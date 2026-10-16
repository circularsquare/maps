// Each city's place list from OpenStreetMap (T-177, T-065): towns, villages and neighbourhoods
// inside the pack's footprint, `packs/<id>.places.json` (pipeline/places.py). The search box finds
// them, and station names come from them where there is no cross street (map/stationNames.ts).
// Fetched per zone when a search or the station tool needs that zone (the view's first), never at
// startup; a failed fetch is forgotten, so the next caller tries again.

import { signal } from "@preact/signals";
import { CITIES, cityNear, zoneBox, type City, type CityId } from "./cities";
import { normalise } from "./search";

export interface Place {
  name: string;
  /** OSM's place kind: city, town, borough, suburb, quarter, village, neighbourhood, hamlet */
  kind: string;
  lng: number;
  lat: number;
  /** OSM's population, 0 when it gives none */
  pop: number;
  /** the pack whose footprint holds it */
  city: CityId;
  /** normalised name and other names (alt_name, short_name...), for matching */
  keys: string[];
}

interface PlaceFile { format: number; kinds: string[]; places: [string, number, number, number, number, string[]?][] }

const files = new Map<CityId, Promise<Place[]>>();
/** Every place list loaded so far, one array per city; replaced as each arrives. */
export const places = signal<ReadonlyMap<CityId, Place[]>>(new Map());

function load(id: CityId): Promise<Place[]> {
  let p = files.get(id);
  if (!p) {
    p = fetch(new URL(`packs/${id}.places.json`, document.baseURI)).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.url}`);
      const f = (await r.json()) as PlaceFile;
      const list = f.places.map(([name, k, lng, lat, pop, others]): Place => ({
        name, kind: f.kinds[k] ?? "neighbourhood", lng, lat, pop, city: id,
        keys: [normalise(name), ...(others ?? []).map(normalise)],
      }));
      places.value = new Map([...places.peek(), [id, list]]);
      return list;
    });
    files.set(id, p);
    p.catch(() => { if (files.get(id) === p) files.delete(id); });
  }
  return p;
}

const loadAll = (ids: CityId[]) => Promise.all(ids.map((id) => load(id).catch(() => []))).then(() => undefined);

/** Metres from (lng, lat) to a zone's box (0 inside it). */
function boxDistance(c: City, lng: number, lat: number): number {
  const [w, s, e, n] = zoneBox(c), kx = 111_320 * Math.cos((lat * Math.PI) / 180);
  return Math.hypot(Math.max(0, w - lng, lng - e) * kx, Math.max(0, s - lat, lat - n) * 111_320);
}

/** T-400: the lists of the zones whose boxes hold (lng, lat) or lie within VIEW_M of it, else the
 * nearest zone's: what the station tool and a first search need. */
const VIEW_M = 30_000;
export function loadPlacesAt(lng: number, lat: number): Promise<void> {
  const ids = CITIES.filter((c) => boxDistance(c, lng, lat) <= VIEW_M).map((c) => c.id);
  return loadAll(ids.length ? ids : [cityNear(lng, lat)]);
}

/** T-400: a search finding too little: the next WIDEN zones' lists not yet asked for, nearest to
 * (lng, lat) first. Resolves false when every list has been asked for. */
const WIDEN = 4;
export async function widenPlaces(lng: number, lat: number): Promise<boolean> {
  const next = CITIES.filter((c) => !files.has(c.id)).sort((a, b) => boxDistance(a, lng, lat) - boxDistance(b, lng, lat)).slice(0, WIDEN);
  if (!next.length) return false;
  await loadAll(next.map((c) => c.id));
  return true;
}

/** Kinds a station may be named after, nearest first within this many metres. */
const NAMING_M: Record<string, number> = {
  neighbourhood: 2500, quarter: 3000, suburb: 4000, borough: 4000, hamlet: 3000, village: 4000, town: 5000, city: 6000,
};

export interface NearPlace { place: Place; d: number; dx: number; dy: number }

/** Every listed place within `maxM` (each way) of a point, with its distance, unsorted. */
function around(lng: number, lat: number, maxM: number): NearPlace[] {
  const kx = 111320 * Math.cos((lat * Math.PI) / 180), ky = 110540;
  const out: NearPlace[] = [];
  for (const list of places.peek().values())
    for (const p of list) {
      const dx = (lng - p.lng) * kx, dy = (lat - p.lat) * ky;
      if (Math.abs(dx) > maxM || Math.abs(dy) > maxM) continue;
      out.push({ place: p, d: Math.hypot(dx, dy), dx, dy });
    }
  return out;
}

/** Places near a point that a station there could be named after, nearest first (loads the lists
 * first). Distance is measured to the place's label point. */
export async function placesNear(lng: number, lat: number): Promise<NearPlace[]> {
  await loadAll(CITIES.filter((c) => boxDistance(c, lng, lat) <= 6000).map((c) => c.id));
  return around(lng, lat, 6000).filter((n) => n.d <= (NAMING_M[n.place.kind] ?? 2500)).sort((a, b) => a.d - b.d);
}

// ---------------------------------------------------------------- name suggestions (T-534)

/** Kinds that divide a city; a station takes one's name only if the place is small. */
const HOOD_KINDS = new Set(["neighbourhood", "quarter", "suburb"]);
/** A subdivision is small when another place of its kind lies within this distance of its label:
 * places of one kind tile a city at about their own size, so this caps its size at about 3 km
 * across (DECISIONS 2026-10-16; Manhattan's nearest fellow suburb is 12 km off). */
const HOOD_PEER_MAX_M = 3000;
/** ...and when OSM gives a population, at most this many people. */
const HOOD_POP_MAX = 200_000;
/** A subdivision names stations within 1.5 times its peer spacing of its label (labels sit off
 * centre), at least this far. */
const HOOD_REACH = 1.5;
const HOOD_REACH_MIN_M = 800;
const GRID_M = 1000;

/** Per list, its places by kind and 1 km cell, built when first needed. */
const grids = new WeakMap<Place[], Map<string, Place[]>>();
const peers = new WeakMap<Place, number>();

function gridOf(list: Place[]): Map<string, Place[]> {
  let g = grids.get(list);
  if (!g) {
    g = new Map();
    for (const p of list) {
      if (!HOOD_KINDS.has(p.kind)) continue;
      const k = `${p.kind}:${Math.floor((p.lng * 111320 * Math.cos((p.lat * Math.PI) / 180)) / GRID_M)}:${Math.floor((p.lat * 110540) / GRID_M)}`;
      (g.get(k) ?? g.set(k, []).get(k)!).push(p);
    }
    grids.set(list, g);
  }
  return g;
}

/** Metres from a subdivision's label to the nearest other place of its kind in its list, or
 * Infinity beyond HOOD_PEER_MAX_M. */
function peerSpacing(p: Place): number {
  const known = peers.get(p);
  if (known !== undefined) return known;
  const list = places.peek().get(p.city);
  let best = Infinity;
  if (list) {
    const g = gridOf(list), kx = 111320 * Math.cos((p.lat * Math.PI) / 180);
    // cells keyed by each place's own latitude scale: near enough at a few km
    const cx = Math.floor((p.lng * kx) / GRID_M), cy = Math.floor((p.lat * 110540) / GRID_M);
    const rings = Math.ceil(HOOD_PEER_MAX_M / GRID_M) + 1;
    for (let i = -rings; i <= rings; i++)
      for (let j = -rings; j <= rings; j++)
        for (const o of g.get(`${p.kind}:${cx + i}:${cy + j}`) ?? []) {
          if (o === p) continue;
          const d = Math.hypot((o.lng - p.lng) * kx, (o.lat - p.lat) * 110540);
          if (d < best) best = d;
        }
  }
  if (best > HOOD_PEER_MAX_M) best = Infinity;
  peers.set(p, best);
  return best;
}

/** The neighbourhoods (OSM's neighbourhood, quarter and suburb) small enough to name a station at
 * a point, nearest first: within its peer spacing (HOOD_PEER_MAX_M at most) and population cap. */
export function hoodsNear(near: NearPlace[]): NearPlace[] {
  return near.filter((n) => {
    if (!HOOD_KINDS.has(n.place.kind) || n.place.pop > HOOD_POP_MAX) return false;
    const s = peerSpacing(n.place);
    return s !== Infinity && n.d <= Math.max(HOOD_REACH * s, HOOD_REACH_MIN_M);
  });
}

/** How far a settlement's name reaches: a city by its population (New York's about 13 km). */
function townReach(p: Place): number {
  if (p.kind === "city") return Math.min(20_000, Math.max(6000, 4500 * Math.sqrt(p.pop / 1e6)));
  return p.kind === "town" ? 5000 : p.kind === "village" ? 4000 : p.kind === "hamlet" ? 3000 : 0;
}

/** The city, town, village or hamlet a point lies in, best first: the label nearest relative to
 * its reach. */
export async function townsNear(lng: number, lat: number): Promise<NearPlace[]> {
  await loadAll(CITIES.filter((c) => boxDistance(c, lng, lat) <= 6000).map((c) => c.id));
  return around(lng, lat, 20_000)
    .map((n) => ({ n, f: n.d / townReach(n.place) }))
    .filter((x) => x.f <= 1)
    .sort((a, b) => a.f - b.f)
    .map((x) => x.n);
}
