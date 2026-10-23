// T-975: realms, the country-sized groups of zones the map shows or leaves dark as one (docs/zones.md
// "Realms"; players see only their names). The list is the hand-edited realms.json; a zone's realm is
// its registry row's `realm` (pipeline/realms.py), else its country's, else, for a country split
// between realms, the realm whose land holds its centre. The land is `packs/realms.geojson`.
//
// A realm is shown when the game has a station in it (blueprint or built), it holds the start zone,
// or the player showed it this session (`showRealm`; not saved). Only what is drawn and loaded for
// viewing follows: demand, the market and money never read this. Everything is shown before the
// first game is in, and for good when the land file is missing (an older build's packs); a zone with
// no realm, or whose realm has no land in the file, is always shown.

import { computed, effect, signal } from "@preact/signals";
import type { Feature, FeatureCollection, MultiPolygon } from "geojson";
import realmsJson from "./realms.json";
import { CITIES, cityById, cityWithin, type CityId } from "./cities";
import { dataUrl } from "./dataBase";
import { world } from "./world";

export interface Realm { id: string; name: string; countries?: string[]; units?: Record<string, string[] | "*"> }
export const REALMS = (realmsJson as { realms: Realm[] }).realms;
const byId = new Map(REALMS.map((r) => [r.id, r]));
export const realmName = (id: string) => byId.get(id)?.name ?? id;
const byCountry = new Map(REALMS.flatMap((r) => (r.countries ?? []).map((cc) => [cc, r.id] as const)));
/** a split country's realms */
const splitOf = new Map<string, string[]>();
for (const r of REALMS) for (const cc of Object.keys(r.units ?? {})) splitOf.set(cc, [...(splitOf.get(cc) ?? []), r.id]);

type Box = [number, number, number, number];
export interface RealmLand {
  id: string;
  /** where its button sits: the point deepest inside its largest part */
  label: [number, number];
  zones: number;
  box: Box;
  /** per polygon, its rings as flat lon, lat pairs, and its box */
  polys: { rings: Float64Array[]; box: Box }[];
  feature: Feature<MultiPolygon>;
}

/** The realms' land by id; null while the file loads. */
export const realmLand = signal<ReadonlyMap<string, RealmLand> | null>(null);
/** The land file could not be read: every zone is shown, as before realms. */
export const realmsOff = signal(false);

export async function loadRealmLand() {
  try {
    const r = await fetch(dataUrl("realms.geojson"));
    if (!r.ok) throw new Error(`${r.status} ${r.url}`);
    const fc = (await r.json()) as FeatureCollection<MultiPolygon, { id: string; zones: number; label: [number, number] }>;
    const out = new Map<string, RealmLand>();
    for (const f of fc.features) {
      const { id, zones, label } = f.properties;
      if (!byId.has(id)) continue;
      const all: Box = [Infinity, Infinity, -Infinity, -Infinity];
      const polys = f.geometry.coordinates.map((poly) => {
        const box: Box = [Infinity, Infinity, -Infinity, -Infinity];
        const rings = poly.map((ring) => {
          const flat = new Float64Array(ring.length * 2);
          ring.forEach(([x, y], i) => {
            flat[2 * i] = x; flat[2 * i + 1] = y;
            if (x < box[0]) box[0] = x; if (y < box[1]) box[1] = y;
            if (x > box[2]) box[2] = x; if (y > box[3]) box[3] = y;
          });
          return flat;
        });
        for (let k = 0; k < 4; k++) all[k] = k < 2 ? Math.min(all[k], box[k]) : Math.max(all[k], box[k]);
        return { rings, box };
      });
      out.set(id, { id, zones, label, box: all, polys, feature: f });
    }
    realmLand.value = out;
  } catch (error) {
    console.warn("Realm land unavailable; every zone is shown", error);
    realmsOff.value = true;
  }
}

function inPolys(l: RealmLand, lon: number, lat: number): boolean {
  const [w, s, e, n] = l.box;
  if (lon < w || lon > e || lat < s || lat > n) return false;
  for (const p of l.polys) {
    const [pw, ps, pe, pn] = p.box;
    if (lon < pw || lon > pe || lat < ps || lat > pn) continue;
    // even-odd over every ring, so holes count
    let inside = false;
    for (const r of p.rings) {
      for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
        const yi = r[i + 1], yj = r[j + 1];
        if (yi > lat !== yj > lat && lon < ((r[j] - r[i]) * (lat - yi)) / (yj - yi) + r[i]) inside = !inside;
      }
    }
    if (inside) return true;
  }
  return false;
}

/** The realm whose land holds (lon, lat), among `ids` if given; null outside all, or before the land is in. */
export function realmAt(lon: number, lat: number, ids?: readonly string[]): string | null {
  const land = realmLand.value;
  if (!land) return null;
  for (const l of ids ? ids.map((id) => land.get(id)).filter((x): x is RealmLand => !!x) : land.values()) if (inPolys(l, lon, lat)) return l.id;
  return null;
}

const splitZones = new Map<CityId, string | null>();
/** A zone's realm, or null for one in none. */
export function zoneRealm(id: CityId): string | null {
  const c = cityById(id);
  if (!c) return null;
  if (c.realm && byId.has(c.realm)) return c.realm;
  const cc = c.country ?? "";
  const whole = byCountry.get(cc);
  if (whole) return whole;
  const among = splitOf.get(cc);
  if (!among || !realmLand.value) return null;
  let r = splitZones.get(id);
  if (r === undefined) splitZones.set(id, (r = realmAt(c.center[0], c.center[1], among) ?? nearestRealm(c.center[0], c.center[1], among)));
  return r;
}

/** Of `ids`, the realm with a land vertex nearest (lon, lat): a centre off the simplified coast
 * (Manhattan, Wenzhou on its estuary). */
function nearestRealm(lon: number, lat: number, ids: readonly string[]): string | null {
  const land = realmLand.value!, k = Math.cos((lat * Math.PI) / 180);
  let best: string | null = null, bd = Infinity;
  for (const id of ids) {
    const l = land.get(id);
    if (!l) continue;
    for (const p of l.polys) for (const r of p.rings) for (let i = 0; i < r.length; i += 2) {
      const d = ((r[i] - lon) * k) ** 2 + (r[i + 1] - lat) ** 2;
      if (d < bd) { bd = d; best = id; }
    }
  }
  return best;
}

/** The realm a station at (lon, lat) builds in: the land's, else (off the simplified coast) its zone's. */
export function realmOfPlace(lon: number, lat: number): string | null {
  const r = realmAt(lon, lat);
  if (r) return r;
  const z = cityWithin(lon, lat);
  return z ? zoneRealm(z) : null;
}

// ---------------------------------------------------------------- which realms are shown

/** The game in hand's start zone's realm; undefined before the first game is in. */
const startRealm = signal<string | null | undefined>(undefined);
/** The first game's first snapshot has come: until then everything shows, as before realms. */
const gameIn = signal(false);
const clicked = signal<ReadonlySet<string>>(new Set());
const built = signal<ReadonlySet<string>>(new Set());

/** The realms shown; null when every zone is (before the first game, or with no land file). */
export const shownRealms = computed<ReadonlySet<string> | null>(() => {
  if (realmsOff.value || !gameIn.value) return null;
  const out = new Set([...built.value, ...clicked.value]);
  const s = startRealm.value;
  if (s) out.add(s);
  return out;
});

/** Whether a zone draws and loads for viewing. Reads signals, so effects that ask follow changes. */
export function zoneShown(id: CityId): boolean {
  const shown = shownRealms.value;
  if (!shown) return true;
  const r = zoneRealm(id);
  if (!r) return true;
  const land = realmLand.value;
  // a realm with no land in the file (zones added since it was written) cannot be drawn dark
  if (land && !land.has(r)) return true;
  return shown.has(r);
}

/** The realms drawn dark with a button: those with land and zones, not shown. */
export const inactiveRealms = computed<RealmLand[]>(() => {
  const shown = shownRealms.value, land = realmLand.value;
  if (!shown || !land) return [];
  return [...land.values()].filter((l) => l.zones > 0 && !shown.has(l.id));
});

/** Zones not shown, for the clock worker's masks fetched ahead of the view. */
export const hiddenZones = (): CityId[] => (shownRealms.peek() ? CITIES.filter((c) => !zoneShown(c.id)).map((c) => c.id) : []);

/** Show a realm for this session (its button, a search pick). */
export function showRealm(id: string | null) {
  if (!id || clicked.peek().has(id)) return;
  clicked.value = new Set([...clicked.peek(), id]);
}
export const showZoneRealm = (zone: CityId) => showRealm(zoneRealm(zone));

/** A game was loaded or started (the clock worker's "opened", sent before its first snapshot):
 * its start zone, or for a save older than its `start` field the zone it opens on. The realms shown
 * by a click go with the game before. */
export function setGameStart(start: CityId | null | undefined) {
  startRealm.value = start ? zoneRealm(start) ?? null : null;
  pendingStart = start ?? null;
  clicked.value = new Set();
  waitFor = world.peek();
}
let waitFor: unknown;
/** a split country's start zone placed once the land is in */
let pendingStart: CityId | null = null;

/** The realms with a station, from each world snapshot (stations placed by position, kept by id). */
export function followStations(): () => void {
  const at = new Map<string, { lng: number; lat: number; realm: string | null }>();
  let landSeen: unknown = null;
  return effect(() => {
    const w = world.value, land = realmLand.value;
    if (startRealm.peek() !== undefined && w !== waitFor && !gameIn.peek()) gameIn.value = true;
    if (land !== landSeen) {
      landSeen = land;
      at.clear();
      if (pendingStart) startRealm.value = zoneRealm(pendingStart) ?? null;
    }
    const next = new Set<string>();
    for (const s of w?.save.stations ?? []) {
      let e = at.get(s.id);
      if (!e || e.lng !== s.lng || e.lat !== s.lat) at.set(s.id, (e = { lng: s.lng, lat: s.lat, realm: realmOfPlace(s.lng, s.lat) }));
      if (e.realm) next.add(e.realm);
    }
    const prev = built.peek();
    if (next.size !== prev.size || [...next].some((r) => !prev.has(r))) built.value = next;
  });
}
