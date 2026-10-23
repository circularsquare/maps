// T-975: realms, the country-sized groups of zones the map shows or leaves dark as one (docs/zones.md
// "Realms"; players see only their names). The list is the hand-edited realms.json; a zone's realm is
// its registry row's `realm` (pipeline/realms.py), else its country's when the country is one realm.
//
// A realm is shown when the game has a station in it (blueprint or built), it holds the start zone,
// or the player showed it this session (`showRealm`; not saved). Only what is drawn and loaded for
// viewing follows: demand, the market and money never read this. Everything is shown before the
// first game is in, and for good when the zones' coarse outlines are missing (T-1006: the shade is cut
// from them); a zone with no realm is always shown.

import { computed, effect, signal } from "@preact/signals";
import type { MultiPolygon } from "geojson";
import realmsJson from "./realms.json";
import { CITIES, cityById, cityNear, zoneBox, zonesTouching, type CityId } from "./cities";
import { world } from "./world";
import { coarseOutlines } from "../map/coarseOutlines";

export interface Realm { id: string; name: string; countries?: string[]; units?: Record<string, string[] | "*"> }
export const REALMS = (realmsJson as { realms: Realm[] }).realms;
const byId = new Map(REALMS.map((r) => [r.id, r]));
export const realmName = (id: string) => byId.get(id)?.name ?? id;
const byCountry = new Map(REALMS.flatMap((r) => (r.countries ?? []).map((cc) => [cc, r.id] as const)));

/** A zone's realm, or null for one in none (a split country's zone the join has not given one). */
export function zoneRealm(id: CityId): string | null {
  const c = cityById(id);
  if (!c) return null;
  if (c.realm && byId.has(c.realm)) return c.realm;
  return byCountry.get(c.country ?? "") ?? null;
}

type Box = [number, number, number, number];
export interface RealmSpot {
  id: string;
  /** where its button sits: the centre of the zone nearest the mean of its zones' centres */
  label: [number, number];
  zones: number;
  box: Box;
}

/** Each realm with zones: its count, box and button point, from the registry. */
export const REALM_SPOTS: ReadonlyMap<string, RealmSpot> = (() => {
  const groups = new Map<string, CityId[]>();
  for (const c of CITIES) {
    const r = zoneRealm(c.id);
    if (r) groups.set(r, [...(groups.get(r) ?? []), c.id]);
  }
  const out = new Map<string, RealmSpot>();
  for (const [id, zones] of groups) {
    const box: Box = [Infinity, Infinity, -Infinity, -Infinity];
    for (const z of zones) {
      const b = zoneBox(cityById(z)!);
      box[0] = Math.min(box[0], b[0]); box[1] = Math.min(box[1], b[1]);
      box[2] = Math.max(box[2], b[2]); box[3] = Math.max(box[3], b[3]);
    }
    let mx = 0, my = 0;
    for (const z of zones) { const [x, y] = cityById(z)!.center; mx += x / zones.length; my += y / zones.length; }
    const k = Math.cos((my * Math.PI) / 180);
    let label: [number, number] = [mx, my], bd = Infinity;
    for (const z of zones) {
      const [x, y] = cityById(z)!.center, d = ((x - mx) * k) ** 2 + (y - my) ** 2;
      if (d < bd) { bd = d; label = [x, y]; }
    }
    out.set(id, { id, label, zones: zones.length, box });
  }
  return out;
})();

/** Every zone's coarse outline (`packs/outlines.json`), the shade's edges; null while it loads. */
export const zoneOutlines = signal<ReadonlyMap<string, MultiPolygon> | null>(null);
/** No outlines to cut the shade from, or the checks' every-zone pages: every zone is shown. */
export const realmsOff = signal(false);

export async function loadRealms() {
  // the checks' pages (`opened=all`: city-browser, every-pack-opens) visit zones in every realm,
  // including new rows the join has not given a realm yet
  if (new URLSearchParams(location.search).get("opened") === "all" || !REALM_SPOTS.size) { realmsOff.value = true; return; }
  const zones = await coarseOutlines();
  if (zones.size) zoneOutlines.value = zones;
  else realmsOff.value = true;
}

function inRings(g: MultiPolygon, lon: number, lat: number): boolean {
  let inside = false;
  for (const poly of g.coordinates) for (const r of poly) {
    for (let i = 0, j = r.length - 1; i < r.length; j = i++) {
      const [xi, yi] = r[i], [xj, yj] = r[j];
      if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** The realm a station at (lon, lat) builds in: the zone whose coarse outline holds it, else (dead
 * space, or before the outlines are in) the zone with the nearest centre. */
export function realmOfPlace(lon: number, lat: number): string | null {
  const outlines = zoneOutlines.peek();
  if (outlines) for (const c of zonesTouching(lon, lat, lon, lat)) {
    const g = outlines.get(c.id);
    if (g && inRings(g, lon, lat)) return zoneRealm(c.id);
  }
  return zoneRealm(cityNear(lon, lat));
}

// ---------------------------------------------------------------- which realms are shown

/** The game in hand's start zone's realm; undefined before the first game is in. */
const startRealm = signal<string | null | undefined>(undefined);
/** The first game's first snapshot has come: until then everything shows, as before realms. */
const gameIn = signal(false);
const clicked = signal<ReadonlySet<string>>(new Set());
const built = signal<ReadonlySet<string>>(new Set());

/** The realms shown; null when every zone is (before the first game, or with no outlines). */
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
  return !r || shown.has(r);
}

/** The realms drawn dark with a button. */
export const inactiveRealms = computed<RealmSpot[]>(() => {
  const shown = shownRealms.value;
  if (!shown) return [];
  return [...REALM_SPOTS.values()].filter((l) => !shown.has(l.id));
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
  clicked.value = new Set();
  waitFor = world.peek();
}
let waitFor: unknown;

/** The realms with a station, from each world snapshot (stations placed by position, kept by id). */
export function followStations(): () => void {
  const at = new Map<string, { lng: number; lat: number; realm: string | null }>();
  let outlinesSeen: unknown = null;
  return effect(() => {
    const w = world.value, outlines = zoneOutlines.value;
    if (startRealm.peek() !== undefined && w !== waitFor && !gameIn.peek()) gameIn.value = true;
    if (outlines !== outlinesSeen) {
      outlinesSeen = outlines;
      at.clear();
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
