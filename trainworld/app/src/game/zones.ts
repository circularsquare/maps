// The zone under the view and the zones beside it (T-276, docs/zones.md 6). Every registry entry
// is a zone, a city or not (`kind`); the focused zone is the one whose boundary holds the view's
// centre or lies within reach of it (`placeAt`; none over dead space, T-318); up to NEARBY more zones whose boundaries are on
// screen show their own bubbles beside the focused zone's (map/commuterView.ts), and the demand
// pool opens them (workers/demandClient.ts). T-404: a zone's boundary is fetched only when a place
// asked about lies near its registry box; until every zone that could hold the place is in, the
// answer falls back to the nearest centre, so it never flips as boundaries arrive one by one.

import { signal } from "@preact/signals";
import { cityNear, cityWithin, hashView, zonesTouching, ZONE_REACH_M, type CityId } from "./cities";
import { perfOff } from "../perfFlags";
import { M_PER_DEG } from "./frame";
import { atZero, distanceTo, footprintOf, loadFootprints, touchesBox, type Footprint } from "../workers/cityReach";

/** Zones beside the focused one that show their bubbles, at most this many. */
export const NEARBY = 3;

/** The zones beside the focused one with their bubbles drawn, nearest first (never the focused one). */
export const nearbyZones = signal<CityId[]>([]);

/** T-871: zones load for viewing (their packs, their exact outlines) only above this zoom, one level
 * out from where their own bubbles show (`OVERVIEW_ZOOM`, map/intercityView.ts); further out the
 * overview's per-zone bubbles need no pack. */
export const ZONE_LOAD_ZOOM = perfOff("zoneZoom") ? -1 : 7.5;

/** The view is zoomed in past `ZONE_LOAD_ZOOM` (map/zoneFocus.ts sets it after each move). */
export const zonesLoadForView = signal((() => { const v = hashView(); return !v || v.zoom > ZONE_LOAD_ZOOM; })());

/** Bumped when boundaries arrive, so the caller can place the view again. */
export const zonesLoaded = signal(0);

const asked = new Set<CityId>();
/** T-884: boundaries arriving in many small batches (a game across Japan's prefectures) re-place
 * everything once per quarter second, not once per batch */
let bumpTimer = 0;
const bumpLoaded = () => {
  if (!bumpTimer) bumpTimer = setTimeout(() => { bumpTimer = 0; zonesLoaded.value++; }, 250) as unknown as number;
};

/** The footprints of `ids`, or null (and they are fetched) while any is not in. Failed ones are left out. */
function footprintsOf(ids: CityId[]): Footprint[] | null {
  const missing = ids.filter((id) => footprintOf(id) === undefined);
  if (missing.length) {
    const fresh = missing.filter((id) => !asked.has(id));
    for (const id of fresh) asked.add(id);
    if (fresh.length) void loadFootprints(fresh).then(bumpLoaded);
    return null;
  }
  return ids.map((id) => footprintOf(id)).filter((f): f is Footprint => !!f);
}

/** The zones whose boundary (lon, lat) can be within `ZONE_REACH_M` of: their boxes say. */
function around(lon: number, lat: number): CityId[] {
  return zonesTouching(lon, lat, lon, lat, ZONE_REACH_M).map((c) => c.id);
}

/** The zone whose boundary holds (lon, lat); else the nearest boundary within reach; else the
 * nearest centre. Always some zone, however far: for lookups that need one. A name or a load for
 * the place uses `placeAt`. */
export function zoneAt(lon: number, lat: number): CityId {
  const fs = footprintsOf(around(lon, lat));
  return (fs && nearestZone(fs, lon, lat)?.id) || cityNear(lon, lat);
}

function nearestZone(fs: Footprint[], lon: number, lat: number): { id: CityId; d: number } | null {
  // T-1348: the first zone at distance 0 wins (none before it can, and the walk below stops there),
  // so look for it before measuring any distance
  for (const f of fs) if (atZero(f, lon, lat)) return { id: f.id, d: 0 };
  let best: CityId | null = null, bestD = Infinity;
  // T-348: a zone whose bounding box is no nearer than the best so far cannot win, so its outline
  // is not walked (with a millimetre to spare for rounding, so the answer is exactly the full walk's)
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180);
  for (const f of fs) {
    const [w, s, e, n] = f.bbox;
    if (Math.hypot(Math.max(0, w - lon, lon - e) * kx, Math.max(0, s - lat, lat - n) * M_PER_DEG) > bestD + 1e-3) continue;
    const d = distanceTo(f, lon, lat);
    if (d < bestD) (bestD = d), (best = f.id);
    if (d === 0) break;
  }
  return best ? { id: best, d: bestD } : null;
}

/** T-318: the zone (lon, lat) is in or within `ZONE_REACH_M` of its boundary; null over dead space.
 * Until the boundaries near it are in, the nearest centre within `CENTRE_REACH_M`. The one answer
 * to "which region is the view in" (the top bar's name, what loads, T-309's local time). */
export function placeAt(lon: number, lat: number): CityId | null {
  const key = lon * 1e3 + lat;
  const hit = placed.get(key);
  if (hit !== undefined && hit.lon === lon && hit.lat === lat && (hit.at < 0 || hit.at === zonesLoaded.peek())) return hit.id;
  const ids = around(lon, lat);
  const fs = ids.length ? footprintsOf(ids) : [];
  const id = !ids.length ? null : !fs ? cityWithin(lon, lat) : (() => { const z = nearestZone(fs, lon, lat); return z && z.d <= ZONE_REACH_M ? z.id : null; })();
  if (placed.size >= 200_000) placed.clear();
  placed.set(key, { lon, lat, id, at: fs ? -1 : zonesLoaded.peek() });
  return id;
}
/** T-884: places asked before, for good once every nearby boundary was in (footprints are fetched
 * once, so no later one changes the answer), else until the next boundaries arrive (`at`). Placing
 * every station and track edge on each money update, walking Japan's prefecture coastlines, froze the
 * page for over a minute while a game loaded. */
const placed = new Map<number, { lon: number; lat: number; id: CityId | null; at: number }>();

/** Zones other than `focus` whose boundaries touch the box, nearest to (lon, lat) first, at most
 * NEARBY; zones whose boundaries are not in yet are left out until they arrive. Only zones `keep`
 * passes are asked about (T-975: none of a realm not shown has its boundary fetched). */
export function zonesInBox(focus: CityId, lon: number, lat: number, w: number, s: number, e: number, n: number, keep: (id: CityId) => boolean = () => true): CityId[] {
  const ids = zonesTouching(w, s, e, n).map((c) => c.id).filter((id) => id !== focus && keep(id));
  footprintsOf(ids);
  return ids
    .map((id) => footprintOf(id))
    .filter((f): f is Footprint => !!f && touchesBox(f, w, s, e, n))
    .map((f) => ({ id: f.id, d: distanceTo(f, lon, lat) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, NEARBY)
    .map((z) => z.id);
}
