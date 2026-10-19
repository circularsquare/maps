// The zone under the view and the zones beside it (T-276, docs/zones.md 6). Every registry entry
// is a zone, a city or not (`kind`); the focused zone is the one whose boundary holds the view's
// centre or lies within reach of it (`placeAt`; none over dead space, T-318); up to NEARBY more zones whose boundaries are on
// screen show their own bubbles beside the focused zone's (map/commuterView.ts), and the demand
// pool opens them (workers/demandClient.ts). T-404: a zone's boundary is fetched only when a place
// asked about lies near its registry box; until every zone that could hold the place is in, the
// answer falls back to the nearest centre, so it never flips as boundaries arrive one by one.

import { signal } from "@preact/signals";
import { cityNear, cityWithin, zonesTouching, ZONE_REACH_M, type CityId } from "./cities";
import { distanceTo, footprintOf, loadFootprints, touchesBox, type Footprint } from "../workers/cityReach";

/** Metres per degree, as workers/cityReach.ts measures. */
const M_PER_DEG = 111_320;

/** Zones beside the focused one that show their bubbles, at most this many. */
export const NEARBY = 3;

/** The zones beside the focused one with their bubbles drawn, nearest first (never the focused one). */
export const nearbyZones = signal<CityId[]>([]);

/** Bumped when boundaries arrive, so the caller can place the view again. */
export const zonesLoaded = signal(0);

const asked = new Set<CityId>();

/** The footprints of `ids`, or null (and they are fetched) while any is not in. Failed ones are left out. */
function footprintsOf(ids: CityId[]): Footprint[] | null {
  const missing = ids.filter((id) => footprintOf(id) === undefined);
  if (missing.length) {
    const fresh = missing.filter((id) => !asked.has(id));
    for (const id of fresh) asked.add(id);
    if (fresh.length) void loadFootprints(fresh).then(() => { zonesLoaded.value++; });
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
  const ids = around(lon, lat);
  if (!ids.length) return null;
  const fs = footprintsOf(ids);
  if (!fs) return cityWithin(lon, lat);
  const z = nearestZone(fs, lon, lat);
  return z && z.d <= ZONE_REACH_M ? z.id : null;
}

/** Zones other than `focus` whose boundaries touch the box, nearest to (lon, lat) first, at most
 * NEARBY, of those `keep` passes; zones whose boundaries are not in yet are left out until they arrive. */
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
