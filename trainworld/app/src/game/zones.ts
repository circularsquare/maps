// The zone under the view and the zones beside it (T-276, docs/zones.md 6). Every registry entry
// is a zone, a city or not (`kind`); the focused zone is the one whose boundary holds the view's
// centre or lies within reach of it (`placeAt`; none over dead space, T-318); up to NEARBY more zones whose boundaries are on
// screen show their own bubbles beside the focused zone's (map/commuterView.ts), and the demand
// pool opens them (workers/demandClient.ts). Boundaries are each pack's `<id>.boundary.geojson`,
// fetched once (game/boundaries.ts); until they are in, the focus falls back to the nearest centre.

import { signal } from "@preact/signals";
import { CITIES, cityNear, cityWithin, ZONE_REACH_M, type CityId } from "./cities";
import { cityBoundary } from "./boundaries";
import { distanceTo, footprint, touchesBox, type Footprint } from "../workers/cityReach";

/** Zones beside the focused one that show their bubbles, at most this many. */
export const NEARBY = 3;

/** The zones beside the focused one with their bubbles drawn, nearest first (never the focused one). */
export const nearbyZones = signal<CityId[]>([]);

let footprints: Footprint[] = [];
let loading: Promise<void> | null = null;
/** Bumped when a boundary arrives, so the caller can place the view again. */
export const zonesLoaded = signal(0);

/** Fetch every registered zone's boundary (the same fetch as the outlines and the reach test). */
export function loadZones(): Promise<void> {
  // all at once, in registry order (a point two boundaries share always goes to the same zone): a
  // focus taken from some boundaries would flip as the others arrive, opening packs for nothing
  return (loading ??= Promise.all(CITIES.map((c) => cityBoundary(c.id).then((fc) => footprint(c.id, fc), (err) => {
    console.warn("zone boundary unavailable", c.id, err);
    return null;
  }))).then((list) => {
    footprints = list.filter((f): f is Footprint => !!f);
    zonesLoaded.value++;
  }));
}

/** The zone whose boundary holds (lon, lat); else the nearest boundary; else the nearest centre.
 * Always some zone, however far: for lookups that need one. A name or a load for the place uses
 * `placeAt`. */
export function zoneAt(lon: number, lat: number): CityId {
  return nearestZone(lon, lat)?.id ?? cityNear(lon, lat);
}

function nearestZone(lon: number, lat: number): { id: CityId; d: number } | null {
  let best: CityId | null = null, bestD = Infinity;
  for (const f of footprints) {
    const d = distanceTo(f, lon, lat);
    if (d < bestD) (bestD = d), (best = f.id);
    if (d === 0) break;
  }
  return best ? { id: best, d: bestD } : null;
}

/** T-318: the zone (lon, lat) is in or within `ZONE_REACH_M` of its boundary; null over dead space.
 * Until the boundaries are in, the nearest centre within `CENTRE_REACH_M`. The one answer to "which
 * region is the view in" (the top bar's name, what loads, T-309's local time). */
export function placeAt(lon: number, lat: number): CityId | null {
  if (!footprints.length) return cityWithin(lon, lat);
  const z = nearestZone(lon, lat);
  return z && z.d <= ZONE_REACH_M ? z.id : null;
}

/** Zones other than `focus` whose boundaries touch the box, nearest to (lon, lat) first, at most NEARBY. */
export function zonesInBox(focus: CityId, lon: number, lat: number, w: number, s: number, e: number, n: number): CityId[] {
  return footprints
    .filter((f) => f.id !== focus && touchesBox(f, w, s, e, n))
    .map((f) => ({ id: f.id, d: distanceTo(f, lon, lat) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, NEARBY)
    .map((z) => z.id);
}
