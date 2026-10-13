// The zone under the view and the zones beside it (T-276, docs/zones.md 6). Every registry entry
// is a zone, a city or not (`kind`); the focused zone is the one whose boundary holds the view's
// centre, or the nearest boundary over dead space; up to NEARBY more zones whose boundaries are on
// screen show their own bubbles beside the focused zone's (map/commuterView.ts), and the demand
// pool opens them (workers/demandClient.ts). Boundaries are each pack's `<id>.boundary.geojson`,
// fetched once (game/boundaries.ts); until they are in, the focus falls back to the nearest centre.

import { signal } from "@preact/signals";
import { CITIES, cityNear, type CityId } from "./cities";
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

/** The zone whose boundary holds (lon, lat); else the nearest boundary; else the nearest centre. */
export function zoneAt(lon: number, lat: number): CityId {
  if (!footprints.length) return cityNear(lon, lat);
  let best: CityId | null = null, bestD = Infinity;
  for (const f of footprints) {
    const d = distanceTo(f, lon, lat);
    if (d < bestD) (bestD = d), (best = f.id);
    if (d === 0) break;
  }
  return best ?? cityNear(lon, lat);
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
