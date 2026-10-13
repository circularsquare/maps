// Which city packs a network can change (T-149). A city's local trips can ride only from a station
// within the walk's 4 km bound of one of its cells (SPEC 4.3, `walk_cutoff_m`); with none, every
// one of its trips walks or drives, it puts no riders on any line, and leaving its pack unloaded
// changes no number anywhere else. So a pack is needed when a built station lies inside its exact
// footprint (packs/<id>.boundary.geojson) or within REACH_M of it. The margin over 4 km covers
// cell centres a little outside the boundary (blocks are spread over the H3 cells they touch)
// and the packs' equirectangular frames.

import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import { CITIES, type CityId } from "../game/cities";
import { cityBoundary } from "../game/boundaries";

export const REACH_M = 4500;
const M_PER_DEG = 111_320;

export interface Footprint {
  id: CityId;
  /** rings as lon, lat pairs (outer rings and holes alike: the even-odd rule needs no difference) */
  rings: Float64Array[];
  /** west, south, east, north, degrees */
  bbox: [number, number, number, number];
}

export function footprint(id: CityId, fc: FeatureCollection<Polygon | MultiPolygon>): Footprint {
  const rings: Float64Array[] = [];
  const bbox: Footprint["bbox"] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const f of fc.features) {
    const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const p of polys) for (const r of p) {
      const a = new Float64Array(2 * r.length);
      r.forEach(([lon, lat], k) => {
        a[2 * k] = lon; a[2 * k + 1] = lat;
        bbox[0] = Math.min(bbox[0], lon); bbox[1] = Math.min(bbox[1], lat);
        bbox[2] = Math.max(bbox[2], lon); bbox[3] = Math.max(bbox[3], lat);
      });
      rings.push(a);
    }
  }
  return { id, rings, bbox };
}

/** Every registered city's footprint. Rejects if any is missing: the caller then loads every pack. */
export async function loadFootprints(): Promise<Footprint[]> {
  return Promise.all(CITIES.map(async ({ id }) => footprint(id, await cityBoundary(id))));
}

/** Inside the footprint, or within REACH_M of its outline. */
function reaches(f: Footprint, lon: number, lat: number): boolean {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), ky = M_PER_DEG;
  const [w, s, e, n] = f.bbox;
  if ((w - lon) * kx > REACH_M || (lon - e) * kx > REACH_M || (s - lat) * ky > REACH_M || (lat - n) * ky > REACH_M) return false;
  let inside = false;
  const r2 = REACH_M * REACH_M;
  for (const ring of f.rings) {
    for (let k = 0, m = ring.length / 2; k < m; k++) {
      const j = (k + 1) % m;
      const ax = (ring[2 * k] - lon) * kx, ay = (ring[2 * k + 1] - lat) * ky;
      const bx = (ring[2 * j] - lon) * kx, by = (ring[2 * j + 1] - lat) * ky;
      // the station is at the origin: distance to segment a-b, then the even-odd crossing test
      const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
      const t = len > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len)) : 0;
      const px = ax + t * dx, py = ay + t * dy;
      if (px * px + py * py <= r2) return true;
      if ((ay > 0) !== (by > 0) && ax - (ay * dx) / dy > 0) inside = !inside;
    }
  }
  return inside;
}

/** Metres from (lon, lat) to the footprint: 0 inside it (T-276, the zone under the view). */
export function distanceTo(f: Footprint, lon: number, lat: number): number {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), ky = M_PER_DEG;
  let inside = false, best = Infinity;
  for (const ring of f.rings) {
    for (let k = 0, m = ring.length / 2; k < m; k++) {
      const j = (k + 1) % m;
      const ax = (ring[2 * k] - lon) * kx, ay = (ring[2 * k + 1] - lat) * ky;
      const bx = (ring[2 * j] - lon) * kx, by = (ring[2 * j + 1] - lat) * ky;
      const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
      const t = len > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len)) : 0;
      best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
      if ((ay > 0) !== (by > 0) && ax - (ay * dx) / dy > 0) inside = !inside;
    }
  }
  return inside ? 0 : best;
}

/** Whether the footprint touches the box (west, south, east, north, degrees): an edge crosses it,
 * or the box's middle is inside. */
export function touchesBox(f: Footprint, w: number, s: number, e: number, n: number): boolean {
  if (f.bbox[0] > e || f.bbox[2] < w || f.bbox[1] > n || f.bbox[3] < s) return false;
  // a segment against the box: Liang-Barsky clipping
  const crosses = (x0: number, y0: number, x1: number, y1: number) => {
    let t0 = 0, t1 = 1;
    const dx = x1 - x0, dy = y1 - y0;
    for (const [p, q] of [[-dx, x0 - w], [dx, e - x0], [-dy, y0 - s], [dy, n - y0]]) {
      if (p === 0) { if (q < 0) return false; continue; }
      const r = q / p;
      if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r; }
      else { if (r < t0) return false; if (r < t1) t1 = r; }
    }
    return true;
  };
  for (const ring of f.rings) {
    for (let k = 0, m = ring.length / 2; k < m; k++) {
      const j = (k + 1) % m;
      if (crosses(ring[2 * k], ring[2 * k + 1], ring[2 * j], ring[2 * j + 1])) return true;
    }
  }
  return distanceTo(f, (w + e) / 2, (s + n) / 2) === 0;
}

/** The cities whose packs some station (lon, lat pairs) can change. */
export function citiesReached(footprints: Footprint[], stations: ArrayLike<number>): Set<CityId> {
  const out = new Set<CityId>();
  for (const f of footprints) {
    for (let k = 0; k < stations.length; k += 2) {
      if (reaches(f, stations[k], stations[k + 1])) { out.add(f.id); break; }
    }
  }
  return out;
}
