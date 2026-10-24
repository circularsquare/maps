// Which city packs a network can change (T-149). A city's local trips ride a line only between two
// stations of its own pack: the home end may ride up to 15 km to a station (`ride_max_m`), but the
// work or activity end walks (`walk_cutoff_m`, 3 km), so a station near its cells is always needed.
// With none, every one of its trips walks or drives, it puts no riders on any line, and leaving its
// pack unloaded changes no number anywhere else. So a pack is needed when a built station lies
// inside its exact footprint (packs/<id>.boundary.geojson) or within REACH_M of it. A station only
// a ride away from a pack adds no rider from it (T-394, notes/T-394.md); the regional market, not
// the pack, carries riders between zones.

import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";
import { CITIES, zoneBox, zonesNearBox, type CityId } from "../game/cities";
import { cityBoundary } from "../game/boundaries";
import { M_PER_DEG } from "../game/frame";

/** The 3 km walk plus a margin: cell centres lie a little outside the boundary (blocks are spread
 * over the H3 cells they touch) and the packs' equirectangular frames differ slightly from this
 * test's. */
export const REACH_M = 4500;

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

/** T-404: footprints fetched so far, shared by the reach test and the focus rule; null: the
 * boundary could not be fetched (not tried again this page). */
const store = new Map<CityId, Footprint | null>();
const loading = new Map<CityId, Promise<void>>();

/** The footprint if fetched: undefined while not, null if it failed. */
export function footprintOf(id: CityId): Footprint | null | undefined {
  return store.get(id);
}

/** Fetch the footprints of `ids` not yet in; resolves once all of them are in or failed. */
export function loadFootprints(ids: Iterable<CityId>): Promise<void> {
  const ps: Promise<void>[] = [];
  for (const id of ids) {
    if (store.has(id)) continue;
    let p = loading.get(id);
    if (!p) {
      p = cityBoundary(id).then((fc) => { store.set(id, footprint(id, fc)); }, (err) => {
        console.warn("zone boundary unavailable", id, err);
        store.set(id, null);
      });
      loading.set(id, p);
    }
    ps.push(p);
  }
  return Promise.all(ps).then(() => {});
}

/** The zones whose registry box, widened by `marginM`, holds some station (lon, lat pairs): the
 * only ones whose footprint the station can be within `marginM` of (the box holds the boundary). */
export function zonesNearStations(stations: ArrayLike<number>, marginM: number): CityId[] {
  // T-1008: each station against the boxes near it only (thousands of stations, every network)
  const near = new Set<number>();
  const dy = marginM / M_PER_DEG;
  for (let k = 0; k < stations.length; k += 2) {
    const lon = stations[k], lat = stations[k + 1], dx = dy / Math.cos((lat * Math.PI) / 180);
    for (const i of zonesNearBox(lon - dx, lat - dy, lon + dx, lat + dy)) {
      if (near.has(i)) continue;
      const [w, s, e, n] = zoneBox(CITIES[i]);
      if (lat >= s - dy && lat <= n + dy && lon >= w - dx && lon <= e + dx) near.add(i);
    }
  }
  return [...near].sort((a, b) => a - b).map((i) => CITIES[i].id);
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

/** T-889: a footprint's outline edges by grid cell over its bbox, so `distanceTo` looks at the
 * edges near the place instead of a whole coastline (placing Japan's stations and track walked every
 * prefecture's outline for each: 4 s of a 5 s freeze on load). `seg` holds ring and index pairs. */
interface EdgeGrid { nx: number; ny: number; cw: number; ch: number; start: Uint32Array; seg: Uint32Array; stamp: Uint32Array; visit: number }
const grids = new WeakMap<Footprint, EdgeGrid | null>();
/** below this many edges the plain walk is as quick */
const GRID_MIN_EDGES = 256;

function edgeGrid(f: Footprint): EdgeGrid | null {
  let g = grids.get(f);
  if (g !== undefined) return g;
  let edges = 0;
  for (const r of f.rings) edges += r.length / 2;
  const [w, s, e, n] = f.bbox;
  if (edges < GRID_MIN_EDGES || !(e > w) || !(n > s)) { grids.set(f, null); return null; }
  // about four edges a cell
  const side = Math.max(1, Math.min(512, Math.round(Math.sqrt(edges / 4))));
  const nx = side, ny = side, cw = (e - w) / nx, ch = (n - s) / ny;
  const col = (x: number) => Math.min(nx - 1, Math.max(0, Math.floor((x - w) / cw)));
  const row = (y: number) => Math.min(ny - 1, Math.max(0, Math.floor((y - s) / ch)));
  const count = new Uint32Array(nx * ny + 1);
  const each = (fn: (cell: number, ri: number, k: number) => void) => {
    f.rings.forEach((ring, ri) => {
      for (let k = 0, m = ring.length / 2; k < m; k++) {
        const j = (k + 1) % m;
        const x0 = col(Math.min(ring[2 * k], ring[2 * j])), x1 = col(Math.max(ring[2 * k], ring[2 * j]));
        const y0 = row(Math.min(ring[2 * k + 1], ring[2 * j + 1])), y1 = row(Math.max(ring[2 * k + 1], ring[2 * j + 1]));
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) fn(y * nx + x, ri, k);
      }
    });
  };
  each((c) => count[c + 1]++);
  for (let c = 0; c < nx * ny; c++) count[c + 1] += count[c];
  const start = count.slice(), fill = count.slice(0, nx * ny), seg = new Uint32Array(2 * start[nx * ny]);
  each((c, ri, k) => { const at = fill[c]++; seg[2 * at] = ri; seg[2 * at + 1] = k; });
  g = { nx, ny, cw, ch, start, seg, stamp: new Uint32Array(edges), visit: 0 };
  grids.set(f, g);
  return g;
}

/** Metres from (lon, lat) to the footprint: 0 inside it (T-276, the zone under the view). */
export function distanceTo(f: Footprint, lon: number, lat: number): number {
  const g = edgeGrid(f);
  return g ? gridDistance(f, g, lon, lat) : walkDistance(f, lon, lat);
}

/** `walkDistance`'s answer from the edges near the place: inside by the edges in the place's row of
 * cells (only they can cross its latitude), else the nearest edge by rings of cells outwards until
 * no nearer one can be left, ties to the first edge in outline order as the walk takes them. */
function gridDistance(f: Footprint, g: EdgeGrid, lon: number, lat: number): number {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), ky = M_PER_DEG;
  const [w, s, , n] = f.bbox;
  const { nx, ny, cw, ch, start, seg, stamp } = g;
  // edge ids for the stamp: each ring's first edge's number
  const base = ringBase(f);
  if (++g.visit === 0xffffffff) { stamp.fill(0); g.visit = 1; }
  let visit = g.visit;
  if (lat >= s && lat <= n) {
    let inside = false;
    const y = Math.min(ny - 1, Math.max(0, Math.floor((lat - s) / ch)));
    for (let c = y * nx, end = c + nx; c < end; c++) {
      for (let q = start[c]; q < start[c + 1]; q++) {
        const ri = seg[2 * q], k = seg[2 * q + 1], id = base[ri] + k;
        if (stamp[id] === visit) continue;
        stamp[id] = visit;
        const ring = f.rings[ri], m = ring.length / 2, j = (k + 1) % m;
        const ay = (ring[2 * k + 1] - lat) * ky, by = (ring[2 * j + 1] - lat) * ky;
        if ((ay > 0) !== (by > 0)) {
          const ax = (ring[2 * k] - lon) * kx, bx = (ring[2 * j] - lon) * kx;
          const dx = bx - ax, dy = by - ay;
          if (ax - (ay * dx) / dy > 0) inside = !inside;
        }
      }
    }
    if (inside) return 0;
    if (++g.visit === 0xffffffff) { stamp.fill(0); g.visit = 1; }
    visit = g.visit;
  }
  const cx = Math.min(nx - 1, Math.max(0, Math.floor((lon - w) / cw))), cy = Math.min(ny - 1, Math.max(0, Math.floor((lat - s) / ch)));
  // every cell r rings out is at least r - 1 whole cells from the place
  const step = Math.min(cw * kx, ch * ky) * (1 - 1e-9);
  let best = Infinity, bestId = Infinity, bx0 = 0, by0 = 0;
  const most = Math.max(cx, nx - 1 - cx, cy, ny - 1 - cy);
  for (let r = 0; r <= most; r++) {
    const lb = Math.max(0, r - 1) * step;
    if (lb * lb > best) break;
    for (let y = cy - r; y <= cy + r; y++) {
      if (y < 0 || y >= ny) continue;
      const edge = y === cy - r || y === cy + r;
      for (let x = cx - r; x <= cx + r; x += edge ? 1 : 2 * r || 1) {
        if (x < 0 || x >= nx) continue;
        const c = y * nx + x;
        for (let q = start[c]; q < start[c + 1]; q++) {
          const ri = seg[2 * q], k = seg[2 * q + 1], id = base[ri] + k;
          if (stamp[id] === visit) continue;
          stamp[id] = visit;
          const ring = f.rings[ri], m = ring.length / 2, j = (k + 1) % m;
          const ax = (ring[2 * k] - lon) * kx, ay = (ring[2 * k + 1] - lat) * ky;
          const bx = (ring[2 * j] - lon) * kx, by = (ring[2 * j + 1] - lat) * ky;
          const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
          const t = len > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len)) : 0;
          const px = ax + t * dx, py = ay + t * dy, d2 = px * px + py * py;
          if (d2 < best || (d2 === best && id < bestId)) (best = d2), (bestId = id), (bx0 = px), (by0 = py);
        }
      }
    }
  }
  return best === Infinity ? Infinity : Math.hypot(bx0, by0);
}

const ringBases = new WeakMap<Footprint, Uint32Array>();
function ringBase(f: Footprint): Uint32Array {
  let b = ringBases.get(f);
  if (!b) {
    b = new Uint32Array(f.rings.length);
    for (let i = 1; i < f.rings.length; i++) b[i] = b[i - 1] + f.rings[i - 1].length / 2;
    ringBases.set(f, b);
  }
  return b;
}

/** Every edge of the outline, in order. */
function walkDistance(f: Footprint, lon: number, lat: number): number {
  const kx = M_PER_DEG * Math.cos((lat * Math.PI) / 180), ky = M_PER_DEG;
  // T-348: the nearest point by its squared distance, and Math.hypot (slow) for that one only:
  // placing 1,756 stations walked New York's outline for each in about 140 ms
  let inside = false, best = Infinity, bx0 = 0, by0 = 0;
  for (const ring of f.rings) {
    for (let k = 0, m = ring.length / 2; k < m; k++) {
      const j = (k + 1) % m;
      const ax = (ring[2 * k] - lon) * kx, ay = (ring[2 * k + 1] - lat) * ky;
      const bx = (ring[2 * j] - lon) * kx, by = (ring[2 * j + 1] - lat) * ky;
      const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
      const t = len > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len)) : 0;
      const px = ax + t * dx, py = ay + t * dy, d2 = px * px + py * py;
      if (d2 < best) (best = d2), (bx0 = px), (by0 = py);
      if ((ay > 0) !== (by > 0) && ax - (ay * dx) / dy > 0) inside = !inside;
    }
  }
  return inside ? 0 : best === Infinity ? Infinity : Math.hypot(bx0, by0);
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
