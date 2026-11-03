// Ferry routes over the basemap's own water (T-2192, notes/T-2177.md 3.3 and "F1 -> F2 route format"). A leg
// between two points on water is searched on a raster of the zoom 12 water tiles round it: the plain shortest
// path, then one where every metre nearer land than `CLEAR_M` costs more, eased off while that path is over
// `DETOUR` times the plain one. A line's legs are then joined, their corners rounded as wide as the water
// allows, and each point given the speed its room and bend allow. Pure: tiles come through a loader, so the
// worker (`workers/ferryWater.worker.ts`) and the Node check share it.

import { WATER_ZOOM, type WaterTile } from "./waterTiles";

export type LonLat = [number, number];
/** a loaded tile (empty where the server has none); called only for tiles a loader has loaded */
export type TileGet = (x: number, y: number, z: number) => WaterTile;
/** loads the tiles given (x, y, z); null when they would not load */
export type TileLoad = (tiles: [number, number, number][]) => Promise<TileGet | null>;

/** wanted distance off land, m: nearer, a metre costs up to 1 + `PENALTY_K` */
export const CLEAR_M = 400;
const PENALTY_K = 3;
/** water narrower than twice this (m of room) pulls a boat to its middle: a metre at the edge costs up to
 * 1 + `CENTRE_K`, always, before the 400 m preference */
const CENTRE_M = 15;
const CENTRE_K = 2;
/** the penalty fades out within this of either end of a leg, m */
const FADE_M = 300;
/** the penalty is eased while its path is over this many times the plain shortest */
export const DETOUR = 1.3;
/** a boat keeps at least this off land, m: a channel 10 m wide at least */
export const HALF_M = 5;
const MIN_CELL_M = 2.5;
const MAX_CELLS = 1.5e6;
/** OSM's water stops at many canal bridges (Amsterdam, Venice), some 40 m wide (Torensluis): a gap under a
 * bridge this long or less is water */
const BRIDGE_GAP_M = 60;
/** widest bend, m */
export const R_MAX_M = 300;
const R_MIN_M = 3;
/** Douglas-Peucker's tolerance, m, and at most this share of the room at the point */
const DP_M = 25;
const DP_ROOM = 0.25;
/** speeds by room (distance to land): `V_OPEN` from `ROOM_OPEN_M`, `V_CHANNEL` at `ROOM_CHANNEL_M` or less */
export const V_OPEN = 25 / 3.6;
export const V_CHANNEL = 10 / 3.6;
const ROOM_OPEN_M = 200;
const ROOM_CHANNEL_M = 20;
/** sideways acceleration on a bend, m/s²: at most √(BEND_A r) */
const BEND_A = 0.3;
/** points on a straight stretch at most this far apart while the speed changes, m */
const STEP_M = 25;
/** a speed change that keeps a point, m/s */
const DV_KEEP = 0.25;
/** arc points at most this far apart, m */
const ARC_STEP_M = 20;
/** the longest leg, m */
export const MAX_LEG_M = 150000;
/** tiles a raster reads at most; a long leg's coarse plan reads coarser tiles to stay under it */
const MAX_TILES = 80;
/** a leg whose box needs a coarser cell than this is searched coarse, its ends fine */
const FINE_CELL_M = 5;
/** half the side of each end's fine box, m */
const END_M = 700;
/** a port click is at most this far from water, m; its berth at most `BERTH_REACH_M` from its landing */
export const PORT_REACH_M = 150;
const BERTH_REACH_M = 300;
/** a berth's room, m: a little more than `HALF_M`, so the coarser raster of a long leg still finds it */
const BERTH_ROOM_M = 6;
/** a boat's acceleration and braking, m/s² (the sim's; here for `runSeconds`) */
export const ACCEL = 0.15;

export type Refusal = "no-water-path" | "far-from-water" | "on-water" | "too-long" | "tiles";

// ------------------------------------------------------------------ frames and tiles

const R_EARTH = 6371008.8;
const D2R = Math.PI / 180;

/** A plane round a point, metres east and north. Good for a leg or a line (tens of km). */
export class Frame {
  readonly kx: number;
  constructor(readonly lon0: number, readonly lat0: number) {
    this.kx = Math.cos(lat0 * D2R);
  }
  x(lon: number) {
    return R_EARTH * (lon - this.lon0) * D2R * this.kx;
  }
  y(lat: number) {
    return R_EARTH * (lat - this.lat0) * D2R;
  }
  lon(x: number) {
    return this.lon0 + x / (R_EARTH * this.kx) / D2R;
  }
  lat(y: number) {
    return this.lat0 + y / R_EARTH / D2R;
  }
}

/** ground metres between two points */
export function metres(a: LonLat, b: LonLat): number {
  const f = new Frame((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
  return Math.hypot(f.x(a[0]) - f.x(b[0]), f.y(a[1]) - f.y(b[1]));
}

const tileX = (lon: number, nt: number) => Math.floor(((lon + 180) / 360) * nt);
const tileY = (lat: number, nt: number) => {
  const s = Math.sin(lat * D2R);
  return Math.floor((0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * nt);
};

/** A raster's place: the plane, its south-west corner (m), cell size (m), cells across and up, and the
 * tiles' zoom (`WATER_ZOOM` unless a long leg's coarse plan reads coarser tiles). */
export interface Box {
  frame: Frame;
  x0: number;
  y0: number;
  cell: number;
  W: number;
  H: number;
  z?: number;
}

/** A box round a centre point covering `x0..x1`, `y0..y1` in its frame, at the finest cell that keeps it
 * under `MAX_CELLS` (or `cell` if given). */
function boxOf(frame: Frame, x0: number, y0: number, x1: number, y1: number, cell?: number): Box {
  const c = cell ?? Math.max(MIN_CELL_M, Math.sqrt(((x1 - x0) * (y1 - y0)) / MAX_CELLS));
  return { frame, x0, y0, cell: c, W: Math.ceil((x1 - x0) / c), H: Math.ceil((y1 - y0) / c) };
}

/** The tiles under a box, x, y, z. */
export function tilesOf(b: Box): [number, number, number][] {
  const z = b.z ?? WATER_ZOOM, nt = 2 ** z;
  const f = b.frame, x1 = b.x0 + b.W * b.cell, y1 = b.y0 + b.H * b.cell;
  const ta = tileX(f.lon(b.x0), nt), tb = tileX(f.lon(x1), nt), tc = tileY(f.lat(y1), nt), td = tileY(f.lat(b.y0), nt);
  const out: [number, number, number][] = [];
  for (let x = ta; x <= tb; x++) for (let y = tc; y <= td; y++) if (y >= 0 && y < nt) out.push([((x % nt) + nt) % nt, y, z]);
  return out;
}

// ------------------------------------------------------------------ the raster

export interface Raster extends Box {
  /** 1 where the cell's centre is in water (bridges filled) */
  water: Uint8Array;
  /** m from the cell's centre to the nearest land cell's centre; large where no land is in the box */
  dist: Float32Array;
  /** cells a bridge filled */
  bridged: number;
  ms: { fill: number; edt: number };
}

/** the room at a cell, m: its distance to land less the half cell a raster cannot see */
const roomOf = (r: Raster, i: number) => r.dist[i] - r.cell / 2;
/** what a cell's centre needs to be sailed through: `HALF_M` to the nearest land cell's centre, so a
 * channel `2 HALF_M` plus a cell wide passes all along, a narrower one only where the cells fall well */
const needOf = (_r: Raster) => HALF_M;
/** what the cell under any point needs, the point being up to half a cell's diagonal off its centre */
const needAnywhere = (r: Box) => HALF_M + 0.71 * r.cell;

const INF = 1e20;
/** at or above this, a transform's value is INF (as Float32 holds it) */
const BIG = 1e19;

/** Squared distances (cells²) from every cell to the nearest cell where `f` is 0, in place (Felzenszwalb
 * and Huttenlocher's exact transform; `f` holds 0 or `INF`). */
function edt(f: Float32Array, W: number, H: number) {
  const n = Math.max(W, H);
  const line = new Float64Array(n), d = new Float64Array(n), z = new Float64Array(n + 1), v = new Int32Array(n);
  const pass = (len: number) => {
    let k = -1;
    for (let q = 0; q < len; q++) {
      if (line[q] >= BIG) continue;
      if (k < 0) {
        k = 0;
        v[0] = q;
        z[0] = -INF;
        z[1] = INF;
        continue;
      }
      let s = 0;
      for (;;) {
        const p = v[k];
        s = (line[q] + q * q - (line[p] + p * p)) / (2 * q - 2 * p);
        // z[0] is -INF, so k stays at 0 or above
        if (s <= z[k]) k--;
        else break;
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    if (k < 0) {
      d.fill(INF, 0, len);
      return;
    }
    k = 0;
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++;
      d[q] = (q - v[k]) * (q - v[k]) + line[v[k]];
    }
  };
  for (let c = 0; c < W; c++) {
    for (let r = 0; r < H; r++) line[r] = f[r * W + c];
    pass(H);
    for (let r = 0; r < H; r++) f[r * W + c] = d[r];
  }
  for (let r = 0; r < H; r++) {
    const o = r * W;
    for (let c = 0; c < W; c++) line[c] = f[o + c];
    pass(W);
    for (let c = 0; c < W; c++) f[o + c] = d[c];
  }
}

/** Fill one polygon feature's rings (cell coordinates) even-odd at cell centres. */
function fillPoly(rings: Float64Array[], W: number, H: number, out: Uint8Array) {
  let ymin = Infinity, ymax = -Infinity, xmin = Infinity, xmax = -Infinity;
  for (const r of rings)
    for (let k = 0; k < r.length; k += 2) {
      if (r[k] < xmin) xmin = r[k];
      if (r[k] > xmax) xmax = r[k];
      if (r[k + 1] < ymin) ymin = r[k + 1];
      if (r[k + 1] > ymax) ymax = r[k + 1];
    }
  if (xmax < 0 || ymax < 0 || xmin > W || ymin > H) return;
  const r0 = Math.max(0, Math.ceil(ymin - 0.5)), r1 = Math.min(H - 1, Math.floor(ymax - 0.5));
  if (r1 < r0) return;
  const rows: number[][] = [];
  for (let i = r0; i <= r1; i++) rows.push([]);
  for (const r of rings) {
    const n = r.length / 2;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xa = r[2 * j], ya = r[2 * j + 1], xb = r[2 * i], yb = r[2 * i + 1];
      if (ya === yb) continue;
      const lo = Math.max(r0, Math.ceil(Math.min(ya, yb) - 0.5)), hi = Math.min(r1, Math.ceil(Math.max(ya, yb) - 0.5) - 1);
      // rows whose centre yc satisfies min <= yc < max
      for (let row = lo; row <= hi; row++) {
        const yc = row + 0.5;
        rows[row - r0].push(xa + ((yc - ya) / (yb - ya)) * (xb - xa));
      }
    }
  }
  for (let row = r0; row <= r1; row++) {
    const xs = rows[row - r0];
    if (xs.length < 2) continue;
    xs.sort((p, q) => p - q);
    const o = row * W;
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const c0 = Math.max(0, Math.ceil(xs[k] - 0.5)), c1 = Math.min(W - 1, Math.floor(xs[k + 1] - 0.5));
      for (let c = c0; c <= c1; c++) out[o + c] = 1;
    }
  }
}

/** The water under a box from loaded tiles, bridges filled, and every cell's distance to land. */
export function buildRaster(get: TileGet, b: Box): Raster {
  const t0 = performance.now();
  const { frame: f, x0, y0, cell, W, H } = b;
  const N = W * H;
  const water = new Uint8Array(N);
  /** bridge segments in cell units, x0, y0, x1, y1 */
  const segs: number[] = [];
  const reachC = BRIDGE_GAP_M / cell;
  for (const [tx, ty, tz] of tilesOf(b)) {
    const t = get(tx, ty, tz);
    const ext = t.extent, nt = 2 ** tz;
    const cx = (px: number) => (f.x(((tx + px / ext) / nt) * 360 - 180) - x0) / cell;
    const cy = (py: number) => {
      const n = Math.PI - (2 * Math.PI * (ty + py / ext)) / nt;
      return (f.y((180 / Math.PI) * Math.atan(Math.sinh(n))) - y0) / cell;
    };
    for (const poly of t.polys) {
      const rings = poly.map((ring) => {
        const out = new Float64Array(ring.length);
        for (let k = 0; k < ring.length; k += 2) {
          out[k] = cx(ring[k]);
          out[k + 1] = cy(ring[k + 1]);
        }
        return out;
      });
      fillPoly(rings, W, H, water);
    }
    for (const line of t.bridges)
      for (let k = 2; k < line.length; k += 2) {
        const xa = cx(line[k - 2]), ya = cy(line[k - 1]), xb = cx(line[k]), yb = cy(line[k + 1]);
        if (Math.max(xa, xb) < -reachC || Math.max(ya, yb) < -reachC || Math.min(xa, xb) > W + reachC || Math.min(ya, yb) > H + reachC) continue;
        segs.push(xa, ya, xb, yb);
      }
  }
  const t1 = performance.now();
  // the gap OSM's water leaves under a bridge: along a bridge line, land with water both ways across the
  // line, BRIDGE_GAP_M or less in all. A road or railway on a viaduct along a shore has water on one side
  // only, and stays land.
  const isWater = (x: number, y: number) => {
    const c = Math.floor(x), r = Math.floor(y);
    return c >= 0 && r >= 0 && c < W && r < H && water[r * W + c] === 1;
  };
  const fills: number[] = [];
  for (let k = 0; k < segs.length; k += 4) {
    const xa = segs[k], ya = segs[k + 1], xb = segs[k + 2], yb = segs[k + 3];
    const L = Math.hypot(xb - xa, yb - ya);
    if (L < 1e-9) continue;
    const dx = (xb - xa) / L, dy = (yb - ya) / L, nx = -dy, ny = dx;
    const n = Math.ceil(L * 4);
    for (let s = 0; s <= n; s++) {
      const px = xa + (xb - xa) * (s / n), py = ya + (yb - ya) * (s / n);
      if (isWater(px, py)) continue;
      let d1 = -1, d2 = -1;
      for (let d = 0.5; d <= reachC; d += 0.5)
        if (isWater(px + nx * d, py + ny * d)) {
          d1 = d;
          break;
        }
      if (d1 < 0) continue;
      for (let d = 0.5; d + d1 <= reachC; d += 0.5)
        if (isWater(px - nx * d, py - ny * d)) {
          d2 = d;
          break;
        }
      if (d2 < 0) continue;
      // dense enough that a diagonal bridge leaves no land cells between its samples
      for (let d = -d2; d <= d1; d += 0.25)
        for (const e of [-0.5, -0.25, 0, 0.25, 0.5]) {
          const c = Math.floor(px + nx * d + dx * e), r = Math.floor(py + ny * d + dy * e);
          if (c >= 0 && r >= 0 && c < W && r < H) fills.push(r * W + c);
        }
    }
  }
  let bridged = 0;
  for (const i of fills)
    if (!water[i]) {
      water[i] = 1;
      bridged++;
    }
  const g = new Float32Array(N);
  for (let i = 0; i < N; i++) g[i] = water[i] ? INF : 0;
  edt(g, W, H);
  for (let i = 0; i < N; i++) g[i] = g[i] >= BIG ? INF : Math.sqrt(g[i]) * cell;
  return { ...b, water, dist: g, bridged, ms: { fill: t1 - t0, edt: performance.now() - t1 } };
}

/** The cell under a plane point, or -1 outside. */
function cellAt(r: Box, x: number, y: number): number {
  const c = Math.floor((x - r.x0) / r.cell), w = Math.floor((y - r.y0) / r.cell);
  return c < 0 || w < 0 || c >= r.W || w >= r.H ? -1 : w * r.W + c;
}

/** The room at a point (longitude, latitude) in a raster, m; -1 outside it or on land. */
export function roomAt(r: Raster, p: LonLat): number {
  const i = cellAt(r, r.frame.x(p[0]), r.frame.y(p[1]));
  return i < 0 || !r.water[i] ? -1 : Math.max(0, roomOf(r, i));
}

// ------------------------------------------------------------------ the search

class Heap {
  k = new Float64Array(1 << 16);
  v = new Int32Array(1 << 16);
  n = 0;
  push(key: number, val: number) {
    if (this.n === this.k.length) {
      const k = new Float64Array(this.n * 2), v = new Int32Array(this.n * 2);
      k.set(this.k);
      v.set(this.v);
      this.k = k;
      this.v = v;
    }
    const K = this.k, V = this.v;
    let i = this.n++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (K[p] <= key) break;
      K[i] = K[p];
      V[i] = V[p];
      i = p;
    }
    K[i] = key;
    V[i] = val;
  }
  pop(): number {
    const K = this.k, V = this.v, top = V[0];
    const n = --this.n;
    if (n > 0) {
      const key = K[n], val = V[n];
      let i = 0;
      for (;;) {
        let m = 2 * i + 1;
        if (m >= n) break;
        if (m + 1 < n && K[m + 1] < K[m]) m++;
        if (K[m] >= key) break;
        K[i] = K[m];
        V[i] = V[m];
        i = m;
      }
      K[i] = key;
      V[i] = val;
    }
    return top;
  }
}

const NB: [number, number][] = [
  [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
  [2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [1, -2], [-1, 2], [-1, -2],
];
const NB_LEN = NB.map(([dx, dy]) => Math.hypot(dx, dy));
const NB_X = Int32Array.from(NB.map((d) => d[0])), NB_Y = Int32Array.from(NB.map((d) => d[1]));
// a knight's move's two crossed cells, as offsets
const NB_M1X = NB.map(([dx]) => Math.trunc(dx / 2)), NB_M1Y = NB.map(([, dy]) => Math.trunc(dy / 2));
const NB_M2X = NB.map(([dx]) => Math.round(dx / 2)), NB_M2Y = NB.map(([, dy]) => Math.round(dy / 2));

interface Search {
  cells: number[];
  /** ground metres along the cells */
  m: number;
  expanded: number;
}

type Buf = { g: Float64Array; from: Int32Array; closed: Uint8Array };

/** A* over 16 neighbours from any of `starts` (each at cost `g0`, cells) to any cell of `goal`, aimed at the
 * cell point (hx, hy). With `goalCost` (cells on from each goal cell, at least its straight distance to
 * (hx, hy)) the search ends at the goal cell of least cost through it; without, at the first goal cell.
 * A step costs its length times the mean of base + k pen at its two cells. */
function astar(r: Raster, pass: Uint8Array, { base, pen }: { base: Float32Array; pen: Float32Array }, k: number, starts: number[], g0: number[], goal: Uint8Array, hx: number, hy: number, goalCost: Float32Array | null, buf: Buf): Search | null {
  const { W, H } = r;
  const { g, from, closed } = buf;
  g.fill(Infinity);
  from.fill(-1);
  closed.fill(0);
  const h = (j: number) => {
    const dx = (j % W) + 0.5 - hx, dy = ((j / W) | 0) + 0.5 - hy;
    return Math.sqrt(dx * dx + dy * dy);
  };
  let best = Infinity;
  const heap = new Heap();
  starts.forEach((s, q) => {
    if (g0[q] < g[s]) {
      g[s] = g0[q];
      heap.push(g0[q] + h(s), s);
    }
  });
  let expanded = 0, end = -1;
  while (heap.n) {
    if (heap.k[0] >= best) break;
    const i = heap.pop();
    if (closed[i]) continue;
    closed[i] = 1;
    expanded++;
    if (goal[i]) {
      if (!goalCost) {
        end = i;
        break;
      }
      if (g[i] + goalCost[i] < best) {
        best = g[i] + goalCost[i];
        end = i;
      }
    }
    const c = i % W, w = (i / W) | 0, wi = base[i] + k * pen[i], gi = g[i];
    for (let n = 0; n < 16; n++) {
      const nc = c + NB_X[n], nr = w + NB_Y[n];
      if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;
      const j = nr * W + nc;
      if (!pass[j] || closed[j]) continue;
      // a knight's move crosses two cells, both sailable
      if (n >= 8 && (!pass[(w + NB_M1Y[n]) * W + c + NB_M1X[n]] || !pass[(w + NB_M2Y[n]) * W + c + NB_M2X[n]])) continue;
      const ng = gi + (NB_LEN[n] * (wi + base[j] + k * pen[j])) / 2;
      if (ng < g[j]) {
        g[j] = ng;
        from[j] = i;
        heap.push(ng + h(j), j);
      }
    }
  }
  if (end < 0) return null;
  const cells: number[] = [];
  for (let i = end; i >= 0; i = from[i]) cells.push(i);
  cells.reverse();
  let m = 0;
  for (let q = 1; q < cells.length; q++) m += Math.hypot((cells[q] % W) - (cells[q - 1] % W), ((cells[q] / W) | 0) - ((cells[q - 1] / W) | 0));
  return { cells, m: m * r.cell, expanded };
}

/** The nearest sailable cell to a plane point within `reach` m, or -1. Where the raster is fine enough to
 * see a pier, the straight line to it must stay in water. */
function nearestPass(r: Raster, pass: Uint8Array, x: number, y: number, reach: number): number {
  const cx = (x - r.x0) / r.cell, cy = (y - r.y0) / r.cell, rad = Math.ceil(reach / r.cell);
  const fc = Math.floor(cx), fr = Math.floor(cy);
  const cand: [number, number][] = [];
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) {
      const c = fc + dx, w = fr + dy;
      if (c < 0 || w < 0 || c >= r.W || w >= r.H) continue;
      const i = w * r.W + c;
      if (!pass[i]) continue;
      const d = ((c + 0.5 - cx) ** 2 + (w + 0.5 - cy) ** 2) * r.cell * r.cell;
      if (d <= reach * reach) cand.push([d, i]);
    }
  cand.sort((p, q) => p[0] - q[0]);
  for (const [, i] of cand.slice(0, 400)) {
    if (r.cell > FINE_CELL_M) return i;
    const px = r.x0 + ((i % r.W) + 0.5) * r.cell, py = r.y0 + (((i / r.W) | 0) + 0.5) * r.cell;
    const n = Math.ceil((Math.hypot(px - x, py - y) / r.cell) * 2) + 1;
    let wet = true;
    for (let s = 0; s <= n && wet; s++) {
      const j = cellAt(r, x + ((px - x) * s) / n, y + ((py - y) * s) / n);
      wet = j >= 0 && r.water[j] === 1;
    }
    if (wet) return i;
  }
  return cand.length ? cand[0][1] : -1;
}

/** Whether a straight stretch between two plane points keeps `HALF_M` off land all along (sampled every
 * half cell). */
function clearLine(r: Raster, ax: number, ay: number, bx: number, by: number): boolean {
  const n = Math.ceil((Math.hypot(bx - ax, by - ay) / r.cell) * 2) + 1, need = needAnywhere(r);
  for (let s = 0; s <= n; s++) {
    const i = cellAt(r, ax + ((bx - ax) * s) / n, ay + ((by - ay) * s) / n);
    if (i < 0 || !r.water[i] || r.dist[i] < need) return false;
  }
  return true;
}

/** Each cell's distance by water from `s`, cells (8 neighbours); Infinity where it cannot be reached. */
function waterDist(r: Raster, pass: Uint8Array, s: number): Float64Array {
  const { W, H } = r;
  const d = new Float64Array(W * H).fill(Infinity);
  const heap = new Heap();
  d[s] = 0;
  heap.push(0, s);
  while (heap.n) {
    const key = heap.k[0], i = heap.pop();
    if (key > d[i]) continue;
    const c = i % W, w = (i / W) | 0;
    for (let n = 0; n < 8; n++) {
      const nc = c + NB_X[n], nr = w + NB_Y[n];
      if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;
      const j = nr * W + nc, nd = key + NB_LEN[n];
      if (pass[j] && nd < d[j]) {
        d[j] = nd;
        heap.push(nd, j);
      }
    }
  }
  return d;
}

/** The cells reachable from `s` on sailable cells (8 neighbours, as the search's). */
function reach(r: Raster, pass: Uint8Array, s: number): Uint8Array {
  const { W, H } = r;
  const seen = new Uint8Array(W * H);
  const q = new Int32Array(W * H);
  let head = 0, tail = 0;
  seen[s] = 1;
  q[tail++] = s;
  while (head < tail) {
    const i = q[head++], c = i % W, w = (i / W) | 0;
    for (let n = 0; n < 8; n++) {
      const nc = c + NB[n][0], nr = w + NB[n][1];
      if (nc < 0 || nr < 0 || nc >= W || nr >= H) continue;
      const j = nr * W + nc;
      if (pass[j] && !seen[j]) {
        seen[j] = 1;
        q[tail++] = j;
      }
    }
  }
  return seen;
}

/** Douglas-Peucker over plane points, splitting where a chord would leave sailable water too, at most
 * `DP_M` (and `DP_ROOM` of the room) off the path. */
function simplify(r: Raster, xs: number[], ys: number[]): number[] {
  const n = xs.length;
  if (n < 3) return xs.map((_, k) => k);
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const st: [number, number][] = [[0, n - 1]];
  while (st.length) {
    const [a, b] = st.pop()!;
    if (b - a < 2) continue;
    const L = Math.hypot(xs[b] - xs[a], ys[b] - ys[a]) || 1e-9;
    let m = -1, md = -1;
    for (let k = a + 1; k < b; k++) {
      const d = Math.abs((xs[b] - xs[a]) * (ys[a] - ys[k]) - (xs[a] - xs[k]) * (ys[b] - ys[a])) / L;
      if (d > md) {
        md = d;
        m = k;
      }
    }
    if (m < 0) continue;
    const i = cellAt(r, xs[m], ys[m]);
    const tol = Math.max(r.cell / 2, Math.min(DP_M, DP_ROOM * (i >= 0 ? roomOf(r, i) : 0)));
    if (md > tol || !clearLine(r, xs[a], ys[a], xs[b], ys[b])) {
      keep[m] = 1;
      st.push([a, m], [m, b]);
    }
  }
  const out: number[] = [];
  for (let k = 0; k < n; k++) if (keep[k]) out.push(k);
  return out;
}

export interface LegStats {
  cells: number;
  /** the finest and coarsest cell searched, m */
  cell: number;
  coarseCell: number;
  /** the coarse plan's tile zoom, 0 when there was none */
  coarseZ: number;
  tiles: number;
  fill: number;
  edt: number;
  search: number;
  searches: number;
  expanded: number;
  /** the penalty's weight the route took (0: the plain shortest; the coarse plan's when there is one) */
  k: number;
  plainM: number;
  bridged: number;
  /** ms in all, without the tiles' fetch */
  ms: number;
}

const newStats = (): LegStats => ({ cells: 0, cell: Infinity, coarseCell: 0, coarseZ: 0, tiles: 0, fill: 0, edt: 0, search: 0, searches: 0, expanded: 0, k: 0, plainM: 0, bridged: 0, ms: 0 });

function addRaster(st: LegStats, r: Raster, tiles: number) {
  st.cells += r.W * r.H;
  st.cell = Math.min(st.cell, r.cell);
  st.coarseCell = Math.max(st.coarseCell, r.cell);
  st.tiles += tiles;
  st.fill += r.ms.fill;
  st.edt += r.ms.edt;
  st.bridged += r.bridged;
}

/** One stretch of a leg's route and the raster it was searched in: longitude, latitude pairs. */
export interface Piece {
  ll: number[];
  box: Box;
  raster: Raster;
}

/** A leg's route before rounding: one piece, or for a leg searched coarse its fine first stretch, the
 * coarse middle and its fine last stretch. A piece's raster is rebuilt from its box when a line is rounded
 * and the raster is gone. */
export type LegRoute =
  | { ok: true; pieces: Piece[]; m: number; stats: LegStats }
  | { ok: false; reason: Refusal; stats?: LegStats };

/** The box a leg between `a` and `b` is searched in; `wide` for the second try. */
export function legBox(a: LonLat, b: LonLat, wide = false): Box {
  const f = new Frame((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
  const ax = f.x(a[0]), ay = f.y(a[1]), bx = f.x(b[0]), by = f.y(b[1]);
  const d = Math.hypot(ax - bx, ay - by);
  const pad = wide ? Math.max(3000, d) : Math.max(800, 0.35 * d);
  return boxOf(f, Math.min(ax, bx) - pad, Math.min(ay, by) - pad, Math.max(ax, bx) + pad, Math.max(ay, by) + pad);
}

/** The fine box round one end of a leg searched coarse. */
function endBox(p: LonLat): Box {
  return boxOf(new Frame(p[0], p[1]), -END_M, -END_M, END_M, END_M, MIN_CELL_M);
}

const boxOnly = (r: Raster): Box => ({ frame: r.frame, x0: r.x0, y0: r.y0, cell: r.cell, W: r.W, H: r.H, z: r.z });

/** The route of one leg between two points on water (berths, or waypoints). Searched on one raster where
 * the leg's box allows a fine cell; otherwise coarse, with each end's first `END_M` or so searched fine.
 * Tried in the leg's box, then, if the waters do not meet there, in a wider one. */
export async function routeLeg(load: TileLoad, a: LonLat, b: LonLat): Promise<LegRoute> {
  if (metres(a, b) > MAX_LEG_M) return { ok: false, reason: "too-long" };
  const t0 = performance.now();
  let fetchMs = 0;
  const timed: TileLoad = async (want) => {
    const t1 = performance.now();
    const get = await load(want);
    fetchMs += performance.now() - t1;
    return get;
  };
  const stats = newStats();
  let last: LegRoute = { ok: false, reason: "no-water-path", stats };
  for (const wide of [false, true]) {
    const box = legBox(a, b, wide);
    let out: LegRoute;
    if (box.cell <= FINE_CELL_M) {
      const tiles = tilesOf(box);
      const get = await timed(tiles);
      if (!get) return { ok: false, reason: "tiles" };
      const r = buildRaster(get, box);
      addRaster(stats, r, tiles.length);
      const p = routeIn(r, a, b, stats, true);
      out = p ? { ok: true, pieces: [p], m: pieceM(p), stats } : { ok: false, reason: "no-water-path", stats };
    } else out = await twoLevel(timed, a, b, box, stats);
    stats.ms = performance.now() - t0 - fetchMs;
    if (out.ok || out.reason !== "no-water-path") return out;
    last = out;
  }
  return last;
}

function pieceM(p: Piece): number {
  let m = 0;
  for (let k = 2; k < p.ll.length; k += 2) m += metres([p.ll[k - 2], p.ll[k - 1]], [p.ll[k], p.ll[k + 1]]);
  return m;
}

function passOf(r: Raster): Uint8Array {
  const N = r.W * r.H, need = needOf(r), pass = new Uint8Array(N);
  for (let i = 0; i < N; i++) pass[i] = r.water[i] && r.dist[i] >= need ? 1 : 0;
  return pass;
}

/** Each sailable cell's cost a metre: `base`, 1 plus the narrow water's pull to the middle, and `pen`, the
 * 400 m preference before its weight, fading out within `FADE_M` of the cell points `ends`. */
function penalty(r: Raster, pass: Uint8Array, ends: [number, number][]): { base: Float32Array; pen: Float32Array } {
  const N = r.W * r.H, pen = new Float32Array(N), base = new Float32Array(N).fill(1);
  for (let i = 0; i < N; i++) {
    if (!pass[i]) continue;
    const room = roomOf(r, i);
    if (room < CENTRE_M) base[i] = 1 + CENTRE_K * (1 - room / CENTRE_M) ** 2;
    const u = 1 - room / CLEAR_M;
    if (u <= 0) continue;
    const c = (i % r.W) + 0.5, w = ((i / r.W) | 0) + 0.5;
    let near = Infinity;
    for (const [x, y] of ends) near = Math.min(near, Math.hypot(c - x, w - y));
    pen[i] = Math.min(1, (near * r.cell) / FADE_M) * u * u;
  }
  return { base, pen };
}

/** The plain shortest path, then the penalised one eased off while it is over `DETOUR` times as long. */
function bestPath(r: Raster, pass: Uint8Array, pen: { base: Float32Array; pen: Float32Array }, starts: number[], g0: number[], goal: Uint8Array, hx: number, hy: number, goalCost: Float32Array | null, st: LegStats, record: boolean): Search | null {
  const N = r.W * r.H;
  const buf = { g: new Float64Array(N), from: new Int32Array(N), closed: new Uint8Array(N) };
  const plain = astar(r, pass, pen, 0, starts, g0, goal, hx, hy, goalCost, buf);
  st.searches++;
  if (!plain) return null;
  st.expanded += plain.expanded;
  if (record) st.plainM += plain.m;
  for (const k of [PENALTY_K, PENALTY_K / 3, PENALTY_K / 10]) {
    const p = astar(r, pass, pen, k, starts, g0, goal, hx, hy, goalCost, buf)!;
    st.searches++;
    st.expanded += p.expanded;
    if (p.m <= DETOUR * plain.m) {
      if (record) st.k = k;
      return p;
    }
  }
  return plain;
}

/** Cells to longitude, latitude pairs, simplified, from exactly `a` to exactly `b`. */
function piece(r: Raster, cells: number[], a: LonLat, b: LonLat): Piece {
  const { W, cell, frame: f } = r;
  const xs = [f.x(a[0])], ys = [f.y(a[1])];
  for (const i of cells) {
    xs.push(r.x0 + ((i % W) + 0.5) * cell);
    ys.push(r.y0 + (((i / W) | 0) + 0.5) * cell);
  }
  xs.push(f.x(b[0]));
  ys.push(f.y(b[1]));
  // the ends join the search's first and last cells straight; the rest is simplified
  const inner = simplify(r, xs.slice(1, -1), ys.slice(1, -1)).map((k) => k + 1);
  const keep = [0, ...inner, xs.length - 1];
  const ll: number[] = [];
  for (let q = 0; q < keep.length; q++) {
    const k = keep[q];
    if (q > 0 && Math.abs(xs[k] - xs[keep[q - 1]]) < 1e-6 && Math.abs(ys[k] - ys[keep[q - 1]]) < 1e-6) continue;
    ll.push(f.lon(xs[k]), f.lat(ys[k]));
  }
  ll[0] = a[0];
  ll[1] = a[1];
  if (ll.length < 4) ll.push(b[0], b[1]);
  ll[ll.length - 2] = b[0];
  ll[ll.length - 1] = b[1];
  return { ll, box: boxOnly(r), raster: r };
}

/** A leg on one raster from point to point, or null when the waters do not meet in it. */
function routeIn(r: Raster, a: LonLat, b: LonLat, st: LegStats, record: boolean): Piece | null {
  const t0 = performance.now();
  const { W, frame: f } = r;
  const pass = passOf(r);
  const ax = f.x(a[0]), ay = f.y(a[1]), bx = f.x(b[0]), by = f.y(b[1]);
  const done = (p: Piece | null) => {
    st.search += performance.now() - t0;
    return p;
  };
  const s = nearestPass(r, pass, ax, ay, BERTH_REACH_M), t = nearestPass(r, pass, bx, by, BERTH_REACH_M);
  if (s < 0 || t < 0 || !reach(r, pass, s)[t]) return done(null);
  const sc = (s % W) + 0.5, sr = ((s / W) | 0) + 0.5, tc = (t % W) + 0.5, tr = ((t / W) | 0) + 0.5;
  const pen = penalty(r, pass, [[sc, sr], [tc, tr]]);
  const goal = new Uint8Array(W * r.H);
  goal[t] = 1;
  const best = bestPath(r, pass, pen, [s], [0], goal, tc, tr, null, st, record)!;
  return done(piece(r, best.cells, a, b));
}

/** A leg searched coarse: the coarse raster's path from water reached from `a` near it to water reached
 * from `b` near it, and fine searches from `a` out to that path and from it in to `b`. */
async function twoLevel(load: TileLoad, a: LonLat, b: LonLat, box0: Box, st: LegStats): Promise<LegRoute> {
  let box: Box | null = null, tiles: [number, number, number][] = [];
  for (const z of [WATER_ZOOM, WATER_ZOOM - 1, WATER_ZOOM - 2]) {
    const bz = { ...box0, z };
    tiles = tilesOf(bz);
    if (tiles.length <= MAX_TILES) {
      box = bz;
      break;
    }
  }
  if (!box) return { ok: false, reason: "too-long", stats: st };
  const fa = endBox(a), fb = endBox(b);
  const ta = tilesOf(fa), tb = tilesOf(fb);
  const get = await load([...tiles, ...ta, ...tb]);
  if (!get) return { ok: false, reason: "tiles", stats: st };
  st.coarseZ = box.z!;
  // each end's fine raster and every cell's distance by water from the end, which is what a coarse start
  // or goal costs (a straight line would cut across a headland)
  const ends: { r: Raster; dist: Float64Array }[] = [];
  for (const [p, fbox, t] of [[a, fa, ta], [b, fb, tb]] as const) {
    const r = buildRaster(get, fbox);
    addRaster(st, r, t.length);
    const t0 = performance.now();
    const pass = passOf(r);
    const s = nearestPass(r, pass, r.frame.x(p[0]), r.frame.y(p[1]), BERTH_REACH_M);
    if (s < 0) {
      st.search += performance.now() - t0;
      return { ok: false, reason: "no-water-path", stats: st };
    }
    ends.push({ r, dist: waterDist(r, pass, s) });
    st.search += performance.now() - t0;
  }
  const R = buildRaster(get, box);
  addRaster(st, R, tiles.length);
  const t0 = performance.now();
  const pass = passOf(R);
  const { W, H, cell, frame: F } = R;
  // the coarse cells either end's fine search reaches, and how far by water: where a coarse search may
  // start or end, at that cost
  const INSET = 100;
  const reached = (e: (typeof ends)[number], x: number, y: number): number => {
    const lon = F.lon(x), lat = F.lat(y), r = e.r;
    const fx = r.frame.x(lon), fy = r.frame.y(lat);
    if (fx < r.x0 + INSET || fy < r.y0 + INSET || fx > r.x0 + r.W * r.cell - INSET || fy > r.y0 + r.H * r.cell - INSET) return Infinity;
    const i = cellAt(r, fx, fy);
    return i < 0 ? Infinity : e.dist[i] * r.cell;
  };
  const starts: number[] = [], g0: number[] = [], goal = new Uint8Array(W * H), goalCost = new Float32Array(W * H);
  const ax = (F.x(a[0]) - R.x0) / cell, ay = (F.y(a[1]) - R.y0) / cell, bx = (F.x(b[0]) - R.x0) / cell, by = (F.y(b[1]) - R.y0) / cell;
  const rad = Math.ceil(END_M / cell);
  let goals = 0;
  for (const [px, py, e, isStart] of [[ax, ay, ends[0], true], [bx, by, ends[1], false]] as const) {
    for (let dy = -rad; dy <= rad; dy++)
      for (let dx = -rad; dx <= rad; dx++) {
        const c = Math.floor(px) + dx, w = Math.floor(py) + dy;
        if (c < 0 || w < 0 || c >= W || w >= H) continue;
        const i = w * W + c;
        if (!pass[i]) continue;
        const m = reached(e, R.x0 + (c + 0.5) * cell, R.y0 + (w + 0.5) * cell);
        if (!Number.isFinite(m)) continue;
        if (isStart) {
          starts.push(i);
          g0.push(m / cell);
        } else {
          goal[i] = 1;
          // at least the straight way to the end, as the search's aim needs
          goalCost[i] = Math.max(m / cell, Math.hypot(c + 0.5 - bx, w + 0.5 - by));
          goals++;
        }
      }
  }
  if (!starts.length || !goals) {
    st.search += performance.now() - t0;
    return { ok: false, reason: "no-water-path", stats: st };
  }
  const pen = penalty(R, pass, [[ax, ay], [bx, by]]);
  const path = bestPath(R, pass, pen, starts, g0, goal, bx, by, goalCost, st, true);
  st.search += performance.now() - t0;
  if (!path) return { ok: false, reason: "no-water-path", stats: st };
  // hand over to the fine searches at the last point of the path still in each end's reach
  const P = path.cells.map((i): LonLat => [F.lon(R.x0 + ((i % W) + 0.5) * cell), F.lat(R.y0 + (((i / W) | 0) + 0.5) * cell)]);
  const inReach = (e: (typeof ends)[number], p: LonLat) => Number.isFinite(reached(e, F.x(p[0]), F.y(p[1])));
  let kA = 0;
  while (kA + 1 < P.length && inReach(ends[0], P[kA + 1])) kA++;
  let kB = P.length - 1;
  while (kB - 1 > 0 && inReach(ends[1], P[kB - 1])) kB--;
  if (kA > kB) {
    // the ends' reaches overlap: hand over once, where both reach
    let j = kB;
    while (j <= kA && !(inReach(ends[0], P[j]) && inReach(ends[1], P[j]))) j++;
    if (j > kA) return { ok: false, reason: "no-water-path", stats: st };
    kA = kB = j;
  }
  const pieces: Piece[] = [];
  const pa = routeIn(ends[0].r, a, P[kA], st, false);
  const pb = routeIn(ends[1].r, P[kB], b, st, false);
  if (!pa || !pb) return { ok: false, reason: "no-water-path", stats: st };
  pieces.push(pa);
  if (kB > kA) {
    const t1 = performance.now();
    pieces.push(piece(R, path.cells.slice(kA + 1, kB), P[kA], P[kB]));
    st.search += performance.now() - t1;
  }
  pieces.push(pb);
  return { ok: true, pieces, m: pieces.reduce((s, p) => s + pieceM(p), 0), stats: st };
}

// ------------------------------------------------------------------ ports

export type PortSnap = { ok: true; landing: LonLat; berth: LonLat } | { ok: false; reason: Refusal };

/** The box a port click near `at` is snapped in. */
function portBox(at: LonLat): Box {
  const f = new Frame(at[0], at[1]);
  const R = PORT_REACH_M + BERTH_REACH_M + 50;
  return boxOf(f, -R, -R, R, R, MIN_CELL_M);
}

/** Where a click by the water puts a port: its landing on the shore (where it is drawn and walked to) and
 * its berth, the nearest water with room for a boat, where routes start and end. */
export async function snapPort(load: TileLoad, at: LonLat): Promise<PortSnap> {
  const box = portBox(at);
  const get = await load(tilesOf(box));
  if (!get) return { ok: false, reason: "tiles" };
  const r = buildRaster(get, box);
  const { W, cell, frame: f } = r;
  const c0 = cellAt(r, 0, 0);
  const onWater = r.water[c0] === 1;
  // the nearest cell of the other kind (water from land, land from water) within reach
  const rad = Math.ceil(PORT_REACH_M / cell);
  const cc = c0 % W, cr = (c0 / W) | 0;
  let other = -1, bd = Infinity;
  for (let dy = -rad; dy <= rad; dy++)
    for (let dx = -rad; dx <= rad; dx++) {
      const i = (cr + dy) * W + cc + dx;
      if (r.water[i] === r.water[c0]) continue;
      const d = dx * dx + dy * dy;
      if (d < bd && d <= rad * rad) {
        bd = d;
        other = i;
      }
    }
  if (other < 0) return { ok: false, reason: onWater ? "on-water" : "far-from-water" };
  // the shore: where the line from the click to that cell first changes kind
  const ox = r.x0 + ((other % W) + 0.5) * cell, oy = r.y0 + (((other / W) | 0) + 0.5) * cell;
  const n = Math.ceil((Math.hypot(ox, oy) / cell) * 4) + 1;
  let lx = ox, ly = oy;
  for (let s = 1; s <= n; s++) {
    const x = (ox * s) / n, y = (oy * s) / n, i = cellAt(r, x, y);
    if (r.water[i] !== r.water[c0]) {
      lx = (ox * (s - 0.5)) / n;
      ly = (oy * (s - 0.5)) / n;
      break;
    }
  }
  // the berth: the nearest cell with room for a boat to the landing
  const brad = Math.ceil(BERTH_REACH_M / cell);
  const lc = Math.floor((lx - r.x0) / cell), lr = Math.floor((ly - r.y0) / cell);
  let berth = -1;
  bd = Infinity;
  for (let dy = -brad; dy <= brad; dy++)
    for (let dx = -brad; dx <= brad; dx++) {
      const c = lc + dx, w = lr + dy;
      if (c < 0 || w < 0 || c >= r.W || w >= r.H) continue;
      const i = w * W + c;
      if (!r.water[i] || roomOf(r, i) < BERTH_ROOM_M) continue;
      const d = dx * dx + dy * dy;
      if (d < bd && d <= brad * brad) {
        bd = d;
        berth = i;
      }
    }
  if (berth < 0) return { ok: false, reason: "far-from-water" };
  const bx = r.x0 + ((berth % W) + 0.5) * cell, by = r.y0 + (((berth / W) | 0) + 0.5) * cell;
  return { ok: true, landing: [f.lon(lx), f.lat(ly)], berth: [f.lon(bx), f.lat(by)] };
}

/** What is under a point: water, and whether a port could go there (land by the water). */
export async function probe(load: TileLoad, at: LonLat): Promise<{ water: boolean; shore: boolean } | null> {
  const f = new Frame(at[0], at[1]);
  const R = PORT_REACH_M + 10;
  const box = boxOf(f, -R, -R, R, R, MIN_CELL_M);
  const get = await load(tilesOf(box));
  if (!get) return null;
  const r = buildRaster(get, box);
  const c0 = cellAt(r, 0, 0);
  if (r.water[c0]) return { water: true, shore: false };
  const rad = PORT_REACH_M / r.cell, cc = c0 % r.W, cr = (c0 / r.W) | 0;
  for (let i = 0; i < r.water.length; i++)
    if (r.water[i] && ((i % r.W) - cc) ** 2 + (((i / r.W) | 0) - cr) ** 2 <= rad * rad) return { water: false, shore: true };
  return { water: false, shore: false };
}

// ------------------------------------------------------------------ a whole line

export const KIND_ROUTE = 0;
export const KIND_PORT = 1;
export const KIND_WAYPOINT = 2;

export interface LinePoint {
  /** a port's berth or a waypoint on water */
  at: LonLat;
  kind: typeof KIND_PORT | typeof KIND_WAYPOINT;
}

/** A line's route in the F1 -> F2 format (notes/T-2177.md): points, their kinds and speeds, m/s. */
export interface LineRoute {
  /** every leg has a route */
  ok: boolean;
  /** longitude, latitude pairs */
  ll: number[];
  kinds: number[];
  speeds: number[];
  /** ground metres */
  m: number;
  /** each leg: its first and last point in `ll`, its metres, and its refusal if it has no route (then
   * it is drawn straight) */
  legs: { from: number; to: number; m: number; reason?: Refusal }[];
}

/** a leg's searched route, its pieces each with the raster it was searched in, or its refusal */
export type LegPart = { ok: true; pieces: { ll: number[]; raster: Raster }[] } | { ok: false; reason: Refusal };
type Part = { ok: true; ll: number[]; raster: Raster } | { ok: false; reason: Refusal };
/** where one piece of a leg meets the next: passed as a waypoint is, written as a route point */
const KIND_JOINT = 3;

interface V {
  x: number;
  y: number;
  /** the leg whose raster holds this point */
  leg: number;
  kind: number;
  /** a bend's radius (m) for points on an arc; Infinity on a straight */
  r: number;
  /** the arc a point is on (consecutive points of one arc share it); 0 for none */
  arc?: number;
  /** a waypoint whose turn could not be eased */
  sharp?: boolean;
  /** on a leg with no route */
  bad?: boolean;
}

/** Join a line's legs: their corners rounded as wide as the water allows, waypoints passed smoothly,
 * then each point's speed. `legs[k]` runs from `points[k]` to `points[k + 1]`. */
export function assembleLine(line: LinePoint[], parts: LegPart[]): LineRoute {
  // each piece of a leg as a leg of its own, joined at joints
  const points: { at: LonLat; kind: number }[] = [line[0]], legs: Part[] = [];
  parts.forEach((L, k) => {
    if (L.ok)
      L.pieces.forEach((p, j) => {
        legs.push({ ok: true, ll: p.ll, raster: p.raster });
        points.push(j + 1 < L.pieces.length ? { at: [p.ll[p.ll.length - 2], p.ll[p.ll.length - 1]], kind: KIND_JOINT } : line[k + 1]);
      });
    else {
      legs.push(L);
      points.push(line[k + 1]);
    }
  });
  const n = points.length;
  let lo = Infinity, hi = -Infinity, la = Infinity, lb = -Infinity;
  for (const p of points) {
    lo = Math.min(lo, p.at[0]);
    hi = Math.max(hi, p.at[0]);
    la = Math.min(la, p.at[1]);
    lb = Math.max(lb, p.at[1]);
  }
  const F = new Frame((lo + hi) / 2, (la + lb) / 2);
  const toLeg = (leg: number, x: number, y: number) => {
    const L = legs[leg];
    if (!L.ok) return -1;
    const r = L.raster;
    return cellAt(r, r.frame.x(F.lon(x)), r.frame.y(F.lat(y)));
  };
  /** sailable in the leg's raster */
  const ok = (leg: number, x: number, y: number) => {
    const L = legs[leg];
    if (!L.ok) return true;
    const i = toLeg(leg, x, y);
    return i >= 0 && L.raster.water[i] === 1 && L.raster.dist[i] >= needAnywhere(L.raster);
  };
  const okLine = (leg: number, ax: number, ay: number, bx: number, by: number) => {
    const L = legs[leg];
    if (!L.ok) return true;
    const step = L.raster.cell / 2, k = Math.ceil(Math.hypot(bx - ax, by - ay) / step) + 1;
    for (let s = 0; s <= k; s++) if (!ok(leg, ax + ((bx - ax) * s) / k, ay + ((by - ay) * s) / k)) return false;
    return true;
  };
  // the corners, leg by leg, each leg's ends shared with the next
  const vs: V[] = [];
  for (let k = 0; k + 1 < n; k++) {
    const L = legs[k];
    const ll = L.ok ? L.ll : [points[k].at[0], points[k].at[1], points[k + 1].at[0], points[k + 1].at[1]];
    for (let q = k === 0 ? 0 : 2; q < ll.length; q += 2) {
      const last = q === ll.length - 2;
      vs.push({ x: F.x(ll[q]), y: F.y(ll[q + 1]), leg: last && k + 2 < n ? k + 1 : k, kind: q === 0 ? points[k].kind : last ? points[k + 1].kind : KIND_ROUTE, r: Infinity, bad: !L.ok });
    }
  }
  // a waypoint is passed straight: two corners either side of it, on the line through it halfway between
  // the directions in and out, take its turn
  const out: V[] = [];
  for (let q = 0; q < vs.length; q++) {
    const Q = vs[q];
    if ((Q.kind !== KIND_WAYPOINT && Q.kind !== KIND_JOINT) || q === 0 || q === vs.length - 1 || Q.bad || vs[q - 1].bad || vs[q + 1].bad) {
      out.push(Q);
      continue;
    }
    const P = vs[q - 1], S = vs[q + 1];
    const lu = Math.hypot(Q.x - P.x, Q.y - P.y), lv = Math.hypot(S.x - Q.x, S.y - Q.y);
    const tx = (Q.x - P.x) / lu + (S.x - Q.x) / lv, ty = (Q.y - P.y) / lu + (S.y - Q.y) / lv, lt = Math.hypot(tx, ty);
    let placed = false;
    if (lt > 0.2) {
      const legIn = legs[Q.leg - 1] ? Q.leg - 1 : Q.leg, legOut = Q.leg;
      for (let L = Math.min(R_MAX_M / 2, lu / 2, lv / 2); L >= 5; L *= 0.6) {
        const e1x = Q.x - (tx / lt) * L, e1y = Q.y - (ty / lt) * L, e2x = Q.x + (tx / lt) * L, e2y = Q.y + (ty / lt) * L;
        if (okLine(legIn, P.x, P.y, e1x, e1y) && okLine(legIn, e1x, e1y, Q.x, Q.y) && okLine(legOut, Q.x, Q.y, e2x, e2y) && okLine(legOut, e2x, e2y, S.x, S.y)) {
          out.push({ x: e1x, y: e1y, leg: legIn, kind: KIND_ROUTE, r: Infinity }, { ...Q, leg: legOut }, { x: e2x, y: e2y, leg: legOut, kind: KIND_ROUTE, r: Infinity });
          placed = true;
          break;
        }
      }
    }
    if (!placed) out.push({ ...Q, sharp: true });
  }
  // round every corner but ports (boats stop there) and waypoints (passed straight, or sharp)
  const pts: V[] = [out[0]];
  let arcs = 0;
  for (let q = 1; q + 1 < out.length; q++) {
    const P = out[q - 1], Q = out[q], S = out[q + 1];
    if (Q.kind !== KIND_ROUTE || Q.bad || S.bad || P.bad) {
      pts.push(Q);
      continue;
    }
    const ux = P.x - Q.x, uy = P.y - Q.y, vx = S.x - Q.x, vy = S.y - Q.y;
    const lu = Math.hypot(ux, uy), lv = Math.hypot(vx, vy);
    const ang = Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / (lu * lv))));
    if (!(lu > 0 && lv > 0) || ang > Math.PI - 1e-3) {
      pts.push(Q);
      continue;
    }
    const t = Math.tan(ang / 2);
    // half of each side, so the next corner's arc has the other half
    const shareU = P.kind === KIND_ROUTE ? 0.5 : 1, shareV = S.kind === KIND_ROUTE ? 0.5 : 1;
    let r = Math.min(R_MAX_M, Math.min(shareU * lu, shareV * lv) * t);
    // a corner no arc fits is taken as a bend as wide as its two sides allow (in a narrow channel the
    // search's own cells make many small kinks)
    const rKink = Math.max(R_MIN_M, r);
    let arc: V[] | null = null;
    for (; r >= R_MIN_M; r *= 0.6) {
      const dd = r / t;
      const t1x = Q.x + (ux / lu) * dd, t1y = Q.y + (uy / lu) * dd, t2x = Q.x + (vx / lv) * dd, t2y = Q.y + (vy / lv) * dd;
      const bx = ux / lu + vx / lv, by = uy / lu + vy / lv, lb = Math.hypot(bx, by);
      const cx = Q.x + ((bx / lb) * r) / Math.sin(ang / 2), cy = Q.y + ((by / lb) * r) / Math.sin(ang / 2);
      const a1 = Math.atan2(t1y - cy, t1x - cx);
      let da = Math.atan2(t2y - cy, t2x - cx) - a1;
      while (da > Math.PI) da -= 2 * Math.PI;
      while (da < -Math.PI) da += 2 * Math.PI;
      const L = legs[Q.leg];
      const step = Math.min(ARC_STEP_M, L.ok ? L.raster.cell / 2 : ARC_STEP_M);
      const k = Math.max(2, Math.ceil((Math.abs(da) * r) / step));
      const cand: V[] = [];
      let fits = true;
      for (let s = 0; s <= k && fits; s++) {
        const a = a1 + (da * s) / k, x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
        if (!ok(Q.leg, x, y)) fits = false;
        cand.push({ x, y, leg: Q.leg, kind: KIND_ROUTE, r });
      }
      if (fits) {
        arc = cand;
        break;
      }
    }
    if (arc) {
      // the route keeps a point every ARC_STEP_M or so; the finer samples were for the water test
      const spacing = ((Math.PI - ang) * r) / (arc.length - 1);
      const every = Math.max(1, Math.floor(ARC_STEP_M / spacing));
      arcs++;
      for (let s = 0; s < arc.length; s++) if (s % every === 0 || s === arc.length - 1) pts.push({ ...arc[s], arc: arcs });
    } else pts.push({ ...Q, r: rKink });
  }
  pts.push(out[out.length - 1]);
  // speeds: each point's room and bend; straight stretches sampled every STEP_M, a point kept where the
  // speed has changed
  const roomV = (leg: number, x: number, y: number) => {
    const L = legs[leg];
    if (!L.ok) return V_CHANNEL;
    const i = toLeg(leg, x, y);
    const room = i < 0 || !L.raster.water[i] ? 0 : Math.max(0, roomOf(L.raster, i));
    const u = Math.max(0, Math.min(1, (room - ROOM_CHANNEL_M) / (ROOM_OPEN_M - ROOM_CHANNEL_M)));
    return V_CHANNEL + u * (V_OPEN - V_CHANNEL);
  };
  const bendV = (p: V) => (p.sharp ? Math.sqrt(BEND_A * R_MIN_M) : Number.isFinite(p.r) ? Math.sqrt(BEND_A * p.r) : Infinity);
  const res: LineRoute = { ok: parts.every((l) => l.ok), ll: [], kinds: [], speeds: [], m: 0, legs: [] };
  const push = (x: number, y: number, kind: number, v: number) => {
    const L = res.ll.length;
    if (L) res.m += Math.hypot(x - F.x(res.ll[L - 2]), y - F.y(res.ll[L - 1]));
    res.ll.push(F.lon(x), F.lat(y));
    res.kinds.push(kind);
    res.speeds.push(v);
  };
  let legNo = 0, legFrom = 0, legM0 = 0;
  for (let q = 0; q < pts.length; q++) {
    const p = pts[q];
    const vp = Math.min(roomV(p.leg, p.x, p.y), bendV(p));
    if (q > 0) {
      const a = pts[q - 1], d = Math.hypot(p.x - a.x, p.y - a.y);
      if (!(a.arc && a.arc === p.arc)) {
        // a straight stretch: a point where the speed has changed, and one within STEP_M of the end
        // when the end is slower or faster, so its speed holds only near it
        const k = Math.ceil(d / STEP_M) - 1;
        let lastV = res.speeds[res.speeds.length - 1];
        for (let s = 1; s <= k; s++) {
          const f = (s * STEP_M) / d;
          const x = a.x + (p.x - a.x) * f, y = a.y + (p.y - a.y) * f, v = roomV(p.leg, x, y);
          if (Math.abs(v - lastV) >= DV_KEEP || (s === k && Math.abs(lastV - vp) >= DV_KEEP)) {
            push(x, y, KIND_ROUTE, v);
            lastV = v;
          }
        }
      }
    }
    push(p.x, p.y, p.kind === KIND_JOINT ? KIND_ROUTE : p.kind, vp);
    if (q > 0 && (p.kind === KIND_PORT || p.kind === KIND_WAYPOINT)) {
      const idx = res.kinds.length - 1;
      const L = parts[legNo];
      res.legs.push({ from: legFrom, to: idx, m: res.m - legM0, reason: L && !L.ok ? L.reason : undefined });
      legNo++;
      legFrom = idx;
      legM0 = res.m;
    }
  }
  return res;
}

/** Seconds a boat takes over a route with its speeds, from rest at the first point to rest at the last
 * (stopping at every port between), accelerating and braking at `ACCEL`. For checks and estimates; the
 * sim has its own. */
export function runSeconds(route: { ll: number[]; kinds: number[]; speeds: number[] }): number {
  const n = route.kinds.length;
  if (n < 2) return 0;
  const F = new Frame(route.ll[0], route.ll[1]);
  const seg: number[] = [], cap: number[] = [];
  for (let i = 1; i < n; i++) {
    seg.push(Math.hypot(F.x(route.ll[2 * i]) - F.x(route.ll[2 * i - 2]), F.y(route.ll[2 * i + 1]) - F.y(route.ll[2 * i - 1])));
    cap.push(Math.min(route.speeds[i - 1], route.speeds[i]));
  }
  // speed at each point: 0 at ports, at most the caps either side, reachable forwards and backwards
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = route.kinds[i] === KIND_PORT ? 0 : Math.min(cap[i - 1] ?? Infinity, cap[i] ?? Infinity);
  for (let i = 1; i < n; i++) v[i] = Math.min(v[i], Math.sqrt(v[i - 1] ** 2 + 2 * ACCEL * seg[i - 1]));
  for (let i = n - 2; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[i + 1] ** 2 + 2 * ACCEL * seg[i]));
  let t = 0;
  for (let i = 0; i < n - 1; i++) {
    // accelerate from v[i], cruise at cap, brake to v[i + 1]
    const c = cap[i], d = seg[i];
    const da = Math.max(0, (c * c - v[i] ** 2) / (2 * ACCEL)), db = Math.max(0, (c * c - v[i + 1] ** 2) / (2 * ACCEL));
    if (da + db <= d) t += (c - v[i]) / ACCEL + (c - v[i + 1]) / ACCEL + (d - da - db) / c;
    else {
      const peak = Math.sqrt((2 * ACCEL * d + v[i] ** 2 + v[i + 1] ** 2) / 2);
      t += (peak - v[i]) / ACCEL + (peak - v[i + 1]) / ACCEL;
    }
  }
  return t;
}
