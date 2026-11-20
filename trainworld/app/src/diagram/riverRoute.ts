// The schematic's narrow rivers (T-3358; docs/ui-diagram.md "The rubber sheet", "Narrow rivers"; notes/T-3358.md).
// Each river course some hop crosses is drawn after the layout as one band of one width: a path at 0, 45 and 90 degrees
// on a half-pitch grid, near its course carried into the drawing, holding every station near it on its own bank by
// "fences" (the drawn straight way between two stations the river does not part on the ground, dear to cross), the
// lines a cost and never walls. Then a light straightening pass, and the stations still on the wrong bank are handed
// back so the lines round them can be laid out again (sheetLines.ts `sheetLayout`). Pure: no DOM, no fetch.

import { filleted, fitOcti, segD } from "./borders";
import { chamfer, type RealWater } from "./waterReal";
import type { DiagramGraph } from "./graph";
import type { DiagramLayout } from "./layout";
import { drawing, pillSize } from "./draw";
import { fillZones, type CarryMap, type MapGrid } from "./sheetThin";

/** a river's course: the river finder's arcs joined the widest way at each junction, never stepping back onto a node
 * already in it; per arc its index, half width (metres), length and start and end along the course, and which way it
 * runs; its points (graph metres) and the crossings of its arcs, along it */
export interface RiverCourse {
  arcs: { arc: number; half: number; l: number; fwd: boolean; s0: number; s1: number }[];
  startNode: number;
  endNode: number;
  pts: Float64Array;
  river: number;
  half: number;
  cross: { e: number; s: number; t: number }[];
}

/** the routing's costs, in pitches of length: the grid's cell, a 45 and a 90 degree bend, each pitch inside another
 * hop's casing and inside one the river crosses on the ground, each cell of a fence, each pitch inside a pill grown by
 * half the band; the corridor (free within freeDev of the carried middle, then ((d - freeDev) / devScale)^2 a pitch, no
 * cell past corrMax), the grid's margin round the middle, a pill's clearance; and the light straightening after, how
 * far (pitches) the path may move to drop a bend (0 none) */
export const ROUTE = { cell: 0.5, B45: 10, B90: 30, wCross: 10, wOwn: 1, fence: 3000, pillCost: 150, freeDev: 2, devScale: 4, margin: 25, clearPad: 0.2, corrMax: 15, straight: 2 };
export type RouteOptions = typeof ROUTE;

/** stations within NEAR_M of a course (and over 30 m off it) hold its banks; pairs of them within PAIR_M are checked,
 * and fenced within FENCE_M, as are pairs with a station within FAR_M of it; a course is routed from EXT_M before its
 * first crossing to EXT_M past its last, or on to its end where that meets wider water within END_M; its ground
 * sampled every STEP_M (metres) */
const NEAR_M = 1500, PAIR_M = 3000, FENCE_M = 2000, FAR_M = 3000, EXT_M = 3000, END_M = 8000, STEP_M = 40;
/** an end on land runs on at least this far past the crossing nearest it, pitches */
const FREE_P = 3;
/** past a routed stretch, the map's own river water under this many pitches across goes (the old bands' least width) */
const THIN_W = 3.5;
/** how far along a river a crossing moves its carried middle, metres */
const CORR_M = 3000;
/** a band's width, pitches: at least three lines' width; a course whose own would be wider than WMAX is the grid's area
 * water (the Hudson, the East River); toward a sea it may widen to WWIDE */
const WMIN = 3 * (5 / 6.4), WMAX = 8, WWIDE = 12;
/** corners rounded at this radius, pitches (the lines' own); a band's inside edge at a bend at least BEND_IN (T-3833) */
const BEND_R = 2.5, BEND_IN = 1;

export function riverCourses(real: RealWater): RiverCourse[] {
  const arcs = real.arcs, byNode = new Map<number, number[]>();
  arcs.forEach((A, i) => { for (const v of [A.a, A.b]) { let l = byNode.get(v); if (!l) byNode.set(v, (l = [])); l.push(i); } });
  const used = new Uint8Array(arcs.length), order = arcs.map((_, i) => i).sort((p, q) => arcs[q].half - arcs[p].half || p - q), out: RiverCourse[] = [];
  const len = (p: ArrayLike<number>) => { let t = 0; for (let q = 2; q < p.length; q += 2) t += Math.hypot(p[q] - p[q - 2], p[q + 1] - p[q - 1]); return t; };
  for (const start of order) {
    if (used[start]) continue;
    used[start] = 1;
    const A0 = arcs[start], seen = new Set([A0.a, A0.b]);
    const list: RiverCourse["arcs"] = [{ arc: start, half: A0.half, l: len(A0.pts), fwd: true, s0: 0, s1: 0 }];
    let endNode = A0.b, startNode = A0.a;
    for (const atEnd of [true, false]) {
      let node = atEnd ? A0.b : A0.a;
      for (;;) {
        let best = -1;
        for (const j of byNode.get(node) ?? []) {
          const B = arcs[j], far = B.a === node ? B.b : B.a;
          if (!used[j] && B.river === A0.river && !seen.has(far) && (best < 0 || B.half > arcs[best].half)) best = j;
        }
        if (best < 0) break;
        used[best] = 1;
        const B = arcs[best], fwdB = B.a === node, far = fwdB ? B.b : B.a;
        seen.add(far);
        const item = { arc: best, half: B.half, l: len(B.pts), fwd: atEnd ? fwdB : !fwdB, s0: 0, s1: 0 };
        if (atEnd) list.push(item);
        else list.unshift(item);
        node = far;
      }
      if (atEnd) endNode = node;
      else startNode = node;
    }
    let s = 0, wsum = 0, np = 0;
    const pts: number[] = [], cross: RiverCourse["cross"] = [];
    for (const it of list) {
      (it.s0 = s), (s += it.l), (it.s1 = s);
      const P = arcs[it.arc].pts, m = P.length / 2;
      for (let k = 0; k < m; k++) { const q = it.fwd ? k : m - 1 - k; if (pts.length && k === 0) continue; pts.push(P[2 * q], P[2 * q + 1]); }
      for (const c of arcs[it.arc].cross) cross.push({ e: c.e, s: it.s0 + (it.fwd ? c.s : it.l - c.s), t: c.t });
      (wsum += it.half * m), (np += m);
    }
    cross.sort((a, b) => a.s - b.s);
    out.push({ arcs: list, startNode, endNode, pts: Float64Array.from(pts), river: A0.river, half: wsum / np, cross });
  }
  return out;
}

/** the nearest point of polyline p to (x, y): its distance, which side (sign), whether it falls inside the line's
 * span (not past an end), and how far along */
export function nearestOn(p: ArrayLike<number>, x: number, y: number): { d: number; side: number; inside: boolean; s: number } {
  let best = Infinity, side = 0, inside = false, bs = 0, cum = 0;
  const m = p.length / 2;
  for (let q = 0; q + 1 < m; q++) {
    const ax = p[2 * q], ay = p[2 * q + 1], dx = p[2 * q + 2] - ax, dy = p[2 * q + 3] - ay, l2 = dx * dx + dy * dy, l = Math.sqrt(l2);
    if (l2 > 1e-12) {
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)), d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
      if (d < best) (best = d), (side = Math.sign(dx * (y - ay) - dy * (x - ax))), (inside = !((q === 0 && t <= 0) || (q === m - 2 && t >= 1))), (bs = cum + t * l);
    }
    cum += l;
  }
  return { d: best, side, inside, s: bs };
}

const segX = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number, dx: number, dy: number) => {
  const d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax), d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
  const d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx), d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
  return d1 * d2 < 0 && d3 * d4 < 0;
};
/** how often segment a-b crosses polyline p */
export function crossCount(ax: number, ay: number, bx: number, by: number, p: ArrayLike<number>): number {
  const x0 = Math.min(ax, bx), x1 = Math.max(ax, bx), y0 = Math.min(ay, by), y1 = Math.max(ay, by);
  let c = 0;
  for (let q = 0; q + 3 < p.length; q += 2) {
    const px = p[q], py = p[q + 1], qx = p[q + 2], qy = p[q + 3];
    if (Math.max(px, qx) < x0 || Math.min(px, qx) > x1 || Math.max(py, qy) < y0 || Math.min(py, qy) > y1) continue;
    if (segX(ax, ay, bx, by, px, py, qx, qy)) c++;
  }
  return c;
}
const polyCross = (a: ArrayLike<number>, b: ArrayLike<number>) => { let c = 0; for (let i = 0; i + 3 < a.length; i += 2) c += crossCount(a[i], a[i + 1], a[i + 2], a[i + 3], b); return c; };
const cumOf = (p: ArrayLike<number>) => { const c = [0]; for (let q = 2; q < p.length; q += 2) c.push(c[c.length - 1] + Math.hypot(p[q] - p[q - 2], p[q + 1] - p[q - 1])); return c; };
const at = (p: ArrayLike<number>, cum: number[], s: number): [number, number] => {
  s = Math.max(0, Math.min(cum[cum.length - 1], s));
  let lo = 0, hi = cum.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (cum[mid] < s) lo = mid; else hi = mid; }
  const l = cum[hi] - cum[lo] || 1, f = (s - cum[lo]) / l;
  return [p[2 * lo] + f * (p[2 * hi] - p[2 * lo]), p[2 * lo + 1] + f * (p[2 * hi + 1] - p[2 * lo + 1])];
};
const lengthOf = (p: ArrayLike<number>) => { let t = 0; for (let q = 0; q + 3 < p.length; q += 2) t += Math.hypot(p[q + 2] - p[q], p[q + 3] - p[q + 1]); return t; };
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
/** bends in a polyline */
export function bendsOf(p: ArrayLike<number>): number {
  let b = 0;
  for (let q = 2; q + 3 < p.length; q += 2) {
    const a = Math.atan2(p[q + 1] - p[q - 1], p[q] - p[q - 2]), c = Math.atan2(p[q + 3] - p[q + 1], p[q + 2] - p[q]);
    if (Math.abs(Math.atan2(Math.sin(c - a), Math.cos(c - a))) > 0.01) b++;
  }
  return b;
}

class Heap {
  k: number[] = [];
  v: number[] = [];
  get size() { return this.k.length; }
  push(k: number, v: number) { const K = this.k, V = this.v; let i = K.length; K.push(k); V.push(v); while (i > 0) { const p = (i - 1) >> 1; if (K[p] <= k) break; K[i] = K[p]; V[i] = V[p]; i = p; } K[i] = k; V[i] = v; }
  pop(): [number, number] {
    const K = this.k, V = this.v, rk = K[0], rv = V[0], lk = K.pop()!, lv = V.pop()!, n = K.length;
    if (n) { let i = 0; for (;;) { let c = 2 * i + 1; if (c >= n) break; if (c + 1 < n && K[c + 1] < K[c]) c++; if (K[c] >= lk) break; K[i] = K[c]; V[i] = V[c]; i = c; } K[i] = lk; V[i] = lv; }
    return [rk, rv];
  }
}
const DX = [1, 1, 0, -1, -1, -1, 0, 1], DY = [0, 1, 1, 1, 0, -1, -1, -1];

/** one pill: its axis's ends and its radius, pitches */
type Pill = [number, number, number, number, number];
interface RouteIn {
  /** the carried middle, pitches, from the end routed from */
  mid: number[];
  /** the band's width at each sample of `mid` */
  wSample: Float64Array;
  pills: Pill[][];
  /** the stations in the grid's box, and their drawn places */
  stations: number[];
  SX: Float64Array;
  SY: Float64Array;
  hops: { pts: number[]; k: number; own: boolean }[];
  fences: number[][];
  /** wider water (the grid's, and bands routed before) an end may meet, or null */
  wetAt: ((x: number, y: number) => boolean) | null;
  startWet: boolean;
  endWet: boolean;
  /** the samples of `mid` at the first and last crossings: a wet start lies before the first, a wet goal past the last */
  kFirst: number;
  kLast: number;
}
/** the path (pitches) and each of its runs' width */
interface RouteOut { path: number[]; segW: number[]; fallback: boolean; ms: number }

/** A* over the half-pitch grid with 8 directions and the bend costs; a station's own place grown by half the band is a
 * wall where a way round exists, else a cost like the rest of its pill */
function routeRiver(r: RouteIn, o: RouteOptions): RouteOut | null {
  const t0 = performance.now(), h = o.cell, { mid, wSample } = r;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let q = 0; q < mid.length; q += 2) (x0 = Math.min(x0, mid[q])), (x1 = Math.max(x1, mid[q])), (y0 = Math.min(y0, mid[q + 1])), (y1 = Math.max(y1, mid[q + 1]));
  (x0 -= o.margin), (y0 -= o.margin), (x1 += o.margin), (y1 += o.margin);
  const W = Math.ceil((x1 - x0) / h) + 1, H = Math.ceil((y1 - y0) / h) + 1, NC = W * H;
  const cx = (c: number) => x0 + (c % W) * h, cy = (c: number) => y0 + Math.floor(c / W) * h;
  const ci = (x: number) => Math.round((x - x0) / h), cj = (y: number) => Math.round((y - y0) / h);
  // each cell's nearest middle sample and its distance (a two-pass chamfer)
  const gd = new Float32Array(NC).fill(1e9), lab = new Int32Array(NC).fill(-1), m = mid.length / 2;
  for (let k = 0; k + 1 < m; k++) {
    const ax = mid[2 * k], ay = mid[2 * k + 1], bx = mid[2 * k + 2], by = mid[2 * k + 3], parts = Math.max(1, Math.ceil((2 * Math.hypot(bx - ax, by - ay)) / h));
    for (let q = 0; q <= parts; q++) {
      const i = ci(ax + ((bx - ax) * q) / parts), j = cj(ay + ((by - ay) * q) / parts);
      if (i >= 0 && j >= 0 && i < W && j < H) (gd[j * W + i] = 0), (lab[j * W + i] = q * 2 < parts ? k : k + 1);
    }
  }
  if (m === 1) { const i = ci(mid[0]), j = cj(mid[1]); if (i >= 0 && j >= 0 && i < W && j < H) (gd[j * W + i] = 0), (lab[j * W + i] = 0); }
  const relax = (c: number, d: number, w: number) => { if (gd[d] + w < gd[c]) (gd[c] = gd[d] + w), (lab[c] = lab[d]); };
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { const c = j * W + i; if (i > 0) relax(c, c - 1, 1); if (j > 0) { relax(c, c - W, 1); if (i > 0) relax(c, c - W - 1, Math.SQRT2); if (i < W - 1) relax(c, c - W + 1, Math.SQRT2); } }
  for (let j = H - 1; j >= 0; j--) for (let i = W - 1; i >= 0; i--) { const c = j * W + i; if (i < W - 1) relax(c, c + 1, 1); if (j < H - 1) { relax(c, c + W, 1); if (i < W - 1) relax(c, c + W + 1, Math.SQRT2); if (i > 0) relax(c, c + W - 1, Math.SQRT2); } }
  const wc = new Float32Array(NC);
  let wMax = 0;
  for (let c = 0; c < NC; c++) (wc[c] = wSample[Math.min(wSample.length - 1, Math.max(0, lab[c]))]), (wMax = Math.max(wMax, wc[c]));
  const step = new Float32Array(NC), pillOf = new Int32Array(NC).fill(-1), fence = new Uint8Array(NC), wall = new Uint8Array(NC);
  for (let c = 0; c < NC; c++) { const dv = Math.max(0, gd[c] * h - o.freeDev) / o.devScale; step[c] = gd[c] * h > o.corrMax ? Infinity : 1 + dv * dv; }
  const stampSeg = (ax: number, ay: number, bx: number, by: number, rad: number, fn: (c: number, d: number) => void) => {
    const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - rad - x0) / h)), i1 = Math.min(W - 1, Math.ceil((Math.max(ax, bx) + rad - x0) / h));
    const j0 = Math.max(0, Math.floor((Math.min(ay, by) - rad - y0) / h)), j1 = Math.min(H - 1, Math.ceil((Math.max(ay, by) + rad - y0) / h));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) fn(j * W + i, segD(x0 + i * h, y0 + j * h, ax, ay, bx, by));
  };
  // the lines: inside a hop's casing (k strokes of 0.78 and 0.2 of casing each side)
  const lineC = new Float32Array(NC);
  for (const hp of r.hops) {
    const rad = (hp.k * 0.78) / 2 + 0.2, add = hp.own ? o.wOwn : o.wCross;
    for (let q = 0; q + 3 < hp.pts.length; q += 2) stampSeg(hp.pts[q], hp.pts[q + 1], hp.pts[q + 2], hp.pts[q + 3], rad, (c, d) => { if (d <= rad) lineC[c] = Math.max(lineC[c], add); });
  }
  for (let c = 0; c < NC; c++) step[c] += lineC[c];
  for (const s of r.stations) {
    for (const p of r.pills[s]) stampSeg(p[0], p[1], p[2], p[3], p[4] + wMax / 2 + o.clearPad, (c, d) => { if (d <= p[4] + wc[c] / 2 + o.clearPad) pillOf[c] = s; });
    stampSeg(r.SX[s], r.SY[s], r.SX[s], r.SY[s], wMax / 2 + o.clearPad, (c, d) => { if (d <= wc[c] / 2 + o.clearPad) wall[c] = 1; });
  }
  // fences: the supercover cells of each drawn same-bank way
  for (const f of r.fences) {
    const ax = f[0], ay = f[1], bx = f[2], by = f[3], n = Math.max(1, Math.ceil((4 * Math.hypot(bx - ax, by - ay)) / h));
    let pi = ci(ax), pj = cj(ay);
    for (let q = 0; q <= n; q++) {
      const i = ci(ax + ((bx - ax) * q) / n), j = cj(ay + ((by - ay) * q) / n);
      if (i !== pi && j !== pj && pi >= 0 && pi < W && j >= 0 && j < H) fence[j * W + pi] = 1;
      if (i >= 0 && j >= 0 && i < W && j < H) fence[j * W + i] = 1;
      (pi = i), (pj = j);
    }
  }
  const near = (x: number, y: number, rad: number) => { const out: number[] = []; for (let j = cj(y - rad); j <= cj(y + rad); j++) for (let i = ci(x - rad); i <= ci(x + rad); i++) if (i >= 0 && j >= 0 && i < W && j < H && Math.hypot(x0 + i * h - x, y0 + j * h - y) <= rad) out.push(j * W + i); return out; };
  const ex = mid[2 * m - 2], ey = mid[2 * m - 1];
  const starts = near(mid[0], mid[1], 1.5);
  if (r.startWet && r.wetAt) for (const c of near(mid[0], mid[1], 6)) if (r.wetAt(cx(c), cy(c)) && pillOf[c] < 0 && lab[c] <= r.kFirst) starts.push(c);
  const goal = new Uint8Array(NC);
  for (const c of near(ex, ey, 1.5)) goal[c] = 1;
  const reach = r.endWet && r.wetAt ? 10 : 1.5;
  if (r.endWet && r.wetAt) for (const c of near(ex, ey, 10)) if (r.wetAt(cx(c), cy(c)) && lab[c] >= r.kLast) goal[c] = 1;
  const bend = [0, o.B45, o.B90];
  const run = (walls: boolean) => {
    const dist = new Float64Array(NC * 8).fill(Infinity), from = new Int32Array(NC * 8).fill(-1), heap = new Heap();
    const hOf = (c: number) => Math.max(0, Math.hypot(cx(c) - ex, cy(c) - ey) - reach);
    const cellCost = (c: number) => { if (walls && wall[c]) return Infinity; let s = step[c]; if (fence[c]) s += o.fence / h; if (pillOf[c] >= 0) s += o.pillCost; return s; };
    for (const c of starts) if (cellCost(c) < Infinity) for (let k = 0; k < 8; k++) (dist[c * 8 + k] = 0), heap.push(hOf(c), c * 8 + k);
    let best = -1;
    while (heap.size) {
      const [f, st] = heap.pop(), c = st >> 3, dk = st & 7, dc = dist[st];
      if (f > dc + hOf(c) + 1e-9) continue;
      if (goal[c]) { best = st; break; }
      const i = c % W, j = (c - i) / W;
      for (let k = 0; k < 8; k++) {
        const tn = Math.min(Math.abs(k - dk), 8 - Math.abs(k - dk));
        if (tn >= 3) continue;
        const ni = i + DX[k], nj = j + DY[k];
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        const nc = nj * W + ni, sc = cellCost(nc);
        if (sc === Infinity) continue;
        const nd = dc + (k & 1 ? h * Math.SQRT2 : h) * sc + bend[tn], ns = nc * 8 + k;
        if (nd < dist[ns]) (dist[ns] = nd), (from[ns] = st), heap.push(nd + hOf(nc), ns);
      }
    }
    if (best < 0) return null;
    const cells: number[] = [];
    for (let st = best; st >= 0; st = from[st]) cells.push(st >> 3);
    return cells.reverse();
  };
  let cells = run(true), fallback = false;
  if (!cells) (cells = run(false)), (fallback = true);
  if (!cells) return null;
  // the light straightening (Anita, 11-20: "straighten/smoothen ever so slightly even when space is tight"): the path
  // fitted again with runs at 0, 45 and 90 degrees straying up to `straight`, a way kept only where the stretch it
  // sweeps over holds no station (so no pair of stations and no hop changes which side of the river it is on), it comes
  // no nearer any station's place or pill than half the band and the clearance (or than the routed path came to it),
  // and it crosses no more lines
  const px = cells.map(cx), py = cells.map(cy);
  /** how near polyline pts comes to station s's place, and to its pill's edge */
  const nearOf = (pts: number[], s: number): [number, number] => {
    let dc = Infinity, dp = Infinity;
    for (let q = 0; q + 3 < pts.length; q += 2) {
      dc = Math.min(dc, segD(r.SX[s], r.SY[s], pts[q], pts[q + 1], pts[q + 2], pts[q + 3]));
      for (const p of r.pills[s]) dp = Math.min(dp, segSegD(pts[q], pts[q + 1], pts[q + 2], pts[q + 3], p[0], p[1], p[2], p[3]) - p[4]);
    }
    return [dc, dp];
  };
  const hopBox = r.hops.map((hp) => { let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity; for (let q = 0; q < hp.pts.length; q += 2) (a = Math.min(a, hp.pts[q])), (c = Math.max(c, hp.pts[q])), (b = Math.min(b, hp.pts[q + 1])), (d = Math.max(d, hp.pts[q + 1])); return [a, b, c, d]; });
  const crossLines = (pts: number[], bx0: number, by0: number, bx1: number, by1: number) => {
    let c = 0;
    r.hops.forEach((hp, q) => { const B = hopBox[q]; if (B[2] >= bx0 && B[0] <= bx1 && B[3] >= by0 && B[1] <= by1) c += polyCross(pts, hp.pts); });
    return c;
  };
  const keeps = (i: number, j: number, bx: number, by: number) => {
    if (j === i + 1) return true;
    const way = bx === bx ? [px[i], py[i], bx, by, px[j], py[j]] : [px[i], py[i], px[j], py[j]];
    const orig: number[] = [];
    for (let q = i; q <= j; q++) orig.push(px[q], py[q]);
    let x0b = Infinity, y0b = Infinity, x1b = -Infinity, y1b = -Infinity;
    for (const v of [orig, way]) for (let q = 0; q < v.length; q += 2) (x0b = Math.min(x0b, v[q])), (x1b = Math.max(x1b, v[q])), (y0b = Math.min(y0b, v[q + 1])), (y1b = Math.max(y1b, v[q + 1]));
    // the swept stretch: the routed path from i to j and the way back
    const ring = orig.slice();
    for (let q = way.length - 4; q >= 2; q -= 2) ring.push(way[q], way[q + 1]);
    const need = wMax / 2 + o.clearPad, reachP = need + 3;
    for (const s of r.stations) {
      const x = r.SX[s], y = r.SY[s];
      if (x < x0b - reachP || x > x1b + reachP || y < y0b - reachP || y > y1b + reachP) continue;
      let inside = false;
      for (let q = 0, m2 = ring.length, k2 = m2 - 2; q < m2; k2 = q, q += 2) if (ring[q + 1] > y !== ring[k2 + 1] > y && x < ((ring[k2] - ring[q]) * (y - ring[q + 1])) / (ring[k2 + 1] - ring[q + 1]) + ring[q]) inside = !inside;
      if (inside) return false;
      const [wc0, wp0] = nearOf(way, s);
      if (wc0 >= need && wp0 >= need) continue;
      const [oc, op] = nearOf(orig, s);
      if (wc0 < Math.min(need, oc) - 1e-9 || wp0 < Math.min(need, op) - 1e-9) return false;
    }
    return crossLines(way, x0b, y0b, x1b, y1b) <= crossLines(orig, x0b, y0b, x1b, y1b);
  };
  let fit: number[];
  if (o.straight > 0) fit = Array.from(fitOcti(px, py, Array<number>(px.length).fill(o.straight), keeps));
  else {
    fit = [];
    for (let q = 0; q < px.length; q++) {
      const L = fit.length;
      if (L >= 4 && Math.abs((fit[L - 2] - fit[L - 4]) * (py[q] - fit[L - 1]) - (fit[L - 1] - fit[L - 3]) * (px[q] - fit[L - 2])) < 1e-9) (fit[L - 2] = px[q]), (fit[L - 1] = py[q]);
      else fit.push(px[q], py[q]);
    }
  }
  // widths along it: each cell's nearest middle sample's, never narrowing toward the end routed to; a vertex put on the
  // fitted path where the width steps (nearest the cell where it does)
  let wNow = wc[cells[0]];
  const steps: [number, number, number][] = [];
  for (let q = 1; q < cells.length; q++) if (wc[cells[q]] > wNow + 1e-6) (wNow = wc[cells[q]]), steps.push([px[q], py[q], wNow]);
  const path: number[] = [fit[0], fit[1]], segW: number[] = [];
  let w = wc[cells[0]], si = 0;
  for (let q = 0; q + 3 < fit.length; q += 2) {
    const ax = fit[q], ay = fit[q + 1], bx = fit[q + 2], by = fit[q + 3], L = Math.hypot(bx - ax, by - ay);
    // the steps falling on this run (their nearest point on it, in order), until one lies nearer a later run
    for (; si < steps.length; si++) {
      const [sx, sy, sw] = steps[si];
      const dq = segD(sx, sy, ax, ay, bx, by);
      let later = false;
      for (let k = q + 2; k + 3 < fit.length && k <= q + 8; k += 2) if (segD(sx, sy, fit[k], fit[k + 1], fit[k + 2], fit[k + 3]) < dq - 1e-9) later = true;
      if (later) break;
      const tt = L > 1e-9 ? Math.max(0, Math.min(1, ((sx - ax) * (bx - ax) + (sy - ay) * (by - ay)) / (L * L))) : 0, tx = ax + tt * (bx - ax), ty = ay + tt * (by - ay);
      if (Math.hypot(tx - path[path.length - 2], ty - path[path.length - 1]) > 1e-6 && Math.hypot(tx - bx, ty - by) > 1e-6) path.push(tx, ty), segW.push(w);
      w = sw;
    }
    path.push(bx, by), segW.push(w);
  }
  return { path, segW, fallback, ms: performance.now() - t0 };
}

/** a polyline's corners rounded (radius up to R, at most half of either run), as points */
export function roundPath(p: ArrayLike<number>, R: number): number[] {
  const m = p.length / 2;
  if (m < 3) return Array.from(p);
  const out = [p[0], p[1]];
  for (let k = 1; k + 1 < m; k++) {
    const ax = p[2 * k - 2], ay = p[2 * k - 1], vx = p[2 * k], vy = p[2 * k + 1], bx = p[2 * k + 2], by = p[2 * k + 3];
    const li = Math.hypot(vx - ax, vy - ay), lo = Math.hypot(bx - vx, by - vy), r = Math.min(R, li / 2, lo / 2);
    if (r <= 1e-6) { out.push(vx, vy); continue; }
    const sx = vx - ((vx - ax) / li) * r, sy = vy - ((vy - ay) / li) * r, ex = vx + ((bx - vx) / lo) * r, ey = vy + ((by - vy) / lo) * r;
    for (let s = 0; s <= 6; s++) { const u = s / 6; out.push((1 - u) ** 2 * sx + 2 * u * (1 - u) * vx + u * u * ex, (1 - u) ** 2 * sy + 2 * u * (1 - u) * vy + u * u * ey); }
  }
  out.push(p[2 * m - 2], p[2 * m - 1]);
  return out;
}

/** one routed river, for checks: its course (an index of ), path (pitches, upstream first, corners square), width, pairs of stations
 * near it (each with whether the river parts them on the ground) and how many are drawn on the wrong banks, the
 * stations under its band, its bends */
export interface RoutedRiver { course: number; arcs: number[]; path: number[]; w: number; wMax: number; pairs: number[]; wrongPairs: number; wrong: number[]; under: number[]; bends: number; ends: [string, string]; hopWrong: number; hopMissed: number; fallback: boolean; ms: number }
export interface RiverRoutes {
  cmds: Map<number, number[]>;
  ends: Map<number, number[]>;
  /** each band as drawn (pitches, corners square) with its width and the arcs of its course (checks) */
  fits: { fit: Float64Array; w: number; arc: number; arcs: number[] }[];
  routed: RoutedRiver[];
  /** the crossed courses not drawn, and why: wider than a band (the grid's area water), no way found, or under 2
   * pitches long */
  skipped: { arcs: number[]; why: string; own: number }[];
  /** stations some routed river leaves on the wrong bank */
  wrong: number[];
  arcs: number;
  crossings: number;
  ms: number;
}

/** Every river course some hop crosses, routed over the drawing: `g` the graph with real places, `lg` the laid out one,
 * `map` the carried map, `G` the drawn grid (whose own narrow water along the routed rivers goes to land). */
export function routeRivers(G: MapGrid, g: DiagramGraph, lg: DiagramGraph, layout: DiagramLayout, k: number, real: RealWater, map: CarryMap, o: RouteOptions = ROUTE): RiverRoutes {
  const t0 = performance.now(), n = g.key.length, { w: GW, h: GH, cell } = G, NC = GW * GH;
  const out: RiverRoutes = { cmds: new Map(), ends: new Map(), fits: [], routed: [], skipped: [], wrong: [], arcs: 0, crossings: 0, ms: 0 };
  const all = riverCourses(real), courses = all.filter((C) => C.cross.length && C.pts.length >= 4);
  if (!courses.length) return out;
  const SX = Float64Array.from(layout.x, (v) => v / k), SY = Float64Array.from(layout.y, (v) => v / k);
  // pills (pitches), each station's
  const d = drawing(lg, layout), pills: Pill[][] = Array.from({ length: n }, () => []);
  d.pills.forEach((p, q) => { const v = d.pillOf[q], [rr, ol] = pillSize(d, v); pills[v].push([p[0] / k, p[1] / k, p[2] / k, p[3] / k, (rr + ol + (p[4] ?? 0)) / k]); });
  for (let i = 0; i < n; i++) if (!pills[i].length) pills[i].push([SX[i], SY[i], SX[i], SY[i], layout.hw[i] / k]);
  // each hop as drawn, pitches, from its a end
  const hopPoly = new Map<number, { pts: number[]; k: number }>();
  for (const pc of layout.pieces) for (let q = 0; q + 1 < pc.nodes.length; q++) {
    const a = pc.nodes[q], b = pc.nodes[q + 1], pts: number[] = [];
    for (let v = pc.at[q]; v <= pc.at[q + 1]; v++) pts.push(pc.verts[v * 2] / k, pc.verts[v * 2 + 1] / k);
    if (a > b) { const r: number[] = []; for (let i = pts.length - 2; i >= 0; i -= 2) r.push(pts[i], pts[i + 1]); hopPoly.set(b * n + a, { pts: r, k: pc.strokes.length }); }
    else hopPoly.set(a * n + b, { pts, k: pc.strokes.length });
  }
  const hopPts = (e: number): ArrayLike<number> => (g.hop && g.hopAt && g.hopAt[e + 1] - g.hopAt[e] >= 2 ? g.hop.subarray(g.hopAt[e] * 2, g.hopAt[e + 1] * 2) : [g.x[g.ea[e]], g.y[g.ea[e]], g.x[g.eb[e]], g.y[g.eb[e]]]);
  const hopBox = Array.from({ length: g.ea.length }, (_, e) => { const p = hopPts(e); let a = Infinity, b = Infinity, c = -Infinity, dd = -Infinity; for (let q = 0; q < p.length; q += 2) (a = Math.min(a, p[q])), (c = Math.max(c, p[q])), (b = Math.min(b, p[q + 1])), (dd = Math.max(dd, p[q + 1])); return [a, b, c, dd]; });
  // the grid's wide water (a disc WMAX across fits it, and what lies within its reach): the sea, and rivers drawn as area
  const M = new Uint8Array(NC);
  for (let c = 0; c < NC; c++) M[c] = +(G.D[c] === 0 || G.D[c] === G.OUT);
  const dIn = chamfer(M, GW, GH, false, 60000), r3 = Math.floor((3 * WMAX) / 2 / cell), core = new Uint8Array(NC);
  for (let c = 0; c < NC; c++) core[c] = +(M[c] === 1 && dIn[c] > r3);
  const dCore = chamfer(core, GW, GH, true, 60000), bandAt = new Uint8Array(NC);
  const gcell = (x: number, y: number) => { const i = Math.floor((x - G.x0) / cell), j = Math.floor((y - G.y0) / cell); return i < 0 || j < 0 || i >= GW || j >= GH ? -1 : j * GW + i; };
  const wetAt = (x: number, y: number) => { const c = gcell(x, y); return c >= 0 && ((M[c] === 1 && dCore[c] <= r3) || bandAt[c] === 1); };
  const stamp = (P: ArrayLike<number>, rad: number, fn: (c: number) => void) => {
    for (let q = 0; q + 1 < P.length; q += 2) {
      const ax = P[q], ay = P[q + 1], bx = q + 3 < P.length ? P[q + 2] : ax, by = q + 3 < P.length ? P[q + 3] : ay;
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - rad - G.x0) / cell)), i1 = Math.min(GW - 1, Math.floor((Math.max(ax, bx) + rad - G.x0) / cell));
      const j0 = Math.max(0, Math.floor((Math.min(ay, by) - rad - G.y0) / cell)), j1 = Math.min(GH - 1, Math.floor((Math.max(ay, by) + rad - G.y0) / cell));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (segD(G.x0 + (i + 0.5) * cell, G.y0 + (j + 0.5) * cell, ax, ay, bx, by) <= rad) fn(j * GW + i);
    }
  };
  const nodeDeg = new Map<number, number>();
  for (const A of real.arcs) for (const v of [A.a, A.b]) nodeDeg.set(v, (nodeDeg.get(v) ?? 0) + 1);
  /** an end: the sea, another river (a junction), open (runs on out of what is looked at) or free (ends on land) */
  const endKind = (v: number) => (real.nodes[v]?.sea ? "sea" : (nodeDeg.get(v) ?? 0) >= 2 ? "join" : real.nodes[v]?.open ? "open" : "free"); // i18n-ignore: end kinds
  const erase = new Uint8Array(NC), wrongSet = new Set<number>(), upArcs = new Set<number>();
  for (const C of courses) out.crossings += C.cross.length;
  courses.forEach((C) => {
    const t1 = performance.now(), cum = cumOf(C.pts), len = cum[cum.length - 1];
    const sMin = C.cross[0].s, sMax = C.cross[C.cross.length - 1].s;
    const kS = endKind(C.startNode), kE = endKind(C.endNode);
    let sA = Math.max(0, sMin - EXT_M), sB = Math.min(len, sMax + EXT_M);
    const wetS = kS === "sea" || kS === "join", wetE = kE === "sea" || kE === "join";
    if (wetS && sMin < END_M) sA = 0;
    if (wetE && len - sMax < END_M) sB = len;
    const toS = sA === 0 && wetS, toE = sB === len && wetE;
    // routed from upstream: a sea end last
    const down = (kS === "sea" && kE !== "sea") || (toS && !toE) ? -1 : 1;
    const GP: number[] = [], SS: number[] = [];
    for (let s = sA; ; s = Math.min(sB, s + STEP_M)) { const [x, y] = at(C.pts, cum, s); GP.push(x, y), SS.push(s); if (s >= sB) break; }
    if (down < 0) {
      const R: number[] = [];
      for (let q = GP.length - 2; q >= 0; q -= 2) R.push(GP[q], GP[q + 1]);
      GP.splice(0, GP.length, ...R), SS.reverse();
    }
    // its middle carried into the drawing, and moved so each crossing lies on its hop as drawn at the hop's share (the
    // river within CORR_M of a crossing, along it, moving with it): the carried map puts a narrow river only roughly
    const mid: number[] = [], fix: number[] = [], carried: number[] = [];
    for (const c of C.cross) {
      if (c.s < sA || c.s > sB) continue;
      const a = g.ea[c.e], b = g.eb[c.e], hp = hopPoly.get(Math.min(a, b) * n + Math.max(a, b));
      if (!hp) continue;
      const [tx, ty] = at(hp.pts, cumOf(hp.pts), (a < b ? c.t : 1 - c.t) * lengthOf(hp.pts)), [rx, ry] = at(C.pts, cum, c.s), [px, py] = map.at(rx, ry);
      fix.push(c.s, tx - px, ty - py);
    }
    for (let q = 0; q < GP.length; q += 2) {
      const [x, y] = map.at(GP[q], GP[q + 1]), s = SS[q >> 1];
      carried.push(x, y);
      let sx = 0, sy = 0, sw = 1 / (CORR_M * CORR_M);
      for (let f = 0; f < fix.length; f += 3) {
        const d = Math.abs(fix[f] - s);
        if (d >= CORR_M) continue;
        const wt = 1 / (d * d + 1) - 1 / (CORR_M * CORR_M);
        (sx += wt * fix[f + 1]), (sy += wt * fix[f + 2]), (sw += wt);
      }
      mid.push(x + sx / sw, y + sy / sw);
    }
    // the drawing's scale (pitches a metre) at its crossings
    const scale = median(C.cross.filter((c) => c.s >= sA && c.s <= sB).map((c) => { const [x, y] = at(C.pts, cum, c.s), [x2, y2] = at(C.pts, cum, c.s + 50), [u, v] = map.at(x, y), [u2, v2] = map.at(x2, y2); return Math.hypot(u2 - u, v2 - v) / Math.max(1, Math.hypot(x2 - x, y2 - y)); }));
    // one width, from the stretch's typical half width (its arcs' length-weighted median)
    const wl: [number, number][] = [];
    for (const a of C.arcs) if (a.s1 > sA && a.s0 < sB) wl.push([a.half, Math.min(a.s1, sB) - Math.max(a.s0, sA)]);
    wl.sort((p, q) => p[0] - q[0]);
    const tot = wl.reduce((t, x) => t + x[1], 0);
    let acc = 0, h0 = wl[0]?.[0] ?? C.half;
    for (const [hh, l] of wl) { acc += l; if (acc >= tot / 2) { h0 = hh; break; } }
    const own = 2 * h0 * scale;
    if (!(own <= WMAX)) return void out.skipped.push({ arcs: C.arcs.map((a) => a.arc), why: "wide", own }); // i18n-ignore: a reason for checks
    const W0 = Math.max(WMIN, own);
    // toward the sea it steps up by powers of 2 where an arc 500 m or longer is at least twice as wide, never back down
    const wSample = new Float64Array(SS.length).fill(W0);
    if ((down < 0 ? kS : kE) === "sea" && (down < 0 ? toS : toE)) {
      let runH = h0;
      for (let q = 0; q < SS.length; q++) {
        const a = C.arcs.find((x) => SS[q] >= x.s0 && SS[q] <= x.s1);
        if (a && a.l >= 500) runH = Math.max(runH, a.half);
        const ratio = runH / h0;
        wSample[q] = ratio >= 2 ? Math.min(WWIDE, W0 * 2 ** Math.floor(Math.log2(ratio))) : W0;
      }
    }
    // the stations near the stretch (projecting inside it, not within 200 m of its ends), and further out
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (let q = 0; q < GP.length; q += 2) (bx0 = Math.min(bx0, GP[q])), (bx1 = Math.max(bx1, GP[q])), (by0 = Math.min(by0, GP[q + 1])), (by1 = Math.max(by1, GP[q + 1]));
    const sLo = sA + 200, sHi = sB - 200, nearSt: number[] = [], far: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = g.x[i], y = g.y[i];
      if (x < bx0 - FAR_M || x > bx1 + FAR_M || y < by0 - FAR_M || y > by1 + FAR_M) continue;
      const r = nearestOn(C.pts, x, y);
      if (r.d <= NEAR_M && r.d >= 30 && r.inside && r.side && r.s >= sLo && r.s <= sHi) nearSt.push(i);
      else if (nearestOn(GP, x, y).d <= FAR_M) far.push(i);
    }
    const pairs: number[] = [], fences: number[][] = [];
    for (let a = 0; a < nearSt.length; a++) for (let b = a + 1; b < nearSt.length; b++) {
      const i = nearSt[a], j = nearSt[b], dd = Math.hypot(g.x[i] - g.x[j], g.y[i] - g.y[j]);
      if (dd > PAIR_M) continue;
      const kc = crossCount(g.x[i], g.y[i], g.x[j], g.y[j], GP);
      pairs.push(i, j, kc % 2);
      if (kc === 0 && dd <= FENCE_M) fences.push([SX[i], SY[i], SX[j], SY[j]]);
    }
    for (const i of nearSt) for (const j of far) {
      if (Math.hypot(g.x[i] - g.x[j], g.y[i] - g.y[j]) <= FENCE_M && crossCount(g.x[i], g.y[i], g.x[j], g.y[j], GP) === 0) fences.push([SX[i], SY[i], SX[j], SY[j]]);
    }
    // the hops and stations in the routing's box (drawn), each hop the river's own where its track crosses the stretch
    let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity;
    for (let q = 0; q < mid.length; q += 2) (mx0 = Math.min(mx0, mid[q])), (mx1 = Math.max(mx1, mid[q])), (my0 = Math.min(my0, mid[q + 1])), (my1 = Math.max(my1, mid[q + 1]));
    const MB = o.margin + 2, inBox = (x: number, y: number) => x >= mx0 - MB && x <= mx1 + MB && y >= my0 - MB && y <= my1 + MB;
    const hops: RouteIn["hops"] = [], ownCross: number[] = [];
    for (let e = 0; e < g.ea.length; e++) {
      const a = g.ea[e], b = g.eb[e];
      if (!inBox(SX[a], SY[a]) && !inBox(SX[b], SY[b])) continue;
      const hp = hopPoly.get(Math.min(a, b) * n + Math.max(a, b));
      if (!hp) continue;
      const B = hopBox[e], gc = B[2] < bx0 || B[0] > bx1 || B[3] < by0 || B[1] > by1 ? 0 : polyCross(hopPts(e), GP);
      hops.push({ pts: hp.pts, k: hp.k, own: gc > 0 }), ownCross.push(gc);
    }
    const stations: number[] = [];
    for (let i = 0; i < n; i++) if (inBox(SX[i], SY[i])) stations.push(i);
    // the samples at the first and last crossings, in the order routed
    const ks = C.cross.filter((c) => c.s >= sA && c.s <= sB).map((c) => { const q = Math.min(SS.length - 1, Math.round((c.s - sA) / STEP_M)); return down < 0 ? SS.length - 1 - q : q; });
    let kFirst = Math.min(...ks), kLast = Math.max(...ks), shift = 0;
    const startWet = down < 0 ? toE : toS, endWet = down < 0 ? toS : toE;
    // an end on land runs on at least FREE_P pitches past the crossing nearest it, straight on where the river stops
    // sooner, so the band passes under that line
    const runOn = (atEnd: boolean) => {
      const m = mid.length / 2, kc = atEnd ? kLast : kFirst;
      let d = 0;
      for (let q = kc; atEnd ? q + 1 < m : q > 0; q += atEnd ? 1 : -1) d += Math.hypot(mid[2 * (q + (atEnd ? 1 : -1))] - mid[2 * q], mid[2 * (q + (atEnd ? 1 : -1)) + 1] - mid[2 * q + 1]);
      if (d >= FREE_P) return;
      const e = atEnd ? m - 1 : 0, f = atEnd ? Math.max(0, Math.min(kc, m - 2)) : Math.min(m - 1, Math.max(kc, 1));
      let ux = mid[2 * e] - mid[2 * f], uy = mid[2 * e + 1] - mid[2 * f + 1];
      const l = Math.hypot(ux, uy);
      if (l < 1e-6) return;
      (ux /= l), (uy /= l);
      const add: number[] = [];
      for (let s = 0.5; s <= FREE_P - d + 1e-9; s += 0.5) add.push(mid[2 * e] + ux * s, mid[2 * e + 1] + uy * s);
      if (atEnd) { mid.push(...add); for (let q = 0; q < add.length / 2; q++) wSampleX.push(wSample[wSample.length - 1]); }
      else {
        const pre: number[] = [];
        for (let q = add.length - 2; q >= 0; q -= 2) pre.push(add[q], add[q + 1]);
        mid.unshift(...pre);
        for (let q = 0; q < add.length / 2; q++) wSampleX.unshift(wSample[0]);
        (kFirst += add.length / 2), (kLast += add.length / 2), (shift += add.length / 2);
      }
    };
    const wSampleX = Array.from(wSample);
    if (!endWet) runOn(true);
    if (!startWet) runOn(false);
    const r = routeRiver({ mid, wSample: Float64Array.from(wSampleX), pills, stations, SX, SY, hops, fences, wetAt, startWet, endWet, kFirst, kLast }, o);
    // a band under 2 pitches long (its crossing already in wider water) is not drawn
    if (!r || lengthOf(r.path) < 2) return void out.skipped.push({ arcs: C.arcs.map((a) => a.arc), why: r ? "short" : "noroute", own }); // i18n-ignore: a reason for checks
    // its arcs past a stretch's end on land (not on toward the sea or another river)
    for (const a of C.arcs) if ((a.s1 <= sA && !toS) || (a.s0 >= sB && !toE)) upArcs.add(a.arc);
    const P = r.path, PR = roundPath(P, BEND_R), segW = r.segW;
    // the pairs drawn on the wrong banks, against the rounded band
    let wrongPairs = 0;
    const wrong: number[] = [];
    for (let q = 0; q < pairs.length; q += 3) {
      const i = pairs[q], j = pairs[q + 1];
      if (crossCount(SX[i], SY[i], SX[j], SY[j], PR) % 2 !== pairs[q + 2]) (wrongPairs++, wrong.push(i, j), wrongSet.add(i), wrongSet.add(j));
    }
    const under: number[] = [];
    for (const i of stations) {
      let hit = false;
      for (const p of pills[i]) for (let q = 0; q + 3 < P.length && !hit; q += 2) if (segSegD(p[0], p[1], p[2], p[3], P[q], P[q + 1], P[q + 2], P[q + 3]) < p[4] + segW[q >> 1] / 2 - 0.05) hit = true;
      if (hit) under.push(i);
    }
    // the hops drawn crossing it an odd number of times where their track does not on the ground, or the other way
    let hopWrong = 0, hopMissed = 0;
    hops.forEach((hp, q) => { const dc = polyCross(hp.pts, PR); if (dc % 2 !== ownCross[q] % 2) dc ? hopWrong++ : hopMissed++; });
    const wMax = Math.max(...segW);
    out.routed.push({ course: all.indexOf(C), arcs: C.arcs.map((a) => a.arc), path: P, w: W0, wMax, pairs, wrongPairs, wrong, under, bends: bendsOf(P), ends: down < 0 ? [kE, kS] : [kS, kE], hopWrong, hopMissed, fallback: r.fallback, ms: performance.now() - t1 });
    // drawn: runs of one width, round at the band's two ends and where the width steps (the wider run's end, T-3833)
    const stubs: number[][] = [[P[0], P[1], P[2], P[3], segW[0]], [P[P.length - 2], P[P.length - 1], P[P.length - 4], P[P.length - 3], segW[segW.length - 1]]];
    for (let q0 = 0, q = 1; q <= segW.length; q++) {
      if (q < segW.length && segW[q] === segW[q0]) continue;
      const w = segW[q0], seg = P.slice(2 * q0, 2 * q + 2), cmd = out.cmds.get(w) ?? out.cmds.set(w, []).get(w)!;
      // the inside of a bend kept round too: its middle's radius at least half the band and BEND_IN more
      filleted(Float64Array.from(seg, (v) => v * k), false, Math.max(BEND_R, w / 2 + BEND_IN) * k, cmd);
      out.fits.push({ fit: Float64Array.from(seg), w, arc: C.arcs[0].arc, arcs: C.arcs.map((a) => a.arc) });
      if (q0 > 0 && segW[q0 - 1] < w) stubs.push([P[2 * q0], P[2 * q0 + 1], P[2 * q0 + 2], P[2 * q0 + 3], w]);
      if (q < segW.length && segW[q] < w) stubs.push([P[2 * q], P[2 * q + 1], P[2 * q - 2], P[2 * q - 1], w]);
      q0 = q;
    }
    for (const [ex, ey, qx, qy, w] of stubs) {
      const l = Math.hypot(qx - ex, qy - ey) || 1, e = out.ends.get(w) ?? out.ends.set(w, []).get(w)!;
      e.push(0, ex * k, ey * k, 1, (ex + ((qx - ex) / l) * 0.01) * k, (ey + ((qy - ey) / l) * 0.01) * k);
    }
    // its band is wider water for the courses routed after it (a tributary meets it), and the grid's own water within
    // its reach and along the stretch's carried middle goes to land where narrower than a band can be
    // (only as far as the band goes: a wet end reached early leaves the water past it, an estuary, as it is)
    const nearestK = (x: number, y: number) => { let b = 0, bd = Infinity; for (let q = 0; q < mid.length; q += 2) { const dd = Math.hypot(mid[q] - x, mid[q + 1] - y); if (dd < bd) (bd = dd), (b = q >> 1); } return Math.min(carried.length / 2 - 1, Math.max(0, b - shift)); };
    const kA = startWet ? nearestK(P[0], P[1]) : 0, kB = endWet ? nearestK(P[P.length - 2], P[P.length - 1]) : carried.length / 2 - 1;
    stamp(carried.slice(2 * Math.min(kA, kB), 2 * Math.max(kA, kB) + 2), 2.5 * h0 * scale + 1.5, (c) => (erase[c] = 1));
    for (let q = 0; q + 3 < P.length; q += 2) {
      const w = segW[q >> 1], seg = [P[q], P[q + 1], P[q + 2], P[q + 3]];
      stamp(seg, w / 2, (c) => (bandAt[c] = 1));
      stamp(seg, w / 2 + 1.5 * cell, (c) => (erase[c] = 1));
    }
    out.arcs += C.arcs.length;
  });
  // past a stretch's end on land, the grid's water along the rest of a routed course (its middle carried, out to 2.5 of
  // its real half width at the drawing's scale and a pitch and a half) goes to land where under THIN_W across, so no
  // strand of the river runs on from the band's end
  if (upArcs.size) {
    const r3t = Math.floor((3 * THIN_W) / 2 / cell), coreT = new Uint8Array(NC), thin = new Uint8Array(NC);
    for (let c = 0; c < NC; c++) coreT[c] = +(M[c] === 1 && dIn[c] > r3t);
    const dCoreT = chamfer(coreT, GW, GH, true, 60000);
    real.arcs.forEach((A, ai) => {
      if (!upArcs.has(ai)) return;
      const P: number[] = [];
      for (let q = 0; q + 1 < A.pts.length; q += 2) P.push(...map.at(A.pts[q], A.pts[q + 1]));
      let sc = 0, cnt = 0;
      for (let q = 0; q + 3 < A.pts.length; q += 2) { const l = Math.hypot(A.pts[q + 2] - A.pts[q], A.pts[q + 3] - A.pts[q + 1]); if (l > 1) (sc += Math.hypot(P[q + 2] - P[q], P[q + 3] - P[q + 1]) / l), cnt++; }
      stamp(P, 2.5 * A.half * (cnt ? sc / cnt : 0) + 1.5, (c) => (thin[c] = 1));
    });
    const fresh = new Uint8Array(NC);
    for (let c = 0; c < NC; c++) if (G.D[c] === 0 && ((erase[c] && dCore[c] > r3) || (thin[c] && dCoreT[c] > r3t))) (G.D[c] = G.NOZ), (fresh[c] = 1);
    fillZones(G, fresh);
  }
  // what the rejoining of the water (sheetThin.ts `rejoinWater`) takes as joined by a band, and leaves as land
  const R = (G.routed ??= new Uint8Array(NC));
  for (let c = 0; c < NC; c++) if (bandAt[c]) R[c] = 1;
  for (let c = 0; c < NC; c++) if (erase[c] && !bandAt[c]) R[c] = 2;
  out.wrong = [...wrongSet].sort((a, b) => a - b);
  out.ms = performance.now() - t0;
  return out;
}

const segSegD = (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) =>
  segX(a, b, c, d, e, f, g, h) ? 0 : Math.min(segD(a, b, e, f, g, h), segD(c, d, e, f, g, h), segD(e, f, a, b, c, d), segD(g, h, a, b, c, d));
