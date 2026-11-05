// The network diagram's water (T-2169; docs/ui-diagram.md "Water"), drawn on the finished drawing from the water on the
// ground (waterReal.ts); the rail drawing is not touched. Rivers are bands of one width: each hop whose track crosses a
// river is crossed once, square to its bundle, near the real share of the hop, from the side the river comes from;
// between those crossings the band is routed on a grid the way a line would be (runs at 0, 45 and 90 degrees, bends
// costing, never through a line's casing or a pill) near the river's own course carried into the drawing by the
// nearest stations' local maps (T-2488's border lines, notes/T-2488.md). Where no such way is found the band has a gap,
// so it never crosses a line the river does not. The sea is waterSea.ts'.

import type { DiagramGraph } from "./graph";
import { pillSize, type Drawing } from "./draw";
import { fitOcti, rounded, segD } from "./borders";
import { hyp } from "./fmath";
import type { RealWater, RiverArc, Crossing } from "./waterReal";
import { diagramSea, type SeaStats } from "./waterSea";

export interface DiagramWater {
  /** the rivers' centre lines as path commands (draw.ts `toPath`), stroked BAND pitches wide with round ends */
  rivers: Float64Array;
  /** the rivers' width, layout units */
  band: number;
  /** the sea's outline as path commands, filled even-odd */
  sea?: Float64Array;
  /** checks only: the sea's rings as fitted */
  seaRings?: Float64Array[];
  seaStats?: SeaStats;
  /** checks only: each river run as routed (x, y pairs, before its corners are rounded), its arc, and the crossings it
   * was routed through (graph edges) */
  runs?: { parts: Float64Array[]; arc: number; edges: number[]; marks: number[] }[];
  /** checks only: sections with no way clear of the lines, from x, y to x, y */
  failed?: number[];
  stats: WaterStats;
}
export interface WaterStats { arcs: number; crossings: number; leftOut: number; turned: number; sections: number; widened: number; noRoute: number; ms: number }

/** a river band's width, pitches */
export const BAND = 2;
/** a route's cost rises within this many pitches of a casing or pill beyond the band's half width */
const GAP = 1;
/** the routing grid's cell, pitches; how far a route may stray from its course; bends' costs (pitches of length) */
const CELL = 1, CORRIDOR = 12, B45 = 6, B90 = 14;
/** straying costs nothing within FREE_DEV pitches, then W_DEV at the corridor's edge (per unit length); nearness to a
 * casing W_PROX at touching */
const FREE_DEV = 8, W_DEV = 0.1, W_PROX = 3;
/** a crossing runs on until this many pitches past the outermost casing (and any other it would end on) */
const RUN_PAST = 0.5;
/** the shares of a hop a crossing may be moved to, from its real one, to end clear of other lines */
const SHARES = [0.2, 0.35, 0.5, 0.65, 0.8];
/** a course longer than this many pitches is routed in parts */
const MAX_GUIDE = 100;
/** a river's course sampled this often on the ground, metres */
const DENSIFY_M = 200;
/** a river is drawn on to the sea where it meets it within this, metres */
const SEA_REACH_M = 4000;
/** a river ending free (not where it meets another drawn river or the sea) runs on this many pitches past its last
 * crossing */
const TAIL = 3;
/** drawn parts of one river are joined where the way between them along it is under this, metres */
const JOIN_M = 15000;
/** corners rounded with this radius, pitches */
const RADIUS = 2.5;

const SX = [1, 1, 0, -1, -1, -1, 0, 1], SY = [0, 1, 1, 1, 0, -1, -1, -1];
const R2 = Math.SQRT1_2;
const UX = [1, R2, 0, -R2, -1, -R2, 0, R2], UY = [0, R2, 1, R2, 0, -R2, -1, -R2];
const dirOf = (dx: number, dy: number) => {
  let best = 0, bd = -Infinity;
  for (let k = 0; k < 8; k++) {
    const v = UX[k] * dx + UY[k] * dy;
    if (v > bd) (bd = v), (best = k);
  }
  return best;
};

/** The rivers of `real` drawn on `d`, the drawing of `g` laid out at `x`, `y`. */
export function diagramWater(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, d: Drawing, real: RealWater, detail = false): DiagramWater {
  const t0 = performance.now();
  const pitch = d.pitch, n = g.key.length;
  const stats: WaterStats = { arcs: 0, crossings: 0, leftOut: 0, turned: 0, sections: 0, widened: 0, noRoute: 0, ms: 0 };
  const failed: number[] = [];

  // ---- which arcs are drawn: those crossed and those joining them; then the sea's within reach
  const arcs = real.arcs, nn = real.nodes.length;
  const lenOf = (a: RiverArc) => {
    let s = 0;
    for (let k = 0; k + 3 < a.pts.length; k += 2) s += hyp(a.pts[k + 2] - a.pts[k], a.pts[k + 3] - a.pts[k + 1]);
    return s;
  };
  const alen = arcs.map(lenOf);
  const drawn = arcs.map((a) => a.cross.length > 0 && a.pts.length >= 4);
  const deg = new Int32Array(nn);
  const count = () => {
    deg.fill(0);
    arcs.forEach((a, q) => drawn[q] && (deg[a.a]++, deg[a.b]++));
  };
  {
    // the drawn parts of a river joined where the way between them on the river is under JOIN_M (a greedy Steiner tree:
    // from each part, the nearest other by Dijkstra over the arcs, joined, until none is within reach)
    const by: number[][] = Array.from({ length: nn }, () => []);
    arcs.forEach((a, q) => a.a !== a.b && (by[a.a].push(q), by[a.b].push(q)));
    const up = Int32Array.from({ length: nn }, (_, i) => i);
    const find = (v: number): number => (up[v] === v ? v : (up[v] = find(up[v])));
    arcs.forEach((a, q) => drawn[q] && (up[find(a.a)] = find(a.b)));
    const inTree = new Uint8Array(nn);
    arcs.forEach((a, q) => drawn[q] && ((inTree[a.a] = 1), (inTree[a.b] = 1)));
    const dist = new Float64Array(nn), via = new Int32Array(nn);
    for (let again = true; again; ) {
      again = false;
      for (let v0 = 0; v0 < nn; v0++) {
        if (!inTree[v0] || find(v0) !== v0) continue;
        // from every node of part v0
        dist.fill(Infinity);
        via.fill(-1);
        const heap = new Heap();
        for (let v = 0; v < nn; v++) if (inTree[v] && find(v) === v0) (dist[v] = 0), heap.push(0, v);
        let hit = -1;
        while (heap.size) {
          const [dv, v] = heap.pop();
          if (dv > dist[v]) continue;
          if (inTree[v] && find(v) !== v0) {
            hit = v;
            break;
          }
          for (const q of by[v]) {
            const w = arcs[q].a === v ? arcs[q].b : arcs[q].a, nd = dv + alen[q];
            if (nd < dist[w] && nd <= JOIN_M) (dist[w] = nd), (via[w] = q), heap.push(nd, w);
          }
        }
        if (hit < 0) continue;
        for (let v = hit; dist[v] > 0; ) {
          const q = via[v];
          drawn[q] = true;
          (inTree[arcs[q].a] = 1), (inTree[arcs[q].b] = 1);
          v = arcs[q].a === v ? arcs[q].b : arcs[q].a;
        }
        up[find(hit)] = v0;
        arcs.forEach((a, q) => drawn[q] && (up[find(a.a)] = find(a.b)));
        again = true;
      }
    }
  }
  count();
  arcs.forEach((a, q) => {
    if (drawn[q] || a.a === a.b) return;
    const ends = [a.a, a.b].filter((v) => deg[v] > 0);
    if (ends.length === 1 && real.nodes[ends[0] === a.a ? a.b : a.a].sea && alen[q] <= SEA_REACH_M) drawn[q] = true;
  });
  count();

  // ---- the warp, and the drawing's obstacles (casings and pills) by squares of OB
  const warp = makeWarp(g, x, y, 20 * pitch, 2, 2.5 * pitch);
  const half = (BAND / 2) * pitch, C = half + GAP * pitch;
  const segs: number[] = [];
  for (const r of d.ribbons) for (let k = 0; k + 3 < r.length; k += 2) segs.push(r[k], r[k + 1], r[k + 2], r[k + 3], pitch / 2);
  d.pills.forEach((p, q) => {
    const [r, o] = pillSize(d, d.pillOf[q]);
    segs.push(p[0], p[1], p[2], p[3], r + o + p[4]);
  });
  const OB = 8 * pitch, obs = new Map<number, number[]>();
  const okey = (i: number, j: number) => (i + 32768) * 65536 + (j + 32768);
  for (let s = 0; s < segs.length; s += 5) {
    const R = segs[s + 4] + C;
    const i0 = Math.floor((Math.min(segs[s], segs[s + 2]) - R) / OB), i1 = Math.floor((Math.max(segs[s], segs[s + 2]) + R) / OB);
    const j0 = Math.floor((Math.min(segs[s + 1], segs[s + 3]) - R) / OB), j1 = Math.floor((Math.max(segs[s + 1], segs[s + 3]) + R) / OB);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const k = okey(i, j), l = obs.get(k);
      if (l) l.push(s);
      else obs.set(k, [s]);
    }
  }
  /** how far the band's middle at (px, py) is past touching the nearest casing or pill, up to C - half (that when none
   * is nearer); below 0 the band overlaps one */
  const CL = C;
  const clearAt = (px: number, py: number) => {
    let best = CL;
    for (const s of obs.get(okey(Math.floor(px / OB), Math.floor(py / OB))) ?? []) best = Math.min(best, segD(px, py, segs[s], segs[s + 1], segs[s + 2], segs[s + 3]) - segs[s + 4]);
    return best;
  };
  /** a point moved off the lines: the nearest within 8 pitches with the full clearance, else the clearest */
  const free = (px: number, py: number): [number, number] => {
    if (clearAt(px, py) >= CL) return [px, py];
    let best: [number, number] = [px, py], bc = clearAt(px, py), bd = Infinity;
    const st = 0.5 * pitch;
    for (let i = -16; i <= 16; i++) for (let j = -16; j <= 16; j++) {
      const qx = px + i * st, qy = py + j * st, c = clearAt(qx, qy), dd = i * i + j * j;
      if (c >= CL ? bc < CL || dd < bd : bc < CL && c > bc) (best = [qx, qy]), (bc = c), (bd = dd);
    }
    return best;
  };

  // ---- per graph edge, its drawn stretches: per line on it, the ribbon's points between its two nodes, a to b
  const stretches = new Map<number, number[][]>();
  {
    const edgeOf = new Map<number, number>();
    for (let e = 0; e < g.ea.length; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
    d.ribbons.forEach((r, li) => {
      const chain = g.lines[d.lineOf[li]].chain, at = d.chainAt[li];
      let p0 = 0;
      for (let p = 1; p <= at.length; p++) {
        if (p < at.length && at[p] === at[p0]) continue;
        const ci = at[p0];
        if (ci + 1 < chain.length) {
          const u = chain[ci], w = chain[ci + 1], e = edgeOf.get(Math.min(u, w) * n + Math.max(u, w));
          if (e !== undefined) {
            const pts: number[] = [];
            for (let q = p0; q <= Math.min(p, at.length - 1); q++) pts.push(r[q * 2], r[q * 2 + 1]);
            if (u !== g.ea[e]) {
              const rev: number[] = [];
              for (let q = pts.length - 2; q >= 0; q -= 2) rev.push(pts[q], pts[q + 1]);
              pts.splice(0, pts.length, ...rev);
            }
            const l = stretches.get(e);
            if (l) l.push(pts);
            else stretches.set(e, [pts]);
          }
        }
        p0 = p;
      }
    });
  }

  /** a crossing drawn: a straight run across the hop's bundle, square to it, from the side the river comes from to
   * the other, its ends clear of the outermost casing */
  type Drawn = { ex: number; ey: number; xx: number; xy: number; dir: number };
  const drawnCross = (c: Crossing): Drawn | null => {
    const st = stretches.get(c.e);
    if (!st) return null;
    const P = st[0];
    let Lt = 0;
    const cum = [0];
    for (let k = 2; k < P.length; k += 2) cum.push((Lt += hyp(P[k] - P[k - 2], P[k + 1] - P[k - 1])));
    if (Lt <= 0) return null;
    const t0 = Math.max(0.25, Math.min(0.75, c.t));
    let first: Drawn | null = null;
    // at the real share of the hop, else the nearest place along it where the run ends clear of other lines
    for (const t of [t0, ...SHARES.filter((u) => Math.abs(u - t0) > 0.05).sort((u, v) => Math.abs(u - t0) - Math.abs(v - t0))]) {
      const want = t * Lt;
      let k = 1;
      while (k < cum.length - 1 && cum[k] < want) k++;
      const f = (want - cum[k - 1]) / Math.max(1e-12, cum[k] - cum[k - 1]);
      const px = P[k * 2 - 2] + (P[k * 2] - P[k * 2 - 2]) * f, py = P[k * 2 - 1] + (P[k * 2 + 1] - P[k * 2 - 1]) * f;
      const du = dirOf(P[k * 2] - P[k * 2 - 2], P[k * 2 + 1] - P[k * 2 - 1]), ux = UX[du], uy = UY[du], nx = -uy, ny = ux;
      let lo = 0, hi = 0;
      for (const Q of st) for (let q = 0; q < Q.length; q += 2) {
        const al = (Q[q] - px) * ux + (Q[q + 1] - py) * uy;
        if (Math.abs(al) > 3 * pitch) continue;
        const o = (Q[q] - px) * nx + (Q[q + 1] - py) * ny;
        (lo = Math.min(lo, o)), (hi = Math.max(hi, o));
      }
      // each end past the hop's own casings, and on past any other line or pill it would end on (a bundle beside),
      // which a place clear of them is looked for first
      let pushed = false;
      const past = (o: number, s: number) => {
        let r = o + pitch / 2 + RUN_PAST * pitch;
        for (let k = 0; k < 40 && clearAt(px + s * nx * r, py + s * ny * r) < RUN_PAST * pitch; k++) (r += 0.25 * pitch), (pushed = true);
        return r;
      };
      const rl = past(hi, 1), rr = past(-lo, -1);
      const [ex, ey, xx, xy] = c.sign > 0 ? [px - nx * rr, py - ny * rr, px + nx * rl, py + ny * rl] : [px + nx * rl, py + ny * rl, px - nx * rr, py - ny * rr];
      const dr = { ex, ey, xx, xy, dir: dirOf(xx - ex, xy - ey) };
      if (!pushed) return dr;
      first ??= dr;
    }
    return first;
  };

  // ---- the course of a stretch of river: its places on the ground laid out as a chain between its two ends as drawn
  const wv = new Float64Array(5);
  const W_ANCHOR = 0.02, W_SIDE = 1;
  const chainFit = (rr: number[], fx: number, fy: number, tx: number, ty: number): number[] => {
    const m = rr.length / 2;
    if (m <= 2) return [fx, fy, tx, ty];
    const vx = new Float64Array(m - 1), vy = new Float64Array(m - 1), axv = new Float64Array(m), ayv = new Float64Array(m);
    const M = new Float64Array(4), o = new Float64Array(5);
    for (let k = 0; k < m; k++) warp.at(rr[k * 2], rr[k * 2 + 1], o), (axv[k] = o[0]), (ayv[k] = o[1]);
    for (let k = 0; k + 1 < m; k++) {
      warp.mapAt((rr[k * 2] + rr[k * 2 + 2]) / 2, (rr[k * 2 + 1] + rr[k * 2 + 3]) / 2, M);
      const dx = rr[k * 2 + 2] - rr[k * 2], dy = rr[k * 2 + 3] - rr[k * 2 + 1];
      (vx[k] = M[0] * dx + M[1] * dy), (vy[k] = M[2] * dx + M[3] * dy);
    }
    const px = new Float64Array(m), py = new Float64Array(m), ws = new Float64Array(m), tsx = new Float64Array(m), tsy = new Float64Array(m);
    (px[0] = fx), (py[0] = fy), (px[m - 1] = tx), (py[m - 1] = ty);
    const cp = new Float64Array(m), dp = new Float64Array(m);
    const solve = (p: Float64Array, v: Float64Array, a: Float64Array, t: Float64Array) => {
      for (let k = 1; k <= m - 2; k++) {
        const b = 2 + W_ANCHOR + ws[k];
        let dd = v[k - 1] - v[k] + W_ANCHOR * a[k] + ws[k] * t[k];
        if (k === 1) dd += p[0];
        if (k === m - 2) dd += p[m - 1];
        const den = b + (k > 1 ? cp[k - 1] : 0);
        cp[k] = -1 / den;
        dp[k] = (dd + (k > 1 ? dp[k - 1] : 0)) / den;
      }
      for (let k = m - 2; k >= 1; k--) p[k] = dp[k] - (k < m - 2 ? cp[k] * p[k + 1] : 0);
    };
    for (let round = 0; round < 4; round++) {
      solve(px, vx, axv, tsx);
      solve(py, vy, ayv, tsy);
      let any = false;
      for (let k = 1; k < m - 1; k++) {
        (o[0] = px[k]), (o[1] = py[k]);
        warp.side(rr[k * 2], rr[k * 2 + 1], o);
        if (hyp(o[0] - px[k], o[1] - py[k]) > 1e-6 * pitch) (ws[k] = W_SIDE), (tsx[k] = o[0]), (tsy[k] = o[1]), (any = true);
      }
      if (!any) break;
    }
    const out: number[] = [];
    for (let k = 0; k < m; k++) out.push(px[k], py[k]);
    return out;
  };
  const warped = (px: number, py: number): [number, number] => (warp.at(px, py, wv), warp.side(px, py, wv), [wv[0], wv[1]]);

  // ---- routing
  const h = CELL * pitch, Rc = CORRIDOR * pitch;
  /** a route clear of the lines, or null where none is found (the band then has a gap: it never crosses a line the
   * river does not) */
  const routeOne = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[]): number[] | null => {
    stats.sections++;
    for (const k of [1, 3]) {
      const r = attempt(sx, sy, dS, ex, ey, dE, guide, k * Rc);
      if (r) return r;
      stats.widened++;
    }
    if (detail) failed.push(sx, sy, ex, ey);
    stats.noRoute++;
    return null;
  };
  const attempt = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[], M: number): number[] | null => {
    let gx0 = Infinity, gy0 = Infinity, gx1 = -Infinity, gy1 = -Infinity;
    for (let k = 0; k < guide.length; k += 2) (gx0 = Math.min(gx0, guide[k])), (gx1 = Math.max(gx1, guide[k])), (gy0 = Math.min(gy0, guide[k + 1])), (gy1 = Math.max(gy1, guide[k + 1]));
    const i0 = Math.floor((gx0 - M - sx) / h) - 1, i1 = Math.ceil((gx1 + M - sx) / h) + 1, j0 = Math.floor((gy0 - M - sy) / h) - 1, j1 = Math.ceil((gy1 + M - sy) / h) + 1;
    const W = i1 - i0 + 1, H = j1 - j0 + 1, NC = W * H;
    // distance to the course: its cells 0, then a chamfer transform (steps 1 and the root of 2, in cells)
    const gd = new Float32Array(NC).fill(1e9);
    for (let k = 0; k + 3 < guide.length; k += 2) {
      const ax = (guide[k] - sx) / h - i0, ay = (guide[k + 1] - sy) / h - j0, bx = (guide[k + 2] - sx) / h - i0, by = (guide[k + 3] - sy) / h - j0;
      const parts = Math.max(1, Math.ceil(2 * hyp(bx - ax, by - ay)));
      for (let q = 0; q <= parts; q++) {
        const ci = Math.round(ax + ((bx - ax) * q) / parts), cj = Math.round(ay + ((by - ay) * q) / parts);
        if (ci >= 0 && cj >= 0 && ci < W && cj < H) gd[cj * W + ci] = 0;
      }
    }
    for (let cj = 0; cj < H; cj++) for (let ci = 0; ci < W; ci++) {
      const c = cj * W + ci;
      let v = gd[c];
      if (ci > 0) v = Math.min(v, gd[c - 1] + 1);
      if (cj > 0) {
        v = Math.min(v, gd[c - W] + 1);
        if (ci > 0) v = Math.min(v, gd[c - W - 1] + Math.SQRT2);
        if (ci < W - 1) v = Math.min(v, gd[c - W + 1] + Math.SQRT2);
      }
      gd[c] = v;
    }
    for (let cj = H - 1; cj >= 0; cj--) for (let ci = W - 1; ci >= 0; ci--) {
      const c = cj * W + ci;
      let v = gd[c];
      if (ci < W - 1) v = Math.min(v, gd[c + 1] + 1);
      if (cj < H - 1) {
        v = Math.min(v, gd[c + W] + 1);
        if (ci < W - 1) v = Math.min(v, gd[c + W + 1] + Math.SQRT2);
        if (ci > 0) v = Math.min(v, gd[c + W - 1] + Math.SQRT2);
      }
      gd[c] = v;
    }
    const goalI = Math.round((ex - sx) / h) - i0, goalJ = Math.round((ey - sy) / h) - j0, goal = goalJ * W + goalI, start = -j0 * W - i0;
    const clr = new Float32Array(NC).fill(NaN);
    const step = new Float64Array(NC);
    const free0 = FREE_DEV * pitch;
    const cost = (c: number) => {
      if (clr[c] !== clr[c]) {
        const od = clearAt(sx + ((c % W) + i0) * h, sy + (Math.floor(c / W) + j0) * h);
        clr[c] = od;
        const dv = Math.max(0, gd[c] * h - free0) / (Rc - free0), pr = Math.max(0, 1 - od / CL);
        step[c] = od < 0 && c !== goal && c !== start ? Infinity : 1 + W_DEV * dv * dv + W_PROX * pr * pr;
      }
      return step[c];
    };
    const NS = NC * 8, dist = new Float64Array(NS).fill(Infinity), from = new Int32Array(NS).fill(-1);
    const heap = new Heap();
    const gxw = sx + (goalI + i0) * h, gyw = sy + (goalJ + j0) * h;
    const hOf = (c: number) => hyp(sx + ((c % W) + i0) * h - gxw, sy + (Math.floor(c / W) + j0) * h - gyw);
    const bendC = [0, B45 * pitch, B90 * pitch];
    for (let k = 0; k < 8; k++) {
      const tn = dS < 0 ? 0 : Math.min(Math.abs(k - dS), 8 - Math.abs(k - dS));
      if (tn < 2) (dist[start * 8 + k] = bendC[tn]), heap.push(hOf(start) + bendC[tn], start * 8 + k);
    }
    let best = Infinity, bestS = -1;
    while (heap.size) {
      const [f, st] = heap.pop();
      if (f >= best) break;
      const c = st >> 3, dk = st & 7, dc = dist[st];
      if (f - hOf(c) > dc + 1e-9) continue;
      if (c === goal) {
        const tn = dE < 0 ? 0 : Math.min(Math.abs(dk - dE), 8 - Math.abs(dk - dE));
        const tot = dc + (tn < 3 ? bendC[tn] : Infinity);
        if (tot < best) (best = tot), (bestS = st);
      }
      const ci = c % W, cj = (c - ci) / W;
      for (let k = 0; k < 8; k++) {
        const tn = Math.min(Math.abs(k - dk), 8 - Math.abs(k - dk));
        if (tn >= 3) continue;
        const ni = ci + SX[k], nj = cj + SY[k];
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        const nc = nj * W + ni, sc = cost(nc);
        if (sc === Infinity) continue;
        const nd = dc + (k & 1 ? h * Math.SQRT2 : h) * sc + bendC[tn], ns = nc * 8 + k;
        if (nd < dist[ns]) (dist[ns] = nd), (from[ns] = st), heap.push(nd + hOf(nc), ns);
      }
    }
    if (bestS < 0) return null;
    const cells: number[] = [];
    for (let st = bestS; st >= 0; st = from[st]) cells.push(st >> 3);
    cells.reverse();
    const px: number[] = [], py: number[] = [];
    let last = -1;
    for (const c of cells) {
      if (c === last) continue;
      last = c;
      px.push(sx + ((c % W) + i0) * h), py.push(sy + (Math.floor(c / W) + j0) * h);
    }
    (px[0] = sx), (py[0] = sy);
    if (px.length < 2) px.push(ex), py.push(ey);
    else (px[px.length - 1] = ex), (py[py.length - 1] = ey);
    return [...fitOcti(px, py, px.map(() => 0.6 * pitch))];
  };

  // ---- the ribbons' segments with their graph edges, by squares of RG: what a piece may not cross
  const RG = 4 * pitch, rib = new Map<number, number[]>(), ribE: number[] = [], ribP: number[] = [];
  {
    const edgeOf = new Map<number, number>();
    for (let e = 0; e < g.ea.length; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
    d.ribbons.forEach((r, li) => {
      const chain = g.lines[d.lineOf[li]].chain, at = d.chainAt[li];
      for (let k = 0; k + 3 < r.length; k += 2) {
        const ci = at[k / 2], u = chain[ci], w = chain[ci + 1], s = ribE.length;
        ribE.push(w === undefined ? -1 : edgeOf.get(Math.min(u, w) * n + Math.max(u, w)) ?? -1);
        ribP.push(r[k], r[k + 1], r[k + 2], r[k + 3]);
        for (let i = Math.floor(Math.min(r[k], r[k + 2]) / RG); i <= Math.floor(Math.max(r[k], r[k + 2]) / RG); i++) for (let j = Math.floor(Math.min(r[k + 1], r[k + 3]) / RG); j <= Math.floor(Math.max(r[k + 1], r[k + 3]) / RG); j++) {
          const l = rib.get(okey(i, j));
          if (l) l.push(s);
          else rib.set(okey(i, j), [s]);
        }
      }
    });
  }
  /** whether polyline p crosses a ribbon of an edge not in `allow` */
  const crossesOther = (p: number[], allow: Set<number>) => {
    for (let k = 0; k + 3 < p.length; k += 2) {
      const ax = p[k], ay = p[k + 1], ux = p[k + 2] - ax, uy = p[k + 3] - ay;
      for (let i = Math.floor(Math.min(ax, p[k + 2]) / RG); i <= Math.floor(Math.max(ax, p[k + 2]) / RG); i++) for (let j = Math.floor(Math.min(ay, p[k + 3]) / RG); j <= Math.floor(Math.max(ay, p[k + 3]) / RG); j++) for (const s of rib.get(okey(i, j)) ?? []) {
        if (allow.has(ribE[s])) continue;
        const cx = ribP[s * 4], cy = ribP[s * 4 + 1], vx = ribP[s * 4 + 2] - cx, vy = ribP[s * 4 + 3] - cy, den = ux * vy - uy * vx;
        if (Math.abs(den) < 1e-12) continue;
        const r = ((cx - ax) * vy - (cy - ay) * vx) / den, t = ((cx - ax) * uy - (cy - ay) * ux) / den;
        if (r >= 0 && r <= 1 && t >= 0 && t <= 1) return true;
      }
    }
    return false;
  };

  // ---- each drawn arc: from its start through each crossing to its end
  const nodeAt = new Map<number, [number, number]>();
  const nodePlace = (v: number) => {
    let p = nodeAt.get(v);
    if (!p) nodeAt.set(v, (p = free(...warped(real.nodes[v].x, real.nodes[v].y))));
    return p;
  };
  const cmds: number[] = [], runs: DiagramWater["runs"] = [];
  arcs.forEach((A, ai) => {
    if (!drawn[ai]) return;
    stats.arcs++;
    const P = A.pts, rs: number[] = [], ss: number[] = [];
    let s = 0;
    for (let k = 0; k + 3 < P.length; k += 2) {
      const sl = hyp(P[k + 2] - P[k], P[k + 3] - P[k + 1]), parts = Math.max(1, Math.ceil(sl / DENSIFY_M));
      for (let q = k ? 1 : 0; q <= parts; q++) {
        const f = q / parts;
        rs.push(P[k] + (P[k + 2] - P[k]) * f, P[k + 1] + (P[k + 3] - P[k + 1]) * f), ss.push(s + sl * f);
      }
      s += sl;
    }
    const total = s;
    const realAt = (at: number): [number, number] => {
      let q = 1;
      while (q < ss.length - 1 && ss[q] < at) q++;
      const f = Math.max(0, Math.min(1, (at - ss[q - 1]) / Math.max(1e-9, ss[q] - ss[q - 1])));
      return [rs[q * 2 - 2] + (rs[q * 2] - rs[q * 2 - 2]) * f, rs[q * 2 - 1] + (rs[q * 2 + 1] - rs[q * 2 - 1]) * f];
    };
    const cr = A.cross.map((c) => [c, drawnCross(c)] as const).filter((q) => q[1]) as [Crossing, Drawn][];
    // ends: a node joining other drawn rivers or at the sea is drawn to; a free end only a little past the last crossing
    const freeEnd = (v: number) => deg[v] <= 1 && !real.nodes[v].sea;
    let s0 = 0, s1 = total;
    if (cr.length && freeEnd(A.a)) s0 = cr[0][0].s;
    if (cr.length && freeEnd(A.b)) s1 = cr[cr.length - 1][0].s;
    // a free end is a short straight run on from the crossing beside it
    const [sx0, sy0] = s0 > 0 ? [NaN, NaN] : nodePlace(A.a);
    const [ex1, ey1] = s1 < total ? [NaN, NaN] : nodePlace(A.b);
    // pieces in order: routes (e -1) and crossings (e their hop); null where no route was found
    const pieces: ({ p: number[]; e: number } | null)[] = [];
    let fromS = s0, fx = sx0, fy = sy0, fd = -1;
    /** the routes from the last place to (tx, ty), arriving the way td; whether the last arrived */
    const leg = (toS: number, tx: number, ty: number, td: number): { got: ({ p: number[]; e: number } | null)[]; ok: boolean } => {
      const got: ({ p: number[]; e: number } | null)[] = [];
      const rr: number[] = [...realAt(fromS)];
      for (let q = 0; q < ss.length; q++) if (ss[q] > fromS && ss[q] < toS) rr.push(rs[q * 2], rs[q * 2 + 1]);
      rr.push(...realAt(toS));
      const guide = simplify(chainFit(rr, fx, fy, tx, ty), 2 * pitch);
      let gl = 0;
      const cut = [0];
      for (let k = 2; k < guide.length; k += 2) {
        gl += hyp(guide[k] - guide[k - 2], guide[k + 1] - guide[k - 1]);
        if (gl > MAX_GUIDE * pitch && k < guide.length - 2) cut.push(k), (gl = 0);
      }
      cut.push(guide.length - 2);
      let ax = fx, ay = fy, ad = fd;
      for (let c = 1; c < cut.length; c++) {
        const lastPart = c === cut.length - 1;
        const [bx, by] = lastPart ? [tx, ty] : free(guide[cut[c]], guide[cut[c] + 1]);
        const sub = [ax, ay, ...guide.slice(cut[c - 1] + 2, cut[c]), bx, by];
        const r = routeOne(ax, ay, ad, bx, by, lastPart ? td : -1, sub);
        got.push(r && { p: r, e: -1 });
        (ax = bx), (ay = by), (ad = -1);
      }
      return { got, ok: !!got[got.length - 1] };
    };
    for (const [c, dc] of cr) {
      if (c.s < s0 || c.s > s1) continue;
      // crossed the way the river crosses on the ground; where the drawing leaves no way round to that side, the other
      // way (the drawing turned the place over), which crosses no other line either
      if (fx !== fx) {
        const ux = UX[dc.dir] * TAIL * pitch, uy = UY[dc.dir] * TAIL * pitch;
        pieces.push({ p: [dc.ex - ux, dc.ey - uy, dc.ex, dc.ey], e: -1 }, { p: [dc.ex, dc.ey, dc.xx, dc.xy], e: c.e });
        (fromS = c.s), (fx = dc.xx), (fy = dc.xy), (fd = dc.dir);
        continue;
      }
      let l = leg(c.s, dc.ex, dc.ey, dc.dir), run = [dc.ex, dc.ey, dc.xx, dc.xy], dir = dc.dir;
      if (!l.ok) {
        const back = leg(c.s, dc.xx, dc.xy, (dc.dir + 4) % 8);
        if (back.ok) (l = back), (run = [dc.xx, dc.xy, dc.ex, dc.ey]), (dir = (dc.dir + 4) % 8), stats.turned++;
      }
      pieces.push(...l.got, { p: run, e: c.e });
      (fromS = c.s), (fx = run[2]), (fy = run[3]), (fd = dir);
    }
    if (ex1 === ex1) pieces.push(...leg(s1, ex1, ey1, -1).got);
    else if (fd >= 0) pieces.push({ p: [fx, fy, fx + UX[fd] * TAIL * pitch, fy + UY[fd] * TAIL * pitch], e: -1 });
    // a piece crossing a line it should not is left out: a route crossing any, a crossing any but its own hop's (so a
    // band never crosses a line the river does not; a crossing left out is counted)
    const parts: { p: number[]; e: Set<number> }[] = [], edges: number[] = [];
    let out = { p: [] as number[], e: new Set<number>() };
    const close = () => {
      if (out.p.length >= 4) parts.push(out);
      out = { p: [], e: new Set() };
    };
    for (const pc of pieces) {
      if (!pc || crossesOther(pc.p, new Set([pc.e]))) {
        if (pc && pc.e >= 0) stats.leftOut++;
        close();
        continue;
      }
      if (pc.e >= 0) edges.push(pc.e), out.e.add(pc.e);
      for (let k = out.p.length ? 2 : 0; k < pc.p.length; k++) out.p.push(pc.p[k]);
    }
    close();
    const vs = parts.map((q) => Float64Array.from(q.p));
    vs.forEach((v, q) => {
      // rounded corners, unless rounding would cut across a line
      let c: number[] = [];
      rounded(v, false, RADIUS * pitch, c);
      if (crossesOther(flatten(c), parts[q].e)) rounded(v, false, 0, (c = []));
      for (const x of c) cmds.push(x);
    });
    stats.crossings += edges.length;
    if (detail) runs.push({ parts: vs, arc: ai, edges, marks: cr.flatMap(([, dc]) => [dc.ex, dc.ey, dc.xx, dc.xy]) });
  });
  stats.ms = performance.now() - t0;
  const sea = diagramSea(g, x, y, d, real, warp.J);
  return { rivers: Float64Array.from(cmds), band: BAND * pitch, sea: sea.cmds, seaRings: detail ? sea.rings : undefined, runs: detail ? runs : undefined, failed: detail ? failed : undefined, stats, seaStats: sea.stats };
}

/** path commands (draw.ts `toPath`) as a polyline, curves sampled */
function flatten(c: number[]): number[] {
  const p: number[] = [];
  let x = 0, y = 0;
  for (let i = 0; i < c.length; ) {
    if (c[i] === 2) {
      for (let s = 1; s <= 6; s++) {
        const u = s / 6, a = (1 - u) * (1 - u), m = 2 * (1 - u) * u, q = u * u;
        p.push(a * x + m * c[i + 1] + q * c[i + 3], a * y + m * c[i + 2] + q * c[i + 4]);
      }
      (x = c[i + 3]), (y = c[i + 4]), (i += 5);
    } else {
      (x = c[i + 1]), (y = c[i + 2]), (i += 3);
      p.push(x, y);
    }
  }
  return p;
}

/** a binary heap of (key, value), least key first */
class Heap {
  k: number[] = [];
  v: number[] = [];
  get size() { return this.k.length; }
  push(key: number, val: number) {
    const k = this.k, v = this.v;
    let i = k.length;
    k.push(key), v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      (k[i] = k[p]), (v[i] = v[p]), (i = p);
    }
    (k[i] = key), (v[i] = val);
  }
  pop(): [number, number] {
    const k = this.k, v = this.v, top: [number, number] = [k[0], v[0]], lk = k.pop()!, lv = v.pop()!;
    if (k.length) {
      let i = 0;
      for (;;) {
        const a = 2 * i + 1, b = a + 1;
        let c = i, ck = lk;
        if (a < k.length && k[a] < ck) (c = a), (ck = k[a]);
        if (b < k.length && k[b] < ck) c = b;
        if (c === i) break;
        (k[i] = k[c]), (v[i] = v[c]), (i = c);
      }
      (k[i] = lk), (v[i] = lv);
    }
    return top;
  }
}

/** a polyline (x, y pairs) with the points within `tol` of the way between their kept neighbours dropped, its ends kept */
function simplify(p: number[], tol: number): number[] {
  const m = p.length / 2;
  if (m <= 2) return p;
  const keep = new Uint8Array(m);
  (keep[0] = 1), (keep[m - 1] = 1);
  const stack = [0, m - 1];
  while (stack.length) {
    const b = stack.pop()!, a = stack.pop()!;
    let worst = -1, wi = -1;
    for (let i = a + 1; i < b; i++) {
      const dd = segD(p[i * 2], p[i * 2 + 1], p[a * 2], p[a * 2 + 1], p[b * 2], p[b * 2 + 1]);
      if (dd > worst) (worst = dd), (wi = i);
    }
    if (worst > tol) (keep[wi] = 1), stack.push(a, wi, wi, b);
  }
  const out: number[] = [];
  for (let i = 0; i < m; i++) if (keep[i]) out.push(p[i * 2], p[i * 2 + 1]);
  return out;
}

// ---- places on the ground carried into the drawing (T-2168's warp, as T-2488 used it): each place by its nearest
// stations' own local maps, blended, the offsets past `reach` growing only as their log

/** the nearest stations blended, a station's weight 1 / (d^2 + EPS^2)^2, metres; a blended map's scale at most MAG_MAX
 * times the stations' median */
const K = 8, EPS_M = 50, MAG_MAX = 2;

export function makeWarp(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, reach: number, sideHops: number, gap: number) {
  const J = localMaps(g, x, y), all = kdTree(g.x, g.y);
  const n = g.key.length, scales = new Float64Array(n);
  for (let i = 0; i < n; i++) scales[i] = Math.sqrt(Math.abs(J[i * 4] * J[i * 4 + 3] - J[i * 4 + 1] * J[i * 4 + 2]));
  const medScale = scales.sort()[n >> 1] || 1;
  const near = new Int32Array(K), nd = new Float64Array(K);
  // each station's reach for the side terms: sideHops of its mean hop on the ground
  const hopR = new Float64Array(n), cnt = new Float64Array(n);
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e], b = g.eb[e], l = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a]);
    (hopR[a] += l), (hopR[b] += l), cnt[a]++, cnt[b]++;
  }
  for (let i = 0; i < n; i++) hopR[i] = cnt[i] ? (sideHops * hopR[i]) / cnt[i] : 0;
  const SK = 16, sn = new Int32Array(SK), sd = new Float64Array(SK);
  return {
    /** `out` [x, y], the drawn place of (sx, sy) on the ground, moved to the side of each station near it on the
     * ground that it is on there (along the station's own map of their offset), by sweeps of projections */
    side(sx: number, sy: number, out: Float64Array) {
      if (!sideHops) return;
      const found = all.nearest(sx, sy, SK, sn, sd);
      for (let sweep = 0; sweep < 8; sweep++) {
        let moved = 0;
        for (let q = 0; q < found; q++) {
          const i = sn[q];
          if (sd[q] > hopR[i] * hopR[i]) continue;
          const ux = sx - g.x[i], uy = sy - g.y[i];
          const ox = J[i * 4] * ux + J[i * 4 + 1] * uy, oy = J[i * 4 + 2] * ux + J[i * 4 + 3] * uy, ol = Math.sqrt(ox * ox + oy * oy);
          if (ol < 1e-9) continue;
          const nx = ox / ol, ny = oy / ol, cur = (out[0] - x[i]) * nx + (out[1] - y[i]) * ny, want = Math.min(ol, gap);
          if (cur < want) (out[0] += (want - cur) * nx), (out[1] += (want - cur) * ny), (moved = 1);
        }
        if (!moved) break;
      }
    },
    /** the blend of the nearest stations' own maps at (sx, sy) on the ground: a, b, d, e of X = a x + b y, Y = d x + e y */
    mapAt(sx: number, sy: number, M: Float64Array) {
      const found = all.nearest(sx, sy, K, near, nd);
      let sw = 0;
      M.fill(0);
      for (let q = 0; q < found; q++) {
        const i = near[q], d2 = nd[q] + EPS_M * EPS_M, w = 1 / (d2 * d2);
        for (let c = 0; c < 4; c++) M[c] += w * J[i * 4 + c];
        sw += w;
      }
      if (sw > 0) for (let c = 0; c < 4; c++) M[c] /= sw;
      const sc = Math.sqrt(Math.abs(M[0] * M[3] - M[1] * M[2]));
      if (sc > MAG_MAX * medScale) for (let c = 0; c < 4; c++) M[c] *= (MAG_MAX * medScale) / sc;
    },
    at(sx: number, sy: number, out: Float64Array) {
      const found = all.nearest(sx, sy, K, near, nd);
      let sw = 0, ax = 0, ay = 0;
      for (let q = 0; q < found; q++) {
        const i = near[q], d2 = nd[q] + EPS_M * EPS_M, w = 1 / (d2 * d2), ux = sx - g.x[i], uy = sy - g.y[i];
        let ox = J[i * 4] * ux + J[i * 4 + 1] * uy, oy = J[i * 4 + 2] * ux + J[i * 4 + 3] * uy;
        const ol = Math.sqrt(ox * ox + oy * oy);
        if (ol > reach) {
          const f = (reach * (1 + Math.log(ol / reach))) / ol;
          (ox *= f), (oy *= f);
        }
        (ax += w * (x[i] + ox)), (ay += w * (y[i] + oy)), (sw += w);
      }
      out[0] = found ? ax / sw : sx;
      out[1] = found ? ay / sw : sy;
    },
    /** the stations nearest a place on the ground */
    nearest: (sx: number, sy: number, k: number, idx: Int32Array, d2: Float64Array) => all.nearest(sx, sy, k, idx, d2),
    J,
  };
}

/** per station, the linear map from offsets on the ground to offsets in the drawing that best fits its hops (least
 * squares, pulled toward its hops' mean scale) */
function localMaps(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>): Float64Array {
  const n = g.key.length, J = new Float64Array(n * 4);
  const nbr: number[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < g.ea.length; e++) nbr[g.ea[e]].push(g.eb[e]), nbr[g.eb[e]].push(g.ea[e]);
  for (let i = 0; i < n; i++) {
    let rxx = 0, rxy = 0, ryy = 0, dxx = 0, dxy = 0, dyx = 0, dyy = 0, sr = 0, sd = 0;
    for (const j of nbr[i]) {
      const rx = g.x[j] - g.x[i], ry = g.y[j] - g.y[i], qx = x[j] - x[i], qy = y[j] - y[i];
      (rxx += rx * rx), (rxy += rx * ry), (ryy += ry * ry);
      (dxx += qx * rx), (dxy += qx * ry), (dyx += qy * rx), (dyy += qy * ry);
      (sr += Math.sqrt(rx * rx + ry * ry)), (sd += Math.sqrt(qx * qx + qy * qy));
    }
    const s = sr > 0 ? sd / sr : 0.005, mu = nbr[i].length ? (rxx + ryy) / nbr[i].length : 1;
    const m00 = rxx + mu, m01 = rxy, m11 = ryy + mu, det = m00 * m11 - m01 * m01;
    const n00 = dxx + mu * s, n01 = dxy, n10 = dyx, n11 = dyy + mu * s;
    const i00 = m11 / det, i01 = -m01 / det, i11 = m00 / det;
    J[i * 4] = n00 * i00 + n01 * i01;
    J[i * 4 + 1] = n00 * i01 + n01 * i11;
    J[i * 4 + 2] = n10 * i00 + n11 * i01;
    J[i * 4 + 3] = n10 * i01 + n11 * i11;
  }
  return J;
}

function kdTree(xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const n = xs.length, ord = Int32Array.from({ length: n }, (_, i) => i);
  const select = (lo: number, hi: number, k: number, c: ArrayLike<number>) => {
    while (hi > lo) {
      const p = c[ord[(lo + hi) >> 1]];
      let i = lo, j = hi;
      while (i <= j) {
        while (c[ord[i]] < p) i++;
        while (c[ord[j]] > p) j--;
        if (i <= j) {
          const t = ord[i];
          (ord[i] = ord[j]), (ord[j] = t), i++, j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return;
    }
  };
  const build = (lo: number, hi: number, depth: number) => {
    if (hi - lo <= 1) return;
    const mid = (lo + hi) >> 1;
    select(lo, hi - 1, mid, depth & 1 ? ys : xs);
    build(lo, mid, depth + 1);
    build(mid + 1, hi, depth + 1);
  };
  build(0, n, 0);
  return {
    nearest(px: number, py: number, k: number, idx: Int32Array, d2: Float64Array): number {
      let found = 0;
      const visit = (lo: number, hi: number, depth: number) => {
        if (hi <= lo) return;
        const mid = (lo + hi) >> 1, i = ord[mid];
        const dx = xs[i] - px, dy = ys[i] - py, dd = dx * dx + dy * dy;
        if (found < k || dd < d2[k - 1]) {
          let s = found < k ? found++ : k - 1;
          while (s > 0 && d2[s - 1] > dd) (d2[s] = d2[s - 1]), (idx[s] = idx[s - 1]), s--;
          (d2[s] = dd), (idx[s] = i);
        }
        const off = depth & 1 ? -dy : -dx;
        if (off < 0) {
          visit(lo, mid, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(mid + 1, hi, depth + 1);
        } else {
          visit(mid + 1, hi, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(lo, mid, depth + 1);
        }
      };
      visit(0, n, 0);
      return found;
    },
  };
}
