// The network diagram's water (T-2169; docs/ui-diagram.md "Water"), drawn on the finished drawing from the water on the
// ground (waterReal.ts); the rail drawing is not touched. The sea is waterSea.ts', drawn first. A river a hop crosses
// is drawn whole (T-2516): along it between its crossings and on to where it meets the sea, as one band. Each hop whose
// track crosses it is crossed once, square to its bundle, near the real share of the hop, from the side the river
// comes from; between those crossings the band is routed on a grid the way a line would be (runs at 0, 45 and 90
// degrees, bends costing, clear of the lines' casings and the pills) near the river's own course carried into the
// drawing by the nearest stations' local maps (T-2488's border lines, notes/T-2488.md). Where the drawing leaves no
// such way, the band crosses the fewest lines it can: it is never left with a gap.

import type { DiagramGraph } from "./graph";
import { pillSize, STROKE_SHARE, type Drawing } from "./draw";
import { fitOcti, rounded, segD } from "./borders";
import { hyp } from "./fmath";
import type { RealWater, RiverArc, Crossing } from "./waterReal";
import { diagramSea, type SeaGrid, type SeaStats } from "./waterSea";

export interface DiagramWater {
  /** the rivers' centre lines as path commands (draw.ts `toPath`), one path per width, stroked with square-cut (butt)
   * ends, so a band's width steps square where it changes */
  rivers: Float64Array[];
  /** each path's width, layout units (T-2528) */
  bands: number[];
  /** per width, a stub (path commands, a hundredth of a pitch long) at each end of a band, stroked with round ends: the
   * band's own ends and where rivers meet stay round */
  riverEnds?: Float64Array[];
  /** the sea's outline as path commands, filled even-odd */
  sea?: Float64Array;
  /** checks only: the sea's rings as fitted */
  seaRings?: Float64Array[];
  seaStats?: SeaStats;
  /** checks only: each river run as routed (x, y pairs, before its corners are rounded), its arc and river, and the
   * crossings it was routed through (graph edges); `ends`, per end of the arc, what it meets: "sea" (a region of the
   * drawn sea, `region`), "river" (another drawn arc), "open" (it runs on off the drawing), "free" (where the river
   * ends on the ground) or "dry" (the sea it meets is not drawn near) */
  runs?: { parts: Float64Array[]; widths: Float64Array[]; wants: Float64Array[]; steps: Float64Array[]; arc: number; river: number; edges: number[]; marks: number[]; falseKeys: string[]; ends: string[]; regions: number[] }[];
  /** checks only: where bands cross lines the river does not, x, y */
  falsePts?: number[];
  /** checks only: the drawn sea on its grid */
  seaGrid?: SeaGrid;
  stats: WaterStats;
}
/** `falseCross`: lines crossed where the river does not cross them (T-2497); `rivers` drawn, `dropped` (crossed but
 * not drawn: narrow ones that would have had to cross other lines), `uncrossed` (found but no hop crosses them);
 * `mouths` drawn into the sea, `dry` ends meeting a sea that is not drawn near, `open` ends running off the drawing */
export interface WaterStats { arcs: number; crossings: number; leftOut: number; turned: number; sections: number; widened: number; noRoute: number; falseCross: number; rivers: number; dropped: number; uncrossed: number; mouths: number; dry: number; open: number; channels: number; ms: number }

/** a river band's least width, pitches: three lines' (T-2516) */
export const BAND = 3 * STROKE_SHARE;
/** T-2528: a band is the river's real width at the drawing's scale round it (`scaleAt`), smoothed along the river over
 * SMOOTH_M either side, between BAND and BAND_CAP pitches; a stretch at least FLOOR_SHARE of the river's widest drawn
 * stretch, times their real widths' ratio (so a big river narrows gently downstream, a narrow branch stays narrow) */
const BAND_CAP = 18, SMOOTH_M = 1500, FLOOR_SHARE = 0.5;
/** a band is narrowed where it would cover a pill or a line it does not cross, to keep CLEAR_MARGIN pitches off it;
 * widths are drawn in steps of WIDTH_STEP pitches, each width held over at least RUN_STEPS steps of a pitch at most
 * (narrowing early where the room shrinks), widening only by WIDEN_MIN pitches or more, and never changing at a bend, so
 * every edge of a band runs at 0, 45 or 90 degrees (Anita, 2026-11-05: water sticks to those) */
const CLEAR_MARGIN = 0.5, WIDTH_STEP = 0.25, RUN_STEPS = 4, WIDEN_MIN = 0.5;
/** a casing within this cosine of a band's way runs alongside it (30 degrees) */
const ALONG = 0.866;
/** a route's cost rises within this many pitches of a casing or pill beyond the band's half width */
const GAP = 1;
/** the routing grid's cell, pitches; how far a route may stray from its course; bends' costs (pitches of length) */
const CELL = 1, CORRIDOR = 12, B45 = 10, B90 = 30;
/** straying costs nothing within FREE_DEV pitches, then W_DEV at the corridor's edge (per unit length); nearness to a
 * casing W_PROX at touching */
const FREE_DEV = 8, W_DEV = 0.3, W_PROX = 1;
/** a way may cross other lines, each pitch inside a casing costing W_CROSS pitches of length, inside a hop the river
 * crosses elsewhere W_OWN, inside a pill W_PILL (T-2497, T-2516); it is looked for within ROUTE_M corridors of its
 * course */
const W_CROSS = 10, W_OWN = 60, W_PILL = 150, ROUTE_M = 2;
/** a crossing runs on until this many pitches past the outermost casing (and any other it would end on) */
const RUN_PAST = 0.5;
/** the shares of a hop a crossing may be moved to, from its real one, to end clear of other lines */
const SHARES = [0.2, 0.35, 0.5, 0.65, 0.8];
/** a course longer than this many pitches is routed in parts */
const MAX_GUIDE = 100;
/** a river's course sampled this often on the ground, metres */
const DENSIFY_M = 200;
/** a river is drawn on to its nearest mouth, and to each other mouth where it is at least MAJOR_M metres wide each side
 * of its middle within MOUTH_M of what is drawn, metres; the sea a mouth meets is looked for within MOUTH_REACH
 * pitches of where its place on the ground is carried */
const MAJOR_M = 150, MOUTH_M = 25000, MOUTH_REACH = 80;
/** a mouth at a sea not drawn near goes on by water on the ground to one that is within this, metres; pieces of one sea
 * drawn apart are joined by a channel within CHANNEL pitches */
const WAY_M = 30000, CHANNEL = 40;
/** a piece of sea under this many square pitches (a sliver the coast's fit leaves) needs no channel */
const CHANNEL_MIN = 60;
/** a river ending free (where it ends on the ground) runs on this many pitches past its last crossing, a wide one
 * TAIL_MAJOR; one running on off what is looked at runs on this far past the drawing's box */
const TAIL = 3, TAIL_MAJOR = 5, OPEN_RUN = 12;
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
  const stats: WaterStats = { arcs: 0, crossings: 0, leftOut: 0, turned: 0, sections: 0, widened: 0, noRoute: 0, falseCross: 0, rivers: 0, dropped: 0, uncrossed: 0, mouths: 0, dry: 0, open: 0, channels: 0, ms: 0 };
  const warp = makeWarp(g, x, y, 20 * pitch, 2, 2.5 * pitch);
  const sea = diagramSea(g, x, y, d, real, warp.J);
  const t0r = performance.now();

  // ---- which arcs are drawn: of each river some hop crosses, those crossed and those joining them, and the ways on to
  // its mouths
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
  const by: number[][] = Array.from({ length: nn }, () => []);
  arcs.forEach((a, q) => a.a !== a.b && (by[a.a].push(q), by[a.b].push(q)));
  {
    // the drawn parts of a river joined (a greedy Steiner tree: from each part, the nearest other by Dijkstra over the
    // arcs, joined, until all are)
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
            if (nd < dist[w]) (dist[w] = nd), (via[w] = q), heap.push(nd, w);
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
  {
    // on to the sea (T-2516): from what is drawn of each river along it to its nearest mouth, and to each wide mouth
    // within MOUTH_M; a river meeting no sea runs on to where it leaves what is looked at, if it does
    const dist = new Float64Array(nn).fill(Infinity), via = new Int32Array(nn).fill(-1), heap = new Heap();
    for (let v = 0; v < nn; v++) if (by[v].some((q) => drawn[q])) (dist[v] = 0), heap.push(0, v);
    while (heap.size) {
      const [dv, v] = heap.pop();
      if (dv > dist[v]) continue;
      for (const q of by[v]) {
        const w = arcs[q].a === v ? arcs[q].b : arcs[q].a, nd = dv + alen[q];
        if (nd < dist[w]) (dist[w] = nd), (via[w] = q), heap.push(nd, w);
      }
    }
    const take = (m: number) => {
      for (let v = m; dist[v] > 0; ) {
        const q = via[v];
        drawn[q] = true;
        v = arcs[q].a === v ? arcs[q].b : arcs[q].a;
      }
    };
    const wide = (v: number) => by[v].some((q) => arcs[q].half >= MAJOR_M);
    const nearest = new Map<number, number>(), open = new Map<number, number>();
    for (let v = 0; v < nn; v++) {
      if (dist[v] === Infinity || !by[v].length) continue;
      const r = arcs[by[v][0]].river, nd = real.nodes[v];
      if (nd.sea && (nearest.get(r) === undefined || dist[v] < dist[nearest.get(r)!])) nearest.set(r, v);
      if (nd.open && (open.get(r) === undefined || dist[v] < dist[open.get(r)!])) open.set(r, v);
    }
    for (const v of nearest.values()) take(v);
    for (const [r, v] of open) if (!nearest.has(r)) take(v);
    // a river meeting no sea and running on off nowhere (it ends on the ground at both ends, or the mask loses it) is
    // not drawn: it would stop short (T-2516)
    const short = new Set<number>();
    arcs.forEach((a, q) => drawn[q] && !nearest.has(a.river) && !open.has(a.river) && short.add(a.river));
    arcs.forEach((a, q) => short.has(a.river) && (drawn[q] = false));
    stats.dropped += short.size;
    for (let v = 0; v < nn; v++) if (real.nodes[v].sea && dist[v] > 0 && dist[v] <= MOUTH_M && wide(v)) take(v);
  }
  count();
  {
    const crossed = new Set<number>(), all = new Set<number>();
    arcs.forEach((a) => (all.add(a.river), a.cross.length && crossed.add(a.river)));
    stats.uncrossed = all.size - crossed.size;
  }

  // ---- the drawing's obstacles (casings and pills) by squares of OB
  const half = (BAND / 2) * pitch, C = half + GAP * pitch;
  const segs: number[] = [];
  for (const r of d.ribbons) for (let k = 0; k + 3 < r.length; k += 2) segs.push(r[k], r[k + 1], r[k + 2], r[k + 3], pitch / 2);
  const pillsFrom = segs.length;
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
  /** what a way pays a unit of length to cross the casings at (px, py) where it has to: dearer through a pill or a hop
   * in `own` (the river crosses those elsewhere) */
  const crossCost = (px: number, py: number, own: Set<number>) => {
    let c = 0;
    for (const s of obs.get(okey(Math.floor(px / OB), Math.floor(py / OB))) ?? []) {
      if (segD(px, py, segs[s], segs[s + 1], segs[s + 2], segs[s + 3]) >= segs[s + 4]) continue;
      c = Math.max(c, s >= pillsFrom ? W_PILL : own.has(ribE[s / 5]) ? W_OWN : W_CROSS);
    }
    return c;
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
  /** a route, crossing another line only where going round costs more (T-2516: a smooth band crossing a line reads
   * better than one winding through the gaps; T-2497: the drawing may have put its two ends on different sides of
   * some line), a hop in `own` (the river's) dearer still */
  const routeOne = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[], own: Set<number>): { p: number[]; fb: boolean } => {
    stats.sections++;
    const r = attempt(sx, sy, dS, ex, ey, dE, guide, ROUTE_M * Rc, own);
    if (r) return { p: r, fb: true };
    stats.noRoute++;
    return { p: [sx, sy, ex, ey], fb: true };
  };
  const attempt = (sx: number, sy: number, dS: number, ex: number, ey: number, dE: number, guide: number[], M: number, own: Set<number> | null): number[] | null => {
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
        const cx = sx + ((c % W) + i0) * h, cy = sy + (Math.floor(c / W) + j0) * h, od = clearAt(cx, cy);
        clr[c] = od;
        const dv = Math.max(0, gd[c] * h - free0) / (Rc - free0), pr = Math.max(0, 1 - od / CL);
        step[c] = 1 + W_DEV * dv * dv + W_PROX * pr * pr;
        if (od < 0 && c !== goal && c !== start) step[c] = own ? step[c] + crossCost(cx, cy, own) : Infinity;
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
  const RG = 4 * pitch, rib = new Map<number, number[]>(), ribE: number[] = [], ribL: number[] = [], ribP: number[] = [];
  {
    const edgeOf = new Map<number, number>();
    for (let e = 0; e < g.ea.length; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
    d.ribbons.forEach((r, li) => {
      const chain = g.lines[d.lineOf[li]].chain, at = d.chainAt[li];
      for (let k = 0; k + 3 < r.length; k += 2) {
        const ci = at[k / 2], u = chain[ci], w = chain[ci + 1], s = ribE.length;
        ribE.push(w === undefined ? -1 : edgeOf.get(Math.min(u, w) * n + Math.max(u, w)) ?? -1), ribL.push(li);
        ribP.push(r[k], r[k + 1], r[k + 2], r[k + 3]);
        for (let i = Math.floor(Math.min(r[k], r[k + 2]) / RG); i <= Math.floor(Math.max(r[k], r[k + 2]) / RG); i++) for (let j = Math.floor(Math.min(r[k + 1], r[k + 3]) / RG); j <= Math.floor(Math.max(r[k + 1], r[k + 3]) / RG); j++) {
          const l = rib.get(okey(i, j));
          if (l) l.push(s);
          else rib.set(okey(i, j), [s]);
        }
      }
    });
  }
  /** the ribbons (`${ribbon}|${edge}`) of edges not in `allow` that polyline p crosses */
  const crossKeys = (p: number[], allow: Set<number>, at?: number[]) => {
    const got = new Set<string>();
    for (let k = 0; k + 3 < p.length; k += 2) {
      const ax = p[k], ay = p[k + 1], ux = p[k + 2] - ax, uy = p[k + 3] - ay;
      for (let i = Math.floor(Math.min(ax, p[k + 2]) / RG); i <= Math.floor(Math.max(ax, p[k + 2]) / RG); i++) for (let j = Math.floor(Math.min(ay, p[k + 3]) / RG); j <= Math.floor(Math.max(ay, p[k + 3]) / RG); j++) for (const s of rib.get(okey(i, j)) ?? []) {
        if (allow.has(ribE[s])) continue;
        const cx = ribP[s * 4], cy = ribP[s * 4 + 1], vx = ribP[s * 4 + 2] - cx, vy = ribP[s * 4 + 3] - cy, den = ux * vy - uy * vx;
        if (Math.abs(den) < 1e-12) continue;
        const r = ((cx - ax) * vy - (cy - ay) * vx) / den, t = ((cx - ax) * uy - (cy - ay) * ux) / den;
        if (r >= 0 && r <= 1 && t >= 0 && t <= 1) got.add(`${ribL[s]}|${ribE[s]}`), at?.push(ax + ux * r, ay + uy * r);
      }
    }
    return got;
  };
  // ---- each drawn arc: from its start through each crossing to its end
  const nodeAt = new Map<number, [number, number]>();
  const nodePlace = (v: number) => {
    let p = nodeAt.get(v);
    if (!p) nodeAt.set(v, (p = free(...warped(real.nodes[v].x, real.nodes[v].y))));
    return p;
  };
  // a mouth is drawn into the sea it meets on the ground (T-2516): the nearest cell of a drawn region of that sea, deep
  // enough for the band's end, to where its place on the ground is carried
  const G = sea.grid, need = (3 * (half + pitch)) / G.h, mouthAt = new Map<number, [number, number, number] | null>();
  const mouthNear = (x: number, y: number, body: number) => {
    const [wx, wy] = warped(x, y), R = Math.ceil((MOUTH_REACH * pitch) / G.h);
    const ci = Math.floor((wx - G.x0) / G.h), cj = Math.floor((wy - G.y0) / G.h);
    let best: [number, number, number] | null = null, bd = Infinity;
    for (let j = Math.max(0, cj - R); j <= Math.min(G.H - 1, cj + R); j++) for (let i = Math.max(0, ci - R); i <= Math.min(G.W - 1, ci + R); i++) {
      const c = j * G.W + i, r = G.reg[c], dd = (i - ci) ** 2 + (j - cj) ** 2;
      if (r < 0 || dd >= bd || G.depth[c] < need || (G.body[r] !== body && G.body[r] >= 0 && body >= 0)) continue;
      (best = [G.x0 + (i + 0.5) * G.h, G.y0 + (j + 0.5) * G.h, r] as [number, number, number]), (bd = dd);
    }
    return best;
  };
  const mouth = (v: number) => {
    if (!mouthAt.has(v)) mouthAt.set(v, mouthNear(real.nodes[v].x, real.nodes[v].y, real.nodes[v].body));
    return mouthAt.get(v)!;
  };
  const mouthAny = (v: number) => mouthNear(real.nodes[v].x, real.nodes[v].y, -1);
  /** the routes along a course on the ground (`rr`, x, y pairs) from (fx, fy), leaving the way fd, to (tx, ty),
   * arriving the way td (-1 either way) */
  const course = (rr: number[], fx: number, fy: number, fd: number, tx: number, ty: number, td: number, own: Set<number>): number[][] => {
    const got: number[][] = [];
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
      got.push(routeOne(ax, ay, ad, bx, by, lastPart ? td : -1, sub, own).p);
      (ax = bx), (ay = by), (ad = -1);
    }
    return got;
  };
  // a mouth at a sea not drawn near goes on by water on the ground to one that is (T-2516: the Hackensack through
  // Newark Bay and the Kill Van Kull to the harbour)
  const drawnSeas = new Set(G.body.filter((b) => b >= 0));
  const wayAt = new Map<number, { pts: number[]; to: [number, number, number] } | null>();
  const way = (v: number) => {
    if (wayAt.has(v)) return wayAt.get(v)!;
    const nd = real.nodes[v], w = real.waterWay(nd.x, nd.y, (b) => b !== nd.body && drawnSeas.has(b), WAY_M);
    let out: { pts: number[]; to: [number, number, number] } | null = null;
    if (w) {
      const n2 = w.pts.length, to = mouthNear(w.pts[n2 - 2], w.pts[n2 - 1], w.body);
      if (to) out = { pts: w.pts, to };
    }
    wayAt.set(v, out);
    return out;
  };
  const [bx0, by0, bx1, by1] = d.box;
  /** a drawn piece of band: its points, the hops it crosses as the river does and the lines it crosses that do not
   * (`${ribbon}|${edge}`), its arc */
  type Part = { p: number[]; w: number[]; e: Set<number>; f: Set<string>; arc: number; river: number; ends: string[]; regions: number[] };
  const runs: DiagramWater["runs"] = [], parts: Part[] = [], marks = new Map<number, number[]>(), falsePts: number[] = [];
  const ways = new Map<number, number>(), dryAt = new Set<number>();
  // T-2528: each drawn arc sampled along its course, with the band's width wanted at each sample (before the room the
  // drawing leaves is looked at)
  const course0: { rs: number[]; ss: number[]; total: number; want: Float64Array }[] = [];
  {
    const minW = BAND * pitch, capW = BAND_CAP * pitch;
    arcs.forEach((A, ai) => {
      if (!drawn[ai]) return;
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
      const m = ss.length, raw = new Float64Array(m), want = new Float64Array(m);
      for (let q = 0; q < m; q++) raw[q] = 2 * A.half * warp.scaleAt(rs[q * 2], rs[q * 2 + 1]);
      for (let q = 0, lo = 0, hi = 0, sum = 0; q < m; q++) {
        while (hi < m && ss[hi] <= ss[q] + SMOOTH_M) sum += raw[hi++];
        while (ss[lo] < ss[q] - SMOOTH_M) sum -= raw[lo++];
        want[q] = sum / (hi - lo);
      }
      course0[ai] = { rs, ss, total: s, want };
    });
    // a river's floor: its widest stretch's share, at the real widths' ratio
    const widest = new Map<number, [number, number]>();
    course0.forEach((c, ai) => {
      if (!c) return;
      const top = Math.max(...c.want), r = arcs[ai].river, w = widest.get(r);
      if (!w || top > w[0]) widest.set(r, [top, arcs[ai].half]);
    });
    course0.forEach((c, ai) => {
      if (!c) return;
      const [top, half] = widest.get(arcs[ai].river)!, floor = FLOOR_SHARE * top * Math.min(1, arcs[ai].half / half);
      for (let q = 0; q < c.want.length; q++) c.want[q] = Math.min(capW, Math.max(minW, floor, c.want[q]));
    });
  }
  const wayW = new Map<number, number>();
  arcs.forEach((A, ai) => {
    if (!drawn[ai]) return;
    stats.arcs++;
    const { rs, ss, total, want } = course0[ai];
    /** the band's width wanted at `at` metres along the arc, and its mean from `a` to `b` */
    const wantAt = (at: number) => {
      let q = 1;
      while (q < ss.length - 1 && ss[q] < at) q++;
      const f = Math.max(0, Math.min(1, (at - ss[q - 1]) / Math.max(1e-9, ss[q] - ss[q - 1])));
      return want[q - 1] + (want[q] - want[q - 1]) * f;
    };
    const wantOver = (a: number, b: number) => {
      if (b - a < DENSIFY_M) return wantAt((a + b) / 2);
      let t = 0, c = 0;
      for (let q = 0; q < ss.length; q++) if (ss[q] >= a && ss[q] <= b) (t += want[q]), c++;
      return c ? t / c : wantAt((a + b) / 2);
    };
    const realAt = (at: number): [number, number] => {
      let q = 1;
      while (q < ss.length - 1 && ss[q] < at) q++;
      const f = Math.max(0, Math.min(1, (at - ss[q - 1]) / Math.max(1e-9, ss[q] - ss[q - 1])));
      return [rs[q * 2 - 2] + (rs[q * 2] - rs[q * 2 - 2]) * f, rs[q * 2 - 1] + (rs[q * 2 + 1] - rs[q * 2 - 1]) * f];
    };
    const cr0 = A.cross.map((c) => [c, drawnCross(c)] as const).filter((q) => q[1]) as [Crossing, Drawn][];
    stats.leftOut += A.cross.length - cr0.length;
    // a crossing run that crosses every line of another hop this river crosses serves for both (two hops side by side
    // in the drawing: PATH's into Christopher Street)
    const also = new Map<Crossing, number[]>(), served = new Set<number>(), cr: [Crossing, Drawn][] = [];
    for (const [c, dc] of cr0) {
      if (served.has(c.e)) continue;
      const lines = new Map<number, number>();
      for (const k of crossKeys([dc.ex, dc.ey, dc.xx, dc.xy], new Set([c.e]))) {
        const e = +k.split("|")[1];
        if (e !== c.e && A.cross.some((o) => o.e === e)) lines.set(e, (lines.get(e) ?? 0) + 1);
      }
      const got = [...lines].filter(([e, m]) => m === (stretches.get(e)?.length ?? -1)).map(([e]) => e);
      for (const e of got) {
        served.add(e);
        const before = cr.findIndex(([o]) => o.e === e);
        if (before >= 0) got.push(...(also.get(cr[before][0]) ?? [])), cr.splice(before, 1);
      }
      also.set(c, [...new Set(got)].filter((e) => e !== c.e)), cr.push([c, dc]);
    }
    const own = new Set(A.cross.map((c) => c.e)), tl = (A.half >= MAJOR_M ? TAIL_MAJOR : TAIL) * pitch;
    // ends: at the sea, into it; where other drawn arcs meet, the node; running on off what is looked at, on past the
    // drawing's box; where the river ends on the ground, a short run on past the crossing beside it
    const ends: string[] = [], regions: number[] = [];
    const endAt = (v: number): [number, number] => {
      const nd = real.nodes[v];
      if (nd.sea) {
        const m = mouth(v);
        if (m) return ends.push("sea"), regions.push(m[2]), stats.mouths++, [m[0], m[1]];
        const w = way(v);
        if (w) return ends.push("way"), ways.set(v, A.river), wayW.set(v, wantAt(v === A.a ? 0 : total)), nodePlace(v);
        // T-2540: else into whatever sea is drawn near (the drawn sea's body is voted by the places its cells are carried
        // back to, which can be a neighbouring sea's)
        const any = mouthAny(v);
        if (any) return ends.push("sea"), regions.push(any[2]), stats.mouths++, [any[0], any[1]];
        return ends.push("dry"), dryAt.add(v), nodePlace(v);
      }
      ends.push(deg[v] > 1 ? "river" : nd.open ? "open" : "free");
      return nodePlace(v);
    };
    const freeEnd = (v: number) => deg[v] <= 1 && !real.nodes[v].sea && !real.nodes[v].open;
    let s0 = 0, s1 = total;
    if (cr.length && freeEnd(A.a)) (s0 = cr[0][0].s), ends.push("free");
    if (cr.length && freeEnd(A.b)) s1 = cr[cr.length - 1][0].s;
    const [sx0, sy0] = s0 > 0 ? [NaN, NaN] : endAt(A.a);
    const [ex1, ey1] = s1 < total ? [NaN, NaN] : endAt(A.b);
    if (s1 < total) ends.push("free");
    // pieces in order: routes (e -1) and crossings (e their hop), each with the width wanted
    type Piece = { p: number[]; e: number; w: number; fb?: boolean; also?: number[] };
    const pieces: Piece[] = [];
    let fromS = s0, fx = sx0, fy = sy0, fd = -1;
    /** the routes from the last place to (tx, ty), arriving the way td */
    const leg = (toS: number, tx: number, ty: number, td: number): Piece[] => {
      const rr: number[] = [...realAt(fromS)];
      for (let q = 0; q < ss.length; q++) if (ss[q] > fromS && ss[q] < toS) rr.push(rs[q * 2], rs[q * 2 + 1]);
      rr.push(...realAt(toS));
      const w = wantOver(fromS, toS);
      return course(rr, fx, fy, fd, tx, ty, td, own).map((p) => ({ p, e: -1, w, fb: true }));
    };
    /** the lines a leg crosses where the river does not */
    const falseOf = (got: Piece[]) => got.reduce((t, pc) => t + (pc.fb ? crossKeys(pc.p, new Set()).size : 0), 0);
    for (const [c, dc] of cr) {
      if (c.s < s0 || c.s > s1) continue;
      if (fx !== fx) {
        const ux = UX[dc.dir] * tl, uy = UY[dc.dir] * tl, w = wantAt(c.s);
        pieces.push({ p: [dc.ex - ux, dc.ey - uy, dc.ex, dc.ey], e: -1, w, fb: true }, { p: [dc.ex, dc.ey, dc.xx, dc.xy], e: c.e, w, also: also.get(c) });
        (fromS = c.s), (fx = dc.xx), (fy = dc.xy), (fd = dc.dir);
        continue;
      }
      // crossed the way the river crosses on the ground; where the drawing leaves no clean way round to that side, the
      // other way (the drawing turned the place over) if that crosses fewer other lines
      let l = leg(c.s, dc.ex, dc.ey, dc.dir), run = [dc.ex, dc.ey, dc.xx, dc.xy], dir = dc.dir;
      const lf = falseOf(l);
      if (lf) {
        const back = leg(c.s, dc.xx, dc.xy, (dc.dir + 4) % 8);
        if (falseOf(back) < lf) (l = back), (run = [dc.xx, dc.xy, dc.ex, dc.ey]), (dir = (dc.dir + 4) % 8), stats.turned++;
      }
      pieces.push(...l, { p: run, e: c.e, w: wantAt(c.s), also: also.get(c) });
      (fromS = c.s), (fx = run[2]), (fy = run[3]), (fd = dir);
    }
    if (ex1 === ex1) pieces.push(...leg(s1, ex1, ey1, -1));
    else if (fd >= 0) pieces.push({ p: [fx, fy, fx + UX[fd] * tl, fy + UY[fd] * tl], e: -1, w: wantAt(fromS), fb: true });
    // an end running on off what is looked at goes on straight, away from the drawing's middle, past its box
    const openRun = (px: number, py: number) => {
      const k = dirOf(px - (bx0 + bx1) / 2, py - (by0 + by1) / 2), M = OPEN_RUN * pitch;
      let t = 0;
      while (t < 4000 * pitch && px + UX[k] * t > bx0 - M && px + UX[k] * t < bx1 + M && py + UY[k] * t > by0 - M && py + UY[k] * t < by1 + M) t += pitch;
      stats.open++;
      return [px, py, px + UX[k] * t, py + UY[k] * t];
    };
    if (pieces.length && ends[ends.length - 1] === "open") {
      const q = pieces[pieces.length - 1].p;
      pieces.push({ p: openRun(q[q.length - 2], q[q.length - 1]), e: -1, w: wantAt(total), fb: true });
    }
    if (pieces.length && ends[0] === "open" && s0 === 0) {
      const r = openRun(pieces[0].p[0], pieces[0].p[1]);
      pieces.unshift({ p: [r[2], r[3], r[0], r[1]], e: -1, w: wantAt(0), fb: true });
    }
    // one band: the lines it crosses where the river does not counted
    const out: Part = { p: [], w: [], e: new Set(), f: new Set(), arc: ai, river: A.river, ends, regions };
    for (const pc of pieces) {
      const allow = new Set(pc.e >= 0 ? [pc.e, ...(pc.also ?? [])] : []), bad = crossKeys(pc.p, allow);
      for (const k of bad) out.f.add(k);
      if (detail && bad.size) crossKeys(pc.p, allow, falsePts);
      for (const e of allow) out.e.add(e);
      for (let k = out.p.length ? 2 : 0; k < pc.p.length; k++) out.p.push(pc.p[k]);
      for (let k = 2; k < pc.p.length; k += 2) out.w.push(pc.w);
    }
    if (out.p.length >= 4) parts.push(out);
    marks.set(ai, cr.flatMap(([, dc]) => [dc.ex, dc.ey, dc.xx, dc.xy]));
  });
  // the ways on by water from mouths at seas not drawn near, each its own piece of band (as an arc of `arcs.length +`
  // its node)
  for (const [v, river] of ways) {
    const w = way(v)!, [fx, fy] = nodePlace(v);
    const p: number[] = [];
    for (const q of course(w.pts, fx, fy, -1, w.to[0], w.to[1], -1, new Set())) for (let k = p.length ? 2 : 0; k < q.length; k++) p.push(q[k]);
    if (p.length < 4) continue;
    stats.mouths++;
    const wv = wayW.get(v) ?? BAND * pitch;
    parts.push({ p, w: p.slice(2).filter((_, k) => k % 2 === 0).map(() => wv), e: new Set(), f: crossKeys(p, new Set()), arc: arcs.length + v, river, ends: ["river", "sea"], regions: [w.to[2]] });
  }
  // a river no wider than MAJOR_M where it is crossed is drawn only where it crosses no other line, else not at all
  // (T-2516: never in pieces; small rivers crossing lines read as noise, T-2497)
  {
    // first a narrow arc at an end of what is drawn of a river (the Harlem River beside the Hudson) that crosses
    // another line, and an arc left ending free with no crossing, until none is
    const dg = new Int32Array(nn);
    for (const q of parts) if (q.arc < arcs.length) dg[arcs[q.arc].a]++, dg[arcs[q.arc].b]++;
    const endFree = (v: number) => dg[v] <= 1 && ((!real.nodes[v].sea && !real.nodes[v].open) || dryAt.has(v));
    for (let again = true; again; ) {
      again = false;
      for (const q of parts) {
        const A = arcs[q.arc];
        if (!A || !q.p.length || A.a === A.b || !(endFree(A.a) || endFree(A.b)) || (q.e.size && !(A.half < MAJOR_M && q.f.size))) continue;
        (stats.leftOut += q.e.size), (q.p.length = 0), dg[A.a]--, dg[A.b]--, (again = true);
      }
    }
    const wide = new Set<number>(), dirty = new Set<number>(), all = new Set<number>();
    arcs.forEach((A) => A.cross.length && A.half >= MAJOR_M && wide.add(A.river));
    // (and a river left meeting no drawn sea: its mouths' seas are not drawn near, nor reached by water)
    const crossed = new Set<number>(), reaches = new Set<number>();
    for (const q of parts) if (q.p.length && q.ends.some((e) => e === "sea" || e === "way" || e === "open")) reaches.add(q.river);
    for (const q of parts) if (q.p.length && q.e.size && reaches.has(q.river)) crossed.add(q.river);
    for (const r of new Set(parts.map((q) => q.river))) if (r >= 0 && !crossed.has(r)) stats.dropped++;
    for (const q of parts) if (q.river >= 0 && !crossed.has(q.river)) q.p.length = 0;
    for (const q of parts) if (q.p.length && q.river >= 0) all.add(q.river), q.f.size && dirty.add(q.river);
    for (const q of parts) if (dirty.has(q.river) && !wide.has(q.river)) (stats.leftOut += q.e.size), (q.p.length = 0);
    for (const r of all) (dirty.has(r) && !wide.has(r) ? stats.dropped++ : stats.rivers++);
    for (const q of parts) if (q.p.length) stats.dry += q.ends.filter((e) => e === "dry").length;
  }
  // pieces of one sea on the ground drawn apart, and not joined by a river still drawn (T-2528: a river left out above
  // had joined them), are joined by a channel where they are within CHANNEL pitches (T-2516; as river -1 - that sea)
  {
    const nR = G.body.length, up = Array.from({ length: nR }, (_, r) => r), find = (r: number): number => (up[r] === r ? r : (up[r] = find(up[r])));
    const first = new Map<number, number>();
    for (const q of parts) if (q.p.length) for (const r of q.regions) first.has(q.river) ? (up[find(r)] = find(first.get(q.river)!)) : first.set(q.river, r);
    const area = new Float64Array(nR);
    for (let c = 0; c < G.reg.length; c++) if (G.reg[c] >= 0) area[G.reg[c]]++;
    const lim = (CHANNEL * pitch) / G.h, dist = new Float32Array(G.reg.length), from = new Int32Array(G.reg.length);
    for (const b of new Set(G.body)) {
      if (b < 0) continue;
      for (let guard = 0; guard < nR; guard++) {
        const mine = G.body.map((x, r) => (x === b && area[r] * G.h * G.h >= CHANNEL_MIN * pitch * pitch ? r : -1)).filter((r) => r >= 0);
        const groups = new Map<number, number>();
        for (const r of mine) groups.set(find(r), (groups.get(find(r)) ?? 0) + area[r]);
        if (groups.size < 2) break;
        const main = [...groups].sort((p, q) => q[1] - p[1])[0][0];
        // outward from the main piece, through anything, to the nearest cell of another piece of the same sea
        dist.fill(Infinity), from.fill(-1);
        const queue: number[] = [];
        for (let c = 0; c < G.reg.length; c++) if (G.reg[c] >= 0 && find(G.reg[c]) === main) (dist[c] = 0), queue.push(c);
        let hit = -1;
        for (let k = 0; k < queue.length && hit < 0; k++) {
          const c = queue[k], i = c % G.W, j = (c - i) / G.W;
          if (dist[c] >= lim) continue;
          for (const nc of [i > 0 ? c - 1 : -1, i < G.W - 1 ? c + 1 : -1, j > 0 ? c - G.W : -1, j < G.H - 1 ? c + G.W : -1]) {
            if (nc < 0 || dist[nc] !== Infinity) continue;
            (dist[nc] = dist[c] + 1), (from[nc] = c), queue.push(nc);
            const r = G.reg[nc];
            if (r >= 0 && G.body[r] === b && find(r) !== main && mine.includes(r)) {
              hit = nc;
              break;
            }
          }
        }
        if (hit < 0) break;
        let s0 = hit;
        while (from[s0] >= 0) s0 = from[s0];
        const at = (c: number): [number, number] => [G.x0 + ((c % G.W) + 0.5) * G.h, G.y0 + (Math.floor(c / G.W) + 0.5) * G.h];
        const [sx, sy] = at(s0), [ex, ey] = at(hit);
        const p = routeOne(sx, sy, -1, ex, ey, -1, [sx, sy, ex, ey], new Set()).p;
        parts.push({ p, w: p.slice(2).filter((_, k) => k % 2 === 0).map(() => BAND * pitch), e: new Set(), f: crossKeys(p, new Set()), arc: arcs.length + nn + G.reg.length + stats.channels, river: -1 - b, ends: ["sea", "sea"], regions: [G.reg[s0], G.reg[hit]] });
        up[find(G.reg[hit])] = main;
        stats.channels++;
      }
    }
  }
  // T-2528: each band at its wanted widths, narrowed where it would cover a pill or a line it does not cross there, the
  // narrowing tapered; drawn as runs of one width
  const byWidth = new Map<number, number[]>(), endsBy = new Map<number, number[]>(), partW = new Map<Part, [Float64Array, Float64Array, Float64Array]>();
  /** the room either side of a band's middle at (px, py), running (ux, uy): to the nearest pill, or casing running
   * alongside (within ALONG of its way) not in `skip` (ribbon segment indices), within `R`; lines across it it passes
   * under, as at a crossing */
  const roomAt = (px: number, py: number, ux: number, uy: number, R: number, skip: Set<number>) => {
    let best = R;
    const i0 = Math.floor((px - R) / OB), i1 = Math.floor((px + R) / OB), j0 = Math.floor((py - R) / OB), j1 = Math.floor((py + R) / OB);
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) for (const s of obs.get(okey(i, j)) ?? []) {
      if (s < pillsFrom) {
        if (skip.has(s / 5)) continue;
        const vx = segs[s + 2] - segs[s], vy = segs[s + 3] - segs[s + 1], vl = hyp(vx, vy);
        if (vl > 0 && Math.abs(vx * ux + vy * uy) < ALONG * vl) continue;
      }
      best = Math.min(best, segD(px, py, segs[s], segs[s + 1], segs[s + 2], segs[s + 3]) - segs[s + 4]);
    }
    return best;
  };
  for (const q of parts) {
    if (!q.p.length) continue;
    // the lines it crosses (as the river does, or where it had to) are no obstacle to it
    const skip = new Set<number>();
    for (let s = 0; s < ribE.length; s++) if (q.e.has(ribE[s]) || q.f.has(`${ribL[s]}|${ribE[s]}`)) skip.add(s);
    // its segments cut into steps of at most a pitch, each at the room there
    const P: number[] = [q.p[0], q.p[1]], want: number[] = [], len: number[] = [];
    for (let k = 0; k + 3 < q.p.length; k += 2) {
      const ax = q.p[k], ay = q.p[k + 1], bx = q.p[k + 2], by = q.p[k + 3], l = hyp(bx - ax, by - ay), m = Math.max(1, Math.ceil(l / pitch));
      for (let z = 1; z <= m; z++) P.push(ax + ((bx - ax) * z) / m, ay + ((by - ay) * z) / m), want.push(q.w[k / 2]), len.push(l / m);
    }
    const ns = want.length, w = new Float64Array(ns);
    for (let k = 0; k < ns; k++) {
      const mx = (P[k * 2] + P[k * 2 + 2]) / 2, my = (P[k * 2 + 1] + P[k * 2 + 3]) / 2, l = Math.max(1e-9, len[k]);
      const room = roomAt(mx, my, (P[k * 2 + 2] - P[k * 2]) / l, (P[k * 2 + 3] - P[k * 2 + 1]) / l, want[k] / 2 + CLEAR_MARGIN * pitch, skip) - CLEAR_MARGIN * pitch;
      w[k] = Math.max(BAND * pitch, Math.min(want[k], 2 * room));
    }
    // runs of one width, in steps (rounded down): each held RUN_STEPS steps, narrowing to the least room ahead, widening
    // only by WIDEN_MIN where that holds as long; only ever narrower than the room allows
    const step = WIDTH_STEP * pitch, qz = Float64Array.from(w, (v) => Math.max(BAND * pitch, Math.floor(v / step) * step));
    const least = (k: number) => {
      let v = Infinity;
      for (let j = k; j < Math.min(ns, k + RUN_STEPS); j++) v = Math.min(v, qz[j]);
      return v;
    };
    for (let k = 0; k < ns; ) {
      const wd = least(k);
      let j = k;
      while (j < ns && qz[j] >= wd && !(j - k >= RUN_STEPS && j + RUN_STEPS <= ns && least(j) >= wd + WIDEN_MIN * pitch)) w[j++] = wd;
      k = j;
    }
    // no width changes at a bend, where square-cut ends would leave a notch: the wider side narrowed past it
    const bendAt = (k: number) => {
      const ax = P[k * 2] - P[k * 2 - 2], ay = P[k * 2 + 1] - P[k * 2 - 1], bx = P[k * 2 + 2] - P[k * 2], by = P[k * 2 + 3] - P[k * 2 + 1];
      return Math.abs(ax * by - ay * bx) > 1e-9 * (hyp(ax, ay) * hyp(bx, by) + 1e-18);
    };
    for (let moved = true, pass = 0; moved && pass < ns; pass++) {
      moved = false;
      for (let k = 1; k < ns; k++) if (w[k] !== w[k - 1] && bendAt(k)) (w[k] > w[k - 1] ? (w[k] = w[k - 1]) : (w[k - 1] = w[k])), (moved = true);
    }
    if (detail) partW.set(q, [w, Float64Array.from(want), Float64Array.from(P)]);
    // the band's two ends round (stubs stroked with round ends), at its end widths
    if (ns)
    for (const [k, px, py, ux, uy] of [[0, P[0], P[1], P[2] - P[0], P[3] - P[1]], [ns - 1, P[ns * 2], P[ns * 2 + 1], P[ns * 2 - 2] - P[ns * 2], P[ns * 2 - 1] - P[ns * 2 + 1]]]) {
      const l = hyp(ux, uy) || 1, s = 0.01 * pitch, e = endsBy.get(w[k]) ?? endsBy.set(w[k], []).get(w[k])!;
      e.push(0, px, py, 1, px + (ux / l) * s, py + (uy / l) * s);
    }
    // runs of one width, each rounded at its corners unless that would cut across a line
    for (let k0 = 0; k0 < ns; ) {
      const ws = w[k0];
      let k1 = k0 + 1;
      while (k1 < ns && w[k1] === ws) k1++;
      const v = Float64Array.from(P.slice(k0 * 2, k1 * 2 + 2)), pts: number[] = [v[0], v[1]];
      // (the steps' points dropped where straight)
      for (let z = 1; z + 1 < v.length / 2; z++) {
        const ux = v[z * 2] - pts[pts.length - 2], uy = v[z * 2 + 1] - pts[pts.length - 1], wx = v[z * 2 + 2] - v[z * 2], wy = v[z * 2 + 3] - v[z * 2 + 1];
        if (Math.abs(ux * wy - uy * wx) > 1e-9 * pitch * pitch) pts.push(v[z * 2], v[z * 2 + 1]);
      }
      pts.push(v[v.length - 2], v[v.length - 1]);
      const pv = Float64Array.from(pts);
      let c: number[] = [];
      rounded(pv, false, RADIUS * pitch, c);
      if (crossKeys(flatten(c), q.e).size > crossKeys(pts, q.e).size) rounded(pv, false, 0, (c = []));
      const into = byWidth.get(ws) ?? byWidth.set(ws, []).get(ws)!;
      for (const x of c) into.push(x);
      k0 = k1;
    }
  }
  const widths = [...byWidth.keys()].sort((a, b) => a - b);
  const byArc = new Map<number, Part[]>();
  for (const q of parts) {
    if (!q.p.length) continue;
    (stats.crossings += q.e.size), (stats.falseCross += q.f.size);
    const l = byArc.get(q.arc);
    if (l) l.push(q);
    else byArc.set(q.arc, [q]);
  }
  if (detail) for (const [ai, l] of byArc) runs.push({ parts: l.map((q) => Float64Array.from(q.p)), widths: l.map((q) => partW.get(q)![0]), wants: l.map((q) => partW.get(q)![1]), steps: l.map((q) => partW.get(q)![2]), arc: ai, river: l[0].river, edges: l.flatMap((q) => [...q.e]), marks: marks.get(ai) ?? [], falseKeys: l.flatMap((q) => [...q.f]), ends: l[0].ends, regions: l[0].regions });
  stats.ms = performance.now() - t0r;
  return { rivers: widths.map((w) => Float64Array.from(byWidth.get(w)!)), bands: widths, riverEnds: widths.map((w) => Float64Array.from(endsBy.get(w) ?? [])), sea: sea.cmds, seaRings: detail ? sea.rings : undefined, seaGrid: detail ? sea.grid : undefined, runs: detail ? runs : undefined, falsePts: detail ? falsePts : undefined, stats, seaStats: sea.stats };
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
/** `scaleAt`'s least reach, metres */
const SCALE_R_M = 1500;

export function makeWarp(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, reach: number, sideHops: number, gap: number) {
  const J = localMaps(g, x, y), all = kdTree(g.x, g.y);
  const n = g.key.length, scales = new Float64Array(n);
  for (let i = 0; i < n; i++) scales[i] = Math.sqrt(Math.abs(J[i * 4] * J[i * 4 + 3] - J[i * 4 + 1] * J[i * 4 + 2]));
  const logs = Float64Array.from(scales, (s) => Math.log(Math.max(1e-9, s)));
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
    /** T-2528: the drawing's scale round a place on the ground, layout units a metre: the nearest stations' own maps'
     * scales, a geometric mean weighted by exp(-d^2 / 2 R^2), R SCALE_R_M or the fourth nearest's distance, so it
     * changes smoothly from place to place */
    scaleAt(sx: number, sy: number): number {
      const found = all.nearest(sx, sy, SK, sn, sd);
      if (!found) return medScale;
      const R2 = Math.max(SCALE_R_M * SCALE_R_M, sd[Math.min(3, found - 1)]);
      let sw = 0, sl = 0;
      for (let q = 0; q < found; q++) {
        const w = Math.exp(-sd[q] / (2 * R2));
        (sl += w * logs[sn[q]]), (sw += w);
      }
      return sw > 0 ? Math.exp(sl / sw) : medScale;
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
