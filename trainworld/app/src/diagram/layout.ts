// The network diagram's layout (T-2160, notes/T-2146.md section 3 steps 2 to 6), run in its own
// worker (workers/diagram.worker.ts):
//   A: real directions with each hop's length evened, by least squares;
//   B: rounds of octilinear least squares from A (each edge toward the nearest free 45 degree
//      direction, straight through two-edge stations, near the last round, kept apart), with room
//      for each edge's bundle of strokes;
//   segments between key nodes, split where their stations fit neither a straight run nor one bend;
//   the snap: one small solve that makes every segment near a 45 degree direction exact;
//   routing: each segment straight, or an orthogonal leg and a diagonal leg (or the reverse), its
//   stations placed along it by their share of its length;
//   the separation (T-2205): the drawing moved, octilinear throughout, until stations keep their
//   spacing from each other and from other pieces;
//   the order of the strokes (one per colour) on each piece, by lineBands.ts' pairwise rule.
// Units: metres-like layout units, x east, y north. A rebuild starts from the last B positions.

import type { DiagramGraph } from "./graph";
import { Terms } from "./lsqr";

export interface LayoutOptions {
  /** a hop's target length is L0 * (real length / L0) ^ alpha: the centre opens, the suburbs close in */
  L0: number;
  alpha: number;
  /** a stroke pitch is the median hop over this */
  puDiv: number;
  /** every edge at least this many times its two ends' bundle half widths */
  room: number;
  /** unconnected stations kept apart by their half widths plus this share of the median hop */
  apart: number;
  roundsA: number;
  roundsB: number;
  /** LSQR iterations a round */
  iters: number;
}

/** round 2's mockups (artifacts/trainworld/t2146/r2-*.png), for comparison */
export const LAYOUT_R2: LayoutOptions = { L0: 1000, alpha: 0.35, puDiv: 12, room: 1.6, apart: 0.5, roundsA: 10, roundsB: 30, iters: 400 };
/** "a little more compact" (Anita on round 2): shorter long hops, thicker strokes against the hops,
 * bundles a little closer (notes/T-2146.md, stage 2) */
export const LAYOUT: LayoutOptions = { L0: 1000, alpha: 0.3, puDiv: 9, room: 1.4, apart: 0.5, roundsA: 10, roundsB: 30, iters: 400 };

/** a segment this near a 45 degree direction is snapped onto it */
const SNAP_DEG = 11;
/** a segment this near is drawn straight; others get two legs */
const STRAIGHT_DEG = 0.5;
/** a segment's stations may lie this share of the median hop off its drawing before it is split */
const TOL_SHARE = 0.5;
const W_LEN = 0.6, W_STR = 0.6, W_PUSH = 3.5, W_SNAP = 30;
/** B pushes apart pairs nearer than this share of their spacing, toward this share of it */
const B_NEAR = 1.2, B_AIM = 1.12;
/** the snap: passes (each drawn, its pairs still too close pushed in the next), solves a pass,
 * LSQR iterations a solve, the push's weight and how far past the spacing it aims */
const SNAP_PASSES = 1, SNAP_ROUNDS = 4, SNAP_ITERS = 600, W_SNAP_PUSH = 2, PUSH_OVER = 1.05;
/** a rebuild's B rounds, and its pull to the last layout */
const WARM_ROUNDS = 10, W_GEO_WARM = 0.05;
/** the spread after drawing: passes, and how far past the spacing a move aims */
const SPREAD_PASSES = 4, SPREAD_OVER = 1.02;
/** a bent piece leaving a node the way another piece does costs this share of its length */
const PORT_COST = 1;
/** the separation (T-2205): passes (each drawn), rounds a pass (the spacing terms linearised again),
 * sweeps a round, sweeps of the fixed terms after each of the spacing terms, and at the end */
const SEP_PASSES = 3, SEP_ROUNDS = 10, SEP_SWEEPS = 60, FIX_REPEAT = 5, FINAL_SWEEPS = 10;
/** pairs this share of their spacing apart are taken in, and pushed this far past it; a station is
 * kept its half width, the leg's and NL_GAP pitches off another piece's leg */
const SEP_NEAR = 1.25, SEP_OVER = 1.15, NL_GAP = 1;
/** stations along a leg keep the larger of this share of their hop's target and ADJ_GAP times their
 * half widths together (so their pills do not touch) */
const SEP_GAP = 0.5, ADJ_GAP = 1.25;
/** a rebuild's spacing terms each move the drawing at most this many hops from the last drawing */
const SEP_CAP = 0.3;
/** a rebuild keeps to the last drawing when at least this share of its nodes were in it */
const PREV_SHARE = 0.9;
const DIRS: [number, number][] = Array.from({ length: 8 }, (_, k) => [Math.cos((k * Math.PI) / 4), Math.sin((k * Math.PI) / 4)]);

export interface Piece {
  /** the graph nodes along it, first to last */
  nodes: Int32Array;
  /** its drawing, x, y pairs: straight, or with one bend; every node is a vertex */
  verts: Float64Array;
  /** per node, its vertex */
  at: Int32Array;
  /** its strokes' colours, left to right looking from its first node to its last */
  cols: string[];
}

export interface DiagramLayout {
  /** per graph node, where it is drawn */
  x: Float64Array;
  y: Float64Array;
  pieces: Piece[];
  /** a stroke pitch, the median hop, layout units */
  pu: number;
  hop: number;
  /** per node, its bundle half width */
  hw: Float64Array;
  /** per node, its place after B, which a rebuild starts from */
  bx: Float64Array;
  by: Float64Array;
  ms: { a: number; b: number; route: number; order: number; total: number };
  straight: number;
  bent: number;
}

const median = (v: ArrayLike<number>) => {
  const s = Float64Array.from(v).sort();
  return s.length ? (s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
};
const octiDev = (dx: number, dy: number) => {
  let a = ((Math.atan2(dy, dx) * 180) / Math.PI) % 45;
  if (a < 0) a += 45;
  return Math.min(a, 45 - a);
};
const nearestDir = (dx: number, dy: number): [number, number] => {
  const k = Math.round(Math.atan2(dy, dx) / (Math.PI / 4));
  return [Math.cos((k * Math.PI) / 4), Math.sin((k * Math.PI) / 4)];
};

/** what both layout stages share: each edge's target length and direction, the stations' half widths */
interface Prep {
  n: number;
  ea: Int32Array;
  eb: Int32Array;
  L: Float64Array;
  gdx: Float64Array;
  gdy: Float64Array;
  hw: Float64Array;
  pu: number;
  dmin: number;
  nbr: number[][];
  adj: Set<number>;
}

function prep(g: DiagramGraph, o: LayoutOptions): Prep {
  const n = g.key.length, m = g.ea.length;
  const L = new Float64Array(m), gdx = new Float64Array(m), gdy = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e], dx = g.x[b] - g.x[a], dy = g.y[b] - g.y[a], gl = Math.max(1, Math.hypot(dx, dy));
    L[e] = o.L0 * (gl / o.L0) ** o.alpha;
    gdx[e] = dx / gl;
    gdy[e] = dy / gl;
  }
  const mid = median(L);
  const pu = mid / o.puDiv;
  const hw = new Float64Array(n).fill(pu);
  for (let e = 0; e < m; e++) {
    const cols = new Set<string>();
    for (let k = g.first[e]; k < g.first[e + 1]; k++) cols.add(g.lines[g.on[k]].colour);
    const h = ((cols.size - 1) / 2) * pu + pu;
    hw[g.ea[e]] = Math.max(hw[g.ea[e]], h);
    hw[g.eb[e]] = Math.max(hw[g.eb[e]], h);
  }
  for (let e = 0; e < m; e++) L[e] = Math.max(L[e], (hw[g.ea[e]] + hw[g.eb[e]]) * o.room);
  const nbr: number[][] = Array.from({ length: n }, () => []);
  const adj = new Set<number>();
  for (let e = 0; e < m; e++) {
    nbr[g.ea[e]].push(e);
    nbr[g.eb[e]].push(e);
    adj.add(g.ea[e] * n + g.eb[e]);
  }
  return { n, ea: g.ea, eb: g.eb, L, gdx, gdy, hw, pu, dmin: o.apart * mid, nbr, adj };
}

/** each edge's octilinear direction: at each node its edges take distinct directions nearest their
 * angles, the surest first; where the two ends disagree, the end deviating less wins */
function chooseDirs(P: Prep, pos: Float64Array): Int8Array {
  const m = P.ea.length, ang = new Float64Array(m);
  for (let e = 0; e < m; e++) ang[e] = Math.atan2(pos[P.eb[e] * 2 + 1] - pos[P.ea[e] * 2 + 1], pos[P.eb[e] * 2] - pos[P.ea[e] * 2]);
  const q = Math.PI / 4;
  const dev = (a: number) => Math.abs(a - Math.round(a / q) * q);
  const propD = new Float64Array(m).fill(Infinity), propK = new Int8Array(m);
  const taken = new Uint8Array(8);
  for (let v = 0; v < P.n; v++) {
    const inc = P.nbr[v].map((e) => [e, P.ea[e] === v ? ang[e] : ang[e] + Math.PI] as const).sort((x, y) => dev(x[1]) - dev(y[1]));
    taken.fill(0);
    for (const [e, a] of inc) {
      let bd = Infinity, bk = -1;
      for (let k = -4; k <= 4; k++) {
        const kk = (((Math.round(a / q) + k) % 8) + 8) % 8;
        if (taken[kk]) continue;
        let d = (a - kk * q + Math.PI) % (2 * Math.PI);
        if (d < 0) d += 2 * Math.PI;
        d = Math.abs(d - Math.PI);
        if (d < bd) bd = d, bk = kk;
      }
      if (bk < 0) bd = 0, bk = (((Math.round(a / q)) % 8) + 8) % 8;
      taken[bk] = 1;
      const ke = P.ea[e] === v ? bk : (bk + 4) % 8;
      if (bd < propD[e]) propD[e] = bd, propK[e] = ke;
    }
  }
  return propK;
}

/** unconnected pairs closer than their half widths plus `dmin` */
function closePairs(P: Prep, pos: Float64Array, out: number[], over = 1): void {
  out.length = 0;
  let hmax = 0;
  for (let i = 0; i < P.n; i++) hmax = Math.max(hmax, P.hw[i]);
  const cell = (P.dmin + 2 * hmax) * over;
  const grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  for (let i = 0; i < P.n; i++) {
    const k = key(Math.floor(pos[i * 2] / cell), Math.floor(pos[i * 2 + 1] / cell));
    const v = grid.get(k);
    if (v) v.push(i);
    else grid.set(k, [i]);
  }
  for (let i = 0; i < P.n; i++) {
    const cx = Math.floor(pos[i * 2] / cell), cy = Math.floor(pos[i * 2 + 1] / cell);
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
      const v = grid.get(key(gx, gy));
      if (!v) continue;
      for (const j of v) {
        if (j <= i || P.adj.has(i * P.n + j)) continue;
        const d = Math.hypot(pos[j * 2] - pos[i * 2], pos[j * 2 + 1] - pos[i * 2 + 1]);
        if (d < (P.dmin + P.hw[i] + P.hw[j]) * over) out.push(i, j);
      }
    }
  }
}

/** rounds of least squares from `start`: option A (real directions) or B (octilinear) */
function lsq(P: Prep, octi: boolean, rounds: number, start: Float64Array, wProx: number, wGeo: number, wOct: number, iters: number): Float64Array {
  let pos: Float64Array = Float64Array.from(start);
  const pairs: number[] = [];
  const I4 = [0, 0, 0, 0], C4 = [0, 0, 0, 0];
  for (let r = 0; r < rounds; r++) {
    const T = new Terms(2 * P.n);
    const K = octi ? chooseDirs(P, pos) : null;
    for (let e = 0; e < P.ea.length; e++) {
      const a = P.ea[e], b = P.eb[e];
      const dx = K ? DIRS[K[e]][0] : P.gdx[e], dy = K ? DIRS[K[e]][1] : P.gdy[e];
      I4[0] = 2 * b, I4[1] = 2 * b + 1, I4[2] = 2 * a, I4[3] = 2 * a + 1;
      C4[0] = dx, C4[1] = dy, C4[2] = -dx, C4[3] = -dy;
      T.add(I4, C4, P.L[e], W_LEN);
      C4[0] = -dy, C4[1] = dx, C4[2] = dy, C4[3] = -dx;
      T.add(I4, C4, 0, octi ? wOct : 1);
    }
    if (octi) {
      for (let v = 0; v < P.n; v++) {
        if (P.nbr[v].length !== 2) continue;
        const [e0, e1] = P.nbr[v];
        const u = P.ea[e0] === v ? P.eb[e0] : P.ea[e0], w = P.ea[e1] === v ? P.eb[e1] : P.ea[e1];
        for (let c = 0; c < 2; c++) T.add([2 * u + c, 2 * v + c, 2 * w + c], [1, -2, 1], 0, W_STR);
      }
    }
    for (let v = 0; v < P.n; v++) for (let c = 0; c < 2; c++) {
      T.add([2 * v + c], [1], pos[2 * v + c], wProx);
      T.add([2 * v + c], [1], start[2 * v + c], wGeo);
    }
    closePairs(P, pos, pairs, B_NEAR);
    for (let k = 0; k < pairs.length; k += 2) {
      const a = pairs[k], b = pairs[k + 1];
      let ux = pos[b * 2] - pos[a * 2], uy = pos[b * 2 + 1] - pos[a * 2 + 1];
      const dl = Math.hypot(ux, uy);
      if (dl > 1e-6) ux /= dl, uy /= dl;
      else ux = 1, uy = 0;
      if (dl < (P.dmin + P.hw[a] + P.hw[b]) * B_AIM) T.add([2 * b, 2 * b + 1, 2 * a, 2 * a + 1], [ux, uy, -ux, -uy], (P.dmin + P.hw[a] + P.hw[b]) * B_AIM, W_PUSH);
    }
    pos = T.solve(pos, iters);
  }
  return pos;
}

// ---- segments, splits, the snap and the routing (route.py)

function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  if (l2 < 1e-12) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
  return Math.hypot(px - ax - dx * t, py - ay - dy * t);
}
/** the farthest of points `pts` (x, y pairs) from the polyline `poly` */
function polyDev(pts: number[], poly: number[]): number {
  let worst = 0;
  for (let i = 0; i < pts.length; i += 2) {
    let best = Infinity;
    for (let j = 0; j + 3 < poly.length; j += 2) best = Math.min(best, segDist(pts[i], pts[i + 1], poly[j], poly[j + 1], poly[j + 2], poly[j + 3]));
    worst = Math.max(worst, best);
  }
  return worst;
}
/** both one-bend drawings from p to q: orthogonal then diagonal, and diagonal then orthogonal */
function twoLegs(px: number, py: number, qx: number, qy: number): [number[], number[]] {
  const dx = qx - px, dy = qy - py, m = Math.min(Math.abs(dx), Math.abs(dy)), sx = Math.sign(dx) || 1, sy = Math.sign(dy) || 1;
  if (Math.abs(dx) >= Math.abs(dy)) return [[px, py, px + dx - sx * m, py, qx, qy], [px, py, px + sx * m, py + dy, qx, qy]];
  return [[px, py, px, py + dy - sy * m, qx, qy], [px, py, px + dx, py + sy * m, qx, qy]];
}

/** chains between key nodes: a branch or a line's end, or where the set of lines changes */
function segments(g: DiagramGraph): number[][] {
  const n = g.key.length, m = g.ea.length;
  const adj: number[][] = Array.from({ length: n }, () => []);
  const sig: string[] = [];
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < m; e++) {
    adj[g.ea[e]].push(g.eb[e]);
    adj[g.eb[e]].push(g.ea[e]);
    sig.push([...g.on.subarray(g.first[e], g.first[e + 1])].sort((a, b) => a - b).join(","));
    edgeOf.set(g.ea[e] * n + g.eb[e], e);
  }
  const ek = (a: number, b: number) => edgeOf.get(Math.min(a, b) * n + Math.max(a, b))!;
  const ends = new Set<number>();
  for (const l of g.lines) if (l.chain.length) ends.add(l.chain[0]).add(l.chain[l.chain.length - 1]);
  const key = new Uint8Array(n);
  for (let v = 0; v < n; v++) {
    const ns = adj[v];
    if (ns.length !== 2 || ends.has(v) || sig[ek(v, ns[0])] !== sig[ek(v, ns[1])]) key[v] = 1;
  }
  const seen = new Uint8Array(m);
  const segs: number[][] = [];
  const walk = (k: number, nx: number) => {
    const path = [k, nx];
    while (!key[path[path.length - 1]]) {
      const [a, b] = adj[path[path.length - 1]];
      const next = a !== path[path.length - 2] ? a : b;
      path.push(next);
      if (next === path[0]) break;
    }
    for (let i = 1; i < path.length; i++) seen[ek(path[i - 1], path[i])] = 1;
    segs.push(path);
  };
  for (let k = 0; k < n; k++) if (key[k]) for (const nx of adj[k]) if (!seen[ek(k, nx)]) walk(k, nx);
  // rings with no key node at all
  for (let e = 0; e < m; e++) if (!seen[e]) {
    key[g.ea[e]] = 1;
    walk(g.ea[e], g.eb[e]);
  }
  return segs;
}

interface Routed { pieces: Piece[]; x: Float64Array; y: Float64Array; straight: number; bent: number; bases: Base[] }

function route(g: DiagramGraph, P: Prep, bx: Float64Array, by: Float64Array, prev: Float64Array | null): Routed {
  const n = g.key.length;
  const hops: number[] = [];
  for (let e = 0; e < g.ea.length; e++) hops.push(Math.hypot(bx[g.eb[e]] - bx[g.ea[e]], by[g.eb[e]] - by[g.ea[e]]));
  const tol = TOL_SHARE * median(hops);
  const raw: number[][] = [];
  for (const s of segments(g)) {
    const fit = (i: number, j: number): number[] => {
      if (j - i <= 1) return [];
      const inner: number[] = [];
      for (let t = i + 1; t < j; t++) inner.push(bx[s[t]], by[s[t]]);
      const ax = bx[s[i]], ay = by[s[i]], cx = bx[s[j]], cy = by[s[j]];
      if (octiDev(cx - ax, cy - ay) < SNAP_DEG && polyDev(inner, [ax, ay, cx, cy]) < tol) return [];
      const [l1, l2] = twoLegs(ax, ay, cx, cy);
      if (Math.min(polyDev(inner, l1), polyDev(inner, l2)) < tol) return [];
      let mm = i + 1, md = -1;
      for (let t = i + 1; t < j; t++) {
        const d = segDist(bx[s[t]], by[s[t]], ax, ay, cx, cy);
        if (d > md) md = d, mm = t;
      }
      return [...fit(i, mm), mm, ...fit(mm, j)];
    };
    const br = [0, ...fit(0, s.length - 1), s.length - 1];
    for (let k = 1; k < br.length; k++) raw.push(s.slice(br[k - 1], br[k] + 1));
  }
  const pieces0 = raw;
  // the snap, on every node: a piece near a 45 degree direction is made exact with its stations on
  // it, stations that share no edge are pushed apart, and the rest stays near B. Pairs still too
  // close once drawn are pushed in the next pass.
  let q: Float64Array = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) (q[i * 2] = bx[i]), (q[i * 2 + 1] = by[i]);
  const q0 = Float64Array.from(q);
  const pushed = new Set<number>();
  const pairs: number[] = [];
  let drawn: Routed | null = null;
  for (let pass = 0; pass < SNAP_PASSES; pass++) {
    for (let it = 0; it < SNAP_ROUNDS; it++) {
      const T = new Terms(q.length);
      for (const p of pieces0) {
        const a = p[0], b = p[p.length - 1];
        if (a === b) continue;
        const vx = q[b * 2] - q[a * 2], vy = q[b * 2 + 1] - q[a * 2 + 1];
        if (octiDev(vx, vy) >= SNAP_DEG) continue;
        const [dx, dy] = nearestDir(vx, vy);
        for (let i = 1; i < p.length; i++) T.add([2 * p[i], 2 * p[i] + 1, 2 * p[i - 1], 2 * p[i - 1] + 1], [-dy, dx, dy, -dx], 0, W_SNAP);
      }
      for (let i = 0; i < q.length; i++) T.add([i], [1], q0[i], 1);
      closePairs(P, q, pairs);
      for (let k = 0; k < pairs.length; k += 2) pushed.add(pairs[k] * n + pairs[k + 1]);
      for (const pk of pushed) {
        const a = Math.floor(pk / n), b = pk % n;
        let ux = q[b * 2] - q[a * 2], uy = q[b * 2 + 1] - q[a * 2 + 1];
        const dl = Math.hypot(ux, uy);
        if (dl > 1e-6) (ux /= dl), (uy /= dl);
        else (ux = 1), (uy = 0);
        const want = (P.dmin + P.hw[a] + P.hw[b]) * PUSH_OVER;
        if (dl >= want) continue;
        T.add([2 * b, 2 * b + 1, 2 * a, 2 * a + 1], [ux, uy, -ux, -uy], want, W_SNAP_PUSH);
      }
      q = T.solve(q, SNAP_ITERS, 1e-9, 1e-9);
    }
    drawn = draw(pieces0, q, n);
    closePairs(P, interleave(drawn.x, drawn.y), pairs);
    if (!pairs.length) break;
    for (let k = 0; k < pairs.length; k += 2) pushed.add(pairs[k] * n + pairs[k + 1]);
  }
  for (let pass = 0; pass < SEP_PASSES; pass++) {
    const s = separate(g, P, pieces0, drawn!, bx, by, prev);
    drawn = draw(pieces0, s.q, n, s.bends);
  }
  return drawn!;
}

/** The separation (T-2205; notes/T-2205.md): the drawing as drawn, moved in terms that keep it
 * octilinear by construction: each piece end free in the plane, each station a distance along its
 * piece's leg from that leg's end, a bent piece's legs meeting where their directions say. Linear terms:
 * straight pieces stay straight, stations keep their order and gaps along their legs, and stations keep
 * their spacing from each other and from other pieces' legs (linearised about the current drawing, redone
 * each round). A first layout meets them by sweeps of projections; a rebuild (`prev`, the last drawing's
 * places) by Hildreth's method from the last drawing, which finds the drawing nearest it, each term's
 * pull capped, so an edit moves little. Returns the places to draw from, and each bent piece's bend. */
function separate(g: DiagramGraph, P: Prep, raw: number[][], r: Routed, bx: Float64Array, by: Float64Array, prev: Float64Array | null): { q: Float64Array; bends: Map<number, [number, number]> } {
  const n = P.n;
  // the variables: x, y of each piece end (and of stations on a piece with no length), then one
  // distance per station along its leg
  const xyVar = new Int32Array(n).fill(-1), sVar = new Int32Array(n).fill(-1);
  let nv = 0;
  const ok = raw.map((p, k) => {
    const L = r.bases[k].line;
    return p[0] !== p[p.length - 1] && Math.hypot(L[L.length - 2] - L[0], L[L.length - 1] - L[1]) > 1e-9;
  });
  raw.forEach((p, k) => {
    for (const v of ok[k] ? [p[0], p[p.length - 1]] : p) if (xyVar[v] < 0) (xyVar[v] = nv), (nv += 2);
  });
  /** per node: the variables its x and y are sums of, with their weights */
  const lin: { vars: number[]; cx: number[]; cy: number[] }[] = Array.from({ length: n }, (_, v) => ({ vars: xyVar[v] >= 0 ? [xyVar[v], xyVar[v] + 1] : [], cx: [1, 0], cy: [0, 1] }));
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < P.ea.length; e++) edgeOf.set(P.ea[e] * n + P.eb[e], e);
  const gapOf = (u: number, w: number) => {
    const e = edgeOf.get(Math.min(u, w) * n + Math.max(u, w));
    return e === undefined ? 0 : Math.max(SEP_GAP * P.L[e], ADJ_GAP * (P.hw[u] + P.hw[w]));
  };
  interface PieceSep { k: number; a: number; z: number; d0: [number, number]; d1: [number, number] | null; leg0: number[]; leg1: number[]; hw: number }
  const ps: PieceSep[] = [];
  const pieceOf = new Int32Array(n).fill(-1);
  const x0: number[] = [];
  // Two pieces leaving a node the same way are drawn over each other and no push along their legs
  // parts them: all but one straight piece among them gets a short first leg 45 degrees off, toward the
  // side its stations lay on after B, which the spacing then lengthens.
  const port = (dx: number, dy: number) => ((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
  const portsAt = new Map<number, number[]>();
  const portOf = (k: number, end: 0 | 1) => {
    const L = r.bases[k].line, m = L.length;
    return end === 0 ? port(L[2] - L[0], L[3] - L[1]) : port(L[m - 4] - L[m - 2], L[m - 3] - L[m - 1]);
  };
  raw.forEach((p, k) => {
    if (!ok[k]) return;
    for (const end of [0, 1] as const) {
      const key = (end ? p[p.length - 1] : p[0]) * 8 + portOf(k, end), v = portsAt.get(key);
      if (v) v.push(k * 2 + end);
      else portsAt.set(key, [k * 2 + end]);
    }
  });
  const turned = new Map<number, [0 | 1, number]>();
  for (const [key, list] of portsAt) {
    if (list.length < 2) continue;
    const node = Math.floor(key / 8), straight = list.filter((ke) => r.bases[ke >> 1].line.length === 4 && !turned.has(ke >> 1));
    if (!straight.length) continue;
    const keep = list.length > straight.length ? -1 : straight.reduce((a, b) => (raw[b >> 1].length > raw[a >> 1].length ? b : a));
    const dir = DIRS[key % 8];
    for (const ke of straight) {
      if (ke === keep) continue;
      const k = ke >> 1, p = raw[k];
      let side = 0;
      for (let i = 1; i < p.length - 1; i++) side += dir[0] * (by[p[i]] - by[node]) - dir[1] * (bx[p[i]] - bx[node]);
      const pref = side >= 0 ? 1 : 7;
      const free = [pref, 8 - pref].map((s) => (key % 8 + s) % 8).find((pp) => !portsAt.has(node * 8 + pp));
      if (free === undefined) continue;
      portsAt.set(node * 8 + free, [ke]);
      turned.set(k, [(ke & 1) as 0 | 1, free]);
    }
  }
  raw.forEach((p, k) => {
    if (!ok[k]) return;
    const b = r.bases[k], L = b.line, a = p[0], z = p[p.length - 1];
    const e = edgeOf.get(Math.min(p[0], p[1]) * n + Math.max(p[0], p[1]));
    let hw = P.pu;
    if (e !== undefined) {
      const cols = new Set<string>();
      for (let j = g.first[e]; j < g.first[e + 1]; j++) cols.add(g.lines[g.on[j]].colour);
      hw = ((cols.size - 1) / 2) * P.pu + P.pu;
    }
    const tn = turned.get(k);
    const bent = L.length === 6 || !!tn;
    let d0 = nearestDir(L[2] - L[0], L[3] - L[1]), d1 = L.length === 6 ? nearestDir(L[4] - L[2], L[5] - L[3]) : null;
    // a turned piece: its stations all on the leg along its old direction, the new leg from its turned end
    let onFirst = (i: number) => !bent || b.s[i] <= b.Ls[1];
    if (tn) {
      const dd = DIRS[tn[1]];
      if (tn[0] === 0) (d1 = d0), (d0 = [dd[0], dd[1]]), (onFirst = () => false);
      else (d1 = [-dd[0], -dd[1]]), (onFirst = () => true);
    }
    const total = b.Ls[b.Ls.length - 1];
    const leg0: number[] = [], leg1: number[] = [];
    for (let i = 1; i < p.length - 1; i++) {
      const v = p[i];
      pieceOf[v] = k;
      sVar[v] = nv++;
      if (onFirst(i)) {
        leg0.push(v);
        x0[sVar[v]] = b.s[i];
        lin[v] = { vars: [xyVar[a], xyVar[a] + 1, sVar[v]], cx: [1, 0, d0[0]], cy: [0, 1, d0[1]] };
      } else {
        leg1.push(v);
        x0[sVar[v]] = total - b.s[i];
        lin[v] = { vars: [xyVar[z], xyVar[z] + 1, sVar[v]], cx: [1, 0, -d1![0]], cy: [0, 1, -d1![1]] };
      }
    }
    leg1.reverse();
    ps.push({ k, a, z, d0, d1, leg0, leg1, hw });
  });
  const X0 = new Float64Array(nv);
  for (let v = 0; v < n; v++) if (xyVar[v] >= 0) (X0[xyVar[v]] = r.x[v]), (X0[xyVar[v] + 1] = r.y[v]);
  for (let v = 0; v < n; v++) if (sVar[v] >= 0) X0[sVar[v]] = x0[sVar[v]];
  // a rebuild: the drawing nearest the last one, where it had the place
  if (prev) {
    for (let v = 0; v < n; v++) if (xyVar[v] >= 0 && !isNaN(prev[v * 2])) (X0[xyVar[v]] = prev[v * 2]), (X0[xyVar[v] + 1] = prev[v * 2 + 1]);
    for (const pc of ps) {
      const legsOf: [number[], number, number, number][] = [[pc.leg0, pc.a, pc.d0[0], pc.d0[1]]];
      if (pc.d1) legsOf.push([pc.leg1, pc.z, -pc.d1[0], -pc.d1[1]]);
      for (const [st, o, dx, dy] of legsOf) for (const v of st) if (!isNaN(prev[v * 2]) && !isNaN(prev[o * 2])) X0[sVar[v]] = dx * (prev[v * 2] - prev[o * 2]) + dy * (prev[v * 2 + 1] - prev[o * 2 + 1]);
    }
  }
  // a row is built as a map from variable to coefficient
  const row = new Map<number, number>();
  const putVar = (i: number, f: number) => row.set(i, (row.get(i) ?? 0) + f);
  const putXY = (u: number, cx: number, cy: number) => (putVar(xyVar[u], cx), putVar(xyVar[u] + 1, cy));
  // A station on a straight piece moved across it moves both the piece's ends alike, so the piece
  // stays straight (its offset across is either end's, the two being equal while it is straight).
  const straightOf: (PieceSep | undefined)[] = [];
  for (const pc of ps) if (!pc.d1) for (const v of pc.leg0) straightOf[v] = pc;
  /** the line a straight piece lies on, its offset along (nx, ny) across it */
  const putAcross = (pc: PieceSep, nx: number, ny: number, f: number) => {
    putXY(pc.a, (f * nx) / 2, (f * ny) / 2);
    putXY(pc.z, (f * nx) / 2, (f * ny) / 2);
  };
  const put = (v: number, nx: number, ny: number, f: number) => {
    const pc = straightOf[v];
    if (pc) {
      const [dx, dy] = pc.d0, al = nx * dx + ny * dy, be = -nx * dy + ny * dx;
      putXY(pc.a, f * al * dx, f * al * dy);
      putVar(sVar[v], f * al);
      putAcross(pc, -be * dy, be * dx, f);
      return;
    }
    const l = lin[v];
    for (let i = 0; i < l.vars.length; i++) row.set(l.vars[i], (row.get(l.vars[i]) ?? 0) + f * (nx * l.cx[i] + ny * l.cy[i]));
  };
  /** a bent piece's legs' lengths from its ends: z - a = l0 d0 + l1 d1 */
  const putLeg = (pc: PieceSep, which: 0 | 1, f: number) => {
    const [d0, d1] = [pc.d0, pc.d1!], c = d0[0] * d1[1] - d0[1] * d1[0];
    // l0 = cross(z - a, d1) / c, l1 = cross(d0, z - a) / c
    const [nx, ny] = which === 0 ? [d1[1] / c, -d1[0] / c] : [-d0[1] / c, d0[0] / c];
    put(pc.z, nx, ny, f);
    put(pc.a, nx, ny, -f);
  };
  const value = (X: Float64Array) => {
    let s = 0;
    for (const [i, c] of row) s += c * X[i];
    return s;
  };
  // places of every node and bend from the variables
  const nb = ps.filter((pc) => pc.d1).length;
  const q = new Float64Array((n + nb) * 2);
  const bendId = new Map<number, number>();
  ps.forEach((pc) => pc.d1 && bendId.set(pc.k, n + bendId.size));
  const place = (X: Float64Array) => {
    for (let v = 0; v < n; v++) {
      const l = lin[v];
      if (!l.vars.length) continue;
      let x = 0, y = 0;
      for (let i = 0; i < l.vars.length; i++) (x += l.cx[i] * X[l.vars[i]]), (y += l.cy[i] * X[l.vars[i]]);
      (q[v * 2] = x), (q[v * 2 + 1] = y);
    }
    for (const pc of ps) {
      if (!pc.d1) continue;
      putLeg(pc, 0, 1);
      const l0 = value(X);
      row.clear();
      const bi = bendId.get(pc.k)!;
      (q[bi * 2] = q[pc.a * 2] + pc.d0[0] * l0), (q[bi * 2 + 1] = q[pc.a * 2 + 1] + pc.d0[1] * l0);
    }
  };
  // the legs as point lists, for the stations near other pieces' legs
  const legs: Leg[] = [], legPiece: PieceSep[] = [], legOrigin: number[] = [];
  for (const pc of ps) {
    if (!pc.d1) {
      legs.push({ pts: [pc.a, ...pc.leg0, pc.z], dx: pc.d0[0], dy: pc.d0[1], piece: pc.k, a: pc.a, z: pc.z });
      legPiece.push(pc), legOrigin.push(pc.a);
      continue;
    }
    const bi = bendId.get(pc.k)!;
    legs.push({ pts: [pc.a, ...pc.leg0, bi], dx: pc.d0[0], dy: pc.d0[1], piece: pc.k, a: pc.a, z: pc.z });
    legs.push({ pts: [pc.z, ...pc.leg1, bi], dx: pc.d1[0], dy: pc.d1[1], piece: pc.k, a: pc.a, z: pc.z });
    legPiece.push(pc, pc), legOrigin.push(pc.a, pc.z);
  }
  const legHw: number[] = [];
  for (const pc of ps) legHw[pc.k] = pc.hw;
  const want = (a: number, b: number) => P.dmin + P.hw[a] + P.hw[b];
  const pairs: number[] = [];
  // each term a row of the variables, its target, and whether it is an equality or a least value
  interface Row { idx: Int32Array; c: Float64Array; t: number; eq: boolean; n2: number; mu: number; key: number }
  const emit = (to: Row[], t: number, eq: boolean, key = -1) => {
    const idx = Int32Array.from(row.keys()), c = Float64Array.from(row.values());
    row.clear();
    let n2 = 0;
    for (let i = 0; i < c.length; i++) n2 += c[i] * c[i];
    if (n2 > 1e-12) to.push({ idx, c, t, eq, n2, mu: 0, key });
  };
  /** Hildreth's row-action method (dual coordinate ascent): each term's own correction taken back,
   * the term met the least way, its correction kept; the sweeps converge to the drawing nearest X0
   * that meets every term, which moves little when X0 moves little */
  const cap = prev ? SEP_CAP * median(P.L) : Infinity;
  const hild = (X: Float64Array, list: Row[]) => {
    let worst = 0;
    for (const r of list) {
      let cz = r.mu * r.n2;
      for (let k = 0; k < r.idx.length; k++) cz += r.c[k] * X[r.idx[k]];
      // each correction at most `cap` long: where the terms cannot all be met, they settle on a
      // compromise rather than drift
      const m = cap / Math.sqrt(r.n2);
      let l = Math.min(m, Math.max(r.eq ? -m : 0, (r.t - cz) / r.n2));
      const step = r.mu + l;
      if (step === 0) continue;
      worst = Math.max(worst, Math.abs(step) * Math.sqrt(r.n2));
      for (let k = 0; k < r.idx.length; k++) X[r.idx[k]] += step * r.c[k];
      r.mu = -l;
    }
    return worst;
  };
  /** sweeps of projections: each term not met moves its variables the least way that meets it */
  const project = (X: Float64Array, list: Row[]) => {
    let worst = 0;
    for (const r of list) {
      let v = -r.t;
      for (let k = 0; k < r.idx.length; k++) v += r.c[k] * X[r.idx[k]];
      if (r.eq ? Math.abs(v) < 1e-12 : v >= 0) continue;
      const l = v / r.n2;
      worst = Math.max(worst, Math.abs(l));
      for (let k = 0; k < r.idx.length; k++) X[r.idx[k]] -= l * r.c[k];
    }
    return worst;
  };
  /** sweeps of `list` then the fixed terms until both are met within `tol` */
  const settle = (X: Float64Array, list: Row[], sweep: typeof project, sweeps = SEP_SWEEPS) => {
    for (let s = 0; s < sweeps; s++) {
      let w = sweep(X, list);
      for (let f = 0; f < FIX_REPEAT; f++) {
        const wf = sweep(X, fixed);
        w = Math.max(w, wf);
        if (wf < tol) break;
      }
      if (w < tol) return;
    }
  };
  const tol = 1e-4 * P.pu;
  // straight pieces stay straight; stations keep their order and gaps along their legs, and a bent
  // piece's legs a length
  const fixed: Row[] = [];
  for (const pc of ps) {
    if (!pc.d1) {
      put(pc.z, -pc.d0[1], pc.d0[0], 1);
      put(pc.a, -pc.d0[1], pc.d0[0], -1);
      emit(fixed, 0, true);
    }
    const legsOf: [number[], number, 0 | 1][] = [[pc.leg0, pc.a, 0]];
    if (pc.d1) legsOf.push([pc.leg1, pc.z, 1]);
    for (const [st, end, which] of legsOf) {
      st.forEach((v, i) => {
        putVar(sVar[v], 1);
        if (i) putVar(sVar[st[i - 1]], -1);
        emit(fixed, gapOf(i ? st[i - 1] : end, v), false);
      });
      const last = st.length ? st[st.length - 1] : -1;
      if (last >= 0) putVar(sVar[last], -1);
      if (pc.d1) (putLeg(pc, which, 1), emit(fixed, 0, false));
      else (put(pc.z, pc.d0[0], pc.d0[1], 1), put(pc.a, pc.d0[0], pc.d0[1], -1), emit(fixed, last >= 0 ? gapOf(last, pc.z) : gapOf(pc.a, pc.z), false));
    }
    if (pc.d1) {
      // the two stations either side of the bend keep their gap round it
      const u0 = pc.leg0.length ? pc.leg0[pc.leg0.length - 1] : pc.a, u1 = pc.leg1.length ? pc.leg1[pc.leg1.length - 1] : pc.z;
      putLeg(pc, 0, 1);
      putLeg(pc, 1, 1);
      if (u0 !== pc.a) putVar(sVar[u0], -1);
      if (u1 !== pc.z) putVar(sVar[u1], -1);
      emit(fixed, gapOf(u0, u1), false);
    }
  }
  const X = Float64Array.from(X0);
  // the spacing terms by key: a pair of stations (below n * n), or a station and another piece's leg.
  // A rebuild keeps a term once taken in, since its correction, dropped, would spring back.
  const taken = new Map<number, Row>();
  const nlKey = (v: number, li: number) => n * n + v * legs.length + li;
  for (let round = 0; round < SEP_ROUNDS; round++) {
    place(X);
    const keys = new Set<number>(taken.keys());
    closePairs(P, q.subarray(0, n * 2), pairs, SEP_NEAR);
    for (let k = 0; k < pairs.length; k += 2) keys.add(pairs[k] * n + pairs[k + 1]);
    nearLegs(P, legs, legHw, pieceOf, q, n, (v, li) => keys.add(nlKey(v, li)));
    const live: Row[] = [];
    for (const key of keys) {
      const old = taken.get(key);
      if (key < n * n) {
        const a = Math.floor(key / n), b = key % n;
        let ux = q[b * 2] - q[a * 2], uy = q[b * 2 + 1] - q[a * 2 + 1];
        const dl = Math.hypot(ux, uy);
        if (dl > 1e-6) (ux /= dl), (uy /= dl);
        else if (old) {
          live.push(old);
          continue;
        } else (ux = 1), (uy = 0);
        put(b, ux, uy, 1);
        put(a, ux, uy, -1);
        emit(live, want(a, b) * SEP_OVER, false, key);
      } else {
        const v = Math.floor((key - n * n) / legs.length), li = (key - n * n) % legs.length, lg = legs[li], o = legOrigin[li];
        let nx = -lg.dy, ny = lg.dx;
        if ((q[v * 2] - q[o * 2]) * nx + (q[v * 2 + 1] - q[o * 2 + 1]) * ny < 0) (nx = -nx), (ny = -ny);
        put(v, nx, ny, 1);
        if (legPiece[li].d1) put(o, nx, ny, -1);
        else putAcross(legPiece[li], nx, ny, -1);
        emit(live, (P.hw[v] + legHw[lg.piece] + NL_GAP * P.pu) * SEP_OVER, false, key);
      }
      const r = live[live.length - 1];
      if (r && r.key === key && old) r.mu = old.mu;
    }
    if (!prev) {
      if (!live.length && round) break;
      settle(X, live, project);
      continue;
    }
    taken.clear();
    for (const r of live) taken.set(r.key, r);
    // X from X0 and every term's correction
    X.set(X0);
    for (const list of [fixed, live]) for (const r of list) if (r.mu) for (let k = 0; k < r.idx.length; k++) X[r.idx[k]] -= r.mu * r.c[k];
    settle(X, live, hild);
  }
  settle(X, [], project, FINAL_SWEEPS);
  place(X);
  const bends = new Map<number, [number, number]>();
  for (const [k, bi] of bendId) bends.set(k, [q[bi * 2], q[bi * 2 + 1]]);
  return { q: q.slice(0, n * 2), bends };
}

/** each station nearer another piece's leg than its own half width, the leg's and NL_GAP pitches,
 * between two of the leg's points: `hit(station, leg, point index after, share along)` */
interface Leg { pts: number[]; dx: number; dy: number; piece: number; a: number; z: number }

function nearLegs(P: Prep, legs: Leg[], legHw: number[], pieceOf: Int32Array, q: Float64Array, n: number, hit: (v: number, leg: number, j: number, t: number) => void): void {
  let hmax = 0;
  for (let i = 0; i < n; i++) hmax = Math.max(hmax, P.hw[i]);
  const reach = 2 * hmax + NL_GAP * P.pu, cell = Math.max(reach, P.dmin);
  const grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(q[i * 2] / cell), Math.floor(q[i * 2 + 1] / cell));
    const v = grid.get(k);
    if (v) v.push(i);
    else grid.set(k, [i]);
  }
  legs.forEach((lg, li) => {
    const ends = new Set([lg.pts[0], lg.pts[lg.pts.length - 1], lg.a, lg.z]);
    for (let j = 1; j < lg.pts.length; j++) {
      const u = lg.pts[j - 1], w = lg.pts[j];
      const ux = q[u * 2], uy = q[u * 2 + 1], wx = q[w * 2], wy = q[w * 2 + 1];
      const sx = wx - ux, sy = wy - uy, l2 = sx * sx + sy * sy;
      if (l2 < 1e-12) continue;
      const x0 = Math.floor((Math.min(ux, wx) - reach) / cell), x1 = Math.floor((Math.max(ux, wx) + reach) / cell);
      const y0 = Math.floor((Math.min(uy, wy) - reach) / cell), y1 = Math.floor((Math.max(uy, wy) + reach) / cell);
      for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) {
        const cand = grid.get(key(gx, gy));
        if (!cand) continue;
        for (const v of cand) {
          if (v === u || v === w || ends.has(v) || pieceOf[v] === lg.piece) continue;
          const t = ((q[v * 2] - ux) * sx + (q[v * 2 + 1] - uy) * sy) / l2;
          if (t <= 0 || t >= 1) continue;
          const d = Math.abs((q[v * 2] - ux) * sy - (q[v * 2 + 1] - uy) * sx) / Math.sqrt(l2);
          if (d < (P.hw[v] + legHw[lg.piece] + NL_GAP * P.pu) * SEP_NEAR) hit(v, li, j, t);
        }
      }
    }
  });
}

const interleave = (x: Float64Array, y: Float64Array) => {
  const o = new Float64Array(x.length * 2);
  for (let i = 0; i < x.length; i++) (o[i * 2] = x[i]), (o[i * 2 + 1] = y[i]);
  return o;
};

/** each piece straight or with one bend between its snapped ends, its stations where they project
 * onto it, in order and at least half an even share of it apart */
function draw(raw: number[][], q: Float64Array, n: number, bends?: Map<number, [number, number]>): Routed {
  const x = new Float64Array(n), y = new Float64Array(n);
  for (const p of raw) for (const k of [p[0], p[p.length - 1]]) (x[k] = q[k * 2]), (y[k] = q[k * 2 + 1]);
  const pieces: Piece[] = [], bases: Base[] = [];
  let straight = 0, bent = 0;
  // each piece leaves a node in one of eight directions; two pieces leaving it the same way would be
  // drawn over each other, so a bent piece takes the legs whose ends are free where it can
  const port = (dx: number, dy: number) => ((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
  const taken = new Set<number>();
  const ends = (l: number[], a: number, b: number) => {
    const m = l.length;
    return [a * 8 + port(l[2] - l[0], l[3] - l[1]), b * 8 + port(l[m - 4] - l[m - 2], l[m - 3] - l[m - 1])];
  };
  const isStraight = (p: number[]) => {
    const vx = x[p[p.length - 1]] - x[p[0]], vy = y[p[p.length - 1]] - y[p[0]];
    return Math.hypot(vx, vy) >= 1e-9 && octiDev(vx, vy) <= STRAIGHT_DEG;
  };
  for (const p of raw) if (isStraight(p)) for (const t of ends([x[p[0]], y[p[0]], x[p[p.length - 1]], y[p[p.length - 1]]], p[0], p[p.length - 1])) taken.add(t);
  for (const [pi, p] of raw.entries()) {
    const ax = x[p[0]], ay = y[p[0]], cx = x[p[p.length - 1]], cy = y[p[p.length - 1]];
    const vx = cx - ax, vy = cy - ay;
    let line: number[];
    if (Math.hypot(vx, vy) < 1e-9) line = [ax, ay, cx, cy];
    else if (isStraight(p)) {
      line = [ax, ay, cx, cy];
      straight++;
    } else {
      const inner: number[] = [];
      for (let i = 1; i < p.length - 1; i++) inner.push(q[p[i] * 2], q[p[i] * 2 + 1]);
      const [l1, l2] = twoLegs(ax, ay, cx, cy);
      const cost = (l: number[]) => (inner.length ? polyDev(inner, l) : 0) + ends(l, p[0], p[p.length - 1]).filter((t) => taken.has(t)).length * PORT_COST * Math.hypot(vx, vy);
      const bd = bends?.get(pi);
      // after the separation, the legs it was solved with
      line = bd ? (Math.hypot(l2[2] - bd[0], l2[3] - bd[1]) < Math.hypot(l1[2] - bd[0], l1[3] - bd[1]) ? l2 : l1) : cost(l2) < cost(l1) ? l2 : l1;
      for (const t of ends(line, p[0], p[p.length - 1])) taken.add(t);
      bent++;
    }
    const nl = line.length / 2;
    const Ls = [0];
    for (let i = 1; i < nl; i++) Ls.push(Ls[i - 1] + Math.hypot(line[i * 2] - line[i * 2 - 2], line[i * 2 + 1] - line[i * 2 - 1]));
    // after the separation, which kept the stations' gaps itself, only their order is held
    const total = Ls[nl - 1], k = p.length - 1, gap = ((bends ? 0.1 : 0.5) * total) / Math.max(1, k);
    const s = [0];
    for (let i = 1; i < k; i++) {
      // the nearest place on the drawing
      let best = Infinity, arc = 0;
      const px = q[p[i] * 2], py = q[p[i] * 2 + 1];
      for (let j = 0; j + 1 < nl; j++) {
        const x0 = line[j * 2], y0 = line[j * 2 + 1], dx = line[j * 2 + 2] - x0, dy = line[j * 2 + 3] - y0, l2 = dx * dx + dy * dy;
        const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        const d = Math.hypot(x0 + dx * t - px, y0 + dy * t - py);
        if (d < best) (best = d), (arc = Ls[j] + t * Math.sqrt(l2));
      }
      s.push(arc);
    }
    s.push(total);
    for (let i = 1; i < k; i++) s[i] = Math.max(s[i], s[i - 1] + gap);
    for (let i = k - 1; i >= 1; i--) s[i] = Math.min(s[i], s[i + 1] - gap);
    const pc: Piece = { nodes: Int32Array.from(p), verts: new Float64Array(0), at: new Int32Array(0), cols: [] };
    const b: Base = { line, Ls, s, gap };
    placeOn(pc, b, x, y);
    pieces.push(pc);
    bases.push(b);
  }
  return { pieces, x, y, straight, bent, bases };
}

/** a piece's drawing (one or two legs), its legs' arc lengths, its stations' arcs, their least gap */
interface Base { line: number[]; Ls: number[]; s: number[]; gap: number }

/** a piece's vertices and its stations' places from its base */
function placeOn(pc: Piece, b: Base, x: Float64Array, y: Float64Array): void {
  const { line, Ls, s } = b, nl = line.length / 2, p = pc.nodes, k = p.length - 1;
  const verts = [line[0], line[1]], at = [0];
  let j = 1;
  for (let i = 1; i < k; i++) {
    while (j < nl - 1 && Ls[j] <= s[i]) verts.push(line[j * 2], line[j * 2 + 1]), j++;
    const [px, py] = arcAt(b, s[i]);
    at.push(verts.length / 2);
    verts.push(px, py);
    x[p[i]] = px;
    y[p[i]] = py;
  }
  while (j < nl) verts.push(line[j * 2], line[j * 2 + 1]), j++;
  at.push(verts.length / 2 - 1);
  pc.verts = Float64Array.from(verts);
  pc.at = Int32Array.from(at);
}

/** the place `a` along a base's drawing, and its direction there */
function arcAt(b: Base, a: number): [number, number, number, number] {
  const { line, Ls } = b, nl = line.length / 2;
  let kk = 0;
  while (kk < nl - 2 && Ls[kk + 1] < a) kk++;
  const l = Math.max(1e-9, Ls[kk + 1] - Ls[kk]), f = (a - Ls[kk]) / l;
  const dx = line[kk * 2 + 2] - line[kk * 2], dy = line[kk * 2 + 3] - line[kk * 2 + 1];
  return [line[kk * 2] + dx * f, line[kk * 2 + 1] + dy * f, dx / l, dy / l];
}

/** Stations still too close once drawn, slid along their pieces apart: each move the least that
 * clears the pair, kept between its neighbours on the piece and taken only if it makes no new close
 * pair. Angles are untouched; key nodes stay. */
function spread(P: Prep, r: Routed): void {
  const n = P.n, onPiece = new Int32Array(n).fill(-1), idx = new Int32Array(n);
  r.pieces.forEach((pc, q) => {
    for (let i = 1; i < pc.nodes.length - 1; i++) (onPiece[pc.nodes[i]] = q), (idx[pc.nodes[i]] = i);
  });
  const want = (a: number, b: number) => P.dmin + P.hw[a] + P.hw[b];
  const pos = interleave(r.x, r.y), pairs: number[] = [];
  /** how many stations `v` would be too close to at (px, py) */
  const crowd = (v: number, px: number, py: number) => {
    let c = 0;
    for (let o = 0; o < n; o++) {
      if (o === v || P.adj.has(Math.min(o, v) * n + Math.max(o, v))) continue;
      const w = want(v, o), dx = pos[o * 2] - px, dy = pos[o * 2 + 1] - py;
      if (Math.abs(dx) < w && Math.abs(dy) < w && dx * dx + dy * dy < w * w) c++;
    }
    return c;
  };
  for (let pass = 0; pass < SPREAD_PASSES; pass++) {
    closePairs(P, pos, pairs);
    if (!pairs.length) return;
    const touched = new Set<number>();
    let moved = 0;
    for (let k = 0; k < pairs.length; k += 2) {
      const a = pairs[k], b = pairs[k + 1];
      const w = want(a, b) * SPREAD_OVER;
      let best: [number, number, number] | null = null;
      for (const [v, o] of [[a, b], [b, a]]) {
        const q = onPiece[v];
        if (q < 0 || touched.has(v)) continue;
        const base = r.bases[q], i = idx[v], s0 = base.s[i];
        const lo = base.s[i - 1] + base.gap, hi = base.s[i + 1] - base.gap;
        // along the leg it is on: |v + t u - o| = w, the root of least size within its room
        const [px, py, ux, uy] = arcAt(base, s0), rx = px - pos[o * 2], ry = py - pos[o * 2 + 1];
        const pr = rx * ux + ry * uy, disc = pr * pr - (rx * rx + ry * ry) + w * w;
        if (disc < 0) continue;
        const now = crowd(v, px, py);
        for (const t of [-pr + Math.sqrt(disc), -pr - Math.sqrt(disc)]) {
          const s1 = s0 + t;
          if (s1 < lo || s1 > hi || (best && Math.abs(t) >= Math.abs(best[2]))) continue;
          const [nx, ny] = arcAt(base, s1);
          if (crowd(v, nx, ny) >= now) continue;
          best = [v, s1, t];
        }
      }
      const apply = (v: number, s1: number) => {
        const q = onPiece[v];
        r.bases[q].s[idx[v]] = s1;
        placeOn(r.pieces[q], r.bases[q], r.x, r.y);
        for (const u of r.pieces[q].nodes) (pos[u * 2] = r.x[u]), (pos[u * 2 + 1] = r.y[u]);
        touched.add(v);
        moved++;
      };
      if (best) {
        apply(best[0], best[1]);
        continue;
      }
      // neither alone has room: both, part of the way each, along their own pieces
      if (onPiece[a] < 0 || onPiece[b] < 0 || onPiece[a] === onPiece[b] || touched.has(a) || touched.has(b)) continue;
      const room = (v: number) => {
        const base = r.bases[onPiece[v]], i = idx[v];
        return [base, base.s[i], base.s[i - 1] + base.gap, base.s[i + 1] - base.gap] as const;
      };
      const [ba, sa, loa, hia] = room(a), [bb, sb, lob, hib] = room(b);
      const before = crowd(a, pos[a * 2], pos[a * 2 + 1]) + crowd(b, pos[b * 2], pos[b * 2 + 1]);
      let pick: [number, number] | null = null;
      for (const f of [0.6, 0.9]) for (const da of [-1, 1]) for (const db of [-1, 1]) {
        if (pick) break;
        const s1a = Math.max(loa, Math.min(hia, sa + da * f * w)), s1b = Math.max(lob, Math.min(hib, sb + db * f * w));
        const [ax, ay] = arcAt(ba, s1a), [bx2, by2] = arcAt(bb, s1b);
        if (Math.hypot(ax - bx2, ay - by2) < want(a, b)) continue;
        const ox = pos[a * 2], oy = pos[a * 2 + 1];
        (pos[a * 2] = ax), (pos[a * 2 + 1] = ay);
        const after = crowd(a, ax, ay) + crowd(b, bx2, by2);
        (pos[a * 2] = ox), (pos[a * 2 + 1] = oy);
        if (after < before) pick = [s1a, s1b];
      }
      if (pick) (apply(a, pick[0]), apply(b, pick[1]));
    }
    if (!moved) return;
  }
}

/** per piece, its strokes (one per colour) left to right looking from its first node to its last:
 * of two lines, follow both ahead to where they part, and the one turning left goes left; if they
 * never part ahead, look back (lineBands.ts' rule, on the diagram's own positions) */
function order(g: DiagramGraph, pieces: Piece[], x: Float64Array, y: Float64Array): void {
  const n = g.key.length;
  const at = g.lines.map((l) => {
    const m = new Map<number, number[]>();
    l.chain.forEach((k, i) => {
      const v = m.get(k);
      if (v) v.push(i);
      else m.set(k, [i]);
    });
    return m;
  });
  const next = (l: number, prev: number, cur: number): number => {
    const c = g.lines[l].chain;
    for (const i of at[l].get(cur) ?? []) {
      if (i > 0 && c[i - 1] === prev && i + 1 < c.length) return c[i + 1];
      if (i + 1 < c.length && c[i + 1] === prev && i > 0) return c[i - 1];
    }
    return -1;
  };
  const turn = (prev: number, cur: number, nx: number) => {
    const ax = x[cur] - x[prev], ay = y[cur] - y[prev], bxx = x[nx] - x[cur], byy = y[nx] - y[cur];
    return Math.atan2(ax * byy - ay * bxx, ax * bxx + ay * byy);
  };
  const walk = (la: number, lb: number, prev: number, cur: number): number => {
    for (let s = 0; s < 400; s++) {
      const na = next(la, prev, cur), nb = next(lb, prev, cur);
      if (na < 0 || nb < 0) return 0;
      if (na !== nb) {
        const ta = turn(prev, cur, na), tb = turn(prev, cur, nb);
        return Math.abs(ta - tb) < 1e-6 ? 0 : ta > tb ? -1 : 1;
      }
      prev = cur;
      cur = na;
    }
    return 0;
  };
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < g.ea.length; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
  for (const pc of pieces) {
    const p = pc.nodes, e = edgeOf.get(Math.min(p[0], p[1]) * n + Math.max(p[0], p[1]));
    if (e === undefined) continue;
    const rep = new Map<string, number>();
    for (let k = g.first[e]; k < g.first[e + 1]; k++) {
      const l = g.on[k], c = g.lines[l].colour, r = rep.get(c);
      if (r === undefined || g.lines[l].chain.length > g.lines[r].chain.length) rep.set(c, l);
    }
    const cols = [...rep.keys()].sort();
    const m = cols.length;
    if (m < 2) {
      pc.cols = cols;
      continue;
    }
    const cmp = (ca: string, cb: string) => {
      const la = rep.get(ca)!, lb = rep.get(cb)!;
      let v = walk(la, lb, p[p.length - 2], p[p.length - 1]);
      if (!v) v = -walk(la, lb, p[1], p[0]);
      return v || (ca < cb ? -1 : 1);
    };
    // by how many of the others each is left of, as lineBands does where partings go round a circle
    const wins = new Int32Array(m), c = new Int8Array(m * m);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
      const r = cmp(cols[i], cols[j]);
      c[i * m + j] = r;
      c[j * m + i] = -r;
      if (r < 0) wins[i]++;
      else wins[j]++;
    }
    pc.cols = cols.map((_, i) => i).sort((i, j) => wins[j] - wins[i] || c[i * m + j]).map((i) => cols[i]);
  }
}

/** a layout's places by node key, for a rebuild: after B, and drawn (`lastPlaces`) */
export type LastPlaces = Map<number, [number, number] | [number, number, number, number]>;

export function lastPlaces(g: DiagramGraph, lay: DiagramLayout): LastPlaces {
  const m: LastPlaces = new Map();
  for (let i = 0; i < g.key.length; i++) m.set(g.key[i], [lay.bx[i], lay.by[i], lay.x[i], lay.y[i]]);
  return m;
}

/** The layout of a graph; `last` the last layout's places: a rebuild starts from its B positions,
 * and its separation keeps to the drawing nearest the last one. */
export function layoutDiagram(g: DiagramGraph, o: LayoutOptions = LAYOUT, last?: LastPlaces | null): DiagramLayout {
  const t0 = performance.now();
  const n = g.key.length;
  const P = prep(g, o);
  const geo = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) geo[i * 2] = g.x[i], (geo[i * 2 + 1] = g.y[i]);
  let start: Float64Array;
  let warm = false;
  if (last && last.size) {
    start = warmStart(g, P, last);
    warm = true;
  } else start = lsq(P, false, o.roundsA, geo, 0.02, 0.001, 1, o.iters);
  const t1 = performance.now();
  // a rebuild starts near where B ended: fewer rounds, held closer to the last layout
  const B = warm ? lsq(P, true, WARM_ROUNDS, start, 0.1, W_GEO_WARM, 3, o.iters) : lsq(P, true, o.roundsB, start, 0.1, 0.01, 3, o.iters);
  const t2 = performance.now();
  const bx = new Float64Array(n), by = new Float64Array(n);
  for (let i = 0; i < n; i++) bx[i] = B[i * 2], (by[i] = B[i * 2 + 1]);
  // the separation keeps near the last drawing only when most of this graph was in it: after a big
  // change (a whole network loaded) it separates afresh
  let prev: Float64Array | null = null;
  if (warm) {
    prev = new Float64Array(n * 2).fill(NaN);
    let known = 0;
    for (let i = 0; i < n; i++) {
      const p = last!.get(g.key[i]);
      if (p && p.length === 4) (prev[i * 2] = p[2]), (prev[i * 2 + 1] = p[3]), known++;
    }
    if (known < PREV_SHARE * n) prev = null;
  }
  const r = route(g, P, bx, by, prev);
  spread(P, r);
  const t3 = performance.now();
  order(g, r.pieces, r.x, r.y);
  const t4 = performance.now();
  return {
    x: r.x, y: r.y, pieces: r.pieces, pu: P.pu, hop: median(P.L), hw: P.hw, bx, by,
    ms: { a: warm ? 0 : t1 - t0, b: t2 - t1, route: t3 - t2, order: t4 - t3, total: t4 - t0 },
    straight: r.straight, bent: r.bent,
  };
}

/** A rebuild's start: each node where the last layout had it; a new one at its placed neighbours'
 * places plus its real offset from each, at the layout's scale there, or failing any, its real
 * place at the old layout's overall scale */
function warmStart(g: DiagramGraph, P: Prep, last: LastPlaces): Float64Array {
  const n = g.key.length, pos = new Float64Array(n * 2), known = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const p = last.get(g.key[i]);
    if (p) (pos[i * 2] = p[0]), (pos[i * 2 + 1] = p[1]), (known[i] = 1);
  }
  // the old layout against the real places, for nodes with nothing placed near: one scale, one shift
  let sx = 0, sy = 0, gx = 0, gy = 0, c = 0;
  for (let i = 0; i < n; i++) if (known[i]) (sx += pos[i * 2]), (sy += pos[i * 2 + 1]), (gx += g.x[i]), (gy += g.y[i]), c++;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) if (known[i]) {
    num += Math.hypot(pos[i * 2] - sx / c, pos[i * 2 + 1] - sy / c);
    den += Math.hypot(g.x[i] - gx / c, g.y[i] - gy / c);
  }
  const scale = den > 0 ? num / den : 1;
  for (let pass = 0; pass < n && known.includes(0); pass++) {
    let grew = false;
    for (let i = 0; i < n; i++) {
      if (known[i]) continue;
      let ax = 0, ay = 0, k = 0;
      for (const e of P.nbr[i]) {
        const o = P.ea[e] === i ? P.eb[e] : P.ea[e];
        if (known[o] !== 1) continue;
        const dx = g.x[i] - g.x[o], dy = g.y[i] - g.y[o], gl = Math.max(1, Math.hypot(dx, dy)), s = P.L[e] / gl;
        ax += pos[o * 2] + dx * s;
        ay += pos[o * 2 + 1] + dy * s;
        k++;
      }
      if (k) (pos[i * 2] = ax / k), (pos[i * 2 + 1] = ay / k), (known[i] = 2), (grew = true);
    }
    for (let i = 0; i < n; i++) if (known[i] === 2) known[i] = 1;
    if (!grew) break;
  }
  for (let i = 0; i < n; i++) if (!known[i]) {
    pos[i * 2] = c ? sx / c + (g.x[i] - gx / c) * scale : g.x[i];
    pos[i * 2 + 1] = c ? sy / c + (g.y[i] - gy / c) * scale : g.y[i];
  }
  return pos;
}
