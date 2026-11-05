// The network diagram's layout (T-2160, notes/T-2146.md section 3 steps 2 to 6), run in its own
// worker (workers/diagram.worker.ts):
//   A: real directions with each hop's length evened, by least squares;
//   the blocks (blocks.ts, T-2407): each city moved as one, and the stations between cities placed;
//   B: rounds of octilinear least squares from A (each edge toward the nearest free 45 degree
//      direction, straight through two-edge stations, near the last round, kept apart), with room
//      for each edge's bundle of strokes;
//   segments between key nodes, split where their stations fit neither a straight run nor one bend;
//   the snap: one small solve that makes every segment near a 45 degree direction exact;
//   routing: each segment straight, or an orthogonal leg and a diagonal leg (or the reverse), its
//   stations placed along it by their share of its length;
//   the separation (T-2205): the drawing moved, octilinear throughout, until stations keep their
//   spacing from each other and from other pieces;
//   the order of the strokes (one per line, lines of a colour on the same track sharing one) on each
//   piece, by lineBands.ts' pairwise rule.
// Units: metres-like layout units, x east, y north. A rebuild starts from the last B positions.

import type { DiagramGraph } from "./graph";
import { Terms } from "./lsqr";
import { atan2, DIRS8 as DIRS, hyp, pow } from "./fmath";
import { localStops } from "./draw";
import { BLOCK_GAP, blockHops, crowded, findBlocks, finishBlocks, heldPairs, placeBlocks, samePairs, tiedBlocks, tieTerms, type Blocks } from "./blocks";

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
  /** a lonely hop's target, as a share of its evened length (T-2289; 1 for none) */
  lonely: number;
  /** a hop's target between two local stops (draw.ts `localStops`), as a share of its evened length (T-2491; 1 or
   * left out for none) */
  local?: number;
  roundsA: number;
  roundsB: number;
  /** LSQR iterations a round */
  iters: number;
}

/** round 2's mockups (artifacts/trainworld/t2146/r2-*.png), for comparison */
export const LAYOUT_R2: LayoutOptions = { L0: 1000, alpha: 0.35, puDiv: 12, room: 1.6, apart: 0.5, lonely: 1, roundsA: 10, roundsB: 30, iters: 400 };
/** "a little more compact" (Anita on round 2): shorter long hops, thicker strokes against the hops,
 * bundles a little closer (notes/T-2146.md, stage 2) */
export const LAYOUT: LayoutOptions = { L0: 1000, alpha: 0.3, puDiv: 9, room: 1.4, apart: 0.5, lonely: 0.4, local: 0.4, roundsA: 10, roundsB: 30, iters: 400 };

/** a segment this near a 45 degree direction is snapped onto it */
const SNAP_DEG = 11;
/** a segment this near is drawn straight; others get two legs */
const STRAIGHT_DEG = 0.5;
/** before the separation, which makes a straight piece exact, a piece this near is drawn straight */
const ROUTE_STRAIGHT_DEG = 5;
/** a segment's stations may lie this share of the median hop off its drawing before it is split */
const TOL_SHARE = 0.5;
const W_LEN = 0.6, W_STR = 0.6, W_PUSH = 3.5, W_SNAP = 30;
/** B pushes apart pairs nearer than this share of their spacing, toward this share of it */
const B_NEAR = 1.2, B_AIM = 1.12;
/** the snap: passes (each drawn, its pairs still too close pushed in the next), solves a pass,
 * LSQR iterations a solve, the push's weight and how far past the spacing it aims */
const SNAP_PASSES = 1, SNAP_ROUNDS = 4, SNAP_ITERS = 600, W_SNAP_PUSH = 2, PUSH_OVER = 1.05;
/** the snap (T-2243): each hop held its length along its direction with this weight, at least this
 * share of its target; a line turned too far at a station drawn back toward its real turn with this (T-2255) */
const W_HOLD = 30, HOLD_MIN = 0.5, W_SNAP_TURN = 20;
/** B: a line through a junction keeps its real turn there with this weight (T-2243) */
const W_TURN = 1;
/** A and B (T-2290): two stations that share no edge and are nearer in reality than this many of their own mean
 * hop (the longer reach of the two) stay their spacing apart along their real bearing, with this weight */
const SIDE_HOPS = 1.5, W_SIDE = 3;
/** A's last ACROSS_ROUNDS rounds (T-2520): such a pair's offset across its real bearing is held at nothing with this
 * weight, so lines running side by side are not sheared against each other */
const W_SIDE_ACROSS = 1, ACROSS_ROUNDS = 3;
/** T-2520: two hops are parallel on the ground when they share no station and no line, their bearings are within
 * 15 degrees of each other (PAR_COS its cosine, written out so every engine agrees), each one's middle lies within
 * PAR_M of the other's line, and they overlap along it by at least half the shorter; their drawn lengths per real
 * metre are averaged with their partners' PAR_ROUNDS times */
const PAR_COS = 0.9659258262890683, PAR_M = 1000, PAR_ROUNDS = 3;
/** T-2534 (`crossTies`): crossings on the ground held in the drawing, and each one's weight in A and B */
const CROSS_COS = 0.9396926207859084, CROSS_END_M = 100, CROSS_DENSE_N = 12, CROSS_DENSE_M = 2500, CROSS_CELL_M = 3000;
const W_CROSS_A = 3, W_CROSS_B = 10, W_CROSS_SNAP = 30, W_CROSS_PT_B = 3, W_CROSS_PT_SNAP = 10;
/** stations within CROSS_SIDE_M of a crossing's longer hop kept on their side of it, at least their half width and
 * CROSS_SIDE_PU pitches off, with weight W_SIDE */
const CROSS_SIDE_M = 1500, CROSS_SIDE_PU = 2;/** a rebuild's B rounds, and its pull to the last layout */
const WARM_ROUNDS = 10, W_GEO_WARM = 0.05;
/** the spread after drawing: passes, and how far past the spacing a move aims */
const SPREAD_PASSES = 4, SPREAD_OVER = 1.02;
/** a bent piece leaving a node the way another piece does costs this share of its length, and each
 * line it folds back at its ends (T-2243) this share; its legs are chosen in this many passes */
const PORT_COST = 1, FOLD_COST = 4, LEG_PASSES = 2;
/** the separation (T-2205): passes (each drawn), rounds a pass (the spacing terms linearised again),
 * sweeps a round, sweeps of the fixed terms after each of the spacing terms, and at the end */
const SEP_PASSES = 3, SEP_ROUNDS = 10, SEP_SWEEPS = 60, FIX_REPEAT = 5, FINAL_SWEEPS = 10;
/** pairs this share of their spacing apart are taken in, and pushed this far past it; a station is
 * kept its half width, the leg's and NL_GAP pitches off another piece's leg */
const SEP_NEAR = 1.25, SEP_OVER = 1.15, NL_GAP = 1;
/** stations along a leg keep the larger of this share of their hop's target and ADJ_GAP times their
 * half widths together (so their pills do not touch) */
const SEP_GAP = 0.5, ADJ_GAP = 1.25;
/** a bent piece's leg with no stations keeps this many pitches where at none the piece would fold a line */
const LEG_MIN = 2;
/** a first layout's loading bar: where A ends and B ends, as shares of the layout's time */
const PROGRESS_A = 0.2, PROGRESS_B = 0.7;
/** a rebuild's spacing terms each move the drawing at most this many hops from the last drawing */
const SEP_CAP = 0.3;
/** a rebuild keeps to the last drawing when at least this share of its nodes were in it */
const PREV_SHARE = 0.9;
/** T-2289: a station is in a city when at least URBAN_N others lie within URBAN_M of it in reality; a
 * city is such stations within URBAN_M of each other. A lonely hop's target is LayoutOptions.lonely of its
 * evened length, at least LONELY_MIN pitches. */
export const URBAN_M = 2500, URBAN_N = 3, ALONE_M = 4000;
/** A's rounds: the options' up to A_ROUNDS_FROM complexes, rising to A_ROUNDS_MAX at A_ROUNDS_AT (T-2289) */
const A_ROUNDS_FROM = 2000, A_ROUNDS_AT = 8000, A_ROUNDS_MAX = 40;
const LONELY_MIN = 4;
/** a local hop's target is at least this many pitches (or its evened length, where shorter) (T-2491) */
const LOCAL_MIN = 3;
/** T-2407: a hop between blocks this many median hops long or more is a segment of its own */
const LONG_HOP = 4;

export interface Piece {
  /** the graph nodes along it, first to last */
  nodes: Int32Array;
  /** its drawing, x, y pairs: straight, or with one bend; every node is a vertex */
  verts: Float64Array;
  /** per node, its vertex */
  at: Int32Array;
  /** its strokes left to right looking from its first node to its last, each the lines (indices into the
   * graph's lines) drawn in it */
  strokes: number[][];
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
  ms: { a: number; b: number; route: number; order: number; total: number; blocks: number };
  straight: number;
  bent: number;
  /** the city blocks (blocks.ts) */
  blocks: Blocks;
}

/** how many strokes an edge's lines are drawn in (graph.ts' `stroke`) */
function strokesOn(g: DiagramGraph, e: number): number {
  const s = new Set<number>();
  for (let k = g.first[e]; k < g.first[e + 1]; k++) s.add(g.stroke[k]);
  return s.size;
}

const median = (v: ArrayLike<number>) => {
  const s = Float64Array.from(v).sort();
  return s.length ? (s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
};
const octiDev = (dx: number, dy: number) => {
  let a = ((atan2(dy, dx) * 180) / Math.PI) % 45;
  if (a < 0) a += 45;
  return Math.min(a, 45 - a);
};
const nearestDir = (dx: number, dy: number): [number, number] => {
  const d = DIRS[((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8];
  return [d[0], d[1]];
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
  /** per node, its pairs of edges that a line runs on through it from one to the other, turning less
   * than a right angle there in reality: edge * m + edge */
  through: Set<number>[];
  /** pairs of stations near each other in reality that share no edge, and the unit real bearing from
   * the first to the second: a, b, ux, uy per pair */
  side: Float64Array;
  /** per node, its city (`cityParts`) */
  city: Int32Array;
  /** per node, 1 where a segment must end though the lines go on (a station between blocks on a long hop) */
  keyNodes?: Uint8Array;
  /** hops parallel on the ground (`parallelHops`): e1, e2, and 1 where they run the same way, -1 where opposite */
  par: Int32Array;
  /** crossings on the ground held (`crossTies`): e1, e2, t1, t2; and the stations near e1 kept on their side: e, v, share, side */
  cross: Float64Array;
  crossSides: Float64Array;
}

function sharesLine(g: DiagramGraph, e1: number, e2: number): boolean {
  for (let i = g.first[e1]; i < g.first[e1 + 1]; i++) for (let j = g.first[e2]; j < g.first[e2 + 1]; j++) if (g.on[i] === g.on[j]) return true;
  return false;
}

/** The pairs of hops parallel on the ground (T-2520: Manhattan's avenue lines), as PAR_COS and PAR_M say */
function parallelHops(g: DiagramGraph): Int32Array {
  const m = g.ea.length, len = new Float64Array(m), mx = new Float64Array(m), my = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e];
    (len[e] = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a])), (mx[e] = (g.x[a] + g.x[b]) / 2), (my[e] = (g.y[a] + g.y[b]) / 2);
  }
  const cell = 2 * PAR_M + 2000, grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  for (let e = 0; e < m; e++) {
    const k = key(Math.floor(mx[e] / cell), Math.floor(my[e] / cell)), v = grid.get(k);
    if (v) v.push(e);
    else grid.set(k, [e]);
  }
  const cosMax = PAR_COS, out: number[] = [];
  for (let e1 = 0; e1 < m; e1++) {
    if (len[e1] < 1) continue;
    const a1 = g.ea[e1], b1 = g.eb[e1], ux = (g.x[b1] - g.x[a1]) / len[e1], uy = (g.y[b1] - g.y[a1]) / len[e1];
    const cx = Math.floor(mx[e1] / cell), cy = Math.floor(my[e1] / cell);
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) for (const e2 of grid.get(key(gx, gy)) ?? []) {
      if (e2 <= e1 || len[e2] < 1) continue;
      const a2 = g.ea[e2], b2 = g.eb[e2];
      if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2) continue;
      const vx = (g.x[b2] - g.x[a2]) / len[e2], vy = (g.y[b2] - g.y[a2]) / len[e2], c = ux * vx + uy * vy;
      if (Math.abs(c) < cosMax) continue;
      if (Math.abs((my[e2] - my[e1]) * ux - (mx[e2] - mx[e1]) * uy) > PAR_M || Math.abs((my[e1] - my[e2]) * vx - (mx[e1] - mx[e2]) * vy) > PAR_M) continue;
      const along = (i: number) => (g.x[i] - g.x[a1]) * ux + (g.y[i] - g.y[a1]) * uy;
      const lo = Math.min(along(a2), along(b2)), hi = Math.max(along(a2), along(b2));
      if (Math.min(len[e1], hi) - Math.max(0, lo) < 0.5 * Math.min(len[e1], len[e2])) continue;
      if (!sharesLine(g, e1, e2)) out.push(e1, e2, c > 0 ? 1 : -1);
    }
  }
  return Int32Array.from(out);
}

/** The crossings on the ground held in the drawing (T-2534): pairs of hops sharing no station and no line whose
 * tracks cross an odd number of times, at more than 20 degrees (CROSS_COS its cosine) and more than CROSS_END_M from
 * every end of either, with fewer than CROSS_DENSE_N stations within CROSS_DENSE_M of the crossing (a city's dense
 * core is left free, where holding every crossing would bend its lines about). `ties`: e1 (the longer track), e2 and
 * the share of each one's track from its first node to the crossing. `sides`: per such longer hop, each station
 * within CROSS_SIDE_M of its track outside dense places, the share of the track at its nearest point and the side
 * it lies on (1 left, -1 right, going from the hop's first node): e, v, share, side. */
export function crossTies(g: DiagramGraph): { ties: Float64Array; sides: Float64Array } {
  const n = g.key.length, m = g.ea.length;
  const pts: number[] = [], at = new Int32Array(m + 1);
  for (let e = 0; e < m; e++) {
    at[e] = pts.length / 2;
    if (g.hop && g.hopAt && g.hopAt[e + 1] - g.hopAt[e] >= 2) for (let k = g.hopAt[e] * 2; k < g.hopAt[e + 1] * 2; k++) pts.push(g.hop[k]);
    else pts.push(g.x[g.ea[e]], g.y[g.ea[e]], g.x[g.eb[e]], g.y[g.eb[e]]);
  }
  at[m] = pts.length / 2;
  const cum = new Float64Array(pts.length / 2);
  for (let e = 0; e < m; e++) for (let i = at[e] + 1; i < at[e + 1]; i++) cum[i] = cum[i - 1] + hyp(pts[i * 2] - pts[i * 2 - 2], pts[i * 2 + 1] - pts[i * 2 - 1]);
  const cell = CROSS_CELL_M, grid = new Map<number, number[]>(), box = new Float64Array(m * 4);
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  for (let e = 0; e < m; e++) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = at[e]; i < at[e + 1]; i++) (x0 = Math.min(x0, pts[i * 2])), (x1 = Math.max(x1, pts[i * 2])), (y0 = Math.min(y0, pts[i * 2 + 1])), (y1 = Math.max(y1, pts[i * 2 + 1]));
    (box[e * 4] = x0), (box[e * 4 + 1] = y0), (box[e * 4 + 2] = x1), (box[e * 4 + 3] = y1);
    for (let cx = Math.floor(x0 / cell); cx <= Math.floor(x1 / cell); cx++) for (let cy = Math.floor(y0 / cell); cy <= Math.floor(y1 / cell); cy++) {
      const k = key(cx, cy), v = grid.get(k);
      if (v) v.push(e);
      else grid.set(k, [e]);
    }
  }
  const sg = new Map<number, number[]>(), R = CROSS_DENSE_M;
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(g.x[i] / R), Math.floor(g.y[i] / R)), v = sg.get(k);
    if (v) v.push(i);
    else sg.set(k, [i]);
  }
  const dense = (px: number, py: number) => {
    let c = 0;
    const cx = Math.floor(px / R), cy = Math.floor(py / R);
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) for (const i of sg.get(key(gx, gy)) ?? []) if ((g.x[i] - px) * (g.x[i] - px) + (g.y[i] - py) * (g.y[i] - py) < R * R) c++;
    return c >= CROSS_DENSE_N;
  };
  const seen = new Set<number>(), out: number[] = [];
  for (const list of grid.values()) for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
    const e1 = Math.min(list[i], list[j]), e2 = Math.max(list[i], list[j]);
    if (seen.has(e1 * m + e2)) continue;
    seen.add(e1 * m + e2);
    if (box[e1 * 4 + 2] < box[e2 * 4] || box[e2 * 4 + 2] < box[e1 * 4] || box[e1 * 4 + 3] < box[e2 * 4 + 1] || box[e2 * 4 + 3] < box[e1 * 4 + 1]) continue;
    const a1 = g.ea[e1], b1 = g.eb[e1], a2 = g.ea[e2], b2 = g.eb[e2];
    if (a1 === a2 || a1 === b2 || b1 === a2 || b1 === b2 || sharesLine(g, e1, e2)) continue;
    let count = 0, t1 = 0, t2 = 0, cos = 1, px = 0, py = 0;
    for (let p = at[e1]; p + 1 < at[e1 + 1]; p++) for (let q = at[e2]; q + 1 < at[e2 + 1]; q++) {
      const ax = pts[p * 2], ay = pts[p * 2 + 1], rx = pts[p * 2 + 2] - ax, ry = pts[p * 2 + 3] - ay;
      const cx = pts[q * 2], cy = pts[q * 2 + 1], sx = pts[q * 2 + 2] - cx, sy = pts[q * 2 + 3] - cy;
      const den = rx * sy - ry * sx;
      if (den === 0) continue;
      const u = ((cx - ax) * sy - (cy - ay) * sx) / den, v = ((cx - ax) * ry - (cy - ay) * rx) / den;
      if (u < 0 || u >= 1 || v < 0 || v >= 1) continue;
      if (count++) continue;
      const lr = hyp(rx, ry), ls = hyp(sx, sy), L1 = cum[at[e1 + 1] - 1], L2 = cum[at[e2 + 1] - 1];
      t1 = L1 > 0 ? (cum[p] + u * lr) / L1 : 0.5;
      t2 = L2 > 0 ? (cum[q] + v * ls) / L2 : 0.5;
      cos = Math.abs((rx * sx + ry * sy) / (lr * ls));
      (px = ax + u * rx), (py = ay + u * ry);
    }
    if (count % 2 === 0 || cos > CROSS_COS) continue;
    let nearEnd = false;
    for (const v of [a1, b1, a2, b2]) if (hyp(g.x[v] - px, g.y[v] - py) < CROSS_END_M) nearEnd = true;
    if (nearEnd || dense(px, py)) continue;
    // the hop with the longer track first
    if (cum[at[e2 + 1] - 1] > cum[at[e1 + 1] - 1]) out.push(e2, e1, t2, t1);
    else out.push(e1, e2, t1, t2);
  }
  // the stations near a crossing's longer hop, outside dense places, on the side of its track they are on
  const sides: number[] = [], had = new Set<number>();
  for (let k = 0; k < out.length; k += 4) {
    const e = out[k], a = g.ea[e], b = g.eb[e];
    if (had.has(e)) continue;
    had.add(e);
    const cdx = g.x[b] - g.x[a], cdy = g.y[b] - g.y[a], cl = hyp(cdx, cdy), L1 = cum[at[e + 1] - 1];
    if (!(cl > 0) || !(L1 > 0)) continue;
    const S = CROSS_SIDE_M;
    for (let cx = Math.floor((box[e * 4] - S) / R); cx <= Math.floor((box[e * 4 + 2] + S) / R); cx++) for (let cy = Math.floor((box[e * 4 + 1] - S) / R); cy <= Math.floor((box[e * 4 + 3] + S) / R); cy++) {
      for (const v of sg.get(key(cx, cy)) ?? []) {
        if (v === a || v === b) continue;
        let bd = Infinity, bs = 0, side = 0;
        for (let p = at[e]; p + 1 < at[e + 1]; p++) {
          const ax = pts[p * 2], ay = pts[p * 2 + 1], rx = pts[p * 2 + 2] - ax, ry = pts[p * 2 + 3] - ay, l2 = rx * rx + ry * ry;
          if (!(l2 > 0)) continue;
          const u = Math.max(0, Math.min(1, ((g.x[v] - ax) * rx + (g.y[v] - ay) * ry) / l2)), d = hyp(ax + u * rx - g.x[v], ay + u * ry - g.y[v]);
          if (d >= bd) continue;
          const l = Math.sqrt(l2), cr = rx * (g.y[v] - ay) - ry * (g.x[v] - ax);
          // a stretch of track turned well off the hop's way says nothing of the side in the drawing
          (bd = d), (bs = (cum[p] + u * l) / L1), (side = (rx * cdx + ry * cdy) / (l * cl) < 0.5 ? 0 : cr > 0 ? 1 : -1);
        }
        if (bd < S && side && bs > 0 && bs < 1 && !dense(g.x[v], g.y[v])) sides.push(e, v, bs, side);
      }
    }
  }
  return { ties: Float64Array.from(out), sides: Float64Array.from(sides) };
}


/** each station near a crossing's longer hop (`crossTies`' sides) that `pos` (x, y pairs) has nearer it than its half width
 * and CROSS_SIDE_PU pitches, or on its other side, pushed out to that on its own side; `dir(e)` the hop's way now */
function crossSideTerms(T: Terms, P: Prep, pos: Float64Array, dir: (e: number) => [number, number], wt: number): void {
  const S = P.crossSides;
  for (let k = 0; k < S.length; k += 4) {
    const e = S[k], v = S[k + 1], s = S[k + 2], a = P.ea[e], b = P.eb[e], [dx, dy] = dir(e), nx = -dy * S[k + 3], ny = dx * S[k + 3];
    const want = P.hw[v] + CROSS_SIDE_PU * P.pu;
    const d = nx * (pos[v * 2] - (1 - s) * pos[a * 2] - s * pos[b * 2]) + ny * (pos[v * 2 + 1] - (1 - s) * pos[a * 2 + 1] - s * pos[b * 2 + 1]);
    if (d < want) T.add([2 * v, 2 * v + 1, 2 * a, 2 * a + 1, 2 * b, 2 * b + 1], [nx, ny, -(1 - s) * nx, -(1 - s) * ny, -s * nx, -s * ny], want, wt);
  }
}

function prep(g: DiagramGraph, o: LayoutOptions): Prep {
  const n = g.key.length, m = g.ea.length;
  const L = new Float64Array(m), gdx = new Float64Array(m), gdy = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e], dx = g.x[b] - g.x[a], dy = g.y[b] - g.y[a], gl = Math.max(1, hyp(dx, dy));
    L[e] = o.L0 * pow(gl / o.L0, o.alpha);
    gdx[e] = dx / gl;
    gdy[e] = dy / gl;
  }
  const mid = median(L);
  const pu = mid / o.puDiv;
  const hw = new Float64Array(n).fill(pu);
  for (let e = 0; e < m; e++) {
    const h = ((strokesOn(g, e) - 1) / 2) * pu + pu;
    hw[g.ea[e]] = Math.max(hw[g.ea[e]], h);
    hw[g.eb[e]] = Math.max(hw[g.eb[e]], h);
  }
  // the lonely stretches between cities drawn shorter, after the pitch is set from every hop (T-2289)
  const parts = cityParts(g);
  const lonely = parts.lonely;
  if (o.lonely < 1) for (let e = 0; e < m; e++) if (lonely[e]) L[e] = Math.max(Math.min(L[e], LONELY_MIN * pu), L[e] * o.lonely);
  // runs of plain stops (drawn small) closer together, so a long local branch is not drawn longer than the core (T-2491)
  if ((o.local ?? 1) < 1) {
    const small = localStops(g);
    for (let e = 0; e < m; e++) if (!lonely[e] && small[g.ea[e]] && small[g.eb[e]]) L[e] = Math.max(Math.min(L[e], LOCAL_MIN * pu), L[e] * o.local!);
  }
  // hops parallel on the ground drawn at one scale, so runs beside each other with stops at different spacings need
  // not bend to fit each other (T-2520)
  const par = parallelHops(g);
  if (par.length) {
    const gl = new Float64Array(m);
    for (let e = 0; e < m; e++) gl[e] = Math.max(1, hyp(g.x[g.eb[e]] - g.x[g.ea[e]], g.y[g.eb[e]] - g.y[g.ea[e]]));
    let s = Float64Array.from(L, (l, e) => l / gl[e]);
    for (let it = 0; it < PAR_ROUNDS; it++) {
      const num = Float64Array.from(s, (v, e) => v * gl[e]), den = Float64Array.from(gl);
      for (let k = 0; k < par.length; k += 3) {
        const a = par[k], b = par[k + 1];
        (num[a] += s[b] * gl[b]), (den[a] += gl[b]), (num[b] += s[a] * gl[a]), (den[b] += gl[a]);
      }
      s = Float64Array.from(num, (v, e) => v / den[e]);
    }
    for (let e = 0; e < m; e++) L[e] = s[e] * gl[e];
  }
  // a hop crossing others on the ground at two places or more is long enough to pass them at their scale, with room
  // either side (T-2534): a long single-line hop drawn short would squeeze the city it crosses
  const { ties: cross, sides: crossSides } = crossTies(g);
  if (cross.length) {
    const lo = new Float64Array(m).fill(1), hi = new Float64Array(m), sc = new Float64Array(m), cnt = new Float64Array(m);
    for (let k = 0; k < cross.length; k += 4) {
      const e = cross[k], o = cross[k + 1];
      (lo[e] = Math.min(lo[e], cross[k + 2])), (hi[e] = Math.max(hi[e], cross[k + 2]));
      (sc[e] += L[o] / Math.max(1, hyp(g.x[g.eb[o]] - g.x[g.ea[o]], g.y[g.eb[o]] - g.y[g.ea[o]]))), cnt[e]++;
    }
    for (let e = 0; e < m; e++) if (cnt[e] >= 2) {
      const gl = hyp(g.x[g.eb[e]] - g.x[g.ea[e]], g.y[g.eb[e]] - g.y[g.ea[e]]);
      L[e] = Math.max(L[e], (hi[e] - lo[e]) * gl * (sc[e] / cnt[e]) + 2 * LONELY_MIN * pu);
    }
  }
  for (let e = 0; e < m; e++) L[e] = Math.max(L[e], (hw[g.ea[e]] + hw[g.eb[e]]) * o.room);
  const nbr: number[][] = Array.from({ length: n }, () => []);
  const adj = new Set<number>();
  for (let e = 0; e < m; e++) {
    nbr[g.ea[e]].push(e);
    nbr[g.eb[e]].push(e);
    adj.add(g.ea[e] * n + g.eb[e]);
  }
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < m; e++) edgeOf.set(g.ea[e] * n + g.eb[e], e);
  const through: Set<number>[] = Array.from({ length: n }, () => new Set());
  for (const l of g.lines) {
    const c = l.chain;
    for (let i = 1; i + 1 < c.length; i++) {
      const u = c[i - 1], v = c[i], w = c[i + 1];
      if (u === w || (g.x[v] - g.x[u]) * (g.x[w] - g.x[v]) + (g.y[v] - g.y[u]) * (g.y[w] - g.y[v]) <= 0) continue;
      const e1 = edgeOf.get(Math.min(u, v) * n + Math.max(u, v)), e2 = edgeOf.get(Math.min(v, w) * n + Math.max(v, w));
      if (e1 === undefined || e2 === undefined) continue;
      through[v].add(e1 * m + e2).add(e2 * m + e1);
    }
  }
  return { n, ea: g.ea, eb: g.eb, L, gdx, gdy, hw, pu, dmin: o.apart * mid, nbr, adj, through, side: sidePairs(g, adj), city: parts.city, par, cross, crossSides };
}

/** The cities and the lonely hops between them (T-2289): `city` per node, its city (-1 for a station in
 * none), and `lonely` per edge, 1 for a hop whose lines are all of one colour and that leaves a city or
 * joins two: the long thin single-line stretches of the countryside, drawn shorter. */
export function cityParts(g: DiagramGraph): { city: Int32Array; lonely: Uint8Array } {
  const n = g.key.length, m = g.ea.length;
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  /** f on each other station within R of station i, from a grid of cell R */
  const within = (R: number) => {
    const grid = new Map<number, number[]>();
    for (let i = 0; i < n; i++) {
      const k = key(Math.floor(g.x[i] / R), Math.floor(g.y[i] / R)), v = grid.get(k);
      if (v) v.push(i);
      else grid.set(k, [i]);
    }
    return (i: number, f: (j: number) => void) => {
      const cx = Math.floor(g.x[i] / R), cy = Math.floor(g.y[i] / R);
      for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++)
        for (const j of grid.get(key(gx, gy)) ?? []) if (j !== i && (g.x[j] - g.x[i]) * (g.x[j] - g.x[i]) + (g.y[j] - g.y[i]) * (g.y[j] - g.y[i]) < R * R) f(j);
    };
  };
  const near = within(URBAN_M);
  const urban = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let c = 0;
    near(i, () => c++);
    urban[i] = c >= URBAN_N ? 1 : 0;
  }
  // alone: no station within ALONE_M whose lines share no colour with this one's
  const cols: Set<string>[] = Array.from({ length: n }, () => new Set());
  for (const l of g.lines) for (const k of l.chain) cols[k].add(l.colour);
  const alone = new Uint8Array(n).fill(1), far = within(ALONE_M);
  for (let i = 0; i < n; i++) far(i, (j) => { if (alone[i] && ![...cols[j]].some((c) => cols[i].has(c))) alone[i] = 0; });
  // a city: urban stations within reach of each other, joined (union-find)
  const up = Int32Array.from({ length: n }, (_, i) => i);
  const root = (i: number): number => {
    while (up[i] !== i) (up[i] = up[up[i]]), (i = up[i]);
    return i;
  };
  for (let i = 0; i < n; i++) if (urban[i]) near(i, (j) => { if (urban[j]) up[root(i)] = root(j); });
  const city = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) if (urban[i]) city[i] = root(i);
  const deg = new Uint16Array(n);
  for (let e = 0; e < m; e++) deg[g.ea[e]]++, deg[g.eb[e]]++;
  const lonely = new Uint8Array(m);
  for (let e = 0; e < m; e++) {
    const c0 = g.lines[g.on[g.first[e]]]?.colour;
    let one = true;
    for (let k = g.first[e] + 1; k < g.first[e + 1]; k++) if (g.lines[g.on[k]].colour !== c0) one = false;
    const a = city[g.ea[e]], b = city[g.eb[e]];
    lonely[e] = one && deg[g.ea[e]] <= 2 && deg[g.eb[e]] <= 2 && (a < 0 || b < 0 || a !== b) && (alone[g.ea[e]] || alone[g.eb[e]]) ? 1 : 0;
  }
  return { city, lonely };
}

/** Stations that share no edge and lie within SIDE_HOPS of either one's mean real hop of each other. Hops
 * are evened per line, so a line of short hops beside one of long hops outruns it, and with nothing
 * between them a station can end up on the far side of its neighbour (T-2290: Fleetwood east of Port Chester). */
function sidePairs(g: DiagramGraph, adj: Set<number>): Float64Array {
  const n = g.key.length, m = g.ea.length;
  const reach = new Float64Array(n), cnt = new Float64Array(n), len = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e];
    len[e] = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a]);
    (reach[a] += len[e]), (reach[b] += len[e]), cnt[a]++, cnt[b]++;
  }
  for (let i = 0; i < n; i++) reach[i] = cnt[i] ? (SIDE_HOPS * reach[i]) / cnt[i] : 0;
  const cell = Math.max(1, 2 * median(len));
  const grid = new Map<number, number[]>();
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(g.x[i] / cell), Math.floor(g.y[i] / cell));
    const v = grid.get(k);
    if (v) v.push(i);
    else grid.set(k, [i]);
  }
  const out: number[] = [], had = new Set<number>();
  for (let i = 0; i < n; i++) {
    const r = reach[i], rc = Math.ceil(r / cell), cx = Math.floor(g.x[i] / cell), cy = Math.floor(g.y[i] / cell);
    for (let gx = cx - rc; gx <= cx + rc; gx++) for (let gy = cy - rc; gy <= cy + rc; gy++) {
      const v = grid.get(key(gx, gy));
      if (!v) continue;
      for (const j of v) {
        if (j === i) continue;
        const a = Math.min(i, j), b = Math.max(i, j), dx = g.x[b] - g.x[a], dy = g.y[b] - g.y[a], d = hyp(dx, dy);
        if (!(d > 0) || d >= r || had.has(a * n + b) || adj.has(a * n + b)) continue;
        had.add(a * n + b);
        out.push(a, b, dx / d, dy / d);
      }
    }
  }
  return Float64Array.from(out);
}

/** each edge's octilinear direction: at each node its edges take distinct directions nearest their
 * angles, the surest first; where the two ends disagree, the end deviating less wins. An edge's angle is the
 * mean of its own and those of the hops parallel to it on the ground, so they choose alike (T-2520). */
function chooseDirs(P: Prep, pos: Float64Array): Int8Array {
  const m = P.ea.length, ang = new Float64Array(m);
  const ux = new Float64Array(m), uy = new Float64Array(m);
  for (let e = 0; e < m; e++) {
    const dx = pos[P.eb[e] * 2] - pos[P.ea[e] * 2], dy = pos[P.eb[e] * 2 + 1] - pos[P.ea[e] * 2 + 1], l = hyp(dx, dy);
    if (l > 0) (ux[e] = dx / l), (uy[e] = dy / l);
  }
  const sx = Float64Array.from(ux), sy = Float64Array.from(uy);
  for (let k = 0; k < P.par.length; k += 3) {
    const a = P.par[k], b = P.par[k + 1], s = P.par[k + 2];
    (sx[a] += s * ux[b]), (sy[a] += s * uy[b]), (sx[b] += s * ux[a]), (sy[b] += s * uy[a]);
  }
  for (let e = 0; e < m; e++) ang[e] = atan2(sy[e], sx[e]);
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
        const d = hyp(pos[j * 2] - pos[i * 2], pos[j * 2 + 1] - pos[i * 2 + 1]);
        if (d < (P.dmin + P.hw[i] + P.hw[j]) * over) out.push(i, j);
      }
    }
  }
}

/** the ends of edges e1 and e2 away from v, and the real turn from e1 into e2 at v: [u, w, cos, sin] */
function throughOf(P: Prep, v: number, e1: number, e2: number): [number, number, number, number] {
  const u = P.ea[e1] === v ? P.eb[e1] : P.ea[e1], w = P.ea[e2] === v ? P.eb[e2] : P.ea[e2];
  // real unit directions u to v and v to w
  const s1 = P.ea[e1] === u ? 1 : -1, s2 = P.ea[e2] === v ? 1 : -1;
  const ix = s1 * P.gdx[e1], iy = s1 * P.gdy[e1], ox = s2 * P.gdx[e2], oy = s2 * P.gdy[e2];
  return [u, w, ix * ox + iy * oy, ix * oy - iy * ox];
}

/** a line from edge e1 through v into e2 keeps its real turn there: (w - v) / L2 = R (v - u) / L1 */
function turnTerm(T: Terms, P: Prep, v: number, e1: number, e2: number, wt: number): void {
  const [u, w, c, s] = throughOf(P, v, e1, e2), L1 = P.L[e1], L2 = P.L[e2], f = (L1 + L2) / 2;
  T.add([2 * w, 2 * v, 2 * v, 2 * v + 1, 2 * u, 2 * u + 1], [f / L2, -f / L2, (-c * f) / L1, (s * f) / L1, (c * f) / L1, (-s * f) / L1], 0, wt);
  T.add([2 * w + 1, 2 * v + 1, 2 * v, 2 * v + 1, 2 * u, 2 * u + 1], [f / L2, -f / L2, (-s * f) / L1, (-c * f) / L1, (s * f) / L1, (c * f) / L1], 0, wt);
}

/** rounds of least squares from `start`: option A (real directions) or B (octilinear); edges marked in `skip`
 * left out */
function lsq(P: Prep, octi: boolean, rounds: number, start: Float64Array, wProx: number, wGeo: number, wOct: number, iters: number, tick?: (done: number) => void, skip?: Uint8Array): Float64Array {
  let pos: Float64Array = Float64Array.from(start);
  const pairs: number[] = [];
  const I4 = [0, 0, 0, 0], C4 = [0, 0, 0, 0];
  for (let r = 0; r < rounds; r++) {
    tick?.(r / rounds);
    const T = new Terms(2 * P.n);
    const K = octi ? chooseDirs(P, pos) : null;
    for (let e = 0; e < P.ea.length; e++) {
      if (skip?.[e]) continue;
      const a = P.ea[e], b = P.eb[e];
      const dx = K ? DIRS[K[e]][0] : P.gdx[e], dy = K ? DIRS[K[e]][1] : P.gdy[e];
      I4[0] = 2 * b, I4[1] = 2 * b + 1, I4[2] = 2 * a, I4[3] = 2 * a + 1;
      C4[0] = dx, C4[1] = dy, C4[2] = -dx, C4[3] = -dy;
      T.add(I4, C4, P.L[e], W_LEN);
      C4[0] = -dy, C4[1] = dx, C4[2] = dy, C4[3] = -dx;
      T.add(I4, C4, 0, octi ? wOct : 1);
    }
    if (octi) {
      // a line through a junction keeps its real turn there, so no direction chosen at either end folds it back (T-2243)
      const m = P.ea.length;
      for (let v = 0; v < P.n; v++) {
        if (P.nbr[v].length === 2) continue;
        for (const k of P.through[v]) {
          const e1 = Math.floor(k / m), e2 = k % m;
          if (e1 < e2) turnTerm(T, P, v, e1, e2, W_TURN);
        }
      }
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
    // near stations that share no edge on the side of each other they are in reality (T-2290)
    const S = P.side;
    for (let k = 0; k < S.length; k += 4) {
      const a = S[k], b = S[k + 1], ux = S[k + 2], uy = S[k + 3], want = P.dmin + P.hw[a] + P.hw[b];
      if ((pos[b * 2] - pos[a * 2]) * ux + (pos[b * 2 + 1] - pos[a * 2 + 1]) * uy < want) T.add([2 * b, 2 * b + 1, 2 * a, 2 * a + 1], [ux, uy, -ux, -uy], want, W_SIDE);
      if (!octi && r >= rounds - ACROSS_ROUNDS) T.add([2 * b, 2 * b + 1, 2 * a, 2 * a + 1], [-uy, ux, uy, -ux], 0, W_SIDE_ACROSS);
    }
    // crossings on the ground held (T-2534): in A on the longer hop's real way, in B on its direction and near the
    // place, with the stations near it on their side
    for (let k = 0; k < P.cross.length; k += 4) {
      const e = P.cross[k], dx = K ? DIRS[K[e]][0] : P.gdx[e], dy = K ? DIRS[K[e]][1] : P.gdy[e];
      tieTerms(T, P.ea, P.eb, P.cross, k, -dy, dx, K ? W_CROSS_B : W_CROSS_A, K ? W_CROSS_PT_B : 0, (v) => v);
    }
    if (K) crossSideTerms(T, P, pos, (e) => [DIRS[K[e]][0], DIRS[K[e]][1]], W_SIDE);
    closePairs(P, pos, pairs, B_NEAR);
    for (let k = 0; k < pairs.length; k += 2) {
      const a = pairs[k], b = pairs[k + 1];
      let ux = pos[b * 2] - pos[a * 2], uy = pos[b * 2 + 1] - pos[a * 2 + 1];
      const dl = hyp(ux, uy);
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
  if (l2 < 1e-12) return hyp(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
  return hyp(px - ax - dx * t, py - ay - dy * t);
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
function segments(g: DiagramGraph, extra?: Uint8Array): number[][] {
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
    if (ns.length !== 2 || ends.has(v) || sig[ek(v, ns[0])] !== sig[ek(v, ns[1])] || extra?.[v]) key[v] = 1;
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

function route(g: DiagramGraph, P: Prep, bx: Float64Array, by: Float64Array, prev: Float64Array | null, tick?: (done: number) => void): Routed {
  const n = g.key.length;
  const hops: number[] = [];
  for (let e = 0; e < g.ea.length; e++) hops.push(hyp(bx[g.eb[e]] - bx[g.ea[e]], by[g.eb[e]] - by[g.ea[e]]));
  const tol = TOL_SHARE * median(hops);
  const raw: number[][] = [];
  for (const s of segments(g, P.keyNodes)) {
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
  // it, every hop keeps its length along its way, stations that share no edge are pushed apart, and the
  // rest stays near B. Pairs still too close once drawn are pushed in the next pass. No line may fold
  // back at a station where it turns less than a right angle in reality (T-2243).
  let q: Float64Array = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) (q[i * 2] = bx[i]), (q[i * 2 + 1] = by[i]);
  const q0 = Float64Array.from(q);
  const edgeOf = new Map<number, number>();
  for (let e = 0; e < P.ea.length; e++) edgeOf.set(P.ea[e] * n + P.eb[e], e);
  const links = lineLinks(g, pieces0);
  /** each hop held its length along its direction (B's, or its snapped piece's), the way the real line
   * runs: the direction terms alone are met by a hop of no length or one turned back, which folds a
   * line over itself at a station (T-2243) */
  const hold = (T: Terms, u: number, w: number, dx: number, dy: number) => {
    const e = edgeOf.get(Math.min(u, w) * n + Math.max(u, w))!, along = dx * (q0[w * 2] - q0[u * 2]) + dy * (q0[w * 2 + 1] - q0[u * 2 + 1]);
    T.add([2 * w, 2 * w + 1, 2 * u, 2 * u + 1], [dx, dy, -dx, -dy], Math.max(along, HOLD_MIN * P.L[e]), W_HOLD);
  };
  const realDot = (u: number, w: number, dx: number, dy: number) => dx * (g.x[w] - g.x[u]) + dy * (g.y[w] - g.y[u]);
  /** a hop's way in B, or its real way where B turned it more than a right angle */
  const holdB = (T: Terms, u: number, w: number) => {
    let dx = q0[w * 2] - q0[u * 2], dy = q0[w * 2 + 1] - q0[u * 2 + 1];
    if (realDot(u, w, dx, dy) < 0) (dx = g.x[w] - g.x[u]), (dy = g.y[w] - g.y[u]);
    const l = hyp(dx, dy);
    if (l > 1e-9) hold(T, u, w, dx / l, dy / l);
  };
  /** which pieces snap and which way (first to last, the way the real line runs): those near a 45 degree
   * direction after B, or now */
  const dev = (k: number, Q: Float64Array) => {
    const p = pieces0[k], a = p[0], b = p[p.length - 1];
    return a === b ? 90 : octiDev(Q[b * 2] - Q[a * 2], Q[b * 2 + 1] - Q[a * 2 + 1]);
  };
  const snapDirs = () => {
    const out = pieces0.map((p, k): [number, number] | null => {
      const Q = dev(k, q0) < SNAP_DEG ? q0 : dev(k, q) < SNAP_DEG ? q : null;
      if (!Q) return null;
      const a = p[0], b = p[p.length - 1], [dx, dy] = nearestDir(Q[b * 2] - Q[a * 2], Q[b * 2 + 1] - Q[a * 2 + 1]);
      return realDot(a, b, dx, dy) < 0 ? [-dx, -dy] : [dx, dy];
    });
    // two snapped pieces leaving a node within 45 degrees of each other fold a line that runs from one
    // to the other: the one further from its direction after B is left to the routing, which bends it
    for (const [ke, list] of links) for (const o of list) {
      const da = out[ke >> 1], db = out[o >> 1];
      if (!da || !db) continue;
      const sa = ke & 1 ? -1 : 1, sb = o & 1 ? -1 : 1;
      if (sa * sb * (da[0] * db[0] + da[1] * db[1]) >= 0.6) out[dev(ke >> 1, q0) > dev(o >> 1, q0) ? ke >> 1 : o >> 1] = null;
    }
    return out;
  };
  // lines turned at a station more than 45 degrees past their real turn, by B or by any round so far, are
  // drawn back toward it (T-2255: the snap itself folded lines B had turned less)
  const bent = new Set<number>(), m = P.ea.length;
  const findBent = (Q: Float64Array) => {
    for (let v = 0; v < n; v++) for (const k of P.through[v]) {
      const e1 = Math.floor(k / m), e2 = k % m;
      if (e1 > e2) continue;
      const [u, w, c, s] = throughOf(P, v, e1, e2);
      const ax = Q[v * 2] - Q[u * 2], ay = Q[v * 2 + 1] - Q[u * 2 + 1], bx2 = Q[w * 2] - Q[v * 2], by2 = Q[w * 2 + 1] - Q[v * 2 + 1];
      if (Math.abs(atan2(ax * by2 - ay * bx2, ax * bx2 + ay * by2)) - Math.abs(atan2(s, c)) > Math.PI / 4) bent.add(v * m * m + k);
    }
  };
  findBent(q0);
  const pushed = new Set<number>();
  const pairs: number[] = [];
  let drawn: Routed | null = null;
  for (let pass = 0; pass < SNAP_PASSES; pass++) {
    for (let it = 0; it < SNAP_ROUNDS; it++) {
      const T = new Terms(q.length);
      const snapDir = snapDirs();
      for (const [k, p] of pieces0.entries()) {
        const d = snapDir[k];
        if (!d) {
          for (let i = 1; i < p.length; i++) holdB(T, p[i - 1], p[i]);
          continue;
        }
        const [dx, dy] = d;
        for (let i = 1; i < p.length; i++) {
          T.add([2 * p[i], 2 * p[i] + 1, 2 * p[i - 1], 2 * p[i - 1] + 1], [-dy, dx, dy, -dx], 0, W_SNAP);
          hold(T, p[i - 1], p[i], dx, dy);
        }
      }
      if (it) findBent(q);
      for (const b of bent) {
        const v = Math.floor(b / (m * m)), k = b % (m * m);
        turnTerm(T, P, v, Math.floor(k / m), k % m, W_SNAP_TURN);
      }
      for (let i = 0; i < q.length; i++) T.add([i], [1], q0[i], 1);
      // crossings on the ground held, along each longer hop's way after B (T-2534)
      const wayB = (e: number): [number, number] => {
        const dx = q0[P.eb[e] * 2] - q0[P.ea[e] * 2], dy = q0[P.eb[e] * 2 + 1] - q0[P.ea[e] * 2 + 1], l = hyp(dx, dy) || 1;
        return [dx / l, dy / l];
      };
      for (let k = 0; k < P.cross.length; k += 4) {
        const [dx, dy] = wayB(P.cross[k]);
        tieTerms(T, P.ea, P.eb, P.cross, k, -dy, dx, W_CROSS_SNAP, W_CROSS_PT_SNAP, (v) => v);
      }
      crossSideTerms(T, P, q, wayB, W_SIDE);
      closePairs(P, q, pairs);
      for (let k = 0; k < pairs.length; k += 2) pushed.add(pairs[k] * n + pairs[k + 1]);
      for (const pk of pushed) {
        const a = Math.floor(pk / n), b = pk % n;
        let ux = q[b * 2] - q[a * 2], uy = q[b * 2 + 1] - q[a * 2 + 1];
        const dl = hyp(ux, uy);
        if (dl > 1e-6) (ux /= dl), (uy /= dl);
        else (ux = 1), (uy = 0);
        const want = (P.dmin + P.hw[a] + P.hw[b]) * PUSH_OVER;
        if (dl >= want) continue;
        T.add([2 * b, 2 * b + 1, 2 * a, 2 * a + 1], [ux, uy, -ux, -uy], want, W_SNAP_PUSH);
      }
      q = T.solve(q, SNAP_ITERS, 1e-9, 1e-9);
    }
    drawn = draw(pieces0, q, n, undefined, links);
    closePairs(P, interleave(drawn.x, drawn.y), pairs);
    if (!pairs.length) break;
    for (let k = 0; k < pairs.length; k += 2) pushed.add(pairs[k] * n + pairs[k + 1]);
  }
  // A pass that folds a line the drawing it started from did not (a bent piece's leg shrunk to nothing, so it
  // leaves its end another way) is done again with those bent pieces' empty legs held (T-2255).
  const keepLegs = new Set<number>();
  for (let pass = 0; pass < SEP_PASSES; pass++) {
    tick?.((pass + 1) / (SEP_PASSES + 1));
    const before = foldedEnds(drawn!, links, 1e-3 * P.pu);
    for (let tries = 0; ; tries++) {
      const s = separate(g, P, pieces0, drawn!, bx, by, prev, links, keepLegs);
      const next = draw(pieces0, s.q, n, s.bends);
      let grew = false;
      if (tries < 2) for (const [ke, o] of foldedEnds(next, links, 1e-3 * P.pu)) {
        if (before.has(ke)) continue;
        for (const k of [ke >> 1, o >> 1]) if (drawn!.bases[k].line.length === 6 && !keepLegs.has(k)) keepLegs.add(k), (grew = true);
      }
      if (!grew) {
        drawn = next;
        break;
      }
    }
  }
  return drawn!;
}

/** per piece end whose port folds a line with a linked piece end's, that end (legs up to `tiny` long left out) */
function foldedEnds(r: Routed, links: Links, tiny: number): Map<number, number> {
  const out = new Map<number, number>();
  for (const [ke, list] of links) for (const o of list) {
    const la = r.bases[ke >> 1].line, lb = r.bases[o >> 1].line;
    if (la.length < 4 || lb.length < 4) continue;
    const d = Math.abs(portOfLine(la, ke & 1, tiny) - portOfLine(lb, o & 1, tiny)) % 8;
    if (Math.min(d, 8 - d) <= 1) out.set(ke, o);
  }
  return out;
}

/** the port (0 to 7, 45 degree steps from east) a drawn piece leaves its first (0) or last (1) end by: its
 * first leg longer than `tiny` from that end */
function portOfLine(l: number[], end: number, tiny: number): number {
  const m = l.length;
  let dx = 0, dy = 0;
  for (let j = 0; j + 3 < m && Math.abs(dx) + Math.abs(dy) < tiny; j += 2) {
    dx = end ? l[m - 4 - j] - l[m - 2 - j] : l[j + 2] - l[j];
    dy = end ? l[m - 3 - j] - l[m - 1 - j] : l[j + 3] - l[j + 1];
  }
  return ((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
}

/** The separation (T-2205; notes/T-2205.md): the drawing as drawn, moved in terms that keep it
 * octilinear by construction: each piece end free in the plane, each station a distance along its
 * piece's leg from that leg's end, a bent piece's legs meeting where their directions say. Linear terms:
 * straight pieces stay straight, stations keep their order and gaps along their legs, and stations keep
 * their spacing from each other and from other pieces' legs (linearised about the current drawing, redone
 * each round). A first layout meets them by sweeps of projections; a rebuild (`prev`, the last drawing's
 * places) by Hildreth's method from the last drawing, which finds the drawing nearest it, each term's
 * pull capped, so an edit moves little. Returns the places to draw from, and each bent piece's bend. */
function separate(g: DiagramGraph, P: Prep, raw: number[][], r: Routed, bx: Float64Array, by: Float64Array, prev: Float64Array | null, links?: Links, keepLegs?: Set<number>): { q: Float64Array; bends: Map<number, [number, number]> } {
  const n = P.n;
  // the variables: x, y of each piece end (and of stations on a piece with no length), then one
  // distance per station along its leg
  const xyVar = new Int32Array(n).fill(-1), sVar = new Int32Array(n).fill(-1);
  let nv = 0;
  const ok = raw.map((p, k) => {
    const L = r.bases[k].line;
    return p[0] !== p[p.length - 1] && hyp(L[L.length - 2] - L[0], L[L.length - 1] - L[1]) > 1e-9;
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
  interface PieceSep { k: number; a: number; z: number; d0: [number, number]; d1: [number, number] | null; leg0: number[]; leg1: number[]; hw: number; held: [number, number] }
  const ps: PieceSep[] = [];
  const pieceOf = new Int32Array(n).fill(-1);
  const x0: number[] = [];
  // Two pieces leaving a node the same way are drawn over each other and no push along their legs
  // parts them: all but one straight piece among them gets a short first leg 45 degrees off, toward the
  // side its stations lay on after B (or the other, or none, where that would fold a line back), which
  // the spacing then lengthens.
  const port = (dx: number, dy: number) => ((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
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
  /** whether leaving piece end `ke`'s node by port `pp` folds back a line that runs on through it */
  const folds = (ke: number, pp: number) => (links?.get(ke) ?? []).some((o) => {
    const t = turned.get(o >> 1), po = t && t[0] === (o & 1) ? t[1] : portOf(o >> 1, (o & 1) as 0 | 1), d = Math.abs(pp - po) % 8;
    return Math.min(d, 8 - d) <= 1;
  });
  for (const [key, list] of portsAt) {
    if (list.length < 2) continue;
    const node = Math.floor(key / 8), straight = list.filter((ke) => r.bases[ke >> 1].line.length === 4 && !turned.has(ke >> 1));
    if (!straight.length) continue;
    const dir = DIRS[key % 8];
    /** the port a piece end turns to: toward its stations' side, else the other, one that folds no line where it can */
    const turnTo = (ke: number) => {
      const p = raw[ke >> 1];
      let side = 0;
      for (let i = 1; i < p.length - 1; i++) side += dir[0] * (by[p[i]] - by[node]) - dir[1] * (bx[p[i]] - bx[node]);
      const pref = side >= 0 ? 1 : 7;
      const frees = [pref, 8 - pref].map((s) => (key % 8 + s) % 8).filter((pp) => !portsAt.has(node * 8 + pp));
      return frees.find((pp) => !folds(ke, pp));
    };
    // the one kept (where all are straight): one that cannot turn without folding a line, else the longest
    const longest = straight.reduce((a, b) => (raw[b >> 1].length > raw[a >> 1].length ? b : a));
    const keep = list.length > straight.length ? -1 : straight.find((ke) => { const t = turnTo(ke); return t === undefined || folds(ke, t); }) ?? longest;
    for (const ke of straight) {
      if (ke === keep) continue;
      const free = turnTo(ke);
      if (free === undefined) continue;
      portsAt.set(node * 8 + free, [ke]);
      turned.set(ke >> 1, [(ke & 1) as 0 | 1, free]);
    }
  }
  raw.forEach((p, k) => {
    if (!ok[k]) return;
    const b = r.bases[k], L = b.line, a = p[0], z = p[p.length - 1];
    const e = edgeOf.get(Math.min(p[0], p[1]) * n + Math.max(p[0], p[1]));
    let hw = P.pu;
    if (e !== undefined) hw = ((strokesOn(g, e) - 1) / 2) * P.pu + P.pu;
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
    // a leg with no stations may shrink to nothing, unless the piece would then leave its end another way
    // that folds a line back there
    const held: [number, number] = [0, 0];
    if (L.length === 6 && !tn && keepLegs?.has(k)) (held[0] = LEG_MIN * P.pu), (held[1] = LEG_MIN * P.pu);
    else if (L.length === 6 && !tn) {
      if (folds(k * 2, port(d1![0], d1![1])) && !folds(k * 2, portOf(k, 0))) held[0] = LEG_MIN * P.pu;
      if (folds(k * 2 + 1, port(-d0[0], -d0[1])) && !folds(k * 2 + 1, portOf(k, 1))) held[1] = LEG_MIN * P.pu;
    }
    ps.push({ k, a, z, d0, d1, leg0, leg1, hw, held });
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
      // a leg with no stations keeps a little length: at none the piece would leave its end another way
      // and could fold a line back there (T-2243)
      if (pc.d1) (putLeg(pc, which, 1), emit(fixed, last >= 0 ? 0 : pc.held[which], false));
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
        const dl = hyp(ux, uy);
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

/** per piece end (piece * 2 + end), the ends of the pieces a line runs on to through that node, where
 * the real line turns less than a right angle there */
type Links = Map<number, number[]>;

function lineLinks(g: DiagramGraph, raw: number[][]): Links {
  const n = g.key.length;
  // a hop arriving at a piece's end: (from * n + to) to that end
  const arrive = new Map<number, number>();
  raw.forEach((p, k) => {
    const m = p.length;
    arrive.set(p[m - 2] * n + p[m - 1], k * 2 + 1);
    arrive.set(p[1] * n + p[0], k * 2);
  });
  const links: Links = new Map();
  const link = (a: number, b: number) => {
    const l = links.get(a);
    if (!l) links.set(a, [b]);
    else if (!l.includes(b)) l.push(b);
  };
  for (const l of g.lines) {
    const c = l.chain;
    for (let i = 1; i + 1 < c.length; i++) {
      const u = c[i - 1], v = c[i], w = c[i + 1];
      const ea = arrive.get(u * n + v), eb = arrive.get(w * n + v);
      if (u === w || ea === undefined || eb === undefined || ea === eb) continue;
      if ((g.x[v] - g.x[u]) * (g.x[w] - g.x[v]) + (g.y[v] - g.y[u]) * (g.y[w] - g.y[v]) <= 0) continue;
      link(ea, eb);
      link(eb, ea);
    }
  }
  return links;
}

/** each piece straight or with one bend between its snapped ends, its stations where they project
 * onto it, in order and at least half an even share of it apart. A bent piece takes the legs that
 * fold no line back at its ends and leave its ends through ports no other piece takes, where it can,
 * then those nearest its stations; after the separation (`bends`), the legs it was solved with. */
function draw(raw: number[][], q: Float64Array, n: number, bends?: Map<number, [number, number]>, links?: Links): Routed {
  const x = new Float64Array(n), y = new Float64Array(n);
  for (const p of raw) for (const k of [p[0], p[p.length - 1]]) (x[k] = q[k * 2]), (y[k] = q[k * 2 + 1]);
  const pieces: Piece[] = [], bases: Base[] = [];
  let straight = 0, bent = 0;
  // each piece leaves a node in one of eight directions; two pieces leaving it the same way would be
  // drawn over each other
  const port = (dx: number, dy: number) => ((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8;
  const portAt = (l: number[], end: number) => {
    const m = l.length;
    return end ? port(l[m - 4] - l[m - 2], l[m - 3] - l[m - 1]) : port(l[2] - l[0], l[3] - l[1]);
  };
  const isStraight = (p: number[]) => {
    const vx = x[p[p.length - 1]] - x[p[0]], vy = y[p[p.length - 1]] - y[p[0]];
    return hyp(vx, vy) >= 1e-9 && octiDev(vx, vy) <= (bends ? STRAIGHT_DEG : ROUTE_STRAIGHT_DEG);
  };
  const lines: number[][] = [], options: ([number[], number[]] | null)[] = [], inner: number[][] = [];
  /** per node * 8 + port, how many pieces leave the node that way */
  const taken = new Map<number, number>();
  const take = (l: number[], p: number[], f: number) => {
    for (const end of [0, 1]) {
      const t = p[end ? p.length - 1 : 0] * 8 + portAt(l, end);
      taken.set(t, (taken.get(t) ?? 0) + f);
    }
  };
  for (const [pi, p] of raw.entries()) {
    const ax = x[p[0]], ay = y[p[0]], cx = x[p[p.length - 1]], cy = y[p[p.length - 1]];
    const vx = cx - ax, vy = cy - ay;
    options.push(null);
    inner.push([]);
    if (hyp(vx, vy) < 1e-9) lines.push([ax, ay, cx, cy]);
    else if (isStraight(p)) {
      lines.push([ax, ay, cx, cy]);
      take(lines[pi], p, 1);
      straight++;
    } else {
      for (let i = 1; i < p.length - 1; i++) inner[pi].push(q[p[i] * 2], q[p[i] * 2 + 1]);
      const [l1, l2] = twoLegs(ax, ay, cx, cy);
      const bd = bends?.get(pi);
      lines.push(bd ? (hyp(l2[2] - bd[0], l2[3] - bd[1]) < hyp(l1[2] - bd[0], l1[3] - bd[1]) ? l2 : l1) : []);
      if (!bd) options[pi] = [l1, l2];
      bent++;
    }
  }
  /** lines folded back at the piece's ends with these legs, against the linked pieces' legs so far */
  const folds = (pi: number, l: number[]) => {
    let f = 0;
    for (const end of [0, 1]) for (const o of links?.get(pi * 2 + end) ?? []) {
      const lo = lines[o >> 1];
      if (lo.length < 4) continue;
      const d = Math.abs(portAt(l, end) - portAt(lo, o & 1)) % 8;
      if (Math.min(d, 8 - d) <= 1) f++;
    }
    return f;
  };
  // a piece only near straight that would fold a line against a straight neighbour is bent instead
  if (!bends) for (const [pi, p] of raw.entries()) {
    const l = lines[pi], vx = l[2] - l[0], vy = l[3] - l[1];
    if (l.length !== 4 || hyp(vx, vy) < 1e-9 || octiDev(vx, vy) <= STRAIGHT_DEG || !folds(pi, l)) continue;
    take(l, p, -1);
    for (let i = 1; i < p.length - 1; i++) inner[pi].push(q[p[i] * 2], q[p[i] * 2 + 1]);
    options[pi] = twoLegs(l[0], l[1], l[2], l[3]);
    lines[pi] = [];
    straight--, bent++;
  }
  const choose = (pi: number) => {
    const p = raw[pi], [l1, l2] = options[pi]!;
    const len = hyp(l1[4] - l1[0], l1[5] - l1[1]);
    const cost = (l: number[]) => {
      let c = (inner[pi].length ? polyDev(inner[pi], l) : 0) + folds(pi, l) * FOLD_COST * len;
      for (const end of [0, 1]) if (taken.get(p[end ? p.length - 1 : 0] * 8 + portAt(l, end))) c += PORT_COST * len;
      return c;
    };
    return cost(l2) < cost(l1) ? l2 : l1;
  };
  // in order, then again each against all the others' choices
  for (let pass = 0; pass < LEG_PASSES; pass++) for (const [pi, o] of options.entries()) {
    if (!o) continue;
    if (lines[pi].length) take(lines[pi], raw[pi], -1);
    lines[pi] = choose(pi);
    take(lines[pi], raw[pi], 1);
  }
  for (const [pi, p] of raw.entries()) {
    const line = lines[pi];
    const nl = line.length / 2;
    const Ls = [0];
    for (let i = 1; i < nl; i++) Ls.push(Ls[i - 1] + hyp(line[i * 2] - line[i * 2 - 2], line[i * 2 + 1] - line[i * 2 - 1]));
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
        const d = hyp(x0 + dx * t - px, y0 + dy * t - py);
        if (d < best) (best = d), (arc = Ls[j] + t * Math.sqrt(l2));
      }
      s.push(arc);
    }
    s.push(total);
    for (let i = 1; i < k; i++) s[i] = Math.max(s[i], s[i - 1] + gap);
    for (let i = k - 1; i >= 1; i--) s[i] = Math.min(s[i], s[i + 1] - gap);
    const pc: Piece = { nodes: Int32Array.from(p), verts: new Float64Array(0), at: new Int32Array(0), strokes: [] };
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
        if (hyp(ax - bx2, ay - by2) < want(a, b)) continue;
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

/** per piece, its strokes (one per line, but for lines of one colour and speed on the same track, which share
 * one) left to right looking from its first node to its last:
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
    return atan2(ax * byy - ay * bxx, ax * bxx + ay * byy);
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
    // lines share a stroke on the piece where they share one on each of its edges (graph.ts' `stroke`)
    const sig = new Map<number, string>();
    for (let k = g.first[e]; k < g.first[e + 1]; k++) sig.set(g.on[k], String(g.stroke[k]));
    for (let i = 2; i < p.length; i++) {
      const f = edgeOf.get(Math.min(p[i - 1], p[i]) * n + Math.max(p[i - 1], p[i]));
      if (f === undefined) continue;
      for (let k = g.first[f]; k < g.first[f + 1]; k++) {
        const s = sig.get(g.on[k]);
        if (s !== undefined) sig.set(g.on[k], s + "," + g.stroke[k]);
      }
    }
    const groups = new Map<string, number[]>();
    for (const [l, s] of sig) {
      const v = groups.get(s);
      if (v) v.push(l);
      else groups.set(s, [l]);
    }
    // each stroke's lines, and the one it is walked by: the longest chain
    const cols = [...groups.values()].map((v) => v.sort((a, b) => a - b)).sort((a, b) => (g.lines[a[0]].colour < g.lines[b[0]].colour ? -1 : g.lines[a[0]].colour > g.lines[b[0]].colour ? 1 : a[0] - b[0]));
    const rep = cols.map((v) => v.reduce((a, b) => (g.lines[b].chain.length > g.lines[a].chain.length ? b : a)));
    const m = cols.length;
    if (m < 2) {
      pc.strokes = cols;
      continue;
    }
    const cmp = (ia: number, ib: number) => {
      const la = rep[ia], lb = rep[ib];
      let v = walk(la, lb, p[p.length - 2], p[p.length - 1]);
      if (!v) v = -walk(la, lb, p[1], p[0]);
      return v || (ia < ib ? -1 : 1);
    };
    // by how many of the others each is left of, as lineBands does where partings go round a circle
    const wins = new Int32Array(m), c = new Int8Array(m * m);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
      const r = cmp(i, j);
      c[i * m + j] = r;
      c[j * m + i] = -r;
      if (r < 0) wins[i]++;
      else wins[j]++;
    }
    // ties in wins by the pairwise rule, by passes of swaps rather than a sort: where they go round in a
    // circle the comparison is no order, and engines' sorts then differ (T-2405)
    const ord = cols.map((_, i) => i).sort((i, j) => wins[j] - wins[i] || i - j);
    for (let pass = 0; pass < m; pass++) {
      let swapped = false;
      for (let t = 0; t + 1 < m; t++) {
        const i = ord[t], j = ord[t + 1];
        if (wins[i] === wins[j] && c[i * m + j] > 0) (ord[t] = j), (ord[t + 1] = i), (swapped = true);
      }
      if (!swapped) break;
    }
    pc.strokes = ord.map((i) => cols[i]);
  }
}

/** a layout's places by node key, for a rebuild: after B, and drawn (`lastPlaces`); and its hops between blocks */
export type LastPlaces = Map<number, [number, number] | [number, number, number, number]> & { blocks?: Set<string> };

export function lastPlaces(g: DiagramGraph, lay: DiagramLayout): LastPlaces {
  const m: LastPlaces = new Map();
  for (let i = 0; i < g.key.length; i++) m.set(g.key[i], [lay.bx[i], lay.by[i], lay.x[i], lay.y[i]]);
  if (lay.blocks.active) m.blocks = heldPairs(lay.blocks);
  return m;
}

/** The layout of a graph; `last` the last layout's places: a rebuild starts from its B positions,
 * and its separation keeps to the drawing nearest the last one. */
export function layoutDiagram(g: DiagramGraph, o: LayoutOptions = LAYOUT, last?: LastPlaces | null, progress?: (share: number) => void): DiagramLayout {
  const t0 = performance.now();
  const n = g.key.length;
  const P = prep(g, o);
  const hop = median(P.L);
  const blocks = findBlocks(g, P.city, P.cross);
  const tied = tiedBlocks(g, blocks, P.cross);
  // with two blocks or more, A lays out each city as if alone, the hops between them left to the blocks' stage
  const skip = blocks.active ? blockHops(g, blocks) : undefined;
  let nA = n;
  if (skip) {
    const size = new Int32Array(blocks.list.length);
    for (let i = 0; i < n; i++) if (blocks.of[i] >= 0) size[blocks.of[i]]++;
    nA = 0;
    for (const s of size) nA = Math.max(nA, s);
  }
  // a big network's A needs more rounds to draw its long stretches as short as their targets (T-2289), a
  // network of blocks as many as its biggest block; the loading bar keeps A's and B's shares of the time in step
  const roundsA = Math.round(Math.min(A_ROUNDS_MAX, Math.max(o.roundsA, o.roundsA + ((A_ROUNDS_MAX - o.roundsA) * (nA - A_ROUNDS_FROM)) / (A_ROUNDS_AT - A_ROUNDS_FROM))));
  const progA = (PROGRESS_B * PROGRESS_A * roundsA) / (PROGRESS_A * roundsA + (PROGRESS_B - PROGRESS_A) * o.roundsA);
  const geo = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) geo[i * 2] = g.x[i], (geo[i * 2 + 1] = g.y[i]);
  const step = (from: number, to: number) => progress && ((d: number) => progress(from + (to - from) * d));
  let start: Float64Array;
  let warm = false;
  if (last && last.size) {
    start = warmStart(g, P, last);
    warm = true;
  } else {
    let PA = P;
    if (skip) {
      const s: number[] = [];
      for (let k = 0; k < P.side.length; k += 4) if (blocks.of[P.side[k]] >= 0 && blocks.of[P.side[k]] === blocks.of[P.side[k + 1]]) s.push(P.side[k], P.side[k + 1], P.side[k + 2], P.side[k + 3]);
      // crossings within one block; those between blocks are the blocks' stage's
      const c: number[] = [];
      for (let k = 0; k < P.cross.length; k += 4) {
        const e1 = P.cross[k], e2 = P.cross[k + 1], b = blocks.of[P.ea[e1]];
        if (b >= 0 && !skip[e1] && !skip[e2] && blocks.of[P.ea[e2]] === b) c.push(e1, e2, P.cross[k + 2], P.cross[k + 3]);
      }
      PA = { ...P, side: Float64Array.from(s), cross: Float64Array.from(c) };
    }
    start = lsq(PA, false, roundsA, geo, 0.02, 0.001, 1, o.iters, step(0, progA), skip);
  }
  const t1 = performance.now();
  // the blocks' stage; a rebuild holds the hops between blocks it had as they are, and with no other places
  // blocks only where boxes crowd
  let mv: Float64Array | null = null;
  if (skip) {
    const held = warm ? last!.blocks ?? null : null;
    if (!held || !samePairs(blocks, held) || crowded(blocks, start, 0, tied)) {
      const linkTarget = (d: number) => {
        const l = o.L0 * pow(d / o.L0, o.alpha);
        return o.lonely < 1 ? Math.max(Math.min(l, LONELY_MIN * P.pu), l * o.lonely) : l;
      };
      mv = placeBlocks(g, blocks, start, P.L, P.gdx, P.gdy, P.pu, linkTarget, held, false, P.cross);
      for (let i = 0; i < n * 2; i++) start[i] += mv[i];
    }
    // the hops between blocks keep the length placed, so B and the drawing do not pull the cities together again;
    // a station between blocks on a long hop ends a segment, so the routing draws each long hop on its own and
    // cannot slide the stops along one long piece into a block
    P.keyNodes = new Uint8Array(n);
    for (let e = 0; e < P.ea.length; e++) if (skip[e]) {
      const a = P.ea[e], b = P.eb[e];
      P.L[e] = Math.max(P.L[e], hyp(start[b * 2] - start[a * 2], start[b * 2 + 1] - start[a * 2 + 1]));
      if (P.L[e] > LONG_HOP * hop) for (const v of [a, b]) if (blocks.of[v] < 0) P.keyNodes[v] = 1;
    }
  }
  const t1b = performance.now();
  // a rebuild starts near where B ended: fewer rounds, held closer to the last layout
  const B = warm ? lsq(P, true, WARM_ROUNDS, start, 0.1, W_GEO_WARM, 3, o.iters) : lsq(P, true, o.roundsB, start, 0.1, 0.01, 3, o.iters, step(progA, PROGRESS_B));
  // B and the drawing widen blocks (their bundles' room): boxes B brought within 3/4 of the gap are moved apart
  // to the gap again, each block and the hops between blocks as B left them
  if (skip && crowded(blocks, B, 0.75 * BLOCK_GAP * P.pu, tied)) {
    const apart = placeBlocks(g, blocks, B, P.L, P.gdx, P.gdy, P.pu, () => 0, heldPairs(blocks), true, P.cross);
    for (let i = 0; i < n * 2; i++) B[i] += apart[i];
    if (mv) for (let i = 0; i < n * 2; i++) mv[i] += apart[i];
    else mv = apart;
    for (let e = 0; e < P.ea.length; e++) if (skip[e]) {
      const a = P.ea[e], b = P.eb[e];
      P.L[e] = Math.max(P.L[e], hyp(B[b * 2] - B[a * 2], B[b * 2 + 1] - B[a * 2 + 1]));
    }
  }
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
      if (p && p.length === 4) (prev[i * 2] = p[2] + (mv ? mv[i * 2] : 0)), (prev[i * 2 + 1] = p[3] + (mv ? mv[i * 2 + 1] : 0)), known++;
    }
    if (known < PREV_SHARE * n) prev = null;
  }
  const r = route(g, P, bx, by, prev, step(PROGRESS_B, 1));
  spread(P, r);
  const t3 = performance.now();
  order(g, r.pieces, r.x, r.y);
  finishBlocks(g, blocks, r.x, r.y);
  const t4 = performance.now();
  return {
    x: r.x, y: r.y, pieces: r.pieces, pu: P.pu, hop, hw: P.hw, bx, by,
    ms: { a: warm ? 0 : t1 - t0, blocks: t1b - t1, b: t2 - t1b, route: t3 - t2, order: t4 - t3, total: t4 - t0 },
    straight: r.straight, bent: r.bent, blocks,
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
    num += hyp(pos[i * 2] - sx / c, pos[i * 2 + 1] - sy / c);
    den += hyp(g.x[i] - gx / c, g.y[i] - gy / c);
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
        const dx = g.x[i] - g.x[o], dy = g.y[i] - g.y[o], gl = Math.max(1, hyp(dx, dy)), s = P.L[e] / gl;
        ax += pos[o * 2] + dx * s;
        ay += pos[o * 2 + 1] + dy * s;
        k++;
      }
      if (k) (pos[i * 2] = ax / k), (pos[i * 2 + 1] = ay / k), (known[i] = 2), (grew = true);
    }
    for (let i = 0; i < n; i++) if (known[i] === 2) known[i] = 1;
    if (!grew) break;
  }
  // the rest: off the nearest placed node in reality, not off the middle of all of them, which in a network of
  // several cities lies between them (T-2417)
  const near = c ? nearestKnown(g, known) : null;
  for (let i = 0; i < n; i++) if (!known[i]) {
    const o = near ? near(i) : -1;
    pos[i * 2] = o >= 0 ? pos[o * 2] + (g.x[i] - g.x[o]) * scale : g.x[i];
    pos[i * 2 + 1] = o >= 0 ? pos[o * 2 + 1] + (g.y[i] - g.y[o]) * scale : g.y[i];
  }
  return pos;
}

/** per node, the nearest node in reality with `known` 1 (by every one: few nodes are placed this way) */
function nearestKnown(g: DiagramGraph, known: Uint8Array): (i: number) => number {
  const ids: number[] = [];
  for (let i = 0; i < g.key.length; i++) if (known[i] === 1) ids.push(i);
  return (i: number) => {
    let best = -1, bd = Infinity;
    for (const j of ids) {
      const d = (g.x[j] - g.x[i]) * (g.x[j] - g.x[i]) + (g.y[j] - g.y[i]) * (g.y[j] - g.y[i]);
      if (d < bd) (bd = d), (best = j);
    }
    return best;
  };
}
