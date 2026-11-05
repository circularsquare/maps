// The network diagram's city blocks (T-2407; notes/T-2407.md, design notes/T-2406.md): each city's stations are
// laid out as if alone and moved as one rigid block; one small solve between A and B places the blocks and the
// stations between them so that each hop between them gets its target along its real bearing, from the edge of its
// block's box, and boxes keep a gap. Blocks are cities by geography, so a piece of a city with no track to the rest
// (Staten Island's railway) rides with it; networks with no track between them are linked to the nearest, so they
// pack the same way. A rebuild moves each block by its move, and nothing when the blocks and the hops between them
// are as before.

import type { DiagramGraph } from "./graph";
import { Terms } from "./lsqr";
import { hyp } from "./fmath";

/** cities (layout.ts' cityParts) with stations within MERGE_M of each other are one block, if it has at least
 * MIN_BLOCK stations */
export const MERGE_M = 15000, MIN_BLOCK = 10;
/** pitches kept between two blocks' boxes, and between a box and a station between blocks */
export const BLOCK_GAP = 16;
/** boxes are kept apart as if this share bigger: the drawing after B widens a block by up to about a fifth */
const BOX_GROW = 0.2;
/** the solve: a hop's weight, a box kept apart, and the pull of each station to where it was, which only picks
 * one place where the hops leave a choice */
const W_HOP = 1, W_BOX = 10, W_FREE = 1e-3;
/** rounds (each moving boxes still overlapping), LSQR iterations in the first and in each later one */
const ROUNDS = 12, ITERS = 2000, ITERS_MORE = 400;
/** a hop leaves its block past the block's stations within this many gaps either side of its way */
const EXIT_BAND = 1;
/** two boxes once held apart stay held while nearer than this share of their spacing */
const HOLD = 1.05;
/** islands linked: at most this many stations of each network tried for the nearest pair */
const LINK_SAMPLE = 64;

export interface Block {
  /** its lowest node key, which names it across rebuilds */
  key: number;
  /** its stations' box as drawn, x0, y0, x1, y1 */
  box: [number, number, number, number];
  /** how far the stage moved it, layout units */
  move: [number, number];
  /** its stations' real places to drawn ones, by least squares: x' = a x + b y + c, y' = d x + e y + f, with
   * a to f in that order (for T-2169's water; notes/T-2407.md) */
  fit: [number, number, number, number, number, number];
}

export interface Blocks {
  /** per node, its block, or -1 for a station between blocks */
  of: Int32Array;
  /** per node, 1 for a station of its block's city (the box is theirs), 0 for a branch or piece taken in */
  core: Uint8Array;
  list: Block[];
  /** whether the stage places blocks: two or more of them */
  active: boolean;
  /** pairs of nodes linking networks with no track between them, nearest first */
  links: Int32Array;
  /** the hops between blocks (or touching a station between them) and the links, as pairs of node keys, lower
   * first: a rebuild holds those it had as they were drawn */
  pairs: Int32Array;
}

/** The blocks of a graph; `city` per node as layout.ts' cityParts gives it (its city's root node, or -1). */
export function findBlocks(g: DiagramGraph, city: Int32Array): Blocks {
  const n = g.key.length, m = g.ea.length;
  const up = Int32Array.from({ length: n }, (_, i) => i);
  const root = (i: number): number => {
    while (up[i] !== i) (up[i] = up[up[i]]), (i = up[i]);
    return i;
  };
  const join = (a: number, b: number) => {
    const ra = root(a), rb = root(b);
    if (ra !== rb) up[Math.max(ra, rb)] = Math.min(ra, rb);
  };
  // cities, and cities within MERGE_M of each other
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  const grid = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    if (city[i] < 0) continue;
    join(i, city[i]);
    const k = key(Math.floor(g.x[i] / MERGE_M), Math.floor(g.y[i] / MERGE_M)), v = grid.get(k);
    if (v) v.push(i);
    else grid.set(k, [i]);
  }
  const R2 = MERGE_M * MERGE_M;
  for (let i = 0; i < n; i++) {
    if (city[i] < 0) continue;
    const cx = Math.floor(g.x[i] / MERGE_M), cy = Math.floor(g.y[i] / MERGE_M);
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
      for (const j of grid.get(key(gx, gy)) ?? []) {
        if (j <= i || root(i) === root(j)) continue;
        const dx = g.x[j] - g.x[i], dy = g.y[j] - g.y[i];
        if (dx * dx + dy * dy < R2) join(i, j);
      }
    }
  }
  const size = new Int32Array(n);
  for (let i = 0; i < n; i++) if (city[i] >= 0) size[root(i)]++;
  const of = new Int32Array(n).fill(-1), core = new Uint8Array(n);
  const blockOfRoot = new Map<number, number>();
  let nb = 0;
  for (let i = 0; i < n; i++) {
    if (city[i] < 0 || size[root(i)] < MIN_BLOCK) continue;
    let b = blockOfRoot.get(root(i));
    if (b === undefined) blockOfRoot.set(root(i), (b = nb++));
    of[i] = b;
    core[i] = 1;
  }
  // stations outside the blocks, joined by their hops into groups: a group whose hops reach one block is taken into
  // it (a branch out of a city), one reaching none is taken into the block with a station nearest it within MERGE_M
  // (a piece of a city with no track to the rest: Staten Island's railway rides with New York), else it is a block
  // of its own (a small network alone); the rest stay between blocks
  const cityGrid = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    if (of[i] < 0) continue;
    const k = key(Math.floor(g.x[i] / MERGE_M), Math.floor(g.y[i] / MERGE_M)), v = cityGrid.get(k);
    if (v) v.push(i);
    else cityGrid.set(k, [i]);
  }
  up.forEach((_, i) => (up[i] = i));
  for (let e = 0; e < m; e++) if (of[g.ea[e]] < 0 && of[g.eb[e]] < 0) join(g.ea[e], g.eb[e]);
  const meets = new Map<number, number>(); // group root -> its one block, or -2 for several
  for (let e = 0; e < m; e++) {
    const a = g.ea[e], b = g.eb[e];
    if ((of[a] < 0) === (of[b] < 0)) continue;
    const r = root(of[a] < 0 ? a : b), blk = of[a] < 0 ? of[b] : of[a], had = meets.get(r);
    meets.set(r, had === undefined || had === blk ? blk : -2);
  }
  // per lone group: the nearest block station within MERGE_M, and its distance squared
  const near = new Map<number, [number, number]>();
  for (let i = 0; i < n; i++) {
    if (of[i] >= 0 || meets.has(root(i))) continue;
    const r = root(i), cx = Math.floor(g.x[i] / MERGE_M), cy = Math.floor(g.y[i] / MERGE_M);
    let best = near.get(r) ?? [-1, R2];
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
      for (const j of cityGrid.get(key(gx, gy)) ?? []) {
        const dx = g.x[j] - g.x[i], dy = g.y[j] - g.y[i], d = dx * dx + dy * dy;
        if (d < best[1]) best = [j, d];
      }
    }
    near.set(r, best);
  }
  const own = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    if (of[i] >= 0) continue;
    const r = root(i), mb = meets.get(r), nr = near.get(r);
    if (mb !== undefined) {
      if (mb >= 0) of[i] = mb;
    } else if (nr && nr[0] >= 0) of[i] = of[nr[0]];
    else {
      let b = own.get(r);
      if (b === undefined) own.set(r, (b = nb++));
      of[i] = b;
      core[i] = 1;
    }
  }
  const list: Block[] = Array.from({ length: nb }, () => ({ key: 0x7fffffff, box: [0, 0, 0, 0], move: [0, 0], fit: [1, 0, 0, 0, 1, 0] }));
  for (let i = 0; i < n; i++) if (of[i] >= 0) list[of[i]].key = Math.min(list[of[i]].key, g.key[i]);
  const active = nb >= 2;
  const links = active ? islandLinks(g, of, nb) : new Int32Array(0);
  const pairs: number[] = [];
  if (active) {
    for (let e = 0; e < m; e++) {
      const a = g.ea[e], b = g.eb[e];
      if (of[a] < 0 || of[a] !== of[b]) pairs.push(Math.min(g.key[a], g.key[b]), Math.max(g.key[a], g.key[b]));
    }
    for (let k = 0; k < links.length; k += 2) {
      const a = g.key[links[k]], b = g.key[links[k + 1]];
      pairs.push(Math.min(a, b), Math.max(a, b));
    }
  }
  return { of, core, list, active, links, pairs: Int32Array.from(pairs) };
}

/** a layout's hops between blocks and links, as `pairs` lists them, for the next rebuild */
export function heldPairs(B: Blocks): Set<string> {
  const s = new Set<string>();
  for (let k = 0; k < B.pairs.length; k += 2) s.add(`${B.pairs[k]},${B.pairs[k + 1]}`);
  return s;
}

/** whether `B`'s hops between blocks and links are the ones in `held` */
export function samePairs(B: Blocks, held: Set<string>): boolean {
  if (held.size * 2 !== B.pairs.length) return false;
  for (let k = 0; k < B.pairs.length; k += 2) if (!held.has(`${B.pairs[k]},${B.pairs[k + 1]}`)) return false;
  return true;
}

/** Links between the networks no track joins (counting a station between blocks with the blocks its hops reach):
 * the shortest real links that join them all (Kruskal), each between the nearest of up to LINK_SAMPLE stations a
 * network. */
function islandLinks(g: DiagramGraph, of: Int32Array, nb: number): Int32Array {
  const n = g.key.length, m = g.ea.length;
  // units: the blocks, then each station between blocks
  const unit = new Int32Array(n);
  let nu = nb;
  for (let i = 0; i < n; i++) unit[i] = of[i] >= 0 ? of[i] : nu++;
  const up = Int32Array.from({ length: nu }, (_, i) => i);
  const root = (i: number): number => {
    while (up[i] !== i) (up[i] = up[up[i]]), (i = up[i]);
    return i;
  };
  for (let e = 0; e < m; e++) {
    const ra = root(unit[g.ea[e]]), rb = root(unit[g.eb[e]]);
    if (ra !== rb) up[Math.max(ra, rb)] = Math.min(ra, rb);
  }
  const members = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = root(unit[i]), v = members.get(r);
    if (v) v.push(i);
    else members.set(r, [i]);
  }
  if (members.size < 2) return new Int32Array(0);
  const comps = [...members.values()].map((v) => {
    if (v.length <= LINK_SAMPLE) return v;
    const s: number[] = [];
    for (let k = 0; k < LINK_SAMPLE; k++) s.push(v[Math.floor((k * v.length) / LINK_SAMPLE)]);
    return s;
  });
  const cand: [number, number, number, number, number][] = [];
  for (let p = 0; p < comps.length; p++) for (let q = p + 1; q < comps.length; q++) {
    let best = Infinity, ba = -1, bb = -1;
    for (const a of comps[p]) for (const b of comps[q]) {
      const d = (g.x[b] - g.x[a]) * (g.x[b] - g.x[a]) + (g.y[b] - g.y[a]) * (g.y[b] - g.y[a]);
      if (d < best) (best = d), (ba = a), (bb = b);
    }
    cand.push([best, p, q, ba, bb]);
  }
  cand.sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
  const cu = Int32Array.from({ length: comps.length }, (_, i) => i);
  const croot = (i: number): number => {
    while (cu[i] !== i) (cu[i] = cu[cu[i]]), (i = cu[i]);
    return i;
  };
  const out: number[] = [];
  for (const [, p, q, a, b] of cand) {
    const rp = croot(p), rq = croot(q);
    if (rp === rq) continue;
    cu[Math.max(rp, rq)] = Math.min(rp, rq);
    out.push(a, b);
  }
  return Int32Array.from(out);
}

/** Whether, at places `pos` (x, y pairs), two blocks' boxes lie nearer than `margin`: a rebuild whose hops between
 * blocks are as before still places blocks where boxes overlap (a city grown into another), and after B where they
 * are nearer than half the gap. */
export function crowded(B: Blocks, pos: Float64Array, margin = 0): boolean {
  const box = blockBoxes(B, pos), nb = B.list.length;
  for (let b = 0; b < nb; b++) for (let c = b + 1; c < nb; c++) {
    if (box[c * 4] - box[b * 4 + 2] < margin && box[b * 4] - box[c * 4 + 2] < margin && box[c * 4 + 1] - box[b * 4 + 3] < margin && box[b * 4 + 1] - box[c * 4 + 3] < margin) return true;
  }
  return false;
}

/** each block's box at places `pos`: x0, y0, x1, y1 per block */
function blockBoxes(B: Blocks, pos: Float64Array): Float64Array {
  const nb = B.list.length, box = new Float64Array(nb * 4);
  for (let b = 0; b < nb; b++) (box[b * 4] = Infinity), (box[b * 4 + 1] = Infinity), (box[b * 4 + 2] = -Infinity), (box[b * 4 + 3] = -Infinity);
  for (let i = 0; i < B.of.length; i++) {
    const b = B.of[i];
    if (b < 0) continue;
    box[b * 4] = Math.min(box[b * 4], pos[i * 2]);
    box[b * 4 + 1] = Math.min(box[b * 4 + 1], pos[i * 2 + 1]);
    box[b * 4 + 2] = Math.max(box[b * 4 + 2], pos[i * 2]);
    box[b * 4 + 3] = Math.max(box[b * 4 + 3], pos[i * 2 + 1]);
  }
  return box;
}

/**
 * The stage: from places `pos` (x, y pairs; A's on a first layout, the last layout's on a rebuild), each block's
 * move and each station between blocks' move, x, y pairs per node, so that each hop between them (`L` its target,
 * `gdx`, `gdy` its real bearing) runs its target along its bearing from where it leaves each end's block plus half
 * the gap, each link `linkTarget` of its real length the same way, and boxes keep the gap. On a rebuild, the hops and links in `held` (the last layout's, `heldPairs`) are held as they are,
 * so only what changed moves anything, and boxes already within the gap of each other but not overlapping are
 * left so, unless `strict` (after B, which grows a block's box, every pair is given the gap again).
 */
export function placeBlocks(g: DiagramGraph, B: Blocks, pos: Float64Array, L: Float64Array, gdx: Float64Array, gdy: Float64Array, pu: number, linkTarget: (d: number) => number, held: Set<string> | null, strict = false): Float64Array {
  const n = g.key.length, nb = B.list.length, of = B.of;
  const unit = new Int32Array(n);
  let nu = nb;
  for (let i = 0; i < n; i++) unit[i] = of[i] >= 0 ? of[i] : nu++;
  const box = blockBoxes(B, pos), pad = (BLOCK_GAP * pu) / 2;
  const kept = (a: number, b: number) => held !== null && held.has(`${Math.min(g.key[a], g.key[b])},${Math.max(g.key[a], g.key[b])}`);
  const members: number[][] = Array.from({ length: nb }, () => []);
  for (let i = 0; i < n; i++) if (of[i] >= 0) members[of[i]].push(i);
  /** from station i along (ux, uy) past the farthest station of its block within EXIT_BAND gaps of that way, plus
   * half the gap; 0 between blocks. Past the block's stations that way rather than to its box's edge, so a hop out
   * of a sparse side of a sprawling block is not drawn across the box's empty corner. */
  const band = EXIT_BAND * BLOCK_GAP * pu;
  const exit = (i: number, ux: number, uy: number) => {
    const b = of[i];
    if (b < 0) return 0;
    const px = pos[i * 2], py = pos[i * 2 + 1];
    let far = 0;
    for (const j of members[b]) {
      const dx = pos[j * 2] - px, dy = pos[j * 2 + 1] - py, along = dx * ux + dy * uy;
      if (along > far && Math.abs(dx * uy - dy * ux) < band) far = along;
    }
    return far + pad;
  };
  // the hops and links: a, b, ux, uy, target; one held is wanted as it is
  const hops: number[] = [];
  const add = (a: number, b: number, ux: number, uy: number, t: number) => {
    if (kept(a, b)) {
      const dx = pos[b * 2] - pos[a * 2], dy = pos[b * 2 + 1] - pos[a * 2 + 1], d = hyp(dx, dy);
      if (d > 0) (ux = dx / d), (uy = dy / d), (t = d);
    }
    hops.push(a, b, ux, uy, t);
  };
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e], b = g.eb[e];
    if (unit[a] === unit[b]) continue;
    add(a, b, gdx[e], gdy[e], L[e] + exit(a, gdx[e], gdy[e]) + exit(b, -gdx[e], -gdy[e]));
  }
  for (let k = 0; k < B.links.length; k += 2) {
    const a = B.links[k], b = B.links[k + 1], dx = g.x[b] - g.x[a], dy = g.y[b] - g.y[a], d = Math.max(1, hyp(dx, dy));
    add(a, b, dx / d, dy / d, linkTarget(d) + exit(a, dx / d, dy / d) + exit(b, -dx / d, -dy / d));
  }
  const cx = new Float64Array(nb), cy = new Float64Array(nb), hx = new Float64Array(nb), hy = new Float64Array(nb);
  for (let b = 0; b < nb; b++) {
    cx[b] = (box[b * 4] + box[b * 4 + 2]) / 2, (cy[b] = (box[b * 4 + 1] + box[b * 4 + 3]) / 2);
    hx[b] = (box[b * 4 + 2] - box[b * 4]) / 2, (hy[b] = (box[b * 4 + 3] - box[b * 4 + 1]) / 2);
  }
  let x: Float64Array = new Float64Array(nu * 2);
  /** pairs of boxes held apart: b * nb + c -> axis (0 x, 1 y) and side (1: c beyond b) */
  const apart = new Map<number, [number, number]>();
  // on a rebuild, boxes nearer than the gap but not overlapping at the start are left so
  const near = new Set<number>();
  /** whether boxes b and c overlap, grown and padded to their spacing with `full` */
  const overlap = (b: number, c: number, full: boolean) => {
    const s = full ? 1 + BOX_GROW : 1, p = full ? 2 * pad : 0;
    return Math.abs(cx[c] + x[c * 2] - cx[b] - x[b * 2]) < (hx[b] + hx[c]) * s + p && Math.abs(cy[c] + x[c * 2 + 1] - cy[b] - x[b * 2 + 1]) < (hy[b] + hy[c]) * s + p;
  };
  if (held && !strict) for (let b = 0; b < nb; b++) for (let c = b + 1; c < nb; c++) if (overlap(b, c, true) && !overlap(b, c, false)) near.add(b * nb + c);
  // each unit held to its place as its stations each would be, so where the hops leave a choice the big cities
  // stay and the small ones move
  const wAnchor = new Float64Array(nu).fill(W_FREE);
  const size = new Float64Array(nb);
  for (let i = 0; i < n; i++) if (of[i] >= 0) size[of[i]]++;
  for (let b = 0; b < nb; b++) wAnchor[b] *= Math.sqrt(size[b]);
  for (let r = 0; r < ROUNDS; r++) {
    const T = new Terms(nu * 2);
    for (let k = 0; k < hops.length; k += 5) {
      const a = hops[k], b = hops[k + 1], ua = unit[a], ub = unit[b], t = hops[k + 4];
      T.add([ub * 2, ua * 2], [1, -1], hops[k + 2] * t - (pos[b * 2] - pos[a * 2]), W_HOP);
      T.add([ub * 2 + 1, ua * 2 + 1], [1, -1], hops[k + 3] * t - (pos[b * 2 + 1] - pos[a * 2 + 1]), W_HOP);
    }
    for (let u = 0; u < nu * 2; u++) T.add([u], [1], 0, wAnchor[u >> 1]);
    // boxes kept apart, along the axis they overlap less on (on a first layout, from the hops' places on)
    if (held || r) for (let b = 0; b < nb; b++) for (let c = b + 1; c < nb; c++) {
      if (near.has(b * nb + c) && !overlap(b, c, false)) continue;
      const dx = cx[c] + x[c * 2] - cx[b] - x[b * 2], dy = cy[c] + x[c * 2 + 1] - cy[b] - x[b * 2 + 1];
      const nx = (hx[b] + hx[c]) * (1 + BOX_GROW) + 2 * pad, ny = (hy[b] + hy[c]) * (1 + BOX_GROW) + 2 * pad;
      let h = apart.get(b * nb + c);
      if (!h) {
        const ox = nx - Math.abs(dx), oy = ny - Math.abs(dy);
        if (ox <= 0 || oy <= 0) continue;
        h = ox < oy ? [0, dx < 0 ? -1 : 1] : [1, dy < 0 ? -1 : 1];
        apart.set(b * nb + c, h);
      } else if ((h[0] ? dy * h[1] : dx * h[1]) >= HOLD * (h[0] ? ny : nx) || Math.abs(h[0] ? dx : dy) >= (h[0] ? nx : ny)) {
        apart.delete(b * nb + c);
        continue;
      }
      const ax = h[0];
      T.add([c * 2 + ax, b * 2 + ax], [1, -1], h[1] * (ax ? ny : nx) - (ax ? cy[c] - cy[b] : cx[c] - cx[b]), W_BOX);
    }
    // the first solve places everything; later ones only move boxes apart
    if (r && !apart.size) break;
    x = T.solve(x, r ? ITERS_MORE : ITERS, 1e-9, 1e-9);
  }
  const mv = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) (mv[i * 2] = x[unit[i] * 2]), (mv[i * 2 + 1] = x[unit[i] * 2 + 1]);
  for (let b = 0; b < nb; b++) (B.list[b].move[0] += x[b * 2]), (B.list[b].move[1] += x[b * 2 + 1]);
  return mv;
}

/** Each block's drawn box and its fit from real to drawn places, from the drawing `x`, `y`. */
export function finishBlocks(g: DiagramGraph, B: Blocks, x: Float64Array, y: Float64Array): void {
  const nb = B.list.length;
  const pos = new Float64Array(x.length * 2);
  for (let i = 0; i < x.length; i++) (pos[i * 2] = x[i]), (pos[i * 2 + 1] = y[i]);
  const box = blockBoxes(B, pos);
  // normal equations per block: sums of 1, x, y, xx, xy, yy over its stations, and of X, Y times 1, x, y
  const S = new Float64Array(nb * 12);
  for (let i = 0; i < x.length; i++) {
    const b = B.of[i];
    if (b < 0) continue;
    const gx = g.x[i], gy = g.y[i], o = b * 12;
    S[o]++, (S[o + 1] += gx), (S[o + 2] += gy), (S[o + 3] += gx * gx), (S[o + 4] += gx * gy), (S[o + 5] += gy * gy);
    (S[o + 6] += x[i]), (S[o + 7] += x[i] * gx), (S[o + 8] += x[i] * gy), (S[o + 9] += y[i]), (S[o + 10] += y[i] * gx), (S[o + 11] += y[i] * gy);
  }
  for (let b = 0; b < nb; b++) {
    const blk = B.list[b];
    blk.box = [box[b * 4], box[b * 4 + 1], box[b * 4 + 2], box[b * 4 + 3]];
    const o = b * 12, k = S[o];
    if (!k) continue;
    // centred: the linear part from the covariances, the shift from the means
    const mx = S[o + 1] / k, my = S[o + 2] / k, sxx = S[o + 3] / k - mx * mx, sxy = S[o + 4] / k - mx * my, syy = S[o + 5] / k - my * my;
    const mX = S[o + 6] / k, mY = S[o + 9] / k;
    const cXx = S[o + 7] / k - mX * mx, cXy = S[o + 8] / k - mX * my, cYx = S[o + 10] / k - mY * mx, cYy = S[o + 11] / k - mY * my;
    const det = sxx * syy - sxy * sxy;
    let a = 1, bb = 0, d = 0, e = 1;
    if (Math.abs(det) > 1e-9 * (sxx * syy + 1e-30)) {
      a = (cXx * syy - cXy * sxy) / det, (bb = (cXy * sxx - cXx * sxy) / det);
      d = (cYx * syy - cYy * sxy) / det, (e = (cYy * sxx - cYx * sxy) / det);
    } else {
      // stations on a line or one place: a scale alone
      const s = sxx + syy > 0 ? Math.sqrt(((cXx + cYy) * (cXx + cYy) + (cXy - cYx) * (cXy - cYx)) / ((sxx + syy) * (sxx + syy))) : 1;
      a = s, (e = s);
    }
    blk.fit = [a, bb, mX - a * mx - bb * my, d, e, mY - d * mx - e * my];
  }
}

/** per edge, 1 for a hop between blocks or touching a station between them */
export function blockHops(g: DiagramGraph, B: Blocks): Uint8Array {
  const out = new Uint8Array(g.ea.length);
  for (let e = 0; e < g.ea.length; e++) {
    const a = B.of[g.ea[e]], b = B.of[g.eb[e]];
    out[e] = a < 0 || a !== b ? 1 : 0;
  }
  return out;
}
