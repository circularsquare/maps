// Edges side by side (T-1799 stage 3, T-1848; notes/T-1799.md): every edge that carries lines is sampled
// every SAMPLE_M, and each sample records the other edges running beside it there, within CORRIDOR_M
// and about the same direction. The record is symmetric (a sample that sees an edge is put in that
// edge's sample at the same place), so both tracks see each other at the same places. `lineBands`
// makes the ribbon at each place from it and smooths it along each line.

import { kAtY } from "./geo";

/** edges this close (m, centre line to centre line) run side by side */
export const CORRIDOR_M = 20;
/** an edge is sampled about this often, m */
export const SAMPLE_M = 25;
/** two edges run the same way when their directions agree to within about 25 degrees */
const COS_MIN = 0.9;
/** a track once beside stays beside out to this many times CORRIDOR_M */
const HOLD = 1.5;
/** a sample with nothing beside (shared, never written to) */
const NONE: number[] = [];

export interface EdgeSamples {
  /** samples, at (i + 0.5) / n of the edge's length a to b */
  n: number;
  /** the edge's length, m */
  len: number;
  /** per sample i, its neighbours from first[i] to first[i + 1] */
  first: Int32Array;
  /** per neighbour: the edge */
  edge: Int32Array;
  /** metres from this edge's centre line to the neighbour's, left of this edge's a to b */
  off: Float32Array;
  /** the neighbour's own sample at this place */
  at: Int32Array;
  /** +1 when the neighbour's a to b runs this edge's way, -1 against it */
  rel: Int8Array;
}

export interface Beside {
  edges: Map<number, EdgeSamples>;
  /** per edge, per edge beside it anywhere: [its offset averaged over every sample where they are
   * side by side (m, left of the first's a to b), rel, samples] */
  pair: Map<number, Map<number, [number, number, number]>>;
}

interface Prep {
  id: number;
  p: Float32Array;
  /** local units per metre */
  k: number;
  /** arc length to each point, local units */
  cum: Float64Array;
  n: number;
  box: [number, number, number, number];
  /** its end nodes, and the unit direction it leaves each in (local units, over CONT_M) */
  a: number;
  b: number;
  da: [number, number];
  db: [number, number];
}

/** how far along an edge the direction it leaves a node in is measured, m */
const CONT_M = 30;

/** one edge's raw record: per sample, [edge, off, at, rel] fours, before the mirror ones are added */
interface Raw { prep: Prep; rec: number[][] }

/** the last call's records by edge, kept while an edge and everything near it is as it was */
let cache = new Map<number, Raw>();
let cacheWithin = 0;
/** what the last call gave, handed back while nothing has changed */
let lastOut: Beside | null = null;

function prepOf(id: number, p: Float32Array, a: number, b: number): Prep | null {
  const m = p.length / 3;
  if (m < 2) return null;
  const cum = new Float64Array(m);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < m; i++) {
    const x = p[i * 3], y = p[i * 3 + 1];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (i) cum[i] = cum[i - 1] + Math.hypot(x - p[i * 3 - 3], y - p[i * 3 - 2]);
  }
  const k = kAtY((y0 + y1) / 2);
  const len = cum[m - 1] / k;
  if (!(len > 0)) return null;
  const leave = (fromA: boolean): [number, number] => {
    const at = (i: number) => (fromA ? i : m - 1 - i), x0 = p[at(0) * 3], y0 = p[at(0) * 3 + 1];
    let i = 1;
    while (i < m - 1 && Math.abs(cum[at(i)] - cum[at(0)]) < CONT_M * k) i++;
    const dx = p[at(i) * 3] - x0, dy = p[at(i) * 3 + 1] - y0, d = Math.hypot(dx, dy) || 1;
    return [dx / d, dy / d];
  };
  return { id, p, k, cum, n: Math.max(1, Math.round(len / SAMPLE_M)), box: [x0, y0, x1, y1], a, b, da: leave(true), db: leave(false) };
}

/** whether `f` carries on from `e` through a node they share (the next stretch of the same track,
 * which is beside `e` only where they meet), rather than leaving it side by side */
function carriesOn(e: Prep, f: Prep): boolean {
  for (const [n, d] of [[e.a, e.da], [e.b, e.db]] as const) {
    const g = f.a === n ? f.da : f.b === n ? f.db : null;
    // arriving along e is the reverse of leaving along it
    if (g && -(d[0] * g[0] + d[1] * g[1]) > 0.5) return true;
  }
  return false;
}

const overlaps = (a: Prep["box"], b: Prep["box"], pad: number) => !(a[2] + pad < b[0] || b[2] + pad < a[0] || a[3] + pad < b[1] || b[3] + pad < a[1]);

/**
 * Per edge of `ids` (edges that carry lines), its samples and what runs beside each. `pts`: each
 * edge's local polyline (x, y, z triples, a to b); `ends` its nodes. `within`: CORRIDOR_M unless a
 * check asks otherwise.
 */
export function besideSamples(ids: readonly number[], pts: ReadonlyMap<number, Float32Array>, ends: ReadonlyMap<number, { a: number; b: number }>, within = CORRIDOR_M): Beside {
  if (within !== cacheWithin) (cache = new Map()), (lastOut = null);
  cacheWithin = within;
  const list: Prep[] = [];
  const raw = new Map<number, Raw>();
  const changed: Prep["box"][] = [];
  const seen = new Set<number>();
  let kMax = 1;
  for (const id of ids) {
    const p = pts.get(id), ed = ends.get(id);
    if (!p || !ed) continue;
    seen.add(id);
    const old = cache.get(id);
    if (old && old.prep.p === p && old.prep.a === ed.a && old.prep.b === ed.b) {
      list.push(old.prep);
      raw.set(id, old);
      if (old.prep.k > kMax) kMax = old.prep.k;
      continue;
    }
    if (old) changed.push(old.prep.box);
    const e = prepOf(id, p, ed.a, ed.b);
    if (!e) continue;
    list.push(e);
    changed.push(e.box);
    if (e.k > kMax) kMax = e.k;
  }
  for (const [id, old] of cache) if (!seen.has(id)) changed.push(old.prep.box);
  const reach = HOLD * within * kMax;
  // edges to work out again: new or moved ones, and any near where something changed
  const redo: Prep[] = [];
  const many = changed.length > 64;
  for (const e of list) {
    const r = raw.get(e.id);
    if (!r || many || (changed.length && changed.some((b) => overlaps(b, e.box, reach)))) redo.push(e);
  }
  if (!redo.length && !changed.length && lastOut && raw.size === cache.size) return lastOut;
  if (redo.length) {
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (const e of redo) {
      bx0 = Math.min(bx0, e.box[0]);
      by0 = Math.min(by0, e.box[1]);
      bx1 = Math.max(bx1, e.box[2]);
      by1 = Math.max(by1, e.box[3]);
    }
    const area: Prep["box"] = [bx0, by0, bx1, by1];
    // a grid of the segments of every edge near those, cells several reaches wide
    const near = redo.length === list.length ? list : list.filter((e) => overlaps(e.box, area, reach));
    const cell = 3 * reach;
    const grid = new Map<number, number[]>();
    // (a small integer, wrapping every 32768 cells: cells that share one are only looked at twice)
    const key = (ix: number, iy: number) => ((ix & 32767) << 15) | (iy & 32767);
    near.forEach((e, ei) => {
      const P = e.p;
      for (let j = 0; j + 1 < P.length / 3; j++) {
        const ax = P[j * 3], ay = P[j * 3 + 1], bx = P[j * 3 + 3], by = P[j * 3 + 4];
        const cx0 = Math.floor(Math.min(ax, bx) / cell), cx1 = Math.floor(Math.max(ax, bx) / cell);
        const cy0 = Math.floor(Math.min(ay, by) / cell), cy1 = Math.floor(Math.max(ay, by) / cell);
        for (let cx = cx0; cx <= cx1; cx++) for (let cy = cy0; cy <= cy1; cy++) {
          const c = key(cx, cy);
          const v = grid.get(c);
          if (v) v.push(ei * 1048576 + j);
          else grid.set(c, [ei * 1048576 + j]);
        }
      }
    });
    const bestD = new Float64Array(near.length).fill(Infinity), bestJ = new Int32Array(near.length), bestT = new Float64Array(near.length);
    const touched: number[] = [];
    const index = new Map(near.map((e, i) => [e.id, i]));
    for (const e of redo) {
      const me = index.get(e.id)!, P = e.p, m = P.length / 3, total = e.cum[m - 1];
      const rec: number[][] = [];
      const carry = new Map<number, boolean>();
      // the other edges in or next to its cells: none (most track in the country) and every sample
      // is empty; else a sample is only looked at near one of their boxes
      const cells = new Set<number>(), cand = new Set<number>();
      for (let j = 0; j + 1 < m; j++) {
        const cx0 = Math.floor(Math.min(P[j * 3], P[j * 3 + 3]) / cell) - 1, cx1 = Math.floor(Math.max(P[j * 3], P[j * 3 + 3]) / cell) + 1;
        const cy0 = Math.floor(Math.min(P[j * 3 + 1], P[j * 3 + 4]) / cell) - 1, cy1 = Math.floor(Math.max(P[j * 3 + 1], P[j * 3 + 4]) / cell) + 1;
        for (let cx = cx0; cx <= cx1; cx++) for (let cy = cy0; cy <= cy1; cy++) cells.add(key(cx, cy));
      }
      for (const c of cells) for (const code of grid.get(c) ?? []) {
        const fi = Math.floor(code / 1048576);
        if (fi !== me) cand.add(fi);
      }
      const boxes: Prep["box"][] = [];
      for (const fi of cand) if (!carriesOn(e, near[fi]) && overlaps(e.box, near[fi].box, reach)) boxes.push(near[fi].box);
      if (!boxes.length) {
        for (let i = 0; i < e.n; i++) rec.push(NONE);
        raw.set(e.id, { prep: e, rec });
        continue;
      }
      let seg = 0;
      for (let i = 0; i < e.n; i++) {
        const want = ((i + 0.5) / e.n) * total;
        while (seg < m - 2 && e.cum[seg + 1] < want) seg++;
        const sl = e.cum[seg + 1] - e.cum[seg], f = sl > 0 ? (want - e.cum[seg]) / sl : 0;
        const dx = P[seg * 3 + 3] - P[seg * 3], dy = P[seg * 3 + 4] - P[seg * 3 + 1], dl = Math.hypot(dx, dy) || 1;
        const px = P[seg * 3] + dx * f, py = P[seg * 3 + 1] + dy * f, tx = dx / dl, ty = dy / dl;
        const r = HOLD * within * e.k;
        let inBox = false;
        for (let q = 0; q < boxes.length && !inBox; q++) {
          const b = boxes[q];
          inBox = px > b[0] - r && px < b[2] + r && py > b[1] - r && py < b[3] + r;
        }
        if (!inBox) {
          rec.push(NONE);
          continue;
        }
        const found: number[] = [];
        for (let cx = Math.floor((px - r) / cell); cx <= Math.floor((px + r) / cell); cx++) for (let cy = Math.floor((py - r) / cell); cy <= Math.floor((py + r) / cell); cy++) {
          const v = grid.get(key(cx, cy));
          if (!v) continue;
          for (const code of v) {
            const fi = Math.floor(code / 1048576);
            if (fi === me) continue;
            let on = carry.get(fi);
            if (on === undefined) carry.set(fi, (on = carriesOn(e, near[fi])));
            if (on) continue;
            const j = code - fi * 1048576, Q = near[fi].p;
            const ax = Q[j * 3], ay = Q[j * 3 + 1], ex = Q[j * 3 + 3] - ax, ey = Q[j * 3 + 4] - ay, l2 = ex * ex + ey * ey;
            const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * ex + (py - ay) * ey) / l2)) : 0;
            const d = (ax + ex * t - px) ** 2 + (ay + ey * t - py) ** 2;
            if (d > r * r) continue;
            if (bestD[fi] === Infinity) touched.push(fi);
            if (d < bestD[fi]) {
              bestD[fi] = d;
              bestJ[fi] = j;
              bestT[fi] = t;
            }
          }
        }
        for (const fi of touched) {
          const F = near[fi], Q = F.p, j = bestJ[fi], t = bestT[fi];
          const ex = Q[j * 3 + 3] - Q[j * 3], ey = Q[j * 3 + 4] - Q[j * 3 + 1], el = Math.hypot(ex, ey) || 1;
          const c = (tx * ex + ty * ey) / el;
          if (Math.abs(c) >= COS_MIN) {
            const qx = Q[j * 3] + ex * t, qy = Q[j * 3 + 1] + ey * t;
            // left of the direction (tx, ty), local y pointing south, is (ty, -tx)
            const off = ((qx - px) * ty - (qy - py) * tx) / e.k;
            const u = (F.cum[j] + t * el) / F.cum[F.cum.length - 1];
            found.push(F.id, off, Math.min(F.n - 1, Math.floor(u * F.n)), c > 0 ? 1 : -1);
          }
          bestD[fi] = Infinity;
        }
        touched.length = 0;
        rec.push(found);
      }
      // past `within` (up to HOLD times it) a neighbour stays beside only along from where it was
      // within, so a track at about `within` does not come and go
      const held = new Map<number, Uint8Array>();
      rec.forEach((v, i) => {
        for (let q = 0; q < v.length; q += 4) {
          let h = held.get(v[q]);
          if (!h) held.set(v[q], (h = new Uint8Array(e.n)));
          h[i] = Math.abs(v[q + 1]) <= within ? 2 : 1;
        }
      });
      for (const h of held.values()) {
        for (let i = 1; i < h.length; i++) if (h[i] === 1 && h[i - 1] === 2) h[i] = 2;
        for (let i = h.length - 2; i >= 0; i--) if (h[i] === 1 && h[i + 1] === 2) h[i] = 2;
      }
      rec.forEach((v, i) => {
        if (!v.length) return;
        const keep: number[] = [];
        for (let q = 0; q < v.length; q += 4) if (held.get(v[q])![i] === 2) keep.push(v[q], v[q + 1], v[q + 2], v[q + 3]);
        rec[i] = keep;
      });
      raw.set(e.id, { prep: e, rec });
    }
  }
  cache = raw;
  // the mirror of each record, where the other edge's sample there did not see this one itself
  // (a sample's list is copied before the first mirror is added to it: the cache keeps the raw ones)
  const rec = new Map<number, number[][]>(), copied = new Map<number, Set<number>>();
  for (const [id, r] of raw) rec.set(id, r.rec.slice());
  for (const [id, r] of raw) {
    r.rec.forEach((v, i) => {
      for (let q = 0; q < v.length; q += 4) {
        const other = rec.get(v[q]);
        if (!other) continue;
        let w = other[v[q + 2]];
        let has = false;
        for (let s = 0; s < w.length && !has; s += 4) has = w[s] === id;
        if (has) continue;
        let c = copied.get(v[q]);
        if (!c) copied.set(v[q], (c = new Set()));
        if (!c.has(v[q + 2])) {
          c.add(v[q + 2]);
          other[v[q + 2]] = w = w.slice();
        }
        w.push(id, -v[q + 1] * v[q + 3], i, v[q + 3]);
      }
    });
  }
  const out: Beside = { edges: new Map(), pair: new Map() };
  // each pair's offset over all the places it is side by side, from the lower id edge
  const sums = new Map<number, Map<number, [number, number, number]>>();
  for (const [id, v] of rec) {
    for (const s of v) for (let q = 0; q < s.length; q += 4) {
      if (s[q] < id) continue;
      let m = sums.get(id);
      if (!m) sums.set(id, (m = new Map()));
      const old = m.get(s[q]);
      if (old) {
        old[0] += s[q + 1];
        old[2]++;
      } else m.set(s[q], [s[q + 1], s[q + 3], 1]);
    }
  }
  const side = (a: number, b: number, x: [number, number, number]) => {
    let m = out.pair.get(a);
    if (!m) out.pair.set(a, (m = new Map()));
    m.set(b, x);
  };
  for (const [id, m] of sums) for (const [f, [sum, rel, n]] of m) {
    side(id, f, [sum / n, rel, n]);
    side(f, id, [(-sum / n) * rel, rel, n]);
  }
  for (const [id, v] of rec) {
    const r = raw.get(id)!;
    let total = 0;
    for (const s of v) total += s.length / 4;
    const first = new Int32Array(v.length + 1), edge = new Int32Array(total), off = new Float32Array(total), at = new Int32Array(total), rel = new Int8Array(total);
    let o = 0;
    v.forEach((s, i) => {
      first[i] = o;
      for (let q = 0; q < s.length; q += 4, o++) {
        edge[o] = s[q];
        off[o] = s[q + 1];
        at[o] = s[q + 2];
        rel[o] = s[q + 3];
      }
    });
    first[v.length] = o;
    out.edges.set(id, { n: r.prep.n, len: r.prep.cum[r.prep.cum.length - 1] / r.prep.k, first, edge, off, at, rel });
  }
  return (lastOut = out);
}
