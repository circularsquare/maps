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
  /** T-4153: the edges whose samples or pairs the call made again, null when any may differ from the last call's (the
   * maps are the last call's, changed in place, when it gives a list) */
  remade?: readonly number[] | null;
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

/** one edge's raw records, before the mirror ones are added: per sample `i`, records `first[i]` to `first[i + 1]` of
 * `edge`, `off`, `at`, `rel` (as `EdgeSamples`', with `off` unrounded); `first` null when no sample has any
 * (T-4184: typed arrays, where a JS array a sample took several times the memory) */
interface Raw { prep: Prep; first: Int32Array | null; edge: Int32Array; off: Float64Array; at: Int32Array; rel: Int8Array }

const NO_EDGE = new Int32Array(0), NO_OFF = new Float64Array(0), NO_REL = new Int8Array(0);
const noRecords = (prep: Prep): Raw => ({ prep, first: null, edge: NO_EDGE, off: NO_OFF, at: NO_EDGE, rel: NO_REL });

/** `rec`: per sample, [edge, off, at, rel] fours */
function packRaw(prep: Prep, rec: number[][]): Raw {
  let total = 0;
  for (const v of rec) total += v.length / 4;
  if (!total) return noRecords(prep);
  const first = new Int32Array(rec.length + 1), edge = new Int32Array(total), off = new Float64Array(total), at = new Int32Array(total), rel = new Int8Array(total);
  let o = 0;
  rec.forEach((v, i) => {
    first[i] = o;
    for (let q = 0; q < v.length; q += 4, o++) {
      edge[o] = v[q];
      off[o] = v[q + 1];
      at[o] = v[q + 2];
      rel[o] = v[q + 3];
    }
  });
  first[rec.length] = o;
  return { prep, first, edge, off, at, rel };
}

/** the last call's records by edge, kept while an edge and everything near it is as it was */
let cache = new Map<number, Raw>();
let cacheWithin = 0;
/** what the last call gave, handed back while nothing has changed */
let lastOut: Beside | null = null;
/** per edge, its samples as last given (the same object while unchanged) and its pair sums */
let made = new Map<number, { samples: EdgeSamples; sums: Map<number, [number, number, number]> }>();
/** per edge, the edges whose own records see it */
let seenBy = new Map<number, Set<number>>();
/** T-4153: `cache`'s edges' boxes in square cells of `size` local units (ids in no set order; one over more than IX_WIDE
 * cells in `wide`), and the most local units a metre of any of them (`kTop`, NaN when to be found again) and how many
 * have it, so an edit's call looks only near what changed */
interface BoxIx { size: number; cells: Map<number, number[]>; wide: Set<number>; kTop: number; kTopN: number }
let boxIx: BoxIx | null = null;
const IX_M = 2000, IX_WIDE = 64;
function ixCells(ix: BoxIx, b: Prep["box"], pad: number, f: (k: number) => void) {
  const s = ix.size;
  for (let cx = Math.floor((b[0] - pad) / s); cx <= Math.floor((b[2] + pad) / s); cx++)
    for (let cy = Math.floor((b[1] - pad) / s); cy <= Math.floor((b[3] + pad) / s); cy++) f(cx * 1_000_003 + cy);
}
const ixWide = (ix: BoxIx, b: Prep["box"]) => !((Math.floor(b[2] / ix.size) - Math.floor(b[0] / ix.size) + 1) * (Math.floor(b[3] / ix.size) - Math.floor(b[1] / ix.size) + 1) <= IX_WIDE);
function ixAdd(ix: BoxIx, e: Prep) {
  if (e.k > ix.kTop) (ix.kTop = e.k), (ix.kTopN = 1);
  else if (e.k === ix.kTop) ix.kTopN++;
  if (ixWide(ix, e.box)) { ix.wide.add(e.id); return; }
  ixCells(ix, e.box, 0, (k) => { const v = ix.cells.get(k); if (v) v.push(e.id); else ix.cells.set(k, [e.id]); });
}
function ixDrop(ix: BoxIx, e: Prep) {
  if (e.k === ix.kTop && --ix.kTopN === 0) ix.kTop = NaN;
  if (ix.wide.delete(e.id)) return;
  ixCells(ix, e.box, 0, (k) => {
    const v = ix.cells.get(k), at = v ? v.indexOf(e.id) : -1;
    if (at < 0) return;
    v![at] = v![v!.length - 1];
    v!.pop();
    if (!v!.length) ix.cells.delete(k);
  });
}
/** the index of `cache`, made once after a whole call */
function ixOf(): BoxIx {
  if (boxIx) return boxIx;
  let k = 1;
  for (const r of cache.values()) if (r.prep.k > k) k = r.prep.k;
  const ix: BoxIx = { size: IX_M * k, cells: new Map(), wide: new Set(), kTop: -Infinity, kTopN: 0 };
  for (const r of cache.values()) ixAdd(ix, r.prep);
  return (boxIx = ix);
}
/** the most local units a metre of `cache`'s edges */
function ixTop(ix: BoxIx): number {
  if (Number.isNaN(ix.kTop)) {
    ix.kTop = -Infinity;
    ix.kTopN = 0;
    for (const r of cache.values()) {
      if (r.prep.k > ix.kTop) (ix.kTop = r.prep.k), (ix.kTopN = 1);
      else if (r.prep.k === ix.kTop) ix.kTopN++;
    }
  }
  return ix.kTop;
}
/** the ids of `cache`'s edges whose boxes may come within `pad` of box `b` (each once) */
function ixNear(ix: BoxIx, b: Prep["box"], pad: number, out: Set<number>) {
  for (const id of ix.wide) out.add(id);
  ixCells(ix, b, pad, (k) => { for (const id of ix.cells.get(k) ?? []) out.add(id); });
}

/** T-4103: the view's anchor moved (`shiftNetCache`): each edge in `moved` is the same track in the moved frame, so its
 * records (metres along and across edges) carry over to its new points, and a next call that finds them unchanged
 * gives what it gave before. Without this a far jump of the view worked every edge out again. */
/** T-4153: what `besideSamples` keeps between calls, one per network cache (map/network.ts), so a build from scratch
 * beside the kept one (`?patch=check`) neither reads nor moves it; the module's variables hold the one in use */
export interface BesideState { cache: Map<number, Raw>; within: number; lastOut: Beside | null; made: typeof made; seenBy: Map<number, Set<number>>; boxIx: BoxIx | null }
export const newBesideState = (): BesideState => ({ cache: new Map(), within: 0, lastOut: null, made: new Map(), seenBy: new Map(), boxIx: null });
const sharedBeside: BesideState = { cache, within: cacheWithin, lastOut, made, seenBy, boxIx };
let inUse = sharedBeside;
function useBeside(st: BesideState) {
  if (st === inUse) return;
  Object.assign(inUse, { cache, within: cacheWithin, lastOut, made, seenBy, boxIx });
  ({ cache, within: cacheWithin, lastOut, made, seenBy, boxIx } = st);
  inUse = st;
}
export function shiftBeside(moved: ReadonlyMap<Float32Array, Float32Array>, st: BesideState = sharedBeside) {
  useBeside(st);
  boxIx = null;
  for (const [id, r] of cache) {
    const p = moved.get(r.prep.p);
    if (!p || p.length !== r.prep.p.length) continue;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < p.length; i += 3) {
      if (p[i] < x0) x0 = p[i];
      if (p[i] > x1) x1 = p[i];
      if (p[i + 1] < y0) y0 = p[i + 1];
      if (p[i + 1] > y1) y1 = p[i + 1];
    }
    cache.set(id, { ...r, prep: { ...r.prep, p, k: kAtY((y0 + y1) / 2), box: [x0, y0, x1, y1] } });
  }
}

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

const sameArr = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
};

/** box tests past which every edge is worked out again (a few ms of tests; an edit that changes that much is mostly redone) */
const MANY_TESTS = 4e6;

const overlaps = (a: Prep["box"], b: Prep["box"], pad: number) => !(a[2] + pad < b[0] || b[2] + pad < a[0] || a[3] + pad < b[1] || b[3] + pad < a[1]);

/**
 * Per edge of `ids` (edges that carry lines), its samples and what runs beside each. `pts`: each
 * edge's local polyline (x, y, z triples, a to b); `ends` its nodes. `within`: CORRIDOR_M unless a
 * check asks otherwise. `changes` (T-4153): the edges whose points, ends or place in `ids` may differ from the last
 * call's, so the call looks again only at those and the edges near them, and changes the last call's maps in place.
 */
export function besideSamples(ids: readonly number[] | ReadonlyMap<number, unknown>, pts: ReadonlyMap<number, Float32Array>, ends: ReadonlyMap<number, { a: number; b: number }>, within = CORRIDOR_M, st: BesideState = sharedBeside, changes?: { edges: Iterable<number>[]; has: (id: number) => boolean }): Beside {
  useBeside(st);
  if (within !== cacheWithin) (cache = new Map()), (lastOut = null), (boxIx = null);
  cacheWithin = within;
  if (changes && lastOut) {
    const out = besideChanged(changes, pts, ends, within);
    if (out) return out;
  }
  const list: Prep[] = [];
  let raw = new Map<number, Raw>();
  const changed: Prep["box"][] = [];
  const seen = new Set<number>();
  let kMax = 1;
  // (T-4153: the edges in id order, which no edit moves, so the order of each sample's records, and the ribbons walked
  // from them, is the same as from scratch whatever order the lines name the edges in)
  const byId = (Array.isArray(ids) ? [...ids] : [...(ids as ReadonlyMap<number, unknown>).keys()]).sort((a, b) => a - b);
  for (const id of byId) {
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
  // edges to work out again: new or moved ones, and any near where something changed (T-4153: each edge tested against
  // every change, past MANY_TESTS all worked out again; a long line removed costs the edges near its track)
  const redo: Prep[] = [];
  const many = changed.length * list.length > MANY_TESTS;
  const all = boxOfAll(changed);
  for (const e of list) {
    const r = raw.get(e.id);
    if (!r || many || (changed.length && overlaps(all, e.box, reach) && changed.some((b) => overlaps(b, e.box, reach)))) redo.push(e);
  }
  if (!redo.length && !changed.length && lastOut && raw.size === cache.size) {
    lastOut.remade = [];
    return lastOut;
  }
  if (redo.length) {
    const area = boxOfAll(redo.map((e) => e.box));
    // (every edge near those, whose segments they are looked for among)
    const near = redo.length === list.length ? list : list.filter((e) => overlaps(e.box, area, reach));
    for (const [id, r] of recordsOf(redo, near, reach, within)) raw.set(id, r);
    // (those new since the last call came last)
    const sorted = new Map<number, Raw>();
    for (const id of byId) {
      const r = raw.get(id);
      if (r) sorted.set(id, r);
    }
    raw = sorted;
  }
  // T-2238: an edge's samples as given (its own records and the mirrors of others') are made again only
  // where its own records or those of an edge seeing it were, so an edit's call costs what it touched
  const whole = !lastOut || many || redo.length === list.length;
  const redone: number[] = [];
  for (const id of raw.keys()) if (whole || raw.get(id) !== cache.get(id)) redone.push(id);
  for (const id of cache.keys()) if (!raw.has(id)) redone.push(id);
  if (whole) seenBy = new Map();
  const remake = seersMoved(redone, (g) => (whole ? undefined : cache.get(g)), (g) => raw.get(g));
  cache = raw;
  // (T-4153: the index an edit's call looks near what changed by, made now so the first edit does not wait for it)
  boxIx = null;
  ixOf();
  for (const id of [...made.keys()]) if (!raw.has(id)) (made.delete(id), seenBy.delete(id));
  remakeSamples(remake);
  const out: Beside = { edges: new Map(), pair: new Map(), remade: null };
  for (const id of raw.keys()) {
    const m = made.get(id)!;
    out.edges.set(id, m.samples);
    for (const [f, x] of m.sums) putPair(out, id, f, x);
  }
  return (lastOut = out);
}

const boxOfAll = (boxes: readonly Prep["box"][]): Prep["box"] => {
  const all: Prep["box"] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const b of boxes) (all[0] = Math.min(all[0], b[0])), (all[1] = Math.min(all[1], b[1])), (all[2] = Math.max(all[2], b[2])), (all[3] = Math.max(all[3], b[3]));
  return all;
};

/** the pair of edges `id` and `f` (f the higher id) side by side: [offset sum, rel, samples], put both ways */
function putPair(out: Beside, id: number, f: number, [sum, rel, n]: [number, number, number]) {
  const side = (a: number, b: number, x: [number, number, number]) => {
    let m = out.pair.get(a);
    if (!m) out.pair.set(a, (m = new Map()));
    m.set(b, x);
  };
  side(id, f, [sum / n, rel, n]);
  side(f, id, [(-sum / n) * rel, rel, n]);
}

/**
 * T-4153: the last call's samples and pairs brought up to date in place where `changes` say edges may differ: those
 * whose points, ends or place among the edges with lines changed are prepared again, and the edges near them worked out
 * again as a whole call would (in id order where order counts). Null when so much changed that a whole call is quicker.
 */
function besideChanged(changes: { edges: Iterable<number>[]; has: (id: number) => boolean }, pts: ReadonlyMap<number, Float32Array>, ends: ReadonlyMap<number, { a: number; b: number }>, within: number): Beside | null {
  const out = lastOut!;
  const changed: Prep["box"][] = [], fresh = new Map<number, Prep>(), gone: number[] = [];
  const looked = new Set<number>();
  for (const list of changes.edges) for (const id of list) {
    if (looked.has(id)) continue;
    looked.add(id);
    const old = cache.get(id), p = pts.get(id), ed = ends.get(id), carries = !!p && !!ed && changes.has(id);
    if (carries && old && old.prep.p === p && old.prep.a === ed!.a && old.prep.b === ed!.b) continue;
    if (old) changed.push(old.prep.box);
    const e = carries ? prepOf(id, p!, ed!.a, ed!.b) : null;
    if (e) {
      fresh.set(id, e);
      changed.push(e.box);
    } else if (old) gone.push(id);
  }
  if (!changed.length) {
    out.remade = [];
    return out;
  }
  // (the most local units a metre of any edge, as a whole call takes it: T-4153, the index's most, unless every edge with
  // it is made again here)
  const ix = ixOf();
  let kMax = 1, n = fresh.size, topFresh = 0;
  const top = ixTop(ix);
  for (const id of fresh.keys()) {
    const r = cache.get(id);
    if (r) { if (r.prep.k === top) topFresh++; } else n++;
  }
  n += cache.size - fresh.size;
  if (topFresh < ix.kTopN) kMax = Math.max(kMax, top);
  else for (const [id, r] of cache) if (!fresh.has(id) && r.prep.k > kMax) kMax = r.prep.k;
  for (const e of fresh.values()) if (e.k > kMax) kMax = e.k;
  if (changed.length * n > MANY_TESTS) return null;
  const reach = HOLD * within * kMax, all = boxOfAll(changed);
  const goneSet = new Set(gone);
  /** (T-4153: the kept edges the index finds near a box, and every edge made again) */
  const nearBy = (boxes: readonly Prep["box"][], fn: (e: Prep) => void) => {
    const ids = new Set<number>();
    for (const b of boxes) ixNear(ix, b, reach, ids);
    for (const id of ids) if (!fresh.has(id) && !goneSet.has(id)) { const r = cache.get(id); if (r) fn(r.prep); }
    for (const e of fresh.values()) fn(e);
  };
  const redo: Prep[] = [];
  nearBy(changed, (e) => { if (fresh.has(e.id) || (overlaps(all, e.box, reach) && changed.some((b) => overlaps(b, e.box, reach)))) redo.push(e); });
  // (in id order, as a whole call works them out)
  redo.sort((a, b) => a.id - b.id);
  const area = boxOfAll(redo.map((e) => e.box)), near: Prep[] = [];
  nearBy([area], (e) => { if (overlaps(e.box, area, reach)) near.push(e); });
  near.sort((a, b) => a.id - b.id);
  const got = recordsOf(redo, near, reach, within);
  const redone = [...got.keys(), ...gone];
  const was = new Map<number, Raw | undefined>();
  for (const g of redone) was.set(g, cache.get(g));
  for (const [id, r] of got) cache.set(id, r);
  for (const id of gone) cache.delete(id);
  for (const g of redone) {
    const o = was.get(g)?.prep, p = cache.get(g)?.prep;
    if (o === p) continue;
    if (o) ixDrop(ix, o);
    if (p) ixAdd(ix, p);
  }
  const remake = seersMoved(redone, (g) => was.get(g), (g) => cache.get(g));
  const oldSums = new Map<number, Map<number, [number, number, number]> | undefined>();
  for (const id of gone) {
    oldSums.set(id, made.get(id)?.sums);
    made.delete(id);
    seenBy.delete(id);
  }
  for (const f of remake) if (!oldSums.has(f)) oldSums.set(f, made.get(f)?.sums);
  const remade = remakeSamples(remake);
  // the samples and pairs of the edges whose own changed, in the last call's maps
  const touched = [...gone, ...remade];
  for (const f of touched) {
    for (const g of oldSums.get(f)?.keys() ?? []) {
      out.pair.get(f)?.delete(g);
      out.pair.get(g)?.delete(f);
    }
    const m = made.get(f);
    if (!m) {
      out.edges.delete(f);
      if (!out.pair.get(f)?.size) out.pair.delete(f);
      continue;
    }
    out.edges.set(f, m.samples);
  }
  for (const f of remade) for (const [g, x] of made.get(f)!.sums) putPair(out, f, g, x);
  out.remade = touched;
  return out;
}

/** each sample's records of the edges `redo` (in id order): the other edges of `near` beside it there */
function recordsOf(redo: readonly Prep[], near: readonly Prep[], reach: number, within: number): Map<number, Raw> {
  const raw = new Map<number, Raw>();
  const cell = 3 * reach;
  // a grid of the segments of every edge near those, cells several reaches wide
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
      raw.set(e.id, noRecords(e));
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
    raw.set(e.id, packRaw(e, rec));
  }
  return raw;
}

/** the edges whose samples are to be made again: each edge redone, and every edge its records named before (`was`) or
 * name now (`now`); `seenBy` brought up to date */
function seersMoved(redone: readonly number[], was: (g: number) => Raw | undefined, now: (g: number) => Raw | undefined): Set<number> {
  const targets = (r: Raw | undefined, f: (t: number) => void) => { if (r) for (let x = 0; x < r.edge.length; x++) f(r.edge[x]); };
  const remake = new Set<number>();
  for (const g of redone) {
    if (now(g)) remake.add(g);
    targets(was(g), (t) => { seenBy.get(t)?.delete(g); remake.add(t); });
    targets(now(g), (t) => {
      let s = seenBy.get(t);
      if (!s) seenBy.set(t, (s = new Set()));
      s.add(g);
      remake.add(t);
    });
  }
  return remake;
}

/** the samples (`made`) of the edges `remake` made again from `cache`; those whose samples changed */
function remakeSamples(remake: ReadonlySet<number>): number[] {
  // per edge to make again, the records of the edges seeing it (sample, place in it), in id order of the edges seeing
  // it, each seeing edge's records read once
  const seers = new Set<number>();
  for (const f of remake) if (cache.has(f)) for (const g of seenBy.get(f) ?? []) if (cache.has(g)) seers.add(g);
  const seenAt = new Map<number, number[]>();
  for (const g of [...seers].sort((a, b) => a - b)) {
    const { first, edge } = cache.get(g)!;
    if (first) for (let i = 0; i + 1 < first.length; i++) for (let x = first[i]; x < first[i + 1]; x++) {
      if (!remake.has(edge[x])) continue;
      let s = seenAt.get(edge[x]);
      if (!s) seenAt.set(edge[x], (s = []));
      s.push(g, i, x);
    }
  }
  const changed: number[] = [];
  for (const f of remake) {
    const r = cache.get(f);
    if (!r) continue;
    // the mirror of each record of an edge seeing f, where f's sample there did not see that edge
    // itself, after that sample's own records
    // (per sample the edges it lists, kept once a mirror comes to it, rather than its list looked
    // through for each: a sample in a bundle of thirty tracks takes hundreds)
    const nS = r.prep.n, F0 = r.first, extra = new Map<number, number[]>(), lists = new Map<number, Set<number>>(), by = seenAt.get(f) ?? [];
    for (let t = 0; t < by.length; t += 3) {
      const g = by[t], i = by[t + 1], U = cache.get(g)!, x = by[t + 2], s = U.at[x];
      let has = lists.get(s);
      if (!has) {
        has = new Set();
        if (F0) for (let y = F0[s]; y < F0[s + 1]; y++) has.add(r.edge[y]);
        lists.set(s, has);
      }
      if (has.has(g)) continue;
      has.add(g);
      let e = extra.get(s);
      if (!e) extra.set(s, (e = []));
      e.push(g, -U.off[x] * U.rel[x], i, U.rel[x]);
    }
    const own = (s: number, each: (edge: number, off: number, at: number, rel: number) => void) => {
      if (F0) for (let y = F0[s]; y < F0[s + 1]; y++) each(r.edge[y], r.off[y], r.at[y], r.rel[y]);
      const e = extra.get(s);
      if (e) for (let y = 0; y < e.length; y += 4) each(e[y], e[y + 1], e[y + 2], e[y + 3]);
    };
    // each pair's offset over all the places it is side by side, from the lower id edge
    const sums = new Map<number, [number, number, number]>();
    let total = 0;
    for (let s = 0; s < nS; s++) own(s, (g, d, _at, rl) => {
      total++;
      if (g < f) return;
      const old = sums.get(g);
      if (old) {
        old[0] += d;
        old[2]++;
      } else sums.set(g, [d, rl, 1]);
    });
    const first = new Int32Array(nS + 1), edge = new Int32Array(total), off = new Float32Array(total), at = new Int32Array(total), rel = new Int8Array(total);
    let o = 0;
    for (let s = 0; s < nS; s++) {
      first[s] = o;
      own(s, (g, d, a, rl) => {
        edge[o] = g;
        off[o] = d;
        at[o] = a;
        rel[o] = rl;
        o++;
      });
    }
    first[nS] = o;
    const len = r.prep.cum[r.prep.cum.length - 1] / r.prep.k, was = made.get(f)?.samples;
    // the same samples as before stay the same object, which tells `lineBands` nothing changed there
    if (was && was.n === r.prep.n && Object.is(was.len, len) && sameArr(was.first, first) && sameArr(was.edge, edge) && sameArr(was.off, off) && sameArr(was.at, at) && sameArr(was.rel, rel)) continue;
    made.set(f, { samples: { n: r.prep.n, len, first, edge, off, at, rel }, sums });
    changed.push(f);
  }
  return changed;
}
