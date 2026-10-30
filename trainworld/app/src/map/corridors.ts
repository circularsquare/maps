// Corridors (T-1799 stage 3, notes/T-1799.md): edges that run side by side along their whole length
// within CORRIDOR_M of each other, such as the two double tracks of a four-track trunk (8 m apart,
// T-1305), are drawn as one ribbon of bands. This finds them: which edges, which way each runs against
// the corridor's direction, and how far each lies from its middle.

import { kAtY } from "./geo";

/** edges this close (m, centre line to centre line) side by side are one corridor */
export const CORRIDOR_M = 20;
/** the share of each edge's length that must lie alongside the other's, for both edges */
const COVER = 0.6;
/** two edges run the same way when their directions agree to within about 25 degrees */
const COS_MIN = 0.9;
/** along an edge, a sample every so many metres */
const SAMPLE_M = 25;
/** a corridor wider than this (m, outermost centre lines) is a throat or yard, not a trunk, and is left
 * as it was: a four-track trunk is 8 m across a pair of edges, a pair of pairs 24 m, the Penn throat's
 * six edges 43 m; any number of edges may chain along a corridor's length, as Queens Boulevard's nine
 * lines on nine edges over 2 m do (Anita) */
const SPAN_M = 30;
/** two edges are only tested when an end of one is within this of an end of the other, m */
const END_M = 60;

export interface CorridorEdge {
  /** the corridor's id */
  id: number;
  /** metres from the corridor's middle to this edge's centre line, left of the corridor's direction */
  y: number;
  /** +1 when the edge's a to b runs the corridor's way, -1 against it */
  s: number;
}

export interface Corridors {
  of: Map<number, CorridorEdge>;
  /** per corridor id: its edges left to right */
  members: Map<number, number[]>;
  /** per corridor id: its direction in local units (x east, y south), unit */
  dir: Map<number, [number, number]>;
}

interface Prep { id: number; p: Float32Array; k: number; len: number; x0: number; y0: number; x1: number; y1: number; fp: number }
interface Pair { cover: number[]; off: number; rel: number }

/** pairs tested last time, keyed by the two edges; kept while both geometries are as they were */
let pairCache = new Map<number, { fa: number; fb: number; res: Pair | null }>();

/** each edge's measures, kept while its points array is the same one (the cache hands the same arrays back) */
const prepCache = new WeakMap<Float32Array, Prep | null>();

const prep = (id: number, p: Float32Array): Prep | null => {
  const old = prepCache.get(p);
  if (old && old.id === id && old.fp === fingerprint(p)) return old;
  if (old === null && p.length < 6) return old;
  const r = measure(id, p);
  prepCache.set(p, r);
  return r;
};

const measure = (id: number, p: Float32Array): Prep | null => {
  const n = p.length / 3;
  if (n < 2) return null;
  let len = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = p[i * 3], y = p[i * 3 + 1];
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
    if (i) len += Math.hypot(x - p[i * 3 - 3], y - p[i * 3 - 2]);
  }
  const k = kAtY((y0 + y1) / 2);
  return { id, p, k, len, x0, y0, x1, y1, fp: fingerprint(p) };
};

/** a number that changes when an edge's points move or are replaced (first, middle and last point) */
const fingerprint = (p: Float32Array) => {
  const n = p.length / 3;
  return n + p[0] * 1.3 + p[1] * 1.7 + p[n * 3 - 3] * 2.9 + p[n * 3 - 2] * 3.1 + p[(n >> 1) * 3] * 0.7 + p[(n >> 1) * 3 + 1] * 0.3;
};

/** how many of `a`'s samples have a parallel stretch of `b` within CORRIDOR_M, and (summed) the lateral
 * offset of that stretch left of `a`'s direction (m) and the sign of its direction against a's */
function against(a: Prep, b: Prep): { hits: number; n: number; off: number; dot: number } {
  const n = Math.max(3, Math.min(80, Math.ceil(a.len / a.k / SAMPLE_M)));
  const P = a.p, Q = b.p, nq = Q.length / 3, reach = CORRIDOR_M * a.k;
  let hits = 0, off = 0, dot = 0, seg = 0, at = 0, segLen = Math.hypot(P[3] - P[0], P[4] - P[1]);
  const maxMiss = Math.floor((1 - COVER) * n);
  let miss = 0;
  for (let i = 0; i < n; i++) {
    const want = ((i + 0.5) / n) * a.len;
    while (at + segLen < want && seg < P.length / 3 - 2) {
      at += segLen;
      seg++;
      segLen = Math.hypot(P[seg * 3 + 3] - P[seg * 3], P[seg * 3 + 4] - P[seg * 3 + 1]);
    }
    const f = segLen > 0 ? (want - at) / segLen : 0;
    const dx = P[seg * 3 + 3] - P[seg * 3], dy = P[seg * 3 + 4] - P[seg * 3 + 1];
    const px = P[seg * 3] + dx * f, py = P[seg * 3 + 1] + dy * f, dl = Math.hypot(dx, dy) || 1;
    const tx = dx / dl, ty = dy / dl;
    // the nearest point on b
    let best = Infinity, qx = 0, qy = 0, ux = 0, uy = 0;
    for (let j = 0; j + 1 < nq; j++) {
      const ax = Q[j * 3], ay = Q[j * 3 + 1], bx = Q[j * 3 + 3], by = Q[j * 3 + 4];
      const ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * ex + (py - ay) * ey) / l2)) : 0;
      const cx = ax + ex * t, cy = ay + ey * t, d = (cx - px) ** 2 + (cy - py) ** 2;
      if (d < best) {
        best = d;
        qx = cx;
        qy = cy;
        const el = Math.sqrt(l2) || 1;
        ux = ex / el;
        uy = ey / el;
      }
    }
    const c = tx * ux + ty * uy;
    if (best <= reach * reach && Math.abs(c) >= COS_MIN) {
      hits++;
      // left of a's direction, in local units (x east, y south) is (ty, -tx)
      off += ((qx - px) * ty - (qy - py) * tx) / a.k;
      dot += c > 0 ? 1 : -1;
    } else if (++miss > maxMiss) break;
  }
  return { hits, n, off: hits ? off / hits : 0, dot };
}

function pairOf(a: Prep, b: Prep): Pair | null {
  const x = against(a, b);
  if (x.hits < COVER * x.n) return null;
  const y = against(b, a);
  if (y.hits < COVER * y.n) return null;
  return { cover: [x.hits / x.n, y.hits / y.n], off: x.off, rel: x.dot >= 0 ? 1 : -1 };
}

/**
 * The corridors among `ids` (edges that carry lines). `pts`: each edge's local polyline (x, y, z
 * triples, a to b). Two edges are side by side when each has at least COVER of its length within
 * CORRIDOR_M of the other, running the same way or the opposite; side by side is passed on, so a
 * third track 8 m further out joins the corridor.
 */
export function findCorridors(ids: readonly number[], pts: ReadonlyMap<number, Float32Array>): Corridors {
  const out: Corridors = { of: new Map(), members: new Map(), dir: new Map() };
  const list: Prep[] = [];
  let kMax = 1;
  for (const id of ids) {
    const p = pts.get(id), e = p && prep(id, p);
    if (e) {
      list.push(e);
      if (e.k > kMax) kMax = e.k;
    }
  }
  if (list.length < 2) return out;
  // side by side edges end near each other (a four-track trunk's two pairs meet the same stations and
  // junctions), so the pairs to test are the ones with an end within END_M of each other: a grid of
  // the edges' two ends, each in the cells within END_M of it
  const cell = 3 * END_M * kMax, pad = END_M * kMax, grid = new Map<number, Prep[]>();
  const key = (ix: number, iy: number) => (ix + 1048576) * 2097152 + (iy + 1048576);
  for (const e of list) {
    const P = e.p, ends = [P[0], P[1], P[P.length - 3], P[P.length - 2]];
    const mine = new Set<number>();
    for (let q = 0; q < 4; q += 2) for (const dx of [-pad, pad]) for (const dy of [-pad, pad]) mine.add(key(Math.floor((ends[q] + dx) / cell), Math.floor((ends[q + 1] + dy) / cell)));
    for (const c of mine) {
      const v = grid.get(c);
      if (v) v.push(e);
      else grid.set(c, [e]);
    }
  }
  const reach = CORRIDOR_M * kMax;
  const seen = new Set<number>(), cache = new Map<number, { fa: number; fb: number; res: Pair | null }>();
  const nbr = new Map<number, { to: number; off: number; rel: number }[]>();
  for (const v of grid.values()) {
    for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) {
      let a = v[i], b = v[j];
      if (a.id > b.id) {
        const t = a;
        a = b;
        b = t;
      }
      if (a.x1 + reach < b.x0 || b.x1 + reach < a.x0 || a.y1 + reach < b.y0 || b.y1 + reach < a.y0) continue;
      // a very different length cannot have both covered
      if (a.len > 4 * b.len || b.len > 4 * a.len) continue;
      const pk = a.id * 33554432 + b.id;
      if (seen.has(pk)) continue;
      seen.add(pk);
      const old = pairCache.get(pk);
      let hit = old && old.fa === a.fp && old.fb === b.fp ? old : null;
      if (!hit) hit = { fa: a.fp, fb: b.fp, res: pairOf(a, b) };
      cache.set(pk, hit);
      const r = hit.res;
      if (!r) continue;
      for (const [from, to, off, rel] of [[a.id, b.id, r.off, r.rel], [b.id, a.id, -r.off * r.rel, r.rel]] as const) {
        let u = nbr.get(from);
        if (!u) nbr.set(from, (u = []));
        u.push({ to, off, rel });
      }
    }
  }
  pairCache = cache;
  const byId = new Map(list.map((e) => [e.id, e]));
  const done = new Set<number>();
  let next = 0;
  for (const start of [...nbr.keys()].sort((a, b) => a - b)) {
    if (done.has(start)) continue;
    // walk the component: each edge's place and direction against the first's
    const place = new Map<number, { y: number; s: number }>([[start, { y: 0, s: 1 }]]);
    const queue = [start];
    done.add(start);
    for (let q = 0; q < queue.length; q++) {
      const me = place.get(queue[q])!;
      for (const n of nbr.get(queue[q]) ?? []) {
        if (place.has(n.to)) continue;
        place.set(n.to, { y: me.y + me.s * n.off, s: me.s * n.rel });
        done.add(n.to);
        queue.push(n.to);
      }
    }
    const ys = [...place.values()].map((v) => v.y);
    const lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo > SPAN_M || place.size < 2) continue;
    const mid = (lo + hi) / 2, id = next++;
    const e0 = byId.get(start)!, p = e0.p, n = p.length / 3;
    const dx = p[n * 3 - 3] - p[0], dy = p[n * 3 - 2] - p[1], dl = Math.hypot(dx, dy) || 1;
    out.dir.set(id, [dx / dl, dy / dl]);
    const members = [...place.keys()].sort((a, b) => place.get(b)!.y - place.get(a)!.y || a - b);
    out.members.set(id, members);
    for (const e of members) out.of.set(e, { id, y: place.get(e)!.y - mid, s: place.get(e)!.s });
  }
  return out;
}
