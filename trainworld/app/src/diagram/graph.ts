// The network diagram's graph (T-2159, notes/T-2146.md section 3 step 1): its nodes are station
// complexes, each line the chain of complexes it runs through, and an edge joins consecutive
// complexes of a chain with the lines on it. A line passes its own stops, the stations on its path
// and the stations whose platforms lie beside its track (within NEAR_M and parallel), found from
// corridors.ts' beside table, so an express on its own tracks shares the local's edges.

import { kAtY } from "../map/geo";
import { SAMPLE_M, type Beside } from "../map/corridors";

/** a platform this close to a line's track (m), and parallel to it, is passed by the line */
export const NEAR_M = 35;
/** |cos| between the platform's track and the line's */
const PARALLEL = 0.9;
/** how far along an edge its direction at a node is measured, m */
const DIR_M = 30;
/** a complex met again this soon after it (m along the line), with one other between, was met once */
const REVISIT_M = 300;
/** lines of one colour and speed share a stroke on a hop where this share of the shorter one's track there
 * is track both run on (T-2405) */
const SAME_TRACK = 0.5;
const NO_COMPLEX = 0xffffffff;
/** each edge's arc lengths, kept while the clock worker keeps its points (map/network.ts' cache) */
const arcs = new Map<number, { p: Float32Array; c: Float64Array }>();
/** the last index of ascending `c` at or below `v` (0 when none is) */
function below(c: Float64Array, v: number): number {
  let lo = 0, hi = c.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (c[mid] <= v) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export interface GraphLine { id: number; ok: boolean; bus?: unknown; path: number[]; stops: number[]; colour: string; fast?: boolean }
export interface GraphEdge { id: number; a: number; b: number }

export interface DiagramGraph {
  /** per node: its key, the lowest station node id among its members on the lines (stable across
   * edits, so a rebuild can start each node where it was) */
  key: Int32Array;
  /** per node: its place, metres east and north of the network's middle */
  x: Float64Array;
  y: Float64Array;
  lines: DiagramLine[];
  /** per edge: its nodes (a < b) and its lines (indices into `lines`) from first[e] to first[e + 1] */
  ea: Int32Array;
  eb: Int32Array;
  first: Int32Array;
  on: Int32Array;
  /** per entry of `on`, the stroke its line is drawn in there: the lowest line of those sharing it, lines of
   * one colour and speed on the same track (T-2405) */
  stroke: Int32Array;
  /** the middle the places are measured from, local units, and local units per metre there */
  mid: [number, number];
  k: number;
  /** per edge, its track on the ground from a to b as one of its lines runs it, simplified (T-2169: where it crosses
   * water): x, y pairs in metres as `x`, `y`, edge e's from hopAt[e] to hopAt[e + 1] (pairs) */
  hop?: Float32Array;
  hopAt?: Int32Array;
}

/** a hop's track is kept to within this of the ground's, metres */
const HOP_TOL_M = 15;

export interface DiagramLine {
  id: number;
  colour: string;
  /** high-speed trains (T-817), drawn with a centre stripe (T-2272) */
  fast?: boolean;
  /** the nodes it runs through in order, first stop to last */
  chain: Int32Array;
  /** per chain entry, 1 where the line stops */
  stop: Uint8Array;
}

/** The rail lines' edges in the order `lineBands` hands them to `besideSamples`, so a call with
 * them is answered from the table the bands just built. */
export function bandEdges(lines: readonly GraphLine[]): number[] {
  const seen = new Set<number>();
  const rail = lines.filter((l) => l.ok && !l.bus && l.path.length).sort((a, b) => a.id - b.id);
  for (const l of rail) for (const p of l.path) seen.add(p >> 1);
  return [...seen];
}

/**
 * The graph. `pts`: each edge's local polyline (x, y, z triples, a to b); `complexOf`: each node's
 * complex, NO_COMPLEX for a node with no platform; `beside`: corridors.ts' table over `bandEdges`.
 */
export function diagramGraph(lines: readonly GraphLine[], edges: readonly GraphEdge[], pts: ReadonlyMap<number, Float32Array>, complexOf: Uint32Array, beside: Beside): DiagramGraph {
  const ends = new Map<number, GraphEdge>();
  for (const e of edges) ends.set(e.id, e);
  const isStation = (n: number) => n < complexOf.length && complexOf[n] !== NO_COMPLEX;
  /** a node's place, from the end of an edge it is on */
  const nodeAt = new Map<number, [number, number]>();
  const placeOf = (n: number): [number, number] | undefined => nodeAt.get(n);
  /** an edge's arc length to each point, local units */
  const cumOf = (e: number): Float64Array | null => {
    const p = pts.get(e);
    if (!p || p.length < 6) return null;
    const had = arcs.get(e);
    if (had && had.p === p) return had.c;
    let c: Float64Array;
    const m = p.length / 3;
    c = new Float64Array(m);
    for (let i = 1; i < m; i++) {
      const dx = p[i * 3] - p[i * 3 - 3], dy = p[i * 3 + 1] - p[i * 3 - 2];
      c[i] = c[i - 1] + Math.sqrt(dx * dx + dy * dy);
    }
    arcs.set(e, { p, c });
    return c;
  };
  for (const e of edges) {
    const p = pts.get(e.id);
    if (!p || p.length < 6) continue;
    if (isStation(e.a) && !nodeAt.has(e.a)) nodeAt.set(e.a, [p[0], p[1]]);
    if (isStation(e.b) && !nodeAt.has(e.b)) nodeAt.set(e.b, [p[p.length - 3], p[p.length - 2]]);
  }
  /** the unit direction edge `e` leaves node `n` in, local units */
  const dirAt = (e: number, n: number): [number, number] | null => {
    const p = pts.get(e), ed = ends.get(e), c = cumOf(e);
    if (!p || !ed || !c) return null;
    const m = p.length / 3, fromA = ed.a === n, at = (i: number) => (fromA ? i : m - 1 - i);
    const reach = DIR_M * kAtY(p[1]);
    let i = 1;
    while (i < m - 1 && Math.abs(c[at(i)] - c[at(0)]) < reach) i++;
    const dx = p[at(i) * 3] - p[at(0) * 3], dy = p[at(i) * 3 + 1] - p[at(0) * 3 + 1], d = Math.sqrt(dx * dx + dy * dy);
    return d > 0 ? [dx / d, dy / d] : null;
  };
  /** per edge with lines: the stations beside it that are not its own ends, [arc from a (local
   * units), node] pairs. Each station's own edges' end samples there say which edges run beside it,
   * and at which of their samples; it is measured against each such edge's polyline round that one. */
  const besideOf = new Map<number, number[]>();
  const tried = new Set<number>();
  for (const [f, F] of beside.edges) {
    const fe = ends.get(f);
    if (!fe || F.first[F.n] === 0) continue;
    for (let side = 0; side < 2; side++) {
      const n = side === 0 ? fe.a : fe.b, s = side === 0 ? 0 : F.n - 1;
      if (!isStation(n)) continue;
      const q = placeOf(n);
      if (!q) continue;
      for (let j = F.first[s]; j < F.first[s + 1]; j++) {
        const e = F.edge[j], ed = ends.get(e), S = beside.edges.get(e), p = pts.get(e), c = cumOf(e);
        if (!ed || !S || !p || !c || n === ed.a || n === ed.b) continue;
        const tk = e * 4294967296 + n;
        if (tried.has(tk)) continue;
        tried.add(tk);
        const m = p.length / 3, kq = kAtY(q[1]), near = NEAR_M * kq, reach = (SAMPLE_M + NEAR_M) * kq;
        const want = ((F.at[j] + 0.5) / S.n) * c[m - 1];
        const i0 = Math.max(0, below(c, want - reach) - 1), i1 = Math.min(m - 2, below(c, want + reach));
        let best = Infinity, arc = 0, ux = 0, uy = 0;
        for (let i = i0; i <= i1; i++) {
          const ax = p[i * 3], ay = p[i * 3 + 1], dx = p[i * 3 + 3] - ax, dy = p[i * 3 + 4] - ay, l2 = dx * dx + dy * dy;
          if (!(l2 > 0)) continue;
          const t = Math.max(0, Math.min(1, ((q[0] - ax) * dx + (q[1] - ay) * dy) / l2));
          const d = (ax + dx * t - q[0]) ** 2 + (ay + dy * t - q[1]) ** 2;
          if (d < best) {
            best = d;
            const l = Math.sqrt(l2);
            arc = c[i] + t * l;
            ux = dx / l;
            uy = dy / l;
          }
        }
        if (best > near * near) continue;
        const g = dirAt(f, n);
        if (!g || Math.abs(g[0] * ux + g[1] * uy) < PARALLEL) continue;
        let v = besideOf.get(e);
        if (!v) besideOf.set(e, (v = []));
        v.push(arc, n);
      }
    }
  }
  const NONE: number[] = [];
  const stationsBeside = (e: number): number[] => besideOf.get(e) ?? NONE;

  // the complexes met, as graph nodes
  const nodeOf = new Map<number, number>();
  const keys: number[] = [], sx: number[] = [], sy: number[] = [], cnt: number[] = [];
  const graphNode = (n: number): number => {
    const cx = complexOf[n];
    let g = nodeOf.get(cx);
    if (g === undefined) {
      nodeOf.set(cx, (g = keys.length));
      keys.push(n);
      sx.push(0), sy.push(0), cnt.push(0);
    } else if (n < keys[g]) keys[g] = n;
    return g;
  };
  const placed = new Set<number>();
  const place = (g: number, n: number) => {
    if (placed.has(n)) return;
    const q = placeOf(n);
    if (!q) return;
    placed.add(n);
    sx[g] += q[0];
    sy[g] += q[1];
    cnt[g]++;
  };

  const rail = lines.filter((l) => l.ok && !l.bus && l.path.length >= 1 && l.stops.length >= 2);
  /** per line: its chain, and per chain entry the range of its path the hops either side run on: the hop
   * from entry i to i + 1 runs on path[lo[i]] to path[hi[i + 1]] */
  type Out = { id: number; colour: string; fast: boolean; path: number[]; chain: number[]; stop: number[]; lo: number[]; hi: number[]; at: number[] };
  const out: Out[] = [];
  /** met: arc, node, lo, hi */
  const met: number[] = [];
  for (const l of rail) {
    met.length = 0;
    let arc = 0;
    l.path.forEach((code, k) => {
      const e = code >> 1, back = (code & 1) === 1, ed = ends.get(e);
      if (!ed) return;
      const from = back ? ed.b : ed.a, to = back ? ed.a : ed.b;
      if (k === 0 && isStation(from)) met.push(arc, from, 0, -1);
      const c = cumOf(e);
      const len = c ? c[c.length - 1] : 0;
      if (c) {
        const s = stationsBeside(e);
        for (let i = 0; i < s.length; i += 2) met.push(arc + (back ? len - s[i] : s[i]), s[i + 1], k, k);
      }
      arc += len;
      if (isStation(to)) met.push(arc, to, k + 1, k);
    });
    const idx = Array.from({ length: met.length / 4 }, (_, i) => i).sort((i, j) => met[i * 4] - met[j * 4] || i - j);
    const stopSet = new Set(l.stops.map((s) => (isStation(s) ? complexOf[s] : NO_COMPLEX)));
    const chain: number[] = [], at: number[] = [], stop: number[] = [], lo: number[] = [], hi: number[] = [], k0 = kAtY(met.length ? placeOf(met[1])?.[1] ?? 0 : 0);
    for (const i of idx) {
      const n = met[i * 4 + 1], a = met[i * 4];
      const g = graphNode(n);
      place(g, n);
      if (chain.length && chain[chain.length - 1] === g) {
        lo[lo.length - 1] = met[i * 4 + 2];
        continue;
      }
      // A, B, A within a short way (a beside platform of A on either side of B): A once
      if (chain.length >= 2 && chain[chain.length - 2] === g && a - at[at.length - 2] < REVISIT_M * k0) continue;
      chain.push(g);
      at.push(a);
      stop.push(stopSet.has(complexOf[n]) ? 1 : 0);
      lo.push(met[i * 4 + 2]);
      hi.push(met[i * 4 + 3]);
    }
    // from the first stop to the last: stations beside it beyond its ends are not the line's
    let i0 = stop.indexOf(1), i1 = stop.lastIndexOf(1);
    if (i0 < 0) continue;
    const cut = <T>(v: T[]) => v.slice(i0, i1 + 1);
    out.push({ id: l.id, colour: l.colour.toLowerCase(), fast: !!l.fast, path: l.path, chain: cut(chain), stop: cut(stop), lo: cut(lo), hi: cut(hi), at: cut(at) });
  }

  // nodes on some kept chain only, numbered again in order of their keys
  const used = new Set<number>();
  for (const l of out) for (const g of l.chain) used.add(g);
  const order = [...used].sort((a, b) => keys[a] - keys[b]);
  const renum = new Map(order.map((g, i) => [g, i]));
  const n = order.length;
  let mx = 0, my = 0;
  const px = new Float64Array(n), py = new Float64Array(n);
  order.forEach((g, i) => {
    px[i] = sx[g] / Math.max(1, cnt[g]);
    py[i] = sy[g] / Math.max(1, cnt[g]);
    mx += px[i];
    my += py[i];
  });
  mx /= Math.max(1, n);
  my /= Math.max(1, n);
  const k = kAtY(my);
  const key = new Int32Array(n), x = new Float64Array(n), y = new Float64Array(n);
  order.forEach((g, i) => {
    key[i] = keys[g];
    x[i] = (px[i] - mx) / k;
    // local y points south
    y[i] = -(py[i] - my) / k;
  });
  const dl: DiagramLine[] = out.map((l) => ({ id: l.id, colour: l.colour, ...(l.fast ? { fast: true } : {}), chain: Int32Array.from(l.chain, (g) => renum.get(g)!), stop: Uint8Array.from(l.stop) }));
  const em = new Map<number, number[]>();
  /** per edge, per line on it, the track edges its hops there run on */
  const runs = new Map<number, Map<number, Set<number>>>();
  dl.forEach((l, li) => {
    const src = out[li];
    for (let i = 1; i < l.chain.length; i++) {
      const a = Math.min(l.chain[i - 1], l.chain[i]), b = Math.max(l.chain[i - 1], l.chain[i]);
      if (a === b) continue;
      const kk = a * n + b;
      let v = em.get(kk);
      if (!v) em.set(kk, (v = []));
      if (v[v.length - 1] !== li) v.push(li);
      let r = runs.get(kk);
      if (!r) runs.set(kk, (r = new Map()));
      let s = r.get(li);
      if (!s) r.set(li, (s = new Set()));
      for (let p = Math.max(0, src.lo[i - 1]); p <= Math.min(src.hi[i], src.path.length - 1); p++) s.add(src.path[p] >> 1);
    }
  });
  const lenOf = (e: number) => { const c = cumOf(e); return c ? c[c.length - 1] : 0; };
  /** whether two lines' hops on an edge run on the same track */
  const sameTrack = (s: Set<number>, t: Set<number>): boolean => {
    let both = 0, ls = 0, lt = 0;
    for (const e of s) (ls += lenOf(e)), t.has(e) && (both += lenOf(e));
    for (const e of t) lt += lenOf(e);
    return both > 0 && both >= SAME_TRACK * Math.min(ls, lt);
  };
  const ek = [...em.keys()].sort((a, b) => a - b);
  const ea = new Int32Array(ek.length), eb = new Int32Array(ek.length), first = new Int32Array(ek.length + 1);
  let total = 0;
  for (const kk of ek) total += em.get(kk)!.length;
  const on = new Int32Array(total), stroke = new Int32Array(total);
  let o = 0;
  ek.forEach((kk, i) => {
    ea[i] = Math.floor(kk / n);
    eb[i] = kk % n;
    first[i] = o;
    const here = [...new Set(em.get(kk)!)], r = runs.get(kk)!;
    // each line's stroke: the lowest line of one colour and speed reached through hops on the same track
    const up = here.map((_, q) => q);
    const find = (q: number): number => (up[q] === q ? q : (up[q] = find(up[q])));
    for (let a = 0; a < here.length; a++) for (let b = a + 1; b < here.length; b++) {
      const la = dl[here[a]], lb = dl[here[b]];
      if (la.colour !== lb.colour || !!la.fast !== !!lb.fast || !sameTrack(r.get(here[a])!, r.get(here[b])!)) continue;
      const ra = find(a), rb = find(b);
      if (ra !== rb) up[Math.max(ra, rb)] = Math.min(ra, rb);
    }
    for (let q = 0; q < here.length; q++) {
      on[o] = here[q];
      // `here` is in line order, so a root is its group's lowest line
      stroke[o++] = here[find(q)];
    }
  });
  first[ek.length] = o;
  const { hop, hopAt } = hopTracks(ek, n, dl, out, ends, pts, cumOf, mx, my, k);
  return { key, x, y, lines: dl, ea, eb, first, on: on.subarray(0, o), stroke: stroke.subarray(0, o), mid: [mx, my], k, hop, hopAt };
}

/** per edge (keys a * n + b in `ek`'s order), the track of the first line on it between its two nodes, in metres
 * from the middle (mx, my), simplified to HOP_TOL_M */
function hopTracks(ek: number[], n: number, dl: DiagramLine[], out: { path: number[]; at: number[] }[], ends: Map<number, GraphEdge>, pts: ReadonlyMap<number, Float32Array>, cumOf: (e: number) => Float64Array | null, mx: number, my: number, k: number): { hop: Float32Array; hopAt: Int32Array } {
  const index = new Map(ek.map((kk, i) => [kk, i]));
  const per: (number[] | undefined)[] = new Array(ek.length);
  dl.forEach((l, li) => {
    const src = out[li];
    // each path entry's start along the line, local units
    const start: number[] = [];
    let arc = 0;
    for (const code of src.path) {
      start.push(arc);
      const c = cumOf(code >> 1);
      arc += c ? c[c.length - 1] : 0;
    }
    for (let i = 1; i < l.chain.length; i++) {
      const u = l.chain[i - 1], w = l.chain[i];
      if (u === w) continue;
      const e = index.get(Math.min(u, w) * n + Math.max(u, w));
      if (e === undefined || per[e]) continue;
      const A = src.at[i - 1], B = src.at[i], p: number[] = [];
      const push = (x: number, y: number) => p.push((x - mx) / k, -(y - my) / k);
      for (let q = 0; q < src.path.length; q++) {
        const code = src.path[q], pe = pts.get(code >> 1), c = cumOf(code >> 1);
        if (!pe || !c || !ends.has(code >> 1)) continue;
        const len = c[c.length - 1], s0 = start[q];
        if (s0 >= B || s0 + len <= A) continue;
        const m = pe.length / 3, back = (code & 1) === 1;
        for (let j = 0; j < m; j++) {
          const jj = back ? m - 1 - j : j, s = s0 + (back ? len - c[jj] : c[jj]);
          if (s >= A && s <= B) push(pe[jj * 3], pe[jj * 3 + 1]);
        }
      }
      if (p.length < 4) continue;
      if (u > w) for (let a = 0, b = p.length - 2; a < b; a += 2, b -= 2) [p[a], p[a + 1], p[b], p[b + 1]] = [p[b], p[b + 1], p[a], p[a + 1]];
      per[e] = simplifyLine(p, HOP_TOL_M);
    }
  });
  const hopAt = new Int32Array(ek.length + 1);
  for (let e = 0; e < ek.length; e++) hopAt[e + 1] = hopAt[e] + (per[e]?.length ?? 0) / 2;
  const hop = new Float32Array(hopAt[ek.length] * 2);
  per.forEach((p, e) => p && hop.set(p, hopAt[e] * 2));
  return { hop, hopAt };
}

/** a polyline (x, y pairs) with the points within `tol` of the way between their kept neighbours dropped (Douglas and
 * Peucker), its ends kept */
export function simplifyLine(p: number[], tol: number): number[] {
  const m = p.length / 2;
  if (m <= 2) return p;
  const keep = new Uint8Array(m);
  (keep[0] = 1), (keep[m - 1] = 1);
  const stack = [0, m - 1];
  while (stack.length) {
    const b = stack.pop()!, a = stack.pop()!;
    const ax = p[a * 2], ay = p[a * 2 + 1], dx = p[b * 2] - ax, dy = p[b * 2 + 1] - ay, l2 = dx * dx + dy * dy;
    let worst = -1, wi = -1;
    for (let i = a + 1; i < b; i++) {
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((p[i * 2] - ax) * dx + (p[i * 2 + 1] - ay) * dy) / l2)) : 0;
      const ex = p[i * 2] - ax - dx * t, ey = p[i * 2 + 1] - ay - dy * t, dd = ex * ex + ey * ey;
      if (dd > worst) (worst = dd), (wi = i);
    }
    if (worst > tol * tol) (keep[wi] = 1), stack.push(a, wi, wi, b);
  }
  const out: number[] = [];
  for (let i = 0; i < m; i++) if (keep[i]) out.push(p[i * 2], p[i * 2 + 1]);
  return out;
}
