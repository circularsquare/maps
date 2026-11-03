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

export interface GraphLine { id: number; ok: boolean; bus?: unknown; path: number[]; stops: number[]; colour: string }
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
  /** the middle the places are measured from, local units, and local units per metre there */
  mid: [number, number];
  k: number;
}

export interface DiagramLine {
  id: number;
  colour: string;
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
  const out: { id: number; colour: string; chain: number[]; stop: number[] }[] = [];
  const met: number[] = [];
  for (const l of rail) {
    met.length = 0;
    let arc = 0;
    l.path.forEach((code, k) => {
      const e = code >> 1, back = (code & 1) === 1, ed = ends.get(e);
      if (!ed) return;
      const from = back ? ed.b : ed.a, to = back ? ed.a : ed.b;
      if (k === 0 && isStation(from)) met.push(arc, from);
      const c = cumOf(e);
      const len = c ? c[c.length - 1] : 0;
      if (c) {
        const s = stationsBeside(e);
        for (let i = 0; i < s.length; i += 2) met.push(arc + (back ? len - s[i] : s[i]), s[i + 1]);
      }
      arc += len;
      if (isStation(to)) met.push(arc, to);
    });
    const idx = Array.from({ length: met.length / 2 }, (_, i) => i).sort((i, j) => met[i * 2] - met[j * 2] || i - j);
    const stopSet = new Set(l.stops.map((s) => (isStation(s) ? complexOf[s] : NO_COMPLEX)));
    const chain: number[] = [], at: number[] = [], stop: number[] = [], k0 = kAtY(met.length ? placeOf(met[1])?.[1] ?? 0 : 0);
    for (const i of idx) {
      const n = met[i * 2 + 1], a = met[i * 2];
      const g = graphNode(n);
      place(g, n);
      if (chain.length && chain[chain.length - 1] === g) continue;
      // A, B, A within a short way (a beside platform of A on either side of B): A once
      if (chain.length >= 2 && chain[chain.length - 2] === g && a - at[at.length - 2] < REVISIT_M * k0) continue;
      chain.push(g);
      at.push(a);
      stop.push(stopSet.has(complexOf[n]) ? 1 : 0);
    }
    // from the first stop to the last: stations beside it beyond its ends are not the line's
    let i0 = stop.indexOf(1), i1 = stop.lastIndexOf(1);
    if (i0 < 0) continue;
    out.push({ id: l.id, colour: l.colour.toLowerCase(), chain: chain.slice(i0, i1 + 1), stop: stop.slice(i0, i1 + 1) });
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
  const dl: DiagramLine[] = out.map((l) => ({ id: l.id, colour: l.colour, chain: Int32Array.from(l.chain, (g) => renum.get(g)!), stop: Uint8Array.from(l.stop) }));
  const em = new Map<number, number[]>();
  dl.forEach((l, li) => {
    for (let i = 1; i < l.chain.length; i++) {
      const a = Math.min(l.chain[i - 1], l.chain[i]), b = Math.max(l.chain[i - 1], l.chain[i]);
      if (a === b) continue;
      const kk = a * n + b;
      let v = em.get(kk);
      if (!v) em.set(kk, (v = []));
      if (v[v.length - 1] !== li) v.push(li);
    }
  });
  const ek = [...em.keys()].sort((a, b) => a - b);
  const ea = new Int32Array(ek.length), eb = new Int32Array(ek.length), first = new Int32Array(ek.length + 1);
  let total = 0;
  for (const kk of ek) total += em.get(kk)!.length;
  const on = new Int32Array(total);
  let o = 0;
  ek.forEach((kk, i) => {
    ea[i] = Math.floor(kk / n);
    eb[i] = kk % n;
    first[i] = o;
    for (const li of new Set(em.get(kk)!)) on[o++] = li;
  });
  first[ek.length] = o;
  return { key, x, y, lines: dl, ea, eb, first, on: on.subarray(0, o), mid: [mx, my], k };
}
