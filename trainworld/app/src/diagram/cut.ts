// T-3409: the schematic of chosen cities. Each line is cut to its runs of stops inside them (the page says which
// stations and bus stops are, by the zone each is in), each run a line of its own with the line's id, its path and stops
// from the run's first stop to its last, and at an end where the line goes on, the way it goes (`cutA`, `cutB`) for the
// stub drawn there. Stations outside are no stations to the graph, so nothing passed or beside outside is drawn. The
// graph is then built from these runs and their track alone, exactly as a network of only them would be. Pure: no DOM.

/** how far along the track past a cut end its way is measured (local units, about a metre each): the next stop if nearer */
const PROBE = 3000;
const NO_COMPLEX = 0xffffffff;

/** a line as the clock worker has it (protocol.ts `ClockLine`), the fields the cut reads */
export interface CutLineIn {
  id: number;
  ok: boolean;
  path: number[];
  stops: number[];
  bus?: { routeLL: number[]; stops: { id: number; lng: number; lat: number; station: number }[] };
}
/** a run: the line's fields with its own path and stops, and where it starts in the line's stops */
export type CutRun<L> = L & { from?: number; cutA?: [number, number]; cutB?: [number, number] };

export interface CutResult<L> {
  lines: CutRun<L>[];
  /** the runs' track, the edges' points as given */
  pts: Map<number, Float32Array>;
  /** `complexOf` with every station outside the chosen cities made no station */
  complexOf: Uint32Array;
}

/** a unit way (east, north) from (x0, y0) to (x1, y1), local units whose y points south; null where they meet */
function way(x0: number, y0: number, x1: number, y1: number): [number, number] | null {
  const dx = x1 - x0, dy = y0 - y1, l = Math.hypot(dx, dy);
  return l > 1e-6 ? [dx / l, dy / l] : null;
}

/**
 * The runs of `lines` inside the chosen cities. `inside`: the station nodes in them (a complex is inside when any of its
 * members is); `insideStops`: the bus and ferry stops in them, by id. `ends` each edge's nodes, `pts` each edge's local
 * points (x, y, z triples from a to b). Null when no run has two stops, so the caller draws the whole network.
 */
export function cutLines<L extends CutLineIn>(lines: readonly L[], ends: ReadonlyMap<number, { a: number; b: number }>, pts: ReadonlyMap<number, Float32Array>, complexOf: Uint32Array, inside: ReadonlySet<number>, insideStops: ReadonlySet<number>): CutResult<L> | null {
  const inCx = new Set<number>();
  for (const n of inside) if (n < complexOf.length && complexOf[n] !== NO_COMPLEX) inCx.add(complexOf[n]);
  const cx = complexOf.slice();
  for (let n = 0; n < cx.length; n++) if (cx[n] !== NO_COMPLEX && !inCx.has(cx[n])) cx[n] = NO_COMPLEX;
  const isIn = (s: number) => s < complexOf.length && complexOf[s] !== NO_COMPLEX ? inCx.has(complexOf[s]) : inside.has(s);
  const out: CutRun<L>[] = [], track = new Map<number, Float32Array>();
  let any = false;
  const take = (l: CutRun<L>) => {
    out.push(l);
    if (l.bus) { if (l.stops.length >= 2) any = true; return; }
    if (l.ok && l.stops.length >= 2) any = true;
    for (const p of l.path) { const q = pts.get(p >> 1); if (q) track.set(p >> 1, q); }
  };
  for (const l of lines) {
    if (l.bus) {
      const b = l.bus, runs = busRuns(b, (s) => (s.station >= 0 ? isIn(s.station) : insideStops.has(s.id)));
      if (runs === "all") take(l);
      else for (const r of runs) take({ ...l, stops: r.stops.map((s) => s.id), bus: { ...b, stops: r.stops, routeLL: r.routeLL }, from: r.from, ...(r.cutA ? { cutA: r.cutA } : {}), ...(r.cutB ? { cutB: r.cutB } : {}) } as CutRun<L>);
      continue;
    }
    const n = l.stops.length;
    const runs = runsOf(n, (i) => isIn(l.stops[i]));
    if (!runs.length) continue;
    if (runs.length === 1 && runs[0][0] === 0 && runs[0][1] === n - 1) { take(l); continue; }
    // the nodes along the path, and each stop's place among them (matched by complex, in order)
    const nodes: number[] = [];
    for (let k = 0; k < l.path.length; k++) {
      const e = ends.get(l.path[k] >> 1);
      if (!e) { nodes.length = 0; break; }
      const back = (l.path[k] & 1) === 1;
      if (!k) nodes.push(back ? e.b : e.a);
      nodes.push(back ? e.a : e.b);
    }
    const same = (a: number, b: number) => a === b || (a < complexOf.length && b < complexOf.length && complexOf[a] !== NO_COMPLEX && complexOf[a] === complexOf[b]);
    const at = new Int32Array(n).fill(-1);
    for (let s = 0, k = 0; s < n && nodes.length; s++) {
      while (k < nodes.length && !same(nodes[k], l.stops[s])) k++;
      if (k >= nodes.length) break;
      at[s] = k;
      // a next stop in the same complex may be met at the same node
      if (s + 1 < n && !same(l.stops[s + 1], l.stops[s])) k++;
    }
    if (at.some((k) => k < 0)) continue; // a stop off its path: left out, the line drawn whole only when all inside (above)
    /** the way the track goes from path node k, walking `step` (+1 or -1) up to node `to` or about PROBE */
    const goes = (k: number, step: number, to: number): [number, number] | null => {
      let x0 = NaN, y0 = 0, fx = NaN, fy = 0;
      for (let q = step > 0 ? k : k - 1; step > 0 ? q < to : q >= to; q += step) {
        const code = l.path[q], p = pts.get(code >> 1);
        if (!p || p.length < 6) continue;
        const m = p.length / 3, rev = ((code & 1) === 1) !== (step < 0);
        for (let j = 0; j < m; j++) {
          const jj = rev ? m - 1 - j : j, x = p[jj * 3], y = p[jj * 3 + 1];
          if (Number.isNaN(x0)) (x0 = x), (y0 = y);
          (fx = x), (fy = y);
          if (Math.hypot(x - x0, y - y0) >= PROBE) return way(x0, y0, fx, fy);
        }
      }
      return Number.isNaN(x0) ? null : way(x0, y0, fx, fy);
    };
    for (const [i, j] of runs) {
      if (j === i) continue;
      const run = { ...l, stops: l.stops.slice(i, j + 1), path: l.path.slice(at[i], at[j]), from: i } as CutRun<L>;
      const A = i > 0 ? goes(at[i], -1, at[i - 1]) : null, B = j < n - 1 ? goes(at[j], 1, at[j + 1]) : null;
      if (A) run.cutA = A;
      if (B) run.cutB = B;
      take(run);
    }
  }
  return any ? { lines: out, pts: track, complexOf: cx } : null;
}

/** the runs of stop indices whose stops `ok` says are inside, as [first, last] */
function runsOf(n: number, ok: (i: number) => boolean): [number, number][] {
  const r: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    if (!ok(i)) continue;
    let j = i;
    while (j + 1 < n && ok(j + 1)) j++;
    r.push([i, j]);
    i = j;
  }
  return r;
}

type BusStop = { lng: number; lat: number };
/** A bus or ferry line's runs of two stops or more inside (`ok`), each with its stops, its route from its first stop to its
 * last (longitude, latitude pairs), where it starts in the line's stops and the ways it goes on; "all" when every stop is
 * inside. Also the page's, for the buses drawn thin (view.ts). */
export function busRuns<S extends BusStop>(b: { routeLL: number[]; stops: S[] }, ok: (s: S) => boolean): "all" | { stops: S[]; routeLL: number[]; from: number; cutA?: [number, number]; cutB?: [number, number] }[] {
  const runs = runsOf(b.stops.length, (i) => ok(b.stops[i]));
  if (runs.length === 1 && runs[0][0] === 0 && runs[0][1] === b.stops.length - 1) return "all";
  // each stop's nearest route point, found from the last one's on (busGraph.ts)
  const at: number[] = [];
  let from = 0;
  for (const s of b.stops) {
    let best = Infinity, bi = from;
    for (let i = from; i + 1 < b.routeLL.length; i += 2) {
      const d = (b.routeLL[i] - s.lng) ** 2 + (b.routeLL[i + 1] - s.lat) ** 2;
      if (d < best) (best = d), (bi = i);
    }
    at.push((from = bi));
  }
  /** the way the route goes from stop i, walking `step` (+2 or -2) up to stop `to` or about PROBE metres */
  const goes = (i: number, step: number, to: number): [number, number] | null => {
    const r = b.routeLL, x0 = b.stops[i].lng, y0 = b.stops[i].lat, c = Math.cos((y0 * Math.PI) / 180), lim = (PROBE / 111_320) ** 2;
    let q = at[i];
    for (let k = at[i] + step; step > 0 ? k <= at[to] : k >= at[to]; k += step) {
      q = k;
      if (((r[k] - x0) * c) ** 2 + (r[k + 1] - y0) ** 2 >= lim) break;
    }
    let dx = (r[q] - x0) * c, dy = r[q + 1] - y0;
    // the route ends at the stop: straight at the next stop
    if (Math.hypot(dx, dy) < 1e-12) (dx = (b.stops[to].lng - x0) * c), (dy = b.stops[to].lat - y0);
    const d = Math.hypot(dx, dy);
    return d > 1e-12 ? [dx / d, dy / d] : null;
  };
  const out: { stops: S[]; routeLL: number[]; from: number; cutA?: [number, number]; cutB?: [number, number] }[] = [];
  for (const [i, j] of runs) {
    if (j === i) continue;
    const lo = Math.min(at[i], at[j]), hi = Math.max(at[i], at[j]);
    const run: (typeof out)[number] = { stops: b.stops.slice(i, j + 1), routeLL: b.routeLL.slice(lo, hi + 2), from: i };
    const A = i > 0 ? goes(i, -2, i - 1) : null, B = j < b.stops.length - 1 ? goes(j, 2, j + 1) : null;
    if (A) run.cutA = A;
    if (B) run.cutB = B;
    out.push(run);
  }
  return out;
}
