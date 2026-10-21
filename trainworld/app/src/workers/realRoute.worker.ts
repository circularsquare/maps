// The real-network router (T-727d, notes/T-727.md): the track tool's snapped route between two
// clicks, off the main thread and off the clock worker. It loads each region's graph
// (`game/realGraph.ts`) when a click or the view first reaches it, and a neighbour's when the search
// reaches a junction the two files share, joined there by OSM node id. Then the fit
// (`TrackApi.trace_route`, sim/src/track/trace.rs) on a TrackApi of its own: the fit needs only the
// world frame, and a long one (New York to Emeryville) would hold up the clock for a second.
//
// The search is A* over chains in each direction: shortest by length, a turn of more than 60
// degrees where two chains meet refused (no running in one branch of a turnout and out the other),
// the other family than the first click's nearest track (metro or rail) at three times its length,
// high-speed chains at HSR of theirs (T-838), and every chain within the snap radius of each click
// a candidate, its distance from the click counted twice. It gives up after 2 s or 3 million states.
//
// The reach (T-839): every chain the search could get to from the first click under the same turn
// rule, sent for the part of it in view so the map can light it up.

import init, { TrackApi } from "../wasm/track/sim";
import { NEW_GAME_FRAME } from "../game/cities";
import { frameKindNumber } from "../game/frame";
import { realGraph, type RealGraph } from "../game/realGraph";

export type ToRouter =
  | { kind: "init"; page: string; graphs: string }
  /** the view: the graphs of the regions in it are fetched ahead of a click */
  | { kind: "view"; w: number; s: number; e: number; n: number }
  /** `ra`, `rb`: the snap radius at each click, ground m */
  | { kind: "route"; seq: number; a: [number, number]; b: [number, number]; ra: number; rb: number; tracks: number; level: number; tol: number }
  /** the track reachable from a first click at `a` (snap radius `ra`), in view box `box` (w, s, e,
   * n), vertices kept `step` degrees apart; null `a` drops it */
  | { kind: "reach"; seq: number; a: [number, number] | null; ra: number; box: [number, number, number, number]; step: number };

export type FromRouter =
  /** the real line, lon, lat pairs; `why` when there is none: "off" no real track within reach of
   * a click (or no graph there), "none" nothing joins them */
  | { kind: "line"; seq: number; line: Float64Array | null; lengthM: number; why?: "off" | "none"; ms: number; states: number }
  /** the fitted corners in `add_route`'s `pis` layout (x, y, radius, level each), null if the fit failed */
  | { kind: "fit"; seq: number; pis: Float64Array | null; worst: number; mean: number; ms: number }
  /** the reachable chains in view: chain i's lon, lat pairs are `xy[off[i] * 2]` up to
   * `xy[off[i + 1] * 2]`, its class `cls[i]` (0 main line, 1 other rail, 2 metro); `chains` reachable
   * in all */
  | { kind: "reach"; seq: number; xy: Float32Array; off: Uint32Array; cls: Uint8Array; chains: number; ms: number };

const M_PER_DEG = 6371008.8 * Math.PI / 180;
const TURN_MAX = 60 * Math.PI / 180;
const OTHER_FAMILY = 3;
const SEARCH_MS = 2000;
const SEARCH_STATES = 3e6;
/** the headings at a chain's ends are taken this far into it, m */
const HEADING_M = 30;
/** the candidate index's cell, degrees */
const CELL = 0.01;
/** a border junction looks for a neighbour's graph within this, degrees */
const BORDER_DEG = 0.05;
/** a high-speed chain's cost per m of it, so a long snap keeps to the LGV or Shinkansen over a
 * slightly shorter classic line (T-838); the heuristic is scaled by it to stay a lower bound */
const HSR = 0.8;

const post = (m: FromRouter, t: Transferable[] = []) => (self as unknown as Worker).postMessage(m, t);

// ---------------------------------------------------------------- regions

interface Outline { cc: string; rings: Float64Array[]; boxes: [number, number, number, number][] }
let outlines: Promise<Outline[]> | null = null;
let pageBase = "", graphBase = "";

function regionOutlines(): Promise<Outline[]> {
  return (outlines ??= (async () => {
    for (const b of [new URL("../noritetsu/", pageBase).href, "https://data.anita.garden/noritetsu/"]) {
      try {
        const r = await fetch(b + "regions.json");
        const j = r.ok ? await r.json() : null;
        if (!j?.regions) continue;
        return Object.entries(j.regions as Record<string, { parts: [number, number][][] }>).map(([cc, r]) => {
          const rings = r.parts.map((ring) => Float64Array.from(ring.flat()));
          const boxes = rings.map((ring) => {
            let w = 180, s = 90, e = -180, n = -90;
            for (let k = 0; k < ring.length; k += 2) (w = Math.min(w, ring[k])), (e = Math.max(e, ring[k])), (s = Math.min(s, ring[k + 1])), (n = Math.max(n, ring[k + 1]));
            return [w, s, e, n] as [number, number, number, number];
          });
          return { cc, rings, boxes };
        });
      } catch {
        // not there: the next
      }
    }
    return [];
  })());
}

function inRing(r: Float64Array, x: number, y: number) {
  let inside = false;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2) {
    const yi = r[i + 1], yj = r[j + 1];
    if (yi > y !== yj > y && x < ((r[j] - r[i]) * (y - yi)) / (yj - yi) + r[i]) inside = !inside;
  }
  return inside;
}

/** The regions whose outline holds the point, else those whose box comes within `pad` degrees. */
async function regionsAt(lon: number, lat: number, pad: number): Promise<string[]> {
  const all = await regionOutlines();
  const inside = all.filter((o) => o.rings.some((r, i) => { const b = o.boxes[i]; return lon >= b[0] && lon <= b[2] && lat >= b[1] && lat <= b[3] && inRing(r, lon, lat); }));
  if (inside.length && pad === 0) return inside.map((o) => o.cc);
  const p = Math.max(pad, 0.02);
  return all.filter((o) => o.boxes.some((b) => lon >= b[0] - p && lon <= b[2] + p && lat >= b[1] - p && lat <= b[3] + p)).map((o) => o.cc);
}

// ---------------------------------------------------------------- the joined graph

/** Every loaded region's chains in one numbering, junctions shared across a border made one. */
const net = {
  J: 0,
  C: 0,
  jLon: new Float64Array(1 << 16),
  jLat: new Float64Array(1 << 16),
  /** a junction another region's file shares */
  border: new Uint8Array(1 << 16),
  ca: new Uint32Array(1 << 16),
  cb: new Uint32Array(1 << 16),
  /** 0 rail, 1 metro */
  fam: new Uint8Array(1 << 16),
  /** the cost of a m of it: 1, or HSR on high-speed track */
  per: new Float32Array(1 << 16),
  /** 0 main line, 1 other rail, 2 metro, as the layer colours them */
  cls: new Uint8Array(1 << 16),
  /** bounding box, w s e n */
  box: new Float32Array(1 << 18),
  len: new Float64Array(1 << 16),
  /** heading leaving a into the chain, and arriving at b along it (radians, east 0, north pi/2) */
  ha: new Float32Array(1 << 16),
  hb: new Float32Array(1 << 16),
  graph: [] as RealGraph[],
  /** each chain's graph, and its number there */
  cg: new Uint16Array(1 << 16),
  cl: new Uint32Array(1 << 16),
  /** the incident chain ends of each junction: chain * 2 (its a end) or chain * 2 + 1 (its b end) */
  adjStart: new Uint32Array(1),
  adj: new Uint32Array(0),
  osm: new Map<number, number>(),
  grid: new Map<number, number[]>(),
};
const tried = new Map<string, Promise<boolean>>();

function grow<T extends Float64Array | Float32Array | Int32Array | Uint32Array | Uint16Array | Uint8Array>(a: T, n: number): T {
  if (a.length >= n) return a;
  const b = new (a.constructor as new (n: number) => T)(Math.max(n, a.length * 2));
  b.set(a);
  return b;
}

const cellKey = (lon: number, lat: number) => (Math.floor(lon / CELL) + 36000) * 40000 + Math.floor(lat / CELL) + 20000;

/** Ground metres east and north from (lon0, lat0), flat over the short distances it is used for. */
const metres = (lon: number, lat: number, lon0: number, lat0: number): [number, number] => [(lon - lon0) * M_PER_DEG * Math.cos(lat0 * Math.PI / 180), (lat - lat0) * M_PER_DEG];

function ground(lon1: number, lat1: number, lon2: number, lat2: number) {
  const [x, y] = metres(lon2, lat2, lon1, lat1);
  return Math.hypot(x, y);
}

/** Load a region's graph and join it in; false if it has none. */
function loadRegion(cc: string): Promise<boolean> {
  let p = tried.get(cc);
  if (!p) {
    p = realGraph(cc, graphBase).then((g) => {
      if (!g) return false;
      join(g);
      return true;
    }, () => false);
    tried.set(cc, p);
  }
  return p;
}

function join(g: RealGraph) {
  const gi = net.graph.push(g) - 1;
  const J = g.jLon.length, C = g.a.length;
  const jmap = new Uint32Array(J).fill(0xffffffff);
  const sharedId = new Map<number, number>();
  for (let i = 0; i < g.shared.length; i++) sharedId.set(g.shared[i], g.sharedId[i]);
  net.jLon = grow(net.jLon, net.J + J);
  net.jLat = grow(net.jLat, net.J + J);
  net.border = grow(net.border, net.J + J);
  for (let j = 0; j < J; j++) {
    const id = sharedId.get(j);
    const known = id === undefined ? undefined : net.osm.get(id);
    if (known !== undefined) {
      jmap[j] = known;
      continue;
    }
    const k = net.J++;
    jmap[j] = k;
    net.jLon[k] = g.jLon[j];
    net.jLat[k] = g.jLat[j];
    net.border[k] = id === undefined ? 0 : 1;
    if (id !== undefined) net.osm.set(id, k);
  }
  const C0 = net.C;
  for (const k of ["ca", "cb", "cl"] as const) net[k] = grow(net[k], C0 + C);
  for (const k of ["len"] as const) net[k] = grow(net[k], C0 + C);
  for (const k of ["ha", "hb"] as const) net[k] = grow(net[k], C0 + C);
  net.fam = grow(net.fam, C0 + C);
  net.per = grow(net.per, C0 + C);
  net.cls = grow(net.cls, C0 + C);
  net.box = grow(net.box, (C0 + C) * 4);
  net.cg = grow(net.cg, C0 + C);
  for (let c = 0; c < C; c++) {
    const gc = C0 + c, v0 = g.off[c], v1 = g.off[c + 1];
    net.ca[gc] = jmap[g.a[c]];
    net.cb[gc] = jmap[g.b[c]];
    net.fam[gc] = g.kind[c] === 2 ? 1 : 0;
    net.per[gc] = g.flags[c] & 8 ? HSR : 1;
    net.cls[gc] = g.kind[c] === 2 ? 2 : g.rank[c] === 0 ? 0 : 1;
    net.cg[gc] = gi;
    net.cl[gc] = c;
    let w = 180, s = 90, e = -180, n = -90;
    for (let v = v0; v < v1; v++) (w = Math.min(w, g.lon[v])), (e = Math.max(e, g.lon[v])), (s = Math.min(s, g.lat[v])), (n = Math.max(n, g.lat[v]));
    net.box.set([w, s, e, n], gc * 4);
    let L = 0, lastKey = -1;
    const cells = (lon: number, lat: number) => {
      const k = cellKey(lon, lat);
      if (k === lastKey) return;
      lastKey = k;
      let list = net.grid.get(k);
      if (!list) net.grid.set(k, (list = []));
      if (list[list.length - 1] !== gc) list.push(gc);
    };
    cells(g.lon[v0], g.lat[v0]);
    for (let v = v0 + 1; v < v1; v++) {
      const x0 = g.lon[v - 1], y0 = g.lat[v - 1], x1 = g.lon[v], y1 = g.lat[v];
      L += ground(x0, y0, x1, y1);
      // a long straight is cut into half cells so every cell it crosses is near one of them
      const steps = Math.ceil(Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0)) / (CELL / 2));
      for (let s = 1; s <= steps; s++) cells(x0 + ((x1 - x0) * s) / steps, y0 + ((y1 - y0) * s) / steps);
    }
    net.len[gc] = L;
    net.ha[gc] = endHeading(g, v0, v1, 1);
    net.hb[gc] = endHeading(g, v1 - 1, v0 - 1, -1) + Math.PI;
  }
  net.C += C;
  // the incidence lists, all of them again: a border junction gains chains from the new region
  const count = new Uint32Array(net.J + 1);
  for (let c = 0; c < net.C; c++) count[net.ca[c] + 1]++, count[net.cb[c] + 1]++;
  for (let j = 0; j < net.J; j++) count[j + 1] += count[j];
  const adj = new Uint32Array(net.C * 2), fill = count.slice(0, net.J);
  for (let c = 0; c < net.C; c++) adj[fill[net.ca[c]]++] = c * 2, adj[fill[net.cb[c]]++] = c * 2 + 1;
  net.adjStart = count;
  net.adj = adj;
}

/** The heading from vertex `from` towards a point HEADING_M along the chain in direction `step`. */
function endHeading(g: RealGraph, from: number, stop: number, step: number) {
  const x0 = g.lon[from], y0 = g.lat[from];
  let v = from + step, d = 0;
  for (; v !== stop; v += step) {
    d = ground(x0, y0, g.lon[v], g.lat[v]);
    if (d >= HEADING_M) break;
  }
  if (v === stop) v -= step;
  const [x, y] = metres(g.lon[v], g.lat[v], x0, y0);
  return Math.atan2(y, x);
}

// ---------------------------------------------------------------- candidates

interface Cand { c: number; /** m along the chain from its a end */ along: number; dist: number; lon: number; lat: number }

/** Every chain within `r` ground m of the point, at its nearest point. */
function candidates(lon: number, lat: number, r: number): Cand[] {
  const dLat = r / M_PER_DEG, dLon = dLat / Math.cos(lat * Math.PI / 180);
  const seen = new Set<number>();
  for (let x = Math.floor((lon - dLon) / CELL) - 1; x <= Math.floor((lon + dLon) / CELL) + 1; x++)
    for (let y = Math.floor((lat - dLat) / CELL) - 1; y <= Math.floor((lat + dLat) / CELL) + 1; y++)
      for (const c of net.grid.get((x + 36000) * 40000 + y + 20000) ?? []) seen.add(c);
  const out: Cand[] = [];
  for (const c of seen) {
    const g = net.graph[net.cg[c]], l = net.cl[c], v0 = g.off[l], v1 = g.off[l + 1];
    let best = Infinity, along = 0, at = 0, bx = 0, by = 0;
    for (let v = v0; v + 1 < v1; v++) {
      const [ax, ay] = metres(g.lon[v], g.lat[v], lon, lat), [cx, cy] = metres(g.lon[v + 1], g.lat[v + 1], lon, lat);
      const dx = cx - ax, dy = cy - ay, l2 = dx * dx + dy * dy, seg = Math.sqrt(l2);
      const t = l2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
      const d = Math.hypot(ax + t * dx, ay + t * dy);
      if (d < best) {
        best = d;
        along = at + t * seg;
        bx = g.lon[v] + (g.lon[v + 1] - g.lon[v]) * t;
        by = g.lat[v] + (g.lat[v + 1] - g.lat[v]) * t;
      }
      at += seg;
    }
    // `at` is the chain's length measured the same way as `along`; scaled onto `len`
    if (best <= r) out.push({ c, along: at > 0 ? (along / at) * net.len[c] : 0, dist: best, lon: bx, lat: by });
  }
  return out.sort((p, q) => p.dist - q.dist);
}

// ---------------------------------------------------------------- the search

/** One step of a route: chain `c` from `from` to `to` m along it (from > to runs it backwards). */
interface Leg { c: number; from: number; to: number }

const turn = (a: number, b: number) => {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
};

let latest = 0;

async function search(seq: number, a: [number, number], b: [number, number], ra: number, rb: number): Promise<{ legs: Leg[] | null; why?: "off" | "none"; states: number; start?: Cand; end?: Cand }> {
  await Promise.all([...(await regionsAt(a[0], a[1], 0)), ...(await regionsAt(b[0], b[1], 0))].map(loadRegion));
  const starts = candidates(a[0], a[1], ra), ends = candidates(b[0], b[1], rb);
  if (!starts.length || !ends.length) return { legs: null, why: "off", states: 0 };
  const fam0 = net.fam[starts[0].c];
  const fac = (c: number) => (net.fam[c] === fam0 ? 1 : OTHER_FAMILY) * net.per[c];
  const endsOn = new Map<number, Cand[]>();
  for (const e of ends) endsOn.set(e.c, [...(endsOn.get(e.c) ?? []), e]);
  let n2 = net.C * 2;
  let g = new Float64Array(n2).fill(Infinity), prev = new Int32Array(n2);
  let best = { total: Infinity, s: -1, last: null as Leg | null, start: null as Cand | null, end: null as Cand | null, direct: false };
  // the heap: f and state
  let hf = new Float64Array(1024), hs = new Int32Array(1024), hn = 0;
  const push = (f: number, s: number) => {
    if (hn === hf.length) hf = grow(hf, hn * 2), hs = grow(hs, hn * 2);
    let i = hn++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (hf[p] <= f) break;
      hf[i] = hf[p], hs[i] = hs[p], i = p;
    }
    hf[i] = f, hs[i] = s;
  };
  const pop = () => {
    const s = hs[0], f = hf[0], lf = hf[--hn], ls = hs[hn];
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= hn) break;
      if (c + 1 < hn && hf[c + 1] < hf[c]) c++;
      if (hf[c] >= lf) break;
      hf[i] = hf[c], hs[i] = hs[c], i = c;
    }
    hf[i] = lf, hs[i] = ls;
    return [f, s] as const;
  };
  const h = (j: number) => HSR * Math.max(0, ground(net.jLon[j], net.jLat[j], b[0], b[1]) - rb);
  const arrive = (s: number) => (s & 1 ? net.ca[s >> 1] : net.cb[s >> 1]);
  starts.forEach((st, i) => {
    const c = st.c, f = fac(c), L = net.len[c];
    // straight along its own chain to an end candidate on it
    for (const e of endsOn.get(c) ?? []) {
      const total = Math.abs(e.along - st.along) * f + 2 * st.dist + 2 * e.dist;
      if (total < best.total) best = { total, s: -1, last: { c, from: st.along, to: e.along }, start: st, end: e, direct: true };
    }
    for (const [s, gc] of [[c * 2, (L - st.along) * f], [c * 2 + 1, st.along * f]] as const) {
      const v = gc + 2 * st.dist;
      if (v < g[s]) {
        g[s] = v;
        prev[s] = -1 - i;
        push(v + h(arrive(s)), s);
      }
    }
  });
  const t0 = performance.now();
  let states = 0;
  const checked = new Set<number>();
  while (hn) {
    const [f, s] = pop();
    if (f >= best.total) break;
    const J = arrive(s);
    if (f > g[s] + h(J) + 1e-6) continue;
    if (++states % 20000 === 0) {
      if (performance.now() - t0 > SEARCH_MS || states > SEARCH_STATES) break;
      await new Promise((r) => setTimeout(r));
      if (seq !== latest) return { legs: null, states };
    }
    if (net.border[J] && !checked.has(J)) {
      // the search reached another region's file: join it before going on
      checked.add(J);
      const C0 = net.C;
      await Promise.all((await regionsAt(net.jLon[J], net.jLat[J], BORDER_DEG)).map(loadRegion));
      if (seq !== latest) return { legs: null, states };
      if (net.C !== C0) {
        n2 = net.C * 2;
        const g2 = new Float64Array(n2).fill(Infinity);
        g2.set(g);
        g = g2;
        prev = grow(prev, n2);
      }
    }
    const c = s >> 1, back = s & 1;
    const hin = back ? net.ha[c] + Math.PI : net.hb[c];
    for (let k = net.adjStart[J]; k < net.adjStart[J + 1]; k++) {
      const end = net.adj[k], c2 = end >> 1, fromB = end & 1;
      const hout = fromB ? net.hb[c2] + Math.PI : net.ha[c2];
      if (turn(hin, hout) > TURN_MAX) continue;
      const f2 = fac(c2), L = net.len[c2];
      for (const e of endsOn.get(c2) ?? []) {
        const total = g[s] + (fromB ? L - e.along : e.along) * f2 + 2 * e.dist;
        if (total < best.total) best = { total, s, last: { c: c2, from: fromB ? L : 0, to: e.along }, start: null, end: e, direct: false };
      }
      const s2 = c2 * 2 + fromB, v = g[s] + L * f2;
      if (v < g[s2]) {
        g[s2] = v;
        prev[s2] = s;
        push(v + h(arrive(s2)), s2);
      }
    }
  }
  if (!best.last) return { legs: null, why: "none", states };
  const legs: Leg[] = [best.last];
  let start = best.start;
  for (let s = best.s; s >= 0; ) {
    const c = s >> 1, back = s & 1, p = prev[s];
    if (p < 0) {
      start = starts[-1 - p];
      legs.push({ c, from: start.along, to: back ? 0 : net.len[c] });
      break;
    }
    legs.push({ c, from: back ? net.len[c] : 0, to: back ? 0 : net.len[c] });
    s = p;
  }
  legs.reverse();
  return { legs, states, start: start!, end: best.end! };
}

// ---------------------------------------------------------------- the reach

/** the chains reachable from the last first click, kept while the view moves; `C` the graph's size then */
let reached: { key: string; C: number; on: Uint8Array; chains: number } | null = null;

/** Every chain a route from (lon, lat) could run over: the search's moves without its costs. */
function reachFrom(lon: number, lat: number, r: number): { on: Uint8Array; chains: number } {
  const n2 = net.C * 2, seen = new Uint8Array(n2), on = new Uint8Array(net.C), stack = new Uint32Array(n2);
  let sp = 0, chains = 0;
  for (const { c } of candidates(lon, lat, r)) {
    if (!on[c]) (on[c] = 1), chains++;
    for (const s of [c * 2, c * 2 + 1]) if (!seen[s]) (seen[s] = 1), (stack[sp++] = s);
  }
  while (sp) {
    const s = stack[--sp], c = s >> 1, back = s & 1, J = back ? net.ca[c] : net.cb[c];
    const hin = back ? net.ha[c] + Math.PI : net.hb[c];
    for (let k = net.adjStart[J]; k < net.adjStart[J + 1]; k++) {
      const end = net.adj[k], c2 = end >> 1, s2 = end;
      if (seen[s2] || turn(hin, end & 1 ? net.hb[c2] + Math.PI : net.ha[c2]) > TURN_MAX) continue;
      seen[s2] = 1;
      stack[sp++] = s2;
      if (!on[c2]) (on[c2] = 1), chains++;
    }
  }
  return { on, chains };
}

/** The reachable chains crossing the box, their vertices thinned to `step` degrees; a chain
 * shorter than that each way (a crossover, zoomed out) is left out. */
function reachIn(on: Uint8Array, box: [number, number, number, number], step: number) {
  const [w, s, e, n] = box, xy: number[] = [], off: number[] = [0], cls: number[] = [];
  const b = net.box;
  for (let c = 0; c < on.length; c++) {
    if (!on[c] || b[c * 4] > e || b[c * 4 + 2] < w || b[c * 4 + 1] > n || b[c * 4 + 3] < s) continue;
    if (b[c * 4 + 2] - b[c * 4] < step && b[c * 4 + 3] - b[c * 4 + 1] < step) continue;
    const g = net.graph[net.cg[c]], l = net.cl[c], v0 = g.off[l], v1 = g.off[l + 1];
    let lx = g.lon[v0], ly = g.lat[v0];
    xy.push(lx, ly);
    for (let v = v0 + 1; v < v1; v++) {
      const x = g.lon[v], y = g.lat[v];
      if (v + 1 < v1 && Math.abs(x - lx) < step && Math.abs(y - ly) < step) continue;
      xy.push(x, y);
      (lx = x), (ly = y);
    }
    off.push(xy.length / 2);
    cls.push(net.cls[c]);
  }
  return { xy: new Float32Array(xy), off: new Uint32Array(off), cls: new Uint8Array(cls) };
}

/** The route's line as lon, lat pairs: the snap point at each click, the track between. */
function lineOf(legs: Leg[], start: Cand, end: Cand): { line: Float64Array; lengthM: number } {
  const out: number[] = [start.lon, start.lat];
  let lengthM = 0;
  for (const { c, from, to } of legs) {
    const g = net.graph[net.cg[c]], l = net.cl[c], v0 = g.off[l], v1 = g.off[l + 1];
    // the chain's vertices with their distance along it, scaled as `candidates` scales
    const n = v1 - v0, at = new Float64Array(n);
    for (let i = 1; i < n; i++) at[i] = at[i - 1] + ground(g.lon[v0 + i - 1], g.lat[v0 + i - 1], g.lon[v0 + i], g.lat[v0 + i]);
    const k = at[n - 1] > 0 ? net.len[c] / at[n - 1] : 1;
    const lo = Math.min(from, to), hi = Math.max(from, to);
    /** the point `d` m along, between the vertices either side */
    const pointAt = (d: number) => {
      let i = 1;
      while (i < n - 1 && at[i] * k < d) i++;
      const t = at[i] > at[i - 1] ? Math.max(0, Math.min(1, (d / k - at[i - 1]) / (at[i] - at[i - 1]))) : 0;
      return [g.lon[v0 + i - 1] + (g.lon[v0 + i] - g.lon[v0 + i - 1]) * t, g.lat[v0 + i - 1] + (g.lat[v0 + i] - g.lat[v0 + i - 1]) * t];
    };
    const pts: number[] = [...pointAt(lo)];
    for (let i = 0; i < n; i++) if (at[i] * k > lo && at[i] * k < hi) pts.push(g.lon[v0 + i], g.lat[v0 + i]);
    pts.push(...pointAt(hi));
    if (to < from) for (let i = pts.length - 2; i >= 0; i -= 2) out.push(pts[i], pts[i + 1]);
    else out.push(...pts);
    lengthM += hi - lo;
  }
  out.push(end.lon, end.lat);
  // the same point twice in a row (a chain's end at a junction, a snap point on a vertex) is one
  const line: number[] = [out[0], out[1]];
  for (let i = 2; i < out.length; i += 2) if (Math.abs(out[i] - line[line.length - 2]) > 1e-9 || Math.abs(out[i + 1] - line[line.length - 1]) > 1e-9) line.push(out[i], out[i + 1]);
  return { line: new Float64Array(line), lengthM };
}

// ---------------------------------------------------------------- messages

let api: TrackApi | null = null;
const apiReady = (async () => {
  await init();
  const a = new TrackApi(NEW_GAME_FRAME.lon, NEW_GAME_FRAME.lat, true);
  a.set_empty_frame(frameKindNumber(NEW_GAME_FRAME.kind));
  api = a;
})();

/** Work this long between looks at the messages, ms (T-754). */
const TRACE_SLICE_MS = 25;

/** The fit in steps of a few ms, yielding every slice so a newer route request is seen; null
 *  once one has come. */
async function traceLatest(seq: number, line: Float64Array, tol: number, tracks: number, level: number): Promise<Float64Array | null> {
  const t = api!.trace_start(line, tol, tracks, level);
  try {
    let slice = performance.now();
    while (!t.step()) {
      if (performance.now() - slice < TRACE_SLICE_MS) continue;
      await new Promise((r) => setTimeout(r, 0));
      if (seq !== latest) return null;
      slice = performance.now();
    }
    return t.result();
  } finally {
    t.free();
  }
}

let queue: Promise<void> = Promise.resolve();
let latestReach = 0;

self.onmessage = (ev: MessageEvent<ToRouter>) => {
  const m = ev.data;
  if (m.kind === "route") latest = m.seq;
  if (m.kind === "reach") latestReach = m.seq;
  queue = queue.then(() => handle(m)).catch((err) => console.error("real route", err));
};

async function handle(m: ToRouter) {
  if (m.kind === "init") {
    pageBase = m.page;
    graphBase = m.graphs;
    void regionOutlines();
  } else if (m.kind === "view") {
    const all = await regionOutlines();
    const cx = (m.w + m.e) / 2, cy = (m.s + m.n) / 2;
    const near = all
      .filter((o) => o.boxes.some((b) => b[0] <= m.e && b[2] >= m.w && b[1] <= m.n && b[3] >= m.s))
      .map((o) => ({ cc: o.cc, d: Math.min(...o.boxes.map((b) => Math.hypot(Math.max(0, b[0] - cx, cx - b[2]), Math.max(0, b[1] - cy, cy - b[3])))) }))
      .sort((p, q) => p.d - q.d)
      .slice(0, 3);
    // in the background: a route asked meanwhile is not held up behind it
    for (const { cc } of near) void loadRegion(cc);
  } else if (m.kind === "reach") {
    if (m.seq !== latestReach) return;
    if (!m.a) return void (reached = null);
    const t0 = performance.now();
    const a = m.a, key = `${a[0]},${a[1]},${m.ra}`;
    const [w, s, e, n] = m.box;
    const here = await regionsAt(a[0], a[1], 0);
    const all = await regionOutlines();
    const inView = all.filter((o) => !here.includes(o.cc) && o.boxes.some((b) => b[0] <= e && b[2] >= w && b[1] <= n && b[3] >= s)).slice(0, 4).map((o) => o.cc);
    await Promise.all([...here, ...inView].map(loadRegion));
    if (m.seq !== latestReach) return;
    if (reached?.key !== key || reached.C !== net.C) reached = { key, C: net.C, ...reachFrom(a[0], a[1], m.ra) };
    const { xy, off, cls } = reachIn(reached.on, m.box, m.step);
    post({ kind: "reach", seq: m.seq, xy, off, cls, chains: reached.chains, ms: performance.now() - t0 }, [xy.buffer, off.buffer, cls.buffer]);
  } else if (m.kind === "route") {
    if (m.seq !== latest) return;
    const t0 = performance.now();
    const r = await search(m.seq, m.a, m.b, m.ra, m.rb);
    if (m.seq !== latest) return;
    if (!r.legs) return post({ kind: "line", seq: m.seq, line: null, lengthM: 0, why: r.why ?? "none", ms: performance.now() - t0, states: r.states });
    const { line, lengthM } = lineOf(r.legs, r.start!, r.end!);
    post({ kind: "line", seq: m.seq, line: line.slice(), lengthM, ms: performance.now() - t0, states: r.states });
    await apiReady;
    if (m.seq !== latest) return;
    const t1 = performance.now();
    const out = await traceLatest(m.seq, line, m.tol, m.tracks, m.level);
    if (!out) return;
    const ms = performance.now() - t1;
    if (out.length < 3) return post({ kind: "fit", seq: m.seq, pis: null, worst: 0, mean: 0, ms });
    const pis = out.slice(3);
    post({ kind: "fit", seq: m.seq, pis, worst: out[0], mean: out[1], ms }, [pis.buffer]);
  }
}
