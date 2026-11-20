// T-3448: the network diagram by region. Stations joined by the player's lines are one region; separate regions whose
// nearest stations are within REGION_JOIN_KM join while their box stays within REGION_MAX_KM, and regions whose boxes
// overlap always join (one could not be placed apart from the other where its geography puts it). Each region is laid out on
// a sheet of its own (the sheet's cells scale with its map, so a city keeps the grid it gets alone), and the drawings
// are put side by side here: each at its own scale brought to one pitch, placed in the direction its middle lies on the
// world map (mercator, no wrap), as close as keeps every two regions' boxes apart. docs/ui-diagram.md "Regions".

import type { DiagramGraph, DiagramLine } from "./graph";
import type { Drawing } from "./draw";
import type { NamePlan, LevelNames } from "./names";
import type { CityNames } from "./cityNames";
import { EARTH_C, makeAnchor, toLngLat, toMerc01 } from "../map/geo";

/** separate regions whose nearest stations are within this join (km) */
export const REGION_JOIN_KM = 500;
/** ... unless the joined region's box would be longer than this (km) on a side */
export const REGION_MAX_KM = 1200;
/** drawn regions keep at least this many pitches between their boxes (the sea drawn round each included) */
export const REGION_GAP_PITCHES = 24;

export const KM_PER_DEG = 111.195;
const RAD = Math.PI / 180;

/** each node's lng, lat */
export function nodeLngLat(g: DiagramGraph): { lng: Float64Array; lat: Float64Array } {
  const a = makeAnchor(g.origin[0], g.origin[1], 0), n = g.key.length, lng = new Float64Array(n), lat = new Float64Array(n);
  for (let i = 0; i < n; i++) [lng[i], lat[i]] = toLngLat(g.mid[0] + g.x[i] * g.k, g.mid[1] - g.y[i] * g.k, a);
  return { lng, lat };
}

const km = (lng1: number, lat1: number, lng2: number, lat2: number): number => {
  let dl = Math.abs(lng1 - lng2);
  if (dl > 180) dl = 360 - dl;
  const c = Math.cos(((lat1 + lat2) / 2) * RAD);
  return KM_PER_DEG * Math.hypot(dl * c, lat1 - lat2);
};
type Box = [number, number, number, number];
const joined = (A: Box, B: Box): Box => [Math.min(A[0], B[0]), Math.min(A[1], B[1]), Math.max(A[2], B[2]), Math.max(A[3], B[3])];
const overlap = (A: Box, B: Box): boolean => A[0] <= B[2] && B[0] <= A[2] && A[1] <= B[3] && B[1] <= A[3];
const sideKm = (b: Box): number =>
  KM_PER_DEG * Math.max(b[3] - b[1], (b[2] - b[0]) * Math.cos(((b[1] + b[3]) / 2) * RAD));

/** The graph's regions: each its nodes, ascending (the clock worker builds each one's graph in a frame of its own, so
 * a city far from the rest has the graph it has alone). */
export function regionGroups(g: DiagramGraph): Int32Array[] {
  const n = g.key.length;
  if (!n) return [];
  const { lng, lat } = nodeLngLat(g);
  const up = Int32Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => {
    while (up[i] !== i) (up[i] = up[up[i]]), (i = up[i]);
    return i;
  };
  for (let e = 0; e < g.ea.length; e++) {
    const a = find(g.ea[e]), b = find(g.eb[e]);
    if (a !== b) up[Math.max(a, b)] = Math.min(a, b);
  }
  for (const l of g.lines) for (let i = 1; i < l.chain.length; i++) {
    const a = find(l.chain[i - 1]), b = find(l.chain[i]);
    if (a !== b) up[Math.max(a, b)] = Math.min(a, b);
  }
  // the components: their boxes and points a fifth of a degree apart (distances between them good to about 20 km)
  const compOf = new Map<number, number>(), comps: { box: [number, number, number, number]; pts: number[] }[] = [], seen: Set<string>[] = [];
  for (let i = 0; i < n; i++) {
    const r = find(i);
    let c = compOf.get(r);
    if (c === undefined) compOf.set(r, (c = comps.length)), comps.push({ box: [Infinity, Infinity, -Infinity, -Infinity], pts: [] }), seen.push(new Set());
    const C = comps[c], b = C.box;
    (b[0] = Math.min(b[0], lng[i])), (b[1] = Math.min(b[1], lat[i])), (b[2] = Math.max(b[2], lng[i])), (b[3] = Math.max(b[3], lat[i]));
    const cell = Math.round(lng[i] * 5) + "," + Math.round(lat[i] * 5);
    if (!seen[c].has(cell)) seen[c].add(cell), C.pts.push(lng[i], lat[i]);
  }
  const K = comps.length;
  // nearby components join, nearest first, while the joined box stays within REGION_MAX_KM
  const joinOf = Int32Array.from({ length: K }, (_, i) => i), boxes = comps.map((c) => [...c.box] as [number, number, number, number]);
  const jfind = (i: number): number => {
    while (joinOf[i] !== i) (joinOf[i] = joinOf[joinOf[i]]), (i = joinOf[i]);
    return i;
  };
  const join = (a: number, b: number) => {
    boxes[Math.min(a, b)] = joined(boxes[a], boxes[b]);
    joinOf[Math.max(a, b)] = Math.min(a, b);
  };
  if (K > 1) {
    const pairs: [number, number, number][] = [], latGap = REGION_JOIN_KM / KM_PER_DEG;
    for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) {
      const A = comps[a], B = comps[b];
      if (overlap(A.box, B.box)) {
        pairs.push([0, a, b]);
        continue;
      }
      if (A.box[1] - B.box[3] > latGap || B.box[1] - A.box[3] > latGap) continue;
      let best = Infinity;
      for (let i = 0; i < A.pts.length && best > 0; i += 2) for (let j = 0; j < B.pts.length; j += 2) best = Math.min(best, km(A.pts[i], A.pts[i + 1], B.pts[j], B.pts[j + 1]));
      if (best <= REGION_JOIN_KM) pairs.push([best, a, b]);
    }
    pairs.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
    for (const [, a0, b0] of pairs) {
      const a = jfind(a0), b = jfind(b0);
      if (a === b) continue;
      if (sideKm(joined(boxes[a], boxes[b])) > REGION_MAX_KM && !overlap(boxes[a], boxes[b])) continue;
      join(a, b);
    }
    // a joined box can come to overlap another region's
    for (let again = true; again; ) {
      again = false;
      for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) {
        const ra = jfind(a), rb = jfind(b);
        if (ra !== rb && overlap(boxes[ra], boxes[rb])) join(ra, rb), (again = true);
      }
    }
  }
  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = jfind(compOf.get(find(i))!);
    let v = groups.get(r);
    if (!v) groups.set(r, (v = []));
    v.push(i);
  }
  return [...groups.values()].map((v) => Int32Array.from(v));
}

/** the regions' graphs, the one with the station nearest the camera (lng, lat) first, then by distance from it */
export function cameraOrder(graphs: DiagramGraph[], camLng: number, camLat: number): DiagramGraph[] {
  const near = graphs.map((g) => {
    const { lng, lat } = nodeLngLat(g);
    let d = Infinity;
    for (let i = 0; i < lng.length; i++) d = Math.min(d, km(camLng, camLat, lng[i], lat[i]));
    return d;
  });
  return graphs.map((g, i) => ({ g, d: near[i] })).sort((a, b) => a.d - b.d || a.g.key[0] - b.g.key[0]).map((x) => x.g);
}

/** a string that changes when the region's graph or its names do (a region the same as before is not laid out again) */
export function regionSig(g: DiagramGraph, names: string[], widths: ArrayLike<number>, extra: string): string {
  let h1 = 0x811c9dc5, h2 = 0x12345;
  const add = (v: number) => {
    const q = Math.round(v * 100);
    h1 = Math.imul(h1 ^ (q | 0), 16777619);
    h2 = (Math.imul(h2, 31) + ((q / 4294967296) | 0) + (q | 0)) | 0;
  };
  const all = (a: ArrayLike<number>) => { add(a.length); for (let i = 0; i < a.length; i++) add(a[i]); };
  all(g.key), all(g.x), all(g.y), all(g.ea), all(g.eb), all(g.on), all(g.stroke), all(widths);
  if (g.hop) all(g.hop);
  for (const l of g.lines) {
    add(l.id), all(l.chain), all(l.stop), add(l.fast ? 1 : 0), add(l.tram ? 1 : 0), add(l.oneWay ? 1 : 0);
    // T-3409: a cut run's place in its line and its stubs
    if (l.from !== undefined) add(l.from);
    if (l.stubA) all(l.stubA);
    if (l.stubB) all(l.stubB);
    for (let i = 0; i < l.colour.length; i++) add(l.colour.charCodeAt(i));
  }
  for (const s of names) for (let i = 0; i < s.length; i++) add(s.charCodeAt(i));
  return `${g.key.length}:${h1 >>> 0}:${h2 >>> 0}:${extra}`;
}

/** a region laid out: its graph, places, drawing and names */
export interface Part { graph: DiagramGraph; x: Float64Array; y: Float64Array; d: Drawing; names: NamePlan }
/** a part's place in the whole: layout units times s, plus (ox, oy) */
export interface Placed { s: number; ox: number; oy: number }

/** the box of path commands (draw.ts `toPath`) */
function cmdsBox(c: Float64Array, b: number[]): void {
  for (let i = 0; i < c.length; ) {
    const n = c[i] === 2 ? 5 : 3;
    for (let q = 1; q < n; q += 2) (b[0] = Math.min(b[0], c[i + q])), (b[1] = Math.min(b[1], c[i + q + 1])), (b[2] = Math.max(b[2], c[i + q])), (b[3] = Math.max(b[3], c[i + q + 1]));
    i += n;
  }
}

/** each part's place, the part at `anchor` as it is: each brought to its pitch, and the others in the direction their
 * middle lies from its, as near as keeps every two boxes (with their sea) REGION_GAP_PITCHES apart */
export function placeParts(parts: Part[], anchor: number): Placed[] {
  if (parts.length === 1) return [{ s: 1, ox: 0, oy: 0 }];
  const P = parts[anchor].d.pitch, gap = REGION_GAP_PITCHES * P;
  const geo = parts.map((p) => {
    const { lng, lat } = nodeLngLat(p.graph);
    let mx = 0, my = 0, lx = 0, ly = 0;
    for (let i = 0; i < lng.length; i++) {
      const [a, b] = toMerc01(lng[i], lat[i]);
      (mx += a), (my += b), (lx += p.x[i]), (ly += p.y[i]);
    }
    const n = Math.max(1, lng.length), s = P / p.d.pitch;
    const b = [...p.d.box];
    if (p.d.water?.sea) cmdsBox(p.d.water.sea, b);
    (lx /= n), (ly /= n);
    // the box about the middle, at the common pitch; y up, as the layout
    return { mx: (mx / n) * EARTH_C, my: (-my / n) * EARTH_C, lx, ly, s, e: [(b[0] - lx) * s, (b[1] - ly) * s, (b[2] - lx) * s, (b[3] - ly) * s] };
  });
  // layout units per mercator metre: the smallest that keeps every two boxes apart along x or y
  let G = 0;
  for (let i = 0; i < geo.length; i++) for (let j = i + 1; j < geo.length; j++) {
    const A = geo[i], B = geo[j], dx = B.mx - A.mx, dy = B.my - A.my;
    const gx = dx > 0 ? (A.e[2] - B.e[0] + gap) / dx : dx < 0 ? (B.e[2] - A.e[0] + gap) / -dx : Infinity;
    const gy = dy > 0 ? (A.e[3] - B.e[1] + gap) / dy : dy < 0 ? (B.e[3] - A.e[1] + gap) / -dy : Infinity;
    const need = Math.min(gx, gy);
    if (Number.isFinite(need)) G = Math.max(G, need);
  }
  const A = geo[anchor];
  return geo.map((p, i) => i === anchor ? { s: 1, ox: 0, oy: 0 } : { s: p.s, ox: A.lx + G * (p.mx - A.mx) - p.s * p.lx, oy: A.ly + G * (p.my - A.my) - p.s * p.ly });
}

/** path commands moved by a placing */
function moveCmds(c: Float64Array, t: Placed): Float64Array {
  const out = new Float64Array(c.length);
  for (let i = 0; i < c.length; ) {
    const n = c[i] === 2 ? 5 : 3;
    out[i] = c[i];
    for (let q = 1; q < n; q += 2) (out[i + q] = c[i + q] * t.s + t.ox), (out[i + q + 1] = c[i + q + 1] * t.s + t.oy);
    i += n;
  }
  return out;
}
const movePairs = (p: Float64Array, t: Placed): Float64Array => {
  const out = new Float64Array(p.length);
  for (let i = 0; i < p.length; i += 2) (out[i] = p[i] * t.s + t.ox), (out[i + 1] = p[i + 1] * t.s + t.oy);
  return out;
};

/** the parts as one: graph, places, drawing and names, as one region's would be; and per node its part. The graph's
 * frame (mid, origin, k) is the first part's: `nearestNode` finds a place's node through the parts. */
export function mergeParts(parts: Part[], at: Placed[]): Part & { partOf: Int32Array; base: number[] } {
  if (parts.length === 1) return { ...parts[0], partOf: new Int32Array(parts[0].graph.key.length), base: [0] };
  const P = parts[0].d.pitch * at[0].s;
  const base: number[] = [], lineBase: number[] = [];
  let nn = 0, nl = 0;
  for (const p of parts) base.push(nn), lineBase.push(nl), (nn += p.graph.key.length), (nl += p.graph.lines.length);
  const partOf = new Int32Array(nn), key = new Int32Array(nn), x = new Float64Array(nn), y = new Float64Array(nn), gx = new Float64Array(nn), gy = new Float64Array(nn), small = new Uint8Array(nn);
  const lines: DiagramLine[] = [], ea: number[] = [], eb: number[] = [], first: number[] = [], on: number[] = [], stroke: number[] = [];
  const d: Drawing = { pitch: P, stroke: parts[0].d.stroke * at[0].s, lines: [], lineOf: [], ribbons: [], chainAt: [], pills: [], pillOf: [], passages: [], passageOf: [], hooks: [], small, box: [Infinity, Infinity, -Infinity, -Infinity] };
  const anyBorders = parts.some((p) => p.d.borders), anyWater = parts.some((p) => p.d.water);
  const bCmds: Float64Array[] = [], bZones: string[] = [], bZoneOf = new Int32Array(nn).fill(-1);
  const water = { rivers: [] as Float64Array[], bands: [] as number[], riverEnds: [] as Float64Array[], sea: [] as Float64Array[], seaRings: [] as Float64Array[], fillets: [] as Float64Array[] };
  const p0 = parts[0].names;
  const at2: { node: number[]; x: number[]; y: number[]; dir: number[]; slide: number[]; lift: number[]; wrap: number[] }[] = p0.levels.map(() => ({ node: [], x: [], y: [], dir: [], slide: [], lift: [], wrap: [] }));
  const text: string[] = [], width: number[] = [], width2: number[] = [];
  let ms = 0;
  const anyCities = parts.some((p) => p.names.cities);
  const cities = { text: [] as string[], width: [] as number[], mx: [] as number[], my: [] as number[], bw: [] as number[], bh: [] as number[], rank: [] as number[], order: [] as number[], want: [] as string[] };
  parts.forEach((p, pi) => {
    const t = at[pi], nb = base[pi], lb = lineBase[pi], g = p.graph, n = g.key.length, pd = p.d;
    partOf.fill(pi, nb, nb + n);
    key.set(g.key, nb), small.set(pd.small, nb), gx.set(g.x, nb), gy.set(g.y, nb);
    for (let i = 0; i < n; i++) (x[nb + i] = p.x[i] * t.s + t.ox), (y[nb + i] = p.y[i] * t.s + t.oy);
    for (const l of g.lines) lines.push({ ...l, chain: l.chain.map((c) => c + nb) });
    for (let e = 0; e < g.ea.length; e++) {
      ea.push(g.ea[e] + nb), eb.push(g.eb[e] + nb), first.push(on.length);
      for (let q = g.first[e]; q < g.first[e + 1]; q++) on.push(g.on[q] + lb), stroke.push(g.stroke[q] + lb);
    }
    // the drawing
    for (const l of pd.lines) d.lines.push({ colour: l.colour, fast: l.fast, tram: l.tram, oneWay: l.oneWay, bus: l.bus, ferry: l.ferry, cmds: moveCmds(l.cmds, t) });
    if (pd.dot) (d.dot ??= new Uint8Array(nn)).set(pd.dot, nb);
    for (const li of pd.lineOf) d.lineOf.push(li + lb);
    for (const r of pd.ribbons) d.ribbons.push(movePairs(r, t));
    for (const c of pd.chainAt) d.chainAt.push(c);
    for (const q of pd.pills) d.pills.push([q[0] * t.s + t.ox, q[1] * t.s + t.oy, q[2] * t.s + t.ox, q[3] * t.s + t.oy, q[4] * t.s]);
    for (const k of pd.pillOf) d.pillOf.push(k + nb);
    for (const q of pd.passages) d.passages.push([q[0] * t.s + t.ox, q[1] * t.s + t.oy, q[2] * t.s + t.ox, q[3] * t.s + t.oy]);
    for (const k of pd.passageOf) d.passageOf.push(k + nb);
    for (const k of pd.hooks) d.hooks.push(k + nb);
    for (const l of pd.thin ?? []) (d.thin ??= []).push({ id: l.id, colour: l.colour, ferry: l.ferry, cmds: moveCmds(l.cmds, t), ribbon: movePairs(l.ribbon, t), stops: movePairs(l.stops, t) });
    const b = pd.box;
    (d.box[0] = Math.min(d.box[0], b[0] * t.s + t.ox)), (d.box[1] = Math.min(d.box[1], b[1] * t.s + t.oy)), (d.box[2] = Math.max(d.box[2], b[2] * t.s + t.ox)), (d.box[3] = Math.max(d.box[3], b[3] * t.s + t.oy));
    if (pd.borders) {
      bCmds.push(moveCmds(pd.borders.cmds, t));
      const zb = bZones.length;
      bZones.push(...pd.borders.zones);
      for (let i = 0; i < n; i++) bZoneOf[nb + i] = pd.borders.zoneOf[i] < 0 ? -1 : pd.borders.zoneOf[i] + zb;
    }
    const wa = pd.water;
    if (wa) {
      for (const r of wa.rivers) water.rivers.push(moveCmds(r, t));
      for (const w of wa.bands) water.bands.push(w * t.s);
      for (const r of wa.riverEnds ?? []) water.riverEnds.push(moveCmds(r, t));
      if (wa.sea) water.sea.push(moveCmds(wa.sea, t));
      if (wa.fillets) water.fillets.push(moveCmds(wa.fillets, t));
      for (const r of wa.seaRings ?? []) water.seaRings.push(movePairs(r, t));
    }
    // the names
    const pn = p.names;
    pn.at.forEach((L: LevelNames, li: number) => {
      const o = at2[li];
      if (!o) return;
      for (let i = 0; i < L.node.length; i++) o.node.push(L.node[i] + nb), o.x.push(L.x[i] * t.s + t.ox), o.y.push(L.y[i] * t.s + t.oy), o.dir.push(L.dir[i]), o.slide.push(L.slide[i]), o.lift.push(L.lift[i] * t.s), o.wrap.push(L.wrap[i]);
    });
    text.push(...pn.text), width.push(...pn.width), width2.push(...pn.width2), (ms += pn.ms);
    const c = pn.cities;
    if (c) {
      const cb = cities.text.length;
      cities.text.push(...c.text), cities.width.push(...c.width), cities.rank.push(...c.rank), cities.want.push(...c.want);
      for (let i = 0; i < c.text.length; i++) cities.mx.push(c.mx[i] * t.s + t.ox), cities.my.push(c.my[i] * t.s + t.oy), cities.bw.push(c.bw[i] * t.s), cities.bh.push(c.bh[i] * t.s);
      for (const o of c.order) cities.order.push(o + cb);
    }
  });
  first.push(on.length);
  if (anyBorders) {
    const all = new Float64Array(bCmds.reduce((s, c) => s + c.length, 0));
    let o = 0;
    for (const c of bCmds) all.set(c, o), (o += c.length);
    d.borders = { cmds: all, zones: bZones, zoneOf: bZoneOf };
  }
  if (anyWater) {
    const join = (cs: Float64Array[]) => {
      const all = new Float64Array(cs.reduce((s, c) => s + c.length, 0));
      let o = 0;
      for (const c of cs) all.set(c, o), (o += c.length);
      return all;
    };
    d.water = { rivers: water.rivers, bands: water.bands, riverEnds: water.riverEnds, sea: join(water.sea), seaRings: water.seaRings, ...(water.fillets.length ? { fillets: join(water.fillets) } : {}), stats: parts.find((p) => p.d.water)!.d.water!.stats };
  }
  const names: NamePlan = {
    levels: p0.levels, faint: p0.faint, fontPx: p0.fontPx, ms, text, width: Float32Array.from(width), width2: Float32Array.from(width2),
    at: at2.map((o) => ({ node: Int32Array.from(o.node), x: Float64Array.from(o.x), y: Float64Array.from(o.y), dir: Uint8Array.from(o.dir), slide: Int8Array.from(o.slide), lift: Float64Array.from(o.lift), wrap: Uint8Array.from(o.wrap) })),
  };
  if (anyCities) {
    const rank = Float64Array.from(cities.rank);
    const c: CityNames = {
      text: cities.text, width: Float32Array.from(cities.width), fontPx: parts.find((p) => p.names.cities)!.names.cities!.fontPx,
      mx: Float64Array.from(cities.mx), my: Float64Array.from(cities.my), bw: Float64Array.from(cities.bw), bh: Float64Array.from(cities.bh), rank,
      order: Int32Array.from(cities.order.sort((a, b) => rank[b] - rank[a] || a - b)), want: cities.want,
    };
    names.cities = c;
  }
  const g0 = parts[0].graph;
  const graph: DiagramGraph = { key, x: gx, y: gy, lines, ea: Int32Array.from(ea), eb: Int32Array.from(eb), first: Int32Array.from(first), on: Int32Array.from(on), stroke: Int32Array.from(stroke), mid: g0.mid, origin: g0.origin, k: g0.k };
  return { graph, x, y, d, names, partOf, base };
}
