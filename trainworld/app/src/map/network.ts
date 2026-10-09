// What the renderer draws, as flat typed arrays in local units (map/geo.ts), built from the clock
// worker's state (T-025): track strokes per edge, line strokes along each line's path, station
// dots, and the keyframe tables trains are positioned from (notes/T-040.md "Keyframes").
//
// Trains: each trip is (profile, departure - epoch); profile p = (line * 2 + run) * 3 + level has
// a phase table (t, s, v, a) and meta (first phase, count, trip seconds, path length m); run 0 of
// each line is resampled every 50 m into `samples`; run 1's offset s is `length - s` on run 0.

import { rgb } from "../game/palette";
import type { ClockState } from "../workers/protocol";
import { toLocal } from "./geo";

/** Colour index of bare track (not a line). */
export const TRACK = 0xffffff;
/** Segment flags. */
export const F_BLUEPRINT = 1;
export const F_FADED = 2;
export const F_BAD = 16;

export interface Strokes {
  /** per segment: x0, y0, x1, y1 */
  seg: Float32Array;
  /** per segment: line id (colour, selection), or TRACK */
  colour: Uint32Array;
  /** per segment: edge id */
  edge: Uint32Array;
  /** per segment: level at its start (height / 8 m, rounded) */
  level: Float32Array;
  flags: Float32Array;
  /** per segment: distance along its edge or line at its start, local units (dashes) */
  dist: Float32Array;
  /** per segment: sideways offset in line widths, to the left of the segment's direction
   * (lines sharing track drawn side by side, T-062); absent = 0 */
  offset?: Float32Array;
  /** line strokes only, 4 per segment (T-084, demand views): the stop-to-stop stretch of run 0 it
   * is on (stop k to k + 1), its slot on the edge (0 = rightmost looking forward), the lines on
   * the edge, and the sign that turns slot order into `offset` (+1 or -1). With these the lines
   * on an edge can be laid side by side again at widths of their own. */
  info?: Float32Array;
  count: number;
}

export interface NetworkBuffers {
  track: Strokes;
  lines: Strokes;
  /** line ids drawn (line strokes), for picking */
  lineIds: string[];
  /** per line id: r, g, b, 255 */
  lineColours: Uint8Array;
  /** per station: x, y, flags (1 transfer, 2 blueprint) */
  stations: Float32Array;
  stationIds: string[];
  stationCount: number;
  /** junction nodes (3+ edge ends, no station): x, y per node, for picking */
  junctions: Float32Array;
  junctionIds: number[];
  // keyframes
  phases: Float32Array;
  meta: Float32Array;
  samples: Float32Array;
  lineTable: Float32Array;
  /** per sample of `samples`: the line's sideways offset there, line widths to the left of run
   * 0's direction (T-062), so trains ride their own stroke; absent = 0 */
  sampleOff?: Float32Array;
  /** per sample of `samples`: a line stroke segment of the same stretch of path (its offset is
   * the sample's), -1 if none (T-084: offsets redone for line widths by load) */
  sampleStroke?: Int32Array;
  /** per edge id: local polyline (x, y, z triples), for snapping and picking */
  edgePts: Map<number, Float32Array>;
}

/** Stroke segments into typed arrays that grow by doubling (number[] pushes cost ~25 ms for a
 * 2,000 km network, T-063). */
class StrokeBuilder {
  private cap = 0;
  count = 0;
  seg = new Float32Array(0);
  colour = new Uint32Array(0);
  edge = new Uint32Array(0);
  level = new Float32Array(0);
  flags = new Float32Array(0);
  dist = new Float32Array(0);
  offset = new Float32Array(0);
  /** 4 per segment (`Strokes.info`), only when built with `withInfo` */
  info = new Float32Array(0);
  constructor(private withInfo = false) {}
  private grow(need: number) {
    if (need <= this.cap) return;
    const cap = Math.max(need, this.cap * 2, 1024);
    const g = <T extends Float32Array | Uint32Array>(a: T, k: number): T => {
      const b = new (a.constructor as any)(cap * k) as T;
      b.set(a);
      return b;
    };
    this.seg = g(this.seg, 4);
    this.colour = g(this.colour, 1);
    this.edge = g(this.edge, 1);
    this.level = g(this.level, 1);
    this.flags = g(this.flags, 1);
    this.dist = g(this.dist, 1);
    this.offset = g(this.offset, 1);
    if (this.withInfo) this.info = g(this.info, 4);
    this.cap = cap;
  }
  /** pts: x, y, z triples; `offset` in line widths to the left of the drawing direction; `info`
   * as `Strokes.info`, for every segment added */
  add(pts: ArrayLike<number>, colour: number, edge: number, flags: number, reverse = false, dist0 = 0, offset = 0, info?: [number, number, number, number]): number {
    const n = pts.length / 3;
    let d = dist0;
    this.grow(this.count + Math.max(0, n - 1));
    if (info && this.withInfo) for (let c = this.count; c < this.count + Math.max(0, n - 1); c++) this.info.set(info, c * 4);
    for (let k = 0; k < n - 1; k++) {
      const i = reverse ? n - 1 - k : k, j = reverse ? i - 1 : i + 1;
      const x0 = pts[i * 3], y0 = pts[i * 3 + 1], x1 = pts[j * 3], y1 = pts[j * 3 + 1];
      const c = this.count++;
      this.seg[c * 4] = x0;
      this.seg[c * 4 + 1] = y0;
      this.seg[c * 4 + 2] = x1;
      this.seg[c * 4 + 3] = y1;
      this.colour[c] = colour;
      this.edge[c] = edge;
      this.level[c] = Math.round(pts[i * 3 + 2] / 8);
      this.flags[c] = flags;
      this.dist[c] = d;
      this.offset[c] = offset;
      d += Math.hypot(x1 - x0, y1 - y0);
    }
    return d;
  }
  build(): Strokes {
    const n = this.count;
    return {
      seg: this.seg.slice(0, n * 4),
      colour: this.colour.slice(0, n),
      edge: this.edge.slice(0, n),
      level: this.level.slice(0, n),
      flags: this.flags.slice(0, n),
      dist: this.dist.slice(0, n),
      offset: this.offset.slice(0, n),
      ...(this.withInfo ? { info: this.info.slice(0, n * 4) } : {}),
      count: n,
    };
  }
}

/**
 * Lines sharing track, side by side (T-062). Each edge gets a direction ("forward") chosen so
 * that it agrees across every node: through a station or a plain node the next edge carries on
 * the same way, and at a junction the branches leaving on one side all point away from it. Lines
 * on an edge then take slots right to left looking forward: lines leaning right first (see
 * below), then by id, so a pair of lines keeps its order along a shared stretch whichever way
 * each edge was drawn. Returns, per edge id, +1 if forward is a -> b, else -1; and per edge, its
 * lines in slot order.
 */
function layoutBundles(s: ClockState, edgePts: Map<number, Float32Array>) {
  // Outward tangent of each edge end at its node, from the drawn polyline.
  const ends = new Map<number, { e: number; atA: boolean; tx: number; ty: number }[]>();
  for (const e of s.edges) {
    const p = edgePts.get(e.id);
    if (!p || p.length < 6) continue;
    const n = p.length / 3;
    const add = (node: number, atA: boolean, tx: number, ty: number) => {
      const l = Math.hypot(tx, ty) || 1;
      const v = ends.get(node) ?? [];
      v.push({ e: e.id, atA, tx: tx / l, ty: ty / l });
      ends.set(node, v);
    };
    add(e.a, true, p[3] - p[0], p[4] - p[1]);
    add(e.b, false, p[(n - 2) * 3] - p[(n - 1) * 3], p[(n - 2) * 3 + 1] - p[(n - 1) * 3 + 1]);
  }
  // side[node][edge end]: +1 if it leaves along the node's reference direction (its first end's)
  const sideOf = (node: number, e: number, atA: boolean) => {
    const v = ends.get(node)!;
    const me = v.find((x) => x.e === e && x.atA === atA)!;
    return me.tx * v[0].tx + me.ty * v[0].ty >= 0 ? 1 : -1;
  };
  const sigma = new Map<number, number>();
  const nodeSign = new Map<number, number>();
  const edgeById = new Map(s.edges.map((e) => [e.id, e]));
  for (const start of ends.keys()) {
    if (nodeSign.has(start)) continue;
    nodeSign.set(start, 1);
    const queue = [start];
    while (queue.length) {
      const n = queue.pop()!;
      const sn = nodeSign.get(n)!;
      for (const end of ends.get(n)!) {
        if (sigma.has(end.e)) continue;
        const e = edgeById.get(end.e)!;
        // forward leaves n when this end leaves along the node's forward direction
        const away = sideOf(n, end.e, end.atA) === sn;
        const sg = end.atA === away ? 1 : -1;
        sigma.set(end.e, sg);
        const m = end.atA ? e.b : e.a;
        if (nodeSign.has(m)) continue;
        const awayM = !end.atA === (sg === 1);
        const k = sideOf(m, end.e, !end.atA);
        nodeSign.set(m, awayM ? k : -k);
        queue.push(m);
      }
    }
  }
  // Which side each line leans to: the middle of its stops, against the edge. A line whose stops
  // lie well to one side (a branch) takes that side of the bundle, so it leaves without crossing
  // the others; lines along the stretch keep id order. Pack metres, y north.
  const node = new Map(s.nodes.map((n) => [n.id, n]));
  const centre = new Map<number, [number, number]>();
  for (const l of s.lines) {
    const pts = l.stops.map((id) => node.get(id)).filter((n) => n !== undefined);
    if (pts.length) centre.set(l.id, [pts.reduce((a, n) => a + n.x, 0) / pts.length, pts.reduce((a, n) => a + n.y, 0) / pts.length]);
  }
  const onEdge = new Map<number, number[]>();
  for (const l of [...s.lines].sort((a, b) => a.id - b.id)) {
    if (!l.ok) continue;
    for (const e of new Set(l.path.map((p) => p >> 1))) {
      const v = onEdge.get(e) ?? [];
      v.push(l.id);
      onEdge.set(e, v);
    }
  }
  for (const [e, v] of onEdge) {
    const ed = edgeById.get(e), a = ed && node.get(ed.a), b = ed && node.get(ed.b);
    if (v.length < 2 || !a || !b) continue;
    const sg = sigma.get(e) ?? 1;
    const fx = (b.x - a.x) * sg, fy = (b.y - a.y) * sg, len = Math.hypot(fx, fy) || 1;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const lean = (id: number) => {
      const c = centre.get(id);
      if (!c) return 0;
      const left = (fx * (c[1] - my) - fy * (c[0] - mx)) / len;
      return left > LEAN_M ? 1 : left < -LEAN_M ? -1 : 0;
    };
    v.sort((p, q) => lean(p) - lean(q) || p - q);
  }
  return { sigma, onEdge };
}

/** A line leans to one side of a shared edge when the middle of its stops is this far off it, m. */
const LEAN_M = 400;

/** A line's slot on an edge, in line widths left of the edge's forward direction (0 alone). */
function slotOf(onEdge: Map<number, number[]>, edge: number, line: number): number {
  const v = onEdge.get(edge);
  if (!v || v.length < 2) return 0;
  return v.indexOf(line) - (v.length - 1) / 2;
}

export function strokesFromPreview(pts: Float32Array, bad: boolean): Strokes {
  const b = new StrokeBuilder();
  b.add(pts, TRACK, 0xffffffff, F_BLUEPRINT | (bad ? F_BAD : 0));
  return b.build();
}

export function networkFromState(s: ClockState): NetworkBuffers {
  const r = s.render;
  const edgePts = new Map<number, Float32Array>();
  const builtEdge = new Map<number, boolean>();
  const track = new StrokeBuilder();
  s.edges.forEach((e, i) => {
    const pts = r.edgePts.subarray(r.edgeOff[i] * 3, r.edgeOff[i + 1] * 3);
    edgePts.set(e.id, pts);
    builtEdge.set(e.id, e.built);
    track.add(pts, TRACK, e.id, e.built ? 0 : F_BLUEPRINT);
  });
  const maxLine = s.lines.reduce((m, l) => Math.max(m, l.id), -1);
  const lineColours = new Uint8Array(Math.max(1, maxLine + 1) * 4);
  const lines = new StrokeBuilder(true);
  const lineIds: string[] = [];
  const { sigma, onEdge } = layoutBundles(s, edgePts);
  const edgeLen = new Map(s.edges.map((e) => [e.id, e.lengthM]));
  const edgeEnds = new Map(s.edges.map((e) => [e.id, e]));
  const sampleOff = new Float32Array(r.samples.length / 2);
  const sampleStroke = new Int32Array(r.samples.length / 2).fill(-1);
  for (const l of s.lines) {
    lineColours.set([...rgb(l.colour), 255], l.id * 4);
    if (!l.ok) continue;
    lineIds.push(String(l.id));
    let d = 0;
    // offset per path entry, line widths left of run 0's direction
    const off = l.path.map((p) => slotOf(onEdge, p >> 1, l.id) * (sigma.get(p >> 1) ?? 1) * ((p & 1) === 1 ? -1 : 1));
    // the first stroke segment of each path entry, and the stop-to-stop stretch it is on (T-084)
    const firstStroke = l.path.map(() => -1);
    let stretch = 0, next = 1;
    l.path.forEach((p, k) => {
      const e = p >> 1;
      const pts = edgePts.get(e);
      const ends = edgeEnds.get(e);
      if (pts) {
        const flags = l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT);
        const v = onEdge.get(e);
        const slots = v && v.length > 1 ? v.length : 1;
        const info: [number, number, number, number] = [Math.min(stretch, Math.max(0, l.stops.length - 2)), slots > 1 ? v!.indexOf(l.id) : 0, slots, (sigma.get(e) ?? 1) * ((p & 1) === 1 ? -1 : 1)];
        if (pts.length >= 6) firstStroke[k] = lines.count;
        d = lines.add(pts, l.id, e, flags, (p & 1) === 1, d, off[k], info);
      }
      // a path entry runs a -> b, or b -> a when its low bit is set
      const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
      if (to === l.stops[next]) {
        stretch++;
        next++;
      }
    });
    // the same offsets on the samples trains ride (run 0 resampled every `step` m)
    const first = r.lineTable[l.id * 4], count = r.lineTable[l.id * 4 + 1], step = r.lineTable[l.id * 4 + 3];
    let j = 0, end = edgeLen.get(l.path[0] >> 1) ?? 0;
    for (let k = 0; k < count; k++) {
      const at = k * step;
      while (j < l.path.length - 1 && at > end) end += edgeLen.get(l.path[++j] >> 1) ?? 0;
      sampleOff[first + k] = off[j] ?? 0;
      sampleStroke[first + k] = firstStroke[j] ?? -1;
    }
  }
  const served = new Map<number, number>();
  for (const l of s.lines) for (const st of new Set(l.stops)) served.set(st, (served.get(st) ?? 0) + 1);
  // the widest bundle at each node, so a station's dot spans its lines
  const bundle = new Map<number, number>();
  for (const e of s.edges) {
    const k = onEdge.get(e.id)?.length ?? 0;
    for (const n of [e.a, e.b]) bundle.set(n, Math.max(bundle.get(n) ?? 0, k));
  }
  const st = s.nodes.filter((n) => n.platform > 0);
  const stations = new Float32Array(st.length * 3);
  // The station dot sits on the track: its node position, from the local polyline of an edge
  // that ends there (exact), else converted from lng/lat by the label code.
  const nodeLocal = new Map<number, [number, number]>();
  s.edges.forEach((e, i) => {
    const a = r.edgeOff[i], b = r.edgeOff[i + 1] - 1;
    nodeLocal.set(e.a, [r.edgePts[a * 3], r.edgePts[a * 3 + 1]]);
    nodeLocal.set(e.b, [r.edgePts[b * 3], r.edgePts[b * 3 + 1]]);
  });
  st.forEach((n, i) => {
    const p = nodeLocal.get(n.id) ?? lngLatLocal(n.lng, n.lat);
    stations.set([p[0], p[1], ((served.get(n.id) ?? 0) > 1 ? 1 : 0) | (n.stationBuilt ? 0 : 2) | (Math.min(bundle.get(n.id) ?? 0, 63) << 2)], i * 3);
  });
  const jn = s.nodes.filter((n) => n.ports >= 3 && n.platform === 0);
  const junctions = new Float32Array(jn.length * 2);
  jn.forEach((n, i) => junctions.set(nodeLocal.get(n.id) ?? lngLatLocal(n.lng, n.lat), i * 2));
  return {
    track: track.build(),
    lines: lines.build(),
    lineIds,
    lineColours,
    stations,
    stationIds: st.map((n) => String(n.id)),
    stationCount: st.length,
    junctions,
    junctionIds: jn.map((n) => n.id),
    phases: r.phases,
    meta: r.meta,
    samples: r.samples,
    lineTable: r.lineTable,
    sampleOff,
    sampleStroke,
    edgePts,
  };
}

function lngLatLocal(lng: number, lat: number): [number, number] {
  return toLocal(lng, lat);
}

// ---------------------------------------------------------------- trains on the CPU

/** Offset along a profile's run at trip time `tau`: the shader's arithmetic in float64. */
export function profileS(n: NetworkBuffers, profile: number, tau: number): number {
  const first = n.meta[profile * 4], count = n.meta[profile * 4 + 1];
  if (!count) return 0;
  let lo = first, hi = first + count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (n.phases[mid * 4] <= tau) lo = mid;
    else hi = mid - 1;
  }
  const t = n.phases[lo * 4], s = n.phases[lo * 4 + 1], v = n.phases[lo * 4 + 2], a = n.phases[lo * 4 + 3];
  let tt = Math.max(0, tau - t);
  if (a < 0) tt = Math.min(tt, v / -a);
  return s + v * tt + 0.5 * a * tt * tt;
}

/** Position (local units) at offset `d` along run 0 of a line. */
export function lineAt(n: NetworkBuffers, line: number, d: number): [number, number] {
  const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], len = n.lineTable[line * 4 + 2], step = n.lineTable[line * 4 + 3];
  if (count < 2) return [n.samples[first * 2], n.samples[first * 2 + 1]];
  const dd = Math.max(0, Math.min(len, d));
  const i0 = Math.min(Math.floor(dd / step), count - 2);
  const segLen = Math.max(1e-3, Math.min(step, len - i0 * step));
  const f = Math.max(0, Math.min(1, (dd - i0 * step) / segLen));
  const si = first + i0;
  return [n.samples[si * 2] + (n.samples[si * 2 + 2] - n.samples[si * 2]) * f, n.samples[si * 2 + 1] + (n.samples[si * 2 + 3] - n.samples[si * 2 + 1]) * f];
}

/** A trip's state at game time `t` (`dep` absolute): null if it is not on the track. `off`: its
 * line's sideways offset there, line widths left of run 0's direction `dx, dy` (T-062). */
export function tripAt(n: NetworkBuffers, profile: number, dep: number, t: number): { x: number; y: number; s: number; run: number; line: number; off: number; dx: number; dy: number } | null {
  const tau = t - dep;
  const tripS = n.meta[profile * 4 + 2];
  if (tau < 0 || tau > tripS || !n.meta[profile * 4 + 1]) return null;
  const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
  const s = profileS(n, profile, tau);
  const len = n.meta[profile * 4 + 3];
  const d = run === 0 ? s : len - s;
  const [x, y] = lineAt(n, line, d);
  const [bx, by] = lineAt(n, line, d - 5), [fx, fy] = lineAt(n, line, d + 5);
  let off = 0;
  if (n.sampleOff) {
    const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
    const i0 = Math.max(0, Math.min(count - 2, Math.floor(d / step)));
    const f = Math.max(0, Math.min(1, d / step - i0));
    off = n.sampleOff[first + i0] * (1 - f) + (n.sampleOff[first + i0 + 1] ?? 0) * f;
  }
  return { x, y, s, run, line, off, dx: fx - bx, dy: fy - by };
}
