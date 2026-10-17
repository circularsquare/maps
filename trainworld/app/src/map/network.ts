// What the renderer draws, as flat typed arrays in local units (map/geo.ts), built from the clock
// worker's state (T-025): track strokes per edge, line strokes along each line's path, station
// dots, and the keyframe tables trains are positioned from (notes/T-040.md "Keyframes").
//
// Trains: each trip is (profile, departure - epoch); profile p = (line * 2 + run) * 3 + level has
// a phase table (t, s, v, a) and meta (first phase, count, trip seconds, path length m); run 0 of
// each line is resampled every 50 m into `samples`; run 1's offset s is `length - s` on run 0.

import { rgb } from "../game/palette";
import type { ClockState } from "../workers/protocol";
import { anchor, toLocal, type Anchor } from "./geo";
import { M_BAD, M_CROSSING, M_JUNCTION, rampMarks } from "./markData";
import { bridgeEnds, PLATFORM_HALF_M, PLATFORM_POINTS, platformPoints, platformShape, reversed } from "./stationGeometry";

/** Colour index of bare track (not a line). */
export const TRACK = 0xffffff;
/** Segment flags. */
export const F_BLUEPRINT = 1;
export const F_FADED = 2;
/** a blueprint switched off (T-333): Construct skips it, drawn grey */
export const F_OFF = 64;
export const F_BAD = 16;
/** Live placement only: solid body/casing without changing its blueprint status. */
export const F_PLACEMENT = 32;

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
  /** Geometry speed limit km/h, per stroke segment (bare track). */
  speed?: Float32Array;
  /** per segment: distance along its edge or line at its start, local units (dashes) */
  dist: Float32Array;
  /** line strokes only, 4 per segment (T-084, demand views): the stop-to-stop stretch of run 0 it
   * is on (stop k to k + 1), then 0, 1, 1 (T-062's slot on the edge, slots and side, unused since
   * T-190 drew lines sharing track on top of each other). */
  info?: Float32Array;
  /** a preview only (T-629): its marks, `MARK_STRIDE` numbers each (map/markData.ts) */
  marks?: Float32Array;
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
  /** Per station: half the platform along its track at the platform's middle, local units (T-144). */
  stationAxis?: Float32Array;
  /** Per station, 4 each (T-189, `PlatformShape`): the platform's middle from the node (x, y; away
   * from the node at a dead end), half the angle its track turns along it (rad, + towards the left
   * of the axis), and half its true width, local units. Absent: straight, centred, no true width. */
  stationShape?: Float32Array;
  /** Per station, PLATFORM_POINTS x, y pairs (T-254): its track's centre line along the platform
   * from the node, local units (`PlatformShape.pts`). Absent: the arc of `stationShape`. */
  stationLine?: Float32Array;
  stationIds: string[];
  stationCount: number;
  /** Per station: its platform's length, m (the room riders have to wait on it, T-219). */
  stationLength?: Float32Array;
  /** The player's links (T-197) drawn as bridges (T-217): per link its two stations' indices here. */
  stationLinkIdx?: Int32Array;
  /** Per link: the bridge's two ends, local units (`bridgeEnds` in stationGeometry.ts). */
  stationLinks?: Float32Array;
  /** junction nodes (3+ edge ends, no station): x, y per node, for picking */
  junctions: Float32Array;
  junctionIds: number[];
  /** the marks for flat crossings, flat junctions, ramps and portals (T-629, T-655), `MARK_STRIDE`
   * numbers each (map/markData.ts) */
  marks?: Float32Array;
  // keyframes
  phases: Float32Array;
  meta: Float32Array;
  samples: Float32Array;
  lineTable: Float32Array;
  /** Actual 20 m cars per train, indexed by line id; absent in old synthetic fixtures. */
  lineCars?: Float32Array;
  /** per sample of `samples`: a line stroke segment of the same stretch of path, -1 if none
   * (T-084: its stretch gives the train's load) */
  sampleStroke?: Int32Array;
  /** per edge id: local polyline (x, y, z triples), for snapping and picking */
  edgePts: Map<number, Float32Array>;
  /** The draw anchor every position here is relative to (T-307, geo.ts): the clock worker's when
   * it built them. Absent (a synthetic network): the current one. */
  anchor?: Anchor;
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
    if (this.withInfo) this.info = g(this.info, 4);
    this.cap = cap;
  }
  /** pts: x, y, z triples; `info` as `Strokes.info`, for every segment added */
  add(pts: ArrayLike<number>, colour: number, edge: number, flags: number, reverse = false, dist0 = 0, info?: [number, number, number, number]): number {
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
      ...(this.withInfo ? { info: this.info.slice(0, n * 4) } : {}),
      count: n,
    };
  }
}

/** Whether a line has any train assigned (its fleet is above zero). */
export const runsTrains = (l: { trains: ArrayLike<number> }) => l.trains[0] + l.trains[1] + l.trains[2] > 0;

/**
 * The order lines sharing track are drawn in, bottom first (T-533; picking follows the same order).
 * Lines with no trains go under all that run trains; within each group the shorter route is on
 * top, ties to the higher id. T-062 drew them side by side, which T-190 dropped (Anita: "for now
 * lets not try to represent interlining. its a hard problem").
 */
export function lineDrawOrder<L extends { id: number; lengthM: number; trains: ArrayLike<number> }>(lines: readonly L[]): L[] {
  return [...lines].sort((a, b) => Number(runsTrains(a)) - Number(runsTrains(b)) || b.lengthM - a.lengthM || a.id - b.id);
}

/** The lines on each edge, by id. */
function linesOnEdges(s: ClockState) {
  const onEdge = new Map<number, number[]>();
  for (const l of [...s.lines].sort((a, b) => a.id - b.id)) {
    if (!l.ok) continue;
    for (const e of new Set(l.path.map((p) => p >> 1))) {
      const v = onEdge.get(e) ?? [];
      v.push(l.id);
      onEdge.set(e, v);
    }
  }
  return onEdge;
}

export function strokesFromPreview(pts: Float32Array, bad: boolean, speed?: Float32Array, placement = false, simMarks?: ArrayLike<number>): Strokes {
  const b = new StrokeBuilder();
  const flags = F_BLUEPRINT | (bad ? F_BAD : 0) | (placement ? F_PLACEMENT : 0);
  // a negative speed is a step with no track (between quad's two pairs' own ends, T-423): not drawn
  let from = 0;
  if (speed) {
    for (let k = 0; k < speed.length; k++) {
      if (speed[k] >= 0) continue;
      b.add(pts.subarray(from * 3, (k + 1) * 3), TRACK, 0xffffffff, flags);
      from = k + 1;
    }
    if (from > 0) speed = speed.filter((v) => v >= 0);
  }
  b.add(pts.subarray(from * 3), TRACK, 0xffffffff, flags);
  const strokes = b.build();
  strokes.marks = previewMarks(pts, simMarks);
  if (speed) {
    if (speed.length !== strokes.count) throw new Error("Preview speed/segment length mismatch");
    strokes.speed = speed;
  }
  return strokes;
}

/** A crossover's two diagonals as the placement preview (T-338), red when it cannot be built. */
export function strokesFromParts(parts: Float32Array[], bad: boolean, placement = true): Strokes {
  const b = new StrokeBuilder();
  for (const pts of parts) b.add(pts, TRACK, 0xffffffff, F_BLUEPRINT | (placement ? F_PLACEMENT : 0) | (bad ? F_BAD : 0));
  return b.build();
}

/** Several edges' local polylines (x, y, z triples) as one stroke set, for the drag-selected
 * blueprint's highlight band (T-333). */
export function strokesFromEdges(lines: Float32Array[]): Strokes {
  const b = new StrokeBuilder();
  for (const pts of lines) b.add(pts, TRACK, 0xffffffff, F_PLACEMENT);
  return b.build();
}

/** A preview's marks (T-629): the ramps and portals of its track, then the crossings and flat
 * junctions the track model found (`sim`: x, y, 0 crossing or 1 junction), all in the blueprint
 * tier. */
export function previewMarks(pts: ArrayLike<number>, sim?: ArrayLike<number>): Float32Array | undefined {
  const m: number[] = [];
  rampMarks(pts, 1, m);
  if (sim) for (let k = 0; k + 2 < sim.length; k += 3) m.push(sim[k], sim[k + 1], sim[k + 2] === 1 ? M_JUNCTION : M_CROSSING, 0, 1, 0);
  return m.length ? new Float32Array(m) : undefined;
}

/** Stroke sets laid end to end, into arrays of exactly their total size. */
function concatStrokes(parts: Strokes[], withInfo: boolean): Strokes {
  let n = 0;
  for (const p of parts) n += p.count;
  const out: Strokes = {
    seg: new Float32Array(n * 4), colour: new Uint32Array(n), edge: new Uint32Array(n), level: new Float32Array(n),
    flags: new Float32Array(n), dist: new Float32Array(n), ...(withInfo ? { info: new Float32Array(n * 4) } : {}), count: n,
  };
  let at = 0;
  for (const p of parts) {
    out.seg.set(p.seg, at * 4);
    out.colour.set(p.colour, at);
    out.edge.set(p.edge, at);
    out.level.set(p.level, at);
    out.flags.set(p.flags, at);
    out.dist.set(p.dist, at);
    if (withInfo) out.info!.set(p.info!, at * 4);
    at += p.count;
  }
  return out;
}

const sameArray = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** Bit for bit, so a NaN height matches itself. */
const sameFloats = (a: Float32Array, b: Float32Array) => {
  if (a.length !== b.length) return false;
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
};

/**
 * What the last build made, per edge, line and station, so an edit rebuilds only the pieces whose
 * inputs changed (T-496). Every entry is checked against its inputs, never trusted, so a reused
 * piece is exactly what a full build would make; entries not seen in a build are dropped.
 */
export interface NetCache {
  edges: Map<number, { pts: Float32Array; a: number; b: number; flags: number; strokes: Strokes; ramps: number[] }>;
  lines: Map<number, { path: number[]; stops: number[]; running: boolean; strokes: Strokes; firstStroke: number[] }>;
  stations: Map<number, { platform: number; lat: number; first: number; ways: number[]; ax: number; ay: number; shape: [number, number, number, number]; line: ArrayLike<number> }>;
}

export const newNetCache = (): NetCache => ({ edges: new Map(), lines: new Map(), stations: new Map() });

/** The renderer's buffers from a state. With `cache` (the clock worker's), pieces whose inputs are
 * as in the last build are reused; the result is the same either way. */
export function networkFromState(s: ClockState, cache: NetCache = newNetCache()): NetworkBuffers {
  const r = s.render;
  const edgePts = new Map<number, Float32Array>();
  const builtEdge = new Map<number, boolean>();
  const offEdge = new Set<number>(s.edges.filter((e) => e.off).map((e) => e.id));
  /** edges whose polyline, ends and flags are as in the last build */
  const kept = new Set<number>();
  const trackParts: Strokes[] = [];
  const edgesSeen: NetCache["edges"] = new Map();
  const speeds = new Float32Array(Math.max(0, r.edgePts.length / 3 - s.edges.length));
  let speedAt = 0;
  const marks: number[] = [];
  s.edges.forEach((e, i) => {
    const pts = r.edgePts.subarray(r.edgeOff[i] * 3, r.edgeOff[i + 1] * 3);
    edgePts.set(e.id, pts);
    builtEdge.set(e.id, e.built);
    const flags = e.built ? 0 : F_BLUEPRINT | (e.off ? F_OFF : 0);
    let c = cache.edges.get(e.id);
    if (c && c.a === e.a && c.b === e.b && c.flags === flags && sameFloats(c.pts, pts)) kept.add(e.id);
    else {
      const b = new StrokeBuilder();
      b.add(pts, TRACK, e.id, flags);
      const ramps: number[] = [];
      rampMarks(pts, e.built ? 0 : 1, ramps);
      c = { pts: pts.slice(), a: e.a, b: e.b, flags, strokes: b.build(), ramps };
    }
    edgesSeen.set(e.id, c);
    trackParts.push(c.strokes);
    for (const v of c.ramps) marks.push(v);
    for (let k = r.edgeOff[i]; k < r.edgeOff[i + 1] - 1; k++) speeds[speedAt++] = r.edgeSpeed?.[k] ?? e.vminKmh;
  });
  cache.edges = edgesSeen;
  const maxLine = s.lines.reduce((m, l) => Math.max(m, l.id), -1);
  const lineColours = new Uint8Array(Math.max(1, maxLine + 1) * 4);
  const lineCars = new Float32Array(r.lineTable.length / 4).fill(8);
  for (const l of s.lines) lineCars[l.id] = Math.max(1, l.cars);
  const lineParts: Strokes[] = [];
  let lineCount = 0;
  const linesSeen: NetCache["lines"] = new Map();
  const lineIds: string[] = [];
  const onEdge = linesOnEdges(s);
  const edgeLen = new Map(s.edges.map((e) => [e.id, e.lengthM]));
  const edgeEnds = new Map(s.edges.map((e) => [e.id, e]));
  const sampleStroke = new Int32Array(r.samples.length / 2).fill(-1);
  for (const l of s.lines) lineColours.set([...rgb(l.colour), 255], l.id * 4);
  for (const l of lineDrawOrder(s.lines)) {
    if (!l.ok) continue;
    lineIds.push(String(l.id));
    let c = cache.lines.get(l.id);
    if (!c || c.running !== l.running || !sameArray(c.path, l.path) || !sameArray(c.stops, l.stops) || !l.path.every((p) => kept.has(p >> 1))) {
      const lines = new StrokeBuilder(true);
      let d = 0;
      // the first stroke segment of each path entry (in this line's strokes), and the stop-to-stop
      // stretch it is on (T-084)
      const firstStroke = l.path.map(() => -1);
      let stretch = 0, next = 1;
      l.path.forEach((p, k) => {
        const e = p >> 1;
        const pts = edgePts.get(e);
        const ends = edgeEnds.get(e);
        if (pts) {
          const flags = l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0));
          // one slot per edge since T-190: no line sits beside another
          const info: [number, number, number, number] = [Math.min(stretch, Math.max(0, l.stops.length - 2)), 0, 1, 1];
          if (pts.length >= 6) firstStroke[k] = lines.count;
          d = lines.add(pts, l.id, e, flags, (p & 1) === 1, d, info);
        }
        // a path entry runs a -> b, or b -> a when its low bit is set
        const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
        if (to === l.stops[next]) {
          stretch++;
          next++;
        }
      });
      c = { path: l.path.slice(), stops: l.stops.slice(), running: l.running, strokes: lines.build(), firstStroke };
    }
    linesSeen.set(l.id, c);
    lineParts.push(c.strokes);
    const base = lineCount;
    lineCount += c.strokes.count;
    // each train sample's stroke segment (run 0 resampled every `step` m)
    const firstStroke = c.firstStroke;
    const first = r.lineTable[l.id * 4], count = r.lineTable[l.id * 4 + 1], step = r.lineTable[l.id * 4 + 3];
    let j = 0, end = edgeLen.get(l.path[0] >> 1) ?? 0;
    for (let k = 0; k < count; k++) {
      const at = k * step;
      while (j < l.path.length - 1 && at > end) end += edgeLen.get(l.path[++j] >> 1) ?? 0;
      const f = firstStroke[j] ?? -1;
      sampleStroke[first + k] = f < 0 ? -1 : base + f;
    }
  }
  cache.lines = linesSeen;
  const served = new Map<number, number>();
  for (const l of s.lines) for (const st of new Set(l.stops)) served.set(st, (served.get(st) ?? 0) + 1);
  // whether a line runs through each node, so a station's box covers it (one line width since T-190)
  const lined = new Set<number>();
  for (const e of s.edges) if (onEdge.get(e.id)?.length) lined.add(e.a).add(e.b);
  const st = s.nodes.filter((n) => n.platform > 0);
  const stations = new Float32Array(st.length * 3);
  const stationAxis = new Float32Array(st.length * 2);
  const stationShape = new Float32Array(st.length * 4);
  const stationLine = new Float32Array(st.length * 2 * PLATFORM_POINTS);
  // Most lines first, then edge id: junction platform direction is stable across rebuilds.
  const platformEdge = new Map<number, number>();
  const waysAt = new Map<number, number[]>();
  for (const e of [...s.edges].sort((a,b) => (onEdge.get(b.id)?.length ?? 0)-(onEdge.get(a.id)?.length ?? 0) || a.id-b.id))
    for (const node of [e.a,e.b]) {
      if (!platformEdge.has(node)) platformEdge.set(node,e.id);
      const v = waysAt.get(node) ?? [];
      v.push(e.id);
      waysAt.set(node, v);
    }
  const edgeOf = new Map(s.edges.map((e) => [e.id, e]));
  /** the track leaving `node` along edge `id`, starting at the node */
  const way = (node: number, id: number) => {
    const p = edgePts.get(id), e = edgeOf.get(id);
    return !p || !e ? new Float32Array(0) : e.a === node ? p : reversed(p);
  };
  // The station dot sits on the track: its node position, from the local polyline of an edge
  // that ends there (exact), else converted from lng/lat by the label code.
  const nodeLocal = new Map<number, [number, number]>();
  s.edges.forEach((e, i) => {
    const a = r.edgeOff[i], b = r.edgeOff[i + 1] - 1;
    nodeLocal.set(e.a, [r.edgePts[a * 3], r.edgePts[a * 3 + 1]]);
    nodeLocal.set(e.b, [r.edgePts[b * 3], r.edgePts[b * 3 + 1]]);
  });
  const stationsSeen: NetCache["stations"] = new Map();
  st.forEach((n, i) => {
    const p = nodeLocal.get(n.id) ?? lngLatLocal(n.lng, n.lat);
    const ids = waysAt.get(n.id) ?? [];
    const first = Math.max(0, ids.indexOf(platformEdge.get(n.id) ?? -1));
    let c = cache.stations.get(n.id);
    if (!c || c.platform !== n.platform || c.lat !== n.lat || c.first !== first || !sameArray(c.ways, ids) || !ids.every((id) => kept.has(id))) {
      // the platform along its track, curved as it is (T-189); local units are metres / cos(lat)
      const perM = 1 / Math.cos(n.lat * Math.PI / 180);
      const sh = ids.length
        ? platformShape(ids.map((id) => way(n.id, id)), first, n.platform / 2 * perM)
        : { mx: 0, my: 0, ax: n.platform / 2 * perM, ay: 0, phi: 0 };
      c = { platform: n.platform, lat: n.lat, first, ways: ids, ax: sh.ax, ay: sh.ay, shape: [sh.mx, sh.my, sh.phi, PLATFORM_HALF_M * perM], line: platformPoints(sh) };
    }
    stationsSeen.set(n.id, c);
    stationAxis.set([c.ax, c.ay], i * 2);
    stationShape.set(c.shape, i * 4);
    stationLine.set(c.line, i * 2 * PLATFORM_POINTS);
    stations.set([p[0], p[1], ((served.get(n.id) ?? 0) > 1 ? 1 : 0) | (n.stationBuilt ? 0 : 2) | (lined.has(n.id) ? 4 : 0) | (n.stationOff ? 8 : 0)], i * 3);
  });
  cache.stations = stationsSeen;
  // the player's links as bridges between the two platforms' nearest points (T-217)
  const stationIndex = new Map(st.map((n, i) => [n.id, i]));
  const shapeAt = (i: number) => ({ mx: stationShape[i * 4], my: stationShape[i * 4 + 1], ax: stationAxis[i * 2], ay: stationAxis[i * 2 + 1], phi: stationShape[i * 4 + 2], pts: stationLine.subarray(i * 2 * PLATFORM_POINTS, (i + 1) * 2 * PLATFORM_POINTS) });
  const linkIdx: number[] = [], linkEnds: number[] = [];
  for (let k = 0; k + 1 < (s.links?.length ?? 0); k += 2) {
    const i = stationIndex.get(s.links[k]), j = stationIndex.get(s.links[k + 1]);
    if (i === undefined || j === undefined) continue;
    linkIdx.push(i, j);
    linkEnds.push(...bridgeEnds(stations[i * 3], stations[i * 3 + 1], shapeAt(i), stations[j * 3], stations[j * 3 + 1], shapeAt(j)));
  }
  const jn = s.nodes.filter((n) => n.ports >= 3 && n.platform === 0);
  const junctions = new Float32Array(jn.length * 2);
  jn.forEach((n, i) => junctions.set(nodeLocal.get(n.id) ?? lngLatLocal(n.lng, n.lat), i * 2));
  // the marks (T-629): a junction at ground level (a flyover is not one) is built once three of its
  // tracks are; a crossing once both of its are
  jn.forEach((n, i) => {
    if (!n.flying) marks.push(junctions[i * 2], junctions[i * 2 + 1], M_JUNCTION, 0, n.builtPorts >= 3 ? 0 : 1, n.level);
  });
  const cx = s.crossings;
  if (cx) for (let k = 0; k + 4 < cx.length; k += 5) marks.push(cx[k], cx[k + 1], cx[k + 2] === 1 ? M_BAD : M_CROSSING, 0, cx[k + 4] === 1 ? 0 : 1, cx[k + 3]);
  return {
    track: { ...concatStrokes(trackParts, false), speed: speeds.subarray(0, speedAt) },
    lines: concatStrokes(lineParts, true),
    lineIds,
    lineColours,
    stations,
    stationAxis,
    stationIds: st.map((n) => String(n.id)),
    stationCount: st.length,
    stationLength: new Float32Array(st.map((n) => n.platform)),
    stationLinkIdx: new Int32Array(linkIdx),
    stationLinks: new Float32Array(linkEnds),
    junctions,
    junctionIds: jn.map((n) => n.id),
    marks: new Float32Array(marks),
    phases: r.phases,
    meta: r.meta,
    samples: r.samples,
    lineTable: r.lineTable,
    lineCars,
    stationShape,
    stationLine,
    sampleStroke,
    edgePts,
    anchor: anchor(),
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

/** A trip's state at game time `t` (`dep` absolute): null if it is not on the track. `dx, dy`:
 * run 0's direction there. */
export function tripAt(n: NetworkBuffers, profile: number, dep: number, t: number, end = -Infinity): { x: number; y: number; s: number; run: number; line: number; dx: number; dy: number } | null {
  const tau = t - dep;
  const tripS = n.meta[profile * 4 + 2];
  // `end`: drawn until then, standing at the terminal for its next departure (T-206)
  if (tau < 0 || (tau > tripS && t > end) || !n.meta[profile * 4 + 1]) return null;
  const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
  const s = profileS(n, profile, tau);
  const len = n.meta[profile * 4 + 3];
  const d = run === 0 ? s : len - s;
  return { ...pointAt(n, line, d), s, run, line };
}

/** Position and direction used by both train and car picking. */
function pointAt(n: NetworkBuffers, line: number, d: number) {
  const [x, y] = lineAt(n, line, d);
  const [bx, by] = lineAt(n, line, d - 5), [fx, fy] = lineAt(n, line, d + 5);
  return { x, y, dx: fx - bx, dy: fy - by };
}

/** A close-up car's centre and bogies, matching the GPU's terminal clamp and 20 m pitch. */
export function trainCarAt(n: NetworkBuffers, at: NonNullable<ReturnType<typeof tripAt>>, car: number) {
  const cars = Math.max(1, n.lineCars?.[at.line] ?? 8);
  const len = n.lineTable[at.line * 4 + 2], total = Math.min(cars * 20, len);
  const d = at.run === 0 ? at.s : len - at.s;
  const centre = Math.max(total / 2, Math.min(len - total / 2, d));
  const carD = centre + (car + 0.5) * total / cars - total / 2;
  const half = Math.max(total / cars / 2 - 0.5, 1);
  return { ...pointAt(n, at.line, carD), back: lineAt(n, at.line, carD - half), front: lineAt(n, at.line, carD + half) };
}
