// What the renderer draws, as flat typed arrays in local units (map/geo.ts), built from the clock
// worker's state (T-025): track strokes per edge, line strokes along each line's path, station
// dots, and the keyframe tables trains are positioned from (notes/T-040.md "Keyframes").
//
// Trains: each trip is (profile, departure - epoch); profile p = (line * 2 + run) * 3 + level has
// a phase table (t, s, v, a) and meta (first phase, count, trip seconds, path length m); run 0 of
// each line is resampled every 50 m into `samples`; run 1's offset s is `length - s` on run 0.

import { rgb } from "../game/palette";
import { hasUpgrade, UPGRADE } from "../game/upgrades";
import type { ClockState } from "../workers/protocol";
import { anchor, kAtY, toLocal, type Anchor } from "./geo";
import { M_BAD, M_CROSSING, M_JUNCTION, rampMarks } from "./markData";
import { annexSides } from "./annexSide";
import { bridgeEnds, PLATFORM_HALF_M, PLATFORM_POINTS, platformPoints, platformShape, reversed, ST_PARK, ST_ROOM, ST_TERMINAL, type PlatformShape } from "./stationGeometry";

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
/** A crossover where lines turn back (T-658d): its lane values are signed, drawn close in only. */
export const F_TURNBACK = 128;
/** High-speed track (T-817), and lines over it: a thin centre line along the stroke. */
export const F_HSR = 256;

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
  /** Built track only (T-1271): per segment, the fastest speed a running line's trains reach at its
   * middle, km/h, either direction; -1 where no line with trains runs. */
  trainSpeed?: Float32Array;
  /** per segment: distance along its edge or line at its start, local units (read on the CPU only) */
  dist: Float32Array;
  /** line strokes only, per segment (T-084, demand views): the stop-to-stop stretch of run 0 it is
   * on (stop k to k + 1) */
  stretch?: Float32Array;
  /** a preview only (T-629): its marks, `MARK_STRIDE` numbers each (map/markData.ts) */
  marks?: Float32Array;
  /** 2 per segment (T-658): how far apart its two tracks are drawn at its start and end, 0 (single
   * track, one track on the centre line) to 255 (double track); absent: all double */
  lane?: Uint8Array;
  count: number;
}

export interface NetworkBuffers {
  track: Strokes;
  lines: Strokes;
  /** line ids drawn (line strokes), for picking */
  lineIds: string[];
  /** per line id: r, g, b, 255 */
  lineColours: Uint8Array;
  /** per station: x, y, flags (1 transfer, 2 blueprint, 4 a line runs through, 8 switched off, 16
   * on double track: a platform each side close in, T-658; ST_ROOM, ST_FLIP, ST_PARK and
   * ST_TERMINAL its upgrades as drawn, T-1416) */
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
  /** per sample of `samples`: how far apart its track's two tracks are drawn there (T-658), 0
   * single track to 1 double, as `Strokes.lane`; absent: 1 */
  sampleLane?: Float32Array;
  /** the crossovers where lines turn back, drawn close in only (T-658d, `turnbackStrokes`) */
  turnbacks?: Strokes;
  /** per edge id: local polyline (x, y, z triples), for snapping and picking */
  edgePts: Map<number, Float32Array>;
  /** The draw anchor every position here is relative to (T-307, geo.ts): the clock worker's when
   * it built them. Absent (a synthetic network): the current one. */
  anchor?: Anchor;
  /** per train sample, its stop-to-stop stretch (`Strokes.stretch` of its stroke, -1 none) */
  sampleStretch?: Float32Array;
  /** T-1029: the clock worker's number for this build, and `patch` when it was posted as changes to
   * the build numbered `patch.base` (`NetPatch`) */
  serial?: number;
  patch?: NetPatch;
}

/**
 * How a build differs from the one before it (T-1029, T-1028). `track` and `lines`: runs of
 * (start, count, start in the old strokes or -1 where the segments are new), covering the new
 * strokes in order. `same`: the fields left out of the message as identical to the old build's
 * (the client puts them back, `mergeNet`). `edgeIds`: the edges in `edgePts` order, with `edgePts`
 * then holding only the edges whose points changed.
 */
export interface NetPatch {
  base: number;
  track: Int32Array;
  lines: Int32Array;
  oldTrack: number;
  oldLines: number;
  same: string[];
  edgeIds: Uint32Array;
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
  /** `Strokes.stretch`, only when built with `withStretch` */
  stretch = new Float32Array(0);
  lane = new Uint8Array(0);
  constructor(private withStretch = false) {}
  private grow(need: number) {
    if (need <= this.cap) return;
    // a fresh builder takes what it is asked for: most hold one edge, and a 1,024 floor made every
    // edge's eight arrays ~50 kB (seconds of allocation on a 2,000-edge load, T-1156)
    const cap = this.cap ? Math.max(need, this.cap * 2) : need;
    const g = <T extends Float32Array | Uint32Array | Uint8Array>(a: T, k: number): T => {
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
    this.lane = g(this.lane, 2);
    if (this.withStretch) this.stretch = g(this.stretch, 1);
    this.cap = cap;
  }
  /** pts: x, y, z triples; `stretch` as `Strokes.stretch`, for every segment added; `lane`: per point of
   * `pts`, 0 single track to 255 double (`Strokes.lane`), or one value for all */
  add(pts: ArrayLike<number>, colour: number, edge: number, flags: number, reverse = false, dist0 = 0, stretch?: number, lane: number | ArrayLike<number> = 255): number {
    const n = pts.length / 3;
    let d = dist0;
    this.grow(this.count + Math.max(0, n - 1));
    if (stretch !== undefined && this.withStretch) this.stretch.fill(stretch, this.count, this.count + Math.max(0, n - 1));
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
      this.lane[c * 2] = typeof lane === "number" ? lane : lane[i];
      this.lane[c * 2 + 1] = typeof lane === "number" ? lane : lane[j];
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
      lane: this.lane.slice(0, n * 2),
      ...(this.withStretch ? { stretch: this.stretch.slice(0, n) } : {}),
      count: n,
    };
  }
}

/** Whether a line has any train assigned (its fleet is above zero). */
export const runsTrains = (l: { trains: ArrayLike<number> }) => l.trains[0] + l.trains[1] + l.trains[2] > 0;

/**
 * The order lines sharing track are drawn in, bottom first (T-533; picking follows the same order).
 * Lines with no trains go under all that run trains; within each group the shorter route is on
 * top, ties to the higher id (T-190).
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

/** `lane`: as `Strokes.lane`, 0 for single track (T-658); `hsr`: high-speed track's look (T-817) */
export function strokesFromPreview(pts: Float32Array, bad: boolean, speed?: Float32Array, placement = false, simMarks?: ArrayLike<number>, lane = 255, hsr = false): Strokes {
  const b = new StrokeBuilder();
  const flags = F_BLUEPRINT | (bad ? F_BAD : 0) | (placement ? F_PLACEMENT : 0) | (hsr ? F_HSR : 0);
  // a negative speed is a step with no track (between quad's two pairs' own ends, T-423): not drawn
  let from = 0;
  if (speed) {
    for (let k = 0; k < speed.length; k++) {
      if (speed[k] >= 0) continue;
      b.add(pts.subarray(from * 3, (k + 1) * 3), TRACK, 0xffffffff, flags, false, 0, undefined, lane);
      from = k + 1;
    }
    if (from > 0) speed = speed.filter((v) => v >= 0);
  }
  b.add(pts.subarray(from * 3), TRACK, 0xffffffff, flags, false, 0, undefined, lane);
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
function concatStrokes(parts: Strokes[], withStretch: boolean): Strokes {
  let n = 0;
  for (const p of parts) n += p.count;
  const out: Strokes = {
    seg: new Float32Array(n * 4), colour: new Uint32Array(n), edge: new Uint32Array(n), level: new Float32Array(n),
    flags: new Float32Array(n), dist: new Float32Array(n), lane: new Uint8Array(n * 2), ...(withStretch ? { stretch: new Float32Array(n) } : {}), count: n,
  };
  let at = 0;
  for (const p of parts) {
    if (p.lane) out.lane!.set(p.lane, at * 2);
    else out.lane!.fill(255, at * 2, (at + p.count) * 2);
    out.seg.set(p.seg, at * 4);
    out.colour.set(p.colour, at);
    out.edge.set(p.edge, at);
    out.level.set(p.level, at);
    out.flags.set(p.flags, at);
    out.dist.set(p.dist, at);
    if (withStretch) out.stretch!.set(p.stretch!, at);
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

/** Where double track narrows to single, m (T-658). */
const TAPER_M = 40;

/** An edge's points as drawn, with their `Strokes.lane` values; `src`, when points were added: per
 * segment, the edge's own segment it lies on. */
interface LanePoints { lpts: Float32Array; lane: number | Uint8Array; src: number[] | null }

/**
 * `lanes`: 1 double track, 2 and 4 narrowing to single track over TAPER_M at the a and b end (half
 * the edge at most each). A point is added where a taper starts, so it never spans a long leg.
 */
function lanePoints(pts: Float32Array, lanes: number): LanePoints {
  const n = pts.length / 3;
  const tapA = (lanes & 2) !== 0, tapB = (lanes & 4) !== 0;
  if (!(lanes & 1) || (!tapA && !tapB) || n < 2) return { lpts: pts, lane: lanes & 1 ? 255 : 0, src: null };
  const at = new Float64Array(n);
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
  const len = at[n - 1];
  const L = Math.max(1e-6, Math.min(TAPER_M * kAtY(pts[1]), tapA && tapB ? len / 2 : len));
  const cuts = [...(tapA ? [L] : []), ...(tapB && (!tapA || len - L > L + 1e-6) ? [len - L] : [])];
  const value = (d: number) => Math.round(255 * Math.max(0, Math.min(1, tapA ? d / L : 1, tapB ? (len - d) / L : 1)));
  const out: number[] = [pts[0], pts[1], pts[2]], lane: number[] = [value(0)], src: number[] = [];
  for (let k = 0; k < n - 1; k++) {
    for (const c of cuts) {
      if (c <= at[k] + 1e-6 || c >= at[k + 1] - 1e-6) continue;
      const f = (c - at[k]) / (at[k + 1] - at[k]);
      for (let j = 0; j < 3; j++) out.push(pts[k * 3 + j] + (pts[k * 3 + 3 + j] - pts[k * 3 + j]) * f);
      lane.push(value(c));
      src.push(k);
    }
    out.push(pts[k * 3 + 3], pts[k * 3 + 4], pts[k * 3 + 5]);
    lane.push(value(at[k + 1]));
    src.push(k);
  }
  return { lpts: new Float32Array(out), lane: new Uint8Array(lane), src };
}

/** Length of a turn-back crossover's diagonal, m (T-658d). */
const TURNBACK_M = 50;

/**
 * The crossovers the game assumes where a line turns back on double track (T-658d; SPEC 6.2: a
 * dead-end terminus uses both its platform tracks, a line ending at a through station crosses back
 * behind the platform): a scissors just outside a dead-end terminus's platform, one diagonal just
 * outside a through station's platform on the side the line arrives from, from the arrival track
 * to the departure track. Drawn along the line's own strokes with `Strokes.lane` signed (`F_TURNBACK`:
 * 0 the left track of run 0's direction, 255 the right).
 */
function turnbackStrokes(s: ClockState, lines: NetCache["lines"], endCount: Map<number, number>, more: (b: StrokeBuilder) => void): Strokes {
  const b = new StrokeBuilder();
  more(b);
  const nodes = new Map(s.nodes.map((n) => [n.id, n]));
  for (const l of s.lines) {
    const st = lines.get(l.id)?.strokes;
    const last = l.stops.length - 1;
    if (!l.ok || !st || !st.count || last < 1 || l.stops[0] === l.stops[last]) continue;
    const segLen = (i: number) => Math.hypot(st.seg[i * 4 + 2] - st.seg[i * 4], st.seg[i * 4 + 3] - st.seg[i * 4 + 1]);
    const D = st.dist[st.count - 1] + segLen(st.count - 1);
    /** run 0's strokes from `s0` to `s1`, the lane going from `v0` to `v1` (-1 left, 1 right) */
    const along = (s0: number, s1: number, v0: number, v1: number, flags: number) => {
      s0 = Math.max(0, s0);
      s1 = Math.min(D, s1);
      if (s1 - s0 < 1e-6) return;
      const pts: number[] = [], lane: number[] = [];
      const push = (x: number, y: number, at: number) => {
        pts.push(x, y, 0);
        lane.push(Math.round(127.5 * (1 + v0 + (v1 - v0) * (at - s0) / (s1 - s0))));
      };
      let edge = -1;
      // `dist` never falls along a line's strokes: start a little before s0, stop past s1 (T-648:
      // a scan of every stroke per call was a third of an edit's buffer build on New York)
      let lo = 0, hi = st.count;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (st.dist[mid] <= s0) lo = mid + 1;
        else hi = mid;
      }
      for (let i = Math.max(0, lo - 2); i < st.count; i++) {
        const a = st.dist[i], len = segLen(i), e = a + len;
        if (a > s1) break;
        if (e < s0 || len < 1e-9) continue;
        if (edge < 0) edge = st.edge[i];
        const x0 = st.seg[i * 4], y0 = st.seg[i * 4 + 1], dx = st.seg[i * 4 + 2] - x0, dy = st.seg[i * 4 + 3] - y0;
        const f0 = Math.max(0, (s0 - a) / len), f1 = Math.min(1, (s1 - a) / len);
        if (!pts.length) push(x0 + dx * f0, y0 + dy * f0, a + len * f0);
        push(x0 + dx * f1, y0 + dy * f1, a + len * f1);
      }
      if (pts.length >= 6) b.add(pts, TRACK, edge, flags | F_TURNBACK, false, 0, undefined, lane);
    };
    for (const atEnd of [false, true]) {
      const node = nodes.get(l.stops[atEnd ? last : 0]);
      const i = atEnd ? st.count - 1 : 0;
      if (!node || !node.platform || (st.lane?.[i * 2 + (atEnd ? 1 : 0)] ?? 255) < 255) continue;
      const k = kAtY(st.seg[i * 4 + (atEnd ? 3 : 1)]);
      const dead = endCount.get(node.id) === 1;
      // the platform's open end at a dead end, else its end on the side the line arrives from
      const p = node.platform * k * (dead ? 1 : 0.5), T = TURNBACK_M * k;
      const s0 = atEnd ? D - p - T : p;
      const flags = node.stationBuilt ? 0 : F_FADED;
      along(s0, s0 + T, -1, 1, flags);
      if (dead) along(s0, s0 + T, 1, -1, flags);
    }
  }
  return b.build();
}

/** The demand level whose run profiles give the Speed view's train speeds (T-1271): the levels
 * differ in how long trains stand, not in how fast they move between stops. */
const SPEED_LEVEL = 1;

/** A run profile's speed at `x` m along its path, km/h: the moving phase holding `x` (phases are
 * t, s, v, a, notes/T-040.md "Keyframes"; a stop's standing phase spans no distance). */
function speedAt(ph: Float32Array, first: number, count: number, x: number): number {
  let lo = first, hi = first + count - 1;
  if (count <= 0) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ph[mid * 4 + 1] <= x) lo = mid;
    else hi = mid - 1;
  }
  const s = ph[lo * 4 + 1], v = ph[lo * 4 + 2], a = ph[lo * 4 + 3];
  return Math.sqrt(Math.max(0, v * v + 2 * a * Math.max(0, x - s))) * 3.6;
}

/** A running line's train speeds (T-1271), per entry of its path: at the middle of each stroke
 * segment of that edge, the faster of its two runs there, km/h; null where the edge is not built.
 * Null for a line with no trains. */
function lineTrainSpeeds(l: ClockState["lines"][number], r: ClockState["render"], strokesOf: (e: number) => Strokes | undefined, lens: number[], built: (e: number) => boolean): (Float32Array | null)[] | null {
  if (!l.ok || !l.running || Math.max(...l.trains) <= 0) return null;
  const runs = [0, 1].map((run) => {
    const m = ((l.id * 2 + run) * 3 + SPEED_LEVEL) * 4;
    return { first: r.meta[m], count: r.meta[m + 1], len: r.meta[m + 3] };
  });
  if (runs.some((p) => !(p.count > 0))) return null;
  let off = 0;
  return l.path.map((p, k) => {
    const e = p >> 1, len = lens[k];
    const st = strokesOf(e);
    let out: Float32Array | null = null;
    if (st && built(e)) {
      const n = st.count;
      out = new Float32Array(n);
      const segLen = (j: number) => Math.hypot(st.seg[j * 4 + 2] - st.seg[j * 4], st.seg[j * 4 + 3] - st.seg[j * 4 + 1]);
      const total = n ? st.dist[n - 1] + segLen(n - 1) : 0;
      for (let j = 0; j < n; j++) {
        const f = total > 0 ? (st.dist[j] + segLen(j) / 2) / total : 0.5;
        const x0 = off + ((p & 1) === 1 ? 1 - f : f) * len;
        out[j] = Math.max(speedAt(r.phases, runs[0].first, runs[0].count, x0), speedAt(r.phases, runs[1].first, runs[1].count, runs[1].len - x0));
      }
    }
    off += len;
    return out;
  });
}

/** The phase tables `lineTrainSpeeds` reads for a line, one after the other with each run's length. */
function speedPhases(id: number, r: ClockState["render"]): Float32Array {
  const parts: number[] = [];
  let n = 0;
  const at: [number, number][] = [];
  for (const run of [0, 1]) {
    const m = ((id * 2 + run) * 3 + SPEED_LEVEL) * 4;
    const first = r.meta[m], count = r.meta[m + 1];
    parts.push(r.meta[m + 1], r.meta[m + 3]);
    at.push([first, count > 0 ? count : 0]);
    n += count > 0 ? count * 4 : 0;
  }
  const out = new Float32Array(4 + n);
  out.set(parts);
  let o = 4;
  for (const [first, count] of at) {
    out.set(r.phases.subarray(first * 4, (first + count) * 4), o);
    o += count * 4;
  }
  return out;
}

/** A terminal's sidings (T-1281), m: how far they run past the platform's end, and over how much of
 * that each leaves its running track. */
const SIDING_M = 160, SIDING_LEAD_M = 40;
/** Where the two sidings lie between the running tracks (`F_TURNBACK` lanes, -1 and 1 the tracks). */
const SIDING_LANE = 0.5;

/**
 * A terminal's two sidings past each end of its platform where track runs on (T-1281), between the
 * running tracks and drawn like the turn-back crossovers. `ways`: the track leaving each station
 * along each of its edges, from the node at the platform's middle.
 */
function terminalSidings(b: StrokeBuilder, s: ClockState, ways: (node: number) => [number, Float32Array][]) {
  for (const n of s.nodes) {
    if (!n.platform || !hasUpgrade(n.upgrades, UPGRADE.terminal)) continue;
    const ws = ways(n.id);
    if (ws.length < 2) continue;
    const perM = 1 / Math.cos(n.lat * Math.PI / 180);
    const flags = n.stationBuilt ? 0 : F_FADED;
    const d0 = n.platform / 2 * perM, d1 = d0 + SIDING_LEAD_M * perM, d2 = d0 + SIDING_M * perM;
    for (const [edge, p] of ws) {
      const pts: number[] = [], at: number[] = [];
      let d = 0;
      const cut = (k: number, t: number) => { for (let j = 0; j < 3; j++) pts.push(p[k * 3 + j] + (p[k * 3 + 3 + j] - p[k * 3 + j]) * t); };
      for (let k = 0; k + 1 < p.length / 3 && d < d2; k++) {
        const len = Math.hypot(p[k * 3 + 3] - p[k * 3], p[k * 3 + 4] - p[k * 3 + 1]);
        for (const c of [d0, d1, d2]) if (len > 0 && c >= d && c < d + len) { cut(k, (c - d) / len); at.push(c); }
        d += len;
        if (d > d0 && d < d2) { cut(k, 1); at.push(d); }
      }
      if (at.length < 2) continue;
      for (const side of [-1, 1]) {
        const lane = at.map((x) => Math.round(127.5 * (1 + side * (x >= d1 ? SIDING_LANE : 1 - (1 - SIDING_LANE) * (x - d0) / (d1 - d0)))));
        b.add(pts, TRACK, edge, flags | F_TURNBACK, false, 0, undefined, lane);
      }
    }
  }
}

/** The lane value (0-1) a fraction `f` of the way along an edge's drawn points. */
function laneAlong(dr: LanePoints, f: number): number {
  if (typeof dr.lane === "number") return dr.lane / 255;
  const p = dr.lpts, n = p.length / 3;
  let total = 0;
  for (let k = 1; k < n; k++) total += Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
  let at = 0;
  const want = f * total;
  for (let k = 1; k < n; k++) {
    const s = Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
    if (at + s >= want || k === n - 1) {
      const u = s > 0 ? Math.max(0, Math.min(1, (want - at) / s)) : 0;
      return (dr.lane[k - 1] + (dr.lane[k] - dr.lane[k - 1]) * u) / 255;
    }
    at += s;
  }
  return dr.lane[0] / 255;
}

/**
 * A track edge's piece of a build: its strokes, ramps and speeds. A new object whenever any of them
 * changes, so its identity says whether its segments are as in the build before (T-1028).
 */
interface EdgeEntry extends LanePoints {
  /** the points as given (the clock worker's own array, the same object while the edge is unchanged) */
  given: Float32Array;
  /** a copy of them, kept */
  pts: Float32Array;
  a: number;
  b: number;
  flags: number;
  lanes: number;
  strokes: Strokes;
  ramps: number[];
  /** the edge speeds given, and per segment of `strokes` its speed limit (`Strokes.speed`) */
  speedGiven: ArrayLike<number> | undefined;
  vmin: number;
  speed: Float32Array;
  /** per segment, `Strokes.trainSpeed` */
  train: Float32Array;
}
interface LineEntry {
  path: number[];
  stops: number[];
  running: boolean;
  strokes: Strokes;
  firstStroke: number[];
}
/** A line's `lineTrainSpeeds` and what they were worked out from. */
interface LineSpeeds {
  path: number[];
  lens: number[];
  ph: Float32Array | null;
  strokes: (Strokes | undefined)[];
  built: boolean[];
  out: (Float32Array | null)[] | null;
}
/** A line's train samples' strokes and lanes (`sampleStroke` less its base, `sampleLane`). */
interface LineSamples {
  c: LineEntry;
  count: number;
  step: number;
  lens: number[];
  rel: Int32Array;
  lane: Float32Array;
}

/**
 * What the last build made, per edge, line and station, so an edit rebuilds only the pieces whose
 * inputs changed (T-496), and the per-line work on them (T-1029). Every entry is checked against its
 * inputs, never trusted, so a reused piece is exactly what a full build would make; entries not seen
 * in a build are dropped. `last`: the build before, its pieces in stroke order, for `NetPatch`.
 */
export interface NetCache {
  edges: Map<number, EdgeEntry>;
  lines: Map<number, LineEntry>;
  stations: Map<number, { platform: number; lat: number; first: number; ways: number[]; ax: number; ay: number; shape: [number, number, number, number]; line: ArrayLike<number> }>;
  speeds: Map<number, LineSpeeds>;
  samples: Map<number, LineSamples>;
  serial: number;
  last: { serial: number; track: EdgeEntry[]; lines: LineEntry[]; trackCount: number; lineCount: number; pts: Map<number, Float32Array> } | null;
}

export const newNetCache = (): NetCache => ({ edges: new Map(), lines: new Map(), stations: new Map(), speeds: new Map(), samples: new Map(), serial: 0, last: null });

/**
 * A network posted as changes (`NetPatch`) made whole from the one before it, `prev`, in place: the
 * fields left out taken from it (the run profiles from `service` when a train count came between),
 * its edges' points from both. False, changing nothing that matters, when `prev` is not the build the
 * changes are to.
 */
export function mergeNet(prev: NetworkBuffers | null, n: NetworkBuffers, service?: { phases: Float32Array; meta: Float32Array } | null): boolean {
  const p = n.patch;
  if (!p) return true;
  if (!prev || prev.serial !== p.base || prev.track.count !== p.oldTrack || prev.lines.count !== p.oldLines) return false;
  const from: Record<string, unknown> = { ...prev, ...(service ?? {}) };
  for (const k of p.same) if (from[k] === undefined) return false;
  const pts = new Map<number, Float32Array>();
  for (const id of p.edgeIds) {
    const v = n.edgePts.get(id) ?? prev.edgePts.get(id);
    if (!v) return false;
    pts.set(id, v);
  }
  for (const k of p.same) (n as unknown as Record<string, unknown>)[k] = from[k];
  n.edgePts = pts;
  return true;
}

/** Runs (start, count, old start or -1) making `parts` (in order, `count` of each) from `old`. */
function strokeRuns<P extends object>(parts: P[], count: (p: P) => number, old: P[] | undefined): Int32Array {
  const at = new Map<P, number>();
  let o = 0;
  for (const p of old ?? []) at.set(p, o), (o += count(p));
  const runs: number[] = [];
  let n = 0;
  for (const p of parts) {
    const c = count(p);
    if (!c) continue;
    const from = at.get(p) ?? -1;
    const k = runs.length - 3;
    if (k >= 0 && (from < 0 ? runs[k + 2] < 0 : runs[k + 2] >= 0 && runs[k + 2] + runs[k + 1] === from)) runs[k + 1] += c;
    else runs.push(n, c, from);
    n += c;
  }
  return new Int32Array(runs);
}

/** The renderer's buffers from a state. With `cache` (the clock worker's), pieces whose inputs are
 * as in the last build are reused, and `patch` says how the strokes differ from it; the buffers are
 * the same either way. */
export function networkFromState(s: ClockState, cache: NetCache = newNetCache()): NetworkBuffers {
  const r = s.render;
  const edgePts = new Map<number, Float32Array>();
  const builtEdge = new Map<number, boolean>();
  const offEdge = new Set<number>(s.edges.filter((e) => e.off).map((e) => e.id));
  const hsrEdge = new Set<number>(s.edges.filter((e) => e.hsr).map((e) => e.id));
  /** edges whose polyline, ends and flags are as in the last build */
  const kept = new Set<number>();
  const edgesSeen: NetCache["edges"] = new Map();
  /** the edges in drawing order */
  const trackOrder: number[] = [];
  const marks: number[] = [];
  // double track narrows to single track where the two meet at a plain node (T-658)
  const endCount = new Map<number, number>(), singleAt = new Set<number>();
  for (const e of s.edges) {
    endCount.set(e.a, (endCount.get(e.a) ?? 0) + 1).set(e.b, (endCount.get(e.b) ?? 0) + 1);
    if (e.tracks === 1) singleAt.add(e.a).add(e.b);
  }
  const taperAt = (node: number) => singleAt.has(node) && endCount.get(node) === 2;
  /** per edge id: the points and lanes its track and lines are drawn with */
  const drawn = new Map<number, LanePoints>();
  // blueprint track after built, so drawn over it and picked first where they overlap, as the
  // station tool prefers it (T-695)
  const order = s.edges.map((_, i) => i).sort((i, j) => Number(!s.edges[i].built) - Number(!s.edges[j].built) || i - j);
  for (const i of order) {
    const e = s.edges[i];
    const given = r.edgeParts ? r.edgeParts[i] : r.edgePts!.subarray(r.edgeOff![i] * 3, r.edgeOff![i + 1] * 3);
    // as in one zero-filled array of every edge's speeds (`ClockRender.edgeSpeed`)
    const sg: ArrayLike<number> | undefined = r.edgeParts ? r.edgeSpeedParts?.[i] : r.edgeSpeed?.subarray(r.edgeOff![i], r.edgeOff![i + 1]);
    builtEdge.set(e.id, e.built);
    const flags = (e.built ? 0 : F_BLUEPRINT | (e.off ? F_OFF : 0)) | (e.hsr ? F_HSR : 0);
    const lanes = (e.tracks === 2 ? 1 : 0) | (e.tracks === 2 && taperAt(e.a) ? 2 : 0) | (e.tracks === 2 && taperAt(e.b) ? 4 : 0);
    let c = cache.edges.get(e.id);
    if (c && c.a === e.a && c.b === e.b && c.flags === flags && c.lanes === lanes && (c.given === given || sameFloats(c.pts, given))) {
      kept.add(e.id);
      if (c.given !== given) c = { ...c, given };
    } else {
      const own = given.slice();
      const lp = lanePoints(own, lanes);
      const b = new StrokeBuilder();
      b.add(lp.lpts, TRACK, e.id, flags, false, 0, undefined, lp.lane);
      const ramps: number[] = [];
      rampMarks(own, e.built ? 0 : 1, ramps);
      const strokes = b.build();
      c = { given, pts: own, a: e.a, b: e.b, flags, lanes, ...lp, strokes, ramps, speedGiven: undefined, vmin: NaN, speed: new Float32Array(0), train: new Float32Array(strokes.count).fill(-1) };
    }
    if (c.speedGiven !== sg || (!sg && c.vmin !== e.vminKmh) || c.speed.length !== c.strokes.count) {
      const n = c.strokes.count, speed = new Float32Array(n), src = c.src;
      for (let j = 0; j < n; j++) {
        const k = src ? src[j] : j;
        speed[j] = sg ? (sg[k] ?? 0) : e.vminKmh;
      }
      c = sameFloats(speed, c.speed) ? { ...c, speedGiven: sg, vmin: e.vminKmh } : { ...c, speedGiven: sg, vmin: e.vminKmh, speed };
    }
    edgesSeen.set(e.id, c);
    edgePts.set(e.id, c.pts);
    drawn.set(e.id, c);
    trackOrder.push(e.id);
    for (const v of c.ramps) marks.push(v);
  }
  const maxLine = s.lines.reduce((m, l) => Math.max(m, l.id), -1);
  const lineColours = new Uint8Array(Math.max(1, maxLine + 1) * 4);
  const lineCars = new Float32Array(r.lineTable.length / 4).fill(8);
  for (const l of s.lines) lineCars[l.id] = Math.max(1, l.cars);
  const lineParts: LineEntry[] = [];
  let lineCount = 0;
  const linesSeen: NetCache["lines"] = new Map();
  const samplesSeen: NetCache["samples"] = new Map();
  const lineIds: string[] = [];
  const onEdge = linesOnEdges(s);
  const edgeLen = new Map(s.edges.map((e) => [e.id, e.lengthM]));
  const edgeEnds = new Map(s.edges.map((e) => [e.id, e]));
  const nSamples = r.samples.length / 2;
  const sampleStroke = new Int32Array(nSamples).fill(-1);
  const sampleLane = new Float32Array(nSamples).fill(1);
  const sampleStretch = new Float32Array(Math.max(1, nSamples)).fill(-1);
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
        const dr = drawn.get(e);
        const ends = edgeEnds.get(e);
        if (dr) {
          const flags = (l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0))) | (hsrEdge.has(e) ? F_HSR : 0);
          const stretchOf = Math.min(stretch, Math.max(0, l.stops.length - 2));
          if (dr.lpts.length >= 6) firstStroke[k] = lines.count;
          d = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, d, stretchOf, dr.lane);
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
    lineParts.push(c);
    const base = lineCount;
    lineCount += c.strokes.count;
    // each train sample's stroke segment (run 0 resampled every `step` m) and lane, per line (T-1029)
    const first = r.lineTable[l.id * 4], count = r.lineTable[l.id * 4 + 1], step = r.lineTable[l.id * 4 + 3];
    const lens = l.path.map((p) => edgeLen.get(p >> 1) ?? 0);
    let sm = cache.samples.get(l.id);
    if (!sm || sm.c !== c || sm.count !== count || sm.step !== step || !sameArray(sm.lens, lens)) {
      const firstStroke = c.firstStroke;
      const rel = new Int32Array(count), lane = new Float32Array(count).fill(1);
      let j = 0, end = lens[0] ?? 0;
      for (let k = 0; k < count; k++) {
        const at = k * step;
        while (j < l.path.length - 1 && at > end) end += lens[++j];
        rel[k] = firstStroke[j] ?? -1;
        // how far apart the tracks are there (T-658), along the edge's drawn points when it tapers
        const p = l.path[j], dr = drawn.get(p >> 1), len = lens[j] ?? 0;
        if (dr) {
          const along = len > 0 ? Math.max(0, Math.min(1, 1 - (end - at) / len)) : 0;
          lane[k] = laneAlong(dr, (p & 1) === 1 ? 1 - along : along);
        }
      }
      sm = { c, count, step, lens, rel, lane };
    }
    samplesSeen.set(l.id, sm);
    const stretchOf = c.strokes.stretch!;
    for (let k = 0; k < count; k++) {
      const f = sm.rel[k];
      sampleStroke[first + k] = f < 0 ? -1 : base + f;
      if (f >= 0) sampleStretch[first + k] = stretchOf[f];
    }
    sampleLane.set(sm.lane, first);
  }
  cache.lines = linesSeen;
  cache.samples = samplesSeen;
  // the fastest train over each built segment (T-1271), from each line's speeds, kept per line
  const speedsSeen: NetCache["speeds"] = new Map();
  const trainOf = new Map<number, Float32Array>();
  for (const l of s.lines) {
    const lens = l.path.map((p) => edgeLen.get(p >> 1) ?? 0);
    const ph = l.ok && l.running && Math.max(...l.trains) > 0 ? speedPhases(l.id, r) : null;
    const strokes = l.path.map((p) => edgesSeen.get(p >> 1)?.strokes), built = l.path.map((p) => builtEdge.get(p >> 1) === true);
    let sp = cache.speeds.get(l.id);
    if (!sp || (sp.ph === null) !== (ph === null) || (ph && !sameFloats(sp.ph!, ph)) || !sameArray(sp.path, l.path) || !sameArray(sp.lens, lens) || strokes.some((x, k) => x !== sp!.strokes[k]) || built.some((x, k) => x !== sp!.built[k]))
      sp = { path: l.path.slice(), lens, ph, strokes, built, out: ph ? lineTrainSpeeds(l, r, (e) => edgesSeen.get(e)?.strokes, lens, (e) => builtEdge.get(e) === true) : null };
    speedsSeen.set(l.id, sp);
    sp.out?.forEach((v, k) => {
      if (!v) return;
      const e = l.path[k] >> 1;
      let t = trainOf.get(e);
      if (!t) trainOf.set(e, (t = new Float32Array(v.length).fill(-1)));
      for (let j = 0; j < v.length; j++) if (v[j] > t[j]) t[j] = v[j];
    });
  }
  cache.speeds = speedsSeen;
  const trackParts: EdgeEntry[] = [];
  let trackCount = 0;
  for (const id of trackOrder) {
    let c = edgesSeen.get(id)!;
    const t = trainOf.get(id);
    if (t ? !sameFloats(t, c.train) : c.train.some((v) => v !== -1)) edgesSeen.set(id, (c = { ...c, train: t ?? new Float32Array(c.strokes.count).fill(-1) }));
    trackParts.push(c);
    trackCount += c.strokes.count;
  }
  cache.edges = edgesSeen;
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
  /** the track leaving `node` along edge `id`, starting at the node */
  const way = (node: number, id: number) => {
    const p = edgePts.get(id), e = edgeEnds.get(id);
    return !p || !e ? new Float32Array(0) : e.a === node ? p : reversed(p);
  };
  const turnbacks = turnbackStrokes(s, linesSeen, endCount, (b) => terminalSidings(b, s, (n) => (waysAt.get(n) ?? []).map((id) => [id, way(n, id)])));
  // The station dot sits on the track: its node position, from the local polyline of an edge
  // that ends there (exact), else converted from lng/lat by the label code.
  const nodeLocal = new Map<number, [number, number]>();
  for (const e of s.edges) {
    const p = edgePts.get(e.id)!, b = p.length / 3 - 1;
    nodeLocal.set(e.a, [p[0], p[1]]);
    nodeLocal.set(e.b, [p[b * 3], p[b * 3 + 1]]);
  }
  const stationsSeen: NetCache["stations"] = new Map();
  st.forEach((n, i) => {
    const p = nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat);
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
    stations.set([p[0], p[1], ((served.get(n.id) ?? 0) > 1 ? 1 : 0) | (n.stationBuilt ? 0 : 2) | (lined.has(n.id) ? 4 : 0) | (n.stationOff ? 8 : 0) | (edgeEnds.get(platformEdge.get(n.id) ?? -1)?.tracks === 2 ? 16 : 0) | (hasUpgrade(n.upgrades, UPGRADE.waitingRoom) ? ST_ROOM : 0) | (hasUpgrade(n.upgrades, UPGRADE.parkRide) ? ST_PARK : 0) | (hasUpgrade(n.upgrades, UPGRADE.terminal) ? ST_TERMINAL : 0)], i * 3);
  });
  cache.stations = stationsSeen;
  annexSides(st, stations, stationLine, stationShape, edgePts, waysAt);
  // the player's links as bridges between the two platforms' nearest points (T-217)
  const stationIndex = new Map(st.map((n, i) => [n.id, i]));
  const shapeAt = (i: number) => platformShapeAt({ stationShape, stationAxis, stationLine }, i);
  const linkIdx: number[] = [], linkEnds: number[] = [];
  for (let k = 0; k + 1 < (s.links?.length ?? 0); k += 2) {
    const i = stationIndex.get(s.links[k]), j = stationIndex.get(s.links[k + 1]);
    if (i === undefined || j === undefined) continue;
    linkIdx.push(i, j);
    linkEnds.push(...bridgeEnds(stations[i * 3], stations[i * 3 + 1], shapeAt(i), stations[j * 3], stations[j * 3 + 1], shapeAt(j)));
  }
  const jn = s.nodes.filter((n) => n.ports >= 3 && n.platform === 0);
  const junctions = new Float32Array(jn.length * 2);
  jn.forEach((n, i) => junctions.set(nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat), i * 2));
  // the marks (T-629): a junction at ground level (a flyover is not one) is built once three of its
  // tracks are; a crossing once both of its are
  jn.forEach((n, i) => {
    if (!n.flying) marks.push(junctions[i * 2], junctions[i * 2 + 1], M_JUNCTION, 0, n.builtPorts >= 3 ? 0 : 1, n.level);
  });
  const cx = s.crossings;
  if (cx) for (let k = 0; k + 4 < cx.length; k += 5) marks.push(cx[k], cx[k + 1], cx[k + 2] === 1 ? M_BAD : M_CROSSING, 0, cx[k + 4] === 1 ? 0 : 1, cx[k + 3]);
  const speed = new Float32Array(trackCount), trainSpeed = new Float32Array(trackCount);
  for (let i = 0, at = 0; i < trackParts.length; at += trackParts[i++].strokes.count) {
    speed.set(trackParts[i].speed, at);
    trainSpeed.set(trackParts[i].train, at);
  }
  const serial = ++cache.serial, last = cache.last;
  const patch: NetPatch | undefined = last ? {
    base: last.serial,
    track: strokeRuns(trackParts, (p) => p.strokes.count, last.track),
    lines: strokeRuns(lineParts, (p) => p.strokes.count, last.lines),
    oldTrack: last.trackCount, oldLines: last.lineCount, same: [], edgeIds: Uint32Array.from(trackOrder),
  } : undefined;
  cache.last = { serial, track: trackParts, lines: lineParts, trackCount, lineCount, pts: edgePts };
  return {
    track: { ...concatStrokes(trackParts.map((p) => p.strokes), false), speed, trainSpeed },
    lines: concatStrokes(lineParts.map((p) => p.strokes), true),
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
    sampleLane,
    sampleStretch,
    turnbacks,
    edgePts,
    anchor: anchor(),
    serial,
    ...(patch ? { patch } : {}),
  };
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

const stationIndexes = new WeakMap<NetworkBuffers, Map<string, number>>();
/** A station id's place in the network's station buffers (`stationIds`), -1 for none. */
export function stationIndexOf(n: NetworkBuffers, id: string): number {
  let m = stationIndexes.get(n);
  if (!m) stationIndexes.set(n, (m = new Map(n.stationIds.map((s, i) => [s, i]))));
  return m.get(id) ?? -1;
}

/** Station `i`'s platform from the buffers: its middle, axis and bend, with its track's own line
 * (`line`) where the buffers have one. */
export function platformShapeAt(n: Pick<NetworkBuffers, "stationShape" | "stationAxis" | "stationLine">, i: number, line = true): PlatformShape {
  const s = n.stationShape, P2 = 2 * PLATFORM_POINTS, L = n.stationLine;
  return { mx: s?.[i * 4] ?? 0, my: s?.[i * 4 + 1] ?? 0, ax: n.stationAxis?.[i * 2] ?? 0, ay: n.stationAxis?.[i * 2 + 1] ?? 0, phi: s?.[i * 4 + 2] ?? 0, pts: line && L && L.length >= (i + 1) * P2 ? L.subarray(i * P2, (i + 1) * P2) : undefined };
}

/** The sample segment of a line (two or more samples) that offset `d` along run 0 is on: its first
 * sample's index, and how far along the segment `d` is, 0-1. */
function sampleSegment(n: NetworkBuffers, line: number, d: number): [number, number] {
  const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], len = n.lineTable[line * 4 + 2], step = n.lineTable[line * 4 + 3];
  const dd = Math.max(0, Math.min(len, d));
  const i0 = Math.min(Math.floor(dd / step), count - 2);
  const segLen = Math.max(1e-3, Math.min(step, len - i0 * step));
  return [first + i0, Math.max(0, Math.min(1, (dd - i0 * step) / segLen))];
}

/** Position (local units) at offset `d` along run 0 of a line. */
export function lineAt(n: NetworkBuffers, line: number, d: number): [number, number] {
  const first = n.lineTable[line * 4];
  if (n.lineTable[line * 4 + 1] < 2) return [n.samples[first * 2], n.samples[first * 2 + 1]];
  const [si, f] = sampleSegment(n, line, d);
  return [n.samples[si * 2] + (n.samples[si * 2 + 2] - n.samples[si * 2]) * f, n.samples[si * 2 + 1] + (n.samples[si * 2 + 3] - n.samples[si * 2 + 1]) * f];
}

/** How far apart the two tracks are at offset `d` along run 0 of a line, 0-1 (`sampleLane`). */
function laneAt(n: NetworkBuffers, line: number, d: number): number {
  if (!n.sampleLane) return 1;
  const first = n.lineTable[line * 4];
  if (n.lineTable[line * 4 + 1] < 2) return n.sampleLane[first] ?? 1;
  const [si, f] = sampleSegment(n, line, d);
  return n.sampleLane[si] + (n.sampleLane[si + 1] - n.sampleLane[si]) * f;
}

/** A trip's state at game time `t` (`dep` absolute): null if it is not on the track. `dx, dy`:
 * run 0's direction there. `shiftM`: metres to the right of its direction of travel on double
 * track, where the renderer draws it on its own track (T-658, `NetworkRenderer.laneShiftM`). */
export function tripAt(n: NetworkBuffers, profile: number, dep: number, t: number, end = -Infinity, shiftM = 0): { x: number; y: number; s: number; run: number; line: number; dx: number; dy: number; shiftM: number } | null {
  const tau = t - dep;
  const tripS = n.meta[profile * 4 + 2];
  // `end`: drawn until then, standing at the terminal for its next departure (T-206)
  if (tau < 0 || (tau > tripS && t > end) || !n.meta[profile * 4 + 1]) return null;
  const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
  const s = profileS(n, profile, tau);
  const len = n.meta[profile * 4 + 3];
  const d = run === 0 ? s : len - s;
  return { ...pointAt(n, line, d, run, shiftM), s, run, line, shiftM };
}

/** Position and direction used by both train and car picking, `shiftM` to the right of travel. */
function pointAt(n: NetworkBuffers, line: number, d: number, run = 0, shiftM = 0) {
  const [x, y] = lineAt(n, line, d);
  const [bx, by] = lineAt(n, line, d - 5), [fx, fy] = lineAt(n, line, d + 5);
  const dx = fx - bx, dy = fy - by;
  const [ox, oy] = rightOf(n, line, d, y, dx, dy, run, shiftM);
  return { x: x + ox, y: y + oy, dx, dy };
}

/** The shift to the right of travel (local y points south, so right of (dx, dy) is (-dy, dx)). */
function rightOf(n: NetworkBuffers, line: number, d: number, y: number, dx: number, dy: number, run: number, shiftM: number): [number, number] {
  const len = Math.hypot(dx, dy);
  if (!shiftM || len < 1e-9) return [0, 0];
  const k = shiftM * laneAt(n, line, d) * kAtY(y, n.anchor ?? anchor()) * (run === 0 ? 1 : -1) / len;
  return [-dy * k, dx * k];
}

/** A close-up car's centre and bogies, matching the GPU's terminal clamp and 20 m pitch. */
export function trainCarAt(n: NetworkBuffers, at: NonNullable<ReturnType<typeof tripAt>>, car: number) {
  const cars = Math.max(1, n.lineCars?.[at.line] ?? 8);
  const len = n.lineTable[at.line * 4 + 2], total = Math.min(cars * 20, len);
  const d = at.run === 0 ? at.s : len - at.s;
  const centre = Math.max(total / 2, Math.min(len - total / 2, d));
  const carD = centre + (car + 0.5) * total / cars - total / 2;
  const half = Math.max(total / cars / 2 - 0.5, 1);
  const p = pointAt(n, at.line, carD, at.run, at.shiftM);
  const [cx, cy] = lineAt(n, at.line, carD), ox = p.x - cx, oy = p.y - cy;
  const [bx, by] = lineAt(n, at.line, carD - half), [fx, fy] = lineAt(n, at.line, carD + half);
  return { ...p, back: [bx + ox, by + oy] as [number, number], front: [fx + ox, fy + oy] as [number, number] };
}
