// What the renderer draws, as flat typed arrays in local units (map/geo.ts), built from the clock
// worker's state (T-025): track strokes per edge, line strokes along each line's path, station
// dots, and the keyframe tables trains are positioned from (notes/T-040.md "Keyframes").
//
// Trains: each trip is (profile, departure - epoch); profile p = (line * 2 + run) * SLOTS + slot has
// a phase table (t, s, v, a) and meta (first phase, count, trip seconds, path length m); run 0 of
// each line is resampled every 50 m into `samples`, with points between where its track bends (`bendAt`, T-4075);
// run 1's offset s is `length - s` on run 0.

import { rgb } from "../game/palette";
import { boatOf } from "../game/boats";
import { hasUpgrade, UPGRADE } from "../game/upgrades";
import type { PreviewAnswer } from "../workers/clockClient";
import type { ClockEdge, ClockState } from "../workers/protocol";
import { anchor, kAtY, toLocal, type Anchor } from "./geo";
import { M_BAD, M_CONVERGE, M_CROSSING, M_DIVERGE, M_JUNCTION, M_RAMPEND, M_REPEAT, MARK_STRIDE, rampMarks } from "./markData";
import { repeatedEdge } from "../game/repeatTrack";
import { newFacilityCache, placeFacilities, type FacilityCache } from "./facilityPlace";
import { fault } from "./selfTest";
import { buildingsDrawn, newEdgeBoxes, type BuildingDraw, type EdgeBoxes } from "./buildingNet";
import { BAND_M_RANGE, BAND_RANGE, BAND_STRIDE, bandKnots, bandWork, KNOT, lineBands, newBandState, profileInto, shiftKept, type BandProfile, type BandState } from "./lineBands";
export { bandWork };
import { BOX_TOUCH_M, DEMAND_MERGE_M, offsetsReach, polylineGap } from "../game/complexes";
import { acrossAt, endClear, patchSplits, singleSplitEnds, splitsWhole, splitValue, TURNBACK_M, type SplitEnd, type SplitState } from "./terminalBays";
import { bridgeEnds, PASS_LEAD_M, passBow, PLATFORM_HALF_M, PLATFORM_POINTS, platformPoints, platformShape, reversed, ST_BUFFER, ST_BUILDING, ST_PARK, ST_ONE, ST_ONE_MID, ST_ONE_NEG, ST_PASSING, ST_SINGLE, ST_ROOM, ST_TERMINAL, ST_TRAM, type PlatformShape } from "./stationGeometry";
import { planStrokes, strokesOfPieces, type StrokeKind, type StrokeMode, type StrokePiece } from "./strokePost";
import { oneWayTrackChevrons, oneWayTrackRuns, trainChevrons, type OneWayEdge } from "./oneWayMarks";
import { stationOutside } from "./stationSide";
import { META_PER_LINE, ROW_SLOT, profileLine, profileOf, profileRun } from "./profiles";

/** A bus line's `lineCars` (T-1464): one car of this share of a train car's 20 m (a bus is 12 m),
 * narrower than a train's (rendererShaders.ts); below 1, so it tells a bus from a train. */
export const BUS_CAR = 0.6;
/** T-2194: a ferry line's: below 0, a boat of this many train cars' 20 m, drawn as one pointed hull; T-3892: per
 * boat size, Small 25 m, Medium 35 m, Large 90 m (`ferry::Boat::len_m`). The shader tells the size by it. */
export const BOAT_CAR = [-1.25, -1.75, -4.5] as const;

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
/** A line segment where its band's shift eases (T-1799, between two knots that differ): strokeLod
 * never merges it with segments either side, whose Traffic view shift may differ. */
export const F_EASE = 512;
/** A passing station's outer track (T-2233, `passingTracks`), in `turnbacks`: its lane value is how far
 * out it is, 0 on the running track to 255 on the outer track; F_PASS_LEFT on the left of its way. */
export const F_PASSING = 1024;
export const F_PASS_LEFT = 2048;
/** Tram track (T-2926), both kinds alike, and lines over it: thinner than rail, with two wires along it (T-2927). */
export const F_TRAM = 4096;
/** Single track splitting through a station (T-3212), and lines over it: its lanes widen the stroke
 * only as double track grows into two tracks, so zoomed out it stays single track's width. */
export const F_SPLIT = 8192;
/** T-3378: a busway (always with `F_TRAM`): a road's grey, one dashed centre line in place of the wires. */
export const F_BUSWAY = 16384;
/** A tram line's `lineCars` is its sections plus this, so the car shader tells it from a train of as many cars
 * (readers of the count use `carsOf`). */
export const TRAM_MARK = 0.5;
/** The cars a `lineCars` entry stands for: a bus (below 1) and a boat (below 0) as given, a tram's without its mark. */
export const carsOf = (raw: number) => (raw >= 1 ? Math.floor(raw) : raw);

/** T-3157: `Strokes.level` packs a segment's heights (m) at its start and end, each in 1/32 levels (8 m) from level -4
 * in 8 bits, as start * 256 + end, exact in a float; the stroke shader blends the level colours along it. */
export const levelCode = (z0: number, z1: number) => {
  const q = (z: number) => Math.max(0, Math.min(255, Math.round((z / 8 + 4) * 32)));
  return q(z0) * 256 + q(z1);
};
/** The whole level a packed segment starts at. */
export const levelAtStart = (code: number) => Math.round(Math.floor(code / 256) / 32 - 4);

export interface Strokes {
  /** per segment: x0, y0, x1, y1 */
  seg: Float32Array;
  /** per segment: line id (colour, selection), or TRACK */
  colour: Uint32Array;
  /** per segment: edge id */
  edge: Uint32Array;
  /** per segment: its height at its start and end (`levelCode`), so the height view blends along a ramp */
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
  /** a preview only (T-3137): the chevrons of the one-way track it would lay, `CHEVRON_STRIDE` numbers each */
  chevrons?: Float32Array;
  /** 2 per segment (T-658): how far apart its two tracks are drawn at its start and end, 0 (single
   * track, one track on the centre line) to 255 (double track); absent: all double */
  lane?: Uint8Array;
  /** line strokes, BAND_STRIDE per segment (T-1799, map/lineBands.ts): its band's shift left of its
   * direction at its start and end, as a share of BAND_RANGE line widths, `1 - its width share` at
   * both, and the metres from its track to its ribbon's middle at both as a share of BAND_M_RANGE
   * (T-2132), all Int16 normalised; absent or 0: one stroke on the centre line */
  band?: Int16Array;
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
   * on double track or single track that splits there (T-3212): a platform each side close in, T-658; ST_ROOM and ST_FLIP the complex's waiting
   * room drawn on this box, T-1528; ST_PARK park and ride here; ST_TERMINAL, T-1416; ST_BUFFER a
   * terminal at a track end, T-1527) */
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
  /** The complexes' car parks (T-1528, map/facilityPlace.ts), LOT_STRIDE numbers each. */
  lots?: Float32Array;
  /** The station buildings as drawn (T-2219, map/buildingNet.ts); their members have ST_BUILDING. */
  buildings?: BuildingDraw[];
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
  /** T-4075: per sample, where the track bends between it and the next (`bendAt`): its first point in `bends` times
   * 32 plus how many (0 none, else 1, 3, 7, 15 or 31, cutting the interval in equal parts); absent: none */
  sampleBend?: Float32Array;
  /** per bend point: its offset (x, y local units) from the straight between its interval's samples */
  bends?: Float32Array;
  /** Actual 20 m cars per train, indexed by line id; absent in old synthetic fixtures. */
  lineCars?: Float32Array;
  /** per sample of `samples`: a line stroke segment of the same stretch of path, -1 if none
   * (T-084: its stretch gives the train's load) */
  sampleStroke?: Int32Array;
  /** per sample of `samples`: how far apart its track's two tracks are drawn there (T-658), 0
   * single track to 1 double, as `Strokes.lane`; absent: 1 */
  sampleLane?: Float32Array;
  /** per sample of `samples`, 2 (T-1799): its line's band's shift there, line widths left of run 0's
   * direction, and metres from its track to its ribbon's middle (T-2132; `lineBands`); absent: 0 */
  sampleBand?: Float32Array;
  /** the edges that run side by side as one ribbon (T-1799 stage 3): `CORRIDOR_STRIDE` numbers each (`LineBands.corridor`) */
  corridor?: Float32Array;
  /** the crossovers where lines turn back, drawn close in only (T-658d, `turnbackStrokes`), and
   * passing stations' outer tracks (T-2233, `F_PASSING`) */
  turnbacks?: Strokes;
  /** T-2991: one-way loops' chevrons, `CHEVRON_STRIDE` numbers each (map/oneWayMarks.ts) */
  oneWay?: Float32Array;
  /** T-3094: chevrons along one-way single track, the same layout */
  oneWayTrack?: Float32Array;
  /** T-2233: per train sample, m past the nearest end of a platform with passing tracks where its line
   * stops on the outer track (below 0 along the platform), where its trains move out to it (`passBow`
   * over PASS_LEAD_M; past the end scaled by PASS_LEAD_M over that side's lead, T-2248); absent or
   * large: none near */
  samplePass?: Float32Array;
  /** T-2233, for the train guard: every passing station on each line's path, PASS_STRIDE numbers
   * each, by line then along run 0: line, offset along run 0 (m), half the platform (m), node, where
   * its trains are there (PASS_THROUGH, PASS_OUTER, PASS_INNER), and the outer track's lead before
   * and after the platform along run 0 (m, `passLead`) */
  passing?: Float32Array;
  /** per edge id: local polyline (x, y, z triples), for snapping and picking */
  edgePts: Map<number, Float32Array>;
  /** The draw anchor every position here is relative to (T-307, geo.ts): the clock worker's when
   * it built them. Absent (a synthetic network): the current one. */
  anchor?: Anchor;
  /** per train sample, its stop-to-stop stretch (`Strokes.stretch` of its stroke, -1 none) */
  sampleStretch?: Float32Array;
  /** T-1965, for the train guard (`trainGuard.ts`): per line id its first entry in `pathEntry` and
   * how many; per entry its edge and direction (edge << 1, low bit set from b to a, as `LineData.path`),
   * where it starts along run 0 and its length, m */
  pathTable?: Int32Array;
  pathEntry?: Int32Array;
  pathFrom?: Float32Array;
  pathLen?: Float32Array;
  /** T-1966, for the train guard: `ClockState.conflicts` */
  conflicts?: Float64Array;
  /** T-3202, for the train guard: `ClockState.singles` */
  singles?: Float64Array;
  /** T-3297: the side trains keep to, 1 right of their travel, -1 left (`ClockState.rightHand`); absent: 1 */
  keep?: number;
  /** T-1029: the clock worker's number for this build, and `patch` when it was posted as changes to
   * the build numbered `patch.base` (`NetPatch`) */
  serial?: number;
  patch?: NetPatch;
  /** T-4284, the page only (`mergeNet`): the edges whose points differ from the build numbered `base`'s, moved or added
   * (`set`) or taken out (`gone`), for the snap grid (map/snapIndex.ts); absent when not known */
  edgeStep?: { base: number; set: number[]; gone: number[] };
  /** the clock worker only (`networkFromState`'s `deferStrokes`): what `track` and `lines` are made from */
  strokePieces?: Record<StrokeKind, StrokePiece[]>;
  /** `?patch=check` only: the strokes whole, which the merged ones are compared with (`mergeNet`) */
  checkStrokes?: Record<StrokeKind, Strokes>;
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
  /** how `track` and `lines` were posted (T-1381, map/strokePost.ts): whole, as the last network's
   * (no arrays), or only the new segments, at `ranges` (start, count pairs; `mergeNet` makes them whole) */
  modes: Record<StrokeKind, StrokeMode>;
  ranges: Record<StrokeKind, Int32Array>;
  /** patch: from, to, count triples of the runs of the last network's segments that moved (T-1509) */
  moves?: Record<StrokeKind, Int32Array>;
  /** T-4280: the lines whose per-sample stretches (`sampleStretch`) differ from the last build's, and those whose lanes,
   * bands or passing values do (`sampleLane`, `sampleBand`, `samplePass`): their samples made again to other values, or
   * their passing stations elsewhere along them */
  stretches: number[];
  lanes: number[];
}

/** a run with no passing places (`linePassing`) */
const NO_PASS: number[] = [];

/** T-4280: whether two runs' samples give other per-sample stretches (1: `sampleStretch`) or lanes or bands (2:
 * `sampleLane`, `sampleBand`) */
function sampleValuesDiffer(a: LineSamples, b: LineSamples): number {
  if (a.count !== b.count) return 3;
  let out = 0;
  if (a.lane.length !== b.lane.length || a.band.length !== b.band.length) out = 2;
  for (let k = 0; !out && k < a.lane.length; k++) if (a.lane[k] !== b.lane[k]) out = 2;
  for (let k = 0; !out && k < a.band.length; k++) if (a.band[k] !== b.band[k]) out = 2;
  const sa = a.c.strokes.stretch, sb = b.c.strokes.stretch;
  for (let k = 0; k < a.count; k++) {
    const fa = a.rel[k], fb = b.rel[k];
    if ((fa < 0 ? -1 : sa?.[fa]) !== (fb < 0 ? -1 : sb?.[fb])) return out | 1;
  }
  return out;
}

/** T-2258: the shortest segment of built track and line strokes, plane m. The shader shifts a band
 * or a parted track sideways along its segment's own normal, and points arrive as float32 plane
 * metres (1-2 m apart far from the frame's origin, as all of Japan is), so a shorter segment, or
 * one of no length where a band's knot meets an edge's end, pointed anywhere and drew its round
 * cap off its band, a rounded prong. A chord this long strays under 0.1 m from a 100 m curve. */
export const MIN_SEG_M = 8;

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
  /** `Strokes.band`, only when built with `withStretch` (line strokes) */
  band = new Int16Array(0);
  /** T-4153: whether the last `add` ended on a point near the last one kept, so its last segment took its distance
   * along from `dist` (as float32) */
  popped = false;
  /** `minSeg`: points closer than this (plane m) to the last one kept are dropped (`MIN_SEG_M`). T-4187: `track`, a kept
   * piece of track, whose colour, edge and flags are one value each, and `oneColour`, a kept line's piece, whose colour is:
   * kept by the piece, not per segment (`TrackStrokes`, `LineStrokes`) */
  constructor(private withStretch = false, private minSeg = 0, private track = false, private oneColour = track) {}
  /** room for `n` segments in all */
  reserve(n: number) {
    this.grow(n);
  }
  private grow(need: number) {
    if (need <= this.cap) return;
    // a fresh builder takes what it is asked for: most hold one edge, and a 1,024 floor made every
    // edge's eight arrays ~50 kB (seconds of allocation on a 2,000-edge load, T-1156)
    const cap = this.cap ? Math.max(need, this.cap * 2) : need;
    const g = <T extends Float32Array | Uint32Array | Uint8Array | Int16Array>(a: T, k: number): T => {
      const b = new (a.constructor as any)(cap * k) as T;
      b.set(a);
      return b;
    };
    this.seg = g(this.seg, 4);
    if (!this.oneColour) this.colour = g(this.colour, 1);
    if (!this.track) this.edge = g(this.edge, 1);
    this.level = g(this.level, 1);
    if (!this.track) this.flags = g(this.flags, 1);
    this.dist = g(this.dist, 1);
    this.lane = g(this.lane, 2);
    if (this.withStretch) this.stretch = g(this.stretch, 1);
    if (this.withStretch) this.band = g(this.band, BAND_STRIDE);
    this.cap = cap;
  }
  /** pts: x, y, z triples; `stretch` as `Strokes.stretch`, for every segment added; `lane`: per point of
   * `pts`, 0 single track to 255 double (`Strokes.lane`), or one value for all; `band`: per point, its
   * shift (line widths), `1 - width share` and metres (`Strokes.band`); `from`, if given, gets each
   * added segment's first segment of `pts` (a to b order), which differ once `minSeg` drops points */
  add(pts: ArrayLike<number>, colour: number, edge: number, flags: number, reverse = false, dist0 = 0, stretch?: number, lane: number | ArrayLike<number> = 255, band?: ArrayLike<number>, ease?: Uint8Array, from?: number[]): number {
    const n = pts.length / 3;
    let d = dist0;
    this.popped = false;
    this.grow(this.count + Math.max(0, n - 1));
    if (stretch !== undefined && this.withStretch) this.stretch.fill(stretch, this.count, this.count + Math.max(0, n - 1));
    const first = this.count, ms = this.minSeg;
    // i: the point the next segment starts from; s: where the last one added started
    let i = reverse ? n - 1 : 0, s = i;
    for (let k = 1; k < n; k++) {
      const j = reverse ? n - 1 - k : k;
      const near = ms > 0 && Math.hypot(pts[j * 3] - pts[i * 3], pts[j * 3 + 1] - pts[i * 3 + 1]) < ms;
      if (near && k < n - 1) continue;
      // the last point near the one kept before it: the segment before ends on it instead
      if (near && this.count > first) {
        this.count--;
        from?.pop();
        d = this.dist[this.count];
        i = s;
        this.popped = true;
      }
      const x0 = pts[i * 3], y0 = pts[i * 3 + 1], x1 = pts[j * 3], y1 = pts[j * 3 + 1];
      const c = this.count++, lo = Math.min(i, j), hi = Math.max(i, j);
      from?.push(lo);
      let eased = false;
      for (let q = lo; ease && q < hi && !eased; q++) eased = ease[q] === 1;
      this.seg[c * 4] = x0;
      this.seg[c * 4 + 1] = y0;
      this.seg[c * 4 + 2] = x1;
      this.seg[c * 4 + 3] = y1;
      if (!this.oneColour) this.colour[c] = colour;
      if (!this.track) {
        this.edge[c] = edge;
        this.flags[c] = eased ? flags | F_EASE : flags;
      }
      this.level[c] = levelCode(pts[i * 3 + 2], pts[j * 3 + 2]);
      this.dist[c] = d;
      this.lane[c * 2] = typeof lane === "number" ? lane : lane[i];
      this.lane[c * 2 + 1] = typeof lane === "number" ? lane : lane[j];
      if (band && this.withStretch) {
        const S = 32767 / BAND_RANGE, o = c * BAND_STRIDE;
        this.band[o] = Math.round(band[i * 3] * S);
        this.band[o + 1] = Math.round(band[j * 3] * S);
        this.band[o + 2] = Math.round(band[i * 3 + 1] * 32767);
        this.band[o + 3] = Math.round(band[j * 3 + 1] * 32767);
        this.band[o + 4] = Math.round(Math.max(-1, Math.min(1, band[i * 3 + 2] / BAND_M_RANGE)) * 32767);
        this.band[o + 5] = Math.round(Math.max(-1, Math.min(1, band[j * 3 + 2] / BAND_M_RANGE)) * 32767);
      }
      d += Math.hypot(x1 - x0, y1 - y0);
      s = i;
      i = j;
    }
    return d;
  }
  /** T-4153: segments `a0` to `a1` of `src`, which one `add` made (`popped` as it left it), added again from
   * `dist0` on, on `stretch`, their distances along worked out as `add` works them out */
  copy(src: LineStrokes, a0: number, a1: number, dist0: number, popped: boolean, stretch: number): number {
    const c0 = this.count, m = a1 - a0;
    this.grow(c0 + m);
    this.seg.set(src.seg.subarray(a0 * 4, a1 * 4), c0 * 4);
    if (!this.oneColour) this.colour.set(src.colour!.subarray(a0, a1), c0);
    this.edge.set(src.edge.subarray(a0, a1), c0);
    this.level.set(src.level.subarray(a0, a1), c0);
    this.flags.set(src.flags.subarray(a0, a1), c0);
    this.lane.set(src.lane!.subarray(a0 * 2, a1 * 2), c0 * 2);
    if (this.withStretch) {
      this.stretch.fill(stretch, c0, c0 + m);
      this.band.set(src.band!.subarray(a0 * BAND_STRIDE, a1 * BAND_STRIDE), c0 * BAND_STRIDE);
    }
    const seg = this.seg;
    let d = dist0;
    for (let c = c0; c < c0 + m; c++) {
      if (popped && c === c0 + m - 1) d = Math.fround(d);
      this.dist[c] = d;
      d += Math.hypot(seg[c * 4 + 2] - seg[c * 4], seg[c * 4 + 3] - seg[c * 4 + 1]);
    }
    this.count = c0 + m;
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
      ...(this.withStretch ? { stretch: this.stretch.slice(0, n), band: this.band.slice(0, n * BAND_STRIDE) } : {}),
      count: n,
    };
  }
  /** T-4187: the strokes as the network's cache keeps them (a line's, `keepLine`, or a piece of track's, `keepTrack`):
   * every array a view of one buffer, since most pieces are a few segments and an array of its own costs more than
   * its numbers */
  keepLine(): LineStrokes {
    return this.pack() as LineStrokes;
  }
  /** (with room for a speed limit per segment beside them, `EdgeEntry.speed`) */
  keepTrack(): { strokes: TrackStrokes; speed: Float32Array } {
    const strokes = this.pack(), n = this.count;
    return { strokes, speed: new Float32Array(strokes.seg.buffer, 24 * n, n) };
  }
  private pack(): TrackStrokes | LineStrokes {
    const n = this.count, line = this.withStretch, track = this.track, colours = !this.oneColour;
    // (4-byte numbers first, then the bands' 2-byte, then the lanes' bytes, each view on its own size's boundary)
    const buf = new ArrayBuffer(4 * n * (track ? 7 : (line ? 9 : 8) + (colours ? 1 : 0)) + (line ? 2 * BAND_STRIDE * n : 0) + 2 * n);
    let at = 0;
    const take = <T extends Float32Array | Uint32Array | Int16Array | Uint8Array>(src: T, len: number): T => {
      const v = new (src.constructor as new (b: ArrayBuffer, o: number, l: number) => T)(buf, at, len);
      v.set(src.subarray(0, len) as never);
      at += len * src.BYTES_PER_ELEMENT;
      return v;
    };
    const seg = take(this.seg, n * 4);
    if (track) {
      const level = take(this.level, n), dist = take(this.dist, n);
      // (the room for its speeds, `keepTrack`)
      at += 4 * n;
      return { seg, level, dist, lane: take(this.lane, n * 2), count: n };
    }
    const colour = colours ? { colour: take(this.colour, n) } : {}, edge = take(this.edge, n), level = take(this.level, n), flags = take(this.flags, n), dist = take(this.dist, n);
    const stretch = line ? take(this.stretch, n) : undefined, band = line ? take(this.band, n * BAND_STRIDE) : undefined;
    return { seg, ...colour, edge, level, flags, dist, lane: take(this.lane, n * 2), ...(line ? { stretch, band } : {}), count: n };
  }
}

/** T-4187: a kept piece of track's strokes (`EdgeEntry.strokes`): no `colour`, `edge` or `flags`, which are TRACK, its
 * edge and its flags at every segment (`EdgeEntry.colour`, `edge`, `flags`), filled in as it is posted */
export type TrackStrokes = Omit<Strokes, "colour" | "edge" | "flags">;
/** T-4187: a kept line's strokes (`LineEntry.strokes`): no `colour` (all the line's id, `LineEntry.colour`) */
export type LineStrokes = Omit<Strokes, "colour"> & { colour?: Uint32Array };

/** T-4187: a copy of kept strokes, every array a view of one new buffer as `StrokeBuilder` keeps them */
function repacked<S extends TrackStrokes>(s: S): S {
  const out: Record<string, unknown> = { ...s };
  type View = Float32Array | Uint32Array | Int16Array | Uint8Array;
  const views = Object.entries(s).filter((e) => ArrayBuffer.isView(e[1])) as [string, View][];
  // (the widest numbers first, each view on its own size's boundary)
  views.sort((a, b) => b[1].BYTES_PER_ELEMENT - a[1].BYTES_PER_ELEMENT);
  const buf = new ArrayBuffer(views.reduce((n, [, v]) => n + v.byteLength, 0));
  let at = 0;
  for (const [k, v] of views) {
    const w = new (v.constructor as new (b: ArrayBuffer, o: number, l: number) => View)(buf, at, v.length);
    w.set(v as never);
    out[k] = w;
    at += v.byteLength;
  }
  return out as S;
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

/** The passages drawn without being built (T-1859): a complex's members whose boxes do not touch
 * are joined like the player's passages (T-217), by the shortest joins between platforms that
 * connect the complex, with boxes that touch and the player's own passages (`links`, station index
 * pairs) counting as joined already. Station index pairs. `cache` (T-2403): each complex's joins, kept
 * while its members, their station pieces (`NetCache.stations`, which hold their platform lines),
 * places and passages stay; a complex's joins depend on those alone, the player's passages joining
 * members of one complex (else everything is worked out again). */
function complexPassages(st: readonly { id: number; lat: number }[], stations: Float32Array, stationLine: Float32Array, complexOf: ArrayLike<number>, links: readonly number[], pieces?: readonly object[], cache?: NetCache["passages"], apartIds: readonly number[] = []): number[] {
  const groups = new Map<number, number[]>();
  st.forEach((n, i) => {
    const c = complexOf[n.id];
    if (c === undefined || c === 0xffffffff) return;
    const g = groups.get(c);
    if (g) g.push(i);
    else groups.set(c, [i]);
  });
  const of = (i: number) => complexOf[st[i].id];
  const linksIn = new Map<number, number[]>();
  let apart = false;
  for (let k = 0; k + 1 < links.length; k += 2) {
    const c = of(links[k]);
    if (c === undefined || c === 0xffffffff || c !== of(links[k + 1])) apart = true;
    else {
      const v = linksIn.get(c);
      if (v) v.push(links[k], links[k + 1]);
      else linksIn.set(c, [links[k], links[k + 1]]);
    }
  }
  // T-4260: pairs the player separated are no join of a complex's tree (one rejoined another way would draw a bridge
  // across them that no button can remove); per complex, so its cached joins stay for the same pairs
  const cutIn = new Map<number, number[]>(), cut = new Set<string>();
  for (let k = 0; k + 1 < apartIds.length; k += 2) {
    const x = apartIds[k], y = apartIds[k + 1], c = complexOf[x];
    if (c === undefined || c === 0xffffffff || c !== complexOf[y]) continue;
    cut.add(x < y ? `${x}|${y}` : `${y}|${x}`);
    const v = cutIn.get(c);
    if (v) v.push(x, y);
    else cutIn.set(c, [x, y]);
  }
  if (apart || !pieces || !cache) cache = undefined;
  const parent = Int32Array.from({ length: st.length }, (_, i) => i);
  const find = (i: number) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const union = (i: number, j: number) => { parent[find(i)] = find(j); };
  for (let k = 0; k + 1 < links.length; k += 2) union(links[k], links[k + 1]);
  const line = (i: number) => stationLine.subarray(i * 2 * PLATFORM_POINTS, (i + 1) * 2 * PLATFORM_POINTS);
  const out: number[] = [];
  const seen: NetCache["passages"] = new Map();
  for (const [c, m] of groups) {
    if (m.length < 2) continue;
    const ln = linksIn.get(c) ?? [], ct = cutIn.get(c) ?? [];
    let hit: PassageEntry | undefined;
    if (cache) {
      const e = cache.get(st[m[0]].id);
      if (e && e.ids.length === m.length && e.links.length === ln.length && e.cut.length === ct.length && e.cut.every((x, k) => x === ct[k]) && m.every((i, a) => e.ids[a] === st[i].id && e.pieces[a] === pieces![i] && e.xy[a * 2] === stations[i * 3] && e.xy[a * 2 + 1] === stations[i * 3 + 1]) && ln.every((i, k) => e.links[k] === st[i].id)) hit = e;
    }
    if (hit) {
      for (const a of hit.joins) out.push(m[a]);
      seen.set(st[m[0]].id, hit);
      continue;
    }
    const from = out.length;
    const reach = m.map((i) => offsetsReach(line(i)));
    const joins: [number, number, number][] = [];
    for (let a = 0; a < m.length; a++)
      for (let b = a + 1; b < m.length; b++) {
        const i = m[a], j = m[b], perM = 1 / Math.cos(st[i].lat * Math.PI / 180);
        const ix = stations[i * 3], iy = stations[i * 3 + 1], jx = stations[j * 3], jy = stations[j * 3 + 1];
        // every pair the complex rule links is nearer than this
        if (Math.hypot(ix - jx, iy - jy) > reach[a] + reach[b] + DEMAND_MERGE_M * perM) continue;
        const gap = polylineGap(ix, iy, line(i), jx, jy, line(j));
        if (gap <= BOX_TOUCH_M * perM) union(i, j);
        else if (!cut.has(st[i].id < st[j].id ? `${st[i].id}|${st[j].id}` : `${st[j].id}|${st[i].id}`)) joins.push([gap, i, j]);
      }
    joins.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
    for (const [, i, j] of joins) if (find(i) !== find(j)) { union(i, j); out.push(i, j); }
    if (cache) {
      const pos = new Map(m.map((i, a) => [i, a]));
      seen.set(st[m[0]].id, {
        ids: m.map((i) => st[i].id), pieces: m.map((i) => pieces![i]), xy: m.flatMap((i) => [stations[i * 3], stations[i * 3 + 1]]),
        links: ln.map((i) => st[i].id), cut: ct, joins: out.slice(from).map((i) => pos.get(i)!),
      });
    }
  }
  if (cache) {
    cache.clear();
    for (const [k, e] of seen) cache.set(k, e);
  }
  return out;
}
/** a complex's joins as last worked out (`complexPassages`): its members' ids, station pieces and
 * places, its passages' station ids in pairs, and the joins as member positions in pairs */
interface PassageEntry { ids: number[]; pieces: object[]; xy: number[]; links: number[]; cut: number[]; joins: number[] }

/** each line row's edges, once each, over its whole plan as it is drawn (T-3474) */
const edgesOfRow = new WeakMap<object, number[]>();
/** How many lines are on each edge: `was` itself when the same. */
function lineCounts(lines: readonly ClockState["lines"][number][], was?: Map<number, number>) {
  const on = new Map<number, number>();
  for (const l of lines) {
    if (!l.ok) continue;
    let v = edgesOfRow.get(l);
    if (!v) edgesOfRow.set(l, (v = [...new Set((l.planPath ?? l.path).map((p) => p >> 1))]));
    for (const e of v) on.set(e, (on.get(e) ?? 0) + 1);
  }
  if (was && was.size === on.size) {
    let same = true;
    for (const [e, v] of on) if (was.get(e) !== v) { same = false; break; }
    if (same) return was;
  }
  return on;
}

type LineRow = ClockState["lines"][number];
type NodeRow = ClockState["nodes"][number];
/** T-4153: the lines whose rows differ from `was`'s (rows are kept by identity while unchanged, workers/stateRows.ts),
 * each as its row in `was` and now, undefined where it came or went */
function lineChanges(was: readonly LineRow[], now: readonly LineRow[]): [LineRow | undefined, LineRow | undefined][] {
  const out: [LineRow | undefined, LineRow | undefined][] = [];
  if (was === now) return out;
  if (was.length === now.length) {
    let i = 0;
    for (; i < now.length; i++) {
      const w = was[i], n = now[i];
      if (w === n) continue;
      if (w.id !== n.id) break;
      out.push([w, n]);
    }
    if (i === now.length) return out;
    out.length = 0;
  }
  const by = new Map<number, LineRow>();
  for (const l of was) by.set(l.id, l);
  for (const l of now) {
    const w = by.get(l.id);
    if (w !== l) out.push([w, l]);
    by.delete(l.id);
  }
  for (const w of by.values()) out.push([w, undefined]);
  return out;
}
/** T-4153: the node ids whose rows differ from `was`'s, came or went (null with no `was`), and each id's index in `now`
 * (`wasIdx`: in `was`) */
function nodeChanges(was: readonly NodeRow[] | undefined, wasIdx: Int32Array | undefined, now: readonly NodeRow[], oldSlots: Map<number, number>): { changed: Set<number> | null; idx: Int32Array; slots: Map<number, number> } {
  let slots = oldSlots;
  const out = new Set<number>();
  if (was && wasIdx) {
    if (was === now) return { changed: out, idx: wasIdx, slots };
    if (was.length === now.length) {
      let i = 0;
      for (; i < now.length; i++) {
        const w = was[i], n = now[i];
        if (w === n) continue;
        if (w.id !== n.id) break;
        out.add(n.id);
      }
      if (i === now.length) return { changed: out, idx: wasIdx, slots };
      out.clear();
    }
  }
  // (node ids can be far apart: each has a slot of its own, `slots`, kept with the cache, and `idx` is by slot; slots of
  // nodes long gone are dropped by numbering the live ones again, `wasIdx` then read through the old numbers)
  const compact = !!wasIdx && slots.size > 2 * now.length + 64;
  if (compact) slots = new Map();
  const slotOf = (id: number) => {
    let k = slots.get(id);
    if (k === undefined) slots.set(id, (k = slots.size));
    return k;
  };
  const ks = new Int32Array(now.length);
  for (let i = 0; i < now.length; i++) ks[i] = slotOf(now[i].id);
  const idx = new Int32Array(slots.size).fill(-1);
  let matched = 0;
  for (let i = 0; i < now.length; i++) {
    const n = now[i], s = ks[i], ws = compact ? oldSlots.get(n.id) : s, j = wasIdx && ws !== undefined && ws < wasIdx.length ? wasIdx[ws] : -1, w = j >= 0 ? was![j] : undefined;
    idx[s] = i;
    if (w) matched++;
    if (w !== n) out.add(n.id);
  }
  if (!was || !wasIdx) return { changed: null, idx, slots };
  if (matched < was.length) for (const w of was) { const k = slots.get(w.id); if (k === undefined || idx[k] < 0) out.add(w.id); }
  return { changed: out, idx, slots };
}
/** each line row's stops, once each, its return run's own too */
const stopsOfRow = new WeakMap<object, number[]>();
const rowStops = (l: LineRow) => {
  let v = stopsOfRow.get(l);
  if (!v) stopsOfRow.set(l, (v = [...new Set([...l.stops, ...(l.backStops ?? [])])]));
  return v;
};
const rowEdgesOnce = (l: LineRow) => {
  let v = edgesOfRow.get(l);
  if (!v) edgesOfRow.set(l, (v = [...new Set((l.planPath ?? l.path).map((p) => p >> 1))]));
  return v;
};
/** how many lines are on each edge (`lineCounts`) and stop at each node, of the rows `rows` */
interface LineCounts { rows: readonly LineRow[]; counts: Map<number, number>; served: Map<number, number> }
/** T-4153: `LineCounts` of `lines`, patched from `kept` by the lines whose rows changed: `edges` and `stops` the ids whose
 * numbers changed (null: worked out whole); a map whose numbers are the same is `kept`'s own */
function countsOf(kept: LineCounts | null, lines: readonly LineRow[]): LineCounts & { edges: Set<number> | null; stops: Set<number> | null } {
  if (!kept) {
    const served = new Map<number, number>();
    for (const l of lines) for (const n of rowStops(l)) served.set(n, (served.get(n) ?? 0) + 1);
    return { rows: lines, counts: lineCounts(lines), served, edges: null, stops: null };
  }
  const dc = new Map<number, number>(), ds = new Map<number, number>();
  const bump = (m: Map<number, number>, k: number, d: number) => m.set(k, (m.get(k) ?? 0) + d);
  for (const [w, n] of lineChanges(kept.rows, lines)) {
    if (w?.ok) for (const e of rowEdgesOnce(w)) bump(dc, e, -1);
    if (n?.ok) for (const e of rowEdgesOnce(n)) bump(dc, e, 1);
    if (w) for (const x of rowStops(w)) bump(ds, x, -1);
    if (n) for (const x of rowStops(n)) bump(ds, x, 1);
  }
  const apply = (was: Map<number, number>, d: Map<number, number>, keys: Set<number>) => {
    for (const [k, v] of d) if (v) keys.add(k);
    if (!keys.size) return was;
    const m = new Map(was);
    for (const k of keys) {
      const v = (m.get(k) ?? 0) + d.get(k)!;
      if (v) m.set(k, v);
      else m.delete(k);
    }
    return m;
  };
  const edges = new Set<number>(), stops = new Set<number>();
  return { rows: lines, counts: apply(kept.counts, dc, edges), served: apply(kept.served, ds, stops), edges, stops };
}

/** `lane`: as `Strokes.lane`, 0 for single track (T-658), or per segment like `speed` (T-3134: the sim's
 * answer, `PreviewAnswer.lanes`); `hsr`: high-speed track's look (T-817) */
export function strokesFromPreview(pts: Float32Array, bad: boolean, speed?: Float32Array, placement = false, simMarks?: ArrayLike<number>, lane: number | ArrayLike<number> = 255, hsr = false, tram = false): Strokes {
  const b = new StrokeBuilder();
  const flags = F_BLUEPRINT | (bad ? F_BAD : 0) | (placement ? F_PLACEMENT : 0) | (hsr ? F_HSR : 0) | (tram ? F_TRAM : 0);
  // the points `from` to `to` with their lanes: each point takes its own segment's, the last its segment before
  const part = (from: number, to: number) => {
    const p = pts.subarray(from * 3, (to + 1) * 3);
    if (typeof lane === "number" || !lane.length) return b.add(p, TRACK, 0xffffffff, flags, false, 0, undefined, typeof lane === "number" ? lane : 255);
    const l = Uint8Array.from({ length: to - from + 1 }, (_, i) => lane[Math.min(from + i, to - 1, lane.length - 1)]);
    return b.add(p, TRACK, 0xffffffff, flags, false, 0, undefined, l);
  };
  // a negative speed is a step with no track (between quad's two pairs' own ends, T-423): not drawn
  let from = 0;
  if (speed) {
    for (let k = 0; k < speed.length; k++) {
      if (speed[k] >= 0) continue;
      part(from, k);
      from = k + 1;
    }
    if (from > 0) speed = speed.filter((v) => v >= 0);
  }
  part(from, pts.length / 3 - 1);
  const strokes = b.build();
  strokes.marks = previewMarks(pts, simMarks);
  if (speed) {
    if (speed.length !== strokes.count) throw new Error("Preview speed/segment length mismatch");
    strokes.speed = speed;
  }
  return strokes;
}

/** One stroke set from a preview of several alignments (`parts`), red when it cannot be built; `hsr`: drawn as high-speed track (T-1949) */
export function previewStrokes(a: PreviewAnswer, hsr = false): Strokes | null {
  const parts = a.parts ?? new Uint32Array([0, a.pts.length / 3]);
  const all: Strokes[] = [];
  for (let k = 0; k + 1 < parts.length; k++) {
    const pts = a.pts.subarray(parts[k] * 3, parts[k + 1] * 3);
    // the track model's crossings and junctions belong to the whole preview: given to the first part
    if (pts.length >= 6) all.push(strokesFromPreview(pts, !a.ok, undefined, false, all.length ? undefined : a.marks, a.partLanes?.[k] ?? 255, hsr));
  }
  if (!all.length) return null;
  if (all.length === 1) return all[0];
  const marks = all.flatMap((s) => (s.marks ? Array.from(s.marks) : []));
  const n = all.reduce((s, x) => s + x.count, 0);
  const out: Strokes = {
    seg: new Float32Array(n * 4), colour: new Uint32Array(n), edge: new Uint32Array(n), level: new Float32Array(n),
    flags: new Float32Array(n), dist: new Float32Array(n), count: n, marks: marks.length ? new Float32Array(marks) : undefined,
    lane: new Uint8Array(n * 2),
  };
  let o = 0;
  for (const s of all) {
    out.seg.set(s.seg.subarray(0, s.count * 4), o * 4);
    out.colour.set(s.colour.subarray(0, s.count), o);
    out.edge.set(s.edge.subarray(0, s.count), o);
    out.level.set(s.level.subarray(0, s.count), o);
    out.flags.set(s.flags.subarray(0, s.count), o);
    out.dist.set(s.dist.subarray(0, s.count), o);
    out.lane!.set(s.lane!.subarray(0, s.count * 2), o * 2);
    o += s.count;
  }
  return out;
}

/** A crossover's two diagonals as the placement preview (T-338), red when it cannot be built. */
export function strokesFromParts(parts: Float32Array[], bad: boolean, placement = true, lane = 255): Strokes {
  const b = new StrokeBuilder();
  for (const pts of parts) b.add(pts, TRACK, 0xffffffff, F_BLUEPRINT | (placement ? F_PLACEMENT : 0) | (bad ? F_BAD : 0), false, 0, undefined, lane);
  return b.build();
}

/** T-4148: edges the track model added, as the strokes the network will lay for them (blueprint, their kind's look, double
 * track's lanes), for drawing them ahead of it; `pts` are local units, x, y, z each */
export function strokesFromPieces(pieces: readonly { edge: number; pts: Float32Array; tracks: 1 | 2; kind: number }[]): Strokes {
  const b = new StrokeBuilder(false, MIN_SEG_M);
  for (const p of pieces) {
    const lp = lanePoints(p.pts.slice(), p.tracks === 2 && p.kind !== 4 ? 1 : 0);
    const flags = F_BLUEPRINT | (p.kind === 1 ? F_HSR : 0) | (p.kind >= 2 ? F_TRAM : 0) | (p.kind === 4 ? F_BUSWAY : 0);
    b.add(lp.lpts, TRACK, p.edge, flags, false, 0, undefined, lp.lane);
  }
  return b.build();
}

/** T-4214 stage E: pieces of the map the tools pick on drawn in place of the drawn edges they replace (map/clickMap.ts
 * `openUnder`), with the look network.ts lays an edge with (built or blueprint, switched off, high speed, tram, busway, its
 * tracks); `pts` are local units, x, y, z each. Its junction tapers and single-track splits wait for the network. */
export function strokesFromRows(pieces: readonly { pts: Float32Array; row: { id: number; built: boolean; off?: boolean; hsr?: boolean; kind?: number; tracks: 1 | 2 } }[]): Strokes {
  const b = new StrokeBuilder(false, MIN_SEG_M);
  for (const { pts, row: e } of pieces) {
    const flags = (e.built ? 0 : F_BLUEPRINT | (e.off ? F_OFF : 0)) | (e.hsr ? F_HSR : 0) | (e.kind ? F_TRAM : 0) | (e.kind === 4 ? F_BUSWAY : 0);
    const lp = lanePoints(pts.slice(), e.kind === 4 ? 0 : e.tracks === 2 ? 1 : 0);
    b.add(lp.lpts, TRACK, e.id, flags, false, 0, undefined, lp.lane);
  }
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
 * junctions the track model found (`sim`: x, y, then 1 for a junction or a crossing's packed level,
 * which carries its tram tracks, T-3535), all in the blueprint tier. */
export function previewMarks(pts: ArrayLike<number>, sim?: ArrayLike<number>): Float32Array | undefined {
  const m: number[] = [];
  rampMarks(pts, 1, m);
  if (sim) for (let k = 0; k + 2 < sim.length; k += 3) m.push(sim[k], sim[k + 1], sim[k + 2] === 1 ? M_JUNCTION : M_CROSSING, 0, 1, sim[k + 2] === 1 ? 0 : sim[k + 2]);
  return m.length ? new Float32Array(m) : undefined;
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

/** Two band profiles that draw alike (T-2238): equal bit for bit but for the knots' Traffic recipe
 * ids, which `bandPoints` reads only as none (-1) or the same as the next knot's. Those ids are
 * numbered across the whole network, so a ribbon gained on one line renumbers every line after it. */
const sameBand = (a: Float32Array, b: Float32Array) => {
  if (a.length !== b.length || a[0] !== b[0]) return false;
  const k0 = 2 + a[0];
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < k0; i++) if (x[i] !== y[i]) return false;
  for (let i = k0; i < x.length; i++) {
    if ((i - k0) % KNOT !== 3) {
      if (x[i] !== y[i]) return false;
    } else if ((a[i] < 0) !== (b[i] < 0) || (i + KNOT < a.length && (a[i] === a[i + KNOT]) !== (b[i] === b[i + KNOT]))) return false;
  }
  return true;
};

/** Half the gap between double track's two tracks when drawn apart, m (renderer.ts `LANE_M`). */
const HALF_GAP_M = 2;
/** How far from a junction a one-way leg's end may be drawn off its centre line, m (T-3120). */
const LEG_REACH_M = 300;
/** T-3191: the length over which a leg's end eases from its track back to its own centre line, m */
const EASE_M = 80;

/** T-3117, T-3120: how a one-way leg's end at a junction is drawn off its centre line (`legShifts`). */
interface LegShift {
  /** per end (a, b): +1 left of a to b in the local frame (right on the map: local y runs south), -1 right, 0 none */
  sign: [number, number];
  /** per end: the other track leaving the node its way on that side, each as its points from the
   * node out and whether it is double */
  near: [{ pts: number[]; double: boolean }[], { pts: number[]; double: boolean }[]];
  /** changes when any of it does, for the cache */
  key: string;
}

/** `LegShift.sign` for an end's join (`ClockEdge.join`, `PreviewAnswer.joins`): 1 the track left looking out from
 * the node along the piece, 2 the right, on the ground; local y runs south, so the ground's left is the local
 * frame's right (`LegShift.sign` -1). `end` 0 is a, whose outward way is a to b; b's runs b to a. */
const joinSign = (join: number, end: number) => (join === 1 ? -1 : join === 2 ? 1 : 0) * (end === 0 ? 1 : -1);

/**
 * Where single track meets double track at a junction, which way its end is drawn off its centre line, so it
 * carries on from its own track of the double. T-3137: the track its end joins (`ClockEdge.join`); for a one-way
 * end with no join (flyovers laid before T-3137) the T-3117 guess, away from the nearest other track leaving the
 * node its way. T-3120: with the tracks it is held off, so `shiftedEnds` keeps it as far from each as the double's
 * own tracks are drawn apart and no further off its line than that needs: those leaving its way on the other side,
 * and for a join to the far track the double it crosses. `parts`: each edge's points, in `edges` order.
/** a leg's end at a node: its edge's index, whether it starts at a, and the way it leaves (unit x, y) */
type LegEnd = { i: number; fromA: boolean; dx: number; dy: number };
const isLeg = (e: ClockEdge) => e.tracks === 1 && (!!e.oneWay || !!e.join);
/** the way an edge's points leave their a end (or b), unit x, y; null for none */
function legDir(pts: ArrayLike<number>, fromA: boolean) {
  const n = pts.length / 3;
  if (n < 2) return null;
  const x0 = fromA ? pts[0] : pts[(n - 1) * 3], y0 = fromA ? pts[1] : pts[(n - 1) * 3 + 1];
  const reach = 20 * kAtY(y0);
  let x1 = x0, y1 = y0;
  for (let k = 1; k < n; k++) {
    const j = fromA ? k : n - 1 - k;
    [x1, y1] = [pts[j * 3], pts[j * 3 + 1]];
    if (Math.hypot(x1 - x0, y1 - y0) >= reach) break;
  }
  const d = Math.hypot(x1 - x0, y1 - y0);
  return d > 1e-9 ? { dx: (x1 - x0) / d, dy: (y1 - y0) / d } : null;
}
/** an end's points from its node out, x, y pairs, as far as a leg's end can be drawn off */
function legFromNode(pts: ArrayLike<number>, fromA: boolean) {
  const n = pts.length / 3, xy: number[] = [];
  if (n < 1) return xy;
  const reach = 1.5 * LEG_REACH_M * kAtY(pts[1]);
  let run = 0;
  for (let k = 0; k < n; k++) {
    const j = fromA ? k : n - 1 - k;
    if (k > 0) run += Math.hypot(pts[j * 3] - xy[xy.length - 2], pts[j * 3 + 1] - xy[xy.length - 1]);
    xy.push(pts[j * 3], pts[j * 3 + 1]);
    if (run > reach) break;
  }
  return xy;
}
/** leg `e`'s (at index `i`) shift, from the ends at its nodes (`endsAt`); null for none */
function legShift(e: ClockEdge, i: number, edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[], endsAt: (node: number) => readonly LegEnd[]): LegShift | null {
  const shift: LegShift = { sign: [0, 0], near: [[], []], key: "" };
  [e.a, e.b].forEach((node, end) => {
    const ends = endsAt(node);
    const join = end === 0 ? e.join & 3 : (e.join >> 2) & 3;
    // T-3190: a join also counts at the loose end of double track, where just the two meet
    if (ends.length < (join ? 2 : 3) || !ends.some((x) => edges[x.i].tracks === 2)) return;
    const me = ends.find((x) => x.i === i && x.fromA === (end === 0));
    if (!me) return;
    if (!join && !e.oneWay) return;
    const cos = Math.cos(Math.PI / 3);
    const along = ends.filter((o) => o !== me && o.dx * me.dx + o.dy * me.dy > cos);
    const left = (o: { dx: number; dy: number }) => me.dx * o.dy - me.dy * o.dx > 0;
    const near = (o: { i: number; fromA: boolean }) => ({ pts: legFromNode(parts[o.i], o.fromA), double: edges[o.i].tracks === 2 });
    const nearest = (os: readonly LegEnd[]) => os.reduce((p, o) => (o.dx * me.dx + o.dy * me.dy > p.dx * me.dx + p.dy * me.dy ? o : p));
    if (join) {
      // `left` is the local frame's (joinSign)
      const toLeft = join === 2, double = (o: { i: number }) => edges[o.i].tracks === 2;
      const doubles = along.filter(double), main = doubles.length ? nearest(doubles) : null;
      // a join to the far track crosses the double it leaves
      const held = along.filter((o) => left(o) !== toLeft || (o === main && left(o) === toLeft));
      shift.sign[end] = joinSign(join, end);
      // leaving at an angle, the double track there
      shift.near[end] = (held.length ? held : ends.filter((o) => o !== me && double(o))).map(near);
      return;
    }
    // T-3209: a leg that carries straight on through the node (a double drawn onto its side split it) stays on its line
    if (ends.some((o) => o !== me && edges[o.i].tracks === 1 && o.dx * me.dx + o.dy * me.dy < -0.9)) return;
    if (!along.length) return;
    const best = nearest(along);
    const otherLeft = left(best);
    // leaving from b runs b to a, so its right is a to b's left
    shift.sign[end] = (otherLeft ? -1 : 1) * (end === 0 ? 1 : -1);
    shift.near[end] = along.filter((o) => left(o) === otherLeft).map(near);
  });
  if (!shift.sign[0] && !shift.sign[1]) return null;
  shift.key = shift.sign.join() + "|" + shift.near.map((n) => n.map((o) => `${o.double}:${o.pts.length}:${o.pts[2]}:${o.pts[3]}:${o.pts[o.pts.length - 2]}`).join(";")).join("|");
  return shift;
}
function legShifts(edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[]): Map<number, LegShift> {
  const out = new Map<number, LegShift>();
  if (!edges.some(isLeg)) return out;
  const at = new Map<number, LegEnd[]>();
  // (T-4153: only the nodes such a leg ends at are read, so only their ends are laid out)
  const wanted = new Set<number>();
  for (const e of edges) if (isLeg(e)) wanted.add(e.a).add(e.b);
  edges.forEach((e, i) => {
    if (!wanted.has(e.a) && !wanted.has(e.b)) return;
    for (const [node, fromA] of [[e.a, true], [e.b, false]] as const) {
      if (!wanted.has(node)) continue;
      const d = legDir(parts[i], fromA);
      if (!d) continue;
      const v = at.get(node) ?? [];
      v.push({ i, fromA, ...d });
      at.set(node, v);
    }
  });
  const endsAt = (n: number) => at.get(n) ?? [];
  edges.forEach((e, i) => {
    if (!isLeg(e)) return;
    const sh = legShift(e, i, edges, parts, endsAt);
    if (sh) out.set(e.id, sh);
  });
  return out;
}
/** T-4153: `prev`'s shifts made again for the legs with an end at a node whose edges changed (`touched`): a leg reads only
 * the ends at its two nodes. `idx`: each id's index in `edges`; `at`: each node's edge ids. */
function patchLegShifts(prev: Map<number, LegShift>, edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[], at: ReadonlyMap<number, { ids: readonly number[] }>, idx: Int32Array, touched: ReadonlySet<number>, removed: readonly ClockEdge[]): Map<number, LegShift> {
  let out = prev;
  const own = () => (out === prev ? (out = new Map(prev)) : out);
  for (const e of removed) if (out.has(e.id)) own().delete(e.id);
  const ends = new Map<number, LegEnd[]>();
  const indexOf = (id: number) => (id < idx.length ? idx[id] : -1);
  const endsAt = (n: number) => {
    let v = ends.get(n);
    if (v) return v;
    v = [];
    // (in edge order, as a whole pass lays them out; an edge from the node to itself both ends, a first)
    const ids = [...new Set(at.get(n)?.ids ?? [])].sort((x, y) => indexOf(x) - indexOf(y));
    for (const id of ids) {
      const i = indexOf(id), e = edges[i];
      if (!e) continue;
      for (const [node, fromA] of [[e.a, true], [e.b, false]] as const) {
        if (node !== n) continue;
        const d = legDir(parts[i], fromA);
        if (d) v.push({ i, fromA, ...d });
      }
    }
    ends.set(n, v);
    return v;
  };
  const redo = new Set<number>();
  for (const n of touched) for (const id of at.get(n)?.ids ?? []) redo.add(id);
  for (const id of redo) {
    const i = indexOf(id), e = i >= 0 ? edges[i] : undefined;
    const sh = e && isLeg(e) ? legShift(e, i, edges, parts, endsAt) : null;
    if (sh) own().set(id, sh);
    else if (out.has(id) && fault !== "legs") own().delete(id);
  }
  return out;
}

/** The distance from (x, y) to the polyline `xy` (x, y pairs). */
function distTo(xy: readonly number[], x: number, y: number): number {
  let best = Infinity;
  for (let k = 0; k + 3 < xy.length; k += 2) {
    const ax = xy[k], ay = xy[k + 1], bx = xy[k + 2], by = xy[k + 3];
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    best = Math.min(best, Math.hypot(ax + dx * t - x, ay + dy * t - y));
  }
  return best;
}

/** `pts` (x, y, z triples) with each end moved (as `legShifts`) off its centre line, by as much as
 * keeps it half the drawn gap of a double's two tracks from each single track beside it and the whole
 * of it from a double's centre line, so the drawn gap never dips below the gap at the node and is the
 * real one once that is wider (T-3120, Anita's "kiss"); points added every few metres there; and per
 * segment, the segment of `pts` it lies on. */
function shiftedEnds(pts: Float32Array, shift: LegShift): { pts: Float32Array; src: number[] } {
  const n = pts.length / 3;
  if (n < 2) return { pts, src: [] };
  const at = new Float64Array(n);
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
  const len = at[n - 1], kk = kAtY(pts[1]), h = HALF_GAP_M * kk, reach = LEG_REACH_M * kk, step = 5 * kk;
  // how far off its line the end `end` holds the point (x, y), `d` along from that end
  const need = (end: number, x: number, y: number, d: number) => {
    if (!shift.sign[end] || d > reach) return 0;
    // a piece shorter than the reach has its far end on the centre line too, or it jogs where the next piece meets it
    // (T-3379): the hold fades out over its far half
    const far = len < reach ? Math.max(0, Math.min(1, (d / len - 0.5) / 0.5)) : 0;
    const kept = 1 - far * far * (3 - 2 * far);
    let m = 0;
    for (const o of shift.near[end]) m = Math.max(m, o.double ? h - distTo(o.pts, x, y) : h - distTo(o.pts, x, y) / 2);
    // fades out short of the reach, should the tracks still be close there
    const t = Math.max(0, Math.min(1, (reach - d) / (0.2 * reach)));
    // T-3191: and never back to the centre line faster than this, so a leg leaving just before the double ends (or at
    // its end) carries on from its track and eases off its line, not bends in the moment the other track is gone
    const e = Math.max(0, Math.min(1, d / (EASE_M * kk)));
    return Math.max(Math.max(0, m) * t * t * (3 - 2 * t), h * (1 - e * e * (3 - 2 * e))) * kept;
  };
  const out: number[] = [], seg: number[] = [];
  const put = (k: number, f: number) => {
    seg.push(k);
    const x = pts[k * 3] + (pts[k * 3 + 3] - pts[k * 3]) * f, y = pts[k * 3 + 1] + (pts[k * 3 + 4] - pts[k * 3 + 1]) * f;
    const z = pts[k * 3 + 2] + (pts[k * 3 + 5] - pts[k * 3 + 2]) * f;
    const dx = pts[k * 3 + 3] - pts[k * 3], dy = pts[k * 3 + 4] - pts[k * 3 + 1], d = Math.hypot(dx, dy) || 1;
    const s = at[k] + (at[k + 1] - at[k]) * f;
    const o = shift.sign[0] * need(0, x, y, s) + shift.sign[1] * need(1, x, y, len - s);
    out.push(x - (dy / d) * o, y + (dx / d) * o, z);
  };
  for (let k = 0; k < n - 1; k++) {
    const seg = at[k + 1] - at[k];
    put(k, 0);
    // points every `step` where either end may hold it off
    if (seg > step && (at[k] < reach || at[k + 1] > len - reach)) {
      for (let d = step; d < seg - 1e-6; d += step) {
        const s0 = at[k] + d;
        if (s0 < reach || s0 > len - reach) put(k, d / seg);
      }
    }
  }
  put(n - 2, 1);
  // each segment drawn: the segment of \`pts\` it lies on
  return { pts: new Float32Array(out), src: seg.slice(0, -1) };
}

/**
 * T-3137: a single-track ghost (`pts`, x, y, z triples) with each end drawn on the track of double track it joins
 * (`PreviewAnswer.joins`, its first two), as built track's ends are (`shiftedEnds`); null when neither joins.
 * `lines`: the points of double track near the ghost (x, y, z triples), each held off at the end it passes
 * through. Per segment, the segment of `pts` it lies on.
 */
export function joinedGhost(pts: Float32Array, joins: ArrayLike<number>, lines: readonly Float32Array[]): { pts: Float32Array; src: number[] } | null {
  const n = pts.length / 3;
  const sign: [number, number] = [joinSign(joins[0] ?? 0, 0), joinSign(joins[1] ?? 0, 1)];
  if (n < 2 || (!sign[0] && !sign[1])) return null;
  const kk = kAtY(pts[1]), reach = 1.5 * LEG_REACH_M * kk;
  const near: LegShift["near"] = [[], []];
  for (const [end, k] of [[0, 0], [1, n - 1]] as const) {
    if (!sign[end]) continue;
    const x = pts[k * 3], y = pts[k * 3 + 1];
    for (const line of lines) {
      // its runs of points within reach of the end, the points either side included
      let run: number[] = [];
      const flush = () => {
        if (run.length >= 4 && distTo(run, x, y) < kk) near[end].push({ pts: run, double: true });
        run = [];
      };
      const m = line.length / 3, inside = (j: number) => Math.hypot(line[j * 3] - x, line[j * 3 + 1] - y) < reach;
      for (let j = 0; j < m; j++) {
        if (inside(j) || (j > 0 && inside(j - 1)) || (j + 1 < m && inside(j + 1))) run.push(line[j * 3], line[j * 3 + 1]);
        else flush();
      }
      flush();
    }
  }
  return shiftedEnds(pts, { sign, near, key: "" });
}

/** T-3137: a route ghost's chevrons (`s`) where every piece it lays would run one way, the same way
 * (`PreviewAnswer.joins` after the first two: 1 along the drawing, 2 against); else none. */
export function ghostChevrons(s: Strokes, joins: ArrayLike<number> | undefined): Float32Array | undefined {
  const ways = joins ? Array.from(joins).slice(2) : [];
  if (!ways.length || !ways[0] || ways.some((w) => w !== ways[0])) return undefined;
  const out: number[] = [];
  oneWayTrackChevrons(s, ways[0] as 1 | 2, [], out);
  return out.length ? new Float32Array(out) : undefined;
}

/** Where double track narrows to single, m (T-658). */
export const TAPER_M = 40;

/** An edge's points as drawn, with their `Strokes.lane` values; `src`, when points were added: per
 * segment, the edge's own segment it lies on. */
interface LanePoints { lpts: Float32Array; lane: number | Uint8Array; src: number[] | null }

/**
 * `lanes`: 1 double track, 2 and 4 narrowing to single track over TAPER_M at the a and b end (half
 * the edge at most each). A point is added where a taper starts, so it never spans a long leg.
 */
export function lanePoints(pts: Float32Array, lanes: number): LanePoints {
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

/**
 * T-3212: single track's points with its splits at stations (`singleSplitEnds`), the edge `lenM`
 * m long: a point at each split's platform end and every ~10 m along its ease, so the drawn ease
 * is smooth (the stroke shader takes lanes straight between points).
 */
function splitPoints(pts: Float32Array, split: readonly SplitEnd[], lenM: number): LanePoints {
  const n = pts.length / 3;
  if (n < 2 || !(lenM > 0)) return { lpts: pts, lane: 0, src: null };
  const at = new Float64Array(n);
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
  const len = at[n - 1];
  if (!(len > 0)) return { lpts: pts, lane: 0, src: null };
  // cuts in metres from a, as shares of the edge's length so they fall where the trains' samples do
  const cutM: number[] = [];
  for (const s of split) {
    if (!Number.isFinite(s.p)) continue;
    const steps = Math.max(1, Math.ceil(s.lead / 10));
    for (let i = 0; i <= steps; i++) {
      const x = s.p + (s.lead * i) / steps, m = s.a ? x : lenM - x;
      if (m > 0 && m < lenM) cutM.push(m);
    }
  }
  const value = (d: number) => Math.round(255 * splitValue(split, (d / len) * lenM, lenM));
  const cuts = cutM.map((m) => (m / lenM) * len).sort((x, y) => x - y);
  const out: number[] = [pts[0], pts[1], pts[2]], lane: number[] = [value(0)], src: number[] = [];
  let c = 0;
  for (let k = 0; k < n - 1; k++) {
    for (; c < cuts.length && cuts[c] < at[k + 1] - 1e-6; c++) {
      if (cuts[c] <= at[k] + 1e-6 || (c > 0 && cuts[c] - cuts[c - 1] < 1e-6)) continue;
      const f = (cuts[c] - at[k]) / (at[k + 1] - at[k]);
      for (let j = 0; j < 3; j++) out.push(pts[k * 3 + j] + (pts[k * 3 + 3 + j] - pts[k * 3 + j]) * f);
      lane.push(value(cuts[c]));
      src.push(k);
    }
    out.push(pts[k * 3 + 3], pts[k * 3 + 4], pts[k * 3 + 5]);
    lane.push(value(at[k + 1]));
    src.push(k);
  }
  if (lane.every((v) => v === lane[0])) return { lpts: pts, lane: lane[0], src: null };
  return { lpts: new Float32Array(out), lane: new Uint8Array(lane), src };
}


/**
 * The crossovers the game assumes where a line turns back on double track (T-658d; SPEC 6.2: a
 * dead-end terminus uses both its platform tracks, a line ending at a through station crosses back
 * behind the platform): a scissors just outside a dead-end terminus's platform, one diagonal just
 * outside a through station's platform on the side the line arrives from, from the arrival track
 * to the departure track. Drawn along the line's own strokes with `Strokes.lane` signed (`F_TURNBACK`:
 * 0 the left track of run 0's direction, 255 the right).
 */
function turnbackStrokes(s: ClockState, lines: NetCache["lines"], endCount: { get(node: number): number | undefined }, more: (push: (calls: StrokeAdd[]) => void) => void, nodes: ReadonlyMap<number, ClockState["nodes"][number]>, kept: NetCache["turnbacks"], seen: NetCache["turnbacks"], was: NetCache["turnStrokes"]): NonNullable<NetCache["turnStrokes"]> {
  // T-4153: the calls in order, each list the kept one where its calls are the same, so the strokes of the same lists are
  // the last build's
  const lists: StrokeAdd[][] = [];
  more((calls) => lists.push(calls));
  for (const l of s.lines) {
    const c = lines.get(l.id), st = c?.strokes;
    const last = l.stops.length - 1;
    if (!l.ok || !c || !st || !st.count || last < 1 || l.stops[0] === l.stops[last]) continue;
    // (T-4153: a line's turn-backs kept while its row, its piece and its end stations are)
    const n0 = nodes.get(l.stops[0]), n1 = nodes.get(l.stops[last]), d0 = endCount.get(l.stops[0]) === 1, d1 = endCount.get(l.stops[last]) === 1;
    const k = kept.get(l.id);
    if (k && k.row === l && k.c === c && k.n0 === n0 && k.n1 === n1 && k.d0 === d0 && k.d1 === d1 && k.rh === s.rightHand) {
      lists.push(k.calls);
      seen.set(l.id, k);
      continue;
    }
    const log = new StrokeLog();
    const entry = { row: l, c, n0, n1, d0, d1, rh: s.rightHand, calls: log.calls };
    seen.set(l.id, entry);
    turnsOf(l, c, st, log);
    if (k && sameCalls(k.calls, log.calls)) entry.calls = k.calls;
    lists.push(entry.calls);
  }
  if (was && was.lists.length === lists.length && was.lists.every((x, i) => x === lists[i])) return was;
  const b = new StrokeBuilder();
  for (const calls of lists) for (const a of calls) b.add(...a);
  return { lists, strokes: b.build() };

  function turnsOf(l: ClockState["lines"][number], c: LineEntry, st: LineEntry["strokes"], log: StrokeLog) {
    const last = l.stops.length - 1;
    const segLen = (i: number) => Math.hypot(st.seg[i * 4 + 2] - st.seg[i * 4], st.seg[i * 4 + 3] - st.seg[i * 4 + 1]);
    // T-3474: a line running part of its plan turns back at the ends of that part: strokes i0 to iEnd
    // of its plan's forward strokes (its return run's own follow them)
    let i0 = 0, iEnd = st.count - 1;
    if (l.planPath) {
      const runStart = l.pathStart ?? 0, runEnd = runStart + l.path.length;
      i0 = c.firstStroke.slice(runStart, runEnd).find((k) => k >= 0) ?? -1;
      iEnd = (c.firstStroke.slice(runEnd).find((k) => k >= 0) ?? c.forwardCount) - 1;
      if (i0 < 0 || iEnd < i0) return;
    }
    const S = l.planPath ? st.dist[i0] : 0, D = st.dist[iEnd] + segLen(iEnd);
    /** run 0's strokes from `s0` to `s1`, the lane going from `v0` to `v1` (-1 left, 1 right) */
    const along = (s0: number, s1: number, v0: number, v1: number, flags: number) => {
      s0 = Math.max(S, s0);
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
      if (pts.length >= 6) log.add(pts, TRACK, edge, flags | F_TURNBACK, false, 0, undefined, lane);
    };
    for (const atEnd of [false, true]) {
      const node = nodes.get(l.stops[atEnd ? last : 0]);
      const i = atEnd ? iEnd : i0;
      if (!node || !node.platform || (st.lane?.[i * 2 + (atEnd ? 1 : 0)] ?? 255) < 255) continue;
      const k = kAtY(st.seg[i * 4 + (atEnd ? 3 : 1)]);
      const dead = endCount.get(node.id) === 1;
      // the platform's open end at a dead end, else its end on the side the line arrives from
      const p = node.platform * k * (dead ? 1 : 0.5), T = TURNBACK_M * k;
      const s0 = atEnd ? D - p - T : S + p;
      const flags = node.stationBuilt ? 0 : F_FADED;
      // from the arrival track to the departure track: with trains keeping left, the other way across (T-3297)
      const hand = s.rightHand === false ? -1 : 1;
      along(s0, s0 + T, -hand, hand, flags);
      if (dead) along(s0, s0 + T, 1, -1, flags);
    }
  }
}
/** T-4153: two lists of stroke calls alike, number for number */
function sameCalls(a: readonly StrokeAdd[], b: readonly StrokeAdd[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as unknown[], y = b[i] as unknown[];
    if (x.length !== y.length) return false;
    for (let j = 0; j < x.length; j++) {
      const u = x[j], v = y[j];
      if (u === v) continue;
      if (!u || !v || typeof u !== "object" || typeof v !== "object" || !sameArray(u as ArrayLike<number>, v as ArrayLike<number>)) return false;
    }
  }
  return true;
}


/** T-3332: which side of a platform's axis (ax, ay), local units with y down, a one-track station's track
 * is on, from its trains' heading (rad, counter-clockwise from east) and the running side: 1 the + side
 * ((-ay, ax)), where trains along the axis keep right, -1 the other. */
export function oneSideAlong(heading: number, left: boolean, ax: number, ay: number): number {
  const along = Math.cos(heading) * ax - Math.sin(heading) * ay >= 0;
  return along !== left ? 1 : -1;
}

/** Length of a single-lead junction's crossover, m (T-3161). */
const SINGLE_LEAD_M = 40;

/**
 * T-3161: where single track that runs both ways joins one track of double track at a junction, the move
 * table also links it to the double's other track, crossing the one it joins (a single-lead junction,
 * track-and-service.md 6.2). Draw that link: a crossover along the double leaving the node on the other
 * side, from the joined track at the node to the other track `SINGLE_LEAD_M` out, as the turn-back
 * crossovers are drawn (`F_TURNBACK`, close in only). `way(node, edge)`: the edge's points from the node out.
 */
function singleLeadStrokes(b: StrokeBuilder, edges: readonly ClockEdge[], way: (node: number, edge: number) => Float32Array, byId: ReadonlyMap<number, ClockEdge>, at: (node: number) => readonly ClockEdge[]) {
  // the way an edge leaves `node`, unit x, y, from its first ~20 m
  const out = (node: number, e: number) => {
    const p = way(node, e), n = p.length / 3;
    if (n < 2) return null;
    const reach = 20 * kAtY(p[1]);
    let k = 1;
    while (k < n - 1 && Math.hypot(p[k * 3] - p[0], p[k * 3 + 1] - p[1]) < reach) k++;
    const dx = p[k * 3] - p[0], dy = p[k * 3 + 1] - p[1], d = Math.hypot(dx, dy);
    return d > 1e-9 ? { dx: dx / d, dy: dy / d, p } : null;
  };
  for (const e of edges) {
    if (e.tracks !== 1 || e.oneWay || !e.join || e.crossover) continue;
    [e.a, e.b].forEach((node, end) => {
      const join = end === 0 ? e.join & 3 : (e.join >> 2) & 3;
      if (!join) return;
      const me = out(node, e.id);
      if (!me) return;
      // the double leaving most nearly the opposite way: trains to or from it use the link
      let best: { id: number; dot: number; p: Float32Array } | null = null;
      for (const o of at(node)) {
        if (o.id === e.id || o.tracks !== 2 || o.crossover) continue;
        const d = out(node, o.id);
        const dot = d ? d.dx * me.dx + d.dy * me.dy : 1;
        if (d && dot < 0 && (!best || dot < best.dot)) best = { id: o.id, dot, p: d.p };
      }
      if (!best) return;
      const p = best.p, n = p.length / 3;
      let len = 0;
      for (let k = 1; k < n; k++) len += Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
      const T = Math.min(SINGLE_LEAD_M * kAtY(p[1]), 0.8 * len);
      // looking out along the double, away from the single, the joined track is on the other hand:
      // join 2 (right looking out along the single) is the double's left (lane -1)
      const v0 = join === 2 ? -1 : 1, v1 = -v0;
      const pts: number[] = [p[0], p[1], 0], lane: number[] = [Math.round(127.5 * (1 + v0))];
      let run = 0;
      for (let k = 1; k < n && run < T; k++) {
        const seg = Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
        const f = run + seg > T ? (T - run) / seg : 1;
        run += seg * f;
        pts.push(p[k * 3 - 3] + (p[k * 3] - p[k * 3 - 3]) * f, p[k * 3 - 2] + (p[k * 3 + 1] - p[k * 3 - 2]) * f, 0);
        lane.push(Math.round(127.5 * (1 + v0 + (v1 - v0) * run / T)));
      }
      const built = e.built && byId.get(best.id)?.built;
      if (pts.length >= 6) b.add(pts, TRACK, best.id, (built ? 0 : F_FADED) | F_TURNBACK, false, 0, undefined, lane);
    });
  }
}

/** The demand level whose run profiles give the Speed view's train speeds (T-1271): the levels
 * differ in how long trains stand, not in how fast they move between stops. */
export const SPEED_LEVEL = 1;

/** A run profile's speed at `x` m along its path, km/h: the moving phase holding `x` (phases are
 * t, s, v, a, notes/T-040.md "Keyframes"; a stop's standing phase spans no distance). */
export function speedAt(ph: Float32Array, first: number, count: number, x: number): number {
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

/** T-4187: `lineTrainSpeeds` in one array, entry k's from `idx[2k]`, `idx[2k + 1]` of them (-1: none), as an array an entry
 * cost more than its numbers */
interface TrainSpeeds { v: Float32Array; idx: Int32Array }
/** A running line's train speeds (T-1271), per entry of its path: at the middle of each stroke
 * segment of that edge, the faster of its two runs there, km/h; none where the edge is not built.
 * Null for a line with no trains. */
function lineTrainSpeeds(l: ClockState["lines"][number], r: ClockState["render"], strokesOf: (e: number) => TrackStrokes | undefined, lens: number[], built: (e: number) => boolean): TrainSpeeds | null {
  if (!l.ok || !l.running || Math.max(...l.trains) <= 0) return null;
  const runs = [0, 1].map((run) => {
    const m = profileOf(l.id, run, ROW_SLOT[SPEED_LEVEL]) * 4;
    return { first: r.meta[m], count: r.meta[m + 1], len: r.meta[m + 3] };
  });
  if (runs.some((p) => !(p.count > 0))) return null;
  // (T-4153: each run's phases found by stepping from the last one found, as `speedAt` finds them while they are in order)
  const at0 = phaseFinder(r.phases, runs[0].first, runs[0].count), at1 = phaseFinder(r.phases, runs[1].first, runs[1].count);
  const idx = new Int32Array(l.path.length * 2);
  let all = 0;
  const sts = l.path.map((p, k) => {
    const st = strokesOf(p >> 1), on = !!st && built(p >> 1);
    idx[k * 2] = all;
    idx[k * 2 + 1] = on ? st!.count : -1;
    if (on) all += st!.count;
    return on ? st : undefined;
  });
  const v = new Float32Array(all);
  let off = 0;
  l.path.forEach((p, k) => {
    const len = lens[k], st = sts[k];
    if (st) {
      const n = st.count, o = idx[k * 2];
      const segLen = (j: number) => Math.hypot(st.seg[j * 4 + 2] - st.seg[j * 4], st.seg[j * 4 + 3] - st.seg[j * 4 + 1]);
      const total = n ? st.dist[n - 1] + segLen(n - 1) : 0;
      for (let j = 0; j < n; j++) {
        const f = total > 0 ? (st.dist[j] + segLen(j) / 2) / total : 0.5;
        const x0 = off + ((p & 1) === 1 ? 1 - f : f) * len;
        v[o + j] = Math.max(at0(x0), at1(runs[1].len - x0));
      }
    }
    off += len;
  });
  return { v, idx };
}

/** T-4153: `speedAt` of one run's phases, the phase found by stepping from the last one found where their starts are in
 * order (the same phase the bisection finds), else by bisection */
function phaseFinder(ph: Float32Array, first: number, count: number): (x: number) => number {
  if (count <= 0) return () => 0;
  const last = first + count - 1;
  let sorted = true;
  for (let i = first + 1; i <= last && sorted; i++) if (!(ph[i * 4 + 1] >= ph[i * 4 - 3])) sorted = false;
  if (!sorted) return (x) => speedAt(ph, first, count, x);
  let i = first;
  return (x) => {
    while (i < last && ph[(i + 1) * 4 + 1] <= x) i++;
    while (i > first && !(ph[i * 4 + 1] <= x)) i--;
    const s = ph[i * 4 + 1], v = ph[i * 4 + 2], a = ph[i * 4 + 3];
    return Math.sqrt(Math.max(0, v * v + 2 * a * Math.max(0, x - s))) * 3.6;
  };
}

/** The phase tables `lineTrainSpeeds` reads for a line, one after the other with each run's length. */
/** T-4153: the profile meta `speedPhases` reads of a line (first phase, count, seconds and length of each run) */
function metaOf(id: number, r: ClockState["render"]): Float64Array {
  const out = new Float64Array(8);
  for (const run of [0, 1]) {
    const m = profileOf(id, run, ROW_SLOT[SPEED_LEVEL]) * 4;
    for (let j = 0; j < 4; j++) out[run * 4 + j] = r.meta[m + j];
  }
  return out;
}
const sameMetaOf = (kept: Float64Array, id: number, r: ClockState["render"]) => {
  for (const run of [0, 1]) {
    const m = profileOf(id, run, ROW_SLOT[SPEED_LEVEL]) * 4;
    for (let j = 0; j < 4; j++) if (!Object.is(kept[run * 4 + j], r.meta[m + j])) return false;
  }
  return true;
};
/** `speedPhases` of a line now is `kept`, bit for bit (`bits`: `r.phases`' own) */
function samePhasesOf(kept: Float32Array, id: number, r: ClockState["render"], bits: Uint32Array): boolean {
  const keptBits = new Uint32Array(kept.buffer, kept.byteOffset, kept.length);
  let n = 4;
  for (const run of [0, 1]) {
    const m = profileOf(id, run, ROW_SLOT[SPEED_LEVEL]) * 4;
    if (!Object.is(kept[run * 2], Math.fround(r.meta[m + 1])) || !Object.is(kept[run * 2 + 1], Math.fround(r.meta[m + 3]))) return false;
    const first = r.meta[m], count = r.meta[m + 1];
    if (!(count > 0)) continue;
    if (n + count * 4 > kept.length) return false;
    for (let j = 0, o = first * 4; j < count * 4; j++) if (keptBits[n + j] !== bits[o + j]) return false;
    n += count * 4;
  }
  return n === kept.length;
}
function speedPhases(id: number, r: ClockState["render"]): Float32Array {
  const parts: number[] = [];
  let n = 0;
  const at: [number, number][] = [];
  for (const run of [0, 1]) {
    const m = profileOf(id, run, ROW_SLOT[SPEED_LEVEL]) * 4;
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


/**
 * Passing tracks' outer pair (T-2233): at each station in `at`, each of its ways from the node, a
 * track either side along the platform that bows back onto the running track over PASS_LEAD_M past
 * its end. The renderer sets how far out the outer track is (`passingState`); lane values say how
 * far along the bow (`passBow`) a point is.
 */
function passingTracks(b: StrokeBuilder, at: readonly ClockState["nodes"][number][], ways: (node: number) => [number, Float32Array][], leadOf: (node: number, edge: number) => number) {
  const LEAD_CUTS = 8;
  for (const n of at) {
    const ws = ways(n.id);
    if (ws.length < 2) continue;
    const perM = 1 / Math.cos(n.lat * Math.PI / 180);
    const flags = F_PASSING | (n.stationBuilt ? 0 : F_FADED);
    const d0 = n.platform / 2 * perM;
    for (const [edge, p] of ws) {
      const d1 = d0 + leadOf(n.id, edge) * perM;
      const cuts = Array.from({ length: LEAD_CUTS + 1 }, (_, j) => d0 + (d1 - d0) * j / LEAD_CUTS);
      const pts: number[] = [p[0], p[1], p[2]], lane: number[] = [255];
      let d = 0;
      for (let k = 0; k + 1 < p.length / 3 && d < d1; k++) {
        const len = Math.hypot(p[k * 3 + 3] - p[k * 3], p[k * 3 + 4] - p[k * 3 + 1]);
        const end = Math.min(d + len, d1);
        for (const c of cuts) if (len > 0 && c > d && c < end) {
          const t = (c - d) / len;
          for (let j = 0; j < 3; j++) pts.push(p[k * 3 + j] + (p[k * 3 + 3 + j] - p[k * 3 + j]) * t);
          lane.push(Math.round(255 * passBow((c - d0) / (d1 - d0))));
        }
        const t = len > 0 ? (end - d) / len : 1;
        for (let j = 0; j < 3; j++) pts.push(p[k * 3 + j] + (p[k * 3 + 3 + j] - p[k * 3 + j]) * t);
        lane.push(Math.round(255 * passBow((end - d0) / (d1 - d0))));
        d += len;
      }
      if (pts.length < 6) continue;
      for (const left of [false, true]) b.add(pts, TRACK, edge, flags | (left ? F_PASS_LEFT : 0), false, 0, undefined, lane);
    }
  }
}

/** Where each line's trains are at passing stations (T-2233), from the stations in `at`: per line, for
 * every one on its path, `PASS_STRIDE` numbers (`NetworkBuffers.passing`); and per station where its
 * trains stand on the outer track: offset, half the platform, the leads before and after (`mine`,
 * for `NetworkBuffers.samplePass`). `inner`: the stations where the line stops on the running track
 * (`innerStops`). */
export const PASS_STRIDE = 7;
export const PASS_THROUGH = 0, PASS_OUTER = 1, PASS_INNER = 2;
function linePassing(l: ClockState["lines"][number], at: ReadonlyMap<number, ClockState["nodes"][number]>, edgeEnds: ReadonlyMap<number, { a: number; b: number }>, lens: readonly number[], inner: ReadonlySet<number> | undefined, leadOf: (node: number, edge: number) => number, out: number[]): number[] {
  const stops = new Set(l.stops.slice(1, -1));
  const mine: number[] = [];
  let d = 0;
  // (a run's ends are where its trains turn, on the running track)
  l.path.forEach((p, k) => {
    const e = edgeEnds.get(p >> 1);
    d += lens[k] ?? 0;
    if (!e || k + 1 >= l.path.length) return;
    const node = (p & 1) ? e.a : e.b, n = at.get(node);
    if (!n || !n.stationBuilt) return;
    const where = !stops.has(node) ? PASS_THROUGH : inner?.has(node) ? PASS_INNER : PASS_OUTER;
    const before = leadOf(node, p >> 1), after = leadOf(node, l.path[k + 1] >> 1);
    out.push(l.id, d, n.platform / 2, node, where, before, after);
    if (where === PASS_OUTER) mine.push(d, n.platform / 2, before, after);
  });
  return mine;
}

/** T-2248: how far past a passing station's platform end (`half`, m) its outer track joins the running
 * track along an edge of `lenM`: PASS_LEAD_M, or what the edge has room for past the platform, shared
 * with the far end's outer track where that is a passing station too (`otherHalf`, else below 0). */
export const PASS_LEAD_MIN_M = 10;
export function passLead(lenM: number, half: number, otherHalf: number): number {
  const room = otherHalf >= 0 ? (lenM - half - otherHalf) / 2 : lenM - half;
  return Math.max(PASS_LEAD_MIN_M, Math.min(PASS_LEAD_M, room));
}

/** each line row's nodes along its path, and the edges (`EdgeSide`) they were read from */
const nodeSeqOf = new WeakMap<object, { side: object; nodes: number[] }>();
/** T-2248: per line, the passing stations in `at` where its trains stop on the running track: beside a
 * slower line's train standing on the outer track (passing-tracks design, notes/T-2228.md 5). Of two
 * lines stopping at a station and sharing the track through it, the slower is the one that stops at
 * more of the stations on the stretch they run together; equal, both stand on the outer track. */
function innerStops(lines: ClockState["lines"], at: ReadonlyMap<number, unknown>, edgeEnds: ReadonlyMap<number, { a: number; b: number }>, side: object): Map<number, Set<number>> {
  const seq = new Map<number, number[]>(), stopsOf = new Map<number, Set<number>>(), by = new Map<number, number[]>();
  for (const l of lines) {
    if (!l.ok || l.bus || !l.path.length) continue;
    // (T-4153: kept with the row while the edges are)
    let kept = nodeSeqOf.get(l);
    if (kept?.side !== side) {
      const nodes: number[] = [];
      for (const [k, p] of l.path.entries()) {
        const e = edgeEnds.get(p >> 1);
        if (!e) break;
        if (k === 0) nodes.push((p & 1) ? e.b : e.a);
        nodes.push((p & 1) ? e.a : e.b);
      }
      nodeSeqOf.set(l, (kept = { side, nodes }));
    }
    const nodes = kept.nodes;
    seq.set(l.id, nodes);
    stopsOf.set(l.id, new Set(l.stops));
    for (const n of new Set(l.stops.slice(1, -1))) if (at.has(n)) by.set(n, [...(by.get(n) ?? []), l.id]);
  }
  const out = new Map<number, Set<number>>();
  for (const [st, ls] of by) {
    for (const a of ls) for (const b of ls) {
      if (a === b) continue;
      const A = seq.get(a)!, B = seq.get(b)!, i = A.indexOf(st), j = B.indexOf(st);
      if (i < 0 || j < 0) continue;
      const meets = (x: number | undefined, y: number | undefined) => x !== undefined && x === y;
      const dir = meets(A[i + 1], B[j + 1]) || meets(A[i - 1], B[j - 1]) ? 1 : meets(A[i + 1], B[j - 1]) || meets(A[i - 1], B[j + 1]) ? -1 : 0;
      if (!dir) continue;
      const sa = stopsOf.get(a)!, sb = stopsOf.get(b)!;
      let na = 0, nb = 0;
      for (const step of [1, -1]) for (let d = step === 1 ? 0 : -1; meets(A[i + d], B[j + d * dir]); d += step) {
        if (sa.has(A[i + d])) na++;
        if (sb.has(A[i + d])) nb++;
      }
      if (na < nb) {
        const v = out.get(a);
        if (v) v.add(st);
        else out.set(a, new Set([st]));
      }
    }
  }
  return out;
}

/** T-4153: `bandPoints`' working arrays, grown as needed; what it returns lies in them until its next call */
const bpScratch = { at: new Float64Array(64), pts: new Float32Array(192), lane: new Uint8Array(64), band: new Float32Array(192), ease: new Uint8Array(64), cuts: [] as number[], ms: [] as number[] };
/**
 * An edge's drawn points for a line on it (T-1799): with a point added at each knot of its band
 * (`lineBands`), and per point its shift and `1 - width share` (`StrokeBuilder.add`'s `band`). `b`:
 * the line's band profile, `k` the path entry, in its travel direction (`reverse`: b to a). (T-4153: views of
 * `bpScratch`, read before the next call.)
 */
function bandPoints(dr: LanePoints, reverse: boolean, b: BandProfile, entry: number): { pts: Float32Array; lane: number | Uint8Array; band: Float32Array; ease: Uint8Array } {
  const p = dr.lpts, n = p.length / 3, S = bpScratch;
  if (S.at.length < n) S.at = new Float64Array(n * 2);
  const at = S.at;
  at[0] = 0;
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
  const perM = kAtY(p[1]), len = at[n - 1] / perM;
  const [o, c] = bandKnots(b, entry);
  // the knots, m along the edge a to b
  const cuts = S.cuts;
  cuts.length = 0;
  for (let i = 0; i < c; i++) cuts.push((reverse ? 1 - b[o + i * KNOT] : b[o + i * KNOT]) * len);
  cuts.sort((x, y) => x - y);
  // the knots' places are in order, so the knot a place falls after is found by bisection (and a profile
  // out of order is read as `profileAt` reads it)
  let sorted = true;
  for (let i = 1; i < c && sorted; i++) if (!(b[o + i * KNOT] >= b[o + (i - 1) * KNOT])) sorted = false;
  /** the first knot, from the second on, with a place at or after `v`; c when none */
  const firstGE = (v: number) => {
    if (!sorted) {
      let i = 1;
      while (i < c && !(v <= b[o + i * KNOT])) i++;
      return i;
    }
    let lo = 1, hi = c;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (v <= b[o + mid * KNOT]) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  };
  // (at most a point per point and per knot)
  const cap = n + c;
  if (S.lane.length < cap) {
    S.pts = new Float32Array(cap * 6);
    S.lane = new Uint8Array(cap * 2);
    S.band = new Float32Array(cap * 6);
    S.ease = new Uint8Array(cap * 2);
  }
  const out = S.pts, lane = S.lane, band = S.band, ms = S.ms;
  ms.length = 0;
  let np = 0;
  const laneAt = (k: number) => (typeof dr.lane === "number" ? dr.lane : dr.lane[k]);
  const push = (x: number, y: number, z: number, m: number, l: number) => {
    out[np * 3] = x;
    out[np * 3 + 1] = y;
    out[np * 3 + 2] = z;
    lane[np] = l;
    ms.push(m);
    const q3 = np * 3;
    np++;
    // `profileAt(b, entry, f)`
    const f = len > 0 ? (reverse ? len - m : m) / len : 0;
    if (c <= 0) (band[q3] = 0), (band[q3 + 1] = 0), (band[q3 + 2] = 0);
    else if (f <= b[o]) (band[q3] = b[o + 1]), (band[q3 + 1] = b[o + 2]), (band[q3 + 2] = b[o + 4]);
    else {
      const i = firstGE(f);
      if (i < c) {
        const x = o + i * KNOT, q = x - KNOT, d = b[x] - b[q], t = d > 0 ? (f - b[q]) / d : 1;
        band[q3] = b[q + 1] + (b[x + 1] - b[q + 1]) * t;
        band[q3 + 1] = b[q + 2] + (b[x + 2] - b[q + 2]) * t;
        band[q3 + 2] = b[q + 4] + (b[x + 4] - b[q + 4]) * t;
      } else {
        const x = o + (c - 1) * KNOT;
        (band[q3] = b[x + 1]), (band[q3 + 1] = b[x + 2]), (band[q3 + 2] = b[x + 4]);
      }
    }
  };
  push(p[0], p[1], p[2], 0, laneAt(0));
  let ci = 0;
  for (let k = 0; k < n - 1; k++) {
    const m0 = at[k] / perM, m1 = at[k + 1] / perM;
    // the cuts strictly inside this segment: those at or before its start are done with
    while (ci < cuts.length && cuts[ci] <= m0 + 1e-3) ci++;
    for (let qi = ci; qi < cuts.length && cuts[qi] < m1 - 1e-3; qi++) {
      const c = cuts[qi];
      const f = (c - m0) / (m1 - m0);
      push(p[k * 3] + (p[k * 3 + 3] - p[k * 3]) * f, p[k * 3 + 1] + (p[k * 3 + 4] - p[k * 3 + 1]) * f, p[k * 3 + 2] + (p[k * 3 + 5] - p[k * 3 + 2]) * f, c, Math.round(laneAt(k) + (laneAt(k + 1) - laneAt(k)) * f));
    }
    push(p[k * 3 + 3], p[k * 3 + 4], p[k * 3 + 5], m1, laneAt(k + 1));
  }
  // per segment, a to b: whether it lies between two knots that differ (in either view)
  const ease = S.ease, ne = Math.max(0, np - 1);
  for (let i = 0; i < ne; i++) {
    const f = len > 0 ? ((ms[i] + ms[i + 1]) / 2) / len : 0, t = reverse ? 1 - f : f;
    const j = c > 1 ? Math.min(firstGE(t), c - 1) : 1;
    const x = o + (j - 1) * KNOT, y = o + j * KNOT;
    ease[i] = c > 1 && (b[x + 1] !== b[y + 1] || b[x + 2] !== b[y + 2] || b[x + 3] !== b[y + 3] || b[x + 3] < 0 || b[x + 4] !== b[y + 4]) ? 1 : 0;
  }
  const same = np * 3 === p.length;
  return { pts: same ? p : out.subarray(0, np * 3), lane: typeof dr.lane === "number" && same ? dr.lane : lane.subarray(0, np), band: band.subarray(0, np * 3), ease: ease.subarray(0, ne) };
}

/** T-4153: per drawn points of a piece of track whose lanes change along it, the points' distances along (`laneAlong`) */
const laneCum = new WeakMap<Float32Array, Float64Array>();
/** The lane value (0-1) a fraction `f` of the way along an edge's drawn points. */
function laneAlong(dr: LanePoints, f: number): number {
  if (typeof dr.lane === "number") return dr.lane / 255;
  const p = dr.lpts, n = p.length / 3;
  if (n < 2) return dr.lane[0] / 255;
  let at = laneCum.get(p);
  if (!at) {
    at = new Float64Array(n);
    for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
    laneCum.set(p, at);
  }
  const want = f * at[n - 1];
  // the first point from the second on as far along as `want`, the last if none
  let lo = 1, hi = n - 1;
  if (!(at[n - 1] >= 0)) while (lo < hi && !(at[lo] >= want)) lo++;
  else while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (at[mid] >= want) hi = mid;
    else lo = mid + 1;
  }
  const k = lo, s = Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
  const u = s > 0 ? Math.max(0, Math.min(1, (want - at[k - 1]) / s)) : 0;
  return (dr.lane[k - 1] + (dr.lane[k] - dr.lane[k - 1]) * u) / 255;
}

/**
 * A track edge's piece of a build: its strokes, ramps and speeds. A new object whenever any of them
 * changes, so its identity says whether its segments are as in the build before (T-1028).
 */
interface EdgeEntry extends LanePoints {
  /** the points as given (the clock worker's own array, the same object while the edge is unchanged) */
  given: Float32Array;
  /** the points it was made from, kept while later given arrays hold the same points: T-4187: `given` as it was (the clock
   * worker never changes such an array), or a copy when the piece is made again from the array it was made from */
  pts: Float32Array;
  /** T-4187: its id, and TRACK: the edge and colour of each of its strokes' segments, whose flags are `flags` */
  edge: number;
  colour: number;
  a: number;
  b: number;
  flags: number;
  lanes: number;
  /** T-3212: its `SplitEnd`s and length as a key, "" for none */
  split: string;
  /** T-3117: `legShifts` for it, or null */
  shift: LegShift | null;
  strokes: TrackStrokes;
  ramps: number[];
  /** the edge speeds given, and per segment of `strokes` its speed limit (`Strokes.speed`) */
  speedGiven: ArrayLike<number> | undefined;
  vmin: number;
  speed: Float32Array;
  /** per segment, `Strokes.trainSpeed`; `NO_TRAIN` for -1 at every one (no running line's train on it) */
  train: Float32Array;
}
/** T-4187: `EdgeEntry.train` of a piece no train runs over, and `ramps` of one with none, shared */
const NO_TRAIN = new Float32Array(0);
const NO_RAMPS = Object.freeze([]) as unknown as number[];
interface LineEntry {
  /** its whole plan's path and stops (`ClockLine.planPath` or `path`, T-3474), which its strokes draw */
  path: number[];
  /** T-3117: `ClockLine.back`, empty for none */
  back: number[];
  stops: number[];
  /** T-3474: where the part its trains run starts in `path` and `stops` (0 for all of it) */
  pathStart: number;
  stopStart: number;
  running: boolean;
  /** high-speed trains: only such a line is drawn in the high-speed look (T-1494) */
  fast: boolean;
  tram: boolean;
  /** its bands (`LineBands.of`), null on no shared track */
  band: Float32Array | null;
  /** T-4153: `LineBands.own`'s profile it was drawn with (the same array: the same band but for recipes' numbers) */
  bandOwn?: Float32Array;
  strokes: LineStrokes;
  /** T-4187: its id, the colour of each of its strokes' segments */
  colour: number;
  /** per entry of `path`, its first stroke segment, -1 none */
  firstStroke: number[];
  /** T-3474: how many of `strokes` the path's entries made (the return run's own follow) */
  forwardCount: number;
  /** T-3250: per entry of `back`, its first stroke segment (the path's own on an edge both use), -1 none */
  backStroke: number[];
  /** T-4153: per entry of `path` drawn, how: `BITS_POPPED`, `BITS_ZEROED`, `BITS_BANDED` */
  bits?: Uint8Array;
}
/** (`LineEntry.bits`) its last segment came back a step (`StrokeBuilder.popped`); its band's width share zeroed (alone on
 * single track); drawn in its band (`bandPoints`) */
const BITS_POPPED = 1, BITS_ZEROED = 2, BITS_BANDED = 4;
/** T-4153: per entry of `p`, the entry of `q` it stands for where the two are alike but for one stretch (-1 inside it) */
function alikeEntries(q: readonly number[], p: readonly number[]): Int32Array {
  const n = Math.min(q.length, p.length), out = new Int32Array(p.length).fill(-1);
  let pre = 0, suf = 0;
  while (pre < n && q[pre] === p[pre]) pre++;
  while (suf < n - pre && q[q.length - 1 - suf] === p[p.length - 1 - suf]) suf++;
  for (let k = 0; k < pre; k++) out[k] = k;
  for (let k = p.length - suf; k < p.length; k++) out[k] = k - p.length + q.length;
  return out;
}
/** T-4153: per entry of a line's path, the stroke segment its strokes end before */
function strokeEnds(c: LineEntry): Int32Array {
  const out = new Int32Array(c.path.length);
  let end = c.forwardCount;
  for (let k = c.path.length - 1; k >= 0; k--) {
    out[k] = end;
    if (c.firstStroke[k] >= 0) end = c.firstStroke[k];
  }
  return out;
}
/** T-4153: entry `ka` of band `a` and `kb` of `b` read alike by `bandPoints`: the same knots (bit for bit), their recipes
 * alike where it reads them (whether one is none, whether two in a row differ) */
function sameKnots(a: BandProfile, ka: number, b: BandProfile, kb: number): boolean {
  const [oa, ca] = bandKnots(a, ka), [ob, cb] = bandKnots(b, kb);
  if (ca !== cb) return false;
  for (let i = 0; i < ca; i++) {
    const x = oa + i * KNOT, y = ob + i * KNOT;
    if (!Object.is(a[x], b[y]) || !Object.is(a[x + 1], b[y + 1]) || !Object.is(a[x + 2], b[y + 2]) || !Object.is(a[x + 4], b[y + 4])) return false;
    if (a[x + 3] < 0 !== b[y + 3] < 0) return false;
    if (i > 0 && (a[x + 3] !== a[x + 3 - KNOT]) !== (b[y + 3] !== b[y + 3 - KNOT])) return false;
  }
  return true;
}
/** A line's `lineTrainSpeeds` and what they were worked out from. */
interface LineSpeeds {
  /** T-4153: the line's row, the phase tables and its own profile meta (`metaOf`) these were worked out with */
  row: ClockState["lines"][number];
  phases: Float32Array;
  meta: Float64Array;
  path: number[];
  lens: number[];
  ph: Float32Array | null;
  strokes: (TrackStrokes | undefined)[];
  built: boolean[];
  out: TrainSpeeds | null;
}
/** A line's train samples' strokes and lanes (`sampleStroke` less its base, `sampleLane`). */
interface LineSamples {
  c: LineEntry;
  /** the path (the row's own array) they are along */
  path: readonly number[];
  count: number;
  step: number;
  lens: number[];
  rel: Int32Array;
  lane: Float32Array;
  /** `sampleBand`'s, empty for a line on no shared track */
  band: Float32Array;
  /** a bus: its lane is the centre line, so a rail line never takes its samples */
  bus: boolean;
  /** T-4153: per entry of `path` its first sample, and `count` past the last (`entrySamples`) */
  ks: Int32Array;
}

/** T-4153: a run's first sample on each entry of its path, and `count` past the last: sample k (k * step metres along)
 * is on the first entry whose end it is not past, or on the last */
function entrySamples(lens: readonly number[], count: number, step: number): Int32Array {
  const n = lens.length, ks = new Int32Array(n + 1);
  let end = 0, k = 0;
  for (let j = 0; j < n - 1; j++) {
    ks[j] = k;
    end += lens[j];
    let e = step > 0 && Number.isFinite(end) ? Math.max(k, Math.min(count, Math.floor(end / step))) : k;
    while (e < count && !(e * step > end)) e++;
    while (e > k && (e - 1) * step > end) e--;
    k = e;
  }
  if (n) ks[n - 1] = k;
  ks[n] = count;
  return ks;
}

/** T-4153: a run's samples made again entry by entry: on the first `upTo` entries (alike from the start of `was`'s)
 * copied from it, on an entry whose lane and band are one value all along filled with it (what each sample there works
 * out to, bit for bit), the rest sample by sample. `bandOf`, `flipOf`: a return run's own samples' entry of the path
 * whose band they take (-1 none), and whether turned round there. */
function makeSamples(c: LineEntry, path: readonly number[], lens: number[], count: number, step: number, strokeOf: readonly number[], bus: boolean, drawn: ReadonlyMap<number, LanePoints>, bandOf: ((j: number) => number) | null, flipOf: ((j: number, f: number) => boolean) | null, was: LineSamples | undefined, upTo: number): LineSamples {
  const n = path.length, ks = entrySamples(lens, count, step), B = c.band;
  const rel = new Int32Array(count), lane = new Float32Array(count).fill(bus ? 0 : 1), band = new Float32Array(B ? count * 2 : 0);
  let end = 0;
  for (let j = 0; j < n; j++) {
    const len = lens[j] ?? 0;
    end += lens[j];
    const ka = ks[j], kb = ks[j + 1];
    if (ka >= kb) continue;
    rel.fill(strokeOf[j] ?? -1, ka, kb);
    let k = ka;
    if (was && j < upTo) {
      let cb = Math.min(kb, was.count);
      // (a sample past the path's end is on its last entry, which need not be where the last path had it)
      if (j === n - 1 && j !== was.path.length - 1) {
        let e = ka;
        while (e < cb && e * step <= end) e++;
        cb = e;
      }
      if (cb > ka) {
        rel.set(was.rel.subarray(ka, cb), ka);
        lane.set(was.lane.subarray(ka, cb), ka);
        if (band.length) band.set(was.band.subarray(ka * 2, cb * 2), ka * 2);
        netWork.kept += cb - ka;
        netWork.keptAll += cb - ka;
        k = cb;
      }
    }
    if (k >= kb) continue;
    const p = path[j], dr = drawn.get(p >> 1), rev = (p & 1) === 1;
    const f = B ? (bandOf ? bandOf(j) : j) : -1, flip = f >= 0 && !!flipOf && flipOf(j, f);
    // how far apart the tracks are (T-658), one value where the drawn piece's lanes are (NaN: they change along it)
    let lv = NaN;
    if (dr) {
      if (typeof dr.lane === "number") lv = dr.lane / 255;
      else if (dr.lpts.length < 6) lv = dr.lane[0] / 255;
      else {
        const L = dr.lane;
        let i = 1;
        while (i < L.length && L[i] === L[0]) i++;
        if (i === L.length) lv = L[0] / 255;
      }
    }
    // its band's shift and metres (T-1799, T-2132), one value where every knot of its entry has the same (NaN: not)
    let bv = NaN, bm = NaN;
    if (f >= 0) {
      const [o, cn] = bandKnots(B!, f);
      if (cn <= 0) (bv = 0), (bm = 0);
      else {
        const v0 = B![o + 1], m0 = B![o + 4];
        let i = Number.isFinite(v0) && Number.isFinite(m0) && !Object.is(v0, -0) && !Object.is(m0, -0) ? 1 : cn + 1;
        while (i < cn && Object.is(B![o + i * KNOT + 1], v0) && Object.is(B![o + i * KNOT + 4], m0)) i++;
        if (i === cn) (bv = flip ? -v0 : v0), (bm = flip ? -m0 : m0);
      }
    }
    const laneOne = !dr || !Number.isNaN(lv), bandOne = f < 0 || !Number.isNaN(bv);
    if (dr && laneOne) lane.fill(lv, k, kb);
    if (f >= 0 && bandOne) for (let q = k; q < kb; q++) (band[q * 2] = bv), (band[q * 2 + 1] = bm);
    if (laneOne && bandOne) continue;
    for (; k < kb; k++) {
      const along = len > 0 ? Math.max(0, Math.min(1, 1 - (end - k * step) / len)) : 0;
      if (!laneOne) lane[k] = laneAlong(dr!, rev ? 1 - along : along);
      if (!bandOne) profileInto(B!, f, flip ? 1 - along : along, band, k * 2, flip);
    }
  }
  return { c, path, count, step, lens, rel, lane, band, bus, ks };
}

/** T-4153: a run's samples' strokes (from `base`, the line's first) and stretches into the network's tables at `first`,
 * a fill per entry (its samples all take its first stroke) */
function writeSamples(sm: LineSamples, first: number, base: number, stretchOf: ArrayLike<number>, stroke: Int32Array, stretch: Float32Array) {
  const ks = sm.ks, rel = sm.rel;
  for (let j = 0; j + 1 < ks.length; j++) {
    const a = ks[j], b = ks[j + 1];
    if (a >= b) continue;
    const f = rel[a];
    if (f < 0) continue;
    stroke.fill(base + f, first + a, first + b);
    stretch.fill(stretchOf[f], first + a, first + b);
  }
}

/** T-2233: per sample of a run `step` m apart, how far it is past the nearest passing place's platform (`linePassing`) */
function passValues(mine: readonly number[], count: number, step: number): Float32Array {
  const out = new Float32Array(count);
  for (let k = 0; k < count; k++) {
    let v = 1e9;
    for (let j = 0; j < mine.length; j += 4) {
      const x = k * step - mine[j], past = Math.abs(x) - mine[j + 1];
      v = Math.min(v, past > 0 ? past * PASS_LEAD_M / mine[x < 0 ? j + 2 : j + 3] : past);
    }
    out[k] = v;
  }
  return out;
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
  /** T-4153: per piece of track, the fastest train over each of its segments (`EdgeEntry.train`) where any runs */
  trains: Map<number, Float32Array>;
  /** per sampled run by its row key (line id times 2, plus 1 for a return run's own way back, trainSlots.ts `rowKey`) */
  samples: Map<number, LineSamples>;
  /** each complex's drawn passages, by its first member's id (`complexPassages`) */
  passages: Map<number, PassageEntry>;
  /** T-3803: the waiting rooms' and car parks' placement, per complex */
  facilities: FacilityCache;
  /** T-4153: the bands' own kept work (map/lineBands.ts), this cache's alone */
  bands: BandState;
  /** T-4153: what the last build read of the edges alone, and of its nodes (`EdgeSide`) */
  side: EdgeSide | null;
  /** the line rows the pieces in `lines` stand for */
  lineRows: Map<number, ClockState["lines"][number]>;
  counts: LineCounts | null;
  /** each node's edges in the order stations take them (`waysAt`), and the edges and counts of lines that order is from */
  ways: { side: EdgeSide; counts: Map<number, number>; ways: Map<number, number[]> } | null;
  /** per line, its turn-backs' strokes (`turnbackStrokes`) and what they were made from */
  turnbacks: Map<number, { row: ClockState["lines"][number]; c: LineEntry; n0: unknown; n1: unknown; d0: boolean; d1: boolean; rh: boolean | undefined; calls: StrokeAdd[] }>;
  /** T-4153: each node id's slot in `nodes.idx` (node ids can be far apart) */
  nodeSlots: Map<number, number>;
  /** T-4153: the station buildings' index of the edges' boxes (map/buildingNet.ts) */
  bldBoxes: EdgeBoxes;
  /** T-4153: the last build's turn-backs' strokes and the lists of calls they were made from */
  turnStrokes: { lists: StrokeAdd[][]; strokes: Strokes } | null;
  /** per single-track station, the side its one platform is on (`stationOutside`) and what it was worked out from */
  outside: Map<number, { side: EdgeSide; ways: Map<number, number[]>; c: unknown; rh: boolean | undefined; v: number; edges: number[]; nodes: number[] }>;
  /** T-4153: the stations' arrays as the last finished build's loop left them (before the buildings and facilities mark
   * them), its stations, each id's index in them, and the running side they are of */
  stArr: { st: readonly ClockState["nodes"][number][]; index: Map<number, number>; rh: boolean | undefined; stations: Float32Array; axis: Float32Array; shape: Float32Array; line: Float32Array } | null;
  /** the passing tracks' strokes (`passingTracks`) and what they were made from */
  passTracks: Map<number, { key: unknown[]; calls: StrokeAdd[] }>;
  /** T-4280: per sampled run (by its row key, as `samples`), where along it its trains pass at passing stations (`linePassing`), when anywhere */
  passOf: Map<number, number[]>;
  /** T-4153: per sampled run (by its row key), its samples' passing values (`passValues`) and what they are of */
  passVals: Map<number, { count: number; step: number; mine: number[]; v: Float32Array }>;
  /** T-4153: per edge the lines whose rows (`edgeRows`, those of the last build) use it, plan, run or way back */
  edgeLines: Map<number, Set<number>>;
  edgeRows: Map<number, ClockState["lines"][number]>;
  /** the nodes a line's edge ends at, and the edges and counts they are from */
  lined: { side: EdgeSide; counts: Map<number, number>; lined: Set<number> } | null;
  nodes: { rows: readonly ClockState["nodes"][number][]; idx?: Int32Array; st?: ClockState["nodes"][number][]; jn?: ClockState["nodes"][number][]; stationIds?: string[] } | null;
  /** T-4153: the last build's pieces of track as a whole, and what they were made from */
  loop: { side: EdgeSide; splits: ReturnType<typeof singleSplitEnds>; split?: SplitState; edgePts: Map<number, Float32Array>; splitEdge: Set<number>; trackOrder: number[]; ramps: number[] } | null;
  serial: number;
  last: { serial: number; track: EdgeEntry[]; lines: LineEntry[]; trackCount: number; lineCount: number; pts: Map<number, Float32Array> } | null;
}

export const newNetCache = (): NetCache => ({ edges: new Map(), lines: new Map(), stations: new Map(), speeds: new Map(), trains: new Map(), samples: new Map(), passages: new Map(), facilities: newFacilityCache(), bands: newBandState(), side: null, lineRows: new Map(), counts: null, ways: null, lined: null, turnbacks: new Map(), nodeSlots: new Map(), bldBoxes: newEdgeBoxes(), turnStrokes: null, outside: new Map(), stArr: null, passTracks: new Map(), passOf: new Map(), passVals: new Map(), edgeLines: new Map(), edgeRows: new Map(), nodes: null, loop: null, serial: 0, last: null });

/**
 * T-1170: the cache's pieces moved to a new draw anchor, so the next build reuses them rather than
 * laying every stroke again. `given` is each edge's points at the new anchor (the clock worker's,
 * exact); the strokes' and ramps' positions are the old ones plus (dx, dy) local units, rounded to
 * float32 again, which can put a point off its exact place by half a float32 step at its distance
 * from the old anchor (about 3 cm at 1,000 km, 15 cm at 4,000 km). Stations keep their pieces: their
 * shapes are measured from their node. An edge whose piece was not made from `was`'s points (the
 * clock worker's at the old anchor) is left out and built anew.
 */
export function shiftNetCache(cache: NetCache, was: ReadonlyMap<number, Float32Array>, given: ReadonlyMap<number, Float32Array>, dx: number, dy: number) {
  const moved = new Map<TrackStrokes, TrackStrokes>();
  // (T-4187: packed again, so the moved segments do not leave the old ones held in the piece's buffer)
  const strokes = <S extends TrackStrokes>(s: S): S => {
    let t = moved.get(s) as S | undefined;
    if (!t) {
      moved.set(s, (t = repacked(s)));
      const seg = t.seg;
      for (let k = 0; k + 1 < seg.length; k += 2) { seg[k] = s.seg[k] + dx; seg[k + 1] = s.seg[k + 1] + dy; }
    }
    return t;
  };
  const movedPts = new Map<Float32Array, Float32Array>();
  for (const [id, c] of cache.edges) {
    const g = given.get(id);
    if (!g || g.length !== c.pts.length || c.given !== was.get(id)) { cache.edges.delete(id); continue; }
    const pts = g;
    movedPts.set(c.pts, pts);
    let lpts = pts;
    if (c.lpts !== c.pts) {
      lpts = new Float32Array(c.lpts.length);
      for (let k = 0; k + 2 < lpts.length; k += 3) { lpts[k] = c.lpts[k] + dx; lpts[k + 1] = c.lpts[k + 1] + dy; lpts[k + 2] = c.lpts[k + 2]; }
    }
    const ramps = c.ramps.length ? c.ramps.slice() : NO_RAMPS;
    for (let k = 0; k + 1 < ramps.length; k += MARK_STRIDE) { ramps[k] = Math.fround(ramps[k] + dx); ramps[k + 1] = Math.fround(ramps[k + 1] + dy); }
    // (its speeds out of the old strokes' buffer, `keepTrack`, so that is let go)
    const speed = c.speed.buffer === c.strokes.seg.buffer ? c.speed.slice() : c.speed;
    cache.edges.set(id, { ...c, given: g, pts, lpts, strokes: strokes(c.strokes), ramps, speed });
  }
  const lines = new Map<LineEntry, LineEntry>();
  for (const [id, c] of cache.lines) {
    const n = { ...c, strokes: strokes(c.strokes) };
    lines.set(c, n);
    cache.lines.set(id, n);
  }
  for (const [id, sm] of cache.samples) {
    const c = lines.get(sm.c);
    if (c) cache.samples.set(id, { ...sm, c });
    else cache.samples.delete(id);
  }
  for (const [id, sp] of cache.speeds) cache.speeds.set(id, { ...sp, strokes: sp.strokes.map((s) => (s ? moved.get(s) ?? s : s)) });
  cache.last = null;
  cache.side = null;
  cache.loop = null;
  cache.stArr = null;
  cache.turnStrokes = null;
  cache.bldBoxes = newEdgeBoxes();
  shiftKept(movedPts, cache.bands);
}

/**
 * A network posted as changes (`NetPatch`) made whole from the one before it, `prev`, in place: the
 * fields left out taken from it (the run profiles from `service` when a train count came between),
 * its edges' points from both. False, changing nothing that matters, when `prev` is not the build the
 * changes are to.
 */
export function mergeNet(prev: NetworkBuffers | null, n: NetworkBuffers, service?: { phases: Float32Array; meta: Float32Array } | null): boolean {
  const p = n.patch;
  if (!p) {
    if (n.checkStrokes) checkMergedStrokes(n, "whole");
    return true;
  }
  if (!prev || prev.serial !== p.base || prev.track.count !== p.oldTrack || prev.lines.count !== p.oldLines) return false;
  const from: Record<string, unknown> = { ...prev, ...(service ?? {}) };
  for (const k of p.same) if (from[k] === undefined) return false;
  const pts = new Map<number, Float32Array>();
  // T-4284: the edges taken out, read off the last build's ids beside these (both ascend, the model's order)
  const was = prev.edgePts.keys(), gone: number[] = [];
  let w = was.next(), last = -Infinity, ordered = true;
  for (const id of p.edgeIds) {
    const v = n.edgePts.get(id) ?? prev.edgePts.get(id);
    if (!v) return false;
    pts.set(id, v);
    if (!(id > last)) ordered = false;
    last = id;
    while (!w.done && w.value < id) { gone.push(w.value); w = was.next(); }
    if (!w.done && w.value === id) w = was.next();
  }
  for (; !w.done; w = was.next()) gone.push(w.value);
  const set = [...n.edgePts.keys()];
  const track = planStrokes("track", prev.track, n.track, p.modes.track, p.ranges.track, p.moves?.track), lines = planStrokes("lines", prev.lines, n.lines, p.modes.lines, p.ranges.lines, p.moves?.lines);
  if (!track || !lines) return false;
  // nothing is written into the last network's arrays unless the whole network can be merged
  track.write();
  lines.write();
  for (const k of p.same) (n as unknown as Record<string, unknown>)[k] = from[k];
  n.edgePts = pts;
  // (the last build's ids out of order: not known, and `gone` cannot be trusted)
  if (ordered && prev.serial !== undefined && gone.length + pts.size === prev.edgePts.size + set.filter((id) => !prev.edgePts.has(id)).length) n.edgeStep = { base: prev.serial, set, gone };
  n.track = track.strokes;
  n.lines = lines.strokes;
  if (n.checkStrokes) checkMergedStrokes(n, `${p.modes.track}/${p.modes.lines}`);
  return true;
}

/** `?patch=check`: the strokes as merged against those the worker made whole beside them (T-1381). */
function checkMergedStrokes(n: NetworkBuffers, modes: string) {
  const bad: string[] = [];
  for (const kind of ["track", "lines"] as const) {
    const a = n[kind], b = n.checkStrokes![kind];
    for (const f of Object.keys(b) as (keyof Strokes)[]) {
      const x = a[f], y = b[f];
      if (!ArrayBuffer.isView(x) || !ArrayBuffer.isView(y)) { if (x !== y) bad.push(`${kind}.${f}`); continue; }
      let same = x.byteLength === y.byteLength;
      if (same) {
        const u = new Uint8Array(x.buffer, x.byteOffset, x.byteLength), v = new Uint8Array(y.buffer, y.byteOffset, y.byteLength);
        for (let i = 0; i < u.length && same; i++) same = u[i] === v[i];
      }
      if (!same) bad.push(`${kind}.${f}`);
    }
  }
  delete n.checkStrokes;
  if (bad.length) console.error(`T-1381 check: strokes posted as ${modes} differ from the whole ones in ${bad.join(", ")}`);
}

/** Runs (start, count, old start or -1) making `parts` (in order, `count` of each) from `old`. */
export function strokeRuns<P extends object>(parts: P[], count: (p: P) => number, old: P[] | undefined): Int32Array {
  // (T-4153: the same pieces in the same order are one run from the start)
  if (old && (parts === old || sameRefs(parts, old))) {
    let n = 0;
    for (const p of parts) n += count(p);
    return new Int32Array(n ? [0, n, 0] : []);
  }
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

/** the same objects at every index */
const sameRefs = (a: readonly unknown[] | undefined, b: readonly unknown[] | undefined) => {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

/** One `StrokeBuilder.add` call, kept to make again. */
type StrokeAdd = Parameters<StrokeBuilder["add"]>;
/** A builder that keeps the calls made to it (`StrokeAdd`), for strokes whose inputs stay while the edges do. */
class StrokeLog {
  calls: StrokeAdd[] = [];
  add(...a: StrokeAdd): number { this.calls.push(a); return 0; }
}

/** T-4153: a node's edges (ids and rows, in edge order, an edge ending there twice listed twice) and what they make it */
interface NodeEnds {
  ids: number[];
  /** the last of its edges in edge order (the one its place is taken from) */
  last: ClockEdge;
  single: number;
  double: number;
  joined: boolean;
  /** its place (local), from the last of its edges' points (the same numbers while this is kept), once asked for */
  xy?: readonly [number, number];
}
function nodeEnds(node: number, ids: number[], rows: ReadonlyMap<number, ClockEdge>): NodeEnds {
  let single = 0, double = 0, joined = false;
  for (const id of ids) {
    const e = rows.get(id)!;
    if (e.tracks === 1) {
      single++;
      if ((e.a === node && (e.join & 3)) || (e.b === node && ((e.join >> 2) & 3))) joined = true;
    } else if (e.tracks === 2) double++;
  }
  return { ids, last: rows.get(ids[ids.length - 1])!, single, double, joined };
}

/** T-4153: what a build reads of the edges alone (their rows and given points). Rows are kept by identity while
 * unchanged (workers/stateRows.ts), so it is kept while every row and points are, and patched at the nodes of an edge
 * that changed (`patchSide`). */
interface EdgeSide {
  /** each edge's row, by id */
  edgeEnds: Map<number, ClockEdge>;
  /** each node's edges */
  at: Map<number, NodeEnds>;
  shifts: Map<number, LegShift>;
  /** `singleLeadStrokes`' calls, once a build asked for them */
  leads: StrokeAdd[] | null;
  /** the last build's edges as given (rows, points, speeds), and each id's index in them; the current build's once it
   * finishes */
  input: { rows: readonly ClockEdge[]; parts: readonly Float32Array[]; speedParts: readonly (ArrayLike<number> | undefined)[] | undefined; idx: Int32Array };
  /** patched from this side, at these nodes */
  from?: EdgeSide;
  touched?: Set<number>;
}
const noInput: EdgeSide["input"] = { rows: [], parts: [], speedParts: undefined, idx: new Int32Array(0) };
function edgeSide(edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[]): EdgeSide {
  const edgeEnds = new Map<number, ClockEdge>(), ids = new Map<number, number[]>();
  for (const e of edges) {
    edgeEnds.set(e.id, e);
    for (const x of [e.a, e.b]) {
      const v = ids.get(x);
      if (v) v.push(e.id);
      else ids.set(x, [e.id]);
    }
  }
  const at = new Map<number, NodeEnds>();
  for (const [n, v] of ids) at.set(n, nodeEnds(n, v, edgeEnds));
  return { edgeEnds, at, shifts: legShifts(edges, parts), leads: null, input: noInput };
}
/** `prev` made the side of `edges` again at the nodes of the edges `diff` names */
function patchSide(prev: EdgeSide, edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[], diff: EdgeDiff): EdgeSide {
  const edgeEnds = new Map(prev.edgeEnds), touched = new Set<number>(), gone = new Set<number>();
  for (const e of diff.removed) { edgeEnds.delete(e.id); touched.add(e.a).add(e.b); gone.add(e.id); }
  for (const id of diff.rows) {
    const was = prev.edgeEnds.get(id), e = edges[diff.idx[id]];
    if (was) touched.add(was.a).add(was.b);
    edgeEnds.set(id, e);
    touched.add(e.a).add(e.b);
    gone.add(id);
  }
  const at = new Map(prev.at);
  for (const n of touched) {
    const ids = (prev.at.get(n)?.ids ?? []).filter((id) => !gone.has(id));
    for (const id of diff.rows) {
      const e = edgeEnds.get(id)!;
      if (e.a === n) ids.push(id);
      if (e.b === n) ids.push(id);
    }
    if (!ids.length) { at.delete(n); continue; }
    ids.sort((x, y) => diff.idx[x] - diff.idx[y]);
    at.set(n, nodeEnds(n, ids, edgeEnds));
  }
  return { edgeEnds, at, shifts: patchLegShifts(prev.shifts, edges, parts, at, diff.idx, touched, diff.removed), leads: null, input: noInput, from: prev, touched };
}

/** T-4153: how the edges differ from the last build's: `rows`, ids whose row is new or whose points' numbers changed;
 * `removed`, the last build's rows of ids gone; `moved`, ids whose points or speeds are new arrays (perhaps of the
 * same numbers), rows the same; `idx`, each id's index now. Null when so many changed that a whole build is quicker. */
interface EdgeDiff { rows: number[]; removed: ClockEdge[]; moved: number[]; idx: Int32Array }
function edgeDiff(prev: EdgeSide, edges: readonly ClockEdge[], parts: readonly Float32Array[], speedParts: readonly (ArrayLike<number> | undefined)[] | undefined): EdgeDiff | null {
  // (mostly the same rows in the same places: told apart by index)
  const was = prev.input;
  if (was.rows.length === edges.length) {
    const rows: number[] = [], moved: number[] = [];
    let same = true;
    for (let i = 0; i < edges.length && same; i++) {
      const e = edges[i], w = was.rows[i];
      if (w !== e) {
        if (w.id !== e.id) same = false;
        else rows.push(e.id);
        continue;
      }
      if (was.parts[i] !== parts[i]) (sameFloats(was.parts[i], parts[i]) ? moved : rows).push(e.id);
      else if (was.speedParts?.[i] !== speedParts?.[i]) moved.push(e.id);
    }
    if (same) return rows.length > edges.length / 4 ? null : { rows, removed: [], moved, idx: was.idx };
  }
  let top = -1;
  for (const e of edges) if (e.id > top) top = e.id;
  const idx = new Int32Array(top + 1).fill(-1), rows: number[] = [], moved: number[] = [];
  let matched = 0;
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i], j = e.id < was.idx.length ? was.idx[e.id] : -1, w = j >= 0 ? was.rows[j] : undefined;
    idx[e.id] = i;
    if (w) matched++;
    if (w !== e) { rows.push(e.id); continue; }
    if (was.parts[j] !== parts[i]) (sameFloats(was.parts[j], parts[i]) ? moved : rows).push(e.id);
    else if (was.speedParts?.[j] !== speedParts?.[i]) moved.push(e.id);
  }
  const removed: ClockEdge[] = [];
  if (matched < was.rows.length) for (const e of was.rows) if (e.id > top || idx[e.id] < 0) removed.push(e);
  if (rows.length + removed.length > edges.length / 4 || moved.length > edges.length / 2) return null;
  return { rows, removed, moved, idx };
}

/** T-4153: the edges whose piece of track may differ from the last build's: those that changed, those at a node whose
 * edges changed (its tracks decide a double's taper), and those whose single track's split or legs' shift changed */
function edgePieceDirty(diff: EdgeDiff, side: EdgeSide, prev: EdgeSide, wasSplits: ReturnType<typeof singleSplitEnds>, splits: ReturnType<typeof singleSplitEnds>): Set<number> {
  const out = new Set<number>(diff.rows);
  for (const id of diff.moved) out.add(id);
  const atNode = (n: number) => { for (const id of side.at.get(n)?.ids ?? []) out.add(id); };
  if (side.touched) for (const n of side.touched) atNode(n);
  if (splits !== wasSplits) {
    for (const [id, v] of splits.ends) {
      const w = wasSplits.ends.get(id);
      if (!w || w.length !== v.length || v.some((x, k) => x.a !== w[k].a || x.p !== w[k].p || x.lead !== w[k].lead)) out.add(id);
    }
    for (const id of wasSplits.ends.keys()) if (!splits.ends.has(id)) out.add(id);
    for (const [a, b] of [[splits.stations, wasSplits.stations], [wasSplits.stations, splits.stations], [splits.full, wasSplits.full], [wasSplits.full, splits.full]]) for (const n of a) if (!b.has(n)) atNode(n);
  }
  if (side.shifts !== prev.shifts) {
    for (const [id, v] of side.shifts) if (prev.shifts.get(id)?.key !== v.key) out.add(id);
    for (const [id, v] of prev.shifts) if (side.shifts.get(id)?.key !== v.key) out.add(id);
  }
  return out;
}

/** Each edge id's index in `edges`, -1 for none. */
function edgeIndex(edges: readonly ClockEdge[]): Int32Array {
  let top = -1;
  for (const e of edges) if (e.id > top) top = e.id;
  const idx = new Int32Array(top + 1).fill(-1);
  edges.forEach((e, i) => { idx[e.id] = i; });
  return idx;
}

/** The edges' ids in drawing order: built track first, then blueprint, each in edge order (T-695). */
function drawOrder(edges: readonly ClockEdge[]): number[] {
  const out: number[] = [];
  for (const e of edges) if (e.built) out.push(e.id);
  for (const e of edges) if (!e.built) out.push(e.id);
  return out;
}

/** The renderer's buffers from a state. With `cache` (the clock worker's), pieces whose inputs are
 * as in the last build are reused, and `patch` says how the strokes differ from it; the buffers are
 * the same either way. `deferStrokes` (the clock worker's): `track` and `lines` hold only their
 * counts, and `strokePieces` the pieces to make them from (map/strokePost.ts). */
export function networkFromState(s: ClockState, cache: NetCache = newNetCache(), deferStrokes = false): NetworkBuffers {
  const g = networkSteps(s, cache, deferStrokes);
  for (;;) {
    const r = g.next();
    if (r.done) return r.value;
  }
}

/** `networkFromState` in pieces (T-3803): the clock worker runs one a task, so an edit can come between them. The
 * cache is written only in the last piece, so one dropped at a yield leaves it as the last finished build left it
 * (the waiting rooms' placement and the passages, kept against their own last inputs, are in the last piece too). */
/** checks/t4153-net.mjs --self-test strokes: a line's entries copied whatever their band's knots, which its check must catch */
let netTrust = false;
export function netSelfTest(on: boolean) { netTrust = on; }
/** checks/t4153-net.mjs --self-test=stations: a station's numbers kept unless its own node's row changed, which its check
 * must catch (a copy of a line changes the stops it serves) */
let stationTrust = false;
export function stationSelfTest(on: boolean) { stationTrust = on; }
/** T-4153: per part of the last build, ms (for measuring the parts an edit's build still pays for) */
export const netLaps: Record<string, number> = {};
/** T-4153: per build, the pieces of track, lines, line samples, stations and line speeds worked out again (counts) */
export const netWork = { edges: 0, lines: 0, entries: 0, samples: 0, kept: 0, stations: 0, speeds: 0, side: "", entriesAll: 0, keptAll: 0 };
let lapT = 0;
const lap = (k: string) => { const t = performance.now(); netLaps[k] = (netLaps[k] ?? 0) + t - lapT; lapT = t; };
export function* networkSteps(s: ClockState, cache: NetCache = newNetCache(), deferStrokes = false): Generator<void, NetworkBuffers, void> {
  try {
    return yield* networkStepsFrom(s, cache, deferStrokes);
  } catch (e) {
    // (T-4153: a kept structure a patch left half made, as a broken invariant does, would throw again at every build)
    console.warn("network build threw; the kept state is dropped and the next build is whole", e);
    Object.assign(cache, newNetCache());
    throw e;
  }
}
function* networkStepsFrom(s: ClockState, cache: NetCache, deferStrokes: boolean): Generator<void, NetworkBuffers, void> {
  for (const k in netLaps) delete netLaps[k];
  netWork.edges = netWork.lines = netWork.entries = netWork.samples = netWork.kept = netWork.stations = netWork.speeds = 0;
  lapT = performance.now();
  const r = s.render;
  const parts = r.edgeParts!, speedParts = r.edgeSpeedParts;
  // T-4153: the edges compared with the last build's by identity (rows are kept while unchanged, workers/stateRows.ts;
  // an edit's own edges' points are read again, into new arrays), and what they alone give patched where they changed
  const prevSide = cache.side, kept0 = cache.loop;
  const diff = prevSide && kept0?.side === prevSide ? edgeDiff(prevSide, s.edges, parts, speedParts) : null;
  const side = !diff ? edgeSide(s.edges, parts) : diff.rows.length || diff.removed.length ? patchSide(prevSide!, s.edges, parts, diff) : prevSide!;
  /** no edge's row or points changed */
  const edgesSame = side === prevSide;
  netWork.side = !diff ? "whole" : edgesSame ? "kept" : "patched";
  const { shifts, at: nodeEnds } = side;
  const edgeEnds = side.edgeEnds;
  const edgeLen = { get: (id: number) => edgeEnds.get(id)?.lengthM };
  const offEdge = { has: (id: number) => !!edgeEnds.get(id)?.off };
  const hsrEdge = { has: (id: number) => !!edgeEnds.get(id)?.hsr };
  const tramEdge = { has: (id: number) => !!edgeEnds.get(id)?.kind };
  const endCount = { get: (node: number) => nodeEnds.get(node)?.ids.length };
  const edgesAt = { get: (node: number) => nodeEnds.get(node)?.ids };
  // (T-4153: the nodes whose rows changed, came or went since the last finished build; null with none)
  const nodesNow = nodeChanges(cache.nodes?.rows, cache.nodes?.idx, s.nodes, cache.nodeSlots), nodeDiff = nodesNow.changed;
  const nodesSame = !!nodeDiff && !nodeDiff.size;
  // T-3212: single track splits into two through a station; where it does, double track meets two
  // (T-4153: worked out again only for the stations whose walks read a node or edge that changed)
  const endsAtRows = (node: number) => nodeEnds.get(node)?.ids.map((id) => edgeEnds.get(id)!) ?? [];
  const nodeAt = (id: number) => { const k = nodesNow.slots.get(id); return k === undefined || k >= nodesNow.idx.length ? -1 : nodesNow.idx[k]; };
  let splitState: SplitState;
  if (edgesSame && nodesSame && kept0?.split) splitState = kept0.split;
  else if (kept0?.split && diff && nodeDiff) {
    const redo = new Set<number>(nodeDiff), tch = side !== prevSide ? side.touched : undefined;
    const rowMoved = new Set<number>(diff.rows);
    for (const e of diff.removed) rowMoved.add(e.id);
    const nodeMoved = (n: number) => nodeDiff.has(n) || !!tch?.has(n), edgeMoved = (e: number) => rowMoved.has(e);
    if (fault !== "splits") for (const [id, sp] of kept0.split.by) if (!redo.has(id) && (sp.nodes.some(nodeMoved) || sp.edges.some(edgeMoved))) redo.add(id);
    splitState = patchSplits(kept0.split, redo, (id) => { const i = nodeAt(id); return i >= 0 ? s.nodes[i] : undefined; }, endsAtRows, nodeAt);
  } else splitState = splitsWhole(s.edges, s.nodes, { get: endsAtRows });
  const splits = splitState.res;
  lap("pre.counts+splits");
  // double track narrows to single track where the two meet at a plain node (T-658); T-3171: also where double track
  // is the only double at a junction of single track (a double branching off single); T-3190: not where single track
  // carries on one track of the double (a join that counts)
  const taperAt = (node: number) => {
    const a = nodeEnds.get(node);
    return !!a && a.single > 0 && (a.ids.length === 2 || a.double === 1) && !a.joined && !splits.stations.has(node) && !splits.full.has(node);
  };
  lap("pre.legShifts");
  yield;
  lapT = performance.now();
  // the pieces of track: worked out again only where an edge, its ends' tracks, its splits or its legs' shifts changed
  // (`edgePieceDirty`), the rest as the last build left them; all of them as they were, points, order and marks too,
  // when none did (`loopSame`)
  const dirty = diff && kept0 ? edgePieceDirty(diff, side, prevSide!, kept0.splits, splits) : null;
  const loopSame = !!dirty && !dirty.size && !diff!.removed.length;
  let edgePts: Map<number, Float32Array>, splitEdge: Set<number>, trackOrder: number[], ramps: number[];
  const builtEdge = { get: (id: number) => edgeEnds.get(id)?.built };
  /** the pieces of track made again (every one not in it is the last build's) */
  const fresh = new Set<number>();
  const gone = new Set(diff?.removed.map((e) => e.id));
  const keptEdge = (id: number) => !fresh.has(id) && (diff ? !gone.has(id) : edgesSeenHas(id));
  /** an edge whose row (its length, whether built) is as in the last build, and its piece of track too */
  const rowsNew = new Set(diff?.rows);
  const sameEdge = (id: number) => !!diff && keptEdge(id) && !rowsNew.has(id);
  /** per edge id: its piece (`EdgeEntry`, whose points and lanes its track and lines are drawn with); copied before
   * any piece is replaced, while it is the cache's own */
  let edgesSeen: NetCache["edges"];
  const edgesSeenHas = (id: number) => edgesSeen.has(id);
  let seenOwn = !loopSame;
  /** `edgesSeen` with `id`'s piece replaced (copied first while the cache's own) */
  const putEdge = (id: number, c: EdgeEntry) => {
    if (!seenOwn) { edgesSeen = new Map(edgesSeen); seenOwn = true; }
    edgesSeen.set(id, c);
  };
  /** the piece of the edge at index `i` */
  const piece = (i: number): EdgeEntry => {
    const e = s.edges[i];
    const given = parts[i];
    const sg: ArrayLike<number> | undefined = speedParts?.[i];
    const flags = (e.built ? 0 : F_BLUEPRINT | (e.off ? F_OFF : 0)) | (e.hsr ? F_HSR : 0) | (e.kind ? F_TRAM : 0) | (e.kind === 4 ? F_BUSWAY : 0) | (e.tracks === 1 && splits.ends.has(e.id) ? F_SPLIT : 0);
    // (a busway is one road, never two tracks parting)
    const lanes = e.kind === 4 ? 0 : (e.tracks === 2 ? 1 : 0) | (e.tracks === 2 && taperAt(e.a) ? 2 : 0) | (e.tracks === 2 && taperAt(e.b) ? 4 : 0);
    const split = e.tracks === 1 ? splits.ends.get(e.id) ?? null : null;
    const splitKey = split ? `${e.lengthM}|` + split.map((x) => `${+x.a},${x.p},${x.lead}`).join("|") : "";
    let c = cache.edges.get(e.id);
    const shift = shifts.get(e.id) ?? null;
    if (c && c.a === e.a && c.b === e.b && c.flags === flags && c.lanes === lanes && c.split === splitKey && c.shift?.key === shift?.key && (c.given === given || sameFloats(c.pts, given))) {
      if (c.given !== given) c = { ...c, given };
    } else {
      netWork.edges++;
      fresh.add(e.id);
      // T-4187: the clock worker's own points, kept as they are when the piece is first made, but a new array whenever it is
      // made again (as when each piece copied them), so what reads them by identity (the bands, the facilities) sees the
      // piece changed, and never an array it saw before (review23: one a dropped build had)
      const own = c ? given.slice() : given;
      const sh = shift ? shiftedEnds(own, shift) : null;
      const lp = split ? splitPoints(sh ? sh.pts : own, split, e.lengthM) : lanePoints(sh ? sh.pts : own, lanes);
      const b = new StrokeBuilder(false, MIN_SEG_M, true), from: number[] = [];
      b.add(lp.lpts, TRACK, e.id, flags, false, 0, undefined, lp.lane, undefined, undefined, from);
      const ramps: number[] = [];
      rampMarks(own, e.built ? 0 : 1, ramps, e.tracks === 2 ? 2 : 1, e.kind ? 1 : e.hsr ? 2 : 0);
      const { strokes, speed } = b.keepTrack();
      // each stroke segment's segment of the edge's given points, for its speed
      const srcL = from.length === lp.lpts.length / 3 - 1 ? lp.src : from.map((k) => (lp.src ? lp.src[k] : k));
      const src = sh ? Array.from({ length: strokes.count }, (_, j) => sh.src[srcL ? srcL[j] : j] ?? 0) : srcL;
      c = { given, pts: own, edge: e.id, colour: TRACK, a: e.a, b: e.b, flags, lanes, split: splitKey, shift, ...lp, src, strokes, ramps: ramps.length ? ramps : NO_RAMPS, speedGiven: sg, vmin: e.vminKmh, speed: speedsOf(src, speed), train: NO_TRAIN };
    }
    if (c.speedGiven !== sg || (!sg && c.vmin !== e.vminKmh) || c.speed.length !== c.strokes.count) {
      const speed = speedsOf(c.src, new Float32Array(c.strokes.count));
      c = sameFloats(speed, c.speed) ? { ...c, speedGiven: sg, vmin: e.vminKmh } : { ...c, speedGiven: sg, vmin: e.vminKmh, speed };
    }
    return c;
    /** per segment of a piece's strokes (`src`: the segment of the given points each is on), its speed limit */
    function speedsOf(src: number[] | null, into: Float32Array): Float32Array {
      for (let j = 0; j < into.length; j++) {
        const k = src ? src[j] : j;
        into[j] = sg ? (sg[k] ?? 0) : e.vminKmh;
      }
      return into;
    }
  };
  if (loopSame) {
    ({ edgePts, splitEdge, trackOrder, ramps } = kept0!);
    edgesSeen = cache.edges;
  } else if (dirty) {
    seenOwn = false;
    // (each of the last build's maps copied only once something in it changes)
    edgesSeen = cache.edges;
    ({ edgePts, splitEdge } = kept0!);
    let ptsOwn = false, splitOwn = false;
    // (T-4153: with rows changed the points map is laid out again in drawing order below, so it is not copied first)
    const reorder = diff!.rows.length > 0 || diff!.removed.length > 0;
    const ownPts = () => { if (!ptsOwn) { edgePts = new Map(edgePts); ptsOwn = true; } return edgePts; };
    const ownSplit = () => { if (!splitOwn) { splitEdge = new Set(splitEdge); splitOwn = true; } return splitEdge; };
    for (const e of diff!.removed) {
      if (!seenOwn) { edgesSeen = new Map(edgesSeen); seenOwn = true; }
      edgesSeen.delete(e.id);
      if (splitEdge.has(e.id)) ownSplit().delete(e.id);
    }
    let rampsSame = !diff!.removed.length;
    for (const id of dirty) {
      const i = diff!.idx[id];
      if (i === undefined || i < 0) continue;
      const was = cache.edges.get(id), c = piece(i);
      if (c.ramps !== was?.ramps) rampsSame = false;
      if (c !== was) putEdge(id, c);
      if (!reorder && c.pts !== edgePts.get(id)) ownPts().set(id, c.pts);
      if (splitEdge.has(id) !== ((c.flags & F_SPLIT) !== 0)) {
        if (c.flags & F_SPLIT) ownSplit().add(id);
        else ownSplit().delete(id);
      }
    }
    // (the drawing order is the edges' own, built first, so it holds while no row did)
    trackOrder = reorder ? drawOrder(s.edges) : kept0!.trackOrder;
    // (T-4153: rows changed in place, none came, went or was built: the same order)
    if (trackOrder !== kept0!.trackOrder && sameArray(trackOrder, kept0!.trackOrder)) trackOrder = kept0!.trackOrder;
    // (in drawing order, as a whole build makes it)
    if (trackOrder !== kept0!.trackOrder) {
      const m = new Map<number, Float32Array>();
      for (let k = 0; k < trackOrder.length; k++) { const id = trackOrder[k]; m.set(id, edgesSeen.get(id)!.pts); }
      edgePts = m;
    } else if (reorder) {
      // (the same order: the kept map with the pieces made again)
      for (const id of dirty) { const c = edgesSeen.get(id); if (c && c.pts !== edgePts.get(id)) ownPts().set(id, c.pts); }
    }
    if (rampsSame && trackOrder === kept0!.trackOrder) ramps = kept0!.ramps;
    else {
      ramps = [];
      for (let k = 0; k < trackOrder.length; k++) {
        const r = edgesSeen.get(trackOrder[k])!.ramps;
        for (let q = 0; q < r.length; q++) ramps.push(r[q]);
      }
    }
  } else {
    edgePts = new Map();
    splitEdge = new Set();
    trackOrder = drawOrder(s.edges);
    ramps = [];
    edgesSeen = new Map();
    // (built track first, so blueprint is drawn over it and picked first where they overlap, as the station tool
    // prefers it, T-695)
    for (const built of [true, false]) s.edges.forEach((e, i) => {
      if (e.built !== built) return;
      const c = piece(i);
      if (c.flags & F_SPLIT) splitEdge.add(e.id);
      edgesSeen.set(e.id, c);
      edgePts.set(e.id, c.pts);
      for (const v of c.ramps) ramps.push(v);
    });
  }
  const marks = ramps.slice();
  const drawn: ReadonlyMap<number, LanePoints> = edgesSeen;
  lap("edges.loop");
  yield;
  lapT = performance.now();
  const maxLine = s.lines.reduce((m, l) => Math.max(m, l.id), -1);
  const lineColours = new Uint8Array(Math.max(1, maxLine + 1) * 4);
  const lineCars = new Float32Array(r.lineTable.length / 4).fill(8);
  for (const l of s.lines) lineCars[l.id] = l.bus?.ferry ? BOAT_CAR[boatOf(l.bus)] : l.bus ? BUS_CAR : l.tram ? Math.max(1, l.cars) + TRAM_MARK : Math.max(1, l.cars);
  const lineParts: LineEntry[] = [];
  let lineCount = 0;
  const linesSeen: NetCache["lines"] = new Map();
  const samplesSeen: NetCache["samples"] = new Map();
  /** T-4280: `NetPatch.stretches` and `lanes`, and each run's passing places (`NetCache.passOf`) */
  const stretchesNew = new Set<number>(), lanesNew = new Set<number>();
  const passOf: NetCache["passOf"] = new Map();
  const lineIds: string[] = [];
  const passValsSeen: NetCache["passVals"] = new Map();
  // T-4153: the lines (by the last finished build's rows) on an edge whose row or piece of track changed: only those and
  // lines whose rows changed look along their paths for what changed
  const touched = new Set<number>();
  if (diff) {
    const mark = (e: number) => { const v = cache.edgeLines.get(e); if (v) for (const id of v) touched.add(id); };
    for (const e of fresh) mark(e);
    for (const e of gone) mark(e);
    for (const e of rowsNew) mark(e);
  }
  /** the row each line's piece stands for now */
  const lineRows: NetCache["lineRows"] = new Map();
  // how many lines are on each edge, over their whole plans, as they are drawn (T-3474)
  // (T-4153: patched from the lines whose rows changed, with the stops each serves; `countsWas` the last finished build's)
  const countsWas = cache.counts?.counts;
  const counted = countsOf(cache.counts, s.lines);
  const onEdge = counted.counts;
  // stations whose passing tracks work: on double track (T-2233, sim `overtake::passing_at`)
  const passingAt = new Map<number, ClockState["nodes"][number]>();
  for (const n of s.nodes) if (n.platform > 0 && hasUpgrade(n.upgrades, UPGRADE.passing) && (nodeEnds.get(n.id)?.double ?? 0) > 0) passingAt.set(n.id, n);
  const innerAt = passingAt.size ? innerStops(s.lines, passingAt, edgeEnds, side) : null;
  lap("lines.setup");
  // T-2248: a passing station's way out along an edge, carried on through plain nodes (two edges) until
  // its outer track's lead fits, and the lead it has room for
  const passWays = new Map<number, { edges: number[]; lead: number }>();
  const passWay = (node: number, edge: number) => {
    const key = node * 2 ** 26 + edge;
    let w = passWays.get(key);
    if (w) return w;
    const half = (passingAt.get(node)?.platform ?? 0) / 2, want = half + PASS_LEAD_M, edges = [edge];
    let e = edgeEnds.get(edge), len = e?.lengthM ?? 0, far = e ? (e.a === node ? e.b : e.a) : -1;
    while (e && len < want && !passingAt.has(far)) {
      const from: number = e.id, next = (edgesAt.get(far) ?? []).filter((id) => id !== from);
      const ne = next.length === 1 ? edgeEnds.get(next[0]) : undefined;
      if (!ne) break;
      edges.push(ne.id);
      len += ne.lengthM;
      far = ne.a === far ? ne.b : ne.a;
      e = ne;
    }
    const o = far !== node ? passingAt.get(far) : undefined;
    w = { edges, lead: passLead(len, half, o ? o.platform / 2 : -1) };
    passWays.set(key, w);
    return w;
  };
  const leadOf = (node: number, edge: number) => (passingAt.has(node) ? passWay(node, edge).lead : PASS_LEAD_M);
  const passing: number[] = [];
  const nSamples = r.samples.length / 2;
  const samplePass = new Float32Array(passingAt.size ? nSamples : 0).fill(1e9);
  const sampleStroke = new Int32Array(nSamples).fill(-1);
  const sampleLane = new Float32Array(nSamples).fill(1);
  const sampleBand = new Float32Array(nSamples * 2);
  const sampleStretch = new Float32Array(Math.max(1, nSamples)).fill(-1);
  for (const l of s.lines) lineColours.set([...rgb(l.colour), 255], l.id * 4);
  lap("lines.arrays");
  // lines sharing track side by side (T-1799)
  // (T-4153: the edges whose row or points may differ from the last finished build's, which the bands' last call most
  // likely had, so the bands look at those alone)
  const changed = dirty && kept0 && prevSide ? { pts: kept0.edgePts, ends: prevSide.edgeEnds, edges: diff!.removed.length ? new Set([...dirty, ...diff!.removed.map((e) => e.id)]) : dirty } : undefined;
  const bands = lineBands(s.lines, s.edges, edgePts, { state: cache.bands, ends: edgeEnds, changed });
  lap("lines.bands");
  for (const l of lineDrawOrder(s.lines)) {
    if (!l.ok) continue;
    lineIds.push(String(l.id));
    let c = cache.lines.get(l.id);
    const band = bands.of.get(l.id) ?? null;
    // T-3474: the whole plan is drawn, faded where its trains do not run (an unbuilt extension, or beyond a gap)
    const planPath = l.planPath ?? l.path, planStops = l.planStops ?? l.stops;
    const runStart = l.pathStart ?? 0, runEnd = runStart + l.path.length, stopStart = l.stopStart ?? 0;
    // (T-4153: a line whose row is the one its piece was made for, over the same pieces of track, is as it was)
    const sameRow = cache.lineRows.get(l.id) === l;
    if (!c || (!sameRow && (c.running !== l.running || c.fast !== l.fast || c.tram !== l.tram || c.pathStart !== runStart || c.stopStart !== stopStart || !sameArray(c.path, planPath) || !sameArray(c.back, l.back ?? []) || !sameArray(c.stops, planStops))) || (!loopSame && !(diff && sameRow && !touched.has(l.id)) && (!planPath.every((p) => keptEdge(p >> 1)) || !(l.back ?? []).every((p) => keptEdge(p >> 1)))) || (c.band === null ? band !== null : band === null || (c.band !== band && c.bandOwn !== bands.own.get(l.id) && !sameBand(c.band, band)))) {
      netWork.lines++;
      const lines = new StrokeBuilder(true, MIN_SEG_M, false, true);
      // (room for about as many segments as the line's last strokes, rather than doubling up to them)
      if (c) lines.reserve(c.strokes.count + 64);
      let d = 0;
      // the first stroke segment of each path entry (in this line's strokes), and the stop-to-stop
      // stretch it is on (T-084), numbered as the part its trains run
      const firstStroke = planPath.map(() => -1);
      // T-4153: an entry as the line's last strokes drew it (the same piece of track, knots of its band and flags) is copied
      // from them, its stretch and distances along worked out again (`LineEntry.bits`)
      const old = c, bits = new Uint8Array(planPath.length);
      const was = old?.bits ? alikeEntries(old.path, planPath) : null, oldEnd = was ? strokeEnds(old!) : null;
      let stretch = 0, next = 1;
      planPath.forEach((p, k) => {
        const e = p >> 1;
        const dr = drawn.get(e);
        const ends = edgeEnds.get(e);
        if (dr) {
          const served = k >= runStart && k < runEnd;
          const flags = (l.running && served ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0))) | (l.fast && hsrEdge.has(e) ? F_HSR : 0) | (l.tram && tramEdge.has(e) ? F_TRAM : 0) | (splitEdge.has(e) ? F_SPLIT : 0);
          // (-1 off the part its trains run: no riders' stretch there)
          const stretchOf = served ? Math.max(0, Math.min(stretch - stopStart, Math.max(0, l.stops.length - 2))) : -1;
          if (dr.lpts.length >= 6) {
            firstStroke[k] = lines.count;
            const banded = !!band && served;
            // T-3158: a line alone on single track keeps a line's width; a ribbon beside it (a flyover's leg by
            // its main) only moves it sideways, as its return leg, which has no band, is drawn
            const zero = banded && ends?.tracks === 1 && (bands.order.get(e)?.length ?? 0) < 2;
            const how = (banded ? BITS_BANDED : 0) | (zero ? BITS_ZEROED : 0), ko = was ? was[k] : -1;
            const a0 = ko >= 0 ? old!.firstStroke[ko] : -1;
            if (a0 >= 0 && (old!.bits![ko] & ~BITS_POPPED) === how && keptEdge(e) && (old!.strokes.flags[a0] & ~F_EASE) === flags && (!banded || netTrust || sameKnots(old!.band!, ko - old!.pathStart, band!, k - runStart))) {
              d = lines.copy(old!.strokes, a0, oldEnd![ko], d, (old!.bits![ko] & BITS_POPPED) !== 0, stretchOf);
              bits[k] = old!.bits![ko];
              netWork.entries++;
              netWork.entriesAll++;
            } else {
              if (banded) {
                const bp = bandPoints(dr, (p & 1) === 1, band!, k - runStart);
                if (zero) for (let q = 1; q < bp.band.length; q += 3) bp.band[q] = 0;
                d = lines.add(bp.pts, l.id, e, flags, (p & 1) === 1, d, stretchOf, bp.lane, bp.band, bp.ease);
              } else d = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, d, stretchOf, dr.lane);
              bits[k] = how | (lines.popped ? BITS_POPPED : 0);
            }
          } else d = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, d, stretchOf, dr.lane);
        }
        // a path entry runs a -> b, or b -> a when its low bit is set
        const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
        if (to === planStops[next]) {
          stretch++;
          next++;
        }
      });
      const forwardCount = lines.count;
      // T-3117: track only the return run takes (a flyover's other leg), its stretch counted from the far end
      const backStroke = (l.back ?? []).map(() => -1);
      if (l.back?.length) {
        const own = new Map<number, number>();
        l.path.forEach((p, k) => { if (!own.has(p >> 1)) own.set(p >> 1, firstStroke[k + runStart]); });
        const last = Math.max(0, l.stops.length - 2);
        let bd = 0, back = 0, nextBack = l.stops.length - 2;
        for (const [j, p] of l.back.entries()) {
          const e = p >> 1, dr = drawn.get(e), ends = edgeEnds.get(e);
          if (own.has(e)) backStroke[j] = own.get(e)!;
          else if (dr) {
            const flags = (l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0))) | (l.tram && tramEdge.has(e) ? F_TRAM : 0) | (splitEdge.has(e) ? F_SPLIT : 0);
            if (dr.lpts.length >= 6) backStroke[j] = lines.count;
            bd = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, bd, Math.max(0, Math.min(last, last - back)), dr.lane);
          }
          const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
          if (nextBack >= 0 && to === l.stops[nextBack]) {
            back++;
            nextBack--;
          }
        }
      }
      c = { path: planPath.slice(), back: (l.back ?? []).slice(), stops: planStops.slice(), pathStart: runStart, stopStart, running: l.running, fast: l.fast, tram: l.tram, band, strokes: lines.keepLine(), colour: l.id, firstStroke, forwardCount, backStroke, bits, bandOwn: bands.own.get(l.id) };
    }
    linesSeen.set(l.id, c);
    lineRows.set(l.id, l);
    lineParts.push(c);
    const base = lineCount;
    lineCount += c.strokes.count;
    // each train sample's stroke segment (run 0 resampled every `step` m) and lane, per line (T-1029);
    // and a return run's own samples, where it takes its own way back (T-3250, `backSlot`)
    let fwdAt: Map<number, number> | null = null;
    const fwdOf = (e: number) => {
      if (!fwdAt) {
        fwdAt = new Map();
        l.path.forEach((p, k) => { if (!fwdAt!.has(p >> 1)) fwdAt!.set(p >> 1, k); });
      }
      return fwdAt.get(e);
    };
    /** T-4153: how many entries of the run's `path` from its start are as `sm`'s were made from: the same entry and piece of
     * track (so lengths and lanes), stroke and band knots */
    const alikeSamples = (sm: LineSamples, path: readonly number[]) => {
      const o = sm.c, n = Math.min(path.length, sm.path.length), band = c!.band;
      if (!o.bits || !band !== !o.band || sm.bus !== !!l.bus) return 0;
      let j = 0;
      while (j < n && path[j] === sm.path[j] && sameEdge(path[j] >> 1) && c!.firstStroke[j + runStart] === o.firstStroke[j + o.pathStart] && (!band || sameKnots(o.band!, j, band, j))) j++;
      return j;
    };
    // (T-4153: a line on no edge that changed, by the last build's rows, which are its own)
    const untouched = !!diff && sameRow && !touched.has(l.id);
    // (T-4285: a run's kept samples and passing places by its row key, line id times 2 plus 1 for a return run's own way
    // back, trainSlots.ts `rowKey`; its line table slot moves when a line is made or the top one goes)
    const runSamples = (slot: number, key: number, path: readonly number[], strokeOf: readonly number[], back: boolean) => {
      const first = r.lineTable[slot * 4], count = r.lineTable[slot * 4 + 1], step = r.lineTable[slot * 4 + 3];
      let sm = cache.samples.get(key);
      // (T-4153: the same path over the same edges has the same lengths)
      const kept = !!sm && sm.c === c && sm.count === count && sm.step === step && sm.path === path && (edgesSame || untouched || path.every((p) => sameEdge(p >> 1)));
      const lens = kept ? sm!.lens : path.map((p) => edgeLen.get(p >> 1) ?? 0);
      if (sm && !kept && sm.c === c && sm.count === count && sm.step === step && sameArray(sm.lens, lens)) sm = { ...sm, path };
      else if (!kept) {
        netWork.samples++;
        // T-4153: the samples on the entries before the first unlike those the last ones were made from are those
        const upTo = sm && !back && sm.step === step ? alikeSamples(sm, path) : 0, was = sm;
        // a return run's own samples take the path's band on an edge both use, turned round where it runs the other way
        sm = makeSamples(c!, path, lens, count, step, strokeOf, !!l.bus, drawn, back ? (j) => fwdOf(path[j] >> 1) ?? -1 : null, back ? (j, f) => (l.path[f] & 1) !== (path[j] & 1) : null, was, upTo);
        // T-4280: named for the renderer where the values it draws with differ (most made again are the same)
        const differ = was ? sampleValuesDiffer(was, sm) : 3;
        if (differ & 1) stretchesNew.add(l.id);
        if (differ & 2) lanesNew.add(l.id);
      }
      samplesSeen.set(key, sm!);
      writeSamples(sm!, first, base, c!.strokes.stretch!, sampleStroke, sampleStretch);
      sampleLane.set(sm!.lane, first);
      if (sm!.band.length) sampleBand.set(sm!.band, first * 2);
      let mine = NO_PASS;
      if (passingAt.size && !l.bus) {
        // (a return run's own stops go on a scratch list: the outer tracks are drawn from its path's)
        // (T-3348: the return run stops at its own platforms where they differ)
        mine = linePassing(back ? { ...l, path: path as number[], stops: (l.backStops ?? l.stops).slice().reverse() } : l, passingAt, edgeEnds, lens, innerAt?.get(l.id), leadOf, back ? [] : passing);
        if (mine.length) {
          // (T-4153: the same passing places along as many samples as far apart give the same values)
          const kp = cache.passVals.get(key);
          const pv = kp && kp.count === count && kp.step === step && sameArray(kp.mine, mine) ? kp : { count, step, mine, v: passValues(mine, count, step) };
          passValsSeen.set(key, pv);
          samplePass.set(pv.v, first);
        }
      }
      // (T-4280: the same passing places along the same samples give the same values)
      if (mine.length) passOf.set(key, mine);
      if (!sameArray(cache.passOf.get(key) ?? NO_PASS, mine)) lanesNew.add(l.id);
    };
    const ts0 = performance.now();
    runSamples(l.id, l.id * 2, l.path, l.planPath ? c.firstStroke.slice(runStart, runEnd) : c.firstStroke, false);
    const slot = backSlotOf(r.lineTable, r.meta, l.id);
    if (slot >= 0 && l.back?.length) runSamples(slot, l.id * 2 + 1, l.back, c.backStroke ?? [], true);
    netLaps["lines.samples"] = (netLaps["lines.samples"] ?? 0) + performance.now() - ts0;
  }
  lap("lines.loop");
  yield;
  lapT = performance.now();
  // the fastest train over each built segment (T-1271), from each line's speeds, kept per line; T-4153: a line is looked
  // at again only when its row, its timing or a piece of its path changed, and a piece's fastest train is folded
  // again only from the lines on a piece whose lines' speeds changed
  const speedsSeen: NetCache["speeds"] = new Map();
  /** pieces whose fastest train may differ from the kept one */
  const trainDirty = new Set<number>();
  const pathDirty = (path: readonly number[]) => { for (const p of path) trainDirty.add(p >> 1); };
  // (T-4153: a line's phase tables compared where they lie, copied out only for a line worked out again)
  const phaseBits = new Uint32Array(r.phases.buffer, r.phases.byteOffset, r.phases.length);
  for (const l of s.lines) {
    let sp = cache.speeds.get(l.id);
    if (sp && sp.row === l && sp.phases === r.phases && sameMetaOf(sp.meta, l.id, r) && (loopSame || (!!diff && !touched.has(l.id)) || sp.path.every((p) => sameEdge(p >> 1)) || sp.path.every((p, k) => { const e = p >> 1; return edgesSeen.get(e)?.strokes === sp!.strokes[k] && (builtEdge.get(e) === true) === sp!.built[k] && (edgeLen.get(e) ?? 0) === sp!.lens[k]; }))) {
      speedsSeen.set(l.id, sp);
      continue;
    }
    const runs = l.ok && l.running && Math.max(...l.trains) > 0;
    let same = !!sp && (sp.ph !== null) === runs && (!runs || samePhasesOf(sp.ph!, l.id, r, phaseBits)) && sameArray(sp.path, l.path);
    let lens: number[] | undefined, strokes: (TrackStrokes | undefined)[] | undefined, built: boolean[] | undefined;
    // (over the same pieces of track their lengths, strokes and building are as they were)
    if (same && !loopSame && !sp!.path.every((p) => sameEdge(p >> 1))) {
      lens = l.path.map((p) => edgeLen.get(p >> 1) ?? 0);
      strokes = l.path.map((p) => edgesSeen.get(p >> 1)?.strokes);
      built = l.path.map((p) => builtEdge.get(p >> 1) === true);
      same = sameArray(sp!.lens, lens) && strokes.every((x, k) => x === sp!.strokes[k]) && built.every((x, k) => x === sp!.built[k]);
    }
    const meta = metaOf(l.id, r);
    if (!same) {
      netWork.speeds++;
      if (sp) pathDirty(sp.path);
      lens ??= l.path.map((p) => edgeLen.get(p >> 1) ?? 0);
      strokes ??= l.path.map((p) => edgesSeen.get(p >> 1)?.strokes);
      built ??= l.path.map((p) => builtEdge.get(p >> 1) === true);
      const ph = runs ? speedPhases(l.id, r) : null;
      sp = { row: l, phases: r.phases, meta, path: l.path.slice(), lens, ph, strokes, built, out: ph ? lineTrainSpeeds(l, r, (e) => edgesSeen.get(e)?.strokes, lens, (e) => builtEdge.get(e) === true) : null };
      pathDirty(sp.path);
    } else sp = { ...sp!, row: l, phases: r.phases, meta };
    speedsSeen.set(l.id, sp);
  }
  for (const [id, sp] of cache.speeds) if (!speedsSeen.has(id)) pathDirty(sp.path);
  // (each dirty piece's fastest train folded from every line on it, in the lines' order, as a whole fold would)
  const trainNew = new Map<number, Float32Array | null>();
  if (trainDirty.size) {
    for (const e of trainDirty) trainNew.set(e, null);
    // (T-4153: only the lines on a dirty piece, by the last build's rows or their own new ones)
    const onDirty = new Set<number>();
    if (diff) for (const e of trainDirty) for (const id of cache.edgeLines.get(e) ?? []) onDirty.add(id);
    for (const l of s.lines) {
      if (diff && !onDirty.has(l.id) && cache.edgeRows.get(l.id) === l) continue;
      const out = speedsSeen.get(l.id)!.out;
      if (!out) continue;
      const { v, idx } = out;
      for (let k = 0; k < idx.length >> 1; k++) {
        const e = l.path[k] >> 1, o = idx[k * 2], n = idx[k * 2 + 1];
        if (n < 0 || !trainDirty.has(e)) continue;
        let t = trainNew.get(e);
        if (!t) trainNew.set(e, (t = new Float32Array(n).fill(-1)));
        for (let j = 0; j < n; j++) if (v[o + j] > t[j]) t[j] = v[o + j];
      }
    }
  }
  const trainOf = (id: number) => (trainNew.has(id) ? trainNew.get(id) ?? undefined : cache.trains.get(id));
  /** `c` with its fastest train as now, `c` itself when that is as it was */
  /** pieces whose fastest train changed, put in `edgesSeen` as the build finishes (only the track's pieces read them) */
  const trainPut = new Map<number, EdgeEntry>();
  const withTrain = (id: number, c: EdgeEntry) => {
    const t = trainOf(id);
    if (t ? sameFloats(t, c.train) : !c.train.some((v) => v !== -1)) return c;
    trainPut.set(id, (c = { ...c, train: t ?? NO_TRAIN }));
    return c;
  };
  let trackParts: EdgeEntry[];
  let trackCount = 0;
  const last0 = cache.last;
  if (loopSame && last0 && last0.track.length === trackOrder.length) {
    // (T-4153: the last build's pieces in their order, those whose fastest train changed replaced)
    trackParts = last0.track;
    trackCount = last0.trackCount;
    if (trainNew.size) {
      trackParts = trackParts.slice();
      for (let k = 0; k < trackOrder.length; k++) if (trainNew.has(trackOrder[k])) trackParts[k] = withTrain(trackOrder[k], trackParts[k]);
    }
  } else {
    trackParts = [];
    for (const id of trackOrder) {
      let c = edgesSeen.get(id)!;
      // (a piece kept as it was, whose lines' speeds are as they were, has its fastest train already)
      if (!keptEdge(id) || trainNew.has(id)) c = withTrain(id, c);
      trackParts.push(c);
      trackCount += c.strokes.count;
    }
  }
  lap("speeds");
  yield;
  lapT = performance.now();
  const served = counted.served;
  // whether a line runs through each node, so a station's box covers it (one line width since T-190)
  // (T-4153: patched at the nodes whose edges changed and the ends of edges whose count of lines did; `linedNew` the nodes
  // that changed, null when worked out whole)
  const kl = cache.lined;
  let lined: Set<number>, linedNew: Set<number> | null = new Set();
  if (kl && kl.side === side && kl.counts === onEdge) lined = kl.lined;
  else if (kl && counted.edges && kl.counts === countsWas && (kl.side === side || side.from === kl.side)) {
    const ln = (lined = new Set(kl.lined)), changed = linedNew!;
    const look = (n: number) => {
      const v = nodeEnds.get(n)?.ids, has = !!v && v.some((id) => !!onEdge.get(id));
      if (has === ln.has(n)) return;
      if (has) ln.add(n);
      else ln.delete(n);
      changed.add(n);
    };
    if (kl.side !== side) for (const n of side.touched!) look(n);
    if (fault !== "lined") for (const e of counted.edges) {
      const x = edgeEnds.get(e);
      if (x) { look(x.a); look(x.b); }
    }
  } else {
    lined = new Set<number>();
    linedNew = null;
    for (const e of s.edges) if (onEdge.get(e.id)) lined.add(e.a).add(e.b);
  }
  const st = nodesSame && cache.nodes!.st ? cache.nodes!.st : s.nodes.filter((n) => n.platform > 0);
  const stationIds = cache.nodes?.stationIds && cache.nodes.st === st ? cache.nodes.stationIds : st.map((n) => String(n.id));
  const stations = new Float32Array(st.length * 3);
  const stationAxis = new Float32Array(st.length * 2);
  const stationShape = new Float32Array(st.length * 4);
  const stationLine = new Float32Array(st.length * 2 * PLATFORM_POINTS);
  // Most lines first, then edge id: junction platform direction is stable across rebuilds. (T-4153: each node's own
  // edges sorted, the same order as one sort of every edge, without it)
  // (T-4153: kept while the edges are, sorted again only at the ends of edges whose count of lines changed)
  const linesAt = (id: number) => onEdge.get(id) ?? 0;
  const wayBefore = (a: number, b: number) => linesAt(b) - linesAt(a) || a - b;
  // (the node's own list when it is in that order already, as most are)
  const sortWays = (v: number[]) => {
    for (let k = 1; k < v.length; k++) if (wayBefore(v[k - 1], v[k]) > 0) return v.slice().sort(wayBefore);
    return v;
  };
  const kw = cache.ways;
  let waysAt: Map<number, number[]>;
  /** T-4153: the nodes whose ways may differ from the last finished build's (null: any may) */
  let waysNew: Set<number> | null = new Set();
  if (kw && kw.side === side && kw.counts === onEdge) waysAt = kw.ways;
  else if (kw && (kw.side === side || side.from === kw.side)) {
    const wa = (waysAt = new Map(kw.ways)), changed = waysNew!;
    const resortAt = (n: number) => {
      const v = nodeEnds.get(n)?.ids;
      if (v) wa.set(n, sortWays(v));
      else wa.delete(n);
      if (wa.get(n) !== kw.ways.get(n)) changed.add(n);
    };
    const resort = (e: number) => { const x = edgeEnds.get(e); if (x) { resortAt(x.a); resortAt(x.b); } };
    if (kw.side !== side) for (const n of side.touched!) resortAt(n);
    if (kw.counts !== onEdge) {
      // (T-4153: the edges whose count changed, when the counts were patched from those `kw` was sorted by)
      if (counted.edges && kw.counts === countsWas) for (const e of counted.edges) resort(e);
      else {
        for (const [e, v] of onEdge) if (kw.counts.get(e) !== v) resort(e);
        for (const e of kw.counts.keys()) if (!onEdge.has(e)) resort(e);
      }
    }
  } else {
    waysNew = null;
    waysAt = new Map();
    for (const [n, v] of nodeEnds) waysAt.set(n, sortWays(v.ids));
  }
  const platformEdge = { get: (node: number) => waysAt.get(node)?.[0] };
  lap("stations.ways");
  /** the track leaving `node` along edge `id`, starting at the node */
  const way = (node: number, id: number) => {
    const p = edgePts.get(id), e = edgeEnds.get(id);
    return !p || !e ? new Float32Array(0) : e.a === node ? p : reversed(p);
  };
  // (the lines' end stops alone are looked up)
  const ends = new Set<number>();
  for (const l of s.lines) if (l.stops.length) ends.add(l.stops[0]).add(l.stops[l.stops.length - 1]);
  const nodeById = new Map<number, ClockState["nodes"][number]>();
  for (const n of s.nodes) if (ends.has(n.id)) nodeById.set(n.id, n);
  const turnbacksSeen: NetCache["turnbacks"] = new Map();
  const passSeen: NetCache["passTracks"] = new Map();
  const turnStrokes = turnbackStrokes(s, linesSeen, endCount, (push) => {
    // a passing station's ways, carried on as far as `passWay` goes
    const waysOut = (n: number) => (waysAt.get(n) ?? []).map((id): [number, Float32Array] => {
      const parts: Float32Array[] = [];
      let at = n;
      for (const e of passWay(n, id).edges) {
        const p = way(at, e), ends = edgeEnds.get(e);
        parts.push(parts.length ? p.subarray(3) : p);
        if (ends) at = ends.a === at ? ends.b : ends.a;
      }
      const out = new Float32Array(parts.reduce((m, p) => m + p.length, 0));
      parts.reduce((o, p) => (out.set(p, o), o + p.length), 0);
      return [id, out];
    });
    // (T-4153: each station's kept while its row, its ways, their pieces' points and its leads are)
    for (const n of passingAt.values()) {
      const ways = waysAt.get(n.id) ?? [], key: unknown[] = [n, ways];
      for (const id of ways) { const w = passWay(n.id, id); key.push(w.lead); for (const e of w.edges) key.push(edgePts.get(e), edgeEnds.get(e)); }
      const k = cache.passTracks.get(n.id);
      let calls = k && sameRefs(k.key, key) ? k.calls : null;
      if (!calls) {
        const log = new StrokeLog();
        passingTracks(log as unknown as StrokeBuilder, [n], waysOut, leadOf);
        calls = log.calls;
      }
      passSeen.set(n.id, { key, calls });
      push(calls);
    }
    // (T-4153: made once while the edges stay)
    if (!side.leads) {
      const log = new StrokeLog();
      singleLeadStrokes(log as unknown as StrokeBuilder, s.edges, way, edgeEnds, (n) => nodeEnds.get(n)?.ids.map((id) => edgeEnds.get(id)!) ?? []);
      // (the last side's own list where alike, so the turn-backs' strokes are known kept)
      side.leads = prevSide?.leads && sameCalls(prevSide.leads, log.calls) ? prevSide.leads : log.calls;
    }
    push(side.leads);
  }, nodeById, cache.turnbacks, turnbacksSeen, cache.turnStrokes);
  const turnbacks = turnStrokes.strokes;
  lap("stations.turnbacks");
  // The station dot sits on the track: its node position, from the local polyline of an edge
  // that ends there (exact), else converted from lng/lat by the label code.
  // (T-4153: the last edge in order to end at a node gives its place, as when each wrote over the one before; kept as
  // numbers, a pair made only when asked for)
  const nodeLocal = {
    get: (id: number): readonly [number, number] | undefined => {
      const a = nodeEnds.get(id);
      if (!a) return undefined;
      if (!a.xy) {
        const e = a.last, p = edgePts.get(e.id)!, k = e.b === id ? p.length - 3 : 0;
        a.xy = [p[k], p[k + 1]];
      }
      return a.xy;
    },
  };
  const edgesOf = (node: number) => (waysAt.get(node) ?? []).map((id) => edgeEnds.get(id)!);
  // T-4153: the stations whose numbers may differ from the last finished build's (`NetCache.stArr`): those at a node whose
  // row, edges, ways, lines through it or stops changed, at an end of a piece of track made again, whose split changed,
  // and single-track ones whose way out to a join read an edge or node that changed; null: every one
  const ka = cache.stArr;
  let stDirty: Set<number> | null = null;
  if (ka && nodeDiff && diff && kept0 && waysNew && linedNew && counted.stops && ka.rh === s.rightHand && (side === prevSide || side.touched)) {
    const d = (stDirty = new Set(nodeDiff));
    const endsOf = (id: number) => { const x = edgeEnds.get(id); if (x && !stationTrust) d.add(x.a).add(x.b); };
    if (side !== prevSide) for (const n of side.touched!) d.add(n);
    for (const id of fresh) endsOf(id);
    for (const id of rowsNew) endsOf(id);
    for (const e of diff.removed) d.add(e.a).add(e.b);
    for (const n of waysNew) d.add(n);
    if (fault !== "st-lined") for (const n of linedNew) d.add(n);
    if (!stationTrust) for (const n of counted.stops) d.add(n);
    if (splits !== kept0.splits) {
      for (const n of splits.stations) if (!kept0.splits.stations.has(n)) d.add(n);
      for (const n of kept0.splits.stations) if (!splits.stations.has(n)) d.add(n);
    }
    const rowMoved = (e: number) => rowsNew.has(e) || gone.has(e);
    const atMoved = (n: number) => waysNew!.has(n) || (side !== prevSide && side.touched!.has(n));
    for (const [id, k] of cache.outside) if (!d.has(id) && (k.edges.some(rowMoved) || k.nodes.some(atMoved))) d.add(id);
  }
  const stIndex = ka && ka.st === st ? ka.index : new Map(st.map((n, i) => [n.id, i]));
  const stationsSeen: NetCache["stations"] = stDirty ? new Map(cache.stations) : new Map();
  const outsideSeen: NetCache["outside"] = stDirty ? new Map(cache.outside) : new Map();
  const oneStation = (n: NodeRow, i: number) => {
    const p = nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat);
    const ids = waysAt.get(n.id) ?? [];
    // (the platform's edge is the first of its ways)
    const first = 0;
    let c = cache.stations.get(n.id);
    if (!c || n.brtStation || c.platform !== n.platform || c.lat !== n.lat || c.first !== first || (c.ways !== ids && !sameArray(c.ways, ids)) || !ids.every(keptEdge)) {
      netWork.stations++;
      // the platform along its track, curved as it is (T-189); local units are metres / cos(lat)
      const perM = 1 / Math.cos(n.lat * Math.PI / 180);
      const sh: PlatformShape = ids.length
        ? platformShape(ids.map((id) => way(n.id, id)), first, n.platform / 2 * perM)
        // T-3378: a BRT station along its road (`heading` counter-clockwise from east; local y runs south)
        : { mx: 0, my: 0, ax: Math.cos(n.brtStation ? n.heading : 0) * n.platform / 2 * perM, ay: -Math.sin(n.brtStation ? n.heading : 0) * n.platform / 2 * perM, phi: 0 };
      c = { platform: n.platform, lat: n.lat, first, ways: ids, ax: sh.ax, ay: sh.ay, shape: [sh.mx, sh.my, sh.phi, PLATFORM_HALF_M * perM], line: platformPoints(sh) };
    }
    // T-3332: a one-track station on double track: the half of the box on its track's side
    const oneSide = n.one === 1 && n.oneHeading !== undefined ? oneSideAlong(n.oneHeading, s.rightHand === false, c.ax, c.ay) : 0;
    // T-3540: a station on single track with no passing loop has one platform, on its outside; one whose track
    // splits round a loop keeps a platform each side, each way's trains stopping on their own track (T-3564)
    const pe = edgeEnds.get(platformEdge.get(n.id) ?? -1);
    const split = splits.stations.has(n.id);
    const single = !oneSide && !split && !!pe && ids.every((id) => edgeEnds.get(id)?.tracks === 1 && !edgeEnds.get(id)?.crossover);
    // (T-4153: kept while the edges, the stations' ways and its own piece are)
    let out = 0;
    if (!oneSide && single) {
      const k = cache.outside.get(n.id), same = !!k && k.side === side && k.ways === waysAt && k.c === c && k.rh === s.rightHand;
      if (same) {
        out = k!.v;
        outsideSeen.set(n.id, k!);
      } else {
        // (T-4153: with the edges and nodes its way out read, so an edit that changes one marks it)
        const edgesRead: number[] = [], nodesRead: number[] = [];
        const at = (m: number) => {
          nodesRead.push(m);
          const v = edgesOf(m);
          for (const e of v) edgesRead.push(e.id);
          return v;
        };
        out = stationOutside(n.id, pe, at, way, c.ax, c.ay, s.rightHand === false);
        outsideSeen.set(n.id, { side, ways: waysAt, c, rh: s.rightHand, v: out, edges: edgesRead, nodes: nodesRead });
      }
    } else if (stDirty) outsideSeen.delete(n.id);
    const sideOf = oneSide || out;
    stationsSeen.set(n.id, c);
    stationAxis[i * 2] = c.ax;
    stationAxis[i * 2 + 1] = c.ay;
    stationShape.set(c.shape, i * 4);
    stationLine.set(c.line, i * 2 * PLATFORM_POINTS);
    stations.set([p[0], p[1], ((served.get(n.id) ?? 0) > 1 ? 1 : 0) | (n.stationBuilt ? 0 : 2) | (lined.has(n.id) ? 4 : 0) | (n.stationOff ? 8 : 0) | ((pe?.tracks === 2 && !oneSide) || split ? 16 : 0) | (sideOf ? ST_ONE | (sideOf < 0 ? ST_ONE_NEG : 0) : 0) | (single ? ST_SINGLE | ST_ONE_MID : 0) | (n.kind || n.brtStation ? ST_TRAM : 0) | (hasUpgrade(n.upgrades, UPGRADE.waitingRoom) ? ST_ROOM : 0) | (hasUpgrade(n.upgrades, UPGRADE.parkRide) ? ST_PARK : 0) | (hasUpgrade(n.upgrades, UPGRADE.terminal) ? ST_TERMINAL | (ids.length === 1 ? ST_BUFFER : 0) : 0) | (passingAt.has(n.id) ? ST_PASSING : 0)], i * 3);
  };
  if (!stDirty) st.forEach(oneStation);
  else {
    const was = ka!, PP = 2 * PLATFORM_POINTS;
    if (was.st === st) {
      stations.set(was.stations);
      stationAxis.set(was.axis);
      stationShape.set(was.shape);
      stationLine.set(was.line);
      for (const id of stDirty) { const i = stIndex.get(id); if (i !== undefined) oneStation(st[i], i); }
    } else {
      // (a station came or went: the others copied from where they were, a run of them at a time)
      const copyRun = (i0: number, j0: number, n: number) => {
        if (!n) return;
        stations.set(was.stations.subarray(j0 * 3, (j0 + n) * 3), i0 * 3);
        stationAxis.set(was.axis.subarray(j0 * 2, (j0 + n) * 2), i0 * 2);
        stationShape.set(was.shape.subarray(j0 * 4, (j0 + n) * 4), i0 * 4);
        stationLine.set(was.line.subarray(j0 * PP, (j0 + n) * PP), i0 * PP);
      };
      let ri = 0, rj = 0, rn = 0;
      for (let i = 0; i < st.length; i++) {
        const n = st[i], j = was.index.get(n.id);
        if (j === undefined || stDirty.has(n.id)) { copyRun(ri, rj, rn); rn = 0; oneStation(n, i); continue; }
        if (rn && i === ri + rn && j === rj + rn) { rn++; continue; }
        copyRun(ri, rj, rn);
        (ri = i), (rj = j), (rn = 1);
      }
      copyRun(ri, rj, rn);
      for (const id of stDirty) if (!stIndex.has(id)) { stationsSeen.delete(id); outsideSeen.delete(id); }
    }
  }
  // (as the loop left them, before the buildings and facilities mark them; copies, as the post can hand the arrays on)
  const stArr: NonNullable<NetCache["stArr"]> = { st, index: stIndex, rh: s.rightHand, stations: stations.slice(), axis: stationAxis.slice(), shape: stationShape.slice(), line: stationLine.slice() };
  lap("stations.loop");
  yield;
  lapT = performance.now();
  // T-2219: the station buildings' members as drawn (their boxes join the floor), which car parks stay out of
  const outlines = buildingsDrawn(s.buildings, s.buildingLL, st, stations, stationLine, edgePts, false);
  lap("rest.buildings1");
  for (let i = 0; i < st.length; i++) if (outlines.memberOf[i] >= 0) stations[i * 3 + 2] += ST_BUILDING;
  // T-1528: one waiting room and one car park per complex
  // (T-3803: kept per complex between the clock worker's builds)
  const lots = placeFacilities(st, stations, stationLine, stationShape, edgePts, waysAt, s.complexOf, outlines.draw, cache.facilities, changed);
  lap("rest.facilities");
  // their insides, which leave room for the waiting rooms just placed
  const bld = outlines.draw.length ? buildingsDrawn(s.buildings, s.buildingLL, st, stations, stationLine, edgePts, true, cache.bldBoxes, changed) : outlines;
  lap("rest.buildings2");
  // the player's links as bridges between the two platforms' nearest points (T-217)
  const stationIndex = stIndex;
  const shapeAt = (i: number) => platformShapeAt({ stationShape, stationAxis, stationLine }, i);
  const linkIdx: number[] = [], linkEnds: number[] = [];
  // none between two stations of one building: its floor joins them (T-2219)
  const oneFloor = (i: number, j: number) => bld.memberOf[i] >= 0 && bld.memberOf[i] === bld.memberOf[j];
  for (let k = 0; k + 1 < (s.links?.length ?? 0); k += 2) {
    const i = stationIndex.get(s.links[k]), j = stationIndex.get(s.links[k + 1]);
    if (i === undefined || j === undefined || oneFloor(i, j)) continue;
    linkIdx.push(i, j);
    linkEnds.push(...bridgeEnds(stations[i * 3], stations[i * 3 + 1], shapeAt(i), stations[j * 3], stations[j * 3 + 1], shapeAt(j)));
  }
  if (s.complexOf) {
    const auto = complexPassages(st, stations, stationLine, s.complexOf, linkIdx, st.map((n) => stationsSeen.get(n.id)!), cache.passages, s.apart);
    for (let k = 0; k + 1 < auto.length; k += 2) {
      const i = auto[k], j = auto[k + 1];
      // T-4079: a BRT station's join with no passage is the bus layer's dashed line (`brtJoins`)
      if (oneFloor(i, j) || st[i].brtStation || st[j].brtStation) continue;
      linkIdx.push(i, j);
      linkEnds.push(...bridgeEnds(stations[i * 3], stations[i * 3 + 1], shapeAt(i), stations[j * 3], stations[j * 3 + 1], shapeAt(j)));
    }
  }
  lap("rest.links");
  // T-2991: a one-way loop's chevrons, on the edges it runs once
  const oneWay: number[] = [];
  for (const l of s.lines) {
    const c = linesSeen.get(l.id);
    if (!c || l.bus || !l.oneWay || !l.loopKind) continue;
    const uses = new Map<number, number>();
    for (const p of l.path) uses.set(p >> 1, (uses.get(p >> 1) ?? 0) + 1);
    const at = l.stops.map((id) => nodeLocal.get(id)).filter((p) => p !== undefined);
    trainChevrons(c.strokes, l.id, uses, at, l.tram ? 0.75 : 1, F_BLUEPRINT, oneWay);
  }
  lap("rest.oneWayLoops");
  const jn = nodesSame && cache.nodes!.jn ? cache.nodes!.jn : s.nodes.filter((n) => n.ports >= 3 && n.platform === 0);
  const junctions = new Float32Array(jn.length * 2);
  jn.forEach((n, i) => junctions.set(nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat), i * 2));
  // the marks (T-629): a junction where moves cross at one level (T-3103) is built once three of
  // its tracks are; a crossing once both of its are
  jn.forEach((n, i) => {
    if (n.crosses) marks.push(junctions[i * 2], junctions[i * 2 + 1], M_JUNCTION, 0, n.builtPorts >= 3 ? 0 : 1, n.level);
  });
  // T-3164: one-way track no train can use, at any node but a station; pointing in wins where both
  for (const n of s.nodes) {
    if (!n.unusable || n.platform > 0) continue;
    const p = nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat);
    marks.push(p[0], p[1], n.unusable & 1 ? M_CONVERGE : M_DIVERGE, 0, n.builtPorts >= n.ports ? 0 : 1, n.level);
  }
  const cx = s.crossings;
  if (cx) for (let k = 0; k + 4 < cx.length; k += 5) marks.push(cx[k], cx[k + 1], cx[k + 2] === 1 ? M_BAD : M_CROSSING, 0, cx[k + 4] === 1 ? 0 : 1, cx[k + 3]);
  // T-3562: a loop at the first piece a line's run goes over twice in the same direction, at the piece's middle
  for (const l of s.lines) {
    if (l.bus || l.twinOf !== undefined) continue;
    const id = repeatedEdge(l), pts = id === null ? undefined : edgePts.get(id);
    if (!pts || pts.length < 6) continue;
    const k = Math.floor((pts.length / 3 - 1) / 2) * 3, f = ((pts.length / 3 - 1) % 2) / 2;
    const z = pts[k + 2] + f * (pts[k + 5] - pts[k + 2]);
    marks.push(pts[k] + f * (pts[k + 3] - pts[k]), pts[k + 1] + f * (pts[k + 4] - pts[k + 1]), M_REPEAT, 0, builtEdge.get(id!) === false ? 1 : 0, Math.max(-3, Math.min(3, Math.round(z / 8))));
  }
  lap("rest.marks");
  // T-3094: chevrons along one-way single track, clear of stations
  const oneWayTrack: number[] = [];
  if (s.edges.some((e) => e.oneWay)) {
    // T-3117: clear of junctions too, where two legs leaving one node would put their arrows side by side;
    // T-3189: and of every mark (ramps, flat crossings and junctions, unusable track), which they hid
    const clear: (readonly [number, number])[] = s.nodes.filter((n) => n.platform > 0 || n.ports >= 3).map((n) => nodeLocal.get(n.id)).filter((p) => p !== undefined);
    const box = [Infinity, Infinity, -Infinity, -Infinity];
    for (const e of s.edges) {
      const c = e.oneWay ? edgesSeen.get(e.id) : undefined;
      if (!c) continue;
      for (let k = 0; k < c.pts.length; k += 3) {
        box[0] = Math.min(box[0], c.pts[k]); box[1] = Math.min(box[1], c.pts[k + 1]);
        box[2] = Math.max(box[2], c.pts[k]); box[3] = Math.max(box[3], c.pts[k + 1]);
      }
    }
    const pad = 50 * kAtY(box[1]);
    for (let k = 0; k < marks.length; k += MARK_STRIDE) {
      const x = marks[k], y = marks[k + 1];
      if (marks[k + 2] === M_RAMPEND) continue;
      if (x >= box[0] - pad && x <= box[2] + pad && y >= box[1] - pad && y <= box[3] + pad) clear.push([x, y]);
    }
    const runs: OneWayEdge[] = [];
    for (const e of s.edges) {
      const c = e.oneWay ? edgesSeen.get(e.id) : undefined;
      if (c) runs.push({ a: e.a, b: e.b, way: e.oneWay as 1 | 2, strokes: c.strokes });
    }
    oneWayTrackRuns(runs, clear, oneWayTrack);
  }
  lap("rest.oneWayTrack");
  // per line its path, and per return run on its own way back (T-3250) that way, at its `backSlot`
  // (with its row key, which its samples are kept by)
  const runPaths: [number, number[], number][] = [];
  for (const l of s.lines) {
    if (!l.ok) continue;
    runPaths.push([l.id, l.path, l.id * 2]);
    const slot = backSlotOf(r.lineTable, r.meta, l.id);
    if (slot >= 0 && l.back?.length) runPaths.push([slot, l.back, l.id * 2 + 1]);
  }
  const pathTable = new Int32Array(Math.max(1, maxLine + 1, r.lineTable.length / 4) * 2);
  let pathN = 0;
  for (const [, path] of runPaths) pathN += path.length;
  const pathEntry = new Int32Array(pathN), pathFrom = new Float32Array(pathN), pathLen = new Float32Array(pathN);
  pathN = 0;
  for (const [id, path, key] of runPaths) {
    pathTable[id * 2] = pathN;
    pathTable[id * 2 + 1] = path.length;
    // (its samples' lengths of the same path, when it has them)
    const sm = samplesSeen.get(key), lens = sm?.path === path ? sm.lens : null;
    let d = 0;
    for (let k = 0; k < path.length; k++) {
      const p = path[k], len = lens ? lens[k] : edgeLen.get(p >> 1) ?? 0;
      pathEntry[pathN] = p;
      pathFrom[pathN] = d;
      pathLen[pathN++] = len;
      d += len;
    }
  }
  lap("rest.pathTable");
  // (the cache's own map changed in place when no piece but a fastest train did: the build is finished)
  for (const [id, c] of trainPut) edgesSeen.set(id, c);
  cache.edges = edgesSeen;
  cache.lines = linesSeen;
  cache.samples = samplesSeen;
  cache.speeds = speedsSeen;
  for (const [e, t] of trainNew) if (t) cache.trains.set(e, t); else cache.trains.delete(e);
  cache.stations = stationsSeen;
  cache.lineRows = lineRows;
  cache.side = side;
  side.input = { rows: s.edges, parts, speedParts, idx: diff?.idx ?? edgeIndex(s.edges) };
  cache.nodes = { rows: s.nodes, idx: nodesNow.idx, st, jn, stationIds };
  cache.nodeSlots = nodesNow.slots;
  // (a patched split state is kept even when no piece changed: its read lists are of the walks as just made)
  cache.loop = loopSame ? (splitState === kept0!.split ? kept0 : { ...kept0!, splits, split: splitState }) : { side, splits, split: splitState, edgePts, splitEdge, trackOrder, ramps };
  cache.counts = { rows: s.lines, counts: onEdge, served };
  cache.stArr = stArr;
  cache.ways = { side, counts: onEdge, ways: waysAt };
  cache.lined = { side, counts: onEdge, lined };
  cache.turnbacks = turnbacksSeen;
  cache.turnStrokes = turnStrokes;
  cache.outside = outsideSeen;
  cache.passTracks = passSeen;
  cache.passOf = passOf;
  cache.passVals = passValsSeen;
  // T-4153: each edge's lines, patched where a line's row changed
  const el = diff ? cache.edgeLines : new Map<number, Set<number>>(), er = diff ? cache.edgeRows : new Map<number, ClockState["lines"][number]>();
  const rowEdges = (l: ClockState["lines"][number]) => {
    const out = new Set<number>();
    for (const p of l.planPath ?? l.path) out.add(p >> 1);
    for (const p of l.path) out.add(p >> 1);
    for (const p of l.back ?? []) out.add(p >> 1);
    return out;
  };
  const unput = (id: number, row: ClockState["lines"][number]) => { for (const e of rowEdges(row)) { const v = el.get(e); if (v) { v.delete(id); if (!v.size) el.delete(e); } } };
  const put = (id: number, row: ClockState["lines"][number]) => { for (const e of rowEdges(row)) { let v = el.get(e); if (!v) el.set(e, (v = new Set())); v.add(id); } };
  for (const l of s.lines) {
    const was = er.get(l.id);
    if (was === l) continue;
    if (was) unput(l.id, was);
    put(l.id, l);
    er.set(l.id, l);
  }
  if (er.size > s.lines.length) {
    const now = new Set(s.lines.map((l) => l.id));
    for (const [id, row] of [...er]) if (!now.has(id)) { unput(id, row); er.delete(id); }
  }
  cache.edgeLines = el;
  cache.edgeRows = er;
  const serial = ++cache.serial, last = cache.last;
  const patch: NetPatch | undefined = last ? {
    base: last.serial,
    track: strokeRuns(trackParts, (p) => p.strokes.count, last.track),
    lines: strokeRuns(lineParts, (p) => p.strokes.count, last.lines),
    oldTrack: last.trackCount, oldLines: last.lineCount, same: [], edgeIds: Uint32Array.from(trackOrder), modes: { track: "whole", lines: "whole" }, ranges: { track: new Int32Array(0), lines: new Int32Array(0) },
    stretches: [...stretchesNew],
    lanes: [...lanesNew],
  } : undefined;
  cache.last = { serial, track: trackParts, lines: lineParts, trackCount, lineCount, pts: edgePts };
  return {
    track: deferStrokes ? { count: trackCount } as Strokes : strokesOfPieces("track", trackParts),
    lines: deferStrokes ? { count: lineCount } as Strokes : strokesOfPieces("lines", lineParts),
    ...(deferStrokes ? { strokePieces: { track: trackParts, lines: lineParts } } : {}),
    lineIds,
    lineColours,
    stations,
    stationAxis,
    stationIds,
    stationCount: st.length,
    stationLength: new Float32Array(st.map((n) => n.platform)),
    stationLinkIdx: new Int32Array(linkIdx),
    stationLinks: new Float32Array(linkEnds),
    lots,
    ...(bld.draw.length ? { buildings: bld.draw } : {}),
    junctions,
    junctionIds: jn.map((n) => n.id),
    marks: new Float32Array(marks),
    phases: r.phases,
    meta: r.meta,
    samples: r.samples,
    lineTable: r.lineTable,
    ...(r.sampleBend ? { sampleBend: r.sampleBend, bends: r.bends } : {}),
    lineCars,
    stationShape,
    stationLine,
    sampleStroke,
    sampleLane,
    sampleBand,
    corridor: bands.corridor,
    sampleStretch,
    pathTable,
    pathEntry,
    pathFrom,
    pathLen,
    conflicts: s.conflicts,
    singles: s.singles,
    keep: s.rightHand === false ? -1 : 1,
    turnbacks,
    ...(oneWay.length ? { oneWay: new Float32Array(oneWay) } : {}),
    ...(oneWayTrack.length ? { oneWayTrack: new Float32Array(oneWayTrack) } : {}),
    ...(passingAt.size ? { samplePass, passing: new Float32Array(passing) } : {}),
    edgePts,
    anchor: anchor(),
    serial,
    ...(patch ? { patch } : {}),
  };
}

// ---------------------------------------------------------------- trains on the CPU

/** The phase of a profile's `count` phases from `first` that trip time `tau` is in: the last one starting by then, else the first. */
function phaseAt(n: NetworkBuffers, first: number, count: number, tau: number): number {
  let lo = first, hi = first + count - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (n.phases[mid * 4] <= tau) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** Offset along a profile's run at trip time `tau`: the shader's arithmetic in float64. */
export function profileS(n: NetworkBuffers, profile: number, tau: number): number {
  const first = n.meta[profile * 4], count = n.meta[profile * 4 + 1];
  if (!count) return 0;
  const lo = phaseAt(n, first, count, tau);
  const t = n.phases[lo * 4], s = n.phases[lo * 4 + 1], v = n.phases[lo * 4 + 2], a = n.phases[lo * 4 + 3];
  let tt = Math.max(0, tau - t);
  if (a < 0) tt = Math.min(tt, v / -a);
  return s + v * tt + 0.5 * a * tt * tt;
}

/** T-4243: a profile's speed at trip time `tau`, m/s, as `profileS` moves it: none before it leaves or once at its last stop. */
export function profileV(n: NetworkBuffers, profile: number, tau: number): number {
  const first = n.meta[profile * 4], count = n.meta[profile * 4 + 1];
  if (!count || tau < 0 || tau >= arriveS(n, profile)) return 0;
  const k = phaseAt(n, first, count, tau);
  const v = n.phases[k * 4 + 2], a = n.phases[k * 4 + 3];
  return Math.max(0, v + a * Math.max(0, tau - n.phases[k * 4]));
}

/** T-4243: the fastest a profile's train runs, m/s: the top of each phase, at its start or where the next one starts. */
export function profileTop(n: NetworkBuffers, profile: number): number {
  const first = n.meta[profile * 4], count = n.meta[profile * 4 + 1], ph = n.phases;
  let top = 0;
  for (let k = first; k < first + count; k++) {
    const v = ph[k * 4 + 2], a = ph[k * 4 + 3];
    top = Math.max(top, v, k + 1 < first + count ? v + a * (ph[k * 4 + 4] - ph[k * 4]) : v);
  }
  return top;
}

/** T-2980: seconds into a profile's trip when it reaches its last stop (its turnaround is stood after
 * it, inside the trip's seconds): its last phase's stop, as the train shader's `arriveAt`. */
export function arriveS(n: NetworkBuffers, profile: number): number {
  const first = n.meta[profile * 4], count = n.meta[profile * 4 + 1];
  if (!count) return n.meta[profile * 4 + 2];
  const k = (first + count - 1) * 4, t = n.phases[k], v = n.phases[k + 2], a = n.phases[k + 3];
  return a < 0 ? t + v / -a : t;
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
  const [bx, by] = bendAt(n, si, f);
  return [n.samples[si * 2] + (n.samples[si * 2 + 2] - n.samples[si * 2]) * f + bx, n.samples[si * 2 + 1] + (n.samples[si * 2 + 3] - n.samples[si * 2 + 1]) * f + by];
}

/** T-4075: how far the track at `f` (0-1) between sample `si` and the next lies off the straight between them, local
 * units: between its bend points, the interval cut in equal parts (the train shader's `bendAt`). */
export function bendAt(n: Pick<NetworkBuffers, "sampleBend" | "bends">, si: number, f: number): [number, number] {
  const code = n.sampleBend?.[si] ?? 0, b = n.bends;
  if (!code || !b) return [0, 0];
  const m = code % 32, start = (code - m) / 32;
  const u = f * (m + 1), j = Math.min(Math.floor(u), m), g = u - j;
  const ax = j > 0 ? b[(start + j - 1) * 2] : 0, ay = j > 0 ? b[(start + j - 1) * 2 + 1] : 0;
  const cx = j < m ? b[(start + j) * 2] : 0, cy = j < m ? b[(start + j) * 2 + 1] : 0;
  return [ax + (cx - ax) * g, ay + (cy - ay) * g];
}

/** `lineAt`, carried on straight past either end (the train shader's `lineAtX`, T-2980). */
function lineAtX(n: NetworkBuffers, line: number, d: number): [number, number] {
  const len = n.lineTable[line * 4 + 2], k = Math.min(20, len);
  if (k > 1e-3 && (d > len || d < 0)) {
    const at = d > len ? len : 0, [ax, ay] = lineAt(n, line, at), [bx, by] = lineAt(n, line, d > len ? len - k : k);
    const f = (d > len ? d - len : -d) / k;
    return [ax + (ax - bx) * f, ay + (ay - by) * f];
  }
  return lineAt(n, line, d);
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
 * run 0's direction there. `shiftM`: metres to the side of its direction of travel it keeps to (`keep`) on double
 * track, where the renderer draws it on its own track (T-658, `NetworkRenderer.laneShiftM`); `bandM`:
 * metres a line width is drawn, where its line's band shifts it (T-1799, `NetworkRenderer.bandShiftM`),
 * by `band` per sample (the Traffic view's own, `NetworkRenderer.sampleBand`), and `bandOn` how far its
 * metres to the ribbon's middle are taken (`NetworkRenderer.bandOn`, T-2132). `passM`: metres out to a
 * passing station's outer track, where its trains stop (T-2233, `NetworkRenderer.passShiftM`). */
export function tripAt(n: NetworkBuffers, profile: number, dep: number, t: number, end = -Infinity, shiftM = 0, bandM = 0, band = n.sampleBand, bandOn = 0, passM = 0, bay?: ArrayLike<number>, bayAt = 0): { x: number; y: number; s: number; run: number; line: number; geo: number; gr: number; dx: number; dy: number; shiftM: number; bandM: number; band?: Float32Array; bandOn: number; passM: number; ex: number; lo: number; hi: number; across: (x: number) => number } | null {
  const tau = t - dep;
  const tripS = n.meta[profile * 4 + 2];
  // `end`: drawn until then, standing at the terminal for its next departure (T-206)
  if (tau < 0 || (tau > tripS && t > end) || !n.meta[profile * 4 + 1]) return null;
  const line = profileLine(profile), run = profileRun(profile);
  const s = profileS(n, profile, tau);
  const len = n.meta[profile * 4 + 3];
  const trainM = trainLengthM(n, line);
  // T-3250: a return run on its own way back is drawn along it, as a run 0 of its own (`geo`, `gr`)
  const { geo, gr, d } = trainFrame(n, profile, s);
  const across = (x: number) => (bay ? acrossAt(bay, bayAt, gr, len, x, trainM) : 1);
  const [pa, pb] = bay ? [bay[bayAt + 1], bay[bayAt + 3]] : [-1, -1];
  // the nearest its middle may come to either end of run 0 (T-3053), as the train shader clamps it
  const lo = endClear(gr === 0 ? pa : pb, trainM), hi = len - endClear(gr === 0 ? pb : pa, trainM);
  return { ...pointAt(n, geo, d, gr, shiftM * across(d), bandM, band, bandOn, passM), s, run, line, geo, gr, shiftM, bandM, band, bandOn, passM, ex: 0, lo, hi, across };
}

/** T-3250: the line table's slot of a line's return run where it takes its own way back (the clock
 * worker's second half of the table, after the `meta` lines), else -1. */
export function backSlotOf(table: ArrayLike<number>, meta: ArrayLike<number>, line: number): number {
  const lines = backBaseOf(table, meta);
  if (!lines || line >= lines) return -1;
  return table[(lines + line) * 4 + 1] >= 2 ? lines + line : -1;
}

/** T-3250: the lines in the line table when it has a second half of return runs' own slots, else 0
 * (the train shader's `u_back_base`). */
export function backBaseOf(table: ArrayLike<number>, meta: ArrayLike<number>): number {
  const lines = meta.length / META_PER_LINE;
  return lines > 0 && table.length >= lines * 8 ? lines : 0;
}

/** Where profile `p`'s train `s` m along its run is drawn: the line table's row (`geo`, its line or
 * its return run's own slot), the run it is drawn as (`gr`) and the offset along that row. */
export function trainFrame(n: Pick<NetworkBuffers, "lineTable" | "meta">, p: number, s: number): { geo: number; gr: number; d: number } {
  const line = profileLine(p), run = profileRun(p);
  const slot = run === 1 ? backSlotOf(n.lineTable, n.meta, line) : -1;
  return slot >= 0 ? { geo: slot, gr: 0, d: s } : { geo: line, gr: run, d: run === 0 ? s : n.meta[p * 4 + 3] - s };
}

/** A line's train as drawn, m: its cars at 20 m, a bus or boat as the train shader has it. */
function trainLengthM(n: NetworkBuffers, line: number): number {
  const raw = n.lineCars?.[line] ?? 8, cars = Math.max(1, raw);
  return Math.min(cars * (raw < 0 ? -raw * 20 : raw < 1 ? raw * 20 : 20), n.lineTable[line * 4 + 2]);
}

/** Position and direction used by both train and car picking, `shiftM` to the side trains keep to (`keep`). */
function pointAt(n: NetworkBuffers, line: number, d: number, run = 0, shiftM = 0, bandM = 0, band = n.sampleBand, bandOn = 0, passM = 0) {
  const [x, y] = lineAtX(n, line, d);
  const [bx, by] = lineAtX(n, line, d - 5), [fx, fy] = lineAtX(n, line, d + 5);
  const dx = fx - bx, dy = fy - by;
  let [ox, oy] = rightOf(n, line, d, y, dx, dy, run, shiftM);
  const len = Math.hypot(dx, dy);
  if (bandM && band && len > 1e-9) {
    // left of run 0's direction (local y points south, so left of (dx, dy) is (dy, -dx))
    const [lw, m] = bandAt(n, band, line, d);
    const k = (lw * bandM + m * bandOn) * kAtY(y, n.anchor ?? anchor()) / len;
    ox += dy * k;
    oy -= dx * k;
  }
  const bow = passM && n.samplePass && len > 1e-9 ? passBow(sampleValue(n, n.samplePass, line, d) / PASS_LEAD_M) : 0;
  if (bow > 0) {
    // the outer track on the side it keeps to, as the train shader's `shiftAt`
    const k = passM * kAtY(y, n.anchor ?? anchor()) * (run === 0 ? 1 : -1) * (n.keep ?? 1) / len;
    ox += (-dy * k - ox) * bow;
    oy += (dx * k - oy) * bow;
  }
  return { x: x + ox, y: y + oy, dx, dy };
}

/** A per-sample value of a line at offset `d` along run 0, between its samples. */
export function sampleValue(n: NetworkBuffers, v: Float32Array, line: number, d: number): number {
  const first = n.lineTable[line * 4];
  if (n.lineTable[line * 4 + 1] < 2) return v[first] ?? 1e9;
  const [si, f] = sampleSegment(n, line, d);
  return v[si] + (v[si + 1] - v[si]) * f;
}

/** A line's band at offset `d` along run 0 (`b`, 2 per sample): [shift, line widths left of it; metres to its ribbon's middle]. */
function bandAt(n: NetworkBuffers, b: Float32Array, line: number, d: number): [number, number] {
  const first = n.lineTable[line * 4];
  if (n.lineTable[line * 4 + 1] < 2) return [b[first * 2] ?? 0, b[first * 2 + 1] ?? 0];
  const [si, f] = sampleSegment(n, line, d);
  return [b[si * 2] + (b[si * 2 + 2] - b[si * 2]) * f, b[si * 2 + 1] + (b[si * 2 + 3] - b[si * 2 + 1]) * f];
}

/** The shift to the side trains keep to, `keep` (local y points south, so right of (dx, dy) is (-dy, dx)). */
function rightOf(n: NetworkBuffers, line: number, d: number, y: number, dx: number, dy: number, run: number, shiftM: number): [number, number] {
  const len = Math.hypot(dx, dy);
  if (!shiftM || len < 1e-9) return [0, 0];
  const k = shiftM * laneAt(n, line, d) * kAtY(y, n.anchor ?? anchor()) * (run === 0 ? 1 : -1) * (n.keep ?? 1) / len;
  return [-dy * k, dx * k];
}

/** A close-up car's centre and bogies, matching the GPU's terminal clamp and 20 m pitch. */
export function trainCarAt(n: NetworkBuffers, at: NonNullable<ReturnType<typeof tripAt>>, car: number) {
  const raw = n.lineCars?.[at.line] ?? 8, cars = Math.max(1, raw);
  const len = n.lineTable[at.geo * 4 + 2], total = Math.min(cars * (raw < 0 ? -raw * 20 : raw < 1 ? raw * 20 : 20), len);
  const d = at.gr === 0 ? at.s : len - at.s;
  const centre = Math.max(at.lo, Math.min(Math.max(at.lo, at.hi), d)) + at.ex;
  const carD = centre + (car + 0.5) * total / cars - total / 2;
  const half = Math.max(total / cars / 2 - 0.5, 1);
  const p = pointAt(n, at.geo, carD, at.gr, at.shiftM * at.across(carD), at.bandM, at.band, at.bandOn, at.passM);
  const [cx, cy] = lineAtX(n, at.geo, carD), ox = p.x - cx, oy = p.y - cy;
  const [bx, by] = lineAtX(n, at.geo, carD - half), [fx, fy] = lineAtX(n, at.geo, carD + half);
  return { ...p, back: [bx + ox, by + oy] as [number, number], front: [fx + ox, fy + oy] as [number, number] };
}
