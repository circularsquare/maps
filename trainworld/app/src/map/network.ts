// What the renderer draws, as flat typed arrays in local units (map/geo.ts), built from the clock
// worker's state (T-025): track strokes per edge, line strokes along each line's path, station
// dots, and the keyframe tables trains are positioned from (notes/T-040.md "Keyframes").
//
// Trains: each trip is (profile, departure - epoch); profile p = (line * 2 + run) * 3 + level has
// a phase table (t, s, v, a) and meta (first phase, count, trip seconds, path length m); run 0 of
// each line is resampled every 50 m into `samples`; run 1's offset s is `length - s` on run 0.

import { rgb } from "../game/palette";
import { hasUpgrade, UPGRADE } from "../game/upgrades";
import type { PreviewAnswer } from "../workers/clockClient";
import type { ClockEdge, ClockState } from "../workers/protocol";
import { anchor, kAtY, toLocal, type Anchor } from "./geo";
import { M_BAD, M_CROSSING, M_JUNCTION, MARK_STRIDE, rampMarks } from "./markData";
import { placeFacilities } from "./facilityPlace";
import { buildingsDrawn, type BuildingDraw } from "./buildingNet";
import { BAND_M_RANGE, BAND_RANGE, BAND_STRIDE, bandKnots, KNOT, lineBands, profileAt, type BandProfile } from "./lineBands";
import { BOX_TOUCH_M, DEMAND_MERGE_M, offsetsReach, polylineGap } from "../game/complexes";
import { acrossAt, endClear, TURNBACK_M } from "./terminalBays";
import { bridgeEnds, PASS_LEAD_M, passBow, PLATFORM_HALF_M, PLATFORM_POINTS, platformPoints, platformShape, reversed, ST_BUFFER, ST_BUILDING, ST_PARK, ST_PASSING, ST_ROOM, ST_TERMINAL, ST_TRAM, type PlatformShape } from "./stationGeometry";
import { planStrokes, strokesOfPieces, type StrokeKind, type StrokeMode, type StrokePiece } from "./strokePost";
import { oneWayTrackChevrons, trainChevrons } from "./oneWayMarks";

/** A bus line's `lineCars` (T-1464): one car of this share of a train car's 20 m (a bus is 12 m),
 * narrower than a train's (rendererShaders.ts); below 1, so it tells a bus from a train. */
export const BUS_CAR = 0.6;
/** T-2194: a ferry line's: below 0, a boat of this many train cars' 20 m (35 m, `ferry::BOAT_LEN_M`),
 * drawn as one pointed hull. */
export const BOAT_CAR = -1.75;

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
/** A tram line's `lineCars` is its sections plus this, so the car shader tells it from a train of as many cars
 * (readers of the count use `carsOf`). */
export const TRAM_MARK = 0.5;
/** The cars a `lineCars` entry stands for: a bus (below 1) and a boat (below 0) as given, a tram's without its mark. */
export const carsOf = (raw: number) => (raw >= 1 ? Math.floor(raw) : raw);

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
   * on double track: a platform each side close in, T-658; ST_ROOM and ST_FLIP the complex's waiting
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
  /** T-1029: the clock worker's number for this build, and `patch` when it was posted as changes to
   * the build numbered `patch.base` (`NetPatch`) */
  serial?: number;
  patch?: NetPatch;
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
  /** `minSeg`: points closer than this (plane m) to the last one kept are dropped (`MIN_SEG_M`) */
  constructor(private withStretch = false, private minSeg = 0) {}
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
    this.colour = g(this.colour, 1);
    this.edge = g(this.edge, 1);
    this.level = g(this.level, 1);
    this.flags = g(this.flags, 1);
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
      this.colour[c] = colour;
      this.edge[c] = edge;
      this.level[c] = Math.round(pts[i * 3 + 2] / 8);
      this.flags[c] = eased ? flags | F_EASE : flags;
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
function complexPassages(st: readonly { id: number; lat: number }[], stations: Float32Array, stationLine: Float32Array, complexOf: ArrayLike<number>, links: readonly number[], pieces?: readonly object[], cache?: NetCache["passages"]): number[] {
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
    const ln = linksIn.get(c) ?? [];
    let hit: PassageEntry | undefined;
    if (cache) {
      const e = cache.get(st[m[0]].id);
      if (e && e.ids.length === m.length && e.links.length === ln.length && m.every((i, a) => e.ids[a] === st[i].id && e.pieces[a] === pieces![i] && e.xy[a * 2] === stations[i * 3] && e.xy[a * 2 + 1] === stations[i * 3 + 1]) && ln.every((i, k) => e.links[k] === st[i].id)) hit = e;
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
        else joins.push([gap, i, j]);
      }
    joins.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
    for (const [, i, j] of joins) if (find(i) !== find(j)) { union(i, j); out.push(i, j); }
    if (cache) {
      const pos = new Map(m.map((i, a) => [i, a]));
      seen.set(st[m[0]].id, {
        ids: m.map((i) => st[i].id), pieces: m.map((i) => pieces![i]), xy: m.flatMap((i) => [stations[i * 3], stations[i * 3 + 1]]),
        links: ln.map((i) => st[i].id), joins: out.slice(from).map((i) => pos.get(i)!),
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
interface PassageEntry { ids: number[]; pieces: object[]; xy: number[]; links: number[]; joins: number[] }

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
export function strokesFromPreview(pts: Float32Array, bad: boolean, speed?: Float32Array, placement = false, simMarks?: ArrayLike<number>, lane = 255, hsr = false, tram = false): Strokes {
  const b = new StrokeBuilder();
  const flags = F_BLUEPRINT | (bad ? F_BAD : 0) | (placement ? F_PLACEMENT : 0) | (hsr ? F_HSR : 0) | (tram ? F_TRAM : 0);
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

/** One stroke set from a preview of several alignments (`parts`), red when it cannot be built. */
export function previewStrokes(a: PreviewAnswer): Strokes | null {
  const parts = a.parts ?? new Uint32Array([0, a.pts.length / 3]);
  const all: Strokes[] = [];
  for (let k = 0; k + 1 < parts.length; k++) {
    const pts = a.pts.subarray(parts[k] * 3, parts[k + 1] * 3);
    // the track model's crossings and junctions belong to the whole preview: given to the first part
    if (pts.length >= 6) all.push(strokesFromPreview(pts, !a.ok, undefined, false, all.length ? undefined : a.marks));
  }
  if (!all.length) return null;
  if (all.length === 1) return all[0];
  const marks = all.flatMap((s) => (s.marks ? Array.from(s.marks) : []));
  const n = all.reduce((s, x) => s + x.count, 0);
  const out: Strokes = {
    seg: new Float32Array(n * 4), colour: new Uint32Array(n), edge: new Uint32Array(n), level: new Float32Array(n),
    flags: new Float32Array(n), dist: new Float32Array(n), count: n, marks: marks.length ? new Float32Array(marks) : undefined,
  };
  let o = 0;
  for (const s of all) {
    out.seg.set(s.seg.subarray(0, s.count * 4), o * 4);
    out.colour.set(s.colour.subarray(0, s.count), o);
    out.edge.set(s.edge.subarray(0, s.count), o);
    out.level.set(s.level.subarray(0, s.count), o);
    out.flags.set(s.flags.subarray(0, s.count), o);
    out.dist.set(s.dist.subarray(0, s.count), o);
    o += s.count;
  }
  return out;
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
/** How far a one-way leg eases from its own track's line in to its centre line, m (T-3117). */
const LEG_EASE_M = 60;

/**
 * T-3117: where one-way single track meets double track at a junction (a flyover's J1, J2 and J3),
 * which way its end is drawn off its centre line, so it carries on from its own track of the double:
 * away from the nearest other track leaving the node its way. Per edge, [a end, b end]: +1 left of
 * a to b, -1 right, 0 none. `parts`: each edge's points, in `edges` order.
 */
function legShifts(edges: readonly ClockEdge[], parts: readonly ArrayLike<number>[]): Map<number, [number, number]> {
  const out = new Map<number, [number, number]>();
  if (!edges.some((e) => e.tracks === 1 && e.oneWay)) return out;
  // per node, each end there: its edge's index and the way it leaves (unit x, y)
  const at = new Map<number, { i: number; dx: number; dy: number }[]>();
  const dir = (pts: ArrayLike<number>, fromA: boolean) => {
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
  };
  edges.forEach((e, i) => {
    for (const [node, fromA] of [[e.a, true], [e.b, false]] as const) {
      const d = dir(parts[i], fromA);
      if (!d) continue;
      const v = at.get(node) ?? [];
      v.push({ i, ...d });
      at.set(node, v);
    }
  });
  edges.forEach((e, i) => {
    if (e.tracks !== 1 || !e.oneWay) return;
    const shift: [number, number] = [0, 0];
    [e.a, e.b].forEach((node, end) => {
      const ends = at.get(node) ?? [];
      if (ends.length < 3 || !ends.some((x) => edges[x.i].tracks === 2)) return;
      const me = ends.find((x) => x.i === i && (end === 0 ? edges[x.i].a === node : edges[x.i].b === node));
      if (!me) return;
      let best: (typeof ends)[number] | null = null, cos = Math.cos(Math.PI / 3);
      for (const o of ends) {
        const c = o.dx * me.dx + o.dy * me.dy;
        if (o !== me && c > cos) [best, cos] = [o, c];
      }
      if (!best) return;
      const otherLeft = me.dx * best.dy - me.dy * best.dx > 0;
      // leaving from b runs b to a, so its right is a to b's left
      shift[end] = (otherLeft ? -1 : 1) * (end === 0 ? 1 : -1);
    });
    if (shift[0] || shift[1]) out.set(e.id, shift);
  });
  return out;
}

/** `pts` (x, y, z triples) with each end moved `shift` (as `legShifts`) half the track gap off to its
 * side, easing back to the line over LEG_EASE_M, points added every few metres there; and per
 * segment, the segment of `pts` it lies on. */
function shiftedEnds(pts: Float32Array, shift: [number, number]): { pts: Float32Array; src: number[] } {
  const n = pts.length / 3;
  if (n < 2) return { pts, src: [] };
  const at = new Float64Array(n);
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
  const len = at[n - 1], kk = kAtY(pts[1]), L = Math.min(LEG_EASE_M * kk, len / 2), step = 5 * kk;
  const ease = (d: number) => {
    const t = Math.max(0, Math.min(1, d / L));
    return 1 - t * t * (3 - 2 * t);
  };
  const off = (s: number) => HALF_GAP_M * kk * (shift[0] * ease(s) + shift[1] * ease(len - s));
  const out: number[] = [], seg: number[] = [];
  const put = (k: number, f: number) => {
    seg.push(k);
    const x = pts[k * 3] + (pts[k * 3 + 3] - pts[k * 3]) * f, y = pts[k * 3 + 1] + (pts[k * 3 + 4] - pts[k * 3 + 1]) * f;
    const z = pts[k * 3 + 2] + (pts[k * 3 + 5] - pts[k * 3 + 2]) * f;
    const dx = pts[k * 3 + 3] - pts[k * 3], dy = pts[k * 3 + 4] - pts[k * 3 + 1], d = Math.hypot(dx, dy) || 1;
    const o = off(at[k] + (at[k + 1] - at[k]) * f);
    out.push(x - (dy / d) * o, y + (dx / d) * o, z);
  };
  for (let k = 0; k < n - 1; k++) {
    const seg = at[k + 1] - at[k];
    put(k, 0);
    // points every `step` where either end eases
    if (seg > step && (at[k] < L || at[k + 1] > len - L)) {
      for (let d = step; d < seg - 1e-6; d += step) {
        const s0 = at[k] + d;
        if (s0 < L || s0 > len - L) put(k, d / seg);
      }
    }
  }
  put(n - 2, 1);
  // each segment drawn: the segment of \`pts\` it lies on
  return { pts: new Float32Array(out), src: seg.slice(0, -1) };
}

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

/** T-2248: per line, the passing stations in `at` where its trains stop on the running track: beside a
 * slower line's train standing on the outer track (passing-tracks design, notes/T-2228.md 5). Of two
 * lines stopping at a station and sharing the track through it, the slower is the one that stops at
 * more of the stations on the stretch they run together; equal, both stand on the outer track. */
function innerStops(lines: ClockState["lines"], at: ReadonlyMap<number, unknown>, edgeEnds: ReadonlyMap<number, { a: number; b: number }>): Map<number, Set<number>> {
  const seq = new Map<number, number[]>(), stopsOf = new Map<number, Set<number>>(), by = new Map<number, number[]>();
  for (const l of lines) {
    if (!l.ok || l.bus || !l.path.length) continue;
    const nodes: number[] = [];
    for (const [k, p] of l.path.entries()) {
      const e = edgeEnds.get(p >> 1);
      if (!e) break;
      if (k === 0) nodes.push((p & 1) ? e.b : e.a);
      nodes.push((p & 1) ? e.a : e.b);
    }
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

/**
 * An edge's drawn points for a line on it (T-1799): with a point added at each knot of its band
 * (`lineBands`), and per point its shift and `1 - width share` (`StrokeBuilder.add`'s `band`). `b`:
 * the line's band profile, `k` the path entry, in its travel direction (`reverse`: b to a).
 */
function bandPoints(dr: LanePoints, reverse: boolean, b: BandProfile, entry: number): { pts: Float32Array; lane: number | Uint8Array; band: Float32Array; ease: Uint8Array } {
  const p = dr.lpts, n = p.length / 3;
  const at = new Float64Array(n);
  for (let k = 1; k < n; k++) at[k] = at[k - 1] + Math.hypot(p[k * 3] - p[k * 3 - 3], p[k * 3 + 1] - p[k * 3 - 2]);
  const perM = kAtY(p[1]), len = at[n - 1] / perM;
  const [o, c] = bandKnots(b, entry);
  // the knots, m along the edge a to b
  const cuts: number[] = [];
  for (let i = 0; i < c; i++) cuts.push((reverse ? 1 - b[o + i * KNOT] : b[o + i * KNOT]) * len);
  cuts.sort((x, y) => x - y);
  const out: number[] = [], lane: number[] = [], band: number[] = [], ms: number[] = [];
  const laneAt = (k: number) => (typeof dr.lane === "number" ? dr.lane : dr.lane[k]);
  const push = (x: number, y: number, z: number, m: number, l: number) => {
    out.push(x, y, z);
    lane.push(l);
    ms.push(m);
    band.push(...profileAt(b, entry, len > 0 ? (reverse ? len - m : m) / len : 0));
  };
  push(p[0], p[1], p[2], 0, laneAt(0));
  for (let k = 0; k < n - 1; k++) {
    const m0 = at[k] / perM, m1 = at[k + 1] / perM;
    for (const c of cuts) {
      if (c <= m0 + 1e-3 || c >= m1 - 1e-3) continue;
      const f = (c - m0) / (m1 - m0);
      push(p[k * 3] + (p[k * 3 + 3] - p[k * 3]) * f, p[k * 3 + 1] + (p[k * 3 + 4] - p[k * 3 + 1]) * f, p[k * 3 + 2] + (p[k * 3 + 5] - p[k * 3 + 2]) * f, c, Math.round(laneAt(k) + (laneAt(k + 1) - laneAt(k)) * f));
    }
    push(p[k * 3 + 3], p[k * 3 + 4], p[k * 3 + 5], m1, laneAt(k + 1));
  }
  // per segment, a to b: whether it lies between two knots that differ (in either view)
  const ease = new Uint8Array(Math.max(0, ms.length - 1));
  for (let i = 0; i < ease.length; i++) {
    const f = len > 0 ? ((ms[i] + ms[i + 1]) / 2) / len : 0, t = reverse ? 1 - f : f;
    let j = 1;
    while (j < c - 1 && b[o + j * KNOT] < t) j++;
    const x = o + (j - 1) * KNOT, y = o + j * KNOT;
    ease[i] = c > 1 && (b[x + 1] !== b[y + 1] || b[x + 2] !== b[y + 2] || b[x + 3] !== b[y + 3] || b[x + 3] < 0 || b[x + 4] !== b[y + 4]) ? 1 : 0;
  }
  return { pts: out.length === p.length ? p : new Float32Array(out), lane: typeof dr.lane === "number" && out.length === p.length ? dr.lane : Uint8Array.from(lane), band: new Float32Array(band), ease };
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
  /** T-3117: `legShifts` for it, or null */
  shift: [number, number] | null;
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
  /** T-3117: `ClockLine.back`, empty for none */
  back: number[];
  stops: number[];
  running: boolean;
  /** high-speed trains: only such a line is drawn in the high-speed look (T-1494) */
  fast: boolean;
  tram: boolean;
  /** its bands (`LineBands.of`), null on no shared track */
  band: Float32Array | null;
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
  /** `sampleBand`'s, empty for a line on no shared track */
  band: Float32Array;
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
  /** each complex's drawn passages, by its first member's id (`complexPassages`) */
  passages: Map<number, PassageEntry>;
  serial: number;
  last: { serial: number; track: EdgeEntry[]; lines: LineEntry[]; trackCount: number; lineCount: number; pts: Map<number, Float32Array> } | null;
}

export const newNetCache = (): NetCache => ({ edges: new Map(), lines: new Map(), stations: new Map(), speeds: new Map(), samples: new Map(), passages: new Map(), serial: 0, last: null });

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
  const moved = new Map<Strokes, Strokes>();
  const strokes = (s: Strokes): Strokes => {
    let t = moved.get(s);
    if (!t) {
      const seg = new Float32Array(s.seg.length);
      for (let k = 0; k + 1 < seg.length; k += 2) { seg[k] = s.seg[k] + dx; seg[k + 1] = s.seg[k + 1] + dy; }
      moved.set(s, (t = { ...s, seg }));
    }
    return t;
  };
  for (const [id, c] of cache.edges) {
    const g = given.get(id);
    if (!g || g.length !== c.pts.length || c.given !== was.get(id)) { cache.edges.delete(id); continue; }
    const pts = g.slice();
    let lpts = pts;
    if (c.lpts !== c.pts) {
      lpts = new Float32Array(c.lpts.length);
      for (let k = 0; k + 2 < lpts.length; k += 3) { lpts[k] = c.lpts[k] + dx; lpts[k + 1] = c.lpts[k + 1] + dy; lpts[k + 2] = c.lpts[k + 2]; }
    }
    const ramps = c.ramps.slice();
    for (let k = 0; k + 1 < ramps.length; k += MARK_STRIDE) { ramps[k] = Math.fround(ramps[k] + dx); ramps[k + 1] = Math.fround(ramps[k + 1] + dy); }
    cache.edges.set(id, { ...c, given: g, pts, lpts, strokes: strokes(c.strokes), ramps });
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
  for (const id of p.edgeIds) {
    const v = n.edgePts.get(id) ?? prev.edgePts.get(id);
    if (!v) return false;
    pts.set(id, v);
  }
  const track = planStrokes("track", prev.track, n.track, p.modes.track, p.ranges.track), lines = planStrokes("lines", prev.lines, n.lines, p.modes.lines, p.ranges.lines);
  if (!track || !lines) return false;
  // nothing is written into the last network's arrays unless the whole network can be merged
  track.write();
  lines.write();
  for (const k of p.same) (n as unknown as Record<string, unknown>)[k] = from[k];
  n.edgePts = pts;
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
 * the same either way. `deferStrokes` (the clock worker's): `track` and `lines` hold only their
 * counts, and `strokePieces` the pieces to make them from (map/strokePost.ts). */
export function networkFromState(s: ClockState, cache: NetCache = newNetCache(), deferStrokes = false): NetworkBuffers {
  const r = s.render;
  const edgePts = new Map<number, Float32Array>();
  const builtEdge = new Map<number, boolean>();
  const offEdge = new Set<number>(s.edges.filter((e) => e.off).map((e) => e.id));
  const hsrEdge = new Set<number>(s.edges.filter((e) => e.hsr).map((e) => e.id));
  const tramEdge = new Set<number>(s.edges.filter((e) => e.kind).map((e) => e.id));
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
  const shifts = legShifts(s.edges, r.edgeParts!);
  /** per edge id: the points and lanes its track and lines are drawn with */
  const drawn = new Map<number, LanePoints>();
  // blueprint track after built, so drawn over it and picked first where they overlap, as the
  // station tool prefers it (T-695)
  const order = s.edges.map((_, i) => i).sort((i, j) => Number(!s.edges[i].built) - Number(!s.edges[j].built) || i - j);
  for (const i of order) {
    const e = s.edges[i];
    const given = r.edgeParts![i];
    const sg: ArrayLike<number> | undefined = r.edgeSpeedParts?.[i];
    builtEdge.set(e.id, e.built);
    const flags = (e.built ? 0 : F_BLUEPRINT | (e.off ? F_OFF : 0)) | (e.hsr ? F_HSR : 0) | (e.kind ? F_TRAM : 0);
    const lanes = (e.tracks === 2 ? 1 : 0) | (e.tracks === 2 && taperAt(e.a) ? 2 : 0) | (e.tracks === 2 && taperAt(e.b) ? 4 : 0);
    let c = cache.edges.get(e.id);
    const shift = shifts.get(e.id) ?? null;
    if (c && c.a === e.a && c.b === e.b && c.flags === flags && c.lanes === lanes && c.shift?.[0] === shift?.[0] && c.shift?.[1] === shift?.[1] && (c.given === given || sameFloats(c.pts, given))) {
      kept.add(e.id);
      if (c.given !== given) c = { ...c, given };
    } else {
      const own = given.slice();
      const sh = shift ? shiftedEnds(own, shift) : null;
      const lp = lanePoints(sh ? sh.pts : own, lanes);
      const b = new StrokeBuilder(false, MIN_SEG_M), from: number[] = [];
      b.add(lp.lpts, TRACK, e.id, flags, false, 0, undefined, lp.lane, undefined, undefined, from);
      const ramps: number[] = [];
      rampMarks(own, e.built ? 0 : 1, ramps);
      const strokes = b.build();
      // each stroke segment's segment of the edge's given points, for its speed
      const srcL = from.length === lp.lpts.length / 3 - 1 ? lp.src : from.map((k) => (lp.src ? lp.src[k] : k));
      const src = sh ? Array.from({ length: strokes.count }, (_, j) => sh.src[srcL ? srcL[j] : j] ?? 0) : srcL;
      c = { given, pts: own, a: e.a, b: e.b, flags, lanes, shift, ...lp, src, strokes, ramps, speedGiven: undefined, vmin: NaN, speed: new Float32Array(0), train: new Float32Array(strokes.count).fill(-1) };
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
  for (const l of s.lines) lineCars[l.id] = l.bus?.ferry ? BOAT_CAR : l.bus ? BUS_CAR : l.tram ? Math.max(1, l.cars) + TRAM_MARK : Math.max(1, l.cars);
  const lineParts: LineEntry[] = [];
  let lineCount = 0;
  const linesSeen: NetCache["lines"] = new Map();
  const samplesSeen: NetCache["samples"] = new Map();
  const lineIds: string[] = [];
  const onEdge = linesOnEdges(s);
  const edgeLen = new Map(s.edges.map((e) => [e.id, e.lengthM]));
  const edgeEnds = new Map(s.edges.map((e) => [e.id, e]));
  // stations whose passing tracks work: on double track (T-2233, sim `overtake::passing_at`)
  const doubleAt = new Set<number>();
  for (const e of s.edges) if (e.tracks === 2) doubleAt.add(e.a).add(e.b);
  const passingAt = new Map<number, ClockState["nodes"][number]>();
  for (const n of s.nodes) if (n.platform > 0 && hasUpgrade(n.upgrades, UPGRADE.passing) && doubleAt.has(n.id)) passingAt.set(n.id, n);
  const innerAt = passingAt.size ? innerStops(s.lines, passingAt, edgeEnds) : null;
  // T-2248: a passing station's way out along an edge, carried on through plain nodes (two edges) until
  // its outer track's lead fits, and the lead it has room for
  const edgesAt = new Map<number, number[]>();
  if (passingAt.size) for (const e of s.edges) for (const x of [e.a, e.b]) edgesAt.set(x, [...(edgesAt.get(x) ?? []), e.id]);
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
  // lines sharing track side by side (T-1799)
  const bands = lineBands(s.lines, s.edges, edgePts);
  for (const l of lineDrawOrder(s.lines)) {
    if (!l.ok) continue;
    lineIds.push(String(l.id));
    let c = cache.lines.get(l.id);
    const band = bands.of.get(l.id) ?? null;
    if (!c || c.running !== l.running || c.fast !== l.fast || c.tram !== l.tram || !sameArray(c.path, l.path) || !sameArray(c.back, l.back ?? []) || !sameArray(c.stops, l.stops) || !l.path.every((p) => kept.has(p >> 1)) || !(l.back ?? []).every((p) => kept.has(p >> 1)) || (c.band === null ? band !== null : band === null || (c.band !== band && !sameBand(c.band, band)))) {
      const lines = new StrokeBuilder(true, MIN_SEG_M);
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
          const flags = (l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0))) | (l.fast && hsrEdge.has(e) ? F_HSR : 0) | (l.tram && tramEdge.has(e) ? F_TRAM : 0);
          const stretchOf = Math.min(stretch, Math.max(0, l.stops.length - 2));
          if (dr.lpts.length >= 6) firstStroke[k] = lines.count;
          if (band && dr.lpts.length >= 6) {
            const bp = bandPoints(dr, (p & 1) === 1, band, k);
            d = lines.add(bp.pts, l.id, e, flags, (p & 1) === 1, d, stretchOf, bp.lane, bp.band, bp.ease);
          } else d = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, d, stretchOf, dr.lane);
        }
        // a path entry runs a -> b, or b -> a when its low bit is set
        const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
        if (to === l.stops[next]) {
          stretch++;
          next++;
        }
      });
      // T-3117: track only the return run takes (a flyover's other leg), its stretch counted from the far end
      if (l.back?.length) {
        const own = new Set(l.path.map((p) => p >> 1)), last = Math.max(0, l.stops.length - 2);
        let bd = 0, back = 0, nextBack = l.stops.length - 2;
        for (const p of l.back) {
          const e = p >> 1, dr = drawn.get(e), ends = edgeEnds.get(e);
          if (dr && !own.has(e)) {
            const flags = (l.running ? 0 : F_FADED | (builtEdge.get(e) ? 0 : F_BLUEPRINT | (offEdge.has(e) ? F_OFF : 0))) | (l.tram && tramEdge.has(e) ? F_TRAM : 0);
            bd = lines.add(dr.lpts, l.id, e, flags, (p & 1) === 1, bd, Math.max(0, Math.min(last, last - back)), dr.lane);
          }
          const to = ends ? ((p & 1) === 1 ? ends.a : ends.b) : -1;
          if (nextBack >= 0 && to === l.stops[nextBack]) {
            back++;
            nextBack--;
          }
        }
      }
      c = { path: l.path.slice(), back: (l.back ?? []).slice(), stops: l.stops.slice(), running: l.running, fast: l.fast, tram: l.tram, band, strokes: lines.build(), firstStroke };
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
      // a bus has one centre line, not two tracks (T-1464)
      const rel = new Int32Array(count), lane = new Float32Array(count).fill(l.bus ? 0 : 1), bandOf = new Float32Array(c.band ? count * 2 : 0);
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
        // its band's shift (T-1799) and metres (T-2132), left of run 0
        if (c.band) {
          const v = profileAt(c.band, j, len > 0 ? Math.max(0, Math.min(1, 1 - (end - at) / len)) : 0);
          bandOf[k * 2] = v[0];
          bandOf[k * 2 + 1] = v[2];
        }
      }
      sm = { c, count, step, lens, rel, lane, band: bandOf };
    }
    samplesSeen.set(l.id, sm);
    const stretchOf = c.strokes.stretch!;
    for (let k = 0; k < count; k++) {
      const f = sm.rel[k];
      sampleStroke[first + k] = f < 0 ? -1 : base + f;
      if (f >= 0) sampleStretch[first + k] = stretchOf[f];
    }
    sampleLane.set(sm.lane, first);
    if (sm.band.length) sampleBand.set(sm.band, first * 2);
    if (passingAt.size && !l.bus) {
      const mine = linePassing(l, passingAt, edgeEnds, lens, innerAt?.get(l.id), leadOf, passing);
      for (let k = 0; k < count && mine.length; k++) {
        let v = 1e9;
        for (let j = 0; j < mine.length; j += 4) {
          const x = k * step - mine[j], past = Math.abs(x) - mine[j + 1];
          v = Math.min(v, past > 0 ? past * PASS_LEAD_M / mine[x < 0 ? j + 2 : j + 3] : past);
        }
        samplePass[first + k] = v;
      }
    }
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
  const turnbacks = turnbackStrokes(s, linesSeen, endCount, (b) => {
    const ways = (n: number) => (waysAt.get(n) ?? []).map((id): [number, Float32Array] => [id, way(n, id)]);
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
    passingTracks(b, [...passingAt.values()], waysOut, leadOf);
  });
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
    stations.set([p[0], p[1], ((served.get(n.id) ?? 0) > 1 ? 1 : 0) | (n.stationBuilt ? 0 : 2) | (lined.has(n.id) ? 4 : 0) | (n.stationOff ? 8 : 0) | (edgeEnds.get(platformEdge.get(n.id) ?? -1)?.tracks === 2 ? 16 : 0) | (n.kind ? ST_TRAM : 0) | (hasUpgrade(n.upgrades, UPGRADE.waitingRoom) ? ST_ROOM : 0) | (hasUpgrade(n.upgrades, UPGRADE.parkRide) ? ST_PARK : 0) | (hasUpgrade(n.upgrades, UPGRADE.terminal) ? ST_TERMINAL | (ids.length === 1 ? ST_BUFFER : 0) : 0) | (passingAt.has(n.id) ? ST_PASSING : 0)], i * 3);
  });
  cache.stations = stationsSeen;
  // T-2219: the station buildings' members as drawn (their boxes join the floor), which car parks stay out of
  const outlines = buildingsDrawn(s.buildings, s.buildingLL, st, stations, stationLine, edgePts, false);
  for (let i = 0; i < st.length; i++) if (outlines.memberOf[i] >= 0) stations[i * 3 + 2] += ST_BUILDING;
  // T-1528: one waiting room and one car park per complex
  const lots = placeFacilities(st, stations, stationLine, stationShape, edgePts, waysAt, s.complexOf, outlines.draw);
  // their insides, which leave room for the waiting rooms just placed
  const bld = outlines.draw.length ? buildingsDrawn(s.buildings, s.buildingLL, st, stations, stationLine, edgePts) : outlines;
  // the player's links as bridges between the two platforms' nearest points (T-217)
  const stationIndex = new Map(st.map((n, i) => [n.id, i]));
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
    const auto = complexPassages(st, stations, stationLine, s.complexOf, linkIdx, st.map((n) => stationsSeen.get(n.id)!), cache.passages);
    for (let k = 0; k + 1 < auto.length; k += 2) {
      const i = auto[k], j = auto[k + 1];
      if (oneFloor(i, j)) continue;
      linkIdx.push(i, j);
      linkEnds.push(...bridgeEnds(stations[i * 3], stations[i * 3 + 1], shapeAt(i), stations[j * 3], stations[j * 3 + 1], shapeAt(j)));
    }
  }
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
  // T-3094: chevrons along one-way single track, clear of stations
  const oneWayTrack: number[] = [];
  if (s.edges.some((e) => e.oneWay)) {
    // T-3117: clear of junctions too, where two legs leaving one node would put their arrows side by side
    const stationAt = s.nodes.filter((n) => n.platform > 0 || n.ports >= 3).map((n) => nodeLocal.get(n.id)).filter((p) => p !== undefined);
    for (const e of s.edges) {
      const c = e.oneWay ? edgesSeen.get(e.id) : undefined;
      if (c) oneWayTrackChevrons(c.strokes, e.oneWay as 1 | 2, stationAt, oneWayTrack);
    }
  }
  const jn = s.nodes.filter((n) => n.ports >= 3 && n.platform === 0);
  const junctions = new Float32Array(jn.length * 2);
  jn.forEach((n, i) => junctions.set(nodeLocal.get(n.id) ?? toLocal(n.lng, n.lat), i * 2));
  // the marks (T-629): a junction where moves cross at one level (T-3103) is built once three of
  // its tracks are; a crossing once both of its are
  jn.forEach((n, i) => {
    if (n.crosses) marks.push(junctions[i * 2], junctions[i * 2 + 1], M_JUNCTION, 0, n.builtPorts >= 3 ? 0 : 1, n.level);
  });
  const cx = s.crossings;
  if (cx) for (let k = 0; k + 4 < cx.length; k += 5) marks.push(cx[k], cx[k + 1], cx[k + 2] === 1 ? M_BAD : M_CROSSING, 0, cx[k + 4] === 1 ? 0 : 1, cx[k + 3]);
  const pathTable = new Int32Array(Math.max(1, maxLine + 1) * 2);
  let pathN = 0;
  for (const l of s.lines) if (l.ok) pathN += l.path.length;
  const pathEntry = new Int32Array(pathN), pathFrom = new Float32Array(pathN), pathLen = new Float32Array(pathN);
  pathN = 0;
  for (const l of s.lines) {
    if (!l.ok) continue;
    pathTable[l.id * 2] = pathN;
    pathTable[l.id * 2 + 1] = l.path.length;
    let d = 0;
    for (const p of l.path) {
      const len = edgeLen.get(p >> 1) ?? 0;
      pathEntry[pathN] = p;
      pathFrom[pathN] = d;
      pathLen[pathN++] = len;
      d += len;
    }
  }
  const serial = ++cache.serial, last = cache.last;
  const patch: NetPatch | undefined = last ? {
    base: last.serial,
    track: strokeRuns(trackParts, (p) => p.strokes.count, last.track),
    lines: strokeRuns(lineParts, (p) => p.strokes.count, last.lines),
    oldTrack: last.trackCount, oldLines: last.lineCount, same: [], edgeIds: Uint32Array.from(trackOrder), modes: { track: "whole", lines: "whole" }, ranges: { track: new Int32Array(0), lines: new Int32Array(0) },
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
    stationIds: st.map((n) => String(n.id)),
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
  return [n.samples[si * 2] + (n.samples[si * 2 + 2] - n.samples[si * 2]) * f, n.samples[si * 2 + 1] + (n.samples[si * 2 + 3] - n.samples[si * 2 + 1]) * f];
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
 * run 0's direction there. `shiftM`: metres to the right of its direction of travel on double
 * track, where the renderer draws it on its own track (T-658, `NetworkRenderer.laneShiftM`); `bandM`:
 * metres a line width is drawn, where its line's band shifts it (T-1799, `NetworkRenderer.bandShiftM`),
 * by `band` per sample (the Traffic view's own, `NetworkRenderer.sampleBand`), and `bandOn` how far its
 * metres to the ribbon's middle are taken (`NetworkRenderer.bandOn`, T-2132). `passM`: metres out to a
 * passing station's outer track, where its trains stop (T-2233, `NetworkRenderer.passShiftM`). */
export function tripAt(n: NetworkBuffers, profile: number, dep: number, t: number, end = -Infinity, shiftM = 0, bandM = 0, band = n.sampleBand, bandOn = 0, passM = 0, bay?: ArrayLike<number>, bayAt = 0): { x: number; y: number; s: number; run: number; line: number; dx: number; dy: number; shiftM: number; bandM: number; band?: Float32Array; bandOn: number; passM: number; ex: number; lo: number; hi: number; across: (x: number) => number } | null {
  const tau = t - dep;
  const tripS = n.meta[profile * 4 + 2];
  // `end`: drawn until then, standing at the terminal for its next departure (T-206)
  if (tau < 0 || (tau > tripS && t > end) || !n.meta[profile * 4 + 1]) return null;
  const line = Math.floor(profile / 6), run = Math.floor(profile / 3) % 2;
  const s = profileS(n, profile, tau);
  const len = n.meta[profile * 4 + 3];
  const trainM = trainLengthM(n, line);
  const across = (x: number) => (bay ? acrossAt(bay, bayAt, run, len, x, trainM) : 1);
  const [pa, pb] = bay ? [bay[bayAt + 1], bay[bayAt + 3]] : [-1, -1];
  // the nearest its middle may come to either end of run 0 (T-3053), as the train shader clamps it
  const lo = endClear(run === 0 ? pa : pb, trainM), hi = len - endClear(run === 0 ? pb : pa, trainM);
  const d = run === 0 ? s : len - s;
  return { ...pointAt(n, line, d, run, shiftM * across(d), bandM, band, bandOn, passM), s, run, line, shiftM, bandM, band, bandOn, passM, ex: 0, lo, hi, across };
}

/** A line's train as drawn, m: its cars at 20 m, a bus or boat as the train shader has it. */
function trainLengthM(n: NetworkBuffers, line: number): number {
  const raw = n.lineCars?.[line] ?? 8, cars = Math.max(1, raw);
  return Math.min(cars * (raw < 0 ? -raw * 20 : raw < 1 ? raw * 20 : 20), n.lineTable[line * 4 + 2]);
}

/** Position and direction used by both train and car picking, `shiftM` to the right of travel. */
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
    // the outer track to the right of its travel, as the train shader's `shiftAt`
    const k = passM * kAtY(y, n.anchor ?? anchor()) * (run === 0 ? 1 : -1) / len;
    ox += (-dy * k - ox) * bow;
    oy += (dx * k - oy) * bow;
  }
  return { x: x + ox, y: y + oy, dx, dy };
}

/** A per-sample value of a line at offset `d` along run 0, between its samples. */
function sampleValue(n: NetworkBuffers, v: Float32Array, line: number, d: number): number {
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

/** The shift to the right of travel (local y points south, so right of (dx, dy) is (-dy, dx)). */
function rightOf(n: NetworkBuffers, line: number, d: number, y: number, dx: number, dy: number, run: number, shiftM: number): [number, number] {
  const len = Math.hypot(dx, dy);
  if (!shiftM || len < 1e-9) return [0, 0];
  const k = shiftM * laneAt(n, line, d) * kAtY(y, n.anchor ?? anchor()) * (run === 0 ? 1 : -1) / len;
  return [-dy * k, dx * k];
}

/** A close-up car's centre and bogies, matching the GPU's terminal clamp and 20 m pitch. */
export function trainCarAt(n: NetworkBuffers, at: NonNullable<ReturnType<typeof tripAt>>, car: number) {
  const raw = n.lineCars?.[at.line] ?? 8, cars = Math.max(1, raw);
  const len = n.lineTable[at.line * 4 + 2], total = Math.min(cars * (raw < 0 ? -raw * 20 : raw < 1 ? raw * 20 : 20), len);
  const d = at.run === 0 ? at.s : len - at.s;
  const centre = Math.max(at.lo, Math.min(Math.max(at.lo, at.hi), d)) + at.ex;
  const carD = centre + (car + 0.5) * total / cars - total / 2;
  const half = Math.max(total / cars / 2 - 0.5, 1);
  const p = pointAt(n, at.line, carD, at.run, at.shiftM * at.across(carD), at.bandM, at.band, at.bandOn, at.passM);
  const [cx, cy] = lineAtX(n, at.line, carD), ox = p.x - cx, oy = p.y - cy;
  const [bx, by] = lineAtX(n, at.line, carD - half), [fx, fy] = lineAtX(n, at.line, carD + half);
  return { ...p, back: [bx + ox, by + oy] as [number, number], front: [fx + ox, fy + oy] as [number, number] };
}
