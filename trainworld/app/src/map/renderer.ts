// WebGL2 network renderer: track and line strokes (one instanced quad per polyline segment, a
// casing under the colour, the theme's (T-039); blueprint solid, lighter and without casing, T-311), stations (platforms along the track, or
// dots zoomed out, T-189), the drawing tool's preview,
// and instanced trains positioned on the GPU from keyframes (shared phase tables, notes/T-040.md)
// and a time uniform. Per frame the CPU sets uniforms and issues a handful of draw calls, whatever
// the train count (notes/T-002.md).
//
// Time precision (T-016, SPEC 7): trip departures are relative to a render epoch at the start of
// the current game hour, so the float32 time uniform stays under 3600 s. The clock worker sends
// each hour's trips (and the next hour's, ahead); when the hour turns, only that buffer is
// re-uploaded. Phase tables change only on edits.

import { BAD_RED, LEVEL_COLOURS, rgbf } from "../game/palette";
import { ANNEX_ROOM, stationAnnexes, stationCoords, PASS_GAP_PX, PASS_TRACK_M, ST_BUFFER, ST_BUILDING, ST_PASSING, ST_ROOM, ST_TERMINAL, TERMINAL_CAP, BOX_WIDTH_ZOOM, boxHalfPxAtBoxZoom, BRIDGE_HALF_PX, bridgeShape, connectorHalfPx, dotRing, spanningPairs, insideStation, PLATFORM_POINTS, platformPoints, SEL_GLOW_PX, SEL_OUTLINE_PX, shapePoints, STATION_BOX_ZOOM, STATIONS_ZOOM, TRAINS_ZOOM, stationRadii, stationReachPx, stationRing, stationShape, type Project, type StationShape, type StationStyle } from "./stationGeometry";
import { shadeColours } from "../game/shades";
import { readBoardHour, trainStands, type BoardHour, type CrowdedStretch } from "../game/boarding";
import { UPGRADE } from "../game/upgrades";
import { PlatformCrowds } from "./platformCrowds";
import { TransferBubbles } from "./transferBubbles";
import { BusLayer } from "./busLayer";
import { MOVING_PLACE_MS } from "./moving";
import { anchor, anchorPsi, EARTH_C, kAtY, localToMercMatrix, mul4, pxPerUnitAt, unitPerPxAt, type Anchor } from "./geo";
import { clipToCss, compile, GpuTimer, interp, localCamera, mercToCss, PROJECT_UNIFORMS, setCamera, uniforms, type Camera, type LocalCamera } from "./gl";
import { GradeMarks } from "./gradeMarks";
import { OneWayChevrons } from "./oneWayMarks";
import { StrokeCache } from "./strokeCache";
import { chunkBoxes, mergeRanges, significance, simplifiedLater, visibleRanges, type ChunkBoxes } from "./strokeLod";
import { StrokeGather } from "./strokeGather";
import { perfOff } from "../perfFlags";
import { ParkingLots } from "./parkingLots";
import { BuildingFloors, STENCIL_BIT } from "./buildingFloors";
import { TrainGuard } from "./trainGuard";
import { LOT_STRIDE } from "./facilityPlace";
import { F_BLUEPRINT, platformShapeAt, type NetworkBuffers, type Strokes } from "./network";
import { BAY_STRIDE } from "./terminalBays";
import { BAND_GAP_PX, BAND_RANGE, BAND_STRIDE, LINE_VIEW_WIDTH, TRAFFIC_PICK_DIV, type TrafficBands } from "./lineBands";
import { BRIDGE_VS, HOT_STATIONS, LANE_SHARE, LINES_W, PROBE_FS, PROBE_VS, SEL_FS, SEL_MAX, SEL_SET_MAX, SEL_STRIDE, SEL_VS, STATION_FS, STATION_VS, STROKE_FS, STROKE_VS, TEX_W, TRAIN_FS, TRAIN_VS } from "./rendererShaders";

/** The render epoch moves to the start of each game hour. */
export const EPOCH_S = 3600;
/** The longest a staged state waits for its trips, wall ms (T-1071). */
const STAGE_WAIT_MS = 2000;
type TripArgs = [epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array, bays?: Float32Array];
const TRACK_RGB = rgbf("#77746f");
const BAD_RGB = rgbf(BAD_RED);
/** Blueprint track (T-322): a light blue middle as wide as a line, with a dark blue rail this
 * share of a line width (at least a pixel) along each edge. */
const BP_MID_RGB = rgbf("#a9cdf3");
const BP_RAIL_RGB = rgbf("#2f62b3");
const BP_RAIL_SHARE = 0.16;
/** The whole blueprint track, rails included, is this share of its full width (T-331); the rails
 * stay at least a pixel. */
const BP_NARROW = 0.75;
/** A train's width on screen by zoom, CSS px, pills and cars alike (about 1.4 lines wide from zoom
 * 12); its true 3 m is wider only from about zoom 19. */
const TRAIN_PX: [number, number][] = [[10, 3], [11, 5.25], [12, 6], [15, 9]];
/** A line's width on screen by zoom, CSS px. Bare track, the casing and a selected line's extra
 * width follow the same steps. */
const LINE_PX: [number, number][] = ([[9, 0.85], [12, 3.9], [15, 6.6]] as [number, number][]).map(([z, px]) => [z, px * LINE_VIEW_WIDTH]);
/** T-1097 (Anita: "a tad more visual weight, not much"): high-speed track and its lines this many
 * times as wide by zoom while drawn as one stroke, and their centre line only on strokes this wide
 * (css px, coming in between the two) */
const HSR_WIDTH: [number, number][] = [[9, 1.4], [11, 1.25], [13, 1]];
const HSR_STRIPE_PX = [3, 5] as const;

/** The Speed view's value per segment (T-1271): the trains' speed where a line with trains runs,
 * else the limit. */
const selSetBuf = new Int32Array(SEL_SET_MAX);
/** T-2980: per trip, no place at either end */
const noBays = (n: number) => new Float32Array(Math.max(1, n) * BAY_STRIDE).fill(-1);
/** T-1812: the lines of a selected group, for the stroke and train shaders' `u_sel_set` */
function setSelLines(gl: WebGL2RenderingContext, u: Record<string, WebGLUniformLocation | null>, ls: readonly number[] | undefined) {
  const n = Math.min(ls?.length ?? 0, SEL_SET_MAX);
  for (let i = 0; i < n; i++) selSetBuf[i] = ls![i];
  gl.uniform1i(u.u_sel_n, n);
  gl.uniform1iv(u.u_sel_set, selSetBuf);
}

function shownSpeed(limit: Float32Array, trains?: Float32Array): Float32Array {
  if (!trains || trains.length !== limit.length) return limit;
  return limit.map((v, i) => (trains[i] >= 0 ? trains[i] : v));
}
/** The ghost of track being placed (T-1038, Anita: "a decent minimum width ... see the track in full
 * length"), drawn as blueprint track with a thin dark outline (T-1054): never thinner than this
 * across its rails, CSS px, with this much outline either side. */
const GHOST_MIN_PX = 4, GHOST_OUTLINE_PX = 1;
/** Double track's two tracks (T-658): their centres 4 m apart on the ground, each a line's width
 * of LANE_SHARE of that (2.4 m) once apart. Where 4 m is LANE_GROW_PX on screen a double track's
 * stroke starts growing to the pair's width, which it reaches at LANE_PART_PX, where the two tracks
 * start parting; they are fully apart at LANE_APART_PX (zoom 16.7 to 17.2 in New York). */
const LANE_M = 4, LANE_GROW_PX = 4, LANE_PART_PX = 7, LANE_APART_PX = 10;
/** Past LANE_APART_PX each track narrows by up to LANE_NARROW (T-713, Anita: "reduce the width of
 * the individual tracks by around 20%" so the riders waiting behind them show), reached at
 * LANE_NARROW_PX; trains keep their 3 m. */
const LANE_NARROW = 0.2, LANE_NARROW_PX = 16;
/** T-1799 (Anita, 2026-10-29): lines sharing track fold into one stroke as a line's width falls
 * through these CSS px zoomed out (about zoom 9.5 to 10) */
const BAND_FOLD_PX = [1.25, 1.75] as const;
/** T-2058: with a line picked, the others' width share and how far their colour is mixed towards the
 * casing's, drawn solid (Anita: "thin and faded"); `?perfOff=muteThin`: at `dimOthers` opacity instead */
const MUTE: readonly [number, number] | null = perfOff("muteThin") ? null : [0.45, 0.5];
/** T-2126: bands side by side with a pixel between (`?perfOff=bandGap`: touching, as before) */
const BAND_GAP = !perfOff("bandGap");
/** T-2980: trains take a place at their line's ends (`?perfOff=bays`: all at the end of the arrival track, as before) */
const BAYS = !perfOff("bays");
const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** How far double track has opened into two tracks at `zoom`, by the size of 4 m on screen at the
 * view's anchor: the gap in CSS px, the growth and parting (0-1), and half the pair's drawn width
 * in CSS px once it has grown (0 before), for picking. */
export function laneState(zoom: number) {
  const gap = LANE_M * kAtY(0) * pxPerUnitAt(zoom);
  const grow = Math.max(0, Math.min(1, (gap - LANE_GROW_PX) / (LANE_PART_PX - LANE_GROW_PX)));
  const t = Math.max(0, Math.min(1, (gap - LANE_PART_PX) / (LANE_APART_PX - LANE_PART_PX)));
  const narrow = 1 - LANE_NARROW * Math.max(0, Math.min(1, (gap - LANE_APART_PX) / (LANE_NARROW_PX - LANE_APART_PX)));
  return { gap, grow, t, narrow, share: LANE_SHARE * narrow, reach: grow > 0 ? 0.5 * (1 + LANE_SHARE * narrow) * gap : 0 };
}
/** Passing stations close in (T-2233), CSS px: how far out the outer tracks are (PASS_TRACK_M on the
 * ground, or a train's drawn width and PASS_GAP_PX if wider, so one passing never touches one standing)
 * and the least half width of their box (that, half a train, the rim and a pixel). */
export function passingState(zoom: number, ringPx: number) {
  const lanes = laneState(zoom), mPx = lanes.gap / LANE_M;
  const train = interp(zoom, TRAIN_PX) * (1 - lanes.t) + 3 * mPx * lanes.t;
  const off = Math.max(PASS_TRACK_M * mPx, train + PASS_GAP_PX);
  return { off, half: off + train / 2 + ringPx + 1, mPx };
}
/** Platform crowds (T-164, T-219): drawn from this zoom, where platforms are (T-204), as each side's
 * half of the box filling from its middle outwards in `CROWD_RGB`, red (BAD_RGB) once that side is
 * full (T-559). */
export const CROWD_ZOOM = STATION_BOX_ZOOM;
/** T-1966: trains wait for each other at flat crossings and crossing junction moves from this zoom,
 * where a train is drawn as its cars at about its true length, so two at a crossing read apart;
 * further out they are drawn through each other (Anita, 2026-11-03: only where we're looking closely). */
export const CONFLICT_ZOOM = STATION_BOX_ZOOM;
const CROWD_RGB = rgbf("#a4a19d");
/** The track drawn through a station box (T-236) is at most this share of the box's white, so the
 * crowd shows beside it. */
const TRACK_IN_BOX = 0.3;
/** Stations as the complex rule's distances group them (game/complexView.ts `dotGroups`): zoomed out
 * each group's dots are joined by connectors, the shortest set that links them all (T-313). */
export interface StationGroups {
  members: string[][];
}

export interface DrawOptions {
  showTrains: boolean;
  byLevel: boolean;
  bySpeed?: boolean;
  /** T-2420: the track coloured by its stretch's profit (`setProfit`) */
  byProfit?: boolean;
  /** selected line id, -1 for none */
  selLine: number;
  /** T-1812: more lines drawn selected, a selected group's (at most SEL_SET_MAX); the bus layer takes `selLine` only */
  selLines?: readonly number[];
  /** the selected train (T-242): its profile and departure (game seconds), drawn over the others
   * with a halo; absent for none */
  selTrain?: { profile: number; dep: number } | null;
  /** selected edge id, -1 for none */
  selEdge: number;
  /** other lines' share of full strength beside the selected line's (T-233 stop editing); 1: none */
  dimOthers?: number;
  /** selected station's index in the station buffer, -1 for none */
  selStation: number;
  /** more stations drawn selected, indices in the station buffer: in complex view, the selected
   * station's whole complex (T-163) */
  selStations?: number[];
  /** riders waiting on platforms (T-164, T-219): each platform filling up as they wait */
  crowds?: boolean;
  /** the marks on flat crossings and junctions (T-673); absent: shown */
  flatMarks?: boolean;
  /** the marks on ramps and portals (T-3048); absent: shown */
  rampMarks?: boolean;
  /** the stations the complex rule's distances group (T-313): zoomed out their dots are joined by
   * connectors; absent, only the player's links are */
  groups?: StationGroups | null;
  /** hovered (T-730): a line's number, a bare track edge's id (-1 none), stations by index */
  hotLine?: number;
  hotEdge?: number;
  hotStations?: number[];
  /** T-2993: bus lines lit as hovered (numbers), the bus tool's route will change with them */
  hotBuses?: number[];
  /** T-1248: lines, track and trains drawn this many times their width; absent: 1 */
  lineScale?: number;
  /** T-1567: every train one pill, white with a rim of its line's colour (the start menu's logo) */
  hollowTrains?: boolean;
  /** T-1700: every box this many px narrower on each side (the start menu's logo) */
  boxShrinkPx?: number;
}

interface StrokeVao {
  vao: WebGLVertexArrayObject;
  count: number;
  /** the per-segment width multipliers, rewritten by `setLayers` */
  wmulBuf: WebGLBuffer;
  /** per segment its traffic mark (T-819, `writeFull`) and where along its run it starts (`fullAttr`) */
  fullBuf: WebGLBuffer;
  /** the network's strokes (T-1028): each attribute's buffer and bytes a segment, in `STROKE_ATTRIBS`
   * order, for the next network to copy its unchanged runs from; and what to free with them */
  attribs?: { buf: WebGLBuffer; bytes: number }[];
  res?: { del: () => void }[];
}

type TexName = "phases" | "meta" | "samples" | "lines" | "colours" | "fill" | "cars" | "stretch" | "stationLine" | "lane";

/** `a_wmul` for `n` plain segments: width 1, no Traffic view band shift (T-1799) */
function plainWmul(n: number): Float32Array {
  const v = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) v[i * 3] = 1;
  return v;
}

/** `a_full` for `n` segments (2 a segment: the traffic mark, from `marks` or 0, and the local units
 * from the start of its run of joined segments of one line and edge to its start, which the dashes
 * are measured along so they keep one length round curves, T-2184); with `start`, segments `start` on,
 * a run taken to begin there (a network piece's first segment always begins one: no two pieces share an
 * edge and line) */
function fullAttr(s: { seg: Float32Array; colour: Uint32Array; edge: Uint32Array }, n: number, marks?: (i: number) => number, start = 0): Float32Array {
  const out = new Float32Array(Math.max(1, n) * 2), g = s.seg;
  let at = 0;
  for (let i = start, o = 0; o < n; i++, o++) {
    if (o > 0 && (s.colour[i] !== s.colour[i - 1] || s.edge[i] !== s.edge[i - 1] || g[i * 4] !== g[i * 4 - 2] || g[i * 4 + 1] !== g[i * 4 - 1])) at = 0;
    if (marks) out[o * 2] = marks(i);
    out[o * 2 + 1] = at;
    at += Math.hypot(g[i * 4 + 2] - g[i * 4], g[i * 4 + 3] - g[i * 4 + 1]);
  }
  return out;
}

/** The stroke attributes a network's strokes keep on the GPU between edits (T-1028), in order. */
const STROKE_ATTRIBS = ["seg", "colour", "edge", "level", "flags", "speed", "lane", "band"] as const;

/** T-1639: the network's strokes drawn simplified below this zoom (plus log2 of the pixel ratio),
 * merged where they stray from the full set by at most LOD_TOL_PX device px; from CULL_ZOOM only the
 * chunks near the view drawn, CULL_MARGIN_PX (CSS) round it covering the widest stroke */
const LOD_TOP_ZOOM = 15, LOD_TOL_PX = 0.25, CULL_ZOOM = 7, CULL_MARGIN_PX = 48;
/** T-1644: a set's visible instances are gathered into one buffer (strokeGather.ts) when there are at most
 * this many; more are drawn by ranges, a gap of RANGE_GAP instances or fewer drawn through */
const GATHER_MAX = 40000, RANGE_GAP = 512;
/** the finest chunk boxes tested at `zoom` (strokeLod.ts level 0 is a chunk): below zoom 12 the globe's
 * projection makes each box test dear, and boxes of 128 or 1024 instances are fine enough */
const cullLevel = (zoom: number) => (zoom < 10 ? 2 : zoom < 12 ? 1 : 0);
/** T-1640: the longest a still picture goes undrawn, ms; the fastest train's speed, m/s; the game
 * seconds a platform's crowd is drawn in steps of */
const STILL_MAX_MS = 1000, TRAIN_MAX_MPS = 100, CROWD_STEP_S = 5;

/** a network stroke set's chunk boxes, and its simplified copies by zoom band (T-1639) */
interface StrokeLodState {
  boxes: ChunkBoxes | null;
  speed: Float32Array | null | undefined;
  sig: Float32Array | null;
  /** how far `sig` is worked out, and whether that work is under way */
  sigAt: number;
  working: boolean;
  bands: Map<number, StrokeBand>;
  /** the zoom band whose copy is being made (`workBand`), null for none */
  making: number | null;
}
/** a simplified copy of a stroke set for one zoom band, with the arrays its buffers hold (T-1644) */
interface StrokeBand { vao: StrokeVao; strokes: Strokes; first: Uint32Array; boxes: ChunkBoxes; v: number; res: { del: () => void }[]; wm: Float32Array | null; fu: Float32Array | null }

/**
 * The demand views drawn by the network renderer (T-084, map/demandLayers.ts), one value per
 * line stroke segment or train sample; null = drawn as without demand.
 */
export interface NetworkLayers {
  /** per line stroke segment: width multiple */
  lineWidth: Float32Array;
  /** per train sample: how full run 0's and run 1's trains are (2 each: 0-1 of the gauge, 2 over
   * full, -1 not known) */
  fill: Float32Array;
  /** the widths are the riders' (the traffic view) */
  traffic?: boolean;
  /** the traffic view's bands side by side (T-1799 stage 2), null where no track is shared */
  bands?: TrafficBands | null;
}

export class NetworkRenderer {
  private gl!: WebGL2RenderingContext;
  private strokeProg!: WebGLProgram;
  private stationProg!: WebGLProgram;
  private trainProg!: WebGLProgram;
  private strokeU!: Record<string, WebGLUniformLocation | null>;
  private stationU!: Record<string, WebGLUniformLocation | null>;
  private trainU!: Record<string, WebGLUniformLocation | null>;
  private probeProg: WebGLProgram | null = null;
  private probeU: Record<string, WebGLUniformLocation | null> = {};
  /** crossings, flat junctions, ramps and portals (T-629, T-655) */
  readonly marks = new GradeMarks();
  readonly oneWay = new OneWayChevrons();
  /** T-3094: along one-way single track */
  readonly oneWayTrack = new OneWayChevrons();
  /** the complexes' car parks (T-1528) */
  readonly lots = new ParkingLots();
  /** the station buildings' floors (T-2218) */
  readonly floors = new BuildingFloors();
  private res: { del: () => void }[] = [];
  private previewRes: { del: () => void }[] = [];
  private track: StrokeVao | null = null;
  private lines: StrokeVao | null = null;
  /** the crossovers where lines turn back (T-658d), drawn close in only */
  private turnback: StrokeVao | null = null;
  private preview: StrokeVao | null = null;
  private highlightRes: { del: () => void }[] = [];
  private refusedRes: { del: () => void }[] = [];
  private refused: StrokeVao | null = null;
  /** the drag-selected blueprint (T-333): an ink band drawn under the track */
  private highlight: StrokeVao | null = null;
  private stationVao: WebGLVertexArrayObject | null = null;
  private trainVao: WebGLVertexArrayObject | null = null;
  private probeVao: WebGLVertexArrayObject | null = null;
  private tripBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: when it stops being drawn, relative to the epoch (T-206) */
  private endBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: its first load and when it reaches its last stop (T-164) */
  private loadBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: its stand at the terminal before its next trip (T-1785, `trainStands`) */
  private standBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: seconds it is drawn behind its timetable (T-1965, `guard.lag`) */
  private lagBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: where it stands at its ends (T-2980, `assignBays`) */
  private bayBuf: WebGLBuffer | null = null;
  private lagDirty = false;
  /** T-1965: trains on the same track drawn one behind the other, never through each other; off
   * with `?perfOff=follow` or live */
  guard = new TrainGuard();
  follow = !perfOff("follow");
  /** game seconds trip `i` of the buffer is drawn behind its timetable (T-1965) */
  tripLag(i: number): number {
    return this.follow && i >= 0 ? this.guard.lag[i] ?? 0 : 0;
  }
  /** the hour's loads (T-164), replaced with the trips; kept apart from `res`, like the trips */
  private loadTex: WebGLTexture | null = null;
  /** T-2420: the Profit view's figure per edge id, and how many ids it holds */
  private profitTex: WebGLTexture | null = null;
  private profitN = 0;
  private profitData: Float32Array | null = null;
  private tex: Record<TexName, WebGLTexture | null> = { phases: null, meta: null, samples: null, lines: null, colours: null, fill: null, cars: null, stretch: null, stationLine: null, lane: null };
  /** The boarding account's hour the trip buffer holds (T-164), or null before demand has riders. */
  board: BoardHour | null = null;
  private crowds = new PlatformCrowds(0);
  private crowdBuf: WebGLBuffer | null = null;
  private dotCrowdBuf: WebGLBuffer | null = null;
  /** the trip buffer's index of a trip (profile, departure - epoch), for the train inspector */
  tripIndex(profile: number, depRel: number): number {
    const d = this.trips;
    for (let i = 0; i < d.length; i += 2) if (d[i] === profile && Math.abs(d[i + 1] - depRel) < 0.01) return i / 2;
    return -1;
  }
  /** whether the last draw drew platform crowds (on, riders known, zoomed in to CROWD_ZOOM), for
   * the crowd hover */
  crowdsShown = false;
  /** The camera is moving (set by the overlay): the crowds are worked out at most every
   * MOVING_PLACE_MS meanwhile (T-1207). */
  moving = false;
  private crowdsAt = 0;
  private crowdsAny = false;
  private maxCars = 8;
  /** Used by picking to follow the same close-up representation. */
  detailedTrains = false;
  private stationShapeBuf: WebGLBuffer | null = null;
  /** how the last draw sized stations, CSS px (picking, the cut-out above the demand bubbles) */
  stationStyle: StationStyle = { boxes: false, radius: stationRadii(12), linePx: 0, ringPx: stationRing(12), boxLinePx: interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx: stationRing(BOX_WIDTH_ZOOM) };
  /** zoomed out below STATIONS_ZOOM at the last draw: trains neither drawn nor picked (T-205) */
  trainsHidden = false;
  private selectedStation = -1;
  /** `DrawOptions.selStations` as a set, and as drawn into `stationSelBuf` (its key) */
  private selectedAlso = new Set<number>();
  private selectedAlsoKey = "";
  private stationSelBuf: WebGLBuffer | null = null;
  /** what `stationSelBuf` and `bridgeFlagBuf` were last written for */
  private selDrawnKey = "";
  /** the groups the connectors were last worked out for (T-313) */
  private groupSrc: { groups: StationGroups | null; net: NetworkBuffers | null } = { groups: null, net: null };
  private bridgeProg!: WebGLProgram;
  private bridgeU: Record<string, WebGLUniformLocation | null> = {};
  private bridgeVao: WebGLVertexArrayObject | null = null;
  private bridgeFlagBuf: WebGLBuffer | null = null;
  /** zoomed out, the connectors between dots (T-313): the two stations of each (index pairs), their
   * dots' centres (local units), and its VAO with the selected and faded flags; rebuilt by
   * syncGroups, so kept apart from `res` */
  connPairs = new Int32Array(0);
  private connEnds = new Float32Array(0);
  private connVao: WebGLVertexArrayObject | null = null;
  private connFlagBuf: WebGLBuffer | null = null;
  private connRes: { del: () => void }[] = [];
  private selProg!: WebGLProgram;
  private selU: Record<string, WebGLUniformLocation | null> = {};
  private selTex: WebGLTexture | null = null;
  private selData = new Float32Array(SEL_MAX * SEL_STRIDE * 4);
  /** Station `i` on screen as the last draw drew it (stationGeometry.ts): its platform, or zoomed
   * out its own dot (T-313). */
  drawnShapeOf(i: number, project: Project): StationShape {
    return this.stationShapeOf(i, project);
  }
  /** The bridges between linked stations (T-217) drawn selected, on screen: both their stations
   * are; zoomed out, the connectors between selected dots (T-313). */
  private selectedBridges(project: Project): StationShape[] {
    const n = this.net;
    if (!n) return [];
    const out: StationShape[] = [];
    if (!this.stationStyle.boxes) {
      const p = this.connPairs, c = this.connEnds, half = connectorHalfPx(this.stationStyle);
      for (let k = 0; k < p.length / 2; k++) {
        if (!this.drawnSelected(p[k * 2]) || !this.drawnSelected(p[k * 2 + 1])) continue;
        const b = bridgeShape(c[k * 4], c[k * 4 + 1], c[k * 4 + 2], c[k * 4 + 3], project, this.stationStyle);
        if (b) out.push({ ...b, halfWidth: half });
      }
      return out;
    }
    const idx = n.stationLinkIdx, ends = n.stationLinks;
    if (!idx || !ends) return out;
    for (let k = 0; k < idx.length / 2; k++) {
      if (!this.drawnSelected(idx[k * 2]) || !this.drawnSelected(idx[k * 2 + 1])) continue;
      const b = bridgeShape(ends[k * 4], ends[k * 4 + 1], ends[k * 4 + 2], ends[k * 4 + 3], project, this.stationStyle);
      if (b) out.push(b);
    }
    return out;
  }
  /** Station `i` on screen as a platform or its own dot (stationGeometry.ts); `plain`: as wide as one
   * without passing tracks. */
  stationShapeOf(i: number, project: Project, plain = false): StationShape {
    const n = this.net!;
    const flag = Math.round(n.stations[i * 3 + 2]);
    return stationShape(n.stations[i * 3], n.stations[i * 3 + 1], platformShapeAt(n, i), n.stationShape?.[i * 4 + 3] ?? 0, project, this.stationStyle, (flag & 1) !== 0, !plain && (flag & ST_PASSING) !== 0);
  }
  private hotList = new Int32Array(HOT_STATIONS);
  /** The bridge program's hovered instances (T-730): the pairs in `pairs` (station indices, two
   * an instance) with both stations hovered. */
  private hotPairs(bu: Record<string, WebGLUniformLocation | null>, pairs: Int32Array, hot: number[]) {
    let k = 0;
    this.hotList.fill(-1);
    if (hot.length > 1) for (let j = 0; j * 2 < pairs.length && k < HOT_STATIONS; j++) {
      if (hot.includes(pairs[j * 2]) && hot.includes(pairs[j * 2 + 1])) this.hotList[k++] = j;
    }
    this.gl.uniform1iv(bu.u_hot_st, this.hotList);
    this.gl.uniform1i(bu.u_hot_n, k);
  }

  /** Whether station `i` is drawn selected: the selected station, or in complex view its complex. */
  private drawnSelected(i: number) {
    return i === this.selectedStation || this.selectedAlso.has(i);
  }
  /** The topmost station drawn at a point, with a small pointer tolerance; demand halos are not
   * targets. Selected stations are drawn last (T-189), so they come first; zoomed far out they are
   * the only ones drawn (T-205). Zoomed out each dot gives its own station (T-313): of dots that
   * overlap, the one whose centre is nearest. */
  stationAt(px: number, py: number, project: Project, padding = 7): number {
    const n = this.net;
    if (!n) return -1;
    const dots = !this.stationStyle.boxes;
    // T-724: on a flat camera, a station whose node is further from the pointer than its platform
    // and shape reach is skipped before its shape is worked out (same answers)
    const [ox, oy] = project(0, 0), [ex, ey] = project(1000, 0), [nx, ny] = project(0, 1000), [fx, fy] = project(1000, 1000);
    const a = (ex - ox) / 1000, c = (ey - oy) / 1000, b = (nx - ox) / 1000, e = (ny - oy) / 1000;
    const flat = Math.abs(fx - (ex + nx - ox)) + Math.abs(fy - (ey + ny - oy)) < 0.01;
    const reach = flat ? this.stationReachNow(n) : null;
    const perLoc = 1.01 * Math.max(Math.hypot(a, c), Math.hypot(b, e));
    const extra = reach ? stationReachPx(this.stationStyle, reach.halfWidth, perLoc) + padding + 2 : 0;
    for (const sel of this.stationStyle.hidden ? [true] : [true, false]) {
      let best = -1, bestD = Infinity;
      for (let i = n.stationCount - 1; i >= 0; i--) {
        if (this.drawnSelected(i) !== sel) continue;
        const annexed = !dots && (n.stations[i * 3 + 2] & ST_ROOM) !== 0;
        if (reach) {
          // an annex reaches about two box widths past the box (T-1416)
          const x = n.stations[i * 3], y = n.stations[i * 3 + 1], r = reach.r[i] * perLoc + extra * (annexed ? 3 : 1);
          const sx = ox + a * x + b * y, sy = oy + c * x + e * y;
          if (Math.abs(sx - px) > r || Math.abs(sy - py) > r) continue;
        }
        const s = this.drawnShapeOf(i, project);
        if (!insideStation(s, px, py, padding) && !(annexed && this.annexesOf(i, s).some((x) => insideStation(x.shape, px, py, padding)))) continue;
        if (!dots) return i;
        const d = Math.hypot(px - s.x, py - s.y);
        if (d < bestD) [best, bestD] = [i, d];
      }
      if (best >= 0) return best;
    }
    return -1;
  }
  private reachOf: { net: NetworkBuffers; r: Float32Array; halfWidth: number } | null = null;
  /** Per station, local units from its node to the farthest point of its platform's centre line,
   * and the widest true half width of any (stationAt's skip, T-724); worked out when first asked
   * for, a few ms where scanning every station's shape cost Japan's network 100 ms a move (T-1020). */
  private stationReachNow(n: NetworkBuffers): { r: Float32Array; halfWidth: number } {
    if (this.reachOf?.net === n) return this.reachOf;
    const r = new Float32Array(n.stationCount), s = n.stationShape;
    let halfWidth = 0;
    for (let i = 0; i < n.stationCount; i++) {
      const sh = platformShapeAt(n, i);
      const q = platformPoints(sh);
      let m = Math.hypot(sh.mx, sh.my) + Math.hypot(sh.ax, sh.ay);
      for (let j = 0; j + 1 < q.length; j += 2) m = Math.max(m, Math.hypot(q[j], q[j + 1]));
      r[i] = m;
      halfWidth = Math.max(halfWidth, s?.[i * 4 + 3] ?? 0);
    }
    this.reachOf = { net: n, r, halfWidth };
    return this.reachOf;
  }
  /** The selected stations drawn, by index: the selected one, or its complex in complex view. */
  private selectedDrawn(): number[] {
    const n = this.net;
    if (!n) return [];
    const out = new Set<number>();
    for (const i of [this.selectedStation, ...this.selectedAlso]) if (i >= 0 && i < n.stationCount) out.add(i);
    return [...out];
  }
  /** The selected stations as drawn, with the bridges or connectors between them (T-217, T-313):
   * the outline and glow go round them, and the demand bubbles are cut away round them so they
   * show above (T-189). */
  selectedShapes(project: Project, offFloors = false): StationShape[] {
    const out: StationShape[] = [];
    for (const i of this.selectedDrawn()) {
      // a station building's members: its floor's outline goes round them (T-2219)
      if (offFloors && Math.round(this.net!.stations[i * 3 + 2]) & ST_BUILDING) continue;
      const s = this.drawnShapeOf(i, project);
      out.push(s, ...this.annexesOf(i, s).map((x) => x.shape));
    }
    return out.concat(this.selectedBridges(project));
  }
  /** Station `i`'s annexes on screen (T-1416), `s` its shape as drawn: none zoomed out. */
  private annexesOf(i: number, s: StationShape) {
    return this.stationStyle.boxes ? stationAnnexes(s, Math.round(this.net!.stations[i * 3 + 2]), this.stationStyle.ringPx) : [];
  }
  /** The upgrade drawn at a point on station `i` (T-1416), its UPGRADE bit, or 0: the waiting room,
   * a terminal's thick end, or the car park whose hover names station `i` (T-1528). */
  upgradeAt(i: number, px: number, py: number, project: Project): number {
    const n = this.net;
    if (!n || !this.stationStyle.boxes || i < 0 || i >= n.stationCount) return 0;
    const s = this.drawnShapeOf(i, project);
    for (const x of this.annexesOf(i, s)) if (insideStation(x.shape, px, py, 1) && x.kind === ANNEX_ROOM) return UPGRADE.waitingRoom;
    const flags = Math.round(n.stations[i * 3 + 2]);
    if ((flags & ST_TERMINAL) && (flags & ST_BUFFER) && insideStation(s, px, py, 2)) {
      const [a] = stationCoords(s, px, py);
      if (a < TERMINAL_CAP * this.stationStyle.ringPx + 2 - s.halfLength) return UPGRADE.terminal;
    }
    if (!insideStation(s, px, py, 2)) {
      const k = this.lotAt(px, py, project);
      if (k >= 0 && this.lots.lots[k * LOT_STRIDE + 8] === i) return UPGRADE.parkRide;
    }
    return 0;
  }
  /** The car park drawn under a point (T-1528), its index in `lots.lots`, or -1: drawn from
   * STATION_BOX_ZOOM in. */
  lotAt(px: number, py: number, project: Project): number {
    return this.stationStyle.boxes ? this.lots.at(px, py, project) : -1;
  }
  /** A selected station drawn at a point, or -1 (picking: it is above the demand bubbles). */
  selectedStationAt(px: number, py: number, project: Project, padding = 2): number {
    for (const i of this.selectedDrawn()) {
      const s = this.drawnShapeOf(i, project);
      if (s && insideStation(s, px, py, padding)) return i;
    }
    return -1;
  }
  /** called when a draw drew a different selected station or complex (the demand canvas cuts the
   * selected stations out of its bubbles, T-189) */
  onSelectionDrawn: () => void = () => {};
  /** A line's width in CSS px at the last draw, 0 when lines are not drawn (picking) */
  slotPx = 0;
  /** the last map draw drew the bus and ferry lines (every track display but Level; picking) */
  busesShown = false;
  /** Half a double track's drawn width in CSS px at the last draw once it has grown to its two
   * tracks' width (T-658), 0 further out (picking) */
  laneReachPx = 0;
  /** Metres a train on double track is drawn to the right of its line's centre line at the last
   * draw (T-658): 2 m once the two tracks have parted, 0 further out (picking) */
  laneShiftM = 0;
  /** T-2233: metres from the centre line to a passing station's outer track as drawn at the last draw,
   * where trains stopping there stand (`passingState`); 0 zoomed out (picking). `passing`: the network
   * has passing tracks; `hadSamplePass`: its trains move out to them. */
  passShiftM = 0;
  private passing = false;
  private hadSamplePass = false;
  /** T-1799: CSS px a band's shift of one line width (`Strokes.band`) is drawn at the last draw, 0
   * where lines sharing track are drawn as one stroke (picking); the same in metres for trains */
  bandPx = 0;
  bandShiftM = 0;
  /** T-2132: how far lines are side by side at the last draw (0-1), which scales a band's metres to
   * its ribbon's middle; and CSS px those metres are drawn, a metre each (picking) */
  bandOn = 0;
  bandMPx = 0;
  /** T-1799: the bands picked against (`Strokes.band`'s layout) and CSS px a unit of their shift is
   * drawn at the last draw; the Traffic view's own (`TrafficBands.pick`) while it is shown */
  get pickBand(): Int16Array | undefined {
    return this.trafficBands?.pick ?? this.net?.lines.band;
  }
  get pickBandPx(): number {
    return this.trafficBands ? this.bandPx * TRAFFIC_PICK_DIV : this.bandPx;
  }
  /** per train sample, its band's shift as drawn (`NetworkBuffers.sampleBand`, or the Traffic view's) */
  get sampleBand(): Float32Array | undefined {
    return this.trafficBands?.sample ?? this.net?.sampleBand;
  }
  /** the most line widths a train is drawn off its line's centre line by its band (train picking's reach) */
  get bandReach(): number {
    return this.trafficBands ? this.trafficReach : BAND_RANGE / 2;
  }
  private trafficBands: TrafficBands | null = null;
  private trafficReach = 0;
  /** per line stroke segment, its width as a multiple of `slotPx` (traffic thickness), or null
   * when all are plain: picking reaches across wide lines (T-227) */
  lineWidths: Float32Array | null = null;
  /** the network has blueprint track: it and its lines are drawn after the rest (T-695) */
  hasBlueprint = false;
  /** ST_ROOM if some box draws a waiting room (T-1416), drawn in a station draw of its own */
  private annexes = 0;
  private levelColours = new Float32Array([3, 2, 1, 0, -1, -2, -3].flatMap((l) => [...rgbf(LEVEL_COLOURS[l as 3])]));
  private get ink() { return rgbf(shadeColours().stationInk); }
  private timer!: GpuTimer;
  /** the canvas has a stencil buffer: the track is drawn through station boxes (T-236) */
  private hasStencil = false;
  /** trips per epoch from the worker, newest version kept, with the boarding account's hour */
  private tripsBy = new Map<number, { version: number; data: Float32Array; ends?: Float32Array; nexts?: Float32Array; bays?: Float32Array; board?: BoardHour }>();
  /** what the trip buffer holds */
  private tripEpoch = NaN;
  private tripVersion = -1;
  tripCount = 0;
  /** the trips in the buffer, profile and departure - epoch (for picking) */
  trips: Float32Array = new Float32Array(0);
  /** per trip in `trips`, when it stops being drawn - epoch (T-206), or null */
  tripEnds: Float32Array | null = null;
  /** per trip in `trips`, the departure - epoch its train takes next, NaN leaving service (T-245) */
  tripNexts: Float32Array | null = null;
  /** per trip in `trips`, BAY_STRIDE numbers: where it stands at its ends (T-2980), or null */
  tripBays: Float32Array | null = null;
  net: NetworkBuffers | null = null;
  /** riders changing trains on their walk (T-1124) */
  readonly bubbles = new TransferBubbles();
  /** bus lines (T-1464): drawn from the world's routes, not the network buffers */
  readonly buses = new BusLayer();
  /** last GPU time for our draws in ms, if the timer extension exists */
  get gpuMs() {
    return this.timer?.ms ?? null;
  }
  /** current render epoch, game seconds; NaN until the first draw */
  epoch = NaN;
  rebases = 0;
  /** the camera of the last draw */
  cam: Camera | null = null;
  /** called with each new epoch the clock enters (the worker sends its trips) */
  onEpoch: (epoch: number) => void = () => {};

  /** T-1638: the strokes kept in a texture between changes (`?perfOff=strokeCache` starts it off) */
  strokeCache = !perfOff("strokeCache");
  /** the map's cache and the complex map snapshot's (overlay.ts) */
  mainCache: StrokeCache | null = null;
  private snapCache: StrokeCache | null = null;
  /** GL uploads and deletions made outside the per-frame passes (`quiet`): what the network,
   * preview, highlight, demand views and the like put on the GPU, so the caches know to redraw */
  private uploads = 0;
  private quiet = false;
  /** T-1639: off with `?perfOff=strokeCull` and `?perfOff=strokeLod` */
  strokeCull = !perfOff("strokeCull");
  strokeLod = !perfOff("strokeLod");
  private lodOf = new Map<StrokeVao, StrokeLodState>();
  /** T-1644: off with `?perfOff=strokeGather` */
  strokeGather = !perfOff("strokeGather");
  /** T-1799: lines sharing track side by side; T-1886: the Settings switch, off by default (main.ts) */
  lineBands = false;
  private trafficWidths = false;
  private gathers = new Map<StrokeVao, StrokeGather>();
  private speedShown = new WeakMap<Strokes, Float32Array>();
  /** the line strokes' demand widths (3 a segment: width, the Traffic view's band shift at its start
   * and end) and traffic marks as last written (null: the defaults), and a count of their writes,
   * which the simplified copies follow */
  private linesWmul: Float32Array | null = null;
  private linesFull: Float32Array | null = null;
  private perSegV = 0;
  /** stroke instances the last stroke redraw drew, and ms spent simplifying (checks read them) */
  strokeInstances = 0;
  lodMs = 0;
  private strokeLoc: Record<string, number> = {};

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.mainCache = this.snapCache = null;
    const g = gl as unknown as Record<string, unknown> & { twCounted?: boolean };
    if (!g.twCounted) {
      g.twCounted = true;
      for (const name of ["bufferData", "bufferSubData", "copyBufferSubData", "texImage2D", "texSubImage2D", "deleteBuffer", "deleteTexture", "deleteVertexArray"]) {
        const f = g[name] as (...args: unknown[]) => unknown;
        g[name] = (...args: unknown[]) => {
          if (!this.quiet) this.uploads++;
          return f.apply(gl, args);
        };
      }
    }
    // a new context: the old texture went with the old one
    this.profitTex = null;
    if (this.profitData) this.setProfit(this.profitData);
    this.strokeProg = compile(gl, STROKE_VS, STROKE_FS);
    for (const a of ["a_seg", "a_colour", "a_edge", "a_level", "a_flags", "a_speed", "a_lane", "a_wmul", "a_full", "a_band", "a_band_m"]) this.strokeLoc[a] = gl.getAttribLocation(this.strokeProg, a);
    this.stationProg = compile(gl, STATION_VS, STATION_FS);
    this.marks.init(gl);
    this.oneWay.init(gl);
    this.oneWayTrack.init(gl);
    this.lots.init(gl);
    this.floors.init(gl);
    this.bubbles.init(gl);
    this.buses.init(gl);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.hasStencil = (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0;
    this.strokeU = uniforms(gl, this.strokeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_casing", "u_casing_rgb", "u_cap", "u_kind", "u_sel_line", "u_sel_n", "u_sel_set", "u_sel_edge", "u_hot", "u_hide", "u_only_sel", "u_bp_part", "u_lanes", "u_lane_mode", "u_pass_off","u_dim", "u_by_level", "u_by_speed", "u_by_profit", "u_profit", "u_profit_n", "u_level_colours", "u_track", "u_bad", "u_sel", "u_bp", "u_bp_mid", "u_bp_rail", "u_bp_off_mid", "u_bp_off_rail", "u_bp_off_fade", "u_line_colours", "u_full_pass", "u_full_px", "u_full_out", "u_full_dash", "u_full_rgb", "u_full_inside", "u_ghost", "u_ghost_layer", "u_ghost_rgb", "u_ghost_min", "u_hsr", "u_wire", "u_band", "u_band_traffic", "u_band_gap", "u_band_m", "u_mute"]);
    this.stationU = uniforms(gl, this.stationProg, [...PROJECT_UNIFORMS, "u_viewport", "u_radius", "u_ring", "u_ink", "u_line", "u_boxes", "u_fixed", "u_pass_half", "u_pass_in", "u_pass", "u_layer", "u_pixel", "u_bad", "u_crowd_rgb", "u_tint", "u_mask", "u_stline", "u_lanes", "u_hot_st", "u_hot_n", "u_room", "u_bld"]);
    this.bridgeProg = compile(gl, BRIDGE_VS, STATION_FS);
    this.bridgeU = uniforms(gl, this.bridgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_ring", "u_bridge", "u_pass", "u_layer", "u_ink", "u_bad", "u_crowd_rgb", "u_mask", "u_hot_st", "u_hot_n"]);
    this.selProg = compile(gl, SEL_VS, SEL_FS);
    this.selU = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_shapes", "u_count", "u_dpr", "u_height", "u_ink", "u_outline", "u_glow"]);
    this.selTex = this.texture(this.selData, "rgba32f", SEL_MAX * SEL_STRIDE, []);
    this.trainU = uniforms(gl, this.trainProg, [...PROJECT_UNIFORMS, "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_phases", "u_meta", "u_samples", "u_lines", "u_line_colours", "u_fill", "u_bad", "u_detail", "u_dpr", "u_cars", "u_loads", "u_stretch", "u_sel_line", "u_sel_n", "u_sel_set", "u_dim", "u_sel_trip", "u_pass", "u_ink", "u_band", "u_psi0", "u_lane_tex", "u_lane", "u_hollow", "u_bands", "u_bands_m", "u_passing"]);
    this.timer = new GpuTimer(gl);
    this.tripBuf = gl.createBuffer();
    this.endBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.endBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1e9]), gl.DYNAMIC_DRAW);
    this.loadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.loadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 0]), gl.DYNAMIC_DRAW);
    this.standBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.standBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, -1]), gl.DYNAMIC_DRAW);
    this.lagBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lagBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(1), gl.DYNAMIC_DRAW);
    this.bayBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bayBuf);
    gl.bufferData(gl.ARRAY_BUFFER, noBays(1), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.loadTex =this.texture(new Float32Array(1), "r32f", TEX_W, []);
  }

  /** T-2420: each edge id's stretch fares over costs for the Profit view (`stretchMoney`'s
   * `ratio`, `NO_PROFIT` where none), or null for none at all. */
  setProfit(ratio: Float32Array | null) {
    const gl = this.gl;
    this.profitData = ratio;
    if (!gl) return;
    if (this.profitTex) gl.deleteTexture(this.profitTex);
    this.profitTex = ratio?.length ? this.texture(ratio, "r32f", TEX_W, []) : null;
    this.profitN = this.profitTex ? ratio!.length : 0;
  }

  private texture(data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number, list = this.res): WebGLTexture {
    const gl = this.gl;
    const comps = kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4;
    const texels = Math.max(1, Math.ceil((data as Float32Array).length / comps));
    const height = Math.max(1, Math.ceil(texels / width));
    const padded = kind === "rgba8" ? new Uint8Array(width * height * 4) : new Float32Array(width * height * comps);
    padded.set(data as any);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    if (kind === "r32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded);
    else if (kind === "rg32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, height, 0, gl.RG, gl.FLOAT, padded);
    else if (kind === "rgba32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, padded);
    list.push({ del: () => gl.deleteTexture(tex) });
    return tex;
  }

  private buffer(data: ArrayBufferView, list = this.res): WebGLBuffer {
    const gl = this.gl;
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    // left bound, WebGL refuses it as a transform feedback target (the probe)
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    list.push({ del: () => gl.deleteBuffer(b) });
    return b;
  }

  private attrib(prog: WebGLProgram, name: string, buf: WebGLBuffer, size: number, type: number, divisor: number, integer = false) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    if (integer) gl.vertexAttribIPointer(loc, size, type, 0, 0);
    else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
    gl.vertexAttribDivisor(loc, divisor);
  }

  /** A stroke set's VAO. `from` (a network's strokes, T-1028): the strokes drawn before and the runs
   * (start, count, old start or -1, `NetPatch`) that make these from them: unchanged runs are copied
   * on the GPU from the old buffers, only the new segments uploaded. `more`: the `a_wmul` and `a_full`
   * values when worked out already, or "later" for an `a_full` written by its caller (T-2403). */
  private strokes(s: Strokes, list = this.res, from?: { old: StrokeVao; runs: Int32Array } | null, more?: { wmul?: Float32Array | null; full?: Float32Array | "later" }): StrokeVao {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    list.push({ del: () => gl.deleteVertexArray(vao) });
    gl.bindVertexArray(vao);
    const p = this.strokeProg;
    const n = s.count;
    const attribs: { buf: WebGLBuffer; bytes: number }[] = [];
    const patch = from && n && from.old.attribs?.length === STROKE_ATTRIBS.length ? from : null;
    /** an attribute's buffer: `get(start, count)` its values for those segments, `per` a segment */
    const make = (get: (start: number, count: number) => ArrayBufferView, per: number, bpe: number, empty: ArrayBufferView): WebGLBuffer => {
      const bytes = per * bpe;
      let buf: WebGLBuffer;
      if (!n) buf = this.buffer(empty, list);
      else if (!patch) buf = this.buffer(get(0, n), list);
      else {
        buf = gl.createBuffer()!;
        list.push({ del: () => gl.deleteBuffer(buf) });
        gl.bindBuffer(gl.COPY_WRITE_BUFFER, buf);
        gl.bufferData(gl.COPY_WRITE_BUFFER, n * bytes, gl.STATIC_DRAW);
        gl.bindBuffer(gl.COPY_READ_BUFFER, patch.old.attribs![attribs.length].buf);
        const r = patch.runs;
        for (let k = 0; k < r.length; k += 3) {
          if (r[k + 2] >= 0) gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, r[k + 2] * bytes, r[k] * bytes, r[k + 1] * bytes);
          else gl.bufferSubData(gl.COPY_WRITE_BUFFER, r[k] * bytes, get(r[k], r[k + 1]));
        }
        gl.bindBuffer(gl.COPY_READ_BUFFER, null);
        gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
      }
      attribs.push({ buf, bytes });
      return buf;
    };
    const sub = <T extends Float32Array | Uint32Array | Uint8Array | Int16Array>(a: T, per: number) => (start: number, count: number) => a.subarray(start * per, (start + count) * per);
    const speed = s.speed;
    const speedOf = speed ? (start: number, count: number) => shownSpeed(speed.subarray(start, start + count), s.trainSpeed?.length === speed.length ? s.trainSpeed.subarray(start, start + count) : undefined) : (_: number, count: number) => new Float32Array(count).fill(160);
    this.attrib(p, "a_seg", make(sub(s.seg, 4), 4, 4, new Float32Array(4)), 4, gl.FLOAT, 1);
    this.attrib(p, "a_colour", make(sub(s.colour, 1), 1, 4, new Uint32Array(1)), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_edge", make(sub(s.edge, 1), 1, 4, new Uint32Array(1)), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_level", make(sub(s.level, 1), 1, 4, new Float32Array(1)), 1, gl.FLOAT, 1);
    this.attrib(p, "a_flags", make(sub(s.flags, 1), 1, 4, new Float32Array(1)), 1, gl.FLOAT, 1);
    this.attrib(p, "a_speed", make(speedOf, 1, 4, new Float32Array(1).fill(160)), 1, gl.FLOAT, 1);
    const wmulBuf = this.buffer(more?.wmul ?? plainWmul(Math.max(1, n)), list);
    this.attrib(p, "a_wmul", wmulBuf, 3, gl.FLOAT, 1);
    let fullBuf: WebGLBuffer;
    if (more?.full instanceof Float32Array) fullBuf = this.buffer(more.full, list);
    else if (more?.full === "later") {
      fullBuf = gl.createBuffer()!;
      list.push({ del: () => gl.deleteBuffer(fullBuf) });
      gl.bindBuffer(gl.ARRAY_BUFFER, fullBuf);
      gl.bufferData(gl.ARRAY_BUFFER, Math.max(1, n) * 8, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    } else if (patch) {
      // as the attributes: a run's values depend on its own segments alone (`fullAttr`)
      fullBuf = gl.createBuffer()!;
      list.push({ del: () => gl.deleteBuffer(fullBuf) });
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, fullBuf);
      gl.bufferData(gl.COPY_WRITE_BUFFER, n * 8, gl.STATIC_DRAW);
      gl.bindBuffer(gl.COPY_READ_BUFFER, patch.old.fullBuf);
      const r = patch.runs;
      for (let k = 0; k < r.length; k += 3) {
        if (r[k + 2] >= 0) gl.copyBufferSubData(gl.COPY_READ_BUFFER, gl.COPY_WRITE_BUFFER, r[k + 2] * 8, r[k] * 8, r[k + 1] * 8);
        else gl.bufferSubData(gl.COPY_WRITE_BUFFER, r[k] * 8, fullAttr(s, r[k + 1], undefined, r[k]));
      }
      gl.bindBuffer(gl.COPY_READ_BUFFER, null);
      gl.bindBuffer(gl.COPY_WRITE_BUFFER, null);
    } else fullBuf = this.buffer(fullAttr(s, n), list);
    this.attrib(p, "a_full", fullBuf, 2, gl.FLOAT, 1);
    const lane = s.lane;
    const laneBuf = make(lane ? sub(lane, 2) : (_, count) => new Uint8Array(count * 2).fill(255), 2, 1, new Uint8Array(2).fill(255));
    const loc = gl.getAttribLocation(p, "a_lane");
    if (loc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, laneBuf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    }
    // lines side by side (T-1799): none on track and the other stroke sets
    const band = s.band;
    const bandBuf = make(band ? sub(band, BAND_STRIDE) : (_, count) => new Int16Array(count * BAND_STRIDE), BAND_STRIDE, 2, new Int16Array(BAND_STRIDE));
    // the shifts and widths, then the metres to the ribbon's middle (T-2132), from one buffer
    for (const [name, size, off] of [["a_band", 4, 0], ["a_band_m", 2, 8]] as const) {
      const bloc = gl.getAttribLocation(p, name);
      if (bloc < 0) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, bandBuf);
      gl.enableVertexAttribArray(bloc);
      gl.vertexAttribPointer(bloc, size, gl.SHORT, true, BAND_STRIDE * 2, off);
      gl.vertexAttribDivisor(bloc, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { vao, count: n, wmulBuf, fullBuf, attribs, res: list };
  }

  /** `?patch=check` (T-1029): each network's strokes read back from the GPU after an edit and
   * compared with the arrays they were made from. */
  checkPatch = false;
  /** networks taken as changes to the one drawn (checks read it) */
  patched = 0;
  private checkStrokes(name: string, v: StrokeVao | null, s: Strokes) {
    if (!v?.attribs || !s.count) return;
    const gl = this.gl, n = s.count;
    const want: ArrayBufferView[] = [s.seg, s.colour, s.edge, s.level, s.flags, s.speed ? shownSpeed(s.speed, s.trainSpeed) : new Float32Array(n).fill(160), s.lane ?? new Uint8Array(n * 2).fill(255), s.band ?? new Int16Array(n * BAND_STRIDE)];
    const bad: string[] = [];
    v.attribs.forEach(({ buf, bytes }, i) => {
      const got = new Uint8Array(n * bytes), w = want[i];
      gl.bindBuffer(gl.COPY_READ_BUFFER, buf);
      gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
      const x = new Uint8Array(w.buffer, w.byteOffset, w.byteLength);
      if (x.length !== got.length || x.some((b, k) => b !== got[k])) bad.push(STROKE_ATTRIBS[i]);
    });
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    if (bad.length) console.error(`T-1028 check: the ${name} strokes patched on the GPU differ from the network's in ${bad.join(", ")}`);
  }

  /** `?patch=check` (T-2403): the track's `a_full` patched by runs against one worked out whole */
  private checkFull(v: StrokeVao | null, s: Strokes) {
    if (!v || !s.count) return;
    const gl = this.gl, want = fullAttr(s, s.count), got = new Float32Array(want.length);
    gl.bindBuffer(gl.COPY_READ_BUFFER, v.fullBuf);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, got);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    const x = new Uint32Array(want.buffer), y = new Uint32Array(got.buffer);
    if (x.some((b, k) => b !== y[k])) console.error("T-2403 check: the track's a_full patched by runs differs from a whole one");
  }

  /** the network's stroke buffers on the GPU and their sizes (checks read them back) */
  gpuDump(): Record<string, [WebGLBuffer, number]> {
    const out: Record<string, [WebGLBuffer, number]> = {};
    for (const [name, v] of [["track", this.track], ["lines", this.lines]] as const)
      v?.attribs?.forEach(({ buf, bytes }, i) => (out[`${name}.${STROKE_ATTRIBS[i]}`] = [buf, Math.max(1, v.count) * bytes]));
    return out;
  }

  /** T-1639: what to draw of network stroke set `s` (made from `src`) this frame: its simplified copy
   * for the zoom band when that has fewer segments, and the instance ranges near the view (null: all). */
  private pickStrokes(s: StrokeVao | null, src: Strokes, lines: boolean, zoom: number, dpr: number, grow: number, project: Project, w: number, h: number): { vao: StrokeVao; ranges: number[] | null } | null {
    if (!s || !s.count) return null;
    let st = this.lodOf.get(s);
    if (!st) this.lodOf.set(s, (st = { boxes: null, speed: undefined, sig: null, sigAt: 0, working: false, bands: new Map(), making: null }));
    let vao = s, boxes = st.boxes, used: StrokeBand | null = null;
    const bz = Math.ceil(zoom + Math.log2(dpr) - 1e-6);
    if (this.strokeLod && grow === 0 && bz < LOD_TOP_ZOOM && (st.sig && st.sigAt >= src.count || this.workLod(s, st, src))) {
      // the full set is drawn until the band is made (T-2403)
      const band = st.bands.get(bz) ?? this.workBand(s, st, src, bz, lines);
      if (band && band.vao.count < s.count) {
        if (lines && band.v !== this.perSegV) {
          band.v = this.perSegV;
          band.wm = this.bandWmul(band.first);
          band.fu = this.bandFull(band.strokes, band.first, 0, band.first.length, this.linesFull);
          this.rewrite(band.vao.wmulBuf, band.wm ?? plainWmul(band.first.length));
          this.rewrite(band.vao.fullBuf, band.fu);
        }
        vao = band.vao;
        boxes = band.boxes;
        used = band;
      }
    }
    if (!this.strokeCull || zoom < CULL_ZOOM) return { vao, ranges: null };
    if (vao === s) boxes = st.boxes ??= chunkBoxes(src.seg, src.count);
    const ranges = visibleRanges(boxes!, vao.count, project, w, h, CULL_MARGIN_PX, cullLevel(zoom));
    if (!ranges) return { vao, ranges: null };
    let total = 0;
    for (let i = 1; i < ranges.length; i += 2) total += ranges[i];
    if (!total) return { vao, ranges };
    if (!this.strokeGather || total > GATHER_MAX) return { vao, ranges: mergeRanges(ranges, RANGE_GAP) };
    let g = this.gathers.get(s);
    if (!g) this.gathers.set(s, (g = new StrokeGather(this.gl, (name) => this.strokeLoc[name])));
    const b = used?.strokes ?? src;
    g.fill({ seg: b.seg, colour: b.colour, edge: b.edge, level: b.level, flags: b.flags, speed: this.shownOf(b), lane: b.lane, band: b.band, wmul: lines ? (used ? used.wm : this.linesWmul) : null, full: lines ? (used ? used.fu : this.linesFull) : null }, ranges, total);
    return { vao: g, ranges: null };
  }

  /** a stroke set's speeds as its buffer holds them, kept per set */
  private shownOf(s: Strokes): Float32Array {
    let v = this.speedShown.get(s);
    if (!v) this.speedShown.set(s, (v = s.speed ? shownSpeed(s.speed, s.trainSpeed?.length === s.speed.length ? s.trainSpeed : undefined) : new Float32Array(s.count).fill(160)));
    return v;
  }

  /** A line band's `a_wmul`: a merged segment's width and shift at its start are its first segment's,
   * at its end its last's; null for plain widths. */
  private bandWmul(f: Uint32Array): Float32Array | null {
    const wm = this.linesWmul;
    if (!wm) return null;
    const m = f.length, all = wm.length / 3, out = new Float32Array(m * 3);
    for (let k = 0; k < m; k++) {
      const i = f[k], last = (k + 1 < m ? f[k + 1] : all) - 1;
      out[k * 3] = wm[i * 3];
      out[k * 3 + 1] = wm[i * 3 + 1];
      out[k * 3 + 2] = wm[last * 3 + 2];
    }
    return out;
  }
  /** A band's `a_full` for merged segments `from` to `to` into `out`, the traffic marks from `fu`
   * (the full set's) when given; `out` made when absent. */
  private bandFull(b: Strokes, f: Uint32Array, from: number, to: number, fu: Float32Array | null, out?: Float32Array): Float32Array {
    const part = fullAttr(b, to - from, fu ? (k) => fu[f[k] * 2] : undefined, from);
    if (!out) {
      if (from === 0 && to === f.length) return part;
      out = new Float32Array(Math.max(1, f.length) * 2);
    }
    out.set(part.subarray(0, (to - from) * 2), from * 2);
    return out;
  }

  /** T-2403: a zoom band's simplified copy made in 8 ms slices between frames after the significance
   * (`workLod`), one band at a time a set; null until it is made. `fullAttr` starts a run at each
   * slice, so the slices of `a_full` end only where a run of the band starts. */
  private workBand(s: StrokeVao, st: StrokeLodState, src: Strokes, bz: number, lines: boolean): StrokeBand | null {
    if (st.making !== null) return null;
    st.making = bz;
    const t0 = performance.now();
    const job = simplifiedLater(src, st.sig!, LOD_TOL_PX / pxPerUnitAt(bz), st.speed ?? null);
    const m = job.strokes.count, b = job.strokes, f = job.first;
    const SLICE = 16384;
    let phase = 0, at = 0, boxes: ChunkBoxes | null = null, v = -1, wm: Float32Array | null = null, fuSrc: Float32Array | null = null, full: Float32Array | null = null;
    // where a run of the band's own starts (`fullAttr`): slices of `a_full` end only there
    const runEnd = (k: number) => {
      while (k < m && k > 0 && b.colour[k] === b.colour[k - 1] && b.edge[k] === b.edge[k - 1] && b.seg[k * 4] === b.seg[k * 4 - 2] && b.seg[k * 4 + 1] === b.seg[k * 4 - 1]) k++;
      return Math.min(k, m);
    };
    const step = (): boolean => {
      if (phase === 0) {
        job.fill(at, at + SLICE);
        at += SLICE;
        if (at >= m) phase = 1, at = 0;
      } else if (phase === 1) {
        boxes = chunkBoxes(b.seg, m);
        if (lines) { v = this.perSegV; wm = this.bandWmul(f); fuSrc = this.linesFull; }
        full = new Float32Array(Math.max(1, m) * 2);
        phase = 2;
      } else if (phase === 2) {
        const to = runEnd(Math.min(m, at + SLICE));
        this.bandFull(b, f, at, to, lines ? fuSrc : null, full!);
        at = to;
        if (at >= m) phase = 3;
      } else {
        const res: { del: () => void }[] = [];
        const vao = this.strokes(b, res, null, { wmul: wm, full: full! });
        st.bands.set(bz, { vao, strokes: b, first: f, boxes: boxes!, v, res, wm: lines ? wm : null, fu: lines ? full : null });
        st.making = null;
        return true;
      }
      return false;
    };
    const run = () => {
      if (this.lodOf.get(s) !== st) return;
      const t1 = performance.now();
      let done = false;
      while (!done && performance.now() - t1 < 8) done = step();
      this.lodMs += performance.now() - t1;
      if (!done) setTimeout(run, 0);
    };
    this.lodMs += performance.now() - t0;
    setTimeout(run, 0);
    return null;
  }

  /** T-1639: the runs' significance worked out in 8 ms slices between frames, so an edit's first
   * draw (the full set, as before) is not held up by it (~80 ms a set on the all-Japan save); false
   * until it is done. */
  private workLod(s: StrokeVao, st: StrokeLodState, src: Strokes): false {
    if (st.working) return false;
    st.working = true;
    const step = () => {
      if (this.lodOf.get(s) !== st) return;
      const t0 = performance.now();
      if (st.speed === undefined) st.speed = src.speed ? shownSpeed(src.speed, src.trainSpeed) : null;
      st.sig ??= new Float32Array(src.count);
      st.sigAt = significance(src, st.speed, st.sig, st.sigAt, t0 + 8);
      this.lodMs += performance.now() - t0;
      // the picture drawn meanwhile is the same, so nothing is redrawn when done: the next
      // stroke redraw takes the copy
      if (st.sigAt < src.count) setTimeout(step, 0);
      else st.working = false;
    };
    setTimeout(step, 0);
    return false;
  }

  /** Point stroke set `v`'s attributes (its VAO bound) at instance `start`: WebGL2 draws have no
   * first instance. */
  private bindAt(v: StrokeVao, start: number) {
    const gl = this.gl, a = v.attribs!;
    const set = (name: string, buf: WebGLBuffer, size: number, type: number, bytes: number, int = false, norm = false, stride = 0, off = 0) => {
      const loc = this.strokeLoc[name];
      if (loc < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      if (int) gl.vertexAttribIPointer(loc, size, type, stride, start * bytes + off);
      else gl.vertexAttribPointer(loc, size, type, norm, stride, start * bytes + off);
    };
    set("a_seg", a[0].buf, 4, gl.FLOAT, 16);
    set("a_colour", a[1].buf, 1, gl.UNSIGNED_INT, 4, true);
    set("a_edge", a[2].buf, 1, gl.UNSIGNED_INT, 4, true);
    set("a_level", a[3].buf, 1, gl.FLOAT, 4);
    set("a_flags", a[4].buf, 1, gl.FLOAT, 4);
    set("a_speed", a[5].buf, 1, gl.FLOAT, 4);
    set("a_lane", a[6].buf, 2, gl.UNSIGNED_BYTE, 2, false, true);
    set("a_band", a[7].buf, 4, gl.SHORT, BAND_STRIDE * 2, false, true, BAND_STRIDE * 2);
    set("a_band_m", a[7].buf, 2, gl.SHORT, BAND_STRIDE * 2, false, true, BAND_STRIDE * 2, 8);
    set("a_wmul", v.wmulBuf, 3, gl.FLOAT, 12);
    set("a_full", v.fullBuf, 2, gl.FLOAT, 8);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private dropLod() {
    for (const st of this.lodOf.values()) for (const b of st.bands.values()) for (const r of b.res) r.del();
    this.lodOf.clear();
    for (const g of this.gathers.values()) g.dispose();
    this.gathers.clear();
  }

  /** Rewrite a dynamic buffer in place (sizes never change between `setNetwork`s). */
  private rewrite(buf: WebGLBuffer, data: Float32Array) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** The demand views (T-084) for the network now drawn, or null for the plain drawing. The
   * arrays must be sized for `this.net` (map/demandLayers.ts builds them from it). */
  setLayers(l: NetworkLayers | null) {
    const n = this.net;
    if (!n || !this.lines) return;
    this.layersSrc = l;
    const lines = this.lines;
    const ok = !!l && l.lineWidth.length === lines.count;
    const tb = ok && l.bands && l.bands.seg.length === lines.count * 2 ? l.bands : null;
    let wm: Float32Array | null = null;
    if (ok) {
      wm = new Float32Array(lines.count * 3);
      for (let i = 0; i < lines.count; i++) {
        wm[i * 3] = l.lineWidth[i];
        if (tb) (wm[i * 3 + 1] = tb.seg[i * 2]), (wm[i * 3 + 2] = tb.seg[i * 2 + 1]);
      }
    }
    if (lines.count) this.rewrite(lines.wmulBuf, wm ?? plainWmul(lines.count));
    this.lineWidths = ok ? l.lineWidth : null;
    this.linesWmul = wm;
    this.trafficWidths = !!l?.traffic;
    // trains on their Traffic view band (T-1799 stage 2): the lane texture's band channel
    if (tb !== this.trafficBands) {
      this.trafficBands = tb;
      this.trafficReach = 0;
      if (tb) for (let i = 0; i < tb.sample.length; i += 2) this.trafficReach = Math.max(this.trafficReach, Math.abs(tb.sample[i]));
      this.putLane(n);
    }
    this.perSegV++;
    const samples = Math.max(1, n.samples.length / 2);
    const upload = (tex: WebGLTexture | null, data: Float32Array, kind: "r32f" | "rg32f") => {
      const gl = this.gl;
      const comps = kind === "r32f" ? 1 : 2;
      const height = Math.max(1, Math.ceil(samples / TEX_W));
      const padded = new Float32Array(TEX_W * height * comps);
      padded.set(data.subarray(0, Math.min(data.length, padded.length)));
      if (kind === "rg32f" && !l) padded.fill(-1);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, height, kind === "r32f" ? gl.RED : gl.RG, gl.FLOAT, padded);
    };
    upload(this.tex.fill, l ? l.fill : new Float32Array(samples * 2), "rg32f");
  }

  /** T-819, T-2236: in the traffic view (`traffic`), the edges whose track is at capacity, drawn
   * with a red edge and a darker body on the lines over them, and the crowded stretches (the
   * boarding account's, each hour), whose red edge and darkening are stronger as their busiest
   * hour's load passes the rated load (rendererShaders.ts `FULL_FLOOR`). */
  setFull(edges: ReadonlySet<number> | null, traffic: boolean) {
    const was = this.fullEdges;
    const same = (!edges || !edges.size) === (!was || !was.size) && (!edges || !was || (edges.size === was.size && [...edges].every((e) => was.has(e))));
    if (same && traffic === this.fullTraffic) return;
    this.fullEdges = edges;
    this.fullTraffic = traffic;
    this.writeFull();
  }
  private fullEdges: ReadonlySet<number> | null = null;
  private fullTraffic = false;
  /** the crowded stretches as last written, line * 4096 + stretch: the busiest hour's load as a
   * share of the rated load and that hour (the line's local time) */
  private fullKeys = new Map<number, CrowdedStretch>();
  private fullDrawn = "";
  /** The crowding drawn on line `line`'s run 0 stretch `s` (T-2236); null where none is. */
  fullStretch(line: number, s: number): CrowdedStretch | null {
    return this.fullTraffic ? this.fullKeys.get(line * 4096 + s) ?? null : null;
  }
  private writeFull() {
    const l = this.lines, n = this.net;
    const keys = new Map<number, CrowdedStretch>();
    const b = this.board, fs = this.fullTraffic ? b?.crowdStretches : undefined;
    for (let k = 0; b && fs && k < fs.length; k += 2) {
      const j = k / 2;
      keys.set(fs[k] * 4096 + fs[k + 1], { load: b.crowdLoads[j] ?? 0, hour: b.crowdHours[j] ?? 0 });
    }
    const drawn = `${n ? 1 : 0}/${l?.count}/${[...(this.fullEdges ?? [])].join(",")}/${[...keys].map(([k, v]) => `${k}:${v.load}:${v.hour}`).join(",")}`;
    this.fullKeys = keys;
    if (!l || !l.count || !n || drawn === this.fullDrawn) return;
    this.fullDrawn = drawn;
    // the shader's a_full: -1 track at capacity, else the stretch's load over the rated load
    const f = this.fullEdges, stretch = n.lines.stretch;
    let any = false;
    const out = fullAttr(n.lines, l.count, (i) => {
      if (f?.size && f.has(n.lines.edge[i])) return (any = true), -1;
      const s = keys.size && stretch ? keys.get(n.lines.colour[i] * 4096 + stretch[i]) : undefined;
      const v = s ? s.load : 0;
      if (v !== 0) any = true;
      return v;
    });
    this.anyFull = any;
    this.rewrite(l.fullBuf, out);
    this.linesFull = out;
    this.perSegV++;
  }
  private anyFull = false;

  /** Replace everything drawn but the trips (an edit). */
  setNetwork(n: NetworkBuffers) {
    const gl = this.gl;
    this.dropPlaced(); // T-1407: the network drawn now has the segment its stand-in showed
    // T-1028: an edit's network posted as changes to the one drawn keeps the strokes' unchanged runs
    // on the GPU and the textures whose data is the same; anything else is uploaded whole
    const was = this.net, p = n.patch;
    const patch = p && was && was !== n && was.serial === p.base && this.track?.count === p.oldTrack && this.lines?.count === p.oldLines ? p : null;
    if (patch) this.patched++;
    for (const r of this.res) r.del();
    this.res = [];
    this.probeVao = null;
    this.net = n;
    this.hide = [n.edgePts.has(this.hide[0]) ? this.hide[0] : -2, n.edgePts.has(this.hide[1]) ? this.hide[1] : -2];
    this.guard.setNetwork(n);
    this.marks.setNetwork(n.marks);
    this.oneWay.setNetwork(n.oneWay);
    this.oneWayTrack.setNetwork(n.oneWayTrack);
    this.lots.setNetwork(n.lots);
    this.floors.setNetwork(n.buildings);
    const oldTrack = this.track, oldLines = this.lines;
    this.track = this.strokes(n.track, [], patch && oldTrack ? { old: oldTrack, runs: patch.track } : null);
    // its a_full is `writeFull`'s, just below
    this.lines = this.strokes(n.lines, [], patch && oldLines ? { old: oldLines, runs: patch.lines } : null, { full: "later" });
    for (const r of [...(oldTrack?.res ?? []), ...(oldLines?.res ?? [])]) r.del();
    this.dropLod();
    this.linesWmul = this.linesFull = null;
    this.perSegV++;
    if (patch && this.checkPatch) {
      this.checkStrokes("track", this.track, n.track);
      this.checkStrokes("lines", this.lines, n.lines);
      this.checkFull(this.track, n.track);
    }
    this.fullDrawn = "";
    this.writeFull();
    this.turnback = n.turnbacks?.count ? this.strokes(n.turnbacks) : null;
    // blueprint track sorts last (network.ts)
    this.hasBlueprint = n.track.count > 0 && (n.track.flags[n.track.count - 1] & F_BLUEPRINT) !== 0;
    this.annexes = 0;
    for (let i = 0; i < n.stationCount; i++) this.annexes |= n.stations[i * 3 + 2] & ST_ROOM;
    this.lineWidths = null;
    const same = new Set(patch?.same ?? []);
    const put = (name: keyof typeof this.tex, field: string | string[] | null, make: () => [ArrayBufferView, "r32f" | "rg32f" | "rgba32f" | "rgba8", number]) => {
      if (field && [field].flat().every((f) => same.has(f)) && this.tex[name]) return;
      this.putTexture(name, ...make());
    };
    put("phases", "phases", () => [n.phases.length ? n.phases : new Float32Array(4), "rgba32f", TEX_W]);
    put("meta", "meta", () => [n.meta.length ? n.meta : new Float32Array(4), "rgba32f", TEX_W]);
    put("samples", "samples", () => [n.samples.length ? n.samples : new Float32Array(2), "rg32f", TEX_W]);
    // the Traffic view's bands were the last network's
    const hadTraffic = this.trafficBands !== null;
    this.trafficBands = null;
    const passChanged = n.samplePass ? !same.has("samplePass") : this.hadSamplePass;
    this.hadSamplePass = !!n.samplePass;
    this.passing = false;
    for (let i = 0; i < n.stationCount && !this.passing; i++) this.passing = (n.stations[i * 3 + 2] & ST_PASSING) !== 0;
    if (hadTraffic || passChanged || !(same.has("sampleLane") && same.has("sampleBand") && this.tex.lane)) this.putLane(n);
    put("lines", "lineTable", () => [n.lineTable.length ? n.lineTable : new Float32Array(4), "rgba32f", LINES_W]);
    put("colours", "lineColours", () => [n.lineColours, "rgba8", LINES_W]);
    put("fill", null, () => [new Float32Array(Math.max(1, n.samples.length / 2) * 2).fill(-1), "rg32f", TEX_W]);
    const cars = n.lineCars ?? new Float32Array(Math.max(1, n.lineTable.length / 4)).fill(8);
    this.maxCars = Math.max(1, ...cars);
    put("cars", "lineCars", () => [cars, "r32f", LINES_W]);
    // each train sample's stop-to-stop stretch (run 0's numbering), where a trip's loads are
    // looked up (T-164)
    put("stretch", "sampleStretch", () => {
      if (n.sampleStretch) return [n.sampleStretch, "r32f", TEX_W];
      const nSamples = Math.max(1, n.samples.length / 2);
      const stretch = new Float32Array(nSamples).fill(-1);
      if (n.sampleStroke && n.lines.stretch)
        for (let k = 0; k < nSamples; k++) {
          const i = n.sampleStroke[k];
          if (i >= 0) stretch[k] = n.lines.stretch[i];
        }
      return [stretch, "r32f", TEX_W];
    });
    this.crowds = new PlatformCrowds(n.stationCount);

    this.stationVao = gl.createVertexArray()!;
    const sv = this.stationVao;
    this.res.push({ del: () => gl.deleteVertexArray(sv) });
    gl.bindVertexArray(this.stationVao);
    this.attrib(this.stationProg, "a_st", this.buffer(n.stations.length ? n.stations : new Float32Array(3)), 3, gl.FLOAT, 1);
    this.attrib(this.stationProg, "a_axis", this.buffer(n.stationAxis?.length ? n.stationAxis : new Float32Array(Math.max(1,n.stationCount)*2)),2,gl.FLOAT,1);
    this.stationShapeBuf = this.buffer(n.stationShape?.length ? n.stationShape : new Float32Array(Math.max(1, n.stationCount) * 4));
    // each platform's track line (T-254), or the arc of its shape where the buffers have none
    const P2 = 2 * PLATFORM_POINTS;
    if (!n.stationLine || n.stationLine.length < n.stationCount * P2) {
      const line = new Float32Array(n.stationCount * P2);
      for (let i = 0; i < n.stationCount; i++)
        line.set(platformPoints(platformShapeAt(n, i, false)), i * P2);
      n.stationLine = line;
    }
    put("stationLine", "stationLine", () => [n.stationLine!.length ? n.stationLine! : new Float32Array(2), "rg32f", TEX_W]);
    this.attrib(this.stationProg, "a_shape", this.stationShapeBuf, 4, gl.FLOAT, 1);
    this.stationSelBuf = this.buffer(new Float32Array(Math.max(1, n.stationCount)));
    this.attrib(this.stationProg, "a_sel", this.stationSelBuf, 1, gl.FLOAT, 1);
    this.groupSrc = { groups: null, net: null };
    this.crowdBuf = this.buffer(this.crowds.crowds);
    this.dotCrowdBuf = this.buffer(this.crowds.dotCrowds);
    this.attrib(this.stationProg, "a_dcrowd", this.dotCrowdBuf, 2, gl.FLOAT, 1);
    this.attrib(this.stationProg, "a_crowd", this.crowdBuf, 4, gl.FLOAT, 1);
    this.selectedAlso = new Set();
    this.selectedAlsoKey = "";
    this.selDrawnKey = "";
    // the player's links as bridges (T-217)
    const links = n.stationLinkIdx?.length ?? 0;
    this.bridgeVao = gl.createVertexArray()!;
    const bv = this.bridgeVao;
    this.res.push({ del: () => gl.deleteVertexArray(bv) });
    gl.bindVertexArray(bv);
    this.attrib(this.bridgeProg, "a_ends", this.buffer(links ? n.stationLinks! : new Float32Array(4)), 4, gl.FLOAT, 1);
    this.bridgeFlagBuf = this.buffer(new Float32Array(Math.max(1, links / 2) * 2));
    this.attrib(this.bridgeProg, "a_bflag", this.bridgeFlagBuf, 2, gl.FLOAT, 1);
    // zoomed out the connectors between dots (T-313) are built by syncGroups

    this.trainVao = gl.createVertexArray()!;
    const tv = this.trainVao;
    this.res.push({ del: () => gl.deleteVertexArray(tv) });
    gl.bindVertexArray(this.trainVao);
    this.attrib(this.trainProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_end", this.endBuf!, 1, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_load", this.loadBuf!, 2, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_stand", this.standBuf!, 4, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_lag", this.lagBuf!, 1, gl.FLOAT, 1);
    this.attrib(this.trainProg, "a_bay", this.bayBuf!, 4, gl.FLOAT, 1);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Only the service changed (a train count, T-229): the run profiles' phase tables and meta are
   * replaced and every other buffer kept, rather than uploading the whole network again. */
  setService(phases: Float32Array, meta: Float32Array) {
    const n = this.net;
    if (!n) return;
    n.phases = phases;
    n.meta = meta;
    this.putTexture("phases", phases.length ? phases : new Float32Array(4), "rgba32f", TEX_W);
    this.putTexture("meta", meta.length ? meta : new Float32Array(4), "rgba32f", TEX_W);
  }

  /** each network texture's GL object, freed when it is replaced (T-1028: kept across edits while its
   * data is the same) */
  private texRes: Partial<Record<TexName, { del: () => void }[]>> = {};
  /** per sample: how far apart its tracks are (T-658) and its line's band's shift as drawn (T-1799),
   * line widths and metres (T-2132) */
  private putLane(n: NetworkBuffers) {
    const m = Math.max(1, n.samples.length / 2), lane = n.sampleLane, band = this.trafficBands?.sample ?? n.sampleBand, v = new Float32Array(m * 4);
    for (let k = 0; k < m; k++) {
      v[k * 4] = lane?.length ? lane[k] ?? 1 : 1;
      v[k * 4 + 1] = band?.[k * 2] ?? 0;
      v[k * 4 + 2] = band?.[k * 2 + 1] ?? 0;
      v[k * 4 + 3] = n.samplePass?.[k] ?? 1e9;
    }
    this.putTexture("lane", v, "rgba32f", TEX_W);
  }

  private putTexture(name: TexName, data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number) {
    for (const r of this.texRes[name] ?? []) r.del();
    const list: { del: () => void }[] = [];
    this.tex[name] = this.texture(data, kind, width, list);
    this.texRes[name] = list;
  }

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    this.previewSrc = s;
    for (const r of this.previewRes) r.del();
    this.previewRes = [];
    this.preview = s && s.count ? this.strokes(s, this.previewRes) : null;
    this.marks.setPreview(s?.marks);
  }

  /** T-1407: the ghost becomes the stand-in for the blueprint segment it was placed as, drawn
   * until the next network (the one with that segment) is set; the tool's next ghost goes in its place.
   * T-2404: several while the pointer moves, their state held for its rest. */
  keepPreviewAsPlaced() {
    if (this.preview) this.placed.push(this.preview);
    this.placedRes.push(...this.previewRes);
    this.preview = null;
    this.previewRes = [];
    this.previewSrc = null;
    this.marks.setPreview(undefined);
  }
  /** T-2404: a placed segment with no ghost of it (the pointer moved on before the click's own was
   * answered), drawn as a stand-in from its edges */
  addPlaced(s: Strokes) {
    if (s.count) this.placed.push(this.strokes(s, this.placedRes));
  }
  /** T-2763: edges the player just removed, not drawn until the network without them is set (or `unhideEdges`). */
  private hide: [number, number] = [-2, -2];
  hideEdge(edge: number) {
    this.hide = [edge, this.hide[0] === edge ? this.hide[1] : this.hide[0]];
    this.uploads++;
  }
  unhideEdges() {
    if (this.hide[0] === -2 && this.hide[1] === -2) return;
    this.hide = [-2, -2];
    this.uploads++;
  }
  private dropPlaced() {
    for (const r of this.placedRes) r.del();
    this.placedRes = [];
    this.placed = [];
  }
  private placed: StrokeVao[] = [];
  private placedRes: { del: () => void }[] = [];

  /** The blueprint track picked by a drag selection (T-333), drawn as an ink band under the track, or null. */
  setHighlight(s: Strokes | null) {
    this.highlightSrc = s;
    for (const r of this.highlightRes) r.del();
    this.highlightRes = [];
    this.highlight = s && s.count ? this.strokes(s, this.highlightRes) : null;
  }

  /** T-3059: the picked pieces a rebuild would refuse, outlined in red over the pick's band, or null. */
  setRefused(s: Strokes | null) {
    this.refusedSrc = s;
    for (const r of this.refusedRes) r.del();
    this.refusedRes = [];
    this.refused = s && s.count ? this.strokes(s, this.refusedRes) : null;
  }

  /** What the setters were last given, for `restore`. */
  private layersSrc: NetworkLayers | null = null;
  private refusedSrc: Strokes | null = null;
  private previewSrc: Strokes | null = null;
  private highlightSrc: Strokes | null = null;

  /** The canvas lost its context and the browser gave it a new one (T-953): every GL object went
   * with the old, so the programs are compiled again and what was drawn is uploaded again. */
  restore() {
    this.res = [];
    this.texRes = {};
    for (const k of Object.keys(this.tex) as TexName[]) this.tex[k] = null;
    this.track = this.lines = null;
    this.lodOf.clear();
    this.previewRes = [];
    this.placedRes = []; // the stand-ins go with the old context; their state is on its way
    this.placed = [];
    this.highlightRes = [];
    this.refusedRes = [];
    this.connRes = [];
    this.connVao = this.connFlagBuf = null;
    this.probeProg = this.probeVao = null;
    this.tripEpoch = NaN;
    this.tripVersion = -1;
    this.init(this.gl);
    const n = this.net, layers = this.layersSrc;
    if (n) {
      this.setNetwork(n);
      if (layers) this.setLayers(layers);
    }
    this.setPreview(this.previewSrc);
    this.setHighlight(this.highlightSrc);
    this.setRefused(this.refusedSrc);
  }

  /** T-1071: states from the worker not drawn yet, oldest first. Each waits for its own trips (the
   * worker sends them just after it), so a train's run profile and its departures change in the
   * same frame; a train count's also waits for the game time it takes effect, a moment ahead of
   * the clock, by when its trips are here. */
  private staged: { version: number; at: number; since: number; apply: () => void; trips: TripArgs[] }[] = [];

  /** Draw a state (`apply` puts it in) from game time `at`, once its trips have come. */
  stage(version: number, at: number, apply: () => void) {
    this.staged.push({ version, at, since: performance.now(), apply, trips: [] });
  }

  /** Put in every staged state now: a state drawn at once comes after them. */
  flushStaged() {
    while (this.staged.length) this.putStaged(this.staged.shift()!);
  }

  /** when the last staged state took effect (checks/t1071-train-count.mjs) */
  stagedAt = NaN;

  private putStaged(s: (typeof this.staged)[number]) {
    this.stagedAt = s.at;
    s.apply();
    for (const t of s.trips) this.putTrips(...t);
  }

  /** Put in the staged states due at `time`, in order. */
  private releaseStaged(time: number) {
    while (this.staged.length) {
      const s = this.staged[0];
      // trips that never come (the worker restarted) hold a state up no longer than this
      const late = performance.now() - s.since > STAGE_WAIT_MS;
      if (time < s.at || !(s.trips.length || late)) break;
      this.putStaged(this.staged.shift()!);
    }
  }

  /** Trips for the hour starting at `epoch` (from the worker); those of a staged state wait with it. */
  setTrips(...t: TripArgs) {
    const version = t[1];
    for (let i = this.staged.length - 1; i >= 0; i--) {
      if (this.staged[i].version <= version) {
        this.staged[i].trips.push(t);
        return;
      }
    }
    this.putTrips(...t);
  }

  private putTrips(epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array, bays?: Float32Array) {
    const old = this.tripsBy.get(epoch);
    if (old && old.version > version) return;
    const hour = board ? readBoardHour(epoch, board) : undefined;
    this.tripsBy.set(epoch, { version, data, ends: ends && ends.length === data.length / 2 ? ends : undefined, nexts: nexts && nexts.length === data.length / 2 ? nexts : undefined, bays: BAYS && bays && bays.length === (data.length / 2) * BAY_STRIDE ? bays : undefined, board: hour && hour.trips === data.length / 2 ? hour : undefined });
    if (epoch === this.tripEpoch) this.tripVersion = -1; // re-upload on the next draw
  }

  /** Make the trip buffer hold the trips of the hour `time` is in, when they are here. */
  private sync(time: number): number {
    const epoch = Math.floor(time / EPOCH_S) * EPOCH_S;
    if (epoch !== this.epoch) {
      this.epoch = epoch;
      this.rebases++;
      for (const k of this.tripsBy.keys()) if (k < epoch) this.tripsBy.delete(k);
      this.onEpoch(epoch);
    }
    const t = this.tripsBy.get(epoch);
    if (t && (this.tripEpoch !== epoch || this.tripVersion !== t.version)) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tripBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.data.length ? t.data : new Float32Array(2), gl.DYNAMIC_DRAW);
      // when each is drawn until (T-206); without them, the end of its run and turnaround
      gl.bindBuffer(gl.ARRAY_BUFFER, this.endBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.ends?.length ? t.ends : new Float32Array(Math.max(1, t.data.length / 2)).fill(-1e9), gl.DYNAMIC_DRAW);
      // each trip's place in the boarding account's loads (T-164), -1 without one
      const offs = t.board ? t.board.offs : new Float32Array(Math.max(1, t.data.length / 2) * 2).fill(-1);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.loadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, offs.length ? offs : new Float32Array([-1, 0]), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.standBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.board ? trainStands(t.board, t.data, t.nexts ?? null) : new Float32Array(Math.max(1, t.data.length / 2) * 4).fill(-1), gl.DYNAMIC_DRAW);
      // where each stands at its ends (T-2980), -1 none
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bayBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.bays?.length ? t.bays : noBays(t.data.length / 2), gl.DYNAMIC_DRAW);
      // every train on its timetable until the guard says otherwise (it keeps each trip's lag)
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lagBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(Math.max(1, t.data.length / 2)), gl.DYNAMIC_DRAW);
      this.lagDirty = true;
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      if (t.board?.loads.length) {
        if (this.loadTex) gl.deleteTexture(this.loadTex);
        this.loadTex = this.texture(t.board.loads, "r32f", TEX_W, []);
      }
      this.board = t.board ?? null;
      if (this.fullTraffic) this.writeFull();
      this.tripEpoch = epoch;
      this.tripVersion = t.version;
      this.tripCount = t.data.length / 2;
      this.trips = t.data;
      this.tripEnds = t.ends ?? null;
      this.tripNexts = t.nexts ?? null;
      this.tripBays = t.bays ?? null;
    }
    // Until the new hour's trips arrive, the last hour's keep running on their own epoch.
    return time - this.tripEpoch;
  }

  /** The epoch the trip buffer is relative to. */
  get tripsEpoch() {
    return this.tripEpoch;
  }

  /** T-1640: how long the picture can go without a redraw while the camera is still, ms: 0 when
   * something visible moves at least a half pixel per frame (the fastest train, about 100 m/s of
   * game time, from TRAINS_ZOOM; platform crowds, 5 game seconds, from STATIONS_ZOOM), else a
   * second, which keeps the hour's trips and staged states current. `speed`: game seconds per
   * second; `lat`: the view's. */
  stillWaitMs(zoom: number, lat: number, speed: number, opt: DrawOptions): number {
    let wait = STILL_MAX_MS;
    if (opt.showTrains && this.tripCount && zoom >= TRAINS_ZOOM) {
      const pxPerM = pxPerUnitAt(zoom) / Math.cos(lat * Math.PI / 180);
      wait = Math.min(wait, (1000 * 0.5) / (TRAIN_MAX_MPS * speed * pxPerM));
    }
    if (opt.crowds && this.board && zoom >= STATIONS_ZOOM) wait = Math.min(wait, (1000 * CROWD_STEP_S) / speed);
    return wait;
  }

  /**
   * Draw with the camera MapLibre last drew (gl.ts `Camera`, flat or globe); `time` is game seconds;
   * `zoom` and `dpr` size things in px. The caller has bound and cleared the target; `view`, device
   * px, when it is not the whole drawing buffer (the complex map's snapshot, overlay.ts).
   */
  draw(cam: Camera, time: number, zoom: number, dpr: number, opt: DrawOptions, view?: readonly [number, number]) {
    const gl = this.gl;
    this.quiet = false;
    this.releaseStaged(time);
    const n = this.net;
    if (!n) return;
    const uTime = this.sync(time);
    this.bubbles.hide();
    // the network's buffers are relative to the anchor they were built at (T-307)
    const a = n.anchor ?? anchor();
    const lc = localCamera(cam, a);
    this.cam = cam;
    const vp = view ?? ([gl.drawingBufferWidth, gl.drawingBufferHeight] as const);
    const pxPerUnit = pxPerUnitAt(zoom) * dpr;

    this.timer.begin();

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const k = opt.lineScale ?? 1;
    const wLine = interp(zoom, LINE_PX) * dpr * k;
    const wTrack = interp(zoom, [[9, 0.4], [12, 1.8], [15, 3]]) * dpr * k;
    const wSel = interp(zoom, [[9, 0.5], [12, 2.1], [15, 3]]) * dpr * k;
    const casing = interp(zoom, [[9, 0.6], [12, 2], [15, 2.4]]) * dpr * k;
    // the stations' sizes this frame, which the track through their boxes needs (T-236)
    const boxes = zoom >= STATION_BOX_ZOOM;
    const hidden = zoom < STATIONS_ZOOM;
    // a dot's rim is thinner than a box's (T-322)
    const ring = boxes ? stationRing(zoom) : dotRing(zoom);
    const boxLinePx = interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx = stationRing(BOX_WIDTH_ZOOM);
    // passing stations close in (T-2233): their outer tracks, the trains stopping there, their boxes
    const pass = boxes && this.passing ? passingState(zoom, ring) : null;
    this.passShiftM = pass ? pass.off / pass.mPx : 0;
    this.stationStyle = { boxes, radius: stationRadii(zoom), linePx: wLine / dpr, ringPx: ring, boxLinePx, boxRingPx, hidden, ...(pass ? { passHalfPx: pass.half } : {}) };
    const project = this.projector(cam, a, vp[0] / dpr, vp[1] / dpr);
    // The track through the boxes (T-236, Anita: "maybe the track on top of the station"): the
    // track and line strokes drawn again, no wider than TRACK_IN_BOX of the box's white and
    // without casing, where the stencil marks a station's white (1 the unselected, 2 the selected).
    const through = n.stationCount > 0 && boxes && this.hasStencil;
    // every box has one width on the ground (T-239); any station's gives it on screen here
    const boxWhite = 2 * ((boxes && n.stationCount ? this.stationShapeOf(0, project, true).halfWidth : 0) - ring);
    const trackCap = Math.max(1, TRACK_IN_BOX * boxWhite) * dpr;
    // T-1638: everything up to the stations changes only with the camera, the network and how it is
    // styled, so it is kept in a texture (layer 0, the track through the boxes layer 1) and drawn
    // again only when one of those changes; GL uploads outside the per-frame passes count as a change
    const cache = this.strokeCache && !this.gl.isContextLost() ? (view ? (this.snapCache ??= new StrokeCache(gl, 2)) : (this.mainCache ??= new StrokeCache(gl, 2))) : null;
    const buses = !view && this.buses.any && !opt.byLevel;
    if (!view) this.busesShown = buses;
    const redraw = !cache || cache.stale(vp[0], vp[1], [zoom, dpr, ...lc.m, ...lc.globe, ...lc.plane, ...lc.at, k, opt.byLevel ? 1 : 0, opt.bySpeed ? 1 : 0, opt.byProfit ? 1 : 0, opt.selLine, (opt.selLines ?? []).reduce((h, v) => (h * 31 + v + 1) | 0, 7), opt.selEdge, opt.hotLine ?? -1, opt.hotEdge ?? -1, (opt.hotBuses ?? []).reduce((h, v) => (h * 31 + v + 1) | 0, 7), this.hide[0], this.hide[1], opt.dimOthers ?? 1, opt.flatMarks !== false ? 1 : 0, opt.rampMarks !== false ? 1 : 0, buses ? this.buses.version : -1, through ? 1 : 0, this.uploads, this.strokeCull ? 1 : 0, this.strokeLod ? 1 : 0, this.strokeGather ? 1 : 0, this.lineBands ? 1 : 0], shadeColours());
    this.quiet = true;
    if (cache && redraw) cache.begin(0);
    // the complexes' car parks under everything else (T-1528)
    if (redraw) this.floors.draw(lc, vp, dpr, zoom);
    if (redraw && zoom >= STATION_BOX_ZOOM) this.lots.draw(lc, vp, dpr);
    // bus lines have no strokes: their own pass, under the track (T-1464, busLayer.ts)
    if (redraw && buses) {
      const sh = shadeColours();
      this.buses.draw(lc, a, vp, zoom, dpr, { wLine, casing, wSel, rail: BP_RAIL_RGB, railShare: BP_RAIL_SHARE, sel: opt.selLine, hot: opt.hotLine ?? -1, lit: opt.hotBuses ?? [], dim: opt.dimOthers ?? 1, shades: sh, fill: rgbf(sh.casing), casingRgb: rgbf(sh.casing),
        paint: opt.bySpeed ? "speed" : opt.byProfit ? "profit" : null, phases: n.phases, meta: n.meta, trackRgb: TRACK_RGB });
    }
    gl.useProgram(this.strokeProg);
    const u = this.strokeU;
    setCamera(gl, u, lc);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1i(u.u_sel_line, opt.selLine);
    setSelLines(gl, u, opt.selLines);
    gl.uniform1i(u.u_sel_edge, opt.selEdge);
    gl.uniform2i(u.u_hot, opt.hotLine ?? -1, opt.hotEdge ?? -1);
    gl.uniform2i(u.u_hide, this.hide[0], this.hide[1]);
    // T-2058: the other lines thinner, faded and solid
    const mute = MUTE && (opt.dimOthers ?? 1) < 1 ? MUTE : null;
    gl.uniform1f(u.u_dim, mute ? 1 : opt.dimOthers ?? 1);
    gl.uniform2f(u.u_mute, mute?.[0] ?? 0, mute?.[1] ?? 0);
    gl.uniform1i(u.u_by_level, opt.byLevel ? 1 : 0);
    gl.uniform1i(u.u_by_speed, opt.bySpeed ? 1 : 0);
    gl.uniform1i(u.u_by_profit, opt.byProfit && this.profitTex ? 1 : 0);
    gl.uniform1i(u.u_profit_n, this.profitTex ? this.profitN : 0);
    gl.activeTexture(gl.TEXTURE12);
    gl.bindTexture(gl.TEXTURE_2D, this.profitTex);
    gl.uniform1i(u.u_profit, 12);
    // Lines sharing track are drawn over each other, the last on top (T-190); picking knows the
    // line width, and that no line strokes are drawn when the track is coloured by height or speed.
    this.slotPx = opt.byLevel || opt.bySpeed || opt.byProfit ? 0 : wLine / dpr;
    // T-1799: lines sharing track side by side in the Line view, folding into one stroke as lines
    // thin out zoomed out and as double track opens into its two tracks close in
    const lanes = laneState(zoom);
    // and in the Traffic view (stage 2) each band its riders' width, the shifts `a_wmul.yz`
    const bandOn = !this.lineBands || opt.byLevel || opt.bySpeed || opt.byProfit || (this.trafficWidths && !this.trafficBands) ? 0 : smoothstep(BAND_FOLD_PX[0], BAND_FOLD_PX[1], wLine / dpr) * (1 - lanes.grow);
    this.bandPx = bandOn * wLine / dpr;
    this.bandShiftM = this.bandPx / (pxPerUnitAt(zoom) * kAtY(0));
    this.bandOn = bandOn;
    this.bandMPx = bandOn * pxPerUnitAt(zoom) * kAtY(0);
    gl.uniform1f(u.u_band, bandOn);
    gl.uniform1f(u.u_band_m, pxPerUnitAt(zoom) * kAtY(0) * dpr);
    gl.uniform1i(u.u_band_traffic, this.trafficWidths ? 1 : 0);
    gl.uniform1f(u.u_band_gap, BAND_GAP ? BAND_GAP_PX * dpr : 0);
    gl.uniform3fv(u.u_level_colours, this.levelColours);
    gl.uniform3fv(u.u_track, TRACK_RGB);
    gl.uniform3fv(u.u_bad, BAD_RGB);
    const shades = shadeColours();
    gl.uniform3fv(u.u_sel, rgbf(shades.mapInk));
    gl.uniform3fv(u.u_casing_rgb, rgbf(shades.casing));
    // T-819, T-1284, T-1549, T-1663: red dashes either side, outside the casing, of a line on full
    // track (this base, plus a share of the line's width in the shader), 0.5 px of casing colour round
    // them; the base scaled down with zoom, so far out they stay a hairline
    const fullPass = this.anyFull;
    const fullScale = interp(zoom, [[9, 0.3], [12, 0.6], [15, 1]]) * dpr * k;
    gl.uniform1i(u.u_full_pass, 0);
    gl.uniform1f(u.u_full_px, 1.75 * fullScale);
    gl.uniform1f(u.u_full_out, 1 * fullScale);
    gl.uniform1f(u.u_full_dash, 16 * dpr);
    gl.uniform3fv(u.u_full_rgb, rgbf(shades.fullTrack));
    gl.uniform1i(u.u_ghost, 0);
    gl.uniform3fv(u.u_ghost_rgb, rgbf(shades.ghost));
    gl.uniform1i(u.u_ghost_layer, -1);
    gl.uniform2f(u.u_ghost_min, GHOST_MIN_PX * dpr, 2 * GHOST_OUTLINE_PX * dpr);
    gl.uniform3f(u.u_hsr, interp(zoom, HSR_WIDTH), HSR_STRIPE_PX[0] * dpr, HSR_STRIPE_PX[1] * dpr);
    gl.uniform3fv(u.u_wire, rgbf(shades.tramWire));
    // blueprint track (T-322): a line-wide light blue middle between two thin dark blue rails
    const bpRail = Math.max(1, BP_NARROW * Math.max(1, BP_RAIL_SHARE * wLine / dpr)) * dpr;
    const bpMid = Math.max(0, BP_NARROW * (wLine + 2 * Math.max(1, BP_RAIL_SHARE * wLine / dpr) * dpr) - 2 * bpRail);
    gl.uniform2f(u.u_bp, bpMid, bpRail);
    gl.uniform3fv(u.u_bp_mid, BP_MID_RGB);
    gl.uniform3fv(u.u_bp_rail, BP_RAIL_RGB);
    gl.uniform3fv(u.u_bp_off_mid, rgbf(shades.offMid));
    gl.uniform3fv(u.u_bp_off_rail, rgbf(shades.offRail));
    gl.uniform3fv(u.u_bp_off_fade, rgbf(shades.offFade));
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
    gl.uniform1i(u.u_line_colours, 2);
    this.laneReachPx = lanes.reach;
    this.laneShiftM = LANE_M / 2 * lanes.t;
    const G = lanes.gap * dpr;
    gl.uniform4f(u.u_lanes, G, lanes.t, lanes.grow, lanes.share * G);
    gl.uniform1f(u.u_pass_off, pass ? pass.off * dpr : 0);
    const modes = lanes.t > 0 ? [1, 2, 3] : [0];
    const picked = new Map<StrokeVao, { vao: StrokeVao; ranges: number[] | null }>();
    if (redraw) {
      this.strokeInstances = 0;
      for (const [s, src, isLines] of [[this.track, n.track, false], [this.lines, n.lines, true]] as const) {
        const p = this.pickStrokes(s, src, isLines, zoom, dpr, lanes.grow, project, vp[0] / dpr, vp[1] / dpr);
        if (s && p) picked.set(s, p);
      }
    }
    // `cap` > 0: the track drawn through station boxes (T-236), no wider than that and no casing
    const stroke = (s0: StrokeVao | null, kind: number, width: number, sel: boolean, cap = 0) => {
      if (!s0 || !s0.count) return;
      const { vao: s, ranges } = picked.get(s0) ?? { vao: s0, ranges: null };
      gl.uniform1i(u.u_kind, kind);
      // each track's width once apart: a line's share of the gap, bare track as much thinner
      gl.uniform3f(u.u_width, width, wSel, lanes.share * G * width / wLine);
      gl.uniform1f(u.u_cap, cap);
      gl.bindVertexArray(s.vao);
      const drawAll = () => {
        if (!ranges) {
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
          this.strokeInstances += s.count;
          return;
        }
        for (let r = 0; r < ranges.length; r += 2) {
          this.bindAt(s, ranges[r]);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, ranges[r + 1]);
          this.strokeInstances += ranges[r + 1];
        }
      };
      // [casing px, selected only, T-819's red band under the lines on full track (1), and under it (2) its outline in the casing's colour, T-1284]
      const passes: [number, number, number][] = cap > 0 ? [[0, 0, 0]] : [[casing, 0, 0], [0, 0, 0]];
      // (T-2184's red edges of bands with riders waiting are drawn by the colour pass, in the line's order, T-2347)
      if (kind === 1 && fullPass && cap === 0) passes.unshift([casing, 0, 2], [casing, 0, 1]);
      if (sel) passes.push(...(cap > 0 ? [[0, 1, 0]] as [number, number, number][] : [[casing, 1, 0], [0, 1, 0]] as [number, number, number][]));
      for (const [cas, only, full] of passes) {
        gl.uniform1f(u.u_casing, cas);
        gl.uniform1i(u.u_only_sel, only);
        gl.uniform1i(u.u_full_pass, full);
        for (const mode of modes) {
          if (cap > 0 && mode === 1) continue;
          if (full && mode > 1) continue;
          gl.uniform1i(u.u_lane_mode, mode);
          drawAll();
        }
      }
      if (ranges) this.bindAt(s, 0);
      gl.uniform1i(u.u_full_pass, 0);
      gl.uniform1i(u.u_lane_mode, 0);
    };
    // Colour by height: the track itself, at line width, and no line strokes over it. Blueprint
    // track and the lines planned on it go over built track and its lines (T-695).
    const trackStrokes = (cap = 0) => {
      for (const part of this.hasBlueprint ? [1, 2] : [0]) {
        gl.uniform1i(u.u_bp_part, part);
        stroke(this.track, 0, opt.byLevel || opt.bySpeed || opt.byProfit ? wLine : wTrack, opt.selEdge >= 0, cap);
        if (!opt.byLevel && !opt.bySpeed && !opt.byProfit) stroke(this.lines, 1, wLine, opt.selLine >= 0 || !!opt.selLines?.length, cap);
      }
      gl.uniform1i(u.u_bp_part, 0);
      // passing stations' outer tracks along their platforms (T-2233)
      if (cap > 0 && pass) stroke(this.turnback, 0, wTrack, false, cap);
    };
    if (redraw) {
    if (this.highlight) {
      // T-1679: picked track gets a selected station's look, an ink outline (2 px) apart from the
      // track by a casing-coloured band (3 px), so it reads apart from a selected line's wider
      // stroke (kind 2 placement strokes in a casing pass take u_sel)
      gl.uniform1i(u.u_kind, 2);
      gl.uniform1f(u.u_cap, 0);
      gl.uniform1i(u.u_only_sel, 0);
      gl.uniform1f(u.u_casing, 0.01 * dpr);
      // round both tracks once they part
      gl.uniform1i(u.u_lane_mode, lanes.t > 0 ? 1 : 0);
      gl.bindVertexArray(this.highlight.vao);
      for (const [px, rgb] of [[10, shades.mapInk], [6, shades.casing]] as const) {
        gl.uniform3fv(u.u_sel, rgbf(rgb));
        gl.uniform3f(u.u_width, Math.max(bpMid + 2 * bpRail, wLine) + px * dpr, 0, lanes.share * G + px * dpr);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.highlight.count);
      }
      if (this.refused) {
        gl.bindVertexArray(this.refused.vao);
        for (const [px, rgb] of [[10, shades.fullTrack], [6, shades.casing]] as const) {
          gl.uniform3fv(u.u_sel, rgbf(rgb));
          gl.uniform3f(u.u_width, Math.max(bpMid + 2 * bpRail, wLine) + px * dpr, 0, lanes.share * G + px * dpr);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.refused.count);
        }
      }
      gl.uniform3fv(u.u_sel, rgbf(shades.mapInk));
      gl.uniform1i(u.u_lane_mode, 0);
    }
    trackStrokes();
    // where lines turn back, over their tracks (T-658d), and passing stations' outer tracks (T-2233)
    if (lanes.t > 0 || pass) stroke(this.turnback, 0, wTrack, false);
    gl.uniform1i(u.u_ghost, 1);
    // the ghost, and a placed segment's stand-in until its network is drawn (T-1407)
    for (const g of [...this.placed, this.preview]) {
      stroke(g, 2, wLine, false);
      if (!g?.count) continue;
      // the ghost: blueprint's outline, rails and middle, each over the last (T-1054)
      gl.uniform1f(u.u_casing, 0);
      gl.uniform1f(u.u_cap, 0);
      gl.uniform1i(u.u_only_sel, 0);
      gl.bindVertexArray(g.vao);
      for (let layer = 0; layer < 3; layer++) {
        gl.uniform1i(u.u_ghost_layer, layer);
        for (const mode of modes) {
          if (mode === 1) continue;
          gl.uniform1i(u.u_lane_mode, mode);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, g.count);
        }
      }
      gl.uniform1i(u.u_ghost_layer, -1);
      gl.uniform1i(u.u_lane_mode, 0);
    }
    gl.uniform1i(u.u_ghost, 0);
    this.marks.draw(lc, vp, zoom, dpr, this.levelColours, opt.flatMarks !== false, opt.rampMarks !== false);
    // T-2991: one-way loops' chevrons on the Line view's strokes (the Traffic view's bands move apart)
    if (!opt.byLevel && !opt.bySpeed && !opt.byProfit && !this.trafficWidths)
      this.oneWay.draw(lc, vp, zoom, dpr, { wLine, band: bandOn, mPx: pxPerUnitAt(zoom) * kAtY(0) * dpr, sel: mute ? opt.selLine : -1 });
    this.oneWayTrack.draw(lc, vp, zoom, dpr, { wLine: wLine * 1.8, band: 0, mPx: pxPerUnitAt(zoom) * kAtY(0) * dpr, sel: -1, rgb: rgbf(shadeColours().mapInk) });
    }
    if (cache) {
      if (redraw) {
        cache.end(0, vp);
        if (through) {
          cache.begin(1);
          gl.useProgram(this.strokeProg);
          trackStrokes(trackCap);
          gl.uniform1f(u.u_cap, 0);
          cache.end(1, vp);
        }
      }
      cache.draw(0);
    }
    // the station buildings' hall crowds (T-2219), over the track and under the boxes
    this.floors.drawHall(lc, vp, dpr, zoom, opt.crowds && this.board && !hidden ? this.crowds.hall : null);

    if (n.stationCount) {
      // Platforms along the track from STATION_BOX_ZOOM, each one shape at every zoom (T-204), dots
      // further out (T-189), one for a whole complex (T-217), and below STATIONS_ZOOM only the
      // selected ones (T-205).
      this.syncGroups(n, opt.groups ?? null);
      const also = (opt.selStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
      const alsoKey = also.join(",");
      const changed = opt.selStation !== this.selectedStation || alsoKey !== this.selectedAlsoKey;
      this.selectedStation = opt.selStation;
      if (alsoKey !== this.selectedAlsoKey) {
        this.selectedAlso = new Set(also);
        this.selectedAlsoKey = alsoKey;
      }
      this.syncSelection(n);
      if (changed) this.onSelectionDrawn();
      const [r, rx] = this.stationStyle.radius.map((v) => v * dpr);
      const su = this.stationU;
      gl.useProgram(this.stationProg);
      setCamera(gl, su, lc);
      gl.uniform2f(su.u_viewport, vp[0], vp[1]);
      gl.uniform2f(su.u_radius, r, rx);
      gl.uniform1f(su.u_ring, ring * dpr);
      gl.uniform3fv(su.u_ink, this.ink);
      gl.uniform3fv(su.u_bad, BAD_RGB);
      gl.uniform3fv(su.u_crowd_rgb, CROWD_RGB);
      gl.uniform1f(su.u_line, wLine);
      gl.uniform3f(su.u_lanes, lanes.gap * dpr, lanes.t, lanes.narrow);
      gl.uniform1f(su.u_boxes, boxes ? 1 : 0);
      const hotSt = (opt.hotStations ?? []).filter((i) => i >= 0 && i < n.stationCount);
      this.hotList.fill(-1);
      hotSt.slice(0, HOT_STATIONS).forEach((i, k) => (this.hotList[k] = i));
      gl.uniform1iv(su.u_hot_st, this.hotList);
      gl.uniform1i(su.u_hot_n, Math.min(hotSt.length, HOT_STATIONS));
      // each platform's track line (T-254)
      gl.activeTexture(gl.TEXTURE11);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.stationLine);
      gl.uniform1i(su.u_stline, 11);
      gl.activeTexture(gl.TEXTURE0);
      // local units per px at BOX_WIDTH_ZOOM (the box's width, T-204)
      const unitAtBox = unitPerPxAt(BOX_WIDTH_ZOOM) / dpr;
      gl.uniform2f(su.u_fixed, (boxHalfPxAtBoxZoom({ boxLinePx, boxRingPx }) - (opt.boxShrinkPx ?? 0)) * dpr, unitAtBox);
      gl.uniform1f(su.u_pass_half, (pass ? pass.half - (opt.boxShrinkPx ?? 0) : 0) * dpr);
      // T-2249: a passing station's riders stand between its running track and the outer track's
      // inner edge (the outer track as the track strokes draw it)
      let passIn = 0;
      if (pass) {
        const G = lanes.gap * dpr, lt = lanes.share * G * wTrack / wLine;
        const wl = lanes.t > 0 ? 0.5 * (Math.max(wTrack, G + lt) - lanes.t * (G - lt)) : wTrack;
        passIn = Math.max(0, pass.off * dpr - 0.5 * Math.max(wl, dpr));
      }
      gl.uniform1f(su.u_pass_in, passIn);
      gl.bindVertexArray(this.stationVao);
      gl.uniform1f(su.u_pixel, dpr);
      gl.uniform1i(su.u_layer, 0);
      // the player's links as bridges between platforms (T-217)
      const bridges = boxes && this.bridgeProg ? (n.stationLinkIdx?.length ?? 0) / 2 : 0;
      if (bridges) {
        this.useBridges(lc, vp, ring * dpr, BRIDGE_HALF_PX * dpr * unitAtBox);
        this.hotPairs(this.bridgeU, n.stationLinkIdx!, hotSt);
      }
      // zoomed out, the connectors between a complex's dots and the player's links (T-313, Anita:
      // "it should look like its made of the same material"): the dots' rim, and their white
      // drawn in the same layer, so dots and connectors are one white shape with one outline
      const connectors = !boxes && this.bridgeProg && this.connVao ? this.connPairs.length / 2 : 0;
      if (connectors) {
        const [x0, y0] = project(n.stations[0], n.stations[1]), [x1, y1] = project(n.stations[0] + 1, n.stations[1]);
        const pxPerLoc = Math.max(1e-9, Math.hypot(x1 - x0, y1 - y0));
        this.useBridges(lc, vp, ring * dpr, (connectorHalfPx(this.stationStyle) - ring) / pxPerLoc);
        this.hotPairs(this.bridgeU, this.connPairs, hotSt);
      }
      const connDraw = (pass: number, layer: number) => {
        gl.useProgram(this.bridgeProg);
        gl.uniform1i(this.bridgeU.u_pass, pass);
        gl.uniform1i(this.bridgeU.u_layer, layer);
        gl.bindVertexArray(this.connVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, connectors);
      };
      // Riders waiting on the platforms (T-164, T-219), where platforms are drawn; zoomed out a pie
      // in each dot (T-228, T-553).
      this.crowdsShown = !!opt.crowds && !!this.board && !hidden;
      let any = false;
      if (this.crowdsShown) {
        const now = performance.now();
        if (this.moving && now - this.crowdsAt < MOVING_PLACE_MS) any = this.crowdsAny;
        else {
          this.crowdsAt = now;
          any = this.crowdsAny = this.crowds.update(n, this.board!, uTime, (c, dc) => {
            if (this.crowdBuf) this.rewrite(this.crowdBuf, c);
            if (this.dotCrowdBuf) this.rewrite(this.dotCrowdBuf, dc);
          });
        }
      }
      const fills = any && boxes;
      gl.useProgram(this.stationProg);
      gl.uniform1f(su.u_tint, any && !boxes ? 1 : 0);
      gl.uniform1f(su.u_mask, 0);
      if (through || (boxes && this.hasStencil)) {
        gl.clearStencil(0);
        gl.clear(gl.STENCIL_BUFFER_BIT);
      }
      // T-2219: a station building's floor in the stencil, where its members' rims are not drawn
      const floored = boxes && this.hasStencil && this.floors.markStencil(lc);
      gl.useProgram(this.stationProg);
      // the annexes joined to the boxes (T-1416), in the same layers so box and annex are one
      // shape; no track runs through them
      const layerDraw = (pass: number, layer: number, withBridges: boolean, annexes = true) => {
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_pass, pass);
        gl.uniform1i(su.u_layer, layer);
        gl.bindVertexArray(this.stationVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
        if (annexes && boxes && this.annexes) {
          for (const [kind, bit] of [[ANNEX_ROOM, ST_ROOM]]) {
            if (!(this.annexes & bit)) continue;
            gl.uniform1i(su.u_room, kind);
            gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
          }
          gl.uniform1i(su.u_room, 0);
        }
        if (bridges && withBridges) {
          gl.useProgram(this.bridgeProg);
          gl.uniform1i(this.bridgeU.u_pass, pass);
          gl.uniform1i(this.bridgeU.u_layer, layer);
          gl.bindVertexArray(this.bridgeVao);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, bridges);
        }
      };
      // the dark shapes; a station building's members' only off its floor (T-2219)
      const rims = (pass: number) => {
        if (!floored) return layerDraw(pass, 0, true);
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_bld, 1);
        layerDraw(pass, 0, true);
        gl.enable(gl.STENCIL_TEST);
        gl.stencilMask(0);
        gl.stencilFunc(gl.EQUAL, 0, STENCIL_BIT);
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_bld, 2);
        layerDraw(pass, 0, false);
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_bld, 0);
        gl.stencilMask(0xff);
        gl.disable(gl.STENCIL_TEST);
      };
      // every station, then the selected ones over them with their outline and glow; each pass the
      // dark shapes, then the white inside every rim, so stations drawn over each other are one white
      // shape with one outline (T-217), then the crowds and the track through them (T-236)
      for (const pass of [0, 1]) {
        if (pass === 0 && hidden) continue;
        if (pass === 1) {
          const shapes = this.selectedShapes(project);
          if (!shapes.length) break;
          // a selected building's outline goes round its floor, not its members' boxes (T-2219)
          if (floored) this.floors.drawSelected(new Set(this.selectedDrawn()), project, vp, dpr, zoom, this.ink);
          this.drawGlow(floored ? this.selectedShapes(project, true) : shapes, dpr, vp);
        }
        // zoomed out the connectors' white goes under the dots' white, so a dot's crowd pie (T-228)
        // stays whole (T-313)
        if (connectors) connDraw(pass, 0);
        rims(pass);
        if (connectors) connDraw(pass, 1);
        layerDraw(pass, 1, true);
        if (!boxes) continue;
        if (through) {
          // mark this pass's white, no colour
          gl.enable(gl.STENCIL_TEST);
          gl.stencilMask(~STENCIL_BIT & 0xff);
          gl.stencilFunc(gl.ALWAYS, pass + 1, 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
          gl.colorMask(false, false, false, false);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 1);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 1);
          }
          layerDraw(pass, 1, true, false);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 0);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 0);
          }
          gl.colorMask(true, true, true, true);
          gl.stencilMask(0xff);
          gl.disable(gl.STENCIL_TEST);
        }
        if (fills) layerDraw(pass, 2, false);
        if (through) {
          gl.enable(gl.STENCIL_TEST);
          gl.stencilFunc(gl.EQUAL, pass + 1, ~STENCIL_BIT & 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
          if (cache) cache.draw(1);
          else {
            gl.useProgram(this.strokeProg);
            trackStrokes(trackCap);
            gl.uniform1f(u.u_cap, 0);
          }
          gl.disable(gl.STENCIL_TEST);
        }
      }
      // riders changing trains walking between stations (T-1124), with the platform crowds
      const placed = this.crowds.stationOfQueue;
      if (this.crowdsShown && boxes && placed) {
        this.bubbles.sync(this.board, n, placed);
        this.bubbles.draw(lc, vp, uTime, 0.5 * boxWhite * dpr, rgbf(shades.walkers), rgbf(shades.casing), dpr);
      }
    }

    // a bus line's stops at train stations, over the stations (T-2948)
    if (buses) this.buses.drawStationStops(lc, vp, dpr);

    // No trains zoomed far out (T-205).
    this.trainsHidden = zoom < TRAINS_ZOOM;
    if (opt.showTrains && this.tripCount && !this.trainsHidden) {
      this.detailedTrains = zoom >= STATION_BOX_ZOOM && !opt.hollowTrains;
      const tw = interp(zoom, TRAIN_PX) * dpr * k;
      this.guardTrains(time, cam, a, project, vp[0] / dpr, vp[1] / dpr, zoom >= CONFLICT_ZOOM);
      gl.useProgram(this.trainProg);
      gl.uniform1i(this.trainU.u_hollow, opt.hollowTrains ? 1 : 0);
      gl.useProgram(this.trainProg);
      setCamera(gl, this.trainU, lc);
      gl.uniform2f(this.trainU.u_viewport, vp[0], vp[1]);
      gl.uniform1f(this.trainU.u_time, uTime);
      gl.uniform1f(this.trainU.u_px_per_unit, pxPerUnit);
      gl.uniform1f(this.trainU.u_psi0, anchorPsi(a));
      // a pill is never shorter than about 2.9 times its width
      gl.uniform2f(this.trainU.u_min_px, tw * 2.2 / 0.75, tw);
      this.bindTrainTextures(this.trainU);
      gl.uniform1i(this.trainU.u_line_colours, 2);
      gl.uniform1i(this.trainU.u_sel_line, opt.selLine);
      setSelLines(gl, this.trainU, opt.selLines);
      gl.uniform1f(this.trainU.u_dim, opt.dimOthers ?? 1);
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.fill);
      gl.uniform1i(this.trainU.u_fill, 6);
      gl.uniform3fv(this.trainU.u_bad, BAD_RGB);
      gl.uniform1i(this.trainU.u_detail, this.detailedTrains ? 1 : 0);
      gl.uniform1f(this.trainU.u_dpr, dpr);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.cars);
      gl.uniform1i(this.trainU.u_cars, 7);
      gl.activeTexture(gl.TEXTURE8);
      gl.bindTexture(gl.TEXTURE_2D, this.loadTex);
      gl.uniform1i(this.trainU.u_loads, 8);
      gl.activeTexture(gl.TEXTURE9);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.stretch);
      gl.uniform1i(this.trainU.u_stretch, 9);
      gl.activeTexture(gl.TEXTURE10);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.lane);
      gl.uniform1i(this.trainU.u_lane_tex, 10);
      // on its own track close in (T-658): the shift at full parting, how far parted
      gl.uniform2f(this.trainU.u_lane, lanes.gap * dpr * 0.5, lanes.t);
      // on its line's band (T-1799)
      gl.uniform1f(this.trainU.u_bands, this.bandPx * dpr);
      gl.uniform1f(this.trainU.u_bands_m, this.bandMPx * dpr);
      gl.uniform2f(this.trainU.u_passing, pass ? pass.off * dpr : 0, pass ? 1 : 0);
      gl.bindVertexArray(this.trainVao);
      // every train but the selected; then the selected one's halo, round the whole consist as one
      // pill; then the selected train over it (T-242)
      const sel = opt.selTrain;
      gl.uniform2f(this.trainU.u_sel_trip, sel ? sel.profile : -1, sel ? sel.dep - this.tripEpoch : 0);
      const halo = shadeColours();
      gl.uniform3fv(this.trainU.u_ink, rgbf(halo.mapInk));
      gl.uniform3fv(this.trainU.u_band, rgbf(halo.casing));
      gl.uniform1i(this.trainU.u_pass, 0);
      const drawTrains = () => gl.drawArraysInstanced(this.detailedTrains ? gl.TRIANGLES : gl.TRIANGLE_STRIP, 0, this.detailedTrains ? 6 * this.maxCars : 4, this.tripCount);
      drawTrains();
      if (sel) {
        gl.uniform1i(this.trainU.u_pass, 1);
        gl.uniform1i(this.trainU.u_detail, 0);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.tripCount);
        gl.uniform1i(this.trainU.u_pass, 2);
        gl.uniform1i(this.trainU.u_detail, this.detailedTrains ? 1 : 0);
        drawTrains();
      }
    }

    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
    this.timer.end();
    this.quiet = false;
  }

  /** T-1965: hold trains in view back behind those ahead on their track (`TrainGuard`), and upload
   * their lags when they change. `w`, `h`: the view, CSS px; `close`: zoomed in to CONFLICT_ZOOM. */
  private guardTrains(time: number, cam: Camera, a: Anchor, project: Project, w: number, h: number, close: boolean) {
    const t0 = performance.now();
    this.guardTrainsOn(time, cam, a, project, w, h, close);
    this.guardMs = performance.now() - t0;
  }
  /** ms the guard took the last frame (checks) */
  guardMs = 0;
  private guardOut: [number, number] = [0, 0];
  private guardTrainsOn(time: number, cam: Camera, a: Anchor, globeProject: Project, w: number, h: number, close: boolean) {
    let changed = this.lagDirty;
    this.lagDirty = false;
    if (this.follow) {
      // the flat map's projection without an array a call (`projector` makes one each)
      const m = cam.globe ? null : mul4(cam.merc, localToMercMatrix(a)), out = this.guardOut;
      const project = m ? (x: number, y: number) => { clipToCss(m, x, y, w, h, out); return out; } : globeProject;
      changed = this.guard.update({ time, epoch: this.tripEpoch, trips: this.trips, ends: this.tripEnds, nexts: this.tripNexts, bays: this.tripBays, project, w, h, close }) || changed;
    } else if (this.guard.lag.some((v) => v !== 0)) {
      this.guard.lag.fill(0);
      changed = true;
    }
    if (!changed || this.guard.lag.length !== this.tripCount || !this.tripCount) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lagBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.guard.lag);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Select the bridge program and set its uniforms for this frame: `ring` and `half` (a bridge's
   * half width in local units) as the caller works them out. */
  private useBridges(lc: LocalCamera, vp: readonly [number, number], ring: number, half: number) {
    const gl = this.gl, bu = this.bridgeU;
    gl.useProgram(this.bridgeProg);
    setCamera(gl, bu, lc);
    gl.uniform2f(bu.u_viewport, vp[0], vp[1]);
    gl.uniform1f(bu.u_ring, ring);
    gl.uniform1f(bu.u_bridge, half);
    gl.uniform3fv(bu.u_ink, this.ink);
    gl.uniform3fv(bu.u_bad, BAD_RGB);
    gl.uniform3fv(bu.u_crowd_rgb, CROWD_RGB);
    gl.uniform1f(bu.u_mask, 0);
  }

  /** Local units of anchor `a` to CSS px on the canvas (y down) through the draw's camera; NaN
   * behind the globe's horizon. */
  private projector(cam: Camera, a: Anchor, w: number, h: number): Project {
    if (!cam.globe) {
      // the flat map: one matrix, as the shaders' u_matrix
      const m = mul4(cam.merc, localToMercMatrix(a));
      return (x, y) => {
        const out: [number, number] = [0, 0];
        clipToCss(m, x, y, w, h, out);
        return out;
      };
    }
    return (x, y) => {
      const out: [number, number] = [0, 0];
      mercToCss(cam, a.merc[0] + x / EARTH_C, a.merc[1] + y / EARTH_C, w, h, out);
      return out;
    };
  }

  /** The outline and glow round the selected shapes' union (T-189, T-217), under them. */
  private drawGlow(shapes: StationShape[], dpr: number, vp: readonly [number, number]) {
    const gl = this.gl;
    const count = Math.min(shapes.length, SEL_MAX), d = this.selData;
    const reach = SEL_OUTLINE_PX + SEL_GLOW_PX + 1;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < count; i++) {
      const s = shapes[i];
      const o = i * SEL_STRIDE * 4;
      d.fill(0, o, o + SEL_STRIDE * 4);
      d.set([s.x, s.y, s.ux, s.uy, s.nx, s.ny, s.halfLength, s.halfWidth, s.dot], o);
      if (!(s.dot > 0)) d.set(shapePoints(s), o + 12);
      // every point of a shape is within its half length plus half width of its middle
      const r = (s.dot > 0 ? s.dot : s.halfLength + s.halfWidth) + reach;
      x0 = Math.min(x0, s.x - r); y0 = Math.min(y0, s.y - r); x1 = Math.max(x1, s.x + r); y1 = Math.max(y1, s.y + r);
    }
    const w = vp[0] / dpr, h = vp[1] / dpr;
    x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(w, x1); y1 = Math.min(h, y1);
    if (x1 <= x0 || y1 <= y0) return;
    gl.activeTexture(gl.TEXTURE10);
    gl.bindTexture(gl.TEXTURE_2D, this.selTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, count * SEL_STRIDE, 1, gl.RGBA, gl.FLOAT, d, 0);
    const u = this.selU;
    gl.useProgram(this.selProg);
    gl.uniform1i(u.u_shapes, 10);
    gl.uniform1i(u.u_count, count);
    gl.uniform4f(u.u_bbox, x0, y0, x1, y1);
    gl.uniform2f(u.u_css, w, h);
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform1f(u.u_height, vp[1]);
    gl.uniform3fv(u.u_ink, this.ink);
    gl.uniform1f(u.u_outline, SEL_OUTLINE_PX);
    gl.uniform1f(u.u_glow, SEL_GLOW_PX);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
  }

  /** The connectors between dots zoomed out (T-313, Anita: "if the dots don't touch, we draw the
   * connector"): within each group of `groups` (game/complexView.ts `dotGroups`, the complex rule's
   * distances) the shortest joins linking all its dots, and each of the player's links (T-197)
   * between two groups from one linked station's dot to the other's. Each runs centre to centre;
   * where dots overlap it is hidden inside them. Rebuilt when the groups or the network change. */
  private syncGroups(n: NetworkBuffers, groups: StationGroups | null) {
    if (this.groupSrc.groups === groups && this.groupSrc.net === n) return;
    this.groupSrc = { groups, net: n };
    const S = n.stations, sh = n.stationShape;
    const mid = (i: number): [number, number] => [S[i * 3] + (sh?.[i * 4] ?? 0), S[i * 3 + 1] + (sh?.[i * 4 + 1] ?? 0)];
    const pairs: number[] = [];
    const groupOf = new Int32Array(Math.max(1, n.stationCount)).fill(-1);
    if (groups) {
      const index = new Map(n.stationIds.map((id, i) => [id, i]));
      groups.members.forEach((ids, k) => {
        const m = ids.map((id) => index.get(id)).filter((i): i is number => i !== undefined);
        for (const i of m) groupOf[i] = k;
        for (const [a, b] of spanningPairs(m.map(mid))) pairs.push(m[a], m[b]);
      });
    }
    const idx = n.stationLinkIdx;
    for (let k = 0; idx && k < idx.length / 2; k++) {
      const i = idx[k * 2], j = idx[k * 2 + 1];
      if (groupOf[i] < 0 || groupOf[i] !== groupOf[j]) pairs.push(i, j);
    }
    this.connPairs = Int32Array.from(pairs);
    const count = pairs.length / 2;
    const ce = (this.connEnds = new Float32Array(Math.max(1, count) * 4));
    for (let k = 0; k < count; k++) ce.set([...mid(pairs[k * 2]), ...mid(pairs[k * 2 + 1])], k * 4);
    // its own buffers and VAO, replaced with the connectors
    for (const r of this.connRes) r.del();
    this.connRes = [];
    this.connVao = null;
    this.connFlagBuf = null;
    if (this.gl && this.bridgeProg) {
      const gl = this.gl;
      const cv = (this.connVao = gl.createVertexArray()!);
      this.connRes.push({ del: () => gl.deleteVertexArray(cv) });
      gl.bindVertexArray(cv);
      this.attrib(this.bridgeProg, "a_ends", this.buffer(ce, this.connRes), 4, gl.FLOAT, 1);
      this.connFlagBuf = this.buffer(new Float32Array(Math.max(1, count) * 2), this.connRes);
      this.attrib(this.bridgeProg, "a_bflag", this.connFlagBuf, 2, gl.FLOAT, 1);
      gl.bindVertexArray(null);
    }
    this.selDrawnKey = "";
  }

  /** Which stations and bridges are drawn selected (`a_sel`, `a_bflag`), when that changed. */
  private syncSelection(n: NetworkBuffers) {
    const key = `${this.selectedStation}/${this.selectedAlsoKey}`;
    if (key === this.selDrawnKey) return;
    this.selDrawnKey = key;
    const flags = new Float32Array(Math.max(1, n.stationCount));
    for (let i = 0; i < n.stationCount; i++) flags[i] = this.drawnSelected(i) ? 1 : 0;
    if (this.stationSelBuf) this.rewrite(this.stationSelBuf, flags);
    // a bridge or connector is selected with both its stations, and faded when either is a blueprint
    const pairFlags = (idx: ArrayLike<number>) => {
      const b = new Float32Array(Math.max(2, idx.length));
      for (let k = 0; k < idx.length / 2; k++) {
        const i = idx[k * 2], j = idx[k * 2 + 1];
        b[k * 2] = this.drawnSelected(i) && this.drawnSelected(j) ? 1 : 0;
        b[k * 2 + 1] = (Math.round(n.stations[i * 3 + 2]) | Math.round(n.stations[j * 3 + 2])) & 2 ? 1 : 0;
      }
      return b;
    };
    const idx = n.stationLinkIdx;
    if (idx?.length && this.bridgeFlagBuf) this.rewrite(this.bridgeFlagBuf, pairFlags(idx));
    if (this.connPairs.length && this.connFlagBuf) this.rewrite(this.connFlagBuf, pairFlags(this.connPairs));
  }

  /** The share of the room to stand that the platform side of queue `q` of the trip buffer holds,
   * as last drawn (0-1); null when crowds are not drawn or the queue has no platform. */
  queueShare(q: number): number | null {
    return this.crowdsShown && this.net ? this.crowds.queueShare(q, this.net) : null;
  }

  /** Zoomed out, the crowd station `i`'s dot shows as a pie (T-228, T-553): riders waiting at its platforms
   * and its fullest platform side's share of the room there; null as platforms or when crowds are not drawn. */
  dotCrowd(i: number): { waiting: number; share: number } | null {
    return this.crowdsShown && !this.stationStyle.boxes ? this.crowds.dot(i) : null;
  }

  private bindTrainTextures(u: Record<string, WebGLUniformLocation | null>) {
    const gl = this.gl;
    const bind = (unit: number, tex: WebGLTexture | null, loc: WebGLUniformLocation | null) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(3, this.tex.phases, u.u_phases);
    bind(4, this.tex.meta, u.u_meta);
    bind(0, this.tex.samples, u.u_samples);
    bind(1, this.tex.lines, u.u_lines);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
  }

  /**
   * GPU positions (local units, x/y pairs; 1e9 when off the track) of every trip in the buffer at
   * game time `time`, read back by transform feedback. For precision checks; changes GL state
   * freely, so only call it on a renderer with a context of its own.
   */
  probe(time: number): Float32Array {
    const gl = this.gl;
    if (!this.probeProg) {
      this.probeProg = compile(gl, PROBE_VS, PROBE_FS, ["v_pos"]);
      this.probeU = uniforms(gl, this.probeProg, ["u_time", "u_phases", "u_meta", "u_samples", "u_lines"]);
    }
    this.releaseStaged(time);
    const uTime = this.sync(time);
    const n = this.tripCount;
    if (!this.probeVao) {
      this.probeVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.probeVao);
      this.attrib(this.probeProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 0);
      this.attrib(this.probeProg, "a_end", this.endBuf!, 1, gl.FLOAT, 0);
      this.attrib(this.probeProg, "a_lag", this.lagBuf!, 1, gl.FLOAT, 0);
      gl.bindVertexArray(null);
    }
    const out = this.buffer(new Float32Array(Math.max(1, n) * 2), []);
    gl.useProgram(this.probeProg);
    gl.uniform1f(this.probeU.u_time, uTime);
    this.bindTrainTextures(this.probeU);
    gl.bindVertexArray(this.probeVao);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, out);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindVertexArray(null);
    const res = new Float32Array(n * 2);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, out);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, res);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    gl.deleteBuffer(out);
    return res;
  }
}
