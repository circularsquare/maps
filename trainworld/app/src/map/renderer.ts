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

import { BAD_RED, LEVEL_COLOURS, NET_INK, rgbf } from "../game/palette";
import { BOX_WIDTH_ZOOM, boxHalfPxAtBoxZoom, BRIDGE_HALF_PX, bridgeShape, connectorHalfPx, dotRing, spanningPairs, insideStation, PLATFORM_POINTS, platformPoints, SEL_GLOW_PX, SEL_OUTLINE_PX, shapePoints, STATION_BOX_ZOOM, STATIONS_ZOOM, TRAINS_ZOOM, stationHaloRadius, stationRadii, stationReachPx, stationRing, stationShape, type Project, type StationShape, type StationStyle } from "./stationGeometry";
import { shadeColours } from "../game/shades";
import { readBoardHour, type BoardHour } from "../game/boarding";
import { PlatformCrowds } from "./platformCrowds";
import { TransferBubbles } from "./transferBubbles";
import { anchor, anchorPsi, EARTH_C, kAtY, localToMercMatrix, mul4, pxPerUnitAt, unitPerPxAt, type Anchor } from "./geo";
import { clipToCss, compile, GpuTimer, interp, localCamera, mercToCss, PROJECT_UNIFORMS, setCamera, uniforms, type Camera, type LocalCamera } from "./gl";
import { GradeMarks } from "./gradeMarks";
import { F_BLUEPRINT, platformShapeAt, type NetworkBuffers, type Strokes } from "./network";
import { BRIDGE_VS, HOT_STATIONS, LANE_SHARE, LINES_W, PROBE_FS, PROBE_VS, SEL_FS, SEL_MAX, SEL_STRIDE, SEL_VS, STATION_FS, STATION_VS, STROKE_FS, STROKE_VS, TEX_W, TRAIN_FS, TRAIN_VS } from "./rendererShaders";

/** The render epoch moves to the start of each game hour. */
export const EPOCH_S = 3600;
/** The longest a staged state waits for its trips, wall ms (T-1071). */
const STAGE_WAIT_MS = 2000;
type TripArgs = [epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array];
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
const LINE_PX: [number, number][] = [[9, 0.85], [12, 3.9], [15, 6.6]];
/** T-1097 (Anita: "a tad more visual weight, not much"): high-speed track and its lines this many
 * times as wide by zoom while drawn as one stroke, and their centre line only on strokes this wide
 * (css px, coming in between the two) */
const HSR_WIDTH: [number, number][] = [[9, 1.4], [11, 1.25], [13, 1]];
const HSR_STRIPE_PX = [3, 5] as const;
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
/** Platform crowds (T-164, T-219): drawn from this zoom, where platforms are (T-204), as each side's
 * half of the box filling from its middle outwards in `CROWD_RGB`, red (BAD_RGB) once that side is
 * full (T-559). */
export const CROWD_ZOOM = STATION_BOX_ZOOM;
const CROWD_RGB = rgbf("#77726d");
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
  /** selected line id, -1 for none */
  selLine: number;
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
  /** the stations the complex rule's distances group (T-313): zoomed out their dots are joined by
   * connectors; absent, only the player's links are */
  groups?: StationGroups | null;
  /** hovered (T-730): a line's number, a bare track edge's id (-1 none), stations by index */
  hotLine?: number;
  hotEdge?: number;
  hotStations?: number[];
  /** T-1248: lines, track and trains drawn this many times their width; absent: 1 */
  lineScale?: number;
}

interface StrokeVao {
  vao: WebGLVertexArrayObject;
  count: number;
  /** the per-segment width multipliers, rewritten by `setLayers` */
  wmulBuf: WebGLBuffer;
  /** per segment 1 where its track is at capacity, rewritten by `setFull` (T-819) */
  fullBuf: WebGLBuffer;
}

/**
 * The demand views drawn by the network renderer (T-084, map/demandLayers.ts), one value per
 * line stroke segment, train sample or station; null = drawn as without demand.
 */
export interface NetworkLayers {
  /** per line stroke segment: width multiple */
  lineWidth: Float32Array;
  /** per train sample: how full run 0's and run 1's trains are (2 each: 0-1 of the gauge, 2 over
   * full, -1 not known) */
  fill: Float32Array;
  /** per station: its demand circle's size, 0-1 of the largest (map/demandLayers.ts `stationScale`) */
  stationSize: Float32Array;
  /** per station: its demand circle's offset from the station, local units (2 each); a station
   * complex's circle sits at the complex's mean position (T-146). Absent = none moved. */
  stationHalo?: Float32Array;
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
  private res: { del: () => void }[] = [];
  private previewRes: { del: () => void }[] = [];
  private track: StrokeVao | null = null;
  private lines: StrokeVao | null = null;
  /** the crossovers where lines turn back (T-658d), drawn close in only */
  private turnback: StrokeVao | null = null;
  private preview: StrokeVao | null = null;
  private highlightRes: { del: () => void }[] = [];
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
  /** the hour's loads (T-164), replaced with the trips; kept apart from `res`, like the trips */
  private loadTex: WebGLTexture | null = null;
  private tex: Record<"phases" | "meta" | "samples" | "lines" | "colours" | "fill" | "cars" | "stretch" | "stationLine" | "lane", WebGLTexture | null> = { phases: null, meta: null, samples: null, lines: null, colours: null, fill: null, cars: null, stretch: null, stationLine: null, lane: null };
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
  private maxCars = 8;
  /** Used by picking to follow the same close-up representation. */
  detailedTrains = false;
  private stationSizeBuf: WebGLBuffer | null = null;
  private stationHaloBuf: WebGLBuffer | null = null;
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
  /** Station `i` on screen as a platform or its own dot (stationGeometry.ts). */
  stationShapeOf(i: number, project: Project): StationShape {
    const n = this.net!;
    const flag = Math.round(n.stations[i * 3 + 2]);
    return stationShape(n.stations[i * 3], n.stations[i * 3 + 1], platformShapeAt(n, i), n.stationShape?.[i * 4 + 3] ?? 0, project, this.stationStyle, (flag & 1) !== 0);
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
        if (reach) {
          const x = n.stations[i * 3], y = n.stations[i * 3 + 1], r = reach.r[i] * perLoc + extra;
          const sx = ox + a * x + b * y, sy = oy + c * x + e * y;
          if (Math.abs(sx - px) > r || Math.abs(sy - py) > r) continue;
        }
        const s = this.drawnShapeOf(i, project);
        if (!insideStation(s, px, py, padding)) continue;
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
  selectedShapes(project: Project): StationShape[] {
    const out: StationShape[] = [];
    for (const i of this.selectedDrawn()) out.push(this.drawnShapeOf(i, project));
    return out.concat(this.selectedBridges(project));
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
  /** Half a double track's drawn width in CSS px at the last draw once it has grown to its two
   * tracks' width (T-658), 0 further out (picking) */
  laneReachPx = 0;
  /** Metres a train on double track is drawn to the right of its line's centre line at the last
   * draw (T-658): 2 m once the two tracks have parted, 0 further out (picking) */
  laneShiftM = 0;
  /** per line stroke segment, its width as a multiple of `slotPx` (traffic thickness), or null
   * when all are plain: picking reaches across wide lines (T-227) */
  lineWidths: Float32Array | null = null;
  /** the network has blueprint track: it and its lines are drawn after the rest (T-695) */
  hasBlueprint = false;
  private levelColours = new Float32Array([3, 2, 1, 0, -1, -2, -3].flatMap((l) => [...rgbf(LEVEL_COLOURS[l as 3])]));
  private ink = rgbf(NET_INK);
  private timer!: GpuTimer;
  /** the canvas has a stencil buffer: the track is drawn through station boxes (T-236) */
  private hasStencil = false;
  /** trips per epoch from the worker, newest version kept, with the boarding account's hour */
  private tripsBy = new Map<number, { version: number; data: Float32Array; ends?: Float32Array; nexts?: Float32Array; board?: BoardHour }>();
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
  net: NetworkBuffers | null = null;
  /** riders changing trains on their walk (T-1124) */
  readonly bubbles = new TransferBubbles();
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

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.strokeProg = compile(gl, STROKE_VS, STROKE_FS);
    this.stationProg = compile(gl, STATION_VS, STATION_FS);
    this.marks.init(gl);
    this.bubbles.init(gl);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.hasStencil = (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0;
    this.strokeU = uniforms(gl, this.strokeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_casing", "u_casing_rgb", "u_cap", "u_kind", "u_sel_line", "u_sel_edge", "u_hot", "u_only_sel", "u_bp_part", "u_lanes", "u_lane_mode", "u_dim", "u_by_level", "u_by_speed", "u_level_colours", "u_track", "u_bad", "u_sel", "u_bp", "u_bp_mid", "u_bp_rail", "u_bp_off_mid", "u_bp_off_rail", "u_bp_off_fade", "u_line_colours", "u_full_pass", "u_full_px", "u_full_rgb", "u_ghost", "u_ghost_layer", "u_ghost_rgb", "u_ghost_min", "u_hsr"]);
    this.stationU = uniforms(gl, this.stationProg, [...PROJECT_UNIFORMS, "u_viewport", "u_radius", "u_ring", "u_ink", "u_line", "u_boxes", "u_fixed", "u_pass", "u_layer", "u_halo", "u_halo_radius", "u_pixel", "u_bad", "u_crowd_rgb", "u_tint", "u_mask", "u_stline", "u_lanes", "u_hot_st", "u_hot_n"]);
    this.bridgeProg = compile(gl, BRIDGE_VS, STATION_FS);
    this.bridgeU = uniforms(gl, this.bridgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_ring", "u_bridge", "u_pass", "u_layer", "u_halo", "u_ink", "u_bad", "u_crowd_rgb", "u_mask", "u_hot_st", "u_hot_n"]);
    this.selProg = compile(gl, SEL_VS, SEL_FS);
    this.selU = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_shapes", "u_count", "u_dpr", "u_height", "u_ink", "u_outline", "u_glow"]);
    this.selTex = this.texture(this.selData, "rgba32f", SEL_MAX * SEL_STRIDE, []);
    this.trainU = uniforms(gl, this.trainProg, [...PROJECT_UNIFORMS, "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_phases", "u_meta", "u_samples", "u_lines", "u_line_colours", "u_fill", "u_bad", "u_detail", "u_dpr", "u_cars", "u_loads", "u_stretch", "u_sel_line", "u_dim", "u_sel_trip", "u_pass", "u_ink", "u_band", "u_psi0", "u_lane_tex", "u_lane"]);
    this.timer = new GpuTimer(gl);
    this.tripBuf = gl.createBuffer();
    this.endBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.endBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1e9]), gl.DYNAMIC_DRAW);
    this.loadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.loadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, 0]), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.loadTex = this.texture(new Float32Array(1), "r32f", TEX_W, []);
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

  private strokes(s: Strokes, list = this.res): StrokeVao {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    list.push({ del: () => gl.deleteVertexArray(vao) });
    gl.bindVertexArray(vao);
    const p = this.strokeProg;
    const one = (a: ArrayBufferView, n: number) => (s.count ? a : new Float32Array(n));
    this.attrib(p, "a_seg", this.buffer(one(s.seg, 4), list), 4, gl.FLOAT, 1);
    this.attrib(p, "a_colour", this.buffer(s.count ? s.colour : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_edge", this.buffer(s.count ? s.edge : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_level", this.buffer(one(s.level, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_flags", this.buffer(one(s.flags, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_speed", this.buffer(s.count && s.speed ? s.speed : new Float32Array(Math.max(1, s.count)).fill(160), list), 1, gl.FLOAT, 1);
    const wmulBuf = this.buffer(new Float32Array(Math.max(1, s.count)).fill(1), list);
    this.attrib(p, "a_wmul", wmulBuf, 1, gl.FLOAT, 1);
    const fullBuf = this.buffer(new Float32Array(Math.max(1, s.count)), list);
    this.attrib(p, "a_full", fullBuf, 1, gl.FLOAT, 1);
    const lane = s.count && s.lane ? s.lane : new Uint8Array(Math.max(1, s.count) * 2).fill(255);
    const loc = gl.getAttribLocation(p, "a_lane");
    if (loc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer(lane, list));
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    return { vao, count: s.count, wmulBuf, fullBuf };
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
    if (lines.count) this.rewrite(lines.wmulBuf, l ? l.lineWidth : new Float32Array(lines.count).fill(1));
    this.lineWidths = l && l.lineWidth.length === lines.count ? l.lineWidth : null;
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
    if (this.stationSizeBuf && n.stationCount) this.rewrite(this.stationSizeBuf, l ? l.stationSize : new Float32Array(n.stationCount));
    if (this.stationHaloBuf && n.stationCount) this.rewrite(this.stationHaloBuf, l?.stationHalo ?? new Float32Array(n.stationCount * 2));
  }

  /** T-819, T-1125: in the traffic view (`traffic`), the edges whose track is at capacity, drawn
   * with a red edge and a darker body on the lines over them, and the stretches where riders wait
   * for a train with room (the boarding account's, each hour), whose red edge and darkening are
   * stronger as that wait grows (T-1160, rendererShaders.ts `FULL_FLOOR`). */
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
  /** the full stretches as last written, line * 4096 + stretch: the share of the day's trains
   * packed there (1 from a WASM build without shares) and the riders' average wait there for a
   * train with room, min (NaN from a WASM build without waits) */
  private fullKeys = new Map<number, { share: number; wait: number }>();
  private fullDrawn = "";
  /** What is drawn on line `line`'s run 0 stretch `s` (T-1125, T-1160); null where nothing is. */
  fullStretch(line: number, s: number): { share: number; wait: number } | null {
    return this.fullTraffic ? this.fullKeys.get(line * 4096 + s) ?? null : null;
  }
  private writeFull() {
    const l = this.lines, n = this.net;
    const keys = new Map<number, { share: number; wait: number }>();
    const b = this.board, fs = this.fullTraffic ? b?.fullStretches : undefined;
    for (let k = 0; b && fs && k < fs.length; k += 2) {
      const j = k / 2;
      keys.set(fs[k] * 4096 + fs[k + 1], { share: b.fullShares.length ? b.fullShares[j] : 1, wait: b.fullWaits.length ? b.fullWaits[j] / 60 : NaN });
    }
    const drawn = `${n ? 1 : 0}/${l?.count}/${[...(this.fullEdges ?? [])].join(",")}/${[...keys].map(([k, v]) => `${k}:${v.share}:${v.wait}`).join(",")}`;
    this.fullKeys = keys;
    if (!l || !l.count || !n || drawn === this.fullDrawn) return;
    this.fullDrawn = drawn;
    // the shader's a_full: -1 track at capacity, else the stretch's wait (min); a WASM build without
    // waits gives the share of full trains as a wait, all of it at 75%
    const out = new Float32Array(l.count), f = this.fullEdges, stretch = n.lines.stretch;
    for (let i = 0; i < l.count; i++) {
      if (f?.size && f.has(n.lines.edge[i])) { out[i] = -1; continue; }
      const s = keys.size && stretch ? keys.get(n.lines.colour[i] * 4096 + stretch[i]) : undefined;
      if (s) out[i] = Number.isNaN(s.wait) ? 15 * Math.min(1, s.share / 0.75) : s.wait;
    }
    this.anyFull = out.some((v) => v !== 0);
    this.rewrite(l.fullBuf, out);
  }
  private anyFull = false;

  /** Replace everything drawn but the trips (an edit). */
  setNetwork(n: NetworkBuffers) {
    const gl = this.gl;
    for (const r of this.res) r.del();
    this.res = [];
    for (const r of this.serviceRes) r.del();
    this.serviceRes = [];
    this.probeVao = null;
    this.net = n;
    this.marks.setNetwork(n.marks);
    this.track = this.strokes(n.track);
    this.lines = this.strokes(n.lines);
    this.fullDrawn = "";
    this.writeFull();
    this.turnback = n.turnbacks?.count ? this.strokes(n.turnbacks) : null;
    // blueprint track sorts last (network.ts)
    this.hasBlueprint = n.track.count > 0 && (n.track.flags[n.track.count - 1] & F_BLUEPRINT) !== 0;
    this.lineWidths = null;
    this.tex.phases = this.texture(n.phases.length ? n.phases : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.meta = this.texture(n.meta.length ? n.meta : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.samples = this.texture(n.samples.length ? n.samples : new Float32Array(2), "rg32f", TEX_W);
    this.tex.lane = this.texture(n.sampleLane?.length ? n.sampleLane : new Float32Array(Math.max(1, n.samples.length / 2)).fill(1), "r32f", TEX_W);
    this.tex.lines = this.texture(n.lineTable.length ? n.lineTable : new Float32Array(4), "rgba32f", LINES_W);
    this.tex.colours = this.texture(n.lineColours, "rgba8", LINES_W);
    this.tex.fill = this.texture(new Float32Array(Math.max(1, n.samples.length / 2) * 2).fill(-1), "rg32f", TEX_W);
    const cars = n.lineCars ?? new Float32Array(Math.max(1, n.lineTable.length / 4)).fill(8);
    this.maxCars = Math.max(1, ...cars);
    this.tex.cars = this.texture(cars, "r32f", LINES_W);
    // each train sample's stop-to-stop stretch (run 0's numbering), where a trip's loads are
    // looked up (T-164)
    const nSamples = Math.max(1, n.samples.length / 2);
    const stretch = new Float32Array(nSamples).fill(-1);
    if (n.sampleStroke && n.lines.stretch)
      for (let k = 0; k < nSamples; k++) {
        const i = n.sampleStroke[k];
        if (i >= 0) stretch[k] = n.lines.stretch[i];
      }
    this.tex.stretch = this.texture(stretch, "r32f", TEX_W);
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
    this.tex.stationLine = this.texture(n.stationLine.length ? n.stationLine : new Float32Array(2), "rg32f", TEX_W);
    this.attrib(this.stationProg, "a_shape", this.stationShapeBuf, 4, gl.FLOAT, 1);
    this.stationSizeBuf = this.buffer(new Float32Array(Math.max(1, n.stationCount)));
    this.attrib(this.stationProg, "a_sz", this.stationSizeBuf, 1, gl.FLOAT, 1);
    this.stationHaloBuf = this.buffer(new Float32Array(Math.max(1, n.stationCount) * 2));
    this.attrib(this.stationProg, "a_halo", this.stationHaloBuf, 2, gl.FLOAT, 1);
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
    for (const r of this.serviceRes) r.del();
    this.serviceRes = [];
    // the textures `setNetwork` made stay in `res` until the next one; these are freed here
    this.tex.phases = this.texture(phases.length ? phases : new Float32Array(4), "rgba32f", TEX_W, this.serviceRes);
    this.tex.meta = this.texture(meta.length ? meta : new Float32Array(4), "rgba32f", TEX_W, this.serviceRes);
  }
  private serviceRes: { del: () => void }[] = [];

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    this.previewSrc = s;
    for (const r of this.previewRes) r.del();
    this.previewRes = [];
    this.preview = s && s.count ? this.strokes(s, this.previewRes) : null;
    this.marks.setPreview(s?.marks);
  }

  /** The blueprint track picked by a drag selection (T-333), drawn as an ink band under the track, or null. */
  setHighlight(s: Strokes | null) {
    this.highlightSrc = s;
    for (const r of this.highlightRes) r.del();
    this.highlightRes = [];
    this.highlight = s && s.count ? this.strokes(s, this.highlightRes) : null;
  }

  /** What the setters were last given, for `restore`. */
  private layersSrc: NetworkLayers | null = null;
  private previewSrc: Strokes | null = null;
  private highlightSrc: Strokes | null = null;

  /** The canvas lost its context and the browser gave it a new one (T-953): every GL object went
   * with the old, so the programs are compiled again and what was drawn is uploaded again. */
  restore() {
    this.res = [];
    this.serviceRes = [];
    this.previewRes = [];
    this.highlightRes = [];
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

  private putTrips(epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array) {
    const old = this.tripsBy.get(epoch);
    if (old && old.version > version) return;
    const hour = board ? readBoardHour(epoch, board) : undefined;
    this.tripsBy.set(epoch, { version, data, ends: ends && ends.length === data.length / 2 ? ends : undefined, nexts: nexts && nexts.length === data.length / 2 ? nexts : undefined, board: hour && hour.trips === data.length / 2 ? hour : undefined });
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
    }
    // Until the new hour's trips arrive, the last hour's keep running on their own epoch.
    return time - this.tripEpoch;
  }

  /** The epoch the trip buffer is relative to. */
  get tripsEpoch() {
    return this.tripEpoch;
  }

  /**
   * Draw with the camera MapLibre last drew (gl.ts `Camera`, flat or globe); `time` is game seconds;
   * `zoom` and `dpr` size things in px. The caller has bound and cleared the target.
   */
  draw(cam: Camera, time: number, zoom: number, dpr: number, opt: DrawOptions) {
    const gl = this.gl;
    this.releaseStaged(time);
    const n = this.net;
    if (!n) return;
    const uTime = this.sync(time);
    this.bubbles.hide();
    // the network's buffers are relative to the anchor they were built at (T-307)
    const a = n.anchor ?? anchor();
    const lc = localCamera(cam, a);
    this.cam = cam;
    const vp =[gl.drawingBufferWidth, gl.drawingBufferHeight] as const;
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
    gl.useProgram(this.strokeProg);
    const u = this.strokeU;
    setCamera(gl, u, lc);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1i(u.u_sel_line, opt.selLine);
    gl.uniform1i(u.u_sel_edge, opt.selEdge);
    gl.uniform2i(u.u_hot, opt.hotLine ?? -1, opt.hotEdge ?? -1);
    gl.uniform1f(u.u_dim, opt.dimOthers ?? 1);
    gl.uniform1i(u.u_by_level, opt.byLevel ? 1 : 0);
    gl.uniform1i(u.u_by_speed, opt.bySpeed ? 1 : 0);
    // Lines sharing track are drawn over each other, the last on top (T-190); picking knows the
    // line width, and that no line strokes are drawn when the track is coloured by height or speed.
    this.slotPx = opt.byLevel || opt.bySpeed ? 0 : wLine / dpr;
    gl.uniform3fv(u.u_level_colours, this.levelColours);
    gl.uniform3fv(u.u_track, TRACK_RGB);
    gl.uniform3fv(u.u_bad, BAD_RGB);
    const shades = shadeColours();
    gl.uniform3fv(u.u_sel, rgbf(shades.mapInk));
    gl.uniform3fv(u.u_casing_rgb, rgbf(shades.casing));
    // T-819: 1.5 px of red either side, outside the casing, of a line on full track
    const fullPass = this.anyFull;
    gl.uniform1i(u.u_full_pass, 0);
    gl.uniform1f(u.u_full_px, 3 * dpr);
    gl.uniform3fv(u.u_full_rgb, rgbf(shades.fullTrack));
    gl.uniform1i(u.u_ghost, 0);
    gl.uniform3fv(u.u_ghost_rgb, rgbf(shades.ghost));
    gl.uniform1i(u.u_ghost_layer, -1);
    gl.uniform2f(u.u_ghost_min, GHOST_MIN_PX * dpr, 2 * GHOST_OUTLINE_PX * dpr);
    gl.uniform3f(u.u_hsr, interp(zoom, HSR_WIDTH), HSR_STRIPE_PX[0] * dpr, HSR_STRIPE_PX[1] * dpr);
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
    const lanes = laneState(zoom);
    this.laneReachPx = lanes.reach;
    this.laneShiftM = LANE_M / 2 * lanes.t;
    const G = lanes.gap * dpr;
    gl.uniform4f(u.u_lanes, G, lanes.t, lanes.grow, lanes.share * G);
    const modes = lanes.t > 0 ? [1, 2, 3] : [0];
    // `cap` > 0: the track drawn through station boxes (T-236), no wider than that and no casing
    const stroke = (s: StrokeVao | null, kind: number, width: number, sel: boolean, cap = 0) => {
      if (!s || !s.count) return;
      gl.uniform1i(u.u_kind, kind);
      // each track's width once apart: a line's share of the gap, bare track as much thinner
      gl.uniform3f(u.u_width, width, wSel, lanes.share * G * width / wLine);
      gl.uniform1f(u.u_cap, cap);
      gl.bindVertexArray(s.vao);
      // [casing px, selected only, T-819's red band under the lines on full track]
      const passes: [number, number, number][] = cap > 0 ? [[0, 0, 0]] : [[casing, 0, 0], [0, 0, 0]];
      if (kind === 1 && fullPass && cap === 0) passes.unshift([casing, 0, 1]);
      if (sel) passes.push(...(cap > 0 ? [[0, 1, 0]] as [number, number, number][] : [[casing, 1, 0], [0, 1, 0]] as [number, number, number][]));
      for (const [cas, only, full] of passes) {
        gl.uniform1f(u.u_casing, cas);
        gl.uniform1i(u.u_only_sel, only);
        gl.uniform1i(u.u_full_pass, full);
        for (const mode of modes) {
          if (cap > 0 && mode === 1) continue;
          if (full && mode > 1) continue;
          gl.uniform1i(u.u_lane_mode, mode);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
        }
      }
      gl.uniform1i(u.u_full_pass, 0);
      gl.uniform1i(u.u_lane_mode, 0);
    };
    // Colour by height: the track itself, at line width, and no line strokes over it. Blueprint
    // track and the lines planned on it go over built track and its lines (T-695).
    const trackStrokes = (cap = 0) => {
      for (const part of this.hasBlueprint ? [1, 2] : [0]) {
        gl.uniform1i(u.u_bp_part, part);
        stroke(this.track, 0, opt.byLevel || opt.bySpeed ? wLine : wTrack, opt.selEdge >= 0, cap);
        if (!opt.byLevel && !opt.bySpeed) stroke(this.lines, 1, wLine, opt.selLine >= 0, cap);
      }
      gl.uniform1i(u.u_bp_part, 0);
    };
    if (this.highlight) {
      // an ink band 3 px beyond the blueprint's rails either side (kind 2 placement strokes in a
      // casing pass are the selection ink)
      gl.uniform1i(u.u_kind, 2);
      gl.uniform3f(u.u_width, bpMid + 2 * bpRail + 6 * dpr, 0, lanes.share * G + 6 * dpr);
      gl.uniform1f(u.u_cap, 0);
      gl.uniform1i(u.u_only_sel, 0);
      gl.uniform1f(u.u_casing, 0.01 * dpr);
      // round both tracks once they part
      gl.uniform1i(u.u_lane_mode, lanes.t > 0 ? 1 : 0);
      gl.bindVertexArray(this.highlight.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.highlight.count);
      gl.uniform1i(u.u_lane_mode, 0);
    }
    trackStrokes();
    // where lines turn back, over their tracks (T-658d)
    if (lanes.t > 0) stroke(this.turnback, 0, wTrack, false);
    gl.uniform1i(u.u_ghost, 1);
    stroke(this.preview, 2, wLine, false);
    if (this.preview?.count) {
      // the ghost: blueprint's outline, rails and middle, each over the last (T-1054)
      gl.uniform1f(u.u_casing, 0);
      gl.uniform1f(u.u_cap, 0);
      gl.uniform1i(u.u_only_sel, 0);
      gl.bindVertexArray(this.preview.vao);
      for (let layer = 0; layer < 3; layer++) {
        gl.uniform1i(u.u_ghost_layer, layer);
        for (const mode of modes) {
          if (mode === 1) continue;
          gl.uniform1i(u.u_lane_mode, mode);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.preview.count);
        }
      }
      gl.uniform1i(u.u_ghost_layer, -1);
      gl.uniform1i(u.u_lane_mode, 0);
    }
    gl.uniform1i(u.u_ghost, 0);
    this.marks.draw(lc, vp, zoom, dpr, this.levelColours, opt.flatMarks !== false);

    if (n.stationCount) {
      // Platforms along the track from STATION_BOX_ZOOM, each one shape at every zoom (T-204), dots
      // further out (T-189), one for a whole complex (T-217), and below STATIONS_ZOOM only the
      // selected ones (T-205).
      const boxes = zoom >= STATION_BOX_ZOOM;
      const hidden = zoom < STATIONS_ZOOM;
      // a dot's rim is thinner than a box's (T-322)
      const ring = boxes ? stationRing(zoom) : dotRing(zoom);
      const boxLinePx = interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx = stationRing(BOX_WIDTH_ZOOM);
      this.stationStyle = { boxes, radius: stationRadii(zoom), linePx: wLine / dpr, ringPx: ring, boxLinePx, boxRingPx, hidden };
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
      const project = this.projector(cam, a, vp[0] / dpr, vp[1] / dpr);
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
      gl.uniform2f(su.u_fixed, boxHalfPxAtBoxZoom({ boxLinePx, boxRingPx }) * dpr, unitAtBox);
      gl.bindVertexArray(this.stationVao);
      // Analytical circles behind the normal, still-selectable platform boxes.
      gl.uniform1f(su.u_halo_radius, stationHaloRadius(zoom) * dpr);
      gl.uniform1f(su.u_pixel, dpr);
      gl.uniform1i(su.u_halo, 1);
      gl.uniform1i(su.u_layer, 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
      gl.uniform1i(su.u_halo, 0);
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
      const any = this.crowdsShown && this.crowds.update(n, this.board!, uTime, (c, dc) => {
        if (this.crowdBuf) this.rewrite(this.crowdBuf, c);
        if (this.dotCrowdBuf) this.rewrite(this.dotCrowdBuf, dc);
      });
      const fills = any && boxes;
      gl.useProgram(this.stationProg);
      gl.uniform1f(su.u_tint, any && !boxes ? 1 : 0);
      gl.uniform1f(su.u_mask, 0);
      // The track through the boxes (T-236, Anita: "maybe the track on top of the station"): the
      // track and line strokes drawn again, no wider than TRACK_IN_BOX of the box's white and
      // without casing, where the stencil marks a station's white (1 the unselected, 2 the selected).
      const through = boxes && this.hasStencil;
      // every box has one width on the ground (T-239); any station's gives it on screen here
      const boxWhite = 2 * ((boxes ? this.stationShapeOf(0, project).halfWidth : 0) - ring);
      const trackCap = Math.max(1, TRACK_IN_BOX * boxWhite) * dpr;
      if (through) {
        gl.clearStencil(0);
        gl.clear(gl.STENCIL_BUFFER_BIT);
      }
      const layerDraw = (pass: number, layer: number, withBridges: boolean) => {
        gl.useProgram(this.stationProg);
        gl.uniform1i(su.u_pass, pass);
        gl.uniform1i(su.u_layer, layer);
        gl.bindVertexArray(this.stationVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
        if (bridges && withBridges) {
          gl.useProgram(this.bridgeProg);
          gl.uniform1i(this.bridgeU.u_pass, pass);
          gl.uniform1i(this.bridgeU.u_layer, layer);
          gl.bindVertexArray(this.bridgeVao);
          gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, bridges);
        }
      };
      // every station, then the selected ones over them with their outline and glow; each pass the
      // dark shapes, then the white inside every rim, so stations drawn over each other are one white
      // shape with one outline (T-217), then the crowds and the track through them (T-236)
      for (const pass of [0, 1]) {
        if (pass === 0 && hidden) continue;
        if (pass === 1) {
          const shapes = this.selectedShapes(project);
          if (!shapes.length) break;
          this.drawGlow(shapes, dpr, vp);
        }
        // zoomed out the connectors' white goes under the dots' white, so a dot's crowd pie (T-228)
        // stays whole (T-313)
        if (connectors) connDraw(pass, 0);
        layerDraw(pass, 0, true);
        if (connectors) connDraw(pass, 1);
        layerDraw(pass, 1, true);
        if (!boxes) continue;
        if (through) {
          // mark this pass's white, no colour
          gl.enable(gl.STENCIL_TEST);
          gl.stencilFunc(gl.ALWAYS, pass + 1, 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
          gl.colorMask(false, false, false, false);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 1);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 1);
          }
          layerDraw(pass, 1, true);
          gl.useProgram(this.stationProg);
          gl.uniform1f(su.u_mask, 0);
          if (bridges) {
            gl.useProgram(this.bridgeProg);
            gl.uniform1f(this.bridgeU.u_mask, 0);
          }
          gl.colorMask(true, true, true, true);
          gl.disable(gl.STENCIL_TEST);
        }
        if (fills) layerDraw(pass, 2, false);
        if (through) {
          gl.enable(gl.STENCIL_TEST);
          gl.stencilFunc(gl.EQUAL, pass + 1, 0xff);
          gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
          gl.useProgram(this.strokeProg);
          trackStrokes(trackCap);
          gl.uniform1f(u.u_cap, 0);
          gl.disable(gl.STENCIL_TEST);
        }
      }
      // riders changing trains walking between stations (T-1124), with the platform crowds
      const placed = this.crowds.stationOfQueue;
      if (this.crowdsShown && boxes && placed) {
        this.bubbles.sync(this.board, n, placed);
        this.bubbles.draw(lc, vp, uTime, 0.5 * boxWhite * dpr, CROWD_RGB, rgbf(shadeColours().casing), dpr);
      }
    }

    // No trains zoomed far out (T-205).
    this.trainsHidden = zoom < TRAINS_ZOOM;
    if (opt.showTrains && this.tripCount && !this.trainsHidden) {
      this.detailedTrains = zoom >= STATION_BOX_ZOOM;
      const tw = interp(zoom, TRAIN_PX) * dpr * k;
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
    gl.uniform1i(bu.u_halo, 0);
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
