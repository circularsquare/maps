// Constants, option types and helpers of the network renderer (renderer.ts), moved out of it (T-2374).

import type { SlotLayout } from "./strokeSlots";
import { BAD_RED, rgbf } from "../game/palette";
import { PASS_GAP_PX, PASS_TRACK_M, STATION_BOX_ZOOM } from "./stationGeometry";
import { kAtY, pxPerUnitAt } from "./geo";
import { interp } from "./gl";
import { type ChunkBoxes } from "./strokeLod";
import { perfOff } from "../perfFlags";
import { type Strokes } from "./network";
import { BAY_STRIDE } from "./terminalBays";
import { LINE_VIEW_WIDTH, type TrafficBands } from "./lineBands";
import { LANE_SHARE, SEL_SET_MAX } from "./rendererShaders";

/** The render epoch moves to the start of each game hour. */
export const EPOCH_S = 3600;
/** The longest a staged state waits for its trips, wall ms (T-1071). */
export const STAGE_WAIT_MS = 2000;
export type TripArgs = [epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array, bays?: Float32Array];
export const TRACK_RGB = rgbf("#77746f");
export const BAD_RGB = rgbf(BAD_RED);
/** Blueprint track (T-322): a light blue middle as wide as a line, with a dark blue rail this
 * share of a line width (at least a pixel) along each edge. */
export const BP_MID_RGB = rgbf("#a9cdf3");
export const BP_RAIL_RGB = rgbf("#2f62b3");
export const BP_RAIL_SHARE = 0.16;
/** The whole blueprint track, rails included, is this share of its full width (T-331); the rails
 * stay at least a pixel. */
export const BP_NARROW = 0.75;
/** A train's width on screen by zoom, CSS px, pills and cars alike (about 1.4 lines wide from zoom
 * 12); its true 3 m is wider only from about zoom 19. */
export const TRAIN_PX: [number, number][] = [[10, 3], [11, 5.25], [12, 6], [15, 9]];
/** A line's width on screen by zoom, CSS px. Bare track, the casing and a selected line's extra
 * width follow the same steps. */
export const LINE_PX: [number, number][] = ([[9, 0.85], [12, 3.9], [15, 6.6]] as [number, number][]).map(([z, px]) => [z, px * LINE_VIEW_WIDTH]);
/** T-1097 (Anita: "a tad more visual weight, not much"): high-speed track and its lines this many
 * times as wide by zoom while drawn as one stroke, and their centre line only on strokes this wide
 * (css px, coming in between the two) */
export const HSR_WIDTH: [number, number][] = [[9, 1.4], [11, 1.25], [13, 1]];
export const HSR_STRIPE_PX = [3, 5] as const;

/** The Speed view's value per segment (T-1271): the trains' speed where a line with trains runs,
 * else the limit. */
export const selSetBuf = new Int32Array(SEL_SET_MAX);
/** T-2980: per trip, no place at either end */
export const noBays = (n: number) => new Float32Array(Math.max(1, n) * BAY_STRIDE).fill(-1);
/** T-1812: the lines of a selected group, for the stroke and train shaders' `u_sel_set` */
export function setSelLines(gl: WebGL2RenderingContext, u: Record<string, WebGLUniformLocation | null>, ls: readonly number[] | undefined) {
  const n = Math.min(ls?.length ?? 0, SEL_SET_MAX);
  for (let i = 0; i < n; i++) selSetBuf[i] = ls![i];
  gl.uniform1i(u.u_sel_n, n);
  gl.uniform1iv(u.u_sel_set, selSetBuf);
}

export function shownSpeed(limit: Float32Array, trains?: Float32Array): Float32Array {
  if (!trains || trains.length !== limit.length) return limit;
  // (a loop: `map` with a closure took 18 ms over her Asia save's track in the frame after each edit, T-4262)
  const out = new Float32Array(limit.length);
  for (let i = 0; i < limit.length; i++) {
    const t = trains[i];
    out[i] = t >= 0 ? t : limit[i];
  }
  return out;
}
/** The ghost of track being placed (T-1038, Anita: "a decent minimum width ... see the track in full
 * length"), drawn as blueprint track with a thin dark outline (T-1054): never thinner than this
 * across its rails, CSS px, with this much outline either side. */
export const GHOST_MIN_PX = 4, GHOST_OUTLINE_PX = 1;
/** Double track's two tracks (T-658): their centres 4 m apart on the ground, each a line's width
 * of LANE_SHARE of that (2.4 m) once apart. Where 4 m is LANE_GROW_PX on screen a double track's
 * stroke starts growing to the pair's width, which it reaches at LANE_PART_PX, where the two tracks
 * start parting; they are fully apart at LANE_APART_PX (zoom 16.7 to 17.2 in New York). */
export const LANE_M = 4, LANE_GROW_PX = 4, LANE_PART_PX = 7, LANE_APART_PX = 10;
/** Past LANE_APART_PX each track narrows by up to LANE_NARROW (T-713, Anita: "reduce the width of
 * the individual tracks by around 20%" so the riders waiting behind them show), reached at
 * LANE_NARROW_PX; trains keep their 3 m. */
export const LANE_NARROW = 0.2, LANE_NARROW_PX = 16;
/** T-1799 (Anita, 2026-10-29): lines sharing track fold into one stroke as a line's width falls
 * through these CSS px zoomed out (about zoom 9.5 to 10) */
export const BAND_FOLD_PX = [1.25, 1.75] as const;
/** T-2058: with a line picked, the others' width share and how far their colour is mixed towards the
 * casing's, drawn solid (Anita: "thin and faded"); `?perfOff=muteThin`: at `dimOthers` opacity instead */
export const MUTE: readonly [number, number] | null = perfOff("muteThin") ? null : [0.45, 0.5];
/** T-2126: bands side by side with a pixel between (`?perfOff=bandGap`: touching, as before) */
export const BAND_GAP = !perfOff("bandGap");
/** T-2980: trains take a place at their line's ends (`?perfOff=bays`: all at the end of the arrival track, as before) */
export const BAYS = !perfOff("bays");
export const smoothstep = (a: number, b: number, x: number) => {
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
export const CROWD_RGB = rgbf("#a4a19d");
/** The track drawn through a station box (T-236) is at most this share of the box's white, so the
 * crowd shows beside it. */
export const TRACK_IN_BOX = 0.3;
/** Stations as the complex rule's distances group them (game/complexView.ts `dotGroups`): zoomed out
 * each group's dots are joined by connectors, the shortest set that links them all (T-313). */
export interface StationGroups {
  members: string[][];
}

export interface DrawOptions {
  showTrains: boolean;
  /** T-3625: false draws no station, bridge, car park or bus stop at a station, and none can be picked; absent: shown */
  showStations?: boolean;
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
  /** T-3890: bus stops lit (ids), the hovered and the selected stop's places */
  hotStops?: number[];
  /** T-1248: lines, track and trains drawn this many times their width; absent: 1 */
  lineScale?: number;
  /** T-1567: every train one pill, white with a rim of its line's colour (the start menu's logo) */
  hollowTrains?: boolean;
  /** T-1700: every box this many px narrower on each side (the start menu's logo) */
  boxShrinkPx?: number;
}

export interface StrokeVao {
  vao: WebGLVertexArrayObject;
  count: number;
  /** the per-segment width multipliers, rewritten by `setLayers` */
  wmulBuf: WebGLBuffer;
  /** T-4262: `wmulBuf` holds the plain widths (`plainWmul`), or else `wmulHeld`, a copy of what it holds */
  wmulPlain?: boolean;
  wmulHeld?: Float32Array | null;
  /** T-4262 stage 2: a network's set, where its segments sit in its buffers (strokeSlots.ts), and the slots drawn */
  layout?: SlotLayout;
  drawn?: number;
  /** T-4262: the last set's culling boxes, whose chunks before segment `upTo` are this set's too (`chunkBoxes`) */
  boxesFrom?: { boxes: ChunkBoxes; upTo: number };
  /** per segment its traffic mark (T-819, `writeFull`) and where along its run it starts (`fullAttr`) */
  fullBuf: WebGLBuffer;
  /** the network's strokes (T-1028): each attribute's buffer and bytes a segment, in `STROKE_ATTRIBS`
   * order, for the next network to copy its unchanged runs from; and what to free with them */
  attribs?: { buf: WebGLBuffer; bytes: number }[];
  res?: { del: () => void }[];
}

export type TexName = "phases" | "meta" | "samples" | "bendAt" | "bends" | "lines" | "colours" | "fill" | "cars" | "stretch" | "stationLine" | "lane" | "slevel";

/** `a_wmul` for `n` plain segments: width 1, no Traffic view band shift (T-1799) */
export function plainWmul(n: number): Float32Array {
  const v = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) v[i * 3] = 1;
  return v;
}

/** `a_full` for `n` segments (2 a segment: the traffic mark, from `marks` or 0, and the local units
 * from the start of its run of joined segments of one line and edge to its start, which the dashes
 * are measured along so they keep one length round curves, T-2184); with `start`, segments `start` on,
 * a run taken to begin there (a network piece's first segment always begins one: no two pieces share an
 * edge and line) */
export function fullAttr(s: { seg: Float32Array; colour: Uint32Array; edge: Uint32Array }, n: number, marks?: (i: number) => number, start = 0): Float32Array {
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
export const STROKE_ATTRIBS = ["seg", "colour", "edge", "level", "flags", "speed", "lane", "band"] as const;

/** T-1639: the network's strokes drawn simplified below this zoom (plus log2 of the pixel ratio),
 * merged where they stray from the full set by at most LOD_TOL_PX device px; from CULL_ZOOM only the
 * chunks near the view drawn, CULL_MARGIN_PX (CSS) round it covering the widest stroke */
export const LOD_TOP_ZOOM = 15, LOD_TOL_PX = 0.25, CULL_ZOOM = 7, CULL_MARGIN_PX = 48;
/** T-1644: a set's visible instances are gathered into one buffer (strokeGather.ts) when there are at most
 * this many; more are drawn by ranges, a gap of RANGE_GAP instances or fewer drawn through */
export const GATHER_MAX = 40000, RANGE_GAP = 512;
/** the finest chunk boxes tested at `zoom` (strokeLod.ts level 0 is a chunk): below zoom 12 the globe's
 * projection makes each box test dear, and boxes of 128 or 1024 instances are fine enough */
export const cullLevel = (zoom: number) => (zoom < 10 ? 2 : zoom < 12 ? 1 : 0);
/** T-1640: the longest a still picture goes undrawn, ms; the fastest train's speed, m/s; the game
 * seconds a platform's crowd is drawn in steps of */
export const STILL_MAX_MS = 1000, TRAIN_MAX_MPS = 100, CROWD_STEP_S = 5;

/** a network stroke set's chunk boxes, and its simplified copies by zoom band (T-1639) */
export interface StrokeLodState {
  boxes: ChunkBoxes | null;
  speed: Float32Array | null | undefined;
  sig: Float32Array | null;
  /** how far `sig` is worked out, and whether that work is under way */
  sigAt: number;
  working: boolean;
  bands: Map<number, StrokeBand>;
  /** the zoom band whose copy is being made (`workBand`), null for none */
  making: number | null;
  /** T-4274: the zoom band whose copy was drawn last and the stroke redraw that drew it (`strokeRedraws`): an edit carries
   * that copy while the view shows it (`carryLod`) */
  drawn?: number;
  drawnAt?: number;
}
/** a simplified copy of a stroke set for one zoom band, with the arrays its buffers hold (T-1644) */
export interface StrokeBand { vao: StrokeVao; strokes: Strokes; first: Uint32Array; boxes: ChunkBoxes; v: number; res: { del: () => void }[]; wm: Float32Array | null; fu: Float32Array | null }

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
