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

import { LEVEL_COLOURS, rgbf } from "../game/palette";
import { ANNEX_ROOM, stationAnnexes, stationCoords, ST_BUFFER, ST_BUILDING, ST_ONE, ST_ONE_MID, ST_ONE_NEG, ST_PASSING, ST_TRAM, ST_ROOM, ST_TERMINAL, TERMINAL_CAP, BOX_WIDTH_ZOOM, bridgeShape, connectorHalfPx, insideStation, PLATFORM_POINTS, platformPoints, STATIONS_ZOOM, TRAINS_ZOOM, stationRadii, stationReachPx, stationRing, stationShape, type Project, type StationShape, type StationStyle } from "./stationGeometry";
import { shadeColours } from "../game/shades";
import { readBoardHour, trainStands, type BoardHour, type CrowdedStretch } from "../game/boarding";
import { UPGRADE } from "../game/upgrades";
import { stepSince, timed } from "../game/stepTimes";
import { checkPatches } from "../game/worldDiff";
import { PlatformCrowds } from "./platformCrowds";
import { TransferBubbles } from "./transferBubbles";
import { BusLayer } from "./busLayer";
import { pxPerUnitAt } from "./geo";
import { compile, GpuTimer, interp, PROJECT_UNIFORMS, uniforms, type Camera } from "./gl";
import { GradeMarks } from "./gradeMarks";
import { OneWayChevrons } from "./oneWayMarks";
import { StrokeCache } from "./strokeCache";
import { StrokeGather } from "./strokeGather";
import { perfOff } from "../perfFlags";
import { ParkingLots } from "./parkingLots";
import { BuildingFloors } from "./buildingFloors";
import { TrainGuard } from "./trainGuard";
import { LOT_BLUEPRINT, LOT_STRIDE } from "./facilityPlace";
import { backBaseOf, F_BLUEPRINT, platformShapeAt, sampleValue, stationIndexOf, type NetPatch, type NetworkBuffers, type Strokes } from "./network";
import { BEND_LIMIT, PHASE_ROOM, bendRows, blankSlots, checkSlots, phaseRows, phasesFromFirst, planSlots, putSlots, sampleRows, slottedLineTable, slottedMeta, trainNamesTest, type SlotLayout, type SlotNames, type SlotSrc, type SlotTex } from "./trainSlots";
import type { EditTouched } from "../workers/protocol";
import { aheadChevrons, aheadEmpty, aheadParts, joinStations, noAhead, standInStations, type AheadModel } from "./lineAhead";
import { BAY_STRIDE } from "./terminalBays";
import { BAND_RANGE, TRAFFIC_PICK_DIV, type TrafficBands } from "./lineBands";
import { BRIDGE_VS, HOT_STATIONS, LINES_W, PROBE_FS, PROBE_VS, SEL_FS, SEL_MAX, SEL_STRIDE, SEL_VS, STATION_FS, STATION_VS, STROKE_FS, STROKE_VS, TEX_W, TRAIN_FS, TRAIN_VS } from "./rendererShaders";
import { codeHidden, hidden as vis, levelShown, lotOff, modesHidden, pickSkip, stationOff, strokeMode } from "./visibility";
import { edgesNear } from "./snapIndex";
import { MARK_STRIDE } from "./markData";
import { EPOCH_S, STAGE_WAIT_MS, TripArgs, LINE_PX, noBays, BAYS, StationGroups, DrawOptions, StrokeVao, TexName, plainWmul, fullAttr, STROKE_ATTRIBS, STILL_MAX_MS, StrokeLodState, NetworkLayers, TRAIN_MAX_MPS, CROWD_STEP_S, type StrokeBand } from "./rendererParts";
export { EPOCH_S, laneState, passingState, CROWD_ZOOM, CONFLICT_ZOOM } from "./rendererParts";
export type { StationGroups, DrawOptions, NetworkLayers } from "./rendererParts";
import { texture, buffer, attrib, strokes, checkStrokes, checkFull, checkTextures, carryLod, dropLodOf, keepTexture, changedRows, unchangedRuns, writeSegs, writeSegRange, type KeptTex, type TexKind } from "./rendererStrokes";
import { drawFrame, bindTrainTextures } from "./rendererDraw";

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

/** T-4400: a mark is on a track that passes within this many local units of it (marks sit on their track's points) */
const MARK_ON_TRACK = 1.5;

export class NetworkRenderer {
  gl!: WebGL2RenderingContext;
  strokeProg!: WebGLProgram;
  stationProg!: WebGLProgram;
  trainProg!: WebGLProgram;
  strokeU!: Record<string, WebGLUniformLocation | null>;
  stationU!: Record<string, WebGLUniformLocation | null>;
  trainU!: Record<string, WebGLUniformLocation | null>;
  probeProg: WebGLProgram | null = null;
  probeU: Record<string, WebGLUniformLocation | null> = {};
  /** crossings, flat junctions, ramps and portals (T-629, T-655) */
  readonly marks = new GradeMarks();
  readonly oneWay = new OneWayChevrons();
  /** T-3094: along one-way single track */
  readonly oneWayTrack = new OneWayChevrons();
  /** T-3137: along the drawing tool's ghost, where the track it lays would run one way */
  readonly oneWayGhost = new OneWayChevrons();
  /** the complexes' car parks (T-1528) */
  readonly lots = new ParkingLots();
  /** the station buildings' floors (T-2218) */
  readonly floors = new BuildingFloors();
  res: { del: () => void }[] = [];
  previewRes: { del: () => void }[] = [];
  track: StrokeVao | null = null;
  lines: StrokeVao | null = null;
  /** the crossovers where lines turn back (T-658d), drawn close in only */
  turnback: StrokeVao | null = null;
  preview: StrokeVao | null = null;
  highlightRes: { del: () => void }[] = [];
  refusedRes: { del: () => void }[] = [];
  refused: StrokeVao | null = null;
  /** the drag-selected blueprint (T-333): an ink band drawn under the track */
  highlight: StrokeVao | null = null;
  stationVao: WebGLVertexArrayObject | null = null;
  trainVao: WebGLVertexArrayObject | null = null;
  probeVao: WebGLVertexArrayObject | null = null;
  tripBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: when it stops being drawn, relative to the epoch (T-206) */
  endBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: its first load and when it reaches its last stop (T-164) */
  loadBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: its stand at the terminal before its next trip (T-1785, `trainStands`) */
  standBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: seconds it is drawn behind its timetable (T-1965, `guard.lag`) */
  lagBuf: WebGLBuffer | null = null;
  /** per trip in the trip buffer: where it stands at its ends (T-2980, `assignBays`) */
  bayBuf: WebGLBuffer | null = null;
  lagDirty = false;
  /** T-1965: trains on the same track drawn one behind the other, never through each other; off
   * with `?perfOff=follow` or live */
  guard = new TrainGuard();
  follow = !perfOff("follow");
  /** game seconds trip `i` of the buffer is drawn behind its timetable (T-1965) */
  tripLag(i: number): number {
    return this.follow && i >= 0 ? this.guard.lag[i] ?? 0 : 0;
  }
  /** the hour's loads (T-164), replaced with the trips; kept apart from `res`, like the trips */
  loadTex: WebGLTexture | null = null;
  /** T-2420: the Profit view's figure per edge id, and how many ids it holds */
  profitTex: WebGLTexture | null = null;
  profitN = 0;
  profitData: Float32Array | null = null;
  tex: Record<TexName, WebGLTexture | null> = { phases: null, meta: null, samples: null, bendAt: null, bends: null, lines: null, colours: null, fill: null, cars: null, stretch: null, stationLine: null, lane: null, slevel: null };
  /** The boarding account's hour the trip buffer holds (T-164), or null before demand has riders. */
  board: BoardHour | null = null;
  crowds = new PlatformCrowds(0);
  crowdBuf: WebGLBuffer | null = null;
  dotCrowdBuf: WebGLBuffer | null = null;
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
  crowdsAt = 0;
  crowdsAny = false;
  maxCars = 8;
  /** Used by picking to follow the same close-up representation. */
  detailedTrains = false;
  stationShapeBuf: WebGLBuffer | null = null;
  /** how the last draw sized stations, CSS px (picking, the cut-out above the demand bubbles) */
  stationStyle: StationStyle = { boxes: false, radius: stationRadii(12), linePx: 0, ringPx: stationRing(12), boxLinePx: interp(BOX_WIDTH_ZOOM, LINE_PX), boxRingPx: stationRing(BOX_WIDTH_ZOOM) };
  /** zoomed out below STATIONS_ZOOM at the last draw: trains neither drawn nor picked (T-205) */
  trainsHidden = false;
  /** T-3625: the stations are switched off in Settings (set each draw) */
  stationsOff = false;
  selectedStation = -1;
  /** `DrawOptions.selStations` as a set, and as drawn into `stationSelBuf` (its key) */
  selectedAlso = new Set<number>();
  selectedAlsoKey = "";
  stationSelBuf: WebGLBuffer | null = null;
  /** what `stationSelBuf` and `bridgeFlagBuf` were last written for */
  selDrawnKey = "";
  /** the groups the connectors were last worked out for (T-313) */
  groupSrc: { groups: StationGroups | null; net: NetworkBuffers | null } = { groups: null, net: null };
  bridgeProg!: WebGLProgram;
  bridgeU: Record<string, WebGLUniformLocation | null> = {};
  bridgeVao: WebGLVertexArrayObject | null = null;
  bridgeFlagBuf: WebGLBuffer | null = null;
  /** zoomed out, the connectors between dots (T-313): the two stations of each (index pairs), their
   * dots' centres (local units), and its VAO with the selected and faded flags; rebuilt by
   * syncGroups, so kept apart from `res` */
  connPairs = new Int32Array(0);
  connEnds = new Float32Array(0);
  connVao: WebGLVertexArrayObject | null = null;
  connFlagBuf: WebGLBuffer | null = null;
  connRes: { del: () => void }[] = [];
  selProg!: WebGLProgram;
  selU: Record<string, WebGLUniformLocation | null> = {};
  selTex: WebGLTexture | null = null;
  selData = new Float32Array(SEL_MAX * SEL_STRIDE * 4);
  /** Station `i` on screen as the last draw drew it (stationGeometry.ts): its platform, or zoomed
   * out its own dot (T-313). */
  drawnShapeOf(i: number, project: Project): StationShape {
    return this.stationShapeOf(i, project);
  }
  /** The bridges between linked stations (T-217) drawn selected, on screen: both their stations
   * are; zoomed out, the connectors between selected dots (T-313). */
  selectedBridges(project: Project): StationShape[] {
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
    const one = (flag & ST_ONE) === 0 ? 0 : (flag & ST_ONE_NEG) !== 0 ? -1 : 1;
    return stationShape(n.stations[i * 3], n.stations[i * 3 + 1], platformShapeAt(n, i), n.stationShape?.[i * 4 + 3] ?? 0, project, this.stationStyle, (flag & 1) !== 0, !plain && (flag & ST_PASSING) !== 0, plain ? 0 : one, { tram: (flag & ST_TRAM) !== 0, mid: (flag & ST_ONE_MID) !== 0 });
  }
  hotList = new Int32Array(HOT_STATIONS);
  /** The bridge program's hovered instances (T-730): the pairs in `pairs` (station indices, two
   * an instance) with both stations hovered. */
  hotPairs(bu: Record<string, WebGLUniformLocation | null>, pairs: Int32Array, hot: number[]) {
    let k = 0;
    this.hotList.fill(-1);
    if (hot.length > 1) for (let j = 0; j * 2 < pairs.length && k < HOT_STATIONS; j++) {
      if (hot.includes(pairs[j * 2]) && hot.includes(pairs[j * 2 + 1])) this.hotList[k++] = j;
    }
    this.gl.uniform1iv(bu.u_hot_st, this.hotList);
    this.gl.uniform1i(bu.u_hot_n, k);
  }

  /** Whether station `i` is drawn selected: the selected station, or in complex view its complex. */
  drawnSelected(i: number) {
    return i === this.selectedStation || this.selectedAlso.has(i);
  }
  /** The topmost station drawn at a point, with a small pointer tolerance; demand halos are not
   * targets. Selected stations are drawn last (T-189), so they come first; zoomed far out they are
   * the only ones drawn (T-205). Zoomed out each dot gives its own station (T-313): of dots that
   * overlap, the one whose centre is nearest. */
  stationAt(px: number, py: number, project: Project, padding = 7): number {
    const n = this.net;
    if (!n || this.stationsOff) return -1;
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
        if (this.drawnSelected(i) !== sel || this.stationHidden(i)) continue;
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
  reachOf: StationReach | null = null;
  /** Per station, local units from its node to the farthest point of its platform's centre line,
   * and the widest true half width of any (stationAt's skip, T-724); worked out when first asked
   * for, a few ms where scanning every station's shape cost Japan's network 100 ms a move (T-1020). */
  stationReachNow(n: NetworkBuffers): { r: Float32Array; halfWidth: number } {
    if (this.reachOf?.net === n) return this.reachOf;
    return timed("station.reach", () => (this.reachOf = stationReachStep(n, this.reachOf, this.checkPatch || checkPatches())));
  }
  /** The selected stations drawn, by index: the selected one, or its complex in complex view. */
  selectedDrawn(): number[] {
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
  annexesOf(i: number, s: StationShape) {
    return this.stationStyle.boxes ? stationAnnexes(s, Math.round(this.net!.stations[i * 3 + 2]), this.stationStyle.ringPx) : [];
  }
  /** The upgrade drawn at a point on station `i` (T-1416), its UPGRADE bit, or 0: the waiting room,
   * a terminal's thick end, or the car park whose hover names station `i` (T-1528). */
  upgradeAt(i: number, px: number, py: number, project: Project): number {
    const n = this.net;
    if (!n || this.stationsOff || !this.stationStyle.boxes || i < 0 || i >= n.stationCount) return 0;
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
    return this.stationStyle.boxes && !this.stationsOff ? this.lots.at(px, py, project, (k) => this.lotHidden(k)) : -1;
  }
  /** A selected station drawn at a point, or -1 (picking: it is above the demand bubbles). */
  selectedStationAt(px: number, py: number, project: Project, padding = 2): number {
    if (this.stationsOff) return -1;
    for (const i of this.selectedDrawn()) {
      if (this.stationHidden(i)) continue;
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
  /** the last map draw drew the bus and ferry lines (picking) */
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
  passing = false;
  hadSamplePass = false;
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
  trafficBands: TrafficBands | null = null;
  trafficReach = 0;
  /** per line stroke segment, its width as a multiple of `slotPx` (traffic thickness), or null
   * when all are plain: picking reaches across wide lines (T-227) */
  lineWidths: Float32Array | null = null;
  /** the network has blueprint track: it and its lines are drawn after the rest (T-695) */
  hasBlueprint = false;
  /** ST_ROOM if some box draws a waiting room (T-1416), drawn in a station draw of its own */
  annexes = 0;
  levelColours = new Float32Array([3, 2, 1, 0, -1, -2, -3].flatMap((l) => [...rgbf(LEVEL_COLOURS[l as 3])]));
  get ink() { return rgbf(shadeColours().stationInk); }
  timer!: GpuTimer;
  /** the canvas has a stencil buffer: the track is drawn through station boxes (T-236) */
  hasStencil = false;
  /** T-3192: the largest stroke cache this context can hold, device px a side; a larger view draws its strokes each frame */
  cacheMax = 0;
  /** trips per epoch from the worker, newest version kept, with the boarding account's hour */
  tripsBy = new Map<number, { version: number; data: Float32Array; ends?: Float32Array; nexts?: Float32Array; bays?: Float32Array; board?: BoardHour }>();
  /** what the trip buffer holds */
  tripEpoch = NaN;
  tripVersion = -1;
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
  snapCache: StrokeCache | null = null;
  /** GL uploads and deletions made outside the per-frame passes (`quiet`): what the network,
   * preview, highlight, demand views and the like put on the GPU, so the caches know to redraw */
  uploads = 0;
  quiet = false;
  /** T-4262: bytes sent to the GPU, copied on it and allocated there since the page opened (checks and profiles read them) */
  gpuBytes = { up: 0, copy: 0, alloc: 0 };
  /** T-4274: the last network's simplified copies carried through its edit (`carryLod`): segments whose significance was
   * worked out, of the set's, merged segments written, copies, ms; null when none were */
  lodCarried: { worked: number; of: number; wrote: number; bands: number; ms: number } | null = null;
  /** T-4262 stage 2: stroke sets laid out whole (a network posted whole, or room run out near an edit) */
  slotLayouts = 0;
  /** T-4262: a network takes the GL objects of the one before and writes only what differs (`?perfOff=keepGl`: each
   * network's textures and stroke buffers made anew, its traffic marks and lanes worked out whole, as before) */
  keepGl = !perfOff("keepGl");
  /** T-1639: off with `?perfOff=strokeCull` and `?perfOff=strokeLod` */
  strokeCull = !perfOff("strokeCull");
  strokeLod = !perfOff("strokeLod");
  lodOf = new Map<StrokeVao, StrokeLodState>();
  /** T-1644: off with `?perfOff=strokeGather` */
  strokeGather = !perfOff("strokeGather");
  /** T-1799: lines sharing track side by side; T-1886: the Settings switch, off by default (main.ts) */
  lineBands = false;
  trafficWidths = false;
  gathers = new Map<StrokeVao, StrokeGather>();
  speedShown = new WeakMap<Strokes, Float32Array>();
  /** the line strokes' demand widths (3 a segment: width, the Traffic view's band shift at its start
   * and end) and traffic marks as last written (null: the defaults), and a count of their writes,
   * which the simplified copies follow */
  linesWmul: Float32Array | null = null;
  linesFull: Float32Array | null = null;
  perSegV = 0;
  /** T-4274: stroke redraws so far (the simplified copy an edit carries is the one the last drew) */
  strokeRedraws = 0;
  /** stroke instances the last stroke redraw drew, and ms spent simplifying (checks read them) */
  strokeInstances = 0;
  lodMs = 0;
  strokeLoc: Record<string, number> = {};

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.mainCache = this.snapCache = null;
    const g = gl as unknown as Record<string, unknown> & { twCounted?: boolean };
    if (!g.twCounted) {
      g.twCounted = true;
      const view = (v: unknown) => (ArrayBuffer.isView(v) ? v : null);
      /** bytes a call sends, copies or allocates (T-4262), as `gpuBytes` counts them */
      const count = (name: string, a: unknown[]) => {
        const b = this.gpuBytes;
        if (name === "bufferData") { const v = view(a[1]); if (v) b.up += v.byteLength; else if (typeof a[1] === "number") b.alloc += a[1]; }
        else if (name === "bufferSubData") { const v = view(a[2]); if (v) b.up += typeof a[4] === "number" ? a[4] * ((v as Float32Array).BYTES_PER_ELEMENT ?? 1) : v.byteLength; }
        else if (name === "copyBufferSubData") b.copy += a[4] as number;
        else if (name === "texImage2D") { const v = view(a[8]); if (v) b.up += v.byteLength; }
        else if (name === "texSubImage2D") {
          const comps = a[6] === gl.RED ? 1 : a[6] === gl.RG ? 2 : 4;
          b.up += (a[4] as number) * (a[5] as number) * comps * (a[7] === gl.FLOAT ? 4 : 1);
        }
      };
      for (const name of ["bufferData", "bufferSubData", "copyBufferSubData", "texImage2D", "texSubImage2D", "deleteBuffer", "deleteTexture", "deleteVertexArray"]) {
        const f = g[name] as (...args: unknown[]) => unknown;
        g[name] = (...args: unknown[]) => {
          if (!this.quiet) this.uploads++;
          count(name, args);
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
    this.oneWayGhost.init(gl);
    this.lots.init(gl);
    this.floors.init(gl);
    this.bubbles.init(gl);
    this.buses.init(gl);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.hasStencil = (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0;
    // (Firefox with fingerprinting resistance says 2048 for both, under many a window's canvas)
    this.cacheMax = Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) ?? 0, gl.getParameter(gl.MAX_TEXTURE_SIZE) ?? 0);
    this.strokeU = uniforms(gl, this.strokeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_casing", "u_casing_rgb", "u_cap", "u_kind", "u_sel_line", "u_sel_n", "u_sel_set", "u_sel_edge", "u_hot", "u_only_sel", "u_bp_part", "u_lanes", "u_lane_mode", "u_pass_off","u_dim", "u_by_level", "u_by_speed", "u_by_profit", "u_profit", "u_profit_n", "u_level_colours", "u_track", "u_bad", "u_sel", "u_bp", "u_bp_mid", "u_bp_rail", "u_bp_off_mid", "u_bp_off_rail", "u_bp_off_fade", "u_line_colours", "u_full_pass", "u_full_px", "u_full_out", "u_full_dash", "u_full_rgb", "u_full_inside", "u_ghost", "u_ghost_layer", "u_ghost_rgb", "u_ghost_min", "u_hsr", "u_wire", "u_band", "u_band_traffic", "u_band_gap", "u_band_m", "u_mute", "u_level_hide", "u_mode_hide", "u_ahead_lines", "u_ahead_edges", "u_ahead_n", "u_ahead"]);
    this.stationU = uniforms(gl, this.stationProg, [...PROJECT_UNIFORMS, "u_viewport", "u_radius", "u_ring", "u_ink", "u_line", "u_boxes", "u_fixed", "u_pass_half", "u_pass_in", "u_pass", "u_layer", "u_pixel", "u_bad", "u_crowd_rgb", "u_tint", "u_mask", "u_stline", "u_lanes", "u_hot_st", "u_hot_n", "u_room", "u_bld", "u_one", "u_one_mid", "u_no_bp"]);
    this.bridgeProg = compile(gl, BRIDGE_VS, STATION_FS);
    this.bridgeU = uniforms(gl, this.bridgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_ring", "u_bridge", "u_pass", "u_layer", "u_ink", "u_bad", "u_crowd_rgb", "u_mask", "u_hot_st", "u_hot_n", "u_no_bp"]);
    // T-3192: the stations' u_stline, which a bridge never reads, on the stations' unit rather than one an integer texture may hold
    gl.useProgram(this.bridgeProg);
    gl.uniform1i(gl.getUniformLocation(this.bridgeProg, "u_stline"), 11);
    this.selProg = compile(gl, SEL_VS, SEL_FS);
    this.selU = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_shapes", "u_count", "u_dpr", "u_height", "u_ink", "u_outline", "u_glow"]);
    this.selTex = texture(this, this.selData, "rgba32f", SEL_MAX * SEL_STRIDE, []);
    this.trainU = uniforms(gl, this.trainProg, [...PROJECT_UNIFORMS, "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_phases", "u_meta", "u_samples", "u_bend_at", "u_bends", "u_lines", "u_back_base", "u_line_colours", "u_fill", "u_bad", "u_detail", "u_dpr", "u_cars", "u_loads", "u_stretch", "u_sel_line", "u_sel_n", "u_sel_set", "u_dim", "u_sel_trip", "u_pass", "u_ink", "u_band", "u_psi0", "u_lane_tex", "u_lane", "u_hollow", "u_bands", "u_bands_m", "u_passing", "u_keep", "u_level_hide", "u_slevel"]);
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
    this.loadTex =texture(this, new Float32Array(1), "r32f", TEX_W, []);
  }

  /** T-2420: each edge id's stretch fares over costs for the Profit view (`stretchMoney`'s
   * `ratio`, `NO_PROFIT` where none), or null for none at all. */
  setProfit(ratio: Float32Array | null) {
    const gl = this.gl;
    this.profitData = ratio;
    if (!gl) return;
    if (this.profitTex) gl.deleteTexture(this.profitTex);
    this.profitTex = ratio?.length ? texture(this, ratio, "r32f", TEX_W, []) : null;
    this.profitN = this.profitTex ? ratio!.length : 0;
  }

  /** `?patch=check` (T-1029): each network's strokes read back from the GPU after an edit and
   * compared with the arrays they were made from. */
  checkPatch = false;
  /** networks taken as changes to the one drawn (checks read it) */
  patched = 0;
  /** the network's stroke buffers on the GPU and their sizes (checks read them back) */
  gpuDump(): Record<string, [WebGLBuffer, number]> {
    const out: Record<string, [WebGLBuffer, number]> = {};
    for (const [name, v] of [["track", this.track], ["lines", this.lines]] as const)
      v?.attribs?.forEach(({ buf, bytes }, i) => (out[`${name}.${STROKE_ATTRIBS[i]}`] = [buf, Math.max(1, v.drawn ?? v.count) * bytes]));
    return out;
  }

  /** The demand views (T-084) for the network now drawn, or null for the plain drawing. The
   * arrays must be sized for `this.net` (map/demandLayers.ts builds them from it). */
  setLayers(l: NetworkLayers | null) {
    const n = this.net, t0 = performance.now();
    if (!n || !this.lines) return;
    this.layersSrc = l;
    const lines = this.lines;
    const ok = !!l && l.lineWidth.length === lines.count;
    const tb = ok && l.bands && l.bands.seg.length === lines.count * 2 ? l.bands : null;
    let wm: Float32Array | null = null;
    let ts = import.meta.env.DEV ? performance.now() : 0;
    // (T-4284: plain widths are known plain without a pass over them)
    if (ok && !(l.plain && !tb && this.keepGl)) {
      wm = new Float32Array(lines.count * 3);
      for (let i = 0; i < lines.count; i++) {
        wm[i * 3] = l.lineWidth[i];
        if (tb) (wm[i * 3 + 1] = tb.seg[i * 2]), (wm[i * 3 + 2] = tb.seg[i * 2 + 1]);
      }
    }
    // T-4262: widths all plain (the Line view's: every width 1, no band shift) are drawn as the plain widths, and
    // only the rows that differ from what the buffer holds are written
    if (wm && this.keepGl && isPlainWmul(wm)) wm = null;
    ts = stepSince("layers.set.wm", ts);
    const wmulChanged = lines.count > 0 && (!this.keepGl || !!wm || !lines.wmulPlain) && writeWmul(this, lines, wm);
    ts = stepSince("layers.set.write", ts);
    lines.wmulPlain = !wm;
    lines.wmulHeld = wm;
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
    ts = stepSince("layers.set.lane", ts);
    if (wmulChanged || !this.keepGl) this.perSegV++;
    // T-4284, T-4314: only the rows whose fill changed are written, when the last fill put is the one they changed from
    const fc = l?.fillChanged, named = fc && this.fillSrc === fc.from ? { keys: fc.keys, from: [], fromRowMap: fc.from } : null;
    this.fillSrc = l ? l.fill : null;
    this.putTrain("fill", n, named);
    stepSince("layers.set.fill", ts);
    this.fillStale = false;
    if (this.checkPatch && this.keepGl) this.checkTrains(["fill", "lane"], n);
    this.layersMs = performance.now() - t0;
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
  fullEdges: ReadonlySet<number> | null = null;
  fullTraffic = false;
  /** the crowded stretches as last written, line * 4096 + stretch: the busiest hour's load as a
   * share of the rated load and that hour (the line's local time) */
  fullKeys = new Map<number, CrowdedStretch>();
  fullDrawn = "";
  /** The crowding drawn on line `line`'s run 0 stretch `s` (T-2236); null where none is. */
  fullStretch(line: number, s: number): CrowdedStretch | null {
    return this.fullTraffic ? this.fullKeys.get(line * 4096 + s) ?? null : null;
  }
  writeFull() {
    const l = this.lines, n = this.net;
    const { keys, key } = this.fullInputs();
    const drawn = `${n ? 1 : 0}/${l?.count}/${key}`;
    this.fullKeys = keys;
    if (!l || !l.count || !n || drawn === this.fullDrawn) return;
    this.fullDrawn = drawn;
    const { out, any } = this.fullValues(n, keys, null, null);
    this.anyFull = any;
    writeSegs(this, l, l.fullBuf, out, 2);
    this.linesFull = out;
    this.fullKey = key;
    this.perSegV++;
  }
  /** T-4262: what the lines' traffic marks were last worked out from (`fullInputs`), so a network's runs copied from
   * the one before keep theirs */
  fullKey: string | null = null;
  /** the crowded stretches (line * 4096 + stretch) drawn, and the marks' inputs as text: the edges at capacity and those */
  private fullInputs(): { keys: Map<number, CrowdedStretch>; key: string } {
    const keys = new Map<number, CrowdedStretch>();
    const b = this.board, fs = this.fullTraffic ? b?.crowdStretches : undefined;
    for (let k = 0; b && fs && k < fs.length; k += 2) {
      const j = k / 2;
      keys.set(fs[k] * 4096 + fs[k + 1], { load: b.crowdLoads[j] ?? 0, hour: b.crowdHours[j] ?? 0 });
    }
    return { keys, key: `${[...(this.fullEdges ?? [])].join(",")}/${[...keys].map(([k, v]) => `${k}:${v.load}:${v.hour}`).join(",")}` };
  }
  /** The lines' `a_full` for `n`: worked out whole, or with `runs` (the patch's) those of the runs kept copied from
   * `old`, the last network's, made from the same inputs (`kept`); `any`: some segment is marked. */
  private fullValues(n: NetworkBuffers, keys: Map<number, CrowdedStretch>, runs: Int32Array | null, old: Float32Array | null): { out: Float32Array; kept: boolean; any: boolean } {
    const L = n.lines, f = this.fullEdges, stretch = L.stretch;
    // the shader's a_full: -1 track at capacity, else the stretch's load over the rated load
    const marked = !!f?.size || keys.size > 0;
    const marks = marked ? (i: number) => {
      if (f?.size && f.has(L.edge[i])) return -1;
      const s = keys.size && stretch ? keys.get(L.colour[i] * 4096 + stretch[i]) : undefined;
      return s ? s.load : 0;
    } : undefined;
    let out: Float32Array;
    if (runs && old) {
      out = new Float32Array(Math.max(1, L.count) * 2);
      for (let k = 0; k < runs.length; k += 3) {
        const s = runs[k], c = runs[k + 1], o = runs[k + 2];
        if (o >= 0) out.set(old.subarray(o * 2, (o + c) * 2), s * 2);
        else out.set(fullAttr(L, c, marks, s).subarray(0, c * 2), s * 2);
      }
    } else out = fullAttr(L, L.count, marks);
    let any = false;
    for (let i = 0; marked && !any && i < L.count; i++) any = out[i * 2] !== 0;
    return { out, kept: !!(runs && old), any };
  }
  anyFull = false;

  /** Replace everything drawn but the trips (an edit). `touched`: its state's (T-4280: the lines whose trains' tables
   * changed, as the worker names them) */
  setNetwork(n: NetworkBuffers, touched?: EditTouched | null) {
    const gl = this.gl, t0 = performance.now();
    this.dropPlaced(); // T-1407: the network drawn now has the segment its stand-in showed
    // (T-4145: what is drawn ahead was cleared by game/pending.ts just before, `setAhead`)
    // T-1028: an edit's network posted as changes to the one drawn keeps the strokes' unchanged runs
    // on the GPU and the textures whose data is the same; anything else is uploaded whole
    const was = this.net, p = n.patch;
    const patch = p && was && was !== n && was.serial === p.base && this.track?.count === p.oldTrack && this.lines?.count === p.oldLines ? p : null;
    if (patch) this.patched++;
    for (const r of this.res) r.del();
    this.res = [];
    this.probeVao = null;
    this.net = n;
    this.guard.setNetwork(n);
    this.marks.setNetwork(n.marks);
    this.oneWay.setNetwork(n.oneWay);
    this.oneWayTrack.setNetwork(n.oneWayTrack);
    this.chevronsAhead = false;
    this.lots.setNetwork(n.lots);
    this.floors.setNetwork(n.buildings);
    const oldTrack = this.track, oldLines = this.lines, wmulWas = this.linesWmul;
    // the lines' traffic marks (T-819); T-4262: on the runs an edit's network copies, the last network's while what
    // they are made from is as it was
    const { keys, key } = this.fullInputs();
    const oldFull = this.linesFull;
    const fullRuns = this.keepGl && patch && oldLines && oldFull && this.fullKey === key && oldFull.length === Math.max(1, oldLines.count) * 2 ? patch.lines : null;
    const full = this.fullValues(n, keys, fullRuns, oldFull);
    // T-4262: each set's segments in slots in its own buffers, an edit's written where it changed them (strokeSlots.ts)
    this.track = strokes(this, n.track, [], patch && oldTrack ? { old: oldTrack, runs: patch.track } : null, { room: true });
    this.lines = strokes(this, n.lines, [], patch && oldLines ? { old: oldLines, runs: patch.lines } : null, { full: full.out, fullKept: full.kept, room: true });
    // a set keeps its simplified copies and culling boxes while none of its segments changed, and the culling boxes
    // of the chunks before its first change
    const linesSame = this.lines === oldLines && !!patch && unchangedRuns(patch.lines, this.lines.count);
    // (T-4274: their simplified copies carried through the edit once the marks and widths are set, below)
    const lod: [StrokeVao, StrokeVao, Int32Array | undefined, Strokes, boolean][] = [];
    for (const [old, now, runs, src, isLines] of [[oldTrack, this.track, patch?.track, n.track, false], [oldLines, this.lines, patch?.lines, n.lines, true]] as const) {
      if (!old) continue;
      if (old !== now || !runs || !unchangedRuns(runs, now.count)) {
        const boxes = this.lodOf.get(old)?.boxes;
        if (boxes && runs && this.keepGl && runs[0] === 0 && runs[2] === 0) now.boxesFrom = { boxes, upTo: runs[1] };
        lod.push([old, now, runs, src, isLines]);
      }
      if (old !== now) for (const r of old.res ?? []) r.del();
    }
    // the demand widths the runs kept had, the new segments' plain, until `setLayers` (T-4262: `wmulHeld`)
    if (this.lines.count && !this.lines.wmulPlain && !this.lines.wmulHeld) {
      writeSegs(this, this.lines, this.lines.wmulBuf, plainWmul(this.lines.count), 3);
      this.lines.wmulPlain = true;
    }
    this.linesWmul = this.lines.wmulPlain ? null : this.lines.wmulHeld ?? null;
    this.linesFull = n.lines.count ? full.out : null;
    this.anyFull = full.any;
    this.fullKeys = keys;
    this.fullKey = key;
    this.fullDrawn = n.lines.count ? `1/${n.lines.count}/${key}` : "";
    const carried: StrokeBand[] = [];
    this.lodCarried = null;
    for (const [old, now, runs, src, isLines] of lod) {
      const c = this.keepGl && this.strokeLod && runs ? carryLod(this, old, now, src, runs, isLines) : null;
      if (c) carried.push(...c);
      else dropLodOf(this, old);
    }
    // (the simplified copies take the widths and marks again when either changed)
    if (!linesSame || !full.kept || this.linesWmul !== wmulWas) this.perSegV++;
    // a copy carried with the marks and plain widths it had, which were written, holds this network's
    for (const b of carried) if (b.v === -2) b.v = full.kept && !this.linesWmul && !wmulWas ? this.perSegV : -1;
    if (patch && this.checkPatch) {
      // (T-4274: a lines copy carried with its widths and marks current; the others are written whole before drawn)
      for (const b of carried) if (b.fu && b.v === this.perSegV) checkFull(this, b.vao, b.strokes, b.fu, b.wm ?? plainWmul(b.strokes.count));
      checkStrokes(this, "track", this.track, n.track);
      checkStrokes(this, "lines", this.lines, n.lines);
      checkFull(this, this.track, n.track);
      checkFull(this, this.lines, n.lines, full.out, this.linesWmul ?? plainWmul(this.lines.count));
      if (full.kept && !sameWords(full.out, this.fullValues(n, keys, null, null).out)) console.error("T-4262 check: the lines' traffic marks kept by runs differ from whole ones");
    }
    // (the picture changed, whatever was uploaded)
    this.uploads++;
    this.turnback = n.turnbacks?.count ? strokes(this, n.turnbacks) : null;
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
    // T-4274: the trains' tables in slots of their own (trainSlots.ts), written only where a line's moved or changed
    const tt = performance.now();
    const keep = this.keepGl, was0 = this.sampleSlots, bendsWere = this.bendSlots, base = backBaseOf(n.lineTable, n.meta);
    const rows = sampleRows(n.lineTable, base);
    this.sampleSlots = planSlots(keep ? was0 : null, rows, TEX_W);
    if (!(keep && this.bendSlots && this.sampleSlots === was0 && same.has("sampleBend") && same.has("bends")))
      this.bendSlots = planSlots(keep ? this.bendSlots : null, n.sampleBend ? bendRows(n.sampleBend, rows) : [], TEX_W, BEND_LIMIT);
    const phasesWere = this.phaseSlots;
    if (!(keep && phasesWere && same.has("meta"))) this.phaseSlots = this.planPhases(n.meta);
    // T-4280: each table written where the worker names a line changed since the network before (or its slot moved)
    const names = patch && was && touched && !touched.all ? trainNames(was, n, touched, patch, bendsWere, this.bendSlots!) : null;
    this.trainNamed = names ? { paths: touched!.paths.length, phases: touched!.phases.length, stretches: patch!.stretches.length, lanes: patch!.lanes.length } : null;
    for (const name of ["samples", "bendAt", "bends", "phases", "stretch"] as const) this.putTrain(name, n, names?.[name]);
    const sw = this.slotTex.samples!.wrote, tw = this.slotTex.stretch!.wrote, stale = this.fillStale ? this.trainsWrote : null;
    // (two networks before a frame, as two staged states due at once: the lines either wrote)
    this.trainsWrote = sw === "all" || tw === "all" || stale === "all" ? "all" : new Set([...(stale ?? []), ...sw, ...tw]);
    put("lines", null, () => [slottedLineTable(n.lineTable.length ? n.lineTable : new Float32Array(4), base, this.sampleSlots!), "rgba32f", LINES_W]);
    if (!(this.phaseSlots === phasesWere && same.has("meta") && this.tex.meta)) this.putTexture("meta", slottedMeta(n.meta.length ? n.meta : new Float32Array(4), this.phaseSlots!), "rgba32f", TEX_W);
    // the Traffic view's bands were the last network's
    this.trafficBands = null;
    this.hadSamplePass = !!n.samplePass;
    this.passing = false;
    for (let i = 0; i < n.stationCount && !this.passing; i++) this.passing = (n.stations[i * 3 + 2] & ST_PASSING) !== 0;
    this.putLane(n, names?.lane);
    this.trainMs = performance.now() - tt;
    put("colours", "lineColours", () => [this.shownColours(n), "rgba8", LINES_W]);
    this.hiddenFor = null;
    // the demand views' fill (T-084) is `setLayers`'s, which follows each network; T-4262: until it comes, the last
    // network's, with the lines written above blank from the next frame (`freshFill`)
    this.fillStale = true;
    const cars = n.lineCars ?? new Float32Array(Math.max(1, n.lineTable.length / 4)).fill(8);
    this.maxCars = Math.max(1, ...cars);
    put("cars", "lineCars", () => [cars, "r32f", LINES_W]);
    this.crowds = new PlatformCrowds(n.stationCount);

    this.stationVao = gl.createVertexArray()!;
    const sv = this.stationVao;
    this.res.push({ del: () => gl.deleteVertexArray(sv) });
    gl.bindVertexArray(this.stationVao);
    attrib(this, this.stationProg, "a_st", buffer(this, n.stations.length ? n.stations : new Float32Array(3)), 3, gl.FLOAT, 1);
    attrib(this, this.stationProg, "a_axis", buffer(this, n.stationAxis?.length ? n.stationAxis : new Float32Array(Math.max(1,n.stationCount)*2)),2,gl.FLOAT,1);
    this.stationShapeBuf = buffer(this, n.stationShape?.length ? n.stationShape : new Float32Array(Math.max(1, n.stationCount) * 4));
    // each platform's track line (T-254), or the arc of its shape where the buffers have none
    const P2 = 2 * PLATFORM_POINTS;
    if (!n.stationLine || n.stationLine.length < n.stationCount * P2) {
      const line = new Float32Array(n.stationCount * P2);
      for (let i = 0; i < n.stationCount; i++)
        line.set(platformPoints(platformShapeAt(n, i, false)), i * P2);
      n.stationLine = line;
    }
    // T-4314: keep `reachOf` chained across edits once asked for, so it stays patched even between hovers
    if (this.reachOf) {
      if (n.patch?.stations && this.reachOf.net.serial === n.patch.base) timed("station.reach", () => (this.reachOf = stationReachStep(n, this.reachOf, this.checkPatch || checkPatches())));
      else this.reachOf = null;
    }
    put("stationLine", "stationLine", () => [n.stationLine!.length ? n.stationLine! : new Float32Array(2), "rg32f", TEX_W]);
    attrib(this, this.stationProg, "a_shape", this.stationShapeBuf, 4, gl.FLOAT, 1);
    this.stationSelBuf = buffer(this, new Float32Array(Math.max(1, n.stationCount)));
    attrib(this, this.stationProg, "a_sel", this.stationSelBuf, 1, gl.FLOAT, 1);
    this.groupSrc = { groups: null, net: null };
    this.crowdBuf = buffer(this, this.crowds.crowds);
    this.dotCrowdBuf = buffer(this, this.crowds.dotCrowds);
    attrib(this, this.stationProg, "a_dcrowd", this.dotCrowdBuf, 2, gl.FLOAT, 1);
    attrib(this, this.stationProg, "a_crowd", this.crowdBuf, 4, gl.FLOAT, 1);
    this.selectedAlso = new Set();
    this.selectedAlsoKey = "";
    this.selDrawnKey = "";
    // the player's links as bridges (T-217)
    const links = n.stationLinkIdx?.length ?? 0;
    this.bridgeVao = gl.createVertexArray()!;
    const bv = this.bridgeVao;
    this.res.push({ del: () => gl.deleteVertexArray(bv) });
    gl.bindVertexArray(bv);
    attrib(this, this.bridgeProg, "a_ends", buffer(this, links ? n.stationLinks! : new Float32Array(4)), 4, gl.FLOAT, 1);
    this.bridgeFlagBuf = buffer(this, new Float32Array(Math.max(1, links / 2) * 2));
    attrib(this, this.bridgeProg, "a_bflag", this.bridgeFlagBuf, 2, gl.FLOAT, 1);
    // zoomed out the connectors between dots (T-313) are built by syncGroups

    this.trainVao = gl.createVertexArray()!;
    const tv = this.trainVao;
    this.res.push({ del: () => gl.deleteVertexArray(tv) });
    gl.bindVertexArray(this.trainVao);
    attrib(this, this.trainProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 1);
    attrib(this, this.trainProg, "a_end", this.endBuf!, 1, gl.FLOAT, 1);
    attrib(this, this.trainProg, "a_load", this.loadBuf!, 2, gl.FLOAT, 1);
    attrib(this, this.trainProg, "a_stand", this.standBuf!, 4, gl.FLOAT, 1);
    attrib(this, this.trainProg, "a_lag", this.lagBuf!, 1, gl.FLOAT, 1);
    attrib(this, this.trainProg, "a_bay", this.bayBuf!, 4, gl.FLOAT, 1);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    if (!aheadEmpty(this.ahead) || this.aheadDrawn) this.buildAhead();
    if (this.checkPatch && this.keepGl) {
      this.texturesRead = checkTextures(this);
      this.checkTrains(["samples", "bendAt", "bends", "phases", "stretch", "lane"], n);
      if (!phasesFromFirst(n.meta, phaseRows(n.meta))) console.error("T-4274 check: a line's phases do not start at its first profile's first phase");
    }
    this.netMs = performance.now() - t0;
  }
  /** `?patch=check`: how many kept textures the last check read back (T-4262) */
  texturesRead = 0;

  /** T-4145: what is drawn ahead of the network (game/pending.ts makes it, map/lineAhead.ts) */
  ahead: AheadModel = noAhead();
  /** the line strokes and track strokes drawn ahead, over the network's */
  aheadVao: StrokeVao | null = null;
  aheadTrackVao: StrokeVao | null = null;
  aheadRes: { del: () => void }[] = [];
  /** per line id and per edge id (1: its network strokes not drawn), and how many ids each covers, 0 none */
  aheadLinesTex: WebGLTexture | null = null;
  aheadEdgesTex: WebGLTexture | null = null;
  aheadN: [number, number] = [0, 0];
  /** station boxes drawn ahead: the station shader's buffers for them, and their centre lines' texture */
  standVao: WebGLVertexArrayObject | null = null;
  standLineTex: WebGLTexture | null = null;
  standCount = 0;
  /** the last `buildAhead` drew something (textures, strokes, boxes or colours), which a network set must take away
   * even when what is drawn ahead is empty by then */
  aheadDrawn = false;
  /** T-4148: the one-way track chevrons drawn are an edit's (`aheadChevrons`), not the network's */
  private chevronsAhead = false;

  /** T-4145: draw `m` ahead of the network from now (until the next call). */
  setAhead(m: AheadModel, build = true) {
    this.ahead = m;
    if (build) this.buildAhead();
  }
  /** T-4114: the version of the state the network drawn came in (main.ts), so a guess at a click knows whether it
   * shows the stops the world holds */
  netVersion = -1;

  private buildAhead() {
    const gl = this.gl;
    for (const r of this.aheadRes) r.del();
    this.aheadRes = [];
    this.aheadVao = this.aheadTrackVao = this.aheadLinesTex = this.aheadEdgesTex = this.standVao = this.standLineTex = null;
    this.aheadN = [0, 0];
    this.standCount = 0;
    const n = this.net, m = this.ahead;
    this.aheadDrawn = !!n && !aheadEmpty(m);
    if (n && !aheadEmpty(m)) {
      const p = aheadParts(n, m);
      if (p.lines) this.aheadVao = strokes(this, p.lines, this.aheadRes);
      if (p.track) this.aheadTrackVao = strokes(this, p.track, this.aheadRes);
      // lines drawn ahead: their own strokes not drawn; edges hidden or drawn with a new look: neither track nor lines on them
      const lines = [...[...m.lines].filter(([, a]) => a.edges).map(([l]) => l), ...m.hideLines];
      const edges = [...m.hideEdges, ...m.edges.keys()];
      const flags = (ids: number[]): [Float32Array, number] => {
        let top = 0;
        for (const id of ids) top = Math.max(top, id + 1);
        const out = new Float32Array(Math.max(1, top));
        for (const id of ids) if (id >= 0) out[id] = 1;
        return [out, top];
      };
      if (lines.length) { const [d, top] = flags(lines); this.aheadLinesTex = texture(this, d, "r32f", LINES_W, this.aheadRes); this.aheadN[0] = top; }
      if (edges.length) { const [d, top] = flags(edges); this.aheadEdgesTex = texture(this, d, "r32f", TEX_W, this.aheadRes); this.aheadN[1] = top; }
      if ((m.stations.length || m.rawStations.length) && this.stationProg) {
        const s = joinStations([...(m.stations.length ? [standInStations(m.stations)] : []), ...m.rawStations]), P2 = 2 * PLATFORM_POINTS;
        this.standVao = gl.createVertexArray()!;
        const v = this.standVao;
        this.aheadRes.push({ del: () => gl.deleteVertexArray(v) });
        gl.bindVertexArray(v);
        attrib(this, this.stationProg, "a_st", buffer(this, s.st, this.aheadRes), 3, gl.FLOAT, 1);
        attrib(this, this.stationProg, "a_axis", buffer(this, s.axis, this.aheadRes), 2, gl.FLOAT, 1);
        attrib(this, this.stationProg, "a_shape", buffer(this, s.shape, this.aheadRes), 4, gl.FLOAT, 1);
        attrib(this, this.stationProg, "a_sel", buffer(this, new Float32Array(Math.max(1, s.count)), this.aheadRes), 1, gl.FLOAT, 1);
        attrib(this, this.stationProg, "a_dcrowd", buffer(this, new Float32Array(Math.max(1, s.count) * 2), this.aheadRes), 2, gl.FLOAT, 1);
        attrib(this, this.stationProg, "a_crowd", buffer(this, new Float32Array(Math.max(1, s.count) * 4), this.aheadRes), 4, gl.FLOAT, 1);
        gl.bindVertexArray(null);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        this.standLineTex = texture(this, s.line.length >= P2 ? s.line : new Float32Array(2), "rg32f", TEX_W, this.aheadRes);
        this.standCount = s.count;
      }
    }
    const chev = n && !aheadEmpty(m) ? aheadChevrons(n, m.edges) : null;
    if (chev) { this.oneWayTrack.setNetwork(chev); this.chevronsAhead = true; }
    else if (this.chevronsAhead) { this.oneWayTrack.setNetwork(n?.oneWayTrack); this.chevronsAhead = false; }
    if (n && this.tex.colours) this.putTexture("colours", this.shownColours(n), "rgba8", LINES_W);
    // stations hidden ahead, with those on a hidden level (the station flags are rebuilt from it)
    if (n && this.hiddenFor === n) this.hideStations(n);
    this.uploads++;
  }

  /** the stations on a hidden level or of hidden kinds only (T-4400), a blueprint the player hid, or hidden ahead of
   * the network (T-4145) */
  private hideStations(n: NetworkBuffers) {
    this.stationHid = new Uint8Array(n.stationCount);
    if (vis.levels || vis.modes || vis.blueprints) {
      for (let i = 0; i < n.stationCount; i++) if (stationOff(n.stationIds[i], (n.stations[i * 3 + 2] & 2) === 0)) this.stationHid[i] = 1;
    }
    for (const id of this.ahead.hideStations) {
      const i = stationIndexOf(n, String(id));
      if (i >= 0) this.stationHid[i] = 1;
    }
    this.lots.setShown(vis.levels || vis.modes || vis.blueprints || this.ahead.hideStations.size ? (k) => this.lotHidden(k) : null);
    this.selDrawnKey = "";
  }

  /** Only the service changed (a train count, T-229): the run profiles' phase tables and meta are
   * replaced and every other buffer kept, rather than uploading the whole network again. */
  setService(phases: Float32Array, meta: Float32Array, touched?: EditTouched | null) {
    const n = this.net;
    if (!n) return;
    // (T-4280: the lines whose tables the worker names as changed since the last it posted)
    const named = touched && !touched.all ? { keys: namedKeys(touched.phases, false), from: [n.phases] } : null;
    n.phases = phases;
    n.meta = meta;
    this.phaseSlots = this.planPhases(meta);
    this.putTrain("phases", n, named);
    if (this.checkPatch && this.keepGl) this.checkTrains(["phases"], n);
    this.putTexture("meta", slottedMeta(meta.length ? meta : new Float32Array(4), this.phaseSlots), "rgba32f", TEX_W);
  }

  /** T-4274: the trains' tables' layouts (trainSlots.ts): per line's run its samples' slot and its bend points', per line
   * its phases'; and the tables written through them */
  sampleSlots: SlotLayout | null = null;
  bendSlots: SlotLayout | null = null;
  phaseSlots: SlotLayout | null = null;
  slotTex: Partial<Record<SlotName, SlotTex>> = {};
  /** the keys the networks' samples and stretches were written for since the fill was last put or blanked, or "all" (`freshFill`) */
  trainsWrote: Set<number> | "all" = "all";
  /** T-4274: bytes sent to each network texture since the page opened (`gpuBytes.up`, by texture) */
  texSent: Record<string, number> = {};
  /** T-4280: per train table, writes that compared the values (no list named the lines changed, or the table was last
   * written from other arrays than the list's), since the page opened: an edit's network adds none */
  trainCompares: Record<string, number> = {};
  /** T-4280: how many lines the last network's lists named, null when it had none */
  trainNamed: { paths: number; phases: number; stretches: number; lanes: number } | null = null;
  /** T-4280: ms of the last `setNetwork`, of its trains' tables, and of the last `setLayers` */
  netMs = 0;
  trainMs = 0;
  layersMs = 0;
  /** the demand views' fill last given (`setLayers`), null: blank */
  private fillSrc: ReadonlyMap<number, Float32Array> | null = null;
  private stretchFor: { n: NetworkBuffers; v: Float32Array } | null = null;

  private planPhases(meta: Float32Array): SlotLayout {
    return planSlots(this.keepGl ? this.phaseSlots : null, phaseRows(meta), TEX_W, Infinity, PHASE_ROOM);
  }
  /** T-4274: train table `name`'s kind, layout and sources for network `n` */
  private trainSrc(name: SlotName, n: NetworkBuffers): [TexKind, SlotLayout, SlotSrc] | null {
    const S = this.sampleSlots, B = this.bendSlots, P = this.phaseSlots;
    if (!S || !B || !P) return null;
    switch (name) {
      case "samples": return ["rg32f", S, { srcs: [n.samples], per: [2] }];
      case "bendAt": return ["r32f", S, { srcs: [n.sampleBend], per: [1], shift: (k) => (B.at.get(k) ?? 0) - (B.first.get(k) ?? 0) }];
      case "bends": return ["rg32f", B, { srcs: [n.bends], per: [2] }];
      case "phases": return ["rgba32f", P, { srcs: [n.phases], per: [4] }];
      case "stretch": return ["r32f", S, { srcs: [this.stretchOf(n)], per: [1] }];
      case "lane": {
        const lane = n.sampleLane?.length ? n.sampleLane : undefined, band = this.trafficBands?.sample ?? n.sampleBand, pass = n.samplePass;
        return ["rgba32f", S, { srcs: [lane, band, pass], per: [1, 2, 1], pack: (out, f, c) => packLane(out, f, c, lane, band, pass) }];
      }
      case "fill": return ["rg32f", S, { srcs: [], per: [2], rowMap: this.fillSrc ?? EMPTY_FILL_ROWS, fillVal: -1 }];
      case "slevel": return ["r32f", S, { srcs: [this.sampleLv ?? undefined], per: [1] }];
    }
  }
  /** T-4274: train table `name` written for network `n` through its layout */
  private putTrain(name: SlotName, n: NetworkBuffers, named?: SlotNames | null) {
    const t = this.trainSrc(name, n);
    if (!t) return;
    const u0 = this.gpuBytes.up;
    const k = putSlots(this, this.slotTex[name] ?? null, t[0], TEX_W, t[1], t[2], named);
    if (k.compared) this.trainCompares[name] = (this.trainCompares[name] ?? 0) + 1;
    this.slotTex[name] = k;
    this.tex[name] = k.tex;
    this.texSent[name] = (this.texSent[name] ?? 0) + this.gpuBytes.up - u0;
  }
  /** `?patch=check` (T-4274): train tables `names` read back, each line's slot against the network's values */
  private checkTrains(names: SlotName[], n: NetworkBuffers) {
    const bad: string[] = [];
    for (const name of names) {
      const k = this.slotTex[name], t = this.trainSrc(name, n);
      if (!k || !t) continue;
      const keys = checkSlots(this, k, TEX_W, t[1], t[2]);
      if (keys.length) bad.push(`${name} (${keys.length} slots, keys ${keys.slice(0, 4).join(" ")})`);
    }
    if (bad.length) console.error(`T-4274 check: the trains' tables in slots differ from the network's: ${bad.join(", ")}`);
  }
  /** each train sample's stop-to-stop stretch (run 0's numbering), where a trip's loads are looked up (T-164) */
  private stretchOf(n: NetworkBuffers): Float32Array {
    if (n.sampleStretch) return n.sampleStretch;
    if (this.stretchFor?.n === n) return this.stretchFor.v;
    const nSamples = Math.max(1, n.samples.length / 2);
    const stretch = new Float32Array(nSamples).fill(-1);
    if (n.sampleStroke && n.lines.stretch)
      for (let k = 0; k < nSamples; k++) {
        const i = n.sampleStroke[k];
        if (i >= 0) stretch[k] = n.lines.stretch[i];
      }
    this.stretchFor = { n, v: stretch };
    return stretch;
  }
  /** T-3420: per sample its height in levels, for the cars on a hidden level (null: none hidden) */
  putLevels(lv: Float32Array | null) {
    this.sampleLv = lv;
    if (lv && this.net) this.putTrain("slevel", this.net);
    else if (this.tex.slevel) this.dropTexture("slevel");
  }

  /** each network texture's GL object, freed when it is replaced (T-1028: kept across edits while its
   * data is the same) */
  texRes: Partial<Record<TexName, { del: () => void }[]>> = {};
  /** per sample: how far apart its tracks are (T-658) and its line's band's shift as drawn (T-1799),
   * line widths and metres (T-2132) */
  putLane(n: NetworkBuffers, named?: SlotNames | null) {
    this.putTrain("lane", n, named);
  }
  /** T-4262: the demand views' fill is the last network's until `setLayers` puts this one's (`freshFill`) */
  fillStale = false;
  private blank: Float32Array | null = null;
  /** per train sample, no fill known (-1), for `m` samples: one array while `m` stays */
  blankFill(m: number): Float32Array {
    if (this.blank?.length !== m * 2) this.blank = new Float32Array(m * 2).fill(-1);
    return this.blank;
  }
  /** Before a frame draws: a network's fill blank while `setLayers` has not put one (T-4262). */
  freshFill() {
    const S = this.sampleSlots;
    if (!this.fillStale || !this.net || !S) return;
    this.fillStale = false;
    // T-4274: blank where the last network's fill is not this one's (a line written or moved since the fill was), else blank whole
    const k = this.slotTex.fill, w = this.trainsWrote;
    if (!k || !this.keepGl || w === "all" || k.layout.base !== S.base || k.rows < Math.ceil(S.cap / TEX_W)) {
      this.fillSrc = null;
      this.putTrain("fill", this.net);
      return;
    }
    const keys: number[] = [];
    for (const [key, a] of S.at) if (w.has(key) || k.layout.at.get(key) !== a || k.layout.count.get(key) !== S.count.get(key)) keys.push(key);
    if (keys.length) blankSlots(this, k, TEX_W, S, keys, -1);
  }

  /** T-4262: the network textures kept from one network to the next (`keepTexture`) */
  keptTex: Partial<Record<TexName, KeptTex>> = {};
  /** Network texture `name` from `data`: the one kept written where it differs (`rows`: start and count pairs of the
   * rows that do, when known), or one made anew. */
  putTexture(name: TexName, data: ArrayBufferView, kind: TexKind, width: number, rows?: readonly number[]) {
    for (const r of this.texRes[name] ?? []) r.del();
    this.texRes[name] = [];
    const k = this.keptTex[name] ?? null;
    if (!this.keepGl) {
      if (k) this.gl.deleteTexture(k.tex);
      delete this.keptTex[name];
      const list: { del: () => void }[] = [];
      this.tex[name] = texture(this, data, kind, width, list);
      this.texRes[name] = list;
      return;
    }
    const u0 = this.gpuBytes.up;
    const next = keepTexture(this, k, data, kind, width, rows);
    this.texSent[name] = (this.texSent[name] ?? 0) + this.gpuBytes.up - u0;
    if (k && next !== k) this.gl.deleteTexture(k.tex);
    this.keptTex[name] = next;
    this.tex[name] = next.tex;
  }
  dropTexture(name: TexName) {
    for (const r of this.texRes[name] ?? []) r.del();
    this.texRes[name] = [];
    const k = this.keptTex[name];
    if (k) this.gl.deleteTexture(k.tex);
    delete this.keptTex[name];
    const s = this.slotTex[name as SlotName];
    if (s) this.gl.deleteTexture(s.tex);
    delete this.slotTex[name as SlotName];
    this.tex[name] = null;
  }

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    this.previewSrc = s;
    this.uncached(() => {
      for (const r of this.previewRes) r.del();
      this.previewRes = [];
      this.preview = s && s.count ? strokes(this, s, this.previewRes) : null;
      this.marks.setPreview(s?.marks);
      this.oneWayGhost.setNetwork(s?.chevrons);
    });
  }

  /** T-4136: the ghost and the placed stand-ins are drawn over the strokes' cache (rendererDraw's `ghostPass`), so their
   * GL work is not counted as a change the cache must be drawn again for */
  private uncached(f: () => void) {
    const q = this.quiet;
    this.quiet = true;
    try { f(); } finally { this.quiet = q; }
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
    this.uncached(() => {
      this.marks.setPreview(undefined);
      this.oneWayGhost.setNetwork(undefined);
    });
  }
  /** T-2404: a placed segment with no ghost of it (the pointer moved on before the click's own was
   * answered), drawn as a stand-in from its edges */
  addPlaced(s: Strokes) {
    if (s.count) this.uncached(() => this.placed.push(strokes(this, s, this.placedRes)));
  }
  /** T-3420: the network and the `vis.v` the line colours' alphas, the station flags and the picks' skips were set for */
  hiddenFor: NetworkBuffers | null = null;
  hiddenV = -1;
  /** per station: hidden (`hideStations`) */
  stationHid = new Uint8Array(0);
  /** T-3420: the line colours with a hidden line's alpha 0 (the stroke and train shaders read it) */
  shownColours(n: NetworkBuffers): Uint8Array {
    // T-4114: a line deleted and not yet out of the network too; T-4145: a line's colour ahead of the network
    const gone = [...this.ahead.lines].filter(([, a]) => !a.edges).map(([l]) => l);
    if (!vis.lines.size && !gone.length && !this.ahead.colours.size) return n.lineColours;
    // a new line (a copy) has an id past the network's colours, or a slot it left unused (alpha 0, not drawn)
    let top = n.lineColours.length;
    for (const l of this.ahead.colours.keys()) top = Math.max(top, l * 4 + 4);
    const c = new Uint8Array(top);
    c.set(n.lineColours);
    for (const [l, rgb] of this.ahead.colours) {
      c.set(rgb, l * 4);
      if (!c[l * 4 + 3]) c[l * 4 + 3] = 255;
    }
    for (const l of [...vis.lines, ...gone]) if (l * 4 + 3 < c.length) c[l * 4 + 3] = 0;
    return c;
  }
  /** Whether station `i` is hidden (T-3420, T-4407). */
  stationHidden(i: number): boolean {
    const n = this.net;
    if (n && (this.hiddenFor !== n || this.hiddenV !== vis.v)) {
      return stationOff(n.stationIds[i], (n.stations[i * 3 + 2] & 2) === 0) || this.ahead.hideStations.has(Number(n.stationIds[i]));
    }
    return this.stationHid[i] === 1;
  }
  /** Whether Park and Ride lot `k` is hidden (T-4407). */
  lotHidden(k: number): boolean {
    const n = this.net, l = this.lots.lots, o = k * LOT_STRIDE;
    if (!n || o + 8 >= l.length) return false;
    const s = l[o + 8];
    return this.stationHidden(s) || lotOff(n.stationIds[s], (l[o + 7] & LOT_BLUEPRINT) === 0);
  }
  sampleLv: Float32Array | null = null;
  /** T-3560: whether a train (as `tripAt` places it) is on a hidden level, not drawn or pickable */
  tripOnOffLevel(at: { geo: number; gr: number; s: number }): boolean {
    const n = this.net;
    if (!vis.levels || !this.sampleLv || !n) return false;
    const d = at.gr === 0 ? at.s : n.lineTable[at.geo * 4 + 2] - at.s;
    return levelShown(sampleValue(n, this.sampleLv, at.geo, d)) < 0.5;
  }
  /** T-3420: per train sample, its height in levels (its line stroke's start), for the cars on a hidden level */
  sampleLevels(n: NetworkBuffers): Float32Array {
    const m = Math.max(1, n.samples.length / 2), out = new Float32Array(m), s = n.sampleStroke;
    let last = 0;
    for (let k = 0; k < m; k++) {
      const i = s ? s[k] : -1;
      if (i >= 0 && i < n.lines.count) last = Math.floor(n.lines.level[i] / 256) / 32 - 4;
      out[k] = last;
    }
    return out;
  }
  /** T-3420: what the player hid, put into the line colours, the station flags, the picks and the trains' heights */
  applyHidden(n: NetworkBuffers) {
    this.hiddenFor = n;
    this.hiddenV = vis.v;
    if (this.tex.colours) this.putTexture("colours", this.shownColours(n), "rgba8", LINES_W);
    this.hideStations(n);
    this.putLevels(vis.levels ? this.sampleLevels(n) : null);
    const L = n.lines, T = n.track;
    if (vis.lines.size || vis.levels || vis.modes || vis.blueprints) {
      const bp = vis.blueprints, modes = vis.modes;
      pickSkip.set(L.seg, (i) => vis.lines.has(L.colour[i]) || codeHidden(L.level[i]) || (bp && (L.flags[i] & F_BLUEPRINT) !== 0));
      if (vis.levels || modes || bp) pickSkip.set(T.seg, (i) => codeHidden(T.level[i]) || (modes & strokeMode(T.flags[i])) !== 0 || (bp && (T.flags[i] & F_BLUEPRINT) !== 0));
      else pickSkip.delete(T.seg);
    } else {
      pickSkip.delete(L.seg);
      pickSkip.delete(T.seg);
    }
    this.hideMarks(n);
  }
  /** T-4400: the junction, crossing and ramp marks that lie on track of hidden kinds only, not drawn or picked */
  private hideMarks(n: NetworkBuffers) {
    const mk = n.marks;
    if (!mk) return;
    if (!vis.modes) {
      pickSkip.delete(mk);
      this.marks.setShown(mk);
      return;
    }
    const off = new Set<number>(), R = MARK_ON_TRACK;
    for (let i = 0; i < mk.length; i += MARK_STRIDE) {
      const x = mk[i], y = mk[i + 1];
      let bits = 0;
      for (const { edge, pts, ks } of edgesNear(n, x, y, R) ?? []) {
        for (const k of ks) {
          const ax = pts[k], ay = pts[k + 1], dx = pts[k + 3] - ax, dy = pts[k + 4] - ay, l2 = dx * dx + dy * dy;
          const u = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
          if ((ax + u * dx - x) ** 2 + (ay + u * dy - y) ** 2 <= R * R) { bits |= vis.edgeModes(edge); break; }
        }
      }
      if (modesHidden(bits)) off.add(i);
    }
    if (!off.size) {
      pickSkip.delete(mk);
      this.marks.setShown(mk);
      return;
    }
    pickSkip.set(mk, (i) => off.has(i));
    const kept: number[] = [];
    for (let i = 0; i < mk.length; i += MARK_STRIDE) if (!off.has(i)) for (let k = 0; k < MARK_STRIDE; k++) kept.push(mk[i + k]);
    this.marks.setShown(new Float32Array(kept));
  }
  dropPlaced() {
    this.uncached(() => { for (const r of this.placedRes) r.del(); });
    this.placedRes = [];
    this.placed = [];
  }
  placed: StrokeVao[] = [];
  placedRes: { del: () => void }[] = [];

  /** The blueprint track picked by a drag selection (T-333), drawn as an ink band under the track, or null. */
  setHighlight(s: Strokes | null) {
    this.highlightSrc = s;
    for (const r of this.highlightRes) r.del();
    this.highlightRes = [];
    this.highlight = s && s.count ? strokes(this, s, this.highlightRes) : null;
  }

  /** T-3059: the picked pieces a rebuild would refuse, outlined in red over the pick's band, or null. */
  setRefused(s: Strokes | null) {
    this.refusedSrc = s;
    for (const r of this.refusedRes) r.del();
    this.refusedRes = [];
    this.refused = s && s.count ? strokes(this, s, this.refusedRes) : null;
  }

  /** What the setters were last given, for `restore`. */
  layersSrc: NetworkLayers | null = null;
  refusedSrc: Strokes | null = null;
  previewSrc: Strokes | null = null;
  highlightSrc: Strokes | null = null;

  /** The canvas lost its context and the browser gave it a new one (T-953): every GL object went
   * with the old, so the programs are compiled again and what was drawn is uploaded again. */
  restore() {
    this.res = [];
    this.texRes = {};
    this.keptTex = {};
    this.slotTex = {};
    this.sampleSlots = this.bendSlots = this.phaseSlots = null;
    this.fullKey = null;
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
   * the clock, by when its trips are here. `loads` (T-3655): its first trips come without loads, so it waits for the
   * ones with them, or goes in with a newer state due (the worker sends those under the newest version). */
  staged: { version: number; at: number; since: number; apply: () => void; trips: TripArgs[]; loads: boolean }[] = [];

  /** Draw a state (`apply` puts it in) from game time `at`, once its trips have come (`loads`: with their loads). */
  stage(version: number, at: number, apply: () => void, loads = false) {
    this.staged.push({ version, at, since: performance.now(), apply, trips: [], loads });
  }

  /** Put in every staged state now: a state drawn at once comes after them. */
  flushStaged() {
    while (this.staged.length) this.putStaged(this.staged.shift()!);
  }

  /** when the last staged state took effect (checks/t1071-train-count.mjs) */
  stagedAt = NaN;

  putStaged(s: (typeof this.staged)[number]) {
    this.stagedAt = s.at;
    s.apply();
    for (const t of s.trips) this.putTrips(...t);
  }

  /** Put in the staged states due at `time`, in order. */
  releaseStaged(time: number) {
    const now = performance.now();
    const due = (i: number): boolean => {
      const s = this.staged[i];
      if (time < s.at) return false;
      // trips that never come (the worker restarted) hold a state up no longer than this
      if (now - s.since > STAGE_WAIT_MS) return true;
      if (!s.loads) return s.trips.length > 0;
      return s.trips.some((t) => t[3]) || (i + 1 < this.staged.length && due(i + 1));
    };
    while (this.staged.length && due(0)) this.putStaged(this.staged.shift()!);
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

  putTrips(epoch: number, version: number, data: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array, bays?: Float32Array) {
    const old = this.tripsBy.get(epoch);
    if (old && old.version > version) return;
    const hour = board ? readBoardHour(epoch, board) : undefined;
    this.tripsBy.set(epoch, { version, data, ends: ends && ends.length === data.length / 2 ? ends : undefined, nexts: nexts && nexts.length === data.length / 2 ? nexts : undefined, bays: BAYS && bays && bays.length === (data.length / 2) * BAY_STRIDE ? bays : undefined, board: hour && hour.trips === data.length / 2 ? hour : undefined });
    if (epoch === this.tripEpoch) this.tripVersion = -1; // re-upload on the next draw
  }

  /** Make the trip buffer hold the trips of the hour `time` is in, when they are here. */
  sync(time: number): number {
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
        this.loadTex = texture(this, t.board.loads, "r32f", TEX_W, []);
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
    if (this.board && zoom >= STATIONS_ZOOM) wait = Math.min(wait, (1000 * CROWD_STEP_S) / speed);
    return wait;
  }

  /**
   * Draw with the camera MapLibre last drew (gl.ts `Camera`, flat or globe); `time` is game seconds;
   * `zoom` and `dpr` size things in px. The caller has bound and cleared the target; `view`, device
   * px, when it is not the whole drawing buffer (the complex map's snapshot, overlay.ts).
   */
  draw(cam: Camera, time: number, zoom: number, dpr: number, opt: DrawOptions, view?: readonly [number, number]) {
    drawFrame(this, cam, time, zoom, dpr, opt, view);
  }

  /** ms the guard took the last frame (checks) */
  guardMs = 0;
  guardOut: [number, number] = [0, 0];
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

  /**
   * GPU positions (local units, x/y pairs; 1e9 when off the track) of every trip in the buffer at
   * game time `time`, read back by transform feedback. For precision checks; changes GL state
   * freely, so only call it on a renderer with a context of its own.
   */
  probe(time: number): Float32Array {
    const gl = this.gl;
    if (!this.probeProg) {
      this.probeProg = compile(gl, PROBE_VS, PROBE_FS, ["v_pos"]);
      this.probeU = uniforms(gl, this.probeProg, ["u_time", "u_phases", "u_meta", "u_samples", "u_bend_at", "u_bends", "u_lines", "u_back_base"]);
    }
    this.releaseStaged(time);
    const uTime = this.sync(time);
    const n = this.tripCount;
    if (!this.probeVao) {
      this.probeVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.probeVao);
      attrib(this, this.probeProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 0);
      attrib(this, this.probeProg, "a_end", this.endBuf!, 1, gl.FLOAT, 0);
      attrib(this, this.probeProg, "a_lag", this.lagBuf!, 1, gl.FLOAT, 0);
      gl.bindVertexArray(null);
    }
    const out = buffer(this, new Float32Array(Math.max(1, n) * 2), []);
    gl.useProgram(this.probeProg);
    gl.uniform1f(this.probeU.u_time, uTime);
    bindTrainTextures(this, this.probeU);
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

/** the lane texture's four numbers of `count` samples from `first` into `v` from 0 (`putLane`) */
function packLane(v: Float32Array, first: number, count: number, lane?: Float32Array, band?: Float32Array, pass?: Float32Array) {
  for (let i = 0; i < count; i++) {
    const k = first + i;
    v[i * 4] = lane ? lane[k] ?? 1 : 1;
    v[i * 4 + 1] = band?.[k * 2] ?? 0;
    v[i * 4 + 2] = band?.[k * 2 + 1] ?? 0;
    v[i * 4 + 3] = pass?.[k] ?? 1e9;
  }
}

/** T-4280: per line id its keys in the per-sample tables (`runs`: its run 0 and its return run's own way back, trainSlots.ts
 * `sampleRows`), else its phase tables' (its id) */
function namedKeys(ids: readonly number[], runs: boolean): Set<number> {
  const out = new Set<number>();
  for (const id of ids) if (runs) out.add(id * 2).add(id * 2 + 1); else out.add(id);
  if (trainNamesTest.drop) for (const k of out) { out.delete(k); break; }
  return out;
}

/** T-4280: the lists the trains' tables are written by, against network `was`'s own arrays: the samples and bends of the
 * lines whose drawn runs changed (`touched.paths`; `bendAt` also where a line's bend points moved slot, since its codes
 * point at them), the phase tables of `touched.phases`, the stretches and lanes of the patch's `stretches` and `lanes` */
function trainNames(was: NetworkBuffers, n: NetworkBuffers, touched: EditTouched, patch: NetPatch, bendsWere: SlotLayout | null, bends: SlotLayout): Partial<Record<SlotName, SlotNames>> {
  const paths = namedKeys(touched.paths, true), bendAt = new Set(paths);
  if (bends !== bendsWere) for (const [k, a] of bends.at) if (bendsWere?.at.get(k) !== a) bendAt.add(k);
  const lane = namedKeys(patch.lanes, true);
  // (near the bend points a code can hold, the clock worker draws the runs past it straight, wherever they lie: compared)
  const cut = Math.max(was.bends?.length ?? 0, n.bends?.length ?? 0) / 2 + 32 >= BEND_LIMIT;
  return {
    samples: { keys: paths, from: [was.samples] },
    ...(cut ? {} : { bendAt: { keys: bendAt, from: [was.sampleBend] }, bends: { keys: paths, from: [was.bends] } }),
    phases: { keys: namedKeys(touched.phases, false), from: [was.phases] },
    stretch: { keys: namedKeys(patch.stretches, true), from: [was.sampleStretch] },
    lane: { keys: lane, from: [was.sampleLane?.length ? was.sampleLane : undefined, was.sampleBand, was.samplePass] },
  };
}

/** the trains' tables kept in slots (trainSlots.ts) */
type SlotName = "samples" | "bendAt" | "bends" | "phases" | "stretch" | "lane" | "fill" | "slevel";

/** two float arrays bit for bit */
function sameWords(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length), y = new Uint32Array(b.buffer, b.byteOffset, b.length);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

/** widths that draw as the plain ones: every width 1, no band shift */
function isPlainWmul(wm: Float32Array): boolean {
  for (let i = 0; i < wm.length; i += 3) if (wm[i] !== 1 || wm[i + 1] !== 0 || wm[i + 2] !== 0) return false;
  return true;
}

/** T-4262: segments a row of the widths' rows compared and written (3 numbers each) */
const WMUL_ROW = 1024;

/** The lines' `a_wmul` as `wm` (null: plain): only the rows that differ from what the buffer holds (`wmulHeld`) when
 * it holds demand widths and `wm` is some; whole otherwise. Whether anything was written. */
function writeWmul(rnd: NetworkRenderer, v: StrokeVao, wm: Float32Array | null): boolean {
  const held = v.wmulPlain ? null : v.wmulHeld;
  if (!rnd.keepGl || !held || !wm || held.length !== wm.length) {
    writeSegs(rnd, v, v.wmulBuf, wm ?? plainWmul(v.count), 3);
    return true;
  }
  const rows = changedRows(held, wm, WMUL_ROW * 3);
  if (!rows.length) return false;
  const gl = rnd.gl;
  gl.bindBuffer(gl.ARRAY_BUFFER, v.wmulBuf);
  for (let i = 0; i + 1 < rows.length; i += 2) writeSegRange(rnd, v, wm, 3, rows[i] * WMUL_ROW, Math.min(v.count, (rows[i] + rows[i + 1]) * WMUL_ROW) - rows[i] * WMUL_ROW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return true;
}

const EMPTY_FILL_ROWS: ReadonlyMap<number, Float32Array> = new Map();

export interface StationReach { net: NetworkBuffers; r: Float32Array; halfWidth: number; maxHwAt: number }

/** T-4314: work counters for the stations' reach pass (last call and totals) */
export const reachWork = { calls: 0, patched: 0, whole: 0, lastStations: 0, stations: 0 };

/** `?reachTest=drop` (checks): skips the first named station in `patch.stations.changed`, which `?patch=check` must catch */
export const reachTest = { drop: typeof location !== "undefined" && new URLSearchParams(location.search).get("reachTest") === "drop" };

function stationReachAt(n: NetworkBuffers, i: number): number {
  const sh = platformShapeAt(n, i);
  const q = platformPoints(sh);
  let m = Math.hypot(sh.mx, sh.my) + Math.hypot(sh.ax, sh.ay);
  for (let j = 0; j + 1 < q.length; j += 2) m = Math.max(m, Math.hypot(q[j], q[j + 1]));
  return m;
}

function stationReachWhole(n: NetworkBuffers): StationReach {
  const r = new Float32Array(n.stationCount), s = n.stationShape;
  let halfWidth = 0, maxHwAt = -1;
  for (let i = 0; i < n.stationCount; i++) {
    const hw = s?.[i * 4 + 3] ?? 0;
    if (hw > halfWidth) { halfWidth = hw; maxHwAt = i; }
    r[i] = stationReachAt(n, i);
  }
  return { net: n, r, halfWidth, maxHwAt };
}

/** T-4314: stations' reach for `n`, patched from `prev` using `n.patch.stations` (touching only the named stations)
 * when `prev` is `n.patch.base`, else built whole; under `check`, compared against a fresh whole build. */
export function stationReachStep(n: NetworkBuffers, prev: StationReach | null, check: boolean): StationReach {
  const sp = prev && n.patch?.stations && prev.net.serial === n.patch.base ? n.patch.stations : null;
  let out: StationReach;
  let touched = 0;
  if (sp) {
    const s = n.stationShape;
    const changed = reachTest.drop && sp.changed.length ? sp.changed.slice(1) : sp.changed;
    let r: Float32Array;
    let halfWidth = 0, maxHwAt = -1;
    if (sp.runs === null && changed.length === 0) {
      r = prev!.r;
      halfWidth = prev!.halfWidth;
      maxHwAt = prev!.maxHwAt;
    } else {
      if (sp.runs === null) {
        r = prev!.r.slice();
        if (prev!.maxHwAt >= 0 && !changed.includes(prev!.maxHwAt)) {
          halfWidth = prev!.halfWidth;
          maxHwAt = prev!.maxHwAt;
        }
      } else {
        r = new Float32Array(n.stationCount);
        for (let k = 0; k + 2 < sp.runs.length; k += 3) {
          const i0 = sp.runs[k], j0 = sp.runs[k + 1], count = sp.runs[k + 2];
          r.set(prev!.r.subarray(j0, j0 + count), i0);
          if (maxHwAt < 0 && prev!.maxHwAt >= j0 && prev!.maxHwAt < j0 + count) {
            halfWidth = prev!.halfWidth;
            maxHwAt = i0 + (prev!.maxHwAt - j0);
          }
        }
      }
      for (const i of changed) {
        r[i] = stationReachAt(n, i);
        touched++;
        if (maxHwAt >= 0) {
          const hw = s?.[i * 4 + 3] ?? 0;
          if (hw > halfWidth) { halfWidth = hw; maxHwAt = i; }
        }
      }
      if (maxHwAt < 0 && n.stationCount > 0) {
        for (let i = 0; i < n.stationCount; i++) {
          const hw = s?.[i * 4 + 3] ?? 0;
          if (hw > halfWidth) { halfWidth = hw; maxHwAt = i; }
        }
      }
    }
    out = { net: n, r, halfWidth, maxHwAt };
    reachWork.patched++;
    if (check) timed("dev.check.reach", () => {
      const whole = stationReachWhole(n);
      if (!sameWords(whole.r, r) || whole.halfWidth !== halfWidth) console.error("T-4314 check: the stations' reach patched from the last network differs from a whole build");
    });
  } else {
    out = stationReachWhole(n);
    touched = n.stationCount;
    reachWork.whole++;
  }
  reachWork.calls++;
  reachWork.lastStations = touched;
  reachWork.stations += touched;
  return out;
}
