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
import { LOT_STRIDE } from "./facilityPlace";
import { F_BLUEPRINT, platformShapeAt, sampleValue, type NetworkBuffers, type Strokes } from "./network";
import { BAY_STRIDE } from "./terminalBays";
import { BAND_RANGE, TRAFFIC_PICK_DIV, type TrafficBands } from "./lineBands";
import { BRIDGE_VS, HOT_STATIONS, LINES_W, PROBE_FS, PROBE_VS, SEL_FS, SEL_MAX, SEL_STRIDE, SEL_VS, STATION_FS, STATION_VS, STROKE_FS, STROKE_VS, TEX_W, TRAIN_FS, TRAIN_VS } from "./rendererShaders";
import { codeHidden, hidden as vis, levelHidden, levelShown, pickSkip } from "./visibility";
import { EPOCH_S, STAGE_WAIT_MS, TripArgs, LINE_PX, noBays, BAYS, StationGroups, DrawOptions, StrokeVao, TexName, plainWmul, fullAttr, STROKE_ATTRIBS, STILL_MAX_MS, StrokeLodState, NetworkLayers, TRAIN_MAX_MPS, CROWD_STEP_S } from "./rendererParts";
export { EPOCH_S, laneState, passingState, CROWD_ZOOM, CONFLICT_ZOOM } from "./rendererParts";
export type { StationGroups, DrawOptions, NetworkLayers } from "./rendererParts";
import { texture, buffer, attrib, strokes, checkStrokes, checkFull, dropLod, rewrite } from "./rendererStrokes";
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
  tex: Record<TexName, WebGLTexture | null> = { phases: null, meta: null, samples: null, lines: null, colours: null, fill: null, cars: null, stretch: null, stationLine: null, lane: null, slevel: null };
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
  reachOf: { net: NetworkBuffers; r: Float32Array; halfWidth: number } | null = null;
  /** Per station, local units from its node to the farthest point of its platform's centre line,
   * and the widest true half width of any (stationAt's skip, T-724); worked out when first asked
   * for, a few ms where scanning every station's shape cost Japan's network 100 ms a move (T-1020). */
  stationReachNow(n: NetworkBuffers): { r: Float32Array; halfWidth: number } {
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
    return this.stationStyle.boxes && !this.stationsOff ? this.lots.at(px, py, project) : -1;
  }
  /** A selected station drawn at a point, or -1 (picking: it is above the demand bubbles). */
  selectedStationAt(px: number, py: number, project: Project, padding = 2): number {
    if (this.stationsOff) return -1;
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
    this.oneWayGhost.init(gl);
    this.lots.init(gl);
    this.floors.init(gl);
    this.bubbles.init(gl);
    this.buses.init(gl);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.hasStencil = (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0;
    // (Firefox with fingerprinting resistance says 2048 for both, under many a window's canvas)
    this.cacheMax = Math.min(gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) ?? 0, gl.getParameter(gl.MAX_TEXTURE_SIZE) ?? 0);
    this.strokeU = uniforms(gl, this.strokeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_casing", "u_casing_rgb", "u_cap", "u_kind", "u_sel_line", "u_sel_n", "u_sel_set", "u_sel_edge", "u_hot", "u_hide", "u_only_sel", "u_bp_part", "u_lanes", "u_lane_mode", "u_pass_off","u_dim", "u_by_level", "u_by_speed", "u_by_profit", "u_profit", "u_profit_n", "u_level_colours", "u_track", "u_bad", "u_sel", "u_bp", "u_bp_mid", "u_bp_rail", "u_bp_off_mid", "u_bp_off_rail", "u_bp_off_fade", "u_line_colours", "u_full_pass", "u_full_px", "u_full_out", "u_full_dash", "u_full_rgb", "u_full_inside", "u_ghost", "u_ghost_layer", "u_ghost_rgb", "u_ghost_min", "u_hsr", "u_wire", "u_band", "u_band_traffic", "u_band_gap", "u_band_m", "u_mute", "u_level_hide"]);
    this.stationU = uniforms(gl, this.stationProg, [...PROJECT_UNIFORMS, "u_viewport", "u_radius", "u_ring", "u_ink", "u_line", "u_boxes", "u_fixed", "u_pass_half", "u_pass_in", "u_pass", "u_layer", "u_pixel", "u_bad", "u_crowd_rgb", "u_tint", "u_mask", "u_stline", "u_lanes", "u_hot_st", "u_hot_n", "u_room", "u_bld", "u_one", "u_one_mid"]);
    this.bridgeProg = compile(gl, BRIDGE_VS, STATION_FS);
    this.bridgeU = uniforms(gl, this.bridgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_ring", "u_bridge", "u_pass", "u_layer", "u_ink", "u_bad", "u_crowd_rgb", "u_mask", "u_hot_st", "u_hot_n"]);
    // T-3192: the stations' u_stline, which a bridge never reads, on the stations' unit rather than one an integer texture may hold
    gl.useProgram(this.bridgeProg);
    gl.uniform1i(gl.getUniformLocation(this.bridgeProg, "u_stline"), 11);
    this.selProg = compile(gl, SEL_VS, SEL_FS);
    this.selU = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_shapes", "u_count", "u_dpr", "u_height", "u_ink", "u_outline", "u_glow"]);
    this.selTex = texture(this, this.selData, "rgba32f", SEL_MAX * SEL_STRIDE, []);
    this.trainU = uniforms(gl, this.trainProg, [...PROJECT_UNIFORMS, "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_phases", "u_meta", "u_samples", "u_lines", "u_back_base", "u_line_colours", "u_fill", "u_bad", "u_detail", "u_dpr", "u_cars", "u_loads", "u_stretch", "u_sel_line", "u_sel_n", "u_sel_set", "u_dim", "u_sel_trip", "u_pass", "u_ink", "u_band", "u_psi0", "u_lane_tex", "u_lane", "u_hollow", "u_bands", "u_bands_m", "u_passing", "u_keep", "u_level_hide", "u_slevel"]);
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
      v?.attribs?.forEach(({ buf, bytes }, i) => (out[`${name}.${STROKE_ATTRIBS[i]}`] = [buf, Math.max(1, v.count) * bytes]));
    return out;
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
    if (lines.count) rewrite(this, lines.wmulBuf, wm ?? plainWmul(lines.count));
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
    rewrite(this, l.fullBuf, out);
    this.linesFull = out;
    this.perSegV++;
  }
  anyFull = false;

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
    this.track = strokes(this, n.track, [], patch && oldTrack ? { old: oldTrack, runs: patch.track } : null);
    // its a_full is `writeFull`'s, just below
    this.lines = strokes(this, n.lines, [], patch && oldLines ? { old: oldLines, runs: patch.lines } : null, { full: "later" });
    for (const r of [...(oldTrack?.res ?? []), ...(oldLines?.res ?? [])]) r.del();
    dropLod(this);
    this.linesWmul = this.linesFull = null;
    this.perSegV++;
    if (patch && this.checkPatch) {
      checkStrokes(this, "track", this.track, n.track);
      checkStrokes(this, "lines", this.lines, n.lines);
      checkFull(this, this.track, n.track);
    }
    this.fullDrawn = "";
    this.writeFull();
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
    put("colours", "lineColours", () => [this.shownColours(n), "rgba8", LINES_W]);
    this.hiddenFor = null;
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
  texRes: Partial<Record<TexName, { del: () => void }[]>> = {};
  /** per sample: how far apart its tracks are (T-658) and its line's band's shift as drawn (T-1799),
   * line widths and metres (T-2132) */
  putLane(n: NetworkBuffers) {
    const m = Math.max(1, n.samples.length / 2), lane = n.sampleLane, band = this.trafficBands?.sample ?? n.sampleBand, v = new Float32Array(m * 4);
    for (let k = 0; k < m; k++) {
      v[k * 4] = lane?.length ? lane[k] ?? 1 : 1;
      v[k * 4 + 1] = band?.[k * 2] ?? 0;
      v[k * 4 + 2] = band?.[k * 2 + 1] ?? 0;
      v[k * 4 + 3] = n.samplePass?.[k] ?? 1e9;
    }
    this.putTexture("lane", v, "rgba32f", TEX_W);
  }

  putTexture(name: TexName, data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number) {
    for (const r of this.texRes[name] ?? []) r.del();
    const list: { del: () => void }[] = [];
    this.tex[name] = texture(this, data, kind, width, list);
    this.texRes[name] = list;
  }

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    this.previewSrc = s;
    for (const r of this.previewRes) r.del();
    this.previewRes = [];
    this.preview = s && s.count ? strokes(this, s, this.previewRes) : null;
    this.marks.setPreview(s?.marks);
    this.oneWayGhost.setNetwork(s?.chevrons);
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
    this.oneWayGhost.setNetwork(undefined);
  }
  /** T-2404: a placed segment with no ghost of it (the pointer moved on before the click's own was
   * answered), drawn as a stand-in from its edges */
  addPlaced(s: Strokes) {
    if (s.count) this.placed.push(strokes(this, s, this.placedRes));
  }
  /** T-3420: the network and the `vis.v` the line colours' alphas, the station flags and the picks' skips were set for */
  hiddenFor: NetworkBuffers | null = null;
  hiddenV = -1;
  /** per station: on a hidden level */
  stationHid = new Uint8Array(0);
  /** T-3420: the line colours with a hidden line's alpha 0 (the stroke and train shaders read it) */
  shownColours(n: NetworkBuffers): Uint8Array {
    if (!vis.lines.size) return n.lineColours;
    const c = n.lineColours.slice();
    for (const l of vis.lines) if (l * 4 + 3 < c.length) c[l * 4 + 3] = 0;
    return c;
  }
  /** Whether station `i` is on a hidden level (T-3420). */
  stationHidden(i: number): boolean {
    return this.stationHid[i] === 1;
  }
  sampleLv: Float32Array | null = null;
  /** T-3560: whether a train (as `tripAt` places it) is on a switched-off level, drawn faint and not pickable */
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
    this.stationHid = new Uint8Array(n.stationCount);
    if (vis.levels) for (let i = 0; i < n.stationCount; i++) {
      const l = vis.stationLevel(n.stationIds[i]);
      if (l !== undefined && levelHidden(l)) this.stationHid[i] = 1;
    }
    this.selDrawnKey = "";
    this.sampleLv = vis.levels ? this.sampleLevels(n) : null;
    if (this.sampleLv) this.putTexture("slevel", this.sampleLv, "r32f", TEX_W);
    else if (this.tex.slevel) { for (const r of this.texRes.slevel ?? []) r.del(); this.texRes.slevel = []; this.tex.slevel = null; }
    const L = n.lines, T = n.track;
    if (vis.lines.size || vis.levels) {
      pickSkip.set(L.seg, (i) => vis.lines.has(L.colour[i]) || codeHidden(L.level[i]));
      if (vis.levels) pickSkip.set(T.seg, (i) => codeHidden(T.level[i]));
      else pickSkip.delete(T.seg);
    } else {
      pickSkip.delete(L.seg);
      pickSkip.delete(T.seg);
    }
  }
  /** T-2763: edges the player just removed, not drawn until the network without them is set (or `unhideEdges`). */
  hide: [number, number] = [-2, -2];
  hideEdge(edge: number) {
    this.hide = [edge, this.hide[0] === edge ? this.hide[1] : this.hide[0]];
    this.uploads++;
  }
  unhideEdges() {
    if (this.hide[0] === -2 && this.hide[1] === -2) return;
    this.hide = [-2, -2];
    this.uploads++;
  }
  dropPlaced() {
    for (const r of this.placedRes) r.del();
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
  staged: { version: number; at: number; since: number; apply: () => void; trips: TripArgs[] }[] = [];

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

  putStaged(s: (typeof this.staged)[number]) {
    this.stagedAt = s.at;
    s.apply();
    for (const t of s.trips) this.putTrips(...t);
  }

  /** Put in the staged states due at `time`, in order. */
  releaseStaged(time: number) {
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
      this.probeU = uniforms(gl, this.probeProg, ["u_time", "u_phases", "u_meta", "u_samples", "u_lines", "u_back_base"]);
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
