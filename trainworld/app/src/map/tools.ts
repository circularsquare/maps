// The map tools (T-023, T-024): what a click does while drawing track, placing stations or adding
// stops to a line. Selecting (picking) is map/pick.ts; main.ts routes clicks here when a tool is
// on.
//
// Track, drawn through points (T-418): the second click places a
// segment from the first, one curve leaving along the track there and passing through the click,
// at once and never changed by later clicks; with Shift held or keepPlacing on (T-445) the next
// segment starts at its end, until a double click, Escape or Enter.
// The first click may start on a track end or on track (a new junction, flat, T-080); a click on
// track or a node joins it there. Backspace takes the last segment of a chain back, Escape, Enter or a
// right click on the map (T-3441) stops. On quad track a segment from a loose quad end carries both pairs on (T-423). While
// drawing, the worker previews the segment (one request in flight): its cost, whether it can be
// built and why not, and the radius and speed of its curve.
//
// Delete (T-096): hovering shows what a click removes (a station or flyover ringed, a stretch of
// track drawn red, its name by the cursor); a click removes blueprint at once and asks first for
// anything constructed (game/remove.ts).
//
// On a touch screen (T-1260) a tap previews and a second tap on it, or the control bar's confirm,
// is the click; the bar and its handle are placed here, drawn by ui/TouchBar.tsx.
//
// Everything drawn is blueprint until constructed (SPEC 6.4).

import { t as tr } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";
import { effect } from "@preact/signals";
import { type Map as MlMap, type MapMouseEvent } from "maplibre-gl";
import { lngLatToPack, packToLngLat } from "../game/coords";
import { distance, money } from "../game/format";
import { activeCity, worldFrameEpoch } from "../game/cities";
import { demandView } from "../game/demand";
import { settleNewLineColour } from "../game/lineColour";
import { metresBetween } from "../game/walks";
import { ask, autoRamp, buildLevel, buildPlatform, buildTramPlatform, buswayTrack, designSpeed, flyoverSpeed, highSpeed, inClickOrder, keepPlacing, lineDraft, newLine, quadTrack, selection, singleTrack, besideOn, snapping, snapMode, snapRealTram, snapRoads, snapStreets, stationKind, stationOnTram, stationOneTrack, tool, tramTrack } from "../game/ui";
import { type Removable } from "../game/remove";
import { loadPlacesAt } from "../game/placeList";
import { warmStreetNames } from "./stationNames";
import { loadRealStationsAt } from "./realStations";
import { pointerLocal } from "./pick";
import { edgesNear, nodesNear } from "./snapIndex";
import { clickNet, clickWorld, patchFresh } from "./clickMap";
import { hidden, levelHidden } from "./visibility";
import { afterStopEdits, edit, queueStopEdit, say, sendStopEdit, stopsQueued, world } from "../game/world";
import { hideBusy, showBusy } from "../game/busy";
import { type PiInput, type RouteEnd } from "../workers/protocol";
import { gameStamp, onRoadsLanded, type ClockClient, type PreviewAnswer } from "../workers/clockClient";
import { roadTileTemplate } from "./busRoads";
import { kAtY, toLngLat, toLocal } from "./geo";
import { laneState } from "./renderer";
import { planStopsOf, type Level } from "../game/types";
import { endHeading } from "./curveCircle";
import { ahead, asRun, editsSentCount, lastSentSeq, setRunBreak, type Ahead, type Show } from "../game/pending";
import { ghostTangent, LOCAL_PX, sameLine, warmGhostLocal, type TangentSpec } from "./ghostLocal";
import { ghostChevrons, joinedGhost, strokesFromPreview, TAPER_M, type Strokes } from "./network";
import { type PlatformShape } from "./stationGeometry";
import { type Overlay } from "./overlay";
import { marksLayer } from "./screenPlace";
import { type StopMarkers, type StopTier } from "./stopMarkers";
import { type BusStopMarkers } from "./busStops";
import { fractionAlong, orderOverlapping, type Layout } from "./stationSnap";
import { RealSnap, type RealClick, type RealRoute } from "./realSnap";
import { StreetSnap, type StreetClick, type StreetRoute } from "./streetSnap";
import { SAME_TAP_PX, touchActions, touchInput } from "./touch";
import { BusDraw } from "./busDraw";
import { FerryDraw } from "./ferryDraw";
import { BuildingDraw } from "./buildingDraw";
import { startBuildingFloors } from "./buildingFloors";
import { playBuild } from "./buildSound";
import { perfOff } from "../perfFlags";
import { guessRow, settleRow } from "../ui/inspector/stopRowGuess";
import { type LngLatish, SNAP_EDGE_PX, TRACK_NODE_PX, TRACK_EDGE_PX, DEAD_PX, DOT_MIN_PX, HELD_NEAR_PX, HELD_START_PX, BESIDE_PX, OVERLAP_PX, type Snap, type Spot, type Verdict, BESIDE_SPACING_M, LANE_HALF_M, type Placed, MAX_PIECE_KM, WAIT_MS, tooFarText, placedAt, standInPts } from "./toolParts";
import { nearTouch, touchConfirm, touchBack, touchStop, dragHandle, placeTouch } from "./toolTouch";
import { autoPick } from "./stationAuto";
import { nameAhead, stationClick, askLegs, placePending, layout } from "./stationTool";
import { freeCancel, freeStarted } from "./stationFree";
import { clearDelHover, deleteHover, deleteClick } from "./toolDelete";
import { clickWaiting, dropClicks, startClickWait, tryClicks, waitClick, waitSpin, type WaitingClick } from "./toolWait";
import { drawBand, placeMarks } from "./toolMarks";

export class Tools {
  start: Placed | null = null;
  /** the starts of the segments placed with Shift, for Backspace (T-418) */
  chain: Placed[] = [];
  /** a segment being worked out before it is sent (a real or street route found first), or one sent on a refused ghost
   * until its answer: clicks wait for it */
  placing = false;
  /** T-4214 stage D: the run's segments sent and not yet answered, where each lies as drawn: a click on one waits for it
   * (`ownTrackOut`) */
  flying: { seq: number; pts: Float32Array }[] = [];
  /** T-4214: map clicks waiting for the map to hold every edit sent before them, in order (map/toolWait.ts) */
  waitingClicks: WaitingClick[] = [];
  waitTimer = 0;
  /** T-4214: clicks being carried out whose tool has not finished sending (a station's name looked up first) */
  clicksOut = 0;
  /** T-4214: the track run being drawn, numbered from 1: its own segments hold none of its clicks */
  run = 0;
  /** T-1407: the end node of a segment answered before its state, until the tools' world has it (-1: none) */
  earlyNode = -1;
  /** T-2404: the points (local x, y, z) of the segments placed early whose state the worker holds
   * while the pointer moves; the snap cannot see them until it lands */
  held: Float32Array[] = [];
  /** a double click's second click came while the segment was still being placed (T-445) */
  endRun = false;
  hover: Snap | null = null;
  /** the track the pointer is beside, while drawing from beside track (T-1685) */
  hoverBeside: { edge: number; lx: number; ly: number } | null = null;
  /** the delete tool: what a click would remove, and the pointer (map px) */
  delHover: (Removable & { lx: number; ly: number }) | null = null;
  delAt: [number, number] = [0, 0];
  lastPreview: PreviewAnswer | null = null;
  /** T-4136: while the pointer has moved on from the spot `lastPreview` answers, the ghost drawn is that answer bent
   * so its end is at the pointer (`bend`), `from` the answer it was made from */
  bent: { from: PreviewAnswer; a: PreviewAnswer } | null = null;
  /** T-4208: from a click's segment being sent until its answer: where the run goes on from (null when it ends at the
   * click). The ghost is drawn from there meanwhile, never from the start being left, which the worker may take a
   * second to answer behind a network it is building. */
  clicked: { onward: { lx: number; ly: number; lng: number; lat: number; dir?: [number, number]; z: number; tangent?: { level: number } } | null } | null = null;
  /** T-4210: a start whose worker answer differed from the ghost laid on the page (water, a ramp): bent from there on */
  localOff: Placed | null = null;
  /** T-4210: the last click's stand-in is the segment as laid (its answer or the page's), not the ghost bent to it */
  stampExact = false;
  /** T-4210: the ghost laid on the page for the newest request, to hold its answer against; the last one laid */
  localFor: { key: string; pts: Float32Array } | null = null;
  private lastLocal: { st: object; h: Snap; g: ReturnType<typeof ghostTangent> } | null = null;
  /** the preview the ghost, its marks and its tag show: the bent one while it stands in */
  get shownPreview(): PreviewAnswer | null {
    if (this.clicked) return null; // T-4208: no answer yet for the ghost going on from the click
    return this.bent && this.bent.from === this.lastPreview ? this.bent.a : this.lastPreview;
  }
  /** where the ghost starts: the run's start, or from a click until its answer where the run goes on from (null: it
   * ends there), T-4208 */
  get ghostStart(): { lng: number; lat: number } | null {
    return this.clicked ? this.clicked.onward : this.start;
  }
  /** snapping to the real network (T-727d) */
  readonly real: RealSnap;
  /** snapping tram track to streets (T-2927) */
  readonly street: StreetSnap;
  /** the route between the clicks is a snapped one, real or street, not an ordinary segment */
  get snapped() {
    return this.real.active || this.street.active;
  }
  /** the bus line tool (T-1464) */
  readonly bus: BusDraw;
  /** the ferry line tool (T-2194) */
  readonly ferry: FerryDraw;
  /** the station building tool (T-2218) */
  readonly building: BuildingDraw;
  /** the request `lastPreview` answers, as JSON */
  previewKey = "";
  /** the newest request asked for (T-830: a click takes only the answer for its own spot) */
  wantKey = "";
  /** where the last station of this session was placed (lng, lat), T-312 */
  lastPlaced: [number, number] | null = null;
  box: HTMLDivElement;
  marks: HTMLDivElement[] = [];
  tag: HTMLDivElement;
  ring: HTMLDivElement;
  editingLine: number | null = null;
  /** the toggle markers on every station while a line's stops are edited (T-233); set by main.ts */
  stops: StopMarkers | null = null;
  busStops: BusStopMarkers | null = null;
  /** T-3731: the busy tag's token while stop clicks are still being sent after stop editing ended */
  savingStops = 0;
  /** T-1260, on touch: the tap whose preview waits to be confirmed (the track's end, a station's
   * spot, what the delete tool would remove) */
  touchAt: LngLatish | null = null;
  /** a station just placed by touch: no ghost at that point until the finger moves or taps elsewhere (T-1756) */
  ghostSpent = false;
  /** the touch control bar's host (ui/TouchBar.tsx draws into it), and the drag handle on `touchAt` */
  readonly touchHost: HTMLDivElement;
  handle: HTMLDivElement;
  touchKey = "";

  constructor(readonly map: MlMap, readonly overlay: Overlay, readonly client: ClockClient) {
    for (const k of ["keydown", "keyup"] as const) window.addEventListener(k, (e) => this.noSnapFrom(e));
    window.addEventListener("blur", () => this.noSnapFrom({ ctrlKey: false, metaKey: false }));
    this.box = marksLayer(map);
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag build-tag";
    this.tag.style.display = "none";
    this.ring =document.createElement("div");
    this.ring.className = "snap-ring";
    this.box.append(this.ring);
    // outside the marks' layer, so no demand tag covers it (T-526)
    map.getContainer().appendChild(this.tag);
    // T-1464: drawing a bus line (map/busDraw.ts); it uses this tag, and the touch bar below
    this.bus = new BusDraw(map, this.box, this.tag, client, overlay, { exit: () => this.exit() });
    effect(() => { if (tool.value !== "bus") this.bus.clear(); });
    this.ferry = new FerryDraw(map, this.box, this.tag, overlay, { exit: () => this.exit() });
    effect(() => { if (tool.value !== "ferry") this.ferry.clear(); });
    this.building = new BuildingDraw(map, this.box, this.tag, overlay, { exit: () => this.exit() });
    effect(() => { if (tool.value !== "building") this.building.clear(); });
    effect(() => { if (tool.value === "track") warmGhostLocal(); });
    startBuildingFloors(overlay);
    overlay.afterMapFrame.push(() => this.place());
    // T-1777: a pointer gone off the map leaves no ghost, ring or tag where it was; T-3130: a segment being
    // drawn keeps its ghost, and so its tag with the price or why it can't be built
    map.getContainer().addEventListener("mouseleave", () => {
      if (touchInput.peek() || (!this.hover && !this.hoverBeside && !this.delHover)) return;
      if (tool.peek() === "track" && this.start) return;
      this.hover = this.hoverBeside = null;
      clearDelHover(this);
      this.place();
    });
    effect(() => { buildPlatform.value; buildTramPlatform.value; stationKind.value; if (tool.value === "station") this.place(); });
    // T-3999: an open-ground station's first end goes with the tool or the kind
    effect(() => { if (tool.value !== "station" || (stationKind.value !== "rail" && stationKind.value !== "tram")) freeCancel(this); });
    // the place lists name stations where the tiles have no streets (T-065); the view's zones' are
    // fetched when the station tool is picked, not at startup (T-400)
    // and the real stations of the view's region, which name a station on one (T-3571)
    effect(() => { if (tool.value === "station") { const c = this.map.getCenter(); void loadPlacesAt(c.lng, c.lat); void loadRealStationsAt(c.lng, c.lat); warmStreetNames(this.map); } });
    // T-2430: the trips near are asked for when the tool is picked and when the demand shown changes,
    // so the figure is in hand when the pointer gets to the track
    effect(() => { demandView.value; activeCity.value; worldFrameEpoch.value; if (tool.value === "station") askLegs(this); });
    map.on("idle", () => {
      if (tool.peek() !== "station") return;
      warmStreetNames(map);
      const c = map.getCenter();
      void loadRealStationsAt(c.lng, c.lat);
    });
    // T-4208: the camera moving under a still pointer (the keys, the zoom buttons, a fly to) left the ghost's end at the
    // pointer's old place on the map, far off once zoomed: it is taken again at the pointer's screen point. A drag of the
    // map moves the pointer too, so its screen point is kept up while a button is held (no hover then).
    addEventListener("mousemove", (e) => {
      if (!e.buttons || !this.pointer) return;
      const r = map.getContainer().getBoundingClientRect();
      [this.pointer.px, this.pointer.py] = [e.clientX - r.left, e.clientY - r.top];
    }, { capture: true, passive: true });
    map.on("move", () => this.cameraMoved());
    // T-1778: the snap ring and ghost are in local units, stale once the draw anchor moves: taken again where the pointer rests
    worldFrameEpoch.subscribe(() => {
      const p = this.pointer;
      if (p && (this.hover || this.hoverBeside) && !touchInput.peek()) this.move({ point: { x: p.px, y: p.py }, lngLat: { lng: p.lng, lat: p.lat } } as MapMouseEvent);
    });
    // T-3441: a right click on empty map is Escape (a run ends, then the tool); a right drag is the map's own
    let rightDown: { x: number; y: number } | null = null;
    map.getContainer().addEventListener("mousedown", (e) => { rightDown = e.button === 2 ? { x: e.clientX, y: e.clientY } : null; }, true);
    map.on("contextmenu", (e) => {
      if (tool.peek() !== "track" || touchInput.peek()) return;
      e.originalEvent.preventDefault();
      const d = rightDown;
      if (d && Math.hypot(e.originalEvent.clientX - d.x, e.originalEvent.clientY - d.y) > 4) return;
      if (this.start) this.reset();
      else this.exit();
    });
    effect(() => { buildLevel.value; autoRamp.value; highSpeed.value; tramTrack.value; quadTrack.value; singleTrack.value; designSpeed.value; flyoverSpeed.value; if (tool.peek() === "track" && this.start) this.real?.active ? this.real.again() : this.street?.active ? this.street.again() : this.requestPreview(); });
    // T-1929: the worker prices road crossings from the basemap's own zoom 14 tiles; the ghost is
    // asked again when more of them land
    let roadsSent = false;
    const sendRoadTiles = () => {
      if (!roadsSent && !perfOff("roads")) void roadTileTemplate(map).then((t) => {
        if (t && !roadsSent) client.post({ kind: "roadTiles", template: t });
        roadsSent ||= !!t;
      });
    };
    map.on("styledata", sendRoadTiles);
    sendRoadTiles();
    onRoadsLanded(() => { if (tool.peek() === "track" && this.start) this.real?.active ? this.real.roadsLanded() : this.requestPreview(); });
    // T-727d: the snap switch's routes along the real network
    this.real = new RealSnap(map, overlay, client, (r) => this.realReq(r), (a, req) => {
      if (a) return this.start && this.showPreview(a, req ? JSON.stringify(req) : "");
      this.lastPreview = null;
      this.clearPreview();
      if (!this.real.active) this.requestPreview();
      this.place();
    }, () => this.place(), () => this.freeLevel());
    effect(() => this.real.setOn(snapMode.value));
    // T-2927: the street snap for tram track, its own route and preview as the real snap's
    this.street = new StreetSnap(map, overlay, client, (r) => this.streetReq(r), (a, req) => {
      if (a) return this.start && this.showPreview(a, req ? JSON.stringify(req) : "");
      // T-2996: an ordinary segment's ghost asked for replaces the street route's when it comes
      if (this.snapped || !this.requestPreview()) {
        this.lastPreview = null;
        this.clearPreview();
      }
      this.place();
    }, () => this.place(), () => this.freeLevel(), () => ({ design: designSpeed.peek(), hsr: highSpeed.peek(), quad: quadTrack.peek() }));
    effect(() => { tool.value; tramTrack.value; snapStreets.value; snapRoads.value; snapRealTram.value; this.streetSwitch(); });
    effect(() => { besideOn.value; this.place(); });
    // T-1260: the touch control bar and the handle that drags the point being previewed
    this.touchHost = document.createElement("div");
    this.touchHost.className = "touch-bar-host";
    this.touchHost.style.display = "none";
    this.handle = document.createElement("div");
    this.handle.className = "touch-handle";
    this.handle.style.display = "none";
    for (const el of [this.touchHost, this.handle]) {
      for (const k of ["pointerdown", "mousedown", "touchstart", "click", "dblclick", "contextmenu"]) el.addEventListener(k, (e) => e.stopPropagation());
    }
    // the bar over everything on the map column (the demand buttons, the search glass)
    document.body.appendChild(this.touchHost);
    map.getContainer().appendChild(this.handle);
    this.handle.addEventListener("pointerdown", (e) => dragHandle(this, e));
    Object.assign(touchActions, { confirm: () => { if (!waitClick(this, "confirm", null)) this.carrying(touchConfirm(this)); }, back: () => touchBack(this), stop: () => touchStop(this), drawn: () => placeTouch(this) });
    startClickWait(this);
    setRunBreak(() => this.dropRun());
    effect(() => {
      tool.value;
      touchInput.value;
      this.touchAt = null;
      this.ghostSpent = false;
      if (tool.value !== "delete") clearDelHover(this);
      this.place();
    });
    // T-1039: a run ends with the track tool however it is left (a game loaded sets the tool directly)
    effect(() => { if (tool.value !== "track" && this.start) setTimeout(() => tool.peek() !== "track" && this.start && this.reset()); });
    effect(() => {
      const active = tool.value === "line" ? lineDraft.value.line : null;
      const w = world.value;
      if (active !== this.editingLine) {
        this.editingLine = active;
        // T-1899: a line is new until its first stop editing ends, and its clicks still out are sent (T-3731)
        const was = newLine.peek()?.line;
        if (was !== undefined && was !== active) void afterStopEdits().then(() => {
          if (newLine.peek()?.line === was && this.editingLine !== was) newLine.value = null;
        });
      }
      if (active === null || !w) return;
      const line = w.save.lines.find((l) => Number(l.id) === active);
      const bus = line && w.lineStats[line.id]?.bus;
      // T-2891: a bus line's stops are edited only while it is by hand
      if (!line || (bus && !bus.hand)) {
        this.exitNow();
        return;
      }
      const stops = planStopsOf(line).map(Number);
      if (!sameStops(stops, lineDraft.peek().stops)) lineDraft.value = { line: active, stops };
    });
    // T-3731: stop editing left with clicks still being sent: the tag says they are on their way
    effect(() => {
      const waiting = stopsQueued.value > 0 && !(tool.value === "line" && lineDraft.value.line !== null);
      if (waiting && !this.savingStops) this.savingStops = showBusy(tr("tools.savingStops"));
      else if (!waiting && this.savingStops) {
        hideBusy(this.savingStops);
        this.savingStops = 0;
      }
    });
  }

  get drawing() {
    return this.start !== null;
  }

  /** The tool's own tag is by the pointer (the track hover moves above it, T-526). */
  get tagShown() {
    return this.tag.style.display !== "none";
  }

  // ---------------------------------------------------------------- snapping

  /** T-1887: Ctrl (Cmd on a Mac) held: the track tool snaps to no track, node or beside spot. */
  noSnap = false;

  /** Follow Ctrl or Cmd from a pointer or key event; a change redraws the ghost where the pointer is. */
  noSnapFrom(e: { ctrlKey: boolean; metaKey: boolean }) {
    const on = e.ctrlKey || e.metaKey;
    if (on === this.noSnap) return;
    this.noSnap = on;
    this.streetSwitch();
    const p = this.pointer;
    if (tool.peek() === "line" && this.busStops?.active && p) this.busStops.hover(p.px, p.py, on); // T-4069
    if (tool.peek() === "track" && p) this.move({ point: { x: p.px, y: p.py }, lngLat: { lng: p.lng, lat: p.lat }, originalEvent: e } as unknown as MapMouseEvent);
  }

  /** T-2927: Ctrl is the free click only for rail; with Tram it flips the street snap */
  get freeClick() {
    return this.noSnap && !tramTrack.peek();
  }

  /** T-2927: the street snap on while the track tool holds Tram, its switch flipped while Ctrl is held. */
  streetSwitch() {
    if (!this.street) return;
    // T-3129: rail follows the roads with Snap to roads on and Ctrl not held
    // T-3069: with tram track snapping to the real network Ctrl does not bring the streets in
    const tram = tramTrack.peek(), on = tool.peek() === "track" && (tram ? !snapRealTram.peek() && snapStreets.peek() !== this.noSnap : snapRoads.peek() && !this.noSnap);
    const profile = tram ? "tram" : "rail";
    if (on === this.street.isOn && (!on || profile === this.street.snapProfile)) return;
    this.street.setOn(on, profile);
    if (on && this.start) this.street.begin([this.start.lng, this.start.lat]);
    if (!on && this.start) this.requestPreview();
  }

  screen(lx: number, ly: number) {
    return this.overlay.toScreen(lx, ly);
  }

  /** T-4214: the network the tools pick on: the drawn one with the edits patched in since (map/clickMap.ts) */
  get pickNet() {
    return clickNet(this.overlay.renderer.net);
  }

  /** What is under the pointer for building: a node, a point on track, or open ground. */
  snap(px: number, py: number, lngLat: { lng: number; lat: number }, nodes = true): Snap {
    const w = clickWorld();
    const n = this.pickNet;
    const [fx, fy] = toLocal(lngLat.lng, lngLat.lat);
    const station = tool.peek() === "station";
    if (station) this.alts = [];
    if (!w || !n || (this.freeClick && tool.peek() === "track")) return { kind: "free", lx: fx, ly: fy };
    const edgePx = station ? SNAP_EDGE_PX : TRACK_EDGE_PX;
    // T-3162: single track picked beside double track drawn as two tracks: each of the two is a branch point across
    // its whole drawn width, so the pair's outer edge is the reach there
    const laneL = !station && tool.peek() === "track" && singleTrack.peek() ? laneState(this.map.getZoom()) : null;
    const laneReach = laneL && laneL.t > 0 ? Math.max(edgePx, (laneL.gap * (laneL.t + laneL.share)) / 2) : edgePx;
    const doubleIds = laneReach > edgePx ? this.doubleEnds(w).ids : null;
    let best = TRACK_NODE_PX * TRACK_NODE_PX, hit: Snap | null = null;
    // only what is near the pointer (T-724); everything with the map pitched
    const at = pointerLocal(this.overlay, px, py);
    // T-1870: a station goes on the track at the pointer, never pulled to a node within the radius
    // (zoomed out that was a long stretch of track); `roomSnap` moves it where the platform needs room
    nodes &&= !station;
    // T-3889: tram track joins train track, so every track is a snap target whichever is picked
    const nearNodes = at && nodes ? nodesNear(w, at[0], at[1], TRACK_NODE_PX * at[2]) : null;
    const nearEdges = at ? edgesNear(n, at[0], at[1], laneReach * at[2]) : null;
    for (const node of nodes ? nearNodes ?? w.nodes.values() : []) {
      if (node.bus || node.port || node.brtStation) continue; // a bus stop, a port (T-2194) or a BRT station (T-3378) is no track to build on (T-1464)
      if (hidden.levels && levelHidden(node.level)) continue; // T-3560
      const [lx, ly] = toLocal(node.lng, node.lat);
      const [x, y] = this.screen(lx, ly);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        best = d;
        hit = { kind: "node", node: node.id, lx, ly };
      }
    }
    // T-3190: zoomed in with single track picked, a double's loose end is picked by either of its two tracks: the
    // pointer over a track within a node radius of the end (along it) carries on that track, not a branch off it
    if (!hit && doubleIds && nodes) {
      for (const [node, { edge, fromA }] of this.doubleEnds(w).loose) {
        const pts = n.edgePts.get(edge), nd = w.nodes.get(node);
        if (!pts || !nd || pts.length < 6 || (hidden.levels && levelHidden(nd.level))) continue;
        const [lx, ly] = toLocal(nd.lng, nd.lat), [x, y] = this.screen(lx, ly);
        const k = fromA ? 3 : pts.length - 6;
        const [x1, y1] = this.screen(pts[k], pts[k + 1]);
        const l = Math.hypot(x1 - x, y1 - y);
        if (!(l > 0)) continue;
        const along = ((px - x) * (x1 - x) + (py - y) * (y1 - y)) / l, across = Math.abs((px - x) * (y1 - y) - (py - y) * (x1 - x)) / l;
        if (Math.abs(along) > TRACK_NODE_PX || across > laneReach) continue;
        const d = along * along + across * across;
        if (d < best || !hit) {
          best = d;
          hit = { kind: "node", node, lx, ly };
        }
      }
    }
    if (hit) return this.withJoin(hit, fx, fy);
    // T-3560: track wholly on a switched-off level is no snap target
    const offEdges = hidden.levels ? new Set(w.edges.filter((e) => levelHidden(e.levelMin) && levelHidden(e.levelMax)).map((e) => e.id)) : null;
    // each edge's nearest point within reach, px squared
    const near: { d: number; hit: Snap }[] = [];
    const tramOnly = station && !this.autoProbe ? stationOnTram.peek() : null, tramOf = station ? this.tramEdgeSet(w) : null, buswayOf = station ? this.buswayEdgeSet(w) : null;
    for (const { edge, pts, ks } of nearEdges ?? [...n.edgePts].map(([edge, pts]) => ({ edge, pts, ks: null }))) {
      if (offEdges?.has(edge)) continue;
      // T-4010: a train station goes on train track only and a tram stop on tram track only
      if (tramOnly !== null && tramOf!.has(edge) !== tramOnly) continue;
      // a busway holds no station: the pointer on one is a BRT station beside it (Auto picks the bus kind)
      if (buswayOf?.has(edge)) continue;
      const er = doubleIds?.has(edge) ? laneReach : edgePx;
      let eb = er * er, eh: Snap | null = null;
      for (let j = 0, k = ks ? ks[0] : 0; ks ? j < ks.length : k + 5 < pts.length; j++, k = ks ? ks[j] : k + 3) {
        const [x0, y0] = this.screen(pts[k], pts[k + 1]);
        const [x1, y1] = this.screen(pts[k + 3], pts[k + 4]);
        const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        const d = (x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2;
        if (d < eb) {
          eb = d;
          eh = { kind: "edge", edge, lx: pts[k] + (pts[k + 3] - pts[k]) * t, ly: pts[k + 1] + (pts[k + 4] - pts[k + 1]) * t };
        }
      }
      if (eh) near.push({ d: Math.sqrt(eb), hit: eh });
    }
    if (!near.length) return { kind: "free", lx: fx, ly: fy };
    near.sort((a, b) => a.d - b.d);
    if (!station) return this.withJoin(near[0].hit, fx, fy);
    // T-469, T-671: what a click can place, in the order it is tried: the tracks within 4 px of the
    // nearest in orderOverlapping's order, then the rest by distance
    const close = near.filter((c) => c.d <= near[0].d + OVERLAP_PX);
    const spots = close.map((c) => { const h = c.hit as Extract<Snap, { kind: "edge" }>; return { edge: h.edge, frac: fractionAlong(n.edgePts.get(h.edge)!, h.lx, h.ly) }; });
    const order = close.length > 1 ? orderOverlapping(spots, this.platFor(close[0].hit as Spot), w.edges, w.nodes, this.lastStationTrack(), layout(this, w)) : close.map((_, i) => i);
    this.alts = [...order.map((i) => close[i].hit as Spot), ...near.slice(close.length, 5).map((c) => c.hit as Spot)];
    return this.alts[0];
  }

  /** `hit` with its `join` for the pointer at (px, py), local units */
  withJoin(hit: Snap, px: number, py: number): Snap {
    if (hit.kind === "free") return hit;
    const j = this.joinPoint(hit, px, py);
    if (!j) return hit;
    return { ...hit, ...(j.join ? { join: j.join } : {}), ...(j.dot ? { dot: j.dot } : {}), ...(j.merge && hit.kind === "node" ? { merge: true as const } : {}) };
  }

  /**
   * T-3137: with single track picked and double track at `hit`, which track of it the pointer (px, py, local units)
   * is on: `join`, the point on that track (on its centre line's normal there, as far off it as the tracks are
   * drawn, 2 m where they are not), and `dot`, where the dot sits, screen px off the hit (the track's drawn offset, at
   * least DOT_MIN_PX, so zoomed out it still reads) (T-3204). Zoomed in the pointer's side picks the track; within
   * DEAD_PX of the centre line zoomed out, and between the tracks at a double's loose end, it takes neither: `merge`
   * at a loose end (both tracks into the one).
   */
  joinPoint(hit: Spot, px: number, py: number): { join?: [number, number]; dot?: [number, number]; merge?: boolean } | undefined {
    if (tool.peek() !== "track" || !singleTrack.peek()) return undefined;
    const lanes = laneState(this.map.getZoom()), t = lanes.t, w = clickWorld(), n = this.pickNet;
    if (!w || !n) return undefined;
    const doubles = this.doubleEnds(w);
    const edges = hit.kind === "edge" ? (doubles.ids.has(hit.edge) ? [hit.edge] : []) : doubles.byNode.get(hit.node) ?? [];
    // the double's centre line at the hit: per edge its segment through it, of those the one nearest the pointer
    let best: { d: number; x: number; y: number; ux: number; uy: number } | null = null;
    for (const e of edges) {
      const pts = n.edgePts.get(e);
      if (!pts) continue;
      let seg: { dh: number; d: number; x: number; y: number; ux: number; uy: number } | null = null;
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const ux = pts[k + 3] - pts[k], uy = pts[k + 4] - pts[k + 1], l2 = ux * ux + uy * uy;
        if (!(l2 > 0)) continue;
        const f = Math.max(0, Math.min(1, ((hit.lx - pts[k]) * ux + (hit.ly - pts[k + 1]) * uy) / l2));
        const dh = Math.hypot(pts[k] + ux * f - hit.lx, pts[k + 1] + uy * f - hit.ly);
        const g = Math.max(0, Math.min(1, ((px - pts[k]) * ux + (py - pts[k + 1]) * uy) / l2));
        const d = Math.hypot(pts[k] + ux * g - px, pts[k + 1] + uy * g - py);
        if (!seg || dh < seg.dh - 1e-6 || (dh <= seg.dh + 1e-6 && d < seg.d)) seg = { dh, d, x: pts[k], y: pts[k + 1], ux, uy };
      }
      if (seg && (!best || seg.d < best.d)) best = seg;
    }
    if (!best) return undefined;
    const ul = Math.hypot(best.ux, best.uy), side = Math.sign(best.ux * (py - best.y) - best.uy * (px - best.x));
    const loose = hit.kind === "node" && doubles.loose.has(hit.node);
    // the pointer's distance from the centre line, screen px
    const [fx, fy] = this.screen(hit.lx, hit.ly), [ex, ey] = this.screen(hit.lx + 1, hit.ly);
    const across = (Math.abs(best.ux * (py - best.y) - best.uy * (px - best.x)) / ul) * Math.hypot(ex - fx, ey - fy);
    const laneOff = (lanes.gap * t) / 2, half = (lanes.share * lanes.gap) / 2;
    const pick = !side ? false : t > 0 && !loose ? true : across > Math.max(DEAD_PX, laneOff - half);
    if (!pick) return loose ? { merge: true } : undefined;
    const off = (LANE_HALF_M * (t > 0 ? t : 1) * kAtY(hit.ly) * side) / ul;
    const near = this.screen(hit.lx - best.uy * (0.01 * side) / ul, hit.ly + best.ux * (0.01 * side) / ul), nl = Math.hypot(near[0] - fx, near[1] - fy) || 1;
    const px0 = Math.max(laneOff, DOT_MIN_PX);
    return { join: [hit.lx - best.uy * off, hit.ly + best.ux * off], dot: [((near[0] - fx) / nl) * px0, ((near[1] - fy) / nl) * px0] };
  }

  /** T-3171: whether the track at (x, y) is single track with no double track there (a double may narrow to it) */
  onlySingle(n: NonNullable<Tools["overlay"]["renderer"]["net"]>, w: NonNullable<ReturnType<typeof world.peek>>, x: number, y: number): boolean {
    const { ids: doubles, singles } = this.doubleEnds(w);
    let single = false;
    const reach = 3 * kAtY(y);
    for (const { edge, pts, ks } of edgesNear(n, x, y, reach) ?? []) {
      // the candidates are the cells' segments: only those within reach
      const near = (ks ?? []).some((k) => {
        const dx = pts[k + 3] - pts[k], dy = pts[k + 4] - pts[k + 1], l2 = dx * dx + dy * dy, f = l2 ? Math.max(0, Math.min(1, ((x - pts[k]) * dx + (y - pts[k + 1]) * dy) / l2)) : 0;
        return Math.hypot(pts[k] + dx * f - x, pts[k + 1] + dy * f - y) <= reach;
      });
      if (!near) continue;
      if (doubles.has(edge)) return false;
      if (singles.has(edge)) single = true;
    }
    return single;
  }

  doubleIndex: { edges: unknown; ids: Set<number>; byNode: Map<number, number[]>; loose: Map<number, { edge: number; fromA: boolean }>; singles: Set<number> } | null = null;
  /** the world's double track (crossovers aside), and its pieces by the nodes at their ends, kept while its edges are the same */
  doubleEnds(w: NonNullable<ReturnType<typeof world.peek>>) {
    if (this.doubleIndex?.edges !== w.edges) {
      const ids = new Set<number>(), singles = new Set<number>(), byNode = new Map<number, number[]>(), ends = new Map<number, number>();
      for (const e of w.edges) {
        for (const k of [e.a, e.b]) ends.set(k, (ends.get(k) ?? 0) + 1);
        if (e.tracks === 1) singles.add(e.id);
        if (e.tracks !== 2 || e.crossover) continue;
        ids.add(e.id);
        for (const k of [e.a, e.b]) byNode.set(k, [...(byNode.get(k) ?? []), e.id]);
      }
      // T-3190: nodes where a double piece just ends
      const loose = new Map<number, { edge: number; fromA: boolean }>();
      for (const [node, es] of byNode) if (es.length === 1 && ends.get(node) === 1) loose.set(node, { edge: es[0], fromA: w.edges.find((e) => e.id === es[0])!.a === node });
      this.doubleIndex = { edges: w.edges, ids, byNode, loose, singles };
    }
    return this.doubleIndex;
  }

  /** T-4105: set while Auto's probe snaps, so the nearest track of either kind answers */
  autoProbe = false;
  autoPick(at?: { px: number; py: number; lng: number; lat: number }) {
    autoPick(this, at);
  }

  tramEdges: { edges: unknown; set: Set<number> } | null = null;
  /** the ids of the world's tram edges (T-2926), kept while its edges are the same */
  tramEdgeSet(w: NonNullable<ReturnType<typeof world.peek>>): Set<number> {
    if (this.tramEdges?.edges !== w.edges) this.tramEdges = { edges: w.edges, set: new Set(w.edges.filter((e) => e.kind === 2 || e.kind === 3).map((e) => e.id)) };
    return this.tramEdges.set;
  }

  buswayEdges: { edges: unknown; set: Set<number> } | null = null;
  buswayEdgeSet(w: NonNullable<ReturnType<typeof world.peek>>): Set<number> {
    if (this.buswayEdges?.edges !== w.edges) this.buswayEdges = { edges: w.edges, set: new Set(w.edges.filter((e) => e.kind === 4).map((e) => e.id)) };
    return this.buswayEdges.set;
  }

  /** whether a station spot is on tram track (T-2926) */
  tramSpot(s: Snap | null | undefined): boolean {
    const w = clickWorld();
    if (!s || !w) return false;
    if (s.kind === "free") return false;
    return s.kind === "node" ? [2, 3].includes(w.nodes.get(s.node)?.kind ?? 0) : this.tramEdgeSet(w).has(s.edge);
  }

  /** the platform length a new station gets at a spot: tram stops have their own (T-2926) */
  platFor(_s?: Snap | null): number {
    return stationOnTram.peek() ? buildTramPlatform.peek() : buildPlatform.peek();
  }

  /** T-1685: with Snap beside on, open ground within BESIDE_PX of track: the nearest piece, and the
   * spot BESIDE_SPACING_M from it on the pointer's side (local units) */
  besideAt(px: number, py: number): { edge: number; lx: number; ly: number } | null {
    const n = this.pickNet;
    const at = pointerLocal(this.overlay, px, py);
    if (!besideOn.peek() || !n || !at || this.freeClick) return null;
    let best = BESIDE_PX * BESIDE_PX, hit: { edge: number; lx: number; ly: number } | null = null;
    const w = clickWorld(), tramOf = w ? this.tramEdgeSet(w) : null, fam = tramTrack.peek();
    for (const { edge, pts, ks } of edgesNear(n, at[0], at[1], BESIDE_PX * at[2]) ?? []) {
      if (tramOf && tramOf.has(edge) !== fam) continue;
      for (let j = 0, k = ks ? ks[0] : 0; ks ? j < ks.length : k + 5 < pts.length; j++, k = ks ? ks[j] : k + 3) {
        const [x0, y0] = this.screen(pts[k], pts[k + 1]);
        const [x1, y1] = this.screen(pts[k + 3], pts[k + 4]);
        const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        const d = (x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2;
        if (d >= best) continue;
        best = d;
        const ux = pts[k + 3] - pts[k], uy = pts[k + 4] - pts[k + 1], ul = Math.hypot(ux, uy) || 1;
        const fx = pts[k] + ux * t, fy = pts[k + 1] + uy * t;
        const side = Math.sign(ux * (at[1] - fy) - uy * (at[0] - fx)) || 1;
        hit = { edge, lx: fx - (uy / ul) * BESIDE_SPACING_M * side, ly: fy + (ux / ul) * BESIDE_SPACING_M * side };
      }
    }
    return hit;
  }

  /** what the station tool would try at the pointer, in order (T-671) */
  alts: Spot[] = [];
  stationTrack: { at: [number, number]; version: number; edges: Set<number> } | null = null;

  /** The edges of the track the last station placed this session is on (T-469): from its node
   * along the track both ways, through plain joints and stations, up to junctions. */
  lastStationTrack(): Set<number> {
    const out = new Set<number>();
    const w = clickWorld();
    const at = this.lastPlaced;
    if (!w || !at) return out;
    // kept per placement and network version: this runs per pointer move
    const kept = this.stationTrack;
    if (kept?.at === at && kept.version === w.version) return kept.edges;
    this.stationTrack = { at, version: w.version, edges: out };
    let start: number | null = null, bd = 2; // metres
    for (const n of w.nodes.values()) {
      if (n.platform <= 0) continue;
      const d = metresBetween(at[0], at[1], n.lng, n.lat);
      if (d < bd) bd = d, start = n.id;
    }
    if (start === null) return out;
    const { byNode, byId: edges } = layout(this, w); // kept per network version: this runs per pointer move
    const todo = [start];
    const seen = new Set(todo);
    while (todo.length && out.size < 2000) {
      const k = todo.pop()!;
      for (const id of byNode.get(k) ?? []) {
        if (out.has(id)) continue;
        out.add(id);
        const e = edges.get(id)!;
        const o = e.a === k ? e.b : e.a;
        if (!seen.has(o) && (w.nodes.get(o)?.ports ?? 0) <= 2) seen.add(o), todo.push(o);
      }
    }
    return out;
  }

  /** T-3332: with the station bar's one-track icon on, the pointer (sim x, y): the platform goes on the
   * track nearer it */
  oneAt(): { one?: { x: number; y: number } } {
    const p = this.pointer;
    return stationOneTrack.peek() && p ? { one: lngLatToPack(p.lng, p.lat) } : {};
  }

  /** T-3332: which side of a platform shape's axis the pointer is on for a one-track ghost on double
   * track (`oneSideShape`'s side), 0 for a platform on both tracks or on single track */
  oneSideAt(lx: number, ly: number, sh: PlatformShape, double: boolean): number {
    const p = this.pointer;
    if (!stationOneTrack.peek() || !double || !p) return 0;
    const [px, py] = toLocal(p.lng, p.lat);
    return Math.sign(-sh.ay * (px - lx) + sh.ax * (py - ly)) || 1;
  }

  pack(lx: number, ly: number) {
    const [lng, lat] = toLngLat(lx, ly);
    return lngLatToPack(lng, lat);
  }

  endOf(s: Snap): RouteEnd {
    if (s.kind === "node") return { kind: "node", node: s.node };
    const p = this.pack(s.lx, s.ly);
    if (s.kind === "edge") return { kind: "edge", edge: s.edge, x: p.x, y: p.y };
    return { kind: "free", x: p.x, y: p.y, level: this.freeLevel() };
  }

  /** The elevation of a segment's end on open ground: the one picked. */
  freeLevel(): Level {
    return buildLevel.peek();
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "select") return;
    if (t === "delete") return deleteHover(this, e.point.x, e.point.y);
    if (t === "bus") return this.bus.move(e);
    if (t === "ferry") return this.ferry.move(e);
    if (t === "building") return this.building.move(e);
    if (e.originalEvent) this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
    if (t === "line") (this.busStops?.active ? this.busStops.hover(e.point.x, e.point.y, this.noSnap) : this.stops?.hover(e.point.x, e.point.y)); // T-233: the toggle markers' tag
    this.streetSwitch();
    this.ghostSpent = false;
    this.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
    if (t === "station") this.autoPick();
    this.hover = t === "line" ? null : this.snap(e.point.x, e.point.y, e.lngLat);
    if (t === "track" && this.hover) this.hover = this.straight(this.hover);
    if (t === "track") this.streetMove(e.point.x, e.point.y);
    // T-1721: from any start, and before the first click for the ring
    this.hoverBeside = t === "track" && this.hover?.kind === "free" && !this.snapped && !this.straightOn() ? this.besideAt(e.point.x, e.point.y) : null;
    if (t === "track") this.real.move(e.point.x, e.point.y);
    if (t === "track" && this.held.length) this.releaseNear(e.point.x, e.point.y);
    if (t === "track" && this.clicked) this.drawOnward();
    else if (t === "track" && this.start) {
      this.requestPreview();
      // T-4136: the ghost follows the pointer at once, its checked shape after
      if (this.lastPreview && this.previewKey !== this.wantKey) this.drawGhost();
    }
    if (t === "station") nameAhead(this, e.point.x, e.point.y);
    this.place();
  }

  /** A click with a tool: carried out now, or once the map holds every edit sent before it (T-4214). */
  click(e: MapMouseEvent) {
    if (!waitClick(this, "click", e)) this.clickNow(e);
  }

  clickNow(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") {
      this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
      this.streetSwitch();
      this.trackClick(e);
    } else if (t === "station") {
      this.autoPick({ px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat });
      this.carrying(stationClick(this, e));
    }
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") this.carrying(deleteClick(this, e));
    else if (t === "bus") this.bus.click(e);
    else if (t === "ferry") this.ferry.click(e);
    else if (t === "building") this.building.click(e);
  }

  // ---------------------------------------------------------------- touch (T-1260)

  /** A tap with a tool on a touch screen: a preview where a click would act, which a second tap
   * on the same spot or the bar's confirm carries out. The first point of a run of track is set at
   * once, as a click sets it; so is a tap on open ground with the station tool (it is refused). */
  tap(e: MapMouseEvent) {
    if (!waitClick(this, "tap", e)) this.tapNow(e);
  }

  tapNow(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "line") return this.clickNow(e);
    if (t === "bus") return this.bus.tap(e);
    if (t === "ferry") return this.ferry.tap(e);
    if (t === "building") return this.building.tap(e);
    if (this.touchAt && nearTouch(this, e.point.x, e.point.y)) return this.carrying(touchConfirm(this));
    const ll = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    if (t === "track") {
      if (this.placing) return;
      if (!this.start) {
        this.move(e);
        this.trackClick(e);
        return;
      }
      // the start tapped with nothing waiting ends the run, as a click on it does (T-707)
      const [sx, sy] = this.screen(...toLocal(this.start.lng, this.start.lat));
      if (!this.touchAt && Math.hypot(sx - e.point.x, sy - e.point.y) < SAME_TAP_PX) return this.reset();
      this.touchAt = ll;
      this.move(e);
    } else if (t === "station") {
      this.autoPick({ px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat });
      if (stationKind.peek() === "bus" || stationKind.peek() === "port" || freeStarted() || this.snap(e.point.x, e.point.y, e.lngLat).kind === "free") {
        this.touchAt = null;
        this.move(e);
        this.carrying(stationClick(this, e));
        return;
      }
      this.touchAt = ll;
      this.move(e);
    } else if (t === "delete") {
      this.touchAt = ll;
      this.move(e);
    }
  }

  /** T-4214: a click is being carried out until `p` settles (its edit sent, or none): the clicks after it wait. */
  carrying(p: Promise<unknown> | undefined) {
    if (!p) return;
    this.clicksOut++;
    void p.finally(() => {
      this.clicksOut--;
      tryClicks(this);
    });
  }

  /** Keys while a tool is on; true if used. */
  key(e: KeyboardEvent): boolean {
    const t = tool.peek();
    if (t === "select") return false;
    if (t === "bus") return this.bus.key(e);
    if (t === "ferry") return this.ferry.key(e);
    if (t === "building") return this.building.key(e);
    if (e.key === "Escape") {
      dropClicks(this);
      if (t === "track" && this.start) this.reset();
      else if (t === "station" && freeStarted()) {
        freeCancel(this);
        this.place();
      }
      else this.exit();
      return true;
    }
    if (t === "track" && this.start) {
      if (e.key === "Backspace" || e.key === "Delete") {
        void this.takeBackSegment();
        return true;
      }
      if (e.key === "Enter") {
        // (after the clicks still waiting, whose segments belong to the run it ends, T-4311)
        inClickOrder(() => this.reset());
        return true;
      }
    }
    if (t === "line" && e.key === "Enter") {
      this.exit();
      return true;
    }
    return false;
  }

  /** Leave the current tool (back to selecting), in its turn after map clicks still waiting (T-4214). */
  exit() {
    inClickOrder(() => this.exitNow());
  }

  exitNow() {
    const d = lineDraft.peek();
    if (tool.peek() === "line" && d.line !== null) selection.value = { kind: "line", line: String(d.line) };
    this.reset();
    lineDraft.value = { line: null, stops: [] };
    ask.value = null;
    tool.value = "select";
  }

  /** Drop the route being drawn (and what the delete tool points at). */
  reset() {
    // T-1039: a ghost asked for this run and not yet worked out is not worked out
    if (this.start) this.client.dropPreview();
    // T-2404: the run's segments held for the pointer's rest go now
    if (this.held.length) this.client.placedNow();
    this.held = [];
    this.flying = [];
    this.start = null;
    this.clicked = null;
    this.chain = [];
    this.delHover = null;
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.real.reset();
    this.street.reset();
    this.touchAt = null;
    this.clearPreview();
    this.place();
  }

  /** A node was dragged (T-734): a route drawn from it starts at its old place, so it is dropped. */
  nodeMoved(node: number) {
    const at = (p: Placed) => endNode(p.end) === node;
    if (this.start && (at(this.start) || this.chain.some(at))) this.reset();
    else if (this.start) this.requestPreview();
  }

  // ---------------------------------------------------------------- track

  trackClick(e: MapMouseEvent) {
    // T-445: a double click ends a run; its first click placed the last segment
    if (e.originalEvent.detail >= 2 && (this.placing || this.chain.length)) {
      if (this.placing) this.endRun = true;
      else this.reset();
      return;
    }
    let s = this.snap(e.point.x, e.point.y, e.lngLat);
    if (!this.start) {
      // T-727d: on open ground near a real line the start goes on the line (the ring)
      const ring = snapping.peek() ? this.real.ring(e.point.x, e.point.y) : null;
      if (ring && s.kind === "free") {
        const [lx, ly] = toLocal(ring[0], ring[1]);
        s = { kind: "free", lx, ly };
      }
      // T-2927: with the street snap on, open ground near a street starts on it
      if (this.street.isOn && s.kind === "free") {
        const [lx, ly] = toLocal(...this.street.landing(e.lngLat.lng, e.lngLat.lat));
        s = { kind: "free", lx, ly };
      }
      // T-1685: open ground beside track starts on the line beside it
      const by = !ring && s.kind === "free" && !this.street.isOn ? this.besideAt(e.point.x, e.point.y) : null;
      if (by) s = { kind: "free", lx: by.lx, ly: by.ly };
      this.start = placedAt(this.endOf(s), s.lx, s.ly, ring);
      this.run++;
      this.start.beside = by?.edge ?? null;
      if (s.kind !== "free" && s.join) this.start.join = s.join;
      this.real.begin(ring);
      this.street.begin([this.start.lng, this.start.lat]);
      this.requestPreview();
      this.place();
      return;
    }
    void this.segmentClick(e, this.straight(s));
  }

  /** Place the segment from the start to this click; with Shift held or keepPlacing on, the next
   * starts at its end. T-4214 stage D: the next one starts there at once, named to the worker as the end of this edit
   * (`endOf`, workers/clockRunEnds.ts), so a run's segments go one per click with no wait for the answer before. */
  async segmentClick(e: MapMouseEvent, s: Snap) {
    const from = this.start;
    if (!from || this.placing) return;
    // T-707: a click on the start ends the run, as a double click does, or drops a start with no
    // segment yet; a segment can never end where it starts, so a loop's end is never this point
    const [sx, sy] = this.screen(...toLocal(from.lng, from.lat)), se = this.endOf(s);
    if (Math.hypot(sx - e.point.x, sy - e.point.y) < 6 || (se.kind === "node" && se.node === endNode(from.end))) {
      this.reset();
      return;
    }
    // T-1811: a segment that joins track (a node, or mid-piece of track or blueprint) ends the run
    // under keep placing; Shift held still carries on from the join
    const chain = e.originalEvent.shiftKey || (keepPlacing.peek() && s.kind === "free");
    const run = this.run;
    this.placing = true;
    this.waitFrom(true);
    this.endRun = false;
    let rc: RealClick, sc: StreetClick = null, p: PreviewAnswer | null = null, sent: Promise<Awaited<ReturnType<typeof edit>>> | null = null, by: { edge: number } | null = null, mark: Ahead | null = null;
    let ow: NonNullable<Tools["clicked"]>["onward"] = null, exact = false, pts: Float32Array | null = null, seq = -1;
    /** the run goes on from this segment's end before its answer (stage D); `hold`: the run waits for the answer as before
     * (a segment sent on a ghost already refused, which the worker will most likely refuse too) */
    let onward = false, hold = false, mapped = false;
    // T-4214 stage E: from its word on, the segment is drawn as the tools' map has it (its patch's new edges, by id), so an
    // edit after it (a delete, a station cutting it) draws on it at once
    const fromMap = (laid: number): Show | null => {
      const f = laid >= 0 ? patchFresh(laid) : null;
      if (!f?.size) return null;
      mapped = true;
      return { track: [], pieces: new Map(f) };
    };
    // T-1001: whatever happens to this click, later clicks are not ignored
    try {
      // T-727d: near real track at both clicks, the real route (waited for if still being found)
      rc = await this.real.click(e.point.x, e.point.y);
      if (rc?.kind === "refuse" || this.start !== from) {
        if (rc?.kind === "refuse") say(tr("tools.noRealTrack"), true);
        return;
      }
      // T-2927: tram track with the street snap on follows the streets
      if (!rc && this.street.isOn) {
        sc = await this.street.click(...this.streetAt(e.point.x, e.point.y, s));
        if (sc?.kind === "refuse" || this.start !== from) {
          if (sc?.kind === "refuse") say(sc.text, true);
          return;
        }
      }
      if (!rc && !sc && this.tooFar(s)) {
        say(tooFarText(), true);
        return;
      }
      by = !rc && !sc && s.kind === "free" ? this.besideAt(e.point.x, e.point.y) : null;
      const req = rc ? this.realReq(rc.route, s) : sc ? this.streetReq(sc.route, s) : this.routeReq(this.endOf(s), by, s);
      // T-830: the quote of this very request; an answer for an earlier pointer position (a slow
      // preview landing late) would give another segment's length and cost
      p = this.previewKey === JSON.stringify(req) ? this.lastPreview : null;
      // T-1474: heard at the click, not at the worker's answer; not when the ghost already shows a refusal
      if (p?.ok !== false) playBuild("snap");
      // (where it lies as drawn, for a click on it before its answer: `flying`; a route with no ghost of its own, through its
      // route's points)
      const [ex, ey] = this.clickEnd(s, rc, sc), route = rc?.kind === "route" ? rc.route : sc?.kind === "route" ? sc.route : null;
      pts = p?.pts ?? (route ? null : this.startLocal(s)?.a.pts ?? this.shownPreview?.pts) ?? this.sentLine(from, route, ex, ey);
      // T-4208: shown placed at the click, the ghost going on from its end; a ghost already refused stays as it is
      if (p?.ok !== false && !rc && !sc && !this.snapped) {
        const stamp = this.clickAhead(p, s, chain, s.kind === "free" && !by);
        exact = this.stampExact;
        mark = stamp ? ahead({ track: [stamp] }, { atWord: fromMap }) : null;
        ow = this.clicked?.onward ?? null;
      } else if (p?.ok !== false && (rc || sc)) {
        // T-4214 stage D: a real or street route's ghost as its snap shows it, the run going on from its end at once
        const st = this.overlay.renderer.previewSrc;
        mark = st?.count ? ahead({ track: [{ ...st, marks: undefined, chevrons: undefined }] }, { atWord: fromMap }) : null;
      }
      // T-1407: answered before its state, so the next ghost is asked at once
      const go = () => asRun(run, () => edit({ op: "route", ...req }, false, true, true));
      const n0 = editsSentCount();
      sent = mark ? mark.send(go) : go();
      if (editsSentCount() > n0) seq = lastSentSeq();
      onward = chain && !this.endRun && seq >= 0 && p?.ok !== false;
      hold = chain && !onward && !this.endRun;
    } finally {
      this.placing = hold;
      this.clicked = null;
      this.place();
      // T-4214: a click waiting for this one goes after the rest of it (the run's new start)
      if (!hold) queueMicrotask(() => tryClicks(this));
    }
    const chainWas = this.chain.slice();
    const next = onward ? this.goOn(from, { kind: "endOf", seq }, s, rc, sc, ow, p, by, e) : null;
    // the run ends at this segment (T-445, T-1811): at once, and back as it was if the segment is refused
    if (!onward && !hold) this.reset();
    if (next && pts) this.flying.push({ seq, pts });
    // (the click is done once sent: a touch confirm waits only for this, stage D; a run waiting for the answer waits for it)
    const answered = async () => {
      let a: Awaited<ReturnType<typeof edit>>;
      try {
        a = await sent;
      } finally {
        // (a click waiting on this segment goes now: its patch, if it has one, came before this answer)
        this.flying = this.flying.filter((f) => f.seq !== seq);
        if (hold) this.placing = false;
        queueMicrotask(() => tryClicks(this));
      }
      if (a.ok && a.node !== undefined && seq >= 0) noteRunNode(seq, a.node);
      // T-4208: the stand-in kept at the click goes with a refusal. T-4210: otherwise it stays until the network holding the
      // segment is drawn (game/pending.ts takes it in that frame, never with an earlier edit's network); a click that beat
      // its answer, with nothing laid on the page, has it swapped for the worker's own edges now (the ghost as shown was bent)
      if (mark && !a.ok) mark.drop();
      if (mark && a.ok && a.early && !exact && !mapped && a.held?.length) {
        mark.set({ track: a.held.map((pts) => strokesFromPreview(pts, false, undefined, true, undefined, singleTrack.peek() ? 0 : 255, highSpeed.peek(), tramTrack.peek())) });
      }
      // T-1407: until the state lands, the ghost just placed stands in for the blueprint it made
      if (a.ok && a.early && this.run === run) {
        // T-2404: its state held while the pointer moves, its edges stand in where no ghost of it can
        // (the ghost as this very answer drew it, not bent to a pointer that moved on meanwhile)
        if (mark) { /* the stand-in kept at the click stays (above) */ }
        else if (p && p.pts.length >= 6) this.overlay.renderer.addPlaced(this.ghostStrokes(p));
        else for (const pts of a.held ?? []) this.overlay.renderer.addPlaced(strokesFromPreview(pts, false, undefined, true, undefined, singleTrack.peek() ? 0 : 255, highSpeed.peek(), tramTrack.peek()));
        this.overlay.invalidate();
        if (a.held) this.held.push(...a.held);
        if (a.node !== undefined) this.earlyNode = a.node;
      }
      const inRun = (st: Placed | null) => !!st && (this.start === st || this.chain.includes(st));
      if (!a.ok) {
        if (this.run !== run || tool.peek() !== "track") return;
        // T-4208: a refused click's run goes on from where it was; stage D: the segments sent on from it are refused with it
        // (the worker refuses a run end of a refused edit), so the run goes back to its start
        if (next && inRun(next)) {
          const i = this.chain.indexOf(from);
          if (i >= 0) this.chain.length = i;
          this.backTo(from);
        } else if (!next && !hold && !this.start) {
          this.chain = chainWas;
          this.backTo(from);
        } else if (mark && this.start === from) {
          this.requestPreview();
          this.drawGhost();
        }
        return;
      }
      say(p ? tr("tools.blueprintCost", { distance: distance(p.lengthM), cost: money(p.cost * 1e6) }) : tr("tools.blueprint"));
      if (next) {
        // (a route the worker took with no end node: the segments from it are refused, so the run ends here)
        if (a.node === undefined && inRun(next)) this.reset();
        return;
      }
      if (!hold || this.start !== from) return;
      if (this.endRun || a.node === undefined) {
        this.reset();
        return;
      }
      this.goOn(from, { kind: "node", node: a.node }, s, rc, sc, ow, p, by, e);
    };
    if (hold) return answered();
    void answered();
  }

  /** The run goes on from the end of the segment just placed from `from` (T-445), `end` naming it to the worker. */
  private goOn(from: Placed, end: RouteEnd, s: Snap, rc: RealClick, sc: StreetClick, ow: NonNullable<Tools["clicked"]>["onward"], p: PreviewAnswer | null, by: { edge: number } | null, e: MapMouseEvent): Placed {
    this.chain.push(from);
    const [ex, ey] = this.clickEnd(s, rc, sc);
    const st = placedAt(end, ex, ey, rc?.kind === "route" ? rc.route.last : snapping.peek() ? this.real.ring(e.point.x, e.point.y) : null);
    st.beside = by?.edge ?? null; // a run beside track carries on beside it
    // T-4210: the heading (and bare end) the ghost from the click took, so nothing jumps as the answer lands
    const eh = !ow?.dir && p ? endHeading(p.verts, p.pts, p.arc, toLngLat, toLocal) : null;
    st.dir = ow?.dir ?? (eh ? [eh.ux, eh.uy] : undefined);
    st.tangent = ow?.tangent;
    st.z = ow?.z;
    this.start = st;
    this.real.begin(st.ring);
    this.street.begin([st.lng, st.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.clearPreview();
    this.requestPreview();
    this.place();
    return st;
  }

  /** T-4214 stage D: the run drawn on from `st` again (a segment refused, and those sent on from it with it). */
  private backTo(st: Placed) {
    this.start = st;
    this.real.begin(st.ring);
    this.street.begin([st.lng, st.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.clearPreview();
    this.requestPreview();
    this.drawGhost();
    this.place();
  }

  /** Backspace while chaining: the last segment undone, drawing on from its start again. */
  async takeBackSegment() {
    // a segment still being placed is not on the undo stack yet, nor a click waiting or being carried out (T-4214)
    if (this.placing || this.waitingClicks.length || this.clicksOut) return;
    const prev = this.chain[this.chain.length - 1];
    if (!prev) {
      this.reset();
      return;
    }
    // (sent before `placing` is set: an undo waits while it is, game/world.ts)
    const sent = asRun(this.run, () => edit({ op: "undo" }, true));
    this.placing = true;
    const a = await sent.finally(() => {
      this.placing = false;
      queueMicrotask(() => tryClicks(this));
    });
    // T-4311: refused, nothing was taken back: the segment it meant was refused, and that answer took the run back already.
    // A run ended or taken back past this segment meanwhile stays as it is.
    const i = this.chain.indexOf(prev);
    if (!a.ok || i < 0) return;
    this.chain.length = i;
    // the undone segment's held state and early node are gone with it
    if (this.held.length) this.client.placedNow();
    this.held = [];
    this.earlyNode = -1;
    this.start = prev;
    this.real.begin(prev.ring);
    this.street.begin([prev.lng, prev.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.requestPreview();
    this.place();
  }

  routeReq(to: RouteEnd, by: { edge: number } | null = this.hoverBeside, at: Snap | null = this.hover) {
    const beside: { beside?: [number, number] } = by && to.kind === "free" ? { beside: [this.start!.beside ?? 0xffffffff, by.edge] } : {};
    return { from: this.start!.end, to, pis: [], single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), flyover: flyoverSpeed.peek(), hsr: highSpeed.peek(), tram: tramTrack.peek(), busway: buswayTrack.peek(), ramp: autoRamp.peek() && !tramTrack.peek(), ...beside, ...this.joinReq(at) };
  }

  /** T-3137: a single-track route's `RouteAsk.join`, from the start's and `at`'s `Snap.join`; nothing when neither has one */
  joinReq(at: Snap | null): { join?: [number | null, number | null, number | null, number | null] } {
    const a = this.start?.join, b = at && at.kind !== "free" ? at.join : undefined;
    if (!singleTrack.peek() || (!a && !b)) return {};
    const p = (j?: [number, number]): [number | null, number | null] => {
      if (!j) return [null, null];
      const q = this.pack(j[0], j[1]);
      return [q.x, q.y];
    };
    return { join: [...p(a), ...p(b)] };
  }

  /** T-2927: the pointer's spot for the street snap: the track it would end on, else where it is. */
  streetAt(px: number, py: number, h: Snap | null = this.hover): [number, number, number, number, boolean] {
    if (h && h.kind !== "free") return [px, py, ...toLngLat(h.lx, h.ly), true];
    const p = this.pointer;
    return [px, py, p?.lng ?? 0, p?.lat ?? 0, false];
  }
  streetMove(px: number, py: number) {
    if (!this.street.isOn) return;
    const [x, y, lng, lat, on] = this.streetAt(px, py);
    this.street.move(x, y, lng, lat, on);
  }

  /** T-2927: a street route's request: from the start, to the tram track under the pointer, else the
   * route's last point (on the street); street tram track. */
  streetReq(r: StreetRoute, at: Snap | null = this.hover) {
    const free = ([lng, lat]: [number, number]): RouteEnd => ({ kind: "free", ...lngLatToPack(lng, lat), level: this.freeLevel() });
    const to = at && at.kind !== "free" ? this.endOf(at) : free(r.last);
    // T-3129: rail on roads is an ordinary route through the fitted corners, as the real snap's
    if (!tramTrack.peek()) return { from: this.start!.end, to, pis: r.pis, single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), hsr: highSpeed.peek(), tram: false, ramp: autoRamp.peek(), ...this.joinReq(at) };
    return { from: this.start!.end, to, pis: r.pis, single: singleTrack.peek(), quad: false, through: true, design: designSpeed.peek(), hsr: false, tram: true, street: true, streetLine: r.line, ramp: false, ...this.joinReq(at) };
  }

  /** A real route's request (T-727d): from the start, or on open ground from the route's first
   * point; to the player's own track under the pointer, else the route's last point. */
  realReq(r: RealRoute, at: Snap | null = this.hover) {
    const free = ([lng, lat]: [number, number]): RouteEnd => ({ kind: "free", ...lngLatToPack(lng, lat), level: this.freeLevel() });
    const se = this.start!.end;
    return { from: se.kind === "free" ? free(r.first) : se, to: at && at.kind !== "free" ? this.endOf(at) : free(r.last), pis: r.pis, single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), flyover: flyoverSpeed.peek(), hsr: highSpeed.peek(), tram: tramTrack.peek(), busway: buswayTrack.peek(), ramp: autoRamp.peek() && !tramTrack.peek(), ...this.joinReq(at) };
  }

  /** Ask for the ghost of a segment from the start to the pointer; true if asked. */
  requestPreview(): boolean {
    if (!this.start || this.snapped || this.clicked) return false;
    const sn = endNode(this.start.end);
    const nodes = clickWorld()?.nodes;
    if (this.earlyNode >= 0 && nodes?.has(this.earlyNode)) this.earlyNode = -1;
    // undone meanwhile; a segment's end answered ahead of its state (T-1407) is not in yet, nor one not yet answered (stage D)
    if (sn !== undefined && sn !== this.earlyNode && !nodes?.has(sn)) return this.reset(), false;
    const h = this.hover;
    if (!h) return false;
    // Nothing to show while the pointer is on the start.
    const [ax, ay] = this.screen(...toLocal(this.start.lng, this.start.lat)), [bx, by] = this.screen(h.lx, h.ly);
    if (Math.hypot(ax - bx, ay - by) < 3) return false;
    // too far for one piece: not asked for, the tag says why (T-1020)
    if (this.tooFar(h)) {
      this.wantKey = "";
      this.lastPreview = null;
      this.previewKey = "";
      this.clearPreview();
      return false;
    }
    const req = this.routeReq(this.endOf(h), undefined, h);
    const key = JSON.stringify(req);
    if (!this.wantKey || this.wantKey === this.previewKey) this.waitFrom(false);
    this.wantKey = key;
    const g = this.startLocal(h);
    this.localFor = g ? { key, pts: g.a.pts } : null;
    const from = this.start;
    this.client.preview(req, (a) => {
      // T-3260: a ghost from a start on track leaves along its heading (its lead-in or tangent corner)
      if (from === this.start && !from.dir && from.end.kind !== "free" && a.verts.length >= 8) {
        const ux = a.verts[4] - a.verts[0], uy = a.verts[5] - a.verts[1], l = Math.hypot(ux, uy);
        if (l > 1e-9) from.dir = [ux / l, uy / l];
      }
      // T-4136: an answer for this run is newer than the one shown (one in flight, answered in order), so it is shown
      // even when the pointer has moved on since it was asked, bent to the pointer until the newest lands (T-830's
      // click still takes only the answer for its own spot)
      if (this.start === from && !this.snapped && this.wantKey) this.showPreview(a, key, this.wantKey === key);
    });
    if (!this.lastPreview) this.drawStandIn(h);
    return true;
  }

  /** T-4210: the drawing tool's settings for a ghost laid on the page; null for the track it does not lay (quad, tram,
   * busway) */
  tangentSpec(): TangentSpec | null {
    if (quadTrack.peek() || tramTrack.peek() || buswayTrack.peek()) return null;
    return { level: this.freeLevel(), design: designSpeed.peek(), hsr: highSpeed.peek(), single: singleTrack.peek() };
  }
  /** T-4210: the tangent segment from a run's bare end `o` (local units, its heading and level) to open ground `h`, laid
   * on the page; null where the worker lays another route (a snap, track or a beside spot under the pointer, another
   * build level) or the code is not loaded yet */
  localGhost(o: { lx: number; ly: number; dir?: [number, number]; tangent?: { level: number } }, h: Snap | null) {
    const sp = this.tangentSpec();
    if (!sp || !o.dir || !o.tangent || o.tangent.level !== sp.level || h?.kind !== "free" || this.snapped || this.real?.active || this.street?.isOn || this.hoverBeside) return null;
    return ghostTangent(o.lx, o.ly, o.dir, h.lx, h.ly, sp);
  }
  /** `localGhost` from the start to `h` (once per pointer spot), unless an answer from this start differed from it */
  startLocal(h: Snap | null = this.hover) {
    const st = this.start;
    if (!st?.tangent || this.localOff === st || !h) return null;
    const l = this.lastLocal;
    if (l?.st === st && l.h === h) return l.g;
    const [lx, ly] = toLocal(st.lng, st.lat);
    const g = this.localGhost({ lx, ly, dir: st.dir, tangent: st.tangent }, h);
    this.lastLocal = { st, h, g };
    return g;
  }
  /** T-4210: answer `a`'s figures on the shape laid on the page for the pointer's spot, in place of bending `a` */
  localAs(a: PreviewAnswer): PreviewAnswer | null {
    const g = this.startLocal();
    return g ? { ...g.a, ok: a.ok, cost: a.cost, mult: a.mult, issues: a.issues, ground: a.ground, noFlyover: a.noFlyover } : null;
  }

  /** T-4143: before the run's first answer there is nothing to bend, and the worker can be busy for seconds (a whole
   * state after a pan's street speeds land, an hour's trips): the ghost is drawn from what the page has, the start, its
   * heading if it has one and the pointer, until the checked answer replaces it. A click still waits for its own. */
  drawStandIn(h: Snap) {
    const g = this.startLocal(h);
    if (g) return this.drawLaid(g.a);
    const st = this.start!, [sx, sy] = toLocal(st.lng, st.lat);
    this.standIn(sx, sy, h, st.dir ?? this.trackHeading(st, sx, sy, h), st.z ?? this.heightOf(st.end, sx, sy));
  }
  /** the stand-in ghost from (sx, sy), leaving along `dir` when given, at height `z` (m), to the pointer's `h` */
  standIn(sx: number, sy: number, h: Snap, dir: [number, number] | undefined, z: number) {
    const pts = standInPts(sx, sy, h.lx, h.ly, dir, z, this.heightOf(this.endOf(h), h.lx, h.ly));
    this.bent = null;
    this.overlay.renderer.setPreview(strokesFromPreview(pts, false, undefined, true, undefined, singleTrack.peek() ? 0 : 255, highSpeed.peek(), tramTrack.peek()));
    this.overlay.invalidate();
  }

  /** T-4210: a ghost laid on the page, drawn as it is (no answer to bend yet) */
  drawLaid(a: PreviewAnswer) {
    this.bent = null;
    this.overlay.renderer.setPreview(this.ghostStrokes(a));
    this.overlay.invalidate();
  }

  /** where a click's segment ends (local units): a real or street route's last point on open ground, else the click's snap */
  clickEnd(s: Snap, rc: RealClick, sc: StreetClick): [number, number] {
    return rc?.kind === "route" && s.kind === "free" ? toLocal(...rc.route.last) : sc?.kind === "route" && s.kind === "free" ? toLocal(...sc.route.last) : [s.lx, s.ly];
  }
  /** T-4208: at a click, the segment's stand-in (T-4210: drawn ahead of the worker by the caller, game/pending.ts, until
   * the network holding it is drawn): this request's own answer when it is in, else the segment laid on the page for
   * the click's spot, else the ghost as shown; and the run's onward start set for the ghost until the answer */
  clickAhead(p: PreviewAnswer | null, s: Snap, chain: boolean, bare: boolean): Strokes | null {
    const r = this.overlay.renderer;
    const laid = this.startLocal(s), shown = this.bent?.a ?? this.lastPreview;
    const q = p ?? laid?.a ?? null;
    this.stampExact = !!q;
    const st = q ? (q.pts.length >= 6 ? this.ghostStrokes(q) : null) : r.previewSrc;
    const stamp = st?.count ? { ...st, marks: undefined, chevrons: undefined } : null;
    this.bent = null;
    // the heading the next start takes (segmentClick), so the ghost keeps its shape when the answer lands: the laid
    // segment's end (the track model's own), else the answer's or the ghost's as shown
    const eh = laid ? { ux: laid.end[0], uy: laid.end[1] } : (q ?? shown) ? endHeading((q ?? shown)!.verts, (q ?? shown)!.pts, (q ?? shown)!.arc, toLngLat, toLocal) : null;
    const { lx, ly } = s, [lng, lat] = toLngLat(lx, ly);
    // the click on open ground is the new segment's bare end, so the next one from it is a tangent route too
    const sp = bare ? this.tangentSpec() : null;
    this.clicked = { onward: chain ? { lx, ly, lng, lat, dir: eh ? [eh.ux, eh.uy] : undefined, z: this.heightOf(this.endOf(s), lx, ly), tangent: sp ? { level: sp.level } : undefined } : null };
    this.drawOnward();
    this.place();
    return stamp;
  }
  /** T-4208: the ghost while a click's answer is awaited, from the click's end to the pointer; none when the run ends there */
  drawOnward() {
    const o = this.clicked?.onward, h = this.hover;
    if (o && h && !this.endRun && !this.tooFar(h, o)) {
      const [ax, ay] = this.screen(o.lx, o.ly), [bx, by] = this.screen(h.lx, h.ly);
      if (Math.hypot(ax - bx, ay - by) >= 3) {
        const g = this.localGhost(o, h);
        return g ? this.drawLaid(g.a) : this.standIn(o.lx, o.ly, h, o.dir, o.z);
      }
    }
    this.clearPreview();
  }
  /** T-4143: the way track leaves a start on track before any answer says (local units, unit length): along the track
   * there, toward the pointer; none on open ground */
  trackHeading(st: Placed, sx: number, sy: number, h: Snap): [number, number] | undefined {
    const n = this.pickNet;
    if (st.end.kind === "free" || !n) return undefined;
    let best = 5 * kAtY(sy), dir: [number, number] | undefined;
    for (const { pts } of edgesNear(n, sx, sy, best) ?? [])
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const x0 = pts[k], y0 = pts[k + 1], dx = pts[k + 3] - x0, dy = pts[k + 4] - y0, l2 = dx * dx + dy * dy;
        if (!l2) continue;
        const t = Math.max(0, Math.min(1, ((sx - x0) * dx + (sy - y0) * dy) / l2)), d = Math.hypot(x0 + t * dx - sx, y0 + t * dy - sy);
        if (d >= best) continue;
        const l = Math.sqrt(l2), sg = dx * (h.lx - sx) + dy * (h.ly - sy) < 0 ? -1 : 1;
        best = d;
        dir = [(sg * dx) / l, (sg * dy) / l];
      }
    return dir;
  }
  /** a route end's height as the page knows it, m: a node's level, the track's height at (lx, ly), the level picked */
  heightOf(e: RouteEnd, lx: number, ly: number): number {
    const node = endNode(e);
    if (node !== undefined) return (clickWorld()?.nodes.get(node)?.level ?? this.freeLevel()) * 8;
    const n = this.pickNet;
    if (e.kind === "edge" && n) {
      let best = Infinity, z = this.freeLevel() * 8;
      for (const { edge, pts } of edgesNear(n, lx, ly, kAtY(ly)) ?? []) {
        if (edge !== e.edge) continue;
        for (let k = 0; k + 2 < pts.length; k += 3) {
          const d = Math.hypot(pts[k] - lx, pts[k + 1] - ly);
          if (d < best) [best, z] = [d, pts[k + 2]];
        }
      }
      return z;
    }
    return this.freeLevel() * 8;
  }

  /** T-3260: Ctrl held while a segment is drawn on from track: the segment runs straight on from its start.
   * T-3608: only with the street snap switch off; with it on, Ctrl just pauses the snap and the end is free. */
  straightOn(): boolean {
    const switchOn = tramTrack.peek() ? snapStreets.peek() && !snapRealTram.peek() : snapRoads.peek();
    return this.noSnap && !!this.start?.dir && !this.real.active && !this.street.isOn && !switchOn;
  }

  /** T-3260: `s` with Ctrl held: open ground moved onto the straight line on from the start (either way, so a
   * branch off the middle of track goes straight either way); anything else as it is */
  straight(s: Snap): Snap {
    const st = this.start;
    if (!st?.dir || s.kind !== "free" || !this.straightOn()) return s;
    const [sx, sy] = toLocal(st.lng, st.lat), [ux, uy] = st.dir;
    const t = (s.lx - sx) * ux + (s.ly - sy) * uy;
    return { kind: "free", lx: sx + t * ux, ly: sy + t * uy };
  }

  /** T-2404: the pointer near a held segment, away from the start the newest one ends at: their
   * state is wanted now, so the snap finds them by the time a click might */
  releaseNear(px: number, py: number) {
    if (!this.nearHeld(px, py)) return;
    this.held = [];
    this.client.placedNow();
  }
  /** whether a held segment is within HELD_NEAR_PX of (px, py), away from the start (dropping the
   * held list once their state has landed) */
  nearHeld(px: number, py: number): boolean {
    if (this.earlyNode < 0 || clickWorld()?.nodes.has(this.earlyNode)) {
      this.held = [];
      return false;
    }
    return this.nearPts(this.held, px, py);
  }
  /** whether one of the polylines `lines` (local x, y, z) is within HELD_NEAR_PX of (px, py), away from the start */
  nearPts(lines: Float32Array[], px: number, py: number): boolean {
    if (this.start) {
      const [sx, sy] = this.screen(...toLocal(this.start.lng, this.start.lat));
      if (Math.hypot(sx - px, sy - py) < HELD_START_PX) return false;
    }
    const r2 = HELD_NEAR_PX * HELD_NEAR_PX;
    for (const pts of lines)
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const [x0, y0] = this.screen(pts[k], pts[k + 1]), [x1, y1] = this.screen(pts[k + 3], pts[k + 4]);
        const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        if ((x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2 < r2) return true;
      }
    return false;
  }
  /** T-4311: what of this run's own track at map point `at` the tools' map lacks yet, which a click there waits for in the
   * click queue (map/toolWait.ts) so its snap finds it: a segment sent and not answered ("sent", `flying`), or one answered
   * ahead of a state the worker holds ("held", T-2404); null for neither */
  ownTrackOut(at: LngLatish | null): "sent" | "held" | null {
    if (!at || (!this.flying.length && !this.held.length)) return null;
    const p = this.map.project([at.lng, at.lat]);
    if (this.flying.length && this.nearPts(this.flying.map((f) => f.pts), p.x, p.y)) return "sent";
    return this.held.length && this.nearHeld(p.x, p.y) ? "held" : null;
  }
  /** T-4311: a segment's line from `from` through a real or street route's points (none: straight) to (ex, ey), local x, y, z
   * (z left 0: only `nearPts` reads it) */
  private sentLine(from: Placed, r: { first: [number, number]; last: [number, number]; pis: PiInput[] } | null, ex: number, ey: number): Float32Array {
    const via = r ? [r.first, ...r.pis.map((q) => packToLngLat(q.x, q.y)), r.last].map(([lng, lat]) => toLocal(lng, lat)) : [];
    return Float32Array.from([toLocal(from.lng, from.lat), ...via, [ex, ey]].flatMap(([x, y]) => [x, y, 0]));
  }
  /** T-4311: the node the run goes on from may be gone (game/pending.ts `breakRun`: an undo or redo from outside the run,
   * the worker's model rebuilt after a panic): the run ends, and the answers still to come for its segments no longer act
   * on the tool (a refused one would put the run back) */
  dropRun() {
    this.run++;
    if (this.start) this.reset();
  }

  /** T-4208: the camera moved: a track or station ghost's end follows the still pointer to the map spot now under it */
  cameraMoved() {
    const p = this.pointer, t = tool.peek();
    if (!p || touchInput.peek() || (t !== "track" && t !== "station") || (!this.hover && !this.hoverBeside)) return;
    const q = this.map.project([p.lng, p.lat]);
    if (Math.hypot(q.x - p.px, q.y - p.py) < 0.5) return;
    this.move({ point: { x: p.px, y: p.py }, lngLat: this.map.unproject([p.px, p.py]) } as MapMouseEvent);
  }

  /** since when the ghost has been behind the pointer, and since when a click has waited (ms) */
  askedAt = 0;
  placingAt = 0;

  waitFrom(click: boolean) {
    if (click) this.placingAt = performance.now();
    else this.askedAt = performance.now();
    setTimeout(() => this.place(), WAIT_MS + 20);
  }

  /** T-1039: a click or the ghost has waited on the worker more than WAIT_MS; the tag shows a spinner. */
  waiting(): boolean {
    const now = performance.now();
    return (this.placing && now - this.placingAt > WAIT_MS) || (!!this.wantKey && this.wantKey !== this.previewKey && now - this.askedAt > WAIT_MS) || !!this.street?.slow || !!this.real?.slow || clickWaiting(this);
  }

  /** Whether a free-drawn piece from the start to `s` would be longer than MAX_PIECE_KM. */
  tooFar(s: Snap, from: { lng: number; lat: number } | null = this.start): boolean {
    if (!from) return false;
    const [lng, lat] = toLngLat(s.lx, s.ly);
    return metresBetween(from.lng, from.lat, lng, lat) > MAX_PIECE_KM * 1000;
  }

  clearPreview() {
    this.bent = null;
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
  }

  /** `newest`: the answer for the newest request (a real or street route's always is); an older one is shown too, and
   * the wait for the newest counts again from it */
  showPreview(a: PreviewAnswer, key: string, newest = true) {
    // T-4210: an answer whose route is not the one laid on the page (water, a ramp, a refusal's) turns that off for the start
    const lf = this.localFor;
    if (lf?.key === key && this.start && a.pts.length >= 6 && !sameLine(lf.pts, a.pts, (x, y) => this.screen(x, y), LOCAL_PX)) this.localOff = this.start;
    this.lastPreview = a;
    this.previewKey = key;
    // a real route's preview (T-727d) answers what the ghost waits for too
    if (newest) this.wantKey = key;
    else this.waitFrom(false);
    this.drawGhost();
    this.place();
  }

  /** T-4136: the ghost drawn: the answer held, bent to the pointer while the pointer has moved on from its spot */
  drawGhost() {
    const a = this.lastPreview;
    if (!a || this.clicked) return;
    const b = this.previewKey !== this.wantKey && !this.snapped ? this.localAs(a) ?? this.bend(a) : null;
    this.bent = b ? { from: a, a: b } : null;
    const g = b ?? a;
    this.overlay.renderer.setPreview(g.pts.length >= 6 ? this.ghostStrokes(g) : null);
    this.overlay.invalidate();
  }

  /** T-4136: answer `a` moved so its end is at the pointer's spot: everything in it turned and scaled about the
   * start, a point by its distance from the start eased from none to all of it when the start has a heading (so the
   * ghost still leaves along it, and the part near the start stays as checked), all of it from a free start (whose
   * segment is straight). The figures are the answer's. Null with nothing to move or the pointer on the start. */
  bend(a: PreviewAnswer): PreviewAnswer | null {
    const h = this.hover, st = this.start, nv = a.verts.length;
    if (!h || !st || nv < 8) return null;
    const [sx, sy] = toLocal(st.lng, st.lat);
    const qx = a.verts[nv - 4] - sx, qy = a.verts[nv - 3] - sy, q2 = qx * qx + qy * qy;
    const [ax, ay] = this.screen(sx, sy), [bx, by] = this.screen(h.lx, h.ly);
    if (q2 < 1e-6 || Math.hypot(ax - bx, ay - by) < 3) return null;
    // the factor (a complex number) that takes the answer's end to the pointer about the start
    const px = h.lx - sx, py = h.ly - sy, kr = (px * qx + py * qy) / q2, ki = (py * qx - px * qy) / q2;
    const reach = Math.sqrt(q2), eased = !!st.dir || st.end.kind !== "free";
    const moved = <T extends Float32Array | Float64Array>(src: T | undefined, stride: number): T | undefined => {
      if (!src) return src;
      const out = src.slice() as T;
      for (let k = 0; k + 1 < out.length; k += stride) {
        const dx = out[k] - sx, dy = out[k + 1] - sy;
        let w = 1;
        if (eased) {
          const t = Math.min(1, Math.hypot(dx, dy) / reach);
          w = t * t * (3 - 2 * t);
        }
        out[k] += w * (kr * dx - ki * dy - dx);
        out[k + 1] += w * (ki * dx + kr * dy - dy);
      }
      return out;
    };
    const arc = moved(a.arc, 4);
    if (arc) arc[2] *= Math.hypot(kr, ki);
    return { ...a, pts: moved(a.pts, 3)!, verts: moved(a.verts, 4)!, arc, marks: moved(a.marks, 3), legs: moved(a.legs, 4), ramps: moved(a.ramps, 3) };
  }

  /** A route preview's ghost: single track as one track close in, double and quad as two each (T-658), as the sim
   * says per segment (T-3134); T-3137: single track's ends on the track of double track they join, as built track's
   * are drawn, and chevrons where it would run one way. */
  ghostStrokes(a: PreviewAnswer) {
    let pts = a.pts, speeds = a.speeds, lanes: number | Uint8Array = a.lanes ?? 255;
    const n = this.pickNet, w = clickWorld(), j = a.joins;
    if (j && (j[0] || j[1]) && n && w && speeds && !speeds.some((v) => v < 0)) {
      const doubles = this.doubleEnds(w).ids, lines = new Map<number, Float32Array>();
      for (const k of [0, pts.length / 3 - 1]) {
        const x = pts[k * 3], y = pts[k * 3 + 1];
        for (const { edge, pts: ep } of edgesNear(n, x, y, kAtY(y)) ?? []) if (doubles.has(edge)) lines.set(edge, ep);
      }
      const sh = joinedGhost(pts, j, [...lines.values()]);
      if (sh) {
        const sp = speeds, ln = lanes;
        pts = sh.pts;
        speeds = Float32Array.from(sh.src, (k) => sp[k]);
        if (typeof ln !== "number") lanes = Uint8Array.from(sh.src, (k) => ln[k]);
      }
    }
    // T-3171: double track leaving or reaching only single track narrows to it over TAPER_M, as built track does
    const taper = n && w && !singleTrack.peek() ? [0, pts.length / 3 - 1].map((k) => this.onlySingle(n, w, pts[k * 3], pts[k * 3 + 1])) : null;
    if (taper && (taper[0] || taper[1]) && (typeof lanes !== "number" || lanes > 0)) {
      const m = pts.length / 3, at = new Float64Array(m), sg = Uint8Array.from({ length: m - 1 }, (_, k) => (typeof lanes === "number" ? lanes : lanes[Math.min(k, lanes.length - 1)]));
      for (let k = 1; k < m; k++) at[k] = at[k - 1] + Math.hypot(pts[k * 3] - pts[k * 3 - 3], pts[k * 3 + 1] - pts[k * 3 - 2]);
      const L = Math.min(TAPER_M * kAtY(pts[1]), at[m - 1] / (taper[0] && taper[1] ? 2 : 1)) || 1;
      for (let k = 0; k < m - 1; k++) {
        const f = Math.max(0, Math.min(1, taper[0] ? at[k] / L : 1, taper[1] ? (at[m - 1] - at[k + 1]) / L : 1));
        sg[k] = Math.round(sg[k] * f);
      }
      lanes = sg;
    }
    const s = strokesFromPreview(pts, !a.ok, speeds, true, a.marks, lanes, highSpeed.peek(), tramTrack.peek());
    s.chevrons = ghostChevrons(s, j);
    return s;
  }

  // ---------------------------------------------------------------- stations

  /** The name a station placed where the pointer rests would get, worked out while it rests, so a
   * click does not wait 10-20 ms for it (T-496); used only if the view, world and tiles are as they
   * were, so it is the name the click would have found. */
  named: { key: string; name: Promise<string> } | null = null;
  namedTimer = 0;
  /** Local trip legs a day per cell, homes and destinations together (T-412), for the demand shown. */
  legs: { key: string; xy: Float32Array; legs: Float32Array; seq: number } | null = null;
  legsAsked = "";
  legsSeq = 0;

  nearKept: { legs: object; lng: number; lat: number; r: number; out: { text: string; legs: number } | null } | null = null;

  // ---------------------------------------------------------------- delete (T-096)

  // ---------------------------------------------------------------- lines

  lineClick(e: MapMouseEvent) {
    if (this.busStops?.active) return this.busStops.click(e.point.x, e.point.y, e.originalEvent.ctrlKey || e.originalEvent.metaKey);
    const n = this.pickNet;
    if (!n) return;
    // a toggle marker (T-233, map/stopMarkers.ts) is on top of the station
    const marker = this.stops?.pick(e.point.x, e.point.y);
    if (marker) {
      this.stops!.activate(marker, e.point.x, e.point.y, e.originalEvent.shiftKey);
      return;
    }
    const station=this.overlay.renderer.stationAt(e.point.x,e.point.y,(x,y)=>this.screen(x,y),7);
    const hit=station>=0?Number(n.stationIds[station]):-1;
    if (hit < 0) {
      say(mobileDevice() ? tr("toolNotes.lineHint", { touch: true }) : tr("tools.lineHelp"));
      return;
    }
    this.toggleStop(hit, e.originalEvent.shiftKey);
  }

  /** Add the station as a stop, or take it out if the line stops there (T-233), every visit of it (T-3109: a click
   * does nothing else). With `oneWay` (a shift-click, T-3711) it goes in as a stop served one way, or a stop of the
   * line is made one way or both ways again. Clicks queue and each reads the latest stops. */
  toggleStop(station: number, oneWay = false) {
    this.queueStop(station, undefined, oneWay);
  }

  /** T-3109, a right click or a long press on a stop: close the line into a loop or open it (`tier` "close", "open",
   * T-798), turn its end round a loop of track or open it again ("turn", "unturn", T-3021). */
  loopStop(station: number, tier: StopTier) {
    this.queueStop(station, tier);
  }

  /** T-3731: the click's line and game stamp are taken now; it is sent however stop editing has ended meanwhile, and
   * dropped only when another game has loaded (its ids are another game's: here, or by the worker, T-3734) or the
   * line is gone. */
  queueStop(station: number, tier?: StopTier, oneWay = false) {
    const id = lineDraft.peek().line, game = gameStamp.peek();
    if (id === null) return;
    if (!tier) {
      const line = clickWorld()?.save.lines.find((l) => Number(l.id) === id);
      // T-3474: the panel's rows are the stops its trains serve, which a guess matches only while they are the plan
      // (a shift-click on a stop keeps the row)
      const on = !!line?.stops.includes(String(station));
      if (line && !line.planStops && !(oneWay && on)) guessRow(String(station), on, line.num);
    }
    // T-4114: a plain click goes at once (the worker reads the stops it holds then); a loop's or a shift-click waits its turn
    const go = () => this.toggle(station, id, game, tier, oneWay);
    void (!tier && !oneWay ? sendStopEdit(go) : queueStopEdit(go)).catch(() => {
      if (game === gameStamp.peek()) say(tr("tools.stopFailed"), true);
    }).finally(() => {
      if (tier) return;
      this.stops?.settleClick(id, station);
      const line = clickWorld()?.save.lines.find((l) => Number(l.id) === id);
      settleRow(String(station), !!line && !line.planStops && line.stops.includes(String(station)), id);
    });
  }

  async toggle(station: number, id: number, game: number, tier?: StopTier, oneWay = false) {
    if (game !== gameStamp.peek()) return;
    const line = clickWorld()?.save.lines.find((l) => Number(l.id) === id);
    if (!line) return;
    const before = planStopsOf(line).map(Number);
    let change: Parameters<typeof edit>[0];
    const n = before.length;
    const closedLoop = n >= 3 && before[0] === before[n - 1];
    if (tier === "turn") {
      // T-3021: the end past this stop turned round its loop
      change = { op: "turnEnd", line: id, station };
    } else if (tier === "open" && closedLoop && new Set(before).size === n - 1 && before.includes(station)) {
      // T-3118: a ring opened at any of its stops ends there: the stretch on from it goes
      const cyc = before.slice(0, n - 1), k = cyc.indexOf(station);
      change = { op: "setStops", line: id, stops: [...cyc.slice(k + 1), ...cyc.slice(0, k + 1)], autoFast: this.autoFast(id) };
    } else if (tier === "unturn" || tier === "open") {
      // a loop opened: a turned line less its way back (from any of its stops, T-3118)
      change = { op: "openLoop", line: id };
    } else if (tier === "close") {
      if ((station !== before[0] && station !== before[n - 1]) || closedLoop) return;
      // T-3203: from either end the loop closes the same way, the last stop on to the first
      change = { op: "setStops", line: id, stops: [...before, before[0]], autoFast: this.autoFast(id) };
    } else if (tier) {
      return;
    } else if (oneWay) {
      // T-3711: served one way, the way that makes sense (the worker's), or both ways again
      change = { op: "oneWayStop", line: id, station, autoFast: this.autoFast(id) };
    } else {
      // T-4114: added where the worker fits it among the stops it holds then (T-648), or taken out, every visit of it
      // (the route between its neighbours the worker's; a loop that loses its first stop closes again on its new
      // first stop, T-3050): no click works from stops an earlier one changed, so none waits for the one before
      change = { op: "toggleStop", line: id, station, autoFast: this.autoFast(id) };
    }
    // T-2766: answered once the world holds it, its network once the clicks stop
    const a = await edit(change, false, false, "rows", game);
    if (!a.ok || game !== gameStamp.peek()) return;
    if (tool.peek() === "line" && lineDraft.peek().line === id) {
      const latest = clickWorld()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: planStopsOf(latest).map(Number) };
    }
    this.noteFast(id);
    void settleNewLineColour(id);
  }

  /** T-1899: whether a stop edit on `id` picks its train type by the track: while it is the new line
   * and the player has not set the type by hand since the last stop edit. */
  autoFast(id: number): boolean {
    const n = newLine.peek();
    if (n?.line !== id) return false;
    const fast = clickWorld()?.lineStats[String(id)]?.fast;
    if (fast !== undefined && fast !== n.fast) {
      newLine.value = null;
      return false;
    }
    return true;
  }

  /** T-1899: the type the new line's stops gave it, to tell a later change by hand. */
  noteFast(id: number) {
    const fast = clickWorld()?.lineStats[String(id)]?.fast;
    if (newLine.peek()?.line === id && fast !== undefined) newLine.value = { line: id, fast };
  }

  // ---------------------------------------------------------------- markers

  /** T-2763: the box of the ghost last drawn, kept for a click to stand in for the station it places */
  lastBox: { lx: number; ly: number; ways: ArrayLike<number>[]; perM: number; plat: number; side: number; tram: boolean; mid: boolean } | null = null;
  /** T-4214 stage E: the spot `lastBox` was laid at (a station clicked before its ghost's verdict is drawn from it) */
  lastBoxAt: Snap | null = null;
  /** T-2763: stations clicked and not yet in the world (the worker takes 0.3 to 3 s on a big network),
   * drawn where they were clicked; the worker's answer takes them away, and a refusal leaves none */
  pendingSt: { b: NonNullable<Tools["lastBox"]>; el: SVGSVGElement }[] = [];
  /** The placement ghost: the platform as it will be drawn, at the configured length along the
   * snapped track and curved with it, as built stations are (T-144, T-189). */
  ghost: SVGSVGElement | null = null;
  /** the last pointer position, in screen px and degrees */
  pointer: { px: number; py: number; lng: number; lat: number } | null = null;

  /** Faint rings 1 and 2 km, or miles in US units (ground distance), round the station being placed (T-312), under the
   * ghost, labelled at their tops; 0.5 and 1 for a tram stop, spaced closer (T-3001). */
  rings: SVGSVGElement | null = null;
  /** Where a station would go and whether a click there is taken (T-311, T-671, T-695). `guess`
   * comes at once from the track's layout (`likelyFirst`), so the first frame is right; `answer`
   * when the worker has tried the same `addStation` the click sends, rolled back, and overrides it.
   * A refused spot stays where the guess put it. */
  guess: Verdict | null = null;
  answer: Verdict | null = null;
  /** what the ghost last showed (null: nothing) */
  shown: Verdict | null = null;
  /** the worker's last answer for each track or node (null: taken) and where it was tried, for
   * this network and platform length (T-695) */
  tried: { scope: string; at: Map<string, { issues: string[] | null; lx: number; ly: number }> } = { scope: "", at: new Map() };
  /** the track's layout as `layoutIssues` reads it, for the world it was made from, with the flat
   * crossings on each edge from the map's marks */
  lay: (Layout & { version: number }) | null = null;
  /** the spots the pointer's candidates (`alts`) are tried at, and the world and platform they were tried in */
  altsKey(h: Spot | null = this.hover as Spot | null) {
    const spots=this.alts.length?this.alts:h?[h]:[];
    return spots.map((h)=>`${h.kind==="node"?`n${h.node}`:`e${h.edge}`}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}`).join("|")+`:${buildPlatform.peek()}:${buildTramPlatform.peek()}:${stationOneTrack.peek()}:${clickWorld()?.version}`;
  }

  /** the worker's answer for the pointer's candidates, else the layout's guess */
  verdict(): Verdict | null {
    const h=this.hover;
    if (!h || h.kind==="free") return null;
    const key=this.altsKey(h);
    return this.answer?.key===key ? this.answer : this.guess?.key===key ? this.guess : null;
  }

  /** Where a station would go (T-695): the worker's spot, else the guessed one. */
  stationSpot(): Snap | null {
    const h=this.hover;
    if (!h || h.kind==="free") return h;
    const v=this.verdict();
    return v ? v.spot : h;
  }

  /** The faint circle the segment being placed curves along (T-418), red when it is refused. */
  circle: SVGSVGElement | null = null;
  /** T-1895: beside the circle, a short line on from the segment's end the way it arrives, where a
   * straight drawn on from there would go; for a straight segment too, except one from a free start,
   * whose line on is its own continuation (T-2008). */
  tangent: SVGSVGElement | null = null;
  /** Move the DOM markers to the camera (after each MapLibre frame and each change). */
  place() {
    // on touch the pointer is a tap, a place on the map: its screen point follows the camera
    const p = this.pointer;
    if (p && touchInput.peek()) {
      const s = this.map.project([p.lng, p.lat]);
      p.px = s.x;
      p.py = s.y;
      if (this.touchAt && tool.peek() === "delete") this.delAt = [s.x, s.y];
    }
    placeMarks(this);
    waitSpin(this);
    if (this.pendingSt.length) placePending(this);
    placeTouch(this);
    drawBand(this);
  }

  /** T-1721: with the track tool and Snap beside on, a faint blue band BESIDE_PX either side of all
   * track, where drawing snaps beside it; redrawn when the view or the network changes. */
  bandEl: HTMLCanvasElement | null = null;
  bandKey = "";
}


/** T-4214 stage D: the node each of this session's route edits ended at, by seq, once answered (the newest 64) */
const runNodes = new Map<number, number>();
function noteRunNode(seq: number, node: number) {
  runNodes.set(seq, node);
  if (runNodes.size > 64) runNodes.delete(runNodes.keys().next().value!);
}
/** a route end's node: its own, or the node the edit it names ended at once that edit is answered */
function endNode(e: RouteEnd): number | undefined {
  return e.kind === "node" ? e.node : e.kind === "endOf" ? runNodes.get(e.seq) : undefined;
}

function sameStops(a: readonly number[], b: readonly number[]) {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}
