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
import { lngLatToPack } from "../game/coords";
import { distance, money } from "../game/format";
import { activeCity, worldFrameEpoch } from "../game/cities";
import { demandView } from "../game/demand";
import { settleNewLineColour } from "../game/lineColour";
import { metresBetween } from "../game/walks";
import { ask, autoRamp, buildLevel, buildPlatform, buildTramPlatform, buswayTrack, designSpeed, flyoverSpeed, highSpeed, keepPlacing, lineDraft, newLine, quadTrack, selection, singleTrack, besideOn, snapping, snapMode, snapRealTram, snapRoads, snapStreets, stationKind, stationOnTram, stationOneTrack, tool, tramTrack } from "../game/ui";
import { type Removable } from "../game/remove";
import { loadPlacesAt } from "../game/placeList";
import { warmStreetNames } from "./stationNames";
import { loadRealStationsAt } from "./realStations";
import { pointerLocal } from "./pick";
import { edgesNear, nodesNear } from "./snapIndex";
import { hidden, levelHidden } from "./visibility";
import { afterStopEdits, edit, queueStopEdit, say, sendStopEdit, stopsQueued, world } from "../game/world";
import { dropAhead, showAhead } from "./aheadShow";
import { edgesWithoutEnd } from "./lineAhead";
import { hideBusy, showBusy } from "../game/busy";
import { type RouteEnd } from "../workers/protocol";
import { gameStamp, onRoadsLanded, type ClockClient, type PreviewAnswer } from "../workers/clockClient";
import { roadTileTemplate } from "./busRoads";
import { kAtY, toLngLat, toLocal } from "./geo";
import { laneState } from "./renderer";
import { planStopsOf, type Level, type LineInput } from "../game/types";
import { endHeading } from "./curveCircle";
import { ghostChevrons, joinedGhost, strokesFromPreview, TAPER_M } from "./network";
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
import { type LngLatish, SNAP_EDGE_PX, TRACK_NODE_PX, TRACK_EDGE_PX, DEAD_PX, DOT_MIN_PX, HELD_NEAR_PX, HELD_START_PX, BESIDE_PX, OVERLAP_PX, type Snap, type Spot, type Verdict, BESIDE_SPACING_M, LANE_HALF_M, type Placed, MAX_PIECE_KM, WAIT_MS, tooFarText, placedAt } from "./toolParts";
import { nearTouch, touchConfirm, touchBack, touchStop, dragHandle, placeTouch } from "./toolTouch";
import { autoPick } from "./stationAuto";
import { nameAhead, stationClick, askLegs, placePending, layout } from "./stationTool";
import { freeCancel, freeStarted } from "./stationFree";
import { clearDelHover, deleteHover, deleteClick } from "./toolDelete";
import { drawBand, placeMarks } from "./toolMarks";

export class Tools {
  start: Placed | null = null;
  /** the starts of the segments placed with Shift, for Backspace (T-418) */
  chain: Placed[] = [];
  /** a segment's edit in flight: clicks wait for it */
  placing = false;
  /** T-1407: the end node of a segment answered before its state, until `world` has it (-1: none) */
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
  /** the preview the ghost, its marks and its tag show: the bent one while it stands in */
  get shownPreview(): PreviewAnswer | null {
    return this.bent && this.bent.from === this.lastPreview ? this.bent.a : this.lastPreview;
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
    Object.assign(touchActions, { confirm: () => void touchConfirm(this), back: () => touchBack(this), stop: () => touchStop(this), drawn: () => placeTouch(this) });
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
        this.exit();
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

  /** What is under the pointer for building: a node, a point on track, or open ground. */
  snap(px: number, py: number, lngLat: { lng: number; lat: number }, nodes = true): Snap {
    const w = world.peek();
    const n = this.overlay.renderer.net;
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
    const tramOnly = station && !this.autoProbe ? stationOnTram.peek() : null, tramOf = station ? this.tramEdgeSet(w) : null;
    for (const { edge, pts, ks } of nearEdges ?? [...n.edgePts].map(([edge, pts]) => ({ edge, pts, ks: null }))) {
      if (offEdges?.has(edge)) continue;
      // T-4010: a train station goes on train track only and a tram stop on tram track only
      if (tramOnly !== null && tramOf!.has(edge) !== tramOnly) continue;
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
    const lanes = laneState(this.map.getZoom()), t = lanes.t, w = world.peek(), n = this.overlay.renderer.net;
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

  /** whether a station spot is on tram track (T-2926) */
  tramSpot(s: Snap | null | undefined): boolean {
    const w = world.peek();
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
    const n = this.overlay.renderer.net;
    const at = pointerLocal(this.overlay, px, py);
    if (!besideOn.peek() || !n || !at || this.freeClick) return null;
    let best = BESIDE_PX * BESIDE_PX, hit: { edge: number; lx: number; ly: number } | null = null;
    const w = world.peek(), tramOf = w ? this.tramEdgeSet(w) : null, fam = tramTrack.peek();
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
    const w = world.peek();
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
    if (t === "track" && this.start) this.requestPreview();
    // T-4136: the ghost follows the pointer at once, its checked shape after
    if (t === "track" && this.start && this.lastPreview && this.previewKey !== this.wantKey) this.drawGhost();
    if (t === "station") nameAhead(this, e.point.x, e.point.y);
    this.place();
  }

  click(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") {
      this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
      this.streetSwitch();
      this.trackClick(e);
    } else if (t === "station") {
      this.autoPick({ px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat });
      void stationClick(this, e);
    }
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") deleteClick(this, e);
    else if (t === "bus") this.bus.click(e);
    else if (t === "ferry") this.ferry.click(e);
    else if (t === "building") this.building.click(e);
  }

  // ---------------------------------------------------------------- touch (T-1260)

  /** A tap with a tool on a touch screen: a preview where a click would act, which a second tap
   * on the same spot or the bar's confirm carries out. The first point of a run of track is set at
   * once, as a click sets it; so is a tap on open ground with the station tool (it is refused). */
  tap(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "line") return this.click(e);
    if (t === "bus") return this.bus.tap(e);
    if (t === "ferry") return this.ferry.tap(e);
    if (t === "building") return this.building.tap(e);
    if (this.touchAt && nearTouch(this, e.point.x, e.point.y)) return void touchConfirm(this);
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
        void stationClick(this, e);
        return;
      }
      this.touchAt = ll;
      this.move(e);
    } else if (t === "delete") {
      this.touchAt = ll;
      this.move(e);
    }
  }

  /** Keys while a tool is on; true if used. */
  key(e: KeyboardEvent): boolean {
    const t = tool.peek();
    if (t === "select") return false;
    if (t === "bus") return this.bus.key(e);
    if (t === "ferry") return this.ferry.key(e);
    if (t === "building") return this.building.key(e);
    if (e.key === "Escape") {
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
        this.reset();
        return true;
      }
    }
    if (t === "line" && e.key === "Enter") {
      this.exit();
      return true;
    }
    return false;
  }

  /** Leave the current tool (back to selecting). */
  exit() {
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
    this.start = null;
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
    const at = (p: Placed) => p.end.kind === "node" && p.end.node === node;
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
    if (this.held.length && this.nearHeld(e.point.x, e.point.y)) {
      this.held = [];
      this.client.placedNow();
      void this.heldLanded().then(() => this.trackClick(e));
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
   * starts at its end. */
  async segmentClick(e: MapMouseEvent, s: Snap) {
    const from = this.start;
    if (!from || this.placing) return;
    // T-707: a click on the start ends the run, as a double click does, or drops a start with no
    // segment yet; a segment can never end where it starts, so a loop's end is never this point
    const [sx, sy] = this.screen(...toLocal(from.lng, from.lat)), se = this.endOf(s);
    if (Math.hypot(sx - e.point.x, sy - e.point.y) < 6 || (from.end.kind === "node" && se.kind === "node" && se.node === from.end.node)) {
      this.reset();
      return;
    }
    // T-1811: a segment that joins track (a node, or mid-piece of track or blueprint) ends the run
    // under keep placing; Shift held still carries on from the join
    const chain = e.originalEvent.shiftKey || (keepPlacing.peek() && s.kind === "free");
    this.placing = true;
    this.waitFrom(true);
    this.endRun = false;
    let rc: RealClick, sc: StreetClick = null, p: PreviewAnswer | null, a: Awaited<ReturnType<typeof edit>>, by: { edge: number } | null = null;
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
      // T-1407: answered before its state, so the next ghost is asked at once
      a = await edit({ op: "route", ...req }, false, true, true);
    } finally {
      this.placing = false;
      this.place();
    }
    // T-1407: until the state lands, the ghost just placed stands in for the blueprint it made
    if (a.ok && a.early) {
      // T-2404: its state held while the pointer moves, its edges stand in where no ghost of it can
      // (the ghost as this very answer drew it, not bent to a pointer that moved on meanwhile)
      if (p) {
        this.overlay.renderer.setPreview(p.pts.length >= 6 ? this.ghostStrokes(p) : null);
        this.bent = null;
        this.overlay.renderer.keepPreviewAsPlaced();
      }
      else for (const pts of a.held ?? []) this.overlay.renderer.addPlaced(strokesFromPreview(pts, false, undefined, true, undefined, singleTrack.peek() ? 0 : 255, highSpeed.peek(), tramTrack.peek()));
      if (a.held) this.held.push(...a.held);
      if (a.node !== undefined) this.earlyNode = a.node;
    }
    if (!a.ok || this.start !== from) return;
    say(p ? tr("tools.blueprintCost", { distance: distance(p.lengthM), cost: money(p.cost * 1e6) }) : tr("tools.blueprint"));
    if (!chain || this.endRun || a.node === undefined) {
      this.reset();
      return;
    }
    this.chain.push(from);
    const [ex, ey] = rc && s.kind === "free" ? toLocal(...rc.route.last) : sc && s.kind === "free" ? toLocal(...sc.route.last) : [s.lx, s.ly];
    this.start = placedAt({ kind: "node", node: a.node }, ex, ey, rc ? rc.route.last : snapping.peek() ? this.real.ring(e.point.x, e.point.y) : null);
    this.start.beside = by?.edge ?? null; // a run beside track carries on beside it
    const eh = p ? endHeading(p.verts, p.pts, p.arc, toLngLat, toLocal) : null;
    if (eh) this.start.dir = [eh.ux, eh.uy];
    this.real.begin(this.start.ring);
    this.street.begin([this.start.lng, this.start.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.clearPreview();
    this.requestPreview();
    this.place();
  }

  /** Backspace while chaining: the last segment undone, drawing on from its start again. */
  async takeBackSegment() {
    // a segment still being placed is not on the undo stack yet
    if (this.placing) return;
    const prev = this.chain.pop();
    if (!prev) {
      this.reset();
      return;
    }
    this.placing = true;
    const a = await edit({ op: "undo" }, true).finally(() => (this.placing = false));
    if (!a.ok) return this.reset();
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
    if (!this.start || this.snapped) return false;
    const se = this.start.end;
    const nodes = world.peek()?.nodes;
    if (this.earlyNode >= 0 && nodes?.has(this.earlyNode)) this.earlyNode = -1;
    // undone meanwhile; a segment's end answered ahead of its state (T-1407) is not in yet
    if (se.kind === "node" && se.node !== this.earlyNode && !nodes?.has(se.node)) return this.reset(), false;
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
    return true;
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
    if (this.earlyNode < 0 || world.peek()?.nodes.has(this.earlyNode)) {
      this.held = [];
      return false;
    }
    if (this.start) {
      const [sx, sy] = this.screen(...toLocal(this.start.lng, this.start.lat));
      if (Math.hypot(sx - px, sy - py) < HELD_START_PX) return false;
    }
    const r2 = HELD_NEAR_PX * HELD_NEAR_PX;
    for (const pts of this.held)
      for (let k = 0; k + 5 < pts.length; k += 3) {
        const [x0, y0] = this.screen(pts[k], pts[k + 1]), [x1, y1] = this.screen(pts[k + 3], pts[k + 4]);
        const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2)) : 0;
        if ((x0 + t * dx - px) ** 2 + (y0 + t * dy - py) ** 2 < r2) return true;
      }
    return false;
  }
  /** T-2404: a click near a held segment waits for their state (at most 3 s), so its snap can find them */
  async heldLanded() {
    const node = this.earlyNode, until = performance.now() + 3000;
    while (node >= 0 && !world.peek()?.nodes.has(node) && performance.now() < until) await new Promise((r) => setTimeout(r, 16));
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
    return (this.placing && now - this.placingAt > WAIT_MS) || (!!this.wantKey && this.wantKey !== this.previewKey && now - this.askedAt > WAIT_MS) || !!this.street?.slow || !!this.real?.slow;
  }

  /** Whether a free-drawn piece from the start to `s` would be longer than MAX_PIECE_KM. */
  tooFar(s: Snap): boolean {
    if (!this.start) return false;
    const [lng, lat] = toLngLat(s.lx, s.ly);
    return metresBetween(this.start.lng, this.start.lat, lng, lat) > MAX_PIECE_KM * 1000;
  }

  clearPreview() {
    this.bent = null;
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
  }

  /** `newest`: the answer for the newest request (a real or street route's always is); an older one is shown too, and
   * the wait for the newest counts again from it */
  showPreview(a: PreviewAnswer, key: string, newest = true) {
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
    if (!a) return;
    const b = this.previewKey !== this.wantKey && !this.snapped ? this.bend(a) : null;
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
    const n = this.overlay.renderer.net, w = world.peek(), j = a.joins;
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
    const n = this.overlay.renderer.net;
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
      const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
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
      const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      settleRow(String(station), !!line && !line.planStops && line.stops.includes(String(station)), id);
    });
  }

  async toggle(station: number, id: number, game: number, tier?: StopTier, oneWay = false) {
    if (game !== gameStamp.peek()) return;
    const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
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
      this.guessPath(line, station);
    }
    // T-2766: answered once the world holds it, its network once the clicks stop
    const a = await edit(change, false, false, "rows", game);
    if (!a.ok && change.op === "toggleStop") dropAhead(id);
    if (!a.ok || game !== gameStamp.peek()) return;
    if (tool.peek() === "line" && lineDraft.peek().line === id) {
      const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: planStopsOf(latest).map(Number) };
    }
    this.noteFast(id);
    void settleNewLineColour(id);
  }

  /** T-4114: a plain click that takes out the line's first or last stop cuts its strokes back at the click (the
   * worker's own path follows a moment later); only with no other click out and the network drawn from the world's stops */
  guessPath(line: LineInput, station: number) {
    const r = this.overlay.renderer, n = r.net;
    // (a network older than the world has an earlier click's stops: its stretches are not these stops')
    if (!n || r.netVersion !== world.peek()?.version || line.planStops || stopsQueued.peek() > 1) return;
    const stops = line.stops.map(Number), k = stops.indexOf(station);
    if (k < 0 || stops[0] === stops[stops.length - 1] || stops.indexOf(station, k + 1) >= 0) return;
    const edges = edgesWithoutEnd(n, Number(line.id), k, stops.length);
    if (edges) showAhead(Number(line.id), edges, false);
  }

  /** T-1899: whether a stop edit on `id` picks its train type by the track: while it is the new line
   * and the player has not set the type by hand since the last stop edit. */
  autoFast(id: number): boolean {
    const n = newLine.peek();
    if (n?.line !== id) return false;
    const fast = world.peek()?.lineStats[String(id)]?.fast;
    if (fast !== undefined && fast !== n.fast) {
      newLine.value = null;
      return false;
    }
    return true;
  }

  /** T-1899: the type the new line's stops gave it, to tell a later change by hand. */
  noteFast(id: number) {
    const fast = world.peek()?.lineStats[String(id)]?.fast;
    if (newLine.peek()?.line === id && fast !== undefined) newLine.value = { line: id, fast };
  }

  // ---------------------------------------------------------------- markers

  /** T-2763: the box of the ghost last drawn, kept for a click to stand in for the station it places */
  lastBox: { lx: number; ly: number; ways: ArrayLike<number>[]; perM: number; plat: number; side: number; tram: boolean; mid: boolean } | null = null;
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
    return spots.map((h)=>`${h.kind==="node"?`n${h.node}`:`e${h.edge}`}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}`).join("|")+`:${buildPlatform.peek()}:${buildTramPlatform.peek()}:${stationOneTrack.peek()}:${world.peek()?.version}`;
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
    if (this.pendingSt.length) placePending(this);
    placeTouch(this);
    drawBand(this);
  }

  /** T-1721: with the track tool and Snap beside on, a faint blue band BESIDE_PX either side of all
   * track, where drawing snaps beside it; redrawn when the view or the network changes. */
  bandEl: HTMLCanvasElement | null = null;
  bandKey = "";
}


function sameStops(a: readonly number[], b: readonly number[]) {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}
