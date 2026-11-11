// The map tools (T-023, T-024): what a click does while drawing track, placing stations or adding
// stops to a line. Selecting (picking) is map/pick.ts; main.ts routes clicks here when a tool is
// on.
//
// Track, drawn through points (T-418): the second click places a
// segment from the first, one curve leaving along the track there and passing through the click,
// at once and never changed by later clicks; with Shift held or keepPlacing on (T-445) the next
// segment starts at its end, until a double click, Escape or Enter.
// The first click may start on a track end or on track (a new junction, flat, T-080); a click on
// track or a node joins it there. Backspace takes the last segment of a chain back, Escape or
// Enter stops. On quad track a segment from a loose quad end carries both pairs on (T-423). While
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

import { t as tr, tNumber } from "../i18n";
import { mobileDevice } from "../workers/deviceMemory";
import { effect } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { EARTH_R } from "../game/frame";
import { describeIssues } from "../game/issues";
import { count, distance, money, unitMetres, unitShort } from "../game/format";
import { activeCity, getWorldFrame, worldFrameEpoch } from "../game/cities";
import { cellModes, cityCells, demandView } from "../game/demand";
import { settleNewLineColour } from "../game/lineColour";
import { metresBetween } from "../game/walks";
import { ask, autoRamp, buildLevel, buildPlatform, buildTramPlatform, designSpeed, highSpeed, keepPlacing, lineDraft, newLine, quadTrack, selection, singleTrack, besideOn, snapping, snapRoads, snapStreets, stationOnTram, tool, tramTrack } from "../game/ui";
import { describeRemovable, removeThing, stationStub, stationTrim, type Removable } from "../game/remove";
import { loadPlacesAt } from "../game/placeList";
import { newStationName, trackDirection, warmStreetNames } from "./stationNames";
import { pointerLocal } from "./pick";
import { edgesNear, nodesNear } from "./snapIndex";
import { edit, say, world } from "../game/world";
import type { RouteEnd } from "../workers/protocol";
import { onRoadsLanded, type ClockClient, type PreviewAnswer } from "../workers/clockClient";
import { roadTileTemplate } from "./busRoads";
import { kAtY, toLngLat, toLocal } from "./geo";
import { laneState } from "./renderer";
import type { Level } from "../game/types";
import { circleOnScreen, drawTangent, endHeading } from "./curveCircle";
import { ghostChevrons, joinedGhost, strokesFromParts, strokesFromPreview } from "./network";
import { PLATFORM_HALF_M, platformShape, reversed, stationPath, stationShape, waysFrom } from "./stationGeometry";
import type { Overlay } from "./overlay";
import { marksLayer, placeAt } from "./screenPlace";
import type { StopMarkers, StopTier } from "./stopMarkers";
import type { BusStopMarkers } from "./busStops";
import { curveLabels, placeCurveLabels, placementTag } from "./trackTagRows";
import { firstAccepted, fractionAlong, layoutIssues, layoutOf, likelyFirst, nearestOn, orderOverlapping, pointAlong, roomNear, type Candidate, type Layout } from "./stationSnap";
import { M_CROSSING, MARK_STRIDE } from "./markData";
import { RealSnap, type RealClick, type RealRoute } from "./realSnap";
import { StreetSnap, type StreetClick, type StreetRoute } from "./streetSnap";
import { TRACK_SPECS } from "../game/trackKinds";
import { SAME_TAP_PX, touchActions, touchCtl, touchInput, type TouchCtl } from "./touch";
import { BusDraw } from "./busDraw";
import { FerryDraw } from "./ferryDraw";
import { BuildingDraw } from "./buildingDraw";
import { buildingAtPack } from "../game/buildingView";
import { BuildingFloors, startBuildingFloors } from "./buildingFloors";
import { playBuild } from "./buildSound";
import { perfOff } from "../perfFlags";
import { logShare } from "./modeBar";

/** the touch control bar's gap from the point it serves, so the finger does not cover it (px) */
const TOUCH_GAP_PX = 44;
type LngLatish = { lng: number; lat: number };

// T-1820: the station tool's reach for track, 1.5 times the earlier 10 (it snaps to track only, T-1870)
const SNAP_EDGE_PX = 15;
// T-1895: the track tool's, smaller, since a click near track more often means open ground than
// a branch; a node (a loose end to carry on from) keeps more
const TRACK_NODE_PX = 12;
const TRACK_EDGE_PX = 8;
/** T-2404: the pointer this near a segment whose state is held (twice the node snap, so the state
 * lands before the pointer reaches what it could snap to), unless this near the start */
const HELD_NEAR_PX = 2 * TRACK_NODE_PX, HELD_START_PX = 3 * TRACK_NODE_PX;
const notOnTrack = () => tr("tools.notOnTrack");
/** T-2271: the station tool's trips-near bar is empty at 1,000 trips and full at 1M, log scaled
 * (New York's lines have 30k-1M within 1 km); its ticks (style.css `.size-bar`) are 10k and 100k */
const NEAR_BAR_LO = 1e3, NEAR_BAR_HI = 1e6;
/** open ground this near track (and further than TRACK_EDGE_PX) draws beside it (T-1685) */
const BESIDE_PX = 40;
/** tracks this much further from the pointer than the nearest still count as overlapping it (T-469) */
const OVERLAP_PX = 4;

/** `join` (T-3137): with single track picked, the point (local units) on the track of double track there that the
 * pointer picks (`Tools.joinPoint`) */
type Snap =
  | { kind: "node"; node: number; lx: number; ly: number; join?: [number, number] }
  | { kind: "edge"; edge: number; lx: number; ly: number; join?: [number, number] }
  | { kind: "free"; lx: number; ly: number };

type Spot = Exclude<Snap, { kind: "free" }>;

/** where a station would go for the candidates `key` (null: no ghost), and whether a click there
 * is taken; `held`: the ghost as it was, until the worker answers */
interface Verdict { key: string; spot: Spot | null; ok: boolean; why: string; held?: boolean }

/** this many candidates under the pointer, not yet tried, and the ghost waits for the worker (T-695) */
const HOLD_AT = 7;
/** the worker's answer for a track tried this close to the pointer's spot stands for it (T-695) */
const KNOWN_PX = 16;
const LAYOUT_ISSUES = new Set(["PlatformTooLong", "PlatformStation", "PlatformJunction", "PlatformJoint", "PlatformCrossing", "StationPorts", "StationJunction"]);
/** track drawn beside track, centre to centre (sim `BESIDE_SPACING`, T-1895) */
const BESIDE_SPACING_M = 12;
/** each of double track's two tracks from its centre line once drawn apart, m (renderer.ts `LANE_M`, sim `LANE_HALF_M`) */
const LANE_HALF_M = 2;

const lengthOf = (pts: ArrayLike<number>) => {
  let l = 0;
  for (let k = 0; k + 5 < pts.length; k += 3) l += Math.hypot(pts[k + 3] - pts[k], pts[k + 4] - pts[k + 1]);
  return l;
};

interface Placed {
  end: RouteEnd;
  /** where it is in degrees: local units go stale when the draw anchor moves (T-1020) */
  lng: number;
  lat: number;
  /** the real network's point the snap starts from here (T-727d), or null */
  ring: [number, number] | null;
  /** the piece of track the start is beside (T-1685), or null */
  beside?: number | null;
  /** T-3137: the track of double track the start joins (`Snap.join`) */
  join?: [number, number];
}

/** The longest piece of free-drawn track one click lays, straight-line km between its ends (T-1020:
 * a piece from London to Shanghai took the clock worker 3-11 s a preview, and every pointer move
 * asked for one). */
const MAX_PIECE_KM = 500;
/** a click or ghost waiting this long on the worker shows the spinner in the tag, ms (T-1039) */
const WAIT_MS = 300;
const tooFarText = () => tr(mobileDevice() ? "tools.tooFarTouch" : "tools.tooFar", { max: distance(MAX_PIECE_KM * 1000) });

/** An x, y, z polyline less the shares `fromA` and `fromB` of its length at its two ends. */
function trimPolyline(pts: Float32Array, fromA: number, fromB: number): Float32Array {
  const n = pts.length / 3, cum = [0];
  for (let i = 1; i < n; i++) cum.push(cum[i - 1] + Math.hypot(pts[i * 3] - pts[i * 3 - 3], pts[i * 3 + 1] - pts[i * 3 - 2]));
  const total = cum[n - 1], s0 = fromA * total, s1 = (1 - fromB) * total;
  const at = (s: number) => {
    let i = 1;
    while (i < n - 1 && cum[i] < s) i++;
    const t = cum[i] > cum[i - 1] ? (s - cum[i - 1]) / (cum[i] - cum[i - 1]) : 0;
    return [0, 1, 2].map((k) => pts[(i - 1) * 3 + k] + (pts[i * 3 + k] - pts[(i - 1) * 3 + k]) * t);
  };
  const out = [...at(s0)];
  for (let i = 0; i < n; i++) if (cum[i] > s0 && cum[i] < s1) out.push(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
  out.push(...at(s1));
  return Float32Array.from(out);
}

function placedAt(end: RouteEnd, lx: number, ly: number, ring: [number, number] | null): Placed {
  const [lng, lat] = toLngLat(lx, ly);
  return { end, lng, lat, ring };
}

export class Tools {
  private start: Placed | null = null;
  /** the starts of the segments placed with Shift, for Backspace (T-418) */
  private chain: Placed[] = [];
  /** a segment's edit in flight: clicks wait for it */
  private placing = false;
  /** T-1407: the end node of a segment answered before its state, until `world` has it (-1: none) */
  private earlyNode = -1;
  /** T-2404: the points (local x, y, z) of the segments placed early whose state the worker holds
   * while the pointer moves; the snap cannot see them until it lands */
  private held: Float32Array[] = [];
  /** a double click's second click came while the segment was still being placed (T-445) */
  private endRun = false;
  private hover: Snap | null = null;
  /** the track the pointer is beside, while drawing from beside track (T-1685) */
  private hoverBeside: { edge: number; lx: number; ly: number } | null = null;
  /** the delete tool: what a click would remove, and the pointer (map px) */
  private delHover: (Removable & { lx: number; ly: number }) | null = null;
  private delAt: [number, number] = [0, 0];
  private lastPreview: PreviewAnswer | null = null;
  /** snapping to the real network (T-727d) */
  readonly real: RealSnap;
  /** snapping tram track to streets (T-2927) */
  readonly street: StreetSnap;
  /** the route between the clicks is a snapped one, real or street, not an ordinary segment */
  private get snapped() {
    return this.real.active || this.street.active;
  }
  /** the bus line tool (T-1464) */
  readonly bus: BusDraw;
  /** the ferry line tool (T-2194) */
  readonly ferry: FerryDraw;
  /** the station building tool (T-2218) */
  readonly building: BuildingDraw;
  /** the request `lastPreview` answers, as JSON */
  private previewKey = "";
  /** the newest request asked for; an older one's answer landing late is dropped (T-830) */
  private wantKey = "";
  /** where the last station of this session was placed (lng, lat), T-312 */
  private lastPlaced: [number, number] | null = null;
  private box: HTMLDivElement;
  private marks: HTMLDivElement[] = [];
  private tag: HTMLDivElement;
  private ring: HTMLDivElement;
  private lineSession = 0;
  private editingLine: number | null = null;
  /** the toggle markers on every station while a line's stops are edited (T-233); set by main.ts */
  stops: StopMarkers | null = null;
  busStops: BusStopMarkers | null = null;
  private stopClicks: Promise<void> = Promise.resolve();
  /** T-1260, on touch: the tap whose preview waits to be confirmed (the track's end, a station's
   * spot, what the delete tool would remove) */
  private touchAt: LngLatish | null = null;
  /** a station just placed by touch: no ghost at that point until the finger moves or taps elsewhere (T-1756) */
  private ghostSpent = false;
  /** the touch control bar's host (ui/TouchBar.tsx draws into it), and the drag handle on `touchAt` */
  readonly touchHost: HTMLDivElement;
  private handle: HTMLDivElement;
  private touchKey = "";

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
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
      this.clearDelHover();
      this.place();
    });
    effect(() => { buildPlatform.value; buildTramPlatform.value; if (tool.value === "station") this.place(); });
    effect(() => { if (tool.value === "station") stationOnTram.value = tramTrack.value; }); // T-3015: also when the type changes with the station tool in hand
    // the place lists name stations where the tiles have no streets (T-065); the view's zones' are
    // fetched when the station tool is picked, not at startup (T-400)
    effect(() => { if (tool.value === "station") { const c = this.map.getCenter(); void loadPlacesAt(c.lng, c.lat); warmStreetNames(this.map); } });
    // T-2430: the trips near are asked for when the tool is picked and when the demand shown changes,
    // so the figure is in hand when the pointer gets to the track
    effect(() => { demandView.value; activeCity.value; worldFrameEpoch.value; if (tool.value === "station") this.askLegs(); });
    map.on("idle", () => tool.peek() === "station" && warmStreetNames(map));
    effect(() => { buildLevel.value; autoRamp.value; highSpeed.value; tramTrack.value; quadTrack.value; singleTrack.value; designSpeed.value; if (tool.peek() === "track" && this.start) this.real?.active ? this.real.again() : this.street?.active ? this.street.again() : this.requestPreview(); });
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
    effect(() => this.real.setOn(snapping.value));
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
    effect(() => { tool.value; tramTrack.value; snapStreets.value; snapRoads.value; this.streetSwitch(); });
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
    this.handle.addEventListener("pointerdown", (e) => this.dragHandle(e));
    Object.assign(touchActions, { confirm: () => void this.touchConfirm(), back: () => this.touchBack(), stop: () => this.touchStop(), drawn: () => this.placeTouch() });
    effect(() => {
      tool.value;
      touchInput.value;
      this.touchAt = null;
      this.ghostSpent = false;
      if (tool.value !== "delete") this.clearDelHover();
      this.place();
    });
    // T-1039: a run ends with the track tool however it is left (a game loaded sets the tool directly)
    effect(() => { if (tool.value !== "track" && this.start) setTimeout(() => tool.peek() !== "track" && this.start && this.reset()); });
    effect(() => {
      const active = tool.value === "line" ? lineDraft.value.line : null;
      const w = world.value;
      if (active !== this.editingLine) {
        this.editingLine = active;
        this.lineSession++;
        // T-1899: a line is new until its first stop editing ends
        if (newLine.peek() && newLine.peek()!.line !== active) newLine.value = null;
      }
      if (active === null || !w) return;
      const line = w.save.lines.find((l) => Number(l.id) === active);
      const bus = line && w.lineStats[line.id]?.bus;
      // T-2891: a bus line's stops are edited only while it is by hand
      if (!line || (bus && !bus.hand)) {
        this.exit();
        return;
      }
      const stops = line.stops.map(Number);
      if (!sameStops(stops, lineDraft.peek().stops)) lineDraft.value = { line: active, stops };
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
  private noSnap = false;

  /** Follow Ctrl or Cmd from a pointer or key event; a change redraws the ghost where the pointer is. */
  noSnapFrom(e: { ctrlKey: boolean; metaKey: boolean }) {
    const on = e.ctrlKey || e.metaKey;
    if (on === this.noSnap) return;
    this.noSnap = on;
    this.streetSwitch();
    const p = this.pointer;
    if (tool.peek() === "track" && p) this.move({ point: { x: p.px, y: p.py }, lngLat: { lng: p.lng, lat: p.lat }, originalEvent: e } as unknown as MapMouseEvent);
  }

  /** T-2927: Ctrl is the free click only for rail; with Tram it flips the street snap */
  private get freeClick() {
    return this.noSnap && !tramTrack.peek();
  }

  /** T-2927: the street snap on while the track tool holds Tram, its switch flipped while Ctrl is held. */
  private streetSwitch() {
    if (!this.street) return;
    // T-3129: rail follows the roads with Snap to roads on and Ctrl not held
    const tram = tramTrack.peek(), on = tool.peek() === "track" && (tram ? snapStreets.peek() !== this.noSnap : snapRoads.peek() && !this.noSnap);
    const profile = tram ? "tram" : "rail";
    if (on === this.street.isOn && (!on || profile === this.street.snapProfile)) return;
    this.street.setOn(on, profile);
    if (on && this.start) this.street.begin([this.start.lng, this.start.lat]);
    if (!on && this.start) this.requestPreview();
  }

  private screen(lx: number, ly: number) {
    return this.overlay.toScreen(lx, ly);
  }

  /** What is under the pointer for building: a node, a point on track, or open ground. */
  private snap(px: number, py: number, lngLat: { lng: number; lat: number }, nodes = true): Snap {
    const w = world.peek();
    const n = this.overlay.renderer.net;
    const [fx, fy] = toLocal(lngLat.lng, lngLat.lat);
    const station = tool.peek() === "station";
    if (station) this.alts = [];
    if (!w || !n || (this.freeClick && tool.peek() === "track")) return { kind: "free", lx: fx, ly: fy };
    const edgePx = station ? SNAP_EDGE_PX : TRACK_EDGE_PX;
    // T-3162: single track picked beside double track drawn as two tracks: each of the two is a branch point across
    // its whole drawn width, so the pair's outer edge is the reach there
    const laneL = !station && tool.peek() === "track" && singleTrack.peek() && !tramTrack.peek() ? laneState(this.map.getZoom()) : null;
    const laneReach = laneL && laneL.t > 0 ? Math.max(edgePx, (laneL.gap * (laneL.t + laneL.share)) / 2) : edgePx;
    const doubleIds = laneReach > edgePx ? this.doubleEnds(w).ids : null;
    let best = TRACK_NODE_PX * TRACK_NODE_PX, hit: Snap | null = null;
    // only what is near the pointer (T-724); everything with the map pitched
    const at = pointerLocal(this.overlay, px, py);
    // T-1870: a station goes on the track at the pointer, never pulled to a node within the radius
    // (zoomed out that was a long stretch of track); `roomSnap` moves it where the platform needs room
    nodes &&= !station;
    // T-2926: tram track and train track never join, so while drawing, only the picked family's are snap targets
    // T-3153: only the track tool filters by family; delete and the rest see all track
    const fam = station || tool.peek() !== "track" ? null : tramTrack.peek();
    const tramOf = fam === null ? null : this.tramEdgeSet(w);
    const nearNodes = at && nodes ? nodesNear(w, at[0], at[1], TRACK_NODE_PX * at[2]) : null;
    const nearEdges = at ? edgesNear(n, at[0], at[1], laneReach * at[2]) : null;
    for (const node of nodes ? nearNodes ?? w.nodes.values() : []) {
      if (node.bus || node.port) continue; // a bus stop or a port (T-2194) is no track to build on (T-1464)
      if (fam !== null && (node.kind ?? 0) >= 2 !== fam) continue;
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
        if (!pts || !nd || pts.length < 6) continue;
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
    // each edge's nearest point within reach, px squared
    const near: { d: number; hit: Snap }[] = [];
    for (const { edge, pts, ks } of nearEdges ?? [...n.edgePts].map(([edge, pts]) => ({ edge, pts, ks: null }))) {
      if (tramOf && tramOf.has(edge) !== fam) continue;
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
    const order = close.length > 1 ? orderOverlapping(spots, this.platFor(close[0].hit as Spot), w.edges, w.nodes, this.lastStationTrack(), this.layout(w)) : close.map((_, i) => i);
    this.alts = [...order.map((i) => close[i].hit as Spot), ...near.slice(close.length, 5).map((c) => c.hit as Spot)];
    return this.alts[0];
  }

  /** `hit` with its `join` for the pointer at (px, py), local units */
  private withJoin(hit: Snap, px: number, py: number): Snap {
    if (hit.kind === "free") return hit;
    const join = this.joinPoint(hit, px, py);
    return join ? { ...hit, join } : hit;
  }

  /**
   * T-3137: with single track picked and double track drawn as two tracks, the point where `hit` meets the double's
   * track nearer the pointer (px, py, local units): on its centre line's normal there, as far off it as the
   * tracks are drawn. None zoomed out, where the track model joins the track beside the route.
   */
  private joinPoint(hit: Spot, px: number, py: number): [number, number] | undefined {
    if (tool.peek() !== "track" || !singleTrack.peek() || tramTrack.peek()) return undefined;
    const t = laneState(this.map.getZoom()).t, w = world.peek(), n = this.overlay.renderer.net;
    if (!(t > 0) || !w || !n) return undefined;
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
    if (!side) return undefined;
    const off = (LANE_HALF_M * t * kAtY(hit.ly) * side) / ul;
    return [hit.lx - best.uy * off, hit.ly + best.ux * off];
  }

  private doubleIndex: { edges: unknown; ids: Set<number>; byNode: Map<number, number[]>; loose: Map<number, { edge: number; fromA: boolean }> } | null = null;
  /** the world's double track (crossovers aside), and its pieces by the nodes at their ends, kept while its edges are the same */
  private doubleEnds(w: NonNullable<ReturnType<typeof world.peek>>) {
    if (this.doubleIndex?.edges !== w.edges) {
      const ids = new Set<number>(), byNode = new Map<number, number[]>(), ends = new Map<number, number>();
      for (const e of w.edges) {
        for (const k of [e.a, e.b]) ends.set(k, (ends.get(k) ?? 0) + 1);
        if (e.tracks !== 2 || e.crossover) continue;
        ids.add(e.id);
        for (const k of [e.a, e.b]) byNode.set(k, [...(byNode.get(k) ?? []), e.id]);
      }
      // T-3190: nodes where a double piece just ends
      const loose = new Map<number, { edge: number; fromA: boolean }>();
      for (const [node, es] of byNode) if (es.length === 1 && ends.get(node) === 1) loose.set(node, { edge: es[0], fromA: w.edges.find((e) => e.id === es[0])!.a === node });
      this.doubleIndex = { edges: w.edges, ids, byNode, loose };
    }
    return this.doubleIndex;
  }

  private tramEdges: { edges: unknown; set: Set<number> } | null = null;
  /** the ids of the world's tram edges (T-2926), kept while its edges are the same */
  private tramEdgeSet(w: NonNullable<ReturnType<typeof world.peek>>): Set<number> {
    if (this.tramEdges?.edges !== w.edges) this.tramEdges = { edges: w.edges, set: new Set(w.edges.filter((e) => e.kind).map((e) => e.id)) };
    return this.tramEdges.set;
  }

  /** whether a station spot is on tram track (T-2926) */
  private tramSpot(s: Snap | null | undefined): boolean {
    const w = world.peek();
    if (!s || !w) return false;
    if (s.kind === "free") return false;
    return s.kind === "node" ? (w.nodes.get(s.node)?.kind ?? 0) >= 2 : this.tramEdgeSet(w).has(s.edge);
  }

  /** the platform length a new station gets at a spot: tram stops have their own (T-2926) */
  private platFor(s: Snap | null | undefined): number {
    return this.tramSpot(s) ? buildTramPlatform.peek() : buildPlatform.peek();
  }

  /** T-1685: with Snap beside on, open ground within BESIDE_PX of track: the nearest piece, and the
   * spot BESIDE_SPACING_M from it on the pointer's side (local units) */
  private besideAt(px: number, py: number): { edge: number; lx: number; ly: number } | null {
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
  private alts: Spot[] = [];
  private stationTrack: { at: [number, number]; version: number; edges: Set<number> } | null = null;

  /** The edges of the track the last station placed this session is on (T-469): from its node
   * along the track both ways, through plain joints and stations, up to junctions. */
  private lastStationTrack(): Set<number> {
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
    const { byNode, byId: edges } = this.layout(w); // kept per network version: this runs per pointer move
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

  private pack(lx: number, ly: number) {
    const [lng, lat] = toLngLat(lx, ly);
    return lngLatToPack(lng, lat);
  }

  private endOf(s: Snap): RouteEnd {
    if (s.kind === "node") return { kind: "node", node: s.node };
    const p = this.pack(s.lx, s.ly);
    if (s.kind === "edge") return { kind: "edge", edge: s.edge, x: p.x, y: p.y };
    return { kind: "free", x: p.x, y: p.y, level: this.freeLevel() };
  }

  /** The elevation of a segment's end on open ground: the one picked. */
  private freeLevel(): Level {
    return buildLevel.peek();
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "select") return;
    if (t === "delete") return this.deleteHover(e.point.x, e.point.y);
    if (t === "bus") return this.bus.move(e);
    if (t === "ferry") return this.ferry.move(e);
    if (t === "building") return this.building.move(e);
    if (t === "line") (this.busStops?.active ? this.busStops : this.stops)?.hover(e.point.x, e.point.y); // T-233: the toggle markers' tag
    if (e.originalEvent) this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
    this.streetSwitch();
    this.ghostSpent = false;
    this.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
    this.hover = t === "line" ? null : this.snap(e.point.x, e.point.y, e.lngLat);
    if (t === "track") this.streetMove(e.point.x, e.point.y);
    // T-1721: from any start, and before the first click for the ring
    this.hoverBeside = t === "track" && this.hover?.kind === "free" && !this.snapped ? this.besideAt(e.point.x, e.point.y) : null;
    if (t === "track") this.real.move(e.point.x, e.point.y);
    if (t === "track" && this.held.length) this.releaseNear(e.point.x, e.point.y);
    if (t === "track" && this.start) this.requestPreview();
    if (t === "station") this.nameAhead(e.point.x, e.point.y);
    this.place();
  }

  click(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") {
      this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
      this.streetSwitch();
      this.trackClick(e);
    } else if (t === "station") void this.stationClick(e);
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") this.deleteClick(e);
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
    if (this.touchAt && this.nearTouch(e.point.x, e.point.y)) return void this.touchConfirm();
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
      if (this.snap(e.point.x, e.point.y, e.lngLat).kind === "free") {
        this.touchAt = null;
        this.move(e);
        void this.stationClick(e);
        return;
      }
      this.touchAt = ll;
      this.move(e);
    } else if (t === "delete") {
      this.touchAt = ll;
      this.move(e);
    }
  }

  private nearTouch(x: number, y: number) {
    const p = this.touchAt && this.map.project([this.touchAt.lng, this.touchAt.lat]);
    return !!p && Math.hypot(p.x - x, p.y - y) < SAME_TAP_PX;
  }

  /** a click's event at a map point, for the touch bar's confirm */
  private evAt(ll: LngLatish): MapMouseEvent {
    const p = this.map.project([ll.lng, ll.lat]);
    return { point: p, lngLat: this.map.unproject(p), originalEvent: { detail: 1, shiftKey: false } } as unknown as MapMouseEvent;
  }

  /** The bar's confirm, or a second tap on the point: what a click there does. */
  private async touchConfirm() {
    const t = tool.peek(), at = this.touchAt;
    if (t === "bus") return this.bus.confirm();
    if (t === "ferry") return this.ferry.confirm();
    if (t === "building") return this.building.confirm();
    if (t === "track") {
      const from = this.start;
      if (!from || !at || this.placing) return;
      this.touchAt = null;
      const ev = this.evAt(at);
      await this.segmentClick(ev, this.hover ?? this.snap(ev.point.x, ev.point.y, ev.lngLat));
      // refused: the end stays, to be moved or confirmed again
      if (this.start === from && !this.touchAt) {
        this.touchAt = at;
        this.place();
      }
    } else if (t === "station") {
      if (at) {
        this.touchAt = null;
        this.place();
        this.ghostSpent = true;
        const placed = this.stationClick(this.evAt(at));
        this.place();
        await placed;
        this.place();
      }
    } else if (t === "delete" && at) {
      this.touchAt = null;
      this.deleteClick(this.evAt(at));
    }
  }

  /** The bar's back button: a segment being drawn (start and waiting end) is dropped whole, the
   * tool staying on (T-1703); else the last segment is taken back (Backspace), else the start goes. */
  private touchBack() {
    if (tool.peek() === "bus") return this.bus.touchBack();
    if (tool.peek() === "ferry") return this.ferry.touchBack();
    if (tool.peek() === "building") return this.building.touchBack();
    if (tool.peek() !== "track") return this.touchStop();
    if (this.touchAt && this.start) {
      this.hover = null;
      this.reset();
    } else if (this.chain.length) void this.takeBackSegment();
    else this.reset();
  }

  /** The bar's stop button: the run ends (its placed track stays) or the placement is dropped; the tool stays on. */
  private touchStop() {
    const t = tool.peek();
    if (t === "bus" || t === "ferry" || t === "building") return this.exit();
    if (t === "track") return this.reset();
    this.touchAt = null;
    this.hover = null;
    if (t === "delete") {
      this.delHover = null;
      this.clearPreview();
    }
    this.place();
  }

  /** Dragging the handle moves the point being previewed; a tap on it confirms. */
  private dragHandle(e: PointerEvent) {
    e.preventDefault();
    const el = this.handle;
    el.setPointerCapture(e.pointerId);
    let moved = false;
    const move = (m: PointerEvent) => {
      const r = this.map.getContainer().getBoundingClientRect();
      const x = m.clientX - r.left, y = m.clientY - r.top;
      if (!moved && Math.hypot(m.clientX - e.clientX, m.clientY - e.clientY) < 6) return;
      moved = true;
      const ll = this.map.unproject([x, y]);
      this.touchAt = { lng: ll.lng, lat: ll.lat };
      this.move({ point: { x, y }, lngLat: ll, originalEvent: m } as unknown as MapMouseEvent);
    };
    const up = (u: PointerEvent) => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      // a tap on the handle is a second tap on the point
      if (!moved && u.type === "pointerup") void this.touchConfirm();
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  }

  /** What the touch bar offers now; null hides it. */
  private touchState(): TouchCtl | null {
    if (!touchInput.peek()) return null;
    const t = tool.peek();
    if (t === "bus") return this.bus.touchState();
    if (t === "ferry") return this.ferry.touchState();
    if (t === "building") return this.building.touchState();
    if (t === "track" && this.start) {
      const pv = this.lastPreview;
      const bad = !!pv && !pv.ok && this.previewKey === this.wantKey && !this.snapped;
      return { confirm: this.touchAt ? !bad && !this.placing : null, confirmTip: tr("tools.placeTrack"), back: !!this.touchAt || this.chain.length > 0, stopTip: tr("tools.stopDrawing") };
    }
    if (t === "station" && this.touchAt && this.hover && this.hover.kind !== "free") return { confirm: this.shown?.ok !== false, confirmTip: tr("tools.placeStation"), back: false, stopTip: tr("tools.cancelStation") };
    if (t === "delete" && this.touchAt && this.delHover) return { confirm: true, confirmTip: tr("tools.remove"), back: false, stopTip: tr("tools.cancelRemove") };
    return null;
  }

  /** The touch bar beside the point it serves, clear of the finger and the map's edges, and the
   * tool's tag beyond it; the drag handle on the point being previewed. */
  private placeTouch() {
    if (!this.touchHost) return; // during construction
    const ctl = this.touchState();
    const key = JSON.stringify(ctl);
    if (key !== this.touchKey) {
      this.touchKey = key;
      touchCtl.value = ctl;
    }
    const host = this.touchHost, t = tool.peek();
    const at = t === "bus" ? this.bus.touchPoint() : t === "ferry" ? this.ferry.touchPoint() : t === "building" ? this.building.touchPoint() : this.touchAt ?? (t === "track" && this.start ? this.start : null);
    if (!ctl || !at) {
      host.style.display = "none";
      this.handle.style.display = "none";
      return;
    }
    const W = this.overlay.cssW, H = this.overlay.cssH;
    const p = this.map.project([at.lng, at.lat]);
    host.style.display = "";
    const bw = host.offsetWidth, bh = host.offsetHeight;
    const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
    // off the side the track comes from: past a waiting end, away from the start (the ghost lies
    // between); at a run's new start, behind it, where the last segment came from (the next
    // segment usually goes on ahead, where the next tap will be)
    const from = this.touchAt ? (t === "track" ? this.start : null) : t === "track" ? this.chain[this.chain.length - 1] ?? null : null;
    const q = from && this.map.project([from.lng, from.lat]);
    const wantAbove = !q ? true : this.touchAt ? q.y > p.y : q.y < p.y;
    const fitsAbove = p.y - TOUCH_GAP_PX - bh >= 8, fitsBelow = p.y + TOUCH_GAP_PX + bh <= H - 8;
    const above = wantAbove ? fitsAbove || !fitsBelow : !fitsBelow && fitsAbove;
    const y = clamp(above ? p.y - TOUCH_GAP_PX - bh : p.y + TOUCH_GAP_PX, 8, H - bh - 8);
    const x = clamp(p.x - bw / 2, 8, W - bw - 8);
    const box = this.map.getContainer().getBoundingClientRect();
    host.style.transform = `translate(${Math.round(box.left + x)}px, ${Math.round(box.top + y)}px)`;
    if (this.tag.style.display !== "none") {
      const tw = this.tag.offsetWidth, th = this.tag.offsetHeight;
      let ty = above ? y - 6 - th : y + bh + 6;
      if (ty < 8 || ty + th > H - 8) ty = above ? p.y + 24 : p.y - 24 - th;
      this.tag.style.transform = `translate(${Math.round(clamp(p.x - tw / 2, 8, W - tw - 8))}px, ${Math.round(ty)}px)`;
    }
    const drag = this.touchAt && t === "track";
    this.handle.style.display = drag ? "" : "none";
    if (drag) this.handle.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px)`;
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

  private trackClick(e: MapMouseEvent) {
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
    void this.segmentClick(e, s);
  }

  /** Place the segment from the start to this click; with Shift held or keepPlacing on, the next
   * starts at its end. */
  private async segmentClick(e: MapMouseEvent, s: Snap) {
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
      if (p) this.overlay.renderer.keepPreviewAsPlaced();
      else for (const pts of a.held ?? []) this.overlay.renderer.addPlaced(strokesFromPreview(pts, false, undefined, true, undefined, singleTrack.peek() && !tramTrack.peek() ? 0 : 255, highSpeed.peek(), tramTrack.peek()));
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
    this.real.begin(this.start.ring);
    this.street.begin([this.start.lng, this.start.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.clearPreview();
    this.requestPreview();
    this.place();
  }

  /** Backspace while chaining: the last segment undone, drawing on from its start again. */
  private async takeBackSegment() {
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
    this.start = prev;
    this.real.begin(prev.ring);
    this.street.begin([prev.lng, prev.lat]);
    this.lastPreview = null;
    this.previewKey = this.wantKey = "";
    this.requestPreview();
    this.place();
  }

  private routeReq(to: RouteEnd, by: { edge: number } | null = this.hoverBeside, at: Snap | null = this.hover) {
    const beside: { beside?: [number, number] } = by && to.kind === "free" ? { beside: [this.start!.beside ?? 0xffffffff, by.edge] } : {};
    return { from: this.start!.end, to, pis: [], single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), hsr: highSpeed.peek(), tram: tramTrack.peek(), ramp: autoRamp.peek() && !tramTrack.peek(), ...beside, ...this.joinReq(at) };
  }

  /** T-3137: a single-track route's `RouteAsk.join`, from the start's and `at`'s `Snap.join`; nothing when neither has one */
  private joinReq(at: Snap | null): { join?: [number | null, number | null, number | null, number | null] } {
    const a = this.start?.join, b = at && at.kind !== "free" ? at.join : undefined;
    if (!singleTrack.peek() || tramTrack.peek() || (!a && !b)) return {};
    const p = (j?: [number, number]): [number | null, number | null] => {
      if (!j) return [null, null];
      const q = this.pack(j[0], j[1]);
      return [q.x, q.y];
    };
    return { join: [...p(a), ...p(b)] };
  }

  /** T-2927: the pointer's spot for the street snap: the track it would end on, else where it is. */
  private streetAt(px: number, py: number, h: Snap | null = this.hover): [number, number, number, number, boolean] {
    if (h && h.kind !== "free") return [px, py, ...toLngLat(h.lx, h.ly), true];
    const p = this.pointer;
    return [px, py, p?.lng ?? 0, p?.lat ?? 0, false];
  }
  private streetMove(px: number, py: number) {
    if (!this.street.isOn) return;
    const [x, y, lng, lat, on] = this.streetAt(px, py);
    this.street.move(x, y, lng, lat, on);
  }

  /** T-2927: a street route's request: from the start, to the tram track under the pointer, else the
   * route's last point (on the street); street tram track. */
  private streetReq(r: StreetRoute, at: Snap | null = this.hover) {
    const free = ([lng, lat]: [number, number]): RouteEnd => ({ kind: "free", ...lngLatToPack(lng, lat), level: this.freeLevel() });
    const to = at && at.kind !== "free" ? this.endOf(at) : free(r.last);
    // T-3129: rail on roads is an ordinary route through the fitted corners, as the real snap's
    if (!tramTrack.peek()) return { from: this.start!.end, to, pis: r.pis, single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), hsr: highSpeed.peek(), tram: false, ramp: autoRamp.peek(), ...this.joinReq(at) };
    return { from: this.start!.end, to, pis: r.pis, single: false, quad: false, through: true, design: designSpeed.peek(), hsr: false, tram: true, street: true, streetLine: r.line, ramp: false };
  }

  /** A real route's request (T-727d): from the start, or on open ground from the route's first
   * point; to the player's own track under the pointer, else the route's last point. */
  private realReq(r: RealRoute, at: Snap | null = this.hover) {
    const free = ([lng, lat]: [number, number]): RouteEnd => ({ kind: "free", ...lngLatToPack(lng, lat), level: this.freeLevel() });
    const se = this.start!.end;
    return { from: se.kind === "free" ? free(r.first) : se, to: at && at.kind !== "free" ? this.endOf(at) : free(r.last), pis: r.pis, single: singleTrack.peek(), quad: quadTrack.peek(), through: true, design: designSpeed.peek(), hsr: highSpeed.peek(), tram: tramTrack.peek(), ramp: autoRamp.peek() && !tramTrack.peek(), ...this.joinReq(at) };
  }

  /** Ask for the ghost of a segment from the start to the pointer; true if asked. */
  private requestPreview(): boolean {
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
    this.client.preview(req, (a) => {
      if (this.start && !this.snapped && this.wantKey === key) this.showPreview(a, key);
    });
    return true;
  }

  /** T-2404: the pointer near a held segment, away from the start the newest one ends at: their
   * state is wanted now, so the snap finds them by the time a click might */
  private releaseNear(px: number, py: number) {
    if (!this.nearHeld(px, py)) return;
    this.held = [];
    this.client.placedNow();
  }
  /** whether a held segment is within HELD_NEAR_PX of (px, py), away from the start (dropping the
   * held list once their state has landed) */
  private nearHeld(px: number, py: number): boolean {
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
  private async heldLanded() {
    const node = this.earlyNode, until = performance.now() + 3000;
    while (node >= 0 && !world.peek()?.nodes.has(node) && performance.now() < until) await new Promise((r) => setTimeout(r, 16));
  }

  /** since when the ghost has been behind the pointer, and since when a click has waited (ms) */
  private askedAt = 0;
  private placingAt = 0;

  private waitFrom(click: boolean) {
    if (click) this.placingAt = performance.now();
    else this.askedAt = performance.now();
    setTimeout(() => this.place(), WAIT_MS + 20);
  }

  /** T-1039: a click or the ghost has waited on the worker more than WAIT_MS; the tag shows a spinner. */
  private waiting(): boolean {
    const now = performance.now();
    return (this.placing && now - this.placingAt > WAIT_MS) || (!!this.wantKey && this.wantKey !== this.previewKey && now - this.askedAt > WAIT_MS) || !!this.street?.slow || !!this.real?.slow;
  }

  /** Whether a free-drawn piece from the start to `s` would be longer than MAX_PIECE_KM. */
  private tooFar(s: Snap): boolean {
    if (!this.start) return false;
    const [lng, lat] = toLngLat(s.lx, s.ly);
    return metresBetween(this.start.lng, this.start.lat, lng, lat) > MAX_PIECE_KM * 1000;
  }

  private clearPreview() {
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
  }

  private showPreview(a: PreviewAnswer, key: string) {
    this.lastPreview = a;
    // a real route's preview (T-727d) answers what the ghost waits for too
    this.previewKey = this.wantKey = key;
    this.overlay.renderer.setPreview(a.pts.length >= 6 ? this.ghostStrokes(a) : null);
    this.overlay.invalidate();
    this.place();
  }

  /** A route preview's ghost: single track as one track close in, double and quad as two each (T-658), as the sim
   * says per segment (T-3134); T-3137: single track's ends on the track of double track they join, as built track's
   * are drawn, and chevrons where it would run one way. */
  private ghostStrokes(a: PreviewAnswer) {
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
    const s = strokesFromPreview(pts, !a.ok, speeds, true, a.marks, lanes, highSpeed.peek(), tramTrack.peek());
    s.chevrons = ghostChevrons(s, j);
    return s;
  }

  // ---------------------------------------------------------------- stations

  /** The name a station placed where the pointer rests would get, worked out while it rests, so a
   * click does not wait 10-20 ms for it (T-496); used only if the view, world and tiles are as they
   * were, so it is the name the click would have found. */
  private named: { key: string; name: Promise<string> } | null = null;
  private namedTimer = 0;
  private nameKey(px: number, py: number) {
    const c = this.map.getCenter();
    return `${px},${py},${c.lng},${c.lat},${this.map.getZoom()},${this.map.getBearing()},${this.map.getPitch()},${world.peek()?.version}`;
  }
  private nameAhead(px: number, py: number) {
    clearTimeout(this.namedTimer);
    this.namedTimer = setTimeout(() => {
      if (tool.peek() !== "station" || !this.map.areTilesLoaded()) return;
      const key = this.nameKey(px, py);
      if (this.named?.key !== key) this.named = { key, name: this.nameFor(px, py) };
    }, 150) as unknown as number;
  }

  private async stationClick(e: MapMouseEvent) {
    const first = this.snap(e.point.x, e.point.y, e.lngLat);
    if (first.kind === "free") {
      // T-1820: stations go on track only
      say(notOnTrack(), true);
      return;
    }
    // T-671, T-695: the spot the ghost shows, else the first the track model takes of what is under the pointer
    const v = this.verdict();
    // T-1474: heard at the click, not when the edit is answered; not when the ghost shows a refusal
    if (!(v && !v.ok && !v.held)) playBuild("station");
    const pend = v?.ok && !v.held && this.hover && this.hover.kind !== "free" ? this.addPending() : null;
    const tries = [...(v?.ok && v.spot && !v.held ? [v.spot] : []), ...this.alts];
    const key = this.nameKey(e.point.x, e.point.y);
    const ahead = this.named?.key === key && this.map.areTilesLoaded() ? this.named.name : null;
    this.named = null;
    const name = await (ahead ?? this.nameFor(e.point.x, e.point.y));
    const stationAt = (p: Exclude<Snap, { kind: "free" }>) => ({ op: "addStation" as const, at: p.kind === "node" ? { kind: "node" as const, node: p.node } : { kind: "edge" as const, edge: p.edge, ...this.pack(p.lx, p.ly) }, platform: this.platFor(p), name });
    const r = await firstAccepted(tries, async (p) => { const a = await edit(stationAt(p), true, true); return a.ok ? null : a.issues; }, (p) => this.roomSnap(p), () => false);
    this.dropPending(pend);
    if (r.spot) {
      this.lastPlaced = toLngLat(r.spot.lx, r.spot.ly); // T-312: the hover tag's "km from" (found again by position)
      say(tr("tools.blueprintStation", { name }));
    } else if (r.issues.length) say(describeIssues(r.issues), true);
  }

  /** Local trip legs a day per cell, homes and destinations together (T-412), for the demand shown. */
  private legs: { key: string; xy: Float32Array; legs: Float32Array; seq: number } | null = null;
  private legsAsked = "";
  private legsSeq = 0;

  /** Ask for the legs of the demand shown, unless held or asked already (T-412). T-2430: an answer
   * for a demand shown before is kept while nothing newer is held (a big zone's answer can take
   * longer than a solve's round, and dropping it left the tag without trips until the solve ended),
   * and one that fails is asked again 2 s later while the station tool is in hand. */
  private askLegs() {
    const dv = demandView.peek();
    if (!dv) return;
    const city = activeCity.peek(), key = `${dv.version}:${dv.round}:${dv.day}:${city}:${worldFrameEpoch.peek()}`;
    if (this.legs?.key === key || this.legsAsked === key) return;
    this.legsAsked = key;
    const seq = ++this.legsSeq;
    void Promise.all([cityCells(), cellModes("all")]).then(([cells, m]) => {
      if (!cells || !m) {
        setTimeout(() => {
          if (this.legsAsked !== key) return;
          this.legsAsked = "";
          if (tool.peek() === "station") this.askLegs();
        }, 2000);
        return;
      }
      // a newer answer is held already, or the zone or frame has changed (its cells are another set)
      if ((this.legs && this.legs.seq > seq) || activeCity.peek() !== city || !key.endsWith(`:${worldFrameEpoch.peek()}`)) return;
      const legs = new Float32Array(cells.n);
      for (const a of [m.homeRail, m.homeWalk, m.homeDrive, m.workRail, m.workWalk, m.workDrive, m.homeNone, m.workNone])
        if (a.length === cells.n) for (let i = 0; i < cells.n; i++) legs[i] += a[i];
      this.legs = { key, xy: cells.xy, legs, seq };
      if (tool.peek() === "station") this.place();
    });
  }

  /** "12k trips within 1km" of the spot (T-412, T-1058; 1 mile in US units), from the cells behind
   * the Homes and Destinations bubbles; null until they are in. A trip with both ends inside counts
   * twice, and a spot near a zone's edge counts the viewed zone's cells only. */
  private tripsNear(lng: number, lat: number): { text: string; legs: number } | null {
    this.askLegs();
    const L = this.legs;
    if (!L) return null;
    // the same answer until the pointer, the unit or the legs change: this runs on every place()
    const r = unitMetres(), kept = this.nearKept;
    if (kept && kept.legs === L && kept.lng === lng && kept.lat === lat && kept.r === r) return kept.out;
    const out = this.sumTripsNear(L, lng, lat, r);
    this.nearKept = { legs: L, lng, lat, r, out };
    return out;
  }

  private nearKept: { legs: object; lng: number; lat: number; r: number; out: { text: string; legs: number } | null } | null = null;

  private sumTripsNear(L: NonNullable<Tools["legs"]>, lng: number, lat: number, r: number): { text: string; legs: number } | null {
    // the radius in world metres each way, so any frame's scale is right here
    const f = getWorldFrame();
    const [x, y] = f.fromLonLat(lng, lat);
    const dLat = (r / EARTH_R) * (180 / Math.PI), dLng = dLat / Math.cos((lat * Math.PI) / 180);
    const [ex] = f.fromLonLat(lng + dLng, lat), [, ny] = f.fromLonLat(lng, lat + dLat);
    const rx = Math.abs(ex - x), ry = Math.abs(ny - y);
    let sum = 0, cells = 0;
    for (let i = 0; i < L.legs.length; i++) {
      const dx = (L.xy[i * 2] - x) / rx, dy = (L.xy[i * 2 + 1] - y) / ry;
      if (dx * dx + dy * dy <= 1) (sum += L.legs[i]), cells++;
    }
    if (!cells) return null;
    // two significant figures: the figure is rough by nature
    const p = 10 ** Math.max(0, Math.floor(Math.log10(Math.max(sum, 1))) - 1), r2 = Math.round(sum / p) * p;
    const n = sum < 50 ? null : r2 >= 10000 ? count(r2) : tNumber(r2);
    return { text: n === null ? tr("tools.nearUnder", { unit: unitShort() }) : tr("tools.near", { n, unit: unitShort() }), legs: sum };
  }

  /** The station placed last in this session (T-312), found by its position so a rename shows. */
  private lastPlacedName(): { name: string; lng: number; lat: number } | null {
    const p = this.lastPlaced;
    if (!p) return null;
    let best: { name: string; lng: number; lat: number } | null = null, bestD = 30;
    for (const s of world.peek()?.save.stations ?? []) {
      const d = metresBetween(p[0], p[1], s.lng, s.lat);
      if (d < bestD) (bestD = d), (best = s);
    }
    if (!best) this.lastPlaced = null; // taken out again
    return best;
  }

  /** A name for a new station where the pointer is (map/stationNames.ts): the cross street, a
   * small neighbourhood, or the fallbacks, skipping names in use. */
  private nameFor(px: number, py: number): Promise<string> {
    const s = this.snap(px, py, this.map.unproject([px, py]));
    const [lng, lat] = toLngLat(s.lx, s.ly);
    const net = this.overlay.renderer.net;
    const track = net ? trackDirection(net, ...toLocal(lng, lat, net.anchor)) : null;
    return newStationName(this.map, lng, lat, track, () => new Set(world.peek()?.save.stations.map((st) => st.name) ?? []));
  }

  // ---------------------------------------------------------------- delete (T-096)

  /** What a click with the delete tool would remove: a station, a flyover, else a stretch of
   * track, else a bus line (train lines are removed from the Lines tab). */
  private deletePick(px: number, py: number): (Removable & { lx: number; ly: number }) | null {
    const w = world.peek();
    const n = this.overlay.renderer.net;
    if (!w || !n) return null;
    const near = (lx: number, ly: number, slop: number) => {
      const [x, y] = this.screen(lx, ly);
      const d = Math.hypot(x - px, y - py);
      return d < slop ? d : Infinity;
    };
    let best = Infinity, hit: (Removable & { lx: number; ly: number }) | null = null;
    const station=this.overlay.renderer.stationAt(px,py,(x,y)=>this.screen(x,y),5);
    if (station>=0) return {kind:"station",node:Number(n.stationIds[station]),lx:n.stations[station*3],ly:n.stations[station*3+1]};
    for (const node of w.nodes.values()) {
      if (!node.flying || node.ports < 3) continue;
      const [lx, ly] = toLocal(node.lng, node.lat);
      const d = near(lx, ly, 10);
      if (d < best) [best, hit] = [d, { kind: "flyover", node: node.id, lx, ly }];
    }
    if (hit) return hit;
    const s = this.snap(px, py, this.map.unproject([px, py]));
    const under = this.stationUnder(s);
    if (under) return under;
    if (s.kind === "edge") return { kind: "track", edge: s.edge, lx: s.lx, ly: s.ly };
    if (s.kind === "node") {
      // a plain node (track end, junction): the stretch nearest to the pointer there
      let e = -1, bd = Infinity;
      const tryPoint = (edge: number, pts: Float32Array, k: number) => {
        const d = near(pts[k], pts[k + 1], 14);
        if (d < bd) [bd, e] = [d, edge];
      };
      const at = pointerLocal(this.overlay, px, py), close = at ? edgesNear(n, at[0], at[1], 14 * at[2]) : null;
      if (close) {
        for (const { edge, pts, ks } of close) for (const k of ks) tryPoint(edge, pts, k), tryPoint(edge, pts, k + 3);
      } else {
        for (const [edge, pts] of n.edgePts) for (let k = 0; k < pts.length; k += 3) tryPoint(edge, pts, k);
      }
      if (e >= 0) return { kind: "track", edge: e, lx: s.lx, ly: s.ly };
    }
    // a bus line has no track: under the rail, so only where no rail is (T-1513)
    const buses = this.overlay.renderer.buses;
    if (buses.any) {
      const line = buses.stopAt(this.overlay, px, py)?.line ?? buses.lineAt(this.overlay, px, py);
      const at = this.overlay.screenToLocal(px, py);
      if (line >= 0 && at) return { kind: "bus", line, lx: at[0], ly: at[1] };
    }
    // T-2218: a station building's floor, under everything else
    if (!BuildingFloors.shownAt(this.map.getZoom())) return null;
    const ll = this.map.unproject([px, py]), q = lngLatToPack(ll.lng, ll.lat), b = buildingAtPack(q.x, q.y);
    const here = b && this.overlay.screenToLocal(px, py);
    if (b && here) return { kind: "building", id: b.id, lx: here[0], ly: here[1] };
    return null;
  }

  /** T-2412: the delete tool on a station's node or its platform's stretch of track (half the platform
   * and 4 m either side, what `stationTrim` keeps; all of a terminus's platform) takes the station,
   * whose own mark is small zoomed out; the track stays, as when the station is clicked. So does a
   * station's short end piece, bare end included, which a removal could only lay back (T-2797). Not
   * where stations are hidden (T-205). */
  private stationUnder(s: Snap): (Removable & { lx: number; ly: number }) | null {
    const w = world.peek(), n = this.overlay.renderer.net;
    if (!w || !n || this.overlay.renderer.stationStyle.hidden || (s.kind !== "node" && s.kind !== "edge")) return null;
    const station = (id: number) => {
      const node = w.nodes.get(id);
      return node && node.platform > 0 && !node.bus && !node.port ? node : null;
    };
    const hit = (id: number) => {
      const node = w.nodes.get(id)!, [lx, ly] = toLocal(node.lng, node.lat);
      return { kind: "station" as const, node: id, lx, ly };
    };
    const stub = (edge: number | undefined) => {
      const id = edge === undefined ? null : stationStub(edge);
      return id !== null && station(id) ? hit(id) : null;
    };
    if (s.kind === "node") {
      if (station(s.node)) return hit(s.node);
      return w.nodes.get(s.node)?.ports === 1 ? stub(w.edges.find((x) => x.a === s.node || x.b === s.node)?.id) : null;
    }
    const held = stub(s.edge);
    if (held) return held;
    const e = w.edges.find((x) => x.id === s.edge), pts = n.edgePts.get(s.edge);
    if (!e || !pts || e.crossover) return null;
    const along = fractionAlong(pts, s.lx, s.ly) * e.lengthM;
    const reach = (id: number, d: number) => {
      const node = station(id);
      return node ? (node.ports <= 1 ? node.platform : node.platform / 2) + 4 - d : -Infinity;
    };
    const ra = reach(e.a, along), rb = reach(e.b, e.lengthM - along);
    if (ra < 0 && rb < 0) return null;
    return hit(ra >= rb ? e.a : e.b);
  }

  /** T-1872: the red stretch is the shared preview, so it goes with the hover whichever way the pointer or tool leaves. */
  private clearDelHover() {
    if (!this.delHover) return;
    this.delHover = null;
    this.clearPreview();
  }

  private deleteHover(px: number, py: number) {
    const h = this.deletePick(px, py);
    const same = (a: Removable | null, b: Removable | null) => JSON.stringify(a && { ...a, lx: 0, ly: 0 }) === JSON.stringify(b && { ...b, lx: 0, ly: 0 });
    const changed = !same(h, this.delHover);
    this.delHover = h;
    this.delAt = [px, py];
    if (changed) {
      // the stretch that would go, drawn red over itself
      // a crossover goes whole: both diagonals (T-391)
      const edgePts = this.overlay.renderer.net?.edgePts;
      const partner = h?.kind === "track" ? world.peek()?.edges.find((x) => x.id === h.edge)?.partner ?? -1 : -1;
      const parts = h?.kind === "track" && edgePts
        ? [h.edge, partner].map((id) => edgePts.get(id)).filter((p): p is Float32Array => !!p && p.length >= 6)
        : [];
      // T-1161: past a station, only what goes
      if (h?.kind === "track" && parts.length) {
        const [ka, kb] = stationTrim(h.edge), len = world.peek()?.edges.find((x) => x.id === h.edge)?.lengthM ?? 0;
        if ((ka || kb) && len > 0) parts[0] = trimPolyline(parts[0], ka / len, kb / len);
      }
      this.overlay.renderer.setPreview(parts.length ? strokesFromParts(parts, true, false) : null);
      this.overlay.invalidate();
    }
    this.map.getCanvas().style.cursor = h ? "pointer" : "crosshair";
    this.place();
  }

  private deleteClick(e: MapMouseEvent) {
    const h = this.deletePick(e.point.x, e.point.y);
    if (!h) return;
    // on touch nothing is shown until the next tap (T-1260)
    void removeThing(h).then(() => touchInput.peek() ? this.touchStop() : tool.peek() === "delete" && this.deleteHover(e.point.x, e.point.y));
  }

  // ---------------------------------------------------------------- lines

  private lineClick(e: MapMouseEvent) {
    if (this.busStops?.active) return this.busStops.click(e.point.x, e.point.y);
    const n = this.overlay.renderer.net;
    if (!n) return;
    // a toggle marker (T-233, map/stopMarkers.ts) is on top of the station
    const marker = this.stops?.pick(e.point.x, e.point.y);
    if (marker) {
      this.stops!.activate(marker, e.point.x, e.point.y);
      return;
    }
    const station=this.overlay.renderer.stationAt(e.point.x,e.point.y,(x,y)=>this.screen(x,y),7);
    const hit=station>=0?Number(n.stationIds[station]):-1;
    if (hit < 0) {
      say(mobileDevice() ? tr("toolNotes.lineHint", { touch: true }) : tr("tools.lineHelp"));
      return;
    }
    this.toggleStop(hit);
  }

  /** Add the station as a stop, or take it out if the line stops there (T-233), every visit of it (T-3109: a click
   * does nothing else). Clicks queue and each reads the latest stops. */
  toggleStop(station: number) {
    this.queueStop(station);
  }

  /** T-3109, a right click or a long press on a stop: close the line into a loop or open it (`tier` "close", "open",
   * T-798), turn its end round a loop of track or open it again ("turn", "unturn", T-3021). */
  loopStop(station: number, tier: StopTier) {
    this.queueStop(station, tier);
  }

  private queueStop(station: number, tier?: StopTier) {
    const session = this.lineSession;
    this.stopClicks = this.stopClicks.then(() => this.toggle(station, session, tier)).catch(() => {
      if (session === this.lineSession) say(tr("tools.stopFailed"), true);
    }).finally(() => { if (!tier) this.stops?.settleClick(station); });
  }

  private async toggle(station: number, session: number, tier?: StopTier) {
    if (session !== this.lineSession || tool.peek() !== "line") return;
    const id = lineDraft.peek().line;
    const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
    if (id === null || !line) return;
    const before = line.stops.map(Number);
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
      if (station !== before[0] || closedLoop) return;
      change = { op: "setStops", line: id, stops: [...before, station], autoFast: this.autoFast(id) };
    } else if (tier) {
      return;
    } else if (!before.includes(station)) {
      // the worker fits it in among the stops it holds then (T-648), so a queued click never works
      // from stops an earlier one changed
      change = { op: "insertStop", line: id, station, autoFast: this.autoFast(id) };
    } else {
      // taking a stop out, every visit of it: the route between its neighbours is the worker's, as the inspector's
      // crosses; a loop that loses its first stop closes again on its new first stop (T-3050)
      const stops = before.filter((s) => s !== station);
      if (closedLoop && stops.length >= 2 && stops[0] !== stops[stops.length - 1]) stops.push(stops[0]);
      change = { op: "setStops", line: id, stops, autoFast: this.autoFast(id) };
    }
    // T-2766: answered once the world holds it (the next click reads its stops), its network once the clicks stop
    const a = await edit(change, false, false, "rows");
    if (a.ok && session === this.lineSession && tool.peek() === "line") {
      const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: latest.stops.map(Number) };
      this.noteFast(id);
      void settleNewLineColour(id);
    }
  }

  /** T-1899: whether a stop edit on `id` picks its train type by the track: while it is the new line
   * and the player has not set the type by hand since the last stop edit. */
  private autoFast(id: number): boolean {
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
  private noteFast(id: number) {
    const fast = world.peek()?.lineStats[String(id)]?.fast;
    if (newLine.peek()?.line === id && fast !== undefined) newLine.value = { line: id, fast };
  }

  // ---------------------------------------------------------------- markers

  private el(i: number, cls: string): HTMLDivElement {
    let m = this.marks[i];
    if (!m) {
      m = this.marks[i] = document.createElement("div");
      this.box.appendChild(m);
    }
    m.className = cls;
    m.textContent = "";
    m.style.display = "";
    return m;
  }

  /** A 1 px SVG in the marks layer (`placeAt` moves it) with an empty child per tag in `kids`, at
   * the back of the layer or the front. */
  private svgMark(kids: string[], back: boolean): SVGSVGElement {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", "1");
    svg.setAttribute("height", "1");
    for (const k of kids) svg.appendChild(document.createElementNS(NS, k));
    if (back) this.box.insertBefore(svg, this.box.firstChild);
    else this.box.appendChild(svg);
    return svg;
  }

  /** T-2763: the box of the ghost last drawn, kept for a click to stand in for the station it places */
  private lastBox: { lx: number; ly: number; ways: ArrayLike<number>[]; perM: number; plat: number } | null = null;
  /** T-2763: stations clicked and not yet in the world (the worker takes 0.3 to 3 s on a big network),
   * drawn where they were clicked; the worker's answer takes them away, and a refusal leaves none */
  private pendingSt: { b: NonNullable<Tools["lastBox"]>; el: SVGSVGElement }[] = [];
  private addPending() {
    const b = this.lastBox;
    if (!b) return null;
    const el = this.svgMark(["path"], false);
    el.setAttribute("class", "station-ghost ok placed");
    const p = { b, el };
    this.pendingSt.push(p);
    this.placePending();
    return p;
  }
  private dropPending(p: { b: unknown; el: SVGSVGElement } | null) {
    if (!p) return;
    p.el.remove();
    this.pendingSt = this.pendingSt.filter((x) => x !== p);
  }
  private placePending() {
    for (const { b, el } of this.pendingSt) {
      const style = { ...this.overlay.renderer.stationStyle, boxes: true };
      const shape = stationShape(b.lx, b.ly, platformShape(b.ways, 0, b.plat / 2 * b.perM), PLATFORM_HALF_M * b.perM, (x, y) => this.screen(x, y), style, false);
      placeAt(el, shape.x, shape.y, 0, 0, false);
      el.firstElementChild!.setAttribute("d", stationPath(shape));
    }
  }

  /** The placement ghost: the platform as it will be drawn, at the configured length along the
   * snapped track and curved with it, as built stations are (T-144, T-189). */
  private ghost: SVGSVGElement | null = null;
  private stationGhost(h: Exclude<Snap,{kind:"free"}>) {
    const n=this.overlay.renderer.net,w=world.peek();
    if (!n || !w) return;
    const candidates=h.kind==="edge"?[h.edge]:w.edges.filter(e=>e.a===h.node||e.b===h.node).map(e=>e.id);
    const usage=new Map<number,Set<number>>();
    for (let i=0;i<n.lines.count;i++) {
      const edge=n.lines.edge[i];
      if (!candidates.includes(edge)) continue;
      if (!usage.has(edge)) usage.set(edge,new Set());
      usage.get(edge)!.add(n.lines.colour[i]);
    }
    candidates.sort((a,b)=>(usage.get(b)?.size??0)-(usage.get(a)?.size??0)||a-b);
    const edge=candidates[0];
    const [,lat]=toLngLat(h.lx,h.ly);
    const perM=1/Math.cos(lat*Math.PI/180);
    // the track either way from the point, as network.ts lays out a built station's platform
    const pts=n.edgePts.get(edge);
    const ways:ArrayLike<number>[]=h.kind==="edge"
      ? pts ? waysFrom(pts,h.lx,h.ly) : []
      : candidates.map((id)=>{ const e=w.edges.find((x)=>x.id===id), p=n.edgePts.get(id); return !e||!p?new Float32Array(0):e.a===h.node?p:reversed(p); });
    // on quad track the pair clicked only (T-1687)
    this.boxGhost(h.lx,h.ly,ways,perM,this.verdict()?.ok===false,null);
  }

  /** The platform box at (lx, ly) along `ways` (the track from the spot each way), and the twin's
   * on quad. */
  private boxGhost(lx: number, ly: number, ways: ArrayLike<number>[], perM: number, bad: boolean, twinAt: { x: number; y: number; ways: ArrayLike<number>[] } | null, plat = this.platFor(this.stationSpot())) {
    this.lastBox={lx,ly,ways,perM,plat};
    const sh=platformShape(ways,0,plat/2*perM);
    // always the full box at its true length, whatever the zoom (T-311): the map's own style
    // draws dots below STATION_BOX_ZOOM
    const style={...this.overlay.renderer.stationStyle,boxes:true};
    const shape=stationShape(lx,ly,sh,PLATFORM_HALF_M*perM,(x,y)=>this.screen(x,y),style,false);
    this.ghost ??= this.svgMark(["path"], false);
    this.ring.style.display="none";
    this.ghost.style.display="";
    this.ghost.setAttribute("class",`station-ghost ${bad?"bad":"ok"}`);
    placeAt(this.ghost,shape.x,shape.y,0,0,false);
    this.ghost.firstElementChild!.setAttribute("d",stationPath(shape));
    let twin=this.ghost.children[1] as SVGPathElement|undefined;
    if (twinAt) {
      const ts=stationShape(twinAt.x,twinAt.y,platformShape(twinAt.ways,0,plat/2*perM),PLATFORM_HALF_M*perM,(x,y)=>this.screen(x,y),style,false);
      if (!twin) twin=this.ghost.appendChild(document.createElementNS("http://www.w3.org/2000/svg","path"));
      twin.setAttribute("d",stationPath(ts));
      twin.setAttribute("transform",`translate(${ts.x-shape.x},${ts.y-shape.y})`);
      twin.style.display="";
    } else if (twin) twin.style.display="none";
  }

  /** the last pointer position, in screen px and degrees */
  private pointer: { px: number; py: number; lng: number; lat: number } | null = null;

  /** Faint rings 1 and 2 km, or miles in US units (ground distance), round the station being placed (T-312), under the
   * ghost, labelled at their tops; 0.5 and 1 for a tram stop, spaced closer (T-3001). */
  private rings: SVGSVGElement | null = null;
  private stationRings(h: Snap) {
    const step = stationOnTram.peek() ? 0.5 : 1;
    if (!this.rings) {
      this.rings=this.svgMark(["circle","text","circle","text"],true);
      this.rings.setAttribute("class","station-rings");
    }
    const unit=circleOnScreen([h.lx,h.ly,unitMetres()],(x,y)=>this.screen(x,y),toLngLat,toLocal);
    if (!unit) return;
    const {x:cx,y:cy,r:km1}=unit;
    this.rings.style.display="";
    placeAt(this.rings,cx,cy,0,0,false);
    const kids=this.rings.children;
    for (let i=0;i<2;i++) {
      const r=km1*step*(i+1), c=kids[i*2], t=kids[i*2+1];
      c.setAttribute("r",String(r));
      t.setAttribute("y",String(-r-3));
      t.textContent=`${tNumber(step*(i+1))} ${unitShort()}`;
    }
  }

  /** Where a station would go and whether a click there is taken (T-311, T-671, T-695). `guess`
   * comes at once from the track's layout (`likelyFirst`), so the first frame is right; `answer`
   * when the worker has tried the same `addStation` the click sends, rolled back, and overrides it.
   * A refused spot stays where the guess put it. */
  private guess: Verdict | null = null;
  private answer: Verdict | null = null;
  /** what the ghost last showed (null: nothing) */
  private shown: Verdict | null = null;
  /** the worker's last answer for each track or node (null: taken) and where it was tried, for
   * this network and platform length (T-695) */
  private tried: { scope: string; at: Map<string, { issues: string[] | null; lx: number; ly: number }> } = { scope: "", at: new Map() };
  private checkGhost() {
    const h=this.hover, w=world.peek(), n=this.overlay.renderer.net;
    if (!h || h.kind==="free" || !w || !n) return;
    const key=this.altsKey(h);
    if (key===this.guess?.key) return;
    const spots=this.alts.length?this.alts:[h];
    const platform=this.platFor(h), lay=this.layout(w);
    const id=(s:Spot)=>s.kind==="node"?`n${s.node}`:`e${s.edge}`;
    const scope=`${platform}:${w.version}`;
    if (this.tried.scope!==scope) this.tried={scope,at:new Map()};
    const tried=this.tried.at;
    // the worker's answer for this track from a spot within KNOWN_PX of this one: a refusal the
    // layout cannot explain stands for this spot; a spot taken there still has the layout checked
    const known=(s:Spot)=>{
      const r=tried.get(id(s));
      if (!r) return undefined;
      const [ax,ay]=this.screen(r.lx,r.ly), [bx,by]=this.screen(s.lx,s.ly);
      return Math.hypot(ax-bx,ay-by)<=KNOWN_PX ? r.issues : undefined;
    };
    const cand=(s:Spot):Candidate=>s.kind==="node"?{kind:"node",node:s.node}:{kind:"edge",edge:s.edge,frac:fractionAlong(n.edgePts.get(s.edge)??[],s.lx,s.ly)};
    const roomOf=(s:Spot)=>this.roomSnap(s);
    const g=likelyFirst(spots,(s)=>known(s)||layoutIssues(cand(s),platform,lay),roomOf);
    let guess:Verdict={key,spot:g.spot??spots[0],ok:!!g.spot,why:g.spot?"":describeIssues(g.issues)||tr("tools.cannotPlace")};
    if (spots.length>=HOLD_AT && !(g.spot?known(g.spot)===null:spots.every((s)=>known(s)))) {
      // a bundle of tracks where the worker has not yet confirmed the guess: there it is often
      // refused for what the page cannot see, so the ghost stays as it was until the answer (if it
      // was by the pointer; else none)
      const was=this.shown, at=was?.spot;
      let near=false;
      if (at) {
        const [ax,ay]=this.screen(at.lx,at.ly), [bx,by]=this.screen(h.lx,h.ly);
        near=Math.hypot(ax-bx,ay-by)<=SNAP_EDGE_PX;
      }
      guess=near&&was ? {...was,key,held:true} : {key,spot:null,ok:true,why:"",held:true};
    }
    this.guess=guess;
    const atOf=(s:Spot)=>s.kind==="node"?{kind:"node" as const,node:s.node}:{kind:"edge" as const,edge:s.edge,...this.pack(s.lx,s.ly)};
    // a track the worker refused a few px away is not asked again, so while the pointer moves each
    // new spot's answer gets further down the list
    const trial=(s:Spot)=>{
      const k=known(s);
      return k ? Promise.resolve(k) : new Promise<string[]|null>((res)=>this.client.checkStation(atOf(s),platform,(ok,issues)=>{
        // refusals the layout explains depend on the spot; the layout finds those itself, and a split
        // refused where the layout finds no room is by a node, not on a ramp (T-1870)
        const byNode=issues.includes("SplitPoint") && layoutIssues(cand(s),platform,lay).length>0;
        if (ok || !issues.every((i)=>LAYOUT_ISSUES.has(i) || (i==="SplitPoint" && byNode))) tried.set(id(s),{issues:ok?null:issues,lx:s.lx,ly:s.ly});
        else tried.delete(id(s));
        res(ok?null:issues);
      }));
    };
    void firstAccepted(spots,trial,roomOf,()=>key!==this.guess?.key).then(({spot,issues})=>{
      if (key!==this.guess?.key) return; // a newer spot is already asked
      const why=spot?"":describeIssues(issues)||tr("tools.cannotPlace");
      this.answer={key,spot:spot??(guess.held?spots[0]:guess.spot),ok:!!spot,why};
      if (tool.peek()==="station" && (this.answer.ok!==guess.ok || this.answer.why!==guess.why || this.answer.spot!==guess.spot)) this.place();
    });
  }

  /** the track's layout as `layoutIssues` reads it, for the world it was made from, with the flat
   * crossings on each edge from the map's marks */
  private lay: (Layout & { version: number }) | null = null;
  private layout(w: NonNullable<ReturnType<typeof world.peek>>) {
    if (this.lay?.version===w.version) return this.lay;
    const net=this.overlay.renderer.net, marks=net?.marks, crossings=new Map<number, number[]>();
    const lay=layoutOf(w.edges,w.nodes,{
      crossingsOn:(e)=>{
        let out=crossings.get(e);
        if (out) return out;
        out=[];
        const pts=net?.edgePts.get(e), edge=lay.byId.get(e);
        if (pts && edge && marks) {
          // local units per metre along this edge; a crossing within 2 m of it is on it
          const total=lengthOf(pts), perM=total/Math.max(1,edge.lengthM);
          let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
          for (let k=0;k<pts.length;k+=3) x0=Math.min(x0,pts[k]),x1=Math.max(x1,pts[k]),y0=Math.min(y0,pts[k+1]),y1=Math.max(y1,pts[k+1]);
          const pad=2*perM;
          for (let k=0;k+MARK_STRIDE<=marks.length;k+=MARK_STRIDE) {
            const x=marks[k],y=marks[k+1];
            if (marks[k+2]!==M_CROSSING || x<x0-pad || x>x1+pad || y<y0-pad || y>y1+pad) continue;
            const [f,off]=nearestOn(pts,x,y);
            if (off<=pad) out.push(f*edge.lengthM);
          }
        }
        crossings.set(e,out);
        return out;
      },
    });
    return this.lay={...lay,version:w.version};
  }

  /** the spots the pointer's candidates (`alts`) are tried at, and the world and platform they were tried in */
  private altsKey(h: Spot | null = this.hover as Spot | null) {
    const spots=this.alts.length?this.alts:h?[h]:[];
    return spots.map((h)=>`${h.kind==="node"?`n${h.node}`:`e${h.edge}`}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}`).join("|")+`:${buildPlatform.peek()}:${buildTramPlatform.peek()}:${world.peek()?.version}`;
  }

  /** the worker's answer for the pointer's candidates, else the layout's guess */
  private verdict(): Verdict | null {
    const h=this.hover;
    if (!h || h.kind==="free") return null;
    const key=this.altsKey(h);
    return this.answer?.key===key ? this.answer : this.guess?.key===key ? this.guess : null;
  }

  /** Where a station would go (T-695): the worker's spot, else the guessed one. */
  private stationSpot(): Snap | null {
    const h=this.hover;
    if (!h || h.kind==="free") return h;
    const v=this.verdict();
    return v ? v.spot : h;
  }

  /** The spots near an edge spot where the platform may have room (`roomNear`, T-671, T-1870). */
  private roomSnap(h: Spot): Spot[] {
    const w=world.peek(), net=this.overlay.renderer.net, pts=h.kind==="edge"?net?.edgePts.get(h.edge):null;
    if (!w || !net || !pts || pts.length<6 || h.kind!=="edge") return [];
    return roomNear({edge:h.edge,frac:fractionAlong(pts,h.lx,h.ly)},this.platFor(h),this.layout(w)).flatMap((c):Spot[]=>{
      if (c.kind==="node") {
        const node=w.nodes.get(c.node);
        if (!node) return [];
        const [lx,ly]=toLocal(node.lng,node.lat);
        return [{kind:"node",node:c.node,lx,ly}];
      }
      const to=net.edgePts.get(c.edge);
      if (!to) return [];
      const [lx,ly]=pointAlong(to,c.frac);
      return [{kind:"edge",edge:c.edge,lx,ly}];
    });
  }

  /** The faint circle the segment being placed curves along (T-418), red when it is refused. */
  private circle: SVGSVGElement | null = null;
  private curveCircle(pv: PreviewAnswer | null) {
    const c = circleOnScreen(pv?.arc, (x, y) => this.screen(x, y), toLngLat, toLocal);
    if (!c) {
      if (this.circle) this.circle.style.display = "none";
      return;
    }
    this.circle ??= this.svgMark(["circle"], true);
    this.circle.style.display = "";
    this.circle.setAttribute("class", "curve-circle" + (pv!.ok ? "" : " bad"));
    placeAt(this.circle, c.x, c.y, 0, 0, false);
    this.circle.firstElementChild!.setAttribute("r", String(c.r));
  }

  /** T-1895: beside the circle, a short line on from the segment's end the way it arrives, where a
   * straight drawn on from there would go; for a straight segment too, except one from a free start,
   * whose line on is its own continuation (T-2008). */
  private tangent: SVGSVGElement | null = null;
  private curveTangent(pv: PreviewAnswer | null) {
    const straightFromFree = this.start?.end.kind === "free" && !(pv?.arc && pv.arc.length >= 3 && pv.arc[2] > 0);
    const h = pv && !straightFromFree ? endHeading(pv.verts, pv.pts, pv.arc, toLngLat, toLocal) : null;
    if (!h) {
      if (this.tangent) this.tangent.style.display = "none";
      return;
    }
    this.tangent ??= this.svgMark(["line"], true);
    drawTangent(this.tangent, h, pv!.ok, (x, y) => this.screen(x, y));
  }

  /** Move the DOM markers to the camera (after each MapLibre frame and each change). */
  place() {
    if (tool.peek() === "station" && this.hover && this.hover.kind !== "free") stationOnTram.value = this.tramSpot(this.stationSpot() as Spot | null);
    // on touch the pointer is a tap, a place on the map: its screen point follows the camera
    const p = this.pointer;
    if (p && touchInput.peek()) {
      const s = this.map.project([p.lng, p.lat]);
      p.px = s.x;
      p.py = s.y;
      if (this.touchAt && tool.peek() === "delete") this.delAt = [s.x, s.y];
    }
    this.placeMarks();
    if (this.pendingSt.length) this.placePending();
    this.placeTouch();
    this.drawBand();
  }

  /** T-1721: with the track tool and Snap beside on, a faint blue band BESIDE_PX either side of all
   * track, where drawing snaps beside it; redrawn when the view or the network changes. */
  private bandEl: HTMLCanvasElement | null = null;
  private bandKey = "";
  private drawBand() {
    const n = this.overlay.renderer.net;
    const on = tool.peek() === "track" && besideOn.peek() && !!n;
    if (!on) {
      if (this.bandEl) this.bandEl.style.display = "none";
      this.bandKey = "";
      return;
    }
    const c = this.map.getCenter(), W = this.overlay.cssW, H = this.overlay.cssH, dpr = devicePixelRatio || 1;
    const key = `${c.lng},${c.lat},${this.map.getZoom()},${this.map.getBearing()},${this.map.getPitch()},${W},${H},${world.peek()?.version},${document.documentElement.dataset.shade}`;
    if (this.bandEl && key === this.bandKey) return;
    this.bandKey = key;
    if (!this.bandEl) {
      this.bandEl = document.createElement("canvas");
      this.bandEl.className = "beside-band";
      this.box.prepend(this.bandEl);
    }
    const el = this.bandEl;
    el.style.display = "";
    if (el.width !== Math.round(W * dpr) || el.height !== Math.round(H * dpr)) {
      el.width = Math.round(W * dpr);
      el.height = Math.round(H * dpr);
      el.style.width = `${W}px`;
      el.style.height = `${H}px`;
    }
    const g = el.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.beginPath();
    for (const pts of n!.edgePts.values()) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      const xy: number[] = [];
      for (let k = 0; k + 2 < pts.length; k += 3) {
        const [x, y] = this.screen(pts[k], pts[k + 1]);
        xy.push(x, y);
        x0 = Math.min(x0, x), y0 = Math.min(y0, y), x1 = Math.max(x1, x), y1 = Math.max(y1, y);
      }
      if (x1 < -BESIDE_PX || y1 < -BESIDE_PX || x0 > W + BESIDE_PX || y0 > H + BESIDE_PX) continue;
      g.moveTo(xy[0], xy[1]);
      for (let k = 2; k < xy.length; k += 2) g.lineTo(xy[k], xy[k + 1]);
    }
    g.lineWidth = BESIDE_PX * 2;
    g.lineCap = g.lineJoin = "round";
    g.strokeStyle = getComputedStyle(el).getPropertyValue("--beside").trim() || "#3d82d6";
    g.globalAlpha = 0.1;
    g.stroke();
  }

  private placeMarks() {
    let k = 0;
    const put = (m: HTMLDivElement, lx: number, ly: number, dx = 0, dy = 0) => {
      const [x, y] = this.screen(lx, ly);
      placeAt(m, x, y, dx, dy);
    };
    const t = tool.peek();
    if (t === "track" && this.start) {
      const pv = this.lastPreview;
      // where the segment starts, which from a station is past its platform (T-1155)
      if (pv && pv.verts.length >= 2) put(this.el(k++, "pi-mark"), pv.verts[0], pv.verts[1]);
      else put(this.el(k++, "pi-mark"), ...toLocal(this.start.lng, this.start.lat));
      if (pv) {
        const from = k;
        k += placeCurveLabels(curveLabels(pv.verts), (x, y) => this.screen(x, y), this.overlay.cssW, this.overlay.cssH, (i) => this.el(from + i, "pi-label"));
      }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";
    this.curveCircle(t === "track" && this.start ? this.lastPreview : null);
    this.curveTangent(t === "track" && this.start ? this.lastPreview : null);

    // the platform ghost shows only while the station tool snaps (below)
    if (this.ghost) this.ghost.style.display = "none";
    if (this.rings) this.rings.style.display = "none";
    if (t === "bus") {
      this.ring.style.display = "none";
      this.bus.place();
      return;
    }
    if (t === "ferry") {
      this.ring.style.display = "none";
      this.ferry.place();
      return;
    }
    if (t === "building") {
      this.ring.style.display = "none";
      this.building.place();
      return;
    }
    // the delete tool: a ring on the station or flyover, and what a click removes by the cursor
    if (t === "delete") {
      const d = this.delHover, what = d && describeRemovable(d);
      if (d && d.kind !== "track" && d.kind !== "bus" && d.kind !== "building") {
        this.ring.className = "snap-ring";
        this.ring.style.display = "";
        put(this.ring, d.lx, d.ly);
      } else this.ring.style.display = "none";
      if (what) this.showTag(tr("tools.removeWhat", { what: what.what }), false, ...this.delAt);
      else this.tag.style.display = "none";
      return;
    }
    // the snap ring and the cursor tag
    if (t === "station") this.checkGhost();
    const h = t === "station" ? (this.ghostSpent ? null : this.stationSpot()) : this.hover;
    const v = t === "station" ? this.verdict() : null;
    if (t === "station") this.shown = h && h.kind !== "free" ? v : null;
    if (t === "station" && h?.kind === "free") {
      // T-1820: open ground takes no station; no ghost, only the reason by the pointer
      this.ring.style.display = "none";
      this.showTag(notOnTrack(), true, ...this.screen(h.lx, h.ly));
      return;
    }
    // T-1721: the track tool's point: red branching off a piece, yellow joining at a node, blue beside track
    const hb = t === "track" && h?.kind === "free" ? this.hoverBeside : null;
    this.ring.className = "snap-ring" + (t !== "track" ? "" : hb ? " beside" : h?.kind === "edge" ? " branch" : h?.kind === "node" ? " join" : "");
    if ((t === "track" || t === "station") && h && h.kind !== "free") {
      this.ring.style.display = "";
      if (t === "station") {
        this.stationRings(h);
        this.stationGhost(h);
      } else put(this.ring, ...(h.join ?? [h.lx, h.ly]));
    } else if (hb) {
      this.ring.style.display = "";
      put(this.ring, hb.lx, hb.ly);
    } else if (t === "track" && this.real.ringAt) {
      this.ring.style.display = "";
      put(this.ring, ...toLocal(...this.real.ringAt));
    } else if (t === "track" && !this.start && h?.kind === "free" && this.street.ringAt) {
      this.ring.style.display = "";
      put(this.ring, ...toLocal(...this.street.ringAt));
    } else this.ring.style.display = "none";
    const pv = this.lastPreview, rt = t === "track" && this.start && h ? this.real.tag() ?? this.street.tag() : null;
    const at = h ? this.screen(h.lx, h.ly) : null;
    if (rt) {
      // T-727d: a real route still being found (the spinner once it takes a while), or none
      const kids: (string | Node)[] = [];
      if (rt.spin) kids.push(Object.assign(document.createElement("i"), { className: "spin" }));
      if (rt.text) kids.push(rt.text);
      if (kids.length) this.showTag(kids, rt.bad, ...at!);
      else this.tag.style.display = "none";
    } else if (t === "track" && this.start && h && !this.snapped && this.tooFar(h)) {
      this.showTag(tooFarText(), true, ...at!);
    } else if (t === "track" && this.start && h && (pv || this.waiting())) {
      const rows: HTMLDivElement[] = pv ? placementTag(pv, buildLevel.peek(), !tramTrack.peek() ? TRACK_SPECS.regular : this.street.active ? TRACK_SPECS.streetTram : TRACK_SPECS.tram) : [document.createElement("div")];
      // T-1039: still working on the newest ghost or the click
      if (this.waiting()) rows[0].prepend(Object.assign(document.createElement("i"), { className: "spin row-dot" }));
      this.showTag(rows, false, ...at!);
    } else if (t === "station" && h && h.kind !== "free" && v?.ok === false) {
      // why a click here would be refused (T-311)
      this.showTag(v.why, true, ...at!);
    } else if (t === "station" && h && h.kind !== "free") {
      const lines = this.helpLines(h, null);
      if (lines.length) this.showTag(lines, false, ...at!);
      else this.tag.style.display = "none";
    } else this.tag.style.display = "none";
  }

  /** The cursor tag with `content`, red when `bad`, by the screen point (x, y). */
  private showTag(content: string | (string | Node)[], bad: boolean, x: number, y: number) {
    this.tag.style.display = "";
    this.tag.className = "tool-tag build-tag" + (bad ? " bad" : "");
    if (typeof content === "string") this.tag.textContent = content;
    else this.tag.replaceChildren(...content);
    placeAt(this.tag, x, y, 14, 12);
  }

  /** The station tool's tag lines: `first` if any, how far from the station placed last (T-312),
   * the trips near (T-412) over a bar of their size on a log scale (T-2271). */
  private helpLines(h: Snap, first: string | null): HTMLDivElement[] {
    const row = (s: string) => Object.assign(document.createElement("div"), { textContent: s });
    const lines = first ? [row(first)] : [];
    const last = this.lastPlacedName(), [lng, lat] = toLngLat(h.lx, h.ly);
    if (last) lines.push(row(tr("tools.fromLast", { distance: distance(metresBetween(lng, lat, last.lng, last.lat)), name: last.name })));
    const trips = this.tripsNear(lng, lat);
    if (trips) {
      const bar = Object.assign(document.createElement("div"), { className: "size-bar" });
      const fill = document.createElement("i");
      fill.style.width = `${(100 * logShare(trips.legs, NEAR_BAR_LO, NEAR_BAR_HI)).toFixed(1)}%`;
      bar.append(fill);
      lines.push(row(trips.text), bar);
    }
    return lines;
  }
}

function sameStops(a: readonly number[], b: readonly number[]) {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}
