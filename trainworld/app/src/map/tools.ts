// The map tools (T-023, T-024): what a click does while drawing track, placing stations or adding
// stops to a line. Selecting (picking) is map/pick.ts; main.ts routes clicks here when a tool is
// on.
//
// Track, drawn through points (T-418; the only way since T-422): the second click places a
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
// Everything drawn is blueprint until constructed (SPEC 6.4).

import { effect } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { count, distance, length, money, moneyShort, speed, unitMetres, unitShort } from "../game/format";
import { activeCity, getWorldFrame, worldFrameEpoch } from "../game/cities";
import { cellModes, cityCells, demandView } from "../game/demand";
import { speedColour } from "../game/palette";
import { metresBetween } from "../game/walks";
import { ask, buildLevel, buildPlatform, designSpeed, highSpeed, keepPlacing, lineDraft, selection, shownDayPhrase, trackCount, tool } from "../game/ui";
import { describeRemovable, removeThing, type Removable } from "../game/remove";
import { loadPlacesAt } from "../game/placeList";
import { newStationName, trackDirection, warmStreetNames } from "./stationNames";
import { pointerLocal } from "./pick";
import { edgesNear, nodesNear } from "./snapIndex";
import { edit, say, world } from "../game/world";
import type { RouteEnd, StationAt } from "../workers/protocol";
import type { ClockClient, PreviewAnswer } from "../workers/clockClient";
import { toLngLat, toLocal } from "./geo";
import { circleOnScreen } from "./curveCircle";
import { strokesFromParts, strokesFromPreview } from "./network";
import { pairBeside, PLATFORM_HALF_M, platformShape, reversed, stationPath, stationShape, waysFrom } from "./stationGeometry";
import type { Overlay } from "./overlay";
import type { StopMarkers, StopTier } from "./stopMarkers";
import { chip, elevationFact, factRow, speedFact } from "./trackTagRows";
import { firstAccepted, fractionAlong, layoutIssues, layoutOf, likelyFirst, nearestOn, orderOverlapping, type Candidate, type Layout } from "./stationSnap";
import { M_CROSSING, MARK_STRIDE } from "./markData";
import { RealSnap, type RealRoute } from "./realSnap";
import { snapping } from "../game/ui";

const SNAP_NODE_PX = 12;
const SNAP_EDGE_PX = 10;
/** tracks this much further from the pointer than the nearest still count as overlapping it (T-469) */
const OVERLAP_PX = 4;

type Snap =
  | { kind: "node"; node: number; lx: number; ly: number }
  | { kind: "edge"; edge: number; lx: number; ly: number }
  | { kind: "free"; lx: number; ly: number };

type Spot = Exclude<Snap, { kind: "free" }>;

/** where a station would go for the candidates `key` (null: no ghost), and whether a click there
 * is taken; `held`: the ghost as it was, until the worker answers */
interface Verdict { key: string; spot: Spot | null; ok: boolean; why: string; held?: boolean }

/** this many candidates under the pointer, not yet tried, and the ghost waits for the worker (T-695) */
const HOLD_AT = 7;
/** the worker's answer for a track tried this close to the pointer's spot stands for it (T-695) */
const KNOWN_PX = 16;
const LAYOUT_ISSUES = new Set(["PlatformTooLong", "PlatformJoint", "PlatformCrossing", "StationPorts"]);
/** a station on open ground's own track runs this far past each platform end (T-794): the track
 * model wants a metre clear, and room for a later route to join without a short piece at the end */
const FREE_MARGIN_M = 10;
/** the express pair's spacing (sim `QUAD_SPACING`) */
const QUAD_SPACING_M = 8;
const EARTH_R = 6371008.8;

const lengthOf = (pts: ArrayLike<number>) => {
  let l = 0;
  for (let k = 0; k + 5 < pts.length; k += 3) l += Math.hypot(pts[k + 3] - pts[k], pts[k + 4] - pts[k + 1]);
  return l;
};

interface Placed {
  end: RouteEnd;
  lx: number;
  ly: number;
}

export class Tools {
  private start: Placed | null = null;
  /** the starts of the segments placed with Shift, for Backspace (T-418) */
  private chain: Placed[] = [];
  /** a segment's edit in flight: clicks wait for it */
  private placing = false;
  /** a double click's second click came while the segment was still being placed (T-445) */
  private endRun = false;
  private hover: Snap | null = null;
  /** the delete tool: what a click would remove, and the pointer (map px) */
  private delHover: (Removable & { lx: number; ly: number }) | null = null;
  private delAt: [number, number] = [0, 0];
  private lastPreview: PreviewAnswer | null = null;
  /** snapping to the real network (T-727d) */
  readonly real: RealSnap;
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
  private stopClicks: Promise<void> = Promise.resolve();

  constructor(private map: MlMap, private overlay: Overlay, private client: ClockClient) {
    this.box = document.createElement("div");
    this.box.className = "tool-marks";
    map.getContainer().appendChild(this.box);
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag build-tag";
    this.tag.style.display = "none";
    this.ring =document.createElement("div");
    this.ring.className = "snap-ring";
    this.box.append(this.ring);
    // outside the marks' layer, so no demand tag covers it (T-526)
    map.getContainer().appendChild(this.tag);
    overlay.afterMapFrame.push(() => this.place());
    effect(() => { buildPlatform.value; buildLevel.value; trackCount.value; highSpeed.value; if (tool.value === "station") this.place(); else { this.freeAt = null; this.clearFree(); } });
    // the place lists name stations where the tiles have no streets (T-065); the view's zones' are
    // fetched when the station tool is picked, not at startup (T-400)
    effect(() => { if (tool.value === "station") { const c = this.map.getCenter(); void loadPlacesAt(c.lng, c.lat); warmStreetNames(this.map); } });
    map.on("idle", () => tool.peek() === "station" && warmStreetNames(map));
    effect(() => { buildLevel.value; trackCount.value; highSpeed.value; designSpeed.value; if (tool.peek() === "track" && this.start) this.real?.active ? this.real.again() : this.requestPreview(); });
    // T-727d: the snap switch's routes along the real network
    this.real = new RealSnap(map, overlay, client, (r) => this.realReq(r), (a, req) => {
      if (a) return this.start && this.showPreview(a, req ? JSON.stringify(req) : "");
      this.lastPreview = null;
      this.overlay.renderer.setPreview(null);
      this.overlay.invalidate();
      if (!this.real.active) this.requestPreview();
      this.place();
    }, () => this.place());
    effect(() => this.real.setOn(snapping.value));
    effect(() => {
      const active = tool.value === "line" ? lineDraft.value.line : null;
      const w = world.value;
      if (active !== this.editingLine) {
        this.editingLine = active;
        this.lineSession++;
      }
      if (active === null || !w) return;
      const line = w.save.lines.find((l) => Number(l.id) === active);
      if (!line) {
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
    if (!w || !n) return { kind: "free", lx: fx, ly: fy };
    let best = SNAP_NODE_PX * SNAP_NODE_PX, hit: Snap | null = null;
    const nodeHits: { d: number; hit: Spot }[] = [];
    // only what is near the pointer (T-724); everything with the map pitched
    const at = pointerLocal(this.overlay, px, py);
    const nearNodes = at && nodes ? nodesNear(w, at[0], at[1], SNAP_NODE_PX * at[2]) : null;
    const nearEdges = at ? edgesNear(n, at[0], at[1], SNAP_EDGE_PX * at[2]) : null;
    for (const node of nodes ? nearNodes ?? w.nodes.values() : []) {
      const [lx, ly] = toLocal(node.lng, node.lat);
      const [x, y] = this.screen(lx, ly);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < SNAP_NODE_PX * SNAP_NODE_PX && station) nodeHits.push({ d, hit: { kind: "node", node: node.id, lx, ly } });
      if (d < best) {
        best = d;
        hit = { kind: "node", node: node.id, lx, ly };
      }
    }
    if (hit && !station) return hit;
    // each edge's nearest point within reach, px squared
    const near: { d: number; hit: Snap }[] = [];
    for (const { edge, pts, ks } of nearEdges ?? [...n.edgePts].map(([edge, pts]) => ({ edge, pts, ks: null }))) {
      let eb = SNAP_EDGE_PX * SNAP_EDGE_PX, eh: Snap | null = null;
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
    if (!near.length && !nodeHits.length) return { kind: "free", lx: fx, ly: fy };
    near.sort((a, b) => a.d - b.d);
    if (!station) return near[0].hit;
    // T-469, T-671: what a click can place, in the order it is tried: the nodes in reach, then the
    // tracks within 4 px of the nearest in orderOverlapping's order, then the rest by distance
    nodeHits.sort((a, b) => a.d - b.d);
    const close = near.filter((c) => c.d <= near[0].d + OVERLAP_PX);
    const spots = close.map((c) => { const h = c.hit as Extract<Snap, { kind: "edge" }>; return { edge: h.edge, frac: fractionAlong(n.edgePts.get(h.edge)!, h.lx, h.ly) }; });
    const order = close.length > 1 ? orderOverlapping(spots, buildPlatform.peek(), w.edges, w.nodes, this.lastStationTrack(), this.layout(w)) : close.map((_, i) => i);
    this.alts = [...nodeHits.slice(0, 3).map((c) => c.hit), ...order.map((i) => close[i].hit as Spot), ...near.slice(close.length, 5).map((c) => c.hit as Spot)];
    return this.alts[0];
  }

  /** what the station tool would try at the pointer, in order (T-671) */
  private alts: Spot[] = [];

  /** The edges of the track the last station placed this session is on (T-469): from its node
   * along the track both ways, through plain joints and stations, up to junctions. */
  private lastStationTrack(): Set<number> {
    const out = new Set<number>();
    const w = world.peek();
    const at = this.lastPlaced;
    if (!w || !at) return out;
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
    return { kind: "free", x: p.x, y: p.y, level: buildLevel.peek() };
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "select") return;
    if (t === "delete") return this.deleteHover(e.point.x, e.point.y);
    if (t === "line") this.stops?.hover(e.point.x, e.point.y); // T-233: the toggle markers' tag
    this.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
    this.hover = t === "line" ? null : this.snap(e.point.x, e.point.y, e.lngLat);
    if (t === "track") this.real.move(e.point.x, e.point.y);
    if (t === "track" && this.start) this.requestPreview();
    if (t === "station") this.nameAhead(e.point.x, e.point.y);
    this.place();
  }

  click(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") this.trackClick(e);
    else if (t === "station") void this.stationClick(e);
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") this.deleteClick(e);
  }

  /** Keys while a tool is on; true if used. */
  key(e: KeyboardEvent): boolean {
    const t = tool.peek();
    if (t === "select") return false;
    if (e.key === "Escape") {
      if (t === "track" && this.start) this.reset();
      else if (t === "station" && this.freeAt) {
        this.freeAt = null;
        this.place();
      } else this.exit();
      return true;
    }
    if (t === "track" && this.start) {
      if (e.key === "Backspace") {
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
    this.start = null;
    this.chain = [];
    this.delHover = null;
    this.lastPreview = null;
    this.previewKey = "";
    this.real.reset();
    this.freeAt = null;
    this.freeDrawn = "";
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
    this.place();
  }

  // ---------------------------------------------------------------- track

  private trackClick(e: MapMouseEvent) {
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
      this.start = { end: this.endOf(s), lx: s.lx, ly: s.ly };
      this.real.begin(ring);
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
    const [sx, sy] = this.screen(from.lx, from.ly), se = this.endOf(s);
    if (Math.hypot(sx - e.point.x, sy - e.point.y) < 6 || (from.end.kind === "node" && se.kind === "node" && se.node === from.end.node)) {
      this.reset();
      return;
    }
    const chain = e.originalEvent.shiftKey || keepPlacing.peek();
    this.placing = true;
    this.endRun = false;
    // T-727d: near real track at both clicks, the real route (waited for if still being found)
    const rc = await this.real.click(e.point.x, e.point.y);
    if (rc?.kind === "refuse" || this.start !== from) {
      this.placing = false;
      if (rc?.kind === "refuse") say("No real track joins these points.", true);
      return;
    }
    const req = rc ? this.realReq(rc.route, s) : this.routeReq(this.endOf(s));
    // T-830: the quote of this very request; an answer for an earlier pointer position (a slow
    // preview landing late) would give another segment's length and cost
    const p = this.previewKey === JSON.stringify(req) ? this.lastPreview : null;
    const a = await edit({ op: "route", ...req });
    this.placing = false;
    if (!a.ok || this.start !== from) return;
    say(p ?`Blueprint added: ${distance(p.lengthM)}, ${money(p.cost * 1e6)} to construct.` : "Blueprint added.");
    if (!chain || this.endRun || a.node === undefined) {
      this.reset();
      return;
    }
    this.chain.push(from);
    const [ex, ey] = rc && s.kind === "free" ? toLocal(...rc.route.last) : [s.lx, s.ly];
    this.start = { end: { kind: "node", node: a.node }, lx: ex, ly: ey };
    this.real.begin(rc ? rc.route.last : snapping.peek() ? this.real.ring(e.point.x, e.point.y) : null);
    this.lastPreview = null;
    this.previewKey = "";
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
    this.requestPreview();
    this.place();
  }

  /** Backspace while chaining: the last segment undone, drawing on from its start again. */
  private async takeBackSegment() {
    const prev = this.chain.pop();
    if (!prev || this.placing) {
      this.reset();
      return;
    }
    this.placing = true;
    const a = await edit({ op: "undo" }, true);
    this.placing = false;
    if (!a.ok) return this.reset();
    this.start = prev;
    this.lastPreview = null;
    this.previewKey = "";
    this.requestPreview();
    this.place();
  }

  private routeReq(to: RouteEnd) {
    return { from: this.start!.end, to, pis: [], single: false, quad: trackCount.peek() === 4, through: true, design: designSpeed.peek(), hsr: highSpeed.peek() };
  }

  /** A real route's request (T-727d): from the start, or on open ground from the route's first
   * point; to the player's own track under the pointer, else the route's last point. */
  private realReq(r: RealRoute, at: Snap | null = this.hover) {
    const free = ([lng, lat]: [number, number]): RouteEnd => ({ kind: "free", ...lngLatToPack(lng, lat), level: buildLevel.peek() });
    const se = this.start!.end;
    return { from: se.kind === "free" ? free(r.first) : se, to: at && at.kind !== "free" ? this.endOf(at) : free(r.last), pis: r.pis, single: false, quad: trackCount.peek() === 4, through: true, design: designSpeed.peek(), hsr: highSpeed.peek() };
  }

  private requestPreview() {
    if (!this.start || this.real.active) return;
    const se = this.start.end;
    if (se.kind === "node" && !world.peek()?.nodes.has(se.node)) return this.reset(); // undone meanwhile
    const h = this.hover;
    if (!h) return;
    // Nothing to show while the pointer is on the start.
    const [ax, ay] = this.screen(this.start.lx, this.start.ly), [bx, by] = this.screen(h.lx, h.ly);
    if (Math.hypot(ax - bx, ay - by) < 3) return;
    const req = this.routeReq(this.endOf(h));
    const key = JSON.stringify(req);
    this.wantKey = key;
    this.client.preview(req, (a) => {
      if (this.start && !this.real.active && this.wantKey === key) this.showPreview(a, key);
    });
  }

  private showPreview(a: PreviewAnswer, key: string) {
    this.lastPreview = a;
    this.previewKey = key;
    // single track previews as one track close in, double and quad as two each (T-658)
    this.overlay.renderer.setPreview(a.pts.length >= 6 ? strokesFromPreview(a.pts, !a.ok, a.speeds, true, a.marks, 255, highSpeed.peek() && trackCount.peek() === 2) : null);
    this.overlay.invalidate();
    this.place();
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
    if (this.freeAt) return this.freeStationClick(e);
    const first = this.snap(e.point.x, e.point.y, e.lngLat);
    if (first.kind === "free") {
      // T-907: open ground, the two clicks are the platform's two ends
      this.freeAt = first;
      this.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
      this.place();
      return;
    }
    // T-671, T-695: the spot the ghost shows, else the first the track model takes of what is under the pointer
    const v = this.verdict();
    const tries = [...(v?.ok && v.spot && !v.held ? [v.spot] : []), ...this.alts];
    const key = this.nameKey(e.point.x, e.point.y);
    const ahead = this.named?.key === key && this.map.areTilesLoaded() ? this.named.name : null;
    this.named = null;
    const name = await (ahead ?? this.nameFor(e.point.x, e.point.y));
    const platform = buildPlatform.peek();
    const stationAt = (p: Exclude<Snap, { kind: "free" }>) => ({ op: "addStation" as const, at: p.kind === "node" ? { kind: "node" as const, node: p.node } : { kind: "edge" as const, edge: p.edge, ...this.pack(p.lx, p.ly) }, platform, name });
    const r = await firstAccepted(tries, async (p) => { const a = await edit(stationAt(p), true); return a.ok ? null : a.issues; }, (p) => (p.kind === "edge" ? this.endSnap(p) : null), () => false);
    if (r.spot) {
      this.lastPlaced = toLngLat(r.spot.lx, r.spot.ly); // T-312: the hover tag's "km from" (found again by position)
      say(`${name} added as a blueprint station.`);
    } else if (r.issues.length) say(describeIssues(r.issues), true);
  }

  /** T-794, T-834: the second click on open ground, the station and its own piece of track, one edit. */
  private async freeStationClick(e: MapMouseEvent) {
    const h = this.freeAt!;
    this.pointer = { px: e.point.x, py: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat };
    const { phi, plat } = this.freeSpan(h);
    const p = this.freePiece(h, phi, plat);
    this.freeAt = null;
    this.lastHeading = phi;
    this.clearFree();
    this.place();
    const name = await this.nameFor(...this.screen(p.mid.lx, p.mid.ly));
    const a = await edit({ op: "addStation", at: p.stationAt, platform: plat, name }, true);
    if (a.ok) {
      this.lastPlaced = toLngLat(p.mid.lx, p.mid.ly);
      say(`${name} added as a blueprint station.`);
    } else say(describeIssues(a.issues), true);
  }

  /** Local trip legs a day per cell, homes and destinations together (T-412), for the demand shown. */
  private legs: { key: string; xy: Float32Array; legs: Float32Array } | null = null;
  private legsAsked = "";

  /** "About 12k trips a day start or end within 800 m" of the spot (T-412), from the cells behind
   * the Homes and Destinations bubbles; null until they are in. A trip with both ends inside counts
   * twice, and a spot near a zone's edge counts the viewed zone's cells only. */
  private tripsNear(lng: number, lat: number): string | null {
    const dv = demandView.peek();
    if (!dv) return null;
    const key = `${dv.version}:${dv.round}:${dv.day}:${activeCity.peek()}:${worldFrameEpoch.peek()}`;
    if (this.legs?.key !== key && this.legsAsked !== key) {
      this.legsAsked = key;
      void Promise.all([cityCells(), cellModes("all")]).then(([cells, m]) => {
        if (this.legsAsked !== key) return;
        if (!cells || !m) {
          setTimeout(() => { if (this.legsAsked === key) this.legsAsked = ""; }, 2000); // asked again then
          return;
        }
        const legs = new Float32Array(cells.n);
        for (const a of [m.homeRail, m.homeWalk, m.homeDrive, m.workRail, m.workWalk, m.workDrive])
          if (a.length === cells.n) for (let i = 0; i < cells.n; i++) legs[i] += a[i];
        this.legs = { key, xy: cells.xy, legs };
        if (tool.peek() === "station") this.place();
      });
    }
    const L = this.legs;
    if (!L) return null;
    // the radius in world metres each way, so any frame's scale is right here
    const f = getWorldFrame(), r = unitMetres() === 1000 ? TRIPS_RADIUS_M : M_PER_HALF_MILE;
    const [x, y] = f.fromLonLat(lng, lat);
    const dLat = (r / 6371008.8) * (180 / Math.PI), dLng = dLat / Math.cos((lat * Math.PI) / 180);
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
    const n = sum < 50 ? null : r2 >= 10000 ? count(r2) : r2.toLocaleString("en-US");
    const within = unitMetres() === 1000 ? `${TRIPS_RADIUS_M} m` : "half a mile";
    const when = shownDayPhrase.peek();
    return n ? `About ${n} trips ${when} start or end within ${within}` : `Under 50 trips ${when} start or end within ${within}`;
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
   * track (lines are removed from the Lines tab). */
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
    if (hit) return hit;
    for (const node of w.nodes.values()) {
      if (!node.flying || node.ports < 3) continue;
      const [lx, ly] = toLocal(node.lng, node.lat);
      const d = near(lx, ly, 10);
      if (d < best) [best, hit] = [d, { kind: "flyover", node: node.id, lx, ly }];
    }
    if (hit) return hit;
    const s = this.snap(px, py, this.map.unproject([px, py]));
    if (s.kind === "edge") return { kind: "track", edge: s.edge, lx: s.lx, ly: s.ly };
    if (s.kind === "node") {
      // a plain node (track end, junction): the stretch nearest to the pointer there
      let e = -1, bd = Infinity;
      for (const [edge, pts] of n.edgePts)
        for (let k = 0; k < pts.length; k += 3) {
          const d = near(pts[k], pts[k + 1], 14);
          if (d < bd) [bd, e] = [d, edge];
        }
      if (e >= 0) return { kind: "track", edge: e, lx: s.lx, ly: s.ly };
    }
    return null;
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
      this.overlay.renderer.setPreview(parts.length ? strokesFromParts(parts, true, false) : null);
      this.overlay.invalidate();
    }
    this.map.getCanvas().style.cursor = h ? "pointer" : "crosshair";
    this.place();
  }

  private deleteClick(e: MapMouseEvent) {
    const h = this.deletePick(e.point.x, e.point.y);
    if (!h) {
      say("Click track, a station or a flyover to remove it.");
      return;
    }
    void removeThing(h).then(() => this.deleteHover(e.point.x, e.point.y));
  }

  // ---------------------------------------------------------------- lines

  private lineClick(e: MapMouseEvent) {
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
      say("Click a circle to add or remove a stop. Press Enter when the line is done.");
      return;
    }
    this.toggleStop(hit);
  }

  /** Add the station as a stop, or take it out if the line stops there (T-233); on the first stop,
   * close the line into a loop or open it when its marker says so (`tier`, T-798). Clicks queue and
   * each reads the latest stops. */
  toggleStop(station: number, tier?: StopTier) {
    const session = this.lineSession;
    this.stopClicks = this.stopClicks.then(() => this.toggle(station, session, tier)).catch(() => {
      if (session === this.lineSession) say("Could not change that stop. Please try again.", true);
    });
  }

  private async toggle(station: number, session: number, tier?: StopTier) {
    if (session !== this.lineSession || tool.peek() !== "line") return;
    const id = lineDraft.peek().line;
    const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
    if (id === null || !line) return;
    const before = line.stops.map(Number);
    if (!before.includes(station)) return this.addStop(station, session);
    const n = before.length;
    const closedLoop = n >= 3 && before[0] === before[n - 1];
    // a loop's first stop opens it; a first stop the marker offers to close closes it (T-798)
    const stops = station === before[0] && closedLoop ? before.slice(0, -1)
      : station === before[0] && tier === "close" && !closedLoop ? [...before, station]
      // taking a stop out: the route between its neighbours is the worker's, as the inspector's crosses
      : before.filter((s) => s !== station);
    const a = await edit({ op: "setStops", line: id, stops });
    if (a.ok && session === this.lineSession && tool.peek() === "line") {
      const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: latest.stops.map(Number) };
    }
  }

  private async addStop(station: number, session: number) {
    if (session !== this.lineSession || tool.peek() !== "line") return;
    const id = lineDraft.peek().line;
    if (id === null) return; // The Lines pane creates the line before entering this tool.
    const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
    if (!line) return;
    if (line.stops.map(Number).includes(station)) {
      say("That station is already a stop on this line.");
      return;
    }
    // the worker fits it in among the stops it holds then (T-648), so a queued click never works
    // from stops an earlier one changed
    const a = await edit({ op: "insertStop", line: id, station });
    if (a.ok && session === this.lineSession && tool.peek() === "line") {
      const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: latest.stops.map(Number) };
    }
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
    // on quad track the pair beside gets its platform too, as the track model places it (T-708)
    const dir=ways.find((p)=>p.length>=6);
    const beside=dir?pairBeside(this.tracksNear(h.lx,h.ly,12*perM,candidates),h.lx,h.ly,dir,perM):null;
    const twin=beside?{x:beside.x,y:beside.y,ways:waysFrom(n.edgePts.get(beside.edge)!,beside.x,beside.y)}:null;
    this.boxGhost(h.lx,h.ly,ways,perM,this.verdict()?.ok===false,twin);
  }

  /** The platform box at (lx, ly) along `ways` (the track from the spot each way), and the twin's
   * on quad. */
  private boxGhost(lx: number, ly: number, ways: ArrayLike<number>[], perM: number, bad: boolean, twinAt: { x: number; y: number; ways: ArrayLike<number>[] } | null, plat = buildPlatform.peek()) {
    const sh=platformShape(ways,0,plat/2*perM);
    // always the full box at its true length, whatever the zoom (T-311): the map's own style
    // draws dots below STATION_BOX_ZOOM
    const style={...this.overlay.renderer.stationStyle,boxes:true};
    const shape=stationShape(lx,ly,sh,PLATFORM_HALF_M*perM,(x,y)=>this.screen(x,y),style,false);
    if (!this.ghost) {
      this.ghost=document.createElementNS("http://www.w3.org/2000/svg","svg");
      this.ghost.setAttribute("width","1");
      this.ghost.setAttribute("height","1");
      this.ghost.appendChild(document.createElementNS("http://www.w3.org/2000/svg","path"));
      this.box.appendChild(this.ghost);
    }
    this.ring.style.display="none";
    this.ghost.style.display="";
    this.ghost.setAttribute("class",`station-ghost ${bad?"bad":"ok"}`);
    this.ghost.style.transform=`translate(${shape.x}px,${shape.y}px)`;
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

  /** T-794, T-834, T-907: a station on open ground lays a straight piece of double track (quad with
   * the tool's quad) at the selected elevation under it, FREE_MARGIN_M longer than the platform at
   * each end. The first click is one end of the platform, which points towards the pointer. */
  private freeAt: Extract<Snap, { kind: "free" }> | null = null;
  /** the heading of the last station placed on open ground, used while the pointer is on the first end */
  private lastHeading = 0;
  private pointer: { px: number; py: number; lng: number; lat: number } | null = null;
  /** The heading from the first end towards the pointer (radians from east, ground), and the set
   * platform length (T-917). */
  private freeSpan(a: Extract<Snap, { kind: "free" }>): { phi: number; plat: number } {
    const p = this.pointer, plat = buildPlatform.peek();
    if (!p) return { phi: this.lastHeading, plat };
    const [sx, sy] = this.screen(a.lx, a.ly);
    if (Math.hypot(p.px - sx, p.py - sy) < 8) return { phi: this.lastHeading, plat };
    const [lng0, lat0] = toLngLat(a.lx, a.ly);
    return { phi: Math.atan2(p.lat - lat0, (p.lng - lng0) * Math.cos((lat0 * Math.PI) / 180)), plat };
  }
  private freePiece(h: Extract<Snap, { kind: "free" }>, phi: number, plat: number) {
    const w = world.peek();
    const [lng0, lat0] = toLngLat(h.lx, h.ly);
    const cos = Math.cos((lat0 * Math.PI) / 180), deg = 180 / Math.PI;
    const level = buildLevel.peek(), quad = trackCount.peek() === 4, z = level * 8;
    // ground metres east and north of the first end to lng, lat
    const at = (e: number, n: number): [number, number] => [lng0 + (e / (EARTH_R * cos)) * deg, lat0 + (n / EARTH_R) * deg];
    const [ce, sn] = [Math.cos(phi), Math.sin(phi)];
    const s0 = -FREE_MARGIN_M, s1 = plat + FREE_MARGIN_M, sm = plat / 2;
    const a = at(s0 * ce, s0 * sn), b = at(s1 * ce, s1 * sn);
    const loc = (p: [number, number]) => toLocal(p[0], p[1]);
    const [ax, ay] = loc(a), [bx, by] = loc(b), [mx, my] = loc(at(sm * ce, sm * sn));
    const track = [new Float32Array([ax, ay, z, bx, by, z])];
    const ways = [new Float32Array([mx, my, z, bx, by, z]), new Float32Array([mx, my, z, ax, ay, z])];
    let twin: { x: number; y: number; ways: ArrayLike<number>[] } | null = null;
    if (quad) {
      // the express pair QUAD_SPACING_M to the left, as the track model lays it (T-255)
      const [ox, on] = [-sn * QUAD_SPACING_M, ce * QUAD_SPACING_M];
      const [qax, qay] = loc(at(s0 * ce + ox, s0 * sn + on)), [qbx, qby] = loc(at(s1 * ce + ox, s1 * sn + on)), [tx, ty] = loc(at(sm * ce + ox, sm * sn + on));
      track.push(new Float32Array([qax, qay, z, qbx, qby, z]));
      twin = { x: tx, y: ty, ways: [new Float32Array([tx, ty, z, qbx, qby, z]), new Float32Array([tx, ty, z, qax, qay, z])] };
    }
    const pa = lngLatToPack(...a), pb = lngLatToPack(...b);
    const stationAt: StationAt = { kind: "free", x0: pa.x, y0: pa.y, x1: pb.x, y1: pb.y, level, quad, hsr: highSpeed.peek() };
    const mid: Extract<Snap, { kind: "free" }> = { kind: "free", lx: mx, ly: my };
    return { stationAt, plat, mid, track, ways, twin, perM: 1 / cos, lengthM: s1 - s0, key: `${JSON.stringify(stationAt)}:${plat}:${w?.version}` };
  }

  /** the worker's answer for the last free piece asked about (T-794) */
  private freeAns: { key: string; ok: boolean; why: string; cost: number } | null = null;
  private freeAsked = "";
  /** the free piece's track drawn as the preview (its key), so it is set once, not every frame */
  private freeDrawn = "";

  private freeGhost(h: Extract<Snap, { kind: "free" }>): { ok: boolean | null; why: string; cost: number; lengthM: number; plat: number; mid: Extract<Snap, { kind: "free" }> } {
    const { phi, plat } = this.freeSpan(h);
    const p = this.freePiece(h, phi, plat);
    if (this.freeAsked !== p.key) {
      this.freeAsked = p.key;
      this.client.checkStation(p.stationAt, plat, (ok, issues, cost) => {
        this.freeAns = { key: p.key, ok, why: ok ? "" : describeIssues(issues) || "Cannot be built here.", cost };
        if (tool.peek() === "station") this.place();
      });
    }
    const ans = this.freeAns?.key === p.key ? this.freeAns : null;
    const bad = ans?.ok === false;
    const drawn = `${p.key}:${bad}`;
    if (this.freeDrawn !== drawn) {
      this.freeDrawn = drawn;
      this.overlay.renderer.setPreview(strokesFromParts(p.track, bad, true));
      this.overlay.invalidate();
    }
    this.boxGhost(p.mid.lx, p.mid.ly, p.ways, p.perM, bad, p.twin, plat);
    return { ok: ans ? ans.ok : null, why: ans?.why ?? "", cost: ans?.cost ?? 0, lengthM: p.lengthM, plat, mid: p.mid };
  }

  /** Take the free piece's track off the preview, if it is there. */
  private clearFree() {
    if (!this.freeDrawn) return;
    this.freeDrawn = "";
    this.overlay.renderer.setPreview(null);
    this.overlay.invalidate();
  }

  /** The drawn tracks whose bounds come within `pad` of (x, y), but `skip`: the twin platform's
   * candidates, bounds kept for one network version. */
  private boxes: { version: number; box: Map<number, [number, number, number, number]> } | null = null;
  private tracksNear(x: number, y: number, pad: number, skip: number[]): [number, ArrayLike<number>][] {
    const n=this.overlay.renderer.net, v=world.peek()?.version ?? -1;
    if (!n) return [];
    if (this.boxes?.version!==v) {
      const box=new Map<number,[number,number,number,number]>();
      for (const [id,p] of n.edgePts) {
        let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
        for (let k=0;k<p.length;k+=3) x0=Math.min(x0,p[k]),x1=Math.max(x1,p[k]),y0=Math.min(y0,p[k+1]),y1=Math.max(y1,p[k+1]);
        box.set(id,[x0,y0,x1,y1]);
      }
      this.boxes={version:v,box};
    }
    const out:[number,ArrayLike<number>][]=[];
    for (const [id,[x0,y0,x1,y1]] of this.boxes.box) {
      if (skip.includes(id) || x<x0-pad || x>x1+pad || y<y0-pad || y>y1+pad) continue;
      const p=n.edgePts.get(id);
      if (p) out.push([id,p]);
    }
    return out;
  }

  /** Faint rings 1 and 2 km, or miles in US units (ground distance), round the station being placed (T-312), under the
   * ghost, labelled at their tops. */
  private rings: SVGSVGElement | null = null;
  private stationRings(h: Snap) {
    const NS="http://www.w3.org/2000/svg";
    if (!this.rings) {
      const r=this.rings=document.createElementNS(NS,"svg");
      r.setAttribute("width","1");
      r.setAttribute("height","1");
      r.setAttribute("class","station-rings");
      for (let i=0;i<2;i++) {
        r.appendChild(document.createElementNS(NS,"circle"));
        r.appendChild(document.createElementNS(NS,"text"));
      }
      this.box.insertBefore(r,this.box.firstChild);
    }
    const [lng,lat]=toLngLat(h.lx,h.ly);
    const [cx,cy]=this.screen(h.lx,h.ly);
    const [ex,ey]=this.screen(...toLocal(lng+unitMetres()/(6371008.8*Math.cos(lat*Math.PI/180))*180/Math.PI,lat));
    const km1=Math.hypot(ex-cx,ey-cy);
    this.rings.style.display="";
    this.rings.style.transform=`translate(${cx}px,${cy}px)`;
    const kids=this.rings.children;
    for (let i=0;i<2;i++) {
      const r=km1*(i+1), c=kids[i*2], t=kids[i*2+1];
      c.setAttribute("r",String(r));
      t.setAttribute("y",String(-r-3));
      t.textContent=`${i+1} ${unitShort()}`;
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
    const platform=buildPlatform.peek(), lay=this.layout(w);
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
    const endOf=(s:Spot)=>s.kind==="edge"?this.endSnap(s):null;
    const g=likelyFirst(spots,(s)=>known(s)||layoutIssues(cand(s),platform,lay),endOf);
    let guess:Verdict={key,spot:g.spot??spots[0],ok:!!g.spot,why:g.spot?"":describeIssues(g.issues)||"Cannot be built here."};
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
        // refusals the layout explains depend on the spot; the layout finds those itself
        if (ok || !issues.every((i)=>LAYOUT_ISSUES.has(i))) tried.set(id(s),{issues:ok?null:issues,lx:s.lx,ly:s.ly});
        else tried.delete(id(s));
        res(ok?null:issues);
      }));
    };
    void firstAccepted(spots,trial,endOf,()=>key!==this.guess?.key).then(({spot,issues})=>{
      if (key!==this.guess?.key) return; // a newer spot is already asked
      const why=spot?"":describeIssues(issues)||"Cannot be built here.";
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
    return spots.map((h)=>`${h.kind==="node"?`n${h.node}`:`e${h.edge}`}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}`).join("|")+`:${buildPlatform.peek()}:${world.peek()?.version}`;
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

  /** The node at the nearer end of the track under an edge spot (T-671). */
  private endSnap(h: Extract<Snap,{kind:"edge"}>): Extract<Snap,{kind:"node"}> | null {
    const w=world.peek(), pts=this.overlay.renderer.net?.edgePts.get(h.edge);
    const e=w?.edges.find((x)=>x.id===h.edge);
    if (!w || !e || !pts || pts.length<6) return null;
    const id=fractionAlong(pts,h.lx,h.ly)<0.5?e.a:e.b, node=w.nodes.get(id);
    if (!node) return null;
    const [lx,ly]=toLocal(node.lng,node.lat);
    return {kind:"node",node:id,lx,ly};
  }

  /** The faint circle the segment being placed curves along (T-418), red when it is refused. */
  private circle: SVGSVGElement | null = null;
  private curveCircle(pv: PreviewAnswer | null) {
    const c = circleOnScreen(pv?.arc, (x, y) => this.screen(x, y), toLngLat, toLocal);
    if (!c) {
      if (this.circle) this.circle.style.display = "none";
      return;
    }
    const NS = "http://www.w3.org/2000/svg";
    if (!this.circle) {
      this.circle = document.createElementNS(NS, "svg");
      this.circle.setAttribute("width", "1");
      this.circle.setAttribute("height", "1");
      this.circle.appendChild(document.createElementNS(NS, "circle"));
      this.box.insertBefore(this.circle, this.box.firstChild);
    }
    this.circle.style.display = "";
    this.circle.setAttribute("class", "curve-circle" + (pv!.ok ? "" : " bad"));
    this.circle.style.transform = `translate(${c.x}px,${c.y}px)`;
    this.circle.firstElementChild!.setAttribute("r", String(c.r));
  }

  /** Move the DOM markers to the camera (after each MapLibre frame and each change). */
  place() {
    let k = 0;
    const put = (m: HTMLDivElement, lx: number, ly: number, dx = 0, dy = 0) => {
      const [x, y] = this.screen(lx, ly);
      m.style.transform = `translate(${Math.round(x + dx)}px, ${Math.round(y + dy)}px)`;
    };
    const t = tool.peek();
    if (t === "track" && this.start) {
      put(this.el(k++, "pi-mark"), this.start.lx, this.start.ly);
      const pv = this.lastPreview;
      if (pv) {
        // only those on screen: a snapped route can have thousands of curves (T-727d)
        // and none over another, so zoomed out a long route keeps a few
        const { clientWidth: W, clientHeight: H } = this.map.getContainer();
        const shown: [number, number][] = [];
        for (const c of curveLabels(pv.verts)) {
          const [x, y] = this.screen(c.x, c.y);
          if (x < -40 || y < -40 || x > W + 40 || y > H + 40) continue;
          if (shown.some(([u, v]) => Math.abs(u - x) < 64 && Math.abs(v - y) < 22)) continue;
          if (shown.push([x, y]) > 300) break;
          const m = this.el(k++, "pi-label");
          setCurveLabel(m, c.kmh);
          put(m, c.x, c.y, 8, 6);
        }
      }
    }
    for (let i = k; i < this.marks.length; i++) this.marks[i].style.display = "none";
    this.curveCircle(t === "track" && this.start ? this.lastPreview : null);

    // the platform ghost shows only while the station tool snaps (below)
    if (this.ghost) this.ghost.style.display = "none";
    if (this.rings) this.rings.style.display = "none";
    // the delete tool: a ring on the station or flyover, and what a click removes by the cursor
    if (t === "delete") {
      const d = this.delHover, what = d && describeRemovable(d);
      if (d && d.kind !== "track") {
        this.ring.style.display = "";
        put(this.ring, d.lx, d.ly);
      } else this.ring.style.display = "none";
      if (what) {
        this.tag.style.display = "";
        this.tag.className = "tool-tag build-tag";
        this.tag.textContent = `Remove ${what.what}`;
        this.tag.style.transform = `translate(${Math.round(this.delAt[0] + 14)}px, ${Math.round(this.delAt[1] + 12)}px)`;
      } else this.tag.style.display = "none";
      return;
    }
    // the snap ring and the cursor tag
    if (t === "station" && !this.freeAt) this.checkGhost();
    const h = t === "station" ? this.freeAt ?? this.stationSpot() : this.hover;
    const v = t === "station" ? this.verdict() : null;
    if (t === "station") this.shown = h && h.kind !== "free" ? v : null;
    if (t === "station" && h?.kind === "free" && !this.freeAt) {
      // T-907: open ground before the first click, the snap ring as a marker and no ghost
      this.clearFree();
      this.ring.style.display = "";
      put(this.ring, h.lx, h.ly);
      this.tag.style.display = "none";
      return;
    }
    if (t === "station" && h?.kind === "free") {
      // T-794, T-907: open ground after the first click, the station with its own piece of track
      this.ring.style.display = "none";
      const g = this.freeGhost(h);
      this.stationRings(g.mid);
      this.tag.style.display = "";
      if (g.ok === false) {
        this.tag.className = "tool-tag build-tag bad";
        this.tag.textContent = g.why;
      } else {
        this.tag.className = "tool-tag build-tag";
        this.tag.replaceChildren(...this.helpLines(g.mid, g.ok ? `${length(g.plat)} platform and ${length(g.lengthM)} of track, ${moneyShort(g.cost * 1e6)}` : null)
          .map((s) => Object.assign(document.createElement("div"), { textContent: s })));
        if (!this.tag.firstChild) this.tag.style.display = "none";
      }
      // T-834: the tag follows the pointer
      const p = this.pointer;
      if (p) this.tag.style.transform = `translate(${Math.round(p.px + 14)}px, ${Math.round(p.py + 12)}px)`;
      else put(this.tag, h.lx, h.ly, 14, 12);
      return;
    }
    if (t === "station") this.clearFree();
    if ((t === "track" || t === "station") && h && h.kind !== "free") {
      this.ring.style.display = "";
      if (t === "station") {
        this.stationRings(h);
        this.stationGhost(h);
      } else put(this.ring, h.lx, h.ly);
    } else if (t === "track" && this.real.ringAt) {
      this.ring.style.display = "";
      put(this.ring, ...toLocal(...this.real.ringAt));
    } else this.ring.style.display = "none";
    const pv = this.lastPreview, rt = t === "track" && this.start && h ? this.real.tag() : null;
    if (rt) {
      // T-727d: a real route still being found (the spinner once it takes a while), or none
      this.tag.className = "tool-tag build-tag" + (rt.bad ? " bad" : "");
      const kids: (string | Node)[] = [];
      if (rt.spin) kids.push(Object.assign(document.createElement("i"), { className: "spin" }));
      if (rt.text) kids.push(rt.text);
      this.tag.style.display = kids.length ? "" : "none";
      this.tag.replaceChildren(...kids);
      put(this.tag, h!.lx, h!.ly, 14, 12);
    } else if (t === "track" && this.start && h && pv) {
      this.tag.style.display = "";
      this.tag.className = "tool-tag build-tag";
      this.tag.replaceChildren(...placementTag(pv, buildLevel.peek()));
      put(this.tag, h.lx, h.ly, 14, 12);
    } else if (t === "station" && h && h.kind !== "free" && v?.ok === false) {
      // why a click here would be refused (T-311)
      this.tag.style.display = "";
      this.tag.className = "tool-tag build-tag bad";
      this.tag.textContent = v.why;
      put(this.tag, h.lx, h.ly, 14, 12);
    } else if (t === "station" && h && h.kind !== "free") {
      const lines = this.helpLines(h, null);
      if (lines.length) {
        this.tag.style.display = "";
        this.tag.className = "tool-tag build-tag";
        this.tag.replaceChildren(...lines.map((s) => Object.assign(document.createElement("div"), { textContent: s })));
        put(this.tag, h.lx, h.ly, 14, 12);
      } else this.tag.style.display = "none";
    } else this.tag.style.display = "none";
  }

  /** The station tool's tag lines: `first` if any, how far from the station placed last (T-312),
   * the trips near (T-412). */
  private helpLines(h: Snap, first: string | null): string[] {
    const lines: string[] = first ? [first] : [];
    const last = this.lastPlacedName(), [lng, lat] = toLngLat(h.lx, h.ly);
    if (last) lines.push(`${distance(metresBetween(lng, lat, last.lng, last.lat))} from ${last.name}`);
    const trips = this.tripsNear(lng, lat);
    if (trips) lines.push(trips);
    return lines;
  }
}

function sameStops(a: readonly number[], b: readonly number[]) {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}

/**
 * One label per curve from a preview's vertices (x, y at the middle of each curve, radius, km/h):
 * a curve fitted as several PIs (one turning more than a right angle) has the same radius at
 * consecutive vertices and gets one label, at its middle one. Curves too gentle to slow a train
 * are not labelled.
 */
export function curveLabels(verts: Float64Array): CurveLabel[] {
  const out: CurveLabel[] = [];
  const nv = verts.length / 4;
  for (let i = 1; i < nv - 1; ) {
    const r = verts[i * 4 + 2];
    let j = i + 1;
    while (j < nv - 1 && r > 0 && Math.abs(verts[j * 4 + 2] - r) < 0.5) j++;
    if (r > 0 && r < 5000) {
      const m = (i + j - 1) >> 1;
      out.push({ x: verts[m * 4], y: verts[m * 4 + 1], radius: r, kmh: verts[i * 4 + 3] });
    }
    i = j;
  }
  return out;
}

export interface CurveLabel {
  x: number;
  y: number;
  /** m, and the curve's speed limit in km/h */
  radius: number;
  kmh: number;
}

/** A curve label's content (T-417; the radius left out, T-530): the speed limit on the speed colour,
 * as the track tool's tag colours the speed. Set only when it changes, since the labels are placed
 * every frame. */
export function setCurveLabel(el: HTMLElement, kmh: number) {
  const key = String(kmh);
  if (el.dataset.curve === key && el.firstChild) return;
  el.dataset.curve = key;
  el.replaceChildren(chip(speed(kmh), speedColour(kmh)));
}

/**
 * T-412: the station tool counts trips within a 10-minute walk: 800 m straight line is about
 * 1,040 m walked at the model's 1.3 detour, 10 minutes at its 6 km/h. US units: half a mile.
 */
const TRIPS_RADIUS_M = 800;
const M_PER_HALF_MILE = 804.672;

/**
 * The track tool's tag (T-411, T-526): length, cost multiplier and price; then the elevation being
 * placed and the route's lowest speed limit (trackTagRows.ts). The multiplier is on yellow growing
 * above 1x. When the segment could not be built ending here, why first.
 */
function placementTag(pv: PreviewAnswer, level: number): HTMLDivElement[] {
  const div = (...kids: (string | Node)[]) => {
    const d = document.createElement("div");
    d.append(...kids);
    return d;
  };
  const out: HTMLDivElement[] = [];
  if (!pv.ok) {
    const why = div(describeIssues(pv.issues) || "Cannot be built here.");
    why.style.color = "var(--bad)"; // style.css's `.tool-tag.bad`, for this line only
    out.push(why);
  }
  const nv = pv.verts.length / 4;
  // the preview's limits already carry the track type's cap (T-995: high-speed track's is 320)
  let top = Infinity;
  if (pv.speeds?.length) {
    for (const v of pv.speeds) if (v >= 0) top = Math.min(top, v);
  } else for (let i = 1; i < nv - 1; i++) if (pv.verts[i * 4 + 2] > 0) top = Math.min(top, pv.verts[i * 4 + 3]);
  if (!Number.isFinite(top)) top = highSpeed.peek() ? HSR_KMH : TOP_KMH;
  if (pv.ok) {
    const dear = Math.min(1, (pv.mult - 1) / 1.5);
    out.push(div(`${distance(pv.lengthM)}, cost `, chip(`${pv.mult.toFixed(2)}x`, pv.mult > 1.005 ? COST_YELLOW : null, 0.2 + 0.6 * dear),
      `, ${moneyShort(pv.cost * 1e6)}`));
  }
  const [first, ...more] = placedLevels(pv.pts, level);
  out.push(factRow(elevationFact(first, ...more), speedFact(top)));
  return out;
}

/** The elevations a segment passes, in order and each once: where it starts, the top of a
 * bridge (T-656) or the bottom of a dip, where it ends (T-707: a ramp names both its levels).
 * `pts` x, y, height (m) triples; `level` when there are none. */
function placedLevels(pts: ArrayLike<number>, level: number): number[] {
  const n = pts.length / 3;
  if (n < 1) return [level];
  const a = pts[2] / 8, b = pts[(n - 1) * 3 + 2] / 8;
  let hi = -Infinity, lo = Infinity;
  for (let i = 2; i < pts.length; i += 3) {
    hi = Math.max(hi, pts[i] / 8);
    lo = Math.min(lo, pts[i] / 8);
  }
  const seq = [a];
  if (hi > Math.max(a, b) + 0.25) seq.push(hi);
  if (lo < Math.min(a, b) - 0.25) seq.push(lo);
  seq.push(b);
  return [...new Set(seq.map(Math.round))];
}

/** The M1 train's top speed, km/h: straight track's limit (track-and-service.md 6.1). */
const TOP_KMH = 160;
/** High-speed track's straight limit, km/h (the sim's `V_HSR`, T-817). */
const HSR_KMH = 320;
const COST_YELLOW = "#f2c200";

