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

import type { Geometry } from "geojson";
import { effect } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { describeIssues } from "../game/issues";
import { count, distance, length, levelLabel, money, moneyShort, speed, unitMetres, unitShort } from "../game/format";
import { activeCity, getWorldFrame, worldFrameEpoch } from "../game/cities";
import { cellModes, cityCells, demandView } from "../game/demand";
import { LEVEL_COLOURS, rgb, speedColour } from "../game/palette";
import type { Level } from "../game/types";
import { metresBetween } from "../game/walks";
import { ask, buildLevel, buildPlatform, keepPlacing, lineDraft, selection, shownDayPhrase, trackCount, tool } from "../game/ui";
import { describeRemovable, removeThing, type Removable } from "../game/remove";
import { loadPlacesAt, placesNear } from "../game/placeList";
import { compass } from "../game/places";
import { edit, say, world } from "../game/world";
import type { RouteEnd } from "../workers/protocol";
import type { ClockClient, CrossoverAnswer, PreviewAnswer } from "../workers/clockClient";
import { toLngLat, toLocal } from "./geo";
import { circleOnScreen } from "./curveCircle";
import { strokesFromParts, strokesFromPreview } from "./network";
import { PLATFORM_HALF_M, platformShape, reversed, stationPath, stationShape, waysFrom } from "./stationGeometry";
import type { Overlay } from "./overlay";
import type { StopMarkers } from "./stopMarkers";

const SNAP_NODE_PX = 12;
const SNAP_EDGE_PX = 10;
/** tracks this much further from the pointer than the nearest still count as overlapping it (T-469) */
const OVERLAP_PX = 4;

type Snap =
  | { kind: "node"; node: number; lx: number; ly: number }
  | { kind: "edge"; edge: number; lx: number; ly: number }
  | { kind: "free"; lx: number; ly: number };

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
  /** the request `lastPreview` answers, as JSON */
  private previewKey = "";
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
    this.tag.className = "tool-tag";
    this.ring = document.createElement("div");
    this.ring.className = "snap-ring";
    this.box.append(this.tag, this.ring);
    overlay.afterMapFrame.push(() => this.place());
    effect(() => { buildPlatform.value; if (tool.value === "station") this.place(); });
    // the place lists name stations where the tiles have no streets (T-065); the view's zones' are
    // fetched when the station tool is picked, not at startup (T-400)
    effect(() => { if (tool.value === "station") { const c = this.map.getCenter(); void loadPlacesAt(c.lng, c.lat); } });
    effect(() => { buildLevel.value; trackCount.value; if (tool.peek() === "track" && this.start) this.requestPreview(); });
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

  // ---------------------------------------------------------------- snapping

  private screen(lx: number, ly: number) {
    return this.overlay.toScreen(lx, ly);
  }

  /** What is under the pointer for building: a node, a point on track, or open ground. */
  private snap(px: number, py: number, lngLat: { lng: number; lat: number }, nodes = true): Snap {
    const w = world.peek();
    const n = this.overlay.renderer.net;
    const [fx, fy] = toLocal(lngLat.lng, lngLat.lat);
    if (!w || !n) return { kind: "free", lx: fx, ly: fy };
    let best = SNAP_NODE_PX * SNAP_NODE_PX, hit: Snap | null = null;
    for (const node of nodes ? w.nodes.values() : []) {
      const [lx, ly] = toLocal(node.lng, node.lat);
      const [x, y] = this.screen(lx, ly);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) {
        best = d;
        hit = { kind: "node", node: node.id, lx, ly };
      }
    }
    if (hit) return hit;
    // each edge's nearest point within reach, px squared
    const near: { d: number; hit: Snap }[] = [];
    for (const [edge, pts] of n.edgePts) {
      let eb = SNAP_EDGE_PX * SNAP_EDGE_PX, eh: Snap | null = null;
      for (let k = 0; k + 5 < pts.length; k += 3) {
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
    if (tool.peek() !== "station") return near[0].hit;
    // T-469: where tracks overlap, the track the last station went on, then blueprint over built
    const close = near.filter((c) => c.d <= near[0].d + OVERLAP_PX);
    if (close.length === 1) return close[0].hit;
    const onLast = this.lastStationTrack();
    const built = new Map(w.edges.map((e) => [e.id, e.built]));
    const rank = (c: { hit: Snap }) => {
      const e = (c.hit as { edge: number }).edge;
      return (onLast.has(e) ? 0 : 2) + (built.get(e) ? 1 : 0);
    };
    return close.reduce((a, b) => (rank(b) < rank(a) ? b : a)).hit;
  }

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
    const byNode = new Map<number, number[]>();
    for (const e of w.edges) for (const k of [e.a, e.b]) (byNode.get(k) ?? byNode.set(k, []).get(k)!).push(e.id);
    const edges = new Map(w.edges.map((e) => [e.id, e]));
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
    this.hover = t === "line" ? null : this.snap(e.point.x, e.point.y, e.lngLat, t !== "crossover");
    if (t === "track" && this.start) this.requestPreview();
    if (t === "crossover") this.checkCrossover();
    if (t === "station") this.nameAhead(e.point.x, e.point.y);
    this.place();
  }

  click(e: MapMouseEvent) {
    const t = tool.peek();
    if (t === "track") this.trackClick(e);
    else if (t === "station") void this.stationClick(e);
    else if (t === "crossover") void this.crossoverClick(e);
    else if (t === "line") this.lineClick(e);
    else if (t === "delete") this.deleteClick(e);
  }

  /** Keys while a tool is on; true if used. */
  key(e: KeyboardEvent): boolean {
    const t = tool.peek();
    if (t === "select") return false;
    if (e.key === "Escape") {
      if (t === "track" && this.start) this.reset();
      else this.exit();
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
    this.cross = null;
    this.crossKey = "";
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
    const s = this.snap(e.point.x, e.point.y, e.lngLat);
    if (!this.start) {
      this.start = { end: this.endOf(s), lx: s.lx, ly: s.ly };
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
    const [sx, sy] = this.screen(from.lx, from.ly);
    if (Math.hypot(sx - e.point.x, sy - e.point.y) < 6) return; // the start itself
    const chain = e.originalEvent.shiftKey || keepPlacing.peek();
    const p = this.lastPreview;
    this.placing = true;
    this.endRun = false;
    const a = await edit({ op: "route", ...this.routeReq(this.endOf(s)) });
    this.placing = false;
    if (!a.ok || this.start !== from) return;
    say(p ? `Blueprint added: ${distance(p.lengthM)}, ${money(p.cost * 1e6)} to construct.` : "Blueprint added.");
    if (!chain || this.endRun || a.node === undefined) {
      this.reset();
      return;
    }
    this.chain.push(from);
    this.start = { end: { kind: "node", node: a.node }, lx: s.lx, ly: s.ly };
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
    return { from: this.start!.end, to, pis: [], single: trackCount.peek() === 1, quad: trackCount.peek() === 4, through: true };
  }

  private requestPreview() {
    if (!this.start) return;
    const se = this.start.end;
    if (se.kind === "node" && !world.peek()?.nodes.has(se.node)) return this.reset(); // undone meanwhile
    const h = this.hover;
    if (!h) return;
    // Nothing to show while the pointer is on the start.
    const [ax, ay] = this.screen(this.start.lx, this.start.ly), [bx, by] = this.screen(h.lx, h.ly);
    if (Math.hypot(ax - bx, ay - by) < 3) return;
    const req = this.routeReq(this.endOf(h));
    const key = JSON.stringify(req);
    this.client.preview(req, (a) => {
      if (this.start) this.showPreview(a, key);
    });
  }

  private showPreview(a: PreviewAnswer, key: string) {
    this.lastPreview = a;
    this.previewKey = key;
    this.overlay.renderer.setPreview(a.pts.length >= 6 ? strokesFromPreview(a.pts, !a.ok, a.speeds, true) : null);
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
    const s = this.snap(e.point.x, e.point.y, e.lngLat);
    if (s.kind === "free") {
      say("Click on track to put a station there.");
      return;
    }
    const key = this.nameKey(e.point.x, e.point.y);
    const ahead = this.named?.key === key && this.map.areTilesLoaded() ? this.named.name : null;
    this.named = null;
    const name = await (ahead ?? this.nameFor(e.point.x, e.point.y));
    const at = s.kind === "node" ? { kind: "node" as const, node: s.node } : { kind: "edge" as const, edge: s.edge, ...this.pack(s.lx, s.ly) };
    const a = await edit({ op: "addStation", at, platform: buildPlatform.peek(), name });
    if (a.ok) {
      this.lastPlaced = toLngLat(s.lx, s.ly); // T-312: the hover tag's "km from" (found again by position)
      say(`${name} added as a blueprint station.`);
    }
  }

  // ---------------------------------------------------------------- crossovers (T-338)

  /** The last crossover preview for the track under the pointer, and what it was asked for. */
  private cross: CrossoverAnswer | null = null;
  private crossKey = "";

  /** Ask the worker what a crossover here would cost or why not, and draw its X as the preview. */
  private checkCrossover() {
    const h = this.hover;
    if (!h || h.kind !== "edge") {
      if (this.cross) {
        this.cross = null;
        this.crossKey = "";
        this.overlay.renderer.setPreview(null);
        this.overlay.invalidate();
      }
      return;
    }
    const key = `${h.edge}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}:${world.peek()?.version}`;
    if (key === this.crossKey) return;
    this.crossKey = key;
    const p = this.pack(h.lx, h.ly);
    this.client.checkCrossover(h.edge, p.x, p.y, (a) => {
      if (key !== this.crossKey || tool.peek() !== "crossover") return;
      this.cross = a;
      this.overlay.renderer.setPreview(a.parts.length ? strokesFromParts(a.parts, !a.ok) : null);
      this.overlay.invalidate();
      this.place();
    });
  }

  private async crossoverClick(e: MapMouseEvent) {
    const s = this.snap(e.point.x, e.point.y, e.lngLat, false);
    if (s.kind !== "edge") {
      say("Click on track beside other track to put a crossover there.");
      return;
    }
    const cost = this.cross?.ok ? this.cross.cost : null;
    const a = await edit({ op: "crossover", edge: s.edge, ...this.pack(s.lx, s.ly) });
    if (a.ok) {
      say(cost !== null ? `Crossover added as a blueprint, ${money(cost * 1e6)} to construct.` : "Crossover added as a blueprint.");
      this.crossKey = "";
      this.checkCrossover();
    }
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

  /**
   * A name for a new station: from zoom 13, where the basemap's tiles carry street names, the
   * nearest named street within 400 m (shortened New York style); else the nearest place on the
   * city's OSM place list (T-065, game/placeList.ts); else a place on the loaded tiles; else
   * "Station N". Names already in use are skipped, so a second station on Broadway takes its
   * cross street and a second one in Williamsburg the next neighbourhood or "North Williamsburg".
   */
  private async nameFor(px: number, py: number): Promise<string> {
    let taken = new Set(world.peek()?.save.stations.map((s) => s.name) ?? []);
    const at = this.map.unproject([px, py]);
    const kx = 111320 * Math.cos((at.lat * Math.PI) / 180), ky = 110540;
    const dist = (g: Geometry): number => {
      const lines: number[][][] = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : g.type === "Point" ? [[g.coordinates]] : [];
      let best = Infinity;
      for (const l of lines) {
        const pts = l.map((c) => [(c[0] - at.lng) * kx, (c[1] - at.lat) * ky]);
        if (pts.length === 1) best = Math.min(best, Math.hypot(pts[0][0], pts[0][1]));
        for (let i = 0; i + 1 < pts.length; i++) {
          const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
          const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
          const t = l2 ? Math.max(0, Math.min(1, (-ax * dx - ay * dy) / l2)) : 0;
          best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
        }
      }
      return best;
    };
    const pickFrom = (sourceLayer: string, maxM: number, ok: (p: Record<string, any>) => boolean): string | null => {
      const layer = this.map.getStyle()?.layers.find((l) => (l as any)["source-layer"] === sourceLayer) as any;
      if (!layer) return null;
      let feats: { geometry: Geometry; properties: Record<string, any> }[] = [];
      try {
        feats = this.map.querySourceFeatures(layer.source, { sourceLayer }) as any;
      } catch {
        return null;
      }
      const best = new Map<string, number>();
      for (const f of feats) {
        const raw = (f.properties?.["name:latin"] || f.properties?.name_en || f.properties?.name) as string | undefined;
        if (!raw || !ok(f.properties)) continue;
        const name = shorten(raw);
        const d = dist(f.geometry);
        if (d <= maxM && d < (best.get(name) ?? Infinity)) best.set(name, d);
      }
      return [...best].sort((a, b) => a[1] - b[1]).map((e) => e[0]).find((n) => !taken.has(n)) ?? null;
    };
    // below zoom 13 the tiles hold only the bigger roads, so the nearest one shown is not the nearest
    const street = this.map.getZoom() >= STREET_NAME_ZOOM ? pickFrom("transportation_name", 400, (p) => p.class !== "motorway") : null;
    if (street) return street;
    const near = await placesNear(at.lng, at.lat).catch(() => []);
    taken = new Set(world.peek()?.save.stations.map((s) => s.name) ?? []); // a station added meanwhile counts
    const listed = placeName(near, taken);
    if (listed) return listed;
    const place = pickFrom("place", 1200, (p) => ["neighbourhood", "quarter", "suburb", "village", "hamlet", "town"].includes(p.class));
    if (place) return place;
    for (let k = 1; ; k++) if (!taken.has(`Station ${k}`)) return `Station ${k}`;
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

  /** Add the station as a stop, or take it out if the line stops there (T-233). Clicks queue and
   * each reads the latest stops. */
  toggleStop(station: number) {
    const session = this.lineSession;
    this.stopClicks = this.stopClicks.then(() => this.toggle(station, session)).catch(() => {
      if (session === this.lineSession) say("Could not change that stop. Please try again.", true);
    });
  }

  private async toggle(station: number, session: number) {
    if (session !== this.lineSession || tool.peek() !== "line") return;
    const id = lineDraft.peek().line;
    const line = world.peek()?.save.lines.find((l) => Number(l.id) === id);
    if (id === null || !line) return;
    const before = line.stops.map(Number);
    if (!before.includes(station)) return this.addStop(station, session);
    // taking a stop out: the route between its neighbours is the worker's, as the inspector's crosses
    const a = await edit({ op: "setStops", line: id, stops: before.filter((s) => s !== station) });
    if (a.ok && session === this.lineSession && tool.peek() === "line") {
      const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (latest) lineDraft.value = { line: id, stops: latest.stops.map(Number) };
    }
  }

  private async addStop(station: number, session: number) {
    // Read the authoritative stops for every queued click: removals and undo may have changed
    // them while the previous click was waiting for the worker.
    for (let attempt = 0; attempt < 3; attempt++) {
      if (session !== this.lineSession || tool.peek() !== "line") return;
      const id = lineDraft.peek().line;
      if (id === null) return; // The Lines pane creates the line before entering this tool.
      const w = world.peek();
      const line = w?.save.lines.find((l) => Number(l.id) === id);
      if (!line) return;
      const before = line.stops.map(Number);
      if (before.includes(station)) {
        say("That station is already a stop on this line.");
        return;
      }
      const result = await this.client.query({ what: "insertStop", line: id, station });
      if (session !== this.lineSession || tool.peek() !== "line" || lineDraft.peek().line !== id) return;
      const current = world.peek()?.save.lines.find((l) => Number(l.id) === id);
      if (!current) return;
      if (world.peek()?.version !== w!.version || !sameStops(current.stops.map(Number), before)) continue;
      if (!result.length) {
        say("No track joins these stations with that stop inserted. Check the track connections.", true);
        return;
      }
      const stops = Array.from(result);
      const a = await edit({ op: "setStops", line: id, stops });
      if (a.ok && session === this.lineSession && tool.peek() === "line") {
        const latest = world.peek()?.save.lines.find((l) => Number(l.id) === id);
        if (latest) lineDraft.value = { line: id, stops: latest.stops.map(Number) };
      }
      return;
    }
    say("The network changed while adding that stop. Please try again.", true);
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
    const sh=platformShape(ways,0,buildPlatform.peek()/2*perM);
    // always the full box at its true length, whatever the zoom (T-311): the map's own style
    // draws dots below STATION_BOX_ZOOM
    const style={...this.overlay.renderer.stationStyle,boxes:true};
    const shape=stationShape(h.lx,h.ly,sh,PLATFORM_HALF_M*perM,(x,y)=>this.screen(x,y),style,false);
    if (!this.ghost) {
      this.ghost=document.createElementNS("http://www.w3.org/2000/svg","svg");
      this.ghost.setAttribute("width","1");
      this.ghost.setAttribute("height","1");
      this.ghost.appendChild(document.createElementNS("http://www.w3.org/2000/svg","path"));
      this.box.appendChild(this.ghost);
    }
    this.checkGhost(h);
    this.ring.style.display="none";
    this.ghost.style.display="";
    this.ghost.setAttribute("class",`station-ghost ${this.ghostOk===false?"bad":"ok"}`);
    this.ghost.style.transform=`translate(${shape.x}px,${shape.y}px)`;
    this.ghost.firstElementChild!.setAttribute("d",stationPath(shape));
  }

  /** Faint rings 1 and 2 km, or miles in US units (ground distance), round the station being placed (T-312), under the
   * ghost, labelled at their tops. */
  private rings: SVGSVGElement | null = null;
  private stationRings(h: Exclude<Snap,{kind:"free"}>) {
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

  /** Whether a click here would place the station (T-311): the worker tries the same `addStation`
   * the click sends and rolls it back. Asked when the spot, the platform length or the network
   * changes, one request in flight (the newest waits), and the last answer stays up meanwhile. */
  private ghostOk: boolean | null = null;
  private ghostWhy = "";
  private ghostKey = "";
  private checkGhost(h: Exclude<Snap,{kind:"free"}>) {
    const platform=buildPlatform.peek();
    const key=`${h.kind==="node"?`n${h.node}`:`e${h.edge}`}:${h.lx.toFixed(1)}:${h.ly.toFixed(1)}:${platform}:${world.peek()?.version}`;
    if (key===this.ghostKey) return;
    this.ghostKey=key;
    const at=h.kind==="node"?{kind:"node" as const,node:h.node}:{kind:"edge" as const,edge:h.edge,...this.pack(h.lx,h.ly)};
    this.client.checkStation(at,platform,(ok,issues)=>{
      if (key!==this.ghostKey) return; // a newer spot is already asked
      const why=ok?"":describeIssues(issues)||"Cannot be built here.";
      if (ok===this.ghostOk && why===this.ghostWhy) return;
      this.ghostOk=ok;
      this.ghostWhy=why;
      if (tool.peek()==="station") this.place();
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
        for (const c of curveLabels(pv.verts)) {
          const m = this.el(k++, "pi-label");
          setCurveLabel(m, c.radius, c.kmh);
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
        this.tag.className = "tool-tag";
        this.tag.textContent = `Remove ${what.what}`;
        this.tag.style.transform = `translate(${Math.round(this.delAt[0] + 14)}px, ${Math.round(this.delAt[1] + 12)}px)`;
      } else this.tag.style.display = "none";
      return;
    }
    // the snap ring and the cursor tag
    const h = this.hover;
    if ((t === "track" || t === "station" || t === "crossover") && h && h.kind !== "free") {
      this.ring.style.display = "";
      if (t === "station") {
        this.stationRings(h);
        this.stationGhost(h);
      } else put(this.ring, h.lx, h.ly);
    } else this.ring.style.display = "none";
    const pv = this.lastPreview;
    const cx = this.cross;
    if (t === "crossover" && h && h.kind === "edge" && cx) {
      this.tag.style.display = "";
      this.tag.className = "tool-tag" + (cx.ok ? "" : " bad");
      this.tag.textContent = cx.ok ? moneyShort(cx.cost * 1e6) : describeIssues(cx.issues) || "Cannot be built here.";
      put(this.tag, h.lx, h.ly, 14, 12);
    } else if (t === "track" && this.start && h && pv) {
      this.tag.style.display = "";
      this.tag.className = "tool-tag";
      this.tag.replaceChildren(...placementTag(pv, buildLevel.peek()));
      put(this.tag, h.lx, h.ly, 14, 12);
    } else if (t === "station" && h && h.kind !== "free" && this.ghostOk === false) {
      // why a click here would be refused (T-311)
      this.tag.style.display = "";
      this.tag.className = "tool-tag bad";
      this.tag.textContent = this.ghostWhy;
      put(this.tag, h.lx, h.ly, 14, 12);
    } else if (t === "station" && h && h.kind !== "free") {
      const lines: string[] = [];
      const last = this.lastPlacedName(), [lng, lat] = toLngLat(h.lx, h.ly);
      // T-312: how far from the station placed last
      if (last) lines.push(`${distance(metresBetween(lng, lat, last.lng, last.lat))} from ${last.name}`);
      const trips = this.tripsNear(lng, lat);
      if (trips) lines.push(trips);
      if (lines.length) {
        this.tag.style.display = "";
        this.tag.className = "tool-tag";
        this.tag.replaceChildren(...lines.map((s) => Object.assign(document.createElement("div"), { textContent: s })));
        put(this.tag, h.lx, h.ly, 14, 12);
      } else this.tag.style.display = "none";
    } else this.tag.style.display = "none";
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

/** A curve label's content (T-417): the radius and speed limit on the speed colour, as the track
 * tool's tag colours the speed. Set only when it changes, since the labels are placed every frame. */
export function setCurveLabel(el: HTMLElement, radius: number, kmh: number) {
  const key = `${radius}:${kmh}`;
  if (el.dataset.curve === key && el.firstChild) return;
  el.dataset.curve = key;
  el.replaceChildren(chip(`${length(radius)}, ${speed(kmh)}`, speedColour(kmh)));
}

/**
 * T-412: the station tool counts trips within a 10-minute walk: 800 m straight line is about
 * 1,040 m walked at the model's 1.3 detour, 10 minutes at its 6 km/h. US units: half a mile.
 */
const TRIPS_RADIUS_M = 800;
const M_PER_HALF_MILE = 804.672;

/** A value in the placement tag with its colour behind it (T-411). */
function chip(text: string, colour: string | null, alpha = 0.5): HTMLSpanElement {
  const s = document.createElement("span");
  s.textContent = text;
  if (colour) {
    const [r, g, b] = rgb(colour);
    s.style.background = `rgba(${r},${g},${b},${alpha})`;
    s.style.padding = "0 3px";
    s.style.borderRadius = "2px";
  }
  return s;
}

/**
 * The track tool's tag (T-411): length, cost multiplier, price and the tightest curve; then the
 * level being placed and the route's lowest speed limit, the multiplier, level and speed on their
 * colours (yellow growing above 1x; the Height and Max speed displays' colours). When the segment
 * could not be built ending here, why first.
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
    why.style.color = "#b23a3a"; // style.css's `.tool-tag.bad`, for this line only
    out.push(why);
  }
  let radius = Infinity;
  const nv = pv.verts.length / 4;
  for (let i = 1; i < nv - 1; i++) if (pv.verts[i * 4 + 2] > 0) radius = Math.min(radius, pv.verts[i * 4 + 2]);
  let top = TOP_KMH;
  if (pv.speeds?.length) for (const v of pv.speeds) if (v >= 0) top = Math.min(top, v);
  else for (let i = 1; i < nv - 1; i++) if (pv.verts[i * 4 + 2] > 0) top = Math.min(top, pv.verts[i * 4 + 3]);
  if (pv.ok) {
    const dear = Math.min(1, (pv.mult - 1) / 1.5);
    out.push(div(`${distance(pv.lengthM)}, cost `, chip(`${pv.mult.toFixed(2)}x`, pv.mult > 1.005 ? COST_YELLOW : null, 0.2 + 0.6 * dear),
      `, ${moneyShort(pv.cost * 1e6)}`, ...(radius < Infinity ? [`, ${length(radius)} curve`] : [])));
  }
  out.push(div(chip(`Level ${levelLabel(level)}`, LEVEL_COLOURS[level as Level] ?? null), ", max speed ", chip(speed(top), speedColour(top))));
  return out;
}

/** The M1 train's top speed, km/h: straight track's limit (track-and-service.md 6.1). */
const TOP_KMH = 160;
const COST_YELLOW = "#f2c200";

/** From this zoom the basemap's tiles carry street names (T-065). */
const STREET_NAME_ZOOM = 13;
/** A taken nearest place gives way to an untaken one at most this much further away. */
const PLACE_SLACK_M = 800;

/**
 * A station's name from the places near it (nearest first, game/placeList.ts): the nearest, or
 * if that name is taken another within PLACE_SLACK_M further, or the nearest with the way the
 * station lies from it ("North Williamsburg"), or with a number ("Williamsburg 2"). Null when no
 * place is near.
 */
export function placeName(near: { place: { name: string }; d: number; dx: number; dy: number }[], taken: Set<string>): string | null {
  if (!near.length) return null;
  const first = near[0];
  const free = near.find((n) => n.d <= first.d + PLACE_SLACK_M && !taken.has(n.place.name));
  if (free) return free.place.name;
  if (first.d > 150) {
    const way = compass(first.dx, first.dy);
    const named = `${way[0].toUpperCase()}${way.slice(1)} ${first.place.name}`;
    if (!taken.has(named)) return named;
  }
  for (let k = 2; ; k++) if (!taken.has(`${first.place.name} ${k}`)) return `${first.place.name} ${k}`;
}

/** "West 42nd Street" -> "W 42nd St". */
function shorten(n: string): string {
  return n
    .replace(/\bWest\b/g, "W")
    .replace(/\bEast\b/g, "E")
    .replace(/\bNorth\b/g, "N")
    .replace(/\bSouth\b/g, "S")
    .replace(/\bStreet\b/g, "St")
    .replace(/\bAvenue\b/g, "Av")
    .replace(/\bBoulevard\b/g, "Blvd")
    .replace(/\bRoad\b/g, "Rd")
    .replace(/\bParkway\b/g, "Pkwy")
    .replace(/\bPlace\b/g, "Pl")
    .replace(/\bExpressway\b/g, "Expy");
}

