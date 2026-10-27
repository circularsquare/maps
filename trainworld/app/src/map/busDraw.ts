// Drawing a bus line (T-1464, design T-1445; editing T-1514): click points, no track. Between two
// clicks the route follows the basemap's roads where a short enough road path exists, else it is
// straight (T-1465, `busRoads.ts`); a click within 100 m of a road lands on it. The clock worker
// quotes the route as it is drawn (`busQuote`): road length, price and where the stops would go; the
// stops and the tag by the pointer show that answer. With an existing line's route loaded
// (`begin(line, ...)`) finishing redraws it (`busRoute`), else it makes the line (`addBusLine`).
//
// The clicked points are handles (docs/ui-building.md "Drawing a bus line"): new points go on at the
// active end (either end; a click on the other one switches), a drag moves a point and re-searches
// the legs either side, a right click on one deletes it. A click on the active end, a double click,
// Enter or the Save/Make button finishes; Escape drops a new route's points or an edit as a whole.
//
// The tool owns no map layer: the route and stops are an SVG over the map in screen pixels, like
// the station ghost (map/tools.ts, which holds the tool's pointer events and the shared tag).

import { signal } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { distance, length, moneyShort, plural } from "../game/format";
import { newBusLine, nextFreeColour } from "../game/lineActions";
import { ask, renameLinePrompt, selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import { keyIsTheirs } from "../ui/keys";
import { BUS_KEY } from "../ui/toolKeys";
import type { ClockClient } from "../workers/clockClient";
import { metresBetween } from "../game/walks";
import { SAME_TAP_PX, touchInput, type TouchCtl } from "./touch";
import type { Overlay } from "./overlay";
import { BusRoads, type Leg } from "./busRoads";
import { setGrabbing } from "./nodeDrag";

/** the sim's `bus::MIN_ROUTE_M` */
const MIN_ROUTE_M = 100;
/** the worker's "no line yet" id for a quote */
const NEW_LINE = 0xffffffff;
/** street stops are drawn from this zoom */
const STREET_ZOOM = 12.5;
/** a click this close (px) to the active end adds nothing */
const SAME_PX = 4;
/** a press this close (px) to a clicked point is on its handle */
const HANDLE_PX = 8;
/** a press on a handle becomes a drag once the pointer has moved this far, px (MapLibre's click tolerance) */
const DRAG_PX = 3;

interface Pt { lng: number; lat: number }
interface Stop { x: number; y: number; kind: number; lng: number; lat: number }
interface Quote { road: number; cost: number; ok: boolean; stops: Stop[]; street: Pt[] }

/** What the tool is doing, for the panels: null while it is off. `line` the line whose route is
 * redrawn (its number), null for a new line; `points` placed so far. */
export const busDraft = signal<{ line: number | null; points: number } | null>(null);

/** `busQuote`'s answer (workers/clock.worker.ts `busQuoteLL`) as stops and street stops. */
function parseQuote(d: Float64Array): Quote {
  const stops: Stop[] = [], street: Pt[] = [];
  let k = 4;
  for (let i = 0; i < d[3]; i++, k += 5) stops.push({ x: d[k], y: d[k + 1], kind: d[k + 2], lng: d[k + 3], lat: d[k + 4] });
  const m = d[k++];
  for (let i = 0; i < m; i++, k += 4) street.push({ lng: d[k + 2], lat: d[k + 3] });
  return { road: d[0], cost: d[1], ok: d[2] === 1, stops, street };
}

/** the sim's `bus::K_CORNER` and `K_ROAD` */
const K_CORNER = 1, K_ROAD = 2;
/** a route point this close (px) to the last drawn one is left out of the drawing */
const DRAW_PX = 1.2;

/** The route as the sim has it: x, y and kind per point (longitude, latitude pairs in `ll`). */
const routeOf = (b: Built): number[] => {
  const out: number[] = [];
  for (let i = 0; i < b.kinds.length; i++) {
    const q = lngLatToPack(b.ll[2 * i], b.ll[2 * i + 1]);
    out.push(q.x, q.y, b.kinds[i]);
  }
  return out;
};

/** The route so far: every point with its kind, the corners (where each clicked point landed, one per
 * point in `pts` order, without the pointer's), whether every leg has its roads, and the run of route
 * points that is the leg to the pointer ([-1, -1] with none). */
interface Built { ll: number[]; kinds: number[]; corners: Pt[]; ready: boolean; dim: [number, number]; sig: string }

export class BusDraw {
  /** the corners as clicked, and the leg after each but the last */
  private pts: Pt[] = [];
  private legs: Leg[] = [];
  /** new points go before the first (true) or after the last */
  private head = false;
  readonly roads: BusRoads;
  /** changes whenever the corners or their legs do */
  private rev = 0;
  private builds = new Map<string, Built>();
  /** the pointer (mouse), and the point a tap previews */
  private at: Pt | null = null;
  private touchAt: Pt | null = null;
  /** the handle under the pointer (an index into `pts`), -1 for none */
  private over = -1;
  /** a press on a handle, and the drag it became: the point's index and the route before it */
  private press: { i: number; x: number; y: number } | null = null;
  private drag: { i: number; pts: Pt[]; legs: Leg[] } | null = null;
  private dragEnded = false;
  /** the line whose route is redrawn, null for a new line */
  private line: number | null = null;
  /** an edit has changed something since it opened */
  private dirty = false;
  private colour = "#1a9e94";
  /** the old road length and whether it is paid for, for a redrawn line's price */
  private was = { road: 0, built: false };
  private quote: Quote | null = null;
  private wantKey = "";
  private wantRoute: number[] = [];
  private inflight = false;
  private busy = false;
  private svg: SVGSVGElement;

  constructor(private map: MlMap, box: HTMLElement, private tag: HTMLElement, private client: ClockClient, private overlay: Overlay, private host: { exit(): void }) {
    this.roads = new BusRoads(map, () => this.loaded());
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "bus-draw");
    this.svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible;display:none";
    box.insertBefore(this.svg, box.firstChild);
    map.on("contextmenu", (e) => {
      if (tool.peek() !== "bus") return;
      e.originalEvent.preventDefault();
      const i = this.handleAt(e.point.x, e.point.y);
      if (i >= 0) this.remove(i);
      else this.back();
    });
    // capture: before MapLibre's own handlers, so a press on a handle does not pan the map
    map.getContainer().addEventListener("mousedown", (e) => this.onDown(e), true);
    addEventListener("mousemove", (e) => this.onDrag(e));
    addEventListener("mouseup", () => this.onUp());
    // B: a new bus line, from any tab. Registered before main.ts's key handler, which would take it
    // for nothing; stopImmediatePropagation keeps it from the others.
    window.addEventListener("keydown", (e) => {
      if (e.key.toLowerCase() !== BUS_KEY || e.ctrlKey || e.metaKey || e.altKey || e.repeat || keyIsTheirs(e) || ask.peek() || !world.peek()) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (tool.peek() !== "bus") newBusLine();
    });
  }

  get active() {
    return tool.peek() === "bus";
  }

  /** the pointer is on a point's handle, or drags one */
  get onHandle() {
    return this.active && (this.over >= 0 || this.drag !== null);
  }

  /** Start the tool: a new line (`line` null) or the route of line number `line` as `routeLL`
   * (longitude, latitude pairs, flat) and its point `kinds` to redraw: its corners stay, and the
   * stretches between them as they are. The caller has left the tool before. */
  begin(line: number | null, routeLL: number[] = [], colour?: string, kinds: number[] = []) {
    this.pts = [];
    this.legs = [];
    const n = routeLL.length >> 1;
    let from = -1;
    for (let i = 0; i < n; i++) {
      if (i > 0 && i < n - 1 && !((kinds[i] ?? K_CORNER) & K_CORNER)) continue;
      this.pts.push({ lng: routeLL[2 * i], lat: routeLL[2 * i + 1] });
      if (from >= 0) this.legs.push({ pts: routeLL.slice(2 * from, 2 * i + 2), road: !!((kinds[i] ?? 0) & K_ROAD), ready: true, m: 0 });
      from = i;
    }
    this.rev++;
    this.head = false;
    this.line = line;
    this.dirty = false;
    this.at = this.touchAt = null;
    this.over = -1;
    this.quote = null;
    this.wantKey = "";
    const c = this.map.getCenter();
    this.colour = colour ?? nextFreeColour([c.lng, c.lat]);
    const st = line === null ? undefined : world.peek()?.lineStats[String(line)]?.bus;
    this.was = { road: st?.roadM ?? 0, built: !!st?.built };
    tool.value = "bus";
    this.changed();
  }

  /** the tool was left (the tool signal is on something else) */
  clear() {
    this.endDrag();
    this.pts = [];
    this.legs = [];
    this.rev++;
    this.at = this.touchAt = null;
    this.over = -1;
    this.quote = null;
    this.wantKey = "";
    this.svg.style.display = "none";
    busDraft.value = null;
  }

  /** The points or legs changed (`edited`: by the player, so an edit has something to save). */
  private changed(edited = false) {
    if (edited) this.dirty = true;
    if (this.pts.length < 2) this.head = false;
    const d = busDraft.peek();
    if (!d || d.line !== this.line || d.points !== this.pts.length) busDraft.value = { line: this.line, points: this.pts.length };
    this.ask();
    this.place();
  }

  // ---------------------------------------------------------------- the quote

  /** where a leg to the pointer goes: none while the pointer is on a handle or a point is dragged */
  private hoverPt(): Pt | null {
    if (touchInput.peek()) return this.touchAt;
    return this.over >= 0 || this.drag ? null : this.at;
  }

  /** the point new ones join (the active end) */
  private endPt(): Pt | undefined {
    return this.head ? this.pts[0] : this.pts[this.pts.length - 1];
  }

  private endIndex(): number {
    return this.head ? 0 : this.pts.length - 1;
  }

  /** The route so far as points, built from the legs and, with the pointer out, the leg to it. */
  private build(withHover: boolean): Built {
    const hover = withHover ? this.hoverPt() : null;
    const key = `${this.rev}|${this.head}|${hover ? hover.lng + "," + hover.lat : ""}`;
    const hit = this.builds.get(key);
    if (hit) return hit;
    const legs = [...this.legs];
    let hoverLeg: Leg | null = null;
    const end = this.endPt();
    if (hover && end) {
      hoverLeg = this.head ? this.roads.leg(hover, end) : this.roads.leg(end, hover);
      if (this.head) legs.unshift(hoverLeg);
      else legs.push(hoverLeg);
    }
    const ll: number[] = [], kinds: number[] = [], corners: Pt[] = [];
    let ready = true;
    const dim: [number, number] = [-1, -1];
    if (!legs.length) {
      // one corner, or the pointer alone: where it would land
      const p = this.pts[0] ?? hover;
      if (p) {
        const c = this.roads.corner(p);
        ready = !!c;
        const q = c ?? p;
        ll.push(q.lng, q.lat);
        kinds.push(K_CORNER);
        if (this.pts.length) corners.push(q);
      }
    }
    legs.forEach((leg, i) => {
      if (!leg.ready) ready = false;
      const n = leg.pts.length;
      const isHover = leg === hoverLeg;
      if (i === 0) {
        ll.push(leg.pts[0], leg.pts[1]);
        kinds.push(K_CORNER);
        if (!(isHover && this.head)) corners.push({ lng: leg.pts[0], lat: leg.pts[1] });
      }
      const from = kinds.length - 1;
      for (let k = 2; k < n; k += 2) {
        ll.push(leg.pts[k], leg.pts[k + 1]);
        kinds.push((leg.road ? K_ROAD : 0) | (k + 2 >= n ? K_CORNER : 0));
      }
      if (isHover) {
        dim[0] = from;
        dim[1] = kinds.length;
      }
      if (!(isHover && !this.head)) corners.push({ lng: leg.pts[n - 2], lat: leg.pts[n - 1] });
    });
    const b: Built = { ll, kinds, corners, ready, dim, sig: `${key}|${ll.length}|${ready}|${hoverLeg ? hoverLeg.road : ""}` };
    if (this.builds.size > 40) this.builds.clear();
    this.builds.set(key, b);
    return b;
  }

  /** Tiles arrived: legs that waited for them are made again. */
  private loaded() {
    let changed = false;
    this.legs.forEach((leg, i) => {
      if (leg.ready) return;
      const now = this.roads.leg(this.pts[i], this.pts[i + 1]);
      if (now.ready) {
        this.legs[i] = now;
        changed = true;
      }
    });
    if (changed) this.rev++;
    this.builds.clear();
    if (tool.peek() !== "bus") return;
    this.ask();
    this.place();
  }

  private ask() {
    const b = this.build(true);
    if (b.kinds.length < 2) {
      this.quote = null;
      this.wantKey = "";
      return;
    }
    if (b.sig === this.wantKey) return;
    this.wantKey = b.sig;
    this.wantRoute = routeOf(b);
    if (!this.inflight) this.fire();
  }

  private fire() {
    const key = this.wantKey, route = this.wantRoute;
    this.inflight = true;
    void this.client.query({ what: "busQuote", line: this.line ?? NEW_LINE, route }).then((d) => {
      this.inflight = false;
      if (tool.peek() !== "bus") return;
      if (key === this.wantKey) {
        this.quote = parseQuote(d);
        this.place();
      } else if (this.wantKey) this.fire();
    });
  }

  // ---------------------------------------------------------------- handles

  /** The clicked point whose handle is under a screen point, -1 for none; the active end first, then
   * the nearest. */
  private handleAt(px: number, py: number): number {
    const cs = this.build(false).corners;
    let best = HANDLE_PX * HANDLE_PX, hit = -1;
    cs.forEach((c, i) => {
      const s = this.map.project([c.lng, c.lat]);
      const d = (s.x - px) ** 2 + (s.y - py) ** 2;
      if (d < best || (d < HANDLE_PX * HANDLE_PX && i === this.endIndex())) {
        best = i === this.endIndex() ? -1 : d;
        hit = i;
      }
    });
    return hit;
  }

  private onDown(e: MouseEvent) {
    if (tool.peek() !== "bus" || this.busy || e.button !== 0 || e.target !== this.map.getCanvas() || touchInput.peek()) return;
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const i = this.handleAt(px, py);
    if (i < 0) return;
    // a press without a drag still reaches MapLibre as a click
    this.map.dragPan.disable();
    this.press = { i, x: px, y: py };
  }

  private onDrag(e: MouseEvent) {
    const p = this.press;
    if (!p) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (!this.drag) {
      if (Math.hypot(px - p.x, py - p.y) < DRAG_PX) return;
      this.drag = { i: p.i, pts: [...this.pts], legs: [...this.legs] };
      setGrabbing(true);
    }
    const ll = this.map.unproject([px, py]);
    this.at = { lng: ll.lng, lat: ll.lat };
    this.movePoint(p.i, this.at);
  }

  /** Point `i` is now at `q`: the legs either side are searched again. */
  private movePoint(i: number, q: Pt) {
    this.pts[i] = q;
    if (i > 0) this.legs[i - 1] = this.roads.leg(this.pts[i - 1], q);
    if (i < this.pts.length - 1) this.legs[i] = this.roads.leg(q, this.pts[i + 1]);
    this.roads.look(q);
    this.rev++;
    this.changed(true);
  }

  private onUp() {
    const was = this.drag;
    this.press = null;
    this.drag = null;
    this.map.dragPan.enable();
    if (!was) return;
    setGrabbing(false);
    // MapLibre may still call a press that came back to its start a click
    this.dragEnded = true;
    setTimeout(() => (this.dragEnded = false));
    this.place();
  }

  /** Drop a drag, the point back where it was; false with none. */
  private cancelDrag(): boolean {
    const d = this.drag;
    if (!d && !this.press) return false;
    if (d) {
      this.pts = d.pts;
      this.legs = d.legs;
      this.rev++;
    }
    this.endDrag();
    this.changed();
    return true;
  }

  private endDrag() {
    if (this.drag) setGrabbing(false);
    if (this.press || this.drag) this.map.dragPan.enable();
    this.press = null;
    this.drag = null;
  }

  /** Delete point `i`; the points either side are joined by a new leg. */
  private remove(i: number) {
    if (this.busy || i < 0 || i >= this.pts.length) return;
    if (i > 0 && i < this.pts.length - 1) this.legs.splice(i - 1, 2, this.roads.leg(this.pts[i - 1], this.pts[i + 1]));
    else if (i === 0) this.legs.shift();
    else this.legs.pop();
    this.pts.splice(i, 1);
    this.over = -1;
    this.rev++;
    this.changed(true);
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    this.at = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    this.roads.look(this.at);
    this.over = this.drag ? this.drag.i : this.handleAt(e.point.x, e.point.y);
    const cur = this.over >= 0 ? "grab" : "crosshair";
    if (this.map.getCanvas().style.cursor !== cur) this.map.getCanvas().style.cursor = cur;
    this.ask();
    this.place();
  }

  click(e: MapMouseEvent) {
    if (this.busy || this.dragEnded) return;
    // a double click: its first click placed the last point
    if (e.originalEvent.detail >= 2) {
      if (this.pts.length >= 2) void this.finish();
      return;
    }
    const i = this.handleAt(e.point.x, e.point.y);
    if (i >= 0) return this.handleClick(i);
    this.add({ lng: e.lngLat.lng, lat: e.lngLat.lat });
  }

  /** A click on a point: the active end finishes, the other end becomes the active one. */
  private handleClick(i: number) {
    const n = this.pts.length;
    if (n < 2) return;
    if (i === this.endIndex()) return void this.finish();
    if (i === 0 || i === n - 1) {
      this.head = i === 0;
      this.over = -1;
      this.rev++;
      this.changed();
    }
  }

  private add(p: Pt) {
    const end = this.endPt();
    if (end) {
      const a = this.map.project([end.lng, end.lat]), b = this.map.project([p.lng, p.lat]);
      if (Math.hypot(a.x - b.x, a.y - b.y) < SAME_PX) return;
    }
    if (this.head) {
      this.pts.unshift(p);
      if (end) this.legs.unshift(this.roads.leg(p, end));
    } else {
      this.pts.push(p);
      if (end) this.legs.push(this.roads.leg(end, p));
    }
    this.rev++;
    this.changed(true);
  }

  /** Take back the active end; with no point placed, leave the tool. */
  private back() {
    if (this.busy) return;
    if (this.pts.length === 0) return this.host.exit();
    this.remove(this.endIndex());
  }

  /** Keys: true if used. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "Escape") {
      if (this.cancelDrag()) return true;
      // an edited route goes back as it was; a new line's points are dropped, and the next Esc leaves
      if (this.line !== null || this.pts.length === 0) this.host.exit();
      else {
        this.pts = [];
        this.legs = [];
        this.rev++;
        this.touchAt = null;
        this.changed();
      }
      return true;
    }
    if (this.drag) return true;
    if (e.key === "Backspace") {
      this.back();
      return true;
    }
    if (e.key === "Enter") {
      void this.finish();
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- touch

  tap(e: MapMouseEvent) {
    const p = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    if (this.pts.length === 0) return this.add(p);
    const near = (q: Pt | null | undefined) => {
      if (!q) return false;
      const s = this.map.project([q.lng, q.lat]);
      return Math.hypot(s.x - e.point.x, s.y - e.point.y) < SAME_TAP_PX;
    };
    if (near(this.touchAt)) return this.confirm();
    if (!this.touchAt && this.pts.length >= 2) {
      // the active end tapped again, with nothing waiting, finishes; the other end becomes the active one
      if (near(this.endPt())) return void this.finish();
      const other = this.head ? this.pts.length - 1 : 0;
      if (near(this.pts[other])) return this.handleClick(other);
    }
    this.touchAt = p;
    this.changed();
  }

  confirm() {
    const p = this.touchAt;
    if (p) {
      this.touchAt = null;
      this.add(p);
    } else void this.finish();
  }

  /** the touch bar's back: the point waiting, else the active end, else the tool */
  touchBack() {
    if (this.touchAt) {
      this.touchAt = null;
      this.changed();
    } else this.back();
  }

  touchState(): TouchCtl | null {
    if (!touchInput.peek() || !this.pts.length) return null;
    const waiting = !!this.touchAt;
    return { confirm: waiting || this.pts.length >= 2 ? !this.busy : null, confirmTip: waiting ? "Add this point" : this.line === null ? "Make the line" : "Save the route", back: true, stopTip: "Cancel", level: false, track: false };
  }

  /** the point the touch bar stands beside */
  touchPoint(): Pt | null {
    return this.touchAt ?? this.endPt() ?? null;
  }

  // ---------------------------------------------------------------- finishing

  /** Wait (a few seconds at most) for the roads under legs still waiting for their tiles. */
  private async settle() {
    for (let i = 0; i < 40 && this.legs.some((l) => !l.ready); i++) {
      await new Promise((r) => setTimeout(r, 100));
      this.loaded();
    }
  }

  /** Make the line, or redraw the route, from the points placed; a route that makes no line is
   * refused with the reason. An edit that changed nothing just closes. */
  async finish() {
    if (this.busy || tool.peek() !== "bus") return;
    this.endDrag();
    if (this.line !== null && !this.dirty) return this.host.exit();
    if (this.pts.length < 2) {
      say("Click at least two points to draw a bus line.", true);
      return;
    }
    this.busy = true;
    try {
      await this.settle();
      if (tool.peek() !== "bus") return;
      const route = routeOf(this.build(false)), line = this.line;
      const q = parseQuote(await this.client.query({ what: "busQuote", line: line ?? NEW_LINE, route }));
      if (tool.peek() !== "bus") return;
      if (!q.ok) {
        let m = 0;
        for (let i = 1; i < this.pts.length; i++) m += metresBetween(this.pts[i - 1].lng, this.pts[i - 1].lat, this.pts[i].lng, this.pts[i].lat);
        say(m < MIN_ROUTE_M ? `A bus route needs at least ${length(MIN_ROUTE_M)}.` : "That route cannot be a bus line. Try fewer points, inside the map.", true);
        return;
      }
      if (line === null) {
        const a = await edit({ op: "addBusLine", route, name: "", colour: this.colour }, true);
        if (!a.ok || a.line === undefined) {
          say("The bus line could not be added. Try another route.", true);
          return;
        }
        say(`Bus line added: ${distance(q.road)}, ${moneyShort(q.cost * 1e6)} to construct.`);
        this.host.exit();
        renameLinePrompt.value = String(a.line);
        selection.value = { kind: "line", line: String(a.line) };
      } else {
        const a = await edit({ op: "busRoute", line, route }, true);
        if (!a.ok) {
          say("The route could not be changed. Try another.", true);
          return;
        }
        say(`Route changed: ${distance(q.road)}.`);
        this.host.exit();
      }
    } finally {
      this.busy = false;
    }
  }

  // ---------------------------------------------------------------- drawing

  /** Draw the route, its stops, the handles and the tag; after each map frame and each change. */
  place() {
    if (tool.peek() !== "bus") {
      this.svg.style.display = "none";
      return;
    }
    const hover = this.hoverPt();
    const b = this.build(true);
    const n = b.kinds.length;
    const q = this.quote && n >= 2 ? this.quote : null;
    const xy = (p: Pt) => this.map.project([p.lng, p.lat]);
    const col = this.colour;
    const parts: string[] = [];
    // the route's points on the screen, leaving out those within a pixel of the last one kept
    const line = (from: number, to: number) => {
      let d = "", lx = NaN, ly = NaN;
      for (let i = from; i < to; i++) {
        const s = xy({ lng: b.ll[2 * i], lat: b.ll[2 * i + 1] });
        if (i > from && i < to - 1 && Math.abs(s.x - lx) < DRAW_PX && Math.abs(s.y - ly) < DRAW_PX) continue;
        d += `${i === from ? "M" : "L"}${s.x.toFixed(1)} ${s.y.toFixed(1)}`;
        lx = s.x;
        ly = s.y;
      }
      return d;
    };
    if (n >= 2) {
      const d = line(0, n);
      const dim = b.dim[0] >= 0 ? line(b.dim[0], b.dim[1]) : "";
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:7px;stroke-opacity:.9"/>`);
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:${col};stroke-width:3.5px"/>`);
      if (dim) parts.push(`<path d="${dim}" fill="none" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:2px;stroke-opacity:.45"/>`);
    }
    const ring = (p: Pt, r: number, w: number) => {
      const s = xy(p);
      return `<circle cx="${s.x.toFixed(1)}" cy="${s.y.toFixed(1)}" r="${r}" style="fill:var(--map-land);stroke:${col};stroke-width:${w}px"/>`;
    };
    const square = (p: Pt) => {
      const s = xy(p);
      return `<rect x="${(s.x - 4.5).toFixed(1)}" y="${(s.y - 4.5).toFixed(1)}" width="9" height="9" rx="2.5" style="fill:var(--map-land);stroke:${col};stroke-width:2.2px"/>`;
    };
    if (q) {
      if (this.map.getZoom() >= STREET_ZOOM) for (const s of q.street) parts.push(ring(s, 2.4, 1.3));
      for (const s of q.stops) {
        if (s.kind === 0) parts.push(square(s));
        else if (s.kind === 1) {
          const p = xy(s);
          parts.push(`<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="9" style="fill:none;stroke:${col};stroke-width:2.5px"/>`);
        } else parts.push(ring(s, 3.6, 1.8));
      }
    } else if (hover && n && this.pts.length === 0) parts.push(square({ lng: b.ll[0], lat: b.ll[1] }));
    // the handles: each clicked point a dot in the line's colour, the active end a ring, the one under
    // the pointer larger
    const end = this.endIndex();
    b.corners.forEach((c, i) => {
      const s = xy(c), big = i === this.over || i === this.drag?.i;
      const at = `cx="${s.x.toFixed(1)}" cy="${s.y.toFixed(1)}"`;
      if (i === end) parts.push(`<circle class="bus-end" ${at} r="${big ? 7 : 6}" style="fill:var(--map-land);stroke:${col};stroke-width:3px"/>`);
      else parts.push(`<circle class="bus-handle" ${at} r="${big ? 6 : 4.5}" style="fill:${col};stroke:var(--map-land);stroke-width:2px"/>`);
    });
    this.svg.style.display = "";
    this.svg.setAttribute("width", String(this.overlay.cssW));
    this.svg.setAttribute("height", String(this.overlay.cssH));
    this.svg.innerHTML = parts.join("");
    this.placeTag(q, n);
  }

  private placeTag(q: Quote | null, n: number) {
    const tag = this.tag;
    const row = (text: string) => Object.assign(document.createElement("div"), { textContent: text });
    const at = touchInput.peek() ? this.touchAt : this.at;
    const show = (rows: HTMLElement[], bad = false) => {
      tag.className = "tool-tag build-tag" + (bad ? " bad" : "");
      tag.replaceChildren(...rows);
      tag.style.display = "";
      const s = this.map.project([at!.lng, at!.lat]);
      tag.style.transform = `translate(${Math.round(s.x + 14)}px, ${Math.round(s.y + 12)}px)`;
    };
    // over a handle: what a click, a drag and a right click do there
    const h = this.drag ? -1 : this.over;
    if (h >= 0 && at && !touchInput.peek()) {
      const ends = this.pts.length >= 2 && (h === 0 || h === this.pts.length - 1);
      const first = !ends ? [] : h === this.endIndex() ? [row(this.line === null ? "Click to make the line" : "Click to save the route")] : [row("Click to draw from this end")];
      return show([...first, row("Drag to move"), row("Right click to delete")]);
    }
    if (!at || !q || n < 2) {
      tag.style.display = "none";
      return;
    }
    if (!q.ok) return show([row(`A bus route needs at least ${length(MIN_ROUTE_M)}.`)], true);
    // a built route redrawn pays only for the road it adds
    const pay = this.was.built ? Math.max(0, q.cost - this.was.road / 1000) : q.cost;
    show([row(`${distance(q.road)} of road, ${moneyShort(pay * 1e6)}${this.was.built ? " more" : ""}`), row(plural(q.stops.length + q.street.length, "stop"))]);
  }
}
