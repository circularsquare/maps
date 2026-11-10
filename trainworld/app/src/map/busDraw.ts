// Drawing a bus line (T-1464, design T-1445; editing T-1514): click points, no track. Between two
// clicks the route follows the basemap's roads (T-1465, `busRoads.ts`); a click within 100 m of a road
// lands on the nearest the route reaches, and a point the roads do not reach from its neighbour is
// refused (T-2451, T-2467): a click, drag or delete waits for the roads it needs, then is made or undone
// with the reason. A leg is drawn along its roads or red and straight, never straight in the line's
// colour; an older save's straight leg that finds no road stays red and keeps the route from being
// saved. The clock worker
// quotes the route as it is drawn (`busQuote`): road length, price and where the stops would go; the
// stops and the tag by the pointer show that answer, and the stations and other bus lines those stops
// connect with are lit as hovered ones are (T-2993, `setToolHot`). With an existing line's route loaded
// (`begin(line, ...)`) finishing redraws it (`busRoute`), else it makes the line (`addBusLine`).
//
// The clicked points are handles (docs/ui-building.md "Drawing a bus line"): new points go on at the
// active end (either end; a click on the other one switches), a drag moves a point and re-searches
// the legs either side, a click on one deletes it (T-3109; a finger: drag, tap to delete). A click on the active
// end, a double click, Enter, Escape or the Save/Make button finishes a new route; Escape drops an edit as a whole.
// A right click (a finger: hold, T-1517) loops (T-3109): on the other end, with three points or more, it closes the
// route into a loop and finishes it (T-2899): the last point is then a copy of the first, which is all the sim keeps
// of a loop (`bus::is_closed`). On a point partway along it joins the active end back to that point and finishes:
// the bus runs out, round the loop and back the way it came, kept as a loop whose last legs are its first ones
// backward (`bus::loop_back_at`), folded back to the way out and the loop here (`begin`). A looped route has no
// active end; a right click where it closes opens it again, as does deleting that point, and taking out a point
// of a loop of three leaves it open. A right click on the active end deletes it.
//
// The tool owns no map layer: the route and stops are an SVG over the map in screen pixels, like
// the station ghost (map/tools.ts, which holds the tool's pointer events and the shared tag).

import { t } from "../i18n";
import { signal } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { distance, length, moneyShort } from "../game/format";
import { editBusRoute, newBusLine, nextFreeColour } from "../game/lineActions";
import { ask, renameLinePrompt, selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import { keyIsTheirs } from "../ui/keys";
import { BUS_KEY } from "../ui/toolKeys";
import type { ClockClient } from "../workers/clockClient";
import { metresBetween } from "../game/walks";
import { SAME_TAP_PX, touchInput, type TouchCtl } from "./touch";
import type { Overlay } from "./overlay";
import { BusRoads, SNAP_M, type BusRefusal, type Leg } from "./busRoads";
import { setGrabbing } from "./nodeDrag";
import { playBuild } from "./buildSound";
import { safeColour } from "./safeColour";
import { setToolHot } from "./hovered";
import { routeMeets } from "../game/busLinks";

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
/** a stop this close (m) to a clicked point is that point, squared */
const NEAR_POINT_SQ_M = 12 * 12;
/** a press on a handle becomes a drag once the pointer has moved this far, px (MapLibre's click tolerance) */
const DRAG_PX = 3;
/** a finger's reach to a handle, and how far it may wander (px) before the press is a drag, not a hold */
const TOUCH_HANDLE_PX = 18;
const TOUCH_SLOP_PX = 8;
/** a finger held still on a handle this long (ms) deletes the point (T-1517) */
const LONG_PRESS_MS = 500;

/** a refusal as the tag over the pointer says it, and as the message after a click (T-2451) */
const refusalTag = (r: BusRefusal) => t(r === "far" ? "busDraw.tagFar" : r === "path" ? "busDraw.tagPath" : "busDraw.tagLong");
const notLoaded = () => t("busDraw.notLoaded");
const refusalSay = (r: BusRefusal) =>
  r === "far" ? t("busDraw.sayFar", { distance: length(SNAP_M) }) : r === "path" ? t("busDraw.sayPath") : t("busDraw.sayLong");

interface Pt { lng: number; lat: number }
/** the draft as it was after an edit, for Ctrl+Z */
interface Snap { pts: Pt[]; legs: Leg[] }
const samePts = (a: Pt[], b: Pt[]) => a.length === b.length && a.every((p, i) => p.lng === b[i].lng && p.lat === b[i].lat);
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

/** the sim's `bus::K_CORNER`, `K_ROAD` and `K_TRUNK` */
const K_CORNER = 1, K_ROAD = 2, K_TRUNK = 4;
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
 * points that is the leg to the pointer ([-1, -1] with none); each leg's run of points and how it is
 * drawn (`runs`: along its roads, red and straight with no road or none found yet, or not at all while
 * the pointer's waits for its roads), and why the pointer's leg, or the pointer alone before the first
 * point, would be refused. */
interface Built {
  ll: number[]; kinds: number[]; corners: Pt[]; ready: boolean; dim: [number, number]; sig: string;
  runs: { from: number; to: number; draw: "road" | "bad" | "none" }[]; hoverRefused: BusRefusal | null; hoverRoad: boolean;
}

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
  private press: { i: number; x: number; y: number; timer?: number } | null = null;
  private drag: { i: number; pts: Pt[]; legs: Leg[] } | null = null;
  private dragEnded = false;
  /** T-3084: a press on a point of the selected bus line while no tool is on; it opens Edit route once it moves */
  private pre: { line: number; at: Pt; x: number; y: number } | null = null;
  /** the draft after its last edit, the steps back from it and the steps forward again; `orig` the route as opened */
  private base: Snap = { pts: [], legs: [] };
  private undos: Snap[] = [];
  private redos: Snap[] = [];
  private orig: Pt[] = [];
  /** the fingers down, and whether the tap that ends a drag or a long press is to be ignored */
  private fingers = new Set<number>();
  private noTap = false;
  /** the line whose route is redrawn, null for a new line */
  private line: number | null = null;
  /** an edit has changed something since it opened */
  private dirty = false;
  private colour = "#1a9e94";
  /** the old road length and whether it is paid for, for a redrawn line's price */
  private was = { road: 0, built: false };
  private quote: Quote | null = null;
  /** T-2993: the quote's transfer lines (`routeMeets`), lng, lat, lng, lat flat */
  private links: number[] = [];
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
      // a held finger makes one too: its own long press has dealt with it (T-1517)
      if (touchInput.peek()) return;
      const i = this.handleAt(e.point.x, e.point.y);
      if (i >= 0) this.rightClick(i);
      else this.back();
    });
    // capture: before MapLibre's own handlers, so a press on a handle does not pan the map
    map.getContainer().addEventListener("mousedown", (e) => this.onDown(e), true);
    addEventListener("mousemove", (e) => (this.openOnDrag(e), this.onDrag(e)));
    addEventListener("mouseup", () => this.onUp());
    // a finger does the same through pointer events: drag a point, or hold it to delete it (T-1517)
    map.getContainer().addEventListener("pointerdown", (e) => this.onTouchDown(e), true);
    addEventListener("pointermove", (e) => e.pointerType === "touch" && this.onDrag(e));
    addEventListener("pointerup", (e) => this.onTouchEnd(e));
    addEventListener("pointercancel", (e) => this.onTouchEnd(e));
    // U (BUS_KEY): a new bus line, from any tab. Registered before main.ts's key handler, which would take it
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
   * (longitude, latitude pairs, flat) and its point `kinds` to redraw: its corners stay, and the road
   * stretches between them as they are; a straight one (saved before T-2451) is searched again. The
   * caller has left the tool before. */
  begin(line: number | null, routeLL: number[] = [], colour?: string, kinds: number[] = []) {
    this.pts = [];
    this.legs = [];
    const n = routeLL.length >> 1;
    let from = -1;
    for (let i = 0; i < n; i++) {
      if (i > 0 && i < n - 1 && !((kinds[i] ?? K_CORNER) & K_CORNER)) continue;
      const p = { lng: routeLL[2 * i], lat: routeLL[2 * i + 1] };
      if (from >= 0) {
        const road = !!((kinds[i] ?? 0) & K_ROAD);
        this.legs.push(road ? { pts: routeLL.slice(2 * from, 2 * i + 2), road, ready: true, m: 0 } : this.roads.leg(this.pts[this.pts.length - 1], p));
      }
      this.pts.push(p);
      from = i;
    }
    // T-3109: a route that loops back is drawn with its way out once, its last point the one the loop comes back to
    const last = this.pts.length - 1, same = (a: Pt, b: Pt) => a.lng === b.lng && a.lat === b.lat;
    if (last >= 4 && same(this.pts[0], this.pts[last])) {
      let k = 0;
      while (2 * (k + 1) < last && same(this.pts[k + 1], this.pts[last - k - 1])) k++;
      if (k >= 1 && last - 2 * k >= 3) {
        this.pts = this.pts.slice(0, last - k + 1);
        this.legs = this.legs.slice(0, last - k);
      }
    }
    this.rev++;
    this.head = false;
    this.line = line;
    this.dirty = false;
    this.orig = [...this.pts];
    this.base = this.snap();
    this.undos = [];
    this.redos = [];
    this.at = this.touchAt = null;
    this.over = -1;
    this.setQuote(null);
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
    this.setQuote(null);
    this.wantKey = "";
    this.svg.style.display = "none";
    busDraft.value = null;
  }

  /** The points or legs changed (`edited`: by the player, so an edit has something to save). */
  private changed(edited = false) {
    if (edited) {
      this.dirty = true;
      if (!this.drag) this.commit();
    }
    if (this.pts.length < 2) this.head = false;
    const d = busDraft.peek();
    const points = this.pts.length - (this.looped() ? 1 : 0);
    if (!d || d.line !== this.line || d.points !== points) busDraft.value = { line: this.line, points };
    this.ask();
    this.place();
  }

  // ---------------------------------------------------------------- the quote

  /** where a leg to the pointer goes: none while the pointer is on a handle or a point is dragged */
  private hoverPt(): Pt | null {
    if (this.looped()) return null;
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

  /** The route is a loop: its last point is its first (T-2899). */
  private closed(): boolean {
    const n = this.pts.length;
    return n >= 4 && this.pts[0].lng === this.pts[n - 1].lng && this.pts[0].lat === this.pts[n - 1].lat;
  }

  /** T-3109: the route loops back: its last point is its point `k` partway along, with three legs or more past
   * it; -1 for none. */
  private joinAt(): number {
    const n = this.pts.length, z = this.pts[n - 1];
    for (let k = 1; k <= n - 4; k++) if (this.pts[k].lng === z.lng && this.pts[k].lat === z.lat) return k;
    return -1;
  }

  /** a loop, or a route that loops back: no end to add points at */
  private looped(): boolean {
    return this.closed() || this.joinAt() >= 0;
  }

  /** where a looped route closes: its last point's handle and, for one that loops back, that point partway along */
  private isSeam(i: number): boolean {
    const n = this.pts.length;
    if (this.closed()) return i === 0 || i === n - 1;
    const k = this.joinAt();
    return k >= 0 && (i === k || i === n - 1);
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
      hoverLeg = this.head ? this.roads.leg(hover, end, "a") : this.roads.leg(end, hover, "b");
      if (this.head) legs.unshift(hoverLeg);
      else legs.push(hoverLeg);
    }
    // T-3109: a route that loops back runs its way out again backward after the loop; those legs have no handles
    const k = this.joinAt(), back = new Set<Leg>();
    if (k >= 0) for (let j = k - 1; j >= 0; j--) {
      const leg = reversedLeg(this.legs[j]);
      back.add(leg);
      legs.push(leg);
    }
    const ll: number[] = [], kinds: number[] = [], corners: Pt[] = [];
    let ready = true;
    const dim: [number, number] = [-1, -1];
    const runs: Built["runs"] = [];
    let hoverRefused: BusRefusal | null = hoverLeg?.refused ?? null;
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
        else if (this.roads.near(p) === false) hoverRefused = "far";
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
        kinds.push((leg.road ? K_ROAD : 0) | (leg.fast?.[k / 2 - 1] ? K_TRUNK : 0) | (k + 2 >= n ? K_CORNER : 0));
      }
      if (isHover) {
        dim[0] = from;
        dim[1] = kinds.length;
      }
      // never a straight leg in the line's colour (T-2467): the pointer's draws nothing until its roads are in
      runs.push({ from, to: kinds.length, draw: leg.road ? "road" : isHover && !leg.ready ? "none" : "bad" });
      if (!(isHover && !this.head) && !back.has(leg)) corners.push({ lng: leg.pts[n - 2], lat: leg.pts[n - 1] });
    });
    const hoverRoad = !!hoverLeg?.road;
    const b: Built = { ll, kinds, corners, ready, dim, runs, hoverRefused, hoverRoad, sig: `${key}|${ll.length}|${ready}|${hoverLeg ? hoverLeg.road : ""}` };
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
    // a leg to the pointer is quoted only once it follows roads
    const withHover = this.build(true);
    const b = withHover.dim[0] >= 0 && !withHover.hoverRoad ? this.build(false) : withHover;
    if (b.kinds.length < 2) {
      this.setQuote(null);
      this.wantKey = "";
      return;
    }
    if (b.sig === this.wantKey) return;
    this.wantKey = b.sig;
    this.wantRoute = routeOf(b);
    if (!this.inflight) this.fire();
  }

  /** A new quote, and with it what the route will connect with lit (T-2993): the stations it stops at
   * and the other bus lines it will change with. */
  private setQuote(q: Quote | null) {
    this.quote = q;
    this.links = [];
    const w = world.peek();
    if (!q || !w) return setToolHot(null);
    const lines = w.save.lines.flatMap((l) => {
      const b = w.lineStats[l.id]?.bus;
      return b && !b.ferry && l.num !== this.line ? [{ line: l.num, stops: b.stops, routeLL: b.routeLL }] : [];
    });
    const m = routeMeets(q.stops, w.save.stations, lines);
    this.links = m.links;
    setToolHot(m);
  }

  private fire() {
    const key = this.wantKey, route = this.wantRoute;
    this.inflight = true;
    void this.client.query({ what: "busQuote", line: this.line ?? NEW_LINE, route }).then((d) => {
      this.inflight = false;
      if (tool.peek() !== "bus") return;
      if (key === this.wantKey) {
        this.setQuote(parseQuote(d));
        this.place();
      } else if (this.wantKey) this.fire();
    });
  }

  // ---------------------------------------------------------------- handles

  /** The clicked point whose handle is under a screen point, -1 for none; the active end first, then
   * the nearest. */
  private handleAt(px: number, py: number, reach = HANDLE_PX): number {
    const cs = this.build(false).corners;
    let best = reach * reach, hit = -1;
    cs.forEach((c, i) => {
      const s = this.map.project([c.lng, c.lat]);
      const d = (s.x - px) ** 2 + (s.y - py) ** 2;
      if (d < best || (d < reach * reach && i === this.endIndex())) {
        best = i === this.endIndex() ? -1 : d;
        hit = i;
      }
    });
    return hit;
  }

  // ---------------------------------------------------------------- undo (T-3084)

  private snap(): Snap {
    return { pts: [...this.pts], legs: [...this.legs] };
  }

  /** An edit is done: the draft before it becomes a step back. A drag that ended where it began is none. */
  private commit() {
    if (samePts(this.base.pts, this.pts) && this.base.legs.length === this.legs.length) return;
    this.undos.push(this.base);
    this.redos = [];
    this.base = this.snap();
  }

  private restore(s: Snap) {
    this.pts = [...s.pts];
    this.legs = [...s.legs];
    this.base = this.snap();
    this.dirty = !samePts(this.pts, this.orig);
    this.touchAt = null;
    this.over = -1;
    this.rev++;
    this.changed();
  }

  /** Ctrl+Z: the draft's last edit goes back; the game's own undo is left alone while the tool is open. */
  private undo(): boolean {
    if (this.busy || this.drag) return true;
    const s = this.undos.pop();
    if (!s) return true;
    this.redos.push(this.snap());
    this.restore(s);
    return true;
  }

  private redo(): boolean {
    if (this.busy || this.drag) return true;
    const s = this.redos.pop();
    if (!s) return true;
    this.undos.push(this.snap());
    this.restore(s);
    return true;
  }

  // ---------------------------------------------------------------- a drag that opens Edit route (T-3084)

  /** What a press at a screen point could drag on the selected bus line while no tool is on: one of its
   * clicked points, or one of its stops (which becomes a point of the route where it stands). */
  private preHit(px: number, py: number): { line: number; at: Pt } | null {
    const w = world.peek(), sel = selection.peek();
    if (!w || sel?.kind !== "line") return null;
    const l = w.save.lines.find((x) => x.id === sel.line), b = l && w.lineStats[l.id]?.bus;
    if (!l || !b || b.ferry || b.routeLL.length < 4) return null;
    const r2 = HANDLE_PX * HANDLE_PX;
    const hit = (lng: number, lat: number) => {
      const s = this.map.project([lng, lat]);
      return (s.x - px) ** 2 + (s.y - py) ** 2 <= r2;
    };
    const n = b.routeLL.length >> 1;
    for (let i = 0; i < n; i++) {
      if (i > 0 && i < n - 1 && !((b.kinds[i] ?? K_CORNER) & K_CORNER)) continue;
      if (hit(b.routeLL[2 * i], b.routeLL[2 * i + 1])) return { line: l.num, at: { lng: b.routeLL[2 * i], lat: b.routeLL[2 * i + 1] } };
    }
    for (const st of b.stops) if (hit(st.lng, st.lat)) return { line: l.num, at: { lng: st.lng, lat: st.lat } };
    return null;
  }

  private preDown(e: MouseEvent) {
    if (tool.peek() !== "select" || this.busy || e.button !== 0 || e.target !== this.map.getCanvas() || touchInput.peek()) return;
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey || ask.peek()) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const h = this.preHit(px, py);
    if (!h) return;
    // a press without a drag still reaches MapLibre as a click
    this.map.dragPan.disable();
    this.pre = { ...h, x: px, y: py };
  }

  /** The press moved: Edit route opens on the line and the drag carries on in it, on the draft. */
  private openOnDrag(e: MouseEvent) {
    const p = this.pre;
    if (!p) return;
    const r = this.map.getContainer().getBoundingClientRect();
    if (Math.hypot(e.clientX - r.left - p.x, e.clientY - r.top - p.y) < DRAG_PX) return;
    this.pre = null;
    const w = world.peek(), l = w?.save.lines.find((x) => x.num === p.line);
    if (!l) return void this.map.dragPan.enable();
    editBusRoute(l);
    if (tool.peek() !== "bus" || this.line !== p.line) return void this.map.dragPan.enable();
    // the nearest clicked point, else the stop becomes one
    let i = -1, best = NEAR_POINT_SQ_M;
    this.pts.forEach((c, k) => {
      const d = metresBetween(c.lng, c.lat, p.at.lng, p.at.lat) ** 2;
      if (d < best) [best, i] = [d, k];
    });
    if (i < 0) i = this.insertPoint(p.at);
    if (i < 0) return void this.map.dragPan.enable();
    this.press = { i, x: p.x, y: p.y };
  }

  /** Put a point on the draft where a stop stands, in the leg nearest to it; its index, -1 for none. */
  private insertPoint(q: Pt): number {
    let k = -1, best = Infinity;
    this.legs.forEach((leg, j) => {
      for (let m = 0; m + 1 < leg.pts.length >> 1; m++) {
        const d = segM(leg.pts[2 * m], leg.pts[2 * m + 1], leg.pts[2 * m + 2], leg.pts[2 * m + 3], q);
        if (d < best) [best, k] = [d, j];
      }
    });
    if (k < 0) return -1;
    this.pts.splice(k + 1, 0, q);
    this.legs.splice(k, 1, this.roads.leg(this.pts[k], q), this.roads.leg(q, this.pts[k + 2]));
    this.rev++;
    return k + 1;
  }

  private onDown(e: MouseEvent) {
    if (tool.peek() !== "bus") return this.preDown(e);
    if ( this.busy || e.button !== 0 || e.target !== this.map.getCanvas() || touchInput.peek()) return;
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const i = this.handleAt(px, py);
    if (i < 0) return;
    // a press without a drag still reaches MapLibre as a click
    this.map.dragPan.disable();
    this.press = { i, x: px, y: py };
  }

  /** A finger on a handle: it drags the point and does not pan the map; held still it does a right click's
   * work there (T-3109); a tap that moves nothing reaches the tap logic as before. A second finger is a pinch. */
  private onTouchDown(e: PointerEvent) {
    if (e.pointerType !== "touch") return;
    this.fingers.add(e.pointerId);
    this.noTap = false;
    if (this.fingers.size > 1) return void this.cancelDrag();
    if (tool.peek() !== "bus" || this.busy || e.target !== this.map.getCanvas() || this.press) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const i = this.handleAt(px, py, TOUCH_HANDLE_PX);
    if (i < 0) return;
    // before MapLibre's touchstart, which the pointerdown precedes
    this.map.dragPan.disable();
    const timer = window.setTimeout(() => this.longPress(), LONG_PRESS_MS);
    this.press = { i, x: px, y: py, timer };
    this.over = i;
    this.place();
  }

  private longPress() {
    const p = this.press;
    if (!p || this.drag) return;
    this.press = null;
    this.map.dragPan.enable();
    this.noTap = true;
    navigator.vibrate?.(15);
    this.rightClick(p.i);
  }

  private onTouchEnd(e: PointerEvent) {
    if (e.pointerType !== "touch") return;
    this.fingers.delete(e.pointerId);
    if (e.type === "pointercancel") this.cancelDrag();
    const edited = this.drag !== null || this.noTap;
    this.onUp();
    this.over = -1;
    if (tool.peek() === "bus") this.place();
    // the browser's tap after a drag or a long press must not add a point
    if (edited) {
      this.noTap = true;
      setTimeout(() => (this.noTap = false), 400);
    }
  }

  private onDrag(e: MouseEvent) {
    const p = this.press;
    if (!p) return;
    const touch = p.timer !== undefined;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (!this.drag) {
      if (Math.hypot(px - p.x, py - p.y) < (touch ? TOUCH_SLOP_PX : DRAG_PX)) return;
      clearTimeout(p.timer);
      this.touchAt = null;
      this.drag = { i: p.i, pts: [...this.pts], legs: [...this.legs] };
      setGrabbing(true);
    }
    const ll = this.map.unproject([px, py]);
    this.at = { lng: ll.lng, lat: ll.lat };
    this.movePoint(p.i, this.at);
  }

  /** Point `i` is now at `q`: the legs either side are searched again, the point landing where the
   * first of them found a road for it (T-2467). */
  private movePoint(i: number, q: Pt) {
    if (this.closed() && (i === 0 || i === this.pts.length - 1)) return this.moveSeam(q);
    const k = this.joinAt();
    if (k >= 0 && (i === k || i === this.pts.length - 1)) return this.moveJoin(k, q);
    let at = q;
    if (i > 0) {
      const leg = this.roads.leg(this.pts[i - 1], q, "b");
      this.legs[i - 1] = leg;
      if (leg.road) at = { lng: leg.pts[leg.pts.length - 2], lat: leg.pts[leg.pts.length - 1] };
    }
    if (i < this.pts.length - 1) {
      const leg = this.roads.leg(at, this.pts[i + 1], i > 0 ? null : "a");
      this.legs[i] = leg;
      if (i === 0 && leg.road) at = { lng: leg.pts[0], lat: leg.pts[1] };
    }
    this.pts[i] = at;
    this.roads.look(q);
    this.rev++;
    this.changed(true);
  }

  /** A loop's first point, which is also its last, is now at `q`: the legs into and out of it are searched again. */
  private moveSeam(q: Pt) {
    const last = this.pts.length - 1;
    const into = this.roads.leg(this.pts[last - 1], q, "b");
    this.legs[last - 1] = into;
    const at = into.road ? { lng: into.pts[into.pts.length - 2], lat: into.pts[into.pts.length - 1] } : q;
    this.legs[0] = this.roads.leg(at, this.pts[1]);
    this.pts[0] = this.pts[last] = at;
    this.roads.look(q);
    this.rev++;
    this.changed(true);
  }

  /** T-3109: the point a route loops back to, point `k` and the last, is now at `q`: the three legs at it are searched again. */
  private moveJoin(k: number, q: Pt) {
    const last = this.pts.length - 1;
    const into = this.roads.leg(this.pts[k - 1], q, "b");
    this.legs[k - 1] = into;
    const at = into.road ? { lng: into.pts[into.pts.length - 2], lat: into.pts[into.pts.length - 1] } : q;
    this.legs[k] = this.roads.leg(at, this.pts[k + 1]);
    this.legs[last - 1] = this.roads.leg(this.pts[last - 1], at);
    this.pts[k] = this.pts[last] = at;
    this.roads.look(q);
    this.rev++;
    this.changed(true);
  }

  /** Wait (a few seconds at most) for `get` to give an answer that is not waiting for tiles. */
  private async whenReady<T>(get: () => T, waiting: (t: T) => boolean): Promise<T> {
    let t = get();
    for (let k = 0; k < 40 && waiting(t); k++) {
      await new Promise((r) => setTimeout(r, 100));
      t = get();
    }
    return t;
  }

  private onUp() {
    if (this.pre) {
      this.pre = null;
      this.map.dragPan.enable();
    }
    const was = this.drag;
    clearTimeout(this.press?.timer);
    this.press = null;
    this.drag = null;
    this.map.dragPan.enable();
    if (!was) return;
    setGrabbing(false);
    // MapLibre may still call a press that came back to its start a click
    this.dragEnded = true;
    setTimeout(() => (this.dragEnded = false));
    playBuild("snap");
    this.place();
    void this.checkDrop(was).then(() => this.commit());
  }

  /** Dropped where the roads do not reach: back where it was, unless its legs had no road before either. */
  private async checkDrop(was: { i: number; pts: Pt[]; legs: Leg[] }) {
    const i = was.i, last = this.pts.length - 1, join = this.joinAt();
    const sides = this.closed() && (i === 0 || i === last) ? [last - 1, 0]
      : join >= 0 && (i === join || i === last) ? [join - 1, join, last - 1]
      : [i - 1, i].filter((k) => k >= 0 && k < this.legs.length);
    if (sides.some((k) => !this.legs[k].ready)) {
      this.busy = true;
      try {
        await this.whenReady(() => (this.loaded(), sides.every((k) => this.legs[k].ready)), (ok) => !ok);
      } finally {
        this.busy = false;
      }
      if (tool.peek() !== "bus") return;
    }
    const unready = sides.some((k) => !this.legs[k].ready);
    const lost = unready ? "tiles" : sides.map((k) => (this.legs[k].refused && !was.legs[k]?.refused ? this.legs[k].refused : undefined)).find((r) => r);
    if (!lost) return;
    this.pts = was.pts;
    this.legs = was.legs;
    this.rev++;
    this.changed();
    say(lost === "tiles" ? notLoaded() : refusalSay(lost), true);
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
    clearTimeout(this.press?.timer);
    this.press = null;
    this.drag = null;
  }

  /** Delete point `i`; the points either side are joined by a new leg. */
  private async remove(i: number) {
    if (this.busy || i < 0 || i >= this.pts.length) return;
    // a loop's first point is also its last, and a route that loops back has its last at its join: taking it out opens the loop
    const wasClosed = this.closed(), wasLooped = this.looped();
    if (this.isSeam(i)) {
      this.legs.pop();
      this.pts.pop();
      playBuild("pop");
      this.over = -1;
      this.rev++;
      return this.changed(true);
    }
    if (i > 0 && i < this.pts.length - 1) {
      const a = this.pts[i - 1], b = this.pts[i + 1];
      let leg = this.roads.leg(a, b);
      if (!leg.ready) {
        this.busy = true;
        try {
          leg = await this.whenReady(() => this.roads.leg(a, b), (l) => !l.ready);
        } finally {
          this.busy = false;
        }
        if (tool.peek() !== "bus" || this.pts[i - 1] !== a || this.pts[i + 1] !== b) return;
        if (!leg.ready) return void say(notLoaded(), true);
      }
      // the points either side must still be joined by road, unless the legs it joins had none either
      if (leg.refused && !this.legs[i - 1].refused && !this.legs[i].refused) return void say(t("busDraw.pathWithout"), true);
      this.legs.splice(i - 1, 2, leg);
    } else if (i === 0) this.legs.shift();
    else this.legs.pop();
    playBuild("pop"); // T-2940
    this.pts.splice(i, 1);
    // a ring of three points is no loop without one, nor a loop of three past where a route loops back: the closing
    // point goes too
    if ((wasClosed && this.pts.length <= 3) || (wasLooped && !this.looped())) {
      this.pts.pop();
      this.legs.pop();
    }
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
    void this.add({ lng: e.lngLat.lng, lat: e.lngLat.lat });
  }

  /** A click on a point (T-3109): the active end, or where a loop closes, finishes; any other point is deleted. */
  private handleClick(i: number) {
    const n = this.pts.length;
    if (n < 2) return;
    if (this.isSeam(i) || (!this.looped() && i === this.endIndex())) return void this.finish();
    void this.remove(i);
  }

  /** What a right click on point `i` does (T-3109), null for nothing: open a loop where it closes, delete the active
   * end, close the loop at the other end, or loop back to a point partway along. */
  private rightAction(i: number): "open" | "delete" | "close" | "loopBack" | null {
    const n = this.pts.length;
    if (n < 2) return null;
    if (this.looped()) return this.isSeam(i) ? "open" : null;
    const end = this.endIndex();
    if (i === end) return "delete";
    if (i === 0 || i === n - 1) return n >= 3 ? "close" : null;
    // the active end joins back to it: the loop past it needs three legs
    return Math.abs(end - i) >= 2 ? "loopBack" : null;
  }

  private rightClick(i: number) {
    const a = this.rightAction(i);
    if (a === "open" || a === "delete") void this.remove(i);
    else if (a === "close") void this.closeLoop();
    else if (a === "loopBack") void this.loopBack(i);
  }

  /** T-3109: join the active end back to point `i` by road and make the line: the bus runs out, round the loop and
   * back. Refused with the reason when the roads do not reach. */
  private async loopBack(i: number) {
    if (this.busy) return;
    if (this.head) {
      // the route turned round, so the active end is its last point
      this.pts.reverse();
      this.legs = this.legs.reverse().map(reversedLeg);
      i = this.pts.length - 1 - i;
      this.head = false;
      this.rev++;
    }
    const a = this.pts[this.pts.length - 1], b = this.pts[i];
    let leg = this.roads.leg(a, b);
    if (!leg.ready) {
      this.busy = true;
      try {
        leg = await this.whenReady(() => this.roads.leg(a, b), (l) => !l.ready);
      } finally {
        this.busy = false;
      }
      if (tool.peek() !== "bus" || this.pts[this.pts.length - 1] !== a || this.pts[i] !== b) return;
    }
    if (!leg.ready) return void say(notLoaded(), true);
    if (leg.refused) return void say(refusalSay(leg.refused), true);
    this.pts.push(b);
    this.legs.push(leg);
    playBuild("snap");
    this.rev++;
    this.changed(true);
    await this.finish();
  }

  /** Join the last point to the first by road, and make the line (T-2899). Refused with the reason when the
   * roads do not reach. */
  private async closeLoop() {
    if (this.busy) return;
    const a = this.pts[this.pts.length - 1], b = this.pts[0];
    let leg = this.roads.leg(a, b);
    if (!leg.ready) {
      this.busy = true;
      try {
        leg = await this.whenReady(() => this.roads.leg(a, b), (l) => !l.ready);
      } finally {
        this.busy = false;
      }
      if (tool.peek() !== "bus" || this.pts[this.pts.length - 1] !== a || this.pts[0] !== b) return;
    }
    if (!leg.ready) return void say(notLoaded(), true);
    if (leg.refused) return void say(refusalSay(leg.refused), true);
    this.pts.push(b);
    this.legs.push(leg);
    playBuild("snap");
    this.rev++;
    this.changed(true);
    await this.finish();
  }

  /** Add a point at the active end where the click lands on a road the route reaches (T-2467): waits
   * for the roads there if they are still loading, then adds it or says why not. */
  private async add(p: Pt) {
    if (this.busy) return;
    if (this.looped()) return void say(t("busDraw.loopClosed"), true);
    const end = this.endPt(), head = this.head;
    if (end) {
      const a = this.map.project([end.lng, end.lat]), b = this.map.project([p.lng, p.lat]);
      if (Math.hypot(a.x - b.x, a.y - b.y) < SAME_PX) return;
    }
    const ask = () => (end ? (head ? this.roads.leg(p, end, "a") : this.roads.leg(end, p, "b")) : this.roads.near(p));
    let got = ask();
    const waiting = (g: Leg | boolean | null) => g === null || (typeof g === "object" && !g.ready);
    if (waiting(got)) {
      this.busy = true;
      try {
        got = await this.whenReady(ask, waiting);
      } finally {
        this.busy = false;
      }
      if (tool.peek() !== "bus" || this.endPt() !== end || this.head !== head) return;
    }
    if (waiting(got)) return void say(notLoaded(), true);
    const leg = typeof got === "object" ? got : null;
    const refused = leg ? leg.refused : got === false ? "far" : undefined;
    if (refused) return void say(refusalSay(refused), true);
    // the point goes where it landed on its road
    const n = leg ? leg.pts.length : 0;
    const at = !leg ? this.roads.corner(p) ?? p : head ? { lng: leg.pts[0], lat: leg.pts[1] } : { lng: leg.pts[n - 2], lat: leg.pts[n - 1] };
    if (head) {
      this.pts.unshift(at);
      if (leg) this.legs.unshift(leg);
    } else {
      this.pts.push(at);
      if (leg) this.legs.push(leg);
    }
    playBuild("snap"); // T-2940: a point placed sounds as track placed does
    this.rev++;
    this.changed(true);
  }

  /** Take back the active end; with no point placed, leave the tool. */
  private back() {
    if (this.busy) return;
    if (this.pts.length === 0) return this.host.exit();
    void this.remove(this.endIndex());
  }

  /** Keys: true if used. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "Escape") {
      if (this.cancelDrag()) return true;
      // Esc places a new line, as it ends a track chain (T-2924); an edited route goes back as it was
      if (this.line === null && this.pts.length >= 2) void this.finish();
      else if (this.line !== null || this.pts.length === 0) this.host.exit();
      else {
        this.pts = [];
        this.legs = [];
        this.rev++;
        this.touchAt = null;
        this.changed();
      }
      return true;
    }
    if ((e.ctrlKey || e.metaKey) && !e.altKey) {
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) return this.undo();
      if (k === "y" || (k === "z" && e.shiftKey)) return this.redo();
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
    if (this.noTap || this.busy) return;
    const p = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    if (this.pts.length === 0) return void this.add(p);
    const near = (q: Pt | null | undefined) => {
      if (!q) return false;
      const s = this.map.project([q.lng, q.lat]);
      return Math.hypot(s.x - e.point.x, s.y - e.point.y) < SAME_TAP_PX;
    };
    if (near(this.touchAt)) return this.confirm();
    if (!this.touchAt && this.pts.length >= 2) {
      // a point tapped, with nothing waiting: as a click on it (the active end finishes, another is deleted)
      const i = this.handleAt(e.point.x, e.point.y, SAME_TAP_PX);
      if (i >= 0) return this.handleClick(i);
    }
    this.touchAt = p;
    this.changed();
  }

  confirm() {
    const p = this.touchAt;
    if (p) {
      this.touchAt = null;
      void this.add(p);
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
    return { confirm: waiting || this.pts.length >= 2 ? !this.busy : null, confirmTip: waiting ? t("busDraw.addPoint") : this.line === null ? t("busDraw.makeLine") : t("busDraw.saveRoute"), back: true, stopTip: t("busDraw.cancel") };
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
      say(t("busDraw.twoPoints"), true);
      return;
    }
    this.busy = true;
    try {
      await this.settle();
      if (tool.peek() !== "bus") return;
      if (this.legs.some((l) => !l.ready)) return void say(notLoaded(), true);
      if (this.legs.some((l) => l.refused)) return void say(t("busDraw.partNoPath"), true);
      const route = routeOf(this.build(false)), line = this.line;
      const q = parseQuote(await this.client.query({ what: "busQuote", line: line ?? NEW_LINE, route }));
      if (tool.peek() !== "bus") return;
      if (!q.ok) {
        let m = 0;
        for (let i = 1; i < this.pts.length; i++) m += metresBetween(this.pts[i - 1].lng, this.pts[i - 1].lat, this.pts[i].lng, this.pts[i].lat);
        say(m < MIN_ROUTE_M ? t("busDraw.minLength", { length: length(MIN_ROUTE_M) }) : t("busDraw.cannotBus"), true);
        return;
      }
      if (line === null) {
        const a = await edit({ op: "addBusLine", route, name: "", colour: this.colour }, true);
        if (!a.ok || a.line === undefined) {
          say(t("busDraw.addFailed"), true);
          return;
        }
        say(t("busDraw.added", { distance: distance(q.road), cost: moneyShort(q.cost * 1e6) }));
        this.host.exit();
        renameLinePrompt.value = String(a.line);
        selection.value = { kind: "line", line: String(a.line) };
      } else {
        const a = await edit({ op: "busRoute", line, route }, true);
        if (!a.ok) {
          say(t("busDraw.changeFailed"), true);
          return;
        }
        say(t("busDraw.changed", { distance: distance(q.road) }));
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
    const col = safeColour(this.colour);
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
      // legs along roads in the line's colour; one with no road, or none found yet, red, dashed and
      // straight as the ferry tool's (T-2467)
      const road = b.runs.filter((r) => r.draw === "road").map((r) => line(r.from, r.to)).join("");
      const bad = b.runs.filter((r) => r.draw === "bad").map((r) => line(r.from, r.to)).join("");
      if (road) {
        parts.push(`<path d="${road}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:7px;stroke-opacity:.9"/>`);
        parts.push(`<path d="${road}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:${col};stroke-width:3.5px"/>`);
      }
      if (bad) {
        parts.push(`<path d="${bad}" fill="none" style="stroke:var(--map-land);stroke-width:3.5px;stroke-opacity:.9"/>`);
        parts.push(`<path d="${bad}" fill="none" stroke-linecap="butt" style="stroke:var(--bad, #d33);stroke-width:3.5px;stroke-dasharray:7 4"/>`);
      }
      if (b.dim[0] >= 0 && b.hoverRoad) parts.push(`<path d="${line(b.dim[0], b.dim[1])}" fill="none" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:2px;stroke-opacity:.45"/>`);
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
      // T-2993: the transfers the route will have, dashed as a built line's (busLayer.ts)
      for (let i = 0; i + 3 < this.links.length; i += 4) {
        const a = xy({ lng: this.links[i], lat: this.links[i + 1] }), z = xy({ lng: this.links[i + 2], lat: this.links[i + 3] });
        parts.push(`<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${z.x.toFixed(1)}" y2="${z.y.toFixed(1)}" class="bus-link" style="stroke:var(--map-ink);stroke-opacity:.5;stroke-width:1.5px;stroke-dasharray:4 2.7"/>`);
      }
      for (const s of q.stops) {
        if (s.kind === 0) parts.push(square(s));
        else if (s.kind === 1) {
          const p = xy(s);
          parts.push(`<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="9" style="fill:none;stroke:${col};stroke-width:2.5px"/>`);
        } else parts.push(ring(s, 3.6, 1.8));
      }
    } else if (hover && n && this.pts.length === 0 && b.ready && !b.hoverRefused) parts.push(square({ lng: b.ll[0], lat: b.ll[1] }));
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
    this.placeTag(q, n, b);
  }

  private placeTag(q: Quote | null, n: number, b: Built) {
    const tag = this.tag;
    const row = (text: string, bad = false) => Object.assign(document.createElement("div"), { textContent: text, className: bad ? "bad" : "" });
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
      const finishes = this.pts.length >= 2 && (this.isSeam(h) || (!this.looped() && h === this.endIndex()));
      const first = row(finishes ? t(this.line === null ? "busDraw.clickMake" : "busDraw.clickSave") : t("busDraw.clickDelete"));
      const right = { open: this.closed() ? "busDraw.rightOpen" : "busDraw.rightStopLooping", delete: "busDraw.rightDelete", close: "busDraw.rightClose", loopBack: "busDraw.rightLoopBack" } as const;
      const a = this.rightAction(h);
      return show([first, row(t("busDraw.dragMove")), ...(a ? [row(t(right[a]))] : [])]);
    }
    // where a click (or a drop) would be refused, the tag says why before it
    const d = this.drag;
    const refusal = d ? [d.i - 1, d.i].map((k) => this.legs[k]?.refused).find((r) => r) : b.hoverRefused;
    if (at && refusal) return show([row(refusalTag(refusal), true)], true);
    if (!at || !q || n < 2) {
      tag.style.display = "none";
      return;
    }
    if (!q.ok) return show([row(t("busDraw.minLength", { length: length(MIN_ROUTE_M) }))], true);
    // a built route redrawn pays only for the road it adds
    const pay = this.was.built ? Math.max(0, q.cost - this.was.road / 1000) : q.cost;
    const rows = [row(t("busDraw.roadCost", { distance: distance(q.road), cost: moneyShort(pay * 1e6), more: this.was.built })), row(t("busDraw.stops", { count: q.stops.length + q.street.length - (this.looped() ? 1 : 0) }))];
    const red = this.legs.find((l) => l.refused)?.refused;
    if (red) rows.unshift(row(refusalTag(red), true));
    show(rows);
  }
}

/** metres from `q` to the segment (x0, y0)-(x1, y1), longitude and latitude */
function segM(x0: number, y0: number, x1: number, y1: number, q: Pt): number {
  const k = Math.cos((q.lat * Math.PI) / 180), ax = (x1 - x0) * k, ay = y1 - y0, bx = (q.lng - x0) * k, by = q.lat - y0;
  const l2 = ax * ax + ay * ay, f = l2 > 0 ? Math.max(0, Math.min(1, (ax * bx + ay * by) / l2)) : 0;
  return Math.hypot(bx - f * ax, by - f * ay) * 111000;
}

/** A leg run the other way: its points backward, each stretch's trunk flag with it (T-3109). */
function reversedLeg(leg: Leg): Leg {
  const pts: number[] = [];
  for (let i = leg.pts.length - 2; i >= 0; i -= 2) pts.push(leg.pts[i], leg.pts[i + 1]);
  return { ...leg, pts, ...(leg.fast ? { fast: [...leg.fast].reverse() } : {}) };
}
