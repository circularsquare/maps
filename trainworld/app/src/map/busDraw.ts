// Drawing a bus line (T-1464, design T-1445): click points, straight between them, no track and no
// snapping. The clock worker quotes the route as it is drawn (`busQuote`): road length, price and
// where the stops would go; the stops and the tag by the pointer show that answer. Double click or
// Enter makes the line (`addBusLine`), or, with an existing line's route loaded (`begin(line, ...)`),
// redraws it (`busRoute`). Backspace and right click take back the last point, Escape drops the lot.
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

/** the sim's `bus::MIN_ROUTE_M` */
const MIN_ROUTE_M = 100;
/** the worker's "no line yet" id for a quote */
const NEW_LINE = 0xffffffff;
/** street stops are drawn from this zoom */
const STREET_ZOOM = 12.5;
/** a click this close (px) to the last point adds nothing */
const SAME_PX = 4;

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

const routeOf = (pts: Pt[]): number[] => pts.flatMap((p) => { const q = lngLatToPack(p.lng, p.lat); return [q.x, q.y]; });

export class BusDraw {
  private pts: Pt[] = [];
  /** the pointer (mouse), and the point a tap previews */
  private at: Pt | null = null;
  private touchAt: Pt | null = null;
  /** the line whose route is redrawn, null for a new line */
  private line: number | null = null;
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
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "bus-draw");
    this.svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible;display:none";
    box.insertBefore(this.svg, box.firstChild);
    map.on("contextmenu", (e) => {
      if (tool.peek() !== "bus") return;
      e.originalEvent.preventDefault();
      this.back();
    });
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

  /** Start the tool: a new line (`line` null) or the route of line number `line` as `routeLL`
   * (longitude, latitude pairs, flat) to redraw. The caller has left the tool before. */
  begin(line: number | null, routeLL: number[] = [], colour?: string) {
    this.pts = [];
    for (let k = 0; k + 1 < routeLL.length; k += 2) this.pts.push({ lng: routeLL[k], lat: routeLL[k + 1] });
    this.line = line;
    this.at = this.touchAt = null;
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
    this.pts = [];
    this.at = this.touchAt = null;
    this.quote = null;
    this.wantKey = "";
    this.svg.style.display = "none";
    busDraft.value = null;
  }

  private changed() {
    const d = busDraft.peek();
    if (!d || d.line !== this.line || d.points !== this.pts.length) busDraft.value = { line: this.line, points: this.pts.length };
    this.ask();
    this.place();
  }

  // ---------------------------------------------------------------- the quote

  private hoverPt(): Pt | null {
    return touchInput.peek() ? this.touchAt : this.at;
  }

  private ask() {
    const all = [...this.pts, ...(this.hoverPt() ? [this.hoverPt()!] : [])];
    if (all.length < 2) {
      this.quote = null;
      this.wantKey = "";
      return;
    }
    const route = routeOf(all), key = route.join(",");
    if (key === this.wantKey) return;
    this.wantKey = key;
    this.wantRoute = route;
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

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    this.at = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    this.ask();
    this.place();
  }

  click(e: MapMouseEvent) {
    if (this.busy) return;
    // a double click: its first click placed the last point
    if (e.originalEvent.detail >= 2) {
      if (this.pts.length >= 2) void this.finish();
      return;
    }
    this.add({ lng: e.lngLat.lng, lat: e.lngLat.lat });
  }

  private add(p: Pt) {
    const last = this.pts[this.pts.length - 1];
    if (last) {
      const a = this.map.project([last.lng, last.lat]), b = this.map.project([p.lng, p.lat]);
      if (Math.hypot(a.x - b.x, a.y - b.y) < SAME_PX) return;
    }
    this.pts.push(p);
    this.changed();
  }

  /** Take back the last point; with none placed, leave the tool. */
  private back() {
    if (this.busy) return;
    if (this.pts.length === 0) return this.host.exit();
    this.pts.pop();
    this.changed();
  }

  /** Keys: true if used. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "Escape") {
      // a redrawn route is dropped as a whole; a new line's points are, and the next Esc leaves
      if (this.line !== null || this.pts.length === 0) this.host.exit();
      else {
        this.pts = [];
        this.touchAt = null;
        this.changed();
      }
      return true;
    }
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
    const near = (q: Pt | null) => {
      if (!q) return false;
      const s = this.map.project([q.lng, q.lat]);
      return Math.hypot(s.x - e.point.x, s.y - e.point.y) < SAME_TAP_PX;
    };
    if (near(this.touchAt)) return this.confirm();
    // the last point tapped again, with nothing waiting, finishes
    if (!this.touchAt && this.pts.length >= 2 && near(this.pts[this.pts.length - 1])) return void this.finish();
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

  /** the touch bar's back: the point waiting, else the last point, else the tool */
  touchBack() {
    if (this.touchAt) {
      this.touchAt = null;
      this.changed();
    } else this.back();
  }

  touchState(): TouchCtl | null {
    if (!touchInput.peek() || !this.pts.length) return null;
    const waiting = !!this.touchAt;
    return { confirm: waiting || this.pts.length >= 2 ? !this.busy : null, confirmTip: waiting ? "Add this point" : "Make the line", back: true, stopTip: "Cancel", level: false, track: false };
  }

  /** the point the touch bar stands beside */
  touchPoint(): Pt | null {
    return this.touchAt ?? this.pts[this.pts.length - 1] ?? null;
  }

  // ---------------------------------------------------------------- finishing

  /** Make the line, or redraw the route, from the points placed; a route that makes no line is
   * refused with the reason. */
  async finish() {
    if (this.busy || tool.peek() !== "bus") return;
    if (this.pts.length < 2) {
      say("Click at least two points to draw a bus line.", true);
      return;
    }
    this.busy = true;
    try {
      const route = routeOf(this.pts), line = this.line;
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

  /** Draw the route, its stops and the tag; after each map frame and each change. */
  place() {
    if (tool.peek() !== "bus") {
      this.svg.style.display = "none";
      return;
    }
    const hover = this.hoverPt();
    const all = hover ? [...this.pts, hover] : this.pts;
    const q = this.quote && all.length >= 2 ? this.quote : null;
    const xy = (p: Pt) => { const s = this.map.project([p.lng, p.lat]); return `${s.x.toFixed(1)} ${s.y.toFixed(1)}`; };
    const col = this.colour;
    const parts: string[] = [];
    if (all.length >= 2) {
      const d = "M" + all.map(xy).join("L");
      const dim = hover && this.pts.length >= 1 ? "M" + [this.pts[this.pts.length - 1], hover].map(xy).join("L") : "";
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:7px;stroke-opacity:.9"/>`);
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:${col};stroke-width:3.5px"/>`);
      if (dim) parts.push(`<path d="${dim}" fill="none" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:2px;stroke-opacity:.45"/>`);
    }
    // the corners the player clicked
    for (const p of this.pts) {
      const [x, y] = xy(p).split(" ");
      parts.push(`<circle cx="${x}" cy="${y}" r="2.6" style="fill:${col};stroke:var(--map-land);stroke-width:1px"/>`);
    }
    const ring = (p: Pt, r: number, w: number) => {
      const [x, y] = xy(p).split(" ");
      return `<circle cx="${x}" cy="${y}" r="${r}" style="fill:var(--map-land);stroke:${col};stroke-width:${w}px"/>`;
    };
    if (q) {
      if (this.map.getZoom() >= STREET_ZOOM) for (const s of q.street) parts.push(ring(s, 2.4, 1.3));
      for (const s of q.stops) {
        const [x, y] = xy(s).split(" ");
        if (s.kind === 0) parts.push(`<rect x="${+x - 4.5}" y="${+y - 4.5}" width="9" height="9" rx="2.5" style="fill:var(--map-land);stroke:${col};stroke-width:2.2px"/>`);
        else if (s.kind === 1) parts.push(`<circle cx="${x}" cy="${y}" r="9" style="fill:none;stroke:${col};stroke-width:2.5px"/>`);
        else parts.push(ring(s, 3.6, 1.8));
      }
    } else if (hover && this.pts.length === 0) {
      const [x, y] = xy(hover).split(" ");
      parts.push(`<rect x="${+x - 4.5}" y="${+y - 4.5}" width="9" height="9" rx="2.5" style="fill:var(--map-land);stroke:${col};stroke-width:2.2px"/>`);
    }
    this.svg.style.display = "";
    this.svg.setAttribute("width", String(this.overlay.cssW));
    this.svg.setAttribute("height", String(this.overlay.cssH));
    this.svg.innerHTML = parts.join("");
    this.placeTag(q, hover, all.length);
  }

  private placeTag(q: Quote | null, hover: Pt | null, n: number) {
    const tag = this.tag;
    if (!hover || !q || n < 2) {
      tag.style.display = "none";
      return;
    }
    const row = (text: string) => Object.assign(document.createElement("div"), { textContent: text });
    tag.style.display = "";
    if (!q.ok) {
      tag.className = "tool-tag build-tag bad";
      tag.replaceChildren(row(`A bus route needs at least ${length(MIN_ROUTE_M)}.`));
    } else {
      tag.className = "tool-tag build-tag";
      // a built route redrawn pays only for the road it adds
      const pay = this.was.built ? Math.max(0, q.cost - this.was.road / 1000) : q.cost;
      tag.replaceChildren(
        row(`${distance(q.road)} of road, ${moneyShort(pay * 1e6)}${this.was.built ? " more" : ""}`),
        row(plural(q.stops.length + q.street.length, "stop")),
      );
    }
    const s = this.map.project([hover.lng, hover.lat]);
    tag.style.transform = `translate(${Math.round(s.x + 14)}px, ${Math.round(s.y + 12)}px)`;
  }
}
