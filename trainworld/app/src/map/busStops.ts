// T-2891: the stops of a bus line placed by hand, edited on the map with a train's stop markers
// (map/stopMarkers.ts, the same marks and tag) while the line tool has the line. A stop is a disc;
// along the route a hollow marker follows the pointer. Stops go only on the route (T-2982): a station
// has no mark and takes no click, and the route within reach of one offers the station's stop, the
// marker snapping to where the route passes nearest it; away from stations, so does another bus line's
// stop within `BUS_XFER_M` of the route (T-2992), its line named in the tag. Like the train markers they take no pointer
// events: map/tools.ts calls `hover` and `click`.

import { t } from "../i18n";
import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { lineDraft, stopEditing } from "../game/ui";
import { edit, stationNames, world } from "../game/world";
import type { ClockBus } from "../workers/protocol";
import { toMerc01 } from "./geo";
import { BUS_XFER_M } from "../game/busLinks";
import { tagLines } from "./labelLayout";
import type { Overlay } from "./overlay";
import { sizeAt } from "./stopMarkers";

/** The route takes a click this near, CSS px. */
const ROUTE_SLOP_PX = 8;
/** `bus::STATION_REACH_M` and `HAND_GAP_M`: a stop placed this near a station is that station's, this
 * near another stop is none. */
const STATION_REACH_M = 200;
const HAND_GAP_M = 50;

/** What is under the pointer: a point of the route (`name` the station whose stop it would be) */
type Hit =
  | { kind: "end" }
  | { kind: "stop"; x: number; y: number; name: string | undefined }
  | { kind: "route"; x: number; y: number; mx: number; my: number; name?: string };

/** A mark on the map: where it is and what a click on it does. */
interface Item {
  mx: number;
  my: number;
  on: boolean;
  el: HTMLDivElement;
  /** on screen at the last `place` */
  sx: number;
  sy: number;
  shown: boolean;
  hit: Hit;
}

const metres = (a: { lng: number; lat: number }, b: { lng: number; lat: number }) =>
  Math.hypot((a.lng - b.lng) * 111_320 * Math.cos((a.lat * Math.PI) / 180), (a.lat - b.lat) * 111_320);

/** A point of a route, in the network's plane and in longitude and latitude. */
interface RoutePoint { x: number; y: number; lng: number; lat: number }

/** The point `k` of the way along route stretch `i`. */
function routePoint(bus: ClockBus, i: number, k: number): RoutePoint {
  const r = bus.route, ll = bus.routeLL;
  return {
    x: r[2 * i] + k * (r[2 * i + 2] - r[2 * i]), y: r[2 * i + 1] + k * (r[2 * i + 3] - r[2 * i + 1]),
    lng: ll[2 * i] + k * (ll[2 * i + 2] - ll[2 * i]), lat: ll[2 * i + 1] + k * (ll[2 * i + 3] - ll[2 * i + 1]),
  };
}

/** The stretch of a route (x, y flat, plane) nearest a plane point, and how far along it: as the sim's `Poly::project`. */
function nearestOnRoute(r: number[], x: number, y: number): [number, number] {
  let best = Infinity, at: [number, number] = [0, 0];
  for (let i = 0; i + 3 < r.length; i += 2) {
    const ax = r[i], ay = r[i + 1], dx = r[i + 2] - ax, dy = r[i + 3] - ay, l2 = dx * dx + dy * dy;
    const k = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    const d = Math.hypot(ax + k * dx - x, ay + k * dy - y);
    if (d < best) (best = d), (at = [i >> 1, k]);
  }
  return at;
}

export class BusStopMarkers {
  private box: HTMLDivElement;
  private tag: HTMLDivElement;
  private ghost: HTMLDivElement;
  private ghostAt: [number, number] | null = null;
  private line: number | null = null;
  private bus: ClockBus | null = null;
  private items: Item[] = [];
  private pool: HTMLDivElement[] = [];
  private merc = new Float64Array(0);
  private size = 0;
  private scratch: [number, number] = [0, 0];
  private names = new Map<string, string>();
  /** the stations the route can stop at, each at the route's point nearest it */
  private reachOn: { p: RoutePoint; node: number }[] = [];
  /** T-2992: other bus lines' stops near enough the route to change at, each at the route's point
   * nearest it, with the line's name */
  private meetOn: { p: RoutePoint; name: string }[] = [];

  constructor(private map: MlMap, private overlay: Overlay) {
    this.box = document.createElement("div");
    this.box.className = "stop-marks";
    this.box.style.display = "none";
    this.ghost = document.createElement("div");
    this.ghost.className = "stop-mark route";
    this.ghost.style.display = "none";
    this.box.appendChild(this.ghost);
    this.tag = document.createElement("div");
    this.tag.className = "tool-tag stop-tag";
    this.tag.style.display = "none";
    map.getContainer().append(this.box, this.tag);
    map.getContainer().addEventListener("mouseleave", () => this.leave());
    overlay.afterMapFrame.push(() => this.place());
    effect(() => {
      const id = stopEditing.value ? lineDraft.value.line : null;
      const w = world.value;
      const l = id !== null && w ? w.save.lines.find((x) => x.num === id) : undefined;
      const bus = l && w ? w.lineStats[l.id]?.bus : undefined;
      this.load(l && bus?.hand ? l.num : null, bus ?? null, l?.colour);
    });
  }

  /** Whether a by-hand bus line is being edited, so these markers have the clicks. */
  get active() {
    return this.line !== null;
  }

  private load(line: number | null, bus: ClockBus | null, colour?: string) {
    this.line = line;
    this.bus = line === null ? null : bus;
    this.box.style.display = line === null ? "none" : "";
    if (colour) this.box.style.setProperty("--c", colour);
    this.items.forEach((m) => { m.el.style.display = "none"; });
    this.items = [];
    if (line === null || !bus) {
      this.leave();
      return;
    }
    const names = stationNames(world.peek()!);
    const merc = new Float64Array(bus.routeLL.length);
    for (let i = 0; i + 1 < bus.routeLL.length; i += 2) {
      const [mx, my] = toMerc01(bus.routeLL[i], bus.routeLL[i + 1]);
      merc[i] = mx;
      merc[i + 1] = my;
    }
    this.merc = merc;
    this.names = names;
    this.reachOn = bus.reach.map((s) => ({ p: routePoint(bus, ...nearestOnRoute(bus.route, s.x, s.y)), node: s.node }));
    this.meetOn = [];
    const w = world.peek()!;
    for (const o of w.save.lines) {
      const ob = w.lineStats[o.id]?.bus;
      if (o.num === line || !ob || ob.ferry) continue;
      for (const s of ob.stops) {
        if (s.station >= 0) continue;
        const p = routePoint(bus, ...nearestOnRoute(bus.route, s.x, s.y));
        if (metres(p, s) <= BUS_XFER_M) this.meetOn.push({ p, name: o.name });
      }
    }
    const add = (lng: number, lat: number, on: boolean, hit: Hit) => {
      const [mx, my] = toMerc01(lng, lat);
      const el = this.pool[this.items.length] ?? (this.pool[this.items.length] = this.box.appendChild(document.createElement("div")));
      el.className = "stop-mark " + (on ? "on" : "route");
      this.items.push({ mx, my, on, el, sx: 0, sy: 0, shown: false, hit });
    };
    bus.stops.forEach((s, i) => {
      const end = i === 0 || i === bus.stops.length - 1;
      add(s.lng, s.lat, true, end ? { kind: "end" } : { kind: "stop", x: s.x, y: s.y, name: names.get(String(s.station)) });
    });
    this.place();
    this.overlay.invalidate();
  }

  /** Move the marks to the camera: after each MapLibre frame and each change. */
  place() {
    if (this.line === null || !this.overlay.camMatrix) return;
    const size = Math.round(sizeAt(this.map.getZoom()));
    if (size !== this.size) {
      this.size = size;
      this.box.style.setProperty("--ms", size + "px");
    }
    const p = this.scratch;
    for (const m of this.items) {
      this.overlay.mercToScreen(m.mx, m.my, p);
      const on = p[0] > -size && p[1] > -size && p[0] < this.overlay.cssW + size && p[1] < this.overlay.cssH + size;
      if (on !== m.shown) {
        m.el.style.display = on ? "" : "none";
        m.shown = on;
      }
      if (!on) continue;
      m.sx = p[0];
      m.sy = p[1];
      m.el.style.transform = `translate(${Math.round(p[0])}px, ${Math.round(p[1])}px)`;
    }
    if (this.ghostAt) {
      this.overlay.mercToScreen(this.ghostAt[0], this.ghostAt[1], p);
      this.ghost.style.transform = `translate(${Math.round(p[0])}px, ${Math.round(p[1])}px)`;
    }
  }

  /** The point of the route nearest a map point (CSS px) within the slop: which stretch, how far along it. */
  private routeAt(px: number, py: number): [number, number] | null {
    const m = this.merc, p = this.scratch;
    let best = ROUTE_SLOP_PX * ROUTE_SLOP_PX, at: [number, number] | null = null;
    this.overlay.mercToScreen(m[0], m[1], p);
    let [ax, ay] = p;
    for (let i = 1; i < m.length >> 1; i++) {
      this.overlay.mercToScreen(m[2 * i], m[2 * i + 1], p);
      const [bx, by] = p;
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const k = l2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
      const ex = ax + k * dx - px, ey = ay + k * dy - py, d2 = ex * ex + ey * ey;
      if (d2 < best) (best = d2), (at = [i - 1, k]);
      (ax = bx), (ay = by);
    }
    return at;
  }

  private pick(px: number, py: number): Hit | null {
    const bus = this.bus;
    if (!bus) return null;
    let best: Item | null = null, bestD = Infinity;
    for (const m of this.items) {
      if (!m.shown) continue;
      const d = Math.hypot(m.sx - px, m.sy - py);
      if (d > this.size / 2 + 2) continue;
      if (d < bestD) (best = m), (bestD = d);
    }
    if (best) return best.hit;
    const at = this.routeAt(px, py);
    if (!at) return null;
    const here = routePoint(bus, ...at);
    // on top of a stop a click adds nothing, and near a station's stop it is that stop
    if (bus.stops.some((s) => metres(s, here) < (s.station >= 0 ? STATION_REACH_M : HAND_GAP_M))) return null;
    // T-2982: near a station, the station's stop, where the route passes nearest the station
    let reach = STATION_REACH_M, near: { p: RoutePoint; node: number } | null = null;
    for (const s of this.reachOn) {
      const d = metres(s.p, here);
      if (d <= reach) (reach = d), (near = s);
    }
    // T-2992: else near another bus line's stop, where the route passes nearest it
    let meet: { p: RoutePoint; name: string } | null = null;
    if (!near) {
      let gap = BUS_XFER_M;
      for (const s of this.meetOn) {
        const d = metres(s.p, here);
        if (d <= gap) (gap = d), (meet = s);
      }
    }
    const p = near?.p ?? meet?.p ?? here, [mx, my] = toMerc01(p.lng, p.lat);
    const name = near ? this.names.get(String(near.node)) ?? t("stopMarkers.station") : meet?.name || undefined;
    return { kind: "route", x: p.x, y: p.y, mx, my, ...(name ? { name } : {}) };
  }

  /** The pointer moved over the map: the follower along the route, the tag and the cursor. */
  hover(px: number, py: number) {
    const hit = this.pick(px, py);
    this.ghostAt = hit?.kind === "route" ? [hit.mx, hit.my] : null;
    this.ghost.style.display = this.ghostAt ? "" : "none";
    this.map.getCanvas().style.cursor = hit && hit.kind !== "end" ? "pointer" : "crosshair";
    if (!hit || hit.kind === "end") return this.hideTag();
    const lines = hit.kind === "stop" ? [...(hit.name ? [hit.name] : []), t("stopMarkers.remove")]
      : [...(hit.name ? [hit.name] : []), t("stopMarkers.add")];
    tagLines(this.tag, lines);
    this.tag.style.display = "";
    this.tag.style.transform = `translate(${Math.round(px + 14)}px, ${Math.round(py + 12)}px)`;
    this.place();
  }

  /** A click on the map. */
  click(px: number, py: number) {
    const hit = this.pick(px, py);
    if (this.line === null || !hit || hit.kind === "end") return;
    this.hideTag();
    void edit({ op: "busHandStop", line: this.line, x: hit.x, y: hit.y, on: hit.kind !== "stop" });
  }

  private hideTag() {
    this.tag.style.display = "none";
  }

  private leave() {
    this.hideTag();
    this.ghostAt = null;
    this.ghost.style.display = "none";
  }
}
