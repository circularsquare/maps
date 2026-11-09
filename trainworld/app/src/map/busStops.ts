// T-2891: the stops of a bus line placed by hand, edited on the map with a train's stop markers
// (map/stopMarkers.ts, the same marks and tag) while the line tool has the line. A stop is a disc; a
// station the route can stop at is the hollow marker; along the route a hollow marker follows the
// pointer. Like the train markers they take no pointer events: map/tools.ts calls `hover` and `click`.

import { t } from "../i18n";
import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { lineDraft, stopEditing } from "../game/ui";
import { edit, stationNames, world } from "../game/world";
import type { ClockBus } from "../workers/protocol";
import { toMerc01 } from "./geo";
import { tagLines } from "./labelLayout";
import type { Overlay } from "./overlay";
import { NEAR_SCALE, sizeAt } from "./stopMarkers";

/** The route takes a click this near, CSS px. */
const ROUTE_SLOP_PX = 8;
/** `bus::STATION_REACH_M` and `HAND_GAP_M`: a stop placed this near a station is that station's, this
 * near another stop is none. */
const STATION_REACH_M = 200;
const HAND_GAP_M = 50;

/** What is under the pointer. */
type Hit =
  | { kind: "end" }
  | { kind: "stop"; x: number; y: number; name: string | undefined }
  | { kind: "station"; x: number; y: number; name: string }
  | { kind: "route"; x: number; y: number; mx: number; my: number };

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
    const stopped = new Set(bus.stops.map((s) => s.station));
    const add = (lng: number, lat: number, on: boolean, hit: Hit) => {
      const [mx, my] = toMerc01(lng, lat);
      const el = this.pool[this.items.length] ?? (this.pool[this.items.length] = this.box.appendChild(document.createElement("div")));
      el.className = "stop-mark " + (on ? "on" : "route");
      this.items.push({ mx, my, on, el, sx: 0, sy: 0, shown: false, hit });
    };
    for (const r of bus.reach) if (!stopped.has(r.node)) add(r.lng, r.lat, false, { kind: "station", x: r.x, y: r.y, name: names.get(String(r.node)) ?? t("stopMarkers.station") });
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
      const r = (m.on ? this.size : this.size * NEAR_SCALE) / 2 + 2;
      if (d > r) continue;
      // a stop over the station marker beside it
      if ((m.on ? 0 : 1) * 1e4 + d < bestD) (best = m), (bestD = (m.on ? 0 : 1) * 1e4 + d);
    }
    if (best) return best.hit;
    const at = this.routeAt(px, py);
    if (!at) return null;
    const [i, k] = at, r = bus.route, ll = bus.routeLL;
    const x = r[2 * i] + k * (r[2 * i + 2] - r[2 * i]), y = r[2 * i + 1] + k * (r[2 * i + 3] - r[2 * i + 1]);
    const here = { lng: ll[2 * i] + k * (ll[2 * i + 2] - ll[2 * i]), lat: ll[2 * i + 1] + k * (ll[2 * i + 3] - ll[2 * i + 1]) };
    // on top of a stop a click adds nothing, and near a station's stop it is that stop
    if (bus.stops.some((s) => metres(s, here) < (s.station >= 0 ? STATION_REACH_M : HAND_GAP_M))) return null;
    const station = bus.reach.find((s) => metres(s, here) <= STATION_REACH_M);
    if (station) return this.items.find((m) => m.hit.kind === "station" && m.hit.x === station.x && m.hit.y === station.y)?.hit ?? null;
    const [mx, my] = toMerc01(here.lng, here.lat);
    return { kind: "route", x, y, mx, my };
  }

  /** The pointer moved over the map: the follower along the route, the tag and the cursor. */
  hover(px: number, py: number) {
    const hit = this.pick(px, py);
    this.ghostAt = hit?.kind === "route" ? [hit.mx, hit.my] : null;
    this.ghost.style.display = this.ghostAt ? "" : "none";
    this.map.getCanvas().style.cursor = hit && hit.kind !== "end" ? "pointer" : "crosshair";
    if (!hit || hit.kind === "end") return this.hideTag();
    const lines = hit.kind === "stop" ? [...(hit.name ? [hit.name] : []), t("stopMarkers.remove")]
      : hit.kind === "station" ? [hit.name, t("stopMarkers.add")]
      : [t("stopMarkers.add")];
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
