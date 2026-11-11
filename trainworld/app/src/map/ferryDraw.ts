// Drawing a ferry line (T-2194, design notes/T-2177.md 3.2): the player places ports only. A click near
// the shore, on land or on the water, puts a new port on the nearest shore (its landing, and its berth in
// the water, from the water worker, `ferry/ferryWater.ts`), shown under the pointer before the click
// (T-2468); a click on or near a port the network has stops there. Between consecutive ports
// the worker finds the route over the basemap's water (T-2192); a leg it cannot find is drawn straight and
// red, and the line cannot be made until it is moved. With an existing line's ports loaded
// (`begin(line, ...)`) finishing redraws it (`ferryRoute`), else it makes the line (`addFerryLine`).
//
// The bus tool's handling where it fits (busDraw.ts, T-3118): new ports go on at the active end, a click on a port
// takes it out, a click on the active end, a double click, Enter, Escape or the Make line button finishes a new line,
// Escape drops an edit as a whole. A finger's tap places at once (a port is a place, not a point to adjust); the
// Loop button makes the next tap a right click (`loopTap`). No dragging: a port is moved by taking it out and placing
// another. A right click loops: on the other end, with three ports or more, it closes the line into a loop and
// finishes it (T-2899): its last port is then the first again; on a port partway along it joins the active end back
// to it and finishes, the boats running out, round and back: sent as the ports out, round and back again
// (`runPorts`), folded back here (`begin`). A looped line has no active end; a right click on any port of a loop
// opens it there (the leg leaving the port goes), on any port of a line that loops back it stops looping, and on the
// active end it takes that port out.
//
// The tool owns no map layer: the route and ports are an SVG over the map in screen pixels, like the bus
// tool's.

import { t } from "../i18n";
import { signal } from "@preact/signals";
import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { lngLatToPack } from "../game/coords";
import { distance, moneyShort } from "../game/format";
import { newFerryLine, nextFreeColour } from "../game/lineActions";
import { moneyState } from "../game/money";
import { ask, loopTap, renameLinePrompt, selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import { metresBetween } from "../game/walks";
import { keyIsTheirs } from "../ui/keys";
import { FERRY_KEY } from "../ui/toolKeys";
import { FerryWater, type LineRoute, type PortSnap } from "../ferry/ferryWater";
import { portReachM } from "../ferry/route";
import type { FerryPortIn } from "../workers/protocol";
import type { Overlay } from "./overlay";
import { newStationName } from "./stationNames";
import { touchInput, type TouchCtl } from "./touch";
import { safeColour } from "./safeColour";
import { playBuild } from "./buildSound";

interface Pt { lng: number; lat: number }

/** A port of the line being drawn: one the network has (`node`), or a new one with its name to come. */
interface DraftPort {
  node: number | null;
  landing: Pt;
  berth: Pt;
  name: string;
  /** a new port's name, while the places list is asked */
  naming?: Promise<string>;
}

/** What a click where the pointer is would do. */
type Under = { kind: "handle"; i: number } | { kind: "port"; node: number; name: string; landing: Pt; berth: Pt }
  | { kind: "shore"; landing: Pt; berth: Pt } | { kind: "water" } | { kind: "far" } | { kind: "unknown" };

/** a click this near (px) to a port's circle is on it; a port within `PORT_NEAR_M` of the click too */
const HANDLE_PX = 10;
const TOUCH_HANDLE_PX = 18;
const PORT_NEAR_M = 30;

const farFromWater = () => t("ferryDraw.tagFar");
const onWater = () => t("ferryDraw.tagOnWater");
const noWaterPath = () => t("ferryDraw.tagPath");

/** What the tool is doing, for the panels: null while it is off. `line` the line whose route is
 * redrawn (its number), null for a new line; `points` the ports placed so far. */
export const ferryDraft = signal<{ line: number | null; points: number } | null>(null);

const same = (a: Pt, b: Pt) => a.lng === b.lng && a.lat === b.lat;

/** A route as the sim takes it: x, y, kind and speed per point. */
function routeArray(r: LineRoute): number[] {
  const route: number[] = [];
  for (let i = 0; i < r.kinds.length; i++) {
    const q = lngLatToPack(r.ll[2 * i], r.ll[2 * i + 1]);
    route.push(q.x, q.y, r.kinds[i], r.speeds[i]);
  }
  return route;
}

export class FerryDraw {
  readonly water: FerryWater;
  private ports: DraftPort[] = [];
  /** new ports go before the first (true) or after the last */
  private head = false;
  /** the route through `ports`' berths, and the berths it was found for */
  private route: LineRoute | null = null;
  private routeKey = "";
  private routing: Promise<void> | null = null;
  /** the pointer, what is under it, and the probe of the water there */
  private at: Pt | null = null;
  private pt: { x: number; y: number } | null = null;
  private under: Under = { kind: "unknown" };
  private probing = false;
  /** where a click at `at` would put a port (T-2468: the pointer's spot is snapped as a click is) */
  private probed: { at: Pt; reach: number; snap: PortSnap } | null = null;
  private line: number | null = null;
  private dirty = false;
  private colour = "#1a7fb0";
  /** a constructed line being redrawn: only its new ports are paid */
  private built = false;
  private busy = false;
  private svg: SVGSVGElement;

  constructor(private map: MlMap, box: HTMLElement, private tag: HTMLElement, private overlay: Overlay, private host: { exit(): void }) {
    this.water = new FerryWater(map);
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "ferry-draw");
    this.svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible;display:none";
    box.insertBefore(this.svg, box.firstChild);
    map.on("contextmenu", (e) => {
      if (tool.peek() !== "ferry") return;
      e.originalEvent.preventDefault();
      if (touchInput.peek()) return;
      const i = this.handleAt(e.point.x, e.point.y);
      if (i >= 0) this.rightClick(i);
      else this.back();
    });
    // O (FERRY_KEY): a new ferry line, from any tab, before main.ts's key handler
    window.addEventListener("keydown", (e) => {
      if (e.key.toLowerCase() !== FERRY_KEY || e.ctrlKey || e.metaKey || e.altKey || e.repeat || keyIsTheirs(e) || ask.peek() || !world.peek()) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (tool.peek() !== "ferry") newFerryLine();
    });
  }

  get active() {
    return tool.peek() === "ferry";
  }

  /** the pointer is on a placed port */
  get onHandle() {
    return this.active && this.under.kind === "handle";
  }

  /** Start the tool: a new line (`line` null) or line number `line`'s ports and route to redraw. The
   * caller has left the tool before. */
  begin(line: number | null, colour?: string) {
    this.ports = [];
    this.route = null;
    this.routeKey = "";
    const st = line === null ? undefined : world.peek()?.lineStats[String(line)]?.bus;
    const names = new Map((world.peek()?.save.stations ?? []).map((s) => [s.num, s.name]));
    if (st?.ferry) {
      for (const p of st.ferry.ports) {
        this.ports.push({ node: p.node, landing: { lng: p.landing[0], lat: p.landing[1] }, berth: { lng: p.berth[0], lat: p.berth[1] }, name: names.get(p.node) ?? "" });
      }
      // the saved route stands until a port changes (routes are never searched again on their own)
      this.route = { ok: true, ll: st.routeLL, kinds: st.kinds, speeds: st.ferry.speeds, m: st.roadM, legs: [] };
      // T-3118: a line that loops back is drawn with its way out once, its last port the one the loop comes back to
      const last = this.ports.length - 1, ps = this.ports;
      if (last >= 4 && ps[0].node === ps[last].node) {
        let k = 0;
        while (2 * (k + 1) < last && ps[k + 1].node === ps[last - k - 1].node) k++;
        if (k >= 1 && last - 2 * k >= 3) this.ports = ps.slice(0, last - k).concat([ps[k]]);
      }
      this.routeKey = this.berthKey();
    }
    this.head = false;
    this.line = line;
    this.dirty = false;
    this.built = !!st?.built;
    this.at = null;
    this.under = { kind: "unknown" };
    const c = this.map.getCenter();
    this.colour = colour ?? nextFreeColour([c.lng, c.lat]);
    tool.value = "ferry";
    this.changed();
  }

  /** the tool was left */
  clear() {
    this.ports = [];
    this.route = null;
    this.routeKey = "";
    this.at = null;
    this.under = { kind: "unknown" };
    this.svg.style.display = "none";
    ferryDraft.value = null;
  }

  private berthKey() {
    return this.runPorts().map((p) => `${p.berth.lng},${p.berth.lat}`).join("|");
  }

  /** T-3118: the ports the boats call at in order: a line that loops back runs its way out again backward. */
  private runPorts(): DraftPort[] {
    const k = this.joinAt();
    return k < 0 ? this.ports : this.ports.concat(this.ports.slice(0, k).reverse());
  }

  /** The ports changed (`edited`: by the player): the route is searched again. */
  private changed(edited = false) {
    if (edited) this.dirty = true;
    if (this.ports.length < 2) this.head = false;
    const d = ferryDraft.peek();
    const points = this.ports.length - (this.looped() ? 1 : 0);
    if (!d || d.line !== this.line || d.points !== points) ferryDraft.value = { line: this.line, points };
    this.reroute();
    this.place();
  }

  private reroute() {
    const key = this.berthKey();
    if (key === this.routeKey) return;
    if (this.ports.length < 2) {
      this.route = null;
      this.routeKey = key;
      return;
    }
    const pts = this.runPorts().map((p) => ({ at: [p.berth.lng, p.berth.lat] as [number, number], kind: 1 as const }));
    const run = this.water.line(pts, "draft").then((r) => {
      if (!r || key !== this.berthKey()) return;
      this.route = r.route;
      this.routeKey = key;
      if (this.routing === run) this.routing = null;
      if (this.active) this.place();
    });
    this.routing = run;
  }

  // ---------------------------------------------------------------- what is under the pointer

  private endIndex(): number {
    return this.head ? 0 : this.ports.length - 1;
  }

  /** The line is a loop: its last port is its first (T-2899). */
  private closed(): boolean {
    const n = this.ports.length;
    return n >= 4 && samePort(this.ports[0], this.ports[n - 1]);
  }

  /** T-3118: the line loops back: its last port is its port `k` partway along, three legs or more past it; -1 for none. */
  private joinAt(): number {
    const n = this.ports.length, z = this.ports[n - 1];
    for (let k = 1; k <= n - 4; k++) if (samePort(this.ports[k], z)) return k;
    return -1;
  }

  /** a loop, or a line that loops back: no end to add ports at */
  private looped(): boolean {
    return this.closed() || this.joinAt() >= 0;
  }

  /** The placed port whose circle is under a screen point, -1 for none; the active end first. */
  private handleAt(px: number, py: number, reach = HANDLE_PX): number {
    let best = reach * reach, hit = -1;
    this.ports.forEach((p, i) => {
      const s = this.map.project([p.landing.lng, p.landing.lat]);
      const d = (s.x - px) ** 2 + (s.y - py) ** 2;
      if (d < best || (d < reach * reach && i === this.endIndex())) {
        best = i === this.endIndex() ? -1 : d;
        hit = i;
      }
    });
    return hit;
  }

  /** The network's port nearest a point within reach, or null. */
  private portAt(px: number, py: number, at: Pt, reach = HANDLE_PX): Extract<Under, { kind: "port" }> | null {
    const w = world.peek();
    if (!w) return null;
    let best = Infinity, hit: Extract<Under, { kind: "port" }> | null = null;
    for (const s of w.save.stations) {
      if (!s.port) continue;
      const q = this.map.project([s.lng, s.lat]);
      const dpx = Math.hypot(q.x - px, q.y - py), dm = metresBetween(s.lng, s.lat, at.lng, at.lat);
      if ((dpx <= reach || dm <= PORT_NEAR_M) && dpx < best) {
        best = dpx;
        hit = { kind: "port", node: s.num, name: s.name, landing: { lng: s.lng, lat: s.lat }, berth: { lng: s.port[0], lat: s.port[1] } };
      }
    }
    return hit;
  }

  /** The network's port whose landing is within `PORT_NEAR_M` of a point, the nearest. */
  private portNear(p: Pt): Extract<Under, { kind: "port" }> | null {
    let best = PORT_NEAR_M, hit: Extract<Under, { kind: "port" }> | null = null;
    for (const s of world.peek()?.save.stations ?? []) {
      if (!s.port) continue;
      const dm = metresBetween(s.lng, s.lat, p.lng, p.lat);
      if (dm <= best) {
        best = dm;
        hit = { kind: "port", node: s.num, name: s.name, landing: { lng: s.lng, lat: s.lat }, berth: { lng: s.port[0], lat: s.port[1] } };
      }
    }
    return hit;
  }

  /** What a click at a screen point would do, as far as is known now (the water is asked after). */
  private underAt(px: number, py: number, at: Pt, touch = false): Under {
    const i = this.handleAt(px, py, touch ? TOUCH_HANDLE_PX : HANDLE_PX);
    if (i >= 0) return { kind: "handle", i };
    const port = this.portAt(px, py, at, touch ? TOUCH_HANDLE_PX : HANDLE_PX);
    if (port) return port;
    // the last answer stands while a newer one is asked, so the tag stays up and only its text changes (T-2273)
    const p = this.probed;
    return p ? this.underOf(p.snap) : { kind: "unknown" };
  }

  private underOf(s: PortSnap): Under {
    if (s.ok) {
      const landing = { lng: s.landing[0], lat: s.landing[1] };
      // a new port this close to one the network has would be that port
      return this.portNear(landing) ?? { kind: "shore", landing, berth: { lng: s.berth[0], lat: s.berth[1] } };
    }
    return { kind: s.reason === "on-water" ? "water" : s.reason === "tiles" ? "unknown" : "far" };
  }

  /** How far off the shore a click at `at` still finds it: a fixed number of screen pixels, so it holds at any zoom (T-3127). */
  private reachAt(at: Pt): number {
    return portReachM((78271.517 * Math.cos((at.lat * Math.PI) / 180)) / 2 ** this.map.getZoom());
  }

  private probeIs(at: Pt): boolean {
    return !!this.probed && same(this.probed.at, at) && this.probed.reach === this.reachAt(at);
  }

  /** Ask the worker where a click at the pointer would put a port; the newest position wins. */
  private probe() {
    if (this.probing || !this.at) return;
    const at = this.at, reach = this.reachAt(at);
    this.probing = true;
    void this.water.port([at.lng, at.lat], reach).then((snap) => {
      this.probing = false;
      if (snap) this.probed = { at, reach, snap };
      if (!this.active) return;
      const k = this.under.kind;
      if (snap && k !== "handle" && k !== "port") {
        this.under = this.underOf(snap);
        this.place();
      }
      if (this.at && !this.probeIs(this.at)) this.probe();
    });
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    this.at = { lng: e.lngLat.lng, lat: e.lngLat.lat };
    this.pt = { x: e.point.x, y: e.point.y };
    this.under = this.underAt(e.point.x, e.point.y, this.at);
    if (!this.probeIs(this.at)) this.probe();
    const cur = this.under.kind === "handle" || this.under.kind === "port" ? "pointer" : "crosshair";
    if (this.map.getCanvas().style.cursor !== cur) this.map.getCanvas().style.cursor = cur;
    this.place();
  }

  click(e: MapMouseEvent) {
    if (this.busy) return;
    if (e.originalEvent.detail >= 2) {
      if (this.ports.length >= 2) void this.finish();
      return;
    }
    void this.press(e.point.x, e.point.y, { lng: e.lngLat.lng, lat: e.lngLat.lat });
  }

  /** A click or a tap at a screen point. */
  private async press(px: number, py: number, at: Pt, touch = false) {
    const u = this.underAt(px, py, at, touch);
    if (u.kind === "handle") return this.handleClick(u.i);
    if (this.looped()) return void say(t("ferryDraw.loopClosed"), true);
    if (u.kind === "port") {
      if (this.ports.some((p) => p.node === u.node)) return;
      return this.add({ node: u.node, landing: u.landing, berth: u.berth, name: u.name });
    }
    this.busy = true;
    try {
      // the spot the pointer's preview showed, else asked now
      const snap = this.probeIs(at) ? this.probed!.snap : await this.water.port([at.lng, at.lat], this.reachAt(at));
      if (!this.active) return;
      if (!snap || (!snap.ok && snap.reason === "tiles")) return void say(t("ferryDraw.notLoaded"), true);
      if (!snap.ok) return void say(t(snap.reason === "on-water" ? "ferryDraw.sayOnWater" : "ferryDraw.sayFar"), true);
      const landing = { lng: snap.landing[0], lat: snap.landing[1] }, berth = { lng: snap.berth[0], lat: snap.berth[1] };
      const near = this.portNear(landing);
      if (near) {
        if (this.ports.some((p) => p.node === near.node)) return;
        return this.add({ node: near.node, landing: near.landing, berth: near.berth, name: near.name });
      }
      const taken = () => new Set([...(world.peek()?.save.stations.map((s) => s.name) ?? []), ...this.ports.map((p) => p.name)]);
      const p: DraftPort = { node: null, landing, berth, name: "" };
      p.naming = newStationName(this.map, landing.lng, landing.lat, null, taken).then((n) => (p.name = n), () => (p.name = t("ferryDraw.port")));
      this.add(p);
    } finally {
      this.busy = false;
    }
  }

  /** A click on a placed port (T-3118): the active end, or where a loop closes, finishes; any other port goes. With
   * the touch Loop button on, a right click's work instead. */
  private handleClick(i: number) {
    const n = this.ports.length;
    if (n < 2) return;
    if (loopTap.peek()) {
      loopTap.value = false;
      return this.rightClick(i);
    }
    if (this.isSeam(i) || (!this.looped() && i === this.endIndex())) return void this.finish();
    this.remove(i);
  }

  /** where a looped line closes: its last port and, for one that loops back, that port partway along */
  private isSeam(i: number): boolean {
    const n = this.ports.length;
    if (this.closed()) return i === 0 || i === n - 1;
    const k = this.joinAt();
    return k >= 0 && (i === k || i === n - 1);
  }

  /** What a right click on port `i` does (T-3118), null for nothing; as the bus tool's. */
  private rightAction(i: number): "open" | "stop" | "delete" | "close" | "loopBack" | null {
    const n = this.ports.length;
    if (n < 2) return null;
    if (this.closed()) return "open";
    if (this.looped()) return "stop";
    const end = this.endIndex();
    if (i === end) return "delete";
    if (i === 0 || i === n - 1) return n >= 3 ? "close" : null;
    return Math.abs(end - i) >= 2 ? "loopBack" : null;
  }

  private rightClick(i: number) {
    const a = this.rightAction(i);
    if (a === "open") this.openAt(i);
    else if (a === "stop") this.remove(this.ports.length - 1);
    else if (a === "delete") this.remove(i);
    else if (a === "close") this.closeLoop();
    else if (a === "loopBack") this.loopBack(i);
  }

  /** T-3118: a loop opened at port `i`: the leg leaving it goes, so the line runs on round from the next port and
   * ends there, its active end. */
  private openAt(i: number) {
    if (this.busy || !this.closed()) return;
    const m = this.ports.length - 1, c = this.ports.slice(0, m);
    this.ports = Array.from({ length: m }, (_, j) => c[((i % m) + 1 + j) % m]);
    this.head = false;
    playBuild("pop");
    this.reunder();
    this.changed(true);
  }

  /** T-3118: join the active end back to port `i` and make the line: the boats run out, round and back. */
  private loopBack(i: number) {
    if (this.busy) return;
    if (this.head) {
      this.ports.reverse();
      i = this.ports.length - 1 - i;
      this.head = false;
    }
    playBuild("station");
    this.ports.push(this.ports[i]);
    this.reunder();
    this.changed(true);
    void this.finish();
  }

  /** Add the first port again at the end and make the line (T-2899). */
  private closeLoop() {
    playBuild("station");
    this.ports.push(this.ports[0]);
    this.reunder();
    this.changed(true);
    void this.finish();
  }

  private add(p: DraftPort) {
    playBuild("station"); // T-2940: a port placed sounds as a station placed does
    if (this.head) this.ports.unshift(p);
    else this.ports.push(p);
    this.reunder();
    this.changed(true);
  }

  private remove(i: number) {
    if (this.busy || i < 0 || i >= this.ports.length) return;
    playBuild("pop");
    // where a loop closes its port is there twice: taking it out opens the loop; a ring of three, or a loop of three
    // past where a line loops back, is no loop without one
    const wasClosed = this.closed(), wasLooped = this.looped();
    if (this.isSeam(i)) this.ports.pop();
    else {
      this.ports.splice(i, 1);
      if ((wasClosed && this.ports.length <= 3) || (wasLooped && !this.looped())) this.ports.pop();
    }
    this.reunder();
    this.changed(true);
  }

  /** The ports changed under a still pointer. */
  private reunder() {
    this.under = this.pt && this.at ? this.underAt(this.pt.x, this.pt.y, this.at) : { kind: "unknown" };
  }

  /** Take back the active end; with no port placed, leave the tool. */
  private back() {
    if (this.busy) return;
    if (this.ports.length === 0) return this.host.exit();
    this.remove(this.endIndex());
  }

  /** Keys: true if used. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "Escape") {
      if (this.line === null && this.ports.length >= 2) void this.finish();
      else if (this.line !== null || this.ports.length === 0) this.host.exit();
      else {
        this.ports = [];
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
    if (this.busy) return;
    void this.press(e.point.x, e.point.y, { lng: e.lngLat.lng, lat: e.lngLat.lat }, true);
  }

  confirm() {
    void this.finish();
  }

  touchBack() {
    this.back();
  }

  touchState(): TouchCtl | null {
    if (!touchInput.peek() || !this.ports.length) return null;
    return { confirm: this.ports.length >= 2 ? !this.busy : null, confirmTip: this.line === null ? t("ferryDraw.makeLine") : t("ferryDraw.saveRoute"), back: true, stopTip: t("ferryDraw.cancel") };
  }

  touchPoint(): Pt | null {
    const p = this.ports[this.endIndex()];
    return p ? p.landing : null;
  }

  // ---------------------------------------------------------------- finishing

  /** The ports not yet constructed that the line would pay for, and their price, US$M. */
  private price(): { count: number; cost: number } {
    const w = world.peek(), m = moneyState.peek();
    const built = new Set((w?.save.stations ?? []).filter((s) => s.built).map((s) => s.num));
    const nodes = new Set<number>();
    let count = 0;
    for (const p of new Set(this.ports)) {
      if (p.node === null) count++;
      else if (!built.has(p.node) && !nodes.has(p.node)) {
        nodes.add(p.node);
        count++;
      }
    }
    return { count, cost: count * (m?.portPrice ?? 10) };
  }

  /** The leg that has no route, as its two ports' names, or null. */
  private refusedLeg(): string | null {
    const r = this.route;
    if (!r || this.routeKey !== this.berthKey()) return null;
    const k = r.legs.findIndex((l) => l.reason);
    if (k < 0) return null;
    const run = this.runPorts(), a = run[k]?.name, b = run[k + 1]?.name;
    return a && b ? t("ferryDraw.pathBetween", { a, b }) : t("ferryDraw.sayPath");
  }

  /** Make the line, or redraw the route, from the ports placed; refused with the reason. An edit that
   * changed nothing just closes. */
  async finish() {
    if (this.busy || tool.peek() !== "ferry") return;
    if (this.line !== null && !this.dirty) return this.host.exit();
    if (this.ports.length < 2) {
      say(t("ferryDraw.twoPorts"), true);
      return;
    }
    this.busy = true;
    try {
      while (this.routing) await this.routing;
      await Promise.all(this.ports.map((p) => p.naming));
      const run = this.runPorts();
      if (tool.peek() !== "ferry") return;
      const r = this.route;
      if (!r || this.routeKey !== this.berthKey()) return;
      const refused = this.refusedLeg();
      if (refused || !r.ok) {
        say(refused ?? t("ferryDraw.sayPath"), true);
        return;
      }
      const route = routeArray(r);
      const ports: FerryPortIn[] = run.map((p, i) => {
        if (p.node !== null) return { node: p.node };
        // a new port closing a loop, or called at again, is the one placed first
        if (run.indexOf(p) < i) return { again: run.indexOf(p) };
        const a = lngLatToPack(p.landing.lng, p.landing.lat), b = lngLatToPack(p.berth.lng, p.berth.lat);
        return { x: a.x, y: a.y, bx: b.x, by: b.y, name: p.name };
      });
      const cost = this.price().cost;
      const line = this.line;
      if (line === null) {
        const a = await edit({ op: "addFerryLine", ports, route, name: "", colour: this.colour }, true);
        if (!a.ok || a.line === undefined) {
          say(t("ferryDraw.addFailed"), true);
          return;
        }
        say(t("ferryDraw.added", { distance: distance(r.m), cost: moneyShort(cost * 1e6) }));
        this.host.exit();
        renameLinePrompt.value = String(a.line);
        selection.value = { kind: "line", line: String(a.line) };
      } else {
        const a = await edit({ op: "ferryRoute", line, ports, route }, true);
        if (!a.ok) {
          say(t("ferryDraw.changeFailed"), true);
          return;
        }
        say(t("ferryDraw.changed", { distance: distance(r.m) }));
        this.host.exit();
      }
    } finally {
      this.busy = false;
    }
  }

  /** T-2275: line `line`'s ports in a new order or fewer of them (`order`: indices into its ports now),
   * from the line panel's list: the route searched again through them and saved, one undoable edit.
   * False when refused (said why). */
  async setPorts(line: number, order: number[]): Promise<boolean> {
    const w = world.peek(), l = w?.save.lines.find((x) => x.num === line);
    const st = l ? w?.lineStats[l.id]?.bus : undefined;
    if (!l || !st?.ferry) return false;
    const ports = order.map((i) => st.ferry!.ports[i]).filter((p) => !!p);
    if (ports.length < 2) {
      say(t("ferryDraw.needTwoPorts"), true);
      return false;
    }
    const got = await this.water.line(ports.map((p) => ({ at: p.berth, kind: 1 as const })));
    if (!got) return false;
    const r = got.route, k = r.legs.findIndex((x) => x.reason);
    if (k >= 0 || !r.ok) {
      const names = new Map((world.peek()?.save.stations ?? []).map((s) => [s.num, s.name]));
      const a = names.get(ports[k]?.node ?? -1), b = names.get(ports[k + 1]?.node ?? -1);
      say(a && b ? t("ferryDraw.pathBetween", { a, b }) : t("ferryDraw.sayPath"), true);
      return false;
    }
    const a = await edit({ op: "ferryRoute", line, ports: ports.map((p) => ({ node: p.node })), route: routeArray(r) }, true);
    if (!a.ok) say(t("ferryDraw.changeFailed"), true);
    else playBuild("pop");
    return a.ok;
  }

  // ---------------------------------------------------------------- drawing

  /** Draw the route, the ports and the tag; after each map frame and each change. */
  place() {
    if (tool.peek() !== "ferry") {
      this.svg.style.display = "none";
      return;
    }
    const xy = (p: Pt) => this.map.project([p.lng, p.lat]);
    const col = safeColour(this.colour);
    const parts: string[] = [];
    const path = (ll: number[], from: number, to: number) => {
      let d = "";
      for (let i = from; i <= to; i++) {
        const s = this.map.project([ll[2 * i], ll[2 * i + 1]]);
        d += `${i === from ? "M" : "L"}${s.x.toFixed(1)} ${s.y.toFixed(1)}`;
      }
      return d;
    };
    const stroke = (d: string, colour: string, dash = false) => {
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:var(--map-land);stroke-width:7px;stroke-opacity:.9"/>`);
      parts.push(`<path d="${d}" fill="none" stroke-linejoin="round" stroke-linecap="${dash ? "butt" : "round"}" style="stroke:${colour};stroke-width:3.5px${dash ? ";stroke-dasharray:7 4" : ""}"/>`);
    };
    const r = this.route && this.routeKey === this.berthKey() ? this.route : null;
    if (r && r.legs.length) {
      for (const leg of r.legs) stroke(path(r.ll, leg.from, leg.to), leg.reason ? "var(--bad, #d33)" : col, !leg.reason);
    } else if (r) stroke(path(r.ll, 0, (r.ll.length >> 1) - 1), col, true);
    else if (this.ports.length >= 2) {
      // searching: straight and grey, as a bus leg waits for its roads
      const ll = this.ports.flatMap((p) => [p.berth.lng, p.berth.lat]);
      parts.push(`<path d="${path(ll, 0, this.ports.length - 1)}" fill="none" style="stroke:var(--muted, #999);stroke-width:2.5px;stroke-dasharray:4 4"/>`);
    }
    // the leg the next click would add, straight
    const end = this.ports[this.endIndex()];
    const u = this.under;
    if (end && this.at && !touchInput.peek() && (u.kind === "shore" || u.kind === "port")) {
      const a = xy(end.berth), b = xy(u.berth);
      parts.push(`<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)}L${b.x.toFixed(1)} ${b.y.toFixed(1)}" fill="none" style="stroke:var(--ferry-ghost);stroke-width:2px;stroke-opacity:.85;stroke-dasharray:4 4"/>`);
    }
    // where a click would put a new port (T-2468), its landing and berth as a port's, in the ghost colour
    if (this.at && !touchInput.peek() && u.kind === "shore") {
      const a = xy(u.landing), b = xy(u.berth);
      parts.push(`<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)}L${b.x.toFixed(1)} ${b.y.toFixed(1)}" style="stroke:var(--ferry-ghost);stroke-width:2.5px"/>`);
      parts.push(`<circle cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}" r="5.5" style="fill:var(--map-land);stroke:var(--ferry-ghost);stroke-width:2.5px"/>`);
    }
    // each port: a line from its landing to its berth and a station circle; the active end larger
    const ei = this.endIndex();
    this.ports.forEach((p, i) => {
      const a = xy(p.landing), b = xy(p.berth), big = i === (u.kind === "handle" ? u.i : -1);
      parts.push(`<path d="M${a.x.toFixed(1)} ${a.y.toFixed(1)}L${b.x.toFixed(1)} ${b.y.toFixed(1)}" style="stroke:${col};stroke-width:2.5px"/>`);
      const at = `cx="${a.x.toFixed(1)}" cy="${a.y.toFixed(1)}"`;
      if (i === ei) parts.push(`<circle class="ferry-end" ${at} r="${big ? 8 : 7}" style="fill:var(--map-land);stroke:${col};stroke-width:3px"/>`);
      else parts.push(`<circle class="ferry-port" ${at} r="${big ? 7 : 5.5}" style="fill:var(--map-land);stroke:${col};stroke-width:2.5px"/>`);
    });
    this.svg.style.display = "";
    this.svg.setAttribute("width", String(this.overlay.cssW));
    this.svg.setAttribute("height", String(this.overlay.cssH));
    this.svg.innerHTML = parts.join("");
    this.placeTag(r);
  }

  private placeTag(r: LineRoute | null) {
    const tag = this.tag, at = this.at;
    const row = (text: string, bad = false) => Object.assign(document.createElement("div"), { textContent: text, className: bad ? "bad" : "" });
    if (!at || touchInput.peek()) {
      tag.style.display = "none";
      return;
    }
    const u = this.under, rows: HTMLElement[] = [];
    let bad = false;
    if (u.kind === "handle") {
      const finishes = this.ports.length >= 2 && (this.isSeam(u.i) || (!this.looped() && u.i === this.endIndex()));
      if (this.ports.length >= 2) rows.push(row(finishes ? t(this.line === null ? "ferryDraw.clickMake" : "ferryDraw.clickSave") : t("ferryDraw.clickDelete")));
      const right = { open: "ferryDraw.rightOpen", stop: "ferryDraw.rightStopLooping", delete: "ferryDraw.rightDelete", close: "ferryDraw.rightClose", loopBack: "ferryDraw.rightLoopBack" } as const;
      const a = this.rightAction(u.i);
      if (a) rows.push(row(t(right[a])));
    } else if (u.kind === "port") rows.push(row(this.ports.some((p) => p.node === u.node) ? u.name : t("ferryDraw.stopAt", { name: u.name })));
    else if (u.kind === "shore") rows.push(row(t("ferryDraw.placePort")));
    else if (u.kind === "water") (rows.push(row(onWater(), true)), (bad = true));
    else if (u.kind === "far") (rows.push(row(farFromWater(), true)), (bad = true));
    if (this.ports.length >= 2 && u.kind !== "handle") {
      if (r && r.legs.some((l) => l.reason)) (rows.push(row(noWaterPath(), true)), (bad = true));
      else if (r) {
        const p = this.price();
        rows.push(row(`${distance(r.m)}, ${moneyShort(p.cost * 1e6)}${this.built ? " more" : ""}`));
        rows.push(row(t("ferryDraw.ports", { count: this.ports.length - (this.looped() ? 1 : 0) })));
      }
    }
    if (!rows.length) {
      tag.style.display = "none";
      return;
    }
    // the whole tag is red only when all it says is a refusal; else the refusal's row is
    tag.className = "tool-tag build-tag" + (bad && rows.every((r) => r.className === "bad") ? " bad" : "");
    tag.replaceChildren(...rows);
    tag.style.display = "";
    const s = this.map.project([at.lng, at.lat]);
    tag.style.transform = `translate(${Math.round(s.x + 14)}px, ${Math.round(s.y + 12)}px)`;
  }
}

/** Two draft ports are one port: the same placed port, or the network's same port. */
function samePort(a: DraftPort, b: DraftPort): boolean {
  return a === b || (a.node !== null && a.node === b.node);
}
