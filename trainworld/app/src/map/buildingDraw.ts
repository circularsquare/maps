// Drawing a station building (T-2218, design notes/T-2208.md 3): started only from a station's upgrade
// row, with the smallest rectangle round its complex's boxes already drawn (`begin`), or from Edit outline
// with a building's own outline. A closed outline is edited by its handles: drag a corner, right click a
// corner to delete it, drag the small handle in an edge's middle to make a new corner. Escape drops a new
// outline, and clicks then place corners as the bus tool's points: a ghost edge follows the pointer, a
// click on the first corner, a double click or Enter closes it, Backspace or a right click takes back the
// last corner. Construct (Enter, or the bar's button) adds it; Save outline reshapes an existing one.
//
// The snap, a light one (Ctrl held: none): the first edge turns to the nearest platform's axis (a station
// within 50 m of its first corner) within 10 degrees; every other edge to that axis or a right angle to it
// within 10 degrees; a corner within 12 px of the first lands on it, and one within 8 px of a line along
// the axis through the first corner lands on that line. With three corners at a right angle the fourth
// shows as a ghost, and Enter takes it. A dragged corner snaps its edges the same way: both where they
// meet within 8 px of the pointer, else the one nearer its line, so it can leave a right angle.
//
// Corners are pack units (the track model's plane); the price, members and refusal come from the track
// model's quote (`buildingQuote`). The tool owns no map layer: the outline is an SVG over the map.

import type { Map as MlMap, MapMouseEvent } from "maplibre-gl";
import { meet, ontoLine, snapCorner, snapLine, SNAP_RAD } from "../game/buildings";
import { buildingById, buildingDraft } from "../game/buildingView";
import { segmentGap } from "../game/complexes";
import { groundScale, lngLatToPack, packToLngLat } from "../game/coords";
import { area, money, plural } from "../game/format";
import { describeIssue, describeIssues } from "../game/issues";
import { shadeColours } from "../game/shades";
import { selection, tool } from "../game/ui";
import { edit, say, world } from "../game/world";
import type { StationId } from "../game/types";
import { queryClock } from "../workers/clockClient";
import { setGrabbing } from "./nodeDrag";
import type { Overlay } from "./overlay";
import { touchInput, type TouchCtl } from "./touch";

type Pt = { x: number; y: number };

/** what a press at the pointer would take hold of */
type Under = { kind: "corner"; i: number } | { kind: "mid"; i: number } | { kind: "first" } | { kind: "free" };

const HANDLE_PX = 9;
const MID_PX = 7;
const CLOSE_PX = 12;
const LINE_PX = 8;
const DRAG_PX = 4;
const PLATFORM_NEAR_M = 50;
/** three corners this near a right angle offer the fourth */
const SQUARE_RAD = (1 * Math.PI) / 180;

interface Quote { key: string; m2: number; price: number; members: number; built: number; issues: string[] }

export class BuildingDraw {
  private pts: Pt[] = [];
  private closed = false;
  /** the building being reshaped, and its outline when the tool began */
  private id: number | null = null;
  private start = "";
  /** the edges' axis (radians, pack frame); null until the first edge has one */
  private axis: number | null = null;
  /** the station whose complex is selected again when the tool ends */
  private from: StationId | null = null;
  /** the pointer: CSS px, and where a click would put a corner */
  private atPx: [number, number] | null = null;
  private at: Pt | null = null;
  private under: Under = { kind: "free" };
  private noSnap = false;
  private press: { i: number; mid: boolean; x: number; y: number } | null = null;
  private drag: { i: number; was: Pt[] } | null = null;
  private dragEnded = false;
  private busy = false;
  private quote: Quote | null = null;
  private asking = false;
  private svg: SVGSVGElement;

  constructor(private map: MlMap, box: HTMLElement, private tag: HTMLElement, private overlay: Overlay, private host: { exit(): void }) {
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "building-draw");
    this.svg.style.cssText = "position:absolute;left:0;top:0;overflow:visible;display:none;pointer-events:none";
    box.insertBefore(this.svg, box.firstChild);
    map.on("contextmenu", (e) => {
      if (!this.active) return;
      e.originalEvent.preventDefault();
      const u = this.underAt(e.point.x, e.point.y);
      if (this.closed) {
        if (u.kind === "corner") this.removeCorner(u.i);
      } else this.back();
    });
    // capture: before MapLibre's own handlers, so a press on a handle does not pan the map
    map.getContainer().addEventListener("mousedown", (e) => this.onDown(e), true);
    addEventListener("mousemove", (e) => this.onDrag(e));
    addEventListener("mouseup", () => this.onUp());
  }

  get active() {
    return tool.peek() === "building";
  }

  /** Start the tool on an outline (pack x, y pairs, flat): building `id`'s to reshape, or a new one's
   * offer (`id` null). `from`: the station whose complex it was started from. */
  begin(corners: number[] | null, id: number | null, from: StationId | null) {
    this.pts = [];
    for (let i = 0; corners && i + 1 < corners.length; i += 2) this.pts.push({ x: corners[i], y: corners[i + 1] });
    this.closed = this.pts.length >= 3;
    this.id = id;
    this.start = this.flat().join();
    this.from = from;
    this.axis = this.pts.length >= 2 ? Math.atan2(this.pts[1].y - this.pts[0].y, this.pts[1].x - this.pts[0].x) : null;
    this.quote = null;
    this.at = this.atPx = null;
    this.under = { kind: "free" };
    tool.value = "building";
    this.changed();
  }

  /** the tool was left */
  clear() {
    this.endDrag();
    this.pts = [];
    this.closed = false;
    this.id = null;
    this.quote = null;
    this.at = this.atPx = null;
    this.svg.style.display = "none";
    buildingDraft.value = null;
  }

  private flat(pts = this.pts): number[] {
    return pts.flatMap((p) => [p.x, p.y]);
  }

  private changed() {
    const d = buildingDraft.peek();
    if (!d || d.id !== this.id || d.corners !== this.pts.length || d.closed !== this.closed) buildingDraft.value = { id: this.id, corners: this.pts.length, closed: this.closed };
    this.ask();
    this.place();
  }

  // ---------------------------------------------------------------- screen and pack

  private toPx(p: Pt): [number, number] {
    const q = this.map.project(packToLngLat(p.x, p.y));
    return [q.x, q.y];
  }

  private toPack(px: number, py: number): Pt {
    const ll = this.map.unproject([px, py]);
    return lngLatToPack(ll.lng, ll.lat);
  }

  /** pack units per CSS px at a screen point */
  private unitsPerPx(px: number, py: number): number {
    const a = this.toPack(px, py), b = this.toPack(px + 1, py);
    return Math.hypot(b.x - a.x, b.y - a.y) || 1;
  }

  // ---------------------------------------------------------------- what is under the pointer

  private underAt(px: number, py: number): Under {
    let best = HANDLE_PX * HANDLE_PX, hit = { kind: "free" } as Under;
    this.pts.forEach((p, i) => {
      const [x, y] = this.toPx(p);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) [best, hit] = [d, { kind: "corner", i }];
    });
    if (hit.kind === "corner") return !this.closed && hit.i === 0 && this.pts.length >= 3 ? { kind: "first" } : hit;
    if (!this.closed) {
      // the first corner closes from a little further out
      const f = this.pts[0];
      if (f && this.pts.length >= 3) {
        const [x, y] = this.toPx(f);
        if (Math.hypot(x - px, y - py) <= CLOSE_PX) return { kind: "first" };
      }
      return hit;
    }
    best = MID_PX * MID_PX;
    for (let i = 0; i < this.pts.length; i++) {
      const [x, y] = this.mid(i);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < best) [best, hit] = [d, { kind: "mid", i }];
    }
    return hit;
  }

  /** edge `i`'s middle (from corner i to the next), CSS px */
  private mid(i: number): [number, number] {
    const a = this.toPx(this.pts[i]), b = this.toPx(this.pts[(i + 1) % this.pts.length]);
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  }

  /** The axis of the platform nearest a point within PLATFORM_NEAR_M, or null. */
  private platformAxis(p: Pt): number | null {
    const k = groundScale(p.y);
    let best = PLATFORM_NEAR_M * k, axis: number | null = null;
    for (const s of world.peek()?.save.stations ?? []) {
      if (!(s.length > 0) || Math.abs(s.x - p.x) > 2000 * k || Math.abs(s.y - p.y) > 2000 * k) continue;
      const h = (s.length / 2) * k, dx = Math.cos(s.heading) * h, dy = Math.sin(s.heading) * h;
      const d = segmentGap(s.x - dx, s.y - dy, s.x + dx, s.y + dy, p.x, p.y, p.x, p.y);
      if (d < best) [best, axis] = [d, s.heading];
    }
    return axis;
  }

  /** Where a click at a screen point would put the next corner, snapped. */
  private snapNext(px: number, py: number): Pt {
    const p = this.toPack(px, py), n = this.pts.length;
    if (this.noSnap || n === 0) return p;
    const prev = this.pts[n - 1];
    if (n === 1) {
      const axis = this.platformAxis(prev);
      const a = axis === null ? null : snapLine(prev, p, axis);
      return a === null ? p : ontoLine(prev, a, p);
    }
    const axis = this.axis!;
    const first = this.pts[0];
    const a = snapLine(prev, p, axis);
    // a line along the axis through the first corner, within LINE_PX of the pointer
    const tol = LINE_PX * this.unitsPerPx(px, py);
    let b: number | null = null, bd = tol;
    for (const dir of [axis, axis + Math.PI / 2]) {
      const q = ontoLine(first, dir, p), d = Math.hypot(q.x - p.x, q.y - p.y);
      if (d <= bd) [bd, b] = [d, dir];
    }
    if (a !== null && b !== null) return meet(prev, a, first, b) ?? ontoLine(prev, a, p);
    if (a !== null) return ontoLine(prev, a, p);
    if (b !== null) return ontoLine(first, b, p);
    return p;
  }

  /** With three corners at a right angle, the fourth that closes a rectangle. */
  private ghostFourth(): Pt | null {
    if (this.closed || this.pts.length !== 3) return null;
    const [a, b, c] = this.pts;
    const u = Math.atan2(a.y - b.y, a.x - b.x), v = Math.atan2(c.y - b.y, c.x - b.x);
    const d = Math.abs(u - v) % Math.PI;
    if (Math.abs(d - Math.PI / 2) > SQUARE_RAD) return null;
    return { x: a.x + c.x - b.x, y: a.y + c.y - b.y };
  }

  // ---------------------------------------------------------------- events

  move(e: MapMouseEvent) {
    this.noSnap = !!(e.originalEvent?.ctrlKey || e.originalEvent?.metaKey);
    if (this.drag) return;
    this.atPx = [e.point.x, e.point.y];
    this.under = this.underAt(e.point.x, e.point.y);
    this.at = this.closed ? null : this.under.kind === "first" ? this.pts[0] : this.snapNext(e.point.x, e.point.y);
    const cur = this.under.kind === "free" ? "crosshair" : "pointer";
    if (this.map.getCanvas().style.cursor !== cur) this.map.getCanvas().style.cursor = cur;
    if (!this.closed && this.pts.length >= 2) this.ask();
    this.place();
  }

  click(e: MapMouseEvent) {
    if (this.busy || this.dragEnded || this.closed) return;
    if (e.originalEvent.detail >= 2) {
      // the first click of the two placed a corner already
      if (this.pts.length >= 3) this.close();
      return;
    }
    this.noSnap = e.originalEvent.ctrlKey || e.originalEvent.metaKey;
    const u = this.underAt(e.point.x, e.point.y);
    if (u.kind === "first") return this.close();
    this.addCorner(this.snapNext(e.point.x, e.point.y));
  }

  tap(e: MapMouseEvent) {
    if (this.busy || this.closed) return;
    const u = this.underAt(e.point.x, e.point.y);
    if (u.kind === "first") return this.close();
    this.addCorner(this.snapNext(e.point.x, e.point.y));
  }

  private addCorner(p: Pt) {
    this.pts.push(p);
    if (this.pts.length === 2) this.axis = Math.atan2(p.y - this.pts[0].y, p.x - this.pts[0].x);
    this.changed();
  }

  private close() {
    if (this.pts.length < 3) return;
    this.closed = true;
    this.at = null;
    this.under = { kind: "free" };
    this.changed();
  }

  private removeCorner(i: number) {
    if (this.busy || this.pts.length <= 3) return;
    this.pts.splice(i, 1);
    this.under = { kind: "free" };
    this.changed();
  }

  /** Take back the last corner; with none placed, leave the tool. */
  private back() {
    if (this.busy) return;
    if (!this.pts.length) return this.host.exit();
    this.pts.pop();
    if (this.pts.length < 2) this.axis = null;
    this.changed();
  }

  /** Keys: true if used. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "Escape") {
      if (this.cancelDrag()) return true;
      // a new outline is dropped first, to be drawn by hand; an edit goes as a whole
      if (this.id === null && this.pts.length) {
        this.pts = [];
        this.closed = false;
        this.axis = null;
        this.quote = null;
        this.changed();
      } else this.host.exit();
      return true;
    }
    if (e.key === "Backspace") {
      if (!this.closed) this.back();
      return true;
    }
    if (e.key === "Enter") {
      if (this.closed) void this.finish();
      else {
        const g = this.ghostFourth();
        if (g) this.pts.push(g);
        this.close();
      }
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- dragging handles

  private onDown(e: MouseEvent) {
    if (!this.active || !this.closed || this.busy || e.button !== 0 || e.target !== this.map.getCanvas() || touchInput.peek()) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const u = this.underAt(px, py);
    if (u.kind !== "corner" && u.kind !== "mid") return;
    this.map.dragPan.disable();
    this.press = { i: u.i, mid: u.kind === "mid", x: px, y: py };
  }

  private onDrag(e: MouseEvent) {
    const p = this.press;
    if (!p) return;
    const r = this.map.getContainer().getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    this.noSnap = e.ctrlKey || e.metaKey;
    if (!this.drag) {
      if (Math.hypot(px - p.x, py - p.y) < DRAG_PX) return;
      const was = this.pts.map((q) => ({ ...q }));
      let i = p.i;
      if (p.mid) {
        const a = this.pts[i], b = this.pts[(i + 1) % this.pts.length];
        this.pts.splice(i + 1, 0, { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
        i++;
      }
      this.drag = { i, was };
      setGrabbing(true);
    }
    const n = this.pts.length, i = this.drag.i;
    const q = this.toPack(px, py);
    this.pts[i] = this.noSnap || this.axis === null ? q : snapCorner(q, this.pts[(i + n - 1) % n], this.pts[(i + 1) % n], this.axis, SNAP_RAD, LINE_PX * this.unitsPerPx(px, py));
    this.atPx = [px, py];
    this.changed();
  }

  private onUp() {
    const was = this.drag;
    this.endDrag();
    if (!was) return;
    // MapLibre may still call a press that came back to its start a click
    this.dragEnded = true;
    setTimeout(() => (this.dragEnded = false));
    this.changed();
  }

  private cancelDrag(): boolean {
    const d = this.drag;
    if (!d && !this.press) return false;
    if (d) this.pts = d.was;
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

  // ---------------------------------------------------------------- touch

  confirm() {
    if (this.closed) void this.finish();
    else this.close();
  }

  touchBack() {
    this.back();
  }

  touchState(): TouchCtl | null {
    if (!touchInput.peek() || !this.pts.length) return null;
    return { confirm: this.closed ? !this.busy : this.pts.length >= 3 ? true : null, confirmTip: this.closed ? (this.id === null ? "Construct" : "Save outline") : "Close the outline", back: !this.closed, stopTip: "Cancel" };
  }

  touchPoint() {
    const p = this.pts[this.pts.length - 1];
    if (!p) return null;
    const [lng, lat] = packToLngLat(p.x, p.y);
    return { lng, lat };
  }

  // ---------------------------------------------------------------- the quote and finishing

  /** The outline the quote is for: closed, or through the pointer while drawing. */
  private quoted(): number[] | null {
    if (this.closed) return this.flat();
    if (this.pts.length >= 2 && this.at && this.under.kind !== "first") return this.flat([...this.pts, this.at]);
    if (this.pts.length >= 3) return this.flat();
    return null;
  }

  /** Ask the track model for the outline's price, members and refusal; one at a time, the newest wins. */
  private ask() {
    if (this.asking) return;
    const c = this.quoted();
    if (!c) {
      this.quote = null;
      return;
    }
    const key = c.join();
    if (this.quote?.key === key) return;
    this.asking = true;
    void queryClock({ what: "buildingQuote", id: this.id, corners: c }).then((d) => {
      this.asking = false;
      if (!this.active) return;
      if (d && d.length >= 4) {
        const kinds = String.fromCharCode(...Array.from(d.subarray(4))).split("\n").filter(Boolean);
        this.quote = { key, m2: d[0], price: d[1], members: d[2], built: d[3], issues: kinds };
      }
      const now = this.quoted();
      if (now && now.join() !== key) this.ask();
      this.place();
    });
  }

  /** Construct the outline, or save the edited one; refused with the reason, the tool staying on. */
  async finish() {
    if (this.busy || !this.active) return;
    if (!this.closed) return void say("Close the outline first.", true);
    const corners = this.flat();
    if (this.id !== null && corners.join() === this.start) return this.host.exit();
    this.busy = true;
    try {
      const id = this.id;
      const a = await edit(id === null ? { op: "addBuilding", corners, name: "" } : { op: "reshapeBuilding", id, corners });
      if (!this.active) return;
      if (!a.ok) {
        say(describeIssues(a.issues, "construction"), true);
        return;
      }
      const name = buildingById(id ?? a.node ?? -1)?.name ?? "";
      if (id === null) say(`Station building built at ${name} for ${money(a.charge * 1e6)}.`);
      else say(a.charge >= 0.0005 ? `${name} station building changed for ${money(a.charge * 1e6)}.` : `${name} station building changed.`);
      const from = this.from;
      this.host.exit();
      if (from !== null) selection.value = { kind: "station", station: from };
    } finally {
      this.busy = false;
    }
  }

  // ---------------------------------------------------------------- drawing

  /** Draw the outline, its handles and the tag; after each map frame and each change. */
  place() {
    if (!this.active) {
      this.svg.style.display = "none";
      return;
    }
    const ink = shadeColours().stationInk;
    const xy = this.pts.map((p) => this.toPx(p));
    const f = (v: number) => v.toFixed(1);
    const parts: string[] = [];
    const d = xy.map(([x, y], i) => `${i ? "L" : "M"}${f(x)} ${f(y)}`).join("") + (this.closed ? "Z" : "");
    const bad = !!this.quote && this.quote.key === this.quoted()?.join() && this.quote.issues.length > 0;
    const edge = bad ? "var(--bad, #d33)" : ink;
    if (this.closed && xy.length >= 3) parts.push(`<path d="${d}" style="fill:#f9f7f3;fill-opacity:.3;stroke:${edge};stroke-width:2px;stroke-linejoin:round"/>`);
    else if (xy.length) {
      // the outline so far, shaded closed through the pointer
      const at = this.at && this.toPx(this.at);
      if (at && xy.length >= 2) parts.push(`<path d="${d}L${f(at[0])} ${f(at[1])}Z" style="fill:#f9f7f3;fill-opacity:.25;stroke:none"/>`);
      if (xy.length >= 2) parts.push(`<path d="${d}" style="fill:none;stroke:${edge};stroke-width:2px;stroke-linejoin:round"/>`);
      if (at && !touchInput.peek()) {
        const [lx, ly] = xy[xy.length - 1];
        parts.push(`<path d="M${f(lx)} ${f(ly)}L${f(at[0])} ${f(at[1])}" style="fill:none;stroke:${edge};stroke-width:2px;stroke-dasharray:5 4"/>`);
      }
      const g = this.ghostFourth();
      if (g) {
        const [gx, gy] = this.toPx(g);
        parts.push(`<path d="M${f(xy[2][0])} ${f(xy[2][1])}L${f(gx)} ${f(gy)}L${f(xy[0][0])} ${f(xy[0][1])}" style="fill:none;stroke:${ink};stroke-width:1.5px;stroke-opacity:.6;stroke-dasharray:3 4"/>`);
        parts.push(`<circle cx="${f(gx)}" cy="${f(gy)}" r="5" style="fill:none;stroke:${ink};stroke-width:1.5px;stroke-opacity:.7"/>`);
      }
    }
    // the edges' middles, then the corners
    const u = this.under, held = this.drag?.i ?? -1;
    if (this.closed && !this.drag) {
      for (let i = 0; i < xy.length; i++) {
        const [mx, my] = this.mid(i), hot = u.kind === "mid" && u.i === i;
        parts.push(`<circle cx="${f(mx)}" cy="${f(my)}" r="${hot ? 5 : 3.5}" style="fill:${ink};fill-opacity:.55;stroke:#fff;stroke-width:1.5px"/>`);
      }
    }
    xy.forEach(([x, y], i) => {
      const hot = (u.kind === "corner" && u.i === i) || (u.kind === "first" && i === 0) || i === held;
      parts.push(`<circle cx="${f(x)}" cy="${f(y)}" r="${hot ? 7 : 5.5}" style="fill:#fff;stroke:${ink};stroke-width:2.5px"/>`);
    });
    this.svg.style.display = "";
    this.svg.setAttribute("width", String(this.overlay.cssW));
    this.svg.setAttribute("height", String(this.overlay.cssH));
    this.svg.innerHTML = parts.join("");
    this.placeTag();
  }

  private placeTag() {
    const tag = this.tag, at = this.atPx;
    if (!at || touchInput.peek()) {
      tag.style.display = "none";
      return;
    }
    const row = (text: string, bad = false) => Object.assign(document.createElement("div"), { textContent: text, className: bad ? "bad" : "" });
    const rows: HTMLElement[] = [];
    const u = this.under;
    if (u.kind === "corner" && this.closed) rows.push(row(this.pts.length > 3 ? "Drag to move, right click to delete" : "Drag to move"));
    else if (u.kind === "mid") rows.push(row("Drag to add a corner"));
    else if (u.kind === "first") rows.push(row("Click to close the outline"));
    const q = this.quote, c = this.quoted();
    if (q && c && q.key === c.join() && !this.drag) {
      if (q.issues.length) rows.push(...q.issues.map((k) => row(describeIssue(k).replace(/\.$/, ""), true)));
      else {
        rows.push(row(area(q.m2)));
        rows.push(row(this.id === null ? money(q.price * 1e6) : `${money(q.price * 1e6)} for the added floor`));
        rows.push(row(plural(q.members, "station") + " inside"));
      }
    } else if (q && this.drag && q.key === c?.join()) rows.push(row(area(q.m2)));
    if (!rows.length) {
      tag.style.display = "none";
      return;
    }
    tag.className = "tool-tag build-tag" + (rows.every((r) => r.className === "bad") ? " bad" : "");
    tag.replaceChildren(...rows);
    tag.style.display = "";
    tag.style.transform = `translate(${Math.round(at[0] + 14)}px, ${Math.round(at[1] + 12)}px)`;
  }
}
