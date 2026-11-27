// T-3740 stage Z2: the Fare zones tool on the map. A click on a station gives it (and its complex) the tool's fare
// zone; a left drag draws a loop, and on release every station inside it takes the zone, one edit for the lot. The
// right or middle button drags the map, as in the Select tool (blueprintBox.ts). On touch a tap paints, one finger
// draws the loop and two fingers pan and zoom as always (MapLibre's touch pan is held to two fingers while the tool
// is on). While the tool is on, the ground around each painted station or complex is tinted in its zone's colour
// (game/palette.ts FARE_ZONE_COLOURS): its cell (map/fareZoneCells.ts), the ground nearer it than any other painted
// one within REACH_M, filled faintly, with a stronger band inside each zone's edge; and each painted station or
// complex has its zone number beside it, clear of the station names. The tint and numbers are on
// two 2D canvases of their own: the tint under the network, the numbers and the loop over it, redrawn on MapLibre's
// frames and when the stations' zones change. Hit tests use every station projected, at any zoom, so a loop drawn
// far out still takes a whole suburb.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { complexes } from "../game/complexView";
import { fareAreas, liveCode, paintChanges, paintRings, paintStations, paintZone, ringPlan, ringsArmed, ringWidth, type RingPlan } from "../game/fareZonePaint";
import { zoneOfCode } from "../game/fareZones";
import { fareZoneColour, inkOn } from "../game/palette";
import { shadeColours } from "../game/shades";
import { mapShade, tool } from "../game/ui";
import { world } from "../game/world";
import { buildCells, type Cell, type Site } from "./fareZoneCells";
import { fitSparse } from "./gl";
import { EARTH_C, kAtLat, toLocal, toMerc01 } from "./geo";
import { BoxGrid } from "./labelLayout";
import type { Overlay } from "./overlay";
import { STATIONS_ZOOM } from "./stationGeometry";
import { stationOff } from "./visibility";

/** ground reach of a station's tint, m */
const REACH_M = 650;
/** the tint's least radius on screen, px, so it still shows zoomed far out; the zoom it is worked out at goes in
 * quarter steps, so the cells are rebuilt only a few times while zooming out past it */
const MIN_R_PX = 7;
const MIN_R_STEP = 4;
/** the fill's strength, and the band's inside each zone's edge (over the fill, together 80%) and its width, CSS px */
const CELL_FILL = 0.22;
const CELL_EDGE = (0.8 - CELL_FILL) / (1 - CELL_FILL);
const RIM_PX = 2.5;
/** the disc's arcs drawn as chords no more than this far off the circle, CSS px */
const SAGITTA_PX = 0.3;
/** a press that moves less than this is a click (px) */
const CLICK_PX = 5;
/** a click this near a station picks it (px) */
const HIT_PX = 14;
/** the zone numbers: radius, and where they may sit from the station, first choice first (px; names go right of or
 * below a station) */
const BADGE_R = 8;
const BADGE_SPOTS: [number, number][] = [[-11, -11], [-11, 11], [-14, 0], [11, -11], [0, -14], [0, 14], [14, 0], [11, 11]];

/** a painted station or complex on screen, CSS px */
interface Painted { x: number; y: number; zone: number }

export class FareZoneTool {
  private tintCanvas: HTMLCanvasElement;
  private marks: HTMLCanvasElement;
  /** the cells and what they were built from (the stations, areas, complexes and the reach floor's zoom) */
  private cells: Cell[] = [];
  private cellsFrom: unknown[] = [];
  /** the zone numbers drawn once each, at the canvas's px ratio */
  private badgeArt = new Map<number, HTMLCanvasElement>();
  private badgeKey = "";
  /** the station names shown, boxes in CSS px (map/stationLabels.ts), which the numbers keep clear of */
  nameBoxes: () => number[][] = () => [];
  /** the last frame's numbers: painted sites on screen, and the boxes of the numbers drawn (CSS px), for checks */
  lastBadges: { sites: number; boxes: number[][] } = { sites: 0, boxes: [] };
  private dirty = false;
  /** the loop being drawn, its points as lng/lat so it stays on the ground when the camera moves, or a press that
   * may still be a click (`start` and `last` are CSS px of the press and of the last point added) */
  private loop: { ll: [number, number][]; start: [number, number]; last: [number, number]; moved: boolean; id: number } | null = null;
  /** a right or middle drag panning the map */
  private pan: { x: number; y: number } | null = null;
  private fingers = new Set<number>();
  /** stations inside the loop as it stands, drawn with a ring */
  private inside: [number, number][] = [];
  private frame = 0;
  private touchPanRule: ((n: number) => boolean) | null = null;
  /** "Set zones by distance": a left press, which pans the map when dragged and picks the centre when not */
  private ringPress: { x: number; y: number; id: number } | null = null;
  /** the station under the pointer while it is armed, and the rings it would give, drawn as a preview */
  private ringHover: number | null = null;
  private ringPreview: { key: string; plan: RingPlan | null } | null = null;

  constructor(private map: MlMap, private overlay: Overlay) {
    this.tintCanvas = document.createElement("canvas");
    this.tintCanvas.className = "overlay fare-tint";
    overlay.canvas.before(this.tintCanvas);
    this.marks = document.createElement("canvas");
    this.marks.className = "overlay fare-marks";
    overlay.canvas.after(this.marks);
    fitSparse(this.tintCanvas, map, false);
    fitSparse(this.marks, map, false);
    overlay.afterMapFrame.push(() => this.draw());
    map.on("resize", () => this.draw());
    effect(() => {
      world.value;
      fareAreas();
      complexes.value;
      paintZone.value;
      mapShade.value;
      ringWidth.value;
      if (!ringsArmed.value) this.ringHover = null;
      const on = tool.value === "fares";
      this.holdTouchPan(on);
      if (!on) { this.cancel(); if (ringsArmed.peek()) ringsArmed.value = false; }
      this.invalidate();
    });
    const host = map.getCanvasContainer();
    host.addEventListener("pointerdown", (e) => this.onDown(e));
    host.addEventListener("contextmenu", (e) => { if (this.active) e.preventDefault(); });
    addEventListener("pointermove", (e) => this.onMove(e));
    addEventListener("pointerup", (e) => this.onUp(e));
    addEventListener("pointercancel", (e) => this.onCancel(e));
  }

  get active() {
    return tool.peek() === "fares";
  }

  /** MapLibre's touch pan needs two fingers while the tool is on, so one finger draws (its cooperative gestures'
   * own rule, put back when the tool goes). */
  private holdTouchPan(on: boolean) {
    const tp = (this.map.dragPan as unknown as { _touchPan?: { _shouldBePrevented: (n: number) => boolean } })._touchPan;
    if (!tp) return;
    if (on && !this.touchPanRule) {
      this.touchPanRule = tp._shouldBePrevented;
      tp._shouldBePrevented = (n: number) => n < 2;
    } else if (!on && this.touchPanRule) {
      tp._shouldBePrevented = this.touchPanRule;
      this.touchPanRule = null;
    }
  }

  /** Esc while drawing a loop: it goes, nothing changes; else "Set zones by distance" is let go. True if either. */
  cancel(): boolean {
    const had = !!this.loop || !!this.pan;
    this.loop = null;
    this.pan = null;
    this.ringPress = null;
    this.inside = [];
    const armed = !had && ringsArmed.peek();
    if (armed) ringsArmed.value = false;
    if (had || armed) this.invalidate();
    return had || armed;
  }

  private point(e: PointerEvent): [number, number] {
    const r = this.map.getCanvasContainer().getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // ---------------------------------------------------------------- the stations on screen

  /** Every station (blueprint ones too) at CSS px: node id, x, y. */
  private stationsOnScreen(): { num: number; x: number; y: number }[] {
    const w = world.peek();
    const out: { num: number; x: number; y: number }[] = [];
    if (!w) return out;
    const at: [number, number] = [NaN, NaN];
    for (const s of w.save.stations) {
      if (stationOff(s.id, s.built)) continue;
      this.overlay.toScreenInto(...toLocal(s.lng, s.lat), at);
      if (Number.isNaN(at[0])) continue;
      out.push({ num: s.num, x: at[0], y: at[1] });
    }
    return out;
  }

  /** The station nearest a point within HIT_PX, or null. */
  private stationAt(x: number, y: number): number | null {
    let best = HIT_PX * HIT_PX, hit: number | null = null;
    for (const s of this.stationsOnScreen()) {
      const d = (s.x - x) ** 2 + (s.y - y) ** 2;
      if (d < best) [best, hit] = [d, s.num];
    }
    return hit;
  }

  /** The loop's points at CSS px with the camera as it is now; NaN points (behind the globe) dropped. */
  private loopPx(lp: { ll: [number, number][] }): [number, number][] {
    const out: [number, number][] = [], at: [number, number] = [NaN, NaN];
    for (const [lng, lat] of lp.ll) {
      this.overlay.toScreenInto(...toLocal(lng, lat), at);
      if (!Number.isNaN(at[0])) out.push([at[0], at[1]]);
    }
    return out;
  }

  /** The stations inside the loop `pts` (CSS px, closed back to its start). */
  private within(pts: [number, number][]): { num: number; x: number; y: number }[] {
    let l = Infinity, t = Infinity, r = -Infinity, b = -Infinity;
    for (const [x, y] of pts) [l, t, r, b] = [Math.min(l, x), Math.min(t, y), Math.max(r, x), Math.max(b, y)];
    return this.stationsOnScreen().filter((s) => s.x >= l && s.x <= r && s.y >= t && s.y <= b && insidePolygon(pts, s.x, s.y));
  }

  // ---------------------------------------------------------------- pointer

  private onDown(e: PointerEvent) {
    if (!this.active) return;
    if (e.pointerType === "touch") {
      this.fingers.add(e.pointerId);
      // a second finger is a pan or pinch: the loop goes
      if (this.fingers.size > 1) return void this.cancel();
    }
    if (this.loop || this.pan) return;
    const [x, y] = this.point(e);
    if (e.pointerType !== "touch" && (e.button === 1 || e.button === 2)) {
      e.preventDefault();
      this.pan = { x, y };
      return;
    }
    if (e.button !== 0) return;
    if (e.pointerType !== "touch") e.preventDefault();
    if (ringsArmed.peek()) {
      // picking the centre: a drag pans, a click picks
      this.pan = { x, y };
      this.ringPress = { x, y, id: e.pointerId };
      return;
    }
    const g = this.map.unproject([x, y]);
    this.loop = { ll: [[g.lng, g.lat]], start: [x, y], last: [x, y], moved: false, id: e.pointerId };
  }

  private onMove(e: PointerEvent) {
    if (!this.active) return;
    const [x, y] = this.point(e);
    if (this.pan) {
      this.map.panBy([this.pan.x - x, this.pan.y - y], { animate: false });
      this.pan = { x, y };
      return;
    }
    const lp = this.loop;
    if (!lp) {
      if (e.pointerType !== "touch" && e.buttons === 0 && e.target === this.map.getCanvas()) {
        const hit = this.map.getZoom() >= STATIONS_ZOOM ? this.stationAt(x, y) : null;
        const c = hit !== null ? "pointer" : "crosshair";
        if (this.map.getCanvas().style.cursor !== c) this.map.getCanvas().style.cursor = c;
        // the rings a click there would give
        if (ringsArmed.peek() && hit !== this.ringHover) {
          this.ringHover = hit;
          this.draw(true);
        }
      }
      return;
    }
    if (e.pointerId !== lp.id) return;
    const [x0, y0] = lp.start, [lx, ly] = lp.last;
    if (!lp.moved && Math.hypot(x - x0, y - y0) < CLICK_PX) return;
    lp.moved = true;
    if (Math.hypot(x - lx, y - ly) >= 3) {
      const g = this.map.unproject([x, y]);
      lp.ll.push([g.lng, g.lat]);
      lp.last = [x, y];
    }
    // the stations it would take follow the loop, at most once a frame
    if (!this.frame) this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.draw(true);
    });
  }

  private onUp(e: PointerEvent) {
    if (e.pointerType === "touch") this.fingers.delete(e.pointerId);
    const rp = this.ringPress;
    this.ringPress = null;
    if (this.pan) {
      this.pan = null;
      if (!rp || rp.id !== e.pointerId || !this.active || !ringsArmed.peek()) return;
      const [x, y] = this.point(e);
      if (Math.hypot(x - rp.x, y - rp.y) >= CLICK_PX || this.map.getZoom() < STATIONS_ZOOM) return;
      const hit = this.stationAt(x, y);
      if (hit !== null) {
        this.ringHover = null;
        void paintRings(hit);
      }
      return;
    }
    const lp = this.loop;
    if (!lp || e.pointerId !== lp.id) return;
    this.loop = null;
    this.inside = [];
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.draw();
    if (!this.active) return;
    if (lp.moved) {
      const pts = this.loopPx(lp);
      if (pts.length < 3) return;
      const nums = this.within(pts).map((s) => s.num);
      if (nums.length) void paintStations(nums);
      return;
    }
    // a click only where stations are drawn, so it never paints one the player can't see (a loop works at any zoom)
    if (this.map.getZoom() < STATIONS_ZOOM) return;
    const [x, y] = this.point(e);
    const hit = this.stationAt(x, y);
    if (hit !== null) void paintStations([hit]);
  }

  private onCancel(e: PointerEvent) {
    if (e.pointerType === "touch") this.fingers.delete(e.pointerId);
    if (this.loop?.id === e.pointerId) this.cancel();
  }

  // ---------------------------------------------------------------- drawing

  invalidate() {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => this.dirty && this.draw());
  }

  /** The painted sites: each complex (or station on its own) at its stations' middle, in the zone of its first
   * station with one (save order), as the station panel and demand take a complex's zone. Mercator 0..1, reach at
   * least `floor`. */
  private sites(floor: number): Site[] {
    const w = world.peek(), areas = fareAreas();
    if (!w || !areas.length) return [];
    const cx = complexes.peek(), groups = new Map<string, { x: number; y: number; lat: number; n: number; code: number }>();
    for (const s of w.save.stations) {
      const code = liveCode(s.fare, areas), c = cx.of.get(s.id), key = c === undefined ? "s" + s.id : String(c);
      const [x, y] = toMerc01(s.lng, s.lat), gr = groups.get(key);
      if (!gr) groups.set(key, { x, y, lat: s.lat, n: 1, code });
      else { gr.x += x; gr.y += y; gr.lat += s.lat; gr.n++; gr.code ||= code; }
    }
    const out: Site[] = [];
    for (const gr of groups.values()) {
      if (!gr.code) continue;
      const reach = REACH_M / (EARTH_C * Math.cos(((gr.lat / gr.n) * Math.PI) / 180));
      out.push({ x: gr.x / gr.n, y: gr.y / gr.n, r: Math.max(reach, floor), code: gr.code });
    }
    return out;
  }

  /** The cells, rebuilt when the paint, the areas, the complexes or (zoomed out past REACH_M's MIN_R_PX) the
   * floor's zoom step change. */
  private cellsNow(): Cell[] {
    const w = world.peek(), areas = fareAreas(), cx = complexes.peek();
    if (!w || !areas.length) return [];
    const zq = Math.floor(this.map.getZoom() * MIN_R_STEP) / MIN_R_STEP;
    // the floor binds only far out: from where REACH_M at the equator (the least reach in Mercator) is over it, none
    const floor = MIN_R_PX / (512 * 2 ** zq), binds = REACH_M / EARTH_C < floor;
    const from = [w.save.stations, areas, cx, binds ? zq : -1];
    if (from.length !== this.cellsFrom.length || from.some((v, i) => v !== this.cellsFrom[i])) {
      this.cellsFrom = from;
      this.cells = buildCells(this.sites(binds ? floor : 0));
    }
    return this.cells;
  }

  /** The tint and the marks; `marksOnly` (a loop being drawn, the camera still) leaves the tint as it is. */
  draw(marksOnly = false) {
    if (!marksOnly) this.dirty = false;
    const on = this.active && !!this.overlay.cam;
    const cells = on ? this.cellsNow() : [];
    if (!marksOnly) fitSparse(this.tintCanvas, this.map, cells.length > 0);
    const rings = on && ringsArmed.peek() ? this.ringsNow() : null;
    fitSparse(this.marks, this.map, on && (cells.length > 0 || !!this.loop || !!rings));
    if (!on) return;
    const W = this.overlay.cssW, H = this.overlay.cssH;
    // device px per CSS px, as fitSparse sizes both canvases
    const k = (this.map.getCanvas().width || W) / Math.max(1, W);
    const shown: Painted[] = [];
    const g = !marksOnly && cells.length ? this.tintCanvas.getContext("2d") : null;
    if (g) {
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, this.tintCanvas.width, this.tintCanvas.height);
      g.setTransform(k, 0, 0, k, 0, 0);
    }
    this.tintCells(g, cells, W, H, shown);
    const m = this.marks.getContext("2d");
    if (!m || !this.marks.width) return;
    m.setTransform(1, 0, 0, 1, 0, 0);
    m.clearRect(0, 0, this.marks.width, this.marks.height);
    m.setTransform(k, 0, 0, k, 0, 0);
    if (rings) this.drawRings(m, rings);
    this.badges(m, shown, k);
    this.drawLoop(m);
  }

  /** The rings the hovered station would give, worked out again only when it, the width, the stations or the areas
   * change. */
  private ringsNow(): RingPlan | null {
    const c = this.ringHover;
    if (c === null) return null;
    const key = `${c}|${ringWidth.peek()}`, from = this.cellsFrom;
    if (this.ringPreview?.key !== key || this.ringFrom !== from) {
      this.ringPreview = { key, plan: ringPlan(c) };
      this.ringFrom = from;
    }
    return this.ringPreview.plan;
  }
  private ringFrom: unknown[] = [];

  /** Each ring of `plan` out to the highest zone it gives, filled faintly in its zone's colour, with a dashed line
   * where one zone meets the next, under the zone numbers. */
  private drawRings(m: CanvasRenderingContext2D, plan: RingPlan) {
    const [cx, cy] = toLocal(plan.at[0], plan.at[1]), k = kAtLat(plan.at[1]), at: [number, number] = [NaN, NaN];
    const ink = shadeColours().stationInk;
    const circle = (rM: number, path: Path2D, back: boolean) => {
      const r = rM * k, n = 120;
      for (let i = 0; i <= n; i++) {
        const a = ((back ? n - i : i) / n) * 2 * Math.PI;
        this.overlay.toScreenInto(cx + r * Math.cos(a), cy + r * Math.sin(a), at);
        if (i === 0) path.moveTo(at[0], at[1]); else path.lineTo(at[0], at[1]);
      }
      path.closePath();
    };
    for (let z = 1; z <= plan.top; z++) {
      const ring = new Path2D();
      circle(z * plan.widthM, ring, false);
      if (z > 1) circle((z - 1) * plan.widthM, ring, true);
      m.globalAlpha = 0.16;
      m.fillStyle = fareZoneColour(z);
      m.fill(ring, "evenodd");
    }
    m.globalAlpha = 0.85;
    m.lineWidth = 1.5;
    m.strokeStyle = ink;
    m.setLineDash([6, 4]);
    for (let z = 1; z <= plan.top; z++) {
      const edge = new Path2D();
      circle(z * plan.widthM, edge, false);
      m.stroke(edge);
    }
    m.setLineDash([]);
    m.globalAlpha = 1;
    // each ring's number, straight north of the centre midway across it
    const px = this.map.getCanvas().width / Math.max(1, this.overlay.cssW), d = 2 * BADGE_R + 2;
    for (let z = 1; z <= plan.top; z++) {
      this.overlay.toScreenInto(cx, cy - (z - 0.5) * plan.widthM * k, at);
      if (Number.isNaN(at[0])) continue;
      let art = this.badgeArt.get(z);
      if (!art) this.badgeArt.set(z, (art = badgeArt(z, ink, px, d)));
      m.drawImage(art, at[0] - d / 2, at[1] - d / 2, d, d);
    }
  }

  /** The cells on screen projected: each code's cells one path, filled at CELL_FILL; then, clipped to that path, its
   * rims (sides facing another code, and the disc's arcs) stroked twice RIM_PX wide, so the band is inside the zone
   * and two zones side by side each keep their own. Fills `shown` with the sites on screen (for the numbers); with
   * no context (`g` null) only that. */
  private tintCells(g: CanvasRenderingContext2D | null, cells: Cell[], W: number, H: number, shown: Painted[]) {
    const fills = new Map<number, Path2D>(), rims = new Map<number, Path2D>();
    const at: [number, number] = [NaN, NaN], pts: number[] = [], rim: boolean[] = [];
    for (const c of cells) {
      this.overlay.mercToScreen(c.x, c.y, at);
      const sx = at[0], sy = at[1];
      this.overlay.mercToScreen(c.x + c.r, c.y, at);
      const rp = Math.hypot(at[0] - sx, at[1] - sy);
      if (!Number.isFinite(rp) || sx < -rp || sy < -rp || sx > W + rp || sy > H + rp) continue;
      if (sx >= 0 && sy >= 0 && sx <= W && sy <= H) shown.push({ x: sx, y: sy, zone: zoneOfCode(c.code) });
      if (!g) continue;
      // the outline's points, each flagged when the stretch ending at it is a rim
      const step = 2 * Math.acos(Math.max(-1, 1 - SAGITTA_PX / Math.max(rp, SAGITTA_PX))) || Math.PI / 6;
      pts.length = rim.length = 0;
      const put = (x: number, y: number, r: boolean) => {
        this.overlay.mercToScreen(c.x + x, c.y + y, at);
        pts.push(at[0], at[1]);
        rim.push(r);
      };
      const p0 = c.pieces[0];
      if (p0.arc) put(c.r * Math.cos(p0.a0), c.r * Math.sin(p0.a0), false);
      else put(p0.x0, p0.y0, false);
      for (const p of c.pieces) {
        if (!p.arc) { put(p.x1, p.y1, p.rim); continue; }
        const n = Math.max(1, Math.ceil((p.a1 - p.a0) / Math.min(step, Math.PI / 6)));
        for (let i = 1; i <= n; i++) { const a = p.a0 + ((p.a1 - p.a0) * i) / n; put(c.r * Math.cos(a), c.r * Math.sin(a), true); }
      }
      if (pts.some(Number.isNaN)) continue;
      let f = fills.get(c.code), e = rims.get(c.code);
      if (!f) fills.set(c.code, (f = new Path2D()));
      if (!e) rims.set(c.code, (e = new Path2D()));
      f.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) f.lineTo(pts[i], pts[i + 1]);
      f.closePath();
      for (let i = 1; i < rim.length; i++) {
        if (!rim[i]) continue;
        if (i === 1 || !rim[i - 1]) e.moveTo(pts[2 * i - 2], pts[2 * i - 1]);
        e.lineTo(pts[2 * i], pts[2 * i + 1]);
      }
    }
    if (!g) return;
    g.lineWidth = 2 * RIM_PX;
    g.lineJoin = g.lineCap = "round";
    for (const [code, f] of fills) {
      const col = fareZoneColour(zoneOfCode(code));
      g.globalAlpha = CELL_FILL;
      g.fillStyle = col;
      g.fill(f);
      g.save();
      g.clip(f);
      g.globalAlpha = CELL_EDGE;
      g.strokeStyle = col;
      g.stroke(rims.get(code)!);
      g.restore();
    }
    g.globalAlpha = 1;
  }

  /** The zone numbers: a disc in the zone's colour beside each painted station or complex, lowest zone first, at
   * the first of BADGE_SPOTS clear of the station names and the numbers already drawn; one with no clear spot is
   * left out (only where stations crowd, zoomed out). */
  private badges(m: CanvasRenderingContext2D, shown: Painted[], k: number) {
    const ink = shadeColours().stationInk, artKey = `${k}|${ink}|${document.fonts.status}`;
    if (this.badgeKey !== artKey) { this.badgeArt.clear(); this.badgeKey = artKey; }
    shown.sort((a, b) => a.zone - b.zone);
    const placed = new BoxGrid(), d = 2 * BADGE_R + 2, boxes: number[][] = [];
    for (const n of this.nameBoxes()) placed.add(n);
    for (const p of shown) {
      let x = NaN, y = NaN, box: number[] = [];
      for (const [ox, oy] of BADGE_SPOTS) {
        [x, y] = [p.x + ox, p.y + oy];
        box = [x - BADGE_R, y - BADGE_R, x + BADGE_R, y + BADGE_R];
        if (!placed.hits(box)) break;
        x = NaN;
      }
      if (Number.isNaN(x)) continue;
      placed.add(box);
      boxes.push(box);
      let art = this.badgeArt.get(p.zone);
      if (!art) this.badgeArt.set(p.zone, (art = badgeArt(p.zone, ink, k, d)));
      m.drawImage(art, x - d / 2, y - d / 2, d, d);
    }
    this.lastBadges = { sites: shown.length, boxes };
  }

  /** The loop being drawn, shaded inside, and a ring on each station it would take in the tool's colour. */
  private drawLoop(m: CanvasRenderingContext2D) {
    const lp = this.loop;
    if (!lp || !lp.moved) return;
    const pts = this.loopPx(lp);
    if (pts.length < 2) return;
    const hit = pts.length >= 3 ? this.within(pts) : [], take = paintChanges(hit.map((s) => s.num));
    this.inside = hit.filter((s) => take.has(s.num)).map((s) => [s.x, s.y]);
    const z = paintZone.peek(), col = z > 0 ? fareZoneColour(z) : "#8a8a8a", ink = shadeColours().stationInk;
    m.beginPath();
    pts.forEach(([x, y], i) => (i ? m.lineTo(x, y) : m.moveTo(x, y)));
    m.closePath();
    m.globalAlpha = 0.18;
    m.fillStyle = col;
    m.fill();
    m.globalAlpha = 1;
    m.lineWidth = 2;
    m.strokeStyle = ink;
    m.setLineDash([6, 4]);
    m.stroke();
    m.setLineDash([]);
    m.lineWidth = 3;
    m.strokeStyle = col;
    for (const [x, y] of this.inside) {
      m.beginPath();
      m.arc(x, y, 9, 0, 2 * Math.PI);
      m.stroke();
    }
  }
}

/** Even-odd: whether (x, y) is inside the polygon `pts`. */
function insidePolygon(pts: [number, number][], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** A zone's number on a disc of its colour, ringed in the station ink, `d` CSS px square at `k` device px each. */
function badgeArt(zone: number, ink: string, k: number, d: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = Math.ceil(d * k);
  const m = c.getContext("2d")!, col = fareZoneColour(zone);
  m.setTransform(c.width / d, 0, 0, c.width / d, 0, 0);
  m.beginPath();
  m.arc(d / 2, d / 2, BADGE_R, 0, 2 * Math.PI);
  m.fillStyle = col;
  m.fill();
  m.lineWidth = 1.5;
  m.strokeStyle = ink;
  m.stroke();
  m.font = "700 11px Nunito, sans-serif";
  m.textAlign = "center";
  m.textBaseline = "middle";
  m.fillStyle = inkOn(col);
  m.fillText(String(zone), d / 2, d / 2 + 0.5);
  return c;
}
