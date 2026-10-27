// Station ridership circles (T-482, T-1448): one circle per station complex, its area the riders a
// day the station ridership choice counts, drawn on a 2D canvas of its own over the demand bubbles'
// canvas and under the station names. Entries are blue and transfers green; a circle is one colour
// between them by its transfers' share, so Total's are shaded and Origins' and Transfers' are the ends
// (DECISIONS 2026-10-26, T-1531).
// Redrawn with each MapLibre frame and when the figures change, never per train frame.

import { effect } from "@preact/signals";
import type { Map as MlMap } from "maplibre-gl";
import { complexes } from "../game/complexView";
import { demandView, type DemandView } from "../game/demand";
import { mixOklab } from "../game/palette";
import { stationUse, useFigure } from "../game/stationUse";
import type { WorldView } from "../game/types";
import { display, shade, stopEditing, type StationUseMode } from "../game/ui";
import { world } from "../game/world";
import { sizeCanvasLike } from "./gl";
import type { Overlay } from "./overlay";
import { stationHaloRadius, STATIONS_ZOOM } from "./stationGeometry";

/** Circle area is proportional to riders a day (T-722, no cap). The radius is
 * `stationHaloRadius(zoom)` times this scale, so STATION_FULL riders have that radius at any zoom; a
 * circle with any riders is never smaller than STATION_MIN_SCALE of it. 140,000 puts the real New
 * York save's biggest total (Penn Station's complex, 681,000) at 100 px radius at zoom 13 (Anita:
 * the largest about 200 px across). */
export const STATION_FULL = 140000, STATION_MIN_SCALE = 0.1;
export function stationScale(ridersPerDay: number): number {
  return ridersPerDay > 0 ? Math.max(STATION_MIN_SCALE, Math.sqrt(ridersPerDay / STATION_FULL)) : 0;
}

/** Entries blue, transfers green, per shade: the two end colours, how opaque a disc is, and the land
 * colour its thin edge is cased in. A circle is one colour between the ends, by its transfers' share. */
export const CIRCLE_COLOURS = {
  light: { entries: "#2456e0", transfers: "#169a50", fill: 0.2, casing: "#f3eee5" },
  dark: { entries: "#8aa6ff", transfers: "#52d68e", fill: 0.26, casing: "#292520" },
};
export const circleColours = () => CIRCLE_COLOURS[shade.peek()];

const mixed = new Map<string, string>();
/** A circle's colour: the entries blue at no transfers, the transfers green at all of them, between
 * by the share (T-1531); 32 steps, kept. */
export function circleColour(share: number, col = circleColours()): string {
  const step = Math.round(Math.min(1, Math.max(0, share)) * 32);
  const key = `${col.entries}${col.transfers}${step}`;
  let c = mixed.get(key);
  if (!c) mixed.set(key, (c = mixOklab(col.entries, col.transfers, step / 32)));
  return c;
}

export interface Circle {
  /** its stations' mean position, local units */
  x: number;
  y: number;
  /** riders a day the mode counts */
  riders: number;
  /** under Total, the transfers among them (the rest are entries); 0 for Origins, all for Transfers */
  transfers: number;
  /** its busiest station's id and name */
  station: string;
  name: string;
}

let circleKey: unknown[] = [];
let circles: Circle[] = [];

/** Every circle with riders (T-630), biggest first; kept until the network, world, solve or display
 * changes, since working it out is a pass over every station's complex (tens of ms on a world save). */
export function circleList(net: NonNullable<Overlay["renderer"]["net"]>, w: WorldView, dv: DemandView, mode: StationUseMode): Circle[] {
  const key = [net, w, dv, mode, complexes.peek(), stationUse.peek()];
  if (key.every((v, i) => v === circleKey[i])) return circles;
  circleKey = key;
  const index = new Map(net.stationIds.map((id, i) => [id, i]));
  const names = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const built = new Set(w.save.stations.filter((s) => s.built).map((s) => s.id));
  const cx = complexes.peek();
  const split = stationUse.peek()?.at;
  const seen = new Set<string>();
  const out: Circle[] = [];
  for (const s of w.save.stations) {
    if (!s.built || seen.has(s.id)) continue;
    const k0 = cx.of.get(s.id);
    const members = k0 !== undefined ? cx.members[k0].filter((m) => built.has(m)) : [s.id];
    let riders = 0, transfers = 0, known = false, top = s.id, topV = -1;
    for (const m of members) {
      seen.add(m);
      const v = useFigure(dv, mode, m);
      if (v === undefined) continue;
      riders += v;
      known = true;
      if (mode === "total") transfers += Math.min(v, split?.get(m)?.transfers ?? 0);
      if (v > topV) (topV = v), (top = m);
    }
    if (!known || riders <= 0) continue;
    let x = 0, y = 0, k = 0;
    for (const m of members) {
      const j = index.get(m);
      if (j === undefined) continue;
      x += net.stations[j * 3];
      y += net.stations[j * 3 + 1];
      k++;
    }
    if (!k) continue;
    out.push({ x: x / k, y: y / k, riders, transfers: mode === "transfers" ? riders : transfers, station: top, name: names.get(top) ?? "Station" });
  }
  circles = out.sort((a, b) => b.riders - a.riders);
  return circles;
}

/** The mode drawn now: none while stops are edited (T-233), as the demand views go then. */
function modeNow(): StationUseMode {
  return stopEditing.peek() ? "none" : display.peek().stationUse;
}

/** The circles on screen at the last draw, CSS px, smallest last. */
interface Placed { c: Circle; x: number; y: number; r: number }

export class StationCircles {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private placed: Placed[] = [];
  private cssW = 0;
  private cssH = 0;
  private dirty = true;

  constructor(private map: MlMap, private overlay: Overlay, below: HTMLCanvasElement) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay stations";
    below.after(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    this.size();
    map.on("resize", () => {
      this.size();
      this.draw();
    });
    overlay.afterMapFrame.push(() => this.draw());
    effect(() => {
      world.value;
      demandView.value;
      stationUse.value;
      complexes.value;
      shade.value;
      stopEditing.value;
      void display.value.stationUse;
      this.invalidate();
    });
  }

  private size() {
    [this.cssW, this.cssH] = sizeCanvasLike(this.canvas, this.map);
  }

  /** The figures or the network changed: draw again on the next animation frame. */
  invalidate() {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => this.dirty && this.draw());
  }

  draw() {
    this.dirty = false;
    const g = this.ctx, k = this.canvas.width / Math.max(1, this.cssW);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.placed = [];
    const net = this.overlay.renderer.net, w = world.peek(), dv = demandView.peek(), mode = modeNow(), zoom = this.map.getZoom();
    if (!net || !w || !dv || mode === "none" || zoom < STATIONS_ZOOM || !this.overlay.cam) return;
    const R = stationHaloRadius(zoom), at: [number, number] = [NaN, NaN];
    for (const c of circleList(net, w, dv, mode)) {
      this.overlay.toScreenInto(c.x, c.y, at);
      const r = R * stationScale(c.riders);
      if (!(r > 0) || Number.isNaN(at[0]) || at[0] < -r || at[1] < -r || at[0] > this.cssW + r || at[1] > this.cssH + r) continue;
      this.placed.push({ c, x: at[0], y: at[1], r });
    }
    g.setTransform(k, 0, 0, k, 0, 0);
    const col = circleColours();
    for (const p of this.placed) drawCircle(g, p.x, p.y, p.r, p.c.transfers / p.c.riders, col);
  }

  /** The circle under a point (CSS px in the map) as last drawn: of those it is inside, the one whose
   * centre it is relatively nearest; null for none. */
  at(px: number, py: number): Circle | null {
    let best: Circle | null = null, bestF = 1;
    for (const p of this.placed) {
      const dx = p.x - px, dy = p.y - py;
      if (dx > p.r || dx < -p.r || dy > p.r || dy < -p.r) continue;
      const f = Math.hypot(dx, dy) / p.r;
      if (f <= bestF) [best, bestF] = [p.c, f];
    }
    return best;
  }
}

/** The one instance, for the hover (networkTips.ts); set by `startDemandViews`. */
export const stationCircles: { now: StationCircles | null } = { now: null };

/** A circle of radius `r` at (x, y), `t` of its riders transfers: one disc in the colour for that
 * share, with a thin faint edge over an even fainter casing of the land, so it holds against the
 * bubbles beneath. */
function drawCircle(g: CanvasRenderingContext2D, x: number, y: number, r: number, t: number, col: (typeof CIRCLE_COLOURS)["light"]) {
  const colour = circleColour(t, col);
  g.beginPath();
  g.arc(x, y, r, 0, FULL);
  g.globalAlpha = col.fill;
  g.fillStyle = colour;
  g.fill();
  g.beginPath();
  g.arc(x, y, Math.max(0.5, r - 0.5), 0, FULL);
  g.globalAlpha = 0.3;
  g.strokeStyle = col.casing;
  g.lineWidth = 2.5;
  g.stroke();
  g.globalAlpha = 0.5;
  g.strokeStyle = colour;
  g.lineWidth = 1;
  g.stroke();
  g.globalAlpha = 1;
}
const FULL = Math.PI * 2;
