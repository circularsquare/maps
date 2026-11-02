// The gap a stop cannot cross (T-1896, docs/ui-building.md "Stops"): after a click on a station no
// track joins to the line being edited, the clock worker's `stop_gap` names the two nearest points of
// the line's track and the station's, and this draws the straight line between them for a few
// seconds, in the trip explorer's look for a straight line (tripView.ts, T-1252), with its open ring
// at both ends. The camera moves only if the gap is not all in view.

import type { Map as MlMap } from "maplibre-gl";
import { NET_INK } from "../game/palette";
import { shadeColours } from "../game/shades";
import { sizeCanvasLike } from "./gl";
import type { Overlay } from "./overlay";

/** Shown at full strength this long, then faded out over FADE_MS. */
const HOLD_MS = 4200;
const FADE_MS = 800;
/** Room kept round the gap when the camera moves to it, CSS px. */
const FRAME_PAD = 90;

export class GapFlash {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  /** the two ends, lng/lat, while one is shown */
  private ends: [[number, number], [number, number]] | null = null;
  private start = 0;
  private raf = 0;
  private timer = 0;

  constructor(private map: MlMap, overlay: Overlay) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay gap";
    // over the network, under the stop markers (style.css .stop-marks, z 3)
    this.canvas.style.zIndex = "2";
    overlay.canvas.after(this.canvas);
    this.ctx = this.canvas.getContext("2d")!;
    sizeCanvasLike(this.canvas, map);
    map.on("resize", () => {
      sizeCanvasLike(this.canvas, map);
      this.draw();
    });
    overlay.afterMapFrame.push(() => {
      if (this.ends) this.draw();
    });
  }

  /** Show the gap from `a` to `b` (lng/lat), framing it if it is not in view. */
  show(a: [number, number], b: [number, number]) {
    this.ends = [a, b];
    this.start = performance.now();
    if (!this.inView(a) || !this.inView(b)) {
      const zoom = this.map.getZoom();
      const bounds: [[number, number], [number, number]] = [[Math.min(a[0], b[0]), Math.min(a[1], b[1])], [Math.max(a[0], b[0]), Math.max(a[1], b[1])]];
      // never closer in than the player was
      this.map.fitBounds(bounds, { padding: FRAME_PAD, maxZoom: zoom, duration: 600 });
    }
    this.tick();
  }

  clear() {
    if (!this.ends) return;
    this.ends = null;
    this.stopTimers();
    this.draw();
  }

  private inView(p: [number, number]): boolean {
    const s = this.map.project(p), c = this.map.getCanvas();
    return s.x >= FRAME_PAD / 2 && s.y >= FRAME_PAD / 2 && s.x <= c.clientWidth - FRAME_PAD / 2 && s.y <= c.clientHeight - FRAME_PAD / 2;
  }

  /** Redraw on each animation frame only while fading; the map's own frames redraw it otherwise. */
  private tick() {
    this.stopTimers();
    if (!this.ends) return;
    const t = performance.now() - this.start;
    if (t >= HOLD_MS + FADE_MS) {
      this.clear();
      return;
    }
    this.draw();
    if (t < HOLD_MS) this.timer = window.setTimeout(() => this.tick(), HOLD_MS - t);
    else this.raf = requestAnimationFrame(() => this.tick());
  }

  private stopTimers() {
    cancelAnimationFrame(this.raf);
    window.clearTimeout(this.timer);
    this.raf = this.timer = 0;
  }

  private draw() {
    const ctx = this.ctx, dpr = this.map.getPixelRatio();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.ends) return;
    const t = performance.now() - this.start;
    ctx.globalAlpha = t <= HOLD_MS ? 1 : Math.max(0, 1 - (t - HOLD_MS) / FADE_MS);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const shades = shadeColours();
    const [a, b] = this.ends.map((p) => this.map.project(p));
    ctx.lineCap = "round";
    ctx.setLineDash([0.1, 7]);
    for (const [colour, width] of [[shades.casing, 6], [shades.mapInk, 3.5]] as const) {
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    for (const p of [a, b]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = NET_INK;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8.5, 0, 2 * Math.PI);
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}
