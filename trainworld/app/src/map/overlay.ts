// The network is drawn on its own WebGL2 canvas over the map (T-015), so MapLibre stays idle while
// trains move and the camera is still: no basemap repaint per train frame (notes/T-002.md).
//
// Staying glued to the basemap: an empty custom layer hands over MapLibre's camera matrix on each
// frame MapLibre draws, and the overlay is drawn right there, in the same frame with the same
// matrix. The clock's animation loop skips its draw when MapLibre has a frame queued, avoiding
// a second draw with the previous camera during panning (T-075). With the camera still, the
// clock loop draws normally and MapLibre stays idle.
//
// Frames (SPEC 7): every frame while the clock runs and the window has focus; 10 fps while visible
// but unfocused; none while hidden or paused, apart from MapLibre's own frames (camera moves,
// tiles) and one-off redraws after a state change (`invalidate`).

import type { Map as MlMap } from "maplibre-gl";
import { clock, SPEEDS } from "../game/clock";
import type { DrawOptions, NetworkRenderer } from "./renderer";
import { anchor, EARTH_C, toLocal } from "./geo";
import { cameraFrom, clipToCss, mercToCss, sizeCanvasLike, type Camera } from "./gl";
import { perfOff } from "../perfFlags";
import { cameraMoving } from "../game/ui";

/** ?perfOff=overlay (T-045, measurement only): skip the WebGL draw. */
const NO_DRAW = perfOff("overlay");
/**
 * Idle frames (T-073, adopted from T-045's ?perfTry=idle): with no trips this hour, no frames at
 * all; the clock wakes once a game minute. New game 13% -> 0.6% of a core. ?perfOff=idle reverts.
 * Skip duplicate clock-loop draws when MapLibre has queued a frame (T-075).
 */
const IDLE = !perfOff("idle");
const NO_DOUBLE = !perfOff("noDouble");
/** ?perfOff=moving (T-1207, measurement only): nothing waits while the camera moves. */
const THROTTLE = !perfOff("moving");

/** Frame interval while the window is visible but does not have focus. */
const UNFOCUSED_MS = 100;
/** A view of the network drawn into another canvas (T-1289, ui/inspector/ComplexMap.tsx). */
export interface Snapshot {
  canvas: HTMLCanvasElement;
  /** its size, CSS px */
  w: number;
  h: number;
  /** its centre, Mercator 0..1, and its zoom */
  mx: number;
  my: number;
  zoom: number;
}

const RING = 120;
export class Ring {
  a = new Float64Array(RING);
  n = 0;
  i = 0;
  push(v: number) {
    this.a[this.i] = v;
    this.i = (this.i + 1) % RING;
    this.n = Math.min(RING, this.n + 1);
  }
  avg() {
    let s = 0;
    for (let k = 0; k < this.n; k++) s += this.a[k];
    return this.n ? s / this.n : 0;
  }
  max() {
    let m = 0;
    for (let k = 0; k < this.n; k++) m = Math.max(m, this.a[k]);
    return m;
  }
  clear() {
    this.n = this.i = 0;
  }
}

export class Overlay {
  readonly canvas: HTMLCanvasElement;
  readonly gl: WebGL2RenderingContext;
  /** The camera MapLibre last drew with (flat or globe, gl.ts). */
  cam: Camera | null = null;
  /** Its flat map's matrix, Mercator 0..1 to clip space, float64: the whole camera while `cam.globe`
   * is null (the flat map, and the globe from zoom 12). */
  camMatrix: ArrayLike<number> | null = null;
  private focused = document.hasFocus();
  private rafId = 0;
  private slowTimer = 0;
  private lastTick = 0;
  private lastFrame = 0;
  // stats for ?debug=1
  readonly drawMs = new Ring();
  readonly frameGap = new Ring();
  frames = 0;
  mapFrames = 0;
  /** matrix of the last overlay draw, for the T-015 lag probe */
  lastDrawnMatrix: ArrayLike<number> | null = null;
  /** called after each MapLibre frame (the camera may have moved), e.g. to move DOM labels */
  readonly afterMapFrame: (() => void)[] = [];
  /** The camera is moving (MapLibre's movestart to moveend): DOM placers re-place at most every
   * `MOVING_PLACE_MS` meanwhile (`placeDue`), and once more on the frame after it stops (T-1207). */
  moving = false;
  /** MapLibre frames drawn while the camera moves: their gaps, ms (the lighter map's auto switch). */
  onMovingFrame: ((gapMs: number) => void) | null = null;
  private lastMapFrame = 0;

  constructor(
    private map: MlMap,
    readonly renderer: NetworkRenderer,
    private options: () => DrawOptions,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay";
    map.getContainer().appendChild(this.canvas);
    // a stencil for the track drawn through the station boxes (T-236, renderer.ts)
    const gl = this.canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: true, alpha: true, stencil: true });
    if (!gl) throw new Error("no WebGL2");
    this.gl = gl;
    renderer.init(gl);
    // Phones drop WebGL contexts (a tab in the background, memory short, a GPU reset): taking the
    // loss lets the browser hand the canvas a new context, which the renderer rebuilds into (T-953).
    this.canvas.addEventListener("webglcontextlost", (e) => e.preventDefault());
    this.canvas.addEventListener("webglcontextrestored", () => {
      renderer.restore();
      this.invalidate();
    });
    this.size();
    map.on("resize", () => {
      this.size();
      this.invalidate();
    });
    const addCamera = () => {
      if (map.getLayer("tw-camera")) return;
      map.addLayer({
        id: "tw-camera",
        type: "custom",
        renderingMode: "2d",
        render: (_gl, args) => {
          this.cam = cameraFrom(args.defaultProjectionData);
          this.camMatrix = this.cam.merc;
          this.mapFrames++;
          const t = performance.now();
          if (this.moving && this.lastMapFrame) this.onMovingFrame?.(t - this.lastMapFrame);
          this.lastMapFrame = this.moving ? t : 0;
          this.draw();
          for (const f of this.afterMapFrame) f();
        },
      });
    };
    if (map.isStyleLoaded()) addCamera();
    else map.on("load", addCamera);
    // MapLibre restores its style after a lost context without custom layers (T-953)
    map.on("style.load", addCamera);
    map.on("movestart", () => {
      if (THROTTLE) this.moving = this.renderer.moving = cameraMoving.now = true;
    });
    // the frame after the move: every placer places in full again
    map.on("moveend", () => {
      this.moving = this.renderer.moving = cameraMoving.now = false;
      map.triggerRepaint();
    });

    window.addEventListener("blur", () => {
      this.focused = false;
      this.lastFrame = 0;
    });
    window.addEventListener("focus", () => {
      this.focused = true;
      this.lastFrame = 0;
      this.kick();
    });
    clock.onChange(() => {
      this.lastFrame = 0;
      this.kick();
      this.invalidate();
    });
  }

  private size() {
    [this.cssW, this.cssH] = sizeCanvasLike(this.canvas, this.map);
  }

  /** The canvas size in CSS px, kept from the last resize: reading clientWidth in `toScreen` forced
   * a layout per call while DOM labels were being rebuilt (T-063). DOM markers read it too (T-496). */
  cssW = 0;
  cssH = 0;

  /** T-1289: a second view drawn with each frame, the station panel's complex map; null for none. */
  private snap: Snapshot | null = null;

  /** Draw `s` along with every frame from now on (null: stop). The caller sizes nothing: the
   * target's backing store is set here, at the map's pixel ratio unless the map's canvas is smaller. */
  setSnapshot(s: Snapshot | null) {
    this.snap = s;
    if (s) this.invalidate();
  }

  /** Draw now with the camera MapLibre last drew. */
  draw() {
    if (!this.cam || !this.renderer.net) return;
    const a = performance.now();
    const gl = this.gl;
    clock.tick();
    const opt = this.options();
    // the snapshot first, so the main draw leaves the renderer's picking state as the map's
    if (this.snap && !NO_DRAW && !gl.isContextLost()) this.drawSnapshot(this.snap, opt);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!NO_DRAW && !gl.isContextLost()) this.renderer.draw(this.cam, clock.now(), this.map.getZoom(), this.map.getPixelRatio(), opt);
    this.lastDrawnMatrix = this.camMatrix;
    const now = performance.now();
    this.drawMs.push(now - a);
    if (this.lastFrame && clock.ticking && now - this.lastFrame < 1000) this.frameGap.push(now - this.lastFrame);
    this.lastFrame = now;
    this.frames++;
  }

  /** The snapshot drawn by the network renderer with a flat camera of its own into the bottom left of
   * this canvas, copied to its target while the frame is still unpresented, then cleared by the
   * frame's own draw: the same shaders, antialiasing and stencil as the map, no second context. */
  private drawSnapshot(s: Snapshot, opt: DrawOptions) {
    const gl = this.gl, cv = this.canvas;
    const dpr = Math.min(this.map.getPixelRatio(), cv.width / s.w, cv.height / s.h);
    const w = Math.floor(s.w * dpr), h = Math.floor(s.h * dpr);
    if (w < 2 || h < 2) return;
    // Mercator 0..1 to clip space, as MapLibre's flat camera at `zoom` (512 px tiles), north up
    const S = 512 * 2 ** s.zoom, kx = (2 * S) / s.w, ky = (2 * S) / s.h;
    const merc = [kx, 0, 0, 0, 0, -ky, 0, 0, 0, 0, 1, 0, -s.mx * kx, s.my * ky, 0, 1];
    gl.viewport(0, 0, w, h);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.renderer.draw({ merc, globe: null, plane: [0, 0, 0, 1], t: 0 }, clock.now(), s.zoom, dpr, opt, [w, h]);
    gl.disable(gl.SCISSOR_TEST);
    const t = s.canvas;
    if (t.width !== w || t.height !== h) {
      t.width = w;
      t.height = h;
    }
    const ctx = t.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(cv, 0, cv.height - h, w, h, 0, 0, w, h);
  }

  /** Something drawn changed (selection, settings, network): redraw once if no loop is running. */
  invalidate() {
    if (!this.rafId) this.rafId = requestAnimationFrame(() => this.tick());
  }

  private tick() {
    this.rafId = 0;
    // MapLibre draws the overlay itself in a frame it renders (camera layer); ours would be a
    // second draw of the same frame.
    if (NO_DOUBLE && (this.map as unknown as { _frameRequest?: unknown })._frameRequest) return this.scheduleNext();
    this.draw();
    this.scheduleNext();
  }

  private frameNow() {
    this.lastTick = performance.now();
    if (!this.rafId) this.rafId = requestAnimationFrame(() => this.tick());
  }

  /** After a frame: the next one at full rate with focus, at 10 fps without, none if stopped. */
  private scheduleNext() {
    if (!clock.ticking) return;
    if (IDLE && this.renderer.net && this.renderer.tripCount === 0) {
      // Nothing moves: one frame when the game minute turns, for the clock.
      if (this.slowTimer) return;
      const t = clock.now();
      const wait = (((Math.floor(t / 60) + 1) * 60 - t) / SPEEDS[clock.speed.peek()]) * 1000;
      this.slowTimer = window.setTimeout(() => {
        this.slowTimer = 0;
        if (clock.ticking) this.frameNow();
      }, Math.max(1, wait));
      return;
    }
    if (this.focused) return this.frameNow();
    if (this.slowTimer) return;
    // The frame itself lands on the next animation frame after the timer, ~8 ms later on average.
    const wait = Math.max(0, UNFOCUSED_MS - 8 - (performance.now() - this.lastTick));
    this.slowTimer = window.setTimeout(() => {
      this.slowTimer = 0;
      if (clock.ticking) this.frameNow();
    }, wait);
  }

  kick() {
    if (clock.ticking) this.frameNow();
  }

  /** local units (of the current anchor, which the drawn network shares, T-307) -> CSS px in the
   * map container, with the camera of the last frame; NaN behind the globe's horizon */
  toScreen(x: number, y: number): [number, number] {
    const out: [number, number] = [NaN, NaN];
    this.toScreenInto(x, y, out);
    return out;
  }

  /** `toScreen` into `out[at]`, `out[at + 1]`, allocating nothing (T-180: hovering a big network
   * projects every segment on each pointer move, and the two arrays per point were most of the
   * cost). */
  toScreenInto(x: number, y: number, out: { [i: number]: number }, at = 0) {
    const c = this.cam;
    if (!c) { out[at] = out[at + 1] = NaN; return; }
    const o = anchor().merc;
    if (c.globe) mercToCss(c, o[0] + x / EARTH_C, o[1] + y / EARTH_C, this.cssW, this.cssH, out, at);
    else clipToCss(c.merc, o[0] + x / EARTH_C, o[1] + y / EARTH_C, this.cssW, this.cssH, out, at);
  }

  /** Mercator 0..1 -> CSS px in the map container, as `toScreenInto`. */
  mercToScreen(mx: number, my: number, out: { [i: number]: number }, at = 0) {
    const c = this.cam;
    if (!c) { out[at] = out[at + 1] = NaN; return; }
    if (c.globe) mercToCss(c, mx, my, this.cssW, this.cssH, out, at);
    else clipToCss(c.merc, mx, my, this.cssW, this.cssH, out, at);
  }

  /** CSS px in the map container -> local units of the current anchor on the globe's surface, or
   * null off it (MapLibre's own unproject, checked by projecting back). */
  screenToLocal(px: number, py: number): [number, number] | null {
    const ll = this.map.unproject([px, py]);
    const [x, y] = toLocal(ll.lng, ll.lat), back: [number, number] = [NaN, NaN];
    this.toScreenInto(x, y, back);
    return Math.hypot(back[0] - px, back[1] - py) < 2 ? [x, y] : null;
  }

  /** The globe is drawn (MapLibre's globe, short of zoom 12 where it is flat). */
  get globe(): boolean {
    return !!this.cam?.globe;
  }
}
