// The demand views' map layer (T-084, T-097, SPEC 8): commuter bubbles, a bubble selection's far
// end and longer trips, as discs on a canvas of their own over the network, with thin lines to a
// selection under them (T-520)
// (Anita: in a demand view the demand is what you want to see, on top). Nothing here moves with
// the clock, so the canvas is drawn only on MapLibre's frames (the camera moved:
// `Overlay.afterMapFrame`, same frame and matrix as the basemap, T-015) and once after its data
// changes. While the game plays with the camera still it is never touched.

import type { Map as MlMap } from "maplibre-gl";
import { rgb } from "../game/palette";
import { shadeColours } from "../game/shades";
import { DISC } from "../workers/commutersProtocol";
import { anchor, EARTH_C, localToMercMatrix, mul4, type Anchor } from "./geo";
import { hot, onHot } from "./hovered";
import type { Overlay } from "./overlay";
import { PLATFORM_POINTS, SEL_GLOW_PX, SEL_OUTLINE_PX, SHAPE_GLSL, shapePoints, type StationShape } from "./stationGeometry";

// The selected station shows above the bubbles (T-189, Anita): after the discs, the canvas is
// cleared where it is drawn, fully over the station and its outline and fading out over its glow,
// so the network canvas below (the station, its glow, a train standing in it) shows through. One
// quad per selected station, in CSS px; the shape is the renderer's (stationGeometry.ts).
const CUT_VS = `#version 300 es
precision highp float;
uniform vec2 u_css;      // the canvas, CSS px
uniform vec2 u_c;        // the station's middle, CSS px (y down)
uniform vec2 u_u;        // along it and across it
uniform vec2 u_n;
uniform vec3 u_box;      // the quad in its frame: half its length, then from and to across
out vec2 v_p;
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  v_p = vec2(corner.x * u_box.x, corner.y < 0.0 ? u_box.y : u_box.z);
  vec2 s = u_c + u_u * v_p.x + u_n * v_p.y;
  gl_Position = vec4(s.x / u_css.x * 2.0 - 1.0, 1.0 - s.y / u_css.y * 2.0, 0.0, 1.0);
}`;

const CUT_FS = `#version 300 es
precision highp float;
uniform vec3 u_size;     // half length, half width, (unused)
uniform float u_dot;
uniform vec2 u_q[${PLATFORM_POINTS}]; // the centre line in the frame, CSS px (T-254)
uniform float u_outline;
uniform float u_glow;
in vec2 v_p;
out vec4 o;
${SHAPE_GLSL}
void main() {
  float d = u_dot > 0.0 ? length(v_p) - u_dot : boxSdf(polyFrame(v_p, u_q), u_size.xy);
  o = vec4(0.0, 0.0, 0.0, 1.0 - clamp((d - u_outline) / u_glow, 0.0, 1.0));
}`;

/** A hovered bubble's colour mixed this far toward white (T-730). */
const HOT_MIX = "0.2";

// A disc per instance: radius in local units, colour and alpha, and an edge (0 none, 2 a dark
// outline of 2 px, a selected bubble's). No minimum size: a tiny count is a tiny disc.
const DISC_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_px_per_unit;
uniform float u_rscale;   // the player's bubble size (T-099)
uniform float u_opacity;  // the player's bubble opacity (T-470)
uniform int u_hot;        // the hovered disc of this set, -1 none (T-730)
in vec3 a_geo;    // x, y, radius (local units)
in vec4 a_col;    // r, g, b, a
in float a_edge;
out vec2 v_p;
out float v_r;
out float v_edge;
out vec4 v_col;
void main() {
  bool hot = gl_InstanceID == u_hot;
  float r = a_geo.z * u_rscale * u_px_per_unit;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  vec4 c = u_matrix * vec4(a_geo.xy, 0.0, 1.0);
  vec2 off = corner * (r + 2.0);
  c.xy += off / (u_viewport * 0.5) * c.w;
  gl_Position = c;
  v_p = off;
  v_r = r;
  v_edge = a_edge;
  v_col = vec4(hot ? mix(a_col.rgb, vec3(1.0), ${HOT_MIX}) : a_col.rgb, a_col.a * u_opacity);
}`;

const DISC_FS = `#version 300 es
precision highp float;
uniform vec3 u_ink;
uniform float u_dpr;
in vec2 v_p;
in float v_r;
in float v_edge;
in vec4 v_col;
out vec4 o;
void main() {
  float d = length(v_p);
  float cover = clamp(v_r - d + 0.5, 0.0, 1.0);
  if (cover <= 0.0) discard;
  vec3 col = v_col.rgb;
  float a = v_col.a;
  if (v_edge > 1.5) {
    // a dark outline 2 px wide, the inside left as it is
    float t = clamp(d - (v_r - 2.0 * u_dpr) + 0.5, 0.0, 1.0);
    col = mix(col, u_ink, t);
    a = mix(a, 1.0, t);
  }
  a *= cover;
  o = vec4(col * a, a);
}`;

/** Floats per line instance: x0, y0, r0, x1, y1, r1 (local units, radii before the bubble size),
 * a stub length factor (0 = a line), alpha, and the colour r, g, b (the first disc's own). A line
 * runs edge to edge between the two discs; a stub runs from the first disc's edge toward the
 * second, `STUB_SCREEN` of the screen long times its factor, at constant width and alpha (T-520,
 * T-739). Its alpha is scaled by the first disc's drawn size (T-524). */
export const LINE = 11;
/** A stub's length (factor 1) as a share of the map's longer side. */
const STUB_SCREEN = 0.06;
/** Line width, CSS px. */
const LINE_PX = 1.5;
/** A line's source disc drawn at this radius (CSS px) or more gives the line its full alpha;
 * smaller ones give it in proportion to the radius. */
const LINE_FULL_PX = 8;

const LINE_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;  // device px
uniform float u_px_per_unit;
uniform float u_rscale;
uniform float u_stub;     // device px
uniform float u_width;    // device px
uniform float u_full;     // device px: the source radius that gives the full alpha
in vec3 a_from;
in vec3 a_to;
in vec2 a_style;          // stub length factor, alpha
in vec3 a_col;
out float v_across;
out float v_alpha;
out vec3 v_col;
void main() {
  vec4 c0 = u_matrix * vec4(a_from.xy, 0.0, 1.0);
  vec4 c1 = u_matrix * vec4(a_to.xy, 0.0, 1.0);
  vec2 p0 = (c0.xy / c0.w * 0.5 + 0.5) * u_viewport;
  vec2 p1 = (c1.xy / c1.w * 0.5 + 0.5) * u_viewport;
  vec2 d = p1 - p0;
  float len = length(d);
  vec2 u = len > 0.0 ? d / len : vec2(1.0, 0.0);
  float s0 = a_from.z * u_rscale * u_px_per_unit;
  float s1 = len - a_to.z * u_rscale * u_px_per_unit;
  if (a_style.x > 0.5) s1 = min(s1, s0 + u_stub * a_style.x);
  float along = (gl_VertexID & 1) == 1 ? s1 : s0;
  float side = gl_VertexID < 2 ? -1.0 : 1.0;
  float half_w = u_width * 0.5 + 1.0;
  vec2 p = p0 + u * along + vec2(-u.y, u.x) * side * half_w;
  // nothing left between the two edges: off screen
  gl_Position = s1 > s0 ? vec4(p / u_viewport * 2.0 - 1.0, 0.0, 1.0) : vec4(2.0, 2.0, 2.0, 1.0);
  v_across = side * half_w;
  v_alpha = a_style.y * clamp(s0 / u_full, 0.0, 1.0);
  v_col = a_col;
}`;

const LINE_FS = `#version 300 es
precision highp float;
uniform float u_width;
in float v_across;
in float v_alpha;
in vec3 v_col;
out vec4 o;
void main() {
  float a = v_alpha * clamp(u_width * 0.5 - abs(v_across) + 0.5, 0.0, 1.0);
  o = vec4(v_col * a, a);
}`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
  return prog;
}

/**
 * Which aggregation level (index into BUBBLE_RES: resolution 9, 8, 7, 6) to show at a zoom. Each
 * step is a 7 times larger area, about 1.4 zoom levels, so the bubbles keep about the same spacing
 * on screen: 40-60 px between neighbours.
 */
export function bubbleLevel(z: number): number {
  return z >= 13.6 ? 0 : z >= 11.8 ? 1 : z >= 10 ? 2 : 3;
}

/** Disc layers, drawn in this order. */
export type DiscLayer = "inter" | "far" | "near";
// "inter": a selection's longer trips (T-118), under the commuters' discs
const LAYERS: DiscLayer[] = ["inter", "far", "near"];

interface DiscSet {
  vao: WebGLVertexArrayObject;
  buf: WebGLBuffer;
  count: number;
  /** the draw anchor the discs' positions are relative to (T-317): drawn with its matrix */
  anchor: Anchor;
  /** the discs it was made from, matched against the hovered disc (T-730) */
  data?: Float32Array;
}

export class DemandOverlay {
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private bubbles: (DiscSet | null)[] = [];
  private sets = new Map<DiscLayer, DiscSet>();
  private showBubbles = false;
  private rafId = 0;
  private blank = true;
  private timerExt: any;
  private query: WebGLQuery | null = null;
  /** last GPU time of a draw, ms (the timer extension, when there is one) */
  gpuMs: number | null = null;
  /** CPU time of the last draw, ms; upload time of the last bubble set, ms */
  drawMs = 0;
  uploadMs = 0;
  draws = 0;
  /** discs in the last draw */
  discsDrawn = 0;
  /** called when the aggregation level on screen changes */
  onLevel: (level: number) => void = () => {};
  private lastLevel = -1;
  /** the selected stations on screen, cut out of the discs (T-189); none when unset */
  cutouts: (() => StationShape[]) | null = null;
  private cutProg: WebGLProgram | null = null;
  private cutU: Record<string, WebGLUniformLocation | null> = {};
  private lineProg: WebGLProgram | null = null;
  private lineU: Record<string, WebGLUniformLocation | null> = {};
  private lines = new Map<DiscLayer, DiscSet>();

  constructor(private map: MlMap, private overlay: Overlay) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay demand";
    // over the network's canvas, under the station names and markers
    overlay.canvas.after(this.canvas);
    const gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, alpha: true });
    if (!gl) throw new Error("no WebGL2");
    this.gl = gl;
    this.prog = compile(gl, DISC_VS, DISC_FS);
    for (const n of ["u_matrix", "u_viewport", "u_px_per_unit", "u_ink", "u_dpr", "u_rscale", "u_opacity", "u_hot"]) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.size();
    map.on("resize", () => {
      this.size();
      this.invalidate();
    });
    overlay.afterMapFrame.push(() => this.draw());
    let lit = hot().disc;
    onHot(() => {
      const d = hot().disc;
      if (d?.discs === lit?.discs && d?.i === lit?.i) return;
      lit = d;
      this.invalidate();
    });
  }

  private size() {
    const dpr = this.map.getPixelRatio();
    const c = this.map.getCanvas();
    this.canvas.width = Math.round(c.clientWidth * dpr);
    this.canvas.height = Math.round(c.clientHeight * dpr);
    this.canvas.style.width = c.clientWidth + "px";
    this.canvas.style.height = c.clientHeight + "px";
  }

  private makeSet(data: Float32Array, a: Anchor): DiscSet {
    const gl = this.gl;
    const buf = gl.createBuffer()!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const geo = gl.getAttribLocation(this.prog, "a_geo"), col = gl.getAttribLocation(this.prog, "a_col");
    gl.enableVertexAttribArray(geo);
    gl.vertexAttribPointer(geo, 3, gl.FLOAT, false, DISC * 4, 0);
    gl.vertexAttribDivisor(geo, 1);
    gl.enableVertexAttribArray(col);
    gl.vertexAttribPointer(col, 4, gl.FLOAT, false, DISC * 4, 12);
    gl.vertexAttribDivisor(col, 1);
    const edge = gl.getAttribLocation(this.prog, "a_edge");
    if (edge >= 0) {
      gl.enableVertexAttribArray(edge);
      gl.vertexAttribPointer(edge, 1, gl.FLOAT, false, DISC * 4, 28);
      gl.vertexAttribDivisor(edge, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { vao, buf, count: data.length / DISC, anchor: a, data };
  }

  private dropSet(s: DiscSet | null | undefined) {
    if (!s) return;
    this.gl.deleteBuffer(s.buf);
    this.gl.deleteVertexArray(s.vao);
  }

  /** The city's bubbles, one disc set per aggregation level, built relative to anchor `a` (T-317);
   * null for none. */
  setBubbles(levels: Float32Array[] | null, a: Anchor = anchor()) {
    const t0 = performance.now();
    for (const s of this.bubbles) this.dropSet(s);
    this.bubbles = (levels ?? []).map((d) => (d.length ? this.makeSet(d, a) : null));
    this.uploadMs = performance.now() - t0;
    this.invalidate();
  }

  /** A nearby zone's bubbles (T-276), drawn under the focused zone's and shown with them; null
   * removes the zone's. */
  private nearby = new Map<string, (DiscSet | null)[]>();
  setNearbyBubbles(zone: string, levels: Float32Array[] | null, a: Anchor = anchor()) {
    for (const s of this.nearby.get(zone) ?? []) this.dropSet(s);
    this.nearby.delete(zone);
    if (levels) this.nearby.set(zone, levels.map((d) => (d.length ? this.makeSet(d, a) : null)));
    this.invalidate();
  }

  /** Commuter bubbles' radius as a multiple of the built one (the settings slider, T-099). */
  bubbleSize = 1;
  setBubbleSize(s: number) {
    if (s === this.bubbleSize) return;
    this.bubbleSize = s;
    this.invalidate();
  }

  /** Commuter and longer-trip bubbles' alpha as a multiple of the built one (the settings slider, T-470). */
  bubbleOpacity = 1;
  setBubbleOpacity(o: number) {
    if (o === this.bubbleOpacity) return;
    this.bubbleOpacity = o;
    this.invalidate();
  }

  /** Show or hide the city's bubbles (kept on the GPU while hidden). */
  setBubblesVisible(v: boolean) {
    if (v === this.showBubbles) return;
    this.showBubbles = v;
    this.invalidate();
  }

  /** One of the small disc layers (a selection's ends, longer trips), built relative to anchor `a`
   * (T-317); null clears it. */
  setDiscs(layer: DiscLayer, data: Float32Array | null, a: Anchor = anchor()) {
    this.dropSet(this.sets.get(layer));
    this.sets.delete(layer);
    if (data && data.length) this.sets.set(layer, this.makeSet(data, a));
    this.invalidate();
  }

  /** Lines toward a selection that go with a disc layer (`LINE` floats each), drawn under every
   * disc, relative to anchor `a`; null clears them. */
  setLines(layer: DiscLayer, data: Float32Array | null, a: Anchor = anchor()) {
    this.dropSet(this.lines.get(layer));
    this.lines.delete(layer);
    if (data && data.length) this.lines.set(layer, this.makeLineSet(data, a));
    this.invalidate();
  }

  private makeLineSet(data: Float32Array, a: Anchor): DiscSet {
    const gl = this.gl;
    if (!this.lineProg) {
      this.lineProg = compile(gl, LINE_VS, LINE_FS);
      for (const n of ["u_matrix", "u_viewport", "u_px_per_unit", "u_rscale", "u_stub", "u_width", "u_full"]) this.lineU[n] = gl.getUniformLocation(this.lineProg, n);
    }
    const buf = gl.createBuffer()!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    ([["a_from", 3, 0], ["a_to", 3, 12], ["a_style", 2, 24], ["a_col", 3, 32]] as const).forEach(([name, n, off]) => {
      const loc = gl.getAttribLocation(this.lineProg!, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, LINE * 4, off);
      gl.vertexAttribDivisor(loc, 1);
    });
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { vao, buf, count: data.length / LINE, anchor: a };
  }

  private drawLines(m0: ArrayLike<number>,matrices: Map<number, Float32Array>, pxPerUnit: number, dpr: number) {
    if (!this.lines.size || !this.lineProg) return;
    const gl = this.gl, u = this.lineU;
    gl.useProgram(this.lineProg);
    gl.uniform2f(u.u_viewport, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.u_px_per_unit, pxPerUnit);
    gl.uniform1f(u.u_rscale, this.bubbleSize);
    gl.uniform1f(u.u_stub, STUB_SCREEN * Math.max(this.canvas.width, this.canvas.height));
    gl.uniform1f(u.u_width, LINE_PX * dpr);
    gl.uniform1f(u.u_full, LINE_FULL_PX * dpr);
    for (const s of this.lines.values()) {
      let m = matrices.get(s.anchor.version);
      if (!m) matrices.set(s.anchor.version, (m = mul4(m0, localToMercMatrix(s.anchor))));
      gl.uniformMatrix4fv(u.u_matrix, false, m);
      gl.bindVertexArray(s.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
    }
    gl.bindVertexArray(null);
  }

  /** Draw once on the next frame (data changed), with the camera of MapLibre's last frame. */
  invalidate() {
    if (!this.rafId)
      this.rafId = requestAnimationFrame(() => {
        this.rafId = 0;
        this.draw();
      });
  }

  private draw() {
    const m0 = this.overlay.camMatrix;
    if (!m0) return;
    const gl = this.gl;
    const zoom = this.map.getZoom(), dpr = this.map.getPixelRatio();
    const level = bubbleLevel(zoom);
    if (level !== this.lastLevel) {
      this.lastLevel = level;
      this.onLevel(level);
    }
    const list: DiscSet[] = [];
    if (this.showBubbles) for (const sets of this.nearby.values()) {
      const n = sets[level];
      if (n) list.push(n);
    }
    const b = this.showBubbles ? this.bubbles[level] : null;
    if (b) list.push(b);
    for (const l of LAYERS) {
      const s = this.sets.get(l);
      if (s) list.push(s);
    }
    if (!list.length && !this.lines.size) {
      // clear once, then leave the canvas alone
      if (!this.blank) {
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        this.blank = true;
      }
      return;
    }
    const t0 = performance.now();
    this.blank = false;
    this.pollTimer();
    const timing = this.timerExt && !this.query;
    if (timing) {
      this.query = gl.createQuery();
      gl.beginQuery(this.timerExt.TIME_ELAPSED_EXT, this.query!);
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    // each set is drawn with the matrix of the anchor it was built at (T-317, map/geo.ts); most
    // draws have one anchor, a switch has two for a moment
    const matrices = new Map<number, Float32Array>();
    const pxPerUnit = ((512 * Math.pow(2, zoom)) / EARTH_C) * dpr;
    this.drawLines(m0, matrices, pxPerUnit, dpr);
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.u_viewport, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.u_px_per_unit, pxPerUnit);
    gl.uniform3fv(this.u.u_ink, rgb(shadeColours().mapInk).map((v) => v / 255));
    gl.uniform1f(this.u.u_dpr, dpr);
    gl.uniform1f(this.u.u_rscale, this.bubbleSize);
    gl.uniform1f(this.u.u_opacity, this.bubbleOpacity);
    this.discsDrawn = 0;
    let uploaded = -1;
    const lit = hot().disc;
    for (const s of list) {
      gl.uniform1i(this.u.u_hot, lit && lit.discs === s.data ? lit.i : -1);
      if (s.anchor.version !== uploaded) {
        uploaded = s.anchor.version;
        let m = matrices.get(uploaded);
        if (!m) matrices.set(uploaded, (m = mul4(m0, localToMercMatrix(s.anchor))));
        gl.uniformMatrix4fv(this.u.u_matrix, false, m);
      }
      gl.bindVertexArray(s.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
      this.discsDrawn += s.count;
    }
    gl.bindVertexArray(null);
    this.cut(dpr);
    if (timing) gl.endQuery(this.timerExt.TIME_ELAPSED_EXT);
    this.drawMs = performance.now() - t0;
    this.draws++;
  }

  /** Clear the discs over the selected stations (T-189), after drawing them. */
  private cut(dpr: number) {
    const shapes = this.cutouts?.() ?? [];
    if (!shapes.length) return;
    const gl = this.gl;
    if (!this.cutProg) {
      this.cutProg = compile(gl, CUT_VS, CUT_FS);
      for (const n of ["u_css", "u_c", "u_u", "u_n", "u_box", "u_size", "u_dot", "u_q", "u_outline", "u_glow"]) this.cutU[n] = gl.getUniformLocation(this.cutProg, n);
    }
    const u = this.cutU;
    gl.useProgram(this.cutProg);
    // what is drawn keeps (1 - alpha) of itself
    gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(u.u_css, this.canvas.width / dpr, this.canvas.height / dpr);
    gl.uniform1f(u.u_outline, SEL_OUTLINE_PX);
    gl.uniform1f(u.u_glow, SEL_GLOW_PX);
    const reach = SEL_OUTLINE_PX + SEL_GLOW_PX + 1;
    for (const s of shapes) {
      // the shape's own centre line (T-254); the quad round it, its ends carried on
      const q = s.dot > 0 ? new Float32Array(2 * PLATFORM_POINTS) : shapePoints(s);
      let lo = Infinity, hi = -Infinity, ax = 0;
      for (let i = 0; i < PLATFORM_POINTS; i++) {
        lo = Math.min(lo, q[i * 2 + 1]); hi = Math.max(hi, q[i * 2 + 1]);
        ax = Math.max(ax, Math.abs(q[i * 2]));
      }
      const r = s.halfWidth + reach + Math.max(0, s.halfLength - ax);
      const box: [number, number, number] = s.dot > 0
        ? [s.dot + reach, -s.dot - reach, s.dot + reach]
        : [ax + r, lo - r, hi + r];
      gl.uniform2fv(u.u_q, q);
      gl.uniform2f(u.u_c, s.x, s.y);
      gl.uniform2f(u.u_u, s.ux, s.uy);
      gl.uniform2f(u.u_n, s.nx, s.ny);
      gl.uniform3f(u.u_box, ...box);
      gl.uniform3f(u.u_size, s.halfLength, s.halfWidth, s.k);
      gl.uniform1f(u.u_dot, s.dot);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  private pollTimer() {
    const gl = this.gl, q = this.query;
    if (!q || !gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return;
    if (!gl.getParameter(this.timerExt.GPU_DISJOINT_EXT)) this.gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
    gl.deleteQuery(q);
    this.query = null;
  }
}
