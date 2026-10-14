// The demand views' map layer (T-084, T-097, SPEC 8): commuter bubbles, a bubble selection's far
// end and a selected station's catchment, as discs on a canvas of their own over the network
// (Anita: in a demand view the demand is what you want to see, on top). Nothing here moves with
// the clock, so the canvas is drawn only on MapLibre's frames (the camera moved:
// `Overlay.afterMapFrame`, same frame and matrix as the basemap, T-015) and once after its data
// changes. While the game plays with the camera still it is never touched.

import type { Map as MlMap } from "maplibre-gl";
import { NET_INK, rgb } from "../game/palette";
import { DISC } from "../workers/commutersProtocol";
import { EARTH_C, HOME_ANCHOR, localToMercMatrix, mul4 } from "./geo";
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

// A disc per instance: radius in local units, colour and alpha, and an edge (0 none, 1 a white
// edge of 1 px, 2 a dark outline of 2 px). No minimum size: a tiny count is a tiny disc.
const DISC_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_px_per_unit;
uniform float u_rscale;   // the player's bubble size (T-099); 1 for a station's discs
in vec3 a_geo;    // x, y, radius (local units)
in vec4 a_col;    // r, g, b, a
in float a_edge;
out vec2 v_p;
out float v_r;
out float v_edge;
out vec4 v_col;
void main() {
  float r = a_geo.z * u_rscale * u_px_per_unit;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  vec4 c = u_matrix * vec4(a_geo.xy, 0.0, 1.0);
  vec2 off = corner * (r + 2.0);
  c.xy += off / (u_viewport * 0.5) * c.w;
  gl_Position = c;
  v_p = off;
  v_r = r;
  v_edge = a_edge;
  v_col = a_col;
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
  } else if (v_edge > 0.5 && v_r > 3.0 * u_dpr) {
    float t = clamp(d - (v_r - u_dpr) + 0.5, 0.0, 1.0);
    col = mix(col, vec3(1.0), t);
  }
  a *= cover;
  o = vec4(col * a, a);
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
export type DiscLayer = "inter" | "far" | "near" | "stationFar" | "stationNear";
// "inter": a selection's or a station's longer trips (T-118), under the commuters' discs
const LAYERS: DiscLayer[] = ["inter", "far", "near", "stationFar", "stationNear"];

interface DiscSet {
  vao: WebGLVertexArrayObject;
  buf: WebGLBuffer;
  count: number;
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

  constructor(private map: MlMap, private overlay: Overlay) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "overlay demand";
    // over the network's canvas, under the station names and markers
    overlay.canvas.after(this.canvas);
    const gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, alpha: true });
    if (!gl) throw new Error("no WebGL2");
    this.gl = gl;
    this.prog = compile(gl, DISC_VS, DISC_FS);
    for (const n of ["u_matrix", "u_viewport", "u_px_per_unit", "u_ink", "u_dpr", "u_rscale"]) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.size();
    map.on("resize", () => {
      this.size();
      this.invalidate();
    });
    overlay.afterMapFrame.push(() => this.draw());
  }

  private size() {
    const dpr = this.map.getPixelRatio();
    const c = this.map.getCanvas();
    this.canvas.width = Math.round(c.clientWidth * dpr);
    this.canvas.height = Math.round(c.clientHeight * dpr);
    this.canvas.style.width = c.clientWidth + "px";
    this.canvas.style.height = c.clientHeight + "px";
  }

  private makeSet(data: Float32Array): DiscSet {
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
    return { vao, buf, count: data.length / DISC };
  }

  private dropSet(s: DiscSet | null | undefined) {
    if (!s) return;
    this.gl.deleteBuffer(s.buf);
    this.gl.deleteVertexArray(s.vao);
  }

  /** The city's bubbles, one disc set per aggregation level; null for none. */
  setBubbles(levels: Float32Array[] | null) {
    const t0 = performance.now();
    for (const s of this.bubbles) this.dropSet(s);
    this.bubbles = (levels ?? []).map((d) => (d.length ? this.makeSet(d) : null));
    this.uploadMs = performance.now() - t0;
    this.invalidate();
  }

  /** A nearby zone's bubbles (T-276), drawn under the focused zone's and shown with them; null
   * removes the zone's. */
  private nearby = new Map<string, (DiscSet | null)[]>();
  setNearbyBubbles(zone: string, levels: Float32Array[] | null) {
    for (const s of this.nearby.get(zone) ?? []) this.dropSet(s);
    this.nearby.delete(zone);
    if (levels) this.nearby.set(zone, levels.map((d) => (d.length ? this.makeSet(d) : null)));
    this.invalidate();
  }

  /** Commuter bubbles' radius as a multiple of the built one (the settings slider, T-099). */
  bubbleSize = 1;
  setBubbleSize(s: number) {
    if (s === this.bubbleSize) return;
    this.bubbleSize = s;
    this.invalidate();
  }

  /** Show or hide the city's bubbles (kept on the GPU while hidden). */
  setBubblesVisible(v: boolean) {
    if (v === this.showBubbles) return;
    this.showBubbles = v;
    this.invalidate();
  }

  /** One of the small disc layers (a selection's ends, a station's catchment); null clears it. */
  setDiscs(layer: DiscLayer, data: Float32Array | null) {
    this.dropSet(this.sets.get(layer));
    this.sets.delete(layer);
    if (data && data.length) this.sets.set(layer, this.makeSet(data));
    this.invalidate();
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
    // each set with its radius scale: the player's bubble size for commuter bubbles, 1 for a
    // station's discs
    const list: [DiscSet, number][] = [];
    if (this.showBubbles) for (const sets of this.nearby.values()) {
      const n = sets[level];
      if (n) list.push([n, this.bubbleSize]);
    }
    const b = this.showBubbles ? this.bubbles[level] : null;
    if (b) list.push([b, this.bubbleSize]);
    for (const l of LAYERS) {
      const s = this.sets.get(l);
      if (s) list.push([s, l === "far" || l === "near" || l === "inter" ? this.bubbleSize : 1]);
    }
    if (!list.length) {
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
    gl.useProgram(this.prog);
    // the demand views' discs stay on the home anchor for now (T-307, map/geo.ts)
    gl.uniformMatrix4fv(this.u.u_matrix, false, mul4(m0, localToMercMatrix(HOME_ANCHOR)));
    gl.uniform2f(this.u.u_viewport, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.u.u_px_per_unit, ((512 * Math.pow(2, zoom)) / EARTH_C) * dpr);
    gl.uniform3fv(this.u.u_ink, rgb(NET_INK).map((v) => v / 255));
    gl.uniform1f(this.u.u_dpr, dpr);
    this.discsDrawn = 0;
    for (const [s, rs] of list) {
      gl.uniform1f(this.u.u_rscale, rs);
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
