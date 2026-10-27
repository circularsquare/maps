// The complexes' car parks (T-1528, notes/T-1528.md): a lot beside the complex, placed by
// map/facilityPlace.ts, drawn by the network renderer under everything else it draws from
// STATION_BOX_ZOOM in. True to size on the ground: rows of 2.5 by 5 m spaces either side of a 6 m
// aisle, a bay to a station with park and ride (at most three), on a surface a little off the land's
// colour. The spaces' lines fade in once they are a few pixels apart (about zoom 16), so further out
// a lot is a faint outlined strip.

import { rgbf } from "../game/palette";
import { shadeColours } from "../game/shades";
import { compile, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type LocalCamera } from "./gl";
import { AISLE, BAY_M, LOT_BLUEPRINT, LOT_OFF, LOT_STRIDE, SPACE_L, SPACE_W } from "./facilityPlace";
import type { Project } from "./stationGeometry";

const VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
in vec4 a_lot;     // centre (local), half its length along it as a vector (local)
in vec4 a_dim;     // half length m, half depth m, bays, flags
out vec2 v_p;      // in the lot's frame, px
flat out vec4 v_dim; // half length m, half depth m, bays, px a metre
flat out float v_alpha;
void main() {
  vec2 half_vp = u_viewport * 0.5;
  vec4 c = project(a_lot.xy);
  vec4 e = project(a_lot.xy + a_lot.zw);
  vec4 q = project(a_lot.xy + vec2(-a_lot.w, a_lot.z));
  vec2 cs = c.xy / c.w * half_vp;
  vec2 axis = e.xy / e.w * half_vp - cs, nrm = q.xy / q.w * half_vp - cs;
  float len = length(axis);
  if (len < 1e-6) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 along = axis / len, across = length(nrm) > 1e-6 ? normalize(nrm) : vec2(-along.y, along.x);
  float k = len / a_dim.x;
  int flags = int(a_dim.w + 0.5);
  v_alpha = (flags & ${LOT_OFF}) != 0 ? 0.3 : (flags & ${LOT_BLUEPRINT}) != 0 ? 0.6 : 1.0;
  v_dim = vec4(a_dim.xyz, k);
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  v_p = corner * (a_dim.xy * k + 1.0);
  c.xy += (along * v_p.x + across * v_p.y) / half_vp * c.w;
  gl_Position = c;
}`;

const FS = `#version 300 es
precision highp float;
uniform vec3 u_fill;
uniform vec3 u_line;
uniform float u_dpr;
in vec2 v_p;
flat in vec4 v_dim;
flat in float v_alpha;
out vec4 o;
// how far a point is from the nearest of lines every \`step\` m, px
float grid(float m, float step, float k) { return abs(fract(m / step + 0.5) - 0.5) * step * k; }
void main() {
  float k = v_dim.w, hl = v_dim.x, hd = v_dim.y;
  float edge = max(abs(v_p.x) - hl * k, abs(v_p.y) - hd * k);
  float cover = clamp(0.5 - edge, 0.0, 1.0);
  if (cover <= 0.0) discard;
  float lw = max(0.75, 0.6 * u_dpr);
  float ink = clamp(lw + 0.5 + edge, 0.0, 1.0);
  vec2 m = v_p / k + vec2(hl, hd);
  // a bay: a row of spaces, the aisle, a row; spaces' side lines across each row
  float t = mod(m.y, ${BAY_M.toFixed(1)});
  float inRow = t < ${SPACE_L.toFixed(1)} || t > ${(SPACE_L + AISLE).toFixed(1)} ? 1.0 : 0.0;
  float spaces = smoothstep(2.5 * u_dpr, 4.5 * u_dpr, ${SPACE_W.toFixed(1)} * k) * inRow * clamp(0.5 * lw + 0.5 - grid(m.x, ${SPACE_W.toFixed(1)}, k), 0.0, 1.0);
  // the line where two bays' rows meet back to back
  float heads = m.y > 1.0 && m.y < 2.0 * hd - 1.0 ? smoothstep(4.0 * u_dpr, 8.0 * u_dpr, ${BAY_M.toFixed(1)} * k) * clamp(0.5 * lw + 0.5 - grid(m.y, ${BAY_M.toFixed(1)}, k), 0.0, 1.0) : 0.0;
  vec3 col = mix(u_fill, u_line, max(ink, max(spaces, heads)));
  float a = cover * v_alpha;
  o = vec4(col * a, a);
}`;

export class ParkingLots {
  private gl!: WebGL2RenderingContext;
  private prog!: WebGLProgram;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject | null = null;
  private buf: WebGLBuffer | null = null;
  private count = 0;
  /** the network's lots, for hit tests */
  lots: Float32Array = new Float32Array(0);

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    // again after a lost context (T-953): the old buffers are gone, the renderer uploads anew
    this.vao = this.buf = null;
    this.count = 0;
    this.prog = compile(gl, VS, FS);
    this.u = uniforms(gl, this.prog, [...PROJECT_UNIFORMS, "u_viewport", "u_fill", "u_line", "u_dpr"]);
  }

  setNetwork(lots: Float32Array | undefined) {
    const gl = this.gl;
    this.lots = lots ?? new Float32Array(0);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.buf) gl.deleteBuffer(this.buf);
    this.vao = this.buf = null;
    this.count = this.lots.length / LOT_STRIDE;
    if (!this.count) return;
    this.vao = gl.createVertexArray()!;
    this.buf = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, this.lots, gl.STATIC_DRAW);
    for (const [name, offset] of [["a_lot", 0], ["a_dim", 16]] as const) {
      const loc = gl.getAttribLocation(this.prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, LOT_STRIDE * 4, offset);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  draw(lc: LocalCamera, vp: readonly [number, number], dpr: number) {
    if (!this.count || !this.vao) return;
    const gl = this.gl, u = this.u, sh = shadeColours();
    gl.useProgram(this.prog);
    setCamera(gl, u, lc);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform3fv(u.u_fill, rgbf(sh.lot));
    gl.uniform3fv(u.u_line, rgbf(sh.lotLine));
    gl.uniform1f(u.u_dpr, dpr);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
  }

  /** The lot drawn under a screen point (CSS px), its index (LOT_STRIDE numbers each), or -1. */
  at(px: number, py: number, project: Project): number {
    for (let i = this.count - 1; i >= 0; i--) {
      const o = i * LOT_STRIDE, l = this.lots;
      const [cx, cy] = project(l[o], l[o + 1]), [ex, ey] = project(l[o] + l[o + 2], l[o + 1] + l[o + 3]);
      const [nx, ny] = project(l[o] - l[o + 3] * l[o + 5] / l[o + 4], l[o + 1] + l[o + 2] * l[o + 5] / l[o + 4]);
      const ax = ex - cx, ay = ey - cy, bx = nx - cx, by = ny - cy;
      const al = Math.hypot(ax, ay), bl = Math.hypot(bx, by);
      if (al < 1e-6 || bl < 1e-6) continue;
      const u = ((px - cx) * ax + (py - cy) * ay) / al, v = ((px - cx) * bx + (py - cy) * by) / bl;
      if (Math.abs(u) <= al + 1 && Math.abs(v) <= bl + 1) return i;
    }
    return -1;
  }
}
