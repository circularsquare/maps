// One-way marks (T-2991): along a loop that runs one way, small marks point the way its vehicles go.
// A bus or ferry route gets arrowheads in its colour sticking out on both sides (drawn by busLayer.ts);
// a train line a tiny white chevron on the line itself, only on the edges it runs once (a balloon's
// tail is run both ways). Marks are placed once per network change, one every STEP_M along the route;
// the shaders keep every 2^n-th so they stay at least GAP_PX apart on screen, and drop any too near a
// stop or station. Both are drawn in the passes the stroke cache keeps (renderer.ts), so a frame
// that redraws nothing draws none of them.

import type { Strokes } from "./network";
import { kAtY } from "./geo";
import { BAND_M_RANGE, BAND_RANGE, BAND_STRIDE } from "./lineBands";
import { compileOptional, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type LocalCamera } from "./gl";

/** Marks are placed every this many ground metres along a route; the shaders keep every 2^n-th. */
export const ONE_WAY_STEP_M = 100;
/** Bus and ferry arrows are this much further apart than train chevrons, step and gap alike (Anita, 2026-11-10: about 20% fewer). */
export const ARROW_SPREAD = 1.25;
/** The least gap between the marks drawn, CSS px. */
export const ONE_WAY_GAP_PX = 150;
/** A bus route's arrows show from this zoom, a train line's chevrons from `CHEVRON_ZOOM` while the
 * line is at least `CHEVRON_MIN_PX` wide there. */
export const ARROW_ZOOM = 13;
export const CHEVRON_ZOOM = 12;
const CHEVRON_MIN_PX = 4.5;

/** The share of marks kept at a zoom: every `oneWayEvery`-th, from device px per ground metre (`spread`: step and gap both that much longer). */
export function oneWayEvery(mPx: number, dpr: number, spread = 1): number {
  const n = Math.ceil(Math.log2((ONE_WAY_GAP_PX * spread * dpr) / Math.max(1e-9, ONE_WAY_STEP_M * spread * mPx)));
  return 2 ** Math.max(0, Math.min(20, n));
}

/** Per chevron: x, y, its direction (a unit vector in local units), its band's shift left (line
 * widths) and its ribbon's middle (m) as `Strokes.band` gives them, `1 - its width share`, the line,
 * its place in the line's marks, the ground metres to the line's nearest stop, its width share
 * (a tram's line is narrower). */
export const CHEVRON_STRIDE = 11;

/** A one-way train line's chevrons, from its strokes (`edgeUses`: how often its path runs each
 * edge; `stops`: its stops' places, local units; strokes with a `skipFlags` flag get none),
 * appended to `out`. */
export function trainChevrons(s: Strokes, line: number, edgeUses: ReadonlyMap<number, number>, stops: readonly (readonly [number, number])[], width: number, skipFlags: number, out: number[]) {
  if (!s.count) return;
  const k = kAtY(s.seg[1]), step = ONE_WAY_STEP_M * k;
  const B = s.band, S = 1 / 32767;
  const firstDist = s.dist[0];
  for (let i = 0; i < s.count; i++) {
    if (edgeUses.get(s.edge[i]) !== 1 || (s.flags[i] & skipFlags) !== 0) continue;
    const x0 = s.seg[i * 4], y0 = s.seg[i * 4 + 1], dx = s.seg[i * 4 + 2] - x0, dy = s.seg[i * 4 + 3] - y0;
    const len = Math.hypot(dx, dy), d0 = s.dist[i] - firstDist;
    if (!(len > 0)) continue;
    // marks at the middle of each step, so none sits on the line's first stop
    for (let m = Math.ceil(d0 / step - 0.5); (m + 0.5) * step < d0 + len; m++) {
      const t = ((m + 0.5) * step - d0) / len;
      if (t < 0) continue;
      const x = x0 + t * dx, y = y0 + t * dy;
      let near = Infinity;
      for (const p of stops) near = Math.min(near, Math.hypot(p[0] - x, p[1] - y));
      const o = i * BAND_STRIDE, lerp = (a: number, b: number) => a + t * (b - a);
      const shift = B ? lerp(B[o], B[o + 1]) * S * BAND_RANGE : 0, wShare = B ? lerp(B[o + 2], B[o + 3]) * S : 0, mid = B ? lerp(B[o + 4], B[o + 5]) * S * BAND_M_RANGE : 0;
      out.push(x, y, dx / len, dy / len, shift, mid, wShare, line, m, near / k, width);
    }
  }
}

/** One-way single track's chevrons (T-3094): one edge's strokes, which run a to b; `way` 2 points them
 * the other way. A mark takes the line field -1, no band shift and the whole width. */
export function oneWayTrackChevrons(s: Strokes, way: 1 | 2, stops: readonly (readonly [number, number])[], out: number[]) {
  if (!s.count) return;
  const k = kAtY(s.seg[1]), step = ONE_WAY_STEP_M * k, sign = way === 2 ? -1 : 1;
  for (let i = 0; i < s.count; i++) {
    const x0 = s.seg[i * 4], y0 = s.seg[i * 4 + 1], dx = s.seg[i * 4 + 2] - x0, dy = s.seg[i * 4 + 3] - y0;
    const len = Math.hypot(dx, dy), d0 = s.dist[i];
    if (!(len > 0)) continue;
    for (let m = Math.ceil(d0 / step - 0.5); (m + 0.5) * step < d0 + len; m++) {
      const t = ((m + 0.5) * step - d0) / len;
      if (t < 0) continue;
      const x = x0 + t * dx, y = y0 + t * dy;
      let near = Infinity;
      for (const p of stops) near = Math.min(near, Math.hypot(p[0] - x, p[1] - y));
      out.push(x, y, sign * dx / len, sign * dy / len, 0, 0, 0, -1, m, near / k, 1);
    }
  }
}

const VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform float u_line;        // a line's width, device px
uniform float u_band;        // how far lines sharing track are drawn side by side (0-1), the stroke shader's
uniform float u_band_m;      // device px a ground metre spans
uniform float u_every;       // every this many-th mark is drawn
uniform float u_clear;       // device px from a stop a mark keeps clear of
uniform float u_min;         // device px a line must be wide to carry one
uniform float u_dpr;
uniform int u_sel;           // the line kept at full strength while others are muted, -1 none
in vec4 a_mark;              // x, y, direction
in vec4 a_band;              // shift (line widths), ribbon middle (m), 1 - width share, line
in vec3 a_more;              // its place in its line's marks, metres to the nearest stop, width share
out vec2 v_p;
flat out float v_h;
flat out float v_t;
void main() {
  float w = u_line * (1.0 - u_band * a_band.z) * a_more.z;
  bool muted = u_sel >= 0 && int(a_band.w + 0.5) != u_sel;
  if (mod(a_more.x, u_every) > 0.5 || a_more.y * u_band_m < u_clear || w < u_min || muted) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec4 c0 = project(a_mark.xy);
  vec4 c1 = project(a_mark.xy + a_mark.zw);
  vec2 half_vp = u_viewport * 0.5;
  vec2 dir = c1.xy / c1.w * half_vp - c0.xy / c0.w * half_vp;
  dir = length(dir) > 1e-6 ? normalize(dir) : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  // the chevron: half as long as it is wide, across about half the line, its stroke about a pixel
  float h = 0.36 * w, t = max(1.4 * u_dpr, 0.19 * w);
  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  vec2 p = corner * (h + t + 1.0);
  vec2 off = nrm * u_band * (u_line * a_band.x + a_band.y * u_band_m) + dir * p.x + nrm * p.y;
  c0.xy += off / half_vp * c0.w;
  gl_Position = c0;
  v_p = p;
  v_h = h;
  v_t = t;
}`;

const FS = `#version 300 es
precision highp float;
uniform vec3 u_rgb;
in vec2 v_p;
flat in float v_h;
flat in float v_t;
out vec4 o;
float seg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
void main() {
  // the point ahead, the arms back at 45 degrees
  vec2 q = vec2(v_p.x, abs(v_p.y));
  float d = seg(q, vec2(0.5 * v_h, 0.0), vec2(-0.5 * v_h, v_h)) - 0.5 * v_t;
  float a = clamp(0.5 - d, 0.0, 1.0);
  if (a <= 0.0) discard;
  o = vec4(u_rgb * a, a);
}`;

export interface ChevronStyle {
  /** a line's width, device px, and how far lines sharing track are drawn apart (the stroke shader's `u_band`) */
  wLine: number;
  band: number;
  /** device px per ground metre (`u_band_m`) */
  mPx: number;
  /** the line kept at full strength while the others are muted (stop editing), -1 none */
  sel: number;
  /** the chevrons' colour, white where absent */
  rgb?: Float32List;
}

/** The train lines' chevrons, drawn by the renderer over the line strokes. */
export class OneWayChevrons {
  private gl: WebGL2RenderingContext | null = null;
  private prog: WebGLProgram | null = null;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject | null = null;
  private buf: WebGLBuffer | null = null;
  private count = 0;
  /** what the last network gave, kept for a new context */
  private data: Float32Array | null = null;

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = compileOptional(gl, VS, FS, "one-way chevrons");
    if (!this.prog) return;
    this.u = uniforms(gl, this.prog, [...PROJECT_UNIFORMS, "u_viewport", "u_line", "u_band", "u_band_m", "u_every", "u_clear", "u_min", "u_dpr", "u_sel", "u_rgb"]);
    this.vao = gl.createVertexArray();
    this.buf = null;
    this.count = 0;
    if (this.data) this.setNetwork(this.data);
  }

  /** The network's chevrons (`NetworkBuffers.oneWay`). */
  setNetwork(data: Float32Array | undefined) {
    this.data = data ?? null;
    const gl = this.gl;
    if (!gl || !this.prog) return;
    const n = data ? data.length / CHEVRON_STRIDE : 0;
    if (!n && !this.count) return;
    this.count = n;
    if (!n) return;
    if (!this.buf) this.buf = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.bufferData(gl.ARRAY_BUFFER, data!, gl.STATIC_DRAW);
    const F = 4, stride = CHEVRON_STRIDE * F;
    for (const [name, size, at] of [["a_mark", 4, 0], ["a_band", 4, 4], ["a_more", 3, 8]] as const) {
      const loc = gl.getAttribLocation(this.prog, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, at * F);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** chevrons in the network (checks read it) */
  get size(): number {
    return this.count;
  }

  draw(lc: LocalCamera, vp: readonly [number, number], zoom: number, dpr: number, s: ChevronStyle) {
    if (!this.count || !this.gl || !this.prog || zoom < CHEVRON_ZOOM) return;
    const gl = this.gl, u = this.u;
    gl.useProgram(this.prog);
    setCamera(gl, u, lc);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1f(u.u_line, s.wLine);
    gl.uniform1f(u.u_band, s.band);
    gl.uniform1f(u.u_band_m, s.mPx);
    gl.uniform1f(u.u_every, oneWayEvery(s.mPx, dpr));
    // clear of a station's box or dot and its rim
    gl.uniform1f(u.u_clear, 14 * dpr);
    gl.uniform1f(u.u_min, CHEVRON_MIN_PX * dpr);
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform1i(u.u_sel, s.sel);
    gl.uniform3fv(u.u_rgb, s.rgb ?? [1, 1, 1]);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
  }
}
