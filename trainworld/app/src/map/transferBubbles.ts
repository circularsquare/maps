// Riders changing trains, drawn (T-1124): when a train stops, those of its riders who change to a
// line at another station walk there as one grey bubble, straight from the platform they left to
// the one they go to, over demand's walk between the two (the boarding account's pulses, T-554),
// and join that platform's crowd as they reach it. A change at the same station walks nothing and
// joins at once. One instanced pass after the stations, built once a game hour from the account's
// slice; each frame only sets the time, and the vertex shader puts each bubble where it is then or
// off the screen.

import { QUEUE_STRIDE, type BoardHour } from "../game/boarding";
import type { NetworkBuffers } from "./network";

/** Floats per bubble: from x, y, to x, y; set off, arrival (s from the hour's epoch), riders, line. */
const STRIDE = 8;
/** A bubble of this many riders is as wide as a platform box's white; the radius goes with the
 * square root of the riders, between these shares of that width's half. */
export const BUBBLE_RIDERS = 250;
const MIN_SHARE = 0.35, MAX_SHARE = 1.3;
/** Walks shorter than this (s) are not drawn. */
const MIN_WALK_S = 1;

const VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_time;
uniform float u_r;           // a box's white half width, device px
in vec4 a_path;              // from, to (local units)
in vec4 a_when;              // set off, arrival, riders, (line)
out vec2 v_p;
flat out float v_r;
void main() {
  float span = a_when.y - a_when.x;
  if (!(u_time >= a_when.x && u_time < a_when.y) || span <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 at = mix(a_path.xy, a_path.zw, (u_time - a_when.x) / span);
  v_r = u_r * clamp(sqrt(a_when.z / ${BUBBLE_RIDERS.toFixed(1)}), ${MIN_SHARE.toFixed(2)}, ${MAX_SHARE.toFixed(2)});
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  v_p = corner * (v_r + 1.5);
  vec4 c = u_matrix * vec4(at, 0.0, 1.0);
  c.xy += v_p / (u_viewport * 0.5) * c.w;
  gl_Position = c;
}`;

const FS = `#version 300 es
precision highp float;
uniform vec3 u_fill;
uniform vec3 u_rim;
uniform float u_rim_px;
in vec2 v_p;
flat in float v_r;
out vec4 o;
void main() {
  float d = length(v_p);
  float a = clamp(v_r + u_rim_px - d + 0.5, 0.0, 1.0);
  if (a <= 0.0) discard;
  vec3 c = mix(u_rim, u_fill, clamp(v_r - d + 0.5, 0.0, 1.0));
  o = vec4(c * a, a);
}`;

export class TransferBubbles {
  private gl!: WebGL2RenderingContext;
  private prog: WebGLProgram | null = null;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private vao: WebGLVertexArrayObject | null = null;
  private pathBuf: WebGLBuffer | null = null;
  private whenBuf: WebGLBuffer | null = null;
  private count = 0;
  /** the bubbles as built, STRIDE floats each, for the hover */
  private data = new Float32Array(0);
  private src: { board: BoardHour | null; net: NetworkBuffers | null } = { board: null, net: null };
  /** ms the last build took (`?debug=1`) */
  buildMs = 0;

  init(gl: WebGL2RenderingContext, compile: (vs: string, fs: string) => WebGLProgram) {
    this.gl = gl;
    this.prog = compile(VS, FS);
    for (const n of ["u_matrix", "u_viewport", "u_time", "u_r", "u_fill", "u_rim", "u_rim_px"]) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.vao = gl.createVertexArray();
    this.pathBuf = gl.createBuffer();
    this.whenBuf = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    for (const [name, buf] of [["a_path", this.pathBuf], ["a_when", this.whenBuf]] as const) {
      const loc = gl.getAttribLocation(this.prog, name);
      if (loc < 0) continue;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(4), gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** The bubbles of `board`'s hour on `net` (each station's platform middle to the other's), when
   * either changed. `station[q]` is queue q's station in the station buffer (-1 for none). */
  sync(board: BoardHour | null, net: NetworkBuffers | null, station: Int32Array | null) {
    if (board === this.src.board && net === this.src.net) return;
    this.src = { board, net };
    const t0 = performance.now();
    const out: number[] = [];
    if (board && net && station && board.pulseFrom.length === board.pulses.length) {
      const index = new Map<number, number>();
      net.stationIds.forEach((id, i) => index.set(Number(id), i));
      const S = net.stations, sh = net.stationShape;
      const mx = (i: number) => S[i * 3] + (sh?.[i * 4] ?? 0), my = (i: number) => S[i * 3 + 1] + (sh?.[i * 4 + 1] ?? 0);
      const P = board.pulses, F = board.pulseFrom, Q = board.queues;
      for (let q = 0; q < board.queueCount; q++) {
        const to = station[q];
        if (to < 0) continue;
        const base = q * QUEUE_STRIDE, first = Q[base + 10], end = first + Q[base + 11];
        for (let p = first; p < end; p++) {
          const walk = F[p * 2 + 1];
          if (!(walk >= MIN_WALK_S) || !(P[p * 2 + 1] >= 1)) continue;
          const from = index.get(F[p * 2]);
          if (from === undefined || from === to) continue;
          out.push(mx(from), my(from), mx(to), my(to), P[p * 2] - walk, P[p * 2], P[p * 2 + 1], Q[base + 1]);
        }
      }
    }
    this.data = Float32Array.from(out);
    this.count = out.length / STRIDE;
    const gl = this.gl;
    if (gl && this.pathBuf && this.whenBuf) {
      const path = new Float32Array(Math.max(1, this.count) * 4), when = new Float32Array(Math.max(1, this.count) * 4);
      for (let i = 0; i < this.count; i++) {
        path.set(this.data.subarray(i * STRIDE, i * STRIDE + 4), i * 4);
        when.set(this.data.subarray(i * STRIDE + 4, i * STRIDE + 8), i * 4);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.pathBuf);
      gl.bufferData(gl.ARRAY_BUFFER, path, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.whenBuf);
      gl.bufferData(gl.ARRAY_BUFFER, when, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }
    this.buildMs = performance.now() - t0;
  }

  /** How many bubbles the hour has. */
  get size() {
    return this.count;
  }

  /** Draw the bubbles walking at `t` (s from the hour's epoch); `r` a box's white half width and
   * `rim` the casing's width, device px. */
  draw(m: ArrayLike<number>, vp: readonly [number, number], t: number, r: number, fill: number[], rim: number[], rimPx: number) {
    if (!this.count || !this.prog) return;
    const gl = this.gl, u = this.u;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(u.u_matrix, false, m as Float32List);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1f(u.u_time, t);
    gl.uniform1f(u.u_r, r);
    gl.uniform3fv(u.u_fill, fill);
    gl.uniform3fv(u.u_rim, rim);
    gl.uniform1f(u.u_rim_px, rimPx);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
    this.last = { m: Array.from(m), t, r, w: vp[0], h: vp[1] };
  }
  private last: { m: number[]; t: number; r: number; w: number; h: number } | null = null;

  /** Not drawn this frame: nothing to hover. */
  hide() {
    this.last = null;
  }

  /** The bubble under (`x`, `y`), shares of the canvas from its top left, as last drawn: its
   * riders and the line they change to; null for none. Only on hover, never per frame. */
  at(fx: number, fy: number): { riders: number; line: number } | null {
    const L = this.last, d = this.data;
    if (!L || !this.count) return null;
    const m = L.m, x = fx * L.w, y = fy * L.h;
    let best: { riders: number; line: number } | null = null, bestD = Infinity;
    for (let i = 0; i < this.count; i++) {
      const o = i * STRIDE, s0 = d[o + 4], s1 = d[o + 5];
      if (!(L.t >= s0 && L.t < s1)) continue;
      const f = (L.t - s0) / (s1 - s0);
      const px = d[o] + (d[o + 2] - d[o]) * f, py = d[o + 1] + (d[o + 3] - d[o + 1]) * f;
      const cx = m[0] * px + m[4] * py + m[12], cy = m[1] * px + m[5] * py + m[13], cw = m[3] * px + m[7] * py + m[15];
      const sx = ((cx / cw + 1) / 2) * L.w, sy = ((1 - cy / cw) / 2) * L.h;
      const r = L.r * Math.max(MIN_SHARE, Math.min(MAX_SHARE, Math.sqrt(d[o + 6] / BUBBLE_RIDERS))) + 2;
      const dd = Math.hypot(sx - x, sy - y);
      if (dd <= r && dd < bestD) {
        bestD = dd;
        best = { riders: d[o + 6], line: d[o + 7] };
      }
    }
    return best;
  }
}
