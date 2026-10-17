// Marks on the map for meeting and climbing track (T-629, T-655), drawn by the network renderer in
// one instanced pass after the track and before stations and trains:
//   - a flat crossing (an X) and a flat junction (a fork) in a round badge. Big on blueprint track
//     and while drawing (the preview), in the blueprint blue; small and quiet on built track, in
//     the map ink. They grow with zoom only part way, by far less than the track's width does.
//   - a ramp (a wedge) where track climbs from level 0 and a tunnel portal (a half disc) where it
//     drops from level 0, at the level 0 end, along the track, in the level's Elevation colour.
// The data is `NetworkBuffers.marks` (map/markData.ts); the hover tag names a mark (`markTip`).

import { levelLabel } from "../game/format";
import { rgb } from "../game/palette";
import { perfOff } from "../perfFlags";
import { shadeColours } from "../game/shades";
import type { Overlay } from "./overlay";
import { M_BAD, M_CROSSING, M_JUNCTION, M_PORTAL, M_RAMP, MARK_STRIDE } from "./markData";
import type { TipLine } from "./networkTips";
import { elevationFact } from "./trackTagRows";

/** ?perfOff=marks (measurement only, src/perfFlags.ts) */
const NO_MARKS = perfOff("marks");

/** Badge radius by zoom, CSS px: blueprint and drawing, then built (the track is 0.85 px wide at
 * zoom 9 and 6.6 at 15, a badge 2 to 3 times larger over that range). */
const LOUD_R: [number, number][] = [[9, 7], [12, 11], [15, 15]];
const QUIET_R: [number, number][] = [[11, 5], [13, 6.5], [15, 8.5]];
const RAMP_R: [number, number][] = [[11, 3.5], [13, 5], [15, 7]];
/** Below these zooms the marks are not drawn, so a region is not covered in them. */
const LOUD_ZOOM = 9, QUIET_ZOOM = 11, RAMP_ZOOM = 11;
/** How opaque a built mark is. */
const QUIET_ALPHA = 0.85;
/** A preview's crossing or junction this close (local units) to one already marked is not drawn. */
const SAME_PLACE = 6;

function interp(z: number, stops: [number, number][]): number {
  if (z <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [z1, v1] = stops[i];
    if (z <= z1) {
      const [z0, v0] = stops[i - 1];
      return v0 + ((v1 - v0) * (z - z0)) / (z1 - z0);
    }
  }
  return stops[stops.length - 1][1];
}

/** The radii (CSS px) of the loud badge, the quiet badge and the ramp mark at a zoom; 0 = not drawn. */
export function markRadii(zoom: number): [number, number, number] {
  return [zoom >= LOUD_ZOOM ? interp(zoom, LOUD_R) : 0, zoom >= QUIET_ZOOM ? interp(zoom, QUIET_R) : 0, zoom >= RAMP_ZOOM ? interp(zoom, RAMP_R) : 0];
}

const VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform vec3 u_r;            // radius in device px: loud badge, quiet badge, ramp mark
in vec4 a_mark;              // x, y, shape, angle
in vec2 a_style;             // tier (0 built, 1 blueprint), level
out vec2 v_p;
flat out vec4 v_m;
flat out vec2 v_s;
flat out float v_r;
void main() {
  int shape = int(a_mark.z + 0.5);
  bool ramp = shape == ${M_RAMP} || shape == ${M_PORTAL};
  float r = ramp ? u_r.z : (a_style.x > 0.5 ? u_r.x : u_r.y);
  if (r <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float rq = ramp ? 1.8 * r + 3.0 : r + 3.0;
  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  vec4 c = u_matrix * vec4(a_mark.xy, 0.0, 1.0);
  vec2 p = corner * rq;
  c.xy += vec2(p.x, -p.y) / (u_viewport * 0.5) * c.w;
  gl_Position = c;
  v_p = p;
  v_m = a_mark;
  v_s = a_style;
  v_r = r;
}`;

const FS = `#version 300 es
precision highp float;
uniform vec3 u_ink;          // built marks
uniform vec3 u_bp;           // blueprint marks
uniform vec3 u_fill;         // the badge, and the halo round a ramp mark
uniform vec3 u_bad;
uniform float u_dpr;
uniform float u_quiet;
uniform vec3 u_level_colours[7];
in vec2 v_p;
flat in vec4 v_m;
flat in vec2 v_s;
flat in float v_r;
out vec4 o;
float cov(float d) { return clamp(0.5 - d, 0.0, 1.0); }
float seg(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a, ba = b - a;
  return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}
float tri(vec2 p, vec2 a, vec2 b, vec2 c) {
  vec2 e0 = b - a, e1 = c - b, e2 = a - c, v0 = p - a, v1 = p - b, v2 = p - c;
  vec2 q0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 q1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 q2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  float s = sign(e0.x * e2.y - e0.y * e2.x);
  vec2 d = min(min(vec2(dot(q0, q0), s * (v0.x * e0.y - v0.y * e0.x)), vec2(dot(q1, q1), s * (v1.x * e1.y - v1.y * e1.x))), vec2(dot(q2, q2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}
void main() {
  int shape = int(v_m.z + 0.5);
  bool blueprint = v_s.x > 0.5;
  float r = v_r;
  vec3 ink = blueprint ? u_bp : u_ink;
  if (shape == ${M_RAMP} || shape == ${M_PORTAL}) {
    // along the track: u along the ramp from its foot, v across it
    vec2 dir = vec2(cos(v_m.w), sin(v_m.w));
    vec2 q = vec2(dot(v_p, dir), dot(v_p, vec2(-dir.y, dir.x)));
    float d = shape == ${M_RAMP} ? tri(q, vec2(0.0, -r), vec2(0.0, r), vec2(1.3 * r, 0.0)) : max(length(q) - r, -q.x);
    float rim = max(1.0, 1.1 * u_dpr);
    float halo = 1.4 * u_dpr;
    vec3 lc = u_level_colours[clamp(3 - int(round(v_s.y)), 0, 6)];
    float body = cov(d), edge = cov(abs(d + rim * 0.5) - rim * 0.5), outer = cov(d - halo);
    vec3 col = mix(u_fill, mix(lc, ink, edge), body);
    float a = outer * (blueprint ? 1.0 : u_quiet);
    o = vec4(col * a, a);
    return;
  }
  float d = length(v_p) - r;
  float ring = cov(abs(d + (blueprint ? 1.5 : 1.0) * u_dpr * 0.5) - (blueprint ? 1.5 : 1.0) * u_dpr * 0.5);
  float g = max(0.8 * u_dpr, 0.1 * r);
  float k = 0.4 * r;
  float glyph;
  if (shape == ${M_JUNCTION}) {
    // a fork: a stem below, two arms up
    vec2 hub = vec2(0.0, 0.1 * r);
    glyph = cov(min(seg(v_p, vec2(0.0, 0.55 * r), hub), min(seg(v_p, hub, vec2(-0.42 * r, -0.55 * r)), seg(v_p, hub, vec2(0.42 * r, -0.55 * r)))) - g);
  } else {
    glyph = cov(min(seg(v_p, vec2(-k, -k), vec2(k, k)), seg(v_p, vec2(-k, k), vec2(k, -k))) - g);
  }
  if (shape == ${M_BAD}) ink = u_bad;
  vec3 col = mix(u_fill, ink, max(ring, glyph));
  float a = cov(d) * (blueprint ? 1.0 : u_quiet);
  o = vec4(col * a, a);
}`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("marks shader: " + gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("marks link: " + gl.getProgramInfoLog(prog));
  return prog;
}

interface Instances {
  vao: WebGLVertexArrayObject;
  buf: WebGLBuffer;
  count: number;
}

const rgbf = (hex: string) => rgb(hex).map((v) => v / 255);

export class GradeMarks {
  private gl!: WebGL2RenderingContext;
  private prog!: WebGLProgram;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private built: Instances | null = null;
  private drawing: Instances | null = null;
  /** the built network's marks, for hit tests */
  marks: Float32Array = new Float32Array(0);

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = compile(gl, VS, FS);
    for (const n of ["u_matrix", "u_viewport", "u_r", "u_ink", "u_bp", "u_fill", "u_bad", "u_dpr", "u_quiet", "u_level_colours"]) this.u[n] = gl.getUniformLocation(this.prog, n);
  }

  private upload(old: Instances | null, data: Float32Array | undefined): Instances | null {
    const gl = this.gl;
    if (old) {
      gl.deleteVertexArray(old.vao);
      gl.deleteBuffer(old.buf);
    }
    if (!data?.length) return null;
    const vao = gl.createVertexArray()!;
    const buf = gl.createBuffer()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const bytes = MARK_STRIDE * 4;
    for (const [name, size, offset] of [["a_mark", 4, 0], ["a_style", 2, 16]] as const) {
      const loc = gl.getAttribLocation(this.prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, bytes, offset);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return { vao, buf, count: data.length / MARK_STRIDE };
  }

  setNetwork(marks: Float32Array | undefined) {
    this.marks = marks ?? new Float32Array(0);
    this.built = this.upload(this.built, marks);
  }

  /** The drawing tool's marks, or none. A crossing or junction where one is already marked is left out. */
  setPreview(marks: Float32Array | undefined) {
    let keep = marks;
    if (marks && this.marks.length) {
      const out: number[] = [];
      for (let i = 0; i < marks.length; i += MARK_STRIDE) {
        const shape = marks[i + 2];
        let dup = false;
        if (shape === M_CROSSING || shape === M_JUNCTION) {
          for (let j = 0; j < this.marks.length && !dup; j += MARK_STRIDE) dup = Math.abs(this.marks[j] - marks[i]) < SAME_PLACE && Math.abs(this.marks[j + 1] - marks[i + 1]) < SAME_PLACE;
        }
        if (!dup) for (let k = 0; k < MARK_STRIDE; k++) out.push(marks[i + k]);
      }
      keep = new Float32Array(out);
    }
    this.drawing = this.upload(this.drawing, keep);
  }

  draw(m: Float32List, vp: readonly [number, number], zoom: number, dpr: number, levelColours: Float32Array) {
    if ((!this.built && !this.drawing) || NO_MARKS) return;
    const [loud, quiet, ramp] = markRadii(zoom);
    if (!loud && !quiet && !ramp) return;
    const gl = this.gl, u = this.u, shades = shadeColours();
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(u.u_matrix, false, m);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform3f(u.u_r, loud * dpr, quiet * dpr, ramp * dpr);
    gl.uniform3fv(u.u_ink, rgbf(shades.markInk));
    gl.uniform3fv(u.u_bp, rgbf(shades.markBlueprint));
    gl.uniform3fv(u.u_fill, rgbf(shades.casing));
    gl.uniform3fv(u.u_bad, rgbf("#d23c3c"));
    gl.uniform1f(u.u_dpr, dpr);
    gl.uniform1f(u.u_quiet, QUIET_ALPHA);
    gl.uniform3fv(u.u_level_colours, levelColours);
    for (const s of [this.built, this.drawing]) {
      if (!s) continue;
      gl.bindVertexArray(s.vao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
    }
    gl.bindVertexArray(null);
  }
}

/** The mark of the built network under a screen point (CSS px in the map), as an index into
 * `marks`, or -1. The nearest of those the point is within. */
export function markAt(ov: Overlay, marks: Float32Array, px: number, py: number, zoom: number): number {
  const [loud, quiet, ramp] = markRadii(zoom);
  let best = -1, bestD = Infinity;
  const at = new Float64Array(2);
  for (let i = 0; i < marks.length; i += MARK_STRIDE) {
    const shape = marks[i + 2], isRamp = shape === M_RAMP || shape === M_PORTAL;
    const r = isRamp ? ramp : marks[i + 4] > 0.5 ? loud : quiet;
    if (r <= 0) continue;
    ov.toScreenInto(marks[i], marks[i + 1], at);
    const d = Math.hypot(at[0] - px, at[1] - py);
    if (d <= Math.max(r, 5) + 1 && d < bestD) [best, bestD] = [i, d];
  }
  return best;
}

/** The hover tag for the mark at `i` of `marks`. */
export function markTip(marks: Float32Array, i: number): TipLine[] {
  const shape = marks[i + 2], level = marks[i + 5], blueprint = marks[i + 4] > 0.5;
  const out: TipLine[] = [];
  if (shape === M_RAMP) out.push(`Ramp up to ${levelLabel(level)}`);
  else if (shape === M_PORTAL) out.push(`Tunnel portal down to ${levelLabel(level)}`);
  else if (shape === M_BAD) out.push("Tracks cross less than one elevation step apart");
  else {
    out.push(shape === M_JUNCTION ? "Flat junction" : "Flat crossing");
    out.push(elevationFact(level));
  }
  if (blueprint && shape !== M_BAD) out.push("Blueprint");
  return out;
}
