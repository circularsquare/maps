// WebGL2 network renderer: track and line strokes (one instanced quad per polyline segment, white
// casing under the colour; blueprint dashed and faded), station dots, the drawing tool's preview,
// and instanced trains positioned on the GPU from keyframes (shared phase tables, notes/T-040.md)
// and a time uniform. Per frame the CPU sets uniforms and issues a handful of draw calls, whatever
// the train count (notes/T-002.md).
//
// Time precision (T-016, SPEC 7): trip departures are relative to a render epoch at the start of
// the current game hour, so the float32 time uniform stays under 3600 s. The clock worker sends
// each hour's trips (and the next hour's, ahead); when the hour turns, only that buffer is
// re-uploaded. Phase tables change only on edits.

import { LEVEL_COLOURS, NET_INK, rgb } from "../game/palette";
import { LOCAL_TO_MERC, MERC_K, EARTH_C, mul4 } from "./geo";
import type { NetworkBuffers, Strokes } from "./network";

const TEX_W = 2048;
const LINES_W = 1024;
/** The render epoch moves to the start of each game hour. */
export const EPOCH_S = 3600;
const TRACK_RGB = rgb("#77746f").map((v) => v / 255);
const BAD_RGB = rgb("#d23c3c").map((v) => v / 255);
const SEL_RGB = rgb(NET_INK).map((v) => v / 255);

const STROKE_VS = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform vec3 u_width;        // normal, extra when selected, (unused)
uniform float u_casing;      // extra px for the white casing pass, 0 for the colour pass
uniform int u_kind;          // 0 track, 1 lines, 2 preview
uniform int u_sel_line;
uniform int u_sel_edge;
uniform int u_only_sel;
uniform int u_by_level;
uniform float u_px_per_unit;
uniform float u_slot;        // px between side-by-side lines (T-062)
uniform vec3 u_level_colours[7];
uniform vec3 u_track;
uniform vec3 u_bad;
uniform vec3 u_sel;
uniform highp sampler2D u_line_colours;
in vec4 a_seg;
in uint a_colour;
in uint a_edge;
in float a_level;
in float a_flags;
in float a_dist;
in float a_offset;           // line widths to the left of the segment's direction
in float a_wmul;             // width as a multiple of the line width (riders, T-084); 1 otherwise
out vec3 v_color;
out float v_alpha;
out float v_dash;
out float v_dashed;
void main() {
  int flags = int(a_flags + 0.5);
  bool sel = (u_kind == 1 && int(a_colour) == u_sel_line) || (u_kind == 0 && int(a_edge) == u_sel_edge);
  bool blueprint = (flags & 1) != 0;
  if ((u_only_sel == 1 && !sel) || (u_casing > 0.0 && blueprint)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float w = u_width.x * a_wmul + (sel ? u_width.y : 0.0) + u_casing;
  int id = gl_VertexID;
  float along = float(id & 1);
  float side = id < 2 ? -1.0 : 1.0;
  vec4 c0 = u_matrix * vec4(a_seg.xy, 0.0, 1.0);
  vec4 c1 = u_matrix * vec4(a_seg.zw, 0.0, 1.0);
  vec2 half_vp = u_viewport * 0.5;
  vec2 s0 = c0.xy / c0.w * half_vp;
  vec2 s1 = c1.xy / c1.w * half_vp;
  vec2 dir = s1 - s0;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 c = along < 0.5 ? c0 : c1;
  vec2 off = nrm * (side * w * 0.5 + a_offset * u_slot) + dir * (along * 2.0 - 1.0) * w * 0.5;
  c.xy += off / half_vp * c.w;
  gl_Position = c;
  v_dash = a_dist * u_px_per_unit + along * len - (along * 2.0 - 1.0) * w * 0.5;
  v_dashed = blueprint ? 1.0 : 0.0;
  v_alpha = (flags & 3) != 0 ? 0.55 : 1.0;
  vec3 lc = u_level_colours[clamp(3 - int(round(a_level)), 0, 6)];
  if (u_casing > 0.0) v_color = vec3(1.0);
  else if ((flags & 16) != 0) v_color = u_bad;
  else if (u_kind == 2 || u_by_level == 1) v_color = lc;
  else if (u_kind == 0) v_color = sel ? u_sel : u_track;
  else v_color = texelFetch(u_line_colours, ivec2(int(a_colour) % ${LINES_W}, int(a_colour) / ${LINES_W}), 0).rgb;
}`;

const STROKE_FS = `#version 300 es
precision mediump float;
in vec3 v_color;
in float v_alpha;
in float v_dash;
in float v_dashed;
out vec4 o;
void main() {
  if (v_dashed > 0.5 && mod(v_dash, 13.0) > 8.0) discard;
  o = vec4(v_color * v_alpha, v_alpha);
}`;

const STATION_VS = `#version 300 es
precision highp float;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform vec2 u_radius;   // normal, transfer (px)
uniform float u_sel;     // index of the selected station, -1 for none
uniform float u_slot;    // px between side-by-side lines (T-062), 0 when lines are not drawn
in vec3 a_st;            // x, y, flags (1 transfer, 2 blueprint, lines side by side here << 2)
in vec2 a_sz;            // the bundle's width here in line widths; riders, 0..1 (T-084)
out vec2 v_p;
out float v_r;
out float v_alpha;
void main() {
  int flags = int(a_st.z + 0.5);
  float r = (flags & 1) != 0 ? u_radius.y : u_radius.x;
  // bigger with more riders (T-084): up to 2.6 times at the busiest
  r *= 1.0 + 1.6 * a_sz.y;
  // span the bundle of lines through it
  if (a_sz.x > 1.5) r = max(r, 0.5 * a_sz.x * u_slot + 1.0);
  if (float(gl_InstanceID) == u_sel) r *= 1.35;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  vec4 c = u_matrix * vec4(a_st.xy, 0.0, 1.0);
  vec2 off = corner * (r + 1.0);
  c.xy += off / (u_viewport * 0.5) * c.w;
  gl_Position = c;
  v_p = off;
  v_r = r;
  v_alpha = (flags & 2) != 0 ? 0.6 : 1.0;
}`;

const STATION_FS = `#version 300 es
precision mediump float;
uniform float u_ring;
uniform vec3 u_ink;
in vec2 v_p;
in float v_r;
in float v_alpha;
out vec4 o;
void main() {
  float d = length(v_p);
  float a = clamp(v_r - d + 0.5, 0.0, 1.0) * v_alpha;
  vec3 col = d > v_r - u_ring ? u_ink : vec3(1.0);
  o = vec4(col * a, a);
}`;

// Shared by the draw shader and the transform-feedback probe, so the probe checks the very
// arithmetic that positions the trains on screen.
const POS_GLSL = `
uniform float u_time;        // game seconds since the render epoch
uniform highp sampler2D u_phases;  // RGBA32F: t, s, v, a
uniform highp sampler2D u_meta;    // RGBA32F per profile: first phase, count, trip s, length m
uniform highp sampler2D u_samples; // RG32F: run 0 of each line, x/y local units
uniform highp sampler2D u_lines;   // RGBA32F per line: first sample, count, length m, step m
in vec2 a_trip;              // profile, departure - epoch (s)

ivec2 tc(int i, int w) { return ivec2(i % w, i / w); }

/** The point at distance d (m) along run 0 of a line (L from the line table). */
vec2 lineAt(vec4 L, float d) {
  float first = L.x, count = L.y, len = L.z, step = L.w;
  float dd = clamp(d, 0.0, len);
  float i0 = clamp(floor(dd / step), 0.0, max(count - 2.0, 0.0));
  float seg = max(min(step, len - i0 * step), 1e-3);
  float f = clamp((dd - i0 * step) / seg, 0.0, 1.0);
  int si = int(first + i0);
  vec2 p0 = texelFetch(u_samples, tc(si, ${TEX_W}), 0).xy;
  vec2 p1 = texelFetch(u_samples, tc(si + 1, ${TEX_W}), 0).xy;
  return mix(p0, p1, f);
}

/** The same for a sampled scalar (the sideways offset, T-062). */
float lineOff(highp sampler2D tex, vec4 L, float d) {
  float first = L.x, count = L.y, len = L.z, step = L.w;
  float dd = clamp(d, 0.0, len);
  float i0 = clamp(floor(dd / step), 0.0, max(count - 2.0, 0.0));
  float seg = max(min(step, len - i0 * step), 1e-3);
  float f = clamp((dd - i0 * step) / seg, 0.0, 1.0);
  int si = int(first + i0);
  return mix(texelFetch(tex, tc(si, ${TEX_W}), 0).x, texelFetch(tex, tc(si + 1, ${TEX_W}), 0).x, f);
}

/** The train's distance along run 0 of its line now; false if it is not on the track. */
bool trainDist(out vec4 L, out float d, out int line) {
  int prof = int(a_trip.x + 0.5);
  vec4 M = texelFetch(u_meta, tc(prof, ${TEX_W}), 0);
  float tau = u_time - a_trip.y;
  if (M.y < 0.5 || tau < 0.0 || tau > M.z) return false;
  int lo = int(M.x), hi = int(M.x + M.y) - 1;
  for (int k = 0; k < 18; k++) {
    if (lo >= hi) break;
    int mid = (lo + hi + 1) / 2;
    if (texelFetch(u_phases, tc(mid, ${TEX_W}), 0).x <= tau) lo = mid; else hi = mid - 1;
  }
  vec4 p = texelFetch(u_phases, tc(lo, ${TEX_W}), 0);
  float tt = max(tau - p.x, 0.0);
  if (p.w < 0.0) tt = min(tt, p.z / -p.w);
  float s = p.y + p.z * tt + 0.5 * p.w * tt * tt;
  line = prof / 6;
  int run = (prof / 3) % 2;
  d = run == 0 ? s : M.w - s;
  L = texelFetch(u_lines, tc(line, ${LINES_W}), 0);
  return true;
}`;

const TRAIN_VS = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 u_matrix;
uniform vec2 u_viewport;
uniform float u_px_per_unit;
uniform vec2 u_min_px;      // minimum length, width in px
uniform vec2 u_size_units;  // real length, width in local units
uniform highp sampler2D u_line_colours;
uniform highp sampler2D u_soff;  // R32F per sample: sideways offset, line widths (T-062)
uniform highp sampler2D u_fill;  // RG32F per sample: how full run 0's and run 1's trains are (T-084)
uniform float u_slot;            // px per line width; 0 when lines are not drawn
${POS_GLSL}
out vec2 v_uv;
out vec2 v_px;
out vec3 v_color;
out vec3 v_empty;
out float v_fill;
out float v_front;

void main() {
  vec4 L;
  float d;
  int line;
  if (!trainDist(L, d, line)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 p = lineAt(L, d);
  vec2 px = max(u_min_px, u_size_units * u_px_per_unit);
  // Heading along the chord between the train's two ends (as its bogies sit on the track), not
  // the sample segment under its centre, which turns in steps on curves (T-047).
  float h = max(0.5 * px.x / u_px_per_unit / ${MERC_K.toFixed(6)}, 1.0);
  vec2 back = lineAt(L, d - h), front = lineAt(L, d + h);
  vec4 c = u_matrix * vec4(p, 0.0, 1.0);
  vec4 cb = u_matrix * vec4(back, 0.0, 1.0);
  vec4 cf = u_matrix * vec4(front, 0.0, 1.0);
  vec2 half_vp = u_viewport * 0.5;
  vec2 sdir = (cf.xy / cf.w - cb.xy / cb.w) * half_vp;
  float sl = length(sdir);
  sdir = sl > 1e-6 ? sdir / sl : vec2(1.0, 0.0);
  vec2 nrm = vec2(-sdir.y, sdir.x);
  int id = gl_VertexID;
  vec2 corner = vec2(float(id & 1) * 2.0 - 1.0, id < 2 ? -1.0 : 1.0);
  // 1 px larger than the train all round, so the soft edge is never cut by the quad (T-047).
  vec2 halfq = px * 0.5 + 1.0;
  // on its line's own stroke where lines run side by side; sdir is run 0's direction, as the
  // offsets are (T-062)
  float side = u_slot > 0.0 ? lineOff(u_soff, L, d) * u_slot : 0.0;
  vec2 off = sdir * corner.x * halfq.x + nrm * (corner.y * halfq.y + side);
  c.xy += off / half_vp * c.w;
  gl_Position = c;
  v_uv = corner * halfq;
  v_px = px;
  vec3 lc = texelFetch(u_line_colours, ivec2(line % ${LINES_W}, line / ${LINES_W}), 0).rgb;
  v_color = lc * 0.62;
  // How full it is (T-084): the stretch it runs, from the sample ahead of it on its way.
  int run = (int(a_trip.x + 0.5) / 3) % 2;
  float i = floor(clamp(d, 0.0, L.z) / L.w) + (run == 0 ? 1.0 : 0.0);
  vec2 f = texelFetch(u_fill, tc(int(L.x + clamp(i, 0.0, L.y - 1.0)), ${TEX_W}), 0).xy;
  v_fill = run == 0 ? f.x : f.y;
  v_empty = mix(lc, vec3(1.0), 0.6);
  v_front = run == 0 ? 1.0 : -1.0;
}`;

// A capsule with a 1 px white edge blended into the core (T-047). With riders known (T-084) the
// core fills from the back like a gauge: dark up to how full the train is, a pale tint of the line
// colour beyond; half way means every seat taken, full means packed to crush; an over-full train
// is full and has a red edge.
const TRAIN_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
in vec2 v_px;
in vec3 v_color;
in vec3 v_empty;
in float v_fill;
in float v_front;
uniform vec3 u_bad;
out vec4 o;
void main() {
  vec2 p = v_uv;
  float r = v_px.y * 0.5;
  float hx = max(v_px.x * 0.5 - r, 0.0);
  float d = length(vec2(max(abs(p.x) - hx, 0.0), p.y)) - r;
  float a = clamp(0.5 - d, 0.0, 1.0);
  vec3 core = v_color;
  vec3 rim = vec3(1.0);
  if (v_fill >= 0.0) {
    float back = p.x * v_front + v_px.x * 0.5;   // px from the back of the train
    core = mix(v_empty, v_color, clamp(min(v_fill, 1.0) * v_px.x - back + 0.5, 0.0, 1.0));
    if (v_fill > 1.5) rim = u_bad;
  }
  vec3 col = mix(core, rim, clamp(d + 1.7, 0.0, 1.0));
  o = vec4(col * a, a);
}`;

const PROBE_VS = `#version 300 es
precision highp float;
precision highp int;
${POS_GLSL}
out vec2 v_pos;
void main() {
  vec4 L;
  float d;
  int line;
  v_pos = trainDist(L, d, line) ? lineAt(L, d) : vec2(1e9);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;

const PROBE_FS = `#version 300 es
precision mediump float;
out vec4 o;
void main() { o = vec4(0.0); }`;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string, feedback?: string[]): WebGLProgram {
  const prog = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  if (feedback) gl.transformFeedbackVaryings(prog, feedback, gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(prog));
  return prog;
}

function uniforms(gl: WebGL2RenderingContext, prog: WebGLProgram, names: string[]) {
  const u: Record<string, WebGLUniformLocation | null> = {};
  for (const n of names) u[n] = gl.getUniformLocation(prog, n);
  return u;
}

/** Piecewise-linear in zoom, like a MapLibre "interpolate" expression. */
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

/** Station sizes without demand: the lines side by side at each (flags >> 2), no riders. */
export function plainStationSize(stations: Float32Array, count: number): Float32Array {
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) out[i * 2] = Math.floor((stations[i * 3 + 2] + 0.5) / 4);
  return out;
}

export interface DrawOptions {
  showTrains: boolean;
  byLevel: boolean;
  /** selected line id, -1 for none */
  selLine: number;
  /** selected edge id, -1 for none */
  selEdge: number;
  /** selected station's index in the station buffer, -1 for none */
  selStation: number;
}

interface StrokeVao {
  vao: WebGLVertexArrayObject;
  count: number;
  /** the per-segment sideways offsets and width multipliers, rewritten by `setLayers` */
  offsetBuf: WebGLBuffer;
  wmulBuf: WebGLBuffer;
}

/**
 * The demand views drawn by the network renderer (T-084, map/demandLayers.ts), one value per
 * line stroke segment, train sample or station; null = drawn as without demand.
 */
export interface NetworkLayers {
  /** per line stroke segment: width multiple and sideways offset (line widths) */
  lineWidth: Float32Array;
  lineOffset: Float32Array;
  /** per train sample: sideways offset, and how full run 0's and run 1's trains are (2 each:
   * 0-1 of the gauge, 2 over full, -1 not known) */
  sampleOff: Float32Array;
  fill: Float32Array;
  /** per station: bundle width in line widths, riders 0-1 (2 each) */
  stationSize: Float32Array;
}

export class NetworkRenderer {
  private gl!: WebGL2RenderingContext;
  private strokeProg!: WebGLProgram;
  private stationProg!: WebGLProgram;
  private trainProg!: WebGLProgram;
  private strokeU!: Record<string, WebGLUniformLocation | null>;
  private stationU!: Record<string, WebGLUniformLocation | null>;
  private trainU!: Record<string, WebGLUniformLocation | null>;
  private probeProg: WebGLProgram | null = null;
  private probeU: Record<string, WebGLUniformLocation | null> = {};
  private res: { del: () => void }[] = [];
  private previewRes: { del: () => void }[] = [];
  private track: StrokeVao | null = null;
  private lines: StrokeVao | null = null;
  private preview: StrokeVao | null = null;
  private stationVao: WebGLVertexArrayObject | null = null;
  private trainVao: WebGLVertexArrayObject | null = null;
  private probeVao: WebGLVertexArrayObject | null = null;
  private tripBuf: WebGLBuffer | null = null;
  private tex: Record<"phases" | "meta" | "samples" | "lines" | "colours" | "soff" | "fill", WebGLTexture | null> = { phases: null, meta: null, samples: null, lines: null, colours: null, soff: null, fill: null };
  private stationSizeBuf: WebGLBuffer | null = null;
  /** CSS px between side-by-side lines at the last draw, 0 when lines are not drawn (picking) */
  slotPx = 0;
  private levelColours = new Float32Array([3, 2, 1, 0, -1, -2, -3].flatMap((l) => rgb(LEVEL_COLOURS[l as 3]).map((v) => v / 255)));
  private ink = new Float32Array(rgb(NET_INK).map((v) => v / 255));
  private timerExt: any = null;
  private pendingQuery: WebGLQuery | null = null;
  /** trips per epoch from the worker, newest version kept */
  private tripsBy = new Map<number, { version: number; data: Float32Array }>();
  /** what the trip buffer holds */
  private tripEpoch = NaN;
  private tripVersion = -1;
  tripCount = 0;
  /** the trips in the buffer, profile and departure - epoch (for picking) */
  trips: Float32Array = new Float32Array(0);
  net: NetworkBuffers | null = null;
  /** last GPU time for our draws in ms, if the timer extension exists */
  gpuMs: number | null = null;
  /** current render epoch, game seconds; NaN until the first draw */
  epoch = NaN;
  rebases = 0;
  /** called with each new epoch the clock enters (the worker sends its trips) */
  onEpoch: (epoch: number) => void = () => {};

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.strokeProg = compile(gl, STROKE_VS, STROKE_FS);
    this.stationProg = compile(gl, STATION_VS, STATION_FS);
    this.trainProg = compile(gl, TRAIN_VS, TRAIN_FS);
    this.strokeU = uniforms(gl, this.strokeProg, ["u_matrix", "u_viewport", "u_width", "u_casing", "u_kind", "u_sel_line", "u_sel_edge", "u_only_sel", "u_by_level", "u_px_per_unit", "u_slot", "u_level_colours", "u_track", "u_bad", "u_sel", "u_line_colours"]);
    this.stationU = uniforms(gl, this.stationProg, ["u_matrix", "u_viewport", "u_radius", "u_ring", "u_ink", "u_sel", "u_slot"]);
    this.trainU = uniforms(gl, this.trainProg, ["u_matrix", "u_viewport", "u_time", "u_px_per_unit", "u_min_px", "u_size_units", "u_phases", "u_meta", "u_samples", "u_lines", "u_line_colours", "u_soff", "u_fill", "u_bad", "u_slot"]);
    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.tripBuf = gl.createBuffer();
  }

  private texture(data: ArrayBufferView, kind: "r32f" | "rg32f" | "rgba32f" | "rgba8", width: number, list = this.res): WebGLTexture {
    const gl = this.gl;
    const comps = kind === "r32f" ? 1 : kind === "rg32f" ? 2 : 4;
    const texels = Math.max(1, Math.ceil((data as Float32Array).length / comps));
    const height = Math.max(1, Math.ceil(texels / width));
    const padded = kind === "rgba8" ? new Uint8Array(width * height * 4) : new Float32Array(width * height * comps);
    padded.set(data as any);
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    if (kind === "r32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded);
    else if (kind === "rg32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, height, 0, gl.RG, gl.FLOAT, padded);
    else if (kind === "rgba32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, padded);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, padded);
    list.push({ del: () => gl.deleteTexture(tex) });
    return tex;
  }

  private buffer(data: ArrayBufferView, list = this.res): WebGLBuffer {
    const gl = this.gl;
    const b = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    // left bound, WebGL refuses it as a transform feedback target (the probe)
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    list.push({ del: () => gl.deleteBuffer(b) });
    return b;
  }

  private attrib(prog: WebGLProgram, name: string, buf: WebGLBuffer, size: number, type: number, divisor: number, integer = false) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    if (integer) gl.vertexAttribIPointer(loc, size, type, 0, 0);
    else gl.vertexAttribPointer(loc, size, type, false, 0, 0);
    gl.vertexAttribDivisor(loc, divisor);
  }

  private strokes(s: Strokes, list = this.res): StrokeVao {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    list.push({ del: () => gl.deleteVertexArray(vao) });
    gl.bindVertexArray(vao);
    const p = this.strokeProg;
    const one = (a: ArrayBufferView, n: number) => (s.count ? a : new Float32Array(n));
    this.attrib(p, "a_seg", this.buffer(one(s.seg, 4), list), 4, gl.FLOAT, 1);
    this.attrib(p, "a_colour", this.buffer(s.count ? s.colour : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_edge", this.buffer(s.count ? s.edge : new Uint32Array(1), list), 1, gl.UNSIGNED_INT, 1, true);
    this.attrib(p, "a_level", this.buffer(one(s.level, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_flags", this.buffer(one(s.flags, 1), list), 1, gl.FLOAT, 1);
    this.attrib(p, "a_dist", this.buffer(one(s.dist, 1), list), 1, gl.FLOAT, 1);
    const offsetBuf = this.buffer(s.count && s.offset ? s.offset : new Float32Array(Math.max(1, s.count)), list);
    this.attrib(p, "a_offset", offsetBuf, 1, gl.FLOAT, 1);
    const wmulBuf = this.buffer(new Float32Array(Math.max(1, s.count)).fill(1), list);
    this.attrib(p, "a_wmul", wmulBuf, 1, gl.FLOAT, 1);
    gl.bindVertexArray(null);
    return { vao, count: s.count, offsetBuf, wmulBuf };
  }

  /** Rewrite a dynamic buffer in place (sizes never change between `setNetwork`s). */
  private rewrite(buf: WebGLBuffer, data: Float32Array) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** The demand views (T-084) for the network now drawn, or null for the plain drawing. The
   * arrays must be sized for `this.net` (map/demandLayers.ts builds them from it). */
  setLayers(l: NetworkLayers | null) {
    const n = this.net;
    if (!n || !this.lines) return;
    const lines = this.lines;
    if (lines.count) {
      this.rewrite(lines.offsetBuf, l ? l.lineOffset : n.lines.offset ?? new Float32Array(lines.count));
      this.rewrite(lines.wmulBuf, l ? l.lineWidth : new Float32Array(lines.count).fill(1));
    }
    const samples = Math.max(1, n.samples.length / 2);
    const upload = (tex: WebGLTexture | null, data: Float32Array, kind: "r32f" | "rg32f") => {
      const gl = this.gl;
      const comps = kind === "r32f" ? 1 : 2;
      const height = Math.max(1, Math.ceil(samples / TEX_W));
      const padded = new Float32Array(TEX_W * height * comps);
      padded.set(data.subarray(0, Math.min(data.length, padded.length)));
      if (kind === "rg32f" && !l) padded.fill(-1);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, height, kind === "r32f" ? gl.RED : gl.RG, gl.FLOAT, padded);
    };
    upload(this.tex.soff, l ? l.sampleOff : n.sampleOff ?? new Float32Array(samples), "r32f");
    upload(this.tex.fill, l ? l.fill : new Float32Array(samples * 2), "rg32f");
    if (this.stationSizeBuf && n.stationCount) this.rewrite(this.stationSizeBuf, l ? l.stationSize : plainStationSize(n.stations, n.stationCount));
  }

  /** Replace everything drawn but the trips (an edit). */
  setNetwork(n: NetworkBuffers) {
    const gl = this.gl;
    for (const r of this.res) r.del();
    this.res = [];
    this.probeVao = null;
    this.net = n;
    this.track = this.strokes(n.track);
    this.lines = this.strokes(n.lines);
    this.tex.phases = this.texture(n.phases.length ? n.phases : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.meta = this.texture(n.meta.length ? n.meta : new Float32Array(4), "rgba32f", TEX_W);
    this.tex.samples = this.texture(n.samples.length ? n.samples : new Float32Array(2), "rg32f", TEX_W);
    this.tex.lines = this.texture(n.lineTable.length ? n.lineTable : new Float32Array(4), "rgba32f", LINES_W);
    this.tex.colours = this.texture(n.lineColours, "rgba8", LINES_W);
    this.tex.soff = this.texture(n.sampleOff?.length ? n.sampleOff : new Float32Array(Math.max(1, n.samples.length / 2)), "r32f", TEX_W);
    this.tex.fill = this.texture(new Float32Array(Math.max(1, n.samples.length / 2) * 2).fill(-1), "rg32f", TEX_W);

    this.stationVao = gl.createVertexArray()!;
    const sv = this.stationVao;
    this.res.push({ del: () => gl.deleteVertexArray(sv) });
    gl.bindVertexArray(this.stationVao);
    this.attrib(this.stationProg, "a_st", this.buffer(n.stations.length ? n.stations : new Float32Array(3)), 3, gl.FLOAT, 1);
    this.stationSizeBuf = this.buffer(n.stationCount ? plainStationSize(n.stations, n.stationCount) : new Float32Array(2));
    this.attrib(this.stationProg, "a_sz", this.stationSizeBuf, 2, gl.FLOAT, 1);

    this.trainVao = gl.createVertexArray()!;
    const tv = this.trainVao;
    this.res.push({ del: () => gl.deleteVertexArray(tv) });
    gl.bindVertexArray(this.trainVao);
    this.attrib(this.trainProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 1);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** The drawing tool's alignment, or null. */
  setPreview(s: Strokes | null) {
    for (const r of this.previewRes) r.del();
    this.previewRes = [];
    this.preview = s && s.count ? this.strokes(s, this.previewRes) : null;
  }

  /** Trips for the hour starting at `epoch` (from the worker). */
  setTrips(epoch: number, version: number, data: Float32Array) {
    const old = this.tripsBy.get(epoch);
    if (old && old.version > version) return;
    this.tripsBy.set(epoch, { version, data });
    if (epoch === this.tripEpoch) this.tripVersion = -1; // re-upload on the next draw
  }

  /** Make the trip buffer hold the trips of the hour `time` is in, when they are here. */
  private sync(time: number): number {
    const epoch = Math.floor(time / EPOCH_S) * EPOCH_S;
    if (epoch !== this.epoch) {
      this.epoch = epoch;
      this.rebases++;
      for (const k of this.tripsBy.keys()) if (k < epoch) this.tripsBy.delete(k);
      this.onEpoch(epoch);
    }
    const t = this.tripsBy.get(epoch);
    if (t && (this.tripEpoch !== epoch || this.tripVersion !== t.version)) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.tripBuf);
      gl.bufferData(gl.ARRAY_BUFFER, t.data.length ? t.data : new Float32Array(2), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      this.tripEpoch = epoch;
      this.tripVersion = t.version;
      this.tripCount = t.data.length / 2;
      this.trips = t.data;
    }
    // Until the new hour's trips arrive, the last hour's keep running on their own epoch.
    return time - this.tripEpoch;
  }

  /** The epoch the trip buffer is relative to. */
  get tripsEpoch() {
    return this.tripEpoch;
  }

  /**
   * Draw. `mercMatrix` maps mercator 0..1 to clip space (MapLibre's
   * `defaultProjectionData.mainMatrix`, float64); `time` is game seconds; `zoom` and `dpr` size
   * things in px. The caller has bound and cleared the target.
   */
  draw(mercMatrix: ArrayLike<number>, time: number, zoom: number, dpr: number, opt: DrawOptions) {
    const gl = this.gl;
    const n = this.net;
    if (!n) return;
    const uTime = this.sync(time);
    const m = mul4(mercMatrix, LOCAL_TO_MERC);
    const vp = [gl.drawingBufferWidth, gl.drawingBufferHeight] as const;
    const pxPerUnit = ((512 * Math.pow(2, zoom)) / EARTH_C) * dpr;

    this.pollTimer();
    const timing = this.timerExt && !this.pendingQuery;
    if (timing) {
      this.pendingQuery = gl.createQuery();
      gl.beginQuery(this.timerExt.TIME_ELAPSED_EXT, this.pendingQuery!);
    }

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const wLine = interp(zoom, [[9, 3.5], [12, 6.5], [15, 11]]) * dpr;
    const wTrack = interp(zoom, [[9, 1.6], [12, 3], [15, 5]]) * dpr;
    const wSel = interp(zoom, [[9, 2], [12, 3.5], [15, 5]]) * dpr;
    const casing = interp(zoom, [[9, 2], [12, 2.5], [15, 3]]) * dpr;
    gl.useProgram(this.strokeProg);
    const u = this.strokeU;
    gl.uniformMatrix4fv(u.u_matrix, false, m);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1i(u.u_sel_line, opt.selLine);
    gl.uniform1i(u.u_sel_edge, opt.selEdge);
    gl.uniform1i(u.u_by_level, opt.byLevel ? 1 : 0);
    gl.uniform1f(u.u_px_per_unit, pxPerUnit);
    // Lines sharing track sit side by side, one line width apart (T-062); not when the track
    // is coloured by height (no line strokes then).
    const slot = opt.byLevel ? 0 : wLine;
    this.slotPx = slot / dpr;
    gl.uniform1f(u.u_slot, slot);
    gl.uniform3fv(u.u_level_colours, this.levelColours);
    gl.uniform3fv(u.u_track, TRACK_RGB);
    gl.uniform3fv(u.u_bad, BAD_RGB);
    gl.uniform3fv(u.u_sel, SEL_RGB);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
    gl.uniform1i(u.u_line_colours, 2);
    const stroke = (s: StrokeVao | null, kind: number, width: number, sel: boolean) => {
      if (!s || !s.count) return;
      gl.uniform1i(u.u_kind, kind);
      gl.uniform3f(u.u_width, width, wSel, 0);
      gl.bindVertexArray(s.vao);
      const passes: [number, number][] = [[casing, 0], [0, 0]];
      if (sel) passes.push([casing, 1], [0, 1]);
      for (const [cas, only] of passes) {
        gl.uniform1f(u.u_casing, cas);
        gl.uniform1i(u.u_only_sel, only);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.count);
      }
    };
    // Colour by height: the track itself, at line width, and no line strokes over it.
    stroke(this.track, 0, opt.byLevel ? wLine : wTrack, opt.selEdge >= 0);
    if (!opt.byLevel) stroke(this.lines, 1, wLine, opt.selLine >= 0);
    stroke(this.preview, 2, wLine, false);

    if (n.stationCount) {
      const r = interp(zoom, [[9, 2.6], [12, 4.8], [15, 7]]) * dpr;
      const rx = interp(zoom, [[9, 3.5], [12, 6.2], [15, 9]]) * dpr;
      gl.useProgram(this.stationProg);
      gl.uniformMatrix4fv(this.stationU.u_matrix, false, m);
      gl.uniform2f(this.stationU.u_viewport, vp[0], vp[1]);
      gl.uniform2f(this.stationU.u_radius, r, rx);
      gl.uniform1f(this.stationU.u_ring, interp(zoom, [[9, 1.2], [13, 2]]) * dpr);
      gl.uniform3fv(this.stationU.u_ink, this.ink);
      gl.uniform1f(this.stationU.u_sel, opt.selStation);
      gl.uniform1f(this.stationU.u_slot, slot);
      gl.bindVertexArray(this.stationVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n.stationCount);
    }

    if (opt.showTrains && this.tripCount) {
      const tw = interp(zoom, [[9, 5], [12, 8], [15, 12]]) * dpr;
      gl.useProgram(this.trainProg);
      gl.uniformMatrix4fv(this.trainU.u_matrix, false, m);
      gl.uniform2f(this.trainU.u_viewport, vp[0], vp[1]);
      gl.uniform1f(this.trainU.u_time, uTime);
      gl.uniform1f(this.trainU.u_px_per_unit, pxPerUnit);
      gl.uniform2f(this.trainU.u_min_px, tw * 2.2, tw);
      gl.uniform2f(this.trainU.u_size_units, 160 * MERC_K, 0);
      this.bindTrainTextures(this.trainU);
      gl.uniform1i(this.trainU.u_line_colours, 2);
      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.soff);
      gl.uniform1i(this.trainU.u_soff, 5);
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, this.tex.fill);
      gl.uniform1i(this.trainU.u_fill, 6);
      gl.uniform3fv(this.trainU.u_bad, BAD_RGB);
      gl.uniform1f(this.trainU.u_slot, slot);
      gl.bindVertexArray(this.trainVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.tripCount);
    }

    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
    if (timing) gl.endQuery(this.timerExt.TIME_ELAPSED_EXT);
  }

  private bindTrainTextures(u: Record<string, WebGLUniformLocation | null>) {
    const gl = this.gl;
    const bind = (unit: number, tex: WebGLTexture | null, loc: WebGLUniformLocation | null) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(3, this.tex.phases, u.u_phases);
    bind(4, this.tex.meta, u.u_meta);
    bind(0, this.tex.samples, u.u_samples);
    bind(1, this.tex.lines, u.u_lines);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.colours);
  }

  /**
   * GPU positions (local units, x/y pairs; 1e9 when off the track) of every trip in the buffer at
   * game time `time`, read back by transform feedback. For precision checks; changes GL state
   * freely, so only call it on a renderer with a context of its own.
   */
  probe(time: number): Float32Array {
    const gl = this.gl;
    if (!this.probeProg) {
      this.probeProg = compile(gl, PROBE_VS, PROBE_FS, ["v_pos"]);
      this.probeU = uniforms(gl, this.probeProg, ["u_time", "u_phases", "u_meta", "u_samples", "u_lines"]);
    }
    const uTime = this.sync(time);
    const n = this.tripCount;
    if (!this.probeVao) {
      this.probeVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.probeVao);
      this.attrib(this.probeProg, "a_trip", this.tripBuf!, 2, gl.FLOAT, 0);
      gl.bindVertexArray(null);
    }
    const out = this.buffer(new Float32Array(Math.max(1, n) * 2));
    gl.useProgram(this.probeProg);
    gl.uniform1f(this.probeU.u_time, uTime);
    this.bindTrainTextures(this.probeU);
    gl.bindVertexArray(this.probeVao);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, out);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindVertexArray(null);
    const res = new Float32Array(n * 2);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, out);
    gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, res);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, null);
    return res;
  }

  private pollTimer() {
    const gl = this.gl;
    const q = this.pendingQuery;
    if (!q) return;
    if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return;
    const disjoint = gl.getParameter(this.timerExt.GPU_DISJOINT_EXT);
    if (!disjoint) this.gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
    gl.deleteQuery(q);
    this.pendingQuery = null;
  }
}
