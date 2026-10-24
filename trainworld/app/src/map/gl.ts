// WebGL and canvas helpers the map's layers share: shader programs, uniform lookup, the GPU timer,
// canvas sizing, and the camera: MapLibre's flat map or globe, to clip space or CSS px (T-033).

import { EARTH_C, localToMercMatrix, mul4, type Anchor } from "./geo";

export function compile(gl: WebGL2RenderingContext, vs: string, fs: string, feedback?: string[]): WebGLProgram {
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

export function uniforms(gl: WebGL2RenderingContext, prog: WebGLProgram, names: string[]) {
  const u: Record<string, WebGLUniformLocation | null> = {};
  for (const n of names) u[n] = gl.getUniformLocation(prog, n);
  return u;
}

/** Piecewise-linear in zoom, like a MapLibre "interpolate" expression: flat before the first stop
 * and after the last. */
export function interp(z: number, stops: [number, number][]): number {
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

/** The time one frame's draws took on the GPU, where the browser has the timer extension. `begin`
 * and `end` bracket the draws; a query still pending when the next frame begins skips that frame's. */
export class GpuTimer {
  /** the last result, ms; null until one has come, or without the extension */
  ms: number | null = null;
  private ext: any;
  private query: WebGLQuery | null = null;
  private timing = false;

  constructor(private gl: WebGL2RenderingContext) {
    this.ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  }

  begin() {
    const gl = this.gl, q = this.query;
    if (q && gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) {
      if (!gl.getParameter(this.ext.GPU_DISJOINT_EXT)) this.ms = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
      gl.deleteQuery(q);
      this.query = null;
    }
    this.timing = !!this.ext && !this.query;
    if (this.timing) {
      this.query = gl.createQuery();
      gl.beginQuery(this.ext.TIME_ELAPSED_EXT, this.query!);
    }
  }

  end() {
    if (this.timing) this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
  }
}

/** Size a canvas laid over the map's: its backing store at the map's pixel ratio, its CSS size the
 * map canvas's. Returns the CSS size. */
export function sizeCanvasLike(canvas: HTMLCanvasElement, map: { getPixelRatio(): number; getCanvas(): HTMLCanvasElement }): [number, number] {
  const dpr = map.getPixelRatio(), c = map.getCanvas(), w = c.clientWidth, h = c.clientHeight;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  return [w, h];
}

/** A point through a camera matrix `m` (column-major, as MapLibre's) to CSS px in a `w` by `h` view,
 * y down, into `out[at]` and `out[at + 1]`. The point is in the space `m` maps from (local units
 * for a matrix made with `localToMercMatrix`, Mercator 0..1 for the camera's). */
export function clipToCss(m: ArrayLike<number>, x: number, y: number, w: number, h: number, out: { [i: number]: number }, at = 0) {
  const cx = m[0] * x + m[4] * y + m[12], cy = m[1] * x + m[5] * y + m[13], cw = m[3] * x + m[7] * y + m[15];
  out[at] = ((cx / cw + 1) / 2) * w;
  out[at + 1] = ((1 - cy / cw) / 2) * h;
}

/** The camera MapLibre last drew with (T-033), from a custom layer's `defaultProjectionData`. On the
 * flat map, and on the globe from zoom 12 where MapLibre has blended it into the flat map, `t` is 0
 * and `merc` is the whole camera. */
export interface Camera {
  /** Mercator 0..1 to clip space, float64: the flat map's camera, which the globe blends to */
  merc: ArrayLike<number>;
  /** the unit sphere to clip space while the globe is drawn (`t` > 0), else null */
  globe: ArrayLike<number> | null;
  /** the plane through the horizon, on the unit sphere: points with dot(p, xyz) + w < 0 are behind it */
  plane: ArrayLike<number>;
  /** 0 the flat map, 1 the globe; between from zoom 11 to 12 (MapLibre's "globe" projection) */
  t: number;
}

const NO_PLANE = [0, 0, 0, 1];

/** The camera out of a custom layer's render arguments. */
export function cameraFrom(d: { mainMatrix: ArrayLike<number>; fallbackMatrix?: ArrayLike<number>; clippingPlane?: ArrayLike<number>; projectionTransition?: number }): Camera {
  const t = d.projectionTransition ?? 0;
  if (t > 0 && d.fallbackMatrix) return { merc: Array.from(d.fallbackMatrix), globe: Array.from(d.mainMatrix), plane: Array.from(d.clippingPlane ?? NO_PLANE), t };
  return { merc: Array.from(d.mainMatrix), globe: null, plane: NO_PLANE, t: 0 };
}

/** Mercator 0..1 through a camera to CSS px, as `clipToCss` (the same arithmetic on the flat map) and
 * the shaders' `project` (PROJECT_GLSL) on the globe. False, with NaN written, for a point behind
 * the globe's horizon. */
export function mercToCss(cam: Camera, mx: number, my: number, w: number, h: number, out: { [i: number]: number }, at = 0): boolean {
  const m = cam.merc, g = cam.globe;
  let cx = m[0] * mx + m[4] * my + m[12], cy = m[1] * mx + m[5] * my + m[13], cw = m[3] * mx + m[7] * my + m[15];
  if (g) {
    const lon = mx * 2 * Math.PI + Math.PI, e = Math.exp(Math.PI - my * 2 * Math.PI), d = e * e + 1;
    const cl = (2 * e) / d, sx = Math.sin(lon) * cl, sy = (e * e - 1) / d, sz = Math.cos(lon) * cl;
    const p = cam.plane, t = cam.t;
    // MapLibre's globe hides the far side by depth from 20% globe up; the blend is closer in than any horizon
    if (t > GLOBE_Z_FROM && sx * p[0] + sy * p[1] + sz * p[2] + p[3] < 0) {
      out[at] = out[at + 1] = NaN;
      return false;
    }
    const gx = g[0] * sx + g[4] * sy + g[8] * sz + g[12], gy = g[1] * sx + g[5] * sy + g[9] * sz + g[13], gw = g[3] * sx + g[7] * sy + g[11] * sz + g[15];
    cx += (gx - cx) * t;
    cy += (gy - cy) * t;
    cw += (gw - cw) * t;
  }
  out[at] = ((cx / cw + 1) / 2) * w;
  out[at + 1] = ((1 - cy / cw) / 2) * h;
  return true;
}
const GLOBE_Z_FROM = 0.2;

/** A place (lng, lat) is on the globe's far side, out of sight: what `mercToCss` returns NaN for. */
export function behindHorizon(cam: Camera | null | undefined, lng: number, lat: number): boolean {
  if (!cam?.globe || cam.t <= GLOBE_Z_FROM) return false;
  const lo = (lng * Math.PI) / 180, la = (lat * Math.PI) / 180, p = cam.plane;
  return Math.sin(lo) * Math.cos(la) * p[0] + Math.sin(la) * p[1] + Math.cos(lo) * Math.cos(la) * p[2] + p[3] < 0;
}

/** A camera for positions in local units of one anchor, as the shaders take it. */
export interface LocalCamera {
  /** local units to clip space on the flat map (u_matrix) */
  m: Float32Array;
  globe: Float32Array;
  plane: Float32Array;
  /** the anchor's mercator x, y, mercator per local unit, `t` (u_globe_at) */
  at: Float32Array;
}

/** `cam` for positions in local units of anchor `a`. */
export function localCamera(cam: Camera, a: Anchor): LocalCamera {
  return {
    m: mul4(cam.merc, localToMercMatrix(a)),
    globe: new Float32Array(cam.globe ?? IDENTITY),
    plane: new Float32Array(cam.plane),
    at: new Float32Array([a.merc[0], a.merc[1], 1 / EARTH_C, cam.globe ? cam.t : 0]),
  };
}
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The uniforms PROJECT_GLSL reads, for `uniforms()`. */
export const PROJECT_UNIFORMS = ["u_matrix", "u_globe", "u_globe_plane", "u_globe_at"];

export function setCamera(gl: WebGL2RenderingContext, u: Record<string, WebGLUniformLocation | null>, c: LocalCamera) {
  gl.uniformMatrix4fv(u.u_matrix, false, c.m);
  gl.uniformMatrix4fv(u.u_globe, false, c.globe);
  gl.uniform4fv(u.u_globe_plane, c.plane);
  gl.uniform4fv(u.u_globe_at, c.at);
}

/** `project(p)`: local units to clip space through a LocalCamera, the flat map's `u_matrix * p`
 * exactly while the globe is off, else MapLibre's globe (its _projection_globe.vertex.glsl:
 * Mercator onto the unit sphere, the horizon's clipping depth, the blend to flat close in). */
export const PROJECT_GLSL = `
uniform mat4 u_matrix;
uniform mat4 u_globe;
uniform vec4 u_globe_plane;
uniform vec4 u_globe_at;
vec4 project(vec2 p) {
  vec4 f = u_matrix * vec4(p, 0.0, 1.0);
  float t = u_globe_at.w;
  if (t <= 0.0) return f;
  vec2 m = u_globe_at.xy + p * u_globe_at.z;
  float lon = m.x * 6.28318531 + 3.14159265;
  float e = exp(3.14159265 - m.y * 6.28318531), d = e * e + 1.0, cl = 2.0 * e / d;
  vec3 s = vec3(sin(lon) * cl, (e * e - 1.0) / d, cos(lon) * cl);
  // the poles, past Mercator's square (the realm shade's caps)
  if (m.y < -1.0) s = vec3(0.0, 1.0, 0.0);
  if (m.y > 2.0) s = vec3(0.0, -1.0, 0.0);
  vec4 g = u_globe * vec4(s, 1.0);
  g.z = (1.0 - (dot(s, u_globe_plane.xyz) + u_globe_plane.w)) * g.w;
  if (t >= 0.999) return g;
  vec4 r = g;
  r.z = mix(0.0, g.z, clamp((t - ${GLOBE_Z_FROM}) / ${(1 - GLOBE_Z_FROM).toFixed(1)}, 0.0, 1.0));
  r.xyw = mix(f.xyw, g.xyw, t);
  return r;
}`;
