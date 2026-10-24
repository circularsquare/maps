// WebGL and canvas helpers the map's layers share: shader programs, uniform lookup, the GPU timer,
// canvas sizing and the camera matrix to CSS px.

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
