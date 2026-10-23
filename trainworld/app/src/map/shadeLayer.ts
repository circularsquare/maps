// T-1025: the realm shade's triangles (boundaryGeometry.outsideZones) drawn as a MapLibre custom layer
// in one colour. A GeoJSON source would tile them, and tiling rounds every vertex to the tile's grid,
// which turns the thinnest triangles inside out; the fill bucket then takes such a triangle for a hole
// in the one before it and earcut joins the two with a long stray triangle. Here every triangle is
// drawn as cut. Positions are doubles split into two floats and taken relative to the view's centre in
// the shader, so the edges hold still at street zooms.
import type { CustomLayerInterface, Map as MlMap } from "maplibre-gl";

const VERTEX = `#version 300 es
in vec2 a_hi;
in vec2 a_lo;
uniform mat4 u_matrix;
uniform vec2 u_center_hi;
uniform vec2 u_center_lo;
uniform float u_wrap;
void main() {
  vec2 p = (a_hi - u_center_hi) + (a_lo - u_center_lo) + vec2(u_wrap, 0.0);
  gl_Position = u_matrix * vec4(p, 0.0, 1.0);
}`;
const FRAGMENT = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }`;

export class ShadeLayer implements CustomLayerInterface {
  readonly type = "custom";
  readonly renderingMode = "2d";
  private map: MlMap | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private buffer: WebGLBuffer | null = null;
  private loc: Record<string, WebGLUniformLocation | null> = {};
  /** hi, lo, hi, lo (x then y) per vertex, waiting for the next upload */
  private data = new Float32Array(0);
  /** vertices: three a triangle (the checks read it) */
  count = 0;
  private stale = true;
  /** premultiplied */
  private color: [number, number, number, number] = [0, 0, 0, 0];
  visible = true;

  constructor(readonly id: string) {}

  /** The triangles, Mercator x, y (0-1) per corner. */
  setTriangles(xy: Float64Array) {
    const n = xy.length / 2, d = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const x = xy[2 * i], y = xy[2 * i + 1], hx = Math.fround(x), hy = Math.fround(y);
      d[4 * i] = hx; d[4 * i + 1] = hy; d[4 * i + 2] = x - hx; d[4 * i + 3] = y - hy;
    }
    this.data = d;
    this.count = n;
    this.stale = true;
    this.map?.triggerRepaint();
  }

  setColor(css: string, opacity: number) {
    const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(css);
    const [r, g, b] = m ? [1, 2, 3].map((i) => parseInt(m[i], 16) / 255) : [0, 0, 0];
    this.color = [r * opacity, g * opacity, b * opacity, opacity];
    this.map?.triggerRepaint();
  }

  onAdd(map: MlMap, gl: WebGL2RenderingContext) {
    this.map = map;
    this.gl = gl;
    const shader = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shade shader: " + gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, shader(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("shade program: " + gl.getProgramInfoLog(p));
    this.program = p;
    for (const u of ["u_matrix", "u_center_hi", "u_center_lo", "u_wrap", "u_color"]) this.loc[u] = gl.getUniformLocation(p, u);
    this.buffer = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const hi = gl.getAttribLocation(p, "a_hi"), lo = gl.getAttribLocation(p, "a_lo");
    gl.enableVertexAttribArray(hi);
    gl.vertexAttribPointer(hi, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(lo);
    gl.vertexAttribPointer(lo, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);
    this.stale = true;
  }

  onRemove(_map: MlMap, gl: WebGL2RenderingContext) {
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.buffer);
    gl.deleteVertexArray(this.vao);
    this.program = this.buffer = this.vao = null;
    this.map = this.gl = null;
  }

  render(gl: WebGL2RenderingContext, args: { defaultProjectionData: { mainMatrix: ArrayLike<number> } }) {
    if (!this.visible || !this.program || !this.map || !this.color[3]) return;
    gl.bindVertexArray(this.vao);
    if (this.stale) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.data, gl.STATIC_DRAW);
      this.stale = false;
    }
    if (this.count) {
      const m = args.defaultProjectionData.mainMatrix;
      const c = this.map.getCenter(), s = Math.sin((Math.max(-85.051129, Math.min(85.051129, c.lat)) * Math.PI) / 180);
      const cx = (c.lng + 180) / 360, cy = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
      // the camera's matrix with the centre moved to the origin, in doubles, then sent as floats
      const t = new Float32Array(16);
      for (let i = 0; i < 16; i++) t[i] = m[i];
      for (let r = 0; r < 4; r++) t[12 + r] = m[r] * cx + m[4 + r] * cy + m[12 + r];
      gl.useProgram(this.program);
      gl.uniformMatrix4fv(this.loc.u_matrix, false, t);
      const hx = Math.fround(cx), hy = Math.fround(cy);
      gl.uniform2f(this.loc.u_center_hi, hx, hy);
      gl.uniform2f(this.loc.u_center_lo, cx - hx, cy - hy);
      gl.uniform4f(this.loc.u_color, ...this.color);
      // the world copies in view
      const b = this.map.getBounds();
      for (let w = Math.floor((b.getWest() + 180) / 360); w <= Math.floor((b.getEast() + 180) / 360); w++) {
        gl.uniform1f(this.loc.u_wrap, w);
        gl.drawArrays(gl.TRIANGLES, 0, this.count);
      }
    }
    gl.bindVertexArray(null);
  }
}
