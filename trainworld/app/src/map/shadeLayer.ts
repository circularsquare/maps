// T-1025: the realm shade's triangles (boundaryGeometry.outsideZones) drawn as a MapLibre custom layer
// in one colour. A GeoJSON source would tile them, and tiling rounds every vertex to the tile's grid,
// which turns the thinnest triangles inside out; the fill bucket then takes such a triangle for a hole
// in the one before it and earcut joins the two with a long stray triangle. Here every triangle is
// drawn as cut. Positions are doubles split into two floats and taken relative to the view's centre in
// the shader, so the edges hold still at street zooms. On the globe (T-033) the triangles are cut
// smaller first, so their edges bend with it as the zone outlines do.
import type { CustomLayerInterface, Map as MlMap } from "maplibre-gl";
import { cameraFrom, compile, PROJECT_GLSL, uniforms } from "./gl";

const VERTEX = `#version 300 es
in vec2 a_hi;
in vec2 a_lo;
${PROJECT_GLSL}
uniform vec2 u_center_hi;
uniform vec2 u_center_lo;
uniform float u_wrap;
void main() {
  vec2 p = (a_hi - u_center_hi) + (a_lo - u_center_lo) + vec2(u_wrap, 0.0);
  gl_Position = project(p);
}`;
const FRAGMENT = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }`;

/** On the globe no triangle edge is longer than this, Mercator units (about 5.6 degrees of longitude):
 * its chord then stays within a third of a pixel of the sphere at world zoom. */
const GLOBE_EDGE = 1 / 64;

export class ShadeLayer implements CustomLayerInterface {
  readonly type = "custom";
  readonly renderingMode = "2d";
  private map: MlMap | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private buffer: WebGLBuffer | null = null;
  private loc: Record<string, WebGLUniformLocation | null> = {};
  private matrix = new Float32Array(16);
  private globeMatrix = new Float32Array(16);
  /** the triangles as set, Mercator x, y per corner */
  private xy: Float64Array<ArrayBufferLike> = new Float64Array(0);
  /** what the buffer holds: the triangles as set, or cut for the globe */
  private uploaded: "flat" | "globe" | null = null;
  private uploadedCount = 0;
  /** of those, the pole caps' vertices, last */
  private capCount = 0;
  /** vertices: three a triangle (the checks read it) */
  count = 0;
  /** premultiplied */
  private color: [number, number, number, number] = [0, 0, 0, 0];
  visible = true;

  constructor(readonly id: string) {}

  /** The triangles, Mercator x, y (0-1) per corner. */
  setTriangles(xy: Float64Array) {
    this.xy = xy;
    this.count = xy.length / 2;
    this.uploaded = null;
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
    const p = (this.program = compile(gl, VERTEX, FRAGMENT));
    this.loc = uniforms(gl, p, ["u_matrix", "u_globe", "u_globe_plane", "u_globe_at", "u_center_hi", "u_center_lo", "u_wrap", "u_color"]);
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
    this.uploaded = null;
  }

  onRemove(_map: MlMap, gl: WebGL2RenderingContext) {
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.buffer);
    gl.deleteVertexArray(this.vao);
    this.program = this.buffer = this.vao = null;
    this.map = this.gl = null;
  }

  render(gl: WebGL2RenderingContext, args: { defaultProjectionData: Parameters<typeof cameraFrom>[0] }) {
    if (!this.visible || !this.program || !this.map || !this.color[3]) return;
    const cam = cameraFrom(args.defaultProjectionData);
    const want = cam.globe ? "globe" : "flat";
    gl.bindVertexArray(this.vao);
    if (this.uploaded !== want) {
      let xy = this.xy;
      this.capCount = 0;
      if (cam.globe && xy.length) {
        // the realms never reach the poles, so the shade has their caps on the globe
        const cut = cutForGlobe(xy), caps = poleCaps();
        xy = new Float64Array(cut.length + caps.length);
        xy.set(cut);
        xy.set(caps, cut.length);
        this.capCount = caps.length / 2;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, split(xy), gl.STATIC_DRAW);
      this.uploaded = want;
      this.uploadedCount = xy.length / 2;
    }
    // the caps only on the whole globe: blending to flat, MapLibre's own poles fade out
    const count = this.uploadedCount - (cam.t >= 0.999 ? 0 : this.capCount);
    if (count > 0) {
      const m = cam.merc;
      const c = this.map.getCenter(), s = Math.sin((Math.max(-85.051129, Math.min(85.051129, c.lat)) * Math.PI) / 180);
      const cx = (c.lng + 180) / 360, cy = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
      // the camera's matrix with the centre moved to the origin, in doubles, then sent as floats
      const t = this.matrix;
      for (let i = 0; i < 16; i++) t[i] = m[i];
      for (let r = 0; r < 4; r++) t[12 + r] = m[r] * cx + m[4 + r] * cy + m[12 + r];
      gl.useProgram(this.program);
      gl.uniformMatrix4fv(this.loc.u_matrix, false, t);
      if (cam.globe) this.globeMatrix.set(cam.globe);
      gl.uniformMatrix4fv(this.loc.u_globe, false, this.globeMatrix);
      gl.uniform4fv(this.loc.u_globe_plane, cam.plane as Float32List);
      // project() takes positions relative to the centre in Mercator units here
      gl.uniform4f(this.loc.u_globe_at, cx, cy, 1, cam.globe ? cam.t : 0);
      const hx = Math.fround(cx), hy = Math.fround(cy);
      gl.uniform2f(this.loc.u_center_hi, hx, hy);
      gl.uniform2f(this.loc.u_center_lo, cx - hx, cy - hy);
      gl.uniform4f(this.loc.u_color, ...this.color);
      // the world copies in view; the globe has one
      const b = this.map.getBounds();
      const w0 = cam.globe ? 0 : Math.floor((b.getWest() + 180) / 360), w1 = cam.globe ? 0 : Math.floor((b.getEast() + 180) / 360);
      for (let w = w0; w <= w1; w++) {
        gl.uniform1f(this.loc.u_wrap, w);
        gl.drawArrays(gl.TRIANGLES, 0, count);
      }
    }
    gl.bindVertexArray(null);
  }
}

/** hi, lo, hi, lo (x then y) per vertex */
function split(xy: Float64Array): Float32Array {
  const n = xy.length / 2, d = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const x = xy[2 * i], y = xy[2 * i + 1], hx = Math.fround(x), hy = Math.fround(y);
    d[4 * i] = hx; d[4 * i + 1] = hy; d[4 * i + 2] = x - hx; d[4 * i + 3] = y - hy;
  }
  return d;
}

/** Triangles from the edges of Mercator's square to the poles, which `project` takes y < -1 and
 * y > 2 for. */
function poleCaps(): number[] {
  const out: number[] = [];
  for (let i = 0; i < 1 / GLOBE_EDGE; i++) {
    const x0 = i * GLOBE_EDGE, x1 = x0 + GLOBE_EDGE, xm = x0 + GLOBE_EDGE / 2;
    out.push(x0, 0, x1, 0, xm, -4, x0, 1, x1, 1, xm, 5);
  }
  return out;
}

/** The triangles halved across their longest edge until none is longer than GLOBE_EDGE. */
export function cutForGlobe(xy: Float64Array): Float64Array {
  const out: number[] = [];
  const cut = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => {
    const ab = (bx - ax) ** 2 + (by - ay) ** 2, bc = (cx - bx) ** 2 + (cy - by) ** 2, ca = (ax - cx) ** 2 + (ay - cy) ** 2;
    const most = Math.max(ab, bc, ca);
    if (most <= GLOBE_EDGE * GLOBE_EDGE) return void out.push(ax, ay, bx, by, cx, cy);
    if (most === ab) {
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      cut(ax, ay, mx, my, cx, cy);
      cut(mx, my, bx, by, cx, cy);
    } else if (most === bc) {
      const mx = (bx + cx) / 2, my = (by + cy) / 2;
      cut(ax, ay, bx, by, mx, my);
      cut(ax, ay, mx, my, cx, cy);
    } else {
      const mx = (cx + ax) / 2, my = (cy + ay) / 2;
      cut(ax, ay, bx, by, mx, my);
      cut(mx, my, bx, by, cx, cy);
    }
  };
  for (let i = 0; i + 5 < xy.length; i += 6) cut(xy[i], xy[i + 1], xy[i + 2], xy[i + 3], xy[i + 4], xy[i + 5]);
  return Float64Array.from(out);
}
