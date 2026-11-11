// The station buildings on the map (T-2218, T-2219; notes/T-2208.md 3 "The drawing"), drawn by the
// network renderer from the network's buildings (map/buildingNet.ts):
// - under the track, with the car parks (the stroke cache's layer): the floor, very light warm white in
//   both themes as the boxes are white in both; the little shops on it; one rim in the stations' ink.
//   From zoom 13 the floor and rim fade in, the floor at 60% until 14; shops fade in from 15 to 15.5 and
//   their glyphs from 17 where a shop is at least SHOP_GLYPH_PX across. A building with no station
//   inside draws at half strength with no shops; the one being reshaped is left to its tool.
// - each frame, over the track and under the boxes: the hall's crowd, a stipple on the free floor
//   nearest the platforms spilling, red once the hall is full; and a selected building's outline and
//   glow (the selected stations' own, SEL_FS, go round boxes only).
// - the floors of buildings with stations inside mark the stencil (STENCIL_BIT), so their members'
//   rims are not drawn on the floor: building and boxes are one white shape with one outline.

import { effect } from "@preact/signals";
import earcut from "earcut";
import { buildingAtPack, buildingDraft, hallWaiting } from "../game/buildingView";
import { SHOP_COLOURS, HALL_CELL_M } from "../game/buildingLayout";
import { BAD_RED, rgb, rgbf } from "../game/palette";
import { shadeColours } from "../game/shades";
import { shade } from "../game/ui";
import { lngLatToPack } from "../game/coords";
import type { BuildingInput } from "../game/types";
import { world } from "../game/world";
import { toLngLat } from "./geo";
import type { Overlay } from "./overlay";
import { CELL_STRIDE, SHOP_STRIDE, type BuildingDraw } from "./buildingNet";
import { compileOptional, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type LocalCamera } from "./gl";
import { SEL_GLOW_PX, SEL_OUTLINE_PX, stationRing, type Project } from "./stationGeometry";

/** below this zoom no building is drawn or picked; full strength from FLOOR_FULL_ZOOM */
export const FLOOR_ZOOM = 13;
const FLOOR_FULL_ZOOM = 14;
/** shops fade in over these zooms, their glyphs over the next */
const SHOP_ZOOM: [number, number] = [15, 15.5], GLYPH_ZOOM: [number, number] = [16.8, 17];
/** a shop shows its glyph only when at least this many CSS px across */
const SHOP_GLYPH_PX = 9;
/** the floor, #f9f7f3 */
const FLOOR_RGB: [number, number, number] = [249 / 255, 247 / 255, 243 / 255];
/** the rim is this many station rims wide, half of it over the floor; wider in the dark theme, where
 * the stations' ink is a mid grey that a thinner rim loses against the floor */
const rimOf = () => (shade.peek() === "dark" ? 2 : 1.5);
/** the stencil bit the floors mark (the track through the boxes uses the bits below it) */
export const STENCIL_BIT = 0x80;
/** corners a selected outline is drawn with at most (the tool allows 24) */
const SEL_MAX = 32;

const FILL_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
in vec3 a_p;       // local x, y, strength
out float v_a;
void main() {
  v_a = a_p.z;
  gl_Position = project(a_p.xy);
}`;

const FILL_FS = `#version 300 es
precision highp float;
uniform vec3 u_rgb;
uniform float u_alpha;
in float v_a;
out vec4 o;
void main() {
  float a = v_a * u_alpha;
  o = vec4(u_rgb * a, a);
}`;

const EDGE_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform float u_half;    // half the rim's width, device px
in vec4 a_seg;           // local a, b
in float a_strength;
out float v_a;
out float v_d;           // device px across the rim from its middle
void main() {
  vec2 half_vp = u_viewport * 0.5;
  vec4 a = project(a_seg.xy), b = project(a_seg.zw);
  vec2 sa = a.xy / a.w * half_vp, sb = b.xy / b.w * half_vp;
  vec2 d = sb - sa;
  float len = length(d);
  if (len < 1e-6) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 u = d / len, n = vec2(-u.y, u.x);
  float w = u_half + 1.0;
  // square ends, so two edges meeting at a corner close it
  float along = (gl_VertexID & 1) == 0 ? -u_half : len + u_half;
  float across = gl_VertexID < 2 ? -w : w;
  vec2 s = sa + u * along + n * across;
  v_d = across;
  v_a = a_strength;
  gl_Position = vec4(s / half_vp * a.w, a.z, a.w);
}`;

const EDGE_FS = `#version 300 es
precision highp float;
uniform vec3 u_rgb;
uniform float u_alpha;
uniform float u_half;
in float v_a;
in float v_d;
out vec4 o;
void main() {
  float cover = clamp(u_half + 0.5 - abs(v_d), 0.0, 1.0);
  float a = cover * v_a * u_alpha;
  if (a <= 0.0) discard;
  o = vec4(u_rgb * a, a);
}`;

// A quad in a frame on the ground (a shop, a crowd cell): its middle, along it as a vector of half its
// length, and half its depth across, local units
const QUAD_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
in vec4 a_quad;          // middle, half its length along it (vector)
in vec4 a_dim;           // half depth (local), then per kind: a shop's colour and glyph; a cell's full flag
out vec2 v_p;            // in its frame, device px
out vec2 v_s;            // from its middle on the screen, device px, y up
flat out vec2 v_half;    // half its size, device px
flat out vec3 v_meta;
void main() {
  vec2 half_vp = u_viewport * 0.5;
  vec4 c = project(a_quad.xy);
  vec2 n = length(a_quad.zw) > 0.0 ? normalize(vec2(-a_quad.w, a_quad.z)) * a_dim.x : vec2(0.0);
  vec4 e = project(a_quad.xy + a_quad.zw), q = project(a_quad.xy + n);
  vec2 cs = c.xy / c.w * half_vp;
  vec2 axis = e.xy / e.w * half_vp - cs, nrm = q.xy / q.w * half_vp - cs;
  float len = length(axis), dep = length(nrm);
  if (len < 1e-6 || dep < 1e-6) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 along = axis / len, across = nrm / dep;
  v_half = vec2(len, dep);
  v_meta = a_dim.yzw;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  v_p = corner * (v_half + 1.0);
  vec2 off = along * v_p.x + across * v_p.y;
  v_s = off;
  c.xy += off / half_vp * c.w;
  gl_Position = c;
}`;

// A shop: its colour, a rim 16% darker, and from zoom 17 a glyph in the rim's colour, upright on the
// screen: a cup, a bag, a bowl or a book
const SHOP_FS = `#version 300 es
precision highp float;
uniform vec3 u_fill[${SHOP_COLOURS.length}];
uniform float u_alpha;
uniform float u_glyph;   // 0-1: glyphs shown
uniform float u_dpr;
in vec2 v_p;
in vec2 v_s;
flat in vec2 v_half;
flat in vec3 v_meta;
out vec4 o;
float box(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float ring(vec2 p, float r, float w) { return abs(length(p) - r) - w; }
float glyph(vec2 p, int k) {
  if (k == 0) {
    // a cup: its body and a handle
    float body = box(p - vec2(-0.15, -0.05), vec2(0.42, 0.5));
    float handle = max(ring(p - vec2(0.3, 0.0), 0.25, 0.09), 0.3 - p.x);
    return min(body, handle);
  }
  if (k == 1) {
    // a bag: its body and a handle over it
    float body = box(p - vec2(0.0, -0.2), vec2(0.6, 0.48));
    float handle = max(ring(p - vec2(0.0, 0.28), 0.28, 0.08), 0.28 - p.y);
    return min(body, handle);
  }
  if (k == 2) {
    // a bowl: the lower half of a disc under a rim
    float bowl = max(length(p - vec2(0.0, 0.1)) - 0.7, p.y - 0.1);
    float rim = box(p - vec2(0.0, 0.14), vec2(0.82, 0.07));
    return min(bowl, rim);
  }
  // a book, open: two pages
  return min(box(p - vec2(-0.34, 0.0), vec2(0.3, 0.5)), box(p - vec2(0.34, 0.0), vec2(0.3, 0.5)));
}
void main() {
  float d = box(v_p, v_half);
  float cover = clamp(0.5 - d, 0.0, 1.0);
  if (cover <= 0.0) discard;
  int c = int(v_meta.x + 0.5);
  vec3 fill = u_fill[c], rim = fill * 0.84;
  float edge = clamp(d + max(1.0, 0.75 * u_dpr) + 0.5, 0.0, 1.0);
  vec3 col = mix(fill, rim, edge);
  float side = 2.0 * min(v_half.x, v_half.y);
  if (u_glyph > 0.0 && side >= ${SHOP_GLYPH_PX.toFixed(1)} * u_dpr) {
    float r = 0.32 * side;
    float g = glyph(v_s / r, int(v_meta.y + 0.5)) * r;
    col = mix(col, rim * 0.92, u_glyph * clamp(0.5 - g, 0.0, 1.0));
  }
  float a = cover * u_alpha;
  o = vec4(col * a, a);
}`;

// The hall's crowd: 2 px dots on a 4 px grid on the screen, so it reads as people, not as building
const CELL_FS = `#version 300 es
precision highp float;
uniform vec3 u_rgb;
uniform vec3 u_bad;
uniform float u_dpr;
uniform float u_alpha;
in vec2 v_p;
in vec2 v_s;
flat in vec2 v_half;
flat in vec3 v_meta;
out vec4 o;
void main() {
  if (abs(v_p.x) > v_half.x + 0.5 || abs(v_p.y) > v_half.y + 0.5) discard;
  vec2 g = mod(gl_FragCoord.xy, 4.0 * u_dpr) - 2.0 * u_dpr;
  float a = clamp(1.0 * u_dpr + 0.5 - length(g), 0.0, 1.0) * u_alpha;
  if (a <= 0.0) discard;
  o = vec4((v_meta.x > 0.5 ? u_bad : u_rgb) * a, a);
}`;

// A selected building: an ink outline round its rim and a glow past it, as the selected stations' (SEL_FS)
const SEL_VS = `#version 300 es
precision highp float;
uniform vec4 u_bbox;     // CSS px, y down: left, top, right, bottom
uniform vec2 u_css;
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), gl_VertexID < 2 ? 0.0 : 1.0);
  vec2 s = mix(u_bbox.xy, u_bbox.zw, corner);
  gl_Position = vec4(s.x / u_css.x * 2.0 - 1.0, 1.0 - s.y / u_css.y * 2.0, 0.0, 1.0);
}`;

const SEL_FS = `#version 300 es
precision highp float;
uniform vec2 u_pts[${SEL_MAX}]; // the outline, CSS px, y down
uniform int u_n;
uniform float u_dpr;
uniform float u_height;  // the canvas, device px
uniform vec3 u_ink;
uniform float u_rim;     // the rim's reach outside the outline, CSS px
uniform float u_outline;
uniform float u_glow;
out vec4 o;
void main() {
  vec2 p = vec2(gl_FragCoord.x, u_height - gl_FragCoord.y) / u_dpr;
  float d = 1e9;
  bool odd = false;
  for (int i = 0; i < ${SEL_MAX}; i++) {
    if (i >= u_n) break;
    vec2 a = u_pts[i], b = u_pts[i + 1 < u_n ? i + 1 : 0];
    vec2 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-9), 0.0, 1.0);
    d = min(d, length(a + ab * t - p));
    if ((a.y > p.y) != (b.y > p.y) && p.x < a.x + (b.x - a.x) * (p.y - a.y) / (b.y - a.y)) odd = !odd;
  }
  if (odd) discard;
  d -= u_rim;
  float line = clamp(u_outline + 0.5 - d, 0.0, 1.0);
  float t = clamp((d - u_outline) / u_glow, 0.0, 1.0);
  float s = max(line, 0.35 * (1.0 - t) * (1.0 - t));
  o = vec4(u_ink * s, s);
}`;

/** Keep the renderer told which building is being reshaped (its tool draws it). */
export function startBuildingFloors(ov: Overlay) {
  effect(() => {
    ov.renderer.floors.setSkip(buildingDraft.value?.id ?? null);
    ov.invalidate();
  });
}

/** The building whose floor is drawn under a screen point (CSS px in the map), or null. */
export function floorAt(ov: Overlay, px: number, py: number): BuildingInput | null {
  if (!BuildingFloors.shownAt(ov.renderer.floors.zoom) || !world.peek()?.save.buildings?.length) return null;
  const here = ov.screenToLocal(px, py);
  if (!here) return null;
  const q = lngLatToPack(...toLngLat(here[0], here[1]));
  return buildingAtPack(q.x, q.y);
}

/** The hall as the platform crowds last worked it out (platformCrowds.ts), per building of the network. */
export interface Hall { id: number; waiting: number; full: boolean; spill: ReadonlySet<number> }

export class BuildingFloors {
  private gl!: WebGL2RenderingContext;
  private fillProg!: WebGLProgram;
  private edgeProg!: WebGLProgram;
  private shopProg!: WebGLProgram;
  private cellProg!: WebGLProgram;
  private selProg!: WebGLProgram;
  private fu: Record<string, WebGLUniformLocation | null> = {};
  private eu: Record<string, WebGLUniformLocation | null> = {};
  private su: Record<string, WebGLUniformLocation | null> = {};
  private cu: Record<string, WebGLUniformLocation | null> = {};
  private xu: Record<string, WebGLUniformLocation | null> = {};
  private fillVao: WebGLVertexArrayObject | null = null;
  private edgeVao: WebGLVertexArrayObject | null = null;
  private shopVao: WebGLVertexArrayObject | null = null;
  private cellVao: WebGLVertexArrayObject | null = null;
  private cellBuf: WebGLBuffer | null = null;
  private bufs: WebGLBuffer[] = [];
  private fillCount = 0;
  /** the floors of buildings with members: the first `markCount` indices of the fill */
  private markCount = 0;
  private edgeCount = 0;
  private shopCount = 0;
  private cellCount = 0;
  private buildings: readonly BuildingDraw[] = [];
  private skip: number | null = null;
  /** the hall last drawn, to upload the crowd's cells only when it changed */
  private hallKey = "";
  private shopColours = new Float32Array(SHOP_COLOURS.flatMap((h) => rgb(h).map((v) => v / 255)));
  private selPts = new Float32Array(SEL_MAX * 2);
  /** the browser refused a program: nothing drawn */
  private off = false;

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    // again after a lost context: the old buffers are gone
    this.fillVao = this.edgeVao = this.shopVao = this.cellVao = this.cellBuf = null;
    this.bufs = [];
    this.hallKey = "";
    const progs = [
      compileOptional(gl, FILL_VS, FILL_FS, "building floors"),
      compileOptional(gl, EDGE_VS, EDGE_FS, "building floors"),
      compileOptional(gl, QUAD_VS, SHOP_FS, "building floors"),
      compileOptional(gl, QUAD_VS, CELL_FS, "building floors"),
      compileOptional(gl, SEL_VS, SEL_FS, "building floors"),
    ];
    // none of the buildings' layers where one is refused (T-3192)
    this.off = progs.some((p) => !p);
    if (this.off) return;
    [this.fillProg, this.edgeProg, this.shopProg, this.cellProg, this.selProg] = progs as WebGLProgram[];
    this.fu = uniforms(gl, this.fillProg, [...PROJECT_UNIFORMS, "u_rgb", "u_alpha"]);
    this.eu = uniforms(gl, this.edgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_half", "u_rgb", "u_alpha"]);
    this.su = uniforms(gl, this.shopProg, [...PROJECT_UNIFORMS, "u_viewport", "u_fill", "u_alpha", "u_glyph", "u_dpr"]);
    this.cu = uniforms(gl, this.cellProg, [...PROJECT_UNIFORMS, "u_viewport", "u_rgb", "u_bad", "u_dpr", "u_alpha"]);
    this.xu = uniforms(gl, this.selProg, ["u_bbox", "u_css", "u_pts", "u_n", "u_dpr", "u_height", "u_ink", "u_rim", "u_outline", "u_glow"]);
    this.upload();
  }

  /** The network's buildings (renderer.ts, on each network). */
  setNetwork(buildings: readonly BuildingDraw[] | undefined) {
    this.buildings = buildings ?? [];
    this.hallKey = "";
    if (this.gl) this.upload();
  }

  /** The building being reshaped, left to its tool. */
  setSkip(id: number | null) {
    if (id === this.skip) return;
    this.skip = id;
    if (this.gl) this.upload();
  }

  /** The buildings drawn (all but the one being reshaped). */
  get drawn(): readonly BuildingDraw[] {
    return this.skip === null ? this.buildings : this.buildings.filter((b) => b.id !== this.skip);
  }

  private upload() {
    const gl = this.gl;
    for (const v of [this.fillVao, this.edgeVao, this.shopVao, this.cellVao]) if (v) gl.deleteVertexArray(v);
    for (const b of this.bufs) gl.deleteBuffer(b);
    if (this.cellBuf) gl.deleteBuffer(this.cellBuf);
    this.fillVao = this.edgeVao = this.shopVao = this.cellVao = this.cellBuf = null;
    this.bufs = [];
    this.fillCount = this.markCount = this.edgeCount = this.shopCount = this.cellCount = 0;
    this.hallKey = "";
    const bs = this.drawn;
    if (!bs.length || this.off) return;
    const verts: number[] = [], index: number[] = [], segs: number[] = [], shops: number[] = [];
    // those with members first, so their floors are the first `markCount` indices
    const order = [...bs].sort((a, b) => Number(!a.members.length) - Number(!b.members.length));
    for (const b of order) {
      const n = b.corners.length / 2;
      if (n < 3) continue;
      const strength = b.members.length ? 1 : 0.5;
      const base = verts.length / 3;
      for (let i = 0; i < n; i++) verts.push(b.corners[2 * i], b.corners[2 * i + 1], strength);
      for (const t of earcut(Array.from(b.corners))) index.push(base + t);
      if (b.members.length) this.markCount = index.length;
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        segs.push(b.corners[2 * i], b.corners[2 * i + 1], b.corners[2 * j], b.corners[2 * j + 1], strength);
      }
      const s = b.shops;
      for (let k = 0; k + SHOP_STRIDE <= s.length; k += SHOP_STRIDE) {
        const hl = s[k + 2] * b.perM, hd = s[k + 3] * b.perM;
        shops.push(s[k], s[k + 1], b.ux * hl, b.uy * hl, hd, s[k + 4], s[k + 5], 0);
      }
    }
    if (!index.length) return;
    const buf = (data: ArrayBufferView, target: number = gl.ARRAY_BUFFER) => {
      const b = gl.createBuffer()!;
      gl.bindBuffer(target, b);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      this.bufs.push(b);
      return b;
    };
    this.fillVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.fillVao);
    buf(new Float32Array(verts));
    const lp = gl.getAttribLocation(this.fillProg, "a_p");
    gl.enableVertexAttribArray(lp);
    gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 12, 0);
    buf(new Uint32Array(index), gl.ELEMENT_ARRAY_BUFFER);
    this.fillCount = index.length;
    gl.bindVertexArray(null);
    this.edgeVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.edgeVao);
    buf(new Float32Array(segs));
    const ls = gl.getAttribLocation(this.edgeProg, "a_seg"), la = gl.getAttribLocation(this.edgeProg, "a_strength");
    gl.enableVertexAttribArray(ls);
    gl.vertexAttribPointer(ls, 4, gl.FLOAT, false, 20, 0);
    gl.vertexAttribDivisor(ls, 1);
    gl.enableVertexAttribArray(la);
    gl.vertexAttribPointer(la, 1, gl.FLOAT, false, 20, 16);
    gl.vertexAttribDivisor(la, 1);
    this.edgeCount = segs.length / 5;
    gl.bindVertexArray(null);
    if (shops.length) {
      this.shopVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.shopVao);
      buf(new Float32Array(shops));
      this.quadAttribs(this.shopProg);
      this.shopCount = shops.length / 8;
      gl.bindVertexArray(null);
    }
    this.cellVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.cellVao);
    this.cellBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cellBuf);
    gl.bufferData(gl.ARRAY_BUFFER, 32, gl.DYNAMIC_DRAW);
    this.quadAttribs(this.cellProg);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** a_quad and a_dim, 8 floats an instance, from the bound buffer */
  private quadAttribs(prog: WebGLProgram) {
    const gl = this.gl;
    for (const [name, offset] of [["a_quad", 0], ["a_dim", 16]] as const) {
      const loc = gl.getAttribLocation(prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 32, offset);
      gl.vertexAttribDivisor(loc, 1);
    }
  }

  /** Whether the floors are drawn (and picked) at a zoom. */
  static shownAt(zoom: number) {
    return zoom >= FLOOR_ZOOM;
  }

  /** the zoom of the last draw */
  zoom = 0;

  /** The floors, shops and rims, under the track (in the stroke cache's layer). */
  draw(lc: LocalCamera, vp: readonly [number, number], dpr: number, zoom: number) {
    this.zoom = zoom;
    if (!this.fillCount || !this.fillVao || !this.edgeVao || zoom < FLOOR_ZOOM) return;
    const gl = this.gl;
    const fade = Math.min(1, (zoom - FLOOR_ZOOM) / (FLOOR_FULL_ZOOM - FLOOR_ZOOM));
    gl.useProgram(this.fillProg);
    setCamera(gl, this.fu, lc);
    gl.uniform3fv(this.fu.u_rgb, FLOOR_RGB);
    gl.uniform1f(this.fu.u_alpha, fade * (zoom < FLOOR_FULL_ZOOM ? 0.6 : 1));
    gl.bindVertexArray(this.fillVao);
    gl.drawElements(gl.TRIANGLES, this.fillCount, gl.UNSIGNED_INT, 0);
    const shopsOn = smooth(SHOP_ZOOM, zoom);
    if (this.shopCount && this.shopVao && shopsOn > 0) {
      gl.useProgram(this.shopProg);
      setCamera(gl, this.su, lc);
      gl.uniform2f(this.su.u_viewport, vp[0], vp[1]);
      gl.uniform3fv(this.su.u_fill, this.shopColours);
      gl.uniform1f(this.su.u_alpha, shopsOn);
      gl.uniform1f(this.su.u_glyph, smooth(GLYPH_ZOOM, zoom));
      gl.uniform1f(this.su.u_dpr, dpr);
      gl.bindVertexArray(this.shopVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.shopCount);
    }
    gl.useProgram(this.edgeProg);
    setCamera(gl, this.eu, lc);
    gl.uniform2f(this.eu.u_viewport, vp[0], vp[1]);
    gl.uniform1f(this.eu.u_half, (rimOf() * stationRing(zoom) * dpr) / 2);
    gl.uniform3fv(this.eu.u_rgb, rgbf(shadeColours().stationInk));
    gl.uniform1f(this.eu.u_alpha, fade);
    gl.bindVertexArray(this.edgeVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.edgeCount);
    gl.bindVertexArray(null);
  }

  /** Mark the floors of buildings with stations inside in the stencil's STENCIL_BIT, nothing drawn;
   * false when there are none. The caller has cleared the stencil. */
  markStencil(lc: LocalCamera): boolean {
    if (!this.markCount || !this.fillVao) return false;
    const gl = this.gl;
    gl.enable(gl.STENCIL_TEST);
    gl.stencilMask(STENCIL_BIT);
    gl.stencilFunc(gl.ALWAYS, STENCIL_BIT, STENCIL_BIT);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
    gl.colorMask(false, false, false, false);
    gl.useProgram(this.fillProg);
    setCamera(gl, this.fu, lc);
    gl.uniform3fv(this.fu.u_rgb, FLOOR_RGB);
    gl.uniform1f(this.fu.u_alpha, 1);
    gl.bindVertexArray(this.fillVao);
    gl.drawElements(gl.TRIANGLES, this.markCount, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
    gl.colorMask(true, true, true, true);
    gl.stencilMask(0xff);
    gl.disable(gl.STENCIL_TEST);
    return true;
  }

  /** The hall's crowd, each frame from zoom 14 while platform crowds are shown (`hall` null: not). */
  drawHall(lc: LocalCamera, vp: readonly [number, number], dpr: number, zoom: number, hall: readonly Hall[] | null) {
    const shown = !!hall && zoom >= FLOOR_FULL_ZOOM;
    publishHall(shown ? hall : null);
    if (!shown || !this.cellVao || !this.cellBuf) return;
    const key = hall.map((h) => `${h.id}:${Math.round(h.waiting)}:${h.full ? 1 : 0}:${[...h.spill].join(",")}`).join("|");
    if (key !== this.hallKey) {
      this.hallKey = key;
      const out: number[] = [];
      const byId = new Map(hall.map((h) => [h.id, h]));
      for (const b of this.drawn) {
        const h = byId.get(b.id), c = b.cells, total = c.length / CELL_STRIDE;
        if (!h || h.waiting < 1 || !total || !b.room) continue;
        const take = Math.min(total, Math.max(1, Math.round((total * h.waiting) / b.room)));
        // the cells nearest the platforms spilling first, nearest first (`cells` is in that order)
        const order: number[] = [];
        for (let k = 0; k < total; k++) if (h.spill.has(c[k * CELL_STRIDE + 2])) order.push(k);
        for (let k = 0; k < total && order.length < take; k++) if (!h.spill.has(c[k * CELL_STRIDE + 2])) order.push(k);
        const half = (HALL_CELL_M / 2) * b.perM;
        for (let j = 0; j < take && j < order.length; j++) {
          const k = order[j] * CELL_STRIDE;
          out.push(c[k], c[k + 1], b.ux * half, b.uy * half, half, h.full ? 1 : 0, 0, 0);
        }
      }
      this.cellCount = out.length / 8;
      if (this.cellCount) {
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.cellBuf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(out), gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
      }
    }
    if (!this.cellCount) return;
    const gl = this.gl, sh = shadeColours();
    gl.useProgram(this.cellProg);
    setCamera(gl, this.cu, lc);
    gl.uniform2f(this.cu.u_viewport, vp[0], vp[1]);
    gl.uniform3fv(this.cu.u_rgb, rgbf(sh.walkers));
    gl.uniform3fv(this.cu.u_bad, rgbf(BAD_RED));
    gl.uniform1f(this.cu.u_dpr, dpr);
    gl.uniform1f(this.cu.u_alpha, 1);
    gl.bindVertexArray(this.cellVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.cellCount);
    gl.bindVertexArray(null);
  }

  /** The outline and glow round each building with a selected station (`selected`, station indices). */
  drawSelected(selected: ReadonlySet<number>, project: Project, vp: readonly [number, number], dpr: number, zoom: number, ink: Float32Array) {
    if (!selected.size || zoom < FLOOR_FULL_ZOOM || this.off) return;
    const gl = this.gl, w = vp[0] / dpr, h = vp[1] / dpr, u = this.xu;
    const rim = (rimOf() * stationRing(zoom)) / 2, reach = rim + SEL_OUTLINE_PX + SEL_GLOW_PX + 1;
    for (const b of this.drawn) {
      if (!b.members.some((i) => selected.has(i))) continue;
      const n = Math.min(SEL_MAX, b.corners.length / 2);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < n; i++) {
        const [x, y] = project(b.corners[2 * i], b.corners[2 * i + 1]);
        if (!Number.isFinite(x + y)) return;
        this.selPts[2 * i] = x;
        this.selPts[2 * i + 1] = y;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      x0 = Math.max(0, x0 - reach); y0 = Math.max(0, y0 - reach); x1 = Math.min(w, x1 + reach); y1 = Math.min(h, y1 + reach);
      if (x1 <= x0 || y1 <= y0) continue;
      gl.useProgram(this.selProg);
      gl.uniform4f(u.u_bbox, x0, y0, x1, y1);
      gl.uniform2f(u.u_css, w, h);
      gl.uniform2fv(u.u_pts, this.selPts);
      gl.uniform1i(u.u_n, n);
      gl.uniform1f(u.u_dpr, dpr);
      gl.uniform1f(u.u_height, vp[1]);
      gl.uniform3fv(u.u_ink, ink);
      gl.uniform1f(u.u_rim, rim);
      gl.uniform1f(u.u_outline, SEL_OUTLINE_PX);
      gl.uniform1f(u.u_glow, SEL_GLOW_PX);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }
}

function smooth([a, b]: [number, number], z: number) {
  const t = Math.min(1, Math.max(0, (z - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** The riders in each hall for the panels and hovers (buildingView.ts `hallWaiting`), written when a
 * rounded count changes. */
let published = "";
function publishHall(hall: readonly Hall[] | null) {
  const key = hall ? hall.map((h) => `${h.id}:${Math.round(h.waiting)}`).join(",") : "";
  if (key === published) return;
  published = key;
  hallWaiting.value = new Map((hall ?? []).map((h) => [h.id, Math.round(h.waiting)]));
}
