// The station buildings on the map (T-2218, for now a plain floor and outline; the full drawing is
// T-2219, notes/T-2208.md 3): each outline a floor of very light warm white, the same in both themes as
// the boxes are, with one rim in the stations' ink, drawn by the network renderer with the car parks,
// under the track and the boxes. From zoom 13 it fades in to full at 14; a building with no station
// inside draws at half strength. The building being reshaped is left out (its tool draws it).
//
// Positions are kept in local units of the anchor they were built at, and the camera is made for that
// anchor at each draw, so a moved anchor needs no rebuild until the buildings change.

import { effect } from "@preact/signals";
import earcut from "earcut";
import { buildingAtPack, buildingDraft, buildingMembersById } from "../game/buildingView";
import { rgbf } from "../game/palette";
import { shadeColours } from "../game/shades";
import { lngLatToPack, packToLngLat } from "../game/coords";
import type { BuildingInput } from "../game/types";
import { world } from "../game/world";
import { anchorVersion } from "./anchorVersion";
import { anchor, toLngLat, toLocal, type Anchor } from "./geo";
import type { Overlay } from "./overlay";
import { compile, localCamera, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type Camera } from "./gl";
import { stationRing } from "./stationGeometry";

/** below this zoom no building is drawn or picked; full strength from FLOOR_FULL_ZOOM */
export const FLOOR_ZOOM = 13;
const FLOOR_FULL_ZOOM = 14;
/** the floor, #f9f7f3 */
const FLOOR_RGB: [number, number, number] = [249 / 255, 247 / 255, 243 / 255];

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

/** Keep the renderer's floors up to date with the world's buildings and the one being reshaped. */
export function startBuildingFloors(ov: Overlay) {
  effect(() => {
    anchorVersion.value;
    const bs = world.value?.save.buildings ?? [], members = buildingMembersById.value;
    const empty = new Set(bs.filter((b) => !(members.get(b.id)?.length)).map((b) => b.id));
    ov.renderer.floors.set(bs, empty, buildingDraft.value?.id ?? null);
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

export class BuildingFloors {
  private gl!: WebGL2RenderingContext;
  private fillProg!: WebGLProgram;
  private edgeProg!: WebGLProgram;
  private fu: Record<string, WebGLUniformLocation | null> = {};
  private eu: Record<string, WebGLUniformLocation | null> = {};
  private fillVao: WebGLVertexArrayObject | null = null;
  private edgeVao: WebGLVertexArrayObject | null = null;
  private bufs: WebGLBuffer[] = [];
  private fillCount = 0;
  private edgeCount = 0;
  private at: Anchor = anchor();
  /** what the buffers were built from, so an unchanged set uploads nothing */
  private key = "";
  private input: { buildings: readonly BuildingInput[]; empty: ReadonlySet<number>; skip: number | null } = { buildings: [], empty: new Set(), skip: null };

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    // again after a lost context: the old buffers are gone
    this.fillVao = this.edgeVao = null;
    this.bufs = [];
    this.key = "";
    this.fillProg = compile(gl, FILL_VS, FILL_FS);
    this.edgeProg = compile(gl, EDGE_VS, EDGE_FS);
    this.fu = uniforms(gl, this.fillProg, [...PROJECT_UNIFORMS, "u_rgb", "u_alpha"]);
    this.eu = uniforms(gl, this.edgeProg, [...PROJECT_UNIFORMS, "u_viewport", "u_half", "u_rgb", "u_alpha"]);
    this.upload();
  }

  /** The buildings to draw: those in `empty` (no station inside) at half strength, `skip` not at all. */
  set(buildings: readonly BuildingInput[], empty: ReadonlySet<number>, skip: number | null) {
    this.input = { buildings, empty, skip };
    if (this.gl) this.upload();
  }

  private upload() {
    const { buildings, empty, skip } = this.input;
    const at = anchor();
    const key = `${at.lng},${at.lat}|${skip}|${buildings.map((b) => `${b.id}:${empty.has(b.id) ? 0 : 1}:${b.corners.join()}`).join("|")}`;
    if (key === this.key) return;
    this.key = key;
    this.at = at;
    const gl = this.gl;
    if (this.fillVao) gl.deleteVertexArray(this.fillVao);
    if (this.edgeVao) gl.deleteVertexArray(this.edgeVao);
    for (const b of this.bufs) gl.deleteBuffer(b);
    this.fillVao = this.edgeVao = null;
    this.bufs = [];
    this.fillCount = this.edgeCount = 0;
    const verts: number[] = [], index: number[] = [], segs: number[] = [];
    for (const b of buildings) {
      if (b.id === skip) continue;
      const n = b.corners.length / 2;
      if (n < 3) continue;
      const strength = empty.has(b.id) ? 0.5 : 1;
      const local: number[] = [];
      for (let i = 0; i < n; i++) local.push(...toLocal(...packToLngLat(b.corners[2 * i], b.corners[2 * i + 1]), at));
      const base = verts.length / 3;
      for (let i = 0; i < n; i++) verts.push(local[2 * i], local[2 * i + 1], strength);
      for (const t of earcut(local)) index.push(base + t);
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        segs.push(local[2 * i], local[2 * i + 1], local[2 * j], local[2 * j + 1], strength);
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
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  /** Whether the floors are drawn (and picked) at a zoom. */
  static shownAt(zoom: number) {
    return zoom >= FLOOR_ZOOM;
  }

  /** the zoom of the last draw */
  zoom = 0;

  draw(cam: Camera, vp: readonly [number, number], dpr: number, zoom: number) {
    this.zoom = zoom;
    if (!this.fillCount || !this.fillVao || !this.edgeVao || zoom < FLOOR_ZOOM) return;
    const gl = this.gl, lc = localCamera(cam, this.at);
    const fade = Math.min(1, (zoom - FLOOR_ZOOM) / (FLOOR_FULL_ZOOM - FLOOR_ZOOM));
    gl.useProgram(this.fillProg);
    setCamera(gl, this.fu, lc);
    gl.uniform3fv(this.fu.u_rgb, FLOOR_RGB);
    gl.uniform1f(this.fu.u_alpha, fade * (zoom < FLOOR_FULL_ZOOM ? 0.6 : 1));
    gl.bindVertexArray(this.fillVao);
    gl.drawElements(gl.TRIANGLES, this.fillCount, gl.UNSIGNED_INT, 0);
    gl.useProgram(this.edgeProg);
    setCamera(gl, this.eu, lc);
    gl.uniform2f(this.eu.u_viewport, vp[0], vp[1]);
    gl.uniform1f(this.eu.u_half, (stationRing(zoom) * dpr) / 2);
    gl.uniform3fv(this.eu.u_rgb, rgbf(shadeColours().stationInk));
    gl.uniform1f(this.eu.u_alpha, fade);
    gl.bindVertexArray(this.edgeVao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.edgeCount);
    gl.bindVertexArray(null);
  }
}
