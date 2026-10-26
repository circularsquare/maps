// How bus lines look on the map (T-1464 stage B1, notes/T-1445.md "What the player sees and pays").
// A bus line has no track, so the network buffers hold no strokes for it; this layer draws its drawn
// route from `lineStats[id].bus` instead, in the renderer's own pass (renderer.ts calls `draw` before
// the track): half a rail line's width in the line's colour on the casing, a planned route as rail
// blueprint is, the 400 m corridor of the hovered or selected line shaded under everything, small
// rings for its street stops from zoom 14 and rounded squares for its ends. A stop at a train station
// draws nothing: the station's own circle is there.
//
// Cost: the geometry is built once per world change and anchor move and kept on the GPU; a frame with
// no bus line returns at the first line of `draw`. Picking and the tags read the same arrays
// (`lineAt`, `stopAt`), on pointer moves only.

import { rgb } from "../game/palette";
import type { ShadeColours } from "../game/shadeTable";
import type { WorldView } from "../game/types";
import { anchor, kAtY, pxPerUnitAt, toLocal, type Anchor } from "./geo";
import { compile, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type LocalCamera } from "./gl";
import type { Overlay } from "./overlay";
import { topSeg } from "./pick";
import { factRow, lineBullet } from "./trackTagRows";
import type { TipLine } from "./networkTips";

/** Street stops are drawn from this zoom, ends from `END_ZOOM`. */
export const BUS_RING_ZOOM = 14;
export const BUS_END_ZOOM = 12;
/** A bus line is this share of a rail line's width, its casing this share of a rail line's. */
const WIDTH_SHARE = 0.5;
const CASING_SHARE = 0.6;
/** Ground metres each side of the route that its riders walk from (demand's corridor). */
export const CORRIDOR_M = 400;
const CORRIDOR_ALPHA = 0.16;
const HOT_MIX = 0.3;
/** Bus stop sizes as shares of the bus line's width, and the least radius, device px */
const RING_SHARE = 1.15, END_SHARE = 1.5, MIN_RING_PX = 2.6;
const PICK_SLOP_PX = 6;

const FLAG_BLUEPRINT = 1;

/** A bus line as the world gives it (lng, lat pairs flat). */
interface BusSource {
  id: number;
  colour: string;
  built: boolean;
  route: number[];
  street: number[];
  /** the stops demand sees, as lng, lat, kind */
  stops: number[];
}

/** The sources' geometry in local units of one anchor. */
interface Built {
  a: Anchor;
  /** per route segment: x0, y0, x1, y1 */
  seg: Float32Array;
  /** per route segment: line id, flags, its colour as r * 65536 + g * 256 + b */
  info: Float32Array;
  count: number;
  /** per stop: x, y, 1 a ring + 2 a blueprint's, its colour packed as `info` */
  stops: Float32Array;
  stopCount: number;
  stopLine: Int32Array;
  /** per stop 1 for an end */
  stopEnd: Uint8Array;
  /** per line id, local units a ground metre spans there */
  k: Map<number, number>;
}

/** What the last draw drew, CSS px, for the hit tests; null when it drew nothing. */
interface Drawn { rings: boolean; ends: boolean; ringR: number; endR: number }

const SEG_VS = `#version 300 es
precision highp float;
precision highp int;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform float u_width;       // device px across (the corridor's whole width in pass 2)
uniform float u_extra;       // added to the selected line
uniform int u_pass;          // 0 casing, 1 colour, 2 corridor
uniform int u_sel;
uniform int u_hot;
uniform int u_only;          // -1 every line, -2 the selected line, else that line only
uniform float u_dim;
uniform float u_alpha;       // the corridor's
uniform vec3 u_casing_rgb;
uniform vec3 u_rail_rgb;
uniform float u_rail;        // a blueprint's rail beyond the colour, device px a side
in vec4 a_seg;
in vec3 a_info;
out vec3 v_color;
out float v_alpha;
out vec4 v_cap;
void main() {
  int id = int(a_info.x + 0.5);
  bool bp = (int(a_info.y + 0.5) & ${FLAG_BLUEPRINT}) != 0;
  bool sel = id == u_sel;
  if ((u_only >= 0 && id != u_only) || (u_only == -2 && !sel)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float p = a_info.z;
  vec3 col = vec3(floor(p / 65536.0), mod(floor(p / 256.0), 256.0), mod(p, 256.0)) / 255.0;
  float w = u_width + (sel && u_pass < 2 ? u_extra : 0.0);
  if (u_pass == 0 && bp) w += 2.0 * u_rail;
  float faint = 1.0;
  if (u_pass < 2) {
    // thinner than a pixel (zoomed far out): a pixel wide and fainter, as the rail strokes are
    faint = clamp(w, 0.05, 1.0);
    w = max(w, 1.0);
  }
  float along = float(gl_VertexID & 1);
  float side = gl_VertexID < 2 ? -1.0 : 1.0;
  vec4 c0 = project(a_seg.xy);
  vec4 c1 = project(a_seg.zw);
  vec2 half_vp = u_viewport * 0.5;
  vec2 s0 = c0.xy / c0.w * half_vp;
  vec2 s1 = c1.xy / c1.w * half_vp;
  vec2 dir = s1 - s0;
  float len = length(dir);
  dir = len > 1e-6 ? dir / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec4 c = along < 0.5 ? c0 : c1;
  vec2 off = nrm * side * w * 0.5 + dir * (along * 2.0 - 1.0) * w * 0.5;
  c.xy += off / half_vp * c.w;
  gl_Position = c;
  float ext = (along * 2.0 - 1.0) * w * 0.5, ua = along * len + ext;
  v_cap = vec4(ua, len - ua, side * w * 0.5, w * 0.5);
  v_alpha = faint * (u_pass == 2 ? u_alpha : (sel ? 1.0 : u_dim));
  if (u_pass == 0) v_color = bp ? u_rail_rgb : u_casing_rgb;
  else {
    v_color = col;
    if (u_pass == 1) {
      if (bp) v_color = mix(v_color, vec3(1.0), 0.45);
      if (id == u_hot) v_color = mix(v_color, vec3(1.0), ${HOT_MIX});
    }
  }
}`;

const SEG_FS = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_alpha;
in vec4 v_cap;
out vec4 o;
void main() {
  float past = -min(v_cap.x, v_cap.y);
  float a = past > 0.0 ? clamp(0.5 - (length(vec2(past, v_cap.z)) - v_cap.w), 0.0, 1.0) * v_alpha : v_alpha;
  o = vec4(v_color * a, a);
}`;

const STOP_VS = `#version 300 es
precision highp float;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform vec2 u_r;            // a ring's outer radius and an end's half side, device px
uniform vec3 u_rail_rgb;
uniform float u_rings;       // 0: the ends only
in vec4 a_stop;              // x, y, 1 a ring + 2 a blueprint's, the colour packed
out vec2 v_p;
flat out vec3 v_color;
flat out float v_end;
flat out float v_r;
void main() {
  int code = int(a_stop.z + 0.5);
  bool ring = (code & 1) != 0, bp = (code & 2) != 0;
  if (ring && u_rings < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float r = ring ? u_r.x : u_r.y;
  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, gl_VertexID < 2 ? -1.0 : 1.0);
  vec2 off = corner * (r + 2.0);
  vec4 c = project(a_stop.xy);
  c.xy += off / (u_viewport * 0.5) * c.w;
  gl_Position = c;
  v_p = off;
  v_r = r;
  v_end = ring ? 0.0 : 1.0;
  float p = a_stop.w;
  v_color = bp ? u_rail_rgb : vec3(floor(p / 65536.0), mod(floor(p / 256.0), 256.0), mod(p, 256.0)) / 255.0;
}`;

const STOP_FS = `#version 300 es
precision highp float;
uniform vec3 u_fill;
uniform float u_rim;         // device px
in vec2 v_p;
flat in vec3 v_color;
flat in float v_end;
flat in float v_r;
out vec4 o;
void main() {
  float d;
  if (v_end > 0.5) {
    float cr = 0.3 * v_r;
    vec2 q = abs(v_p) - vec2(v_r - cr);
    d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - cr;
  } else d = length(v_p) - v_r;
  float cover = clamp(0.5 - d, 0.0, 1.0);
  if (cover <= 0.0) discard;
  vec3 c = mix(u_fill, v_color, clamp(d + u_rim + 0.5, 0.0, 1.0));
  o = vec4(c * cover, cover);
}`;

const pack = (colour: string) => {
  const [r, g, b] = rgb(colour);
  return r * 65536 + g * 256 + b;
};

const sameNums = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

/** How the renderer styles the pass: its own widths and colours, which this file does not import. */
export interface BusStyle {
  /** a rail line's width and casing, device px (`renderer.draw`'s `wLine` and `casing`), and a selected line's extra */
  wLine: number;
  casing: number;
  wSel: number;
  /** blueprint's rail colour and its share of a line's width (at least a pixel) */
  rail: Float32Array;
  railShare: number;
  sel: number;
  hot: number;
  dim: number;
  shades: ShadeColours;
  fill: Float32Array;
  casingRgb: Float32Array;
}

export class BusLayer {
  private gl: WebGL2RenderingContext | null = null;
  private segProg: WebGLProgram | null = null;
  private stopProg: WebGLProgram | null = null;
  private su: Record<string, WebGLUniformLocation | null> = {};
  private pu: Record<string, WebGLUniformLocation | null> = {};
  private segVao: WebGLVertexArrayObject | null = null;
  private stopVao: WebGLVertexArrayObject | null = null;
  private bufs: WebGLBuffer[] = [];
  private lines: BusSource[] = [];
  private byId = new Map<number, BusSource>();
  private built: Built | null = null;
  /** the build the GPU holds */
  private onGpu: Built | null = null;
  private last: Drawn | null = null;

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.segProg = compile(gl, SEG_VS, SEG_FS);
    this.stopProg = compile(gl, STOP_VS, STOP_FS);
    this.su = uniforms(gl, this.segProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_extra", "u_pass", "u_sel", "u_hot", "u_only", "u_dim", "u_alpha", "u_casing_rgb", "u_rail_rgb", "u_rail"]);
    this.pu = uniforms(gl, this.stopProg, [...PROJECT_UNIFORMS, "u_viewport", "u_r", "u_rail_rgb", "u_rings", "u_fill", "u_rim"]);
    this.segVao = gl.createVertexArray();
    this.stopVao = gl.createVertexArray();
    this.bufs = [];
    this.onGpu = null;
  }

  /** Take the world's bus lines; true when what is drawn changed. */
  set(w: WorldView | null): boolean {
    const next: BusSource[] = [];
    if (w) {
      for (const l of w.save.lines) {
        const b = w.lineStats[l.id]?.bus;
        if (!b || b.routeLL.length < 4) continue;
        const stops: number[] = [];
        for (const s of b.stops) stops.push(s.lng, s.lat, s.kind);
        next.push({ id: l.num, colour: l.colour, built: b.built, route: b.routeLL, street: b.streetLL, stops });
      }
    }
    const was = this.lines;
    if (was.length === next.length && was.every((x, i) => {
      const y = next[i];
      return x.id === y.id && x.colour === y.colour && x.built === y.built && sameNums(x.route, y.route) && sameNums(x.street, y.street) && sameNums(x.stops, y.stops);
    })) return false;
    this.lines = next;
    this.byId = new Map(next.map((l) => [l.id, l]));
    this.built = null;
    return true;
  }

  /** There is a bus line to draw. */
  get any(): boolean {
    return this.lines.length > 0;
  }

  has(line: number): boolean {
    return this.byId.has(line);
  }

  private local(a: Anchor): Built {
    if (this.built && this.built.a === a) return this.built;
    let nSeg = 0, nStop = 0;
    for (const l of this.lines) {
      nSeg += l.route.length / 2 - 1;
      nStop += l.street.length / 2 + l.stops.length / 3;
    }
    const seg = new Float32Array(nSeg * 4), info = new Float32Array(nSeg * 3);
    const rings: number[] = [], ends: number[] = [], ringLine: number[] = [], endLine: number[] = [];
    const k = new Map<number, number>();
    let at = 0;
    for (const l of this.lines) {
      const col = pack(l.colour), flags = l.built ? 0 : FLAG_BLUEPRINT, code = l.built ? 0 : 2;
      let px = NaN, py = NaN, sumY = 0;
      for (let i = 0; i + 1 < l.route.length; i += 2) {
        const [x, y] = toLocal(l.route[i], l.route[i + 1], a);
        sumY += y;
        if (i > 0) {
          seg.set([px, py, x, y], at * 4);
          info.set([l.id, flags, col], at * 3);
          at++;
        }
        px = x;
        py = y;
      }
      k.set(l.id, kAtY(sumY / (l.route.length / 2), a));
      for (let i = 0; i + 1 < l.street.length; i += 2) {
        const [x, y] = toLocal(l.street[i], l.street[i + 1], a);
        rings.push(x, y, 1 + code, col);
        ringLine.push(l.id);
      }
      for (let i = 0; i + 2 < l.stops.length; i += 3) {
        const kind = l.stops[i + 2];
        // a station's own circle stands for a stop at a station
        if (kind === 1) continue;
        const [x, y] = toLocal(l.stops[i], l.stops[i + 1], a);
        if (kind === 0) (ends.push(x, y, code, col), endLine.push(l.id));
        else (rings.push(x, y, 1 + code, col), ringLine.push(l.id));
      }
    }
    const stopCount = (rings.length + ends.length) / 4;
    this.built = {
      a, seg, info, count: at, stops: Float32Array.from([...rings, ...ends]), stopCount,
      stopLine: Int32Array.from([...ringLine, ...endLine]),
      stopEnd: Uint8Array.from([...ringLine.map(() => 0), ...endLine.map(() => 1)]),
      k,
    };
    return this.built;
  }

  private upload(b: Built) {
    const gl = this.gl!;
    for (const x of this.bufs) gl.deleteBuffer(x);
    this.bufs = [];
    const put = (data: Float32Array, name: string, size: number, prog: WebGLProgram) => {
      const buf = gl.createBuffer()!;
      this.bufs.push(buf);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data.length ? data : new Float32Array(size), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    };
    gl.bindVertexArray(this.segVao);
    put(b.seg, "a_seg", 4, this.segProg!);
    put(b.info, "a_info", 3, this.segProg!);
    gl.bindVertexArray(this.stopVao);
    put(b.stops, "a_stop", 4, this.stopProg!);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    this.onGpu = b;
  }

  /** Draw the bus lines in the renderer's frame: the shaded corridors first (under everything the
   * renderer draws), then casings, colours and stops. Leaves the blend as it found it. */
  draw(lc: LocalCamera, a: Anchor, vp: readonly [number, number], zoom: number, dpr: number, s: BusStyle) {
    this.last = null;
    if (!this.lines.length || !this.gl || !this.segProg) return;
    const gl = this.gl;
    const b = this.local(a);
    if (!b.count) return;
    if (this.onGpu !== b) this.upload(b);
    const bus = s.wLine * WIDTH_SHARE;
    const u = this.su;
    gl.useProgram(this.segProg);
    setCamera(gl, u, lc);
    gl.uniform2f(u.u_viewport, vp[0], vp[1]);
    gl.uniform1i(u.u_sel, s.sel);
    gl.uniform1i(u.u_hot, s.hot);
    gl.uniform1f(u.u_dim, s.dim);
    gl.uniform1f(u.u_extra, s.wSel * WIDTH_SHARE);
    gl.uniform3fv(u.u_casing_rgb, s.casingRgb);
    gl.uniform3fv(u.u_rail_rgb, s.rail);
    gl.uniform1f(u.u_rail, Math.max(1 * dpr, s.railShare * bus));
    gl.bindVertexArray(this.segVao);
    // the corridor of the hovered and the selected line, a pale band; MAX blending makes where its
    // segments and caps overlap one band, edges included
    const lit = [...new Set([s.hot, s.sel].filter((id) => this.byId.has(id)))];
    if (lit.length) {
      gl.blendEquation(gl.MAX);
      gl.uniform1i(u.u_pass, 2);
      gl.uniform1f(u.u_alpha, CORRIDOR_ALPHA);
      const perM = pxPerUnitAt(zoom) * dpr;
      for (const id of lit) {
        gl.uniform1i(u.u_only, id);
        gl.uniform1f(u.u_width, 2 * CORRIDOR_M * (b.k.get(id) ?? 1) * perM);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
      }
      gl.blendEquation(gl.FUNC_ADD);
    }
    const body = (only: number) => {
      gl.uniform1i(u.u_only, only);
      gl.uniform1i(u.u_pass, 0);
      gl.uniform1f(u.u_width, bus + s.casing * CASING_SHARE);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
      gl.uniform1i(u.u_pass, 1);
      gl.uniform1f(u.u_width, bus);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
    };
    body(-1);
    // the selected line again over the rest, as the rail lines' (renderer.ts)
    if (this.byId.has(s.sel)) body(-2);
    gl.bindVertexArray(null);

    const rings = zoom >= BUS_RING_ZOOM, ends = zoom >= BUS_END_ZOOM;
    if (ends && b.stopCount) {
      const ringR = Math.max(MIN_RING_PX * dpr, bus * RING_SHARE), endR = Math.max(MIN_RING_PX * 1.25 * dpr, bus * END_SHARE);
      const p = this.pu;
      gl.useProgram(this.stopProg);
      setCamera(gl, p, lc);
      gl.uniform2f(p.u_viewport, vp[0], vp[1]);
      gl.uniform2f(p.u_r, ringR, endR);
      gl.uniform3fv(p.u_rail_rgb, s.rail);
      gl.uniform3fv(p.u_fill, s.fill);
      gl.uniform1f(p.u_rim, Math.max(1, 0.45 * bus));
      gl.bindVertexArray(this.stopVao);
      gl.uniform1f(p.u_rings, rings ? 1 : 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.stopCount);
      gl.bindVertexArray(null);
      this.last = { rings, ends, ringR: ringR / dpr, endR: endR / dpr };
    }
  }

  /** The bus line under the pointer (CSS px), or -1. */
  lineAt(ov: Overlay, px: number, py: number): number {
    if (!this.lines.length) return -1;
    const b = this.local(anchor());
    if (!b.count) return -1;
    const [i] = topSeg(ov, b.seg, b.count, px, py, PICK_SLOP_PX * PICK_SLOP_PX + 1);
    return i < 0 ? -1 : b.info[i * 3];
  }

  /** Up to `max` bus lines under the pointer, nearest first, for the resting hover's list. */
  linesAt(ov: Overlay, px: number, py: number, max: number, skip: (id: number) => boolean): number[] {
    if (!this.lines.length) return [];
    const b = this.local(anchor());
    const out: number[] = [];
    while (out.length < max && b.count) {
      const [i] = topSeg(ov, b.seg, b.count, px, py, PICK_SLOP_PX * PICK_SLOP_PX + 1, undefined, 0, (j) => !skip(b.info[j * 3]) && !out.includes(b.info[j * 3]));
      if (i < 0) break;
      out.push(b.info[i * 3]);
    }
    return out;
  }

  /** The stop drawn under the pointer: its line, and whether it is an end. */
  stopAt(ov: Overlay, px: number, py: number): { line: number; end: boolean } | null {
    const L = this.last, b = this.built;
    if (!L || !b || !b.stopCount) return null;
    const p = SCRATCH;
    // the ends were drawn last, so they are tried first
    for (let i = b.stopCount - 1; i >= 0; i--) {
      const end = b.stopEnd[i] === 1;
      if (end ? !L.ends : !L.rings) continue;
      ov.toScreenInto(b.stops[i * 4], b.stops[i * 4 + 1], p, 0);
      const r = (end ? L.endR : L.ringR) + 3;
      if ((p[0] - px) ** 2 + (p[1] - py) ** 2 <= r * r) return { line: b.stopLine[i], end };
    }
    return null;
  }
}
const SCRATCH = new Float64Array(2);

/** A tag row of a line's badge and its name, as the resting hover's list has them. */
function nameRow(w: WorldView, line: number): HTMLElement | string {
  const l = w.save.lines.find((x) => x.num === line);
  if (!l) return `Line ${line}`;
  const name = Object.assign(document.createElement("span"), { textContent: l.name });
  return factRow(lineBullet(l, true), name);
}

/** The tag for a bus line (hovering its route). */
export function busLineTip(w: WorldView, line: number): TipLine[] {
  const l = w.save.lines.find((x) => x.num === line);
  const built = l ? w.lineStats[l.id]?.bus?.built : true;
  return built ? [nameRow(w, line)] : [nameRow(w, line), "Not running yet"];
}

/** The tag for one of its stops. */
export function busStopTip(w: WorldView, line: number, end: boolean): TipLine[] {
  return [end ? "End of the line" : "Bus stop", nameRow(w, line)];
}
