// How bus lines look on the map (T-1464 stage B1, notes/T-1445.md "What the player sees and pays").
// A bus line has no track, so the network buffers hold no strokes for it; this layer draws its drawn
// route from `lineStats[id].bus` instead, in the renderer's own pass (renderer.ts calls `draw` before
// the track): half a rail line's width in the line's colour on the casing, a planned route as rail
// blueprint is, the 300 m corridor of the selected line as a thin outline under everything, small
// rings for its street stops from zoom 14 and rounded squares for its ends. A stop at a train station
// draws nothing: the station's own circle is there.
//
// A ferry line (T-2194) has a bus line's row, so it is drawn here too: its route dashed on a continuous
// casing, a solid stroke from each port's landing to its berth, no stops (its ports are stations) and
// no corridor.
//
// In the Speed and Profit track displays (T-2452) a running line's route takes the display's colour in
// place of its own, its width, casing and dashes kept: Speed by its vehicles' run profiles as the rail
// track's (`map/network.ts` `speedAt`), Profit by each stretch's fares over costs (`stretchMoney`'s
// `buses`); the plain track grey where there is no figure, and on a ferry's landings.
//
// Cost: the geometry is built once per world change and anchor move and kept on the GPU; a frame with
// no bus line returns at the first line of `draw`. Picking and the tags read the same arrays
// (`lineAt`, `stopAt`), on pointer moves only.

import { t } from "../i18n";
import { busSpeedColour, ferrySpeedColour, profitColour, rgb } from "../game/palette";
import type { ShadeColours } from "../game/shadeTable";
import type { WorldView } from "../game/types";
import { anchor, kAtY, pxPerUnitAt, toLocal, type Anchor } from "./geo";
import { compile, PROJECT_GLSL, PROJECT_UNIFORMS, setCamera, uniforms, type LocalCamera } from "./gl";
import type { Overlay } from "./overlay";
import { SPEED_LEVEL, speedAt } from "./network";
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
export const CORRIDOR_M = 300;
/** The selected line's corridor outline: opacity of the line's colour, and thickness in CSS px (T-1490). */
const OUTLINE_ALPHA = 0.75;
const OUTLINE_PX = 1;
const HOT_MIX = 0.3;
/** Bus stop sizes as shares of the bus line's width, and the least radius, device px */
const RING_SHARE = 1.15, END_SHARE = 1.5, MIN_RING_PX = 2.6;
const PICK_SLOP_PX = 6;

const FLAG_BLUEPRINT = 1;
/** T-2194: a ferry line's route, dashed, and the short solid line from a port's landing to its berth */
const FLAG_FERRY = 2, FLAG_SPUR = 4;
/** a ferry's dashes: one dash and gap this many line widths long, the dash this share of it */
const DASH_WIDTHS = 3.2, DASH_ON = 0.6;

/** A bus line as the world gives it (lng, lat pairs flat). */
interface BusSource {
  id: number;
  colour: string;
  built: boolean;
  route: number[];
  street: number[];
  /** the stops demand sees, as lng, lat, kind */
  stops: number[];
  /** T-2194: a ferry line, and each port's landing and berth (lng, lat, lng, lat) */
  ferry: boolean;
  spurs: number[];
  /** the stops riders are counted between, in run 0's order (lng, lat): a ferry's berths, a bus's demand stops */
  marks: number[];
  /** T-2452: those stops' places along each run's profile, m (`LineStats.stopS`), while it runs
   * vehicles; else null */
  stopS: number[][] | null;
}

/** The sources' geometry in local units of one anchor. */
interface Built {
  a: Anchor;
  /** per route segment: x0, y0, x1, y1 */
  seg: Float32Array;
  /** per route segment: line id, flags, its colour as r * 65536 + g * 256 + b, local units along its
   * line's route at its start (a ferry's dashes run on across its points) */
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
  /** per route segment: local units along its line's route at its start */
  cum: Float64Array;
  /** per line id, its first segment and their number */
  range: Map<number, [number, number]>;
  /** per line id, each of its stops' place along the route, in `cum`'s units (made when asked) */
  stopAlong: Map<number, number[]>;
}

/** What the last draw drew, CSS px, for the hit tests; null when it drew nothing. */
interface Drawn { rings: boolean; ends: boolean; ringR: number; endR: number }

const SEG_VS = `#version 300 es
precision highp float;
precision highp int;
${PROJECT_GLSL}
uniform vec2 u_viewport;
uniform float u_width;       // device px across (the corridor's edge in pass 2)
uniform float u_add;         // the casing beyond the line, device px (pass 0)
uniform float u_extra;       // added to the selected line
uniform int u_pass;          // 0 casing, 1 colour, 2 corridor outline (drawn three times, see draw())
uniform int u_sel;
uniform int u_hot;
uniform int u_only;          // -1 every line, -2 the selected line, else that line only
uniform float u_dim;
uniform float u_alpha;       // the corridor's
uniform vec3 u_casing_rgb;
uniform vec3 u_rail_rgb;
uniform float u_rail;        // a blueprint's rail beyond the colour, device px a side
uniform float u_px_per_unit; // device px a local unit spans
in vec4 a_seg;
in vec4 a_info;
in float a_w;                // T-2274: the Traffic view's width by riders, a share of the line's own
in float a_col;              // T-2452: the Speed or Profit view's colour packed as a_info.z; -1 the line's own, -2 plain track
uniform vec3 u_track_rgb;
out vec3 v_color;
out float v_alpha;
out vec4 v_cap;
out float v_dash;            // T-2194: device px along a ferry's route, -1 elsewhere
flat out float v_period;
void main() {
  int id = int(a_info.x + 0.5);
  int flags = int(a_info.y + 0.5);
  bool bp = (flags & ${FLAG_BLUEPRINT}) != 0;
  bool sel = id == u_sel;
  if ((u_only >= 0 && id != u_only) || (u_only == -2 && !sel) || (u_pass == 2 && (flags & ${FLAG_FERRY | FLAG_SPUR}) != 0)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  bool paint = u_pass == 1 && !bp;
  float p = paint && a_col >= 0.0 ? a_col : a_info.z;
  vec3 col = paint && a_col < -1.5 ? u_track_rgb : vec3(floor(p / 65536.0), mod(floor(p / 256.0), 256.0), mod(p, 256.0)) / 255.0;
  float w = u_pass < 2 ? u_width * a_w + (u_pass == 0 ? u_add : 0.0) + (sel ? u_extra : 0.0) : u_width;
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
  v_period = max(${DASH_WIDTHS.toFixed(1)} * w, 6.0);
  v_dash = u_pass == 1 && (flags & ${FLAG_FERRY}) != 0 ? a_info.w * u_px_per_unit + ua : -1.0;
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
in float v_dash;
flat in float v_period;
out vec4 o;
void main() {
  float past = -min(v_cap.x, v_cap.y);
  float a = past > 0.0 ? clamp(0.5 - (length(vec2(past, v_cap.z)) - v_cap.w), 0.0, 1.0) * v_alpha : v_alpha;
  if (v_dash >= 0.0) {
    // a ferry's dash, its ends softened over a pixel
    float f = mod(v_dash, v_period), on = ${DASH_ON.toFixed(2)} * v_period;
    a *= clamp(min(f, on - f) + 0.5, 0.0, 1.0);
  }
  if (a <= 0.0) discard;   // the stencil passes must not mark a cap's empty corners
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
    float cr = 0.36 * v_r;       // the bus bullet's corner share of the side (BUS_CORNER, ui/bullet.ts) x 2
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
  /** T-2452: the track display colouring the routes, with the run profiles (the network's keyframes)
   * the Speed view reads, and the plain track's colour */
  paint: BusPaintKind;
  phases: Float32Array;
  meta: Float32Array;
  trackRgb: Float32Array;
}

export type BusPaintKind = "speed" | "profit" | null;

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
  /** counts `set`s that changed the lines (the renderer's stroke cache, T-1638) */
  version = 0;
  private byId = new Map<number, BusSource>();
  private built: Built | null = null;
  /** the build the GPU holds */
  private onGpu: Built | null = null;
  private last: Drawn | null = null;

  init(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.segProg = compile(gl, SEG_VS, SEG_FS);
    this.stopProg = compile(gl, STOP_VS, STOP_FS);
    this.su = uniforms(gl, this.segProg, [...PROJECT_UNIFORMS, "u_viewport", "u_width", "u_add", "u_extra", "u_pass", "u_sel", "u_hot", "u_only", "u_dim", "u_alpha", "u_casing_rgb", "u_rail_rgb", "u_rail", "u_px_per_unit", "u_track_rgb"]);
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
        const stops: number[] = [], spurs: number[] = [];
        // a ferry's stops are its ports, drawn as the stations they are
        if (!b.ferry) for (const s of b.stops) stops.push(s.lng, s.lat, s.kind);
        for (const p of b.ferry?.ports ?? []) spurs.push(...p.landing, ...p.berth);
        const marks = b.ferry ? b.ferry.ports.flatMap((p) => p.berth) : b.stops.flatMap((s) => [s.lng, s.lat]);
        const st = w.lineStats[l.id], n = marks.length / 2;
        const runs = st.status === "running" && Math.max(l.tph.high, l.tph.medium, l.tph.low) > 0 && st.stopS.length === 2 && st.stopS.every((s) => s.length === n);
        next.push({ id: l.num, colour: l.colour, built: b.built, route: b.routeLL, street: b.streetLL, stops, ferry: !!b.ferry, spurs, marks, stopS: runs ? st.stopS : null });
      }
    }
    const was = this.lines;
    const sameS = (a: number[][] | null, b: number[][] | null) => a === b || (!!a && !!b && a.length === b.length && a.every((s, k) => sameNums(s, b[k])));
    if (was.length === next.length && was.every((x, i) => {
      const y = next[i];
      return x.id === y.id && x.colour === y.colour && x.built === y.built && sameNums(x.route, y.route) && sameNums(x.street, y.street) && sameNums(x.stops, y.stops) && sameNums(x.spurs, y.spurs) && sameNums(x.marks, y.marks) && sameS(x.stopS, y.stopS);
    })) return false;
    this.lines = next;
    this.byId = new Map(next.map((l) => [l.id, l]));
    this.built = null;
    this.version++;
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
      nSeg += l.route.length / 2 - 1 + l.spurs.length / 4;
      nStop += l.street.length / 2 + l.stops.length / 3;
    }
    const seg = new Float32Array(nSeg * 4), info = new Float32Array(nSeg * 4);
    const rings: number[] = [], ends: number[] = [], ringLine: number[] = [], endLine: number[] = [];
    const k = new Map<number, number>();
    const cum = new Float64Array(nSeg), range = new Map<number, [number, number]>();
    let at = 0;
    for (const l of this.lines) {
      const col = pack(l.colour), flags = (l.built ? 0 : FLAG_BLUEPRINT) | (l.ferry ? FLAG_FERRY : 0), code = l.built ? 0 : 2;
      let px = NaN, py = NaN, sumY = 0, run = 0;
      const first = at;
      for (let i = 0; i + 1 < l.route.length; i += 2) {
        const [x, y] = toLocal(l.route[i], l.route[i + 1], a);
        sumY += y;
        if (i > 0) {
          seg.set([px, py, x, y], at * 4);
          info.set([l.id, flags, col, run], at * 4);
          cum[at] = run;
          run += Math.hypot(x - px, y - py);
          at++;
        }
        px = x;
        py = y;
      }
      range.set(l.id, [first, at - first]);
      for (let i = 0; i + 3 < l.spurs.length; i += 4) {
        const [x0, y0] = toLocal(l.spurs[i], l.spurs[i + 1], a), [x1, y1] = toLocal(l.spurs[i + 2], l.spurs[i + 3], a);
        seg.set([x0, y0, x1, y1], at * 4);
        info.set([l.id, (flags & ~FLAG_FERRY) | FLAG_SPUR, col, 0], at * 4);
        at++;
      }
      k.set(l.id, kAtY(sumY / (l.route.length / 2), a));
      for (let i = 0; i + 1 < l.street.length; i += 2) {
        const [x, y] = toLocal(l.street[i], l.street[i + 1], a);
        rings.push(x, y, 1 + code, col);
        ringLine.push(l.id);
      }
      for (let i = 0; i + 2 < l.stops.length; i += 3) {
        const kind = l.stops[i + 2];
        // T-2938: a line's two ends are drawn as ends even where they are at a station; the station's own
        // circle stands for any other stop at a station
        const end = kind === 0 || (kind === 1 && (i === 0 || i + 3 >= l.stops.length));
        if (kind === 1 && !end) continue;
        const [x, y] = toLocal(l.stops[i], l.stops[i + 1], a);
        if (end) (ends.push(x, y, code, col), endLine.push(l.id));
        else (rings.push(x, y, 1 + code, col), ringLine.push(l.id));
      }
    }
    const stopCount = (rings.length + ends.length) / 4;
    this.built = {
      a, seg, info, count: at, stops: Float32Array.from([...rings, ...ends]), stopCount,
      stopLine: Int32Array.from([...ringLine, ...endLine]),
      stopEnd: Uint8Array.from([...ringLine.map(() => 0), ...endLine.map(() => 1)]),
      k, cum, range, stopAlong: new Map(),
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
    put(b.info, "a_info", 4, this.segProg!);
    put(this.segWidths(b), "a_w", 1, this.segProg!);
    this.wBuf = this.bufs[this.bufs.length - 1];
    this.widthsFor = b;
    put(new Float32Array(b.count).fill(-1), "a_col", 1, this.segProg!);
    this.cBuf = this.bufs[this.bufs.length - 1];
    this.coloursFor = [b];
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
    if (this.widthsFor !== b && this.wBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.wBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.segWidths(b), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      this.widthsFor = b;
    }
    const cKey = [b, s.paint, s.paint === "speed" ? s.phases : null, s.paint === "speed" ? s.meta : null, s.paint === "profit" ? this.profit : null];
    if (this.cBuf && (cKey.length !== this.coloursFor.length || cKey.some((v, i) => v !== this.coloursFor[i]))) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.cBuf);
      gl.bufferData(gl.ARRAY_BUFFER, this.segColours(b, s), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      this.coloursFor = cKey;
    }
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
    gl.uniform3fv(u.u_track_rgb, s.trackRgb);
    gl.uniform1f(u.u_rail, Math.max(1 * dpr, s.railShare * bus));
    gl.uniform1f(u.u_px_per_unit, pxPerUnitAt(zoom) * dpr);
    gl.bindVertexArray(this.segVao);
    // The selected line's corridor as an outline of its union (T-1490): the stencil gets 1 inside the
    // corridor widened by half the outline and 0 inside it narrowed by half, so what stays 1 is a ring
    // along the edge, however the segments and caps overlap; the colour pass then draws the ring once
    // (the stencil goes to 0 where it draws).
    if (this.byId.has(s.sel) && !this.byId.get(s.sel)!.ferry && (gl.getParameter(gl.STENCIL_BITS) ?? 0) > 0) {
      const edge = 2 * CORRIDOR_M * (b.k.get(s.sel) ?? 1) * pxPerUnitAt(zoom) * dpr, half = 0.5 * OUTLINE_PX * dpr;
      gl.uniform1i(u.u_pass, 2);
      gl.uniform1i(u.u_only, s.sel);
      gl.uniform1f(u.u_alpha, 1);
      gl.enable(gl.STENCIL_TEST);
      gl.clearStencil(0);
      gl.clear(gl.STENCIL_BUFFER_BIT);
      gl.colorMask(false, false, false, false);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      for (const [ref, width] of [[1, edge + half * 2], [0, Math.max(0, edge - half * 2)]] as const) {
        if (width <= 0) continue;
        gl.stencilFunc(gl.ALWAYS, ref, 0xff);
        gl.uniform1f(u.u_width, width);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
      }
      gl.colorMask(true, true, true, true);
      gl.stencilFunc(gl.EQUAL, 1, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.ZERO);
      gl.uniform1f(u.u_alpha, OUTLINE_ALPHA);
      gl.uniform1f(u.u_width, edge + half * 2);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
      gl.disable(gl.STENCIL_TEST);
    }
    const body = (only: number) => {
      gl.uniform1i(u.u_only, only);
      gl.uniform1f(u.u_width, bus);
      gl.uniform1i(u.u_pass, 0);
      gl.uniform1f(u.u_add, s.casing * CASING_SHARE);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
      gl.uniform1i(u.u_pass, 1);
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
    return i < 0 ? -1 : b.info[i * 4];
  }

  /** Up to `max` bus lines under the pointer, nearest first, for the resting hover's list. */
  linesAt(ov: Overlay, px: number, py: number, max: number, skip: (id: number) => boolean): number[] {
    if (!this.lines.length) return [];
    const b = this.local(anchor());
    const out: number[] = [];
    while (out.length < max && b.count) {
      const [i] = topSeg(ov, b.seg, b.count, px, py, PICK_SLOP_PX * PICK_SLOP_PX + 1, undefined, 0, (j) => !skip(b.info[j * 4]) && !out.includes(b.info[j * 4]));
      if (i < 0) break;
      out.push(b.info[i * 4]);
    }
    return out;
  }

  /** The stretch between the line's demand stops that the pointer (CSS px) is over, counted from its
   * first stop as the line's riders are; -1 where it cannot tell. */
  stretchAt(ov: Overlay, px: number, py: number, line: number): number {
    const b = this.local(anchor());
    const r = b.range.get(line), at = ov.screenToLocal(px, py);
    if (!r || !r[1] || !at) return -1;
    const stops = this.stopsAlong(b, line);
    if (!stops) return -1;
    return stretchOf(stops, alongRoute(b, r, at[0], at[1]));
  }

  /** Line `line`'s counted stops' places along its route, in `cum`'s units (kept with the build). */
  private stopsAlong(b: Built, line: number): number[] | null {
    const l = this.byId.get(line), r = b.range.get(line);
    if (!l || !r || !r[1]) return null;
    let stops = b.stopAlong.get(line);
    if (!stops) {
      stops = [];
      // each stop searched for from the one before, so a route passing a stop twice counts it where it stops
      let from = r[0];
      for (let i = 0; i + 1 < l.marks.length; i += 2) {
        const [x, y] = toLocal(l.marks[i], l.marks[i + 1], b.a);
        const [s, seg] = nearestAlong(b, from, r[0] + r[1], x, y);
        stops.push(s);
        from = seg;
      }
      b.stopAlong.set(line, stops);
    }
    return stops;
  }

  /** T-2274: the Traffic view's widths, per line its stretches' widths as shares of a rail line's
   * (`demandLayers.ts` `busWidths`); null draws every line at its own width. */
  setTraffic(widths: Map<number, Float32Array> | null) {
    if (widths === this.traffic) return;
    this.traffic = widths;
    this.widthsFor = null;
    this.version++;
  }
  private traffic: Map<number, Float32Array> | null = null;
  /** the build `wBuf` holds widths for */
  private widthsFor: Built | null = null;
  private wBuf: WebGLBuffer | null = null;

  /** Per route segment the Traffic view's width share (1 outside it). */
  private segWidths(b: Built): Float32Array {
    const out = new Float32Array(b.count).fill(1);
    const t = this.traffic;
    if (!t) return out;
    for (const [line, ws] of t) {
      const r = b.range.get(line);
      const stops = r && this.stopsAlong(b, line);
      if (!r || !stops || ws.length !== stops.length - 1) continue;
      for (let i = r[0]; i < r[0] + r[1]; i++) {
        const mid = b.cum[i] + 0.5 * Math.hypot(b.seg[i * 4 + 2] - b.seg[i * 4], b.seg[i * 4 + 3] - b.seg[i * 4 + 1]);
        out[i] = (ws[stretchOf(stops, mid)] ?? WIDTH_SHARE) / WIDTH_SHARE;
      }
    }
    return out;
  }

  /** T-2452: the Profit view's figure per running line's number, each stretch's fares over costs
   * (`stretchMoney`'s `buses`; `NO_PROFIT` where none); null outside the view. */
  setProfit(ratios: Map<number, { ratio: Float32Array }> | null) {
    if (ratios === this.profit) return;
    this.profit = ratios;
    this.version++;
  }
  private profit: Map<number, { ratio: Float32Array }> | null = null;
  private cBuf: WebGLBuffer | null = null;
  /** what the colours `cBuf` holds were made from */
  private coloursFor: unknown[] = [];
  /** the run profiles the last draw had, for the hover's speeds */
  private keyframes: { phases: Float32Array; meta: Float32Array } | null = null;

  /** Per route segment the display's colour packed (-1 the line's own, -2 plain track). */
  private segColours(b: Built, s: BusStyle): Float32Array {
    const out = new Float32Array(b.count).fill(-1);
    if (!s.paint) return out;
    if (s.paint === "speed") this.keyframes = { phases: s.phases, meta: s.meta };
    for (const l of this.lines) {
      const r = b.range.get(l.id), stops = r && this.stopsAlong(b, l.id);
      if (!l.built || !r || !stops) continue;
      // a ferry's landings have no figure of their own
      for (let i = r[0] + r[1]; i < b.count && b.info[i * 4] === l.id; i++) out[i] = -2;
      const ratio = this.profit?.get(l.id)?.ratio;
      for (let i = r[0]; i < r[0] + r[1]; i++) {
        const mid = b.cum[i] + 0.5 * Math.hypot(b.seg[i * 4 + 2] - b.seg[i * 4], b.seg[i * 4 + 3] - b.seg[i * 4 + 1]);
        const k = stretchOf(stops, mid);
        let v: number | null = null;
        if (s.paint === "speed") {
          const f = stops[k + 1] > stops[k] ? (mid - stops[k]) / (stops[k + 1] - stops[k]) : 0.5;
          v = runSpeed(l, k, f, s.phases, s.meta);
          out[i] = v === null ? -2 : pack(l.ferry ? ferrySpeedColour(v) : busSpeedColour(v));
        } else {
          v = ratio && k < ratio.length && ratio[k] < 1e29 ? ratio[k] : null;
          out[i] = v === null ? -2 : pack(profitColour(v));
        }
      }
    }
    return out;
  }

  /** T-2452: the speed the Speed view draws on line `line` under the pointer, km/h; null where its
   * vehicles have no run profile (none run, or the game loading). */
  speedAt(ov: Overlay, px: number, py: number, line: number): number | null {
    const b = this.local(anchor());
    const l = this.byId.get(line), r = b.range.get(line), at = ov.screenToLocal(px, py);
    const stops = l && this.stopsAlong(b, line);
    if (!l || !r || !stops || !at || !this.keyframes) return null;
    const s = alongRoute(b, r, at[0], at[1]), k = stretchOf(stops, s);
    const f = stops[k + 1] > stops[k] ? Math.max(0, Math.min(1, (s - stops[k]) / (stops[k + 1] - stops[k]))) : 0.5;
    return runSpeed(l, k, f, this.keyframes.phases, this.keyframes.meta);
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

/** The place along segments `from` to `to` (exclusive) nearest a local point, in `cum`'s units, and
 * that segment. */
function nearestAlong(b: Built, from: number, to: number, x: number, y: number): [number, number] {
  let best = Infinity, s = 0, at = from;
  for (let i = from; i < to; i++) {
    const x0 = b.seg[i * 4], y0 = b.seg[i * 4 + 1], dx = b.seg[i * 4 + 2] - x0, dy = b.seg[i * 4 + 3] - y0, len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / len2)) : 0;
    const d = Math.hypot(x0 + t * dx - x, y0 + t * dy - y);
    if (d < best) { best = d; s = b.cum[i] + t * Math.sqrt(len2); at = i; }
  }
  return [s, at];
}

const alongRoute = (b: Built, r: [number, number], x: number, y: number) => nearestAlong(b, r[0], r[0] + r[1], x, y)[0];

/** T-2452: the faster of a line's two runs at `f` of the way along run 0's stretch `k`, km/h, from
 * the run profiles as the rail track's Speed view reads them; null without a profile. */
function runSpeed(l: BusSource, k: number, f: number, phases: Float32Array, meta: Float32Array): number | null {
  const S = l.stopS, n = S?.[0].length ?? 0;
  if (!S || k < 0 || k > n - 2) return null;
  let best: number | null = null;
  for (const run of [0, 1]) {
    const m = ((l.id * 2 + run) * 3 + SPEED_LEVEL) * 4;
    const first = meta[m], count = meta[m + 1];
    if (!(count > 0)) continue;
    // run 1 runs the stops backwards: its stretch n - 2 - k is run 0's k
    const j = run === 0 ? k : n - 2 - k, g = run === 0 ? f : 1 - f;
    const x = S[run][j] + g * (S[run][j + 1] - S[run][j]);
    best = Math.max(best ?? 0, speedAt(phases, first, count, x));
  }
  return best;
}

/** The stretch a place along the route is on, between stops at `stops` (ascending). */
function stretchOf(stops: number[], s: number): number {
  let k = -1;
  for (const v of stops) if (v <= s) k++;
  return Math.max(0, Math.min(stops.length - 2, k));
}

/** A tag row of a line's badge and its name, as the resting hover's list has them. */
function nameRow(w: WorldView, line: number): HTMLElement | string {
  const l = w.save.lines.find((x) => x.num === line);
  if (!l) return t("busLayer.line", { num: line });
  const name = Object.assign(document.createElement("span"), { textContent: l.name });
  return factRow(lineBullet(l, true), name);
}

/** The tag for a bus line (hovering its route): as a rail line's, its name, that it is a bus, and
 * `rows` (its riders there) once it runs. */
export function busLineTip(w: WorldView, line: number, rows: TipLine[] = []): TipLine[] {
  const l = w.save.lines.find((x) => x.num === line);
  const b = l ? w.lineStats[l.id]?.bus : undefined, built = b ? b.built : true;
  const kind = b?.ferry ? t("busLayer.ferryLine") : t("busLayer.busLine");
  return built ? [nameRow(w, line), kind, ...rows] : [nameRow(w, line), kind, t("busLayer.notRunning")];
}

/** The tag for one of its stops. */
export function busStopTip(w: WorldView, line: number, end: boolean): TipLine[] {
  return [end ? t("busLayer.endOfLine") : t("busLayer.busStop"), nameRow(w, line)];
}
