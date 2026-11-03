// The network diagram drawn (T-2160, T-2161; docs/ui-diagram.md): each line one continuous ribbon
// through its pieces, at its slot's offset, corners rounded so a bundle's bends are concentric; each
// station a white pill with an ink outline along an axis its lines cross, its stops where they cross it.
// Paths are in layout units (y north), drawn through one canvas transform.

import type { DiagramGraph } from "./graph";
import type { DiagramLayout, Piece } from "./layout";

/** a stroke is this share of a pitch wide; the rest is the gap between neighbours */
export const STROKE_SHARE = 5 / 6.4;
/** a corner's radius is this many stroke widths, plus the bundle's half width and the offset */
const RADIUS_W = 6;
/** a pill's half width and its outline, in stroke widths */
export const PILL_R = 0.95, PILL_OUTLINE = 0.32;
/** stop points of one station this many pitches apart or nearer share a pill */
const PILL_JOIN = 1.6;
/** stops this many pitches apart or nearer share a pill when no line crosses between them */
const PILL_MERGE = 4;
/** a line running along a pill this many pitches off its axis still lies under its edge */
const ALONG_SLACK = 1.25;
/** a station's axis may pass this many pitches to the side of its place, tried in steps of 1 / AXIS_STEPS
 * pitches, at this cost (pill length) a pitch, where no axis through its place crosses all its lines */
const AXIS_SHIFT = 2, AXIS_STEPS = 2, SHIFT_COST = 1;
/** a passage between pills of one station is this share of a pill wide */
const PASSAGE = 0.45;
/** pills are drawn from a pitch this many device px */
const PILL_MIN_PX = 2.5;

export interface Drawing {
  /** a pitch, stroke width, layout units */
  pitch: number;
  stroke: number;
  /** per line its path as commands (`toPath`), built in the layout worker and posted, the Path2D made
   * on the page when first painted */
  lines: { colour: string; cmds: Float64Array; path?: Path2D }[];
  /** per line, its index in the graph's lines */
  lineOf: number[];
  /** per line, its drawn ribbon sampled: x, y pairs (the names keep off them) */
  ribbons: Float64Array[];
  /** per ribbon point, the index in its line's chain of the last node passed (a click's stretch) */
  chainAt: Int32Array[];
  /** per pill: its two ends and how far its points lie off its axis */
  pills: [number, number, number, number, number][];
  /** per pill, the graph node (station) it is for */
  pillOf: number[];
  /** joins between pills of one station: two centres */
  passages: [number, number, number, number][];
  /** per passage, its station */
  passageOf: number[];
  box: [number, number, number, number];
}

/** a polyline offset left by `o`, one point per point (mitred, capped) */
function offsetKeep(v: Float64Array, o: number): Float64Array {
  const n = v.length / 2, out = new Float64Array(v.length);
  const dir = (i: number, step: number): [number, number] | null => {
    for (; i >= 0 && i < n - 1; i += step) {
      const dx = v[i * 2 + 2] - v[i * 2], dy = v[i * 2 + 3] - v[i * 2 + 1], l = Math.hypot(dx, dy);
      if (l > 1e-9) return [dx / l, dy / l];
    }
    return null;
  };
  for (let i = 0; i < n; i++) {
    let a = i > 0 ? dir(i - 1, -1) : null, b = i < n - 1 ? dir(i, 1) : null;
    a ??= b;
    b ??= a;
    if (!a || !b) {
      out[i * 2] = v[i * 2];
      out[i * 2 + 1] = v[i * 2 + 1];
      continue;
    }
    let mx = -a[1] - b[1], my = a[0] + b[0];
    const l = Math.hypot(mx, my);
    if (l < 1e-9) (mx = -a[1]), (my = a[0]);
    else (mx /= l), (my /= l);
    const c = Math.max(0.4, mx * -a[1] + my * a[0]);
    out[i * 2] = v[i * 2] + (mx * o) / c;
    out[i * 2 + 1] = v[i * 2 + 1] + (my * o) / c;
  }
  return out;
}

type V = [number, number, number, number, number]; // x, y, offset left of travel, bundle half width, node (-1 none)

export function drawing(g: DiagramGraph, lay: DiagramLayout): Drawing {
  const pitch = lay.pu, W = pitch * STROKE_SHARE, R0 = RADIUS_W * W;
  const pieceOf = new Map<number, [Piece, number, number]>();
  const n = g.key.length;
  for (const r of lay.pieces) {
    const p = r.nodes;
    for (let i = 0; i + 1 < p.length; i++) {
      pieceOf.set(p[i] * n + p[i + 1], [r, i, 1]);
      pieceOf.set(p[i + 1] * n + p[i], [r, i + 1, -1]);
    }
  }
  const offs = new Map<Piece, Map<string, [number, Float64Array, number]>>();
  for (const r of lay.pieces) {
    const m = new Map<string, [number, Float64Array, number]>(), k = r.cols.length;
    r.cols.forEach((c, j) => {
      const o = ((k - 1) / 2 - j) * pitch;
      m.set(c, [o, offsetKeep(r.verts, o), ((k - 1) / 2) * pitch]);
    });
    offs.set(r, m);
  }
  const lines: Drawing["lines"] = [];
  const stopsAt = new Map<number, [number, number][]>();
  const Vs: [DiagramGraph["lines"][number], V[]][] = [];
  for (const ln of g.lines) {
    const ch = ln.chain, c = ln.colour;
    const V: V[] = [];
    let i = 0;
    while (i < ch.length - 1) {
      const first = pieceOf.get(ch[i] * n + ch[i + 1]);
      if (!first) { i++; continue; }
      const [r, a, d] = first;
      let j = i;
      while (j < ch.length - 1) {
        const x = pieceOf.get(ch[j] * n + ch[j + 1]);
        if (!x || x[0] !== r || x[2] !== d) break;
        j++;
      }
      const b = a + d * (j - i);
      const [o, op, hw] = offs.get(r)!.get(c) ?? [0, r.verts, 0];
      const va = r.at[a], vb = r.at[b];
      const nodeAt = new Map<number, number>();
      for (let t = 0; t <= j - i; t++) nodeAt.set(r.at[a + d * t], i + t);
      const run: V[] = [];
      for (let t = va; d === 1 ? t <= vb : t >= vb; t += d) {
        const ci = nodeAt.get(t);
        run.push([op[t * 2], op[t * 2 + 1], o * d, hw, ci === undefined ? -1 : ci]);
      }
      if (V.length && run.length > 1) {
        // join at the shared node: the corner where the two runs' offset lines meet, or a 45 degree jog
        const p1 = V[V.length - 1], p0 = V.length > 1 ? V[V.length - 2] : null, q1 = run[0], q2 = run[1];
        let d1x = p0 ? p1[0] - p0[0] : q2[0] - q1[0], d1y = p0 ? p1[1] - p0[1] : q2[1] - q1[1];
        let d2x = q2[0] - q1[0], d2y = q2[1] - q1[1];
        const l1 = Math.hypot(d1x, d1y), l2 = Math.hypot(d2x, d2y);
        if (l1 > 1e-9 && l2 > 1e-9) {
          d1x /= l1, d1y /= l1, d2x /= l2, d2y /= l2;
          const cr = d1x * d2y - d1y * d2x;
          if (Math.abs(cr) > 0.02) {
            const t = ((q1[0] - p1[0]) * d2y - (q1[1] - p1[1]) * d2x) / cr;
            const xx = p1[0] + d1x * t, xy = p1[1] + d1y * t;
            if (Math.hypot(xx - p1[0], xy - p1[1]) < 6 * pitch + 1e-9) {
              V[V.length - 1] = [xx, xy, p1[2], p1[3], p1[4]];
              run.shift();
            }
          } else if (Math.hypot(q1[0] - p1[0], q1[1] - p1[1]) > 1e-6) {
            const jog = Math.hypot(q1[0] - p1[0], q1[1] - p1[1]), ta = Math.min(jog, 0.45 * l1), tb = Math.min(jog, 0.45 * l2);
            V[V.length - 1] = [p1[0] - d1x * ta, p1[1] - d1y * ta, p1[2], p1[3], p1[4]];
            run[0] = [q1[0] + d2x * tb, q1[1] + d2y * tb, q1[2], q1[3], q1[4]];
          } else run.shift();
        }
      }
      V.push(...run);
      i = j;
    }
    Vs.push([ln, V]);
  }
  // Each station's pill lies along an axis through its place, of the four octilinear ones the one
  // every line stopping there crosses, with the shortest pill; a line's stop is where its drawn
  // ribbon crosses the axis, and a line ending there has its last run carried on, or cut back, to it.
  const ribs = Vs.map(([, V]) => {
    const r: Ribbon = { pts: [], vi: [] };
    return { r, cmds: filleted(V, R0, r) };
  });
  type Occ = { li: number; i0: number; i1: number; end: -1 | 0 | 1 };
  const occAt = new Map<number, Occ[]>();
  Vs.forEach(([ln, V], li) => {
    const seenAt = new Set<number>();
    V.forEach((v, i) => {
      if (v[4] < 0 || !ln.stop[v[4]] || seenAt.has(v[4])) return;
      seenAt.add(v[4]);
      let i1 = i;
      while (i1 + 1 < V.length && V[i1 + 1][4] === v[4]) i1++;
      const k = ln.chain[v[4]];
      let o = occAt.get(k);
      if (!o) occAt.set(k, (o = []));
      o.push({ li, i0: i, i1, end: V.length < 2 ? -1 : i === 0 ? 0 : i1 === V.length - 1 ? 1 : -1 });
    });
  });
  /** where a line's end leg, carried on or cut back, meets an axis */
  const endOn = (V: V[], end: 0 | 1, cx: number, cy: number, ax: [number, number]): [number, number] | null => {
    const v = V[end ? V.length - 1 : 0], o = V[end ? V.length - 2 : 1];
    const dx = v[0] - o[0], dy = v[1] - o[1], cr = dx * ax[1] - dy * ax[0];
    if (Math.abs(cr) < 0.2 * Math.hypot(dx, dy)) return null;
    const t = ((cx - o[0]) * ax[1] - (cy - o[1]) * ax[0]) / cr;
    return t > 0.3 && Math.abs(t - 1) * Math.hypot(dx, dy) < 3 * pitch ? [o[0] + dx * t, o[1] + dy * t] : null;
  };
  const AX: [number, number][] = [0, 1, 2, 3].map((q) => [Math.cos((q * Math.PI) / 4), Math.sin((q * Math.PI) / 4)]);
  const axisOf = new Map<number, [number, number]>();
  /** per station, the point its axis passes through */
  const originOf = new Map<number, [number, number]>();
  const dirty = new Set<number>();
  for (const [k, occ] of occAt) {
    let best = Infinity, bq = 0, bo = 0;
    // A line that turns at the station, or ends along an axis, can miss every axis through its place
    // (a round stop off the pill, joined by a passage, reads as a hook, T-2206): the axis may also pass
    // up to AXIS_SHIFT pitches to one side, at a small cost a pitch.
    AX.forEach((ax, q) => {
      for (let s = 0; s <= 2 * AXIS_SHIFT * AXIS_STEPS; s++) {
        const off = ((s % 2 ? 1 : -1) * Math.ceil(s / 2) * pitch) / AXIS_STEPS;
        const cx = lay.x[k] - ax[1] * off, cy = lay.y[k] + ax[0] * off;
        let miss = 0, t0 = Infinity, t1 = -Infinity;
        let beside = 0;
        for (const o of occ) {
          const s = stopOn(ribs[o.li].r, o.i0 - 1, o.i1 + 1, cx, cy, ax, pitch);
          let p = o.end >= 0 ? endOn(Vs[o.li][1], o.end as 0 | 1, cx, cy, ax) : s.cross;
          if (!p) {
            // a line beside the axis, close enough to lie under the pill's edge
            const across = Math.abs((s.near[0] - cx) * ax[1] - (s.near[1] - cy) * ax[0]);
            if (across > ALONG_SLACK * pitch) {
              miss++;
              continue;
            }
            p = s.near;
            beside += across;
          }
          const t = (p[0] - cx) * ax[0] + (p[1] - cy) * ax[1];
          (t0 = Math.min(t0, t)), (t1 = Math.max(t1, t));
        }
        const cost = miss * 1e3 * pitch + (t1 > t0 ? t1 - t0 : 0) + beside + Math.abs(off) * SHIFT_COST + q * 1e-6;
        if (cost < best) (best = cost), (bq = q), (bo = off);
        // no miss on the axis through the place: no need to look to the side
        if (!s && !miss) break;
      }
    });
    const ax = AX[bq], cx = lay.x[k] - ax[1] * bo, cy = lay.y[k] + ax[0] * bo;
    axisOf.set(k, ax);
    originOf.set(k, [cx, cy]);
    for (const o of occ) {
      if (o.end < 0) continue;
      const V = Vs[o.li][1], p = endOn(V, o.end as 0 | 1, cx, cy, ax), at = o.end ? V.length - 1 : 0;
      if (p) (V[at] = [p[0], p[1], V[at][2], V[at][3], V[at][4]]), dirty.add(o.li);
    }
  }
  for (const li of dirty) {
    const r: Ribbon = { pts: [], vi: [] };
    ribs[li] = { r, cmds: filleted(Vs[li][1], R0, r) };
  }
  const ribbons: Float64Array[] = [], lineOf: number[] = [], chainAt: Int32Array[] = [];
  ribs.forEach(({ r, cmds }, li) => {
    if (!cmds) return;
    lines.push({ colour: Vs[li][0].colour, cmds });
    ribbons.push(Float64Array.from(r.pts));
    lineOf.push(li);
    const V = Vs[li][1], at = new Int32Array(r.vi.length);
    let c = 0;
    r.vi.forEach((v, j) => (at[j] = c = V[v][4] >= 0 ? V[v][4] : c));
    chainAt.push(at);
  });
  for (const [k, occ] of occAt) {
    const ax = axisOf.get(k)!, [cx, cy] = originOf.get(k)!, list: [number, number][] = [];
    for (const o of occ) {
      const s = stopOn(ribs[o.li].r, o.i0 - 1, o.i1 + 1, cx, cy, ax, pitch);
      list.push(s.cross ?? s.near);
    }
    stopsAt.set(k, list);
  }
  const pills: Drawing["pills"] = [], passages: Drawing["passages"] = [], pillOf: number[] = [], passageOf: number[] = [];
  for (const [k, raw] of stopsAt) {
    const firstPassage = passages.length;
    const ax = axisOf.get(k)!, [cx, cy] = originOf.get(k)!;
    // along the axis: where each stop is, and how far off it (only a line running along it is)
    const at = raw.map(([px, py]) => [(px - cx) * ax[0] + (py - cy) * ax[1], (px - cx) * ax[1] - (py - cy) * ax[0], px, py]).sort((a, b) => a[0] - b[0]);
    const on = at.filter((p) => Math.abs(p[1]) <= ALONG_SLACK * pitch), off = at.filter((p) => Math.abs(p[1]) > ALONG_SLACK * pitch);
    const centres: [number, number][] = [];
    // stops a little further apart share a pill too where no line crosses the axis between them, so a
    // lone stop past the end of a pill is not a dot of its own (T-2206)
    const clear = (ta: number, tb: number) => !crossesAny(ribs, cx + ax[0] * (ta + 0.3 * pitch), cy + ax[1] * (ta + 0.3 * pitch), cx + ax[0] * (tb - 0.3 * pitch), cy + ax[1] * (tb - 0.3 * pitch));
    for (let i = 0; i < on.length; ) {
      let j = i;
      while (j + 1 < on.length && (on[j + 1][0] - on[j][0] < PILL_JOIN * pitch || (on[j + 1][0] - on[j][0] < PILL_MERGE * pitch && clear(on[j][0], on[j + 1][0])))) j++;
      const t0 = on[i][0], t1 = on[j][0];
      pills.push([cx + ax[0] * t0, cy + ax[1] * t0, cx + ax[0] * t1, cy + ax[1] * t1, 0]);
      pillOf.push(k);
      centres.push([cx + ax[0] * (t0 + t1) / 2, cy + ax[1] * (t0 + t1) / 2]);
      i = j + 1;
    }
    for (let q = 1; q < centres.length; q++) passages.push([centres[q - 1][0], centres[q - 1][1], centres[q][0], centres[q][1]]);
    // a line running along the axis, off it: a round stop on its own ribbon, joined to the nearest pill
    const dots: [number, number][] = [];
    for (const [, , px, py] of off) {
      if (dots.some((d) => Math.hypot(d[0] - px, d[1] - py) < 0.5 * pitch)) continue;
      dots.push([px, py]);
      pills.push([px, py, px, py, 0]);
      pillOf.push(k);
      let bq = -1, bd = Infinity;
      centres.forEach((c, q) => {
        const d = Math.hypot(c[0] - px, c[1] - py);
        if (d < bd) (bd = d), (bq = q);
      });
      if (bq >= 0) passages.push([centres[bq][0], centres[bq][1], px, py]);
      else centres.push([px, py]);
    }
    for (let q = firstPassage; q < passages.length; q++) passageOf.push(k);
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    x0 = Math.min(x0, lay.x[i]), x1 = Math.max(x1, lay.x[i]), y0 = Math.min(y0, lay.y[i]), y1 = Math.max(y1, lay.y[i]);
  }
  return { pitch, stroke: W, lines, lineOf, ribbons, chainAt, pills, pillOf, passages, passageOf, box: [x0, y0, x1, y1] };
}

/** a line's commands as a Path2D: 0 move to x, y; 1 line to x, y; 2 a quadratic through cx, cy to x, y */
function toPath(c: Float64Array): Path2D {
  const p = new Path2D();
  for (let i = 0; i < c.length; ) {
    const op = c[i];
    if (op === 2) p.quadraticCurveTo(c[i + 1], c[i + 2], c[i + 3], c[i + 4]), (i += 5);
    else (op ? p.lineTo(c[i + 1], c[i + 2]) : p.moveTo(c[i + 1], c[i + 2])), (i += 3);
  }
  return p;
}

/** whether any ribbon crosses the segment from (x0, y0) to (x1, y1) */
function crossesAny(ribs: { r: Ribbon }[], x0: number, y0: number, x1: number, y1: number): boolean {
  if ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) < 1e-18) return false;
  const lx = Math.min(x0, x1), hx = Math.max(x0, x1), ly = Math.min(y0, y1), hy = Math.max(y0, y1);
  const side = (ax: number, ay: number, bx: number, by: number, px: number, py: number) => (bx - ax) * (py - ay) - (by - ay) * (px - ax);
  for (const { r } of ribs) {
    const p = r.pts;
    for (let j = 0; j + 3 < p.length; j += 2) {
      if (Math.max(p[j], p[j + 2]) < lx || Math.min(p[j], p[j + 2]) > hx || Math.max(p[j + 1], p[j + 3]) < ly || Math.min(p[j + 1], p[j + 3]) > hy) continue;
      const d1 = side(x0, y0, x1, y1, p[j], p[j + 1]), d2 = side(x0, y0, x1, y1, p[j + 2], p[j + 3]);
      // a ribbon running along the segment does not cross it
      if (Math.abs(d1 - d2) < 0.2 * Math.hypot(x1 - x0, y1 - y0) * Math.hypot(p[j + 2] - p[j], p[j + 3] - p[j + 1])) continue;
      const d3 = side(p[j], p[j + 1], p[j + 2], p[j + 3], x0, y0), d4 = side(p[j], p[j + 1], p[j + 2], p[j + 3], x1, y1);
      if ((d1 > 0) !== (d2 > 0) && (d3 > 0) !== (d4 > 0)) return true;
    }
  }
  return false;
}

/** a line's drawn ribbon, sampled: x, y pairs, and per point the index in V it belongs to */
interface Ribbon { pts: number[]; vi: number[] }
const CURVE_SAMPLES = 8;

/** a path through V with each corner rounded, its radius by offset so a bundle's bends are concentric;
 * `out` gets the same path sampled */
function filleted(V: V[], R0: number, out: Ribbon): Float64Array | null {
  const P: V[] = [], src: number[] = [];
  V.forEach((v, i) => {
    if (!P.length || Math.hypot(v[0] - P[P.length - 1][0], v[1] - P[P.length - 1][1]) > 1e-9) P.push(v), src.push(i);
  });
  if (P.length < 2) return null;
  const cmd: number[] = [];
  const path = {
    moveTo: (x: number, y: number) => cmd.push(0, x, y),
    lineTo: (x: number, y: number) => cmd.push(1, x, y),
    quadraticCurveTo: (cx: number, cy: number, x: number, y: number) => cmd.push(2, cx, cy, x, y),
  };
  const at = (x: number, y: number, i: number) => {
    out.pts.push(x, y);
    out.vi.push(src[i]);
  };
  path.moveTo(P[0][0], P[0][1]);
  at(P[0][0], P[0][1], 0);
  for (let i = 1; i < P.length - 1; i++) {
    let ax = P[i][0] - P[i - 1][0], ay = P[i][1] - P[i - 1][1], bx = P[i + 1][0] - P[i][0], by = P[i + 1][1] - P[i][1];
    const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
    ax /= la, ay /= la, bx /= lb, by /= lb;
    const th = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
    if (Math.abs(th) < Math.PI / 180) {
      at(P[i][0], P[i][1], i);
      continue;
    }
    const r = R0 + P[i][3] + (th > 0 ? -P[i][2] : P[i][2]);
    const t = Math.min(Math.max(0, r) * Math.tan(Math.abs(th) / 2), 0.49 * la, 0.49 * lb);
    const x0 = P[i][0] - ax * t, y0 = P[i][1] - ay * t, x2 = P[i][0] + bx * t, y2 = P[i][1] + by * t;
    path.lineTo(x0, y0);
    path.quadraticCurveTo(P[i][0], P[i][1], x2, y2);
    for (let k = 0; k <= CURVE_SAMPLES; k++) {
      const u = k / CURVE_SAMPLES, a = (1 - u) * (1 - u), b = 2 * (1 - u) * u, c = u * u;
      at(a * x0 + b * P[i][0] + c * x2, a * y0 + b * P[i][1] + c * y2, i);
    }
  }
  path.lineTo(P[P.length - 1][0], P[P.length - 1][1]);
  at(P[P.length - 1][0], P[P.length - 1][1], P.length - 1);
  return Float64Array.from(cmd);
}

/** where a ribbon's points from V index `i0` to `i1` cross the line through (cx, cy) along `ax`,
 * nearest that place; failing a crossing, its point nearest the line */
function stopOn(r: Ribbon, i0: number, i1: number, cx: number, cy: number, ax: [number, number], reach: number): { cross: [number, number] | null; near: [number, number] } {
  const side = (j: number) => (r.pts[j * 2] - cx) * ax[1] - (r.pts[j * 2 + 1] - cy) * ax[0];
  let best: [number, number] | null = null, bd = Infinity, near: [number, number] = [cx, cy], nd = Infinity;
  for (let j = 0; j < r.vi.length; j++) {
    if (r.vi[j] < i0 || r.vi[j] > i1) continue;
    const s0 = side(j), dc = Math.hypot(r.pts[j * 2] - cx, r.pts[j * 2 + 1] - cy);
    if (dc < nd) (nd = dc), (near = [r.pts[j * 2], r.pts[j * 2 + 1]]);
    if (j + 1 >= r.vi.length || r.vi[j + 1] > i1) continue;
    const s1 = side(j + 1);
    if ((s0 > 0) === (s1 > 0) && s0 !== 0) continue;
    const f = s0 === s1 ? 0 : s0 / (s0 - s1), px = r.pts[j * 2] + (r.pts[j * 2 + 2] - r.pts[j * 2]) * f, py = r.pts[j * 2 + 1] + (r.pts[j * 2 + 3] - r.pts[j * 2 + 1]) * f;
    const d = Math.hypot(px - cx, py - cy);
    if (d < bd) (bd = d), (best = [px, py]);
  }
  // a crossing much farther off than the ribbon's nearest point is another stretch of it
  return { cross: best && bd < nd + 2 * reach ? best : null, near };
}

export interface DiagramTheme { bg: string; ink: string; pill: string; selected: string }

/** What is picked (T-2163): its lines drawn full and the rest muted, as the map draws a picked line's
 * others; its stations' pills and names full and the rest faded. */
export interface DiagramFocus {
  /** drawn lines (indices into `lines`) drawn full, null for all */
  lines: Set<number> | null;
  /** stations (graph nodes) drawn full, null for all */
  stations: Set<number> | null;
  /** stations outlined as selected */
  selected: Set<number>;
  /** a drawn line under the pointer, drawn full over the rest; -1 none */
  hot: number;
}
/** the muted lines: their width's share and opacity over the background (the map's, renderer.ts MUTE) */
export const MUTE_WIDTH = 0.45, MUTE_ALPHA = 0.5;
/** faded pills' outlines and names: their opacity */
export const FADE_ALPHA = 0.35;

/** the drawing on `ctx` at `scale` px per layout unit, (cx, cy) in the canvas's middle */
export function paint(ctx: CanvasRenderingContext2D, d: Drawing, w: number, h: number, cx: number, cy: number, scale: number, theme: DiagramTheme, focus?: DiagramFocus): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  ctx.setTransform(scale, 0, 0, -scale, w / 2 - cx * scale, h / 2 + cy * scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // casings first, so neighbours sit a gap apart and bundles read as one ribbon
  const full = (i: number) => !focus?.lines || focus.lines.has(i);
  const ribbon = (i: number, casing: boolean) => {
    const l = d.lines[i];
    if (casing) {
      ctx.strokeStyle = theme.bg;
      ctx.lineWidth = d.pitch;
      ctx.stroke((l.path ??= toPath(l.cmds)));
    }
    ctx.strokeStyle = l.colour;
    ctx.lineWidth = d.stroke;
    ctx.stroke((l.path ??= toPath(l.cmds)));
  };
  ctx.strokeStyle = theme.bg;
  ctx.lineWidth = d.pitch;
  for (const l of d.lines) ctx.stroke((l.path ??= toPath(l.cmds)));
  if (focus?.lines) {
    ctx.globalAlpha = MUTE_ALPHA;
    ctx.lineWidth = d.stroke * MUTE_WIDTH;
    d.lines.forEach((l, i) => {
      if (full(i) || i === focus.hot) return;
      ctx.strokeStyle = l.colour;
      ctx.stroke(l.path!);
    });
    ctx.globalAlpha = 1;
    for (const i of focus.lines) if (i !== focus.hot && d.lines[i]) ribbon(i, true);
  } else for (let i = 0; i < d.lines.length; i++) if (i !== focus?.hot) ribbon(i, false);
  if (focus && focus.hot >= 0 && d.lines[focus.hot]) ribbon(focus.hot, true);
  // far out a pill is a speck on its lines: none
  if (d.pitch * scale < PILL_MIN_PX) return;
  const rp = PILL_R * d.stroke, ol = PILL_OUTLINE * d.stroke;
  const faded = (k: number) => !!focus?.stations && !focus.stations.has(k) && !focus.selected.has(k);
  const outline = (k: number) => (focus?.selected.has(k) ? theme.selected : theme.ink);
  const bar = (x0: number, y0: number, x1: number, y1: number, width: number, colour: string, alpha = 1) => {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    // a round stop: a zero-length stroke's caps are not drawn
    if (x0 === x1 && y0 === y1) {
      ctx.fillStyle = colour;
      ctx.arc(x0, y0, width / 2, 0, 2 * Math.PI);
      ctx.fill();
      return;
    }
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  };
  // a selected station's outline is half again as thick
  const olOf = (k: number) => (focus?.selected.has(k) ? 1.5 * ol : ol);
  d.passages.forEach(([x0, y0, x1, y1], q) => bar(x0, y0, x1, y1, rp * PASSAGE * 2 + 2 * olOf(d.passageOf[q]), outline(d.passageOf[q]), faded(d.passageOf[q]) ? FADE_ALPHA : 1));
  // a faded pill's white over the background, so it recedes in the dark theme too
  const fill = (x0: number, y0: number, x1: number, y1: number, width: number, k: number) => {
    if (faded(k)) bar(x0, y0, x1, y1, width, theme.bg);
    bar(x0, y0, x1, y1, width, theme.pill, faded(k) ? FADE_ALPHA : 1);
  };
  d.passages.forEach(([x0, y0, x1, y1], q) => fill(x0, y0, x1, y1, rp * PASSAGE * 2, d.passageOf[q]));
  d.pills.forEach((p, i) => bar(p[0], p[1], p[2], p[3], 2 * rp + 2 * olOf(d.pillOf[i]) + 2 * p[4], outline(d.pillOf[i]), faded(d.pillOf[i]) ? FADE_ALPHA : 1));
  d.pills.forEach((p, i) => fill(p[0], p[1], p[2], p[3], 2 * rp + 2 * p[4], d.pillOf[i]));
  ctx.globalAlpha = 1;
}

/** what is under a point (layout units) at `scale` px a layout unit, `slop` px of reach: a station
 * (graph node) by its pill, else a drawn line and where along its ribbon (the point's index) */
export function pickAt(d: Drawing, x: number, y: number, scale: number, slop: number): { station: number } | { line: number; at: number } | null {
  const reach = slop / scale;
  if (d.pitch * scale >= PILL_MIN_PX) {
    const R = (PILL_R + PILL_OUTLINE) * d.stroke + reach;
    let best = -1, bd = Infinity;
    d.pills.forEach((p, i) => {
      const dd = segDist(x, y, p[0], p[1], p[2], p[3]) - p[4];
      if (dd < R && dd < bd) (bd = dd), (best = d.pillOf[i]);
    });
    if (best >= 0) return { station: best };
  }
  let line = -1, at = -1, bd = d.stroke / 2 + reach;
  d.ribbons.forEach((r, i) => {
    for (let j = 0; j + 3 < r.length; j += 2) {
      // a cheap box test before the distance
      if (Math.min(r[j], r[j + 2]) - bd > x || Math.max(r[j], r[j + 2]) + bd < x || Math.min(r[j + 1], r[j + 3]) - bd > y || Math.max(r[j + 1], r[j + 3]) + bd < y) continue;
      const dd = segDist(x, y, r[j], r[j + 1], r[j + 2], r[j + 3]);
      if (dd < bd) (bd = dd), (line = i), (at = j / 2);
    }
  });
  return line >= 0 ? { line, at } : null;
}

function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return Math.hypot(px - ax - dx * t, py - ay - dy * t);
}
