// The network diagram drawn (T-2160, T-2161; docs/ui-diagram.md): each line one continuous ribbon
// through its pieces, at its slot's offset, corners rounded so a bundle's bends are concentric; each
// station a white pill with an ink outline along an axis its lines cross, its stops where they cross it.
// Paths are in layout units (y north), drawn through one canvas transform.

import type { DiagramGraph } from "./graph";
import type { DiagramLayout, Piece } from "./layout";
import { atan2, DIRS8, hyp } from "./fmath";
import type { DiagramBorders } from "./borders";
import type { DiagramWater } from "./water";

/** a stroke is this share of a pitch wide; the rest is the gap between neighbours */
export const STROKE_SHARE = 5 / 6.4;
/** a corner's radius is this many stroke widths, plus the bundle's half width and the offset */
const RADIUS_W = 6;
/** at a station, so a line turning there passes near its place and under its pill (T-2244) */
const RADIUS_STATION_W = 3;
/** a pill's half width and its outline, in stroke widths */
export const PILL_R = 0.95, PILL_OUTLINE = 0.32;
/** a local stop's (`small`, T-2272) */
export const SMALL_R = 0.5, SMALL_OUTLINE = 0.24;
/** a high-speed line's centre stripe, in the background colour, this share of its stroke (T-2272) */
const FAST_CORE = 0.34;
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
/** `apart`'s passes, each moving a piece at most once (T-2405) */
const APART_PASSES = 4;
/** pills are drawn from a pitch this many device px (T-2272) */
const PILL_MIN_PX = 0.625;

export interface Drawing {
  /** a pitch, stroke width, layout units */
  pitch: number;
  stroke: number;
  /** per line its path as commands (`toPath`), built in the layout worker and posted, the Path2D made
   * on the page when first painted */
  lines: { colour: string; fast: boolean; cmds: Float64Array; path?: Path2D }[];
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
  /** stations with a round stop off their axis (a hook, for checks) */
  hooks: number[];
  /** per graph node, 1 for a local stop, drawn small (`localStops`) */
  small: Uint8Array;
  box: [number, number, number, number];
  /** the zones' borders (borders.ts), set in the layout worker after the names; the Path2D made on the page when
   * first painted */
  borders?: DiagramBorders & { path?: Path2D };
  /** the water (water.ts), set on the page when the layout worker posts it; the Path2Ds made when first painted */
  water?: DiagramWater & { riverPaths?: Path2D[]; endPaths?: Path2D[]; seaPath?: Path2D };
}

/** a polyline offset left by `o`, one point per point (mitred, capped) */
function offsetKeep(v: Float64Array, o: number): Float64Array {
  const n = v.length / 2, out = new Float64Array(v.length);
  const dir = (i: number, step: number): [number, number] | null => {
    for (; i >= 0 && i < n - 1; i += step) {
      const dx = v[i * 2 + 2] - v[i * 2], dy = v[i * 2 + 3] - v[i * 2 + 1], l = hyp(dx, dy);
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
    const l = hyp(mx, my);
    if (l < 1e-9) (mx = -a[1]), (my = a[0]);
    else (mx /= l), (my /= l);
    const c = Math.max(0.4, mx * -a[1] + my * a[0]);
    out[i * 2] = v[i * 2] + (mx * o) / c;
    out[i * 2 + 1] = v[i * 2 + 1] + (my * o) / c;
  }
  return out;
}

/** Local stops (T-2272), drawn small: stations one line stops at, or passed by a line that does not stop
 * while the lines stopping are all of one colour. A line's end never is one. */
export function localStops(g: DiagramGraph): Uint8Array {
  const n = g.key.length, stops = new Uint16Array(n), passed = new Uint8Array(n), end = new Uint8Array(n), many = new Uint8Array(n);
  const colour: (string | null)[] = new Array(n).fill(null), seen = new Int32Array(n).fill(-1);
  g.lines.forEach((ln, li) => {
    const c = ln.chain;
    if (!c.length) return;
    end[c[0]] = end[c[c.length - 1]] = 1;
    c.forEach((k, i) => {
      if (!ln.stop[i]) return void (passed[k] = 1);
      if (seen[k] === li) return;
      seen[k] = li;
      stops[k]++;
      if (colour[k] === null) colour[k] = ln.colour;
      else if (colour[k] !== ln.colour) many[k] = 1;
    });
  });
  const small = new Uint8Array(n);
  for (let k = 0; k < n; k++) small[k] = stops[k] > 0 && !end[k] && (stops[k] === 1 || (passed[k] && !many[k])) ? 1 : 0;
  return small;
}

/** a station's pill: its fill's half width and its outline, layout units */
export function pillSize(d: Drawing, k: number): [number, number] {
  return d.small[k] ? [SMALL_R * d.stroke, SMALL_OUTLINE * d.stroke] : [PILL_R * d.stroke, PILL_OUTLINE * d.stroke];
}

type V = [number, number, number, number, number]; // x, y, offset left of travel, bundle half width, node (-1 none)

/** Pieces leaving a node the same way (where the layout could turn none of them without folding a line)
 * would be drawn over each other: drawn as one bundle there instead (T-2405). Of each such group the piece
 * with the most strokes stays where it is, and the others go beside it, each to the side its stations lie
 * on in reality, nearest first. Per piece, how far its strokes are moved, in pitches left of its travel. */
export function besideShifts(g: DiagramGraph, lay: DiagramLayout): Float64Array {
  const out = new Float64Array(lay.pieces.length);
  const tiny = 1e-6 * lay.pu;
  const groups = new Map<number, [number, 0 | 1][]>();
  lay.pieces.forEach((pc, k) => {
    const v = pc.verts, m = v.length / 2;
    if (m < 2 || pc.nodes[0] === pc.nodes[pc.nodes.length - 1]) return;
    for (const end of [0, 1] as const) {
      let dx = 0, dy = 0;
      for (let j = 1; j < m && Math.abs(dx) + Math.abs(dy) < tiny; j++) {
        const a = end ? m - j : j - 1, b = end ? m - 1 - j : j;
        (dx = v[b * 2] - v[a * 2]), (dy = v[b * 2 + 1] - v[a * 2 + 1]);
      }
      if (Math.abs(dx) + Math.abs(dy) < tiny) continue;
      const port = ((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8, key = pc.nodes[end ? pc.nodes.length - 1 : 0] * 8 + port;
      const list = groups.get(key);
      if (list) list.push([k, end]);
      else groups.set(key, [[k, end]]);
    }
  });
  for (const [key, list] of groups) {
    if (list.length < 2) continue;
    const node = Math.floor(key / 8), [ux, uy] = DIRS8[key % 8];
    // which side of the way out each piece's stations lie in reality, left positive
    const side = list.map(([k]) => {
      const p = lay.pieces[k].nodes;
      let s = 0;
      for (const v of p) if (v !== node) s += (ux * (g.y[v] - g.y[node]) - uy * (g.x[v] - g.x[node])) / Math.max(1, hyp(g.x[v] - g.x[node], g.y[v] - g.y[node]));
      return s / Math.max(1, p.length - 1);
    });
    const width = (q: number) => lay.pieces[list[q][0]].strokes.length;
    const main = list.reduce((a, _, q) => (width(q) > width(a) ? q : a), 0);
    const rest = list.map((_, q) => q).filter((q) => q !== main).sort((a, b) => Math.abs(side[a]) - Math.abs(side[b]) || list[a][0] - list[b][0]);
    let left = width(main) / 2, right = width(main) / 2;
    for (const q of rest) {
      const w = width(q), toLeft = side[q] >= side[main];
      const s = toLeft ? left + w / 2 : -(right + w / 2);
      if (toLeft) left += w;
      else right += w;
      // left of the way out is left of travel from the piece's first node, right from its last
      const t = list[q][1] ? -s : s, k = list[q][0];
      if (Math.abs(t) > Math.abs(out[k])) out[k] = t;
    }
  }
  apart(g, lay, out);
  return out;
}

/** Legs of two pieces still drawn along each other, nearer than their bundles' half widths together (the
 * layout's spacing failed there), over more than a pitch: the piece with fewer strokes moved aside until the
 * two bundles sit side by side, the way it already lies, else the way its stations lie in reality
 * (`besideShifts`' second step, T-2405). Moves `shift` in place. */
function apart(g: DiagramGraph, lay: DiagramLayout, shift: Float64Array): void {
  const pu = lay.pu, tiny = 1e-6 * pu;
  // each leg: its piece, axis (0 to 3, a direction and its reverse), its way along the axis (1 or -1), where
  // it lies across the axis, and its span along it
  const legs: { k: number; q: number; sgn: number; c: number; t0: number; t1: number }[] = [];
  lay.pieces.forEach((pc, k) => {
    const v = pc.verts;
    for (let j = 2; j < v.length; j += 2) {
      const dx = v[j] - v[j - 2], dy = v[j + 1] - v[j - 1], l = hyp(dx, dy);
      if (l < pu) continue;
      const p = ((Math.round(atan2(dy, dx) / (Math.PI / 4)) % 8) + 8) % 8, q = p % 4, [ux, uy] = DIRS8[q];
      const ta = ux * v[j - 2] + uy * v[j - 1], tb = ux * v[j] + uy * v[j + 1];
      legs.push({ k, q, sgn: p < 4 ? 1 : -1, c: (-uy * (v[j - 2] + v[j]) + ux * (v[j - 1] + v[j + 1])) / 2, t0: Math.min(ta, tb), t1: Math.max(ta, tb) });
    }
  });
  if (legs.length < 2) return;
  const width = (k: number) => lay.pieces[k].strokes.length;
  let wmax = 1;
  for (const pc of lay.pieces) wmax = Math.max(wmax, pc.strokes.length);
  const cell = wmax * pu;
  /** where a leg's bundle's middle lies across its axis, its piece's shift taken */
  const mid = (L: (typeof legs)[number]) => L.c + shift[L.k] * pu * L.sgn;
  for (let pass = 0; pass < APART_PASSES; pass++) {
    const buckets = new Map<number, number[]>();
    legs.forEach((L, i) => {
      const key = L.q * 1e9 + Math.floor(mid(L) / cell);
      const b = buckets.get(key);
      if (b) b.push(i);
      else buckets.set(key, [i]);
    });
    const moved = new Set<number>();
    for (let i = 0; i < legs.length; i++) {
      const A = legs[i], key = A.q * 1e9 + Math.floor(mid(A) / cell);
      for (const nb of [key - 1, key, key + 1]) for (const j of buckets.get(nb) ?? []) {
        const B = legs[j];
        if (j <= i || B.k === A.k || B.q !== A.q || moved.has(A.k) || moved.has(B.k)) continue;
        if (Math.min(A.t1, B.t1) - Math.max(A.t0, B.t0) <= pu) continue;
        const need = ((width(A.k) + width(B.k)) / 2) * pu;
        if (Math.abs(mid(B) - mid(A)) >= need - tiny) continue;
        const [M, R] = width(A.k) < width(B.k) || (width(A.k) === width(B.k) && A.k > B.k) ? [A, B] : [B, A];
        const rel = mid(M) - mid(R);
        let dir = rel > tiny ? 1 : rel < -tiny ? -1 : 0;
        if (!dir) {
          // the way the moved piece's stations lie off the other's in reality
          const [ux, uy] = DIRS8[M.q], r0 = lay.pieces[R.k].nodes[0];
          let s = 0;
          for (const v of lay.pieces[M.k].nodes) s += -uy * (g.x[v] - g.x[r0]) + ux * (g.y[v] - g.y[r0]);
          dir = s < 0 ? -1 : 1;
        }
        shift[M.k] += ((dir * need - rel) / pu) * M.sgn;
        moved.add(M.k);
      }
    }
    if (!moved.size) return;
  }
}

export function drawing(g: DiagramGraph, lay: DiagramLayout): Drawing {
  const pitch = lay.pu, W = pitch * STROKE_SHARE, R0 = RADIUS_W * W, RS = RADIUS_STATION_W * W;
  const pieceOf = new Map<number, [Piece, number, number]>();
  const n = g.key.length;
  for (const r of lay.pieces) {
    const p = r.nodes;
    for (let i = 0; i + 1 < p.length; i++) {
      pieceOf.set(p[i] * n + p[i + 1], [r, i, 1]);
      pieceOf.set(p[i + 1] * n + p[i], [r, i + 1, -1]);
    }
  }
  /** per piece, per line on it: its stroke's offset, the piece's drawing at that offset, the bundle's half width */
  const offs = new Map<Piece, Map<number, [number, Float64Array, number]>>();
  const shift = besideShifts(g, lay);
  lay.pieces.forEach((r, pi) => {
    const m = new Map<number, [number, Float64Array, number]>(), k = r.strokes.length, sh = shift[pi] * pitch;
    r.strokes.forEach((ls, j) => {
      const o = ((k - 1) / 2 - j) * pitch + sh, s: [number, Float64Array, number] = [o, offsetKeep(r.verts, o), ((k - 1) / 2) * pitch + Math.abs(sh)];
      for (const l of ls) m.set(l, s);
    });
    offs.set(r, m);
  });
  const lines: Drawing["lines"] = [];
  const stopsAt = new Map<number, [number, number][]>();
  const Vs: [DiagramGraph["lines"][number], V[]][] = [];
  for (const [li, ln] of g.lines.entries()) {
    const ch = ln.chain;
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
      const [o, op, hw] = offs.get(r)!.get(li) ?? [0, r.verts, 0];
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
        const l1 = hyp(d1x, d1y), l2 = hyp(d2x, d2y);
        if (l1 > 1e-9 && l2 > 1e-9) {
          d1x /= l1, d1y /= l1, d2x /= l2, d2y /= l2;
          const cr = d1x * d2y - d1y * d2x;
          if (Math.abs(cr) > 0.02) {
            const t = ((q1[0] - p1[0]) * d2y - (q1[1] - p1[1]) * d2x) / cr;
            const xx = p1[0] + d1x * t, xy = p1[1] + d1y * t;
            if (hyp(xx - p1[0], xy - p1[1]) < 6 * pitch + 1e-9) {
              V[V.length - 1] = [xx, xy, p1[2], p1[3], p1[4]];
              run.shift();
            }
          } else if (hyp(q1[0] - p1[0], q1[1] - p1[1]) > 1e-6) {
            const jog = hyp(q1[0] - p1[0], q1[1] - p1[1]), ta = Math.min(jog, 0.45 * l1), tb = Math.min(jog, 0.45 * l2);
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
    return { r, cmds: filleted(V, R0, r, RS) };
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
    if (Math.abs(cr) < 0.2 * hyp(dx, dy)) return null;
    const t = ((cx - o[0]) * ax[1] - (cy - o[1]) * ax[0]) / cr;
    return t > 0.3 && Math.abs(t - 1) * hyp(dx, dy) < 3 * pitch ? [o[0] + dx * t, o[1] + dy * t] : null;
  };
  const AX: [number, number][] = [0, 1, 2, 3].map((q) => [DIRS8[q][0], DIRS8[q][1]]);
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
    ribs[li] = { r, cmds: filleted(Vs[li][1], R0, r, RS) };
  }
  const ribbons: Float64Array[] = [], lineOf: number[] = [], chainAt: Int32Array[] = [];
  ribs.forEach(({ r, cmds }, li) => {
    if (!cmds) return;
    lines.push({ colour: Vs[li][0].colour, fast: !!Vs[li][0].fast, cmds });
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
  const pills: Drawing["pills"] = [], passages: Drawing["passages"] = [], pillOf: number[] = [], passageOf: number[] = [], hooks: number[] = [];
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
    if (off.length && (on.length || off.length > 1)) hooks.push(k);
    for (const [, , px, py] of off) {
      if (dots.some((d) => hyp(d[0] - px, d[1] - py) < 0.5 * pitch)) continue;
      dots.push([px, py]);
      pills.push([px, py, px, py, 0]);
      pillOf.push(k);
      let bq = -1, bd = Infinity;
      centres.forEach((c, q) => {
        const d = hyp(c[0] - px, c[1] - py);
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
  return { pitch, stroke: W, lines, lineOf, ribbons, chainAt, pills, pillOf, passages, passageOf, hooks, small: localStops(g), box: [x0, y0, x1, y1] };
}

/** a line's (or the borders') commands as a Path2D: 0 move to x, y; 1 line to x, y; 2 a quadratic through cx, cy
 * to x, y */
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
      if (Math.abs(d1 - d2) < 0.2 * hyp(x1 - x0, y1 - y0) * hyp(p[j + 2] - p[j], p[j + 3] - p[j + 1])) continue;
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
function filleted(V: V[], R0: number, out: Ribbon, RS = R0): Float64Array | null {
  const P: V[] = [], src: number[] = [];
  V.forEach((v, i) => {
    if (!P.length || hyp(v[0] - P[P.length - 1][0], v[1] - P[P.length - 1][1]) > 1e-9) P.push(v), src.push(i);
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
    const la = hyp(ax, ay), lb = hyp(bx, by);
    ax /= la, ay /= la, bx /= lb, by /= lb;
    const sn = ax * by - ay * bx, cs = ax * bx + ay * by, th = atan2(sn, cs);
    if (Math.abs(th) < Math.PI / 180) {
      at(P[i][0], P[i][1], i);
      continue;
    }
    const r = (P[i][4] >= 0 ? RS : R0) + P[i][3] + (th > 0 ? -P[i][2] : P[i][2]);
    // tan(|th| / 2) from the unit vectors' sine and cosine
    const t = Math.min(Math.max(0, r) * (cs > -1 + 1e-12 ? Math.abs(sn) / (1 + cs) : 1e16), 0.49 * la, 0.49 * lb);
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
    const s0 = side(j), dc = hyp(r.pts[j * 2] - cx, r.pts[j * 2 + 1] - cy);
    if (dc < nd) (nd = dc), (near = [r.pts[j * 2], r.pts[j * 2 + 1]]);
    if (j + 1 >= r.vi.length || r.vi[j + 1] > i1) continue;
    const s1 = side(j + 1);
    if ((s0 > 0) === (s1 > 0) && s0 !== 0) continue;
    const f = s0 === s1 ? 0 : s0 / (s0 - s1), px = r.pts[j * 2] + (r.pts[j * 2 + 2] - r.pts[j * 2]) * f, py = r.pts[j * 2 + 1] + (r.pts[j * 2 + 3] - r.pts[j * 2 + 1]) * f;
    const d = hyp(px - cx, py - cy);
    if (d < bd) (bd = d), (best = [px, py]);
  }
  // a crossing much farther off than the ribbon's nearest point is another stretch of it
  return { cross: best && bd < nd + 2 * reach ? best : null, near };
}

/** `edge`: a thin outline round a pill's ring, so the ink ring shows on a dark background (T-2272) */
export interface DiagramTheme {
  bg: string; ink: string; pill: string; selected: string; edge?: string;
  /** the zones' borders' colour */
  border?: string;
  /** the water's colour: the map's, WATER_SHARE of the way from the background */
  water?: string;
}
/** the water is drawn this share of the way from the background to the map's water, so a river or coast drawn a little
 * wrong shows less (T-2516) */
export const WATER_SHARE = 0.55;
/** zone borders (T-2464): this share of a pitch wide at this opacity; below a device px, a px wide and fainter by its
 * width's share, though never below BORDER_THIN of it (the lines' rule, T-2408, with a higher floor) */
const BORDER_W = 0.4, BORDER_ALPHA = 0.126, BORDER_THIN = 0.5;
/** the edge's width, in stroke widths */
const EDGE_W = 0.16;

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
/** a line thinner than a device px is drawn a px wide at its width's share of its opacity, at least this (T-2408) */
const THIN_ALPHA = 0.25;
/** faded pills' outlines and names: their opacity */
export const FADE_ALPHA = 0.35;

/** the drawing on `ctx` at `scale` px per layout unit, (cx, cy) in the canvas's middle */
export function paint(ctx: CanvasRenderingContext2D, d: Drawing, w: number, h: number, cx: number, cy: number, scale: number, theme: DiagramTheme, focus?: DiagramFocus, under?: (ctx: CanvasRenderingContext2D) => void): void {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  ctx.setTransform(scale, 0, 0, -scale, w / 2 - cx * scale, h / 2 + cy * scale);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // the water under everything (T-2169): the sea filled, the rivers as bands, at least a device px wide
  const wa = d.water;
  if (wa && theme.water) {
    ctx.fillStyle = ctx.strokeStyle = theme.water;
    if (wa.sea?.length) ctx.fill((wa.seaPath ??= toPath(wa.sea)), "evenodd");
    // square-cut where a band's width steps, round at its ends (T-2528)
    const rp = (wa.riverPaths ??= wa.rivers.map(toPath)), re = (wa.endPaths ??= (wa.riverEnds ?? []).map(toPath));
    ctx.lineCap = "butt";
    for (let k = 0; k < rp.length; k++) {
      ctx.lineWidth = Math.max(wa.bands[k], 1 / scale);
      ctx.stroke(rp[k]);
    }
    ctx.lineCap = "round";
    for (let k = 0; k < re.length; k++) {
      ctx.lineWidth = Math.max(wa.bands[k], 1 / scale);
      ctx.stroke(re[k]);
    }
  }
  // the zones' borders under everything else; the lines' casings cut them
  const bd = d.borders;
  if (bd && theme.border && bd.cmds.length) {
    ctx.strokeStyle = theme.border;
    const px = BORDER_W * d.pitch * scale;
    ctx.lineWidth = px < 1 ? 1 / scale : BORDER_W * d.pitch;
    ctx.globalAlpha = BORDER_ALPHA * (px < 1 ? Math.max(BORDER_THIN, px) : 1);
    ctx.stroke((bd.path ??= toPath(bd.cmds)));
    ctx.globalAlpha = 1;
  }
  if (under) (ctx.save(), under(ctx), ctx.restore());
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
    const thin = width(d.stroke, 1);
    ctx.stroke((l.path ??= toPath(l.cmds)));
    ctx.globalAlpha = 1;
    // a high-speed line: two strokes of its colour either side of a stripe of the background
    if (l.fast && !thin) {
      ctx.strokeStyle = theme.bg;
      ctx.lineWidth = d.stroke * FAST_CORE;
      ctx.stroke(l.path);
    }
  };
  /** a stroke `w` wide at opacity `alpha`: under a device px, a px wide and fainter by as much, though never
   * below THIN_ALPHA of it, so no line vanishes far out (T-2408); whether it was */
  const width = (w: number, alpha: number): boolean => {
    const px = w * scale;
    ctx.lineWidth = px < 1 ? 1 / scale : w;
    ctx.globalAlpha = px < 1 ? alpha * Math.max(THIN_ALPHA, px) : alpha;
    return px < 1;
  };
  ctx.strokeStyle = theme.bg;
  ctx.lineWidth = d.pitch;
  for (const l of d.lines) ctx.stroke((l.path ??= toPath(l.cmds)));
  if (focus?.lines) {
    width(d.stroke * MUTE_WIDTH, MUTE_ALPHA);
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
  const faded = (k: number) => !!focus?.stations && !focus.stations.has(k) && !focus.selected.has(k);
  const outline = (k: number) => (focus?.selected.has(k) ? theme.selected : theme.ink);
  const bar = (x0: number, y0: number, x1: number, y1: number, width: number, colour: string, alpha = 1) => {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    // a round stop: a zero-length stroke's caps are not drawn, nor a stroke whose ends differ in the last
    // bits (T-2290: West Mount Vernon and Melrose drew no pill)
    if (hyp(x1 - x0, y1 - y0) < 1e-6 * width) {
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
  const olOf = (k: number) => (focus?.selected.has(k) ? 1.5 : 1) * pillSize(d, k)[1];
  const rp = (k: number) => pillSize(d, k)[0];
  if (theme.edge) {
    const e = 2 * EDGE_W * d.stroke, edge = theme.edge;
    d.passages.forEach(([x0, y0, x1, y1], q) => bar(x0, y0, x1, y1, rp(d.passageOf[q]) * PASSAGE * 2 + 2 * olOf(d.passageOf[q]) + e, edge, faded(d.passageOf[q]) ? FADE_ALPHA : 1));
    d.pills.forEach((p, i) => bar(p[0], p[1], p[2], p[3], 2 * rp(d.pillOf[i]) + 2 * olOf(d.pillOf[i]) + 2 * p[4] + e, edge, faded(d.pillOf[i]) ? FADE_ALPHA : 1));
  }
  d.passages.forEach(([x0, y0, x1, y1], q) => bar(x0, y0, x1, y1, rp(d.passageOf[q]) * PASSAGE * 2 + 2 * olOf(d.passageOf[q]), outline(d.passageOf[q]), faded(d.passageOf[q]) ? FADE_ALPHA : 1));
  // a faded pill's white over the background, so it recedes in the dark theme too
  const fill = (x0: number, y0: number, x1: number, y1: number, width: number, k: number) => {
    if (faded(k)) bar(x0, y0, x1, y1, width, theme.bg);
    bar(x0, y0, x1, y1, width, theme.pill, faded(k) ? FADE_ALPHA : 1);
  };
  d.passages.forEach(([x0, y0, x1, y1], q) => fill(x0, y0, x1, y1, rp(d.passageOf[q]) * PASSAGE * 2, d.passageOf[q]));
  d.pills.forEach((p, i) => bar(p[0], p[1], p[2], p[3], 2 * rp(d.pillOf[i]) + 2 * olOf(d.pillOf[i]) + 2 * p[4], outline(d.pillOf[i]), faded(d.pillOf[i]) ? FADE_ALPHA : 1));
  d.pills.forEach((p, i) => fill(p[0], p[1], p[2], p[3], 2 * rp(d.pillOf[i]) + 2 * p[4], d.pillOf[i]));
  ctx.globalAlpha = 1;
}

/** what is under a point (layout units) at `scale` px a layout unit, `slop` px of reach: a station
 * (graph node) by its pill, else a drawn line and where along its ribbon (the point's index) */
export function pickAt(d: Drawing, x: number, y: number, scale: number, slop: number): { station: number } | { line: number; at: number } | null {
  const reach = slop / scale;
  if (d.pitch * scale >= PILL_MIN_PX) {
    let best = -1, bd = Infinity;
    d.pills.forEach((p, i) => {
      const [r, o] = pillSize(d, d.pillOf[i]);
      // measured from the outline, so a small stop beside a big one is still picked
      const dd = segDist(x, y, p[0], p[1], p[2], p[3]) - p[4] - r - o;
      if (dd < reach && dd < bd) (bd = dd), (best = d.pillOf[i]);
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
  return hyp(px - ax - dx * t, py - ay - dy * t);
}
