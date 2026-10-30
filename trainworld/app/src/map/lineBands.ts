// Lines sharing track drawn side by side (T-1799, notes/T-1799.md): each edge's lines in an order,
// left to right, and each line's sideways shift along its path in line widths, which the stroke and
// train shaders scale by the line width on screen.
//
// The order is the pairwise rule of metro-line crossing minimisation: of two lines on an edge, follow
// both ahead to the node where they part and put the one turning more to the left on the left; if
// they never part ahead, the same looking back; same route both ways, by line id. "Ahead" is the lower
// id line's own direction, whichever way each edge was drawn. One parting decides
// every edge of a stretch they share, so bands cross only at a node where their routes make them.

import { findCorridors } from "./corridors";
import { kAtY } from "./geo";

/** T-1821: the Line view's line width as a multiple of the width the Traffic view's scale is set
 * against (Anita, 2026-10-29: "raise normal line thickness by like 30% relative to traffic mode") */
export const LINE_VIEW_WIDTH = 1.3;
/** a band's width as a share of a line's (Anita, 2026-10-29) */
export const BAND_SHARE = 0.8;
/** line widths a metre across is drawn as, for the gap between the centre lines of the edges of a
 * corridor (8 m between a four-track trunk's pairs): 0.4 of a line width for 8 m, which is exact near
 * zoom 14.5, 1.4 px of overlap at zoom 13 and 0.9 px of gap at 15, a few px of gap by 16 (a metre is
 * fixed on the ground, a line width is not; notes/T-1799.md, stage 3) */
export const BAND_LW_PER_M = 0.05;
/** where a line's slot changes at a node, its shift eases over this much of the edge, m */
export const BAND_RAMP_M = 30;
/** a band's shift is stored as a share of this many line widths (`Strokes.band`, Int16 normalised) */
export const BAND_RANGE = 16;
/** per path entry in `LineBands.of`: shift at its start, along it and at its end (line widths, left of
 * travel), and the same three of `1 - width share` */
export const BAND_STRIDE = 6;

interface BandLine { id: number; ok: boolean; path: number[]; bus?: unknown }
interface BandEdge { id: number; a: number; b: number }

export interface LineBands {
  /** per line id: BAND_STRIDE numbers per path entry; absent for a line on no shared track */
  of: Map<number, Float32Array>;
  /** per edge with two or more lines: their ids left to right in its a to b direction */
  order: Map<number, number[]>;
  /** the pairwise rule on an edge between two of its lines (checks/t1799-bands.mjs): < 0 the first on the left in a to b */
  rule: (edge: number, x: number, y: number) => number;
  /** per edge in a corridor, CORRIDOR_STRIDE numbers: the edge, its corridor, how far its centre line is
   * from the corridor's middle (line widths, left of the corridor's direction), and that direction (local x, y) */
  corridor: Float32Array;
}

export const CORRIDOR_STRIDE = 5;


/** how far along an edge its leaving direction is measured, m, so a short first segment does not decide it */
const BEARING_M = 60;

/** The bands of every line. `pts`: each edge's local polyline (x, y, z triples, a to b). */
export function lineBands(lines: readonly BandLine[], edges: readonly BandEdge[], pts: ReadonlyMap<number, Float32Array>): LineBands {
  const ends = new Map(edges.map((e) => [e.id, e]));
  /** per edge: the lines on it, each with its first entry in its path */
  const on = new Map<number, { line: BandLine; k: number }[]>();
  const rail = lines.filter((l) => l.ok && !l.bus && l.path.length);
  for (const l of [...rail].sort((a, b) => a.id - b.id)) {
    l.path.forEach((p, k) => {
      const e = p >> 1;
      let v = on.get(e);
      if (!v) on.set(e, (v = []));
      if (!v.some((x) => x.line === l)) v.push({ line: l, k });
    });
  }
  // the unit direction an edge leaves a node in, y up (local y points south)
  const leave = new Map<number, [number, number]>();
  const dirAt = (edge: number, node: number): [number, number] => {
    const key = edge * 2 + (ends.get(edge)?.a === node ? 0 : 1);
    let v = leave.get(key);
    if (v) return v;
    const p = pts.get(edge), e = ends.get(edge);
    v = [1, 0];
    if (p && e && p.length >= 6) {
      const n = p.length / 3, fromA = e.a === node;
      const at = (i: number) => (fromA ? i : n - 1 - i);
      const x0 = p[at(0) * 3], y0 = p[at(0) * 3 + 1];
      let x = x0, y = y0, d = 0;
      const reach = BEARING_M * kAtY(y0);
      for (let i = 1; i < n && d < reach; i++) {
        const nx = p[at(i) * 3], ny = p[at(i) * 3 + 1];
        d += Math.hypot(nx - x, ny - y);
        x = nx;
        y = ny;
      }
      const l = Math.hypot(x - x0, y - y0);
      if (l > 0) v = [(x - x0) / l, -(y - y0) / l];
    }
    leave.set(key, v);
    return v;
  };
  /** how far left (rad) leaving along `out` turns from arriving along `inEdge`, at `node` */
  const turn = (inEdge: number, out: number, node: number) => {
    const [ax, ay] = dirAt(inEdge, node), [bx, by] = dirAt(out, node);
    // arriving is the reverse of leaving along the edge in
    return Math.atan2(-ax * by + ay * bx, -ax * bx - ay * by);
  };
  /** line `l`'s entries walked from entry k in direction `fwd` of its path: the edge and the node it
   * is entered from, step by step; null past the path's end */
  const step = (l: BandLine, k: number, fwd: boolean, j: number): [number, number] | null => {
    const i = fwd ? k + j : k - j;
    if (i < 0 || i >= l.path.length) return null;
    const p = l.path[i], e = ends.get(p >> 1);
    if (!e) return null;
    // entered from a when run forwards a to b (low bit clear), or walked backwards b to a
    const fromA = ((p & 1) === 0) === fwd;
    return [p >> 1, fromA ? e.a : e.b];
  };
  /** of two lines on `edge`, walking towards `toB` (its b end) or away: < 0 when `x` turns left of
   * `y` where they part, > 0 right, 0 if they never part */
  const part = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }, toB: boolean): number => {
    // each line's own direction on this edge: walk its path forwards when it runs the way we walk
    const fx = ((x.line.path[x.k] & 1) === 0) === toB, fy = ((y.line.path[y.k] & 1) === 0) === toB;
    let prev = edge;
    const max = Math.min(x.line.path.length, y.line.path.length);
    for (let j = 1; j <= max; j++) {
      const a = step(x.line, x.k, fx, j), b = step(y.line, y.k, fy, j);
      if (!a || !b) return 0;
      if (a[0] !== b[0] || a[1] !== b[1]) {
        if (a[1] !== b[1]) return 0;
        const ta = turn(prev, a[0], a[1]), tb = turn(prev, b[0], b[1]);
        return ta > tb ? -1 : ta < tb ? 1 : 0;
      }
      prev = a[0];
    }
    return 0;
  };
  const order = new Map<number, number[]>();
  /** < 0: `x` left of `y` in the edge's a to b direction. Looked at first ahead in the lower-id
   * line's own direction, so every edge of a stretch the two share asks the same parting node
   * whatever way each edge was drawn; a walk towards a sees left of travel as right of a to b. */
  const compare = (edge: number, x: { line: BandLine; k: number }, y: { line: BandLine; k: number }) => {
    const lead = x.line.id < y.line.id ? x : y;
    const toB = (lead.line.path[lead.k] & 1) === 0;
    const first = part(edge, x, y, toB), second = first ? 0 : part(edge, x, y, !toB);
    // never parting: the lower id on the left of the lead line's travel
    return (toB ? first : -first) || (toB ? -second : second) || (toB ? 1 : -1) * (x.line.id - y.line.id);
  };
  for (const [edge, v] of on) {
    if (v.length < 2) continue;
    // by how many of the others each is left of: the rule's own order where it is one, and where
    // three or more lines' partings go round in a circle (some pair has to cross somewhere), a
    // fixed order rather than whatever a sort makes of a comparison that contradicts itself
    const m = v.length, wins = new Int32Array(m), c = new Int8Array(m * m);
    for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
      const r = Math.sign(compare(edge, v[i], v[j]));
      c[i * m + j] = r;
      c[j * m + i] = -r;
      if (r < 0) wins[i]++;
      else if (r > 0) wins[j]++;
    }
    const idx = v.map((_, i) => i).sort((i, j) => wins[j] - wins[i] || c[i * m + j] || v[i].line.id - v[j].line.id);
    const sorted = idx.map((i) => v[i]);
    order.set(edge, sorted.map((s) => s.line.id));
  }
  // each line's shift on each edge it is on, line widths left of the edge's a to b direction
  const pos = new Map<number, Map<number, number>>();
  const cnt = new Map<number, number>();
  const cor = on.size > 1 ? findCorridors([...on.keys()], pts) : null;
  for (const [edge, o] of order) {
    if (cor?.of.has(edge)) continue;
    pos.set(edge, new Map(o.map((id, i) => [id, ((o.length - 1) / 2 - i) * BAND_SHARE])));
    cnt.set(edge, o.length);
  }
  const corr: number[] = [];
  for (const [cid, members] of cor?.members ?? []) {
    // side by side (T-1799 stage 3): the left edge's lines on the left, one ribbon centred on the
    // corridor's middle; an edge's centre line is off the middle by `y` m, which `BAND_LW_PER_M` turns
    // into line widths so the bands of neighbouring edges touch
    const seq: [number, number][] = [];
    for (const e of members) {
      const ids = order.get(e) ?? on.get(e)!.map((x) => x.line.id);
      for (const id of cor!.of.get(e)!.s > 0 ? ids : [...ids].reverse()) seq.push([e, id]);
    }
    const K = seq.length, dir = cor!.dir.get(cid)!;
    seq.forEach(([e, id], g) => {
      const c = cor!.of.get(e)!;
      let m = pos.get(e);
      if (!m) pos.set(e, (m = new Map()));
      m.set(id, c.s * (((K - 1) / 2 - g) * BAND_SHARE - c.y * BAND_LW_PER_M));
      cnt.set(e, K);
    });
    for (const e of members) corr.push(e, cid, cor!.of.get(e)!.y * BAND_LW_PER_M, dir[0], dir[1]);
  }
  const of = new Map<number, Float32Array>();
  for (const l of rail) {
    const n = l.path.length;
    // per entry: its shift along the edge, left of this line's travel, its width share, lines there
    const mid = new Float32Array(n), q = new Float32Array(n), count = new Int32Array(n);
    let any = false;
    for (let k = 0; k < n; k++) {
      const e = l.path[k] >> 1, sl = pos.get(e);
      count[k] = cnt.get(e) ?? on.get(e)?.length ?? 1;
      if (!sl) continue;
      any = true;
      const s = sl.get(l.id)!;
      mid[k] = (l.path[k] & 1) === 1 ? -s : s;
      q[k] = 1 - BAND_SHARE;
    }
    if (!any) continue;
    const out = new Float32Array(n * BAND_STRIDE);
    for (let k = 0; k < n; k++) {
      // at a node the line keeps the bigger bundle's place into the edge with fewer lines and eases
      // to its own there (T-1799 stage 4: where lines part they leave side by side and close up as
      // they curve out, not snap together first); between equal ones, half way
      const atNode = (j: number): [number, number] => {
        if (j < 0 || j >= n) return [mid[k], q[k]];
        if (count[j] > count[k]) return [mid[j], q[j]];
        if (count[j] < count[k]) return [mid[k], q[k]];
        return [(mid[j] + mid[k]) / 2, (q[j] + q[k]) / 2];
      };
      const [s0, q0] = atNode(k - 1), [s1, q1] = atNode(k + 1);
      out.set([s0, mid[k], s1, q0, q[k], q1], k * BAND_STRIDE);
    }
    of.set(l.id, out);
  }
  const rule = (edge: number, x: number, y: number) => {
    const v = on.get(edge), a = v?.find((s) => s.line.id === x), b = v?.find((s) => s.line.id === y);
    return a && b ? compare(edge, a, b) : 0;
  };
  return { of, order, rule, corridor: new Float32Array(corr) };
}

/** `TrafficBands.pick` holds a shift as a share of this many times BAND_RANGE line widths, since a
 * trunk of busy lines is wider than BAND_RANGE */
export const TRAFFIC_PICK_DIV = 4;

export interface TrafficBands {
  /** per line stroke segment, 2: its band's shift at its start and end, line widths left of its direction */
  seg: Float32Array;
  /** per train sample: its line's band's shift there, line widths left of run 0 (as `sampleBand`) */
  sample: Float32Array;
  /** per segment, `Strokes.band`'s layout for picking: the shifts over TRAFFIC_PICK_DIV, full width */
  pick: Int16Array;
}

interface TrafficStrokes { seg: Float32Array; colour: Uint32Array; edge: Uint32Array; band?: Int16Array; count: number }

/**
 * The Traffic view's bands (T-1799 stage 2): on an edge the Line view draws side by side, the same
 * order, each band as wide as its line's riders (`width`, per segment, line widths), so the ribbon is
 * their sum; eased at nodes as the Line view's (`lineBands`' rule on which side of a node the ease
 * sits, over BAND_RAMP_M, network.ts `bandPoints` cutting the segments there). The order is read back
 * from the Line view's shifts in `L.band`. `perM`: local units per metre at a y. `samples` and
 * `sampleStroke` and `corridor` as `NetworkBuffers`'. A corridor's edges are one ribbon (stage 3).
 * Null when no edge is shared.
 */
export function trafficBands(L: TrafficStrokes, width: Float32Array, perM: (y: number) => number, samples: Float32Array, sampleStroke: Int32Array | undefined, corridor?: Float32Array): TrafficBands | null {
  const n = L.count, B = L.band, seg = L.seg;
  if (!B || B.length < n * 4) return null;
  const S = BAND_RANGE / 32767;
  // runs: one line's consecutive segments on one edge, one path entry each, a line's in path order
  const start: number[] = [];
  for (let i = 0; i < n; ) {
    let j = i + 1;
    while (j < n && L.edge[j] === L.edge[i] && L.colour[j] === L.colour[i]) j++;
    start.push(i);
    i = j;
  }
  const R = start.length;
  start.push(n);
  const runOf = new Int32Array(n);
  const len = new Float64Array(R), mid = new Float64Array(R), sgn = new Int8Array(R).fill(1), count = new Int32Array(R).fill(1);
  const T = new Float64Array(R), off = new Float64Array(R);
  const cor = new Map<number, { id: number; off: number; dx: number; dy: number }>();
  for (let i = 0; corridor && i + CORRIDOR_STRIDE <= corridor.length; i += CORRIDOR_STRIDE) {
    cor.set(corridor[i], { id: corridor[i + 1], off: corridor[i + 2], dx: corridor[i + 3], dy: corridor[i + 4] });
  }
  const byEdge = new Map<number, number[]>();
  let any = false;
  for (let r = 0; r < R; r++) {
    const i0 = start[r], i1 = start[r + 1];
    let at = 0;
    for (let i = i0; i < i1; i++) {
      runOf[i] = r;
      at += Math.hypot(seg[i * 4 + 2] - seg[i * 4], seg[i * 4 + 3] - seg[i * 4 + 1]);
    }
    len[r] = at;
    // the Line view's shift at the end point nearest the middle, which lies between the eases
    let best = Infinity, shared = false;
    at = 0;
    for (let i = i0; i < i1; i++) {
      for (const e of [0, 1]) {
        const d = Math.abs(at - len[r] / 2);
        if (d < best) (best = d), (mid[r] = B[i * 4 + e] * S), (shared = B[i * 4 + 2 + e] > 0);
        if (e === 0) at += Math.hypot(seg[i * 4 + 2] - seg[i * 4], seg[i * 4 + 3] - seg[i * 4 + 1]);
      }
    }
    if (!shared) continue;
    any = true;
    // a corridor's edges are one ribbon (negative keys), any other edge its own
    const c = cor.get(L.edge[i0]);
    const e = c ? -1 - c.id : L.edge[i0];
    let v = byEdge.get(e);
    if (!v) byEdge.set(e, (v = []));
    v.push(r);
  }
  if (!any) return null;
  for (const v of byEdge.values()) {
    // each run's direction against the first's (a corridor's: against its direction), by their
    // chords: the shifts in one frame
    const chord = (r: number) => {
      const a = start[r], b = start[r + 1] - 1;
      return [seg[b * 4 + 2] - seg[a * 4], seg[b * 4 + 3] - seg[a * 4 + 1]];
    };
    const c0 = cor.get(L.edge[start[v[0]]]);
    const [rx, ry] = c0 ? [c0.dx, c0.dy] : chord(v[0]);
    const first = new Map<number, number>();
    for (const r of v) {
      const [x, y] = chord(r);
      sgn[r] = x * rx + y * ry >= 0 ? 1 : -1;
      // where the edge's centre line lies from the ribbon's middle, in the same frame
      off[r] = cor.get(L.edge[start[r]])?.off ?? 0;
      if (!first.has(L.colour[start[r]])) first.set(L.colour[start[r]], r);
    }
    if (first.size < 2) continue;
    const at = (r: number) => mid[r] * sgn[r] + off[r];
    const lines = [...first.values()].sort((a, b) => at(b) - at(a));
    let total = 0;
    for (const r of lines) total += width[start[r]];
    const own = new Map<number, number>();
    let left = 0;
    for (const r of lines) {
      const w = width[start[r]];
      own.set(L.colour[start[r]], total / 2 - left - w / 2);
      left += w;
    }
    for (const r of v) {
      T[r] = (own.get(L.colour[start[r]])! - off[r]) * sgn[r];
      count[r] = first.size;
    }
  }
  const out = new Float32Array(n * 2), pick = new Int16Array(n * 4);
  const P = 32767 / (BAND_RANGE * TRAFFIC_PICK_DIV);
  const ease = new Float64Array(BAND_STRIDE);
  for (let r = 0; r < R; r++) {
    const i0 = start[r], i1 = start[r + 1];
    const same = (q: number) => q >= 0 && q < R && L.colour[start[q]] === L.colour[i0];
    if (!T[r] && !(same(r - 1) && T[r - 1]) && !(same(r + 1) && T[r + 1])) continue;
    // as lineBands' `atNode`: the edge with fewer lines takes the bigger one's place at the node, half way between equals
    const atNode = (q: number) => (!same(q) ? T[r] : count[q] > count[r] ? T[q] : count[q] < count[r] ? T[r] : (T[q] + T[r]) / 2);
    ease[0] = atNode(r - 1);
    ease[1] = T[r];
    ease[2] = atNode(r + 1);
    const k = perM(seg[i0 * 4 + 1]), lm = len[r] / k;
    let at = 0;
    for (let i = i0; i < i1; i++) {
      const l = Math.hypot(seg[i * 4 + 2] - seg[i * 4], seg[i * 4 + 3] - seg[i * 4 + 1]);
      const s0 = bandAlong(ease, 0, at / k, lm)[0], s1 = bandAlong(ease, 0, (at + l) / k, lm)[0];
      at += l;
      out[i * 2] = s0;
      out[i * 2 + 1] = s1;
      pick[i * 4] = Math.round(s0 * P);
      pick[i * 4 + 1] = Math.round(s1 * P);
    }
  }
  // a train sample's run is its stroke's (the first segment of its path entry); where it is along
  // the run, by walking the run's segments forwards with the line's samples
  const ns = samples.length / 2, sample = new Float32Array(ns);
  let lastRun = -1, s = 0;
  for (let q = 0; sampleStroke && q < ns; q++) {
    const f = sampleStroke[q];
    if (f < 0 || f >= n) continue;
    const r = runOf[f], i1 = start[r + 1];
    if (r !== lastRun) (lastRun = r), (s = start[r]);
    if (out[start[r] * 2] === 0 && out[(i1 - 1) * 2 + 1] === 0 && T[r] === 0) continue;
    const x = samples[q * 2], y = samples[q * 2 + 1];
    for (;;) {
      const ax = seg[s * 4], ay = seg[s * 4 + 1], dx = seg[s * 4 + 2] - ax, dy = seg[s * 4 + 3] - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / l2 : 0;
      if (t > 1 && s < i1 - 1) {
        s++;
        continue;
      }
      const u = Math.max(0, Math.min(1, t));
      sample[q] = out[s * 2] + (out[s * 2 + 1] - out[s * 2]) * u;
      break;
    }
  }
  return { seg: out, sample, pick };
}

/** A path entry's band at `t` metres from its start (its travel direction) on an edge `len` m long:
 * [shift, 1 - width share]. `b`: its BAND_STRIDE numbers at `o`. */
export function bandAlong(b: ArrayLike<number>, o: number, t: number, len: number): [number, number] {
  const r = Math.min(BAND_RAMP_M, len / 2);
  if (r > 0 && t < r) {
    const f = Math.max(0, t / r);
    return [b[o] + (b[o + 1] - b[o]) * f, b[o + 3] + (b[o + 4] - b[o + 3]) * f];
  }
  if (r > 0 && t > len - r) {
    const f = Math.min(1, (t - (len - r)) / r);
    return [b[o + 1] + (b[o + 2] - b[o + 1]) * f, b[o + 4] + (b[o + 5] - b[o + 4]) * f];
  }
  return [b[o + 1], b[o + 4]];
}
