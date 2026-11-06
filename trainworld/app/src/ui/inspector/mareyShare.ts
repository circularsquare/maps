// T-2660: the timetable chart's other lines. Where each runs over the chart line's track, as a map
// from its run 0 offsets onto the chart line's, from the network's path table (`pathTable`, the
// train guard's); and where trains of other routes cross the chart line's path at grade, from the
// train guard's conflicts (`TrackApi.guard_conflicts`: flat crossings, and junction moves that cross).

import type { NetworkBuffers } from "../../map/network";

/** as the train guard's: a move's key from its arriving and leaving path entries */
const MOVE_KEY = 2 ** 26;

/** A stretch another line shares: its run 0 offsets `a` to `b` are the chart line's `c + sg * (s - a)`. */
export interface Shared { a: number; b: number; c: number; sg: 1 | -1 }
/** Trains of line `o` passing an at-grade conflict on the chart line's path: `sO` along `o`'s run 0,
 * `sL` along the chart line's; `run` the run that makes the conflicting move, -1 both. */
export interface Pass { o: number; run: number; sO: number; sL: number }

interface Index {
  /** per edge: line, path entry index pairs */
  byEdge: Map<number, number[]>;
  /** per junction move: line, run, offset along run 0 triples */
  byMove: Map<number, number[]>;
}
const indexes = new WeakMap<NetworkBuffers, Index>();

function indexOf(n: NetworkBuffers): Index | null {
  const T = n.pathTable, E = n.pathEntry, PF = n.pathFrom;
  if (!T || !E || !PF) return null;
  const had = indexes.get(n);
  if (had) return had;
  const byEdge = new Map<number, number[]>(), byMove = new Map<number, number[]>();
  const push = (m: Map<number, number[]>, key: number, ...v: number[]) => {
    const a = m.get(key);
    if (a) a.push(...v);
    else m.set(key, v);
  };
  for (let l = 0; 2 * l + 1 < T.length; l++) {
    const f = T[2 * l], c = T[2 * l + 1];
    // a bus runs on roads
    if (!c || (n.lineCars?.[l] ?? 8) < 1) continue;
    for (let k = f; k < f + c; k++) {
      push(byEdge, E[k] >> 1, l, k);
      if (k + 1 < f + c) {
        push(byMove, E[k] * MOVE_KEY + E[k + 1], l, 0, PF[k + 1]);
        push(byMove, (E[k + 1] ^ 1) * MOVE_KEY + (E[k] ^ 1), l, 1, PF[k + 1]);
      }
    }
  }
  const ix = { byEdge, byMove };
  indexes.set(n, ix);
  return ix;
}

/** How far along run 0 path entry `k` is at `frac` of its edge from the edge's node a. */
function along(n: NetworkBuffers, k: number, frac: number): number {
  return n.pathFrom![k] + ((n.pathEntry![k] & 1) === 0 ? frac : 1 - frac) * n.pathLen![k];
}

/** Every other rail line over `line`'s track, with the stretches it shares, kept to where `line`'s
 * run 0 offset is between `d0` and `d1` (its first and last stops). */
export function sharedStretches(n: NetworkBuffers, line: number, d0: number, d1: number): Map<number, Shared[]> {
  const out = new Map<number, Shared[]>();
  const ix = indexOf(n);
  const T = n.pathTable, E = n.pathEntry, PF = n.pathFrom, PL = n.pathLen;
  if (!ix || !T || !E || !PF || !PL || 2 * line + 1 >= T.length) return out;
  const f = T[2 * line], cnt = T[2 * line + 1];
  const raw = new Map<number, Shared[]>();
  for (let kL = f; kL < f + cnt; kL++) {
    const on = ix.byEdge.get(E[kL] >> 1);
    if (!on) continue;
    for (let i = 0; i < on.length; i += 2) {
      const o = on[i], kO = on[i + 1];
      if (o === line) continue;
      const same = ((E[kO] ^ E[kL]) & 1) === 0;
      const s: Shared = { a: PF[kO], b: PF[kO] + PL[kO], c: same ? PF[kL] : PF[kL] + PL[kL], sg: same ? 1 : -1 };
      const v = raw.get(o);
      if (v) v.push(s);
      else raw.set(o, [s]);
    }
  }
  for (const [o, list] of raw) {
    list.sort((x, y) => x.a - y.a);
    const merged: Shared[] = [];
    for (const s of list) {
      const p = merged[merged.length - 1];
      if (p && p.sg === s.sg && Math.abs(s.a - p.b) < 0.5 && Math.abs(s.c - (p.c + p.sg * (s.a - p.a))) < 0.5) p.b = s.b;
      else merged.push({ ...s });
    }
    const kept: Shared[] = [];
    for (const s of merged) {
      const lo = Math.max(s.a, s.sg === 1 ? s.a + d0 - s.c : s.a + s.c - d1);
      const hi = Math.min(s.b, s.sg === 1 ? s.a + d1 - s.c : s.a + s.c - d0);
      if (hi - lo > 1) kept.push({ a: lo, b: hi, c: s.c + s.sg * (lo - s.a), sg: s.sg });
    }
    if (kept.length) out.set(o, kept);
  }
  return out;
}

/** The other lines' trains that cross `line`'s path at grade between offsets `d0` and `d1`. */
export function gradePasses(n: NetworkBuffers, line: number, d0: number, d1: number): Pass[] {
  const ix = indexOf(n), c = n.conflicts;
  if (!ix || !c || !c.length) return [];
  const seen = new Set<string>();
  const out: Pass[] = [];
  const add = (p: Pass) => {
    const key = `${p.o} ${p.run} ${Math.round(p.sO)} ${Math.round(p.sL)}`;
    if (p.sL <= d0 || p.sL >= d1 || seen.has(key)) return;
    seen.add(key);
    out.push(p);
  };
  const nx = c[0];
  for (let x = 0; x < nx; x++) {
    const k = 1 + 4 * x;
    for (const [own, atOwn, other, atOther] of [[c[k], c[k + 1], c[k + 2], c[k + 3]], [c[k + 2], c[k + 3], c[k], c[k + 1]]]) {
      const mine = ix.byEdge.get(own), theirs = ix.byEdge.get(other);
      if (!mine || !theirs) continue;
      for (let i = 0; i < mine.length; i += 2) {
        if (mine[i] !== line) continue;
        const sL = along(n, mine[i + 1], atOwn);
        for (let j = 0; j < theirs.length; j += 2) if (theirs[j] !== line) add({ o: theirs[j], run: -1, sO: along(n, theirs[j + 1], atOther), sL });
      }
    }
  }
  const k0 = 1 + 4 * nx, np = c[k0] ?? 0;
  for (let j = 0; j < np; j++) {
    const k = k0 + 1 + 5 * j;
    const m1 = c[k + 1] * MOVE_KEY + c[k + 2], m2 = c[k + 3] * MOVE_KEY + c[k + 4];
    for (const [a, b] of [[m1, m2], [m2, m1]]) {
      const mine = ix.byMove.get(a), theirs = ix.byMove.get(b);
      if (!mine || !theirs) continue;
      for (let i = 0; i < mine.length; i += 3) {
        if (mine[i] !== line) continue;
        for (let q = 0; q < theirs.length; q += 3) if (theirs[q] !== line) add({ o: theirs[q], run: theirs[q + 1], sO: theirs[q + 2], sL: mine[i + 2] });
      }
    }
  }
  return out;
}

/** The part of a (time, offset) polyline whose offset is between `lo` and `hi`, as SVG path data,
 * each point placed by `x` and `y`; straight between points. */
export function clipPath(pts: number[], lo: number, hi: number, x: (t: number) => number, y: (s: number) => number): string {
  let d = "", open = false;
  const pt = (cmd: string, t: number, s: number) => `${cmd}${x(t).toFixed(1)} ${y(s).toFixed(1)}`;
  for (let i = 0; i + 3 < pts.length; i += 2) {
    const t1 = pts[i], s1 = pts[i + 1], t2 = pts[i + 2], s2 = pts[i + 3];
    let u0 = 0, u1 = 1;
    if (s2 !== s1) {
      const ua = (lo - s1) / (s2 - s1), ub = (hi - s1) / (s2 - s1);
      u0 = Math.max(0, Math.min(ua, ub));
      u1 = Math.min(1, Math.max(ua, ub));
    } else if (s1 < lo || s1 > hi) u0 = 2;
    if (u0 > u1) { open = false; continue; }
    if (!open || u0 > 0) d += pt("M", t1 + (t2 - t1) * u0, s1 + (s2 - s1) * u0);
    d += pt("L", t1 + (t2 - t1) * u1, s1 + (s2 - s1) * u1);
    open = u1 >= 1;
  }
  return d;
}
