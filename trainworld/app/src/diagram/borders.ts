// The network diagram's faint zone borders (T-2168; docs/ui-diagram.md "Zone borders", notes/T-2168.md): the
// coarse outline of each zone holding a station (`packs/outlines.json`, the map's own), warped into the drawing
// (warp.ts) and drawn under the lines, fading out away from the stations so a border shows where things are built.
// A border two zones share is one arc, warped and drawn once.

import type { DiagramGraph } from "./graph";
import type { Blocks } from "./blocks";
import { makeWarp } from "./warp";
import { hyp } from "./fmath";

/** a zone's outline in graph metres: its rings, x, y pairs, open */
export interface ZoneShape { id: string; rings: Float64Array[] }

export interface DiagramBorders {
  /** the borders' polylines (layout units, x, y pairs, NaN between lines) by how strongly they are drawn: level k
   * at (k + 1) / LEVELS of the borders' opacity */
  levels: Float64Array[];
  /** the zones drawn, and per graph node its zone among them (-1 for none) */
  zones: string[];
  zoneOf: Int32Array;
  /** checks only (`zoneBorders(..., true)`): per zone drawn its whole rings as drawn, and each arc once */
  rings?: Float64Array[][];
  arcs?: Arc[];
}

/** a run of border between two zones (or a zone and none, -1): drawn x, y and opacity per point, the zones on its
 * left and right as it runs */
export interface Arc { p: Float64Array; left: number; right: number }

/** a station outside every outline is in the zone whose outline is nearest within this (cities.ts ZONE_REACH_M) */
const REACH_M = 40_000;
/** a border is drawn full within this many pitches of a drawn station, and fades out by FAR_PITCHES */
const NEAR_PITCHES = 6, FAR_PITCHES = 16;
export const LEVELS = 4;
/** and in reality within this many of the nearest station's median hops (at least NEAR_M), fading out by twice that */
const NEAR_HOPS = 1, NEAR_M = 1500;
/** each station's map in the warp reaches at most this share of FAR_PITCHES */
const REACH = 1.25;
/** densified so each piece is at most this many pitches drawn and STEP_M metres real */
const STEP_PITCHES = 0.75, STEP_M = 500;
/** simplified within this many pitches, then rounded off by corner cutting */
const SIMPLIFY_PITCHES = 0.6, ROUNDS = 2;

/** The borders of the zones holding stations, warped into the layout `x`, `y` (`pitch` layout units). */
export function zoneBorders(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, blocks: Blocks, pitch: number, shapes: ZoneShape[], detail = false): DiagramBorders {
  const n = g.key.length;
  // each station's zone: the outline holding it, else the nearest within reach
  const box = shapes.map((s) => {
    const b = [Infinity, Infinity, -Infinity, -Infinity];
    for (const r of s.rings) for (let k = 0; k < r.length; k += 2) (b[0] = Math.min(b[0], r[k])), (b[1] = Math.min(b[1], r[k + 1])), (b[2] = Math.max(b[2], r[k])), (b[3] = Math.max(b[3], r[k + 1]));
    return b;
  });
  const zoneAll = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const px = g.x[i], py = g.y[i];
    let best = -1, bd = REACH_M;
    for (let z = 0; z < shapes.length && zoneAll[i] < 0; z++) {
      const b = box[z];
      if (px < b[0] - bd || px > b[2] + bd || py < b[1] - bd || py > b[3] + bd) continue;
      if (px >= b[0] && px <= b[2] && py >= b[1] && py <= b[3] && inRings(shapes[z].rings, px, py)) zoneAll[i] = z;
      else {
        const d = distRings(shapes[z].rings, px, py);
        if (d < bd) (bd = d), (best = z);
      }
    }
    if (zoneAll[i] < 0) zoneAll[i] = best;
  }
  const drawnOf = new Int32Array(shapes.length).fill(-1), zones: string[] = [];
  for (let i = 0; i < n; i++) if (zoneAll[i] >= 0 && drawnOf[zoneAll[i]] < 0) (drawnOf[zoneAll[i]] = zones.length), zones.push(shapes[zoneAll[i]].id);
  const zoneOf = Int32Array.from(zoneAll, (z) => (z < 0 ? -1 : drawnOf[z]));
  const levels: number[][] = Array.from({ length: LEVELS }, () => []);
  if (!zones.length) return { levels: levels.map((l) => Float64Array.from(l)), zones, zoneOf };

  // the drawn zones' vertices, by real place, and their edges once
  const vid = new Map<string, number>(), vx: number[] = [], vy: number[] = [];
  const idOf = (px: number, py: number) => {
    const k = `${Math.round(px)},${Math.round(py)}`;
    let v = vid.get(k);
    if (v === undefined) vid.set(k, (v = vx.length)), vx.push(px), vy.push(py);
    return v;
  };
  const ringsV: { zone: number; v: number[]; ccw: boolean }[] = [];
  shapes.forEach((s, z) => {
    if (drawnOf[z] < 0) return;
    for (const r of s.rings) {
      const v: number[] = [];
      for (let k = 0; k < r.length; k += 2) {
        const id = idOf(r[k], r[k + 1]);
        if (v[v.length - 1] !== id) v.push(id);
      }
      while (v.length > 1 && v[0] === v[v.length - 1]) v.pop();
      let area = 0;
      for (let k = 0; k < v.length; k++) {
        const a = v[k], b = v[(k + 1) % v.length];
        area += vx[a] * vy[b] - vx[b] * vy[a];
      }
      if (v.length >= 3) ringsV.push({ zone: drawnOf[z], v, ccw: area > 0 });
    }
  });
  const nbrs: Set<number>[] = vx.map(() => new Set());
  for (const { v } of ringsV) for (let k = 0; k < v.length; k++) {
    const a = v[k], b = v[(k + 1) % v.length];
    nbrs[a].add(b), nbrs[b].add(a);
  }
  // each station's median real hop: how far from it a border still counts as near in reality
  const hops: number[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < g.ea.length; e++) {
    const d = hyp(g.x[g.eb[e]] - g.x[g.ea[e]], g.y[g.eb[e]] - g.y[g.ea[e]]);
    hops[g.ea[e]].push(d), hops[g.eb[e]].push(d);
  }
  const hop = Float64Array.from(hops, (h) => (h.sort((p, q) => p - q), h.length ? h[h.length >> 1] : 1000));
  const near = NEAR_PITCHES * pitch, far = FAR_PITCHES * pitch;
  /** a border's opacity at a warped place: full near a station both as drawn and in reality */
  const alpha = (o: Float64Array) => {
    const d = o[2], a = d <= near ? 1 : d >= far ? 0 : (far - d) / (far - near);
    const r1 = Math.max(NEAR_M, NEAR_HOPS * hop[o[4]]), r = o[3], b = r <= r1 ? 1 : r >= 2 * r1 ? 0 : (2 * r1 - r) / r1;
    return Math.min(a, b);
  };
  // borders show by the blocks' stations only (all of them when there are no blocks): between blocks the drawing
  // moves stations too far from where they lie for a border to keep its sides
  const seen = blocks.list.length ? Uint8Array.from(blocks.of, (b) => (b >= 0 ? 1 : 0)) : undefined;
  const warp = makeWarp(g, x, y, REACH * far, seen), o = new Float64Array(5);
  // per vertex: drawn x, y and opacity
  const wv = new Float64Array(vx.length * 3).fill(NaN);
  const warped = (v: number) => {
    if (wv[v * 3] !== wv[v * 3]) {
      warp.at(vx[v], vy[v], o);
      (wv[v * 3] = o[0]), (wv[v * 3 + 1] = o[1]), (wv[v * 3 + 2] = alpha(o));
    }
    return v * 3;
  };
  // arcs: runs between vertices where borders meet (not two neighbours), each kept once in one direction, with the
  // zones on its left and right
  const arcs = new Map<string, Arc>();
  const zoneArcs: { arc: Arc; back: boolean }[][][] = zones.map(() => []);
  for (const { zone, v, ccw } of ringsV) {
    const len = v.length, node = v.map((id) => nbrs[id].size !== 2);
    const start = node.indexOf(true);
    const runs: number[][] = [];
    if (start < 0) {
      // a ring meeting no other: one closed run from its lowest vertex (kept toward its lower neighbour, below)
      let lo = 0;
      for (let k = 1; k < len; k++) if (v[k] < v[lo]) lo = k;
      const run: number[] = [];
      for (let k = 0; k <= len; k++) run.push(v[(lo + k) % len]);
      runs.push(run);
    } else {
      for (let k = 0; k < len; ) {
        const run = [v[(start + k) % len]];
        do run.push(v[(start + ++k) % len]);
        while (!node[(start + k) % len]);
        runs.push(run);
      }
    }
    const parts: { arc: Arc; back: boolean }[] = [];
    for (const run of runs) {
      const m = run.length;
      const back = run[0] > run[m - 1] || (run[0] === run[m - 1] && m > 2 && run[1] > run[m - 2]);
      const c = back ? run.slice().reverse() : run, key = c.join(",");
      let arc = arcs.get(key);
      if (!arc) arcs.set(key, (arc = { p: arcOf(c), left: -1, right: -1 }));
      // the zone lies left of a counterclockwise ring
      if (ccw !== back) arc.left = zone;
      else arc.right = zone;
      parts.push({ arc, back });
    }
    zoneArcs[zone].push(parts);
  }

  /** a run of vertices densified, warped, simplified and rounded: x, y, opacity per point */
  function arcOf(run: number[]): Float64Array {
    const pts: number[] = [];
    for (let k = 0; k + 1 < run.length; k++) {
      const a = run[k], b = run[k + 1], wa = warped(a), wb = warped(b);
      if (!k) pts.push(wv[wa], wv[wa + 1], wv[wa + 2]);
      const real = hyp(vx[b] - vx[a], vy[b] - vy[a]), drawn = hyp(wv[wb] - wv[wa], wv[wb + 1] - wv[wa + 1]);
      const parts = Math.min(1000, Math.max(1, Math.ceil(real / STEP_M), Math.ceil(drawn / (STEP_PITCHES * pitch))));
      for (let q = 1; q < parts; q++) {
        const t = q / parts;
        warp.at(vx[a] + (vx[b] - vx[a]) * t, vy[a] + (vy[b] - vy[a]) * t, o);
        pts.push(o[0], o[1], alpha(o));
      }
      pts.push(wv[wb], wv[wb + 1], wv[wb + 2]);
    }
    return round(simplify(pts, SIMPLIFY_PITCHES * pitch), ROUNDS);
  }

  // the levels: each arc's pieces by the opacity at their middles
  for (const { p: arc } of arcs.values()) {
    let cur = -1;
    for (let k = 0; k + 3 < arc.length; k += 3) {
      const lv = Math.round(((arc[k + 2] + arc[k + 5]) / 2) * LEVELS) - 1;
      if (lv !== cur && cur >= 0) levels[cur].push(NaN, NaN);
      if (lv >= 0 && lv !== cur) levels[lv].push(arc[k], arc[k + 1]);
      if (lv >= 0) levels[lv].push(arc[k + 3], arc[k + 4]);
      cur = lv;
    }
    if (cur >= 0) levels[cur].push(NaN, NaN);
  }
  const out: DiagramBorders = { levels: levels.map((l) => Float64Array.from(l)), zones, zoneOf };
  if (detail) {
    out.arcs = [...arcs.values()];
    out.rings = zoneArcs.map((rs) => rs.map((parts) => {
      const r: number[] = [];
      for (const { arc, back } of parts) {
        const m = arc.p.length / 3;
        for (let q = 0; q < m - 1; q++) {
          const k = (back ? m - 1 - q : q) * 3;
          r.push(arc.p[k], arc.p[k + 1]);
        }
      }
      return Float64Array.from(r);
    }));
  }
  return out;
}

/** Douglas-Peucker on x, y, d triples within `tol`, ends kept */
function simplify(p: number[], tol: number): number[] {
  const m = p.length / 3;
  if (m <= 2) return p;
  const keep = new Uint8Array(m);
  keep[0] = keep[m - 1] = 1;
  const stack = [0, m - 1];
  while (stack.length) {
    const j = stack.pop()!, i = stack.pop()!;
    const ax = p[i * 3], ay = p[i * 3 + 1], dx = p[j * 3] - ax, dy = p[j * 3 + 1] - ay, l2 = dx * dx + dy * dy;
    let best = -1, bd = tol * tol;
    for (let k = i + 1; k < j; k++) {
      const qx = p[k * 3] - ax, qy = p[k * 3 + 1] - ay, t = l2 > 0 ? Math.max(0, Math.min(1, (qx * dx + qy * dy) / l2)) : 0;
      const ex = qx - t * dx, ey = qy - t * dy, e = ex * ex + ey * ey;
      if (e > bd) (bd = e), (best = k);
    }
    if (best >= 0) (keep[best] = 1), stack.push(i, best, best, j);
  }
  const out: number[] = [];
  for (let k = 0; k < m; k++) if (keep[k]) out.push(p[k * 3], p[k * 3 + 1], p[k * 3 + 2]);
  return out;
}

/** corners cut `rounds` times (Chaikin), ends kept, as a Float64Array of x, y, d */
function round(p: number[], rounds: number): Float64Array {
  let cur = p;
  for (let r = 0; r < rounds; r++) {
    const m = cur.length / 3;
    if (m < 3) break;
    const nx: number[] = [cur[0], cur[1], cur[2]];
    for (let k = 0; k + 1 < m; k++) {
      const a = k * 3, b = a + 3;
      for (const t of [0.25, 0.75]) {
        if ((k === 0 && t === 0.25) || (k === m - 2 && t === 0.75)) continue;
        nx.push(cur[a] + (cur[b] - cur[a]) * t, cur[a + 1] + (cur[b + 1] - cur[a + 1]) * t, cur[a + 2] + (cur[b + 2] - cur[a + 2]) * t);
      }
    }
    nx.push(cur[(m - 1) * 3], cur[(m - 1) * 3 + 1], cur[(m - 1) * 3 + 2]);
    cur = nx;
  }
  return Float64Array.from(cur);
}

/** whether (x, y) is inside rings (even-odd) */
export function inRings(rings: ArrayLike<number>[], x: number, y: number): boolean {
  let inside = false;
  for (const r of rings) {
    const m = r.length;
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) {
      const yi = r[i + 1], yj = r[j + 1];
      if (yi > y !== yj > y && x < ((r[j] - r[i]) * (y - yi)) / (yj - yi) + r[i]) inside = !inside;
    }
  }
  return inside;
}

/** the distance from (x, y) to the rings' edges */
function distRings(rings: Float64Array[], x: number, y: number): number {
  let best = Infinity;
  for (const r of rings) {
    const m = r.length;
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) {
      const ax = r[j], ay = r[j + 1], dx = r[i] - ax, dy = r[i + 1] - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      best = Math.min(best, hyp(ax + t * dx - x, ay + t * dy - y));
    }
  }
  return best;
}
