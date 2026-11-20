// The zone borders of the network diagram (T-2464; docs/ui-diagram.md "Zone borders"): the shapes, the zone each station
// is in, and the fitting of octilinear runs the rubber sheet's map uses (sheetLines.ts `sheetMap` draws the borders).

import type { DiagramGraph } from "./graph";
import { DIRS8, hyp } from "./fmath";

/** a zone's outline in graph metres: its rings, x, y pairs, open */
export interface ZoneShape { id: string; rings: Float64Array[] }

export interface DiagramBorders {
  /** the borders as path commands (draw.ts `toPath`), layout units */
  cmds: Float64Array;
  /** the zones drawn, and per graph node its zone among them (-1 for none) */
  zones: string[];
  zoneOf: Int32Array;
}

/** a station outside every outline is in the zone whose outline is nearest within this (cities.ts ZONE_REACH_M) */
const REACH_M = 40_000;

/** per graph node its shape's index: the outline holding it, else the nearest within REACH_M, else -1 */
export function stationZones(g: DiagramGraph, shapes: ZoneShape[]): Int32Array {
  const n = g.key.length;
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
        const dd = distRings(shapes[z].rings, px, py);
        if (dd < bd) (bd = dd), (best = z);
      }
    }
    if (zoneAll[i] < 0) zoneAll[i] = best;
  }
  return zoneAll;
}

/** A polyline through px, py fitted with runs at 0, 45 and 90 degrees, its ends kept: from each point, the farthest
 * point reachable by one run or two (one straight, one diagonal) passing within `tol` of every point between (and
 * that `keeps` allows); runs going on the same way joined. Vertices as x, y pairs. */
export function fitOcti(px: number[], py: number[], tol: number[], keeps?: (i: number, j: number, bx: number, by: number) => boolean, outer?: { tol: number[]; left: boolean }): Float64Array {
  const m = px.length, out: number[] = [px[0], py[0]];
  let dirX = 0, dirY = 0;
  /** the ways from point i to point j (one run, or a bend at bx, by), each that fits */
  const ways = (i: number, j: number): { bx: number; by: number; dev: number; fx: number; fy: number }[] => {
    const dx = px[j] - px[i], dy = py[j] - py[i], L = hyp(dx, dy);
    if (L < 1e-12) return [{ bx: NaN, by: NaN, dev: 0, fx: 0, fy: 0 }];
    // the two octilinear directions either side of (dx, dy), and its parts along them
    let k = 0, bestDot = -Infinity;
    for (let q = 0; q < 8; q++) {
      const dot = DIRS8[q][0] * dx + DIRS8[q][1] * dy;
      if (dot > bestDot) (bestDot = dot), (k = q);
    }
    const ax = DIRS8[k][0], ay = DIRS8[k][1], cross = ax * dy - ay * dx;
    const out: { bx: number; by: number; dev: number; fx: number; fy: number }[] = [];
    if (Math.abs(cross) < 1e-9 * L) {
      const dev = deviation(i, j, NaN, NaN);
      if (dev >= 0) out.push({ bx: NaN, by: NaN, dev, fx: ax, fy: ay });
      return out;
    }
    const k2 = (k + (cross > 0 ? 1 : 7)) % 8, cx = DIRS8[k2][0], cy = DIRS8[k2][1];
    // dx = a A + b C
    const det = ax * cy - ay * cx, a = (dx * cy - dy * cx) / det, b = (ax * dy - ay * dx) / det;
    for (const [bx, by, fx, fy] of [[px[i] + a * ax, py[i] + a * ay, ax, ay], [px[i] + b * cx, py[i] + b * cy, cx, cy]]) {
      const dev = deviation(i, j, bx, by);
      if (dev >= 0) out.push({ bx, by, dev, fx, fy });
    }
    return out;
  };
  /** the worst share of its tolerance any point between i and j strays from the path (and the bend from the
   * points), or -1 where one strays past it */
  const deviation = (i: number, j: number, bx: number, by: number): number => {
    const bent = bx === bx;
    let worst = 0, bendNear = Infinity, bendTol = 0;
    for (let k = i + 1; k < j; k++) {
      const e1 = segD(px[k], py[k], px[i], py[i], bent ? bx : px[j], bent ? by : py[j]), e2 = bent ? segD(px[k], py[k], bx, by, px[j], py[j]) : Infinity;
      let e = Math.min(e1, e2), t = tol[k];
      // on the region's edge the way may pass further out than in: it fills notches, never nears the lines (T-2491)
      // (or less far, where its outward tolerance is the smaller: the sea's coast along a strait, T-2540)
      if (outer && (e > t || outer.tol[k] < t)) {
        const [ax, ay, cx, cy] = e1 <= e2 ? [px[i], py[i], bent ? bx : px[j], bent ? by : py[j]] : [bx, by, px[j], py[j]];
        if (((cx - ax) * (py[k] - ay) - (cy - ay) * (px[k] - ax) < 0) === outer.left) t = outer.tol[k];
      }
      if (e > t) return -1;
      worst = Math.max(worst, e / t);
      if (bent) {
        e = hyp(px[k] - bx, py[k] - by);
        if (e < bendNear) (bendNear = e), (bendTol = outer ? (outer.tol[k] < tol[k] ? outer.tol[k] : Math.max(tol[k], outer.tol[k])) : tol[k]);
      }
    }
    if (bent && j > i + 1 && bendNear > bendTol + 1e-9) return -1;
    if (keeps && !keeps(i, j, bx, by)) return -1;
    return worst;
  };
  const pick = (i: number, j: number) => {
    const w = ways(i, j);
    if (!w.length) return null;
    // a way going on as the last run went, else the nearest
    const on = (p: { fx: number; fy: number }) => +(snap(p.fx) === dirX && snap(p.fy) === dirY);
    w.sort((p, q) => on(q) - on(p) || p.dev - q.dev);
    return w[0];
  };
  for (let i = 0; i < m - 1; ) {
    // lo fits, hi does not (m: past the end): doubling, then halving (the fits are mostly monotone)
    let lo = i + 1, hi = m, step = 1;
    while (lo + step < m) {
      if (!pick(i, lo + step)) {
        hi = lo + step;
        break;
      }
      (lo += step), (step *= 2);
    }
    if (hi === m && lo < m - 1) {
      if (pick(i, m - 1)) lo = m - 1;
      else hi = m - 1;
    }
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (pick(i, mid)) lo = mid;
      else hi = mid;
    }
    const w = pick(i, lo)!;
    if (w.bx === w.bx) out.push(w.bx, w.by);
    out.push(px[lo], py[lo]);
    const lx = px[lo] - out[out.length - 4], ly = py[lo] - out[out.length - 3], ll = hyp(lx, ly);
    if (ll > 0) (dirX = snap(lx / ll)), (dirY = snap(ly / ll));
    i = lo;
  }
  return Float64Array.from(joinStraight(out));
}

const snap = (v: number) => Math.round(v * 1e6) / 1e6;

/** vertices with those where the way goes straight on (or doubles back on itself by nothing) dropped */
function joinStraight(v: number[]): number[] {
  const out: number[] = [v[0], v[1]];
  for (let k = 2; k < v.length; k += 2) {
    const n = out.length;
    if (hyp(v[k] - out[n - 2], v[k + 1] - out[n - 1]) < 1e-9) continue;
    if (n >= 4) {
      const ax = out[n - 2] - out[n - 4], ay = out[n - 1] - out[n - 3], bx = v[k] - out[n - 2], by = v[k + 1] - out[n - 1];
      if (Math.abs(ax * by - ay * bx) < 1e-9 * hyp(ax, ay) * hyp(bx, by) && ax * bx + ay * by > 0) {
        (out[n - 2] = v[k]), (out[n - 1] = v[k + 1]);
        continue;
      }
    }
    out.push(v[k], v[k + 1]);
  }
  return out;
}

/** the polyline `v` as path commands, its corners rounded with radius `r` (less where its runs are short) */
export function rounded(v: Float64Array, closed: boolean, r: number, cmd: number[]): void {
  let m = v.length / 2;
  if (closed && m > 2 && hyp(v[0] - v[m * 2 - 2], v[1] - v[m * 2 - 1]) < 1e-9) m--;
  if (m < 2) return;
  const X = (k: number) => v[((k + m) % m) * 2], Y = (k: number) => v[((k + m) % m) * 2 + 1];
  /** where the curve round corner k starts and ends, or null for no corner */
  const corner = (k: number): [number, number, number, number] | null => {
    const ax = X(k) - X(k - 1), ay = Y(k) - Y(k - 1), bx = X(k + 1) - X(k), by = Y(k + 1) - Y(k);
    const la = hyp(ax, ay), lb = hyp(bx, by);
    if (la < 1e-12 || lb < 1e-12) return null;
    const sn = (ax * by - ay * bx) / (la * lb), cs = (ax * bx + ay * by) / (la * lb);
    if (Math.abs(sn) < 1e-9 && cs > 0) return null;
    const t = Math.min(r * (cs > -1 + 1e-12 ? Math.abs(sn) / (1 + cs) : 1e16), 0.49 * la, 0.49 * lb);
    return [X(k) - (ax / la) * t, Y(k) - (ay / la) * t, X(k) + (bx / lb) * t, Y(k) + (by / lb) * t];
  };
  if (!closed) {
    cmd.push(0, v[0], v[1]);
    for (let k = 1; k < m - 1; k++) {
      const c = corner(k);
      if (!c) continue;
      cmd.push(1, c[0], c[1], 2, X(k), Y(k), c[2], c[3]);
    }
    cmd.push(1, X(m - 1), Y(m - 1));
    return;
  }
  const c0 = corner(0);
  cmd.push(0, c0 ? c0[2] : X(0), c0 ? c0[3] : Y(0));
  for (let k = 1; k <= m; k++) {
    const c = corner(k);
    if (!c) cmd.push(1, X(k), Y(k));
    else cmd.push(1, c[0], c[1], 2, X(k), Y(k), c[2], c[3]);
  }
}

/** the polyline `v` as path commands, each corner a circular arc of radius up to `r` drawn as curves of at most 45
 * degrees each (T-3833: `rounded`'s single curve comes to a point at a 135-degree turn). Where two corners share a run
 * too short for both, the run is split between them in proportion to what each asks, so both get the same radius. */
export function filleted(v0: ArrayLike<number>, closed: boolean, r: number, cmd: number[]): void {
  // vertices where the way goes straight on dropped first, so a run is whole between its two corners
  const v = joinStraight(Array.from(v0));
  let m = v.length / 2;
  if (closed && m > 2 && hyp(v[0] - v[m * 2 - 2], v[1] - v[m * 2 - 1]) < 1e-9) m--;
  if (closed && m > 2) {
    // and the start, when the ring goes straight on through it
    const ax = v[0] - v[m * 2 - 2], ay = v[1] - v[m * 2 - 1], bx = v[2] - v[0], by = v[3] - v[1];
    if (Math.abs(ax * by - ay * bx) < 1e-9 * hyp(ax, ay) * hyp(bx, by) && ax * bx + ay * by > 0) (v.splice(0, 2), m--);
  }
  if (m < 2) return;
  const X = (k: number) => v[((k + m) % m) * 2], Y = (k: number) => v[((k + m) % m) * 2 + 1];
  // per vertex: the run before it, its turn (radians, signed) and the tangent length it asks for
  const len = new Float64Array(m), turn = new Float64Array(m), ask = new Float64Array(m);
  for (let k = 0; k < m; k++) len[k] = hyp(X(k) - X(k - 1), Y(k) - Y(k - 1));
  for (let k = 0; k < m; k++) {
    if (!closed && (k === 0 || k === m - 1)) continue;
    const la = len[k], lb = len[(k + 1) % m];
    if (la < 1e-12 || lb < 1e-12) continue;
    const ax = (X(k) - X(k - 1)) / la, ay = (Y(k) - Y(k - 1)) / la, bx = (X(k + 1) - X(k)) / lb, by = (Y(k + 1) - Y(k)) / lb;
    const th = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
    if (Math.abs(th) < 1e-9 || Math.abs(th) > Math.PI - 1e-6) continue;
    (turn[k] = th), (ask[k] = r * Math.tan(Math.abs(th) / 2));
  }
  const tan = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    if (!ask[k]) continue;
    const p = ask[(k - 1 + m) % m], q = ask[(k + 1) % m], la = len[k], lb = len[(k + 1) % m];
    tan[k] = Math.min(ask[k], (0.999 * la * ask[k]) / (ask[k] + (closed || k > 1 ? p : 0)), (0.999 * lb * ask[k]) / (ask[k] + (closed || k < m - 2 ? q : 0)));
  }
  const arc = (k: number) => {
    const t = tan[k], th = turn[k], la = len[k], lb = len[(k + 1) % m];
    const ax = (X(k) - X(k - 1)) / la, ay = (Y(k) - Y(k - 1)) / la, bx = (X(k + 1) - X(k)) / lb, by = (Y(k + 1) - Y(k)) / lb;
    const n = Math.max(1, Math.ceil(Math.abs(th) / (Math.PI / 4) - 1e-6)), d = th / n, rad = t / Math.tan(Math.abs(th) / 2), h = rad * Math.tan(Math.abs(d) / 2);
    let px = X(k) - ax * t, py = Y(k) - ay * t, dx = ax, dy = ay;
    if (n === 1) return void cmd.push(2, X(k), Y(k), X(k) + bx * t, Y(k) + by * t);
    for (let i = 1; i <= n; i++) {
      const c = Math.cos(d * i), s = Math.sin(d * i), ex = i === n ? bx : ax * c - ay * s, ey = i === n ? by : ax * s + ay * c;
      const qx = px + dx * h, qy = py + dy * h, nx = i === n ? X(k) + bx * t : qx + ex * h, ny = i === n ? Y(k) + by * t : qy + ey * h;
      cmd.push(2, qx, qy, nx, ny);
      (px = nx), (py = ny), (dx = ex), (dy = ey);
    }
  };
  if (!closed) {
    cmd.push(0, v[0], v[1]);
    for (let k = 1; k < m - 1; k++) {
      if (!tan[k]) continue;
      const la = len[k];
      cmd.push(1, X(k) - ((X(k) - X(k - 1)) / la) * tan[k], Y(k) - ((Y(k) - Y(k - 1)) / la) * tan[k]);
      arc(k);
    }
    cmd.push(1, X(m - 1), Y(m - 1));
    return;
  }
  const lb0 = len[1 % m];
  cmd.push(0, tan[0] ? X(0) + ((X(1) - X(0)) / lb0) * tan[0] : X(0), tan[0] ? Y(0) + ((Y(1) - Y(0)) / lb0) * tan[0] : Y(0));
  for (let k = 1; k <= m; k++) {
    const kk = k % m;
    if (!tan[kk]) { cmd.push(1, X(k), Y(k)); continue; }
    const la = len[kk];
    cmd.push(1, X(k) - ((X(k) - X(k - 1)) / la) * tan[kk], Y(k) - ((Y(k) - Y(k - 1)) / la) * tan[kk]);
    arc(kk);
  }
}

export function segD(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return hyp(px - ax - dx * t, py - ay - dy * t);
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
    for (let i = 0, j = m - 2; i < m; j = i, i += 2) best = Math.min(best, segD(x, y, r[j], r[j + 1], r[i], r[i + 1]));
  }
  return best;
}
