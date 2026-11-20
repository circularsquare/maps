// Where the water's separately drawn shapes meet (T-3833; docs/ui-diagram.md "Water"): a river's band crossing the
// sea's coast or another band leaves a corner of land between them, which no shape's own rounding reaches. Each such
// corner is filled with water along an arc (a fillet), drawn over the sea and the bands as its own patch that reaches a
// little into the water either side, so no seam shows. Pure: no DOM.

import { filleted } from "./borders";
import { hyp } from "./fmath";

/** a fillet's radius (pitches, the coast's bends being 3 and a band's ends half its width); how near a station's centre
 * a fillet may come; how far a patch reaches into the water beside it */
export const FILLET = { r: 1.5, clear: 0.75, overlap: 0.04 };

/** a coast's small features: a run shorter than `short` pitches is taken out (its neighbours run on to meet, or one
 * moves onto the other's line at a step, or a nib or notch under `short` deep is cut off at its base, as is one under
 * `thin` wide and `deep` deep), each move at most `move` pitches, never sweeping over a station's centre or bringing
 * the coast nearer one than `clear` (or than it already was); a ring left under `tiny` square pitches goes */
export const COAST = { short: 3, move: 3, thin: 1.2, deep: 8, clear: 1.2, tiny: 6 };

/** A closed coast ring fitted at 0, 45 and 90 degrees (x, y pairs, pitches) with its small features taken out
 * (`COAST`), so each corner left has room to round (T-3833: a run of a pitch between two turns rounds to a radius of
 * half a pitch, which reads as a sharp step). Stations' centres `sx, sy`, pitches. Still at 0, 45 and 90 degrees. */
export function cleanRing(v: ArrayLike<number>, sx: ArrayLike<number>, sy: ArrayLike<number>, o = COAST): Float64Array {
  const X: number[] = [], Y: number[] = [];
  for (let k = 0; k < v.length; k += 2) X.push(v[k]), Y.push(v[k + 1]);
  if (X.length > 1 && hyp(X[0] - X[X.length - 1], Y[0] - Y[Y.length - 1]) < 1e-9) X.pop(), Y.pop();
  // stations near the ring, by 4-pitch buckets
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let k = 0; k < X.length; k++) (x0 = Math.min(x0, X[k])), (x1 = Math.max(x1, X[k])), (y0 = Math.min(y0, Y[k])), (y1 = Math.max(y1, Y[k]));
  const BK = 4, bk = new Map<number, number[]>(), pad = o.move + o.clear + 1;
  for (let s = 0; s < sx.length; s++) {
    if (sx[s] < x0 - pad || sx[s] > x1 + pad || sy[s] < y0 - pad || sy[s] > y1 + pad) continue;
    const kk = Math.floor(sx[s] / BK) * 65536 + Math.floor(sy[s] / BK);
    let l = bk.get(kk);
    if (!l) bk.set(kk, (l = []));
    l.push(s);
  }
  const stationsIn = (qx0: number, qy0: number, qx1: number, qy1: number): number[] => {
    const out: number[] = [];
    if (!bk.size) return out;
    for (let i = Math.floor(qx0 / BK); i <= Math.floor(qx1 / BK); i++) for (let j = Math.floor(qy0 / BK); j <= Math.floor(qy1 / BK); j++) for (const s of bk.get(i * 65536 + j) ?? []) if (sx[s] >= qx0 && sx[s] <= qx1 && sy[s] >= qy0 && sy[s] <= qy1) out.push(s);
    return out;
  };
  const segDist = (x: number, y: number, ax: number, ay: number, bx: number, by: number) => {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    return hyp(x - ax - dx * t, y - ay - dy * t);
  };
  /** whether the coast may go from old points `a` to new points `b` (both from the same vertex before to the same
   * after): no station's centre in the area between, none nearer the new than `clear` unless the old was as near */
  const may = (a: number[], b: number[]): boolean => {
    let qx0 = Infinity, qy0 = Infinity, qx1 = -Infinity, qy1 = -Infinity;
    for (const p of [a, b]) for (let k = 0; k < p.length; k += 2) (qx0 = Math.min(qx0, p[k])), (qx1 = Math.max(qx1, p[k])), (qy0 = Math.min(qy0, p[k + 1])), (qy1 = Math.max(qy1, p[k + 1]));
    const poly = [...a];
    for (let k = b.length - 2; k >= 0; k -= 2) poly.push(b[k], b[k + 1]);
    for (const s of stationsIn(qx0 - o.clear, qy0 - o.clear, qx1 + o.clear, qy1 + o.clear)) {
      const x = sx[s], y = sy[s];
      let dNew = Infinity, dOld = Infinity, inside = false;
      for (let k = 0; k + 3 < b.length; k += 2) dNew = Math.min(dNew, segDist(x, y, b[k], b[k + 1], b[k + 2], b[k + 3]));
      for (let k = 0; k + 3 < a.length; k += 2) dOld = Math.min(dOld, segDist(x, y, a[k], a[k + 1], a[k + 2], a[k + 3]));
      if (dNew < Math.min(o.clear, dOld) - 1e-9) return false;
      for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) if (poly[i + 1] > y !== poly[j + 1] > y && x < ((poly[j] - poly[i]) * (y - poly[i + 1])) / (poly[j + 1] - poly[i + 1]) + poly[i]) inside = !inside;
      if (inside) return false;
    }
    return true;
  };
  /** where the line through (ax, ay) along (ux, uy) meets the one through (bx, by) along (vx, vy): the distances along each */
  const meet = (ax: number, ay: number, ux: number, uy: number, bx: number, by: number, vx: number, vy: number): [number, number] | null => {
    const den = ux * vy - uy * vx;
    if (Math.abs(den) < 1e-9) return null;
    return [((bx - ax) * vy - (by - ay) * vx) / den, ((bx - ax) * uy - (by - ay) * ux) / den];
  };
  /** vertices where the way goes straight on, doubles back (a spike), or repeats, dropped */
  const tidy = () => {
    for (let changed = true; changed && X.length > 3; ) {
      changed = false;
      for (let k = 0; k < X.length && X.length > 3; k++) {
        const m = X.length, a = (k - 1 + m) % m, b = (k + 1) % m;
        const ax = X[k] - X[a], ay = Y[k] - Y[a], bx = X[b] - X[k], by = Y[b] - Y[k], la = hyp(ax, ay), lb = hyp(bx, by);
        if (la < 1e-9 || lb < 1e-9 || Math.abs(ax * by - ay * bx) < 1e-9 * la * lb) (X.splice(k, 1), Y.splice(k, 1)), (changed = true), k--;
      }
    }
  };
  tidy();
  for (let pass = 0, changed = true; changed && pass < 8; pass++) {
    changed = false;
    for (let k = 0; k < X.length && X.length > 4; k++) {
      const m = X.length, i0 = (k - 1 + m) % m, i1 = k, i2 = (k + 1) % m, i3 = (k + 2) % m, im = (k - 2 + m) % m;
      // run i1 to i2 short; the runs before (i0 to i1) and after (i2 to i3)
      const L = hyp(X[i2] - X[i1], Y[i2] - Y[i1]);
      if (L >= o.short) continue;
      const la = hyp(X[i1] - X[i0], Y[i1] - Y[i0]), lb = hyp(X[i3] - X[i2], Y[i3] - Y[i2]);
      const ux = (X[i1] - X[i0]) / la, uy = (Y[i1] - Y[i0]) / la, vx = (X[i3] - X[i2]) / lb, vy = (Y[i3] - Y[i2]) / lb;
      const par = Math.abs(ux * vy - uy * vx) < 1e-6;
      let from = -1, n = 0, put: number[] = [];
      if (!par) {
        // the runs either side run on to meet
        const t = meet(X[i1], Y[i1], ux, uy, X[i2], Y[i2], vx, vy);
        if (!t || t[0] <= -la + 1e-6 || t[1] >= lb - 1e-6) continue;
        const px = X[i1] + ux * t[0], py = Y[i1] + uy * t[0];
        if (hyp(px - X[i1], py - Y[i1]) > o.move || hyp(px - X[i2], py - Y[i2]) > o.move) continue;
        (from = i0), (n = 2), (put = [px, py]);
      } else if (ux * vx + uy * vy > 0) {
        // a step: the shorter of the runs either side moved onto the other's line
        if (la <= lb) {
          const lp = hyp(X[i0] - X[im], Y[i0] - Y[im]), wx = (X[i0] - X[im]) / lp, wy = (Y[i0] - Y[im]) / lp, t = meet(X[im], Y[im], wx, wy, X[i2], Y[i2], vx, vy);
          if (!t || t[0] <= 1e-6 || t[1] >= lb - 1e-6) continue;
          const px = X[im] + wx * t[0], py = Y[im] + wy * t[0];
          if (hyp(px - X[i0], py - Y[i0]) > o.move) continue;
          (from = im), (n = 3), (put = [px, py]);
        } else {
          const i4 = (k + 3) % m, ln = hyp(X[i4] - X[i3], Y[i4] - Y[i3]), wx = (X[i4] - X[i3]) / ln, wy = (Y[i4] - Y[i3]) / ln, t = meet(X[i1], Y[i1], ux, uy, X[i3], Y[i3], wx, wy);
          if (!t || t[0] <= -la + 1e-6 || t[1] >= ln - 1e-6) continue;
          const px = X[i1] + ux * t[0], py = Y[i1] + uy * t[0];
          if (hyp(px - X[i3], py - Y[i3]) > o.move) continue;
          (from = i0), (n = 3), (put = [px, py]);
        }
      } else {
        // a nib or notch L wide: cut off at its base where under `short` deep on one side
        if (Math.min(la, lb) >= (L < o.thin ? o.deep : o.short)) continue;
        (from = i0), (n = 2), (put = la <= lb ? [X[i2] + vx * la, Y[i2] + vy * la] : [X[i1] - ux * lb, Y[i1] - uy * lb]);
      }
      // from vertex `from`, the next n vertices replaced by `put`; checked against the stations
      const old: number[] = [X[from], Y[from]], nw: number[] = [X[from], Y[from]];
      for (let q = 1; q <= n; q++) old.push(X[(from + q) % m], Y[(from + q) % m]);
      const after = (from + n + 1) % m;
      old.push(X[after], Y[after]);
      nw.push(...put, X[after], Y[after]);
      if (!may(old, nw)) continue;
      // nor running over the rest of its own ring
      let over = false;
      const span = new Set<number>([from, after]);
      for (let q = 1; q <= n; q++) span.add((from + q) % m);
      for (let k = 0; k < m && !over; k++) {
        const k2 = (k + 1) % m;
        if (span.has(k) || span.has(k2)) continue;
        for (let q = 0; q + 3 < nw.length && !over; q += 2) {
          const ax = nw[q], ay = nw[q + 1], bx = nw[q + 2], by = nw[q + 3], d = (px: number, py: number, qx: number, qy: number, rx: number, ry: number) => (qx - px) * (ry - py) - (qy - py) * (rx - px);
          over = d(ax, ay, bx, by, X[k], Y[k]) * d(ax, ay, bx, by, X[k2], Y[k2]) < 0 && d(X[k], Y[k], X[k2], Y[k2], ax, ay) * d(X[k], Y[k], X[k2], Y[k2], bx, by) < 0;
        }
      }
      if (over) continue;
      // replace, keeping the ring's start where it was
      const idx: number[] = [];
      for (let q = 1; q <= n; q++) idx.push((from + q) % m);
      const pos = Math.min(...idx), wrap = idx.some((q) => q < from);
      for (const q of [...idx].sort((p, r) => r - p)) X.splice(q, 1), Y.splice(q, 1);
      const at = wrap ? 0 : pos;
      for (let q = put.length - 2; q >= 0; q -= 2) X.splice(at, 0, put[q]), Y.splice(at, 0, put[q + 1]);
      tidy();
      changed = true;
      k = Math.max(-1, k - 3);
    }
  }
  // a ring left under `tiny` square pitches with no station near (an islet or a pond a corner or two across) goes
  let area = 0;
  for (let k = 0, j = X.length - 1; k < X.length; j = k++) area += X[j] * Y[k] - X[k] * Y[j];
  if (Math.abs(area) / 2 < o.tiny && !stationsIn(x0 - o.clear, y0 - o.clear, x1 + o.clear, y1 + o.clear).length) return new Float64Array(0);
  const out = new Float64Array(X.length * 2 + 2);
  for (let k = 0; k < X.length; k++) (out[2 * k] = X[k]), (out[2 * k + 1] = Y[k]);
  (out[X.length * 2] = X[0]), (out[X.length * 2 + 1] = Y[0]);
  return out;
}

/** path commands (0 move, 1 line, 2 quadratic) as subpaths of points, divided by `k`; each curve in `n` pieces */
function flatten(cmd: ArrayLike<number>, k: number, n = 6): number[][] {
  const out: number[][] = [];
  let cur: number[] = [];
  for (let i = 0; i < cmd.length; ) {
    if (cmd[i] === 0) (cur = [cmd[i + 1] / k, cmd[i + 2] / k]), out.push(cur), (i += 3);
    else if (cmd[i] === 1) cur.push(cmd[i + 1] / k, cmd[i + 2] / k), (i += 3);
    else {
      const ax = cur[cur.length - 2], ay = cur[cur.length - 1], cx = cmd[i + 1] / k, cy = cmd[i + 2] / k, bx = cmd[i + 3] / k, by = cmd[i + 4] / k;
      for (let t = 1; t <= n; t++) { const u = t / n; cur.push((1 - u) ** 2 * ax + 2 * u * (1 - u) * cx + u * u * bx, (1 - u) ** 2 * ay + 2 * u * (1 - u) * cy + u * u * by); }
      i += 5;
    }
  }
  return out;
}

/** an edge of the water: from (ax, ay) to (bx, by), `nx, ny` its normal toward land; its shape (a sea ring, or a band's
 * side or end) and its neighbours along that shape's outline (-1 none) */
interface Edge { ax: number; ay: number; bx: number; by: number; nx: number; ny: number; shape: number; prev: number; next: number }

/** The fillets where bands meet the sea or each other, as path commands (layout units, filled nonzero): `sea` the coast
 * (water on each ring's left), `rivers` the bands' middles per width `bands`, `ends` their round ends, all layout units,
 * `K` layout units a pitch; stations' centres `sx, sy` in pitches. */
export function bandFillets(sea: ArrayLike<number>, rivers: ArrayLike<number>[], bands: number[], ends: ArrayLike<number>[], K: number, sx: ArrayLike<number>, sy: ArrayLike<number>, o = FILLET): { cmds: number[]; made: number; tight: number; tightAt: number[] } {
  const E: Edge[] = [];
  const add = (ax: number, ay: number, bx: number, by: number, nx: number, ny: number, shape: number) => {
    const l = hyp(bx - ax, by - ay);
    if (l < 1e-9) return;
    E.push({ ax, ay, bx, by, nx, ny, shape, prev: -1, next: -1 });
  };
  /** an outline's edges added in order, each linked to the next */
  const chain = (pts: number[], closed: boolean, shape: number, left: boolean) => {
    const e0 = E.length, m = pts.length / 2;
    for (let q = 0; q + 1 < m || (closed && q < m); q++) {
      const a = q, b = (q + 1) % m, ax = pts[2 * a], ay = pts[2 * a + 1], bx = pts[2 * b], by = pts[2 * b + 1], l = hyp(bx - ax, by - ay);
      if (l < 1e-9) continue;
      // water on the left: land on the right
      add(ax, ay, bx, by, left ? (by - ay) / l : -(by - ay) / l, left ? -(bx - ax) / l : (bx - ax) / l, shape);
    }
    for (let e = e0; e + 1 < E.length; e++) (E[e].next = e + 1), (E[e + 1].prev = e);
    if (closed && E.length - e0 > 1) (E[E.length - 1].next = e0), (E[e0].prev = E.length - 1);
  };
  let shape = 0;
  for (const ring of flatten(sea, K)) chain(ring, true, shape++, true);
  const firstBand = shape;
  // each band's sides (a run of one width, its middle flattened) and its round ends as rings
  const caps: [number, number, number][] = [];
  rivers.forEach((cmd, wi) => {
    const hw = bands[wi] / K / 2;
    for (const p of flatten(cmd, K)) {
      const m = p.length / 2;
      if (m < 2) continue;
      const L: number[] = [], R: number[] = [];
      for (let q = 0; q < m; q++) {
        // the side's offset at a vertex: the mean of its segments' normals, scaled to keep the side at hw
        const a = Math.max(0, q - 1), b = Math.min(m - 1, q + 1);
        let tx = p[2 * b] - p[2 * a], ty = p[2 * b + 1] - p[2 * a + 1];
        const tl = hyp(tx, ty) || 1;
        (tx /= tl), (ty /= tl);
        let s = 1;
        if (q > 0 && q < m - 1) {
          const ux = p[2 * q] - p[2 * q - 2], uy = p[2 * q + 1] - p[2 * q - 1], ul = hyp(ux, uy) || 1;
          s = 1 / Math.max(0.5, (ux * tx + uy * ty) / ul);
        }
        L.push(p[2 * q] - ty * hw * s, p[2 * q + 1] + tx * hw * s), R.push(p[2 * q] + ty * hw * s, p[2 * q + 1] - tx * hw * s);
      }
      // the left side runs with the band on its right, the right side reversed likewise
      chain(L, false, shape, false);
      const Rr: number[] = [];
      for (let q = m - 1; q >= 0; q--) Rr.push(R[2 * q], R[2 * q + 1]);
      chain(Rr, false, shape, false);
      shape++;
    }
    const e = ends[wi];
    if (e) for (const s of flatten(e, K)) caps.push([s[0], s[1], hw]);
  });
  const capOf = new Map<number, [number, number, number]>();
  for (const c of caps) {
    const [cx, cy, r] = c, ring: number[] = [];
    for (let q = 0; q < 24; q++) ring.push(cx + r * Math.cos((q * Math.PI) / 12), cy + r * Math.sin((q * Math.PI) / 12));
    capOf.set(shape, c);
    chain(ring, true, shape++, true);
  }
  // edges by 2-pitch buckets
  const BK = 2, buckets = new Map<number, number[]>(), key = (i: number, j: number) => i * 1048576 + j;
  E.forEach((e, k) => {
    for (let j = Math.floor(Math.min(e.ay, e.by) / BK); j <= Math.floor(Math.max(e.ay, e.by) / BK); j++) for (let i = Math.floor(Math.min(e.ax, e.bx) / BK); i <= Math.floor(Math.max(e.ax, e.bx) / BK); i++) {
      let l = buckets.get(key(i, j));
      if (!l) buckets.set(key(i, j), (l = []));
      l.push(k);
    }
  });
  const near = (x0: number, y0: number, x1: number, y1: number, fn: (k: number) => void) => {
    const seen = new Set<number>();
    for (let j = Math.floor(y0 / BK); j <= Math.floor(y1 / BK); j++) for (let i = Math.floor(x0 / BK); i <= Math.floor(x1 / BK); i++) for (const k of buckets.get(key(i, j)) ?? []) if (!seen.has(k)) seen.add(k), fn(k);
  };
  // whether a point is water: inside a sea ring (even-odd, by its own buckets' rows), a band or an end
  const seaE = E.filter((e) => e.shape < firstBand);
  const inSea = (x: number, y: number) => {
    let c = false;
    for (const e of seaE) if (e.ay > y !== e.by > y && x < ((e.bx - e.ax) * (y - e.ay)) / (e.by - e.ay) + e.ax) c = !c;
    return c;
  };
  const bandSegs: [number, number, number, number, number][] = [];
  rivers.forEach((cmd, wi) => { const hw = bands[wi] / K / 2; for (const p of flatten(cmd, K)) for (let q = 0; q + 3 < p.length; q += 2) bandSegs.push([p[q], p[q + 1], p[q + 2], p[q + 3], hw]); });
  const segDist = (x: number, y: number, ax: number, ay: number, bx: number, by: number) => {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
    return hyp(x - ax - dx * t, y - ay - dy * t);
  };
  const inBand = (x: number, y: number) => bandSegs.some(([ax, ay, bx, by, hw]) => segDist(x, y, ax, ay, bx, by) < hw) || caps.some(([cx, cy, r]) => hyp(x - cx, y - cy) < r);
  // sea rings are many and long: a point's sea test only when a band is near, so that test is rare
  const wet = (x: number, y: number) => inBand(x, y) || inSea(x, y);
  /** how far an outline runs straight on from edge k's point (x, y) along (ux, uy), over its next edges in line */
  const straight = (k: number, x: number, y: number, ux: number, uy: number) => {
    let e = E[k], fwd = (e.bx - e.ax) * ux + (e.by - e.ay) * uy > 0, L = fwd ? hyp(e.bx - x, e.by - y) : hyp(e.ax - x, e.ay - y);
    for (let s = 0; s < 64; s++) {
      const nk = fwd ? e.next : e.prev;
      if (nk < 0) break;
      const f = E[nk], fx = f.bx - f.ax, fy = f.by - f.ay, fl = hyp(fx, fy), dot = ((fwd ? fx : -fx) * ux + (fwd ? fy : -fy) * uy) / fl;
      if (dot < 0.9998) break;
      (L += fl), (e = f);
    }
    return L;
  };
  const cmds: number[] = [], done: number[] = [], tightAt: number[] = [];
  let made = 0, tight = 0;
  for (let a = 0; a < E.length; a++) {
    const A = E[a];
    if (A.shape < firstBand) continue;
    near(Math.min(A.ax, A.bx), Math.min(A.ay, A.by), Math.max(A.ax, A.bx), Math.max(A.ay, A.by), (b) => {
      const B = E[b];
      // a band's own sides meet too, inside a bend tighter than half its width
      if (b === a || b === A.prev || b === A.next || (B.shape >= firstBand && b < a)) return;
      // where the two edges cross
      const rx = A.bx - A.ax, ry = A.by - A.ay, sx2 = B.bx - B.ax, sy2 = B.by - B.ay, den = rx * sy2 - ry * sx2;
      if (Math.abs(den) < 1e-12) return;
      const t = ((B.ax - A.ax) * sy2 - (B.ay - A.ay) * sx2) / den, u = ((B.ax - A.ax) * ry - (B.ay - A.ay) * rx) / den;
      if (t < 0 || t > 1 || u < 0 || u > 1) return;
      const X = A.ax + rx * t, Y = A.ay + ry * t;
      for (let q = 0; q < done.length; q += 2) if (hyp(done[q] - X, done[q + 1] - Y) < 0.05) return;
      // the corner of land: along each edge toward the other's land side
      const la = hyp(rx, ry), lb = hyp(sx2, sy2);
      let ux = rx / la, uy = ry / la, vx = sx2 / lb, vy = sy2 / lb;
      if (ux * B.nx + uy * B.ny < 0) (ux = -ux), (uy = -uy);
      if (vx * A.nx + vy * A.ny < 0) (vx = -vx), (vy = -vy);
      const cos = ux * vx + uy * vy;
      if (cos > 0.999 || cos < -0.999) return;
      const half = Math.acos(cos) / 2, bx = ux + vx, by = uy + vy, bl = hyp(bx, by);
      // the corner must be land, both its sides the two edges' water
      const d = 0.03;
      if (wet(X + (bx / bl) * d, Y + (by / bl) * d)) return;
      if (!wet(X + ux * d * 2 - A.nx * d, Y + uy * d * 2 - A.ny * d) || !wet(X + vx * d * 2 - B.nx * d, Y + vy * d * 2 - B.ny * d)) return;
      done.push(X, Y);
      /** whether a station's centre is within `clear` of the patch, which lies in the triangle X, (ax, ay), (cx, cy) */
      const hits = (ax: number, ay: number, cx: number, cy: number) => {
        const R0 = Math.max(hyp(ax - X, ay - Y), hyp(cx - X, cy - Y)) + o.clear;
        for (let s = 0; s < sx.length; s++) {
          const px = sx[s], py = sy[s];
          if (Math.abs(px - X) > R0 || Math.abs(py - Y) > R0) continue;
          const s1 = (ax - X) * (py - Y) - (ay - Y) * (px - X), s2 = (cx - ax) * (py - ay) - (cy - ay) * (px - ax), s3 = (X - cx) * (py - cy) - (Y - cy) * (px - cx);
          if ((s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0) || segDist(px, py, ax, ay, cx, cy) < o.clear) return true;
        }
        return false;
      };
      const e = o.overlap;
      // a band's round end against a straight side: the arc touches the side and the end's circle (T-3833)
      const capA = capOf.get(A.shape), capB = capOf.get(B.shape);
      if (capA || capB) {
        if (capA && capB) return void (tight++, tightAt.push(X, Y));
        const [cx, cy, R] = (capA ?? capB)!, k = capA ? b : a, lx = capA ? vx : ux, ly = capA ? vy : uy, nx = E[k].nx, ny = E[k].ny;
        // a band's own sides touch its end's circle, no corner there
        if (Math.abs(Math.abs((cx - X) * nx + (cy - Y) * ny) - R) < 0.02) return;
        const room = 0.95 * straight(k, X, Y, lx, ly);
        for (let rho = o.r; rho > 0.05; rho *= 0.7) {
          const Dx = X + nx * rho - cx, Dy = Y + ny * rho - cy, vd = lx * Dx + ly * Dy, disc = vd * vd - (Dx * Dx + Dy * Dy) + (R + rho) ** 2;
          if (disc < 0) continue;
          const s = -vd + Math.sqrt(disc);
          if (s <= 0 || s > room) continue;
          const Fx = X + lx * s + nx * rho, Fy = Y + ly * s + ny * rho, Lx = X + lx * s, Ly = Y + ly * s, Cx = cx + ((Fx - cx) * R) / (R + rho), Cy = cy + ((Fy - cy) * R) / (R + rho);
          if (hits(Lx, Ly, Cx, Cy)) continue;
          // wound counter-clockwise: X, along the side, around the arc, back by the circle
          const ccw = (Lx - X) * (Cy - Y) - (Ly - Y) * (Cx - X) > 0, cnx = (X - cx) / R, cny = (Y - cy) / R, onx = (Cx - cx) / R, ony = (Cy - cy) / R;
          const x0 = X - (nx + cnx) * e, y0 = Y - (ny + cny) * e;
          const ends: [number, number, number, number][] = [[Lx, Ly, Lx - nx * e, Ly - ny * e], [Cx, Cy, Cx - onx * e, Cy - ony * e]];
          if (!ccw) ends.reverse();
          cmds.push(0, x0 * K, y0 * K, 1, ends[0][2] * K, ends[0][3] * K, 1, ends[0][0] * K, ends[0][1] * K);
          const t0 = Math.atan2(ends[0][1] - Fy, ends[0][0] - Fx);
          let dt = Math.atan2(ends[1][1] - Fy, ends[1][0] - Fx) - t0;
          if (dt > Math.PI) dt -= 2 * Math.PI;
          if (dt < -Math.PI) dt += 2 * Math.PI;
          const n = Math.max(1, Math.ceil(Math.abs(dt) / (Math.PI / 4))), hc = rho / Math.cos(dt / n / 2);
          for (let i = 1; i <= n; i++) {
            const mid = t0 + (dt * (i - 0.5)) / n, end = t0 + (dt * i) / n;
            cmds.push(2, (Fx + hc * Math.cos(mid)) * K, (Fy + hc * Math.sin(mid)) * K, i === n ? ends[1][0] * K : (Fx + rho * Math.cos(end)) * K, i === n ? ends[1][1] * K : (Fy + rho * Math.sin(end)) * K);
          }
          cmds.push(1, ends[1][2] * K, ends[1][3] * K, 1, x0 * K, y0 * K);
          made++;
          return;
        }
        return void (tight++, tightAt.push(X, Y));
      }
      let T = Math.min(o.r / Math.tan(half), 0.95 * straight(a, X, Y, ux, uy), 0.95 * straight(b, X, Y, vx, vy));
      // shrunk while a station's centre is within `clear` of it
      while (T > 0.05 && hits(X + ux * T, Y + uy * T, X + vx * T, Y + vy * T)) T *= 0.7;
      if (T <= 0.05) return void (tight++, tightAt.push(X, Y));
      // each patch wound counter-clockwise (filled nonzero, so overlapping patches stay filled): from the side whose
      // run comes first that way
      const ccw = ux * vy - uy * vx > 0;
      const [px, py, pnx, pny, qx, qy, qnx, qny] = ccw ? [ux, uy, A.nx, A.ny, vx, vy, B.nx, B.ny] : [vx, vy, B.nx, B.ny, ux, uy, A.nx, A.ny];
      const Px = X + px * T, Py = Y + py * T, Qx = X + qx * T, Qy = Y + qy * T, arc: number[] = [];
      filleted([Px, Py, X, Y, Qx, Qy], false, T * Math.tan(half), arc);
      // X pushed into the water both ways, along the first side, the arc, back along the other
      const x0 = X - (pnx + qnx) * e, y0 = Y - (pny + qny) * e;
      cmds.push(0, x0 * K, y0 * K, 1, (Px - pnx * e) * K, (Py - pny * e) * K);
      for (let q = 0; q < arc.length; ) {
        const op = arc[q], n = op === 2 ? 4 : 2;
        cmds.push(op === 0 ? 1 : op);
        for (let z = 1; z <= n; z++) cmds.push(arc[q + z] * K);
        q += n + 1;
      }
      cmds.push(1, (Qx - qnx * e) * K, (Qy - qny * e) * K, 1, x0 * K, y0 * K);
      made++;
    });
  }
  return { cmds, made, tight, tightAt };
}
