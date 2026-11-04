// The network diagram's warp (T-2168; notes/T-2168.md): any real place (graph metres) to a drawn one, so shapes
// from the map (zone borders now, water later, T-2169) can be drawn on the diagram. One function for the whole
// drawing, so a border two zones share is warped once, and a line through blocks and the gaps between them stays
// one line.
//
// Each station is a control: its real place, its drawn place and a linear map of its own (its hops' drawn vectors
// against their real ones, drawn toward their scale). A place is the blend of its nearest stations' maps, weights
// 1/d^4: by a station it lands by the station, on the side it had in reality, and between stations it follows the
// drawing's own stretch. Each station's map reaches at most `reach` layout units, so a place far from every station
// stays near the stations' edge rather than shooting off past other blocks.

import type { DiagramGraph } from "./graph";

/** the nearest stations blended */
const K = 8;
/** a station's weight 1 / (d^2 + EPS^2)^2, metres */
const EPS_M = 50;

export interface Warp {
  /** the drawn place of real place (x, y), into `out` [x, y]; in `out` [2] the drawn distance from there to the
   * nearest drawn station `seen` (layout units), in [3] and [4] the real distance to the nearest station `seen`
   * (metres) and that station */
  at(x: number, y: number, out: Float64Array): void;
}

/** The warp of a laid out graph: `x`, `y` drawn places per node; `seen` (per node, 1 or 0) picks the stations
 * `at` measures distances to, all when left out. */
export function makeWarp(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, reach: number, seen?: ArrayLike<number>): Warp {
  const J = localMaps(g, x, y);
  const all = kdTree(g.x, g.y);
  const pick: number[] = [];
  for (let i = 0; i < g.key.length; i++) if (!seen || seen[i]) pick.push(i);
  const dx = Float64Array.from(pick, (i) => x[i]), dy = Float64Array.from(pick, (i) => y[i]);
  const rx = Float64Array.from(pick, (i) => g.x[i]), ry = Float64Array.from(pick, (i) => g.y[i]);
  const drawnTree = kdTree(dx, dy), realTree = kdTree(rx, ry);
  const near = new Int32Array(K), nd = new Float64Array(K), one = new Int32Array(1), oned = new Float64Array(1);
  return {
    at(sx: number, sy: number, out: Float64Array) {
      const found = all.nearest(sx, sy, K, near, nd);
      let sw = 0, ax = 0, ay = 0;
      for (let q = 0; q < found; q++) {
        const i = near[q], d2 = nd[q] + EPS_M * EPS_M, w = 1 / (d2 * d2), ux = sx - g.x[i], uy = sy - g.y[i];
        let ox = J[i * 4] * ux + J[i * 4 + 1] * uy, oy = J[i * 4 + 2] * ux + J[i * 4 + 3] * uy;
        const ol = Math.sqrt(ox * ox + oy * oy);
        if (ol > reach) (ox *= reach / ol), (oy *= reach / ol);
        (ax += w * (x[i] + ox)), (ay += w * (y[i] + oy)), (sw += w);
      }
      out[0] = found ? ax / sw : sx;
      out[1] = found ? ay / sw : sy;
      if (!pick.length) {
        (out[2] = Infinity), (out[3] = Infinity), (out[4] = -1);
        return;
      }
      drawnTree.nearest(out[0], out[1], 1, one, oned);
      out[2] = Math.sqrt(oned[0]);
      realTree.nearest(sx, sy, 1, one, oned);
      out[3] = Math.sqrt(oned[0]);
      out[4] = pick[one[0]];
    },
  };
}

/** per station a linear map (a, b, d, e: X = a x + b y) from its hops' drawn vectors against their real ones, drawn
 * toward its hops' own scale */
function localMaps(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>): Float64Array {
  const n = g.key.length, J = new Float64Array(n * 4);
  const nbr: number[][] = Array.from({ length: n }, () => []);
  for (let e = 0; e < g.ea.length; e++) nbr[g.ea[e]].push(g.eb[e]), nbr[g.eb[e]].push(g.ea[e]);
  for (let i = 0; i < n; i++) {
    let rxx = 0, rxy = 0, ryy = 0, dxx = 0, dxy = 0, dyx = 0, dyy = 0, sr = 0, sd = 0;
    for (const j of nbr[i]) {
      const rx = g.x[j] - g.x[i], ry = g.y[j] - g.y[i], qx = x[j] - x[i], qy = y[j] - y[i];
      (rxx += rx * rx), (rxy += rx * ry), (ryy += ry * ry);
      (dxx += qx * rx), (dxy += qx * ry), (dyx += qy * rx), (dyy += qy * ry);
      (sr += Math.sqrt(rx * rx + ry * ry)), (sd += Math.sqrt(qx * qx + qy * qy));
    }
    const s = sr > 0 ? sd / sr : 0.005, mu = nbr[i].length ? (rxx + ryy) / nbr[i].length : 1;
    const m00 = rxx + mu, m01 = rxy, m11 = ryy + mu, det = m00 * m11 - m01 * m01;
    const n00 = dxx + mu * s, n01 = dxy, n10 = dyx, n11 = dyy + mu * s;
    const i00 = m11 / det, i01 = -m01 / det, i11 = m00 / det;
    J[i * 4] = n00 * i00 + n01 * i01;
    J[i * 4 + 1] = n00 * i01 + n01 * i11;
    J[i * 4 + 2] = n10 * i00 + n11 * i01;
    J[i * 4 + 3] = n10 * i01 + n11 * i11;
  }
  return J;
}

/** A k-d tree over points (x, y): `nearest` fills `idx`, `d2` with the k nearest, nearest first, and gives
 * how many it found. */
function kdTree(xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const n = xs.length, ord = Int32Array.from({ length: n }, (_, i) => i);
  const select = (lo: number, hi: number, k: number, c: ArrayLike<number>) => {
    while (hi > lo) {
      const p = c[ord[(lo + hi) >> 1]];
      let i = lo, j = hi;
      while (i <= j) {
        while (c[ord[i]] < p) i++;
        while (c[ord[j]] > p) j--;
        if (i <= j) {
          const t = ord[i];
          (ord[i] = ord[j]), (ord[j] = t), i++, j--;
        }
      }
      if (k <= j) hi = j;
      else if (k >= i) lo = i;
      else return;
    }
  };
  // ord[lo..hi) is split at its middle on axis depth % 2, each half built the same way
  const build = (lo: number, hi: number, depth: number) => {
    if (hi - lo <= 1) return;
    const mid = (lo + hi) >> 1;
    select(lo, hi - 1, mid, depth & 1 ? ys : xs);
    build(lo, mid, depth + 1);
    build(mid + 1, hi, depth + 1);
  };
  build(0, n, 0);
  return {
    nearest(px: number, py: number, k: number, idx: Int32Array, d2: Float64Array): number {
      let found = 0;
      const visit = (lo: number, hi: number, depth: number) => {
        if (hi <= lo) return;
        const mid = (lo + hi) >> 1, i = ord[mid];
        const dx = xs[i] - px, dy = ys[i] - py, dd = dx * dx + dy * dy;
        if (found < k || dd < d2[k - 1]) {
          let s = found < k ? found++ : k - 1;
          while (s > 0 && d2[s - 1] > dd) (d2[s] = d2[s - 1]), (idx[s] = idx[s - 1]), s--;
          (d2[s] = dd), (idx[s] = i);
        }
        const off = depth & 1 ? -dy : -dx;
        if (off < 0) {
          visit(lo, mid, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(mid + 1, hi, depth + 1);
        } else {
          visit(mid + 1, hi, depth + 1);
          if (found < k || off * off < d2[k - 1]) visit(lo, mid, depth + 1);
        }
      };
      visit(0, n, 0);
      return found;
    },
  };
}

