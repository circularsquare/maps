// The sea and lakes on the network diagram (T-2169; docs/ui-diagram.md "Water"), decided in the drawing the way the
// zone borders are (borders.ts): on a grid over the drawing, each cell is sea where the places on the ground its nearest
// drawn stations' local maps carry it to are sea (waterReal.ts `seaAt`), voted by nearness; every pill and the ground
// round it is land. Small specks are dropped, and the coast is fitted with runs at 0, 45 and 90 degrees, free to stray
// seaward but hardly landward, so every station stays on land even where the shape departs from the real coast.

import type { DiagramGraph } from "./graph";
import { pillSize, type Drawing } from "./draw";
import { fitOcti, inRings, rounded } from "./borders";
import { hyp } from "./fmath";
import type { RealWater } from "./waterReal";

/** the grid's cell, pitches (larger for big drawings, so at most MAX_CELLS cells) */
const CELL = 1.5, MAX_CELLS = 1_000_000;
/** the sea is drawn this far past the drawing's box, pitches */
const MARGIN = 40;
/** stations voting for a cell; ground within LAND_M of a pill's edge is land, pitches */
const VOTERS = 4, LAND = 2.5;
/** cells looked up together, a side, and the stations nearest their middle each cell's voters are taken from */
const BLOCK = 4, CANDIDATES = 10;
/** each station looks for the sea in RAYS ways on the ground, FACE_HOPS of its mean hop far and at least FACE_M, in
 * steps of STEP_M at least, metres */
const RAYS = 16, FACE_HOPS = 3, FACE_M = 3000, STEP_M = 100;
/** sea specks and lakes under this many square pitches, and land without a station under LAND_MIN, are dropped */
const SEA_MIN = 60, LAND_MIN = 30;
/** a fitted coast strays at most this far landward and seaward from the grid's, pitches */
const TOL_LAND = 2, TOL_SEA = 6;
/** corners rounded with this radius, pitches */
const RADIUS = 3;

export interface SeaStats { cells: number; cell: number; sea: number; rings: number; refit: number; ms: number }

/** The sea in the drawing `d` of `g` laid out at `x`, `y`, from `real`: path commands, filled even-odd. */
export function diagramSea(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, d: Drawing, real: RealWater, J: Float64Array): { cmds: Float64Array; rings: Float64Array[]; stats: SeaStats } {
  const t0 = performance.now();
  const n = g.key.length, pitch = d.pitch;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) (x0 = Math.min(x0, x[i])), (x1 = Math.max(x1, x[i])), (y0 = Math.min(y0, y[i])), (y1 = Math.max(y1, y[i]));
  (x0 -= MARGIN * pitch), (y0 -= MARGIN * pitch), (x1 += MARGIN * pitch), (y1 += MARGIN * pitch);
  let h = CELL * pitch;
  if (((x1 - x0) / h) * ((y1 - y0) / h) > MAX_CELLS) h = Math.sqrt(((x1 - x0) * (y1 - y0)) / MAX_CELLS);
  const W = Math.ceil((x1 - x0) / h), H = Math.ceil((y1 - y0) / h), NC = W * H;

  // each station's map from the drawing back to the ground: the inverse of its local map
  const inv = new Float64Array(n * 4), ok = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const a = J[i * 4], b = J[i * 4 + 1], c = J[i * 4 + 2], e = J[i * 4 + 3], det = a * e - b * c;
    if (!(Math.abs(det) > 1e-18)) continue;
    (inv[i * 4] = e / det), (inv[i * 4 + 1] = -b / det), (inv[i * 4 + 2] = -c / det), (inv[i * 4 + 3] = a / det), (ok[i] = 1);
  }
  // per station and each of RAYS ways on the ground, how far to the sea, up to FACE_HOPS of its mean hop (at least
  // FACE_M); Infinity for no sea that near
  const face = new Float64Array(n), cnt = new Float64Array(n);
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e], b = g.eb[e], l = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a]);
    (face[a] += l), (face[b] += l), cnt[a]++, cnt[b]++;
  }
  const toSea = new Float64Array(n * RAYS).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const reach = Math.max(FACE_M, cnt[i] ? (FACE_HOPS * face[i]) / cnt[i] : 0), step = Math.max(STEP_M, reach / 40);
    for (let r = 0; r < RAYS; r++) {
      const ux = Math.cos((2 * Math.PI * r) / RAYS), uy = Math.sin((2 * Math.PI * r) / RAYS);
      for (let s = step; s <= reach; s += step) if (real.seaAt(g.x[i] + ux * s, g.y[i] + uy * s) === 1) {
        toSea[i * RAYS + r] = s;
        break;
      }
    }
  }
  // a cell is sea where its nearest drawn stations, voting by nearness, each find it past the sea the way it lies from
  // them on the ground (through their maps back to the ground)
  // (the nearest stations of each BLOCK by BLOCK square of cells looked up once, CANDIDATES of them, and each cell's
  // nearest taken among those)
  const kd = kdTree(x, y), cand = new Int32Array(CANDIDATES), cd = new Float64Array(CANDIDATES);
  const near = new Int32Array(VOTERS), nd = new Float64Array(VOTERS);
  const cls = new Uint8Array(NC);
  for (let bj = 0; bj < H; bj += BLOCK) for (let bi = 0; bi < W; bi += BLOCK) {
    const nc = kd.nearest(x0 + (bi + BLOCK / 2) * h, y0 + (bj + BLOCK / 2) * h, CANDIDATES, cand, cd);
    for (let j = bj; j < Math.min(H, bj + BLOCK); j++) for (let i = bi; i < Math.min(W, bi + BLOCK); i++) {
    const px = x0 + (i + 0.5) * h, py = y0 + (j + 0.5) * h;
    let found = 0;
    for (let q = 0; q < nc; q++) {
      const k = cand[q], dd = (x[k] - px) ** 2 + (y[k] - py) ** 2;
      if (found < VOTERS || dd < nd[VOTERS - 1]) {
        let s = found < VOTERS ? found++ : VOTERS - 1;
        while (s > 0 && nd[s - 1] > dd) (nd[s] = nd[s - 1]), (near[s] = near[s - 1]), s--;
        (nd[s] = dd), (near[s] = k);
      }
    }
    let sea = 0, land = 0;
    for (let q = 0; q < found; q++) {
      const k = near[q];
      if (!ok[k]) continue;
      const ox = px - x[k], oy = py - y[k], w = 1 / (nd[q] + pitch * pitch);
      const rx = inv[k * 4] * ox + inv[k * 4 + 1] * oy, ry = inv[k * 4 + 2] * ox + inv[k * 4 + 3] * oy;
      const ray = Math.round((Math.atan2(ry, rx) / (2 * Math.PI)) * RAYS + RAYS) % RAYS;
      if (toSea[k * RAYS + ray] <= hyp(rx, ry)) sea += w;
      else land += w;
    }
    cls[j * W + i] = sea > land ? 1 : 0;
    }
  }
  // every pill and the ground round it is land
  const forced = new Uint8Array(NC);
  d.pills.forEach((p, q) => {
    const [r, o] = pillSize(d, d.pillOf[q]), R = r + o + p[4] + LAND * pitch;
    const i0 = Math.max(0, Math.floor((Math.min(p[0], p[2]) - R - x0) / h)), i1 = Math.min(W - 1, Math.floor((Math.max(p[0], p[2]) + R - x0) / h));
    const j0 = Math.max(0, Math.floor((Math.min(p[1], p[3]) - R - y0) / h)), j1 = Math.min(H - 1, Math.floor((Math.max(p[1], p[3]) + R - y0) / h));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      // a cell is land when any of it is within R of the pill's axis
      const cx = x0 + (i + 0.5) * h, cy = y0 + (j + 0.5) * h;
      if (segD(cx, cy, p[0], p[1], p[2], p[3]) <= R + h * Math.SQRT1_2) (forced[j * W + i] = 1), (cls[j * W + i] = 0);
    }
  });
  // smoothed (a majority of each 3 by 3, twice), then small pieces of either dropped
  const tmp = new Uint8Array(NC);
  for (let pass = 0; pass < 2; pass++) {
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      let s = 0, m = 0;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ni = i + di, nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        (s += cls[nj * W + ni]), m++;
      }
      tmp[j * W + i] = forced[j * W + i] ? 0 : s * 2 > m ? 1 : 0;
    }
    cls.set(tmp);
  }
  const area = h * h / (pitch * pitch);
  dropSmall(cls, W, H, 1, SEA_MIN / area, forced);
  dropSmall(cls, W, H, 0, LAND_MIN / area, forced);

  // the coast: unit edges between sea and land cells, sea on the left, chained into rings at the grid's corners
  const W1 = W + 1, out = new Map<number, number[]>(), ea: number[] = [], eb: number[] = [];
  const sea = (i: number, j: number) => i >= 0 && j >= 0 && i < W && j < H && cls[j * W + i] === 1;
  const edge = (a: number, b: number) => {
    const e = ea.length;
    ea.push(a), eb.push(b);
    const l = out.get(a);
    if (l) l.push(e);
    else out.set(a, [e]);
  };
  let seaCells = 0;
  for (let j = -1; j < H; j++) for (let i = -1; i < W; i++) {
    const s = sea(i, j);
    if (s && i >= 0 && j >= 0) seaCells++;
    // the edge east of cell (i, j), between it and (i + 1, j); and north, between it and (i, j + 1). Corner (i, j) is
    // the cell's south-west one: corner id (j + 1) * W1 + (i + 1) shifted by one so -1 fits
    const c = (ci: number, cj: number) => (cj + 1) * (W1 + 1) + (ci + 1);
    if (s !== sea(i + 1, j)) s ? edge(c(i + 1, j), c(i + 1, j + 1)) : edge(c(i + 1, j + 1), c(i + 1, j));
    if (s !== sea(i, j + 1)) s ? edge(c(i + 1, j + 1), c(i, j + 1)) : edge(c(i, j + 1), c(i + 1, j + 1));
  }
  const cxy = (v: number): [number, number] => [x0 + ((v % (W1 + 1)) - 1) * h, y0 + (Math.floor(v / (W1 + 1)) - 1) * h];
  const used = new Uint8Array(ea.length), rings: Float64Array[] = [], cmds: number[] = [];
  const pillPts: number[] = [];
  for (const p of d.pills) pillPts.push((p[0] + p[2]) / 2, (p[1] + p[3]) / 2);
  let refit = 0;
  for (let e0 = 0; e0 < ea.length; e0++) {
    if (used[e0]) continue;
    const px: number[] = [], py: number[] = [];
    let e = e0;
    for (let guard = 0; guard <= ea.length; guard++) {
      used[e] = 1;
      const [ax, ay] = cxy(ea[e]), [bx, by] = cxy(eb[e]);
      px.push((ax + bx) / 2), py.push((ay + by) / 2);
      // at the end corner: the next unused edge, turning left first (so touching seas stay apart)
      const l = out.get(eb[e]) ?? [];
      const dx = bx - ax, dy = by - ay;
      let next = -1, bestTurn = -Infinity;
      for (const f of l) {
        if (used[f] && f !== e0) continue;
        const [fx, fy] = cxy(ea[f]), [gx, gy] = cxy(eb[f]), ux = gx - fx, uy = gy - fy;
        const turn = dx * uy - dy * ux + (dx * ux + dy * uy) * 1e-3;
        if (turn > bestTurn) (bestTurn = turn), (next = f);
      }
      if (next < 0 || next === e0) break;
      e = next;
    }
    if (px.length < 4) continue;
    // start where the ring turns, its point farthest south-west, and close it
    let s = 0;
    for (let k = 1; k < px.length; k++) if (px[k] + py[k] < px[s] + py[s]) s = k;
    const qx = [...px.slice(s), ...px.slice(0, s + 1)], qy = [...py.slice(s), ...py.slice(0, s + 1)];
    const fit = (tl: number, ts: number) => fitOcti(qx, qy, qx.map(() => tl * pitch), undefined, { tol: qx.map(() => ts * pitch), left: true });
    let v = fit(TOL_LAND, TOL_SEA);
    // a fit that took a pill into the sea is done again tighter
    if (pillsIn(v, pillPts, cls, W, H, x0, y0, h)) (v = fit(0.3, 0.3)), refit++;
    rings.push(v);
    rounded(v, true, RADIUS * pitch, cmds);
  }
  return { cmds: Float64Array.from(cmds), rings, stats: { cells: NC, cell: h / pitch, sea: seaCells, rings: rings.length, refit, ms: performance.now() - t0 } };
}

/** whether ring v holds a pill's middle that its cell says is land (so the fit moved the coast over it) */
function pillsIn(v: Float64Array, pts: number[], cls: Uint8Array, W: number, H: number, x0: number, y0: number, h: number): boolean {
  let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
  for (let k = 0; k < v.length; k += 2) (bx0 = Math.min(bx0, v[k])), (bx1 = Math.max(bx1, v[k])), (by0 = Math.min(by0, v[k + 1])), (by1 = Math.max(by1, v[k + 1]));
  for (let k = 0; k < pts.length; k += 2) {
    const px = pts[k], py = pts[k + 1];
    if (px < bx0 || px > bx1 || py < by0 || py > by1) continue;
    const i = Math.floor((px - x0) / h), j = Math.floor((py - y0) / h);
    if (i < 0 || j < 0 || i >= W || j >= H || cls[j * W + i] === 1) continue;
    if (inRings([v], px, py)) return true;
  }
  return false;
}

/** pieces of class `c` (4-connected) under `min` cells turned to the other class, but never a forced land cell to sea */
function dropSmall(cls: Uint8Array, W: number, H: number, c: number, min: number, forced: Uint8Array) {
  const seen = new Uint8Array(W * H);
  for (let s = 0; s < W * H; s++) {
    if (seen[s] || cls[s] !== c) continue;
    const q = [s];
    seen[s] = 1;
    let pinned = false;
    for (let k = 0; k < q.length; k++) {
      const v = q[k], i = v % W, j = (v - i) / W;
      if (forced[v]) pinned = true;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ni = i + di, nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        const nv = nj * W + ni;
        if (!seen[nv] && cls[nv] === c) (seen[nv] = 1), q.push(nv);
      }
    }
    if (q.length < min && !(c === 0 && pinned)) for (const v of q) cls[v] = 1 - c;
  }
}

function segD(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
  return hyp(px - ax - dx * t, py - ay - dy * t);
}

function kdTree(xs: ArrayLike<number>, ys: ArrayLike<number>) {
  const n = xs.length, ord = Int32Array.from({ length: n }, (_, i) => i);
  const build = (lo: number, hi: number, depth: number) => {
    if (hi - lo <= 1) return;
    const c = depth & 1 ? ys : xs, sub = Array.from(ord.subarray(lo, hi)).sort((a, b) => c[a] - c[b]);
    ord.set(sub, lo);
    const mid = (lo + hi) >> 1;
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
