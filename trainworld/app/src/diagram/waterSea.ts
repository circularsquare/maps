// The sea and lakes on the network diagram (T-2169; docs/ui-diagram.md "Water"), decided in the drawing the way the
// zone borders are (borders.ts): on a grid over the drawing, each cell is sea where the places on the ground its nearest
// drawn stations' local maps carry it to are sea (waterReal.ts `seaAt`), voted by nearness; every pill and the ground
// round it is land. Small specks are dropped, and the coast is fitted with runs at 0, 45 and 90 degrees, free to stray
// seaward but hardly landward, so every station stays on land even where the shape departs from the real coast.

import type { DiagramGraph } from "./graph";
import { pillSize, type Drawing } from "./draw";
import { fitOcti, inRings, rounded } from "./borders";
import { hyp } from "./fmath";
import { chamfer, type RealWater } from "./waterReal";

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
/** the coast smoothed at this scale before it is fitted: inlets and notches under about twice it across go, pitches */
const SMOOTH = 4;
/** a fitted coast strays at most TOL_SEA seaward from the grid's, and landward its clearance from the lines and pills
 * less CLEAR, within TOL_MIN and TOL_LAND, pitches (T-2497: wide, so it runs straight for long like a line) */
const TOL_SEA = 12, TOL_LAND = 8, TOL_MIN = 1, CLEAR = 1;
/** corners rounded with this radius, pitches */
const RADIUS = 4;

export interface SeaStats { cells: number; cell: number; sea: number; rings: number; refit: number; ms: number }

/** The sea in the drawing `d` of `g` laid out at `x`, `y`, from `real`: path commands, filled even-odd. */
export function diagramSea(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, d: Drawing, real: RealWater, J: Float64Array): { cmds: Float64Array; rings: Float64Array[]; grid: SeaGrid; stats: SeaStats } {
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
  const toSea = new Float64Array(n * RAYS).fill(Infinity), toBody = new Int32Array(n * RAYS).fill(-1);
  for (let i = 0; i < n; i++) {
    const reach = Math.max(FACE_M, cnt[i] ? (FACE_HOPS * face[i]) / cnt[i] : 0), step = Math.max(STEP_M, reach / 40);
    for (let r = 0; r < RAYS; r++) {
      const ux = Math.cos((2 * Math.PI * r) / RAYS), uy = Math.sin((2 * Math.PI * r) / RAYS);
      for (let s = step; s <= reach; s += step) if (real.seaAt(g.x[i] + ux * s, g.y[i] + uy * s) === 1) {
        toSea[i * RAYS + r] = s;
        toBody[i * RAYS + r] = real.seaBody(g.x[i] + ux * s, g.y[i] + uy * s);
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
  const cls = new Uint8Array(NC), cellBody = new Int32Array(NC).fill(-1);
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
    let sea = 0, land = 0, bw = 0;
    for (let q = 0; q < found; q++) {
      const k = near[q];
      if (!ok[k]) continue;
      const ox = px - x[k], oy = py - y[k], w = 1 / (nd[q] + pitch * pitch);
      const rx = inv[k * 4] * ox + inv[k * 4 + 1] * oy, ry = inv[k * 4 + 2] * ox + inv[k * 4 + 3] * oy;
      const ray = Math.round((Math.atan2(ry, rx) / (2 * Math.PI)) * RAYS + RAYS) % RAYS;
      if (toSea[k * RAYS + ray] <= hyp(rx, ry)) {
        sea += w;
        if (w > bw) (bw = w), (cellBody[j * W + i] = toBody[k * RAYS + ray]);
      } else land += w;
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
  // the lines' casings, and each cell's clearance from them and the land round the pills (chamfer steps, 3 a cell)
  const block = forced.slice();
  for (const r of d.ribbons) for (let k = 0; k + 3 < r.length; k += 2) {
    const ax = r[k], ay = r[k + 1], bx = r[k + 2], by = r[k + 3], R = pitch / 2 + h * Math.SQRT1_2;
    const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - R - x0) / h)), i1 = Math.min(W - 1, Math.floor((Math.max(ax, bx) + R - x0) / h));
    const j0 = Math.max(0, Math.floor((Math.min(ay, by) - R - y0) / h)), j1 = Math.min(H - 1, Math.floor((Math.max(ay, by) + R - y0) / h));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (segD(x0 + (i + 0.5) * h, y0 + (j + 0.5) * h, ax, ay, bx, by) <= R) block[j * W + i] = 1;
  }
  const clearCap = Math.min(65000, 3 * Math.ceil((TOL_LAND + CLEAR) * pitch / h) + 3);
  const clear = chamfer(block, W, H, true, clearCap);
  // smoothed (a majority of each 3 by 3, twice; then opened and closed by SMOOTH), then small pieces of either dropped
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
  {
    const rc = 3 * Math.max(1, Math.round((SMOOTH * pitch) / h)), cap = rc + 8;
    const grown = (src: Uint8Array, from1: boolean) => Uint8Array.from(chamfer(src, W, H, from1, cap), (v) => +(v <= rc));
    // opened: the sea shrunk and grown back; then closed: grown and shrunk back; the land round the pills kept
    const open = grown(Uint8Array.from(chamfer(cls, W, H, false, cap), (v) => +(v > rc)), true);
    const closed = grown(grown(open, true), false);
    for (let c = 0; c < NC; c++) cls[c] = forced[c] ? 0 : 1 - closed[c];
  }
  const area = h * h / (pitch * pitch);
  dropSmall(cls, W, H, 1, SEA_MIN / area, forced);
  dropSmall(cls, W, H, 0, LAND_MIN / area, forced);

  // the coast: unit edges between sea and land cells, sea on the left, chained into rings at the grid's corners
  // (each edge with the landward tolerance its land cell's clearance allows)
  const W1 = W + 1, out = new Map<number, number[]>(), ea: number[] = [], eb: number[] = [], et: number[] = [];
  const sea = (i: number, j: number) => i >= 0 && j >= 0 && i < W && j < H && cls[j * W + i] === 1;
  const tolAt = (i: number, j: number) => {
    const cl = i >= 0 && j >= 0 && i < W && j < H ? (clear[j * W + i] / 3) * h : Infinity;
    return Math.max(TOL_MIN * pitch, Math.min(TOL_LAND * pitch, cl - CLEAR * pitch));
  };
  const edge = (a: number, b: number, t: number) => {
    const e = ea.length;
    ea.push(a), eb.push(b), et.push(t);
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
    if (s !== sea(i + 1, j)) s ? edge(c(i + 1, j), c(i + 1, j + 1), tolAt(i + 1, j)) : edge(c(i + 1, j + 1), c(i + 1, j), tolAt(i, j));
    if (s !== sea(i, j + 1)) s ? edge(c(i + 1, j + 1), c(i, j + 1), tolAt(i, j + 1)) : edge(c(i, j + 1), c(i + 1, j + 1), tolAt(i, j));
  }
  const cxy = (v: number): [number, number] => [x0 + ((v % (W1 + 1)) - 1) * h, y0 + (Math.floor(v / (W1 + 1)) - 1) * h];
  const used = new Uint8Array(ea.length), rings: Float64Array[] = [], cells: [number[], number[]][] = [], cmds: number[] = [];
  const pillPts: number[] = [];
  for (const p of d.pills) pillPts.push((p[0] + p[2]) / 2, (p[1] + p[3]) / 2);
  // the pills' ends and middles, by squares of PG: a fitted run may not move any of them to its other side
  const PG = 8 * pitch, pg = new Map<number, number[]>();
  const gkey = (i: number, j: number) => (i + 32768) * 65536 + (j + 32768);
  for (const p of d.pills) for (const [qx, qy] of [[p[0], p[1]], [p[2], p[3]], [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2]]) {
    const key = gkey(Math.floor(qx / PG), Math.floor(qy / PG)), l = pg.get(key);
    if (l) l.push(qx, qy);
    else pg.set(key, [qx, qy]);
  }
  const keepsSides = (px: number[], py: number[]) => (i: number, j: number, bx: number, by: number) => {
    const poly = px.slice(i, j + 1).flatMap((x, k) => [x, py[i + k]]);
    if (bx === bx) poly.push(bx, by);
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (let k = 0; k < poly.length; k += 2) (bx0 = Math.min(bx0, poly[k])), (bx1 = Math.max(bx1, poly[k])), (by0 = Math.min(by0, poly[k + 1])), (by1 = Math.max(by1, poly[k + 1]));
    for (let gx = Math.floor(bx0 / PG); gx <= Math.floor(bx1 / PG); gx++) for (let gy = Math.floor(by0 / PG); gy <= Math.floor(by1 / PG); gy++) {
      const l = pg.get(gkey(gx, gy));
      if (l) for (let q = 0; q < l.length; q += 2) if (l[q] >= bx0 && l[q] <= bx1 && l[q + 1] >= by0 && l[q + 1] <= by1 && inRings([poly], l[q], l[q + 1])) return false;
    }
    return true;
  };
  let refit = 0;
  for (let e0 = 0; e0 < ea.length; e0++) {
    if (used[e0]) continue;
    const px: number[] = [], py: number[] = [], pt: number[] = [];
    let e = e0;
    for (let guard = 0; guard <= ea.length; guard++) {
      used[e] = 1;
      const [ax, ay] = cxy(ea[e]), [bx, by] = cxy(eb[e]);
      px.push((ax + bx) / 2), py.push((ay + by) / 2), pt.push(et[e]);
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
    const rot = (a: number[]) => [...a.slice(s), ...a.slice(0, s + 1)];
    const qx = rot(px), qy = rot(py);
    cells.push([qx, qy]);
    rings.push(fitOcti(qx, qy, rot(pt), keepsSides(qx, qy), { tol: qx.map(() => TOL_SEA * pitch), left: true }));
  }
  // each ring keeps every pill on its own side, but two rings fitted near each other can still cross: a ring near a
  // pill left in the sea is fitted again, tight
  for (let pass = 0; pass < 2; pass++) {
    const redo = new Set<number>();
    for (let k = 0; k < pillPts.length; k += 2) {
      const qx = pillPts[k], qy = pillPts[k + 1];
      if (!inRings(rings, qx, qy)) continue;
      rings.forEach((v, q) => {
        const m = v.length;
        for (let z = 0; z < m; z += 2) if (segD(qx, qy, v[z], v[z + 1], v[(z + 2) % m], v[(z + 3) % m]) < (TOL_SEA + TOL_LAND + 2) * pitch) return void redo.add(q);
      });
    }
    if (!redo.size) break;
    for (const q of redo) {
      const [qx, qy] = cells[q], tight = qx.map(() => 0.3 * pitch);
      (rings[q] = fitOcti(qx, qy, tight, keepsSides(qx, qy), { tol: tight, left: true })), refit++;
    }
  }
  for (const v of rings) rounded(v, true, RADIUS * pitch, cmds);
  // the sea as drawn back on the grid (even-odd, by rows), in connected regions, each with the sea on the ground most
  // of its voted cells found (T-2516: rivers are drawn into the region of the sea they meet)
  const wet = new Uint8Array(NC);
  {
    const xs: number[][] = Array.from({ length: H }, () => []);
    for (const v of rings) {
      const m = v.length / 2;
      for (let k = 0; k < m; k++) {
        const ax = v[k * 2], ay = v[k * 2 + 1], bx = v[((k + 1) % m) * 2], by = v[((k + 1) % m) * 2 + 1];
        if (ay === by) continue;
        const lo = Math.max(0, Math.ceil((Math.min(ay, by) - y0) / h - 0.5)), hi = Math.min(H - 1, Math.floor((Math.max(ay, by) - y0) / h - 0.5));
        for (let j = lo; j <= hi; j++) {
          const yc = y0 + (j + 0.5) * h;
          if (ay > yc !== by > yc) xs[j].push(ax + ((yc - ay) * (bx - ax)) / (by - ay));
        }
      }
    }
    for (let j = 0; j < H; j++) {
      const l = xs[j].sort((a, b) => a - b);
      for (let q = 0; q + 1 < l.length; q += 2) for (let i = Math.max(0, Math.ceil((l[q] - x0) / h - 0.5)); i <= Math.min(W - 1, Math.floor((l[q + 1] - x0) / h - 0.5)); i++) wet[j * W + i] = 1;
    }
  }
  const reg = new Int32Array(NC).fill(-1), body: number[] = [];
  for (let s = 0; s < NC; s++) {
    if (!wet[s] || reg[s] >= 0) continue;
    const id = body.length, q = [s], votes = new Map<number, number>();
    reg[s] = id;
    for (let k = 0; k < q.length; k++) {
      const v = q[k], i = v % W, j = (v - i) / W;
      if (cellBody[v] >= 0) votes.set(cellBody[v], (votes.get(cellBody[v]) ?? 0) + 1);
      for (const nv of [i > 0 ? v - 1 : -1, i < W - 1 ? v + 1 : -1, j > 0 ? v - W : -1, j < H - 1 ? v + W : -1]) if (nv >= 0 && wet[nv] && reg[nv] < 0) (reg[nv] = id), q.push(nv);
    }
    let best = -1, bc = 0;
    for (const [b, c] of votes) if (c > bc) (best = b), (bc = c);
    body.push(best);
  }
  const grid: SeaGrid = { x0, y0, h, W, H, reg, body, depth: chamfer(wet, W, H, false, 65000) };
  return { cmds: Float64Array.from(cmds), rings, grid, stats: { cells: NC, cell: h / pitch, sea: seaCells, rings: rings.length, refit, ms: performance.now() - t0 } };
}

/** the sea as drawn on a grid of cells `h` wide from (x0, y0): each cell's region (-1 land), each region's sea on the
 * ground (-1 unknown), each cell's depth into the sea (chamfer steps, 3 a cell) */
export interface SeaGrid { x0: number; y0: number; h: number; W: number; H: number; reg: Int32Array; body: number[]; depth: Uint16Array }

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
