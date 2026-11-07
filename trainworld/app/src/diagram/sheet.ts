// The rubber sheet (T-2566, T-2568; docs/ui-diagram.md "Rubber sheet"): one continuous warp of the real plane, which
// the stations follow (and, in later stages, the water and the zones' borders). A lattice of square cells over the
// stations' box, each cell two triangles mapped linearly, solved by least squares: a cell's uneven stretch costs most
// on land near lines, less on land without lines, little on water, and its size is pulled to a field of the scale its
// stations ask for (their room and the distortion dial), bounded so it at most doubles every D drawn pitches. The
// rules are T-2566's rounds 2 and 3 (notes/T-2566.md); the solver is T-2568's (notes/T-2568.md). On a country map a
// cell's rest shape is narrower than its real square where its columns or rows are empty of networks (the cut, T-2609).
//
// The lattice is a quadtree: cells as fine as the finest lattice where the field changes, along coasts near lines and
// on small islands, up to `coarse` finest cells a side elsewhere. A coarse cell's weights are its finest cells' means,
// so it stands for them exactly while the map is linear across it. The quadtree cut at each depth is a level; a round
// is one least-squares solve on a level, by conjugate gradients with a multigrid V-cycle (Galerkin coarse levels, the
// coarser cuts). `schedule` says which rounds run on which level: the early rounds, where the map moves far, on
// coarser levels; the finest last, with the holds on cells near folding frozen so the sheet settles. Pure: no DOM, no
// fetch.

import type { DiagramGraph } from "./graph";
import type { GroundSea, WaterMask } from "./waterReal";

export interface SheetOptions {
  /** the distortion dial, as the layout's (T-2556) */
  dial: number;
  /** the scale doubles at most every D drawn pitches; and at most maxStep times across one finest cell */
  D: number;
  maxStep: number;
  /** room: (strokes + gap) pitches between near stations, times `core` */
  gap: number;
  core: number;
  /** `core` on a map 600 km across or more */
  coreCountry: number;
  /** the three kinds of cell (land near lines, land without, water): the cost of stretching unevenly (`a`), and the
   * pull of the size to the field's (`s`); on land near lines the pull is times the cell's stations over the median
   * cell's, between 0.3 and `sMax` */
  aLine: number;
  sLine: number;
  sMax: number;
  aLand: number;
  sLand: number;
  aWater: number;
  sWater: number;
  /** the cut, on a map 600 km across or more: the map's columns (rows) farther than cutKm across from every network
   * station (one with cutN others within cutKm; 0: off) are drawn narrow, past cutKm the width halving every cutD drawn
   * pitches at the floor's scale, to cutMin at least (T-2609, T-2629) */
  cutKm: number;
  cutN: number;
  cutD: number;
  cutMin: number;
  /** on such a map, land farther than freeKm from every network station costs aWater to stretch unevenly (0: off) */
  freeKm: number;
  /** on such a map, the floor falls away from the networks: each node's the best of the network stations' scales times
   * (1 + distance / farKm)^-farP, at most the floor and at least farMin of it; a station of no network asks at most the
   * floor where it is; the size pull on land near lines is times farS; and the rounds are `scheduleFar` (T-2625; farP 0: off) */
  farP: number;
  farKm: number;
  farMin: number;
  farS: number;
  scheduleFar: string;
  /** a cell near folding, or under `lo` of its planned size, is held with up to this weight */
  wFold: number;
  lo: number;
  /** land within nearKm of a station or track is near lines; a cell is land from `landShare` land */
  nearKm: number;
  landShare: number;
  /** a land mass of at most islandKm2 with a station: one size all over (the `islandQ` point of what it asks, times
   * `islandBoost`), each cell mostly on it costing `aIsland` to stretch unevenly */
  islandKm2: number;
  islandQ: number;
  islandBoost: number;
  aIsland: number;
  /** finest cells along the box's longer side, for a map under 600 km across and for a bigger one; the ground's
   * raster on a map under 600 km across, metres */
  nCity: number;
  nCountry: number;
  cityRaster: number;
  /** the field's smoothing radius and the near-lines reach are at least 3/4 of the box's longer side over refN - 1 (0: off) */
  refN: number;
  /** the floor times this on a map 600 km across or more */
  floorCountry: number;
  /** a quadtree cell stays whole where its field changes by under `flat` times across it, or it is drawn under
   * `drawn` pitches; at most `coarse` finest cells a side, and `nearMax` near lines (0: no limit) */
  flat: number;
  flatCountry: number;
  drawn: number;
  coarse: number;
  nearMax: number;
  /** if above 0, a cell splits where its field changes by more than this many pitches across it (instead of flat and drawn) */
  drift: number;
  /** cells on a coast near lines at most this many finest cells a side, on a city map and a country map */
  coastCity: number;
  coastCountry: number;
  /** every cell at the finest size (for comparing with the uniform lattice) */
  uniform: boolean;
  /** the rounds, comma separated, on level K0 + L (L <= 0, 0 the finest): "eL:R:A" R rounds with each hold moving the
   * share A of the way to what the stretch asks now; "L:R:K" R rounds with each hold the larger of what the stretch
   * asks and K times the last; "raise" (cores that came out smaller than the field asks get their wish raised by 80%
   * of the shortfall); "sN" up to N rounds on the finest level with the holds frozen (tightening if `tighten`);
   * "fL:N" frozen, never tightening; "hL:N" holds tightened once, then frozen; "uN" only if a cell folded, up to N
   * frozen rounds that tighten */
  schedule: string;
  /** rounds stop when the stations' p90 move (the drawing's shift taken out) is under this, pitches */
  settle: number;
  /** whether a frozen hold may still tighten */
  tighten: boolean;
  /** frozen rounds mix this many of the last rounds (Anderson; 0: none) */
  aa: number;
  /** whether a tightened hold starts the mixing afresh; frozen rounds that tighten remake the coarser levels every galEvery rounds */
  aaReset: boolean;
  galEvery: number;
  /** a frozen hold tightens only where the uneven stretch is at least this share of the even part (or the cell is under lo of its size) */
  tightFrom: number;
  /** each solve's tolerance (residual over right side) and iteration cap, in moving rounds and in frozen ones */
  tolMove: number;
  itMove: number;
  tol: number;
  maxIt: number;
}

export const SHEET: SheetOptions = {
  dial: 0.5, D: 40, maxStep: 1.6, gap: 4, core: 1.8, coreCountry: 2.1,
  aLine: 4, sLine: 1, sMax: 6, aLand: 2, sLand: 0.1, aWater: 0.1, sWater: 0.1,
  cutKm: 10, cutN: 2, cutD: 20, cutMin: 0.02, freeKm: 25, farP: 0.5, farKm: 2, farMin: 0.1, farS: 3,
  wFold: 10, lo: 0.3, nearKm: 3, landShare: 0.4,
  islandKm2: 300, islandQ: 0.75, islandBoost: 2, aIsland: 80,
  nCity: 512, nCountry: 1024, cityRaster: 50, refN: 256, floorCountry: 1.2, flat: 1.15, flatCountry: 1.4, drawn: 12, coarse: 16, nearMax: 0, drift: 0, coastCity: 1, coastCountry: 4, uniform: false,
  schedule: "e-2:10:0.3,raise,e-2:6:0.3,e0:5:0.3,s12", scheduleFar: "e-2:10:0.3,raise,e-2:6:0.3,raise,e-2:4:0.3,e0:5:0.3,s12", // i18n-ignore: solver schedules
  settle: 0.5, tighten: true, aa: 0, aaReset: true, galEvery: 1, tightFrom: 0.4, tolMove: 1e-3, itMove: 4, tol: 1e-3, maxIt: 6,
};

/** what the sheet reads of the ground: the zones' water masks, the graph's frame in mercator, the sea and land masses */
export interface SheetGround {
  masks: WaterMask[];
  frame: [number, number, number];
  sea: Pick<GroundSea, "isSea" | "massAt"> | null;
}

export interface Sheet {
  /** per graph node, its place on the sheet, pitches */
  x: Float64Array;
  y: Float64Array;
  /** the lattice: its finest cell (metres) and origin; per vertex its real place (finest-cell units from the origin,
   * i + j (gx + 1)) and drawn place; per cell its corner (i, j, finest cells) and side, and its land share */
  mesh: { h: number; x0: number; y0: number; gx: number; nx: number; ny: number; key: Int32Array; qx: Float64Array; qy: Float64Array; cell: Int32Array; land: Float32Array };
  /** the ground as the sheet read it (T-2567; the worker keeps it and does not post it) */
  raster?: SheetRaster;
  info: {
    stations: number; leaves: number; vertices: number; free: number; rounds: number;
    stages: string[]; folds: number; nearFold: number; islands: number; turnedDeg: number;
    cellKm: number; rasterM: number; floorPerKm: number; capKm: number; dispP90Km: number; overCap: number;
    ms: Record<string, number>;
  };
}

/** the ground's raster, `rr` metres a cell from (x0, y0), `w` by `h`: per row (rowStart) its land runs [runA, runB) of
 * cells, each run's land mass (`mass`: the same number for runs joined side by side); per row (covStart) the spans
 * [covA, covB) a mask covers (elsewhere the ground's sea, a guess past the masks, decided water) */
export interface SheetRaster { x0: number; y0: number; rr: number; w: number; h: number; rowStart: Int32Array; runA: Int32Array; runB: Int32Array; mass: Int32Array; covStart: Int32Array; covA: Int32Array; covB: Int32Array }

/** per land mass (`mass` numbers, which are run numbers), its coast in raster cell sides, and how much of that coast is cut: a masked land
 * cell beside an unmasked water cell, the mask's edge with the ground's guessed sea past it (T-2591) */
export function massCoasts(R: SheetRaster): { coast: Float64Array; cut: Float64Array } {
  const nM = R.mass.length, coast = new Float64Array(nM), cut = new Float64Array(nM);
  /** cells of [p, q) row j's masks cover */
  const covered = (j: number, p: number, q: number) => {
    let s = 0;
    for (let t = R.covStart[j]; t < R.covStart[j + 1]; t++) s += Math.max(0, Math.min(q, R.covB[t]) - Math.max(p, R.covA[t]));
    return s;
  };
  const isCov = (j: number, i: number) => covered(j, i, i + 1) > 0;
  for (let j = 0; j < R.h; j++) for (let r = R.rowStart[j]; r < R.rowStart[j + 1]; r++) {
    const a = R.runA[r], b = R.runB[r], m = R.mass[r];
    if (a > 0) { coast[m]++; if (isCov(j, a) && !isCov(j, a - 1)) cut[m]++; }
    if (b < R.w) { coast[m]++; if (isCov(j, b - 1) && !isCov(j, b)) cut[m]++; }
    for (const jj of [j - 1, j + 1]) {
      if (jj < 0 || jj >= R.h) continue;
      // the water of row jj under [a, b): the gaps between its land runs
      let p = a;
      const wet = (q: number) => {
        if (q <= p) return;
        coast[m] += q - p;
        // masked here, unmasked there: covered(j) minus covered by both
        for (let t = R.covStart[j]; t < R.covStart[j + 1]; t++) {
          const u = Math.max(p, R.covA[t]), v = Math.min(q, R.covB[t]);
          if (v > u) cut[m] += v - u - covered(jj, u, v);
        }
      };
      for (let s = R.rowStart[jj]; s < R.rowStart[jj + 1] && p < b; s++) {
        if (R.runB[s] <= p) continue;
        if (R.runA[s] >= b) break;
        wet(Math.min(b, R.runA[s]));
        p = Math.max(p, R.runB[s]);
      }
      wet(b);
    }
  }
  return { coast, cut };
}

/** the sheet's box reaches this far past its stations' box `diagM` metres across, metres */
export const sheetPadM = (diagM: number) => Math.max(8000, 0.08 * diagM);

/** the stations' box [w, s, e, n] (degrees) widened to the sheet's box, and a quarter more for the frame's scale: the
 * box whose zones' water masks the sheet reads, so no ground it reads is a guess where a zone has a mask (T-2591) */
export function sheetZoneBox([w, s, e, n]: [number, number, number, number]): [number, number, number, number] {
  const M = 111320, c = Math.cos((Math.min(80, Math.max(Math.abs(s), Math.abs(n))) * Math.PI) / 180);
  const pad = 1.25 * sheetPadM(Math.hypot((e - w) * M * Math.cos((((s + n) / 2) * Math.PI) / 180), (n - s) * M));
  return [w - pad / (M * c), s - pad / M, e + pad / (M * c), n + pad / M];
}

/** the land mass at a real place, its number plus 1; 0 on water or off the raster */
export function massAt(R: SheetRaster, x: number, y: number): number {
  const i = Math.floor((x - R.x0) / R.rr), j = Math.floor((y - R.y0) / R.rr);
  if (i < 0 || j < 0 || i >= R.w || j >= R.h) return 0;
  let lo = R.rowStart[j], hi = R.rowStart[j + 1] - 1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (R.runB[m] <= i) lo = m + 1; else if (R.runA[m] > i) hi = m - 1; else return R.mass[m] + 1; }
  return 0;
}

/** the sheet's drawn place at every node of its finest lattice inside the box ((nx + 1) by (ny + 1), from the mesh's
 * origin), linear in each quadtree cell's two triangles */
export function sheetFinest(s: Sheet): { gw: number; gh: number; qx: Float64Array; qy: Float64Array } {
  const { gx, nx, ny, key, qx, qy, cell } = s.mesh, GW = gx + 1, nc = cell.length / 3, W = nx + 1, H = ny + 1;
  const X = new Float64Array(W * H), Y = new Float64Array(W * H);
  for (let c = 0; c < nc; c++) {
    const i0 = cell[3 * c], j0 = cell[3 * c + 1], sz = cell[3 * c + 2];
    const a = findKey(key, j0 * GW + i0), b = findKey(key, j0 * GW + i0 + sz), cc = findKey(key, (j0 + sz) * GW + i0 + sz), d = findKey(key, (j0 + sz) * GW + i0);
    for (let j = j0; j <= Math.min(ny, j0 + sz); j++) for (let i = i0; i <= Math.min(nx, i0 + sz); i++) {
      const fi = (i - i0) / sz, fj = (j - j0) / sz, v = j * W + i;
      if (fi >= fj) (X[v] = (1 - fi) * qx[a] + (fi - fj) * qx[b] + fj * qx[cc]), (Y[v] = (1 - fi) * qy[a] + (fi - fj) * qy[b] + fj * qy[cc]);
      else (X[v] = (1 - fj) * qx[a] + fi * qx[cc] + (fj - fi) * qx[d]), (Y[v] = (1 - fj) * qy[a] + fi * qy[cc] + (fj - fi) * qy[d]);
    }
  }
  return { gw: W, gh: H, qx: X, qy: Y };
}

const RAD = Math.PI / 180, EARTH_R = 6371008.8;
/** the quadtree's roots are 2^K0 finest cells a side */
const K0 = 7;

function quantile(v: ArrayLike<number>, f: number): number {
  const s = Float64Array.from(v).sort();
  return s[Math.min(s.length - 1, Math.floor(f * s.length))];
}
const median = (v: ArrayLike<number>) => quantile(v, 0.5);

/** 2 by 2 blocks in compressed rows: row i's blocks at rs[i] .. rs[i + 1], columns `col` (sorted), values
 * val[4k .. 4k + 3] (xx, xy, yx, yy); `dg` the diagonal block's slot per row, `inv` its inverse */
interface Bsr { n: number; rs: Int32Array; col: Int32Array; val: Float64Array; dg: Int32Array; inv: Float64Array }

/** a Bsr's pattern from rows held in a flat buffer, `cap` slots a row, `len` used */
function bsrOf(n: number, buf: Int32Array, cap: number, len: Int32Array): Bsr {
  const rs = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) rs[i + 1] = rs[i] + len[i];
  const col = new Int32Array(rs[n]), dg = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const r = buf.subarray(i * cap, i * cap + len[i]).sort();
    col.set(r, rs[i]);
    for (let k = rs[i]; k < rs[i + 1]; k++) if (col[k] === i) dg[i] = k;
  }
  return { n, rs, col, val: new Float64Array(rs[n] * 4), dg, inv: new Float64Array(n * 4) };
}
function slotOf(A: Bsr, i: number, j: number): number {
  let lo = A.rs[i], hi = A.rs[i + 1] - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1, c = A.col[m];
    if (c === j) return m;
    if (c < j) lo = m + 1;
    else hi = m - 1;
  }
  return -1;
}
function mulBsr(A: Bsr, x: Float64Array, y: Float64Array) {
  const { n, rs, col, val } = A;
  for (let i = 0; i < n; i++) {
    let a = 0, b = 0;
    for (let k = rs[i], v = 4 * k; k < rs[i + 1]; k++, v += 4) {
      const j = 2 * col[k], xj = x[j], yj = x[j + 1];
      (a += val[v] * xj + val[v + 1] * yj), (b += val[v + 2] * xj + val[v + 3] * yj);
    }
    (y[2 * i] = a), (y[2 * i + 1] = b);
  }
}
/** the diagonal blocks' inverses, for the smoother; a row with nothing in it gets a tiny ridge */
function invDiag(A: Bsr) {
  for (let i = 0; i < A.n; i++) {
    const v = A.dg[i] * 4;
    let a = A.val[v], b = A.val[v + 1], c = A.val[v + 2], d = A.val[v + 3];
    if (Math.abs(a) + Math.abs(d) < 1e-300) (a = 1), (d = 1), (b = 0), (c = 0);
    const det = a * d - b * c, k = i * 4;
    (A.inv[k] = d / det), (A.inv[k + 1] = -b / det), (A.inv[k + 2] = -c / det), (A.inv[k + 3] = a / det);
  }
}
/** block Gauss-Seidel on A x = r, forward or backward */
function gs(A: Bsr, x: Float64Array, r: Float64Array, forward: boolean) {
  const n = A.n, rs = A.rs, col = A.col, val = A.val, dg = A.dg, inv = A.inv;
  const step = forward ? 1 : -1;
  for (let i = forward ? 0 : n - 1; forward ? i < n : i >= 0; i += step) {
    let a = r[2 * i], b = r[2 * i + 1];
    const di = dg[i], e = rs[i + 1];
    for (let k = rs[i]; k < e; k++) {
      if (k === di) continue;
      const j = 2 * col[k], v = 4 * k, xj = x[j], yj = x[j + 1];
      a -= val[v] * xj + val[v + 1] * yj;
      b -= val[v + 2] * xj + val[v + 3] * yj;
    }
    const q = i * 4;
    x[2 * i] = inv[q] * a + inv[q + 1] * b;
    x[2 * i + 1] = inv[q + 2] * a + inv[q + 3] * b;
  }
}

/** a prolongation from a level's free vertices to the next finer level's (scalar per vertex, the same for x and y):
 * rows `rs`, `col`, `w`, and the transpose `ts`, `tcol`, `tw` */
interface Prol { rs: Int32Array; col: Int32Array; w: Float64Array; ts: Int32Array; tcol: Int32Array; tw: Float64Array }
function prolOf(nf: number, nc: number, rs: Int32Array, col: Int32Array, w: Float64Array): Prol {
  const ts = new Int32Array(nc + 1);
  for (let k = 0; k < rs[nf]; k++) ts[col[k] + 1]++;
  for (let c = 0; c < nc; c++) ts[c + 1] += ts[c];
  const at = ts.slice(0, nc), tcol = new Int32Array(rs[nf]), tw = new Float64Array(rs[nf]);
  for (let i = 0; i < nf; i++) for (let k = rs[i]; k < rs[i + 1]; k++) { const p = at[col[k]]++; (tcol[p] = i), (tw[p] = w[k]); }
  return { rs, col, w, ts, tcol, tw };
}

/** one level of the quadtree: its cells (corner i, j and side, finest cells), vertices (keys i + j (gx + 1), sorted),
 * which are free, and each vertex as free vertices with weights (es, ei, ew; a hanging one as the middle of the edge
 * it hangs on) */
interface Level {
  ci: Int32Array; cj: Int32Array; cs: Int32Array;
  key: Int32Array; free: Int32Array; nF: number;
  es: Int32Array; ei: Int32Array; ew: Float64Array;
}

function findKey(keys: Int32Array, k: number): number {
  let lo = 0, hi = keys.length - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (keys[m] === k) return m;
    if (keys[m] < k) lo = m + 1;
    else hi = m - 1;
  }
  return -1;
}

// a triangle's rows over its corners' x, y (6-vectors, over the cell's side): uneven1 = Jxx - Jyy, uneven2 = Jxy + Jyx,
// even1 = (Jxx + Jyy) / 2, even2 = (Jyx - Jxy) / 2. Corners 0, 1, 2 are (a, b, c) in tri 0 and (a, c, d) in tri 1 of a
// cell a b c d (counter-clockwise from its low corner), Jx. = (x[p0] - x[p1], x[q0] - x[q1]) / side
const ROWS = [0, 1].map((tri) => {
  const [p0, p1, q0, q1] = tri === 0 ? [1, 0, 2, 1] : [1, 2, 2, 0];
  const r = Array.from({ length: 4 }, () => new Float64Array(6));
  const add = (row: number, corner: number, xy: number, v: number) => (r[row][corner * 2 + xy] += v);
  add(0, p0, 0, 1), add(0, p1, 0, -1), add(0, q0, 1, -1), add(0, q1, 1, 1);
  add(1, q0, 0, 1), add(1, q1, 0, -1), add(1, p0, 1, 1), add(1, p1, 1, -1);
  add(2, p0, 0, 0.5), add(2, p1, 0, -0.5), add(2, q0, 1, 0.5), add(2, q1, 1, -0.5);
  add(3, p0, 1, 0.5), add(3, p1, 1, -0.5), add(3, q0, 0, -0.5), add(3, q1, 0, 0.5);
  return r;
});
/** ROWS split into their parts along the cell's rows (the p pair) and its columns (the q pair), so a cell whose rest
 * shape is a rectangle kx by ky of its side (the cut, T-2609) has rows RP / kx + RQ / ky */
const [RP, RQ] = [0, 1].map((part) => [0, 1].map((tri) => {
  const [p0, p1, q0, q1] = tri === 0 ? [1, 0, 2, 1] : [1, 2, 2, 0];
  const r = Array.from({ length: 4 }, () => new Float64Array(6));
  const add = (row: number, corner: number, xy: number, v: number, along: number) => { if (along === part) r[row][corner * 2 + xy] += v; };
  add(0, p0, 0, 1, 0), add(0, p1, 0, -1, 0), add(0, q0, 1, -1, 1), add(0, q1, 1, 1, 1);
  add(1, q0, 0, 1, 1), add(1, q1, 0, -1, 1), add(1, p0, 1, 1, 0), add(1, p1, 1, -1, 0);
  add(2, p0, 0, 0.5, 0), add(2, p1, 0, -0.5, 0), add(2, q0, 1, 0.5, 1), add(2, q1, 1, -0.5, 1);
  add(3, p0, 1, 0.5, 0), add(3, p1, 1, -0.5, 0), add(3, q0, 0, -0.5, 1), add(3, q1, 0, 0.5, 1);
  return r;
}));
const gram = (tri: number, rows: number[]) => { const M = new Float64Array(36); for (const k of rows) for (let p = 0; p < 6; p++) for (let q = 0; q < 6; q++) M[p * 6 + q] += ROWS[tri][k][p] * ROWS[tri][k][q]; return M; };
const GA = [gram(0, [0, 1]), gram(1, [0, 1])], GS = [gram(0, [2, 3]), gram(1, [2, 3])];
/** a triangle's rows for rest shape (kx, ky) into `R` (4 rows of 6), and its Gram matrices of rows 0-1 and 2-3 */
function restRows(tri: number, kx: number, ky: number, R: Float64Array[], Ga: Float64Array, Gs: Float64Array) {
  const ix = 1 / kx, iy = 1 / ky;
  for (let k = 0; k < 4; k++) for (let p = 0; p < 6; p++) R[k][p] = RP[tri][k][p] * ix + RQ[tri][k][p] * iy;
  for (let p = 0; p < 6; p++) for (let q = 0; q < 6; q++) {
    Ga[p * 6 + q] = R[0][p] * R[0][q] + R[1][p] * R[1][q];
    Gs[p * 6 + q] = R[2][p] * R[2][q] + R[3][p] * R[3][q];
  }
}

/** The sheet of graph `g` on the ground `ground`, with options `o`. */
export function rubberSheet(g: DiagramGraph, ground: SheetGround, o: SheetOptions = SHEET): Sheet {
  const T0 = performance.now(), ms: Record<string, number> = {};
  let tl = T0;
  const tick = (k: string) => { const t = performance.now(); ms[k] = Math.round(t - tl); tl = t; };
  const n = g.key.length, PX = g.x, PY = g.y, nE = g.ea.length;

  // ---- the box: the stations' with a margin
  let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
  for (let i = 0; i < n; i++) (bx0 = Math.min(bx0, PX[i])), (bx1 = Math.max(bx1, PX[i])), (by0 = Math.min(by0, PY[i])), (by1 = Math.max(by1, PY[i]));
  if (!(n > 1)) (bx0 = by0 = -1000), (bx1 = by1 = 1000);
  const E = Math.hypot(bx1 - bx0, by1 - by0), pad = sheetPadM(E);
  (bx0 -= pad), (by0 -= pad), (bx1 += pad), (by1 += pad);
  const capM = 2000 * Math.pow(E / 4.5e6, 0.62) * 1000;

  // ---- what each station needs, pitches per metre: the dial's hop lengths (the median over its hops), or its room
  // beside its 4 nearest (strokes + gap pitches over their distance, the median, times `core`), the larger
  const strokes = new Uint16Array(n), deg = new Int32Array(n + 1), hopLen = new Float64Array(2 * nE), hopAt = new Int32Array(n + 1);
  const Tlen = (l: number) => 1000 * Math.pow(l / 1000, 1 - 0.7 * o.dial);
  const tg = new Float64Array(nE);
  {
    const seen = new Int32Array(Math.max(1, g.lines.length)).fill(-1);
    for (let e = 0; e < nE; e++) {
      const a = g.ea[e], b = g.eb[e];
      deg[a + 1]++, deg[b + 1]++;
      let st = 0;
      for (let k = g.first[e]; k < g.first[e + 1]; k++) if (seen[g.stroke[k]] !== e) (seen[g.stroke[k]] = e), st++;
      (strokes[a] = Math.max(strokes[a], st)), (strokes[b] = Math.max(strokes[b], st));
    }
    for (let i = 0; i < n; i++) deg[i + 1] += deg[i];
    hopAt.set(deg);
    for (let e = 0; e < nE; e++) {
      const a = g.ea[e], b = g.eb[e], l = Math.max(50, Math.hypot(PX[b] - PX[a], PY[b] - PY[a]));
      (hopLen[hopAt[a]++] = l), (hopLen[hopAt[b]++] = l), (tg[e] = Tlen(l));
    }
  }
  const medT = nE ? median(tg) : 1000, Pm = medT / 9;
  const sSt = new Float64Array(n), d4 = new Float64Array(n);
  {
    // stations in 2 km cells, by cell
    const cs = 2000, key = (i: number, j: number) => i * 100003 + j, order = new Int32Array(n), keys = new Float64Array(n);
    for (let i = 0; i < n; i++) (keys[i] = key(Math.floor(PX[i] / cs), Math.floor(PY[i] / cs))), (order[i] = i);
    order.sort((a, b) => keys[a] - keys[b]);
    const span = new Map<number, [number, number]>();
    for (let t = 0; t < n; ) { let u = t; while (u < n && keys[order[u]] === keys[order[t]]) u++; span.set(keys[order[t]], [t, u]); t = u; }
    const bd = new Float64Array(4), bj = new Int32Array(4), tmp = new Float64Array(64);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let k = deg[i]; k < deg[i + 1] && m < 64; k++) tmp[m++] = Tlen(hopLen[k]) / hopLen[k] / Pm;
      const sDial = m ? median(tmp.subarray(0, m)) : 9 / medT;
      const ci = Math.floor(PX[i] / cs), cj = Math.floor(PY[i] / cs);
      let cnt = 0;
      for (let r = 1; r < 400; r++) {
        for (let a = ci - r; a <= ci + r; a++) for (let b = cj - r; b <= cj + r; b++) {
          if (Math.max(Math.abs(a - ci), Math.abs(b - cj)) !== r && r > 1) continue;
          const sp = span.get(key(a, b));
          if (!sp) continue;
          for (let t = sp[0]; t < sp[1]; t++) {
            const j = order[t];
            if (j === i) continue;
            const d = Math.max(30, Math.sqrt((PX[j] - PX[i]) ** 2 + (PY[j] - PY[i]) ** 2));
            if (cnt === 4 && d >= bd[3]) continue;
            let p = cnt < 4 ? cnt++ : 3;
            while (p > 0 && bd[p - 1] > d) (bd[p] = bd[p - 1]), (bj[p] = bj[p - 1]), p--;
            (bd[p] = d), (bj[p] = j);
          }
        }
        if ((cnt >= 4 && (r - 1) * cs > bd[3]) || cnt >= n - 1) break;
      }
      d4[i] = cnt ? bd[cnt - 1] : 5000;
      for (let k = 0; k < cnt; k++) tmp[k] = ((strokes[i] + strokes[bj[k]]) / 2 + o.gap) / bd[k];
      const sRoom = cnt ? (E < 6e5 ? o.core : o.coreCountry) * median(tmp.subarray(0, cnt)) : 0;
      sSt[i] = Math.max(sDial, sRoom);
    }
  }
  const sFloor = Math.pow(0.25, o.dial) * (9 / medT) * (E < 6e5 ? 1 : o.floorCountry), lf = Math.log(sFloor);
  tick("needs");

  // ---- the finest lattice and the field on its nodes: the geometric mean of the stations' scales under Gaussians,
  // blended with the floor, then bounded (`bound`)
  const NXF = E < 6e5 ? o.nCity : o.nCountry;
  const h = Math.max(bx1 - bx0, by1 - by0) / (NXF - 1);
  // the field's smoothing and the reach of "near lines" at least 3/4 of a cell of T-2566's 256-node lattice, whatever
  // the finest cell: on a country map the finer lattice is for room in the cores, not a sharper field (T-2568)
  const hRef = o.refN > 1 ? Math.max(bx1 - bx0, by1 - by0) / (o.refN - 1) : 0;
  const NCX = Math.ceil((bx1 - bx0) / h), NCY = Math.ceil((by1 - by0) / h);
  const R0 = 1 << K0, NRX = Math.ceil(NCX / R0), NRY = Math.ceil(NCY / R0), GX = NRX * R0, GY = NRY * R0, GW = GX + 1, GH = GY + 1, GN = GW * GH;
  // the cut (T-2609): a column (row) of finest cells whose middle is farther than cutKm across from every network
  // station (one with cutN others within cutKm; a lonely stop on a long line is not one) has a rest width (height)
  // under a cell's side, 1 / k growing linearly past cutKm so the width halves every cutD drawn pitches at the floor's
  // scale (twice the field's own rate, T-2629), at least cutMin. Empty bands across the whole map are drawn narrow; a cell's rest
  // shape is its columns' by its rows'. Only on a map 600 km across or more: a city map's margins are its context
  // (Long Island's east end)
  const cutM = E < 6e5 ? 0 : o.cutKm * 1000, kxF = new Float64Array(GX).fill(1), kyF = new Float64Array(GY).fill(1), xs: number[] = [], ys: number[] = [], netSt = new Uint8Array(n);
  if (cutM > 0) {
    const H = new Map<number, number[]>(), key = (i: number, j: number) => i * 100003 + j;
    for (let s = 0; s < n; s++) { const k = key(Math.floor(PX[s] / cutM), Math.floor(PY[s] / cutM)); let l = H.get(k); if (!l) H.set(k, (l = [])); l.push(s); }
    for (let s = 0; s < n; s++) {
      let c = 0;
      const ci = Math.floor(PX[s] / cutM), cj = Math.floor(PY[s] / cutM);
      for (let a = -1; a <= 1 && c < o.cutN; a++) for (let b = -1; b <= 1 && c < o.cutN; b++) for (const t of H.get(key(ci + a, cj + b)) ?? []) if (t !== s && Math.hypot(PX[t] - PX[s], PY[t] - PY[s]) < cutM && ++c >= o.cutN) break;
      if (c >= o.cutN) xs.push(PX[s]), ys.push(PY[s]), (netSt[s] = 1);
    }
    const rate = (sFloor * Math.LN2) / o.cutD;
    const band = (v: number[], k: Float64Array, o0: number) => {
      if (!v.length) return;
      v.sort((a, b) => a - b);
      for (let i = 0, p = 0; i < k.length; i++) {
        const c = o0 + (i + 0.5) * h;
        while (p + 1 < v.length && v[p + 1] <= c) p++;
        const d = Math.min(Math.abs(c - v[p]), p + 1 < v.length ? Math.abs(v[p + 1] - c) : Infinity);
        k[i] = d <= cutM ? 1 : Math.max(o.cutMin, 1 / (1 + (d - cutM) * rate));
      }
    };
    band(xs.slice(), kxF, bx0), band(ys.slice(), kyF, by0);
  }
  /** prefix sums of the columns' and rows' rest sizes, so a cell's is its finest cells' mean */
  const kxS = new Float64Array(GX + 1), kyS = new Float64Array(GY + 1);
  for (let i = 0; i < GX; i++) kxS[i + 1] = kxS[i] + kxF[i];
  for (let j = 0; j < GY; j++) kyS[j + 1] = kyS[j] + kyF[j];
  // the floor per lattice node on a country map (T-2625): the best of the network stations' own scales fallen as
  // (1 + distance / farKm)^-farP, never above the floor, nor under farMin of it. A scale falling as r^-p round a network
  // is what a map can draw with its uneven stretch held to 1 / (1 - p) (the radial stretch r^(1-p)), 2:1 at p 0.5,
  // inside the fold guard; a fall from the floor itself, or faster, asks the ring round a network for more and the
  // sheet folds or never settles (notes/T-2625.md)
  const far = cutM > 0 && o.farP > 0 && xs.length > 0, LfN = new Float64Array(GN).fill(lf);
  if (far) {
    // each node carries its best station and that station's distance from neighbour to neighbour, as a chamfer distance
    const sL = new Float64Array(GN).fill(-Infinity), sD = new Float64Array(GN).fill(Infinity), r0 = o.farKm * 1000, hd2 = h * Math.SQRT2;
    const val = (L: number, d: number) => L - o.farP * Math.log(1 + d / r0);
    for (let s = 0; s < n; s++) {
      if (!netSt[s]) continue;
      const fi = (PX[s] - bx0) / h, fj = (PY[s] - by0) / h, ls = Math.log(sSt[s]);
      for (let j = Math.max(0, Math.floor(fj)); j <= Math.min(GY, Math.ceil(fj)); j++) for (let i = Math.max(0, Math.floor(fi)); i <= Math.min(GX, Math.ceil(fi)); i++) {
        const v = j * GW + i, d = Math.hypot(i - fi, j - fj) * h;
        if (val(ls, d) > val(sL[v], sD[v])) (sL[v] = ls), (sD[v] = d);
      }
    }
    for (let pass = 0; pass < 4; pass++) {
      const s = pass & 1 ? -1 : 1;
      for (let jj = 0; jj < GH; jj++) {
        const j = s > 0 ? jj : GH - 1 - jj;
        for (let ii = 0; ii < GW; ii++) {
          const i = s > 0 ? ii : GW - 1 - ii, v = j * GW + i;
          let bL = sL[v], bD = sD[v], bv = val(bL, bD);
          const take = (u: number, step: number) => { const c = val(sL[u], sD[u] + step); if (c > bv) (bv = c), (bL = sL[u]), (bD = sD[u] + step); };
          if (i - s >= 0 && i - s < GW) take(v - s, h);
          if (j - s >= 0 && j - s < GH) {
            const u = v - s * GW;
            take(u, h);
            if (i - s >= 0 && i - s < GW) take(u - s, hd2);
            if (i + s >= 0 && i + s < GW) take(u + s, hd2);
          }
          (sL[v] = bL), (sD[v] = bD);
        }
      }
    }
    const lmin = lf + Math.log(o.farMin);
    for (let v = 0; v < GN; v++) LfN[v] = Math.max(lmin, Math.min(lf, val(sL[v], sD[v])));
  }
  const floorAt = (x: number, y: number) => LfN[Math.min(GY, Math.max(0, Math.round((y - by0) / h))) * GW + Math.min(GX, Math.max(0, Math.round((x - bx0) / h)))];
  const Lf = new Float64Array(GN);
  {
    const Lw = new Float64Array(GN), Ws = new Float64Array(GN), gx = new Float64Array(GW), gy = new Float64Array(GH);
    for (let s = 0; s < n; s++) {
      const R = Math.max(1500, d4[s], 0.75 * h, 0.75 * hRef), ls = far && !netSt[s] ? Math.min(Math.log(sSt[s]), floorAt(PX[s], PY[s])) : Math.log(sSt[s]), k2 = 1 / (2 * R * R);
      const i0 = Math.max(0, Math.floor((PX[s] - 3 * R - bx0) / h)), i1 = Math.min(GX, Math.ceil((PX[s] + 3 * R - bx0) / h));
      const j0 = Math.max(0, Math.floor((PY[s] - 3 * R - by0) / h)), j1 = Math.min(GY, Math.ceil((PY[s] + 3 * R - by0) / h));
      for (let i = i0; i <= i1; i++) { const dx = bx0 + i * h - PX[s]; gx[i] = Math.exp(-dx * dx * k2); }
      for (let j = j0; j <= j1; j++) { const dy = by0 + j * h - PY[s]; gy[j] = Math.exp(-dy * dy * k2); }
      for (let j = j0; j <= j1; j++) for (let i = i0, v = j * GW + i0; i <= i1; i++, v++) { const w = gx[i] * gy[j]; (Lw[v] += w * ls), (Ws[v] += w); }
    }
    const W0 = 0.05;
    for (let v = 0; v < GN; v++) Lf[v] = Math.max(LfN[v], (Lw[v] + W0 * LfN[v]) / (Ws[v] + W0));
  }
  // between neighbouring nodes the field may fall at most to double every D drawn pitches and maxStep a cell: sweeps
  // forward and back raising the low side, as T-2566's prototype, each node looked at again only once a neighbour moved
  const dirty = new Uint8Array(GN);
  const bound = (F: Float64Array) => {
    const c = Math.LN2 / o.D, lstep = Math.log(o.maxStep), cs = c * h, cd = c * h * Math.SQRT2, ld = lstep * Math.SQRT2;
    for (let sweep = 0; sweep < 24; sweep++) {
      const fwd = !(sweep & 1);
      let any = 0;
      for (let jj = 0; jj < GH; jj++) {
        const j = fwd ? jj : GH - 1 - jj;
        for (let ii = 0; ii < GW; ii++) {
          const i = fwd ? ii : GW - 1 - ii, v = j * GW + i;
          if (!dirty[v]) continue;
          dirty[v] = 0;
          let f = F[v], moved = false;
          for (let q = 0; q < 8; q++) {
            const di = q === 0 || q === 2 || q === 3 ? 1 : q === 4 || q === 6 || q === 7 ? -1 : 0;
            const dj = q === 1 || q === 2 || q === 7 ? 1 : q === 3 || q === 5 || q === 6 ? -1 : 0;
            const a = i + di, b = j + dj;
            if (a < 0 || b < 0 || a >= GW || b >= GH) continue;
            const fu = F[b * GW + a];
            if (fu <= f) continue;
            const diag = di !== 0 && dj !== 0;
            const need = Math.max(fu - (diag ? cd : cs) * Math.exp((fu + f) / 2), fu - (diag ? ld : lstep));
            if (need > f + 1e-9) (f = need), (moved = true);
          }
          if (!moved) continue;
          F[v] = f;
          any++;
          for (let b = Math.max(0, j - 1); b <= Math.min(GY, j + 1); b++) for (let a = Math.max(0, i - 1); a <= Math.min(GX, i + 1); a++) dirty[b * GW + a] = 1;
          dirty[v] = 0;
        }
      }
      if (!any) break;
    }
  };
  dirty.fill(1);
  bound(Lf);
  tick("field");

  // ---- the ground, as T-2566's prototype read it: a raster of `rr` metres, water where a mask's pixel under a cell's
  // middle is (on a country map only mask water whose run the ground calls sea or a strait), else the ground's sea
  // (once per 2 km block). Read row by row as land runs, each joined to the runs it touches in the row before, so
  // land masses come out of a union-find over runs; per finest cell its land share
  const [fx, fy, fk] = ground.frame;
  const mk = ground.masks.map((m) => { const s = EARTH_R * RAD * Math.cos(m.lat * RAD); return { m, B: (s * 360) / (fk * m.cell), A: (s * (360 * fx - 180 - m.lon) - m.x0) / m.cell }; });
  const latOf = (y: number) => (2 * Math.atan(Math.exp(Math.PI * (1 - 2 * (fy - y / fk)))) - Math.PI / 2) / RAD;
  const sea = ground.sea;
  const seaAt = (x: number, y: number) => (sea ? sea.isSea(x, y) || sea.massAt(x, y) < 0 : false);
  const city = E < 6e5, rr = city ? o.cityRaster : Math.max(200, E / 6000, h / 2);
  const RW = Math.ceil((bx1 - bx0) / rr), RH = Math.ceil((by1 - by0) / rr);
  // a row's spans of raster cells as packed keys start * 2^16 + end (RW is under 2^16), sorted and merged
  let wK = new Float64Array(1024), nW = 0, cK = new Float64Array(64), nC = 0;
  const pushW = (a: number, b: number) => { if (nW >= wK.length) { const t = new Float64Array(wK.length * 2); t.set(wK); wK = t; } wK[nW++] = a * 65536 + b; };
  const pushC = (a: number, b: number) => { if (nC >= cK.length) { const t = new Float64Array(cK.length * 2); t.set(cK); cK = t; } cK[nC++] = a * 65536 + b; };
  /** the water spans (wK) and covered spans (cK) of the raster row at graph y, from each mask: its pixel under each
   * cell's middle */
  const maskRow = (y: number) => {
    const lat = latOf(y);
    (nW = 0), (nC = 0);
    for (const { m, A, B } of mk) {
      const r = Math.floor((EARTH_R * (lat - m.lat) * RAD - m.y0) / m.cell);
      if (r < 0 || r >= m.h) continue;
      const cA = A + B * (bx0 + 0.5 * rr), cS = B * rr;
      const c0 = Math.max(0, Math.ceil(-cA / cS)), c1 = Math.min(RW, Math.ceil((m.w - cA) / cS));
      if (c1 > c0) pushC(c0, c1);
      const lo = m.rows[r], hi = m.rows[r + 1];
      if (hi <= lo) continue;
      let a = lo / 2, b = hi / 2;
      while (a < b) { const mid = (a + b) >> 1; if (m.runs[mid * 2 + 1] <= cA) a = mid + 1; else b = mid; }
      const cEnd = cA + cS * RW;
      for (let p = a; p < hi / 2; p++) {
        const s = m.runs[p * 2], e = m.runs[p * 2 + 1];
        if (s >= cEnd) break;
        const i0 = Math.max(0, Math.ceil((s - cA) / cS)), i1 = Math.min(RW, Math.ceil((e - cA) / cS));
        if (i1 <= i0) continue;
        if (!city && !seaAt(bx0 + ((i0 + i1) / 2) * rr, y)) continue;
        pushW(i0, i1);
      }
    }
  };
  /** spans packed in K[0..k) sorted and merged where they touch or overlap, in place; their count */
  const merge = (K: Float64Array, k: number) => {
    if (k < 2) return k;
    K.subarray(0, k).sort();
    let m = 0, a = Math.floor(K[0] / 65536), b = K[0] - a * 65536;
    for (let t = 1; t < k; t++) {
      const s = Math.floor(K[t] / 65536), e = K[t] - s * 65536;
      if (s <= b) { if (e > b) b = e; continue; }
      K[m++] = a * 65536 + b;
      (a = s), (b = e);
    }
    K[m++] = a * 65536 + b;
    return m;
  };
  // the raster columns (rows) whose middles fall in each finest column (row); those past the last go to the last
  const colAt = new Int32Array(NCX + 1), rowAt = new Int32Array(NCY + 1);
  for (let c = 0; c < NCX; c++) colAt[c] = Math.min(RW, Math.max(0, Math.ceil((c * h) / rr - 0.5)));
  for (let c = 0; c < NCY; c++) rowAt[c] = Math.min(RH, Math.max(0, Math.ceil((c * h) / rr - 0.5)));
  (colAt[NCX] = RW), (rowAt[NCY] = RH);
  const fineCol = (i: number) => Math.min(NCX - 1, Math.floor(((i + 0.5) * rr) / h));
  const NFC = GX * GY, landC = new Float32Array(NFC), ilC = new Float32Array(NFC);
  let runA = new Int32Array(1 << 16), runB = new Int32Array(1 << 16), par = new Int32Array(1 << 16), nR = 0;
  let covA = new Int32Array(1024), covB = new Int32Array(1024), nCov = 0;
  const rowStart = new Int32Array(RH + 1), covStart = new Int32Array(RH + 1);
  {
    const BS = Math.max(1, Math.round(2000 / rr)), BWr = Math.floor(RW / BS) + 1, blk = new Int8Array(BWr * (Math.floor(RH / BS) + 1)).fill(-1);
    const find = (v: number) => { while (par[v] !== v) (par[v] = par[par[v]]), (v = par[v]); return v; };
    for (let j = 0; j < RH; j++) {
      const y = by0 + (j + 0.5) * rr, cj = Math.min(NCY - 1, Math.floor(((j + 0.5) * rr) / h)), bj = Math.floor(j / BS) * BWr, yb = by0 + (Math.floor(j / BS) + 0.5) * BS * rr, base = cj * GX;
      maskRow(y);
      nC = merge(cK, nC);
      covStart[j] = nCov;
      if (nCov + nC > covA.length) {
        const g2 = (v: Int32Array) => { const w = new Int32Array(Math.max(v.length * 2, nCov + nC)); w.set(v); return w; };
        (covA = g2(covA)), (covB = g2(covB));
      }
      for (let t = 0; t < nC; t++) { const s = Math.floor(cK[t] / 65536); (covA[nCov] = s), (covB[nCov++] = cK[t] - s * 65536); }
      // where no mask covers: the ground's answer, a block at a time
      const open = (a: number, b: number) => {
        for (let bi = Math.floor(a / BS); bi * BS < b; bi++) {
          const k = bj + bi;
          if (blk[k] < 0) blk[k] = +seaAt(bx0 + (bi + 0.5) * BS * rr, yb);
          if (blk[k]) pushW(Math.max(a, bi * BS), Math.min(b, (bi + 1) * BS));
        }
      };
      let prev = 0;
      for (let t = 0; t < nC; t++) { const s = Math.floor(cK[t] / 65536), e = cK[t] - s * 65536; if (s > prev) open(prev, s); if (e > prev) prev = e; }
      if (prev < RW) open(prev, RW);
      nW = merge(wK, nW);
      rowStart[j] = nR;
      const p1 = nR;
      let p = j ? rowStart[j - 1] : 0;
      // the land runs: the gaps between the water spans
      prev = 0;
      for (let t = 0; t <= nW; t++) {
        const s = t < nW ? Math.floor(wK[t] / 65536) : RW, e = t < nW ? wK[t] - s * 65536 : RW;
        if (s > prev) {
          const a = prev, b = s;
          if (nR >= runA.length) {
            const g2 = (v: Int32Array) => { const w = new Int32Array(v.length * 2); w.set(v); return w; };
            (runA = g2(runA)), (runB = g2(runB)), (par = g2(par));
          }
          (runA[nR] = a), (runB[nR] = b), (par[nR] = nR);
          while (p < p1 && runB[p] <= a) p++;
          for (let q = p; q < p1 && runA[q] < b; q++) { const ra = find(q), rb = find(nR); if (ra !== rb) par[ra] = rb; }
          nR++;
          for (let ci = fineCol(a), c1 = fineCol(b - 1); ci <= c1; ci++) { const lo = colAt[ci], hi = colAt[ci + 1]; landC[base + ci] += (b < hi ? b : hi) - (a > lo ? a : lo); }
        }
        if (e > prev) prev = e;
      }
    }
    (rowStart[RH] = nR), (covStart[RH] = nCov);
    for (let r = 0; r < nR; r++) par[r] = find(r);
    for (let cj = 0; cj < NCY; cj++) for (let ci = 0; ci < NCX; ci++) {
      const all = (colAt[ci + 1] - colAt[ci]) * (rowAt[cj + 1] - rowAt[cj]);
      landC[cj * GX + ci] = all ? landC[cj * GX + ci] / all : 0;
    }
  }
  /** the land run under raster cell (i, j), -1 water */
  const runAt = (i: number, j: number) => {
    if (i < 0 || j < 0 || i >= RW || j >= RH) return -1;
    let lo = rowStart[j], hi = rowStart[j + 1] - 1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (runB[m] <= i) lo = m + 1; else if (runA[m] > i) hi = m - 1; else return m; }
    return -1;
  };
  // land masses of at most islandKm2 with a station on them (its real place, or the nearest land within 1 km where
  // that is water: piers, the harbour's edge)
  tick("raster");
  const islands: { i0: number; j0: number; i1: number; j1: number }[] = [];
  const islandOf = new Int32Array(nR);
  if (o.islandKm2 > 0) {
    const area = new Float64Array(nR);
    for (let r = 0; r < nR; r++) area[par[r]] += runB[r] - runA[r];
    const limit = (o.islandKm2 * 1e6) / (rr * rr);
    for (let s = 0; s < n; s++) {
      let r = runAt(Math.floor((PX[s] - bx0) / rr), Math.floor((PY[s] - by0) / rr));
      for (let d = rr; r < 0 && d <= 1000; d += rr) for (let a = 0; a < 24 && r < 0; a++) r = runAt(Math.floor((PX[s] + d * Math.cos((a * Math.PI) / 12) - bx0) / rr), Math.floor((PY[s] + d * Math.sin((a * Math.PI) / 12) - by0) / rr));
      if (r < 0) continue;
      const root = par[r];
      if (islandOf[root] || area[root] > limit) continue;
      islands.push({ i0: RW, j0: RH, i1: 0, j1: 0 });
      islandOf[root] = islands.length;
    }
    for (let j = 0; j < RH && islands.length; j++) {
      const cj = Math.min(NCY - 1, Math.floor(((j + 0.5) * rr) / h));
      for (let r = rowStart[j]; r < rowStart[j + 1]; r++) {
        const id = islandOf[par[r]];
        if (!id) continue;
        const isl = islands[id - 1], a = runA[r], b = runB[r];
        (isl.i0 = Math.min(isl.i0, a)), (isl.i1 = Math.max(isl.i1, b - 1)), (isl.j0 = Math.min(isl.j0, j)), (isl.j1 = Math.max(isl.j1, j));
        for (let ci = fineCol(a), c1 = fineCol(b - 1), base = cj * GX; ci <= c1; ci++) { const lo = colAt[ci], hi = colAt[ci + 1]; ilC[base + ci] += (b < hi ? b : hi) - (a > lo ? a : lo); }
      }
    }
  }
  tick("ground");

  // ---- per finest cell: near lines (within nearKm of a station or a track's point, cell middle to the point's cell
  // middle, plus 0.71 of a cell), its stations, its kind (0 land near lines, 1 land with none, 2 water, and every cell
  // outside the box), whether mostly on an island, and its weights
  const nearC = new Uint8Array(NFC), cntC = new Float32Array(NFC), islC = new Uint8Array(NFC), kindC = new Uint8Array(NFC).fill(2);
  {
    const R = Math.max(o.nearKm * 1000, 0.75 * h, 0.75 * hRef), rc = Math.ceil(R / h), lim = (R + 0.71 * h) / h, disk: number[] = [];
    for (let b = -rc; b <= rc; b++) for (let a = -rc; a <= rc; a++) if (Math.hypot(a, b) <= lim) disk.push(a, b);
    const seed = new Uint8Array(NFC), seeds: number[] = [];
    const put = (x: number, y: number) => {
      const ci = Math.floor((x - bx0) / h), cj = Math.floor((y - by0) / h);
      if (ci < 0 || cj < 0 || ci >= NCX || cj >= NCY || seed[cj * GX + ci]) return;
      seed[cj * GX + ci] = 1, seeds.push(ci, cj);
    };
    for (let s = 0; s < n; s++) put(PX[s], PY[s]);
    const step = Math.min(R, h) / 2;
    for (let e = 0; e < nE; e++) {
      const hp = g.hop, ha = g.hopAt;
      const pts: ArrayLike<number> = hp && ha && ha[e + 1] - ha[e] >= 2 ? hp.subarray(ha[e] * 2, ha[e + 1] * 2) : [PX[g.ea[e]], PY[g.ea[e]], PX[g.eb[e]], PY[g.eb[e]]];
      for (let k = 0; k + 3 < pts.length; k += 2) {
        const m = Math.ceil(Math.hypot(pts[k + 2] - pts[k], pts[k + 3] - pts[k + 1]) / step);
        for (let t = 0; t <= m; t++) put(pts[k] + ((pts[k + 2] - pts[k]) * t) / m, pts[k + 1] + ((pts[k + 3] - pts[k + 1]) * t) / m);
      }
    }
    for (let k = 0; k < seeds.length; k += 2) {
      const ci = seeds[k], cj = seeds[k + 1];
      for (let q = 0; q < disk.length; q += 2) {
        const a = ci + disk[q], b = cj + disk[q + 1];
        if (a >= 0 && b >= 0 && a < NCX && b < NCY) nearC[b * GX + a] = 1;
      }
    }
    for (let s = 0; s < n; s++) cntC[Math.min(NCY - 1, Math.floor((PY[s] - by0) / h)) * GX + Math.min(NCX - 1, Math.floor((PX[s] - bx0) / h))]++;
    for (let cj = 0; cj < NCY; cj++) for (let ci = 0; ci < NCX; ci++) {
      const c = cj * GX + ci;
      kindC[c] = landC[c] >= o.landShare ? (nearC[c] ? 0 : 1) : 2;
      if (o.aIsland && ilC[c] && landC[c] >= o.landShare && ilC[c] >= 0.5 * landC[c] * (colAt[ci + 1] - colAt[ci]) * (rowAt[cj + 1] - rowAt[cj])) islC[c] = 1;
    }
  }
  // on a country map, land farther than freeKm from every network station (the cut's) is as free to stretch unevenly as
  // water, even near a lonely line or inside a zone with stations elsewhere (T-2609: the Los Angeles zone's empty east)
  const free = new Uint8Array(NFC), freeM = o.freeKm * 1000;
  if (cutM > 0 && freeM > 0) {
    const dd = new Float32Array(NFC).fill(Infinity), hd2 = h * Math.SQRT2;
    for (let k = 0; k < xs.length; k++) {
      const i = Math.floor((xs[k] - bx0) / h), j = Math.floor((ys[k] - by0) / h);
      if (i >= 0 && j >= 0 && i < GX && j < GY) dd[j * GX + i] = 0;
    }
    for (let pass = 0; pass < 2; pass++) {
      const s = pass ? -1 : 1;
      for (let jj = 0; jj < GY; jj++) {
        const j = s > 0 ? jj : GY - 1 - jj;
        for (let ii = 0; ii < GX; ii++) {
          const i = s > 0 ? ii : GX - 1 - ii, v = j * GX + i;
          let d = dd[v];
          if (i - s >= 0 && i - s < GX) d = Math.min(d, dd[v - s] + h);
          if (j - s >= 0 && j - s < GY) {
            const u = v - s * GX;
            d = Math.min(d, dd[u] + h);
            if (i - s >= 0 && i - s < GX) d = Math.min(d, dd[u - s] + hd2);
            if (i + s >= 0 && i + s < GX) d = Math.min(d, dd[u + s] + hd2);
          }
          dd[v] = d;
        }
      }
    }
    for (let c = 0; c < NFC; c++) free[c] = +(dd[c] > freeM);
  }
  const aC = new Float32Array(NFC), sC = new Float32Array(NFC);
  {
    const nz: number[] = [];
    for (let c = 0; c < NFC; c++) if (cntC[c] > 0) nz.push(cntC[c]);
    const sLine = far ? o.sLine * o.farS : o.sLine;
    const cm = nz.length ? median(nz) : 1, AW = [o.aLine, o.aLand, o.aWater], SW = [o.sLine, o.sLand, o.sWater];
    for (let c = 0; c < NFC; c++) {
      const k = kindC[c];
      (aC[c] = islC[c] ? o.aIsland : free[c] && k !== 2 ? o.aWater : AW[k]), (sC[c] = k === 0 ? sLine * Math.min(o.sMax, Math.max(0.3, cntC[c] / cm)) : SW[k]);
    }
  }
  // islands keep one size: the nodes on each, and the ring a cell round it raised to it
  const islNodes: Int32Array[] = [];
  for (let k = 0; k < islands.length; k++) {
    const isl = islands[k], id = k + 1, vs: number[] = [];
    const ni0 = Math.max(0, Math.floor((isl.i0 * rr) / h)), ni1 = Math.min(GX, Math.ceil(((isl.i1 + 1) * rr) / h));
    const nj0 = Math.max(0, Math.floor((isl.j0 * rr) / h)), nj1 = Math.min(GY, Math.ceil(((isl.j1 + 1) * rr) / h));
    for (let j = nj0; j <= nj1; j++) for (let i = ni0; i <= ni1; i++) {
      const r = runAt(Math.floor((i * h) / rr), Math.floor((j * h) / rr));
      if (r >= 0 && islandOf[par[r]] === id) vs.push(j * GW + i);
    }
    islNodes.push(Int32Array.from(vs));
  }
  const ownNode = new Uint8Array(GN);
  const uniformIslands = (F: Float64Array, boost: boolean) => {
    for (const vs of islNodes) {
      if (!vs.length) continue;
      const vals = new Float64Array(vs.length);
      for (let a = 0; a < vs.length; a++) vals[a] = F[vs[a]];
      const t = quantile(vals, o.islandQ) + (boost ? Math.log(o.islandBoost) : 0);
      for (const v of vs) (F[v] = t), (ownNode[v] = 1), (dirty[v] = 1);
      for (const v of vs) {
        const i = v % GW, j = (v / GW) | 0;
        for (let b = Math.max(0, j - 1); b <= Math.min(GY, j + 1); b++) for (let a = Math.max(0, i - 1); a <= Math.min(GX, i + 1); a++) {
          const u = b * GW + a;
          if (!ownNode[u] && F[u] < t) (F[u] = t), (dirty[u] = 1);
        }
      }
      for (const v of vs) ownNode[v] = 0;
    }
    // what moved, and its neighbours, looked at again
    for (let v = 0; v < GN; v++) if (dirty[v] === 1) dirty[v] = 2;
    for (let v = 0; v < GN; v++) if (dirty[v] === 2) { const i = v % GW, j = (v / GW) | 0; for (let b = Math.max(0, j - 1); b <= Math.min(GY, j + 1); b++) for (let a = Math.max(0, i - 1); a <= Math.min(GX, i + 1); a++) if (!dirty[b * GW + a]) dirty[b * GW + a] = 1; }
    for (let v = 0; v < GN; v++) if (dirty[v]) dirty[v] = 1;
    bound(F);
  };
  if (islNodes.length) uniformIslands(Lf, true);
  tick("cells");

  // ---- the quadtree: a pyramid of the finest cells' facts, cells split from the roots down, then 2:1 across edges
  const IN = 1, K0B = 2, K1B = 4, K2B = 8, NEAR = 16, ISL = 32;
  const pyr: { w: number; fmin: Float32Array; fmax: Float32Array; bits: Uint8Array }[] = [];
  {
    const fmin = new Float32Array(NFC), fmax = new Float32Array(NFC), bits = new Uint8Array(NFC);
    for (let j = 0; j < GY; j++) for (let i = 0; i < GX; i++) {
      const c = j * GX + i, v = j * GW + i, a = Lf[v], b = Lf[v + 1], cc = Lf[v + GW], d = Lf[v + GW + 1];
      let lo = a < b ? a : b, hi = a < b ? b : a;
      if (cc < lo) lo = cc;
      if (cc > hi) hi = cc;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
      (fmin[c] = lo), (fmax[c] = hi);
      if (i < NCX && j < NCY) bits[c] = IN | (kindC[c] === 0 ? K0B : kindC[c] === 1 ? K1B : K2B) | (nearC[c] ? NEAR : 0) | (ilC[c] ? ISL : 0);
    }
    pyr[K0] = { w: GX, fmin, fmax, bits };
    for (let d = K0 - 1; d >= 0; d--) {
      const F = pyr[d + 1], w = F.w >> 1, hh = GY >> (K0 - d), m = w * hh;
      const L = { w, fmin: new Float32Array(m), fmax: new Float32Array(m), bits: new Uint8Array(m) };
      for (let j = 0; j < hh; j++) for (let i = 0; i < w; i++) {
        const c = j * w + i, a = 2 * j * F.w + 2 * i, b = a + 1, cc = a + F.w, dd = cc + 1;
        let lo = F.fmin[a], hi = F.fmax[a];
        for (const u of [b, cc, dd]) (lo = F.fmin[u] < lo ? F.fmin[u] : lo), (hi = F.fmax[u] > hi ? F.fmax[u] : hi);
        (L.fmin[c] = lo), (L.fmax[c] = hi), (L.bits[c] = F.bits[a] | F.bits[b] | F.bits[cc] | F.bits[dd]);
      }
      pyr[d] = L;
    }
  }
  const lflat = Math.log(E < 6e5 ? o.flat : o.flatCountry);
  // leaves: corner and side (finest cells); leafOf per finest cell (-1 none)
  const leafOf = new Int32Array(NFC).fill(-1);
  let LI = new Int32Array(4096), LJ = new Int32Array(4096), LS = new Int32Array(4096), nLeaf = 0;
  const addLeaf = (i0: number, j0: number, s: number) => {
    if (nLeaf >= LI.length) {
      const g2 = (v: Int32Array) => { const w = new Int32Array(v.length * 2); w.set(v); return w; };
      (LI = g2(LI)), (LJ = g2(LJ)), (LS = g2(LS));
    }
    (LI[nLeaf] = i0), (LJ[nLeaf] = j0), (LS[nLeaf] = s);
    for (let j = j0; j < j0 + s; j++) leafOf.fill(nLeaf, j * GX + i0, j * GX + i0 + s);
    return nLeaf++;
  };
  const inBox = (i0: number, j0: number, s: number) => { const d = K0 - Math.log2(s), L = pyr[d]; return (L.bits[(j0 / s) * L.w + i0 / s] & IN) !== 0; };
  {
    const split = (d: number, ci: number, cj: number) => {
      const L = pyr[d], c = cj * L.w + ci, b = L.bits[c];
      if (!(b & IN)) return;
      const side = 1 << (K0 - d);
      const go = d < K0 && (o.uniform || side > o.coarse || (o.nearMax > 0 && !!(b & NEAR) && side > o.nearMax) || !!(b & ISL) || (!!(b & NEAR) && !!(b & K2B) && !!(b & (K0B | K1B)) && side > (E < 6e5 ? o.coastCity : o.coastCountry)) ||
        (o.drift > 0 ? (Math.exp(L.fmax[c]) - Math.exp(L.fmin[c])) * side * h > o.drift : L.fmax[c] - L.fmin[c] > lflat && Math.exp(L.fmax[c]) * side * h > o.drawn));
      if (!go) return void addLeaf(ci * side, cj * side, side);
      for (let q = 0; q < 4; q++) split(d + 1, 2 * ci + (q & 1), 2 * cj + (q >> 1));
    };
    for (let rj = 0; rj < NRY; rj++) for (let ri = 0; ri < NRX; ri++) split(0, ri, rj);
    // 2:1: a leaf with an edge neighbour under half its side splits; its neighbours and children are looked at again
    const dead = new Uint8Array(1 << 20);
    const queue: number[] = [];
    for (let id = 0; id < nLeaf; id++) queue.push(id);
    const around = (id: number, fn: (nb: number) => void) => {
      const i0 = LI[id], j0 = LJ[id], s = LS[id];
      let last = -1;
      for (let t = 0; t < s; t++) for (const [a, b] of [[i0 - 1, j0 + t], [i0 + s, j0 + t], [i0 + t, j0 - 1], [i0 + t, j0 + s]]) {
        if (a < 0 || b < 0 || a >= GX || b >= GY) continue;
        const nb = leafOf[b * GX + a];
        if (nb >= 0 && nb !== last) (last = nb), fn(nb);
      }
    };
    while (queue.length) {
      const id = queue.pop()!;
      if (dead[id] || LS[id] < 4) continue;
      let need = false;
      around(id, (nb) => { if (2 * LS[nb] < LS[id]) need = true; });
      if (!need) continue;
      dead[id] = 1;
      const i0 = LI[id], j0 = LJ[id], hs = LS[id] >> 1;
      for (let j = j0; j < j0 + 2 * hs; j++) leafOf.fill(-1, j * GX + i0, j * GX + i0 + 2 * hs);
      for (let q = 0; q < 4; q++) {
        const a = i0 + (q & 1) * hs, b = j0 + (q >> 1) * hs;
        if (inBox(a, b, hs)) queue.push(addLeaf(a, b, hs));
      }
      around(id, (nb) => queue.push(nb));
    }
    // the live leaves, in order
    let m = 0;
    for (let id = 0; id < nLeaf; id++) if (!dead[id]) (LI[m] = LI[id]), (LJ[m] = LJ[id]), (LS[m] = LS[id]), m++;
    nLeaf = m;
    leafOf.fill(-1);
    for (let id = 0; id < nLeaf; id++) for (let j = LJ[id]; j < LJ[id] + LS[id]; j++) leafOf.fill(id, j * GX + LI[id], j * GX + LI[id] + LS[id]);
  }
  tick("quadtree");

  // ---- the levels: the leaves cut at each depth (a 2:1 quadtree stays 2:1 when cut), and the prolongations between
  const stamp = new Int32Array(GN).fill(-1);
  const buildLevel = (d: number): Level => {
    const side = 1 << (K0 - d);
    let ci = new Int32Array(nLeaf), cj = new Int32Array(nLeaf), cs = new Int32Array(nLeaf), nc = 0;
    for (let id = 0; id < nLeaf; id++) {
      if (LS[id] >= side) { (ci[nc] = LI[id]), (cj[nc] = LJ[id]), (cs[nc] = LS[id]), nc++; continue; }
      const a = LI[id] - (LI[id] % side), b = LJ[id] - (LJ[id] % side), k = b * GW + a;
      // a leaf's ancestor is its first leaf's, by the leaves' order within each cut cell, so stamp it
      if (stamp[k] === -2) continue;
      (stamp[k] = -2), (ci[nc] = a), (cj[nc] = b), (cs[nc] = side), nc++;
    }
    for (let c = 0; c < nc; c++) if (cs[c] === side && stamp[cj[c] * GW + ci[c]] === -2) stamp[cj[c] * GW + ci[c]] = -1;
    (ci = ci.slice(0, nc)), (cj = cj.slice(0, nc)), (cs = cs.slice(0, nc));
    const keys = new Int32Array(nc * 4);
    let nk = 0;
    for (let c = 0; c < nc; c++) {
      const a = cj[c] * GW + ci[c], s = cs[c];
      for (const k of [a, a + s, a + s * GW, a + s * GW + s]) if (stamp[k] === -1) (stamp[k] = -3), (keys[nk++] = k);
    }
    const key = keys.slice(0, nk).sort(), nv = nk;
    for (let v = 0; v < nv; v++) stamp[key[v]] = v;
    const m1 = new Int32Array(nv).fill(-1), m2 = new Int32Array(nv).fill(-1);
    for (let c = 0; c < nc; c++) {
      const s = cs[c];
      if (s < 2) continue;
      const a = cj[c] * GW + ci[c], hs = s >> 1;
      const mids = [a + hs, a, a + s, a + s * GW + hs, a + s * GW, a + s * GW + s, a + hs * GW, a, a + s * GW, a + hs * GW + s, a + s, a + s * GW + s];
      for (let q = 0; q < 12; q += 3) { const v = stamp[mids[q]]; if (v >= 0) (m1[v] = stamp[mids[q + 1]]), (m2[v] = stamp[mids[q + 2]]); }
    }
    const free = new Int32Array(nv).fill(-1);
    let nF = 0;
    for (let v = 0; v < nv; v++) if (m1[v] < 0) free[v] = nF++;
    // each vertex as free vertices: a hanging one through its edge's ends, which may hang in turn
    const exp: (number[] | null)[] = new Array(nv).fill(null);
    const expOf = (v: number): number[] => {
      let e = exp[v];
      if (e) return e;
      if (m1[v] < 0) e = [free[v], 1];
      else {
        e = [];
        for (const u of [m1[v], m2[v]]) {
          const eu = expOf(u);
          for (let t = 0; t < eu.length; t += 2) {
            let k = 0;
            while (k < e.length && e[k] !== eu[t]) k += 2;
            if (k < e.length) e[k + 1] += 0.5 * eu[t + 1];
            else e.push(eu[t], 0.5 * eu[t + 1]);
          }
        }
      }
      return (exp[v] = e);
    };
    let tot = 0;
    for (let v = 0; v < nv; v++) tot += expOf(v).length / 2;
    const es = new Int32Array(nv + 1), ei = new Int32Array(tot), ew = new Float64Array(tot);
    for (let v = 0, k = 0; v < nv; v++) {
      const e = exp[v]!;
      for (let t = 0; t < e.length; t += 2, k++) (ei[k] = e[t]), (ew[k] = e[t + 1]);
      es[v + 1] = k;
    }
    for (let v = 0; v < nv; v++) stamp[key[v]] = -1;
    return { ci, cj, cs, key, free, nF, es, ei, ew };
  };
  const LV: Level[] = [];
  for (let d = 0; d <= K0; d++) LV.push(buildLevel(d));
  const PR: Prol[] = [];
  for (let d = 0; d < K0; d++) {
    const C = LV[d], F = LV[d + 1];
    for (let v = 0; v < C.key.length; v++) stamp[C.key[v]] = v;
    const cap = 16, rc = new Int32Array(F.nF * cap), rw = new Float64Array(F.nF * cap), rl = new Int32Array(F.nF), got = new Uint8Array(F.nF);
    const fromC = (f: number, k: number, w: number) => {
      const v = stamp[k];
      for (let t = C.es[v]; t < C.es[v + 1]; t++) {
        const c = C.ei[t], base = f * cap;
        let q = 0;
        while (q < rl[f] && rc[base + q] !== c) q++;
        if (q < rl[f]) rw[base + q] += w * C.ew[t];
        else if (rl[f] >= cap) throw new Error("sheet: a prolongation row too wide");
        else (rc[base + rl[f]] = c), (rw[base + rl[f]] = w * C.ew[t]), rl[f]++;
      }
    };
    for (let v = 0; v < F.key.length; v++) {
      const f = F.free[v];
      if (f >= 0 && stamp[F.key[v]] >= 0) fromC(f, F.key[v], 1), (got[f] = 1);
    }
    // vertices new at d + 1: the middles of the edges and of the diagonals of cells split there
    const side = 1 << (K0 - d), hs = side >> 1;
    for (let c = 0; c < C.ci.length; c++) {
      if (C.cs[c] !== side) continue;
      const a = C.cj[c] * GW + C.ci[c];
      const mids = [a + hs * GW + hs, a, a + side * GW + side, a + hs, a, a + side, a + side * GW + hs, a + side * GW, a + side * GW + side, a + hs * GW, a, a + side * GW, a + hs * GW + side, a + side, a + side * GW + side];
      for (let q = 0; q < 15; q += 3) {
        if (stamp[mids[q]] >= 0) continue;
        const fv = findKey(F.key, mids[q]);
        if (fv < 0) continue;
        const f = F.free[fv];
        if (f < 0 || got[f]) continue;
        fromC(f, mids[q + 1], 0.5), fromC(f, mids[q + 2], 0.5), (got[f] = 1);
      }
    }
    const rs = new Int32Array(F.nF + 1);
    for (let f = 0; f < F.nF; f++) rs[f + 1] = rs[f] + rl[f];
    const col = new Int32Array(rs[F.nF]), w = new Float64Array(rs[F.nF]);
    for (let f = 0; f < F.nF; f++) for (let q = 0; q < rl[f]; q++) (col[rs[f] + q] = rc[f * cap + q]), (w[rs[f] + q] = rw[f * cap + q]);
    PR.push(prolOf(F.nF, C.nF, rs, col, w));
    for (let v = 0; v < C.key.length; v++) stamp[C.key[v]] = -1;
  }
  tick("levels");

  // per cell means of the finest cells' facts, by summed-area tables over the finest cells
  const sat = (val: (c: number) => number) => {
    const S = new Float64Array((GX + 1) * (GY + 1));
    for (let j = 0; j < GY; j++) {
      let run = 0;
      for (let i = 0; i < GX; i++) (run += val(j * GX + i)), (S[(j + 1) * (GX + 1) + i + 1] = S[j * (GX + 1) + i + 1] + run);
    }
    return S;
  };
  const rect = (S: Float64Array, i0: number, j0: number, s: number) => { const W = GX + 1; return S[(j0 + s) * W + i0 + s] - S[j0 * W + i0 + s] - S[(j0 + s) * W + i0] + S[j0 * W + i0]; };
  const SA = sat((c) => aC[c]), SS = sat((c) => sC[c]), SL = sat((c) => landC[c]), SK = sat((c) => +(kindC[c] === 0));
  const Lcur = Float64Array.from(Lf);
  /** per finest cell: its size pull times its planned size (the field at its corners, their mean log) */
  let SP = new Float64Array(0);
  const replan = () => { SP = sat((c) => { const i = c % GX, j = (c - i) / GX, v = j * GW + i; return sC[c] * Math.exp((Lcur[v] + Lcur[v + 1] + Lcur[v + GW] + Lcur[v + GW + 1]) / 4); }); };
  replan();
  tick("tables");

  // ---- a problem on one level: its triangles, matrix and multigrid hierarchy, the places of its vertices
  interface Prob {
    d: number; L: Level; nV: number; nF: number; NL: number; NT: number;
    tv: Int32Array; tside: Float64Array; cellA: Float64Array; cellS: Float64Array; cellK0: Float64Array; sizeT: Float64Array; kx: Float64Array; ky: Float64Array;
    eS: Int32Array; eC: Int32Array; eF: Int32Array; eW: Float64Array; sS: Int32Array; sSlot: Int32Array; A: Bsr; AL: Bsr[]; bufs: { x: Float64Array; b: Float64Array; r: Float64Array }[];
    dense: (b: Float64Array, x: Float64Array) => void; QX: Float64Array; QY: Float64Array; XY: Float64Array; rhs: Float64Array;
    guard: Float64Array; stTri: Int32Array; stW: Float64Array; pin: number; built: boolean;
  }
  const cellOfTmp = new Int32Array(NFC);
  const makeProb = (d: number): Prob => {
    const tp0 = performance.now();
    const L = LV[d], NL = L.ci.length, NT = 2 * NL, nV = L.key.length, nF = L.nF;
    for (let v = 0; v < nV; v++) stamp[L.key[v]] = v;
    const tv = new Int32Array(NT * 3), tside = new Float64Array(NT), cellA = new Float64Array(NL), cellS = new Float64Array(NL), cellK0 = new Float64Array(NL), sizeT = new Float64Array(NL), kx = new Float64Array(NL), ky = new Float64Array(NL);
    for (let c = 0; c < NL; c++) {
      const i0 = L.ci[c], j0 = L.cj[c], s = L.cs[c], a = j0 * GW + i0, b = a + s, cc = a + s * GW + s, dd = a + s * GW, m = s * s;
      (tv[6 * c] = stamp[a]), (tv[6 * c + 1] = stamp[b]), (tv[6 * c + 2] = stamp[cc]), (tv[6 * c + 3] = stamp[a]), (tv[6 * c + 4] = stamp[cc]), (tv[6 * c + 5] = stamp[dd]);
      (tside[2 * c] = s), (tside[2 * c + 1] = s);
      (kx[c] = (kxS[Math.min(GX, i0 + s)] - kxS[i0]) / s), (ky[c] = (kyS[Math.min(GY, j0 + s)] - kyS[j0]) / s);
      (cellA[c] = rect(SA, i0, j0, s) / m), (cellS[c] = rect(SS, i0, j0, s) / m), (cellK0[c] = rect(SK, i0, j0, s) / m);
    }
    for (let v = 0; v < nV; v++) stamp[L.key[v]] = -1;
    // per triangle its corners as free vertices (eC the corner, eF the free vertex, eW the weight; a hanging corner
    // spreads), and per pair of those the slot of its block in the matrix (sSlot from sS[t], row-major over the pairs)
    const eS = new Int32Array(NT + 1);
    for (let t = 0; t < NT; t++) { let k = 0; for (let c = 0; c < 3; c++) { const v = tv[3 * t + c]; k += L.es[v + 1] - L.es[v]; } eS[t + 1] = eS[t] + k; }
    const eC = new Int32Array(eS[NT]), eF = new Int32Array(eS[NT]), eW = new Float64Array(eS[NT]), sS = new Int32Array(NT + 1);
    for (let t = 0; t < NT; t++) {
      let k = eS[t];
      for (let c = 0; c < 3; c++) { const v = tv[3 * t + c]; for (let q = L.es[v]; q < L.es[v + 1]; q++, k++) (eC[k] = c), (eF[k] = L.ei[q]), (eW[k] = L.ew[q]); }
      const m = eS[t + 1] - eS[t];
      sS[t + 1] = sS[t] + m * m;
    }
    const sSlot = new Int32Array(sS[NT]);
    // each free vertex's triangles
    const incS = new Int32Array(nF + 1);
    for (let k = 0; k < eS[NT]; k++) incS[eF[k] + 1]++;
    for (let f = 0; f < nF; f++) incS[f + 1] += incS[f];
    const incAt = incS.slice(0, nF), incK = new Int32Array(eS[NT]);
    for (let t = 0; t < NT; t++) for (let k = eS[t]; k < eS[t + 1]; k++) incK[incAt[eF[k]]++] = k;
    // rows: each free vertex with the free vertices it shares a triangle with
    const seen = new Int32Array(nF).fill(-1), posOf = new Int32Array(nF), rs = new Int32Array(nF + 1), tri = new Int32Array(eS[NT]);
    for (let t = 0; t < NT; t++) for (let k = eS[t]; k < eS[t + 1]; k++) tri[k] = t;
    let colBuf = new Int32Array(nF * 10), nCol = 0;
    const rowTmp: number[] = [];
    for (let f = 0; f < nF; f++) {
      rowTmp.length = 0;
      (seen[f] = f), rowTmp.push(f);
      for (let u = incS[f]; u < incS[f + 1]; u++) {
        const t = tri[incK[u]];
        for (let k = eS[t]; k < eS[t + 1]; k++) { const J = eF[k]; if (seen[J] !== f) (seen[J] = f), rowTmp.push(J); }
      }
      rowTmp.sort((a, b) => a - b);
      if (nCol + rowTmp.length > colBuf.length) { const c2 = new Int32Array(colBuf.length * 2); c2.set(colBuf); colBuf = c2; }
      for (let q = 0; q < rowTmp.length; q++) (colBuf[nCol + q] = rowTmp[q]), (posOf[rowTmp[q]] = nCol + q);
      nCol += rowTmp.length;
      rs[f + 1] = nCol;
      // the slots of this row's pairs in each of its triangles
      for (let u = incS[f]; u < incS[f + 1]; u++) {
        const kp = incK[u], t = tri[kp], m = eS[t + 1] - eS[t], base = sS[t] + (kp - eS[t]) * m;
        for (let kq = eS[t]; kq < eS[t + 1]; kq++) sSlot[base + kq - eS[t]] = posOf[eF[kq]];
      }
    }
    const col = colBuf.slice(0, nCol), dg = new Int32Array(nF);
    for (let f = 0; f < nF; f++) for (let k = rs[f]; k < rs[f + 1]; k++) if (col[k] === f) dg[f] = k;
    const A: Bsr = { n: nF, rs, col, val: new Float64Array(nCol * 4), dg, inv: new Float64Array(nF * 4) };
    // the coarser levels' patterns, P^T A P row by row
    const AL: Bsr[] = new Array(d + 1);
    AL[d] = A;
    let mark = new Int32Array(1);
    for (let e = 0; e <= d; e++) if (LV[e].nF + 1 > mark.length) mark = new Int32Array(LV[e].nF + 1);
    mark.fill(-1);
    for (let e = d - 1; e >= 0; e--) {
      const P = PR[e], Fm = AL[e + 1], nc = LV[e].nF, cap = 64, b2 = new Int32Array(nc * cap), l2 = new Int32Array(nc);
      for (let I = 0; I < nc; I++) {
        const base = I * cap;
        for (let t = P.ts[I]; t < P.ts[I + 1]; t++) {
          const i = P.tcol[t];
          for (let k = Fm.rs[i]; k < Fm.rs[i + 1]; k++) {
            const j = Fm.col[k];
            for (let u = P.rs[j]; u < P.rs[j + 1]; u++) {
              const J = P.col[u];
              if (mark[J] === I) continue;
              mark[J] = I;
              if (l2[I] >= cap) throw new Error("sheet: a coarse row too wide");
              b2[base + l2[I]++] = J;
            }
          }
        }
        if (mark[I] !== I) (mark[I] = I), (b2[base + l2[I]++] = I);
      }
      mark.fill(-1);
      AL[e] = bsrOf(nc, b2, cap, l2);
    }
    // stations in this level's cells
    cellOfTmp.fill(-1);
    for (let c = 0; c < NL; c++) for (let j = L.cj[c]; j < L.cj[c] + L.cs[c]; j++) cellOfTmp.fill(c, j * GX + L.ci[c], j * GX + L.ci[c] + L.cs[c]);
    const stTri = new Int32Array(n * 3), stW = new Float64Array(n * 3);
    for (let s = 0; s < n; s++) {
      let fi = (PX[s] - bx0) / h, fj = (PY[s] - by0) / h;
      const c = Math.max(0, cellOfTmp[Math.min(NCY - 1, Math.max(0, Math.floor(fj))) * GX + Math.min(NCX - 1, Math.max(0, Math.floor(fi)))]), sz = L.cs[c];
      (fi = Math.min(1, Math.max(0, (fi - L.ci[c]) / sz))), (fj = Math.min(1, Math.max(0, (fj - L.cj[c]) / sz)));
      const k = 3 * s, t = fi >= fj ? 6 * c : 6 * c + 3;
      (stTri[k] = tv[t]), (stTri[k + 1] = tv[t + 1]), (stTri[k + 2] = tv[t + 2]);
      if (fi >= fj) (stW[k] = 1 - fi), (stW[k + 1] = fi - fj), (stW[k + 2] = fj);
      else (stW[k] = 1 - fj), (stW[k + 1] = fi), (stW[k + 2] = fj - fi);
    }
    // the vertex nearest the prototype lattice's middle pins the drawing's place
    let pin = 0, bd = Infinity;
    const mi = (NCX + 1) >> 1, mj = (NCY + 1) >> 1;
    for (let v = 0; v < nV; v++) {
      if (L.free[v] < 0) continue;
      const k = L.key[v], dd = (k % GW - mi) ** 2 + (((k / GW) | 0) - mj) ** 2;
      if (dd < bd) (bd = dd), (pin = v);
    }
    const P: Prob = {
      d, L, nV, nF, NL, NT, tv, tside, cellA, cellS, cellK0, sizeT, kx, ky, eS, eC, eF, eW, sS, sSlot, A, AL,
      bufs: AL.map((C) => ({ x: new Float64Array(C.n * 2), b: new Float64Array(C.n * 2), r: new Float64Array(C.n * 2) })),
      dense: () => {}, QX: new Float64Array(nV), QY: new Float64Array(nV), XY: new Float64Array(nF * 2), rhs: new Float64Array(nF * 2),
      guard: new Float64Array(NT), stTri, stW, pin, built: false,
    };
    planOf(P);
    ms.prob = (ms.prob ?? 0) + Math.round(performance.now() - tp0);
    return P;
  };
  const planOf = (P: Prob) => {
    const L = P.L;
    for (let c = 0; c < P.NL; c++) P.sizeT[c] = rect(SP, L.ci[c], L.cj[c], L.cs[c]) / rect(SS, L.ci[c], L.cj[c], L.cs[c]);
  };
  const galerkin = (P: Prob) => {
    const { AL, d } = P;
    let mark = new Int32Array(1);
    for (let e = 0; e < d; e++) if (LV[e].nF + 1 > mark.length) mark = new Int32Array(LV[e].nF + 1);
    for (let e = d - 1; e >= 0; e--) {
      const Pr = PR[e], Fm = AL[e + 1], C = AL[e], val = C.val, fv = Fm.val;
      val.fill(0);
      for (let I = 0; I < C.n; I++) {
        for (let k = C.rs[I]; k < C.rs[I + 1]; k++) mark[C.col[k]] = 4 * k;
        for (let t = Pr.ts[I]; t < Pr.ts[I + 1]; t++) {
          const i = Pr.tcol[t], pI = Pr.tw[t];
          for (let k = Fm.rs[i], v = 4 * k; k < Fm.rs[i + 1]; k++, v += 4) {
            const j = Fm.col[k], a0 = pI * fv[v], a1 = pI * fv[v + 1], a2 = pI * fv[v + 2], a3 = pI * fv[v + 3];
            for (let u = Pr.rs[j]; u < Pr.rs[j + 1]; u++) {
              const w = Pr.w[u], q = mark[Pr.col[u]];
              (val[q] += w * a0), (val[q + 1] += w * a1), (val[q + 2] += w * a2), (val[q + 3] += w * a3);
            }
          }
        }
      }
      invDiag(C);
    }
    invDiag(P.A);
    // the coarsest level by a dense Cholesky factor
    const C = AL[0], m = C.n * 2, M = new Float64Array(m * m);
    for (let i = 0; i < C.n; i++) for (let k = C.rs[i]; k < C.rs[i + 1]; k++) {
      const j = C.col[k], v = k * 4;
      (M[2 * i * m + 2 * j] += C.val[v]), (M[2 * i * m + 2 * j + 1] += C.val[v + 1]), (M[(2 * i + 1) * m + 2 * j] += C.val[v + 2]), (M[(2 * i + 1) * m + 2 * j + 1] += C.val[v + 3]);
    }
    let tr = 0;
    for (let i = 0; i < m; i++) tr += M[i * m + i];
    for (let i = 0; i < m; i++) M[i * m + i] += (1e-9 * tr) / m + 1e-300;
    for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
      let s = M[i * m + j];
      for (let k = 0; k < j; k++) s -= M[i * m + k] * M[j * m + k];
      M[i * m + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / M[j * m + j];
    }
    const y = new Float64Array(m);
    P.dense = (b, x) => {
      for (let i = 0; i < m; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= M[i * m + k] * y[k]; y[i] = s / M[i * m + i]; }
      for (let i = m - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < m; k++) s -= M[k * m + i] * x[k]; x[i] = s / M[i * m + i]; }
    };
  };
  const vcycle = (P: Prob, d: number) => {
    const C = P.AL[d], B = P.bufs[d];
    if (d === 0) return P.dense(B.b, B.x);
    B.x.fill(0);
    gs(C, B.x, B.b, true);
    mulBsr(C, B.x, B.r);
    for (let u = 0; u < B.r.length; u++) B.r[u] = B.b[u] - B.r[u];
    const Pr = PR[d - 1], D = P.bufs[d - 1];
    for (let I = 0; I < P.AL[d - 1].n; I++) {
      let a = 0, b = 0;
      for (let t = Pr.ts[I]; t < Pr.ts[I + 1]; t++) { const i = 2 * Pr.tcol[t], w = Pr.tw[t]; (a += w * B.r[i]), (b += w * B.r[i + 1]); }
      (D.b[2 * I] = a), (D.b[2 * I + 1] = b);
    }
    vcycle(P, d - 1);
    for (let i = 0; i < C.n; i++) {
      let a = 0, b = 0;
      for (let u = Pr.rs[i]; u < Pr.rs[i + 1]; u++) { const J = 2 * Pr.col[u], w = Pr.w[u]; (a += w * D.x[J]), (b += w * D.x[J + 1]); }
      (B.x[2 * i] += a), (B.x[2 * i + 1] += b);
    }
    gs(C, B.x, B.b, false);
  };
  /** conjugate gradients on A x = rhs from x, the V-cycle as preconditioner; its iterations */
  const pcg = (P: Prob, tol: number, maxIt: number): number => {
    const { A, rhs, XY: x } = P, m = x.length, r = new Float64Array(m), z = new Float64Array(m), p = new Float64Array(m), qv = new Float64Array(m), top = P.bufs[P.d];
    mulBsr(A, x, qv);
    let bn = 0;
    for (let i = 0; i < m; i++) (r[i] = rhs[i] - qv[i]), (bn += rhs[i] * rhs[i]);
    bn = Math.sqrt(bn) || 1;
    const prec = () => { top.b.set(r); vcycle(P, P.d); z.set(top.x); };
    prec();
    p.set(z);
    let rz = 0;
    for (let i = 0; i < m; i++) rz += r[i] * z[i];
    let it = 0;
    for (; it < maxIt; it++) {
      mulBsr(A, p, qv);
      let pq = 0;
      for (let i = 0; i < m; i++) pq += p[i] * qv[i];
      const al = rz / pq;
      let rn = 0;
      for (let i = 0; i < m; i++) (x[i] += al * p[i]), (r[i] -= al * qv[i]), (rn += r[i] * r[i]);
      if (Math.sqrt(rn) / bn < tol) { it++; break; }
      prec();
      let rz2 = 0;
      for (let i = 0; i < m; i++) rz2 += r[i] * z[i];
      const be = rz2 / rz;
      rz = rz2;
      for (let i = 0; i < m; i++) p[i] = z[i] + be * p[i];
    }
    return it;
  };
  const fromXY = (P: Prob) => {
    const { L, QX, QY, XY } = P;
    for (let v = 0; v < P.nV; v++) {
      let x = 0, y = 0;
      for (let k = L.es[v]; k < L.es[v + 1]; k++) { const f = 2 * L.ei[k], w = L.ew[k]; (x += w * XY[f]), (y += w * XY[f + 1]); }
      (QX[v] = x), (QY[v] = y);
    }
  };
  const toXY = (P: Prob) => { for (let v = 0; v < P.nV; v++) { const f = P.L.free[v]; if (f >= 0) (P.XY[2 * f] = P.QX[v]), (P.XY[2 * f + 1] = P.QY[v]); } };
  const stationsOf = (P: Prob, X: Float64Array, Y: Float64Array) => {
    const { stTri: T, stW: W, QX, QY } = P;
    for (let s = 0, k = 0; s < n; s++, k += 3) {
      (X[s] = W[k] * QX[T[k]] + W[k + 1] * QX[T[k + 1]] + W[k + 2] * QX[T[k + 2]]), (Y[s] = W[k] * QY[T[k]] + W[k + 1] * QY[T[k + 1]] + W[k + 2] * QY[T[k + 2]]);
    }
  };
  const loc = new Float64Array(36), lb = new Float64Array(6), rR = Array.from({ length: 4 }, () => new Float64Array(6)), rGa = new Float64Array(36), rGs = new Float64Array(36);
  let nearFold = 0;
  /** with the holds frozen, a hold may still tighten (never loosen), where a cell has come nearer folding than its
   * hold allows; whether any did */
  const tighten = (P: Prob): boolean => {
    const { tv, QX, QY, guard } = P;
    let any = false;
    for (let t = 0; t < P.NT; t++) {
      const s = P.tside[t] * h, a = tv[3 * t], b = tv[3 * t + 1], cc = tv[3 * t + 2], sz = P.sizeT[t >> 1];
      let ja: number, jb: number, jc: number, jd: number;
      if (t & 1) (ja = (QX[b] - QX[cc]) / s), (jb = (QX[cc] - QX[a]) / s), (jc = (QY[b] - QY[cc]) / s), (jd = (QY[cc] - QY[a]) / s);
      else (ja = (QX[b] - QX[a]) / s), (jb = (QX[cc] - QX[b]) / s), (jc = (QY[b] - QY[a]) / s), (jd = (QY[cc] - QY[b]) / s);
      { const kx = P.kx[t >> 1], ky = P.ky[t >> 1]; (ja /= kx), (jc /= kx), (jb /= ky), (jd /= ky); }
      const Ee = (ja + jd) / 2, He = (jc - jb) / 2, ce = Math.sqrt(Ee * Ee + He * He), ua = (ja - jd) / 2, ub = (jb + jc) / 2, un = Math.sqrt(ua * ua + ub * ub);
      const g1 = Math.min(1, Math.max(0, (un / Math.max(ce, 1e-12) - 0.4) / 0.5)), g2 = Math.min(1, Math.max(0, (o.lo * sz - ce) / (o.lo * sz))), gm = Math.max(g1, g2), w = o.wFold * gm * gm;
      if (w > 1.1 * guard[t] + 0.01 * o.wFold && (un >= o.tightFrom * ce || ce < o.lo * sz)) (guard[t] = w), (any = true);
    }
    return any;
  };
  /** the next solve's right side, and with `keep` its matrix too: each triangle's even part to its planned size at its
   * current turn; its hold the larger of what its stretch asks now and its last hold times `keep` (null: holds as they
   * are) */
  const assemble = (P: Prob, keep: number | null, matrix: boolean) => {
    const { A, rhs, tv, QX, QY, L, guard } = P;
    if (matrix) A.val.fill(0);
    rhs.fill(0);
    nearFold = 0;
    for (let t = 0; t < P.NT; t++) {
      const c = t >> 1, sz = P.sizeT[c], s = P.tside[t] * h, a = tv[3 * t], b = tv[3 * t + 1], cc = tv[3 * t + 2], tri = t & 1;
      let ja: number, jb: number, jc: number, jd: number;
      if (tri) (ja = (QX[b] - QX[cc]) / s), (jb = (QX[cc] - QX[a]) / s), (jc = (QY[b] - QY[cc]) / s), (jd = (QY[cc] - QY[a]) / s);
      else (ja = (QX[b] - QX[a]) / s), (jb = (QX[cc] - QX[b]) / s), (jc = (QY[b] - QY[a]) / s), (jd = (QY[cc] - QY[b]) / s);
      const kx = P.kx[c], ky = P.ky[c], rest = kx !== 1 || ky !== 1;
      if (rest) (ja /= kx), (jc /= kx), (jb /= ky), (jd /= ky);
      const Ee = (ja + jd) / 2, He = (jc - jb) / 2, ce = Math.sqrt(Ee * Ee + He * He);
      if (keep !== null) {
        const ua = (ja - jd) / 2, ub = (jb + jc) / 2, un = Math.sqrt(ua * ua + ub * ub);
        const g1 = Math.min(1, Math.max(0, (un / Math.max(ce, 1e-12) - 0.4) / 0.5)), g2 = Math.min(1, Math.max(0, (o.lo * sz - ce) / (o.lo * sz))), gm = Math.max(g1, g2);
        guard[t] = keep >= 0 ? Math.max(o.wFold * gm * gm, keep * guard[t]) : (1 + keep) * guard[t] - keep * o.wFold * gm * gm;
      }
      const gd = guard[t];
      if (gd > 0.1 * o.wFold) nearFold++;
      if (rest) restRows(tri, kx, ky, rR, rGa, rGs);
      const ws = (P.cellS[c] + gd) / (h * h), R2 = rest ? rR[2] : ROWS[tri][2], R3 = rest ? rR[3] : ROWS[tri][3];
      const ct = ce > 1e-12 ? Ee / ce : 1, st = ce > 1e-12 ? He / ce : 0, side = P.tside[t] * h, f2 = ws * side * sz * ct, f3 = ws * side * sz * st;
      for (let k = 0; k < 6; k++) lb[k] = f2 * R2[k] + f3 * R3[k];
      if (matrix) {
        const wa = (P.cellA[c] + gd) / (h * h), Ga = rest ? rGa : GA[tri], Gs = rest ? rGs : GS[tri];
        for (let k = 0; k < 36; k++) loc[k] = wa * Ga[k] + ws * Gs[k];
      }
      // each corner's share to its free vertices, each pair's block to its slot
      const e0 = P.eS[t], e1 = P.eS[t + 1], m = e1 - e0;
      for (let kp = e0; kp < e1; kp++) {
        const fp = 2 * P.eF[kp], wp = P.eW[kp], cp = P.eC[kp];
        (rhs[fp] += wp * lb[2 * cp]), (rhs[fp + 1] += wp * lb[2 * cp + 1]);
        if (!matrix) continue;
        const base = P.sS[t] + (kp - e0) * m;
        for (let kq = e0; kq < e1; kq++) {
          const w = wp * P.eW[kq], sl = P.sSlot[base + kq - e0] * 4, r0 = 12 * cp + 2 * P.eC[kq], r1 = r0 + 6;
          (A.val[sl] += w * loc[r0]), (A.val[sl + 1] += w * loc[r0 + 1]), (A.val[sl + 2] += w * loc[r1]), (A.val[sl + 3] += w * loc[r1 + 1]);
        }
      }
    }
    const PIN = 1e-4, fp = L.free[P.pin], v = P.pin;
    if (matrix) { const sl = A.dg[fp] * 4; (A.val[sl] += PIN), (A.val[sl + 3] += PIN); }
    (rhs[2 * fp] += PIN * QX[v]), (rhs[2 * fp + 1] += PIN * QY[v]);
  };
  const SXa = new Float64Array(n), SYa = new Float64Array(n), BX = new Float64Array(n), BY = new Float64Array(n), mv = new Float64Array(n);
  let turned = 0, rounds = 0;
  const unturn = (P: Prob) => {
    stationsOf(P, SXa, SYa);
    let px = 0, py = 0, qx = 0, qy = 0, dot = 0, crs = 0;
    for (let i = 0; i < n; i++) (px += PX[i]), (py += PY[i]), (qx += SXa[i]), (qy += SYa[i]);
    (px /= n), (py /= n), (qx /= n), (qy /= n);
    for (let i = 0; i < n; i++) { const ax = PX[i] - px, ay = PY[i] - py, bx = SXa[i] - qx, by = SYa[i] - qy; (dot += ax * bx + ay * by), (crs += ax * by - ay * bx); }
    const ang = Math.atan2(crs, dot), c = Math.cos(-ang), s = Math.sin(-ang);
    for (let v = 0; v < P.nV; v++) { const x = P.QX[v] - qx, y = P.QY[v] - qy; (P.QX[v] = qx + c * x - s * y), (P.QY[v] = qy + s * x + c * y); }
    turned += ang;
  };
  const stages: string[] = [];
  /** how many of P's triangles are folded */
  const foldsOf = (P: Prob) => {
    let folds = 0;
    const { QX, QY, tv } = P;
    for (let t = 0; t < P.NT; t++) {
      const s = P.tside[t] * h, a = tv[3 * t], b = tv[3 * t + 1], cc = tv[3 * t + 2];
      const J = t & 1 ? [(QX[b] - QX[cc]) / s, (QX[cc] - QX[a]) / s, (QY[b] - QY[cc]) / s, (QY[cc] - QY[a]) / s] : [(QX[b] - QX[a]) / s, (QX[cc] - QX[b]) / s, (QY[b] - QY[a]) / s, (QY[cc] - QY[b]) / s];
      if (J[0] * J[3] - J[1] * J[2] <= 0) folds++;
    }
    return folds;
  };
  /** up to `k` rounds on P: with `keep` its holds updated each round (as assemble says), else frozen (the matrix made
   * again only where a hold tightens, and with `mix` the rounds sped up by Anderson mixing of the last rounds: each
   * next guess the mix of the last rounds' results whose differences best cancel the last change) */
  const runRounds = (P: Prob, k: number, keep: number | null, tight = o.tighten, mix = o.aa) => {
    const moves: number[] = [], its: number[] = [], tm = [0, 0, 0], m2 = P.nF * 2;
    const M = keep === null ? mix : 0, dF: Float64Array[] = [], dG: Float64Array[] = [];
    let fPrev: Float64Array | null = null, gPrev: Float64Array | null = null, fnPrev = Infinity;
    const xIn = new Float64Array(m2);
    for (let it = 0; it < k; it++) {
      let t0 = performance.now();
      const tightened = keep === null && tight && tighten(P), matrix = keep !== null || tightened || !P.built;
      if (tightened && o.aaReset) (dF.length = 0), (dG.length = 0), (fPrev = null);
      assemble(P, keep, matrix);
      tm[0] += performance.now() - t0, (t0 = performance.now());
      if (matrix) {
        // a tightened hold changes the finest matrix; the coarser levels keep up only every galEvery rounds
        if (!tightened || !P.built || it % o.galEvery === 0) galerkin(P);
        else invDiag(P.A);
        P.built = keep === null;
      }
      tm[1] += performance.now() - t0, (t0 = performance.now());
      stationsOf(P, BX, BY);
      toXY(P);
      xIn.set(P.XY);
      its.push(keep === null ? pcg(P, o.tol, o.maxIt) : pcg(P, o.tolMove, o.itMove));
      tm[2] += performance.now() - t0;
      fromXY(P);
      unturn(P);
      if (M) {
        // g = this round's result, f = g - x; the next x = g - dG gamma, gamma the least squares fit of f by dF
        toXY(P);
        const g = P.XY.slice(), f = new Float64Array(m2);
        let fn = 0;
        for (let u = 0; u < m2; u++) (f[u] = g[u] - xIn[u]), (fn += f[u] * f[u]);
        if (fn > 4 * fnPrev) (dF.length = 0), (dG.length = 0), (fPrev = null);
        if (fPrev && gPrev) {
          const a = new Float64Array(m2), b = new Float64Array(m2);
          for (let u = 0; u < m2; u++) (a[u] = f[u] - fPrev[u]), (b[u] = g[u] - gPrev[u]);
          dF.push(a), dG.push(b);
          if (dF.length > M) dF.shift(), dG.shift();
        }
        (fPrev = f), (gPrev = g), (fnPrev = fn);
        const m = dF.length;
        if (m) {
          const N = new Float64Array(m * m), r = new Float64Array(m);
          for (let i = 0; i < m; i++) {
            for (let j = 0; j <= i; j++) { let s = 0; const A = dF[i], B = dF[j]; for (let u = 0; u < m2; u++) s += A[u] * B[u]; N[i * m + j] = N[j * m + i] = s; }
            let s = 0;
            for (let u = 0; u < m2; u++) s += dF[i][u] * f[u];
            r[i] = s;
          }
          let tr = 0;
          for (let i = 0; i < m; i++) tr += N[i * m + i];
          for (let i = 0; i < m; i++) N[i * m + i] += 1e-10 * tr + 1e-300;
          // Gaussian elimination, m is small
          for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) { const q = N[j * m + i] / N[i * m + i]; for (let c = i; c < m; c++) N[j * m + c] -= q * N[i * m + c]; r[j] -= q * r[i]; }
          const gam = new Float64Array(m);
          for (let i = m - 1; i >= 0; i--) { let s = r[i]; for (let c = i + 1; c < m; c++) s -= N[i * m + c] * gam[c]; gam[i] = s / N[i * m + i]; }
          for (let i = 0; i < m; i++) { const B = dG[i], w = gam[i]; for (let u = 0; u < m2; u++) P.XY[u] -= w * B[u]; }
          fromXY(P);
        }
      }
      stationsOf(P, SXa, SYa);
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) (mx += SXa[i] - BX[i]), (my += SYa[i] - BY[i]);
      (mx /= n), (my /= n);
      for (let i = 0; i < n; i++) { const dx = SXa[i] - BX[i] - mx, dy = SYa[i] - BY[i] - my; mv[i] = Math.sqrt(dx * dx + dy * dy); }
      const m90 = quantile(mv, 0.9);
      moves.push(Math.round(m90 * 100) / 100);
      rounds++;
      if (m90 < o.settle) break;
    }
    if (!stages.length) stages.push(`hierarchy rows/blocks ${P.AL.map((C) => `${C.n}/${C.rs[C.n]}`).join(" ")}`); // i18n-ignore: debug readout
    stages.push(`level ${P.d - K0} ${keep === null ? "frozen" : `keep ${keep}`}: ${moves.length} rounds, moves ${moves.join(" ")}, solver ${its.join(" ")}, ms ${tm.map(Math.round).join(" ")}`); // i18n-ignore: debug readout
  };
  /** the next level's problem from P: its places by prolongation, each triangle's hold its parent triangle's */
  const finer = (P: Prob): Prob => {
    const Q = makeProb(P.d + 1), Pr = PR[P.d];
    for (let f = 0; f < Q.nF; f++) {
      let x = 0, y = 0;
      for (let u = Pr.rs[f]; u < Pr.rs[f + 1]; u++) { const c = 2 * Pr.col[u], w = Pr.w[u]; (x += w * P.XY[c]), (y += w * P.XY[c + 1]); }
      (Q.XY[2 * f] = x), (Q.XY[2 * f + 1] = y);
    }
    fromXY(Q);
    cellOfTmp.fill(-1);
    const L = P.L;
    for (let c = 0; c < P.NL; c++) for (let j = L.cj[c]; j < L.cj[c] + L.cs[c]; j++) cellOfTmp.fill(c, j * GX + L.ci[c], j * GX + L.ci[c] + L.cs[c]);
    for (let c = 0; c < Q.NL; c++) {
      const i0 = Q.L.ci[c], j0 = Q.L.cj[c], s = Q.L.cs[c], pc = cellOfTmp[j0 * GX + i0], pi = L.ci[pc], pj = L.cj[pc], ps = L.cs[pc];
      for (let tri = 0; tri < 2; tri++) {
        const fi = (i0 + s * (tri ? 1 / 3 : 2 / 3) - pi) / ps, fj = (j0 + s * (tri ? 2 / 3 : 1 / 3) - pj) / ps;
        Q.guard[2 * c + tri] = P.guard[2 * pc + (fi >= fj ? 0 : 1)];
      }
    }
    return Q;
  };
  /** cores that came out smaller than the field asks get their wish raised by 80% of the shortfall (at most 0.5 in
   * log): per vertex the mean log size of the cells round it that are mostly land near lines, spread over each cell's
   * finest nodes; then bounded again, islands one size again, and the cells' plans made again */
  const raise = (P: Prob) => {
    const { L, tv, QX, QY } = P, lr = new Float64Array(P.nV), cnt = new Float64Array(P.nV), up = new Float64Array(P.nV), add = new Float64Array(GN);
    for (let c = 0; c < P.NL; c++) {
      if (P.cellK0[c] < 0.5) continue;
      let ar = 0;
      for (let tri = 0; tri < 2; tri++) {
        const t = 2 * c + tri, s = L.cs[c] * h, a = tv[3 * t], b = tv[3 * t + 1], cc = tv[3 * t + 2];
        const J = tri ? [(QX[b] - QX[cc]) / s, (QX[cc] - QX[a]) / s, (QY[b] - QY[cc]) / s, (QY[cc] - QY[a]) / s] : [(QX[b] - QX[a]) / s, (QX[cc] - QX[b]) / s, (QY[b] - QY[a]) / s, (QY[cc] - QY[b]) / s];
        ar += (J[0] * J[3] - J[1] * J[2]) / (2 * P.kx[c] * P.ky[c]);
      }
      const lsz = Math.log(Math.sqrt(Math.max(1e-30, ar)));
      for (const v of [tv[6 * c], tv[6 * c + 1], tv[6 * c + 2], tv[6 * c + 5]]) (lr[v] += lsz), cnt[v]++;
    }
    for (let v = 0; v < P.nV; v++) if (cnt[v]) { const k = L.key[v], got = lr[v] / cnt[v]; if (got < Lf[k]) up[v] = Math.min(0.5, 0.8 * (Lf[k] - got)); }
    for (let c = 0; c < P.NL; c++) {
      const i0 = L.ci[c], j0 = L.cj[c], s = L.cs[c], a = up[tv[6 * c]], b = up[tv[6 * c + 1]], cc = up[tv[6 * c + 2]], d = up[tv[6 * c + 5]];
      if (!(a || b || cc || d)) continue;
      for (let j = j0; j <= j0 + s; j++) for (let i = i0; i <= i0 + s; i++) {
        const u = (i - i0) / s, w = (j - j0) / s, k = j * GW + i;
        add[k] = Math.max(add[k], (1 - u) * (1 - w) * a + u * (1 - w) * b + u * w * cc + (1 - u) * w * d);
      }
    }
    for (let k = 0; k < GN; k++) if (add[k] > 0) {
      Lcur[k] += add[k];
      const i = k % GW, j = (k - i) / GW;
      for (let b = Math.max(0, j - 1); b <= Math.min(GY, j + 1); b++) for (let a = Math.max(0, i - 1); a <= Math.min(GX, i + 1); a++) dirty[b * GW + a] = 1;
    }
    bound(Lcur);
    if (islNodes.length) uniformIslands(Lcur, false);
    replan();
    planOf(P);
  };

  // ---- the rounds, as the schedule says; the first problem starts from the field's scale integrated along the
  // finest nodes' rows and columns
  let P: Prob | null = null;
  for (const st of (far ? o.scheduleFar : o.schedule).split(",")) {
    if (st === "raise") { if (P) raise(P); continue; }
    // "L:R:K" moving rounds; "sN" frozen on the finest; "fL:N" frozen on level K0 + L, holds as they are; "hL:N" holds
    // taken from the map as it is, then frozen rounds that may tighten them
    const kind = st[0] === "s" ? "s" : st[0] === "f" ? "f" : st[0] === "h" ? "h" : st[0] === "e" ? "e" : st[0] === "u" ? "u" : "m";
    const parts = (kind === "m" ? st : st.slice(1)).split(":").map(Number);
    const lev = kind === "s" || kind === "u" ? K0 : K0 + parts[0];
    if (!P) {
      P = makeProb(lev);
      // only along the rows and columns that hold a vertex
      const rowUsed = new Uint8Array(GH), colUsed = new Uint8Array(GW), IX = new Float64Array(GN), IY = new Float64Array(GN);
      for (let v = 0; v < P.nV; v++) { const k = P.L.key[v], i = k % GW; (colUsed[i] = 1), (rowUsed[(k - i) / GW] = 1); }
      for (let j = 0; j <= GY; j++) if (rowUsed[j]) for (let i = 1; i <= GX; i++) IX[j * GW + i] = IX[j * GW + i - 1] + Math.exp(Lf[j * GW + i]) * h * kxF[i - 1];
      for (let i = 0; i <= GX; i++) if (colUsed[i]) for (let j = 1; j <= GY; j++) IY[j * GW + i] = IY[(j - 1) * GW + i] + Math.exp(Lf[j * GW + i]) * h * kyF[j - 1];
      for (let v = 0; v < P.nV; v++) (P.QX[v] = IX[P.L.key[v]]), (P.QY[v] = IY[P.L.key[v]]);
      toXY(P);
    }
    while (P.d < lev) P = finer(P);
    if (kind === "s") runRounds(P, parts[0], null);
    else if (kind === "f") runRounds(P, parts[1], null, false);
    else if (kind === "h") { tighten(P), (P.built = false), runRounds(P, parts[1], null, false); }
    else if (kind === "u") {
      // where a cell folded: rounds with holds that may tighten (no mixing), until settled
      if (foldsOf(P)) (P.built = false), runRounds(P, parts[0], null, true, 0);
    }
    else if (kind === "e") runRounds(P, parts[1], -parts[2]);
    else runRounds(P, parts[1], parts[2] ?? 0.9);
  }
  if (!P) P = makeProb(K0);
  while (P.d < K0) P = finer(P);
  tick("solve");

  // ---- the result
  stationsOf(P, SXa, SYa);
  const folds = foldsOf(P);
  // displacement from the best uniform scaling of the whole map, real km
  let pmx = 0, pmy = 0, qmx = 0, qmy = 0;
  for (let i = 0; i < n; i++) (pmx += PX[i]), (pmy += PY[i]), (qmx += SXa[i]), (qmy += SYa[i]);
  (pmx /= n || 1), (pmy /= n || 1), (qmx /= n || 1), (qmy /= n || 1);
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) (num += (PX[i] - pmx) * (SXa[i] - qmx) + (PY[i] - pmy) * (SYa[i] - qmy)), (den += (PX[i] - pmx) ** 2 + (PY[i] - pmy) ** 2);
  const Ssim = num / (den || 1), ds = new Float64Array(n);
  for (let i = 0; i < n; i++) ds[i] = Math.hypot(SXa[i] - (Ssim * (PX[i] - pmx) + qmx), SYa[i] - (Ssim * (PY[i] - pmy) + qmy)) / Ssim;
  let overCap = 0;
  for (let v = 0; v < P.nV; v++) { const k = P.L.key[v], ex = Ssim * (bx0 + (k % GW) * h - pmx) + qmx, ey = Ssim * (by0 + ((k / GW) | 0) * h - pmy) + qmy; if (Math.hypot(P.QX[v] - ex, P.QY[v] - ey) / Ssim > capM) overCap++; }
  const cell = new Int32Array(P.NL * 3), land = new Float32Array(P.NL);
  for (let c = 0; c < P.NL; c++) (cell[3 * c] = P.L.ci[c]), (cell[3 * c + 1] = P.L.cj[c]), (cell[3 * c + 2] = P.L.cs[c]), (land[c] = rect(SL, P.L.ci[c], P.L.cj[c], P.L.cs[c]) / (P.L.cs[c] * P.L.cs[c]));
  tick("result");
  ms.total = Math.round(performance.now() - T0);
  return {
    x: SXa, y: SYa,
    mesh: { h, x0: bx0, y0: by0, gx: GX, nx: NCX, ny: NCY, key: P.L.key, qx: P.QX, qy: P.QY, cell, land },
    raster: { x0: bx0, y0: by0, rr, w: RW, h: RH, rowStart: rowStart.slice(), runA: runA.slice(0, nR), runB: runB.slice(0, nR), mass: par.slice(0, nR), covStart, covA: covA.slice(0, nCov), covB: covB.slice(0, nCov) },
    info: {
      stations: n, leaves: nLeaf, vertices: P.nV, free: P.nF, rounds, stages, folds, nearFold, islands: islands.length,
      turnedDeg: +((turned * 180) / Math.PI).toFixed(2), cellKm: +(h / 1000).toFixed(3), rasterM: Math.round(rr), floorPerKm: +(sFloor * 1000).toFixed(2),
      capKm: Math.round(capM / 1000), dispP90Km: +(quantile(ds, 0.9) / 1000).toFixed(1), overCap, ms,
    },
  };
}

/** the sheet's map of real places (graph metres) to drawn ones (pitches): linear in each cell's triangle, places past
 * the lattice as on its edge's cells */
export function sheetWarp(s: Sheet): (x: number, y: number) => [number, number] {
  const { h, x0, y0, gx, key, qx, qy, cell } = s.mesh, GW = gx + 1, nc = cell.length / 3;
  let gy = 0;
  for (let c = 0; c < nc; c++) gy = Math.max(gy, cell[3 * c + 1] + cell[3 * c + 2]);
  const leafOf = new Int32Array(gx * gy).fill(-1);
  for (let c = 0; c < nc; c++) { const i0 = cell[3 * c], j0 = cell[3 * c + 1], sz = cell[3 * c + 2]; for (let j = j0; j < j0 + sz; j++) leafOf.fill(c, j * gx + i0, j * gx + i0 + sz); }
  const vAt = (i: number, j: number) => findKey(key, j * GW + i);
  return (x, y) => {
    let fi = (x - x0) / h, fj = (y - y0) / h;
    let ci = Math.min(gx - 1, Math.max(0, Math.floor(fi))), cj = Math.min(gy - 1, Math.max(0, Math.floor(fj)));
    let c = leafOf[cj * gx + ci];
    // a place over no cell: the nearest cell along its row or column
    for (let r = 1; c < 0 && r < Math.max(gx, gy); r++) for (const [a, b] of [[ci - r, cj], [ci + r, cj], [ci, cj - r], [ci, cj + r]]) if (c < 0 && a >= 0 && b >= 0 && a < gx && b < gy && leafOf[b * gx + a] >= 0) (c = leafOf[b * gx + a]), (ci = a), (cj = b);
    const i0 = cell[3 * c], j0 = cell[3 * c + 1], sz = cell[3 * c + 2];
    (fi = Math.min(1, Math.max(0, (fi - i0) / sz))), (fj = Math.min(1, Math.max(0, (fj - j0) / sz)));
    const a = vAt(i0, j0), b = vAt(i0 + sz, j0), cc = vAt(i0 + sz, j0 + sz), d = vAt(i0, j0 + sz);
    if (fi >= fj) return [(1 - fi) * qx[a] + (fi - fj) * qx[b] + fj * qx[cc], (1 - fi) * qy[a] + (fi - fj) * qy[b] + fj * qy[cc]];
    return [(1 - fj) * qx[a] + fi * qx[cc] + (fj - fi) * qx[d], (1 - fj) * qy[a] + fi * qy[cc] + (fj - fi) * qy[d]];
  };
}
