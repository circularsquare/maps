// The sea and lakes on the network diagram (T-2169; docs/ui-diagram.md "Water"), decided in the drawing the way the
// zone borders are (borders.ts): on a grid over the drawing, each cell is sea where the places on the ground its nearest
// drawn stations' local maps carry it to are sea (waterReal.ts `isSea`), voted by nearness; every pill and the ground
// round it is land; where the drawing squeezes two land masses or the sea between two stations together, a strait is
// kept open (T-2540). Small specks are dropped, and the coast is fitted with runs at 0, 45 and 90 degrees, free to stray
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
/** sea specks and lakes under this many square pitches, and land without a station under LAND_MIN, are dropped */
const SEA_MIN = 60, LAND_MIN = 30;
/** the coast smoothed at this scale before it is fitted: inlets and notches under about twice it across go, pitches */
const SMOOTH = 4;
/** a fitted coast strays at most TOL_SEA seaward from the grid's, and landward its clearance from the lines and pills
 * less CLEAR, within TOL_MIN and TOL_LAND, pitches (T-2497: wide, so it runs straight for long like a line) */
const TOL_SEA = 12, TOL_LAND = 8, TOL_MIN = 1, CLEAR = 1;
/** corners rounded with this radius, pitches */
const RADIUS = 4;
/** a station on water on the ground takes the land mass found nearest, in steps of MASS_STEP out to MASS_FIND, metres */
const MASS_STEP = 200, MASS_FIND = 2000;
/** a strait never takes a pill nor PILL_CORE pitches round it; along one the coast strays at most STRAIT_TOL seaward
 * (or 0.4 of a cell) */
const PILL_CORE = 0.5, STRAIT_TOL = 0.5;
/** seaward, a coast strays at most this share of the way across the sea to the next land */
const GAP_SHARE = 0.35;
/** two stations have the sea between them where the straight way on the ground crosses BETWEEN_M of sea in one run,
 * and BETWEEN_SHARE of its length; looked at every BETWEEN_STEP_M (or a sixtieth of the way), metres */
const BETWEEN_M = 2000, BETWEEN_SHARE = 0.3, BETWEEN_STEP_M = 250, SQUEEZE = 0.5;
/** a station's region (`regionMaps`): the stations on its land mass within REGION_M of the middle of its REGION_CELL
 * square, metres, at least REGION_MIN */
const REGION_M = 50000, REGION_MIN = 3, REGION_CELL = 10000;
/** an island with no station is drawn by the stations within OWN_M of the one nearest it, on its land mass, metres */
const OWN_M = 15000;
/** T-2528 (`farShore`): the far shore's station is within FAR_SHORE_M of the land carried to; the way across looked at
 * every FAR_STEP_M (or a thirtieth of it), metres */
const FAR_SHORE_M = 8000, FAR_STEP_M = 500;

export interface SeaStats { cells: number; cell: number; sea: number; straits: number; rings: number; refit: number; ms: number }

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
  // T-2528: land a voter's map carries a cell to across the sea, nearest on the ground a station of the far shore that
  // is not among the cell's voters, is that station's to draw (it is drawn elsewhere): to this voter it is sea. A shore
  // drawn further from the far one than the water between them at its scale (Connecticut's from Long Island's) would
  // otherwise draw the far shore's land into the water between them.
  const gkd = kdTree(g.x, g.y), g1 = new Int32Array(1), gd1 = new Float64Array(1);
  const farShore = (k: number, gx: number, gy: number, found: number) => {
    const L = hyp(gx - g.x[k], gy - g.y[k]);
    if (L < BETWEEN_M || !gkd.nearest(gx, gy, 1, g1, gd1) || gd1[0] > FAR_SHORE_M * FAR_SHORE_M) return false;
    const s = g1[0];
    if (s === k || mass[s] !== mass[k]) return false;
    for (let q = 0; q < found; q++) if (near[q] === s) return false;
    const steps = Math.max(2, Math.ceil(L / Math.max(FAR_STEP_M, L / 30)));
    let run = 0;
    for (let z = 1; z < steps; z++) {
      if (real.isSea(g.x[k] + ((gx - g.x[k]) * z) / steps, g.y[k] + ((gy - g.y[k]) * z) / steps)) {
        if ((++run * L) / steps >= Math.max(BETWEEN_M, BETWEEN_SHARE * L)) return true;
      } else run = 0;
    }
    return false;
  };
  // a cell is sea where its nearest drawn stations, voting by nearness, each find the sea on the ground at the place
  // their maps carry it back to (T-2540)
  // (the nearest stations of each BLOCK by BLOCK square of cells looked up once, CANDIDATES of them, and each cell's
  // nearest taken among those)
  const kd = kdTree(x, y), cand = new Int32Array(CANDIDATES), cd = new Float64Array(CANDIDATES);
  const near = new Int32Array(VOTERS), nd = new Float64Array(VOTERS);
  const cls = new Uint8Array(NC), cellBody = new Int32Array(NC).fill(-1);
  // each station's land mass on the ground, and each cell's: its nearest drawn station's (T-2540)
  const mass = new Int32Array(n), cellMass = new Int32Array(NC).fill(-1), cellNear = new Int32Array(NC).fill(-1);
  for (let i = 0; i < n; i++) {
    let m = real.massAt(g.x[i], g.y[i]);
    for (let s = MASS_STEP; m < 0 && s <= MASS_FIND; s += MASS_STEP) for (let a = 0; a < 8 && m < 0; a++) m = real.massAt(g.x[i] + s * Math.cos((a * Math.PI) / 4), g.y[i] + s * Math.sin((a * Math.PI) / 4));
    mass[i] = m;
  }
  // past its own hops' reach (their mean drawn length), a station's map back to the ground is its region's (T-2550): a
  // city's map carried far out over empty land drew that land at the city's scale (a spit 180 pitches long east of
  // Hakodate, Awaji at Kobe's scale)
  const reach = new Float64Array(n), hops = new Float64Array(n);
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e], b = g.eb[e], l = hyp(x[b] - x[a], y[b] - y[a]);
    (reach[a] += l), (reach[b] += l), hops[a]++, hops[b]++;
  }
  for (let i = 0; i < n; i++) reach[i] = hops[i] ? reach[i] / hops[i] : 0;
  const far = regionMaps(g, x, y, mass, inv), peopled = new Set(mass);
  // per cell, whether most of its land votes found an island with no station (land drawn with no station on it is kept
  // only as such an island, T-2550)
  const isle = new Uint8Array(NC);
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
    let sea = 0, land = 0, bw = 0, lone = 0;
    if (found) cellNear[j * W + i] = near[0];
    for (let q = 0; q < found; q++) if (mass[near[q]] >= 0) { cellMass[j * W + i] = mass[near[q]]; break; }
    for (let q = 0; q < found; q++) {
      const k = near[q];
      if (!ok[k]) continue;
      const ox = px - x[k], oy = py - y[k], w = 1 / (nd[q] + pitch * pitch), r = Math.sqrt(nd[q]);
      const f = r > reach[k] ? reach[k] / r : 1, lx = ox * f, ly = oy * f, fx = ox - lx, fy = oy - ly;
      const rx = inv[k * 4] * lx + inv[k * 4 + 1] * ly + far[k * 4] * fx + far[k * 4 + 1] * fy;
      const ry = inv[k * 4 + 2] * lx + inv[k * 4 + 3] * ly + far[k * 4 + 2] * fx + far[k * 4 + 3] * fy;
      const gx = g.x[k] + rx, gy = g.y[k] + ry;
      if (real.isSea(gx, gy)) {
        sea += w;
        // (the far grid's sea knows no body)
        const b = w > bw ? real.seaBody(gx, gy) : -1;
        if (b >= 0) (bw = w), (cellBody[j * W + i] = b);
      } else {
        // land of another mass with stations on it is that mass's stations' to draw, and an island with none the
        // stations' near the one nearest it: to any other station it is sea, so each is drawn once (T-2550)
        const m = real.massAt(gx, gy), o = m >= 0 && !peopled.has(m) ? real.owner(m) : -1;
        if (m >= 0 && m !== mass[k] && peopled.has(m)) sea += w;
        else if (o >= 0 && (mass[o] !== mass[k] || (g.x[o] - g.x[k]) ** 2 + (g.y[o] - g.y[k]) ** 2 > OWN_M * OWN_M)) sea += w;
        else if (o < 0 && farShore(k, gx, gy, found)) sea += w;
        else (land += w), o >= 0 && (lone += w);
      }
    }
    cls[j * W + i] = sea > land ? 1 : 0;
    isle[j * W + i] = lone * 2 > land ? 1 : 0;
    }
  }
  // every pill and the ground round it is land (`core`: the pill and PILL_CORE round it, which a strait never takes)
  const forced = new Uint8Array(NC), core = new Uint8Array(NC);
  d.pills.forEach((p, q) => {
    const [r, o] = pillSize(d, d.pillOf[q]), R = r + o + p[4] + LAND * pitch, Rc = r + o + p[4] + PILL_CORE * pitch + 0.5 * h;
    const i0 = Math.max(0, Math.floor((Math.min(p[0], p[2]) - R - x0) / h)), i1 = Math.min(W - 1, Math.floor((Math.max(p[0], p[2]) + R - x0) / h));
    const j0 = Math.max(0, Math.floor((Math.min(p[1], p[3]) - R - y0) / h)), j1 = Math.min(H - 1, Math.floor((Math.max(p[1], p[3]) + R - y0) / h));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      // a cell is land when any of it is within R of the pill's axis
      const cx = x0 + (i + 0.5) * h, cy = y0 + (j + 0.5) * h, dd = segD(cx, cy, p[0], p[1], p[2], p[3]);
      if (dd <= R + h * Math.SQRT1_2) (forced[j * W + i] = 1), (cls[j * W + i] = 0);
      if (dd <= Rc) core[j * W + i] = 1;
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
  // straits (T-2540): where cells nearest stations on different land masses on the ground meet, or nearest two
  // stations with the sea between them on the ground, the sea, but in the pills' cores: so two land masses are never
  // drawn as one, and a bay or strait the drawing squeezes stays open
  const strait = new Uint8Array(NC), between = new Map<number, boolean>();
  const seaBetween = (a: number, b: number) => {
    const key = Math.min(a, b) * n + Math.max(a, b);
    let v = between.get(key);
    if (v === undefined) {
      const L = hyp(g.x[b] - g.x[a], g.y[b] - g.y[a]), steps = Math.max(2, Math.ceil(L / Math.max(BETWEEN_STEP_M, L / 60)));
      let run = 0, best = 0;
      for (let s = 1; s < steps; s++) {
        if (real.isSea(g.x[a] + ((g.x[b] - g.x[a]) * s) / steps, g.y[a] + ((g.y[b] - g.y[a]) * s) / steps)) best = Math.max(best, ++run);
        else run = 0;
      }
      // only where the drawing squeezes them together: drawn nearer than SQUEEZE of their distance on the ground at
      // their own maps' scales (elsewhere the sea found through the maps shows the water)
      const sa = Math.sqrt(Math.abs(J[a * 4] * J[a * 4 + 3] - J[a * 4 + 1] * J[a * 4 + 2])), sb = Math.sqrt(Math.abs(J[b * 4] * J[b * 4 + 3] - J[b * 4 + 1] * J[b * 4 + 2]));
      const squeezed = hyp(x[b] - x[a], y[b] - y[a]) < SQUEEZE * L * Math.sqrt(sa * sb);
      between.set(key, (v = squeezed && (best * L) / steps >= Math.max(BETWEEN_M, BETWEEN_SHARE * L)));
    }
    return v;
  };
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const c = j * W + i, m = cellMass[c], a = cellNear[c];
    if (a < 0 || core[c] || cls[c]) continue;
    for (const nc of [i > 0 ? c - 1 : -1, i < W - 1 ? c + 1 : -1, j > 0 ? c - W : -1, j < H - 1 ? c + W : -1]) {
      if (nc < 0) continue;
      const b = cellNear[nc];
      if (b < 0 || b === a) continue;
      if ((m >= 0 && cellMass[nc] >= 0 && cellMass[nc] !== m) || seaBetween(a, b)) (strait[c] = 1), (cls[c] = 1);
    }
  }
  let straitCells = 0;
  for (let c = 0; c < NC; c++) straitCells += strait[c];
  const area = h * h / (pitch * pitch);
  dropSmall(cls, W, H, 1, SEA_MIN / area, forced);
  dropSmall(cls, W, H, 0, LAND_MIN / area, forced, isle);

  // the coast: unit edges between sea and land cells, sea on the left, chained into rings at the grid's corners
  // (each edge with the landward tolerance its land cell's clearance allows)
  const W1 = W + 1, out = new Map<number, number[]>(), ea: number[] = [], eb: number[] = [], et: number[] = [], es: number[] = [];
  const sea = (i: number, j: number) => i >= 0 && j >= 0 && i < W && j < H && cls[j * W + i] === 1;
  const tolAt = (i: number, j: number) => {
    const cl = i >= 0 && j >= 0 && i < W && j < H ? (clear[j * W + i] / 3) * h : Infinity;
    return Math.max(TOL_MIN * pitch, Math.min(TOL_LAND * pitch, cl - CLEAR * pitch));
  };
  // seaward, a coast strays little along a strait (within a cell of one), so it never closes it
  const nearStrait = new Uint8Array(NC);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) if (strait[j * W + i] && cls[j * W + i]) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) if (i + di >= 0 && i + di < W && j + dj >= 0 && j + dj < H) nearStrait[(j + dj) * W + i + di] = 1;
  // and never more than GAP_SHARE of the way across the sea to the next land straight out from it, so a fitted coast
  // never closes a strait or a bay that the grid has open
  const gapCap = Math.ceil((2 * TOL_SEA * pitch) / h);
  const seaTol = (i: number, j: number, di: number, dj: number) => {
    if (i < 0 || j < 0 || i >= W || j >= H) return TOL_SEA * pitch;
    if (nearStrait[j * W + i]) return Math.max(STRAIT_TOL * pitch, 0.4 * h);
    let k = 0;
    for (let a = i, b = j; k < gapCap && a >= 0 && b >= 0 && a < W && b < H && cls[b * W + a] === 1; k++, a += di, b += dj);
    const open = k === gapCap || i + di * k < 0 || j + dj * k < 0 || i + di * k >= W || j + dj * k >= H;
    return open ? TOL_SEA * pitch : Math.max(0.4 * h, Math.min(TOL_SEA * pitch, GAP_SHARE * k * h));
  };
  const edge = (a: number, b: number, t: number, ts: number) => {
    const e = ea.length;
    ea.push(a), eb.push(b), et.push(t), es.push(ts);
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
    if (s !== sea(i + 1, j)) s ? edge(c(i + 1, j), c(i + 1, j + 1), tolAt(i + 1, j), seaTol(i, j, -1, 0)) : edge(c(i + 1, j + 1), c(i + 1, j), tolAt(i, j), seaTol(i + 1, j, 1, 0));
    if (s !== sea(i, j + 1)) s ? edge(c(i + 1, j + 1), c(i, j + 1), tolAt(i, j + 1), seaTol(i, j, 0, -1)) : edge(c(i, j + 1), c(i + 1, j + 1), tolAt(i, j), seaTol(i, j + 1, 0, 1));
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
    const px: number[] = [], py: number[] = [], pt: number[] = [], ps: number[] = [];
    let e = e0;
    for (let guard = 0; guard <= ea.length; guard++) {
      used[e] = 1;
      const [ax, ay] = cxy(ea[e]), [bx, by] = cxy(eb[e]);
      px.push((ax + bx) / 2), py.push((ay + by) / 2), pt.push(et[e]), ps.push(es[e]);
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
    rings.push(fitOcti(qx, qy, rot(pt), keepsSides(qx, qy), { tol: rot(ps), left: true }));
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
        // (a row more each way: the test below decides, so a vertex on a row's middle counts once, T-2540)
        const lo = Math.max(0, Math.ceil((Math.min(ay, by) - y0) / h - 0.5) - 1), hi = Math.min(H - 1, Math.floor((Math.max(ay, by) - y0) / h - 0.5) + 1);
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
  return { cmds: Float64Array.from(cmds), rings, grid, stats: { cells: NC, cell: h / pitch, sea: seaCells, straits: straitCells, rings: rings.length, refit, ms: performance.now() - t0 } };
}

/** the sea as drawn on a grid of cells `h` wide from (x0, y0): each cell's region (-1 land), each region's sea on the
 * ground (-1 unknown), each cell's depth into the sea (chamfer steps, 3 a cell) */
export interface SeaGrid { x0: number; y0: number; h: number; W: number; H: number; reg: Int32Array; body: number[]; depth: Uint16Array }

/** per station, the map from drawn offsets back to the ground over its region (T-2550): the inverse of the similarity
 * that best fits its region's stations on the ground (REGION_M), or its own map (`inv`) where that
 * draws the ground smaller, or there are too few */
function regionMaps(g: DiagramGraph, x: ArrayLike<number>, y: ArrayLike<number>, mass: Int32Array, inv: Float64Array): Float64Array {
  const n = g.key.length, out = inv.slice(), R = REGION_M, R2 = R * R;
  const key = (cx: number, cy: number) => (cx + 32768) * 65536 + (cy + 32768);
  const grid = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(g.x[i] / R), Math.floor(g.y[i] / R)), v = grid.get(k);
    if (v) v.push(i);
    else grid.set(k, [i]);
  }
  // (one fit per REGION_CELL square and land mass, over the stations within REGION_M of the square's middle)
  const S: number[] = [], fits = new Map<string, [number, number]>();
  const fitOf = (cx: number, cy: number, m: number): [number, number] => {
    const id = `${cx},${cy},${m}`, have = fits.get(id);
    if (have) return have;
    S.length = 0;
    const mx = (cx + 0.5) * REGION_CELL, my = (cy + 0.5) * REGION_CELL, gcx = Math.floor(mx / R), gcy = Math.floor(my / R);
    let gx = 0, gy = 0, dx = 0, dy = 0;
    for (let a = gcx - 1; a <= gcx + 1; a++) for (let b = gcy - 1; b <= gcy + 1; b++) for (const j of grid.get(key(a, b)) ?? []) {
      if (mass[j] !== m || (g.x[j] - mx) ** 2 + (g.y[j] - my) ** 2 > R2) continue;
      S.push(j), (gx += g.x[j]), (gy += g.y[j]), (dx += x[j]), (dy += y[j]);
    }
    let fit: [number, number] = [0, 0];
    if (S.length >= REGION_MIN) {
      (gx /= S.length), (gy /= S.length), (dx /= S.length), (dy /= S.length);
      let nr = 0, ni = 0, dd = 0;
      for (const j of S) {
        const ux = g.x[j] - gx, uy = g.y[j] - gy, vx = x[j] - dx, vy = y[j] - dy;
        (nr += ux * vx + uy * vy), (ni += ux * vy - uy * vx), (dd += ux * ux + uy * uy);
      }
      if (dd > 0) fit = [nr / dd, ni / dd];
    }
    fits.set(id, fit);
    return fit;
  };
  for (let i = 0; i < n; i++) {
    const [a, b] = fitOf(Math.floor(g.x[i] / REGION_CELL), Math.floor(g.y[i] / REGION_CELL), mass[i]), s2 = a * a + b * b;
    // (the inverse's scale against the station's own: a larger one carries a drawn offset farther on the ground)
    const own = Math.abs(inv[i * 4] * inv[i * 4 + 3] - inv[i * 4 + 1] * inv[i * 4 + 2]);
    if (!(s2 > 0) || 1 / s2 < own) continue;
    (out[i * 4] = a / s2), (out[i * 4 + 1] = b / s2), (out[i * 4 + 2] = -b / s2), (out[i * 4 + 3] = a / s2);
  }
  return out;
}

/** pieces of class `c` (4-connected) under `min` cells turned to the other class, but never a forced land cell to sea;
 * with `isle`, land holding no pill goes too unless most of it is an island with no station (T-2550) */
function dropSmall(cls: Uint8Array, W: number, H: number, c: number, min: number, forced: Uint8Array, isle?: Uint8Array) {
  const seen = new Uint8Array(W * H);
  for (let s = 0; s < W * H; s++) {
    if (seen[s] || cls[s] !== c) continue;
    const q = [s];
    seen[s] = 1;
    // (land holding a pill stays; a strait cut off inside the land, where pills left it no way through, goes)
    let pinned = false, lone = 0;
    for (let k = 0; k < q.length; k++) {
      const v = q[k], i = v % W, j = (v - i) / W;
      if (c === 0 && forced[v]) pinned = true;
      if (isle) lone += isle[v];
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const ni = i + di, nj = j + dj;
        if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
        const nv = nj * W + ni;
        if (!seen[nv] && cls[nv] === c) (seen[nv] = 1), q.push(nv);
      }
    }
    if (!pinned && (q.length < min || (isle && lone * 2 < q.length))) for (const v of q) cls[v] = 1 - c;
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
