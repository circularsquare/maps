// The network diagram's lines on the rubber sheet (T-2567; docs/ui-diagram.md "The rubber sheet"; notes/T-2567.md).
// The stations start at their places on the sheet (sheet.ts), and today's octilinear stages (layout.ts: B, the snap,
// the routing, the separation) run as a rebuild from there: every hop's target its length on the sheet, no A and no
// blocks' stage, B holding each station to its sheet place, and stations of two land masses with water between them
// keeping most of their sheet gap. Then the carry: a field of moves over the sheet's finest lattice, stiff on land and
// nearly free on water, met at each station's drawn place, so the water and the zones' borders follow the stations;
// a station whose drawn place is not on its own land in the carried map is met harder until every one is. Last, the
// ground carried through that map onto a grid over the drawing, its thin pieces and narrow rivers drawn as bands
// (sheetThin.ts, T-2580): the sea's coast and the zones' borders, fitted with runs at 0, 45 and 90 degrees. Pure: no
// DOM, no fetch.

import type { DiagramGraph } from "./graph";
import { dialled, layoutDiagram, type DiagramLayout, type LastPlaces } from "./layout";
import { massAt, massCoasts, sheetFinest, sheetWarp, type Sheet, type SheetRaster } from "./sheet";
import { fitOcti, rounded, segD, stationZones, type DiagramBorders, type ZoneShape } from "./borders";
import { chamfer, type RealWater } from "./waterReal";
import type { DiagramWater } from "./water";
import { riverBands, smoothEdges, thinPass, THIN, type CarryMap, type MapGrid, type RiverBands, type ThinOptions, type ThinStats } from "./sheetThin";

export interface SheetLinesOptions {
  /** the sheet's dial (sheet.ts SHEET.dial): sets the pitch, metres of a median target */
  dial: number;
  /** B's pull of each station to its sheet place (a rebuild's is 0.05) */
  wGeo: number;
  /** stations of two land masses within seaR pitches on the sheet, with water between them on the ground, keep at
   * least seaKeep of their sheet gap (their 6 nearest such) */
  seaKeep: number;
  seaR: number;
  /** the carry: links on land and on water, its rounds on a city map and a country map, and how much harder a station
   * off its own land is met each round */
  lamLand: number;
  lamWater: number;
  carryCity: number;
  carryCountry: number;
  wUp: number;
  /** its solves' tolerance (residual over right side) */
  carryTol: number;
  /** the drawn grid's cells at most, on a city map and a country map */
  cellsCity: number;
  cellsCountry: number;
  /** a station still off its own land has its land carried to it from at most this many pitches */
  bendMax: number;
  /** every zone's borders and, on a country map, all the water; otherwise only borders of zones a line passes through
   * and water nearest them (T-2609: in a cut band other zones are squeezed into slivers) */
  allZones: boolean;
  /** how far a zone border's fit may stray from the traced border, pitches */
  borderTol: number;
}

export const SHEET_LINES: SheetLinesOptions = {
  dial: 0.5, wGeo: 2, seaKeep: 0.8, seaR: 40, lamLand: 5, lamWater: 0.004, carryCity: 8, carryCountry: 12, wUp: 8, carryTol: 1e-4,
  cellsCity: 4e6, cellsCountry: 12e6, bendMax: 8, allZones: false, borderTol: 24,
};

/** the stations laid out on the sheet and the map carried with them */
export interface SheetLines {
  /** the graph with the sheet's places (layout units, `k` a pitch) */
  graph: DiagramGraph;
  k: number;
  layout: DiagramLayout;
  carry: Carry;
  info: { seaPairs: number; carryRounds: number[]; carryIters: number[]; offLand: number; ms: Record<string, number> };
}

/** the carried map: per node of the sheet's finest lattice ((gw by gh) from (x0, y0), h metres a cell) its drawn place,
 * pitches; per station its land mass (0 none) and its place on it */
export interface Carry { gw: number; gh: number; x0: number; y0: number; h: number; rx: Float64Array; ry: Float64Array; sx: Float64Array; sy: Float64Array; ownMass: Int32Array; landX: Float64Array; landY: Float64Array; raster: SheetRaster }

/** the river finder's cell for the map's narrow rivers (waterReal.ts `realWater`), metres: on a city map 25, so the
 * Harlem River is found whole (T-2580); a country map draws no narrow rivers (0), only water wide enough to show as
 * area (T-2566's design: on all Japan a band drawn from rivers' courses crossed a line wrongly about as often as rightly) */
export const riverCellM = (g: DiagramGraph) => (cityMap(g) ? 25 : 0);

/** a map under 600 km across (its stations' box), as sheet.ts tells a city map from a country's */
function cityMap(g: DiagramGraph): boolean {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < g.key.length; i++) (x0 = Math.min(x0, g.x[i])), (x1 = Math.max(x1, g.x[i])), (y0 = Math.min(y0, g.y[i])), (y1 = Math.max(y1, g.y[i]));
  return !(Math.hypot(x1 - x0, y1 - y0) >= 6e5);
}

/** a land mass with no station is drawn as water when the masks' edge cuts it for at least CUT_WIDTH of its width
 * (the square root of its area): only part of it is known (T-2591) */
const MG_STALE = 20, CUT_WIDTH = 0.5;

const quantile = (v: ArrayLike<number>, f: number) => { const s = Float64Array.from(v).sort(); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : 0; };

/** Lays graph `g` out on `sheet` and carries the map with its stations; `progress` (0 to 1) for a loading bar. */
export function sheetLines(g: DiagramGraph, sheet: Sheet, o: SheetLinesOptions = SHEET_LINES, progress?: (share: number) => void): SheetLines {
  const t0 = performance.now(), ms: Record<string, number> = {};
  let tl = t0;
  const tick = (k: string) => { const t = performance.now(); ms[k] = Math.round(t - tl); tl = t; };
  const R = sheet.raster!, n = g.key.length, nE = g.ea.length, PX = g.x, PY = g.y, WX = sheet.x, WY = sheet.y;
  // the pitch in metres of target, as the sheet's (its places are in these pitches)
  const tg = new Float64Array(nE);
  for (let e = 0; e < nE; e++) { const l = Math.max(50, Math.hypot(PX[g.eb[e]] - PX[g.ea[e]], PY[g.eb[e]] - PY[g.ea[e]])); tg[e] = 1000 * Math.pow(l / 1000, 1 - 0.7 * o.dial); }
  const K = nE ? quantile(tg, 0.5) / 9 : 100;

  // each station's land mass: its real place's, or the nearest land within 1 km (piers, the harbour's edge)
  const ownMass = new Int32Array(n), landX = Float64Array.from(PX), landY = Float64Array.from(PY);
  for (let s = 0; s < n; s++) {
    let m = massAt(R, PX[s], PY[s]);
    for (let r = R.rr; !m && r <= 1000; r += R.rr) for (let a = 0; a < 24; a++) {
      const x = PX[s] + r * Math.cos((a * Math.PI) / 12), y = PY[s] + r * Math.sin((a * Math.PI) / 12), v = massAt(R, x, y);
      if (v) { (m = v), (landX[s] = x), (landY[s] = y); break; }
    }
    ownMass[s] = m;
  }

  // ---- the octilinear stages from the sheet
  const warp = sheetWarp(sheet);
  let hop2 = g.hop;
  if (g.hop) { hop2 = new Float32Array(g.hop.length); for (let k = 0; k < g.hop.length; k += 2) { const [x, y] = warp(g.hop[k], g.hop[k + 1]); (hop2[k] = x * K), (hop2[k + 1] = y * K); } }
  const g2: DiagramGraph = { ...g, x: Float64Array.from(WX, (v) => v * K), y: Float64Array.from(WY, (v) => v * K), hop: hop2 };
  const hl = new Float64Array(nE);
  for (let e = 0; e < nE; e++) hl[e] = Math.max(1e-3, Math.hypot(WX[g.eb[e]] - WX[g.ea[e]], WY[g.eb[e]] - WY[g.ea[e]]));
  const medH = nE ? quantile(hl, 0.5) : 1;
  // water-gap pairs: stations of two land masses near each other on the sheet with water on the straight way between
  // them on the ground
  const sp: number[] = [];
  {
    const H = new Map<number, number[]>(), key = (x: number, y: number) => Math.floor(x / o.seaR) * 100003 + Math.floor(y / o.seaR);
    for (let i = 0; i < n; i++) { const k = key(WX[i], WY[i]); let l = H.get(k); if (!l) H.set(k, (l = [])); l.push(i); }
    const wet = (a: number, b: number) => {
      const l = Math.hypot(PX[b] - PX[a], PY[b] - PY[a]), m = Math.ceil(l / R.rr);
      for (let t = 1; t < m; t++) if (!massAt(R, PX[a] + ((PX[b] - PX[a]) * t) / m, PY[a] + ((PY[b] - PY[a]) * t) / m)) return true;
      return false;
    };
    const cand: [number, number][] = [];
    for (let a = 0; a < n; a++) {
      if (!ownMass[a]) continue;
      cand.length = 0;
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) for (const b of H.get(key(WX[a] + i * o.seaR, WY[a] + j * o.seaR)) ?? []) {
        if (b <= a || !ownMass[b] || ownMass[b] === ownMass[a]) continue;
        const d = Math.hypot(WX[b] - WX[a], WY[b] - WY[a]);
        if (d < o.seaR && d > 1e-6) cand.push([d, b]);
      }
      cand.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
      for (const [d, b] of cand.slice(0, 6)) if (wet(a, b)) sp.push(a, b, (WX[b] - WX[a]) / d, (WY[b] - WY[a]) / d, o.seaKeep * d * K);
    }
  }
  tick("prep");
  const last: LastPlaces = new Map();
  for (let i = 0; i < n; i++) last.set(g.key[i], [WX[i] * K, WY[i] * K, WX[i] * K, WY[i] * K]);
  // the layout about 0.6 of the time, the carry the rest
  const layout = layoutDiagram(g2, { ...dialled(0), puDiv: medH, sheet: { pairs: Float64Array.from(sp), wGeo: o.wGeo } }, last, progress && ((s) => progress(0.6 * s)), null);
  tick("layout");

  // ---- the carry
  const fin = sheetFinest(sheet), gw = fin.gw, gh = fin.gh, N = gw * gh, { h, x0, y0 } = sheet.mesh;
  const SX = Float64Array.from(layout.x, (v) => v / K), SY = Float64Array.from(layout.y, (v) => v / K);
  const rx = new Float64Array(N), ry = new Float64Array(N);
  const bn = new Int32Array(3 * n), bw = new Float64Array(3 * n);
  for (let s = 0; s < n; s++) {
    let fi = (landX[s] - x0) / h, fj = (landY[s] - y0) / h;
    const i = Math.min(gw - 2, Math.max(0, Math.floor(fi))), j = Math.min(gh - 2, Math.max(0, Math.floor(fj)));
    (fi = Math.min(1, Math.max(0, fi - i))), (fj = Math.min(1, Math.max(0, fj - j)));
    const a = j * gw + i, k = 3 * s;
    if (fi >= fj) (bn[k] = a), (bn[k + 1] = a + 1), (bn[k + 2] = a + gw + 1), (bw[k] = 1 - fi), (bw[k + 1] = fi - fj), (bw[k + 2] = fj);
    else (bn[k] = a), (bn[k + 1] = a + gw + 1), (bn[k + 2] = a + gw), (bw[k] = 1 - fj), (bw[k + 1] = fi), (bw[k + 2] = fj - fi);
  }
  const lamH = new Float64Array(N), lamV = new Float64Array(N);
  for (let j = 0; j < gh; j++) for (let i = 0; i < gw; i++) {
    const v = j * gw + i, x = x0 + i * h, y = y0 + j * h;
    if (i + 1 < gw) lamH[v] = massAt(R, x + h / 2, y) ? o.lamLand : o.lamWater;
    if (j + 1 < gh) lamV[v] = massAt(R, x, y + h / 2) ? o.lamLand : o.lamWater;
  }
  // each station's move: its drawn place less where the sheet puts its place on its land
  const dx = new Float64Array(n), dy = new Float64Array(n);
  for (let s = 0, k = 0; s < n; s++, k += 3) {
    const wx = bw[k] * fin.qx[bn[k]] + bw[k + 1] * fin.qx[bn[k + 1]] + bw[k + 2] * fin.qx[bn[k + 2]];
    const wy = bw[k] * fin.qy[bn[k]] + bw[k + 1] * fin.qy[bn[k + 1]] + bw[k + 2] * fin.qy[bn[k + 2]];
    (dx[s] = SX[s] - wx), (dy[s] = SY[s] - wy);
  }
  // the moves of both axes solved at once, interleaved (x then y per node): each pass over the matrix serves both
  const ws = new Float64Array(n).fill(1), U = new Float64Array(2 * N), rhs = new Float64Array(2 * N), U0y = new Float64Array(N);
  const carry: Carry = { gw, gh, x0, y0, h, rx, ry, sx: SX, sy: SY, ownMass, landX, landY, raster: R };
  const rounds = cityMap(g) ? o.carryCity : o.carryCountry;
  const bads: number[] = [], its: number[] = [];
  const tm = [0, 0, 0];
  const A = carryMatrix(gw, gh, lamH, lamV, bn, bw, ws);
  // the multigrid made once; a later round changes only a few stations' weights, and is made again only where its
  // solve does not settle in MG_STALE iterations
  const solver = (carrySolverMaker ?? jsCarrySolver)(gw, gh, A);
  let fresh = true;
  const touched: number[] = [];
  tm[0] += performance.now() - tl;
  const tol = o.carryTol;
  try {
    for (let round = 0; round <= rounds; round++) {
      let ta = performance.now();
      rhs.fill(0);
      for (let s = 0, k = 0; s < n; s++, k += 3) for (let c = 0; c < 3; c++) (rhs[2 * bn[k + c]] += ws[s] * bw[k + c] * dx[s]), (rhs[2 * bn[k + c] + 1] += ws[s] * bw[k + c] * dy[s]);
      // the axes as if solved one after the other: x first; where x does not settle under the old multigrid, it is made
      // again and y solved from where it was with the new one, else y goes on under the old one as far as it can
      if (!fresh) for (let v = 0; v < N; v++) U0y[v] = U[2 * v + 1];
      const it = solver.solve(rhs, U, tol, fresh ? 60 : MG_STALE, [true, true]);
      if (!fresh && (it[0] >= MG_STALE || it[1] >= MG_STALE)) {
        const tb = performance.now(), redoY = it[0] >= MG_STALE;
        solver.remake(), (fresh = true), (tm[0] += performance.now() - tb);
        if (redoY) {
          for (let v = 0; v < N; v++) U[2 * v + 1] = U0y[v];
          it[1] = 0;
        }
        const more = solver.solve(rhs, U, tol, 60, [it[0] >= MG_STALE, redoY || it[1] >= MG_STALE]);
        (it[0] += more[0]), (it[1] += more[1]);
      }
      its.push(it[0], it[1]);
      for (let v = 0; v < N; v++) (rx[v] = fin.qx[v] + U[2 * v]), (ry[v] = fin.qy[v] + U[2 * v + 1]);
      tm[1] += performance.now() - ta, (ta = performance.now());
      let bad = 0;
      for (let s = 0, k = 0; s < n; s++, k += 3) {
        if (!ownMass[s] || drawnMass(carry, s, 12) === ownMass[s]) continue;
        // its weight times wUp, in the matrix too
        const dw = ws[s] * (o.wUp - 1);
        ws[s] *= o.wUp;
        for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {
          const u = bn[k + p], w = bn[k + q], di = (w % gw) - (u % gw), dj = Math.floor(w / gw) - Math.floor(u / gw), at = u * 9 + (dj + 1) * 3 + di + 1;
          A[at] += dw * bw[k + p] * bw[k + q];
          touched.push(at);
        }
        bad++;
      }
      if (touched.length) solver.changed(touched), (touched.length = 0);
      bads.push(bad);
      progress?.(0.6 + (0.4 * (round + 1)) / (rounds + 1));
      fresh = false;
      tm[2] += performance.now() - ta;
      if (!bad) break;
    }
  } finally {
    solver.free();
  }
  tick("carry");
  (ms.carryBuild = Math.round(tm[0])), (ms.carrySolve = Math.round(tm[1])), (ms.carryCheck = Math.round(tm[2]));
  let offLand = 0;
  for (let s = 0; s < n; s++) if (ownMass[s] && drawnMass(carry, s, 20) !== ownMass[s]) offLand++;
  ms.total = Math.round(performance.now() - t0);
  return { graph: g2, k: K, layout, carry, info: { seaPairs: sp.length / 5, carryRounds: bads, carryIters: its, offLand, ms } };
}

/** the zones' outlines as spans along the raster's rows (each row's middle): per row its spans [a, b) of x, and zone */
interface ZoneSpans { at: Int32Array; a: Float64Array; b: Float64Array; z: Int32Array }
function zoneSpans(R: SheetRaster, shapes: ZoneShape[]): ZoneSpans {
  const hits: number[][] = Array.from({ length: R.h }, () => []);
  shapes.forEach((sh, zi) => {
    for (const r of sh.rings) for (let a = 0, m = r.length; a < m; a += 2) {
      const b = (a + 2) % m, ya = r[a + 1], yb = r[b + 1];
      if (ya === yb) continue;
      const j0 = Math.max(0, Math.ceil((Math.min(ya, yb) - R.y0) / R.rr - 0.5)), j1 = Math.min(R.h - 1, Math.floor((Math.max(ya, yb) - R.y0) / R.rr - 0.5));
      for (let j = j0; j <= j1; j++) { const y = R.y0 + (j + 0.5) * R.rr; if ((ya > y) !== (yb > y)) hits[j].push(zi, r[a] + ((y - ya) * (r[b] - r[a])) / (yb - ya)); }
    }
  });
  const at = new Int32Array(R.h + 1), A: number[] = [], B: number[] = [], Z: number[] = [];
  for (let j = 0; j < R.h; j++) {
    const byZ = new Map<number, number[]>(), hh = hits[j];
    for (let k = 0; k < hh.length; k += 2) { let l = byZ.get(hh[k]); if (!l) byZ.set(hh[k], (l = [])); l.push(hh[k + 1]); }
    for (const [zi, xs] of [...byZ].sort((p, q) => p[0] - q[0])) {
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) A.push(xs[k]), B.push(xs[k + 1]), Z.push(zi);
    }
    at[j + 1] = A.length;
  }
  return { at, a: Float64Array.from(A), b: Float64Array.from(B), z: Int32Array.from(Z) };
}

/** the sheet's water and zone borders: the ground carried through the carried map onto a grid over the drawing, each
 * station's pill on its own land and zone, cleaned, and the coasts and borders fitted at 0, 45 and 90 degrees (round 3
 * of T-2577). The sea only with `withWater`. */
export function sheetMap(lines: SheetLines, g: DiagramGraph, shapes: ZoneShape[], withWater: boolean, o: SheetLinesOptions = SHEET_LINES, detail = false, real: RealWater | null = null, th: ThinOptions = THIN): { water: DiagramWater | null; borders: DiagramBorders; info: Record<string, number>; grid?: { d: Uint8Array | Uint16Array; w: number; h: number; x0: number; y0: number; cell: number }; borderFits?: Float64Array[]; riverFits?: RiverBands["fits"] } {
  const t0 = performance.now();
  const C = lines.carry, R = C.raster, K = lines.k, n = g.key.length, SX = C.sx, SY = C.sy, { gw, gh, h, x0, y0, rx: RXn, ry: RYn } = C;
  // labels: WATER, a zone (its index plus 1), NOZ land in no zone, OUT off the carried map
  const wide = shapes.length > 250, OUT = wide ? 65534 : 254, NOZ = wide ? 65535 : 255;
  const zs = zoneSpans(R, shapes);
  // a land mass with no station that the masks' edge cuts across is drawn as sea: another country's coast at the edge
  // of the masks (Korea and China's coast round all Japan) would show as land ending in the masks' straight edges. Land
  // the masks hold whole stays, stations or not (Pennsylvania past the Delaware, Sado, Governors Island)
  const { cut: dropped, km2: cutKm2 } = cutLand(C);
  const labAt = (x: number, y: number) => {
    const m = massAt(R, x, y);
    if (!m || dropped[m - 1]) return 0;
    const j = Math.floor((y - R.y0) / R.rr);
    if (j < 0 || j >= R.h) return NOZ;
    for (let k = zs.at[j]; k < zs.at[j + 1]; k++) if (x >= zs.a[k] && x < zs.b[k]) return zs.z[k] + 1;
    return NOZ;
  };
  const zoneAll = stationZones(g, shapes), ownLab = new Int32Array(n);
  for (let s = 0; s < n; s++) ownLab[s] = zoneAll[s] >= 0 ? zoneAll[s] + 1 : NOZ;
  const isZ = (v: number) => v > 0 && v < OUT;

  // ---- the drawn grid: the real raster carried through the carried map
  let qx0 = Infinity, qy0 = Infinity, qx1 = -Infinity, qy1 = -Infinity;
  for (let v = 0; v < RXn.length; v++) (qx0 = Math.min(qx0, RXn[v])), (qx1 = Math.max(qx1, RXn[v])), (qy0 = Math.min(qy0, RYn[v])), (qy1 = Math.max(qy1, RYn[v]));
  const hd = Math.max(1, Math.sqrt(((qx1 - qx0) * (qy1 - qy0)) / (cityMap(g) ? o.cellsCity : o.cellsCountry)));
  const GW = Math.ceil((qx1 - qx0) / hd), GH = Math.ceil((qy1 - qy0) / hd), NC = GW * GH;
  const D = wide ? new Uint16Array(NC).fill(OUT) : new Uint8Array(NC).fill(OUT), DM = new Int32Array(NC).fill(-1);
  for (let j = 0; j + 1 < gh; j++) for (let i = 0; i + 1 < gw; i++) {
    const a = j * gw + i;
    for (let t = 0; t < 2; t++) {
      const p = a, q = t ? a + gw + 1 : a + 1, r = t ? a + gw : a + gw + 1;
      const x1 = RXn[p], y1 = RYn[p], x2 = RXn[q], y2 = RYn[q], x3 = RXn[r], y3 = RYn[r];
      const det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
      if (Math.abs(det) < 1e-12) continue;
      const c0 = Math.max(0, Math.floor((Math.min(x1, x2, x3) - qx0) / hd)), c1 = Math.min(GW - 1, Math.ceil((Math.max(x1, x2, x3) - qx0) / hd));
      const r0 = Math.max(0, Math.floor((Math.min(y1, y2, y3) - qy0) / hd)), r1 = Math.min(GH - 1, Math.ceil((Math.max(y1, y2, y3) - qy0) / hd));
      const pxr = x0 + (p % gw) * h, pyr = y0 + Math.floor(p / gw) * h, qxr = x0 + (q % gw) * h, qyr = y0 + Math.floor(q / gw) * h, sxr = x0 + (r % gw) * h, syr = y0 + Math.floor(r / gw) * h;
      for (let gj = r0; gj <= r1; gj++) for (let gi = c0; gi <= c1; gi++) {
        const X = qx0 + (gi + 0.5) * hd, Y = qy0 + (gj + 0.5) * hd;
        const l1 = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / det, l2 = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / det, l3 = 1 - l1 - l2;
        if (l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9) continue;
        const ux = l1 * pxr + l2 * qxr + l3 * sxr, uy = l1 * pyr + l2 * qyr + l3 * syr, c = gj * GW + gi;
        D[c] = labAt(ux, uy);
        DM[c] = massAt(R, ux, uy);
      }
    }
  }
  const t1 = performance.now();
  const cellOf = (x: number, y: number) => { const i = Math.floor((x - qx0) / hd), j = Math.floor((y - qy0) / hd); return i < 0 || j < 0 || i >= GW || j >= GH ? -1 : j * GW + i; };
  // stations whose drawn place is not on their own land in the carried map, and how far that land is
  const offLand: { s: number; dist: number }[] = [];
  for (let s = 0; s < n; s++) {
    if (!C.ownMass[s] || drawnMass(C, s, 20) === C.ownMass[s]) continue;
    let dist = -1;
    const ci = Math.floor((SX[s] - qx0) / hd), cj = Math.floor((SY[s] - qy0) / hd);
    for (let r = 1; r <= 80 && dist < 0; r++) for (let b = cj - r; b <= cj + r && dist < 0; b++) for (let a = ci - r; a <= ci + r; a++) {
      if (Math.max(Math.abs(a - ci), Math.abs(b - cj)) !== r || a < 0 || b < 0 || a >= GW || b >= GH) continue;
      if (DM[b * GW + a] === C.ownMass[s]) { dist = r * hd; break; }
    }
    offLand.push({ s, dist });
  }
  const pill = new Uint8Array(NC);
  const disk = (s: number, rad: number, fn: (c: number) => void) => {
    const ci = Math.floor((SX[s] - qx0) / hd), cj = Math.floor((SY[s] - qy0) / hd), rc = Math.ceil(rad / hd);
    for (let b = cj - rc; b <= cj + rc; b++) for (let a = ci - rc; a <= ci + rc; a++) {
      if (a < 0 || b < 0 || a >= GW || b >= GH) continue;
      if (Math.hypot((a - ci) * hd, (b - cj) * hd) <= Math.max(rad, 0.5 * hd)) fn(b * GW + a);
    }
  };
  // a station's pill and 1.5 pitches round it are land (at least half a pitch more than a drawn cell, so a coarse
  // grid's coast cannot pass over the station), of its own zone where it was water
  const PR = Math.max(1.5, 0.5 + hd);
  for (let s = 0; s < n; s++) disk(s, PR, (c) => { pill[c] = 1; if (D[c] === 0 || D[c] === OUT) D[c] = ownLab[s]; });
  // a station still off its own land has that land carried to it a pill wide from the nearest cell of it, within bendMax
  let bent = 0;
  for (const b of offLand) {
    if (b.dist < 0 || b.dist > o.bendMax) continue;
    const s = b.s, ci = Math.floor((SX[s] - qx0) / hd), cj = Math.floor((SY[s] - qy0) / hd);
    let best = -1, bd = Infinity;
    const r = Math.ceil(b.dist / hd) + 1;
    for (let jj = cj - r; jj <= cj + r; jj++) for (let ii = ci - r; ii <= ci + r; ii++) {
      if (ii < 0 || jj < 0 || ii >= GW || jj >= GH) continue;
      const c = jj * GW + ii;
      if (DM[c] === C.ownMass[s]) { const d = Math.hypot(ii - ci, jj - cj); if (d < bd) (bd = d), (best = c); }
    }
    if (best < 0) continue;
    const bx = best % GW, by = (best - bx) / GW, m = Math.ceil(bd * 2) + 1, rad = Math.max(1, Math.ceil(PR / hd));
    for (let t = 0; t <= m; t++) {
      const i0 = Math.round(ci + ((bx - ci) * t) / m), j0 = Math.round(cj + ((by - cj) * t) / m);
      for (let b2 = -rad; b2 <= rad; b2++) for (let a2 = -rad; a2 <= rad; a2++) {
        const i = i0 + a2, j = j0 + b2;
        if (i < 0 || j < 0 || i >= GW || j >= GH || Math.hypot(a2, b2) > rad) continue;
        const c = j * GW + i;
        (DM[c] = C.ownMass[s]), (pill[c] = 1), (D[c] = ownLab[s]);
      }
    }
    bent++;
  }
  // cleaned: a 3 by 3 majority
  {
    const old = D.slice();
    // the labels round a cell in the order first met, with their counts (the first of the most wins)
    const lab = new Int32Array(9), cnt = new Int32Array(9);
    for (let j = 1; j + 1 < GH; j++) for (let i = 1; i + 1 < GW; i++) {
      const c = j * GW + i, o0 = old[c];
      if (o0 === OUT || pill[c]) continue;
      let w = 0;
      for (let b = -GW; b <= GW; b += GW) for (let a = -1; a <= 1; a++) if (old[c + b + a] === 0) w++;
      if (o0 === 0 && w <= 3) {
        let m = 0;
        for (let b = -GW; b <= GW; b += GW) for (let a = -1; a <= 1; a++) {
          const v = old[c + b + a];
          if (v === 0 || v === OUT) continue;
          let k = 0;
          while (k < m && lab[k] !== v) k++;
          if (k === m) (lab[m] = v), (cnt[m++] = 0);
          cnt[k]++;
        }
        if (m) { let bv = 0, bc = -1; for (let k = 0; k < m; k++) if (cnt[k] > bc) (bc = cnt[k]), (bv = lab[k]); D[c] = bv; }
      } else if (o0 !== 0 && w >= 6) D[c] = 0;
    }
  }
  // narrow rivers a hop crosses as bands (with the water), the grid's own water under them taken out; thin water and
  // thin land drawn again as bands; nibs and notches smoothed (sheetThin.ts)
  const G: MapGrid = { D, w: GW, h: GH, cell: hd, x0: qx0, y0: qy0, OUT, NOZ, pill, band: new Uint8Array(NC) };
  const thinStats: ThinStats = { pieces: 0, bands: 0, skipped: 0, ms: 0 };
  const rivers = real && withWater ? riverBands(G, lines.graph, lines.layout, K, real, carryMap(C), th) : null;
  // on a coarse grid (a country's) a band is at least a cell and a half wide, and thin is four cells
  const thG: ThinOptions = { ...th, thinW: Math.max(th.thinW, 4 * hd), minW: Math.max(th.minW, 1.5 * hd), tolMax: Math.max(th.tolMax, 2 * hd) };
  // water under 40 square pitches and land with no station under 30 dropped, before the bands and after
  const dropSmall = () => {
    const comp = new Int32Array(NC).fill(-1), area: number[] = [], hasSt: number[] = [], wetK: boolean[] = [], qq = new Int32Array(NC);
    for (let s0 = 0; s0 < NC; s0++) {
      if (comp[s0] >= 0 || D[s0] === OUT) continue;
      const wet = D[s0] === 0, id = area.length;
      let head = 0, tail = 0, st = 0;
      (qq[tail++] = s0), (comp[s0] = id);
      const take = (u: number) => { if (comp[u] < 0 && D[u] !== OUT && (D[u] === 0) === wet) (comp[u] = id), (qq[tail++] = u); };
      while (head < tail) {
        const v = qq[head++], i = v % GW;
        if (pill[v]) st = 1;
        if (i > 0) take(v - 1);
        if (i < GW - 1) take(v + 1);
        if (v >= GW) take(v - GW);
        if (v + GW < NC) take(v + GW);
      }
      area.push(tail * hd * hd), hasSt.push(st), wetK.push(wet);
    }
    for (let c = 0; c < NC; c++) { const k = comp[c]; if (k < 0) continue; if (wetK[k] && area[k] < 40) D[c] = NOZ; else if (!wetK[k] && !hasSt[k] && area[k] < 30) D[c] = 0; }
    for (let pass = 0; pass < 3; pass++) for (let c = 0; c < NC; c++) {
      if (D[c] !== NOZ) continue;
      const i = c % GW;
      if (i > 0 && isZ(D[c - 1])) D[c] = D[c - 1];
      else if (i < GW - 1 && isZ(D[c + 1])) D[c] = D[c + 1];
      else if (c >= GW && isZ(D[c - GW])) D[c] = D[c - GW];
      else if (c + GW < NC && isZ(D[c + GW])) D[c] = D[c + GW];
    }
  };
  dropSmall();
  thinPass(G, true, thG, thinStats);
  thinPass(G, false, thG, thinStats);
  smoothEdges(G);
  dropSmall();
  // every station on its own zone's side: where its pill is in another zone, its own zone is carried to it along the
  // shortest way over land from the nearest of its own cells, a pill wide, and the pill painted its own
  let zoneFixed = 0;
  {
    const seen = new Int32Array(NC).fill(-1), from = new Int32Array(NC);
    for (let s = 0; s < n; s++) {
      const c0 = cellOf(SX[s], SY[s]), z = ownLab[s];
      if (c0 < 0 || !isZ(z) || D[c0] === z) continue;
      zoneFixed++;
      const qq = [c0];
      (seen[c0] = s), (from[c0] = -1);
      let hit = -1;
      for (let k = 0; k < qq.length && hit < 0 && k < 200000; k++) {
        const v = qq[k], i = v % GW;
        for (const u of [i > 0 ? v - 1 : -1, i < GW - 1 ? v + 1 : -1, v - GW, v + GW]) {
          if (u < 0 || u >= NC || seen[u] === s || D[u] === OUT || D[u] === 0) continue;
          (seen[u] = s), (from[u] = v);
          if (D[u] === z && !pill[u]) { hit = u; break; }
          qq.push(u);
        }
      }
      const rad = Math.max(1, Math.ceil(0.75 / hd));
      if (hit >= 0) for (let v = from[hit]; v >= 0; v = from[v]) {
        const i = v % GW, j = (v - i) / GW;
        for (let b = -rad; b <= rad; b++) for (let a = -rad; a <= rad; a++) {
          const u = (j + b) * GW + i + a;
          if (i + a >= 0 && i + a < GW && j + b >= 0 && j + b < GH && D[u] !== 0 && D[u] !== OUT && !(pill[u] && D[u] !== z)) D[u] = z;
        }
      }
      disk(s, 1.5, (c) => { if (D[c] !== 0 && D[c] !== OUT) D[c] = z; pill[c] = 1; });
      D[c0] = z;
    }
    for (let s = 0; s < n; s++) { const c = cellOf(SX[s], SY[s]); if (c >= 0 && isZ(ownLab[s])) (D[c] = ownLab[s]), (pill[c] = 1); }
  }
  const t2 = performance.now();

  // ---- the zones a line passes through: a station's, or one a hop's track crosses (T-2609)
  const held = new Uint8Array(OUT + 1).fill(o.allZones ? 1 : 0);
  for (let s = 0; s < n; s++) if (zoneAll[s] >= 0) held[zoneAll[s] + 1] = 1;
  for (let e = 0; e < g.ea.length && !o.allZones; e++) {
    const pts: ArrayLike<number> = g.hop && g.hopAt && g.hopAt[e + 1] - g.hopAt[e] >= 2 ? g.hop.subarray(g.hopAt[e] * 2, g.hopAt[e + 1] * 2) : [g.x[g.ea[e]], g.y[g.ea[e]], g.x[g.eb[e]], g.y[g.eb[e]]];
    for (let k = 0; k + 3 < pts.length; k += 2) {
      const m = Math.max(1, Math.ceil(Math.hypot(pts[k + 2] - pts[k], pts[k + 3] - pts[k + 1]) / R.rr));
      for (let t = 0; t <= m; t++) { const z = labAt(pts[k] + ((pts[k + 2] - pts[k]) * t) / m, pts[k + 1] + ((pts[k + 3] - pts[k + 1]) * t) / m); if (isZ(z)) held[z] = 1; }
    }
  }

  // ---- coasts and borders
  const wm = new Uint8Array(NC);
  for (let c = 0; c < NC; c++) wm[c] = +(D[c] === 0 || D[c] === OUT);
  // on a country map, water only where its nearest land is a zone a line passes through; the rest is left plain, its
  // edge fitted like a coast (T-2609)
  if (!o.allZones && !cityMap(g)) {
    const own = new Int8Array(NC).fill(-1), q = new Int32Array(NC);
    let qn = 0;
    for (let c = 0; c < NC; c++) if (!wm[c]) (own[c] = isZ(D[c]) && held[D[c]] ? 1 : 0), (q[qn++] = c);
    for (let k = 0; k < qn; k++) {
      const v = q[k], i = v % GW;
      for (const u of [i > 0 ? v - 1 : -1, i < GW - 1 ? v + 1 : -1, v - GW, v + GW]) if (u >= 0 && u < NC && wm[u] && own[u] < 0) (own[u] = own[v]), (q[qn++] = u);
    }
    for (let c = 0; c < NC; c++) if (wm[c] && own[c] !== 1) wm[c] = 0;
  }
  const stm = new Uint8Array(NC);
  for (let s = 0; s < n; s++) { const c = cellOf(SX[s], SY[s]); if (c >= 0) stm[c] = 1; }
  const dS = chamfer(stm, GW, GH, true, 60000);
  const VW = GW + 1, VX = (v: number) => qx0 + (v % VW) * hd, VY = (v: number) => qy0 + Math.floor(v / VW) * hd;
  const seaCmd: number[] = [], scaled = (f: Float64Array) => Float64Array.from(f, (v) => v * K);
  const seaRings: Float64Array[] = [];
  if (withWater) {
    // the coast's edges, water on the left, by vertex
    const eA: number[] = [], eB: number[] = [], outs = new Map<number, number[]>();
    const add = (a: number, b: number) => { let l = outs.get(a); if (!l) outs.set(a, (l = [])); l.push(eA.length); eA.push(a), eB.push(b); };
    const V = (i: number, j: number) => j * VW + i, wet = (i: number, j: number) => i >= 0 && j >= 0 && i < GW && j < GH && wm[j * GW + i] === 1;
    for (let j = 0; j < GH; j++) for (let i = -1; i < GW; i++) { const l = wet(i, j), r = wet(i + 1, j); if (l !== r) (l ? add(V(i + 1, j), V(i + 1, j + 1)) : add(V(i + 1, j + 1), V(i + 1, j))); }
    for (let j = -1; j < GH; j++) for (let i = 0; i < GW; i++) { const d = wet(i, j), u = wet(i, j + 1); if (d !== u) (u ? add(V(i, j + 1), V(i + 1, j + 1)) : add(V(i + 1, j + 1), V(i, j + 1))); }
    const used = new Uint8Array(eA.length);
    const run = (x: number, y: number, ux: number, uy: number, wetW: boolean) => { let t = 0; for (;;) { if (t >= 40) break; const c = cellOf(x + ux * t, y + uy * t); if (!(c >= 0 && (wm[c] === 1) === wetW)) break; t += 0.5 * hd; } return t; };
    for (let e0 = 0; e0 < eA.length; e0++) {
      if (used[e0]) continue;
      const ring = [eA[e0]];
      let e = e0;
      for (;;) {
        used[e] = 1;
        const a = eA[e], b = eB[e];
        ring.push(b);
        if (b === ring[0]) break;
        const opts = (outs.get(b) ?? []).filter((f) => !used[f]);
        if (!opts.length) break;
        const ax = VX(b) - VX(a), ay = VY(b) - VY(a), turn = (f: number) => (VX(eB[f]) - VX(b)) * ay - (VY(eB[f]) - VY(b)) * ax;
        opts.sort((p, q) => (turn(p) < turn(q) ? 1 : -1));
        e = opts[0];
      }
      if (ring.length < 5) continue;
      const px = ring.map(VX), py = ring.map(VY), m = ring.length - 1;
      // landward, at most 0.35 of the land's width behind and clear of the pills; seaward, 0.35 of the water's ahead;
      // at least a cell's step either way where that is narrow enough (water on the ring's left)
      const tolL: number[] = [], tolS: number[] = [];
      ring.forEach((v, k) => {
        const a2 = ring[(k - 3 + m) % m], b2 = ring[(k + 3) % m], tx = VX(b2) - VX(a2), ty = VY(b2) - VY(a2), tln = Math.hypot(tx, ty) || 1, nx = -ty / tln, ny = tx / tln;
        const x = VX(v), y = VY(v), ww = run(x + nx * 0.25 * hd, y + ny * 0.25 * hd, nx, ny, true), wl = run(x - nx * 0.25 * hd, y - ny * 0.25 * hd, -nx, -ny, false);
        const i = v % VW, j = Math.floor(v / VW), c = Math.min(GH - 1, j) * GW + Math.min(GW - 1, i), clear = (dS[c] / 3) * hd, step = th.tolCell * hd;
        tolL.push(Math.max(Math.min(Math.max(0.2, clear - (PR - 0.3)), Math.max(0.25 * hd, Math.min(step, 0.45 * wl))), Math.min(10, 0.35 * wl, clear - PR)));
        tolS.push(Math.max(0.25 * hd, Math.min(step, 0.45 * ww), Math.min(10, 0.35 * ww)));
      });
      const fit = scaled(fitOcti(px, py, tolL, undefined, { tol: tolS, left: true }));
      rounded(fit, true, 3 * K, seaCmd);
      seaRings.push(fit);
    }
  }
  const t3 = performance.now();
  const borderCmd: number[] = [], borderFits: Float64Array[] = [];
  let bends = 0, wrongSide = 0;
  {
    const adj = new Map<number, number[]>(), V = (i: number, j: number) => j * VW + i;
    const zl = (i: number, j: number) => (i < 0 || j < 0 || i >= GW || j >= GH ? 0 : D[j * GW + i]);
    const link = (a: number, b: number) => { let l = adj.get(a); if (!l) adj.set(a, (l = [])); l.push(b); let k = adj.get(b); if (!k) adj.set(b, (k = [])); k.push(a); };
    const edge = (p: number, q: number) => p !== q && isZ(p) && isZ(q) && (held[p] === 1 || held[q] === 1);
    for (let j = 0; j < GH; j++) for (let i = 0; i + 1 < GW; i++) if (edge(zl(i, j), zl(i + 1, j))) link(V(i + 1, j), V(i + 1, j + 1));
    for (let j = 0; j + 1 < GH; j++) for (let i = 0; i < GW; i++) if (edge(zl(i, j), zl(i, j + 1))) link(V(i, j + 1), V(i + 1, j + 1));
    const coastal = (v: number) => { const i = v % VW, j = Math.floor(v / VW); return !isZ(zl(i - 1, j - 1)) || !isZ(zl(i, j - 1)) || !isZ(zl(i - 1, j)) || !isZ(zl(i, j)); };
    const isEnd = (v: number) => adj.get(v)!.length !== 2 || coastal(v);
    const used = new Set<number>(), NV = VW * (GH + 1), ek = (a: number, b: number) => (a < b ? a * NV + b : b * NV + a);
    // the stations by 16-pitch bucket, for the fit's checks
    const BK = 16, bucket = new Map<number, number[]>(), bkey = (i: number, j: number) => i * 65536 + j;
    for (let s = 0; s < n; s++) { const k = bkey(Math.floor(SX[s] / BK), Math.floor(SY[s] / BK)); let l = bucket.get(k); if (!l) bucket.set(k, (l = [])); l.push(s); }
    const near = (x0: number, y0: number, x1: number, y1: number, fn: (s: number) => boolean) => {
      for (let bj = Math.floor(y0 / BK); bj <= Math.floor(y1 / BK); bj++) for (let bi = Math.floor(x0 / BK); bi <= Math.floor(x1 / BK); bi++) for (const s of bucket.get(bkey(bi, bj)) ?? []) if (SX[s] >= x0 && SX[s] <= x1 && SY[s] >= y0 && SY[s] <= y1 && fn(s)) return true;
      return false;
    };
    /** whether (x, y) is inside the polygon of points px, py from i to j (and the bend), even-odd */
    const inside = (px: number[], py: number[], i: number, j: number, bx: number, by: number, x: number, y: number) => {
      let c = false, ax = bx === bx ? bx : px[j], ay = bx === bx ? by : py[j];
      const step = (qx: number, qy: number) => { if (qy > y !== ay > y && x < ((ax - qx) * (y - qy)) / (ay - qy) + qx) c = !c; (ax = qx), (ay = qy); };
      for (let k = i; k <= j; k++) step(px[k], py[k]);
      if (bx === bx) step(bx, by);
      return c;
    };
    const walk = (s: number, nx: number) => {
      const ch = [s];
      let a = s, b = nx;
      for (;;) {
        used.add(ek(a, b)), ch.push(b);
        if (isEnd(b) || b === s) break;
        const c = adj.get(b)!.find((c2) => !used.has(ek(b, c2)));
        if (c === undefined) break;
        (a = b), (b = c);
      }
      return ch;
    };
    const chains: number[][] = [];
    for (const [v, ns] of adj) if (isEnd(v)) for (const u of ns) if (!used.has(ek(v, u))) chains.push(walk(v, u));
    for (const [v, ns] of adj) for (const u of ns) if (!used.has(ek(v, u))) chains.push(walk(v, u));
    // each chain fitted straying up to borderTol pitches (T-2629: one long zig and one long zag over many short
    // steps), a way kept only where it leaves every station on its side, at least a pitch from it (or as far as the
    // traced border was), and runs over the chain's own zones' land
    const tol = Math.max(0.2 * hd, o.borderTol);
    for (const ch of chains) {
      if (ch.length < 3) continue;
      const px = ch.map(VX), py = ch.map(VY), labs = new Set<number>();
      for (let k = 0; k + 1 < ch.length; k++) {
        const a = ch[k], b = ch[k + 1], ia = a % VW, ja = (a - ia) / VW, ib = b % VW, jb = (b - ib) / VW;
        if (ja === jb) labs.add(zl(Math.min(ia, ib), ja - 1)), labs.add(zl(Math.min(ia, ib), ja));
        else labs.add(zl(ia - 1, Math.min(ja, jb))), labs.add(zl(ia, Math.min(ja, jb)));
      }
      const keeps = (i: number, j: number, bx: number, by: number) => {
        const bent = bx === bx, wx = bent ? [px[i], bx, px[j]] : [px[i], px[j]], wy = bent ? [py[i], by, py[j]] : [py[i], py[j]];
        // on the chain's zones: samples along the way, half a cell apart, but a cell and a half from its ends
        for (let k = 0; k + 1 < wx.length; k++) {
          const L = Math.hypot(wx[k + 1] - wx[k], wy[k + 1] - wy[k]), m = Math.ceil(L / (0.5 * hd));
          for (let t = 0; t <= m; t++) {
            const x = wx[k] + ((wx[k + 1] - wx[k]) * t) / m, y = wy[k] + ((wy[k + 1] - wy[k]) * t) / m;
            if (Math.hypot(x - px[i], y - py[i]) < 1.5 * hd || Math.hypot(x - px[j], y - py[j]) < 1.5 * hd) continue;
            const c = cellOf(x, y);
            if (c < 0 || !labs.has(D[c])) return false;
          }
        }
        let x0 = bent ? bx : Infinity, y0 = bent ? by : Infinity, x1 = bent ? bx : -Infinity, y1 = bent ? by : -Infinity;
        for (let k = i; k <= j; k++) (x0 = Math.min(x0, px[k])), (x1 = Math.max(x1, px[k])), (y0 = Math.min(y0, py[k])), (y1 = Math.max(y1, py[k]));
        return !near(x0 - 1, y0 - 1, x1 + 1, y1 + 1, (s) => {
          const x = SX[s], y = SY[s];
          let dw = Infinity;
          for (let k = 0; k + 1 < wx.length; k++) dw = Math.min(dw, segD(x, y, wx[k], wy[k], wx[k + 1], wy[k + 1]));
          if (dw < 1) {
            let dr = Infinity;
            for (let k = i; k < j; k++) dr = Math.min(dr, segD(x, y, px[k], py[k], px[k + 1], py[k + 1]));
            if (dw < Math.min(1, dr) - 1e-9) return true;
          }
          return inside(px, py, i, j, bx, by, x, y);
        });
      };
      const fit = fitOcti(px, py, Array<number>(ch.length).fill(tol), keeps);
      // stations between the traced chain and its fit, on the wrong side (none, by keeps; counted as a check)
      if (detail) {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (let k = 0; k < ch.length; k++) (x0 = Math.min(x0, px[k])), (x1 = Math.max(x1, px[k])), (y0 = Math.min(y0, py[k])), (y1 = Math.max(y1, py[k]));
        const qx = [...px], qy = [...py];
        for (let k = fit.length - 4; k >= 2; k -= 2) qx.push(fit[k]), qy.push(fit[k + 1]);
        near(x0 - tol, y0 - tol, x1 + tol, y1 + tol, (s) => { if (inside(qx, qy, 0, qx.length - 1, NaN, NaN, SX[s], SY[s])) wrongSide++; return false; });
      }
      const bf = scaled(fit);
      bends += Math.max(0, bf.length / 2 - 2);
      if (detail) borderFits.push(bf);
      rounded(bf, ch[0] === ch[ch.length - 1], 3 * K, borderCmd);
    }
  }
  // the zones drawn, in the order their first station comes
  const drawnOf = new Int32Array(shapes.length).fill(-1), zones: string[] = [];
  for (let s = 0; s < n; s++) if (zoneAll[s] >= 0 && drawnOf[zoneAll[s]] < 0) (drawnOf[zoneAll[s]] = zones.length), zones.push(shapes[zoneAll[s]].id);
  const zoneOf = Int32Array.from(zoneAll, (z) => (z < 0 ? -1 : drawnOf[z]));
  let wrongZone = 0;
  for (let s = 0; s < n; s++) { const c = cellOf(SX[s], SY[s]); if (c >= 0 && isZ(ownLab[s]) && D[c] !== ownLab[s]) wrongZone++; }
  const t4 = performance.now();
  const stats = { arcs: rivers?.arcs ?? 0, crossings: rivers?.crossings ?? 0, leftOut: 0, turned: 0, sections: 0, widened: 0, noRoute: 0, falseCross: 0, rivers: rivers?.fits.length ?? 0, dropped: 0, uncrossed: 0, mouths: 0, dry: 0, open: 0, channels: 0, ms: Math.round(t3 - t0) };
  // the rivers' bands per width, in layout units
  const widths = rivers ? [...rivers.cmds.keys()].sort((a, b) => a - b) : [];
  const water: DiagramWater | null = withWater
    ? { rivers: widths.map((w) => Float64Array.from(rivers!.cmds.get(w)!)), bands: widths.map((w) => w * K), riverEnds: widths.map((w) => Float64Array.from(rivers!.ends.get(w) ?? [])), sea: Float64Array.from(seaCmd), seaRings, stats }
    : null;
  return {
    water,
    borders: { cmds: Float64Array.from(borderCmd), zones, zoneOf },
    riverFits: detail ? rivers?.fits : undefined,
    info: { cell: +hd.toFixed(2), cells: NC, cutKm2: Math.round(cutKm2), offLand: offLand.length, bent, zoneFixed, wrongZone, wrongSide, bends, rings: seaRings.length, riverBands: rivers?.fits.length ?? 0, thinPieces: thinStats.pieces, thinBands: thinStats.bands, thinSkipped: thinStats.skipped, msThin: Math.round(thinStats.ms), msGrid: Math.round(t1 - t0), msClean: Math.round(t2 - t1), msCoast: Math.round(t3 - t2), msBorders: Math.round(t4 - t3) },
    grid: detail ? { d: D, w: GW, h: GH, x0: qx0, y0: qy0, cell: hd } : undefined,
    borderFits: detail ? borderFits : undefined,
  };
}

/** the land masses (per `mass` number, 1) drawn as sea: no station on them and the masks' edge cutting them for at
 * least CUT_WIDTH of their width; and their area, km2 */
export function cutLand(C: Carry): { cut: Uint8Array; km2: number } {
  const R = C.raster, N = R.mass.length, area = new Float64Array(N), has = new Uint8Array(N), out = new Uint8Array(N), { cut } = massCoasts(R);
  for (let r = 0; r < N; r++) area[R.mass[r]] += (R.runB[r] - R.runA[r]) * R.rr * R.rr;
  for (let s = 0; s < C.ownMass.length; s++) if (C.ownMass[s]) has[C.ownMass[s] - 1] = 1;
  let km2 = 0;
  for (let m = 0; m < N; m++) if (area[m] && !has[m] && cut[m] * R.rr >= CUT_WIDTH * Math.sqrt(area[m])) (out[m] = 1), (km2 += area[m] / 1e6);
  return { cut: out, km2 };
}

/** the carried map at a place on the ground (graph metres), in pitches, through its lattice's triangles */
export function carryMap(C: Carry): CarryMap {
  const { gw, gh, h, x0, y0, rx, ry } = C;
  const tri = (x: number, y: number) => {
    let fi = (x - x0) / h, fj = (y - y0) / h;
    const i = Math.min(gw - 2, Math.max(0, Math.floor(fi))), j = Math.min(gh - 2, Math.max(0, Math.floor(fj)));
    (fi -= i), (fj -= j);
    return { a: j * gw + i, fi, fj };
  };
  return {
    at(x, y) {
      const { a, fi, fj } = tri(x, y), b = a + 1, c = a + gw + 1, d = a + gw;
      if (fi >= fj) return [(1 - fi) * rx[a] + (fi - fj) * rx[b] + fj * rx[c], (1 - fi) * ry[a] + (fi - fj) * ry[b] + fj * ry[c]];
      return [(1 - fj) * rx[a] + fi * rx[c] + (fj - fi) * rx[d], (1 - fj) * ry[a] + fi * ry[c] + (fj - fi) * ry[d]];
    },
  };
}

/** the land mass (number plus 1, 0 water) under station s's drawn place in the carried map: the place found in the
 * carried triangles within `wn` cells of its place on its own land; its own mass if any triangle there gives it; -1 if
 * none holds it */
export function drawnMass(C: Carry, s: number, wn: number): number {
  const { gw, gh, h, x0, y0, rx, ry } = C, X = C.sx[s], Y = C.sy[s];
  const ci = Math.floor((C.landX[s] - x0) / h), cj = Math.floor((C.landY[s] - y0) / h);
  let found = -1;
  for (let j = Math.max(0, cj - wn); j < Math.min(gh - 1, cj + wn); j++) for (let i = Math.max(0, ci - wn); i < Math.min(gw - 1, ci + wn); i++) {
    const a = j * gw + i;
    for (let t = 0; t < 2; t++) {
      const p = a, q = t ? a + gw + 1 : a + 1, r = t ? a + gw : a + gw + 1;
      const x1 = rx[p], y1 = ry[p], x2 = rx[q], y2 = ry[q], x3 = rx[r], y3 = ry[r], det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
      if (Math.abs(det) < 1e-12) continue;
      const l1 = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / det, l2 = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / det, l3 = 1 - l1 - l2;
      if (l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9) continue;
      const pi = p % gw, pj = (p - pi) / gw, qi = q % gw, qj = (q - qi) / gw, ri = r % gw, rj = (r - ri) / gw;
      const m = massAt(C.raster, x0 + h * (l1 * pi + l2 * qi + l3 * ri), y0 + h * (l1 * pj + l2 * qj + l3 * rj));
      if (m === C.ownMass[s]) return m;
      found = m;
    }
  }
  return found;
}

/** the carry's solver over the matrix A it was made with (its multigrid made with it): `changed` after the caller
 * changed A's entries `at`, `remake` its multigrid from A as it is, `solve` as pcg9 below */
export interface CarrySolver {
  changed(at: number[]): void;
  remake(): void;
  solve(b: Float64Array, x: Float64Array, tol: number, maxIt: number, on: [boolean, boolean]): [number, number];
  free(): void;
}
export type CarrySolverMaker = (gw: number, gh: number, A: Float64Array) => CarrySolver;

function jsCarrySolver(gw: number, gh: number, A: Float64Array): CarrySolver {
  // the reweighting only scales the stations' stencils, so which nodes have corner terms stays as it is
  const corner = cornersOf(A);
  let mg = multigrid(gw, gh, A);
  return { changed() {}, remake: () => void (mg = multigrid(gw, gh, A)), solve: (b, x, tol, maxIt, on) => pcg9(gw, gh, A, corner, b, x, mg, tol, maxIt, on), free() {} };
}
/** the solver sheetLines uses: the JavaScript's below unless the diagram worker sets the WASM one (carryWasm.ts,
 * sim/src/diagram.rs), which gives the same bits faster (T-2589) */
let carrySolverMaker: CarrySolverMaker | null = null;
export const setCarrySolver = (make: CarrySolverMaker | null) => void (carrySolverMaker = make);

// ---- the carry's least squares: a 9-point matrix on the lattice (slot (dj + 1) * 3 + di + 1 per node), solved by
// conjugate gradients with a multigrid V-cycle (operator-dependent prolongation, Galerkin coarse levels, Gauss-Seidel),
// both axes at once: vectors hold x and y per node, interleaved. A node's terms are summed in one fixed order (itself,
// west, east, the row below, the row above), the same on the lattice's edges with the terms off it left out; inside it
// the loops need no tests.
function carryMatrix(gw: number, gh: number, lamH: Float64Array, lamV: Float64Array, bn: Int32Array, bw: Float64Array, ws: Float64Array): Float64Array {
  const N = gw * gh, A = new Float64Array(N * 9);
  for (let v = 0; v < N; v++) {
    const a = lamH[v], b = lamV[v];
    if (a) (A[v * 9 + 4] += a), (A[(v + 1) * 9 + 4] += a), (A[v * 9 + 5] -= a), (A[(v + 1) * 9 + 3] -= a);
    if (b) (A[v * 9 + 4] += b), (A[(v + gw) * 9 + 4] += b), (A[v * 9 + 7] -= b), (A[(v + gw) * 9 + 1] -= b);
  }
  for (let s = 0, k = 0; s < ws.length; s++, k += 3) for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {
    const u = bn[k + p], w = bn[k + q], di = (w % gw) - (u % gw), dj = Math.floor(w / gw) - Math.floor(u / gw);
    A[u * 9 + (dj + 1) * 3 + di + 1] += ws[s] * bw[k + p] * bw[k + q];
  }
  return A;
}
/** y = A x at node (i, j) on the lattice's edge, or from - A x */
function mulEdge(nx: number, ny: number, A: Float64Array, x: Float64Array, y: Float64Array, i: number, j: number, from: Float64Array | null) {
  const v = j * nx + i, b = v * 9, p = 2 * v, q = 2 * nx;
  let sx = A[b + 4] * x[p], sy = A[b + 4] * x[p + 1];
  if (i > 0) (sx += A[b + 3] * x[p - 2]), (sy += A[b + 3] * x[p - 1]);
  if (i + 1 < nx) (sx += A[b + 5] * x[p + 2]), (sy += A[b + 5] * x[p + 3]);
  if (j > 0) {
    (sx += A[b + 1] * x[p - q]), (sy += A[b + 1] * x[p - q + 1]);
    if (i > 0) (sx += A[b] * x[p - q - 2]), (sy += A[b] * x[p - q - 1]);
    if (i + 1 < nx) (sx += A[b + 2] * x[p - q + 2]), (sy += A[b + 2] * x[p - q + 3]);
  }
  if (j + 1 < ny) {
    (sx += A[b + 7] * x[p + q]), (sy += A[b + 7] * x[p + q + 1]);
    if (i > 0) (sx += A[b + 6] * x[p + q - 2]), (sy += A[b + 6] * x[p + q - 1]);
    if (i + 1 < nx) (sx += A[b + 8] * x[p + q + 2]), (sy += A[b + 8] * x[p + q + 3]);
  }
  if (from) (y[p] = from[p] - sx), (y[p + 1] = from[p + 1] - sy);
  else (y[p] = sx), (y[p + 1] = sy);
}
/** per node whether any of its corner terms is not 0 (on the finest level only stations' nodes have them: a term of 0
 * left out adds nothing) */
function cornersOf(A: Float64Array): Uint8Array {
  const N = A.length / 9, f = new Uint8Array(N);
  for (let v = 0, b = 0; v < N; v++, b += 9) f[v] = +(A[b] !== 0 || A[b + 2] !== 0 || A[b + 6] !== 0 || A[b + 8] !== 0);
  return f;
}
function mul9(nx: number, ny: number, A: Float64Array, x: Float64Array, y: Float64Array, corner: Uint8Array, from: Float64Array | null = null) {
  const q = 2 * nx;
  for (let j = 0; j < ny; j++) {
    if (j === 0 || j + 1 === ny || nx < 3) {
      for (let i = 0; i < nx; i++) mulEdge(nx, ny, A, x, y, i, j, from);
      continue;
    }
    mulEdge(nx, ny, A, x, y, 0, j, from);
    for (let v = j * nx + 1, b = v * 9, p = 2 * v, end = j * nx + nx - 1; v < end; v++, b += 9, p += 2) {
      const a1 = A[b + 1], a3 = A[b + 3], a4 = A[b + 4], a5 = A[b + 5], a7 = A[b + 7];
      let sx = a4 * x[p], sy = a4 * x[p + 1];
      (sx += a3 * x[p - 2]), (sy += a3 * x[p - 1]);
      (sx += a5 * x[p + 2]), (sy += a5 * x[p + 3]);
      (sx += a1 * x[p - q]), (sy += a1 * x[p - q + 1]);
      if (corner[v]) {
        const a0 = A[b], a2 = A[b + 2], a6 = A[b + 6], a8 = A[b + 8];
        (sx += a0 * x[p - q - 2]), (sy += a0 * x[p - q - 1]);
        (sx += a2 * x[p - q + 2]), (sy += a2 * x[p - q + 3]);
        (sx += a7 * x[p + q]), (sy += a7 * x[p + q + 1]);
        (sx += a6 * x[p + q - 2]), (sy += a6 * x[p + q - 1]);
        (sx += a8 * x[p + q + 2]), (sy += a8 * x[p + q + 3]);
      } else (sx += a7 * x[p + q]), (sy += a7 * x[p + q + 1]);
      if (from) (y[p] = from[p] - sx), (y[p + 1] = from[p + 1] - sy);
      else (y[p] = sx), (y[p + 1] = sy);
    }
    mulEdge(nx, ny, A, x, y, nx - 1, j, from);
  }
}
/** one Gauss-Seidel step at node (i, j) on the lattice's edge */
function gsEdge(nx: number, ny: number, A: Float64Array, x: Float64Array, r: Float64Array, i: number, j: number) {
  const v = j * nx + i, b = v * 9, p = 2 * v, q = 2 * nx;
  let sx = r[p], sy = r[p + 1];
  if (i > 0) (sx -= A[b + 3] * x[p - 2]), (sy -= A[b + 3] * x[p - 1]);
  if (i + 1 < nx) (sx -= A[b + 5] * x[p + 2]), (sy -= A[b + 5] * x[p + 3]);
  if (j > 0) {
    (sx -= A[b + 1] * x[p - q]), (sy -= A[b + 1] * x[p - q + 1]);
    if (i > 0) (sx -= A[b] * x[p - q - 2]), (sy -= A[b] * x[p - q - 1]);
    if (i + 1 < nx) (sx -= A[b + 2] * x[p - q + 2]), (sy -= A[b + 2] * x[p - q + 3]);
  }
  if (j + 1 < ny) {
    (sx -= A[b + 7] * x[p + q]), (sy -= A[b + 7] * x[p + q + 1]);
    if (i > 0) (sx -= A[b + 6] * x[p + q - 2]), (sy -= A[b + 6] * x[p + q - 1]);
    if (i + 1 < nx) (sx -= A[b + 8] * x[p + q + 2]), (sy -= A[b + 8] * x[p + q + 3]);
  }
  const d = A[b + 4];
  (x[p] = d > 0 ? sx / d : 0), (x[p + 1] = d > 0 ? sy / d : 0);
}
function gs9(nx: number, ny: number, A: Float64Array, x: Float64Array, r: Float64Array, fwd: boolean, corner: Uint8Array) {
  const q = 2 * nx;
  for (let jj = 0; jj < ny; jj++) {
    const j = fwd ? jj : ny - 1 - jj;
    if (j === 0 || j + 1 === ny || nx < 3) {
      for (let ii = 0; ii < nx; ii++) gsEdge(nx, ny, A, x, r, fwd ? ii : nx - 1 - ii, j);
      continue;
    }
    // inside the row (first to last, or back) the terms off the lattice need no tests, and a node with no corner terms
    // skips them
    const v0 = fwd ? j * nx + 1 : j * nx + nx - 2, dv = fwd ? 1 : -1;
    gsEdge(nx, ny, A, x, r, fwd ? 0 : nx - 1, j);
    for (let k = 0, v = v0; k < nx - 2; k++, v += dv) {
      const b = v * 9, p = 2 * v;
      const a1 = A[b + 1], a3 = A[b + 3], a5 = A[b + 5], a7 = A[b + 7], d = A[b + 4];
      let sx = r[p], sy = r[p + 1];
      (sx -= a3 * x[p - 2]), (sy -= a3 * x[p - 1]);
      (sx -= a5 * x[p + 2]), (sy -= a5 * x[p + 3]);
      (sx -= a1 * x[p - q]), (sy -= a1 * x[p - q + 1]);
      if (corner[v]) {
        const a0 = A[b], a2 = A[b + 2], a6 = A[b + 6], a8 = A[b + 8];
        (sx -= a0 * x[p - q - 2]), (sy -= a0 * x[p - q - 1]);
        (sx -= a2 * x[p - q + 2]), (sy -= a2 * x[p - q + 3]);
        (sx -= a7 * x[p + q]), (sy -= a7 * x[p + q + 1]);
        (sx -= a6 * x[p + q - 2]), (sy -= a6 * x[p + q - 1]);
        (sx -= a8 * x[p + q + 2]), (sy -= a8 * x[p + q + 3]);
      } else (sx -= a7 * x[p + q]), (sy -= a7 * x[p + q + 1]);
      (x[p] = d > 0 ? sx / d : 0), (x[p + 1] = d > 0 ? sy / d : 0);
    }
    gsEdge(nx, ny, A, x, r, fwd ? nx - 1 : 0, j);
  }
}
/** one V-cycle of the levels from (nx, ny, A) as a preconditioner: z from r */
function multigrid(nx0: number, ny0: number, A0: Float64Array): (r: Float64Array, z: Float64Array) => void {
  // per level its matrix, and (but the coarsest) each node's 4 coarse parents and weights (pc, pw; unused slots weight 0,
  // after the node's pn used ones)
  const lev: { nx: number; ny: number; A: Float64Array; corner: Uint8Array; x: Float64Array; b: Float64Array; r: Float64Array; pc: Int32Array; pw: Float64Array; pn: Uint8Array }[] = [];
  let nx = nx0, ny = ny0, A = A0;
  for (;;) {
    const N = nx * ny, L = { nx, ny, A, corner: cornersOf(A), x: new Float64Array(2 * N), b: new Float64Array(2 * N), r: new Float64Array(2 * N), pc: new Int32Array(0), pw: new Float64Array(0), pn: new Uint8Array(0) };
    lev.push(L);
    if (nx <= 5 && ny <= 5) break;
    const cx = (nx >> 1) + 1, cy = (ny >> 1) + 1, C = new Float64Array(cx * cy * 9), pc = new Int32Array(4 * N), pw = new Float64Array(4 * N);
    // the prolongation follows the matrix (black box multigrid, Dendy 1982): a node between two coarse ones takes them by
    // its stencil's couplings toward each, collapsed across; a node between four, by its own stencil over its
    // neighbours as they are taken. So a land mass coupled to its neighbours only through water is moved as one.
    const at = (i: number, j: number, k: number) => A[(j * nx + i) * 9 + k];
    const line = (i: number, j: number, horiz: boolean): [number, number] => {
      // past the lattice's last node (an even count: the coarse level's last node has no fine one)
      if (i >= nx || j >= ny) return [0, 0];
      const c = horiz ? at(i, j, 1) + at(i, j, 4) + at(i, j, 7) : at(i, j, 3) + at(i, j, 4) + at(i, j, 5);
      const lo = horiz ? at(i, j, 0) + at(i, j, 3) + at(i, j, 6) : at(i, j, 0) + at(i, j, 1) + at(i, j, 2);
      const hi = horiz ? at(i, j, 2) + at(i, j, 5) + at(i, j, 8) : at(i, j, 6) + at(i, j, 7) + at(i, j, 8);
      return c > 1e-300 ? [-lo / c, -hi / c] : [0.5, 0.5];
    };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = j * nx + i, b = 4 * v, I = i >> 1, J = j >> 1, oi = i & 1, oj = j & 1;
      if (!oi && !oj) (pc[b] = J * cx + I), (pw[b] = 1);
      else if (oi && !oj) { const [w0, w1] = line(i, j, true); (pc[b] = J * cx + I), (pw[b] = w0), (pc[b + 1] = J * cx + I + 1), (pw[b + 1] = w1); }
      else if (!oi && oj) { const [w0, w1] = line(i, j, false); (pc[b] = J * cx + I), (pw[b] = w0), (pc[b + 1] = (J + 1) * cx + I), (pw[b + 1] = w1); }
      else {
        // corners SW, SE, NW, NE; the edge neighbours W, E (vertical lines), S, N (horizontal) by their own weights
        const c = at(i, j, 4), [wS, wN] = line(i - 1, j, false), [eS, eN] = line(i + 1, j, false), [sW, sE] = line(i, j - 1, true), [nW, nE] = line(i, j + 1, true);
        const aW = at(i, j, 3), aE = at(i, j, 5), aS = at(i, j, 1), aN = at(i, j, 7);
        const k = c > 1e-300 ? -1 / c : 0;
        (pc[b] = J * cx + I), (pw[b] = k * (at(i, j, 0) + aW * wS + aS * sW));
        (pc[b + 1] = J * cx + I + 1), (pw[b + 1] = k * (at(i, j, 2) + aE * eS + aS * sE));
        (pc[b + 2] = (J + 1) * cx + I), (pw[b + 2] = k * (at(i, j, 6) + aW * wN + aN * nW));
        (pc[b + 3] = (J + 1) * cx + I + 1), (pw[b + 3] = k * (at(i, j, 8) + aE * eN + aN * nE));
        if (c <= 1e-300) for (let q = 0; q < 4; q++) pw[b + q] = 0.25;
      }
    }
    // C = P^T A P
    for (let v = 0; v < N; v++) {
      for (let k = 0; k < 9; k++) {
        const a = A[v * 9 + k];
        if (!a) continue;
        const w = v + (Math.floor(k / 3) - 1) * nx + (k % 3) - 1;
        for (let p = 0; p < 4; p++) {
          const wu = pw[4 * v + p] * a;
          if (!wu) continue;
          const U = pc[4 * v + p], UI = U % cx, UJ = (U - UI) / cx;
          for (let p2 = 0; p2 < 4; p2++) {
            const ww = pw[4 * w + p2];
            if (!ww) continue;
            const U2 = pc[4 * w + p2], I2 = U2 % cx, J2 = (U2 - I2) / cx;
            C[U * 9 + (J2 - UJ + 1) * 3 + (I2 - UI + 1)] += wu * ww;
          }
        }
      }
    }
    const pn = new Uint8Array(N);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) pn[j * nx + i] = (i & 1 ? 2 : 1) * (j & 1 ? 2 : 1);
    (L.pc = pc), (L.pw = pw), (L.pn = pn);
    (nx = cx), (ny = cy), (A = C);
  }
  // the coarsest by a dense Cholesky factor
  const L = lev[lev.length - 1], m = L.nx * L.ny, M = new Float64Array(m * m);
  for (let v = 0; v < m; v++) for (let k = 0; k < 9; k++) {
    const i2 = (v % L.nx) + (k % 3) - 1, j2 = Math.floor(v / L.nx) + Math.floor(k / 3) - 1;
    if (i2 >= 0 && j2 >= 0 && i2 < L.nx && j2 < L.ny) M[v * m + j2 * L.nx + i2] += L.A[v * 9 + k];
  }
  let tr = 0;
  for (let i = 0; i < m; i++) tr += M[i * m + i];
  for (let i = 0; i < m; i++) M[i * m + i] += (1e-9 * tr) / m + 1e-300;
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
    let s = M[i * m + j];
    for (let k = 0; k < j; k++) s -= M[i * m + k] * M[j * m + k];
    M[i * m + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / M[j * m + j];
  }
  const yv = new Float64Array(m);
  const dense = (b: Float64Array, x: Float64Array) => {
    for (let a = 0; a < 2; a++) {
      for (let i = 0; i < m; i++) { let s = b[2 * i + a]; for (let k = 0; k < i; k++) s -= M[i * m + k] * yv[k]; yv[i] = s / M[i * m + i]; }
      for (let i = m - 1; i >= 0; i--) { let s = yv[i]; for (let k = i + 1; k < m; k++) s -= M[k * m + i] * x[2 * k + a]; x[2 * i + a] = s / M[i * m + i]; }
    }
  };
  const cycle = (d: number) => {
    const F = lev[d];
    if (d === lev.length - 1) return dense(F.b, F.x);
    F.x.fill(0);
    gs9(F.nx, F.ny, F.A, F.x, F.b, true, F.corner);
    mul9(F.nx, F.ny, F.A, F.x, F.r, F.corner, F.b);
    const C = lev[d + 1];
    C.b.fill(0);
    const { pc, pw, pn } = F, N = F.nx * F.ny, Cb = C.b, Cx = C.x, Fr = F.r, Fx = F.x;
    // a parent of weight 0 (the unused slots) adds nothing either way, so only the node's pn are read
    for (let v = 0; v < N; v++) {
      const rx = Fr[2 * v], ry = Fr[2 * v + 1];
      if (rx || ry) for (let p = 4 * v, e = p + pn[v]; p < e; p++) { const c = 2 * pc[p], w = pw[p]; (Cb[c] += w * rx), (Cb[c + 1] += w * ry); }
    }
    cycle(d + 1);
    for (let v = 0; v < N; v++) {
      let sx = 0, sy = 0;
      for (let p = 4 * v, e = p + pn[v]; p < e; p++) { const c = 2 * pc[p], w = pw[p]; (sx += w * Cx[c]), (sy += w * Cx[c + 1]); }
      (Fx[2 * v] += sx), (Fx[2 * v + 1] += sy);
    }
    gs9(F.nx, F.ny, F.A, F.x, F.b, false, F.corner);
  };
  return (r, z) => { lev[0].b.set(r); cycle(0); z.set(lev[0].x); };
}
/** conjugate gradients on A x = b from x for each axis `on` (x and y interleaved), preconditioned by `prec`; each
 * axis's iterations. An axis settled or off is left as it is while the other goes on. */
function pcg9(nx: number, ny: number, A: Float64Array, corner: Uint8Array, b: Float64Array, x: Float64Array, prec: (r: Float64Array, z: Float64Array) => void, tol: number, maxIt: number, on: [boolean, boolean]): [number, number] {
  const m = x.length, r = new Float64Array(m), z = new Float64Array(m), p = new Float64Array(m), q = new Float64Array(m);
  const it: [number, number] = [0, 0], go = [on[0] && maxIt > 0, on[1] && maxIt > 0];
  mul9(nx, ny, A, x, q, corner);
  let bn0 = 0, bn1 = 0;
  for (let i = 0; i < m; i += 2) (r[i] = b[i] - q[i]), (bn0 += b[i] * b[i]), (r[i + 1] = b[i + 1] - q[i + 1]), (bn1 += b[i + 1] * b[i + 1]);
  const bn = [Math.sqrt(bn0) || 1, Math.sqrt(bn1) || 1];
  prec(r, z);
  p.set(z);
  let rz0 = 0, rz1 = 0;
  for (let i = 0; i < m; i += 2) (rz0 += r[i] * z[i]), (rz1 += r[i + 1] * z[i + 1]);
  const rz = [rz0, rz1];
  while (go[0] || go[1]) {
    mul9(nx, ny, A, p, q, corner);
    let pq0 = 0, pq1 = 0;
    for (let i = 0; i < m; i += 2) (pq0 += p[i] * q[i]), (pq1 += p[i + 1] * q[i + 1]);
    const pq = [pq0, pq1];
    for (let a = 0; a < 2; a++) {
      if (!go[a]) continue;
      if (!(pq[a] > 0)) { go[a] = false; continue; }
      const al = rz[a] / pq[a];
      let rn = 0;
      for (let i = a; i < m; i += 2) (x[i] += al * p[i]), (r[i] -= al * q[i]), (rn += r[i] * r[i]);
      if (Math.sqrt(rn) / bn[a] < tol) (it[a]++, (go[a] = false));
    }
    if (!go[0] && !go[1]) break;
    prec(r, z);
    let rz20 = 0, rz21 = 0;
    for (let i = 0; i < m; i += 2) (rz20 += r[i] * z[i]), (rz21 += r[i + 1] * z[i + 1]);
    const rz2 = [rz20, rz21];
    for (let a = 0; a < 2; a++) {
      if (!go[a]) continue;
      const be = rz2[a] / rz[a];
      rz[a] = rz2[a];
      for (let i = a; i < m; i += 2) p[i] = z[i] + be * p[i];
      if (++it[a] >= maxIt) go[a] = false;
    }
  }
  return it;
}
