// The water on the ground the network diagram draws (T-2169; docs/ui-diagram.md "Water"): from the zones' water masks
// (each pack's 25 m raster, the one the track model prices water with), near the network only. Two grids in the
// graph's metres: a coarse one (COARSE_M) far round the stations says where the sea is (lakes too: water wider than
// SEA_M everywhere), and a fine one (FINE_M) near the track finds the rivers: water wider than RIVER_M that is not sea,
// thinned to centre lines and traced as arcs between the places where rivers meet or end. Each arc's crossings with the
// hops' tracks say which hops cross it, where. Pure: no DOM, no fetch.

import type { DiagramGraph } from "./graph";
import { simplifyLine } from "./graph";
import { hyp } from "./fmath";

/** a zone's water mask as its pack has it: row runs (pairs start, end) of water pixels, row 0 south, in the pack's
 * own equirectangular frame at (lon, lat) */
export interface WaterMask { cell: number; x0: number; y0: number; w: number; h: number; rows: Uint32Array; runs: Uint16Array; lon: number; lat: number }
/** the graph's frame in mercator, as diagram.worker.ts' ZoneFrame: graph metres are ((mx - f[0]) f[2], -(my - f[1]) f[2]) */
export type MercFrame = [number, number, number];

/** the fine grid's cell, metres; water three cells across (RIVER_M) or more is a river's */
const FINE_M = 40;
/** the coarse grid's cell, metres; water SEA_M across everywhere within is sea, and river water within SEA_NEAR_M of
 * the sea is the sea's (so a coast's fringe is no river) */
const COARSE_M = 200, SEA_M = 2000, SEA_NEAR_M = 400;
/** a river's end at the sea meets the sea nearest it within this many coarse cells */
const BODY_NEAR = 5;
/** a river ending free this near another river is joined to it, or this near the sea ends at it, metres (T-2516) */
const GAP_M = 400, SEA_GAP_M = 1000;
/** rivers are looked for this near a hop's track or a station, and followed on up to RIVER_FAR_M from a station; the
 * sea this near a station, metres */
const RIVER_NEAR_M = 3000, RIVER_FAR_M = 12000, SEA_NEAR_STATION_M = 15000;
/** a river's branch ending free shorter than this, or this many of the river's half widths where it leaves, is
 * dropped (thinning leaves such spurs); a river shorter than RIVER_MIN_M in all is a pond */
const SPUR_M = 400, SPUR_WIDTHS = 3, RIVER_MIN_M = 600;
/** a hop's crossings of two arcs this near each other along it are one, where rivers meet, metres */
const MEET_M = 600;
/** tiles' sides, cells */
const FT = 128, CT = 128;
/** a coarse cell some mask covers (else what is there is not known); one in water that parts land masses (the sea, or
 * water at least STRAIT coarse cells from land somewhere across, so about 600 m wide: rivers narrow upstream, so only
 * straits and channels between seas part land, T-2540) */
const COVERED = 8, PARTS = 16, STRAIT = 2;
/** the sea far from the stations (T-2540): a grid of at least FAR_M cells, at most FAR_CELLS, over the coarse tiles and
 * FAR_PAD past them, each place no mask covers taking the nearest known cell's */
const FAR_M = 1000, FAR_CELLS = 1_500_000, FAR_PAD = 100_000;
const EARTH_R = 6371008.8, RAD = Math.PI / 180;

export interface RiverArc {
  /** its centre line, x, y pairs in graph metres, from node a to node b */
  pts: Float64Array;
  a: number;
  b: number;
  /** the river it is part of: arcs of one connected water, between the seas */
  river: number;
  /** its half width, metres (the median along it) */
  half: number;
  /** its crossings with the hops, by distance along it */
  cross: Crossing[];
}
export interface Crossing {
  /** metres along the arc */
  s: number;
  /** the graph edge, and the share of its track's length from its a end */
  e: number;
  t: number;
  /** 1: the river goes from the hop's right (a to b) to its left; -1 the other way */
  sign: number;
}
export interface RealWater {
  arcs: RiverArc[];
  /** per node: where it is, graph metres, whether a river ends there at the sea, and which sea (`seaBody`; -1 none
   * found); `open`, where a river runs on out of what is looked at (far from the stations, or out of the masks) */
  nodes: { x: number; y: number; sea: boolean; body: number; open: boolean }[];
  /** whether a place is sea (1), land or river (0), or unknown (-1: no mask says, or too far from the stations) */
  seaAt(x: number, y: number): number;
  /** whether a place is sea, anywhere (T-2540): where no mask says, as the nearest place one does */
  isSea(x: number, y: number): boolean;
  /** which land mass a place is on: one id per land connected on the ground round water that parts it (the sea, and
   * straits about 600 m wide or more; T-2540), -1 in such water */
  massAt(x: number, y: number): number;
  /** which sea a place is in: one id per sea connected on the ground (T-2516), -1 where it is not sea */
  seaBody(x: number, y: number): number;
  /** the shortest way by water (any water on the coarse grid) from a place to a sea `ok` takes, within `maxM`: its
   * places (x, y pairs, graph metres) and that sea; null if none */
  waterWay(x: number, y: number, ok: (body: number) => boolean, maxM: number): { pts: number[]; body: number } | null;
  /** the hops that cross some river an even number of times, so are counted as crossing none */
  evenDropped: number;
  ms: { coarse: number; fine: number; skeleton: number; crossings: number };
}

/** the masks' rows in graph metres: per mask, its row at a graph y and its columns at graph x (col = A + B x) */
function maskRows(masks: WaterMask[], f: MercFrame) {
  const [fx, fy, k] = f;
  const cols = masks.map((m) => {
    const c0 = Math.cos(m.lat * RAD), s = EARTH_R * RAD * c0;
    // lon = 360 (fx + x / k) - 180
    return { B: (s * 360) / (k * m.cell), A: (s * (360 * fx - 180 - m.lon) - m.x0) / m.cell };
  });
  const latOf = (y: number) => {
    const my = fy - y / k;
    return (2 * Math.atan(Math.exp(Math.PI * (1 - 2 * my))) - Math.PI / 2) / RAD;
  };
  /** fills `out` (cells of `h` metres from graph x `x0`, `n` of them) with 1 where some mask's pixel under the cell's
   * middle at graph y `y` is water, and `cov` with 1 where some mask covers it */
  return (y: number, x0: number, h: number, n: number, out: Uint8Array, off: number, cov?: Uint8Array) => {
    const lat = latOf(y);
    for (let q = 0; q < masks.length; q++) {
      const m = masks[q], r = Math.floor((EARTH_R * (lat - m.lat) * RAD - m.y0) / m.cell);
      if (r < 0 || r >= m.h) continue;
      const { A, B } = cols[q];
      // the row's columns under the cells: cell i's middle is at column A + B (x0 + (i + 0.5) h)
      const cA = A + B * (x0 + 0.5 * h), cS = B * h;
      if (cov) for (let i = Math.max(0, Math.ceil(-cA / cS)); i < Math.min(n, Math.ceil((m.w - cA) / cS)); i++) cov[off + i] = 1;
      const lo = m.rows[r], hi = m.rows[r + 1];
      if (hi <= lo) continue;
      // the first run ending after the first cell's column
      let a = lo / 2, b = hi / 2;
      while (a < b) {
        const mid = (a + b) >> 1;
        if (m.runs[mid * 2 + 1] <= cA) a = mid + 1;
        else b = mid;
      }
      const cEnd = cA + cS * n;
      for (let p = a; p < hi / 2; p++) {
        const s = m.runs[p * 2], e = m.runs[p * 2 + 1];
        if (s >= cEnd) break;
        const i0 = Math.max(0, Math.ceil((s - cA) / cS)), i1 = Math.min(n, Math.ceil((e - cA) / cS));
        for (let i = i0; i < i1; i++) out[off + i] = 1;
      }
    }
  };
}

/** whether some mask covers a place in graph metres */
function maskCover(masks: WaterMask[], f: MercFrame) {
  const [fx, fy, k] = f;
  return (x: number, y: number) => {
    const lat = (2 * Math.atan(Math.exp(Math.PI * (1 - 2 * (fy - y / k)))) - Math.PI / 2) / RAD, lon = 360 * (fx + x / k) - 180;
    for (const m of masks) {
      const r = (EARTH_R * (lat - m.lat) * RAD - m.y0) / m.cell, c = (EARTH_R * RAD * Math.cos(m.lat * RAD) * (lon - m.lon) - m.x0) / m.cell;
      if (r >= 0 && r < m.h && c >= 0 && c < m.w) return true;
    }
    return false;
  };
}

/** chamfer distances (3 a side step, 4 a diagonal) from cells where `src` is 0 (or 1 with `from1`), w by h */
export function chamfer(src: Uint8Array, w: number, h: number, from1: boolean, cap: number): Uint16Array {
  const d = new Uint16Array(w * h);
  for (let c = 0; c < w * h; c++) d[c] = (src[c] !== 0) === from1 ? 0 : cap;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const c = j * w + i;
    let v = d[c];
    if (!v) continue;
    if (i > 0) v = Math.min(v, d[c - 1] + 3);
    if (j > 0) {
      v = Math.min(v, d[c - w] + 3);
      if (i > 0) v = Math.min(v, d[c - w - 1] + 4);
      if (i < w - 1) v = Math.min(v, d[c - w + 1] + 4);
    }
    d[c] = Math.min(v, cap);
  }
  for (let j = h - 1; j >= 0; j--) for (let i = w - 1; i >= 0; i--) {
    const c = j * w + i;
    let v = d[c];
    if (!v) continue;
    if (i < w - 1) v = Math.min(v, d[c + 1] + 3);
    if (j < h - 1) {
      v = Math.min(v, d[c + w] + 3);
      if (i < w - 1) v = Math.min(v, d[c + w + 1] + 4);
      if (i > 0) v = Math.min(v, d[c + w - 1] + 4);
    }
    d[c] = v;
  }
  return d;
}

/** The water near the network of `g` (laid out or not: only its places on the ground are read), from `masks`. */
export function realWater(g: DiagramGraph, masks: WaterMask[], frame: MercFrame): RealWater {
  const t0 = performance.now();
  const n = g.key.length, rowOf = maskRows(masks, frame), inMasks = maskCover(masks, frame);
  const hop = g.hop ?? new Float32Array(0), hopAt = g.hopAt ?? new Int32Array(g.ea.length + 1);

  // ---- the sea, on the coarse grid: tiles round the stations, each worked with a halo wide enough for the opening
  const hc = COARSE_M, rs = Math.round(SEA_M / 2 / hc), near = Math.round(SEA_NEAR_M / hc), HALO = 2 * rs + near + 2;
  const ctiles = new Map<number, Uint8Array>();
  const tkey = (i: number, j: number) => (i + 32768) * 65536 + (j + 32768);
  {
    const R = SEA_NEAR_STATION_M, side = CT * hc;
    const want = new Set<number>();
    const mark = (x: number, y: number, r: number, side: number, into: Set<number>) => {
      for (let i = Math.floor((x - r) / side); i <= Math.floor((x + r) / side); i++) for (let j = Math.floor((y - r) / side); j <= Math.floor((y + r) / side); j++) into.add(tkey(i, j));
    };
    for (let i = 0; i < n; i++) mark(g.x[i], g.y[i], R, side, want);
    const W = CT + 2 * HALO, buf = new Uint8Array(W * W), cov = new Uint8Array(W * W);
    for (const key of want) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768;
      buf.fill(0), cov.fill(0);
      const x0 = (ti * CT - HALO) * hc;
      for (let j = 0; j < W; j++) rowOf((tj * CT - HALO + j + 0.5) * hc, x0, hc, W, buf, j * W, cov);
      let any = 0;
      for (let c = 0; c < W * W; c++) any |= buf[c];
      const out = new Uint8Array(CT * CT);
      if (any) {
        // cores: water at least rs cells from land; sea: within rs of a core; near the sea: within rs + near
        const dl = chamfer(buf, W, W, false, 3 * (rs + 2));
        const core = new Uint8Array(W * W);
        for (let c = 0; c < W * W; c++) core[c] = dl[c] >= 3 * rs ? 1 : 0;
        const dc = chamfer(core, W, W, true, 3 * (rs + near + 2));
        // water parting land (T-2540): the sea, and water STRAIT cells across or more
        for (let c = 0; c < W * W; c++) core[c] = dl[c] >= 3 * STRAIT ? 1 : 0;
        const dw = chamfer(core, W, W, true, 3 * (STRAIT + 2));
        for (let j = 0; j < CT; j++) for (let i = 0; i < CT; i++) {
          const c = (j + HALO) * W + i + HALO;
          const cls = dc[c] <= 3 * rs && buf[c] ? 3 : dc[c] <= 3 * (rs + near) ? 2 : buf[c] ? 1 : 0;
          out[j * CT + i] = cls | (buf[c] ? 4 : 0) | (cls === 3 || (buf[c] && dw[c] <= 3 * STRAIT) ? PARTS : 0);
        }
      }
      for (let j = 0; j < CT; j++) for (let i = 0; i < CT; i++) if (cov[(j + HALO) * W + i + HALO]) out[j * CT + i] |= COVERED;
      ctiles.set(key, out);
    }
  }
  /** the coarse cell's class at a place: 3 sea, 2 near the sea, 1 other water, 0 land; -1 no tile; with 4 added where
   * it is water (`raw`) */
  const coarseAt = (x: number, y: number, raw = false) => {
    const i = Math.floor(x / hc), j = Math.floor(y / hc), t = ctiles.get(tkey(Math.floor(i / CT), Math.floor(j / CT)));
    return t ? t[(j - Math.floor(j / CT) * CT) * CT + (i - Math.floor(i / CT) * CT)] & (raw ? 7 : 3) : -1;
  };
  // each sea connected on the ground (4 neighbours, across tiles) one body (T-2516)
  const labs = new Map<number, Int32Array>(), par: number[] = [];
  const root = (v: number): number => {
    while (par[v] !== v) (par[v] = par[par[v]]), (v = par[v]);
    return v;
  };
  const join = (a: number, b: number) => void (par[root(a)] = root(b));
  for (const [key, t] of ctiles) {
    const lab = new Int32Array(CT * CT).fill(-1);
    for (let c = 0; c < CT * CT; c++) {
      if ((t[c] & 3) !== 3) continue;
      const l = c % CT > 0 ? lab[c - 1] : -1, u = c >= CT ? lab[c - CT] : -1;
      if (l < 0 && u < 0) lab[c] = par.length, par.push(par.length);
      else (lab[c] = l >= 0 ? l : u), l >= 0 && u >= 0 && join(l, u);
    }
    labs.set(key, lab);
  }
  for (const [key, lab] of labs) {
    const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768, e = labs.get(tkey(ti + 1, tj)), nb = labs.get(tkey(ti, tj + 1));
    for (let q = 0; q < CT; q++) {
      if (e && lab[q * CT + CT - 1] >= 0 && e[q * CT] >= 0) join(lab[q * CT + CT - 1], e[q * CT]);
      if (nb && lab[(CT - 1) * CT + q] >= 0 && nb[q] >= 0) join(lab[(CT - 1) * CT + q], nb[q]);
    }
  }
  const bodyId = new Map<number, number>();
  const seaBody = (x: number, y: number) => {
    const i = Math.floor(x / hc), j = Math.floor(y / hc), lab = labs.get(tkey(Math.floor(i / CT), Math.floor(j / CT)));
    const v = lab ? lab[(j - Math.floor(j / CT) * CT) * CT + (i - Math.floor(i / CT) * CT)] : -1;
    if (v < 0) return -1;
    const r = root(v);
    let id = bodyId.get(r);
    if (id === undefined) bodyId.set(r, (id = bodyId.size));
    return id;
  };
  /** the sea nearest a place within BODY_NEAR coarse cells, -1 none */
  const bodyNear = (x: number, y: number, r = BODY_NEAR) => {
    let best = -1, bd = Infinity;
    for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
      const dd = di * di + dj * dj, b = dd < bd ? seaBody(x + di * hc, y + dj * hc) : -1;
      if (b >= 0) (best = b), (bd = dd);
    }
    return best;
  };
  const { isSea, massAt } = farGround();
  const t1 = performance.now();

  /** the sea anywhere, and the land masses (T-2540): the coarse tiles where a mask covers them; elsewhere a far grid,
   * the masks' wide water where they cover it, else the nearest known cell's. Land masses are land connected round
   * the water that parts them (PARTS), across the tiles and the far grid. */
  function farGround() {
    const cellAt = (x: number, y: number) => {
      const i = Math.floor(x / hc), j = Math.floor(y / hc), t = ctiles.get(tkey(Math.floor(i / CT), Math.floor(j / CT)));
      return t ? t[(j - Math.floor(j / CT) * CT) * CT + (i - Math.floor(i / CT) * CT)] : 0;
    };
    let tx0 = Infinity, ty0 = Infinity, tx1 = -Infinity, ty1 = -Infinity;
    for (const key of ctiles.keys()) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768;
      (tx0 = Math.min(tx0, ti)), (ty0 = Math.min(ty0, tj)), (tx1 = Math.max(tx1, ti + 1)), (ty1 = Math.max(ty1, tj + 1));
    }
    if (!(tx0 < tx1)) return { isSea: () => false, massAt: () => 0 };
    const fx0 = tx0 * CT * hc - FAR_PAD, fy0 = ty0 * CT * hc - FAR_PAD, fx1 = tx1 * CT * hc + FAR_PAD, fy1 = ty1 * CT * hc + FAR_PAD;
    const fh = Math.max(FAR_M, Math.sqrt(((fx1 - fx0) * (fy1 - fy0)) / FAR_CELLS));
    const FW = Math.ceil((fx1 - fx0) / fh), FH = Math.ceil((fy1 - fy0) / fh), NF = FW * FH;
    // the masks' water, opened by a cell (so rivers go), where they cover it
    const raw = new Uint8Array(NF), cov = new Uint8Array(NF);
    for (let j = 0; j < FH; j++) rowOf(fy0 + (j + 0.5) * fh, fx0, fh, FW, raw, j * FW, cov);
    const core = new Uint8Array(NF);
    for (let j = 1; j < FH - 1; j++) for (let i = 1; i < FW - 1; i++) {
      const c = j * FW + i;
      core[c] = raw[c] & raw[c - 1] & raw[c + 1] & raw[c - FW] & raw[c + FW] & raw[c - FW - 1] & raw[c - FW + 1] & raw[c + FW - 1] & raw[c + FW + 1];
    }
    // 0 land, 1 sea, 2 not known yet; filled from the known cells outward (breadth first, so the nearest by steps);
    // `tiled`, cells whose middle a tile holds
    const far = new Uint8Array(NF).fill(2), tiled = new Uint8Array(NF), q: number[] = [];
    for (let j = 0; j < FH; j++) for (let i = 0; i < FW; i++) {
      const c = j * FW + i, v = cellAt(fx0 + (i + 0.5) * fh, fy0 + (j + 0.5) * fh);
      if (v & COVERED) (far[c] = (v & 3) === 3 ? 1 : 0), (tiled[c] = 1), q.push(c);
      else if (cov[c] && raw[c]) {
        // a mask's wide water (its land far out at sea may only be what it leaves out, so not known)
        let wide = 0;
        for (let dj = -1; dj <= 1 && !wide; dj++) for (let di = -1; di <= 1; di++) if (i + di >= 0 && i + di < FW && j + dj >= 0 && j + dj < FH && core[c + dj * FW + di]) (wide = 1);
        if (wide) (far[c] = 1), q.push(c);
      }
    }
    for (let k = 0; k < q.length; k++) {
      const c = q[k], i = c % FW, v = far[c];
      for (const nc of [i > 0 ? c - 1 : -1, i < FW - 1 ? c + 1 : -1, c - FW, c + FW]) if (nc >= 0 && nc < NF && far[nc] === 2) (far[nc] = v), q.push(nc);
    }
    if (!q.length) far.fill(0);
    const farAt = (x: number, y: number) => {
      const i = Math.min(FW - 1, Math.max(0, Math.floor((x - fx0) / fh))), j = Math.min(FH - 1, Math.max(0, Math.floor((y - fy0) / fh)));
      return j * FW + i;
    };
    // land masses: the tiles' land labelled (4 neighbours), joined across tiles, and to the far grid's land where no
    // tile is
    const par: number[] = [];
    const root = (v: number): number => {
      while (par[v] !== v) (par[v] = par[par[v]]), (v = par[v]);
      return v;
    };
    const join = (a: number, b: number) => void (par[root(a)] = root(b));
    const parts = (t: Uint8Array, c: number, x: number, y: number) => (t[c] & COVERED ? (t[c] & PARTS) !== 0 : far[farAt(x, y)] === 1);
    const labs = new Map<number, Int32Array>();
    for (const [key, t] of ctiles) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768, lab = new Int32Array(CT * CT).fill(-1);
      for (let c = 0; c < CT * CT; c++) {
        if (parts(t, c, (ti * CT + (c % CT) + 0.5) * hc, (tj * CT + Math.floor(c / CT) + 0.5) * hc)) continue;
        const l = c % CT > 0 ? lab[c - 1] : -1, u = c >= CT ? lab[c - CT] : -1;
        if (l < 0 && u < 0) (lab[c] = par.length), par.push(par.length);
        else (lab[c] = l >= 0 ? l : u), l >= 0 && u >= 0 && join(l, u);
      }
      labs.set(key, lab);
    }
    const farBase = par.length;
    for (let c = 0; c < NF; c++) par.push(c + farBase);
    const farLand = (c: number) => !tiled[c] && far[c] === 0;
    for (let j = 0; j < FH; j++) for (let i = 0; i < FW; i++) {
      const c = j * FW + i;
      if (!farLand(c)) continue;
      if (i + 1 < FW && farLand(c + 1)) join(farBase + c, farBase + c + 1);
      if (j + 1 < FH && farLand(c + FW)) join(farBase + c, farBase + c + FW);
    }
    for (const [key, lab] of labs) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768;
      const nb = [labs.get(tkey(ti - 1, tj)), labs.get(tkey(ti + 1, tj)), labs.get(tkey(ti, tj - 1)), labs.get(tkey(ti, tj + 1))];
      for (let q = 0; q < CT; q++) {
        // [this tile's cell, the neighbour tile's cell, the place just past the edge]
        const sides: [number, number, number, number][] = [
          [q * CT, q * CT + CT - 1, (ti * CT - 0.5) * hc, (tj * CT + q + 0.5) * hc],
          [q * CT + CT - 1, q * CT, (ti * CT + CT + 0.5) * hc, (tj * CT + q + 0.5) * hc],
          [q, (CT - 1) * CT + q, (ti * CT + q + 0.5) * hc, (tj * CT - 0.5) * hc],
          [(CT - 1) * CT + q, q, (ti * CT + q + 0.5) * hc, (tj * CT + CT + 0.5) * hc],
        ];
        sides.forEach(([c, oc, ox, oy], s) => {
          if (lab[c] < 0) return;
          const o = nb[s];
          if (o) o[oc] >= 0 && join(lab[c], o[oc]);
          else {
            const f = farAt(ox, oy);
            if (farLand(f)) join(lab[c], farBase + f);
          }
        });
      }
    }
    const ids = new Map<number, number>();
    const idOf = (v: number) => {
      const r = root(v);
      let id = ids.get(r);
      if (id === undefined) ids.set(r, (id = ids.size));
      return id;
    };
    return {
      isSea: (x: number, y: number) => {
        const v = cellAt(x, y);
        return v & COVERED ? (v & 3) === 3 : far[farAt(x, y)] === 1;
      },
      massAt: (x: number, y: number) => {
        const i = Math.floor(x / hc), j = Math.floor(y / hc), key = tkey(Math.floor(i / CT), Math.floor(j / CT)), lab = labs.get(key);
        if (lab) {
          const v = lab[(j - Math.floor(j / CT) * CT) * CT + (i - Math.floor(i / CT) * CT)];
          return v < 0 ? -1 : idOf(v);
        }
        const f = farAt(x, y);
        return farLand(f) ? idOf(farBase + f) : -1;
      },
    };
  }

  // ---- rivers, on the fine grid near the track: water three cells across (an opening by a 3 by 3 square), less the
  // sea and its fringe. Per tile, 1 river, 2 wide water by the sea.
  // The tiles near the track are made first; a river is followed on into tiles out to RIVER_FAR_M from a station, made
  // as its flood reaches them (T-2516: so it is found whole, to the sea), and where it runs on past those or out of
  // the masks its end is open.
  const h = FINE_M, ftiles = new Map<number, Uint8Array | null>(), want = new Set<number>(), allowed = new Set<number>();
  const HF = 2, FW = FT + 2 * HF, buf = new Uint8Array(FW * FW), ero = new Uint8Array(FW * FW);
  const fineTile = (key: number): Uint8Array | null => {
    const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768, W = FW;
    buf.fill(0);
    const x0 = (ti * FT - HF) * h;
    for (let j = 0; j < W; j++) rowOf((tj * FT - HF + j + 0.5) * h, x0, h, W, buf, j * W);
    ero.fill(0);
    let any = 0;
    for (let j = 1; j < W - 1; j++) for (let i = 1; i < W - 1; i++) {
      const c = j * W + i;
      if (buf[c] && buf[c - 1] && buf[c + 1] && buf[c - W] && buf[c + W] && buf[c - W - 1] && buf[c - W + 1] && buf[c + W - 1] && buf[c + W + 1]) (ero[c] = 1), (any = 1);
    }
    if (!any) return null;
    const out = new Uint8Array(FT * FT);
    let rivers = 0;
    for (let j = 0; j < FT; j++) for (let i = 0; i < FT; i++) {
      const c = (j + HF) * W + i + HF;
      if (!(ero[c] || ero[c - 1] || ero[c + 1] || ero[c - W] || ero[c + W] || ero[c - W - 1] || ero[c - W + 1] || ero[c + W - 1] || ero[c + W + 1])) continue;
      const sea = coarseAt((ti * FT + i + 0.5) * h, (tj * FT + j + 0.5) * h) >= 2;
      out[j * FT + i] = sea ? 2 : 1;
      if (!sea) rivers++;
    }
    return rivers ? out : null;
  };
  {
    const side = FT * h;
    const mark = (x: number, y: number, R: number, into: Set<number>) => {
      for (let i = Math.floor((x - R) / side); i <= Math.floor((x + R) / side); i++) for (let j = Math.floor((y - R) / side); j <= Math.floor((y + R) / side); j++) into.add(tkey(i, j));
    };
    for (let i = 0; i < n; i++) mark(g.x[i], g.y[i], RIVER_NEAR_M, want), mark(g.x[i], g.y[i], RIVER_FAR_M, allowed);
    for (let e = 0; e < g.ea.length; e++) for (let q = hopAt[e]; q + 1 < hopAt[e + 1]; q++) {
      const ax = hop[q * 2], ay = hop[q * 2 + 1], bx = hop[q * 2 + 2], by = hop[q * 2 + 3], parts = Math.ceil(hyp(bx - ax, by - ay) / 1000);
      for (let p = 1; p <= parts; p++) mark(ax + ((bx - ax) * p) / parts, ay + ((by - ay) * p) / parts, RIVER_NEAR_M, want);
    }
    for (const key of want) allowed.add(key), ftiles.set(key, fineTile(key));
  }
  const fineAt = (i: number, j: number) => {
    const ti = Math.floor(i / FT), tj = Math.floor(j / FT), key = tkey(ti, tj);
    let t = ftiles.get(key);
    if (t === undefined) ftiles.set(key, (t = allowed.has(key) ? fineTile(key) : null));
    return t ? t[(j - tj * FT) * FT + (i - ti * FT)] : 0;
  };
  /** whether a fine cell is past where rivers are followed: out of the allowed tiles or of every mask */
  const beyond = (i: number, j: number) => !allowed.has(tkey(Math.floor(i / FT), Math.floor(j / FT))) || !inMasks((i + 0.5) * h, (j + 0.5) * h);
  const t2 = performance.now();

  // ---- each river (8-connected river cells) thinned and traced
  const nodes: RealWater["nodes"] = [], arcs: RiverArc[] = [];
  let rivers = 0;
  const seen = new Map<number, Uint8Array>();
  const seenAt = (i: number, j: number) => {
    const key = tkey(Math.floor(i / FT), Math.floor(j / FT));
    let s = seen.get(key);
    if (!s) seen.set(key, (s = new Uint8Array(FT * FT)));
    return [s, (j - Math.floor(j / FT) * FT) * FT + (i - Math.floor(i / FT) * FT)] as const;
  };
  // only rivers some hop's track passes over: from the river cells under the tracks
  const seeds: number[] = [];
  for (let e = 0; e < g.ea.length; e++) for (let q = hopAt[e]; q + 1 < hopAt[e + 1]; q++) {
    const ax = hop[q * 2], ay = hop[q * 2 + 1], bx = hop[q * 2 + 2], by = hop[q * 2 + 3], parts = Math.ceil((2 * hyp(bx - ax, by - ay)) / h);
    for (let p = 0; p <= parts; p++) {
      const i = Math.floor((ax + ((bx - ax) * p) / parts) / h), j = Math.floor((ay + ((by - ay) * p) / parts) / h);
      if (fineAt(i, j) === 1) seeds.push(i, j);
    }
  }
  {
    for (let z = 0; z < seeds.length; z += 2) {
      const si = seeds[z], sj = seeds[z + 1], [s0, c0] = seenAt(si, sj);
      if (s0[c0]) continue;
      // the river's cells, by a flood
      s0[c0] = 1;
      const ci: number[] = [si], cj: number[] = [sj];
      let i0 = si, i1 = si, j0 = sj, j1 = sj;
      for (let q = 0; q < ci.length; q++) {
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
          const ni = ci[q] + di, nj = cj[q] + dj;
          if (fineAt(ni, nj) !== 1) continue;
          const [s, cc] = seenAt(ni, nj);
          if (s[cc]) continue;
          s[cc] = 1;
          ci.push(ni), cj.push(nj);
          (i0 = Math.min(i0, ni)), (i1 = Math.max(i1, ni)), (j0 = Math.min(j0, nj)), (j1 = Math.max(j1, nj));
        }
      }
      if (ci.length * h * h < RIVER_M_AREA) continue;
      traceRiver(ci, cj, i0 - 2, j0 - 2, i1 - i0 + 5, j1 - j0 + 5);
    }
  }
  bridgeGaps();
  const t3 = performance.now();

  // ---- crossings: each arc against the hops' tracks near it
  const HC = 2000, segs = new Map<number, number[]>();
  for (let e = 0; e < g.ea.length; e++) for (let q = hopAt[e]; q + 1 < hopAt[e + 1]; q++) {
    const ax = hop[q * 2], ay = hop[q * 2 + 1], bx = hop[q * 2 + 2], by = hop[q * 2 + 3];
    for (let i = Math.floor(Math.min(ax, bx) / HC); i <= Math.floor(Math.max(ax, bx) / HC); i++) for (let j = Math.floor(Math.min(ay, by) / HC); j <= Math.floor(Math.max(ay, by) / HC); j++) {
      const l = segs.get(tkey(i, j));
      if (l) l.push(e, q);
      else segs.set(tkey(i, j), [e, q]);
    }
  }
  const hopLen = new Float64Array(g.ea.length), segAt = new Float64Array(hop.length / 2);
  for (let e = 0; e < g.ea.length; e++) for (let q = hopAt[e]; q + 1 < hopAt[e + 1]; q++) {
    segAt[q] = hopLen[e];
    hopLen[e] += hyp(hop[q * 2 + 2] - hop[q * 2], hop[q * 2 + 3] - hop[q * 2 + 1]);
  }
  let evenDropped = 0;
  for (const A of arcs) {
    const P = A.pts, found = new Map<number, Crossing[]>();
    let s = 0;
    for (let k = 0; k + 3 < P.length; k += 2) {
      const ax = P[k], ay = P[k + 1], bx = P[k + 2], by = P[k + 3], sl = hyp(bx - ax, by - ay), ux = bx - ax, uy = by - ay;
      const cand = new Map<number, number>();
      for (let i = Math.floor(Math.min(ax, bx) / HC); i <= Math.floor(Math.max(ax, bx) / HC); i++) for (let j = Math.floor(Math.min(ay, by) / HC); j <= Math.floor(Math.max(ay, by) / HC); j++) {
        const l = segs.get(tkey(i, j));
        if (l) for (let z = 0; z < l.length; z += 2) cand.set(l[z + 1], l[z]);
      }
      for (const [q, e] of cand) {
        const cx = hop[q * 2], cy = hop[q * 2 + 1], dx = hop[q * 2 + 2] - cx, dy = hop[q * 2 + 3] - cy;
        const den = ux * dy - uy * dx;
        if (Math.abs(den) < 1e-12) continue;
        const r = ((cx - ax) * dy - (cy - ay) * dx) / den, tt = ((cx - ax) * uy - (cy - ay) * ux) / den;
        if (r < 0 || r >= 1 || tt < 0 || tt >= 1) continue;
        const c: Crossing = { s: s + r * sl, e, t: hopLen[e] > 0 ? (segAt[q] + tt * hyp(dx, dy)) / hopLen[e] : 0.5, sign: dx * uy - dy * ux > 0 ? 1 : -1 };
        const l = found.get(e);
        if (l) l.push(c);
        else found.set(e, [c]);
      }
      s += sl;
    }
    for (const l of found.values()) {
      if (l.length % 2 === 0) {
        evenDropped++;
        continue;
      }
      l.sort((p, q) => p.s - q.s);
      A.cross.push(l[l.length >> 1]);
    }
  }
  // a hop crossing where rivers meet crosses two arcs within a short way: it crosses the wider one only
  {
    const byHop = new Map<number, [RiverArc, Crossing][]>();
    for (const A of arcs) for (const c of A.cross) {
      const l = byHop.get(c.e);
      if (l) l.push([A, c]);
      else byHop.set(c.e, [[A, c]]);
    }
    for (const [e, l] of byHop) {
      if (l.length < 2) continue;
      l.sort((p, q) => q[0].half - p[0].half);
      const kept: number[] = [];
      for (const [A, c] of l) {
        if (kept.some((t) => Math.abs(t - c.t) * hopLen[e] < MEET_M)) A.cross.splice(A.cross.indexOf(c), 1);
        else kept.push(c.t);
      }
    }
  }
  for (const A of arcs) A.cross.sort((p, q) => p.s - q.s);
  const t4 = performance.now();

  return {
    arcs, nodes, evenDropped,
    seaAt: (x, y) => {
      const c = coarseAt(x, y);
      return c < 0 ? -1 : c === 3 ? 1 : 0;
    },
    isSea,
    massAt,
    seaBody,
    waterWay(x, y, ok, maxM) {
      const key = (i: number, j: number) => (i + 1e6) * 4e6 + (j + 1e6);
      const wet = (i: number, j: number) => coarseAt((i + 0.5) * hc, (j + 0.5) * hc, true) >= 4;
      const dist = new Map<number, number>(), from = new Map<number, number>(), hp: [number, number, number][] = [];
      const push = (d: number, i: number, j: number) => {
        hp.push([d, i, j]);
        for (let c = hp.length - 1; c > 0; ) {
          const p = (c - 1) >> 1;
          if (hp[p][0] <= hp[c][0]) break;
          [hp[p], hp[c]] = [hp[c], hp[p]];
          c = p;
        }
      };
      const pop = () => {
        const top = hp[0], last = hp.pop()!;
        if (hp.length) {
          hp[0] = last;
          for (let c = 0; ; ) {
            const a = 2 * c + 1, b = a + 1;
            let m = c;
            if (a < hp.length && hp[a][0] < hp[m][0]) m = a;
            if (b < hp.length && hp[b][0] < hp[m][0]) m = b;
            if (m === c) break;
            [hp[m], hp[c]] = [hp[c], hp[m]];
            c = m;
          }
        }
        return top;
      };
      const i0 = Math.floor(x / hc), j0 = Math.floor(y / hc);
      for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) if (wet(i0 + di, j0 + dj)) dist.set(key(i0 + di, j0 + dj), 0), push(0, i0 + di, j0 + dj);
      while (hp.length) {
        const [dv, i, j] = pop();
        if (dv > (dist.get(key(i, j)) ?? Infinity) || dv > maxM) continue;
        const b = coarseAt((i + 0.5) * hc, (j + 0.5) * hc) === 3 ? seaBody((i + 0.5) * hc, (j + 0.5) * hc) : -1;
        if (b >= 0 && ok(b)) {
          const pts: number[] = [];
          for (let k: number | undefined = key(i, j); k !== undefined; k = from.get(k)) {
            const ci = Math.floor(k / 4e6) - 1e6, cj = k - (ci + 1e6) * 4e6 - 1e6;
            pts.push((ci + 0.5) * hc, (cj + 0.5) * hc);
          }
          const out: number[] = [];
          for (let q = pts.length - 2; q >= 0; q -= 2) out.push(pts[q], pts[q + 1]);
          return { pts: simplifyLine(out, hc), body: b };
        }
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
          if ((!di && !dj) || !wet(i + di, j + dj)) continue;
          const nd = dv + (di && dj ? Math.SQRT2 : 1) * hc, k = key(i + di, j + dj);
          if (nd < (dist.get(k) ?? Infinity)) dist.set(k, nd), from.set(k, key(i, j)), push(nd, i + di, j + dj);
        }
      }
      return null;
    },
    ms: { coarse: t1 - t0, fine: t2 - t1, skeleton: t3 - t2, crossings: t4 - t3 },
  };

  /** a river ending free where the mask has a short gap (a bridge drawn as land, the river narrowing under three cells
   * for a while) is joined on (T-2516): to another river within GAP_M, by a straight arc to its nearest point, the
   * two then one river; else, within SEA_GAP_M of the sea, it ends at the sea */
  function bridgeGaps() {
    const deg = new Int32Array(nodes.length);
    for (const A of arcs) deg[A.a]++, deg[A.b]++;
    const G = 1000, grid = new Map<number, number[]>();
    const index = (q: number) => {
      const P = arcs[q].pts;
      for (let k = 0; k < P.length; k += 2) {
        const key = tkey(Math.floor(P[k] / G), Math.floor(P[k + 1] / G)), l = grid.get(key);
        if (l) l.push(q, k);
        else grid.set(key, [q, k]);
      }
    };
    arcs.forEach((_, q) => index(q));
    const same = arcs.map((A) => A.river), up = Array.from({ length: rivers }, (_, r) => r);
    const top = (r: number): number => (up[r] === r ? r : (up[r] = top(up[r])));
    const riverOf = new Int32Array(nodes.length).fill(-1);
    arcs.forEach((A) => (riverOf[A.a] = riverOf[A.b] = A.river));
    for (let v = 0; v < nodes.length; v++) {
      const nd = nodes[v];
      if (deg[v] !== 1 || nd.sea || nd.open) continue;
      let bq = -1, bk = -1, bd = GAP_M;
      for (let i = Math.floor((nd.x - GAP_M) / G); i <= Math.floor((nd.x + GAP_M) / G); i++) for (let j = Math.floor((nd.y - GAP_M) / G); j <= Math.floor((nd.y + GAP_M) / G); j++) {
        const l = grid.get(tkey(i, j));
        if (l) for (let z = 0; z < l.length; z += 2) {
          const q = l[z], k = l[z + 1], P = arcs[q].pts;
          if (top(same[q]) === top(riverOf[v]) || k >= P.length) continue;
          const dd = hyp(P[k] - nd.x, P[k + 1] - nd.y);
          if (dd < bd) (bd = dd), (bq = q), (bk = k);
        }
      }
      if (bq >= 0) {
        // split the arc at that point (unless it is an end), and join the two by a straight arc
        const A = arcs[bq], P = A.pts;
        let w: number;
        if (bk === 0) w = A.a;
        else if (bk === P.length - 2) w = A.b;
        else {
          w = nodes.length;
          nodes.push({ x: P[bk], y: P[bk + 1], sea: false, body: -1, open: false });
          const B: RiverArc = { pts: P.slice(bk), a: w, b: A.b, river: A.river, half: A.half, cross: [] };
          (A.pts = P.slice(0, bk + 2)), (A.b = w);
          arcs.push(B), same.push(B.river), index(arcs.length - 1);
        }
        arcs.push({ pts: Float64Array.of(nd.x, nd.y, nodes[w].x, nodes[w].y), a: v, b: w, river: riverOf[v], half: A.half, cross: [] });
        same.push(riverOf[v]);
        up[top(same[bq])] = top(riverOf[v]);
        continue;
      }
      for (let s = COARSE_M / 2; s <= SEA_GAP_M; s += COARSE_M / 2) {
        let hit = false;
        for (let q = 0; q < 16 && !hit; q++) hit = coarseAt(nd.x + s * Math.cos((q * Math.PI) / 8), nd.y + s * Math.sin((q * Math.PI) / 8)) === 3;
        if (hit) {
          (nd.sea = true), (nd.body = bodyNear(nd.x, nd.y, BODY_NEAR + Math.ceil(SEA_GAP_M / COARSE_M)));
          break;
        }
      }
    }
    arcs.forEach((A, q) => (A.river = top(same[q])));
  }

  /** one river's cells (global cell indices) thinned in a raster of its own (origin ox, oy, w by hh cells) and traced
   * into `nodes` and `arcs` */
  function traceRiver(ci: number[], cj: number[], ox: number, oy: number, w: number, hh: number) {
    const img = new Uint8Array(w * hh);
    for (let q = 0; q < ci.length; q++) img[(cj[q] - oy) * w + ci[q] - ox] = 1;
    const half = chamfer(img, w, hh, false, 60000);
    thin(img, w, hh);
    // the skeleton as a graph: 8 neighbours, but a diagonal step only where no side step goes round it
    const SIDE = [1, w, -1, -w], DIAG = [w + 1, w - 1, -w - 1, -w + 1];
    const nbr: number[] = [];
    const nbrs = (c: number) => {
      nbr.length = 0;
      for (const o of SIDE) if (img[c + o]) nbr.push(c + o);
      for (let q = 0; q < 4; q++) {
        const o = DIAG[q], d = c + o;
        // diagonal q sits between side steps q and q + 1 (east-north, north-west, ...)
        if (img[d] && !img[c + SIDE[q]] && !img[c + SIDE[(q + 1) % 4]]) nbr.push(d);
      }
      return nbr;
    };
    const bySea = (c: number) => {
      const i = (c % w) + ox, j = Math.floor(c / w) + oy;
      for (let dj = -2; dj <= 2; dj++) for (let di = -2; di <= 2; di++) if (fineAt(i + di, j + dj) === 2) return true;
      return false;
    };
    const byEdge = (c: number) => {
      const i = (c % w) + ox, j = Math.floor(c / w) + oy;
      for (let dj = -3; dj <= 3; dj += 3) for (let di = -3; di <= 3; di += 3) if (beyond(i + di, j + dj)) return true;
      return false;
    };
    // nodes: cells with other than two neighbours, those touching one node
    const nodeOf = new Map<number, number>();
    const local: { cells: number[]; sea: boolean; open: boolean }[] = [];
    const isNode = (c: number) => nbrs(c).length !== 2;
    for (let c = 0; c < w * hh; c++) {
      if (!img[c] || nodeOf.has(c) || !isNode(c)) continue;
      const id = local.length, cells = [c];
      nodeOf.set(c, id);
      for (let q = 0; q < cells.length; q++) for (const d of [...nbrs(cells[q])]) if (!nodeOf.has(d) && isNode(d)) nodeOf.set(d, id), cells.push(d);
      local.push({ cells, sea: cells.some(bySea), open: cells.some(byEdge) });
    }
    type LArc = { cells: number[]; a: number; b: number; len: number };
    const larcs: LArc[] = [], used = new Uint8Array(w * hh);
    const lenOf = (cells: number[]) => {
      let len = 0;
      for (let q = 1; q < cells.length; q++) len += hyp((cells[q] % w) - (cells[q - 1] % w), Math.floor(cells[q] / w) - Math.floor(cells[q - 1] / w));
      return len * h;
    };
    /** from node cell `c` through `d` to the next node */
    const walk = (c: number, d: number) => {
      const cells = [c, d];
      let prev = c, cur = d;
      while (!nodeOf.has(cur)) {
        used[cur] = 1;
        const nx = nbrs(cur).find((x) => x !== prev);
        if (nx === undefined) return;
        cells.push(nx);
        (prev = cur), (cur = nx);
      }
      larcs.push({ cells, a: nodeOf.get(c)!, b: nodeOf.get(cur)!, len: lenOf(cells) });
    };
    local.forEach((nd) => {
      for (const c of nd.cells) for (const d of [...nbrs(c)]) if (!nodeOf.has(d) && !used[d]) walk(c, d);
    });
    // rings with no node: a node on each
    for (let c = 0; c < w * hh; c++) {
      if (!img[c] || used[c] || nodeOf.has(c)) continue;
      nodeOf.set(c, local.length);
      local.push({ cells: [c], sea: bySea(c), open: false });
      const d = nbrs(c)[0];
      if (d !== undefined) walk(c, d);
    }
    // spurs: arcs with a free end (not at the sea), short for the river's width where they leave, dropped until
    // none is; rings short too
    const alive = larcs.map((l) => l.a !== l.b || l.len >= SPUR_M);
    const degree = new Int32Array(local.length);
    const recount = () => {
      degree.fill(0);
      larcs.forEach((l, q) => alive[q] && (degree[l.a]++, degree[l.b]++));
    };
    recount();
    for (let round = 0; round < 6; round++) {
      let cut = false;
      larcs.forEach((l, q) => {
        if (!alive[q] || l.a === l.b) return;
        const freeA = degree[l.a] === 1 && !local[l.a].sea && !local[l.a].open, freeB = degree[l.b] === 1 && !local[l.b].sea && !local[l.b].open;
        if (freeA === freeB) return;
        const at = freeA ? l.cells[l.cells.length - 1] : l.cells[0];
        if (l.len < Math.max(SPUR_M, SPUR_WIDTHS * (half[at] / 3) * h)) (alive[q] = false), (cut = true);
      });
      if (!cut) break;
      recount();
    }
    let total = 0;
    larcs.forEach((l, q) => alive[q] && (total += l.len));
    if (total < RIVER_MIN_M) return;
    // chains through nodes left with two arcs joined
    const at: number[][] = local.map(() => []);
    larcs.forEach((l, q) => alive[q] && (at[l.a].push(q), at[l.b].push(q)));
    const through = (id: number) => at[id].length === 2 && at[id][0] !== at[id][1];
    const far = (q: number, id: number) => (larcs[q].a === id ? larcs[q].b : larcs[q].a);
    const done = new Uint8Array(larcs.length), kept = new Map<number, number>();
    const keepNode = (id: number) => {
      let k = kept.get(id);
      if (k === undefined) {
        const cs = local[id].cells;
        let sx = 0, sy = 0;
        for (const c of cs) (sx += (c % w) + ox + 0.5), (sy += Math.floor(c / w) + oy + 0.5);
        kept.set(id, (k = nodes.length));
        const nx = (sx / cs.length) * h, ny = (sy / cs.length) * h;
        nodes.push({ x: nx, y: ny, sea: local[id].sea, body: local[id].sea ? bodyNear(nx, ny) : -1, open: local[id].open && !local[id].sea });
      }
      return k;
    };
    larcs.forEach((_, q) => {
      if (!alive[q] || done[q]) return;
      // back to a node that is not a through node (or round a ring of them to q again)
      let arc = q, from = larcs[q].a;
      for (let guard = 0; through(from) && guard < larcs.length; guard++) {
        const other = at[from][0] === arc ? at[from][1] : at[from][0];
        if (other === q) break;
        from = far(other, from);
        arc = other;
      }
      // then forward, joining arcs through through nodes
      const start = from, cells: number[] = [];
      for (let guard = 0; guard <= larcs.length; guard++) {
        const L = larcs[arc];
        done[arc] = 1;
        const fwd = L.a === from, cs = fwd ? L.cells : [...L.cells].reverse();
        for (let z = cells.length ? 1 : 0; z < cs.length; z++) cells.push(cs[z]);
        from = fwd ? L.b : L.a;
        if (!through(from)) break;
        const nx = at[from][0] === arc ? at[from][1] : at[from][0];
        if (done[nx]) break;
        arc = nx;
      }
      const p: number[] = [], hw: number[] = [];
      for (const c of cells) p.push(((c % w) + ox + 0.5) * h, (Math.floor(c / w) + oy + 0.5) * h), hw.push((half[c] / 3) * h);
      hw.sort((u, v) => u - v);
      // the ends at the nodes' middles
      const a = keepNode(start), b = keepNode(from);
      (p[0] = nodes[a].x), (p[1] = nodes[a].y), (p[p.length - 2] = nodes[b].x), (p[p.length - 1] = nodes[b].y);
      arcs.push({ pts: Float64Array.from(simplifyLine(p, h)), a, b, river: rivers, half: hw[hw.length >> 1], cross: [] });
    });
    rivers++;
  }
}

/** water at least this many square metres in all, before thinning, can be a river */
const RIVER_M_AREA = RIVER_MIN_M * 3 * FINE_M;

/** Zhang and Suen's thinning of a 0/1 raster (its edge cells 0), in place */
function thin(img: Uint8Array, w: number, h: number) {
  const del: number[] = [];
  let cand: number[] = [];
  for (let c = 0; c < w * h; c++) if (img[c]) cand.push(c);
  for (let pass = 0; ; pass++) {
    let changed = false;
    for (const step of [0, 1]) {
      del.length = 0;
      const next: number[] = [];
      for (const c of cand) {
        if (!img[c]) continue;
        const p2 = img[c - w], p3 = img[c - w + 1], p4 = img[c + 1], p5 = img[c + w + 1], p6 = img[c + w], p7 = img[c + w - 1], p8 = img[c - 1], p9 = img[c - w - 1];
        const b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
        if (b < 2 || b > 6) {
          next.push(c);
          continue;
        }
        const a = (+(!p2 && p3)) + (+(!p3 && p4)) + (+(!p4 && p5)) + (+(!p5 && p6)) + (+(!p6 && p7)) + (+(!p7 && p8)) + (+(!p8 && p9)) + (+(!p9 && p2));
        if (a !== 1) {
          next.push(c);
          continue;
        }
        if (step === 0 ? p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0 : p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0) del.push(c);
        else next.push(c);
      }
      for (const c of del) img[c] = 0;
      if (del.length) changed = true;
      cand = next;
    }
    if (!changed) break;
  }
}
