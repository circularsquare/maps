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
/** rivers are looked for this near a hop's track or a station, the sea this near a station, metres */
const RIVER_NEAR_M = 3000, SEA_NEAR_STATION_M = 15000;
/** a river's branch ending free shorter than this, or this many of the river's half widths where it leaves, is
 * dropped (thinning leaves such spurs); a river shorter than RIVER_MIN_M in all is a pond */
const SPUR_M = 400, SPUR_WIDTHS = 3, RIVER_MIN_M = 600;
/** a hop's crossings of two arcs this near each other along it are one, where rivers meet, metres */
const MEET_M = 600;
/** tiles' sides, cells */
const FT = 128, CT = 128;
const EARTH_R = 6371008.8, RAD = Math.PI / 180;

export interface RiverArc {
  /** its centre line, x, y pairs in graph metres, from node a to node b */
  pts: Float64Array;
  a: number;
  b: number;
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
  /** per node: where it is, graph metres, and whether a river ends there at the sea */
  nodes: { x: number; y: number; sea: boolean }[];
  /** whether a place is sea (1), land or river (0), or unknown (-1: no mask says, or too far from the stations) */
  seaAt(x: number, y: number): number;
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
   * middle at graph y `y` is water */
  return (y: number, x0: number, h: number, n: number, out: Uint8Array, off: number) => {
    const lat = latOf(y);
    for (let q = 0; q < masks.length; q++) {
      const m = masks[q], r = Math.floor((EARTH_R * (lat - m.lat) * RAD - m.y0) / m.cell);
      if (r < 0 || r >= m.h) continue;
      const { A, B } = cols[q];
      // the row's columns under the cells: cell i's middle is at column A + B (x0 + (i + 0.5) h)
      const cA = A + B * (x0 + 0.5 * h), cS = B * h;
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

/** chamfer distances (3 a side step, 4 a diagonal) from cells where `src` is 0 (or 1 with `from1`), w by h */
function chamfer(src: Uint8Array, w: number, h: number, from1: boolean, cap: number): Uint16Array {
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
  const n = g.key.length, rowOf = maskRows(masks, frame);
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
    const W = CT + 2 * HALO, buf = new Uint8Array(W * W);
    for (const key of want) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768;
      buf.fill(0);
      const x0 = (ti * CT - HALO) * hc;
      for (let j = 0; j < W; j++) rowOf((tj * CT - HALO + j + 0.5) * hc, x0, hc, W, buf, j * W);
      let any = 0;
      for (let c = 0; c < W * W; c++) any |= buf[c];
      const out = new Uint8Array(CT * CT);
      if (any) {
        // cores: water at least rs cells from land; sea: within rs of a core; near the sea: within rs + near
        const dl = chamfer(buf, W, W, false, 3 * (rs + 2));
        const core = new Uint8Array(W * W);
        for (let c = 0; c < W * W; c++) core[c] = dl[c] >= 3 * rs ? 1 : 0;
        const dc = chamfer(core, W, W, true, 3 * (rs + near + 2));
        for (let j = 0; j < CT; j++) for (let i = 0; i < CT; i++) {
          const c = (j + HALO) * W + i + HALO;
          out[j * CT + i] = dc[c] <= 3 * rs && buf[c] ? 3 : dc[c] <= 3 * (rs + near) ? 2 : buf[c] ? 1 : 0;
        }
      }
      ctiles.set(key, out);
    }
  }
  /** the coarse cell's class at a place: 3 sea, 2 near the sea, 1 other water, 0 land; -1 no tile */
  const coarseAt = (x: number, y: number) => {
    const i = Math.floor(x / hc), j = Math.floor(y / hc), t = ctiles.get(tkey(Math.floor(i / CT), Math.floor(j / CT)));
    return t ? t[(j - Math.floor(j / CT) * CT) * CT + (i - Math.floor(i / CT) * CT)] : -1;
  };
  const t1 = performance.now();

  // ---- rivers, on the fine grid near the track: water three cells across (an opening by a 3 by 3 square), less the
  // sea and its fringe. Per tile, 1 river, 2 wide water by the sea.
  const h = FINE_M, ftiles = new Map<number, Uint8Array>();
  {
    const side = FT * h, want = new Set<number>(), R = RIVER_NEAR_M;
    const mark = (x: number, y: number) => {
      for (let i = Math.floor((x - R) / side); i <= Math.floor((x + R) / side); i++) for (let j = Math.floor((y - R) / side); j <= Math.floor((y + R) / side); j++) want.add(tkey(i, j));
    };
    for (let i = 0; i < n; i++) mark(g.x[i], g.y[i]);
    for (let e = 0; e < g.ea.length; e++) for (let q = hopAt[e]; q + 1 < hopAt[e + 1]; q++) {
      const ax = hop[q * 2], ay = hop[q * 2 + 1], bx = hop[q * 2 + 2], by = hop[q * 2 + 3], parts = Math.ceil(hyp(bx - ax, by - ay) / 1000);
      for (let p = 1; p <= parts; p++) mark(ax + ((bx - ax) * p) / parts, ay + ((by - ay) * p) / parts);
    }
    const HF = 2, W = FT + 2 * HF, buf = new Uint8Array(W * W), ero = new Uint8Array(W * W);
    for (const key of want) {
      const ti = Math.floor(key / 65536) - 32768, tj = (key % 65536) - 32768;
      buf.fill(0);
      const x0 = (ti * FT - HF) * h;
      for (let j = 0; j < W; j++) rowOf((tj * FT - HF + j + 0.5) * h, x0, h, W, buf, j * W);
      ero.fill(0);
      let any = 0;
      for (let j = 1; j < W - 1; j++) for (let i = 1; i < W - 1; i++) {
        const c = j * W + i;
        if (buf[c] && buf[c - 1] && buf[c + 1] && buf[c - W] && buf[c + W] && buf[c - W - 1] && buf[c - W + 1] && buf[c + W - 1] && buf[c + W + 1]) (ero[c] = 1), (any = 1);
      }
      if (!any) continue;
      const out = new Uint8Array(FT * FT);
      let rivers = 0;
      for (let j = 0; j < FT; j++) for (let i = 0; i < FT; i++) {
        const c = (j + HF) * W + i + HF;
        if (!(ero[c] || ero[c - 1] || ero[c + 1] || ero[c - W] || ero[c + W] || ero[c - W - 1] || ero[c - W + 1] || ero[c + W - 1] || ero[c + W + 1])) continue;
        const sea = coarseAt((ti * FT + i + 0.5) * h, (tj * FT + j + 0.5) * h) >= 2;
        out[j * FT + i] = sea ? 2 : 1;
        if (!sea) rivers++;
      }
      if (rivers) ftiles.set(key, out);
    }
  }
  const fineAt = (i: number, j: number) => {
    const ti = Math.floor(i / FT), tj = Math.floor(j / FT), t = ftiles.get(tkey(ti, tj));
    return t ? t[(j - tj * FT) * FT + (i - ti * FT)] : 0;
  };
  const t2 = performance.now();

  // ---- each river (8-connected river cells) thinned and traced
  const nodes: RealWater["nodes"] = [], arcs: RiverArc[] = [];
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
    ms: { coarse: t1 - t0, fine: t2 - t1, skeleton: t3 - t2, crossings: t4 - t3 },
  };

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
    // nodes: cells with other than two neighbours, those touching one node
    const nodeOf = new Map<number, number>();
    const local: { cells: number[]; sea: boolean }[] = [];
    const isNode = (c: number) => nbrs(c).length !== 2;
    for (let c = 0; c < w * hh; c++) {
      if (!img[c] || nodeOf.has(c) || !isNode(c)) continue;
      const id = local.length, cells = [c];
      nodeOf.set(c, id);
      for (let q = 0; q < cells.length; q++) for (const d of [...nbrs(cells[q])]) if (!nodeOf.has(d) && isNode(d)) nodeOf.set(d, id), cells.push(d);
      local.push({ cells, sea: cells.some(bySea) });
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
      local.push({ cells: [c], sea: bySea(c) });
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
        const freeA = degree[l.a] === 1 && !local[l.a].sea, freeB = degree[l.b] === 1 && !local[l.b].sea;
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
        nodes.push({ x: (sx / cs.length) * h, y: (sy / cs.length) * h, sea: local[id].sea });
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
      arcs.push({ pts: Float64Array.from(simplifyLine(p, h)), a, b, half: hw[hw.length >> 1], cross: [] });
    });
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
