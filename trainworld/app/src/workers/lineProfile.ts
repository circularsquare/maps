// T-2734: what the line inspector's cross-section draws, from the track model: the line's centre line
// with its height over the ground, its stops along it, and where other lines' track crosses it.

import type { TrackApi } from "../wasm/track/sim";

/** grid cell for finding crossings, m */
const CELL = 250;
/** an intersection this close to an end of the other line's edge is a junction there, m */
const JUNCTION_M = 3;
/** track meeting at under about 12 degrees runs alongside rather than across: the real saves' parallel
 * lines weave over each other's centre lines along a shared corridor (sin 12°) */
const SHALLOW_SIN = 0.2;

/** work done before the worker turns to its other messages, ms (Anita, 2026-11-07: the cross section
 * is flavour and never holds up anything else) */
const SLICE_MS = 6;

/** how often along the line the water masks are read, m */
const WATER_STEP_M = 20;
const EARTH_R = 6371008.8, RAD = Math.PI / 180;

/** a zone's water mask as its pack has it (the clock worker's `WaterRaster`): row runs of water cells,
 * row 0 south, in the pack's own frame at (lon, lat) */
export interface WaterMaskRows { cell: number; x0: number; y0: number; w: number; h: number; rows: Uint32Array; runs: Uint16Array; lon: number; lat: number }

/** 1 where some mask says water at a place, 0 where one covers it as land, -1 where none covers it */
function waterAt(masks: WaterMaskRows[], lng: number, lat: number): number {
  let covered = false;
  for (const m of masks) {
    const r = Math.floor((EARTH_R * (lat - m.lat) * RAD - m.y0) / m.cell);
    const c = (EARTH_R * RAD * Math.cos(m.lat * RAD) * (lng - m.lon) - m.x0) / m.cell;
    if (r < 0 || r >= m.h || c < 0 || c >= m.w) continue;
    covered = true;
    for (let p = m.rows[r] / 2; p < m.rows[r + 1] / 2; p++) {
      if (m.runs[p * 2] > c) break;
      if (c < m.runs[p * 2 + 1]) return 1;
    }
  }
  return covered ? 0 : -1;
}

/**
 * One Float64Array: counts nP, nS, nC, nW; then nP points (s m along the line, lng, lat, track height
 * over the ground m); nS stops (s, node); nC crossings (s, the other track's height m, n, n line ids);
 * nW stretches (s from, s to, 1 water or -1 no mask known). Empty for a line with no path, or when
 * `current` turns false (a new track model) part way. Worked in slices with the worker's other messages
 * between them, so it never delays an answer. `masksUnder` gives the water masks of the zones under
 * world x, y pairs.
 */
export async function lineProfile(api: TrackApi, line: number, lngLat: (x: number, y: number) => [number, number], current: () => boolean,
  masksUnder: (xy: number[]) => Promise<WaterMaskRows[]>): Promise<Float64Array> {
  let t0 = 0;
  /** false once the track model has been replaced */
  const pause = async () => {
    if (t0 && performance.now() - t0 < SLICE_MS) return true;
    await new Promise((r) => setTimeout(r, 0));
    t0 = performance.now();
    return current();
  };
  const none = new Float64Array(0);
  if (!(await pause())) return none;
  const path = api.line_path(line);
  if (!path.length) return new Float64Array(0);
  const own = new Set<number>();
  const xs: number[] = [], ys: number[] = [], hs: number[] = [];
  for (const p of path) {
    const e = p >> 1;
    own.add(e);
    const r = api.edge_render(e), n = r.length / 3;
    for (let j = 0; j < n; j++) {
      const k = (p & 1 ? n - 1 - j : j) * 3;
      if (xs.length && j === 0 && Math.hypot(r[k] - xs[xs.length - 1], r[k + 1] - ys[ys.length - 1]) < 0.5) continue;
      xs.push(r[k]), ys.push(r[k + 1]), hs.push(r[k + 2]);
    }
  }
  const nP = xs.length;
  const s = new Float64Array(nP);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < nP; i++) {
    if (i) s[i] = s[i - 1] + Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    x0 = Math.min(x0, xs[i]), y0 = Math.min(y0, ys[i]), x1 = Math.max(x1, xs[i]), y1 = Math.max(y1, ys[i]);
  }

  // stops: each placed at the nearest point of the line past the stop before
  const nodes = api.nodes_info();
  const nodeXY = new Map<number, [number, number]>();
  for (let i = 0; i < nodes.length; i += 10) nodeXY.set(nodes[i], [nodes[i + 1], nodes[i + 2]]);
  const stops: number[] = [];
  let from = 0;
  for (const node of api.line_stops(line)) {
    if (!(await pause())) return none;
    const q = nodeXY.get(node);
    if (!q) continue;
    let best = Infinity, bestS = s[from], bestI = from;
    for (let i = from; i + 1 < nP; i++) {
      const dx = xs[i + 1] - xs[i], dy = ys[i + 1] - ys[i], L2 = dx * dx + dy * dy;
      const t = L2 > 0 ? Math.max(0, Math.min(1, ((q[0] - xs[i]) * dx + (q[1] - ys[i]) * dy) / L2)) : 0;
      const d = Math.hypot(xs[i] + t * dx - q[0], ys[i] + t * dy - q[1]);
      if (d < best - 1e-6) best = d, bestS = s[i] + t * (s[i + 1] - s[i]), bestI = i;
    }
    stops.push(bestS, node);
    from = bestI;
  }

  // crossings: other rail lines' edges off this line's path
  const linesOf = new Map<number, number[]>();
  const info = api.lines_info();
  for (let i = 0; i < info.length; i += 22) {
    const id = info[i];
    if (!(await pause())) return none;
    if (id === line || api.is_bus(id)) continue;
    for (const p of api.line_path(id)) {
      const e = p >> 1;
      if (own.has(e)) continue;
      const l = linesOf.get(e);
      if (!l) linesOf.set(e, [id]);
      else if (!l.includes(id)) l.push(id);
    }
  }
  const crosses: number[] = [];
  let nC = 0;
  if (linesOf.size) {
    const grid = new Map<number, number[]>();
    const key = (cx: number, cy: number) => cx * 1048576 + cy;
    for (let i = 0; i + 1 < nP; i++) {
      const ax = Math.floor(Math.min(xs[i], xs[i + 1]) / CELL), bx = Math.floor(Math.max(xs[i], xs[i + 1]) / CELL);
      const ay = Math.floor(Math.min(ys[i], ys[i + 1]) / CELL), by = Math.floor(Math.max(ys[i], ys[i + 1]) / CELL);
      for (let cx = ax; cx <= bx; cx++) for (let cy = ay; cy <= by; cy++) {
        const k = key(cx, cy), c = grid.get(k);
        if (c) c.push(i);
        else grid.set(k, [i]);
      }
    }
    const edges = api.edges_info();
    for (let i = 0; i < edges.length; i += 10) {
      const e = edges[i], lines = linesOf.get(e);
      if (!lines) continue;
      const a = nodeXY.get(edges[i + 1]), len = edges[i + 5];
      if (!a || a[0] < x0 - len || a[0] > x1 + len || a[1] < y0 - len || a[1] > y1 + len) continue;
      if (!(await pause())) return none;
      const r = api.edge_render(e), n = r.length / 3;
      let along = 0, total = 0;
      for (let j = 0; j + 1 < n; j++) total += Math.hypot(r[j * 3 + 3] - r[j * 3], r[j * 3 + 4] - r[j * 3 + 1]);
      const hitS: number[] = [];
      for (let j = 0; j + 1 < n; j++) {
        const px = r[j * 3], py = r[j * 3 + 1], qx = r[j * 3 + 3], qy = r[j * 3 + 4];
        const segLen = Math.hypot(qx - px, qy - py);
        const ax = Math.floor(Math.min(px, qx) / CELL), bx = Math.floor(Math.max(px, qx) / CELL);
        const ay = Math.floor(Math.min(py, qy) / CELL), by = Math.floor(Math.max(py, qy) / CELL);
        const seen = new Set<number>();
        for (let cx = ax; cx <= bx; cx++) for (let cy = ay; cy <= by; cy++) {
          for (const si of grid.get(key(cx, cy)) ?? []) {
            if (seen.has(si)) continue;
            seen.add(si);
            const hit = intersect(xs[si], ys[si], xs[si + 1], ys[si + 1], px, py, qx, qy);
            if (!hit) continue;
            const ux = xs[si + 1] - xs[si], uy = ys[si + 1] - ys[si];
            if (Math.abs(ux * (qy - py) - uy * (qx - px)) < SHALLOW_SIN * Math.hypot(ux, uy) * segLen) continue;
            const [t, u] = hit, at = along + u * segLen;
            if (at < JUNCTION_M || at > total - JUNCTION_M) continue;
            const sAt = s[si] + t * (s[si + 1] - s[si]);
            // a crossing on a shared vertex is found by both segments either side
            if (hitS.some((v) => Math.abs(v - sAt) < 1)) continue;
            hitS.push(sAt);
            crosses.push(sAt, r[j * 3 + 2] + u * (r[j * 3 + 5] - r[j * 3 + 2]), lines.length, ...lines);
            nC++;
          }
        }
        along += segLen;
      }
    }
  }

  // water: the zones' masks read every WATER_STEP_M, as stretches of water and of no mask
  const xy: number[] = [];
  for (let i = 0; i < nP; i++) xy.push(xs[i], ys[i]);
  const masks = await masksUnder(xy).catch(() => []);
  if (!current()) return none;
  const wet: number[] = [];
  let runKind = 0, runFrom = 0, i = 0;
  const len = nP ? s[nP - 1] : 0;
  for (let at = 0; nP >= 2 && at <= len + WATER_STEP_M / 2; at += WATER_STEP_M) {
    if (!(await pause())) return none;
    const a = Math.min(at, len);
    while (i + 2 < nP && s[i + 1] < a) i++;
    const t = s[i + 1] > s[i] ? (a - s[i]) / (s[i + 1] - s[i]) : 0;
    const [lng, lat] = lngLat(xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]));
    const kind = waterAt(masks, lng, lat);
    if (kind !== runKind) {
      if (runKind) wet.push(runFrom, a, runKind);
      runKind = kind, runFrom = a;
    }
  }
  if (runKind) wet.push(runFrom, len, runKind);

  const out = new Float64Array(4 + nP * 4 + stops.length + crosses.length + wet.length);
  out[0] = nP, out[1] = stops.length / 2, out[2] = nC, out[3] = wet.length / 3;
  let k = 4;
  for (let q = 0; q < nP; q++) {
    const [lng, lat] = lngLat(xs[q], ys[q]);
    out[k++] = s[q], out[k++] = lng, out[k++] = lat, out[k++] = hs[q];
  }
  out.set(stops, k);
  out.set(crosses, k + stops.length);
  out.set(wet, k + stops.length + crosses.length);
  return out;
}

/** Where segments ab and pq cross: the share along each, or null (touching ends count). */
function intersect(ax: number, ay: number, bx: number, by: number, px: number, py: number, qx: number, qy: number): [number, number] | null {
  const rx = bx - ax, ry = by - ay, sx = qx - px, sy = qy - py;
  const d = rx * sy - ry * sx;
  if (Math.abs(d) < 1e-9) return null;
  const t = ((px - ax) * sy - (py - ay) * sx) / d, u = ((px - ax) * ry - (py - ay) * rx) / d;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? [t, u] : null;
}
