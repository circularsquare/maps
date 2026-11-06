// T-080 harness, reused for T-082: the worker's modules in node over the real data (as
// checks/ktrips.mjs). CANOE=<folder>/ reads another canoe data set (the A/B of T-082).
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL, fileURLToPath } from 'node:url';

// Run from this folder (node measure.mjs ...); results go to ./results/. The T-080 runs read a
// copy of data/canoe/ taken 2026-11-05 (files of 2026-10-26): CANOE=<folder> to do the same.
// T-088: js/ from JSDIR (default the repo's, for a copy being tried), results in ./results/;
// WACC scales the water units' access (a variant knob of the prototype, gone from js/ since)
export const ROOT = fileURLToPath(new URL('../../', import.meta.url)).replace(/\\/g, '/');
export const SCR = fileURLToPath(new URL('./', import.meta.url)).replace(/\\/g, '/');
export const JS = process.env.JSDIR || ROOT + 'js/';
const CANOE = process.env.CANOE || ROOT + 'data/canoe/';

globalThis.fetch = async (u) => {
    let p = String(u);
    if (p.startsWith('file://')) p = p.slice(7);
    else if (!/^[A-Za-z]:\//.test(p)) p = ROOT + p;
    try { return new Response(readFileSync(p)); } catch { return new Response(null, { status: 404 }); }
};
const imp = (m) => import(pathToFileURL(JS + m).href);
export { imp };
export const { cfg, setConfig } = await imp('config.js');
export const { makeLevel, snap, setHazard, sampleNode } = await imp('levels.js');
export const { solve, cellTimes } = await imp('search.js');
export const { loadWorld, loadWindowWater } = await imp('canoedata.js');
export const { buildCanoe, shoreAt } = await imp('canoe.js');
export const T = await imp('tiles.js');
export const Wn = await imp('windows.js');

export const meta = JSON.parse(readFileSync(ROOT + 'data/grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: CANOE, hazardBase: ROOT + 'data/hazard/' });
if (process.env.WACC) cfg.waterAccess = Number(process.env.WACC);
if (process.env.ACC) cfg.access = Number(process.env.ACC);
if (process.env.SFREE) cfg.startFree = true;
T.setTiles(JSON.parse(readFileSync(ROOT + 'data/tiles/index.json', 'utf8')));

const g10 = gunzipSync(readFileSync(ROOT + 'data/grid.bin.gz'));
export const grid10 = new Uint8Array(g10.buffer, g10.byteOffset, g10.length);
if (cfg.roads && meta.roads) T.applyRoads(grid10, gunzipSync(readFileSync(ROOT + 'data/' + meta.roads)));
let world = null;
// The L10 world level (wraps), with land hazard, and its canoe layer on demand.
export async function world10(canoe) {
    if (!world) {
        const cell = (meta.north - meta.south) / meta.height;
        world = makeLevel(grid10, meta.width, meta.height, meta.west, meta.north, cell, true);
        world.hzCodes = await T.loadHazardWorld();
        setHazard(world, world.hzCodes);
    }
    if (canoe && !world.cn) { const d = await loadWorld(world.W, world.H); buildCanoe(world, d, d.sea); world.sea = d.sea; }
    return world;
}

// A box at level f (cells of f/120 degree), edges on 0.25 degree.
export function win(f, west, south, east, north) {
    const cell = f / 120;
    return { f, cell, west, east, north, south, col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
}
export function boxAround(lat, lng, halfLat, halfLon) {
    const a = 0.25;
    return { west: Math.floor((lng - halfLon) / a) * a, east: Math.ceil((lng + halfLon) / a) * a,
             south: Math.max(-60, Math.floor((lat - halfLat) / a) * a), north: Math.min(85, Math.ceil((lat + halfLat) / a) * a) };
}

// L10 cut to a box (no wrap needed for our boxes), speeds and hazard codes.
export async function cut10(w) {
    const W = await world10(false);
    const g = new Uint8Array(w.cols * w.rows), h = new Uint8Array(w.cols * w.rows);
    for (let r = 0; r < w.rows; r++) for (let c = 0; c < w.cols; c++) {
        const i = (w.row0 + r) * W.W + ((w.col0 + c) % W.W + W.W) % W.W;
        g[r * w.cols + c] = grid10[i]; h[r * w.cols + c] = W.hzCodes[i];
    }
    return { grid: g, hz: h };
}

// The speed grid and hazard band of a box at level f (10, 3, 1, 0.1), from the page's data.
export async function loadBox(w) {
    if (w.f === 10) return cut10(w);
    const [grid, hz] = await Promise.all([T.loadWindow(w), T.loadHazardWindow(w)]);
    return { grid, hz };
}

export function level(w, grid, hz) {
    const L = makeLevel(grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    if (hz) setHazard(L, hz); else L.hq = null;
    return L;
}

// Solve a level from a point, start seed only (no ring). canoe: build the window's canoe layer.
export async function solveFrom(L, lat, lng, canoe = false) {
    if (canoe && !L.cn) {
        const W = await world10(true);
        const data = await loadWindowWater(L.win, L.grid, (x, y) => { const s = shoreAt(W, x, y); return s === undefined ? 78 : s; });
        buildCanoe(L, data, W.sea);
    }
    const s = snap(L, lng, lat, L.win ? Math.ceil(10 / L.win.f) : 8);
    if (s < 0) return null;
    const res = await solve(L, { idx: [s], w: [0], t: [0], hz: [0] }, () => true, canoe);
    L.res = res;
    L.dist = cellTimes(L, res, canoe);
    return L.dist;
}

// Time at the cell holding a point; a water cell takes the least time of its 3 x 3 (reported as `near`).
export function timeAt(L, lat, lng) {
    const W = L.W, H = L.H;
    let c = Math.floor((lng - L.west) / L.cellDeg), r = Math.floor((L.north - lat) / L.cellDeg);
    if (L.wrap) c = ((c % W) + W) % W;
    if (r < 0 || r >= H || c < 0 || c >= W) return { t: NaN, near: false };
    const i = r * W + c;
    if (L.dist[i] < Infinity) return { t: L.dist[i], near: false };
    let best = Infinity;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
        best = Math.min(best, L.dist[rr * W + cc]);
    }
    return { t: best, near: true };
}

export const KM = 111.195;
export function offset(lat, lng, km, bearingDeg) {
    const b = bearingDeg * Math.PI / 180;
    const dlat = km * Math.cos(b) / KM, dlon = km * Math.sin(b) / (KM * Math.cos(lat * Math.PI / 180));
    return [lat + dlat, lng + dlon];
}

// Upscale a speed grid (uint8 km/h x 40) by k: mean time per metre over land cells, land where a
// quarter of the cells are (tools/build_grid.py).
export function upscale(g, W, H, k) {
    const w = W / k, h = H / k, out = new Uint8Array(w * h);
    for (let R = 0; R < h; R++) for (let C = 0; C < w; C++) {
        let s = 0, n = 0;
        for (let r = R * k; r < R * k + k; r++) for (let c = C * k; c < C * k + k; c++) { const v = g[r * W + c]; if (v) { s += 1 / v; n++; } }
        if (n >= 0.25 * k * k) out[R * w + C] = Math.max(1, Math.min(255, Math.round(n / s)));
    }
    return out;
}

// Seeded random
export function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

export function quant(a, p) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
