// T-088's harness (notes/T-088/lib.mjs) with T-011's directed walking: the worker's modules in node
// over the real data, as checks/levels.mjs reads it. Kept for re-measuring after canoe data changes (T-102).
//   TILES=<name>  walking levels from data/tiles_<name>/ (e.g. pre_t011), else the served data/
//   ACC=<x>       cfg.access (the knob of T-082)
//   OUT=<folder>/ where results/ goes (default this folder)
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL, fileURLToPath } from 'node:url';

export const ROOT = fileURLToPath(new URL('../../', import.meta.url)).replace(/\\/g, '/');
export const SCR = (process.env.OUT || fileURLToPath(new URL('./', import.meta.url))).replace(/\\/g, '/').replace(/\/?$/, '/');
export const JS = process.env.JSDIR || ROOT + 'js/';
const SUB = (process.env.TILES || '').replace(/[^a-z0-9_-]/gi, '');
export const BASE = ROOT + 'data/' + (SUB ? 'tiles_' + SUB + '/' : '');
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
export const { makeLevel, snap, setHazard, sampleNode, setDirs } = await imp('levels.js');
export const { solve, cellTimes } = await imp('search.js');
export const { loadWorld, loadWindowWater } = await imp('canoedata.js');
export const { buildCanoe, shoreAt } = await imp('canoe.js');
export const T = await imp('tiles.js');

export const meta = JSON.parse(readFileSync(BASE + 'grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: CANOE, hazardBase: ROOT + 'data/hazard/' });
if (process.env.ACC) cfg.access = Number(process.env.ACC);
T.setTiles(JSON.parse(readFileSync(BASE + 'tiles/index.json', 'utf8')), BASE + 'tiles/');

const gz = (f) => { const b = gunzipSync(readFileSync(BASE + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
export const grid10 = gz('grid.bin.gz');
if (cfg.roads && meta.roads) T.applyRoads(grid10, gz(meta.roads));
let world = null;
// The L10 world level (wraps), with land hazard and directions, and its canoe layer on demand.
export async function world10(canoe) {
    if (!world) {
        const cell = (meta.north - meta.south) / meta.height;
        world = makeLevel(grid10, meta.width, meta.height, meta.west, meta.north, cell, true);
        setDirs(world, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);
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

// The speed grid, direction planes and hazard band of an L3 or L1 box, from the page's data.
export async function loadBox(w) {
    const dirs = {};
    const [grid, hz] = await Promise.all([T.loadWindow(w, dirs), T.loadHazardWindow(w)]);
    return { grid, hz, dirs };
}

export function level(w, box) {
    const L = makeLevel(box.grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    setDirs(L, box.dirs.d, box.dirs.e);
    if (box.hz) setHazard(L, box.hz); else L.hq = null;
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

// Seeded random
export function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

export function quant(a, p) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
