// L3 and L1 tiles (data/tiles/: speeds and road bytes), L0.1 tiles (data/fine/) and the land
// hazard band (data/hazard/): fetched on demand, kept in a cache, cut into a window's grid.
// The fetch URL is relative to the worker entry, worker.js.
import { cfg } from './config.js';

let tiles = null;                // data/tiles/index.json, with a Set of tiles per level

export function setTiles(index) {
    tiles = index;
    for (const k in tiles.levels) {
        const lv = tiles.levels[k];
        lv.set = new Set(lv.tiles);
        if (lv.roads) lv.roads.set = new Set(lv.roads.tiles);
    }
}

// Roads (docs/roads.md, "The page part") folded into a level's speeds, in place. The road byte b
// of a land cell is its time with roads over its time without, x 250; of a water cell, a walkable
// bridge's speed over 5 km/h, x 250, which makes the cell land; 0 no road. Speeds are km/h x 40.
const BRIDGE_PER_B = 5 * 40 / 250;
export function applyRoads(grid, roads) {
    for (let i = 0, n = grid.length; i < n; i++) {
        const b = roads[i];
        if (!b) continue;
        const s = grid[i];
        const v = s ? Math.round(s * 250 / b) : Math.round(b * BRIDGE_PER_B);
        grid[i] = v < 1 ? 1 : v > 255 ? 255 : v;
    }
    return grid;
}

// A gzipped file as bytes.
export async function fetchGz(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ' ' + r.status);
    return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

// One cache for speed, canoe and hazard tiles: url -> { p: Promise<Uint8Array>, bytes (expected size) },
// oldest first.
const tileCache = new Map();
let cacheBytes = 0;
export function getCached(url, bytes) {
    let e = tileCache.get(url);
    if (e) { tileCache.delete(url); tileCache.set(url, e); return e.p; }
    e = { p: fetchGz(url), bytes };
    e.p.catch(() => { if (tileCache.get(url) === e) { tileCache.delete(url); cacheBytes -= e.bytes; } });
    tileCache.set(url, e);
    cacheBytes += bytes;
    while (cacheBytes > 250e6 && tileCache.size > 1) {
        const [k, old] = tileCache.entries().next().value;
        tileCache.delete(k);
        cacheBytes -= old.bytes;
    }
    return e.p;
}

function getTile(f, path, tr, tc) {
    const T = tiles.levels[f].tile;
    return getCached('data/tiles/' + path + '/' + tr + '_' + tc + '.bin.gz', T * T);
}

// Each 5-degree tile overlapping window w at tile size T: calls fn(tr, tc, tcw) with tc
// unwrapped (for window positions) and tcw wrapped (for file names).
export function forTiles(w, T, TX, fn) {
    for (let tr = Math.floor(w.row0 / T); tr <= Math.floor((w.row0 + w.rows - 1) / T); tr++)
        for (let tc = Math.floor(w.col0 / T); tc <= Math.floor((w.col0 + w.cols - 1) / T); tc++)
            fn(tr, tc, ((tc % TX) + TX) % TX);
}

export async function loadWindow(w) {
    if (w.f === 0.1) return loadFineWindow(w);
    const lv = tiles.levels[w.f], T = lv.tile, TX = tiles.tilesX;
    const rd = cfg.roads && lv.roads;
    const grid = new Uint8Array(w.cols * w.rows), roads = rd ? new Uint8Array(w.cols * w.rows) : null;
    const jobs = [];
    forTiles(w, T, TX, (tr, tc, tcw) => {
        const id = tr * TX + tcw;
        if (lv.set.has(id)) jobs.push(getTile(w.f, lv.path, tr, tcw).then(t => cutInto(grid, w, t, 0, T, tr, tc)));
        if (rd && rd.set.has(id)) jobs.push(getTile(w.f, rd.path, tr, tcw).then(t => cutInto(roads, w, t, 0, T, tr, tc)));
    });
    await Promise.all(jobs);
    return roads ? applyRoads(grid, roads) : grid;
}

// ---- L0.1: 1 x 1 degree tiles of data/fine/ (docs/fine.md), turned into speeds here ----

// LC100 class -> off-road km/h (docs/surface.md, tools/surface/common.py LC_SPEED); 0 = not land
const LC_KMH = new Float32Array(256);
for (const [cls, v] of [[111, 3.24], [113, 3.24], [112, 1.62], [114, 4.0], [115, 3.24], [116, 3.24],
    [121, 4.0], [123, 4.0], [125, 4.0], [124, 4.86], [126, 4.86], [122, 3.0], [20, 4.2], [30, 4.86],
    [40, 2.5], [50, 5.0], [60, 3.0], [70, 1.62], [90, 2.0], [100, 4.86]]) LC_KMH[cls] = v;
const ROAD_WALK = 7, ROAD_BRIDGE = 16;   // path, local, main; a walkable bridge (tools/fine/build.py)
const ROAD_KMH = 5;                      // MAP's road speed, any class
const SCALE = 40;                        // stored speed = km/h x 40, as every level

// data/fine/index.json with mask (tile name -> block mask), or null when there is none. A town
// tile holds only some 0.25 degree blocks: bit 4 i + j for block row i from the north, column j
// from the west; the other blocks are all 0 (docs/fine.md, "Town blocks").
const FULL = 0xFFFF;
let fine;
export function fineIndex() {
    if (fine === undefined) fine = fetch('data/fine/index.json')
        .then(r => r.ok ? r.json() : null)
        .then(ix => {
            if (!ix) return null;
            ix.mask = new Map(ix.tiles.map(n => [n, FULL]));
            for (const n in ix.partial || {}) ix.mask.set(n, ix.partial[n]);
            ix.block = ix.block || ix.tile;
            ix.roadSet = new Set(ix.roads);
            return ix;
        })
        .catch(() => null);
    return fine;
}

// Calls fn(r0, r1, c0, c1, present) for each block overlapping rows [row0, row1) and columns
// [col0, col1) of the L0.1 world grid (row 0 at 85 N, column 0 at 180 W), clipped to them.
function forBlocks(ix, row0, row1, col0, col1, fn) {
    const T = ix.tile, B = ix.block, nb = T / B;
    for (let br = Math.floor(row0 / B); br * B < row1; br++)
        for (let bc = Math.floor(col0 / B); bc * B < col1; bc++) {
            const tr = Math.floor(br / nb), tc = Math.floor(bc / nb);
            const m = ix.mask.get(fineName(tr, ((tc % 360) + 360) % 360)) || 0;
            const bit = (br - tr * nb) * nb + (bc - tc * nb);
            fn(Math.max(row0, br * B), Math.min(row1, (br + 1) * B),
               Math.max(col0, bc * B), Math.min(col1, (bc + 1) * B), (m >> bit) & 1);
        }
}

// Share of the box (west, east, north, south, degrees) in fine blocks, by cell count.
export function fineShare(ix, b) {
    const k = ix.tile;
    const row0 = Math.floor((85 - b.north) * k), row1 = Math.ceil((85 - b.south) * k);
    const col0 = Math.floor((b.west + 180) * k), col1 = Math.ceil((b.east + 180) * k);
    let n = 0;
    forBlocks(ix, row0, row1, col0, col1, (r0, r1, c0, c1, on) => { if (on) n += (r1 - r0) * (c1 - c0); });
    return n / Math.max(1, (row1 - row0) * (col1 - col0));
}

// tile row tr (0 at 85N), column tcw (0 at 180W) -> its name, as n40w074 (south-west corner)
export function fineName(tr, tcw) {
    const lat = 84 - tr, lon = tcw - 180;
    return (lat >= 0 ? 'n' : 's') + String(Math.abs(lat)).padStart(2, '0') +
           (lon >= 0 ? 'e' : 'w') + String(Math.abs(lon)).padStart(3, '0');
}

// Decoded speed tiles, name -> Promise<Uint8Array>, oldest first, ~100 MB.
const fineCache = new Map();
function fineTile(ix, name) {
    let p = fineCache.get(name);
    if (p) { fineCache.delete(name); fineCache.set(name, p); return p; }
    const T = ix.tile;
    p = Promise.all([fetchGz('data/fine/' + name + '.land.bin.gz'),
                     ix.roadSet.has(name) ? fetchGz('data/fine/' + name + '.roads.bin.gz') : null])
        .then(([land, roads]) => fineSpeed(land, roads, T, ix.terrain, ix.tfScale));
    p.catch(() => { if (fineCache.get(name) === p) fineCache.delete(name); });
    fineCache.set(name, p);
    while (fineCache.size * T * T > 100e6) fineCache.delete(fineCache.keys().next().value);
    return p;
}

// Land cover speed, raised to road speed where a walkable way crosses land; water is 0 except
// under a walkable bridge; times the L1 terrain factor of the cell's L1 cell.
function fineSpeed(land, roads, T, TF, tfScale) {
    const out = new Uint8Array(T * T), k = T / TF, tf = land.subarray(T * T);
    for (let r = 0; r < T; r++) {
        const tfRow = ((r / k) | 0) * TF;
        for (let c = 0, i = r * T; c < T; c++, i++) {
            const cls = land[i], rd = roads ? roads[i] : 0;
            let kmh = LC_KMH[cls];
            if (cls === 80 || cls === 200) kmh = rd & ROAD_BRIDGE ? ROAD_KMH : 0;
            else if (rd & ROAD_WALK) kmh = Math.max(kmh, ROAD_KMH);
            if (!kmh) continue;
            const v = Math.round(kmh * tf[tfRow + ((c / k) | 0)] / tfScale * SCALE);
            out[i] = v < 1 ? 1 : v > 255 ? 255 : v;
        }
    }
    return out;
}

// The window's speeds: fine blocks from their tiles; every other block (absent from a partial
// tile, or no tile) takes the speed of the L1 cell it lies in, roads folded in as at L1, so the
// window needs only some fine blocks.
async function loadFineWindow(w) {
    const ix = await fineIndex(), T = ix.tile;
    const grid = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    forTiles(w, T, 360, (tr, tc, tcw) => {
        const name = fineName(tr, tcw);
        if (!ix.mask.has(name)) return;
        jobs.push(fineTile(ix, name).then(t => cutInto(grid, w, t, 0, T, tr, tc)));
    });
    const gaps = [];
    forBlocks(ix, w.row0, w.row0 + w.rows, w.col0, w.col0 + w.cols, (r0, r1, c0, c1, on) => {
        if (!on) gaps.push([r0 - w.row0, r1 - w.row0, c0 - w.col0, c1 - w.col0]);
    });
    let l1 = null, w1 = null;
    if (gaps.length) {
        const k = 10;   // L0.1 cells per L1 cell
        const col0 = Math.floor(w.col0 / k), row0 = Math.floor(w.row0 / k);
        w1 = { f: 1, cell: 1 / 120, col0, row0, cols: Math.floor((w.col0 + w.cols - 1) / k) - col0 + 1,
               rows: Math.floor((w.row0 + w.rows - 1) / k) - row0 + 1 };
        jobs.push(loadWindow(w1).then(g => { l1 = g; }));
    }
    await Promise.all(jobs);
    for (const [r0, r1, c0, c1] of gaps)
        for (let r = r0; r < r1; r++) {
            const src = (Math.floor((w.row0 + r) / 10) - w1.row0) * w1.cols - w1.col0;
            for (let c = c0, i = r * w.cols + c0; c < c1; c++, i++) grid[i] = l1[src + Math.floor((w.col0 + c) / 10)];
        }
    return grid;
}

// ---- land hazard (docs/hazard.md 4): data/hazard/, uint8 codes on the speed tiles' grid ----
// A tile the index does not list has no hazard anywhere (all 0).

let hzIndex = null;
const hzUrl = (p) => cfg.hazardBase + p;
function hazardIndex() {
    return hzIndex || (hzIndex = fetch(hzUrl('index.json')).then(r => {
        if (!r.ok) throw new Error('hazard index ' + r.status);
        return r.json();
    }).then(ix => {
        for (const k in ix.levels) ix.levels[k].set = new Set(ix.levels[k].tiles);
        return ix;
    }).catch(e => { hzIndex = null; throw e; }));
}

// The L10 world band, row-major from the north-west, as the speed grid.
export async function loadHazardWorld() {
    const ix = await hazardIndex();
    return fetchGz(hzUrl(ix.world.file));
}

// The band cut into window w; an L0.1 window takes the L1 pixel under each cell.
export async function loadHazardWindow(w) {
    const ix = await hazardIndex();
    if (w.f === 0.1) {
        const k = 10;   // L0.1 cells per L1 cell
        const col0 = Math.floor(w.col0 / k), row0 = Math.floor(w.row0 / k);
        const w1 = { f: 1, col0, row0, cols: Math.floor((w.col0 + w.cols - 1) / k) - col0 + 1,
                     rows: Math.floor((w.row0 + w.rows - 1) / k) - row0 + 1 };
        const c1 = await hazardCut(ix, w1), out = new Uint8Array(w.cols * w.rows);
        for (let r = 0; r < w.rows; r++) {
            const src = (Math.floor((w.row0 + r) / k) - row0) * w1.cols;
            for (let c = 0, i = r * w.cols; c < w.cols; c++, i++) out[i] = c1[src + Math.floor((w.col0 + c) / k) - col0];
        }
        return out;
    }
    return hazardCut(ix, w);
}

async function hazardCut(ix, w) {
    const lv = ix.levels[w.f], T = lv.tile, TX = ix.tilesX;
    const out = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!lv.set.has(tr * TX + tcw)) return;
        jobs.push(getCached(hzUrl(lv.path + '/' + tr + '_' + tcw + '.bin.gz'), T * T).then(t => cutInto(out, w, t, 0, T, tr, tc)));
    });
    await Promise.all(jobs);
    return out;
}

// Copy band `band` of a T x T tile (bands one after another) into a window-sized array.
export function cutInto(dst, w, t, band, T, tr, tc) {
    const r0 = Math.max(0, tr * T - w.row0), r1 = Math.min(w.rows, (tr + 1) * T - w.row0);
    const c0 = Math.max(0, tc * T - w.col0), c1 = Math.min(w.cols, (tc + 1) * T - w.col0);
    const base = band * T * T;
    for (let r = r0; r < r1; r++) {
        const src = base + (r + w.row0 - tr * T) * T + (c0 + w.col0 - tc * T);
        dst.set(t.subarray(src, src + (c1 - c0)), r * w.cols + c0);
    }
}
