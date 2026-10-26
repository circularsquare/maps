// L3 and L1 tiles (data/tiles/) and L0.1 tiles (data/fine/): fetched on demand, kept in a
// cache, cut into a window's grid.
// The fetch URL is relative to the worker entry, worker.js.

let tiles = null;                // data/tiles/index.json, with a Set of tiles per level

export function setTiles(index) {
    tiles = index;
    for (const k in tiles.levels) tiles.levels[k].set = new Set(tiles.levels[k].tiles);
}

// A gzipped file as bytes.
export async function fetchGz(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ' ' + r.status);
    return new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
}

// One cache for speed and canoe tiles: url -> { p: Promise<Uint8Array>, bytes (expected size) },
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

function getTile(f, tr, tc) {
    const lv = tiles.levels[f];
    return getCached('data/tiles/' + lv.path + '/' + tr + '_' + tc + '.bin.gz', lv.tile * lv.tile);
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
    const grid = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    forTiles(w, T, TX, (tr, tc, tcw) => {
        if (!lv.set.has(tr * TX + tcw)) return;
        jobs.push(getTile(w.f, tr, tcw).then(t => cutInto(grid, w, t, 0, T, tr, tc)));
    });
    await Promise.all(jobs);
    return grid;
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

let fine;                                // data/fine/index.json with a Set, or null when there is none
export function fineIndex() {
    if (fine === undefined) fine = fetch('data/fine/index.json')
        .then(r => r.ok ? r.json() : null)
        .then(ix => { if (ix) { ix.set = new Set(ix.tiles); ix.roadSet = new Set(ix.roads); } return ix; })
        .catch(() => null);
    return fine;
}

// tile row tr (0 at 85N), column tcw (0 at 180W) -> its name, as n40w074 (south-west corner)
export function fineName(tr, tcw) {
    const lat = 84 - tr, lon = tcw - 180;
    return (lat >= 0 ? 'n' : 's') + String(Math.abs(lat)).padStart(2, '0') +
           (lon >= 0 ? 'e' : 'w') + String(Math.abs(lon)).padStart(3, '0');
}

// Every tile of window w exists.
export function fineCovers(ix, w) {
    let ok = true;
    forTiles(w, ix.tile, 360, (tr, tc, tcw) => { if (!ix.set.has(fineName(tr, tcw))) ok = false; });
    return ok;
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

async function loadFineWindow(w) {
    const ix = await fineIndex(), T = ix.tile;
    const grid = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    forTiles(w, T, 360, (tr, tc, tcw) => {
        const name = fineName(tr, tcw);
        if (!ix.set.has(name)) return;
        jobs.push(fineTile(ix, name).then(t => cutInto(grid, w, t, 0, T, tr, tc)));
    });
    await Promise.all(jobs);
    return grid;
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
