// L3 and L1 tiles (data/tiles/): fetched on demand, kept in a cache, cut into a window's grid.
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
