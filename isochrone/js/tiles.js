// L3 and L1 tiles (data/tiles/): fetched on demand, kept in a cache, cut into a window's grid.
// The fetch URL is relative to the worker entry, worker.js.

let tiles = null;                // data/tiles/index.json, with a Set of tiles per level

export function setTiles(index) {
    tiles = index;
    for (const k in tiles.levels) tiles.levels[k].set = new Set(tiles.levels[k].tiles);
}

const tileCache = new Map();     // "f/tr_tc" -> Promise<Uint8Array>, oldest first
let cacheBytes = 0;
function getTile(f, tr, tc) {
    const key = f + '/' + tr + '_' + tc;
    let p = tileCache.get(key);
    if (p) { tileCache.delete(key); tileCache.set(key, p); return p; }
    const lv = tiles.levels[f];
    p = fetch('data/tiles/' + lv.path + '/' + tr + '_' + tc + '.bin.gz')
        .then(r => { if (!r.ok) throw new Error('tile ' + key + ' ' + r.status);
                     return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(); })
        .then(b => new Uint8Array(b));
    p.catch(() => tileCache.delete(key));
    tileCache.set(key, p);
    cacheBytes += lv.tile * lv.tile;
    while (cacheBytes > 200e6 && tileCache.size > 1) {
        const k = tileCache.keys().next().value;
        tileCache.delete(k);
        const s = tiles.levels[k.split('/')[0]].tile;
        cacheBytes -= s * s;
    }
    return p;
}

export async function loadWindow(w) {
    const lv = tiles.levels[w.f], T = lv.tile, TX = tiles.tilesX;
    const grid = new Uint8Array(w.cols * w.rows);
    const jobs = [];
    for (let tr = Math.floor(w.row0 / T); tr <= Math.floor((w.row0 + w.rows - 1) / T); tr++) {
        for (let tc = Math.floor(w.col0 / T); tc <= Math.floor((w.col0 + w.cols - 1) / T); tc++) {
            const tcw = ((tc % TX) + TX) % TX;
            if (!lv.set.has(tr * TX + tcw)) continue;
            jobs.push(getTile(w.f, tr, tcw).then(t => {
                // overlap of tile (tr, tc) with the window, in window cells
                const r0 = Math.max(0, tr * T - w.row0), r1 = Math.min(w.rows, (tr + 1) * T - w.row0);
                const c0 = Math.max(0, tc * T - w.col0), c1 = Math.min(w.cols, (tc + 1) * T - w.col0);
                for (let r = r0; r < r1; r++) {
                    const src = (r + w.row0 - tr * T) * T + (c0 + w.col0 - tc * T);
                    grid.set(t.subarray(src, src + (c1 - c0)), r * w.cols + c0);
                }
            }));
        }
    }
    await Promise.all(jobs);
    return grid;
}
