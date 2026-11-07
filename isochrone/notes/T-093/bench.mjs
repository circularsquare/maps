// T-093: the heap search against js/search.js, on the page's levels in node, by CPU time (BY=wall
// for wall time). The heap search is search_heap_src.js beside this file, which is not kept:
//   git show 86b67450:isochrone/js/search.js > isochrone/notes/T-093/search_heap_src.js
//   node isochrone/notes/T-093/bench.mjs [--only=A,B] [--reps=3] [--extra=other_search.js,...]
// (writes search_heap.mjs and the extras' copies here, imports rewritten to js/).
import { readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = 'C:/Users/anita/projects/maps/isochrone/';
const SCR = fileURLToPath(new URL('./', import.meta.url)).replace(/\\/g, '/');
const ARGS = process.argv.slice(2);
const opt = (k, d) => { const a = ARGS.find((x) => x.startsWith('--' + k + '=')); return a ? a.slice(k.length + 3) : d; };
const ONLY = opt('only', '') ? opt('only', '').split(',') : null;
const REPS = +opt('reps', '3');
const HEAPSRC = opt('heap', 'search_heap_src.js');
const NEWSRC = opt('new', '');

globalThis.fetch = async (u) => {
    let p = String(u);
    if (p.startsWith('file://')) p = p.slice(7);
    else if (!/^[A-Za-z]:\//.test(p)) p = ROOT + p;
    try { return new Response(readFileSync(p)); } catch { return new Response(null, { status: 404 }); }
};
const JSURL = pathToFileURL(ROOT + 'js/').href;
const imp = (m) => import(JSURL + m);
const rewrite = (src, out) => { writeFileSync(SCR + out, readFileSync(SCR + src, 'utf8').replaceAll("from './", "from '" + JSURL)); return import(pathToFileURL(SCR + out).href); };
const HEAP = await rewrite(HEAPSRC, 'search_heap.mjs');
const NEW = NEWSRC ? await rewrite(NEWSRC, 'search_new.mjs') : await imp('search.js');
const EXTRA = opt('extra', '') ? opt('extra', '').split(',') : [];
const XMODS = [];
for (const [i, f] of EXTRA.entries()) XMODS.push([f, await rewrite(f, `search_x${i}.mjs`)]);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, sampleNode, setDirs, setSlope, setHazard } = await imp('levels.js');
const T = await imp('tiles.js');
const { loadWorld, loadWindowWater } = await imp('canoedata.js');
const { buildCanoe, shoreAt } = await imp('canoe.js');

const meta = JSON.parse(readFileSync(ROOT + 'data/grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: ROOT + 'data/canoe/', hazardBase: ROOT + 'data/hazard/', fineBase: 'data/fine/' });
T.setTiles(JSON.parse(readFileSync(ROOT + 'data/tiles/index.json', 'utf8')), ROOT + 'data/tiles/');
const gz = (f) => { const b = gunzipSync(readFileSync(ROOT + 'data/' + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };

let world = null;
async function world10(canoe) {
    if (!world) {
        const g = gz('grid.bin.gz');
        if (cfg.roads && meta.roads) T.applyRoads(g, gz(meta.roads));
        world = makeLevel(g, meta.width, meta.height, meta.west, meta.north, (meta.north - meta.south) / meta.height, true);
        setDirs(world, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);
        setHazard(world, await T.loadHazardWorld());
    }
    if (canoe && !world.cn) { const d = await loadWorld(world.W, world.H); buildCanoe(world, d, d.sea); world.sea = d.sea; }
    return world;
}
const win = (f, west, south, east, north) => {
    const cell = f / 120;
    return { f, cell, west, east, north, south, col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
};
async function windowLevel(w, canoe) {
    const dirs = {};
    const [grid, hzc] = await Promise.all([T.loadWindow(w, dirs), T.loadHazardWindow(w)]);
    const L = makeLevel(grid, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    setDirs(L, dirs.d, dirs.e);
    if (dirs.z) setSlope(L, dirs.z, dirs.g, dirs.slope);
    setHazard(L, hzc);
    if (canoe) {
        const W10 = await world10(true);
        const data = await loadWindowWater(w, grid, (x, y) => { const s = shoreAt(W10, x, y); return s === undefined ? 78 : s; });
        buildCanoe(L, data, W10.sea);
    }
    return L;
}
const startSeeds = (L, lng, lat, R) => { const s = snap(L, lng, lat, R); return { idx: [s], w: [0], t: [0], hz: [0] }; };
function ringSeeds(L, C, origin) {
    const w = L.win, seeds = { idx: [], w: [], t: [], hz: [], pred: [] }, out = [0, 0, 0];
    const add = (r, c) => {
        const i = r * w.cols + c;
        if (!L.grid[i]) return;
        const got = sampleNode(C, w.west + (c + 0.5) * w.cell, w.north - (r + 0.5) * w.cell, false, out);
        if (got && out[1] <= cfg.maxHours && out[2] <= cfg.hzStop) { seeds.idx.push(i); seeds.w.push(out[0]); seeds.t.push(out[1]); seeds.hz.push(out[2]); seeds.pred.push(255); }
    };
    for (let c = 0; c < w.cols; c++) { add(0, c); add(w.rows - 1, c); }
    for (let r = 1; r < w.rows - 1; r++) { add(r, 0); add(r, w.cols - 1); }
    const s = snap(L, origin[0], origin[1], 100);
    if (s >= 0) { seeds.idx.push(s); seeds.w.push(0); seeds.t.push(0); seeds.hz.push(0); seeds.pred.push(0); }
    return seeds;
}

const CLUJ = [23.60, 46.77], CONSTANTA = [28.63, 44.17];
let L1cluj = null;
const CASES = {
    A: { name: 'L10 walk, hazard on', hazard: true, canoe: false, make: async () => { const L = await world10(false); return [L, startSeeds(L, ...CLUJ, 8)]; } },
    B: { name: 'L10 walk, hazard off', hazard: false, canoe: false, make: async () => { const L = await world10(false); return [L, startSeeds(L, ...CLUJ, 8)]; } },
    C: { name: 'L10 canoe, hazard on', hazard: true, canoe: true, make: async () => { const L = await world10(true); return [L, startSeeds(L, ...CONSTANTA, 8)]; } },
    D: { name: 'L1 window 1950 x 1020 Cluj, hazard on', hazard: true, canoe: false, make: async () => {
        const L = await windowLevel(win(1, 15.5, 42.5, 31.75, 51), false); L1cluj = L; return [L, startSeeds(L, ...CLUJ, 10)]; } },
    E: { name: 'L0.1 window 1680 x 900 Cluj (ring from L1), hazard on', hazard: true, canoe: false, make: async () => {
        const L = await windowLevel(win(0.1, 22.90, 46.40, 24.30, 47.15), false); return [L, ringSeeds(L, L1cluj, CLUJ)]; } },
    F: { name: 'L0.1 window 1680 x 900 Cluj (ring from L1), hazard off', hazard: false, canoe: false, make: async () => {
        const L = await windowLevel(win(0.1, 22.90, 46.40, 24.30, 47.15), false); return [L, ringSeeds(L, L1cluj, CLUJ)]; } },
    G: { name: 'L1 canoe window 1440 x 840 Constanta, hazard on', hazard: true, canoe: true, make: async () => {
        const L = await windowLevel(win(1, 23, 40, 35, 47), true); return [L, startSeeds(L, ...CONSTANTA, 10)]; } },
};

// CPU time of this process, ms (BY=wall for wall time): the machine is shared, so wall time waits on other jobs
const now = process.env.BY === 'wall' ? () => performance.now() : () => { const u = process.cpuUsage(); return (u.user + u.system) / 1000; };
const q = (a, p) => { if (!a.length) return NaN; const s = Float64Array.from(a).sort(); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
function compare(L, a, b, canoe) {
    const ta = HEAP.cellTimes(L, a, canoe), tb = NEW.cellTimes(L, b, canoe);
    let reached = 0, reachDiff = 0, differ = 0, maxRel = 0, maxAbs = 0, maxAt = -1;
    const rels = [];
    for (let i = 0; i < ta.length; i++) {
        const x = ta[i], y = tb[i];
        if (x === Infinity && y === Infinity) continue;
        reached++;
        if ((x === Infinity) !== (y === Infinity)) { reachDiff++; continue; }
        if (x === y) continue;
        differ++;
        const rel = Math.abs(x - y) / Math.max(x, y);
        rels.push(rel);
        if (rel > maxRel) { maxRel = rel; maxAt = i; }
        maxAbs = Math.max(maxAbs, Math.abs(x - y));
    }
    let wDiff = 0, pDiff = 0, hDiff = 0, nodes = 0;
    for (let i = 0; i < a.w.length; i++) {
        if (a.w[i] === Infinity && b.w[i] === Infinity) continue;
        nodes++;
        if (a.w[i] !== b.w[i]) wDiff++;
        if (a.pred[i] !== b.pred[i]) pDiff++;
        if (a.hz[i] !== b.hz[i]) hDiff++;
    }
    return { reached, reachDiff, differ, maxRel, medRel: q(rels, 0.5), maxAbs, at: maxAt >= 0 ? `${maxAt} (${ta[maxAt]} vs ${tb[maxAt]})` : '-', nodes, wDiff, pDiff, hDiff };
}

for (const key of Object.keys(CASES)) {
    if (ONLY && !ONLY.includes(key) && !(key === 'D' && ONLY.some((k) => k === 'E' || k === 'F'))) continue;
    const C = CASES[key];
    cfg.hazard = C.hazard;
    const [L, seeds] = await C.make();
    const ms = { heap: [], bucket: [] }, xms = XMODS.map(() => []), xres = [];
    let ra, rb;
    for (let rep = 0; rep < REPS; rep++) {
        let t0 = now();
        ra = await HEAP.solve(L, seeds, () => true, C.canoe);
        ms.heap.push(now() - t0);
        t0 = now();
        rb = await NEW.solve(L, seeds, () => true, C.canoe);
        ms.bucket.push(now() - t0);
        for (const [i, [, X]] of XMODS.entries()) {
            t0 = now();
            xres[i] = await X.solve(L, seeds, () => true, C.canoe);
            xms[i].push(now() - t0);
        }
    }
    const qi = { ...NEW.queueInfo };
    if (key === 'D') L.res = ra;
    const cmp = compare(L, ra, rb, C.canoe);
    const med = (a) => q(a, 0.5);
    console.log(`${key} ${C.name}: ${L.W} x ${L.H} = ${(L.W * L.H / 1e6).toFixed(2)} M cells${C.canoe ? `, ${L.cn.E} canoe nodes beyond` : ''}, ${seeds.idx.length} seeds`);
    console.log(`   heap   ms ${ms.heap.map((x) => x.toFixed(0)).join(' ')}  median ${med(ms.heap).toFixed(0)} min ${Math.min(...ms.heap).toFixed(0)}`);
    console.log(`   bucket ms ${ms.bucket.map((x) => x.toFixed(0)).join(' ')}  median ${med(ms.bucket).toFixed(0)} min ${Math.min(...ms.bucket).toFixed(0)}  (x${(med(ms.heap) / med(ms.bucket)).toFixed(2)}, by min x${(Math.min(...ms.heap) / Math.min(...ms.bucket)).toFixed(2)})`);
    console.log(`   queue: exact ${qi.exact}, d ${qi.d.toPrecision(4)} h, ${qi.buckets} buckets used, ${qi.again} queued again, ${qi.entries} entries`);
    console.log(`   cells reached ${cmp.reached}, reach differs ${cmp.reachDiff}, time differs ${cmp.differ} (max rel ${cmp.maxRel.toExponential(2)}, median rel ${cmp.medRel}, max abs ${cmp.maxAbs.toExponential(2)} h at ${cmp.at})`);
    console.log(`   nodes ${cmp.nodes}: w differs ${cmp.wDiff}, hazard differs ${cmp.hDiff}, pred differs ${cmp.pDiff}`);
    for (const [i, [f, X]] of XMODS.entries()) {
        const c2 = compare(L, ra, xres[i], C.canoe), qx = X.queueInfo || {};
        console.log(`   ${f} ms ${xms[i].map((x) => x.toFixed(0)).join(' ')}  median ${med(xms[i]).toFixed(0)} min ${Math.min(...xms[i]).toFixed(0)}  (x${(med(ms.heap) / med(xms[i])).toFixed(2)}, by min x${(Math.min(...ms.heap) / Math.min(...xms[i])).toFixed(2)}); exact ${qx.exact}, again ${qx.again}; ` +
            `time differs ${c2.differ} (max rel ${c2.maxRel.toExponential(2)}), reach differs ${c2.reachDiff}, w differs ${c2.wDiff}, pred differs ${c2.pDiff}`);
    }
}
process.exit(0);
