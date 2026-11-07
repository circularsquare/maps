// L0.1 against the L1 round it (T-039): the worker's modules in node, from the maps root:
//   node isochrone/checks/fine.mjs [--fine=<name>] [--only=Cluj,Brig] [--noroads]
// --fine reads data/fine_<name>/ (the page's ?fine= switch), else data/fine/; L1 from the served
// data/tiles/. Walking, hazard off. For each start, an L1 box +-0.6 degrees solved from the start
// alone, then L0.1 windows of 0.15 x 0.15 degrees (east-west / cos lat), as the page solves them:
//   centred: round the start, seeded on its ring from L1 and from the start;
//   off:     25 km east, west, north and south, not holding the start, seeded on the ring only.
// Per window, over its cells with 100 m data reached at both levels (the start's at least 0.25 h
// out): the median of t(L0.1) / t(L1) (inside), and the same on the ring a quarter of the way in
// (step). Near 1 is no seam. Also the solve time per window and the share of cells with 100 m data.
// Prints a line a start and a summary; no pass or fail (the numbers go in a task's notes).
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');
const ARGS = process.argv.slice(2);
const opt = (k, d) => { const a = ARGS.find((x) => x.startsWith('--' + k + '=')); return a ? a.slice(k.length + 3) : d; };
const SUB = opt('fine', '').replace(/[^a-z0-9_-]/gi, '');
const ONLY = opt('only', '') ? opt('only', '').split(',') : null;

globalThis.fetch = async (u) => {
    let p = String(u);
    if (!/^[A-Za-z]:\//.test(p)) p = ROOT + p;
    try { return new Response(readFileSync(p)); } catch { return new Response(null, { status: 404 }); }
};
const imp = (m) => import(pathToFileURL(ROOT + 'js/' + m).href);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, sampleNode, setDirs, setSlope } = await imp('levels.js');
const { solve, cellTimes, P_SEED } = await imp('search.js');
const T = await imp('tiles.js');

const meta = JSON.parse(readFileSync(ROOT + 'data/grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, fineBase: 'data/' + (SUB ? 'fine_' + SUB : 'fine') + '/' });
cfg.hazard = false;
if (ARGS.includes('--noroads')) cfg.roads = false;
T.setTiles(JSON.parse(readFileSync(ROOT + 'data/tiles/index.json', 'utf8')), ROOT + 'data/tiles/');

const STARTS = [
    ['Cluj', 46.77, 23.60], ['New York', 40.758, -73.985], ['Innsbruck', 47.268, 11.393],
    ['Brig', 46.316, 7.988], ['Genoa', 44.41, 8.93], ['Pittsburgh', 40.44, -79.99],
];
const KM = 111.195, HALF1 = 0.6, SIDE = 0.15, OFF_KM = 25;

const win = (f, west, south, east, north) => {
    const cell = f / 120;
    return { f, cell, west, east, north, south, col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
};
// a box of whole cells round (lat, lng), half-sides in degrees, edges on multiples of a
const box = (lat, lng, hLat, hLon, a) => ({ west: Math.floor((lng - hLon) / a) * a, east: Math.ceil((lng + hLon) / a) * a,
    south: Math.floor((lat - hLat) / a) * a, north: Math.ceil((lat + hLat) / a) * a });
const q = (a, p) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const fmt = (a) => a.length ? `${q(a, 0.5).toFixed(3)} (${q(a, 0.1).toFixed(2)}-${q(a, 0.9).toFixed(2)})` : '-';

async function fineWindow(lat, lng, L1, withStart, origin) {
    const cos = Math.cos(lat * Math.PI / 180);
    const b = box(lat, lng, SIDE / 2, SIDE / 2 / cos, 0.05);
    const w = win(0.1, b.west, b.south, b.east, b.north);
    const dirs = {};
    const g = await T.loadWindow(w, dirs);
    const L = makeLevel(g, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    if (dirs.z) setSlope(L, dirs.z, dirs.g, dirs.slope);
    const seeds = { idx: [], w: [], t: [], hz: [], pred: [] }, o = [0, 0, 0];
    const add = (r, c) => {
        const i = r * w.cols + c;
        if (g[i] && sampleNode(L1, w.west + (c + 0.5) * w.cell, w.north - (r + 0.5) * w.cell, false, o)) {
            seeds.idx.push(i); seeds.w.push(o[0]); seeds.t.push(o[1]); seeds.hz.push(o[2]); seeds.pred.push(P_SEED);
        }
    };
    for (let c = 0; c < w.cols; c++) { add(0, c); add(w.rows - 1, c); }
    for (let r = 1; r < w.rows - 1; r++) { add(r, 0); add(r, w.cols - 1); }
    if (withStart) {
        const s = snap(L, origin.lng, origin.lat, 100);
        if (s >= 0) { seeds.idx.push(s); seeds.w.push(0); seeds.t.push(0); seeds.hz.push(0); seeds.pred.push(0); }
    }
    const t0 = performance.now();
    const res = await solve(L, seeds, () => true, false);
    const ms = performance.now() - t0;
    // cells inside a block with 100 m data (by the index's block masks, so stage 1 tiles count the same)
    const ix = await T.fineIndex();
    const fineAt = (i) => {
        const R = w.row0 + Math.floor(i / w.cols), C = w.col0 + i % w.cols;
        const m = ix.mask.get(T.fineName(Math.floor(R / 1200), Math.floor(C / 1200) % 360)) || 0;
        return (m >> (4 * Math.floor((R % 1200) / 300) + Math.floor((C % 1200) / 300))) & 1;
    };
    const qr = Math.round(w.rows / 4), qc = Math.round(w.cols / 4);
    const inside = [], step = [];
    let nf = 0;
    for (let r = 0; r < w.rows; r++) for (let c = 0; c < w.cols; c++) {
        const i = r * w.cols + c;
        if (!g[i]) continue;
        if (!fineAt(i)) continue;
        nf++;
        const t = res.t[i];
        if (!(t < Infinity) || !sampleNode(L1, w.west + (c + 0.5) * w.cell, w.north - (r + 0.5) * w.cell, false, o)) continue;
        if (o[1] < 0.25) continue;
        inside.push(t / o[1]);
        const ring = ((r === qr || r === w.rows - 1 - qr) && c >= qc && c < w.cols - qc) || ((c === qc || c === w.cols - 1 - qc) && r >= qr && r < w.rows - qr);
        if (ring) step.push(t / o[1]);
    }
    let land = 0;
    for (let i = 0; i < g.length; i++) if (g[i]) land++;
    return { inside: q(inside, 0.5), step: q(step, 0.5), ms, cells: w.cols * w.rows, share: nf / Math.max(1, land), slope: !!dirs.z };
}

const offset = (lat, lng, km, bear) => [lat + km * Math.cos(bear * Math.PI / 180) / KM, lng + km * Math.sin(bear * Math.PI / 180) / (KM * Math.cos(lat * Math.PI / 180))];
console.log(`fine: ${cfg.fineBase}, roads ${cfg.roads ? 'on' : 'off'}`);
const all = { centred: [], off: [], step: [], ms: [] };
for (const [name, lat, lng] of STARTS) {
    if (ONLY && !ONLY.includes(name)) continue;
    const cos = Math.cos(lat * Math.PI / 180);
    const b = box(lat, lng, HALF1, HALF1 / cos, 0.25);
    const w1 = win(1, b.west, b.south, b.east, b.north);
    const dirs = {};
    const L1 = makeLevel(await T.loadWindow(w1, dirs), w1.cols, w1.rows, w1.west, w1.north, w1.cell, false);
    L1.win = w1;
    setDirs(L1, dirs.d, dirs.e);
    const s = snap(L1, lng, lat, 100);
    L1.res = await solve(L1, { idx: [s], w: [0], t: [0], hz: [0] }, () => true, false);
    L1.dist = cellTimes(L1, L1.res, false);
    const origin = { lat, lng };
    const c = await fineWindow(lat, lng, L1, true, origin);
    const offs = [];
    for (const bear of [90, 270, 0, 180]) {
        const [la, lo] = offset(lat, lng, OFF_KM, bear);
        offs.push(await fineWindow(la, lo, L1, false, origin));
    }
    const off = offs.map((x) => x.inside).filter((x) => x === x), stp = [c, ...offs].map((x) => x.step).filter((x) => x === x);
    const ms = [c, ...offs].map((x) => x.ms);
    all.centred.push(c.inside); all.off.push(...off); all.step.push(...stp); all.ms.push(...ms.map((m, i) => [m, [c, ...offs][i].cells]));
    console.log(`${name.padEnd(11)} slope ${c.slope ? 'on ' : 'off'} 100 m share ${[c, ...offs].map((x) => (100 * x.share).toFixed(0) + '%').join(' ')}` +
        `  centred ${c.inside.toFixed(3)}  off ${off.map((x) => x.toFixed(3)).join(' ')}  step ${fmt(stp)}` +
        `  solve ${ms.map((m) => Math.round(m)).join(' ')} ms (${(c.cells / 1e6).toFixed(2)} M cells)`);
}
if (!ONLY || ONLY.length > 1) {
    console.log(`all: centred ${fmt(all.centred)}  off ${fmt(all.off)}  step ${fmt(all.step)}  solve ${fmt(all.ms.map(([m]) => m / 1000))} s`);
}
process.exit(0);
