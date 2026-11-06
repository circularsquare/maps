// Level agreement (T-080, T-011): the worker's modules in node over a build's walking levels, from
// the maps root:
//   node isochrone/checks/levels.mjs [--tiles=<name>] [--only=Brig,Cusco] [--half=3.3] [--seams] [--updown] [--noroads]
// --tiles reads data/tiles_<name>/ (the page's ?tiles= switch), else the served data/. Walking,
// hazard off (T-080: the ratios are the same with it on).
//   pure:   for each start, L10 (the world), L3 and L1 (a box +-half degrees round the start), each
//           solved from the start alone; the time at destinations 20-300 km out on 12 bearings, on L1
//           land. Ratios L1/L10, L1/L3, L3/L10: median and 10-90%, per start and per kind of land.
//   seams:  L1 windows of ~170 x 110 km, 120-200 km out and not holding the start, seeded on their
//           ring from L10 as the page does, against the whole L1 box (inside) and against L10 a
//           quarter of the way in (step); median of each. Near 1 is no seam.
//   updown: walks between a low and a high place, both ways, at L1 and L10 (directed: uphill slower).
// Prints a line a start and a summary; no pass or fail (the numbers go in a task's notes).
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');
const ARGS = process.argv.slice(2);
const opt = (k, d) => { const a = ARGS.find((x) => x.startsWith('--' + k + '=')); return a ? a.slice(k.length + 3) : d; };
const SUB = opt('tiles', '').replace(/[^a-z0-9_-]/gi, '');
const BASE = ROOT + 'data/' + (SUB ? 'tiles_' + SUB + '/' : '');
const ONLY = opt('only', '') ? opt('only', '').split(',') : null;
const HALF = +opt('half', '3.3');

globalThis.fetch = async (u) => {
    let p = String(u);
    if (p.startsWith('file://')) p = p.slice(7);
    else if (!/^[A-Za-z]:\//.test(p)) p = ROOT + p;
    try { return new Response(readFileSync(p)); } catch { return new Response(null, { status: 404 }); }
};
const imp = (m) => import(pathToFileURL(ROOT + 'js/' + m).href);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, sampleNode, setDirs } = await imp('levels.js');
const { solve, cellTimes, P_SEED } = await imp('search.js');
const T = await imp('tiles.js');

const meta = JSON.parse(readFileSync(BASE + 'grid.json', 'utf8'));
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: ROOT + 'data/canoe/', hazardBase: ROOT + 'data/hazard/' });
cfg.hazard = false;
if (ARGS.includes('--noroads')) cfg.roads = false;
T.setTiles(JSON.parse(readFileSync(BASE + 'tiles/index.json', 'utf8')), BASE + 'tiles/');
const gz = (f) => { const b = gunzipSync(readFileSync(BASE + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
const L10 = makeLevel(gz('grid.bin.gz'), meta.width, meta.height, meta.west, meta.north, (meta.north - meta.south) / meta.height, true);
if (cfg.roads && meta.roads) T.applyRoads(L10.grid, gz(meta.roads));
setDirs(L10, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);

const STARTS = [
    ['Poltava', 'farmland', 49.59, 34.55], ['Ames', 'farmland', 42.03, -93.62], ['Pampa', 'farmland', -34.59, -60.95],
    ['Ludhiana', 'farmland', 30.90, 75.85], ['Brig', 'mountains', 46.32, 7.99], ['Cusco', 'mountains', -13.53, -71.97],
    ['Pokhara', 'mountains', 28.21, 83.99], ['Dillon', 'mountains', 39.63, -106.04], ['Jyvaskyla', 'lakes', 62.24, 25.75],
    ['Sudbury', 'lakes', 46.49, -80.99], ['Bemidji', 'lakes', 47.47, -94.88], ['Tamanrasset', 'desert', 22.79, 5.53],
    ['Alice Springs', 'desert', -23.70, 133.88], ['Dalanzadgad', 'desert', 43.57, 104.42], ['Kisangani', 'forest', 0.52, 25.19],
    ['Coari', 'forest', -4.08, -63.14], ['Ust-Ilimsk', 'forest', 58.00, 102.66], ['Cluj', 'city', 46.77, 23.60],
    ['New York', 'city', 40.758, -73.985], ['London', 'city', 51.507, -0.128], ['Lagos', 'city', 6.52, 3.38],
    ['Delhi', 'city', 28.61, 77.21], ['Khulna', 'coast/delta', 22.82, 89.55], ['Utrecht', 'coast/delta', 52.09, 5.12],
    ['Sogndal', 'coast/delta', 61.23, 7.10],
];
// [name, low lat, lng, high lat, lng]: valley town and a place well above it, a day's walk or less
const UPDOWN = [
    ['Brig - Simplon Pass', 46.316, 7.988, 46.250, 8.033], ['Visp - Zermatt', 46.293, 7.882, 46.020, 7.749],
    ['Kathmandu - Nagarkot', 27.700, 85.320, 27.715, 85.520], ['Pokhara - Ghandruk', 28.209, 83.986, 28.375, 83.807],
    ['Pisac - Cusco', -13.422, -71.851, -13.532, -71.968], ['Martigny - Grand St Bernard', 46.101, 7.072, 45.869, 7.170],
];
const DISTS = [20, 40, 80, 160, 300], BEARINGS = 12, KM = 111.195;

const win = (f, west, south, east, north) => {
    const cell = f / 120;
    return { f, cell, west, east, north, south, col0: Math.round((west + 180) / cell), cols: Math.round((east - west) / cell),
             row0: Math.round((85 - north) / cell), rows: Math.round((north - south) / cell) };
};
const box = (lat, lng, hLat, hLon) => ({ west: Math.floor((lng - hLon) * 4) / 4, east: Math.ceil((lng + hLon) * 4) / 4,
    south: Math.max(-60, Math.floor((lat - hLat) * 4) / 4), north: Math.min(85, Math.ceil((lat + hLat) * 4) / 4) });
async function level(w) {
    const dirs = {};
    const g = await T.loadWindow(w, dirs);
    const L = makeLevel(g, w.cols, w.rows, w.west, w.north, w.cell, false);
    L.win = w;
    setDirs(L, dirs.d, dirs.e);
    return L;
}
async function solveFrom(L, lat, lng) {
    const s = snap(L, lng, lat, L.win ? Math.ceil(10 / L.win.f) : 8);
    if (s < 0) return false;
    L.res = await solve(L, { idx: [s], w: [0], t: [0], hz: [0] }, () => true, false);
    L.dist = cellTimes(L, L.res, false);
    return true;
}
function timeAt(L, lat, lng) {
    let c = Math.floor((lng - L.west) / L.cellDeg), r = Math.floor((L.north - lat) / L.cellDeg);
    if (L.wrap) c = ((c % L.W) + L.W) % L.W;
    if (r < 0 || r >= L.H || c < 0 || c >= L.W) return NaN;
    let best = L.dist[r * L.W + c];
    if (best < Infinity) return best;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < L.H && cc >= 0 && cc < L.W) best = Math.min(best, L.dist[rr * L.W + cc]);
    }
    return best;
}
const offset = (lat, lng, km, b) => [lat + km * Math.cos(b * Math.PI / 180) / KM, lng + km * Math.sin(b * Math.PI / 180) / (KM * Math.cos(lat * Math.PI / 180))];
const q = (a, p) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const fmt = (a) => a.length ? `${q(a, 0.5).toFixed(2)} (${q(a, 0.1).toFixed(2)}-${q(a, 0.9).toFixed(2)})` : '-';
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

console.log(`levels: ${BASE.slice(ROOT.length)}, L10 directions ${L10.sd ? 'on' : 'off'}`);
const byKind = {}, all = { r110: [], r13: [], r310: [], inside: [], step: [] };
const t0 = performance.now();
for (const [name, kind, lat, lng] of STARTS) {
    if (ONLY && !ONLY.includes(name)) continue;
    const cos = Math.cos(lat * Math.PI / 180), b = box(lat, lng, HALF, HALF / cos);
    const w1 = win(1, b.west, b.south, b.east, b.north), w3 = win(3, b.west, b.south, b.east, b.north);
    const [L1, L3] = await Promise.all([level(w1), level(w3)]);
    seed = 12345 + Math.round(lat * 100);
    const dests = [];
    for (const D of DISTS) for (let j = 0; j < BEARINGS; j++) {
        const [la, lo] = offset(lat, lng, D * (0.85 + 0.3 * rnd()), 360 * (j + rnd()) / BEARINGS);
        if (la > b.north - 0.3 || la < b.south + 0.3 || lo < b.west + 0.3 || lo > b.east - 0.3) continue;
        const c = Math.floor((lo - w1.west) / w1.cell), r = Math.floor((w1.north - la) / w1.cell);
        if (L1.grid[r * w1.cols + c]) dests.push([la, lo]);
    }
    for (const L of [L10, L3, L1]) await solveFrom(L, lat, lng);
    const k = byKind[kind] || (byKind[kind] = { r110: [], r13: [], r310: [], inside: [], step: [] });
    const one = { r110: [], r13: [], r310: [] };
    for (const [la, lo] of dests) {
        const t1 = timeAt(L1, la, lo), t3 = timeAt(L3, la, lo), t10 = timeAt(L10, la, lo);
        if (!(t1 < cfg.maxHours && t3 < cfg.maxHours && t10 < cfg.maxHours) || t1 <= 0) continue;
        one.r110.push(t1 / t10); one.r13.push(t1 / t3); one.r310.push(t3 / t10);
    }
    for (const x in one) { k[x].push(...one[x]); all[x].push(...one[x]); }
    let seamTxt = '';
    if (ARGS.includes('--seams')) {
        const ins = [], stp = [], o = [0, 0, 0];
        for (const [dkm, bear] of [[120, 90], [120, 270], [200, 0], [200, 180]]) {
            const [cl, cg] = offset(lat, lng, dkm, bear);
            const ww = win(1, Math.floor((cg - 0.75 / cos) * 4) / 4, Math.floor((cl - 0.5) * 4) / 4, Math.ceil((cg + 0.75 / cos) * 4) / 4, Math.ceil((cl + 0.5) * 4) / 4);
            if (ww.west < b.west || ww.east > b.east || ww.south < b.south || ww.north > b.north) continue;
            const L = await level(ww);
            const seeds = { idx: [], w: [], t: [], hz: [], pred: [] };
            const add = (r, c) => {
                const i = r * ww.cols + c;
                if (L.grid[i] && sampleNode(L10, ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell, false, o)) {
                    seeds.idx.push(i); seeds.w.push(o[0]); seeds.t.push(o[1]); seeds.hz.push(o[2]); seeds.pred.push(P_SEED);
                }
            };
            for (let c = 0; c < ww.cols; c++) { add(0, c); add(ww.rows - 1, c); }
            for (let r = 1; r < ww.rows - 1; r++) { add(r, 0); add(r, ww.cols - 1); }
            const rr = await solve(L, seeds, () => true, false);
            const dr = ww.row0 - w1.row0, dc = ww.col0 - w1.col0, qr = Math.round(ww.rows / 4), qc = Math.round(ww.cols / 4);
            const inside = [], step = [];
            for (let r = 0; r < ww.rows; r++) for (let c = 0; c < ww.cols; c++) {
                const t = rr.t[r * ww.cols + c];
                if (!(t < Infinity)) continue;
                const tw = L1.dist[(r + dr) * w1.cols + c + dc];
                if (tw < Infinity && tw > 0) inside.push(t / tw);
                const ring = ((r === qr || r === ww.rows - 1 - qr) && c >= qc && c < ww.cols - qc) || ((c === qc || c === ww.cols - 1 - qc) && r >= qr && r < ww.rows - qr);
                if (ring && sampleNode(L10, ww.west + (c + 0.5) * ww.cell, ww.north - (r + 0.5) * ww.cell, false, o)) step.push(t / o[1]);
            }
            if (inside.length) ins.push(q(inside, 0.5));
            if (step.length) stp.push(q(step, 0.5));
        }
        k.inside.push(...ins); k.step.push(...stp); all.inside.push(...ins); all.step.push(...stp);
        seamTxt = `  seams inside ${fmt(ins)}, step ${fmt(stp)}`;
    }
    console.log(`${name.padEnd(14)} ${kind.padEnd(12)} n ${String(one.r110.length).padStart(3)}  L1/L10 ${fmt(one.r110)}  L1/L3 ${fmt(one.r13)}  L3/L10 ${fmt(one.r310)}${seamTxt}`);
}
if (!ONLY || ONLY.length > 1) {
    console.log('-- by kind');
    for (const kind in byKind) { const k = byKind[kind]; console.log(`${kind.padEnd(12)} n ${String(k.r110.length).padStart(4)}  L1/L10 ${fmt(k.r110)}  L1/L3 ${fmt(k.r13)}  L3/L10 ${fmt(k.r310)}${k.inside.length ? `  seams inside ${fmt(k.inside)}, step ${fmt(k.step)}` : ''}`); }
    const w10 = (a) => a.length ? (100 * a.filter((x) => x >= 0.9 && x <= 1.1).length / a.length).toFixed(0) + '%' : '-';
    console.log(`${'all'.padEnd(12)} n ${String(all.r110.length).padStart(4)}  L1/L10 ${fmt(all.r110)} ${w10(all.r110)} within 10%  L1/L3 ${fmt(all.r13)} ${w10(all.r13)}  L3/L10 ${fmt(all.r310)}${all.inside.length ? `  seams inside ${fmt(all.inside)}, step ${fmt(all.step)}` : ''}`);
}
if (ARGS.includes('--updown')) {
    console.log('-- up and down (hours; up / down)');
    for (const [name, la, lo, ha, ho] of UPDOWN) {
        const cos = Math.cos((la + ha) / 2 * Math.PI / 180);
        const b = box((la + ha) / 2, (lo + ho) / 2, 0.6, 0.6 / cos);
        const L1 = await level(win(1, b.west, b.south, b.east, b.north));
        const row = [];
        for (const L of [L1, L10]) {
            await solveFrom(L, la, lo); const up = timeAt(L, ha, ho);
            await solveFrom(L, ha, ho); const down = timeAt(L, la, lo);
            row.push(`${L === L1 ? 'L1' : 'L10'} up ${up.toFixed(2)} down ${down.toFixed(2)} (${(up / down).toFixed(2)})`);
        }
        console.log(`${name.padEnd(28)} L1 dirs ${L1.sd ? 'on ' : 'off'}  ${row.join('   ')}`);
    }
}
console.log(`done in ${((performance.now() - t0) / 1000).toFixed(0)} s`);
process.exit(0);
