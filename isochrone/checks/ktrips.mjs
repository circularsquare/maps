// The K check trips (notes/T-005.md, notes/T-006.md, walking: notes/T-042.md, T-067.md): the
// worker's modules in node over the real L10 data, with the land danger band, from the maps root:
//   node isochrone/checks/ktrips.mjs trips -1 0 300 1000   -> each trip at each K (-1: hazard off)
//   node isochrone/checks/ktrips.mjs time [K]              -> canoe solve times, hazard off and on
//   ONLY=cyprus ...                                        -> Constanta to Cyprus and Izmir, by edge kind
//   ONLY=lakes ...                                         -> lake crossings (T-043), by edge kind
//   ONLY=rivers ...                                        -> river trips (T-031, T-083), by edge kind
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url)).replace(/\\/g, '/');
globalThis.fetch = async (u) => {
    const p = String(u).replace(/^file:\/\//, '');
    return new Response(readFileSync(p));
};
const imp = (m) => import(pathToFileURL(ROOT + 'js/' + m).href);
const { cfg, setConfig } = await imp('config.js');
const { makeLevel, snap, setHazard, setDirs } = await imp('levels.js');
const { solve, cellTimes, MDC, MDR, P_GRID, P_XFER, P_DOWN, P_UP } = await imp('search.js');
const { loadWorld } = await imp('canoedata.js');
const { buildCanoe, eachCanoeNode, canoeFrom } = await imp('canoe.js');
const { loadHazardWorld, applyRoads } = await imp('tiles.js');

const meta = JSON.parse(readFileSync(ROOT + 'data/grid.json', 'utf8'));
// CANOE=<folder>/ and HAZARD=<folder>/ read another canoe or land hazard data set (an A/B before a swap)
setConfig({ bands: [], alpha: 190, maxHours: 1024, meta, canoeBase: process.env.CANOE || ROOT + 'data/canoe/', hazardBase: process.env.HAZARD || ROOT + 'data/hazard/' });
const grid = gunzipSync(readFileSync(ROOT + 'data/grid.bin.gz'));
const cell = (meta.north - meta.south) / meta.height;
const L = makeLevel(new Uint8Array(grid.buffer, grid.byteOffset, grid.length), meta.width, meta.height, meta.west, meta.north, cell, true);
if (cfg.roads && meta.roads) applyRoads(L.grid, gunzipSync(readFileSync(ROOT + 'data/' + meta.roads)));
// directed walking, as the page reads it (index.html loadGrid, js/session.js)
const gz = (f) => { const b = gunzipSync(readFileSync(ROOT + 'data/' + f)); return new Uint8Array(b.buffer, b.byteOffset, b.length); };
if (!process.env.SYMMETRIC) setDirs(L, meta.dirs ? gz(meta.dirs) : null, cfg.roads && meta.dirsRoads ? gz(meta.dirsRoads) : null);
setHazard(L, await loadHazardWorld());
const d = await loadWorld(L.W, L.H);
let t0 = performance.now();
buildCanoe(L, d, d.sea);
console.error('build', Math.round(performance.now() - t0), 'ms, E', L.cn.E);
const { W, H } = L, N = W * H;
const alive = () => true;
const lonOf = (c) => L.west + (c + 0.5) * L.cellDeg, latOf = (r) => L.north - (r + 0.5) * L.cellDeg;
const SH = new Float32Array(256);
for (let s = 0; s < 256; s++) SH[s] = s <= 20 ? s : 20 + 10 * (Math.min(s, 78) - 20);

async function run(lat, lng, canoe = true) {
    const s = snap(L, lng, lat, 8);
    const t = performance.now();
    const res = await solve(L, { idx: [s], w: [0], t: [0], hz: [0] }, alive, canoe);
    return { res, ms: performance.now() - t };
}

function nodeCell(n) { return n < N ? n : L.cn.cellOf[n - N]; }
function trace(res, n) {
    const path = [];
    for (let guard = 0; guard < 1e6; guard++) {
        path.push(n);
        const p = res.pred[n];
        if (p === 0 || p === 255) break;
        const c = nodeCell(n), r = (c / W) | 0, col = c - r * W;
        let dc, dr, other = false;
        if (p >= P_UP) { const k = p - P_UP; dc = MDC[k]; dr = MDR[k]; }
        else if (p >= P_DOWN) { const k = p - P_DOWN; dc = MDC[k]; dr = MDR[k]; }
        else if (p >= P_XFER) { const j = p - P_XFER; dc = j % 3 - 1; dr = ((j / 3) | 0) - 1; other = true; }
        else { const k = p - P_GRID; dc = MDC[k]; dr = MDR[k]; }
        const pc = (r - dr) * W + ((col - dc + W) % W);
        const canoeHere = n >= N || !L.grid[n];
        const canoePrev = other ? !canoeHere : canoeHere;
        n = canoePrev ? canoeFrom(L, res, n, p, pc) : pc;
    }
    return path.reverse();
}

function dest(res, lat, lng) {
    // the least-time node in the 3 x 3 cells round the point, land or canoe
    const r0 = Math.floor((L.north - lat) / L.cellDeg), c0 = Math.floor((lng - L.west) / L.cellDeg);
    let best = -1, bt = Infinity;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const c = (r0 + dr) * W + ((c0 + dc + W) % W);
        const ns = [L.grid[c] ? c : -1];
        eachCanoeNode(L.cn, c, (m) => { ns.push(m); });
        for (const n of ns) if (n >= 0 && res.t[n] < bt) { bt = res.t[n]; best = n; }
    }
    return best;
}

function describe(res, path) {
    let maxOff = 0, far = 0, pad = 0, carry = 0, pts = [];
    for (let i = 1; i < path.length; i++) {
        const a = path[i - 1], b = path[i], dt = res.t[b] - res.t[a];
        const cb = nodeCell(b);
        const canoe = b >= N || !L.grid[b];
        if (canoe) {
            pad += dt;
            const km = SH[L.cn.shore[cb]];
            if (km > maxOff) maxOff = km;
            if (km > 20) far += dt;
        } else carry += dt;
    }
    const by = { sea: 0, lake: 0, down: 0, up: 0, carry: 0, xfer: 0 }, segs = [];
    let last = null;
    for (let i = 1; i < path.length; i++) {
        const b = path[i], p = res.pred[b], dt = res.t[b] - res.t[path[i - 1]];
        const cb = nodeCell(b), k = L.cn.info[cb] & 0xFF;
        const kind = p >= P_UP ? 'up' : p >= P_DOWN ? 'down' : p >= P_XFER ? 'xfer' : (b >= N || !L.grid[b]) ? (k >= 3 ? 'lake' : 'sea') : 'carry';
        by[kind] += dt;
        if (kind !== last && kind !== 'xfer') { const r = (cb / W) | 0; segs.push(kind + '@' + latOf(r).toFixed(1) + ',' + lonOf(cb - r * W).toFixed(1)); last = kind; }
    }
    if (process.env.ONLY) console.log('    by kind (h):', Object.entries(by).map(([k, v]) => k + ' ' + v.toFixed(0)).join(', '), '\n    segments:', segs.join(' '));
    const step = Math.max(1, Math.floor(path.length / 12));
    for (let i = 0; i < path.length; i += step) {
        const c = nodeCell(path[i]), r = (c / W) | 0;
        pts.push(latOf(r).toFixed(1) + ',' + lonOf(c - r * W).toFixed(1));
    }
    return { pad, carry, far, maxOff, pts };
}

// ONLY=lakes: lake crossings for the lake temperature (notes/T-043.md)
const LAKE_TRIPS = [
    ['Yellowknife-Fort Resolution (Great Slave)', [62.45, -114.37], [61.17, -113.67]],
    ['Listvyanka-Ust-Barguzin (Baikal)', [51.85, 104.87], [53.42, 109.02]],
    ['Puno-Copacabana (Titicaca)', [-15.84, -70.02], [-16.17, -69.09]],
    ['Polson-Somers (Flathead)', [47.69, -114.16], [48.08, -114.23]],
    ['Ibanez-Chile Chico (General Carrera)', [-46.30, -71.94], [-46.54, -71.73]],
    ['West Thumb-Lake Village (Yellowstone)', [44.42, -110.57], [44.55, -110.40]],
    ['Karakul east-west (Karakul)', [39.01, 73.55], [39.05, 73.33]],
    ['Divnogorsk-Krasnoturansk (Krasnoyarsk Reservoir)', [55.96, 92.38], [54.32, 91.56]],
];
// ONLY=rivers: river trips for the gradient and desert rivers (notes/T-031.md)
const RIVER_TRIPS = [
    ['Belem-Manaus', [-1.45, -48.50], [-3.13, -60.02]],
    ['Manaus-Belem', [-3.13, -60.02], [-1.45, -48.50]],
    ['Kisangani-Kinshasa', [0.52, 25.19], [-4.32, 15.31]],
    ['Kinshasa-Banana', [-4.32, 15.31], [-6.01, 12.40]],
    ['Matadi-Kinshasa', [-5.82, 13.45], [-4.32, 15.31]],
    ['St Louis-New Orleans', [38.63, -90.19], [29.95, -90.07]],
    ['New Orleans-St Louis', [29.95, -90.07], [38.63, -90.19]],
    ['Vienna-Sulina', [48.21, 16.37], [45.16, 29.73]],
    ['Lees Ferry-Pearce Ferry', [36.87, -111.60], [36.11, -113.99]],
    ['Skardu-Tarbela', [35.30, 75.63], [34.09, 72.70]],
    ['Elverum-Fredrikstad', [60.88, 11.56], [59.21, 10.94]],
    ['Longreach-Windorah', [-23.44, 144.25], [-25.42, 142.65]],
    ['Windorah-Lake Eyre', [-25.42, 142.65], [-28.37, 137.37]],
    ['Bourke-Wentworth', [-30.09, 145.94], [-34.11, 141.92]],
    ['Mopti-Gao', [14.49, -4.20], [16.27, -0.04]],
    ['Khartoum-Aswan', [15.60, 32.53], [24.09, 32.90]],
    // small rivers both ways (T-073): Lewis and Clark's Jefferson and Beaverhead, Schoolcraft's Mississippi headwaters
    ['Three Forks-Camp Fortunate', [45.925, -111.515], [45.006, -112.852]],
    ['Camp Fortunate-Three Forks', [45.006, -112.852], [45.925, -111.515]],
    ['Cass Lake-Itasca', [47.449, -94.482], [47.240, -95.208]],
    ['Itasca-Cass Lake', [47.240, -95.208], [47.449, -94.482]],
];
const TRIPS = process.env.ONLY === 'lakes' ? LAKE_TRIPS : process.env.ONLY === 'rivers' ? RIVER_TRIPS : process.env.ONLY === 'cyprus' ? [['Constanta-Kyrenia', [44.17, 28.63], [35.34, 33.32]], ['Constanta-Izmir', [44.17, 28.63], [38.42, 27.14]]] : [
    ['Brest-A Coruna', [48.39, -4.49], [43.37, -8.40]],
    ['Dover-Calais', [51.12, 1.31], [50.95, 1.85]],
    ['Tarifa-Tangier', [36.01, -5.60], [35.78, -5.81]],
    ['Su\'ao-Yonaguni', [24.60, 121.86], [24.47, 123.00]],
    ['Marseille-Barcelona', [43.30, 5.37], [41.38, 2.18]],
    ['Bergen-Reykjavik', [60.39, 5.32], [64.15, -21.94]],
    ['Bergen-St John\'s', [60.39, 5.32], [47.56, -52.71]],
    ['Brest-St John\'s', [48.39, -4.49], [47.56, -52.71]],
];
// walking starts, where the land danger decides the route (notes/T-042.md, T-067.md)
const WALKS = process.env.ONLY ? [] : [
    ['Tamanrasset-Agadez', [22.79, 5.53], [16.97, 7.99]],
    ['Tamanrasset-In Salah', [22.79, 5.53], [27.20, 2.47]],
    ['Cairo-Khartoum', [30.04, 31.24], [15.50, 32.56]],
    ['Greenland east-Kangerlussuaq', [65.60, -38.20], [67.01, -50.70]],
    ['Leh-Manali', [34.16, 77.58], [32.24, 77.19]],
];

// REV=1 walks each trip the other way (a check that directed walking differs uphill and downhill)
if (process.env.REV) for (let i = 0; i < WALKS.length; i++) WALKS[i] = [WALKS[i][0] + ' (reverse)', WALKS[i][2], WALKS[i][1]];
const mode = process.argv[2] || 'trips';
if (mode === 'time') {
    const STARTS = [['Cluj', 46.77, 23.6], ['Lisbon', 38.72, -9.14], ['Jakarta', -6.1, 106.8], ['Brest', 48.39, -4.49], ['Constanta', 44.17, 28.63]];
    cfg.K = Number(process.argv[3] || 1000);
    for (const [name, lat, lng] of STARTS) {
        const ms = { false: [], true: [] }, nodes = {};
        for (let i = 0; i < 4; i++) for (const hz of [false, true]) {
            cfg.hazard = hz;
            const { res, ms: m } = await run(lat, lng);
            ms[hz].push(Math.round(m));
            let reached = 0;
            for (let j = 0; j < res.t.length; j++) if (res.t[j] <= 1024) reached++;
            nodes[hz] = (reached / 1e6).toFixed(2);
        }
        console.log(`${name.padEnd(10)} off ${ms.false.join('/')} ms (${nodes.false} M)   on ${ms.true.join('/')} ms (${nodes.true} M)`);
    }
} else {
    const Ks = process.argv.slice(3).map(Number);
    const settings = Ks.map(k => k < 0 ? ['hazard off', false, 1000] : ['K=' + k, true, k]);
    const starts = new Map();
    for (const [, a] of TRIPS) starts.set(a.join(), a);
    for (const [label, hz, K] of settings) {
        cfg.hazard = hz; cfg.K = K;
        console.log(`\n== ${label}`);
        for (const a of starts.values()) {
            const { res, ms } = await run(a[0], a[1]);
            // westmost reached sea at the latitude band 40-55N, for the Atlantic reach
            let west = 0;
            for (let r = 0; r < H; r++) {
                const lat = latOf(r);
                if (lat < 40 || lat > 55) continue;
                for (let c = 0; c < W; c++) { const lon = lonOf(c); if (lon > -80 && lon < west && res.t[r * W + c] <= 1024 && !L.grid[r * W + c]) west = lon; }
            }
            console.log(`start ${a.join(',')}: ${Math.round(ms)} ms, westmost sea reached 40-55N ${west.toFixed(1)}`);
            for (const [name, s, b] of TRIPS) {
                if (s.join() !== a.join()) continue;
                const n = dest(res, b[0], b[1]);
                if (n < 0 || !(res.t[n] <= 1024)) { console.log(`  ${name}: not reached`); continue; }
                const path = trace(res, n);
                const ds = describe(res, path);
                console.log(`  ${name}: ${(res.t[n] / 8).toFixed(1)} days (${res.t[n].toFixed(0)} h), survival ${(100 * Math.exp(-res.hz[n])).toFixed(1)}% (risk ${(100 * -Math.expm1(-res.hz[n])).toFixed(3)}%), ` +
                    `paddled ${ds.pad.toFixed(0)} h (${ds.far.toFixed(0)} h past 20 km), carried ${ds.carry.toFixed(0)} h, farthest ${ds.maxOff} km out\n    via ${ds.pts.join(' ')}`);
            }
        }
        let from = null, res = null;
        for (const [name, s, b] of WALKS) {
            if (s.join() !== from) { from = s.join(); res = (await run(s[0], s[1], false)).res; console.log(`walking from ${from}`); }
            // the land cell nearest the point, as notes/T-042/trips.mjs
            const n = snap(L, b[1], b[0], 3);
            if (n < 0 || !(res.t[n] <= 1024)) { console.log(`  ${name}: not reached`); continue; }
            console.log(`  ${name}: ${(res.t[n] / 8).toFixed(1)} days (${res.t[n].toFixed(0)} h), survival ${(100 * Math.exp(-res.hz[n])).toFixed(1)}%`);
        }
    }
}
process.exit(0);   // the search's MessageChannel would keep node alive
