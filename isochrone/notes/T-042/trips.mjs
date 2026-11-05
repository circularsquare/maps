// T-042 sample: walking check trips over the real L10 grid with a land hazard band (docs/hazard.md).
// A copy of js/search.js's walking loop with weight = time + K x hazard; js/ itself is unchanged.
//   node isochrone/notes/T-042/trips.mjs [world|nat|osm|natjul|osmjul] [K ...]   (from the maps root; -1 = hazard off)
//   CLUJ=1 ...   also the Cluj reach statistics
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url)).replace(/\\/g, '/');
const HERE = fileURLToPath(new URL('./', import.meta.url)).replace(/\\/g, '/');
const imp = (m) => import(pathToFileURL(ROOT + 'js/' + m).href);
const { makeLevel, snap } = await imp('levels.js');
const { MDC, MDR } = await imp('search.js');

const meta = JSON.parse(readFileSync(ROOT + 'data/grid.json', 'utf8'));
const grid = new Uint8Array(gunzipSync(readFileSync(ROOT + 'data/grid.bin.gz')));
const L = makeLevel(grid, meta.width, meta.height, meta.west, meta.north, (meta.north - meta.south) / meta.height, true);
const { W, H, dxKm, dyKm } = L, N = W * H;
const inv = new Float32Array(256).fill(-1);
for (let v = 1; v < 256; v++) inv[v] = meta.scale / v;
const variant = process.argv[2] || 'nat';
// 'world' reads the built band (T-066, data/hazard/l10.bin.gz); the others the T-042 sample
const OUT = ROOT + 'data/raw/t042/', WORLD = variant === 'world';
const codes = new Uint8Array(gunzipSync(readFileSync(WORLD ? ROOT + 'data/hazard/l10.bin.gz' : OUT + 'hazard_' + variant + '.bin.gz')));
const rate = new Float32Array(N);
for (let i = 0; i < N; i++) rate[i] = codes[i] ? Math.pow(10, (codes[i] - 240) / 40) : 0;
// extra per-cell layers for describing a path (float32, not coded)
const f32 = (k) => {
    if (WORLD) return new Float32Array(N);     // no factor layers in the built band: the breakdown reads 0
    const b = readFileSync(OUT + 'l10_' + k + '.f32'); return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
};
const thirstL = f32('thirst_' + variant), iceL = f32('ice'), altL = f32('alt'), coldL = f32('cold');

const E0C = new Int32Array(16), E0R = new Int32Array(16), E1C = new Int32Array(16), E1R = new Int32Array(16);
for (let k = 8, i = 0; i < 4; i++) {
    const sx = i < 2 ? 1 : -1, sy = i % 2 ? -1 : 1;
    E0C[k] = 0; E0R[k] = sy; E1C[k] = sx; E1R[k] = sy; k++;
    E0C[k] = sx; E0R[k] = 0; E1C[k] = sx; E1R[k] = sy; k++;
}

const stepKm = new Float32Array(H * 16);
for (let r = 0; r < H; r++) for (let k = 0; k < 16; k++) {
    const r2 = r + MDR[k];
    if (r2 < 0 || r2 >= H) continue;
    const dx = MDC[k] * (dxKm[r] + dxKm[r2]) * 0.5, dy = MDR[k] * dyKm;
    stepKm[r * 16 + k] = Math.sqrt(dx * dx + dy * dy);
}
const MAXH = 1024, HZSTOP = Math.log(100);

function solve(start, K, hazardOn) {
    const q = new Float32Array(N);     // hazard per km walked: rate x hours per km
    if (hazardOn) for (let i = 0; i < N; i++) if (grid[i]) q[i] = rate[i] * inv[grid[i]];
    const w = new Float64Array(N).fill(Infinity), t = new Float64Array(N).fill(Infinity), hz = new Float64Array(N);
    const pred = new Uint8Array(N), done = new Uint8Array(N);
    const heap = [];   // binary heap of [w, node]
    const push = (d, v) => { heap.push([d, v]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= d) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { let c = 2 * i + 1; if (c >= heap.length) break; if (c + 1 < heap.length && heap[c + 1][0] < heap[c][0]) c++; if (heap[c][0] >= heap[i][0]) break; [heap[c], heap[i]] = [heap[i], heap[c]]; i = c; } } return top; };
    w[start] = 0; t[start] = 0; push(0, start);
    const wStop = MAXH + K * HZSTOP;
    while (heap.length) {
        const [wu, u] = pop();
        if (done[u] || wu > w[u]) continue;
        done[u] = 1;
        if (wu > wStop) break;
        const tu = t[u], hu = hz[u];
        if (tu > MAXH || hu > HZSTOP) continue;
        const r = (u / W) | 0, c = u - r * W, invU = inv[grid[u]], qU = q[u];
        for (let k = 0; k < 16; k++) {
            const r2 = r + MDR[k];
            if (r2 < 0 || r2 >= H) continue;
            const c2 = (c + MDC[k] + W) % W, v = r2 * W + c2;
            if (done[v] || !grid[v]) continue;
            let mean, mq;
            if (k < 8) { mean = (invU + inv[grid[v]]) * 0.5; mq = (qU + q[v]) * 0.5; }
            else {
                const e0 = (r + E0R[k]) * W + (c + E0C[k] + W) % W, e1 = (r + E1R[k]) * W + (c + E1C[k] + W) % W;
                if (!grid[e0] || !grid[e1]) continue;
                mean = (invU + inv[grid[v]] + inv[grid[e0]] + inv[grid[e1]]) * 0.25;
                mq = (qU + q[v] + q[e0] + q[e1]) * 0.25;
            }
            const len = stepKm[r * 16 + k], dt = len * mean, dh = len * mq, nw = wu + dt + K * dh;
            if (nw < w[v]) { w[v] = nw; t[v] = tu + dt; hz[v] = hu + dh; pred[v] = 1 + k; push(nw, v); }
        }
    }
    return { w, t, hz, pred };
}

function trace(res, n) {
    const path = [n];
    while (res.pred[n]) {
        const k = res.pred[n] - 1, r = (n / W) | 0, c = n - r * W;
        n = (r - MDR[k]) * W + (c - MDC[k] + W) % W;
        path.push(n);
    }
    return path.reverse();
}
const latOf = (cell) => L.north - (((cell / W) | 0) + 0.5) * L.cellDeg;
const lonOf = (cell) => L.west + (cell % W + 0.5) * L.cellDeg;

function describe(res, path) {
    // hazard by factor along the path (each step's time x the end cell's factor rates) and the worst cell
    const by = { thirst: 0, ice: 0, alt: 0, cold: 0 };
    let worst = 0, worstAt = '', km = 0;
    for (let i = 1; i < path.length; i++) {
        const b = path[i], dt = res.t[b] - res.t[path[i - 1]];
        by.thirst += dt * thirstL[b]; by.ice += dt * iceL[b]; by.alt += dt * altL[b]; by.cold += dt * coldL[b];
        if (rate[b] > worst) { worst = rate[b]; worstAt = latOf(b).toFixed(1) + ',' + lonOf(b).toFixed(1); }
        const k = res.pred[b] - 1;
        km += stepKm[(((b / W) | 0) - MDR[k]) * 16 + k];
    }
    const step = Math.max(1, Math.floor(path.length / 10)), pts = [];
    for (let i = 0; i < path.length; i += step) pts.push(latOf(path[i]).toFixed(1) + ',' + lonOf(path[i]).toFixed(1));
    pts.push(latOf(path[path.length - 1]).toFixed(1) + ',' + lonOf(path[path.length - 1]).toFixed(1));
    return { by, worst, worstAt, km, pts };
}

const TRIPS = [
    ['Tamanrasset', [22.79, 5.53], [['Agadez', 16.97, 7.99], ['In Salah', 27.20, 2.47]]],
    ['Cairo', [30.04, 31.24], [['Khartoum', 15.50, 32.56], ['Siwa', 29.20, 25.52]]],
    ['Cluj', [46.77, 23.60], [['Vienna', 48.21, 16.37], ['Athens', 37.98, 23.73], ['Kyiv', 50.45, 30.52]]],
    ['Greenland east', [65.60, -38.20], [['Kangerlussuaq', 67.01, -50.70]]],
    ['Yakutsk', [62.03, 129.73], [['Magadan', 59.56, 150.80], ['Verkhoyansk', 67.55, 133.39]]],
    ['Kathmandu', [27.71, 85.32], [['Lhasa', 29.65, 91.17]]],
    ['Leh', [34.16, 77.58], [['Manali', 32.24, 77.19]]],
    ['Alice Springs', [-23.70, 133.88], [['Adelaide', -34.93, 138.60], ['Darwin', -12.46, 130.84]]],
    ['Riyadh', [24.71, 46.68], [['Muscat', 23.59, 58.41]]],
    ['Tashkent', [41.30, 69.24], [['Orenburg', 51.77, 55.10]]],
    ['Altar', [30.71, -111.83], [['Tucson', 32.22, -110.97]]],
];

const Ks = process.argv.slice(3).map(Number);
const only = process.env.ONLY;
for (const [sname, s, dests] of TRIPS) {
    if (only && !sname.startsWith(only)) continue;
    const si = snap(L, s[1], s[0], 3);
    console.log(`\n## ${sname} (${variant})  start cell ${latOf(si).toFixed(2)},${lonOf(si).toFixed(2)}`);
    for (const K of Ks) {
        const hzOn = K >= 0, KK = Math.max(K, 0);
        const t0 = performance.now();
        const res = solve(si, KK, hzOn);
        const ms = performance.now() - t0;
        let reached = 0;
        for (let i = 0; i < N; i++) if (grid[i] && res.t[i] <= MAXH && res.hz[i] <= HZSTOP) reached++;
        console.log(`  ${K < 0 ? 'off   ' : 'K=' + K}: ${Math.round(ms)} ms, land cells reached ${reached}`);
        for (const [dname, lat, lng] of dests) {
            const di = snap(L, lng, lat, 3);
            if (!(res.t[di] <= MAXH) || res.hz[di] > HZSTOP + 5) { console.log(`    ${dname}: not reached`); continue; }
            const path = trace(res, di), ds = describe(res, path);
            const surv = 100 * Math.exp(-res.hz[di]);
            console.log(`    ${dname}: ${(res.t[di] / 8).toFixed(1)} d (${res.t[di].toFixed(0)} h), ${ds.km.toFixed(0)} km, survival ${surv < 0.1 ? surv.toExponential(1) : surv.toFixed(1)}%` +
                `  [thirst ${ds.by.thirst.toFixed(3)}, ice ${ds.by.ice.toFixed(3)}, alt ${ds.by.alt.toFixed(3)}, cold ${ds.by.cold.toFixed(3)}; worst ${ds.worst.toExponential(1)}/h at ${ds.worstAt}]` +
                `\n      via ${ds.pts.join(' ')}`);
        }
    }
}

if (process.env.CLUJ) {
    const si = snap(L, 23.60, 46.77, 3);
    const off = solve(si, 0, false), on = solve(si, 1000, true);
    let a = 0, b = 0, both = 0, sumR = 0, moved = 0;
    for (let i = 0; i < N; i++) {
        if (!grid[i]) continue;
        const ra = off.t[i] <= MAXH, rb = on.t[i] <= MAXH && on.hz[i] <= HZSTOP;
        if (ra) a++; if (rb) b++;
        if (ra && rb && off.t[i] > 8) { both++; const x = on.t[i] / off.t[i]; sumR += x; if (x > 1.05) moved++; }
    }
    console.log(`\nCluj reach: off ${a} cells, K=1000 ${b}; mean time ratio ${(sumR / both).toFixed(3)} over ${both}, ${(100 * moved / both).toFixed(1)}% over 5% slower`);
}
process.exit(0);
