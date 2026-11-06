// Corrected coarse grids for the pure_walk pairs: L3' and L10' made from the L1 box by different
// rules, each solved from the start only (hazard off); the L1 time is the target.
//   node proto.mjs > results/proto.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';

H.cfg.hazard = false;
const pure = JSON.parse(readFileSync(H.SCR + 'results/pure_walk.json', 'utf8'));
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const M = { 3: Number(process.env.M3 || 3), 10: Number(process.env.M10 || 10) };

// power mean of hours per km over land cells, the build's land rule
function powerUp(g, W, H_, k, p) {
    const w = W / k, h = H_ / k, out = new Uint8Array(w * h);
    for (let R = 0; R < h; R++) for (let C = 0; C < w; C++) {
        let s = 0, n = 0;
        for (let r = R * k; r < R * k + k; r++) for (let c = C * k; c < C * k + k; c++) {
            const v = g[r * W + c]; if (!v) continue;
            const x = 40 / v; s += p === 0 ? Math.log(x) : Math.pow(x, p); n++;
        }
        if (n < 0.25 * k * k) continue;
        const mean = p === 0 ? Math.exp(s / n) : Math.pow(s / n, 1 / p);
        out[R * w + C] = Math.max(1, Math.min(255, Math.round(40 / mean)));
    }
    return out;
}
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
const out = [];
for (const reg of pure) {
    if (only && !only.includes(reg.name)) continue;
    const { lat, lng } = reg;
    const t0 = performance.now();
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north);
    const g = (await H.loadBox(w1)).grid;
    const dests = reg.dests.map(d => ({ D: d.D, lat: d.lat, lng: d.lng, t1: d.t1off }));
    const tc = {};
    for (const k of [3, 10]) {
        const w = H.win(k, box.west, box.south, box.east, box.north);
        const base = H.upscale(g, w1.cols, w1.rows, k);
        const X = crossings(g, w1.cols, w1.rows, k, M[k], w1.north, w1.cell);
        const G = {
            cur: base,
            p0: powerUp(g, w1.cols, w1.rows, k, 0),
            pm1: powerUp(g, w1.cols, w1.rows, k, -1),
            xmin: gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i])),
            xmean: gridFrom(base, X, (X, i) => geo(X.ewMean[i], X.nsMean[i])),
            xmid: gridFrom(base, X, (X, i) => geo(geo(X.ewMin[i], X.nsMin[i]), geo(X.ewMean[i], X.nsMean[i]))),
        };
        for (const [name, grid] of Object.entries(G)) {
            const L = H.level(w, grid, null);
            await H.solveFrom(L, lat, lng, false);
            for (const d of dests) d[`t${k}${name}`] = H.timeAt(L, d.lat, d.lng).t;
        }
        tc[k] = ((performance.now() - t0) / 1000).toFixed(0);
    }
    out.push({ name: reg.name, kind: reg.kind, lat, lng, dests });
    console.error(`${reg.name}: ${tc[3]} s, ${tc[10]} s`);
}
writeFileSync(H.SCR + `results/proto${process.env.OUT || ''}.json`, JSON.stringify(out));
process.exit(0);
