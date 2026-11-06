// Directional crossing slowness (two values a cell) against the one-value grids, same pairs.
//   node proto2.mjs > results/proto2.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';
import { solveDir } from './dirsearch.mjs';

H.cfg.hazard = false;
const pure = JSON.parse(readFileSync(H.SCR + 'results/pure_walk.json', 'utf8'));
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const M = { 3: Number(process.env.M3 || 3), 10: Number(process.env.M10 || 10) };
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
    for (const k of [3, 10]) {
        const w = H.win(k, box.west, box.south, box.east, box.north);
        const base = H.upscale(g, w1.cols, w1.rows, k);
        const X = crossings(g, w1.cols, w1.rows, k, M[k], w1.north, w1.cell);
        const L = H.level(w, base, null);
        const s = H.snap(L, lng, lat, Math.ceil(10 / k));
        const n = base.length;
        const mk = (fx, fy) => {
            const sx = new Float32Array(n).fill(Infinity), sy = new Float32Array(n).fill(Infinity);
            for (let i = 0; i < n; i++) {
                if (!base[i]) continue;
                sx[i] = fx(i); sy[i] = fy(i);
                if (!(sx[i] < Infinity) && !(sy[i] < Infinity)) sx[i] = sy[i] = 40 / base[i];
            }
            return [sx, sy];
        };
        const V = {
            dmin: mk(i => X.ewMin[i], i => X.nsMin[i]),
            dmid: mk(i => geo(X.ewMin[i], X.ewMean[i]), i => geo(X.nsMin[i], X.nsMean[i])),
        };
        for (const [name, [sx, sy]] of Object.entries(V)) {
            L.dist = solveDir(L, sx, sy, s);
            for (const d of dests) d[`t${k}${name}`] = H.timeAt(L, d.lat, d.lng).t;
        }
        // the isotropic ones for reference
        for (const [name, grid] of Object.entries({
            xmin: gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i])),
            xmid: gridFrom(base, X, (X, i) => geo(geo(X.ewMin[i], X.nsMin[i]), geo(X.ewMean[i], X.nsMean[i]))) })) {
            const L2 = H.level(w, grid, null);
            await H.solveFrom(L2, lat, lng, false);
            for (const d of dests) d[`t${k}${name}`] = H.timeAt(L2, d.lat, d.lng).t;
        }
    }
    out.push({ name: reg.name, kind: reg.kind, lat, lng, dests });
    console.error(`${reg.name}: ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + `results/proto2${process.env.OUT || ''}.json`, JSON.stringify(out));
process.exit(0);
