// The canoe with T-081's crossing-time speeds at L3 and L10 inside each box (notes/T-080.md):
// L1/L10 and L1/L3 canoe ratios with the coarse speeds as built and as T-081 would make them.
//   CANOE=<folder>/ ONLY=... node with081.mjs tag  -> results/w081_<tag>.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from '../T-080/homog.mjs';

H.cfg.hazard = true;
const only = process.env.ONLY ? process.env.ONLY.split(',') : null;
const pure = JSON.parse(readFileSync(H.SCR + 'results/pure_canoe.json', 'utf8'));
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
const out = [];
for (const reg of pure) {
    if (only && !only.includes(reg.name)) continue;
    const { lat, lng } = reg;
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north), w3 = H.win(3, box.west, box.south, box.east, box.north);
    const w10 = H.win(10, box.west, box.south, box.east, box.north);
    const [b1, b3] = await Promise.all([H.loadBox(w1), H.loadBox(w3)]);
    const dests = reg.dests.map(d => ({ D: d.D, lat: d.lat, lng: d.lng, t1: d.t1on }));
    const L10 = await H.world10(true);
    const saved = new Map();
    for (const v of ['now', 'x']) {
        const L3 = H.level(w3, v === 'x' ? gridFrom(b3.grid, crossings(b1.grid, w1.cols, w1.rows, 3, 1, w1.north, w1.cell), (X, i) => geo(X.ewMin[i], X.nsMin[i])) : b3.grid, b3.hz);
        if (v === 'x') {
            const base10 = H.upscale(b1.grid, w1.cols, w1.rows, 10);
            const X = crossings(b1.grid, w1.cols, w1.rows, 10, 5, w1.north, w1.cell);
            const fix = gridFrom(base10, X, (X, i) => geo(X.ewMin[i], X.nsMin[i]));
            for (let r = 0; r < w10.rows; r++) for (let c = 0; c < w10.cols; c++) {
                const i = (w10.row0 + r) * L10.W + w10.col0 + c, g = L10.grid[i];
                if (!g || !fix[r * w10.cols + c]) continue;
                saved.set(i, g);
                L10.grid[i] = fix[r * w10.cols + c];
            }
            H.setHazard(L10, L10.hzCodes);
        }
        for (const [lv, L] of [[10, L10], [3, L3]]) {
            await H.solveFrom(L, lat, lng, true);
            for (const d of dests) d[`t${lv}${v}`] = H.timeAt(L, d.lat, d.lng).t;
        }
        L3.cn = null;
    }
    for (const [i, g] of saved) L10.grid[i] = g;
    H.setHazard(L10, L10.hzCodes);
    for (const d of dests) { d.t1now = d.t1; d.t1x = d.t1; }
    out.push({ name: reg.name, kind: reg.kind, dests });
    console.error(reg.name, 'done');
}
writeFileSync(H.SCR + `results/w081_${process.argv[2] || 'all'}.json`, JSON.stringify(out));
process.exit(0);
