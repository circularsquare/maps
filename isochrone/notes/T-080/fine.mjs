// L0.1 against L1 (MAP) and against its own upscales (L1', L3', L10' made from L0.1 by the
// build rule), in the Cluj and New York tiles. Walking, hazard off.
//   node fine.mjs > results/fine.json
import { writeFileSync } from 'node:fs';
import * as H from './lib.mjs';

H.cfg.hazard = false;
const REG = [
    ['Cluj', 'city', 46.77, 23.60, [23, 46, 25, 48]],
    ['Beclean', 'farmland/hills', 47.18, 24.18, [23, 46, 25, 48]],
    ['Midtown', 'city', 40.758, -73.985, [-75, 40, -73, 42]],
    ['Middletown NY', 'farmland/hills', 41.45, -74.42, [-75, 40, -73, 42]],
];
const DISTS = [5, 10, 20, 40, 80];
const out = [];
for (const [name, kind, lat, lng, [w, s, e, n]] of REG) {
    const t0 = performance.now();
    const w01 = H.win(0.1, w, s, e, n), w1 = H.win(1, w, s, e, n);
    const [b01, b1] = await Promise.all([H.loadBox(w01), H.loadBox(w1)]);
    const lv = {
        '01': H.level(w01, b01.grid, null),
        '1m': H.level(w1, b1.grid, null),
        '1u': H.level(w1, H.upscale(b01.grid, w01.cols, w01.rows, 10), null),
        '3u': H.level(H.win(3, w, s, e, n), H.upscale(b01.grid, w01.cols, w01.rows, 30), null),
        '10u': H.level(H.win(10, w, s, e, n), H.upscale(b01.grid, w01.cols, w01.rows, 100), null),
    };
    // MAP chain in the same box too
    lv['3m'] = H.level(H.win(3, w, s, e, n), H.upscale(b1.grid, w1.cols, w1.rows, 3), null);
    lv['10m'] = H.level(H.win(10, w, s, e, n), H.upscale(b1.grid, w1.cols, w1.rows, 10), null);
    const r = H.rng(777 + Math.round(lat * 100));
    const dests = [];
    for (const D of DISTS) for (let j = 0; j < 16; j++) {
        const [la, lo] = H.offset(lat, lng, D * (0.85 + 0.3 * r()), 360 * (j + r()) / 16);
        if (la > n - 0.05 || la < s + 0.05 || lo < w + 0.05 || lo > e - 0.05) continue;
        const c = Math.floor((lo - w01.west) / w01.cell), rr = Math.floor((w01.north - la) / w01.cell);
        if (!b01.grid[rr * w01.cols + c]) continue;
        dests.push({ D, lat: la, lng: lo });
    }
    for (const [k, L] of Object.entries(lv)) {
        await H.solveFrom(L, lat, lng, false);
        for (const d of dests) d['t' + k + 'off'] = H.timeAt(L, d.lat, d.lng).t;
        L.dist = null; L.res = null;
    }
    out.push({ name, kind, lat, lng, dests });
    console.error(`${name}: ${dests.length} dests, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
writeFileSync(H.SCR + 'results/fine.json', JSON.stringify(out));
process.exit(0);
