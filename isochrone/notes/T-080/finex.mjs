// L0.1 against L1 and L10 made from it by the mean rule and by crossing times (Cluj, New York).
//   node finex.mjs > results/finex.json
import { readFileSync, writeFileSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';

H.cfg.hazard = false;
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
const fine = JSON.parse(readFileSync(H.SCR + 'results/fine.json', 'utf8'));
const BOX = { Cluj: [23, 46, 25, 48], Beclean: [23, 46, 25, 48], Midtown: [-75, 40, -73, 42], 'Middletown NY': [-75, 40, -73, 42] };
const out = [];
for (const reg of fine) {
    const [w, s, e, n] = BOX[reg.name];
    const w01 = H.win(0.1, w, s, e, n);
    const g = (await H.loadBox(w01)).grid;
    const dests = reg.dests.map(d => ({ D: d.D, lat: d.lat, lng: d.lng, t1: d.t01off, t1m: d.t1moff }));
    for (const [k, f, m] of [[10, 1, 5], [100, 10, 20]]) {
        const wf = H.win(f, w, s, e, n);
        const base = H.upscale(g, w01.cols, w01.rows, k);
        const X = crossings(g, w01.cols, w01.rows, k, m, w01.north, w01.cell);
        for (const [name, grid] of [['cur', base], ['xmin', gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i]))]]) {
            const L = H.level(wf, grid, null);
            await H.solveFrom(L, reg.lat, reg.lng, false);
            for (const d of dests) d[`t${f}${name}`] = H.timeAt(L, d.lat, d.lng).t;
        }
    }
    out.push({ name: reg.name, kind: reg.kind, dests });
    console.error(reg.name, 'done');
}
writeFileSync(H.SCR + 'results/finex.json', JSON.stringify(out));
process.exit(0);
