// Time of the crossing build per coarse land cell, and gzip size of a corrected box against now.
import { gzipSync } from 'node:zlib';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
for (const [name, lat, lng] of [['Poltava', 49.59, 34.55], ['Cusco', -13.53, -71.97], ['Jyvaskyla', 62.24, 25.75]]) {
    const box = H.boxAround(lat, lng, 3.3, 3.3 / Math.cos(lat * Math.PI / 180));
    const w1 = H.win(1, box.west, box.south, box.east, box.north);
    const g = (await H.loadBox(w1)).grid;
    for (const [k, m] of [[10, 10], [10, 5], [3, 3], [3, 1]]) {
        const base = H.upscale(g, w1.cols, w1.rows, k);
        const t0 = performance.now();
        const X = crossings(g, w1.cols, w1.rows, k, m, w1.north, w1.cell);
        const ms = performance.now() - t0;
        const land = base.reduce((s, v) => s + (v ? 1 : 0), 0);
        const fix = gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i]));
        let up = 0; for (let i = 0; i < base.length; i++) if (base[i]) up += fix[i] / base[i];
        console.log(`${name} k${k} m${m}: ${land} land cells, ${(1000 * ms / land).toFixed(0)} us a cell; gz now ${gzipSync(base).length} fixed ${gzipSync(fix).length}; mean speed x${(up / land).toFixed(2)}`);
    }
}
process.exit(0);
