// Time fields for a picture: L1, L10 now and L10 from crossing times, one start (hazard off).
//   node fields.mjs <lat> <lng> <tag>
import { writeFileSync, mkdirSync } from 'node:fs';
import * as H from './lib.mjs';
import { crossings, gridFrom } from './homog.mjs';

H.cfg.hazard = false;
const lat = Number(process.argv[2]), lng = Number(process.argv[3]), tag = process.argv[4];
const geo = (a, b) => a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b);
const box = H.boxAround(lat, lng, 1.5, 1.5 / Math.cos(lat * Math.PI / 180));
const w1 = H.win(1, box.west, box.south, box.east, box.north), w10 = H.win(10, box.west, box.south, box.east, box.north);
// a bigger box to solve in, so the picture's edge is not the search's edge
const big = H.boxAround(lat, lng, 3, 3 / Math.cos(lat * Math.PI / 180));
const b1 = H.win(1, big.west, big.south, big.east, big.north), b10 = H.win(10, big.west, big.south, big.east, big.north);
const g = (await H.loadBox(b1)).grid;
const base = H.upscale(g, b1.cols, b1.rows, 10);
const X = crossings(g, b1.cols, b1.rows, 10, 10, b1.north, b1.cell);
const fix = gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i]));
mkdirSync(H.SCR + 'fields', { recursive: true });
function crop(dist, wb, w) {
    const out = new Float32Array(w.cols * w.rows);
    const dc = w.col0 - wb.col0, dr = w.row0 - wb.row0;
    for (let r = 0; r < w.rows; r++) for (let c = 0; c < w.cols; c++) out[r * w.cols + c] = dist[(r + dr) * wb.cols + c + dc];
    return out;
}
for (const [name, wb, w, grid] of [['l1', b1, w1, g], ['l10', b10, w10, base], ['l10x', b10, w10, fix]]) {
    const L = H.level(wb, grid, null);
    const d = await H.solveFrom(L, lat, lng, false);
    writeFileSync(H.SCR + `fields/${tag}_${name}.bin`, Buffer.from(crop(d, wb, w).buffer));
}
writeFileSync(H.SCR + `fields/${tag}.json`, JSON.stringify({ lat, lng, box, l1: [w1.cols, w1.rows], l10: [w10.cols, w10.rows] }));
process.exit(0);
