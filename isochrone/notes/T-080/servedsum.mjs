// What the page shows on zooming in, from served.mjs: zoom 6 / zoom 4 and zoom 8 / zoom 4 at each
// destination, by kind and distance, and whether the start was inside the window.
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';
const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const num = (x) => x === null || x === undefined ? Infinity : x;
const cell = (rs) => rs.length ? `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)}) n${rs.length}` : '-';
const P = [['z6', 'z4'], ['z8', 'z4'], ['z8', 'z6']];
function rows(label, sel) {
    const r = P.map(([a, b]) => {
        const rs = [];
        for (const reg of data) for (const d of reg.dests) if (sel(reg, d)) {
            const ta = num(d[a].t), tb = num(d[b].t);
            if (ta < Infinity && tb < Infinity && tb > 0) rs.push(ta / tb);
        }
        return cell(rs);
    });
    console.log(`| ${label} | ${r.join(' | ')} |`);
}
console.log('| | zoom 6 / zoom 4 | zoom 8 / zoom 4 | zoom 8 / zoom 6 |\n|---|---|---|---|');
for (const k of [...new Set(data.map(r => r.kind))]) rows(k, (r) => r.kind === k);
rows('all', () => true);
for (const D of [20, 80, 300]) rows(D + ' km', (r, d) => d.D === D);
rows('start inside the L1 window', (r, d) => d.in8 && d.in8[1]);
rows('start outside the L1 window', (r, d) => d.in8 && !d.in8[1]);
let lv = {};
for (const r of data) for (const d of r.dests) { const k = 'z8 read L' + d.z8.f; lv[k] = (lv[k] || 0) + 1; }
console.log(lv);
process.exit(0);
