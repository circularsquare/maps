// Tables of fine/coarse time ratios from a measure.mjs result.
//   node analyze.mjs results/pure_walk.json [tag=on] [pairs=1/10,3/10,1/3] [minD=20]
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';

const file = process.argv[2];
const tag = process.argv[3] || 'on';
const pairs = (process.argv[4] || '1/10,3/10,1/3').split(',').map(s => s.split('/'));
const minD = Number(process.argv[5] || 20);
const data = JSON.parse(readFileSync(file, 'utf8'));
const fmt = (x) => isNaN(x) ? '-' : x.toFixed(2);
function stats(rs) {
    if (!rs.length) return '- | - | - | 0';
    const gm = Math.exp(rs.reduce((s, x) => s + Math.log(x), 0) / rs.length);
    return `${fmt(quant(rs, 0.5))} | ${fmt(quant(rs, 0.1))}-${fmt(quant(rs, 0.9))} | ${fmt(gm)} | ${rs.length}`;
}
function ratios(sel, [a, b]) {
    const rs = []; let onlyA = 0, onlyB = 0;
    for (const reg of data) for (const d of reg.dests) {
        if (!sel(reg, d)) continue;
        const num = (x) => x === null || x === undefined ? Infinity : x;   // JSON writes Infinity as null
        const ta = num(d['t' + a + tag]), tb = num(d['t' + b + tag]);
        const ra = ta < Infinity, rb = tb < Infinity;
        if (ra && rb && tb > 0) rs.push(ta / tb);
        else if (ra && !rb) onlyA++;
        else if (rb && !ra) onlyB++;
    }
    return { rs, onlyA, onlyB };
}
const head = (first) => `| ${first} | ` + pairs.map(([a, b]) => `L${a}/L${b} median | p10-p90 | geo mean | n`).join(' | ') + ' |\n|' + '---|'.repeat(1 + 4 * pairs.length);
console.log(`\n${file}, hazard ${tag}, destinations from ${minD} km\n\nBy kind of land\n`);
console.log(head('kind'));
const kinds = [...new Set(data.map(r => r.kind))];
for (const k of [...kinds, 'all']) {
    const row = pairs.map(p => stats(ratios((r, d) => (k === 'all' || r.kind === k) && d.D >= minD, p).rs));
    console.log(`| ${k} | ${row.join(' | ')} |`);
}
console.log('\nBy distance (all kinds)\n');
console.log(head('km'));
for (const D of [...new Set(data.flatMap(r => r.dests.map(d => d.D)))].sort((x, y) => x - y)) {
    const row = pairs.map(p => stats(ratios((r, d) => d.D === D, p).rs));
    console.log(`| ${D} | ${row.join(' | ')} |`);
}
console.log('\nBy start (from ' + minD + ' km)\n');
console.log(head('start'));
for (const reg of data) {
    const row = pairs.map(p => stats(ratios((r, d) => r === reg && d.D >= minD, p).rs));
    console.log(`| ${reg.name} (${reg.kind}) | ${row.join(' | ')} |`);
}
console.log('\nReached at one level only (all distances)\n');
for (const p of pairs) {
    const x = ratios(() => true, p);
    console.log(`L${p[0]} only ${x.onlyA}, L${p[1]} only ${x.onlyB}`);
}
process.exit(0);
