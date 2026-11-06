// Side by side L1/L10 and L1/L3 medians per kind and per start, several measure.mjs results.
//   node cmp.mjs tag1 tag2 ...   (results/pure_canoe_<tag>.json)
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';
const tags = process.argv.slice(2);
const sets = tags.map(t => JSON.parse(readFileSync(new URL(`./results/pure_canoe_${t}.json`, import.meta.url), 'utf8')));
const num = (x) => x === null || x === undefined ? Infinity : x;
function med(data, sel, a, b) {
    const rs = [];
    for (const reg of data) for (const d of reg.dests) {
        if (!sel(reg, d) || d.D < 20) continue;
        const ta = num(d['t' + a + 'on']), tb = num(d['t' + b + 'on']);
        if (ta < Infinity && tb < Infinity && tb > 0) rs.push(ta / tb);
    }
    return rs.length ? `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)})` : '-';
}
console.log(`| | ${tags.map(t => `L1/L10 ${t}`).join(' | ')} | ${tags.map(t => `L1/L3 ${t}`).join(' | ')} |`);
console.log('|---|' + '---|'.repeat(2 * tags.length));
const kinds = [...new Set(sets[0].map(r => r.kind))];
for (const k of [...kinds, 'all']) {
    const sel = (r) => k === 'all' || r.kind === k;
    console.log(`| ${k} | ${sets.map(s => med(s, sel, 1, 10)).join(' | ')} | ${sets.map(s => med(s, sel, 1, 3)).join(' | ')} |`);
}
for (const reg of sets[0]) {
    const sel = (r) => r.name === reg.name;
    console.log(`| ${reg.name} | ${sets.map(s => med(s, sel, 1, 10)).join(' | ')} | ${sets.map(s => med(s, sel, 1, 3)).join(' | ')} |`);
}
// mean |ln ratio| over every destination, and over the starts' medians
const mabs = (data, a, b) => {
    let s = 0, n = 0; const meds = [];
    for (const reg of data) {
        const rs = [];
        for (const d of reg.dests) { if (d.D < 20) continue; const ta = num(d['t' + a + 'on']), tb = num(d['t' + b + 'on']); if (ta < Infinity && tb < Infinity && tb > 0) { rs.push(ta / tb); s += Math.abs(Math.log(ta / tb)); n++; } }
        if (rs.length) meds.push(Math.abs(Math.log(quant(rs, 0.5))));
    }
    return `${(s / n).toFixed(3)} / ${(meds.reduce((x, y) => x + y, 0) / meds.length).toFixed(3)}`;
};
console.log(`| mean abs ln (dests / start medians) | ${sets.map(s => mabs(s, 1, 10)).join(' | ')} | ${sets.map(s => mabs(s, 1, 3)).join(' | ')} |`);
process.exit(0);
