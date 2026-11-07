// T-088's cmp.mjs: L1/L10 and L1/L3 medians (10-90%) per kind and per start, side by side, trips
// from 20 km, hazard on.
//   OUT=<folder>/ node cmp.mjs [walk|canoe] tag1 tag2 ...   (reads <OUT>results/pure_<mode>_<tag>.json)
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const SCR = (process.env.OUT || fileURLToPath(new URL('./', import.meta.url))).replace(/\\/g, '/').replace(/\/?$/, '/');
const quant = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const [mode, ...tags] = process.argv.slice(2);
const sets = tags.map(t => JSON.parse(readFileSync(SCR + `results/pure_${mode}_${t}.json`, 'utf8')));
const num = (x) => x === null || x === undefined ? Infinity : x;
function ratios(data, sel, a, b) {
    const rs = [];
    for (const reg of data) for (const d of reg.dests) {
        if (!sel(reg, d) || d.D < 20) continue;
        const ta = num(d['t' + a + 'on']), tb = num(d['t' + b + 'on']);
        if (ta < Infinity && tb < Infinity && tb > 0) rs.push(ta / tb);
    }
    return rs;
}
const med = (rs) => rs.length ? `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)})` : '-';
console.log(`| | ${tags.map(t => `L1/L10 ${t}`).join(' | ')} | ${tags.map(t => `L1/L3 ${t}`).join(' | ')} |`);
console.log('|---|' + '---|'.repeat(2 * tags.length));
const kinds = [...new Set(sets[0].map(r => r.kind))];
for (const k of [...kinds, 'all']) {
    const sel = (r) => k === 'all' || r.kind === k;
    console.log(`| ${k} | ${sets.map(s => med(ratios(s, sel, 1, 10))).join(' | ')} | ${sets.map(s => med(ratios(s, sel, 1, 3))).join(' | ')} |`);
}
for (const reg of sets[0]) {
    const sel = (r) => r.name === reg.name;
    console.log(`| ${reg.name} | ${sets.map(s => med(ratios(s, sel, 1, 10))).join(' | ')} | ${sets.map(s => med(ratios(s, sel, 1, 3))).join(' | ')} |`);
}
// mean |ln ratio| over every destination
const mabs = (data, a, b) => { const rs = ratios(data, () => true, a, b); return (rs.reduce((s, x) => s + Math.abs(Math.log(x)), 0) / rs.length).toFixed(3); };
console.log(`| mean abs ln | ${sets.map(s => mabs(s, 1, 10)).join(' | ')} | ${sets.map(s => mabs(s, 1, 3)).join(' | ')} |`);
process.exit(0);
