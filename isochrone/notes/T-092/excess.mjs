// The canoe's L1/L10 past walking's: per destination (canoe L1/L10) / (walk L1/L10), same starts and
// destinations; median (10-90%) per kind and for the river and lake starts, and mean abs ln.
//   OUT=<folder>/ node excess.mjs <walk tag> <canoe tag> ...
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const SCR = (process.env.OUT || fileURLToPath(new URL('./', import.meta.url))).replace(/\\/g, '/').replace(/\/?$/, '/');
const quant = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const [wt, ...ct] = process.argv.slice(2);
const load = (m, t) => JSON.parse(readFileSync(SCR + `results/pure_${m}_${t}.json`, 'utf8'));
const walk = load('walk', wt), sets = ct.map(t => load('canoe', t));
const ok = (x) => x > 0 && x < Infinity;
function rs(cs, sel) {
    const out = [];
    cs.forEach((reg, i) => {
        if (!sel(reg)) return;
        const wr = walk[i];
        reg.dests.forEach((d, j) => {
            const w = wr.dests[j];
            if (d.D < 20 || !w || w.lat !== d.lat) return;
            if (ok(d.t1on) && ok(d.t10on) && ok(w.t1on) && ok(w.t10on)) out.push((d.t1on / d.t10on) / (w.t1on / w.t10on));
        });
    });
    return out;
}
const med = (a) => a.length ? `${quant(a, 0.5).toFixed(2)} (${quant(a, 0.1).toFixed(2)}-${quant(a, 0.9).toFixed(2)})` : '-';
console.log(`| canoe / walk L1/L10 | ${ct.join(' | ')} |`);
console.log('|---|' + '---|'.repeat(ct.length));
const kinds = [...new Set(walk.map(r => r.kind))];
for (const k of [...kinds, 'all']) console.log(`| ${k} | ${sets.map(s => med(rs(s, r => k === 'all' || r.kind === k))).join(' | ')} |`);
for (const n of ['Coari', 'Khulna', 'Kisangani', 'Lagos', 'Jyvaskyla', 'New York', 'Utrecht', 'Bemidji', 'Sudbury', 'Sogndal'])
    console.log(`| ${n} | ${sets.map(s => med(rs(s, r => r.name === n))).join(' | ')} |`);
console.log(`| mean abs ln | ${sets.map(s => { const a = rs(s, () => true); return (a.reduce((x, y) => x + Math.abs(Math.log(y)), 0) / a.length).toFixed(3); }).join(' | ')} |`);
process.exit(0);
