// What a global or per-kind speed factor would leave: L1 / (L10 or L3 now x factor), the factor
// being the median ratio (fitted on the same pairs, so this flatters it).
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';
const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const num = (x) => x === null || x === undefined ? Infinity : x;
for (const k of ['10', '3']) {
    const all = [], byKind = {};
    for (const r of data) for (const d of r.dests) {
        if (d.D < 20) continue;
        const a = num(d.t1), b = num(d['t' + k + 'cur']);
        if (!(a < Infinity && b < Infinity && b > 0)) continue;
        all.push(a / b); (byKind[r.kind] ||= []).push(a / b);
    }
    const g = quant(all, 0.5);
    const show = (rs) => `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)}) ${Math.round(100 * rs.filter(x => x > 0.9 && x < 1.1).length / rs.length)}%`;
    console.log(`L1/L${k}: global factor ${g.toFixed(2)} leaves ${show(all.map(x => x / g))}`);
    const per = [];
    for (const [kind, rs] of Object.entries(byKind)) { const f = quant(rs, 0.5); per.push(...rs.map(x => x / f)); }
    console.log(`L1/L${k}: per-kind factors leave ${show(per)}`);
}
process.exit(0);
