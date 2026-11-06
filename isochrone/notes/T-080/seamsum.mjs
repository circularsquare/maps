import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';
const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const v = { cur: { inside: [], step: [] }, fix: { inside: [], step: [] } };
for (const r of data) for (const w of r.wins) for (const k of ['cur', 'fix']) for (const m of ['inside', 'step']) {
    const x = w[k][m][1]; if (x === x && x !== null) v[k][m].push(x);
}
const s = (a) => `${quant(a, 0.5).toFixed(2)} (${quant(a, 0.1).toFixed(2)}-${quant(a, 0.9).toFixed(2)}) n${a.length}`;
for (const k of ['cur', 'fix']) console.log(k, 'inside', s(v[k].inside), '| step', s(v[k].step));
process.exit(0);
