// Summarise a .cpuprofile: self ticks by function, then by line for files matching a pattern.
// node prof.mjs <file.cpuprofile> [pattern]
import { readFileSync } from 'node:fs';
const [file, pat = 'search'] = process.argv.slice(2);
const P = JSON.parse(readFileSync(file, 'utf8'));
const byFn = new Map(), byLine = new Map();
let total = 0;
for (const n of P.nodes) {
    const cf = n.callFrame, key = `${cf.functionName || '(anon)'} ${cf.url.split('/').pop()}:${cf.lineNumber + 1}`;
    byFn.set(key, (byFn.get(key) || 0) + n.hitCount);
    total += n.hitCount;
    if (cf.url.includes(pat) && n.positionTicks) for (const t of n.positionTicks) {
        const k = `${cf.url.split('/').pop()}:${t.line}`;
        byLine.set(k, (byLine.get(k) || 0) + t.ticks);
    }
}
console.log('total ticks', total);
for (const [k, v] of [...byFn].sort((a, b) => b[1] - a[1]).slice(0, 15)) console.log(String(v).padStart(7), (100 * v / total).toFixed(1).padStart(5) + '%', k);
console.log('-- lines');
for (const [k, v] of [...byLine].sort((a, b) => b[1] - a[1]).slice(0, 40)) console.log(String(v).padStart(7), (100 * v / total).toFixed(1).padStart(5) + '%', k);
// ranges: file:a-b,...
const fileTot = new Map();
for (const [k, v] of byLine) { const f = k.split(':')[0]; fileTot.set(f, (fileTot.get(f) || 0) + v); }
for (const spec of process.argv.slice(4)) {
    const [f, rr] = spec.split(':');
    let s = 0;
    for (const part of rr.split(',')) { const [a, b] = part.split('-').map(Number); for (let l = a; l <= (b || a); l++) s += byLine.get(`${f}:${l}`) || 0; }
    console.log(`${spec}: ${s} of ${fileTot.get(f)} (${(100 * s / fileTot.get(f)).toFixed(1)}% of the file's line ticks)`);
}
