// Canoe (or walk) L1/L10 median by trip distance, for named starts and for all starts.
//   OUT=<folder>/ node bydist.mjs <canoe|walk> <tag> [start names comma separated]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const SCR = (process.env.OUT || fileURLToPath(new URL('./', import.meta.url))).replace(/\\/g, '/').replace(/\/?$/, '/');
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)].toFixed(2) : '-'; };
const [mode, tag, names] = process.argv.slice(2);
const data = JSON.parse(readFileSync(SCR + `results/pure_${mode}_${tag}.json`, 'utf8'));
const sels = [['all', () => true], ...(names ? names.split(',') : []).map(n => [n, (r) => r.name === n])];
const Ds = [5, 10, 20, 40, 80, 160, 300];
console.log(`| ${mode} ${tag} L1/L10 by distance (km) | ${Ds.join(' | ')} |`);
console.log('|---|' + '---|'.repeat(Ds.length));
for (const [n, sel] of sels) {
    const row = Ds.map(D => {
        const rs = [];
        for (const r of data) if (sel(r)) for (const d of r.dests) if (d.D === D && d.t1on < Infinity && d.t10on < Infinity && d.t10on > 0) rs.push(d.t1on / d.t10on);
        return med(rs);
    });
    console.log(`| ${n} | ${row.join(' | ')} |`);
}
process.exit(0);
