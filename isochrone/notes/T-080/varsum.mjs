// Summary of variants.json: median (p10-p90) of a level pair's ratio per variant and kind of land.
//   node varsum.mjs results/variants.json 1/10 [minD=20] [variants]
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';

const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const [a, b] = (process.argv[3] || '1/10').split('/');
const minD = Number(process.argv[4] || 20);
const VARS = (process.argv[5] || 'real,uni,flat,fill,fcap,ffloor').split(',');
const kinds = [...new Set(data.map(r => r.kind)), 'all'];
const cell = (rs) => rs.length ? `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)})` : '-';
console.log(`\nL${a}/L${b}, from ${minD} km: median (p10-p90)\n`);
console.log('| kind | ' + VARS.join(' | ') + ' |\n|' + '---|'.repeat(VARS.length + 1));
const num = (x) => x === null || x === undefined ? Infinity : x;
for (const k of [...kinds, ...data.map(r => r.name)]) {
    const row = VARS.map(v => {
        const rs = [];
        for (const r of data) if ((k === 'all' || r.kind === k || r.name === k)) for (const d of r.dests) {
            if (d.D < minD) continue;
            const ta = num(d[`t${a}${v}`]), tb = num(d[`t${b}${v}`]);
            if (ta < Infinity && tb < Infinity && tb > 0) rs.push(ta / tb);
        }
        return cell(rs);
    });
    console.log(`| ${k} | ${row.join(' | ')} |`);
}
process.exit(0);
