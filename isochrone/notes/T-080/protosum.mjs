// L1 / corrected coarse ratios from proto.json: median (p10-p90), and the share within 10%.
//   node protosum.mjs results/proto.json 10 [minD=20] [variants]
import { readFileSync } from 'node:fs';
import { quant } from './lib.mjs';

const data = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const k = process.argv[3] || '10';
const minD = Number(process.argv[4] || 20);
const keys = Object.keys(data[0].dests[0]).filter(x => x.startsWith('t' + k) && x.length > 1 + k.length).map(x => x.slice(1 + k.length));
const VARS = process.argv[5] ? process.argv[5].split(',') : keys;
const num = (x) => x === null || x === undefined ? Infinity : x;
const cell = (rs) => {
    if (!rs.length) return '-';
    const w = rs.filter(x => x > 0.9 && x < 1.1).length / rs.length;
    return `${quant(rs, 0.5).toFixed(2)} (${quant(rs, 0.1).toFixed(2)}-${quant(rs, 0.9).toFixed(2)}) ${Math.round(100 * w)}%`;
};
console.log(`\nL1 / L${k} variant, from ${minD} km: median (p10-p90), share within 10%\n`);
console.log('| kind | ' + VARS.join(' | ') + ' |\n|' + '---|'.repeat(VARS.length + 1));
const kinds = [...new Set(data.map(r => r.kind)), 'all'];
for (const kk of [...kinds, ...(process.env.STARTS ? data.map(r => r.name) : [])]) {
    const row = VARS.map(v => {
        const rs = [];
        for (const r of data) if (kk === 'all' || r.kind === kk || r.name === kk) for (const d of r.dests) {
            if (d.D < minD) continue;
            const a = num(d.t1), b = num(d[`t${k}${v}`]);
            if (a < Infinity && b < Infinity && b > 0) rs.push(a / b);
        }
        return cell(rs);
    });
    console.log(`| ${kk} | ${row.join(' | ')} |`);
}
process.exit(0);
