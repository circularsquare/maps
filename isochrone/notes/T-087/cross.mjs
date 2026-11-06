// T-087: 1 km speeds by the best crossing time of the 100 m grid (T-080's rule, notes/T-080/homog.mjs).
//   node isochrone/notes/T-087/cross.mjs <tiles dir> [margin=5] [truth=f_cap] [tile,tile,...] [suffix]
// Writes <tile>/l1_xmin.bin (crossing the 100 m grid with roads) and l1_xminplain.bin (the grid without
// roads, f_plain), uint8 km/h x 40, on the land cells of l1_none.bin; geometric mean of the east-west
// and north-south best crossings. Prints seconds and microseconds per land cell.
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { crossings, gridFrom } from '../T-080/homog.mjs';

const dir = process.argv[2], M = +(process.argv[3] || 5), TRUTH = process.argv[4] || 'f_cap';
const tiles = process.argv[5] ? process.argv[5].split(',') : readdirSync(dir).sort();
const geo = (a, b) => (a < Infinity && b < Infinity ? Math.sqrt(a * b) : Math.min(a, b));
let totSec = 0, totCells = 0;
for (const name of tiles) {
  const d = join(dir, name);
  if (!existsSync(join(d, 'meta.json'))) continue;
  const meta = JSON.parse(readFileSync(join(d, 'meta.json'), 'utf8'));
  const base = new Uint8Array(readFileSync(join(d, 'l1_none.bin')));
  let land = 0; for (let i = 0; i < base.length; i++) if (base[i]) land++;
  const res = {};
  for (const [src, outName] of [[TRUTH, 'xmin'], ['f_plain', 'xminplain']]) {
    const g = new Uint8Array(readFileSync(join(d, src + '.bin')));
    const t0 = performance.now();
    const X = crossings(g, 1200, 1200, 10, M, meta.lat + 1, 1 / 1200);
    const sec = (performance.now() - t0) / 1000;
    writeFileSync(join(d, 'l1_' + outName + (process.argv[6] || '') + '.bin'), gridFrom(base, X, (X, i) => geo(X.ewMin[i], X.nsMin[i])));
    res[outName] = sec;
    totSec += sec; totCells += land;
  }
  console.log(name, 'land cells', land, 'roads', res.xmin.toFixed(1) + ' s', (1e6 * res.xmin / land).toFixed(0) + ' us/cell',
    'plain', res.xminplain.toFixed(1) + ' s', (1e6 * res.xminplain / land).toFixed(0) + ' us/cell');
}
console.log('mean', (1e6 * totSec / totCells).toFixed(0), 'us per land cell per grid');
process.exit(0);
