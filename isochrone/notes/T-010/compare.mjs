// T-010: each 1 km road rule against 100 m with roads, with the page's own search
// (the T-047 'road' comparison, notes/T-047/compare.mjs, with the 100 m solve shared by every rule).
//   node isochrone/notes/T-010/compare.mjs <dir of sample.py> [rule,rule,...|all] [f_road|f_plain] > out.json
// Five starts a tile, the search inside the tile. Per rule: median |t100 - t1| / t1 (cells over
// 0.5 h), the share of cells reached at both whose colour band differs, the ratio of summed times
// (100 m / 1 km; above 1 means the 1 km map is too fast), each also for urban and other cells.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { cfg } from '../../js/config.js';
import { makeLevel, snap } from '../../js/levels.js';
import { solve } from '../../js/search.js';

const dir = process.argv[2];
const only = process.argv[3] && process.argv[3] !== 'all' ? process.argv[3].split(',') : null;   // optional rule list
const FINE = process.argv[4] || 'f_road';   // the 100 m grid compared against
cfg.maxHours = 1024;
cfg.inv = new Float32Array(256).fill(-1);
for (let v = 1; v < 256; v++) cfg.inv[v] = 40 / v;
const EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const band = (h) => { let k = 0; while (k < EDGES.length && h > EDGES[k]) k++; return k; };
const STARTS = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];
const pct = (a, p) => a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : null;
const seeds = (i) => ({ idx: [i], w: [0], t: [0], hz: [0] });

const out = [];
for (const name of readdirSync(dir).sort()) {
  const d = join(dir, name);
  if (!existsSync(join(d, 'meta.json'))) continue;
  const meta = JSON.parse(readFileSync(join(d, 'meta.json'), 'utf8'));
  if (only) meta.l1 = meta.l1.filter((k) => only.includes(k));
  const west = meta.lon, north = meta.lat + 1;
  const g01 = new Uint8Array(readFileSync(join(d, FINE + '.bin')));
  const urban = new Uint8Array(readFileSync(join(d, 'urban.bin')));
  const L01 = makeLevel(g01, 1200, 1200, west, north, 1 / 1200, false);
  const grids = {};
  for (const k of meta.l1) grids[k] = new Uint8Array(readFileSync(join(d, 'l1_' + k + '.bin')));
  const acc = {};
  for (const k of meta.l1) acc[k] = { u: { rel: [], both: 0, band: 0, s1: 0, s01: 0 }, o: { rel: [], both: 0, band: 0, s1: 0, s01: 0 }, only: 0, all: 0 };
  let starts = 0;
  for (const [fx, fy] of STARTS) {
    const lng = west + fx, lat = north - fy;
    const s01 = snap(L01, lng, lat, 100);
    if (s01 < 0) continue;
    const t01 = (await solve(L01, seeds(s01), () => true, false)).t.slice();
    starts++;
    for (const k of meta.l1) {
      const g1 = grids[k];
      const L1 = makeLevel(g1, 120, 120, west, north, 1 / 120, false);
      const s1 = snap(L1, lng, lat, 10);
      if (s1 < 0) continue;
      const t1 = (await solve(L1, seeds(s1), () => true, false)).t;
      const A = acc[k];
      for (let r = 0; r < 1200; r++) for (let c = 0; c < 1200; c++) {
        const i01 = r * 1200 + c, i1 = ((r / 10) | 0) * 120 + ((c / 10) | 0);
        const h01 = t01[i01], h1 = t1[i1];
        const in01 = h01 <= 1024, in1 = h1 <= 1024;
        if (in01 && in1) {
          const B = urban[i01] ? A.u : A.o;
          B.both++; A.all++;
          if (band(h01) !== band(h1)) B.band++;
          if (h1 >= 0.5) B.rel.push(Math.abs(h01 - h1) / h1);
          B.s1 += h1; B.s01 += h01;
        } else if ((in01 && g1[i1]) || (in1 && g01[i01])) { A.only++; A.all++; }
      }
    }
  }
  const res = { label: meta.label, name, starts, share: meta.share, bits: meta.bits_agree, rules: {} };
  const f = (B) => { B.rel.sort((x, y) => x - y); return { n: B.both, rel: pct(B.rel, 0.5), band: B.band / Math.max(1, B.both), ratio: B.s01 / Math.max(1e-9, B.s1) }; };
  for (const k of meta.l1) {
    const A = acc[k];
    const all = { rel: A.u.rel.concat(A.o.rel), both: A.u.both + A.o.both, band: A.u.band + A.o.band, s1: A.u.s1 + A.o.s1, s01: A.u.s01 + A.o.s01 };
    res.rules[k] = { all: f(all), urban: f(A.u), other: f(A.o), only: A.only / Math.max(1, A.all) };
  }
  out.push(res);
  console.error(meta.label, Object.entries(res.rules).map(([k, v]) => `${k} ${v.all.rel?.toFixed(3)}/${v.all.band.toFixed(2)}/${v.all.ratio.toFixed(2)}`).join('  '));
}
console.log(JSON.stringify(out, null, 1));
process.exit(0);
