// T-047: the 1 km and the 100 m answer inside each sample tile, with the page's own search.
//   node isochrone/notes/T-047/compare.mjs <dir of sample.py> > out.json
// Five starts a tile (centre and the four quarter points, snapped to land); the search stays
// inside the tile at both levels. For every 100 m cell reached at either level (L1 read at the
// 100 m cell, as the page paints it): the relative time difference, whether the colour band
// differs, and whether only one level reaches it.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { cfg } from '../../js/config.js';
import { makeLevel, snap } from '../../js/levels.js';
import { solve } from '../../js/search.js';

const dir = process.argv[2];
cfg.maxHours = 1024;
cfg.inv = new Float32Array(256).fill(-1);
for (let v = 1; v < 256; v++) cfg.inv[v] = 40 / v;
const EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const band = (h) => { let k = 0; while (k < EDGES.length && h > EDGES[k]) k++; return k; };
const PAIRS = [['l1_our', 'f_plain'], ['l1_our', 'f_glo'], ['l1_map', 'f_road'], ['l1_map', 'f_road_glo']];
const STARTS = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]];

const pct = (a, p) => a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : null;
const out = [];
for (const name of readdirSync(dir).sort()) {
  const d = join(dir, name);
  if (!existsSync(join(d, 'meta.json'))) continue;
  const meta = JSON.parse(readFileSync(join(d, 'meta.json'), 'utf8'));
  const west = meta.lon, north = meta.lat + 1;
  const res = { ...meta, name, pairs: {} };
  // land shares
  const fp = new Uint8Array(readFileSync(join(d, 'f_plain.bin')));
  let land = 0; for (const v of fp) if (v) land++;
  res.land = land / fp.length;
  for (const [a, b] of PAIRS) {
    if (!meta.have.includes(a) || !meta.have.includes(b)) continue;
    const g1 = new Uint8Array(readFileSync(join(d, a + '.bin'))), g01 = new Uint8Array(readFileSync(join(d, b + '.bin')));
    const L1 = makeLevel(g1, 120, 120, west, north, 1 / 120, false);
    const L01 = makeLevel(g01, 1200, 1200, west, north, 1 / 1200, false);
    const rel = [], acc = { both: 0, band: 0, only1: 0, only01: 0, ms: 0, sum1: 0, sum01: 0 };
    let starts = 0;
    for (const [fx, fy] of STARTS) {
      const lng = west + fx, lat = north - fy;
      const s1 = snap(L1, lng, lat, 10), s01 = snap(L01, lng, lat, 100);
      if (s1 < 0 || s01 < 0) continue;
      const t0 = performance.now();
      const r1 = await solve(L1, { idx: [s1], w: [0], t: [0], hz: [0] }, () => true, false);
      const r01 = await solve(L01, { idx: [s01], w: [0], t: [0], hz: [0] }, () => true, false);
      acc.ms += performance.now() - t0;
      const t1 = r1.t.slice(), t01 = r01.t.slice();
      starts++;
      for (let r = 0; r < 1200; r++) for (let c = 0; c < 1200; c++) {
        const h01 = t01[r * 1200 + c], h1 = t1[((r / 10) | 0) * 120 + ((c / 10) | 0)];
        const in01 = h01 <= 1024, in1 = h1 <= 1024;
        if (in01 && in1) {
          acc.both++;
          if (band(h01) !== band(h1)) acc.band++;
          if (h1 >= 0.5) rel.push(Math.abs(h01 - h1) / h1);
          acc.sum1 += h1; acc.sum01 += h01;
        } else if (in01 && g1[((r / 10) | 0) * 120 + ((c / 10) | 0)]) acc.only01++;
        else if (in1 && g01[r * 1200 + c]) acc.only1++;
      }
    }
    if (!starts) continue;
    rel.sort((x, y) => x - y);
    const all = acc.both + acc.only1 + acc.only01;
    res.pairs[a + '/' + b] = {
      starts, relMedian: pct(rel, 0.5), relP90: pct(rel, 0.9),
      bandDiff: acc.band / Math.max(1, acc.both),
      onlyOne: (acc.only1 + acc.only01) / Math.max(1, all),
      ratio: acc.sum01 / Math.max(1e-9, acc.sum1),
      msPerSolvePair: Math.round(acc.ms / starts),
    };
  }
  out.push(res);
  console.error(meta.label, JSON.stringify(res.pairs));
}
console.log(JSON.stringify(out, null, 1));
process.exit(0);
