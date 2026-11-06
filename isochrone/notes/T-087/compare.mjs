// T-087: 1 km road rules against the 100 m solve, with the page's own search.
//   node isochrone/notes/T-087/compare.mjs <tiles dir> <truth> <nstarts> [rule,rule,...|all] [tile,tile,...] > out.json
// Per tile: nstarts random land starts (fixed per tile); the 100 m solve is cached beside the grids
// (t100_<truth>_<k>.f32). Each rule's 1 km solve starts at the 1 km cell holding the start. Pairs:
// every third 100 m cell (rows and columns) reached at both with the 1 km time h1 >= 0.5 h.
// Per rule: n, median |h01 - h1| / h1 ("rel"), median ln(h01 / h1) ("bias": above 0, the 1 km map is
// too fast), ratio of summed times 100 m / 1 km ("ratio"), 90th percentile of rel, and bias by h1 band.
import { readFileSync, readdirSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { cfg } from '../../js/config.js';
import { makeLevel } from '../../js/levels.js';
import { solve } from '../../js/search.js';

const dir = process.argv[2], TRUTH = process.argv[3], NS = +process.argv[4];
const ruleArg = process.argv[5] || 'all';
const tileArg = process.argv[6];
cfg.maxHours = 1024;
cfg.inv = new Float32Array(256).fill(-1);
for (let v = 1; v < 256; v++) cfg.inv[v] = 40 / v;
const BANDS = [[0.5, 2], [2, 8], [8, 32], [32, 1024]];
const seeds = (i) => ({ idx: [i], w: [0], t: [0], hz: [0] });
const pct = (a, p) => a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : null;

function rng(seed) {   // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const hash = (s) => { let h = 2166136261; for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619); return h >>> 0; };

const out = [];
const tiles = tileArg ? tileArg.split(',') : readdirSync(dir).sort();
for (const name of tiles) {
  const d = join(dir, name);
  if (!existsSync(join(d, 'meta.json'))) continue;
  const meta = JSON.parse(readFileSync(join(d, 'meta.json'), 'utf8'));
  const west = meta.lon, north = meta.lat + 1;
  const g01 = new Uint8Array(readFileSync(join(d, TRUTH + '.bin')));
  const L01 = makeLevel(g01, 1200, 1200, west, north, 1 / 1200, false);
  let rules = readdirSync(d).filter((f) => f.startsWith('l1_') && f.endsWith('.bin')).map((f) => f.slice(3, -4));
  if (ruleArg !== 'all') rules = ruleArg.split(',');
  const grids = {}, levels = {};
  for (const k of rules) {
    grids[k] = new Uint8Array(readFileSync(join(d, 'l1_' + k + '.bin')));
    levels[k] = makeLevel(grids[k], 120, 120, west, north, 1 / 120, false);
  }
  const acc = {};
  for (const k of rules) acc[k] = { rel: [], lr: [], bands: BANDS.map(() => []), s1: 0, s01: 0, n: 0 };
  const rand = rng(hash(name));
  let used = 0;
  for (let k = 0; k < NS; k++) {
    let r0, c0, tries = 0;
    do { r0 = 180 + Math.floor(rand() * 840); c0 = 180 + Math.floor(rand() * 840); tries++; } while (!g01[r0 * 1200 + c0] && tries < 100);
    if (!g01[r0 * 1200 + c0]) continue;
    const cache = join(d, `t100_${TRUTH}_${k}.f32`);
    let t01;
    if (existsSync(cache)) {
      const b = readFileSync(cache); t01 = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    } else {
      t01 = (await solve(L01, seeds(r0 * 1200 + c0), () => true, false)).t.slice();
      writeFileSync(cache, Buffer.from(t01.buffer));
    }
    used++;
    const i1 = Math.floor(r0 / 10) * 120 + Math.floor(c0 / 10);
    for (const key of rules) {
      const g1 = grids[key];
      if (!g1[i1]) continue;
      const t1 = (await solve(levels[key], seeds(i1), () => true, false)).t;
      const A = acc[key];
      for (let r = 1; r < 1200; r += 3) for (let c = 1; c < 1200; c += 3) {
        const h01 = t01[r * 1200 + c], h1 = t1[((r / 10) | 0) * 120 + ((c / 10) | 0)];
        if (!(h01 <= 1024) || !(h1 <= 1024) || h1 < 0.5 || h01 <= 0) continue;
        A.rel.push(Math.abs(h01 - h1) / h1);
        const lr = Math.log(h01 / h1);
        A.lr.push(lr);
        for (let b = 0; b < BANDS.length; b++) if (h1 >= BANDS[b][0] && h1 < BANDS[b][1]) { A.bands[b].push(lr); break; }
        A.s1 += h1; A.s01 += h01; A.n++;
      }
    }
  }
  const res = { name, starts: used, rules: {} };
  for (const k of rules) {
    const A = acc[k];
    A.rel.sort((x, y) => x - y); A.lr.sort((x, y) => x - y);
    res.rules[k] = { n: A.n, rel: pct(A.rel, 0.5), rel90: pct(A.rel, 0.9), bias: pct(A.lr, 0.5), ratio: A.s01 / Math.max(1e-9, A.s1),
      bands: A.bands.map((b) => { b.sort((x, y) => x - y); return pct(b, 0.5); }) };
    A.rel = A.lr = A.bands = null;
  }
  out.push(res);
  console.error(name, used, rules.map((k) => `${k} ${res.rules[k].rel?.toFixed(3)}/${res.rules[k].ratio.toFixed(2)}`).join('  '));
}
console.log(JSON.stringify(out));
process.exit(0);
