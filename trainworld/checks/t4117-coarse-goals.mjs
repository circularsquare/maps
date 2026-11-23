// T-4117: stars and goals count coarse figures. A region shown from coarse data earns its star and goals as one shown
// fine does (game/goalRules.ts `goalsDue`, which the clock client posts from; the city first in achievements.ts reads
// no level either); figures not yet final
// earn nothing, and what the money state already records is not earned again. With the setting every zone is
// coarse, so a region star held back from coarse data would never come on a phone.
//   node trainworld/checks/t4117-coarse-goals.mjs      (from the maps root; no browser, a few seconds)
// Prints PASS or FAIL lines and exits 1 on failure.
import { loadApp } from './lib.mjs';

globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const store = new Map();
globalThis.localStorage ??= { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) };

import { readFile } from 'node:fs/promises';
const { goalsDue } = await loadApp("export { goalsDue } from '@app/src/game/goalRules.ts';");

let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const key = (gs) => gs.map((g) => g.zone + ':' + (g.share ?? 'star')).sort().join(',');

const views = new Map([
  ['tokyo', { final: true, split: { train: 24.6 } }],
  ['osaka', { final: true, split: { train: 9.9 } }],
  ['nagoya', { final: false, split: { train: 35 } }],
  ['sapporo', { final: true, split: { train: 41 } }],
]);
// (every zone here is coarse as far as the rule is concerned: it takes no level)
const none = { stars: {}, goals: {} };
check('coarse zones earn their star and goals', key(goalsDue(views, none)) === 'sapporo:20,sapporo:30,sapporo:40,sapporo:star,tokyo:20,tokyo:star', key(goalsDue(views, none)));
check('a zone under 10% earns nothing', !goalsDue(views, none).some((g) => g.zone === 'osaka'));
check('figures not yet final earn nothing', !goalsDue(views, none).some((g) => g.zone === 'nagoya'));
const had = { stars: { tokyo: 100, sapporo: 50 }, goals: { tokyo: { 20: 120 }, sapporo: { 20: 60, 30: 70 } } };
check('what was earned is not earned again', key(goalsDue(views, had)) === 'sapporo:40', key(goalsDue(views, had)));
// the places that award them read no level of detail (T-1223 held stars back from `roughZones`)
for (const f of ['app/src/workers/clockClient.ts', 'app/src/game/achievements.ts', 'app/src/game/goalRules.ts']) {
  const src = await readFile(new URL('../' + f, import.meta.url), 'utf8');
  check(`${f} reads no roughZones`, !src.includes('roughZones'));
}

console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
