// T-1791 stage 2: the group view's layout (game/groupLayout.ts) on the real save: the 4, 5 and 6
// with a copy of the 4, and two unrelated lines. Prints each diagram as text. Run from the maps root:
//   node trainworld/checks/t1791-layout.mjs
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

const { buildComplexes } = await loadApp('src/game/complexes.ts');
const { groupLayout, pathPlaces } = await loadApp('src/game/groupLayout.ts');

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(REAL_SAVE));
const api = new TrackApi(-73.985, 40.758, true);
if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error('save did not load');
const ni = api.nodes_info(), geo = [];
for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const cx = buildComplexes(geo, links);
const key = (s) => { const c = cx.of.get(s); return c === undefined ? s : `c${c}`; };

const li = api.lines_info(), ids = [];
for (let i = 0; i < li.length; i += 22) ids.push(li[i]);
const short = (n) => n.replace(/^subway /, '');
const byName = (n) => ids.find((x) => short(api.line_name(x)) === n);
const placesOf = (l) => {
  const p = [...api.passed_stations(l)], passed = [];
  for (let k = 0; k + 1 < p.length; k += 2) passed.push({ gap: p[k], node: p[k + 1] });
  return pathPlaces([...api.line_stops(l)].map(String), passed, key);
};
const nm = (s) => api.node_name(Number(s)).split(' / ')[0];

let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const show = (title, lines) => {
  const lay = groupLayout(lines);
  console.log(`\n== ${title}: ${lay.rows.length} rows`);
  for (const r of lay.rows) console.log('  ' + r.map((c) => (c ? (c.stop ? 'o' : '-') : ' ')).join(' ') + '  ' + [...new Set(r.filter(Boolean).map((c) => nm(c.station)))].join(' | '));
  return lay;
};

{
  const ls = ['4', '5', '6'].map(byName);
  check(ls.every((l) => l !== undefined), 'the 4, 5 and 6 are in the save');
  const lines = [...ls.map((l) => ({ num: l, places: placesOf(l) })), { num: 999, places: placesOf(ls[0]) }];
  const lay = show('4, 5, 6 and a copy of the 4', lines);
  const count = (k, stop) => lay.rows.filter((r) => r[k] && r[k].stop === stop).length;
  lines.forEach((x, k) => check(count(k, true) === x.places.filter((p) => p.stop).length, `column ${k}: every stop once (${count(k, true)})`));
  check(lay.rows.every((r, i) => r.some(Boolean)) , 'no empty row');
  check(lay.rows.filter((r) => r[0] && r[3]).length === lay.rows.filter((r) => r[0]).length, 'the copy of the 4 shares every row of the 4');
  const shared = lay.rows.filter((r) => r[0] && r[1]).length;
  check(shared >= 10, `the 4 and the 5 share rows (${shared})`);
  const tail4 = lay.rows.findIndex((r) => r[0] && nm(r[0].station) === 'Woodlawn');
  check(lay.breaks[0].length === 1 && lay.breaks[0][0] <= tail4, `the 4's Bronx branch goes below, its column broken (${lay.breaks[0]})`);
}
{
  // a local and an express on one track: the express passes b and d, a short turn stops at a to c
  const P = (s, stop = true) => ({ key: s, station: s, stop });
  const lay = groupLayout([
    { num: 1, places: ['a', 'b', 'c', 'd', 'e'].map((s) => P(s)) },
    { num: 2, places: [P('e'), P('d', false), P('c'), P('b', false), P('a')] },
    { num: 3, places: ['a', 'b', 'c'].map((s) => P(s)) },
    { num: 4, places: ['x', 'c', 'y'].map((s) => P(s)) },
  ]);
  console.log('\n== synthetic\n' + lay.rows.map((r) => '  ' + r.map((c) => (c ? (c.stop ? 'o' : '-') : ' ')).join(' ') + '  ' + (r.find(Boolean)?.station ?? '')).join('\n'));
  check(lay.rows.length === 7 && lay.rows.every((r) => !r[1] || r[1].stop === (r[1].station !== 'b' && r[1].station !== 'd')), 'the express is turned round and its passes marked');
  check(lay.rows[lay.spans[2][0]][2].station === 'a' && lay.rows[lay.spans[2][1]][2].station === 'c', 'the short turn spans a to c');
  check(lay.rows[2][3]?.station === 'x' && lay.rows[6][3]?.station === 'y' && lay.breaks[3][0] === 6, 'a crossing line: its own station before the shared one above it, the one after it at the bottom, broken');
}
{
  const a = byName('7'), b = byName('A');
  const lay = show('7 and A, unrelated', [{ num: a, places: placesOf(a) }, { num: b, places: placesOf(b) }]);
  check(lay.spans[0][0] === 0 && lay.spans[1][0] === 0, 'unrelated lines side by side from the top');
}
console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);
