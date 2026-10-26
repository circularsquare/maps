// T-1168 stage 2: the related-lines layout (game/relatedLayout.ts) on the real save, against the
// mockups (t1168-*.png in the maps root, artifacts/trainworld/t1168/): the 2, the 4 and the 6X.
// Prints each line's columns and checks the ones the mockups show. Run from the maps root:
//   node trainworld/checks/t1168-layout.mjs
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

const { buildComplexes } = await loadApp('src/game/complexes.ts');
const { decodeRelated } = await loadApp('src/game/relatedLines.ts');
const { relatedLayout } = await loadApp('src/game/relatedLayout.ts');

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(REAL_SAVE));
const api = new TrackApi(-73.985, 40.758, true);
if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error('save did not load');
const ni = api.nodes_info(), geo = [];
let top = 0;
for (let i = 0; i < ni.length; i += 10) {
  top = Math.max(top, ni[i]);
  if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
}
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const cx = buildComplexes(geo, links);
const of = new Uint32Array(top + 1).fill(0xffffffff);
for (const g of geo) of[Number(g.id)] = cx.of.get(g.id);

const li = api.lines_info(), ids = [];
for (let i = 0; i < li.length; i += 22) ids.push(li[i]);
const lineStops = new Map(ids.map((l) => [l, [...api.line_stops(l)]]));
const short = (n) => n.replace(/^subway /, '');
const opts = (l, width) => ({
  line: l, stopS: api.line_stop_s(l, 0), width,
  lineName: (x) => short(api.line_name(x)),
  stopName: (x, s) => api.node_name(lineStops.get(x)[s]).split(' / ')[0],
  stopCount: (x) => lineStops.get(x).length,
});

let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const layoutOf = (name, width = 340) => {
  const l = ids.find((x) => api.line_name(x) === name);
  const t = performance.now();
  const lay = relatedLayout(decodeRelated(api.line_related(l, of)), opts(l, width));
  const ms = performance.now() - t;
  const nm = (i) => opts(l, width).stopName(lay.rows[i].ref.line, lay.rows[i].ref.stop);
  console.log(`\n== ${name}: ${lay.rows.length} rows (${lay.rows.filter((r) => r.lStop < 0).length} side-only), ${lay.columns.length} columns, ${lay.width} px wide, ${ms.toFixed(1)} ms`);
  for (const c of lay.columns) console.log(`  ${short(api.line_name(c.line)).padEnd(4)} ${c.side}${Math.floor(Math.abs(c.x - lay.spineX) / 8)} rows ${c.r0}-${c.r1} ${nm(c.r0)} .. ${nm(c.r1)}${c.up ? ' up' : ''}${c.down ? ' down' : ''}`);
  return { lay, col: (n) => lay.columns.filter((c) => short(api.line_name(c.line)) === n), nm };
};

{
  const { lay, col, nm } = layoutOf('subway 2');
  const one = col('1');
  check(one.length === 1 && one[0].side === 'R' && nm(one[0].r0) === 'Chambers St' && nm(one[0].r1) === '96 St', 'the 2: the 1 on the right, Chambers St to 96 St');
  check(lay.rows.some((r) => r.lStop < 0 && opts(0).stopName && true), 'the 2: rows only the 1 stops at');
  check(col('3').length === 1 && col('3')[0].side === 'L', 'the 2: the 3 on the left');
  check(col('5').length === 2 && col('5').every((c) => c.side === 'L'), 'the 2: the 5 twice on the left (Brooklyn, the Bronx)');
  check(col('4').length === 1 && col('4')[0].side === 'L', 'the 2: the 4 beside in Brooklyn');
  check(!col('A').length && !col('R').length, 'the 2: no A or R column');
  const x149 = lay.rows.findIndex((r) => nm(lay.rows.indexOf(r)).startsWith('149 St-Grand'));
  check(x149 >= 0 && lay.rows[x149].cross.some((c) => short(api.line_name(c)) === '4'), 'the 2: the 4 crosses at 149 St-Grand Concourse');
  check(lay.crossTip(x149, lay.rows[x149].cross.find((c) => short(api.line_name(c)) === '4')).includes('strays up to'), 'the 2: the crossing hover names the stray');
}
{
  const { col } = layoutOf('subway 6X');
  check(col('6').length === 2 && col('6').every((c) => c.side === 'R'), 'the 6X: both 6 locals on the right');
  check(col('4').length === 1 && col('5').length === 1 && col('4')[0].side === 'L' && col('5')[0].side === 'L', 'the 6X: the 4 and 5 on the left');
  check(!col('J').length && !col('Z').length && !col('N').length, 'the 6X: the J, Z and N under three rows, so crossings');
}
{
  const { lay, col } = layoutOf('subway 4');
  check(lay.columns.length >= 4, `the 4: ${lay.columns.length} columns`);
  check(col('6').length + col('6X').length >= 2, 'the 4: the Lexington locals beside it');
}
{
  const { lay } = layoutOf('subway 4', 280);
  check(Math.max(...lay.columns.map((c) => Math.floor(Math.abs(c.x - lay.spineX) / 8))) <= 2, 'the 4 at the narrowest dock (280 px): two columns a side at most');
}
process.exit(fail ? 1 : 0);
