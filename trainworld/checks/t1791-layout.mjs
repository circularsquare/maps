// T-1791, T-1809: the group view's layout (game/groupLayout.ts). The 4, 5 and 6 with a copy of the 4
// on the real save, a Tokyo group of lines that merge and part (data/packs/real/tyo.save), and
// synthetic cases for the rules: termini and meetings named, skip-stop drawn apart, express stops
// named, branches continuous with their stops together. Prints each diagram as text. From the maps root:
//   node trainworld/checks/t1791-layout.mjs
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

const { buildComplexes } = await loadApp('src/game/complexes.ts');
const { groupLayout, pathPlaces, PITCH, SPREAD, NAME_H } =await loadApp('src/game/groupLayout.ts');

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });

function open(file, lng, lat) {
  const raw = gunzipSync(readFileSync(file));
  const api = new TrackApi(lng, lat, true);
  if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error(`${file} did not load`);
  const ni = api.nodes_info(), geo = [];
  for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
  const fl = api.station_links(), links = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = buildComplexes(geo, links);
  const key = (s) => { const c = cx.of.get(s); return c === undefined ? s : `c${c}`; };
  const li = api.lines_info(), ids = [];
  for (let i = 0; i < li.length; i += 22) ids.push(li[i]);
  const byName = (n) => ids.find((x) => api.line_name(x).replace(/^subway /, '') === n);
  const placesOf = (l) => {
    const p = [...api.passed_stations(l)], passed = [];
    for (let k = 0; k + 1 < p.length; k += 2) passed.push({ gap: p[k], node: p[k + 1] });
    return pathPlaces([...api.line_stops(l)].map(String), passed, key);
  };
  // places where a line outside `group` stops
  const transfer = (group) => {
    const at = new Set();
    for (const l of ids) if (!group.includes(l)) for (const s of api.line_stops(l)) at.add(key(String(s)));
    return (k) => at.has(k);
  };
  const nm = (s) => api.node_name(Number(s)).split(' / ')[0];
  return { api, byName, placesOf, transfer, nm };
}

let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
/** the diagram as text: a character per line at its x (o named stop, . tick stop, - passed, | through) */
const draw = (title, lay, nm = (s) => s) => {
  console.log(`\n== ${title}: ${lay.rows.length} rows, ${lay.rows.filter((r) => r.named).length} named, ${lay.width} by ${Math.round(lay.height)} px`);
  const cols = Math.ceil(lay.width / 3) + 1;
  lay.rows.forEach((row, r) => {
    const s = Array(cols).fill(' ');
    lay.xs.forEach((xk, k) => {
      if (!Number.isFinite(xk[r])) return;
      const c = row.cells[k];
      s[Math.round(xk[r] / 3)] = !c ? '|' : !c.stop ? '-' : row.named ? 'o' : '.';
    });
    const name = row.named ? [...new Set(row.cells.filter(Boolean).map((c) => nm(c.station)))].join(' / ') : '';
    console.log('  ' + s.join('') + '  ' + name);
  });
};
/** a column's stops in row order are its places in path order (or reversed): every stop once */
const everyStop = (lay, k, places) => {
  const got = lay.rows.map((row) => row.cells[k]).filter((c) => c?.stop).map((c) => c.station);
  const want = places.filter((p) => p.stop).map((p) => p.station);
  const loop = want.length > 2 && want[0] === want[want.length - 1];
  const w = loop ? want.slice(0, -1) : want;
  return JSON.stringify(got) === JSON.stringify(w) || JSON.stringify(got) === JSON.stringify([...w].reverse());
};
/** a column's x is set on every row of its span, nowhere else, and its path is one stroke */
const continuous = (lay, k) => {
  const [a, b] = lay.spans[k];
  return lay.xs[k].every((x, r) => Number.isFinite(x) === (r >= a && r <= b)) && (lay.paths[k].match(/M/g) ?? []).length === 1;
};
/** a branch's own stops are one block of rows: no other line's own stop between its first and last */
const ownBlock = (lay, k) => {
  const own = lay.rows.map((row, r) => (row.cells[k]?.stop && row.cells.filter((c) => c?.stop).length === 1 ? r : -1)).filter((r) => r >= 0);
  const blocks = [];
  for (const r of own) { if (blocks.length && blocks.at(-1)[1] >= r - 1 - lay.rows.slice(blocks.at(-1)[1] + 1, r).filter((row) => !row.cells.some((c) => c?.stop)).length) blocks.at(-1)[1] = r; else blocks.push([r, r]); }
  // between two of its own blocks there must be a shared row (where it meets others again)
  return blocks.every((bl, i) => i === 0 || lay.rows.slice(blocks[i - 1][1] + 1, bl[0]).some((row) => row.cells[k]));
};
const namedAt = (lay, k, r) => lay.rows[r].named && !!lay.rows[r].cells[k];

// ---- synthetic cases
const P = (s, stop = true) => ({ key: s, station: s, stop, index: 0 });
const seqOf = (str) => str.split(' ').map((s) => (s.startsWith('(') ? P(s.slice(1, -1), false) : P(s)));
{
  // her Tokyo case in small: Y and F come in from their own branches and meet at m, 1 joins at j; Y
  // runs express (skips l1 l2), Y leaves first at s, then 1 and F part at t
  const Y = seqOf('y1 y2 y3 m a j b (l1) (l2) c d s ya yb');
  const F = seqOf('f1 f2 f3 f4 m a j b l1 l2 c d s e t f9 f8');
  const L1 = seqOf('o1 o2 j b l1 l2 c d s e t g1 g2 g3');
  const lines = [{ num: 1, places: L1 }, { num: 2, places: Y }, { num: 3, places: F }];
  const lay = groupLayout(lines);
  draw('synthetic: three branches in, Y express, out again (short)', lay);
  const row = (s) => lay.rows.findIndex((r) => r.cells.some((c) => c?.station === s));
  lines.forEach((l, k) => check(everyStop(lay, k, l.places), `synthetic ${l.num}: every stop once, in order`));
  lines.forEach((l, k) => check(continuous(lay, k), `synthetic ${l.num}: one continuous bar`));
  lines.forEach((l, k) => check(ownBlock(lay, k), `synthetic ${l.num}: its own stops in one block per branch`));
  check(['y1', 'f1', 'o1', 'yb', 'f8', 'g3'].every((s) => lay.rows[row(s)].named), 'termini named');
  check(lay.rows[row('m')].named && lay.rows[row('j')].named, 'meetings named (m, j)');
  check(lay.rows[row('s')].named && lay.rows[row('t')].named, 'partings named (s, t)');
  check(lay.rows[row('b')].named && lay.rows[row('c')].named, 'the stops either side of the stretch Y skips named (b, c)');
  check(!lay.rows[row('l1')].named && lay.rows[row('l1')].h < NAME_H, 'a stop no rule names is a tick (l1)');
  check(!lay.rows[row('a')].named, 'a plain shared stop is a tick (a)');
  const [y, f, one] = [1, 2, 0];
  const r = row('d');
  check(lay.together(y, f, r) && lay.together(f, one, r), 'Y, F and 1 together at d');
  check(Math.abs(lay.xs[f][r] - lay.xs[one][r]) === PITCH, `F and 1 side by side at d, ${PITCH} px apart`);
  const o = lay.rows[r].order;
  check(o.indexOf(f) === 1 || o.indexOf(one) === 1, `the express Y outside the locals at d (order ${o.map((k) => lines[k].num)})`);
  const ry = row('y2');
  check(Math.abs(lay.xs[y][ry] - lay.xs[f][ry]) >= SPREAD - 0.01, `Y and F apart on their branches (${(lay.xs[y][ry] - lay.xs[f][ry]).toFixed(1)} px)`);
  check(row('g1') === row('f9'), "after t, the ticks of F's and 1's tails share a row (f9 and g1)");
  check(Number.isFinite(lay.xs[y][ry]) && lay.bullets.filter((b) => b.col === y).length === 2, "Y's bullets at both ends");
  const all = groupLayout(lines, { all: true });
  check(all.rows.every((x) => x.named), 'all stops: every row named');
  check(all.height > lay.height, `all stops: a longer diagram (${Math.round(lay.height)} to ${Math.round(all.height)} px)`);
}
{
  // T-1817: T1 runs with T3 from the north and with T2 after they meet: T1 between them
  const T1 = seqOf('a1 a2 a3 m x1 x2 x3 x4 x5 x6 e');
  const T2 = seqOf('b1 b2 b3 m x1 x2 x3 x4 x5 x6 e');
  const T3 = seqOf('a1 a2 a3 m x1 z1');
  const lay = groupLayout([{ num: 1, places: T1 }, { num: 2, places: T2 }, { num: 3, places: T3 }]);
  draw('synthetic: a line between two it shares with', lay);
  const rb = lay.rows.findIndex((r) => r.cells[1]?.station === 'b2'), ra = lay.rows.findIndex((r) => r.cells[0]?.station === 'a2');
  check(lay.rows[rb].order.indexOf(0) === 1, `T1 in the middle beside T2's branch (order ${lay.rows[rb].order.map((k) => k + 1)})`);
  check(Math.sign(lay.xs[0][ra] - lay.xs[2][ra]) === Math.sign(lay.xs[1][rb] - lay.xs[0][rb]), 'T1 on the side of T3 facing T2 from the top');
}
{
  // skip-stop on one track: A stops at 1 3 5 7, B at 2 4 6 7, each passing the other's
  const A = seqOf('0 1 (2) 3 (4) 5 (6) 7 8');
  const B = seqOf('0 (1) 2 (3) 4 (5) 6 7 8');
  const lay = groupLayout([{ num: 1, places: A }, { num: 2, places: B }]);
  draw('synthetic: skip-stop', lay);
  const r4 = lay.rows.findIndex((r) => r.cells[1]?.station === '4');
  check(!lay.together(0, 1, r4) && Math.abs(lay.xs[0][r4] - lay.xs[1][r4]) >= SPREAD, 'skip-stop lines drawn apart between the stops both make');
  check(lay.together(0, 1, 0) && lay.together(0, 1, lay.rows.length - 1), '...and together where both stop');
}
{
  // a local and an express on one track, the express turned round, a short turn, a line crossing at c
  const lay = groupLayout([
    { num: 1, places: seqOf('a b c d e') },
    { num: 2, places: seqOf('e (d) c (b) a') },
    { num: 3, places: seqOf('a b c') },
    { num: 4, places: seqOf('x c y') },
  ]);
  draw('synthetic: local, express, short turn, crossing', lay);
  check(lay.rows.length === 7 && lay.rows.every((r) => !r.cells[1] || r.cells[1].stop === (r.cells[1].station !== 'b' && r.cells[1].station !== 'd')), 'the express is turned round and its passes kept');
  check(lay.rows[lay.spans[2][0]].cells[2].station === 'a' && lay.rows[lay.spans[2][1]].cells[2].station === 'c', 'the short turn spans a to c');
  check([0, 1, 2, 3].every((k) => continuous(lay, k)), 'every bar continuous (the crossing line too)');
}

// ---- New York: the 4, 5, 6 and a copy of the 4
{
  const ny = open(REAL_SAVE, -73.985, 40.758);
  const ls = ['4', '5', '6'].map(ny.byName);
  check(ls.every((l) => l !== undefined), 'the 4, 5 and 6 are in the save');
  const lines = [...ls.map((l) => ({ num: l, places: ny.placesOf(l) })), { num: 999, places: ny.placesOf(ls[0]) }];
  const lay = groupLayout(lines, { transfer: ny.transfer(ls) });
  draw('4, 5, 6 and a copy of the 4', lay, ny.nm);
  lines.forEach((x, k) => check(everyStop(lay, k, x.places), `NYC column ${k}: every stop once, in order`));
  lines.forEach((x, k) => check(continuous(lay, k), `NYC column ${k}: one continuous bar`));
  check(lay.rows.filter((r) => r.cells[0] && r.cells[3]).length === lay.rows.filter((r) => r.cells[0]).length, 'the copy of the 4 shares every row of the 4');
  const both = lay.rows.map((r, i) => i).filter((i) => lay.rows[i].cells[0]?.stop && lay.rows[i].cells[3]?.stop);
  check(both.every((i) => lay.together(0, 3, i) && Math.abs(lay.xs[0][i] - lay.xs[3][i]) === PITCH), 'the 4 and its copy run side by side, PITCH apart');
  const ends = (k) => [lay.spans[k][0], lay.spans[k][1]].every((r) => namedAt(lay, k, r));
  check([0, 1, 2, 3].every(ends), 'every terminus named');
  const named = lay.rows.filter((r) => r.named).length, stops = lay.rows.filter((r) => r.cells.some((c) => c?.stop)).length;
  check(named < stops, `fewer named rows than stop rows (${named} of ${stops})`);
  const usq = lay.rows.findIndex((r) => r.cells.some((c) => c && /14 St-Union Sq/.test(ny.nm(c.station))));
  check(usq >= 0 && lay.rows[usq].named, '14 St-Union Sq named');
}
// ---- Tokyo: the Tokaido, Yokosuka, Shonan-Shinjuku and Ueno-Tokyo lines, which meet and part
{
  const ty = open(new URL('../data/packs/real/tyo.save', import.meta.url), 139.7, 35.68);
  const names = ['東海道本線', '横須賀線', '湘南新宿ライン', '上野東京ライン（1）'];
  const ls = names.map(ty.byName);
  check(ls.every((l) => l !== undefined), 'the Tokyo lines are in tyo.save');
  const lines = ls.map((l) => ({ num: l, places: ty.placesOf(l) }));
  const lay = groupLayout(lines, { transfer: ty.transfer(ls) });
  draw('Tokyo: Tokaido, Yokosuka, Shonan-Shinjuku, Ueno-Tokyo', lay, ty.nm);
  lines.forEach((x, k) => check(everyStop(lay, k, x.places), `Tokyo ${names[k]}: every stop once, in order`));
  lines.forEach((x, k) => check(continuous(lay, k), `Tokyo ${names[k]}: one continuous bar`));
  lines.forEach((x, k) => check(ownBlock(lay, k), `Tokyo ${names[k]}: its own stops in one block per branch`));
  check(lines.every((_, k) => [lay.spans[k][0], lay.spans[k][1]].every((r) => namedAt(lay, k, r))), 'every terminus named');
}
console.log(fail ? `\n${fail} FAILED` : '\nall passed');
process.exit(fail ? 1 : 0);
