// T-4441 measuring only: the train work after a click run on running track, in Node, the worker's own wasm model.
// node drive.mjs <wasm pkg dir> <save> [edge] [clicks]
import {readFileSync, writeFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {pathToFileURL} from 'node:url';
const [pkg, savePath, edgeArg = '1482', clicksArg = '4', outArg] = process.argv.slice(2);
const mod = await import(pathToFileURL(pkg + '/sim.js').href);
await mod.default({module_or_path: readFileSync(pkg + '/sim_bg.wasm')});
const {TrackApi} = mod;
const raw = gunzipSync(readFileSync(savePath));
if (raw.subarray(0, 4).toString() !== 'TWG1') throw new Error('not a save');
const hl = raw.readUInt32LE(4);
const header = JSON.parse(raw.subarray(8, 8 + hl).toString('utf8'));
let net = new Uint8Array(raw.subarray(8 + hl));
if (new TextDecoder().decode(net.subarray(0, 4)) !== 'TWT6') { const c = TrackApi.convert_save_frame(net, 2); if (c.length) net = c; }
const a = new TrackApi(-73.985, 40.758, true);
if (header.level !== undefined) a.set_load_level(header.level);
let t = performance.now();
if (!a.load(net)) throw new Error('load failed');
console.log('load ms', (performance.now() - t).toFixed(0), 'header clock', header.clock, 'level', header.level);
a.set_now(header.clock);
a.take_phases();
const linesN = a.lines_info().length / 22, edgeN = a.edges_info().length / 10;
console.log('lines', linesN, 'edges', edgeN);
// quiet start
t = performance.now(); a.settle_owed(); console.log('first settle_owed ms', (performance.now() - t).toFixed(1), 'phases dropped', a.take_phases().length);
const parse = s => Object.fromEntries(s.split(';').filter(Boolean).map(e => { const [k, v] = e.split('='); const [ms, n] = v.split('*'); return [k, [+ms, +n]]; }));
const edgeSet = () => { const i = a.edges_info(), m = new Map(); for (let k = 0; k < i.length; k += 10) m.set(i[k], {a: i[k+1], b: i[k+2], len: i[k+5]}); return m; };
const clicks = [];
let now = header.clock;
const N = +clicksArg;
const poly = a.edge_polyline(+edgeArg, 0);
const seg = []; let tot = 0;
for (let i = 2; i < poly.length; i += 2) { const l = Math.hypot(poly[i] - poly[i-2], poly[i+1] - poly[i-1]); seg.push(l); tot += l; }
const at = f => { let s = f * tot; for (let i = 0; i < seg.length; i++) { if (s <= seg[i] || i === seg.length - 1) { const u = Math.min(1, s / seg[i]); return [poly[2*i] + (poly[2*i+2] - poly[2*i]) * u, poly[2*i+1] + (poly[2*i+3] - poly[2*i+1]) * u]; } s -= seg[i]; } };
console.log('edge', edgeArg, 'polyline pts', poly.length / 2, 'length m', Math.round(tot));
let pieces = [+edgeArg];
for (let k = 0; k < N; k++) {
  const f = 0.15 + 0.2 * k, [x, y] = at(f);
  const before = edgeSet();
  now += 0.3; a.set_now(now);
  a.set_station_one(new Float64Array(0));
  const owedBefore = a.owes();
  // the piece that takes the point: the first whose ghost check passes (the ghost's own trial leaves the model as it is)
  let edge = pieces.find(e => a.preview_station(2, e, x, y, 200) === 0);
  if (edge === undefined) { console.log('no piece takes point', k, pieces, a.issue_kinds()); break; }
  t = performance.now();
  const r = a.add_station(2, edge, x, y, 200, '');
  const ms = performance.now() - t;
  const ph = parse(a.take_phases());
  const after = edgeSet();
  const fresh = [...after.keys()].filter(e => !before.has(e));
  pieces = [...pieces, ...fresh];
  clicks.push({k, edge, r, ms, owedBefore, owesAfter: a.owes(), ph, fresh, issues: r ? a.issue_kinds() : undefined});
  console.log('click', k + 1, 'edge', edge, 'r', r, 'add_station ms', ms.toFixed(1), 'owed before', owedBefore, 'phases', Object.keys(ph).length);
  if (r) break;
}
now += 0.3; a.set_now(now);
t = performance.now();
const any = a.settle_owed();
const final = {ms: performance.now() - t, any, ph: parse(a.take_phases())};
console.log('final settle_owed ms', final.ms.toFixed(1), any);
const out = {save: savePath, lines: linesN, edges: edgeN, clicks, final};
writeFileSync(outArg || (pkg + '/../run.json'), JSON.stringify(out, null, 1));
