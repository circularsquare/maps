// T-1899: a fresh city's track built with the edit API the game uses (routes, then stations on
// them), and stop-adding click orders tried on it with the track model (WASM) in Node: 8 sequences
// of stations that lie along one way through the track, 15 click orders each, every one added with
// `insert_stop` as a click adds it. Before T-1899, 17 of the 120 came out going back and forth.
//   node trainworld/checks/t1899-stop-orders.mjs [clicks|build|debug] [tag]
// `build` writes the city as a game file (../artifacts/trainworld/t1899/fresh-city.save) for
// t1899-stop-logic.mjs; `clicks` (the default) exits 1 if any order backtracks.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { gunzipSync, gzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t1899/', import.meta.url));
const { default: init, TrackApi } = await import(pathToFileURL(ROOT + 'app/src/wasm/track/sim.js').href);
await init({ module_or_path: readFileSync(ROOT + 'app/src/wasm/track/sim_bg.wasm') });
const api = new TrackApi(-73.985, 40.758, true);
api.set_cash(1e9);
const must = (r, what) => { if (r !== 0) throw new Error(`${what}: ${api.issue_kinds()}`); };
const free = (x, y) => [0, 0, x, y, 0];
const nodeEnd = (n) => [1, n, 0, 0, 0];
const edgeEnd = (e, x, y) => [2, e, x, y, 0];
const pis = (pts) => new Float64Array(pts.flatMap(([x, y]) => [x, y, 0, 0]));
const edges = () => { const e = api.edges_info(), out = []; for (let i = 0; i < e.length; i += 10) out.push({ id: e[i], a: e[i + 1], b: e[i + 2], len: e[i + 5] }); return out; };
const nodes = () => { const n = api.nodes_info(), out = []; for (let i = 0; i < n.length; i += 10) out.push({ id: n[i], x: n[i + 1], y: n[i + 2], ports: n[i + 4], platform: n[i + 7] }); return out; };
const nearestNode = (x, y) => nodes().reduce((b, n) => (Math.hypot(n.x - x, n.y - y) < Math.hypot(b.x - x, b.y - y) ? n : b));
// the edge nearest a point, by its polyline
const edgeAt = (x, y) => {
  let best = null, bd = Infinity;
  for (const e of edges()) {
    const p = api.edge_polyline(e.id, 0);
    for (let k = 0; k + 3 < p.length; k += 2) {
      const [ax, ay, bx, by] = [p[k], p[k + 1], p[k + 2], p[k + 3]];
      const t = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 || 1)));
      const d = Math.hypot(ax + t * (bx - ax) - x, ay + t * (by - ay) - y);
      if (d < bd) [bd, best] = [d, e.id];
    }
  }
  return best;
};
const route = (from, to, via, hsr = false, what = 'route') => {
  api.set_route_tangent(true);
  api.set_design_speed(hsr ? 320 : 160);
  api.set_route_hsr(hsr);
  must(api.add_route(new Float64Array([...from, ...to]), pis(via), 2, false), what);
  return api.route_end();
};
const stationAt = (x, y, name) => {
  const n = nodes().find((m) => Math.hypot(m.x - x, m.y - y) < 5);
  if (n) must(api.add_station(1, n.id, 0, 0, 200, name), 'station ' + name);
  else must(api.add_station(2, edgeAt(x, y), x, y, 200, name), 'station ' + name);
  return nearestNode(x, y).id;
};

// ---- the city, 25 km north-west of Times Square, x east and y north in metres
const X0 = -25000, Y0 = 25000;
const P = (x, y) => [X0 + x, Y0 + y];
// main line west to east, 0 to 14 km
route(free(...P(0, 0)), free(...P(14000, 0)), [], false, 'main');
// the way round: leaves the main line at 2.5 km heading east, north through three towns, back
// onto the main line at 11.5 km
route(edgeEnd(edgeAt(...P(2500, 0)), ...P(2500, 0)), edgeEnd(edgeAt(...P(11500, 0)), ...P(11500, 0)),
  [P(4500, 400), P(4500, 9000), P(9500, 9000), P(9500, 400)], false, 'way round');
// a branch off the main line's east end, north-east
route(edgeEnd(edgeAt(...P(12800, 0)), ...P(12800, 0)), free(...P(17000, 4500)), [P(14500, 300)], false, 'branch');
// a high-speed line from the main line's west end, 60 km west and south
route(nodeEnd(nearestNode(...P(0, 0)).id), free(...P(-60000, -12000)), [P(-20000, 0)], true, 'high speed');
// a short line of its own 80 m north of Centre, from the way round's first town
route(free(...P(5500, 80)), free(...P(8500, 80)), [], false, 'market spur');
const S = {};
for (const [k, x, y] of [['Westgate', 0, 0], ['Mill Lane', 1500, 0], ['Centre', 7000, 0], ['Eastfield', 13200, 0],
  ['Northby', 4500, 4000], ['Upton', 7000, 9000], ['Hillcrest', 9500, 4000],
  ['Branch Park', 15500, 3000], ['Far End', 17000, 4500],
  ['Ridgeway', -30000, -3000], ['Lakeport', -60000, -12000], ['Station 14', 14000, 0],
  ['Centre Market', 7000, 80], ['Market End', 8500, 80]]) S[k] = stationAt(...P(x, y), k);
console.log('stations', JSON.stringify(S));
must(api.construct_all(), 'construct');

const mode = process.argv[2] ?? 'clicks';
if (mode === 'debug') {
  for (const n of nodes()) console.log('node', n.id, Math.round(n.x - X0), Math.round(n.y - Y0), 'ports', n.ports, 'plat', n.platform, api.node_name(n.id));
  for (const e of edges()) { const p = api.edge_polyline(e.id, 0); console.log('edge', e.id, e.a, '->', e.b, Math.round(e.len), 'poly', p.length / 2, Math.round(p[0] - X0), Math.round(p[1] - Y0), Math.round(p.at(-2) - X0), Math.round(p.at(-1) - Y0)); }
  const id = api.new_line_id();
  must(api.set_line_with_trains(id, new Uint32Array([S.Westgate, S['Station 14']]), 0, 0, 0, 30, 180, 0, 0xff0000, 'probe'), 'line');
  console.log('Westgate-Station 14 path', [...api.line_path(id)].map((p) => `${p >> 1}${p & 1 ? '-' : '+'}`).join(' '), 'passed', [...api.passed_stations(id)].join(','));
  process.exit(0);
}
if (mode === 'build') {
  // the game file: T-007's header (a New York game) with this track
  const raw = gunzipSync(readFileSync(ROOT + 'T-007-real-nyc.save'));
  const n = raw.readUInt32LE(4);
  const header = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
  const h = Buffer.from(JSON.stringify({ ...header, cash: 1e6, clock: 8 * 3600 }), 'utf8');
  const head = Buffer.alloc(8);
  raw.copy(head, 0, 0, 4);
  head.writeUInt32LE(h.length, 4);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(OUT + 'fresh-city.save', gzipSync(Buffer.concat([head, h, Buffer.from(api.save())])));
  writeFileSync(OUT + 'fresh-city-stations.json', JSON.stringify(S));
  console.log('saved', OUT + 'fresh-city.save', nodes().length, 'nodes', edges().length, 'edges');
} else {
  // each sequence lies along one way through the track, in order
  const SEQ = {
    'way round': ['Westgate', 'Mill Lane', 'Northby', 'Upton', 'Hillcrest', 'Eastfield', 'Station 14'],
    'way round, short': ['Mill Lane', 'Upton', 'Eastfield'],
    'way round, towns': ['Mill Lane', 'Northby', 'Upton', 'Hillcrest', 'Eastfield'],
    'main': ['Westgate', 'Mill Lane', 'Centre', 'Eastfield', 'Station 14'],
    'branch': ['Mill Lane', 'Centre', 'Branch Park', 'Far End'],
    'way round to the branch': ['Westgate', 'Northby', 'Upton', 'Hillcrest', 'Branch Park', 'Far End'],
    'high speed': ['Lakeport', 'Ridgeway', 'Westgate', 'Mill Lane', 'Centre'],
    'high speed and way round': ['Lakeport', 'Ridgeway', 'Westgate', 'Northby', 'Upton', 'Eastfield'],
  };
  let seed = 4242;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const shuffle = (a) => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const lenOf = (id) => { const f = api.lines_info(); for (let i = 0; i < f.length; i += 22) if (f[i] === id) return f[i + 9]; return NaN; };
  const name = new Map(Object.entries(S).map(([k, v]) => [v, k]));
  const rows = [];
  let total = 0, bad = 0;
  for (const [label, names] of Object.entries(SEQ)) {
    const seq = names.map((k) => S[k]);
    const orders = [[seq[0], seq.at(-1), ...seq.slice(1, -1)], [seq.at(-1), seq[0], ...seq.slice(1, -1)], [...seq.slice(1, -1), seq[0], seq.at(-1)]];
    for (let k = 0; k < 12; k++) orders.push(shuffle(seq));
    let nb = 0;
    for (const order of orders) {
      const id = api.new_line_id();
      must(api.set_line_with_trains(id, new Uint32Array([]), 0, 0, 0, 30, 180, 0, 0xff0000, 'probe'), 'line');
      for (const s of order) { const ins = api.insert_stop(id, s); if (ins.length) must(api.set_stops(id, Uint32Array.from(ins)), 'stops'); }
      const got = [...api.line_stops(id)];
      const rev = api.line_reversals(id).length + api.line_end_turns(id).length;
      const straight = got.join() === seq.join() || got.join() === seq.slice().reverse().join();
      const isBad = rev > 0 || !straight;
      total++;
      if (isBad) { bad++; nb++; rows.push({ label, order: order.map((s) => name.get(s)), got: got.map((s) => name.get(s)), rev, km: Math.round(lenOf(id) / 100) / 10 }); }
      api.remove_line(id);
    }
    console.log(`${label}: ${nb}/${orders.length} orders backtrack or come out in another order`);
  }
  console.log(`${bad}/${total} in all`);
  for (const r of rows.slice(0, 6)) console.log('  ' + JSON.stringify(r));
  mkdirSync(OUT, { recursive: true });
  writeFileSync(OUT + `fresh-clicks-${process.argv[3] ?? 'x'}.json`, JSON.stringify(rows, null, 1));
  console.log(`${bad ? 'FAIL' : 'PASS'}  no order backtracks or comes out in another order`);
  process.exit(bad ? 1 : 0);
}
