// T-1382: the clock worker's state rows (workers/stateRows.ts) read incrementally against a full read,
// after random series of edits on a real network in the track model (WASM, as the worker holds it):
// routes drawn from free ground and from nodes, stations added, renamed and removed, lines made, copied,
// restyled, re-stopped, removed, train counts and headways, track and stations constructed, blueprint
// switched off, doubled up, links, upgrades, nodes moved, edges deleted, undo and redo, and route
// previews (which apply an edit and take it back), bus lines drawn and redrawn. Two models take the same
// edits in step (reading a line's build cost runs a trial that settles headway lines a little differently
// each time, so one model read twice is not the same twice); one is read with rows kept from the steps
// before, the other whole with a new `StateRows()`, and after each edit, applied or refused, the two must
// be equal field for field.
//   node trainworld/checks/t1382-rows.mjs [seeds] [steps] [nyc|japan|tokyo|asia] (from the maps root; 4 seeds of 30 steps, about 30 s; 60 of 60 for a long run;
//     tokyo: lines timed train by train, T-3896, 12 seeds of 10 steps; asia: the 2,647-station Asia save, T-4113)
// T-4113: the incremental read is the bounded one (workers/regions.ts): only records in the regions an edit dirtied,
// the records it touched and those whose bulk numbers moved are read again, with the worker's platform lines and
// complexes as the env, so their neighbour effects are covered. Every few steps a read with a focus (a rows state's)
// comes first and leaves regions dirty, which the read after must take. After each read the regions must be the
// ones a fresh build from the same nodes and edges makes. CAP=n sets the regions' station cap, BUDGET=n the focus
// budget (low here, so focus reads defer); the work line reports the records read again against the whole.
//   --self-test-touched holds back the model's touched records: the check must fail.
//   node trainworld/checks/t1382-rows.mjs --self-test                      the check must fail with the
//     model's list of changed lines held back, and --self-test-names with its list of changed names
//     (they prove the check can tell); --self-test-phases holds the same list back and compares only the
//     phase tables (T-1540), --self-test-trials the list of lines changed since the last take, which the
//     phase tables are read from after the trials that price a planned line
// T-1540: after every step the phase tables the rows lay out line by line (`StateRows.phases`) must equal
// the model's whole `phases()` byte for byte, and be the same array object as last time when no line's
// tables changed.
// T-4130: each seed first clears a built junction's crossing from far away (a line through it broken where no station
// is, with its path kept, so only the line's no longer counting moves the junction's row; `crossDone`), and the random
// edits break lines too (breakLine). New York's save has no built junction with a crossing, so the case runs on Asia's.
// T-4125: after every step the page's nearest track node and junction to the capacity tags and to points near nodes
// (map/snapIndex.ts `nearestNode`, from its grids) must be the ones a scan of every node finds.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const { StateRows, upgradesOf, Regions, PlatformLines, buildComplexes } = await loadApp("export { StateRows, upgradesOf } from '@app/src/workers/stateRows.ts'; export { Regions } from '@app/src/workers/regions.ts'; export { PlatformLines } from '@app/src/workers/platformLines.ts'; export { buildComplexes } from '@app/src/game/complexes.ts';");
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const { nearestNode, toLocal } = await loadApp("export { nearestNode } from '@app/src/map/snapIndex.ts'; export { toLocal } from '@app/src/map/geo.ts';");

const SELF = process.argv.includes('--self-test-touched') ? 'touched' : process.argv.includes('--self-test') ? 'lines' : process.argv.includes('--self-test-names') ? 'names' : process.argv.includes('--self-test-phases') ? 'phases' : process.argv.includes('--self-test-trials') ? 'trials' : null;
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const SEEDS = Number(args[0] ?? 4), STEPS = Number(args[1] ?? 30), SETUP = args[2] ?? 'nyc';
// T-3896: Tokyo's has lines timed train by train (T-3346), which New York's has none of
const SAVE = SETUP === 'asia' ? ARTIFACTS + 't3600-asianita-1852.save' : SETUP === 'japan' ? ARTIFACTS + 't737-japan-trains.save' : SETUP === 'tokyo' ? fileURLToPath(new URL('fixtures/t3302-tokyo-day77.save', import.meta.url)) : REAL_SAVE;
const raw = gunzipSync(readFileSync(SAVE));
const saveBytes = raw.subarray(8 + raw.readUInt32LE(4));

if (process.env.CAP) StateRows.regionCap = Number(process.env.CAP);
StateRows.focusBudget = Number(process.env.BUDGET ?? 200);
/** T-4113: the worker's platform lines (one per model, as the worker keeps one) and complexes, beside the env */
function envFor(api) {
  const pl = new PlatformLines();
  return {
    ...env,
    pl,
    platformLine: (a, id, x, y, platform) => pl.of(a, id, x, y, platform, 1),
    complexKey: (a) => a.station_links().join(),
    complexes: (rows) => {
      const fl = api.station_links(), links = [];
      for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
      return buildComplexes(rows.map((r) => ({ id: String(r.id), x: r.x, y: r.y, heading: r.heading, length: r.platform, ...(r.line ? { line: r.line } : {}) })), links);
    },
  };
}
const env = {
  lngLat: (x, y) => [x / 1000, y / 1000],
  periodHours: new Float64Array([4, 6, 4, 4, 6]),
  // the worker's bus row, shortened: a bus line's stops that are no station join the nodes
  busRow: (a, id, nodes) => {
    const r = a.bus_route(id), st = a.bus_stops(id), stops = [];
    for (let k = 0; k + 7 < st.length; k += 8) {
      const x = { id: st[k], x: st[k + 1], y: st[k + 2], kind: st[k + 4], station: st[k + 5] };
      stops.push(x);
      if (x.station < 0 && !nodes.some((n) => n.id === x.id)) nodes.push({ id: x.id, x: x.x, y: x.y, platform: 0, bus: true });
    }
    return { built: r[0] === 1, route: Array.from(r), stops, street: Array.from(a.bus_street(id)), roadM: a.bus_road_m(id) };
  },
};
const FRAME = 'frame';
// WHOLE=1: always read the phase tables in one call (the way used when an edit changes many lines)
if (process.env.WHOLE === '1') StateRows.wholePhases = 0;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

/** a read as the worker's `buildState` makes it: the platform lines told the edges and then the lines */
const read = (rows, api, e = env, focus) => {
  rows.bind(api, FRAME);
  rows.begin(api, focus);
  const edges = rows.edges(api);
  e.pl?.sync(edges);
  const nodes = rows.nodes(api, e, upgradesOf(api));
  const lines = rows.lines(api, e, nodes);
  e.pl?.setLines(lines);
  return { edges, nodes, lines };
};
/** the first row where two reads differ, for the message */
function firstDifference(a, b) {
  for (const k of ['edges', 'nodes', 'lines']) {
    if (a[k].length !== b[k].length) return `${k}: ${a[k].length} rows against ${b[k].length}`;
    for (let i = 0; i < a[k].length; i++) {
      const x = JSON.stringify(a[k][i]), y = JSON.stringify(b[k][i]);
      if (x === y) continue;
      const ra = a[k][i], rb = b[k][i];
      const fields = Object.keys({ ...ra, ...rb }).filter((f) => JSON.stringify(ra[f]) !== JSON.stringify(rb[f]));
      return `${k} row ${i} (id ${ra.id}) differs in ${fields.join(', ')}: kept ${JSON.stringify(ra[fields[0]])?.slice(0, 80)} against ${JSON.stringify(rb[fields[0]])?.slice(0, 80)}`;
    }
  }
  return null;
}

/** the first place two phase tables differ, or null */
function phaseDifference(a, b) {
  if (a.length !== b.length) return `${a.length} floats against ${b.length}`;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return `float ${i} (phase ${Math.floor(i / 4)}) is ${a[i]} against ${b[i]}`;
  return null;
}

/** T-4125: the page's nearest track node and junction to a capacity tag (map/snapIndex.ts `nearestNode`, from its
 * grids; the tram labels and the junction links) against a scan of every node, on the nodes as the read left them */
async function nearDifference(rows, api, rnd) {
  const ll = (x, y) => [x / 111320, Math.max(-80, Math.min(80, y / 111320))];
  const all = rows.nodes.map((n) => { const [lng, lat] = ll(n.x, n.y); return { id: n.id, lng, lat, ports: n.ports ?? 0, ...(n.bus ? { bus: true } : {}), ...(n.port ? { port: true } : {}), ...(n.brtStation ? { brtStation: true } : {}), ...(n.kind ? { kind: n.kind } : {}) }; });
  const w = { nodes: new Map(all.map((n) => [n.id, n])) }, at = all.map((n) => toLocal(n.lng, n.lat));
  const pts = [], m = api.markers_in(0);
  for (let k = 0; k < m.length && pts.length < 150; k += 7) pts.push(toLocal(...ll(m[k], m[k + 1])));
  for (let k = 0; k < 40; k++) { const [x, y] = at[Math.floor(rnd() * at.length)]; pts.push([x + (rnd() - 0.5) * 4000, y + (rnd() - 0.5) * 4000]); }
  const sets = { track: (n) => !n.bus && !n.port && !n.brtStation, junction: (n) => n.ports >= 3 };
  // a small set's grid is built when the thread is next idle
  for (const set in sets) nearestNode(w, 0, 0, set);
  await new Promise((r) => setTimeout(r, 60));
  for (const [set, keep] of Object.entries(sets))
    for (const [x, y] of pts) {
      let best = Infinity, hit = null;
      all.forEach((n, i) => { if (!keep(n)) return; const d = (at[i][0] - x) ** 2 + (at[i][1] - y) ** 2; if (d < best) { best = d; hit = n; } });
      const got = nearestNode(w, x, y, set);
      if ((got?.id ?? null) !== (hit?.id ?? null)) return `the nearest ${set} node to (${x.toFixed(0)}, ${y.toFixed(0)}) is ${got?.id} from the grid against ${hit?.id} from a scan`;
    }
  nearChecks += pts.length * 2;
  return null;
}
let nearChecks = 0;

let crossCleared = 0, crossJunctions = 0, failures = 0, reads = 0, rowsKept = 0, rowsTotal = 0, phasesSame = 0;
let focusReads = 0, focusDeferred = 0, workDone = 0, workWhole = 0, leavesTaken = 0, msBounded = 0, msFull = 0, leavesEnd = 0;
const pick2 = (r, a) => a[Math.floor(r() * a.length)];
const applied = {}, refused = {};

/** a model with the save loaded, and the random edits to make to it (the same seed gives the same edits to the same model) */
function makeModel(seed) {
  const rnd = rng(seed * 104729);
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const api = new TrackApi(SETUP === 'japan' ? 139.7 : -73.985, SETUP === 'japan' ? 35.7 : 40.758, true);
  if (!api.load(saveBytes)) throw new Error('save did not load');
  api.set_unlimited(true);
  // T-1540: water under the network, so that the cost sums ask a mask. Stripes of water a few cells
  // wide over the nodes' extent; `water(k)` adds one shifted by k (a second mask, as zones load one by one)
  const ext = (() => { const ni = api.nodes_info(); const b = [Infinity, Infinity, -Infinity, -Infinity]; for (let i = 0; i < ni.length; i += 10) { b[0] = Math.min(b[0], ni[i + 1]); b[1] = Math.min(b[1], ni[i + 2]); b[2] = Math.max(b[2], ni[i + 1]); b[3] = Math.max(b[3], ni[i + 2]); } return b; })();
  const water = (k) => {
    // (a cell that keeps the mask a few thousand cells across on a save as wide as the Asia one)
    const cell = Math.max(150, Math.ceil(Math.max(ext[2] - ext[0], ext[3] - ext[1]) / 3000)), w = Math.min(60000, Math.ceil((ext[2] - ext[0]) / cell) + 4), h = Math.min(60000, Math.ceil((ext[3] - ext[1]) / cell) + 4);
    const row = new Uint32Array(h + 1), xs = [];
    for (let r = 0; r < h; r++) {
      row[r] = xs.length;
      for (let c0 = ((r * 3 + k) % 9) - 9; c0 < w; c0 += 9) { const a = Math.max(0, c0), b = Math.min(w, c0 + 3 + (k % 2)); if (b > a) xs.push(a, b); }
    }
    row[h] = xs.length;
    api.append_water(cell, ext[0] - 2 * cell, ext[1] - 2 * cell, w, h, row, Uint16Array.from(xs), 0, 0, 1, 1);
  };
  water(0);

  const model = () => {
    const ni = api.nodes_info(), ei = api.edges_info(), li = api.lines_info();
    const nodes = [], stations = [], edges = [], lines = [];
    for (let i = 0; i < ni.length; i += 10) { nodes.push({ id: ni[i], x: ni[i + 1], y: ni[i + 2], platform: ni[i + 7], built: ni[i + 8] === 1, ports: ni[i + 4], builtPorts: ni[i + 5] }); if (ni[i + 7] > 0) stations.push(nodes[nodes.length - 1]); }
    for (let i = 0; i < ei.length; i += 10) edges.push({ id: ei[i], a: ei[i + 1], b: ei[i + 2], tracks: ei[i + 3], built: ei[i + 4] === 1 });
    for (let i = 0; i < li.length; i += 22) lines.push({ id: li[i], running: li[i + 3] === 1, ok: li[i + 2] === 1, bus: api.is_bus(li[i]) });
    return { nodes, stations, edges, lines };
  };
  const free = (x, y) => [0, 0, x, y, 0];
  const nearby = (m) => { const n = pick(m.nodes); const ang = rnd() * 2 * Math.PI, d = 150 + rnd() * 2500; return [n.x + Math.cos(ang) * d, n.y + Math.sin(ang) * d]; };
  const names = ['Alder', 'Birch', 'Cedar', 'Alder', 'Elm', ''];

  const ops = {
    routeFree: (m) => { const [x, y] = nearby(m), [x2, y2] = nearby(m); return api.add_route(new Float64Array([...free(x, y), ...free(x2, y2)]), new Float64Array(), rnd() < 0.3 ? 1 : 2, false); },
    routeFromNode: (m) => { const n = pick(m.nodes); const [x, y] = [n.x + (rnd() - 0.5) * 3000, n.y + (rnd() - 0.5) * 3000]; return api.add_route(new Float64Array([1, n.id, 0, 0, 0, ...free(x, y)]), new Float64Array(), 2, false); },
    stationAtNode: (m) => api.add_station(1, pick(m.nodes.filter((n) => n.ports === 2 && !n.platform).concat(m.nodes.slice(0, 1))).id, 0, 0, pick([120, 200, 300]), pick(names)),
    stationOnEdge: (m) => { const e = pick(m.edges); const a = m.nodes.find((n) => n.id === e.a), b = m.nodes.find((n) => n.id === e.b); return api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, pick(names)); },
    removeStation: (m) => api.remove_station(pick(m.stations).id),
    rename: (m) => { const s = pick(m.stations); return api.set_station_props(s.id, s.platform, pick(names) + (rnd() < 0.5 ? ' Rd' : '')); },
    stationsFree: () => 0,
    setStops: (m) => {
      const l = pick(m.lines); const stops = [...api.line_stops(l.id)];
      if (stops.length < 3) return -1;
      if (rnd() < 0.5) stops.pop(); else stops.splice(1 + Math.floor(rnd() * (stops.length - 2)), 1);
      return api.set_stops(l.id, Uint32Array.from(stops));
    },
    addLine: (m) => {
      const l = pick(m.lines); const stops = [...api.line_stops(l.id)];
      if (stops.length < 3) return -1;
      const from = Math.floor(rnd() * (stops.length - 2)), to = from + 2 + Math.floor(rnd() * (stops.length - from - 1));
      return api.set_line_with_trains(api.new_line_id(), Uint32Array.from(stops.slice(from, to)), 6, 6, 3, 30, 180, 8, 0x3366cc, 'Test ' + Math.floor(rnd() * 99));
    },
    copyLine: (m) => api.copy_line(pick(m.lines).id, api.new_line_id(), 'Copy ' + Math.floor(rnd() * 99)),
    removeLine: (m) => api.remove_line(pick(m.lines).id),
    trains: (m) => { const l = pick(m.lines); return api.set_train_counts(l.id, 1 + Math.floor(rnd() * 12), 1 + Math.floor(rnd() * 8), 1 + Math.floor(rnd() * 4)); },
    headway: (m) => { const l = pick(m.lines); return rnd() < 0.6 ? api.set_auto_headway(l.id, 300 + Math.floor(rnd() * 600), 600, 900) : api.clear_auto_headway(l.id); },
    lineFast: (m) => api.set_line_fast(pick(m.lines).id, rnd() < 0.5),
    lineCars: (m) => api.set_line_cars(pick(m.lines).id, Math.floor(rnd() * 12)),
    lineLook: (m) => { const l = pick(m.lines); return rnd() < 0.5 ? api.set_line_look(l.id, 'Look ' + Math.floor(rnd() * 99), Math.floor(rnd() * 0xffffff)) : api.set_line_badge(l.id, 'Badge ' + Math.floor(rnd() * 9), Math.floor(rnd() * 0xffffff), 'K'); },
    construct: (m) => { const es = m.edges.filter((e) => !e.built); const st = m.stations.filter((s) => !s.built); const e = es.slice(0, 1 + Math.floor(rnd() * 4)), s = st.slice(0, Math.floor(rnd() * 3)); return api.construct(Uint32Array.from(e.map((x) => x.id)), Uint32Array.from(s.map((x) => x.id))); },
    constructLine: (m) => api.construct_line(pick(m.lines).id),
    constructAll: () => (rnd() < 0.15 ? api.construct_all() : -1),
    blueprintOff: (m) => { const es = m.edges.filter((e) => !e.built); return es.length ? api.set_blueprint_off(Uint32Array.from([pick(es).id]), new Uint32Array(), rnd() < 0.5) : -1; },
    doubleUp: (m) => api.double_up(pick(m.edges).id),
    deleteEdge: (m) => api.delete_edge(pick(m.edges.filter((e) => !e.built).concat(m.edges.slice(0, 1))).id),
    flying: (m) => api.set_flying(pick(m.nodes).id, rnd() < 0.5),
    moveNode: (m) => { const n = pick(m.nodes.filter((x) => x.builtPorts === 0).concat(m.nodes.slice(0, 1))); return api.move_node(n.id, n.x + (rnd() - 0.5) * 120, n.y + (rnd() - 0.5) * 120); },
    link: (m) => { const a = pick(m.stations), b = m.stations.filter((s) => s !== a).sort((p, q) => Math.hypot(p.x - a.x, p.y - a.y) - Math.hypot(q.x - a.x, q.y - a.y))[0] ?? a; return rnd() < 0.7 ? api.link_stations(a.id, b.id) : api.unlink_stations(a.id, b.id); },
    upgrade: (m) => api.set_station_upgrade(Uint32Array.from([pick(m.stations).id]), pick([1, 2, 4]), rnd() < 0.7),
    undo: () => (api.can_undo() ? api.undo() : -1),
    redo: () => api.redo(),
    preview: (m) => { const [x, y] = nearby(m), [x2, y2] = nearby(m); api.preview_route(new Float64Array([...free(x, y), ...free(x2, y2)]), new Float64Array(), 2, false); return 0; },
    busLine: (m) => { const [x, y] = nearby(m); const ang = rnd() * 2 * Math.PI; return api.set_bus_line(api.new_line_id(), Float64Array.from([x, y, 1, x + Math.cos(ang) * 2500, y + Math.sin(ang) * 2500, 1]), 480, 600, 1200, 30, 120, 0x2d7dd2, 'Bus ' + Math.floor(rnd() * 99)); },
    busRoute: (m) => { const l = m.lines.filter((x) => x.bus); if (!l.length) return -1; const [x, y] = nearby(m); return api.set_bus_route(pick(l).id, Float64Array.from([x, y, 1, x + 2200, y - 900, 1])); },
    fleet: () => { api.set_fleet(Math.floor(rnd() * 400)); return 0; },
    // T-4113: what only an edge's own calls answer (its one-way setting, its kind)
    oneWay: (m) => { const es = m.edges.filter((e) => e.tracks === 1); return es.length ? api.set_edge_oneway(pick(es).id, pick([0, 1, 2, 3])) : -1; },
    edgeKind: (m) => { const es = m.edges.filter((e) => !e.built); return es.length ? api.set_edges_kind(Uint32Array.from([pick(es).id, pick(es).id]), pick([1, 2, 3])) : -1; },
    // T-4130: a piece of some train line's track, built or not, deleted (the line stops counting, with its path kept)
    breakLine: (m) => { const ls = m.lines.filter((l) => l.ok && !l.bus && api.line_path(l.id).length); if (!ls.length) return -1; const p = api.line_path(pick(ls).id); return api.delete_edge(p[Math.floor(rnd() * p.length)] >> 1); },
  };
  return { api, model, water, edit: (kind) => ops[kind](model()) };
}

const kinds = ['routeFree', 'routeFromNode', 'stationAtNode', 'stationOnEdge', 'removeStation', 'rename', 'setStops', 'addLine', 'copyLine', 'removeLine', 'trains', 'headway', 'lineFast', 'lineCars', 'lineLook', 'construct', 'constructLine', 'constructAll', 'blueprintOff', 'doubleUp', 'deleteEdge', 'flying', 'moveNode', 'link', 'upgrade', 'undo', 'redo', 'preview', 'busLine', 'busRoute', 'fleet', 'oneWay', 'edgeKind', 'breakLine'];
const weights = { breakLine: 2, oneWay: 2, edgeKind: 2, busLine: 2, busRoute: 2, undo: 3, redo: 2, trains: 3, headway: 2, rename: 3, setStops: 3, routeFree: 2, routeFromNode: 2, stationAtNode: 3, removeStation: 2, construct: 3 };
const kindPool = kinds.flatMap((k) => Array(weights[k] ?? 1).fill(k));

for (let seed = 1; seed <= SEEDS; seed++) {
  const A = makeModel(seed), B = makeModel(seed);
  B.api.set_wet_cache(false);
  const eA = envFor(A.api), eB = envFor(B.api);
  if (SELF === 'touched') A.api.take_touched = () => Uint32Array.of(0);
  if (SELF === 'lines' || SELF === 'phases') A.api.take_changed_lines = () => [];
  if (SELF === 'names') A.api.take_name_changes = () => [];
  if (SELF === 'trials' || SELF === 'phases') A.api.peek_changed_lines = () => [];
  const choose = rng(seed * 7919);
  const rows = new StateRows();
  let before = read(rows, A.api, eA);
  let lastPhases = rows.phases(A.api);
  const d0 = firstDifference(before, read(new StateRows(), B.api, eB));
  if (d0) { console.log(`FAIL  seed ${seed}: the first read differs from a fresh one: ${d0}`); failures++; continue; }

  // T-4130: a junction's crossing cleared from far away (crossDone)
  if (!SELF) { const b = crossDone(seed, rows, A, B, eA, eB, before); if (!b) continue; before = b; }
  for (let step = 0; step < STEPS; step++) {
    const kind = kindPool[Math.floor(choose() * kindPool.length)];
    const m = A.model();
    if (!m.nodes.length || !m.stations.length || !m.edges.length || !m.lines.length) break;
    let r, rB;
    if (step === 14) { A.water(1); B.water(1); }
    try { r = A.edit(kind); rB = B.edit(kind); } catch (e) { console.log(`FAIL  seed ${seed} step ${step} ${kind}: the model threw ${e.message}`); failures++; break; }
    if (r !== rB) { console.log(`FAIL  seed ${seed} step ${step} ${kind}: the two models answered ${r} and ${rB}`); failures++; break; }
    if (process.env.VERBOSE) console.log('seed', seed, 'step', step, kind, r === 0 ? 'applied' : 'refused ' + r);
    (r === 0 ? applied : refused)[kind] = ((r === 0 ? applied : refused)[kind] ?? 0) + 1;
    // T-4113: a rows state's read first now and then (the view about a station), its regions left dirty taken by the next
    if (choose() < 0.3) {
      const st = pick2(choose, m.stations), rect = [st.x - 3000, st.y - 3000, st.x + 3000, st.y + 3000];
      // half of them after an "everything" (as a load's), which a focus does first and leaves the rest for after
      if (choose() < 0.5) rows.regions.markAll();
      read(rows, A.api, eA, { rect });
      if (rows.work.left > 0) focusDeferred++;
      focusReads++;
      read(new StateRows(), B.api, eB);
    }
    const tA0 = performance.now();
    const now = read(rows, A.api, eA);
    const tA1 = performance.now();
    const w = rows.work;
    workDone += w.edges + w.nodes + w.lines + w.buses;
    leavesTaken += w.leaves;
    const full = read(new StateRows(), B.api, eB);
    msBounded += tA1 - tA0; msFull += performance.now() - tA1;
    reads++;
    workWhole += full.edges.length + full.nodes.length + full.lines.length;
    // the regions are what a fresh build from the same nodes and edges makes (so an undo gives the same ones back)
    const fresh = new Regions(rows.regions.cap);
    fresh.sync(A.api.nodes_info(), A.api.edges_info());
    if (fresh.shape() !== rows.regions.shape()) { failures++; console.log(`FAIL  seed ${seed} step ${step} ${kind}: the regions differ from a fresh build (${rows.regions.size} leaves against ${fresh.size})`); break; }
    rowsTotal += now.edges.length + now.nodes.length + now.lines.length;
    const old = new Set([...before.edges, ...before.nodes, ...before.lines]);
    for (const k of ['edges', 'nodes', 'lines']) for (const r of now[k]) if (old.has(r)) rowsKept++;
    // the one-pass cost sums are the two passes' to the bit, and the two models agree on them
    const cA = A.api.costs(), cB = B.api.costs();
    // (model A keeps the water under its nodes, B asks the mask every time; total_cost and built_cost
    // ask every time as well, in two passes; the quote is read from both, in step, after the rows)
    const qA = A.api.blueprint_cost_items(), qB = B.api.blueprint_cost_items();
    const dc = !Object.is(cA[0], A.api.total_cost()) || !Object.is(cA[1], A.api.built_cost()) ? `costs() ${cA} against total_cost ${A.api.total_cost()} and built_cost ${A.api.built_cost()}` : !Object.is(cA[0], cB[0]) || !Object.is(cA[1], cB[1]) ? `the models' costs differ: ${cA} against ${cB}` : qA.length !== qB.length || qA.some((v, i) => !Object.is(v, qB[i])) ? `the models' construct-all quotes differ: ${Array.from(qA).slice(0, 14)} against ${Array.from(qB).slice(0, 14)}` : null;
    if (dc) { failures++; console.log(`FAIL  seed ${seed} step ${step} ${kind}: ${dc}`); break; }
    const phasesNow = rows.phases(A.api);
    const dp = phaseDifference(phasesNow, B.api.phases());
    if (phasesNow === lastPhases) phasesSame++;
    lastPhases = phasesNow;
    const d = (SELF === 'phases' ? null : firstDifference(now, full) ?? (JSON.stringify(now) === JSON.stringify(full) ? null : 'serialised rows differ')) ?? (dp ? `the phase tables differ: ${dp}` : null);
    if (d) {
      failures++;
      console.log(`${SELF ? 'detected' : 'FAIL'}  seed ${seed} step ${step} ${kind} (${r === 0 ? 'applied' : 'refused'}): ${d}`);
      break;
    }
    // (its own random points: the edits stay the ones each seed always made)
    const dn = SELF ? null : await nearDifference(now, A.api, rng(seed * 7907 + step));
    if (dn) { failures++; console.log(`FAIL  seed ${seed} step ${step} ${kind}: ${dn}`); break; }
    before = now;
  }
  leavesEnd = rows.regions.size;
}

/** T-4130's case, for the seed's models: the last read, or null on a failure. A built junction's crossing taken away
 * far from it: a line through it is broken by deleting its track farthest from the junction, so no edit touches the
 * junction or its region and only the line's no longer counting (its path kept) clears the crossing. An attempt that
 * reroutes the line, or leaves the crossing, is undone and the next junction tried. */
function crossDone(seed, rows, A, B, eA, eB, before) {
  // (a read with no edit first: the one after a load carries every line's track into the next)
  let last = read(rows, A.api, eA);
  const at = new Map(before.nodes.map((n) => [n.id, n]));
  const ends = new Map(before.edges.map((e) => [e.id, e]));
  const lines = before.lines.filter((l) => l.ok && !A.api.is_bus(l.id) && !A.api.is_ferry(l.id)).map((l) => ({ id: l.id, edges: [...A.api.line_path(l.id)].map((v) => v >> 1) }));
  const through = (n) => lines.filter((l) => l.edges.some((e) => ends.get(e)?.a === n || ends.get(e)?.b === n));
  // (a blueprint junction shows its crossing with no line); the fewest lines first
  const cands = before.nodes.filter((n) => n.crosses && n.builtPorts >= 3).map((n) => ({ n, ls: through(n.id) })).filter((c) => c.ls.length).sort((p, q) => p.ls.length - q.ls.length || p.n.id - q.n.id);
  crossJunctions += cands.length ? 1 : 0;
  const compare = (what) => {
    const now = read(rows, A.api, eA), full = read(new StateRows(), B.api, eB);
    reads++;
    const d = firstDifference(now, full);
    if (d) { console.log(`FAIL  seed ${seed} crossing case, ${what}: ${d}`); failures++; return null; }
    last = now;
    return full;
  };
  for (const { n, ls } of cands.slice(0, 8)) {
    const l = ls[(seed - 1) % ls.length];
    const far = (e) => { const r = ends.get(e), a = at.get(r.a), b = at.get(r.b); return Math.hypot((a.x + b.x) / 2 - n.x, (a.y + b.y) / 2 - n.y); };
    // (track with no station at either end: a station's would be trimmed and the path move with it)
    const plain = l.edges.filter((x) => { const r = ends.get(x); return at.get(r.a).platform === 0 && at.get(r.b).platform === 0; });
    if (!plain.length) continue;
    const e = plain.reduce((p, q) => (far(q) > far(p) ? q : p));
    const r = A.api.delete_edge(e), rB = B.api.delete_edge(e);
    if (r !== rB) { console.log(`FAIL  seed ${seed} crossing case: the two models answered ${r} and ${rB}`); failures++; return null; }
    if (r !== 0) continue;
    const full = compare(`junction ${n.id}, line ${l.id} broken at edge ${e} ${far(e).toFixed(0)} m away`);
    if (!full) return null;
    const broken = !full.lines.find((x) => x.id === l.id)?.ok, cleared = !full.nodes.find((x) => x.id === n.id)?.crosses;
    if (process.env.VERBOSE) console.log('crossing case: junction', n.id, 'of', cands.length, 'line', l.id, 'edge', e, Math.round(far(e)), 'm away; line broken', broken, 'crossing cleared', cleared);
    if (broken && cleared) { crossCleared++; return last; }
    A.api.undo(); B.api.undo();
    if (!compare('undone')) return null;
  }
  return last;
}

const total = (o) => Object.values(o).reduce((a, b) => a + b, 0);
console.log(`${reads} reads over ${SEEDS} seeds; edits applied ${total(applied)} (${Object.entries(applied).map(([k, v]) => `${k} ${v}`).join(', ')}); refused ${total(refused)}`);
if (SELF) {
  console.log(`${failures ? 'PASS' : 'FAIL'}  t1382 rows self-test (${SELF}): ${failures ? 'the check fails when the list of changes is held back' : 'the check did not notice the held-back list'}`);
  process.exit(failures ? 0 : 1);
}
const keptShare = rowsKept / Math.max(1, rowsTotal);
// (the crossing case must have happened where the save has a built junction with a crossing: Asia's has, New York's none)
const bad = failures || keptShare < 0.9 || phasesSame === 0 || (crossJunctions > 0 && crossCleared === 0);
console.log(`T-4113 work: ${workDone} records read again of ${workWhole} (${(100 * workDone / Math.max(1, workWhole)).toFixed(1)}%), regions of at most ${StateRows.regionCap} stations (${leavesEnd} at the end, ${(leavesTaken / Math.max(1, reads)).toFixed(1)} read again per read); a bounded read ${(msBounded / Math.max(1, reads)).toFixed(1)} ms against a fresh full one's ${(msFull / Math.max(1, reads)).toFixed(1)} ms (the check's env); ${focusReads} focus reads, ${focusDeferred} of them left regions for the next read`);
console.log(`${bad ? 'FAIL' : 'PASS'}  t1382 rows (${SETUP}): ${reads} incremental reads equal a full read; ${crossCleared} of ${SEEDS} seeds cleared a junction's crossing from far away${crossJunctions ? '' : ' (the save has no built junction with a crossing)'}; ${(100 * keptShare).toFixed(1)}% of the rows were the kept ones (the cache is used); the phase tables equal the whole ones and were the kept array in ${phasesSame} of ${reads}; ${nearChecks} nearest nodes from the page's grids equal a scan's`);
process.exit(bad ? 1 : 0);
