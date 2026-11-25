// T-4153: the drawn network built from the clock worker's cache (map/network.ts `networkSteps`, the bands' kept orders and
// lines, the station buildings' kept layouts) against one built from scratch, after every edit of a random series on a
// real save: the worker's own code (`stateSteps` in workers/clockMain.ts) run in Node on the track model (WASM), with
// `?patch=check` on, which builds every network a second time from a new cache and compares the two byte for byte
// (T-1029, `checkNet`), and works the bands out whole beside the kept ones (T-2238, `setBandCheck`). Between full
// states: builds dropped part way (an edit came between their pieces, as in the worker), rows states, undo and redo
// chains, and now and then a move of the draw anchor. Any check's error fails the run.
//   node trainworld/checks/t4153-net.mjs [seeds] [steps] [asia|nyc]   (from the maps root; 2 seeds of 12 steps on asia,
//     about 10 min; nyc about 2 min)
//   --self-test (asia: New York's real save has no track two lines share): the kept band orders are trusted however
//     their lines move, and a copy of the line sharing the most track must fail the check
//   --self-test=strokes (asia): a line's strokes are copied entry by entry whatever their band's knots, and a station
//     on shared track in the middle of the longest line (which moves the knots after it) must fail the check
//   --self-test=splice (asia): a line worked out again sweeps only its entries that changed, not those beside them its
//     change reaches, and the same station must fail the check
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { loadApp, REAL_SAVE, ARTIFACTS } from './lib.mjs';

// (what a post transfers is detached, as in the worker: an array the worker keeps and posts by mistake shows up, T-4187)
globalThis.postMessage = (m, o) => { for (const b of o?.transfer ?? []) if (b instanceof ArrayBuffer && !b.detached && b.byteLength) b.transfer(); };
globalThis.self ??= globalThis;
globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const app = await loadApp("export * from '@app/src/workers/clockMain.ts'; export { putApi, syncFrame, frame } from '@app/src/workers/clockState.ts'; export { putPatchMode, posted, sender } from '@app/src/workers/clockPost.ts'; export { setAnchor, makeAnchor, anchor } from '@app/src/map/geo.ts'; export { setBandCheck, bandSelfTest, spliceSelfTest, bandWork } from '@app/src/map/lineBands.ts'; export { netWork, netSelfTest } from '@app/src/map/network.ts'; export { default as initSim, TrackApi } from '@app/src/wasm/track/sim.js';");
await app.initSim({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });

const SELF = process.argv.includes('--self-test') ? 'orders' : process.argv.includes('--self-test=strokes') ? 'strokes' : process.argv.includes('--self-test=splice') ? 'splice' : '';
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const SEEDS = Number(args[0] ?? 2), STEPS = Number(args[1] ?? 12), SAVE = args[2] ?? 'asia';
const raw = gunzipSync(readFileSync(SAVE === 'asia' ? ARTIFACTS + 't3600-asianita-1852.save' : REAL_SAVE));
const saveBytes = raw.subarray(8 + raw.readUInt32LE(4));
const errors = [];
console.error = (...a) => errors.push(a.join(' '));
app.putPatchMode('check');
app.setBandCheck(true);
if (SELF === 'orders') app.bandSelfTest(true);
if (SELF === 'strokes') app.netSelfTest(true);
if (SELF === 'splice') app.spliceSelfTest(true);

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

function load() {
  const api = new app.TrackApi(SAVE === 'asia' ? 139.7 : -73.985, SAVE === 'asia' ? 35.7 : 40.758, true);
  if (!api.load(saveBytes)) throw new Error('save did not load');
  api.set_unlimited(true);
  app.putApi(api);
  app.syncFrame();
  app.forget(true);
  const ni = api.nodes_info(), xs = [], ys = [];
  for (let i = 0; i < ni.length; i += 10) { xs.push(ni[i + 1]); ys.push(ni[i + 2]); }
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
  const [lng, lat] = app.frame.toLonLat(xs[xs.length >> 1], ys[ys.length >> 1]);
  app.setAnchor(app.makeAnchor(Math.round(lng), Math.round(lat), app.anchor().version + 1));
  return api;
}
/** the worker's anchor message in check mode (clock.worker.ts "anchor"): laid exactly again at the new anchor */
function moveAnchor(dLng, dLat) {
  const a = app.anchor();
  app.setAnchor(app.makeAnchor(a.lng + dLng, a.lat + dLat, a.version + 1));
  app.putShiftFrom(null);
  app.putShiftWas(new Map());
  app.edgeDrawn.clear();
  app.lineDrawn.clear();
  app.posted.net = null;
  app.sender.forget();
}
/** a full state's pieces; `stopAt`: dropped after that many, as when an edit comes between them */
function state(stopAt = Infinity) {
  const g = app.stateSteps(false, undefined, false);
  for (let k = 0; ; k++) {
    if (k >= stopAt) return false;
    if (g.next().done) return true;
  }
}

function ops(api, rnd) {
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const model = () => {
    const ni = api.nodes_info(), ei = api.edges_info(), li = api.lines_info(), nodes = [], edges = [], lines = [];
    for (let i = 0; i < ni.length; i += 10) nodes.push({ id: ni[i], x: ni[i + 1], y: ni[i + 2], ports: ni[i + 4], platform: ni[i + 7] });
    for (let i = 0; i < ei.length; i += 10) edges.push({ id: ei[i], a: ei[i + 1], b: ei[i + 2], tracks: ei[i + 3], built: ei[i + 4] === 1 });
    for (let i = 0; i < li.length; i += 22) lines.push({ id: li[i], ok: li[i + 2] === 1, bus: api.is_bus(li[i]) });
    return { nodes, edges, lines, stations: nodes.filter((n) => n.platform > 0), rail: lines.filter((l) => l.ok && !l.bus) };
  };
  const free = (x, y) => [0, 0, x, y, 0];
  const near = (m) => { const n = pick(m.nodes), t = rnd() * 2 * Math.PI, d = 150 + rnd() * 2500; return [n.x + Math.cos(t) * d, n.y + Math.sin(t) * d]; };
  return {
    routeFree: (m) => { const [x, y] = near(m), [x2, y2] = near(m); return api.add_route(new Float64Array([...free(x, y), ...free(x2, y2)]), new Float64Array(), rnd() < 0.3 ? 1 : 2, false); },
    routeFromNode: (m) => { const n = pick(m.nodes); return api.add_route(new Float64Array([1, n.id, 0, 0, 0, ...free(n.x + (rnd() - 0.5) * 3000, n.y + (rnd() - 0.5) * 3000)]), new Float64Array(), 2, false); },
    stationAtNode: (m) => { const c = m.nodes.filter((n) => n.ports === 2 && !n.platform); return c.length ? api.add_station(1, pick(c).id, 0, 0, pick([120, 200, 300]), 'Alder') : -1; },
    stationOnLine: (m) => { const l = pick(m.rail); if (!l) return -1; const p = api.line_path(l.id); if (!p.length) return -1; const e = m.edges.find((x) => x.id === p[Math.floor(rnd() * p.length)] >> 1); if (!e) return -1; const a = m.nodes.find((n) => n.id === e.a), b = m.nodes.find((n) => n.id === e.b); return api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, 'Birch'); },
    removeStation: (m) => api.remove_station(pick(m.stations).id),
    setStops: (m) => { const l = pick(m.rail); if (!l) return -1; const s = [...api.line_stops(l.id)]; if (s.length < 3) return -1; s.splice(1 + Math.floor(rnd() * (s.length - 2)), 1); return api.set_stops(l.id, Uint32Array.from(s)); },
    reorder: (m) => { const l = pick(m.rail); if (!l) return -1; const s = [...api.line_stops(l.id)]; if (s.length < 4) return -1; const k = 1 + Math.floor(rnd() * (s.length - 3)); [s[k], s[k + 1]] = [s[k + 1], s[k]]; return api.set_stops(l.id, Uint32Array.from(s)); },
    copyLine: (m) => { const l = pick(m.rail); return l ? api.copy_line(l.id, api.new_line_id(), 'Copy') : -1; },
    removeLine: (m) => { const l = pick(m.rail); return l ? api.remove_line(l.id) : -1; },
    trains: (m) => { const l = pick(m.lines); return api.set_train_counts(l.id, 1 + Math.floor(rnd() * 12), 1 + Math.floor(rnd() * 8), 1 + Math.floor(rnd() * 4)); },
    construct: (m) => { const es = m.edges.filter((e) => !e.built).slice(0, 4); return es.length ? api.construct(Uint32Array.from(es.map((x) => x.id)), new Uint32Array()) : -1; },
    constructLine: (m) => { const l = pick(m.rail); return l ? api.construct_line(l.id) : -1; },
    doubleUp: (m) => api.double_up(pick(m.edges).id),
    deleteEdge: (m) => { const es = m.edges.filter((e) => !e.built); return api.delete_edge(pick(es.length ? es : m.edges).id); },
    breakLine: (m) => { const l = pick(m.rail); if (!l) return -1; const p = api.line_path(l.id); return p.length ? api.delete_edge(p[Math.floor(rnd() * p.length)] >> 1) : -1; },
    moveNode: (m) => { const n = pick(m.nodes); return api.move_node(n.id, n.x + (rnd() - 0.5) * 120, n.y + (rnd() - 0.5) * 120); },
    link: (m) => { const a = pick(m.stations), b = m.stations.filter((s) => s !== a).sort((p, q) => Math.hypot(p.x - a.x, p.y - a.y) - Math.hypot(q.x - a.x, q.y - a.y))[0] ?? a; return rnd() < 0.7 ? api.link_stations(a.id, b.id) : api.unlink_stations(a.id, b.id); },
    upgrade: (m) => api.set_station_upgrade(Uint32Array.from([pick(m.stations).id]), pick([1, 2, 4]), rnd() < 0.7),
    oneWay: (m) => { const es = m.edges.filter((e) => e.tracks === 1); return es.length ? api.set_edge_oneway(pick(es).id, pick([0, 1, 2, 3])) : -1; },
    // (the edit bench's: a piece of track off a node, then a station on it, which cuts it in two)
    trackStation: (m) => {
      const [x, y] = near(m), t = rnd() * 2 * Math.PI, x2 = x + Math.cos(t) * 1500, y2 = y + Math.sin(t) * 1500;
      if (api.add_route(new Float64Array([...free(x, y), ...free(x2, y2)]), new Float64Array(), rnd() < 0.3 ? 1 : 2, false) !== 0) return -1;
      const m2 = model(), at = new Map(m2.nodes.map((q) => [q.id, q]));
      let best = null, bd = Infinity;
      for (const e of m2.edges) {
        const a = at.get(e.a), b = at.get(e.b);
        if (!a || !b) continue;
        const d = Math.hypot((a.x + b.x) / 2 - (x + x2) / 2, (a.y + b.y) / 2 - (y + y2) / 2);
        if (d < bd) { bd = d; best = { e, a, b }; }
      }
      return best ? api.add_station(2, best.e.id, (best.a.x + best.b.x) / 2, (best.a.y + best.b.y) / 2, 160, 'Cedar') : -1;
    },
    // (T-4153 stage 4: a station on track a long line shares, partway along it, so the strokes before it are copied)
    trunkStation: (m) => {
      const count = new Map();
      for (const l of m.rail) for (const p of new Set(Array.from(api.line_path(l.id), (q) => q >> 1))) count.set(p, (count.get(p) ?? 0) + 1);
      const long = [...m.rail].sort((a, b) => api.line_path(b.id).length - api.line_path(a.id).length).slice(0, 5), l = pick(long);
      if (!l) return -1;
      const path = Array.from(api.line_path(l.id)), shared = path.filter((q) => count.get(q >> 1) > 1);
      const e = m.edges.find((x) => x.id === pick(shared.length ? shared : path) >> 1);
      if (!e) return -1;
      const a = m.nodes.find((n) => n.id === e.a), b = m.nodes.find((n) => n.id === e.b);
      return api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, 'Fir');
    },
    // (T-4153 stage 6: a station a few entries from where a long line's track changes how many lines it carries, inside
    // the curves its bands ease over there; and one on a lone shared edge between track the line has alone, a short run)
    nearChange: (m) => {
      const count = new Map();
      for (const l of m.rail) for (const p of new Set(Array.from(api.line_path(l.id), (q) => q >> 1))) count.set(p, (count.get(p) ?? 0) + 1);
      const l = pick([...m.rail].sort((a, b) => api.line_path(b.id).length - api.line_path(a.id).length).slice(0, 8));
      if (!l) return -1;
      const path = Array.from(api.line_path(l.id), (q) => q >> 1), at = [];
      for (let k = 1; k < path.length; k++) if (count.get(path[k]) !== count.get(path[k - 1])) at.push(k);
      if (!at.length) return -1;
      for (let t = 0; t < 20; t++) {
        const k = Math.max(0, Math.min(path.length - 1, pick(at) + Math.floor(rnd() * 5) - 2)), e = m.edges.find((x) => x.id === path[k]);
        if (!e) continue;
        const a = m.nodes.find((n) => n.id === e.a), b = m.nodes.find((n) => n.id === e.b);
        if (api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, 'Gum') === 0) return 0;
      }
      return -1;
    },
    shortShared: (m) => {
      const count = new Map();
      for (const l of m.rail) for (const p of new Set(Array.from(api.line_path(l.id), (q) => q >> 1))) count.set(p, (count.get(p) ?? 0) + 1);
      const c = [];
      for (const l of m.rail) { const path = Array.from(api.line_path(l.id), (q) => q >> 1); for (let k = 1; k + 1 < path.length; k++) if (count.get(path[k]) > 1 && count.get(path[k - 1]) === 1 && count.get(path[k + 1]) === 1) c.push(path[k]); }
      for (let t = 0; t < 20 && c.length; t++) {
        const e = m.edges.find((x) => x.id === pick(c));
        if (!e) continue;
        const a = m.nodes.find((n) => n.id === e.a), b = m.nodes.find((n) => n.id === e.b);
        if (api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, 'Hazel') === 0) return 0;
      }
      return -1;
    },
    undo: () => (api.can_undo() ? api.undo() : -1),
    redo: () => api.redo(),
    model,
  };
}
const weights = { nearChange: 3, shortShared: 2, trunkStation: 3, stationAtNode: 3, stationOnLine: 3, removeStation: 3, copyLine: 3, removeLine: 2, setStops: 2, undo: 4, redo: 2, trackStation: 3, routeFromNode: 2, deleteEdge: 2, doubleUp: 2, moveNode: 2, construct: 2, oneWay: 2 };

let builds = 0, dropped = 0, rowsStates = 0, moves = 0, failures = 0;
const applied = {};
for (let seed = 1; seed <= SEEDS && !failures; seed++) {
  const api = load(), rnd = rng(seed * 7919), o = ops(api, rnd);
  const pool = Object.keys(o).filter((k) => k !== 'model').flatMap((k) => Array(weights[k] ?? 1).fill(k));
  state();
  if (SELF === 'strokes' || SELF === 'splice') {
    // (a station on track the longest line shares, in its middle: the knots of its band after it move a little)
    const m = o.model(), count = new Map(), at = new Map(m.nodes.map((q) => [q.id, q]));
    for (const l of m.rail) for (const p of new Set(Array.from(api.line_path(l.id), (q) => q >> 1))) count.set(p, (count.get(p) ?? 0) + 1);
    const best = m.rail.reduce((a, b) => (api.line_path(b.id).length > api.line_path(a.id).length ? b : a)), path = Array.from(api.line_path(best.id));
    const k = path.findIndex((q, i) => i >= path.length / 3 && count.get(q >> 1) > 1), e = m.edges.find((x) => x.id === path[k] >> 1);
    const a = at.get(e.a), b = at.get(e.b);
    api.add_station(2, e.id, (a.x + b.x) / 2, (a.y + b.y) / 2, 160, 'Birch');
    app.forget();
    state();
    if (errors.length) { console.log(`detected  seed ${seed}, a station on edge ${e.id} of line ${best.id}: ${errors[0].slice(0, 300)}`); failures++; break; }
  }
  if (SELF === 'orders') {
    // (a copy of the line sharing the most track with others: every edge it shares gains a line its kept order lacks)
    const m = o.model(), count = new Map();
    for (const l of m.rail) for (const p of new Set(Array.from(api.line_path(l.id), (q) => q >> 1))) count.set(p, (count.get(p) ?? 0) + 1);
    const shared = (l) => Array.from(api.line_path(l.id)).filter((q) => count.get(q >> 1) > 1).length;
    const best = m.rail.reduce((a, b) => (shared(b) > shared(a) ? b : a));
    api.copy_line(best.id, api.new_line_id(), 'Copy');
    app.forget();
    state();
    if (errors.length) { console.log(`detected  seed ${seed}, line ${best.id} copied: ${errors[0].slice(0, 300)}`); failures++; break; }
  }
  for (let step = 0; step < STEPS && !failures; step++) {
    const kind = pool[Math.floor(rnd() * pool.length)];
    let r;
    try { r = o[kind](o.model()); } catch (e) { console.log(`FAIL  seed ${seed} step ${step} ${kind}: the model threw ${e.message}`); failures++; break; }
    app.forget();
    if (r === 0) applied[kind] = (applied[kind] ?? 0) + 1;
    const x = rnd();
    // (a build dropped part way, its edit's follow-up begun when the next edit came: the cache is as the last finished one left it)
    if (x < 0.25) { state(1 + Math.floor(rnd() * 6)); dropped++; continue; }
    if (x < 0.35 && app.sendRows()) rowsStates++;
    if (x > 0.93) { moveAnchor((rnd() - 0.5) * 0.4, (rnd() - 0.5) * 0.4); moves++; }
    state();
    builds++;
    if (errors.length) { console.log(`${SELF ? 'detected' : 'FAIL'}  seed ${seed} step ${step} ${kind} (${r === 0 ? 'applied' : 'refused'}): ${errors.slice(0, 3).join(' | ').slice(0, 600)}`); failures++; }
    else if (process.env.VERBOSE) console.log('seed', seed, 'step', step, kind, r === 0 ? 'applied' : 'refused ' + r);
  }
}
console.log(`${builds} checked builds, ${dropped} dropped part way, ${rowsStates} rows states, ${moves} anchor moves; edits applied: ${Object.entries(applied).map(([k, v]) => k + ' ' + v).join(', ')}; pairs of lines kept on edges where their paths changed: ${app.bandWork.passedAll}; line entries' strokes copied: ${app.netWork.entriesAll}, line samples copied: ${app.netWork.keptAll}; band builds that followed the network's changed edges: ${app.bandWork.followed}; lines worked out again in part: ${app.bandWork.splicedAll} (their entries worked out ${app.bandWork.sweptAll}, copied ${app.bandWork.copiedAll})`);
if (SELF) {
  const what = SELF === 'strokes' ? "lines' entries are copied whatever their knots" : SELF === 'splice' ? 'only the entries that changed are worked out again' : 'the kept orders are trusted';
  console.log(`${failures ? 'PASS' : 'FAIL'}  t4153 net self-test (${SELF}): ${failures ? `the check fails when ${what}` : `the check did not notice when ${what}`}`);
  process.exit(failures ? 0 : 1);
}
console.log(`${failures ? 'FAIL' : 'PASS'}  t4153 net (${SAVE}): every build from the cache equals one from scratch`);
process.exit(failures ? 1 : 0);
