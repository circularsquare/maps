// T-409: the market's pairs split by track network. Runs the real demand worker in Node (as
// t149-worker-check.mjs does) with the market, Boston and Philadelphia open, on three networks: a
// Boston line alone, a Philadelphia line alone, and both (no track between them). With both, the
// two lines must solve as two clusters, and each line's loads, its city's figures and its part of
// the market must equal those of the network holding that line alone, to the bit; the market's
// rail is the two parts' added, its trips counted once. Then an edit to the Boston line, timed to
// the worker's done, split and with the market's zones as one (`oneMarket`, `?perfOff=split`).
// Run from the maps root: node trainworld/checks/t409-worker-check.mjs
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { rolldown } from '../app/node_modules/rolldown/dist/index.mjs';
const artifacts = new URL('../../artifacts/trainworld/t409/', import.meta.url);
mkdirSync(artifacts, { recursive: true });
const compiled = new URL('demand-worker.mjs', artifacts), wrapper = new URL('worker-adapter.mjs', artifacts);
const wasm = new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url);
const bundle = await rolldown({ input: fileURLToPath(new URL('../app/src/workers/demand.worker.ts', import.meta.url)), platform: 'browser' });
await bundle.write({ file: fileURLToPath(compiled), format: 'esm' }); await bundle.close();
writeFileSync(wrapper, `import {parentPort} from 'node:worker_threads';
import {readFileSync} from 'node:fs';
globalThis.onmessage=null;
globalThis.location={search:'',hash:'',href:'http://localhost/'};
globalThis.postMessage=(data,options={})=>parentPort.postMessage(data,options.transfer??[]);
globalThis.fetch=async()=>new Response(readFileSync(${JSON.stringify(fileURLToPath(wasm))}),{headers:{'Content-Type':'application/wasm'}});
await import(${JSON.stringify(compiled.href)});
parentPort.on('message',data=>globalThis.onmessage({data}));
parentPort.postMessage({kind:'boot'});
`);
function start() {
  const w = new Worker(wrapper), messages = [], waiters = [];
  w.on('message', (msg) => { messages.push(msg); for (const f of [...waiters]) f(msg); });
  w.on('error', (e) => { console.error(e); process.exit(1); });
  const next = (predicate, from = 0) => new Promise((resolve, reject) => {
    const existing = messages.slice(from).find(predicate); if (existing) return resolve(existing);
    const timeout = setTimeout(() => reject(new Error('worker timed out')), 300000);
    const f = (m) => { if (predicate(m)) { clearTimeout(timeout); waiters.splice(waiters.indexOf(f), 1); resolve(m); } }; waiters.push(f);
  });
  return { w, messages, next };
}
const registry = JSON.parse(readFileSync(new URL('../app/src/game/registry.json', import.meta.url)));
const pack = (id) => ({ id, header: readFileSync(new URL(`../data/packs/${id}.json`, import.meta.url), 'utf8'), bin: Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`, import.meta.url))).buffer });
// T-377: the zones' market packs; T-776: the world market's rows, which the page sends with the first
const ROWS = readFileSync(new URL('../app/src/game/market-rows.json', import.meta.url), 'utf8');
const market = ['bos', 'phl'].map((id) => ({ id, pack: readFileSync(new URL(`../data/packs/${id}.market.json`, import.meta.url), 'utf8') }));
const FRAME = { kind: 'equirect', lon: -73.985, lat: 40.758 }, R = 6371008.8, RAD = Math.PI / 180;
const xyOf = ([lon, lat]) => [R * (lon - FRAME.lon) * RAD * Math.cos(FRAME.lat * RAD), R * (lat - FRAME.lat) * RAD];
const LINES = {
  bos: [[-71.0552, 42.3523], [-71.0752, 42.3474], [-71.1001, 42.3480], [-71.1210, 42.3510]],
  phl: [[-75.1817, 39.9557], [-75.1636, 39.9526], [-75.1530, 39.9510], [-75.1340, 39.9690]],
};
/** a network of the given lines, each 4 stations, 3 minutes a hop, `tph` trains an hour */
function network(ids, tph = { bos: [12, 10, 6], phl: [12, 10, 6] }) {
  const stXY = [], lineN = [], stops = [], times = [], tphs = [], cars = [], hopKm = [];
  for (const id of ids) {
    const s0 = stXY.length / 2, pts = LINES[id].map(xyOf), m = pts.length;
    for (const p of pts) stXY.push(...p);
    lineN.push(m);
    for (let k = 0; k < m; k++) stops.push(s0 + k);
    const t = new Array(6 * m).fill(0);
    for (let lev = 0; lev < 3; lev++) for (let k = 0; k < m - 1; k++) { t[lev * m + k] = 180; t[(3 + lev) * m + k + 1] = 180; }
    times.push(...t);
    tphs.push(...tph[id]);
    cars.push(6);
    const km = pts.slice(1).map((p, k) => Math.hypot(p[0] - pts[k][0], p[1] - pts[k][1]) / 1000);
    hopKm.push(...km, 0, ...km.slice().reverse(), 0);
  }
  return { frame: FRAME, stXY: new Float32Array(stXY), complex: new Uint32Array(stXY.length / 2).map((_, i) => i), lineN: new Uint32Array(lineN), stops: new Uint32Array(stops), times: new Float32Array(times), tph: new Float32Array(tphs), cars: new Float32Array(cars), fare: { base: 2.9, perKm: 0.1 }, hopKm: new Float32Array(hopKm) };
}
const ROUNDS = 3;
const finals = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);
async function worker() {
  const k = start(); await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [pack('bos'), pack('phl')], intercity: market, rows: ROWS, periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
}
async function solve(k, version, net, extra = {}) {
  const from = k.messages.length, t0 = performance.now();
  k.w.postMessage({ kind: 'solve', version, net, rounds: ROUNDS, ...extra });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  const plan = k.messages.slice(from).find((m) => m.kind === 'plan' && m.version === version);
  return { ms: Math.round(performance.now() - t0), results: finals(k.messages.slice(from), version), clusters: done.clusters, plan };
}
const same = (x, y, what) => { assert.equal(x.length, y.length, `${what} length`); for (let i = 0; i < x.length; i++) assert(Object.is(x[i], y[i]), `${what}[${i}] ${x[i]} vs ${y[i]}`); };
const noMs = (v) => Array.from(v).map((x, i) => (i === 4 ? 0 : x));

const out = {};
const run = async (ids) => { const k = await worker(); const r = await solve(k, 1, network(ids)); await k.w.terminate(); return r; };
const [bos, phl] = [await run(['bos']), await run(['phl'])];
const kb = await worker();
const both = await solve(kb, 1, network(['bos', 'phl']));
assert.equal(both.plan.solving.length, 2, `two clusters: ${JSON.stringify(both.plan.solving)}`);
const nb = 2 * 4; // route nodes of a 4-stop line
for (let q = 0; q < 5; q++) {
  const [B, b, p] = [both.results[q], bos.results[q], phl.results[q]];
  for (const f of ['seg', 'board', 'alight', 'loadOfCrush']) {
    same(B[f].slice(0, nb), b[f], `period ${q} Boston line ${f}`);
    same(B[f].slice(nb), p[f], `period ${q} Philadelphia line ${f}`);
  }
  same(noMs(B.citySummary.bos), noMs(b.citySummary.bos), `period ${q} Boston's summary`);
  same(noMs(B.citySummary.phl), noMs(p.citySummary.phl), `period ${q} Philadelphia's summary`);
  const [m, mb, mp] = [B.intercitySummary, b.intercitySummary, p.intercitySummary];
  assert(mb[2] > 0 && mp[2] > 0, `period ${q}: both parts carry longer trips`);
  assert.equal(m[2], mb[2] + mp[2], `period ${q} market rail`);
  assert.equal(m[1], mb[1], `period ${q} market trips once`);
  assert.equal(mb[1], mp[1]);
  same(B.intercityParts, B.intercityParts.map((v, i) => (i % 2 ? b.intercityParts[i] + p.intercityParts[i] : b.intercityParts[i])), `period ${q} market parts`);
  same(B.corridors, B.corridors.map((v, i) => (i % 4 === 3 ? b.corridors[i] + p.corridors[i] : b.corridors[i])), `period ${q} corridors`);
}
out.figures = { periods: 5, lineLoadsCitySummariesMarketParts: 'identical', marketRail: both.results.map((r, q) => [r.intercitySummary[2], bos.results[q].intercitySummary[2], phl.results[q].intercitySummary[2]]) };
// an edit to the Boston line: split, Philadelphia's cluster is kept
const edit = network(['bos', 'phl'], { bos: [14, 10, 6], phl: [12, 10, 6] });
const back = network(['bos', 'phl']);
const timed = async (k, v0, extra) => {
  const runs = [];
  for (let i = 0; i < 2; i++) {
    const r = await solve(k, v0 + i, i % 2 ? back : edit, extra);
    runs.push({ ms: r.ms, clusters: r.clusters, solving: r.plan.solving });
  }
  return runs;
};
out.split = { first: both.ms, edits: await timed(kb, 2, {}) };
await kb.w.terminate();
const ko = await worker();
const one = await solve(ko, 1, network(['bos', 'phl']), { oneMarket: true });
assert.equal(one.plan.solving.length, 1, 'one cluster through the market');
out.oneMarket = { first: one.ms, edits: await timed(ko, 2, { oneMarket: true }), marketRailDiff: one.results.map((r, q) => r.intercitySummary[2] - both.results[q].intercitySummary[2]) };
await ko.w.terminate();
writeFileSync(new URL('t409-worker-check.json', artifacts), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
console.log('PASS t409-worker-check');
