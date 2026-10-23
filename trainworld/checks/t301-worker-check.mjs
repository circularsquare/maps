// T-301: closing and reopening packs in the real demand worker (Node, as t409-worker-check.mjs).
// A worker holding the market, Boston and Philadelphia solves a line in each; then
//   - Philadelphia closed, the Boston line alone: the same figures to the bit as a worker that kept
//     Philadelphia open (unreached, it carries nobody), Philadelphia's summary gone;
//   - Philadelphia reopened, both lines: the same as the first solve, to the bit;
//   - Philadelphia closed while a solve that covers it runs: the solve ends normally, the close
//     follows it; and a close cancelled by a reopen before that solve ends keeps the pack;
//   - the market closed and reopened: no market figures while closed, the first solve's after;
//   - a zone opened bare in the aux worker, closed and reopened: the same figures, and WASM memory
//     no larger than before the close (the freed space is reused).
// Run from the maps root: node trainworld/checks/t301-worker-check.mjs
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { rolldown } from '../app/node_modules/rolldown/dist/index.mjs';
const artifacts = new URL('../../artifacts/trainworld/t301/', import.meta.url);
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
  let nextId = 1;
  const ask = async (q) => { const id = nextId++; w.postMessage({ ...q, id }); return next((m) => m.kind === q.kind && m.id === id); };
  return { w, messages, next, ask };
}
const registry = JSON.parse(readFileSync(new URL('../app/src/game/registry.json', import.meta.url)));
const pack = (id) => ({ id, header: readFileSync(new URL(`../data/packs/${id}.json`, import.meta.url), 'utf8'), bin: Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`, import.meta.url))).buffer });
// T-377: zones' market packs, opened and closed with their city packs as the page does
const market = (ids) => ids.map((id) => ({ id, pack: readFileSync(new URL(`../data/packs/${id}.market.json`, import.meta.url), 'utf8') }));
const FRAME = { kind: 'equirect', lon: -73.985, lat: 40.758 }, R = 6371008.8, RAD = Math.PI / 180;
const xyOf = ([lon, lat]) => [R * (lon - FRAME.lon) * RAD * Math.cos(FRAME.lat * RAD), R * (lat - FRAME.lat) * RAD];
const LINES = {
  bos: [[-71.0552, 42.3523], [-71.0752, 42.3474], [-71.1001, 42.3480], [-71.1210, 42.3510]],
  phl: [[-75.1817, 39.9557], [-75.1636, 39.9526], [-75.1530, 39.9510], [-75.1340, 39.9690]],
};
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
async function worker(init = { packs: [pack('bos'), pack('phl')], intercity: market(['bos', 'phl']), periods: [0, 1, 2, 3, 4] }) {
  const k = start(); await k.next((m) => m.kind === 'boot');
  // T-776: the world market's rows, which the page sends with its first market pack
  k.w.postMessage({ kind: 'init', rows: readFileSync(new URL('../app/src/game/market-rows.json', import.meta.url), 'utf8'), ...init });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
}
async function solve(k, version, net) {
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version, net, rounds: ROUNDS });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  return { results: finals(k.messages.slice(from), version), mem: done.memBytes };
}
const send = async (k, msg, reply) => { const from = k.messages.length; k.w.postMessage(msg); const r = await k.next((m) => m.kind === reply || m.kind === 'failed', from); assert.equal(r.kind, reply, r.error); return r; };
const same = (x, y, what) => { assert.equal(x.length, y.length, `${what} length`); for (let i = 0; i < x.length; i++) assert(Object.is(x[i], y[i]), `${what}[${i}] ${x[i]} vs ${y[i]}`); };
const noMs = (v) => Array.from(v).map((x, i) => (i === 4 ? 0 : x));
/** every figure of two solves' finals, the cities in `ids` */
function sameFinals(A, B, ids, what, market = true) {
  assert.equal(A.length, 5); assert.equal(B.length, 5);
  for (let q = 0; q < 5; q++) {
    const [a, b] = [A[q], B[q]];
    for (const f of ['seg', 'board', 'alight', 'loadOfCrush']) same(a[f], b[f], `${what}: period ${q} ${f}`);
    for (const f of ['seg', 'board', 'alight']) same(a.parts[f], b.parts[f], `${what}: period ${q} parts ${f}`);
    for (const id of ids) { same(noMs(a.citySummary[id]), noMs(b.citySummary[id]), `${what}: period ${q} ${id} summary`); same(a.cityParts[id], b.cityParts[id], `${what}: period ${q} ${id} parts`); }
    if (market) for (const f of ['intercitySummary', 'intercityParts', 'corridorShares', 'corridorRail']) same(f === 'intercitySummary' ? noMs(a[f]) : a[f], f === 'intercitySummary' ? noMs(b[f]) : b[f], `${what}: period ${q} ${f}`);
  }
}
const MiB = (b) => Math.round(b / 1048576);
const out = { mem: {} };

// reference: Philadelphia kept open while only the Boston line is built
const kr = await worker();
const refBos = await solve(kr, 1, network(['bos']));
await kr.w.terminate();

const k = await worker();
const both = await solve(k, 1, network(['bos', 'phl']));
out.mem.opened = MiB(both.mem);
await send(k, { kind: 'close', ids: ['phl'], markets: ['phl'] }, 'closed');
const bosOnly = await solve(k, 2, network(['bos']));
sameFinals(bosOnly.results, refBos.results, ['bos'], 'phl closed, bos line');
assert(bosOnly.results.every((r) => !r.citySummary.phl), 'a closed pack has no summary');
await send(k, { kind: 'open', packs: [pack('phl')], intercity: market(['phl']) }, 'opened');
const again = await solve(k, 3, network(['bos', 'phl']));
sameFinals(again.results, both.results, ['bos', 'phl'], 'phl reopened');
out.mem.reopened = MiB(again.mem);

// a close while a solve covering the pack runs: deferred to its end
{
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version: 4, net: network(['bos', 'phl'], { bos: [14, 10, 6], phl: [12, 10, 6] }), rounds: ROUNDS });
  await k.next((m) => m.kind === 'plan' && m.version === 4, from);
  k.w.postMessage({ kind: 'close', ids: ['phl'], markets: ['phl'] });
  const done = await k.next((m) => (m.kind === 'done' && m.version === 4) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  assert(finals(k.messages.slice(from), 4).every((r) => r.citySummary.phl), 'the solve in hand still covers it');
  const r = await solve(k, 5, network(['bos']));
  sameFinals(r.results, refBos.results, ['bos'], 'phl closed during a solve');
  assert(r.results.every((x) => !x.citySummary.phl), 'closed once the solve ended');
}
// a close cancelled by a reopen before the solve in hand ends
await send(k, { kind: 'open', packs: [pack('phl')], intercity: market(['phl']) }, 'opened');
await solve(k, 6, network(['bos', 'phl']));
{
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version: 7, net: network(['bos', 'phl'], { bos: [14, 10, 6], phl: [12, 10, 6] }), rounds: ROUNDS });
  await k.next((m) => m.kind === 'plan' && m.version === 7, from);
  k.w.postMessage({ kind: 'close', ids: ['phl'], markets: ['phl'] });
  k.w.postMessage({ kind: 'open', packs: [pack('phl')], intercity: market(['phl']) });
  const done = await k.next((m) => (m.kind === 'done' && m.version === 7) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  const r = await solve(k, 8, network(['bos', 'phl']));
  sameFinals(r.results, both.results, ['bos', 'phl'], 'close cancelled by a reopen');
}
// the market
await send(k, { kind: 'close', ids: [], markets: ['bos', 'phl'] }, 'closed');
const noMarket = await solve(k, 9, network(['bos', 'phl']));
assert(noMarket.results.every((r) => r.intercitySummary.length === 0), 'no market figures while closed');
await send(k, { kind: 'open', packs: [], intercity: market(['bos', 'phl']) }, 'opened');
const marketBack = await solve(k, 10, network(['bos', 'phl']));
sameFinals(marketBack.results, both.results, ['bos', 'phl'], 'market reopened');
out.mem.end = MiB(marketBack.mem);
await k.w.terminate();

// a bare zone in the aux worker
const aux = await worker({ packs: [], periods: [], aux: true, groups: [[0, 1, 2, 3, 4]] });
const b1 = await send(aux, { kind: 'open', packs: [pack('phl')], bare: true }, 'opened');
const before = await aux.ask({ kind: 'bare', city: 'phl', version: 0, round: 0, day: 0, purpose: -1 });
const c1 = await send(aux, { kind: 'close', ids: ['phl'], bare: true }, 'closed');
assert.equal((await aux.ask({ kind: 'bare', city: 'phl', version: 0, round: 0, day: 0, purpose: -1 })).commute, null, 'closed bare: no answer');
const b2 = await send(aux, { kind: 'open', packs: [pack('phl')], bare: true }, 'opened');
const after = await aux.ask({ kind: 'bare', city: 'phl', version: 0, round: 0, day: 0, purpose: -1 });
assert.deepEqual(b2.bare.phl.summary.map(noMs), b1.bare.phl.summary.map(noMs), 'bare summaries');
assert.deepEqual(b2.bare.phl.parts, b1.bare.phl.parts, 'bare parts');
same(after.commute, before.commute, 'bare commuters by mode');
after.parts.forEach((p, i) => same(p, before.parts[i], `bare everyday legs ${i}`));
// open, close, open of another zone: the second fits in the space the first freed
const b3 = await send(aux, { kind: 'open', packs: [pack('bos')], bare: true }, 'opened');
await send(aux, { kind: 'close', ids: ['bos', 'phl'], bare: true }, 'closed');
const b4 = await send(aux, { kind: 'open', packs: [pack('bos')], bare: true }, 'opened');
out.aux = { phlOpened: MiB(b1.memBytes), phlClosed: MiB(c1.memBytes), phlReopened: MiB(b2.memBytes), plusBos: MiB(b3.memBytes), bosAloneAfterClosingBoth: MiB(b4.memBytes), reopenMs: Math.round(b2.openMs) };
assert(b2.memBytes <= b1.memBytes, `reopening reuses the freed space: ${b2.memBytes} > ${b1.memBytes}`);
assert(b4.memBytes <= b3.memBytes, 'reopening after closing both grows nothing');
await aux.w.terminate();
writeFileSync(new URL('t301-worker-check.json', artifacts), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
console.log('PASS t301-worker-check');
