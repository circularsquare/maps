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
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, startWorker, network, workerPack as pack, marketPacks as market, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t301');
const start = () => startWorker(wrapper);
const ROUNDS = 3;
const finals = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);
async function worker(init = { packs: [pack('bos'), pack('phl')], intercity: market(['bos', 'phl']), periods: [0, 1, 2, 3, 4] }) {
  const k = start(); await k.next((m) => m.kind === 'boot');
  // T-776: the world market's rows, which the page sends with its first market pack
  k.w.postMessage({ kind: 'init', rows: marketRows(), ...init });
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
writeFileSync(join(ARTIFACTS, 't301', 't301-worker-check.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
console.log('PASS t301-worker-check');
