// T-409: the market's pairs split by track network. Runs the real demand worker in Node (as
// t149-worker-check.mjs does) with the market, Boston and Philadelphia open, on three networks: a
// Boston line alone, a Philadelphia line alone, and both (no track between them). With both, the
// two lines must solve as two clusters, and each line's loads, its city's figures and its part of
// the market must equal those of the network holding that line alone, to the bit; the market's
// rail is the two parts' added, its trips counted once. Then an edit to the Boston line, timed to
// the worker's done, split and with the market's zones as one (`oneMarket`, `?perfOff=split`).
// Run from the maps root: node trainworld/checks/t409-worker-check.mjs
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, startWorker, network, workerPack as pack, marketPacks, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t409');
const start = () => startWorker(wrapper);
// T-377: the zones' market packs; T-776: the world market's rows, which the page sends with the first
const ROWS = marketRows();
const market = marketPacks(['bos', 'phl']);
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
  // T-1008: the corridors with rail (row, rail, commute rail), each part's added up
  const railOf = (r) => { const m = new Map(); for (let i = 0; i < r.corridorRail.length; i += 3) m.set(r.corridorRail[i], [r.corridorRail[i + 1], r.corridorRail[i + 2]]); return m; };
  const [rB, rb, rp] = [railOf(B), railOf(b), railOf(p)];
  same([...rB.keys()], [...new Set([...rb.keys(), ...rp.keys()])].sort((x, y) => x - y), `period ${q} corridors with rail`);
  for (const [j, [r, c]] of rB) same([r, c], [(rb.get(j)?.[0] ?? 0) + (rp.get(j)?.[0] ?? 0), (rb.get(j)?.[1] ?? 0) + (rp.get(j)?.[1] ?? 0)], `period ${q} corridor ${j}`);
  same(B.corridorShares, b.corridorShares, `period ${q} corridor shares`);
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
writeFileSync(join(ARTIFACTS, 't409', 't409-worker-check.json'), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
console.log('PASS t409-worker-check');
