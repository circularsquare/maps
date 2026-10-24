// T-149: the demand worker opening packs one by one. Runs the real worker module in Node (as
// t066-worker-check.mjs does) twice: one worker opens every city at init, as before T-149; the
// other starts empty, opens the market and New York, solves, then opens Philadelphia and solves
// again. A network inside New York, with no station within reach of any other pack, must give
// exactly the same line loads, New York and market results either way: leaving a city with no
// construction unloaded changes no number. Also: a city opened after a solve answers no query
// until a solve covers it, and a cells query for a city not open answers empty instead of failing.
// Run from the maps root: node trainworld/checks/t149-worker-check.mjs
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { bundleDemandWorker, startWorker, workerPack as pack, marketPacks as market, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t149');
const start = () => startWorker(wrapper, { timeout: 120000 });
// New York, Philadelphia and Boston stand for the registry: every zone at once is 2.3 GB of packs,
// more than a worker's 4 GB of WASM holds
const ZONES = ['nyc', 'phl', 'bos'];
// T-776: the world market's rows, which the page sends with the first
const ROWS = marketRows();
// a three-station line across Midtown (world frame = the New York pack's origin, equirectangular)
const times = new Float32Array(18); for (let lev = 0; lev < 3; lev++) { times[lev * 3] = 60; times[lev * 3 + 1] = 60; times[(lev + 3) * 3 + 1] = 60; times[(lev + 3) * 3 + 2] = 60; }
const net = { frame: { kind: 'equirect', lon: -73.985, lat: 40.758 }, stXY: new Float32Array([0, 0, 4000, 0, 8000, 0]), lineN: new Uint32Array([3]), stops: new Uint32Array([0, 1, 2]), times, tph: new Float32Array([12, 12, 12]), cars: new Float32Array([4]), fare: { base: 1, perKm: .1 }, hopKm: new Float32Array([4, 4, 0, 0, 4, 4]) };
const ROUNDS = 2;
// the solve's final: cluster results (T-392) are the early figures of single clusters; the last
// result of each period stands for it (the worker sends the final's rounds twice at present)
const results = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);
const t = () => performance.now();

// before T-149: every pack at init
const all = start(); await all.next((m) => m.kind === 'boot');
let a0 = t();
all.w.postMessage({ kind: 'init', packs: ZONES.map(pack), intercity: market(ZONES), rows: ROWS, periods: [0, 1, 2, 3, 4] });
const readyAll = await all.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(readyAll.kind, 'ready', readyAll.error);
const openAllMs = t() - a0;
all.w.postMessage({ kind: 'solve', version: 1, net, rounds: ROUNDS });
await all.next((m) => (m.kind === 'done' && m.version === 1) || m.kind === 'failed');
const ref = results(all.messages, 1); assert.equal(ref.length, 5);
await all.w.terminate();

// T-149: start empty, then the market and New York, solve; then Philadelphia, solve again
const lazy = start(); await lazy.next((m) => m.kind === 'boot');
a0 = t();
lazy.w.postMessage({ kind: 'init', packs: [], rows: ROWS, periods: [0, 1, 2, 3, 4] });
lazy.w.postMessage({ kind: 'open', packs: [], intercity: market(['nyc']) });
lazy.w.postMessage({ kind: 'open', packs: [pack('nyc')] });
const ready = await lazy.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(ready.kind, 'ready', ready.error);
assert.deepEqual(Object.keys(ready.cityInfo), []);
const openedMarket = await lazy.next((m) => m.kind === 'opened' && m.markets.length);
const openedNyc = await lazy.next((m) => m.kind === 'opened' && m.cityInfo.nyc);
const openNycMs = t() - a0;
// cells, zones, trips, walk share (the fifth is the opening's ms)
assert.deepEqual(openedNyc.cityInfo.nyc.slice(0, 4), readyAll.cityInfo.nyc.slice(0, 4), 'New York info is the same opened alone');
lazy.w.postMessage({ kind: 'solve', version: 1, net, rounds: ROUNDS });
await lazy.next((m) => (m.kind === 'done' && m.version === 1) || m.kind === 'failed');
assert(!lazy.messages.some((m) => m.kind === 'failed'), lazy.messages.find((m) => m.kind === 'failed')?.error);
const one = results(lazy.messages, 1); assert.equal(one.length, 5);
const same = (x, y, what) => { assert.equal(x.length, y.length, what); for (let k = 0; k < x.length; k++) assert(Object.is(x[k], y[k]) || Math.abs(x[k] - y[k]) <= 1e-6 * Math.max(1, Math.abs(y[k])), `${what}[${k}] ${x[k]} vs ${y[k]}`); };
let maxDiff = 0;
for (let q = 0; q < 5; q++) {
  for (const f of ['seg', 'board', 'alight', 'loadOfCrush']) { same(one[q][f], ref[q][f], `period ${q} ${f}`); for (let k = 0; k < ref[q][f].length; k++) maxDiff = Math.max(maxDiff, Math.abs(one[q][f][k] - ref[q][f][k])); }
  same(one[q].citySummary.nyc.filter((_, k) => k !== 4), ref[q].citySummary.nyc.filter((_, k) => k !== 4), `period ${q} New York summary`);
  same(one[q].intercitySummary.filter((_, k) => k !== 4), ref[q].intercitySummary.filter((_, k) => k !== 4), `period ${q} market summary`);
  // T-1008: the corridors with rail by their places (each worker's table sent once, on its first result)
  const railByPlaces = (r, msgs) => { const t = msgs.find((m) => m.corridorTable)?.corridorTable, v = r.corridorRail ?? []; const out = []; for (let i = 0; i < v.length; i += 3) out.push(t.from[v[i]], t.to[v[i]], v[i + 1], v[i + 2]); return out; };
  same(railByPlaces(one[q], lazy.messages), railByPlaces(ref[q], all.messages), `period ${q} corridors with rail`);
  same(one[q].corridorShares, ref[q].corridorShares, `period ${q} corridor shares`);
  assert.deepEqual(Object.keys(one[q].citySummary), ['nyc']);
  // the other zones had no rail at all in the full solve
  for (const id of Object.keys(ref[q].citySummary)) if (id !== 'nyc') assert.equal(ref[q].citySummary[id][2], 0, `${id} rail in the full solve`);
}
// cells for a city not open: an empty answer, never a pool failure
lazy.w.postMessage({ kind: 'cells', id: 7, city: 'bos', frame: net.frame });
const cells = await lazy.next((m) => m.kind === 'cells' && m.id === 7); assert.equal(cells.xy.length, 0);
// Philadelphia opens after the solve: it answers nothing until a solve covers it
let from = lazy.messages.length;
lazy.w.postMessage({ kind: 'open', packs: [pack('phl')], intercity: market(['phl']) });
await lazy.next((m) => m.kind === 'opened' && m.cityInfo.phl, from);
lazy.w.postMessage({ kind: 'purposes', id: 8, city: 'phl', version: 1, round: ROUNDS, end: 0, cells: new Uint32Array([0]) });
const before = await lazy.next((m) => m.kind === 'purposes' && m.id === 8); assert.equal(before.work, null);
lazy.w.postMessage({ kind: 'solve', version: 2, net, rounds: ROUNDS });
await lazy.next((m) => (m.kind === 'done' && m.version === 2) || m.kind === 'failed', from);
const two = results(lazy.messages, 2); assert.equal(two.length, 5);
for (let q = 0; q < 5; q++) {
  same(two[q].seg, one[q].seg, `period ${q} seg after Philadelphia opened`);
  same(two[q].citySummary.nyc.filter((_, k) => k !== 4), one[q].citySummary.nyc.filter((_, k) => k !== 4), `period ${q} New York after Philadelphia opened`);
  assert.deepEqual(Object.keys(two[q].citySummary).sort(), ['nyc', 'phl']);
  same(two[q].citySummary.phl.filter((_, k) => k !== 4), ref[q].citySummary.phl.filter((_, k) => k !== 4), `period ${q} Philadelphia as in the full solve`);
}
lazy.w.postMessage({ kind: 'purposes', id: 9, city: 'phl', version: 2, round: ROUNDS, end: 0, cells: new Uint32Array([0]) });
const after = await lazy.next((m) => m.kind === 'purposes' && m.id === 9); assert(after.work?.length === 30);
await lazy.w.terminate();
const result = {
  sameLoadsNycMarketCorridors: true, maxAbsDiffLoads: maxDiff, otherCitiesRailInFullSolve: 0,
  lateCityAnswersOnlyAfterCoveringSolve: true, cellsForUnopenedCityEmpty: true,
  openMs: { everyPackAtInit: Math.round(openAllMs), marketAndNewYorkOnly: Math.round(openNycMs), marketOpenMs: Math.round(openedMarket.openMs), newYorkOpenMs: Math.round(openedNyc.openMs) },
  wasmMemoryMiB: { everyPack: Math.round(ref.at(-1).memBytes / 1048576), newYorkOnly: Math.round(one.at(-1).memBytes / 1048576) },
};
writeFileSync(new URL('../../artifacts/trainworld/notes-outputs/T-149-worker-check.json', import.meta.url), JSON.stringify(result, null, 2)); console.log(result);
