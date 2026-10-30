// T-1862: a cluster whose lines moved only by settle noise keeps its result. Runs the real demand
// worker in Node (as t409-worker-check.mjs) with Boston, Philadelphia and their market open, a test
// line in each (two clusters). Boston's run times and trains an hour are moved by noise while
// Philadelphia is edited: Boston must be kept, its figures the same to the bit as before. Noise that
// adds up past the bound (trains an hour 0.2%, run times 0.5 s, against the values Boston last
// solved on), +1% trains, +1 s run time, a span step, a car more and a moved stop must each re-solve
// Boston, its figures then equal to the bit to a fresh worker's on the same network. A zone opened
// again at another level beside Boston's noise is still solved; `exactKeep` (`?perfOff=noise`)
// re-solves on noise. Run from the maps root: node trainworld/checks/t1862-noise-keep.mjs
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, startWorker, network, workerPack as pack, marketPacks, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t1862');
const ROWS = marketRows(), MARKET = marketPacks(['bos', 'phl']);
const ROUNDS = 3, PERIODS = 5;
const finals = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);
async function worker() {
  const k = startWorker(wrapper); await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [pack('bos'), pack('phl')], intercity: MARKET, rows: ROWS, periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
}
// the page always sends the ids (T-1733), which keep `capRatio` and the market's route nodes across solves
const IDS = { lines: ['bos', 'phl'], stations: Array.from({ length: 8 }, (_, i) => `s${i}`) };
async function solve(k, version, net, extra = {}) {
  const from = k.messages.length, t0 = performance.now();
  k.w.postMessage({ kind: 'solve', version, net, ids: IDS, rounds: ROUNDS, ...extra });
  // (T-1733: an edit's whole solve waits for the page's word that the local results are shown)
  k.w.postMessage({ kind: 'whole', version });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  const plan = k.messages.slice(from).find((m) => m.kind === 'plan' && m.version === version);
  const results = finals(k.messages.slice(from), version);
  assert.equal(results.length, PERIODS, `v${version}: a final per period`);
  return { ms: Math.round(performance.now() - t0), results, clusters: done.clusters, kept: plan.kept, solving: plan.solving.flat() };
}
// Boston's line is the network's first: route nodes 0-7
const NB = 8;
function bosDiff(A, B) {
  const bad = [];
  for (let q = 0; q < PERIODS; q++) {
    for (const f of ['seg', 'board', 'alight']) for (let i = 0; i < NB; i++) if (!Object.is(A[q][f][i], B[q][f][i])) bad.push(`p${q} ${f}[${i}] ${A[q][f][i]} vs ${B[q][f][i]}`);
    const a = Array.from(A[q].citySummary.bos), b = Array.from(B[q].citySummary.bos);
    for (let i = 0; i < Math.max(a.length, b.length); i++) if (i !== 4 && !Object.is(a[i], b[i])) bad.push(`p${q} citySummary.bos[${i}] ${a[i]} vs ${b[i]}`);
  }
  return bad;
}
const riders = (r) => r.results.reduce((t, m) => t + Array.from(m.board.subarray(0, NB)).reduce((a, b) => a + b, 0), 0);

// The network: Boston's line moved by `noise` (run times + s, trains an hour times f) and changed by
// `edit`; Philadelphia's trains an hour `phl`
const BASE = network(['bos', 'phl']);
function net({ dt = 0, f = 1, phl = [12, 10, 6], edit = null } = {}) {
  const n = { ...BASE, stXY: BASE.stXY.slice(), times: BASE.times.slice(), tph: BASE.tph.slice(), cars: BASE.cars.slice() };
  for (let k = 0; k < 6 * 4; k++) if (n.times[k] > 0) n.times[k] += dt;
  for (let k = 0; k < 3; k++) n.tph[k] *= f;
  n.tph.set(phl, 3);
  if (edit) edit(n);
  return n;
}
const EDITS = {
  'trains +1%': (n) => { for (let k = 0; k < 3; k++) n.tph[k] *= 1.01; },
  'run times +1 s': (n) => { for (let k = 0; k < 6 * 4; k++) if (n.times[k] > 0) n.times[k] += 1; },
  // one train less in period 0's day on Boston's route nodes (T-842, T-1764: steps of 0.05)
  'a span step': (n) => { const s = new Float32Array(PERIODS * 2 * 8).fill(1); for (let r = 0; r < NB; r++) s[r] = 0.95; n.span = s; },
  'a car more': (n) => { n.cars[0] += 1; },
  'a stop moved 100 m': (n) => { n.stXY[2] += 100; },
};

const log = [], fail = [];
const check = (ok, what) => { log.push(`${ok ? 'PASS' : 'FAIL'} ${what}`); console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail.push(what); };
const keptBos = (r) => r.kept.includes('bos') && !r.solving.includes('bos');
const fresh = async (n) => { const k = await worker(); const r = await solve(k, 1, n); await k.w.terminate(); return r; };
const vsFresh = async (r, n, what) => {
  const f = await fresh(n), bad = bosDiff(r.results, f.results);
  check(!bad.length, `${what}: Boston's figures equal a fresh worker's to the bit${bad.length ? ` (${bad.length} differ: ${bad.slice(0, 3).join('; ')})` : ''}`);
};

const k = await worker();
let v = 0;
const v1 = await solve(k, ++v, net());
check(!v1.kept.length && v1.solving.includes('bos') && v1.solving.includes('phl'), `first solve: both clusters solved (${v1.ms} ms, Boston ${Math.round(riders(v1))} boardings)`);

// (a) Philadelphia edited, Boston moved by noise: Boston kept, to the bit
const STEP_F = 1.00045, STEP_S = 0.12;
const a = await solve(k, ++v, net({ dt: 0.1, f: STEP_F, phl: [14, 10, 6] }));
check(keptBos(a) && a.solving.includes('phl') && a.clusters[0] === 1, `noise (0.1 s, 0.045%) beside a Philadelphia edit: Boston kept, Philadelphia solved (${a.ms} ms, clusters kept/solved ${JSON.stringify(a.clusters)})`);
const aBad = bosDiff(a.results, v1.results);
check(!aBad.length, `noise: Boston's figures the same to the bit as before${aBad.length ? ` (${aBad.slice(0, 3).join('; ')})` : ''}`);

// (c) trains an hour adding up by 0.045% a step against the values solved on: kept to 0.18%, solved at 0.225%
let last;
for (let s = 2; s <= 5; s++) {
  last = await solve(k, ++v, net({ dt: 0.1, f: STEP_F ** s, phl: [14, 10, 6] }));
  const pct = ((STEP_F ** s - 1) * 100).toFixed(3);
  check(s < 5 ? keptBos(last) : !keptBos(last), `trains an hour ${pct}% off what Boston solved on: ${s < 5 ? 'kept' : 'solved'} (${last.ms} ms)`);
}
await vsFresh(last, net({ dt: 0.1, f: STEP_F ** 5, phl: [14, 10, 6] }), 'trains an hour past 0.2%');
// run times adding up by 0.12 s a step: kept to 0.48 s, solved at 0.6 s
const F5 = STEP_F ** 5;
for (let s = 1; s <= 5; s++) {
  last = await solve(k, ++v, net({ dt: 0.1 + STEP_S * s, f: F5, phl: [14, 10, 6] }));
  check(s < 5 ? keptBos(last) : !keptBos(last), `run times ${(STEP_S * s).toFixed(2)} s off what Boston solved on: ${s < 5 ? 'kept' : 'solved'} (${last.ms} ms)`);
}
const N0 = { dt: 0.1 + STEP_S * 5, f: F5, phl: [14, 10, 6] };
await vsFresh(last, net(N0), 'run times past 0.5 s');

// (b) each real change re-solves Boston, each from the state the one before left
const applied = [];
for (const [what, edit] of Object.entries(EDITS)) {
  applied.push(edit);
  const n = net({ ...N0, edit: (x) => applied.forEach((e) => e(x)) });
  const r = await solve(k, ++v, n);
  check(!keptBos(r), `${what}: Boston solved (${r.ms} ms)`);
  await vsFresh(r, n, what);
}
const NB0 = { ...N0, edit: (x) => applied.forEach((e) => e(x)) };

// (e) `?perfOff=noise`: noise re-solves
const e0 = await solve(k, ++v, net({ ...NB0, dt: N0.dt + 0.05 }), { exactKeep: true });
check(!keptBos(e0), `exactKeep (?perfOff=noise): noise of 0.05 s re-solves Boston (${e0.ms} ms)`);

// (d) Philadelphia opened again at the coarse level beside Boston's noise: Philadelphia solved, Boston kept
const from = k.messages.length;
k.w.postMessage({ kind: 'open', packs: [{ ...pack('phl'), level: 1 }] });
const op = await k.next((m) => m.kind === 'opened' || m.kind === 'failed', from); assert.equal(op.kind, 'opened', op.error);
const d = await solve(k, ++v, net({ ...NB0, dt: N0.dt + 0.1 }));
check(keptBos(d) && d.solving.includes('phl'), `Philadelphia reopened coarse, Boston's noise: Boston kept, Philadelphia solved (${d.ms} ms)`);
await k.w.terminate();

writeFileSync(join(ARTIFACTS, 't1862', 'noise-keep.log'), log.join('\n') + '\n');
console.log(fail.length ? `t1862-noise-keep FAIL (${fail.length})` : 't1862-noise-keep PASS');
process.exit(fail.length ? 1 : 0);
