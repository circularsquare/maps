// T-1850: is the small market difference in kept.mjs the corridor table (T-1354: kept from a
// worker's first composition) rather than the kept part's own composition? A worker that composes
// Boston and Philadelphia first (the fresh worker's corridors), then Boston alone, solves Boston,
// then Philadelphia's market again: Boston's part kept from the Boston-only composition.
import assert from 'node:assert/strict';
import { bundleDemandWorker, startWorker, network, workerPack as pack, marketPacks as market, marketRows } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const wrapper = await bundleDemandWorker('t1850');
const ROUNDS = 3;
const finals = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);
async function worker(init) {
  const k = startWorker(wrapper); await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', rows: marketRows(), periods: [0, 1, 2, 3, 4], ...init });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
}
async function solve(k, version, net, extra = {}) {
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version, net, rounds: ROUNDS, ...extra });
  if (extra.barrier) k.w.postMessage({ kind: 'go', version, round: 99, group: 9 });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  const plan = k.messages.slice(from).find((m) => m.kind === 'plan' && m.version === version);
  return { results: finals(k.messages.slice(from), version), clusters: done.clusters, solving: plan.solving, kept: plan.kept };
}
const send = async (k, msg, reply) => { const from = k.messages.length; k.w.postMessage(msg); const r = await k.next((m) => m.kind === reply || m.kind === 'failed', from); assert.equal(r.kind, reply, r.error); return r; };
function allDiff(A, B, what) {
  let n = 0;
  const fields = ['seg', 'board', 'alight', 'loadOfCrush', 'intercitySummary', 'intercityParts', 'corridorShares', 'corridorRail', 'elsewhere', 'corridorKey'];
  for (let q = 0; q < 5; q++) for (const f of fields) {
    const a = A[q][f], b = B[q][f];
    const x = a && typeof a === 'object' ? Array.from(a) : [a], y = b && typeof b === 'object' ? Array.from(b) : [b];
    for (let i = 0; i < Math.max(x.length, y.length); i++) if (!(f === 'intercitySummary' && i === 4) && !Object.is(x[i], y[i])) { n++; if (q === 0) console.log(`  q0 ${f}[${i}] ${x[i]} vs ${y[i]}`); }
  }
  for (let q = 0; q < 5; q++) for (const id of ['bos', 'phl']) {
    const a = A[q].citySummary[id], b = B[q].citySummary[id];
    if (!a || !b) continue;
    for (let i = 0; i < a.length; i++) if (i !== 4 && !Object.is(a[i], b[i])) n++;
  }
  console.log(`${what}: ${n} figures differ`);
}
const k = await worker({ packs: [pack('bos'), pack('phl')], intercity: market(['bos', 'phl']) });
await send(k, { kind: 'close', ids: [], markets: ['phl'] }, 'closed');
const v1 = await solve(k, 1, network(['bos', 'phl']));
console.log('v1 (market of bos alone, corridors of both)', JSON.stringify(v1.clusters), JSON.stringify(v1.solving));
await send(k, { kind: 'open', packs: [], intercity: market(['phl']) }, 'opened');
const v2 = await solve(k, 2, network(['bos', 'phl']), { barrier: true, first: 'phl' });
console.log('v2 (phl market back)', JSON.stringify(v2.clusters), JSON.stringify(v2.solving), 'kept', JSON.stringify(v2.kept));
await k.w.terminate();
const f = await worker({ packs: [pack('bos'), pack('phl')], intercity: market(['bos', 'phl']) });
const ref = await solve(f, 1, network(['bos', 'phl']));
allDiff(v2.results, ref.results, 'v2 (bos part kept from the bos-only composition) against a fresh worker');
await f.w.terminate();
