// T-1850: does opening a far zone's city and market packs dirty a cluster that shares no rider?
// Real demand worker in Node (as t409/t301). Boston solved alone, then Philadelphia's packs opened
// (city first, market after, as the page does), solves with and without the barrier; Boston's
// figures compared to the bit with its first solve and with a fresh worker that opened everything.
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
  if (process.env.EXACT) extra = { ...extra, exactMarket: true };
  const from = k.messages.length, t0 = performance.now();
  k.w.postMessage({ kind: 'solve', version, net, rounds: ROUNDS, ...extra });
  // the barrier's rounds all allowed at once (the page sends `go` per round as it publishes)
  if (extra.barrier) k.w.postMessage({ kind: 'go', version, round: 99, group: 9 });
  const done = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(done.kind, 'done', done.error);
  const plan = k.messages.slice(from).find((m) => m.kind === 'plan' && m.version === version);
  return { ms: Math.round(performance.now() - t0), results: finals(k.messages.slice(from), version), clusters: done.clusters, periods: done.periods, solving: plan.solving, kept: plan.kept };
}
const send = async (k, msg, reply) => { const from = k.messages.length; k.w.postMessage(msg); const r = await k.next((m) => m.kind === reply || m.kind === 'failed', from); assert.equal(r.kind, reply, r.error); return r; };
const nb = 8;
function diff(A, B, what) {
  let n = 0, worst = 0;
  for (let q = 0; q < 5; q++) {
    for (const f of ['seg', 'board', 'alight']) for (let i = 0; i < nb; i++) { const a = A[q][f][i], b = B[q][f][i]; if (!Object.is(a, b)) { n++; worst = Math.max(worst, Math.abs(a - b) / Math.max(1e-9, Math.abs(b))); } }
    const a = Array.from(A[q].citySummary.bos), b = Array.from(B[q].citySummary.bos);
    for (let i = 0; i < a.length; i++) if (i !== 4 && !Object.is(a[i], b[i])) n++;
  }
  console.log(`${what}: ${n} Boston figures differ, worst ${(100 * worst).toFixed(4)}%`);
}
// the market's figures, which add its parts up (some fields taken once, from the first part)
function marketDiff(A, B, what) {
  for (let q = 0; q < 5; q++) {
    const out = [];
    for (const f of ['intercitySummary', 'intercityParts', 'corridorShares', 'corridorRail', 'elsewhere', 'corridorKey']) {
      const a = A[q][f], b = B[q][f];
      if (a === undefined && b === undefined) continue;
      const x = typeof a === 'object' && a ? Array.from(a) : [a], y = typeof b === 'object' && b ? Array.from(b) : [b];
      const bad = [];
      for (let i = 0; i < Math.max(x.length, y.length); i++) if (!(f === 'intercitySummary' && i === 4) && !Object.is(x[i], y[i])) bad.push(`${i}: ${x[i]} vs ${y[i]}`);
      if (bad.length) out.push(`${f} [${bad.slice(0, 6).join('; ')}${bad.length > 6 ? ' ...' + bad.length : ''}]`);
    }
    console.log(`${what} period ${q}: ${out.length ? out.join(' ') : 'market figures identical'}`);
  }
}
const show = (tag, r) => console.log(`${tag}: ${r.ms} ms, clusters kept/solved ${JSON.stringify(r.clusters)}, periods kept/solved ${JSON.stringify(r.periods)}, solving ${JSON.stringify(r.solving)} kept ${JSON.stringify(r.kept)}`);

const k = await worker({ packs: [pack('bos')], intercity: market(['bos']) });
const v1 = await solve(k, 1, network(['bos']));
show('v1 bos alone', v1);
await send(k, { kind: 'open', packs: [pack('phl')] }, 'opened');
const v2 = await solve(k, 2, network(['bos', 'phl']), { barrier: true, first: 'phl' });
show('v2 phl city open, barrier', v2);
await send(k, { kind: 'open', packs: [], intercity: market(['phl']) }, 'opened');
const v3 = await solve(k, 3, network(['bos', 'phl']), { barrier: true, first: 'phl' });
show('v3 phl market open, barrier', v3);
const v4 = await solve(k, 4, network(['bos', 'phl']));
show('v4 same network, no barrier', v4);
diff(v3.results, v1.results, 'v3 against v1');
await k.w.terminate();
const f = await worker({ packs: [pack('bos'), pack('phl')], intercity: market(['bos', 'phl']) });
const ref = await solve(f, 1, network(['bos', 'phl']));
show('fresh worker, both', ref);
diff(v3.results, ref.results, 'v3 (kept) against a fresh worker with both');
diff(v1.results, ref.results, 'v1 (bos alone) against a fresh worker with both');
marketDiff(v3.results, ref.results, 'market v3 (bos part kept from the old composition) against fresh');
marketDiff(v4.results, ref.results, 'market v4 against fresh');
await f.w.terminate();
