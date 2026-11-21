// T-3925: an edit's solve stopped part way by a newer network, then solved on that newer network, gives the same
// figures to the bit as one solve of it. Runs the real demand worker in Node (as t1862-noise-keep.mjs) with Boston,
// Philadelphia and their market open, a test line in each (two clusters). Network A is solved; edit B (Boston's
// trains) is sent and stopped by C at three points: in B's local stage, at its whole solve's first unit, and halfway
// through its whole solve. C either takes Boston back to A and edits Philadelphia (the cluster B touched now holds
// neither A's solve nor B's), or edits Boston further. Each C's final results, every period and every field, must
// equal those of a worker that solved A then C with nothing stopped, and of a fresh worker that solved C alone.
// One solving worker: the two-worker shared periods (`partner`) are not covered here.
// Run from the maps root: node trainworld/checks/t3925-restart-solve.mjs
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, startWorker, network, workerPack as pack, marketPacks, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t3925');
const ROWS = marketRows(), MARKET = marketPacks(['bos', 'phl']);
const ROUNDS = 3, PERIODS = 5;
const IDS = { lines: ['bos', 'phl'], stations: Array.from({ length: 8 }, (_, i) => `s${i}`) };
const finals = (msgs, version) => [...new Map(msgs.filter((m) => m.kind === 'result' && !m.cluster && m.version === version && m.summary[0] === ROUNDS).map((m) => [m.period, m])).values()].sort((a, b) => a.period - b.period);

async function worker() {
  const k = startWorker(wrapper); await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [pack('bos'), pack('phl')], intercity: MARKET, rows: ROWS, periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed'); assert.equal(r.kind, 'ready', r.error);
  return k;
}
let X = {};
const post = (k, version, net, whole = true) => {
  k.w.postMessage({ kind: 'solve', version, net, ids: IDS, rounds: ROUNDS, ...X });
  // (T-1733: an edit's whole solve waits for the page's word that its local results are shown)
  if (whole) k.w.postMessage({ kind: 'whole', version });
};
async function done(k, version, from) {
  const d = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(d.kind, 'done', d.error);
  const res = finals(k.messages.slice(from), version);
  assert.equal(res.length, PERIODS, `v${version}: a final per period`);
  return res;
}
async function solve(k, version, net) {
  const from = k.messages.length;
  post(k, version, net);
  return done(k, version, from);
}

// every field of every period's final, numbers compared with Object.is (the version and timings left out)
const SKIP = new Set(['version', 'kind', 'ms', 'setupMs', 'memBytes', 'heap']);
// (a summary's [4] is the period's ms)
const TIMED = /\.(summary|intercitySummary|citySummary\.\w+)$/;
function diff(a, b, path, out) {
  if (out.length > 8) return out;
  if (ArrayBuffer.isView(a) || Array.isArray(a)) {
    if (!(ArrayBuffer.isView(b) || Array.isArray(b)) || a.length !== b.length) { out.push(`${path}: length ${a.length} vs ${b?.length}`); return out; }
    for (let i = 0; i < a.length; i++) {
      if (i === 4 && TIMED.test(path)) continue;
      if (typeof a[i] === 'object' && a[i] !== null) diff(a[i], b[i], `${path}[${i}]`, out);
      else if (!Object.is(a[i], b[i])) { out.push(`${path}[${i}] ${a[i]} vs ${b[i]}`); if (out.length > 8) return out; }
    }
  } else if (a && typeof a === 'object') {
    if (!b || typeof b !== 'object') { out.push(`${path}: ${typeof a} vs ${b}`); return out; }
    for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) if (!SKIP.has(key)) diff(a[key], b[key], `${path}.${key}`, out);
  } else if (!Object.is(a, b)) out.push(`${path}: ${a} vs ${b}`);
  return out;
}
const diffAll = (A, B) => A.flatMap((r, q) => diff(r, B[q], `p${q}`, []));
const boardings = (res) => res.reduce((t, m) => t + Array.from(m.board).reduce((a, b) => a + b, 0), 0);

const A = network(['bos', 'phl']);
const B = network(['bos', 'phl'], { bos: [15, 12, 8], phl: [12, 10, 6] });
const C = {
  'Boston back, Philadelphia edited': network(['bos', 'phl'], { bos: [12, 10, 6], phl: [16, 12, 8] }),
  'Boston edited further': network(['bos', 'phl'], { bos: [20, 15, 10], phl: [12, 10, 6] }),
};
// where B is stopped: the first message of B's (from `from`) that matches
const CUTS = {
  'local stage': (v) => (m) => m.kind === 'plan' && m.version === v,
  'whole solve, first unit': (v) => (m) => m.kind === 'work' && m.version === v && m.stage === 1 && m.busy,
  'whole solve, halfway': (v) => (m) => m.kind === 'work' && m.version === v && m.stage === 1 && m.of[1] > 0 && m.done[1] >= m.of[1] / 2,
};

const log = [], fail = [];
const check = (ok, what) => { const line = `${ok ? 'PASS' : 'FAIL'} ${what}`; log.push(line); console.log(line); if (!ok) fail.push(what); };

// Each case twice: as the page sends it (these small clusters are then solved whole at once), and with every touched
// cluster allowed a local stage first (`localShare`, dev `?localShare`), so the frozen loads and kept periods a cut
// solve leaves behind are read too
for (const [mode, extra] of [['', {}], ['local stages, ', { localShare: [2] }]]) {
  X = extra;
  // the references: A then C with nothing stopped, and C alone on a fresh worker
  const ref = {}, fresh = {};
  for (const [name, c] of Object.entries(C)) {
    const k = await worker();
    await solve(k, 1, A);
    ref[name] = await solve(k, 2, c);
    await k.w.terminate();
    const f = await worker();
    fresh[name] = await solve(f, 1, c);
    await f.w.terminate();
    const d = diffAll(ref[name], fresh[name]);
    check(!d.length, `${mode}${name}: A then C equals C alone to the bit${d.length ? ` (${d.slice(0, 3).join('; ')})` : ''}`);
  }

  for (const [cut, at] of Object.entries(CUTS)) {
    for (const [name, c] of Object.entries(C)) {
      const k = await worker();
      await solve(k, 1, A);
      const from = k.messages.length, t0 = performance.now();
      post(k, 2, B, cut !== 'local stage');
      const m = await k.next(at(2), from);
      const cutAt = performance.now() - t0, wholeDone = m.kind === 'work' ? `${m.done[1]} of ${m.of[1]} whole units done` : 'local stage';
      const from3 = k.messages.length;
      post(k, 3, c);
      const res = await done(k, 3, from3);
      const bDone = k.messages.slice(from).some((x) => x.kind === 'done' && x.version === 2);
      const plan2 = k.messages.slice(from).find((x) => x.kind === 'plan' && x.version === 2);
      const plan3 = k.messages.slice(from3).find((x) => x.kind === 'plan' && x.version === 3);
      await k.w.terminate();
      const how = (p) => (p?.local ? `${p.local} cluster local first` : 'whole at once');
      check(!bDone, `${mode}${cut}, ${name}: B (${how(plan2)}) stopped (${wholeDone}, ${Math.round(cutAt)} ms in), never finished; C ${how(plan3)} in ${Math.round(performance.now() - t0 - cutAt)} ms`);
      const dr = diffAll(res, ref[name]), df = diffAll(res, fresh[name]);
      check(!dr.length, `${mode}${cut}, ${name}: C's figures equal A then C to the bit (${Math.round(boardings(res))} boardings)${dr.length ? `: ${dr.slice(0, 3).join('; ')}` : ''}`);
      check(!df.length, `${mode}${cut}, ${name}: and C alone on a fresh worker${df.length ? `: ${df.slice(0, 3).join('; ')}` : ''}`);
    }
  }
}

mkdirSync(join(ARTIFACTS, 't3925'), { recursive: true });
writeFileSync(join(ARTIFACTS, 't3925', 'restart-solve.log'), log.join('\n') + '\n');
console.log(fail.length ? `t3925-restart-solve FAIL (${fail.length})` : 't3925-restart-solve PASS');
process.exit(fail.length ? 1 : 0);
