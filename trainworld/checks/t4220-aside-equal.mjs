// T-4220 stage 5: a side job's figures during a refresh equal a main-pool local solve on the same base.
// Runs the real demand worker in Node (as t3925-restart-solve.mjs) with Boston, Philadelphia and their market open:
// one main worker solves A, ends a city edit B locally, and starts the refresh (sending its snapshot); a side worker
// with only the fresh zone open solves edit C against that snapshot. Each C's result is compared against C solved
// locally on a fresh worker holding B's state: every line's riders, every station's boardings, and every period's
// seg/board/alight must match within 1e-6. Covers C as a one-level train edit (kept periods) and C as a stop edit
// (all periods solved, route nodes remapped), both with Boston and Philadelphia as two clusters (frozen market) and
// joined in one cluster via `oneMarket` (frozen city + frozen market). Stage 4 refuses intercity edits, so C during an
// intercity edit's whole solve is omitted here; the T-3925 cut and edit-batching cases belong to stage 6.
// Run from the maps root: node trainworld/checks/t4220-aside-equal.mjs
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, bundleDemandWorker, loadApp, startWorker, network, workerPack as pack, marketPacks, marketRows } from './lib.mjs';

const wrapper = await bundleDemandWorker('t4220');
const { ShownStore, dayViews, dayResultOf, PERIODS, TIMES_PER_STOP } = await loadApp(`
  export { ShownStore } from '@app/src/workers/demandShown.ts';
  export { dayViews, dayResultOf } from '@app/src/workers/demandSums.ts';
  export { PERIODS, TIMES_PER_STOP } from '@app/src/game/periods.ts';
`);
const PERIOD_STRIDE = 6;

const ROWS = marketRows(), MARKET = marketPacks(['bos', 'phl']);
const ROUNDS = 3, TOL = 1e-6;
const CITIES = ['bos', 'phl'];
const IDS_AB = { lines: ['bos', 'phl'], stations: Array.from({ length: 8 }, (_, i) => `s${i}`) };
const IDS_STOP = { lines: ['bos', 'phl'], stations: Array.from({ length: 9 }, (_, i) => `s${i}`) };

// Convert checks/lib.mjs's 3-level network helper to the T-3542 5-period format (5 tph per line, 10 times per stop).
function makeNet(tphByLevel, bosInfill = false) {
  const base = network(['bos', 'phl']);
  const n0 = base.lineN[0], n1 = base.lineN[1];
  const stCount = base.stXY.length / 2;
  let stXY = base.stXY, complex = base.complex;
  let bosStops = Array.from(base.stops.slice(0, n0));
  let bosHops = [180, 180, 180];
  if (bosInfill) {
    // Add an infill stop (s8) on Boston's line between stops 1 and 2, splitting that 180 s hop into 95 s + 95 s
    // (10 s shift < RUN_SHIFT_MIN 3 min; total run 9.17 min < WINDOW_EDIT_MIN 20 min, so it stays a city edit).
    stXY = new Float32Array(base.stXY.length + 2);
    stXY.set(base.stXY);
    const s1 = base.stops[1], s2 = base.stops[2];
    stXY[2 * stCount] = (base.stXY[2 * s1] + base.stXY[2 * s2]) / 2;
    stXY[2 * stCount + 1] = (base.stXY[2 * s1 + 1] + base.stXY[2 * s2 + 1]) / 2;
    complex = new Uint32Array(stCount + 1);
    complex.set(base.complex);
    complex[stCount] = stCount;
    bosStops = [base.stops[0], base.stops[1], stCount, base.stops[2], base.stops[3]];
    bosHops = [180, 95, 95, 180];
  }
  const phlStops = Array.from(base.stops.slice(n0));
  const phlHops = [180, 180, 180];
  const newN0 = bosStops.length;
  const lineN = new Uint32Array([newN0, n1]);
  const stops = new Uint32Array([...bosStops, ...phlStops]);

  const times = new Float32Array(TIMES_PER_STOP * (newN0 + n1));
  const fillTimes = (stopOff, m, hops) => {
    const baseOff = TIMES_PER_STOP * stopOff;
    for (let q = 0; q < PERIODS.length; q++) {
      for (let k = 0; k < m - 1; k++) {
        times[baseOff + q * m + k] = hops[k];
        times[baseOff + (PERIODS.length + q) * m + k + 1] = hops[k];
      }
    }
  };
  fillTimes(0, newN0, bosHops);
  fillTimes(newN0, n1, phlHops);

  const expandTph = ([hi, med, lo]) => [hi, hi, hi, med, lo];
  const tph = new Float32Array([...expandTph(tphByLevel.bos), ...expandTph(tphByLevel.phl)]);

  const kmOf = (a, b) => Math.hypot(stXY[2 * b] - stXY[2 * a], stXY[2 * b + 1] - stXY[2 * a + 1]) / 1000;
  const bosKm = bosStops.slice(1).map((s, k) => kmOf(bosStops[k], s));
  const phlKm = phlStops.slice(1).map((s, k) => kmOf(phlStops[k], s));
  const hopKm = new Float32Array([...bosKm, 0, ...bosKm.slice().reverse(), 0, ...phlKm, 0, ...phlKm.slice().reverse(), 0]);

  return { frame: base.frame, stXY, complex, lineN, stops, times, tph, cars: base.cars.slice(), fare: { ...base.fare }, hopKm };
}

function layoutOf(net, ids) {
  const out = [];
  let o = 0, s = 0;
  for (let i = 0; i < net.lineN.length; i++) {
    const n = net.lineN[i];
    const stops = [];
    for (let k = 0; k < n; k++) stops.push(ids.stations[net.stops[s + k]]);
    out.push({ id: ids.lines[i], o, n, stops });
    o += 2 * n;
    s += n;
  }
  return out;
}

async function mainWorker() {
  const k = startWorker(wrapper);
  await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [pack('bos'), pack('phl')], intercity: MARKET, rows: ROWS, periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed');
  assert.equal(r.kind, 'ready', r.error);
  return { k, ready: r };
}

async function sideWorker() {
  const k = startWorker(wrapper);
  await k.next((m) => m.kind === 'boot');
  k.w.postMessage({ kind: 'init', packs: [], periods: [0, 1, 2, 3, 4] });
  const r = await k.next((m) => m.kind === 'ready' || m.kind === 'failed');
  assert.equal(r.kind, 'ready', r.error);
  return k;
}

async function solveWhole(k, version, net, ids, extra = {}) {
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version, net, ids, rounds: ROUNDS, ...extra });
  k.w.postMessage({ kind: 'whole', version });
  const d = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(d.kind, 'done', d.error);
  return d;
}

async function solveLocalEnd(k, version, net, ids, extra = {}) {
  const from = k.messages.length;
  k.w.postMessage({ kind: 'solve', version, net, ids, rounds: ROUNDS, ...extra });
  const plan = await k.next((m) => (m.kind === 'plan' && m.version === version) || m.kind === 'failed', from);
  assert.equal(plan.kind, 'plan', plan.error);
  assert.ok(plan.local > 0 && plan.final, `v${version}: expected city edit ending locally (local=${plan.local}, final=${plan.final})`);
  k.w.postMessage({ kind: 'endLocal', version });
  const d = await k.next((m) => (m.kind === 'done' && m.version === version) || m.kind === 'failed', from);
  assert.equal(d.kind, 'done', d.error);
  assert.ok(d.localHeld, `v${version}: expected localHeld`);
  const results = k.messages.slice(from).filter((m) => m.kind === 'result' && m.version === version && m.cluster?.local);
  assert.equal(results.length, PERIODS.length, `v${version}: expected ${PERIODS.length} local period results, got ${results.length}`);
  return { done: d, results: results.sort((a, b) => a.period - b.period) };
}

function reduceCluster(results, net, ids, version, ready) {
  const res = PERIODS.map((_, q) => results.find((r) => r.period === q));
  assert.ok(res.every(Boolean), 'missing a period result');
  const own = new Set(res[0].cluster.lines);
  const layout = layoutOf(net, ids).filter((_, i) => own.has(i));
  const peakHourFactor = PERIODS.map((_, q) => ready.periods[q * PERIOD_STRIDE + 5] ?? 1);
  const [d] = dayViews({
    version,
    round: ROUNDS,
    final: true,
    layout,
    res: res.map((r) => dayResultOf(r)),
    dayTable: ready.days,
    peakHourFactor,
    solved: [],
    places: 0,
    days: [0],
    noCorridors: true,
  });
  const lines = Object.fromEntries([...d.view.lines].map(([id, l]) => [id, l.ridersPerDay]));
  const stations = Object.fromEntries([...d.view.stations].map(([id, st]) => [id, st.boardings]));
  return { lines, stations };
}

function maxMapDiff(a, b) {
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
  let max = 0, worst = '', aVal = 0, bVal = 0, total = 0;
  for (const k of keys) {
    assert.ok(k in a && k in b, `key ${k} missing on one side`);
    total += a[k];
    const d = Math.abs(a[k] - b[k]);
    if (d > max) { max = d; worst = k; aVal = a[k]; bVal = b[k]; }
  }
  return { max, worst, aVal, bVal, count: keys.length, total };
}

function maxArrayDiff(sideRes, refRes) {
  let max = 0, worst = '';
  for (let q = 0; q < PERIODS.length; q++) {
    const s = sideRes[q], r = refRes[q];
    for (const field of ['seg', 'board', 'alight']) {
      assert.equal(s[field].length, r[field].length, `p${q}.${field} length mismatch`);
      for (let i = 0; i < s[field].length; i++) {
        const d = Math.abs(s[field][i] - r[field][i]);
        if (d > max) { max = d; worst = `p${q}.${field}[${i}] (${s[field][i]} vs ${r[field][i]})`; }
      }
    }
  }
  return { max, worst };
}

const A = makeNet({ bos: [12, 10, 6], phl: [12, 10, 6] });
const B = makeNet({ bos: [15, 12, 8], phl: [12, 10, 6] });
const CASES = [
  {
    name: 'one-level train edit (kept periods)',
    net: makeNet({ bos: [15, 14, 8], phl: [12, 10, 6] }),
    ids: IDS_AB,
    expectKeep: (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4),
    expectSolve: [3],
    expectSums: [],
  },
  {
    name: 'stop edit (all periods solved)',
    net: makeNet({ bos: [15, 12, 8], phl: [12, 10, 6] }, true),
    ids: IDS_STOP,
    expectKeep: 0,
    expectSolve: [0, 1, 2, 3, 4],
    expectSums: ['bos'],
  },
];

const log = [], fail = [];
const check = (ok, what) => { const line = `${ok ? 'PASS' : 'FAIL'} ${what}`; log.push(line); console.log(line); if (!ok) fail.push(what); };

const MODES = [
  { label: 'two clusters', extra: {}, expectLines: 1 },
  { label: 'shared cluster (frozen phl)', extra: { oneMarket: true }, expectLines: 2 },
];

for (const mode of MODES) {
  const { k: main, ready } = await mainWorker();
  await solveWhole(main, 1, A, IDS_AB, mode.extra);
  await solveLocalEnd(main, 2, B, IDS_AB, mode.extra);

  const refreshFrom = main.messages.length;
  main.w.postMessage({ kind: 'solve', version: 3, net: B, ids: IDS_AB, rounds: ROUNDS, refresh: true, snap: true, ...mode.extra });
  const shownMsg = await main.next((m) => (m.kind === 'shown' && m.version === 3) || m.kind === 'failed', refreshFrom);
  assert.equal(shownMsg.kind, 'shown', shownMsg.error);
  const store = new ShownStore();
  assert.equal(store.merge(3, 0, shownMsg.snap), 'held', 'snapshot merged into ShownStore');
  const shown = store.forAside();
  assert.ok(shown, 'ShownStore.forAside() returned a snapshot');
  check(true, `${mode.label}: refresh snapshot v3 held (${shown.pops.length} populations, ${shown.zones.length} zones, ${(shownMsg.snap.bytes / 1024).toFixed(0)} KB)`);

  const side = await sideWorker();
  const openedOnSide = new Set();
  let sentGen = -1;
  let sideVersion = 10;

  for (const tc of CASES) {
    sideVersion++;
    const tag = `${mode.label}, ${tc.name}`;
    const planFrom = side.messages.length;
    side.w.postMessage({
      kind: 'aside',
      version: sideVersion,
      net: tc.net,
      ids: tc.ids,
      rounds: ROUNDS,
      cities: CITIES,
      gen: shown.gen,
      ...(sentGen !== shown.gen ? { shown } : {}),
    });
    sentGen = shown.gen;
    const planMsg = await side.next((m) => (m.kind === 'asidePlan' && m.version === sideVersion) || m.kind === 'failed', planFrom);
    assert.equal(planMsg.kind, 'asidePlan', planMsg.error);
    const { plan } = planMsg;
    check(!plan.refused, `${tag}: plan accepted${plan.refused ? ` (refused ${plan.refused}: ${plan.why})` : ''}`);
    if (plan.refused) continue;

    check(
      plan.fresh.map((z) => z.id).join(',') === 'bos' &&
        plan.clusters.length === 1 &&
        plan.clusters[0].keep === tc.expectKeep &&
        plan.clusters[0].solve.join(',') === tc.expectSolve.join(','),
      `${tag}: plan fresh=[${plan.fresh.map((z) => z.id)}], keep=${plan.clusters[0]?.keep}, solve=[${plan.clusters[0]?.solve}]`,
    );

    const toOpen = plan.fresh.filter((z) => !openedOnSide.has(z.id));
    if (toOpen.length) {
      const openFrom = side.messages.length;
      side.w.postMessage({ kind: 'open', packs: toOpen.map((z) => ({ ...pack(z.id), level: z.level })) });
      const op = await side.next((m) => m.kind === 'opened' || m.kind === 'failed', openFrom);
      assert.equal(op.kind, 'opened', op.error);
      for (const z of toOpen) openedOnSide.add(z.id);
    }

    const solveFrom = side.messages.length;
    side.w.postMessage({
      kind: 'solve',
      version: sideVersion,
      net: tc.net,
      ids: tc.ids,
      rounds: ROUNDS,
      aside: { gen: shown.gen, cities: CITIES },
    });
    const asideDone = await side.next((m) => (m.kind === 'asideDone' && m.version === sideVersion) || m.kind === 'failed', solveFrom);
    assert.equal(asideDone.kind, 'asideDone', asideDone.error);
    assert.ok(!asideDone.plan.refused && !asideDone.stale, `${tag}: side job finished cleanly`);
    const sideRes = side.messages.slice(solveFrom).filter((m) => m.kind === 'result' && m.version === sideVersion && m.cluster?.aside).sort((a, b) => a.period - b.period);
    assert.equal(sideRes.length, PERIODS.length, `${tag}: expected ${PERIODS.length} aside results`);
    check((asideDone.sums ?? []).join(',') === tc.expectSums.join(','), `${tag}: bubble sums=[${(asideDone.sums ?? []).join(',')}] (expected [${tc.expectSums.join(',')}])`);

    // Reference: a fresh worker holding B's state, then solving C locally
    const { k: ref } = await mainWorker();
    await solveWhole(ref, 1, A, IDS_AB, mode.extra);
    await solveLocalEnd(ref, 2, B, IDS_AB, mode.extra);
    const { results: refRes } = await solveLocalEnd(ref, 3, tc.net, tc.ids, mode.extra);
    await ref.w.terminate();

    const sideView = reduceCluster(sideRes, tc.net, tc.ids, sideVersion, ready);
    const refView = reduceCluster(refRes, tc.net, tc.ids, 3, ready);
    const lineDiff = maxMapDiff(sideView.lines, refView.lines);
    const stDiff = maxMapDiff(sideView.stations, refView.stations);
    const arrDiff = maxArrayDiff(sideRes, refRes);

    check(
      lineDiff.count === mode.expectLines && lineDiff.total > 0 && lineDiff.max <= TOL,
      `${tag}: lines (${lineDiff.count}, ${Math.round(lineDiff.total)} riders) max diff ${lineDiff.max.toExponential(3)} <= ${TOL}${lineDiff.worst ? ` (worst ${lineDiff.worst}: ${lineDiff.aVal} vs ${lineDiff.bVal})` : ''}`,
    );
    check(
      stDiff.total > 0 && stDiff.max <= TOL,
      `${tag}: stations (${stDiff.count}, ${Math.round(stDiff.total)} boardings) max diff ${stDiff.max.toExponential(3)} <= ${TOL}${stDiff.worst ? ` (worst ${stDiff.worst}: ${stDiff.aVal} vs ${stDiff.bVal})` : ''}`,
    );
    check(
      arrDiff.max <= TOL,
      `${tag}: period seg/board/alight max diff ${arrDiff.max.toExponential(3)} <= ${TOL}${arrDiff.worst ? ` (worst ${arrDiff.worst})` : ''}`,
    );
  }

  const refreshDone = await main.next((m) => (m.kind === 'done' && m.version === 3) || m.kind === 'failed', refreshFrom);
  assert.equal(refreshDone.kind, 'done', refreshDone.error);
  await main.w.terminate();
  await side.w.terminate();
}

mkdirSync(join(ARTIFACTS, 't4220'), { recursive: true });
writeFileSync(join(ARTIFACTS, 't4220', 'aside-equal.log'), log.join('\n') + '\n');
console.log(fail.length ? `t4220-aside-equal FAIL (${fail.length})` : 't4220-aside-equal PASS');
process.exit(fail.length ? 1 : 0);
