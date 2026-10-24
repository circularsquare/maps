// T-325: a zone no station reaches, opened bare in the aux worker, against the same zone solved by
// the solving workers. Runs the real worker module in Node (as t149-worker-check.mjs does): two
// solving workers (periods 0, 2, 4 and 1, 3) and one aux worker hold the market and Philadelphia
// and solve a Midtown line, so Philadelphia is open but unreached; the aux worker also opens
// Philadelphia bare. Every figure the page builds for the zone must be the same to the bit:
// each period's city summary and purpose parts (the city views), each solving worker's everyday
// legs per cell (weekday and weekend, every purpose kind alone), and the commuters by mode; and
// (T-336) a bubble selection's far end: commutes, everyday purposes, longer trips.
// Run from the maps root: node trainworld/checks/t325-bare-check.mjs
import assert from 'node:assert/strict';
import { writeFileSync, mkdirSync } from 'node:fs';
import { bundleDemandWorker, startWorker, workerPack as pack, marketPacks, marketRows } from './lib.mjs';
const wrapper = await bundleDemandWorker('t325');
const start = () => startWorker(wrapper);
const ZONE = process.env.ZONE ?? 'phl';
// T-377: the market packs of New York (the line) and the zone; every registered zone's is more than a worker holds
const market = marketPacks(['nyc', ZONE]);
// a three-station line across Midtown, as t149-worker-check.mjs
const times = new Float32Array(18); for (let lev = 0; lev < 3; lev++) { times[lev * 3] = 60; times[lev * 3 + 1] = 60; times[(lev + 3) * 3 + 1] = 60; times[(lev + 3) * 3 + 2] = 60; }
const net = { frame: { kind: 'equirect', lon: -73.985, lat: 40.758 }, stXY: new Float32Array([0, 0, 4000, 0, 8000, 0]), complex: new Uint32Array([0, 1, 2]), lineN: new Uint32Array([3]), stops: new Uint32Array([0, 1, 2]), times, tph: new Float32Array([12, 12, 12]), cars: new Float32Array([4]), fare: { base: 1, perKm: .1 }, hopKm: new Float32Array([4, 4, 0, 0, 4, 4]) };
const ROUNDS = 2, GROUPS = [[0, 2, 4], [1, 3]];
const bits = (a) => new Uint32Array(a.buffer, a.byteOffset, a.length);
const sameBits = (x, y, what) => {
  assert.equal(x.length, y.length, `${what}: length ${x.length} vs ${y.length}`);
  const a = bits(Float32Array.from(x)), b = bits(Float32Array.from(y));
  for (let k = 0; k < a.length; k++) assert.equal(a[k], b[k], `${what}[${k}] ${x[k]} vs ${y[k]}`);
};
const sameNumbers = (x, y, what) => { assert.equal(x.length, y.length, what); for (let k = 0; k < x.length; k++) assert(Object.is(x[k], y[k]) || x[k] === y[k], `${what}[${k}] ${x[k]} vs ${y[k]}`); };

const solving = GROUPS.map(() => start());
const aux = start();
for (const s of [...solving, aux]) await s.next((m) => m.kind === 'boot');
const t0 = performance.now();
solving.forEach((s, i) => { s.w.postMessage({ kind: 'init', packs: [], periods: GROUPS[i] }); s.w.postMessage({ kind: 'open', packs: [], intercity: market, rows: marketRows() }); s.w.postMessage({ kind: 'open', packs: [pack(ZONE)] }); });
aux.w.postMessage({ kind: 'init', packs: [], periods: [], aux: true, groups: GROUPS });
aux.w.postMessage({ kind: 'open', packs: [], intercity: new Uint8Array(0) });
aux.w.postMessage({ kind: 'open', packs: [pack(ZONE)] });
aux.w.postMessage({ kind: 'open', packs: [pack(ZONE)], bare: true });
for (const s of solving) await s.next((m) => m.kind === 'opened' && m.cityInfo[ZONE]);
const bareOpened = await aux.next((m) => m.kind === 'opened' && m.bare);
const bareOpenMs = bareOpened.openMs;
for (const s of [...solving, aux]) s.w.postMessage({ kind: 'solve', version: 1, net, rounds: ROUNDS });
for (const s of [...solving, aux]) await s.next((m) => (m.kind === 'done' && m.version === 1) || m.kind === 'failed');
for (const s of [...solving, aux]) assert(!s.messages.some((m) => m.kind === 'failed'), s.messages.find((m) => m.kind === 'failed')?.error);
const solveMs = performance.now() - t0;

// unreached: no rail in the solve
const fig = bareOpened.bare[ZONE];
const solvedInfo = (await solving[0].next((m) => m.kind === 'opened' && m.cityInfo[ZONE])).cityInfo[ZONE];
sameNumbers(bareOpened.cityInfo[ZONE].slice(0, 4), solvedInfo.slice(0, 4), 'city info (base split)');
let checked = { summaries: 0, parts: 0, everyday: 0, kinds: 0, commute: 0 };
for (let i = 0; i < GROUPS.length; i++) {
  for (const round of [0, ROUNDS]) {
    for (const q of GROUPS[i]) {
      const r = solving[i].messages.find((m) => m.kind === 'result' && m.version === 1 && m.period === q && m.summary[0] === round);
      assert(r, `result period ${q} round ${round}`);
      assert.equal(r.citySummary[ZONE][2], 0, `${ZONE} rail in the solve, period ${q}`);
      const want = r.citySummary[ZONE], got = fig.summary[q];
      // all but the round and the stage's ms
      sameNumbers(got.filter((_, k) => k !== 0 && k !== 4), want.filter((_, k) => k !== 0 && k !== 4), `summary period ${q} round ${round}`);
      const cp = r.cityParts[ZONE], own = fig.parts[q];
      sameNumbers(own, cp.slice(0, own.length), `parts period ${q} round ${round}`);
      assert(cp.slice(own.length).every((v) => v === 0), `parts beyond the zone's kinds are zero, period ${q}`);
      checked.summaries++; checked.parts++;
    }
  }
}
// everyday legs per solving worker: weekday from the round's `sums`, weekend and each kind asked
const kinds = fig.keys.length - 1;
for (const day of [0, 1]) {
  const b = await aux.ask({ kind: 'bare', city: ZONE, version: 1, round: ROUNDS, day, purpose: -1 });
  assert.deepEqual(b.masks, GROUPS.map((g) => g.reduce((m, q) => m | 1 << q, 0)));
  for (let i = 0; i < GROUPS.length; i++) {
    const want = day === 0
      ? solving[i].messages.find((m) => m.kind === 'sums' && m.version === 1 && m.round === ROUNDS && m.city === ZONE).everyday
      : (await solving[i].ask({ kind: 'subSums', city: ZONE, version: 1, round: ROUNDS, day })).everyday;
    sameBits(b.parts[i], want, `everyday legs, day ${day}, worker ${i}`);
    checked.everyday++;
  }
  for (let k = 0; k <= kinds; k++) {
    const bk = await aux.ask({ kind: 'bare', city: ZONE, version: 1, round: ROUNDS, day, purpose: k });
    for (let i = 0; i < GROUPS.length; i++) {
      const want = (await solving[i].ask({ kind: 'purposeModes', city: ZONE, version: 1, round: ROUNDS, day, purpose: k })).modes;
      sameBits(bk.parts[i], want, `kind ${k} legs, day ${day}, worker ${i}`);
      checked.kinds++;
    }
  }
  // commuters by mode: the aux worker's answer from the solving workers' summed sums
  const parts = [];
  for (let i = 0; i < GROUPS.length; i++) parts.push(day === 0
    ? solving[i].messages.find((m) => m.kind === 'sums' && m.version === 1 && m.round === ROUNDS && m.city === ZONE).sums
    : (await solving[i].ask({ kind: 'subSums', city: ZONE, version: 1, round: ROUNDS, day })).sums);
  const sums = parts[0].slice(); for (const p of parts.slice(1)) for (let k = 0; k < sums.length; k++) sums[k] += p[k];
  const solved = (await aux.ask({ kind: 'cellModes', city: ZONE, version: 1, round: ROUNDS, day, sums })).modes;
  sameBits(b.commute, solved, `commuters by mode, day ${day}`);
  checked.commute++;
}
// T-336: a bubble selection's far end. Solved: every solving worker's `flowRail` added up in period
// order and spread by one worker (`flows`), each worker's `purposes`, and the longer trips (`icSel`)
// from the zone's own cells. Bare: the aux worker's `bareFlows`, and `icSel` asked of the solving
// workers with the squares the aux worker works out (`bareSquares`).
const cellsReply = await aux.ask({ kind: 'cells', city: ZONE, frame: net.frame });
const nCells = cellsReply.xy.length / 2;
const selections = {
  block: Uint32Array.from({ length: 60 }, (_, k) => Math.floor(nCells / 2) + k),
  spread: Uint32Array.from({ length: Math.floor(nCells / 97) }, (_, k) => k * 97),
  one: Uint32Array.of(Math.floor(nCells / 3)),
};
checked.selections = 0; checked.longer = 0;
const byMaskOrder = (rs) => [...rs].sort((a, b) => a.mask - b.mask);
for (const [name, cells] of Object.entries(selections)) {
  for (const end of [0, 1]) {
    for (const day of [0, 1]) {
      const what = `${name}, end ${end}, day ${day}`;
      const q = { city: ZONE, version: 1, round: ROUNDS, day, end, cells };
      const rail = byMaskOrder(await Promise.all(solving.map((s) => s.ask({ kind: 'flowRail', ...q }))));
      let sums = null;
      for (const r of rail) { assert(r.sums, `flowRail ${what}`); if (!sums) sums = r.sums.slice(); else for (let k = 0; k < sums.length; k++) sums[k] += r.sums[k]; }
      const solvedFlows = (await solving[0].ask({ kind: 'flows', ...q, sums })).packed;
      const purposes = byMaskOrder(await Promise.all(solving.map((s) => s.ask({ kind: 'purposes', ...q }))));
      const b = await aux.ask({ kind: 'bareFlows', ...q });
      assert(solvedFlows && b.packed, `flows answered, ${what}`);
      sameBits(b.packed, solvedFlows, `far end (commutes), ${what}`);
      const order = b.masks.map((m, i) => [m, i]).sort((x, y) => x[0] - y[0]).map(([, i]) => i);
      assert.deepEqual(order.map((i) => b.masks[i]), purposes.map((r) => r.mask), `masks, ${what}`);
      order.forEach((i, j) => {
        sameBits(b.work[i], purposes[j].work, `work timing, ${what}, part ${j}`);
        sameBits(b.everyday[i], purposes[j].everyday, `everyday far end, ${what}, part ${j}`);
        assert.deepEqual(b.names, purposes[j].names, `purpose names, ${what}`);
      });
      checked.selections++;
      // longer trips and commutes between zones: the squares from the bare copy, asked of the market
      const sq = await aux.ask({ kind: 'bareSquares', ...q });
      const solvedIc = byMaskOrder(await Promise.all(solving.map((s) => s.ask({ kind: 'icSel', ...q }))));
      if (!sq.xy) { assert(solvedIc.every((r) => !r.packed), `outside the market, ${what}`); continue; }
      const bareIc = byMaskOrder(await Promise.all(solving.map((s) => s.ask({ kind: 'icSel', ...q, sq: { xy: sq.xy, share: sq.share } }))));
      solvedIc.forEach((r, j) => {
        assert(r.packed && bareIc[j].packed, `icSel answered, ${what}`);
        assert.equal(r.mask, bareIc[j].mask);
        const x = new Uint32Array(r.packed.buffer), y = new Uint32Array(bareIc[j].packed.buffer);
        assert.equal(x.length, y.length, `longer trips length, ${what}`);
        for (let k = 0; k < x.length; k++) assert.equal(x[k], y[k], `longer trips, ${what}, part ${j} [${k >> 1}]`);
      });
      checked.longer++;
    }
  }
}
// not open bare: null, never a failure
const none = await aux.ask({ kind: 'bare', city: 'bos', version: 1, round: ROUNDS, day: 0, purpose: -1 });
assert.equal(none.commute, null);
assert.equal((await aux.ask({ kind: 'bareFlows', city: 'bos', version: 1, round: ROUNDS, day: 0, end: 0, cells: Uint32Array.of(0) })).packed, null);
assert.equal((await aux.ask({ kind: 'bareSquares', city: 'bos', version: 1, round: ROUNDS, day: 0, end: 0, cells: Uint32Array.of(0) })).xy, null);
const mem = { solvingWorkerMiB: Math.round(solving[0].messages.filter((m) => m.kind === 'done').at(-1).memBytes / 1048576), auxWorkerMiB: Math.round(aux.messages.filter((m) => m.kind === 'done').at(-1).memBytes / 1048576) };
for (const s of [...solving, aux]) await s.w.terminate();
const result = { zone: ZONE, groups: GROUPS, kinds, checked, identical: true, bareOpenMs: Math.round(bareOpenMs), setupAndSolveMs: Math.round(solveMs), ...mem };
mkdirSync(new URL('../../artifacts/trainworld/notes-outputs/', import.meta.url), { recursive: true });
writeFileSync(new URL(`../../artifacts/trainworld/notes-outputs/T-325-bare-check-${ZONE}.json`, import.meta.url), JSON.stringify(result, null, 2)); console.log(result);
