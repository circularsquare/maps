// T-325: a zone no station reaches, opened bare in the aux worker, against the same zone solved by
// the solving workers. Runs the real worker module in Node (as t149-worker-check.mjs does): two
// solving workers (periods 0, 2, 4 and 1, 3) and one aux worker hold the market and Philadelphia
// and solve a Midtown line, so Philadelphia is open but unreached; the aux worker also opens
// Philadelphia bare. Every figure the page builds for the zone must be the same to the bit:
// each period's city summary and purpose parts (the city views), each solving worker's everyday
// legs per cell (weekday and weekend, every purpose kind alone), and the commuters by mode.
// Run from the maps root: node trainworld/checks/t325-bare-check.mjs
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { Worker } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { rolldown } from '../app/node_modules/rolldown/dist/index.mjs';
const artifacts = new URL('../../artifacts/trainworld/t325/', import.meta.url);
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
const ZONE = process.env.ZONE ?? 'phl';
const pack = (id) => ({ id, header: readFileSync(new URL(`../data/packs/${id}.json`, import.meta.url), 'utf8'), bin: Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`, import.meta.url))).buffer });
const market = readFileSync(new URL(`../data/packs/${registry.market.pack}.json`, import.meta.url), 'utf8');
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
solving.forEach((s, i) => { s.w.postMessage({ kind: 'init', packs: [], periods: GROUPS[i] }); s.w.postMessage({ kind: 'open', packs: [], intercity: market }); s.w.postMessage({ kind: 'open', packs: [pack(ZONE)] }); });
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
// not open bare: null, never a failure
const none = await aux.ask({ kind: 'bare', city: 'bos', version: 1, round: ROUNDS, day: 0, purpose: -1 });
assert.equal(none.commute, null);
const mem = { solvingWorkerMiB: Math.round(solving[0].messages.filter((m) => m.kind === 'done').at(-1).memBytes / 1048576), auxWorkerMiB: Math.round(aux.messages.filter((m) => m.kind === 'done').at(-1).memBytes / 1048576) };
for (const s of [...solving, aux]) await s.w.terminate();
const result = { zone: ZONE, groups: GROUPS, kinds, checked, identical: true, bareOpenMs: Math.round(bareOpenMs), setupAndSolveMs: Math.round(solveMs), ...mem };
mkdirSync(new URL('../../artifacts/trainworld/notes-outputs/', import.meta.url), { recursive: true });
writeFileSync(new URL(`../../artifacts/trainworld/notes-outputs/T-325-bare-check-${ZONE}.json`, import.meta.url), JSON.stringify(result, null, 2)); console.log(result);
