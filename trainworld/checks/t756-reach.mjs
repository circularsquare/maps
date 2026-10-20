// T-756: each population solves on the stations within its reach. Two modes, from the maps root:
//   node trainworld/checks/t756-reach.mjs capture <save> <label>   (headless Chrome on its own Vite)
//   node trainworld/checks/t756-reach.mjs compare <label> [wide|reach|both]
// `capture` loads the save, records the network the page sends its demand workers and the packs they
// open, and writes ../artifacts/trainworld/t756/<label>.net.json. `compare` runs the real demand
// worker in Node on it (WORKERS threads, default 3, the periods dealt as the page deals them), once
// with every population on its whole cluster (`wide`, `?perfOff=reach`) and once on its reach, and
// writes the figures, times and memory of each and their differences to <label>.compare.json. A run
// already written is read back instead of solved again unless FRESH=1. Fails when a figure moves
// more than the tolerance (TOL, default 0.005): the network's and the market's rail and walked trips
// and each city's rail trips per period within TOL (a city's within 10 trips either way), each
// line's day of boardings within 2 x TOL of the line's (or of 1% of the busiest line's).
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { mkdtemp } from 'node:fs/promises';
import { Worker } from 'node:worker_threads';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t756/', import.meta.url));
mkdirSync(OUT, { recursive: true });
const [mode, a1, a2, a3] = process.argv.slice(2);
const TOL = Number(process.env.TOL ?? 0.005);

const enc = (v) => ({ t: v.constructor.name, b: Buffer.from(v.buffer, v.byteOffset, v.byteLength).toString('base64') });
const dec = (o) => { const b = Buffer.from(o.b, 'base64'); const ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); return new globalThis[o.t](ab); };

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? readdirSync(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

async function capture(save, label) {
  const port = Number(process.env.PORT ?? 6165);
  let vite = null, base = process.env.URL;
  if (!base) {
    vite = spawn(process.execPath, [join(HERE, 'dev-nowatch.mjs'), String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    vite.stdout.on('data', (d) => { out += d; });
    vite.stderr.on('data', (d) => { out += d; });
    const deadline = Date.now() + 60000;
    while (!/localhost:/.test(out)) {
      if (vite.exitCode !== null || Date.now() > deadline) throw new Error('Vite did not start: ' + out.slice(-500));
      await new Promise((r) => setTimeout(r, 200));
    }
    base = `http://localhost:${port}/`;
  }
  const { chromium } = await playwright();
  const profile = await mkdtemp(join(tmpdir(), 'tw-t756-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 } });
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      window.__t756 = { solves: [], packs: new Set(), markets: new Set() };
      const orig = Worker.prototype.postMessage;
      Worker.prototype.postMessage = function (m, t) {
        try {
          if (m && (m.kind === 'init' || m.kind === 'open') && !m.bare && !m.aux) {
            for (const p of m.packs ?? []) window.__t756.packs.add(p.id);
            for (const p of m.intercity ?? []) window.__t756.markets.add(p.id);
          }
          if (m && m.kind === 'close') { for (const id of m.ids ?? []) if (!m.bare) window.__t756.packs.delete(id); for (const id of m.markets ?? []) window.__t756.markets.delete(id); }
          if (m && m.kind === 'solve') {
            const n = m.net, c = (v) => Array.from(v);
            window.__t756.solves.push({ at: performance.now(), version: m.version, packs: [...window.__t756.packs], markets: [...window.__t756.markets],
              net: { frame: n.frame, fare: n.fare, stXY: c(n.stXY), complex: c(n.complex), lineN: c(n.lineN), stops: c(n.stops), times: c(n.times), tph: c(n.tph), cars: c(n.cars), hopKm: c(n.hopKm) } });
          }
        } catch (e) { console.error('t756 hook', e); }
        return orig.call(this, m, t);
      };
    });
    await page.goto(`${base}?debug=1&new=1&opened=all&demandWorkers=2&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
    await page.waitForTimeout(3000);
    const bytes = readFileSync(save);
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    // the last solve once none has come for 30 s
    const deadline = Date.now() + 15 * 60000;
    for (;;) {
      const s = await page.evaluate(() => ({ n: window.__t756.solves.filter((x) => x.net.lineN.length > 0).length, last: window.__t756.solves.at(-1)?.at ?? 0, now: performance.now() }));
      if (s.n && s.now - s.last > 30000) break;
      if (Date.now() > deadline) throw new Error('no settled solve');
      await page.waitForTimeout(3000);
    }
    const last = await page.evaluate(() => { const s = window.__t756.solves.at(-1); return { version: s.version, packs: s.packs, markets: s.markets, net: s.net }; });
    const F = { stXY: Float32Array, complex: Uint32Array, lineN: Uint32Array, stops: Uint32Array, times: Float32Array, tph: Float32Array, cars: Float32Array, hopKm: Float32Array };
    const net = { frame: last.net.frame, fare: last.net.fare };
    for (const [k, T] of Object.entries(F)) net[k] = enc(T.from(last.net[k]));
    const file = join(OUT, `${label}.net.json`);
    writeFileSync(file, JSON.stringify({ save, packs: last.packs, markets: last.markets, net }));
    console.log(`captured ${label}: ${last.net.stXY.length / 2} stations, ${last.net.lineN.length} lines, packs ${last.packs.join(',')}, markets ${last.markets.length} -> ${file}`);
  } finally {
    await ctx.close();
    if (vite) vite.kill();
  }
}

// ---- the worker in Node (as t409-worker-check.mjs)
async function bundleWorker() {
  const { rolldown } = await import(pathToFileURL(join(HERE, '../app/node_modules/rolldown/dist/index.mjs')).href);
  const compiled = join(OUT, 'demand-worker.mjs'), wrapper = join(OUT, 'worker-adapter.mjs');
  const wasm = fileURLToPath(new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url));
  const bundle = await rolldown({ input: fileURLToPath(new URL('../app/src/workers/demand.worker.ts', import.meta.url)), platform: 'browser' });
  await bundle.write({ file: compiled, format: 'esm' });
  await bundle.close();
  writeFileSync(wrapper, `import {parentPort} from 'node:worker_threads';
import {readFileSync} from 'node:fs';
globalThis.onmessage=null;
globalThis.location={search:'',hash:'',href:'http://localhost/'};
globalThis.postMessage=(data,options={})=>parentPort.postMessage(data,options.transfer??[]);
globalThis.fetch=async()=>new Response(readFileSync(${JSON.stringify(wasm)}),{headers:{'Content-Type':'application/wasm'}});
await import(${JSON.stringify(pathToFileURL(compiled).href)});
parentPort.on('message',data=>globalThis.onmessage({data}));
parentPort.postMessage({kind:'boot'});
`);
  return wrapper;
}

function start(wrapper) {
  const w = new Worker(wrapper, { resourceLimits: { maxOldGenerationSizeMb: 4096 } }), messages = [], waiters = [];
  w.on('message', (msg) => { messages.push(msg); for (const f of [...waiters]) f(msg); });
  w.on('error', (e) => { messages.push({ kind: 'failed', error: String(e) }); for (const f of [...waiters]) f({ kind: 'failed', error: String(e) }); });
  const next = (predicate, from = 0) => new Promise((resolve) => {
    const existing = messages.slice(from).find(predicate); if (existing) return resolve(existing);
    const f = (m) => { if (predicate(m) || m.kind === 'failed') { waiters.splice(waiters.indexOf(f), 1); resolve(m); } }; waiters.push(f);
  });
  return { w, messages, next };
}

const ROUNDS = 3;
const packOf = (id) => ({ id, header: readFileSync(new URL(`../data/packs/${id}.json`, import.meta.url), 'utf8'), bin: Uint8Array.from(readFileSync(new URL(`../data/packs/${id}.bin`, import.meta.url))).buffer });
const marketOf = (id) => ({ id, pack: readFileSync(new URL(`../data/packs/${id}.market.json`, import.meta.url), 'utf8') });

async function solveIn(wrapper, cap, extra) {
  const W = Number(process.env.WORKERS ?? 3);
  const periods = Array.from({ length: W }, (_, i) => [0, 1, 2, 3, 4].filter((q) => q % W === i));
  const net = { frame: cap.net.frame, fare: cap.net.fare };
  for (const k of ['stXY', 'complex', 'lineN', 'stops', 'times', 'tph', 'cars', 'hopKm']) net[k] = dec(cap.net[k]);
  const t0 = performance.now();
  const ks = await Promise.all(periods.map(async (ps) => {
    const k = start(wrapper);
    await k.next((m) => m.kind === 'boot');
    k.w.postMessage({ kind: 'init', packs: cap.packs.map(packOf), intercity: cap.markets.map(marketOf), rows: readFileSync(new URL('../app/src/game/market-rows.json', import.meta.url), 'utf8'), periods: ps });
    const r = await k.next((m) => m.kind === 'ready');
    if (r.kind !== 'ready') throw new Error('init failed: ' + r.error);
    return k;
  }));
  const tOpen = performance.now() - t0;
  const t1 = performance.now();
  const dones = await Promise.all(ks.map(async (k) => {
    const from = k.messages.length;
    k.w.postMessage({ kind: 'solve', version: 1, net, rounds: ROUNDS, first: cap.packs[0], ...extra });
    const d = await k.next((m) => (m.kind === 'done' && m.version === 1), from);
    if (d.kind !== 'done') throw new Error('solve failed: ' + d.error);
    return { d, ms: performance.now() - t1, results: k.messages.slice(from).filter((m) => m.kind === 'result' && !m.cluster && m.summary[0] === ROUNDS) };
  }));
  const results = new Map();
  for (const { results: rs } of dones) for (const r of rs) results.set(r.period, r);
  for (const k of ks) await k.w.terminate();
  const per = [...results.values()].sort((a, b) => a.period - b.period);
  // the figures compared: per period the city summaries, the market's, and per line boardings and
  // riders over its hops (seg) summed over the day's periods
  const L = net.lineN.length, lineOf = new Uint32Array(2 * net.stops.length);
  let o = 0;
  for (let l = 0; l < L; l++) { for (let r = 0; r < 2 * net.lineN[l]; r++) lineOf[o + r] = l; o += 2 * net.lineN[l]; }
  const lineBoard = new Float64Array(L), lineSeg = new Float64Array(L);
  for (const r of per) for (let i = 0; i < r.board.length; i++) { lineBoard[lineOf[i]] += r.board[i]; lineSeg[lineOf[i]] += r.seg[i]; }
  return {
    openMs: Math.round(tOpen), solveMs: Math.round(Math.max(...dones.map((x) => x.ms))), workerMB: dones.map((x) => Math.round(x.d.memBytes / 1048576)), clusters: dones[0].d.clusters,
    periods: per.map((r) => ({ period: r.period, summary: r.summary.slice(0, 4).concat(r.summary.slice(5)), citySummary: r.citySummary, intercitySummary: r.intercitySummary })),
    lineBoard: Array.from(lineBoard), lineSeg: Array.from(lineSeg),
  };
}

function compareRuns(a, b) {
  const rel = (x, y) => Math.abs(x - y) / Math.max(Math.abs(x), Math.abs(y), 1e-9);
  const out = { totals: [], cities: {}, market: [], lines: {} };
  let worst = 0;
  for (let i = 0; i < a.periods.length; i++) {
    const [p, q] = [a.periods[i], b.periods[i]];
    // trips, rail, walked: the network's, each city's, the market's
    out.totals.push([1, 2, 3].map((k) => [p.summary[k], q.summary[k], rel(p.summary[k], q.summary[k])]));
    worst = Math.max(worst, ...[2, 3].map((k) => rel(p.summary[k], q.summary[k])));
    for (const id of Object.keys(p.citySummary)) {
      const [x, y] = [p.citySummary[id][2], q.citySummary[id]?.[2] ?? 0];
      (out.cities[id] ??= []).push([x, y, rel(x, y)]);
      // a small city (a few hundred rail trips) may move by a few trips
      if (Math.abs(x - y) > 10) worst = Math.max(worst, rel(x, y));
    }
    if (p.intercitySummary.length) { out.market.push([p.intercitySummary[2], q.intercitySummary[2], rel(p.intercitySummary[2], q.intercitySummary[2])]); worst = Math.max(worst, rel(p.intercitySummary[2], q.intercitySummary[2])); }
  }
  const boardA = a.lineBoard.reduce((s, v) => s + v, 0), boardB = b.lineBoard.reduce((s, v) => s + v, 0);
  // a line's boardings: off by under TOL of the line, or under TOL x 10 of the line's when that is
  // a small line (under 1% of the busiest line's)
  const top = Math.max(...a.lineBoard);
  let lineWorst = 0, lineWorstAt = -1, absMax = 0;
  a.lineBoard.forEach((x, l) => {
    const y = b.lineBoard[l], d = Math.abs(x - y);
    absMax = Math.max(absMax, d);
    const r = d / Math.max(x, y, top * 0.01);
    if (r > lineWorst) { lineWorst = r; lineWorstAt = l; }
  });
  out.lines = { boardings: [boardA, boardB, rel(boardA, boardB)], worstLine: lineWorstAt, worstShare: lineWorst, worstAbs: absMax, busiest: top };
  out.worst = Math.max(worst, rel(boardA, boardB));
  out.pass = out.worst <= TOL && lineWorst <= 2 * TOL;
  return out;
}

async function compare(label, which = 'both') {
  const cap = JSON.parse(readFileSync(join(OUT, `${label}.net.json`), 'utf8'));
  const wrapper = await bundleWorker();
  const runs = {};
  for (const m of which === 'both' ? ['wide', 'reach'] : [which]) {
    const file = join(OUT, `${label}.${m}.json`);
    if (existsSync(file) && process.env.FRESH !== '1') { runs[m] = JSON.parse(readFileSync(file, 'utf8')); continue; }
    console.log(`${label} ${m}: solving`);
    runs[m] = await solveIn(wrapper, cap, m === 'wide' ? { wide: true } : {});
    writeFileSync(file, JSON.stringify(runs[m]));
    console.log(`${label} ${m}: open ${runs[m].openMs} ms, solve ${runs[m].solveMs} ms, worker MB ${runs[m].workerMB}`);
  }
  if (!runs.wide || !runs.reach) return;
  const c = compareRuns(runs.wide, runs.reach);
  c.time = { wide: runs.wide.solveMs, reach: runs.reach.solveMs };
  c.workerMB = { wide: runs.wide.workerMB, reach: runs.reach.workerMB };
  writeFileSync(join(OUT, `${label}.compare.json`), JSON.stringify(c, null, 1));
  const pct = (x) => (100 * x).toFixed(3) + '%';
  console.log(`${c.pass ? 'PASS' : 'FAIL'}  t756 ${label}: solve ${c.time.wide} -> ${c.time.reach} ms, worker MB ${c.workerMB.wide} -> ${c.workerMB.reach}; boardings ${Math.round(c.lines.boardings[0])} vs ${Math.round(c.lines.boardings[1])} (${pct(c.lines.boardings[2])}), worst total ${pct(c.worst)}, worst line ${pct(c.lines.worstShare)} (line ${c.lines.worstLine}, ${Math.round(c.lines.worstAbs)} riders at most)`);
  if (!c.pass) process.exitCode = 1;
}

if (mode === 'capture') await capture(a1, a2);
else if (mode === 'compare') await compare(a1, a2);
else if (mode === 'solve') {
  // one run, no comparison: `solve <label> reach|wide`
  await compare(a1, a2 ?? 'reach');
} else { console.error('usage: capture <save> <label> | compare <label> [wide|reach|both]'); process.exit(2); }
void a3;
