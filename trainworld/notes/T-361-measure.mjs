// T-361: one load of the real save at N solving workers: startup (no cache) to first estimate and
// final, main-thread long tasks while loading, WASM memory per worker, Chrome process memory, then
// one edit (a line's train count) and its re-solve.
//   MODE=prepare|run  BASE=<url>  OUT=<dir>  N=<workers>  LABEL=<name>  EXTRA=<more edits>  PLAYWRIGHT_MODULE=...
// BASE: a scratch build (`vite build --minify false --outDir <scratch>`) served by checks/static-serve.mjs.
// OUT holds the persistent profile (run MODE=prepare once to store the real save) and run-<LABEL>.json.
import { readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6161/';
const MODE = process.env.MODE ?? 'run';
const N = process.env.N ?? '5';
const LABEL = process.env.LABEL ?? `w${N}`;
const SAVE = process.env.SAVE ?? 'C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save';
const PS = new URL('./T-361-procmem.ps1', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const profile = resolve(OUT, 'profile');
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });

await ctx.addInitScript(() => {
  window.__lt = [];
  window.__frames = [];
  const tick = (t) => { window.__frames.push(t); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([Math.round(e.startTime), Math.round(e.duration)]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
  // every worker message carrying WASM memory: [ms, worker #, kind, period, round, MB]
  window.__mem = [];
  const W = window.Worker;
  let wn = 0;
  window.Worker = class extends W {
    constructor(...a) {
      super(...a);
      const k = wn++, url = String(a[0]);
      this.addEventListener('message', (e) => {
        const m = e.data;
        if (m && typeof m.memBytes === 'number') window.__mem.push([Math.round(performance.now()), k, m.kind, m.period ?? -1, m.summary ? m.summary[0] : -1, Math.round(m.memBytes / 1048576), url.slice(-40)]);
      });
    }
  };
  window.__world = 0;
  const iv = setInterval(() => { if (window.tw?.world?.value) { window.__world = performance.now(); clearInterval(iv); } }, 50);
  // the "player": the map nudged 3 px and back every 50 ms once it exists (as t348-lag.mjs)
  let dir = 1;
  setInterval(() => { const m = window.tw?.map; if (!m) return; m.panBy([3 * dir, 0], { duration: 0 }); dir = -dir; }, 50);
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/openfreemap|AJAXError|favicon/.test(m.text())) errors.push(m.text()); });

const waitSolved = (timeout = 300000) => page.waitForFunction(() => {
  const d = window.twDemand, s = d?.log?.at?.(-1) ?? d?.log?.[0], v = d?.view?.();
  return s && v && v.final && !d.client.waiting && d.client.cur && d.client.cur.published === 3;
}, undefined, { timeout, polling: 200 });
const procMem = () => {
  try { const j = JSON.parse(execFileSync('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', PS, '-Profile', profile], { encoding: 'utf8' })); return Array.isArray(j) ? j : [j]; }
  catch (e) { return [{ error: String(e).slice(0, 200) }]; }
};
const workerMem = () => page.evaluate(() => {
  const c = window.twDemand.client;
  return { computeN: c.computeN, start: c.start.workers.map((w) => w && Math.round(w.memBytes / 1048576)), solve: (c.cur?.memBytes ?? []).map((b) => b && Math.round(b / 1048576)) };
});
const lt = (from, to) => page.evaluate(([from, to]) => {
  const l = window.__lt.filter(([s]) => s >= from && s <= to);
  const f = window.__frames.filter((t) => t >= from && t <= to);
  let over = 0; for (let i = 1; i < f.length; i++) { const g = f[i] - f[i - 1]; if (g > 50) over += g; }
  return { n: l.length, ms: l.reduce((a, [, d]) => a + d, 0), max: l.reduce((a, [, d]) => Math.max(a, d), 0), fps: +(f.length / ((to - from) / 1000)).toFixed(1), gapMs: Math.round(over) };
}, [from, to]);

const out = { label: LABEL, n: N, base: BASE, errors };
try {
  if (MODE === 'prepare') {
    await page.goto(`${BASE}?debug=1&demandWorkers=${N}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile(SAVE);
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await waitSolved();
    log('prepared');
  } else {
    // perfOff=nocache: an unknown flag, so the T-328 cache key misses and every load solves fresh
    await page.goto(`${BASE}?debug=1&demandWorkers=${N}&perfOff=nocache&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved();
    const s0 = await page.evaluate(() => { const c = window.twDemand.client, s = c.cur; return { v: s.version, sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), final: s.refinedAt && Math.round(s.refinedAt), world: Math.round(window.__world), now: performance.now(), periodMs: s.periodMs, setupMs: s.setupMs }; });
    out.startup = s0;
    out.loadingLT = await lt(s0.world, s0.final ?? s0.now);
    out.memAfterStartup = await workerMem();
    await page.waitForTimeout(3000);
    out.procAfterStartup = procMem();
    // the edit: one more train at each level on a running line with trains
    const e = await page.evaluate(async () => {
      const tw = window.tw, d = window.twDemand, c = d.client;
      const w = tw.world.value, stats = w.lineStats instanceof Map ? (id) => w.lineStats.get(id) : (id) => w.lineStats?.[id];
      const l = w.save.lines.find((x) => stats(x.id)?.status === 'running' && stats(x.id).trainsNeeded[0] > 0);
      const trains = stats(l.id).trainsNeeded.map((t) => t + 2);
      const v0 = c.cur.version;
      const t0 = performance.now();
      const r = await tw.client.edit({ op: 'setTrainCount', line: l.num, trains });
      return { line: l.id, name: l.name, trains, ok: r?.ok, v0, t0 };
    });
    await page.waitForFunction((v0) => window.twDemand.client.cur?.version > v0, e.v0, { timeout: 60000, polling: 50 });
    await waitSolved();
    const s1 = await page.evaluate(() => { const s = window.twDemand.client.cur; return { v: s.version, sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), final: s.refinedAt && Math.round(s.refinedAt), periodMs: s.periodMs }; });
    out.edit = { ...e, ...s1, firstAfterEdit: s1.first && Math.round(s1.first - e.t0), finalAfterEdit: s1.final && Math.round(s1.final - e.t0) };
    out.editLT = await lt(e.t0, s1.final);
    out.memAfterEdit = await workerMem();
    out.procAfterEdit = procMem();
    // EXTRA=k: k more edits (alternating +2 / back), WASM memory after each, to see whether it creeps
    out.extra = [];
    for (let i = 0; i < Number(process.env.EXTRA ?? 0); i++) {
      const v0 = await page.evaluate(async ([line, trains, i]) => { const v0 = window.twDemand.client.cur.version; await window.tw.client.edit({ op: 'setTrainCount', line: Number(line), trains: trains.map((t) => t + (i % 2 ? 0 : 1)) }); return v0; }, [e.line, e.trains, i]);
      await page.waitForFunction((v0) => window.twDemand.client.cur?.version > v0, v0, { timeout: 60000, polling: 50 });
      await waitSolved();
      out.extra.push(await workerMem());
    }
    if (out.extra.length) { out.procAfterExtra = procMem(); log('extra', JSON.stringify(out.extra.map((m) => m.solve))); }
    out.memTimeline = await page.evaluate(() => window.__mem);
    log(LABEL, 'first', s0.first, 'final', s0.final, 'lt', JSON.stringify(out.loadingLT), 'edit first', out.edit.firstAfterEdit, 'final', out.edit.finalAfterEdit, 'mem', JSON.stringify(out.memAfterEdit));
  }
} finally {
  await ctx.close();
}
await writeFile(join(OUT, `${MODE}-${LABEL}.json`), JSON.stringify(out, null, 1));
log('written', `${MODE}-${LABEL}.json`, 'errors', errors.slice(0, 5));
