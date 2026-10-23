// T-737: Japan's whole real network (709 lines, 10,171 stations from N02; notes/T-737.md) loaded and
// played at 240x and 960x: load time, demand's first solve (its times and worker memory), frame times
// while panning and zooming across Japan, main-thread long tasks, the threads' busy time.
//   node trainworld/checks/t737-japan.mjs [url]      (from the maps root)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port JP_PORT or 6165). About 4 minutes,
// QUICK=1 (one speed, three scenes) about 1.5. The save is ../artifacts/trainworld/t737-japan.save (SAVE
// to override; t737-japan-cities.save is the seven city packs only; built by `pipeline/realnet.py net jpn`,
// `geom jpn`, the realsave bin with --city jpn, `gamesave jpn`, notes/T-737.md). Env: LABEL, OUT (default
// ../artifacts/trainworld/t737), VIEW=WxH, SPEEDS (default "1,2": indices into the clock's SPEEDS),
// THROTTLE, CPUPROFILE=1 (the main thread's profile per scene in OUT), DEMAND_WORKERS (default 2), NOSOLVE=1 (play at once; the clock is held while demand solves,
// so this measures panning during a solve). Before the scenes it waits up to 35 min for demand's final
// (the clock runs only then) or a demand worker's failure: whole Japan's final comes in about 19 minutes
// with 3.2-3.4 GB a worker (T-761), the seven cities' in about 6.5 (T-756). A demand failure is reported
// as `demandErrors`, not failing the check; DEMAND_STRICT=1 fails on it. Fails on page errors, a load
// that never ends, and the budget below.
import { spawn } from 'node:child_process';
import { readFile, readdir, mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const SAVE = process.env.SAVE ?? fileURLToPath(new URL('../../artifacts/trainworld/t737-japan.save', import.meta.url));
const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t737/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const [W, H] = (process.env.VIEW ?? '1400x900').split('x').map(Number);
const SPEED_IX = (process.env.SPEEDS ?? (process.env.QUICK === '1' ? '1' : '1,2')).split(',').map(Number);
/** main-thread ms per drawn frame over the moving scenes, and the share of frames over 50 ms */
const BUDGET = { mainMsPerFrame: Number(process.env.BUDGET_MS ?? 14), slowShare: Number(process.env.BUDGET_SLOW ?? 0.05) };

const ALL = [
  { name: 'Tokyo z11 still', at: [139.75, 35.68], zoom: 11, drag: 'still' },
  { name: 'Tokyo z12 pan', at: [139.73, 35.69], zoom: 12, drag: 'pan' },
  { name: 'Tokyo z14 pan', at: [139.70, 35.69], zoom: 14, drag: 'pan' },
  { name: 'Osaka z11 pan', at: [135.50, 34.69], zoom: 11, drag: 'pan' },
  { name: 'Japan z6 still', at: [137.5, 36.5], zoom: 5.6, drag: 'still' },
  { name: 'Japan z6 pan', at: [137.5, 36.5], zoom: 5.6, drag: 'pan' },
  { name: 'Kansai z8 pan', at: [135.6, 34.8], zoom: 8, drag: 'pan' },
  { name: 'Nagoya zoom 6-12', at: [136.88, 35.17], zoom: 6, drag: 'wheel' },
];
const SCENES = process.env.QUICK === '1' ? ALL.filter((s) => /Tokyo z12|Japan z6 pan|zoom/.test(s.name)) : ALL;

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

async function startVite() {
  const port = Number(process.env.JP_PORT ?? 6165);
  const child = spawn(process.execPath, [join(HERE, 'dev-nowatch.mjs'), String(port)], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '';
  child.stdout.on('data', (d) => { out += d; });
  child.stderr.on('data', (d) => { out += d; });
  const deadline = Date.now() + 60000;
  while (!/localhost:/.test(out)) {
    if (child.exitCode !== null || Date.now() > deadline) throw new Error('Vite did not start: ' + out.slice(-500));
    await new Promise((r) => setTimeout(r, 200));
  }
  return { child, url: `http://localhost:${port}/` };
}

/** stop a trace and sum the top-level tasks per thread name (ms, overlaps merged) */
async function traceThreads(cdp) {
  const done = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  await cdp.send('Tracing.end');
  const { stream } = await done;
  const chunks = [];
  for (;;) {
    const { data, eof, base64Encoded } = await cdp.send('IO.read', { handle: stream, size: 1 << 22 });
    chunks.push(base64Encoded ? Buffer.from(data, 'base64').toString() : data);
    if (eof) break;
  }
  await cdp.send('IO.close', { handle: stream });
  const ev = JSON.parse(chunks.join('')).traceEvents ?? [];
  const names = new Map();
  for (const e of ev) if (e.ph === 'M' && e.name === 'thread_name') names.set(`${e.pid}:${e.tid}`, e.args.name);
  const by = new Map();
  for (const e of ev) {
    if (e.ph !== 'X' || !/RunTask|ThreadControllerImpl::RunTask/.test(e.name)) continue;
    const k = `${e.pid}:${e.tid}`;
    if (!by.has(k)) by.set(k, []);
    by.get(k).push(e);
  }
  const out = {};
  for (const [k, list] of by) {
    const name = names.get(k) ?? '?';
    if (!/CrRendererMain|CrGpuMain|VizCompositorThread|DedicatedWorker/.test(name)) continue;
    list.sort((a, b) => a.ts - b.ts);
    let end = -Infinity, ms = 0, max = 0;
    for (const e of list) { if (e.ts >= end) { ms += (e.dur ?? 0) / 1000; max = Math.max(max, (e.dur ?? 0) / 1000); end = e.ts + (e.dur ?? 0); } }
    (out[name] ??= []).push({ ms: Math.round(ms), max: Math.round(max) });
  }
  for (const k of Object.keys(out)) out[k].sort((a, b) => b.ms - a.ms);
  return out;
}

const pct = (a, p) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const r1 = (x) => Math.round(x * 10) / 10;
const mb = (b) => Math.round(b / 1048576);
const log = (...a) => console.log(...a);

async function run(BASE) {
  const { chromium } = await playwright();
  const profile = await mkdtemp(join(tmpdir(), 'tw-t737-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: W, height: H }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-precise-memory-info'] });
  const errors = [];
  const out = { label: LABEL, base: BASE, budget: BUDGET, view: [W, H], errors, scenes: [] };
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
      window.__frames = [];
      const tick = (t) => { window.__frames.push(t); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      window.__lt = [];
      try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
      // every clock message: kind, arrival, main-thread ms handling it, the worker's own ms where given
      window.__msgs = [];
      const hook = setInterval(() => {
        const c = window.tw?.client;
        if (!c || c.__hooked) return;
        c.__hooked = true;
        clearInterval(hook);
        const orig = c.receive.bind(c);
        c.receive = (m) => {
          const t0 = performance.now();
          orig(m);
          const rec = { k: m.kind, t: t0, ms: performance.now() - t0 };
          if (m.kind === 'state') { rec.w = m.state.ms; rec.service = !!m.state.service; }
          if (m.kind === 'trips') { rec.n = m.trips.length; rec.prep = m.prepMs; rec.board = m.boardMs; }
          if (m.kind === 'loaded') rec.error = m.error;
          window.__msgs.push(rec);
        };
      }, 5);
    });
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    // a bare minified class name ("Nn") is maplibre logging a failed tile fetch as its error object
    page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError|noritetsu|^\w{1,3}$/.test(m.text()) && !/maplibre-gl/.test(m.location()?.url ?? '')) errors.push(m.text().slice(0, 300) + ' @ ' + (m.location()?.url ?? '').replace(/.*\//, '')); });
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    const metric = async () => {
      const { metrics } = await cdp.send('Performance.getMetrics');
      const m = Object.fromEntries(metrics.map((x) => [x.name, x.value]));
      return { task: m.TaskDuration * 1000, script: m.ScriptDuration * 1000, heap: m.JSHeapUsedSize };
    };
    const sep = BASE.includes('?') ? '&' : '?';
    // two solving workers, the target machine's (notes/perf-survey.md)
    await page.goto(`${BASE}${sep}debug=1&new=1&opened=all&demandWorkers=${process.env.DEMAND_WORKERS ?? 2}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.client.__hooked, undefined, { timeout: 180000 });
    out.gpu = await page.evaluate(() => { const gl = document.createElement('canvas').getContext('webgl2'); const e = gl?.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : '?'; });
    out.cores = await page.evaluate(() => navigator.hardwareConcurrency);
    await page.waitForTimeout(3000);

    // ---- load the save
    const bytes = await readFile(SAVE);
    await page.evaluate(([at, zoom]) => void window.tw.map.jumpTo({ center: at, zoom }), [[139.75, 35.68], 10]);
    const tImport = await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; const t = performance.now(); window.tw.client.post({ kind: 'import', bytes: b }, [b]); return t; }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 400, undefined, { timeout: 300000, polling: 100 });
    const load = await page.evaluate((t0) => {
      const m = window.__msgs.filter((x) => x.t >= t0);
      const loaded = m.find((x) => x.k === 'loaded');
      const state = m.find((x) => x.k === 'state');
      const lt = window.__lt.filter(([s]) => s >= t0);
      const w = window.tw.world.value.save;
      return { loadedMs: loaded && Math.round(loaded.t - t0), loadError: loaded?.error ?? null, stateMs: state && Math.round(state.t - t0), stateMainMs: state && Math.round(state.ms), stateWorkerMs: state?.w,
        linesMs: Math.round(performance.now() - t0), lines: w.lines.length, stations: w.stations?.length, longTasks: lt.length, longMax: Math.round(Math.max(0, ...lt.map(([, d]) => d))) };
    }, tImport);
    out.load = load;
    log('load', JSON.stringify(load));

    // ---- demand
    const tD = await page.evaluate(() => performance.now());
    const samples = [];
    let final = false;
    const deadline = Date.now() + 35 * 60000;
    if (process.env.NOSOLVE !== '1') {
      while (Date.now() < deadline) {
        const s = await page.evaluate(() => {
          const d = window.twDemand, l = d?.log?.at(-1), v = d?.view?.();
          return { t: Math.round(performance.now()), final: !!(l && v && v.version === l.version && v.final && !d.client.waiting), round: v?.round ?? null,
            workers: (d.start?.workers ?? []).map((w) => Math.round((w.memBytes ?? 0) / 1048576)), opened: [...(d.client.opened ?? [])].length,
            solveMem: (l?.memBytes ?? []).map((b) => Math.round(b / 1048576)), heap: Math.round((performance.memory?.usedJSHeapSize ?? 0) / 1048576) };
        });
        samples.push(s);
        if (s.final) { final = true; break; }
        if (errors.some((e) => /demand: worker/.test(e))) break;
        if (samples.length % 15 === 0) log(`  demand at ${Math.round((s.t - tD) / 1000)} s: round ${s.round}, solve MB ${s.solveMem}, opened ${s.opened}`);
        await page.waitForTimeout(2000);
      }
      if (!final) errors.push('demand never reached its final');
    }
    out.demand = await page.evaluate((t0) => {
      const d = window.twDemand;
      const lt = window.__lt.filter(([s]) => s >= t0);
      return {
        computeN: d.client.computeN,
        workers: (d.start?.workers ?? []).map((w) => ({ wasmMs: Math.round(w.wasmMs ?? 0), openMs: Math.round(w.openMs ?? 0), memMB: Math.round((w.memBytes ?? 0) / 1048576) })),
        opened: [...(d.client.opened ?? [])],
        loads: (d.start?.loads ?? []).length,
        solves: d.log.map((s) => ({ v: s.version, cities: s.cities.length, arrived: Math.round(s.arrivedAt - t0), first: s.firstAt && Math.round(s.firstAt - t0), refined: s.refinedAt && Math.round(s.refinedAt - t0),
          published: s.published && Math.round(s.published - t0), periodMs: s.periodMs.map((p) => p.map((x) => Math.round(x))), setupMs: s.setupMs.map((x) => Math.round(x)),
          memMB: s.memBytes.map((b) => Math.round(b / 1048576)), lines: s.svc.lines.length, stations: s.svc.stations.length })),
        longTasks: lt.length, longMs: Math.round(lt.reduce((a, [, x]) => a + x, 0)), longMax: Math.round(Math.max(0, ...lt.map(([, x]) => x))),
        finalAt: Math.round(performance.now() - t0),
      };
    }, tD);
    out.demand.samples = samples;
    log('demand', JSON.stringify({ ...out.demand, samples: undefined, solves: out.demand.solves.slice(-2) }));

    // ---- play
    if (process.env.THROTTLE) await cdp.send('Emulation.setCPUThrottlingRate', { rate: Number(process.env.THROTTLE) });
    const cx = W / 2, cy = H / 2;
    for (const speed of SPEED_IX) {
      await page.evaluate((i) => { window.tw.clock.setSpeed(i); window.tw.clock.running.value = true; }, speed);
      for (const sc of SCENES) {
        await page.evaluate(([at, zoom]) => void window.tw.map.jumpTo({ center: at, zoom }), [sc.at, sc.zoom]);
        await page.waitForTimeout(2500);
        await cdp.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { includedCategories: ['toplevel', '__metadata'], recordMode: 'recordContinuously' } });
        if (process.env.CPUPROFILE === '1') { await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start'); }
        const m0 = await metric();
        const t0 = await page.evaluate(() => performance.now());
        const g0 = await page.evaluate(() => window.tw.clock.now());
        await page.mouse.move(cx, cy);
        if (sc.drag === 'pan') {
          for (let rep = 0; rep < 2; rep++) for (const dir of [1, -1]) {
            await page.mouse.move(cx - dir * 200, cy - dir * 60);
            await page.mouse.down();
            for (let i = 1; i <= 50; i++) { await page.mouse.move(cx - dir * 200 + dir * 8 * i, cy - dir * 60 + dir * 2.4 * i); await page.waitForTimeout(16); }
            await page.mouse.up();
            await page.waitForTimeout(150);
          }
        } else if (sc.drag === 'still') {
          await page.waitForTimeout(8000);
        } else {
          for (const dy of [-100, 100]) { for (let i = 0; i < 30; i++) { await page.mouse.wheel(0, dy); await page.waitForTimeout(40); } await page.waitForTimeout(400); }
        }
        const t1 = await page.evaluate(() => performance.now());
        const g1 = await page.evaluate(() => window.tw.clock.now());
        const m1 = await metric();
        const threads = await traceThreads(cdp);
        if (process.env.CPUPROFILE === '1') { const { profile: prof } = await cdp.send('Profiler.stop'); await mkdir(OUT, { recursive: true }); await writeFile(join(OUT, `main-${LABEL}-${speed}-${sc.name.replace(/\W+/g, '_')}.cpuprofile`), JSON.stringify(prof)); }
        const st = await page.evaluate(([t0, t1]) => {
          const f = window.__frames.filter((t) => t >= t0 && t <= t1);
          const iv = [];
          for (let i = 1; i < f.length; i++) iv.push(f[i] - f[i - 1]);
          const lt = window.__lt.filter(([s]) => s >= t0 && s <= t1);
          const ms = window.__msgs.filter((x) => x.t >= t0 && x.t <= t1);
          const by = {};
          for (const x of ms) { const b = (by[x.k] ??= { n: 0, mainMs: 0, mainMax: 0 }); b.n++; b.mainMs += x.ms; b.mainMax = Math.max(b.mainMax, x.ms); if (x.prep != null) b.prepMax = Math.max(b.prepMax ?? 0, x.prep); if (x.n != null) b.tripsMax = Math.max(b.tripsMax ?? 0, x.n); }
          for (const b of Object.values(by)) { b.mainMs = Math.round(b.mainMs); b.mainMax = Math.round(b.mainMax); }
          return { iv, longTasks: lt.length, longMs: lt.reduce((a, [, d]) => a + d, 0), longMax: Math.max(0, ...lt.map(([, d]) => d)), msgs: by };
        }, [t0, t1]);
        const frames = st.iv.length + 1;
        const secs = (t1 - t0) / 1000;
        const s = {
          speed, name: sc.name, secs: r1(secs), frames, fps: r1(frames / secs), gameMinPerSec: r1((g1 - g0) / 60 / secs),
          mainMsPerFrame: r1((m1.task - m0.task) / frames), mainCorePct: Math.round((100 * (m1.task - m0.task)) / (t1 - t0)),
          p50: r1(pct(st.iv, 0.5)), p95: r1(pct(st.iv, 0.95)), max: Math.round(Math.max(0, ...st.iv)), over50: st.iv.filter((v) => v > 50).length, intervals: st.iv.length,
          longTasks: st.longTasks, longMs: Math.round(st.longMs), longMax: Math.round(st.longMax), heapMB: mb(m1.heap), msgs: st.msgs,
          threadsPctOfWall: Object.fromEntries(Object.entries(threads).map(([k, v]) => [k, v.slice(0, 4).map((x) => `${Math.round((100 * x.ms) / (t1 - t0))}% (max ${x.max})`)])),
          mainMs: m1.task - m0.task, moving: sc.drag !== 'still',
        };
        out.scenes.push(s);
        log(`${SPEEDS_NAME[speed]} ${sc.name.padEnd(18)} fps ${String(s.fps).padStart(5)}  main ${String(s.mainMsPerFrame).padStart(5)} ms/frame (${s.mainCorePct}% core)  p95 ${s.p95}  max ${s.max}  >50ms ${s.over50}  long ${s.longTasks}/${s.longMs} ms (max ${s.longMax})  game ${s.gameMinPerSec} min/s  msgs ${JSON.stringify(s.msgs)}  threads ${JSON.stringify(s.threadsPctOfWall)}`);
      }
    }
    out.end = await page.evaluate(() => ({ heapMB: Math.round((performance.memory?.usedJSHeapSize ?? 0) / 1048576), workers: (window.twDemand.start?.workers ?? []).map((w) => Math.round((w.memBytes ?? 0) / 1048576)) }));
  } finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
  return out;
}
const SPEEDS_NAME = ['60x', '240x', '960x'];

const given = process.argv[2];
const vite = given ? null : await startVite();
let out;
try {
  out = await run(given ?? vite.url);
} finally {
  if (vite) vite.child.kill();
}
const moving = out.scenes.filter((s) => s.moving);
const frames = moving.reduce((a, s) => a + s.frames, 0);
const mainMsPerFrame = frames ? moving.reduce((a, s) => a + s.mainMs, 0) / frames : 0;
const slowShare = moving.reduce((a, s) => a + s.over50, 0) / Math.max(1, moving.reduce((a, s) => a + s.intervals, 0));
out.total = { frames, mainMsPerFrame: r1(mainMsPerFrame), slowShare: Math.round(slowShare * 1000) / 1000 };
if (process.env.DEMAND_STRICT !== '1') {
  out.demandErrors = out.errors.filter((e) => /demand: worker|demand never reached/.test(e));
  out.errors = out.errors.filter((e) => !/demand: worker|demand never reached/.test(e));
  if (out.demandErrors.length) log(`demand workers failed: ${out.demandErrors.slice(0, 2).join(' | ')}`);
}
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `japan-${LABEL}.json`), JSON.stringify(out, null, 1));
const ok =mainMsPerFrame <= BUDGET.mainMsPerFrame && slowShare <= BUDGET.slowShare && !out.errors.length;
log(`${ok ? 'PASS' : 'FAIL'}  Japan at speed  (${r1(mainMsPerFrame)} ms main thread a frame, budget ${BUDGET.mainMsPerFrame}; ${(100 * slowShare).toFixed(1)}% of frames over 50 ms, budget ${100 * BUDGET.slowShare}%; load ${out.load?.linesMs} ms; demand final ${out.demand?.finalAt} ms${out.errors.length ? '; errors: ' + out.errors.slice(0, 3).join(' | ') : ''})`);
process.exit(ok ? 0 : 1);
