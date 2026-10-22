// T-714: frame time while panning the real save (New York, 109 lines), with the clock running.
// Fails when the main thread's time per frame, or the slow frames, go over the budget below.
//   node trainworld/checks/t714-pan.mjs [url]      (from the maps root)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port PAN_PORT or 6192) and stops it at the
// end. Headless Chrome with a throwaway profile; Playwright as in smoke.mjs. About two minutes.
// Env: LABEL (output name), OUT (default ../artifacts/trainworld/t714), CPUPROFILE=1 (a .cpuprofile per
// pan), NOSOLVE=1 (pan without waiting for demand's final), PAN_REPS (drags per scene, default 2),
// THROTTLE=4 (the main thread 4x slower), TRACE=1 (busy ms a frame on the renderer, GPU and viz
// threads), VIEW=WxH and DPR, DISPLAY (settings JSON), SETUP and PROBE (page expressions), SPEED, HOUR.
// The overlay's GPU time comes from renderer.ts's timer query. Two still scenes (camera at rest,
// clock running) are reported but not budgeted.
//
// Each scene: a jump, 2.5 s for tiles, then real mouse drags (press, 8 px per 16 ms one way and back)
// or wheel zooms. Per scene: frames from requestAnimationFrame, the main thread's busy time from
// CDP Performance.getMetrics (TaskDuration) over the frames drawn, long tasks, the overlay's own draw.
import { spawn } from 'node:child_process';
import { readFile, readdir, mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = fileURLToPath(new URL('.', import.meta.url));
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t714/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const REPS = Number(process.env.PAN_REPS ?? 2);
const [W, H] = (process.env.VIEW ?? '1400x900').split('x').map(Number);
const DPR = Number(process.env.DPR ?? 1);
/** The budget, over the drag and zoom scenes together: main-thread ms per drawn frame, and the share
 * of frame intervals over 50 ms. A 60 Hz frame is 16.7 ms; every build of 2026-10-18/19 measured
 * 4.7-5.7 ms and 0-0.1% on Anita's machine (notes/perf-survey.md, T-714). */
const BUDGET = { mainMsPerFrame: Number(process.env.BUDGET_MS ?? 12), slowShare: Number(process.env.BUDGET_SLOW ?? 0.05) };

const SCENES = [
  { name: 'Manhattan z12', at: [-73.975, 40.76], zoom: 12, drag: 'pan' },
  { name: 'Manhattan z14', at: [-73.985, 40.75], zoom: 14, drag: 'pan' },
  { name: 'Midtown z16', at: [-73.988, 40.752], zoom: 16, drag: 'pan' },
  { name: 'Brooklyn z12', at: [-73.95, 40.66], zoom: 12, drag: 'pan' },
  { name: 'Brooklyn z14', at: [-73.985, 40.69], zoom: 14, drag: 'pan' },
  { name: 'Downtown Brooklyn z16', at: [-73.988, 40.692], zoom: 16, drag: 'pan' },
  { name: 'zoom 11-16 and back', at: [-73.98, 40.74], zoom: 11, drag: 'wheel' },
  // the camera still, the clock running: what playing costs between pans (not in the budget)
  { name: 'still z12', at: [-73.975, 40.76], zoom: 12, drag: 'still' },
  { name: 'still z15', at: [-73.985, 40.75], zoom: 15, drag: 'still' },
];

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
  const port = Number(process.env.PAN_PORT ?? 6192);
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

const TRACE = process.env.TRACE === '1';

/** TRACE=1: stop the trace and sum the top-level tasks per thread (renderer main, GPU main, viz), ms */
async function traceThreads(cdp) {
  const done = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  await cdp.send('Tracing.end');
  const { stream } = await done;
  let text = '';
  for (;;) {
    const { data, eof, base64Encoded } = await cdp.send('IO.read', { handle: stream, size: 1 << 22 });
    text += base64Encoded ? Buffer.from(data, 'base64').toString() : data;
    if (eof) break;
  }
  await cdp.send('IO.close', { handle: stream });
  const ev = JSON.parse(text).traceEvents ?? [];
  const names = new Map();
  for (const e of ev) if (e.ph === 'M' && e.name === 'thread_name') names.set(`${e.pid}:${e.tid}`, e.args.name);
  const byThread = new Map();
  for (const e of ev) {
    if (e.ph !== 'X' || !/RunTask/.test(e.name)) continue;
    const k = `${e.pid}:${e.tid}`;
    if (!byThread.has(k)) byThread.set(k, []);
    byThread.get(k).push(e);
  }
  const out = {};
  for (const [k, list] of byThread) {
    const name = names.get(k) ?? '?';
    if (!/CrRendererMain|CrGpuMain|VizCompositorThread|Compositor$/.test(name)) continue;
    list.sort((a, b) => a.ts - b.ts);
    let end = -Infinity, ms = 0;
    for (const e of list) { if (e.ts >= end) { ms += (e.dur ?? 0) / 1000; end = e.ts + (e.dur ?? 0); } }
    out[name] = Math.max(out[name] ?? 0, ms);
  }
  return out;
}

const pct = (a, p) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const r1 = (x) => Math.round(x * 10) / 10;

async function run(BASE) {
  const { chromium } = await playwright();
  const profile = await mkdtemp(join(tmpdir(), 'tw-pan-'));
  const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: W, height: H }, deviceScaleFactor: DPR, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
  const errors = [];
  const out = { label: LABEL, base: BASE, budget: BUDGET, scenes: [], errors };
  await mkdir(OUT, { recursive: true });
  try {
    const page = await ctx.newPage();
    // DISPLAY: more display settings as JSON, e.g. '{"realNetwork":true}'
    await page.addInitScript((more) => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', ...more }, v: 2 })); } catch { /* */ }
      window.__frames = [];
      // the overlay's GPU time (renderer.ts's timer query), where the browser has the timer
      window.__gpu = [];
      const tick = (t) => { window.__frames.push(t); const g = window.tw?.renderer?.gpuMs; if (g != null) window.__gpu.push([t, g]); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      window.__lt = [];
      window.__fontLoads = 0;
      document.fonts?.addEventListener?.('loadingdone', () => window.__fontLoads++);
      try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
    }, JSON.parse(process.env.DISPLAY ?? '{}'));
    page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
    // maplibre logs a failed tile fetch as its minified error object, whose text is a class name
    page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError|noritetsu/.test(m.text()) && !/maplibre-gl/.test(m.location()?.url ?? '')) errors.push(m.text() + ' @ ' + (m.location()?.url ?? '')); });
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    const sep = BASE.includes('?') ? '&' : '?';
    await page.goto(`${BASE}${sep}debug=1&new=1&opened=all&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    out.gpu = await page.evaluate(() => { const gl = document.createElement('canvas').getContext('webgl2'); const e = gl?.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : '?'; });
    const bytes = await readFile(SAVE);
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 120000 });
    if (process.env.NOSOLVE !== '1') {
      await page.waitForFunction(() => {
        const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
        return s && v && v.version === s.version && v.final && !d.client.waiting;
      }, undefined, { timeout: 300000, polling: 250 }).catch(() => errors.push('demand never reached its final; panned anyway'));
    }
    await page.waitForTimeout(2000);
    const metric = async () => {
      const { metrics } = await cdp.send('Performance.getMetrics');
      const m = Object.fromEntries(metrics.map((x) => [x.name, x.value]));
      return { task: m.TaskDuration * 1000, script: m.ScriptDuration * 1000, layout: m.LayoutDuration * 1000, style: m.RecalcStyleDuration * 1000 };
    };
    const cx = W / 2, cy = H / 2;
    // SETUP: a JS expression awaited in the page before the scenes (diagnostics: a tool, a selection)
    if (process.env.SETUP) console.log('setup', await page.evaluate((x) => Promise.resolve((0, eval)(x)).then((v) => JSON.stringify(v ?? null)), process.env.SETUP));
    // SPEED=0|1|2: the clock's speed (60x, 240x, 960x) while panning; HOUR: the game hour to pan at
    if (process.env.SPEED) await page.evaluate((i) => window.tw.clock.setSpeed(i), Number(process.env.SPEED));
    if (process.env.HOUR) await page.evaluate((h) => window.tw.clock.set(Math.floor(window.tw.clock.now() / 86400) * 86400 + h * 3600), Number(process.env.HOUR));
    // THROTTLE=4: a slower machine's main thread, so a cost shows as dropped frames
    if (process.env.THROTTLE) await cdp.send('Emulation.setCPUThrottlingRate', { rate: Number(process.env.THROTTLE) });
    for (const sc of SCENES) {
      // braces: jumpTo returns the map, which Playwright would serialise for a second or more
      await page.evaluate(([at, zoom]) => { window.tw.map.jumpTo({ center: at, zoom }); }, [sc.at, sc.zoom]);
      await page.waitForTimeout(2500);
      if (process.env.CPUPROFILE === '1') { await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 200 }); await cdp.send('Profiler.start'); }
      if (TRACE) await cdp.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { includedCategories: ['toplevel', 'disabled-by-default-gpu.service', 'gpu', 'viz', '__metadata'], recordMode: 'recordContinuously' } });
      await page.evaluate(() => window.tw.reset?.());
      const m0 = await metric();
      const t0 = await page.evaluate(() => performance.now());
      await page.mouse.move(cx, cy);
      if (sc.drag === 'pan') {
        for (let rep = 0; rep < REPS; rep++) {
          for (const dir of [1, -1]) {
            await page.mouse.move(cx - dir * 200, cy - dir * 60);
            await page.mouse.down();
            for (let i = 1; i <= 50; i++) {
              await page.mouse.move(cx - dir * 200 + dir * 8 * i, cy - dir * 60 + dir * 2.4 * i);
              await page.waitForTimeout(16);
            }
            await page.mouse.up();
            await page.waitForTimeout(150);
          }
        }
      } else if (sc.drag === 'still') {
        await page.waitForTimeout(6000);
      } else {
        for (let rep = 0; rep < REPS; rep++) {
          for (const dy of [-100, 100]) {
            for (let i = 0; i < 25; i++) { await page.mouse.wheel(0, dy); await page.waitForTimeout(40); }
            await page.waitForTimeout(400);
          }
        }
      }
      const t1 = await page.evaluate(() => performance.now());
      const m1 = await metric();
      const threads = TRACE ? await traceThreads(cdp) : null;
      if (process.env.CPUPROFILE === '1') {
        const { profile: prof } = await cdp.send('Profiler.stop');
        await writeFile(join(OUT, `pan-${LABEL}-${sc.name.replace(/\W+/g, '_')}.cpuprofile`), JSON.stringify(prof));
      }
      const st = await page.evaluate(([t0, t1]) => {
        const f = window.__frames.filter((t) => t >= t0 && t <= t1);
        const iv = [];
        for (let i = 1; i < f.length; i++) iv.push(f[i] - f[i - 1]);
        const lt = window.__lt.filter(([s]) => s >= t0 && s <= t1);
        const g = window.__gpu.filter(([t]) => t >= t0 && t <= t1).map(([, v]) => v);
        return { iv, longTasks: lt.length, longMs: lt.reduce((a, [, d]) => a + d, 0), overlay: window.tw.stats?.(), gpu: g.length ? g.reduce((a, b) => a + b, 0) / g.length : null };
      }, [t0, t1]);
      const frames = st.iv.length + 1;
      const s = {
        name: sc.name, secs: r1((t1 - t0) / 1000), frames, fps: r1(frames / ((t1 - t0) / 1000)),
        mainMsPerFrame: r1((m1.task - m0.task) / frames), scriptMsPerFrame: r1((m1.script - m0.script) / frames),
        layoutMs: Math.round(m1.layout - m0.layout), styleMs: Math.round(m1.style - m0.style),
        p50: r1(pct(st.iv, 0.5)), p95: r1(pct(st.iv, 0.95)), max: Math.round(Math.max(0, ...st.iv)),
        over50: st.iv.filter((v) => v > 50).length, longTasks: st.longTasks, longMs: Math.round(st.longMs),
        overlayDrawMs: st.overlay ? r1(st.overlay.drawMs) : null, mapFrames: st.overlay?.mapFrames ?? null, overlayGpuMs: st.gpu === null ? null : Math.round(st.gpu * 100) / 100,
        mainMs: m1.task - m0.task, intervals: st.iv.length, still: sc.drag === 'still',
        mainCorePct: Math.round((100 * (m1.task - m0.task)) / (t1 - t0)),
      };
      if (threads) s.threadMsPerFrame = Object.fromEntries(Object.entries(threads).map(([k, v]) => [k, r1(v / frames)]));
      // PROBE: a JS expression evaluated in the page after each scene, logged (diagnostics)
      if (process.env.PROBE) s.probe = await page.evaluate((x) => { try { return JSON.stringify((0, eval)(x)); } catch (e) { return 'probe error: ' + e.message; } }, process.env.PROBE);
      if (s.probe) console.log('  probe', s.probe);
      out.scenes.push(s);
      console.log(`${sc.name.padEnd(24)} fps ${String(s.fps).padStart(5)}  main ${String(s.mainMsPerFrame).padStart(5)} ms/frame (script ${s.scriptMsPerFrame}, ${s.mainCorePct}% of a core)  p50 ${s.p50}  p95 ${s.p95}  max ${s.max}  >50ms ${s.over50}  long ${s.longTasks}/${s.longMs} ms  overlay ${s.overlayDrawMs} ms, gpu ${s.overlayGpuMs ?? '-'} ms${threads ? '  threads ' + JSON.stringify(s.threadMsPerFrame) : ''}`);
    }
  } finally {
    await ctx.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
  return out;
}

const given = process.argv[2];
const vite = given ? null : await startVite();
let out;
try {
  out = await run(given ?? vite.url);
} finally {
  if (vite) vite.child.kill();
}
const drags = out.scenes.filter((s) => !s.still);
const frames = drags.reduce((a, s) => a + s.frames, 0);
const mainMsPerFrame = drags.reduce((a, s) => a + s.mainMs, 0) / frames;
const slowShare = drags.reduce((a, s) => a + s.over50, 0) / drags.reduce((a, s) => a + s.intervals, 0);
out.total = { frames, mainMsPerFrame: r1(mainMsPerFrame), slowShare: Math.round(slowShare * 1000) / 1000 };
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `pan-${LABEL}.json`), JSON.stringify(out, null, 1));
const ok = mainMsPerFrame <= BUDGET.mainMsPerFrame && slowShare <= BUDGET.slowShare && !out.errors.length;
console.log(`${ok ? 'PASS' : 'FAIL'}  pan frame time  (${r1(mainMsPerFrame)} ms main thread a frame, budget ${BUDGET.mainMsPerFrame}; ${(100 * slowShare).toFixed(1)}% of frames over 50 ms, budget ${100 * BUDGET.slowShare}%; ${frames} frames${out.errors.length ? '; errors: ' + out.errors.slice(0, 3).join(' | ') : ''}; gpu ${out.gpu})`);
process.exit(ok ? 0 : 1);
