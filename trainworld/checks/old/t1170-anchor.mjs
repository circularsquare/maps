// T-1170: what a move of the draw anchor costs. A save imported into a new game (paused, one demand
// worker), then the camera jumped between cities at zoom 12, each far enough to move the anchor: per
// jump the ms from the jump to the first frame drawn at the new anchor, the long tasks from the jump
// to 3 s after that frame (count, longest, total) and the clock worker's state and buffer ms for that state; then a
// track piece laid by the edit API right after a jump (the first click in a new city), jump to drawn.
// A CPU profile over the jumps goes to OUT. Env: SAVE (default the real New York save), CITIES
// ("lon,lat;lon,lat;..."), THROTTLE (CDP CPU rate, default 1), LABEL, OUT, LINES (lines the save has
// at least, default 50), SETTLE (s after the load before the first jump, default 25), QUERY (more URL parameters: `patch=shiftcheck` logs how far the drawn pieces
// moved to each new anchor are from a build from scratch, `shiftChecks` in the JSON; `patch=check`
// lays every anchor's pieces exactly, as before T-1170).
//   node trainworld/checks/old/t1170-anchor.mjs <page url>      (from the maps root; about 2 minutes, Japan 3)
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, collectErrors } from '../lib.mjs';

const SAVE = process.env.SAVE ?? fileURLToPath(new URL('../../T-007-real-nyc.save', import.meta.url));
const OUT = process.env.OUT ?? fileURLToPath(new URL('../../../artifacts/trainworld/t1170/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const LINES = Number(process.env.LINES ?? 50);
const THROTTLE = Number(process.env.THROTTLE ?? 1);
const CITIES = (process.env.CITIES ?? '-71.06,42.36;-75.16,39.95;-73.98,40.75;-71.06,42.36;-75.16,39.95').split(';').map((s) => s.split(',').map(Number));
const BASE = process.argv[2] ?? 'http://localhost:6163/';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const { ctx, close } = await launchChrome({ prefix: 'tw-t1170-' });
const errors = [];
const out = { label: LABEL, base: BASE, save: SAVE, throttle: THROTTLE, jumps: [], clicks: [], errors };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  // with QUERY=patch=shiftcheck: how far the pieces moved to each new anchor are from a build from scratch
  out.shiftChecks = [];
  page.on('console', (m) => { if (/T-1170 check/.test(m.text())) out.shiftChecks.push(m.text()); });
  await page.addInitScript(() => {
    window.__lt = [];
    try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
    // the draw anchor of the network drawn, by frame
    window.__anch = [];
    const tick = (t) => { const v = window.tw?.renderer?.net?.anchor?.version; const l = window.__anch.at(-1); if (v !== undefined && (!l || l[1] !== v)) window.__anch.push([t, v]); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    window.__st = [];
    const hook = setInterval(() => {
      const c = window.tw?.client;
      if (!c || c.__hooked) return;
      c.__hooked = true;
      clearInterval(hook);
      const orig = c.receive.bind(c);
      c.receive = (m) => { if (m?.kind === 'state') window.__st.push([performance.now(), m.state?.ms ?? m.ms ?? null, m.state?.net?.anchor?.version ?? null]); return orig(m); };
    }, 5);
  });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1${process.env.QUERY ? '&' + process.env.QUERY : ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.client?.__hooked, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction((n) => (window.tw.world.value?.save?.lines?.length ?? 0) >= n, LINES, { timeout: 300000, polling: 200 });
  const settle = Number(process.env.SETTLE ?? 25);
  log(`loaded; settling ${settle} s`);
  await page.waitForTimeout(settle * 1000);
  const cdp = await ctx.newCDPSession(page);
  if (THROTTLE > 1) await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
  await cdp.send('Profiler.start');
  const jump = async (lon, lat) => {
    const v0 = await page.evaluate(() => window.tw.renderer.net?.anchor?.version ?? -1);
    const t0 = await page.evaluate(([lon, lat]) => { const t = performance.now(); window.tw.map.jumpTo({ center: [lon, lat], zoom: 12 }); return t; }, [lon, lat]);
    await page.waitForFunction((v0) => (window.__anch.at(-1)?.[1] ?? -1) !== v0, v0, { timeout: 60000, polling: 50 });
    return t0;
  };
  const stats = (t0, ms) => page.evaluate(([t0, ms]) => {
    const lt = window.__lt.filter(([s]) => s >= t0 - 5 && s <= t0 + ms);
    const drawn = window.__anch.find(([t]) => t >= t0);
    const st = window.__st.filter(([t]) => t >= t0 && t <= t0 + ms);
    return { drawnMs: drawn ? Math.round(drawn[0] - t0) : null, longTasks: lt.length, longMax: Math.round(Math.max(0, ...lt.map(([, d]) => d))), longTotal: Math.round(lt.reduce((a, [, d]) => a + d, 0)), worker: st.map(([, ms]) => ms) };
  }, [t0, ms]);
  for (const [lon, lat] of CITIES) {
    const t0 = await jump(lon, lat);
    await page.waitForTimeout(3000);
    const s = await stats(t0, (await page.evaluate(() => performance.now())) - t0);
    out.jumps.push({ to: [lon, lat], ...s });
    log('jump', lon, lat, JSON.stringify(s));
  }
  // the first click in a new city: a piece of track asked the moment the camera lands
  for (const [lon, lat] of CITIES.slice(0, 2)) {
    const r = await page.evaluate(async ([lon, lat]) => {
      const tw = window.tw, v0 = tw.renderer.net?.anchor?.version;
      const t0 = performance.now();
      tw.map.jumpTo({ center: [lon, lat], zoom: 12 });
      await new Promise((d) => requestAnimationFrame(d));
      const p = tw.lngLatToPack(lon + 0.01, lat + 0.004), q = tw.lngLatToPack(lon + 0.02, lat + 0.006);
      const a = await tw.client.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
      const edited = performance.now() - t0;
      // drawn: the frame after a network holding the piece's edge at the new anchor
      const n0 = tw.world.value.edges.length;
      await new Promise((d) => { const f = () => (tw.renderer.net?.anchor?.version !== v0 && tw.world.value.edges.length >= n0 ? requestAnimationFrame(() => d()) : requestAnimationFrame(f)); f(); });
      return { ok: a.ok, editedMs: Math.round(edited), drawnMs: Math.round(performance.now() - t0), worker: Object.fromEntries(Object.entries(tw.client.editLog.at(-1)?.worker ?? {}).map(([k, v]) => [k, Math.round(v)])) };
    }, [lon + 0.3, lat + 0.3]);
    await page.waitForTimeout(3000);
    out.clicks.push({ at: [lon, lat], ...r });
    log('first click', JSON.stringify(r));
  }
  const { profile } = await cdp.send('Profiler.stop');
  await mkdir(OUT, { recursive: true });
  await writeFile(join(OUT, `anchor-${LABEL}.cpuprofile`), JSON.stringify(profile));
} finally {
  await close();
}
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `anchor-${LABEL}.json`), JSON.stringify(out, null, 1));
log('written', join(OUT, `anchor-${LABEL}.json`), 'errors', errors.slice(0, 5), 'shift checks', out.shiftChecks);
