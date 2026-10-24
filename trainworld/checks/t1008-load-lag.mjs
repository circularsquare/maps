// T-1008: main-thread long tasks after a big save loads (notes/T-1001.md "T-1008").
//   node trainworld/checks/t1008-load-lag.mjs <page url>      (from the maps root)
// A new game (`?new=1`, a throwaway profile), then the save imported; from the import on it records
// long tasks (over 50 ms), frame gaps and each demand-client message's main-thread ms, with a CPU
// profile of the main thread over the first WINDOW seconds (top functions by self and inclusive ms).
// With FINAL=1 (default) it then waits for demand's final and records the game's figures, for
// comparing builds. Env: SAVE (default trainworld/T-007-real-nyc.save), LABEL, OUT (default
// ../artifacts/trainworld/t1008), WINDOW (s, default 30), DEMAND_WORKERS (default 2), FINAL, AT
// ("lon,lat,zoom" to look at while loading), LINES (lines the loaded save has at least, default 50),
// CONTINUE=1 (the save stored in the slot and solved first, then the page opened again with
// `?continue=1`: the window starts at that navigation, the demand cache shown first).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal } from './lib.mjs';

const SAVE = process.env.SAVE ?? fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t1008/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const WINDOW = Number(process.env.WINDOW ?? 30) * 1000;
const FINAL = process.env.FINAL !== '0';
const LINES = Number(process.env.LINES ?? 50);
const CONTINUE = process.env.CONTINUE === '1';
const BASE = process.argv[2] ?? 'http://localhost:6131/';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** self and inclusive ms per function (name @ file:line) from a CPU profile */
function profileTop(prof, n = 30) {
  const byId = new Map(prof.nodes.map((x) => [x.id, x]));
  const parent = new Map();
  for (const x of prof.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const selfUs = new Map();
  for (let i = 0; i < prof.samples.length; i++) selfUs.set(prof.samples[i], (selfUs.get(prof.samples[i]) ?? 0) + (prof.timeDeltas[i + 1] ?? 0));
  const name = (x) => `${x.callFrame.functionName || '(anon)'} ${x.callFrame.url.replace(/.*\//, '').replace(/\?.*/, '')}:${x.callFrame.lineNumber + 1}`;
  const self = new Map(), incl = new Map();
  for (const [id, us] of selfUs) {
    const x = byId.get(id);
    self.set(name(x), (self.get(name(x)) ?? 0) + us);
    const seen = new Set();
    for (let c = id; c !== undefined; c = parent.get(c)) { const k = name(byId.get(c)); if (!seen.has(k)) { seen.add(k); incl.set(k, (incl.get(k) ?? 0) + us); } }
  }
  const top = (m) => [...m].filter(([k]) => !/^\((root|program|idle|anon\) :0)/.test(k)).sort((a, b) => b[1] - a[1]).slice(0, n).map(([k, us]) => [k, Math.round(us / 1000)]);
  return { self: top(self), inclusive: top(incl), idleMs: Math.round([...self].filter(([k]) => /^\(idle/.test(k)).reduce((a, [, v]) => a + v, 0) / 1000) };
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t1008-' });
const errors = [];
const out = { label: LABEL, base: BASE, save: SAVE, windowMs: WINDOW, errors };
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
    window.__frames = [];
    const tick = (t) => { window.__frames.push(t); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    window.__lt = [];
    try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
    window.__dm = [];
    const hook = setInterval(() => {
      const c = window.twDemand?.client;
      if (!c || c.__hooked) return;
      c.__hooked = true;
      clearInterval(hook);
      const orig = c.receive.bind(c);
      c.receive = (i, m) => { const t0 = performance.now(); try { return orig(i, m); } finally { window.__dm.push([t0, performance.now() - t0, m?.kind ?? '?']); } };
    }, 5);
  });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError|^\w{1,3}$/.test(m.text()) && !/maplibre-gl/.test(m.location()?.url ?? '')) errors.push(m.text().slice(0, 300)); });
  const cdp = await ctx.newCDPSession(page);
  const workers = `demandWorkers=${process.env.DEMAND_WORKERS ?? 2}`;
  const solved = (timeout) => demandFinal(page, { timeout, polling: 250 });
  let t0;
  if (CONTINUE) {
    // the save stored in the slot and solved once, so its demand cache (T-328) is written; then the
    // page opened again on it, as a returning player's is
    await page.goto(`${BASE}?debug=1&continue=1&${workers}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    const bytes = await readFile(SAVE);
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await importSave(page, bytes);
    await page.waitForFunction((n) => (window.tw.world.value?.save?.lines?.length ?? 0) >= n, LINES, { timeout: 300000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await solved(20 * 60000);
    await page.waitForTimeout(5000);
    log('stored and solved');
    await cdp.send('Profiler.enable');
    await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
    await page.goto(`${BASE}?debug=1&continue=1&${workers}&v=${Date.now()}`, { waitUntil: 'commit' });
    await cdp.send('Profiler.start');
    t0 = 0;
  } else {
    await page.goto(`${BASE}?debug=1&new=1&${workers}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client?.__hooked, undefined, { timeout: 180000 });
    if (process.env.AT) { const [lon, lat, z] = process.env.AT.split(',').map(Number); await page.evaluate(([lon, lat, z]) => void window.tw.map.jumpTo({ center: [lon, lat], zoom: z }), [lon, lat, z]); }
    await page.waitForTimeout(4000);
    const bytes = await readFile(SAVE);
    await cdp.send('Profiler.enable');
    await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
    await cdp.send('Profiler.start');
    t0 = await importSave(page, bytes);
  }
  await page.waitForFunction((n) => (window.tw?.world?.value?.save?.lines?.length ?? 0) >= n, LINES, { timeout: 300000, polling: 100 });
  out.linesAtMs = Math.round(await page.evaluate(() => performance.now()) - t0);
  const left = t0 + WINDOW - (await page.evaluate(() => performance.now()));
  if (left > 0) await page.waitForTimeout(left);
  const { profile: prof } = await cdp.send('Profiler.stop');
  await mkdir(OUT, { recursive: true });
  await writeFile(join(OUT, `main-${LABEL}.cpuprofile`), JSON.stringify(prof));
  out.profile = profileTop(prof);
  const stats = (from, to) => page.evaluate(([from, to]) => {
    const f = window.__frames.filter((t) => t >= from && t <= to);
    let maxGap = 0, gaps = 0;
    for (let i = 1; i < f.length; i++) { const g = f[i] - f[i - 1]; maxGap = Math.max(maxGap, g); if (g > 50) gaps++; }
    const lt = window.__lt.filter(([s]) => s >= from && s <= to);
    const dm = window.__dm.filter(([s]) => s >= from && s <= to);
    const by = {};
    for (const [, ms, k] of dm) { const b = (by[k] ??= { n: 0, ms: 0, max: 0 }); b.n++; b.ms += ms; b.max = Math.max(b.max, ms); }
    for (const b of Object.values(by)) { b.ms = Math.round(b.ms); b.max = Math.round(b.max); }
    return { secs: +((to - from) / 1000).toFixed(1), fps: +(f.length / ((to - from) / 1000)).toFixed(1), frameGapsOver50: gaps, maxFrameGap: Math.round(maxGap),
      longTasks: lt.length, longMax: Math.round(Math.max(0, ...lt.map(([, d]) => d))), longTotal: Math.round(lt.reduce((a, [, d]) => a + d, 0)),
      blocked: Math.round(lt.reduce((a, [, d]) => a + Math.max(0, d - 50), 0)), demandMsgs: by, lt: lt.map(([s, d]) => [Math.round(s - from), Math.round(d)]) };
  }, [from, to]);
  out.window = await stats(t0, t0 + WINDOW);
  log('window', JSON.stringify({ ...out.window, lt: undefined }));
  log('self', JSON.stringify(out.profile.self.slice(0, 12)));
  log('inclusive', JSON.stringify(out.profile.inclusive.slice(0, 20)));
  if (process.env.EDITS === '1') {
    // a track segment laid near AT (or the view's centre), then a station on its end: the long tasks
    // in the 6 s after each
    out.edits = [];
    await cdp.send('Profiler.start');
    for (const kind of ['track', 'station', 'track']) {
      const t = await page.evaluate(async ([kind, k]) => {
        const tw = window.tw, c = tw.client, m = tw.map.getCenter();
        const p = tw.lngLatToPack(m.lng + 0.01 * k, m.lat + 0.004), q = tw.lngLatToPack(m.lng + 0.01 * k + 0.008, m.lat + 0.006);
        const t0 = performance.now();
        if (kind === 'track') await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: -1 }, to: { kind: 'free', x: q.x, y: q.y, level: -1 }, pis: [], single: false });
        else {
          let best = null, bd = 1e12;
          for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - q.x, n.y - q.y); if (d < bd) { bd = d; best = n.id; } }
          await c.edit({ op: 'addStation', at: { kind: 'node', node: best }, platform: 200, name: '' });
        }
        return t0;
      }, [kind, out.edits.length]);
      await page.waitForTimeout(6000);
      const s = await stats(t, t + 6000);
      out.edits.push({ kind, ...s, lt: s.lt });
      log('edit', kind, JSON.stringify({ ...s, lt: s.lt }));
    }
    out.editProfile = profileTop((await cdp.send('Profiler.stop')).profile);
    log('edits inclusive', JSON.stringify(out.editProfile.inclusive.slice(0, 25)));
  }
  if (FINAL) {
    await solved(20 * 60000);
    await page.waitForTimeout(3000);
    const t1 = await page.evaluate(() => performance.now());
    out.finalAtMs = Math.round(t1 - 3000 - t0);
    out.toFinal = await stats(t0, t1);
    out.figures = await page.evaluate(() => {
      const r = (x) => Math.round(x * 1000) / 1000;
      return window.twDemand.days().map((d) => {
        const v = d.view, ic = d.intercity;
        let c = ic?.corridors;
        // T-1008's table and pairs with rail, as dense columns
        if (c?.table) {
          const t = c.table, n = t.n, cols = { n, trips: new Float64Array(n), rail: new Float64Array(n), commutes: new Float64Array(n), commuteRail: new Float64Array(n), air: new Float64Array(n), car: new Float64Array(n) };
          const lr = new Float64Array(n);
          c.railAt.forEach((j, k) => { lr[j] = c.longerRail[k]; cols.commuteRail[j] = c.commuteRail[k]; });
          for (let j = 0; j < n; j++) {
            const tl = t.longer[j] * c.longerScale, cm = t.commutes[j] * c.commuteScale, rest = tl - lr[j];
            cols.trips[j] = tl + cm; cols.rail[j] = lr[j] + cols.commuteRail[j]; cols.commutes[j] = cm; cols.air[j] = rest * t.air[j]; cols.car[j] = rest * t.car[j] + cm - cols.commuteRail[j];
          }
          c = cols;
        }
        const sum = (a) => (a ? r(a.reduce((x, y) => x + y, 0)) : null);
        let riders = 0; for (const l of v?.lines?.values() ?? []) riders += l.ridersPerDay;
        return v && { final: v.final, train: r(v.split.train), walk: r(v.split.walk), trips: r(v.tripsPerDay), rail: r(v.railTripsPerDay), riders: r(riders), cities: d.cities?.size,
          inter: ic && { trips: r(ic.tripsPerDay), rail: r(ic.railTripsPerDay), commutes: r(ic.commuteTripsPerDay), commuteRail: r(ic.commuteRailPerDay) },
          corridors: c && { n: c.n, trips: sum(c.trips), rail: sum(c.rail), commutes: sum(c.commutes), commuteRail: sum(c.commuteRail), air: sum(c.air), car: sum(c.car) } };
      });
    });
    log('final at', out.finalAtMs, 'ms', JSON.stringify({ ...out.toFinal, lt: undefined }));
    log('figures', JSON.stringify(out.figures[0]));
  }
} finally {
  await close();
}
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `load-${LABEL}.json`), JSON.stringify(out, null, 1));
log('written', join(OUT, `load-${LABEL}.json`), 'errors', errors.slice(0, 5));
