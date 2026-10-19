// T-348: main-thread long tasks and frame gaps while demand loads (notes/T-348.md).
//   MODE=prepare | startup | pan     BASE=<page url>  OUT=<output folder>  LABEL=<name>
//   BUBBLES=on|off (Destinations bubbles, weekday)   TRACE=0 to skip the Chrome trace
//   PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs node trainworld/checks/t348-lag.mjs
// One persistent profile (OUT/profile) keeps the real save between runs: run MODE=prepare first.
// startup: navigation to the newest solve's final + 3 s. pan: the save solved, then a jump to
//   Philadelphia, 15 s, a jump to Toronto, 15 s. Throughout, the page pans the map 3 px one way
//   then back every 50 ms (input the main thread must keep up with), counts frames with
//   requestAnimationFrame, and records long tasks (PerformanceObserver). With TRACE (default on)
//   a Chrome trace with CPU samples goes to OUT/trace-<MODE>-<LABEL>.json for t348-analyze.mjs.
// Writes OUT/<MODE>-<LABEL>.json: frames, long tasks, the demand events.
import { readFile, writeFile } from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import { join, resolve } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const OUT = process.env.OUT ?? '.';
const BASE = process.env.BASE ?? 'http://localhost:6148/';
const MODE = process.env.MODE ?? 'startup';
const LABEL = process.env.LABEL ?? 'run';
const BUBBLES = process.env.BUBBLES ?? 'on';
const TRACE = process.env.TRACE !== '0';
// QS: more query, e.g. '&perfOff=nocache' (an unknown flag: the demand cache's key misses, T-328)
const QS = process.env.QS ?? '';
const profile = resolve(OUT, process.env.PROFILE ?? 'profile');
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist', ...(process.env.CHROME_ARGS ? process.env.CHROME_ARGS.split(' ') : [])] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

await ctx.addInitScript(([bubbles]) => {
  try {
    const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
    s.v = 2;
    s.display = { ...(s.display || {}), commuters: bubbles !== 'off', commuterEnd: 'work', dayShown: 'weekday' };
    localStorage.setItem('trainworld-ui', JSON.stringify(s));
  } catch { /* */ }
  const ev = (window.__ev = []);
  const mark = (k, v) => ev.push([Math.round(performance.now()), k, v]);
  window.__mark = mark;
  window.__frames = [];
  const tick = (t) => { window.__frames.push(t); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  window.__lt = [];
  try {
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([Math.round(e.startTime), Math.round(e.duration)]); }).observe({ type: 'longtask', buffered: true });
  } catch { /* */ }
  let lastView = '', world = false, clock = false, lastBar = '';
  setInterval(() => {
    if (!world && window.tw?.world?.value) { world = true; mark('world'); }
    if (!clock && world && window.tw?.clock?.held && window.tw.clock.held.peek() === false) { clock = true; mark('clock'); }
    const d = window.twDemand;
    if (!d) return;
    const v = d.view?.();
    const key = v ? `${v.version}:${v.round}:${v.final}` : '';
    if (key !== lastView) { lastView = key; mark('view', key); }
    const bar = document.querySelector('.top-bar')?.textContent?.match(/(Updating demand[^A-Z]*|Estimate[^A-Z]*|Loading[^A-Z]*)/)?.[0] ?? '';
    if (bar !== lastBar) { lastBar = bar; mark('bar', bar); }
  }, 50);
  // the "player": the map nudged 3 px and back every 50 ms once it exists
  let dir = 1;
  window.__jiggle = setInterval(() => {
    const m = window.tw?.map;
    if (!m || window.__noJiggle) return;
    m.panBy([3 * dir, 0], { duration: 0 });
    dir = -dir;
  }, 50);
}, [BUBBLES]);

const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/openfreemap|AJAXError|favicon/.test(m.text())) errors.push(m.text()); });
const cdp = await ctx.newCDPSession(page);

const CATS = ['devtools.timeline', 'disabled-by-default-devtools.timeline', 'v8', 'v8.execute', 'disabled-by-default-v8.cpu_profiler', 'blink.user_timing', 'toplevel', 'disabled-by-default-v8.gc', 'loading'];
async function traceStart() {
  if (!TRACE) return;
  await cdp.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { includedCategories: CATS, recordMode: 'recordContinuously' } });
}
async function traceStop(name) {
  if (!TRACE) return;
  const done = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  await cdp.send('Tracing.end');
  const { stream } = await done;
  const file = join(OUT, `trace-${name}.json`);
  const ws = createWriteStream(file);
  for (;;) {
    const { data, eof, base64Encoded } = await cdp.send('IO.read', { handle: stream, size: 1 << 20 });
    ws.write(base64Encoded ? Buffer.from(data, 'base64') : data);
    if (eof) break;
  }
  await cdp.send('IO.close', { handle: stream });
  await new Promise((r) => ws.end(r));
  log('trace', file);
}

const waitSolved = (timeout = 180000) => page.waitForFunction(() => {
  const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
  return s && v && v.version === s.version && v.final && !d.client.waiting;
}, undefined, { timeout, polling: 200 });

/** frames and long tasks between two performance.now() times */
const windowStats = (from, to) => page.evaluate(([from, to]) => {
  const f = window.__frames.filter((t) => t >= from && t <= to);
  let gaps50 = 0, gaps100 = 0, maxGap = 0, over = 0;
  for (let i = 1; i < f.length; i++) { const g = f[i] - f[i - 1]; if (g > 50) { gaps50++; over += g; } if (g > 100) gaps100++; maxGap = Math.max(maxGap, g); }
  const lt = window.__lt.filter(([s]) => s >= from && s <= to);
  return { secs: +((to - from) / 1000).toFixed(1), frames: f.length, fps: +(f.length / ((to - from) / 1000)).toFixed(1), gaps50, gaps100, maxGap: Math.round(maxGap), msInGaps50: Math.round(over),
    longTasks: lt.length, longMs: lt.reduce((a, [, d]) => a + d, 0), longMax: lt.reduce((a, [, d]) => Math.max(a, d), 0), lt };
}, [from, to]);

const out = { mode: MODE, label: LABEL, base: BASE, bubbles: BUBBLES, errors };
try {
  if (MODE === 'prepare') {
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const bytes = await readFile(new URL('../T-007-real-nyc.save', import.meta.url));
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    // and solve it once, so the demand cache (T-328) holds this build's final
    await waitSolved(240000);
    await page.waitForTimeout(3000);
    log('save stored and solved');
  } else if (MODE === 'startup') {
    await traceStart();
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved(240000);
    await page.waitForTimeout(3000);
    const end = await page.evaluate(() => performance.now());
    await traceStop(`${MODE}-${LABEL}`);
    out.ev = await page.evaluate(() => window.__ev);
    const final = out.ev.filter(([, k, v]) => k === 'view' && /:true$/.test(v)).at(-1)?.[0] ?? end;
    const first = out.ev.find(([, k]) => k === 'world')?.[0] ?? 0;
    out.cache = await page.evaluate(() => window.twDemand.client.cacheLog);
    out.solves = await page.evaluate(() => window.twDemand.log.map((s) => ({ v: s.version, sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), final: s.refinedAt && Math.round(s.refinedAt) })));
    out.computeN = await page.evaluate(() => window.twDemand.client.computeN);
    out.hc = await page.evaluate(() => navigator.hardwareConcurrency);
    // from the network drawn to the fresh final, and the 3 s after
    out.loading = await windowStats(first, final);
    out.after = await windowStats(final, end);
    log('startup', JSON.stringify({ ...out.loading, lt: undefined }), 'after', JSON.stringify({ ...out.after, lt: undefined }));
  } else if (MODE === 'resolve') {
    // the same network solved again once settled: the workers busy as at startup, nothing new opened
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved(240000);
    await page.waitForTimeout(3000);
    await traceStart();
    const t0 = await page.evaluate(() => { window.twDemand.resubmit(); return performance.now(); });
    await page.waitForTimeout(500);
    await waitSolved(240000);
    const t1 = await page.evaluate(() => performance.now());
    await traceStop(`${MODE}-${LABEL}`);
    out.resolve = await windowStats(t0, t1);
    log('resolve', JSON.stringify({ ...out.resolve, lt: undefined }));
  } else if (MODE === 'reach') {
    // the save solved, a jump to Toronto (opened bare), then a short line built there: Toronto is
    // fetched again for every worker, opened in all and solved with the rest (T-325)
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    await waitSolved(240000);
    await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.39, 43.67], zoom: 12 }));
    await page.waitForTimeout(6000);
    await traceStart();
    const t0 = await page.evaluate(async () => {
      const PTS = [[-79.3806, 43.6453], [-79.3832, 43.6560], [-79.3866, 43.6710]];
      const tw = window.tw, c = tw.client;
      const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
      const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
      const t0 = performance.now();
      window.__mark('build');
      await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
      await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[1]) }, to: { kind: 'free', x: xy[2][0], y: xy[2][1], level: -1 }, pis: [], single: false });
      for (const p of xy) await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' });
      await c.edit({ op: 'constructAll' });
      const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Lag test', colour: '#0a7d3b' });
      await c.edit({ op: 'setTrainCount', line: r.line, trains: [10, 8, 6] });
      return t0;
    });
    await page.waitForFunction(() => window.twDemand.client.opened.has?.('tor') ?? [...window.twDemand.client.opened].includes('tor'), undefined, { timeout: 120000, polling: 200 }).catch(() => log('tor never opened in all'));
    await page.waitForTimeout(1000);
    await waitSolved(240000);
    await page.waitForTimeout(2000);
    const t1 = await page.evaluate(() => performance.now());
    await traceStop(`${MODE}-${LABEL}`);
    out.reach = await windowStats(t0, t1);
    out.reach.loads = await page.evaluate((t0) => window.twDemand.start.loads.filter((l) => l.requestedAt >= t0).map((l) => ({ item: l.item, bare: !!l.bare, req: Math.round(l.requestedAt - t0), dl: l.downloadedAt && Math.round(l.downloadedAt - t0), open: l.openedAt && Math.round(l.openedAt - t0) })), t0);
    out.reach.solves = await page.evaluate((t0) => window.twDemand.log.filter((s) => s.sentAt >= t0).map((s) => ({ v: s.version, cities: s.cities.join('+'), sent: Math.round(s.sentAt - t0), first: s.firstAt && Math.round(s.firstAt - t0), final: s.refinedAt && Math.round(s.refinedAt - t0) })), t0);
    log('reach', JSON.stringify({ ...out.reach, lt: out.reach.lt.map(([s, d]) => [Math.round(s - t0), d]) }));
  } else if (MODE === 'pan') {
    await page.goto(`${BASE}?debug=1&continue=1${QS}&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand, undefined, { timeout: 60000 });
    // EARLY=1: the pan starts once the network is drawn, while the startup solve runs
    if (process.env.EARLY === '1') await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 120000 });
    else { await waitSolved(240000); await page.waitForTimeout(3000); }
    await traceStart();
    out.steps = [];
    // WAY: [[name, lon, lat, zoom], ...]; STEP: seconds at each
    const WAY = JSON.parse(process.env.WAY ?? 'null') ?? [['Philadelphia', -75.16, 39.95, 11], ['Toronto', -79.38, 43.65, 11]];
    for (const [name, lon, lat, zoom] of WAY) {
      const t0 = await page.evaluate(([lon, lat, zoom, name]) => { window.__mark('jump', name); window.tw.map.jumpTo({ center: [lon, lat], zoom }); return performance.now(); }, [lon, lat, zoom, name]);
      await page.waitForTimeout(Number(process.env.STEP ?? 15) * 1000);
      const t1 = await page.evaluate(() => performance.now());
      const st = await windowStats(t0, t1);
      st.name = name;
      st.loads = await page.evaluate((t0) => window.twDemand.start.loads.filter((l) => l.requestedAt >= t0 - 50).map((l) => ({ item: l.item, bare: !!l.bare, req: Math.round(l.requestedAt - t0), dl: l.downloadedAt && Math.round(l.downloadedAt - t0), open: l.openedAt && Math.round(l.openedAt - t0) })), t0);
      st.solves = await page.evaluate((t0) => window.twDemand.log.filter((s) => s.sentAt >= t0).map((s) => ({ v: s.version, sent: Math.round(s.sentAt - t0), final: s.refinedAt && Math.round(s.refinedAt - t0) })), t0);
      st.t0 = Math.round(t0);
      out.steps.push(st);
      log(name, JSON.stringify({ ...st, lt: st.lt.map(([s, d]) => [Math.round(s - t0), d]) }));
    }
    await traceStop(`${MODE}-${LABEL}`);
    out.ev = await page.evaluate(() => window.__ev);
  }
} finally {
  await ctx.close();
}
await writeFile(join(OUT, `${MODE}-${LABEL}.json`), JSON.stringify(out, null, 1));
log('written', `${MODE}-${LABEL}.json`, 'errors', errors.slice(0, 5));
