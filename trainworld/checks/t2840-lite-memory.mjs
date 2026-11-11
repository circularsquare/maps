// T-2840: what Map detail Low saves. Map detail forced High ("off") or Low ("on") through the stored display setting
// (Auto never involved), interleaved High, Low, High, Low. Desktop: the real New York save at DPR 2. Phone: an
// emulated Pixel 7 (DPR 3), a new game with its first line built. Each run settles at the demand final, then goes to
// z12 and z14 over New York and waits for tiles; then renderer, GPU and utility process memory (OS, by profile path),
// the page's JS heap and the demand workers' WASM are read, and a 4 s pan is timed at 1x and 4x CPU throttle.
//   TW_CORES=2 node trainworld/checks/t2840-lite-memory.mjs http://localhost:6177/ [desktop|phone|both] [rounds]
// Writes artifacts/trainworld/t2840/runs.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ARTIFACTS, GL, REAL_SAVE, launchChrome, importSave, demandFinal, acrossLine, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
const WHICH = process.argv[3] ?? 'both';
const ROUNDS = Number(process.argv[4] ?? 2);
const MB = 1048576;
const run = promisify(execFile);
const OUT = join(ARTIFACTS, 't2840');

async function procMemory(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; '{0} {1} {2}' -f $t, $_.PrivatePageCount, $_.WorkingSetSize }`;
  const { stdout } = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps]);
  const by = {};
  for (const l of stdout.trim().split(/\r?\n/).filter(Boolean)) {
    const [type, priv, ws] = l.trim().split(' ');
    const e = (by[type] ??= { priv: 0, ws: 0 });
    e.priv += +priv / MB;
    e.ws += +ws / MB;
  }
  return by;
}

const view = (lng, lat, zoom) => ({ lng, lat, zoom });
const STOPS = [view(-73.98, 40.75, 12), view(-73.985, 40.758, 14)];

async function pan(page, cdp, rate) {
  await cdp.send('Emulation.setCPUThrottlingRate', { rate });
  await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.99, 40.75], zoom: 13 }); });
  await page.waitForTimeout(3000);
  const gaps = await page.evaluate(() => new Promise((done) => {
    const map = window.tw.map, gaps = [];
    let last = performance.now(), t0 = last, lng = -73.99;
    map.on('render', function f() {
      const now = performance.now();
      gaps.push(now - last);
      last = now;
      if (now - t0 > 4000) { map.off('render', f); done(gaps); }
    });
    const step = () => {
      if (performance.now() - t0 > 4000) return;
      lng += 0.00025;
      map.jumpTo({ center: [lng, 40.75] });
      requestAnimationFrame(step);
    };
    step();
  }));
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  const s = [...gaps].sort((a, b) => a - b);
  return { frames: gaps.length, medianMs: +s[s.length >> 1].toFixed(1), p90Ms: +s[Math.floor(s.length * 0.9)].toFixed(1) };
}

async function once(kind, detail, round, phone) {
  const lite = detail === 'Low' ? 'on' : 'off';
  const options = kind === 'phone' ? { size: [phone.vw, phone.vh], ...phone.opts } : { size: [1400, 900], deviceScaleFactor: 2 };
  const chrome = await launchChrome({ prefix: `tw-t2840${kind[0]}-`, args: GL, ...options });
  const label = `${kind} ${detail} #${round}`;
  try {
    await chrome.ctx.addInitScript((lm) => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ v: 12, display: { lighterMap: lm } })); } catch { /* none */ }
    }, lite);
    const page = await chrome.ctx.newPage();
    const cdp = await chrome.ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    await page.goto(`${BASE}?debug=1&new=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.map && window.twDemand?.client, undefined, { timeout: 240000 });
    if (kind === 'desktop') {
      await importSave(page, await readFile(REAL_SAVE));
      await page.waitForTimeout(3000);
      await demandFinal(page, { timeout: 480000 });
    } else {
      await page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 240000 }).catch(() => {});
      await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.98, 40.75], zoom: 13 }); });
      await page.waitForTimeout(2000);
      const line = await acrossLine(page, [-73.98, 40.75], 2);
      if (line === null) throw new Error('no line built');
      await page.evaluate(() => window.tw.clock.setRunning(true));
      await demandFinal(page, { timeout: 480000, strict: false }).catch(() => {});
    }
    await page.waitForTimeout(3000);
    const lm = await page.evaluate(() => ({ lighter: JSON.parse(localStorage.getItem('trainworld-ui')).display?.lighterMap, ratio: window.tw.map.getPixelRatio(), dpr: devicePixelRatio, layers: window.tw.map.getStyle().layers.filter((l) => window.tw.map.getLayoutProperty(l.id, 'visibility') !== 'none').length }));
    for (const v of STOPS) {
      await page.evaluate((v) => { window.tw.map.jumpTo({ center: [v.lng, v.lat], zoom: v.zoom }); }, v);
      await page.waitForTimeout(7000);
    }
    await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.985, 40.758], zoom: 14 }); });
    await page.waitForTimeout(3000);
    await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
    const { metrics } = await cdp.send('Performance.getMetrics');
    const heap = (metrics.find((x) => x.name === 'JSHeapUsedSize')?.value ?? 0) / MB;
    const wasm = await page.evaluate(() => (window.twDemand.client.start?.workers ?? []).reduce((s, w) => s + (w?.memBytes ?? 0), 0) / 1048576);
    const proc = await procMemory(chrome.profile);
    const frames1 = await pan(page, cdp, 1);
    const frames4 = await pan(page, cdp, 4);
    const r = { label, kind, detail, round, map: lm, heapMB: +heap.toFixed(0), wasmMB: +wasm.toFixed(0), proc: Object.fromEntries(Object.entries(proc).map(([k, v]) => [k, { privMB: +v.priv.toFixed(0), wsMB: +v.ws.toFixed(0) }])), pan1x: frames1, pan4x: frames4 };
    console.log(JSON.stringify(r));
    return r;
  } finally { await chrome.close(); }
}

await mkdir(OUT, { recursive: true });
const { devices } = await playwright();
const px = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete px.defaultBrowserType;
const phone = { vw: px.viewport.width, vh: px.viewport.height, opts: (() => { const o = { ...px }; delete o.viewport; return o; })() };
const results = [];
const kinds = WHICH === 'both' ? ['desktop', 'phone'] : [WHICH];
for (const kind of kinds) {
  for (let i = 1; i <= ROUNDS; i++) {
    for (const detail of ['High', 'Low']) {
      try { results.push(await once(kind, detail, i, phone)); } catch (e) { console.log('FAILED', kind, detail, i, String(e).slice(0, 300)); }
      await writeFile(join(OUT, `runs-${WHICH}.json`), JSON.stringify(results, null, 1));
    }
  }
}
