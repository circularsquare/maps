// T-2794: what a new game's load takes on an emulated phone (Pixel size, touch, DPR 3, Android user agent).
//   TW_CORES=3 LABEL=before node trainworld/checks/t2794-mobile-memory.mjs http://localhost:6133/
// Start menu, "Start in New York", demand's final, then a few camera moves to fill the tile cache. Each stage
// and the peak: Chrome's private bytes by process type (Windows, by profile path), the page's JS heap, the demand
// workers' WASM, the workers running, MapLibre's tiles and the canvases' backing size. DESKTOP=1 for the 1400x900
// page at DPR 1, QS adds URL switches. This is desktop Chrome pretending, so the process figures stand in for a
// phone's; the GPU process's figure leaves out what sits in the graphics card. Writes ../artifacts/trainworld/t2794/LABEL.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ARTIFACTS, GL, launchChrome, collectErrors, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6133/';
const LABEL = process.env.LABEL ?? 'run', QS = process.env.QS ?? '', DESKTOP = !!process.env.DESKTOP;
const OUT = join(ARTIFACTS, 't2794');
const MB = 1048576;
const run = promisify(execFile);

async function chromeMemory(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; '{0} {1} {2}' -f $t, $_.PrivatePageCount, $_.WorkingSetSize }`;
  const { stdout } = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps]);
  const by = {};
  let total = 0;
  for (const l of stdout.trim().split(/\r?\n/).filter(Boolean)) {
    const [type, priv] = l.trim().split(' ');
    const mb = Math.round(+priv / MB);
    by[type] = (by[type] ?? 0) + mb;
    total += mb;
  }
  return { total, renderer: by.renderer ?? 0, gpu: by['gpu-process'] ?? 0, by };
}

/** Chrome's own accounting (memory-infra): per process, the biggest allocators by effective size, in MB */
async function memoryDump(cdp) {
  const done = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  await cdp.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { recordMode: 'recordAsMuchAsPossible', includedCategories: ['disabled-by-default-memory-infra'], memoryDumpConfig: { triggers: [] } } });
  await new Promise((r) => setTimeout(r, 1000));
  await cdp.send('Tracing.requestMemoryDump', { deterministic: true, levelOfDetail: 'detailed' });
  await cdp.send('Tracing.end');
  const { stream } = await done;
  let text = '';
  for (;;) { const c = await cdp.send('IO.read', { handle: stream, size: 1 << 22 }); text += c.base64Encoded ? Buffer.from(c.data, 'base64').toString() : c.data; if (c.eof) break; }
  await cdp.send('IO.close', { handle: stream });
  const events = (JSON.parse(text).traceEvents ?? []);
  const names = new Map(events.filter((e) => e.ph === 'M' && e.name === 'process_name').map((e) => [e.pid, e.args.name]));
  const perPid = new Map();
  for (const e of events) if (e.ph === 'v' && e.args?.dumps?.allocators) perPid.set(e.pid, e.args.dumps.allocators);
  const report = {};
  for (const [pid, a] of perPid) {
    const keys = Object.keys(a), leaf = keys.filter((k) => !keys.some((o) => o.startsWith(k + '/')));
    const by = {};
    for (const k of leaf) {
      const v = parseInt(a[k].attrs?.effective_size?.value ?? '0', 16);
      if (!v) continue;
      const p = k.split('/'), g = p[0] === 'v8' ? p.slice(0, p[1] === 'workers' ? 3 : 2).join('/') : p[0] === 'partition_alloc' ? p.slice(0, 3).join('/') : p[0] === 'gpu' ? p.slice(0, 2).join('/') : p[0] === 'canvas' || p[0] === 'webgl' ? p.slice(0, 2).join('/') : p[0];
      by[g] = (by[g] ?? 0) + v;
    }
    const top = Object.entries(by).map(([k, v]) => [k, Math.round(v / MB)]).filter(([, v]) => v >= 3).sort((x, y) => y[1] - x[1]);
    report[`${names.get(pid) ?? 'process'} ${pid}`] = top;
    const raw = (prefix) => keys.filter((k) => k.startsWith(prefix) && k.split('/').length === prefix.split('/').length + 1).map((k) => [k.slice(prefix.length), Math.round(parseInt(a[k].attrs?.size?.value ?? '0', 16) / MB)]).filter(([, v]) => v >= 2);
    console.log(names.get(pid) ?? pid, JSON.stringify(top), 'partitions', JSON.stringify(raw('partition_alloc/partitions/')), 'malloc', JSON.stringify(raw('malloc/')), 'v8/main', JSON.stringify(raw('v8/main/')));
  }
  await writeFile(join(OUT, `${LABEL}-dump.json`), JSON.stringify(report, null, 1));
}

await mkdir(OUT, { recursive: true });
const { devices } = await playwright();
const phone = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete phone.defaultBrowserType;
const options = DESKTOP ? { size: [1400, 900] } : { size: [phone.viewport.width, phone.viewport.height], ...phone };
delete options.viewport;
// LOWEND=1: Chrome's low-end device mode, closer to how a phone's browser spends memory (its collector less lazy)
const chrome = await launchChrome({ prefix: 'tw-t2794-', args: [...GL, ...(process.env.LOWEND ? ['--enable-low-end-device-mode'] : [])], ...options });
const errors = [], stages = [], peak = { total: 0, renderer: 0, gpu: 0 };
let sampling = true;
const sampler = (async () => {
  while (sampling) {
    try {
      const m = await chromeMemory(chrome.profile);
      for (const k of ['total', 'renderer', 'gpu']) peak[k] = Math.max(peak[k], m[k]);
    } catch { /* the browser is closing */ }
    await new Promise((r) => setTimeout(r, 1500));
  }
})();
try {
  const page = await chrome.ctx.newPage();
  const crashed = new Promise((r) => page.on('crash', () => r('crash')));
  collectErrors(page, errors, /favicon|openfreemap|AJAXError/, /maplibre-gl/);
  const got = new Map();
  page.on('response', (r) => { const n = Number(r.headers()['content-length'] ?? 0); if (n > 300000) got.set(r.url().replace(/^.*\/(.*?)(\?.*)?$/, '$1'), Math.round(n / 1024 / 1024 * 10) / 10); });
  const cdp = await chrome.ctx.newCDPSession(page);
  await cdp.send('Performance.enable');
  const measure = async (where) => {
    await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
    const { metrics } = await cdp.send('Performance.getMetrics');
    const heap = Math.round((metrics.find((x) => x.name === 'JSHeapUsedSize')?.value ?? 0) / MB);
    const st = await page.evaluate(() => {
      const c = window.twDemand?.client, map = window.tw?.map;
      const tms = Object.entries(map?.style?.tileManagers ?? {}).map(([id, m]) => [id, Object.keys(m._inViewTiles?._tiles ?? m._tiles ?? {}).length, (m._outOfViewCache ?? m._cache)?.order?.length ?? 0, m._maxTileCacheSize ?? (m._outOfViewCache ?? m._cache)?.max]).filter(([, a, b]) => a + b > 0);
      const canvases = [...document.querySelectorAll('canvas')].filter((x) => x.width * x.height > 40000).map((x) => [x.className || x.id, x.width, x.height]);
      return { opened: c ? [...c.opened].length + '/' + (c.opened.size ? [...c.opened].slice(0, 12).join(',') : '') : null, wasmMB: c?.start.workers.map((w) => Math.round((w?.memBytes ?? 0) / 1048576)), tiles: tms, canvases, pixelRatio: map?.getPixelRatio?.(), dpr: devicePixelRatio, deviceMemory: navigator.deviceMemory };
    }).catch((e) => ({ evalError: String(e).slice(0, 200) }));
    const r = { where, heap, workers: page.workers().map((w) => w.url().replace(/^.*\/(.*?)(\?.*)?$/, '$1')), ...st, proc: await chromeMemory(chrome.profile) };
    stages.push(r);
    console.log(where, `total ${r.proc.total} renderer ${r.proc.renderer} gpu ${r.proc.gpu} heap ${r.heap} wasm ${r.wasmMB} workers ${r.workers.length} opened ${r.opened}`);
    console.log('   ', r.workers.join(' '));
    console.log('   ', JSON.stringify(r.tiles), JSON.stringify(r.canvases), 'ratio', r.pixelRatio);
  };
  await page.goto(`${BASE}?debug=1&demandWorkers=1${QS}&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
  await page.waitForTimeout(3000);
  await measure('menu');
  if (process.env.DUMP === 'all') await memoryDump(cdp);
  await page.tap('.start-menu .new-game .start-new');
  const ready = page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu') && document.querySelector('.guide'), undefined, { timeout: 240000 });
  if ((await Promise.race([ready, crashed])) === 'crash') throw new Error('page crashed during load');
  await measure('game up');
  const final = page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 240000 });
  if ((await Promise.race([final, crashed])) === 'crash') throw new Error('page crashed before demand final');
  await page.waitForTimeout(3000);
  await measure('demand final');
  if (process.env.DUMP) await memoryDump(cdp);
  for (const [lng, lat, z] of [[-73.94, 40.7, 12], [-73.8, 40.75, 13], [-74.1, 40.72, 11], [-73.96, 40.78, 14], [-73.96, 40.75, 11.2]]) {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [[lng, lat], z]);
    await page.waitForTimeout(4000);
  }
  await page.waitForTimeout(3000);
  await measure('after camera moves');
  // the tutorial's own weight: a three-station line built and the clock running
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const pts = [[-73.99, 40.73], [-73.98, 40.75], [-73.97, 40.77], [-73.955, 40.79]].map(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    await c.edit({ op: 'route', from: { kind: 'free', x: pts[0][0], y: pts[0][1], level: -1 }, to: { kind: 'free', x: pts[1][0], y: pts[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < pts.length; k++) { await pause(400); await c.edit({ op: 'route', from: { kind: 'node', node: near(pts[k - 1]) }, to: { kind: 'free', x: pts[k][0], y: pts[k][1], level: -1 }, pis: [], single: false }); }
    await pause(700);
    for (const p of pts.slice(0, 3)) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(300); }
    await c.edit({ op: 'constructAll' });
    await pause(800);
    return c.edit({ op: 'addLine', stops: pts.slice(0, 3).map(near), name: 'Tutorial', colour: '#0a7d3b' });
  });
  if (made?.line !== undefined) await page.evaluate((id) => window.tw.client.edit({ op: 'setTrainCount', line: id, trains: [4, 2, 1] }), made.line);
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await page.waitForTimeout(45000);
  await measure('line built, clock running 45 s');
  console.log('downloads over 0.3 MB', JSON.stringify([...got].sort((a, b) => b[1] - a[1])));
  console.log('peak', JSON.stringify(peak), 'errors', errors.length, errors.slice(0, 5).join(' | '));
} catch (e) {
  console.log('FAILED', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  sampling = false;
  await sampler;
  await writeFile(join(OUT, `${LABEL}.json`), JSON.stringify({ label: LABEL, qs: QS, peak, stages, errors }, null, 1));
  await chrome.close();
}
