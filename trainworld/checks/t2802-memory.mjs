// T-2802: where an emulated phone's memory goes, by process and by JS context (the page and each worker), through
// a new game's load and the tutorial's first line. Run against a scratch production build (the dev server's own
// weight misleads; notes/T-2802.md says how):
//   TW_CORES=4 LABEL=before [LOWEND=1] [DUMP=1] node trainworld/checks/t2802-memory.mjs http://127.0.0.1:6211/
// The flow is t2794-mobile-memory.mjs's: start menu, "Start in New York", demand's final, camera moves, then a
// three-station line with the clock running 45 s. Every second: Chrome's private bytes by process type, and each
// context's V8 heap (used) and ArrayBuffer backing stores (WASM memories included) from Runtime.getHeapUsage over a
// raw CDP connection, so the peak can be split by context. DUMP=1 adds Chrome's memory-infra accounting at the
// stages; NOGC=1 forces no collection anywhere (the natural figures, with a lazy collector's leftovers). Writes ../artifacts/trainworld/t2802/LABEL.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ARTIFACTS, GL, launchChrome, collectErrors, playwright } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://127.0.0.1:6211/';
const LABEL = process.env.LABEL ?? 'run', QS = process.env.QS ?? '', NOGC = !!process.env.NOGC;
const OUT = join(ARTIFACTS, 't2802');
const MB = 1048576;
const PORT = 9300 + Math.floor(Math.random() * 500);
const run = promisify(execFile);
const mb = (x) => Math.round(x / MB);

async function chromeMemory(profile) {
  const needle = profile.replace(/'/g, "''");
  const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${needle}*' } | ForEach-Object { $t = if ($_.CommandLine -match '--type=(\\S+)') { $matches[1] } else { 'browser' }; '{0} {1}' -f $t, $_.PrivatePageCount }`;
  const { stdout } = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps]);
  const by = {};
  let total = 0;
  for (const l of stdout.trim().split(/\r?\n/).filter(Boolean)) {
    const [type, priv] = l.trim().split(' ');
    by[type] = (by[type] ?? 0) + mb(+priv);
    total += mb(+priv);
  }
  return { total, renderer: by.renderer ?? 0, gpu: by['gpu-process'] ?? 0 };
}

/** a raw CDP connection to the browser, so a worker's session can be spoken to (Playwright's CDPSession can't) */
async function rawCdp(port) {
  let v;
  for (let k = 0; k < 50 && !v; k++) v = await fetch(`http://127.0.0.1:${port}/json/version`).then((r) => r.json()).catch(() => new Promise((r) => setTimeout(r, 200)));
  const ws = new WebSocket(v.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const wait = new Map();
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    const w = m.id && wait.get(m.id);
    if (!w) return;
    wait.delete(m.id);
    if (m.error) w.j(new Error(m.error.message)); else w.r(m.result);
  };
  const send = (method, params = {}, sessionId) => new Promise((r, j) => {
    const i = ++id;
    wait.set(i, { r, j });
    ws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }));
    setTimeout(() => { if (wait.delete(i)) j(new Error('timeout ' + method)); }, 15000);
  });
  return { send, close: () => ws.close() };
}

const sessions = new Map();
const shortName = (url) => url.replace(/^.*\/(.*?)(\?.*)?$/, '$1').replace(/-[A-Za-z0-9_-]{8}\.js$/, '.js') || url;
/** a context's WASM memories, MB (they are not in its backing stores): every WebAssembly.Memory it holds, by
 * queryObjects, which collects garbage first, so only at the stages */
async function wasmOf(cdp, s) {
  try {
    const { result: proto } = await cdp.send('Runtime.evaluate', { expression: 'WebAssembly.Memory.prototype' }, s);
    const { objects } = await cdp.send('Runtime.queryObjects', { prototypeObjectId: proto.objectId }, s);
    const { result } = await cdp.send('Runtime.callFunctionOn', { objectId: objects.objectId, functionDeclaration: 'function () { return this.reduce((a, m) => a + m.buffer.byteLength, 0); }', returnByValue: true }, s);
    return mb(result.value);
  } catch { return -1; }
}

/** each context's heap: [name, used MB, ArrayBuffer backing MB] */
async function contexts(cdp, withWasm = false) {
  const { targetInfos } = await cdp.send('Target.getTargets', { filter: [{}] });
  const out = [];
  for (const t of targetInfos) {
    if (t.type !== 'page' && t.type !== 'worker') continue;
    if (t.type === 'page' && !t.url.startsWith(BASE)) continue;
    try {
      let s = sessions.get(t.targetId);
      if (!s) {
        s = (await cdp.send('Target.attachToTarget', { targetId: t.targetId, flatten: true })).sessionId;
        sessions.set(t.targetId, s);
      }
      const h = await cdp.send('Runtime.getHeapUsage', {}, s);
      const row = [t.type === 'page' ? 'page' : shortName(t.url), mb(h.usedSize), mb(h.backingStorageSize ?? 0)];
      if (withWasm) row.push(await wasmOf(cdp, s));
      out.push(row);
    } catch { /* a worker that just ended */ }
  }
  return out.sort((a, b) => b[1] + b[2] - a[1] - a[2]);
}

/** Chrome's own accounting (memory-infra) for the renderer: the biggest allocators by effective size, in MB */
async function memoryDump(cdp, where) {
  const done = new Promise((r) => cdp.once('Tracing.tracingComplete', r));
  await cdp.send('Tracing.start', { transferMode: 'ReturnAsStream', traceConfig: { recordMode: 'recordAsMuchAsPossible', includedCategories: ['disabled-by-default-memory-infra'], memoryDumpConfig: { triggers: [] } } });
  await new Promise((r) => setTimeout(r, 1000));
  await cdp.send('Tracing.requestMemoryDump', { deterministic: true, levelOfDetail: 'detailed' });
  await cdp.send('Tracing.end');
  const { stream } = await done;
  let text = '';
  for (;;) { const c = await cdp.send('IO.read', { handle: stream, size: 1 << 22 }); text += c.base64Encoded ? Buffer.from(c.data, 'base64').toString() : c.data; if (c.eof) break; }
  await cdp.send('IO.close', { handle: stream });
  const events = JSON.parse(text).traceEvents ?? [];
  const names = new Map(events.filter((e) => e.ph === 'M' && e.name === 'process_name').map((e) => [e.pid, e.args.name]));
  const report = {};
  for (const e of events) {
    const a = e.ph === 'v' && e.args?.dumps?.allocators;
    if (!a || !/Renderer/i.test(names.get(e.pid) ?? '')) continue;
    const keys = Object.keys(a), leaf = keys.filter((k) => !keys.some((o) => o.startsWith(k + '/')));
    const by = {};
    for (const k of leaf) {
      const v = parseInt(a[k].attrs?.effective_size?.value ?? '0', 16);
      if (!v) continue;
      const p = k.split('/'), g = p[0] === 'v8' ? p.slice(0, p[1] === 'workers' ? 3 : 2).join('/') : p[0] === 'partition_alloc' ? p.slice(0, 3).join('/') : ['gpu', 'canvas', 'webgl', 'blink_gc', 'malloc'].includes(p[0]) ? p.slice(0, 2).join('/') : p[0];
      by[g] = (by[g] ?? 0) + v;
    }
    const raw = (prefix) => keys.filter((k) => k.startsWith(prefix) && k.split('/').length === prefix.split('/').length + 1).map((k) => [k.slice(prefix.length), mb(parseInt(a[k].attrs?.size?.value ?? '0', 16))]).filter(([, v]) => v >= 2);
    by.raw = { partitions: raw('partition_alloc/partitions/'), malloc: raw('malloc/'), v8: raw('v8/'), v8workers: raw('v8/workers/'), blink_gc: raw('blink_gc/') };
    report[`${names.get(e.pid)} ${e.pid}`] = Object.entries(by).map(([k, v]) => [k, k === 'raw' ? v : mb(v)]).filter(([, v]) => typeof v === 'object' || v >= 2).sort((x, y) => (+y[1] || -1) - (+x[1] || -1));
  }
  console.log('   dump', where, JSON.stringify(report));
  return report;
}

await mkdir(OUT, { recursive: true });
const { devices } = await playwright();
const phone = { ...devices['Pixel 7'], deviceScaleFactor: 3 };
delete phone.defaultBrowserType;
const options = { size: [phone.viewport.width, phone.viewport.height], ...phone };
delete options.viewport;
const chrome = await launchChrome({ prefix: 'tw-t2802-', args: [...GL, `--remote-debugging-port=${PORT}`, ...(process.env.LOWEND ? ['--enable-low-end-device-mode'] : [])], ...options });
const raw = await rawCdp(PORT);
const errors = [], stages = [], timeline = [], dumps = {};
let stage = 'menu', sampling = true, peak = null;
const t0 = Date.now();
const sampler = (async () => {
  while (sampling) {
    try {
      const [proc, ctx] = await Promise.all([chromeMemory(chrome.profile), contexts(raw)]);
      const s = { s: Math.round((Date.now() - t0) / 1000), stage, ...proc, ctx };
      timeline.push(s);
      if (!peak || s.renderer > peak.renderer) peak = s;
    } catch { /* the browser is closing */ }
    await new Promise((r) => setTimeout(r, 1000));
  }
})();
try {
  const page = await chrome.ctx.newPage();
  const crashed = new Promise((r) => page.on('crash', () => r('crash')));
  collectErrors(page, errors, /favicon|openfreemap|AJAXError/, /maplibre-gl/);
  const got = new Map();
  page.on('response', (r) => { const n = Number(r.headers()['content-length'] ?? 0); if (n > 300000) got.set(r.url().replace(/^.*\/(.*?)(\?.*)?$/, '$1'), Math.round(n / MB * 10) / 10); });
  const cdp = await chrome.ctx.newCDPSession(page);
  const measure = async (where) => {
    // a forced collection (here, queryObjects or a deterministic dump) hides what a lazy collector leaves: NOGC=1 for the natural figures
    if (!NOGC) await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
    await new Promise((r) => setTimeout(r, 500));
    // each demand worker's WASM memory and its allocator's live and peak bytes, MB
    const wasm = await page.evaluate(() => window.twDemand?.client?.mem?.map((m) => m && [m.wasm, ...(m.heap ?? [])].map((x) => Math.round(x / 1048576)).join('/'))).catch(() => null);
    const r = { where, ...(await chromeMemory(chrome.profile)), ctx: await contexts(raw, !NOGC), demandWasm: wasm };
    stages.push(r);
    console.log(where, `renderer ${r.renderer} gpu ${r.gpu} total ${r.total} demand wasm/live/peak ${wasm}`);
    console.log('    used+backing+wasm', r.ctx.map(([n, u, b, w]) => `${n} ${u}+${b}+${w}`).join(', '));
    if (process.env.DUMP && !NOGC) dumps[where] = await memoryDump(cdp, where);
  };
  await page.goto(`${BASE}?debug=1&demandWorkers=1${QS}&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
  await page.waitForTimeout(3000);
  await measure('menu');
  stage = 'load';
  await page.tap('.start-menu .new-game .start-new');
  const ready = page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu') && document.querySelector('.guide'), undefined, { timeout: 240000 });
  if ((await Promise.race([ready, crashed])) === 'crash') throw new Error('page crashed during load');
  stage = 'demand';
  await measure('game up');
  const final = page.waitForFunction(() => window.twDemand?.view?.()?.final, undefined, { timeout: 240000 });
  if ((await Promise.race([final, crashed])) === 'crash') throw new Error('page crashed before demand final');
  await page.waitForTimeout(3000);
  await measure('demand final');
  stage = 'camera';
  for (const [lng, lat, z] of [[-73.94, 40.7, 12], [-73.8, 40.75, 13], [-74.1, 40.72, 11], [-73.96, 40.78, 14], [-73.96, 40.75, 11.2]]) {
    await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); }, [[lng, lat], z]); // T-2802: never return the map: Playwright serialises its whole object graph, about 1 GB of garbage on the page
    await page.waitForTimeout(4000);
  }
  await page.waitForTimeout(3000);
  await measure('after camera moves');
  stage = 'line';
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
  const peakOf = (st) => timeline.filter((s) => s.stage === st).reduce((a, s) => (!a || s.renderer > a.renderer ? s : a), null);
  for (const st of ['load', 'demand', 'camera', 'line']) {
    const p = peakOf(st);
    if (p) console.log(`peak in ${st}: renderer ${p.renderer} at ${p.s} s:`, p.ctx.map(([n, u, b]) => `${n} ${u}+${b}`).join(', '));
  }
  console.log('downloads over 0.3 MB', JSON.stringify([...got].sort((a, b) => b[1] - a[1])));
  console.log('peak', peak?.renderer, 'at', peak?.s, 's in', peak?.stage);
  console.log('errors', errors.length, errors.slice(0, 5).join(' | '));
} catch (e) {
  console.log('FAILED', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  sampling = false;
  await sampler;
  await writeFile(join(OUT, `${LABEL}.json`), JSON.stringify({ label: LABEL, qs: QS, lowend: !!process.env.LOWEND, peak, stages, dumps, timeline, errors }, null, 1));
  raw.close();
  await chrome.close();
}
