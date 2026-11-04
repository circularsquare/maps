// T-2122: a new game in a big zone on an emulated phone (or desktop), 1, 3 and 5 lines built, then
// the game exported and loaded again in a fresh page; memory sampled throughout: every Chrome process
// of this profile (private bytes), each worker's JS heap (CDP Runtime.getHeapUsage) and WASM (the
// client's memory()). env: BASE, ZONE (jak), PHONE (1: 412x915 touch viewport), DM (deviceMemory
// override, '' = leave), HC (hardwareConcurrency override), WORKERS (demandWorkers, '' = none), EXTRA,
// TAG, STEPS ("1,3,5,load"), PORT (CDP), CAP (MB: stop and count as a crash once the renderer passes it)
import { writeFile, readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { join } from 'node:path';
import { launchChrome, demandFinal } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6160';
const ZONE = process.env.ZONE ?? 'jak';
const PHONE = process.env.PHONE !== '0';
const DM = process.env.DM ?? (PHONE ? '4' : '');
const HC = process.env.HC ?? (PHONE ? '8' : '');
const WORKERS = process.env.WORKERS ?? '';
const EXTRA = process.env.EXTRA ?? '';
const TAG = process.env.TAG ?? `${ZONE}-${PHONE ? 'phone' : 'desk'}`;
const STEPS = (process.env.STEPS ?? '1,3,5,load').split(',');
const PORT = process.env.PORT ?? '9431';
const CAP = Number(process.env.CAP ?? 0);
// SCR: a scratch folder for the browser profile; OUT: where JSON, logs and the exported save go
const SCR = process.env.SCR ?? 'C:/Users/anita/AppData/Local/Temp/t2122';
const OUT = process.env.OUT ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/t2122';
const Z = JSON.parse(await readFile(new URL('./lines.json', import.meta.url), 'utf8'))[ZONE];

const opts = PHONE ? { size: [412, 915], deviceScaleFactor: 2.625, isMobile: true, hasTouch: true, ...(process.env.UA ? { userAgent: process.env.UA } : {}) } : {};
const { ctx, close, profile } = await launchChrome({ prefix: join(SCR, `prof-${TAG}-`), ...opts, acceptDownloads: true, args: ['--enable-unsafe-swiftshader', '--enable-precise-memory-info', `--remote-debugging-port=${PORT}`] });
const pdir = profile.split(/[\\/]/).pop();
const t0 = Date.now();
const log = (...a) => console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s`, ...a);
if (DM || HC) await ctx.addInitScript(([dm, hc]) => {
  if (dm) Object.defineProperty(Navigator.prototype, 'deviceMemory', { get: () => (dm === 'none' ? undefined : Number(dm)), configurable: true });
  if (hc) Object.defineProperty(Navigator.prototype, 'hardwareConcurrency', { get: () => Number(hc), configurable: true });
}, [DM, HC]);

function procs() {
  return new Promise((res) => {
    const ps = `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${pdir}*' } | ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId, $_.PrivatePageCount, ($(if ($_.CommandLine -match '--type=([a-z-]+)') { $matches[1] } else { 'browser' })) }`;
    execFile('powershell.exe', ['-NoProfile', '-Command', ps], { timeout: 15000 }, (err, out) => {
      if (err) return res([]);
      res(out.trim().split(/\r?\n/).filter(Boolean).map((l) => { const [pid, priv, type] = l.split('|'); return { pid: +pid, mb: Math.round(+priv / 1048576), type }; }));
    });
  });
}

// raw CDP: each worker's and page's JS heap
const ver = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json();
const ws = new WebSocket(ver.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let nid = 1;
const wait = new Map();
ws.addEventListener('message', (ev) => { const m = JSON.parse(ev.data); if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); } });
const send = (method, params = {}, sessionId) => new Promise((r) => { const id = nid++; wait.set(id, r); ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) })); setTimeout(() => { if (wait.has(id)) { wait.delete(id); r({ timeout: true }); } }, 3000); });
const sessions = new Map();
async function heaps() {
  const t = await send('Target.getTargets');
  const out = await Promise.all((t.result?.targetInfos ?? []).filter((ti) => /worker|page/.test(ti.type) && /localhost/.test(ti.url)).map(async (ti) => {
    let s = sessions.get(ti.targetId);
    if (!s) { const a = await send('Target.attachToTarget', { targetId: ti.targetId, flatten: true }); s = a.result?.sessionId; if (!s) return null; sessions.set(ti.targetId, s); }
    const h = await send('Runtime.getHeapUsage', {}, s);
    return { id: ti.targetId.slice(0, 6), type: ti.type, url: ti.url.replace(/^.*\//, '').replace(/\?.*$/, '').slice(0, 30), used: h.result ? Math.round(h.result.usedSize / 1048576) : null, total: h.result ? Math.round(h.result.totalSize / 1048576) : null, ab: h.result?.backingStorageSize !== undefined ? Math.round(h.result.backingStorageSize / 1048576) : undefined };
  }));
  return out.filter(Boolean);
}

let page, crashed = false, phase = 'start';
const samples = [];
async function pageMem() {
  if (!page || crashed) return null;
  try {
    return await Promise.race([page.evaluate(() => {
      const c = window.twDemand?.client;
      return { workers: c?.memory?.() ?? null, focus: c?.fpool ? (c.fpool.mem ?? []).map((b) => Math.round((b ?? 0) / 1048576)) : null, n: c?.computeN };
    }), new Promise((r) => setTimeout(() => r({ timeout: true }), 4000))]);
  } catch (e) { return { err: String(e).slice(0, 80) }; }
}
let sampling = true;
(async () => {
  while (sampling) {
    const t = (Date.now() - t0) / 1000;
    const [p, m, js] = await Promise.all([procs(), pageMem(), heaps().catch(() => [])]);
    const rend = Math.max(0, ...p.filter((x) => x.type === 'renderer').map((x) => x.mb));
    const total = p.reduce((a, x) => a + x.mb, 0);
    samples.push({ t, phase, total, rend, procs: p, page: m, js });
    if (CAP && rend > CAP && !crashed) { crashed = true; log(`OVER CAP: renderer ${rend} MB > ${CAP}`); }
    if (process.env.VERBOSE) log(phase, 'rend', rend, 'total', total, JSON.stringify(m?.workers?.map((w) => w.wasm)), js.map((x) => `${x.url}:${x.used}`).join(' '));
    await new Promise((r) => setTimeout(r, 1500));
  }
})();

function attach(p) {
  p.on('crash', () => { crashed = true; log('PAGE CRASH'); });
  p.on('console', (m) => { const t = m.text(); if (/demand:|memory|panic|failed|T-1197|workers:/.test(t) && !/\[vite\]/.test(t)) log('console', m.type(), t.slice(0, 240)); });
  p.on('pageerror', (e) => log('pageerror', e.message.slice(0, 200)));
}
const q = `debug=1&paused=1&mode=sandbox&opened=all${WORKERS ? `&demandWorkers=${WORKERS}` : ''}${EXTRA}`;
page = await ctx.newPage();
attach(page);
phase = 'open';
await page.goto(`${BASE}/?${q}&new=1&v=${Date.now()}`);
await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });
log('pool', await page.evaluate(() => [window.twDemand.client.computeN, navigator.deviceMemory, navigator.hardwareConcurrency]));
await page.evaluate((ll) => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); }, Z.centre);
await page.waitForFunction((id) => window.twDemand.client.start.loads.find((l) => l.item === id)?.openedAt, ZONE, { timeout: 300000 });
await new Promise((r) => setTimeout(r, 8000));
log('zone open');

const res = { tag: TAG, url: q, phone: PHONE, dm: DM, hc: HC, steps: {} };
const summarise = (ph) => {
  const ss = samples.filter((s) => s.phase === ph);
  if (!ss.length) return null;
  const pk = ss.reduce((a, s) => (s.rend > a.rend ? s : a), ss[0]);
  const wasm = {}, js = {};
  for (const s of ss) {
    for (const w of s.page?.workers ?? []) wasm[w.aux ? 'aux' : 'w' + w.worker] = Math.max(wasm[w.aux ? 'aux' : 'w' + w.worker] ?? 0, w.wasm);
    (s.page?.focus ?? []).forEach((b, i) => { wasm['focus' + i] = Math.max(wasm['focus' + i] ?? 0, b); });
    for (const h of s.js ?? []) { const k = `${h.url}#${h.id}`; js[k] = Math.max(js[k] ?? 0, h.total ?? 0); }
  }
  return { peakRenderer: pk.rend, peakAll: Math.max(...ss.map((s) => s.total)), atPeak: { procs: pk.procs, workers: pk.page?.workers, js: pk.js }, wasmPeak: wasm, jsTotalPeak: js, secs: ss.at(-1).t - ss[0].t };
};
res.steps.open = summarise('open');

// the browser's own breakdown, by worker url and kind (a cross-origin isolated page)
async function uaMem() {
  try {
    return await Promise.race([page.evaluate(async () => {
      if (!performance.measureUserAgentSpecificMemory) return null;
      const m = await performance.measureUserAgentSpecificMemory();
      const by = {};
      for (const b of m.breakdown) {
        const url = (b.attribution[0]?.url ?? '').replace(/^.*\//, '').replace(/\?.*$/, '').slice(0, 30) || '-';
        const k = `${url}:${b.types.join('+') || '-'}`;
        by[k] = (by[k] ?? 0) + Math.round(b.bytes / 1048576);
      }
      return { total: Math.round(m.bytes / 1048576), by };
    }), new Promise((r) => setTimeout(() => r({ timeout: true }), 60000))]);
  } catch (e) { return { err: String(e).slice(0, 80) }; }
}

let built = 0;
async function buildTo(n) {
  phase = `L${n}`;
  const t = Date.now();
  const v = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  const lines = Z.lines.slice(built, n);
  await page.evaluate(async ([lines, first]) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    for (const [k, L] of lines.entries()) {
      const pts = L.stops.map((s) => t.lngLatToPack(s[1], s[2]));
      const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
      const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
      const nodes = [];
      for (let i = 1; i < path.length; i++) {
        const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: L.level } : { kind: 'node', node: nodes.at(-1) };
        await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: L.level }, pis: [], single: false });
        const e = t.world.value.edges.at(-1);
        if (i === 1) nodes.push(e.a);
        nodes.push(e.b);
      }
      const at = nodes.slice(1, -1);
      for (let i = 0; i < L.stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: L.stops[i][0] });
      const line = (await run({ op: 'addLine', stops: at, name: `T2122 ${first + k + 1}`, colour: ['#2a7ab0', '#c0392b', '#27ae60', '#8e44ad', '#d35400'][(first + k) % 5] })).line;
      await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    }
    await run({ op: 'constructAll' });
  }, [lines, built]);
  built = n;
  let ok = true;
  try { await Promise.race([demandFinal(page, { after: v, timeout: 600000 }), new Promise((_, rej) => { const i = setInterval(() => { if (crashed) { clearInterval(i); rej(Error('crash')); } }, 500); })]); }
  catch (e) { ok = false; log('no final:', String(e).slice(0, 100)); }
  const finalS = (Date.now() - t) / 1000;
  if (ok) await new Promise((r) => setTimeout(r, 15000));
  const rail = ok ? await page.evaluate(() => Math.round(window.twDemand.view().railTripsPerDay)).catch(() => null) : null;
  const uasm = ok ? await uaMem() : null;
  const tables = await page.evaluate(() => [...window.twDemand.client.corridorTables.keys()]).catch(() => null);
  res.steps[`L${n}`] = { ok, finalS, rail, tables, uasm, ...summarise(`L${n}`) };
  log('tables', JSON.stringify(tables));
  log(`L${n}`, ok ? 'final' : 'FAILED', finalS, 's rail', rail, 'renderer peak', res.steps[`L${n}`].peakRenderer, 'wasm', JSON.stringify(res.steps[`L${n}`].wasmPeak));
  return ok;
}

for (const s of STEPS) {
  if (s === 'load') break;
  if (!(await buildTo(Number(s)))) break;
}

if (STEPS.includes('load') && !crashed) {
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const d = await dl;
  const file = join(OUT, `${TAG}.save`);
  await d.saveAs(file);
  log('exported', file);
  await page.close();
  await new Promise((r) => setTimeout(r, 4000));
  page = await ctx.newPage();
  attach(page);
  phase = 'load';
  const t = Date.now();
  await page.goto(`${BASE}/?${q}&new=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });
  const bytes = await readFile(file);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  let ok = true;
  try { await Promise.race([demandFinal(page, { lines: 0, timeout: 600000 }), new Promise((_, rej) => { const i = setInterval(() => { if (crashed) { clearInterval(i); rej(Error('crash')); } }, 500); })]); }
  catch (e) { ok = false; log('load: no final', String(e).slice(0, 100)); }
  const finalS = (Date.now() - t) / 1000;
  if (ok) await new Promise((r) => setTimeout(r, 15000));
  const rail = ok ? await page.evaluate(() => Math.round(window.twDemand.view().railTripsPerDay)).catch(() => null) : null;
  res.steps.load = { ok, finalS, rail, uasm: ok ? await uaMem() : null, ...summarise('load') };
  log('load', ok ? 'final' : 'FAILED', finalS, 's rail', rail, 'renderer peak', res.steps.load.peakRenderer, 'wasm', JSON.stringify(res.steps.load.wasmPeak));
}
sampling = false;
await new Promise((r) => setTimeout(r, 2000));
res.crashed = crashed;
const brief = Object.fromEntries(Object.entries(res.steps).map(([k, v]) => [k, v && { ok: v.ok, finalS: v.finalS, rail: v.rail, peakRenderer: v.peakRenderer, peakAll: v.peakAll, wasm: v.wasmPeak, js: v.jsTotalPeak }]));
log(JSON.stringify(brief));
await writeFile(`${OUT}/${TAG}.json`, JSON.stringify({ ...res, brief, samples }, null, 1));
ws.close();
await Promise.race([close(), new Promise((r) => setTimeout(r, 30000))]);
process.exit(0);
