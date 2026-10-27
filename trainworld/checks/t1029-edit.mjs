// T-1029, T-1028, T-1030: a track edit's cost on a big network, and whether patched states and strokes
// match a full rebuild (notes/T-1029.md).
//   node trainworld/checks/t1029-edit.mjs <page url> [setup] [label]      (from the maps root)
// setup: nyc (the real save) or japan (../artifacts/trainworld/t737-japan-trains.save). The clock is
// paused; after the load (Japan: JAPAN_WAIT s, demand not waited for; New York: demand's final) the CPU
// is throttled THROTTLE times (default 6) and a fixed series of edits runs: track added free and from
// its end, its end dragged, a station on a running line's track, stations added and removed, a train count,
// a line's last stop dropped, track removed, undo and redo. Per edit: the clock worker's ms by step,
// the main thread's on the state (taking it, `setNetwork`), then six pointer moves with the track tool
// (its snap and the pick grids), all under a CPU profile whose named functions are summed.
// DUMP=1: after the series, unthrottled, the world, the renderer's buffers and what is on the GPU, as
// hashes per field, to OUT/<label>-<setup>-dump.json; compare two labels with `--compare a b`.
// PATCH (default unset: the app's own default) goes into the URL as `patch=` (0 full, check both).
// Output: ../artifacts/trainworld/t1029/<label>-<setup>.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1029');
await mkdir(OUT, { recursive: true });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

if (process.argv[2] === '--compare') {
  const [a, b] = await Promise.all(process.argv.slice(3, 5).map(async (f) => JSON.parse(await readFile(join(OUT, f), 'utf8'))));
  let diff = 0;
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) { diff++; console.log('differs', k, JSON.stringify(a[k])?.slice(0, 200), JSON.stringify(b[k])?.slice(0, 200)); }
  console.log(`${diff ? 'FAIL' : 'PASS'}  ${Object.keys(a).length} fields, ${diff} differ`);
  process.exit(diff ? 1 : 0);
}

const BASE = process.argv[2] ?? 'http://localhost:6129/';
const SETUP = process.argv[3] ?? 'nyc';
const LABEL = process.argv[4] ?? 'run';
const THROTTLE = Number(process.env.THROTTLE ?? 6);
const JAPAN = fileURLToPath(new URL('../../artifacts/trainworld/t737-japan-trains.save', import.meta.url));
// open ground beside each network, and where the view sits for the pointer moves
const AT = { nyc: [-74.13, 40.585], japan: [140.0, 35.9] };

const NAMES = ['takeStateNow', 'toWorld', 'setNetwork', 'CellIndex', 'carry', 'carryNow', 'topSeg', 'edgesNear', 'nodesNear', 'followTrack', 'receive', 'takeRows', 'mergeNet'];
function named(prof, names) {
  const byId = new Map(prof.nodes.map((x) => [x.id, x]));
  const parent = new Map();
  for (const x of prof.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const out = Object.fromEntries(names.map((n) => [n, 0]));
  let busy = 0;
  for (let i = 0; i < prof.samples.length; i++) {
    const us = prof.timeDeltas[i + 1] ?? 0, seen = new Set();
    const f0 = byId.get(prof.samples[i]).callFrame.functionName;
    if (f0 !== '(idle)' && f0 !== '(program)' && f0 !== '(root)') busy += us;
    for (let c = prof.samples[i]; c !== undefined; c = parent.get(c)) { const f = byId.get(c).callFrame.functionName; if (f in out && !seen.has(f)) { seen.add(f); out[f] += us; } }
  }
  for (const k in out) out[k] = Math.round(out[k] / 1000);
  out.busy = Math.round(busy / 1000);
  return out;
}
function profileTop(prof, n = 40) {
  const byId = new Map(prof.nodes.map((x) => [x.id, x]));
  const parent = new Map();
  for (const x of prof.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const nm = (x) => `${x.callFrame.functionName || '(anon)'} ${x.callFrame.url.replace(/.*\//, '').replace(/\?.*/, '')}:${x.callFrame.lineNumber + 1}`;
  const incl = new Map(), self = new Map();
  for (let i = 0; i < prof.samples.length; i++) {
    const us = prof.timeDeltas[i + 1] ?? 0, x0 = byId.get(prof.samples[i]);
    if (/^\((idle|program|root)/.test(x0.callFrame.functionName)) continue;
    self.set(nm(x0), (self.get(nm(x0)) ?? 0) + us);
    const seen = new Set();
    for (let c = prof.samples[i]; c !== undefined; c = parent.get(c)) { const k = nm(byId.get(c)); if (!seen.has(k)) { seen.add(k); incl.set(k, (incl.get(k) ?? 0) + us); } }
  }
  const top = (m) => [...m].sort((a, b) => b[1] - a[1]).slice(0, n).map(([k, us]) => [k, Math.round(us / 1000)]);
  return { inclusive: top(incl), self: top(self) };
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t1029-', size: [1400, 900] });
const errors = [];
const out = { label: LABEL, setup: SETUP, base: BASE, throttle: THROTTLE, errors, edits: [] };
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  // ?patch=check: a difference found in the clock worker is a console error there
  page.on('worker', (w) => { try { w.on('console', (m) => { if (m.type() === 'error') errors.push('worker: ' + m.text().slice(0, 300)); }); } catch { /* an older Playwright */ } });
  page.on('console', (msg) => {
    const t = msg.text();
    if (/^T-(1029|1381)/.test(t)) log('page:', t.slice(0, 400));
    if (msg.type() === 'error' && !/favicon|openfreemap|AJAXError|^\w{1,3}$/.test(t) && !/maplibre-gl/.test(msg.location()?.url ?? '')) errors.push(t.slice(0, 300));
  });
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
  });
  const patch = process.env.PATCH ? `&patch=${process.env.PATCH}` : '';
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1${patch}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.client.preview.length === 2, undefined, { timeout: 180000 });
  await page.waitForTimeout(1000);
  await importSave(page, await readFile(SETUP === 'japan' ? JAPAN : REAL_SAVE));
  await page.waitForFunction((n) => window.tw.client.lastLoad?.from === 'file' && (window.tw.world.value?.save?.lines?.length ?? 0) >= n, SETUP === 'japan' ? 500 : 100, { timeout: 600000, polling: 500 });
  if (SETUP === 'nyc') await demandFinal(page, { timeout: 600000 }).catch(() => log('no demand final in 10 min'));
  else await page.waitForTimeout(Number(process.env.JAPAN_WAIT ?? 60) * 1000);
  out.network = await page.evaluate(() => { const w = window.tw.world.value, n = window.tw.renderer.net; return { lines: w.save.lines.length, stations: w.save.stations.length, edges: w.edges.length, trackSegs: n?.track.count, lineSegs: n?.lines.count }; });
  log(SETUP, 'ready', JSON.stringify(out.network));
  // the view on open ground beside the network, the track tool on, keep placing off
  await page.evaluate(async ([lng, lat]) => {
    const ui = await import('/src/game/ui.ts');
    window.tw.clock.setRunning(false);
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 14 });
    ui.keepPlacing.value = false;
    ui.tool.value = 'track';
    // the main thread's ms on each state and each setNetwork
    const m = (window.__m = { states: [], nets: [] });
    const c = window.tw.client, f = c.takeStateNow, r = window.tw.renderer, sn = r.setNetwork;
    c.takeStateNow = function (msg) { const t = performance.now(); try { return f.call(this, msg); } finally { m.states.push(performance.now() - t); } };
    r.setNetwork = function (n) { const t = performance.now(); try { return sn.call(this, n); } finally { m.nets.push(performance.now() - t); } };
  }, AT[SETUP]);
  await page.waitForTimeout(3000);
  const cdp = await ctx.newCDPSession(page);
  // WORKER_PROFILE=1: the clock worker profiled over the whole series too, through the page's target
  // (Target.sendMessageToTarget, since a Playwright session cannot be opened on a worker)
  let wprof = null;
  if (process.env.WORKER_PROFILE === '1') {
    const pending = new Map();
    let id = 1, workerSession = null;
    cdp.on('Target.receivedMessageFromTarget', (e) => { const m = JSON.parse(e.message); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
    const attached = new Promise((res) => cdp.on('Target.attachedToTarget', (e) => { if (e.targetInfo.type === 'worker' && /clock/.test(e.targetInfo.url)) { workerSession = e.sessionId; res(); } }));
    await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
    await Promise.race([attached, new Promise((r) => setTimeout(r, 5000))]);
    if (workerSession) {
      const wsend = (method, params = {}) => new Promise((res) => { const i = id++; pending.set(i, res); cdp.send('Target.sendMessageToTarget', { sessionId: workerSession, message: JSON.stringify({ id: i, method, params }) }); });
      await wsend('Profiler.enable');
      await wsend('Profiler.setSamplingInterval', { interval: 250 });
      await wsend('Profiler.start');
      wprof = { stop: async () => (await wsend('Profiler.stop')).result.profile };
    } else log('no clock worker target');
  }
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });

  // the series: each step a function of what the steps before it made (`ctx`)
  const steps = ['add', 'extend', 'drag', 'lineStation', 'station', 'station2', 'removeStation', 'trains', 'dropStop', 'removeEdge', 'undo', 'undo2', 'redo', 'undo3'];
  await page.evaluate(([lng, lat]) => { window.__ctx = { at: [lng, lat] }; }, AT[SETUP]);
  for (const step of steps) {
    await cdp.send('Profiler.start');
    const r = await page.evaluate(async (step) => {
      const tw = window.tw, c = tw.client, k = window.__ctx, w = () => tw.world.value;
      const pack = (dLng, dLat) => tw.lngLatToPack(k.at[0] + dLng, k.at[1] + dLat);
      let op = null;
      if (step === 'add') { const a = pack(0, 0), b = pack(0.004, 0.001); op = { op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false }; }
      else if (step === 'extend') { const b = pack(0.008, -0.001); op = { op: 'route', from: { kind: 'node', node: k.end }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false }; }
      else if (step === 'drag') { const n = w().nodes.get(k.end); op = { op: 'moveNode', node: k.end, x: n.x + 60, y: n.y - 40 }; }
      else if (step === 'lineStation') {
        // a plain node in the middle of a running line's path
        const ws = w();
        const line = ws.save.lines.find((l) => ws.lineStats[l.id]?.status === 'running' && l.stops.length > 3);
        const stops = new Set(line.stops.map(Number));
        const onLine = ws.edges.filter((e) => e.lines.includes(line.id));
        const ends = new Set([Number(line.stops[0]), Number(line.stops[line.stops.length - 1])]);
        const ids = onLine.flatMap((e) => [e.a, e.b]);
        // built track cannot be moved: a station on a plain node of the line, between its stops
        const cand = ids.find((id) => { const n = ws.nodes.get(id); return n && !stops.has(id) && !ends.has(id) && n.ports === 2 && !n.platform; });
        if (cand !== undefined) op = { op: 'addStation', at: { kind: 'node', node: cand }, platform: 120, name: '' };
      }
      else if (step === 'station') op = { op: 'addStation', at: { kind: 'node', node: k.end }, platform: 200, name: '' };
      else if (step === 'station2') op = { op: 'addStation', at: { kind: 'node', node: k.start }, platform: 200, name: '' };
      else if (step === 'removeStation') op = { op: 'removeStation', node: k.start };
      else if (step === 'trains') { const ws = w(), line = ws.save.lines.find((l) => ws.lineStats[l.id]?.status === 'running'); const t = ws.lineStats[line.id].trainsNeeded; op = { op: 'setTrainCount', line: line.num, trains: [t[0] + 1, t[1] + 1, t[2]] }; }
      else if (step === 'dropStop') { const ws = w(), line = ws.save.lines.filter((l) => ws.lineStats[l.id]?.status === 'running' && l.stops.length > 3)[1]; op = { op: 'setStops', line: line.num, stops: line.stops.slice(0, -1).map(Number) }; }
      else if (step === 'removeEdge') op = { op: 'removeEdge', edge: k.lastEdge };
      else op = { op: step.startsWith('undo') ? 'undo' : 'redo' };
      if (!op) return { step, skipped: true };
      const edgesBefore = new Set(w().edges.map((e) => e.id));
      const m = window.__m;
      m.states.length = 0; m.nets.length = 0;
      const t = performance.now();
      const a = await c.edit(op);
      const wall = performance.now() - t;
      await new Promise((r) => setTimeout(r, 300));
      if (step === 'add' || step === 'extend') {
        const fresh = w().edges.filter((e) => !edgesBefore.has(e.id));
        if (fresh.length) { k.lastEdge = fresh[fresh.length - 1].id; const e = fresh[fresh.length - 1]; if (step === 'add') { k.start = e.a; k.end = e.b; } else k.end = a.node ?? (e.a === k.end ? e.b : e.a); }
        if (a.node !== undefined && step === 'add') k.end = a.node;
      }
      const e = c.editLog.at(-1);
      const r2 = (v) => Math.round(v * 10) / 10;
      return { step, ok: a.ok, wall: Math.round(wall), worker: Object.fromEntries(Object.entries(e?.worker ?? {}).map(([k, v]) => [k, r2(v)])), take: m.states.map(r2), setNetwork: m.nets.map(r2) };
    }, step);
    // six pointer moves with the track tool: its snap and the pick grids for the new network
    const t0 = Date.now();
    for (let i = 0; i < 6; i++) { await page.mouse.move(box[0] - 150 + 50 * i, box[1] + (i % 2 ? 20 : -20)); await page.waitForTimeout(60); }
    await page.waitForTimeout(1500);
    const { profile } = await cdp.send('Profiler.stop');
    r.movesMs = Date.now() - t0;
    r.named = named(profile, NAMES);
    if (process.env.TOP === '1') r.top = profileTop(profile, 25);
    out.edits.push(r);
    if (!r.skipped && !r.ok) failed = true;
    log(step.padEnd(14), r.skipped ? 'skipped' : `ok ${r.ok} worker ${JSON.stringify(r.worker)} take ${JSON.stringify(r.take)} setNetwork ${JSON.stringify(r.setNetwork)} named ${JSON.stringify(r.named)}`);
  }
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  out.patched = await page.evaluate(() => window.tw.renderer.patched ?? null);
  log('networks taken as changes', out.patched);
  if (wprof) {
    const p = await wprof.stop();
    await writeFile(join(OUT, `${LABEL}-${SETUP}-worker.cpuprofile`), JSON.stringify(p));
    out.workerTop = profileTop(p, 40);
    log('worker inclusive', JSON.stringify(out.workerTop.inclusive.slice(0, 40)));
    log('worker self', JSON.stringify(out.workerTop.self.slice(0, 25)));
  }
  const sum = (f) => out.edits.filter((e) => !e.skipped && e.step !== 'trains').reduce((s, e) => s + f(e), 0);
  const n = out.edits.filter((e) => !e.skipped && e.step !== 'trains').length;
  out.summary = {
    trackEdits: n,
    workerTotal: Math.round(sum((e) => e.worker.total ?? 0) / n), workerState: Math.round(sum((e) => e.worker.state ?? 0) / n), workerBuffers: Math.round(sum((e) => e.worker.buffers ?? 0) / n),
    take: Math.round(sum((e) => e.take.reduce((a, b) => a + b, 0)) / n), setNetwork: Math.round(sum((e) => e.setNetwork.reduce((a, b) => a + b, 0)) / n),
    mainBusy: Math.round(sum((e) => e.named.busy) / n), cellIndex: Math.round(sum((e) => e.named.CellIndex) / n), topSeg: Math.round(sum((e) => e.named.topSeg) / n), receive: Math.round(sum((e) => e.named.receive) / n),
  };
  log('mean per track edit', JSON.stringify(out.summary));

  if (process.env.DUMP === '1') {
    await page.waitForTimeout(3000);
    const dump = await page.evaluate(async () => {
      const tw = window.tw, r = tw.renderer, n = r.net, gl = r.gl;
      const h = async (x) => {
        const bytes = x instanceof ArrayBuffer ? new Uint8Array(x) : ArrayBuffer.isView(x) ? new Uint8Array(x.buffer, x.byteOffset, x.byteLength) : new TextEncoder().encode(JSON.stringify(x, (k, v) => v instanceof Map ? [...v] : v instanceof Set ? [...v] : ArrayBuffer.isView(v) ? Array.from(v) : v));
        const d = await crypto.subtle.digest('SHA-256', bytes.slice());
        return [...new Uint8Array(d)].slice(0, 8).map((b) => b.toString(16).padStart(2, '0')).join('') + `/${bytes.length}`;
      };
      const out = {};
      const wv = { ...tw.world.value, version: 0, save: { ...tw.world.value.save, clock: 0, version: 0 } };
      for (const k of Object.keys(wv).sort()) out['world.' + k] = await h(wv[k]);
      for (const k of Object.keys(n).sort()) {
        const v = n[k];
        if (k === 'anchor' || k === 'patch' || k === 'serial') continue;
        if (v && typeof v === 'object' && 'seg' in v && 'count' in v) { for (const f of Object.keys(v).sort()) out[`net.${k}.${f}`] = await h(v[f]); }
        else if (k === 'edgePts') out['net.edgePts'] = await h([...v].map(([id, p]) => [id, Array.from(p)]));
        else out['net.' + k] = await h(v);
      }
      // what is on the GPU: each stroke VAO's attribute buffers, read back
      const read = (buf, bytes) => { const o = new Uint8Array(bytes); gl.bindBuffer(gl.COPY_READ_BUFFER, buf); gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, o); gl.bindBuffer(gl.COPY_READ_BUFFER, null); return o; };
      const gpu = r.gpuDump?.() ?? null;
      if (gpu) for (const [k, [buf, bytes]] of Object.entries(gpu)) out['gpu.' + k] = await h(read(buf, bytes));
      return out;
    });
    await writeFile(join(OUT, `${LABEL}-${SETUP}-dump.json`), JSON.stringify(dump, null, 1));
    log('dump', Object.keys(dump).length, 'fields');
  }
} catch (e) {
  failed = true;
  out.error = String(e?.stack ?? e);
  log('error', out.error);
} finally {
  await close();
}
await writeFile(join(OUT, `${LABEL}-${SETUP}.json`), JSON.stringify(out, null, 1));
if (errors.length) { log('page errors', errors.slice(0, 5)); failed = true; }
console.log(`${failed ? 'FAIL' : 'PASS'}  t1029 ${SETUP} ${LABEL}`);
process.exit(failed ? 1 : 0);
