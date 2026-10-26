// T-1380: what a station edit costs on a big network, main thread and clock worker, at 6x CPU
// throttling (notes/T-1029.md "What is left", notes/T-1380.md).
//   node trainworld/checks/t1380-edit.mjs <page url> [nyc|japan] [label]      (from the maps root)
// The save imported, the clock paused, then at THROTTLE (default 6) a fixed series of station edits:
// stations added, one moved, one moved into another's complex, links made and undone, stations
// removed, a line's stops changed, undo and redo. Per edit: the clock worker's ms (the edit, then the
// snapshot it schedules, whose `buildComplexes` is named), the main thread's ms on the state, in
// `StationLabels.set` and in the world's `complexes`, each under a CPU profile of the page and of the
// clock worker. Output: ../artifacts/trainworld/t1380/<label>-<setup>.json
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1380');
await mkdir(OUT, { recursive: true });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const BASE = process.argv[2] ?? 'http://localhost:6400/';
const SETUP = process.argv[3] ?? 'japan';
const LABEL = process.argv[4] ?? 'run';
const THROTTLE = Number(process.env.THROTTLE ?? 6);
const JAPAN = fileURLToPath(new URL('../../artifacts/trainworld/t737-japan-trains.save', import.meta.url));
const AT = { nyc: [-74.13, 40.585], japan: [140.0, 35.9] };

const NAMES = ['takeStateNow', 'toWorld', 'setNetwork', 'labelGroups', 'changed', 'buildComplexes', 'patchComplexes', 'patchLabels', 'complexes', 'set', 'buildSnapshot', 'complexOfNodes', 'postState', 'buildState', 'complexesOf', 'update', 'renumber', 'plan', 'apply', 'patch', 'rebuild', 'place'];
// names other code uses too: counted only in the files this task changed
const OWN = new Set(['set', 'update', 'plan', 'apply', 'patch', 'rebuild', 'place', 'renumber']);
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
    for (let c = prof.samples[i]; c !== undefined; c = parent.get(c)) {
      const x = byId.get(c), f = x.callFrame.functionName;
      if (OWN.has(f) && !/stationLabels|complexBuilder|labelNames/.test(x.callFrame.url)) continue;
      if (f in out && !seen.has(f)) { seen.add(f); out[f] += us; }
    }
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
  const incl = new Map();
  for (let i = 0; i < prof.samples.length; i++) {
    const us = prof.timeDeltas[i + 1] ?? 0, x0 = byId.get(prof.samples[i]);
    if (/^\((idle|program|root)/.test(x0.callFrame.functionName)) continue;
    const seen = new Set();
    for (let c = prof.samples[i]; c !== undefined; c = parent.get(c)) { const k = nm(byId.get(c)); if (!seen.has(k)) { seen.add(k); incl.set(k, (incl.get(k) ?? 0) + us); } }
  }
  return [...incl].sort((a, b) => b[1] - a[1]).slice(0, n).map(([k, us]) => [k, Math.round(us / 1000)]);
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t1380-', size: [1400, 900] });
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
    if (/^T-(1029|1380)/.test(t)) log('page:', t.slice(0, 400));
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
  out.network = await page.evaluate(() => { const w = window.tw.world.value; return { lines: w.save.lines.length, stations: w.save.stations.length, links: w.save.links.length }; });
  log(SETUP, 'ready', JSON.stringify(out.network));
  await page.evaluate(async ([lng, lat]) => {
    const ui = await import('/src/game/ui.ts');
    window.tw.clock.setRunning(false);
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 14 });
    ui.keepPlacing.value = false;
    ui.tool.value = 'track';
    const m = (window.__m = { states: [], sets: [] });
    const c = window.tw.client, f = c.takeStateNow, l = window.tw.labels, ls = l.set;
    c.takeStateNow = function (msg) { const t = performance.now(); try { return f.call(this, msg); } finally { m.states.push(performance.now() - t); } };
    l.set = function (...a) { const t = performance.now(); try { return ls.apply(this, a); } finally { m.sets.push(performance.now() - t); } };
  }, AT[SETUP]);
  await page.waitForTimeout(3000);
  const cdp = await ctx.newCDPSession(page);
  // the clock worker profiled through the page's target, started and stopped per step
  const pending = new Map();
  let id = 1, workerSession = null;
  cdp.on('Target.receivedMessageFromTarget', (e) => { const m = JSON.parse(e.message); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const attached = new Promise((res) => cdp.on('Target.attachedToTarget', (e) => { if (e.targetInfo.type === 'worker' && /clock/.test(e.targetInfo.url)) { workerSession = e.sessionId; res(); } }));
  await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: false });
  await Promise.race([attached, new Promise((r) => setTimeout(r, 5000))]);
  const wsend = (method, params = {}) => new Promise((res) => { const i = id++; pending.set(i, res); cdp.send('Target.sendMessageToTarget', { sessionId: workerSession, message: JSON.stringify({ id: i, method, params }) }); });
  if (!workerSession) throw new Error('no clock worker target');
  await wsend('Profiler.enable');
  await wsend('Profiler.setSamplingInterval', { interval: 250 });
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: THROTTLE });
  await cdp.send('Profiler.enable');
  await cdp.send('Profiler.setSamplingInterval', { interval: 500 });

  const steps = (process.env.STEPS ?? 'add,extend,station,station2,moveStation,add2,station3,construct,link,unlink,removeStation,removeMerged,setStops,undo,undo2,redo').split(',');
  await page.evaluate(([lng, lat]) => { window.__ctx = { at: [lng, lat] }; }, AT[SETUP]);
  for (const step of steps) {
    await cdp.send('Profiler.start');
    await wsend('Profiler.start');
    const r = await page.evaluate(async (step) => {
      const tw = window.tw, c = tw.client, k = window.__ctx, w = () => tw.world.value;
      const pack = (dLng, dLat) => tw.lngLatToPack(k.at[0] + dLng, k.at[1] + dLat);
      const node = (id) => w().nodes.get(id);
      let op = null;
      if (step === 'add') { const a = pack(0, 0), b = pack(0.004, 0.001); op = { op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false }; }
      else if (step === 'extend') { const b = pack(0.008, -0.001); op = { op: 'route', from: { kind: 'node', node: k.end }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false }; }
      else if (step === 'station') op = { op: 'addStation', at: { kind: 'node', node: k.end }, platform: 200, name: '' };
      else if (step === 'station2') op = { op: 'addStation', at: { kind: 'node', node: k.start }, platform: 200, name: '' };
      else if (step === 'moveStation') { const n = node(k.end); op = { op: 'moveNode', node: k.end, x: n.x + 150, y: n.y + 60 }; }
      else if (step === 'add2') { const n = node(k.start); op = { op: 'route', from: { kind: 'free', x: n.x, y: n.y + 33, level: 0 }, to: { kind: 'free', x: n.x - 300, y: n.y + 400, level: 0 }, pis: [], single: false }; }
      else if (step === 'station3') op = { op: 'addStation', at: { kind: 'node', node: k.near }, platform: 200, name: '' };
      else if (step === 'construct') op = { op: 'constructAll' };
      else if (step === 'link' || step === 'unlink') {
        if (step === 'link') {
          // two built stations 300 to 900 m apart, not yet in one complex
          const st = w().save.stations.filter((s) => s.built), by = new Map();
          for (const s of st) { const key = `${Math.floor(s.x / 1000)},${Math.floor(s.y / 1000)}`; (by.get(key) ?? by.set(key, []).get(key)).push(s); }
          const cx = (await import('/src/game/complexView.ts')).complexes.value;
          for (const [key, v] of by) {
            for (const a of v) for (const b of v) {
              const d = Math.hypot(a.x - b.x, a.y - b.y);
              if (a.num < b.num && d > 300 && d < 900 && cx.of.get(a.id) !== cx.of.get(b.id) && !k.pair) k.pair = [a.num, b.num];
            }
            if (k.pair) break;
          }
        }
        if (k.pair) op = { op: 'link', a: k.pair[0], b: k.pair[1], on: step === 'link' };
      }
      else if (step === 'removeStation') op = { op: 'removeStation', node: k.start };
      else if (step === 'removeMerged') op = { op: 'removeStation', node: k.near };
      else if (step === 'setStops') { const ws = w(), line = ws.save.lines.filter((l) => ws.lineStats[l.id]?.status === 'running' && l.stops.length > 3)[1]; op = { op: 'setStops', line: line.num, stops: line.stops.slice(0, -1).map(Number) }; }
      else op = { op: step.startsWith('undo') ? 'undo' : 'redo' };
      if (!op) return { step, skipped: true };
      const edgesBefore = new Set(w().edges.map((e) => e.id));
      const m = window.__m;
      m.states.length = 0; m.sets.length = 0;
      const t = performance.now();
      const a = await c.edit(op);
      const wall = performance.now() - t;
      await new Promise((r) => setTimeout(r, 400));
      if (step === 'add' || step === 'extend') {
        const fresh = w().edges.filter((e) => !edgesBefore.has(e.id));
        if (fresh.length) { const e = fresh[fresh.length - 1]; if (step === 'add') { k.start = e.a; k.end = e.b; } else k.end = a.node ?? (e.a === k.end ? e.b : e.a); }
        if (a.node !== undefined && step === 'add') k.end = a.node;
      }
      if (step === 'add2') {
        // the new route's node nearer the first station
        const fresh = w().edges.filter((e) => !edgesBefore.has(e.id)), e = fresh[fresh.length - 1], n0 = node(k.start);
        if (e) k.near = Math.hypot(node(e.a).x - n0.x, node(e.a).y - n0.y) < Math.hypot(node(e.b).x - n0.x, node(e.b).y - n0.y) ? e.a : e.b;
      }
      const e = c.editLog.at(-1);
      const r2 = (v) => Math.round(v * 10) / 10;
      return { step, ok: a.ok, issues: a.issues, wall: Math.round(wall), worker: Object.fromEntries(Object.entries(e?.worker ?? {}).map(([k, v]) => [k, r2(v)])), take: m.states.map(r2), labelsSet: m.sets.map(r2), stations: w().save.stations.length };
    }, step);
    // the snapshot follows the state after SNAPSHOT_DEBOUNCE_MS; let it finish
    await page.waitForTimeout(2500);
    const { profile } = await cdp.send('Profiler.stop');
    const wprofile = (await wsend('Profiler.stop')).result.profile;
    r.main = named(profile, NAMES);
    r.workerNamed = named(wprofile, NAMES);
    if (process.env.TOP === '1') { r.mainTop = profileTop(profile, Number(process.env.TOPN ?? 25)); r.workerTop = profileTop(wprofile, Number(process.env.TOPN ?? 25)); }
    out.edits.push(r);
    if (!r.skipped && !r.ok) log('edit refused:', step, JSON.stringify(r.issues));
    log(step.padEnd(14), r.skipped ? 'skipped' : `ok ${r.ok} stations ${r.stations} worker ${JSON.stringify(r.worker)} take ${JSON.stringify(r.take)} labels.set ${JSON.stringify(r.labelsSet)}`);
    if (!r.skipped) log('   main', JSON.stringify(r.main), '\n   worker', JSON.stringify(r.workerNamed));
  }
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
  const ok = out.edits.filter((e) => !e.skipped && e.ok);
  const sum = (f) => Math.round(ok.reduce((s, e) => s + f(e), 0) / Math.max(1, ok.length));
  out.summary = { edits: ok.length, take: sum((e) => e.take.reduce((a, b) => a + b, 0)), labelsSet: sum((e) => e.labelsSet.reduce((a, b) => a + b, 0)), complexesMain: sum((e) => e.main.buildComplexes + e.main.update), complexesWorker: sum((e) => e.workerNamed.buildComplexes + e.workerNamed.update), workerBusy: sum((e) => e.workerNamed.busy) };
  log('mean per station edit', JSON.stringify(out.summary));
} catch (e) {
  failed = true;
  out.error = String(e?.stack ?? e);
  log('error', out.error);
} finally {
  await close();
}
await writeFile(join(OUT, `${LABEL}-${SETUP}.json`), JSON.stringify(out, null, 1));
if (errors.length) { log('page errors', errors.slice(0, 5)); failed = true; }
console.log(`${failed ? 'FAIL' : 'PASS'}  t1380 ${SETUP} ${LABEL}`);
process.exit(failed ? 1 : 0);
