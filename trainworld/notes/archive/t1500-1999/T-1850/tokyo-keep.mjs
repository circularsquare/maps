// T-1862: Tokyo real-network save, a 3-station Ceuta line; then a fresh load of the save with the line.
// Records plan/done messages, the time demandUpdating takes to clear, and every line's riders at the
// load's final, after the line, and at the fresh load's final. env: BASE, WORKERS, OUT, PERF (perfOff)
import { readFile, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { launchChrome, importSave, demandFinal, buildLine, collectErrors, ANGLE } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6161/';
const SAVE = 'C:/Users/anita/projects/maps/trainworld/data/packs/real/tyo.save';
const W = process.env.WORKERS ?? '2';
const PERF = process.env.PERF ? `&perfOff=${process.env.PERF}` : '';
const OUT = process.env.OUT ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/t1862/tokyo-ceuta.json';
const reg = JSON.parse(readFileSync('C:/Users/anita/projects/maps/trainworld/app/src/game/registry.json', 'utf8'));
const z = (reg.cities ?? reg.zones ?? reg).find?.((c) => c.id === 'ceu');
const [lon, lat] = z.center;
const dLng = 1.2 / (111.32 * Math.cos((lat * Math.PI) / 180)), dLat = 1.2 / 111.32;
const pts = [[lon - dLng, lat - dLat * 0.2], [lon, lat], [lon + dLng, lat + dLat * 0.2]];

const hook = () => {
  const log = (window.__wlog = []);
  let n = 0;
  const sum = (m) => {
    switch (m.kind) {
      case 'solve': return { v: m.version, barrier: !!m.barrier, only: m.only, freshParts: !!m.freshParts, exactKeep: !!m.exactKeep, lines: m.net ? m.net.lineN.length : null };
      case 'plan': return { v: m.version, solving: (m.solving ?? []).map((c) => c.length + ':' + c.slice(0, 3).join(',')), kept: (m.kept ?? []).length, keptIds: (m.kept ?? []).slice(0, 6), local: m.local };
      case 'done': return { v: m.version, clusters: m.clusters, periods: m.periods };
      case 'failed': case 'died': return { error: String(m.error ?? '') };
      default: return null;
    }
  };
  const keep = new Set(['solve', 'plan', 'done', 'failed', 'died', 'whole']);
  const W0 = window.Worker;
  window.Worker = class extends W0 {
    constructor(url, opts) {
      super(url, opts);
      this.__id = `${n++}`;
      super.addEventListener('message', (e) => { const m = e.data; if (m && keep.has(m.kind)) log.push({ t: Math.round(performance.now()), w: this.__id, dir: '<', kind: m.kind, ...sum(m) }); });
    }
    postMessage(m, t) {
      try { if (m && keep.has(m.kind)) log.push({ t: Math.round(performance.now()), w: this.__id, dir: '>', kind: m.kind, ...sum(m) }); } catch {}
      return super.postMessage(m, t);
    }
  };
};
const riders = (page) => page.evaluate(() => { const v = window.twDemand.view(); return { version: v.version, final: v.final, lines: Object.fromEntries([...v.lines].map(([id, l]) => [id, { r: l.ridersPerDay, b: Array.from(l.boardings ?? []) }])) }; });

const errors = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t1862-', args: ANGLE });
const out = { workers: W, perf: PERF };
try {
  const page = await ctx.newPage();
  await page.addInitScript(hook);
  collectErrors(page, errors);
  const clog = [];
  page.on('console', (m) => { const t = m.text(); if (/T1862|T1733|T1609/.test(t)) clog.push(t.slice(0, 300)); });
  await page.goto(`${BASE}?debug=1&continue=1&paused=1&demandWorkers=${W}${PERF}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });
  const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 400000 });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 120000 });
  await page.goto(`${BASE}?debug=1&demandWorkers=${W}${PERF}`);
  await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
  let t = Date.now();
  await page.evaluate(() => document.querySelector('.start-menu .start-save').click());
  await demandFinal(page, { held: true, timeout: 600000 });
  out.loadS = (Date.now() - t) / 1000;
  console.log('load final after s', out.loadS);
  await page.waitForTimeout(3000);
  out.loadFinal = await riders(page);
  await page.evaluate(([lon, lat]) => window.tw.map.jumpTo({ center: [lon, lat], zoom: 12 }), [lon, lat]);
  await page.waitForTimeout(6000);
  await page.evaluate(async () => { window.__d = await import('/src/game/demand.ts'); });
  const v0 = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
  const r = await buildLine(page, pts, { trains: [10, 8, 6], wait: { route: 300, station: 150, settle: 300 } });
  console.log('buildLine ok', r?.ok, JSON.stringify(r?.issues ?? []));
  const t0 = Date.now(), pt0 = await page.evaluate(() => performance.now());
  // the time the last solve's final clears demandUpdating, once no newer solve has come for 15 s
  const settle = async (t0, v0) => {
    const clears = [];
    let clearedVer = -1, clearedAt = 0;
    for (; Date.now() - t0 < 400000;) {
      const s = await page.evaluate(() => ({ upd: window.__d.demandUpdating.value, ver: window.twDemand.log.at(-1)?.version ?? 0, view: window.twDemand.view()?.version, final: !!window.twDemand.view()?.final, waiting: !!window.twDemand.client.waiting })).catch(() => null);
      const clear = s && !s.upd && s.ver > v0 && s.view === s.ver && s.final && !s.waiting;
      if (clear && s.ver !== clearedVer) { clearedVer = s.ver; clearedAt = Date.now(); clears.push([s.ver, clearedAt - t0]); }
      if (s && !clear) clearedVer = -1;
      if (clearedVer > 0 && Date.now() - clearedAt > 15000) return { ms: clearedAt - t0, clears };
      await page.waitForTimeout(250);
    }
    return { ms: null, clears };
  };
  const first = await settle(t0, v0);
  out.updMs = first.ms;
  out.clears = first.clears;
  console.log('demandUpdating cleared after ms', out.updMs, JSON.stringify(out.clears));
  // a later edit to the Ceuta line: its trains
  const v1 = await page.evaluate(() => window.twDemand.log.at(-1)?.version ?? 0);
  const t1 = Date.now();
  await page.evaluate((line) => window.tw.client.edit({ op: 'setTrainCount', line, trains: [12, 8, 6] }), r.line);
  const second = await settle(t1, v1);
  out.trainsMs = second.ms;
  out.trainsClears = second.clears;
  console.log('after a trains edit, demandUpdating cleared after ms', out.trainsMs, JSON.stringify(out.trainsClears));
  out.afterLine = await riders(page);
  const log = await page.evaluate(() => window.__wlog);
  out.log = log.filter((x) => x.t > pt0 - 15000).map((x) => ({ ...x, t: x.t - Math.round(pt0) }));
  out.console = clog;
  out.fresh = out.afterLine;
  // the save with the line, loaded fresh
  if (!process.env.NOFRESH) {
    const b1 = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, b1, { timeout: 120000 });
    await page.goto(`${BASE}?debug=1&demandWorkers=${W}${PERF}`);
    await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
    t = Date.now();
    await page.evaluate(() => document.querySelector('.start-menu .start-save').click());
    await demandFinal(page, { held: true, timeout: 600000 });
    out.freshS = (Date.now() - t) / 1000;
    await page.waitForTimeout(3000);
    out.fresh = await riders(page);
  }
  out.errors = errors;
  await writeFile(OUT, JSON.stringify(out));
  for (const x of out.log) console.log(JSON.stringify(x));
  console.log('errors', JSON.stringify(errors.slice(0, 5)));
} finally { await close(); }
