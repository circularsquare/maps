// T-1850: the Tokyo real-network save, then a first line in Ceuta: every message between the page
// and the demand workers that says why a cluster is solved again (solve flags, opens and closes with
// levels, periods, plan, done). Based on T-1847's ceuta.mjs.
// env: BASE, SAVE, WORKERS, OUT
import { readFile, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { launchChrome, importSave, demandFinal, buildLine, collectErrors, ANGLE } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6150/';
const SAVE = process.env.SAVE ?? 'C:/Users/anita/projects/maps/trainworld/data/packs/real/tyo.save';
const W = process.env.WORKERS ?? '2';
const OUT = process.env.OUT ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/t1850/tokyo-ceuta.json';
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
      case 'solve': return { v: m.version, barrier: !!m.barrier, first: m.first, only: m.only, freshParts: !!m.freshParts, fresh: !!m.fresh, st: m.net ? m.net.stXY.length / 2 : null, lines: m.net ? m.net.lineN.length : null };
      case 'open': return { packs: (m.packs ?? []).map((p) => p.id + (p.level ? '@' + p.level : '')), markets: (m.intercity ?? []).map((x) => x.id), bare: !!m.bare, first: m.first };
      case 'close': return { ids: m.ids, markets: m.markets, bare: !!m.bare };
      case 'periods': return { periods: m.periods, mkt: m.mkt, share: m.share };
      case 'plan': return { v: m.version, solving: (m.solving ?? []).map((c) => c.length + ':' + c.slice(0, 3).join(',')), kept: (m.kept ?? []).length, ahead: m.ahead ? m.ahead.slice(0, 3) : undefined, local: m.local };
      case 'done': return { v: m.version, clusters: m.clusters, periods: m.periods };
      case 'opened': return { cities: Object.keys(m.cityInfo ?? {}), markets: m.markets, rejected: m.rejected, openMs: Math.round(m.openMs ?? 0) };
      case 'failed': case 'died': return { error: String(m.error ?? '') };
      default: return null;
    }
  };
  const keep = new Set(['solve', 'open', 'close', 'periods', 'init', 'plan', 'done', 'opened', 'failed', 'whole', 'freshParts']);
  const W0 = window.Worker;
  window.Worker = class extends W0 {
    constructor(url, opts) {
      super(url, opts);
      this.__id = `${n++}:${String(url).split('/').pop().split('?')[0]}`;
      super.addEventListener('message', (e) => { const m = e.data; if (m && keep.has(m.kind)) log.push({ t: Math.round(performance.now()), w: this.__id, dir: '<', kind: m.kind, ...sum(m) }); });
    }
    postMessage(m, t) {
      try {
        if (m && m.kind === 'solve' && m.net && this.__id.startsWith('2:')) {
          const plain = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v]));
          (window.__nets ??= {})[m.version] = { net: plain(m.net), ids: m.ids };
        } if (m && keep.has(m.kind)) log.push({ t: Math.round(performance.now()), w: this.__id, dir: '>', kind: m.kind, ...(m.kind === 'init' ? { packs: (m.packs ?? []).length, periods: m.periods, share: m.share, mkt: m.mkt } : sum(m)) }); } catch {}
      return super.postMessage(m, t);
    }
  };
};

const errors = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t1850-', args: ANGLE });
try {
  const page = await ctx.newPage();
  await page.addInitScript(hook);
  collectErrors(page, errors);
  const clog = [];
  page.on('console', (m) => clog.push(m.type() + ' ' + m.text().slice(0, 300)));
  page.on('pageerror', (e) => clog.push('pageerror ' + String(e).slice(0, 300)));
  await page.goto(`${BASE}?debug=1&continue=1&paused=1&demandWorkers=${W}`);
  if (process.env.PROBE) { await page.waitForTimeout(30000); console.log(JSON.stringify(errors), clog.join('\n')); process.exit(0); }
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });
  const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 400000 });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 120000 });
  await page.goto(`${BASE}?debug=1&demandWorkers=${W}`);
  await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
  const t = Date.now();
  await page.evaluate(() => document.querySelector('.start-menu .start-save').click());
  await demandFinal(page, { held: true, timeout: 600000 });
  console.log('save final after s', (Date.now() - t) / 1000);
  await page.waitForTimeout(3000);
  const tJump = await page.evaluate(([lon, lat]) => { window.tw.map.jumpTo({ center: [lon, lat], zoom: 12 }); return performance.now(); }, [lon, lat]);
  await page.waitForTimeout(6000);
  await page.evaluate(async () => { window.__d = await import('/src/game/demand.ts'); });
  const b0 = await page.evaluate(() => performance.now());
  const r = await buildLine(page, pts, { trains: [10, 8, 6], wait: { route: 300, station: 150, settle: 300 } });
  console.log('buildLine ok', r?.ok, JSON.stringify(r?.issues ?? []));
  const t0 = await page.evaluate(() => performance.now());
  let upd = null;
  if (process.env.QUICK) {
    await page.waitForTimeout(5000);
    const nets = await page.evaluate(() => window.__nets);
    await writeFile(OUT.replace('.json', '-nets.json'), JSON.stringify(nets));
    console.log('nets of versions', Object.keys(nets).join(','));
  }
  for (const w0 = Date.now(); !process.env.QUICK && Date.now() - w0 < 300000;) {
    const s = await page.evaluate(() => ({ upd: window.__d.demandUpdating.value, vfin: window.__d.viewFinal.value })).catch(() => null);
    if (s && !s.upd && Date.now() - w0 > 3000) { upd = Date.now() - w0; break; }
    await page.waitForTimeout(1000);
  }
  console.log('demandUpdating cleared after ms', upd);
  await page.waitForTimeout(3000);
  const log = await page.evaluate(() => window.__wlog);
  const rel = (x) => ({ ...x, t: x.t - Math.round(t0) });
  const out = { tJump: Math.round(tJump - t0), build: Math.round(b0 - t0), upd, log: log.map(rel), errors };
  await writeFile(OUT, JSON.stringify(out, null, 1));
  for (const x of out.log.filter((x) => x.t > out.tJump - 1000)) console.log(JSON.stringify(x));
  console.log('errors', JSON.stringify(errors.slice(0, 5)));
} finally { await close(); }
