// T-1169 scratch: load a save on the instrumented build, collect each demand worker's tagged heap
// reports (BroadcastChannel "t1169") until the weekday final, and write them as JSON.
//   node t1169-measure.mjs <base url> <save> <out.json> [timeout min]
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, NOISE } from './lib.mjs';

const [BASE, SAVE, OUT, MIN = '45'] = process.argv.slice(2);
const QS = process.env.QS ?? '';
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1169-' });
const errors = [];
let result = {};
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    const D = (window.__diag = { light: [], full: {}, done: [] });
    const ch = new BroadcastChannel('t1169');
    ch.onmessage = (e) => {
      const m = e.data;
      const r = JSON.parse(m.report);
      D.light.push({ w: m.w, aux: m.aux, kind: m.kind, t: m.t, live: r.live, peak: r.peak, mem: r.mem, blocks: r.blocks, ncities: m.cities.length, nbare: m.bare.length });
      const full = { w: m.w, aux: m.aux, mine: m.mine, kind: m.kind, t: m.t, cities: m.cities, bare: m.bare, version: m.version, ...r };
      D.full[`${m.w}:${m.kind}`] = full;
      if (m.kind === 'done') D.done.push(full);
      if (D.done.length > 12) D.done.shift();
    };
  });
  collectErrors(page, errors, NOISE, /glyphs/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${QS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`${secs()} s loaded`);
  if (process.env.WINDOW) {
    await page.waitForFunction(() => Object.values(window.__diag.full).some((r) => r.kind === 'counts' && r.counts.some((c) => c[0] === 'mkt.window_done')), undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
    result = await page.evaluate(() => Object.values(window.__diag.full).filter((r) => r.kind === 'counts').map((r) => ({ w: r.w, mine: r.mine, counts: r.counts, tags: r.tags, live: r.live, peak: r.peak })));
    result = { windows: result, finalSecs: Number(secs()) };
    throw new Error('stopped at the first window count (WINDOW)');
  }
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  // let the last reports arrive
  await page.waitForTimeout(3000);
  result = await page.evaluate(() => {
    const d = window.twDemand, v = d.view(), c = d.client;
    return { rail: Math.round(v?.railTripsPerDay ?? 0), status: String(c.status), computeN: c.computeN, memory: c.memory(), opened: [...c.opened], stations: window.tw.world.value.save.stations.length, lines: window.tw.world.value.save.lines.length, diag: window.__diag };
  });
  result.finalSecs = Number(secs());
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
} finally {
  result.errors = errors;
  await writeFile(OUT, JSON.stringify(result));
  console.log(`wrote ${OUT}; ${errors.length} errors`);
  await close();
}
