// T-1558 scratch: T-1169's measure.mjs plus t328's figure signature of the weekday final.
//   node t1558-measure.mjs <base url> <save> <out.json> [timeout min]
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, NOISE } from './lib.mjs';

const [BASE, SAVE, OUT, MIN = '45'] = process.argv.slice(2);
const src = await readFile(new URL('./t328-cache-check.mjs', import.meta.url), 'utf8');
const a = src.indexOf('// a signature of every figure a view holds');
const b = src.indexOf('window.__ev = [];');
const SIG = src.slice(a, b);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1558-' });
const errors = [];
let result = {};
try {
  const page = await ctx.newPage();
  await page.addInitScript(`(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    ${SIG}
    const D = (window.__diag = { light: [], full: {}, done: [] });
    const ch = new BroadcastChannel('t1169');
    ch.onmessage = (e) => {
      const m = e.data;
      const r = JSON.parse(m.report);
      D.light.push({ w: m.w, aux: m.aux, kind: m.kind, t: m.t, live: r.live, peak: r.peak, mem: r.mem, blocks: r.blocks, ncities: m.cities.length, nbare: m.bare.length });
      const full = { w: m.w, aux: m.aux, mine: m.mine, kind: m.kind, t: m.t, cities: m.cities, bare: m.bare, version: m.version, ...r };
      D.full[m.w + ':' + m.kind] = full;
      if (m.kind === 'done') D.done.push(full);
      if (D.done.length > 12) D.done.shift();
    };
  })()`);
  collectErrors(page, errors, NOISE, /glyphs/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`${secs()} s loaded`);
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  await page.waitForTimeout(3000);
  result = await page.evaluate(async () => {
    const d = window.twDemand, v = d.view(), c = d.client;
    const sig = window.__sig();
    const modes = await window.__modesSig();
    return { rail: v?.railTripsPerDay ?? 0, sig, modes, nums: Array.from(window.__lastNums), status: String(c.status), computeN: c.computeN, memory: c.memory(), opened: [...c.opened], stations: window.tw.world.value.save.stations.length, lines: window.tw.world.value.save.lines.length, diag: window.__diag };
  });
  result.finalSecs = Number(secs());
} catch (err) {
  console.log('ERROR', String(err));
  result.error = String(err);
} finally {
  result.errors = errors;
  await writeFile(OUT, JSON.stringify(result));
  console.log(`wrote ${OUT}; ${errors.length} errors`);
  if (result.sig) console.log('SIG', JSON.stringify(result.sig.sections), result.sig.hash, 'modes', JSON.stringify(result.modes), 'rail', result.rail);
  await close();
}
