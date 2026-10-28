// T-1594 scratch: T-1169's measure-sig.mjs plus the solve log's timings (per period and round,
// per worker setup) and the packs' load log, to see where a cold load's time goes.
//   node t1594-measure.mjs <base url> <save> <out.json> [timeout min]
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, NOISE } from './lib.mjs';

const [BASE, SAVE, OUT, MIN = '45'] = process.argv.slice(2);
const src = await readFile(new URL('./t328-cache-check.mjs', import.meta.url), 'utf8');
const a = src.indexOf('// a signature of every figure a view holds');
const b = src.indexOf('window.__ev = [];');
const SIG = src.slice(a, b);
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0);
const { ctx, close } = await launchChrome({ prefix: 'tw-t782-' });
const errors = [];
let result = {};
try {
  const page = await ctx.newPage();
  await page.addInitScript(`(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {}
    ${SIG}
  })()`);
  collectErrors(page, errors, NOISE, /glyphs/);
  page.on('console', (m) => { if (/panicked|out of memory|solving workers to|T1594|T782/.test(m.text())) console.log(`${secs()} s: ${m.text().slice(0, 300)}`); });
  await page.goto(`${BASE}?debug=1&new=1&paused=1${process.env.QS ?? ''}&v=${Date.now()}`, { timeout: 300000, waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const tImport = await page.evaluate(() => performance.now());
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  console.log(`${secs()} s loaded`);
  await page.waitForFunction(() => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    // (a solve of the zone in view alone, before its neighbours and the market open, is not the load's final)
    return String(d.client.status).startsWith('failed') || (!!v && s && s.markets?.length > 0 && v.version === s.version && v.final && !d.client.waiting);
  }, undefined, { timeout: Number(MIN) * 60000, polling: 2000 });
  console.log(`${secs()} s final`);
  const tFinal = await page.evaluate(() => performance.now());
  await page.waitForTimeout(3000);
  result = await page.evaluate(async () => {
    const d = window.twDemand, v = d.view(), c = d.client;
    const sig = window.__sig();
    const modes = await window.__modesSig();
    const log = d.log.map((s) => ({ version: s.version, sentAt: s.sentAt, arrivedAt: s.arrivedAt, roundAt: s.roundAt, periodMs: s.periodMs, setupMs: s.setupMs, clusters: s.clusters, periods: s.periods, published: s.published, rough: s.rough, cities: s.cities?.length, markets: s.markets?.length }));
    const lines = Object.fromEntries([...v.lines].map(([k, l]) => [k, [l.ridersPerDay, l.boardings, l.fullestPct]]));
    const stations = Object.fromEntries([...v.stations].map(([k, s]) => [k, [s.boardings, s.alightings]]));
    const ic = d.intercity();
    const market = ic ? [ic.tripsPerDay, ic.railTripsPerDay, ic.commuteTripsPerDay ?? 0, ic.commuteRailPerDay ?? 0] : null;
    const cities = Object.fromEntries([...d.cities()].map(([k, c]) => [k, c.railTripsPerDay]));
    return { rail: v?.railTripsPerDay ?? 0, lineFig: lines, stationFig: stations, market, cityRail: cities, sig, modes, status: String(c.status), computeN: c.computeN, memory: c.memory(), stations: window.tw.world.value.save.stations.length, lines: window.tw.world.value.save.lines.length, log, start: c.start };
  });
  result.tImport = tImport;
  result.tFinal = tFinal;
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
