// T-2867: a demand worker that cannot get more memory costs the pool workers, never the game. A simulated
// failed grow (`twDemand.client.debugPanic(i, when, true)`, the error text a real one gives) on the real
// save: during the load, in an idle solving worker and in the aux worker, the pool gets smaller and the
// game reaches final figures equal to the trip to a load with no failure (the pool's size changes no
// figure). With one solving worker left, a failure twice over reaches the out of memory screen, whose
// button plays on with intercity trips off: the page comes back to the same game with no market, to
// final figures. Also: `addressLimited()` is false in this 64-bit Chrome.
// Usage (maps root): node trainworld/checks/t2867-memory-fallback.mjs [dev server url]. About 6 minutes.
// Dev server only: `debugPanic` is compiled out of a production build.
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, NOISE, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6135/';
let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const WORKERS = 3;
const url = (extra = '') => `${BASE}?debug=1&new=1&paused=1&demandWorkers=${WORKERS}${extra}&v=${Date.now()}`;

const { ctx, close } = await launchChrome({ prefix: 'tw-t2867-' });
const errors = [];
const state = (page) => page.evaluate(() => {
  const d = window.twDemand, v = d.view();
  return { version: v?.version, final: !!v?.final, rail: Math.round(v?.railTripsPerDay ?? 0), status: String(d.client.status), n: d.client.computeN, workers: d.client.workers.length };
});
const open = async (page, extra) => {
  await page.goto(url(extra));
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
};
const load = async (page) => {
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
};
const settled = (page, after) => page.waitForFunction((after) => {
  const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version)), v = d.view();
  return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting && v.version > after && !d.client.respawn.size);
}, after, { timeout: 420000, polling: 250 });

try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs/);

  await open(page);
  report('64-bit Chrome is not taken for a 32-bit browser', await page.evaluate(() => window.twDemand.addressLimited() === false));
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100 });
  const clean = await state(page);
  console.log(`      no failure: ${clean.rail.toLocaleString('en-US')} rail trips a day, ${clean.n} solving workers`);
  report('the pool starts at the size asked', clean.n === WORKERS, `${clean.n}`);

  // during the load, in a solving worker's first solve
  await open(page);
  await page.evaluate(() => window.twDemand.client.debugPanic(1, 'solve', true));
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100, timeout: 420000 });
  await settled(page, -1);
  let now = await state(page);
  report('a failed grow during the load: fewer workers, same final figures', now.n < WORKERS && now.rail === clean.rail && !now.status.startsWith('failed'), `${now.n} solving workers, ${now.rail.toLocaleString('en-US')} rail trips, ${now.status}`);

  // idle, aux: first bring a fresh pool of WORKERS up
  await open(page);
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100 });
  let last = await state(page);
  const before = last.n;
  await page.evaluate((i) => window.twDemand.client.debugPanic(i, 'now', true), last.n);
  await settled(page, last.version);
  now = await state(page);
  report('a failed grow in the aux worker: fewer workers, same final figures', now.n < before && now.rail === clean.rail && !now.status.startsWith('failed'), `${before} to ${now.n} solving workers, ${now.rail.toLocaleString('en-US')} rail trips`);
  const riders = await page.evaluate(async () => { const d = window.twDemand, st = window.tw.world.value.save.stations[37]?.id; return !!(st && await d.stationRiders(st)); });
  report('a station is still answered', riders);

  // the player's own words on the failure screen or notice: the way on keeps demand
  await page.goto(url());
  await page.waitForFunction(() => window.tw?.failDemand, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.failDemand('worker 4: RuntimeError: unreachable executed (out of memory: 57318568 bytes could not be allocated), 184 MB of WASM memory'));
  await page.waitForSelector('.loading-box.failed, .demand-stopped-note', { timeout: 30000 });
  const shown = page.locator('.loading-box.failed, .demand-stopped-note');
  const words = (await shown.innerText()).replace(/\n+/g, ' | ');
  report('the failure offers intercity trips off, and no "Play without demand"', /Play without intercity trips/.test(words) && !/Play without demand/.test(words), words.slice(0, 260));

  // a game with a slot, one solving worker: replaced once, as any panic; a second failure before it has solved stops the pool
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100, timeout: 420000 });
  await page.evaluate(() => window.tw.client.saveBeforeReload());
  last = await state(page);
  report('the pool has one solving worker', last.n === 1 && !last.status.startsWith('failed'), `${last.n}, ${last.rail.toLocaleString('en-US')} rail trips`);
  await page.evaluate(() => window.twDemand.client.debugPanic(0, 'now', true));
  await page.waitForFunction(() => window.twDemand.client.respawn.size > 0, undefined, { timeout: 30000 });
  await page.evaluate(() => window.twDemand.client.debugPanic(0, 'now', true));
  await page.waitForFunction(() => String(window.twDemand.client.status).startsWith('failed'), undefined, { timeout: 60000 });
  await page.waitForTimeout(500);
  const note = page.locator('.demand-stopped-note');
  report('out of memory with one worker: the notice comes up', await note.count() === 1, (await note.innerText().catch(() => '')).replace(/\n/g, ' | '));
  const button = note.getByRole('button', { name: 'Play without intercity trips' });
  report('the notice offers to play without intercity trips', await button.count() === 1);
  await Promise.all([page.waitForNavigation({ timeout: 60000 }), button.click()]);
  await page.waitForFunction(() => window.tw?.world?.value?.save?.lines?.length > 0 && window.twDemand?.client, undefined, { timeout: 180000 });
  await demandFinal(page, { polling: 250, lines: 100, timeout: 420000 });
  const off = await page.evaluate(() => ({ off: window.twDemand.marketOff(), markets: [...window.twDemand.client.opened].filter((x) => x.startsWith('market:')).length, rail: Math.round(window.twDemand.view().railTripsPerDay) }));
  report('the game comes back with intercity trips off and reaches final figures', off.off && off.markets === 0 && off.rail > 0, `${off.rail.toLocaleString('en-US')} rail trips a day (${(100 * (1 - off.rail / clean.rail)).toFixed(1)}% fewer), ${off.markets} market packs open`);

  const unexpected = errors.filter((e) => !/panicked, replaced|out of memory|load failed|demand: worker/.test(e));
  report('no other console errors', unexpected.length === 0, unexpected.slice(0, 5).join(' | '));
} finally {
  await close();
}
console.log(failed ? `T2867 FAILED: ${failed}` : 'T2867 PASSED');
process.exit(failed ? 1 : 0);
