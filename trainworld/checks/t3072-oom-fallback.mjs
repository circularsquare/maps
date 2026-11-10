// T-3072: a solving worker that runs out of memory costs the pool workers, never the game, and the
// smaller pool is remembered for the tab's session. The real save with 3 solving workers; one worker's
// grow is made to fail (`debugPanic`, the error text a real failed grow gives) during the load. Checks:
// the game reaches final figures equal to a load with no failure, on fewer workers, with no failure
// screen or notice shown and no console error but the pool's own warning; the tab remembers the lower
// count; a pool sized afterwards (no `demandWorkers` in the address) is no larger than that count.
// Usage (maps root): node trainworld/checks/t3072-oom-fallback.mjs [dev server url]. About 4 minutes.
// Dev server only: `debugPanic` is compiled out of a production build.
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, NOISE, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6198/';
let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const WORKERS = 3;
const url = (extra = '') => `${BASE}?debug=1&new=1&paused=1${extra}&v=${Date.now()}`;

const { ctx, close } = await launchChrome({ prefix: 'tw-t3072-' });
const errors = [];
const state = (page) => page.evaluate(() => {
  const d = window.twDemand, v = d.view();
  return { final: !!v?.final, rail: Math.round(v?.railTripsPerDay ?? 0), status: String(d.client.status), n: d.client.computeN, cap: sessionStorage.getItem('anitabuilder-pool-cap') };
});
const open = async (page, extra) => {
  await page.goto(url(extra));
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
};
const load = async (page) => {
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
};
const settled = (page) => page.waitForFunction(() => {
  const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version)), v = d.view();
  return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting && !d.client.respawn.size);
}, undefined, { timeout: 420000, polling: 250 });

try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs/);

  await open(page, `&demandWorkers=${WORKERS}`);
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100 });
  const clean = await state(page);
  console.log(`      no failure: ${clean.rail.toLocaleString('en-US')} rail trips a day, ${clean.n} solving workers`);
  report('the pool starts at 3 and remembers nothing', clean.n === WORKERS && clean.cap === null, `${clean.n}, cap ${clean.cap}`);

  await open(page, `&demandWorkers=${WORKERS}`);
  await page.evaluate(() => window.twDemand.client.debugPanic(1, 'solve', true));
  await load(page);
  await demandFinal(page, { polling: 250, lines: 100, timeout: 420000 });
  await settled(page);
  const now = await state(page);
  report('one worker out of memory: fewer workers, same final figures', now.n < WORKERS && now.n >= 1 && now.final && now.rail === clean.rail && !now.status.startsWith('failed'), `${now.n} solving workers, ${now.rail.toLocaleString('en-US')} rail trips, ${now.status}`);
  report('no failure screen or notice', await page.locator('.loading-box.failed, .demand-stopped-note').count() === 0);
  report('the tab remembers the lower count', Number(now.cap) === now.n, `cap ${now.cap}`);

  // the page's own address gives the pool size (the check's launcher adds demandWorkers); take it off
  await open(page);
  const again = await page.evaluate(() => { history.replaceState(null, '', location.pathname + '?debug=1'); return window.twDemand.poolSize(); });
  report('a pool made after the fall is no larger than the remembered count', again <= Number(now.cap), `${again} against cap ${now.cap}`);

  const unexpected = errors.filter((e) => !/out of memory|demand: worker/.test(e));
  report('no other console errors', unexpected.length === 0, unexpected.slice(0, 5).join(' | '));
} finally {
  await close();
}
console.log(failed ? `T3072 FAILED: ${failed}` : 'T3072 PASSED');
process.exit(failed ? 1 : 0);
