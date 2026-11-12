// T-3314, T-3315: the "figures stopped" notice and the no-SIMD screen.
//  1. normal play (a line built, the clock running, a minute) never shows the notice even with the thresholds
//     cut to 30 s (`?figuresMs=30000`), and the longest quiet seen stays under them
//  2. an edit the clock worker never answers (a fake entry in the client's unanswered list) shows the notice
//     after the threshold; answering it hides the notice; Reload reloads the page
//  3. every demand download held unanswered, a line built: the demand pool waits with no step, the notice shows
//  4. a browser whose `WebAssembly.validate` refuses the SIMD probe gets the loading screen's own sentence, no buttons
//   node trainworld/checks/t3314-figures-stopped.mjs [base url]      (screenshots with SHOTS=1)
import { collectErrors, launchChrome, demandFinal, acrossLine, ARTIFACTS } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6175/').replace(/\?.*$/, '');
const Q = (ms) => `${BASE}?new=1&debug=1&acct=off&notrack&figuresMs=${ms}`;
const SHOTS = !!process.env.SHOTS;
const PACK = /\/packs\/(?!terrain-world\/)[^?]*(\.bin|\.market\.json)(\?|$)/;
const NOTE = '.demand-stopped-note';
let fails = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) fails++; };
const peaks = (page) => page.evaluate(() => ({ clock: window.tw.figures.peaks.clock / 1000, demand: window.tw.figures.peaks.demand / 1000, on: window.tw.figures.stopped.value }));
const up = async (page, url) => {
  await page.goto(url);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value, null, { timeout: 120000 });
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t3314-' });
try {
  const errors = [];
  let hold = false;
  await ctx.route((u) => PACK.test(u.href) && !/\/real\//.test(u.href), (route) => {
    if (hold && !route.request().headers().range) return; // never answered
    return route.continue();
  });
  const page = ctx.pages()[0] ?? await ctx.newPage();
  collectErrors(page, errors);

  // 1. normal play
  await up(page, Q(30000));
  await demandFinal(page, { strict: false, timeout: 180000 });
  check((await acrossLine(page, [-73.98, 40.75])) !== null, 'a line built');
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await page.waitForTimeout(60000);
  let p = await peaks(page);
  console.log(`      peaks clock ${p.clock.toFixed(1)} s, demand ${p.demand.toFixed(1)} s`);
  check(!p.on && !(await page.$(NOTE)), 'normal play: no notice with the thresholds at 30 s');
  check(p.clock < 30 && p.demand < 30, 'normal play: the longest quiet is under 30 s');

  // 2. the clock worker's unanswered edit
  await page.evaluate(() => window.tw.client.sent.set(999999, { what: 'check', at: performance.now() }));
  await page.waitForSelector(NOTE, { timeout: 45000 }).then(() => check(true, 'an edit unanswered for 30 s shows the notice'), () => check(false, 'the notice never showed for an unanswered edit'));
  const text = await page.$eval(NOTE, (e) => e.textContent).catch(() => '');
  console.log(`      notice: ${text}`);
  check(!/autosave/.test(text), 'nothing saved yet (?new=1): the notice names no autosave');
  await page.evaluate(() => { window.tw.lastSaved.value = Date.now() - 24000; });
  await page.waitForFunction((sel) => /autosave from 2\ds ago is kept/.test(document.querySelector(sel)?.textContent ?? ''), NOTE, { timeout: 5000 })
    .then(() => check(true, 'saved 24 s ago: "Your last autosave from 24s ago is kept."'), () => check(false, 'the notice did not give the autosave age'));
  await page.waitForTimeout(3000);
  check(/autosave from (2[6-9]|3\d)s ago/.test(await page.$eval(NOTE, (e) => e.textContent)), 'the age counts up while the notice shows');
  if (SHOTS) {
    await page.screenshot({ path: `${ARTIFACTS}t3314_notice_light.png` });
    const shade = await page.evaluate(() => { const s = document.documentElement.dataset.shade ?? null; document.documentElement.dataset.shade = 'dark'; return s; });
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${ARTIFACTS}t3314_notice_dark.png` });
    await page.evaluate((s) => { if (s === null) delete document.documentElement.dataset.shade; else document.documentElement.dataset.shade = s; }, shade);
    if (process.env.SHOT_ONLY) { await close(); process.exit(0); }
  }
  await page.evaluate(() => window.tw.client.sent.delete(999999));
  await page.waitForFunction((sel) => !document.querySelector(sel), NOTE, { timeout: 15000 }).then(() => check(true, 'answered: the notice goes'), () => check(false, 'the notice stayed after the edit was answered'));
  await page.evaluate(() => window.tw.client.sent.set(999999, { what: 'check', at: performance.now() }));
  await page.waitForSelector(NOTE, { timeout: 45000 });
  await Promise.all([page.waitForNavigation({ timeout: 30000 }).then(() => check(true, 'Reload reloads the page'), () => check(false, 'Reload did not reload')), page.click(`${NOTE} button`)]);

  // 3. every demand download held
  await up(page, Q(30000));
  await demandFinal(page, { strict: false, timeout: 180000 });
  hold = true;
  const line = await acrossLine(page, [-73.98, 40.75]);
  check(line !== null, `a line built while downloads are held (line ${line})`);
  const t0 = Date.now();
  await page.waitForSelector(NOTE, { timeout: 150000 }).then(() => check(true, `held demand downloads show the notice after ${Math.round((Date.now() - t0) / 1000)} s`), () => check(false, 'the notice never showed with demand downloads held'));
  p = await peaks(page);
  if (!(await page.$(NOTE))) console.log('      state: ' + await page.evaluate(() => `${window.twDemand.client.diagnosis()} | clock ${window.tw.client.diagnosis()} | stopped ${window.tw.figures.stopped.value}`));
  hold = false;
  console.log(`      peaks clock ${p.clock.toFixed(1)} s, demand ${p.demand.toFixed(1)} s`);

  // 4. no SIMD (Safari before 16.4)
  const old = await ctx.newPage();
  await old.addInitScript(() => { const v = WebAssembly.validate; WebAssembly.validate = (b) => (b && b.length === 31 ? false : v.call(WebAssembly, b)); });
  collectErrors(old, []);
  await old.goto(`${BASE}?new=1&debug=1&acct=off&notrack`);
  await old.waitForSelector('.loading-screen.loading-screen, .loading-screen', { timeout: 30000 });
  const box = await old.$eval('.loading-box', (e) => ({ text: e.textContent, buttons: e.querySelectorAll('button').length }));
  console.log(`      screen: ${box.text}`);
  check(/missing a feature the game needs/.test(box.text) && box.buttons === 0, 'no SIMD: the loading screen says so, with no buttons');
  if (SHOTS) await old.screenshot({ path: `${ARTIFACTS}t3315_no_simd.png` });

  const real = errors.filter((e) => !/net::ERR|Failed to fetch|AbortError|TimeoutError|signal timed out|No data for|not loaded|WebGL2|SIMD/.test(e));
  check(!real.length, `no page errors${real.length ? ': ' + real.slice(0, 3).join(' | ') : ''}`);
} finally {
  await close();
}
console.log(fails ? `${fails} failed` : 'all passed');
process.exit(fails ? 1 : 0);
