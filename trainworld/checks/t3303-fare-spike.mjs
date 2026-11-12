// T-3303: a huge fare set while riders are counted must never pay out. New game, two built lines in Manhattan,
// demand solved; then the fare curve is set 1e9 times the start's (a player's typed "90 billion"). Income a game
// hour and the cash are sampled every 40 ms until demand's next final. PASS when the curve is held to FARE_MAX
// (game/fareCurve.ts), income never passes 1.05 times what it was, the cash gained over the window is no more than
// the same window's at the old income, and riders end under 1% of before.
//   node trainworld/checks/t3303-fare-spike.mjs [url]
// from the maps root, on your own Vite (port 6161 by default); about 2 minutes.
import { acrossLine, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6161/';
const MUL = 1e9;
const b = await launchChrome({ prefix: 'tw-t3303-' });
const errors = [];
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  await acrossLine(page, [-73.985, 40.758], 2);
  await acrossLine(page, [-73.985, 40.77], 2);
  await page.waitForTimeout(3000);
  await demandFinal(page, { timeout: 900000, polling: 1000, lines: 0 });
  await page.waitForTimeout(5000);
  await page.evaluate(async () => {
    const H = [4, 6, 4, 4, 6], tw = window.tw;
    const { liveMoney } = await import('/src/game/money.ts');
    const orig = tw.moneyState.value.fares.nodes.map((n) => [...n]);
    const T = (window.__t = {
      orig,
      income: () => Math.max(...tw.incomeByDay.value.map((i) => i.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 / 24)),
      cash: () => liveMoney.value.cash * 1e6,
      riders: () => { let r = 0; for (const dm of window.twDemand.view().lines.values()) r += dm.ridersPerDay; return r; },
      view: () => { const v = window.twDemand.view(); return { version: v.version, final: v.final }; },
      samples: [],
    });
    setInterval(() => T.samples.push({ at: performance.now(), ...T.view(), income: T.income(), cash: T.cash() }), 40);
    tw.clock.setRunning(true);
    tw.clock.setMultiple(100);
  });
  const v0 = await page.evaluate(() => ({ ...window.__t.view(), income: window.__t.income(), riders: window.__t.riders() }));
  await page.waitForTimeout(3000);
  const mark = await page.evaluate(() => ({ at: performance.now(), cash: window.__t.cash() }));
  await page.evaluate((mul) => window.tw.client.edit({ op: 'fares', nodes: window.__t.orig.map(([km, f]) => [km, f * mul]) }), MUL);
  await page.waitForFunction(() => window.tw.moneyState.value.fares.nodes[0][1] !== window.__t.orig[0][1], undefined, { timeout: 30000, polling: 50 });
  await page.waitForFunction((v) => { const x = window.__t.view(); return x.final && x.version > v; }, v0.version, { timeout: 600000, polling: 250 });
  await page.waitForTimeout(8000);
  const out = await page.evaluate(async () => {
    const { FARE_MAX, tidyNodes } = await import('/src/game/fareCurve.ts');
    const s = window.__t.samples;
    return { s, end: { riders: window.__t.riders(), cash: window.__t.cash() }, top: Math.max(...window.tw.moneyState.value.fares.nodes.map((n) => n[1])), FARE_MAX, tidy: tidyNodes([[0, 9e10], [20, 9e10], [100, 9e10]]).map((n) => n[1]) };
  });
  const peak = Math.max(...out.s.map((x) => x.income));
  const secs = (out.s.at(-1).at - mark.at) / 1000;
  // what the same wall time earns at 100x on the old income (a game hour is 36 s at that speed)
  const bound = (v0.income * secs * 100) / 60 / 60 * 1.1 + 5e6;
  console.log(`before: income ${v0.income.toExponential(3)}/game hour, riders ${Math.round(v0.riders)}; window ${secs.toFixed(1)} s; peak income ${peak.toExponential(3)}; cash gained ${(out.end.cash - mark.cash).toExponential(3)}; riders after ${out.end.riders.toExponential(2)}; curve top ${out.top}`);
  check('the curve is held to FARE_MAX', out.top <= out.FARE_MAX && out.tidy.every((v) => v === out.FARE_MAX), `top ${out.top}, tidy ${out.tidy}`);
  check('income never passes 1.05 times what it was', peak <= v0.income * 1.05, `${(peak / v0.income).toFixed(3)}x`);
  check('the cash gained is no more than the old income earns', out.end.cash - mark.cash <= bound, `${(out.end.cash - mark.cash).toExponential(2)} against ${bound.toExponential(2)}`);
  check('riders end under 1% of before', out.end.riders < v0.riders * 0.01, `${out.end.riders.toExponential(2)} of ${Math.round(v0.riders)}`);
} catch (e) {
  console.log('FAILED', e.message);
  failed++;
} finally {
  check('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  await b.close();
  process.exitCode = failed ? 1 : 0;
}
