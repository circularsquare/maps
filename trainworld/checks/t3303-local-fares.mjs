// T-3303: after a fare edit, intermediate results before the whole solve's final (a local result when one runs, or
// whole-solve period steps in a one-city game where T-3512 skips the local stage) hold lines priced at the old fares
// (`takesMarketless` in publishCluster / publishStep), or unsolved periods / longer trips pay the new fare they never
// chose. New game, three lines in Manhattan solved, then the curve set MUL times (300 by default, under FARE_MAX).
// Every 50 ms until the fare's final: the shown view, each line's solved fares, the income and the cash.
// PASS when the fare solve showed a pre-final step or local result, no line is priced at the new fares before that
// solve's final, income never passes 1.05 times what it was, the cash gained is no more than twice the old income over
// the game hours passed, and after the final every line is priced at the new fares with riders under 1% of before.
// `perfOff=fareHold` (the second argument) turns the fix off: it should FAIL.
//   node trainworld/checks/t3303-local-fares.mjs [url] [extra query]
// from the maps root, on your own Vite (port 6168 by default); about 2 minutes.
import { acrossLine, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6168/';
const EXTRA = process.argv[3] ? `&${process.argv[3]}` : '';
const MUL = Number(process.env.MUL ?? 300);
const b = await launchChrome({ prefix: 'tw-t3303l-' });
const errors = [];
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2${EXTRA}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  await acrossLine(page, [-73.985, 40.758], 2);
  await acrossLine(page, [-73.985, 40.77], 2);
  await acrossLine(page, [-73.97, 40.785], 2);
  await demandFinal(page, { timeout: 900000, polling: 500, lines: 0 });
  const v0 = await page.evaluate((mul) => {
    const H = [4, 6, 4, 4, 6], tw = window.tw;
    const orig = tw.moneyState.value.fares.nodes.map((n) => [...n]);
    const T = (window.__t = {
      orig, base: orig[0][1] * mul, samples: [],
      income: () => Math.max(...tw.incomeByDay.value.map((i) => i.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 / 24)),
      cash: () => tw.moneyState.value.cash * 1e6,
      riders: () => { let r = 0; for (const dm of window.twDemand.view().lines.values()) r += dm.ridersPerDay; return r; },
      // the shown view and how many of its lines (on any kind of day) are priced at the new fares
      view: () => {
        const v = window.twDemand.view();
        let priced = 0;
        for (const d of window.twDemand.days()) for (const l of d.view?.lines.values() ?? []) if (l.solvedFares?.nodes?.[0]?.[1] === T.base) priced++;
        return { version: v.version, final: v.final, priced };
      },
    });
    const at = { ...T.view(), income: T.income(), riders: T.riders(), cash: T.cash(), g: tw.clock.now() };
    setInterval(() => T.samples.push({ ...T.view(), income: T.income() }), 50);
    tw.client.edit({ op: 'fares', nodes: orig.map(([km, f]) => [km, f * mul]) });
    tw.clock.setRunning(true);
    tw.clock.setMultiple(100);
    return at;
  }, MUL);
  // the fare's own solve: the first one sent at the new fares
  await page.waitForFunction(() => window.twDemand.log.some((s) => s.svc?.fare?.nodes?.[0]?.[1] === window.__t.base), undefined, { timeout: 30000, polling: 20 });
  const fv = await page.evaluate(() => window.twDemand.log.find((s) => s.svc?.fare?.nodes?.[0]?.[1] === window.__t.base).version);
  await page.waitForFunction((fv) => { const v = window.twDemand.view(); return v.final && v.version >= fv; }, fv, { timeout: 600000, polling: 50 });
  await page.waitForTimeout(3000);
  const out = await page.evaluate((fv) => {
    const s = window.twDemand.log.find((x) => x.version === fv);
    const steps = Array.from(s?.clusterRes?.values() ?? []).reduce((n, e) => n + (e.steps?.shown.length ?? 0), 0);
    const days = window.twDemand.days();
    let lines = 0;
    for (const d of days) lines += d.view?.lines.size ?? 0;
    return {
      s: window.__t.samples, riders: window.__t.riders(), income: window.__t.income(), cash: window.__t.cash(),
      g: window.tw.clock.now(), local: s?.localShown ?? 0, steps, pricedAfter: window.__t.view().priced, linesAfter: lines,
    };
  }, fv);
  const early = out.s.filter((x) => x.version < fv && x.priced > 0);
  const before = out.s.filter((x) => x.version < fv);
  const peak = Math.max(...out.s.map((x) => x.income), v0.income);
  const hours = (out.g - v0.g) / 3600, gained = out.cash - v0.cash, bound = 2 * v0.income * hours + 5e6;
  console.log(`before: income ${v0.income.toExponential(3)}/game hour, riders ${Math.round(v0.riders)}, fare solve v${fv}; ${before.length} samples before its final over ${hours.toFixed(1)} game hours; peak ${peak.toExponential(3)}; cash gained ${gained.toExponential(3)}; riders after ${out.riders.toExponential(2)}; income after ${out.income.toExponential(2)}`);
  check('the fare solve showed a pre-final step or local result', out.local + out.steps >= 1, `localShown ${out.local}, stepsShown ${out.steps}`);
  check('no line is priced at the new fares before its final', early.length === 0, early.length ? `${early.length} samples, first on v${early[0].version}, ${early[0].priced} lines` : '');
  check('income never passes 1.05 times what it was', peak <= v0.income * 1.05, `${(peak / v0.income).toFixed(3)}x`);
  check('the cash gained is no more than twice the old income', gained <= bound, `${gained.toExponential(2)} against ${bound.toExponential(2)}`);
  check('figures settle to the full solve at the new fares', out.pricedAfter === out.linesAfter && out.linesAfter > 0 && out.riders < v0.riders * 0.01 && out.income < v0.income * 0.01, `priced ${out.pricedAfter}/${out.linesAfter}, riders ${out.riders.toExponential(2)}, income ${out.income.toExponential(2)}`);
} catch (e) {
  console.log('FAILED', e.message);
  failed++;
} finally {
  check('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  await b.close();
  process.exitCode = failed ? 1 : 0;
}
