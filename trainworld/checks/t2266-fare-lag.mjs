// T-2266: riders pay the fares their demand was solved with. On the real save: raise the fare curve 10x,
// then (while that is solving) set it to 5x. The fare controls (moneyState) show each at once; fare income a
// game hour stays what it was with the old fares until riders solved with the new curve show (a cluster's
// final, then the solve's final) and ends at the new-fare figure, every line priced
// at 5x. The 10x curve, overtaken by the second edit, is never used.
//   node trainworld/checks/t2266-fare-lag.mjs [url] [save] [out.json]
// from the maps root, on your own Vite (port 6172 by default), DEMAND_WORKERS demand workers (2 by default);
// about 5 minutes. Income is game US$ (the economy's 100x) per game hour, weekday, 24-hour mean.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6172/';
const SAVE = process.argv[3] ?? REAL_SAVE;
const OUT = process.argv[4] ?? `${ARTIFACTS}t2266/fare-lag.json`;
const WORKERS = process.env.DEMAND_WORKERS ?? 2;

const b = await launchChrome({ prefix: 'tw-t2266-' });
const errors = [];
const out = { when: new Date().toISOString(), base: BASE, save: SAVE };
let failed = 0;
const check = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { timeout: 1800000, polling: 1000, lines: 0 });
  await page.waitForTimeout(5000);

  // helpers in the page: income a game hour (24-hour mean, weekday), the fare controls' factor on the start curve,
  // and how many lines are priced at each factor
  await page.evaluate(async () => {
    const H = [4, 6, 4, 4, 6];
    const { lineFares } = await import('/src/game/fareInEffect.ts');
    // T-2286: each line's stamp of its solve's fares, as the demand client's `daysOf` writes it
    const markSolveFares = (days, fares) => { for (const d of days) if (d.view) for (const l of d.view.lines.values()) l.solvedFares = fares; };
    const { demandByDay } = await import('/src/game/demand.ts');
    const tw = window.tw;
    const orig = tw.moneyState.value.fares.nodes.map((n) => [...n]);
    const k0 = orig.findIndex((n) => n[1] > 0);
    const T = (window.__t = {
      orig,
      income: () => tw.incomeByDay.value[0].perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 / 24,
      factor: (nodes) => Math.round(nodes[k0][1] / orig[k0][1] * 100) / 100,
      controls: () => T.factor(tw.moneyState.value.fares.nodes),
      mix: () => {
        const cur = tw.moneyState.value.fares, m = {};
        for (const dm of window.twDemand.view().lines.values()) { const f = T.factor(lineFares(dm, cur).nodes); m[f] = (m[f] ?? 0) + 1; }
        return m;
      },
      view: () => { const v = window.twDemand.view(); return { version: v.version, final: v.final }; },
      now: () => ({ ...T.view(), income: T.income(), controls: T.controls(), mix: T.mix() }),
      edit: (mul) => tw.client.edit({ op: 'fares', nodes: orig.map(([km, f]) => [km, f * mul]) }),
      // what the old behaviour showed right after an edit: the shown riders priced at the current fares
      // (their marks moved to them for an instant, then put back; the views are copies so income is worked out again)
      spike: () => {
        const days = demandByDay.peek(), old = days[0].view.lines.values().next().value.solvedFares;
        const copy = () => demandByDay.value = demandByDay.peek().map((d) => ({ ...d, view: d.view && { ...d.view } }));
        markSolveFares(days, tw.moneyState.value.fares);
        copy();
        const v = T.income();
        markSolveFares(demandByDay.peek(), old);
        copy();
        return v;
      },
      samples: [],
    });
    setInterval(() => T.samples.push({ at: Math.round(performance.now()), ...T.now() }), 500);
  });
  const show = (x) => `controls ${x.controls}x, lines priced at ${JSON.stringify(x.mix)}, income ${Math.round(x.income).toLocaleString()}/game hour`;

  const v0 = await page.evaluate(() => window.__t.now());
  out.before = v0;
  console.log(`before (demand version ${v0.version}): ${show(v0)}`);

  await page.evaluate(() => window.__t.edit(10));
  await page.waitForFunction(() => window.__t.controls() === 10, undefined, { timeout: 30000, polling: 100 });
  await page.waitForTimeout(1500);
  const e10 = await page.evaluate(() => window.__t.now());
  out.after10 = e10;
  const spike = await page.evaluate(() => window.__t.spike());
  out.incomeWithoutTheRule = spike;
  console.log(`10x set: ${show(e10)}; without the rule the same riders would pay ${Math.round(spike).toLocaleString()}/game hour (${(spike / v0.income).toFixed(2)}x)`);
  check('without the rule the same moment would spike', spike / v0.income > 2, `${(spike / v0.income).toFixed(2)}x`);
  check('fare controls show the new curve at once', e10.controls === 10, `${e10.controls}x`);
  check('income keeps the old fares after the edit', Math.abs(e10.income / v0.income - 1) < 0.001, `${(e10.income / v0.income).toFixed(4)} of before`);

  // a second edit while the first is being solved
  await page.waitForTimeout(3000);
  await page.evaluate(() => window.__t.edit(5));
  await page.waitForFunction(() => window.__t.controls() === 5, undefined, { timeout: 30000, polling: 100 });
  await page.waitForTimeout(1500);
  const e5 = await page.evaluate(() => window.__t.now());
  out.after5 = e5;
  console.log(`then 5x: ${show(e5)}`);
  check('fare controls show the second edit, income still on the old fares', e5.controls === 5 && Math.abs(e5.income / v0.income - 1) < 0.001);

  // the final of the solve made with the last curve
  await demandFinal(page, { timeout: 1800000, polling: 1000, after: v0.version });
  await page.waitForTimeout(3000);
  const f = await page.evaluate(() => window.__t.now());
  const seen = await page.evaluate(() => window.__t.samples);
  out.final = f;
  out.samples = seen;
  console.log(`final (demand version ${f.version}): ${show(f)}`);
  const oldOnly = seen.filter((s) => Object.keys(s.mix).length === 1 && s.mix[1] !== undefined);
  const worstOld = Math.max(0, ...oldOnly.map((s) => Math.abs(s.income / v0.income - 1)));
  const mixed = seen.filter((s) => Object.keys(s.mix).length > 1).length;
  console.log(`${seen.length} samples: ${oldOnly.length} with every line on the old fares (worst income change ${(worstOld * 100).toFixed(3)}%), ${mixed} with some lines on the new`);
  check('income does not move while every line is on the old fares', oldOnly.length > 5 && worstOld < 0.001, `${oldOnly.length} samples, worst ${(worstOld * 100).toFixed(3)}%`);
  const peak = Math.max(...seen.map((s) => s.income));
  check('income before the final stays far under the no-rule spike', peak < spike / 2, `highest ${Math.round(peak).toLocaleString()} against ${Math.round(spike).toLocaleString()}`);
  check('the 10x curve was never used', !seen.some((s) => s.mix[10] !== undefined));
  check('after the final every line is priced at the last curve (5x)', Object.keys(f.mix).length === 1 && f.mix[5] > 0 && f.controls === 5, JSON.stringify(f.mix));
  check('income moved to the new-fare figure', Math.abs(f.income / v0.income - 1) > 0.02, `${(f.income / v0.income).toFixed(3)} of before`);
} catch (e) {
  console.log('FAILED', e.message);
  out.error = e.message;
  failed++;
} finally {
  out.errors = errors.slice(0, 5);
  check('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));
  await mkdir(dirname(OUT), { recursive: true });
  await writeFile(OUT, JSON.stringify(out, null, 1));
  await b.close();
  process.exitCode = failed ? 1 : 0;
}
