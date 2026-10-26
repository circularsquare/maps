// T-1300: the fare panel's estimate of rail trips and fare income a day against demand's own solve.
// The real save's weekday: from each solved curve, the estimate for the next one, then that curve
// set and solved; the estimate's error against the solve, step by step.
//   node trainworld/checks/t1300-fare-estimate.mjs <url of a dev server>
// Writes artifacts/trainworld/t1300/estimate.json; PASS when every step is within 10%.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2];
if (!BASE) throw new Error('usage: t1300-fare-estimate.mjs <url>');
// each step's curve from the save's: every fare times x, the base fare alone, or the rates alone
const STEPS = [['all x1.25', 'all', 1.25], ['all x1.5', 'all', 1.5], ['all x2', 'all', 2], ['all x0.75', 'all', 0.75], ['all x1', 'all', 1], ['base x2', 'base', 2], ['rates x2', 'rates', 2], ['all x1 again', 'all', 1]];

const { ctx, close } = await launchChrome({ prefix: 'tw-t1300-' });
let fails = 0;
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 500, lines: 100 });
  const start = await page.evaluate(() => window.tw.moneyState.value.fares.nodes);
  const curve = (kind, x) => kind === 'all' ? start.map(([k, f]) => [k, f * x])
    : kind === 'base' ? start.map(([k, f]) => [k, f + start[0][1] * (x - 1)])
    : start.map(([k, f]) => [k, start[0][1] + (f - start[0][1]) * x]);
  const rows = [];
  for (const [name, kind, x] of STEPS) {
    const to = curve(kind, x);
    const r = await page.evaluate(async (to) => {
      const { estimateFares } = await import('/src/game/fareEstimate.ts');
      const { setFares } = await import('/src/game/world.ts');
      const est = estimateFares(to);
      const v0 = window.twDemand.view().version;
      await setFares(to);
      return { est, v0 };
    }, to);
    await demandFinal(page, { polling: 500, after: r.v0 });
    // income follows the new riders once the money state has the new curve
    await page.waitForTimeout(1500);
    const now = await page.evaluate(async () => {
      const { estimateFares } = await import('/src/game/fareEstimate.ts');
      const e = estimateFares(window.tw.moneyState.value.fares.nodes);
      return { riders: e.ridersNow, fares: e.faresNow };
    });
    const er = r.est.riders / now.riders - 1, ef = r.est.fares / now.fares - 1;
    // the error as a share of the change, where the change is big enough to measure it by
    const dr = now.riders - r.est.ridersNow, df = now.fares - r.est.faresNow;
    const row = { name, before: { riders: r.est.ridersNow, fares: r.est.faresNow }, estimate: { riders: r.est.riders, fares: r.est.fares }, solved: now, riderErr: er, fareErr: ef, riderErrOfChange: Math.abs(dr) > 1e-3 * now.riders ? (r.est.riders - r.est.ridersNow - dr) / dr : null, fareErrOfChange: Math.abs(df) > 1e-3 * now.fares ? (r.est.fares - r.est.faresNow - df) / df : null };
    rows.push(row);
    const ok = Math.abs(er) <= 0.1 && Math.abs(ef) <= 0.1;
    if (!ok) fails++;
    const pc = (v) => (v === null ? '-' : (100 * v).toFixed(1) + '%');
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: riders ${Math.round(r.est.ridersNow)} -> est ${Math.round(r.est.riders)}, solved ${Math.round(now.riders)} (${pc(er)}; ${pc(row.riderErrOfChange)} of the change); fares $${(r.est.faresNow / 1e6).toFixed(2)}M -> est $${(r.est.fares / 1e6).toFixed(2)}M, solved $${(now.fares / 1e6).toFixed(2)}M (${pc(ef)}; ${pc(row.fareErrOfChange)} of the change)`);
  }
  const dir = join(ARTIFACTS, 't1300');
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'estimate.json'), JSON.stringify(rows, null, 1) + '\n');
} finally {
  await close();
}
console.log(`${fails ? 'FAIL' : 'PASS'} t1300 fare estimate: ${STEPS.length - fails} of ${STEPS.length} steps within 10%`);
process.exitCode = fails ? 1 : 0;
