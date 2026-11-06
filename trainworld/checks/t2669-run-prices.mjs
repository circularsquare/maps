// T-2669, T-2670: running costs at each country's prices. A new game in San Diego with one short line
// there and one in Tijuana: the Tijuana line's running cost factor is Mexico's (0.39), San Diego's 1,
// and the cash drains at the lines' world running cost times their factors.
//   node trainworld/checks/t2669-run-prices.mjs [url]     (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure. A minute or two.
import { launchChrome, collectErrors, buildLine, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t2669-', size: [1200, 800], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&city=snd`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });
  const sd = await buildLine(page, [[-117.162, 32.715], [-117.150, 32.725], [-117.140, 32.735]], { name: 'San Diego', trains: [12, 6, 0] });
  const tj = await buildLine(page, [[-117.037, 32.533], [-117.020, 32.527], [-117.004, 32.511]], { name: 'Tijuana', trains: [12, 6, 0] });
  report('two lines built', !!sd?.ok && !!tj?.ok, JSON.stringify({ sd: sd?.issues, tj: tj?.issues }));
  // the region split reaches the worker, which prices each line and sends the money state again
  await page.waitForFunction(([a, b]) => window.tw.lineRunFactor(a) === 1 && Math.abs(window.tw.lineRunFactor(b) - 0.39) < 0.005, [sd.line, tj.line], { timeout: 60000 }).catch(() => {});
  const got = await page.evaluate(async ([a, b]) => {
    const tw = window.tw;
    await new Promise((r) => setTimeout(r, 1500));
    const w = tw.world.value, m = tw.moneyState.value;
    const lines = w.save.lines.map((l) => ({ num: l.num, f: tw.lineRunFactor(l.num), tph: l.tph.high, km: w.lineStats[l.id]?.km, cars: w.lineStats[l.id]?.cars, status: w.lineStats[l.id]?.status }));
    const want = lines.filter((l) => l.status === 'running').reduce((s, l) => s + l.tph * 2 * l.km * (m.runCostTrainKm + m.runCostCarKm * l.cars) * m.economy / 1e6 * l.f, 0);
    return { lines, want, got: m.runningPerHour[0], fA: tw.lineRunFactor(a), fB: tw.lineRunFactor(b) };
  }, [sd.line, tj.line]);
  report('San Diego line at world prices, Tijuana line at Mexico\'s 0.39x', got.fA === 1 && Math.abs(got.fB - 0.39) < 0.005, JSON.stringify({ sd: got.fA, tj: got.fB }));
  report('the cash drains at world running cost times each line\'s factor', got.want > 0 && Math.abs(got.got - got.want) <= 1e-6 * got.want + 1e-9, `running an hour at the busy level $${(got.got * 1e6).toFixed(0)} against $${(got.want * 1e6).toFixed(0)} worked out ${JSON.stringify(got.lines)}`);
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
console.log(results.every(Boolean) ? 'T-2669 PASSED' : 'T-2669 FAILED');
process.exitCode = results.every(Boolean) ? 0 : 1;
