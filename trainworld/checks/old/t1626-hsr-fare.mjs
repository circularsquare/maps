// T-1626: the high-speed fare multiplier. A new game shows no "High speed" fare row; once a line runs
// high-speed trains it does. On that line, multipliers 1.0, 1.5 and 2.0 each re-solve demand: its
// riders fall and its fare income a rider rises as the multiplier goes up. Prints the figures, a
// shot of the Money tab to artifacts/trainworld/t1626-money.png, PASS or FAIL lines; exits 1 on a
// failure. About three minutes.
//   node trainworld/checks/t1626-hsr-fare.mjs [url]     (from the maps root; default localhost:6176)
import { join } from 'node:path';
import { ARTIFACTS, ANGLE, collectErrors, demandFinal, launchChrome } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6176/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1626-', size: [1300, 900], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await page.click('button.tab[aria-label="Money"]');
  await page.waitForSelector('ul.fare-rows > li');
  const hsrRow = () => page.$$eval('ul.fare-rows > li', (lis) => {
    const li = lis.find((l) => l.querySelector('.fr-what')?.textContent === 'High speed');
    return li ? li.querySelector('.stepper .v')?.textContent ?? '' : null;
  });
  report('new game: no high-speed fare row', (await hsrRow()) === null);

  // a high-speed line east from Midtown, three stations about 15 km apart
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const pts = [[-73.99, 40.75], [-73.83, 40.70], [-73.64, 40.72]].map(([lng, lat]) => tw.lngLatToPack(lng, lat));
    const near = (p) => [...tw.world.value.nodes.values()].sort((m, n) => Math.hypot(m.x - p.x, m.y - p.y) - Math.hypot(n.x - p.x, n.y - p.y))[0].id;
    const r0 = await c.edit({ op: 'route', from: { kind: 'free', x: pts[0].x, y: pts[0].y, level: 1 }, to: { kind: 'free', x: pts[1].x, y: pts[1].y, level: 1 }, pis: [], single: false, through: true, hsr: true });
    await sleep(500);
    const r1 = await c.edit({ op: 'route', from: { kind: 'node', node: near(pts[1]) }, to: { kind: 'free', x: pts[2].x, y: pts[2].y, level: 1 }, pis: [], single: false, through: true, hsr: true });
    await sleep(500);
    for (const p of pts) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await sleep(250); }
    const built = await c.edit({ op: 'constructAll' });
    await sleep(600);
    const line = await c.edit({ op: 'addLine', stops: pts.map(near), name: 'Check', colour: '#d33682' });
    await sleep(300);
    await c.edit({ op: 'setTrainCount', line: line.line, trains: [8, 6, 4] });
    await sleep(300);
    return { ok: [r0?.ok, r1?.ok, built?.ok, line?.ok], line: line?.line, issues: built?.issues };
  });
  console.log(JSON.stringify(made));
  await page.waitForTimeout(800);
  report('an ordinary line: still no high-speed fare row', (await hsrRow()) === null);
  await page.evaluate((l) => window.tw.client.edit({ op: 'lineFast', line: l, fast: true }), made.line);
  await page.waitForFunction((l) => window.tw.world.value.lineStats[l]?.fast, made.line, { timeout: 20000 });
  await page.waitForTimeout(800);
  const shown = await hsrRow();
  report('a high-speed line: the row shows the default 1.50x', shown === '1.50x', String(shown));

  const figures = async (mul) => {
    const v0 = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
    await page.evaluate((m) => window.tw.client.edit({ op: 'fareHsr', mul: m }), mul);
    await page.waitForFunction((m) => window.tw.moneyState.value.fares.hsr === m, mul, { timeout: 20000 });
    await demandFinal(page, { after: v0 });
    await page.waitForTimeout(1000);
    return page.evaluate(async (l) => {
      const { incomeByDay } = await import('/src/game/money.ts');
      const { PERIODS } = await import('/src/game/periods.ts');
      const lm = incomeByDay.value[0].lines.get(l) ?? incomeByDay.value[0].lines.get(String(l));
      const hours = PERIODS.map((p) => p.to - p.from);
      const dm = window.twDemand.view().lines.get(String(l));
      const day = (a) => a.reduce((s, v, p) => s + v * (hours ? hours[p] : 1), 0);
      const econ = window.tw.moneyState.value.economy;
      const fx = (dm?.fareX ?? []).reduce((s, v) => s + v, 0) * econ;
      return { riders: dm?.ridersPerDay ?? 0, fares: lm ? day(lm.perHour) * 1e6 / econ : 0, rides: lm ? day(lm.rides) : 0, fx: fx / econ };
    }, made.line);
  };
  const out = {};
  for (const m of [1, 1.5, 2]) {
    out[m] = await figures(m);
    console.log(`x${m.toFixed(1)}: riders ${Math.round(out[m].riders)} a day, fares $${Math.round(out[m].fares).toLocaleString("en-US")} a day (curve extra $${Math.round(out[m].fx).toLocaleString("en-US")}), $${(out[m].fares / out[m].rides).toFixed(2)} a ride`);
  }
  const [a, b, c] = [out[1], out[1.5], out[2]];
  report('riders fall as the multiplier rises', a.riders > b.riders && b.riders > c.riders, `${Math.round(a.riders)} > ${Math.round(b.riders)} > ${Math.round(c.riders)}`);
  report('fares a ride rise as the multiplier rises', a.fares / a.rides < b.fares / b.rides && b.fares / b.rides < c.fares / c.rides);
  report('fare income rises from 1.0 to 1.5', b.fares > a.fares, `${Math.round(a.fares)} -> ${Math.round(b.fares)}`);

  await page.evaluate((m) => window.tw.client.edit({ op: 'fareHsr', mul: m }), 1.75);
  await page.waitForTimeout(800);
  await page.$eval('ul.fare-rows', (e) => e.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  const box = await page.$eval('ul.fare-rows', (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  const shot = join(ARTIFACTS, 't1626-money.png');
  await page.screenshot({ path: shot, clip: { x: Math.max(0, box.x - 12), y: Math.max(0, box.y - 260), width: box.width + 24, height: box.height + 280 } });
  console.log('shot', shot);
  report('a set multiplier shows with its back button', (await hsrRow()) === '1.75x' && !!(await page.$('ul.fare-rows button[aria-label="Back to the usual multiple"]')));
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-1626 PASSED' : 'T-1626 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
