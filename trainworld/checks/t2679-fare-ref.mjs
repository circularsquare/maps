import { launchChrome, collectErrors, buildLine, demandFinal, registry, ANGLE, ARTIFACTS } from './lib.mjs';
// T-2679: the Money tab's fare multiples are of the reference country's fares, the first built station's country.
// A new game in New York with a line there and one in Tijuana: the US row reads a plain 1.00x with its hover,
// Mexico 0.40x and Japan 0.63x as steppers; making Mexico the reference reads Mexico 1.00x and the US 2.52x;
// with Mexico at 0.45x and Japan at 0.65x, switching changes no fare: a New York and a Tijuana trip, each line's
// fare income a day and the network's are the same to the last digit, and demand does not solve again.
//   node trainworld/checks/t2679-fare-ref.mjs [url]     (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure; pictures in artifacts/trainworld/t2679/. About two minutes.

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const OUT = `${ARTIFACTS}t2679/`;
const tokyo = registry().cities.find((c) => /tokyo/i.test(c.id) || /^Tokyo/.test(c.name))?.id;
const NY = [-73.99, 40.73], TJ = [-117.02, 32.515];
const { ctx, close } = await launchChrome({ prefix: 'tw-t2679-', size: [1300, 900], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
const across = ([lng, lat], name) => {
  const d = 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
  return [-1, 0, 1].map((k) => [lng + k * d, lat]);
};

/** the country row as shown with `city` selected */
async function rowAt(page, city) {
  await page.evaluate(async (id) => { const { selection } = await import('/src/game/ui.ts'); selection.value = { kind: 'city', city: id }; }, city);
  await page.click('button.tab[data-guide-tab="money"]').catch(() => {});
  await page.waitForTimeout(400);
  return page.$$eval('ul.fare-rows > li', (lis) => {
    const li = lis.find((l) => / fares$/.test(l.querySelector('.fr-what')?.textContent ?? ''));
    if (!li) return null;
    const v = li.querySelector('.stepper .v') ?? li.querySelector('.fr-ref');
    return { name: li.querySelector('.fr-what').textContent, value: v?.textContent, tip: v?.dataset.tip ?? null, stepper: !!li.querySelector('.stepper'), ref: !!li.querySelector('button[aria-label="Use as the reference country"]') };
  });
}
const clickRef = (page) => page.$$eval('ul.fare-rows button[aria-label="Use as the reference country"]', (bs) => bs[0]?.click());
const clickPlus = (page, n) => page.evaluate(async (n) => {
  for (let k = 0; k < n; k++) {
    const li = [...document.querySelectorAll('ul.fare-rows > li')].find((l) => / fares$/.test(l.querySelector('.fr-what')?.textContent ?? ''));
    li.querySelectorAll('.stepper > button.btn')[1].click();
    await new Promise((r) => setTimeout(r, 400));
  }
}, n);

/** fares charged and income a day, everything that a reference switch must leave */
const figures = (page) => page.evaluate(async () => {
  const { fareScale } = await import('/src/game/income.ts');
  const { roundedDefault } = await import('/src/game/fareDefault.ts');
  const { fareAt, scaleNodes } = await import('/src/game/fareCurve.ts');
  const { incomeByDay } = await import('/src/game/money.ts');
  const { PERIODS } = await import('/src/game/periods.ts');
  const f = window.tw.moneyState.value.fares;
  const { ref, ...rest } = f;
  const charged = (c) => roundedDefault(f, c)?.nodes ?? scaleNodes(f.nodes, fareScale(c, f.scales));
  const trip = (c, km) => Math.round(fareAt(charged(c), km) * 1e6) / 1e6;
  const day = (perHour) => perHour.reduce((a, v, p) => a + v * (PERIODS[p].to - PERIODS[p].from), 0) * 1e6;
  const inc = incomeByDay.value[0];
  const w = window.tw.world.value;
  const lines = Object.fromEntries(w.save.lines.map((l) => [l.name, day(inc.lines.get(l.id)?.perHour ?? [])]));
  const d = window.twDemand;
  return {
    ref: ref ?? null, fares: JSON.stringify(rest),
    trips: { us4: trip('US', 4), us20: trip('US', 20), mx4: trip('MX', 4), mx20: trip('MX', 20), jp20: trip('JP', 20) },
    lines, network: day(inc.perHour), demand: d.view()?.version,
  };
});
const sameFigures = (a, b) => a.fares === b.fares && JSON.stringify(a.trips) === JSON.stringify(b.trips) && JSON.stringify(a.lines) === JSON.stringify(b.lines) && a.network === b.network && a.demand === b.demand;
const show = (f) => `ref ${f.ref}; trips ${Object.entries(f.trips).map(([k, v]) => `${k} $${v.toFixed(4)}`).join(', ')}; a day: ${Object.entries(f.lines).map(([k, v]) => `${k} $${v.toFixed(2)}`).join(', ')}, network $${f.network.toFixed(2)}; demand v${f.demand}`;

try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });
  const ny = await buildLine(page, across(NY), { name: 'NY', trains: [6, 4, 3] });
  await page.waitForTimeout(1000);
  const pinned = await page.evaluate(() => window.tw.moneyState.value.fares.ref);
  report('the first line, in New York, makes the US the reference country', ny?.ok !== false && pinned === 'US', `${JSON.stringify(ny?.issues ?? [])} ref ${pinned}`);
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 11 }), TJ);
  await page.waitForTimeout(3000);
  const tj = await buildLine(page, across(TJ), { name: 'TJ', trains: [6, 4, 3] });
  report('a line in Tijuana built; the reference stays the US', tj?.ok !== false && (await page.evaluate(() => window.tw.moneyState.value.fares.ref)) === 'US', JSON.stringify(tj?.issues ?? []));
  await demandFinal(page, { lines: 1 });
  await page.waitForTimeout(1500);

  // 1. defaults, the US the reference
  const usRow = await rowAt(page, 'nyc'), mxRow = await rowAt(page, 'tij'), jpRow = await rowAt(page, tokyo);
  report('US row: a plain 1.00x, no stepper or reference button, hover names it the reference',
    usRow?.name === 'United States fares' && usRow.value === '1.00x' && !usRow.stepper && !usRow.ref && usRow.tip === 'Reference country\nOther countries\' fares are set as a multiple of United States fares.', JSON.stringify(usRow));
  report('Mexico row: 0.40x on a stepper with a reference button; hover "Mexico fares as a multiple of United States fares"',
    mxRow?.name === 'Mexico fares' && mxRow.value === '0.40x' && mxRow.stepper && mxRow.ref && mxRow.tip === 'Mexico fares as a multiple of United States fares', JSON.stringify(mxRow));
  report('Japan row: 0.63x', jpRow?.name === 'Japan fares' && jpRow.value === '0.63x', JSON.stringify(jpRow));
  // selecting Tokyo opens its zone, which demand solves
  await page.waitForTimeout(2000);
  await demandFinal(page, { lines: 1 });
  await page.waitForTimeout(1500);
  const d0 = await figures(page);
  console.log(`  defaults, US the reference: ${show(d0)}`);
  await rowAt(page, 'tij');
  await page.$eval('ul.fare-rows > li:last-child', (li) => li.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  const shotUs = await page.locator('ul.fare-rows').screenshot();
  await clickRef(page);
  await page.waitForTimeout(800);
  const d1 = await figures(page);
  const mxRef = await rowAt(page, 'tij'), usRel = await rowAt(page, 'nyc');
  report('Mexico made the reference: Mexico a plain 1.00x, the US 2.52x', d1.ref === 'MX' && mxRef?.value === '1.00x' && !mxRef.stepper && usRel?.value === '2.52x' && usRel.tip === 'United States fares as a multiple of Mexico fares', `${JSON.stringify(mxRef)} ${JSON.stringify(usRel)}`);
  report('defaults: the switch changes no fare, income or demand', sameFigures(d0, d1), show(d1));
  await clickRef(page); // the US back
  await page.waitForTimeout(800);

  // 2. Mexico at 0.45x, Japan at 0.65x
  await rowAt(page, 'tij');
  await clickPlus(page, 1);
  await rowAt(page, tokyo);
  await clickPlus(page, 1);
  await page.evaluate(async (id) => { const { selection } = await import('/src/game/ui.ts'); selection.value = null; }, null);
  await demandFinal(page, { lines: 1 });
  await page.waitForTimeout(1500);
  const set = await page.evaluate(async () => {
    const { fareScale } = await import('/src/game/income.ts');
    const s = window.tw.moneyState.value.fares.scales;
    return { ref: window.tw.moneyState.value.fares.ref, mx: fareScale('MX', s) / fareScale('US', s), jp: fareScale('JP', s) / fareScale('US', s), scales: s };
  });
  report('the steppers set Mexico 0.45x and Japan 0.65x, exactly that many times US fares', set.ref === 'US' && Math.abs(set.mx - 0.45) < 1e-12 && Math.abs(set.jp - 0.65) < 1e-12, JSON.stringify(set));
  const rowsA = [await rowAt(page, 'nyc'), await rowAt(page, 'tij'), await rowAt(page, tokyo)].map((r) => r?.value);
  const a = await figures(page);
  console.log(`  set, US the reference: ${show(a)}`);
  await rowAt(page, 'tij');
  await page.$eval('ul.fare-rows > li:last-child', (li) => li.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  const shotMxSet = await page.locator('ul.fare-rows').screenshot();
  await clickRef(page);
  await page.waitForTimeout(800);
  const b = await figures(page);
  console.log(`  set, Mexico the reference: ${show(b)}`);
  const rowsB = [await rowAt(page, 'nyc'), await rowAt(page, 'tij'), await rowAt(page, tokyo)].map((r) => r?.value);
  report('rows: US 1.00x, Mexico 0.45x, Japan 0.65x (from 0.63x, to the next 0.05x); Mexico the reference: US 2.22x, Mexico 1.00x, Japan 1.44x', JSON.stringify(rowsA) === '["1.00x","0.45x","0.65x"]' && JSON.stringify(rowsB) === '["2.22x","1.00x","1.44x"]', `${JSON.stringify(rowsA)} ${JSON.stringify(rowsB)}`);
  report('Mexico made the reference: the same fares to the last digit (New York and Tijuana trips, each line a day, the network), no new solve', sameFigures(a, b) && b.ref === 'MX');
  const shotUsRel = await (async () => { await rowAt(page, 'nyc'); await page.$eval('ul.fare-rows > li:last-child', (li) => li.scrollIntoView({ block: 'center' })); await page.waitForTimeout(300); return page.locator('ul.fare-rows').screenshot(); })();
  const shotMxRef = await (async () => { await rowAt(page, 'tij'); await page.waitForTimeout(300); return page.locator('ul.fare-rows').screenshot(); })();
  // the four views side by side
  const img = (buf, cap) => `<figure><figcaption>${cap}</figcaption><img src="data:image/png;base64,${buf.toString('base64')}"></figure>`;
  const sheet = await ctx.newPage();
  await sheet.setViewportSize({ width: 760, height: 300 });
  await sheet.setContent(`<style>figure{margin:0}</style><body style="margin:12px;font:13px system-ui;background:#fff;display:grid;grid-template-columns:repeat(2,auto);gap:16px 24px;justify-content:start;align-content:start">${img(shotUs, 'US the reference, viewing Tijuana: Mexico at its default')}${img(shotMxSet, 'US the reference, viewing Tijuana: Mexico set to 0.45x')}${img(shotUsRel, 'Mexico the reference, viewing New York')}${img(shotMxRef, 'Mexico the reference, viewing Tijuana')}</body>`);
  await sheet.screenshot({ path: `${OUT}fare-rows.png`, fullPage: true });
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-2679 PASSED' : 'T-2679 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
