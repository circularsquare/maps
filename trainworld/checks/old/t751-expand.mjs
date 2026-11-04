// T-751, T-1844: a refused + on train length (platforms too short) shakes, knocks and offers
// "Expand all"; the button lengthens the platforms, charges them and lengthens the trains; with too
// little cash it refuses the same way. Steppers at a limit shake and knock too.
//   node trainworld/checks/old/t751-expand.mjs [url] [shots folder]    (from the maps root; a running dev server)
import { mkdirSync } from 'node:fs';
import { NOISE, acrossLine, collectErrors, launchChrome } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6193/';
const SHOTS = process.argv[3] ?? 'artifacts/trainworld/t751';
mkdirSync(SHOTS, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t751-', size: [1400, 900] });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.40, 40.64], zoom: 14 }));
  await page.waitForTimeout(1500);
  const line = await acrossLine(page, [-74.40, 40.64], 2);
  check('a line built', line !== null);
  if (line === null) throw new Error('no line');
  const id = String(line);
  const settle = async () => {
    await page.waitForFunction(async () => {
      const w = await import('/src/game/world.ts');
      return w.pendingCars.value.size === 0 && w.pendingTrains.value.size === 0 && w.pendingHeadway.value.size === 0;
    }, undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };
  const info = () => page.evaluate((id) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.id === id), s = w.lineStats[id];
    const st = new Map(w.save.stations.map((x) => [x.id, x]));
    return { cars: s.cars, lens: l.stops.map((x) => st.get(x).length), built: l.stops.map((x) => st.get(x).built), cash: window.tw.moneyState?.value?.cash };
  }, id);
  const cash = () => page.evaluate(async () => (await import('/src/game/money.ts')).liveMoney.value.cash);

  // platforms 120, 120, 200 m: the line fits 6 cars; the trains run at 6
  const nodes = await page.evaluate((id) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.id === id), st = new Map(w.save.stations.map((x) => [x.id, x]));
    return l.stops.map((x) => ({ num: st.get(x).num, name: st.get(x).name }));
  }, id);
  for (const [k, len] of [[0, 120], [1, 120]]) {
    await page.evaluate(async ([n, len]) => (await (await import('/src/game/world.ts')).edit({ op: 'stationProps', node: n.num, platform: len, name: n.name }, true)), [nodes[k], len]);
  }
  await page.evaluate(async (line) => (await import('/src/game/world.ts')).edit({ op: 'lineCars', line, cars: 0 }, true), line);
  await page.evaluate(async (id) => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; ui.selection.value = { kind: 'line', line: id }; }, id);
  await page.waitForSelector('button[aria-label="More train length"]');
  await page.waitForTimeout(1000);
  const i0 = await info();
  check('line fits 6 cars', i0.cars === 6, JSON.stringify(i0));

  // count the refusal knocks
  await page.evaluate(async () => { window.__knocks = 0; (await import('/src/game/world.ts')).setRefuseHook(() => { window.__knocks++; }); });
  const more = 'button[aria-label="More train length"]';
  const shaking = () => page.evaluate(() => !!document.querySelector('.stepper.shake'));
  await page.click(more);
  await page.waitForTimeout(80);
  check('+ at the cap shakes the stepper', await shaking());
  check('+ at the cap knocks', (await page.evaluate(() => window.__knocks)) === 1);
  const row = page.locator('li', { hasText: 'too short for' });
  await row.waitFor({ timeout: 5000 });
  const text = await row.innerText();
  console.log('row:', JSON.stringify(text));
  check('row names 2 stations and 7-car trains', /2 stations are too short for 7-car trains/.test(text));
  await page.waitForTimeout(500);
  const btn = row.locator('button');
  check('button has a price', /Expand all for \$/.test(await btn.innerText()), await btn.innerText());
  await page.screenshot({ path: `${SHOTS}/1-offer.png` }).catch(() => {});

  // too little cash: refused, nothing built
  const cashBefore = await cash();
  await page.evaluate(async () => { const m = await import('/src/game/money.ts'); window.__cash = m.liveMoney.value; m.liveMoney.value = { ...m.liveMoney.value, cash: 0.5 }; });
  const k0 = await page.evaluate(() => window.__knocks);
  await btn.click();
  await page.waitForTimeout(80);
  check('Expand all with no money shakes the button', await page.evaluate(() => !!document.querySelector('button.shake')));
  await page.waitForTimeout(600);
  check('and knocks once', (await page.evaluate(() => window.__knocks)) === k0 + 1);
  const i1 = await info();
  check('platforms unchanged', i1.lens[0] === 120 && i1.lens[1] === 120, JSON.stringify(i1.lens));
  await page.screenshot({ path: `${SHOTS}/2-refused.png` }).catch(() => {});
  await page.evaluate(async () => { const m = await import('/src/game/money.ts'); m.liveMoney.value = window.__cash; });

  // with the money: platforms to 140, charged, trains to 7 cars
  const price = await btn.innerText();
  const cashPre = await cash();
  await btn.click();
  await settle();
  const i2 = await info();
  const cashPost = await cash();
  check('platforms now 140 m', i2.lens[0] === 140 && i2.lens[1] === 140 && i2.lens[2] === 200, JSON.stringify(i2.lens));
  check('trains now 7 cars', i2.cars === 7, `cars ${i2.cars}`);
  console.log('button said', price, '; cash', cashPre, '->', cashPost, '; spent M', (cashPre - cashPost).toFixed(2));
  check('cash went down', cashPost < cashPre);
  check('the row is gone', (await row.count()) === 0);
  await page.screenshot({ path: `${SHOTS}/3-expanded.png` }).catch(() => {});

  // undo after a paid platform edit: the history was cleared, so nothing happens to the trains
  await page.evaluate(async () => (await import('/src/game/world.ts')).edit({ op: 'undo' }, true));
  await page.waitForTimeout(1000);
  const i3 = await info();
  console.log('after undo', JSON.stringify(i3));

  // limits: 1 car and the headway/train count at zero shake and knock
  const k1 = await page.evaluate(() => window.__knocks);
  await page.evaluate(async (line) => (await import('/src/game/world.ts')).edit({ op: 'lineCars', line, cars: 1 }, true), line);
  await page.waitForTimeout(800);
  await page.click('button[aria-label="Less train length"]');
  await page.waitForTimeout(80);
  check('- at 1 car shakes and knocks', (await shaking()) && (await page.evaluate(() => window.__knocks)) === k1 + 1);
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
