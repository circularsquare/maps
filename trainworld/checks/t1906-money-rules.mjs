import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { launchChrome, collectErrors, importSave, registry, REAL_SAVE, ANGLE, ARTIFACTS } from './lib.mjs';
// T-1906: a new game starts with round fares for where it starts (New York $2.50 base, Tokyo $1.30),
// every other country at its own start fares by income, and every dial at 1.00x; the real save keeps its fares and stored scales and reads
// 1.00x too; a loan whose payment a day (T-2022) is past the last week's profit a day is refused, with the
// reason in its hover.
//   node trainworld/checks/t1906-money-rules.mjs [url]     (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure; pictures in artifacts/trainworld/t1906/. About a minute.

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const OUT = `${ARTIFACTS}t1906/`;
const tokyo = registry().cities.find((c) => /tokyo/i.test(c.id) || /^Tokyo/.test(c.name))?.id;
const { ctx, close } = await launchChrome({ prefix: 'tw-t1906-', size: [1300, 900], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
const near = (a, b) => Math.abs(a - b) < 1e-6;
const sameNodes = (a, b) => a.length === b.length && a.every(([k, v], i) => near(k, b[i][0]) && near(v, b[i][1]));

async function open(page, query) {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2${query}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });
  await page.click('button.tab[data-guide-tab="money"]');
  await page.waitForSelector('ul.fare-rows > li');
}
const scaleRow = (page) => page.$$eval('ul.fare-rows > li', (lis) => {
  const li = lis.find((l) => / fares$/.test(l.querySelector('.fr-what')?.textContent ?? ''));
  return li && { name: li.querySelector('.fr-what').textContent, value: li.querySelector('.stepper .v')?.textContent };
});
const loanButtons = (page) => page.$$eval('button.btn.text', (bs) => bs.filter((b) => /^\$\d+B$/.test(b.textContent)).map((b) => ({ text: b.textContent, disabled: b.disabled, tip: b.dataset.tip })));
const scalesIn = (page) => page.evaluate(async () => {
  const { fareScale } = await import('/src/game/income.ts');
  const { roundedDefault } = await import('/src/game/fareDefault.ts');
  const f = window.tw.moneyState.value.fares;
  const cs = ['US', 'JP', 'IN', 'DE', 'CN'];
  return {
    nodes: f.nodes, scales: f.scales,
    at: Object.fromEntries(cs.map((c) => [c, fareScale(c, f.scales)])),
    base: Object.fromEntries(cs.map((c) => [c, roundedDefault(f, c)?.base ?? f.nodes[0][1] * fareScale(c, f.scales)])),
  };
});
const BASES = { US: 2.5, JP: 1.3, IN: 0.22, DE: 1.9, CN: 0.65 };
const basesOk = (b) => Object.entries(BASES).every(([c, v]) => near(b[c], v));

try {
  const page = await ctx.newPage();
  collectErrors(page, errors);

  // 1. a new game in New York
  await open(page, '');
  const ny = await scalesIn(page);
  report('New York: fares $2.50, $6.50 at 20 km, $22.50 at 100 km', sameNodes(ny.nodes, [[0, 2.5], [20, 6.5], [100, 22.5]]), JSON.stringify(ny.nodes));
  report('New York: start income scale stored, each country at its own default base (US 2.50, JP 1.30, IN 0.22, DE 1.90, CN 0.65)', JSON.stringify(ny.scales) === '{"@":11.6}' && ny.at.US === 1 && basesOk(ny.base), JSON.stringify(ny));
  const own = await page.evaluate(async () => {
    const { setFareScale } = await import('/src/game/world.ts');
    const { fareScale, storedScale, shownScale } = await import('/src/game/income.ts');
    await setFareScale('JP', storedScale(2, 'JP', window.tw.moneyState.value.fares.scales));
    await new Promise((r) => setTimeout(r, 500));
    const s = window.tw.moneyState.value.fares.scales;
    const out = { scales: s, jp: fareScale('JP', s), shown: shownScale(fareScale('JP', s), 'JP', s), old: [fareScale('US', { '*': 1 }), fareScale('JP', { '*': 1, JP: 2 })] };
    await setFareScale('JP', null);
    await new Promise((r) => setTimeout(r, 500));
    return out;
  });
  report(`Japan's dial set to 2.00x: stored 2, charged twice its default, reads 2.00x; a "*" save still reads`, own.scales.JP === 2 && near(own.jp, 2 * 6.1 / 11.6) && near(own.shown, 2) && own.old[0] === 1 && own.old[1] === 2, JSON.stringify(own));
  const nyRow = await scaleRow(page);
  report('New York: the panel reads "United States fares 1.00x"', nyRow?.name === 'United States fares' && nyRow?.value === '1.00x', JSON.stringify(nyRow));
  const loans = await loanButtons(page);
  report('New York day 1: $1B allowed, $5B and $10B refused with the reason', loans.length === 3 && !loans[0].disabled && loans[1].disabled && loans[2].disabled && /full day of fares/.test(loans[1].tip), JSON.stringify(loans));
  const r = await page.evaluate(async () => {
    const { borrow } = await import('/src/game/world.ts');
    const big = await borrow(5000);
    const loanAfterBig = window.tw.moneyState.value.loan;
    return { big, loanAfterBig };
  });
  report('the worker refuses a $5B loan on day 1', r.loanAfterBig === null, JSON.stringify(r));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [139.7, 35.69], zoom: 10 }));
  await page.waitForFunction(() => [...document.querySelectorAll('ul.fare-rows .fr-what')].some((e) => e.textContent === 'Japan fares'), undefined, { timeout: 30000 }).catch(() => {});
  const jpRow = await scaleRow(page);
  report('New York game, viewing Tokyo: "Japan fares 1.00x"', jpRow?.name === 'Japan fares' && jpRow?.value === '1.00x', JSON.stringify(jpRow));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.99, 40.73], zoom: 11 }));
  await page.waitForTimeout(800);
  const b5 = await page.$$('button.btn.text');
  for (const b of b5) if ((await b.textContent()) === '$5B') {
    await b.scrollIntoViewIfNeeded();
    const box = await b.boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  }
  await page.waitForTimeout(600);
  await page.screenshot({ path: `${OUT}new-nyc-money.png` });

  // 2. a new game in Tokyo
  await open(page, `&city=${tokyo}`);
  const tk = await scalesIn(page);
  report(`Tokyo (${tokyo}): base $1.30`, near(tk.nodes[0][1], 1.3), JSON.stringify(tk.nodes));
  report('Tokyo: start income scale stored, each country at its own default base (US 2.50)', JSON.stringify(tk.scales) === '{"@":6.1}' && tk.at.JP === 1 && basesOk(tk.base), JSON.stringify(tk));
  const tkRow = await scaleRow(page);
  report('Tokyo: "Japan fares 1.00x"', tkRow?.name === 'Japan fares' && tkRow?.value === '1.00x', JSON.stringify(tkRow));
  await page.$eval('ul.fare-rows > li:last-child', (li) => li.scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}new-tokyo-money.png` });

  // 3. the profit rule
  const rule = await page.evaluate(async () => {
    const { loanProfit, loanAllowed } = await import('/src/game/loan.ts');
    const days = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((day) => ({ day, fares: 100, running: 40, upkeep: 20, trains: 500, build: 900 }));
    const p = loanProfit(days, 9);
    const young = loanProfit([{ day: 1, fares: 0, running: 0, trains: 0, build: 50 }], 2);
    const loss = loanProfit([{ day: 1, fares: 10, running: 40, trains: 0, build: 0 }], 2);
    return { p, k1: loanAllowed(1000, p), k5: loanAllowed(5000, p), young, y1: loanAllowed(1000, young), y5: loanAllowed(5000, young), loss, l1: loanAllowed(1000, loss) };
  });
  report('profit a day: fares less running and upkeep over the last 7 whole days', rule.p?.perDay === 40 && rule.p?.days === 7, JSON.stringify(rule.p));
  report('$40M a day of profit: $1B ($22M a day, T-2022) allowed, $5B ($111M) refused', rule.k1 && !rule.k5);
  report('no day with fares yet: only $1B', rule.young === null && rule.y1 && !rule.y5);
  report('a loss: no loan', !rule.l1, JSON.stringify(rule.loss));

  // 4. the real save keeps its fares
  const raw = gunzipSync(readFileSync(REAL_SAVE));
  const hdr = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + new DataView(raw.buffer, raw.byteOffset + 4, 4).getUint32(0, true))));
  await open(page, '');
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  const real = await scalesIn(page);
  const want = (km) => hdr.fares.base + hdr.fares.perKm * km;
  const fares = await page.evaluate(async () => {
    const { fareAt } = await import('/src/game/fareCurve.ts');
    const { fareScale } = await import('/src/game/income.ts');
    const f = window.tw.moneyState.value.fares;
    return [0, 10, 20, 50, 100].map((km) => [km, fareAt(f.nodes, km) * fareScale('US', f.scales)]);
  });
  report('real save: New York fares as saved (within the income scale rounding)', fares.every(([km, v]) => Math.abs(v / want(km) - 1) < 0.005), fares.map(([km, v]) => `${km} km $${v.toFixed(3)} (want $${want(km).toFixed(3)})`).join('; '));
  report('real save: no scales stored, income defaults (US 11.6, JP 6.1, IN 1)', Object.keys(real.scales ?? {}).length === 0 && real.at.US === 11.6 && real.at.JP === 6.1 && real.at.IN === 1, JSON.stringify(real));
  await page.click('button.tab[data-guide-tab="money"]').catch(() => {});
  await page.waitForSelector('ul.fare-rows > li');
  const realRow = await scaleRow(page);
  report('real save: the panel reads 1.00x', realRow?.value === '1.00x', JSON.stringify(realRow));
  const realLoans = await loanButtons(page);
  console.log(`  real save loan buttons: ${JSON.stringify(realLoans)}`);
  await page.screenshot({ path: `${OUT}real-save-money.png` });
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-1906 PASSED' : 'T-1906 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
