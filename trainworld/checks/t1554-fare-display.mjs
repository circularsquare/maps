import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { launchChrome, collectErrors, importSave, REAL_SAVE, ANGLE } from './lib.mjs';
// T-1554: the fare scale is shown as a multiple of Japan's fares (Japan's default 1.00x, India's 0.16x)
// under a heading naming the country, and nothing stored changes. Fares per km in the real save's New
// York are the save's own base $1.50 + $0.10 a km within the income scale's rounding (0.4%, as before); the stored scale
// (baseline units, India's) still has its income defaults, and a typed shown value reads back as typed.
//   node trainworld/checks/t1554-fare-display.mjs [url]     (from the maps root; default localhost:6170)
// Prints PASS or FAIL lines and exits 1 on failure. About a minute.

const BASE = process.argv[2] ?? 'http://localhost:6170/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1554-', size: [1300, 850], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
try {
  // the save's own fares: an old one, base and rate in New York's dollars, no scales
  const raw = gunzipSync(readFileSync(REAL_SAVE));
  const hdr = JSON.parse(new TextDecoder().decode(raw.subarray(8, 8 + new DataView(raw.buffer, raw.byteOffset + 4, 4).getUint32(0, true))));
  const want = (km) => hdr.fares.base + hdr.fares.perKm * km;

  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);

  const got = await page.evaluate(async () => {
    const { fareScale, incomeScale, shownScale, storedScale } = await import('/src/game/income.ts');
    const { fareAt } = await import('/src/game/fareCurve.ts');
    const f = window.tw.moneyState.value.fares;
    const us = fareScale('US', f.scales);
    return {
      scales: f.scales, baseline: f.baseline,
      fares: [0, 10, 20, 50, 100].map((km) => [km, fareAt(f.nodes, km) * us]),
      income: { IN: incomeScale('IN'), US: incomeScale('US'), JP: incomeScale('JP'), CN: incomeScale('CN') },
      shown: { JP: shownScale(incomeScale('JP')), IN: shownScale(incomeScale('IN')), US: shownScale(incomeScale('US')) },
      kept: fareScale('XX', { XX: 2.8 }),
      typed: [0.17, 1, 1.25, 3.2].map((v) => shownScale(storedScale(v))),
    };
  });
  // New York's scale is the income's 11.6, where the old rule's unrounded one was 11.64: about 0.4% over the save's own $1.50 + $0.10 a km, as before this change
  const same = got.fares.every(([km, fare]) => Math.abs(fare / want(km) - 1) < 0.005);
  report('real save: New York fares per km match the save (within the income scale rounding)', same, got.fares.map(([km, v]) => `${km} km $${v.toFixed(3)} (want $${want(km).toFixed(3)})`).join('; '));
  report('real save: stored scales and baseline untouched', Object.keys(got.scales).length === 0 && got.baseline === 'IN', JSON.stringify([got.scales, got.baseline]));
  report('stored income scales unchanged (IN 1, US 11.6, JP 6.1, CN 3.1)', got.income.IN === 1 && got.income.US === 11.6 && got.income.JP === 6.1 && got.income.CN === 3.1, JSON.stringify(got.income));
  report('a saved tenth reads back the same', got.kept === 2.8, String(got.kept));
  report('shown scale: Japan 1.00, India about 0.16, the US 1.90', got.shown.JP === 1 && Math.abs(got.shown.IN - 0.164) < 0.001 && Math.abs(got.shown.US - 1.902) < 0.001, JSON.stringify(got.shown));
  report('a typed shown value reads back as typed', got.typed.every((v, i) => Math.abs(v - [0.17, 1, 1.25, 3.2][i]) < 0.006), JSON.stringify(got.typed));

  // the panel: the heading names the country in view and the stepper reads in Japan's terms
  await page.goto(`${BASE}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.click('button.tab[aria-label="Money"]');
  const row = () => page.$eval('ul.fare-rows > li', (li) => ({ name: li.querySelector('.fr-head')?.textContent, value: li.querySelector('.stepper .v')?.textContent }));
  await page.waitForSelector('ul.fare-rows > li');
  const home = await row();
  report('new game: heading names the country, the stepper reads in Japan terms', home.name === 'Fares in United States' && home.value === '1.90x', JSON.stringify(home));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [139.7, 35.69], zoom: 10 }));
  await page.waitForFunction(() => document.querySelector('ul.fare-rows > li .fr-head')?.textContent === 'Fares in Japan', undefined, { timeout: 30000 }).catch(() => {});
  const jp = await row();
  report('in Tokyo: "Fares in Japan" at 1.00x', jp.name === 'Fares in Japan' && jp.value === '1.00x', JSON.stringify(jp));
  await page.click('.fare-rows button[aria-label^="More"]');
  await page.waitForTimeout(500);
  const up = await row();
  report('one step up reads 1.05x', up.value === '1.05x', JSON.stringify(up));
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-1554 PASSED' : 'T-1554 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
