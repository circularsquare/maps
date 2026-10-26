// T-1286: the regional market weighs each trip's fare at its origin's fare scale over its income's.
// The real save's weekday final at the income's scales, then with one country's scale set to
// FACTOR times its income's: each zone's local rail trips, the market's rail trips (longer trips
// and commute legs) on corridors touching that country and on the rest, and the network's.
//   node trainworld/checks/t1286-market-scale.mjs <url> [country US] [factor 2]
// Writes artifacts/trainworld/t1286/<host-port>-<country>-<factor>.json; run it on a build without
// T-1286 too and compare: local figures should match, the market's differ only at the country set.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, importSave, demandFinal, marketRows, loadApp } from './lib.mjs';

const [BASE, COUNTRY = 'US', FACTOR = '2'] = process.argv.slice(2);
if (!BASE) throw new Error('usage: t1286-market-scale.mjs <url> [country] [factor]');
const { incomeScale } = await loadApp("export { incomeScale } from '@app/src/game/income.ts';");
const rows = JSON.parse(marketRows());
const countryOf = [];
for (const r of Object.values(rows)) countryOf[r.place] = r.country;

const { ctx, close } = await launchChrome({ prefix: 'tw-t1286-' });
try {
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`console ${m.type()}: ${m.text().slice(0, 300)}`); });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  const sep = BASE.includes('?') ? '&' : '?';
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 500, lines: 100 });

  const figures = (country) => page.evaluate(([country, countryOf]) => {
    const d = window.twDemand.days()[0];
    const local = {};
    for (const [id, v] of d.cities) local[id] = v.railTripsPerDay;
    const c = d.intercity?.corridors;
    let mine = 0, rest = 0;
    if (c) {
      for (let k = 0; k < c.railAt.length; k++) {
        const j = c.railAt[k], r = c.longerRail[k] + c.commuteRail[k];
        if (countryOf[c.table.from[j]] === country || countryOf[c.table.to[j]] === country) mine += r; else rest += r;
      }
    }
    return { version: d.view.version, rail: d.view.railTripsPerDay, intercityRail: d.intercity?.railTripsPerDay ?? 0, corridorRail: { [country]: mine, rest }, local };
  }, [country, countryOf]);

  const before = await figures(COUNTRY);
  const income = await page.evaluate((c) => window.tw.moneyState.value.fares.scales?.[c] ?? null, COUNTRY);
  if (income !== null) throw new Error(`the save sets ${COUNTRY}'s scale already (${income})`);
  // the income's scale, as the Money tab shows it
  const inc = incomeScale(COUNTRY);
  const scale = { inc, s: Math.round(inc * Number(FACTOR) * 10) / 10 };
  await page.evaluate(([c, s]) => window.tw.client.edit({ op: 'fareScale', country: c, scale: s }), [COUNTRY, scale.s]);
  await demandFinal(page, { polling: 500, after: before.version });
  const after = await figures(COUNTRY);

  const out = { country: COUNTRY, incomeScale: scale.inc, scale: scale.s, before, after };
  const dir = join(ARTIFACTS, 't1286');
  await mkdir(dir, { recursive: true });
  const file = join(dir, `${new URL(BASE).host.replace(':', '-')}-${COUNTRY}-${FACTOR}.json`);
  await writeFile(file, JSON.stringify(out, null, 1) + '\n');
  const pct = (a, b) => (b ? (100 * (a - b) / b).toFixed(2) + '%' : 'n/a');
  const localBefore = Object.values(before.local).reduce((s, v) => s + v, 0), localAfter = Object.values(after.local).reduce((s, v) => s + v, 0);
  console.log(`${COUNTRY} scale ${scale.inc} -> ${scale.s}: network ${Math.round(before.rail)} -> ${Math.round(after.rail)} (${pct(after.rail, before.rail)}); local ${Math.round(localBefore)} -> ${Math.round(localAfter)} (${pct(localAfter, localBefore)}); market ${Math.round(before.intercityRail)} -> ${Math.round(after.intercityRail)} (${pct(after.intercityRail, before.intercityRail)}); corridors touching ${COUNTRY} ${Math.round(before.corridorRail[COUNTRY])} -> ${Math.round(after.corridorRail[COUNTRY])}, the rest ${Math.round(before.corridorRail.rest)} -> ${Math.round(after.corridorRail.rest)}`);
  console.log(file);
} finally {
  await close();
}
