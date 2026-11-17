// T-3567: a line's own fare multiple, set through its panel. On the real save a busy rail line's panel shows a
// Fare multiple row at 1.00x, right above the revenue and expenses chart (T-3566, the panel's last section).
// Typed 1.2 it reads 1.20x and, once demand's final is in, the line's fare a ride is higher and its riders fewer;
// 5 is held to 3.00x, 0 reads 0.00x with the fare a ride below the start's; the undo button puts it back to
// 1.00x and every figure is as it started (rail trips, each line's riders and revenue, to the bit). At 1.20x
// again the game is exported and opened again: the row still reads 1.20x and the save's fares carry it. Shots of
// the panel's bottom, light and dark, in artifacts/trainworld/. From the maps root (or a tree's root):
//   node trainworld/checks/t3567-line-fare.mjs [url] [line number]   (about six minutes; default localhost:6144, line 14)
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, importSave } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6144/';
const NUM = Number(process.argv[3] ?? 14);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const g = await openGame({ url: BASE, save: REAL_SAVE, saveVia: 'post', prefix: 'tw-t3567-', ui: { shade: 'light', v: 15 } });
const page = g.page;
/** the network's and line NUM's figures: rail trips a day, each line's riders and fare revenue a day (US$M) */
const figures = () => page.evaluate(async (num) => {
  const money = await import('/src/game/money.ts'), periods = await import('/src/game/periods.ts');
  const hours = periods.PERIODS.map((p) => p.to - p.from);
  const v = window.twDemand.view(), w = window.tw.world.value, inc = money.incomeByDay.value[0];
  const day = (m) => (m ? m.perHour.reduce((a, x, p) => a + x * hours[p], 0) : null);
  const lines = {};
  for (const l of w.save.lines) lines[l.num] = [v.lines.get(l.id)?.ridersPerDay ?? null, day(inc?.lines.get(l.id))];
  const [riders, revenue] = lines[num];
  return { version: v.version, rail: v.railTripsPerDay, lines, riders, revenue, perRide: revenue * 1e6 / 100 / riders, own: money.moneyState.value?.fares?.lines ?? null };
}, NUM);
const row = async () => (await g.see.panel())?.rows['Fare multiple'] ?? null;
/** type into the row's value, then wait for demand's final on the new fares */
const typeFare = async (text) => {
  const v = await page.evaluate(() => window.twDemand.view().version);
  await g.act.type('Type fare multiple', text, { within: '#inspector' });
  await g.waitDemand({ after: v });
  await page.waitForTimeout(2000);
};
/** the panel scrolled to its bottom (the fare row and the chart), shot */
const shot = async (file) => {
  await page.evaluate(() => { const el = document.querySelector('#inspector svg.line-money')?.nextElementSibling; el?.scrollIntoView({ block: 'end' }); });
  await page.waitForTimeout(500);
  return g.shot(file, { element: '#inspector' });
};
const fmt = (f) => `${Math.round(f.riders).toLocaleString('en-US')} riders, $${f.revenue.toFixed(2)}M a day, $${f.perRide.toFixed(3)} a ride`;
try {
  await g.waitDemand({ lines: 50 });
  // the save's line days are older than its clock: a few days run on (the paused clock moved, which settles the
  // hours on the way) give the chart some
  const days = async () => page.evaluate((num) => (window.tw.moneyState.value?.lineDays?.[String(num)]?.length ?? 0) / 3, NUM);
  if ((await days()) < 4) {
    const t0 = await page.evaluate(() => window.tw.clock.now());
    for (let k = 1; k <= 5; k++) { await page.evaluate((t) => window.tw.clock.set(t), t0 + k * 86400); await page.waitForTimeout(2500); }
    await page.waitForFunction((num) => (window.tw.moneyState.value?.lineDays?.[String(num)]?.length ?? 0) >= 12, NUM, { timeout: 60000, polling: 500 }).catch(() => {});
    await g.waitDemand({ lines: 50 });
  }
  console.log(`  line ${NUM} has ${await days()} days kept`);
  await page.waitForTimeout(2000);
  const name = await page.evaluate((num) => window.tw.world.value.save.lines.find((l) => l.num === num)?.name, NUM);
  await g.act.openLine(name);
  check((await row()) === '1.00x', `line ${NUM} (${name}): the row reads ${await row()}`);
  // the row, then the chart's heading, the chart and its dates, and nothing after
  const order = await page.evaluate(() => {
    const ul = [...document.querySelectorAll('#inspector ul.rows')].find((u) => u.textContent.includes('Fare multiple'));
    const out = [];
    for (let e = ul; e; e = e.nextElementSibling) out.push(e.tagName.toLowerCase() + '.' + [...e.classList].join('.'));
    return out.join(' ');
  });
  check(order === 'ul.rows div.h svg.hist.line-money div.hist-x.small.muted', `the fare row sits right above the chart, at the panel's bottom (${order})`);
  const f0 = await figures();
  console.log(`  at 1.00x: ${fmt(f0)}`);

  await typeFare('1.2');
  const f12 = await figures();
  check((await row()) === '1.20x' && f12.own?.[NUM] === 1.2, `typed 1.2: the row reads ${await row()}, saved fares ${JSON.stringify(f12.own)}`);
  check(f12.perRide > f0.perRide && f12.riders < f0.riders, `at 1.20x: ${fmt(f12)}`);
  await shot('t3567_line_light.png');

  await typeFare('5');
  const f3 = await figures();
  check((await row()) === '3.00x' && f3.own?.[NUM] === 3, `typed 5: held to ${await row()}`);
  check(f3.perRide > f12.perRide && f3.riders < f12.riders, `at 3.00x: ${fmt(f3)}`);

  await typeFare('0');
  const fz = await figures();
  check((await row()) === '0.00x', `typed 0: the row reads ${await row()}`);
  check(fz.perRide < f0.perRide && fz.riders > f0.riders && fz.revenue > 0, `at 0.00x (the base is still paid): ${fmt(fz)}`);

  const v = await page.evaluate(() => window.twDemand.view().version);
  await g.act.click('Back to 1.00x', { within: '#inspector' });
  await g.waitDemand({ after: v });
  await page.waitForTimeout(2000);
  const f1 = await figures();
  check((await row()) === '1.00x' && f1.own === null, `the undo button: back to ${await row()}, no multiples kept (${JSON.stringify(f1.own)})`);
  check(f1.rail === f0.rail && JSON.stringify(f1.lines) === JSON.stringify(f0.lines), `back at 1.00x every figure is as it started (rail ${f1.rail} vs ${f0.rail})`);

  // save and open again
  await typeFare('1.2');
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  const saved = await page.evaluate(async (data) => { const s = await import('/src/workers/saveFile.ts'); return (await s.decodeSave(Uint8Array.from(data))).header.fares.lines ?? null; }, Array.from(bytes));
  check(saved?.[NUM] === 1.2, `the exported save's fares carry it: ${JSON.stringify(saved)}`);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await g.waitDemand({ lines: 50 });
  await page.waitForTimeout(2000);
  await g.act.openLine(name);
  const back = await figures();
  check((await row()) === '1.20x' && back.own?.[NUM] === 1.2, `opened again: the row reads ${await row()}`);
  check(Math.abs(back.revenue - f12.revenue) < 1e-6 * f12.revenue, `opened again: ${fmt(back)}`);

  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(600);
  await shot('t3567_line_dark.png');
  const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
  check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
  console.log(`  shots: ${join(ARTIFACTS, 't3567_line_light.png')}, t3567_line_dark.png`);
} catch (e) {
  check(false, `ran to the end: ${e.stack ?? e.message}`);
} finally {
  await g.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
