// T-2684: years in the calendar. A new game's top bar has no year in year 1 (its hover has it); a
// forged two-year ledger makes the Money tab's all-time history label Year 1, Year 2 and Year 3 under
// its charts; a short ledger has none. Screenshots in artifacts/trainworld/yearsdest/.
//   node trainworld/checks/t2684-years.mjs [url]     (about 1 minute)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, launchChrome, collectErrors } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6161/';
const OUT = join(ARTIFACTS, 'yearsdest');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];

const chrome = await launchChrome({ prefix: 'tw-t2684-', size: [1400, 900], args: GL });
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors, NOISE);
  await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu') && window.tw.moneyState.value, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  const date = await page.textContent('.clock .date');
  check(/^\w{3} \w{3} \d+$/.test(date.trim()), `year 1's date has no year ("${date.trim()}")`);

  const forge = (days) => page.evaluate(async (days) => {
    const m = window.tw.moneyState.value;
    const ledger = [];
    for (let d = 1; d <= days; d++) ledger.push({ day: d, fares: 1 + (d % 30) / 10 + d / 400, running: 1.5 + (d % 7) / 10, upkeep: 0.4, trains: 0, build: d % 90 === 0 ? 3 : 0, rides: 20000 + d * 30 + (d % 13) * 500 });
    window.tw.moneyState.value = { ...m, ledger };
    (await import('/src/game/ui.ts')).tab.value = 'money';
  }, days);
  await forge(60);
  await page.waitForSelector('.hist', { timeout: 20000 });
  check(!(await page.$('.hist-years')), 'a 60-day history has no year labels');
  await forge(800);
  await page.waitForSelector('.hist-years', { timeout: 20000 });
  const marks = await page.$$eval('.hist-years span', (s) => s.map((x) => x.textContent));
  console.log('year marks', JSON.stringify(marks));
  check(marks.join() === 'Year 1,Year 2,Year 3', 'the 800-day history labels Year 1, Year 2, Year 3');
  await page.evaluate(() => document.querySelector('.hist-x').scrollIntoView({ block: 'end' }));
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(OUT, 'money-two-years.png'), clip: { x: 0, y: 0, width: 420, height: 900 } });

  // the top bar past year 1: day 400 is Feb 2 of year 2
  const cal = await page.evaluate(async () => { const d = await import('/src/game/days.ts'); return [d.yearOf(1), d.yearOf(339), d.yearOf(340), d.yearOf(704), d.yearOf(705), d.withYear(d.dateName(400), 400)]; });
  check(cal.join() === '1,1,2,2,3,Mon Mar 2, year 2', `the year turns on January 1 (${cal})`);
} finally {
  await chrome.close();
}
check(errors.length === 0, `no console errors (${errors.length})`);
process.exit(fail ? 1 : 0);
