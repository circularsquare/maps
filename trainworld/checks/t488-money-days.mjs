// T-488: the money history's days and hours stay one entry each, in order, with fares and running
// costs on every whole day while trains run. Imports the real save into a ?new=1 game, runs the clock
// fast over a few days, and pulls the main thread's clock back across midnight (as drift between the
// two threads' clocks did, giving zero days out of order), then pauses, resumes and changes speed.
//   node trainworld/checks/t488-money-days.mjs [url]     (from the maps root; default localhost:6135)
// Prints PASS or FAIL and exits 1 on failure. About two minutes, most of it demand on the real save.
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6135/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t488-', size: [1200, 800], args: ANGLE });
let ok = false;
try {
  const page = await ctx.newPage();
  const wait = (ms) => page.waitForTimeout(ms);
  await page.goto(`${BASE}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  // (T-840: the blank game's final view still shows while the save's solve runs; the clock's hold
  // ends with the save's own final)
  await demandFinal(page, { strict: false, held: true });
  // the fastest speed made a game day about 4 wall seconds (the module is the app's own under Vite)
  await page.evaluate(async () => { const c = await import('/src/game/clock.ts'); c.SPEEDS[2] = 20000; });
  const run = (ms) => page.evaluate(() => window.tw.clock.setSpeed(2)).then(() => wait(ms));
  const pause = () => page.evaluate(() => window.tw.clock.setRunning(false)).then(() => wait(400));
  // run into day 3, then wait for 00:00-00:10 and pull the clock back 20 game minutes, twice
  for (let k = 0; k < 2; k++) {
    await run(3000);
    await page.waitForFunction(() => window.tw.clock.now() % 86400 < 600, undefined, { timeout: 30000, polling: 'raf' });
    await page.evaluate(() => { const c = window.tw.clock; c.set(c.now() - 1200); });
    await wait(300);
  }
  await pause();
  await run(2500);
  await page.evaluate(() => window.tw.clock.setSpeed(1));
  await wait(500);
  await run(4500);
  await pause();
  const m = await page.evaluate(() => { const m = window.tw.moneyState.value; return { days: m.ledger, hours: m.hours.map((h) => h.hour), now: window.tw.clock.now() }; });
  const days = m.days.map((d) => d.day);
  const inOrder = (a) => a.every((x, i) => !i || x > a[i - 1]);
  // the first day (loaded mid-day) and today are partial; every other day ran whole
  const whole = m.days.slice(1, -1);
  const empty = whole.filter((d) => !(d.fares > 0 && d.running > 0)).map((d) => d.day);
  console.log(`days ${days.join(' ')}; hours ${m.hours.length} from ${m.hours[0]} to ${m.hours.at(-1)}; clock day ${Math.floor(m.now / 86400)}`);
  for (const d of m.days) console.log(`  day ${d.day}: fares ${d.fares.toFixed(1)}, running ${d.running.toFixed(1)}`);
  ok = inOrder(days) && inOrder(m.hours) && empty.length === 0 && whole.length >= 2;
  console.log(`${ok ? 'PASS' : 'FAIL'}  money days in order ${inOrder(days)}, hours in order ${inOrder(m.hours)}, whole days ${whole.length}, empty whole days [${empty.join(', ')}]`);
} finally {
  await close();
}
process.exit(ok ? 0 : 1);
