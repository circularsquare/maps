// T-2092: a new game with the tutorial card opens on the Build tab; a new rail line's first trains
// (or first headway) at one level go to every level still at 0, after which levels are independent.
// A new game from the start menu in New York, a line built, then train counts set through the same
// functions the line panel's steppers call (game/world.ts). Screenshots in artifacts/trainworld/t2092/.
//   node trainworld/checks/t2092-first-service.mjs [url]     (about 2 minutes)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, NOISE, launchChrome, collectErrors, buildLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6161/';
const OUT = join(ARTIFACTS, 't2092');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t2092-', size: [1400, 900], args: GL });
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors, NOISE);
  await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 }).catch(async (e) => { await page.screenshot({ path: join(OUT, 'no-menu.png') }); throw e; });
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  const tabNow = () => page.evaluate(async () => (await import('/src/game/ui.ts')).tab.value);
  check(!!(await page.$('.guide')), 'the tutorial card shows');
  check((await tabNow()) === 'build', `a new game with the card opens on the Build tab (${await tabNow()})`);
  await page.screenshot({ path: join(OUT, 'after-game-start.png') });

  const mod = (fn, arg) => page.evaluate(async ([src, arg]) => (0, eval)(src)(await import('/src/game/world.ts'), arg), [fn.toString(), arg]);
  const trains = (id) => page.evaluate((id) => window.tw.world.value.lineStats[id]?.trainsNeeded, id);
  const settle = (ms = 1200) => page.waitForTimeout(ms);
  await page.evaluate(() => window.tw.clock.setRunning(false));

  // line A: trains by count
  const a = await buildLine(page, [[-73.985, 40.735], [-73.98, 40.755], [-73.97, 40.775]], { name: 'A', wait: { route: 400, station: 300, settle: 700 } });
  check(!!a && a.ok !== false, `line A built (${JSON.stringify(a?.issues ?? [])})`);
  const A = String(a.line);
  await settle();
  check(JSON.stringify(await trains(A)) === '[0,0,0]', `a new line has no trains (${await trains(A)})`);
  await mod((w, id) => w.setTrainCount(id, 'high', 4), A);
  await settle();
  const t1 = await trains(A);
  check(JSON.stringify(t1) === '[4,4,4]', `4 trains at High gives 4 at every level (${t1})`);

  // it runs at 21:00 on a Saturday (the screenshot): low is in force then, and has trains
  await page.evaluate(() => { const c = window.tw.clock; const day = Math.floor(c.now() / 86400); const toSat = (6 - (day % 7) + 7) % 7; c.set((day + toSat) * 86400 + 21 * 3600); c.setRunning(true); });
  await settle(6000);
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await page.evaluate(() => { document.querySelector('[data-guide-tab="lines"]').click(); });
  await settle(500);
  await page.click('#dock .pane .rows li').catch(() => {});
  await settle(1500);
  await page.screenshot({ path: join(OUT, 'after-line-A.png') });

  // levels are independent after that
  await mod((w, id) => w.setTrainCount(id, 'low', 0), A);
  await settle();
  check(JSON.stringify(await trains(A)) === '[4,4,0]', `Low set to 0 stays 0 (${await trains(A)})`);
  await mod((w, id) => w.stepTrainCount(id, 'high', 1), A);
  await settle();
  const t2 = await trains(A);
  check(JSON.stringify(t2) === '[5,4,0]', `an existing line with 0 at Low keeps 0 when High changes (${t2})`);

  // line B: a new line gets a headway first
  const b = await buildLine(page, [[-73.99, 40.72], [-73.995, 40.70], [-74.0, 40.69]], { name: 'B', wait: { route: 400, station: 300, settle: 700 } });
  check(!!b && b.ok !== false, `line B built (${JSON.stringify(b?.issues ?? [])})`);
  const B = String(b.line);
  await settle();
  await mod((w, id) => w.setAutoTrains(id, true), B);
  await settle();
  await mod((w, id) => w.setHeadway(id, 'high', 600), B);
  await settle();
  const t3 = await trains(B);
  const auto = await page.evaluate((id) => window.tw.world.value.lineStats[id]?.autoS, B);
  check(t3.every((n) => n > 0), `a headway at High on a new line starts trains at every level (${t3}, headways ${auto})`);
  check(auto && auto.every((s) => s === 600), `the headway is the same at every level (${auto})`);

  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
