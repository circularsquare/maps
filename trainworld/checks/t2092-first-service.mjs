// T-2092, T-2123: a new game with the tutorial card opens on the Build tab; a new rail line's trains
// (or headway) set at one level stay at that level alone (T-2123 took back T-2092's fill of every
// level: the tutorial's last steps ask for trains at each level instead). A new game from the start
// menu in New York, a line built, then train counts set through the same functions the line panel's
// steppers call (game/world.ts). Screenshots in artifacts/trainworld/t2092/.
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
  check(JSON.stringify(t1) === '[4,0,0]', `4 trains at High leaves Medium and Low at 0 (${t1})`);
  await mod((w, id) => w.stepTrainCount(id, 'high', 1), A);
  await settle();
  const t2 = await trains(A);
  check(JSON.stringify(t2) === '[5,0,0]', `a step at High changes High alone (${t2})`);
  await mod((w, id) => w.setTrainCount(id, 'low', 2), A);
  await settle();
  const t3 = await trains(A);
  check(JSON.stringify(t3) === '[5,0,2]', `Low set on its own (${t3})`);

  // line B: a new line gets a headway first
  const b = await buildLine(page, [[-73.99, 40.72], [-73.995, 40.70], [-74.0, 40.69]], { name: 'B', wait: { route: 400, station: 300, settle: 700 } });
  check(!!b && b.ok !== false, `line B built (${JSON.stringify(b?.issues ?? [])})`);
  const B = String(b.line);
  await settle();
  await mod((w, id) => w.setAutoTrains(id, true), B);
  await settle();
  await mod((w, id) => w.setHeadway(id, 'high', 600), B);
  await settle();
  const t4 = await trains(B);
  const auto = await page.evaluate((id) => window.tw.world.value.lineStats[id]?.autoS, B);
  check(t4[0] > 0 && t4[1] === 0 && t4[2] === 0, `a headway at High on a new line starts trains at High alone (${t4}, headways ${auto})`);

  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
