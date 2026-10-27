// T-1526: every click on a line panel stepper counts. Five quick clicks on "More train length" (all five
// in one task, then five real mouse clicks the other way), "More trains on the line" and "More headway"
// each move the value by five once the worker has answered. Before the fix the train length stepper
// counted from the length of the last render, so quick clicks dropped.
//   node trainworld/checks/t1526-steppers.mjs [url]     (from the maps root; a running dev server)
import { NOISE, acrossLine, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6143/';
const { ctx, close } = await launchChrome({ prefix: 'tw-t1526-', size: [1400, 900] });
const errors = [];
let fails = 0;
const check = (name, ok, detail = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, NOISE, /\/fonts\/[^/]+\/\d+-\d+\.pbf$/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.40, 40.64], zoom: 14 }));
  await page.waitForTimeout(1500);
  const line = await acrossLine(page, [-74.40, 40.64], 2);
  check('a line built', line !== null);
  if (line === null) throw new Error('no line');
  const id = String(line);
  const stats = () => page.evaluate((id) => { const s = window.tw.world.value.lineStats[id]; return { cars: s.cars, trains: s.trainsNeeded.slice(), autoS: s.autoS ? s.autoS.slice() : null }; }, id);
  /** waits until no stepper edit is waiting or in flight and the world has answered */
  const settle = async () => {
    await page.waitForFunction(async () => {
      const w = await import('/src/game/world.ts');
      return w.pendingCars.value.size === 0 && w.pendingTrains.value.size === 0 && w.pendingHeadway.value.size === 0;
    }, undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };

  // two cars to start from; the platforms (200 m) fit more than seven
  await page.evaluate(async ([id, line]) => { const w = await import('/src/game/world.ts'); await w.edit({ op: 'lineCars', line, cars: 2 }, true); }, [id, line]);
  await page.evaluate(async (id) => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; ui.selection.value = { kind: 'line', line: id }; }, id);
  await page.waitForSelector('button[aria-label="More train length"]');
  await page.waitForTimeout(800);
  const s0 = await stats();
  check('train length starts at 2', s0.cars === 2, JSON.stringify(s0));

  const more = 'button[aria-label="More train length"]', less = 'button[aria-label="Less train length"]';
  await page.evaluate((sel) => { const b = document.querySelector(sel); for (let i = 0; i < 5; i++) b.click(); }, more);
  await settle();
  const s1 = await stats();
  check('five clicks together on More train length: +5', s1.cars === 7, `cars ${s1.cars}`);
  for (let i = 0; i < 5; i++) await page.click(less, { delay: 0 });
  await settle();
  const s2 = await stats();
  check('five quick mouse clicks on Less train length: -5', s2.cars === 2, `cars ${s2.cars}`);

  // the count and headway steppers fold their clicks the same way
  const moreTrains = 'button[aria-label="More trains on the line"]';
  await page.evaluate((sel) => { const b = document.querySelector(sel); for (let i = 0; i < 5; i++) b.click(); }, moreTrains);
  await settle();
  const s3 = await stats();
  const sum = (a) => a.reduce((x, y) => x + y, 0);
  check('five clicks together on More trains: +5', sum(s3.trains) - sum(s2.trains) === 5, `${s2.trains} -> ${s3.trains}`);

  await page.evaluate(async (id) => { const w = await import('/src/game/world.ts'); w.setAutoTrains(id, true); }, id);
  await settle();
  await page.waitForSelector('button[aria-label="More headway"]');
  const h0 = (await stats()).autoS;
  await page.evaluate((sel) => { const b = document.querySelector(sel); for (let i = 0; i < 5; i++) b.click(); }, 'button[aria-label="More headway"]');
  await settle();
  const h1 = (await stats()).autoS;
  const want = await page.evaluate(async (h0) => {
    const w = await import('/src/game/world.ts');
    // the first row's headway, stepped five times from where it began
    let s = h0[0];
    for (let i = 0; i < 5; i++) s = w.nextHeadway(s, 1);
    return s;
  }, h0);
  check('five clicks together on More headway: five steps', !!h0 && !!h1 && h1[0] === want, `${h0?.[0]} -> ${h1?.[0]}, wanted ${want}`);
  check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(fails ? 1 : 0);
