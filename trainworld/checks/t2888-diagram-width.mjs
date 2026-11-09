// T-2888: the diagram pauses the game while it is up and gives the clock back as it was; its line width slider.
//   node trainworld/checks/t2888-diagram-width.mjs [url]     (about 1 minute; pictures in artifacts/trainworld/t2888/)
import { mkdirSync, readFileSync } from 'node:fs';
import { ARTIFACTS, REAL_SAVE, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const base = process.argv[2] ?? 'http://localhost:6125/';
const OUT = `${ARTIFACTS}t2888/`;
mkdirSync(OUT, { recursive: true });
const errors = [];
let fails = 0;
const check = (name, ok, extra = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${extra ? ' ' + extra : ''}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-diagwidth-', size: [1400, 900] });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE);
  await page.goto(`${base}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.waitForFunction(() => !window.tw.clock.held.value, undefined, { timeout: 300000 });
  await page.waitForTimeout(1500);

  const state = () => page.evaluate(() => { const c = window.tw.clock; return { running: c.running.value, rate: c.chosenRate.value, custom: c.custom.value, speed: c.speed.value, now: c.now(), shown: !!document.querySelector('.diagram-canvas')?.offsetParent }; });
  /** the game clock over a wall second */
  const moves = async () => { const a = (await state()).now; await page.waitForTimeout(1000); return (await state()).now - a; };
  const diagram = async (on) => {
    if (on) await page.click('[data-guide-tab="cities"]');
    const found = await page.evaluate((on) => { const b = document.querySelector(on ? '.map-diagram' : '.diagram-exit'); b?.click(); return !!b; }, on);
    if (!found) throw new Error('no diagram button ' + on);
    await page.waitForTimeout(400);
  };

  // 1. running at a slider speed: paused while up, the same speed after
  await page.evaluate(() => window.tw.clock.setMultiple(7));
  const before = await state();
  check('runs before', before.running && (await moves()) > 1, `rate ${before.rate}`);
  await diagram(true);
  const up = await state();
  check('diagram up pauses', up.shown && !up.running);
  check('clock frozen while up', Math.abs(await moves()) < 1e-6);
  await diagram(false);
  const after = await state();
  check('closing resumes at the same speed', !after.shown && after.running && after.custom === before.custom && after.rate === before.rate && (await moves()) > 1, `custom ${after.custom}, rate ${after.rate}`);

  // 2. paused already: stays paused through open and close
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await diagram(true);
  await diagram(false);
  const stay = await state();
  check('paused before stays paused', !stay.running && stay.custom === before.custom && Math.abs(await moves()) < 1e-6);

  // 3. the player starts the clock while it is up: it stays on after closing
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await diagram(true);
  await page.evaluate(() => window.tw.clock.setRunning(true));
  await diagram(false);
  check('started while up stays running', (await state()).running && (await moves()) > 1);

  // 4. the slider: the minimum, the middle and the largest; kept in the browser
  await diagram(true);
  await page.waitForFunction(() => !document.querySelector('.status-tag'), undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  const set = (v) => page.evaluate((v) => { const i = document.querySelector('.diagram-width input'); i.value = String(v); i.dispatchEvent(new Event('input', { bubbles: true })); return [i.min, i.max]; }, v);
  const [min, max] = (await set(1)).map(Number);
  check('slider range', min === 1 && max > 1.1 && max <= 1.4, `${min} to ${max}`);
  const canvas = page.locator('.diagram-canvas');
  const box = await canvas.boundingBox();
  const mid = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await page.mouse.move(mid.x, mid.y);
  await page.screenshot({ path: `${OUT}full_min.png` });
  for (let i = 0; i < 2; i++) { await page.mouse.wheel(0, -300); await page.waitForTimeout(150); }
  await page.waitForTimeout(500);
  const clip = { x: box.x + box.width / 2 - 450, y: box.y + box.height / 2 - 280, width: 900, height: 560 };
  await page.screenshot({ path: `${OUT}zoom_min.png`, clip });
  await set(max);
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}zoom_max.png`, clip });
  for (let i = 0; i < 2; i++) { await page.mouse.wheel(0, 300); await page.waitForTimeout(150); }
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}full_max.png` });
  const kept = await page.evaluate(() => localStorage.getItem('anitabuilder-diagram-width'));
  check('width kept', Number(kept) === max, kept);
  await diagram(false);
  check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
console.log(fails ? `${fails} FAILED` : 'ALL PASS');
process.exit(fails ? 1 : 0);
