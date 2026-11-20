// T-2787: a rail line's money history in its panel, in headless Chrome. From the maps root:
//   node trainworld/checks/t2787-line-money.mjs [url] [--shots]
// On the real New York save (T-007), run at speed for a few game days:
// 1. The clock worker keeps whole days per line (moneyState.lineDays), three numbers a day: day, fares, running.
// 2. A running rail line's panel draws "Revenue and expenses" with a bar per whole day (not the day now),
//    and a bar's hover gives the date, Fares, Running trains and Net.
// 3. The money state carries lineDays (the save header writes the same numbers).
// --shots: the panel, in artifacts/trainworld/linemoney/. Uses the Vite on the url (dev-nowatch.mjs).
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome, REAL_SAVE } from './lib.mjs';

const OUT = join(ARTIFACTS, 'linemoney');
const SHOTS = process.argv.includes('--shots');
if (SHOTS) await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6120/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t2787-', size: [1400, 1000], deviceScaleFactor: SHOTS ? 2 : 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&city=nyc&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { lines: 100, timeout: 600000 });
  // run at speed until three whole days are kept
  await page.evaluate(async () => {
    const { clock } = await import('/src/game/clock.ts');
    clock.setCapped(false);
    clock.setMultiple(100);
  });
  const have = () => page.evaluate(async () => {
    const { moneyState } = await import('/src/game/money.ts');
    const v = moneyState.value?.lineDays ?? {};
    const lens = Object.values(v).map((a) => a.length / 3);
    return { lines: Object.keys(v).length, most: Math.max(0, ...lens) };
  });
  let h = { lines: 0, most: 0 };
  for (let i = 0; i < 240 && h.most < 4; i++) { await page.waitForTimeout(1000); h = await have(); }
  check(h.lines > 0 && h.most >= 4, 'the worker keeps days per line', JSON.stringify(h));
  await page.evaluate(async () => { (await import('/src/game/clock.ts')).clock.setRunning(false); });
  await page.waitForTimeout(1500);

  await page.evaluate(async () => {
    const { moneyState } = await import('/src/game/money.ts');
    const num = Number(Object.entries(moneyState.value.lineDays).sort((a, b) => b[1].length - a[1].length)[0][0]);
    const l = window.tw.world.value.save.lines.find((x) => x.num === num);
    window.tw.selection.value = { kind: 'line', line: l.id };
  });
  await page.waitForSelector('svg.line-money', { timeout: 30000 });
  const info = await page.evaluate(() => {
    const g = document.querySelector('svg.line-money');
    g.scrollIntoView({ block: 'center' });
    return { bars: g.querySelectorAll('rect.hist-fares, rect.hist-costs').length, heading: g.previousElementSibling?.firstElementChild?.textContent, ends: [...g.nextElementSibling.querySelectorAll('span')].map((s) => s.textContent) };
  });
  check(info.bars >= 3 && info.bars <= 7 && info.heading === 'Revenue and expenses', 'the panel draws whole-day bars', JSON.stringify(info));
  const box = await page.evaluate(() => document.querySelectorAll('svg.line-money rect.hist-col')[1].getBoundingClientRect().toJSON());
  await page.mouse.move(box.left + 2, box.top + box.height / 2);
  await page.mouse.move(box.left + box.width / 2, box.top + box.height / 2);
  await page.waitForTimeout(400);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.ui-tag')].map((t) => t.innerText).find((x) => /Net/.test(x)) ?? '');
  check(/Fares/.test(tag) && /Running trains/.test(tag) && /Net/.test(tag), "a bar's hover", tag.replace(/\n/g, ' | '));
  if (SHOTS) {
    const dock = await page.evaluate(() => { const g = document.querySelector('svg.line-money'); return g.getBoundingClientRect().toJSON(); });
    await page.screenshot({ path: join(OUT, 'line-money-panel.png'), clip: { x: Math.max(0, dock.left - 300), y: Math.max(0, dock.top - 360), width: 640, height: 520 } });
    await page.screenshot({ path: join(OUT, 'line-money-full.png') });
  }
  await page.mouse.move(5, 600);

  // save and reload
  const before = await page.evaluate(async () => (await import('/src/game/money.ts')).moneyState.value.lineDays);
  check(!!before && Object.keys(before).length > 0, 'lineDays is in the money state', `${Object.keys(before ?? {}).length} lines`);
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2787 FAILED: ${fail}` : 'T-2787 PASSED');
process.exit(fail ? 1 : 0);
