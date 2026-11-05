// T-2444: the Money tab's history plots on the real save, with 30 days of figures drawn from the save's
// own day (scaled, with a build day), light and dark, cropped to artifacts/trainworld/profit2/money-<label>-<shade>.png.
//   node trainworld/checks/t2444-money-plot.mjs [url] [label]     (from the maps root; default localhost:6184, "after")
// Also hovers a column and prints the tag. Does not wait for demand.
import { mkdirSync, readFileSync } from 'node:fs';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6184/';
const LABEL = process.argv[3] ?? 'after';
const OUT = process.env.OUT_DIR ?? `${ARTIFACTS}profit2/`;
mkdirSync(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t2444-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.evaluate(async () => {
    const { moneyState, setMoneyState } = await import('/src/game/money.ts');
    const v = moneyState.value;
    const ledger = Array.from({ length: 30 }, (_, k) => {
      const wk = (k + 3) % 7 >= 5, f = 3000 * (wk ? 0.8 : 1) * (0.9 + 0.1 * Math.sin(k)) * Math.min(1, 0.4 + k / 12);
      return { day: k + 1, fares: f, running: 1300 + 20 * (k % 3), upkeep: 240, loan: 0, trains: k === 11 ? 1500 : k === 20 ? 400 : k % 5 === 0 ? 120 : 0, build: k === 11 ? 5200 : k % 4 === 0 ? 250 : 0, rides: 12e6 * (wk ? 0.8 : 1) };
    });
    setMoneyState({ ...v, ledger });
    (await import('/src/game/listScope.ts')).listScope.value = 'world';
    (await import('/src/game/ui.ts')).tab.value = 'money';
  });
  await page.waitForTimeout(1500);
  console.log('ledger days', await page.evaluate(async () => (await import('/src/game/money.ts')).moneyState.value.ledger.length), 'cols', await page.evaluate(() => document.querySelectorAll('rect.hist-col').length));
  await page.waitForSelector('svg.hist', { timeout: 30000 });
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.waitForTimeout(800);
    const box = await page.evaluate(() => {
      const a = document.querySelector('.hist-title'), b = document.querySelector('.hist-x');
      a.scrollIntoView({ block: 'center' });
      const r1 = a.getBoundingClientRect(), r2 = b.getBoundingClientRect();
      return { x: r1.left - 6, y: r1.top - 6, width: r1.width + 12, height: r2.bottom - r1.top + 12 };
    });
    await page.screenshot({ path: `${OUT}money-${LABEL}-${shade}.png`, clip: box });
  }
  const col = await page.evaluate(() => document.querySelectorAll('rect.hist-col')[11].getBoundingClientRect().toJSON());
  await page.mouse.move(col.left - 20, col.top + 20);
  await page.mouse.move(col.left + col.width / 2, col.top + 20);
  await page.waitForTimeout(400);
  console.log(await page.evaluate(() => [...document.querySelectorAll('.ui-tag')].find((t) => t.style.display !== 'none')?.innerText.replace(/\n/g, ' | ') ?? 'no tag'));
  console.log(errors.length ? `errors: ${errors.slice(0, 3).join(' | ')}` : 'no console errors');
} finally {
  await close();
}
