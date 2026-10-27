// T-1523: the real save, NJCL's panel, hover on its High and Medium crowding dots, light and dark.
// usage: node shot.mjs <port>
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const TREE = 'C:/Users/anita/projects/maps/.tw-trees/t1523/trainworld';
const lib = await import(pathToFileURL(`${TREE}/checks/lib.mjs`).href);
const port = process.argv[2];
const BASE = `http://localhost:${port}/`;
const SCRATCH = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/ee78dc43-077b-4b8e-9cf9-ac5bb7367793/scratchpad/t1523/';
const OUT = 'C:/Users/anita/projects/maps/artifacts/trainworld/';

const { ctx, close } = await lib.launchChrome({ prefix: SCRATCH + 'prof-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  lib.collectErrors(page, errors);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.tw?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  await lib.importSave(page, await readFile(lib.REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await lib.demandFinal(page, { polling: 500, lines: 100, timeout: 900000 });
  await page.waitForTimeout(20000);
  const id = await page.evaluate(() => window.tw.world.value.save.lines.find((l) => l.num === 82).id);
  for (const shade of ['light', 'dark']) {
    await page.evaluate((shade) => { window.tw.shade.value = shade; window.tw.selection.value = null; }, shade);
    await page.waitForTimeout(1500);
    await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, id);
    await page.waitForSelector('.crowd-dot', { timeout: 60000 });
    await page.waitForTimeout(1500);
    const dots = page.locator('.crowd-dot');
    const tips = [];
    for (let k = 0; k < Math.min(2, await dots.count()); k++) {
      tips.push(await dots.nth(k).getAttribute('data-tip'));
      await dots.nth(k).hover();
      await page.waitForTimeout(800);
      await page.screenshot({ path: `${OUT}t1523-njcl-${['high', 'medium'][k]}-${shade}.png` });
      await page.mouse.move(5, 5);
      await page.waitForTimeout(300);
    }
    console.log(shade, JSON.stringify(tips, null, 1));
  }
  console.log('errors', errors.length, errors.slice(0, 5).join(' | '));
} finally {
  await close();
}
