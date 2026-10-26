// T-1300: screenshots of the Money tab's fare estimate on the real save, a drag of every fare to 1.5x
// under way, light and dark, with the riders row's hover; to artifacts/trainworld/t1300/.
//   node trainworld/checks/t1300-shots.mjs <url of a dev server>
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2];
if (!BASE) throw new Error('usage: t1300-shots.mjs <url>');
const OUT = join(ARTIFACTS, 't1300');
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1300s-' });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, dock: { dockW: 320, paneH: 820 }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 500, lines: 100 });
  await page.click('button.tab[aria-label="Money"]');
  await page.waitForTimeout(500);
  const none = await page.$$eval('ul.rows li', (ls) => ls.some((l) => /Riders a day/.test(l.textContent)));
  console.log(`no estimate rows before a drag: ${!none}`);
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.evaluate(async () => {
      const { farePreview } = await import('/src/ui/FareGraph.tsx');
      const m = window.tw.moneyState.value;
      const s = 2.8;
      farePreview.value = m.fares.nodes.map(([k, f]) => [k, f * s * 1.5]);
    });
    await page.waitForTimeout(400);
    const g = await page.$('svg.fare-graph');
    await g.evaluate((el) => el.scrollIntoView({ block: 'start' }));
    await page.waitForTimeout(200);
    const box = await g.boundingBox();
    const rows = await page.$$('ul.rows');
    let bottom = box.y + box.height;
    for (const r of rows) {
      const t = await r.textContent();
      if (/Riders a day/.test(t)) { const b = await r.boundingBox(); bottom = b.y + b.height; }
    }
    const clip = { x: box.x - 14, y: box.y - 30, width: box.width + 28, height: bottom - box.y + 40 };
    await page.screenshot({ path: join(OUT, `estimate-${shade}.png`), clip });
    const val = await page.$('li:has-text("Riders a day") .val span');
    await val.hover();
    await page.waitForTimeout(700);
    const vp = page.viewportSize();
    const hx = Math.max(0, clip.x - 260);
    await page.screenshot({ path: join(OUT, `estimate-hover-${shade}.png`), clip: { x: hx, y: clip.y, width: Math.min(vp.width - hx, clip.width + 520), height: clip.height + 90 } });
    await page.mouse.move(5, 5);
    const text = await page.$$eval('ul.rows li', (ls) => ls.filter((l) => /a day/.test(l.textContent)).map((l) => l.textContent));
    console.log(shade, JSON.stringify(text));
  }
} finally {
  await close();
}
