// T-3225: every real network loads from the new game sheet and reaches playable. For each city: a fresh profile's
// new game sheet, Sandbox, the city picked in the sheet's Real world network list, Start; then seconds
// from Start to the lines drawn (`drawn`), the first demand final (`final`) and the clock running
// (`playable`), with the save's line and station counts and any page errors. From the maps root:
//   node trainworld/checks/t3225-real-load.mjs [url] [city ...]      (default: every city in the list)
// TW_CORES / TW_CORES_WANT as checks/cores.mjs. Writes artifacts/trainworld/t3225/load-<city>.json
// and a shot of the sheet (sheet.png) and of each city once playable (<city>.png).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't3225');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6136/';
const all = JSON.parse(await readFile(new URL('../app/src/game/realNetworks.json', import.meta.url), 'utf8'));
const cities = process.argv.length > 3 ? process.argv.slice(3) : all;
const SHOTS = process.env.SHOTS !== '0';
const rows = [];
for (const city of cities) {
  const chrome = await launchChrome({ prefix: 'tw-t3225-', size: [1400, 900] });
  const errors = [];
  const row = { city };
  let page;
  try {
    page = await chrome.ctx.newPage();
    collectErrors(page, errors);
    await page.goto(`${url}?debug=1&v=${Date.now()}`);
    // a fresh profile opens on the new game sheet (t2850-main-menu.mjs)
    await page.waitForSelector('.start-menu .new-game-mode', { timeout: 120000 });
    await page.click('.start-menu .new-game-mode [data-mode="sandbox"]');
    await page.selectOption('.start-menu .new-game-real select', city);
    if (SHOTS && city === cities[0]) await page.screenshot({ path: join(OUT, 'sheet.png') });
    const t0 = await page.evaluate(() => {
      window.__t = {};
      const t = performance.now();
      const mark = (k) => { if (!(k in window.__t)) window.__t[k] = Math.round(performance.now() - t); };
      const id = setInterval(() => {
        const w = window.tw?.world?.value;
        if ((w?.save?.lines?.length ?? 0) > 0) mark('drawn');
        const v = window.twDemand?.view?.();
        if (v?.final && window.__t.drawn !== undefined) mark('final');
        if (window.__t.final !== undefined && window.tw?.clock?.held?.peek() === false && !window.tw.clock.capped?.peek()) { mark('playable'); clearInterval(id); }
      }, 20);
      document.querySelector('.start-menu .new-game-start').click();
      return t;
    });
    await page.waitForFunction(() => window.__t?.playable !== undefined, undefined, { timeout: 900000, polling: 1000 });
    Object.assign(row, await page.evaluate(() => {
      const w = window.tw.world.value;
      return { ...Object.fromEntries(Object.entries(window.__t).map(([k, v]) => [k, +(v / 1000).toFixed(1)])),
        lines: w.save.lines.length, stations: w.save.stations.length, sandbox: window.tw.mode?.peek?.() ?? null };
    }));
    if (SHOTS) {
      await page.waitForTimeout(3000);
      await page.screenshot({ path: join(OUT, `${city}.png`) });
    }
    void t0;
  } catch (e) {
    row.failed = e.message.split('\n')[0];
    await page?.screenshot({ path: join(OUT, `failed-${city}.png`) }).catch(() => {});
  } finally {
    row.errors = errors.slice(0, 5);
    await chrome.close();
  }
  rows.push(row);
  console.log(JSON.stringify(row));
  await writeFile(join(OUT, `load-${city}.json`), JSON.stringify(row, null, 1));
}
const bad = rows.filter((r) => r.failed || r.errors.length);
console.log(`${rows.length - bad.length} of ${rows.length} loaded and reached playable with no errors${bad.length ? '; not: ' + bad.map((r) => r.city).join(' ') : ''}`);
process.exit(bad.length ? 1 : 0);
