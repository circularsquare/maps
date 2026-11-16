// T-3491: the zone outlines at a few zooms on desktop, a screenshot each, with the page's and MapLibre's worker's
// heap after a collection there, so a change to how outlines are held can be compared by eye and by MB.
//   TW_CORES=4 LABEL=after [QS='&opened=all'] node trainworld/checks/t3491-outlines.mjs http://127.0.0.1:6178/
// Writes ../artifacts/trainworld/t3491/LABEL-<view>.png and LABEL.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, launchChrome, collectErrors } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://127.0.0.1:6178/';
// QS='&opened=all' shows every realm, so the world's outlines draw
const LABEL = process.env.LABEL ?? 'run', QS = process.env.QS ?? '';
const OUT = join(ARTIFACTS, 't3491');
const MB = 1048576;
const VIEWS = [['world', [10, 30], 2.2], ['europe', [10, 50], 4.5], ['germany', [9, 50.5], 7], ['nyc', [-73.95, 40.75], 9]];

await mkdir(OUT, { recursive: true });
const chrome = await launchChrome({ prefix: 'tw-t3491-', args: GL, size: [1400, 900] });
const errors = [], rows = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&demandWorkers=1${QS}&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .new-game', { timeout: 180000 });
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 240000 });
  await page.waitForTimeout(4000);
  // the guide's panel covers part of the map: closed if it offers a close
  await page.evaluate(() => document.querySelector('.guide .close, .guide button[aria-label="Close"]')?.click());
  const cdp = await chrome.ctx.newCDPSession(page);
  for (const [name, c, z] of VIEWS) {
    await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); }, [c, z]);
    await page.waitForTimeout(7000);
    await page.screenshot({ path: join(OUT, `${LABEL}-${name}.png`) });
    await cdp.send('HeapProfiler.collectGarbage');
    const h = await cdp.send('Runtime.getHeapUsage');
    const src = await page.evaluate(() => { const s = window.tw.map.getSource('tw-city-boundaries'); const d = s?._data; const f = d?.updateable ? [...d.updateable.values()] : d?.geojson?.features ?? []; let pts = 0; for (const x of f) for (const l of x.geometry.coordinates) pts += l.length; return { features: f.length, points: pts }; });
    rows.push({ view: name, pageUsedMB: Math.round(h.usedSize / MB), pageBuffersMB: Math.round((h.backingStorageSize ?? 0) / MB), ...src });
    console.log(name, JSON.stringify(rows.at(-1)));
  }
} catch (e) {
  console.log('FAILED', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await writeFile(join(OUT, `${LABEL}.json`), JSON.stringify({ label: LABEL, rows, errors }, null, 1));
  console.log('errors', errors.length, errors.slice(0, 3).join(' | '));
  await chrome.close();
}
