// T-2261: the diagram's graph does not depend on the camera's draw anchor. A save is opened, the diagram built; the
// camera then goes far away (the draw anchor moves) and back to another place (it moves again), and the diagram is
// built again: the graph's places must be the same bit for bit.
//   node trainworld/checks/t2261-graph-anchor.mjs [url]      (about 1 minute; needs a dev server)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, NOISE, launchChrome, collectErrors, importSave } from './lib.mjs';

const base = process.argv[2] ?? 'http://localhost:6146/';
const errors = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t2261-', size: [1400, 900] });
let failed = 0;
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {}
  });
  collectErrors(page, errors, NOISE);
  await page.goto(`${base}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  // T-3448: the graphs as the clock worker posts them (one per region), hashed; a reopen with the network unchanged lays
  // nothing out again, so the clock worker's posts are what is compared
  const last = () => page.evaluate(() => {
    const gs = window.twGraphs.graphs;
    let h = 0, n = 0;
    for (const g of gs) {
      n += g.key.length;
      for (const v of new Uint32Array(new Float64Array([...g.x, ...g.y]).buffer)) h = (Math.imul(h, 31) + v) >>> 0;
    }
    return { n, hash: h, origin: gs.map((g) => g.origin.join(' ')).join('; ') };
  });
  const built = async (n) => {
    await page.waitForFunction((n) => window.twGraphs?.n >= n && window.twDiagram?.layouts >= 1, n, { timeout: 300000, polling: 500 });
    await page.waitForTimeout(3000);
    return last();
  };
  await page.keyboard.press('m');
  const a = await built(1);
  await page.keyboard.press('m');
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-122.4, 37.8], zoom: 11 }));
  await page.waitForTimeout(4000);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.1, 41.1], zoom: 11 }));
  await page.waitForTimeout(4000);
  const before = await page.evaluate(() => window.twGraphs.n);
  await page.keyboard.press('m');
  await page.waitForFunction((n) => window.twGraphs.n > n, before, { timeout: 300000, polling: 500 });
  await page.waitForTimeout(3000);
  const b = await last();
  const same = a.hash === b.hash && a.n === b.n;
  if (!same) failed++;
  console.log(`${same ? 'PASS' : 'FAIL'} the graph's places are the same after the draw anchor moved: ${a.n} complexes, hash ${a.hash} then ${b.hash}, origin ${a.origin} then ${b.origin}`);
  if (errors.length) console.log('page errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
