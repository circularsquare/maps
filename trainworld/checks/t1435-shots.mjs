// T-1435: where people drive. The real save (New York) and a new game in London with Drives on:
// screenshots light and dark zoomed in, at the city's scale and zoomed out into
// artifacts/trainworld/t1435/, a hover on the biggest arc, the workers' timings for the drive
// queries, and no console errors. Uses the Vite already on `port` (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1435-shots.mjs <port>
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, demandFinal, acrossLine } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1435');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6143);
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

/** drawn pixels on the arcs' canvas (every 4th pixel each way), and the most opaque one, CSS px */
const arcPixels = (page) => page.evaluate(() => {
  const c = document.querySelector('#map canvas.overlay.drives');
  if (!c || !c.width) return { n: 0 };
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data, k = c.width / c.clientWidth;
  let n = 0, best = null, ba = 0;
  for (let y = 0; y < c.height; y += 4) for (let x = 0; x < c.width; x += 4) {
    const a = d[(y * c.width + x) * 4 + 3];
    if (a > 0) n++;
    if (a > ba) { ba = a; best = [x / k, y / k]; }
  }
  return { n, best };
});
const waitArcs = async (page, label, timeout = 120000) => {
  const t = Date.now();
  while (Date.now() - t < timeout) {
    const p = await arcPixels(page);
    if (p.n > 50) return { ...p, s: (Date.now() - t) / 1000 };
    await page.waitForTimeout(500);
  }
  return { n: 0, s: timeout / 1000 };
};
const timings = (page) => page.evaluate(() => (window.twDemand?.client?.qlog ?? []).filter((q) => q.kind.startsWith('drive')).map((q) => `${q.kind} ${q.replies.map((r) => `w${r.w} ${Math.round(r.ms)} ms`).join(', ')}`));

async function views(page, label, spots) {
  const map = await page.$('#map');
  const box = await map.boundingBox();
  for (const [name, center, zoom] of spots) {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [center, zoom]);
    await page.waitForTimeout(1500);
    const p = await waitArcs(page, label, 60000);
    check(p.n > 50, `${label} ${name}: arcs drawn (${p.n} sampled pixels, ${p.s.toFixed(1)} s)`);
    for (const s of ['light', 'dark']) {
      await page.evaluate((s) => { window.tw.shade.value = s; }, s);
      await page.waitForTimeout(600);
      await page.mouse.move(box.x + 2, box.y + 2);
      await page.screenshot({ path: join(OUT, `${label}-${name}-${s}.png`) });
    }
    await page.evaluate(() => { window.tw.shade.value = 'light'; });
    await page.waitForTimeout(400);
    const q = await arcPixels(page);
    if (q.best) {
      await page.mouse.move(box.x + q.best[0], box.y + q.best[1], { steps: 3 });
      await page.waitForTimeout(700);
      const tag = await page.evaluate(() => { const t = document.querySelector('.demand-tag'); return t && t.style.display !== 'none' ? t.innerText : ''; });
      check(/drives .* between .* and /.test(tag) && /faster by train/.test(tag), `${label} ${name}: hover (${tag.replace(/\n/g, ' | ')})`);
      await page.screenshot({ path: join(OUT, `${label}-${name}-hover.png`) });
    }
  }
}

async function session(label, run) {
  const chrome = await launchChrome({ prefix: 'tw-t1435-', size: [1400, 900] });
  const errors = [];
  try {
    const page = await chrome.ctx.newPage();
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', drives: true, commuters: true, commuterEnd: 'home' }, v: 10 })); } catch {} });
    collectErrors(page, errors);
    await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await run(page);
    console.log(label, 'timings', JSON.stringify(await timings(page)));
    check(errors.length === 0, `${label}: no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  } finally {
    await chrome.close();
  }
}

await session('nyc', async (page) => {
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250, lines: 100 });
  await views(page, 'nyc', [['z10', [-73.93, 40.74], 10], ['z12', [-73.97, 40.75], 12], ['z7', [-74.6, 40.6], 7]]);
});
await session('london', async (page) => {
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-0.12, 51.5], zoom: 10 }));
  await page.waitForTimeout(3000);
  await views(page, 'london', [['z10', [-0.12, 51.5], 10], ['z11', [-0.1, 51.51], 11.5]]);
  // zoomed out the corridors need the market's result, which comes with the first station
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-0.12, 51.51], zoom: 12 }));
  check((await acrossLine(page, [-0.12, 51.51], 3)) !== null, 'london: a line built');
  await demandFinal(page, { polling: 250, lines: 0 });
  await views(page, 'london-line', [['z10', [-0.12, 51.5], 10], ['z7', [-1.2, 52.2], 7]]);
});
process.exit(fail ? 1 : 0);
