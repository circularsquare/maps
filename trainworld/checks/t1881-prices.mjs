// T-1881: the track tool's tag over the named ghosts of T-1944 and T-1889 (Midtown at ground, the
// East River at +1) and over real hills that must stay dear (Nob Hill, the Peak), each in a new game
// in its zone. Prints each tag; screenshots of them to artifacts/trainworld/t1881/. No checks.
// Usage: node trainworld/checks/t1881-prices.mjs <label> [url] [ghost,ghost] (default http://localhost:6158/)

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const LABEL = process.argv[2] ?? 'now';
const BASE = process.argv[3] ?? 'http://localhost:6158/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1881');
mkdirSync(OUT, { recursive: true });
// name, zone, from, to, level key (Shift+4 ground, Shift+3 +1), zoom
const ALL = [
  ['6th-ave', 'nyc', [-73.9845, 40.7485], [-73.9845, 40.7645], 4, 14],
  ['45th-st', 'nyc', [-73.9693, 40.7518], [-73.9942, 40.7623], 4, 14],
  ['east-river-42nd', 'nyc', [-73.9700, 40.7490], [-73.9440, 40.7490], 3, 14],
  ['broadway-fidi', 'nyc', [-74.0137, 40.7049], [-74.0072, 40.7127], 4, 15],
  ['chicago-state-st', 'cic', [-87.6278, 41.8758], [-87.6278, 41.8881], 4, 14],
  ['nob-hill', 'snf', [-122.4030, 37.7929], [-122.4230, 37.7905], 4, 14],
  ['hk-peak', 'hon', [114.1590, 22.2795], [114.1495, 22.2712], 4, 14],
];
const only = process.argv[4]?.split(',');
const GHOSTS = only ? ALL.filter(([n]) => only.includes(n)) : ALL;

const { ctx, close } = await launchChrome({ prefix: 'tw-t1881p-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  let zone = null;
  const px = (x, y) => page.evaluate(([x, y]) => { const r = window.tw.map.getCanvas().getBoundingClientRect(), p = window.tw.map.project([x, y]); return [r.left + p.x, r.top + p.y]; }, [x, y]);
  const tagText = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag')].map((t) => t.textContent).join(' | '));
  for (const [name, z, a, b, key, zoom] of GHOSTS) {
    if (z !== zone) {
      await page.goto(`${BASE}${sep}debug=1&new=1&city=${z}&paused=1&demandWorkers=1&v=${Date.now()}`);
      await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
      zone = z;
    }
    await page.evaluate(([c, zoom]) => void window.tw.map.jumpTo({ center: c, zoom }), [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], zoom]);
    await page.waitForTimeout(4000);
    await page.keyboard.press('Escape');
    await page.keyboard.press('1');
    await page.keyboard.press(`Shift+Digit${key}`);
    await page.waitForTimeout(300);
    const [x0, y0] = await px(...a), [x1, y1] = await px(...b);
    await page.mouse.move(x0, y0);
    await page.mouse.click(x0, y0);
    await page.waitForTimeout(400);
    await page.mouse.move(x1, y1, { steps: 6 });
    await page.waitForTimeout(6000); // the terrain and road tiles land and the ghost is asked again
    await page.mouse.move(x1 + 1, y1, { steps: 1 });
    await page.waitForTimeout(1500);
    const tag = await tagText();
    await page.screenshot({ path: join(OUT, `tag-${name}-${LABEL}.png`), clip: { x: Math.max(0, Math.min(x0, x1) - 120), y: Math.max(0, Math.min(y0, y1) - 120), width: Math.min(1100, Math.abs(x1 - x0) + 480), height: Math.min(800, Math.abs(y1 - y0) + 300) } });
    await page.keyboard.press('Escape');
    await page.keyboard.press('Escape');
    console.log(`${name}: ${tag}`);
  }
  if (errors.length) console.log(`page errors: ${errors.slice(0, 3).join(' | ')}`);
} finally {
  await close();
}
