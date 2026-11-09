// T-1854: the terrain tint over a valley, for a before/after pair of the class grids. Wuppertal's
// Wupper valley (about 1 km wide, the Schwebebahn along its floor) and a side valley of the Inn above
// Innsbruck, and the dense centres of Tokyo, Nagoya and Shanghai (which must stay flat), with the
// track tool on and Terrain shown. Screenshots only, no checks.
// Usage: node trainworld/checks/t1854-valley.mjs <label> [url] [spot,spot] (default http://localhost:6185/, every spot)

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from '../lib.mjs';

const LABEL = process.argv[2] ?? 'now';
const BASE = process.argv[3] ?? 'http://localhost:6185/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1854');
mkdirSync(OUT, { recursive: true });
const ALL = [['wuppertal', 7.15, 51.258, 12.6], ['innsbruck', 11.43, 47.29, 12.2],
  ['tokyo', 139.755, 35.68, 12.2], ['nagoya', 136.91, 35.17, 12.2], ['shanghai', 121.47, 31.23, 12.2]];
const only = process.argv[4]?.split(',');
const SPOTS = only ? ALL.filter(([n]) => only.includes(n)) : ALL;

const { ctx, close } = await launchChrome({ prefix: 'tw-t1854-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.addInitScript(() => {
    try {
      const s = JSON.parse(localStorage.getItem('trainworld-ui') || '{}');
      s.display = { ...(s.display ?? {}), hillshade: true };
      s.v = s.v ?? 10;
      localStorage.setItem('trainworld-ui', JSON.stringify(s));
    } catch { /* private window */ }
  });
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  for (const [name, lng, lat, zoom] of SPOTS) {
    await page.evaluate(([lng, lat, zoom]) => { window.tw.shade.value = 'light'; window.tw.map.jumpTo({ center: [lng, lat], zoom }); }, [lng, lat, zoom]);
    await page.waitForTimeout(1500);
    await page.mouse.click(22, 16); // the Build tab, then the track tool
    await page.waitForTimeout(300);
    await page.keyboard.press('1');
    await page.waitForTimeout(6000);
    await page.screenshot({ path: join(OUT, `${name}-${LABEL}.png`) });
    await page.keyboard.press('Escape');
    console.log(`wrote ${join(OUT, `${name}-${LABEL}.png`)}`);
  }
  if (errors.length) console.log(`page errors: ${errors.slice(0, 3).join(' | ')}`);
} finally {
  await close();
}
