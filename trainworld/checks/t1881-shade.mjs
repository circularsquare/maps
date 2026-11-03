// T-1881: the hillshade (Settings, Terrain) and the terrain tint over dense centres, which must read
// flat, and real hills, which must not: a before/after pair. Per spot two pictures, the shade alone and
// with the track tool on (the tint). `?rawdem=1` on the url draws the Terrain Tiles as they come.
// Screenshots only, no checks.
// Usage: node trainworld/checks/t1881-shade.mjs <label> [url] [spot,spot] (default http://localhost:6158/, every spot)

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const LABEL = process.argv[2] ?? 'now';
const BASE = process.argv[3] ?? 'http://localhost:6158/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 't1881');
mkdirSync(OUT, { recursive: true });
// a new game in each spot's zone: a zone outside the game's realms loads no class grids
const ALL = [['tokyo', 139.755, 35.68, 13, 'tyo'], ['nagoya', 136.905, 35.17, 13, 'ngo'], ['shanghai', 121.48, 31.232, 13, 'sha'],
  ['midtown', -73.982, 40.755, 13.5, 'nyc'], ['hongkong', 114.165, 22.28, 13, 'hon'], ['sf', -122.425, 37.78, 13, 'snf'],
  ['seoul', 126.99, 37.556, 12.5, 'seo']];
const only = process.argv[4]?.split(',');
const SPOTS = only ? ALL.filter(([n]) => only.includes(n)) : ALL;

const { ctx, close } = await launchChrome({ prefix: 'tw-t1881-' });
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
  for (const [name, lng, lat, zoom, zone] of SPOTS) {
    await page.goto(`${BASE}${sep}debug=1&new=1&city=${zone}&paused=1&demandWorkers=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.evaluate(([lng, lat, zoom, theme]) => { window.tw.shade.value = theme; window.tw.map.jumpTo({ center: [lng, lat], zoom }); }, [lng, lat, zoom, process.env.THEME ?? 'light']);
    // the class grids come with the view, then the shade's tiles
    await page.waitForTimeout(9000);
    await page.screenshot({ path: join(OUT, `${name}-shade-${LABEL}.png`) });
    // the relief alone, shaded harder: every layer but the land, the shade and the water hidden
    const was = await page.evaluate(() => {
      const m = window.tw.map, keep = new Set(['background', 'hillshade', 'water_over_shade']);
      const out = {};
      for (const l of m.getStyle().layers) {
        out[l.id] = m.getLayoutProperty(l.id, 'visibility') ?? 'visible';
        if (!keep.has(l.id)) m.setLayoutProperty(l.id, 'visibility', 'none');
      }
      out.__ex = m.getPaintProperty('hillshade', 'hillshade-exaggeration');
      m.setPaintProperty('hillshade', 'hillshade-exaggeration', 1);
      return out;
    });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `${name}-relief-${LABEL}.png`), clip: { x: 300, y: 76, width: 1100, height: 824 } });
    await page.evaluate((was) => {
      const m = window.tw.map;
      for (const [id, v] of Object.entries(was)) if (id !== '__ex' && m.getLayer(id)) m.setLayoutProperty(id, 'visibility', v);
      m.setPaintProperty('hillshade', 'hillshade-exaggeration', was.__ex);
    }, was);
    await page.waitForTimeout(500);
    await page.mouse.click(22, 16); // the Build tab, then the track tool
    await page.waitForTimeout(300);
    await page.keyboard.press('1');
    await page.waitForTimeout(3000);
    await page.screenshot({ path: join(OUT, `${name}-tint-${LABEL}.png`) });
    await page.keyboard.press('Escape');
    const stats = await page.evaluate(() => JSON.stringify(window.twDemFlat ?? null));
    console.log(`wrote ${name}-shade-${LABEL}.png, ${name}-tint-${LABEL}.png; terrain tiles ${stats}`);
  }
  if (errors.length) console.log(`page errors: ${errors.slice(0, 3).join(' | ')}`);
} finally {
  await close();
}
