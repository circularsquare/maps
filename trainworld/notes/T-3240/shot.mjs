// T-3242: a zone's city panel (its top destinations) and the long-trip names, screenshotted.
//   node shot.mjs <tag> [zone lon lat ...]     (Vite on 6133)
import { launchChrome, ARTIFACTS } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const tag = process.argv[2];
const zones = [];
for (let i = 3; i + 2 < process.argv.length + 1; i += 3) zones.push([process.argv[i], +process.argv[i + 1], +process.argv[i + 2]]);
const OUT = join(ARTIFACTS, 't3242');
mkdirSync(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t3242-', size: [1400, 900] });
const out = {};
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://localhost:6133/?debug=1&new=1&opened=all&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  for (const [id, lon, lat] of zones) {
    await page.evaluate(([lon, lat]) => window.tw.map.jumpTo({ center: [lon, lat], zoom: 9.5 }), [lon, lat]);
    await page.waitForFunction((id) => window.twDemand.client.start.loads.find((l) => l.item === id)?.openedAt, id, { timeout: 300000 });
    await page.evaluate((id) => window.tw.select({ kind: 'city', city: id }), id);
    const h = await page.waitForFunction(() => {
      const li = [...document.querySelectorAll('li')].find((l) => l.textContent.trim() === 'Top destinations');
      if (!li) return null;
      const rows = [...li.parentElement.querySelectorAll('li')].slice(1).map((l) => l.querySelector('.grow')?.textContent);
      return rows.length ? rows : null;
    }, undefined, { timeout: 300000, polling: 1000 });
    out[id] = await h.jsonValue();
    await page.waitForTimeout(1500);
    const li = page.locator('li', { hasText: /^Top destinations$/ }).first();
    const panel = li.locator('xpath=ancestor::*[contains(@class,"inspector") or contains(@class,"panel")][1]');
    const target = (await panel.count()) ? panel : page;
    await target.screenshot({ path: join(OUT, `t3242_${id}_${tag}.png`) });
    console.log(id, JSON.stringify(out[id]));
  }
  // the trip explorer's names for a few places, as the app now gives them
  out.names = await page.evaluate(async () => {
    const m = await import('/src/game/demand.ts');
    const ids = { jnj: 0, jgx: 0, osa: 0, lod: 0, kra: 0, dev: 0, poz: 0 };
    const reg = (await import('/src/game/cities.ts')).CITIES;
    return Object.fromEntries(Object.keys(ids).map((z) => { const c = reg.find((c) => c.id === z); return [z, c ? [m.INTERCITY_PLACES[c.place], m.intercityPlaceName(c.place)] : null]; }));
  });
  console.log('names', JSON.stringify(out.names));
  if (errors.length) console.log('page errors', errors.slice(0, 5));
} finally {
  writeFileSync(join(OUT, `t3242_${tag}.json`), JSON.stringify(out, null, 1));
  await close();
}
