// T-3055: the tram green hint beside the water works in Sakura-ku, Saitama, drawn with the new filter (the
// classes the street graph keeps) and then the old one (every service way). From the maps root:
//   node trainworld/checks/t3055-green-hint.mjs [url] [lng lat zoom]
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6169/';
const at = process.argv[4] ? [+process.argv[3], +process.argv[4], +(process.argv[5] ?? 16)] : [139.594, 35.8695, 16];
const chrome = await launchChrome({ prefix: 'tw-t3055-', size: [1000, 800] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(async ([a]) => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track';
    ui.tramTrack.value = true;
    ui.snapStreets.value = true;
    window.tw.map.jumpTo({ center: [a[0], a[1]], zoom: a[2] });
  }, [at]);
  await page.waitForTimeout(6000);
  const layer = await page.evaluate(() => window.tw.map.getLayer('tram_roads_hint') ? JSON.stringify(window.tw.map.getFilter('tram_roads_hint')) : null);
  console.log('hint layer filter:', layer);
  await page.screenshot({ path: 'artifacts/trainworld/t3055-after.png' });
  await page.evaluate(() => window.tw.map.setFilter('tram_roads_hint', ['any',
    ['in', ['get', 'class'], ['literal', ['trunk', 'primary', 'secondary', 'tertiary', 'minor', 'busway']]],
    ['all', ['==', ['get', 'class'], 'service'], ['!', ['has', 'service']], ['!', ['has', 'access']], ['!', ['has', 'brunnel']]]]));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'artifacts/trainworld/t3055-before.png' });
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
