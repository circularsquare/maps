// T-1265: a network in Hong Kong's northwest (Tuen Mun, Yuen Long, Kam Tin, Tsuen Wan), whose
// stations are all nearer Shenzhen's centre than Hong Kong's but inside Hong Kong's outline, is saved
// and its slot names Hong Kong only; the page's own `cityWithin` agrees for each stop, and the line's
// country is HK. The saves panel goes to artifacts/trainworld/t1265/saves.png.
//   node trainworld/checks/t1265-hk-preview.mjs [url]   (from the maps root; default localhost:6167)
// Prints PASS or FAIL lines and exits 1 on a failure. About two minutes.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, buildLine, ARTIFACTS, ANGLE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6167/').replace(/\/?$/, '/');
const OUT = join(ARTIFACTS, 't1265');
await mkdir(OUT, { recursive: true });

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const STOPS = [[113.977, 22.391], [113.973, 22.411], [114.004, 22.459], [114.03, 22.445], [114.063, 22.441], [114.03, 22.40], [114.118, 22.373]];

const { ctx, close } = await launchChrome({ prefix: 'tw-t1265-', size: [1300, 900], args: ANGLE });
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}#12/22.42/114.03`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);

  const places = await page.evaluate(async (stops) => {
    const { cityWithin, centreWithin } = await import('/src/game/cities.ts');
    return stops.map(([lng, lat]) => `${cityWithin(lng, lat)}/${centreWithin(lng, lat)}`);
  }, STOPS);
  report('each stop is in Hong Kong by outline, Shenzhen by nearest centre', places.every((p) => p.startsWith('hon/')) && places.some((p) => p.endsWith('/shz')), `(${places.join(' ')})`);

  const built = await buildLine(page, STOPS, { name: 'Tuen Ma', colour: '#9a3b26' });
  report('line built', built?.line !== undefined, `(${JSON.stringify(built)})`);
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForTimeout(3000);
  const slot = await page.evaluate(async () => {
    const s = (await (await import('/src/workers/saveFile.ts')).listSlots())[0];
    return { cities: s?.cities, panels: (s?.panels ?? []).map((p) => `${p.city}:${p.stations}`) };
  });
  report('the slot names Hong Kong only', JSON.stringify(slot.cities) === '["hon"]' && slot.panels.length === 1 && slot.panels[0].startsWith('hon:'), `(${JSON.stringify(slot)})`);
  const country = await page.evaluate(async () => {
    const { lineCountry } = await import('/src/game/holidays.ts');
    const l = window.tw.world.value.save.lines.at(-1);
    return l ? lineCountry(l.stops) : null;
  });
  report("the line keeps Hong Kong's holidays", country === 'HK', `(${country})`);

  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector('.saves-panel .save-files');
  await page.waitForTimeout(800);
  await page.locator('.saves-panel').screenshot({ path: join(OUT, 'saves.png') });
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  console.log(`screenshot in ${OUT}`);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
