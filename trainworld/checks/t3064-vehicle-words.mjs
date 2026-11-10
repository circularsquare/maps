// T-3064: a line's panel in train and tram words. Opens the real save, selects a line, screenshots its panel,
// then marks the line a tram in the page's copy of the world (nothing saved) and screenshots again; prints the
// messages that take a vehicle for tram and bus. From the maps root:
//   node trainworld/checks/t3064-vehicle-words.mjs [url]
import { readFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, REAL_SAVE } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6190/';
const chrome = await launchChrome({ prefix: 'tw-t3064-', size: [1400, 1000] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  await page.evaluate(() => { const w = window.tw.world.value; window.tw.selection.value = { kind: 'line', line: w.save.lines[0].id }; });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'artifacts/trainworld/t3064-train-line.png' });
  await page.evaluate(() => { const w = window.tw.world.value; const id = w.save.lines[0].id; w.lineStats[id].tram = true; window.tw.world.value = { ...w }; });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'artifacts/trainworld/t3064-tram-line.png' });
  const out = await page.evaluate(async () => {
    const { t } = await import('/src/i18n/index.ts');
    const r = {};
    for (const v of ['train', 'tram', 'bus']) {
      r[v] = [
        t('lineInspector.reach', { speed: '80 km/h', vehicle: v }), t('lineInspector.fits', { length: '60 m', name: 'Canal St', cars: '3 cars', vehicle: v }),
        t('schedule.leftBehind', { vehicle: v }), t('schedule.limit', { full: false, what: 'Platform', time: '4m', vehicle: v }),
        t('stationInspector.noTrain', { vehicle: v }), t('oneRider.tip', { vehicle: v }), t('trackHover.noVehicles', { vehicles: v === 'bus' ? 'buses' : v + 's' }),
      ];
    }
    return r;
  });
  console.log(JSON.stringify(out, null, 1));
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
