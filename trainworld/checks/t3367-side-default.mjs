// T-3367: the new game sheet never asks the running side; the game takes it from the start.
//   A new game in New York keeps right, in Tokyo left; Choose on map follows the browser's time zone (Asia/Tokyo
//   left, America/New_York right, an unknown one right).
// Pictures in artifacts/trainworld/t3367-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6198):
//   node trainworld/checks/t3367-side-default.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6198/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

/** Starts a new game from the sheet with `pick`, returns the side the game keeps (1 right, -1 left). */
async function sideOf(timezoneId, pick, shotName) {
  const chrome = await launchChrome({ prefix: 'tw-t3367-', size: [1400, 900], deviceScaleFactor: 1, timezoneId });
  const page = await chrome.ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  try {
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
    await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
    await page.waitForSelector('.new-game, .start-new', { timeout: 120000 });
    if (!(await page.$('.new-game'))) await page.click('button.start-new');
    await page.waitForSelector('.new-game');
    check(!(await page.$('.new-game .side-switch')), `${timezoneId}: no side control on the sheet`);
    await pick(page);
    await page.waitForTimeout(400);
    if (shotName) await page.screenshot({ path: join(ARTIFACTS, `t3367-${shotName}.png`) });
    await page.click('.new-game-start');
    await page.waitForFunction(() => window.tw?.world?.value && window.tw.renderer?.net, undefined, { timeout: 240000 });
    await page.waitForTimeout(1500);
    return await page.evaluate(() => window.tw.renderer.net.keep ?? 1);
  } finally {
    const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
    if (bad.length) check(false, 'no console errors', bad.slice(0, 3).join(' | '));
    await chrome.close?.();
  }
}

try {
  const city = (id) => (page) => page.click(`.new-game button[data-city="${id}"]`);
  const map = (page) => page.click('.new-game-map');
  check(await sideOf('America/New_York', city('nyc'), 'sheet-after') === 1, 'New York starts keeping right');
  check(await sideOf('America/New_York', city('tyo')) === -1, 'Tokyo starts keeping left');
  check(await sideOf('Asia/Tokyo', map) === -1, 'Choose on map in a Tokyo time zone keeps left');
  check(await sideOf('America/New_York', map) === 1, 'Choose on map in a New York time zone keeps right');
} catch (e) {
  console.log('FAIL', e?.stack ?? e);
  fail++;
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
