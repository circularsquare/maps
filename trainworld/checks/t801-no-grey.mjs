// T-801: no grey city bubbles. A start on the map in a fresh profile (demand None) turns Homes on and
// shows the zoomed-out bubbles of every zone; a click on one selects it. In a new New York game
// (start menu): zoomed out over the Northeast, Philadelphia and the rest are ordinary city bubbles;
// a pan over Philadelphia loads its pack for the view, and no market pack (no station reaches it).
// Screenshots to artifacts/trainworld/t801-*.png.
//   BASE=http://localhost:6141/ node trainworld/checks/t801-no-grey.mjs   (from the maps root)
import { readdir, mkdtemp, rm, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright');
}
const { chromium } = await playwright();
const BASE = process.env.BASE ?? 'http://localhost:6141/';
const SHOTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
await mkdir(SHOTS, { recursive: true });
const PHL = [-75.1636, 39.9526];

const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const profile = await mkdtemp(join(tmpdir(), 'tw-t801-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) errors.push(m.text()); });

const loadsOf = (id) => page.evaluate((id) => window.twDemand.start.loads.filter((l) => l.item === id && l.openedAt !== undefined).length, id);
/** the page point of (lng, lat) */
const at = (ll) => page.evaluate((ll) => {
  const p = window.tw.map.project(ll), r = window.tw.map.getContainer().getBoundingClientRect();
  return { x: p.x + r.left, y: p.y + r.top };
}, ll);
const tagText = () => page.evaluate(() => [...document.querySelectorAll('.demand-tag')].filter((e) => e.style.display !== 'none' && !e.classList.contains('track-tag')).map((e) => e.textContent).join(' | '));
const hoverTag = async (ll) => {
  const p = await at(ll);
  await page.mouse.move(p.x + 300, p.y + 300);
  await page.waitForTimeout(200);
  await page.mouse.move(p.x, p.y);
  await page.waitForTimeout(500);
  return tagText();
};

try {
  // 1. a start on the map, first in the fresh profile (demand None): Homes goes on, the zones'
  // bubbles show, and a click selects one
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  const homesBefore = await page.evaluate(() => window.tw?.display?.value?.commuters ?? null);
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await page.waitForTimeout(4000);
  const homesAfter = await page.evaluate(() => window.tw.display.value.commuters);
  const buttons = await page.$$eval('.demand-selector button', (b) => b.slice(0, 3).map((x) => `${x.textContent}:${x.getAttribute('aria-disabled')}`));
  report('a map start turns Homes on, buttons usable', homesBefore !== true && homesAfter === true && buttons.every((b) => b.endsWith(':false')), JSON.stringify({ homesBefore, homesAfter, buttons }));
  await page.screenshot({ path: join(SHOTS, 't801-world-light.png') });
  const LON = [-0.1276, 51.5072];
  const lonTag = await hoverTag(LON);
  // zoomed this far out the bubble on top near London may be a neighbouring zone's
  report('a map start shows city bubbles', /trips per day within /.test(lonTag), JSON.stringify(lonTag));
  const p = await at(LON);
  await page.mouse.click(p.x, p.y);
  await page.waitForTimeout(800);
  const sel = await page.evaluate(() => window.tw.selection?.peek?.() ?? null);
  report('a click on one selects it', sel?.kind === 'city', JSON.stringify(sel));
  await page.screenshot({ path: join(SHOTS, 't801-world-pick-light.png') });

  // 2. a new New York game from the start menu
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  report('no grey layer', await page.evaluate(() => window.tw.greyAt === undefined));

  // 3. zoomed out over the Northeast: Philadelphia is an ordinary city bubble
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; window.tw.map.jumpTo({ center: [-74.6, 40.3], zoom: 8 }); });
  await page.waitForTimeout(2500);
  const tag = await hoverTag(PHL);
  report('Philadelphia zoomed out is a city bubble', /within Philadelphia/.test(tag) && !/Click to open/.test(tag), JSON.stringify(tag));
  await page.mouse.move(10, 450);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.6, 40.9], zoom: 6.6 }));
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(SHOTS, 't801-northeast-light.png') });
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(SHOTS, 't801-northeast-dark.png') });
  await page.evaluate(() => { window.tw.shade.value = 'light'; });

  // 4. a pan over Philadelphia, Homes on, loads it for the view
  await page.evaluate((ll) => window.tw.map.jumpTo({ center: ll, zoom: 10.5 }), PHL);
  const loaded = await page.waitForFunction(() => window.twDemand.start.loads.some((l) => l.item === 'phl' && l.openedAt !== undefined), undefined, { timeout: 90000 }).then(() => true, () => false);
  await page.waitForTimeout(3000);
  // T-801: viewing alone opens no market pack (no station reaches Philadelphia)
  const markets = await page.evaluate(() => window.twDemand.start.loads.filter((l) => l.item.startsWith('market:')).map((l) => l.item));
  report('a pan over Philadelphia loads it, no market pack', loaded && markets.length === 0, `(${await loadsOf('phl')} loads opened, markets ${JSON.stringify(markets)})`);
  await page.screenshot({ path: join(SHOTS, 't801-phl-zoomed-light.png') });
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, e.message.split('\n')[0]);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
}
console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
process.exit(results.every(Boolean) ? 0 : 1);
