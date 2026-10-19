// T-729: cities a game has not opened are grey bubbles. In a new New York game (start menu):
// Philadelphia is grey, the hover lightens it and says "Click to open", a pan over it loads nothing,
// and a click opens it without moving the camera. The real save, imported, starts its list with the
// zone in view and the zones its stations reach; a grey city clicked there is in the list after a
// reload. A start on the map has every city grey. Screenshots in light and dark.
//   BASE=http://localhost:6150/ node trainworld/checks/t729-grey.mjs   (from the maps root)
import { readFile, readdir, mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
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
const BASE = process.env.BASE ?? 'http://localhost:6150/';
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const SHOTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
await mkdir(SHOTS, { recursive: true });
const PHL = [-75.1636, 39.9526];

const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const profile = await mkdtemp(join(tmpdir(), 'tw-t729-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) errors.push(m.text()); });

const opened = () => page.evaluate(() => window.tw.opened());
const loadsOf = (id) => page.evaluate((id) => window.twDemand.start.loads.filter((l) => l.item === id || l.item === 'market:' + id).length, id);
/** the page point of (lng, lat) */
const at = (ll) => page.evaluate((ll) => {
  const p = window.tw.map.project(ll), r = window.tw.map.getContainer().getBoundingClientRect();
  return { x: p.x + r.left, y: p.y + r.top };
}, ll);
/** the mean brightness of a small square of the page (0-255) */
async function brightness(x, y, r = 4) {
  const png = await page.screenshot({ clip: { x: x - r, y: y - r, width: 2 * r, height: 2 * r } });
  return page.evaluate(async (b64) => {
    const img = new Image();
    img.src = 'data:image/png;base64,' + b64;
    await img.decode();
    const c = new OffscreenCanvas(img.width, img.height), g = c.getContext('2d');
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, img.width, img.height).data;
    let s = 0;
    for (let i = 0; i < d.length; i += 4) s += (d[i] + d[i + 1] + d[i + 2]) / 3;
    return s / (d.length / 4);
  }, png.toString('base64'));
}
const tagText = () => page.evaluate(() => [...document.querySelectorAll('.demand-tag')].filter((e) => e.style.display !== 'none' && !e.classList.contains('track-tag')).map((e) => e.textContent).join(' | '));

try {
  // 1. a new New York game from the start menu
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && !document.querySelector('.start-menu') && Array.isArray(window.tw.opened()), undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  const o1 = await opened();
  report('new New York game opens New York only', o1.length === 1 && o1[0] === 'nyc', JSON.stringify(o1));

  // 2. Philadelphia grey, its hover lightens it
  await page.evaluate((ll) => { window.tw.clock.setRunning?.(false); window.tw.map.jumpTo({ center: [-74.6, 40.3], zoom: 8 }); }, PHL);
  await page.waitForTimeout(1500);
  const p = await at(PHL);
  const sx = p.x - 10, sy = p.y - 10; // a corner of the disc, clear of the tag
  await page.mouse.move(p.x + 300, p.y + 300);
  await page.waitForTimeout(300);
  const before = await brightness(sx, sy);
  await page.mouse.move(p.x, p.y);
  await page.waitForTimeout(400);
  const tag = await tagText();
  const after = await brightness(sx, sy);
  report('Philadelphia is grey with its tag', /Philadelphia/.test(tag) && /Click to open/.test(tag), JSON.stringify(tag));
  report('hover lightens it', after > before + 3, `(${before.toFixed(1)} to ${after.toFixed(1)})`);
  await page.screenshot({ path: join(SHOTS, 't729-grey-hover-light.png') });
  await page.mouse.move(p.x + 300, p.y + 300);
  await page.waitForTimeout(300);
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; window.tw.map.jumpTo({ center: [-73.6, 40.9], zoom: 6.6 }); });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(SHOTS, 't729-northeast-light.png') });
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(SHOTS, 't729-northeast-dark.png') });
  await page.evaluate(() => { window.tw.map.jumpTo({ center: [-74.6, 40.3], zoom: 8 }); });
  await page.waitForTimeout(800);
  { const q = await at(PHL); await page.mouse.move(q.x, q.y); }
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(SHOTS, 't729-grey-hover-dark.png') });
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await page.mouse.move(10, 450);

  // 3. a pan over Philadelphia, Homes on, loads nothing
  await page.evaluate((ll) => window.tw.map.jumpTo({ center: ll, zoom: 10.5 }), PHL);
  await page.waitForTimeout(8000);
  const panLoads = await loadsOf('phl');
  const focus = await page.evaluate(() => window.twDemand.activeCity.peek());
  report('a pan over grey Philadelphia loads nothing', focus === 'phl' && panLoads === 0, `(focus ${focus}, ${panLoads} loads)`);
  await page.screenshot({ path: join(SHOTS, 't729-phl-zoomed-light.png') });

  // 4. a click opens it, the camera staying
  const q = await at(PHL);
  const cam0 = await page.evaluate(() => { const c = window.tw.map.getCenter(); return [c.lng, c.lat, window.tw.map.getZoom()]; });
  await page.mouse.click(q.x, q.y);
  await page.waitForTimeout(300);
  const cam1 = await page.evaluate(() => { const c = window.tw.map.getCenter(); return [c.lng, c.lat, window.tw.map.getZoom()]; });
  const o2 = await opened();
  report('a click opens it without a camera move', o2.includes('phl') && cam0.every((v, i) => Math.abs(v - cam1[i]) < 1e-9), JSON.stringify({ opened: o2, cam0, cam1 }));
  const shown = await page.waitForFunction(() => window.twDemand.start.loads.some((l) => l.item === 'phl' && l.openedAt !== undefined), undefined, { timeout: 60000 }).then(() => true, () => false);
  report('it loads once clicked', shown);
  await page.waitForTimeout(2500);
  await page.screenshot({ path: join(SHOTS, 't729-phl-opened-light.png') });

  // 5. the real save imported (a slot of its own, written at once): zone in view plus the reached
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.985, 40.758], zoom: 10.5 }));
  await page.waitForTimeout(1500);
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 100, undefined, { timeout: 180000 });
  await page.waitForFunction(() => window.twDemand.client.knownReach() !== null, undefined, { timeout: 120000, polling: 200 });
  await page.waitForTimeout(1000);
  const mig = await page.evaluate(() => ({ opened: window.tw.opened(), reached: [...window.twDemand.client.knownReach()] }));
  const want = new Set(['nyc', ...mig.reached]);
  report('the real save starts with the view and the reached zones', mig.opened.length === want.size && mig.opened.every((id) => want.has(id)), JSON.stringify(mig));
  const grey = ['phl', 'bos', 'hfd', 'tor'].find((id) => !mig.opened.includes(id));
  const ll = { phl: PHL, bos: [-71.0589, 42.3601], hfd: [-72.6851, 41.7637], tor: [-79.3832, 43.6534] }[grey];
  await page.evaluate((ll) => window.tw.map.jumpTo({ center: ll, zoom: 9 }), ll);
  await page.waitForTimeout(1500);
  const g = await at(ll);
  await page.mouse.click(g.x, g.y);
  await page.waitForTimeout(500);
  const savedBefore = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, savedBefore, { timeout: 60000 });
  await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value?.save?.lines?.length > 100 && Array.isArray(window.tw.opened?.()), undefined, { timeout: 180000 });
  const o3 = await opened();
  report(`a clicked city (${grey}) is in the saved list after a reload`, o3.includes(grey) && mig.opened.every((id) => o3.includes(id)), JSON.stringify(o3));

  // 6. a start on the map: every city grey
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu') && Array.isArray(window.tw.opened()), undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  const o4 = await opened();
  report('a start on the map has every city grey', o4.length === 0, JSON.stringify(o4));
  await page.screenshot({ path: join(SHOTS, 't729-world-light.png') });
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, e.message.split('\n')[0]);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
}
await writeFile(join(SHOTS, 't729-grey.json'), JSON.stringify({ results, errors }, null, 1));
console.log(results.every(Boolean) ? 'ALL PASS' : 'SOME FAILED');
