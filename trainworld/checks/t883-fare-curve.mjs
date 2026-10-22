// T-883: the fare curve reaches demand. On the real save (a ?new=1 game, the weekday's final):
// the default straight fare; the same line through three nodes (no change at all); a cap beyond
// 500 km (the curve's own path in the sim, but no New York trip is that long, so within 0.05%);
// a cap at $4.50 beyond 30 km (only longer trips get cheaper, so more rail trips). Screenshots of
// the Money tab's graph in light and dark go to artifacts/trainworld/fares2/.
//   node trainworld/checks/t883-fare-curve.mjs [url]   (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure. About five minutes.
import { mkdir, mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = (process.argv[2] ?? 'http://localhost:6150/').replace(/\/?$/, '/');
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));
const OUT = 'artifacts/trainworld/fares2/';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';

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

let failed = false;
const report = (name, ok, detail) => { if (!ok) failed = true; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { chromium } = await playwright();
await mkdir(OUT, { recursive: true });
const profile = await mkdtemp(join(tmpdir(), 'tw-t883-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(SAVE);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });

  /** the next final after `after` (a demand log version), its rail trips a day */
  const final = async (after) => {
    await page.waitForFunction((after) => {
      const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
      return s && v && v.version === s.version && v.version > after && v.final && !d.client.waiting && v.lines.size > 100;
    }, after, { timeout: 400000, polling: 250 });
    return page.evaluate(() => { const v = window.twDemand.view(); return { version: v.version, rail: v.railTripsPerDay }; });
  };
  const setNodes = (nodes) => page.evaluate((nodes) => window.tw.client.edit({ op: 'fares', nodes }), nodes);

  const a = await final(-1);
  report('default fare', Math.round(a.rail) === 11342122, `(${Math.round(a.rail).toLocaleString('en-US')} rail trips a day; want 11,342,122)`);

  // the same fares give the demand nothing new to solve, so no new final may come
  await setNodes([[1, 1.6], [10, 2.5], [100, 11.5]]);
  await page.waitForTimeout(15000);
  const b = await final(-1);
  report('the same line through three nodes', Math.round(b.rail) === Math.round(a.rail), `(${Math.round(b.rail).toLocaleString('en-US')})`);

  await setNodes([[1, 1.6], [500, 51.5], [1000, 51.5]]);
  const c = await final(b.version);
  const dc = (c.rail - a.rail) / a.rail;
  report('a cap beyond 500 km', Math.abs(dc) < 5e-4, `(${Math.round(c.rail).toLocaleString('en-US')}, ${(100 * dc).toFixed(3)}%)`);

  await setNodes([[1, 1.6], [30, 4.5], [100, 4.5]]);
  const d = await final(c.version);
  const dd = (d.rail - a.rail) / a.rail;
  report('a cap at $4.50 beyond 30 km', dd > 0, `(${Math.round(d.rail).toLocaleString('en-US')}, ${(100 * dd).toFixed(2)}% more rail trips)`);

  // screenshots of the Money tab's graph with the cap, light then dark
  await page.click('button.tab[aria-label="Money"]');
  for (const shade of ['light', 'dark']) {
    await page.evaluate((shade) => { document.documentElement.dataset.shade = shade; window.tw.ui?.shade && (window.tw.ui.shade.value = shade); }, shade);
    const g = await page.$('svg.fare-graph');
    await g.scrollIntoViewIfNeeded();
    await page.waitForTimeout(400);
    const box = await g.boundingBox();
    await page.screenshot({ path: `${OUT}money-cap-${shade}.png`, clip: { x: box.x - 8, y: box.y - 24, width: box.width + 16, height: box.height + 130 } });
  }
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await setNodes([[1, 1.6], [100, 11.5]]);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
