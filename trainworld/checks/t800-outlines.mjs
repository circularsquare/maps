// T-800: every zone has an outline from the start (packs/outlines.json, pipeline/zone_outlines.py).
// A new New York game, then a jump to Europe: the outline source should hold the Europe zones at
// once, before any of their own boundary files have been fetched, and each zone's own outline
// replaces its coarse one when its pack arrives. NOFILE=1 blocks the coarse file (the old behaviour).
// Screenshots to artifacts/trainworld/t800-<TAG>-*.png.
//   BASE=http://localhost:6146/ [NOFILE=1 TAG=before] node trainworld/checks/t800-outlines.mjs   (from the maps root)
import { readdir, mkdtemp, mkdir } from 'node:fs/promises';
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
const BASE = process.env.BASE ?? 'http://localhost:6146/';
const TAG = process.env.TAG ?? (process.env.NOFILE ? 'before' : 'after');
const SHOTS = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
await mkdir(SHOTS, { recursive: true });
const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};

const profile = await mkdtemp(join(tmpdir(), 'tw-t800-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) errors.push(m.text()); });
let boundaryFetches = 0;
page.on('request', (r) => { if (r.url().includes('.boundary.geojson')) boundaryFetches++; });
if (process.env.NOFILE) await page.route('**/packs/outlines.json*', (r) => r.abort());
// a slow connection: each zone's boundary file takes up to DELAY ms (the local server answers at once)
const DELAY = Number(process.env.DELAY ?? 0);
if (DELAY) await page.route('**/*.boundary.geojson*', async (r) => { await new Promise((ok) => setTimeout(ok, Math.random() * DELAY)); await r.continue(); });

/** zone ids with an outline feature in the rendered view */
const drawn = () => page.evaluate(() => {
  const f = window.tw.map.querySourceFeatures('tw-city-boundaries');
  return [...new Set(f.map((x) => x.properties.id))];
});
const settle = (ms) => page.waitForTimeout(ms);

try {
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await settle(4000);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74, 41], zoom: 6.5 }));
  await settle(3000);
  await page.screenshot({ path: join(SHOTS, `t800-${TAG}-1-newyork.png`) });
  const ny = await drawn();
  console.log(`New York view, zoom 6.5: ${ny.length} zones drawn`);

  // the jump to Europe: no boundary file of a European zone has been asked for yet
  const fetched0 = boundaryFetches;
  await page.evaluate(() => window.tw.map.jumpTo({ center: [8, 49], zoom: 5.2 }));
  await settle(700);
  await page.screenshot({ path: join(SHOTS, `t800-${TAG}-2-europe-at-once.png`) });
  const first = await drawn();
  console.log(`Europe, 0.7 s after the jump: ${first.length} zones drawn, ${boundaryFetches - fetched0} boundary files fetched since`);
  await settle(8000);
  await page.screenshot({ path: join(SHOTS, `t800-${TAG}-3-europe-settled.png`) });
  const later = await drawn();
  console.log(`Europe, settled: ${later.length} zones drawn, ${boundaryFetches - fetched0} boundary files fetched since`);

  await page.evaluate(() => window.tw.map.jumpTo({ center: [10, 50], zoom: 3.2 }));
  await settle(1500);
  await page.screenshot({ path: join(SHOTS, `t800-${TAG}-4-wide.png`) });
  const wide = await drawn();
  console.log(`Wide view, zoom 3.2: ${wide.length} zones drawn`);

  if (!process.env.NOFILE) {
    report('Europe has outlines at once', first.length >= 40, `${first.length}`);
    report('the outline source holds the whole world', wide.length >= 100, `${wide.length}`);
  }
  const bad = errors.filter((e) => !(process.env.NOFILE && /ERR_FAILED/.test(e)));
  report('no console errors', bad.length === 0, bad.slice(0, 3).join(' | '));
} finally {
  await ctx.close();
}
process.exit(results.every(Boolean) ? 0 : 1);
