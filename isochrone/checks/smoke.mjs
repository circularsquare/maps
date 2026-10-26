// Smoke check for the isochrone page: one PASS or FAIL line per check, exit 1 if any failed.
//   node isochrone/checks/smoke.mjs [url]      (from the maps root; default http://localhost:8800/isochrone/)
// Playwright from PLAYWRIGHT_MODULE (a file URL of playwright/index.mjs), else `playwright`, else the npx cache.
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const URL_ = process.argv[2] ?? 'http://localhost:8800/isochrone/';
const CHROME = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
// console errors that are not the page's: the favicon the server has none of
const NOISE = /favicon/;
const NAIROBI = [36.82, -1.29];
// a hover on a point a few km from the start: minutes or hours, never days
const TIP = /^(\d+ min|\d+ h( \d+ min)?)$/;

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

const results = [];
const report = (name, ok, detail = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
};
const t0 = Date.now();
const secs = () => ((Date.now() - t0) / 1000).toFixed(0) + ' s';
const first = (e) => String(e.message ?? e).split('\n')[0];

const profile = await mkdtemp(join(tmpdir(), 'iso-smoke-'));
process.on('exit', () => { try { rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }); } catch { /* left in the temp folder */ } });
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) process.on(sig, () => process.exit(1));

let ctx = null;
const errors = [];
try {
  const { chromium } = await playwright();
  ctx = await chromium.launchPersistentContext(profile, {
    executablePath: CHROME, headless: true, viewport: { width: 1400, height: 900 },
    args: ['--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !NOISE.test(m.location()?.url ?? '')) errors.push(m.text()); });

  /** wait for a solveLog entry of this level after the first `from` entries; its figures */
  const entry = async (level, from, timeout) => {
    await page.waitForFunction(([level, from]) => window.solveLog.slice(from).some((e) => e.level === level), [level, from], { timeout, polling: 100 });
    return page.evaluate(([level, from]) => window.solveLog.slice(from).find((e) => e.level === level), [level, from]);
  };
  const logLen = () => page.evaluate(() => window.solveLog.length);
  const sep = URL_.includes('?') ? '&' : '?';

  // 1. the page loads and the default solve from Cluj-Napoca arrives
  let up = false;
  try {
    await page.goto(`${URL_}${sep}v=${Date.now()}`);
    const e = await entry(10, 0, 60000);
    up = true;
    report('page loads, default solve', true, `(L10 solve ${e.solveMs} ms, paint ${e.paintMs} ms, ${secs()})`);
  } catch (e) {
    report('page loads, default solve', false, first(e));
  }

  if (!up) {
    for (const n of ['click gives a new L10 solve', 'zoom 6 gives an L3 window', 'zoom 8 gives an L1 window', 'hover shows a time']) report(n, false, '(page did not load)');
  } else {
    // 2. a click elsewhere: Nairobi, at a zoom with no windows
    try {
      const n = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), NAIROBI);
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, NAIROBI);
      await page.mouse.click(xy[0], xy[1]);
      const e = await entry(10, n, 30000);
      report('click gives a new L10 solve', true, `(L10 solve ${e.solveMs} ms)`);
    } catch (e) {
      report('click gives a new L10 solve', false, first(e));
    }

    // 3. zoom 6 gives an L3 window, zoom 8 an L1 window
    try {
      const n = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 6 }), NAIROBI);
      const e = await entry(3, n, 30000);
      report('zoom 6 gives an L3 window', true, `(${e.cols} x ${e.rows}, fetch ${e.fetchMs} ms, solve ${e.solveMs} ms)`);
    } catch (e) {
      report('zoom 6 gives an L3 window', false, first(e));
    }
    try {
      const n = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 8 }), NAIROBI);
      const e = await entry(1, n, 30000);
      report('zoom 8 gives an L1 window', true, `(${e.cols} x ${e.rows}, fetch ${e.fetchMs} ms, solve ${e.solveMs} ms)`);
    } catch (e) {
      report('zoom 8 gives an L1 window', false, first(e));
    }

    // 4. hover on a point a little east of the start reads the finest level
    try {
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x + 30, p.y]; }, NAIROBI);
      await page.mouse.move(xy[0] - 5, xy[1]);
      await page.mouse.move(xy[0], xy[1], { steps: 3 });
      await page.waitForFunction(() => document.getElementById('tip').style.display === 'block', undefined, { timeout: 10000, polling: 100 });
      const text = await page.evaluate(() => document.getElementById('tip').textContent);
      const finest = await page.evaluate(() => [1, 3, 10].find((k) => layers[k].shown));
      report('hover shows a time', TIP.test(text) && finest === 1, `("${text}" from L${finest})`);
    } catch (e) {
      report('hover shows a time', false, first(e));
    }
  }
  report('no console or page errors', errors.length === 0, errors.length ? `(${errors.length}: ${errors.slice(0, 3).join(' | ').slice(0, 400)})` : '');
} catch (e) {
  report('smoke check ran', false, first(e));
} finally {
  try { await ctx?.close(); } catch { /* already gone */ }
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => {});
}
const failed = results.filter((ok) => !ok).length;
console.log(failed ? `SMOKE FAILED: ${failed} of ${results.length} checks, ${secs()}` : `SMOKE PASSED: ${results.length} checks, ${secs()}`);
process.exit(failed ? 1 : 0);
