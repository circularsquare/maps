// Smoke check for the isochrone page: one PASS or FAIL line per check, exit 1 if any failed.
//   node isochrone/checks/smoke.mjs [url] [--canoe=<folder>]
// (from the maps root; default http://localhost:8800/isochrone/). The canoe checks run when the
// page serves data/canoe/index.json, or on data/canoe/<folder>/ with --canoe=<folder> (a test set).
// Playwright from PLAYWRIGHT_MODULE (a file URL of playwright/index.mjs), else `playwright`, else the npx cache.
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const ARGS = process.argv.slice(2);
const URL_ = ARGS.find((a) => !a.startsWith('--')) ?? 'http://localhost:8800/isochrone/';
const CANOE_SUB = (ARGS.find((a) => a.startsWith('--canoe=')) ?? '').slice(8);
// the canoe start: Constanta, on the Black Sea; a sea point off it and a point inland
const CONSTANTA = [28.63, 44.17], OFFSHORE = [30.2, 43.6];
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

  // 5. canoe: a start on the Black Sea coast solves, the sea is painted, a hover at sea shows a time
  let canoeOn = !!CANOE_SUB;
  if (!canoeOn) {
    try { canoeOn = await page.evaluate(async () => (await fetch('data/canoe/index.json', { method: 'HEAD' })).ok); } catch { /* no canoe data */ }
  }
  if (canoeOn) {
    const NAMES = ['canoe start solves', 'canoe paints the sea', 'canoe hover at sea shows a time', 'canoe zoom 8 gives an L1 window with sea'];
    try {
      const q = `mode=canoe${CANOE_SUB ? '&canoedata=' + CANOE_SUB : ''}&v=${Date.now()}`;
      await page.goto(`${URL_}${sep}${q}`);
      await page.waitForFunction(() => window.solveLog?.some((e) => e.level === 10), undefined, { timeout: 60000, polling: 100 });
      const n = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), CONSTANTA);
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, CONSTANTA);
      await page.mouse.click(xy[0], xy[1]);
      const e = await entry(10, n, 60000);
      report(NAMES[0], e.mode === 'canoe', `(L10 ${e.mode}, data ${e.loadMs} ms, solve ${e.solveMs} ms)`);
      // the L10 canvas pixel and time at a point off the coast
      const sea = await page.evaluate((c) => {
        const m = layers[10].m, lv = layers[10];
        const merc = (lat) => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
        const x = Math.floor((c[0] - m.west) / (m.east - m.west) * m.cw);
        const y = Math.floor((merc(m.north) - merc(c[1])) / (merc(m.north) - merc(m.south)) * m.ch);
        const col = Math.floor((c[0] - m.west) / m.cellDeg), row = Math.floor((m.north - c[1]) / m.cellDeg);
        return { alpha: lv.img.data[(y * m.cw + x) * 4 + 3], h: m.dist[row * m.W + col] };
      }, OFFSHORE);
      report(NAMES[1], sea.alpha > 0 && sea.h > 0 && sea.h <= 1024, `(alpha ${sea.alpha}, ${sea.h.toFixed(1)} h)`);
      const p = await page.evaluate((c) => { const q = map.project(c); return [q.x, q.y]; }, OFFSHORE);
      await page.mouse.move(p[0] - 5, p[1]);
      await page.mouse.move(p[0], p[1], { steps: 3 });
      await page.waitForFunction(() => document.getElementById('tip').style.display === 'block', undefined, { timeout: 10000, polling: 100 });
      const text = await page.evaluate(() => document.getElementById('tip').textContent);
      report(NAMES[2], /\d/.test(text), `("${text}")`);
      const n1 = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 8 }), [CONSTANTA[0] + 0.3, CONSTANTA[1]]);
      const e1 = await entry(1, n1, 60000);
      const h1 = await page.evaluate((c) => {
        const m = layers[1].m;
        return m.dist[Math.floor((m.north - c[1]) / m.cellDeg) * m.W + Math.floor((c[0] - m.west) / m.cellDeg)];
      }, [CONSTANTA[0] + 0.4, CONSTANTA[1]]);
      report(NAMES[3], h1 > 0 && h1 <= 1024, `(${e1.cols} x ${e1.rows}, fetch ${e1.fetchMs} ms, solve ${e1.solveMs} ms; sea ${h1.toFixed(1)} h)`);
    } catch (e) {
      report('canoe checks', false, first(e));
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
