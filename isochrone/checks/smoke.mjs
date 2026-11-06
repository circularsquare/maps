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
// a walking start in the Sahara and a point south-west of it with survival ~96% (land hazard, T-067)
const TAMANRASSET = [5.53, 22.79], SAHARA_SW = [2.5, 18.8];
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

  // 1b. corridors of the default solve arrive with lines, drawn, and the switch hides them
  if (up) {
    try {
      await page.waitForFunction(() => window.corrLog?.some((e) => e.level === 10), undefined, { timeout: 30000, polling: 100 });
      const c = await page.evaluate(() => window.corrLog.find((e) => e.level === 10));
      const drawn = await page.evaluate(async () => (await map.getSource('corridors').getData()).features.length);
      await page.click('#corrbtn');
      const off = await page.evaluate(() => map.getLayoutProperty('corridors', 'visibility'));
      await page.click('#corrbtn');
      const on = await page.evaluate(() => map.getLayoutProperty('corridors', 'visibility'));
      report('corridors arrive with lines', c.lines > 20 && drawn > 20 && off === 'none' && on === 'visible',
        `(${c.lines} lines, ${drawn} drawn, ${c.ms} ms; switch ${off}/${on})`);
    } catch (e) {
      report('corridors arrive with lines', false, first(e));
    }
  }

  if (!up) {
    for (const n of ['click gives a new L10 solve', 'zoom 6 gives an L3 window', 'zoom 8 gives an L1 window', 'hover shows a time', 'walking hover in the Sahara shows survival']) report(n, false, '(page did not load)');
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
      await page.click('#corrbtn');   // off: a corridor line under the pointer would show its speed, not the time
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x + 30, p.y]; }, NAIROBI);
      await page.mouse.move(xy[0] - 5, xy[1]);
      await page.mouse.move(xy[0], xy[1], { steps: 3 });
      await page.waitForFunction(() => document.getElementById('tip').style.display === 'block', undefined, { timeout: 10000, polling: 100 });
      const text = await page.evaluate(() => document.getElementById('tip').firstChild.textContent);
      await page.click('#corrbtn');
      const finest = await page.evaluate(() => [1, 3, 10].find((k) => layers[k].shown));
      report('hover shows a time', TIP.test(text) && finest === 1, `("${text}" from L${finest})`);
    } catch (e) {
      report('hover shows a time', false, first(e));
    }

    // 4a. land hazard: a walking start at Tamanrasset, a hover ~600 km south-west in the Sahara shows survival
    try {
      const n = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 4 }), [4, 21]);
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, TAMANRASSET);
      await page.mouse.click(xy[0], xy[1]);
      await entry(10, n, 30000);
      await page.click('#corrbtn');
      const p = await page.evaluate((c) => { const q = map.project(c); return [q.x, q.y]; }, SAHARA_SW);
      await page.mouse.move(p[0] - 5, p[1]);
      await page.evaluate(() => { window.lastRoute = null; });
      await page.mouse.move(p[0], p[1], { steps: 3 });
      await page.waitForFunction(() => window.lastRoute, undefined, { timeout: 10000, polling: 100 });
      const r = await page.evaluate(() => window.lastRoute);
      await page.click('#corrbtn');
      const s = /^survival (\d+)%$/.exec(r.split);
      report('walking hover in the Sahara shows survival', r.mode === 'walk' && s && +s[1] < 100, `("${r.split}", ${(r.t / 8).toFixed(1)} days)`);
    } catch (e) {
      report('walking hover in the Sahara shows survival', false, first(e));
    }
  }

  // 4b. L0.1: a walking start in Midtown Manhattan at zoom 12 gives a 100 m window, Brooklyn is
  // reached over the East River bridges in hours, and hover reads L0.1 (when data/fine/ has the tile)
  let fineOn = false;
  if (up) {
    try { fineOn = await page.evaluate(async () => { const r = await fetch('data/fine/index.json'); return r.ok && (await r.json()).tiles.includes('n40w074'); }); } catch { /* no fine tiles */ }
  }
  if (fineOn) {
    const MIDTOWN = [-73.985, 40.758], BROOKLYN = [-73.99, 40.695];
    try {
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 12 }), [-73.98, 40.73]);
      const n = await logLen();
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, MIDTOWN);
      await page.mouse.click(xy[0], xy[1]);
      const e = await entry(0.1, n, 60000);
      const h = await page.evaluate((c) => {
        const m = layers[0.1].m;
        return m.dist[Math.floor((m.north - c[1]) / m.cellDeg) * m.W + Math.floor((c[0] - m.west) / m.cellDeg)];
      }, BROOKLYN);
      report('zoom 12 in Manhattan gives an L0.1 window, Brooklyn reached', h > 0.5 && h < 4,
        `(${e.cols} x ${e.rows}, fetch ${e.fetchMs} ms, solve ${e.solveMs} ms; Brooklyn ${h.toFixed(2)} h)`);
      const p = await page.evaluate((c) => { const q = map.project(c); return [q.x, q.y]; }, BROOKLYN);
      await page.mouse.move(p[0] - 5, p[1]);
      await page.mouse.move(p[0], p[1], { steps: 3 });
      await page.waitForFunction(() => document.getElementById('tip').style.display === 'block', undefined, { timeout: 10000, polling: 100 });
      const text = await page.evaluate(() => document.getElementById('tip').firstChild.textContent);
      const finest = await page.evaluate(() => [0.1, 1, 3, 10].find((k) => layers[k].shown));
      report('hover reads L0.1', TIP.test(text) && finest === 0.1, `("${text}" from L${finest})`);
    } catch (e) {
      report('L0.1 checks', false, first(e));
    }
  }

  // 4c. L0.1 in a partial town tile (Lagos, n06e003: its southern row of blocks has no 100 m data
  // and is filled from L1): a walking start at zoom 12 gives an L0.1 window
  let lagosOn = false;
  if (up) {
    try { lagosOn = await page.evaluate(async () => { const r = await fetch('data/fine/index.json'); return r.ok && 'n06e003' in ((await r.json()).partial ?? {}); }); } catch { /* no fine tiles */ }
  }
  if (lagosOn) {
    try {
      const LAGOS = [3.38, 6.50];
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 12 }), LAGOS);
      const n = await logLen();
      const xy = await page.evaluate((c) => { const p = map.project(c); return [p.x, p.y]; }, LAGOS);
      await page.mouse.click(xy[0], xy[1]);
      const e = await entry(0.1, n, 60000);
      report('zoom 12 in Lagos (a partial tile) gives an L0.1 window', true, `(${e.cols} x ${e.rows}, fetch ${e.fetchMs} ms, solve ${e.solveMs} ms)`);
    } catch (e) {
      report('zoom 12 in Lagos (a partial tile) gives an L0.1 window', false, first(e));
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
      await page.evaluate(() => { window.lastRoute = null; });
      await page.mouse.move(p[0], p[1], { steps: 3 });
      await page.waitForFunction(() => document.getElementById('tip').style.display === 'block', undefined, { timeout: 10000, polling: 100 });
      const text = await page.evaluate(() => document.getElementById('tip').textContent);
      report(NAMES[2], /\d/.test(text), `("${text}")`);
      // the route to that point comes back from the worker with its split
      try {
        await page.waitForFunction(() => window.lastRoute, undefined, { timeout: 10000, polling: 100 });
        const r = await page.evaluate(() => window.lastRoute);
        report('canoe hover gives a route and a split', r.mode === 'canoe' && r.points >= 2 && /paddled \d/.test(r.split),
          `(${r.points} points, "${r.split}")`);
      } catch (e) {
        report('canoe hover gives a route and a split', false, first(e));
      }
      const n1 = await logLen();
      await page.evaluate((c) => map.jumpTo({ center: c, zoom: 8 }), [CONSTANTA[0] + 0.3, CONSTANTA[1]]);
      const e1 = await entry(1, n1, 60000);
      const h1 = await page.evaluate((c) => {
        const m = layers[1].m;
        return m.dist[Math.floor((m.north - c[1]) / m.cellDeg) * m.W + Math.floor((c[0] - m.west) / m.cellDeg)];
      }, [CONSTANTA[0] + 0.4, CONSTANTA[1]]);
      report(NAMES[3], h1 > 0 && h1 <= 1024, `(${e1.cols} x ${e1.rows}, fetch ${e1.fetchMs} ms, solve ${e1.solveMs} ms; sea ${h1.toFixed(1)} h)`);
      // the mode buttons re-solve the same start and keep the URL
      try {
        const got = [];
        for (const m of ['walk', 'canoe']) {
          const n2 = await logLen();
          await page.click(`#controls [data-mode=${m}]`);
          const e2 = await entry(10, n2, 60000);
          const st = await page.evaluate(() => ({ url: location.search, title: document.getElementById('title').textContent }));
          got.push(e2.mode === m && /mode=canoe/.test(st.url) === (m === 'canoe') ? m : `${m}: ${e2.mode} ${st.url} "${st.title}"`);
        }
        report('mode buttons switch walking and canoe', got.join() === 'walk,canoe', `(${got.join(', ')})`);
      } catch (e) {
        report('mode buttons switch walking and canoe', false, first(e));
      }
      // after the session's starts and windows, the canoe start's L10 corridors still lead back to
      // the start: each line ends at the start's cell or on another line (T-090: counts left dirty by
      // one start broke later starts' lines into pieces)
      try {
        await page.waitForFunction(() => corr[10] && corr[10].originId === originId, undefined, { timeout: 60000, polling: 100 });
        const r = await page.evaluate(() => {
          const c = corr[10], m = layers[10].m, o = originMarker.getLngLat();
          const key = (x, y) => x.toFixed(4) + ',' + y.toFixed(4);
          const seen = new Map();
          for (let i = 0; i + 1 < c.off.length; i++) for (let j = c.off[i]; j < c.off[i + 1]; j++) {
            const k = key(c.xy[2 * j], c.xy[2 * j + 1]);
            seen.set(k, (seen.get(k) || 0) + 1);
          }
          const sx = m.west + (Math.floor((o.lng - m.west) / m.cellDeg) + 0.5) * m.cellDeg;
          const sy = m.north - (Math.floor((m.north - o.lat) / m.cellDeg) + 0.5) * m.cellDeg;
          let loose = 0;
          for (let i = 0; i + 1 < c.off.length; i++) {
            const j = c.off[i + 1] - 1, x = c.xy[2 * j], y = c.xy[2 * j + 1];
            if ((Math.abs(x - sx) < 1e-3 && Math.abs(y - sy) < 1e-3) || seen.get(key(x, y)) > 1) continue;
            loose++;
          }
          return { lines: c.s.length, loose };
        });
        report('canoe corridors lead back to the start', r.lines > 20 && r.loose === 0, `(${r.lines} lines, ${r.loose} loose ends)`);
      } catch (e) {
        report('canoe corridors lead back to the start', false, first(e));
      }
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
