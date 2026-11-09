// T-2870: the status tag says "Loading map…" once the basemap's vector tiles have been missing for 3 s, and goes when they land.
//   1. normal tiles: the tag never says it through a load and a pan;
//   2. tiles held 0.6 s each: a pan loads within the threshold, the tag stays away;
//   3. tiles held 7 s each: the tag shows after about 3 s and goes once the tiles arrive.
// Screenshot to artifacts/trainworld/basemapload/. About one minute.
// Usage: node trainworld/checks/t2870-basemap-wait.mjs [url] (default http://localhost:6150/)

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const sep = BASE.includes('?') ? '&' : '?';
const OUT = join(ARTIFACTS, 'basemapload');
mkdirSync(OUT, { recursive: true });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) failed++; };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));

const net = { delay: 0 };
const { ctx, close } = await launchChrome({ prefix: 'tw-t2870-', size: [1300, 900] });
try {
  await ctx.route(/tiles\.openfreemap\.org\/planet\/.*\.pbf/, async (route) => {
    try {
      if (net.delay) await hold(net.delay);
      await route.continue();
    } catch { /* the page went */ }
  });
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|Failed to load resource/);
  await page.goto(`${BASE}${sep}debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  // every tag text seen, polled in the page, so a brief show is not missed
  await page.evaluate(() => {
    window.seen = [];
    setInterval(() => {
      const t = document.querySelector('.status-tag .demand-busy')?.textContent;
      if (t === 'Loading map…') window.seen.push(Math.round(performance.now()));
    }, 100);
  });
  const mapTag = () => page.evaluate(() => document.querySelector('.status-tag .demand-busy')?.textContent === 'Loading map…');
  const settle = async () => { await page.waitForFunction(() => window.tw.map.areTilesLoaded?.() ?? true, undefined, { timeout: 60000 }).catch(() => {}); await hold(500); };
  const clearSeen = () => page.evaluate(() => { window.seen = []; });
  const seen = () => page.evaluate(() => window.seen.length);
  const jump = (lng, lat, z) => page.evaluate(([a, b, c]) => window.tw.map.jumpTo({ center: [a, b], zoom: c }), [lng, lat, z]);

  // 1. normal tiles
  await settle();
  await clearSeen();
  for (const [lng, lat] of [[-73.95, 40.75], [-74.1, 40.65], [-73.8, 40.85], [-74.0, 40.7]]) { await jump(lng, lat, 11); await hold(1500); }
  check(!(await seen()) && !(await mapTag()), `normal tiles: the tag never showed (${await seen()} samples)`);

  // 2. tiles held 0.6 s: a pan stays under the threshold
  net.delay = 600;
  await clearSeen();
  for (const [lng, lat] of [[-71.1, 42.35], [-71.3, 42.5], [-71.0, 42.2]]) { await jump(lng, lat, 11); await hold(3000); }
  check(!(await seen()), `tiles held 0.6 s: the tag never showed (${await seen()} samples)`);

  // 3. tiles held 7 s
  net.delay = 0;
  await page.waitForFunction(() => window.tw.map.isSourceLoaded('openmaptiles') && !document.querySelector('.status-tag'), undefined, { timeout: 60000 }).catch(() => {});
  await hold(1000);
  net.delay = 7000;
  await clearSeen();
  await page.evaluate(() => {
    window.t3 = { start: null, shown: null };
    const m = window.tw.map;
    const id = setInterval(() => {
      const now = performance.now();
      if (window.t3.start === null && !m.isSourceLoaded('openmaptiles')) window.t3.start = now;
      if (window.t3.start !== null && window.t3.shown === null && document.querySelector('.status-tag .demand-busy')?.textContent === 'Loading map…') window.t3.shown = now;
      if (window.t3.shown !== null) clearInterval(id);
    }, 20);
  });
  await jump(-77.04, 38.9, 11.6);
  await page.waitForFunction(() => window.t3.shown !== null, undefined, { timeout: 8000 }).catch(() => {});
  const { start, shown } = await page.evaluate(() => window.t3);
  const shownAt = start !== null && shown !== null ? Math.round(shown - start) : null;
  check(shownAt !== null && shownAt >= 2700 && shownAt <= 4500, `tiles held 7 s: the tag shows ${shownAt} ms after the tiles went missing (about 3000)`);
  await page.screenshot({ path: join(OUT, 'loading-map.png'), clip: { x: 0, y: 0, width: 1300, height: 220 } });
  const t1 = Date.now();
  let goneAt = null;
  while (Date.now() - t1 < 15000 && goneAt === null) { if (!(await mapTag())) goneAt = Date.now() - t1; else await hold(100); }
  check(goneAt !== null && goneAt < 8000, `it goes ${goneAt} ms after it showed, once the tiles are in`);
  net.delay = 0;
  check(errors.length === 0, `no console errors (${errors.slice(0, 3).join(' | ')})`);
} finally {
  await close();
}
console.log(failed ? `${failed} FAILED` : 'ALL PASS');
process.exit(failed ? 1 : 0);
