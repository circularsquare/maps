// T-3479: the "Terrain could not load" tag is clicked to try again at once. A new game paused in New York with
// every terrain download failing: the tag's hover says to click; a click asks for terrain again within a second
// and the tag says "Downloading terrain…" meanwhile; the try after that failure comes 2 s on (the wait started
// over); with the downloads working again, a click brings the terrain in and the tag goes. About a minute.
//   TW_CORES=2 node trainworld/checks/t3479-terrain-retry.mjs http://localhost:6160/
import { join } from 'node:path';
import { mkdirSync } from 'node:fs';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6160/';
const OUT = join(ARTIFACTS, 'terrainnet');
mkdirSync(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const hold = (ms) => new Promise((d) => setTimeout(d, ms));
let block = true;
const asked = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t3479-', size: [1300, 900] });
try {
  await ctx.route(/\/terrain-world\/[0-9_-]+\.terrain/, async (route) => {
    asked.push(Date.now());
    try { if (block) return await route.abort('internetdisconnected'); await route.continue(); } catch { /* the page went */ }
  });
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|ERR_INTERNET_DISCONNECTED|Failed to load resource/);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug-root { display: none !important; }' });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.98, 40.75], zoom: 12 }));
  const tag = () => page.evaluate(() => { const t = document.querySelector('.status-tag'); return t ? { text: t.querySelector('.demand-busy')?.textContent ?? '', tip: t.getAttribute('data-tip'), button: t.tagName === 'BUTTON' } : null; });
  const tagIs = async (text, ms) => { for (const end = Date.now() + ms; Date.now() < end; await hold(100)) if ((await tag())?.text === text) return true; return false; };
  check(await tagIs('Terrain could not load', 60000), `blocked: "Terrain could not load" (${JSON.stringify(await tag())})`);
  const t1 = await tag();
  check(t1?.tip === 'Check your internet connection. Click to try again, or wait up to 30 s.', `its hover says to click (${t1?.tip})`);
  check(t1?.button, 'the tag is a button');
  await page.screenshot({ path: join(OUT, 't3479-failed.png'), clip: { x: 0, y: 0, width: 1300, height: 220 } });
  // let the wait grow past a few seconds, so a try within a second of the click is the click's
  await hold(9000);
  const n0 = asked.length, c0 = Date.now();
  await page.click('.status-tag');
  const shown = await tagIs('Downloading terrain…', 800);
  for (const end = Date.now() + 1000; Date.now() < end && asked.length === n0; await hold(50));
  check(asked.length > n0, `a click asks again at once (${asked.length - n0} asked, first after ${asked.length > n0 ? asked[n0] - c0 : '-'} ms)`);
  check(shown, 'the tag says "Downloading terrain…" after the click');
  check(await tagIs('Terrain could not load', 5000), 'and "Terrain could not load" once that try fails');
  const n1 = asked.length, f1 = Date.now();
  for (const end = Date.now() + 4500; Date.now() < end && asked.length === n1; await hold(100));
  check(asked.length > n1, `the next try by itself comes about 2 s after (${asked.length > n1 ? asked[n1] - f1 : '-'} ms)`);
  block = false;
  await hold(500);
  await page.click('.status-tag').catch(() => {});
  let gone = false;
  for (const end = Date.now() + 20000; Date.now() < end && !gone; await hold(250)) gone = !/terrain/i.test((await tag())?.text ?? '');
  check(gone, `unblocked: a click brings the terrain in and the tag goes (${JSON.stringify(await tag())})`);
  check(errors.length === 0, `no page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally { await close(); }
process.exitCode = fail ? 1 : 0;
