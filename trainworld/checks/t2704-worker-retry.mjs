// Run from maps: node trainworld/checks/t2704-worker-retry.mjs <url>      (about a minute)
// T-2704: a demand worker whose code did not download is tried again (demandClient.ts `scriptFailed`).
// Headless Chrome on a ?new=1 game:
// 1. the first download of the demand worker's script is cut off: a new worker takes its place and
//    demand still reaches ready;
// 2. every download of it cut off: the pool stops after its tries, with the reason on the failure screen;
// 3. T-2710: the same for the clock worker (clockClient.ts `spawnWorker`): cut once, the game starts;
//    cut on every try, the failure screen.
import assert from 'node:assert/strict';
import { launchChrome } from './lib.mjs';

const url = (process.argv[2] ?? 'http://localhost:6171/').replace(/\/?$/, '/');
const { ctx, close } = await launchChrome({ prefix: 'tw-workerretry-', size: [1200, 800] });
let failed = false;
const SCRIPT = /demand\.worker\.(ts|js)/;

/** demand's status once it is ready or failed, or 'failure screen' */
const outcome = (page) => page.waitForFunction(() => {
  const s = String(window.twDemand?.client?.status ?? '');
  return s === 'ready' || s.startsWith('failed') || document.querySelector('.loading-screen') ? s || 'failure screen' : null;
}, undefined, { timeout: 180000, polling: 500 }).then((h) => h.jsonValue());

try {
  // 1
  const page = await ctx.newPage();
  let cut = 0, asked = 0;
  await page.route(SCRIPT, (route) => {
    asked++;
    if (cut) return route.continue();
    cut++;
    return route.abort('connectionreset');
  });
  const warned = [];
  page.on('console', (m) => { if (m.text().includes('did not download; trying again')) warned.push(m.text()); });
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  const status = await outcome(page);
  assert.equal(cut, 1, 'one script download cut');
  assert.equal(status, 'ready', `demand after its worker's script was cut: ${status}`);
  assert.ok(warned.length === 1, `one retry in the console: ${JSON.stringify(warned)}`);
  console.log(`PASS  a cut worker download is tried again  (${asked} script requests, "${warned[0]}")`);
  await page.close();

  // 2
  const p2 = await ctx.newPage();
  await p2.route(SCRIPT, (route) => route.abort('connectionreset'));
  await p2.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  const why = await p2.waitForFunction(() => document.querySelector('.loading-screen .load-why')?.textContent, undefined, { timeout: 120000, polling: 250 }).then((h) => h.jsonValue());
  assert.match(why, /could not be downloaded/, `the reason: ${why}`);
  console.log(`PASS  a worker that never downloads stops the load with its reason  ("${why}")`);
  await p2.close();

  // 3 (T-2710): the clock worker's script cut off once, then on every try
  const CLOCK = /clock\.worker\.(ts|js)/;
  const p3 = await ctx.newPage();
  let cut3 = 0;
  await p3.route(CLOCK, (route) => { if (cut3) return route.continue(); cut3++; return route.abort('connectionreset'); });
  const warned3 = [];
  p3.on('console', (m) => { if (m.text().startsWith('clock worker: its script did not download')) warned3.push(m.text()); });
  await p3.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  const s3 = await outcome(p3);
  const world = await p3.waitForFunction(() => !!window.tw?.world?.value || !!document.querySelector('.loading-screen'), undefined, { timeout: 120000, polling: 250 })
    .then(() => p3.evaluate(() => !!window.tw?.world?.value));
  assert.equal(cut3, 1, 'one clock script download cut');
  assert.ok(s3 === 'ready' && world && warned3.length === 1, `after the clock worker's script was cut: demand ${s3}, game ${world}, ${JSON.stringify(warned3)}`);
  console.log(`PASS  a cut clock worker download is tried again  (the game and demand start)`);
  await p3.close();

  const p4 = await ctx.newPage();
  await p4.route(CLOCK, (route) => route.abort('connectionreset'));
  await p4.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  const why4 = await p4.waitForFunction(() => document.querySelector('.loading-screen .load-why')?.textContent, undefined, { timeout: 120000, polling: 250 }).then((h) => h.jsonValue());
  assert.match(why4, /could not start/, `the reason: ${why4}`);
  console.log(`PASS  a clock worker that never downloads stops the load with its reason  ("${why4}")`);
  await p4.close();
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message.split('\n')[0]);
} finally {
  await close();
}
console.log(failed ? 't2704-worker-retry: FAILED' : 't2704-worker-retry: PASSED');
process.exit(failed ? 1 : 0);
