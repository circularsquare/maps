// Run from maps: node trainworld/checks/loadfail.mjs [port]      (default 6191; about a minute)
// A load that meets a transient failure (loadfail, 2026-11-06: Anita's North America save failed once
// at "Failed to load" and opened fine on the next two tries). Own Vite on `port`, headless Chrome:
// 1. the first download of each WASM file (track model, demand) is cut off; the workers try again
//    (workers/wasmInit.ts) and the game and its demand still start.
// 2. a load that cannot finish (`?quietMs=1`, the watchdog at once) shows its reason under the
//    failure sentence (`.load-why`) and logs it to the console.
import assert from 'node:assert/strict';
import { launchChrome, startVite } from './lib.mjs';

const port = Number(process.argv[2] ?? 6191);
const { child, url } = await startVite(port);
const { ctx, close } = await launchChrome({ prefix: 'tw-loadfail-', size: [1200, 800] });
let failed = false;
try {
  // 1
  const page = await ctx.newPage();
  const cut = new Map();
  await page.route(/sim_bg\.wasm/, (route) => {
    const path = new URL(route.request().url()).pathname;
    if (cut.has(path)) return route.continue();
    cut.set(path, 1);
    return route.abort('connectionreset');
  });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const status = await page.waitForFunction(() => {
    const s = String(window.twDemand?.client?.status ?? '');
    return s === 'ready' || s.startsWith('failed') || document.querySelector('.loading-screen') ? s || 'failure screen' : null;
  }, undefined, { timeout: 180000, polling: 500 }).then((h) => h.jsonValue());
  assert.ok(cut.size >= 2, `cut ${cut.size} WASM downloads (${[...cut.keys()].join(', ')}), wanted the track model's and demand's`);
  assert.equal(status, 'ready', `demand after its WASM download was cut: ${status}`);
  assert.deepEqual(errors, [], 'page errors');
  console.log(`PASS  a cut WASM download is tried again  (${[...cut.keys()].map((p) => p.split('/').slice(-2).join('/')).join(', ')})`);
  await page.close();

  // 2
  const p2 = await ctx.newPage();
  const logged = [];
  p2.on('console', (m) => { if (m.type() === 'error' && m.text().startsWith('load failed:')) logged.push(m.text()); });
  await p2.goto(`${url}?debug=1&new=1&paused=1&quietMs=1&v=${Date.now()}`);
  const why = await p2.waitForFunction(() => document.querySelector('.loading-screen .load-why')?.textContent, undefined, { timeout: 120000, polling: 250 }).then((h) => h.jsonValue());
  assert.ok(why.length > 0, 'a reason under the failure sentence');
  assert.ok(logged.some((l) => l.includes(why)), `the reason in the console: ${JSON.stringify(logged)}`);
  console.log(`PASS  the failure screen gives its reason  ("${why}")`);
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message.split('\n')[0]);
} finally {
  await close();
  child.kill();
}
console.log(failed ? 'loadfail: FAILED' : 'loadfail: PASSED');
process.exit(failed ? 1 : 0);
