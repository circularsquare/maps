// T-3160: what the Tokyo real-network start takes on an emulated iPhone (an iOS user agent, 390 x 844 at 3x, touch, no
// navigator.deviceMemory, as Safari), run against a production build (the dev server overstates a phone by about
// 0.4 GB, notes/T-3160.md stage 3): `perf_bench.mjs build <tree>` prints one, `static-serve.mjs <dist> <port>` serves it.
//   node trainworld/checks/t3160-phone-start.mjs http://127.0.0.1:6212/ [label]
// A new game, then data/packs/real/tyo.save posted as an import (the method of stages 3-6, so the aux worker also holds
// the new game's New York); the demand final; 30 s more. Every 3 s: Chrome's private bytes (all processes, the tab) and
// each demand version's rail trips as it publishes. At the final and 30 s after it: each context's JS heap, buffers and
// WASM after a collection, and each demand worker's WASM, heap in use and the peak in use. Cores from the ledger
// (TW_CORES_WANT, 2 by default). Writes ../artifacts/trainworld/t3160/<label>.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';
import { chromeMemory, contexts, rawCdp } from './lib/memory.mjs';
import { STATE } from './lib/reads.mjs';

process.env.TW_CORES_WANT ??= '2';
const BASE = process.argv[2] ?? 'http://127.0.0.1:6212/';
const LABEL = process.argv[3] ?? 'run';
const SAVE = new URL('../data/packs/real/tyo.save', import.meta.url);
const OUT = join(ARTIFACTS, 't3160');
const PORT = 9300 + Math.floor(Math.random() * 500);
const IOS = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1';

await mkdir(OUT, { recursive: true });
const chrome = await launchChrome({ prefix: 'tw-t3160-', size: [390, 844], args: [...GL, `--remote-debugging-port=${PORT}`], deviceScaleFactor: 3, isMobile: true, hasTouch: true, userAgent: IOS });
await chrome.ctx.addInitScript(() => { try { delete Navigator.prototype.deviceMemory; } catch {} });
const raw = await rawCdp(PORT);
const errors = [], stages = [], timeline = [], versions = [];
let stage = 'new game', sampling = true, page = null;
const t0 = Date.now();
const secs = () => Math.round((Date.now() - t0) / 100) / 10;
const sampler = (async () => {
  while (sampling) {
    try {
      const m = await chromeMemory(chrome.profile);
      timeline.push({ s: secs(), stage, ...m });
      const v = page && await page.evaluate(STATE.trips).catch(() => null);
      const last = versions[versions.length - 1];
      if (v && v.railTripsPerDay !== null && (!last || last.version !== v.version || last.round !== v.round || last.final !== v.final)) {
        versions.push({ s: secs(), ...v });
        console.log(`  ${secs()} s: v${v.version} round ${v.round}${v.final ? ' final' : ''}, rail ${v.railTripsPerDay}`);
      }
    } catch { /* the browser is closing */ }
    await new Promise((r) => setTimeout(r, 3000));
  }
})();
try {
  page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  const crashed = new Promise((r) => page.on('crash', () => r('crash')));
  const cdp = await chrome.ctx.newCDPSession(page);
  const measure = async (where) => {
    await cdp.send('HeapProfiler.collectGarbage').catch(() => {});
    await new Promise((r) => setTimeout(r, 500));
    const demand = await page.evaluate(() => window.twDemand?.client?.memory?.()).catch(() => null);
    const r = { where, s: secs(), ...(await chromeMemory(chrome.profile)), ctx: await contexts(raw, BASE, true), demand };
    stages.push(r);
    console.log(`${where} (${r.s} s): all ${r.total}, tab ${r.renderer}, gpu ${r.gpu} MB`);
    console.log('    used+buffers+wasm', r.ctx.map(([n, u, b, w]) => `${n} ${u}+${b}+${w}`).join(', '));
    console.log('    demand workers', JSON.stringify(demand));
  };
  await page.goto(`${BASE}?debug=1&new=1&v=${Date.now()}`);
  await page.waitForFunction(STATE.ready, undefined, { timeout: 300000 });
  const device = await page.evaluate(() => ({ ua: navigator.userAgent.slice(0, 40), deviceMemory: navigator.deviceMemory ?? null, dpr: devicePixelRatio, touch: navigator.maxTouchPoints, workers: new URLSearchParams(location.search).get('demandWorkers') }));
  console.log('device', JSON.stringify(device));
  await page.waitForTimeout(3000);
  stage = 'tokyo load';
  const tImport = secs();
  await importSave(page, await readFile(SAVE));
  const loaded = page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  if ((await Promise.race([loaded, crashed])) === 'crash') throw new Error('page crashed during the load');
  const tLoaded = secs();
  const final = demandFinal(page, { timeout: 900000 });
  if ((await Promise.race([final, crashed])) === 'crash') throw new Error('page crashed before the demand final');
  const tFinal = secs();
  stage = 'after the final';
  const trips = await page.evaluate(STATE.trips);
  console.log(`import ${tImport} s, game up ${tLoaded} s, trips calculated ${tFinal} s (${Math.round((tFinal - tImport) * 10) / 10} s after the import): ${JSON.stringify(trips)}`);
  await measure('final');
  await page.waitForTimeout(30000);
  await measure('30 s after the final');
  const load = timeline.filter((x) => x.stage !== 'new game');
  const peak = (k) => load.reduce((a, x) => (x[k] > a[k] ? x : a), load[0]);
  console.log(`peak through the load: all ${peak('total').total} MB at ${peak('total').s} s, tab ${peak('renderer').renderer} MB at ${peak('renderer').s} s`);
  console.log('errors', errors.length, errors.slice(0, 5).join(' | '));
  await writeFile(join(OUT, `${LABEL}.json`), JSON.stringify({ label: LABEL, base: BASE, device, tImport, tLoaded, tFinal, trips, versions, peak: { total: peak('total'), tab: peak('renderer') }, stages, timeline, errors }, null, 1));
} catch (e) {
  console.log('FAILED', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  sampling = false;
  await sampler;
  raw.close();
  await chrome.close();
}
