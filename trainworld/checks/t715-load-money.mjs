// T-715: loading a game never charges or freezes its money. Part 1: the real save in a slot, a copy of it run on three game days, then the original opened from the saves panel while the
// later copy is open, twice (before T-715 the later game's clock reached the worker after the load,
// so the gap's running costs were charged with no fares and the money then stood still until the
// clock caught up). Each time the cash must load as saved, move while the clock runs with fares coming
// in (the same network's income is sent again after a load), and no ledger day may lie after the
// clock. Part 2: a save whose ledgers run three days past its clock (as two such loads left them, each
// day's running cost twice) loads with those days gone and both charges back in the cash.
//   node trainworld/checks/t715-load-money.mjs [url]     (from the maps root; default localhost:6165)
// Prints PASS or FAIL lines and exits 1 on a failure. About two minutes.
import { readFile, mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6165/';
const SAVE = fileURLToPath(new URL('../T-007-real-nyc.save', import.meta.url));

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
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

/** The real save with day and hour ledger entries for the three days after its clock as two such
 * loads leave them (no fares, each day's running cost twice, a build charge on the second), the last
 * hour's running 0 so they end on its hour. */
function damagedSave(file, daily) {
  const raw = gunzipSync(file);
  const n = raw.readUInt32LE(4);
  const h = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
  const D = Math.floor(h.clock / 86400), H = Math.floor(h.clock / 3600);
  const day = (d, extra = {}) => ({ day: d, fares: 0, running: 2 * daily, trains: 0, build: 0, rides: 0, ...extra });
  h.ledger = [...(h.ledger ?? []).filter((d) => d.day <= D), day(D + 1), day(D + 2, { build: 50 }), day(D + 3)];
  const hours = [];
  for (let k = H + 1; k <= H + 73; k++) hours.push({ hour: k, fares: 0, running: k === H + 73 ? 0 : 5, trains: 0, build: 0, rides: 0 });
  h.hours = [...(h.hours ?? []).filter((e) => e.hour <= H), ...hours];
  const json = Buffer.from(JSON.stringify(h), 'utf8');
  const head = Buffer.alloc(8);
  head.write('TWG1', 0, 'latin1');
  head.writeUInt32LE(json.length, 4);
  return { bytes: gzipSync(Buffer.concat([head, json, raw.subarray(8 + n)])), cash: h.cash, day: D };
}

const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-t715-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1200, height: 800 }, args: ['--use-angle=d3d11'] });
try {
  const page = await ctx.newPage();
  const wait = (ms) => page.waitForTimeout(ms);
  // the cash as the top bar shows it, the clock's day and the days the ledger has money on
  const money = () => page.evaluate(async () => {
    const M = await import('/src/game/money.ts');
    const m = window.tw.moneyState.value, t = window.tw.clock.now();
    return { cash: M.liveMoney.value.cash, fares: M.liveMoney.value.today?.fares ?? 0, day: Math.floor(t / 86400), days: m.ledger.filter((d) => d.fares || d.running || d.trains || d.build).map((d) => d.day) };
  });
  // the cash a slot was last written with (its saves-list summary)
  const slotCash = (key) => page.evaluate(async (key) => (await (await import('/src/workers/saveFile.ts')).listSlots()).find((s) => s.key === key)?.cash, key);
  const settled = () => page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return v && v.final && !d.client.waiting && !window.tw.clock.held.peek(); }, undefined, { timeout: 300000, polling: 300 });
  const pause = async () => { await page.evaluate(() => window.tw.clock.setRunning(false)); await wait(500); };
  const open = async (key) => {
    await page.evaluate((key) => { window.__load = window.tw.client.lastLoad; window.tw.client.post({ kind: 'saveSlot', action: 'open', key }); }, key);
    await page.waitForFunction(() => window.tw.client.lastLoad !== window.__load, undefined, { timeout: 120000 });
    await settled();
    await wait(500);
  };
  const importSave = async (bytes) => {
    await page.evaluate((data) => { window.__load = window.tw.client.lastLoad; const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
    await page.waitForFunction(() => window.tw.client.lastLoad !== window.__load && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  };

  // Part 1: the real save, loaded into a slot of its own (T-743)
  await page.goto(`${BASE}?debug=1`);
  await page.waitForSelector('.start-menu .start-new', { timeout: 60000 });
  await page.click('.start-menu .start-new');
  await page.click('.start-menu .new-game-cities [data-city="nyc"]'); // T-728: the new game sheet
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(await readFile(SAVE));
  await settled();
  const realKey = await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
  await pause();
  await page.evaluate(() => window.tw.client.post({ kind: 'saveSlot', action: 'create', name: 'later' }));
  await wait(2000);
  const laterKey = await page.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots()).find((s) => s.name === 'later')?.key);
  await page.evaluate(async () => { const c = await import('/src/game/clock.ts'); c.SPEEDS[2] = 20000; window.tw.clock.setSpeed(2); });
  await page.waitForFunction(() => window.tw.clock.now() > 4.5 * 86400, undefined, { timeout: 60000, polling: 200 });
  await page.evaluate(() => window.tw.clock.setSpeed(0));
  await pause();
  for (let round = 1; round <= 2; round++) {
    const saved = await slotCash(realKey);
    await open(realKey);
    const a = await money();
    await page.evaluate(() => window.tw.clock.setRunning(true));
    await wait(3000);
    await pause();
    const b = await money();
    const future = b.days.filter((d) => d > b.day);
    report(`original opened from a later game, ${round}`, Math.abs(a.cash - saved) < 0.01 && b.cash !== a.cash && b.fares > a.fares && future.length === 0,
      `saved ${saved.toFixed(2)}, loaded ${a.cash.toFixed(2)}, after 3 s ${b.cash.toFixed(2)} with fares today ${a.fares.toFixed(2)} to ${b.fares.toFixed(2)} (US$M), clock day ${b.day}, ledger days after it [${future.join(' ')}]`);
    if (round < 2) await open(laterKey);
  }

  // Part 2: a save already damaged that way, twice over; the real save's day of running costs first
  const run = await page.evaluate(async () => {
    const m = window.tw.moneyState.value, P = await import('/src/game/periods.ts');
    const LEVEL = { high: 0, medium: 1, low: 2 };
    let daily = 0;
    for (let h = 0; h < 24; h++) daily += m.runningPerHour[LEVEL[P.PERIODS[P.periodAt(h * 3600)].demand]];
    return { daily, most: Math.max(...m.runningPerHour) };
  });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bad = damagedSave(await readFile(SAVE), run.daily);
  await importSave(bad.bytes);
  await wait(1000);
  const r = await page.evaluate(() => { const m = window.tw.moneyState.value; return { cash: m.cash, days: m.ledger.map((d) => d.day), build: m.ledger.reduce((s, d) => s + d.build, 0) }; });
  const back = r.cash - bad.cash, want = 2 * 3 * run.daily;
  report('a save with ledger days after its clock', Math.abs(back - want) <= 2 * run.most + 0.01 && Math.max(...r.days) <= bad.day && r.build >= 50,
    `cash back ${back.toFixed(2)}, twice three days' running ${want.toFixed(2)} (US$M, within two hours' ${(2 * run.most).toFixed(2)}), ledger days [${r.days.join(' ')}], clock day ${bad.day}, build kept ${r.build.toFixed(0)}`);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(failed ? 1 : 0);
