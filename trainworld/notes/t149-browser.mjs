// T-149 behaviour in the browser: the startup screen blocks keys and clicks and holds the clock;
// a pan to Boston (no construction there) loads it without blocking or holding the clock and
// leaves New York's numbers unchanged; loading another save holds the clock until its demand is
// complete and loads the cities its stations reach. Isolated temporary profile. From the maps root:
//   URL=http://localhost:5341/ node trainworld/notes/t149-browser.mjs
// Writes artifacts/trainworld/notes-outputs/T-149-browser.json and screenshots in ../artifacts/trainworld/.
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.URL ?? 'http://localhost:8800/trainworld/';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't149b-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = {}, errors = [];
const shots = 'artifacts/trainworld/';
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${base}?debug=1&new=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForSelector('.loading-screen');
  await page.waitForFunction(() => window.tw?.world.value, undefined, { timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: shots + 'T-149-loading.png' });
  // nothing gets through while it is up
  const before = await page.evaluate(() => ({ minute: window.tw.clock.minute.peek(), held: window.tw.clock.held.peek(), tool: document.querySelector('.tool-pick .btn.on')?.textContent ?? null }));
  await page.keyboard.press('1');
  await page.keyboard.press(' ');
  await page.mouse.click(650, 450);
  await page.mouse.wheel(0, -400);
  await page.waitForTimeout(2500);
  const during = await page.evaluate(() => ({ minute: window.tw.clock.minute.peek(), running: window.tw.clock.running.peek(), held: window.tw.clock.held.peek(), zoom: window.tw.map.getZoom(), screen: document.querySelector('.loading-screen')?.textContent }));
  assert.equal(before.held, true); assert.equal(during.minute, before.minute, 'clock held'); assert.equal(during.running, true, 'space did not pause');
  assert(Math.abs(during.zoom - 11.2) < 1e-6, 'wheel did not zoom');
  out.startup = { before, during };
  await page.waitForFunction(() => !document.querySelector('.loading-screen'), undefined, { timeout: 180000 });
  const ready = await page.evaluate(() => ({ round: window.twDemand.view().round, cities: [...window.twDemand.cities().keys()], held: window.tw.clock.held.peek(), minute: window.tw.clock.minute.peek() }));
  assert.equal(ready.round, 3); assert.deepEqual(ready.cities, ['nyc']); assert.equal(ready.held, false);
  await page.waitForTimeout(2500);
  const ticking = await page.evaluate(() => window.tw.clock.minute.peek());
  assert(ticking > ready.minute, 'clock runs after the screen goes');
  out.ready = { ...ready, minuteAfter2_5s: ticking };
  // a small line in Midtown, so New York's numbers are worth comparing
  await page.evaluate(async () => {
    const t = window.tw, run = async (op) => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    await run({ op: 'route', from: { kind: 'free', x: -3000, y: 0, level: -2 }, to: { kind: 'free', x: 3000, y: 0, level: -2 }, pis: [], single: false });
    const edge = t.world.value.edges.at(-1);
    await run({ op: 'addStation', at: { kind: 'node', node: edge.a }, platform: 200, name: 'West' });
    await run({ op: 'addStation', at: { kind: 'node', node: edge.b }, platform: 200, name: 'East' });
    const line = (await run({ op: 'addLine', stops: [edge.a, edge.b], name: 'Check', colour: '#17a488' })).line;
    await run({ op: 'setTrainCount', line, trains: [4, 4, 2] }); await run({ op: 'constructAll' });
  });
  await page.waitForFunction(() => window.twDemand.view()?.lines.size === 1 && window.twDemand.view().round === 3, undefined, { timeout: 180000 });
  const nycBefore = await page.evaluate(() => { const c = window.twDemand.cities().get('nyc'), v = window.twDemand.view(); return { split: c.split, rail: c.railTripsPerDay, line: [...v.lines.values()][0].ridersPerDay, version: v.version }; });
  // pan to Boston: it loads, the game keeps running, New York's numbers stay
  const m0 = await page.evaluate(() => window.tw.clock.minute.peek());
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-71.0589, 42.3601], zoom: 11.2 }));
  await page.waitForFunction(() => document.querySelector('.demand-busy')?.textContent?.startsWith('Loading Boston'), undefined, { timeout: 20000 });
  const busy = await page.evaluate(() => ({ text: document.querySelector('.demand-busy').textContent, held: window.tw.clock.held.peek(), screen: !!document.querySelector('.loading-screen'), pending: document.querySelector('.demand-selector .demand-pending')?.title }));
  const bar = await page.$('#topbar');
  await bar.screenshot({ path: shots + 'T-149-topbar-boston.png' });
  await page.waitForFunction(() => window.twDemand.cities().has('bos') && window.twDemand.view().round === 3 && !document.querySelector('.demand-busy')?.textContent?.startsWith('Loading'), undefined, { timeout: 180000 });
  const m1 = await page.evaluate(() => window.tw.clock.minute.peek());
  const nycAfter = await page.evaluate(() => { const c = window.twDemand.cities().get('nyc'), v = window.twDemand.view(); return { split: c.split, rail: c.railTripsPerDay, line: [...v.lines.values()][0].ridersPerDay, version: v.version, bos: window.twDemand.cities().get('bos').split }; });
  assert.equal(busy.held, false); assert.equal(busy.screen, false); assert(m1 > m0, 'clock ran during the Boston load');
  assert(nycAfter.version > nycBefore.version, 'solved again with Boston');
  assert.equal(nycAfter.rail, nycBefore.rail, 'New York rail unchanged'); assert.equal(nycAfter.line, nycBefore.line, 'line riders unchanged');
  assert.equal(nycAfter.bos.train, 0, 'Boston has no rail');
  out.pan = { busy, nycBefore, nycAfter, minutes: [m0, m1] };
  // another save: its clock waits for its demand, and its constructed cities load
  const bytes = await readFile('trainworld/T-125-northeast.save');
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.clock.held.peek(), undefined, { timeout: 20000 });
  const heldAt = await page.evaluate(() => ({ minute: window.tw.clock.minute.peek() }));
  await page.waitForFunction((v) => !window.tw.clock.held.peek() && window.twDemand.view().version > v, v0, { timeout: 300000 });
  const loaded = await page.evaluate(() => ({ round: window.twDemand.view().round, lines: window.twDemand.view().lines.size, cities: [...window.twDemand.cities().keys()], minute: window.tw.clock.minute.peek(), stations: window.tw.world.value.save.stations.filter((s) => s.built).length }));
  assert.equal(loaded.round, 3, 'released at the last round');
  out.importNortheast = { heldAt, loaded };
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
out.errors = errors;
console.log(JSON.stringify(out, null, 1));
await writeFile('artifacts/trainworld/notes-outputs/T-149-browser.json', JSON.stringify(out, null, 1));
