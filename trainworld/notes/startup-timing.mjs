// Startup timing (T-172/T-174's method, docs/adding-a-city.md step 9): a production build in a
// scratch folder served by notes/static-serve.mjs (Range, no-store), headless Chrome with its own
// temporary profile, default demand pool. A new game RUNS times, then the real NYC save
// (T-007-real-nyc.save) RUNS times. Marks are ms since navigation: world = track drawn, clock =
// clock released (first estimate), r3 = crowding round 3. Writes OUT/startup-LABEL.json.
//   PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs URL=http://127.0.0.1:6101/ LABEL=before RUNS=5 OUT=<scratch> node notes/startup-timing.mjs
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.URL ?? 'http://127.0.0.1:6101/';
const label = process.env.LABEL ?? 'run';
const runs = Number(process.env.RUNS ?? 5);
const OUT = process.env.OUT ?? '.';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? OUT, 'chrome-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const probe = () => {
  window.__t = {};
  const mark = (k) => { if (!(k in window.__t)) window.__t[k] = Math.round(performance.now()); };
  const id = setInterval(() => {
    if (window.tw?.world?.value) mark('world');
    const v = window.twDemand?.view?.();
    if (v) mark('r' + v.round);
    if (window.__t.world !== undefined && window.tw?.clock?.held && window.tw.clock.held.peek() === false && window.__t.r0 !== undefined) mark('clock');
    if (v?.round === 3 && window.__t.clock !== undefined) clearInterval(id);
  }, 5);
};
await ctx.addInitScript(probe);
const errors = [];
const page = await ctx.newPage();
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const waitDone = () => page.waitForFunction(() => window.__t?.r3 !== undefined && window.__t.clock !== undefined, undefined, { timeout: 240000 });
const read = () => page.evaluate(() => ({
  marks: window.__t,
  cities: [...(window.twDemand.cities?.() ?? new Map()).keys()],
  lines: window.twDemand.view()?.lines.size,
  loads: JSON.parse(JSON.stringify(window.twDemand.client?.start?.loads ?? [])),
}));
const result = { label, base, newGame: [], nycSave: [], errors };
try {
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1&new=1&v=${Date.now()}`);
    await waitDone();
    result.newGame.push(await read());
  }
  const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
  const before = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => (window.twDemand.view()?.lines.size ?? 0) > 100, undefined, { timeout: 240000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1&v=${Date.now()}`);
    await waitDone();
    result.nycSave.push(await read());
  }
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const brief = (r) => r.map((x) => ({ world: x.marks.world, clock: x.marks.clock, r0: x.marks.r0, r3: x.marks.r3, cities: x.cities.join('+'), lines: x.lines }));
console.log(JSON.stringify({ label, newGame: brief(result.newGame), nycSave: brief(result.nycSave), errors }, null, 1));
await writeFile(join(OUT, `startup-${label}.json`), JSON.stringify(result, null, 1));
