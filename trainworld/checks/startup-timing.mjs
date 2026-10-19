// Startup timing (T-172/T-174's method, docs/adding-a-city.md step 9): a production build in a
// scratch folder served by checks/static-serve.mjs (Range, no-store), headless Chrome with its own
// temporary profile, default demand pool. A new game RUNS times, then the real NYC save
// (T-007-real-nyc.save) RUNS times. Marks are ms since navigation: world = track drawn, clock =
// clock released (the fresh final, T-347), final = the first final view, r3 = crowding round 3 (a
// new game's bare view has none). Writes OUT/startup-LABEL.json.
// MENU=1 (T-677): every load goes through the start menu, New game or the save's card clicked as soon
// as the menu is up; `pick` is the click, and `fromPick` the marks less it. Without it the loads
// skip the menu (`?new=1`, `?continue=1`).
//   PLAYWRIGHT_MODULE=file:///.../playwright/index.mjs URL=http://127.0.0.1:6101/ LABEL=before RUNS=5 OUT=<scratch> node trainworld/checks/startup-timing.mjs
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.URL ?? 'http://127.0.0.1:6101/';
const label = process.env.LABEL ?? 'run';
const runs = Number(process.env.RUNS ?? 5);
const OUT = process.env.OUT ?? '.';
const MENU = process.env.MENU === '1';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? OUT, 'chrome-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const probe = () => {
  window.__t = {};
  const mark = (k) => { if (!(k in window.__t)) window.__t[k] = Math.round(performance.now()); };
  const id = setInterval(() => {
    if (window.tw?.world?.value) mark('world');
    const v = window.twDemand?.view?.();
    if (v) mark('r' + v.round);
    if (v?.final) mark('final');
    if (window.__t.world !== undefined && window.tw?.clock?.held && window.tw.clock.held.peek() === false && window.__t.final !== undefined) mark('clock');
    if (v?.round === 3 && window.__t.clock !== undefined) clearInterval(id);
  }, 5);
};
await ctx.addInitScript(probe);
const errors = [];
const page = await ctx.newPage();
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const waitDone = () => page.waitForFunction(() => window.__t?.final !== undefined && window.__t.clock !== undefined, undefined, { timeout: 240000 });
/** the menu's New game (`which` = 'new') or its first saved game ('save'), clicked once it is up */
async function pick(which) {
  const sel = which === 'new' ? '.start-menu .start-new' : '.start-menu .start-save';
  await page.waitForSelector(sel, { timeout: 60000 });
  if (which === 'new') { // T-728: New game opens the sheet; the pick is its Start, New York
    await page.click(sel);
    await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  }
  const go = which === 'new' ? '.start-menu .new-game .start-new' : sel;
  await page.evaluate((sel) => { window.__t.pick = Math.round(performance.now()); document.querySelector(sel).click(); }, go);
}
const read = async () => {
  await page.waitForTimeout(1000); // r3, if it comes soon after
  return page.evaluate(() => ({
    marks: window.__t,
    cities: [...(window.twDemand.cities?.() ?? new Map()).keys()],
    lines: window.twDemand.view()?.lines.size,
    loads: JSON.parse(JSON.stringify(window.twDemand.client?.start?.loads ?? [])),
  }));
};
const result = { label, base, menu: MENU, newGame: [], nycSave: [], errors };
try {
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1${MENU ? '' : '&new=1'}&v=${Date.now()}`);
    if (MENU) await pick('new');
    await waitDone();
    result.newGame.push(await read());
  }
  // the real save, imported into a game that owns the original slot
  await page.goto(`${base}?debug=1&continue=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 120000 });
  const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
  const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => (window.twDemand.view()?.lines.size ?? 0) > 100, undefined, { timeout: 240000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 60000 });
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1${MENU ? '' : '&continue=1'}&v=${Date.now()}`);
    if (MENU) await pick('save');
    await waitDone();
    result.nycSave.push(await read());
  }
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const less = (m, k) => (m[k] === undefined ? undefined : m[k] - (m.pick ?? 0));
const brief = (r) => r.map((x) => ({ pick: x.marks.pick, world: x.marks.world, clock: x.marks.clock, final: x.marks.final, r3: x.marks.r3,
  ...(x.marks.pick !== undefined ? { fromPick: { world: less(x.marks, 'world'), final: less(x.marks, 'final'), clock: less(x.marks, 'clock') } } : {}), cities: x.cities.join('+'), lines: x.lines }));
console.log(JSON.stringify({ label, menu: MENU, newGame: brief(result.newGame), nycSave: brief(result.nycSave), errors }, null, 1));
await writeFile(join(OUT, `startup-${label}.json`), JSON.stringify(result, null, 1));
