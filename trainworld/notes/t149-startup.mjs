// T-149: time to first interactive at startup, for a new game and for the real New York save
// (T-007-real-nyc.save, 109 lines), on a served build of the app. Isolated, temporary Chrome
// profile; nothing of Anita's is touched. Run from the maps root:
//   URL=http://localhost:5341/ LABEL=after WORKERS=3 node trainworld/notes/t149-startup.mjs
// PLAYWRIGHT_MODULE may point at a playwright index.mjs. Writes artifacts/trainworld/notes-outputs/T-149-<LABEL>.json.
//
// Marks are performance.now() in the page (ms since navigation): `world` the clock worker's
// first state (the track is drawn), `r0`..`r3` the demand result published at each round,
// `ready` the loading screen gone (after T-149 only; before T-149 nothing blocked input).
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const base = process.env.URL ?? 'http://localhost:8800/trainworld/';
const label = process.env.LABEL ?? 'run';
const workers = process.env.WORKERS ? `&demandWorkers=${process.env.WORKERS}` : '';
const runs = Number(process.env.RUNS ?? 3);
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't149-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const probe = () => {
  window.__t = {};
  const mark = (k) => { if (!(k in window.__t)) window.__t[k] = Math.round(performance.now()); };
  const id = setInterval(() => {
    if (window.tw?.world?.value) mark('world');
    const v = window.twDemand?.view?.();
    if (v) mark('r' + v.round);
    const gate = document.querySelector('.loading-screen');
    if (gate) { mark('gateShown'); window.__pct = [...(window.__pct ?? []), gate.textContent]; }
    if (window.__t.gateShown !== undefined && !gate) mark('ready');
    if (v?.round === 3 && window.__t.world !== undefined && (!window.__t.gateShown || window.__t.ready !== undefined)) clearInterval(id);
  }, 5);
};
await ctx.addInitScript(probe);
const errors = [];
const page = await ctx.newPage();
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const waitDone = () => page.waitForFunction(() => window.__t?.r3 !== undefined && window.__t.world !== undefined && (window.__t.gateShown === undefined || window.__t.ready !== undefined), undefined, { timeout: 180000 });
const read = () => page.evaluate(() => ({
  marks: window.__t,
  cities: [...(window.twDemand.cities?.() ?? new Map()).keys()],
  lines: window.twDemand.view()?.lines.size,
  start: JSON.parse(JSON.stringify(window.twDemand.start)),
  steps: [...new Set(window.__pct ?? [])].slice(0, 400),
  clockMinute: window.tw.clock.minute.peek(),
}));
const result = { label, base, workers: process.env.WORKERS ?? 'default', newGame: [], nycSave: [], errors };
try {
  // a new game, several times (the first run also warms the HTTP cache like a returning player)
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1&new=1${workers}&v=${Date.now()}`);
    await waitDone();
    result.newGame.push(await read());
  }
  // put the real New York network in the autosave, then start the page on it
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  const before = await page.evaluate(() => window.tw.lastSaved.peek());
  await page.evaluate((data) => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => (window.twDemand.view()?.lines.size ?? 0) > 100, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
  for (let i = 0; i < runs; i++) {
    await page.goto(`${base}?debug=1${workers}&v=${Date.now()}`);
    await waitDone();
    result.nycSave.push(await read());
  }
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const brief = (r) => r.map((x) => ({ ...x.marks, cities: x.cities.length, lines: x.lines }));
console.log(JSON.stringify({ label, newGame: brief(result.newGame), nycSave: brief(result.nycSave), errors }, null, 1));
await writeFile(`artifacts/trainworld/notes-outputs/T-149-${label}.json`, JSON.stringify(result, null, 1));
