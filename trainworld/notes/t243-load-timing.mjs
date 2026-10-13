// T-243: where does startup time go, and how does the loading card's progress track it?
// Samples the card (text, and the bar's fill once it has one) every 25 ms from navigation until
// the clock is released, plus the demand client's load log and solve log.
// Needs Playwright (PLAYWRIGHT_MODULE, e.g. the npx cache's playwright/index.mjs) and OUT, a scratch
// folder for the throwaway Chrome profile and the results (load-LABEL.json). notes/T-243.md.
//   BASE=http://localhost:8800/trainworld/ LABEL=before RUNS=3 OUT=<scratch> node notes/t243-load-timing.mjs
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:8800/trainworld/';
const LABEL = process.env.LABEL ?? 'run';
const RUNS = Number(process.env.RUNS ?? 3);
const HERE = process.env.OUT ?? '.';
const profile = await mkdtemp(join(HERE, 'chrome-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
await ctx.addInitScript(() => {
  window.__s = [];
  let last = '';
  const id = setInterval(() => {
    const c = document.querySelector('.loading-card');
    const fill = document.querySelector('.loading-card .load-fill');
    const txt = c ? c.innerText.replace(/\s+/g, ' ') + (fill ? ' | ' + fill.style.width : '') : '';
    if (window.tw?.world?.value && window.__world === undefined) window.__world = Math.round(performance.now());
    if (txt !== last) { window.__s.push([Math.round(performance.now()), txt]); last = txt; }
    if (window.__world !== undefined && window.tw?.clock?.held?.peek?.() === false && window.__clock === undefined) { window.__clock = Math.round(performance.now()); clearInterval(id); }
  }, 25);
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', e => errors.push(e.message));
const read = () => page.evaluate(() => ({
  world: window.__world, clock: window.__clock, samples: window.__s,
  loads: JSON.parse(JSON.stringify(window.twDemand.start.loads)),
  workers: JSON.parse(JSON.stringify(window.twDemand.start.workers)),
  solves: window.twDemand.log.map(s => ({ v: s.version, arrived: Math.round(s.arrivedAt), sent: Math.round(s.sentAt), first: s.firstAt && Math.round(s.firstAt), cities: s.cities, periodMs: s.periodMs, setupMs: s.setupMs })),
}));
const wait = () => page.waitForFunction(() => window.__clock !== undefined, undefined, { timeout: 300000 });
const out = { label: LABEL, base: BASE, newGame: [], nycSave: [], errors };
try {
  for (let i = 0; i < RUNS; i++) {
    await page.goto(`${BASE}?debug=1&new=1&v=${Date.now()}`);
    await wait();
    out.newGame.push(await read());
    console.log('new', i, out.newGame.at(-1).world, out.newGame.at(-1).clock);
  }
  const bytes = await readFile('C:/Users/anita/projects/maps/trainworld/T-007-real-nyc.save');
  await page.evaluate(data => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 100, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  for (let i = 0; i < RUNS; i++) {
    await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
    await wait();
    out.nycSave.push(await read());
    console.log('nyc', i, out.nycSave.at(-1).world, out.nycSave.at(-1).clock);
  }
} finally {
  await ctx.close();
}
await writeFile(join(HERE, `load-${LABEL}.json`), JSON.stringify(out, null, 1));
for (const k of ['newGame', 'nycSave']) for (const r of out[k]) {
  console.log(`\n${k} world ${r.world} clock ${r.clock}`);
  for (const l of r.loads) console.log('  load', l.item, Math.round(l.requestedAt), Math.round(l.downloadedAt), Math.round(l.openedAt));
  for (const s of r.solves) console.log('  solve', s.v, 'arrived', s.arrived, 'sent', s.sent, 'first', s.first, JSON.stringify(s.periodMs), 'setup', JSON.stringify(s.setupMs));
  console.log('  workers', JSON.stringify(r.workers.map(w => [Math.round(w.wasmMs), Math.round(w.openMs), Math.round(w.warmMs)])));
  for (const [t, txt] of r.samples) console.log('   ', t, txt);
}
console.log('errors', errors.slice(0, 5));
