// T-229 browser check: how long changing a line's train count takes, and what blocks the page.
// On New York's real network (T-007 save) with the clock running: the subway G selected, its
// high-demand stepper clicked through the inspector, and for each click: when the stepper's number
// changed (from the click), the clock worker's time by step (`client.editLog`), the main thread's
// long tasks (over 50 ms) in the next 8 s, and when demand's new result landed.
// Needs a Vite dev server (BASE, default http://localhost:6932; file watching off) and Playwright
// (PLAYWRIGHT_MODULE). PROFILE_DIR: where the throwaway Chrome profile goes. LABEL names the run.
//   node trainworld/notes/t229-browser.mjs
// Results: artifacts/trainworld/notes-outputs/t229-browser-<LABEL>.json; screenshot artifacts/trainworld/T-229-<LABEL>.png.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6932';
const LABEL = process.env.LABEL ?? 'run';
const LINE = process.env.LINE ?? 'subway G';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't229-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = { label: LABEL, clicks: [] };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round === 3, undefined, { timeout: 300000 });
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size > 100, v0, { timeout: 600000 });
  // riders on every line's account, then the line selected and the clock running at 08:10
  await page.waitForTimeout(3000);
  const line = await page.evaluate(name => {
    const tw = window.tw, l = tw.world.value.save.lines.find(l => l.name === name);
    const pts = l.stops.map(s => tw.world.value.save.stations.find(st => st.id === s));
    tw.map.jumpTo({ center: [pts[Math.floor(pts.length / 2)].lng, pts[Math.floor(pts.length / 2)].lat], zoom: 12.5 });
    tw.select({ kind: 'line', line: l.id });
    tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8 * 3600 + 600);
    tw.clock.setSpeed(0);
    tw.clock.setRunning(true);
    return { id: l.id, name: l.name };
  }, LINE);
  out.line = line;
  await page.waitForTimeout(4000);
  // long tasks on the main thread, from now on
  await page.evaluate(() => {
    window.__long = [];
    new PerformanceObserver(l => { for (const e of l.getEntries()) window.__long.push({ at: e.startTime, ms: e.duration }); }).observe({ type: 'longtask', buffered: false });
  });
  const button = page.getByRole('button', { name: 'More trains on the entire line' }).first();
  const value = () => page.locator('.stepper .v').first().textContent();
  for (let k = 0; k < 3; k++) {
    const before = await value();
    const dv = await page.evaluate(() => window.twDemand.view().version);
    const t0 = await page.evaluate(() => { window.__t0 = performance.now(); window.__seen = null; const el = document.querySelector('.stepper .v'); const was = el.textContent; const mo = new MutationObserver(() => { if (el.textContent !== was && window.__seen === null) window.__seen = performance.now() - window.__t0; }); mo.observe(el, { characterData: true, childList: true, subtree: true }); return window.__t0; });
    await page.evaluate(() => window.tw.reset());
    await button.click();
    await page.waitForFunction(() => window.__seen !== null, undefined, { timeout: 30000 });
    const seen = await page.evaluate(() => window.__seen);
    let demandMs = null;
    try {
      await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3, dv, { timeout: 120000 });
      demandMs = await page.evaluate(t0 => performance.now() - t0, t0);
    } catch { /* no new result */ }
    await page.waitForTimeout(3000);
    const res = await page.evaluate(t0 => ({
      long: window.__long.filter(e => e.at >= t0).map(e => ({ at: Math.round(e.at - t0), ms: Math.round(e.ms) })),
      edit: window.tw.client.editLog.at(-1),
      frames: (() => { const s = window.tw.stats(); return { draws: s.draws, gapMs: +s.gap.toFixed(1), worstGapMs: +window.tw.overlay.frameGap.max().toFixed(1) }; })(),
    }), t0);
    out.clicks.push({ before, after: await value(), stepperMs: Math.round(seen), demandMs: demandMs && Math.round(demandMs), ...res });
    console.log(JSON.stringify(out.clicks.at(-1)));
  }
  // two quick clicks: both must count
  const before = Number(await value());
  await button.click();
  await button.click();
  await page.waitForTimeout(2500);
  out.doubleClick = { before, after: Number(await value()) };
  await page.screenshot({ path: join(ART, `T-229-${LABEL}.png`) });
  out.errors = errors.filter(e => !/favicon|DevTools/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', `t229-browser-${LABEL}.json`), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
