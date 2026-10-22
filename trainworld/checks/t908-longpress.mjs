// T-908: in a ?new=1 New York game with home bubbles on, real mouse events: a quick drag from a
// bubble pans; a press held 0.6 s (still, or jittering 2 or 4 px) then dragged draws a box that
// selects bubbles without panning; the controls panel opens, Esc closes it and keeps the selection,
// a press outside or its button closes it. Screenshots t908-*.png in artifacts/trainworld/help/.
//   node trainworld/checks/t908-longpress.mjs [url] [shotdir]      (about a minute)
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const BASE = process.argv[2] ?? 'http://localhost:6110/';
const SHOTS = process.argv[3] ?? 'C:/Users/anita/projects/maps/artifacts/trainworld/help';
const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
async function playwright() {
  try { return await import('playwright'); } catch {}
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no playwright');
}
const { chromium } = await playwright();
const profile = await mkdtemp(join(tmpdir(), 'tw-help-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: CHROME, headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist', '--renderer-process-limit=1'] });
const errors = [];
let fails = 0;
const report = (name, ok, d = '') => { if (!ok) fails++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${d}`); };
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError/.test(m.text())) errors.push(m.text()); });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { commuters: true, commuterEnd: 'home' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews, undefined, { timeout: 180000 });
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true }; window.tw.map.jumpTo({ center: [-73.985, 40.755], zoom: 13.2 }); });
  await page.waitForFunction(() => window.twViews.stats.commuters.probe?.(), undefined, { timeout: 180000, polling: 500 });
  await page.waitForTimeout(2500);
  const rect = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; });
  // a point on a bubble near the middle
  const at = await page.evaluate(({ w, h }) => {
    let best = null, bd = 1e9;
    for (let y = 100; y < h - 100; y += 8) for (let x = 200; x < w - 200; x += 8) {
      if (!window.twViews.hover(x, y)) continue;
      const d = Math.hypot(x - w / 2, y - h / 2);
      if (d < bd) { bd = d; best = [x, y]; }
    }
    return best;
  }, rect);
  if (!at) throw new Error('no bubble found');
  const X = rect.x + at[0], Y = rect.y + at[1];
  const center = () => page.evaluate(() => { const c = window.tw.map.getCenter(); return [c.lng, c.lat]; });
  const reset = () => page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-73.985, 40.755], zoom: 13.2 }); });

  // A. quick drag from a bubble: a pan, no selection
  await reset(); await page.waitForTimeout(500);
  let c0 = await center();
  await page.mouse.move(X, Y); await page.mouse.down();
  for (let k = 1; k <= 10; k++) { await page.mouse.move(X + 15 * k, Y + 8 * k); await page.waitForTimeout(16); }
  await page.mouse.up(); await page.waitForTimeout(400);
  let c1 = await center(), sel = await page.evaluate(() => window.tw.selection.value);
  report('quick drag pans', Math.hypot(c1[0] - c0[0], c1[1] - c0[1]) > 1e-4 && sel?.kind !== 'commuters', `moved ${(Math.hypot(c1[0] - c0[0], c1[1] - c0[1]) * 1e4).toFixed(1)}e-4, sel ${sel?.kind}`);

  // B. still press 600 ms then drag: a box, no pan
  for (const [name, jitter] of [['still', 0], ['jitter 2px', 2], ['jitter 4px', 4]]) {
    await reset(); await page.waitForTimeout(500);
    c0 = await center();
    await page.mouse.move(X, Y); await page.mouse.down();
    for (let t = 0; t < 12; t++) { await page.mouse.move(X + (t % 2 ? jitter : -jitter), Y + (t % 3 === 0 ? jitter : 0)); await page.waitForTimeout(50); }
    await page.mouse.move(X, Y);
    const armed = await page.evaluate(() => { const b = document.querySelector('.demand-box'); return b && b.style.display !== 'none'; });
    if (name === 'still') await page.screenshot({ path: join(SHOTS, 't908-armed.png') });
    for (let k = 1; k <= 10; k++) { await page.mouse.move(X + 18 * k, Y + 12 * k); await page.waitForTimeout(16); }
    if (name === 'still') await page.screenshot({ path: join(SHOTS, 't908-box.png') });
    await page.mouse.up(); await page.waitForTimeout(600);
    c1 = await center(); sel = await page.evaluate(() => { const s = window.tw.selection.value; return s && { kind: s.kind, areas: s.areas }; });
    const moved = Math.hypot(c1[0] - c0[0], c1[1] - c0[1]);
    report(`long press ${name} then drag`, moved < 1e-6 && sel?.kind === 'commuters' && sel.areas > 1, `armed ${armed}, moved ${(moved * 1e4).toFixed(2)}e-4, sel ${JSON.stringify(sel)}`);
  }
  // C. the controls panel: opens, Esc closes it and keeps the selection; a press outside closes it
  const selBefore = await page.evaluate(() => window.tw.selection.value?.kind);
  await page.click('.tab.help');
  await page.waitForTimeout(300);
  const shown = await page.$('.help-panel');
  await page.screenshot({ path: join(SHOTS, 't908-help.png') });
  await page.keyboard.press('Escape'); await page.waitForTimeout(200);
  const afterEsc = { panel: !!(await page.$('.help-panel')), sel: await page.evaluate(() => window.tw.selection.value?.kind) };
  report('help opens, Esc closes it only', !!shown && !afterEsc.panel && afterEsc.sel === selBefore, JSON.stringify({ selBefore, afterEsc }));
  await page.click('.tab.help'); await page.waitForTimeout(200);
  await page.mouse.click(rect.x + rect.w - 60, rect.y + rect.h - 60); await page.waitForTimeout(200);
  const afterOut = !!(await page.$('.help-panel'));
  await page.click('.tab.help'); await page.waitForTimeout(200);
  await page.click('.tab.help'); await page.waitForTimeout(200);
  const afterToggle = !!(await page.$('.help-panel'));
  report('help closes on a press outside and on its button', !afterOut && !afterToggle, `${afterOut} ${afterToggle}`);
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.click('.tab.help'); await page.waitForTimeout(400);
  await page.screenshot({ path: join(SHOTS, 't908-help-dark.png') });
  if (process.argv[4] === 'shot') await page.screenshot({ path: join(SHOTS, 't908-after.png') });
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('ran', false, e.message.split('\n')[0]);
} finally {
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
process.exit(fails ? 1 : 0);
