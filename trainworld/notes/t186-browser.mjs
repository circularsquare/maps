// T-186 browser check: zoomed far out with Homes on, every city is one longer-trips bubble (the
// focused one too); a click on one selects that city and draws one bubble per other city. On New
// York's real network (T-007 save) in the real app.
// Needs a Vite dev server (BASE) and Playwright (PLAYWRIGHT_MODULE). PROFILE_DIR: where the
// throwaway Chrome profile goes. Run from the maps root: node trainworld/notes/t186-browser.mjs.
// Screenshots: artifacts/trainworld/T-186-*.png. Results: artifacts/trainworld/notes-outputs/t186-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6571';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't186-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
const NYC = [-73.985, 40.758], PHL = [-75.1636, 39.9526];
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 300000 });
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100 && window.twDemand.intercity()?.round === 3, undefined, { timeout: 600000 });
  await page.evaluate(async () => {
    window.__iv = await import('/src/map/intercityView.ts');
    const d = document.querySelector('#debug'); if (d) d.style.display = 'none';
    window.tw.clock.setRunning(false);
    window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home', intercity: true, bubbleSize: 1 };
  });
  const state = () => page.evaluate(() => ({ zoom: window.tw.map.getZoom(), overview: window.__iv.cityOverview.value, sel: window.tw.selection.value, labels: [...document.querySelectorAll('.intercity-label')].map(e => e.textContent) }));
  const hover = (ll) => page.evaluate((ll) => { const p = window.tw.map.project(ll); return window.twViews.hover(p.x, p.y); }, ll);

  // 1. zoomed out between New York and Philadelphia: one bubble per city, New York included
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.3, 40.6], zoom: 7.6 }));
  await page.waitForTimeout(2500);
  const s1 = await state();
  result.overview = { ...s1, nyc: await hover(NYC), phl: await hover(PHL) };
  check('overview on at zoom 7.6', s1.overview, s1);
  check('New York, the focused city, has its own bubble', s1.labels.some(l => l.startsWith('New York')) && /between New York and other places/.test(result.overview.nyc?.[0] ?? ''), result.overview);
  await page.screenshot({ path: join(ART, 'T-186-1-overview.png') });

  // 2. a click on Philadelphia's bubble selects it; one bubble per other city
  const pp = await page.evaluate((ll) => { const p = window.tw.map.project(ll); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, PHL);
  await page.mouse.click(pp[0], pp[1]);
  await page.waitForTimeout(1500);
  const s2 = await state();
  result.philadelphia = { ...s2, nyc: await hover(NYC), inspector: await page.evaluate(() => document.querySelector('#inspector')?.innerText) };
  check('a click selects Philadelphia', s2.sel?.kind === 'city' && s2.sel.city === 'phl', s2.sel);
  check('New York is drawn as a destination of Philadelphia', /between Philadelphia and New York/.test(result.philadelphia.nyc?.[0] ?? ''), result.philadelphia.nyc);
  await page.screenshot({ path: join(ART, 'T-186-2-philadelphia.png') });

  // 3. a click on New York's bubble selects New York
  const pn = await page.evaluate((ll) => { const p = window.tw.map.project(ll); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, NYC);
  await page.mouse.click(pn[0], pn[1]);
  await page.waitForTimeout(1500);
  const s3 = await state();
  result.newYork = { ...s3, inspector: await page.evaluate(() => document.querySelector('#inspector')?.innerText) };
  check('a click on New York selects it', s3.sel?.kind === 'city' && s3.sel.city === 'nyc', s3.sel);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.2, 41.4], zoom: 7.2 }));
  await page.waitForTimeout(2000);
  await page.screenshot({ path: join(ART, 'T-186-3-newyork.png') });

  // 4. an empty click clears it; zoomed in past the threshold the city's own bubbles are back
  await page.keyboard.press('Escape');
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 9.0 }), NYC);
  await page.waitForTimeout(2500);
  const s4 = await state();
  check('cleared, and no overview at zoom 9', s4.sel === null && !s4.overview, s4);
  await page.screenshot({ path: join(ART, 'T-186-4-zoom9.png') });

  // 5. the whole region (the City pane's view)
  await page.evaluate(() => window.tw.map.fitBounds([[-77.74, 38.4], [-70.36, 42.86]], { padding: 40, duration: 0 }));
  await page.waitForTimeout(2500);
  const s5 = await state();
  result.region = s5;
  check('overview on for the whole region', s5.overview && s5.labels.length >= 8, s5);
  await page.screenshot({ path: join(ART, 'T-186-5-region.png') });

  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't186-browser.json'), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
