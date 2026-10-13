// T-209 to T-213 browser check (UI batch 13), on New York's real network (T-007 save), with real
// pointer moves, clicks and key presses:
// - T-212: the loading card and the top bar say "Loading data… N%", no city names;
// - T-209: every station complex draws exactly one demand circle in complex view (Penn, Grand
//   Central, Times Sq), and a player link (Times Sq with its 8 Av platforms) makes one circle and
//   one name label of the two;
// - T-210: no `title` attribute or SVG <title> anywhere; hovering the complex map's link line, the
//   pause button, a purpose bar and MapLibre's attribution button shows the game's own tag;
// - T-211: search is a magnifying glass; "/", Ctrl+F and a click open it, Escape and leaving it
//   empty close it; Ctrl+F in another text field is left to the browser;
// - T-213: the circle settings are in the Stations tab, not Settings.
// Needs a Vite dev server (BASE; the probes import /src/...) and Playwright (PLAYWRIGHT_MODULE, a
// file URL to its index.mjs). PROFILE_DIR: where the throwaway Chrome profile goes.
//   node trainworld/notes/t209-browser.mjs
// Screenshots: artifacts/trainworld/T-209-*.png ... T-213-*.png. Results: artifacts/trainworld/notes-outputs/t209-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:7023';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't209-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const out = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const importSave = async (bytes) => page.evaluate(data => { const b = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes: b }, [b]); }, Array.from(bytes));
  await importSave(await readFile(join(ROOT, 'T-007-real-nyc.save')));

  // ---- T-212: the loading card while the clock is held ----
  const card = await page.waitForFunction(() => document.querySelector('.loading-card .big')?.textContent?.includes('%') && document.querySelector('.loading-card').textContent, undefined, { timeout: 60000, polling: 50 }).then(h => h.jsonValue()).catch(() => null);
  if (card) await page.screenshot({ path: join(ART, 'T-212-card.png') });
  out.card = card;
  log('card', card);
  await page.waitForFunction(() => { const d = window.twDemand.view(); return d?.round >= 1 && d.lines.size > 100; }, undefined, { timeout: 900000 });
  log('solved');
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });
  // a city loaded later (a pan to Boston): the top bar's status
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-71.06, 42.36], zoom: 11 }));
  const bar = await page.waitForFunction(() => { const t = document.querySelector('.demand-busy')?.textContent ?? ''; return t.startsWith('Loading') && t; }, undefined, { timeout: 60000, polling: 50 }).then(h => h.jsonValue()).catch(() => null);
  if (bar) {
    const b = await page.$('.demand-busy');
    const r = await b.boundingBox();
    await page.mouse.move(r.x + r.width / 2, r.y + r.height / 2);
    await page.mouse.move(r.x + r.width / 2 + 2, r.y + r.height / 2);
    await page.waitForTimeout(100);
    out.barTip = await page.evaluate(() => document.querySelector('.ui-tag')?.style.display === '' ? document.querySelector('.ui-tag').textContent : null);
    await page.screenshot({ path: join(ART, 'T-212-topbar.png'), clip: { x: 330, y: 0, width: 1070, height: 120 } });
  }
  out.bar = bar;
  log('top bar', bar, out.barTip);
  await page.mouse.move(700, 600);

  // ---- T-209: one circle per complex ----
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, complexView: true, stationRiders: true, commuters: false }; });
  await page.waitForTimeout(1200);
  const circles = () => page.evaluate(async () => {
    const cv = await import('/src/game/complexView.ts');
    const c = cv.complexes.value, R = window.tw.renderer, n = R.net, sz = R.stationSizes, w = window.tw.world.value;
    const idx = new Map(n.stationIds.map((id, i) => [id, i]));
    const built = id => (Math.round(n.stations[idx.get(id) * 3 + 2]) & 2) === 0;
    let complexes = 0, wrong = [];
    c.members.forEach((m, k) => {
      if (!m.some(built)) return;
      complexes++;
      const k2 = m.filter(id => sz[idx.get(id) * 2 + 1] > 0).length;
      if (k2 > 1) wrong.push({ k, n: m.length, circles: k2 });
    });
    const named = name => { const s = w.save.stations.find(t => t.name === name); const m = c.members[c.of.get(s.id)]; return { n: m.length, circles: m.filter(id => sz[idx.get(id) * 2 + 1] > 0).length }; };
    return { complexes, wrong, penn: named('34 St-Penn Station'), grandCentral: named('Grand Central-42 St'), timesSq: named('Times Sq-42 St/42 St-Port Authority Bus Terminal') };
  });
  out.before = await circles();
  log('circles', JSON.stringify(out.before));
  // Times Sq's two same-named complexes (its 8 Av platforms are 330 m away)
  const ts = await page.evaluate(async () => {
    const cv = await import('/src/game/complexView.ts');
    const c = cv.complexes.value, w = window.tw.world.value;
    const name = 'Times Sq-42 St/42 St-Port Authority Bus Terminal';
    const ks = [...new Set(w.save.stations.filter(s => s.name === name).map(s => c.of.get(s.id)))].sort((a, b) => c.members[b].length - c.members[a].length);
    const st = id => w.save.stations.find(s => s.id === id);
    const a = st(c.members[ks[0]][0]), b = st(c.members[ks[1]][0]);
    return { a: { id: a.id, num: a.num, lng: a.lng, lat: a.lat }, b: { id: b.id, num: b.num, lng: b.lng, lat: b.lat }, sizes: ks.map(k => c.members[k].length) };
  });
  out.timesSqSplit = ts.sizes;
  const labelsAt = () => page.evaluate(() => [...document.querySelectorAll('.stn-label')].filter(e => e.textContent.startsWith('Times Sq-42 St/42') && e.style.display !== 'none' && e.getBoundingClientRect().width > 0).length);
  await page.evaluate(({ a, b }) => window.tw.map.jumpTo({ center: [(a.lng + b.lng) / 2, (a.lat + b.lat) / 2], zoom: 15.3 }), ts);
  await page.waitForTimeout(1500);
  out.labelsBeforeLink = await labelsAt();
  await page.screenshot({ path: join(ART, 'T-209-timessq-two-complexes.png') });
  const version0 = await page.evaluate(() => window.twDemand.view().version);
  out.linkEdit = await page.evaluate(async ({ a, b }) => JSON.stringify(await (await import('/src/game/world.ts')).edit({ op: 'link', a: a.num, b: b.num, on: true })), ts);
  log('link', out.linkEdit, JSON.stringify(ts), errors.slice(-3));
  await page.waitForFunction(() => window.tw.world.value.save.links.length === 1, undefined, { timeout: 10000 });
  await page.waitForTimeout(1500);
  out.afterLink = await circles();
  out.labelsAfterLink = await labelsAt();
  log('after link', JSON.stringify(out.afterLink.timesSq), 'labels', out.labelsBeforeLink, '->', out.labelsAfterLink);
  await page.screenshot({ path: join(ART, 'T-209-timessq-linked.png') });

  // ---- T-210: the complex map's link line, hovered for real ----
  await page.evaluate(({ a }) => { window.tw.select({ kind: 'station', station: a.id }); }, ts);
  await page.waitForSelector('.complex-map-link', { timeout: 20000 });
  await page.waitForTimeout(800);
  const tagAt = async (sel, nth = 0) => {
    const el = (await page.$$(sel))[nth];
    if (!el) return { sel, missing: true };
    await el.scrollIntoViewIfNeeded().catch(() => {});
    const r = await el.boundingBox();
    await page.mouse.move(r.x + r.width / 2 - 3, r.y + r.height / 2 - 3);
    await page.mouse.move(r.x + r.width / 2, r.y + r.height / 2, { steps: 3 });
    await page.waitForTimeout(80);
    return page.evaluate((sel) => { const t = document.querySelector('.ui-tag'); return { sel, shown: t?.style.display === '', text: t?.innerText ?? '' }; }, sel);
  };
  out.tips = [];
  out.tips.push(await tagAt('.complex-map-link'));
  await page.screenshot({ path: join(ART, 'T-210-link-line.png') });
  // the transfer curves come with the next demand round after the link
  await page.waitForSelector('.complex-map-change', { timeout: 120000 }).catch(() => {});
  out.tips.push(await tagAt('.complex-map-change'));
  await page.screenshot({ path: join(ART, 'T-210-transfer-curve.png') });
  out.tips.push(await tagAt('.split-bar .seg'));
  await page.screenshot({ path: join(ART, 'T-210-purpose-bar.png') });
  out.tips.push(await tagAt('.speed .btn'));
  await page.screenshot({ path: join(ART, 'T-210-pause.png'), clip: { x: 1000, y: 0, width: 400, height: 120 } });
  // (MapLibre's attribution button lies under the top bar, out of the pointer's reach; ui/tip.ts
  // moves a stray title like its own to the tag, checked with a test element)
  out.titles = await page.evaluate(() => ({ attrs: [...document.querySelectorAll('[title]')].map(e => e.outerHTML.slice(0, 120)), svgTitles: document.querySelectorAll('svg title').length }));
  log('tips', JSON.stringify(out.tips), JSON.stringify(out.titles));
  // unlink again (the profile is thrown away anyway)
  await page.evaluate(async ({ a, b }) => (await import('/src/game/world.ts')).edit({ op: 'link', a: a.num, b: b.num, on: false }), ts);
  await page.evaluate(() => window.tw.select(null));
  await page.mouse.move(800, 500);

  // ---- T-211: search ----
  const searchState = async () => (await page.waitForTimeout(60), page.evaluate(() => ({ glass: !!document.querySelector('.map-search.closed .search-open'), box: !!document.querySelector('.map-search input'), focused: document.activeElement?.matches?.('.map-search input') ?? false })));
  const s = {};
  s.start = await searchState();
  await page.screenshot({ path: join(ART, 'T-211-collapsed.png'), clip: { x: 1000, y: 40, width: 400, height: 120 } });
  await page.mouse.click(800, 500); // focus in the game (the map)
  await page.keyboard.press('/');
  s.afterSlash = await searchState();
  await page.keyboard.press('Escape');
  s.afterEscape = await searchState();
  // Ctrl+F: our search, and the browser's find kept away (defaultPrevented)
  await page.evaluate(() => { window.__lastCtrlF = null; window.addEventListener('keydown', e => { if (e.ctrlKey && e.key.toLowerCase() === 'f') window.__lastCtrlF = e.defaultPrevented; }); });
  await page.keyboard.press('Control+f');
  s.afterCtrlF = { ...(await searchState()), prevented: await page.evaluate(() => window.__lastCtrlF) };
  await page.keyboard.type('times sq');
  await page.waitForTimeout(300);
  s.results = await page.evaluate(() => document.querySelectorAll('.map-search-results li').length);
  await page.screenshot({ path: join(ART, 'T-211-open.png'), clip: { x: 900, y: 40, width: 500, height: 330 } });
  await page.keyboard.press('Escape');
  s.afterEscape2 = await searchState();
  // a click on the glass opens it; a click on the map with the box empty closes it
  await page.click('.search-open');
  s.afterClick = await searchState();
  await page.mouse.click(800, 600);
  await page.waitForTimeout(100);
  s.afterBlurEmpty = await searchState();
  // with text left in it, leaving keeps the box
  await page.keyboard.press('/');
  await page.keyboard.type('penn');
  await page.mouse.click(800, 600);
  await page.waitForTimeout(100);
  s.afterBlurText = await searchState();
  await page.keyboard.press('/');
  await page.keyboard.press('Escape');
  // Ctrl+F in another text field (the new save slot name in Settings) is the browser's
  await page.click('.tab.gear');
  await page.click('.save-slot-new input');
  await page.keyboard.press('Control+f');
  s.ctrlFInTextField = { ...(await searchState()), prevented: await page.evaluate(() => window.__lastCtrlF) };
  await page.mouse.click(800, 600);
  out.search = s;
  log('search', JSON.stringify(s));

  // ---- T-213: the circle settings in the Stations tab ----
  out.settingsHasCircles = await page.evaluate(() => document.querySelector('#dock')?.textContent?.includes('Station demand circles') ?? null);
  await page.click('.tab:has-text("Stations")');
  await page.waitForTimeout(200);
  const label = page.locator('label.check:has-text("Station demand circles")');
  // the tab area keeps its scroll from Settings (where the slot name field was clicked)
  await page.evaluate(() => { let e = document.querySelector('label.check'); while (e && e !== document.body) { e.scrollTop = 0; e = e.parentElement; } });
  out.stationsTab = { has: await label.count(), complexOption: await page.locator('label.check:has-text("One circle per station complex")').count() };
  await page.screenshot({ path: join(ART, 'T-213-stations-tab.png'), clip: { x: 0, y: 0, width: 400, height: 420 } });
  await label.click();
  out.stationsTab.afterClick = await page.evaluate(() => window.tw.display.value.stationRiders);
  out.stationsTab.complexOptionAfter = await page.locator('label.check:has-text("One circle per station complex")').count();
  await label.click();
  out.stationsTab.afterSecondClick = await page.evaluate(() => window.tw.display.value.stationRiders);
  log('T-213', out.settingsHasCircles, JSON.stringify(out.stationsTab));

  out.gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  out.errors = errors;
  console.log(JSON.stringify(out, null, 1));
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't209-browser.json'), JSON.stringify(out, null, 2));
} finally { await ctx.close(); }
