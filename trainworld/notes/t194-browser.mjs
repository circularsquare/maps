// T-192/T-193/T-194 browser check on New York's real network (T-007 save), with real pointer
// moves: the demand selector without the bridge, the merged school and university purpose, complex
// view by default, the mirrored timing charts, and the network tag over a station, a demand circle,
// a train and a platform crowd.
// Needs a Vite dev server (BASE) and Playwright (PLAYWRIGHT_MODULE). PROFILE_DIR: where the
// throwaway Chrome profile goes. Run from the maps root: node trainworld/notes/t194-browser.mjs.
// Screenshots: artifacts/trainworld/T-19{2,3,4}-*.png. Results: artifacts/trainworld/notes-outputs/t194-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6731';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't194-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2, args: ['--enable-unsafe-swiftshader'] });
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 300000 });
  result.complexDefault = await page.evaluate(() => window.tw.display.value.complexView);
  check('complex view is the default in a fresh browser', result.complexDefault === true, result.complexDefault);
  const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 600000 });
  await page.evaluate(() => {
    const d = document.querySelector('#debug'); if (d) d.style.display = 'none';
    window.tw.clock.setRunning(false);
  });
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.left, r.top]; });
  /** a real pointer move to map px, then the network tag's text after its rest */
  const tagAt = async (x, y) => {
    await page.mouse.move(box[0] + x - 3, box[1] + y - 3);
    await page.mouse.move(box[0] + x, box[1] + y);
    await page.waitForTimeout(350);
    return page.evaluate(() => { const t = document.querySelector('.track-tag'); return t && t.style.display !== 'none' ? [...t.children].map(c => c.textContent) : null; });
  };
  const stationPx = (name) => page.evaluate((name) => {
    const all = window.tw.world.value.save.stations.filter(s => s.built);
    const st = window.twDemand.view().stations;
    const busy = (s) => { const d = st.get(s.id); return d ? d.boardings + d.alightings : 0; };
    const s = all.filter(s => s.name.includes(name)).sort((a, b) => busy(b) - busy(a))[0] ?? all.sort((a, b) => busy(b) - busy(a))[0];
    const p = window.tw.map.project(window.tw.packToLngLat(s.x, s.y));
    return [p.x, p.y, s.id];
  }, name);

  // ---- T-192: the selector, the merged purpose ----
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'work' }; window.tw.map.jumpTo({ center: [-73.985, 40.755], zoom: 12.5 }); });
  await page.waitForTimeout(2500);
  const options = await page.$$eval('select[aria-label="Trip purpose on the map"] option', os => os.map(o => o.textContent));
  result.options = options;
  check('one School and university entry, no University', options.includes('School and university') && !options.includes('University') && !options.includes('School'), options);
  await page.locator('.demand-selector').screenshot({ path: join(ART, 'T-192-1-selector.png') });
  result.selectorGap = await page.evaluate(() => {
    const sel = document.querySelector('.demand-selector'), dest = [...sel.querySelectorAll('.btn')].pop(), f = sel.querySelector('.purpose-filter');
    return { background: getComputedStyle(sel).backgroundColor, gap: f.getBoundingClientRect().left - dest.getBoundingClientRect().right };
  });
  check('selector has no background behind the gap', /rgba\(0, 0, 0, 0\)|transparent/.test(result.selectorGap.background), result.selectorGap);
  await page.selectOption('select[aria-label="Trip purpose on the map"]', 'school');
  await page.waitForTimeout(4000);
  const mid = await page.evaluate(() => [window.tw.map.getContainer().clientWidth / 2, window.tw.map.getContainer().clientHeight / 2]);
  let bubble = null;
  for (let dx = -200; dx <= 200 && !bubble; dx += 25) for (let dy = -150; dy <= 150 && !bubble; dy += 25) {
    const t = await page.evaluate(([x, y]) => window.twViews.hover(x, y), [mid[0] + dx, mid[1] + dy]);
    if (t) bubble = { at: [mid[0] + dx, mid[1] + dy], t };
  }
  result.schoolBubble = bubble;
  check('school and university bubbles drawn, hover names the merged purpose', /school and university trips/.test(bubble?.t?.[0] ?? ''), bubble);
  await page.screenshot({ path: join(ART, 'T-192-2-school-filter.png') });

  // ---- T-193: a bubble selection's purposes ----
  await page.selectOption('select[aria-label="Trip purpose on the map"]', 'all');
  await page.waitForTimeout(3000);
  let pickAt = null;
  for (let dx = -100; dx <= 100 && !pickAt; dx += 20) for (let dy = -100; dy <= 100 && !pickAt; dy += 20) {
    const t = await page.evaluate(([x, y]) => window.twViews.hover(x, y), [mid[0] + dx, mid[1] + dy]);
    if (t) pickAt = [mid[0] + dx, mid[1] + dy];
  }
  await page.mouse.click(box[0] + pickAt[0], box[1] + pickAt[1]);
  await page.waitForFunction(() => document.querySelectorAll('.purpose-mirror').length > 0, undefined, { timeout: 60000 });
  await page.waitForTimeout(500);
  result.breakdown = await page.evaluate(() => ({
    rows: [...document.querySelectorAll('.purpose-item .purpose-heading')].map(e => e.innerText),
    charts: document.querySelectorAll('.purpose-mirror').length,
    pie: [...document.querySelectorAll('.purpose-pie title')].map(e => e.textContent),
    firstUp: document.querySelector('.purpose-mirror .up')?.title, firstDown: document.querySelector('.purpose-mirror .down')?.title,
  }));
  check('school and university is one row, one mirrored chart a purpose', result.breakdown.rows.filter(r => /School/.test(r)).length === 1 && !result.breakdown.rows.some(r => /University/.test(r)) && result.breakdown.charts === result.breakdown.rows.length, result.breakdown);
  check('pie slices give counts', result.breakdown.pie.every(t => /trip legs a day/.test(t)), result.breakdown.pie);
  const inspector = page.locator('#inspector');
  const chart = page.locator('.purpose-item').first();
  await chart.scrollIntoViewIfNeeded();
  await page.locator('.purpose-mirror .up').first().hover();
  await inspector.screenshot({ path: join(ART, 'T-193-1-purposes.png') });
  await page.locator('.purpose-item').last().scrollIntoViewIfNeeded();
  await inspector.screenshot({ path: join(ART, 'T-193-2-charts.png') });
  await page.keyboard.press('Escape');
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: false }; });

  // ---- T-194: the network tag ----
  // a station with demand circles on (complex view: the complex's count)
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, stationRiders: true }; window.tw.map.jumpTo({ center: [-73.987, 40.757], zoom: 12.6 }); });
  await page.waitForTimeout(2000);
  const ts = await stationPx('Times Sq');
  await page.evaluate(([x, y]) => window.tw.map.setCenter(window.tw.map.unproject([x, y])), ts);
  await page.waitForTimeout(1500);
  ts.splice(0, 2, ...(await stationPx('Times Sq')).slice(0, 2));
  // trains off: a train standing in the station wins the pick (and gets its own tag, below)
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, trains: false }; });
  await page.waitForTimeout(300);
  result.station = await tagAt(ts[0], ts[1]);
  check('station tag gives the circle count', /boardings and alightings a day/.test(result.station?.join(' ') ?? ''), result.station);
  await page.screenshot({ path: join(ART, 'T-194-1-station.png'), clip: { x: box[0] + ts[0] - 180, y: box[1] + ts[1] - 160, width: 560, height: 320 } });
  // a demand circle away from the station: try round the circle's centre
  let circle = null;
  for (const r of [26, 34, 20, 42]) for (let a = 0; a < 16 && !circle; a++) {
    const x = ts[0] + r * Math.cos(a * Math.PI / 8), y = ts[1] + r * Math.sin(a * Math.PI / 8);
    const t = await tagAt(x, y);
    if (t && /boardings and alightings/.test(t.join(' ')) && await page.evaluate(([x, y]) => !window.tw.pick(window.tw.overlay, x, y, true), [x, y])) circle = { at: [x, y], t };
  }
  result.circle = circle;
  check('demand circle tag away from any station', !!circle, circle);
  if (circle) await page.screenshot({ path: join(ART, 'T-194-2-circle.png'), clip: { x: box[0] + circle.at[0] - 180, y: box[1] + circle.at[1] - 160, width: 560, height: 320 } });
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, stationRiders: false, trains: true }; });
  await page.waitForTimeout(300);

  // a train: the busiest in view
  const train = await page.evaluate(() => {
    const r = window.tw.renderer, n = r.net, t = window.tw.clock.now();
    const W = window.tw.map.getContainer().clientWidth, H = window.tw.map.getContainer().clientHeight;
    for (let i = 0; i < r.tripCount; i++) {
      const at = window.tw.tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch, t);
      if (!at) continue;
      const [x, y] = window.tw.overlay.toScreen(at.x, at.y);
      if (x > 200 && x < W - 200 && y > 150 && y < H - 150) return [x, y];
    }
    return null;
  });
  result.train = train ? await tagAt(train[0], train[1]) : null;
  check('train tag gives riders aboard', /riders (aboard|on a train)/.test(result.train?.join(' ') ?? ''), result.train);
  if (train) await page.screenshot({ path: join(ART, 'T-194-3-train.png'), clip: { x: box[0] + train[0] - 180, y: box[1] + train[1] - 160, width: 560, height: 320 } });

  // a platform crowd band, zoomed in where platforms are drawn
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.9905, 40.7505], zoom: 15.5 }));
  await page.waitForTimeout(2500);
  const crowd = await page.evaluate(() => {
    const r = window.tw.renderer, n = r.net, W = window.tw.map.getContainer().clientWidth, H = window.tw.map.getContainer().clientHeight;
    const project = (x, y) => window.tw.overlay.toScreen(x, y);
    const out = [];
    for (let i = 0; i < n.stationCount; i++) {
      const s = r.stationShapeOf(i, project);
      if (s.x < 150 || s.x > W - 150 || s.y < 150 || s.y > H - 150) continue;
      out.push([s.x + s.nx * (s.halfWidth + 9), s.y + s.ny * (s.halfWidth + 9)], [s.x - s.nx * (s.halfWidth + 9), s.y - s.ny * (s.halfWidth + 9)]);
    }
    return out;
  });
  let band = null;
  for (const [x, y] of crowd) {
    const t = await tagAt(x, y);
    if (t && /waiting/i.test(t.join(' '))) { band = { at: [x, y], t }; break; }
  }
  result.crowd = band;
  check('crowd band tag gives riders waiting', !!band, band);
  if (band) await page.screenshot({ path: join(ART, 'T-194-4-crowd.png'), clip: { x: box[0] + band.at[0] - 180, y: box[1] + band.at[1] - 160, width: 560, height: 320 } });

  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
  await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', 't194-browser.json'), JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
