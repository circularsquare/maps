// T-146 browser check: station complexes on New York's real network (T-007 save) in the real app.
// Needs the app served (BASE, default http://localhost:8800/trainworld; a Vite dev server works:
// the probe imports /src/game/complexView.ts, so use dev for that part) and Playwright
// (PLAYWRIGHT_MODULE). Run from the maps root: node trainworld/notes/t146-browser.mjs
// Screenshots: artifacts/trainworld/T-146-*.png.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:8800/trainworld';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't146-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 300000 });
  const stats = await page.evaluate(async () => {
    const cv = await import('/src/game/complexView.ts');
    const c = cv.complexes.value, st = window.tw.world.value.save.stations, name = new Map(st.map(s => [s.id, s.name]));
    const multi = c.members.filter(m => m.length > 1).sort((a, b) => b.length - a.length);
    const sizes = {};
    for (const m of c.members) sizes[m.length] = (sizes[m.length] ?? 0) + 1;
    window.__cv = cv;
    return { stations: st.length, complexes: c.members.length, sizes, biggest: multi.slice(0, 12).map(m => m.map(id => name.get(id))) };
  });
  // a complex with Times Sq in it, else the biggest
  const pick = await page.evaluate(() => {
    const c = window.__cv.complexes.value, st = window.tw.world.value.save.stations;
    const ts = st.find(s => /Times Sq/.test(s.name) && c.members[c.of.get(s.id)].length > 1) ?? st.find(s => c.members[c.of.get(s.id)].length === Math.max(...c.members.map(m => m.length)));
    return { id: ts.id, name: ts.name, lng: ts.lng, lat: ts.lat, members: c.members[c.of.get(ts.id)] };
  });
  const sums = await page.evaluate(async ({ id, members }) => {
    const d = window.twDemand.client;
    const one = await Promise.all(members.map(m => d.stationRiders(m, 20)));
    const all = await d.stationRiders(id, 20, members);
    const seen = new Set(); let home = 0, work = 0;
    for (const r of one) { if (!r) continue; const k = r.home.total + '|' + r.work.total; if (seen.has(k)) continue; seen.add(k); home += r.home.total; work += r.work.total; }
    const ic = await d.intercityStation(id, members), ic1 = await Promise.all(members.map(m => d.intercityStation(m)));
    const icSeen = new Set(); let icRail = 0;
    for (const r of ic1) { if (!r) continue; const k = r.starts.rail + '|' + r.ends.rail; if (icSeen.has(k)) continue; icSeen.add(k); icRail += r.starts.rail + r.ends.rail; }
    const dv = window.twDemand.view(); let board = 0;
    for (const m of members) { const s = dv.stations.get(m); if (s) board += s.boardings; }
    return { distinctDemandStations: seen.size, homeSum: home, homeComplex: all?.home.total, workSum: work, workComplex: all?.work.total, icSum: icRail, icComplex: ic ? ic.starts.rail + ic.ends.rail : null, boardings: board };
  }, pick);
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });
  const shot = async (name, clip) => { await page.waitForTimeout(2500); await page.screenshot({ path: 'artifacts/trainworld/T-146-' + name + '.png', clip }); };
  const inspector = async () => page.evaluate(() => document.querySelector('#inspector')?.innerText ?? '');
  await page.evaluate(({ id, lng, lat }) => {
    window.tw.display.value = { ...window.tw.display.value, complexView: false, stationRiders: true, commuters: false };
    window.tw.select({ kind: 'station', station: id });
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 14.6 });
  }, pick);
  await shot('station-view');
  const stationText = await inspector();
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, complexView: true }; });
  await shot('complex-view');
  const complexText = await inspector();
  const scopes = await page.evaluate(async () => (await import('/src/map/demandViews.ts')).stationView.value?.scope);
  // circles alone, nothing selected, over Midtown
  await page.evaluate(() => { window.tw.select(null); window.tw.map.jumpTo({ center: [-73.985, 40.755], zoom: 13.4 }); });
  await shot('circles-complex');
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, complexView: false }; });
  await shot('circles-station');
  // a station on its own: no switch
  const lone = await page.evaluate(async () => {
    const c = window.__cv.complexes.value, st = window.tw.world.value.save.stations;
    const s = st.find(s => c.members[c.of.get(s.id)].length === 1 && window.twDemand.view().stations.get(s.id));
    window.tw.display.value = { ...window.tw.display.value, complexView: true };
    window.tw.select({ kind: 'station', station: s.id });
    await new Promise(r => setTimeout(r, 1500));
    return { name: s.name, buttons: [...document.querySelectorAll('#inspector button')].map(b => b.innerText) };
  });
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { stats, pick, sums, stationText, complexText, scopes, lone, errors, gl };
  console.log(JSON.stringify(result, null, 1));
  await writeFile('artifacts/trainworld/notes-outputs/t146-browser.json', JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
