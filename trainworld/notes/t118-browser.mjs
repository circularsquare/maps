// T-118 acceptance: longer trips of a bubble selection and of a station, in the real app.
// Needs `python serve.py` at the maps root (localhost:8800/trainworld/) and playwright.
// Run from the maps root: node trainworld/notes/t118-browser.mjs   (isolated profile, own browser)
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const profile = await mkdtemp(join(tmpdir(), 't118-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 } });
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); if (m.text().startsWith('DBG')) console.log(m.text()); });
  const missing = [];
  page.on('response', r => { if (r.status() === 404) missing.push(r.url()); });
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client);
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 240000 });
  // Philadelphia service: one express line City Hall area, plus a long line to Trenton and New York
  const fixture = await page.evaluate(async () => {
    const t = window.tw, run = async op => { const a = await t.client.edit(op); if (!a.ok) throw Error(JSON.stringify({ op, a })); return a; };
    const R = 6371008.8, rad = Math.PI / 180, x = R * (-75.1636 + 73.985) * rad * Math.cos(40.758 * rad), y = R * (39.9526 - 40.758) * rad;
    await run({ op: 'route', from: { kind: 'free', x, y: y - 2000, level: -2 }, to: { kind: 'free', x, y: y + 2000, level: -2 }, pis: [], single: false });
    const edge = t.world.value.edges.at(-1);
    await run({ op: 'addStation', at: { kind: 'node', node: edge.a }, platform: 200, name: 'Philadelphia south' });
    await run({ op: 'addStation', at: { kind: 'node', node: edge.b }, platform: 200, name: 'Philadelphia north' });
    const line = (await run({ op: 'addLine', stops: [edge.a, edge.b], name: 'Philadelphia check', colour: '#17a488' })).line;
    await run({ op: 'setTrainCount', line, trains: [4, 4, 2] }); await run({ op: 'constructAll' }); return { line };
  });
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 109, undefined, { timeout: 240000 });
  // select every New York cell within 3 km of Times Square (homes)
  const sel = async (city, lng, lat, km, end) => page.evaluate(async ([city, lng, lat, km, end]) => {
    let probe = null;
    const R = 6371008.8, rad = Math.PI / 180;
    // as a player would: focus the city, show its bubbles, wait for them, then select
    window.tw.map.jumpTo({ center: [lng, lat], zoom: 11.2 });
    window.twDemand.activeCity.value = city;
    window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: end };
    const b0 = window.twViews.stats.commuters.builds;
    for (let i = 0; i < 100 && window.twViews.stats.commuters.builds === b0; i++) await new Promise(r => setTimeout(r, 100));
    await new Promise(r => setTimeout(r, 500));
    const c = await window.twDemand.cityCells();
    const px = R * (lng + 73.985) * rad * Math.cos(40.758 * rad), py = R * (lat - 40.758) * rad;
    const cells = [];
    for (let i = 0; i < c.n; i++) if (Math.hypot(c.xy[2 * i] - px, c.xy[2 * i + 1] - py) < km * 1000) cells.push(i);
    const r = await window.twDemand.intercitySelection(end, Uint32Array.from(cells));
    const ll = (x, y) => [-73.985 + x / (R * Math.cos(40.758 * rad) * rad), 40.758 + y / (R * rad)].map(v => +v.toFixed(3));
    const mean = (ids) => ll(ids.reduce((a, i) => a + c.xy[2 * i], 0) / ids.length, ids.reduce((a, i) => a + c.xy[2 * i + 1], 0) / ids.length);
    const f = await window.twDemand.flows(end, Uint32Array.from(cells));
    probe = { selAt: mean(cells), farAt: f && mean(Array.from(f.cells).slice(0, 200)), farCells: f && f.cells.length };
    window.tw.select({ kind: 'commuters', end, cells, areas: 1 });
    await new Promise(r => setTimeout(r, 1500));
    const pt = window.tw.map.project([lng, lat]);
    probe.hoverAtSelection = window.twViews.hover(pt.x, pt.y);
    probe.city = window.twDemand.activeCity.value;
    probe.center = window.tw.map.getCenter();
    return { cells: cells.length, r, probe };
  }, [city, lng, lat, km, end]);
  const check = (name, r) => {
    const f = r.starts, g = r.ends;
    for (const d of [f, g]) {
      const sumAll = d.places.reduce((a, p) => a + p.all, 0), sumRail = d.places.reduce((a, p) => a + p.rail, 0);
      if (Math.abs(sumAll - d.all) > 1e-6 * Math.max(1, d.all) || Math.abs(sumRail - d.rail) > 1e-6 * Math.max(1, d.rail) || d.rail > d.all + 1e-6) throw Error(name + ' conservation ' + JSON.stringify(d));
    }
  };
  const nyc = await sel('nyc', -73.985, 40.758, 3, 'home');
  check('nyc', nyc.r);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'artifacts/trainworld/T-118-selection-nyc.png' });
  const phl = await sel('phl', -75.1636, 39.9526, 3, 'home');
  check('phl', phl.r);
  await page.evaluate(() => { document.querySelector('#debug').style.display = 'none'; window.tw.map.flyTo({ center: [-75.1636, 39.9526], zoom: 10.5, animate: false }); });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'artifacts/trainworld/T-118-selection-phl.png' });
  await page.evaluate(() => { window.tw.map.fitBounds([[-75.6, 39.6], [-73.7, 41.1]], { padding: 40, animate: false }); });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: 'artifacts/trainworld/T-118-selection-phl-region.png' });
  const station = await page.evaluate(async () => {
    // the station with the most longer-trip rail riders
    let best = null;
    for (const st of window.tw.world.value.save.stations) {
      const r = await window.twDemand.client.intercityStation(String(st.id));
      if (r && (!best || r.starts.rail + r.ends.rail > best.starts.rail + best.ends.rail)) best = { ...r, name: st.name };
    }
    window.tw.select({ kind: 'station', station: best.stationId });
    window.tw.map.flyTo({ center: [-74.2, 40.4], zoom: 8.4, animate: false });
    await new Promise(r => setTimeout(r, 2500));
    return best;
  });
  await page.screenshot({ path: 'artifacts/trainworld/T-118-station.png' });
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { fixture, nyc: { probe: nyc.probe, cells: nyc.cells, starts: nyc.r.starts, ends: nyc.r.ends, matched: nyc.r.matched }, phl: { probe: phl.probe, cells: phl.cells, starts: phl.r.starts, ends: phl.r.ends, matched: phl.r.matched }, station, errors, missing, gl };
  console.log(JSON.stringify(result));
  await writeFile('artifacts/trainworld/notes-outputs/T-118-browser.json', JSON.stringify(result, null, 2));
  if (errors.filter(e => !/404/.test(e)).length || gl) throw Error(JSON.stringify({ errors, gl }));
} finally { await ctx.close(); }
