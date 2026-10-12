// T-120 acceptance: longer-trip bubbles per far zone, drawn like the commuter bubbles, in the real
// app. Adapted from t118-browser.mjs. Needs `python serve.py` at the maps root
// (localhost:8800/trainworld/) and playwright (PLAYWRIGHT_MODULE).
// Run from the maps root: node trainworld/notes/t120-browser.mjs   (isolated profile, own browser)
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const profile = await mkdtemp(join(tmpdir(), 't120-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1300, height: 900 } });
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto('http://localhost:8800/trainworld/?debug=1&new=1&paused=1');
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client);
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 240000 });
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
  const sel = async (city, lng, lat, km, end) => page.evaluate(async ([city, lng, lat, km, end]) => {
    const R = 6371008.8, rad = Math.PI / 180;
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
    window.tw.select({ kind: 'commuters', end, cells, areas: 1 });
    await new Promise(r => setTimeout(r, 1500));
    return { cells: cells.length, r };
  }, [city, lng, lat, km, end]);
  const check = (name, r) => {
    for (const d of [r.starts, r.ends]) {
      const sum = (k, l) => l.reduce((a, p) => a + p[k], 0);
      for (const k of ['all', 'rail']) {
        if (Math.abs(sum(k, d.zones) - d[k]) > 1e-6 * Math.max(1, d[k]) || Math.abs(sum(k, d.places) - d[k]) > 1e-6 * Math.max(1, d[k])) throw Error(name + ' zones do not add up ' + k + ' ' + sum(k, d.zones) + ' ' + d[k]);
      }
      if (d.zones.some(z => z.rail > z.all + 1e-6 || !isFinite(z.x) || !isFinite(z.y))) throw Error(name + ' zone rail above all');
    }
  };
  const zoneStats = (d) => ({ zones: d.zones.length, over05: d.zones.filter(z => z.all >= 0.5).length, over2: d.zones.filter(z => z.all >= 2).length, top: d.zones.slice(0, 3), all: d.all, rail: d.rail });
  const shot = async (name) => { await page.waitForTimeout(2000); await page.screenshot({ path: 'artifacts/trainworld/T-120-' + name + '.png' }); };
  const hide = () => page.evaluate(() => { document.querySelector('#debug').style.display = 'none'; });
  const fit = (b) => page.evaluate(b => window.tw.map.fitBounds(b, { padding: 40, animate: false }), b);

  const nyc = await sel('nyc', -73.985, 40.758, 3, 'home');
  check('nyc', nyc.r);
  await hide();
  await fit([[-75.6, 39.6], [-73.2, 41.4]]);
  await shot('timessquare-region');
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.985, 40.758], zoom: 9.2 }));
  await shot('timessquare-near');
  const timesJobs = await sel('nyc', -73.985, 40.758, 3, 'work');
  check('nyc jobs', timesJobs.r);
  await fit([[-75.6, 39.6], [-73.2, 41.4]]);
  await shot('timessquare-jobs-region');

  const phl = await sel('phl', -75.1636, 39.9526, 3, 'home');
  check('phl', phl.r);
  await hide();
  await fit([[-75.6, 39.6], [-73.7, 41.1]]);
  await shot('philadelphia-region');
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.6, 40.3], zoom: 8.6 }));
  await shot('philadelphia-mid');
  // the bubble size slider scales them
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, bubbleSize: 2 }; });
  await shot('philadelphia-size2');
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, bubbleSize: 1 }; });
  // hover: find the screen position of the biggest far zone and ask the finder
  const hov = await page.evaluate(async () => {
    const r = await window.twDemand.intercitySelection('home', Uint32Array.from(window.tw.selection.value.cells));
    const z = r.starts.zones[0];
    const R = 6371008.8, rad = Math.PI / 180;
    const ll = [-73.985 + z.x / (R * Math.cos(40.758 * rad) * rad), 40.758 + z.y / (R * rad)];
    const pt = window.tw.map.project(ll);
    return { zone: z, ll, hover: window.twViews.hover(pt.x, pt.y) };
  });

  const station = await page.evaluate(async () => {
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
  check('station', station);
  await shot('station');
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { fixture, nyc: { cells: nyc.cells, matched: nyc.r.matched, starts: zoneStats(nyc.r.starts), ends: zoneStats(nyc.r.ends) }, nycJobs: { starts: zoneStats(timesJobs.r.starts), ends: zoneStats(timesJobs.r.ends) }, phl: { cells: phl.cells, starts: zoneStats(phl.r.starts), ends: zoneStats(phl.r.ends) }, hov, station: { name: station.name, stationId: station.stationId, starts: zoneStats(station.starts), ends: zoneStats(station.ends) }, errors, gl };
  console.log(JSON.stringify(result, null, 1));
  await writeFile('trainworld/notes/T-120-browser.json', JSON.stringify(result, null, 2));
  if (errors.filter(e => !/404/.test(e)).length || gl) throw Error(JSON.stringify({ errors, gl }));
} finally { await ctx.close(); }
