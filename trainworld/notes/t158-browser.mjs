// T-158 browser check: the map's purpose filter on New York's real network (T-007 save) in the
// real app. Needs a Vite dev server (BASE, default http://localhost:5547; the probe imports
// /src modules, so a build will not do) and Playwright (PLAYWRIGHT_MODULE). Run from the maps
// root: node trainworld/notes/t158-browser.mjs. Screenshots: artifacts/trainworld/T-158-*.png.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:5547';
const WORKERS = process.env.WORKERS ?? '3';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't158-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = 'artifacts/trainworld/T-158-';
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });
  const bytes = await readFile('trainworld/T-007-real-nyc.save');
  await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
  await page.waitForFunction(() => window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size > 100, undefined, { timeout: 400000 });
  await page.evaluate(async () => {
    window.__ui = await import('/src/game/ui.ts');
    window.__th = await import('/src/map/trackHover.ts');
    window.__dm = await import('/src/game/demand.ts');
  });

  // 1. per-purpose totals against all purposes: every line's riders and loads, every station's boardings
  const sums = await page.evaluate(() => {
    const dv = window.twDemand.view(), keys = [...dv.purposes.keys()];
    const w = window.tw.world.value, lineName = new Map(w.save.lines.map(l => [l.id, l.name])), stName = new Map(w.save.stations.map(s => [s.id, s.name]));
    const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
    let worstLine = 0, worstLoad = 0, worstStation = 0;
    for (const [id, l] of dv.lines) {
      const sum = keys.reduce((s, k) => s + (dv.purposes.get(k).lines.get(id)?.ridersPerDay ?? 0), 0);
      worstLine = Math.max(worstLine, rel(sum, l.ridersPerDay));
      l.loads.forEach((p, q) => p.forEach((v, j) => {
        const s = keys.reduce((a, k) => a + (dv.purposes.get(k).lines.get(id)?.loads[q][j] ?? 0), 0);
        worstLoad = Math.max(worstLoad, rel(s, v));
      }));
    }
    for (const [id, s] of dv.stations) {
      const b = keys.reduce((a, k) => a + (dv.purposes.get(k).stations.get(id)?.boardings ?? 0), 0);
      const al = keys.reduce((a, k) => a + (dv.purposes.get(k).stations.get(id)?.alightings ?? 0), 0);
      worstStation = Math.max(worstStation, rel(b, s.boardings), rel(al, s.alightings));
    }
    const top = [...dv.lines].sort((a, b) => b[1].ridersPerDay - a[1].ridersPerDay)[0];
    const topSt = [...dv.stations].sort((a, b) => b[1].boardings - a[1].boardings)[0];
    const line = { name: lineName.get(top[0]), all: Math.round(top[1].ridersPerDay), ...Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).lines.get(top[0])?.ridersPerDay ?? 0)])) };
    line.sum = keys.reduce((s, k) => s + line[k], 0);
    const station = { name: stName.get(topSt[0]), all: Math.round(topSt[1].boardings), ...Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).stations.get(topSt[0])?.boardings ?? 0)])) };
    station.sum = keys.reduce((s, k) => s + station[k], 0);
    let network = { all: 0 };
    for (const l of dv.lines.values()) network.all += l.ridersPerDay;
    for (const k of keys) { network[k] = 0; for (const l of dv.purposes.get(k).lines.values()) network[k] += l.ridersPerDay; }
    network = Object.fromEntries(Object.entries(network).map(([k, v]) => [k, Math.round(v)]));
    return { keys, worstLine, worstLoad, worstStation, line, station, network };
  });
  console.log(JSON.stringify(sums, null, 1));

  // 2. bubbles: each purpose's legs per cell, summed, against all purposes
  const bubbles = await page.evaluate(async () => {
    const dm = window.__dm, tot = (m) => { let h = 0, a = 0; for (const k of ['homeRail', 'homeWalk', 'homeDrive']) for (const v of m[k]) h += v; for (const k of ['workRail', 'workWalk', 'workDrive']) for (const v of m[k]) a += v; return [h, a]; };
    const all = await dm.cellModes('all'), res = { all: all && tot(all).map(Math.round) };
    let h = 0, a = 0;
    for (const p of ['work', 'school', 'university', 'shopping', 'social', 'longer']) {
      const m = await dm.cellModes(p);
      if (!m) { res[p] = null; continue; }
      const t = tot(m); res[p] = t.map(Math.round); h += t[0]; a += t[1];
    }
    res.sum = [Math.round(h), Math.round(a)];
    return res;
  });
  console.log(JSON.stringify(bubbles));

  // 3. a selection's far end by purpose: cells within 600 m of Times Sq
  const sel = await page.evaluate(async () => {
    const dm = window.__dm, cells = await dm.cityCells();
    const st = window.tw.world.value.save.stations.find(s => /Times Sq/.test(s.name));
    const pick = [];
    for (let c = 0; c < cells.n; c++) if (Math.hypot(cells.xy[2 * c] - st.x, cells.xy[2 * c + 1] - st.y) < 600) pick.push(c);
    const f = await dm.flows('home', Uint32Array.from(pick));
    if (!f) return null;
    const res = { cells: pick.length, all: Math.round(f.rail + f.walk + f.drive) };
    let s = 0;
    for (const [k, e] of f.byPurpose) { res[k] = Math.round(e.rail + e.walk + e.drive); s += e.rail + e.walk + e.drive; }
    res.sum = Math.round(s);
    window.__sel = { kind: 'commuters', end: 'home', cells: pick, areas: 1 };
    return res;
  });
  console.log(JSON.stringify(sel));

  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; });
  const shot = async (name, wait = 2500, clip) => { await page.waitForTimeout(wait); await page.screenshot({ path: out + name + '.png', clip }); };
  const setPurpose = (p) => page.evaluate(p => { window.__ui.purposeFilter.value = p; }, p);

  // 4. network drawings under each purpose: Midtown, traffic widths and station circles
  await page.evaluate(() => {
    window.tw.display.value = { ...window.tw.display.value, trackColour: 'traffic', stationRiders: true, complexView: false, commuters: false };
    window.tw.select(null);
    window.tw.map.jumpTo({ center: [-73.975, 40.745], zoom: 12.3 });
  });
  for (const p of ['all', 'work', 'school', 'shopping', 'social', 'longer']) { await setPurpose(p); await shot('network-' + p); }
  await setPurpose('work');
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, complexView: true }; });
  await shot('network-work-complex');
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, complexView: false }; });

  // 5. the tooltip: rest the pointer on the busiest line's busiest stretch under a filter
  const tip = await page.evaluate(() => {
    const dv = window.twDemand.view(), w = window.tw.world.value, map = window.tw.map, th = window.__th;
    const st = new Map(w.save.stations.map(s => [s.id, s]));
    // a Manhattan stretch: both stops inside the view
    const box = map.getContainer().getBoundingClientRect();
    let best = null;
    for (const l of w.save.lines) {
      const d = dv.lines.get(l.id); if (!d) continue;
      for (let k = 0; k < l.stops.length - 1; k++) {
        const a = st.get(l.stops[k]), b = st.get(l.stops[k + 1]);
        const pa = map.project([a.lng, a.lat]), pb = map.project([b.lng, b.lat]);
        if (Math.hypot(pa.x - pb.x, pa.y - pb.y) < 40) continue;
        const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
        if (mx < 450 || my < 80 || mx > box.width - 60 || my > box.height - 60) continue;
        const day = th.dailyRiders(dv, l.id, l.stops.length, k);
        if (day !== null && (!best || day > best.day)) best = { line: l.name, id: l.id, k, mx, my, day };
      }
    }
    if (!best) return null;
    // the stroke may curve away from the stops' midpoint: look nearby for the line's own tag
    for (let r = 0; r <= 24; r += 2) for (let a = 0; a < 16; a++) {
      const x = best.mx + r * Math.cos(a * Math.PI / 8), y = best.my + r * Math.sin(a * Math.PI / 8);
      const t = th.trackTip(window.tw.overlay, x, y);
      // page coordinates for the pointer: the map sits right of the dock
      if (t && t[0] === best.line) return { ...best, x: x + box.left, y: y + box.top, tip: t };
    }
    return { ...best, x: null };
  });
  const tips = {};
  if (tip?.x !== null && tip) {
    for (const p of ['all', 'shopping', 'longer']) {
      await setPurpose(p);
      await page.mouse.move(tip.x + 30, tip.y + 30);
      await page.waitForTimeout(300);
      await page.mouse.move(tip.x, tip.y);
      await page.waitForTimeout(900);
      tips[p] = await page.evaluate(() => { const t = document.querySelector('.track-tag'); return t && t.style.display !== 'none' ? t.innerText : null; });
      await page.screenshot({ path: out + 'tooltip-' + p + '.png', clip: { x: Math.max(0, tip.x - 200), y: Math.max(0, tip.y - 120), width: 520, height: 260 } });
    }
    await page.mouse.move(5, 890);
  }
  console.log(JSON.stringify({ tip, tips }));

  // 6. bubbles, homes, all purposes then school, then a selection under shopping
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, stationRiders: false, commuters: true, commuterEnd: 'home' }; window.tw.map.jumpTo({ center: [-73.95, 40.72], zoom: 11.2 }); });
  await setPurpose('all');
  await shot('bubbles-all', 6000);
  await setPurpose('school');
  await shot('bubbles-school', 8000);
  await setPurpose('shopping');
  await page.waitForTimeout(6000);
  await page.evaluate(() => { window.tw.select(window.__sel); window.tw.map.jumpTo({ center: [-73.975, 40.75], zoom: 12 }); });
  await shot('selection-shopping', 6000);
  const inspector = await page.evaluate(() => document.querySelector('#inspector')?.innerText ?? '');
  // the control itself
  await page.evaluate(() => window.tw.select(null));
  const ctl = await page.evaluate(() => { const r = document.querySelector('.demand-selector').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  await page.screenshot({ path: out + 'control.png', clip: { x: ctl.x - 6, y: ctl.y - 6, width: ctl.width + 12, height: ctl.height + 12 }, scale: 'device' });
  const options = await page.evaluate(() => [...document.querySelectorAll('.demand-selector select option')].map(o => o.textContent));
  await setPurpose('all');
  const gl = await page.evaluate(() => window.tw.renderer.gl.getError());
  const result = { sums, bubbles, sel, tip, tips, inspector: inspector.slice(0, 900), ctl, options, errors, gl };
  console.log(JSON.stringify({ inspector: result.inspector, options, errors, gl }, null, 1));
  await writeFile('artifacts/trainworld/notes-outputs/t158-browser.json', JSON.stringify(result, null, 2));
} finally { await ctx.close(); }
