// T-142 browser check: everyday, school, university, medical and airport purposes in all nine cities, in the real
// app. A new game, then a 5 km elevated test line in each of the eight cities other than New York (as
// t137-browser.mjs), constructed; at round 3 every city must earn riders of every purpose, and
// every line's and station's purposes must add up to its total (T-158's sums). Then a view of
// Philadelphia and Washington, opened at runtime, gives their everyday opening time in Chrome.
// Needs a Vite dev server (BASE, default http://localhost:6012; the probe imports /src modules)
// and Playwright (PLAYWRIGHT_MODULE). Run from the maps root: node trainworld/notes/t142-browser.mjs.
// Screenshots: artifacts/trainworld/T-142-*.png.
import { writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:6012';
const WORKERS = process.env.WORKERS ?? '3';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't142-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = process.env.OUT ?? 'artifacts/trainworld/T-142b-';
// T-142's second slice (airports, medical): one more line, Philadelphia's centre to its airport
const AIRPORT_LINE = [[-75.1636, 39.9526], [-75.2437, 39.8768]];
const CITIES = [['phl', 'Philadelphia', [-75.1636, 39.9526]], ['bos', 'Boston', [-71.0589, 42.3601]], ['wdc', 'Washington', [-77.0369, 38.9072]], ['bal', 'Baltimore', [-76.6122, 39.2904]], ['pvd', 'Providence', [-71.4128, 41.8240]], ['nhv', 'New Haven', [-72.9279, 41.3083]], ['hfd', 'Hartford', [-72.6851, 41.7637]], ['spf', 'Springfield', [-72.5898, 42.1015]]];
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });

  // 1. opening a city at runtime: view Philadelphia, then Washington (each opens its pack, everyday layer included)
  result.opening = {};
  for (const [id, , ll] of [CITIES[0], CITIES[2]]) {
    const t0 = await page.evaluate(([id, ll]) => { window.tw.map.jumpTo({ center: ll, zoom: 11 }); return performance.now(); }, [id, ll]);
    await page.waitForFunction(id => window.twDemand.client.start.loads.find(l => l.item === id)?.openedAt, id, { timeout: 120000 });
    result.opening[id] = await page.evaluate(([id, t0]) => { const l = window.twDemand.client.start.loads.find(l => l.item === id); return { download: Math.round(l.downloadedAt - l.requestedAt), open: Math.round(l.openedAt - l.downloadedAt), fromPan: Math.round(l.openedAt - t0) }; }, [id, t0]);
  }
  check('Philadelphia and Washington open at runtime', true, result.opening);

  // 2. a test line in each of the eight cities, constructed
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async ([cities, airport]) => {
    const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    for (const [, name, ll] of cities) {
      const p = t.lngLatToPack(...ll);
      await run({ op: 'route', from: { kind: 'free', x: p.x - 2500, y: p.y, level: 1 }, to: { kind: 'free', x: p.x + 2500, y: p.y, level: 1 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      await run({ op: 'addStation', at: { kind: 'node', node: e.a }, platform: 200, name: name + ' west' });
      await run({ op: 'addStation', at: { kind: 'node', node: e.b }, platform: 200, name: name + ' east' });
      const line = (await run({ op: 'addLine', stops: [e.a, e.b], name: name + ' test', colour: '#17a488' })).line;
      await run({ op: 'setTrainCount', line, trains: [4, 4, 2] });
    }
    const [a, b] = airport.map(ll => t.lngLatToPack(...ll));
    await run({ op: 'route', from: { kind: 'free', x: a.x, y: a.y - 600, level: 1 }, to: { kind: 'free', x: b.x, y: b.y, level: 1 }, pis: [], single: false });
    const e = t.world.value.edges.at(-1);
    await run({ op: 'addStation', at: { kind: 'node', node: e.a }, platform: 200, name: 'Airport line centre' });
    await run({ op: 'addStation', at: { kind: 'node', node: e.b }, platform: 200, name: 'Airport line terminal' });
    const line = (await run({ op: 'addLine', stops: [e.a, e.b], name: 'Philadelphia airport test', colour: '#a8a032' })).line;
    await run({ op: 'setTrainCount', line, trains: [4, 4, 2] });
    await run({ op: 'constructAll' });
  }, [CITIES, AIRPORT_LINE]);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3 && window.twDemand.view().lines.size === 9, before, { timeout: 400000 });
  await page.evaluate(async () => { window.__ui = await import('/src/game/ui.ts'); window.__dm = await import('/src/game/demand.ts'); });

  // 3. each line's riders by purpose; purposes add up to the line, every load and station
  const sums = await page.evaluate(() => {
    const dv = window.twDemand.view(), keys = [...dv.purposes.keys()], w = window.tw.world.value;
    const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
    let worstLine = 0, worstLoad = 0, worstStation = 0;
    const lines = {};
    for (const l of w.save.lines) {
      const d = dv.lines.get(l.id); if (!d) continue;
      const by = Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0)]));
      lines[l.name] = { all: Math.round(d.ridersPerDay), ...by };
      worstLine = Math.max(worstLine, rel(keys.reduce((s, k) => s + (dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0), 0), d.ridersPerDay));
      d.loads.forEach((p, q) => p.forEach((v, j) => { worstLoad = Math.max(worstLoad, rel(keys.reduce((a, k) => a + (dv.purposes.get(k).lines.get(l.id)?.loads[q][j] ?? 0), 0), v)); }));
    }
    for (const [id, s] of dv.stations) {
      const b = keys.reduce((a, k) => a + (dv.purposes.get(k).stations.get(id)?.boardings ?? 0), 0);
      const al = keys.reduce((a, k) => a + (dv.purposes.get(k).stations.get(id)?.alightings ?? 0), 0);
      worstStation = Math.max(worstStation, rel(b, s.boardings), rel(al, s.alightings));
    }
    return { keys, worstLine, worstLoad, worstStation, lines };
  });
  result.sums = sums;
  check('purposes add up to every line, load and station', sums.worstLine < 1e-4 && sums.worstLoad < 1e-4 && sums.worstStation < 1e-4, { worstLine: sums.worstLine, worstLoad: sums.worstLoad, worstStation: sums.worstStation });
  for (const [, name] of CITIES) {
    const l = sums.lines[name + ' test'];
    // T-192: school and university are one shown purpose, "school" (its members summed)
    check(name + ' line carries every purpose', l && ['work', 'shopping', 'social', 'school', 'medical'].every(k => l[k] > 0), l);
  }
  const al = sums.lines['Philadelphia airport test'];
  check('the Philadelphia airport line carries airport trips', al && al.airport > 0, al);

  // 4. city bubbles by purpose and a selection's purposes, in Philadelphia (focus there)
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-75.1636, 39.9526], zoom: 12 }));
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'phl', undefined, { timeout: 60000 });
  await page.waitForFunction(() => !window.tw.map.isMoving());
  const bubbles = await page.evaluate(async () => {
    const dm = window.__dm, tot = (m) => { let h = 0, a = 0; for (const k of ['homeRail', 'homeWalk', 'homeDrive']) for (const v of m[k]) h += v; for (const k of ['workRail', 'workWalk', 'workDrive']) for (const v of m[k]) a += v; return [h, a]; };
    const all = await dm.cellModes('all'), res = { all: all && tot(all).map(Math.round) };
    let h = 0, a = 0;
    // T-192: "school" already includes university, which is still queryable on its own (not summed)
    for (const p of ['work', 'school', 'university', 'shopping', 'social', 'airport', 'medical']) { const m = await dm.cellModes(p); if (!m) { res[p] = null; continue; } const t = tot(m); res[p] = t.map(Math.round); if (p !== 'university') { h += t[0]; a += t[1]; } }
    res.sum = [Math.round(h), Math.round(a)];
    return res;
  });
  result.phlBubbles = bubbles;
  check('Philadelphia bubbles: purposes add up to all', bubbles.all && Math.abs(bubbles.sum[0] - bubbles.all[0]) <= 2e-4 * bubbles.all[0] && ['school', 'university', 'shopping', 'social', 'airport', 'medical'].every(p => bubbles[p]?.[0] > 0), bubbles);
  const sel = await page.evaluate(async () => {
    const dm = window.__dm, cells = await dm.cityCells();
    const st = window.tw.world.value.save.stations.find(s => s.name === 'Philadelphia east');
    const pick = [];
    for (let c = 0; c < cells.n; c++) if (Math.hypot(cells.xy[2 * c] - st.x, cells.xy[2 * c + 1] - st.y) < 800) pick.push(c);
    const f = await dm.flows('home', Uint32Array.from(pick));
    if (!f) return null;
    const res = { cells: pick.length, all: Math.round(f.rail + f.walk + f.drive), rail: Math.round(f.rail) };
    let s = 0;
    for (const [k, e] of f.byPurpose) { res[k] = Math.round(e.rail + e.walk + e.drive); s += e.rail + e.walk + e.drive; }
    res.sum = Math.round(s);
    window.__sel = { kind: 'commuters', end: 'home', cells: pick, areas: 1 };
    return res;
  });
  result.phlSelection = sel;
  check('Philadelphia selection: purposes add up to its legs', sel && Math.abs(sel.sum - sel.all) <= 1e-3 * sel.all && Object.keys(sel).length > 6, sel);
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; window.tw.select(window.__sel); });
  await page.waitForTimeout(6000);
  await page.screenshot({ path: out + 'phl-selection.png' });
  result.inspector = (await page.evaluate(() => document.querySelector('#inspector')?.innerText ?? '')).slice(0, 1200);
  check('Philadelphia inspector names the everyday purposes', /School and university/.test(result.inspector) && /Shopping/.test(result.inspector) && /Medical/.test(result.inspector) && /Airport/.test(result.inspector), result.inspector.slice(0, 300));
  // 5. the purpose filter on airport and on medical, Philadelphia focused (T-142b screenshots)
  for (const p of ['airport', 'medical']) {
    await page.evaluate(p => { window.tw.select(null); window.__ui.purposeFilter.value = p; window.tw.map.jumpTo({ center: [-75.20, 39.92], zoom: 11 }); }, p);
    await page.waitForTimeout(5000);
    await page.screenshot({ path: out + 'filter-' + p + '.png' });
  }
  await page.evaluate(() => { window.__ui.purposeFilter.value = 'all'; });
  result.cities = await page.evaluate(() => Object.fromEntries([...window.twDemand.cities()].map(([id, v]) => [id, { rail: Math.round(v.railTripsPerDay ?? 0) }])));
  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
} finally {
  await writeFile(process.env.JSON_OUT ?? 'artifacts/trainworld/notes-outputs/T-142b-browser.json', JSON.stringify(result, null, 2));
  await ctx.close();
}
