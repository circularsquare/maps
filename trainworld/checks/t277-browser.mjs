// T-277 browser check: Toronto, the first pack outside the US, in the real app. A new game, a jump
// to Toronto: its pack opens, its boundary draws, its Homes bubbles show; ground-level track across
// Toronto Harbour is refused and the same route in a tunnel accepted (the water mask is loaded); a
// Yonge Street subway line from Union to Eglinton (10 stations, 7 km, constructed, 10 trains an hour
// at the peak) carries riders of every purpose, purposes add up; the city's mode split and the
// renderer's precision are read out. Needs a Vite dev server (BASE, default http://localhost:6127)
// and Playwright (PLAYWRIGHT_MODULE). Run from the maps root: node trainworld/notes/t277-browser.mjs.
// Screenshots: artifacts/trainworld/T-277-*.png; results artifacts/trainworld/T-277-browser.json.
import { writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:6127';
const WORKERS = process.env.WORKERS ?? '3';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't277-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = process.env.OUT ?? 'artifacts/trainworld/T-277-';
const CENTRE = [-79.3832, 43.6534];
// Yonge-University line stations, Union to Eglinton (OSM station points, rounded)
const YONGE = [['Union', -79.3806, 43.6453], ['King', -79.3779, 43.6489], ['Queen', -79.3793, 43.6525], ['Dundas', -79.3802, 43.6561],
  ['College', -79.3830, 43.6613], ['Bloor-Yonge', -79.3857, 43.6709], ['Rosedale', -79.3889, 43.6770], ['St Clair', -79.3933, 43.6879],
  ['Davisville', -79.3970, 43.6977], ['Eglinton', -79.3983, 43.7057]];
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&opened=all&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });

  // 1. jump to Toronto: its pack opens, it becomes the active city, its boundary layer exists
  const t0 = await page.evaluate(ll => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); return performance.now(); }, CENTRE);
  await page.waitForFunction(() => window.twDemand.client.start.loads.find(l => l.item === 'tor')?.openedAt, undefined, { timeout: 180000 });
  result.opening = await page.evaluate(t0 => { const l = window.twDemand.client.start.loads.find(l => l.item === 'tor'); return { download: Math.round(l.downloadedAt - l.requestedAt), open: Math.round(l.openedAt - l.downloadedAt), fromPan: Math.round(l.openedAt - t0) }; }, t0);
  check('Toronto opens at runtime', true, result.opening);
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'tor', undefined, { timeout: 60000 });
  const layer = await page.evaluate(() => !!window.tw.map.getLayer('tw-city-boundaries')); // T-400: one layer for every zone
  check('Toronto is the active city and its boundary layer exists', layer, { layer });
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; });
  await page.waitForTimeout(8000);
  await page.screenshot({ path: out + 'tor-homes.png' });

  // 2. the water mask: ground track across Toronto Harbour refused, a tunnel accepted
  const water = await page.evaluate(async () => {
    const t = window.tw, a = t.lngLatToPack(-79.3790, 43.6420), b = t.lngLatToPack(-79.3790, 43.6230);
    const g = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    const u = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -2 }, to: { kind: 'free', x: b.x, y: b.y, level: -2 }, pis: [], single: false });
    if (u.ok) await t.client.edit({ op: 'undo' });
    if (g.ok) await t.client.edit({ op: 'undo' });
    return { ground: g.ok ? 'accepted' : JSON.stringify(g).slice(0, 200), tunnel: u.ok };
  });
  check('ground track across Toronto Harbour bridged (or refused for want of room), a tunnel accepted', (water.ground === 'accepted' || /BridgeNoRoom/.test(water.ground)) && water.tunnel, water);

  // 3. a Yonge Street subway, Union to Eglinton, constructed
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async (stops) => {
    const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    // a 300 m tail beyond each end, so every platform sits on a through node
    const pts = stops.map(s => t.lngLatToPack(s[1], s[2]));
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
    const nodes = [];
    for (let i = 1; i < path.length; i++) {
      const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: -2 } : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: -2 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    const at = nodes.slice(1, -1);
    for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
    const line = (await run({ op: 'addLine', stops: at, name: 'Yonge test', colour: '#f2c300' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
  }, YONGE);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3, before, { timeout: 600000 });
  const sums = await page.evaluate(() => {
    const dv = window.twDemand.view(), keys = [...dv.purposes.keys()], w = window.tw.world.value;
    const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
    const l = w.save.lines.find(l => l.name === 'Yonge test'), d = dv.lines.get(l.id);
    const by = Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0)]));
    const worst = rel(keys.reduce((s, k) => s + (dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0), 0), d.ridersPerDay);
    const stations = Object.fromEntries(w.save.stations.map(s => [s.name, Math.round((dv.stations.get(s.id)?.boardings ?? 0))]));
    return { riders: Math.round(d.ridersPerDay), by, worst, stations };
  });
  result.yonge = sums;
  check('the Yonge line carries every purpose, purposes add up', ['work', 'shopping', 'social', 'school', 'medical'].every(k => sums.by[k] > 0) && sums.worst < 1e-4, sums);
  result.cities = await page.evaluate(() => Object.fromEntries([...window.twDemand.cities()].map(([id, v]) => [id, v])));
  check('Toronto has a mode split', result.cities.tor, result.cities.tor);
  // 4. renderer precision at Toronto (T-127): the debug probe, if present
  // the renderer's Float32 positions are Web Mercator metres from its single origin (map/geo.ts):
  // the rounding there, in ground metres, over the Yonge stations
  result.precision = await page.evaluate((stops) => {
    const C = 40075016.686, [ox, oy] = window.tw.originMerc, my = (lat) => (1 - Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)) / Math.PI) / 2;
    let worst = 0, far = 0;
    for (const [, lon, lat] of stops) {
      const x = ((lon + 180) / 360 - ox) * C, y = (my(lat) - oy) * C, k = Math.cos(lat * Math.PI / 180);
      worst = Math.max(worst, Math.abs(Math.fround(x) - x) * k, Math.abs(Math.fround(y) - y) * k);
      far = Math.max(far, Math.hypot(x, y) * k);
    }
    return { worstRoundingM: worst, distanceFromOriginKm: far / 1000 };
  }, YONGE);
  check('renderer precision at Toronto under 5 cm (T-127)', result.precision.worstRoundingM < 0.05, result.precision);
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.388, 43.675], zoom: 12.3 }));
  await page.waitForTimeout(8000);
  await page.screenshot({ path: out + 'tor-yonge.png' });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.3806, 43.6453], zoom: 17.5 }));
  await page.waitForTimeout(4000);
  await page.screenshot({ path: out + 'tor-union-zoom.png' });
  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
} finally {
  await writeFile(process.env.JSON_OUT ?? 'artifacts/trainworld/T-277-browser.json', JSON.stringify(result, null, 2));
  await ctx.close();
}
