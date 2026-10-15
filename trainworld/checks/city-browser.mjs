// A new city in the real app (T-364, from T-288's London check): a new game, a jump to the city's
// centre: its pack opens, it becomes the active city with its boundary layer, the top bar names it
// and gives its no-rail split; ground-level track across a water crossing is refused and the same
// route in a tunnel accepted; a test line through the centre (deep tunnel, constructed, 10 trains an
// hour at the peak) reaches a final solve and carries riders of every purpose, purposes adding up;
// no page errors. Cities are the table below: add one per new city.
//
//     node trainworld/checks/city-browser.mjs <id>        (from the maps root)
//
// Needs a Vite dev server (BASE, default http://localhost:6125) and Playwright (PLAYWRIGHT_MODULE,
// else `playwright`, else the npx cache). Screenshots artifacts/trainworld/<task>-<id>-homes.png and
// -line.png; results <task>-<id>-browser.json (OUT and JSON_OUT override).
import { writeFile, mkdtemp, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

// name: as the top bar shows it; water: two points either side of a river or bay; line: [station, lon, lat]
const CITIES = {
  lon: {
    task: 'T-288', name: 'London', centre: [-0.1246, 51.5073], water: [[-0.1100, 51.5125], [-0.1080, 51.5040]],
    lineName: 'Central test', colour: '#dc241f', view: [-0.14, 51.514],
    line: [['Notting Hill Gate', -0.1967, 51.5090], ['Queensway', -0.1874, 51.5107], ['Lancaster Gate', -0.1756, 51.5119],
      ['Marble Arch', -0.1584, 51.5136], ['Bond Street', -0.1494, 51.5142], ['Oxford Circus', -0.1415, 51.5152],
      ['Tottenham Court Road', -0.1308, 51.5165], ['Holborn', -0.1200, 51.5174], ['Chancery Lane', -0.1111, 51.5185],
      ["St Paul's", -0.0976, 51.5146], ['Bank', -0.0886, 51.5133], ['Liverpool Street', -0.0823, 51.5178]],
  },
  par: {
    task: 'T-289', name: 'Paris', centre: [2.3488, 48.8534], water: [[2.3190, 48.8652], [2.3190, 48.8610]],
    lineName: 'Ligne 1 test', colour: '#ffcd00', view: [2.345, 48.861],
    line: [['Charles de Gaulle Etoile', 2.2950, 48.8738], ['George V', 2.3006, 48.8720], ['Franklin D. Roosevelt', 2.3095, 48.8689],
      ['Champs-Elysees Clemenceau', 2.3145, 48.8676], ['Concorde', 2.3229, 48.8656], ['Tuileries', 2.3297, 48.8645],
      ['Palais Royal', 2.3364, 48.8625], ['Louvre Rivoli', 2.3409, 48.8609], ['Chatelet', 2.3475, 48.8585],
      ['Hotel de Ville', 2.3522, 48.8573], ['Saint-Paul', 2.3610, 48.8551], ['Bastille', 2.3691, 48.8531],
      ['Gare de Lyon', 2.3733, 48.8448], ['Reuilly Diderot', 2.3872, 48.8472], ['Nation', 2.3959, 48.8484]],
  },
};

async function playwright() {
  if (process.env.PLAYWRIGHT_MODULE) return import(process.env.PLAYWRIGHT_MODULE);
  try { return await import('playwright'); } catch { /* not installed here */ }
  const npx = join(process.env.LOCALAPPDATA ?? '', 'npm-cache', '_npx');
  for (const d of existsSync(npx) ? await readdir(npx) : []) {
    const p = join(npx, d, 'node_modules', 'playwright', 'index.mjs');
    if (existsSync(p)) return import(pathToFileURL(p).href);
  }
  throw new Error('no Playwright: set PLAYWRIGHT_MODULE to a file URL of playwright/index.mjs');
}

const id = process.argv[2];
const C = CITIES[id];
if (!C) { console.error(`usage: city-browser.mjs <${Object.keys(CITIES).join('|')}>`); process.exit(2); }
const { chromium } = await playwright();
const BASE = process.env.BASE ?? 'http://localhost:6125';
const WORKERS = process.env.WORKERS ?? '3';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), `cb-${id}-`));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const out = process.env.OUT ?? `artifacts/trainworld/${C.task}-${id}-`;
const IGNORE = /favicon|openfreemap|AJAXError/;
const result = { city: id, checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
const topBar = (page) => page.evaluate(() => ({ city: document.querySelector('.topbar .city, .city')?.textContent ?? null, split: document.querySelector('.split-row')?.getAttribute('data-tip') ?? null }));
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error' && !IGNORE.test(m.text())) errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&demandWorkers=${WORKERS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });

  // 1. jump to the city: its pack opens, it becomes the active city, its boundary layer exists
  const t0 = await page.evaluate(ll => { window.tw.map.jumpTo({ center: ll, zoom: 11.2 }); return performance.now(); }, C.centre);
  await page.waitForFunction(id => window.twDemand.client.start.loads.find(l => l.item === id)?.openedAt, id, { timeout: 300000 });
  result.opening = await page.evaluate(([id, t0]) => { const l = window.twDemand.client.start.loads.find(l => l.item === id); return { download: Math.round(l.downloadedAt - l.requestedAt), open: Math.round(l.openedAt - l.downloadedAt), fromPan: Math.round(l.openedAt - t0) }; }, [id, t0]);
  check(`${C.name} opens at runtime`, true, result.opening);
  await page.waitForFunction(id => window.twDemand.activeCity.value === id, id, { timeout: 60000 });
  const layer = await page.evaluate(id => !!window.tw.map.getLayer(`tw-city-boundary-${id}`), id);
  check(`${C.name} is the active city and its boundary layer exists`, layer, { layer });
  await page.evaluate(() => { const d = document.querySelector('#debug'); if (d) d.style.display = 'none'; window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' }; });
  // the no-rail split: wait until the top bar has the city's own figures
  await page.waitForFunction(name => /Train .*Walking .*Driving/.test(document.querySelector('.split-row')?.getAttribute('data-tip') ?? '') && (document.querySelector('.split-row')?.getAttribute('data-tip') ?? '').includes(name) && !/–/.test(document.querySelector('.split-row .pct')?.textContent ?? '–'), C.name, { timeout: 300000 });
  await page.waitForTimeout(6000);
  result.noRail = { topBar: await topBar(page), cities: await page.evaluate(id => window.twDemand.cities().get(id) ?? null, id) };
  check(`the top bar names ${C.name} and gives its split`, (result.noRail.topBar.city ?? '').includes(C.name) && /Walking/.test(result.noRail.topBar.split ?? ''), result.noRail);
  await page.screenshot({ path: out + 'homes.png' });

  // 2. the water mask: ground track across the crossing refused, a tunnel accepted
  const water = await page.evaluate(async ([p, q]) => {
    const t = window.tw, a = t.lngLatToPack(p[0], p[1]), b = t.lngLatToPack(q[0], q[1]);
    const g = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    const u = await t.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -2 }, to: { kind: 'free', x: b.x, y: b.y, level: -2 }, pis: [], single: false });
    if (u.ok) await t.client.edit({ op: 'undo' });
    return { ground: g.ok ? 'accepted' : JSON.stringify(g).slice(0, 200), tunnel: u.ok };
  }, C.water);
  check('ground track across the water refused, a tunnel accepted', /GroundOverWater/.test(water.ground) && water.tunnel, water);

  // 3. a test line through the centre, constructed
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async ([stops, name, colour]) => {
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
    const line = (await run({ op: 'addLine', stops: at, name, colour })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
  }, [C.line, C.lineName, C.colour]);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3 && window.twDemand.view()?.final, before, { timeout: 900000 });
  const sums = await page.evaluate(name => {
    const dv = window.twDemand.view(), keys = [...dv.purposes.keys()], w = window.tw.world.value;
    const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
    const l = w.save.lines.find(l => l.name === name), d = dv.lines.get(l.id);
    const by = Object.fromEntries(keys.map(k => [k, Math.round(dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0)]));
    const worst = rel(keys.reduce((s, k) => s + (dv.purposes.get(k).lines.get(l.id)?.ridersPerDay ?? 0), 0), d.ridersPerDay);
    const stations = Object.fromEntries(w.save.stations.map(s => [s.name, Math.round((dv.stations.get(s.id)?.boardings ?? 0))]));
    return { final: !!dv.final, riders: Math.round(d.ridersPerDay), by, worst, stations };
  }, C.lineName);
  result.line = sums;
  check(`the ${C.lineName} reaches a final solve with riders`, sums.final && sums.riders > 0, { final: sums.final, riders: sums.riders });
  check(`the ${C.lineName} carries every purpose, purposes add up`, ['work', 'shopping', 'social', 'school', 'medical'].every(k => sums.by[k] > 0) && sums.worst < 1e-4, sums.by);
  await page.waitForTimeout(4000);
  result.withLine = { topBar: await topBar(page), cities: await page.evaluate(id => window.twDemand.cities().get(id) ?? null, id) };
  check(`${C.name} has a mode split with the line`, result.withLine.cities, result.withLine);
  await page.evaluate(ll => window.tw.map.jumpTo({ center: ll, zoom: 12.6 }), C.view);
  await page.waitForTimeout(8000);
  await page.screenshot({ path: out + 'line.png' });
  result.errors = errors;
  check('no page errors', errors.length === 0, errors);
} finally {
  await writeFile(process.env.JSON_OUT ?? `artifacts/trainworld/${C.task}-${id}-browser.json`, JSON.stringify(result, null, 2));
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const pass = result.checks.length === 8 && result.checks.every(c => c.ok);
console.log(pass ? `PASS city-browser ${id}` : `FAIL city-browser ${id}`);
process.exit(pass ? 0 : 1);
