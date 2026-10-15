// T-394's experiment (notes/T-394.md): does opening a zone's pack for a station past the 4.5 km
// reach add its riders? A new game viewed in Boston (so Worcester is not the viewed zone); a test
// line from a station 6.2 km east of Worcester's outline to Back Bay, tunnelled and constructed.
// Prints the final solve's cities, the station's riders and Worcester's rail trips. Its checks
// pass only with cityReach.ts's REACH_M widened to 15,500 (they show the zone opened); the riders
// are the same either way.
//
//     node trainworld/notes/T-394-reach-ab.mjs        (from the maps root)
//
// Needs a Vite dev server (BASE, default http://localhost:6194) and Playwright (PLAYWRIGHT_MODULE,
// else `playwright`, else the npx cache). Results artifacts/trainworld/T-394-browser.json.
import { writeFile, mkdir, mkdtemp, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const STOPS = [['Outlying test', -71.44, 42.238], ['Wellesley test', -71.2, 42.3], ['Back Bay test', -71.0752, 42.3473]];
const ZONE = 'wor', VIEW = [-71.06, 42.36];

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

const { chromium } = await playwright();
const BASE = process.env.BASE ?? 'http://localhost:6194';
const profile = await mkdtemp(join(tmpdir(), 'tw-t394-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const IGNORE = /favicon|openfreemap|AJAXError/;
const result = { checks: [] };
const check = (name, ok, detail) => { result.checks.push({ name, ok: !!ok, detail }); console.log(ok ? 'ok  ' : 'FAIL', name, JSON.stringify(detail ?? '')); };
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !IGNORE.test(m.text())) errors.push(m.text()); });
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client, undefined, { timeout: 240000 });

  // view Boston: its pack opens and is the viewed zone
  await page.evaluate((ll) => window.tw.map.jumpTo({ center: ll, zoom: 10.5 }), VIEW);
  await page.waitForFunction(() => window.twDemand.client.start.loads.find((l) => l.item === 'bos')?.openedAt, undefined, { timeout: 300000 });
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'bos', undefined, { timeout: 60000 });
  check('Boston is the viewed zone', true);

  // the test line: a 300 m tail beyond each end so every platform sits on a through node, tunnelled
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  const ids = await page.evaluate(async ([stops, name]) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const pts = stops.map((s) => t.lngLatToPack(s[1], s[2]));
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
    const line = (await run({ op: 'addLine', stops: at, name, colour: '#cc3333' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
    return { line, stations: Object.fromEntries(t.world.value.save.stations.map((s) => [s.name, s.id])) };
  }, [STOPS, 'T-394 test']);
  await page.waitForFunction((v) => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3 && window.twDemand.view()?.final, before, { timeout: 900000 });
  await page.waitForTimeout(3000);

  const out = await page.evaluate(async ([ids, zone]) => {
    const d = window.twDemand, v = d.view(), s = d.log.at(-1);
    const sid = ids.stations['Outlying test'];
    const r = await d.stationRiders(sid);
    return {
      version: v.version, final: !!v.final, solveCities: s.cities, bare: d.client.isBare(zone), opened: d.client.opened.has(zone),
      zoneRail: Math.round(d.cities().get(zone)?.railTripsPerDay ?? -1), bosRail: Math.round(d.cities().get('bos')?.railTripsPerDay ?? -1),
      stationBoardings: Math.round(v.stations.get(sid)?.boardings ?? 0), homeCommuters: Math.round(r?.home.total ?? 0), rode: Math.round(r?.rode ?? 0),
    };
  }, [ids, ZONE]);
  result.after = out;
  console.log(JSON.stringify(out));
  check(`the final solve covers ${ZONE}`, out.final && out.solveCities.includes(ZONE), { solveCities: out.solveCities });
  check(`${ZONE} is open in the workers, not bare`, out.opened && !out.bare, { opened: out.opened, bare: out.bare });
  check('commuters ride to the station 10 km outside the zone', out.rode > 0, { rode: out.rode, homeCommuters: out.homeCommuters, boardings: out.stationBoardings });
  check('no page errors', errors.length === 0, errors);
} finally {
  await mkdir('artifacts/trainworld', { recursive: true });
  await writeFile(process.env.JSON_OUT ?? 'artifacts/trainworld/T-394-browser.json', JSON.stringify(result, null, 2));
  await ctx.close();
  await rm(profile, { recursive: true, force: true }).catch(() => {});
}
const pass = result.checks.length === 5 && result.checks.every((c) => c.ok);
console.log(pass ? 'PASS t394-browser' : 'FAIL t394-browser');
process.exit(pass ? 0 : 1);
