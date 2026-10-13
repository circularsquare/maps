// T-277 rail share check: a rough TTC subway (Yonge, Union to Eglinton, 12 stations, the starting $6B buys no more; Bloor-Danforth,
// Jane to Main Street, 24 stations; both 20 trains an hour at the peak, tunnels), new game, round 3.
// Against the real pre-pandemic subway: about 1.2M riders a weekday on the whole system (Line 1
// about 0.7M), the census's 2016 GTHA transit commute share about 20-25% (all transit, buses and
// GO included). Same setup as t277-browser.mjs (BASE, PLAYWRIGHT_MODULE). Results:
// artifacts/trainworld/T-277-network.json, screenshot T-277-tor-network.png.
import { writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const BASE = process.env.BASE ?? 'http://localhost:6127';
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't277n-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--enable-unsafe-swiftshader'] });
const YONGE = [['Union', -79.3806, 43.6453], ['King', -79.3779, 43.6489], ['Queen', -79.3793, 43.6525], ['Dundas', -79.3802, 43.6561],
  ['College', -79.3830, 43.6613], ['Wellesley', -79.3840, 43.6654], ['Bloor-Yonge', -79.3857, 43.6709], ['Rosedale', -79.3889, 43.6770], ['Summerhill', -79.3907, 43.6823],
  ['St Clair', -79.3933, 43.6879], ['Davisville', -79.3970, 43.6977], ['Eglinton', -79.3983, 43.7057]
];
const BLOOR = [['Jane', -79.4842, 43.6500], ['Runnymede', -79.4759, 43.6516], ['High Park', -79.4666, 43.6540], ['Keele', -79.4597, 43.6555],
  ['Dundas West', -79.4527, 43.6566], ['Lansdowne', -79.4426, 43.6591], ['Dufferin', -79.4355, 43.6604], ['Ossington', -79.4263, 43.6624],
  ['Christie', -79.4183, 43.6641], ['Bathurst', -79.4111, 43.6660], ['Spadina', -79.4038, 43.6673], ['St George', -79.3995, 43.6683],
  ['Bay', -79.3901, 43.6702], ['Bloor-Yonge B', -79.3857, 43.6712], ['Sherbourne', -79.3764, 43.6722], ['Castle Frank', -79.3689, 43.6737],
  ['Broadview', -79.3583, 43.6768], ['Chester', -79.3523, 43.6783], ['Pape', -79.3449, 43.6799], ['Donlands', -79.3374, 43.6812],
  ['Greenwood', -79.3302, 43.6826], ['Coxwell', -79.3231, 43.6842], ['Woodbine', -79.3127, 43.6863], ['Main Street', -79.3013, 43.6890]];
const result = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(`${BASE}/?debug=1&new=1&paused=1&demandWorkers=3&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round >= 0, undefined, { timeout: 240000 });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.39, 43.70], zoom: 11.2 }));
  await page.waitForFunction(() => window.twDemand.client.start.loads.find(l => l.item === 'tor')?.openedAt, undefined, { timeout: 180000 });
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
  await page.evaluate(async ([lines]) => {
    const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    for (const [name, stops, level, colour] of lines) {
      const pts = stops.map(s => t.lngLatToPack(s[1], s[2]));
      const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
      const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
      const nodes = [];
      for (let i = 1; i < path.length; i++) {
        const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level } : { kind: 'node', node: nodes.at(-1) };
        await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level }, pis: [], single: false });
        const e = t.world.value.edges.at(-1);
        if (i === 1) nodes.push(e.a);
        nodes.push(e.b);
      }
      const at = nodes.slice(1, -1);
      for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
      const line = (await run({ op: 'addLine', stops: at, name, colour })).line;
      await run({ op: 'setTrainCount', line, trains: [20, 20, 10] });
    }
    await run({ op: 'constructAll' });
  }, [[['Yonge', YONGE, -2, '#f2c300'], ['Bloor-Danforth', BLOOR, -1, '#00a54f']]]);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view()?.round === 3, before, { timeout: 900000 });
  result.lines = await page.evaluate(() => {
    const dv = window.twDemand.view(), w = window.tw.world.value;
    return Object.fromEntries(w.save.lines.map(l => [l.name, Math.round(dv.lines.get(l.id)?.ridersPerDay ?? 0)]));
  });
  result.tor = await page.evaluate(() => { const v = window.twDemand.cities().get('tor'); return { split: v.split, tripsPerDay: Math.round(v.tripsPerDay), railTripsPerDay: Math.round(v.railTripsPerDay) }; });
  result.work = await page.evaluate(async () => {
    const dm = await import('/src/game/demand.ts'), m = await dm.cellModes('work');
    if (!m) return null;
    let rail = 0, all = 0;
    for (const k of ['homeRail', 'homeWalk', 'homeDrive']) for (const v of m[k]) { all += v; if (k === 'homeRail') rail += v; }
    return { legs: Math.round(all), rail: Math.round(rail), share: rail / all };
  });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-79.40, 43.70], zoom: 11.6 }));
  await page.waitForTimeout(8000);
  await page.screenshot({ path: 'artifacts/trainworld/T-277-tor-network.png' });
  result.errors = errors;
  console.log(JSON.stringify(result, null, 1));
} finally {
  await writeFile('artifacts/trainworld/T-277-network.json', JSON.stringify(result, null, 2));
  await ctx.close();
}
