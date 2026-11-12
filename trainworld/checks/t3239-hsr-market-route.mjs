// T-3239: the regional market's route search weighs fares as the market's riders do. Since T-1906 a
// game's curve is in its start country's dollars, and the market's weight was minutes a dollar of
// India's units, so its route choice saw every fare km 11.6x too heavy in a US start (7.3x in Japan):
// any high-speed multiplier above 1 sent every longer trip to a slower parallel line.
// 1. No browser: for a US start, a Japan start and a save from before T-1906, the market's minutes a
//    curve dollar (`marketFareWeight`) is every country's own at its default scale (within the scales'
//    tenths), and the mode choice's weight (that times `fareFactors`) is each country's at its scale.
// 2. A sandbox game: Tokyo - Koriyama (about 200 km), a high-speed line and an ordinary one held to
//    80 km/h, two stations side by side at each end; at the default 1.5x the high-speed line, 110
//    minutes faster, carries at least 90% of the longer trips between them.
//   node trainworld/checks/t3239-hsr-market-route.mjs [url]   (from the maps root; default localhost:6132; about 2 minutes)
// Prints PASS or FAIL lines and exits 1 on a failure.
import { collectErrors, demandFinal, launchChrome, loadApp } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6132/';
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const inc = await loadApp("export { marketFareWeight, fareFactors, fareScale, defaultScale, fareWeight, incomeScale } from '@app/src/game/income.ts';");
const COUNTRIES = ['US', 'JP', 'IN', 'PL', 'CN', 'MX', 'GB', 'NG'];
for (const [name, scales] of [['a US start', { '@': 11.6 }], ['a Japan start', { '@': 7.3 }], ['a Japan start, Poland at 1.4x', { '@': 7.3, PL: 1.4 }], ['a save from before T-1906', undefined]]) {
  const w = inc.marketFareWeight(scales), f = inc.fareFactors(scales);
  let worstRoute = 0, worstMode = 0;
  for (const c of COUNTRIES) {
    const own = inc.fareWeight(c) * inc.defaultScale(c, scales), set = inc.fareWeight(c) * inc.fareScale(c, scales);
    worstRoute = Math.max(worstRoute, Math.abs(w / own - 1));
    worstMode = Math.max(worstMode, Math.abs((w * (f[c] ?? 1)) / set - 1));
  }
  report(`${name}: the market's route search weighs a curve dollar as riders do at their default`, worstRoute < 0.06, `(${w.toFixed(2)} min a dollar, worst ${(100 * worstRoute).toFixed(1)}% off)`);
  report(`${name}: its mode choice weighs each country's fares at their scale`, worstMode < 0.06, `(worst ${(100 * worstMode).toFixed(1)}% off)`);
}

const A = [139.7671, 35.6812], B = [140.3888, 37.3981];
const { ctx, close } = await launchChrome({ prefix: 'tw-t3239-', size: [1200, 800] });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 10 }), A);
  const made = await page.evaluate(async ([A, B]) => {
    const tw = window.tw, c = tw.client;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const ids = () => new Set(tw.world.value.nodes.keys());
    const nearIn = (set, p) => [...set].sort((m, n) => { const a = tw.world.value.nodes.get(m), b = tw.world.value.nodes.get(n); return Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y); })[0];
    const pa = tw.lngLatToPack(...A), pb = tw.lngLatToPack(...B);
    const dx = pb.x - pa.x, dy = pb.y - pa.y, L = Math.hypot(dx, dy), ox = (dy / L) * 60, oy = (-dx / L) * 60;
    // the ordinary track 60 m beside the high-speed one: the stations side by side are one demand station
    const lay = async (hsr, ox, oy) => {
      const before = ids();
      await c.edit({ op: 'route', from: { kind: 'free', x: pa.x + ox, y: pa.y + oy, level: 1 }, to: { kind: 'free', x: pb.x + ox, y: pb.y + oy, level: 1 }, pis: [], single: false, through: true, hsr });
      await sleep(800);
      const fresh = new Set([...ids()].filter((i) => !before.has(i)));
      return [nearIn(fresh, { x: pa.x + ox, y: pa.y + oy }), nearIn(fresh, { x: pb.x + ox, y: pb.y + oy })];
    };
    const h = await lay(true, 0, 0), g = await lay(false, ox, oy);
    for (const n of [...h, ...g]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }); await sleep(300); }
    await c.edit({ op: 'constructAll' });
    await sleep(800);
    const lh = await c.edit({ op: 'addLine', stops: h, name: 'High speed', colour: '#d33682' });
    await sleep(400);
    const lg = await c.edit({ op: 'addLine', stops: g, name: 'Ordinary', colour: '#268bd2' });
    await sleep(400);
    await c.edit({ op: 'lineFast', line: lh.line, fast: true });
    await c.edit({ op: 'setHeadway', line: lh.line, seconds: [900, 1200, 1800] });
    await c.edit({ op: 'lineFast', line: lg.line, fast: false });
    await c.edit({ op: 'setHeadway', line: lg.line, seconds: [450, 600, 1200] });
    await c.edit({ op: 'lineLimit', line: lg.line, kmh: 80 });
    await c.edit({ op: 'fareHsr', mul: 1.5 });
    await sleep(1500);
    return { lh: lh?.line, lg: lg?.line };
  }, [A, B]);
  await demandFinal(page, { timeout: 600000 });
  const r = await page.evaluate(([lh, lg]) => {
    const v = window.twDemand.view(), p = v.purposes?.get('longer')?.lines, s = window.tw.world.value.lineStats;
    const run = (l) => (s[l]?.fromStartByLevel?.[0]?.at(-1) ?? NaN) / 60;
    return { hsr: p?.get(String(lh))?.ridersPerDay ?? 0, reg: p?.get(String(lg))?.ridersPerDay ?? 0, hsrMin: run(lh), regMin: run(lg), mul: window.tw.moneyState.value.fares.hsr };
  }, [made.lh, made.lg]);
  const share = r.hsr / Math.max(1e-9, r.hsr + r.reg);
  report('Tokyo - Koriyama at 1.5x: the high-speed line carries the longer trips', r.mul === 1.5 && r.hsr + r.reg > 100 && share >= 0.9,
    `(${Math.round(r.hsr)} against ${Math.round(r.reg)} a day, ${(100 * share).toFixed(1)}%; ${r.hsrMin.toFixed(0)} against ${r.regMin.toFixed(0)} minutes)`);
} catch (e) {
  report('ran to the end', false, e.message.split('\n')[0]);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-3239 PASSED' : 'T-3239 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
