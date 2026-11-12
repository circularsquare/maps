// T-3239: the regional market's route search weighs fares as its longer trips do. Since T-1906 a
// game's curve is in its start country's dollars, and the market's weight was minutes a dollar of
// India's units, so its route choice saw every fare km 11.6x too heavy in a US start (7.3x in Japan).
// Then (Anita's option A) the search charges a longer trip's weight, 0.4 of a local one's, so a long
// trip on a high-speed line no longer changes to a slower line for the last stretch to save its fare.
// 1. No browser: for a US start, a Japan start (with and without a scale set) and a save from before
//    T-1906, the market's route search weighs a curve dollar at 0.4 of every country's own at its
//    default scale (within the scales' tenths), the commutes' mode choice (that times `fareFactors`)
//    at each country's own at its scale, and every market row's country has a factor.
// 2. A sandbox game: a high-speed line Sendai - Omiya - Tokyo, an ordinary line Omiya - Tokyo beside
//    it, both at their top speed; at the default 1.5x, under 5% of the longer trips Sendai -> Tokyo
//    by rail change at Omiya (74% did with the search at a local trip's weight, nearly all before).
//   node trainworld/checks/t3239-hsr-market-route.mjs [url]   (from the maps root; default localhost:6132; about 2 minutes)
// Prints PASS or FAIL lines and exits 1 on a failure.
import { readFileSync } from 'node:fs';
import { collectErrors, demandFinal, launchChrome, loadApp } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6132/';
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const inc = await loadApp("export { marketFareWeight, fareFactors, fareScale, defaultScale, fareWeight, LONG_TRIP_MONEY } from '@app/src/game/income.ts';");
const rows = JSON.parse(readFileSync(new URL('../app/src/game/market-rows.json', import.meta.url), 'utf8'));
const rowCountries = [...new Set(Object.values(rows).map((r) => r.country))];
for (const [name, scales] of [['a US start', { '@': 11.6 }], ['a Japan start', { '@': 7.3 }], ['a Japan start, Poland at 1.4x', { '@': 7.3, PL: 1.4 }], ['a save from before T-1906', undefined]]) {
  const w = inc.marketFareWeight(scales), f = inc.fareFactors(scales);
  let worstRoute = 0, worstMode = 0;
  for (const c of rowCountries) {
    const own = inc.fareWeight(c) * inc.defaultScale(c, scales), set = inc.fareWeight(c) * inc.fareScale(c, scales);
    worstRoute = Math.max(worstRoute, Math.abs(w / (inc.LONG_TRIP_MONEY * own) - 1));
    const e = Math.abs((w * (f[c] ?? NaN)) / set - 1);
    worstMode = Math.max(worstMode, Number.isNaN(e) ? Infinity : e);
  }
  report(`${name}: the market's route search weighs a curve dollar at 0.4 of riders' own`, worstRoute < 0.06, `(${w.toFixed(2)} min a dollar, worst ${(100 * worstRoute).toFixed(1)}% off over ${rowCountries.length} countries)`);
  report(`${name}: commutes between zones weigh each country's fares at their scale`, worstMode < 0.06, `(worst ${(100 * worstMode).toFixed(1)}% off)`);
}

const SENDAI = [140.8822, 38.2601], OMIYA = [139.6239, 35.9063], TOKYO = [139.7671, 35.6812];
const { ctx, close } = await launchChrome({ prefix: 'tw-t3239-', size: [1200, 800] });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 10 }), TOKYO);
  const made = await page.evaluate(async ([S, O, T]) => {
    const tw = window.tw, c = tw.client;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const ids = () => new Set(tw.world.value.nodes.keys());
    const nearIn = (set, p) => [...set].sort((m, n) => { const a = tw.world.value.nodes.get(m), b = tw.world.value.nodes.get(n); return Math.hypot(a.x - p.x, a.y - p.y) - Math.hypot(b.x - p.x, b.y - p.y); })[0];
    const route = async (from, q, hsr) => {
      const b = ids();
      await c.edit({ op: 'route', from, to: { kind: 'free', x: q.x, y: q.y, level: 1 }, pis: [], single: false, through: true, hsr });
      await sleep(800);
      return new Set([...ids()].filter((i) => !b.has(i)));
    };
    const ps = tw.lngLatToPack(...S), po = tw.lngLatToPack(...O), pt = tw.lngLatToPack(...T);
    // Tokyo - Omiya straight on 10 km, then bending north and on to Sendai, no corner over 30 degrees
    const d = Math.hypot(po.x - pt.x, po.y - pt.y), pp = { x: po.x + ((po.x - pt.x) / d) * 10000, y: po.y + ((po.y - pt.y) / d) * 10000 };
    const n1 = await route({ kind: 'free', x: pt.x, y: pt.y, level: 1 }, po, true);
    const hT = nearIn(n1, pt), hO = nearIn(n1, po);
    let at = hO;
    for (const q of [pp, tw.lngLatToPack(139.5728, 36.2567), ps]) at = nearIn(await route({ kind: 'node', node: at }, q, true), q);
    const ox = ((po.y - pt.y) / d) * 60, oy = (-(po.x - pt.x) / d) * 60;
    const n3 = await route({ kind: 'free', x: po.x + ox, y: po.y + oy, level: 1 }, { x: pt.x + ox, y: pt.y + oy }, false);
    const gO = nearIn(n3, { x: po.x + ox, y: po.y + oy }), gT = nearIn(n3, { x: pt.x + ox, y: pt.y + oy });
    for (const n of [at, hO, hT, gO, gT]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }); await sleep(300); }
    await c.edit({ op: 'constructAll' });
    await sleep(800);
    const lh = await c.edit({ op: 'addLine', stops: [at, hO, hT], name: 'High speed', colour: '#d33682' });
    await sleep(400);
    const lg = await c.edit({ op: 'addLine', stops: [gO, gT], name: 'Ordinary', colour: '#268bd2' });
    await sleep(400);
    await c.edit({ op: 'lineFast', line: lh.line, fast: true });
    await c.edit({ op: 'setHeadway', line: lh.line, seconds: [900, 1200, 1800] });
    await c.edit({ op: 'lineFast', line: lg.line, fast: false });
    await c.edit({ op: 'setHeadway', line: lg.line, seconds: [450, 600, 1200] });
    await c.edit({ op: 'fareHsr', mul: 1.5 });
    await sleep(1500);
    const { REGISTRY } = await import('/src/game/cities.ts');
    const pl = REGISTRY.market.places;
    return { ok: !!(lh?.ok && lg?.ok), lg: lg.line, sendai: pl.findIndex((p) => p.name === 'Sendai'), tokyo: pl.findIndex((p) => p.name === 'Tokyo') };
  }, [SENDAI, OMIYA, TOKYO]);
  if (!made.ok || made.sendai < 0 || made.tokyo < 0) throw new Error(`not built: ${JSON.stringify(made)}`);
  await demandFinal(page, { timeout: 600000 });
  const r = await page.evaluate(async (m) => {
    const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), m.tokyo, -1, { place: m.sendai, gid: -1 });
    let rail = 0, change = 0;
    for (const p of ps ?? []) {
      if (!p) continue;
      rail += p.rail;
      for (const rt of p.routes) if (rt.legs.some((l) => l.kind === 'ride' && l.line === String(m.lg))) change += p.rail * rt.share;
    }
    return { rail, change, mul: window.tw.moneyState.value.fares.hsr };
  }, made);
  report('Sendai -> Tokyo at 1.5x: longer trips stay on the high-speed line past Omiya', r.mul === 1.5 && r.rail > 100 && r.change < 0.05 * r.rail,
    `(${Math.round(r.rail)} by rail a day, ${Math.round(r.change)} change at Omiya)`);
} catch (e) {
  report('ran to the end', false, e.message.split('\n')[0]);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-3239 PASSED' : 'T-3239 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
