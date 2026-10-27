import { launchChrome, demandFinal, buildLine, collectErrors, ANGLE } from './lib.mjs';
// T-1515: with the select tool, lines (rail and bus), stations and trains are picked before the
// demand bubbles drawn over them; a bubble still takes a click where none of those is near. In a
// ?new=1 game with the commuter bubbles on at zoom 15 in Midtown, a rail line and a bus line cross
// bubbles: a click on either line inside a bubble selects the line and no bubble tag shows there, a
// click on the middle station selects it, and a click on a bubble well away from the network selects
// the bubble with its tag showing. Trains are off so a passing train cannot take the line's click.
//   node trainworld/checks/t1515-click-priority.mjs [url]     (from the maps root; default localhost:6115)
// Prints PASS or FAIL lines and exits 1 on failure. About two minutes.

const BASE = process.argv[2] ?? 'http://localhost:6115/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t1515-', size: [1200, 800], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 180000 });
  const [lng, lat] = [-73.985, 40.75];
  const dLng = 0.6 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const rail = await buildLine(page, [-1, 0, 1].map((k) => [lng + k * dLng, lat]), { trains: [6, 4, 3] });
  const bus = await page.evaluate(async ([from, to]) => {
    const tw = window.tw, c = tw.client;
    const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
    const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
    const con = await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    return { line: add.line, ok: add.ok && con.ok };
  }, [[lng + 0.004, lat + 0.006], [lng + 0.004, lat + 0.0003]]);
  await page.evaluate(([lng, lat]) => {
    const tw = window.tw;
    tw.display.value = { ...tw.display.value, commuters: true, trains: false };
    tw.map.jumpTo({ center: [lng, lat + 0.002], zoom: 15 });
  }, [lng, lat]);
  await demandFinal(page, { strict: false });
  await page.waitForTimeout(3000);
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  // points on a grid: what the network picks there and whether a bubble lies under the pointer
  const grid = await page.waitForFunction(([rail, bus]) => {
    const tw = window.tw, W = tw.map.getContainer().clientWidth, H = tw.map.getContainer().clientHeight;
    const out = { rail: null, bus: null, station: null, bubble: null };
    // the middle of the map, clear of the panels and bars round its edges
    for (let y = H * 0.25; y < H * 0.75; y += 3) for (let x = W * 0.25; x < W * 0.75; x += 3) {
      const net = tw.pick(tw.overlay, x, y, false);
      const under = window.twViews.click(x, y, false, true);
      if (!under) continue;
      if (net?.kind === 'line' && net.line === String(rail) && !out.rail) out.rail = { x, y, under: under.kind };
      if (net?.kind === 'line' && net.line === String(bus) && !out.bus) out.bus = { x, y, under: under.kind };
      if (net?.kind === 'station' && !out.station) out.station = { x, y, under: under.kind };
      // a bubble with nothing of the network within 20 px
      if (!net && !out.bubble && [[20, 0], [-20, 0], [0, 20], [0, -20], [14, 14], [-14, -14], [14, -14], [-14, 14]].every(([dx, dy]) => !tw.pick(tw.overlay, x + dx, y + dy, false))) out.bubble = { x, y, under: under.kind };
    }
    return out.rail && out.bus && out.station && out.bubble ? out : null;
  }, [rail.line, bus.line], { timeout: 120000, polling: 2000 }).then((h) => h.jsonValue());
  const clickAt = async ({ x, y }) => {
    const net = await page.evaluate(([x, y]) => { window.tw.selection.value = null; return window.tw.pick(window.tw.overlay, x, y, false); }, [x, y]);
    if (process.env.DEBUG) console.log('at', x, y, JSON.stringify(net), JSON.stringify(await page.evaluate(() => window.tw.map.getCenter())));
    await page.mouse.move(box.left + x, box.top + y);
    await page.waitForTimeout(150);
    const tag = await page.evaluate(() => { const t = document.querySelector('.demand-tag:not(.rest-list)'); return !!t && t.style.display !== 'none' && t.textContent.trim() !== ''; });
    await page.mouse.click(box.left + x, box.top + y);
    await page.waitForTimeout(300);
    return { sel: await page.evaluate(() => window.tw.selection.value), tag };
  };
  const r = await clickAt(grid.rail);
  report('click on a rail line inside a bubble selects the line', r.sel?.kind === 'line' && r.sel.line === String(rail.line), `(at ${grid.rail.x},${grid.rail.y} over a ${grid.rail.under} bubble: ${JSON.stringify(r.sel)})`);
  report('no bubble tag over the rail line', !r.tag);
  const b = await clickAt(grid.bus);
  report('click on a bus line inside a bubble selects the bus line', bus.ok && b.sel?.kind === 'line' && b.sel.line === String(bus.line), `(${JSON.stringify(b.sel)})`);
  report('no bubble tag over the bus line', !b.tag);
  const s = await clickAt(grid.station);
  report('click on a station inside a bubble selects the station', s.sel?.kind === 'station', `(${JSON.stringify(s.sel)})`);
  const o = await clickAt(grid.bubble);
  report('click on a bubble away from the network selects the bubble', !!o.sel && o.sel.kind === grid.bubble.under, `(${JSON.stringify(o.sel)})`);
  report('the bubble tag shows away from the network', o.tag);
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-1515 PASSED' : 'T-1515 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
