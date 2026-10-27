import { launchChrome, demandFinal, buildLine, collectErrors, ANGLE } from './lib.mjs';
// T-1550: zoomed out to the zone bubbles (zoom 8.5 and below), a click and the hover tag pick the
// bubble before a line or station under it; closer in (T-1515's order) the line or station wins. A
// ?new=1 game with a three-station line in Midtown, Destinations on: at zoom 7 and at 8.4 a point
// where the network and a bubble meet selects the bubble and shows its tag; at zoom 12 the same
// kind of point selects the network.
//   node trainworld/checks/t1550-far-bubbles.mjs [url]     (from the maps root; default localhost:6170)
// Prints PASS or FAIL lines and exits 1 on failure. About two minutes.

const BASE = process.argv[2] ?? 'http://localhost:6170/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t1550-', size: [1200, 800], args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 180000 });
  const [lng, lat] = [-73.985, 40.75];
  const dLng = 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const rail = await buildLine(page, [-1, 0, 1].map((k) => [lng + k * dLng, lat]), { trains: [6, 4, 3] });
  await page.evaluate(() => { window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'work', trains: false }; });
  await demandFinal(page, { strict: false });

  /** a point where the network and a bubble (of kind `under`) both lie, found on a one-pixel grid round the line */
  const findMeet = async (zoom, under) => {
    await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [[lng, lat], zoom]);
    await page.waitForTimeout(2500);
    return page.waitForFunction((under) => {
      const tw = window.tw, W = tw.map.getContainer().clientWidth, H = tw.map.getContainer().clientHeight;
      const out = { line: null, station: null };
      for (let y = H / 2 - 60; y < H / 2 + 60; y++) for (let x = W / 2 - 80; x < W / 2 + 80; x++) {
        const net = tw.pick(tw.overlay, x, y, false);
        if (net?.kind !== 'line' && net?.kind !== 'station') continue;
        const hit = window.twViews.click(x, y, false, true);
        if (!hit || (under && hit.kind !== under)) continue;
        // well inside the network's reach, not on its edge
        if (![[2, 0], [-2, 0], [0, 2], [0, -2]].every(([dx, dy]) => tw.pick(tw.overlay, x + dx, y + dy, false)?.kind === net.kind)) continue;
        if (net.kind === 'line' && !out.line) out.line = { x, y };
        if (net.kind === 'station' && !out.station) out.station = { x, y };
      }
      return out.line || out.station ? out : null;
    }, under, { timeout: 60000, polling: 1000 }).then((h) => h.jsonValue());
  };
  const clickAt = async ({ x, y }) => {
    const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
    await page.evaluate(() => { window.tw.selection.value = null; });
    await page.mouse.move(box.left + x - 30, box.top + y);
    await page.mouse.move(box.left + x, box.top + y);
    await page.waitForTimeout(300);
    if (process.env.DEBUG) console.log('at', x, y, JSON.stringify(await page.evaluate(async ([x, y]) => [window.tw.pick(window.tw.overlay, x, y, false), window.tw.map.getZoom(), (await import('/src/map/intercityView.ts')).cityOverview.peek()], [x, y])));
    const tag = await page.evaluate(() => { const t = document.querySelector('.demand-tag:not(.rest-list)'); return !!t && t.style.display !== 'none' && t.textContent.trim() !== ''; });
    await page.mouse.click(box.left + x, box.top + y);
    await page.waitForTimeout(300);
    return { sel: await page.evaluate(() => window.tw.selection.value), tag };
  };

  for (const zoom of [7, 8.4]) {
    const m = await findMeet(zoom, 'city');
    const r = await clickAt(m.station ?? m.line);
    report(`zoom ${zoom}: a click where the network meets a bubble selects the bubble`, r.sel?.kind === 'city', JSON.stringify(r.sel));
    report(`zoom ${zoom}: the bubble tag shows there`, r.tag);
  }
  const m = await findMeet(15, null);
  const r = await clickAt(m.station ?? m.line);
  report('zoom 15: the same click selects the network', r.sel?.kind === 'station' || r.sel?.kind === 'line', JSON.stringify(r.sel));
  report('zoom 15: no bubble tag there', !r.tag);
} catch (e) {
  report('ran to the end', false, e.message);
} finally {
  report('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await close();
}
console.log(results.every(Boolean) ? 'T-1550 PASSED' : 'T-1550 FAILED');
process.exit(results.length && results.every(Boolean) ? 0 : 1);
