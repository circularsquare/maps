// T-3481: zoomed out, the connectors between a complex's dots keep their width on the globe when the
// game's first station is far from the view. The width was measured on screen at that station, which
// behind the globe gave NaN and near its edge a width 10-100 times too great: long straight bars
// across the map (players with stations in two cities and the globe on). On the real New York save
// (every station in New York), the connectors' half width (u_bridge, local units) with the globe at
// Tokyo, Bangkok and New York must be the flat map's at the same place and zoom. Run from the maps root
// with a Vite of this tree:
//   node trainworld/checks/t3481-globe-connectors.mjs http://localhost:6171/
import { readFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, REAL_SAVE } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6171/';
const chrome = await launchChrome({ prefix: 'tw-t3481-', size: [1400, 900] });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
let failed = 0;
try {
  await page.goto(url + '?debug=1&v=' + Date.now());
  await page.waitForSelector('.start-menu .new-game', { timeout: 120000 });
  await page.click('.start-menu .new-game-cities [data-city="nyc"]');
  await page.click('.start-menu .new-game .start-new');
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0 && window.tw.renderer.net, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  const read = (projection, at, zoom) => page.evaluate(async ([projection, at, zoom]) => {
    const tw = window.tw, d = tw.display;
    if (d.value.projection !== projection) d.value = { ...d.value, projection };
    tw.map.jumpTo({ center: at, zoom });
    for (let i = 0; i < 40; i++) {
      await new Promise((r) => setTimeout(r, 100));
      if ((tw.map.getProjection()?.type ?? 'mercator') === (projection === 'globe' ? 'globe' : 'mercator')) break;
    }
    tw.map.triggerRepaint();
    await new Promise((r) => setTimeout(r, 1500));
    const r = tw.renderer;
    return { conns: r.connPairs.length / 2, t: r.cam?.t ?? 0, bridge: r.gl.getUniform(r.bridgeProg, r.bridgeU.u_bridge) };
  }, [projection, at, zoom]);
  const places = [['New York', [-73.98, 40.75]], ['Tokyo', [139.76, 35.68]], ['Bangkok', [100.52, 13.75]]];
  for (const [name, at] of places) for (const zoom of [10.5, 11.5, 11.85]) {
    const flat = await read('flat', at, zoom), globe = await read('globe', at, zoom);
    const ok = flat.conns > 0 && Number.isFinite(globe.bridge) && Math.abs(globe.bridge / flat.bridge - 1) < 0.05;
    if (!ok) failed++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} z${zoom}: globe (blend ${globe.t.toFixed(2)}) ${globe.bridge?.toFixed?.(2) ?? globe.bridge}, flat ${flat.bridge.toFixed(2)} local units, ${flat.conns} connectors`);
  }
} catch (e) {
  failed++;
  console.log('FAIL', e?.stack ?? e);
}
if (errors.length) console.log('page errors: ' + errors.slice(0, 5).join(' | '));
await chrome.close();
console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
