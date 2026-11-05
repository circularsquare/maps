// T-2462: the Speed display's own colour scale for bus and ferry routes. Loads real-ferries.save, adds three
// bus lines as t2452-bus-views.mjs does, waits for demand, prints each bus and ferry line's hovered speed, and
// takes the Speed display light and dark into artifacts/trainworld/smallui/. From the maps root:
//   node trainworld/checks/t2462-bus-speed.mjs [dev url]      (claim 3 cores; 2 demand workers)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 'smallui');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6192/';
const BUSES = [
  [[-73.9905, 40.7145], [-73.9840, 40.7175], [-73.9765, 40.7200]],
  [[-73.9600, 40.7140], [-73.9520, 40.7070], [-73.9450, 40.7000]],
  [[-73.9560, 40.7440], [-73.9450, 40.7470], [-73.9350, 40.7500]],
];
const VIEW = { center: [-73.968, 40.728], zoom: 13.2 };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = await launchChrome({ prefix: 'tw-t2462-', size: [1300, 850], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(join(ARTIFACTS, 't2195', 'real-ferries.save')));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.evaluate(async (BUSES) => {
    const tw = window.tw, c = tw.client;
    for (const [i, pts] of BUSES.entries()) {
      const route = pts.flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
      const add = await c.edit({ op: 'addBusLine', route, name: '', colour: ['#d7263d', '#7b3fa0', '#1f8a70'][i] });
      if (add.ok) await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    }
  }, BUSES);
  await wait(1500);
  await demandFinal(page, { timeout: 1200000, polling: 1000, lines: 100 });
  await page.evaluate(() => { const tw = window.tw; tw.display.value = { ...tw.display.value, trains: false, dayShown: 'weekday', trackColour: 'speed' }; });
  const speeds = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, out = [];
    const { trackTip } = await import('/src/map/trackHover.ts');
    for (const l of w.save.lines) {
      const b = w.lineStats[l.id]?.bus;
      if (!b || !b.built) continue;
      const n = b.routeLL.length / 2 - 1, found = [];
      for (let i = 0; i < n; i++) {
        const at = [(b.routeLL[2 * i] + b.routeLL[2 * i + 2]) / 2, (b.routeLL[2 * i + 1] + b.routeLL[2 * i + 3]) / 2];
        tw.map.jumpTo({ center: at, zoom: 13.5 });
        await new Promise((r) => setTimeout(r, 400));
        const p = tw.map.project(at);
        const t = (trackTip(tw.overlay, p.x, p.y, 13.5) ?? []).map((x) => (typeof x === 'string' ? x : x.textContent)).join(' | ');
        const m = /Fastest (?:bus|boat) (\d+)/.exec(t);
        if (m) found.push(+m[1]);
        if (found.length >= 8) break;
      }
      out.push(`${l.name} ${b.ferry ? 'ferry' : 'bus'}: ${found.join(' ')}`);
    }
    return out;
  });
  console.log(speeds.join('\n'));
  for (const shade of ['light', 'dark']) {
    await page.evaluate(([s, VIEW]) => { window.tw.shade.value = s; window.tw.map.jumpTo(VIEW); }, [shade, VIEW]);
    await wait(2500);
    await page.screenshot({ path: join(OUT, `speed-${shade}.png`) });
  }
} finally {
  console.log(errors.length ? `errors: ${errors.slice(0, 3).join(' | ')}` : 'no console errors');
  await chrome.close();
}
