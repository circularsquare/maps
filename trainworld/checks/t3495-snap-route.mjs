// T-3495: the track tool snaps to each kind of drawn rail. For each place, the Real world network at All
// with snap on: finds a drawn feature of each class (used rail, unused rail "n", metro, light rail), clicks
// a point on it, rests the pointer on a point further along the same feature and reports whether a route is
// found (spot.phase done, a real line of length > 0). GRAPHS=<dir> serves that folder's .rgr files in place of
// the data pack's (to test a rebuild before it is shipped). From the maps root:
//   [GRAPHS=dir] [ONLY=london,chicago] node trainworld/checks/t3495-snap-route.mjs [url]
import { readFile, readdir } from 'node:fs/promises';
import { launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6189/';
const PLACES = { london: [-0.12, 51.5], chicago: [-87.65, 41.88], tokyo: [139.75, 35.68], berlin: [13.4, 52.5], sydney: [151.2, -33.87] };
const only = process.env.ONLY?.split(',');
const chrome = await launchChrome({ prefix: 'tw-t3495r-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  if (process.env.GRAPHS) {
    for (const f of await readdir(process.env.GRAPHS)) if (f.endsWith('.rgr')) {
      const body = await readFile(process.env.GRAPHS + '/' + f);
      await page.route('**/realgraph/' + f, (r) => r.fulfill({ status: 200, body, contentType: 'application/octet-stream' }));
    }
  }
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  for (const [name, c] of Object.entries(PLACES)) {
    if (only && !only.includes(name)) continue;
    await page.evaluate(async ([c]) => {
      const ui = await import('/src/game/ui.ts');
      ui.display.value = { ...ui.display.value, realNetwork: 'all' };
      ui.keepPlacing.value = true; ui.snapReal.value = true; ui.tool.value = 'track';
      window.tw.map.jumpTo({ center: c, zoom: 13 });
    }, [c]);
    await page.waitForTimeout(10000);
    // candidate (start, end) screen points per class: a feature's points with 60 to 300 px between them
    const cands = await page.evaluate(() => {
      const map = window.tw.map, r = map.getContainer().getBoundingClientRect();
      const layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real-')).map((l) => l.id);
      const out = {};
      for (const f of map.queryRenderedFeatures(undefined, { layers })) {
        const p = f.properties, cls = (p.k === 'rail' && p.n && !p.x ? 'rail-unused' : p.k) + (p.r === 1 ? '' : '');
        if (f.geometry.type !== 'LineString') continue;
        const pts = f.geometry.coordinates.map((q) => map.project(q)).filter((q) => q.x > 40 && q.y > 100 && q.x < r.width - 40 && q.y < r.height - 40);
        if (pts.length < 3) continue;
        const a = pts[0], b = pts[pts.length - 1], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 60 || d > 300 || !window.tw.tools.real.ring(a.x, a.y) || !window.tw.tools.real.ring(b.x, b.y)) continue;
        (out[cls] ??= []).push([[r.left + a.x, r.top + a.y], [r.left + b.x, r.top + b.y]]);
      }
      return out;
    });
    for (const cls of ['rail', 'rail-unused', 'subway', 'light_rail']) {
      const list = cands[cls] ?? [];
      let ok = 0, tried = 0, last = null;
      for (const [a, b] of list.slice(0, 6)) {
        tried++;
        await page.mouse.move(...a); await page.waitForTimeout(250);
        await page.mouse.down(); await page.mouse.up(); await page.waitForTimeout(400);
        const began = await page.evaluate(() => !!window.tw.tools.real.from);
        for (let i = 1; i <= 8; i++) { await page.mouse.move(a[0] + ((b[0] - a[0]) * i) / 8, a[1] + ((b[1] - a[1]) * i) / 8); await page.waitForTimeout(30); }
        await page.waitForTimeout(1500);
        const s = await page.evaluate(() => { const r = window.tw.tools.real; return { phase: r.spot?.phase, len: r.last?.lengthM, ring: !!r.ringAt }; });
        if (began && s.phase === 'done' && s.len > 0) ok++;
        last = { began, ...s };
        await page.keyboard.press('Escape'); await page.waitForTimeout(300);
        await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.snapReal.value = true; ui.tool.value = 'track'; });
        await page.waitForTimeout(300);
      }
      console.log(name.padEnd(9), cls.padEnd(12), tried ? ok + '/' + tried + ' routed' : 'no candidate in view', last ? JSON.stringify(last) : '');
    }
  }
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
