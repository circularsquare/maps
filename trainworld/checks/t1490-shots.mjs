// T-1490: a selected bus line's 400 m range. A new game in New York with a bent bus line; screenshots
// light and dark of it selected and of the same line unselected, at two zooms, into
// artifacts/trainworld/t1490/ named by the label. From the maps root:
//   node trainworld/checks/t1490-shots.mjs <url> <before|after>
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1490');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6190/';
const label = process.argv[3] ?? 'after';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t1490-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const [lng, lat] = [-73.87, 40.75];
  await page.evaluate(([a, b]) => window.tw.map.jumpTo({ center: [a, b], zoom: 12 }), [lng, lat]);
  await page.waitForTimeout(2500);
  // a bent route: five points, so the corridor is a union of overlapping capsules
  const pts = [[lng - 0.03, lat + 0.02], [lng - 0.01, lat + 0.012], [lng + 0.005, lat + 0.03], [lng + 0.02, lat + 0.015], [lng + 0.035, lat + 0.022]];
  const id = await page.evaluate(async (pts) => {
    const tw = window.tw, c = tw.client;
    const route = pts.flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; });
    const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
    await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    return add.line;
  }, pts);
  await page.waitForTimeout(1500);
  const map = await page.$('#map');
  const box = await map.boundingBox();
  const shade = async (s) => { await page.evaluate((s) => { window.tw.shade.value = s; }, s); await page.waitForTimeout(700); };
  const select = (on) => page.evaluate(([on, id]) => { window.tw.selection.value = on ? { kind: 'line', line: String(id) } : null; }, [on, id]);
  const at = (p) => page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p);
  for (const [zoom, w, h] of [[12.6, 760, 420], [14.5, 520, 340]]) {
    await page.evaluate(([a, b, z]) => window.tw.map.jumpTo({ center: [a, b], zoom: z }), [lng + 0.0025, lat + 0.021, zoom]);
    await page.waitForTimeout(2000);
    const [x, y] = await at([lng + 0.0025, lat + 0.021]);
    for (const s of ['light', 'dark']) {
      await shade(s);
      for (const on of [true, false]) {
        await select(on);
        await page.mouse.move(box.x + 5, box.y + 5);
        await page.waitForTimeout(800);
        const clip = { x: Math.max(0, box.x + x - w / 2), y: Math.max(0, box.y + y - h / 2), width: w, height: h };
        await page.screenshot({ path: join(OUT, `${label}-z${zoom}-${on ? 'selected' : 'unselected'}-${s}.png`), clip });
      }
    }
  }
  await select(false);
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
