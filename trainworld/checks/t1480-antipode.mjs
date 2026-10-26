// T-1480: a hidden realm's circle on the globe's far side stays hidden at every zoom. A new game in
// Wuhan on the globe, zoomed from 6 to 13 (and an eased zoom 9 to 13): no circle of a realm other than
// China's own shows (Brazil, near Wuhan's antipode, showed mid-screen from zoom 11.8 to 12 while the
// globe blended into the flat map). Zoomed out over east Asia, Japan's and Korea's circles still show.
//   node trainworld/checks/t1480-antipode.mjs [url]     (from the maps root; default localhost:6150)
// About a minute. Screenshot to ../artifacts/trainworld/t1479/t1480-check.png.
import { join } from 'node:path';
import { launchChrome, collectErrors, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1480-', size: [1300, 850] });
try {
  const page = await ctx.newPage();
  const errors = [];
  collectErrors(page, errors);
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', projection: 'globe' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?new=1&city=wuh&debug=1`);
  await page.waitForFunction(() => window.tw?.world?.value && !document.querySelector('.start-menu'), undefined, { timeout: 180000 });
  await page.waitForFunction(() => !document.querySelector('.loading-card'), undefined, { timeout: 300000 });
  await page.waitForTimeout(2000);
  /** the realm circles drawn inside the map, by name */
  const onScreen = () => page.evaluate(() => {
    const el = document.querySelector('#map'), W = el.clientWidth, H = el.clientHeight;
    return [...document.querySelectorAll('.realm-button')].filter((e) => e.style.display !== 'none' && e.offsetParent).flatMap((e) => {
      const m = /translate\((-?\d+)px, (-?\d+)px\)/.exec(e.style.transform);
      return m && +m[1] > 0 && +m[1] < W && +m[2] > 0 && +m[2] < H ? [e.querySelector('button').getAttribute('aria-label').replace(/^Show /, '')] : [];
    });
  });
  const far = [];
  for (const zoom of [6, 8, 10, 11, 11.3, 11.6, 11.8, 11.9, 12, 12.5, 13]) {
    for (const c of [[114.3, 30.6], [114.0, 30.4], [114.6, 30.8]]) {
      await page.evaluate(([c, zoom]) => window.tw.map.jumpTo({ center: c, zoom }), [c, zoom]);
      await page.waitForTimeout(200);
      for (const n of await onScreen()) if (!/China/.test(n)) far.push(`${n} at z${zoom}`);
    }
  }
  await page.evaluate(() => window.tw.map.jumpTo({ center: [114.3, 30.6], zoom: 9 }));
  await page.waitForTimeout(400);
  await page.evaluate(() => window.tw.map.easeTo({ zoom: 13, duration: 3000 }));
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(100);
    for (const n of await onScreen()) if (!/China/.test(n)) far.push(`${n} easing`);
  }
  report('no far realm\'s circle over Wuhan at any zoom', far.length === 0, far.slice(0, 4).join(', '));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [125, 35], zoom: 4 }));
  await page.waitForTimeout(1500);
  const near = await onScreen();
  await page.screenshot({ path: join(ARTIFACTS, 't1479', 't1480-check.png') });
  report('zoomed out over east Asia, the near realms keep their circles', near.includes('Japan') && !near.includes('Brazil'), JSON.stringify(near));
  report('no console errors', errors.length === 0, errors.slice(0, 2).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
