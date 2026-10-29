// T-1804: a city's loading bar stays on its spot through a zoom. A bar is faked on the zone in view
// (zoneLoads), the map zooms in and out, and each frame the bar's place on the screen is compared with
// where the zone's centre projects (plus the bar's fixed drop below it). From the maps root:
//   node trainworld/checks/t1804-bars.mjs [url]   (about 30 s; screenshots to artifacts/trainworld/t1804/)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, GL, launchChrome, collectErrors } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6173/';
const OUT = join(ARTIFACTS, 't1804');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t1804-', size: [1200, 800], args: GL });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.map && window.tw?.world?.value, undefined, { timeout: 180000 });
  await page.waitForTimeout(4000);
  const city = await page.evaluate(async () => {
    const c = await import('/src/game/cities.ts'), l = await import('/src/game/loading.ts');
    const id = c.activeCity.value;
    l.zoneLoads.value = new Map([[id, 0.4]]);
    return { id, center: c.cityById(id).center };
  });
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 9 }), city.center);
  await page.waitForSelector('.zone-load', { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, 'before-zoom.png') });

  // zoom in and out; every frame, where the bar is against where it should be
  const run = () => page.evaluate(async ([center]) => {
    const map = window.tw.map, el = document.querySelector('.zone-load'), box = map.getContainer().getBoundingClientRect();
    const samples = [];
    const frame = () => {
      const r = el.getBoundingClientRect(), p = map.project(center);
      // the bar is 96 px wide, centred on its x; its top edge sits 4 px above the place (style.css .zone-load .load-bar)
      samples.push({ free: p.x > 60 && p.x < box.width - 60 && p.y + 24 > 60 && p.y + 24 < box.height - 60, z: map.getZoom(), bx: r.left - box.left, by: r.top - box.top, dx: (r.left + r.width / 2 - box.left) - p.x, dy: (r.top - box.top) - (p.y + 24 - 4) });
    };
    let on = true;
    const loop = () => { if (on) { frame(); requestAnimationFrame(loop); } };
    requestAnimationFrame(loop);
    const around = map.unproject([box.width * 0.2, box.height * 0.25]);
    map.easeTo({ zoom: 11, duration: 900, around });
    await new Promise((d) => map.once('moveend', d));
    map.easeTo({ zoom: 8, duration: 900, around });
    await new Promise((d) => map.once('moveend', d));
    on = false;
    return samples;
  }, [city.center]);
  const samples = await run();
  // the bar's fixed drop and width differ from the guess above, so frames are compared with the first, at rest
  // frames where the bar is held at the screen edge (its zone's centre is off the screen) are left out
  const free = samples.filter((s) => s.free), d0 = free[0], drift = free.map((s) => Math.hypot(s.dx - d0.dx, s.dy - d0.dy));
  const worst = Math.max(...drift), mean = drift.reduce((a, v) => a + v, 0) / drift.length;
  console.log(`${free.length} of ${samples.length} frames count; the bar drifts from its spot by ${mean.toFixed(1)} px on average, ${worst.toFixed(1)} px at worst`);
  const zs = samples.map((s) => s.z), bxs = samples.map((s) => s.bx);
  console.log(`zoom ${Math.min(...zs).toFixed(2)} to ${Math.max(...zs).toFixed(2)}; the bar's left edge went ${Math.min(...bxs).toFixed(0)} to ${Math.max(...bxs).toFixed(0)} px`);
  check(Math.max(...zs) - Math.min(...zs) > 2, 'the map zoomed');
  check(free.length > 20, 'frames sampled during the zoom');
  check(worst <= 2, `the bar holds its place (worst ${worst.toFixed(1)} px, whole-pixel rounding allowed)`);
  await page.screenshot({ path: join(OUT, 'after-zoom.png') });
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
