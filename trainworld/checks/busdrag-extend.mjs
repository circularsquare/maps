// Bus Edit route starts in plain editing (no leg follows the pointer); a click on the route's end enters the
// extend mode (the leg follows, clicks add points); Escape goes back to plain editing, still in Edit route;
// a point drags in plain editing. Pictures: artifacts/trainworld/busdrag-{plain,extend}.png.
//   node trainworld/checks/busdrag-extend.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6199/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const ROUTE = [[-73.9930, 40.7300], [-73.9850, 40.7390], [-73.9770, 40.7300]];

const chrome = await launchChrome({ prefix: 'tw-busdrag-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);
const box = () => page.$('#map').then((m) => m.boundingBox());
const at = async (p) => { const b = await box(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const active = () => page.evaluate(() => window.tw.tools.bus.active);
const pts = () => page.evaluate(() => window.tw.tools.bus.pts.map((p) => [p.lng, p.lat]));
// the leg to the pointer is drawn when the build with the pointer has more route points than the one without
const hoverLeg = () => page.evaluate(() => { const b = window.tw.tools.bus; return b.build(true).ll.length > b.build(false).ll.length; });
const tagText = () => page.evaluate(() => { const t = document.querySelector('.tool-tag.build-tag'); return t && t.style.display !== 'none' ? t.textContent : ''; });
const wander = async (x, y) => { await page.mouse.move(x, y, { steps: 4 }); await page.waitForTimeout(900); };

try {
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.map.fitBounds([[-73.998, 40.725], [-73.972, 40.755]], { padding: 100, duration: 0 }));
  await settle();
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-bus');
  await page.waitForTimeout(300);
  for (let k = 0; k < 3; k++) {
    const q = await at(ROUTE[k]);
    await page.mouse.move(q.x - 5, q.y - 5);
    await page.mouse.move(q.x, q.y, { steps: 3 });
    await page.waitForTimeout(500);
    if (k === 1) check(await hoverLeg(), 'a new line starts in extend mode: the leg follows the pointer');
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.bus.pts.length >= n, k + 1, { timeout: 40000 }).catch(() => {});
  }
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 1 && !window.tw.tools.bus.active, undefined, { timeout: 60000 });
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const id = await page.evaluate(() => window.tw.world.value.save.lines[0].id);
  await page.evaluate((i) => { window.tw.selection.value = { kind: 'line', line: i }; }, id);
  await page.waitForTimeout(500);
  await settle(300);

  await page.evaluate(() => [...document.querySelectorAll('button.btn.text')].find((b) => b.textContent.includes('Edit route')).click());
  await page.waitForFunction(() => window.tw.tools.bus.active, undefined, { timeout: 10000 }).catch(() => {});
  await page.waitForTimeout(800);
  check(await active(), 'Edit route opens the bus tool');
  const p0 = await pts();
  check(p0.length === 3, 'the route has its three points', `${p0.length}`);

  // plain editing: nothing follows the pointer, a click on empty ground adds nothing
  const free = { x: 1100, y: 250 };
  await wander(free.x - 30, free.y - 30);
  await wander(free.x, free.y);
  check(!(await hoverLeg()), 'plain editing: no leg follows the pointer');
  await page.screenshot({ path: join(ARTIFACTS, 'busdrag-plain.png') });
  await page.mouse.click(free.x, free.y);
  await page.waitForTimeout(600);
  check((await pts()).length === 3, 'plain editing: a click on empty ground adds no point');

  // the end: hover tag, then a click enters extend mode
  const end = await at(p0[2]);
  await wander(end.x + 25, end.y + 25);
  await wander(end.x, end.y);
  check((await tagText()).includes('Click to extend the route'), 'the tag over the end says Click to extend the route', await tagText());
  await page.mouse.click(end.x, end.y);
  await page.waitForTimeout(500);
  check(await active(), 'a click on the end keeps Edit route open');
  await wander(free.x - 30, free.y - 30);
  await wander(free.x, free.y);
  check(await hoverLeg(), 'after a click on the end: the leg follows the pointer');
  await page.screenshot({ path: join(ARTIFACTS, 'busdrag-extend.png') });

  // Escape: back to plain editing, not out of Edit route
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  await wander(free.x + 20, free.y + 20);
  check(await active(), 'Escape in extend mode stays in Edit route');
  check(!(await hoverLeg()), 'Escape in extend mode: no leg follows the pointer');

  // extend again and add a point by click
  await page.mouse.click(end.x, end.y);
  await page.waitForTimeout(400);
  const nu = await at([-73.9760, 40.7265]);
  await wander(nu.x - 10, nu.y - 10);
  await page.mouse.click(nu.x, nu.y);
  await page.waitForFunction(() => window.tw.tools.bus.pts.length >= 4, undefined, { timeout: 30000 }).catch(() => {});
  check((await pts()).length === 4, 'extend mode: a click adds a point', `${(await pts()).length}`);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(400);

  // plain editing: drag a point
  const mid = await pts();
  const m = await at(mid[1]);
  await wander(m.x + 15, m.y + 15);
  await wander(m.x, m.y);
  await page.mouse.down();
  await page.mouse.move(m.x + 6, m.y, { steps: 3 });
  await page.mouse.move(m.x + 60, m.y - 10, { steps: 8 });
  await page.waitForTimeout(1000);
  await page.mouse.up();
  await page.waitForTimeout(1000);
  const after = await pts();
  check(after.length === 4 && Math.abs(after[1][0] - mid[1][0]) > 0.0004, 'plain editing: a point drags', `${mid[1]} -> ${after[1]}`);
  check(await active(), 'still in Edit route after the drag');

  // Escape in plain editing leaves Edit route
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  check(!(await active()), 'Escape in plain editing leaves Edit route');
} finally {
  console.log('page errors', JSON.stringify(errors));
  await chrome.close?.();
}
process.exit(fail ? 1 : 0);
