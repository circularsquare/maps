// T-3366: a bus route can be reshaped by dragging from anywhere on its path. A line is drawn in a new game; then
// (1) with the line selected and no tool on, a press on the path between two points that moves opens Edit route and
// a new point is put there and follows the pointer, the route searched again through it; (2) in Edit route a second
// drag from the path adds another, and a drag of the new point moves it like any other; (3) Ctrl+Z takes each back,
// Ctrl+Y returns it; (4) a click on the point deletes it; (5) a press on the path that does not move is a click and
// opens nothing; the hover tag says "Drag to add a point". Pictures in artifacts/trainworld/t3366-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3366-bus-drag.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6199/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const ROUTE = [[-73.9930, 40.7300], [-73.9850, 40.7390], [-73.9770, 40.7300]];

const chrome = await launchChrome({ prefix: 'tw-t3366-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);
const box = () => page.$('#map').then((m) => m.boundingBox());
const at = async (p) => { const b = await box(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const bus = () => page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines[0]; const st = l && w.lineStats[l.id]; return l ? { num: l.num, id: l.id, route: st.bus.routeLL, stops: st.bus.stops.map((s) => [s.lng, s.lat]) } : null; });
const tool = () => page.evaluate(() => window.tw.tools.bus.active ? 'bus' : 'other');
const pts = () => page.evaluate(() => window.tw.tools.bus.pts.map((p) => [p.lng, p.lat]));
const legs = () => page.evaluate(() => window.tw.tools.bus.legs.map((l) => ({ road: l.road, ready: l.ready, refused: l.refused ?? null, n: l.pts.length / 2 })));
const same = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-7);
const tagText = () => page.evaluate(() => { const t = document.querySelector('.tool-tag.build-tag'); return t && t.style.display !== 'none' ? t.textContent : ''; });

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
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.bus.pts.length >= n, k + 1, { timeout: 40000 }).catch(() => {});
  }
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 1 && !window.tw.tools.bus.active, undefined, { timeout: 60000 });
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const before = await bus();
  check(!!before && before.route.length >= 6, 'a planned bus line is made', `${before?.route.length}`);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
  await page.waitForTimeout(500);
  await settle(300);

  // a spot on the path far (in px) from every point and stop: the middle of the longest clear stretch
  const spot = async (route, avoid) => {
    const av = [];
    for (const p of avoid) av.push(await at(p));
    let best = null;
    for (let i = 0; i + 1 < route.length / 2; i++) {
      for (const f of [0.25, 0.5, 0.75]) {
        const p = [route[2 * i] + f * (route[2 * i + 2] - route[2 * i]), route[2 * i + 1] + f * (route[2 * i + 3] - route[2 * i + 1])];
        const q = await at(p);
        const d = Math.min(...av.map((a) => Math.hypot(a.x - q.x, a.y - q.y)));
        if (!best || d > best.d) best = { d, p, q };
      }
    }
    return best;
  };
  const s1 = await spot(before.route, [...ROUTE, ...before.stops]);
  check(s1.d > 20, 'a spot on the path is clear of every point and stop', `${Math.round(s1.d)} px`);
  await page.screenshot({ path: join(ARTIFACTS, 't3366-before.png') });

  // 1. select tool: a drag from the path opens Edit route and adds a point at the pointer
  const a = s1.q, to = { x: a.x + 10, y: a.y - 95 };
  check((await tool()) !== 'bus', 'not in the bus tool before the drag');
  await page.mouse.move(a.x, a.y);
  await page.waitForTimeout(300);
  await page.mouse.down();
  await page.mouse.move(a.x, a.y - 12, { steps: 4 });
  await page.mouse.move(to.x, to.y, { steps: 10 });
  await page.waitForTimeout(1500);
  check((await tool()) === 'bus', 'the drag opens Edit route');
  const d1 = await pts();
  check(d1.length === 4, 'a fourth point is added to the draft', `${d1.length}`);
  const want = await page.evaluate(([x, y]) => { const b = document.querySelector('#map').getBoundingClientRect(); const l = window.tw.map.unproject([x - b.x, y - b.y]); return [l.lng, l.lat]; }, [to.x, to.y]);
  check(d1.length === 4 && Math.hypot(d1[1][0] - want[0], (d1[1][1] - want[1])) < 0.0015, 'the new point is between the first and last and follows the pointer', JSON.stringify(d1[1]));
  check(same((await bus()).route, before.route), 'the game route is unchanged while dragging');
  const lg = await legs();
  check(lg.length === 3 && lg.every((l) => l.ready && l.road), 'the three legs are searched again on the roads', JSON.stringify(lg));
  await page.screenshot({ path: join(ARTIFACTS, 't3366-during.png') });
  await page.mouse.up();
  await page.waitForTimeout(1200);
  check((await tool()) === 'bus' && (await pts()).length === 4, 'Edit route stays open after the drop');
  await page.screenshot({ path: join(ARTIFACTS, 't3366-after.png') });

  // 2. in the tool: the tag over the path, then a second drag from the path adds another point
  const live = await page.evaluate(() => window.tw.tools.bus.legs.map((l) => Array.from(l.pts)));
  const s2 = await spot(live.flatMap((l) => l).length ? live[live.length - 1] : [], [...(await pts())]);
  await page.mouse.move(s2.q.x + 40, s2.q.y + 40);
  await page.mouse.move(s2.q.x, s2.q.y, { steps: 4 });
  await page.waitForTimeout(700);
  check((await tagText()).includes('Drag to add a point'), 'the tag over the path says Drag to add a point', await tagText());
  await page.screenshot({ path: join(ARTIFACTS, 't3366-hover.png') });
  await page.mouse.down();
  await page.mouse.move(s2.q.x, s2.q.y + 12, { steps: 4 });
  await page.mouse.move(s2.q.x - 60, s2.q.y + 70, { steps: 8 });
  await page.waitForTimeout(1200);
  await page.mouse.up();
  await page.waitForTimeout(1000);
  const d2 = await pts();
  check(d2.length === 5, 'a second drag from the path adds a fifth point', `${d2.length}`);

  // the first new point drags like any other
  const p1 = await at(d2[1]);
  await page.mouse.move(p1.x, p1.y);
  await page.waitForTimeout(300);
  await page.mouse.down();
  await page.mouse.move(p1.x + 6, p1.y, { steps: 3 });
  await page.mouse.move(p1.x + 70, p1.y - 10, { steps: 8 });
  await page.waitForTimeout(1000);
  await page.mouse.up();
  await page.waitForTimeout(1000);
  const d3 = await pts();
  check(d3.length === 5 && Math.abs(d3[1][0] - d2[1][0]) > 0.0004, 'the new point moves again', `${d2[1]} -> ${d3[1]}`);

  // 3. Ctrl+Z steps back through each, Ctrl+Y returns
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(500);
  check(same((await pts()).flat(), d2.flat()), 'Ctrl+Z puts the moved point back');
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(500);
  check((await pts()).length === 4, 'Ctrl+Z takes the second added point out');
  await page.keyboard.press('Control+y');
  await page.waitForTimeout(500);
  check(same((await pts()).flat(), d2.flat()), 'Ctrl+Y brings it back');

  // 4. a click on a new point deletes it
  const p2 = await at(d2[1]);
  await page.mouse.move(p2.x + 20, p2.y + 20);
  await page.mouse.move(p2.x, p2.y, { steps: 3 });
  await page.waitForTimeout(400);
  await page.mouse.click(p2.x, p2.y);
  await page.waitForFunction(() => window.tw.tools.bus.pts.length === 4, undefined, { timeout: 15000 }).catch(() => {});
  check((await pts()).length === 4, 'a click on an added point deletes it, as on any point', `${(await pts()).length}`);

  // Done saves the route; one undo in the game restores the old route
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.tw.tools.bus.active, undefined, { timeout: 30000 });
  await page.waitForTimeout(800);
  const after = await bus();
  check(!same(after.route, before.route), 'Done applies the new route');
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(800);
  check(same((await bus()).route, before.route), 'Ctrl+Z after Done restores the old route');
  await page.keyboard.press('Control+y');
  await page.waitForTimeout(800);

  // 5. a press on the path that does not move is a click: nothing opens
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
  await page.waitForTimeout(500);
  const s3 = await spot((await bus()).route, [...(await pts().catch(() => [])), ...(await bus()).stops, ...ROUTE]);
  await page.mouse.click(s3.q.x, s3.q.y);
  await page.waitForTimeout(500);
  check((await tool()) !== 'bus', 'a click on the path opens nothing');
} finally {
  console.log('page errors', JSON.stringify(errors));
  await chrome.close?.();
}
process.exit(fail ? 1 : 0);
