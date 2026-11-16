// // T-3370: in Edit route the network hover is hidden over the route path, and a finger drags from the path and from a point.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3370-bus-touch.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6199/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const ROUTE = [[-73.9930, 40.7300], [-73.9850, 40.7390], [-73.9770, 40.7300]];

const chrome = await launchChrome({ prefix: 'tw-t3370-', size: [1400, 900], deviceScaleFactor: 1 });
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


  const spot = async (route, avoid) => {
    const av = [];
    for (const p of avoid) av.push(await at(p));
    let best = null;
    for (let i = 0; i + 1 < route.length / 2; i++) for (const f of [0.25, 0.5, 0.75]) {
      const p = [route[2 * i] + f * (route[2 * i + 2] - route[2 * i]), route[2 * i + 1] + f * (route[2 * i + 3] - route[2 * i + 1])];
      const q = await at(p);
      const d = Math.min(...av.map((a) => Math.hypot(a.x - q.x, a.y - q.y)));
      if (!best || d > best.d) best = { d, p, q };
    }
    return best;
  };
  const s1 = await spot(before.route, [...ROUTE, ...before.stops]);
  const a = s1.q;
  // open Edit route with the mouse (drag from the path)
  await page.mouse.move(a.x, a.y);
  await page.waitForTimeout(300);
  await page.mouse.down();
  await page.mouse.move(a.x, a.y - 12, { steps: 4 });
  await page.mouse.move(a.x + 10, a.y - 95, { steps: 10 });
  await page.waitForTimeout(1200);
  await page.mouse.up();
  await page.waitForTimeout(1000);
  check((await tool()) === 'bus' && (await pts()).length === 4, 'Edit route open with four points');
  // hover over the path: the network's own hover is hidden
  const live = await page.evaluate(() => window.tw.tools.bus.legs.map((l) => Array.from(l.pts)));
  const s2 = await spot(live[live.length - 1], await pts());
  await page.mouse.move(s2.q.x + 40, s2.q.y + 40);
  await page.mouse.move(s2.q.x, s2.q.y, { steps: 4 });
  await page.waitForTimeout(700);
  check(await page.evaluate(() => window.tw.tools.bus.onHandle), 'over the path the bus tool owns the hover');
  const tips = await page.evaluate(() => [...document.querySelectorAll('.tool-tag, .map-tag, [class*="tag"]')].filter((e) => e.offsetParent && e.textContent.trim()).map((e) => e.className + ': ' + e.textContent.trim().slice(0, 60)));
  console.log('visible tags over the path', JSON.stringify(tips));
  check(!tips.some((x) => /Bus line|Not running|Bus 1/.test(x)), 'no network hover over the path', JSON.stringify(tips));
  await page.screenshot({ path: join(ARTIFACTS, 't3370-hover.png') });
  // touch: switch to a touch pointer and drag from the path
  const cdp = await chrome.ctx.newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
  const s3 = await spot((await page.evaluate(() => window.tw.tools.bus.legs.map((l) => Array.from(l.pts)))).flat().length ? (await page.evaluate(() => window.tw.tools.bus.legs.map((l) => Array.from(l.pts))))[0] : [], await pts());
  const tp = (type, x, y) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y, id: 1 }] });
  const n0 = (await pts()).length;
  await tp('touchStart', s3.q.x, s3.q.y);
  for (let k = 1; k <= 12; k++) { await tp('touchMove', s3.q.x + k * 4, s3.q.y + k * 6); await page.waitForTimeout(40); }
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(ARTIFACTS, 't3370-touch-during.png') });
  await tp('touchEnd');
  await page.waitForTimeout(1000);
  const n1 = (await pts()).length;
  check(n1 === n0 + 1, 'a finger dragging from the path adds a point', n0 + ' -> ' + n1);
  const mapMoved = await page.evaluate(() => window.tw.map.isMoving());
  check(!mapMoved, 'the map did not pan');
  // a finger on an existing point drags it
  const p = await pts();
  const i = p.findIndex((_, k) => k > 0 && k < p.length - 1);
  const q0 = await at(p[i]);
  await tp('touchStart', q0.x, q0.y);
  for (let k = 1; k <= 10; k++) { await tp('touchMove', q0.x + k * 5, q0.y - k * 3); await page.waitForTimeout(40); }
  await page.waitForTimeout(600);
  await tp('touchEnd');
  await page.waitForTimeout(1000);
  const p2 = await pts();
  check(p2.length === p.length && Math.abs(p2[i][0] - p[i][0]) > 0.0002, 'a finger on a point drags it', JSON.stringify([p[i], p2[i]]));
  await page.screenshot({ path: join(ARTIFACTS, 't3370-touch-after.png') });
} finally {
  console.log('page errors', JSON.stringify(errors));
  await chrome.close?.();
}
process.exit(fail ? 1 : 0);
