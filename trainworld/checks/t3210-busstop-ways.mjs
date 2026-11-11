// T-3210, T-3211: a hand-placed bus stop on a stretch the bus runs both ways serves both ways, and the hover says
// "One way stop" only on a one-way stop. A new game in New York (Nutley NJ): a bus loop whose way out and way back
// are drawn by hand along one road, 10 m apart, set to Manual. Stops placed on the road (both passes) and on the loop
// (one pass). Screenshots artifacts/trainworld/t3210-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3210-busstop-ways.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6114/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const LL = [-74.16, 40.82];

const chrome = await launchChrome({ prefix: 'tw-t3210-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
collectErrors(page, errors);

const busOf = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), b = l && w.lineStats[l.id]?.bus;
  return b ? { id: l.id, hand: b.hand, stops: b.stops.map((s) => ({ id: s.id, lng: s.lng, lat: s.lat })), route: Array.from(b.routeLL), ways: w.lineStats[l.id].oneWayAt } : null;
}, num);
const until = async (num, test) => { for (let k = 0; k < 100; k++) { const s = await busOf(num); if (s && test(s)) return s; await wait(200); } return busOf(num); };
const screen = (lng, lat) => page.evaluate(([lng, lat]) => { const p = window.tw.map.project([lng, lat]); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, [lng, lat]);
const click = async (lng, lat) => { const [x, y] = await screen(lng, lat); await page.mouse.move(x, y); await wait(250); await page.mouse.click(x, y); };
const tagText = () => page.evaluate(() => [...document.querySelectorAll('.stop-tag')].filter((t) => t.style.display !== 'none').map((t) => t.innerText).join(' / '));
const mode = (name) => page.locator('.h .group button').filter({ hasText: new RegExp(`^${name}$`) });
const unit = (ll, dx, dy) => page.evaluate(([ll, dx, dy]) => { const o = window.tw.lngLatToPack(ll[0], ll[1]); const p = window.tw.packToLngLat(o.x + dx, o.y + dy); return [p[0] ?? p.lng, p[1] ?? p.lat]; }, [ll, dx, dy]);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 300000 });
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 14 }), LL);
  await wait(1500);

  // way out along y = 0, the loop, way back along y = 10 m (not the same points)
  const P = [[0, 0], [600, 0], [1200, 0], [1600, 0], [1600, 500], [1200, 500], [1200, 10], [600, 10], [0, 0]];
  const made = await page.evaluate(async ([ll, P]) => {
    const o = window.tw.lngLatToPack(ll[0], ll[1]);
    const route = P.flatMap(([dx, dy]) => [o.x + dx, o.y + dy, 1]);
    const r = await window.tw.client.edit({ op: 'addBusLine', route, name: 'Nutley', colour: '#2e8b57' });
    return { ok: r.ok, line: r.line, issues: r.issues };
  }, [LL, P]);
  check(made.ok, 'a bus loop with its way back drawn beside its way out is made', JSON.stringify(made));
  const id = (await busOf(made.line)).id;
  await page.evaluate((id) => { window.tw.tools.exit(); window.tw.selection.value = { kind: 'line', line: id }; }, id);
  await wait(800);
  await mode('Manual').click();
  let s = await until(made.line, (x) => x.hand);
  await page.waitForFunction(() => document.documentElement.classList.contains('stop-editing'), undefined, { timeout: 10000 }).catch(() => null);
  await wait(800);
  await page.evaluate((route) => {
    const lngs = route.filter((_, i) => i % 2 === 0), lats = route.filter((_, i) => i % 2 === 1);
    window.tw.map.fitBounds([[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]], { padding: 200, duration: 0 });
  }, s.route);
  await wait(1500);
  const n0 = s.stops.length;

  // a stop on the road both passes share
  const stem = await unit(LL, 600, 5);
  await click(...stem);
  s = await until(made.line, (b) => b.stops.length >= n0 + 2);
  check(s.stops.length === n0 + 2, 'a click on the shared road adds a stop on each pass', `${n0} -> ${s.stops.length}`);
  const near = (st, p) => Math.hypot((st.lng - p[0]) * 84000, (st.lat - p[1]) * 111000) < 30;
  const stemStops = s.stops.filter((st) => near(st, stem));
  check(stemStops.length === 2 && stemStops.every((st) => s.ways[String(st.id)] === undefined), 'both of them serve both ways', JSON.stringify(stemStops.map((st) => s.ways[String(st.id)])));

  // a stop on the loop, passed once
  const loopPt = await unit(LL, 1600, 250);
  await click(...loopPt);
  s = await until(made.line, (b) => b.stops.length >= n0 + 3);
  const loopStop = s.stops.find((st) => near(st, loopPt));
  check(!!loopStop && s.ways[String(loopStop.id)] !== undefined, 'a stop on the loop alone stays one way', JSON.stringify(loopStop && s.ways[String(loopStop.id)]));

  // hovers
  await page.mouse.move(700, 880);
  await wait(600);
  await page.screenshot({ path: join(ARTIFACTS, 't3210-stops.png') });
  const hover = async (st) => { const [x, y] = await screen(st.lng, st.lat); await page.mouse.move(x, y); await wait(600); return tagText(); };
  const hStem = await hover(stemStops[0]);
  await page.screenshot({ path: join(ARTIFACTS, 't3210-hover-both.png') });
  const hLoop = await hover(loopStop);
  await page.screenshot({ path: join(ARTIFACTS, 't3210-hover-oneway.png') });
  console.log('stem hover:', JSON.stringify(hStem), ' loop hover:', JSON.stringify(hLoop));
  check(/remove/i.test(hStem) && !/one way/i.test(hStem), 'the shared road stop hover has no one way line', hStem);
  check(/one way stop/i.test(hLoop) && /remove/i.test(hLoop) && hLoop.toLowerCase().indexOf('one way') < hLoop.toLowerCase().indexOf('remove'), 'the loop stop hover says One way stop above the removal line', hLoop);
  await page.evaluate(() => window.tw.tools.exit());
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3210-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3210 FAILED (${fail})` : 'T-3210 PASSED');
process.exit(fail ? 1 : 0);
