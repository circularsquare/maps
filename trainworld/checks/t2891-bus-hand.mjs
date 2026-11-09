// T-2891: a bus line's stops placed by hand. On the real save: By hand clears the automatic stops down
// to the two ends and opens the stop editing, a click along the route adds a stop, a click on a stop
// takes it out, undo and redo carry it, a save round-trips it, and leaving goes back to the automatic
// stops. The real save has no line by hand (`save.rs` tests check the bytes).
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t2891-bus-hand.mjs [dev url]
// About 3 minutes. Pictures in artifacts/trainworld/t2891/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2891');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6260/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t2891-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);

const state = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), b = l && w.lineStats[l.id].bus;
  return b ? { id: l.id, hand: b.hand, spacing: b.spacing, street: b.street.length / 2, reach: b.reach.length, kinds: b.stops.map((s) => s.kind), stops: b.stops.map((s) => [s.lng, s.lat]) } : null;
}, num);
const until = async (num, test) => { for (let k = 0; k < 150; k++) { const s = await state(num); if (s && test(s)) return s; await wait(200); } return state(num); };
const point = (lng, lat) => page.evaluate(([lng, lat]) => { const p = window.tw.map.project([lng, lat]); const r = window.tw.map.getContainer().getBoundingClientRect(); return [p.x + r.left, p.y + r.top]; }, [lng, lat]);
const click = async (lng, lat) => { const [x, y] = await point(lng, lat); await page.mouse.move(x, y); await wait(150); await page.mouse.click(x, y); };
const stepper = (dir) => page.locator('.stepper').filter({ has: page.locator(`button[aria-label*="Stop spacing"]`) }).locator('button')[dir < 0 ? 'first' : 'last']();
const exported = async () => {
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  return readFile(await (await dl).path());
};

try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(3000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));

  const none = await page.evaluate(() => { const w = window.tw.world.value; return w.save.lines.filter((l) => w.lineStats[l.id].bus?.hand).length; });
  check(none === 0, 'the real save has no line by hand', `${none}`);

  // the bus line: a real one, or one drawn along two stations of a rail line
  let num = await page.evaluate(() => { const w = window.tw.world.value; const l = w.save.lines.find((x) => { const b = w.lineStats[x.id].bus; return b && !b.ferry && b.stops.length >= 3 && b.roadM < 40000; }); return l ? l.num : null; });
  if (num === null) {
    const made = await page.evaluate(async () => {
      const w = window.tw.world.value;
      const rail = w.save.lines.find((l) => l.stops.length >= 6 && !w.lineStats[l.id].bus);
      const pts = rail.stops.slice(0, 5).map((s) => w.nodes.get(s) ?? w.nodes.get(Number(s))).filter(Boolean).flatMap((n) => [n.x, n.y, 1]);
      // and on 3 km past the last, off every station, for a plain stop
      pts.push(pts[pts.length - 3] + 3000, pts[pts.length - 2], 1);
      const a = await window.tw.client.edit({ op: 'addBusLine', route: pts, name: 'Check bus', colour: '#c0392b' });
      return { ok: a.ok, line: a.line, issues: a.issues, pts: pts.slice(0, 9) };
    });
    check(made.ok, 'a bus line to work on, drawn over a rail line', JSON.stringify(made));
    num = made.line;
    await wait(1500);
  }
  const before = await until(num, (s) => s.stops.length >= 2);
  const auto = await state(num);
  check(auto.kinds.length >= 3 && !auto.hand, 'an automatic line to start from', `${auto.kinds.length} stops, ${auto.street} street stops`);
  const stops = auto.stops;
  // frame the route
  await page.evaluate(([id, stops]) => {
    const lngs = stops.map((s) => s[0]), lats = stops.map((s) => s[1]);
    window.tw.tools.exit();
    window.tw.selection.value = { kind: 'line', line: id };
    window.tw.map.fitBounds([[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]], { padding: 200, duration: 0 });
  }, [auto.id, stops]);
  await wait(2500);

  // By hand: one step up from Auto
  const toAuto = auto.spacing === 0;
  check(toAuto, 'the line starts on Auto');
  await stepper(1).click();
  let s = await until(num, (x) => x.hand);
  check(s.hand && s.kinds.length === 2 && s.street === 0, 'By hand leaves the two ends', `${s.kinds} ${s.street}`);
  await wait(800);
  const editing = await page.evaluate(() => [window.tw.tool?.value ?? null, document.documentElement.classList.contains('stop-editing')]);
  check(editing[1], 'and opens the stop editing', JSON.stringify(editing));
  await page.waitForTimeout(500);
  const panel = await page.locator('.stepper .v, .stepper').filter({ hasText: 'Manual' }).count();
  check(panel > 0, 'the stepper reads Manual');

  // a click along the route adds, twice
  const route = await page.evaluate((id) => { const b = window.tw.world.value.lineStats[id].bus; return b.routeLL; }, auto.id);
  const n = route.length;
  const along = (f) => [route[n - 4] + f * (route[n - 2] - route[n - 4]), route[n - 3] + f * (route[n - 1] - route[n - 3])];
  const reach = await page.evaluate((id) => window.tw.world.value.lineStats[id].bus.reach, auto.id);
  const offBy = (f) => { const [lng, lat] = along(f); return Math.min(...reach.map((r) => Math.hypot((r.lng - lng) * 84000, (r.lat - lat) * 111000))); };
  const open = [...Array(19).keys()].map((k) => (k + 1) / 20).reduce((a, b) => (offBy(b) > offBy(a) ? b : a));
  console.log(`farthest from a station along the spur: ${offBy(open).toFixed(0)} m at ${open}`);
  await click(...along(open));
  s = await until(num, (x) => x.kinds.length === 3);
  check(s.kinds.length === 3 && s.kinds[1] === 2, 'a click on the open road adds a plain stop', `${s.kinds}`);
  await click(route[4], route[5]);
  s = await until(num, (x) => x.kinds.length === 4);
  check(s.kinds.length === 4 && s.kinds[1] === 1, 'a click near a station stops at the station', `${s.kinds}`);
  await wait(800);
  const [gx, gy] = await point(...along(0.8));
  await page.mouse.move(gx, gy);
  await wait(400);
  await page.screenshot({ path: join(OUT, 'placed.png') });
  const placedKinds = s.kinds;

  // a click on a stop removes it, and undo brings it back
  const victim = s.stops[1];
  await click(...victim);
  s = await until(num, (x) => x.kinds.length === 3);
  check(s.kinds.length === 3, 'a click on a stop takes it out', `${s.kinds}`);
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  s = await until(num, (x) => x.kinds.length === 4);
  check(s.hand && s.kinds.length === 4, 'undo brings it back', `${s.kinds}`);
  await page.evaluate(() => window.tw.client.edit({ op: 'redo' }));
  s = await until(num, (x) => x.kinds.length === 3);
  check(s.kinds.length === 3, 'redo takes it out again');
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  s = await until(num, (x) => x.kinds.length === 4);

  // the save round-trips
  const saved = await exported();
  await importSave(page, saved);
  await wait(4000);
  await page.waitForFunction(() => window.tw?.world?.value?.save.lines.length > 50, undefined, { timeout: 120000 });
  await wait(2000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const back = await until(num, (x) => x.hand && x.kinds.length === 4);
  check(back.hand && JSON.stringify(back.kinds) === JSON.stringify(placedKinds), 'a save keeps the stops', `${back.kinds}`);

  // leaving: one step down is Auto
  await page.evaluate((id) => { window.tw.tools.exit(); window.tw.selection.value = { kind: 'line', line: id }; }, back.id);
  await wait(800);
  await stepper(-1).click();
  s = await until(num, (x) => !x.hand);
  await wait(500);
  check(!s.hand && JSON.stringify(s.kinds) === JSON.stringify(auto.kinds) && s.street === auto.street, 'leaving goes back to the automatic stops', `${s.kinds.length} stops, ${s.street} street`);
  const toolOut = await page.evaluate(() => document.documentElement.classList.contains('stop-editing'));
  check(!toolOut, 'and the editing is over');
  check(before.kinds.length === auto.kinds.length, 'the line was the same before and after');
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(OUT, 'died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2891 FAILED: ${fail}` : 'T-2891 PASSED');
process.exit(fail ? 1 : 0);
