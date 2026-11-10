// T-3084: editing a bus route. A bus line is drawn with the mouse, then (1) a drag on one of its points
// outside Edit route opens Edit route with the drag applied to the draft; (2) while editing, the line's
// current route is drawn muted under the draft; (3) Done applies it as one edit and Ctrl+Z restores the old
// route; Cancel (Escape) throws a drag away. Pictures light and dark in artifacts/trainworld/t3084-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3084-bus-edit.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6174/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

// three points along Manhattan streets: the middle one is dragged
const ROUTE = [[-73.9930, 40.7300], [-73.9850, 40.7390], [-73.9770, 40.7300]];
const DROP = [-73.9850, 40.7500];

const chrome = await launchChrome({ prefix: 'tw-t3084-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);
const box = () => page.$('#map').then((m) => m.boundingBox());
const at = async (p) => { const b = await box(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const bus = () => page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines[0]; const st = l && w.lineStats[l.id]; return l ? { num: l.num, id: l.id, route: st.bus.routeLL, built: st.bus.built } : null; });
const tool = () => page.evaluate(() => window.tw.tools.bus.active ? 'bus' : 'other');
const pts = () => page.evaluate(() => window.tw.tools.bus.pts.map((p) => [p.lng, p.lat]));
const same = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-7);

const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);

async function run(shade) {
  await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, shade);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 180000 });
  await page.evaluate(([pts]) => window.tw.map.fitBounds([[-73.998, 40.725], [-73.972, 40.755]], { padding: 100, duration: 0 }), [ROUTE]);
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
  // Enter finishes: makes the line
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 1 && !window.tw.tools.bus.active, undefined, { timeout: 60000 });
  await page.keyboard.press('Enter'); // keeps the name
  await page.waitForTimeout(500);
  const before = await bus();
  check(!!before && before.route.length >= 6, 'a planned bus line is made', `${before?.route.length}`);
  // the line is selected; leave the panel's name box, then drag the middle point on the map
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
  await page.waitForTimeout(500);
  await settle(300);
  const dragTo = async (from, to, hold = 1500) => {
    const a = await at(from), b = await at(to);
    await page.mouse.move(a.x, a.y);
    await page.waitForTimeout(300);
    await page.mouse.down();
    await page.mouse.move(a.x, a.y - 12, { steps: 4 });
    await page.mouse.move(b.x, b.y, { steps: 10 });
    await page.waitForTimeout(hold);
  };
  check((await tool()) !== 'bus', 'not in the bus tool before the drag', await tool());
  // 1. a drag on a point of the selected line opens Edit route with the drag on the draft
  await dragTo(ROUTE[1], DROP);
  check((await tool()) === 'bus', 'the drag opens Edit route', await tool());
  const live = await bus();
  check(same(live.route, before.route), 'the route in the game is not changed while dragging');
  const dp = await pts();
  check(dp.length === 3 && Math.abs(dp[1][1] - DROP[1]) < 0.002, 'the draft middle point follows the pointer', JSON.stringify(dp[1]));
  await page.screenshot({ path: join(ARTIFACTS, `t3084-drag-${shade}.png`) });
  await page.mouse.up();
  await page.waitForTimeout(1200);
  check((await tool()) === 'bus' && (await pts()).length === 3, 'Edit route stays open after the drop');
  await page.screenshot({ path: join(ARTIFACTS, `t3084-${shade}.png`) });
  // Ctrl+Z inside Edit route takes the drag back, Ctrl+Y puts it again
  const moved = await pts();
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(500);
  const undone = await pts();
  check(Math.abs(undone[1][1] - ROUTE[1][1]) < 0.002, 'Ctrl+Z in Edit route puts the point back', JSON.stringify(undone[1]));
  check(same(live.route, (await bus()).route) && (await page.evaluate(() => window.tw.world.value.save.lines.length)) === 1, 'no undo ran in the game');
  await page.keyboard.press('Control+y');
  await page.waitForTimeout(500);
  check(same(moved.flat(), (await pts()).flat()), 'Ctrl+Y brings the drag back');
  // Done: Enter saves; the route is the new one, one undo step
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.tw.tools.bus.active, undefined, { timeout: 30000 });
  await page.waitForTimeout(800);
  const after = await bus();
  check(!same(after.route, before.route), 'Done applies the new route');
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(800);
  check(same((await bus()).route, before.route), 'Ctrl+Z after Done restores the old route in one step');
  await page.keyboard.press('Control+y');
  await page.waitForTimeout(800);
  check(same((await bus()).route, after.route), 'redo brings the dragged route back');
  await page.keyboard.press('Control+z');
  await page.waitForTimeout(800);
  // Cancel: a drag then Escape leaves the route as it was
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
  await page.waitForTimeout(500);
  await dragTo(ROUTE[1], DROP, 800);
  await page.mouse.up();
  await page.waitForTimeout(800);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(600);
  check((await tool()) !== 'bus' && same((await bus()).route, before.route), 'Escape throws the drag away');
  // a stop that is not a clicked point can be dragged too: it becomes a point of the draft
  const stop = await page.evaluate(() => { const w = window.tw.world.value, b = w.lineStats[w.save.lines[0].id].bus; const q = window.tw.tools.bus; return b.stops.map((s) => [s.lng, s.lat]); });
  const corner = (p) => ROUTE.some((c) => Math.abs(c[0] - p[0]) < 2e-4 && Math.abs(c[1] - p[1]) < 2e-4);
  const plain = stop.filter((p) => !corner(p));
  console.log('stops', stop.length, 'not corners', plain.length);
  if (plain.length) {
    await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
    await page.waitForTimeout(400);
    await dragTo(plain[0], [plain[0][0] + 0.0008, plain[0][1] + 0.0016], 800);
    await page.mouse.up();
    await page.waitForTimeout(800);
    check((await tool()) === 'bus' && (await pts()).length === 4, 'a stop dragged becomes a point of the draft', `${(await pts()).length}`);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(400);
  }
  // a press that does not move stays a click: no Edit route
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, before.id);
  const m0 = await at(ROUTE[1]);
  await page.mouse.click(m0.x, m0.y);
  await page.waitForTimeout(500);
  check((await tool()) !== 'bus', 'a click on a point opens nothing');

  // ferries: Edit route mutes the line's current route too
  const WHITEHALL = [-74.0130, 40.7012], ST_GEORGE = [-74.0715, 40.6437];
  await page.evaluate(() => window.tw.tools.exit?.());
  await page.evaluate(([a, c]) => window.tw.map.fitBounds([[Math.min(a[0], c[0]), Math.min(a[1], c[1])], [Math.max(a[0], c[0]), Math.max(a[1], c[1])]], { padding: 140, duration: 0 }), [WHITEHALL, ST_GEORGE]);
  await settle();
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-ferry');
  await page.waitForTimeout(300);
  for (const [k, pt] of [WHITEHALL, ST_GEORGE].entries()) {
    const q = await at(pt);
    await page.mouse.move(q.x - 4, q.y - 4);
    await page.mouse.move(q.x, q.y, { steps: 4 });
    await page.waitForTimeout(1500);
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.ferry.ports.length >= n, k + 1, { timeout: 30000 }).catch(() => {});
  }
  await page.waitForFunction(() => { const f = window.tw.tools.ferry; return f.route && f.routeKey === f.berthKey(); }, undefined, { timeout: 60000 }).catch(() => {});
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 2 && !window.tw.tools.ferry.active, undefined, { timeout: 90000 });
  await page.keyboard.press('Enter');
  const fid = await page.evaluate(() => window.tw.world.value.save.lines.find((l) => window.tw.world.value.lineStats[l.id].bus.ferry).id);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, fid);
  await page.waitForTimeout(600);
  await page.locator('button:has-text("Edit route")').first().click();
  await page.waitForFunction(() => window.tw.tools.ferry.active, undefined, { timeout: 20000 });
  await page.waitForTimeout(1200);
  const fnum = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id).num, fid);
  check(await page.evaluate((n) => window.tw.renderer.buses.muted === n, fnum), 'Edit route on a ferry line mutes its current route');
  await page.screenshot({ path: join(ARTIFACTS, `t3084-ferry-${shade}.png`) });
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  check(await page.evaluate(() => window.tw.renderer.buses.muted === -1), 'leaving Edit route draws it in colour again');
}

try {
  await run(process.env.SHADE ?? 'light');
} finally {
  console.log('page errors', JSON.stringify(errors));
  await chrome.close?.();
}
process.exit(fail ? 1 : 0);
