// T-1517: editing a bus route with a finger, in Manhattan. A line is made with the mouse and opened with
// Edit route; then, by CDP touch events (Input.dispatchTouchEvent): a short press on a point deletes
// nothing, one finger on a point drags it without panning the map, one finger on empty ground still pans,
// a second finger during a drag puts the point back, a point held still for half a second is deleted and
// the lift adds no point, and a press that wanders before the half second is a drag, not a delete.
// Uses the Vite already on `port` (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1517-bus-touch.mjs [port]
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const OUT = join(ARTIFACTS, 't1517');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 7017);
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const A = [-73.9965, 40.7510], B = [-73.9795, 40.7680], C = [-73.9570, 40.7720];
const metres = (a, b) => { const k = Math.cos((a[1] * Math.PI) / 180); return Math.hypot((a[0] - b[0]) * 111320 * k, (a[1] - b[1]) * 110574); };

const chrome = await launchChrome({ prefix: process.env.PROFILE_PREFIX ?? 'tw-t1517-', profile: process.env.PROFILE_DIR, size: [1400, 900], deviceScaleFactor: 2, hasTouch: true });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, process.env.SHADE ?? 'light');
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const at = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return { x: Math.round(box.x + x), y: Math.round(box.y + y) }; };
  const state = () => page.evaluate(() => {
    const b = window.tw.tools.bus, c = window.tw.map.getCenter();
    return { tool: b.active, pts: b.pts.map((p) => [p.lng, p.lat]), legs: b.legs.length, touchAt: !!b.touchAt, centre: [c.lng, c.lat], lines: window.tw.world.value.save.lines.length, grab: window.tw.map.dragPan.isEnabled() };
  });
  const settle = async () => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 5 });
  const pt = (p, id = 1) => ({ x: Math.round(p.x), y: Math.round(p.y), id });
  const touch = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts });
  /** a finger down on `from`, held `holdMs`, moved to `to` in `steps`, lifted */
  const finger = async (from, { to = null, holdMs = 0, steps = 10, liftHeld = 0 } = {}) => {
    await touch('touchStart', [pt(from)]);
    if (holdMs) await page.waitForTimeout(holdMs);
    if (to) for (let i = 1; i <= steps; i++) { await touch('touchMove', [pt({ x: from.x + ((to.x - from.x) * i) / steps, y: from.y + ((to.y - from.y) * i) / steps })]); await page.waitForTimeout(16); }
    if (liftHeld) await page.waitForTimeout(liftHeld);
    await touch('touchEnd', []);
    await page.waitForTimeout(600);
  };

  // ---- a line made with the mouse, then opened for editing
  await page.evaluate(([a, c]) => window.tw.map.fitBounds([[Math.min(a[0], c[0]) - 0.004, a[1] - 0.008], [c[0] + 0.004, c[1] + 0.004]], { padding: 120, maxZoom: 16, duration: 0 }), [A, C]);
  await settle();
  await page.evaluate(() => { const b = window.tw.tools.bus; const c = window.tw.map.getCenter(); b.roads.look(c); });
  await page.waitForTimeout(3000);
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-bus');
  await page.waitForTimeout(300);
  for (const p of [A, B, C]) {
    const q = await at(p);
    await page.mouse.move(q.x, q.y, { steps: 6 });
    await page.waitForTimeout(250);
    await page.mouse.click(q.x, q.y);
    await page.waitForTimeout(350);
  }
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => !window.tw.tools.bus.active && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 20000 });
  await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.selection.value = null; });
  await page.waitForTimeout(300);
  await page.evaluate(() => { const l = window.tw.world.value.save.lines.at(-1); window.tw.selection.value = { kind: 'line', line: l.id }; });
  await page.waitForTimeout(600);
  await page.click('button:has-text("Edit route")');
  await page.waitForTimeout(500);
  let s = await state();
  if (!s.tool) {
    await page.click('button:has-text("Edit route")');
    await page.waitForTimeout(500);
    s = await state();
  }
  check(s.tool && s.pts.length === 3, 'edit: Edit route opens the three points', `tool ${s.tool}, ${s.pts.length} points`);
  const start = s.pts.map((p) => [...p]);
  const centre0 = s.centre;

  // ---- a short press on a middle point: it is a tap, nothing is deleted or moved
  await finger(await at(start[1]), { holdMs: 150 });
  s = await state();
  check(s.pts.length === 3 && metres(s.pts[1], start[1]) < 1 && s.grab, 'touch: a short press on a point deletes and moves nothing');

  // ---- one finger drags a point, the map stays
  const to = [start[1][0] + 0.006, start[1][1] - 0.003];
  await finger(await at(start[1]), { to: await at(to) });
  s = await state();
  check(s.pts.length === 3 && s.legs === 2 && metres(s.pts[1], to) < 30 && metres(s.pts[0], start[0]) < 1 && metres(s.pts[2], start[2]) < 1, 'touch: a finger drags a point, the others stay', `moved ${metres(s.pts[1], start[1]).toFixed(0)} m, ${metres(s.pts[1], to).toFixed(0)} m from the target`);
  check(metres(s.centre, centre0) < 1, 'touch: the drag did not pan the map');
  check(!s.touchAt && s.grab, 'touch: no preview point is left waiting and the map pans again');
  const moved = s.pts.map((p) => [...p]);
  await page.screenshot({ path: join(OUT, `drag-${process.env.SHADE ?? 'light'}.png`), clip: { x: box.x, y: box.y, width: box.width, height: box.height } });

  // ---- one finger on empty ground pans
  {
    const empty = { x: box.x + 150, y: box.y + box.height - 120 };
    await finger(empty, { to: { x: empty.x + 120, y: empty.y - 80 } });
    const s2 = await state();
    check(metres(s2.centre, centre0) > 20 && s2.pts.length === 3, 'touch: a finger on empty ground still pans the map', `${metres(s2.centre, centre0).toFixed(0)} m`);
    await page.evaluate((c) => window.tw.map.jumpTo({ center: c }), centre0);
    await page.waitForTimeout(500);
  }

  // ---- a second finger during a drag: pinch, the point goes back
  {
    const p = await at(moved[1]);
    await touch('touchStart', [pt(p)]);
    for (let i = 1; i <= 5; i++) { await touch('touchMove', [pt({ x: p.x + i * 8, y: p.y + i * 4 })]); await page.waitForTimeout(16); }
    await touch('touchStart', [pt({ x: p.x + 40, y: p.y + 20 }), pt({ x: p.x + 200, y: p.y + 200 }, 2)]);
    await page.waitForTimeout(200);
    await touch('touchEnd', []);
    await page.waitForTimeout(600);
    s = await state();
    check(s.pts.length === 3 && metres(s.pts[1], moved[1]) < 1, 'touch: a second finger puts a dragged point back');
    await page.evaluate((c) => window.tw.map.jumpTo({ center: c }), centre0);
    await page.waitForTimeout(500);
  }

  // ---- a press that wanders before half a second is a drag, not a delete
  {
    const to2 = [moved[1][0] - 0.003, moved[1][1] + 0.002];
    await finger(await at(moved[1]), { to: await at(to2), holdMs: 0, steps: 4, liftHeld: 700 });
    s = await state();
    check(s.pts.length === 3 && metres(s.pts[1], to2) < 30, 'touch: a press that moves is a drag; holding at the end deletes nothing', `${s.pts.length} points`);
  }

  // ---- held still: the point is deleted, and the lift adds nothing
  {
    const before = s.pts.map((p) => [...p]);
    await finger(await at(before[0]), { holdMs: 750 });
    s = await state();
    check(s.pts.length === 2 && s.legs === 1 && metres(s.pts[0], before[1]) < 1 && metres(s.pts[1], before[2]) < 1, 'touch: a point held still for half a second is deleted', `${s.pts.length} points`);
    check(!s.touchAt, 'touch: the lift after a delete adds and previews no point');
    check(metres(s.centre, centre0) < 1, 'touch: the long press did not pan the map');
  }

  // ---- a tap on empty ground still previews a point (the old tap logic is intact)
  {
    const empty = { x: box.x + 200, y: box.y + 200 };
    await finger(empty);
    s = await state();
    check(s.touchAt && s.pts.length === 2, 'touch: a tap on empty ground previews a point as before');
  }

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-1517 FAILED: ${fail}` : 'T-1517 PASSED');
process.exit(fail ? 1 : 0);
