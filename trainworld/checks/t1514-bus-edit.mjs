// T-1514: drawing and editing a bus route with the mouse, in Manhattan. The Build tab's Bus button
// starts the tool; a new route is finished each way (a click on the last point, a double click, the
// Make line button, Enter); Escape drops a new route's points, then leaves; Edit route opens the
// route, a click on the first point makes it the end new points join, a point added there goes first,
// a dragged point moves with the legs either side searched again, a right click deletes a point, and
// the inspector's Save route saves it (it dropped the edits before T-1514); Escape on an edit puts
// the route back. Screenshots of the handles in light and dark go to artifacts/trainworld/t1514/.
// Uses the Vite already on `port` (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1514-bus-edit.mjs [port]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1514');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 7014);
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

// Manhattan: 8th Ave at 34th St, Columbus Circle, the East Side; and points near them
const A = [-73.9965, 40.7510], B = [-73.9795, 40.7680], C = [-73.9570, 40.7720];
const P = (k) => [A[0] + k * 0.004, A[1] - 0.006 + k * 0.002];
const metres = (a, b) => { const k = Math.cos((a[1] * Math.PI) / 180); return Math.hypot((a[0] - b[0]) * 111320 * k, (a[1] - b[1]) * 110574); };

const chrome = await launchChrome({ prefix: process.env.PROFILE_PREFIX ?? 'tw-t1514-', profile: process.env.PROFILE_DIR, size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const at = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  const settle = async () => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };
  const state = () => page.evaluate(() => {
    const tw = window.tw, b = tw.tools.bus;
    return { tool: b.active, pts: b.pts.map((p) => [p.lng, p.lat]), legs: b.legs.length, head: b.head, corners: b.build(false).corners.map((p) => [p.lng, p.lat]), lines: tw.world.value.save.lines.length };
  });
  /** move the pointer there in steps, as a hand does, then click */
  let last = null;
  const clickAt = async (p, opts = {}) => {
    const [x, y] = await at(p);
    if (last) await page.mouse.move(last[0], last[1]);
    await page.mouse.move(x, y, { steps: last ? 12 : 1 });
    await page.waitForTimeout(250);
    await page.mouse.click(x, y, opts);
    await page.waitForTimeout(350);
    last = [x, y];
  };
  const corner = async (i) => (await state()).corners.at(i);
  const fresh = async () => {
    await page.evaluate(() => { document.activeElement?.blur?.(); window.tw.tools.exit?.(); window.tw.selection.value = null; });
    await page.waitForTimeout(200);
    last = null;
  };
  const newLineDone = (n) => page.waitForFunction((n) => window.tw.world.value.save.lines.length === n && !window.tw.tools.bus.active, n, { timeout: 20000 }).then(() => true, () => false);
  const lastLine = () => page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines[w.save.lines.length - 1], b = w.lineStats[l.id].bus;
    return { id: l.id, num: l.num, route: b.routeLL, kinds: b.kinds };
  });
  const cornersOf = (l) => { const out = []; for (let i = 0; i < l.kinds.length; i++) if (l.kinds[i] & 1) out.push([l.route[2 * i], l.route[2 * i + 1]]); return out; };

  await page.evaluate(([a, c]) => window.tw.map.fitBounds([[Math.min(a[0], c[0]) - 0.004, a[1] - 0.008], [c[0] + 0.004, c[1] + 0.004]], { padding: 120, maxZoom: 16, duration: 0 }), [A, C]);
  await settle();
  // the roads under the view load before the clicks, so every leg is searched at once
  await page.evaluate(() => { const b = window.tw.tools.bus; const c = window.tw.map.getCenter(); b.roads.look(c); });
  await page.waitForTimeout(3000);

  // ---- the Construct (Build) tab's Bus button
  await page.click('button.tab[aria-label="Build"]');
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, 'build-tab.png'), clip: { x: 0, y: 0, width: 420, height: 420 } });
  await page.click('button.tool-bus');
  await page.waitForTimeout(300);
  let s = await state();
  check(s.tool && s.pts.length === 0, 'Build tab: the Bus button starts a new bus line');
  const onBtn = await page.$eval('button.tool-bus', (b) => b.classList.contains('on'));
  check(onBtn, 'Build tab: the Bus button shows as on while drawing');

  // ---- finishing: a click on the last point placed
  const n0 = s.lines;
  await clickAt(A); await clickAt(B); await clickAt(C);
  await page.waitForTimeout(600);
  const c3 = await corner(-1);
  await clickAt(c3);
  check(await newLineDone(n0 + 1), 'finish: a click on the last point makes the line');
  const first = await lastLine();
  check(cornersOf(first).length === 3, 'finish: the line has the three clicked points', `${cornersOf(first).length}`);

  // ---- a double click
  await fresh();
  await page.keyboard.press('b');
  await page.waitForTimeout(200);
  check((await state()).tool, 'B starts the bus tool');
  await clickAt(P(0)); await clickAt(P(1));
  { const [x, y] = await at(P(2)); await page.mouse.move(x, y, { steps: 10 }); await page.waitForTimeout(300); await page.mouse.dblclick(x, y); }
  check(await newLineDone(n0 + 2), 'finish: a double click makes the line');
  check(cornersOf(await lastLine()).length === 3, 'finish: the double click placed its point first');

  // ---- the Make line button
  await fresh();
  await page.keyboard.press('b');
  await clickAt(P(0)); await clickAt(P(2));
  await page.click('#tool-notes button:has-text("Make line")');
  check(await newLineDone(n0 + 3), 'finish: the Make line button makes the line');

  // ---- Enter
  await fresh();
  await page.keyboard.press('b');
  await clickAt(P(1)); await clickAt(P(3));
  await page.keyboard.press('Enter');
  check(await newLineDone(n0 + 4), 'finish: Enter makes the line');

  // ---- Escape on a new route: the points go, the tool stays; again, the tool goes
  await fresh();
  await page.keyboard.press('b');
  await clickAt(P(0)); await clickAt(P(3));
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  s = await state();
  check(s.tool && s.pts.length === 0, 'Escape: a new route is dropped, the tool stays');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  s = await state();
  check(!s.tool && s.lines === n0 + 4, 'Escape again: the tool is left, no line made');

  // ---- editing the first line from its far end
  await fresh();
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, first.id);
  await page.waitForTimeout(500);
  await page.click('button:has-text("Edit route")');
  await page.waitForTimeout(400);
  s = await state();
  check(s.tool && s.pts.length === 3 && !s.head, 'edit: Edit route opens the route, drawing on from its last point');
  // a click on the first point makes it the end new points join
  await clickAt(s.corners[0]);
  s = await state();
  check(s.tool && s.head, 'edit: a click on the first point makes it the active end');
  const Z = [A[0] - 0.006, A[1] - 0.004];
  await clickAt(Z);
  s = await state();
  check(s.pts.length === 4 && metres(s.pts[0], Z) < 5, 'edit: a new point goes before the first', `${s.pts.length} points`);
  // the handles, hovered, in light and dark
  { const [x, y] = await at(s.corners[2]); await page.mouse.move(x, y, { steps: 6 }); await page.waitForTimeout(500); last = [x, y]; }
  // round the hovered point and the new first one
  const clipAround = async () => { const cs = (await state()).corners; const xy = await Promise.all([cs[0], cs[2]].map(at)); const xs = xy.map((p) => p[0]), ys = xy.map((p) => p[1]); const x0 = Math.max(box.x, Math.min(...xs) - 80), y0 = Math.max(box.y, Math.min(...ys) - 80); return { x: x0, y: y0, width: Math.min(box.x + box.width, Math.max(...xs) + 260) - x0, height: Math.min(box.y + box.height, Math.max(...ys) + 80) - y0 }; };
  await page.screenshot({ path: join(OUT, 'edit-handles-light.png'), clip: await clipAround() });
  await page.evaluate(() => { window.tw.shade.value = 'dark'; });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: join(OUT, 'edit-handles-dark.png'), clip: await clipAround() });
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  await page.waitForTimeout(600);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.tool-tag')].find((t) => t.style.display !== 'none')?.textContent ?? '');
  check(/Drag to move/.test(tag) && /Right click to delete/.test(tag), 'edit: a point\'s hover says how to move and delete it', tag);

  // drag the middle point (index 2, the old A->B corner B) east
  const before = await state();
  const centre = await page.evaluate(() => { const c = window.tw.map.getCenter(); return [c.lng, c.lat]; });
  const from = before.corners[2], to = [from[0] + 0.006, from[1] - 0.002];
  { const [x, y] = await at(from), [x2, y2] = await at(to); await page.mouse.move(x, y, { steps: 4 }); await page.mouse.down(); await page.mouse.move(x2, y2, { steps: 12 }); await page.waitForTimeout(300); await page.mouse.up(); await page.waitForTimeout(500); last = [x2, y2]; }
  s = await state();
  check(s.pts.length === 4 && s.legs === 3 && metres(s.pts[2], to) < 30 && metres(s.corners[1], before.corners[1]) < 1 && metres(s.corners[3], before.corners[3]) < 1, 'edit: a dragged point moves, the others stay', `moved ${metres(s.pts[2], from).toFixed(0)} m`);
  check(metres(centre, await page.evaluate(() => { const c = window.tw.map.getCenter(); return [c.lng, c.lat]; })) < 1, 'edit: the drag did not pan the map');
  // Escape during a drag puts the point back
  { const c = s.corners[2], [x, y] = await at(c), [x2, y2] = await at([c[0], c[1] + 0.003]); await page.mouse.move(x, y, { steps: 4 }); await page.mouse.down(); await page.mouse.move(x2, y2, { steps: 8 }); await page.keyboard.press('Escape'); await page.mouse.up(); await page.waitForTimeout(400); }
  const s2 = await state();
  check(s2.tool && metres(s2.pts[2], s.pts[2]) < 1, 'edit: Escape during a drag puts the point back, editing goes on');
  // right click a point: deleted (index 1, the old first point A)
  await clickAt(s2.corners[1], { button: 'right' });
  s = await state();
  check(s.pts.length === 3 && s.legs === 2 && metres(s.pts[0], Z) < 5, 'edit: a right click deletes the point', `${s.pts.length} points`);
  // the inspector's button saves
  const saveBtn = await page.evaluateHandle(() => [...document.querySelectorAll('button')].find((b) => /Save route/.test(b.textContent) && !b.closest('#tool-notes')));
  await saveBtn.asElement().click();
  await page.waitForFunction(() => !window.tw.tools.bus.active, undefined, { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(1200);
  const saved = await lastLine().then(() => page.evaluate((id) => { const b = window.tw.world.value.lineStats[id].bus; return { route: b.routeLL, kinds: b.kinds }; }, first.id));
  const sc = cornersOf(saved);
  check(sc.length === 3 && metres(sc[0], Z) < 120 && metres(sc[1], to) < 120, 'edit: Save route in the inspector saves the edits', `${sc.length} points, first ${metres(sc[0], Z).toFixed(0)} m from the click`);

  // ---- Escape on an edit: the route as it was
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, first.id);
  await page.waitForTimeout(400);
  await page.click('button:has-text("Edit route")');
  await page.waitForTimeout(300);
  last = null;
  await clickAt([C[0] + 0.004, C[1] + 0.002]);
  s = await state();
  check(s.pts.length === 4, 'edit again: a point added at the last point');
  await page.keyboard.press('Escape');
  await page.waitForTimeout(800);
  const after = await page.evaluate((id) => { const b = window.tw.world.value.lineStats[id].bus; return { route: b.routeLL, kinds: b.kinds, tool: window.tw.tools.bus.active }; }, first.id);
  check(!after.tool && after.route.length === saved.route.length && after.route.every((v, i) => Math.abs(v - saved.route[i]) < 1e-9), 'Escape on an edit: the route is as it was');

  // ---- an edit finished with no change leaves the route alone
  await page.click('button:has-text("Edit route")');
  await page.waitForTimeout(300);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(600);
  const same = await page.evaluate((id) => { const b = window.tw.world.value.lineStats[id].bus; return { route: b.routeLL, tool: window.tw.tools.bus.active }; }, first.id);
  check(!same.tool && same.route.length === saved.route.length, 'an edit with no change closes as it was');

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-1514 FAILED: ${fail}` : 'T-1514 PASSED');
process.exit(fail ? 1 : 0);
