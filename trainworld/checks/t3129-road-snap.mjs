// T-3129: rail track snaps to the main roads (Snap to roads in the track bar). With the real tool driven by
// the mouse on a new game, for each route in ROUTES: a first click on a road, the pointer over a second place
// along it, the ghost's route (phase, price tag, corners, fit tolerance, slowest curve), then the second click
// placing it. Prints a table and screenshots artifacts/trainworld/t3129-<name>.png; asserts in the checks
// below. Set ROUTES from the command line for a sweep:  ROUTES="name,lng,lat,lng,lat,level,design,zoom;..."
// From the maps root, on a Vite of your own:   node trainworld/checks/t3129-road-snap.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6116/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// name, from, to (lng, lat), level, target speed km/h, zoom
const DEFAULT = [
  ['chicago-danryan', [-87.6308, 41.8191], [-87.6307, 41.8131], 0, 100, 14.5],
  ['manhattan-34th-viaduct', [-73.99346, 40.7522], [-73.98455, 40.74845], 1, 60, 14.5],
  ['manhattan-tunnel', [-74.00037, 40.7617], [-73.99394, 40.75898], -1, 60, 14.5],
  ['bangkok-primary', [100.57491, 13.74704], [100.58473, 13.74481], 1, 80, 14.5],
];
const ROUTES = process.env.ROUTES
  ? process.env.ROUTES.split(';').map((r) => { const [n, a, b, c, d, lv, v, z] = r.split(','); return [n, [+a, +b], [+c, +d], +lv, +v, +(z ?? 14)]; })
  : DEFAULT;

const chrome = await launchChrome({ prefix: 'tw-t3129-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { tram: false, roads: true, level: 0, keep: true, tracks: 2 } })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const screen = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  const rows = [];
  /** one route: the first click, the pointer on B, the ghost's figures, then (with `place`) the second click */
  const route = async (name, A, B, level, design, zoom, place = false) => {
    await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [[(A[0] + B[0]) / 2, (A[1] + B[1]) / 2], zoom]);
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
    await wait(1500);
    await page.evaluate(async ([level, design]) => {
      const ui = await import('/src/game/ui.ts');
      ui.tool.value = 'track'; ui.tramTrack.value = false; ui.snapReal.value = false; ui.snapBeside.value = false; ui.snapRoads.value = true;
      ui.buildLevel.value = level; ui.keepPlacing.value = true; ui.designSpeed.value = design;
    }, [level, design]);
    await wait(800);
    const [x0, y0] = await screen(A), [x1, y1] = await screen(B);
    await page.mouse.move(x0, y0, { steps: 3 });
    await wait(1500);
    await page.mouse.click(x0, y0);
    await wait(400);
    await page.mouse.move(x0 + (x1 - x0) * 0.9, y0 + (y1 - y0) * 0.9, { steps: 4 });
    await page.mouse.move(x1, y1, { steps: 4 });
    await page.waitForFunction(() => { const s = window.tw.tools.street; return (s.phase === 'done' && window.tw.tools.lastPreview) || s.phase === 'none' || s.phase === 'off'; }, undefined, { timeout: 40000 }).catch(() => {});
    await wait(800);
    const r = await page.evaluate(() => {
      const T = window.tw.tools, pv = T.lastPreview, l = T.street.last;
      return { phase: T.street.phase, ok: pv?.ok, issues: pv?.issues, tag: T.tag.style.display !== 'none' ? T.tag.innerText.replace(/\n+/g, ' | ') : '', last: l, profile: T.street.snapProfile };
    });
    await page.screenshot({ path: join(ARTIFACTS, `t3129-${name}.png`) });
    rows.push({ name, ...r });
    console.log(name, JSON.stringify(r));
    if (r.phase === 'done' && place) {
      const before = await page.evaluate(() => window.tw.world.value.edges.length);
      await page.mouse.click(x1, y1);
      await wait(1500);
      r.placed = await page.evaluate((before) => ({ before, after: window.tw.world.value.edges.length }), before);
    }
    await page.keyboard.press('Escape');
    return r;
  };
  for (const [name, A, B, level, design, zoom] of ROUTES) {
    await route(name, A, B, level, design, zoom);
    await page.evaluate(() => window.tw.client.edit?.({ op: 'undo' })).catch(() => {});
    await wait(300);
  }
  if (!process.env.ROUTES) {
    // two routes that cross at ground level: the second is ramped or flown over the first (T-1615, T-3092)
    await route('cross-1', [-73.99346, 40.7522], [-73.98455, 40.74845], 0, 60, 14.5, true);
    const second = await route('cross-2', [-73.98504, 40.75356], [-73.9905, 40.7445], 0, 60, 14.5, true);
    check(second.phase === 'done' && second.ok === true, 'a second route across the first can be built (ramp or flyover)', `${JSON.stringify(second.issues)} ${second.tag}`);
    check(second.placed && second.placed.after > second.placed.before, 'and placing it adds track', JSON.stringify(second.placed));
  }
  // the bar's switch, its exclusions, the hint layer and Ctrl
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.tramTrack.value = false; ui.snapRoads.value = false; ui.snapReal.value = true; ui.snapBeside.value = false;
  });
  await wait(500);
  const btn = page.locator('button[aria-label="Snap to roads"]');
  check(await btn.count() === 1 && (await btn.getAttribute('aria-pressed')) === 'false', 'the bar has Snap to roads, off');
  await btn.click();
  await wait(500);
  const st = await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return { roads: ui.snapRoads.value, real: ui.snapReal.value, beside: ui.snapBeside.value, hint: !!window.tw.map.getLayer('rail_roads_hint') }; });
  check(st.roads && !st.real && !st.beside && st.hint, 'turning it on turns Snap to real network off and tints the roads', JSON.stringify(st));
  check(await page.evaluate(() => window.tw.tools.street.isOn && window.tw.tools.street.snapProfile === 'rail'), 'the street snap runs the rail profile');
  await page.keyboard.down('Control');
  await wait(300);
  check(await page.evaluate(() => !window.tw.tools.street.isOn), 'holding Ctrl draws by hand');
  await page.keyboard.up('Control');
  await wait(300);
  check(await page.evaluate(() => window.tw.tools.street.isOn), 'and letting go snaps again');
  await page.locator('button[aria-label="Snap beside track"]').click();
  await wait(300);
  check(await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return !ui.snapRoads.value && ui.snapBeside.value; }), 'Snap beside track turns it off');
  await page.screenshot({ path: join(ARTIFACTS, 't3129-bar.png') });
  for (const r of rows) {
    check(r.profile === 'rail', `${r.name}: the rail profile`, r.profile);
    check(r.phase === 'done' && r.ok === true, `${r.name}: a route follows the road and can be built`, `${r.phase} ${JSON.stringify(r.issues)} ${r.tag}`);
  }
  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3129-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3129 FAILED: ${fail}` : 'T-3129 PASSED');
process.exit(fail ? 1 : 0);
