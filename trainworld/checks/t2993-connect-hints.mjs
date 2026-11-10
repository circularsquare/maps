// T-2993: while a bus route is drawn, the stations it will stop at and the bus lines it will change with
// are lit. On a new game in New York (Brooklyn): a rail line of two stations and a bus line A, east-west;
// then the bus tool, driven by the mouse, from south of A to north of it, passing about 100 m from the
// first station: that station and line A lit, nothing once the tool is left. Pictures light and dark in
// artifacts/trainworld/t2993-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t2993-connect-hints.mjs [dev url]
// About 2 minutes (no demand solve).
import { ARTIFACTS, buildLine, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6480/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
// Church Avenue at Nostrand: the rail line runs north from here, the bus line A along Church Avenue
const C = [-73.9497, 40.6505], DLAT = 1 / 111320, DLNG = 1 / (111320 * Math.cos((C[1] * Math.PI) / 180));
const at = (east, north) => [C[0] + east * DLNG, C[1] + north * DLAT];

const chrome = await launchChrome({ prefix: 'tw-t2993-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
collectErrors(page, errors);
const settle = async (ms = 900) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const lit = () => page.evaluate(async () => { const h = (await import('/src/map/hovered.ts')).hot(); return { stations: h.stations, buses: h.buses }; });
const CLIP = { x: 400, y: 100, width: 800, height: 700 };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 180000 });
  const rail = await buildLine(page, [at(-400, 300), at(-400, 1500)], { name: 'Rail' });
  check(rail.ok, 'a rail line of two stations');
  const A = await page.evaluate(async ([a, b]) => {
    const tw = window.tw, p = tw.lngLatToPack(...a), q = tw.lngLatToPack(...b);
    const r = await tw.client.edit({ op: 'addBusLine', route: [p.x, p.y, 1, q.x, q.y, 1], name: 'Check A', colour: '#c0392b' });
    const c = await tw.client.edit({ op: 'construct', edges: [], stations: [], buses: [r.line] });
    return { ok: r.ok && c.ok, num: r.line };
  }, [at(-1500, 0), at(1500, 0)]);
  check(A.ok, 'a bus line A along the crossing street');
  const st = await page.evaluate(() => window.tw.world.value.save.stations.map((s) => ({ id: s.id, lng: s.lng, lat: s.lat })));
  const near = st.reduce((b, s) => (Math.abs(s.lat - at(0, 300)[1]) < Math.abs(b.lat - at(0, 300)[1]) ? s : b));

  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 14.5 }); }, at(-280, 200));
  await settle(2500);
  const box = await (await page.$('#map')).boundingBox();
  const screen = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  await page.getByText('Calculating demand').waitFor({ state: 'hidden', timeout: 300000 }).catch(() => console.log('the demand dialog stayed'));
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  const s0 = at(-280, -500), s1 = at(-280, 900);
  await page.mouse.move(...(await screen(s0)));
  await page.waitForTimeout(2500);
  await page.mouse.click(...(await screen(s0)));
  const end = await screen(s1);
  await page.mouse.move(end[0], end[1], { steps: 25 });
  let h = { stations: [], buses: [] };
  for (let i = 0; i < 60 && !(h.buses.length && h.stations.length); i++) { await page.waitForTimeout(400); h = await lit(); }
  check(h.stations.includes(near.id), 'the station the route passes is lit', JSON.stringify(h));
  check(h.buses.includes(A.num), 'and the bus line it crosses', JSON.stringify(h));
  const links = await page.evaluate(() => document.querySelectorAll('svg .bus-link').length);
  check(links >= 1, 'the station stop has its dashed transfer line in the preview', `${links} drawn`);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(ARTIFACTS, 't2993-drawing-light.png'), clip: CLIP });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; window.tw.overlay.invalidate(); });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(ARTIFACTS, 't2993-drawing-dark.png'), clip: CLIP });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; window.tw.overlay.invalidate(); });
  // a route away from both: nothing lit
  const far = await screen(at(-280, -480));
  await page.mouse.move(far[0], far[1], { steps: 10 });
  for (let i = 0; i < 30; i++) { await page.waitForTimeout(300); h = await lit(); if (!h.buses.length) break; }
  check(!h.buses.length && !h.stations.includes(near.id), 'a short route away from both lights neither', JSON.stringify(h));
  await page.evaluate(() => window.tw.tools.exit?.());
  await page.waitForTimeout(500);
  h = await lit();
  check(!h.buses.length && !h.stations.length, 'leaving the tool lights nothing', JSON.stringify(h));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't2993-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2993 FAILED: ${fail}` : 'T-2993 PASSED');
process.exit(fail ? 1 : 0);
