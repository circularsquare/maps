// T-3172: buses and trams may use any road in the basemap data (service ways, tracks, pedestrian streets,
// bus guideways), not only the streets and motorways. Against the old road list (written out below):
// - Dubrovnik's Stradun (a pedestrian street) and Heathrow's service roads were "too far from a road"
//   and now take a bus leg; Copenhagen's Stroget (pedestrian) is a shorter route than round it.
// - a tram click (`BusRoads.street`) lands on the Stradun now.
// - the real bus tool in Dubrovnik: refused with the old list ("Too far from a road"), a two-point route
//   with the new one. Footways, steps and cycleways stay out (no road under a click on one).
// Screenshots artifacts/trainworld/t3172-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t3172-any-road.mjs [port]
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';
import { join } from 'node:path';

const port = Number(process.argv[2] ?? 6116);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3172-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`http://localhost:${port}/?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.map && window.tw.tools?.bus, undefined, { timeout: 180000 });

  // --- legs, old list against new
  const cases = [
    { name: 'dubrovnik-stradun', a: [18.1078, 42.6411], b: [18.1105, 42.6406] },
    { name: 'heathrow-service', a: [-0.4540, 51.4700], b: [-0.4400, 51.4690] },
    { name: 'stroget', a: [12.5693, 55.6786], b: [12.5790, 55.6797] },
    { name: 'times-square', a: [-73.9855, 40.7580], b: [-73.9857, 40.7590] },
  ];
  const legs = await page.evaluate(async (cases) => {
    const { BusRoads } = await import('/src/map/busRoads.ts');
    const { MOTORWAY, TRUNK, LINK } = await import('/src/map/roadGraph.ts');
    const { BUS_CLASSES } = await import('/src/map/roadTiles.ts');
    const oldKeep = (cls, brunnel, service, access) => cls === 'trunk' ? TRUNK : BUS_CLASSES.has(cls) ? 1 : cls === 'motorway' ? MOTORWAY : cls === 'service' && !service && !access && !brunnel ? LINK : 0;
    const map = window.tw.map, olds = new BusRoads(map, () => {}, oldKeep), news = new BusRoads(map, () => {});
    const run = async (br, a, b) => {
      for (let i = 0; i < 100; i++) {
        const l = br.leg({ lng: a[0], lat: a[1] }, { lng: b[0], lat: b[1] }, 'b');
        if (l.ready) return { road: l.road, m: Math.round(l.m), refused: l.refused ?? null };
        await new Promise((r) => setTimeout(r, 300));
      }
      return { timeout: true };
    };
    const out = {};
    for (const c of cases) out[c.name] = { old: await run(olds, c.a, c.b), new: await run(news, c.a, c.b) };
    // a tram click on the Stradun
    const st = async (br, p) => { for (let i = 0; i < 100; i++) { const r = br.street({ lng: p[0], lat: p[1] }); if (r !== undefined) return r; await new Promise((r) => setTimeout(r, 300)); } };
    out.tram = { old: await st(olds, [18.1090, 42.6409]), new: await st(news, [18.1090, 42.6409]) };
    return out;
  }, cases);
  for (const n of ['dubrovnik-stradun', 'heathrow-service']) {
    const r = legs[n];
    check(!r.old.road && r.old.refused === 'far' && r.new.road, `${n}: refused before, a road leg now`, JSON.stringify(r));
  }
  check(legs.stroget.new.road && legs.stroget.new.m < 0.8 * legs.stroget.old.m, 'Stroget: the pedestrian street is a much shorter route', JSON.stringify(legs.stroget));
  check(legs['times-square'].new.road && legs['times-square'].new.m <= legs['times-square'].old.m, 'a street leg is no longer', JSON.stringify(legs['times-square']));
  check(legs.tram.old === null && legs.tram.new !== null, 'a tram click on the Stradun lands on it now', JSON.stringify(legs.tram));

  // --- the real bus tool in Dubrovnik, the old list then the new one
  const box = await (await page.$('#map')).boundingBox();
  const screen = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  const settle = async () => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(800); };
  const tag = () => page.evaluate(() => { const t = window.tw.tools.bus.tag; return t.style.display !== 'none' ? t.textContent : ''; });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [18.1092, 42.6409], zoom: 16 }));
  await settle();
  const a = [18.1078, 42.6411], b = [18.1105, 42.6406];
  const setKeep = (old) => page.evaluate(async (old) => {
    const { MOTORWAY, TRUNK, LINK } = await import('/src/map/roadGraph.ts');
    const { BUS_CLASSES, DRIVE_CLASSES, PEDESTRIAN } = await import('/src/map/roadTiles.ts');
    const r = window.tw.tools.bus.roads;
    r.keep = old
      ? (cls, brunnel, service, access) => cls === 'trunk' ? TRUNK : BUS_CLASSES.has(cls) ? 1 : cls === 'motorway' ? MOTORWAY : cls === 'service' && !service && !access && !brunnel ? LINK : 0
      : (cls, _b, _s, _a, sub) => cls === 'trunk' ? TRUNK : DRIVE_CLASSES.has(cls) || (cls === 'path' && sub === PEDESTRIAN) ? 1 : cls === 'motorway' ? MOTORWAY : 0;
    r.tiles.clear(); r.graph.clear(); r.legs.clear(); r.snaps.clear(); r.template = null;
  }, old);
  const waitTag = async (want) => { for (let i = 0; i < 40; i++) { await page.waitForTimeout(300); const t = await tag(); if (want(t)) return t; } return tag(); };
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  await setKeep(true);
  await page.mouse.move(...(await screen(a)));
  const before = await waitTag((t) => t !== '');
  await page.screenshot({ path: join(ARTIFACTS, 't3172-dubrovnik-before.png') });
  check(before === 'Too far from a road', 'old list: a point on the Stradun is refused', JSON.stringify(before));
  await setKeep(false);
  await page.mouse.move(...(await screen([18.1079, 42.6411])));
  await page.mouse.move(...(await screen(a)));
  await page.waitForTimeout(3000);
  const after = await tag();
  check(after !== 'Too far from a road', 'new list: the same point is not refused', JSON.stringify(after));
  await page.mouse.click(...(await screen(a)));
  await page.waitForTimeout(800);
  await page.mouse.move(...(await screen(b)), { steps: 20 });
  await waitTag((t) => t !== '');
  await page.waitForTimeout(800);
  await page.mouse.click(...(await screen(b)));
  await page.waitForTimeout(1200);
  const route = await page.evaluate(() => { const t = window.tw.tools.bus; return { pts: t.pts.length, legs: t.legs.map((l) => ({ road: l.road, m: Math.round(l.m) })) }; });
  await page.mouse.move(box.x + 40, box.y + box.height - 60);
  await page.waitForTimeout(500);
  await page.screenshot({ path: join(ARTIFACTS, 't3172-dubrovnik-after.png') });
  check(route.pts === 2 && route.legs[0]?.road, 'new list: a two-point route along the Stradun', JSON.stringify(route));
  await page.evaluate(() => window.tw.tools.exit?.());

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
