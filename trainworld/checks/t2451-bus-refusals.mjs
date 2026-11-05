// T-2451, T-2467, T-2468, T-2469: the bus and ferry tools place only where they can. With the real tools
// driven by the mouse in New York:
// - bus: a first point over the Hudson is refused (no road within 100 m), the tag saying so before the
//   click; a point on 8th Avenue, then the pointer over the Hudson again (refused, the leg dashed red,
//   never straight in the line's colour); a route up the avenues makes a line whose every stretch follows
//   roads; Hoboken from 14th Street across the Hudson (roads on both sides, the tunnels too far round) is
//   refused as "No road path found."; legs over the Verrazzano and the Brooklyn Bridge follow roads.
// - ferry: the pointer on the East River 140 m off the Manhattan shore shows the port it would place on
//   the shore ("Place port"), and the click places it there.
// Screenshots go to artifacts/trainworld/ferrybus/. Uses the Vite already on `port` (dev-nowatch.mjs).
// From the maps root:
//   node trainworld/checks/t2451-bus-refusals.mjs [port]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 'ferrybus');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6187);
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const metres = (a, b) => { const k = Math.cos((a[1] * Math.PI) / 180); return Math.hypot((a[0] - b[0]) * 111320 * k, (a[1] - b[1]) * 110574); };

const chrome = await launchChrome({ prefix: process.env.PROFILE_PREFIX ?? 'tw-t2451-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  // one solving worker: this check needs no demand
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
  const box = await (await page.$('#map')).boundingBox();
  const screen = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  const settle = async () => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };
  const state = (tool = 'bus') => page.evaluate(async (tool) => {
    const b = window.tw.tools[tool], { notice } = await import('/src/game/world.ts');
    const tag = b.tag;
    return { pts: (b.pts ?? b.ports).length, notice: notice.value?.text ?? '', bad: !!notice.value?.bad, tag: tag && tag.style.display !== 'none' ? tag.textContent : '', tagBad: !!tag?.classList.contains('bad') };
  }, tool);
  const clearNotice = () => page.evaluate(async () => { const { notice } = await import('/src/game/world.ts'); notice.value = null; });
  // walk the pointer to a place and wait for the tool's answer there
  const hover = async (from, to, ready) => {
    await page.mouse.move(...(await screen(from)));
    const [x, y] = await screen(to);
    await page.mouse.move(x, y, { steps: 25 });
    for (let i = 0; i < 50; i++) {
      await page.waitForTimeout(300);
      if (i >= 4 && await page.evaluate(ready)) break;
    }
    return [x, y];
  };
  const busReady = () => { const b = window.tw.tools.bus; return b.build(true).ready && b.tag.style.display !== 'none'; };
  // what the bus tool draws in the line's colour, red, and how long its straight pieces are
  const drawn = () => page.evaluate(() => {
    const paths = [...document.querySelectorAll('svg.bus-draw path')].map((p) => ({ d: p.getAttribute('d'), style: p.getAttribute('style') }));
    return { red: paths.filter((p) => /--bad/.test(p.style)).length, colour: paths.filter((p) => /stroke:#/.test(p.style)).map((p) => p.d) };
  });

  // --- the bus tool: a first point over the Hudson
  const ave = [-73.9900, 40.7575], hudson = [-74.0075, 40.7700];
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-74.0020, 40.7620], zoom: 14.4 }); });
  await settle();
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  let at = await hover(ave, hudson, busReady);
  let s = await state();
  check(s.tag === 'Too far from a road' && s.tagBad, 'a first point over the Hudson: the tag says why before the click', JSON.stringify(s.tag));
  await clearNotice();
  await page.mouse.click(...at);
  await page.waitForTimeout(400);
  s = await state();
  check(s.pts === 0 && s.notice === 'Too far from a road. Click within 100 m of one.', 'and the click adds nothing, saying why', `${s.pts} points, ${JSON.stringify(s.notice)}`);

  // --- a point on 8th Avenue, then over the Hudson again
  await page.mouse.move(...(await screen(ave)));
  await page.waitForTimeout(1500);
  await page.mouse.click(...(await screen(ave)));
  await page.waitForTimeout(800);
  check((await state()).pts === 1, 'a click on 8th Avenue adds a point');
  at = await hover(ave, hudson, busReady);
  s = await state();
  let dr = await drawn();
  await page.screenshot({ path: join(OUT, 'bus-refused-hover.png') });
  check(s.tag === 'Too far from a road' && s.tagBad, 'over the Hudson the tag says why before the click', JSON.stringify(s.tag));
  check(dr.red > 0 && dr.colour.length === 0, 'the leg to the pointer is red and dashed, none in the line\'s colour', JSON.stringify({ red: dr.red, colour: dr.colour.length }));
  await clearNotice();
  await page.mouse.click(...at);
  await page.waitForTimeout(400);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-refused-click.png') });
  check(s.pts === 1 && s.notice === 'Too far from a road. Click within 100 m of one.' && s.bad, 'the click over the Hudson adds nothing and says why', `${s.pts} points, ${JSON.stringify(s.notice)}`);

  // --- a route along the avenues still makes a line, every stretch on roads
  const ave2 = [-73.9960, 40.7525];
  await hover(hudson, ave2, busReady);
  dr = await drawn();
  check(dr.red === 0 && dr.colour.length > 0, 'the leg to 34th Street is drawn along the roads in the line\'s colour', JSON.stringify({ red: dr.red, colour: dr.colour.length }));
  await page.mouse.click(...(await screen(ave2)));
  await page.waitForTimeout(800);
  check((await state()).pts === 2, 'a second point down the avenue is added');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0 && !window.tw.tools.bus.active, undefined, { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(1000);
  const line = await page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines[w.save.lines.length - 1];
    return l ? w.lineStats[l.id].bus.kinds : null;
  });
  check(!!line && line.slice(1).every((k) => k & 2), 'the line is made, every stretch on roads', line ? `${line.length} points` : 'no line');
  await page.keyboard.press('Escape');
  await page.evaluate(() => { window.tw.selection.value = null; });

  // --- Hoboken from 14th Street: roads on both sides of the Hudson, the tunnels too far round
  const west14 = [-74.0085, 40.7425], hoboken = [-74.0275, 40.7440];
  await page.evaluate(() => { window.tw.map.jumpTo({ center: [-74.0180, 40.7432], zoom: 14.6 }); });
  await settle();
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  await page.mouse.move(...(await screen(west14)));
  await page.waitForTimeout(1500);
  await page.mouse.click(...(await screen(west14)));
  await page.waitForTimeout(800);
  at = await hover(west14, hoboken, busReady);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-no-road-path-hover.png') });
  check(s.tag === 'No road path found' && s.tagBad, 'to Hoboken the tag says there is no road path', JSON.stringify(s.tag));
  await clearNotice();
  await page.mouse.click(...at);
  await page.waitForTimeout(600);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-no-road-path-click.png') });
  check(s.pts === 1 && s.notice === 'No road path found.', 'and the click is refused', `${s.pts} points, ${JSON.stringify(s.notice)}`);
  await page.evaluate(() => window.tw.tools.exit?.());

  // --- bridges: legs over the Verrazzano (a motorway) and the Brooklyn Bridge follow roads
  for (const [name, a, b, center, zoom] of [
    ['verrazzano', [-74.0300, 40.6100], [-74.0600, 40.6060], [-74.0450, 40.6080], 13.6],
    ['brooklyn', [-74.0040, 40.7120], [-73.9905, 40.6990], [-73.9972, 40.7055], 14.2],
  ]) {
    await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [center, zoom]);
    await settle();
    await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
    await page.mouse.move(...(await screen(a)));
    await page.waitForTimeout(1500);
    await page.mouse.click(...(await screen(a)));
    await page.waitForTimeout(800);
    const end = await hover(a, b, busReady);
    await page.mouse.click(...end);
    await page.waitForTimeout(1500);
    const r = await page.evaluate(() => { const b = window.tw.tools.bus; return { pts: b.pts.length, legs: b.legs.map((l) => ({ road: l.road, m: Math.round(l.m) })) }; });
    await page.mouse.move(box.x + 40, box.y + box.height - 60);
    await page.waitForTimeout(400);
    await page.screenshot({ path: join(OUT, `bus-bridge-${name}.png`) });
    const straight = metres(a, b);
    check(r.pts === 2 && r.legs[0]?.road && r.legs[0].m < 3 * straight, `${name}: the leg follows roads over the water`, `${JSON.stringify(r.legs)}, straight ${Math.round(straight)} m`);
    await page.evaluate(() => window.tw.tools.exit?.());
  }

  // --- the ferry tool: the pointer on the East River by the Manhattan shore
  const river = [-73.9695, 40.7440];
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-73.9700, 40.7450], zoom: 15 }); });
  await settle();
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.ferry.begin(null); });
  await hover([-73.9720, 40.7450], river, () => { const f = window.tw.tools.ferry; return f.under.kind !== 'unknown' && f.tag.style.display !== 'none'; });
  s = await state('ferry');
  const ghost = await page.evaluate(() => { const u = window.tw.tools.ferry.under; return u.kind === 'shore' ? [u.landing.lng, u.landing.lat] : null; });
  await page.screenshot({ path: join(OUT, 'ferry-snap-hover.png') });
  check(s.tag === 'Place port' && !!ghost, 'on the water by the shore the tag offers a port and shows where it lands', `${JSON.stringify(s.tag)}, landing ${ghost ? Math.round(metres(ghost, river)) + ' m off' : 'none'}`);
  await page.mouse.click(...(await screen(river)));
  await page.waitForTimeout(1500);
  const placed = await page.evaluate(() => window.tw.tools.ferry.ports.map((p) => [p.landing.lng, p.landing.lat]));
  check(placed.length === 1 && !!ghost && metres(placed[0], ghost) < 1, 'the click places the port where the preview showed', JSON.stringify(placed));
  await page.evaluate(() => window.tw.tools.exit?.());

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
