// T-2451: the bus tool refuses a point the roads do not reach. In Manhattan, with the real tool driven by
// the mouse: a point on 8th Avenue, then the pointer out over the Hudson (no road within 100 m: the tag
// says so before the click, the click adds nothing and the message says why), then St George from the
// Battery (roads on both, none between under the detour cap: "No road path found."), and a route along the avenues still
// makes a line whose every stretch follows roads. Screenshots go to artifacts/trainworld/ferrybus/. Uses
// the Vite already on `port` (dev-nowatch.mjs). From the maps root:
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
  const state = () => page.evaluate(async () => {
    const b = window.tw.tools.bus, { notice } = await import('/src/game/world.ts');
    const tag = b.tag;
    return { pts: b.pts.length, notice: notice.value?.text ?? '', bad: !!notice.value?.bad, tag: tag && tag.style.display !== 'none' ? tag.textContent : '', tagBad: !!tag?.classList.contains('bad') };
  });
  const clearNotice = () => page.evaluate(async () => { const { notice } = await import('/src/game/world.ts'); notice.value = null; });
  // walk the pointer to a place, let its roads load, and say what the tool shows
  const hover = async (from, to) => {
    await page.mouse.move(...(await screen(from)));
    const [x, y] = await screen(to);
    await page.mouse.move(x, y, { steps: 25 });
    // until the leg to the pointer has its roads (the tag waits for them), 15 s at most
    for (let i = 0; i < 50; i++) {
      await page.waitForTimeout(300);
      await page.mouse.move(x + (i % 2), y);
      if (i >= 4 && await page.evaluate(() => { const b = window.tw.tools.bus; return b.build(true).ready && b.tag.style.display !== 'none'; })) break;
    }
    return [x, y];
  };

  // --- Manhattan: 8th Avenue at 42nd Street, then out over the Hudson
  const ave = [-73.9900, 40.7575], hudson = [-74.0150, 40.7610];
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-74.0020, 40.7590], zoom: 14.6 }); });
  await settle();
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  await page.mouse.move(...(await screen(ave)));
  await page.waitForTimeout(1500);
  await page.mouse.click(...(await screen(ave)));
  await page.waitForTimeout(500);
  check((await state()).pts === 1, 'a click on 8th Avenue adds a point');
  const at = await hover(ave, hudson);
  let s = await state();
  await page.screenshot({ path: join(OUT, 'bus-refused-hover.png') });
  check(s.tag === 'Too far from a road' && s.tagBad, 'over the Hudson the tag says why before the click', JSON.stringify(s.tag));
  await clearNotice();
  await page.mouse.click(...at);
  await page.waitForTimeout(400);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-refused-click.png') });
  check(s.pts === 1, 'the click over the Hudson adds nothing', `${s.pts} points`);
  check(s.notice === 'Too far from a road. Click within 100 m of one.' && s.bad, 'and the message says why', JSON.stringify(s.notice));

  // --- a route along the avenues still makes a line, every stretch on roads
  // 9th Avenue at 34th Street (clear of the debug panel at the top right)
  const ave2 = [-73.9960, 40.7525];
  await hover(hudson, ave2);
  await page.mouse.click(...(await screen(ave2)));
  await page.waitForTimeout(500);
  check((await state()).pts === 2, 'a second point up the avenue is added');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0 && !window.tw.tools.bus.active, undefined, { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(1000);
  const line = await page.evaluate(() => {
    const w = window.tw.world.value, l = w.save.lines[w.save.lines.length - 1];
    return l ? w.lineStats[l.id].bus.kinds : null;
  });
  check(!!line && line.slice(1).every((k) => k & 2), 'the line is made, every stretch on roads', line ? `${line.length} points` : 'no line');
  await page.evaluate(() => { window.tw.renameLinePrompt.value = null; window.tw.selection.value = null; }).catch(() => {});
  await page.keyboard.press('Escape');

  // --- the Battery to St George: roads at both ends, the only road between (by Brooklyn and the
  // Verrazzano) over 3 times the straight line
  const battery = [-74.0135, 40.7025], island = [-74.0790, 40.6415];
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: [-74.0463, 40.6720], zoom: 12.3 }); });
  await settle();
  await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); });
  await page.mouse.move(...(await screen(battery)));
  await page.waitForTimeout(1500);
  await page.mouse.click(...(await screen(battery)));
  await page.waitForTimeout(500);
  const at2 = await hover(battery, island);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-no-road-path-hover.png') });
  check(s.tag === 'No road path found' && s.tagBad, 'to St George the tag says there is no road path', JSON.stringify(s.tag));
  await clearNotice();
  await page.mouse.click(...at2);
  await page.waitForTimeout(400);
  s = await state();
  await page.screenshot({ path: join(OUT, 'bus-no-road-path-click.png') });
  check(s.pts === 1 && s.notice === 'No road path found.', 'and the click is refused', `${s.pts} points, ${JSON.stringify(s.notice)}`);
  await page.evaluate(() => window.tw.tools.exit?.());

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
