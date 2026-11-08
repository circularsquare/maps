// T-2764 and T-2765 in headless Chrome.
//   node trainworld/checks/t2764-line-colour-quick.mjs [url]      (from the maps root; about a minute)
// 1. Five new lines made one after another without waiting (empty, as when the panel is closed at once)
//    get five different colours.
// 2. Dragging a loose track end with the select tool shows the tangent line (an SVG line in the marks
//    layer); dragging a node with track on both sides shows none.
// Screenshot of the tangent while dragging: artifacts/trainworld/linecolour/tangent-drag.png.
import { mkdir } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6160/';
const OUT = `${ARTIFACTS}linecolour/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t2764-', size: [1200, 800], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);

  // 1: colours of lines made back to back
  const colours = await page.evaluate(async () => {
    const { createLine } = await import('/src/game/lineActions.ts');
    const before = new Set(window.tw.world.value.save.lines.map((l) => l.num));
    await Promise.all([1, 2, 3, 4, 5].map(() => createLine()));
    return window.tw.world.value.save.lines.filter((l) => !before.has(l.num)).map((l) => [l.colour, l.name]);
  });
  report(colours.length === 5 && new Set(colours.map((c) => c[0])).size === 5, 'five lines made at once have five colours', colours.map((c) => c[0]).join(' '));
  report(new Set(colours.map((c) => c[1])).size === 5, 'and five names', colours.map((c) => c[1]).join(', '));

  // 2: the tangent while dragging a loose end
  await page.evaluate(() => window.tw.tools.exit());
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client, a = tw.lngLatToPack(-73.968, 40.778);
    const pi = { x: a.x + 400, y: a.y }, b = { x: pi.x + 400 * Math.cos(Math.PI / 3), y: pi.y + 400 * Math.sin(Math.PI / 3) };
    await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [{ x: pi.x, y: pi.y, level: 0 }], single: false });
    const e = tw.world.value.edges.at(-1);
    return { a: e.a, b: e.b };
  });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'select'; });
  const ends = { a: made.a, b: made.b };
  for (const which of ['b', 'a']) {
    const node = ends[which];
    const ll = await page.evaluate((n) => { const x = window.tw.world.value.nodes.get(n); window.tw.map.jumpTo({ center: [x.lng, x.lat], zoom: 15 }); return [x.lng, x.lat]; }, node);
    await page.waitForTimeout(1500);
    const r = await page.evaluate((ll) => { const m = window.tw.map, rc = m.getContainer().getBoundingClientRect(), p = m.project(ll); return { x: rc.left + p.x, y: rc.top + p.y }; }, ll);
    await page.mouse.move(r.x, r.y);
    await page.waitForTimeout(200);
    await page.mouse.down();
    for (let i = 1; i <= 8; i++) { await page.mouse.move(Math.round(r.x + 6 * i), Math.round(r.y + 3 * i)); await page.waitForTimeout(60); }
    await page.waitForTimeout(1200);
    const shown = await page.evaluate(() => [...document.querySelectorAll('.tool-marks > svg')].filter((s) => s.style.display !== 'none' && s.firstElementChild?.tagName === 'line' && (Number(s.firstElementChild.getAttribute('x2')) !== 0 || Number(s.firstElementChild.getAttribute('y2')) !== 0)).length);
    if (which === 'b') await page.screenshot({ path: `${OUT}tangent-drag.png` });
    report(shown >= 1, `a tangent shows while dragging the loose end ${which}`, `${shown} lines`);
    await page.mouse.up();
    await page.waitForTimeout(900);
    await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true));
    await page.waitForTimeout(700);
  }
} catch (e) {
  failed = true;
  console.log('FAIL', e.message);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
