// T-1513: the delete tool removes a bus line. A planned bus line goes at once with a click on its route
// (and comes back with undo); a constructed one asks first and goes after Remove, with nothing refunded.
// Hovering the route says what a click removes. Uses the Vite already on `port` (dev-nowatch.mjs). From
// the maps root:
//   node trainworld/checks/t1513-bus-delete.mjs [port]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1512');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6312);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const chrome = await launchChrome({ prefix: 'tw-t1513-', profile: process.env.PROFILE_DIR, size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`http://localhost:${port}/?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const [lng, lat] = [-73.87, 40.75];
  await page.evaluate(([a, b]) => window.tw.map.jumpTo({ center: [a, b], zoom: 13.5 }), [lng, lat]);
  await page.waitForFunction(() => window.tw.map.loaded());
  const make = (from, to, build) => page.evaluate(async ([from, to, build]) => {
    const tw = window.tw;
    const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
    const add = await tw.client.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
    if (build) await tw.client.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
    return add.line;
  }, [from, to, build]);
  const planned = await make([lng - 0.02, lat + 0.01], [lng + 0.02, lat + 0.01], false);
  const built = await make([lng - 0.02, lat - 0.01], [lng + 0.02, lat - 0.01], true);
  await page.waitForTimeout(1500);
  const lines = () => page.evaluate(() => window.tw.world.value.save.lines.map((l) => l.num));
  check((await lines()).includes(planned) && (await lines()).includes(built), 'two bus lines made, one planned and one constructed');

  const box = await (await page.$('#map')).boundingBox();
  const at = (p) => page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p);
  await page.evaluate(async () => { const { tool } = await import('/src/game/ui.ts'); tool.value = 'delete'; });
  await page.waitForTimeout(300);
  const tagText = () => page.evaluate(() => { return [...document.querySelectorAll('.tool-tag')].filter((t) => t.style.display !== 'none').map((t) => t.innerText).join(' | '); });

  // planned: hover names it, a click removes it at once
  const [x1, y1] = await at([lng - 0.005, lat + 0.01]);
  await page.mouse.move(box.x + x1 - 20, box.y + y1 + 20);
  await page.mouse.move(box.x + x1, box.y + y1, { steps: 4 });
  await page.waitForTimeout(400);
  const tag = await tagText();
  check(/Remove Bus \d+/.test(tag), 'hovering a planned bus route says what a click removes', tag.replace(/\n/g, ' / '));
  await page.screenshot({ path: join(OUT, 'delete-hover.png') });
  await page.mouse.click(box.x + x1, box.y + y1);
  await page.waitForTimeout(1200);
  check(!(await lines()).includes(planned), 'a click on a planned bus route removes the line at once');
  check(await page.evaluate(() => window.tw.world.value.save.lines.length === 1), 'the other bus line is still there');
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await page.waitForTimeout(1200);
  check((await lines()).includes(planned), 'undo brings it back');

  // constructed: asks first, nothing refunded
  const [x2, y2] = await at([lng - 0.005, lat - 0.01]);
  await page.mouse.move(box.x + x2 - 20, box.y + y2 + 20);
  await page.mouse.move(box.x + x2, box.y + y2, { steps: 4 });
  await page.waitForTimeout(400);
  await page.mouse.click(box.x + x2, box.y + y2);
  await page.waitForTimeout(800);
  const ask = await page.evaluate(async () => { const { ask } = await import('/src/game/ui.ts'); const a = ask.peek(); return a && { text: a.text, yes: a.yes }; });
  check(ask && /Nothing is refunded/.test(ask.text), 'a constructed bus line asks first', JSON.stringify(ask));
  check((await lines()).includes(built), 'it is still there while the question stands');
  const cash0 = await page.evaluate(() => window.tw.world.value.save.cash);
  await page.evaluate(async () => { const { ask } = await import('/src/game/ui.ts'); ask.peek()?.run(); });
  await page.waitForTimeout(1200);
  check(!(await lines()).includes(built), 'after Remove the constructed line is gone');
  const cash1 = await page.evaluate(() => window.tw.world.value.save.cash);
  check(cash1 <= cash0, 'nothing is refunded', `${cash0} -> ${cash1}`);

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
