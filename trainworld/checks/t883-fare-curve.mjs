// T-883, T-1112: the fare curve reaches demand, and the Money tab's two-node control works. On the
// real save (a ?new=1 game, the weekday's final): the default fare; the same line through other
// nodes (no change at all); an older three-node curve kept as its first and last (here the default
// again); twice the rate per km (fewer rail trips); a line below $0 at 0 km lifted at its first node; $0 at 0 km
// then $0.10 a km (more rail trips). Then in the Money tab: dragging the second
// node sends nothing until release, then the new fare; Ctrl+Z, Ctrl+Y and the undo button take it
// back and put it back (T-1130); wheel zoom puts it past the plot as an
// arrow, and a click on the arrow zooms back out. Screenshots of the fare graph (its default view,
// zoomed with the arrow, mid-drag), light and dark, go to artifacts/trainworld/t1111/.
//   node trainworld/checks/t883-fare-curve.mjs [url]   (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure. About five minutes.
import { mkdir, readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6150/').replace(/\/?$/, '/');
const OUT = 'artifacts/trainworld/t1111/';
const DEFAULT_RAIL = 12796392;

let failed = false;
const report = (name, ok, detail) => { if (!ok) failed = true; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t883-' });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });

  /** the next final after `after` (a demand log version), its rail trips a day */
  const final = async (after) => {
    await demandFinal(page, { after, lines: 100, timeout: 400000, polling: 250 });
    return page.evaluate(() => { const v = window.twDemand.view(); return { version: v.version, rail: v.railTripsPerDay }; });
  };
  const setNodes = (nodes) => page.evaluate((nodes) => window.tw.client.edit({ op: 'fares', nodes }), nodes);
  const nodesNow = () => page.evaluate(() => JSON.stringify(window.tw.moneyState.value.fares.nodes));
  const fmt = (v) => Math.round(v).toLocaleString('en-US');

  const a = await final(-1);
  report('default fare', Math.round(a.rail) === DEFAULT_RAIL, `(${fmt(a.rail)} rail trips a day; want ${fmt(DEFAULT_RAIL)})`);

  // the same fares give the demand nothing new to solve, so no new final may come
  await setNodes([[10, 2.5], [100, 11.5]]);
  await page.waitForTimeout(15000);
  const b = await final(-1);
  report('the same line through other nodes', Math.round(b.rail) === Math.round(a.rail), `(${fmt(b.rail)})`);

  await setNodes([[1, 1.6], [30, 4.5], [100, 11.5]]);
  await page.waitForTimeout(15000);
  const c = await final(-1);
  const cn = await nodesNow();
  report('three nodes keep the first and last', cn === '[[1,1.6],[100,11.5]]' && Math.round(c.rail) === Math.round(a.rail), `(nodes ${cn}, ${fmt(c.rail)})`);

  await setNodes([[1, 1.7], [100, 21.5]]);
  const d = await final(c.version);
  const dd = (d.rail - a.rail) / a.rail;
  report('twice the rate per km', dd < 0, `(${fmt(d.rail)}, ${(100 * dd).toFixed(2)}%)`);

  // a line that would be below $0 at 0 km is lifted at the first node, so demand and income agree
  await setNodes([[5, 0], [100, 9.5]]);
  await page.waitForFunction(() => JSON.stringify(window.tw.moneyState.value.fares.nodes) === '[[5,0.48],[100,9.5]]', undefined, { timeout: 30000 });
  report('a line below $0 at 0 km is lifted', true, '(nodes [[5,0.48],[100,9.5]])');
  const e0 = await final(d.version);

  await setNodes([[1, 0.1], [100, 10]]);
  const e = await final(e0.version);
  const de = (e.rail - a.rail) / a.rail;
  report('$0 at 0 km, then $0.10 a km', Math.round(e.rail) === 14078830, `(${fmt(e.rail)}, ${(100 * de).toFixed(2)}% more rail trips)`);

  await setNodes([[1, 1.6], [100, 11.5]]);
  await page.waitForFunction(() => JSON.stringify(window.tw.moneyState.value.fares.nodes) === '[[1,1.6],[100,11.5]]', undefined, { timeout: 30000 });

  // the Money tab
  await page.click('button.tab[aria-label="Money"]');
  const shade = async (s) => {
    await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s);
    await page.waitForTimeout(300);
  };
  const shot = async (name, sel, extra = 0) => {
    const g = await page.$(sel);
    await g.scrollIntoViewIfNeeded();
    await page.waitForTimeout(200);
    const box = await g.boundingBox();
    await page.screenshot({ path: `${OUT}${name}.png`, clip: { x: box.x - 8, y: box.y - 30, width: box.width + 16, height: box.height + 40 + extra } });
  };
  const graph = 'svg.fare-graph';
  for (const s of ['light', 'dark']) { await shade(s); await shot(`fare-default-${s}`, graph, 110); }
  await shade('light');

  // drag the second node up: nothing is sent until the pointer is released
  const knobs = await page.$$(`${graph} .fg-knob`);
  report('two nodes on the graph', knobs.length === 2, `(${knobs.length})`);
  const before = await nodesNow();
  const k = await knobs[1].boundingBox();
  const kx = k.x + k.width / 2, ky = k.y + k.height / 2;
  await page.mouse.move(kx, ky);
  await page.mouse.down();
  await page.mouse.move(kx + 10, ky - 15, { steps: 4 });
  await page.mouse.move(kx + 20, ky - 30, { steps: 4 });
  await page.waitForTimeout(300);
  const mid = await nodesNow();
  for (const s of ['light', 'dark']) { await shade(s); await shot(`fare-drag-${s}`, graph, 110); }
  await shade('light');
  await page.mouse.up();
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== before, before, { timeout: 30000 });
  const after = JSON.parse(await nodesNow());
  report('a drag sends the fare once, on release', mid === before && after[1][1] > 11.5 && after[1][0] > 100 && after[0][1] === 1.6, `(before ${before}, mid-drag ${mid}, after ${JSON.stringify(after)})`);

  // T-1130: Ctrl+Z takes the fare change back while it is the last edit, Ctrl+Y puts it back
  const undoBtn = `button[aria-label="Undo the fare change"]`;
  const enabled = async () => !(await page.$eval(undoBtn, (b) => b.disabled));
  const wasOn = await enabled();
  await page.keyboard.press('Control+z');
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) === before, before, { timeout: 30000 }).catch(() => {});
  const undone = await nodesNow();
  const offAfter = !(await enabled());
  await page.keyboard.press('Control+y');
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== before, before, { timeout: 30000 }).catch(() => {});
  const redone = await nodesNow();
  report('Ctrl+Z and Ctrl+Y undo and redo a fare change', wasOn && undone === before && offAfter && redone === JSON.stringify(after), `(button on ${wasOn}, undone ${undone}, button off after ${offAfter}, redone ${redone})`);
  await page.click(undoBtn);
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) === before, before, { timeout: 30000 }).catch(() => {});
  report('the undo button takes it back', (await nodesNow()) === before, `(${await nodesNow()})`);
  // redo it so the zoom below has a far node to put past the plot
  await page.keyboard.press('Control+y');
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== before, before, { timeout: 30000 }).catch(() => {});

  // zoom in with the wheel until the second node is past the plot: an arrow at the edge
  const gb = await (await page.$(graph)).boundingBox();
  await page.mouse.move(gb.x + gb.width / 2, gb.y + gb.height / 2);
  for (let i = 0; i < 6; i++) { await page.mouse.wheel(0, -150); await page.waitForTimeout(80); }
  await page.waitForTimeout(300);
  const off = await page.$$(`${graph} .fg-off`);
  report('zoomed in, the far node is an arrow', off.length === 1 && (await page.$$(`${graph} .fg-knob`)).length === 1, `(${off.length} arrows)`);
  for (const s of ['light', 'dark']) { await shade(s); await shot(`fare-zoomed-${s}`, graph, 110); }
  await shade('light');
  if (off.length) {
    const ab = await off[0].boundingBox();
    await page.mouse.click(ab.x + ab.width / 2, ab.y + ab.height / 2);
    await page.waitForTimeout(300);
    report('a click on the arrow zooms out to the node', (await page.$$(`${graph} .fg-off`)).length === 0, '');
  }
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  await setNodes([[1, 1.6], [100, 11.5]]);
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
