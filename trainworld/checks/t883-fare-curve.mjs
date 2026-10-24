// T-883, T-1112, T-1143: the fare curve reaches demand and income, and the Money tab's three-node
// control works. On the real save (a ?new=1 game, the weekday's final): the default fare (three
// nodes on $1.50 + $0.10 a km); the same line through other nodes and an older save's two nodes
// (no change at all); twice the rate per km (fewer rail trips); a curve steeper past 20 km (fewer
// rail trips, demand's extra fare counted in income: the weekday's income is the base on each trip,
// the first stretch's rate on the passenger-km and demand's extra, and the extra is within what the
// steeper rate could add). Then in the Money tab: arrows on hover the ways a node drags; a drag
// moves one way only (the base fare up and down, the others by the drag's first pixels) and sends
// nothing until release; Ctrl+Z, Ctrl+Y and the undo button (T-1130); zoom keeps the line's slope
// on screen; zoomed in, a node past the plot is an arrow, and a click on it zooms out to it.
// Screenshots (default, hover, mid-drag, zoomed in and out), light and dark, go to
// artifacts/trainworld/t1143/.
//   node trainworld/checks/t883-fare-curve.mjs [url]   (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on failure. About five minutes.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6150/').replace(/\/?$/, '/');
const OUT = join(ARTIFACTS, 't1143');
/** the real save's rail trips a day at the default fare (the data of 2026-10-24) */
const DEFAULT_RAIL = Number(process.env.T883_RAIL ?? 12783191);
const DEFAULT = '[[0,1.5],[20,3.5],[100,11.5]]';

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
  const waitNodes = (want) => page.waitForFunction((want) => JSON.stringify(window.tw.moneyState.value.fares.nodes) === want, want, { timeout: 30000 }).catch(() => {});
  const fmt = (v) => Math.round(v).toLocaleString('en-US');
  /** the weekday's fare income a day against the base, rate and demand's extra on its riders, US$ */
  const money = () => page.evaluate(() => {
    const tw = window.tw, w = tw.world.value, m = tw.moneyState.value, dv = window.twDemand.view();
    const hours = [4, 6, 4, 4, 6];
    const income = tw.incomeByDay.value[0].perHour.reduce((a, v, p) => a + v * hours[p] * 1e6, 0);
    const boardings = w.save.lines.reduce((a, l) => a + (dv.lines.get(l.id)?.ridersPerDay ?? 0), 0);
    const perBoarding = boardings > 0 ? Math.min(1, dv.railTripsPerDay / boardings) : 1;
    let trips = 0, pkm = 0, extra = 0;
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id], dm = dv.lines.get(l.id), n = l.stops.length;
      if (!st || st.status !== 'running' || !dm || st.stopS[0]?.length !== n || st.stopS[1]?.length !== n) continue;
      const km = [];
      for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push((s[k + 1] - s[k]) / 1000);
      for (const seg of dm.loads) if (seg.length === km.length) pkm += seg.reduce((a, v, k) => a + v * km[k], 0);
      trips += dm.ridersPerDay * perBoarding;
      extra += (dm.fareX ?? []).reduce((a, b) => a + b, 0);
    }
    const f = m.fares;
    return { income, expected: (f.base * trips + f.perKm * pkm + extra) * m.economy, extra: extra * m.economy, pkm, economy: m.economy };
  });

  const a = await final(-1);
  const an = await nodesNow();
  const am = await money();
  report('default fare', Math.round(a.rail) === DEFAULT_RAIL && an === DEFAULT && am.extra === 0, `(${fmt(a.rail)} rail trips a day; want ${fmt(DEFAULT_RAIL)}; nodes ${an}; extra ${am.extra})`);

  // the same fares give the demand nothing new to solve, so no new final may come
  await setNodes([[0, 1.5], [50, 6.5], [200, 21.5]]);
  await page.waitForTimeout(15000);
  const b = await final(-1);
  report('the same line through other nodes', Math.round(b.rail) === Math.round(a.rail), `(${fmt(b.rail)})`);

  await setNodes([[1, 1.6], [100, 11.5]]);
  await waitNodes(DEFAULT);
  await page.waitForTimeout(15000);
  const c = await final(-1);
  const cn = await nodesNow();
  report('an older two-node curve becomes three on its line', cn === DEFAULT && Math.round(c.rail) === Math.round(a.rail), `(nodes ${cn}, ${fmt(c.rail)})`);

  await setNodes([[0, 1.5], [20, 5.5], [100, 21.5]]);
  const d = await final(c.version);
  const dd = (d.rail - a.rail) / a.rail;
  report('twice the rate per km', dd < 0, `(${fmt(d.rail)}, ${(100 * dd).toFixed(2)}%)`);

  // steeper past 20 km: fewer riders, and the extra fare demand charged is in the income
  await setNodes([[0, 1.5], [20, 3.5], [100, 27.5]]);
  const e = await final(d.version);
  const de = (e.rail - a.rail) / a.rail;
  report('steeper past 20 km', e.rail < a.rail && e.rail > d.rail, `(${fmt(e.rail)}, ${(100 * de).toFixed(2)}%)`);
  const em = await money();
  const gap = Math.abs(em.income - em.expected) / em.expected;
  report('income is the base, the rate and demand\'s extra', gap < 1e-6 && em.extra > 0 && em.extra <= 0.2 * em.pkm * em.economy * 1.0001,
    `(income $${fmt(em.income)} a day, expected $${fmt(em.expected)}, extra $${fmt(em.extra)}, ${(100 * em.extra / em.income).toFixed(1)}% of it)`);

  await setNodes(JSON.parse(DEFAULT));
  await waitNodes(DEFAULT);

  // the Money tab
  await page.click('button.tab[aria-label="Money"]');
  const shade = async (s) => {
    await page.evaluate((s) => { document.documentElement.dataset.shade = s; window.tw.shade.value = s; }, s);
    await page.waitForTimeout(300);
  };
  const graph = 'svg.fare-graph';
  const shot = async (name, extra = 0) => {
    const g = await page.$(graph);
    await g.scrollIntoViewIfNeeded();
    await page.waitForTimeout(200);
    const box = await g.boundingBox();
    await page.screenshot({ path: join(OUT, `${name}.png`), clip: { x: box.x - 8, y: box.y - 30, width: box.width + 16, height: box.height + 40 + extra } });
  };
  const both = async (name, extra = 0) => { for (const s of ['light', 'dark']) { await shade(s); await shot(`${name}-${s}`, extra); } await shade('light'); };
  await both('fare-default', 150);

  const knobs = await page.$$(`${graph} .fg-knob`);
  report('three nodes on the graph', knobs.length === 3, `(${knobs.length})`);
  const centre = async (i) => { const k = (await page.$$(`${graph} .fg-knob`))[i]; const bb = await k.boundingBox(); return [bb.x + bb.width / 2, bb.y + bb.height / 2]; };
  const hints = () => page.$$eval(`${graph} .fg-hint`, (h) => h.length);

  // hovering a node shows the ways it drags: the base fare two arrows, the others four
  const [b0x, b0y] = await centre(0);
  await page.mouse.move(b0x, b0y);
  await page.waitForTimeout(150);
  const baseHints = await hints();
  const [k1x, k1y] = await centre(1);
  await page.mouse.move(k1x, k1y);
  await page.waitForTimeout(150);
  const nodeHints = await hints();
  report('hover arrows: up and down on the base fare, all four on the others', baseHints === 2 && nodeHints === 4, `(${baseHints}, ${nodeHints})`);
  await both('fare-hover', 150);

  // drag the second node mostly sideways: it moves in distance only, and nothing is sent until release
  await page.mouse.move(k1x, k1y);
  await page.mouse.down();
  await page.mouse.move(k1x + 6, k1y - 2, { steps: 2 });
  await page.mouse.move(k1x + 40, k1y - 14, { steps: 6 });
  await page.waitForTimeout(300);
  const mid = await nodesNow();
  const midHints = await hints();
  await both('fare-drag', 150);
  await page.mouse.up();
  await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== before, DEFAULT, { timeout: 30000 }).catch(() => {});
  const after = JSON.parse(await nodesNow());
  report('a sideways drag moves the distance only, sent once on release', mid === DEFAULT && midHints === 2 && after[1][0] > 20 && after[1][1] === 3.5 && after[0][1] === 1.5 && after[2][0] === 100,
    `(mid-drag ${mid}, ${midHints} arrows, after ${JSON.stringify(after)})`);

  // T-1130: Ctrl+Z takes the fare change back while it is the last edit, Ctrl+Y puts it back
  const undoBtn = `button[aria-label="Undo the fare change"]`;
  const enabled = async () => !(await page.$eval(undoBtn, (b) => b.disabled));
  const wasOn = await enabled();
  await page.keyboard.press('Control+z');
  await waitNodes(DEFAULT);
  const undone = await nodesNow();
  const offAfter = !(await enabled());
  await page.keyboard.press('Control+y');
  await waitNodes(JSON.stringify(after));
  const redone = await nodesNow();
  report('Ctrl+Z and Ctrl+Y undo and redo a fare change', wasOn && undone === DEFAULT && offAfter && redone === JSON.stringify(after), `(button on ${wasOn}, undone ${undone}, button off after ${offAfter}, redone ${redone})`);
  await page.click(undoBtn);
  await waitNodes(DEFAULT);
  report('the undo button takes it back', (await nodesNow()) === DEFAULT, `(${await nodesNow()})`);

  // the third node mostly up: fare only; the base fare sideways and up: fare only, still at 0 km
  const drag = async (i, dx, dy) => {
    const [x, y] = await centre(i);
    const before = await nodesNow();
    await page.mouse.move(x, y);
    await page.mouse.down();
    await page.mouse.move(x + dx / 8, y + dy / 8, { steps: 2 });
    await page.mouse.move(x + dx, y + dy, { steps: 6 });
    await page.mouse.up();
    await page.waitForFunction((before) => JSON.stringify(window.tw.moneyState.value.fares.nodes) !== before, before, { timeout: 30000 }).catch(() => {});
    return JSON.parse(await nodesNow());
  };
  const up = await drag(2, 6, -30);
  report('an upward drag moves the fare only', up[2][0] === 100 && up[2][1] > 11.5 && up[1][0] === 20, `(${JSON.stringify(up)})`);
  const baseMoved = await drag(0, 30, -5);
  report('the base fare stays at 0 km', baseMoved[0][0] === 0 && baseMoved[0][1] > 1.5 && baseMoved[0][1] <= baseMoved[1][1], `(${JSON.stringify(baseMoved)})`);
  await setNodes(JSON.parse(DEFAULT));
  await waitNodes(DEFAULT);
  await page.waitForTimeout(300);

  // zoom keeps the slope: the first stretch's slope on screen at the default view, out and in
  const slope = () => page.evaluate((g) => {
    const k = [...document.querySelectorAll(`${g} .fg-knob`)].map((c) => [+c.getAttribute('cx'), +c.getAttribute('cy')]);
    const span = (c) => (c[1][1] - c[0][1]) / (c[1][0] - c[0][0]);
    if (k.length >= 2) return span(k);
    const m = document.querySelector(`${g} .fg-curve`).getAttribute('d').match(/-?[\d.]+/g).map(Number);
    return span([[m[0], m[1]], [m[2], m[3]]]);
  }, graph);
  const gb = await (await page.$(graph)).boundingBox();
  await page.mouse.move(gb.x + gb.width / 2, gb.y + gb.height / 2);
  const s0 = await slope();
  for (let i = 0; i < 8; i++) { await page.mouse.wheel(0, 150); await page.waitForTimeout(80); }
  await page.waitForTimeout(300);
  const sOut = await slope();
  await both('fare-zoomed-out', 150);
  for (let i = 0; i < 22; i++) { await page.mouse.wheel(0, -150); await page.waitForTimeout(80); }
  await page.waitForTimeout(300);
  const sIn = await slope();
  const off = await page.$$(`${graph} .fg-off`);
  await both('fare-zoomed-in', 150);
  const same = (v) => Math.abs(v / s0 - 1) < 0.01;
  report('zoom keeps the slope on screen', same(sOut) && same(sIn), `(default ${s0.toFixed(4)}, out ${sOut.toFixed(4)}, in ${sIn.toFixed(4)})`);
  report('zoomed in, the far nodes are arrows', off.length === 2 && (await page.$$(`${graph} .fg-knob`)).length === 1, `(${off.length} arrows)`);
  if (off.length) {
    const before = off.length;
    const ab = await off[0].boundingBox();
    await page.mouse.click(ab.x + ab.width / 2, ab.y + ab.height / 2);
    await page.waitForTimeout(300);
    const left = (await page.$$(`${graph} .fg-off`)).length;
    report('a click on an arrow zooms out to its node', left < before, `(${before} arrows, then ${left})`);
  }
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
