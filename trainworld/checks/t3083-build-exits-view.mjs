// T-3083: starting to build closes an open line or station view. On a new game in New York, two stations
// and a line: with the line view open the Track tool (key 1) closes it; with a station view open the bus
// key (U) closes it; the Station tool, Delete tool and ferry key close the same; the kept cases stay open
// (Edit stops on a line, a city view under a build tool, Select with a line open).
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3083-build-exits-view.mjs [dev url]
import { collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6173/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3083-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: -1 });
    const at = (n) => ({ kind: 'node', node: n });
    const A = P(-1500, 0), B = P(1500, 0);
    await cl.edit({ op: 'route', from: free(A), to: free(B), pis: [], single: false });
    await pause(400);
    const nA = near(A), nB = near(B);
    for (const n of [nA, nB]) { await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' }); await pause(200); }
    await cl.edit({ op: 'constructAll' });
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nB], name: 'Exit check', colour: '#7b3fb3' });
    await pause(800);
    return { a: nA, line: line.line, ok: line.ok };
  }, C);
  check(built.ok, 'a line is built');
  if (!built.ok) throw new Error('no line');
  const lineId = String(built.line);

  const sel = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); return { kind: ui.selection.peek()?.kind ?? null, tool: ui.tool.peek() }; });
  const open = async (s) => {
    await page.keyboard.press('Escape');
    await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; });
    await page.evaluate((s) => window.tw.select(s), s);
    await page.waitForTimeout(150);
  };
  const lineSel = { kind: 'line', line: lineId };
  const stationId = await page.evaluate((n) => String(window.tw.world.value.save.stations.find((s) => s.num === n)?.id ?? n), built.a);
  const stSel = { kind: 'station', station: stationId };

  await open(lineSel);
  check((await sel()).kind === 'line', 'line view opens');
  await page.keyboard.press('1');
  await page.waitForTimeout(150);
  let s = await sel();
  check(s.tool === 'track' && s.kind === null, 'Track tool closes the line view', JSON.stringify(s));

  await open(stSel);
  check((await sel()).kind === 'station', 'station view opens');
  await page.keyboard.press('u');
  await page.waitForTimeout(300);
  s = await sel();
  check(s.tool === 'bus' && s.kind === null, 'bus key closes the station view', JSON.stringify(s));
  await page.keyboard.press('Escape');

  await open(stSel);
  await page.keyboard.press('2');
  await page.waitForTimeout(150);
  s = await sel();
  check(s.tool === 'station' && s.kind === null, 'Station tool closes the station view', JSON.stringify(s));

  await open(lineSel);
  await page.keyboard.press('o');
  await page.waitForTimeout(300);
  s = await sel();
  check(s.tool === 'ferry' && s.kind === null, 'ferry key closes the line view', JSON.stringify(s));
  await page.keyboard.press('Escape');

  await open(lineSel);
  await page.keyboard.press('3');
  await page.waitForTimeout(150);
  s = await sel();
  check(s.tool === 'delete' && s.kind === null, 'Delete tool closes the line view', JSON.stringify(s));

  await open(lineSel);
  await page.locator('.tool-bus').first().click().catch(() => {});
  await page.waitForTimeout(300);
  s = await sel();
  check(s.tool === 'bus' && s.kind === null, 'Build tab Bus button closes the line view', JSON.stringify(s));
  await page.keyboard.press('Escape');

  // kept
  await open(lineSel);
  await page.keyboard.press('s');
  await page.waitForTimeout(200);
  s = await sel();
  check(s.tool === 'line' && s.kind === 'line', 'Edit stops keeps the line view', JSON.stringify(s));
  await page.keyboard.press('Escape');

  await open(lineSel);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'select'; });
  s = await sel();
  check(s.tool === 'select' && s.kind === 'line', 'Select tool keeps the line view', JSON.stringify(s));

  await open({ kind: 'world' });
  await page.keyboard.press('1');
  await page.waitForTimeout(150);
  s = await sel();
  check(s.tool === 'track' && s.kind === 'world', 'a non-network view (world) stays under a build tool', JSON.stringify(s));

  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('FAIL exception', e.message);
  fail++;
} finally {
  await chrome.close?.();
}
console.log(fail ? `FAIL ${fail}` : 'PASS all');
process.exit(fail ? 1 : 0);
