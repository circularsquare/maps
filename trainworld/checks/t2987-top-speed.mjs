// T-2987: a line's speed limit. On a new game in New York, a straight 14 km line of two stations built with the edit
// API. In the line panel the Top speed row shows what its trains reach (160 km/h); the Limit button beside it adds a
// Speed limit row, whose stepper is clicked down to 120. Checks the round trip grows, the top speed row follows,
// the drawn trains never go faster than the limit (their positions probed from the renderer), a save that reloads
// with the limit, and that turning Limit off restores the first round trip. Screenshots of the panel light and dark
// in artifacts/trainworld/t2987-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t2987-top-speed.mjs [dev url]
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6470/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const C = [-73.99, 40.74], HALF = 7000;

const chrome = await launchChrome({ prefix: 'tw-t2987-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const stats = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num);
  if (!l) return null;
  const st = w.lineStats[l.id];
  return { id: l.id, status: st.status, limit: st.limit, round: st.roundTripByLevel[1], reached: Math.max(...st.stretches.map((s) => s.reached)) };
}, num);
const select = async (id) => { await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, id); await page.waitForTimeout(1200); };
/** the fastest any drawn train goes over `n` probes `step` game seconds apart, km/h (the plane's metres are Mercator's: ground = x cos lat) */
const fastest = (n, step) => page.evaluate(([n, step, lat]) => {
  const rd = window.tw.renderer, t0 = window.tw.clock.now();
  const ps = Array.from({ length: n }, (_, k) => rd.probe(t0 + k * step));
  let best = 0;
  for (let i = 0; i < rd.tripCount; i++) {
    for (let k = 1; k < n; k++) {
      const [a, b] = [ps[k - 1], ps[k]];
      if (a[i * 2] >= 1e8 || b[i * 2] >= 1e8) continue;
      best = Math.max(best, (Math.hypot(b[i * 2] - a[i * 2], b[i * 2 + 1] - a[i * 2 + 1]) / step) * 3.6 * Math.cos((lat * Math.PI) / 180));
    }
  }
  return best;
}, [n, step, C[1]]);
const panel = () => page.locator('.top-speed').first();
const shot = async (file) => {
  await panel().scrollIntoViewIfNeeded().catch(() => {});
  const p = await page.$('#inspector, .inspector');
  await page.waitForTimeout(600);
  await (p ?? page).screenshot({ path: join(ARTIFACTS, file) });
};
const waitStats = async (num, pred, what) => {
  for (let k = 0; k < 60; k++) {
    const s = await stats(num);
    if (s && pred(s)) return;
    await page.waitForTimeout(500);
  }
  check(false, `waited for ${what}`);
};

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const built = await page.evaluate(async ([c, h]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const a = { x: o.x - h, y: o.y }, b = { x: o.x + h, y: o.y };
    const r = await cl.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: -1 }, to: { kind: 'free', x: b.x, y: b.y, level: -1 }, pis: [], single: false });
    await pause(400);
    const ns = [near(a), near(b)];
    for (const n of ns) { await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }); await pause(200); }
    await cl.edit({ op: 'constructAll' });
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: ns, name: 'Limited', colour: '#0a7d3b' });
    await pause(500);
    return { ok: r.ok && line.ok, line: line.line };
  }, [C, HALF]);
  check(built.ok, 'the straight line is built with the edit API');
  await edit({ op: 'setTrainCount', line: built.line, trains: [6, 4, 2] });
  await page.waitForTimeout(1000);
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 11.5 }); }, C);
  await select((await stats(built.line)).id);

  let s0 = await stats(built.line);
  check(s0.limit === 0 && Math.abs(s0.reached - 160) < 1, 'a new line has no limit and reaches 160 km/h', `limit ${s0.limit}, reached ${s0.reached}`);
  const row = panel();
  await row.waitFor({ timeout: 20000 });
  check(/Top speed/.test(await row.innerText()) && /160 km\/h/.test(await row.innerText()), 'the Top speed row shows 160 km/h', (await row.innerText()).replace(/\s+/g, ' '));
  check((await row.locator('.stepper').count()) === 0, 'no Speed limit row while Limit is off');
  await shot('t2987-off-light.png');

  // free-running speed of the drawn trains, then limited
  await page.evaluate(() => { const tw = window.tw; tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); tw.clock.setRunning(false); tw.overlay.invalidate(); });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  const free = await fastest(40, 15);
  check(free > 140 && free <= 161, 'drawn trains reach about 160 km/h unlimited', `${free.toFixed(1)} km/h`);

  await row.locator('button', { hasText: /^Limit$/ }).click();
  await waitStats(built.line, (s) => s.limit > 0, 'the limit to turn on');
  let s1 = await stats(built.line);
  check(s1.limit === 160, 'Limit on starts at the current top speed', `${s1.limit}`);
  check((await row.locator('.stepper').count()) === 1, 'a Speed limit row with a stepper appears');
  const minus = row.locator('.stepper button').first();
  for (let i = 0; i < 4; i++) { await minus.click(); await page.waitForTimeout(500); }
  await waitStats(built.line, (s) => s.limit === 120, 'the limit to reach 120');
  const s2 = await stats(built.line);
  check(s2.limit === 120, 'four steps down make it 120 km/h', `${s2.limit}`);
  check(s2.round > s0.round * 1.05, 'its round trip grows', `${s0.round.toFixed(0)} s -> ${s2.round.toFixed(0)} s`);
  check(Math.abs(s2.reached - 120) < 1, 'the Top speed row follows: 120', `${s2.reached}`);
  check(/120 km\/h/.test(await row.innerText()), 'the panel shows 120 km/h', (await row.innerText()).replace(/\s+/g, ' '));
  await page.evaluate(() => { const tw = window.tw; tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); tw.overlay.invalidate(); });
  await page.waitForTimeout(1500);
  const lim = await fastest(40, 15);
  check(lim > 100 && lim <= 121, 'drawn trains never go faster than 120 km/h', `${lim.toFixed(1)} km/h`);
  await shot('t2987-on-light.png');
  const before = await stats(built.line);
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const loaded = await stats(built.line);
  check(loaded?.limit === 120 && Math.abs(loaded.round - before.round) < 1, 'a saved game loads the line with its 120 km/h limit and round trip', `${loaded?.limit}, ${loaded?.round}`);

  // dark screenshot of the panel with the limit on, then Limit off restores the round trip
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await select(loaded.id);
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 11.5 }), C);
  await shot('t2987-on-dark.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
  await panel().locator('button', { hasText: /^Limit$/ }).click();
  await waitStats(built.line, (s) => s.limit === 0, 'the limit to turn off');
  const off = await stats(built.line);
  check(off.limit === 0 && Math.abs(off.round - s0.round) < 1, 'turning Limit off removes the limit and restores the first round trip', `${s0.round.toFixed(0)} s -> ${off.round.toFixed(0)} s`);
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't2987-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2987 FAILED: ${fail}` : 'T-2987 PASSED');
process.exit(fail ? 1 : 0);
