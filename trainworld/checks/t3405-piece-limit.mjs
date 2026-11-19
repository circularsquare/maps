// T-3405: a speed limit on a piece of track. On a new game in New York, a straight line of three stations
// (A, M halfway, B) built with the edit API, so the track is two pieces of about 5 km. The M-B piece's panel shows Top
// speed 160 km/h; the selection bar's Speed limit (T-3746: the piece clicked is the pick) opens a stepper at 160
// storing no limit (T-3525), clicked down to 60, and the panel then shows the limit.
// Checks M to B takes what a stop-to-stop run at 60 instead of 160 adds and the round trip grows, the stop-time hover
// names the track's limit, the cash is unchanged, a save reloads with the limit and the same round trip, and turning
// Limit off restores the first round trip exactly. Screenshots of the panel in artifacts/trainworld/t3405_*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3405-piece-limit.mjs [dev url]
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6164/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const C = [-73.99, 40.74], HALF = 7000;

const chrome = await launchChrome({ prefix: 'tw-t3405-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const stats = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num);
  if (!l) return null;
  const st = w.lineStats[l.id];
  return { id: l.id, status: st.status, round: st.roundTripByLevel[1], mb: st.fromStartS[2] - st.fromStartS[1] - st.dwellS, mbM: st.stopS[0][2] - st.stopS[0][1], whys: st.stretches.map((s) => s.why), limits: st.stretches.map((s) => Math.round(s.limit)) };
}, num);
const limited = () => page.evaluate(() => window.tw.world.value.edges.filter((e) => e.limit).map((e) => [e.id, e.limit]));
const cash = () => page.evaluate(() => window.tw.world.value.save.cash);
const selectEdge = async (edge) => { await page.evaluate((edge) => { window.tw.selection.value = { kind: 'track', edge }; }, edge); await page.waitForTimeout(1000); };
const panel = () => page.locator('.top-speed').first();
// T-3746: the limit is set in the selection bar of the piece clicked; the panel shows it
const bar = () => page.locator('.blueprint-bar .bp-limit').first();
const shot = async (file) => {
  const p = await page.$('#inspector, .inspector');
  await page.waitForTimeout(600);
  await (p ?? page).screenshot({ path: join(ARTIFACTS, file) });
};
const waitFor = async (fn, what) => {
  for (let k = 0; k < 60; k++) {
    if (await fn()) return true;
    await page.waitForTimeout(500);
  }
  check(false, `waited for ${what}`);
  return false;
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
    const ends = [near(a), near(b)];
    for (const n of ends) { await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' }); await pause(200); }
    const e0 = tw.world.value.edges[0].id;
    const mid = await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: e0, x: o.x, y: o.y }, platform: 200, name: '' });
    await pause(400);
    const m = near(o);
    await cl.edit({ op: 'constructAll' });
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [ends[0], m, ends[1]], name: 'Limited', colour: '#0a7d3b' });
    await pause(500);
    const w = tw.world.value;
    // the piece from M to B
    const mb = w.edges.find((e) => (e.a === m && e.b === ends[1]) || (e.b === m && e.a === ends[1]));
    return { ok: r.ok && mid.ok && line.ok, line: line.line, mb: mb?.id, pieces: w.edges.length };
  }, [C, HALF]);
  check(built.ok && built.mb !== undefined && built.pieces === 2, 'the line over two pieces is built with the edit API', JSON.stringify(built));
  await page.evaluate((num) => window.tw.client.edit({ op: 'setTrainCount', line: num, trains: [6, 4, 2] }), built.line);
  await page.waitForTimeout(1500);
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 11.5 }); }, C);
  const s0 = await stats(built.line);
  const cash0 = await cash();
  await selectEdge(built.mb);
  const row = panel();
  await row.waitFor({ timeout: 20000 });
  const text0 = (await row.innerText()).replace(/\s+/g, ' ');
  check(/Top speed 160 km\/h/.test(text0) && !/Speed limit/.test(text0), 'the panel shows Top speed 160 km/h and no limit', text0);
  await bar().waitFor({ timeout: 10000 });
  check((await bar().locator('.stepper').count()) === 0, 'no stepper in the bar while Speed limit is off');
  await shot('t3405_off_light.png');

  await bar().locator('button', { hasText: /^Speed limit$/ }).click();
  await page.waitForTimeout(600);
  // T-3525: a limit at the track's top is none, so Speed limit on opens the stepper at 160 and stores nothing
  check((await limited()).length === 0, 'Speed limit on stores no limit at the track\'s 160', JSON.stringify(await limited()));
  check((await bar().locator('.stepper').count()) === 1 && /160 km\/h/.test(await bar().locator('.stepper').innerText()), 'a stepper at 160 appears in the bar');
  const minus = bar().locator('.stepper button').first();
  for (let i = 0; i < 10; i++) { await minus.click(); await page.waitForTimeout(350); }
  await waitFor(async () => (await limited())[0]?.[1] === 60, 'the limit to reach 60');
  await minus.click();
  await page.waitForTimeout(600);
  await minus.click();
  await page.waitForTimeout(600);
  await minus.click();
  await waitFor(async () => (await limited())[0]?.[1] === 40, 'the limit to stop at 40');
  check((await limited())[0]?.[1] === 40, 'the stepper stops at 40 km/h', JSON.stringify(await limited()));
  const plus = bar().locator('.stepper button').last();
  await plus.click(); await page.waitForTimeout(600); await plus.click();
  await waitFor(async () => (await limited())[0]?.[1] === 60, 'the limit back at 60');
  await waitFor(async () => (await stats(built.line)).round > s0.round + 100, 'the round trip to grow');
  const s1 = await stats(built.line);
  // M to B, stop to stop: speeding up at 1.0 m/s2 to 60 km/h and 0.5 above, braking at 0.9, cruising between
  const runS = (len, v) => { const v1 = Math.min(v, 60 / 3.6), d = v1 * v1 / 2 + (v * v - v1 * v1) / (2 * 0.5) + v * v / (2 * 0.9); return v1 + (v - v1) / 0.5 + v / 0.9 + (len - d) / v; };
  const len = s0.mbM, want = runS(len, 60 / 3.6) - runS(len, 160 / 3.6);
  check(Math.abs(s1.mb - s0.mb - want) < 2, 'M to B takes what a stop-to-stop run at 60 instead of 160 adds', `${s0.mb.toFixed(1)} s -> ${s1.mb.toFixed(1)} s, +${(s1.mb - s0.mb).toFixed(1)}; ${want.toFixed(1)} by the train's motion over ${len.toFixed(0)} m`);
  check(s1.round > s0.round + want, 'the round trip grows', `${s0.round.toFixed(1)} s -> ${s1.round.toFixed(1)} s`);
  check(s1.whys[1] === 4 && s1.limits[1] === 60 && s1.whys[0] !== 4, 'the stop-time hover names the track\'s limit on M-B only', JSON.stringify(s1));
  check(Math.abs((await cash()) - cash0) < 1e-6, 'setting the limit cost nothing', `${cash0} -> ${await cash()}`);
  const text1 = (await row.innerText()).replace(/\s+/g, ' ');
  check(/Top speed 160 km\/h ?Speed limit 60 km\/h/.test(text1), 'the panel shows Top speed 160 and the 60 km/h limit', text1);
  await shot('t3405_on_light.png');

  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const back = await limited();
  const s2 = await stats(built.line);
  check(back.length === 1 && back[0][1] === 60 && Math.abs(s2.round - s1.round) < 0.01, 'a saved game loads with the 60 km/h piece and the same round trip', `${JSON.stringify(back)}, ${s2.round}`);

  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await selectEdge(back[0][0]);
  await panel().waitFor({ timeout: 20000 });
  await shot('t3405_on_dark.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
  await bar().locator('button', { hasText: /^Speed limit$/ }).click();
  await waitFor(async () => (await limited()).length === 0, 'the limit to turn off');
  await waitFor(async () => Math.abs((await stats(built.line)).round - s0.round) < 0.01, 'the first round trip');
  const off = await stats(built.line);
  check(Math.abs(off.round - s0.round) < 0.01, 'turning Limit off restores the first round trip exactly', `${s0.round} -> ${off.round}`);
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3405_died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3405 FAILED: ${fail}` : 'T-3405 PASSED');
process.exit(fail ? 1 : 0);
