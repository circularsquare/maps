// T-3201: one train alone on single track that no other line uses shows no "Single track ... is full" alert and no red
// headway; the lone train's cap stays at least 2 (T-3165) so one held train can still show; with two trains on the same
// stretch the limit is back. A new game, a two-station single-track line.
// Screenshots artifacts/trainworld/t3201-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3201-lone-train.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6113/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3201-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const lng = -71.06, lat = 42.36, dLng = 4 / (111.32 * Math.cos((lat * Math.PI) / 180));
    const xy = [[lng, lat], [lng + dLng, lat]].map(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: true });
    await pause(600);
    for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(300); }
    await c.edit({ op: 'constructAll' });
    await pause(600);
    const r = await c.edit({ op: 'addLine', stops: xy.map(near), name: 'Lone', colour: '#0a7d3b' });
    if (r?.line === undefined) return { ok: false };
    await c.edit({ op: 'setTrainCount', line: r.line, trains: [1, 1, 1] });
    await pause(1500);
    const l = tw.world.value.save.lines.find((x) => x.num === r.line);
    tw.selection.value = { kind: 'line', line: l.id };
    return { ok: true, num: r.line, id: l.id };
  });
  check(made.ok, 'a single-track line is built');
  await wait(2500);
  const info = () => page.evaluate(async ([num, id]) => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const { trainCap } = await import('/src/game/world.ts');
    const st = window.tw.world.value.lineStats[id];
    const d = await queryClock({ what: 'lineHeld', line: num });
    const s = d[21];
    return { trains: st.trainsNeeded.slice(), wait: d[0], limitS: s, trip: st.roundTripByLevel[0], cap: trainCap(st, st.roundTripByLevel[0], s > 0 ? s : null),
      red: [...document.querySelectorAll('table td .bad')].length, text: document.body.innerText.match(/[^\n]*(is full)[^\n]*/g) ?? [] };
  }, [made.num, made.id]);
  const one = await info();
  console.log('one train', JSON.stringify(one));
  check(one.trains[0] === 1, 'one train', JSON.stringify(one.trains));
  check(!(one.wait > 0), 'no train is held', `${one.wait}`);
  check(one.text.length === 0, 'no "is full" alert in the panel', JSON.stringify(one.text));
  check(one.red === 0, 'no red headway');
  check(one.cap >= 2, 'the lone train\'s cap is at least 2', `${one.cap}`);
  await page.screenshot({ path: join(ARTIFACTS, 't3201-lone.png') });
  // two trains on the same stretch: the margin counts again
  await page.evaluate(async (num) => { await window.tw.client.edit({ op: 'setTrainCount', line: num, trains: [4, 4, 4] }); }, made.num);
  await wait(3000);
  const four = await info();
  console.log('four trains', JSON.stringify(four));
  check(four.limitS > 0 && four.trip / four.limitS < 4, 'with four trains the single track limits the line', `limit ${Math.round(four.limitS)} s, trip ${Math.round(four.trip)} s`);
  await page.screenshot({ path: join(ARTIFACTS, 't3201-four.png') });
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
