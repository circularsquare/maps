// T-3297: the running side a player switches (T-3367: the new game sheet has no control; the side each start gets
// is checked by t3367-side-default.mjs).
//   1. The new game sheet shows no side control.
//   2. A sandbox keeping right with a flyover branch and a line over it; Settings' Trains keep Left asks, then
//      switches: the game keeps left (`rightHand`, the drawn `keep`), the flyover's legs run the other way, neither
//      junction has a flat badge, the line runs, the list opens with its rows (the legs and the line under Fixed,
//      nothing to fix), and a row's click shows the place; switching back puts the legs as they were.
//   3. A sandbox from the real Singapore network keeps Left whichever side its save keeps.
// Pictures in artifacts/trainworld/t3297-side-*.png.
// From the maps root (or the tree's), on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6169):
//   node trainworld/checks/t3297-side-switch.mjs [dev url]
// About 3 minutes.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6169/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3297s-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3297-side-${name}.png`) });

try {
  // ---- 1. the new game sheet
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.new-game, .start-new', { timeout: 120000 });
  if (!(await page.$('.new-game'))) await page.click('button.start-new');
  // T-3367: the sheet has no side control; the sides each start gets are checked by t3367-side-default.mjs
  check(!(await page.$('.new-game .side-switch')), 'the new game sheet has no running side control');

  // ---- 2. a game with a flyover, switched
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const res = [];
    const go = async (op, ms = 400) => { const r = await cl.edit(op); res.push([op.op, r]); await pause(ms); return r; };
    const W = P(-3000, 0), E = P(3000, 0), S = P(-1200, 0), B = P(1800, 2200);
    await go({ op: 'route', from: { kind: 'free', ...W, level: 0 }, to: { kind: 'free', ...E, level: 0 }, pis: [], single: false });
    const main = tw.world.value.edges.find((e) => e.tracks === 2);
    await go({ op: 'addStation', at: { kind: 'edge', edge: main.id, ...S }, platform: 200, name: '' });
    await go({ op: 'route', from: { kind: 'node', node: near(S) }, to: { kind: 'free', ...B, level: 0 }, pis: [], single: false, ramp: true, design: 160, flyover: 20 }, 600);
    for (const p of [W, E, B]) await go({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }, 200);
    await go({ op: 'constructAll' }, 500);
    const line = await go({ op: 'addLine', stops: [near(W), near(B)], name: 'Branch', colour: '#f28c00' }, 800);
    const w = tw.world.value;
    const legs = w.edges.filter((e) => e.tracks === 1).map((e) => ({ id: e.id, set: e.setWay, way: e.oneWay }));
    return { refused: res.filter(([, r]) => !r.ok).map(([op, r]) => `${op}: ${JSON.stringify(r.issues ?? r).slice(0, 80)}`), line: line.line, legs, at: tw.packToLngLat(S.x, S.y) };
  }, [RC]);
  check(!built.refused.length && built.legs.length >= 2, 'a sandbox keeping right with a flyover branch and a line', built.refused.join(' | ') || `${built.legs.length} legs`);
  const state = () => page.evaluate((line) => {
    const tw = window.tw, w = tw.world.value;
    const st = w.lineStats[String(line)];
    return {
      keep: tw.renderer.net?.keep ?? 1,
      legs: w.edges.filter((e) => e.tracks === 1).map((e) => ({ id: e.id, set: e.setWay })),
      flat: tw.client.lastNodes.filter((n) => n.crosses).length,
      status: st?.status,
    };
  }, built.line);
  const before = await state();
  check(before.keep === 1 && before.flat === 0 && before.status !== 'broken', 'keeping right, no flat badge, the line runs', JSON.stringify(before).slice(0, 160));
  await page.click('button.tab[aria-label="Settings"]');
  await page.waitForSelector('.side-switch');
  await page.click('.side-switch .btn:not(.on)');
  await page.waitForSelector('dialog .btn', { timeout: 10000 });
  const askText = await page.evaluate(() => document.querySelector('dialog[open]')?.textContent ?? '');
  check(/Switch trains, trams and buses to the left\?/.test(askText), 'the question asks first', askText.slice(0, 120));
  await shot('ask');
  await page.getByRole('button', { name: 'Switch side' }).click();
  await page.waitForSelector('dialog.side-sites-panel', { timeout: 30000 });
  await page.waitForTimeout(1500);
  await shot('list');
  const list = await page.evaluate(() => {
    const d = document.querySelector('dialog.side-sites-panel');
    return { title: d?.querySelector('h2')?.textContent ?? '', rows: [...(d?.querySelectorAll('.side-sites li') ?? [])].map((li) => li.textContent.trim()), nothing: !!d?.querySelector('p.muted') };
  });
  check(list.title === 'Switched to the left', 'the list opens', list.title);
  check(list.nothing && list.rows.filter((r) => /One-way track turned around/.test(r)).length === built.legs.length && list.rows.some((r) => /Branch rerouted/.test(r)),
    'nothing to fix; the legs turned around and the line rerouted under Fixed', list.rows.join(' | '));
  const after = await state();
  check(after.keep === -1 && after.flat === 0 && after.status !== 'broken', 'keeping left, still no flat badge, the line runs', JSON.stringify(after).slice(0, 160));
  check(after.legs.every((l) => { const b = before.legs.find((x) => x.id === l.id); return b && l.set === 3 - b.set; }), 'every leg runs the other way, ids kept', JSON.stringify([before.legs, after.legs]));
  // a row's click shows the place and closes the list
  const z0 = await page.evaluate(() => window.tw.map.getCenter());
  await page.click('dialog.side-sites-panel .side-sites li .btn');
  await page.waitForTimeout(800);
  const moved = await page.evaluate((z) => { const c = window.tw.map.getCenter(); return Math.hypot(c.lng - z.lng, c.lat - z.lat); }, z0);
  check(!(await page.$('dialog.side-sites-panel')) && moved >= 0, 'a row\'s click closes the list and shows the place');
  await page.evaluate((at) => { const c = Array.isArray(at) ? at : [at.lng, at.lat]; window.tw.map.jumpTo({ center: c, zoom: 16.4 }); }, built.at);
  await page.waitForTimeout(1500);
  await shot('map-left');
  check(await page.evaluate(() => !!document.querySelector('[data-side-sites]')), 'Settings offers the list again');
  // and back
  await page.click('.side-switch .btn:not(.on)');
  await page.waitForSelector('dialog .btn', { timeout: 10000 });
  await page.getByRole('button', { name: 'Switch side' }).click();
  await page.waitForSelector('dialog.side-sites-panel', { timeout: 30000 });
  await page.waitForTimeout(1000);
  const back = await state();
  check(back.keep === 1 && back.flat === 0 && back.legs.every((l) => before.legs.find((x) => x.id === l.id)?.set === l.set), 'switched back: right, the legs as they were', JSON.stringify(back).slice(0, 160));

  // ---- 3. a real network keeps the sheet's side: Singapore, Left (a save shipped right-hand is switched as it opens)
  await page.goto(`${BASE}?debug=1&v=${Date.now()}`);
  await page.waitForSelector('.new-game, .start-new', { timeout: 120000 });
  if (!(await page.$('.new-game'))) await page.click('button.start-new');
  await page.click('.new-game button[data-mode="sandbox"]');
  await page.selectOption('.new-game-real select', 'sin');
  await page.waitForTimeout(150);
  await page.click('.new-game-start');
  await page.waitForFunction(() => window.tw?.world?.value?.edges?.length > 0 && window.tw.renderer?.net, undefined, { timeout: 240000 });
  await page.waitForTimeout(2000);
  const sin = await page.evaluate(() => {
    const tw = window.tw, w = tw.world.value;
    return { keep: tw.renderer.net.keep ?? 1, lines: w.save.lines.length, broken: Object.values(w.lineStats).filter((s) => s.status === 'broken').length };
  });
  check(sin.keep === -1 && sin.lines > 0 && sin.broken === 0, 'the real Singapore network starts keeping left, whichever side its save keeps', JSON.stringify(sin));
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  console.log('FAIL', e?.stack ?? e);
  fail++;
} finally {
  await chrome.close?.();
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
