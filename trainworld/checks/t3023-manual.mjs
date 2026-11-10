// T-3023/T-3024: manual stop sequence ("Edit every stop in order"). On a new game in New York a stem
// A - B - J - C - D along x with a spur from the junction J to a dead-end station E, and a line A, B, C, D.
// In stop editing: the toggle writes the whole cycle (A B C D C B A, numbered rows, one way, no Direction
// row); the cross on the way back's C skips that visit only; the "+" on C then a click on E's marker puts
// E in after C (the spur leaves facing east, so it cannot follow B); no U-turn icon shows on the line's reversals; a list that is a plain out-and-back folds
// back through the toggle to the open line, and one undo takes the switch back. Times the add.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3023-manual.mjs [dev url]
// About a minute. Screenshots artifacts/trainworld/t3023-*.png.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6171/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3023-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const out = { when: new Date().toISOString() };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const built = await page.evaluate(async (c) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lv = -1;
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: lv });
    const at = (n) => ({ kind: 'node', node: n });
    const res = [];
    const go = async (from, to, ps = []) => { res.push(await cl.edit({ op: 'route', from, to, pis: ps, single: false })); await pause(300); };
    const A = P(-3000, 0), B = P(-1500, 0), J = P(-500, 0), Cc = P(1000, 0), D = P(2500, 0), E = P(-500, 1800);
    await go(free(A), free(B));
    const nA = near(A), nB = near(B);
    await go(at(nB), free(J));
    const nJ = near(J);
    await go(at(nJ), free(Cc));
    const nC = near(Cc);
    await go(at(nC), free(D));
    const nD = near(D);
    await go(at(nJ), free(E));
    const nE = near(E);
    for (const n of [nA, nB, nC, nD, nE]) { res.push(await cl.edit({ op: 'addStation', at: at(n), platform: 160, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [nA, nB, nC, nD], name: 'Manual check', colour: '#7b3fb3' });
    await pause(800);
    return { n: { A: nA, B: nB, C: nC, D: nD, E: nE }, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 140)) };
  }, C);
  check(built.ok && !built.refused.length, 'the layout is built with the edit API', built.refused.join(' | '));
  if (!built.ok) throw new Error('no line');
  const N = built.n, id = built.line;
  const info = () => page.evaluate((num) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
    return { stops: l.stops.map(Number), loop: st.loop, oneWay: st.oneWay, manual: st.manual, folds: st.manualFolds, status: st.status, round: Math.round(st.roundTripByLevel[0]), reversals: st.reversals };
  }, id);
  const s0 = await info();
  console.log('open line', JSON.stringify(s0));

  await page.evaluate(async ([num, c]) => {
    const tw = window.tw, ui = await import('/src/game/ui.ts');
    const l = tw.world.value.save.lines.find((x) => x.num === num);
    tw.selection.value = { kind: 'line', line: l.id };
    ui.tool.value = 'line';
    ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
    tw.map.jumpTo({ center: [c[0] + 0.004, c[1] + 0.005], zoom: 13.6 });
  }, [id, C]);
  await page.waitForTimeout(2500);

  const panel = () => page.evaluate(() => ({
    rows: [...document.querySelectorAll('.stops > li[data-stop]')].map((li) => ({ n: li.querySelector('.visit-n')?.textContent ?? '', name: li.querySelector('.stop-name')?.textContent ?? '', cross: !!li.querySelector('button.danger'), plus: !!li.querySelector('button.btn.icon.mini:not(.danger)') })),
    adding: [...document.querySelectorAll('.stops > li.adding')].map((li) => li.textContent),
    direction: [...document.querySelectorAll('.rows > li > .grow')].some((e) => e.textContent === 'Direction'),
    revIcons: document.querySelectorAll('.stops .rev-stop').length,
    mapIcons: [...document.querySelectorAll('.rev-mark')].filter((e) => e.style.display !== 'none').length,
    toggle: (() => { const b = document.querySelector('.manual-row button'); return b ? { text: b.textContent, on: b.classList.contains('on'), disabled: b.disabled } : null; })(),
  }));

  // the toggle
  const p0 = await panel();
  check(p0.toggle && p0.toggle.text === 'Exact edits' && !p0.toggle.on, 'the toggle is at the top of the list, off', JSON.stringify(p0.toggle));
  await page.locator('.manual-row button').click();
  await page.waitForTimeout(1500);
  const s1 = await info();
  const want = [N.A, N.B, N.C, N.D, N.C, N.B, N.A];
  check(JSON.stringify(s1.stops) === JSON.stringify(want) && s1.manual && s1.oneWay, 'the toggle writes the whole cycle, one way', JSON.stringify(s1));
  check(Math.abs(s1.round - s0.round) <= 5, 'the lap is the old round trip', `${s0.round} -> ${s1.round}`);
  const p1 = await panel();
  check(p1.rows.length === 7 && p1.rows.map((r) => r.n).join() === '1,2,3,4,5,6,7', 'the list shows every visit, numbered', p1.rows.map((r) => `${r.n} ${r.name}`).join(' / '));
  check(!p1.direction, 'the Direction row is hidden');
  check(p1.toggle?.on && !p1.toggle.disabled, 'the toggle is on and can be switched back (a plain out-and-back)', JSON.stringify(p1.toggle));
  check(!p1.rows[0].cross && !p1.rows[6].cross && !p1.rows[3].cross && p1.rows[4].cross, 'no cross on the ends or the turning stop, one on the way back', p1.rows.map((r) => (r.cross ? 'x' : '-')).join(''));
  await page.screenshot({ path: join(ARTIFACTS, 't3023-manual.png') });

  // a visit out: C on the way back
  await page.locator('.stops > li[data-stop="4"] button.danger').click();
  await page.waitForTimeout(1500);
  const s2 = await info();
  check(JSON.stringify(s2.stops) === JSON.stringify([N.A, N.B, N.C, N.D, N.B, N.A]) && s2.manual, 'the cross takes that one visit out', s2.stops.join());
  const p2 = await panel();
  check(p2.toggle?.on && !p2.toggle.disabled, 'the toggle can always switch back', JSON.stringify(p2.toggle));
  // T-3079: a list that does not fold goes back to a normal line with the same stops; undo returns to exact
  await page.locator('.manual-row button').click();
  await page.waitForTimeout(1500);
  const sOff = await info();
  check(!sOff.manual && JSON.stringify(sOff.stops) === JSON.stringify(s2.stops), 'switching back keeps the stops as a normal line', JSON.stringify(sOff));
  await edit({ op: 'undo' });
  await page.waitForTimeout(1200);
  check((await info()).manual, 'undo puts exact edits back');

  // a visit in: + on C, then a click on E's marker
  const markerAt = (n) => page.evaluate((n) => {
    const tw = window.tw, node = tw.world.value.nodes.get(String(n)) ?? [...tw.world.value.nodes.values()].find((x) => Number(x.id) === n);
    const [lng, lat] = tw.packToLngLat(node.x, node.y);
    const p = tw.map.project([lng, lat]);
    const rect = tw.map.getContainer().getBoundingClientRect();
    let best = null, bd = 1e9;
    for (const el of document.querySelectorAll('.stop-marks > div')) {
      if (el.style.display === 'none') continue;
      const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(el.style.transform);
      if (!m) continue;
      const d = Math.hypot(+m[1] - p.x, +m[2] - p.y);
      if (d < bd) { bd = d; best = { x: +m[1] + rect.left, y: +m[2] + rect.top, d }; }
    }
    return best;
  }, n);
  await page.locator('.stops > li[data-stop="2"] button.btn.icon.mini:not(.danger)').click();
  await page.waitForTimeout(400);
  const p3 = await panel();
  check(p3.adding.length === 1 && /Pick a station on the map/.test(p3.adding[0]), 'the + opens a row that waits for a station', p3.adding.join());
  const mk = await markerAt(N.E);
  check(mk && mk.d < 30, 'E has a marker on screen', JSON.stringify(mk));
  const t0 = Date.now();
  await page.mouse.click(mk.x, mk.y);
  await page.waitForFunction((n) => window.tw.world.value.save.lines.some((l) => l.stops.length === 7), undefined, { timeout: 15000 }).catch(() => {});
  out.addMs = Date.now() - t0;
  const s3 = await info();
  check(JSON.stringify(s3.stops) === JSON.stringify([N.A, N.B, N.C, N.E, N.D, N.B, N.A]) && s3.manual, 'the station is put in after C', s3.stops.join());
  console.log('add took ms', out.addMs, 'reversals', JSON.stringify(s3.reversals));
  const p4 = await panel();
  check(p4.adding.length === 0, 'the waiting row is gone');
  check(s3.reversals.length > 0, 'the trains reverse at the spur end (a reversal on a manual line)', JSON.stringify(s3.reversals));
  check(p4.revIcons === 0 && p4.mapIcons === 0, 'no U-turn icon in the list or on the map', `${p4.revIcons} / ${p4.mapIcons}`);
  await page.screenshot({ path: join(ARTIFACTS, 't3023-added.png') });

  // a manual list that is a plain out-and-back again folds back
  await edit({ op: 'orderStops', line: id, stops: [N.A, N.B, N.C, N.B, N.A] });
  await page.waitForTimeout(1200);
  const s4 = await info();
  const p5 = await panel();
  check(s4.manual && s4.folds && p5.toggle && !p5.toggle.disabled, 'a plain out-and-back can switch back', JSON.stringify({ ...s4, toggle: p5.toggle }));
  await edit({ op: 'orderStops', line: id, stops: want });
  await page.waitForTimeout(1200);
  await page.locator('.manual-row button').click();
  await page.waitForTimeout(1500);
  const s5 = await info();
  check(JSON.stringify(s5.stops) === JSON.stringify(s0.stops) && !s5.manual && !s5.oneWay && s5.loop === null, 'switching back gives the open line again', JSON.stringify(s5));
  check(Math.abs(s5.round - s0.round) <= 5, 'its round trip is the one it started with', `${s0.round} -> ${s5.round}`);
  await edit({ op: 'undo' });
  await page.waitForTimeout(1200);
  const s6 = await info();
  check(s6.manual && s6.stops.length === 7, 'one undo takes the switch back', JSON.stringify(s6));
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3023-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
await writeFile(join(ARTIFACTS, 't3023-manual.json'), JSON.stringify(out, null, 1)).catch(() => {});
console.log(fail ? `T-3023 FAILED: ${fail}` : 'T-3023 PASSED');
process.exit(fail ? 1 : 0);
