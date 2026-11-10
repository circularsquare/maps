// T-3023/T-3024: manual stop sequence ("Edit every stop in order"). On a new game in New York a stem
// A - B - J - C - D along x with a spur from the junction J to a dead-end station E, and a line A, B, C, D.
// In stop editing: the toggle writes the whole cycle (A B C D C B A, numbered rows, one way, no Direction
// row); the cross on the way back's C skips that visit only; the "+" on C then a click on E's marker puts
// E in after C (the spur leaves facing east, so it cannot follow B); no U-turn icon shows on the line's reversals; a list that is a plain out-and-back folds
// back through the toggle to the open line, and one undo takes the switch back. Times the add.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3023-manual.mjs [dev url] [dark]
// T-3089/T-3080: hovering a row lights its station; every visit has a cross (the first, the last and a turning one too);
// a stop served one way gets the half-filled marker; the numbers beside the stations on the map (a click takes that visit
// out), the empty circles (a click puts a stop in) and a click on a stop do what the cross does. Screenshots t3089-*.png.
// About a minute. Screenshots artifacts/trainworld/t3023-*.png.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6171/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];
const SHADE = process.argv.includes('dark') ? 'dark' : 'light';

const chrome = await launchChrome({ prefix: 'tw-t3023-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const out = { when: new Date().toISOString() };

try {
  await page.goto(`${BASE}?debug=1&exact&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
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
  check(p1.rows.every((r) => r.cross), 'every visit has a cross, the ends and the turning stop too', p1.rows.map((r) => (r.cross ? 'x' : '-')).join(''));
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
  // ---- T-3089 / T-3080
  const stationId = (n) => page.evaluate((n) => window.tw.world.value.save.stations.find((s) => s.num === n).id, n);
  const markerBg = (n) => page.evaluate((n) => {
    const tw = window.tw, node = tw.world.value.nodes.get(String(n)) ?? [...tw.world.value.nodes.values()].find((x) => Number(x.id) === n);
    const [lng, lat] = tw.packToLngLat(node.x, node.y), p = tw.map.project([lng, lat]);
    let best = null, bd = 1e9;
    for (const el of document.querySelectorAll('.stop-marks > div.stop-mark')) {
      const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(el.style.transform);
      if (!m || el.style.display === 'none') continue;
      const d = Math.hypot(+m[1] - p.x, +m[2] - p.y);
      if (d < bd) { bd = d; best = el; }
    }
    return best ? { bg: best.style.background, cls: best.className } : null;
  }, n);
  const visitMarks = () => page.evaluate(() => {
    const rect = window.tw.map.getContainer().getBoundingClientRect();
    return [...document.querySelectorAll('.stop-marks > .visit-mark')].filter((e) => e.style.display !== 'none').map((e) => {
      const m = /translate\((-?[\d.]+)px, (-?[\d.]+)px\)/.exec(e.style.transform) ?? [0, 0, 0];
      return { add: e.classList.contains('add'), text: e.textContent, x: +m[1] + rect.left, y: +m[2] + rect.top };
    });
  });
  const settle = async (len) => { await page.waitForFunction((a) => window.tw.world.value.save.lines.find((l) => l.num === a[0]).stops.length === a[1], [id, len], { timeout: 15000 }).catch(() => {}); await page.waitForTimeout(400); };
  const bId = await stationId(N.B);
  const hotNow = () => page.evaluate(async () => (await import('/src/map/hovered.ts')).hot().stations);
  await page.locator('.stops > li[data-stop="1"]').hover();
  await page.waitForTimeout(300);
  check((await hotNow()).includes(bId), 'hovering a row lights its station on the map', JSON.stringify(await hotNow()));
  check((await markerBg(N.B))?.cls.includes('hot'), 'the station marker takes the light');
  await page.screenshot({ path: join(ARTIFACTS, `t3089-hover-${SHADE}.png`) });
  await page.mouse.move(700, 20);
  await page.waitForTimeout(300);
  check((await hotNow()).length === 0, 'the light goes when the pointer leaves the row');
  check((await page.evaluate(() => window.tw.selection.value.kind)) === 'line', 'the selection stays the line');

  // the ends and the turning visit go like any other
  await page.locator('.stops > li[data-stop="0"] button.danger').click();
  await settle(5);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.B, N.C, N.D, N.C, N.B]), 'the first visit goes: the line now starts where its next stop is', (await info()).stops.join());
  await edit({ op: 'undo' }); await settle(7);
  await page.locator('.stops > li[data-stop="6"] button.danger').click();
  await settle(5);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.B, N.C, N.D, N.C, N.B]), 'the last row does the same as the first', (await info()).stops.join());
  await edit({ op: 'undo' }); await settle(7);
  await page.locator('.stops > li[data-stop="3"] button.danger').click();
  await settle(5);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.A, N.B, N.C, N.B, N.A]), 'the turning visit goes: the line is shorter', (await info()).stops.join());
  await edit({ op: 'undo' }); await settle(7);

  // B on the way back out: B is served one way, so its marker is half filled
  const two = await markerBg(N.B);
  check(!two.bg.includes('linear-gradient'), 'a stop served both ways has a whole dot', two.bg);
  await page.locator('.stops > li[data-stop="5"] button.danger').click();
  await settle(6);
  await page.waitForTimeout(600);
  const half = await markerBg(N.B);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.A, N.B, N.C, N.D, N.C, N.A]) && half.bg.includes('linear-gradient'), 'a stop left with one way has a half-filled dot', half.bg);

  // the numbers on the map
  await page.evaluate(([c]) => { window.tw.map.jumpTo({ center: [c[0] - 0.003, c[1] + 0.0004], zoom: 13.6 }); }, [C]);
  await page.waitForTimeout(1200);
  console.log('passed', JSON.stringify(await page.evaluate((num) => { const w = window.tw.world.value; return w.lineStats[w.save.lines.find((l) => l.num === num).id].passed; }, id)));
  const vm = await visitMarks();
  console.log('map marks', JSON.stringify(vm.map((v) => (v.add ? 'o' : v.text))));
  check(vm.filter((v) => !v.add).map((v) => v.text).sort().join('|') === '1, 6|2|3|4|5', 'every visit has its number on the map', vm.map((v) => v.text).join('|'));
  check(vm.filter((v) => v.add).length >= 1, 'an empty circle shows where a direction has no stop', String(vm.filter((v) => v.add).length));
  await page.screenshot({ path: join(ARTIFACTS, `t3089-numbers-${SHADE}.png`) });
  const five = vm.find((v) => v.text === '5');
  await page.mouse.click(five.x, five.y);
  await settle(5);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.A, N.B, N.C, N.D, N.A]), 'a click on a number takes that visit out', (await info()).stops.join());
  await page.mouse.move(5, 5);
  await page.waitForTimeout(500);
  const circles = (await visitMarks()).filter((v) => v.add);
  check(circles.length >= 2, 'the empty circles are there for the way back', String(circles.length));
  const bm = await page.evaluate((n) => { const tw = window.tw, node = tw.world.value.nodes.get(String(n)) ?? [...tw.world.value.nodes.values()].find((x) => Number(x.id) === n); const [lng, lat] = tw.packToLngLat(node.x, node.y), p = tw.map.project([lng, lat]), r = tw.map.getContainer().getBoundingClientRect(); return { x: p.x + r.left, y: p.y + r.top }; }, N.B);
  circles.sort((a, b) => Math.hypot(a.x - bm.x, a.y - bm.y) - Math.hypot(b.x - bm.x, b.y - bm.y));
  await page.mouse.click(circles[0].x, circles[0].y);
  await settle(6);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.A, N.B, N.C, N.D, N.B, N.A]), 'a click on an empty circle puts the stop in', (await info()).stops.join());
  // a click on a stop is the cross
  const mb = await markerAt(N.B);
  await page.mouse.click(mb.x, mb.y);
  await settle(5);
  check(JSON.stringify((await info()).stops) === JSON.stringify([N.A, N.C, N.D, N.B, N.A]), 'a click on a stop takes out one visit, its first, as its cross does', (await info()).stops.join());
  await edit({ op: 'orderStops', line: id, stops: want });
  await settle(7);
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
