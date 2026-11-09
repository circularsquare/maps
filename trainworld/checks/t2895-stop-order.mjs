// T-2895: a line's stop order set by hand. On the real save: a stop row dragged to a new place keeps its
// order when another stop goes and comes back, undo and redo carry it, a save round-trips it, Sort stops
// gives the order back to the game, and the loaded save has no line ordered by hand. On track built for
// it (a Y, in blueprint): each refusal says which two stops and leaves the list as it was.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t2895-stop-order.mjs [dev url]
// About 2 minutes. Pictures in artifacts/trainworld/t2895/.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2895');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6220/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const chrome = await launchChrome({ prefix: 'tw-t2895-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);

const lineOf = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = l && w.lineStats[l.id];
  return l ? { id: l.id, stops: l.stops.map(Number), hand: !!st.hand, reversals: st.reversals.length, status: st.status } : null;
}, num);
const until = async (num, f, arg) => { await page.waitForFunction(f, [num, arg], { timeout: 60000 }).catch(() => {}); await wait(300); return lineOf(num); };
const handIs = (num, hand) => until(num, ([num, hand]) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return !!l && !!w.lineStats[l.id].hand === hand; }, hand);
const stopsAre = (num, stops) => until(num, ([num, s]) => JSON.stringify(window.tw.world.value.save.lines.find((x) => x.num === num)?.stops.map(Number)) === s, JSON.stringify(stops));
const editStops = async (id, plain = false) => {
  await page.evaluate((id) => { window.tw.tools.exit(); window.tw.selection.value = { kind: 'line', line: id }; }, id);
  await wait(800);
  await page.locator('.line-stop-actions button').first().click();
  // a broken line has no rows in the related lines view: the plain list shows them
  if (plain && (await page.locator('.related-toggle.on').count())) await page.locator('.related-toggle.on').click();
  await page.waitForSelector('.order li[data-stop]', { timeout: 20000 }).catch(async (e) => {
    await page.screenshot({ path: join(OUT, 'no-list.png') });
    console.log(await page.evaluate(() => JSON.stringify({ tool: window.tw.tools.active ?? null, rows: document.querySelectorAll('li[data-stop]').length, order: document.querySelectorAll('.order').length, actions: [...document.querySelectorAll('.line-stop-actions button')].map((b) => b.innerText) })));
    throw e;
  });
  await wait(600);
};
/** Drag stop `from`'s row so it drops into the gap before stop `gap` (the list may be drawn last stop first). */
const dragRow = async (from, gap, n, shot) => {
  const row = (i) => page.locator(`.order li[data-stop="${i}"]`).first();
  await row(Math.min(gap, n - 1)).scrollIntoViewIfNeeded();
  await row(from).scrollIntoViewIfNeeded();
  await wait(200);
  const y = async (i) => { const b = await row(i).boundingBox(); return [b.y, b.height]; };
  const flipped = (await y(0))[0] > (await y(1))[0];
  // the gap before stop `gap` is the top half of its row, or the bottom half of the row before; flipped, the other halves
  const target = gap < n ? gap : n - 1, before = gap < n;
  const [ty, th] = await y(target);
  const dropY = ty + th * (before !== flipped ? 0.25 : 0.75);
  const name = await row(from).locator('.stop-name').boundingBox();
  const x = name.x + 12, y0 = name.y + name.height / 2;
  await page.mouse.move(x, y0);
  await page.mouse.down();
  for (let k = 1; k <= 10; k++) { await page.mouse.move(x, y0 + ((dropY - y0) * k) / 10); await wait(30); }
  if (shot) await page.screenshot({ path: join(OUT, shot) });
  await page.mouse.up();
  await wait(400);
};
const listOrder = () => page.evaluate(() => [...document.querySelectorAll('.order li[data-stop]')].map((li) => Number(li.dataset.stop)));
const exported = async () => {
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  return readFile(await (await dl).path());
};

try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(3000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));

  // ---- the old save: nothing ordered by hand, and no block for it when it is saved again
  const loaded = await page.evaluate(() => { const w = window.tw.world.value; return { lines: w.save.lines.length, hand: w.save.lines.filter((l) => w.lineStats[l.id].hand).length }; });
  check(loaded.lines > 50 && loaded.hand === 0, 'the real save loads with no line ordered by hand', `${loaded.lines} lines, ${loaded.hand} by hand`);

  // ---- a row dragged
  const pick = await page.evaluate(() => {
    const w = window.tw.world.value;
    const l = w.save.lines.find((l) => { const st = w.lineStats[l.id]; return !st.bus && !st.loop && st.status === 'running' && st.reversals.length === 0 && st.endTurns.length === 0 && l.stops.length >= 8 && l.stops.length <= 30 && new Set(l.stops).size === l.stops.length; });
    return l && { id: l.id, num: l.num, name: l.name, stops: l.stops.map(Number) };
  });
  check(!!pick, 'a running line of 8 to 30 stops that never reverses', pick?.name);
  const s = pick.stops, n = s.length;
  await editStops(pick.id);
  check((await page.locator('.sort-stops').count()) === 0, 'no Sort stops button on a line in the game\'s order');
  const want = [s[0], s[2], s[3], s[1], ...s.slice(4)];
  await dragRow(1, 4, n, 'drag.png');
  let now = await stopsAre(pick.num, want);
  now = await handIs(pick.num, true);
  check(same(now.stops, want) && now.hand, 'the second stop dragged below the fourth goes there, and the line is ordered by hand', `${s.slice(0, 5)} -> ${now.stops.slice(0, 5)}`);
  check(now.reversals >= 1 && now.status !== 'broken', 'its trains reverse, with no question asked', `${now.reversals} reversals, ${now.status}`);
  check((await page.locator('.sort-stops').count()) === 1, 'the Sort stops button shows');
  check(same(await listOrder(), [...Array(n).keys()]) || same(await listOrder(), [...Array(n).keys()].reverse()), 'the list shows the new order');
  await wait(1200);
  await page.screenshot({ path: join(OUT, 'dragged.png') });
  await page.locator('.sort-stops').scrollIntoViewIfNeeded();
  await wait(300);
  await page.screenshot({ path: join(OUT, 'sort-button.png') });

  // ---- undo and redo
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  now = await handIs(pick.num, false);
  check(same(now.stops, s) && !now.hand, 'undo puts the stops and the game\'s order back');
  await page.evaluate(() => window.tw.client.edit({ op: 'redo' }));
  now = await handIs(pick.num, true);
  check(same(now.stops, want) && now.hand, 'redo brings the hand order back');

  // ---- a stop taken out elsewhere and put back by the click's edit: the dragged order stays
  const far = n - 2;
  await page.locator(`.order li[data-stop="${far}"] button.danger`).first().click();
  const less = want.filter((_, i) => i !== far);
  now = await stopsAre(pick.num, less);
  check(same(now.stops, less) && now.hand, 'a stop taken out elsewhere leaves the dragged order', `${now.stops.slice(0, 5)}`);
  const added = await page.evaluate(([line, station]) => window.tw.client.edit({ op: 'insertStop', line, station }), [pick.num, want[far]]);
  now = await until(pick.num, ([num, k]) => window.tw.world.value.save.lines.find((x) => x.num === num)?.stops.length === k, n);
  check(added.ok && now.stops.length === n && same(now.stops.filter((x) => x !== want[far]), less) && now.hand, 'a stop added elsewhere fits in and leaves the dragged order', `${now.stops.slice(0, 5)} ... ${now.stops.slice(-3)}`);
  const kept = now.stops;

  // ---- the save carries it
  const bytes = await exported();
  await page.evaluate(() => { window.tw.tools.exit(); window.tw.client.lastLoad = null; });
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(3000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const back = await page.evaluate((name) => {
    const w = window.tw.world.value, by = w.save.lines.filter((l) => w.lineStats[l.id].hand);
    const names = new Map(w.save.stations.map((x) => [x.num, x.name]));
    return by.map((l) => ({ id: l.id, num: l.num, name: l.name, stops: l.stops.map((x) => names.get(Number(x))) }));
  }, pick.name);
  const keptNames = await page.evaluate((ids) => { const names = new Map(window.tw.world.value.save.stations.map((x) => [x.num, x.name])); return ids.map((i) => names.get(i)); }, kept).catch(() => null);
  check(back.length === 1 && back[0].name === pick.name && back[0].stops.length === n, 'loaded again, that line alone is ordered by hand', back.map((b) => b.name).join(', '));
  check(back.length === 1 && (keptNames === null || back[0].stops.slice(0, 4).join('|') === [s[0], s[2], s[3], s[1]].map((i) => keptNames[kept.indexOf(i)]).join('|')), 'and its stops are in the dragged order', back[0]?.stops.slice(0, 5).join(', '));

  // ---- Sort stops
  const again = back[0];
  await editStops(again.id);
  const before = (await lineOf(again.num)).stops;
  await page.locator('.sort-stops').click();
  now = await handIs(again.num, false);
  const straight = await page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return { reversals: w.lineStats[l.id].reversals.length, names: l.stops.map((x) => w.save.stations.find((q) => q.num === Number(x))?.name) }; }, again.num);
  const first = [...again.stops].sort().join('|') === [...straight.names].sort().join('|');
  check(!now.hand && straight.reversals === 0 && first && !same(now.stops, before), 'Sort stops puts the stops in the game\'s order again, no train reversing', straight.names.slice(0, 5).join(', '));
  check((await page.locator('.sort-stops').count()) === 0, 'and the button goes');
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  now = await handIs(again.num, true);
  check(now.hand && same(now.stops, before), 'undo brings the hand order back');
  await page.evaluate(() => window.tw.tools.exit());

  // ---- refusals, on a Y built for them: S - T - K - J - M along a line, a branch J - B leaving J towards M
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const o = tw.lngLatToPack(-74.62, 39.86);
    const at = (x, y) => ({ kind: 'free', x: o.x + x, y: o.y + y, level: -1 });
    const near = (x, y) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - o.x - x, n.y - o.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const out = [];
    const xs = [-3000, 0, 1500, 3000, 6000];
    out.push(await c.edit({ op: 'route', from: at(xs[0], 0), to: at(xs[1], 0), pis: [], single: false }));
    for (let k = 2; k < xs.length; k++) { await pause(300); out.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(xs[k - 1], 0) }, to: at(xs[k], 0), pis: [], single: false })); }
    await pause(300);
    out.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(3000, 0) }, to: at(5600, 1500), pis: [], single: false }));
    await pause(400);
    const st = { S: near(-3000, 0), T: near(0, 0), M: near(6000, 0), B: near(5600, 1500) };
    for (const [name, node] of Object.entries(st)) { out.push(await c.edit({ op: 'addStation', at: { kind: 'node', node }, platform: 200, name: `Check ${name}` })); await pause(200); }
    const y = await c.edit({ op: 'addLine', stops: [st.B, st.T, st.M], name: 'Check Y', colour: '#0a7d3b' });
    const flat = await c.edit({ op: 'addLine', stops: [st.S, st.T, st.M], name: 'Check flat', colour: '#7d0a3b' });
    await pause(600);
    const tk = tw.world.value.edges.find((e) => [e.a, e.b].includes(st.T) && [e.a, e.b].includes(near(1500, 0)));
    return { st, y: y.line, flat: flat.line, tk: tk?.id, bad: out.filter((a) => !a.ok).map((a) => a.issues.join(',')), yOk: y.ok, flatOk: flat.ok };
  });
  check(made.bad.length === 0 && made.yOk && made.flatOk && made.tk !== undefined, 'a Y of blueprint track with two lines on it', JSON.stringify(made));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.60, 39.865], zoom: 12.3 }));
  const yLine = await lineOf(made.y);
  check(yLine.reversals === 1, 'B, T, M reverses at T');
  const noticeText = () => page.evaluate(() => document.querySelector('#tool-notes')?.innerText ?? '');
  await editStops(yLine.id);
  await dragRow(1, 0, 3);
  await wait(800);
  let text = await noticeText();
  now = await lineOf(made.y);
  check(text.includes("Trains can't get from Check B to Check M without turning back at a junction."), 'T dragged first is refused, naming the two stops', JSON.stringify(text.split('\n')[0]));
  check(same(now.stops, yLine.stops) && !now.hand && same((await listOrder()).slice().sort(), [0, 1, 2]), 'and the row is back where it was');
  await page.screenshot({ path: join(OUT, 'refused-junction.png') });
  await page.evaluate(() => window.tw.tools.exit());
  // the track between T and K goes: the flat line is broken, and S and T have no track to M
  const cut = await page.evaluate((edge) => window.tw.client.edit({ op: 'removeEdge', edge }), made.tk);
  await wait(800);
  const flat = await lineOf(made.flat);
  check(cut.ok && flat.status === 'broken', 'with the track past T removed, S, T, M is broken', `${cut.ok} ${flat.status}`);
  await editStops(flat.id, true);
  await dragRow(2, 0, 3);
  await wait(800);
  text = await noticeText();
  now = await lineOf(made.flat);
  check(text.includes('No track joins Check M and Check S.'), 'M dragged first is refused: no track joins M and S', JSON.stringify(text.split('\n')[0]));
  check(same(now.stops, flat.stops) && !now.hand, 'and the stops are as they were');
  await page.screenshot({ path: join(OUT, 'refused-no-track.png') });
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2895 FAILED: ${fail}` : 'T-2895 PASSED');
process.exit(fail ? 1 : 0);
