// T-1813: the Lines and Stations tabs' dragging, and the group bullet. Real mouse drags on the real
// save: a drop on a row's middle groups, a drop between rows reorders (and takes a line out of its
// group), a drop between an open group's rows reorders the group; a station row dragged between two
// others; the hand order surviving an export and reload, a header click dropping it; the group pill's
// text and colour by default ("Tokaido Main Line" reads "Tokaido") and set by hand, kept on reload;
// then the shared Tokyo start, a group made there. Screenshots light and dark in
// artifacts/trainworld/t1813/. From the maps root:
//   node trainworld/checks/t1813-drag.mjs [url]   (about two minutes)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6185/';
const OUT = join(ARTIFACTS, 't1813');
const TOKYO = new URL('../data/packs/real/tyo.save', import.meta.url);
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1813-', size: [1400, 900], args: GL, acceptDownloads: true });
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const load = async (bytes) => {
    await page.evaluate(() => { window.tw.client.lastLoad = null; });
    await importSave(page, bytes);
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.waitForTimeout(3000);
  };
  await load(await readFile(REAL_SAVE));
  const settle = (ms = 1200) => page.waitForTimeout(ms);
  const tab = (t) => page.evaluate(async (t) => { const ui = await import('/src/game/ui.ts'); ui.tab.value = t; ui.selection.value = null; }, t);
  const hand = () => page.evaluate(async () => (await import('/src/game/handOrder.ts')).handOrder.value);
  const line = (name) => page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.name === n); return l && { id: l.id, num: l.num, group: l.group ?? 0, groupSymbol: l.groupSymbol ?? '', groupColour: l.groupColour ?? '', colour: l.colour }; }, name);
  const rows = (sel = '.lines-box .rows > li') => page.$$eval(sel, (els) => els.map((e) => e.dataset.drop ?? ''));
  const box = async (sel) => {
    await page.$eval(sel, (e) => e.scrollIntoView({ block: 'center' }));
    await settle(200);
    return page.$eval(sel, (e) => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; });
  };
  /** drag the row `from` onto the row `to` at `f` of its height (0.5 the middle, 0.1 the gap above, 0.9 below) */
  const drag = async (from, to, f, shot) => {
    const a = await box(from);
    await page.mouse.move(a.x + a.w * 0.4, a.y + a.h / 2);
    await page.mouse.down();
    await page.mouse.move(a.x + a.w * 0.4, a.y + a.h / 2 + 10, { steps: 3 });
    const b = await page.$eval(to, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
    await page.mouse.move(b.x + b.w * 0.4, b.y + b.h * f, { steps: 6 });
    await settle(250);
    if (shot) await page.screenshot({ path: join(OUT, shot), clip: { x: 0, y: Math.max(0, b.y - 120), width: 440, height: 260 } });
    const marked = await page.$eval(to, (e) => e.className);
    await page.mouse.up();
    await settle();
    return marked;
  };
  const L = (n) => `.lines-box li[data-drop="l:${n}"]`;
  const G = (g) => `.lines-box li[data-drop="g:${g}"]`;

  check((await hand()).lines.length === 0 && (await hand()).stations.length === 0, 'the real save loads with no hand order');
  await tab('lines');
  await page.waitForSelector(L((await line('4')).id));
  const [l1, l4, l5, l6, l7] = await Promise.all(['1', '4', '5', '6', '7'].map(line));

  // 1. a drop on the middle of a row groups
  let m = await drag(L(l5.id), L(l4.id), 0.5, 'light-drop-on.png');
  let a4 = await line('4'), a5 = await line('5');
  check(m.includes('drop-on') && a4.group && a4.group === a5.group, `drop on the middle groups 5 with 4 (mark "${m}", groups ${a4.group}/${a5.group})`);
  check((await hand()).lines.length === 0, 'grouping alone makes no hand order');
  const g = a4.group;
  m = await drag(L(l6.id), G(g), 0.5);
  check((await line('6')).group === g, `drop on the group row's middle adds 6 (mark "${m}")`);

  // 2. a drop between rows reorders, and the sort shows none
  m = await drag(L(l7.id), L(l1.id), 0.1, 'light-drop-gap.png');
  let order = await rows();
  let h = await hand();
  check(m.includes('drop-before') && order.indexOf('l:' + l7.id) === order.indexOf('l:' + l1.id) - 1 && h.lines.indexOf(l7.id) === h.lines.indexOf(l1.id) - 1, `drop above 1 puts 7 just above it (mark "${m}")`);
  check(await page.$$eval('.lines-box .sort-btn.on', (e) => e.length) === 0, 'no sort column is on while the hand order holds');

  // 3. inside an open group: a reorder of the group's lines
  check(await page.$eval(`${G(g)} .chevron`, (e) => e.getAttribute('aria-expanded')) === 'true', 'a group shows open by default (T-1818)');
  await settle();
  m = await drag(L(l6.id), L(l4.id), 0.1, 'light-drop-in-group.png');
  order = await rows();
  const at = (id) => order.indexOf('l:' + id);
  check(m.includes('drop-before') && (await line('6')).group === g && at(l6.id) === order.indexOf('g:' + g) + 1 && at(l4.id) < at(l5.id), `6 moved above 4 inside the group, still in it (rows ${order.slice(order.indexOf('g:' + g), order.indexOf('g:' + g) + 4).join(' ')})`);
  await page.screenshot({ path: join(OUT, 'light-lines-tab.png'), clip: { x: 0, y: 40, width: 440, height: 520 } });

  // 4. a member dropped between rows outside the group leaves it
  m = await drag(L(l5.id), L(l1.id), 0.9);
  order = await rows();
  check((await line('5')).group === 0 && (await line('4')).group === g && at(l5.id) === at(l1.id) + 1, `5 dropped below 1 leaves the group and sits there (mark "${m}")`);

  // 5. the group's bullet: text and colour by default, then a long name
  const pill = () => page.$eval(`${G(g)} .group-pill`, (e) => ({ text: e.textContent, colour: getComputedStyle(e).color, border: getComputedStyle(e).borderTopColor, bg: getComputedStyle(e).backgroundColor }));
  const rgb = (hex) => `rgb(${parseInt(hex.slice(1, 3), 16)}, ${parseInt(hex.slice(3, 5), 16)}, ${parseInt(hex.slice(5, 7), 16)})`;
  let p = await pill();
  check(p.text === '4' && p.colour === rgb(l4.colour) && p.border === p.colour && p.bg === 'rgba(0, 0, 0, 0)', `default pill reads "4" in the 4's colour, see-through (${JSON.stringify(p)})`);
  const edit = (op) => page.evaluate(async (op) => (await import('/src/game/world.ts')).edit(op), op);
  await edit({ op: 'groupName', group: g, name: 'Tokaido Main Line' });
  await settle();
  p = await pill();
  check(p.text === 'Tokaido', `"Tokaido Main Line" reads "${p.text}"`);

  // 6. set by hand in the group view's title
  await page.evaluate(async (g) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'group', group: g }; }, g);
  await page.waitForSelector('#inspector .badge-edit .group-pill');
  await settle(600);
  await page.click('#inspector .badge-edit');
  await page.waitForSelector('#inspector input[aria-label="Group bullet"]');
  await page.click('#inspector input[aria-label="Group bullet"]');
  await page.keyboard.type('Lex');
  await page.keyboard.press('Enter');
  await settle();
  await edit({ op: 'groupLook', group: g, symbol: 'Lex', colour: '#ee352e' });
  await settle();
  await page.screenshot({ path: join(OUT, 'light-group-title.png'), clip: { x: 0, y: 380, width: 440, height: 360 } });
  const gl = await line('4');
  p = await page.$eval('#inspector .title .group-pill', (e) => ({ text: e.textContent, colour: getComputedStyle(e).color }));
  check(gl.groupSymbol === 'Lex' && gl.groupColour === '#ee352e' && p.text === 'Lex' && p.colour === rgb('#ee352e'), `bullet set by hand: ${JSON.stringify([gl.groupSymbol, gl.groupColour, p])}`);
  await page.evaluate(async (n) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'line', line: n }; }, l4.id);
  await page.waitForSelector('#inspector .line-group .group-pill');
  await settle(600);
  const lp = await page.$$eval('#inspector .line-group .chip', (e) => e.length);
  check(lp === 0 && (await page.$eval('#inspector .line-group .group-pill', (e) => e.textContent)) === 'Lex', `the line panel's group row shows the one pill, no chips (${lp})`);
  await page.screenshot({ path: join(OUT, 'light-line-panel.png'), clip: { x: 0, y: 380, width: 440, height: 300 } });

  // 7. stations: a row dragged above the first
  await tab('stations');
  await page.waitForSelector('.rows.stations > li[data-drop]');
  let st = await rows('.rows.stations > li[data-drop]');
  const third = st[2];
  m = await drag(`.rows.stations > li[data-drop="${third}"]`, `.rows.stations > li[data-drop="${st[0]}"]`, 0.1, 'light-station-drop.png');
  st = await rows('.rows.stations > li[data-drop]');
  check(m.includes('drop-before') && st[0] === third && (await hand()).stations.length > 0, `station row moved to the top (mark "${m}")`);

  // 8. export and reload: the hand orders and the bullet stay
  const before = await hand();
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await page.evaluate(async () => (await import('/src/game/handOrder.ts')).takeHandOrder([], []));
  await load(bytes);
  const after = await hand();
  const r4 = await line('4');
  check(after.lines.join() === before.lines.join() && after.stations.join() === before.stations.join() && after.lines.length > 100, `hand orders kept on reload (${after.lines.length} lines, ${after.stations.length} stations)`);
  check(r4.groupSymbol === 'Lex' && r4.groupColour === '#ee352e' && (await line('6')).group === r4.group, `the bullet kept on reload: ${JSON.stringify(r4)}`);
  await tab('stations');
  await settle();
  check((await rows('.rows.stations > li[data-drop]'))[0] === third, 'the station order shows after reload');

  // 9. a header click sorts again and drops the hand order
  await tab('lines');
  await settle();
  await page.click('.lines-box .sort-btn:has-text("Riders")');
  await settle();
  check((await hand()).lines.length === 0 && await page.$$eval('.lines-box .sort-btn.on', (e) => e.length) === 1, 'a header click drops the hand order');
  await edit({ op: 'groupLook', group: r4.group, symbol: '', colour: null });
  await settle();
  const back = await line('4');
  check(back.groupSymbol === '' && back.groupColour === '', 'the bullet back to its defaults');

  // 10. dark
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await settle(800);
  await page.$eval(G(r4.group), (e) => e.scrollIntoView({ block: 'center' }));
  await page.screenshot({ path: join(OUT, 'dark-lines-tab.png'), clip: { x: 0, y: 40, width: 440, height: 520 } });
  await page.evaluate(async (g) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'group', group: g }; }, r4.group);
  await settle(1000);
  await page.screenshot({ path: join(OUT, 'dark-group-title.png'), clip: { x: 0, y: 380, width: 440, height: 300 } });

  // 11. the shared Tokyo start: two lines grouped by a drag, the pill from their names
  await load(await readFile(TOKYO));
  await tab('lines');
  await settle(1500);
  // the first two rows, so both are on screen
  const shown = (await rows()).filter((k) => k.startsWith('l:')).slice(0, 2).map((k) => k.slice(2));
  const ty = await page.evaluate((ids) => ids.map((id) => window.tw.world.value.save.lines.find((l) => l.id === id)).map((l) => ({ id: l.id, name: l.name, colour: l.colour })), shown);
  console.log('tokyo lines', JSON.stringify(ty));
  const [ta, tb] = ty;
  await drag(L(tb.id), L(ta.id), 0.5);
  const tg = await page.evaluate((id) => window.tw.world.value.save.lines.find((l) => l.id === id)?.group ?? 0, ta.id);
  check(tg > 0, `Tokyo: ${tb.name} dropped on ${ta.name} groups them`);
  if (tg) {
    p = await page.$eval(G(tg), (e) => ({ text: e.querySelector('.group-pill').textContent, name: e.querySelector('.line-name').textContent }));
    console.log('tokyo pill', JSON.stringify(p));
    // two colours, one line each: the busier line's colour (the older line's when level)
    const want = await page.evaluate(async (ids) => {
      const t = (await import('/src/game/today.ts')).today.value;
      const ls = ids.map((id) => window.tw.world.value.save.lines.find((l) => l.id === id)).sort((a, b) => a.num - b.num);
      const r = (l) => t.get(l.id)?.riders ?? 0;
      return (r(ls[1]) > r(ls[0]) ? ls[1] : ls[0]).colour;
    }, [ta.id, tb.id]);
    const col = await page.$eval(`${G(tg)} .group-pill`, (e) => getComputedStyle(e).color);
    check(p.text === p.name.replace(/(本線|ライン|線)$/u, '') && p.text.length > 0, `Tokyo pill "${p.text}" for "${p.name}"`);
    // T-1817: the pill's colour is kept legible on the shade, a little lighter or darker
    const near = (a, b) => { const p = (s) => s.match(/\d+/g).map(Number); return p(a).every((v, i) => Math.abs(v - p(b)[i]) <= 24); };
    check(near(col, rgb(want)), `Tokyo pill in the busier line's colour ${want} (${col})`);
    await page.$eval(G(tg), (e) => e.scrollIntoView({ block: 'center' }));
    await page.screenshot({ path: join(OUT, 'dark-tokyo-lines.png'), clip: { x: 0, y: 40, width: 440, height: 520 } });
    await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
    await settle(800);
    await page.screenshot({ path: join(OUT, 'light-tokyo-lines.png'), clip: { x: 0, y: 40, width: 440, height: 520 } });
  }
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
