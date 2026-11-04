// T-1855: line groups survive. On Tokyo's real network in a slot game (not ?new=1): three groups, one
// named and given a bullet by hand; a save by hand writes them into the slot's bytes (LGR1, LGL1: the
// account gets these same bytes, cloud.ts); a reload of the page opens them; dragging a group's row
// into a gap only moves it (it once sent "leave group" for line Number("g:N") = NaN, which the worker's
// Uint32Array made line 0, so the group holding line 0 dissolved); an export and import keeps them.
//   node trainworld/checks/old/t1855-groups-kept.mjs [url]   (from the maps root; default localhost:6189; about a minute)
import { readFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, GL } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6189/';
const TOKYO = new URL('../../data/packs/real/tyo.save', import.meta.url);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1855-', size: [1400, 900], args: GL, acceptDownloads: true });
const open = async () => {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  return page;
};
/** group number -> its lines, name and hand bullet, as the UI sees them */
const groups = (page) => page.evaluate(() => {
  const by = {};
  for (const l of window.tw.world.value.save.lines) if (l.group) (by[l.group] ??= { lines: [], name: l.groupName || '', sym: l.groupSymbol || '', colour: l.groupColour || '' }).lines.push(l.num);
  return by;
});
const edit = (page, op) => page.evaluate(async (op) => (await import('/src/game/world.ts')).edit(op), op);
const active = (page) => page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
const has = (t, tag) => { for (let i = 0; i + 3 < t.length; i++) if (t[i] === tag.charCodeAt(0) && t[i + 1] === tag.charCodeAt(1) && t[i + 2] === tag.charCodeAt(2) && t[i + 3] === tag.charCodeAt(3)) return true; return false; };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
try {
  // 1. a slot game from Tokyo's real network, three groups
  let page = await open();
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, await readFile(TOKYO));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const nums = await page.evaluate(() => window.tw.world.value.save.lines.map((l) => l.num).sort((a, b) => a - b));
  check(nums[0] === 0, 'the game has a line 0');
  for (const pair of [[nums[0], nums[1]], [nums[2], nums[3]], [nums[4], nums[5]]]) await edit(page, { op: 'lineGroup', lines: pair, group: 'new' });
  await page.waitForTimeout(1000);
  const gOf = (by, n) => Number(Object.entries(by).find(([, v]) => v.lines.includes(n))?.[0] ?? 0);
  const gB = gOf(await groups(page), nums[2]);
  await edit(page, { op: 'groupName', group: gB, name: 'Named B' });
  await edit(page, { op: 'groupLook', group: gB, symbol: 'NB', colour: '#ee352e' });
  await page.waitForTimeout(1500);
  const made = await groups(page);
  check(Object.keys(made).length === 3 && made[gB].name === 'Named B' && made[gB].sym === 'NB' && made[gB].colour === '#ee352e', `three groups made: ${JSON.stringify(made)}`);

  // 2. a save by hand: the slot's bytes carry them
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
  await page.waitForTimeout(3000);
  const key = await active(page);
  const tags = await page.evaluate(async (key) => {
    const sf = await import('/src/workers/saveFile.ts');
    const r = await sf.readStored(key);
    const { track } = await sf.decodeSave(r.bytes);
    return Array.from(new Uint8Array(track.buffer ?? track, track.byteOffset ?? 0, track.byteLength ?? track.length));
  }, key);
  check(has(tags, 'LGR1') && has(tags, 'LGL1'), `slot ${key} holds the groups (LGR1 ${has(tags, 'LGR1')}, LGL1 ${has(tags, 'LGL1')})`);
  await page.close();

  // 3. the page reloaded opens the slot with its groups
  page = await open();
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'autosave' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  check(same(await groups(page), made), `reload keeps the groups: ${JSON.stringify(await groups(page))}`);

  // 4. a group's row dragged into a gap moves it and changes no group (line 0's included)
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tab.value = 'lines'; ui.selection.value = null; });
  await page.waitForTimeout(1500);
  const rows = await page.$$eval('.lines-box .rows > li', (els) => els.map((e) => e.dataset.drop ?? ''));
  const grouped = new Set(Object.values(made).flatMap((v) => v.lines.map(String)));
  const target = rows.slice(4).find((r) => r.startsWith('l:') && !grouped.has(r.slice(2)));
  const rect = (sel) => page.$eval(sel, (e) => { e.scrollIntoView({ block: 'center' }); const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; });
  const a = await rect(`.lines-box li[data-drop="g:${gB}"]`);
  await page.mouse.move(a.x + a.w * 0.4, a.y + a.h / 2);
  await page.mouse.down();
  await page.mouse.move(a.x + a.w * 0.4, a.y + a.h / 2 + 10, { steps: 3 });
  const b = await page.$eval(`.lines-box li[data-drop="${target}"]`, (e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.mouse.move(b.x + b.w * 0.4, b.y + b.h * 0.9, { steps: 6 });
  await page.waitForTimeout(300);
  await page.mouse.up();
  await page.waitForTimeout(2000);
  const after = await page.$$eval('.lines-box .rows > li', (els) => els.map((e) => e.dataset.drop ?? ''));
  check(after.indexOf(`g:${gB}`) === after.indexOf(target) + 1, `group B's row moved below ${target}`);
  check(same(await groups(page), made), `a group row's drag keeps every group: ${JSON.stringify(await groups(page))}`);

  // 5. export and import
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  check(same(await groups(page), made), 'export and import keep the groups');
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
