// T-3543: one schedule button, and default headways. A new game in New York with a line built through the
// harness (lib/play.mjs): the line panel has no copy or paste schedule button and one schedule button, a size
// bigger than the chart button beside it. In the schedule window the line's headways are set (the Headway switch),
// "Set as default" makes them the default (the window shows them), a second line drawn after starts with those
// headways. The game is exported and opened again: the default holds (the save's header carries it) and a line
// drawn then starts with it. "Use default" gives a line the default (one undo each way; off with no default or a match). Shots: the line
// panel (review/t3543.png) and the schedule window. From the tree root, on a Vite of your own:
//   node trainworld/checks/t3543-default-headways.mjs [dev url]      (about three minutes)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, importSave } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6170/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32];
const REVIEW = join(ARTIFACTS, 'review');
await mkdir(REVIEW, { recursive: true });

const g = await openGame({ url: BASE, city: 'nyc', ui: { shade: 'light', display: { dayShown: 'weekday' }, v: 2 }, prefix: 'tw-t3543-' });
const page = g.page;
/** a line's headways in seconds as the game has them: null on counts */
const headwaysOf = (name) => page.evaluate(async (n) => {
  const w = await import('/src/game/world.ts'), world = window.tw.world.value;
  const l = world.save.lines.find((x) => x.name === n);
  return l ? w.headwayShown(l.id) : undefined;
}, name);
const defaultNow = () => page.evaluate(async () => (await import('/src/game/newHeadway.ts')).newHeadway.value.rail ?? null);
const windowText = () => page.evaluate(() => document.querySelector('.sched-default')?.innerText ?? null);
const closeWindow = async () => { await page.keyboard.press('Escape'); await page.waitForTimeout(400); };
try {
  await g.act.camera(C, 13);
  await g.act.tool('track', { type: 'regular', tracks: 2, snap: { real: false, roads: false, beside: false }, keepPlacing: false });
  const A = P(-1.5, 0), B = P(0, 0.3), D = P(1.5, 0);
  await g.act.drawTrack([A, B, D]);
  for (const p of [A, B, D]) await g.act.placeStation(p);
  await g.act.construct();
  await g.act.newLine([A, B, D], { name: 'First', zoom: 15 });
  check((await g.see.panel())?.title === 'First', 'the first line is built and its panel open');
  check((await defaultNow()) === null, 'no default at the start');

  // the line panel's buttons
  const buttons = await page.evaluate(() => [...document.querySelectorAll('#inspector .h .sched-copy button')].map((b) => ({ label: b.getAttribute('aria-label'), w: b.getBoundingClientRect().width, h: b.getBoundingClientRect().height })));
  const labels = buttons.map((b) => b.label);
  check(!labels.some((x) => /Copy the schedule|Paste the schedule/.test(x ?? '')), 'no copy or paste schedule button', labels.join(' | '));
  check(!(await page.getByRole('button', { name: /Paste a schedule|Paste the schedule|Copy the schedule/ }).count()), 'none of them anywhere on the page');
  const sched = buttons.find((b) => b.label === 'Detailed schedule'), other = buttons.find((b) => b.label !== 'Detailed schedule');
  check(!!sched && !!other && sched.w > other.w && sched.w - other.w < 4 && other.w >= 18, 'the schedule button is a little bigger than the chart button (Anita, 2026-11-17)', JSON.stringify(buttons));
  await g.shot('t3543_line_panel_full.png', { element: '#inspector' });
  await page.evaluate(() => document.querySelector('#inspector .h .sched-copy')?.scrollIntoView({ block: 'center' }));
  const head = await page.evaluate(() => { const r = document.querySelector('#inspector .h .sched-copy').parentElement.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.screenshot({ path: join(REVIEW, 't3543.png'), clip: { x: Math.max(0, head.x - 12), y: Math.max(0, head.y - 60), width: Math.min(420, head.w + 24), height: 220 } });

  // the window: Headway, three headways, then Set as default
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  check(/Default not set/.test((await windowText()) ?? ''), 'a fresh game: the window says Default not set', await windowText());
  check(!(await page.getByRole('button', { name: 'Clear default' }).count()), 'no Clear default button');
  check(await page.getByRole('button', { name: 'Use default' }).isDisabled(), 'Use default is off while there is no default');
  check((await headwaysOf('First')) === null, 'the first line runs on counts');
  const setBtn = page.getByRole('button', { name: 'Set as default' });
  check(await setBtn.isEnabled(), 'Set as default is on for a line on counts');
  await page.evaluate(async () => {
    const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'First');
    for (const [d, n] of [['high', 3], ['medium', 2], ['low', 1]]) { w.setTrainCount(l.id, d, n); await new Promise((r) => setTimeout(r, 700)); }
  });
  await page.waitForFunction(() => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === 'First'); return l && w.lineStats[l.id]?.trainsNeeded?.join() === '3,2,1'; }, undefined, { timeout: 30000 });
  const counts = await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'First'); return [...(w.trainsShown(l.id) ?? window.tw.world.value.lineStats[l.id].trainsNeeded)]; });
  await setBtn.click();
  await page.waitForTimeout(2500);
  const sw = await headwaysOf('First');
  check(Array.isArray(sw) && sw.some((x) => x > 0) && (await defaultNow())?.join() === sw.join(), 'Set as default on counts: the line switched to headway and it is the default', JSON.stringify({ sw, def: await defaultNow() }));
  const after = await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'First'); return [...window.tw.world.value.lineStats[l.id].trainsNeeded]; });
  check(counts.join() === after.join(), 'same trains as before the switch', `${counts} -> ${after}`);
  check(await page.getByRole('button', { name: 'Use default' }).isDisabled(), 'Use default is off: the line is the default');
  await g.shot('t3588_window.png', { element: 'dialog.sched-panel' });
  await page.locator('.sched-default').scrollIntoViewIfNeeded();
  await page.locator('dialog.sched-panel').screenshot({ path: join(REVIEW, 't3588.png') });
  await page.evaluate(async () => {
    const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'First');
    w.setHeadway(l.id, 'high', 300); await new Promise((r) => setTimeout(r, 700));
    w.setHeadway(l.id, 'medium', 480); await new Promise((r) => setTimeout(r, 700));
    w.setHeadway(l.id, 'low', 900);
  });
  await page.waitForTimeout(1500);
  const set = await headwaysOf('First');
  check(JSON.stringify(set) === '[300,480,900]', 'the first line runs 5, 8 and 15 min', JSON.stringify(set));
  await setBtn.click();
  await page.waitForTimeout(500);
  check(JSON.stringify(await defaultNow()) === '[300,480,900]', 'Set as default stores them', JSON.stringify(await defaultNow()));
  check(/5 min, 8 min and 15 min/.test((await windowText()) ?? ''), 'the window shows them', await windowText());
  check(await setBtn.isDisabled(), 'Set as default is off while they already are the default');
  await g.shot('t3543_window.png', { element: 'dialog.sched-panel' });
  await closeWindow();

  // a line drawn now starts with them
  await g.act.newLine([A, B], { name: 'Second', zoom: 15 });
  await page.waitForTimeout(1500);
  const second = await headwaysOf('Second');
  check(JSON.stringify(second) === '[300,480,900]', 'a new line starts with the default headways', JSON.stringify(second));
  const shown = await g.see.line();
  check(shown?.schedule?.map((r) => r.headway).join(',') === '5 min,8 min,15 min', 'its panel shows 5, 8 and 15 min', JSON.stringify(shown?.schedule));

  // save and open again
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  const header = await page.evaluate(async (data) => (await import('/src/workers/saveFile.ts')).decodeSave(Uint8Array.from(data)).then((d) => d.header.newHeadway?.rail ?? null), Array.from(bytes));
  check(JSON.stringify(header) === '[300,480,900]', 'the exported save carries the default', JSON.stringify(header));
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length >= 2, undefined, { timeout: 300000 });
  await page.waitForTimeout(2500);
  check(JSON.stringify(await defaultNow()) === '[300,480,900]', 'opened again: the default holds', JSON.stringify(await defaultNow()));
  await g.act.camera(C, 13);
  await g.act.newLine([B, D], { name: 'Third', zoom: 15 });
  await page.waitForTimeout(1500);
  check(JSON.stringify(await headwaysOf('Third')) === '[300,480,900]', 'a line drawn after the reload starts with it', JSON.stringify(await headwaysOf('Third')));

  // weekends: Apply to all lines asks, says it covers new lines, and a line drawn after starts on it
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Third'); w.setWeekendShare(l.id, 60); });
  await page.waitForTimeout(1200);
  await page.getByRole('button', { name: 'Apply to all lines' }).click();
  await page.waitForTimeout(400);
  const askText = await page.evaluate(() => document.body.innerText.match(/Set weekends to .*new lines too\?/)?.[0] ?? null);
  check(!!askText, 'the question says new lines too', String(askText));
  await page.getByRole('button', { name: 'Apply to all lines' }).last().click();
  await page.waitForTimeout(1500);
  await closeWindow();
  await g.act.newLine([A, B], { name: 'Weekender', zoom: 15 });
  await page.waitForTimeout(1500);
  const wshare = await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Weekender'); return w.weekendShown(l.id)?.share ?? null; });
  check(wshare === 60, 'a new line starts on the weekend percent set for all lines', String(wshare));

  // stepping one weekend headway leaves the other levels as shown, in whole minutes
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Third'); w.setWeekendShare(l.id, 70); });
  await page.waitForTimeout(1500);
  const shownRest = () => page.evaluate(async () => { const l = window.tw.world.value.save.lines.find((x) => x.name === 'Third'), st = window.tw.world.value.lineStats[l.id]; return st.restTph.map((t) => (t > 0 ? Math.round(3600 / t) : 0)); });
  const wkBefore = await shownRest();
  await page.evaluate(async () => {
    const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Third'), st = window.tw.world.value.lineStats[l.id];
    const h = st.restTph[0] > 0 ? 3600 / st.restTph[0] : 0;
    w.stepWeekendHeadway(l.id, 'high', 1, st.roundTripByLevel[0], 0, false, 99, h);
  });
  await page.waitForTimeout(1500);
  const wkAfter = await shownRest();
  check(wkBefore[1] === wkAfter[1] && wkBefore[2] === wkAfter[2] && wkAfter[0] % 60 === 0 && wkAfter[0] !== wkBefore[0], 'a weekend headway step leaves the other two levels as shown, whole minutes', JSON.stringify({ wkBefore, wkAfter }));
  await closeWindow();

  // Use default: a line on counts, then one with its own headways; one undoable edit each
  await g.act.newLine([A, D], { name: 'Fourth', zoom: 15 });
  await page.waitForTimeout(1500);
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Fourth'); w.setAutoTrains(l.id, false); });
  await page.waitForTimeout(1500);
  check((await headwaysOf('Fourth')) === null, 'the fourth line is on counts');
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  const useBtn = page.getByRole('button', { name: 'Use default' });
  check(await useBtn.isEnabled(), 'Use default is on for a line on counts');
  await useBtn.click();
  await page.waitForTimeout(2000);
  check(JSON.stringify(await headwaysOf('Fourth')) === '[300,480,900]', 'Use default gives the line the default headways', JSON.stringify(await headwaysOf('Fourth')));
  check(await useBtn.isDisabled(), 'Use default is off once the line matches');
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await page.waitForTimeout(2000);
  check((await headwaysOf('Fourth')) === null, 'one undo puts the line back on counts', JSON.stringify(await headwaysOf('Fourth')));
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'), l = window.tw.world.value.save.lines.find((x) => x.name === 'Fourth'); w.setAutoTrains(l.id, true); await new Promise((r) => setTimeout(r, 1500)); w.setHeadway(l.id, 'high', 420); });
  await page.waitForTimeout(2000);
  check(await useBtn.isEnabled(), 'Use default is on for a line with other headways', JSON.stringify(await headwaysOf('Fourth')));
  await useBtn.click();
  await page.waitForTimeout(2000);
  check(JSON.stringify(await headwaysOf('Fourth')) === '[300,480,900]', 'Use default replaces the other headways', JSON.stringify(await headwaysOf('Fourth')));
  await page.evaluate(() => window.tw.client.edit({ op: 'undo' }));
  await page.waitForTimeout(2000);
  check((await headwaysOf('Fourth'))?.[0] === 420, 'one undo restores the line\'s own headways', JSON.stringify(await headwaysOf('Fourth')));
  await closeWindow();

  const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
  check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
} catch (e) {
  check(false, `ran to the end: ${e.stack ?? e.message}`);
} finally {
  await g.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
