// T-3543: one schedule button, and default headways. A new game in New York with a line built through the
// harness (lib/play.mjs): the line panel has no copy or paste schedule button and one schedule button, a size
// bigger than the chart button beside it. In the schedule window the line's headways are set (the Headway switch),
// "Set as default" makes them the default (the window shows them), a second line drawn after starts with those
// headways. The game is exported and opened again: the default holds (the save's header carries it) and a line
// drawn then starts with it. "Use game default" clears it and the next line starts on counts. Shots: the line
// panel (review/t3543.png) and the schedule window. From the tree root, on a Vite of your own:
//   node trainworld/checks/t3543-default-headways.mjs [dev url]      (about three minutes)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, importSave } from './lib.mjs';
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6164/';
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
const defaultNow = () => page.evaluate(async () => (await import('/src/game/newHeadway.ts')).newHeadway.value);
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
  check(!!sched && !!other && sched.w > other.w && sched.w - other.w <= 8, 'the schedule button is a little bigger than the chart button', JSON.stringify(buttons));
  await g.shot('t3543_line_panel_full.png', { element: '#inspector' });
  await page.evaluate(() => document.querySelector('#inspector .h .sched-copy')?.scrollIntoView({ block: 'center' }));
  const head = await page.evaluate(() => { const r = document.querySelector('#inspector .h .sched-copy').parentElement.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  await page.screenshot({ path: join(REVIEW, 't3543.png'), clip: { x: Math.max(0, head.x - 12), y: Math.max(0, head.y - 60), width: Math.min(420, head.w + 24), height: 220 } });

  // the window: Headway, three headways, then Set as default
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  check(/Game default/.test((await windowText()) ?? ''), 'the window says Game default', await windowText());
  const setBtn = page.getByRole('button', { name: 'Set as default' });
  check(await setBtn.isDisabled(), 'Set as default is off while the line runs on counts');
  await page.locator('dialog.sched-panel').getByRole('button', { name: 'Headway', exact: true }).first().click();
  await page.waitForTimeout(800);
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
  const header = await page.evaluate(async (data) => (await import('/src/workers/saveFile.ts')).decodeSave(Uint8Array.from(data)).then((d) => d.header.newHeadway ?? null), Array.from(bytes));
  check(JSON.stringify(header) === '[300,480,900]', 'the exported save carries the default', JSON.stringify(header));
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length >= 2, undefined, { timeout: 300000 });
  await page.waitForTimeout(2500);
  check(JSON.stringify(await defaultNow()) === '[300,480,900]', 'opened again: the default holds', JSON.stringify(await defaultNow()));
  await g.act.camera(C, 13);
  await g.act.newLine([B, D], { name: 'Third', zoom: 15 });
  await page.waitForTimeout(1500);
  check(JSON.stringify(await headwaysOf('Third')) === '[300,480,900]', 'a line drawn after the reload starts with it', JSON.stringify(await headwaysOf('Third')));

  // back to the game's own default
  await g.act.click('Detailed schedule');
  await page.waitForSelector('dialog.sched-panel .sched-default', { timeout: 15000 });
  await page.getByRole('button', { name: 'Use game default' }).click();
  await page.waitForTimeout(500);
  check((await defaultNow()) === null && /Game default/.test((await windowText()) ?? ''), 'Use game default clears it', await windowText());
  await closeWindow();
  await g.act.newLine([A, D], { name: 'Fourth', zoom: 15 });
  await page.waitForTimeout(1500);
  check((await headwaysOf('Fourth')) === null, 'the next line starts on counts again');

  const errs = g.pageErrors().filter((e) => !/WebGL|GPU|ResizeObserver/.test(e));
  check(errs.length === 0, `no page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
} catch (e) {
  check(false, `ran to the end: ${e.stack ?? e.message}`);
} finally {
  await g.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
