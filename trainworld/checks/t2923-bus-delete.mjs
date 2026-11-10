// T-2923: a built bus line's delete dialog (Keep leaves the line; Delete line removes it), and undo
// after deleting a built bus (final: the history is cleared, undo does nothing); also the paste hover naming the line pasted
// on ("buses"), not the line copied from ("boats"). On the real save.
//   node trainworld/checks/t2923-bus-delete.mjs [dev url]     (from the maps root; about 2 minutes)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6430/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t2923-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
const line = (num) => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return l ? { built: !!w.lineStats[l.id].bus?.built, id: l.id } : null; }, num);
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(3000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const made = await page.evaluate(async () => {
    const w = window.tw.world.value;
    const rail = w.save.lines.find((l) => l.stops.length >= 6 && !w.lineStats[l.id].bus);
    const pts = rail.stops.slice(0, 5).map((s) => w.nodes.get(s) ?? w.nodes.get(Number(s))).filter(Boolean).flatMap((n) => [n.x, n.y, 1]);
    const a = await window.tw.client.edit({ op: 'addBusLine', route: pts, name: 'Check bus', colour: '#c0392b' });
    const b = a.ok ? await window.tw.client.edit({ op: 'constructLine', line: a.line }) : a;
    return { ok: a.ok && b.ok, line: a.line, issues: [...(a.issues ?? []), ...(b.issues ?? [])] };
  });
  check(made.ok, 'a built bus line', JSON.stringify(made));
  const num = made.line;
  await wait(1500);
  check((await line(num))?.built === true, 'the line is built');
  const select = () => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); window.tw.selection.value = { kind: 'line', line: l.id }; }, num);
  const del = () => page.locator('button.danger', { hasText: /^Delete line$/ }).first();
  const dialog = page.locator('dialog.confirmation');

  // Keep
  await select();
  await del().waitFor({ timeout: 20000 });
  await del().click();
  await dialog.waitFor({ timeout: 10000 });
  check(/Nothing is refunded/.test(await dialog.innerText()), 'the dialog warns that nothing is refunded');
  await dialog.locator('button', { hasText: /^Keep line$/ }).click();
  await wait(1000);
  check(!(await dialog.count()), 'Keep closes the dialog');
  check((await line(num))?.built === true, 'Keep leaves the built line');

  // Delete, then undo
  await del().click();
  await dialog.waitFor({ timeout: 10000 });
  await dialog.locator('button.danger').click();
  for (let k = 0; k < 50 && (await line(num)); k++) await wait(200);
  check(!(await line(num)), 'Delete line removes the built line');
  check(await page.evaluate(() => window.tw.selection.value === null), 'the selection is cleared');
  // final, like constructed track (SPEC 6.4): the undo history is cleared, so undo does nothing
  check(await page.evaluate(() => !window.tw.world.value.canUndo), 'deleting a built bus clears the undo history');
  const u = await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true));
  await wait(1000);
  check(!u.ok && !(await line(num)), 'undo then changes nothing', JSON.stringify({ ok: u.ok }));
  // a planned bus line is not final: it goes at once and undo brings it back
  const planned = await page.evaluate(async () => {
    const w = window.tw.world.value, E = (o) => window.tw.client.edit(o, o.op === 'undo');
    const rail = w.save.lines.find((l) => l.stops.length >= 6 && !w.lineStats[l.id].bus);
    const pts = rail.stops.slice(0, 5).map((s) => w.nodes.get(s) ?? w.nodes.get(Number(s))).filter(Boolean).flatMap((n) => [n.x, n.y, 1]);
    const a = await E({ op: 'addBusLine', route: pts, name: 'Planned', colour: '#c0392b' });
    const r = await E({ op: 'removeLine', line: a.line });
    const u = await E({ op: 'undo' });
    return { r: r.ok, u: u.ok, line: a.line };
  });
  await wait(1000);
  check(planned.r && planned.u && !!(await line(planned.line)), 'undo brings a planned bus line back', JSON.stringify(planned));

  // the paste hover follows the line pasted on
  const text = await page.evaluate(async (num) => {
    const m = await import('/src/game/scheduleCopy.ts');
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num) ?? w.save.lines.find((x) => w.lineStats[x.id].bus), st = w.lineStats[l.id];
    const boat = { from: 'Ferry', headway: null, trains: [4, 3, 2], cars: 1, fast: false, vehicle: 'boat' };
    return { bus: m.scheduleText(boat, false, m.vehicleOf(st)), own: m.scheduleText(boat, false) };
  }, num);
  check(/buses/.test(text.bus) && !/boats/.test(text.bus), 'a ferry copy pasted on a bus says buses', text.bus);
  check(/boats/.test(text.own), 'a ferry copy on its own says boats', text.own);
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} finally { await chrome.close(); }
process.exit(fail ? 1 : 0);
