// T-3414 (the issues list sits 4 px above its toggle) and
// T-3416: a very long line or group name stays editable (the pencil and the box stay reachable in the panel
// and the Lines list). Real save; shots in artifacts/trainworld/t3416/. node trainworld/checks/t3416-long-names.mjs [url]
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6131/';
const OUT = join(ARTIFACTS, 't3416');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log((ok ? 'PASS' : 'FAIL') + ' ' + what); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3416-', size: [1400, 1000], args: GL });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(BASE + '?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 4, undefined, { timeout: 120000 });
  await page.waitForTimeout(3000);
  await page.evaluate(async () => { window.__m = { ui: await import('/src/game/ui.ts'), groups: await import('/src/game/lineGroups.ts') }; });
  const long = 'W'.repeat(120);
  const inView = (sel) => page.evaluate((sel) => {
    const e = document.querySelector(sel); if (!e) return null;
    const r = e.getBoundingClientRect(), p = document.querySelector('#inspector').getBoundingClientRect();
    return { l: r.left, r: r.right, pr: p.right, vis: r.right <= p.right + 1 && r.width > 0 };
  }, sel);
  // a line
  const lineId = await page.evaluate(() => window.tw.world.value.save.lines[0].id);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, lineId);
  await page.waitForTimeout(800);
  await page.click('#inspector .title > .btn.icon:last-child');
  await page.keyboard.type(long);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1200);
  await page.locator('#inspector').screenshot({ path: join(OUT, 'line-panel.png'), timeout: 5000 });
  const pen = await inView('#inspector .title > .btn.icon:last-child');
  check(pen?.vis, 'line panel: pencil inside the panel ' + JSON.stringify(pen));
  const nm = await page.evaluate(() => { const e = document.querySelector('#inspector .title .name'); return e && { sw: e.scrollWidth, cw: e.clientWidth, pr: document.querySelector('#inspector').getBoundingClientRect().right, r: e.getBoundingClientRect().right }; });
  console.log('name box', JSON.stringify(nm));
  check(nm && nm.sw <= nm.cw + 1, 'line panel: the name text fits its box');
  await page.click('#inspector .title > .btn.icon:last-child');
  await page.waitForTimeout(300);
  await page.locator('#inspector').screenshot({ path: join(OUT, 'line-panel-edit.png'), timeout: 5000 });
  const ed = await inView('#inspector .title > .btn.icon:last-child');
  check(ed?.vis, 'line panel editing: done button inside the panel ' + JSON.stringify(ed));
  await page.keyboard.press('Escape');
  // the Lines tab
  await page.evaluate(() => { window.__m.ui.tab.value = 'lines'; window.tw.selection.value = null; });
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(OUT, 'lines-tab.png'), clip: { x: 0, y: 0, width: 520, height: 600 } });
  // a group of two lines made here
  await page.evaluate(async () => { const w = await import('/src/game/world.ts'); const ls = window.tw.world.value.save.lines; await w.edit({ op: 'lineGroup', lines: [ls[3].num, ls[4].num], group: 'new' }); });
  await page.waitForTimeout(1000);
  const g = await page.evaluate(() => { const m = window.__m.groups.lineGroups(window.tw.world.value.save.lines); const e = [...m.entries()].find(([, ls]) => ls.length >= 1); return e ? e[0] : null; });
  console.log('group', g);
  if (g !== null) {
    await page.evaluate((g) => { window.tw.selection.value = { kind: 'group', group: g }; }, g);
    await page.waitForTimeout(800);
    await page.click('#inspector .title > .btn.icon:last-child');
    await page.keyboard.type(long);
    await page.keyboard.press('Enter');
    await page.waitForTimeout(1200);
    await page.locator('#inspector').screenshot({ path: join(OUT, 'group-panel.png'), timeout: 5000 });
    const gp = await inView('#inspector .title > .btn.icon:last-child');
    check(gp?.vis, 'group panel: buttons inside the panel ' + JSON.stringify(gp));
    const nb = await page.$$eval('#inspector .title .btn', (b) => b.map((x) => { const r = x.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.right), Math.round(r.width)]; }));
    console.log('title buttons', JSON.stringify(nb));
    await page.evaluate(() => { window.__m.ui.tab.value = 'lines'; window.tw.selection.value = null; });
    await page.waitForTimeout(800);
    await page.screenshot({ path: join(OUT, 'lines-tab-group.png'), clip: { x: 0, y: 0, width: 520, height: 600 } });
  }
  // T-3414: the open issues list and its toggle
  await page.evaluate(() => { window.tw.selection.value = null; });
  const tb = await page.$('.trouble-btn');
  if (tb) {
    if (!(await page.$('.trouble-list'))) await tb.click();
    await page.waitForTimeout(500);
    const gap = await page.evaluate(() => { const l = document.querySelector('.trouble-list')?.getBoundingClientRect(), b = document.querySelector('.trouble-btn')?.getBoundingClientRect(); return l && b ? Math.round((b.top - l.bottom) * 10) / 10 : null; });
    check(gap !== null && Math.abs(gap - 4) <= 0.6, 'issues list is 4 px above its toggle: ' + gap);
    await page.locator('.trouble').screenshot({ path: join(OUT, 'trouble-gap.png'), timeout: 5000 });
  } else console.log('no issues in this save, trouble gap not checked');
  check(errors.length === 0, 'no console errors ' + errors.slice(0, 2).join(' | '));
} finally { await chrome.close(); }
process.exit(fail ? 1 : 0);
