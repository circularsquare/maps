// T-1509: edits that change a line's length (a stop dropped, put back, undone) post the strokes as a patch
// with moved runs, not whole; ?patch=check compares each merged network with the whole one and logs an error
// when they differ. On the real save (about 3 minutes).
//   node trainworld/checks/t1509-line-moves.mjs [dev url]     (from the maps root)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6145/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = await launchChrome({ prefix: 'tw-t1509-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=1&paused=1&patch=check&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client && window.tw.tools, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 50, undefined, { timeout: 300000 });
  await wait(5000);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await wait(2000);
  const lines = await page.evaluate(() => {
    const w = window.tw.world.value;
    return w.save.lines.filter((l) => l.stops.length >= 5 && !w.lineStats[l.id].bus).slice(0, 12).map((l) => l.num);
  });
  const modes = [];
  const edit = async (op, early = false) => {
    const r = await page.evaluate(([op, early]) => window.tw.client.edit(op, early), [op, early]);
    await wait(1500);
    const e = await page.evaluate(() => { const l = window.tw.client.editLog; return l[l.length - 1]?.worker; });
    modes.push(e?.modes);
    return r;
  };
  let done = 0;
  for (const num of lines) {
    const stops = await page.evaluate((num) => { const w = window.tw.world.value; return w.save.lines.find((l) => l.num === num).stops.map(Number); }, num);
    const a = await edit({ op: 'dropStop', line: num, index: 1 });
    if (a.ok) {
      done++;
      await edit({ op: 'undo' }, true);
      await edit({ op: 'redo' }, true);
      await edit({ op: 'undo' }, true);
    }
  }
  check(done >= 6, 'line edits applied', `${done} of ${lines.length}`);
  check(!errors.some((e) => /T-1381 check|T-1029/.test(e)), 'merged strokes equal the whole ones', errors.filter((e) => /T-1381|T-1029/.test(e)).slice(0, 3).join(' | '));
  const lineModes = modes.filter((m) => m !== undefined).map((m) => m % 10);
  console.log('lines modes (0 whole, 1 same, 2 patch):', lineModes.join(' '));
  check(lineModes.filter((m) => m === 2).length > lineModes.filter((m) => m === 0).length, 'most line edits post a patch');
  check(errors.filter((e) => !/T-1381|T-1029/.test(e)).length === 0, 'no other page errors', errors.slice(0, 3).join(' | '));
} finally { await chrome.close(); }
process.exit(fail ? 1 : 0);
