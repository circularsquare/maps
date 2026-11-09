// T-1812: a symbol typed by hand sticks. Real clicks and typing in the line editor on the real save:
// set the symbol (then Done), rename then set it, set it then rename, a rename of a line whose symbol
// was set, a line with no symbol of its own following its name, and an export and reload. From the
// maps root:
//   node trainworld/checks/t1812-symbol.mjs [url]   (about a minute)
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, GL, launchChrome, collectErrors, importSave } from '../lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6187/';
const OUT = join(ARTIFACTS, 't1812');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const errors = [];
const chrome = await launchChrome({ prefix: 'tw-t1812s-', size: [1400, 900], args: GL, acceptDownloads: true });
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(4000);
  const look = (name) => page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.name === n); return l ? { num: l.num, name: l.name, symbol: l.symbol ?? '', letter: l.letter } : null; }, name);
  const byNum = (num) => page.evaluate((n) => { const l = window.tw.world.value.save.lines.find((x) => x.num === n); return l ? { num: l.num, name: l.name, symbol: l.symbol ?? '', letter: l.letter } : null; }, num);
  const open = async (num) => {
    await page.evaluate((n) => { window.tw.selection.value = { kind: 'line', line: String(n) }; }, num);
    await page.waitForSelector('#inspector .badge-edit', { timeout: 30000 });
    await page.waitForTimeout(600);
    await page.dblclick('#inspector .badge-edit');
    await page.waitForSelector('#inspector input[aria-label="Line symbol"]');
  };
  const typeIn = async (sel, text, delay = 0) => { await page.click(sel); await page.keyboard.press('Control+A'); await page.keyboard.type(text, { delay }); };
  const SYM = '#inspector input[aria-label="Line symbol"]', NAME = '#inspector input.name-edit', DONE = '#inspector button[aria-label="Done"]';
  const settle = (ms = 1500) => page.waitForTimeout(ms);
  const sample = await page.evaluate(() => window.tw.world.value.save.lines.slice(0, 8).map((l) => [l.num, l.name, l.symbol ?? '', l.letter]));
  console.log('lines', JSON.stringify(sample));
  // choose lines with no symbol of their own
  const pool = await page.evaluate(() => window.tw.world.value.save.lines.filter((l) => !l.symbol && /^\d$/.test(l.name)).map((l) => l.num));
  console.log('pool', pool);
  if (pool.length < 6) throw new Error('too few plain lines');
  const [A, B, C, D, E, F] = pool;

  // 1. the symbol alone, then Done
  await open(A);
  await typeIn(SYM, 'Q');
  await page.click(DONE);
  await settle();
  let r = await byNum(A);
  check(r.symbol === 'Q' && r.letter === 'Q', `1 symbol then Done: ${JSON.stringify(r)}`);

  // 2. a rename, then the symbol typed at once and Done at once, then after a beat
  for (const [line, wait, sym] of [[B, 0, 'R'], [C, 200, 'T'], [D, 500, 'U']]) {
    await open(line);
    await typeIn(NAME, 'Alpha' + line);
    await page.click(SYM);
    await page.keyboard.press('Control+A');
    await page.keyboard.type(sym);
    await page.waitForTimeout(wait);
    await page.click(DONE);
    await settle(2500);
    r = await byNum(line);
    check(r.symbol === sym && r.name === 'Alpha' + line, `2 rename then symbol, Done after ${wait} ms: ${JSON.stringify(r)}`);
  }

  // 3. the symbol, then the name, Enter
  await open(E);
  await typeIn(SYM, 'S');
  await typeIn(NAME, 'Beta');
  await page.keyboard.press('Enter');
  await settle(2500);
  r = await byNum(E);
  check(r.symbol === 'S' && r.name === 'Beta', `3 symbol then rename: ${JSON.stringify(r)}`);

  // 4. a line with a hand symbol renamed from the pencil: it keeps the symbol
  await page.evaluate((n) => { window.tw.selection.value = { kind: 'line', line: String(n) }; }, A);
  await page.waitForSelector('#inspector .title button[aria-label="Rename"]');
  await page.waitForTimeout(600);
  await page.click('#inspector .title button[aria-label="Rename"]');
  await page.keyboard.press('Control+A');
  await page.keyboard.type('Gamma');
  await page.keyboard.press('Enter');
  await settle(2500);
  r = await byNum(A);
  check(r.symbol === 'Q' && r.letter === 'Q' && r.name === 'Gamma', `4 rename of a hand symbol: ${JSON.stringify(r)}`);

  // 5. a symbol typed equal to the letter its name gives is still a hand symbol
  await open(F);
  const auto = (await byNum(F)).letter;
  await typeIn(SYM, auto);
  await page.click(DONE);
  await settle();
  await page.click('#inspector .title button[aria-label="Rename"]');
  await page.keyboard.press('Control+A');
  await page.keyboard.type('Zeta');
  await page.keyboard.press('Enter');
  await settle(2500);
  r = await byNum(F);
  check(r.symbol === auto && r.letter === auto && r.name === 'Zeta', `5 symbol equal to the auto letter "${auto}" then rename: ${JSON.stringify(r)}`);

  // 6. a line never given a symbol follows its name
  const G = (await page.evaluate((used) => window.tw.world.value.save.lines.find((l) => !l.symbol && /^\d$/.test(l.name) && !used.includes(l.num))?.num, pool.slice(0, 6)));
  await page.evaluate((n) => { window.tw.selection.value = { kind: 'line', line: String(n) }; }, G);
  await page.waitForSelector('#inspector .title button[aria-label="Rename"]');
  await page.waitForTimeout(600);
  await page.click('#inspector .title button[aria-label="Rename"]');
  await page.keyboard.press('Control+A');
  await page.keyboard.type('Omega');
  await page.keyboard.press('Enter');
  await settle(2500);
  r = await byNum(G);
  check(r.symbol === '' && r.letter === 'O' && r.name === 'Omega', `6 no symbol follows the name: ${JSON.stringify(r)}`);

  // 7. clearing the symbol hands the badge back to the name
  await open(A);
  await typeIn(SYM, '');
  await page.keyboard.press('Backspace');
  await page.click(DONE);
  await settle(2000);
  r = await byNum(A);
  check(r.symbol === '' && r.letter === 'G', `7 blank symbol restores the name's letter: ${JSON.stringify(r)}`);
  await open(A);
  await typeIn(SYM, 'Q');
  await page.click(DONE);
  await settle();

  // 8. export and reload
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const file = await (await dl).path();
  const bytes = await readFile(file);
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const after = await Promise.all([A, B, E, F, G].map(byNum));
  check(after[0].symbol === 'Q' && after[1].symbol === 'R' && after[2].symbol === 'S' && after[3].symbol === auto && after[4].symbol === '', `8 reload keeps the hand symbols: ${JSON.stringify(after.map((x) => [x.name, x.symbol, x.letter]))}`);
  await open(A);
  await page.screenshot({ path: join(OUT, 'symbol-editor.png'), clip: { x: 0, y: 380, width: 420, height: 300 } });
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
