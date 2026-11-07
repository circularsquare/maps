// Run from maps: node trainworld/checks/t2703-storeread.mjs <url>      (about a minute)
// T-2703: a saved game whose read fails at startup. The real save goes into a slot of a throwaway
// profile, then the page opens it with reads made to fail (`?debug=1&readFails=N`, saveFile.ts):
// 1. every try failing: the failure screen gives the storage reason (`.load-why`), and the slot is
//    byte-identical afterwards, with no other slot written;
// 2. one failure, then success: the slot's game opens (its 109 lines);
// 3. T-2705: the start menu with its slot list failing every try says so (`.start-unread`), not
//    opening on the new game sheet; failing once, it shows the saved game.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { launchChrome, REAL_SAVE } from './lib.mjs';

const url = (process.argv[2] ?? 'http://localhost:6171/').replace(/\/?$/, '/');
const KEY = 'slot-t2703';
const { ctx, close } = await launchChrome({ prefix: 'tw-storeread-', size: [1200, 800] });
let failed = false;

/** the whole store as JSON (bytes as arrays), in a page of the app's origin */
const dump = (page) => page.evaluate(() => new Promise((res, rej) => {
  const r = indexedDB.open('anitabuilder', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('saves');
  r.onerror = () => rej(r.error);
  r.onsuccess = () => {
    const s = r.result.transaction('saves', 'readonly').objectStore('saves');
    const keys = s.getAllKeys(), vals = s.getAll();
    vals.onsuccess = () => {
      const out = {};
      keys.result.forEach((k, i) => { out[k] = JSON.stringify(vals.result[i], (_, v) => v instanceof Uint8Array ? Array.from(v) : v); });
      r.result.close();
      res(out);
    };
    vals.onerror = () => rej(vals.error);
  };
}));

try {
  const bytes = Array.from(await readFile(REAL_SAVE));
  const blank = await ctx.newPage();
  await blank.goto(url + '@vite/client');
  await blank.evaluate(({ key, data }) => new Promise((res, rej) => {
    const r = indexedDB.open('anitabuilder', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('saves');
    r.onerror = () => rej(r.error);
    r.onsuccess = () => {
      const tx = r.result.transaction('saves', 'readwrite');
      tx.objectStore('saves').put({ bytes: Uint8Array.from(data), at: 1, day: 0, name: 'T-2703 probe' }, key);
      tx.oncomplete = () => { r.result.close(); res(); };
      tx.onerror = () => rej(tx.error);
    };
  }), { key: KEY, data: bytes });
  const before = await dump(blank);
  assert.ok(before[KEY], 'the probe slot was written');

  // 1
  const p1 = await ctx.newPage();
  const tries = [];
  p1.on('console', (m) => { if (m.text().startsWith('T-2703:')) tries.push(m.text()); });
  const t0 = Date.now();
  await p1.goto(`${url}?debug=1&paused=1&slot=${KEY}&readFails=99&v=${t0}`);
  const why = await p1.waitForFunction(() => document.querySelector('.loading-screen .load-why')?.textContent, undefined, { timeout: 120000, polling: 250 }).then((h) => h.jsonValue());
  const waited = Date.now() - t0;
  assert.match(why, /browser storage/, `the reason: ${why}`);
  assert.ok(tries.some((l) => l.includes('after 4 tries')), `the tries in the console: ${JSON.stringify(tries)}`);
  await p1.waitForTimeout(3000); // anything the failed page might still write
  await p1.close();
  const after = await dump(blank);
  assert.deepEqual(Object.keys(after).sort(), Object.keys(before).sort(), 'no slot added or removed');
  assert.ok(after[KEY] === before[KEY], 'the slot is byte-identical');
  console.log(`PASS  a read that keeps failing stops with its reason, the slot untouched  (${waited} ms; "${why}")`);

  // 2
  const p2 = await ctx.newPage();
  await p2.goto(`${url}?debug=1&paused=1&slot=${KEY}&readFails=1&v=${Date.now()}`);
  const lines = await p2.waitForFunction(() => {
    if (document.querySelector('.loading-screen .load-why')) return -1;
    const w = window.tw?.world?.value;
    return w && w.save.lines.length > 0 ? w.save.lines.length : null;
  }, undefined, { timeout: 180000, polling: 250 }).then((h) => h.jsonValue());
  assert.equal(lines, 109, 'lines of the opened game (-1: the failure screen)');
  console.log(`PASS  one failed read then success opens the slot's game  (${lines} lines)`);
  await p2.close();

  // 3 (T-2705): the start menu, its list of saved games failing every try, then once
  const menu = async (fails) => {
    const p = await ctx.newPage();
    await p.goto(`${url}?debug=1&listFails=${fails}&v=${Date.now()}`);
    await p.waitForSelector('.start-menu .start-new', { timeout: 120000 });
    await p.waitForTimeout(300);
    const r = await p.evaluate(() => ({ unread: document.querySelector('.start-unread')?.textContent ?? null, cards: document.querySelectorAll('.start-save').length, sheet: !!document.querySelector('.new-game-box') }));
    await p.close();
    return r;
  };
  const m1 = await menu(99);
  assert.ok(m1.unread && !m1.sheet && m1.cards === 0, `the menu with its list failing: ${JSON.stringify(m1)}`);
  console.log(`PASS  a slot list that keeps failing says so on the menu  ("${m1.unread}")`);
  const m2 = await menu(1);
  assert.ok(!m2.unread && m2.cards === 1, `the menu with one failed list: ${JSON.stringify(m2)}`);
  console.log(`PASS  one failed list then success shows the saved game  (${m2.cards} card)`);
  await blank.close();
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message.split('\n')[0]);
} finally {
  await close();
}
console.log(failed ? 't2703-storeread: FAILED' : 't2703-storeread: PASSED');
process.exit(failed ? 1 : 0);
