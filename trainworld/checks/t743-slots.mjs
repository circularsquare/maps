// T-743, T-744: New game, Open a file and a backup's restore never write over a saved game, and each
// slot keeps two rotating backups. Through the saves panel's own buttons (each reloads the page into
// its game, T-1196): the real save loaded from a
// file opens in a slot of its own; fares changed there twice, the backups (every 4 s here, `?backupMs=`)
// rotate; "Open as a copy" (under Earlier versions) opens the older backup in a new slot; New game opens an empty game in a
// new slot; recovered-day26.save loads into another; every slot left behind keeps its bytes. Deleting a
// slot deletes its backups. A panel screenshot with the backups goes to artifacts/trainworld/.
//   node trainworld/checks/t743-slots.mjs [url]     (from the maps root; default localhost:6150)
// Prints PASS or FAIL lines and exits 1 on a failure. A few minutes.
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const ART = fileURLToPath(new URL('../../artifacts/trainworld/', import.meta.url));
const RECOVERED = join(ART, 'recovered-day26.save');
const BACKUP_MS = 4000;

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t743-', size: [1300, 900], args: ANGLE });
try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const wait = (ms) => page.waitForTimeout(ms);
  /** the active slot, the listed slots and the active slot's backups */
  const state = () => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, slots: s.slots.map((x) => ({ key: x.key, name: x.name, at: x.at })), backups: s.backups };
  });
  /** a stored record's bytes as a SHA-256, and its header's fare and line count */
  const stored = (key) => page.evaluate(async (key) => {
    const S = await import('/src/workers/saveFile.ts');
    const r = await S.readStored(key);
    if (!r) return null;
    const h = [...new Uint8Array(await crypto.subtle.digest('SHA-256', r.bytes))].map((b) => b.toString(16).padStart(2, '0')).join('');
    const { header } = await S.decodeSave(r.bytes);
    return { hash: h.slice(0, 16), at: r.at, base: header.fares.base, lines: r.lines, thumb: !!r.thumb, name: r.name };
  }, key);
  const allKeys = () => page.evaluate(() => new Promise((res, rej) => {
    const o = indexedDB.open('anitabuilder');
    o.onsuccess = () => { const q = o.result.transaction('saves').objectStore('saves').getAllKeys(); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); };
  }));
  const settled = () => demandFinal(page, { strict: false, held: true, polling: 300 });
  const panel = async () => {
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
    await page.waitForSelector('.saves-panel .save-files');
    await wait(300);
  };
  /** after a game change (T-1196: the page reloads into it), the new active slot */
  const switched = async (from) => {
    for (const end = Date.now() + 180000; Date.now() < end; await wait(300)) {
      try {
        const a = await page.evaluate(async () => window.tw?.world?.value ? (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active : null);
        if (a && a !== from && a.startsWith('slot-')) return a;
      } catch { /* the page is reloading */ }
    }
    throw new Error(`no switch from ${from}`);
  };
  const loadFile = async (path) => {
    const from = (await state()).active;
    await panel();
    const [fc] = await Promise.all([page.waitForEvent('filechooser'), page.click('.saves-panel .save-files button:has-text("Open a file")')]);
    await fc.setFiles(path);
    const key = await switched(from);
    await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await settled();
    return key;
  };
  /** write the game now (as the page does when hidden) and let it land */
  const flush = async () => { await page.evaluate(() => window.tw.client.post({ kind: 'flush' })); await wait(1500); };
  const fares = async (base) => { await page.evaluate((base) => window.tw.client.post({ kind: 'edit', seq: 990000 + Math.round(base * 100), op: { op: 'fares', base, perKm: 0.1 } }), base); await flush(); };

  // a fresh profile with nothing saved opens the original slot, empty and unwritten
  await page.goto(`${BASE}?debug=1&paused=1&backupMs=${BACKUP_MS}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await wait(1500);
  const first = (await state()).active;

  // 1. Open a file: the real save in a slot of its own, named after the file
  const A = await loadFile(REAL_SAVE);
  let s = await state();
  const aSlot = s.slots.find((x) => x.key === A);
  const lines = await page.evaluate(() => window.tw.world.value.save.lines.length);
  report('load a file opens a new slot', A !== first && A.startsWith('slot-') && aSlot?.name === 'T-007-real-nyc' && lines === 109,
    `(from ${first} to ${A}, named "${aSlot?.name}", ${lines} lines)`);

  // 2. backups: the first at the slot's first write, then one per change at least BACKUP_MS apart, the older replaced
  // a change is backed up only when the newest backup is BACKUP_MS old by its own stamp, taken after the
  // write's encode and picture, so wait on the stamp and not on the clock (T-1410: under load a fixed wait fell short)
  const aged = async () => {
    for (const end = Date.now() + 60000; Date.now() < end; await wait(250)) {
      const newest = (await state()).backups[0];
      if (newest && await page.evaluate((at) => Date.now() - at > 500, newest.at + BACKUP_MS)) return;
    }
    throw new Error('no backup aged past the interval');
  };
  await aged();
  await fares(2);
  await aged();
  await fares(2.5);
  s = await state();
  const keys = await allKeys();
  const bk = await Promise.all(s.backups.map((b) => stored(b.key)));
  const aNow = await stored(A);
  report('two rotating backups', s.backups.length === 2 && bk[0]?.base === 2.5 && bk[1]?.base === 2 && aNow.base === 2.5 && bk.every((b) => !b.thumb)
    && !s.slots.some((x) => x.key.startsWith('backup:')) && keys.filter((k) => String(k).startsWith(`backup:${A}:`)).length === 2,
    `(backups ${s.backups.map((b, i) => `${b.key} fares ${bk[i]?.base}`).join(', ')}; slot fares ${aNow.base}; listed slots ${s.slots.length})`);

  // the panel shows them under the playing card, once its earlier versions are open (T-1395)
  await panel();
  await page.click('.save-slots li.on .save-versions');
  const rows = await page.locator('.save-slots li.on .save-backup').count();
  await page.screenshot({ path: join(ART, 't743-backups.png') });
  report('panel lists the backups', rows === 2, `(${rows} rows; ${join(ART, 't743-backups.png')})`);

  // 3. Open as a copy: the older backup opens in a new slot, the original untouched
  const aBefore = await stored(A);
  const older = s.backups[1].key;
  await page.click(`.save-backup[data-backup="${older}"] button`);
  const E = await switched(A);
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await settled();
  await flush();
  const e = await stored(E), aAfter = await stored(A);
  report('restore as a copy', E !== A && e?.base === 2 && /^T-007-real-nyc at /.test(e?.name ?? '') && aAfter.hash === aBefore.hash,
    `(new slot "${e?.name}" fares ${e?.base}; original ${aBefore.hash} -> ${aAfter.hash})`);

  // 4. New game from the panel: an empty game in a new slot; the game left keeps its bytes
  const eBefore = await stored(E);
  await panel();
  await page.click('.saves-panel .save-files button:has-text("New game")');
  await page.click('.saves-panel .new-game .start-new'); // T-728: the new game sheet, its defaults
  const B = await switched(E);
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 0, undefined, { timeout: 60000 });
  await wait(1500);
  const eAfter = await stored(E);
  s = await state();
  report('new game opens a new slot', B !== E && B.startsWith('slot-') && eAfter?.hash === eBefore.hash && s.slots.some((x) => x.key === E) && (await stored(A)).hash === aBefore.hash,
    `(from ${E} to ${B}; left game ${eBefore.hash} -> ${eAfter?.hash}; ${s.slots.length} slots listed)`);

  // 5. Open a file again, from the new game: Anita's recovered game in another slot, named after its file
  const C = await loadFile(RECOVERED);
  await flush();
  const c = await stored(C);
  s = await state();
  report('recovered save opens in a new slot', C !== B && c?.name === 'recovered-day26' && (c?.lines ?? 0) > 0 && (await stored(E)).hash === eBefore.hash && (await stored(A)).hash === aBefore.hash,
    `(slot ${C} "${c?.name}", ${c?.lines} lines; ${s.slots.length} slots listed)`);

  // 6. deleting a slot deletes its backups
  await page.evaluate((key) => window.tw.client.post({ kind: 'saveSlot', action: 'delete', key }), A);
  await wait(1500);
  const left = (await allKeys()).filter((k) => String(k) === A || String(k).startsWith(`backup:${A}:`));
  report('delete takes the backups', left.length === 0, `(${left.length} keys left for ${A})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
