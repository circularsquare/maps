// T-1829: storage full or cleared while the game is open. (1) Storage full (slot writes in the clock
// worker refused with QuotaExceededError until a demand cache goes): a save drops the demand caches
// and goes through; full with no room to make, it says so, and saves once there is room. (2) The
// database cleared under a running game (site data cleared; deleted from another page): the next
// save writes the game into a new database under its name and account tie, and no notice is left.
// Also (T-1827 removed): deleting a slot, or a tombstone from the account, deletes it and its backups
// for good, and `deleted:` keys T-1827 left are swept when a page starts.
//   node trainworld/checks/t1829-storage.mjs [url]     (from the maps root; default localhost:6199)
// Prints PASS or FAIL lines and exits 1 on a failure. About three minutes.
import { launchChrome, collectErrors, acrossLine, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6199/';
const ORIGIN = new URL(BASE).origin;

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1829-', size: [1300, 900], args: ANGLE });
const errors = [];
try {
  const ready = (page) => page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const open = async () => {
    const page = await ctx.newPage();
    collectErrors(page, errors);
    await page.goto(`${BASE}?debug=1&paused=1&backupMs=50&v=${Date.now()}`);
    await ready(page);
    await page.waitForTimeout(1500);
    return page;
  };
  const state = (page) => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, writable: s.writable, reason: s.reason ?? '', error: s.error ?? '', slots: s.slots.map((x) => x.key) };
  });
  const sf = (page, f, arg) => page.evaluate(async ([src, arg]) => {
    const sf = await import('/src/workers/saveFile.ts');
    return (0, eval)(src)(sf, arg);
  }, [f.toString(), arg]);
  const keys = (page) => sf(page, async (sf) => {
    await sf.readStored('x'); // opens the connection
    const db = await new Promise((res, rej) => { const r = indexedDB.open('anitabuilder', 1); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
    const all = await new Promise((res) => { const r = db.transaction('saves').objectStore('saves').getAllKeys(); r.onsuccess = () => res(r.result); });
    db.close();
    return all;
  });
  /** the network bytes' hash and the summary of a stored record */
  const facts = (page, key) => sf(page, async (sf, key) => {
    const r = await sf.readStored(key);
    if (!r) return null;
    const { track } = await sf.decodeSave(r.bytes);
    const d = await crypto.subtle.digest('SHA-256', track);
    return { track: [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join(''), name: r.name, lines: r.lines, stations: r.stations, at: r.at };
  }, key);
  const post = (page, action, key, name) => page.evaluate(async ([a, k, n]) => (await import('/src/workers/clockClient.ts')).manageSaveSlot(a, k, n), [action, key, name]);
  const saveNow = async (page) => { await post(page, 'save'); await page.waitForTimeout(1500); };
  const fares = async (page, base) => {
    await page.evaluate((base) => window.tw.client.post({ kind: 'edit', seq: 990000 + Math.round(base * 100), op: { op: 'fares', base, perKm: 0.1 } }), base);
    await page.waitForTimeout(1500);
  };

  // ---- 0. deleting is for good (T-1827 removed)
  const a = await open();
  const line = await acrossLine(a, [-73.985, 40.758], 2);
  for (const end = Date.now() + 4000; Date.now() < end;) { await post(a, 'save'); await a.waitForTimeout(250); }
  await a.waitForTimeout(1000);
  const home = (await state(a)).active;
  const homeBackups = await sf(a, (sf, k) => sf.listBackups(k), home);
  await post(a, 'create', undefined, 'Other game');
  await a.waitForTimeout(3000);
  await post(a, 'delete', home);
  await a.waitForTimeout(2000);
  const left = (await keys(a)).filter((k) => k.includes(home) || k.startsWith('deleted:') || k.startsWith('backup:deleted:'));
  report('deleting a slot deletes it and its backups for good', line !== null && homeBackups.length > 0 && !(await state(a)).slots.includes(home) && left.length === 0,
    `(line ${line}; backups before ${homeBackups.length}; keys left ${JSON.stringify(left)})`);
  const tomb = await a.evaluate(async () => {
    const sf = await import('/src/workers/saveFile.ts'), cloud = await import('/src/workers/cloud.ts');
    const key = 'slot-t1829-tomb', at = 5000;
    const bytes = (await sf.readStored((await import('/src/workers/clockClient.ts')).saveSlotsState.value.active)).bytes;
    await sf.writeSlot(key, { bytes, at, day: 3, name: 'Tombstone check' });
    await sf.writeSync(key, { id: 'g-tomb', acct: 'acct-check', rev: 2, syncedAt: at });
    const c = { base: '', token: '', acct: 'acct-check', device: 'd', on: false };
    const wrote = await cloud.settleSlot(c, key, { games: [], tombstones: [{ id: 'g-tomb', at: 1 }], used: 0, limit: 0, maxGames: 0 });
    return { wrote, gone: !await sf.readStored(key), sync: await sf.readSync(key) };
  });
  const tombLeft = (await keys(a)).filter((k) => k.includes('t1829-tomb') || k.startsWith('deleted:'));
  report('a tombstone from the account deletes the slot for good', tomb.wrote && tomb.gone && !tomb.sync && tombLeft.length === 0, `(${JSON.stringify(tomb)}; keys left ${JSON.stringify(tombLeft)})`);

  // what T-1827 left in a browser is swept at start
  const stale = [`deleted:${Date.now() - 86400_000}-x`, `backup:deleted:${Date.now() - 86400_000}-x:t0`];
  await sf(a, async (sf, stale) => { for (const k of stale) await sf.writeStored(k, { bytes: new Uint8Array([1]), at: 1, day: 1, name: 'Old' }); }, stale);
  await a.close();
  const c = await open();
  const ks = await keys(c);
  report('T-1827 leftovers are swept at start', stale.every((k) => !ks.includes(k)), `(left ${JSON.stringify(stale.filter((k) => ks.includes(k)))})`);

  // ---- 2a. storage full: the demand caches go and the save goes through
  let demand = [];
  for (const end = Date.now() + 150000; Date.now() < end && !demand.length; await c.waitForTimeout(2000)) demand = (await keys(c)).filter((k) => k.startsWith('demand:'));
  // Chrome's CDP quota override does not take here (Storage.overrideQuotaForOrigin leaves writes
  // going through), so the clock worker's store is made full by hand: a slot write throws
  // QuotaExceededError until a demand cache is deleted
  const clock = c.workers().find((w) => /clock\.worker/.test(w.url()));
  await clock.evaluate(() => {
    self.__full = true; self.__refused = 0;
    const put = IDBObjectStore.prototype.put, del = IDBObjectStore.prototype.delete;
    IDBObjectStore.prototype.put = function (v, k) {
      if (self.__full && typeof k === 'string' && (k === 'autosave' || k.startsWith('slot-'))) { self.__refused++; throw new DOMException('The quota has been exceeded.', 'QuotaExceededError'); }
      return put.call(this, v, k);
    };
    IDBObjectStore.prototype.delete = function (k) { if (typeof k === 'string' && k.startsWith('demand:')) self.__full = false; return del.call(this, k); };
  });
  const atBefore = (await facts(c, (await state(c)).active)).at;
  await fares(c, 3.2);
  await saveNow(c);
  const st3 = await state(c), f3 = await facts(c, st3.active);
  const refused = await clock.evaluate(() => self.__refused);
  const demandAfter = (await keys(c)).filter((k) => k.startsWith('demand:'));
  report('storage full: demand caches dropped and the save goes through', demand.length > 0 && refused === 1 && f3.at > atBefore && st3.writable && !st3.error && demandAfter.length === 0,
    `(demand caches ${demand.length} -> ${demandAfter.length}; writes refused ${refused}; saved ${f3.at > atBefore}; ${JSON.stringify({ writable: st3.writable, error: st3.error })})`);
  // full with nothing to drop: the save says it failed, and the next one after room comes goes through
  await clock.evaluate(() => { self.__full = true; self.__refused = 0; self.__sticky = true; });
  await clock.evaluate(() => {
    const del = IDBObjectStore.prototype.delete;
    IDBObjectStore.prototype.delete = function (k) { if (self.__sticky) return del.call(this, '__none__'); return del.call(this, k); };
  });
  await fares(c, 3.6);
  await saveNow(c);
  const st4 = await state(c), refused4 = await clock.evaluate(() => self.__refused);
  await clock.evaluate(() => { self.__full = false; self.__sticky = false; });
  await saveNow(c);
  const st5 = await state(c);
  report('storage full for good: the save says so, then saves once there is room', refused4 === 2 && !!st4.error && !st5.error && st5.writable,
    `(writes refused ${refused4}; error "${st4.error}"; after room ${JSON.stringify({ writable: st5.writable, error: st5.error })})`);
  const cdp = await ctx.newCDPSession(c);

  // ---- 2b. the database cleared under the running game
  const active = (await state(c)).active;
  const rec = { id: 'g-clear', acct: 'acct-check', rev: 4, syncedAt: 1, local: true };
  await sf(c, (sf, [k, r]) => sf.writeSync(k, r), [active, rec]);
  await post(c, 'rename', active, 'Clear check');
  await c.waitForTimeout(1500);
  const cleared = async (how) => {
    if (how === 'site data') await cdp.send('Storage.clearDataForOrigin', { origin: ORIGIN, storageTypes: 'indexeddb' });
    else await c.evaluate(() => new Promise((res) => { const r = indexedDB.deleteDatabase('anitabuilder'); r.onsuccess = r.onerror = r.onblocked = () => res(); }));
    await c.waitForTimeout(1000);
    const gone = !(await facts(c, active));
    await fares(c, how === 'site data' ? 3.3 : 3.5);
    await saveNow(c);
    const st = await state(c), f = await facts(c, active), sync = await sf(c, (sf, k) => sf.readSync(k), active);
    const notice = await c.locator('.save-notice').count();
    report(`${how} cleared under the game: saved again under its name and tie, no notice`, gone && !!f && f.name === 'Clear check' && sync?.id === rec.id && sync?.local === true && st.writable && !st.error && !st.reason && notice === 0,
      `(gone first ${gone}; after: ${JSON.stringify({ name: f?.name, sync: sync?.id, writable: st.writable, error: st.error, reason: st.reason, notice })})`);
  };
  await cleared('site data');
  await cleared('deleteDatabase');

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
