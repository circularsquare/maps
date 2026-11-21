// T-1826: save safety. (1) A slot's backups reach back in four tiers (`planBackups` over two weeks of
// simulated play, then live with `?backupMs=50`, which scales 10 min to 50 ms), and a slot replaced
// from the account keeps its old version as a `pull` backup. (2) T-4014: a save from a newer build (the
// level one above this build's, an extra block after the known ones, an unknown header field) does not
// open: the import answers the newer line, the game in hand plays on in its slot and no slot is added;
// a save without a level opens as before. (3) Closing a normal tab does not ask.
//   node trainworld/checks/t1826-saves.mjs [url]     (from the maps root; default localhost:6189)
// Prints PASS or FAIL lines and exits 1 on a failure. About three minutes.
import { launchChrome, collectErrors, importSave, acrossLine, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6189/';
const NEWER = 'This game was saved on a newer game version. Reload the page to update.';

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const { ctx, close } = await launchChrome({ prefix: 'tw-t1826-', size: [1300, 900], args: ANGLE, acceptDownloads: true });
const errors = [];
try {
  const open = async (extra = '') => {
    const page = await ctx.newPage();
    collectErrors(page, errors);
    await page.goto(`${BASE}?debug=1&paused=1&backupMs=50${extra}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.waitForTimeout(1500);
    return page;
  };
  const state = (page) => page.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return { active: s.active, writable: s.writable, reason: s.reason ?? '', slots: s.slots.length };
  });
  const hashOf = (page, key) => page.evaluate(async (key) => {
    const r = await (await import('/src/workers/saveFile.ts')).readStored(key);
    if (!r) return null;
    const d = await crypto.subtle.digest('SHA-256', r.bytes);
    return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
  }, key);
  const backups = (page, key) => page.evaluate(async (key) => (await import('/src/workers/saveFile.ts')).listBackups(key), key);
  const asking = (page) => page.evaluate(async () => (await import('/src/game/saveSafety.ts')).askingBeforeLeaving.peek());
  const saveNow = (page) => page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
  const fares = async (page, base) => {
    await page.evaluate((base) => window.tw.client.post({ kind: 'edit', seq: 990000 + Math.round(base * 100), op: { op: 'fares', base, perKm: 0.1 } }), base);
    await page.waitForTimeout(1500);
  };
  /** the active slot's save rewritten: `block` adds an unknown block after the network, `level` the
   * level one above this build's, `field` a header field this build does not know, `old` drops the level
   * as saves before T-1826 had none */
  const craft = (page, key, kind) => page.evaluate(async ([key, kind]) => {
    const sf = await import('/src/workers/saveFile.ts');
    const { header, track } = await sf.decodeSave((await sf.readStored(key)).bytes);
    let t = track;
    if (kind === 'block') { t = new Uint8Array(track.length + 8); t.set(track); t.set([90, 90, 90, 49, 3, 1, 2, 3], track.length); }
    const h = { ...header };
    if (kind === 'level') h.level = sf.SAVE_LEVEL + 1;
    if (kind === 'field') h.futureField = 1;
    if (kind === 'old') delete h.level;
    return Array.from(await sf.encodeSave(h, t));
  }, [key, kind]);
  /** a file posted as Open a file does; the page's answer (the clock worker's `loaded` message) */
  const importInto = async (page, bytes) => {
    const was = (await state(page)).active;
    await page.evaluate(() => { window.__t1826 = window.tw.client.lastLoad; });
    await importSave(page, Uint8Array.from(bytes));
    await page.waitForFunction(async (was) => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active !== was || (window.tw.client.lastLoad !== window.__t1826 && window.tw.client.lastLoad?.from === 'file'), was, { timeout: 120000, polling: 300 });
    await page.waitForTimeout(1000);
    return { ...await state(page), said: await page.evaluate((t) => document.body.innerText.includes(t), NEWER) };
  };
  /** close the page as a person would; the beforeunload dialog's type if one came */
  const closeAsked = async (page) => {
    let type = null;
    page.on('dialog', async (d) => { type = d.type(); await d.accept(); });
    await page.click('#topbar', { position: { x: 4, y: 4 } }).catch(() => {}); // a user gesture, which Chrome needs to ask
    await page.close({ runBeforeUnload: true });
    for (let i = 0; i < 30 && !page.isClosed(); i++) await pause(200);
    return { type, closed: page.isClosed() };
  };

  // ---- 1. backups
  const a = await open();
  const sim = await a.evaluate(async () => {
    const { planBackups, BACKUP_STEPS } = await import('/src/workers/saveFile.ts');
    const M = 60_000;
    // two hours of play a day for 14 days, an autosave a minute, then the tiers' ages at the last save
    let tiers = [null, null, null, null], now = 0;
    for (let d = 0; d < 14; d++) for (let m = 0; m < 120; m++) {
      now = d * 1440 * M + 18 * 60 * M + m * M;
      tiers = planBackups(tiers, { at: now }, BACKUP_STEPS) ?? tiers;
    }
    return tiers.map((t) => t ? (now - t.at) / M : null);
  });
  const [s0, s1, s2, s3] = sim;
  report('tiers reach about 10 min, 1 h, 1 day and 1 week', s0 !== null && s0 <= 10 && s1 >= 10 && s1 <= 70 && s2 >= 60 && s2 <= 1440 + 120 && s3 >= 1440 && s3 <= 8 * 1440,
    `(ages in minutes after 14 days of 2 h a day: ${sim.map((x) => x?.toFixed(0)).join(', ')})`);

  const line = await acrossLine(a, [-73.985, 40.758], 2);
  await saveNow(a);
  await a.waitForTimeout(1500);
  const home = (await state(a)).active;
  for (const end = Date.now() + 25000; Date.now() < end;) { await saveNow(a); await a.waitForTimeout(250); }
  // the version list drops a backup with no player change since the slot's save (workers/versions.ts, T-2037: the clock
  // alone moving on is the same version), so a real edit goes in before the last save, after the backups exist
  // twice: a save also backs up the state it writes, so the second edit leaves every backup older than the slot
  for (const at of [[-73.95, 40.78], [-73.92, 40.8]]) { await acrossLine(a, at, 2); await saveNow(a); await a.waitForTimeout(1500); }
  const live = await backups(a, home);
  const tierKeys = live.filter((b) => /:t\d$/.test(b.key)).map((b) => b.key.slice(-2)).sort();
  // a tier saved a moment after the same player change holds the slot's own version, which the list leaves out
  // (workers/versions.ts, T-2037); so the rows are the tiers whose time and edit both differ from the slot's
  const slot = await a.evaluate(async (k) => { const x = (await (await import('/src/workers/saveFile.ts')).listSlots()).find((x) => x.key === k); return { at: x?.at, edit: x?.edit ?? x?.at }; }, home);
  const listed = live.filter((b) => /:t\d$/.test(b.key) && b.at !== slot.at && (b.edit ?? b.at) !== slot.edit).length;
  await a.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  // T-2037: listed under the card's "other versions" line
  await a.waitForSelector('.save-slots li.on .save-versions-line');
  await a.click('.save-slots li.on .save-versions-line');
  await a.waitForTimeout(800);
  const rows = await a.locator('.save-slots li.on .save-version-list li[data-kind="backup"] .save-version > span').allTextContents();
  report('live backups fill four tiers, listed with their age', line !== null && tierKeys.join() === 't0,t1,t2,t3' && rows.length === listed && listed >= 2 && rows.every((r) => /^Saved .* · Day \d+/.test(r)),
    `(line ${line}; tiers ${tierKeys.join()}; ages ${live.map((b) => ((Date.now() - b.at) / 1000).toFixed(1) + ' s').join(', ')}; rows ${JSON.stringify(rows)})`);
  await a.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });

  // a slot replaced from the account keeps what it held as its `pull` backup
  const pulled = await a.evaluate(async (home) => {
    const sf = await import('/src/workers/saveFile.ts'), cloud = await import('/src/workers/cloud.ts');
    const bytes = (await sf.readStored(home)).bytes, key = 'slot-t1826-pull';
    await sf.writeSlot(key, { bytes, at: 1000, day: 1, name: 'Pull check' });
    await cloud.writeFetched({ base: '', token: '', acct: 'check', device: 'd', on: false }, key, { stored: { bytes, at: 2000, day: 2, name: 'Pull check' }, id: 'g', rev: 1, savedAt: 2 });
    const list = await sf.listBackups(key), now = await sf.readStored(key);
    await sf.deleteSlot(key);
    return { list, at: now.at };
  }, home);
  // T-3395: a whole pull newer than the slot's good copy (none here) is its good copy too
  const pulls = pulled.list.filter((b) => !b.good);
  report('a pull from the account leaves a backup', pulled.at === 2000 && pulls.length === 1 && pulls[0].pull && pulls[0].at === 1000 && pulled.list.some((b) => b.good && b.at === 1000), `(${JSON.stringify(pulled)})`);

  // ---- 3a. a change autosave will write: no question on close
  await fares(a, 3.1);
  const askA = await asking(a);
  const c1 = await closeAsked(a);
  report('closing a normal tab does not ask', !askA && c1.type === null && c1.closed, `(asking ${askA}, dialog ${c1.type}, closed ${c1.closed})`);

  // ---- 2. newer saves (T-4014): refused, the game in hand kept in its slot, nothing written
  const b = await open();
  const old = await importInto(b, await craft(b, (await state(b)).active, 'old'));
  report('a save without a level opens as before', !old.said && old.writable, `(${JSON.stringify(old)})`);
  const base = old.active;
  // the game in hand still autosaves; what matters is that its slot holds this build's game, not the newer file
  const own = (key) => b.evaluate(async (key) => {
    const sf = await import('/src/workers/saveFile.ts');
    const { header } = await sf.decodeSave((await sf.readStored(key)).bytes);
    return header.level === sf.SAVE_LEVEL && !('futureField' in header);
  }, key);
  for (const kind of ['level', 'block', 'field']) {
    const file = await craft(b, base, kind);
    const r = await importInto(b, file);
    const ownGame = await own(base), notFile = await hashOf(b, base) !== await b.evaluate(async (bytes) => [...new Uint8Array(await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes)))].map((x) => x.toString(16).padStart(2, '0')).join(''), file);
    report(`a newer save (${kind}) does not open`, r.said && r.active === base && r.slots === old.slots && r.writable && ownGame && notFile,
      `(${JSON.stringify(r)}; slot still this build's ${ownGame}, not the file ${notFile})`);
  }
  // the save check calls such a save "other" (never damaged, never a good copy)
  const checks = await b.evaluate(async (bytes) => (await import('/src/workers/saveFile.ts')).checkSave(Uint8Array.from(bytes)), await craft(b, base, 'level'));
  report('the save check calls a newer level "other"', checks === 'other', `(${checks})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
