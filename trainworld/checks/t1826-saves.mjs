// T-1826: save safety. (1) A slot's backups reach back in four tiers (`planBackups` over two weeks of
// simulated play, then live with `?backupMs=50`, which scales 10 min to 50 ms), and a slot replaced
// from the account keeps its old version as a `pull` backup. (2) A save from a newer build (an extra
// block after the known ones, or a higher `level`) opens with autosave off and the notice, its slot
// left byte for byte through an edit and a minute and more of waiting, and Download gives the file as
// loaded; a save without a level opens as before; Make a copy plays on in a slot of this build's.
// (3) Closing the page asks first only while a change cannot be saved.
//   node trainworld/checks/t1826-saves.mjs [url]     (from the maps root; default localhost:6189)
// Prints PASS or FAIL lines and exits 1 on a failure. About three minutes.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, acrossLine, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6189/';
const NEWER = 'This game was saved by a newer version. Reload the page to update.';

let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const sha = (bytes) => createHash('sha256').update(Buffer.from(bytes)).digest('hex');
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
    return { active: s.active, writable: s.writable, newer: !!s.newer, reason: s.reason ?? '' };
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
  /** the active slot's save rewritten: `block` adds an unknown block after the network, `level` a
   * higher level, `old` drops the level as saves before T-1826 had none */
  const craft = (page, key, kind) => page.evaluate(async ([key, kind]) => {
    const sf = await import('/src/workers/saveFile.ts');
    const { header, track } = await sf.decodeSave((await sf.readStored(key)).bytes);
    let t = track;
    if (kind === 'block') { t = new Uint8Array(track.length + 8); t.set(track); t.set([90, 90, 90, 49, 3, 1, 2, 3], track.length); }
    const h = { ...header };
    if (kind === 'level') h.level = 99;
    if (kind === 'old') delete h.level;
    return Array.from(await sf.encodeSave(h, t));
  }, [key, kind]);
  const importInto = async (page, bytes) => {
    const was = (await state(page)).active;
    await importSave(page, Uint8Array.from(bytes));
    await page.waitForFunction(async (was) => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active !== was, was, { timeout: 120000, polling: 300 });
    await page.waitForTimeout(2500);
    return state(page);
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
  const live = await backups(a, home);
  const tierKeys = live.filter((b) => /:t\d$/.test(b.key)).map((b) => b.key.slice(-2)).sort();
  await a.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await a.waitForSelector('.save-slots li.on .save-versions');
  await a.click('.save-slots li.on .save-versions');
  await a.waitForTimeout(800);
  const rows = await a.locator('.save-backups .save-backup span.muted').allTextContents();
  report('live backups fill four tiers, listed with their age', line !== null && tierKeys.join() === 't0,t1,t2,t3' && rows.length === 4 && rows.every((r) => /(ago|Just now) · Day \d+/.test(r)),
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
  report('a pull from the account leaves a backup', pulled.at === 2000 && pulled.list.length === 1 && pulled.list[0].pull && pulled.list[0].at === 1000, `(${JSON.stringify(pulled)})`);

  // ---- 3a. a change autosave will write: no question on close
  await fares(a, 3.1);
  const askA = await asking(a);
  const c1 = await closeAsked(a);
  report('closing a normal tab does not ask', !askA && c1.type === null && c1.closed, `(asking ${askA}, dialog ${c1.type}, closed ${c1.closed})`);

  // ---- 2. newer saves
  const b = await open();
  const base = (await state(b)).active;
  const old = await importInto(b, await craft(b, base, 'old'));
  report('a save without a level opens as before', !old.newer && old.writable, `(${JSON.stringify(old)})`);
  const lev = await importInto(b, await craft(b, base, 'level'));
  report('a higher level opens with autosave off', lev.newer && !lev.writable && lev.reason === NEWER, `(${JSON.stringify(lev)})`);
  const file = await craft(b, base, 'block');
  const nb = await importInto(b, file);
  const kept = await hashOf(b, nb.active);
  const notice = await b.locator('.save-notice span').first().textContent().catch(() => '');
  report('an extra block opens with autosave off and the notice', nb.newer && !nb.writable && notice === NEWER && kept === sha(file),
    `(${JSON.stringify(nb)}; top bar "${notice}"; slot as the file ${kept === sha(file)})`);
  await fares(b, 3.4);
  const askB = await asking(b);
  await b.waitForTimeout(65000);
  const after = await hashOf(b, nb.active), nbk = await backups(b, nb.active);
  report('a minute after an edit the newer slot is as it was', after === sha(file) && nbk.length === 0 && askB, `(same bytes ${after === sha(file)}, backups ${nbk.length}, asking ${askB})`);
  const [dl] = await Promise.all([b.waitForEvent('download'), b.evaluate(async () => (await import('/src/workers/clockClient.ts')).exportGame())]);
  const got = await readFile(await dl.path());
  report('Download gives the newer file as loaded', sha(got) === sha(file), `(${got.length} bytes, file ${file.length})`);
  // ---- 3b. a change nothing can write: the page asks
  const c2 = await closeAsked(b);
  report('closing with an unsaved change asks', c2.type === 'beforeunload', `(dialog ${c2.type}, closed ${c2.closed})`);

  // ---- 2b. Make a copy plays on in this build's slot; the newer slot is still as it was
  const c = await open();
  const reopened = await state(c);
  await c.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Copy check'));
  await c.waitForTimeout(3000);
  const copy = await state(c);
  report('Make a copy autosaves in a slot of its own', reopened.newer && reopened.active === nb.active && !copy.newer && copy.writable && copy.active !== nb.active && await hashOf(c, nb.active) === sha(file),
    `(reopened ${JSON.stringify(reopened)}; copy ${JSON.stringify(copy)})`);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
