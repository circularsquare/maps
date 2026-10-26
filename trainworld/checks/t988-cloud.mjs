// T-988 stage 2: games on the account, against a local shonei-server (dev sign-in) with two browser
// profiles as two devices, A and B. A loads the real save signed out, signs in and adds it at the
// sign-in question; B signs in on the start menu and opens it from its account card, and A stops
// autosaving; a save by hand on A takes it back and B stops. A then plays with the server out of
// reach ("Not on your account yet") while B saves, and on its next save keeps its version as a fork
// ("<name> at <time>") and catches up. A deletes the old game and B's copy goes at its next start
// menu. A game made signed out and kept "here only" at the next sign-in shows the crossed cloud.
// Stage 3: that cloud's click adds the game; the playing game's account backup is listed and restored
// as a copy; the account panel counts games and room; signing out with Remove clears the browser.
// Screenshots of the sign-in question and the saves panel, light and dark, go to artifacts/trainworld/;
// stage 3's, light and dark at desktop and phone width, to artifacts/trainworld/t988s3/.
//   node trainworld/checks/t988-cloud.mjs [app url] [--server]     (from the maps root; default localhost:6169)
// Without --server it builds ../shonei-server and runs it on 127.0.0.1:8083 in a temp folder (the
// page's `?acct=local`); with --server one is already running there in dev mode (no SHONEI_SECRET).
// Prints PASS or FAIL lines and exits 1 on a failure. A few minutes.
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, REAL_SAVE, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const OWN_SERVER = !args.includes('--server');
const APP = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:6169/';
const SERVER = 'http://127.0.0.1:8083/';
const Q = 'debug=1&acct=local&demandWorkers=1';
const SUB = 't988-check-' + Date.now().toString(36);

let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 400) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

// ---- the server
let server = null, serverDir = '';
async function startServer() {
  server = spawn(join(serverDir, 'shonei.exe'), [], { cwd: serverDir, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: '' } });
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the local server did not start');
}
function stopServer() { server?.kill(); server = null; }
if (OWN_SERVER) {
  if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error('a server already answers on 127.0.0.1:8083: stop it, or pass --server to use it');
  serverDir = mkdtempSync(join(tmpdir(), 'tw-t988-srv-'));
  // shonei-server beside the maps repo (an agent worktree is two folders further in)
  let src = MAPS;
  for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_games.go')); up++) src = join(src, '..');
  src = join(src, '..', 'shonei-server');
  const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
  const b = spawnSync(go, ['build', '-o', join(serverDir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
  if (b.status !== 0) throw new Error(`go build in ${src} failed: ${b.stderr ?? b.error}`);
  for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(serverDir, f));
  await startServer();
}
const cfg = await (await fetch(SERVER + 'noritetsu/config')).json();
report('server keeps games', cfg.trainworldGames === true && cfg.dev === true, JSON.stringify({ games: cfg.trainworldGames, dev: cfg.dev }));
// the check's own session on the test account, to read what the account holds
const token = (await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub: SUB }) })).json()).token;
const account = async () => (await fetch(SERVER + 'trainworld/games', { headers: { Authorization: 'Bearer ' + token } })).json();

// ---- the devices
const errors = [];
const devices = [];
async function device(name) {
  const d = await launchChrome({ prefix: `tw-t988-${name}-`, size: [1300, 900], args: ANGLE });
  d.name = name;
  d.games = 0; // requests to /trainworld/games
  d.ctx.on('request', (r) => { if (r.url().startsWith(SERVER + 'trainworld/games')) d.games++; });
  devices.push(d);
  return d;
}
async function open(d, query = '') {
  d.page ??= await d.ctx.newPage();
  const p = d.page;
  if (!p.listening) {
    p.listening = true;
    p.on('pageerror', (e) => errors.push(`${d.name}: ${e.message}`));
    p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource/.test(m.text())) errors.push(`${d.name}: ${m.text()}`); });
  }
  await p.goto(`${APP}?${Q}${query}&v=${Date.now()}`);
  return p;
}
const inGame = (p) => p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
const state = (p) => p.evaluate(async () => {
  const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
  return { active: s.active, writable: s.writable, reason: s.reason ?? '', slots: s.slots.map((x) => ({ key: x.key, name: x.name, at: x.at, cloud: x.cloud ?? null })) };
});
const slotOf = async (p, key) => (await state(p)).slots.find((s) => s.key === key);
const deviceId = (p) => p.evaluate(() => JSON.parse(localStorage.getItem('trainworld.device.v1')));
const panel = async (p, on = true) => {
  await p.evaluate(async (on) => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = on; }, on);
  if (on) { await p.waitForSelector('.saves-panel .save-files'); await p.waitForTimeout(800); }
};
const accountPanel = (p, on = true) => p.evaluate(async (on) => { (await import('/src/game/account.ts')).accountOpen.value = on; }, on);
const theme = (p, shade) => p.evaluate(async (shade) => { (await import('/src/game/ui.ts')).shade.value = shade; }, shade);
// an edit that changes the game, then the page's hide: the autosave, and the account's upload
let fare = 1.5;
const change = async (p) => {
  fare += 0.25;
  await p.evaluate((base) => window.tw.client.post({ kind: 'edit', seq: 980000 + Math.round(base * 100), op: { op: 'fares', base, perKm: 0.1 } }), fare);
  await p.evaluate(() => window.tw.client.post({ kind: 'flush' }));
};
const saveByHand = (p) => p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
const signInDev = async (p) => {
  await p.waitForSelector('dialog.acct-panel[open] input[name="sub"]', { timeout: 30000 });
  await p.fill('dialog.acct-panel input[name="sub"]', SUB);
  await p.click('dialog.acct-panel .acct-row button[type="submit"]');
};
const shot = (p, name) => p.screenshot({ path: join(ARTIFACTS, name) });

const A = await device('a'), B = await device('b');
try {
  // 1. A, signed out: the real save into a slot of its own; nothing asks for the account's games
  let a = await open(A, '&new=1&paused=1');
  await inGame(a);
  await panel(a);
  const [fc] = await Promise.all([a.waitForEvent('filechooser'), a.click('.saves-panel .save-files button:has-text("Load a file")')]);
  await fc.setFiles(REAL_SAVE);
  const G = await until(async () => { const s = await state(a); return s.active.startsWith('slot-') ? s.active : null; }, 180000, 500);
  await inGame(a);
  await a.waitForTimeout(1500);
  report('signed out: no account games requests', !!G && A.games === 0, `(slot ${G}, ${A.games} requests)`);

  // 2. sign in on A: the question, "Don't add", "Are you sure", then add it
  await accountPanel(a);
  await signInDev(a);
  const q1 = await until(() => a.locator('.acct-merge p').textContent(), 20000);
  await a.waitForTimeout(400);
  await shot(a, 't988-question-light.png');
  await theme(a, 'dark');
  await a.waitForTimeout(400);
  await shot(a, 't988-question-dark.png');
  await theme(a, 'light');
  await a.click('.acct-merge button:has-text("Don\'t add")');
  const q2 = await a.locator('.acct-merge p').textContent();
  await a.click('.acct-merge button:has-text("Add it")');
  const up = await until(async () => { const l = await account(); return l.games.length === 1 ? l : null; }, 30000);
  const gid = up?.games[0]?.id;
  report('sign-in question adds the game', /^Add T-007-real-nyc to your account\?$/.test(q1 ?? '') && /^Are you sure\?/.test(q2 ?? '') && up?.games[0]?.name === 'T-007-real-nyc' && up?.games[0]?.pictureRev > 0,
    `("${q1}" / "${q2}"; account ${JSON.stringify(up?.games.map((g) => [g.name, g.rev, g.pictureRev]))})`);
  await accountPanel(a, false);
  await panel(a);
  const line = await until(() => a.locator('.save-slots li.on .save-account').textContent(), 10000);
  report('indicator on A', /^Saved to your account \d/.test(line ?? ''), `("${line}")`);
  await panel(a, false);

  // 3. B signs in on the start menu: the account's game is a card there; opening it claims it
  let b = await open(B);
  await b.waitForSelector('.start-menu .acct-start', { timeout: 60000 });
  await b.click('.start-menu .acct-start');
  await signInDev(b);
  await b.waitForTimeout(500);
  await accountPanel(b, false);
  const card = await until(() => b.locator(`.start-save[data-slot="account:${gid}"]`).count(), 20000);
  const cardText = card ? await b.locator(`.start-save[data-slot="account:${gid}"]`).textContent() : '';
  await b.click(`.start-save[data-slot="account:${gid}"]`);
  await inGame(b);
  await b.evaluate(() => window.tw.clock.setRunning(false));
  const devA = await deviceId(a), devB = await deviceId(b);
  const claimed = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return g?.writer === devB ? g : null; }, 20000);
  const sb = await state(b);
  report('saved on A, opened on B', !!card && /T-007-real-nyc/.test(cardText) && !!claimed && sb.slots.find((s) => s.key === sb.active)?.cloud?.id === gid,
    `(card ${card}, writer ${claimed ? 'B' : 'not B'}, B plays ${sb.slots.find((s) => s.key === sb.active)?.cloud?.id})`);

  // 4. A's next upload hears "elsewhere": autosave stops there
  await change(a);
  const gaveWay = await until(async () => { const s = await state(a); return !s.writable && /another device/.test(s.reason) ? s : null; }, 20000);
  const atA = (await slotOf(a, G)).at;
  await change(a);
  await a.waitForTimeout(2500);
  report('B opening it stops A', !!gaveWay && (await slotOf(a, G)).at === atA, `("${gaveWay?.reason ?? (await state(a)).reason}")`);

  // 5. a save by hand on A takes it back, and B stops
  await saveByHand(a);
  const back = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return (await state(a)).writable && g?.writer === devA ? g : null; }, 20000);
  await change(b);
  const bStops = await until(async () => { const s = await state(b); return !s.writable && /another device/.test(s.reason) ? s : null; }, 20000);
  report('save by hand on A takes it back, B stops', !!back && !!bStops, `(writer ${back ? 'A' : '?'}, rev ${back?.rev}; B "${bStops?.reason ?? ''}")`);

  // 6. A plays with the server out of reach: the game opened then cannot claim, its saves stay here
  if (OWN_SERVER) stopServer(); else await A.ctx.route(SERVER + '**', (r) => r.abort());
  a = await open(A, '&paused=1');
  await inGame(a);
  await a.waitForTimeout(1500);
  await change(a);
  const offline = await until(async () => { const s = await slotOf(a, G); return s?.cloud?.error ? s : null; }, 20000);
  await panel(a);
  const offLine = await a.locator('.save-slots li.on .save-account').textContent().catch(() => '');
  const offTip = await a.locator('.save-slots li.on .save-account').getAttribute('data-tip').catch(() => '');
  await panel(a, false);
  report('offline: Not on your account yet', !!offline && offLine === 'Not on your account yet' && /could not be reached|Offline/.test(offTip ?? ''), `("${offLine}", "${offTip}")`);

  // 7. back online, B saves first; A's next save keeps its version as a fork and catches up
  if (OWN_SERVER) await startServer(); else await A.ctx.unroute(SERVER + '**');
  await saveByHand(b);
  const revB = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return g?.writer === devB ? g.rev : null; }, 20000);
  await change(a);
  const forked = await until(async () => { const s = await state(a); return s.active !== G ? s : null; }, 30000);
  const fork = forked?.slots.find((s) => s.key === forked.active);
  const afterFork = await until(async () => { const l = await account(); return l.games.length === 2 && l.games.some((g) => g.id === fork?.cloud?.id) ? l : null; }, 30000);
  const old = await slotOf(a, G);
  report('fork kept as a new slot', !!fork && /^T-007-real-nyc at \d{1,2}:\d\d/.test(fork.name) && old?.cloud?.rev === revB && old?.cloud?.id === gid,
    `(playing "${fork?.name}", old slot rev ${old?.cloud?.rev} of B's ${revB})`);
  const caught = await until(async () => { const s = await slotOf(a, fork?.key); return s?.cloud?.okAt && !s.cloud.error ? s : null; }, 20000);
  report('offline, then catches up', !!afterFork && !!caught, `(account ${JSON.stringify(afterFork?.games.map((g) => [g.name, g.rev]))})`);

  // 8. B on its start menu; A deletes the old game; B's copy goes at its next start menu
  b = await open(B);
  await b.waitForSelector('.start-menu', { timeout: 60000 });
  await b.waitForTimeout(3000);
  await panel(a);
  await a.click(`.save-slots li[data-slot="${G}"] button:has-text("Delete")`);
  const ask = await a.locator(`.save-slots li[data-slot="${G}"] .callout p`).textContent();
  await a.click(`.save-slots li[data-slot="${G}"] .callout button:has-text("Delete saved game")`);
  const gone = await until(async () => { const l = await account(); return !l.games.some((g) => g.id === gid) && l.tombstones.some((t) => t.id === gid) ? l : null; }, 20000);
  await panel(a, false);
  b = await open(B);
  await b.waitForSelector('.start-menu', { timeout: 60000 });
  const bGone = await until(async () => {
    const slots = await b.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots()).map((s) => s.cloud?.id));
    return !slots.includes(gid) && (await b.locator(`.start-save[data-slot="account:${fork?.cloud?.id}"]`).count()) ? slots : null;
  }, 20000);
  report('delete crosses over', /from this browser and your account\?$/.test(ask ?? '') && !!gone && !!bGone, `("${ask}"; B's slots ${JSON.stringify(bGone)})`);

  // 9. A game made signed out and kept here only at the next sign-in: the crossed cloud (A's page
  // opened while the server was down has no sign-in form, so it is opened again first)
  a = await open(A, '&paused=1');
  await inGame(a);
  await a.waitForTimeout(1500);
  await accountPanel(a);
  await a.click('dialog.acct-panel button:has-text("Sign out")');
  // stage 3: the account's games are in this browser, so whether they stay is asked first
  await a.click('dialog.acct-panel .acct-signout button:has-text("Keep games")');
  await panel(a);
  // the copy's name is asked only once Save a copy is clicked (T-1295)
  await a.click('button:text-is("Save a copy")');
  await a.fill('.save-slot-new input', 'Kept here');
  await a.click('.save-slot-new button[type="submit"]');
  await until(async () => (await state(a)).slots.some((s) => s.name === 'Kept here'), 20000);
  await panel(a, false);
  await accountPanel(a);
  await signInDev(a);
  const q3 = await until(() => a.locator('.acct-merge p').textContent(), 20000);
  await a.click('.acct-merge button:has-text("Don\'t add")');
  await a.click('.acct-merge button:has-text("Keep it here only")');
  await a.waitForTimeout(1000);
  await accountPanel(a, false);
  // the fork is the game shown playing in the screenshots: the page reloads into it
  const kept = (await state(a)).slots.find((s) => s.name === 'Kept here');
  await a.evaluate(async (key) => (await import('/src/workers/clockClient.ts')).manageSaveSlot('open', key), fork.key);
  await until(async () => (await state(a)).active === fork.key, 120000, 600);
  await inGame(a);
  await a.waitForTimeout(2000);
  await panel(a);
  const mark = await a.locator(`.save-slots li[data-slot="${kept?.key}"] .save-cloud`).getAttribute('data-tip').catch(() => null);
  const playingLine = await a.locator('.save-slots li.on .save-account').textContent().catch(() => '');
  await shot(a, 't988-saves-light.png');
  await theme(a, 'dark');
  await a.waitForTimeout(500);
  await shot(a, 't988-saves-dark.png');
  await theme(a, 'light');
  const listed = (await account()).games.map((g) => g.name);
  report('kept here only: the crossed cloud', q3 === 'Add Kept here to your account?' && /^Only in this browser/.test(mark ?? '') && !listed.includes('Kept here') && /^Saved to your account/.test(playingLine ?? ''),
    `("${q3}", mark "${mark}", playing "${playingLine}", account ${JSON.stringify(listed)})`);

  // ---- stage 3
  const S3 = join(ARTIFACTS, 't988s3');
  mkdirSync(S3, { recursive: true });
  const shots = async (name, before = async () => {}) => {
    for (const size of [[1300, 900, 'desktop'], [390, 844, 'phone']]) {
      await a.setViewportSize({ width: size[0], height: size[1] });
      for (const sh of ['light', 'dark']) {
        await theme(a, sh);
        await a.waitForTimeout(500);
        await before();
        await a.screenshot({ path: join(S3, `${name}-${size[2]}-${sh}.png`) });
      }
    }
    await theme(a, 'light');
    await a.setViewportSize({ width: 1300, height: 900 });
  };
  // 10. the crossed cloud on a game kept here only is a button: a click adds it to the account
  await shots('add', () => a.locator(`.save-slots li[data-slot="${kept?.key}"]`).scrollIntoViewIfNeeded());
  await a.click(`.save-slots li[data-slot="${kept?.key}"] button.save-add`);
  const added = await until(async () => (await account()).games.find((g) => g.name === 'Kept here'), 30000);
  const markGone = await until(async () => (await a.locator(`.save-slots li[data-slot="${kept?.key}"] .save-cloud`).count()) === 0, 15000);
  report('Add on a kept-here card', !!added && !!markGone, `(account ${JSON.stringify((await account()).games.map((g) => g.name))})`);
  await panel(a, false);

  // 11. the playing game's account backups listed under it beside this browser's, one restored as a copy
  const fid = fork?.cloud?.id;
  await change(a);
  await saveByHand(a);
  const withBackup = await until(async () => { const g = (await account()).games.find((x) => x.id === fid); return g?.backups?.length ? g : null; }, 30000);
  await panel(a);
  const row = await until(() => a.locator('.save-slots li.on .save-backup[data-account-backup]').count(), 10000);
  const rowTip = row ? await a.locator('.save-slots li.on .save-backup[data-account-backup] > span').first().getAttribute('data-tip') : '';
  await shots('saves', () => a.locator('.save-slots li.on').scrollIntoViewIfNeeded());
  report('account backups listed', !!withBackup && !!row && /on your account from/.test(rowTip ?? ''), `(backups ${JSON.stringify(withBackup?.backups?.map((b) => b.rev))}, rows ${row}, "${rowTip}")`);
  const before3 = (await account()).games.length;
  await a.locator('.save-slots li.on .save-backup[data-account-backup] button').first().click();
  const restored = await until(async () => { const s = await state(a); const x = s.slots.find((y) => y.key === s.active); return x && x.key !== fork?.key && x.name.startsWith(`${fork?.name} at `) ? x : null; }, 120000, 600);
  await inGame(a);
  const restoredUp = await until(async () => { const l = await account(); return l.games.length === before3 + 1 && l.games.some((g) => g.name === restored?.name) ? l : null; }, 30000);
  report('account backup restored as a copy', !!restored && !!restoredUp, `(playing "${restored?.name}", account ${JSON.stringify(restoredUp?.games.map((g) => g.name))})`);

  // 12. the account panel counts the account's games and the room they take; Sign out asks
  await a.waitForTimeout(1500);
  await accountPanel(a);
  const usage = await until(() => a.locator('dialog.acct-panel .acct-games').textContent(), 15000);
  await a.click('dialog.acct-panel button:has-text("Sign out")');
  const outQ = await until(() => a.locator('dialog.acct-panel .acct-signout p').textContent(), 10000);
  await shots('signout');
  report('account panel: games and room; the sign-out question', /^\d+ games on your account · [\d.]+ MB of 64 MB$/.test(usage ?? '') && outQ === 'Remove your games from this browser?', `("${usage}", "${outQ}")`);

  // 13. Remove: the page goes to the start menu, the account's games leave this browser (none is
  // deleted from the account) and the page is signed out
  const onAccount = (await account()).games.length;
  await a.click('dialog.acct-panel .acct-signout button:has-text("Remove games")');
  await a.waitForURL((u) => !/paused=1/.test(u.search) && !/slot=|load=/.test(u.search), { timeout: 60000 }).catch(() => {});
  await a.waitForSelector('.start-menu', { timeout: 60000 });
  const removed = await until(async () => {
    const r = await a.evaluate(async () => ({
      slots: (await (await import('/src/workers/saveFile.ts')).listSlots()).map((s) => ({ name: s.name, acct: s.cloud?.acct ?? null })),
      signedIn: !!(await import('/src/game/account.ts')).session.value,
    }));
    return !r.signedIn && !r.slots.some((s) => s.acct) ? r : null;
  }, 60000, 600);
  const kept3 = (await account()).games.length;
  report('Remove at sign-out', !!removed && kept3 === onAccount, `(slots left ${JSON.stringify(removed?.slots)}, account ${kept3} of ${onAccount})`);

  report('no page errors', errors.length === 0, errors.slice(0, 4).join(' | '));
} catch (err) {
  report('ran to the end', false, String(err?.stack ?? err));
} finally {
  for (const d of devices) await d.close().catch(() => {});
  if (OWN_SERVER) { stopServer(); await pause(500); rmSync(serverDir, { recursive: true, force: true }); }
}
process.exit(failed ? 1 : 0);
