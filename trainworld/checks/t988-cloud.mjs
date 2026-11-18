// T-988 stage 2 (sync rules of T-2037): games on the account, against a local shonei-server (dev sign-in)
// with two browser profiles as two devices, A and B. A loads the real save signed out, signs in and adds it
// at the sign-in question; B signs in on the start menu and opens it from its account card; B only ran its
// clock, so A's next player change saves over B's claim and A carries on; a save by hand on A and a player
// change on B stop B. A then plays with the server out of reach ("Not on your account yet") while a third
// device C joins and uploads a player change to the game; A's next change splits its game off in place (same slot, a new
// account game of the same origin), one card with the account's copy as another version. A deletes that
// version and C's slot of it goes at its next start menu. A game made signed out and kept "here only" at
// the next sign-in shows the crossed cloud.
// Stage 3: the playing game's account backup is a row of its card and is
// opened as a new version; the account panel counts games and room; signing out with Remove clears the browser.
// T-1395's panel: the head's account icon, the playing card's icons, Make a copy naming the copy in place.
// Screenshots of the sign-in question and the saves panel, light and dark, go to artifacts/trainworld/;
// stage 3's, light and dark at desktop and phone width, to artifacts/trainworld/t988s3/; the saves
// panel signed out (built-a) and signed in with earlier versions open (built-b) to t1395/.
//   node trainworld/checks/t988-cloud.mjs [app url] [--server]     (from the maps root; default localhost:6169)
// Without --server it builds ../shonei-server and runs it on 127.0.0.1:8083 in a temp folder (the
// page's `?acct=local`); with --server one is already running there in dev mode (SHONEI_SECRET set, SHONEI_DEV=1).
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
  server = spawn(join(serverDir, 'shonei.exe'), [], { cwd: serverDir, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1' } });
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
  await p.goto(`${APP}?${Q}${query}&v=${Date.now()}`, { waitUntil: 'domcontentloaded', timeout: 120000 });
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
// light and dark at desktop and phone width: `${dir}/${name}-{desktop,phone}-{light,dark}.png`
const shotsOf = async (p, dir, name, before = async () => {}) => {
  mkdirSync(dir, { recursive: true });
  for (const size of [[1300, 900, 'desktop'], [390, 844, 'phone']]) {
    await p.setViewportSize({ width: size[0], height: size[1] });
    for (const sh of ['light', 'dark']) {
      await theme(p, sh);
      await p.waitForTimeout(500);
      await before();
      await p.screenshot({ path: join(dir, `${name}-${size[2]}-${sh}.png`) });
    }
  }
  await theme(p, 'light');
  await p.setViewportSize({ width: 1300, height: 900 });
};
const T1395 = join(ARTIFACTS, 't1395');
const tipOf = (p, sel) => p.locator(sel).first().getAttribute('data-tip').catch(() => null);

const A = await device('a'), B = await device('b');
try {
  // 1. A, signed out: the real save into a slot of its own; nothing asks for the account's games
  let a = await open(A, '&new=1&paused=1');
  await inGame(a);
  await panel(a);
  const [fc] = await Promise.all([a.waitForEvent('filechooser'), a.click('.saves-panel .save-files button:has-text("Open a file")')]);
  await fc.setFiles(REAL_SAVE);
  const G = await until(async () => { const s = await state(a); return s.active.startsWith('slot-') ? s.active : null; }, 180000, 500);
  await inGame(a);
  await a.waitForTimeout(1500);
  report('signed out: no account games requests', !!G && A.games === 0, `(slot ${G}, ${A.games} requests)`);
  // T-1395: signed out, the head's crossed cloud asks to sign in; the playing card's icons
  await panel(a);
  const headOut = await tipOf(a, '.saves-head .save-head-mark');
  const icons = await a.locator('.save-slots li.on .save-actions button').evaluateAll((bs) => bs.map((b) => b.dataset.tip?.replace(/ \(Ctrl\+S\).*/, '') ?? b.textContent));
  await shotsOf(a, T1395, 'built-a');
  report('signed out: head and card icons', headOut === 'Kept in this browser only. Sign in to keep your games on your account.' && icons.join('|') === 'Make a copy of this save|Download as a file|Delete|Save this game now',
    `("${headOut}"; ${icons.join(', ')})`);
  await panel(a, false);

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
  const line = await until(async () => { const t = await tipOf(a, '.save-slots li.on .save-when .save-account'); return /^On your account/.test(t ?? '') ? t : null; }, 10000);
  const headIn = await tipOf(a, '.saves-head .save-head-mark');
  report('indicator on A', /^On your account, saved \d/.test(line ?? '') && /^Signed in/.test(headIn ?? ''), `("${line}", head "${headIn}")`);
  await panel(a, false);

  // 3. B signs in on the start menu: the account's game is a card there; opening it claims it
  let b = await open(B);
  // a device with no games opens on the new-game screen; its Back leads to the menu with the account button
  await b.waitForSelector('.start-menu', { timeout: 60000 });
  await b.waitForSelector('.start-menu .acct-start', { timeout: 60000 });
  await b.click('.start-menu .acct-start');
  await signInDev(b);
  await b.waitForTimeout(500);
  await accountPanel(b, false);
  // the new-game sheet (all a device with no games shows) has no cards: Back leads to the menu that lists them
  await until(() => b.locator('.start-menu button:text-is("Back")').click({ timeout: 2000 }).then(() => true), 8000);
  const card = await until(() => b.locator(`.start-menu li[data-origin] .start-save[data-slot="account:${gid}"]`).count(), 20000);
  const cardText = card ? await b.locator(`.start-save[data-slot="account:${gid}"]`).textContent() : '';
  await b.click(`.start-save[data-slot="account:${gid}"]`);
  await inGame(b);
  await b.evaluate(() => window.tw.clock.setRunning(false));
  const devA = await deviceId(a), devB = await deviceId(b);
  const claimed = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return g?.writer === devB ? g : null; }, 20000);
  const sb = await state(b);
  report('saved on A, opened on B', !!card && /T-007-real-nyc/.test(cardText) && !!claimed && sb.slots.find((s) => s.key === sb.active)?.cloud?.id === gid,
    `(card ${card}, writer ${claimed ? 'B' : 'not B'}, B plays ${sb.slots.find((s) => s.key === sb.active)?.cloud?.id})`);

  // 4. B only opened the game and ran its clock: A's next player change saves over B's claim and A carries on
  const revClaimed = claimed?.rev ?? 0;
  await change(a);
  const overB = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return g?.writer === devA && g.rev > revClaimed ? g : null; }, 150000, 1000);
  const sa = await state(a);
  report('B opened it idle: A saves over it and carries on', !!overB && sa.writable, `(writer ${overB ? 'A' : '?'}, rev ${overB?.rev} after ${revClaimed}; A writable ${sa.writable} "${sa.reason}")`);

  // 5. a save by hand on A takes it back, and B stops
  await saveByHand(a);
  const back = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return (await state(a)).writable && g?.writer === devA ? g : null; }, 20000);
  await change(b);
  const bStops = await until(async () => { const s = await state(b); return !s.writable && /another device/.test(s.reason) ? s : null; }, 20000);
  report('save by hand on A takes it back, B stops', !!back && !!bStops, `(writer ${back ? 'A' : '?'}, rev ${back?.rev}; B "${bStops?.reason ?? ''}")`);

  // 6. A plays with the server out of reach: the game opened then cannot claim, its saves stay here
  const revA = (await account()).games.find((g) => g.id === gid)?.rev;
  if (OWN_SERVER) stopServer(); else await A.ctx.route(SERVER + '**', (r) => r.abort());
  a = await open(A, '&paused=1');
  await inGame(a);
  await a.waitForTimeout(1500);
  await change(a);
  const offline = await until(async () => { const s = await slotOf(a, G); return s?.cloud?.error ? s : null; }, 20000);
  await panel(a);
  const offWarn = await a.locator('.save-slots li.on .save-account.warn').count();
  const offTip = await tipOf(a, '.save-slots li.on .save-account');
  await panel(a, false);
  report('offline: Not on your account yet', !!offline && offWarn === 1 && /^Not on your account yet\. .*(could not be reached|Offline)/.test(offTip ?? ''), `(warn mark ${offWarn}, "${offTip}")`);

  // 7. back online. B, stopped, keeps a change of its own, so a third device C joins, opens the game (writer, idle)
  // and makes a player change that goes up; A, whose change never went up, changes again: its game is split off in
  // place, the account's copy kept as another version
  await B.close();
  if (OWN_SERVER) await startServer(); else await A.ctx.unroute(SERVER + '**');
  const C = await device('c');
  let c = await open(C);
  await c.waitForSelector('.start-menu .acct-start', { timeout: 60000 });
  await c.click('.start-menu .acct-start');
  await signInDev(c);
  await c.waitForTimeout(500);
  await accountPanel(c, false);
  await until(() => c.locator('.start-menu button:text-is("Back")').click({ timeout: 2000 }).then(() => true), 8000);
  await c.waitForSelector(`.start-menu li[data-origin] .start-save[data-slot="account:${gid}"]`, { timeout: 30000 });
  await c.click(`.start-save[data-slot="account:${gid}"]`);
  await inGame(c);
  await c.evaluate(() => window.tw.clock.setRunning(false));
  const devC = await deviceId(c);
  await until(async () => (await account()).games.find((x) => x.id === gid)?.writer === devC, 20000);
  await change(c);
  const revC = await until(async () => { const g = (await account()).games.find((x) => x.id === gid); return g?.writer === devC && g.rev > revA ? g.rev : null; }, 150000, 1000);
  await change(a);
  const split = await until(async () => { const s = await slotOf(a, G); const l = await account(); return s?.cloud?.id && s.cloud.id !== gid && l.games.length === 2 && l.games.some((g) => g.id === s.cloud.id) ? { s, l } : null; }, 150000, 1000);
  const fork = split ? { key: G, name: split.s.name, cloud: split.s.cloud } : null;
  const oldGame = split?.l.games.find((g) => g.id === gid);
  report('conflict split off in place', !!split && (await state(a)).active === G && split.s.name === 'T-007-real-nyc' && oldGame?.rev === revC && oldGame?.name === 'T-007-real-nyc',
    `(slot ${G} now on ${split?.s.cloud.id}, account ${JSON.stringify(split?.l.games.map((g) => [g.name, g.id, g.rev]))}, C's rev ${revC})`);
  const caught = await until(async () => { const s = await slotOf(a, G); return s?.cloud?.okAt && !s.cloud.error ? s : null; }, 20000);
  // one card with the account's copy as its other version (grouped by origin)
  await panel(a);
  await a.waitForSelector('.save-slots li.on .save-versions-line', { timeout: 10000 });
  if (await a.locator('.save-slots li.on .save-versions-line[aria-expanded="false"]').count()) await a.click('.save-slots li.on .save-versions-line');
  const cards = await a.locator('.save-slots > li[data-origin]').count();
  const versionRows = await until(() => a.locator(`.save-slots li.on .save-version-list li[data-kind=account][data-version="account:${gid}"]`).count(), 10000);
  report('offline, then catches up as one card', !!caught && cards === 1 && versionRows === 1, `(cards ${cards}, account-version rows ${versionRows})`);
  await panel(a, false);

  // 8. C on its start menu; A deletes the old version (the account's copy under the old id); C's slot of it goes
  // at its next start menu, and the game kept on the account is a card there
  c = await open(C);
  await c.waitForSelector('.start-menu', { timeout: 60000 });
  await c.waitForTimeout(3000);
  await panel(a);
  const oldRow = `.save-slots li.on .save-version-list li[data-version="account:${gid}"]`;
  if (await a.locator('.save-slots li.on .save-versions-line[aria-expanded="false"]').count()) await a.click('.save-slots li.on .save-versions-line');
  await a.click(`${oldRow} .save-delete`);
  const ask = await a.locator(`${oldRow} .callout p`).textContent();
  await a.click(`${oldRow} .callout button.danger`);
  const gone = await until(async () => { const l = await account(); return !l.games.some((g) => g.id === gid) && l.tombstones.some((t) => t.id === gid) ? l : null; }, 20000);
  await panel(a, false);
  c = await open(C);
  await c.waitForSelector('.start-menu', { timeout: 60000 });
  const cGone = await until(async () => {
    const slots = await c.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots()).map((s) => s.cloud?.id));
    return !slots.includes(gid) && (await c.locator(`.start-save[data-slot="account:${fork?.cloud?.id}"]`).count()) ? slots : null;
  }, 20000);
  report('delete crosses over', /^Delete this version from your account\?$/.test(ask ?? '') && !!gone && !!cGone, `("${ask}"; C's slots ${JSON.stringify(cGone)})`);

  // 9. A game made signed out and kept here only at the next sign-in: the crossed cloud (A's page
  // opened while the server was down has no sign-in form, so it is opened again first)
  a = await open(A, '&paused=1');
  await inGame(a);
  await a.waitForTimeout(1500);
  await accountPanel(a);
  await a.click('dialog.acct-panel button:has-text("Sign out")');
  // T-1851: signing out is direct, the games stay in this browser
  await panel(a);
  // T-1395: Make a copy makes "<game> copy" at once, the slot played, with its name open for typing
  const preCopy = await state(a);
  const copiedFrom = preCopy.slots.find((s) => s.key === preCopy.active)?.name;
  await a.click('.save-slots li.on .save-copy');
  await a.waitForSelector('.save-slots li.on .save-name-field', { timeout: 20000 });
  const copyNamed = await a.evaluate(() => { const f = document.querySelector('.save-slots li.on .save-name-field'); return { value: f.value, selected: f.selectionEnd - f.selectionStart === f.value.length, focused: document.activeElement === f }; });
  const copyState = await state(a);
  await a.fill('.save-slots li.on .save-name-field', 'Kept here');
  await a.press('.save-slots li.on .save-name-field', 'Enter');
  const copyKept = await until(async () => (await state(a)).slots.some((s) => s.name === 'Kept here'), 20000);
  report('Make a copy names it in place', copyNamed.value === `${copiedFrom} copy` && copyNamed.selected && copyNamed.focused && copyState.slots.find((s) => s.key === copyState.active)?.name === `${copiedFrom} copy` && !!copyKept,
    `(${JSON.stringify(copyNamed)}, playing "${copyState.slots.find((s) => s.key === copyState.active)?.name}")`);
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
  // a copy shares its game's origin: it is a version of the card (a row of its list), with no account mark of its own
  if (await a.locator('.save-slots li.on .save-versions-line[aria-expanded="false"]').count()) await a.click('.save-slots li.on .save-versions-line');
  const keptRow = await until(() => a.locator(`.save-slots li.on .save-version-list li[data-version="${kept?.key}"][data-kind=slot]`).count(), 10000);
  const playingLine = await tipOf(a, '.save-slots li.on .save-account');
  await shot(a, 't988-saves-light.png');
  await theme(a, 'dark');
  await a.waitForTimeout(500);
  await shot(a, 't988-saves-dark.png');
  await theme(a, 'light');
  const listed = (await account()).games.map((g) => g.name);
  report('kept here only: a version of the card', q3 === 'Add Kept here to your account?' && keptRow === 1 && !listed.includes('Kept here') && /^On your account/.test(playingLine ?? ''),
    `("${q3}", version rows of it ${keptRow}, playing "${playingLine}", account ${JSON.stringify(listed)})`);

  // ---- stage 3
  const shots = (name, before) => shotsOf(a, join(ARTIFACTS, 't988s3'), name, before);
  // 10. (the crossed cloud's Add button is on a card of its own origin; a copy is a version of its game's card, so
  // it has none and there is nothing to click) the saves panel with the version row open, light and dark
  await shots('kept-version', () => a.locator('.save-slots li.on').scrollIntoViewIfNeeded());
  await panel(a, false);

  // 11. the playing game's account backups are rows among the card's other versions (data-kind=accountBackup);
  // one opened as a new version under the game's name
  const fid = fork?.cloud?.id;
  await change(a);
  await saveByHand(a);
  const withBackup = await until(async () => { const g = (await account()).games.find((x) => x.id === fid); return g?.backups?.length ? g : null; }, 30000);
  await panel(a);
  await a.waitForSelector('.save-slots li.on .save-versions-line', { timeout: 10000 });
  if (await a.locator('.save-slots li.on .save-versions-line[aria-expanded="false"]').count()) await a.click('.save-slots li.on .save-versions-line');
  const BK = '.save-slots li.on .save-version-list li[data-kind=accountBackup]';
  const row = await until(() => a.locator(BK).count(), 10000);
  const rowTip = row ? await tipOf(a, `${BK} .num`) : '';
  const rowText = row ? await a.locator(BK).first().textContent() : '';
  await shotsOf(a, T1395, 'built-b', () => a.locator('.save-slots li.on').scrollIntoViewIfNeeded());
  report('account backups: rows of the card', !!withBackup && !!row && /on your account/i.test(rowTip ?? '') && /^Saved (just now|\d+ \w+ ago|(Today|\w{3} \d+) at \d{1,2}:\d\d).*Open$/.test(rowText ?? ''),
    `(backups ${JSON.stringify(withBackup?.backups?.map((b) => b.rev))}, rows ${row}, "${rowText}", "${rowTip}")`);
  const before3 = (await account()).games.length;
  await a.locator(`${BK} button.save-version-open`).first().click();
  const restored = await until(async () => { const s = await state(a); const x = s.slots.find((y) => y.key === s.active); return x && x.key !== fork?.key && x.name === 'T-007-real-nyc' ? x : null; }, 120000, 600); // opened under the game's name, a new version of its card
  await inGame(a);
  const restoredUp = await until(async () => { const l = await account(); return l.games.length === before3 + 1 && l.games.some((g) => g.name === restored?.name) ? l : null; }, 30000);
  report('account backup restored as a copy', !!restored && !!restoredUp, `(playing "${restored?.name}", account ${JSON.stringify(restoredUp?.games.map((g) => g.name))})`);

  // 12. the account panel counts the account's games and the room they take (T-1851: the sign-out
  // question and "Remove games" are gone)
  await a.waitForTimeout(1500);
  await accountPanel(a);
  const usage = await until(() => a.locator('dialog.acct-panel .acct-games').first().textContent(), 15000);
  report('account panel: games and room', /^\d+ games? on your account · [\d.]+ MB of 64 MB$/.test(usage ?? ''), `("${usage}")`);

  report('no page errors', errors.length === 0, errors.slice(0, 4).join(' | '));
} catch (err) {
  report('ran to the end', false, String(err?.stack ?? err));
  for (const d of devices) await d.page?.screenshot({ path: join(ARTIFACTS, `t988-failed-${d.name}.png`) }).catch(() => {});
} finally {
  for (const d of devices) await d.close().catch(() => {});
  if (OWN_SERVER) { stopServer(); await pause(500); rmSync(serverDir, { recursive: true, force: true }); }
}
process.exit(failed ? 1 : 0);
