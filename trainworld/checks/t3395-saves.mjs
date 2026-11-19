// T-3395 stage 2: the game side of protecting saves from damage, against a local shonei-server with the
// save check (trainworldSaveCheck), two browser profiles: S signed in (dev sign-in), L signed out. Made-up
// damaged saves are the real NYC save cut to 60%; a save "the server stored before its check" is made by
// cutting its file in the server's scratch store after a whole upload.
//   S: a damaged slot edited here is refused 422 and keeps its bytes, with the sync note, and is never sent
//      again (one PUT over two loads); a slot whose account copy is damaged is left as it is; an account game
//      whose copy is damaged says so on Open, lists its pinned copy as "Last undamaged copy", and "Open a
//      copy" opens it as a new game; one with no pin says none is left; a damaged tied slot that fails to
//      load goes back to the start menu (T-3773), no slot is added and nothing is uploaded over the account's
//      copy, and its card there lists the pinned copy, which opens as a copy.
//   L: a damaged slot's newest intact backup becomes its good copy and opens as a copy; with no backup the
//      card says none is left; a damaged file says "This save is damaged.", a non-save "not a saved game";
//      a damaged slot never becomes a pull backup; autosave backups keep a good copy.
// Screenshots: artifacts/trainworld/t3395-sync-note-{light,dark}.png, t3395-restore-{light,dark}.png.
//   O (with --old-exe or --old-rev, a server from before the check): an account game whose copy is damaged
//      says so on Open, with no pinned row and no "none left" (the old server's missing pin says nothing); a
//      damaged slot edited here is never sent (the old server would store it), marked, with the sync note.
//   Also: with the codec worker blocked a whole save is "unchecked", never damaged, and a good slot is
//   neither marked nor kept aside; deleting a slot removes the copies kept aside as unreadable.
//   node trainworld/checks/t3395-saves.mjs [app url] [--exe <shonei.exe>] [--old-exe <shonei.exe>]   (from the maps root)
// Without --exe it builds ../shonei-server's committed HEAD (a git clone of it in a temp folder, so no tar);
// --old-rev <commit> builds the old server the same way (372a8ce: the server in production before the check). The server runs
// on 127.0.0.1:8083 (the page's ?acct=local) in a temp data folder. Prints PASS/FAIL lines; exit 1 on a failure.
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { launchChrome, REAL_SAVE, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const EXE = opt('--exe');
/** a shonei-server from before the save check (372a8ce), for the last phase, as an exe or a commit to build;
 * skipped without either */
let OLD_EXE = opt('--old-exe');
const OLD_REV = opt('--old-rev');
const APP = args.find((a, i) => !a.startsWith('--') && !['--exe', '--old-exe', '--old-rev'].includes(args[i - 1])) ?? 'http://localhost:6156/';
const SERVER = 'http://127.0.0.1:8083/';
const Q = 'debug=1&acct=local&demandWorkers=1';
const SUB = 't3395-' + Date.now().toString(36);

let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log((ok ? 'PASS' : 'FAIL') + '  ' + name + '  ' + detail); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 400) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

// ---- the saves
const REAL = new Uint8Array(readFileSync(REAL_SAVE));
const DAMAGED = REAL.slice(0, Math.floor(REAL.length * 0.6));
/** a whole gzip and header whose network is cut to 60%: the main thread's check (no track model) passes it,
 * the server's refuses it */
const NETCUT = (() => {
  const raw = gunzipSync(REAL), n = raw.readUInt32LE(4);
  return new Uint8Array(gzipSync(raw.subarray(0, 8 + n + Math.floor((raw.length - 8 - n) * 0.6))));
})();
/** the real save at save level `level`: past what the server's check reads, it is stored unverified and never pinned
 * (the browser only ever meets these copies cut short) */
const atLevel = (level) => {
  const raw = gunzipSync(REAL), n = raw.readUInt32LE(4);
  const h = JSON.parse(raw.subarray(8, 8 + n).toString('utf8'));
  h.level = level;
  const json = Buffer.from(JSON.stringify(h)), head = Buffer.alloc(8);
  head.write('TWG1', 0, 'latin1'); head.writeUInt32LE(json.length, 4);
  return new Uint8Array(gzipSync(Buffer.concat([head, json, raw.subarray(8 + n)])));
};

// ---- the server
let server = null;
const dataDir = mkdtempSync(join(tmpdir(), 'tw-t3395-srv-'));
if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error('a server already answers on 127.0.0.1:8083: stop it first');
let src = MAPS;
for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_games.go')); up++) src = join(src, '..');
src = join(src, '..', 'shonei-server');
/** shonei-server at commit `rev` built into the temp folder: a shared clone (git only, no tar) checked out there */
function buildServer(rev, name) {
  const tree = join(dataDir, 'src-' + name);
  if (spawnSync('git', ['clone', '-q', '--shared', '--no-checkout', src, tree]).status !== 0) throw new Error('git clone failed');
  const co = spawnSync('git', ['-C', tree, 'checkout', '-q', rev], { encoding: 'utf8' });
  if (co.status !== 0) throw new Error('git checkout ' + rev + ' failed: ' + co.stderr);
  const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
  const out = join(dataDir, name + '.exe');
  const b = spawnSync(go, ['build', '-p', '1', '-o', out, '.'], { cwd: tree, encoding: 'utf8' });
  if (b.status !== 0) throw new Error('go build failed: ' + b.stderr);
  return out;
}
const head = spawnSync('git', ['-C', src, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).stdout.trim();
const exe = EXE ?? buildServer(head, 'shonei');
/** one past the level the server's check reads (`twSaveLevel` at the server's HEAD); a fixed level went stale as the
 * server learned each new one */
const PAST_LEVEL = Number(/twSaveLevel\s*=\s*(\d+)/.exec(spawnSync('git', ['-C', src, 'show', head + ':trainworld_savecheck.go'], { encoding: 'utf8' }).stdout)?.[1]) + 1;
const LEVEL19 = atLevel(PAST_LEVEL);
if (!OLD_EXE && OLD_REV) OLD_EXE = buildServer(OLD_REV, 'shonei-old');
const run = join(dataDir, 'run');
mkdirSync(run);
for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(run, f));
server = spawn(exe, [], { cwd: run, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1' } });
if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the local server did not start');
const cfg = await (await fetch(SERVER + 'noritetsu/config')).json();
report('server has the save check', cfg.trainworldSaveCheck >= 1 && cfg.trainworldGames === true && cfg.dev === true, JSON.stringify({ check: cfg.trainworldSaveCheck }));

const signin = await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub: SUB }) })).json();
const token = signin.token, ACCT = signin.account.id;
const auth = { Authorization: 'Bearer ' + token };
const DEV = 'checkdevice0001';
const account = async () => (await fetch(SERVER + 'trainworld/games', { headers: auth })).json();
const gameOf = async (id) => (await account()).games.find((g) => g.id === id);
async function put(id, bytes, baseRev, name) {
  const fd = new FormData();
  fd.append('meta', JSON.stringify({ name, facts: { at: Date.now(), day: 3, origin: id, edit: Date.now() }, baseRev, device: DEV, force: false }));
  fd.append('save', new Blob([bytes]), 'game.save');
  const r = await fetch(SERVER + 'trainworld/games/' + id, { method: 'PUT', headers: auth, body: fd });
  return { status: r.status, json: await r.json().catch(() => null) };
}
function findFile(dir, name) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (e === name) return p;
    if (statSync(p).isDirectory()) { const f = findFile(p, name); if (f) return f; }
  }
  return null;
}
/** the stored file of game id's rev cut short, as a save stored before the check */
function cutStored(id, rev) {
  const p = findFile(run, id + '.r' + rev + '.save');
  if (!p) throw new Error('no stored file for ' + id + ' rev ' + rev);
  const b = readFileSync(p);
  writeFileSync(p, b.subarray(0, Math.floor(b.length * 0.6)));
}

const G1 = 'chkrefused0001', G2 = 'chkdownload002', G3 = 'chknoneleft003', G4 = 'chkfallback004', G5 = 'chkpinned00005', G7 = 'chknotsent0007';
const NAMES = { [G1]: 'Refused upload', [G2]: 'Download check', [G3]: 'None left', [G4]: 'Failed load', [G5]: 'Pinned restore', [G7]: 'Never sent' };
const r1 = await put(G1, REAL, 0, NAMES[G1]);
report('a whole save goes up and is pinned', r1.status === 200 && r1.json?.check === 'intact', JSON.stringify(r1.json));
await put(G2, REAL, 0, NAMES[G2]);
const r2 = await put(G2, LEVEL19, 1, NAMES[G2]);
report('a level ' + PAST_LEVEL + ' save is stored unverified', r2.status === 200 && r2.json?.check === 'unverified', JSON.stringify(r2.json));
cutStored(G2, 2);
await put(G3, LEVEL19, 0, NAMES[G3]);
cutStored(G3, 1);
await put(G4, REAL, 0, NAMES[G4]);
await put(G7, REAL, 0, NAMES[G7]);
await put(G5, REAL, 0, NAMES[G5]);
await put(G5, LEVEL19, 1, NAMES[G5]);
cutStored(G5, 2);
const before = await account();
report('the account: G2 and G5 current revs cut, pinned at rev 1; G3 with no pin',
  before.games.find((g) => g.id === G2)?.pinned?.rev === 1 && before.games.find((g) => g.id === G5)?.pinned?.rev === 1 && !before.games.find((g) => g.id === G3)?.pinned,
  JSON.stringify(before.games.map((g) => [g.id, g.rev, g.check, g.pinned?.rev])));

// ---- the browsers
const errors = [];
const devices = [];
async function device(name) {
  const d = await launchChrome({ prefix: 'tw-t3395-' + name + '-', size: [1300, 900], args: ANGLE });
  d.name = name;
  d.puts = {}; // PUTs by game id
  d.ctx.on('request', (r) => {
    const m = r.method() === 'PUT' && r.url().startsWith(SERVER + 'trainworld/games/') ? r.url().slice((SERVER + 'trainworld/games/').length).split(/[/?]/)[0] : null;
    if (m) d.puts[m] = (d.puts[m] ?? 0) + 1;
  });
  devices.push(d);
  return d;
}
async function open(d, query = '') {
  d.page ??= await d.ctx.newPage();
  const p = d.page;
  if (!p.listening) {
    p.listening = true;
    p.on('pageerror', (e) => errors.push(d.name + ': ' + e.message));
    p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource|did not load|T-1826/.test(m.text())) errors.push(d.name + ': ' + m.text()); });
  }
  await p.goto(APP + '?' + Q + query + '&v=' + Date.now(), { waitUntil: 'domcontentloaded', timeout: 120000 });
  return p;
}
const inGame = (p) => p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
const startMenu = (p, saves = true) => p.waitForSelector(saves ? '.start-menu .start-saves' : '.start-menu', { timeout: 120000 });
const theme = (p, shade) => p.evaluate(async (shade) => { (await import('/src/game/ui.ts')).shade.value = shade; }, shade);
/** what IndexedDB holds for a slot (bytes length, damaged, its record) and every key */
const storeOf = (p, key) => p.evaluate(async (key) => {
  const sf = await import('/src/workers/saveFile.ts');
  const s = await sf.readStored(key), rec = await sf.readSync(key);
  const keys = await new Promise((res) => { const r = indexedDB.open('anitabuilder'); r.onsuccess = () => { const q = r.result.transaction('saves').objectStore('saves').getAllKeys(); q.onsuccess = () => { res(q.result); r.result.close(); }; }; });
  return { bytes: s?.bytes.length ?? null, at: s?.at ?? null, damaged: !!s?.damaged, rec, keys };
}, key);
const slotsOf = (p) => p.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots()).map((s) => ({ key: s.key, name: s.name, lines: s.lines, damaged: !!s.damaged, cloud: s.cloud ?? null })));
/** slots written into this browser: [key, bytes, name, at, edit, sync record or null, backups {tier: [bytes, at]}] */
const inject = (p, slots) => p.evaluate(async (slots) => {
  const sf = await import('/src/workers/saveFile.ts');
  for (const [key, bytes, name, at, edit, rec, backups] of slots) {
    await sf.writeStored(key, { bytes: Uint8Array.from(bytes), at, day: 3, name, edit, origin: rec?.id ?? key, cities: ['nyc'], lines: 109, stations: 1756, cash: 100 });
    if (rec) await sf.writeSync(key, rec);
    for (const [tier, [b, bat]] of Object.entries(backups ?? {})) await sf.writeStored('backup:' + key + ':' + tier, { bytes: Uint8Array.from(b), at: bat, day: 2, edit: bat, origin: rec?.id ?? key, lines: 109, stations: 1756, cash: 100 });
  }
}, slots);
const shot = (p, name) => p.screenshot({ path: join(ARTIFACTS, name) });
const text = (p, sel) => p.locator(sel).first().innerText().catch(() => '');

const S = await device('s'), L = await device('l');
try {
  // ---- S: signed in
  let s = await open(S);
  await startMenu(s, false);
  const t0 = Date.now() - 3600_000, now = Date.now();
  await inject(s, [
    // G1: written here after its last sync (a player change), its network damaged inside a whole gzip, which
    // only the server's check (or the clock worker's) sees
    ['slot-g1', Array.from(NETCUT), NAMES[G1], now, now, { id: G1, acct: ACCT, rev: 1, syncedAt: t0, edit: t0 }],
    // G7: the same, its gzip cut short as Safari's damaged autosaves: the check here stops it before any request
    ['slot-g7', Array.from(DAMAGED), NAMES[G7], now, now, { id: G7, acct: ACCT, rev: 1, syncedAt: t0, edit: t0 }],
    // G2: as last synced at rev 1; the account has a rev 2 whose file is damaged
    ['slot-g2', Array.from(REAL), NAMES[G2], t0, t0, { id: G2, acct: ACCT, rev: 1, syncedAt: t0, edit: t0 }],
    // G4: as last synced, but damaged here (so it fails to load)
    ['slot-g4', Array.from(DAMAGED), NAMES[G4], t0 + 1000, t0 + 1000, { id: G4, acct: ACCT, rev: 1, syncedAt: t0 + 1000, edit: t0 + 1000 }],
  ]);
  // signed in, as the device that wrote the account's copies (so G1's change here is an ordinary upload, not a fork)
  await s.evaluate(([token, account, dev]) => { localStorage.setItem('noritetsu.auth.v1', JSON.stringify({ token, account })); localStorage.setItem('trainworld.device.v1', JSON.stringify(dev)); }, [token, signin.account, DEV]);
  s = await open(S);
  await startMenu(s);
  const g1 = await until(async () => { const x = await storeOf(s, 'slot-g1'); return x.rec?.error ? x : null; }, 30000);
  console.log('      ' + await s.locator('.save-damaged').first().evaluate((e) => e.outerHTML).catch((e) => String(e)));
  report('refused upload: the slot keeps its bytes', g1 && g1.bytes === NETCUT.length, JSON.stringify(g1 && { bytes: g1.bytes, damaged: g1.damaged }));
  report('refused upload: the sync record says why and which copy', g1?.rec?.error === 'This copy is damaged, so your account kept the last good one' && g1.rec.refused === g1.at && g1.rec.rev === 1, JSON.stringify(g1?.rec));
  const a1 = await gameOf(G1);
  report('refused upload: the account kept its copy', a1?.rev === 1 && a1?.pinned?.rev === 1, JSON.stringify({ rev: a1?.rev, pinned: a1?.pinned?.rev }));
  const qdir = findFile(run, 'quarantine');
  report('refused upload: the server kept the refused bytes aside', !!qdir && readdirSync(qdir).some((f) => f.endsWith('.save')), qdir ?? 'none');
  await until(() => text(s, 'li[data-origin] button[data-slot="slot-g1"] .save-damaged'), 10000);
  const note = await text(s, 'li[data-origin] button[data-slot="slot-g1"] .save-damaged');
  report('sync note on the card', note === 'This copy is damaged, so your account kept the last good one', JSON.stringify(note));
  for (const sh of ['light', 'dark']) { await theme(s, sh); await s.waitForTimeout(500); await s.locator('li[data-origin]:has(button[data-slot="slot-g1"])').scrollIntoViewIfNeeded(); await shot(s, 't3395-sync-note-' + sh + '.png'); }
  await theme(s, 'light');
  const g7 = await until(async () => { const x = await storeOf(s, 'slot-g7'); return x.rec?.error && x.damaged ? x : null; }, 30000);
  report('damaged slot: never sent, marked, with the sync note', !!g7 && !S.puts[G7] && g7.bytes === DAMAGED.length && g7.rec.error === 'This copy is damaged, so your account kept the last good one', JSON.stringify({ puts: S.puts[G7] ?? 0, damaged: g7?.damaged, rec: g7?.rec }));
  report('damaged slot: the account game is untouched', (await gameOf(G7))?.rev === 1);
  const g2 = await storeOf(s, 'slot-g2');
  report('damaged download: the slot is left as it was', g2.bytes === REAL.length && g2.rec?.rev === 1 && !g2.keys.includes('backup:slot-g2:pull'), JSON.stringify({ bytes: g2.bytes, rev: g2.rec?.rev }));

  // G3: on the account only, its only copy damaged and never pinned
  await s.click('button.start-save[data-slot="account:' + G3 + '"]');
  const refused3 = await until(() => text(s, '.start-refused'), 20000);
  report('damaged account game: Open says it is damaged', refused3 === 'This save is damaged.', JSON.stringify(refused3));
  const none3 = await until(() => text(s, 'li[data-origin]:has(button[data-slot="account:' + G3 + '"]) .save-none-left'), 10000);
  report('damaged account game with no pin: none left', none3 === 'No undamaged copy of this game is left', JSON.stringify(none3));
  report('no slot was made for it', !(await slotsOf(s)).some((x) => x.cloud?.id === G3));

  // G5: on the account only, current rev damaged, pinned rev 1
  await s.click('button.start-save[data-slot="account:' + G5 + '"]');
  const row5 = 'li[data-origin]:has(button[data-slot="account:' + G5 + '"]) li[data-kind="pinned"]';
  await until(() => s.locator(row5).count(), 20000);
  const label5 = await text(s, row5 + ' .save-version');
  report('damaged account game: its pinned copy is listed', /^Last undamaged copy · Saved/.test(label5.replace(/\s+/g, ' ')) && (await text(s, row5 + ' .save-version-open')) === 'Open a copy', JSON.stringify(label5));
  await s.click(row5 + ' .save-version-open');
  await inGame(s);
  const opened5 = await until(async () => { const l = await slotsOf(s); const act = await s.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active); const x = l.find((y) => y.key === act); return x?.lines ? { x, act } : null; }, 60000);
  report('the pinned copy opens as a new game', !!opened5 && opened5.x.lines > 100 && opened5.act.startsWith('slot-') && opened5.x.cloud?.id !== G5, JSON.stringify(opened5?.x));
  const a5 = await gameOf(G5);
  report('the damaged game on the account is untouched', a5?.rev === 2, 'rev ' + a5?.rev);

  // loads again: the refused copy is not sent again
  s = await open(S);
  await startMenu(s);
  await s.waitForTimeout(6000);
  report('the refused bytes are sent once over two loads; the cut-short ones never', S.puts[G1] === 1 && !S.puts[G7], JSON.stringify(S.puts));

  // G4: opening the damaged tied slot: back to the start menu, the slot as it was (T-3773)
  const putsBefore = S.puts[G4] ?? 0;
  const keys4 = (await slotsOf(s)).map((x) => x.key).sort().join();
  s = await open(S, '&slot=slot-g4&paused=1');
  const said = await until(() => s.evaluate(() => document.querySelector('.start-menu .start-refused')?.textContent ?? ''), 60000);
  report('failed load: back on the start menu, saying so', said === 'Failed load could not be opened.', JSON.stringify(said));
  await startMenu(s);
  await s.waitForTimeout(8000);
  const g4 = await storeOf(s, 'slot-g4');
  report('failed load: no slot added', (await slotsOf(s)).map((x) => x.key).sort().join() === keys4, JSON.stringify((await slotsOf(s)).map((x) => x.key)));
  report('failed load: the damaged slot is kept and marked', g4.bytes === DAMAGED.length && g4.damaged && g4.rec?.id === G4 && g4.keys.some((k) => String(k).startsWith('unreadable-')), JSON.stringify({ bytes: g4.bytes, damaged: g4.damaged }));
  const a4 = await gameOf(G4);
  report('failed load: nothing went up over the account game', (S.puts[G4] ?? 0) === putsBefore && a4?.rev === 1, JSON.stringify({ puts: S.puts[G4] ?? 0, rev: a4?.rev }));
  const card4 = '.start-saves > li:has(button[data-slot="slot-g4"])';
  const row4 = card4 + ' li[data-kind="pinned"]';
  await until(() => s.locator(row4).count(), 10000);
  report('menu: the damaged card says so', (await text(s, card4 + ' .save-damaged')) === 'This save is damaged.');
  report('menu: its last undamaged copy is listed, open', /^Last undamaged copy · Saved/.test((await text(s, row4 + ' .save-version')).replace(/\s+/g, ' ')) && (await text(s, row4 + ' .save-version-open')) === 'Open a copy');
  for (const sh of ['light', 'dark']) { await theme(s, sh); await s.waitForTimeout(500); await s.locator(card4).scrollIntoViewIfNeeded(); await shot(s, 't3395-restore-' + sh + '.png'); }
  await theme(s, 'light');
  await s.click(row4 + ' .save-version-open');
  await inGame(s);
  const opened4 = await until(async () => { const l = await slotsOf(s); const act = await s.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active); const x = l.find((y) => y.key === act); return x?.lines && act !== 'slot-g4' ? { x, act } : null; }, 60000);
  report('Open a copy: the pinned copy plays in a slot of its own', !!opened4 && opened4.x.lines > 100, JSON.stringify(opened4?.x));
  const after4 = await storeOf(s, 'slot-g4');
  report('Open a copy: the damaged original stays', after4.bytes === DAMAGED.length && (await gameOf(G4))?.rev === 1);

  // ---- L: signed out
  let l = await open(L);
  await startMenu(l, false);
  await inject(l, [
    ['slot-l1', Array.from(DAMAGED), 'Local damaged', now, now, null, { t1: [Array.from(DAMAGED), now - 600_000], t2: [Array.from(REAL), now - 7200_000] }],
    ['slot-l2', Array.from(DAMAGED), 'Local none left', now - 5000, now - 5000, null, {}],
  ]);
  // the module-level rules, in the page
  const unit = await l.evaluate(async ([real, damaged]) => {
    const sf = await import('/src/workers/saveFile.ts'), cc = await import('/src/workers/clockClient.ts');
    const R = Uint8Array.from(real), D = Uint8Array.from(damaged), junk = new Uint8Array(500).map((_, i) => i * 7);
    const out = { checks: [await sf.checkSave(R), await sf.checkSave(D), await sf.checkSave(junk)] };
    out.fileDamaged = await cc.fileChoice(new File([D], 'x.save'));
    out.fileJunk = await cc.fileChoice(new File([junk], 'y.save'));
    await sf.writeStored('slot-u1', { bytes: D, at: 1000, day: 1 });
    await sf.backupBeforeReplace('slot-u1', { at: 2000, bytes: R });
    out.pullOfDamaged = !!await sf.readStored('backup:slot-u1:pull');
    out.keptAside = !!await sf.readStored(sf.unreadableKey('slot-u1', 1000));
    await sf.writeStored('slot-u2', { bytes: R, at: 1000, day: 1 });
    await sf.backupBeforeReplace('slot-u2', { at: 2000, bytes: R.slice() });
    out.pullOfWhole = !!await sf.readStored('backup:slot-u2:pull');
    out.goodOfWhole = (await sf.readStored('backup:slot-u2:good'))?.at;
    await sf.writeBackup('slot-u3', { bytes: R, at: Date.now(), day: 1 });
    out.backupGood = !!await sf.readStored('backup:slot-u3:good') && !!await sf.readStored('backup:slot-u3:t0');
    for (const k of ['slot-u1', 'slot-u2', 'slot-u3']) await sf.deleteSlot(k);
    out.asideAfterDelete = !!await sf.readStored(sf.unreadableKey('slot-u1', 1000));
    return out;
  }, [Array.from(REAL), Array.from(DAMAGED)]);
  report('checkSave: whole, cut short, not a save', JSON.stringify(unit.checks) === '["intact","damaged","other"]', JSON.stringify(unit.checks));
  report('a damaged file says so; a non-save says it is not one', unit.fileDamaged === 'This save is damaged.' && unit.fileJunk === 'This file is not a saved game.', JSON.stringify([unit.fileDamaged, unit.fileJunk]));
  report('a damaged slot never becomes a pull backup (kept aside instead)', !unit.pullOfDamaged && unit.keptAside, JSON.stringify(unit));
  report('a whole slot does, and is the good copy', unit.pullOfWhole && unit.goodOfWhole === 1000);
  report('deleting a slot removes its copy kept aside', unit.keptAside && !unit.asideAfterDelete);
  report('an autosave backup keeps the good copy beside tier 0', unit.backupGood);

  // T-3773: each fails to open and the page goes back to the start menu
  const backOnMenu = async (name) => /could not be opened/.test(await until(() => l.evaluate(() => document.querySelector('.start-menu .start-refused')?.textContent ?? ''), 60000) ?? '') && !!await startMenu(l);
  l = await open(L, '&slot=slot-l2&paused=1');
  report('a damaged browser game goes back to the menu', await backOnMenu());
  l = await open(L, '&slot=slot-l1&paused=1');
  report('another damaged browser game goes back to the menu', await backOnMenu());
  const l1 = await until(async () => { const x = await storeOf(l, 'slot-l1'); return x.keys.includes('backup:slot-l1:good') ? x : null; }, 15000);
  const good = await l.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).readStored('backup:slot-l1:good'))?.bytes.length);
  report('damaged slot: its newest whole backup becomes the good copy', !!l1 && l1.damaged && good === REAL.length, JSON.stringify({ good }));
  const cardL = (key) => `.start-saves > li:has(button[data-slot="${key}"])`;
  await until(() => l.locator(cardL('slot-l1') + ' li[data-version]').count(), 10000);
  const none2 = await text(l, cardL('slot-l2') + ' .save-none-left');
  report('a damaged browser game with no whole backup: none left', none2 === 'No undamaged copy of this game is left', JSON.stringify(none2));
  const row1 = cardL('slot-l1') + ' li[data-version="backup:slot-l1:good"]';
  report('menu: the good copy is listed first, as the last undamaged copy',
    (await l.locator(cardL('slot-l1') + ' li[data-version]').first().getAttribute('data-version')) === 'backup:slot-l1:good' && /^Last undamaged copy/.test(await text(l, row1)));
  await l.click(row1 + ' .save-version-open');
  await inGame(l);
  const openedL = await until(async () => { const ls = await slotsOf(l); const act = await l.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active); const x = ls.find((y) => y.key === act); return x?.lines && !['slot-l1', 'slot-l2'].includes(act) ? x : null; }, 60000);
  report('the good copy opens as a copy; the damaged slot stays', !!openedL && openedL.lines > 100 && (await storeOf(l, 'slot-l1')).bytes === DAMAGED.length, JSON.stringify(openedL));
  // slot-l2 failed to load, so its bytes were kept aside; deleting the game takes them too
  const asideOf = async (key) => (await storeOf(l, key)).keys.filter((k) => String(k).startsWith('unreadable-') && String(k).endsWith(':' + key));
  const aside2 = await asideOf('slot-l2');
  await l.evaluate(async () => (await import('/src/workers/saveFile.ts')).deleteSlot('slot-l2'));
  report('deleting a game removes its copies kept aside as unreadable', aside2.length === 1 && (await asideOf('slot-l2')).length === 0, JSON.stringify(aside2));

  // the codec worker blocked: nothing can be checked on the page, and nothing is called damaged
  const b = await L.ctx.newPage();
  await b.route('**/gzip.worker*', (r) => r.abort());
  await b.goto(APP + '?new=1&debug=1&demandWorkers=1&v=' + Date.now(), { waitUntil: 'domcontentloaded', timeout: 120000 });
  await b.waitForTimeout(3000);
  const blocked = await b.evaluate(async (real) => {
    const sf = await import('/src/workers/saveFile.ts'), cc = await import('/src/workers/clockClient.ts');
    const R = Uint8Array.from(real);
    const out = { check: await sf.checkSave(R) };
    try { await sf.decodeSave(R); out.decode = 'ok'; } catch (e) { out.decode = String(e.message); }
    out.file = await cc.fileChoice(new File([R], 'z.save'));
    await sf.writeStored('slot-b1', { bytes: R, at: 1000, day: 3, name: 'Blocked' });
    out.slot = await sf.checkSlot('slot-b1');
    out.marked = !!(await sf.readStored('slot-b1'))?.damaged;
    await sf.backupBeforeReplace('slot-b1', { at: 2000, bytes: R.slice(0, 100) });
    const keys = await new Promise((res) => { const r = indexedDB.open('anitabuilder'); r.onsuccess = () => { const q = r.result.transaction('saves').objectStore('saves').getAllKeys(); q.onsuccess = () => { res(q.result); r.result.close(); }; }; });
    out.keys = keys.filter((k) => String(k).includes('slot-b1'));
    await sf.deleteSlot('slot-b1');
    return out;
  }, Array.from(REAL));
  await b.close();
  report('codec worker blocked: a whole save is unchecked, never damaged', blocked.check === 'unchecked' && blocked.slot === 'unchecked' && !blocked.marked && /could not be read/.test(blocked.decode) && blocked.file === 'Failed to load :( Try reloading the page.', JSON.stringify({ ...blocked, keys: undefined }));
  report('codec worker blocked: the slot becomes its pull backup, nothing kept aside', blocked.keys.includes('backup:slot-b1:pull') && !blocked.keys.some((k) => String(k).startsWith('unreadable-')), JSON.stringify(blocked.keys));
  // ---- O: a server from before the check
  if (OLD_EXE) {
    server.kill();
    await pause(1500);
    const run2 = join(dataDir, 'run-old');
    mkdirSync(run2);
    for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(run2, f));
    server = spawn(OLD_EXE, [], { cwd: run2, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1' } });
    if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the old server did not start');
    const ocfg = await (await fetch(SERVER + 'noritetsu/config')).json();
    report('old server: no save check', ocfg.trainworldSaveCheck === undefined && ocfg.trainworldGames === true);
    const os = await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub: SUB + 'o' }) })).json();
    const G6 = 'chkoldserver06';
    const fd = new FormData();
    fd.append('meta', JSON.stringify({ name: 'Old server', facts: { at: Date.now(), day: 3, origin: G6 }, baseRev: 0, device: DEV, force: false }));
    fd.append('save', new Blob([DAMAGED]), 'game.save');
    const up = await fetch(SERVER + 'trainworld/games/' + G6, { method: 'PUT', headers: { Authorization: 'Bearer ' + os.token }, body: fd });
    report('old server: takes a damaged upload, as before', up.status === 200, 'status ' + up.status);
    // G8: whole on the account at rev 1, cut short here after a player change: the old server would take it
    const G8 = 'chkoldnosend08';
    const fd8 = new FormData();
    fd8.append('meta', JSON.stringify({ name: 'Old server cut here', facts: { at: Date.now(), day: 3, origin: G8 }, baseRev: 0, device: DEV, force: false }));
    fd8.append('save', new Blob([REAL]), 'game.save');
    const up8 = await fetch(SERVER + 'trainworld/games/' + G8, { method: 'PUT', headers: { Authorization: 'Bearer ' + os.token }, body: fd8 });
    const O = await device('o');
    let o = await open(O);
    await startMenu(o, false);
    const t8 = Date.now() - 3600_000, n8 = Date.now();
    // a game kept here only, so the menu lists saved games (a browser with none opens on the new game sheet)
    await inject(o, [
      ['slot-o1', Array.from(REAL), 'Kept here', Date.now(), Date.now(), { id: 'chkkeptlocal01', rev: 0, syncedAt: 0, local: true }],
      ['slot-o2', Array.from(DAMAGED), 'Old server cut here', n8, n8, { id: G8, acct: os.account.id, rev: 1, syncedAt: t8, edit: t8 }],
    ]);
    await o.evaluate(([token, account, dev]) => { localStorage.setItem('noritetsu.auth.v1', JSON.stringify({ token, account })); localStorage.setItem('trainworld.device.v1', JSON.stringify(dev)); }, [os.token, os.account, DEV]);
    o = await open(O);
    await startMenu(o);
    await o.click('button.start-save[data-slot="account:' + G6 + '"]');
    const said6 = await until(() => text(o, '.start-refused'), 20000);
    await o.waitForTimeout(1000);
    const card6 = 'li[data-origin]:has(button[data-slot="account:' + G6 + '"])';
    report('old server: Open says damaged, no pinned row, no "none left"', said6 === 'This save is damaged.' && await o.locator(card6 + ' li[data-kind="pinned"]').count() === 0 && await o.locator(card6 + ' .save-none-left').count() === 0, JSON.stringify(said6));
    report('old server: no slot made for it', !(await slotsOf(o)).some((x) => x.cloud?.id === G6));
    const o2 = await until(async () => { const x = await storeOf(o, 'slot-o2'); return x.rec?.error && x.damaged ? x : null; }, 30000);
    const a8 = (await (await fetch(SERVER + 'trainworld/games', { headers: { Authorization: 'Bearer ' + os.token } })).json()).games.find((g) => g.id === G8);
    const bytes8 = (await (await fetch(SERVER + 'trainworld/games/' + G8, { headers: { Authorization: 'Bearer ' + os.token } })).arrayBuffer()).byteLength;
    report('old server: a damaged slot is never sent (no PUT at all)', up8.status === 200 && Object.keys(O.puts).length === 0 && a8?.rev === 1 && bytes8 === REAL.length, JSON.stringify({ puts: O.puts, rev: a8?.rev, bytes8 }));
    report('old server: the slot is marked, with the sync note', !!o2 && o2.bytes === DAMAGED.length && o2.rec.error === 'This copy is damaged, so your account kept the last good one', JSON.stringify(o2?.rec));
  } else console.log('SKIP  old server phase (no --old-exe or --old-rev)');
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
} finally {
  report('no page errors', errors.length === 0, errors.slice(0, 5).join(' | '));
  for (const d of devices) await d.close?.().catch(() => {});
  server?.kill();
  await pause(500);
  try { rmSync(dataDir, { recursive: true, force: true }); } catch { /* left in temp */ }
}
console.log(failed ? 'FAIL' : 'ALL PASS');
process.exit(failed ? 1 : 0);
