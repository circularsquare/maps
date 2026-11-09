// T-1832: the "Saved only in this browser" dot beside the search glass after an hour of play (here
// `?keepMs=3000`, three seconds). Against a local shonei-server (as t988-cloud.mjs), one browser profile:
//   - a `?new=1` page that owns no slot shows no dot;
//   - a game saved into a slot shows it, with its tip, just left of the search glass; a reload keeps it;
//   - a download, or the game added to the account, takes it away for good;
//   - clicking it opens the account panel;
//   - with accounts off (`?acct=off`) a game on no account shows the download-only tip and a click downloads it.
//   node trainworld/checks/t1832-keep.mjs [app url] [--server]     (from the maps root; default localhost:6191)
// Prints PASS or FAIL lines and exits 1 on a failure. Screenshots go to artifacts/trainworld/t1832/.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const OWN_SERVER = !args.includes('--server');
const APP = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:6191/';
const SERVER = 'http://127.0.0.1:8083/';
const Q = 'debug=1&acct=local&keepMs=3000';
const SUB = 't1832-check-' + Date.now().toString(36);
const OUT = join(ARTIFACTS, 't1832');
mkdirSync(OUT, { recursive: true });

let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 400) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

let server = null;
if (OWN_SERVER) {
  if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error('a server already answers on 127.0.0.1:8083: stop it, or pass --server');
  const dir = mkdtempSync(join(tmpdir(), 'tw-t1832-srv-'));
  let src = MAPS;
  for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_games.go')); up++) src = join(src, '..');
  src = join(src, '..', 'shonei-server');
  const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
  const b = spawnSync(go, ['build', '-o', join(dir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
  if (b.status !== 0) throw new Error(`go build failed: ${b.stderr ?? b.error}`);
  for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dir, f));
  server = spawn(join(dir, 'shonei.exe'), [], { cwd: dir, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1' } });
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the local server did not start');
}

const errors = [];
const d = await launchChrome({ prefix: 'tw-t1832-', size: [1300, 900], args: ANGLE });
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource/.test(m.text())) errors.push(m.text()); });
const go = (query) => p.goto(`${APP}?${Q}${query}&v=${Date.now()}`);
const goOff = (query) => p.goto(`${APP}?debug=1&acct=off&keepMs=3000${query}&v=${Date.now()}`);
const inGame = async () => {
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await p.addStyleTag({ content: '#debug-root { display: none !important; }' }); // the debug panel covers the map's corner
};
const dot = () => p.locator('.keep-dot');
const dotIn = (ms) => dot().waitFor({ state: 'visible', timeout: ms }).then(() => true, () => false);
const theme = (shade) => p.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; }, shade);
const active = () => p.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
const forget = () => p.evaluate(() => localStorage.removeItem('anitabuilder-keep'));
const TIP = 'Saved only in this browser!\nIf your browser data is cleared, this game will be lost. You can make an account to save it online, or download a copy locally.';
const TIP_OFF = 'Saved only in this browser!\nIf your browser data is cleared, this game will be lost. You can download a copy to keep it.';

try {
  // 1. a `?new=1` page owns no slot: no dot however long it runs
  await go('&new=1');
  await inGame();
  report('?new=1: no dot', !await dotIn(14000), '(14 s)');

  // 2. saved into a slot: the dot comes, with its tip, right of the search glass
  await p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Keep check'));
  const slot = await until(async () => { const a = await active(); return a.startsWith('slot-') ? a : null; }, 30000, 500);
  const shown = await dotIn(20000);
  report('saved game: dot after the hour', !!slot && shown, `(${slot})`);
  const tip = await dot().getAttribute('data-tip').catch(() => null);
  report('tip words', tip === TIP, JSON.stringify(tip));
  const [db, sb] = await Promise.all([dot().boundingBox(), p.locator('.search-open').boundingBox()]);
  report('dot sits just right of the search glass (T-1868)', !!db && !!sb && sb.x + sb.width <= db.x && db.x - (sb.x + sb.width) < 16 && Math.abs((db.y + db.height / 2) - (sb.y + sb.height / 2)) < 3,
    JSON.stringify({ dot: db, glass: sb }));
  for (const sh of ['light', 'dark']) {
    await theme(sh);
    await p.mouse.move(5, 500);
    await p.waitForTimeout(300);
    await p.screenshot({ path: join(OUT, `dot-${sh}.png`), clip: { x: 250, y: 40, width: 400, height: 300 } });
    await p.mouse.move(db.x + db.width / 2, db.y + db.height / 2);
    await p.waitForTimeout(900);
    await p.screenshot({ path: join(OUT, `dot-hover-${sh}.png`), clip: { x: 250, y: 40, width: 550, height: 340 } });
  }
  await theme('light');
  await p.mouse.move(5, 500);
  // no way to close it: a reload keeps it
  await go('&continue=1');
  await inGame();
  report('after a reload: the dot is still there', await dotIn(20000));

  // 3. downloading the game (the saved games panel's Download) takes it away for good
  const [dl] = await Promise.all([p.waitForEvent('download', { timeout: 20000 }), p.evaluate(async () => (await import('/src/workers/clockClient.ts')).exportGame())]);
  report('Download saves a file', /^trainsworld-.*\.save$/.test(dl.suggestedFilename()), dl.suggestedFilename());
  report('dot goes after Download', !await dotIn(1500));
  await go('&continue=1');
  await inGame();
  report('after a reload and a wait: no dot', !await dotIn(16000));

  // 4. the dot opens the account panel; adding the game to the account takes it away
  await forget();
  await go('&continue=1');
  await inGame();
  await dotIn(20000);
  await dot().click();
  const opened = await p.locator('dialog.acct-panel[open]').waitFor({ timeout: 10000 }).then(() => true, () => false);
  report('the dot opens the account panel', opened);
  await p.waitForSelector('dialog.acct-panel[open] input[name="sub"]', { timeout: 30000 });
  await p.fill('dialog.acct-panel input[name="sub"]', SUB);
  await p.click('dialog.acct-panel .acct-row button[type="submit"]');
  await p.waitForSelector('.acct-merge button:text-is("Add")', { timeout: 30000 });
  await p.click('.acct-merge button:text-is("Add")');
  await p.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = false; });
  const onAcct = await until(() => p.evaluate(async () => {
    const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
    return !!s.slots.find((x) => x.key === s.active)?.cloud?.acct;
  }), 30000);
  report('game is on the account', !!onAcct);
  report('dot goes once the game is on the account', !await dotIn(1500));
  await forget();
  await go('&continue=1');
  await inGame();
  report('signed in and on the account: no dot', !await dotIn(16000));

  // 5. accounts off (`?acct=off`): the same game, already on an account, shows no dot; a new game kept only
  // here shows the download-only tip, and a click downloads it and takes the dot away
  await goOff('&continue=1');
  await inGame();
  report('accounts off, game on an account: no dot', !await dotIn(16000));
  await goOff('&new=1');
  await inGame();
  await p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Keep check 2'));
  report('accounts off: dot after the hour', await dotIn(30000));
  const tip2 = await dot().getAttribute('data-tip').catch(() => null);
  report('accounts off: tip words', tip2 === TIP_OFF, JSON.stringify(tip2));
  const b2 = await dot().boundingBox();
  for (const sh of ['light', 'dark']) {
    await theme(sh);
    await p.mouse.move(b2.x + b2.width / 2, b2.y + b2.height / 2);
    await p.waitForTimeout(900);
    await p.screenshot({ path: join(OUT, `dot-off-hover-${sh}.png`), clip: { x: 750, y: 0, width: 550, height: 200 } });
  }
  await theme('light');
  const [dl2] = await Promise.all([p.waitForEvent('download', { timeout: 20000 }), dot().click()]);
  report('accounts off: a click downloads the game', /^trainsworld-.*\.save$/.test(dl2.suggestedFilename()), dl2.suggestedFilename());
  report('accounts off: dot goes after the download', !await dotIn(1500));
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
  await p.screenshot({ path: join(OUT, 'failure.png') }).catch(() => {});
} finally {
  await d.close?.().catch(() => {});
  server?.kill();
}
process.exit(failed ? 1 : 0);
