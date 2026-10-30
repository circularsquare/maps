// T-1832: the "Saved in this browser only" card after an hour of play (here `?keepMs=3000`, three
// seconds). Against a local shonei-server (as t988-cloud.mjs) with one browser profile:
//   - a `?new=1` page that owns no slot shows no card;
//   - a game saved into a slot shows it (Sign in, Download), light and dark screenshots;
//   - the x closes it for good: a reload and a wait show none;
//   - a game downloaded from the card, or signed in and added to the account, never shows it;
//   - Sign in opens the account panel.
//   node trainworld/checks/t1832-keep.mjs [app url] [--server]     (from the maps root; default localhost:6199)
// Prints PASS or FAIL lines and exits 1 on a failure. Screenshots go to artifacts/trainworld/t1832/.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const OWN_SERVER = !args.includes('--server');
const APP = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:6199/';
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
  server = spawn(join(dir, 'shonei.exe'), [], { cwd: dir, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: '' } });
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the local server did not start');
}

const errors = [];
const d = await launchChrome({ prefix: 'tw-t1832-', size: [1300, 900], args: ANGLE });
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource/.test(m.text())) errors.push(m.text()); });
const go = (query) => p.goto(`${APP}?${Q}${query}&v=${Date.now()}`);
const inGame = async () => {
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await p.addStyleTag({ content: '#debug-root { display: none !important; }' }); // the debug panel covers the card
};
const card = () => p.locator('.keep-card');
const cardIn = (ms) => card().waitFor({ state: 'visible', timeout: ms }).then(() => true, () => false);
const keepStore = () => p.evaluate(() => localStorage.getItem('anitabuilder-keep'));
const theme = (shade) => p.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; }, shade);
const active = () => p.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
const forget = () => p.evaluate(() => localStorage.removeItem('anitabuilder-keep'));

try {
  // 1. a `?new=1` page owns no slot: no card however long it runs
  await go('&new=1');
  await inGame();
  report('?new=1: no card', !await cardIn(14000), '(14 s)');

  // 2. saved into a slot: the card comes, with Sign in and Download
  await p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Keep check'));
  const slot = await until(async () => { const a = await active(); return a.startsWith('slot-') ? a : null; }, 30000, 500);
  const shown = await cardIn(20000);
  const text = shown ? (await card().innerText()).replace(/\s+/g, ' ') : '';
  report('saved game: card after the hour', !!slot && shown, `(${slot})`);
  console.log('     card:', text);
  report('card words and buttons', /Saved in this browser only/.test(text) && /Sign in/.test(text) && /Download/.test(text));
  await p.screenshot({ path: join(OUT, 'card-light.png') });
  await theme('dark');
  await p.waitForTimeout(400);
  await p.screenshot({ path: join(OUT, 'card-dark.png') });
  await theme('light');

  // 3. the x closes it for good
  await card().locator('.keep-x').click();
  report('x closes the card', !await cardIn(1500));
  const stored = JSON.parse(await keepStore());
  report('x is kept for the slot', stored?.[slot]?.closed === true, JSON.stringify(stored));
  await go('&continue=1');
  await inGame();
  report('after a reload and a wait: still closed', !await cardIn(16000));

  // 4. Download from the card hides it for good
  await forget();
  await go('&continue=1');
  await inGame();
  report('card comes back with a fresh count', await cardIn(20000));
  const [dl] = await Promise.all([p.waitForEvent('download', { timeout: 20000 }), card().locator('button:has-text("Download")').click()]);
  report('Download saves a file', /^trainsworld-.*\.save$/.test(dl.suggestedFilename()), dl.suggestedFilename());
  report('card goes after Download', !await cardIn(1500));
  await go('&continue=1');
  await inGame();
  report('after a reload and a wait: no card', !await cardIn(16000));

  // 5. Sign in opens the account panel; adding the game to the account ends the card for good
  await forget();
  await go('&continue=1');
  await inGame();
  await cardIn(20000);
  await card().locator('button:has-text("Sign in")').click();
  const opened = await p.locator('dialog.acct-panel[open]').waitFor({ timeout: 10000 }).then(() => true, () => false);
  report('Sign in opens the account panel', opened);
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
  await forget();
  await go('&continue=1');
  await inGame();
  report('signed in and on the account: no card', !await cardIn(16000));
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
  await p.screenshot({ path: join(OUT, 'failure.png') }).catch(() => {});
} finally {
  await d.close?.().catch(() => {});
  server?.kill();
}
process.exit(failed ? 1 : 0);
