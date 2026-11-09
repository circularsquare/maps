// T-2738: a card's delete removes every version of the game, in this browser and on the account, in
// one go, against a local shonei-server (dev sign-in) with every request to it slowed by LAT ms, as
// over the internet. Each game is seeded as slots tied to the account and account-only versions of
// one origin. (1) The start menu's delete. (2) The saves panel's delete of a card not being played
// while the server refuses deletes: the panel says so, the card stays gone, and the account's copies
// go at the next load. (3) The saves panel's delete of the game being played (the page goes to the
// start menu). After each: no slot or account game of that origin, and no card after a reload and the
// start menu's sync. Also the confirm's buttons, "Delete" (red) and "Keep".
//   node trainworld/checks/t2738-delete.mjs [app url] [--server]     (from the maps root; default localhost:6190)
// Without --server it builds ../shonei-server and runs it on 127.0.0.1:8083 in a temp folder (the
// page's `?acct=local`); with --server one is already running there in dev mode (SHONEI_SECRET set, SHONEI_DEV=1).
// Prints PASS or FAIL lines and exits 1 on a failure. About two minutes.
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, REAL_SAVE, ANGLE, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const OWN_SERVER = !args.includes('--server');
const APP = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:6190/';
const SERVER = 'http://127.0.0.1:8083/';
const Q = 'debug=1&acct=local&demandWorkers=1';
const SUB = 't2738-check-' + Date.now().toString(36);
const LAT = 150;

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
if (OWN_SERVER) {
  if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error('a server already answers on 127.0.0.1:8083: stop it, or pass --server to use it');
  serverDir = mkdtempSync(join(tmpdir(), 'tw-t2738-srv-'));
  let src = MAPS;
  for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_games.go')); up++) src = join(src, '..');
  src = join(src, '..', 'shonei-server');
  const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
  const b = spawnSync(go, ['build', '-o', join(serverDir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
  if (b.status !== 0) throw new Error(`go build in ${src} failed: ${b.stderr ?? b.error}`);
  for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(serverDir, f));
  await startServer();
}
const token = (await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub: SUB }) })).json()).token;
const account = async () => (await fetch(SERVER + 'trainworld/games', { headers: { Authorization: 'Bearer ' + token } })).json();
const onAccount = async (origin) => (await account()).games.filter((g) => g.facts?.origin === origin).map((g) => g.id);

const bytes = Array.from(readFileSync(REAL_SAVE));
const errors = [];
const d = await launchChrome({ prefix: 'tw-t2738-', size: [1300, 900], args: ANGLE });
// every request to the account server waits LAT ms; `refuse` makes the deletes fail
let refuse = false;
await d.ctx.route(SERVER + '**', async (r) => {
  await pause(LAT);
  if (refuse && r.request().method() === 'DELETE') return r.abort();
  return r.continue().catch(() => {});
});
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource/.test(m.text())) errors.push(m.text()); });
const menu = async () => {
  await p.goto(`${APP}?${Q}&v=${Date.now()}`);
  await p.waitForSelector('.start-menu', { timeout: 120000 });
};
/** the start menu settled against the account (its list and the slots read again) */
const settled = () => p.evaluate(async () => (await import('/src/game/accountGames.ts')).settleOnStart());
const inBrowser = (origin) => p.evaluate(async (o) => (await (await import('/src/workers/saveFile.ts')).listSlots()).filter((s) => s.origin === o).map((s) => s.key), origin);
const queued = () => p.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).readCloudDeletes()).map((x) => x.id));
/** game `origin` as `slots` slots tied to the account and `only` versions on the account alone */
const seed = (origin, slots, only) => p.evaluate(async ([origin, slots, only, bytes]) => {
  const sf = await import('/src/workers/saveFile.ts'), cloud = await import('/src/workers/cloud.ts'), acct = await import('/src/game/account.ts');
  const ctx = acct.cloudContext();
  if (!ctx?.on) throw new Error('not signed in');
  const b = Uint8Array.from(bytes), now = Date.now(), keys = [];
  for (let i = 0; i < slots + only; i++) {
    const key = `slot-t2738-${origin}-${i}`, at = now - i * 3600_000;
    await sf.writeSlot(key, { bytes: b, at, edit: at, origin, day: 30 - i, name: 'Delete ' + origin, cities: ['nyc'], lines: 10, stations: 40 + i, cash: 1000 });
    await sf.writeSync(key, { id: cloud.newGameId(), acct: ctx.acct, rev: 0, syncedAt: 0 });
    if (await cloud.upload(ctx, key) !== 'ok') throw new Error('upload failed');
    if (i >= slots) await sf.deleteSlot(key); else keys.push(key);
  }
  return keys;
}, [origin, slots, only, bytes]);
const card = (sel, origin) => p.locator(`${sel} li[data-origin="${origin}"]`).count();

try {
  await menu();
  // a first visit opens on the new game sheet, without the menu's account button
  await until(() => p.evaluate(async () => (await import('/src/game/account.ts')).accountState.value === 'on'), 30000);
  await p.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = true; });
  await p.waitForSelector('dialog.acct-panel[open] input[name="sub"]', { timeout: 30000 });
  await p.fill('dialog.acct-panel input[name="sub"]', SUB);
  await p.click('dialog.acct-panel .acct-row button[type="submit"]');
  await until(() => p.evaluate(async () => !!(await import('/src/game/account.ts')).cloudContext()?.on), 20000);
  await p.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = false; });

  // ---- 1. the start menu's delete: two slots and two account-only versions
  await seed('menu', 2, 2);
  report('seeded: four versions on the account', (await onAccount('menu')).length === 4, `(${(await onAccount('menu')).length})`);
  await menu();
  await settled();
  await until(() => card('.start-saves', 'menu'), 20000);
  await p.click('.start-saves li[data-origin="menu"] .start-delete');
  const ask = await p.locator('.start-saves li[data-origin="menu"] .callout p').textContent();
  const buttons = await p.locator('.start-saves li[data-origin="menu"] .callout button').allTextContents();
  const red = await p.locator('.start-saves li[data-origin="menu"] .callout button').first().evaluate((b) => getComputedStyle(b).color);
  const ink = await p.locator('.start-saves li[data-origin="menu"] .callout button').nth(1).evaluate((b) => getComputedStyle(b).color);
  report('the question and its buttons', /^Delete "Delete menu" and its \d+ other versions from this browser and your account\?$/.test(ask ?? '') && buttons.join('|') === 'Delete|Keep' && red !== ink,
    `("${ask}"; ${buttons.join(', ')}; ${red} vs ${ink})`);
  await p.click('.start-saves li[data-origin="menu"] .callout button:first-child');
  const gone1 = await until(async () => !(await onAccount('menu')).length && !(await inBrowser('menu')).length && !(await card('.start-saves', 'menu')), 20000);
  await menu();
  await settled();
  await p.waitForTimeout(1000);
  report('start menu: every version gone, here and on the account, after a reload', !!gone1 && !(await card('.start-saves', 'menu')) && !(await onAccount('menu')).length && !(await inBrowser('menu')).length,
    `(account ${(await onAccount('menu')).length}, browser ${(await inBrowser('menu')).length}, card ${await card('.start-saves', 'menu')})`);

  // ---- 2, 3: in a game; the panel deletes another game while the server refuses, then the game played
  const playKeys = await seed('play', 3, 1);
  await seed('other', 2, 1);
  await menu();
  await settled();
  await until(() => card('.start-saves', 'play'), 20000);
  await p.click(`.start-save[data-slot="${playKeys[0]}"]`);
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await p.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await p.waitForSelector('.saves-panel .save-files');
  await until(() => card('.save-slots', 'other'), 20000);
  refuse = true;
  await p.click('.save-slots li[data-origin="other"] .save-actions .save-delete');
  const panelButtons = await p.locator('.save-slots li[data-origin="other"] > .callout button').allTextContents();
  await p.click('.save-slots li[data-origin="other"] > .callout button:first-child');
  const said = await until(() => p.locator('.saves-panel .callout.bad, .saves-panel .save-delete-failed').first().textContent(), 15000);
  const otherHere = await until(async () => !(await inBrowser('other')).length && !(await card('.save-slots', 'other')), 15000);
  report('panel, server refusing: says so, nothing left here', panelButtons.join('|') === 'Delete|Keep' && /account/.test(said ?? '') && !!otherHere && (await onAccount('other')).length === 3,
    `(${panelButtons.join(', ')}; "${said}"; browser ${(await inBrowser('other')).length}, card ${await card('.save-slots', 'other')}, queued ${(await queued()).length})`);
  refuse = false;

  // the game being played, with a change on its way up
  await p.evaluate(() => window.tw.client.post({ kind: 'edit', seq: 990001, op: { op: 'fares', base: 2.25, perKm: 0.1 } }));
  await p.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await p.click('.save-slots li[data-origin="play"] .save-actions .save-delete');
  await p.click('.save-slots li[data-origin="play"] > .callout button:first-child');
  await p.waitForSelector('.start-menu', { timeout: 60000 });
  await settled();
  await p.waitForTimeout(1500);
  const first = { account: (await onAccount('play')).length, browser: (await inBrowser('play')).length, card: await card('.start-saves', 'play') };
  await menu();
  await settled();
  await p.waitForTimeout(1000);
  report('panel, game being played: every version gone, after the page leaves and after a reload', !first.account && !first.browser && !first.card && !(await onAccount('play')).length && !(await card('.start-saves', 'play')),
    `(at the menu ${JSON.stringify(first)}; after a reload account ${(await onAccount('play')).length}, card ${await card('.start-saves', 'play')})`);
  report('the refused deletes went at the next load', !(await onAccount('other')).length && !(await card('.start-saves', 'other')) && !(await queued()).length,
    `(account ${(await onAccount('other')).length}, queued ${(await queued()).length})`);
  report('no page errors', errors.length === 0, errors.slice(0, 4).join(' | '));
} catch (err) {
  report('ran to the end', false, String(err?.stack ?? err));
} finally {
  await d.close().catch(() => {});
  if (OWN_SERVER) { server?.kill(); await pause(500); rmSync(serverDir, { recursive: true, force: true }); }
}
process.exit(failed ? 1 : 0);
