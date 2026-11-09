// T-2614: a save that cannot reach the account. Against a local shonei-server (as t988-cloud.mjs), one
// browser profile signed in, a game tied to the account, the clock paused:
//   1. with the account's game routes cut, a change and a save by hand: the slot in this browser has
//      the change, the record says why it did not go up, and (here after `?keepMs=1000`, else two
//      minutes) the yellow dot beside the search glass says "Not on your account yet" with the reason;
//   2. the routes back: the next minute's retry sends it with nothing else happening; the dot goes and
//      the account has the change;
//   3. cut again, a change, the dot; back, and a click on the dot sends it at once.
//   node trainworld/checks/t2614-cloud-fail.mjs [app url] [--server]   (from the maps root; default localhost:6153)
// Without --server it builds ../shonei-server and runs it on 127.0.0.1:8083 in a temp folder. Prints
// PASS or FAIL lines and exits 1 on a failure. Screenshots to artifacts/trainworld/terrainnet/. About three minutes.
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const args = process.argv.slice(2);
const OWN_SERVER = !args.includes('--server');
// the account server answers pages on localhost 6100-6199 only (shonei-server's cors.go)
const APP = args.find((a) => !a.startsWith('--')) ?? 'http://localhost:6153/';
const SERVER = 'http://127.0.0.1:8083/';
const Q = 'debug=1&acct=local&keepMs=1000&paused=1&demandWorkers=1';
const SUB = 't2614-check-' + Date.now().toString(36);
const OUT = join(ARTIFACTS, 'terrainnet');
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
  const dir = mkdtempSync(join(tmpdir(), 'tw-t2614-srv-'));
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
const signin = await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub: SUB }) })).json();
const accountGames = async () => (await (await fetch(SERVER + 'trainworld/games', { headers: { Authorization: 'Bearer ' + signin.token } })).json()).games;

const errors = [];
const d = await launchChrome({ prefix: 'tw-t2614-', size: [1300, 900], args: ANGLE });
let cut = false;
await d.ctx.route(/127\.0\.0\.1:8083\/trainworld\/games/, async (route) => {
  try { if (cut) await route.abort('internetdisconnected'); else await route.continue(); } catch { /* the page went */ }
});
if (process.env.T2614_LOG) d.ctx.on('requestfinished', async (r) => { if (r.url().startsWith(SERVER)) console.log('   ', r.method(), r.url().slice(SERVER.length), (await r.response())?.status()); });
await d.ctx.addInitScript((s) => { if (!localStorage.getItem('noritetsu.auth.v1')) localStorage.setItem('noritetsu.auth.v1', s); }, JSON.stringify({ token: signin.token, account: signin.account }));
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource|ERR_INTERNET_DISCONNECTED/.test(m.text())) errors.push(m.text()); });
const go = (query) => p.goto(`${APP}?${Q}${query}&v=${Date.now()}`);
const inGame = async () => {
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await p.addStyleTag({ content: '#debug-root { display: none !important; }' });
};
const playing = () => p.evaluate(async () => {
  const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
  const slot = s.slots.find((x) => x.key === s.active);
  return slot ? { key: slot.key, at: slot.at, edit: slot.edit, cloud: slot.cloud ?? null } : null;
});
const dot = () => p.locator('.keep-dot');
const dotIn = (ms) => dot().waitFor({ state: 'visible', timeout: ms }).then(() => true, () => false);
const dotGone = (ms) => dot().waitFor({ state: 'detached', timeout: ms }).then(() => true, () => false);
/** a player change: a short blueprint segment in Manhattan */
let nth = 0;
const change = () => p.evaluate(async (k) => {
  const tw = window.tw, a = tw.lngLatToPack(-73.99 + k * 0.004, 40.75), b = tw.lngLatToPack(-73.987 + k * 0.004, 40.752);
  return (await tw.client.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 1 }, to: { kind: 'free', x: b.x, y: b.y, level: 1 }, pis: [], single: false })).ok;
}, nth++);
const saveByHand = () => p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
const TIP = 'Not on your account yet\nThe server could not be reached. This game is saved in this browser. Click to try again.';

try {
  // a slot of its own, tied to the account, opened in a page that writes it
  await go('&new=1');
  await inGame();
  await p.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Cloud check'));
  const key = await until(async () => { const s = await playing(); return s?.key.startsWith('slot-') ? s.key : null; }, 30000, 500);
  await p.evaluate(async ([key, acct]) => {
    const [sf, cl] = await Promise.all([import('/src/workers/saveFile.ts'), import('/src/workers/cloud.ts')]);
    await sf.writeSync(key, { id: cl.newGameId(), acct, rev: 0, syncedAt: 0 });
  }, [key, signin.account.id]);
  await go(`&slot=${key}`);
  await inGame();
  const up = await until(async () => { const s = await playing(); return s?.cloud?.okAt && !s.cloud.error ? s : null; }, 60000, 500);
  report('the game is on the account', !!up && up.key === key, JSON.stringify(up?.cloud ?? await playing()));
  if (!up) console.log(await p.evaluate(async () => { const a = await import('/src/game/account.ts'); return JSON.stringify({ state: a.accountState.value, cfg: a.accountConfig.value, ctx: a.cloudContext() }); }));
  if (!up) throw new Error('no game on the account to test with');
  const id = up?.cloud?.id;
  const revThere = async () => (await accountGames()).find((g) => g.id === id)?.rev ?? -1;
  const rev0 = await revThere();

  // 1. the account cut off: a change and a save by hand
  cut = true;
  report('a change', await change());
  await saveByHand();
  const behind = await until(async () => { const s = await playing(); return s?.cloud?.error ? s : null; }, 30000, 500);
  report('the record says why it did not go up', behind?.cloud?.error === 'The server could not be reached', JSON.stringify(behind?.cloud));
  report('the slot in this browser has the change', !!behind && (behind.edit ?? 0) > (up?.edit ?? 0) && behind.at > up.at, `edit ${up?.edit} -> ${behind?.edit}`);
  report('the dot comes', await dotIn(20000));
  const tip = await dot().getAttribute('data-tip').catch(() => null);
  report('its words', tip === TIP, JSON.stringify(tip));
  const box = await dot().boundingBox();
  if (box) {
    await p.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await p.waitForTimeout(900);
    await p.screenshot({ path: join(OUT, 't2614-dot-hover.png'), clip: { x: 250, y: 40, width: 650, height: 320 } });
    await p.mouse.move(5, 500);
  }
  report('the account still has the old copy', await revThere() === rev0, `rev ${rev0}`);

  // 2. back: the retry a minute on sends it, with the clock paused and nothing else happening
  cut = false;
  const t0 = Date.now();
  report('the dot goes once it is sent again', await dotGone(80000), `${((Date.now() - t0) / 1000).toFixed(0)} s`);
  const rev1 = await until(async () => { const r = await revThere(); return r > rev0 ? r : null; }, 10000);
  report('the account has the change', !!rev1, `rev ${rev0} -> ${rev1}`);
  const ok = await playing();
  report('the record is clear', !ok?.cloud?.error && ok?.cloud?.okAt > (up?.cloud?.okAt ?? 0), JSON.stringify(ok?.cloud));

  // 3. cut again; back; a click on the dot sends it at once
  cut = true;
  await change();
  await saveByHand();
  report('cut again: the dot comes', await dotIn(30000));
  cut = false;
  await dot().click();
  report('a click on the dot sends it', await dotGone(10000));
  const rev2 = await until(async () => { const r = await revThere(); return r > (rev1 ?? rev0) ? r : null; }, 10000);
  report('the account has the second change', !!rev2, `rev ${rev1} -> ${rev2}`);
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
  await p.screenshot({ path: join(OUT, 't2614-failure.png') }).catch(() => {});
} finally {
  await d.close?.().catch(() => {});
  server?.kill();
}
process.exit(failed ? 1 : 0);
