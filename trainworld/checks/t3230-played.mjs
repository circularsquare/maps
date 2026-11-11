// T-3230: time played, against a mocked server (the server's part, `played` seconds on the account's games list and on
// the 403 "too-new" answer, is shonei-server's). A stored game counts the seconds it is open here (localStorage
// `trainworld.gameSeconds`) and its saved-games card says "N min played"; the signed-in account panel says the
// account's "3 h 20 min played" from the games list; the Share dialog's too-new error ends "You've played 25 min." from
// the answer, from the games list when the answer has none, and leaves the sentence off when neither says.
// Screenshots: artifacts/trainworld/t3230-{card,account,share}-{light,dark}.png
//   node trainworld/checks/t3230-played.mjs [url]          (needs a Vite: dev-nowatch.mjs <port>; TW_CORES=2; about 2 minutes)
import { readFileSync } from 'node:fs';
import { launchChrome, collectErrors, importSave, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6118/';
const SERVER = 'http://127.0.0.1:8083';
const CORS = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': '*' };
const json = (body, status = 200) => ({ status, headers: { ...CORS, 'content-type': 'application/json' }, body: JSON.stringify(body) });

let failed = 0;
const ok = (c, msg) => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) failed++; };

const errors = [];
let hidden = false, gamesOn = false, listPlayed = 12020, answerPlayed = 1500;
const { ctx, close } = await launchChrome({ prefix: 'tw-t3230-', size: [1400, 900] });
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|Failed to load resource|ERR_/);
  await page.addInitScript(() => {
    try {
      if (!localStorage.getItem('trainworld-ui')) localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 }));
      localStorage.setItem('trainworld.acct-api', 'local');
      localStorage.setItem('noritetsu.auth.v1', JSON.stringify({ token: 'tok', account: { id: 'acct1', username: 'anita', logins: ['password'], email: 'a@example.com', pendingEmail: '' } }));
    } catch {}
  });
  await page.route(`${SERVER}/**`, async (route) => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: CORS });
    if (path === '/noritetsu/config') return route.fulfill(json({ googleClientId: '', dev: false, password: true, signIn: true, trainworld: true, trainworldCommunity: true, trainworldGames: gamesOn }));
    if (path === '/noritetsu/refresh') return route.fulfill(json({ token: 'tok', account: { id: 'acct1', username: 'anita', logins: ['password'], email: 'a@example.com', pendingEmail: '' } }));
    if (path === '/trainworld/me') return route.fulfill(json({ board: { on: true, name: 'player', ...(hidden ? { nameHidden: true } : {}) } }));
    if (path === '/trainworld/games' && req.method() === 'GET') return route.fulfill(json({ games: [], tombstones: [], used: 0, limit: 20 << 20, maxGames: 20, ...(listPlayed === null ? {} : { played: listPlayed }) }));
    if (path === '/trainworld/community/list') return route.fulfill(json({ view: 'mine', seed: 'x', count: 0, rows: [], left: 3, perDay: 3 }));
    if (path === '/trainworld/community' && req.method() === 'POST') return route.fulfill(json({ error: 'play a little longer before sharing', code: 'too-new', ...(answerPlayed === null ? {} : { played: answerPlayed }) }, 403));
    return route.fulfill(json({}));
  });
  await page.goto(`${BASE}?debug=1&paused=1&acct=local&community&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForFunction(async () => (await import('/src/game/account.ts')).accountState.value === 'on', undefined, { timeout: 30000 });
  await importSave(page, readFileSync(REAL_SAVE));
  await page.waitForTimeout(6000);
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Played check'));
  await page.waitForFunction(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return s.writable && s.slots.find((x) => x.key === s.active)?.name === 'Played check'; }, undefined, { timeout: 120000, polling: 500 });
  await page.waitForFunction(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return s.active !== 'autosave' && s.slots.some((x) => x.key === s.active && x.name === 'Played check'); }, undefined, { timeout: 120000, polling: 500 });
  const activeKey = () => page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
  const stored = async () => page.evaluate((k) => { try { return JSON.parse(localStorage.getItem('trainworld.gameSeconds') || '{}')[k] ?? 0; } catch { return -1; } }, await activeKey());

  // the game counts while it is open and the tab is visible
  const before = await stored();
  await page.waitForTimeout(40000);
  const after = await stored();
  ok(after - before >= 15 && after - before <= 45, `40 s open counted ${after - before} s (flushed every 15 s)`);

  const shot = async (name) => {
    for (const shade of ['light', 'dark']) {
      await page.evaluate(async (s) => { (await import('/src/game/ui.ts')).shade.value = s; }, shade);
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${ARTIFACTS}t3230-${name}-${shade}.png` });
    }
  };

  // the card
  console.log('      active', await activeKey());
  await page.evaluate(async (k) => { localStorage.setItem('trainworld.gameSeconds', JSON.stringify({ [k]: 12020 })); }, await activeKey());
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector('.save-card');
  const card = await page.locator('.save-card', { hasText: 'Played check' }).first().innerText();
  console.log('      card:', JSON.stringify(card));
  ok(/Day \d+.* · 3 h 20 min played/.test(card), 'the card says "3 h 20 min played" after the day and cities');
  await shot('card');
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });

  // the account's figures need the games list, so the page opens again with games on
  gamesOn = true;
  await page.goto(`${BASE}?debug=1&paused=1&acct=local&community&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForFunction(async () => (await import('/src/game/account.ts')).accountState.value === 'on', undefined, { timeout: 30000 });

  // the account panel
  await page.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = true; });
  await page.waitForSelector('.acct-account');
  await page.waitForFunction(() => /played/.test(document.querySelector('.acct-account')?.textContent ?? ''), undefined, { timeout: 15000 }).catch(() => {});
  const acct = await page.locator('.acct-account').innerText();
  console.log('      account:', JSON.stringify(acct));
  ok(/3 h 20 min played/.test(acct), 'the account panel says "3 h 20 min played"');
  ok(!/hidden from others/.test(acct), 'no hidden-name line while the server does not say');
  await shot('account');
  hidden = true;
  await page.evaluate(async () => { const a = await import('/src/game/account.ts'); a.accountOpen.value = false; await new Promise((r) => setTimeout(r, 300)); a.accountOpen.value = true; });
  await page.waitForFunction(() => /hidden from others/.test(document.querySelector('.acct-account')?.textContent ?? ''), undefined, { timeout: 15000 }).catch(() => {});
  ok(/Your name is hidden from others\. Pick another\./.test(await page.locator('.acct-account').innerText()), 'a hidden name shows "Your name is hidden from others. Pick another."');
  await page.screenshot({ path: `${ARTIFACTS}t3230-namehidden.png` });
  hidden = false;
  await page.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = false; });

  // the Share dialog's error
  await page.evaluate(async () => { (await import('/src/game/community.ts')).shareAsk.value = true; });
  const send = page.locator('.cm-ask .cm-ask-foot .btn.text').first();
  await send.waitFor({ timeout: 30000 });
  const msg = async () => { await send.click(); await page.waitForFunction(() => document.querySelector('.cm-ask .acct-msg')?.textContent, undefined, { timeout: 30000 }); return page.locator('.cm-ask .acct-msg').first().innerText(); };
  let m = await msg();
  ok(m === "You can share after playing with an account for 60 minutes. You've played 25 min.", `answer's figure: ${m}`);
  await shot('share');
  answerPlayed = null;
  m = await msg();
  ok(m === "You can share after playing with an account for 60 minutes. You've played 3 h 20 min.", `list's figure when the answer has none: ${m}`);
  listPlayed = null;
  await page.evaluate(async () => { const g = await import('/src/game/accountGames.ts'); g.accountList.value = { ...g.accountList.value, played: undefined }; });
  m = await msg();
  ok(m === 'You can share after playing with an account for 60 minutes.', `no figure: ${m}`);
} finally {
  await close();
}
ok(errors.length === 0, `no console or page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
console.log(failed ? 'FAIL t3230-played' : 'PASS t3230-played');
process.exit(failed ? 1 : 0);
