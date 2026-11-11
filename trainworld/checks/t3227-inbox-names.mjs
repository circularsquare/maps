// T-3227/T-3229: the inbox's Names tab and the waiting-reports counts, against a local shonei-server in a temp folder.
// Builds ../shonei-server, runs it on its own port with Community and feedback threads on, seeds three players (names,
// boards, shares), puts reports on two shares, then drives the real inbox page (localhost:8800/twfeedback/, pointed at
// the local server with ?server=). Needs `python serve.py` running. Also checks the server side: a hidden name reads
// "player" on the board and the card, and Unhide brings it back.
// Screenshots: artifacts/trainworld/t3227-{feedback-count,names,names-hidden}.png
//   node trainworld/checks/t3227-inbox-names.mjs
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { launchChrome, ARTIFACTS, MAPS } from './lib.mjs';

const PORT = Number(process.env.T3227_PORT ?? 8127);
const SERVER = `http://127.0.0.1:${PORT}/`;
const INBOX = process.env.T3227_INBOX ?? 'http://localhost:8800/twfeedback/';
const KEY = 'check-admin-key-t3227-0123456789abcdef';

let failed = 0;
const ok = (c, msg) => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) failed++; };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 300) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error(`a server already answers on ${SERVER}: set T3227_PORT`);
const dir = mkdtempSync(join(tmpdir(), 'tw-t3227-srv-'));
const src = join(MAPS, '..', 'shonei-server');
const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
const b = spawnSync(go, ['build', '-o', join(dir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
if (b.status !== 0) throw new Error(`go build failed: ${b.stderr ?? b.error}`);
for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dir, f));
let server = null;
const startServer = async () => {
  server = spawn(join(dir, 'shonei.exe'), [], {
    cwd: dir, stdio: 'ignore',
    env: { ...process.env, SHONEI_ADDR: `127.0.0.1:${PORT}`, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1', SHONEI_COMMUNITY: 'preview', SHONEI_COMMUNITY_TESTERS: ['alice', 'pissbomb', 'bob'].map((s) => 'dev:t3227-' + s).join(','),
      SHONEI_FEEDBACK_THREADS: '1', SHONEI_FEEDBACK_ADMIN_KEY: KEY, RESEND_API_KEY: '' },
  });
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok))) throw new Error('the local server did not start');
};
const stopServer = async () => {
  try { server?.kill(); } catch { /* gone */ }
  await until(() => fetch(SERVER + 'noritetsu/config').then(() => false, () => true), 10000);
  await pause(300);
};
process.on('exit', () => { try { server?.kill(); } catch { /* gone */ } });

function drawing() {
  const n = 10, pills = [];
  for (let i = 0; i < n; i++) pills.push(i * 256, 0, i * 256, 256, 0, i);
  return {
    v: 1, pitch: 256, stroke: 200,
    lines: [{ c: '#d23f31', d: [0, 0, 0, 1, 2304, 0] }, { c: '#1f6fd1', f: 1, d: [0, 0, 256, 2, 128, 256, 256, 512, 1, 2304, 512] }],
    names: [...Array(n)].map((_, i) => `Stop ${String.fromCharCode(65 + i)}`), small: Array(n).fill(0), pills, passages: [0, 0, 256, 0, 1],
    fontPx: 192, width: Array(n).fill(480), width2: Array(n).fill(0), levels: [112, 224], faint: 0,
    at: [{ node: [0], x: [0], y: [0], dir: [2], slide: [-1], lift: [10], wrap: [0] }, { node: [], x: [], y: [], dir: [], slide: [], lift: [], wrap: [] }],
  };
}
const json = { 'Content-Type': 'application/json' };
const PEOPLE = [['alice', 'Alice Chen', 'aliceharbour01', 'Harbour loop', 9e5], ['pissbomb', 'pissbomb', 'pissbombgame01', 'Big pile', 5e5], ['bob', 'Bob', 'bobcafeline001', 'Cafe Line', 7e5]];
let page = null, close = async () => {};
try {
  await startServer();
  const tok = {}, ids = {}, shares = {};
  const gz = gzipSync(JSON.stringify(drawing()));
  for (const [sub, name, game, title, points] of PEOPLE) {
    const r = await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: json, body: JSON.stringify({ provider: 'dev', sub: 't3227-' + sub }) })).json();
    tok[sub] = r.token; ids[sub] = r.account?.id;
    const auth = { Authorization: 'Bearer ' + r.token };
    const s = await fetch(SERVER + 'trainworld/score/' + game, { method: 'PUT', headers: { ...auth, ...json }, body: JSON.stringify({ mode: 'normal', day: 12, points, cities: { nyc: points } }) });
    if (!s.ok) throw new Error(`score ${game}: ${s.status} ${await s.text()}`);
    const o = await fetch(SERVER + 'trainworld/optin', { method: 'PUT', headers: { ...auth, ...json }, body: JSON.stringify({ on: true, name }) });
    if (!o.ok) throw new Error(`optin ${sub}: ${o.status} ${await o.text()}`);
    const form = new FormData();
    form.append('meta', JSON.stringify({ game, name, title, copy: false }));
    form.append('drawing', new Blob([gz]), 'drawing.json.gz');
    const u = await fetch(SERVER + 'trainworld/community', { method: 'POST', headers: auth, body: form });
    if (!u.ok) throw new Error(`share ${game}: ${u.status} ${await u.text()}`);
    shares[sub] = (await u.json()).id;
  }
  // Reports need 60 minutes of play from each reporter, so the store is edited between two runs of the server.
  await stopServer();
  const idx = join(dir, 'trainworld', 'community', 'index.json');
  const ix = JSON.parse(readFileSync(idx, 'utf8'));
  const at = Math.floor(Date.now() / 1000) - 600;
  for (const sh of ix.shares) {
    if (sh.id === shares.pissbomb) { sh.reports = [1, 2, 3].map((i) => ({ by: 'x' + i, reason: 'offensive', at: at + i })); sh.hidden = 'auto'; }
    if (sh.id === shares.bob) sh.reports = [1, 2].map((i) => ({ by: 'y' + i, reason: i === 1 ? 'fake-score' : 'offensive', at: at + i }));
  }
  writeFileSync(idx, JSON.stringify(ix));
  await startServer();

  const board = async () => (await (await fetch(SERVER + 'trainworld/board?board=points')).json()).rows.map((r) => r.name);
  ok((await board()).includes('pissbomb'), `before: the board shows pissbomb (${await board()})`);

  ({ ctx: page, close } = await launchChrome({ prefix: 'tw-t3227-', size: [1100, 760], args: [] }));
  const p = await page.newPage();
  await p.addInitScript((k) => { try { localStorage.setItem('twfeedback.adminKey', k); } catch {} }, KEY);
  await p.goto(`${INBOX}?server=${encodeURIComponent(SERVER)}`);
  await p.waitForSelector('#app:not([hidden])', { timeout: 20000 });
  await p.waitForFunction(() => document.title.startsWith('('), undefined, { timeout: 10000 });
  const title = await p.title();
  ok(title === '(2) trainworld feedback inbox', `page title carries the count: "${title}"`);
  const ping = await p.textContent('#ping');
  ok(ping === '2 reported shares waiting: 1 hidden, 1 still showing', `top of the feedback list: "${ping}"`);
  ok((await p.textContent('#tabCount')) === '2', 'count on the Community tab');
  await p.screenshot({ path: join(ARTIFACTS, 't3227-feedback-count.png') });

  await p.click('#tabNames');
  await p.waitForSelector('.nrow');
  const rows = async () => p.evaluate(() => [...document.querySelectorAll('.nrow')].map((r) => ({ name: r.querySelector('.name').textContent, off: r.classList.contains('off'), btn: r.querySelector('button').textContent })));
  ok(JSON.stringify((await rows()).map((r) => r.name)) === JSON.stringify(['Alice Chen', 'Bob', 'pissbomb']), `three names, in order (${(await rows()).map((r) => r.name)})`);
  await p.screenshot({ path: join(ARTIFACTS, 't3227-names.png') });
  await p.fill('#nameQ', 'piss');
  await p.waitForFunction(() => document.querySelectorAll('.nrow').length === 1, undefined, { timeout: 5000 });
  ok((await p.textContent('#nameCount')) === '1 name', `search: "${await p.textContent('#nameCount')}"`);
  await p.click('.nrow button');
  await p.waitForFunction(() => document.querySelector('.nrow.off'), undefined, { timeout: 5000 });
  ok((await rows())[0].btn === 'Unhide', 'the row now offers Unhide');
  ok(!(await board()).includes('pissbomb') && (await board()).includes('player'), `after Hide name: the board reads ${await board()}`);
  const cardId = shares.pissbomb;
  ok((await (await fetch(SERVER + 'trainworld/community/' + cardId)).status) === 404, 'the reported share stays hidden by reports (card 404)');
  await p.fill('#nameQ', '');
  await p.waitForFunction(() => document.querySelectorAll('.nrow').length === 3, undefined, { timeout: 5000 });
  ok((await p.textContent('#nameCount')) === '3 names, 1 hidden', `count: "${await p.textContent('#nameCount')}"`);
  await p.screenshot({ path: join(ARTIFACTS, 't3227-names-hidden.png') });
  await p.click('.nrow.off button');
  await p.waitForFunction(() => !document.querySelector('.nrow.off'), undefined, { timeout: 5000 });
  ok((await board()).includes('pissbomb'), `after Unhide: ${await board()}`);
  await p.click('#tabCommunity');
  await p.waitForSelector('.share');
  ok((await p.locator('.share').count()) === 2, 'Community tab lists both reported shares');
} finally {
  await close();
  try { server?.kill(); } catch { /* gone */ }
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
