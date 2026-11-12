// T-3326: screenshots on the feedback form. Against a local shonei-server with the feedback threads on
// (built from ../shonei-server, own temp store, admin key below), one browser profile:
//   - the Attach screenshots button and the counter show, 0/4;
//   - paste (a real ClipboardEvent), the file picker and a drop each add thumbnails; a fifth is refused with the
//     "Only 4 screenshots fit" line; text pasted into the message still pastes as text;
//   - a 6000 px picture is sent at 1920 px or less, under 1.5 MB, as WebP or JPEG;
//   - the remove button takes a thumbnail away; sending delivers the rest and the inbox lists them, each
//     fetchable; a file that is not an image gets the refusal line;
//   - the inbox page (tools/feedback-inbox/, served here on 6156) shows the three as small thumbnails that link to the full image;
//   - a server that does not report trainworldFeedbackImages (a stub config) shows no attach control.
//   node trainworld/checks/t3326-feedback-shots.mjs [app url]     (from the tree's root; default localhost:6155)
// Prints PASS or FAIL lines and exits 1 on a failure. Screenshots go to artifacts/trainworld/t3326-*.png.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, copyFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { launchChrome, ANGLE, ARTIFACTS, MAPS, NOISE } from './lib.mjs';

const APP = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'http://localhost:6155/';
const SERVER = 'http://127.0.0.1:8083/';
const ADMIN = 'check-admin-key-0123456789abcdef0123';
mkdirSync(ARTIFACTS, { recursive: true });
const shot = (name) => join(ARTIFACTS, `t3326-${name}.png`);

let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 300) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error('a server already answers on 127.0.0.1:8083: stop it first');
const dir = mkdtempSync(join(tmpdir(), 'tw-t3326-srv-'));
let src = MAPS;
for (let up = 0; up < 5 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_games.go')); up++) src = join(src, '..');
src = join(src, '..', 'shonei-server');
const goBin = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
const b = spawnSync(goBin, ['build', '-o', join(dir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
if (b.status !== 0) throw new Error(`go build failed: ${b.stderr ?? b.error}`);
for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dir, f));
const server = spawn(join(dir, 'shonei.exe'), [], { cwd: dir, stdio: 'ignore', env: { ...process.env, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1',
  SHONEI_FEEDBACK_THREADS: '1', SHONEI_FEEDBACK_ADMIN_KEY: ADMIN, SHONEI_FEEDBACK_DIR: join(dir, 'feedback') } });
if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok), 30000)) throw new Error('the local server did not start');
const admin = (path) => fetch(SERVER + 'trainworld/inbox/' + path, { headers: { Authorization: 'Bearer ' + ADMIN } });

const errors = [];
const d = await launchChrome({ prefix: 'tw-t3326-', size: [1100, 900], args: ANGLE });
const p = await d.ctx.newPage();
p.on('pageerror', (e) => errors.push(e.message));
p.on('console', (m) => { if (m.type() === 'error' && !NOISE.test(m.text()) && !/127\.0\.0\.1:8083|Failed to load resource/.test(m.text())) errors.push(m.text()); });

// a noisy picture drawn in the page, `w` x `h`, as PNG bytes and handed over as base64 in chunks (a 6000 px PNG is large)
async function pngBytes(w, h) {
  return Buffer.from(await p.evaluate(async ([w, h]) => {
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const x = c.getContext('2d');
    const g = x.createLinearGradient(0, 0, w, h); g.addColorStop(0, '#2a7'); g.addColorStop(1, '#a42');
    x.fillStyle = g; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 400; i++) { x.fillStyle = `hsl(${i * 37 % 360} 70% 50%)`; x.fillRect(Math.random() * w, Math.random() * h, 40 + Math.random() * 200, 20 + Math.random() * 100); }
    x.fillStyle = '#fff'; x.font = `${Math.round(h / 6)}px sans-serif`; x.fillText(`${w}x${h}`, w / 20, h / 2);
    const bytes = new Uint8Array(await (await new Promise((r) => c.toBlob(r, 'image/png'))).arrayBuffer());
    let s = ''; for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode(...bytes.subarray(i, i + 32768));
    return btoa(s);
  }, [w, h]), 'base64');
}

const thumbs = () => p.locator('.fb-thumbs li').count();
const counter = () => p.locator('.fb-shot-count').textContent().catch(() => null);
const msgText = () => p.locator('.fb-panel .acct-msg').first().textContent({ timeout: 1500 }).catch(() => '');
const inGame = () => p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 240000 });
const openForm = async () => {
  await p.evaluate(async () => { (await import('/src/game/ui.ts')).tab.value = 'settings'; (await import('/src/ui/FeedbackPanel.tsx')).feedbackOpen.value = true; });
  await p.locator('.fb-panel textarea[name=message]').waitFor({ timeout: 15000 });
};
const paste = (files, text = '') => p.evaluate(async ([files, text]) => {
  const dt = new DataTransfer();
  for (const f of files) dt.items.add(new File([Uint8Array.from(atob(f.b64), (c) => c.charCodeAt(0))], f.name, { type: f.type }));
  if (text) dt.setData('text/plain', text);
  const ev = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
  document.querySelector('.fb-panel textarea[name=message]').dispatchEvent(ev);
  return ev.defaultPrevented;
}, [files.map((f) => ({ name: f.name, type: f.type, b64: f.buf.toString('base64') })), text]);
const drop = (files) => p.evaluate((files) => {
  const dt = new DataTransfer();
  for (const f of files) dt.items.add(new File([Uint8Array.from(atob(f.b64), (c) => c.charCodeAt(0))], f.name, { type: f.type }));
  const form = document.querySelector('.fb-panel form');
  form.dispatchEvent(new DragEvent('dragover', { dataTransfer: dt, bubbles: true, cancelable: true }));
  form.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
}, files.map((f) => ({ name: f.name, type: f.type, b64: f.buf.toString('base64') })));

try {
  const small = await pngBytes(800, 500);
  const big = await pngBytes(6000, 3000);
  await p.goto(`${APP}?debug=1&acct=local&new=1&v=${Date.now()}`);
  await inGame();
  await until(() => p.evaluate(async () => (await import('/src/game/account.ts')).accountConfig.value?.trainworldFeedbackImages), 40000);
  await openForm();
  await p.addStyleTag({ content: '#debug-root { display: none !important; }' });
  report('attach button shown', await p.getByRole('button', { name: 'Attach screenshots' }).isVisible());
  report('counter reads 0/4', (await counter()) === '0/4', await counter());
  await p.screenshot({ path: shot('form-empty') });

  // text paste is left alone
  const prevented = await paste([], 'just words');
  report('pasting text is not intercepted', prevented === false);

  // paste one big picture: shrunk to 1920 and under 1.5 MB
  await paste([{ name: 'big.png', type: 'image/png', buf: big }]);
  await until(async () => (await thumbs()) === 1, 20000);
  report('paste adds a thumbnail', (await thumbs()) === 1);
  const enc = await p.evaluate(async () => {
    const src = document.querySelector('.fb-thumbs img').src;
    const blob = await (await fetch(src)).blob();
    const bm = await createImageBitmap(blob);
    return { type: blob.type, size: blob.size, w: bm.width, h: bm.height };
  });
  report('big picture shrunk to 1920 and under 1.5 MB', Math.max(enc.w, enc.h) === 1920 && enc.size <= 1.5 * 1024 * 1024 && /image\/(webp|jpeg)/.test(enc.type), JSON.stringify(enc));

  // picker: two files
  await p.locator('.fb-file').setInputFiles([
    { name: 'a.png', mimeType: 'image/png', buffer: small }, { name: 'b.png', mimeType: 'image/png', buffer: small }]);
  await until(async () => (await thumbs()) === 3, 15000);
  report('the picker adds two', (await thumbs()) === 3 && (await counter()) === '3/4', await counter());

  // drop one, then a fifth and sixth in one go
  await drop([{ name: 'd.png', type: 'image/png', buf: small }]);
  await until(async () => (await thumbs()) === 4, 15000);
  report('a drop adds one', (await thumbs()) === 4 && (await counter()) === '4/4');
  report('attach button disabled at 4', await p.getByRole('button', { name: 'Attach screenshots' }).isDisabled());
  await paste([{ name: 'e.png', type: 'image/png', buf: small }]);
  await pause(800);
  report('a fifth is refused with the line', (await thumbs()) === 4 && (await msgText()) === 'Only 4 screenshots fit. The rest were left out.', await msgText());
  await p.locator('.fb-panel textarea[name=message]').fill('the screenshots check');
  await p.screenshot({ path: shot('form-four') });

  // remove one
  await p.locator('.fb-thumbs li').nth(1).locator('.fb-thumb-x').click();
  report('remove takes one away', (await thumbs()) === 3 && (await counter()) === '3/4');

  // a file that is not a picture
  await p.locator('.fb-file').setInputFiles([{ name: 'x.txt', mimeType: 'text/plain', buffer: Buffer.from('hello') }]);
  await pause(500);
  report('a text file gets the refusal line', (await msgText()) === "That file isn't an image the game can send. Use PNG, JPEG or WebP.", await msgText());

  // send
  await p.locator('.fb-panel button[type=submit]').click();
  await p.locator('.fb-panel').waitFor({ state: 'hidden', timeout: 20000 }).catch(() => {});
  const list = await (await admin('threads')).json();
  const row = list.threads?.[0];
  report('a thread arrived', !!row, JSON.stringify(row ?? list).slice(0, 120));
  const th = row ? await (await admin('thread/' + row.id)).json() : {};
  report('the inbox lists 3 screenshots', (th.images ?? []).length === 3, JSON.stringify(th.images));
  let allOk = true;
  for (const im of th.images ?? []) {
    const r = await admin(`thread/${row.id}/image/${im.n}`);
    const bytes = Buffer.from(await r.arrayBuffer());
    allOk &&= r.ok && bytes.length === im.bytes && r.headers.get('content-type') === im.type;
  }
  report('each is fetchable at its size and type', allOk);
  report('the first (the 1920 px one) is under 1.5 MB', (th.images?.[0]?.bytes ?? 1e9) <= 1.5 * 1024 * 1024, String(th.images?.[0]?.bytes));
  report('the form is empty again', await openForm().then(async () => (await thumbs()) === 0 && (await counter()) === '0/4'));
  await p.keyboard.press('Escape');

  // the inbox page: thumbnails that open the full image
  const page = readFileSync(join(MAPS, 'trainworld/tools/feedback-inbox/index.html'));
  const inbox = createServer((q, r) => { r.setHeader('Content-Type', 'text/html'); r.end(page); }).listen(6156);
  try {
    const ip = await d.ctx.newPage();
    await ip.setViewportSize({ width: 1000, height: 800 });
    await ip.goto('http://localhost:6156/?server=http://127.0.0.1:8083/');
    await ip.evaluate((k) => localStorage.setItem('twfeedback.adminKey', k), ADMIN);
    await ip.goto(`http://localhost:6156/?server=http://127.0.0.1:8083/#${row.id}`);
    await ip.locator('#pics img').first().waitFor({ timeout: 15000 });
    await until(async () => (await ip.locator('#pics img').count()) === 3, 10000);
    const sizes = await ip.locator('#pics img').evaluateAll((a) => a.map((i) => [i.naturalWidth, i.width, i.height]));
    report('the inbox shows 3 thumbnails of about 96 px', sizes.length === 3 && sizes.every((z) => z[0] > 96 && z[1] >= 90 && z[1] <= 96 && z[2] === z[1]), JSON.stringify(sizes));
    report('each thumbnail links to the full image', await ip.locator('#pics a').evaluateAll((a) => a.every((x) => x.target === '_blank' && x.href.startsWith('blob:'))));
    await ip.screenshot({ path: shot('inbox') });
    await ip.close();
  } finally { inbox.close(); }

  // a server without the key: no control
  await p.route('**/noritetsu/config', async (route) => {
    const r = await route.fetch(); const j = await r.json(); delete j.trainworldFeedbackImages;
    await route.fulfill({ response: r, json: j });
  });
  await p.goto(`${APP}?debug=1&acct=local&new=1&v=${Date.now()}`);
  await inGame();
  await until(() => p.evaluate(async () => (await import('/src/game/account.ts')).accountConfig.value?.trainworldFeedbackThreads), 40000);
  await openForm();
  report('no attach control on a server that does not report it', (await p.locator('.fb-shots').count()) === 0);

  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} catch (e) {
  report('check ran', false, String(e?.stack ?? e));
  await p.screenshot({ path: shot('failure') }).catch(() => {});
} finally {
  await d.close?.().catch(() => {});
  server.kill();
}
process.exit(failed ? 1 : 0);
