// T-3224: Community's search, against a local shonei-server. The check builds ../shonei-server and runs it on its own
// port in a temp folder (Community in preview, the three sharers as testers), shares five networks through the real
// routes, and points the page's `?acct=local` server (127.0.0.1:8083) at it by forwarding every request. Signed out, it
// searches a title (accents folded), a sharer's display name (all of their shares, not one), something nothing matches,
// and clears the box back to the normal page; a search goes up once typing pauses. A server without
// `trainworldCommunitySearch` hides the box, and one that ignores `q` still shows the list.
// Screenshots: artifacts/trainworld/t3224-{desktop,phone}-{light,dark}.png (a title search) and t3224-nomatch-*.png.
//   node trainworld/checks/t3224-community-search.mjs [app url]     (from the maps root; needs a Vite: dev-nowatch.mjs <port>)
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { launchChrome, collectErrors, ARTIFACTS, MAPS } from './lib.mjs';

const APP = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'http://localhost:6117/';
const PORT = Number(process.env.T3224_PORT ?? 8124);
const SERVER = `http://127.0.0.1:${PORT}/`;
const PAGE_SERVER = 'http://127.0.0.1:8083/';

let failed = 0;
const ok = (c, msg) => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) failed++; };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f, ms = 30000, step = 300) {
  for (const end = Date.now() + ms; ; await pause(step)) {
    let v; try { v = await f(); } catch { v = null; }
    if (v || Date.now() > end) return v;
  }
}

// ---- the server
if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error(`a server already answers on ${SERVER}: set T3224_PORT`);
const dir = mkdtempSync(join(tmpdir(), 'tw-t3224-srv-'));
let src = MAPS;
for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_community.go')); up++) src = join(src, '..');
src = join(src, '..', 'shonei-server');
const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
const b = spawnSync(go, ['build', '-o', join(dir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
if (b.status !== 0) throw new Error(`go build in ${src} failed: ${b.stderr ?? b.error}`);
for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dir, f));
const SUBS = ['t3224-alice', 't3224-bob', 't3224-zoe'];
const server = spawn(join(dir, 'shonei.exe'), [], {
  cwd: dir, stdio: 'ignore',
  env: { ...process.env, SHONEI_ADDR: `127.0.0.1:${PORT}`, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1', SHONEI_COMMUNITY: 'preview', SHONEI_COMMUNITY_TESTERS: SUBS.map((s) => 'dev:' + s).join(','), RESEND_API_KEY: '' },
});
const stop = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stop);

/** a small valid drawing: two lines and ten stations (shonei-server's ctDrawing) */
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
const SHARES = [ // sub, display name, game, title, points in nyc
  [0, 'Alice Chen', 'aliceharbour01', 'Harbour loop', 9e5],
  [0, 'Alice Chen', 'alicesecond002', 'Second try', 3e5],
  [1, 'Bob', 'bobcafeline001', 'Café Line', 5e5],
  [1, 'Bob', 'bobbigcafe0002', 'Big Cafe', 1e5],
  [2, 'Zoë', 'zoenighttram01', 'Night trams', 7e5],
];
let pageErrors = [];
try {
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok))) throw new Error('the local server did not start');
  const tokens = [];
  for (const sub of SUBS) {
    const r = await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub }) });
    tokens.push((await r.json()).token);
  }
  const gz = gzipSync(JSON.stringify(drawing()));
  for (const [who, name, game, title, points] of SHARES) {
    const auth = { Authorization: 'Bearer ' + tokens[who] };
    const s = await fetch(SERVER + 'trainworld/score/' + game, { method: 'PUT', headers: { ...auth, 'Content-Type': 'application/json' }, body: JSON.stringify({ mode: 'normal', day: 12, points, cities: { nyc: points } }) });
    if (!s.ok) throw new Error(`score ${game}: ${s.status} ${await s.text()}`);
    const form = new FormData();
    form.append('meta', JSON.stringify({ game, name, title, copy: false }));
    form.append('drawing', new Blob([gz]), 'drawing.json.gz');
    const u = await fetch(SERVER + 'trainworld/community', { method: 'POST', headers: auth, body: form });
    if (!u.ok) throw new Error(`share ${game}: ${u.status} ${await u.text()}`);
  }
  const cfg = await (await fetch(SERVER + 'noritetsu/config')).json();
  ok(cfg.trainworldCommunitySearch === true && cfg.trainworldCommunityPreview === true, `the server says it searches (${JSON.stringify({ search: cfg.trainworldCommunitySearch, preview: cfg.trainworldCommunityPreview })})`);

  // ---- the page, signed out
  const { ctx, close } = await launchChrome({ prefix: 'tw-t3224-', size: [1300, 900] });
  try {
    const lists = [];
    let mode = 'real'; // 'real', 'old' (config without the flag), 'ignore' (the list never sees q)
    await ctx.route(`${PAGE_SERVER}**`, async (route) => {
      const req = route.request(), url = new URL(req.url().replace(PAGE_SERVER, SERVER));
      if (url.pathname === '/trainworld/community/list') {
        lists.push(url.searchParams.get('q'));
        if (mode === 'ignore') url.searchParams.delete('q');
      }
      const res = await route.fetch({ url: url.href });
      if (url.pathname === '/noritetsu/config' && mode !== 'real') {
        const j = await res.json();
        delete j.trainworldCommunitySearch;
        if (mode === 'ignore') j.trainworldCommunitySearch = true;
        return route.fulfill({ response: res, body: JSON.stringify(j) });
      }
      return route.fulfill({ response: res });
    });
    const page = await ctx.newPage();
    collectErrors(page, pageErrors, /favicon|openfreemap|AJAXError|Failed to load resource|ERR_/);
    await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
    const open = async () => {
      await page.goto(`${APP}?debug=1&paused=1&acct=local&community&v=${Date.now()}`);
      await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
      await page.waitForFunction(async () => (await import('/src/game/account.ts')).accountState.value === 'on', undefined, { timeout: 30000 });
      await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = true; });
      await page.waitForSelector('.cm-card', { timeout: 30000 });
    };
    const titles = () => page.evaluate(() => [...document.querySelectorAll('.cm-card .cm-card-name strong')].map((e) => e.textContent).sort());
    const names = () => page.evaluate(() => [...document.querySelectorAll('.cm-card .cm-by')].map((e) => e.textContent));
    /** types into the box as a player would, then waits for the list to settle */
    const search = async (text) => {
      const box = page.locator('.cm-search');
      await box.fill('');
      if (text) await box.pressSequentially(text, { delay: 60 });
      await page.waitForTimeout(900);
      // the list's own Loading line (cards say it too, in a span, while their drawing comes)
      await page.waitForFunction(() => ![...document.querySelectorAll('.community-panel p')].some((p) => p.textContent === 'Loading…'), undefined, { timeout: 15000 }).catch(() => {});
    };
    await open();
    ok((await page.locator('.cm-search').count()) === 1, 'the search box shows');
    const lay = await page.evaluate(() => {
      const r = (s) => document.querySelector(s).getBoundingClientRect(), bar = r('.cm-bar'), box = r('.cm-search'), tab = r('.cm-views .btn');
      return { right: Math.round(bar.right - box.right), mid: Math.round((box.top + box.bottom) / 2 - (tab.top + tab.bottom) / 2), ph: document.querySelector('.cm-search').placeholder };
    });
    ok(lay.right <= 1 && Math.abs(lay.mid) <= 1, `at the bar's right end, on the tabs' row (right gap ${lay.right}px, centre off ${lay.mid}px)`);
    ok(lay.ph === 'Search', `placeholder "${lay.ph}"`);
    ok((await titles()).length === 3, `no search: one share per person (${await titles()})`);

    let before = lists.length;
    await search('cafe');
    const sent = lists.slice(before).filter((q) => q !== null);
    ok(sent.length === 1 && sent[0] === 'cafe', `a search goes up once typing pauses (${JSON.stringify(lists.slice(before))})`);
    ok(JSON.stringify(await titles()) === JSON.stringify(['Big Cafe', 'Café Line']), `title search, accents folded: ${await titles()}`);
    for (const [shade, size, label] of [['light', [1300, 900], 'desktop'], ['dark', [1300, 900], 'desktop'], ['light', [390, 844], 'phone'], ['dark', [390, 844], 'phone']]) {
      await page.setViewportSize({ width: size[0], height: size[1] });
      await page.evaluate(async (sh) => { (await import('/src/game/ui.ts')).shade.value = sh; }, shade);
      await page.waitForTimeout(600);
      await page.screenshot({ path: `${ARTIFACTS}t3224-${label}-${shade}.png` });
      if (label === 'phone' && shade === 'light') {
        const ph = await page.evaluate(() => {
          const r = (s) => document.querySelector(s).getBoundingClientRect(), box = r('.cm-search'), where = r('.cm-where'), bar = r('.cm-bar');
          return { sameRow: Math.abs(box.top - where.top) < 4, inside: box.right <= bar.right + 1 && box.left >= bar.left - 1, w: Math.round(box.width) };
        });
        ok(ph.sameRow && ph.inside && ph.w >= 90, `phone: beside the place picker, inside the bar (${JSON.stringify(ph)})`);
      }
    }
    await page.setViewportSize({ width: 1300, height: 900 });
    await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });

    await search('alice');
    ok(JSON.stringify(await titles()) === JSON.stringify(['Harbour loop', 'Second try']) && (await names()).every((n) => n === 'Alice Chen'), `name search: both of her shares (${await titles()})`);
    await page.getByRole('tab', { name: 'Biggest' }).click();
    await page.waitForTimeout(800);
    const ranks = await page.evaluate(() => [...document.querySelectorAll('.cm-card')].map((c) => [c.querySelector('strong')?.textContent, c.querySelector('.lb-rank')?.textContent ?? null]));
    ok(JSON.stringify(ranks) === JSON.stringify([['Harbour loop', '1'], ['Second try', null]]), `Biggest keeps the search, points order, board ranks (${JSON.stringify(ranks)})`);
    await page.getByRole('tab', { name: 'Random' }).click();
    await page.waitForTimeout(800);

    await search('nothing like it');
    const empty = await page.locator('.community-panel p.muted').first().textContent().catch(() => '');
    ok((await page.locator('.cm-card').count()) === 0 && empty === 'No networks match that search', `no match says so ("${empty}")`);
    for (const [shade, size, label] of [['light', [1300, 900], 'desktop'], ['dark', [390, 844], 'phone']]) {
      await page.setViewportSize({ width: size[0], height: size[1] });
      await page.evaluate(async (sh) => { (await import('/src/game/ui.ts')).shade.value = sh; }, shade);
      await page.waitForTimeout(500);
      await page.screenshot({ path: `${ARTIFACTS}t3224-nomatch-${label}-${shade}.png` });
    }
    await page.setViewportSize({ width: 1300, height: 900 });
    await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });

    before = lists.length;
    await search('');
    ok((await titles()).length === 3 && lists.slice(before).at(-1) === null, `cleared: the normal page again (${await titles()})`);
    await search('zoe');
    await page.locator('.cm-search').press('Escape');
    await page.waitForTimeout(800);
    ok((await page.locator('.cm-search').inputValue()) === '' && (await titles()).length === 3 && await page.evaluate(async () => (await import('/src/game/community.ts')).communityOpen.value), 'Escape clears the box and keeps Community open');

    // a server without search: no box, the list as before
    mode = 'old';
    await open();
    ok((await page.locator('.cm-search').count()) === 0 && (await titles()).length === 3, 'a server without trainworldCommunitySearch: no box, cards as before');
    // a server that ignores q: the unfiltered list, no error
    mode = 'ignore';
    await open();
    await search('alice');
    ok((await titles()).length === 3 && (await page.locator('.community-panel .acct-msg.bad').count()) === 0, `a server that ignores q: the unfiltered list (${await titles()})`);
  } finally {
    await close();
  }
} finally {
  stop();
  await pause(500);
  try { rmSync(dir, { recursive: true, force: true }); } catch { /* the exe may linger a moment */ }
}
ok(pageErrors.length === 0, `no console or page errors${pageErrors.length ? ': ' + pageErrors.slice(0, 3).join(' | ') : ''}`);
console.log(failed ? 'FAIL t3224-community-search' : 'PASS t3224-community-search');
process.exit(failed ? 1 : 0);
