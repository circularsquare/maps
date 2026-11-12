// T-3279: flags after names, against a local shonei-server (built from ../shonei-server on its own port, Community in
// preview, as t3224 does). Three players with names, scores and a share each; the page is signed in as the first, with
// the browser in Tokyo's time zone. The account panel shows an empty outline and then the display name; the outline
// opens the picker past the panel: Japan first, then every country, then other places (Tokyo among them, its country in
// grey), then the game's own few last. Typing filters; a pick saves at once and the outline becomes the flag. The
// leaderboard rows and Community cards show the flag after the name (smaller after the small grey name on a card). A
// server without `trainworldFlagSet` gets no prefecture or New York City in the picker.
// Screenshots: artifacts/trainworld/t3279/{account,picker-top,picker-places,picker-spain,picker-indonesia,picker-end,board,community}-{light,dark}.png
//   node trainworld/checks/t3279-flags.mjs [app url]     (from the maps root; needs a Vite: dev-nowatch.mjs <port>)
import { spawn, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { launchChrome, collectErrors, ARTIFACTS, MAPS } from './lib.mjs';

const APP = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'http://localhost:6152/';
const PORT = Number(process.env.T3279_PORT ?? 8127);
const SERVER = `http://127.0.0.1:${PORT}/`;
const PAGE_SERVER = 'http://127.0.0.1:8083/';
const OUT = join(ARTIFACTS, 't3279');
mkdirSync(OUT, { recursive: true });

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
if (await fetch(SERVER + 'noritetsu/config').then(() => true, () => false)) throw new Error(`a server already answers on ${SERVER}: set T3279_PORT`);
const dir = mkdtempSync(join(tmpdir(), 'tw-t3279-srv-'));
let src = MAPS;
for (let up = 0; up < 4 && !existsSync(join(src, '..', 'shonei-server', 'trainworld_flags.go')); up++) src = join(src, '..');
src = join(src, '..', 'shonei-server');
const go = existsSync('C:/Program Files/Go/bin/go.exe') ? 'C:/Program Files/Go/bin/go.exe' : 'go';
const b = spawnSync(go, ['build', '-o', join(dir, 'shonei.exe'), '.'], { cwd: src, encoding: 'utf8' });
if (b.status !== 0) throw new Error(`go build in ${src} failed: ${b.stderr ?? b.error}`);
for (const f of ['traders.json', 'traderstock.json']) if (existsSync(join(src, f))) copyFileSync(join(src, f), join(dir, f));
const PLAYERS = [ // sub, display name, flag, game, title, points in nyc
  ['t3279-anita', 'anita', '', 'anitagame0001', 'Tokyo', 9e5],
  ['t3279-bob', 'Bob Hoskins', 'gb', 'bobgame000001', 'Night trams', 6e5],
  ['t3279-zoe', 'Zoë', 'chequered', 'zoegame000001', '', 3e5],
];
const server = spawn(join(dir, 'shonei.exe'), [], {
  cwd: dir, stdio: 'ignore',
  env: { ...process.env, SHONEI_ADDR: `127.0.0.1:${PORT}`, SHONEI_SECRET: 'local-check', SHONEI_DEV: '1', SHONEI_COMMUNITY: 'preview', SHONEI_COMMUNITY_TESTERS: PLAYERS.map((p) => 'dev:' + p[0]).join(','), RESEND_API_KEY: '' },
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
const api = async (tok, method, path, body) => {
  const r = await fetch(SERVER + path, { method, headers: { Authorization: 'Bearer ' + tok, 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${await r.text()}`);
  return r.json();
};

let pageErrors = [];
try {
  if (!await until(() => fetch(SERVER + 'noritetsu/config').then((r) => r.ok))) throw new Error('the local server did not start');
  const sessions = [];
  const gz = gzipSync(JSON.stringify(drawing()));
  for (const [sub, name, flag, game, title, points] of PLAYERS) {
    const s = await (await fetch(SERVER + 'noritetsu/signin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider: 'dev', sub }) })).json();
    sessions.push(s);
    await api(s.token, 'PUT', 'trainworld/score/' + game, { mode: 'normal', day: 12, points, cities: { nyc: points } });
    await api(s.token, 'PUT', 'trainworld/optin', { on: true, name });
    if (flag) await api(s.token, 'PUT', 'trainworld/flag', { flag });
    const form = new FormData();
    form.append('meta', JSON.stringify({ game, name, title, copy: false }));
    form.append('drawing', new Blob([gz]), 'drawing.json.gz');
    const u = await fetch(SERVER + 'trainworld/community', { method: 'POST', headers: { Authorization: 'Bearer ' + s.token }, body: form });
    if (!u.ok) throw new Error(`share ${game}: ${u.status} ${await u.text()}`);
  }
  const cfg = await (await fetch(SERVER + 'noritetsu/config')).json();
  ok(cfg.trainworldFlags === true && cfg.trainworldFlagSet === 3, `the server takes flag list 3 (${JSON.stringify({ flags: cfg.trainworldFlags, set: cfg.trainworldFlagSet })})`);

  // ---- the page, signed in as the first player, in Tokyo
  const { ctx, close } = await launchChrome({ prefix: 'tw-t3279-', size: [1300, 900], deviceScaleFactor: 2, timezoneId: 'Asia/Tokyo', locale: 'en-US' });
  try {
    let old = false; // a server from before flag list 2: config without trainworldFlagSet
    await ctx.route(`${PAGE_SERVER}**`, async (route) => {
      const url = new URL(route.request().url().replace(PAGE_SERVER, SERVER));
      const res = await route.fetch({ url: url.href });
      if (url.pathname === '/noritetsu/config' && old) {
        const j = await res.json();
        delete j.trainworldFlagSet;
        return route.fulfill({ response: res, body: JSON.stringify(j) });
      }
      return route.fulfill({ response: res });
    });
    const page = await ctx.newPage();
    collectErrors(page, pageErrors, /favicon|openfreemap|AJAXError|Failed to load resource|ERR_/);
    await page.addInitScript((s) => {
      try {
        if (!localStorage.getItem('trainworld-ui')) localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 }));
        localStorage.setItem('noritetsu.auth.v1', JSON.stringify(s));
      } catch {}
    }, sessions[0]);
    const load = async () => {
      await page.goto(`${APP}?debug=1&paused=1&acct=local&community&v=${Date.now()}`);
      await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 180000 });
      await page.waitForFunction(async () => (await import('/src/game/account.ts')).accountState.value === 'on', undefined, { timeout: 30000 });
    };
    const shade = (s) => page.evaluate(async (sh) => { (await import('/src/game/ui.ts')).shade.value = sh; }, s);
    /** both shades, cropped to the union of `sel`'s boxes (the panel, the picker) with a margin */
    const shots = async (name, sel, between) => {
      for (const s of ['light', 'dark']) {
        await shade(s); await page.waitForTimeout(500);
        if (between) { await between(); await page.waitForTimeout(1200); } // the list's pictures load as they scroll into view
        const clip = await page.evaluate((q) => {
          const rs = [...document.querySelectorAll(q)].map((e) => e.getBoundingClientRect()).filter((r) => r.width);
          const x = Math.max(0, Math.min(...rs.map((r) => r.left)) - 16), y = Math.max(0, Math.min(...rs.map((r) => r.top)) - 16);
          return { x, y, width: Math.min(innerWidth, Math.max(...rs.map((r) => r.right)) + 16) - x, height: Math.min(innerHeight, Math.max(...rs.map((r) => r.bottom)) + 16) - y };
        }, sel);
        await page.screenshot({ path: join(OUT, `${name}-${s}.png`), clip });
      }
      await shade('light');
    };
    const openAccount = async () => {
      await page.evaluate(async () => { const a = await import('/src/game/account.ts'); a.accountOpen.value = false; await new Promise((r) => setTimeout(r, 200)); a.accountOpen.value = true; });
      await page.waitForSelector('.acct-flag-btn', { timeout: 20000 });
    };
    const pickerNames = () => page.evaluate(() => [...document.querySelectorAll('.flag-pop li')].map((li) => li.querySelector('.flag-pop-name').textContent + (li.classList.contains('gap') ? ' |' : '')));

    await load();
    await openAccount();
    const row = await page.evaluate(() => {
      const r = (e) => e.getBoundingClientRect(), btn = document.querySelector('.acct-flag-btn'), name = document.querySelector('.acct-flag-name');
      return { none: !!btn.querySelector('.flag-none'), name: name?.textContent, after: r(name).left > r(btn).right, mid: Math.round((r(name).top + r(name).bottom) / 2 - (r(btn).top + r(btn).bottom) / 2), label: !!document.querySelector('.acct-flag-row')?.textContent.includes('Flag') };
    });
    ok(row.none && row.name === 'anita' && row.after && Math.abs(row.mid) <= 2 && !row.label, `account: an empty outline, then "anita", no "Flag" label (${JSON.stringify(row)})`);
    await shots('account', '.acct-panel');

    // the picker
    await page.click('.acct-flag-btn');
    await page.waitForSelector('.flag-pop li');
    let names = await pickerNames();
    const at = (n) => names.findIndex((x) => x.replace(' |', '') === n);
    ok(names[0] === 'Japan', `first: the time zone's country (${names.slice(0, 3)})`);
    ok(at('Afghanistan') === 1 && names[1].endsWith('|') && at('Zimbabwe') > at('Japan', 1), 'then every country A to Z, after a gap');
    const places = names.slice(at('Zimbabwe') + 1, at('Train'));
    ok(names[at('Zimbabwe') + 1].endsWith('|') && places.length === 79 && places.some((x) => x.startsWith('Tokyo')) && places.some((x) => x.startsWith('New York City')) && places.some((x) => x.startsWith('European Union')), `then 79 other places, Tokyo and New York City among them (${places.slice(0, 6)})`);
    ok(JSON.stringify(names.slice(at('Train')).map((x) => x.replace(' |', ''))) === JSON.stringify(['Train', 'Earth', 'Pride', 'Transgender', 'United Nations', 'Pirate', 'Checkered']), `the game's own few last (${names.slice(at('Train'))})`);
    ok(names.filter((x) => x.startsWith('Japan')).length === 2 && !names.includes('No flag'), 'Japan at the top and in its place; no "No flag" while none is picked');
    const tokyoWhere = await page.evaluate(() => [...document.querySelectorAll('.flag-pop li')].find((li) => li.querySelector('.flag-pop-name').textContent === 'Tokyo')?.querySelector('.flag-pop-where')?.textContent);
    ok(tokyoWhere === 'Japan', `Tokyo's grey note: ${tokyoWhere}`);
    const geo = await page.evaluate(() => {
      const r = (s) => document.querySelector(s).getBoundingClientRect(), pop = r('.flag-pop'), panel = r('.acct-panel');
      // the search takes focus only with a mouse (a phone's keyboard would cover the list); headless Chrome has none
      const fine = matchMedia('(pointer: fine)').matches;
      return { pop: [Math.round(pop.top), Math.round(pop.bottom), Math.round(pop.height)], panel: [Math.round(panel.top), Math.round(panel.bottom)], inView: pop.bottom <= innerHeight && pop.right <= innerWidth && pop.left >= 0, focus: !fine || !!document.activeElement?.closest('.flag-pop') };
    });
    ok(geo.inView && geo.pop[2] >= 300 && geo.pop[1] > geo.panel[1] && geo.focus, `the picker is tall, inside the window and reaches past the panel (${JSON.stringify(geo)})`);
    await shots('picker-top', '.acct-panel, .flag-pop');
    const scrollTo = (text) => page.evaluate((t) => { const li = [...document.querySelectorAll('.flag-pop li')].find((l) => l.querySelector('.flag-pop-name').textContent === t); li.scrollIntoView({ block: 'start' }); li.parentElement.scrollTop -= 40; }, text);
    await shots('picker-places', '.acct-panel, .flag-pop', () => scrollTo('European Union'));
    await shots('picker-end', '.acct-panel, .flag-pop', () => page.evaluate(() => { const ul = document.querySelector('.flag-pop ul'); ul.scrollTop = ul.scrollHeight; }));
    const broken = await page.evaluate(async () => {
      const ul = document.querySelector('.flag-pop ul'), bad = [];
      for (let y = 0; y < ul.scrollHeight; y += ul.clientHeight) { ul.scrollTop = y; await new Promise((r) => setTimeout(r, 120)); }
      await new Promise((r) => setTimeout(r, 1500));
      for (const img of ul.querySelectorAll('img')) if (img.complete && img.naturalWidth === 0) bad.push(img.src);
      return { bad, n: ul.querySelectorAll('img').length };
    });
    ok(broken.bad.length === 0, `every picture in the list loads (${broken.n} pictures; broken ${broken.bad.slice(0, 5)})`);

    // typing filters
    await page.fill('.flag-pop input', 'tok');
    names = await pickerNames();
    ok(JSON.stringify(names) === JSON.stringify(['Tokelau', 'Tokushima |', 'Tokyo']), `"tok": ${names}`);
    await page.fill('.flag-pop input', 'japan');
    names = await pickerNames();
    ok(names[0] === 'Japan' && names.length === 48 && names[1].endsWith('|'), `"japan": Japan, then its 47 prefectures (${names.length})`);
    await page.fill('.flag-pop input', 'spain');
    names = await pickerNames();
    ok(names[0] === 'Spain' && names.length === 18 && ['Catalonia', 'Basque Country', 'Galicia', 'Valencia', 'Andalusia', 'Madrid', 'Canary Islands'].every((n) => names.some((x) => x.startsWith(n))), `"spain": Spain, then its 17 regions (${names.length})`);
    await shots('picker-spain', '.acct-panel, .flag-pop');
    await page.fill('.flag-pop input', 'indonesia');
    await shots('picker-indonesia', '.acct-panel, .flag-pop');
    await page.fill('.flag-pop input', 'spain');
    for (const [q, n] of [['portugal', 5], ['indonesia', 4], ['france', 4]]) {
      await page.fill('.flag-pop input', q);
      names = await pickerNames();
      ok(names.length === n, `"${q}": the country and ${n - 1} places (${names})`);
    }
    await page.fill('.flag-pop input', 'tokyo');
    await page.press('.flag-pop input', 'Enter');
    await until(() => page.evaluate(() => !!document.querySelector('.acct-flag-btn img.flag')));
    const set = await api(sessions[0].token, 'GET', 'trainworld/me');
    ok(set.board.flag === 'jp-13' && (await page.locator('.flag-pop').count()) === 0, `Enter picks Tokyo, saved at once (${set.board.flag}), the picker closes`);
    await page.click('.acct-flag-btn');
    names = await pickerNames();
    ok(names[0] === 'No flag' && names[1] === 'Japan' && names[2] === 'Afghanistan |', `with a flag, "No flag" comes first (${names.slice(0, 3)})`);
    await page.keyboard.press('Escape');
    ok((await page.locator('.flag-pop').count()) === 0 && await page.evaluate(async () => (await import('/src/game/account.ts')).accountOpen.value), 'Escape closes the picker and keeps the panel');
    await shots('account-set', '.acct-panel');
    await page.evaluate(async () => { (await import('/src/game/account.ts')).accountOpen.value = false; });

    // the leaderboard
    await page.evaluate(async () => { (await import('/src/game/account.ts')).boardsOpen.value = true; });
    await page.waitForSelector('.lb-rows li .named', { timeout: 30000 });
    await page.waitForFunction(() => document.querySelectorAll('.lb-rows li .flag').length >= 3, undefined, { timeout: 15000 }).catch(() => {});
    const lb = await page.evaluate(() => [...document.querySelectorAll('.lb-rows li')].map((li) => {
      const t = li.querySelector('.named-text'), f = li.querySelector('.flag'), r = (e) => e.getBoundingClientRect();
      return { name: t?.textContent, flag: f?.getAttribute('alt') ?? '', after: f ? r(f).left > r(t).right : null, h: f ? Math.round(parseFloat(getComputedStyle(f).height)) : 0, mid: f ? Math.round((r(f).top + r(f).bottom) / 2 - (r(t).top + r(t).bottom) / 2) : 0 };
    }));
    ok(lb.length === 3 && lb.every((r) => r.after && r.h === 12 && Math.abs(r.mid) <= 1), `board rows: the flag after the name, 12 px high, centred on it (${JSON.stringify(lb)})`);
    ok(JSON.stringify(lb.map((r) => r.flag)) === JSON.stringify(['Tokyo', 'United Kingdom', 'Checkered']), `board flags: ${lb.map((r) => r.flag)}`);
    await shots('board', '.lb-panel');
    await page.evaluate(async () => { (await import('/src/game/account.ts')).boardsOpen.value = false; });

    // Community
    await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = true; });
    await page.waitForSelector('.cm-card', { timeout: 30000 });
    await page.waitForFunction(() => document.querySelectorAll('.cm-card .flag').length >= 3, undefined, { timeout: 15000 }).catch(() => {});
    const cards = await page.evaluate(() => [...document.querySelectorAll('.cm-card')].map((c) => {
      const t = c.querySelector('.named-text'), f = c.querySelector('.flag'), r = (e) => e.getBoundingClientRect();
      return { name: t?.textContent, small: !!f?.classList.contains('small'), grey: !!t?.closest('.cm-by'), after: f ? r(f).left > r(t).right : null, h: f ? Math.round(parseFloat(getComputedStyle(f).height)) : 0 };
    }));
    ok(cards.length === 3 && cards.every((c) => c.after && (c.grey ? c.small && c.h === 9 : !c.small && c.h === 12)), `cards: the flag after the name, 9 px after a small grey name, 12 px after a bold one (${JSON.stringify(cards)})`);
    await shots('community', '.community-panel .cm-cards');
    await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = false; });

    // a phone: the picker fits the window
    await page.setViewportSize({ width: 390, height: 844 });
    await openAccount();
    await page.click('.acct-flag-btn');
    await page.waitForSelector('.flag-pop li');
    const phone = await page.evaluate(() => { const r = document.querySelector('.flag-pop').getBoundingClientRect(); return { left: Math.round(r.left), right: Math.round(r.right), top: Math.round(r.top), bottom: Math.round(r.bottom), w: innerWidth, h: innerHeight }; });
    ok(phone.left >= 0 && phone.right <= phone.w && phone.top >= 0 && phone.bottom <= phone.h && phone.bottom - phone.top >= 300, `phone: the picker fits the window (${JSON.stringify(phone)})`);
    await shots('phone', '.acct-panel, .flag-pop');
    await page.keyboard.press('Escape');
    await page.setViewportSize({ width: 1300, height: 900 });

    // a server from before list 2: no prefectures or New York City offered
    old = true;
    await load();
    await openAccount();
    await page.click('.acct-flag-btn');
    await page.waitForSelector('.flag-pop li');
    names = await pickerNames();
    const oldPlaces = names.slice(at('Zimbabwe') + 1, names.findIndex((x) => x.startsWith('Train')));
    ok(!names.some((x) => /^(Tokyo|Osaka|New York City|Hokkaido|Catalonia|Lisbon|Jakarta|Brittany)/.test(x)) && names.some((x) => x.startsWith('Amsterdam')), `an older server: no prefectures or New York City (${oldPlaces.length} other places)`);
    ok(await page.evaluate(() => !!document.querySelector('.acct-flag-btn img.flag')), 'the flag it already has still shows');
  } finally {
    await close();
  }
} finally {
  stop();
  await pause(500);
  try { rmSync(dir, { recursive: true, force: true }); } catch { /* the exe may linger a moment */ }
}
ok(pageErrors.length === 0, `no console or page errors${pageErrors.length ? ': ' + pageErrors.slice(0, 3).join(' | ') : ''}`);
console.log(failed ? 'FAIL t3279-flags' : 'PASS t3279-flags');
process.exit(failed ? 1 : 0);
