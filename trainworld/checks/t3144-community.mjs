// T-3144: Community round 3, the page's part, against a mocked server (the server's part is shonei-server's Go tests).
// Cards read "<title bold> <name grey>" (a share from before titles: the name alone), the rank has no blank before it,
// the Sandbox tag sits at the row's left edge, the Share dialog's title starts as the game's name and goes up as typed,
// and a refused title shows its error.
//   node trainworld/checks/t3144-community.mjs [url]          (needs a Vite: dev-nowatch.mjs <port>; TW_CORES=2)
import { readFileSync } from 'node:fs';
import { launchChrome, collectErrors, importSave, REAL_SAVE, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6195/';
const SERVER = 'http://127.0.0.1:8083';
const CORS = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': '*' };
const json = (body, status = 200) => ({ status, headers: { ...CORS, 'content-type': 'application/json' }, body: JSON.stringify(body) });

const at = Math.floor(Date.now() / 1000);
const ROWS = [
  { id: 'aaaaaaa1', rev: 1, name: 'anita', title: 'Amsterdam', city: '', points: 0, sandbox: true, lines: 6, stations: 41, day: 1, likes: 1, at, rank: 1 },
  { id: 'bbbbbbb2', rev: 1, name: 'bob', city: 'nyc', points: 5e6, lines: 20, stations: 300, day: 30, likes: 1, at, rank: 2 },
  { id: 'ccccccc3', rev: 1, name: 'a rather long display name', title: 'A title that is also rather long for a card', city: 'nyc', points: 4e6, lines: 5, stations: 90, day: 9, likes: 3, at, rank: 3 },
];

let failed = 0;
const ok = (c, msg) => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) failed++; };

const errors = [], listUrls = [], posts = [];
let refuseTitle = false;
const { ctx, close } = await launchChrome({ prefix: 'tw-t3144-', size: [1400, 900] });
try {
  const page = await ctx.newPage();
  collectErrors(page, errors, /favicon|openfreemap|AJAXError|Failed to load resource|ERR_/);
  await page.addInitScript(() => {
    try {
      localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 }));
      localStorage.setItem('trainworld.acct-api', 'local');
      localStorage.setItem('noritetsu.auth.v1', JSON.stringify({ token: 'tok', account: { id: 'acct1', logins: ['pw:anita'], email: '', pendingEmail: '' } }));
    } catch {}
  });
  await page.route(`${SERVER}/**`, async (route) => {
    const req = route.request(), url = new URL(req.url()), path = url.pathname;
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: CORS });
    if (path === '/noritetsu/config') return route.fulfill(json({ googleClientId: '', dev: false, password: true, signIn: true, trainworld: true, trainworldCommunity: true }));
    if (path === '/noritetsu/refresh') return route.fulfill(json({ token: 'tok', account: { id: 'acct1', logins: ['pw:anita'], email: '', pendingEmail: '' } }));
    if (path === '/trainworld/me') return route.fulfill(json({ board: { on: true, name: 'anita' } }));
    if (path === '/trainworld/community/list') {
      const view = url.searchParams.get('view');
      listUrls.push(url.search);
      return route.fulfill(json({ view, city: url.searchParams.get('city'), seed: 'x', count: ROWS.length, rows: view === 'mine' ? [] : ROWS, left: 3, perDay: 3 }));
    }
    if (path === '/trainworld/community' && req.method() === 'POST') {
      const body = req.postData() ?? '';
      const m = /name="meta"[\s\S]*?\r\n\r\n([\s\S]*?)\r\n--/.exec(body);
      posts.push(m ? JSON.parse(m[1]) : null);
      return route.fulfill(refuseTitle ? json({ error: 'please pick a different title', code: 'bad-title' }, 400) : json({ id: 'ddddddd4', rev: 1 }));
    }
    if (/^\/trainworld\/community\/\w+\/\d+$/.test(path)) return route.fulfill(json({ error: 'none' }, 404));
    return route.fulfill(json({}));
  });
  await page.goto(`${BASE}?debug=1&paused=1&acct=local&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForFunction(async () => (await import('/src/game/account.ts')).accountState.value === 'on', undefined, { timeout: 30000 });

  await page.waitForFunction(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.writable, undefined, { timeout: 60000 });
  // the list
  await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = true; });
  try { await page.waitForSelector('.cm-card', { timeout: 20000 }); }
  catch (e) {
    console.log(await page.evaluate(async () => { const a = await import('/src/game/account.ts'), c = await import('/src/game/community.ts'); return JSON.stringify({ state: a.accountState.value, cfg: a.accountConfig.value, session: !!a.session.value, on: c.communityOn(), open: c.communityOpen.value, modal: !!document.querySelector('.community-panel') }); }));
    console.log(errors.join(' | '));
    await page.screenshot({ path: `${ARTIFACTS}t3144-fail.png` });
    throw e;
  }
  await page.getByRole('tab', { name: 'Biggest' }).click();
  await page.waitForTimeout(400);
  await page.waitForSelector('.cm-card');
  await page.waitForTimeout(300);
  ok(listUrls.some((u) => /view=top/.test(u) && /city=(&|$)/.test(u)), `Biggest in the world asks with an empty place (${listUrls.at(-1)})`);
  const cards = await page.evaluate(() => [...document.querySelectorAll('.cm-card')].map((c) => {
    const q = (s) => c.querySelector(s), cs = (e) => e && getComputedStyle(e);
    const rank = q('.lb-rank'), strong = q('.cm-card-name strong'), by = q('.cm-by'), tag = q('.save-sandbox'), facts = q('.cm-facts');
    return {
      strong: strong?.textContent, strongWeight: +cs(strong).fontWeight, strongColor: cs(strong).color,
      by: by?.textContent ?? null, byWeight: by ? +cs(by).fontWeight : null, byColor: by ? cs(by).color : null,
      rankW: rank?.getBoundingClientRect().width, rankLeft: rank?.getBoundingClientRect().left, factsLeft: facts.getBoundingClientRect().left,
      tagLeft: tag?.getBoundingClientRect().left ?? null, tagMargin: tag ? cs(tag).marginLeft : null,
      nameBox: q('.cm-card-name').getBoundingClientRect().width, overflow: q('.cm-card-name').scrollWidth > q('.cm-card-name').clientWidth + 1,
    };
  }));
  ok(cards[0].strong === 'Amsterdam' && cards[0].by === 'anita', 'a titled card reads its title, then the name');
  ok(cards[0].strongWeight >= 600 && cards[0].byWeight === 400, `title bold (${cards[0].strongWeight}), name not (${cards[0].byWeight})`);
  ok(cards[0].byColor !== cards[0].strongColor, `name is grey (${cards[0].byColor} against ${cards[0].strongColor})`);
  ok(cards[1].strong === 'bob' && cards[1].by === null, 'a share with no title reads the name alone');
  ok(cards.every((c) => c.rankW <= 14 && Math.abs(c.rankLeft - c.factsLeft) <= 1), `the rank starts at the card's text edge (widths ${cards.map((c) => c.rankW).join(', ')})`);
  ok(cards[0].tagMargin === '0px' && Math.abs(cards[0].tagLeft - cards[0].factsLeft) <= 1, `the Sandbox tag sits at the row's left edge (margin ${cards[0].tagMargin}, ${cards[0].tagLeft} against ${cards[0].factsLeft})`);
  ok(cards[2].overflow === false, 'a long title and name stay inside the card');
  ok((await page.locator('.cm-note-foot').textContent()) === "Sandbox games aren't counted", 'Biggest ends with the sandbox note');
  await page.getByRole('tab', { name: 'Random' }).click();
  await page.waitForSelector('.cm-card');
  await page.waitForTimeout(300);
  ok((await page.locator('.cm-note-foot').count()) === 0, 'the note is on Biggest only');
  await page.getByRole('tab', { name: 'Biggest' }).click();
  await page.waitForSelector('.cm-card');
  await page.screenshot({ path: `${ARTIFACTS}t3144-cards.png` });

  // the opened share
  await page.locator('.cm-open').first().click();
  await page.waitForSelector('.cm-detail');
  const det = await page.evaluate(() => {
    const who = document.querySelector('.cm-detail-who'), tag = who.querySelector('.save-sandbox');
    return { text: who.querySelector('.cm-heading')?.textContent, tagLeft: tag?.getBoundingClientRect().left, whoLeft: who.getBoundingClientRect().left };
  });
  ok(det.text === 'Amsterdamanita', `an opened share is headed the same way (${det.text})`);
  ok(Math.abs(det.tagLeft - det.whoLeft) <= 1, 'the tag is at the left edge when opened too');
  await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = false; });

  // the Share dialog, with the real save loaded
  const was = await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).saveSlotsState.value.active);
  await importSave(page, readFileSync(REAL_SAVE));
  const ready = () => page.waitForFunction(async () => { const c = await import('/src/workers/clockClient.ts'); return !!window.tw?.world?.value && c.saveSlotsState.value.writable; }, undefined, { timeout: 120000, polling: 500 });
  await page.waitForTimeout(6000);
  await ready();
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'Amsterdam check'));
  await page.waitForTimeout(6000);
  await ready();
  await page.waitForFunction(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return !!s.slots.find((x) => x.key === s.active); }, undefined, { timeout: 60000 });
  const slotName = () => page.evaluate(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return s.slots.find((x) => x.key === s.active)?.name; });
  await page.waitForFunction(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return s.slots.find((x) => x.key === s.active)?.name === 'Amsterdam check'; }, undefined, { timeout: 30000 }).catch(() => {});
  const gameName = await slotName();
  await page.evaluate(async () => { (await import('/src/game/community.ts')).shareAsk.value = true; });
  const field = page.getByLabel('Title');
  try { await field.waitFor({ timeout: 15000 }); }
  catch (e) { await page.screenshot({ path: `${ARTIFACTS}t3144-fail.png` }); console.log(errors.join(' | '), gameName); throw e; }
  console.log('      slots', await page.evaluate(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return JSON.stringify({ active: s.active, slots: s.slots.map((x) => [x.key, x.name]) }); }));
  ok((await field.inputValue()) === gameName && !!gameName, `the title starts as the game's name (${gameName})`);
  ok((await field.getAttribute('maxlength')) === '40', 'the title field stops at 40 characters');
  await page.screenshot({ path: `${ARTIFACTS}t3144-share.png` });
  await field.fill('My Amsterdam');
  const send = page.locator('.cm-ask .cm-ask-foot .btn.text').first();
  await send.click();
  await page.waitForFunction(() => document.querySelector('.cm-ask .acct-msg'), undefined, { timeout: 180000 });
  console.log('      message:', await page.locator('.cm-ask .acct-msg').textContent());
  ok(posts.length === 1 && posts[0]?.title === 'My Amsterdam', `the typed title goes up (${JSON.stringify(posts[0])})`);
  // blank goes up as the game's name
  await field.fill('');
  await send.click();
  for (let i = 0; i < 600 && posts.length < 2; i++) await page.waitForTimeout(300);
  ok(posts[1]?.title === gameName, `a blank title goes up as the game's name (${JSON.stringify(posts[1])})`);
  // refused
  refuseTitle = true;
  await field.fill('nope');
  await send.click();
  await page.waitForFunction(() => /different title/.test(document.querySelector('.cm-ask .acct-msg')?.textContent ?? ''), undefined, { timeout: 180000 });
  ok(true, 'a refused title says "Please pick a different title."');
} finally {
  await close();
}
ok(errors.length === 0, `no console or page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
console.log(failed ? 'FAIL t3144-community' : 'PASS t3144-community');
process.exit(failed ? 1 : 0);
