// T-3257: View on a shared network, against a mocked server. View loads the network as a game in no slot (the saves
// list is unchanged), the saves panel shows it as its own card with only Download and Save, the top bar says it isn't
// saved, opening another game asks first (Stay keeps it), Save makes it a saved game named "<title> - <name>", and a
// viewed game left with "Leave without saving" adds nothing.
//   node trainworld/checks/t3257-view.mjs [url]          (needs a Vite: dev-nowatch.mjs <port>; TW_CORES=2)
import { readFileSync } from 'node:fs';
import { launchChrome, collectErrors, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6144/';
const SERVER = 'http://127.0.0.1:8083';
const SAVE = readFileSync(new URL('../../artifacts/trainworld/t3099-anita-amsterdam.save', import.meta.url));
const CORS = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': '*' };
const json = (body, status = 200) => ({ status, headers: { ...CORS, 'content-type': 'application/json' }, body: JSON.stringify(body) });
const at = Math.floor(Date.now() / 1000);
const ROW = { id: 'h3kzm9qa', rev: 1, name: 'kanocifer', title: 'Guangzhou', city: '', points: 0, sandbox: true, copy: true, lines: 6, stations: 41, day: 1, likes: 1, at };
const NAME = 'Guangzhou - kanocifer';

let failed = 0;
const ok = (c, msg) => { console.log(`  ${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) failed++; };
const errors = [];
const { ctx, close } = await launchChrome({ prefix: 'tw-t3257-', size: [1400, 900] });
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
    const req = route.request(), path = new URL(req.url()).pathname;
    if (req.method() === 'OPTIONS') return route.fulfill({ status: 204, headers: CORS });
    if (path === '/noritetsu/config') return route.fulfill(json({ googleClientId: '', dev: false, password: true, signIn: true, trainworld: true, trainworldCommunity: true }));
    if (path === '/noritetsu/refresh') return route.fulfill(json({ token: 'tok', account: { id: 'acct1', logins: ['pw:anita'], email: '', pendingEmail: '' } }));
    if (path === '/trainworld/me') return route.fulfill(json({ board: { on: true, name: 'anita' } }));
    if (path === '/trainworld/community/list') return route.fulfill(json({ view: 'random', city: '', seed: 'x', count: 1, rows: [ROW], left: 3, perDay: 3 }));
    if (path === `/trainworld/community/${ROW.id}/${ROW.rev}/save`) return route.fulfill({ status: 200, headers: { ...CORS, 'content-type': 'application/octet-stream' }, body: SAVE });
    if (path.startsWith('/trainworld/community/')) return route.fulfill(json({ error: 'none' }, 404));
    return route.fulfill(json({}));
  });
  const slots = () => page.evaluate(async () => { const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value; return { active: s.active, writable: s.writable, viewing: s.viewing?.name ?? null, names: s.slots.map((x) => x.name) }; });
  // the game is up and the saves state is `kind`: writable, playing the slot `name`, viewing `name`, or saved as `name`
  // (polled from here: the page reloads between games, and waitForFunction takes an async predicate's promise as true)
  const gameUp = async (kind, name = '') => {
    for (const end = Date.now() + 180000; Date.now() < end; await page.waitForTimeout(500)) {
      const up = await page.evaluate(async ([kind, name]) => {
        if (!window.tw?.world?.value) return false;
        const s = (await import('/src/workers/clockClient.ts')).saveSlotsState.value;
        const playing = s.slots.some((x) => x.key === s.active && x.name === name);
        return kind === 'writable' ? s.writable : kind === 'playing' ? playing : kind === 'viewing' ? s.viewing?.name === name : !s.viewing && s.writable && playing;
      }, [kind, name]).catch(() => false);
      if (up) return;
    }
    throw new Error(`gave up waiting for ${kind} ${name}`);
  };
  const view = async () => {
    await page.evaluate(async () => { (await import('/src/game/community.ts')).communityOpen.value = true; });
    await page.waitForSelector('.cm-card', { timeout: 30000 });
    await page.locator('.cm-open').first().click();
    await page.waitForSelector('.cm-open-copy');
    ok((await page.locator('.cm-open-copy').textContent()) === 'View', 'the button on an opened share says View');
    await page.locator('.cm-open-copy').click();
  };

  await page.goto(`${BASE}?debug=1&paused=1&acct=local&v=${Date.now()}`);
  await gameUp('writable');
  // a game of the player's own to come back to
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('create', undefined, 'My game'));
  await gameUp('playing', 'My game');
  const before = await slots();
  console.log('      before', JSON.stringify(before));

  // View
  await view();
  await gameUp('viewing', NAME);
  const v = await slots();
  ok(v.active === '' && !v.writable, `a viewed game is in no slot (active "${v.active}")`);
  ok(JSON.stringify(v.names.sort()) === JSON.stringify([...before.names].sort()), `View added no saved game (${v.names.join(', ')})`);
  const notice = await page.locator('.save-notice').textContent();
  ok(notice === `${NAME} isn't saved.Save`, `the top bar says it isn't saved, with Save (${notice})`);

  // the saves panel's card
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector('.save-view .save-thumb img', { timeout: 30000 }).catch(() => {});
  const card = await page.evaluate(() => {
    const c = document.querySelector('.save-view');
    return c && { name: c.querySelector('.save-name')?.textContent, mark: c.querySelector('.save-playing')?.textContent, when: c.querySelector('.save-when')?.textContent,
      buttons: [...c.querySelectorAll('.save-actions button')].map((b) => b.getAttribute('aria-label') || b.textContent), picture: !!c.querySelector('.save-thumb img'),
      first: document.querySelector('.save-slots > li') === c, copyIcon: !!document.querySelector('.save-files .save-copy') };
  });
  console.log('      card', JSON.stringify(card));
  ok(card?.first && card.name.startsWith(NAME) && card.mark === 'Viewing' && card.when === 'Not saved', 'its card is first, named, marked Viewing and Not saved');
  ok(JSON.stringify(card?.buttons) === JSON.stringify(['Download as a file', 'Save']), `its only buttons are Download and Save (${card?.buttons})`);
  ok(card?.picture, 'its card has a picture');
  ok(!card?.copyIcon, 'no Make a copy icon above the cards');
  await page.addStyleTag({ content: '#debug { display: none !important; }' });
  await page.screenshot({ path: `${ARTIFACTS}t3257/view-card.png` });

  // another game asks first; Stay keeps the view
  await page.locator('li[data-slot]:not(.save-view) .save-open').first().click();
  await page.waitForSelector('dialog.confirmation', { timeout: 10000 });
  const q = await page.locator('#confirmation-text').textContent();
  ok(q === `${NAME} won't be saved.`, `Open asks first (${q})`);
  ok((await page.locator('dialog.confirmation .btn.danger').textContent()) === 'Leave without saving', 'the red button says Leave without saving');
  await page.screenshot({ path: `${ARTIFACTS}t3257/leave-ask.png` });
  await page.locator('dialog.confirmation .btn:not(.danger)').click();
  await page.waitForTimeout(1500);
  ok((await slots()).viewing === NAME, 'Stay keeps the viewed game');

  // Save
  await page.keyboard.press('Control+s');
  await gameUp('saved', NAME);
  const saved = await slots();
  ok(saved.names.length === before.names.length + 1, `Save made one saved game (${saved.names.join(', ')})`);
  ok((await page.locator('.save-notice').count()) === 0, 'the top bar notice has gone');
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; });
  await page.waitForSelector('li.on .save-playing');
  await page.waitForTimeout(500);
  ok((await page.locator('.save-view').count()) === 0 && (await page.locator('li.on .save-name').first().textContent()).startsWith(NAME), 'the saved view is the Playing card');
  await page.screenshot({ path: `${ARTIFACTS}t3257/saved.png` });
  await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });

  // View again from a saved game: no question; then leave without saving
  await view();
  await gameUp('viewing', NAME);
  ok(true, 'View from a saved game went without a question');
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).leaveToMenu(() => {}));
  await page.waitForSelector('dialog.confirmation', { timeout: 10000 });
  await page.locator('dialog.confirmation .btn.danger').click();
  await page.waitForFunction(() => !window.tw?.world?.value || document.querySelector('.start-menu'), undefined, { timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(4000);
  const names = await page.evaluate(async () => (await import('/src/workers/saveFile.ts')).listSlots().then((l) => l.map((s) => s.name)));
  ok(names.length === saved.names.length, `leaving without saving added nothing (${names.join(', ')})`);
} finally {
  await close();
}
ok(errors.length === 0, `no console or page errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
console.log(failed ? 'FAIL t3257-view' : 'PASS t3257-view');
process.exit(failed ? 1 : 0);
