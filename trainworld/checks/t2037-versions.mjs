// T-2037, T-2030: one card per game. (1) The origin rule: saves from before T-2037 named "<name> at
// <time>" or "<name> copy" join the save named <name>; "Tokyo 2" stays a game of its own. (2) The
// listing: Anita's Tokyo slots of 2026-11-02 make one card, its top the version with the latest player
// change (not the one an idle tab's clock saved last), "N other versions" opening to a row each.
// (3) The conflict rule and (4) the idle-tab rule, through `settleSlot` against a fake account server:
// a slot with only the clock's saves never goes up and takes a newer account copy; a player change
// here goes up over a copy the clock alone moved on, and splits off as a version of the same origin
// when the account's copy has a player change of its own. Then pictures of the start menu and the
// saves panel, light and dark, collapsed and opened, to artifacts/trainworld/t2037/.
//   node trainworld/checks/t2037-versions.mjs [url]     (from the maps root; default localhost:6153)
// Runs in a throwaway Chrome profile. Prints PASS or FAIL lines and exits 1 on a failure.
import { readFile, mkdir } from 'node:fs/promises';
import { launchChrome, collectErrors, REAL_SAVE, ARTIFACTS, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6153/';
const DIR = ARTIFACTS + 't2037/';
await mkdir(DIR, { recursive: true });
let failed = false;
const report = (name, ok, detail = '') => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const bytes = Array.from(await readFile(REAL_SAVE));
const H = 3600_000, NOW = Date.now();

// Anita's slots (2026-11-02), with sync ids as the account had them; `edit` absent as before T-2037,
// but on the idle tab's "Tokyo", whose clock saved last (`at`) long after its last player change
const SEED = [
  { key: 'slot-t2037-tokyo', name: 'Tokyo', at: NOW - 10 * 60_000, edit: NOW - 50 * H, stations: 120, id: 'tokyoaaaa001' },
  { key: 'slot-t2037-t102', name: 'Tokyo at 1:02 PM', at: NOW - 70 * H, stations: 96, id: 'tokyobbbb002' },
  { key: 'slot-t2037-t230', name: 'Tokyo at 2:30 PM', at: NOW - 66 * H, stations: 101, id: 'tokyocccc003' },
  { key: 'slot-t2037-t617', name: 'Tokyo at 6:17 PM', at: NOW - 2 * H, edit: NOW - 2 * H, stations: 164, id: 'tokyodddd004' },
  { key: 'slot-t2037-tokyo2', name: 'Tokyo 2', at: NOW - 30 * H, stations: 12, id: 'tokyoeeee005' },
  { key: 'slot-t2037-paris', name: 'Paris', at: NOW - 5 * H, stations: 40, id: 'parisaaaa001' },
  { key: 'slot-t2037-pcopy', name: 'Paris copy', at: NOW - 4 * H, stations: 44 },
];

const { ctx, close } = await launchChrome({ prefix: 'tw-t2037-', size: [1300, 950], args: ANGLE });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&demandWorkers=1&v=${NOW}`);
  await page.waitForSelector('.start-menu', { timeout: 120000 });

  // ---- 1, 2: the origin rule and the listing, on the seeded slots
  const listed = await page.evaluate(async ([seed, bytes]) => {
    const sf = await import('/src/workers/saveFile.ts'), v = await import('/src/workers/versions.ts');
    const b = Uint8Array.from(bytes);
    for (const s of seed) {
      await sf.writeSlot(s.key, { bytes: b, at: s.at, day: 30, name: s.name, cities: ['tokyo'], lines: 20, stations: s.stations, cash: 4000, ...(s.edit ? { edit: s.edit } : {}) });
      if (s.id) await sf.writeSync(s.key, { id: s.id, acct: 'acct-t2037', rev: 3, syncedAt: s.at });
    }
    const slots = (await sf.listSlots()).filter((s) => s.key.startsWith('slot-t2037-'));
    const groups = v.groupVersions(slots, []);
    return groups.map((g) => ({ origin: g.origin, top: g.top.slot.key, others: g.others.map((o) => o.slot.key) }));
  }, [SEED, bytes]);
  const tokyo = listed.find((g) => g.others.length === 3);
  report('origin rule: the three forks join "Tokyo", "Paris copy" joins "Paris", "Tokyo 2" stands alone',
    listed.length === 3 && tokyo?.origin === 'tokyoaaaa001' && listed.some((g) => g.top === 'slot-t2037-tokyo2' && !g.others.length) && listed.some((g) => g.origin === 'parisaaaa001' && g.others.length === 1),
    JSON.stringify(listed));
  report('the latest player change heads the card, not the idle tab\'s later clock save', tokyo?.top === 'slot-t2037-t617', `(top ${tokyo?.top})`);

  // ---- 3, 4: the conflict and idle-tab rules through settleSlot, the account a fake server in the page
  const rules = await page.evaluate(async (bytes) => {
    const sf = await import('/src/workers/saveFile.ts'), cloud = await import('/src/workers/cloud.ts'), v = await import('/src/workers/versions.ts');
    const b = Uint8Array.from(bytes), puts = [], gets = [];
    const games = new Map();
    const real = window.fetch;
    window.fetch = async (url, init = {}) => {
      if (!String(url).startsWith('http://fake.t2037/')) return real(url, init);
      const path = String(url).slice('http://fake.t2037/'.length), m = /^trainworld\/games\/([a-z0-9]+)(\/picture)?$/.exec(path);
      if (m?.[2]) return new Response(null, { status: 404 });
      if (m && init.method === 'PUT') {
        const meta = JSON.parse(init.body.get('meta'));
        puts.push({ id: m[1], force: meta.force, baseRev: meta.baseRev, origin: meta.facts?.origin, edit: meta.facts?.edit });
        const g = games.get(m[1]), rev = (g?.rev ?? 0) + 1;
        games.set(m[1], { ...(g ?? { id: m[1], name: meta.name }), rev, facts: meta.facts, writer: meta.device });
        return Response.json({ rev, savedAt: Math.floor(Date.now() / 1000), used: 1 });
      }
      if (m) { gets.push(m[1]); return new Response(b, { status: 200, headers: { 'X-Rev': String(games.get(m[1])?.rev ?? 0), 'X-Saved-At': String(Math.floor(Date.now() / 1000)) } }); }
      return new Response(null, { status: 404 });
    };
    const ctx = { base: 'http://fake.t2037/', token: 't', acct: 'acct-t2037', device: 'dev-a', on: true };
    const T = 1_700_000_000_000;
    const game = (id, rev, edit, writer = 'dev-a') => ({ id, name: 'Rule', rev, writer, writerAt: 0, savedAt: Math.floor(T / 1000), saveBytes: b.length, facts: { at: edit + 60_000, edit, day: 31, stations: 50 } });
    /** a slot last synced at rev 5 with its player change at T; `edit` its latest change now, `at` its last write */
    const run = async (name, { edit, at }, g) => {
      const key = 'slot-t2037-rule-' + name, id = 'rule' + name;
      await sf.writeSlot(key, { bytes: b, at, day: 30, name: 'Rule ' + name, edit, origin: 'origin-rules', stations: 40, lines: 4, cities: [], cash: 1 });
      await sf.writeSync(key, { id, acct: ctx.acct, rev: 5, syncedAt: T + 1000, edit: T });
      games.clear(); puts.length = 0; gets.length = 0;
      const ag = g(id);
      games.set(id, ag);
      const wrote = await cloud.settleSlot(ctx, key, { games: [ag], tombstones: [], used: 0, limit: 1, maxGames: 50 });
      const [s, rec] = await Promise.all([sf.readStored(key), sf.readSync(key)]);
      const out = { wrote, puts: puts.map((p) => ({ ...p })), gets: [...gets], id: rec?.id, oldId: id, origin: s?.origin, at: s?.at, edit: s?.edit };
      await sf.deleteSlot(key);
      return out;
    };
    const idle = { edit: T, at: T + 9 * 3600_000 }, edited = { edit: T + 3 * 3600_000, at: T + 3 * 3600_000 };
    const r = {
      idleSame: await run('a', idle, (id) => game(id, 5, T)),
      idleMoved: await run('b', idle, (id) => game(id, 6, T + 2 * 3600_000, 'dev-b')),
      clockThere: await run('c', edited, (id) => game(id, 7, T, 'dev-b')),
      bothChanged: await run('d', edited, (id) => game(id, 6, T + 3600_000, 'dev-b')),
      openedThere: await run('e', edited, (id) => game(id, 5, T, 'dev-b')),
      mine: await run('f', edited, (id) => game(id, 5, T)),
    };
    // the elsewhere rule (clock.worker upNow): a writer at this slot's rev changed nothing it lacks
    const rec = { id: 'x', rev: 5, syncedAt: T + 1000, edit: T };
    r.elsewhere = { idleWriter: cloud.editedThere(game('x', 5, T), rec), clockWriter: cloud.editedThere(game('x', 6, T), rec), busyWriter: cloud.editedThere(game('x', 6, T + 1), rec) };
    // the split version and the account's copy still make one card, the later change on top
    const split = { key: 'slot-t2037-split', name: 'Rule d', at: T + 3 * 3600_000, day: 30, edit: T + 3 * 3600_000, origin: 'origin-rules', cloud: { id: r.bothChanged.id, rev: 1, syncedAt: 0 } };
    const g = v.groupVersions([split], [{ ...game(r.bothChanged.oldId, 6, T + 3600_000, 'dev-b'), name: 'Rule d', facts: { edit: T + 3600_000, origin: 'origin-rules' } }]);
    r.splitCard = { cards: g.length, top: g[0]?.top.kind, others: g[0]?.others.map((o) => o.kind) };
    window.fetch = real;
    return r;
  }, bytes);
  const { idleSame, idleMoved, clockThere, bothChanged, openedThere, mine } = rules;
  report('idle tab: clock saves alone never go up', !idleSame.wrote && !idleSame.puts.length, JSON.stringify(idleSame));
  report('idle tab: a newer account copy replaces clock saves', idleMoved.wrote && !idleMoved.puts.length && idleMoved.gets.length === 1 && idleMoved.id === idleMoved.oldId, JSON.stringify(idleMoved));
  report('a player change goes up over a copy the clock alone moved on', clockThere.puts.length === 1 && clockThere.puts[0].force === true && clockThere.puts[0].id === clockThere.oldId && clockThere.id === clockThere.oldId, JSON.stringify(clockThere));
  report('changed on both sides: this version splits off, same origin, no new card', bothChanged.puts.length === 1 && bothChanged.id !== bothChanged.oldId && bothChanged.puts[0].id === bothChanged.id && bothChanged.puts[0].baseRev === 0 && bothChanged.puts[0].origin === 'origin-rules' && !bothChanged.gets.length,
    JSON.stringify(bothChanged));
  report('opened on another device from this rev: split off too', openedThere.puts.length === 1 && openedThere.id !== openedThere.oldId, JSON.stringify(openedThere));
  report('this device the writer at its rev: goes up as before', mine.puts.length === 1 && mine.puts[0].force === false && mine.id === mine.oldId && mine.puts[0].edit === mine.edit, JSON.stringify(mine));
  report('elsewhere from an idle writer does not count as a change there', !rules.elsewhere.idleWriter && !rules.elsewhere.clockWriter && rules.elsewhere.busyWriter, JSON.stringify(rules.elsewhere));
  report('a split version and the account copy share one card, the later change on top', rules.splitCard.cards === 1 && rules.splitCard.top === 'slot', JSON.stringify(rules.splitCard));

  // ---- the start menu, then the game, the saves panel and pictures
  await page.reload();
  await page.waitForSelector('.start-saves li');
  const cards = await page.locator('.start-saves > li').count();
  const line = await page.locator('.start-saves > li[data-origin="tokyoaaaa001"] .save-versions-line').textContent().catch(() => '');
  const topKey = await page.locator('.start-saves > li[data-origin="tokyoaaaa001"] .start-save').getAttribute('data-slot').catch(() => '');
  report('start menu: one card per game', cards === 3 && topKey === 'slot-t2037-t617' && line === '3 other versions', `(${cards} cards, top ${topKey}, line "${line}")`);

  // the game opened from the card, saved by hand to draw its picture, which the other seeds borrow
  await page.click('.start-saves > li[data-origin="tokyoaaaa001"] .start-save');
  await page.waitForFunction(() => window.tw?.world?.value, undefined, { timeout: 240000 });
  await page.waitForTimeout(3000);
  await page.evaluate(async () => (await import('/src/workers/clockClient.ts')).manageSaveSlot('save'));
  await page.waitForTimeout(4000);
  const after = await page.evaluate(async () => {
    const sf = await import('/src/workers/saveFile.ts');
    const s = await sf.readStored('slot-t2037-t617');
    for (const k of ['slot-t2037-tokyo', 'slot-t2037-t102', 'slot-t2037-t230', 'slot-t2037-tokyo2', 'slot-t2037-paris', 'slot-t2037-pcopy']) {
      const o = await sf.readStored(k);
      if (o) await sf.writeStored(k, { ...o, thumb: s.thumb, panels: s.panels, shade: s.shade });
    }
    return { origin: s.origin, name: s.name, edit: s.edit, at: s.at, thumb: !!s.thumb };
  });
  report('the game opened is stamped with its origin, the fork\'s time off its name; a save by hand keeps the latest player change',
    after.origin === 'tokyoaaaa001' && after.name === 'Tokyo' && after.edit === SEED[3].edit && after.at > SEED[3].at && after.thumb, JSON.stringify(after));

  const shot = async (name) => { await page.waitForTimeout(500); await page.screenshot({ path: `${DIR}${name}.png` }); };
  for (const sh of ['light', 'dark']) {
    await page.evaluate(async (sh) => { (await import('/src/game/ui.ts')).shade.value = sh; (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = true; }, sh);
    await page.waitForSelector('.save-slots li.on');
    const open = await page.locator('.save-slots li.on .save-versions-line').getAttribute('aria-expanded');
    if (open === 'true') await page.click('.save-slots li.on .save-versions-line');
    await shot(`panel-${sh}-collapsed`);
    await page.click('.save-slots li.on .save-versions-line');
    await shot(`panel-${sh}-open`);
    if (sh === 'light') {
      const rows = await page.locator('.save-slots li.on .save-version-list > li').count();
      const playingTop = await page.locator('.save-slots li.on').getAttribute('data-slot');
      const line = await page.locator('.save-slots li.on .save-versions-line').textContent();
      // the backup written with the hand save is the card's own save, so it is not listed
      report('saves panel: the playing version heads its card, the others listed, not its own save again', playingTop === 'slot-t2037-t617' && rows === 3 && line === '3 other versions', `(top ${playingTop}, ${rows} rows, "${line}")`);
      const facts = await page.locator('.save-slots .save-info').allTextContents();
      report('no trailing dot after the day when no city follows', facts.every((t) => !/Day \d+ ·\s*(\d|$)/.test(t)), JSON.stringify(facts.map((t) => t.match(/Day \d+[^\d]{0,4}/)?.[0])));
    }
    await page.evaluate(async () => { (await import('/src/ui/SavesPanel.tsx')).savesOpen.value = false; });
  }
  // the start menu in both themes: a fresh page in this profile (the game's slot let go first)
  await page.close();
  for (const sh of ['light', 'dark']) {
    const m = await ctx.newPage();
    collectErrors(m, errors);
    await m.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, sh);
    await m.goto(`${BASE}?debug=1&demandWorkers=1&v=${Date.now()}`);
    await m.waitForSelector('.start-saves li');
    await m.waitForTimeout(1500);
    await m.screenshot({ path: `${DIR}menu-${sh}-collapsed.png` });
    await m.click('.start-saves > li[data-origin="tokyoaaaa001"] .save-versions-line');
    await m.waitForTimeout(500);
    await m.screenshot({ path: `${DIR}menu-${sh}-open.png` });
    await m.close();
  }
  report('no console errors', !errors.length, errors.slice(0, 5).join(' | '));
} finally {
  await close();
}
process.exit(failed ? 1 : 0);
