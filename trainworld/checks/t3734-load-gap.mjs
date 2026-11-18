// T-3734 stage 1: the gap while another game loads into the page (T-3687, T-3722). Normal game A is played in a
// slot; a sandbox game SB is kept in another. From A, SB is opened three ways the player can: Open a file, a
// backup's restore and Community's Open (a copy of a share, viewed in no slot).
//   1. A score send in the gap (T-3687): during each load the page's real `sendScores` is called on every change of
//      the money, the slot list or the account state. No sandbox mark may go out under A's game id (the server
//      would latch A as a sandbox for good). The account server is a stand-in (?acct=local answered by a route);
//      A's figures are on it as the page remembers them.
//   2. Save and quit in the gap (T-3722): Open a file, and a backup's restore, then Save and quit to main menu
//      pressed while the incoming game is still opening (the track model's script, wasm/track/sim.js, is held
//      until just after the press: a stand-in for a slow load that puts the leave before the worker has read the
//      picked game). The start menu must list the incoming game in a slot of its own.
// The share's own sandbox mark (account.ts `gameForShare`) is not covered here.
//   node trainworld/checks/t3734-load-gap.mjs [url]     (from the maps root; default localhost:6177; ONLY=score or leave)
// Prints PASS / FAIL lines and exits 1 on a FAIL. About two minutes (each half about one): seven page loads.
import { launchChrome, collectErrors, ANGLE, acrossLine } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6177/';
const ACCT = 'acct-t3734';
/** ONLY=score or ONLY=leave runs one half */
const ONLY = process.env.ONLY ?? '';
let failed = false;
const report = (name, ok, detail) => { failed ||= !ok; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const { ctx, close } = await launchChrome({ prefix: 'tw-t3734-gap-', size: [1300, 900], args: ANGLE });
const errors = [];
let stall = null;
try {
  // ---- the account server, answered here: config on, a session, every score PUT recorded
  const puts = [];
  const CORS = { 'access-control-allow-origin': '*', 'access-control-allow-headers': '*', 'access-control-allow-methods': 'GET, POST, PUT, DELETE, OPTIONS' };
  await ctx.route(/^http:\/\/127\.0\.0\.1:8083\//, async (route) => {
    const req = route.request(), path = new URL(req.url()).pathname.slice(1), method = req.method();
    const json = (body, status = 200) => route.fulfill({ status, headers: { ...CORS, 'content-type': 'application/json' }, body: JSON.stringify(body) });
    if (method === 'OPTIONS') return route.fulfill({ status: 204, headers: CORS });
    if (path === 'noritetsu/config') return json({ signIn: true, trainworld: true });
    if (path === 'noritetsu/refresh') return json({ token: 'tok-t3734', account: { id: ACCT, logins: ['dev'], email: '', pendingEmail: '' } });
    if (path.startsWith('trainworld/score/') && method === 'PUT') { puts.push({ id: path.slice(17), body: req.postData() }); return json({ listed: true }); }
    return json({});
  });
  // ---- the track model's script, held while `stall` is set (the stand-in slow load; a worker's own fetch of the
  // .wasm is not seen by routes, its module imports are)
  await ctx.route(/\/wasm\/track\/sim\.js/, async (route) => {
    if (stall) { stall.n++; await stall.held; }
    await route.continue().catch(() => {});
  });
  const stallNow = () => { let go; stall = { n: 0, held: new Promise((r) => { go = r; }) }; stall.go = () => { go(); stall = null; }; };

  await ctx.addInitScript(([acct]) => {
    try {
      if (!localStorage.getItem('noritetsu.auth.v1')) localStorage.setItem('noritetsu.auth.v1', JSON.stringify({ token: 'tok-t3734', account: { id: acct, logins: ['dev'], email: '', pendingEmail: '' } }));
    } catch {}
    // during a load from the slot named in 't3734-old': the real sendScores on every change it reads
    window.__gap = { calls: 0, unrankedWhileOld: 0 };
    const old = localStorage.getItem('t3734-old');
    if (!old) return;
    (async () => {
      while (!window.tw?.client || !window.tw?.moneyState) await new Promise((r) => setTimeout(r, 10));
      const acc = await import('/src/game/account.ts'), cc = await import('/src/workers/clockClient.ts');
      const fire = () => {
        const m = window.tw.moneyState.peek(), s = cc.saveSlotsState.peek();
        window.__gap.calls++;
        if (m && (m.mode === 'sandbox' || m.copy) && s.active === old && acc.accountState.peek() === 'on') window.__gap.unrankedWhileOld++;
        void acc.sendScores(true);
      };
      window.tw.moneyState.subscribe(fire); cc.saveSlotsState.subscribe(fire); acc.accountState.subscribe(fire);
    })();
  }, [ACCT]);

  const page = await ctx.newPage();
  collectErrors(page, errors);
  const q = 'debug=1&paused=1&lang=en&acct=local';
  /** one of the page's own calls by name (the page's CSP refuses code sent as text) */
  const call = (what, a) => page.evaluate(async ([what, a]) => {
    const cc = await import('/src/workers/clockClient.ts'), sf = await import('/src/workers/saveFile.ts'), ss = await import('/src/game/sharedSave.ts');
    const s = cc.saveSlotsState.peek();
    if (what === 'slots') return { active: s.active, writable: s.writable, viewing: !!s.viewing, keys: s.slots.map((x) => x.key) };
    if (what === 'create') return cc.manageSaveSlot('create', undefined, a);
    if (what === 'open') return void cc.manageSaveSlot('open', a);
    if (what === 'restore') return void cc.manageSaveSlot('restore', a);
    if (what === 'file') return void cc.importGame(new File([(await sf.readStored(a)).bytes], 'gap check.save'));
    if (what === 'view') return void cc.openView(await ss.copyFromShare((await sf.readStored(a)).bytes, 'abcdefgh'), 'viewed');
    throw new Error('no call ' + what);
  }, [what, a]);
  const slotsNow = () => call('slots');
  // the slot list's module kept on the page, for the waits below (a wait's test cannot await)
  const up = async () => {
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.moneyState?.value && window.twDemand?.client, undefined, { timeout: 180000 });
    await page.evaluate(async () => { window.__cc = await import('/src/workers/clockClient.ts'); });
  };
  /** Make a copy of the game in hand named `name`; its slot */
  const keep = async (name) => {
    await call('create', name);
    await page.waitForFunction((name) => { const s = window.__cc.saveSlotsState.peek(); return s.slots.some((x) => x.key === s.active && x.name === name); }, name, { timeout: 60000 });
    return (await slotsNow()).active;
  };
  const reloadedBy = async (what, a) => {
    const t = await page.evaluate(() => performance.timeOrigin);
    await call(what, a).catch(() => {}); // the page may go before it answers
    await page.waitForFunction((t) => performance.timeOrigin !== t, t, { timeout: 60000 }).catch(async () => { throw new Error(`no reload after ${what}: ` + JSON.stringify(await page.evaluate(() => ({ notice: window.tw?.notice?.value ?? null, slots: window.__cc?.saveSlotsState.peek().error, url: location.href }))) + ' ' + errors.slice(-3).join(' | ')); });
  };
  /** back in slot A, playing, with A's figures on the account as sent */
  const intoA = async (A, id) => {
    const was = await slotsNow().catch((e) => ({ e: String(e) }));
    // the slot a plain page opens is the last one opened (a viewed copy would ask before Open could leave it)
    if (was.active !== A) {
      await page.evaluate((A) => localStorage.setItem('anitabuilder-save-slot', A), A);
      await page.goto(`${BASE}?${q}&v=${Date.now()}`);
      await up();
    }
    await page.waitForFunction((A) => { const s = window.__cc.saveSlotsState.peek(); return s.active === A && s.writable; }, A, { timeout: 60000 }).catch(async (e) => { throw new Error('not back in A: ' + JSON.stringify(await slotsNow())); });
    await page.evaluate(([A, id, acct]) => {
      const all = JSON.parse(localStorage.getItem('trainworld.score-games.v1') || '{}');
      all[A] = { id, acct, sent: JSON.stringify([5000, { nyc: 5000 }]), at: Date.now() };
      localStorage.setItem('trainworld.score-games.v1', JSON.stringify(all));
      localStorage.setItem('t3734-old', A);
    }, [A, id, ACCT]);
  };

  // ---- set-up: SB (a sandbox with a line) and A (a normal game), each in a slot
  await page.goto(`${BASE}?new=1&city=nyc&mode=sandbox&${q}&v=${Date.now()}`);
  await up();
  if (await acrossLine(page, [-73.985, 40.758], 2) === null) throw new Error('set-up: no line in SB');
  const SB = await keep('SB');
  await page.goto(`${BASE}?new=1&city=nyc&${q}&v=${Date.now()}`);
  await up();
  const A = await keep('A');
  // a backup of SB to restore (written as autosave writes one)
  const SBB = await page.evaluate(async (k) => { const sf = await import('/src/workers/saveFile.ts'); const r = await sf.readStored(k); await sf.writeBackup(k, { ...r, at: r.at - 60000 }); return (await sf.listBackups(k))[0]?.key; }, SB);
  if (!SBB) throw new Error('set-up: no backup of SB');

  const arg = { file: SB, restore: SBB, view: SB };

  // ---- 1. score sends in the gap
  for (const kind of ONLY === 'leave' ? [] : ['file', 'restore', 'view']) {
    const id = `game${kind}t3734`;
    await intoA(A, id);
    puts.length = 0;
    await reloadedBy(kind, arg[kind]);
    await up();
    await page.waitForFunction((A) => { const s = window.__cc.saveSlotsState.peek(); return s.active !== A || !!s.viewing; }, A, { timeout: 60000 }).catch(() => {});
    await pause(1500);
    const gap = await page.evaluate(() => window.__gap);
    const marked = puts.filter((p) => p.id === id && /"mode":"sandbox"|"copy":true/.test(p.body ?? ''));
    const entry = await page.evaluate(([A]) => JSON.parse(localStorage.getItem('trainworld.score-games.v1') || '{}')[A], [A]);
    report(`${kind}: no sandbox mark sent for the game left`, marked.length === 0 && !entry?.sandbox,
      `(PUTs ${JSON.stringify(puts.map((p) => [p.id, (p.body ?? '').slice(0, 40)]))}; sends tried ${gap.calls}, with the incoming unranked money and A still listed ${gap.unrankedWhileOld}; A's entry ${JSON.stringify(entry)}; now ${JSON.stringify(await slotsNow())})`);
  }
  await page.evaluate(() => localStorage.removeItem('t3734-old'));

  // ---- 2. Save and quit to main menu while the incoming game opens
  for (const kind of ONLY === 'score' ? [] : ['file', 'restore']) {
    await intoA(A, 'unused');
    await page.evaluate(() => localStorage.removeItem('t3734-old'));
    const before = (await slotsNow()).keys;
    stallNow();
    await reloadedBy(kind, arg[kind]);
    let pressed = false;
    for (let i = 0; i < 40 && !pressed; i++) {
      try {
        await page.locator('button.tab.save').first().click({ timeout: 1500 });
        await page.locator('.save-main-menu').click({ timeout: 1500 });
        pressed = true;
      } catch { await pause(250); }
    }
    await pause(500);
    const stalled = stall?.n ?? 0;
    stall?.go();
    await page.locator('.start-menu').waitFor({ timeout: 90000 }).catch(() => {});
    const now = await page.evaluate(async () => (await (await import('/src/workers/saveFile.ts')).listSlots()).map((s) => ({ key: s.key, name: s.name })));
    const added = now.filter((s) => !before.includes(s.key));
    report(`${kind}: Save and quit while it opens keeps the incoming game`, pressed && stalled > 0 && added.length === 1 && now.length === before.length + 1,
      `(pressed ${pressed}; track script loads held ${stalled}; menu up ${await page.locator('.start-menu').count() > 0}; slots ${before.length} -> ${now.length}, new ${JSON.stringify(added)})`);
  }

  const real = errors.filter((e) => !/WebGL|GPU|127\.0\.0\.1:8083/i.test(e));
  report('no page errors', real.length === 0, real.slice(0, 3).join(' | '));
} finally {
  stall?.go?.();
  await close();
}
process.exit(failed ? 1 : 0);
