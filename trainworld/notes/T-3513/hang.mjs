// settle2: a program of ops on one game, every console line kept, a state dump when a wait times out.
// env: NAME BASE LIB SAVE(optional) QS OPS(json array) REPEAT(times the ops run) WAITMAX(s, default 900)
// ops: {t:'train',k:'low|med|high|all',line:'busiest'} {t:'undo'} {t:'wait'} {t:'sleep',s}
import { writeFileSync, readFileSync, appendFileSync, mkdirSync } from 'node:fs';
const LIB = process.env.LIB ?? 'file:///C:/Users/anita/projects/maps/.tw-trees/crowdfast/trainworld/checks/';
const { launchChrome, collectErrors, importSave, demandFinal, NOISE, REAL_SAVE } = await import(LIB + 'lib.mjs');
const SCR = 'C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/ca3a5e8b-220b-4b3d-a5e9-87f500e687dc/scratchpad/crowdfast/';
const name = process.env.NAME, BASE = process.env.BASE ?? 'http://localhost:6160/';
const SAVE = process.env.SAVE === 'none' ? null : (process.env.SAVE ?? REAL_SAVE);
const OPS = JSON.parse(process.env.OPS ?? '[]');
const REPEAT = Number(process.env.REPEAT ?? 1);
const WAITMAX = Number(process.env.WAITMAX ?? 900) * 1000;
mkdirSync(SCR + 'out', { recursive: true });
const profile = (SCR + 'profile-' + name).replace(/\//g, '\\');
const say = (...a) => { const l = new Date().toISOString().slice(11, 19) + ' ' + name + ' ' + a.join(' '); console.log(l); appendFileSync(SCR + 'out/runs.log', l + '\n'); };
const sec = (s) => new Promise((r) => setTimeout(r, s * 1000));
const out = { name, ops: OPS, qs: process.env.QS ?? '', errors: [], steps: [] };
const logs = [];
const { ctx, close } = await launchChrome({ profile });
const FIG = async () => {
  const d = window.twDemand, v = d.view();
  const h = (arrs) => { let x = 2166136261 >>> 0; for (const a of arrs) { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); for (let i = 0; i < u.length; i++) { x ^= u[i]; x = Math.imul(x, 16777619) >>> 0; } } return x.toString(16); };
  const lines = {};
  for (const [id, l] of v.lines) lines[id] = { r: l.ridersPerDay, b: (l.boardings ?? []).reduce((a, x) => a + x, 0), h: h(l.loads ?? []) };
  let stH = 0; { const s = JSON.stringify([...v.stations].map(([k, x]) => [k, x.boardings ?? x.ridersPerDay ?? null])); for (let i = 0; i < s.length; i++) stH = (Math.imul(stH, 31) + s.charCodeAt(i)) | 0; }
  const ic = d.intercity?.();
  return { version: v.version, final: v.final, rail: v.railTripsPerDay, trips: v.tripsPerDay, icRail: ic?.railTripsPerDay ?? null, icTrips: ic?.tripsPerDay ?? null, nLines: v.lines.size, lines, stH, calc: !!d.client.waiting };
};
// a shallow picture of the client and its current solve, for a wait that never ends
const DUMP = () => {
  const d = window.twDemand, c = d.client;
  const flat = (o) => { const r = {}; if (!o) return o; for (const k of Object.keys(o)) { const v = o[k]; const t = typeof v; if (v === null || t === 'number' || t === 'boolean' || t === 'string') r[k] = v; else if (Array.isArray(v)) r[k] = v.length <= 12 ? JSON.parse(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? `typed ${x.length}` : x))) : `array ${v.length}`; else if (v instanceof Map || v instanceof Set) r[k] = `${v.constructor.name} ${v.size}`; else if (t === 'object') r[k] = v.constructor?.name ?? 'object'; } return r; };
  let cur = null; try { cur = flat(c.cur); } catch (e) { cur = String(e); }
  const log = d.log.slice(-6).map(flat);
  const v = d.view();
  return { client: flat(c), cur, log, view: { version: v?.version, final: v?.final, round: v?.round } };
};
// the game side of a wait that never ends: snapshots from the clock worker, the world's trains, the demand signals
const GAME = async () => {
  const dm = await import('/src/game/demand.ts'), cc = await import('/src/workers/clockClient.ts');
  const w = window.tw.world.peek?.() ?? window.tw.world.value, s = cc.networkSnapshot?.peek?.();
  const l0 = s?.lines?.[0];
  return { snaps: window.__snaps ?? null, snapVersion: s?.version, snapLine0: l0 && { id: l0.id, tph: l0.tph }, worldVersion: w?.version,
    trains: Object.fromEntries(Object.entries(w?.lineStats ?? {}).slice(0, 3).map(([k, x]) => [k, x.trainsNeeded])),
    networkWorld: dm.networkWorld.peek(), finalWorld: dm.finalWorld.peek(), editsSettling: dm.editsSettling.peek(), demandHeld: dm.demandHeld.peek(), demandUpdating: dm.demandUpdating.peek(),
    running: window.tw.clock?.running?.peek?.(), lastKeyLen: window.twDemand.client.lastKey?.length, pausedSnap: !!window.twDemand.client.pausedSnap };
};
let pend = null;
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  collectErrors(page, out.errors, NOISE, /glyphs/);
  const t00 = Date.now();
  page.on('console', (m) => { const t = m.text(); if (/^\[vite\]|Download the|glyph|water mask/.test(t)) return; logs.push([Date.now() - t00, t.slice(0, 300)]); });
  page.on('worker', (w) => { try { w.on('console', (m) => logs.push([Date.now() - t00, 'W ' + m.text().slice(0, 300)])); } catch {} });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(BASE + '?debug=1&new=1&paused=1' + (process.env.QS ?? '') + '&v=' + Date.now(), { timeout: 180000 });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(async () => { const cc = await import('/src/workers/clockClient.ts'); cc.onNetworkSnapshot((s) => (window.__snaps ??= []).push([Math.round(performance.now()), s.version, s.lines?.[0]?.tph?.join('/')])); });
  await sec(8);
  const t0 = Date.now();
  if (SAVE) {
    await importSave(page, readFileSync(SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
    await demandFinal(page, { timeout: 3600000, polling: 100, lines: 0 });
    out.load = (Date.now() - t0) / 1000;
    say('load final', out.load, 'qs', process.env.QS ?? '');
  }
  // T-3165's lowering of counts over the cap lands a second or two after a load, as ordinary undoable
  // edits: the ops start once the world has not changed for 5 s, so an undo takes back the op's edit
  {
    let wv = await page.evaluate(() => window.tw.world.value.version), since = Date.now();
    const tq = Date.now();
    while (Date.now() - since < 5000 && Date.now() - tq < 90000) {
      await sec(0.5);
      const v = await page.evaluate(() => window.tw.world.value.version);
      if (v !== wv) { wv = v; since = Date.now(); }
    }
    out.quietAfter = (Date.now() - tq) / 1000;
    if (SAVE) await demandFinal(page, { timeout: WAITMAX, polling: 100, lines: 0 });
  }
  out.fig0 = await page.evaluate(FIG);
  out.startedAt = Date.now() - t00;
  const busiest = SAVE ? Object.entries(out.fig0.lines).sort((a, b) => b[1].r - a[1].r) : [];
  let prof = null;
  if (process.env.PROF) {
    const cdp = await ctx.newCDPSession(page);
    const { targetInfos } = await cdp.send('Target.getTargets');
    const workers = targetInfos.filter((t) => t.type === 'worker' && /demand\.worker/.test(t.url));
    let nextId = 1; const waiting = new Map();
    cdp.on('Target.receivedMessageFromTarget', (e) => { const m = JSON.parse(e.message); if (m.id && waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
    const sessions = [];
    for (const w of workers) { const { sessionId } = await cdp.send('Target.attachToTarget', { targetId: w.targetId, flatten: false }); sessions.push(sessionId); }
    const send = (sessionId, method, params = {}) => new Promise((done) => { const id = nextId++; waiting.set(id, done); cdp.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id, method, params }) }); });
    for (const s of sessions) { await send(s, 'Profiler.enable'); await send(s, 'Profiler.setSamplingInterval', { interval: 1000 }); await send(s, 'Profiler.start'); }
    prof = { sessions, send };
    say('profiling', workers.length, 'workers');
  }
  // the busiest line's trains as each op should leave them (an undo: as before the op it takes back)
  const trainsOf = () => page.evaluate((id) => { const w = window.tw.world.value, l = w.save.lines.find((x) => String(x.id) === String(id)); return JSON.stringify(w.lineStats[l.id].trainsNeeded); }, busiest[0]?.[0]);
  const stack = [];
  let v0 = await page.evaluate(() => window.twDemand.view().version), wall0 = Date.now(), li = logs.length;
  for (let rep = 0; rep < REPEAT; rep++) for (const op of OPS) {
    let res = '';
    if (op.t === 'sleep') { await sec(op.s); continue; }
    if (op.t === 'relevel') {
      // a zone's level changes on its own (T1223 log, ?debug=1), then that solve's final
      const tl = Date.now();
      let nudged = 0;
      while (!logs.slice(li).some(([, x]) => /T1223 levels change/.test(x)) && Date.now() - tl < 1800000) {
        await sec(5);
        // past 11 minutes, the view moves a little each minute, in case the level check waits on a move
        if (Date.now() - tl > 660000 && Date.now() - tl > 660000 + 60000 * nudged) { nudged++; await page.evaluate((k) => { const c = window.tw.map.getCenter(); window.tw.map.jumpTo({ center: [c.lng + (k % 2 ? 0.001 : -0.001), c.lat] }); }, nudged); }
      }
      out.nudged = nudged;
      const line = logs.slice(li).find(([, x]) => /T1223 levels change/.test(x));
      say('relevel', line ? line[1].slice(0, 160) : 'NONE after 30 min', 'at', line ? (line[0] - out.startedAt) / 1000 : null, 's after the load final');
      await sec(5);
      const relevelV = await demandFinal(page, { timeout: 400000, polling: 100, after: v0, lines: null }).then(() => 'new version', () => demandFinal(page, { timeout: WAITMAX, polling: 100 }).then(() => 'same version'));
      say('relevel solve:', relevelV);
      const f = await page.evaluate(FIG);
      out.steps.push({ rep, op: 'wait', relevel: line?.[1] ?? null, settle: (Date.now() - wall0) / 1000, fig: f, lod: await page.evaluate(() => window.twDemand.client.lodState()), log: logs.slice(li).map(([t, x]) => [t - (wall0 - t00), x]) });
      say('relevel final, rail', f.rail, 'v', f.version);
      await sec(3);
      v0 = await page.evaluate(() => window.twDemand.view().version); wall0 = Date.now(); li = logs.length;
      continue;
    }
    if (op.t === 'wait') {
      // a solve newer than the shown one must start within 20 s (an op demand reads nothing of is a harness fault)
      const started = await page.waitForFunction((v0) => window.twDemand.client.version > v0, v0, { timeout: 20000, polling: 100 }).then(() => true, () => false);
      if (!started) { say('NO SOLVE rep', rep, 'within 20 s of the ops'); throw new Error('no solve started'); }
      try {
        await demandFinal(page, { timeout: WAITMAX, polling: 100, after: v0, lines: null });
      } catch (e) {
        out.hang = { rep, at: (Date.now() - wall0) / 1000, dump: await page.evaluate(DUMP).catch((x) => String(x)), game: await page.evaluate(GAME).catch((x) => String(x)), log: logs.slice(li).map(([t, x]) => [t - (wall0 - t00), x]) };
        say('HANG rep', rep, 'after', out.hang.at, 's');
        throw e;
      }
      const settle = (Date.now() - wall0) / 1000;
      const f = await page.evaluate(FIG);
      const work = await page.evaluate(() => { const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version)); return { version: s?.version, work: s?.work, localPlans: s?.localPlans, wholeSent: s?.wholeSent, localShownAt: s?.localShownAt, arrivedAt: s?.arrivedAt, refinedAt: s?.refinedAt }; }).catch((x) => String(x));
      out.steps.push({ rep, op: 'wait', settle, fig: f, work, game: await page.evaluate(GAME).catch((x) => String(x)), log: logs.slice(li).map(([t, x]) => [t - (wall0 - t00), x]) });
      say('rep', rep, 'final after', settle, 's rail', f.rail, 'v', f.version);
      await sec(3);
      v0 = await page.evaluate(() => window.twDemand.view().version); wall0 = Date.now(); li = logs.length;
      continue;
    }
    const before = await trainsOf();
    res = await page.evaluate(async ([op, busiest]) => {
      const tw = window.tw, w = tw.world.peek?.() ?? tw.world.value;
      if (op.t === 'undo') return JSON.stringify(await tw.client.edit({ op: 'undo' }))?.slice(0, 120);
      if (op.t === 'train') {
        const id = op.line === 'second' ? busiest[1][0] : op.line && op.line !== 'busiest' ? op.line : busiest[0][0];
        const l = w.save.lines.find((x) => String(x.id) === String(id));
        const had = tw.world.value.lineStats[l.id].trainsNeeded;
        const lv = { high: 0, med: 1, low: 2 }[op.k];
        const t = lv !== undefined ? had.map((x, i) => (i === lv ? x + (op.d ?? 1) : x)) : had.map((x) => x + (op.d ?? 1));
        return 'line ' + id + ' ' + JSON.stringify(had) + '->' + JSON.stringify(t) + ' ' + JSON.stringify(await tw.client.edit({ op: 'setTrainCount', line: l.num, trains: t }))?.slice(0, 100);
      }
      return 'unknown';
    }, [op, busiest]);
    let want = null;
    if (op.t === 'train') { want = res.match(/->(\[[^\]]*\])/)?.[1] ?? null; stack.push(before); }
    else if (op.t === 'undo') want = stack.pop() ?? null;
    let after = await trainsOf();
    for (let k = 0; k < 30 && want && after !== want; k++) { await sec(0.1); after = await trainsOf(); }
    say('rep', rep, JSON.stringify(op), res.slice(0, 120), 'trains', before, '->', after);
    out.steps.push({ rep, op, res: res.slice(0, 300), at: Date.now() - t00, before, after });
    if (want && want !== after) { say('ASSERT rep', rep, 'trains', after, 'want', want); throw new Error('trains not as the op implies'); }
  }
  if (prof) { for (let k = 0; k < prof.sessions.length; k++) { const r = await prof.send(prof.sessions[k], 'Profiler.stop'); writeFileSync(SCR + 'out/' + name + '-w' + k + '.cpuprofile', JSON.stringify(r.result.profile)); } say('profiles written'); }
  out.errorsN = out.errors.length;
} catch (e) { out.error = String(e.stack ?? e).slice(0, 800); say('error', out.error.slice(0, 200)); } finally {
  out.errors = out.errors.slice(0, 20);
  writeFileSync(SCR + 'out/' + name + '.json', JSON.stringify(out, null, 1));
  await close();
}
say('done');
process.exit(0);
