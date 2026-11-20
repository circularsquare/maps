// T-3388 trace (from T-3512's): one save loaded to final, then train count edits, each timed to its final with the
// solve's own timestamps and every worker console line. env: NAME, SAVE, BASE, LIB (checks dir URL),
// EDITS (comma list of kinds: all, high, undo), OUT dir
import { writeFileSync, readFileSync, appendFileSync, mkdirSync } from 'node:fs';
const LIB = process.env.LIB ?? 'file:///C:/Users/anita/projects/maps/.tw-trees/trainrate/trainworld/checks/';
const { launchChrome, collectErrors, importSave, demandFinal, NOISE, REAL_SAVE } = await import(LIB + 'lib.mjs');
const SCR = 'C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f4ff94a8-c06c-4457-a593-caeb3269d7c2/scratchpad/trainrate/browser/';
const name = process.env.NAME ?? 'run';
const SAVE = process.env.SAVE ?? REAL_SAVE;
const BASE = process.env.BASE ?? 'http://localhost:6182/';
const EDITS = (process.env.EDITS ?? 'all,undo,high,undo').split(',');
const OUT = SCR + 'out/';
mkdirSync(OUT, { recursive: true });
const profile = (SCR + 'profile-' + name).replace(/\//g, '\\');
const say = (...a) => { const l = new Date().toISOString().slice(11, 19) + ' ' + name + ' ' + a.join(' '); console.log(l); appendFileSync(OUT + 'runs.log', l + '\n'); };
const sec = (s) => new Promise((r) => setTimeout(r, s * 1000));
const out = { name, save: SAVE, errors: [], edits: [] };
const logs = [];
const { ctx, close } = await launchChrome({ profile });
const FIG = async () => {
  const d = window.twDemand, v = d.view();
  const h = (arrs) => { let x = 2166136261 >>> 0; for (const a of arrs) { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); for (let i = 0; i < u.length; i++) { x ^= u[i]; x = Math.imul(x, 16777619) >>> 0; } } return x.toString(16); };
  const lines = {};
  for (const [id, l] of v.lines) lines[id] = { r: l.ridersPerDay, b: (l.boardings ?? []).reduce((a, x) => a + x, 0), h: h(l.loads ?? []) };
  let points = null;
  try { const m = await import('/src/game/points.ts'); points = Object.fromEntries(m.pointsScore.value); } catch (e) { points = 'err ' + String(e).slice(0, 80); }
  const ic = d.intercity?.();
  let stH = 0; { const s = JSON.stringify([...v.stations].map(([k, x]) => [k, x.boardings ?? x.ridersPerDay ?? null])); for (let i = 0; i < s.length; i++) stH = (Math.imul(stH, 31) + s.charCodeAt(i)) | 0; }
  return { version: v.version, round: v.round, rail: v.railTripsPerDay, trips: v.tripsPerDay, split: v.split, icRail: ic?.railTripsPerDay ?? null, icTrips: ic?.tripsPerDay ?? null, nLines: v.lines.size, lines, points, stH };
};
// the newest solve's timestamps, relative to `t0` (page performance.now())
const SOLVE = (t0) => {
  const d = window.twDemand, s = d.log.findLast((x) => !d.client.fversions?.has(x.version));
  const r = (x) => (x == null ? null : Math.round(x - t0));
  // the network solved, hashed: typed arrays by their bytes, the rest as JSON
  let nh = 2166136261 >>> 0;
  const eat = (u) => { for (let i = 0; i < u.length; i++) { nh ^= u[i]; nh = Math.imul(nh, 16777619) >>> 0; } };
  const walk = (x, depth) => {
    if (x == null || depth > 4) return;
    if (ArrayBuffer.isView(x)) return eat(new Uint8Array(x.buffer, x.byteOffset, x.byteLength));
    if (typeof x !== 'object') return eat(new TextEncoder().encode(String(x)));
    for (const k of Object.keys(x).sort()) { eat(new TextEncoder().encode(k)); walk(x[k], depth + 1); }
  };
  try { walk(s.svc, 0); } catch {}
  const cr = s.clusterRes ? [...s.clusterRes].map(([k, v]) => ({ k: k.slice(0, 60), at: r(v.at), roundAt: v.roundAt?.map(r) })) : null;
  return { netHash: nh.toString(16), version: s.version, arrivedAt: r(s.arrivedAt), sentAt: r(s.sentAt), firstAt: r(s.firstAt), refinedAt: r(s.refinedAt), earlyAt: r(s.earlyAt), roundAt: s.roundAt?.map(r), roundFirstAt: s.roundFirstAt?.map(r), localShownAt: r(s.localShownAt), heldFrom: r(s.heldFrom), wholeSent: s.wholeSent, localPlans: s.localPlans, clusters: s.clusters, periods: s.periods, local: s.local, work: s.work, setupMs: s.setupMs, periodMs: s.periodMs, clusterRes: cr, nLog: d.log.length, versions: d.log.slice(-6).map((x) => x.version) };
};
try {
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  collectErrors(page, out.errors, NOISE, /glyphs/);
  const t00 = Date.now();
  page.on('console', (m) => { const t = m.text(); if (/^\[vite\]|Download the|glyph/.test(t)) return; logs.push([Date.now() - t00, t.slice(0, 400)]); });
  page.on('worker', (w) => { try { w.on('console', (m) => logs.push([Date.now() - t00, 'W ' + m.text().slice(0, 400)])); } catch {} });
  await page.addInitScript(() => { const W = window.Worker; window.__ed = []; window.Worker = class extends W { constructor(...a) { super(...a); this.addEventListener('message', (e) => { const d = e.data; if (d && d.kind === 'edited') window.__ed.push({ t: performance.now() - (window.__t0 ?? 0), ms: d.ms }); }); } }; });
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  await page.goto(BASE + '?debug=1&new=1&paused=1' + (process.env.QS ?? '') + '&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await sec(8);
  const t0 = Date.now();
  await importSave(page, readFileSync(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 3600000, polling: 100, lines: 0 });
  out.final = (Date.now() - t0) / 1000;
  out.workers = await page.evaluate(() => ({ computeN: window.twDemand.client.computeN, poolN: window.twDemand.client.poolN }));
  say('load final', out.final, 's workers', JSON.stringify(out.workers));
  out.fig0 = await page.evaluate(FIG);
  const busiest = process.env.LINE ?? Object.entries(out.fig0.lines).sort((a, b) => b[1].r - a[1].r)[0][0];
  out.busiest = busiest;
  for (const kind of EDITS) {
    await sec(Number(process.env.GAP ?? 5));
    const v0 = await page.evaluate(() => window.twDemand.view().version);
    const wall0 = Date.now();
    const li = logs.length;
    const res = await page.evaluate(async ([kind, id]) => {
      const tw = window.tw, w = tw.world.peek?.() ?? tw.world.value;
      window.__t0 = performance.now();
      if (kind === 'undo') return JSON.stringify(await tw.client.edit({ op: 'undo' }))?.slice(0, 200);
      const l = w.save.lines.find((x) => String(x.id) === id);
      const had = w.lineStats[l.id].trainsNeeded;
      const lv = { high: 0, med: 1, low: 2 }[kind];
      const t = lv !== undefined ? had.map((x, i) => (i === lv ? x + 1 : x)) : had.map((x) => x + 1);
      return 'line ' + id + ' num ' + l.num + ' trains ' + JSON.stringify(had) + ' -> ' + JSON.stringify(t) + ' ' + JSON.stringify(await tw.client.edit({ op: 'setTrainCount', line: l.num, trains: t }))?.slice(0, 200);
    }, [kind, busiest]);
    const tAns = (Date.now() - wall0) / 1000;
    // the line's riders as first changed (the local result), when
    await demandFinal(page, { timeout: 1800000, polling: 100, after: v0 });
    const settle = (Date.now() - wall0) / 1000;
    const solve = await page.evaluate(`(${SOLVE})(window.__t0)`);
    const f = await page.evaluate(FIG);
    const elog = logs.slice(li).map(([t, x]) => [t - (wall0 - t00), x]);
    const ed = await page.evaluate(() => window.__ed.slice(-3));
    out.edits.push({ kind, res, tAns, settle, solve, fig: f, log: elog, ed });
    say(kind, 'settle', settle, 's', res.slice(0, 100), 'arrived', solve.arrivedAt, 'local', solve.localShownAt, 'refined', solve.refinedAt);
  }
} catch (e) { out.error = String(e.stack ?? e).slice(0, 600); say('error', out.error); } finally {
  out.errorsN = out.errors.length; out.errors = out.errors.slice(0, 20);
  out.loadLog = logs.slice(0, 400);
  writeFileSync(OUT + name + '.json', JSON.stringify(out, null, 1));
  await close();
}
say('done');
process.exit(0);
