// T-2773: after a schedule or stop change, does any line's riders or money as the Lines tab shows them, or the fare
// income booked, fall below its settled figure while the new answer is worked out? And does the new answer land?
//   TW_CORES=4 node trainworld/checks/t2773-pending-dip.mjs [save] [url] [edits]     (from the maps root)
// save: default the real New York save (Anita's Japan one: her Downloads' trainsworld-tokyo-day-414.save).
// edits: comma list of headway, trains, stop, each on the busiest fast line (else the busiest; the second for trains);
// "stop" drops a middle stop as the line tool does (a "rows" edit, T-2766). "@view" after one (headway@view) first puts
// the camera on the line's middle stop zoomed in; "@move" moves it along the line's stops after the edit (zones refined
// aside meanwhile, T-2605). Default: the three, then each @move.
// The clock runs a while, then stops for the edits, so the riders counted over the past 24 hours move only with the
// answers; TW_CLOCK=run keeps it running at 4 game minutes a second (those counts then drift several percent as a
// weekend comes into the window). TW_MARGIN: the dip allowed (default 5%, 10% with the clock running).
// TW_PERFOFF=settledHold runs without the hold, keepFigures with the demand route before T-2793 (the before).
// TW_TAG names the output files (artifacts/trainworld/riderdip/<tag>-<edit>.json).
// Every 250 ms each running line's riders and fares a day as shown, and the fare income posted to the clock for it.
// Fails (exit 1) when a line falls more than the margin below the smaller of its figure before the edit and after the
// final, or ends more than 80% below its figure before, or the edit's final does not land (still pending or solving at the
// time limit, or the final shown not of a network made since the edit).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { launchChrome, importSave, ANGLE, ARTIFACTS, REAL_SAVE } from './lib.mjs';

const SAVE = process.argv[2] || REAL_SAVE;
const BASE = process.argv[3] ?? 'http://localhost:6206/';
const EDITS = (process.argv[4] ?? 'headway,trains,stop,headway@move,trains@move,stop@move').split(',');
const MARGIN = Number(process.env.TW_MARGIN ?? (process.env.TW_CLOCK === 'run' ? 0.1 : 0.05));
const FINAL_FALL = 0.8;
const OUT = ARTIFACTS + `riderdip/${process.env.TW_TAG ?? 'run'}-`;
await mkdir(ARTIFACTS + 'riderdip', { recursive: true });
const stamp = () => new Date().toISOString().slice(11, 19);

const { ctx, close } = await launchChrome({ prefix: 'tw-t2773-', size: [1300, 850], args: ANGLE });
const failures = [];
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  page.on('console', (m) => { if (m.type() === 'error') console.log('console.error', m.text().slice(0, 300)); });
  await page.goto(`${BASE}?debug=1&new=1${process.env.TW_PERFOFF ? `&perfOff=${process.env.TW_PERFOFF}` : ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(async () => {
    const [td, lf, ui, wd, dm] = await Promise.all(['today', 'lineFigures', 'ui', 'world', 'demand'].map((m) => import(`/src/game/${m}.ts`)));
    const tw = window.tw, R = (window.__rec = { td, lf, ui, wd, dm, posted: new Map() });
    // the income the clock is paid per line (the first kind of day's), as posted
    const post = tw.client.post.bind(tw.client);
    tw.client.post = (msg, tr) => {
      if (msg.kind === 'income') R.posted = new Map(msg.lines.map((l) => [String(l.line), l.perHour[0].reduce((a, b) => a + b, 0)]));
      return post(msg, tr);
    };
    // as the Lines tab shows them: riders (the pending figure while pending), fares a day once the profit is known
    R.snap = () => {
      const w = tw.world.value, o = {};
      for (const l of w.save.lines) {
        if (w.lineStats[l.id]?.status !== 'running') continue;
        const d = td.today.value.get(l.id), k = td.pendingOf(l);
        const riders = k ? td.pendingRiders(l) : d?.riders;
        o[l.id] = [riders ?? 0, d && td.profitKnown(l) ? d.dayFares : 0, R.posted.get(l.id) ?? 0, k ? 1 : 0];
      }
      return o;
    };
    // the line each kind of edit is made on: the busiest fast line (else the busiest), the second for a train count, the
    // busiest fast one with more than six stops for a stop
    R.pick = (kind) => {
      const w = tw.world.value, running = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id]?.bus);
      const rid = (l) => td.today.value.get(l.id)?.riders ?? 0;
      const fast = running.filter((l) => w.lineStats[l.id].fast).sort((a, b) => rid(b) - rid(a));
      const busy = [...running].sort((a, b) => rid(b) - rid(a));
      return kind === 'stop' ? (fast.find((x) => x.stops.length > 6) ?? busy.find((x) => x.stops.length > 6)) : kind === 'trains' ? (fast[1] ?? busy[1]) : (fast[0] ?? busy[0]);
    };
    R.busy = () => { const d = window.twDemand, v = d?.view?.(); return !(v?.final && !d.client.waiting) || d.client.respawn?.size > 0; };
  });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 5, undefined, { timeout: 300000, polling: 250 });
  console.log(stamp(), 'imported', await page.evaluate(() => window.tw.world.value.save.lines.length), 'lines');
  await page.evaluate(() => { window.__rec.ui.selection.value = null; window.__rec.ui.tab.value = 'lines'; });
  const settled = () => page.waitForFunction(() => {
    const R = window.__rec, ok = !R.busy() && !R.lf.linesPending.value.size && R.td.today.value.size > 3 && !document.querySelector('.load-card, .loading-card');
    R.quiet = ok ? (R.quiet ?? 0) + 1 : 0;
    return R.quiet > 12;
  }, undefined, { timeout: 900000, polling: 500 });
  await settled();
  // the clock runs, as in play; then (unless TW_CLOCK=run) stops, so the past-24-hour counts move only with the answers
  await page.evaluate(() => window.tw.clock.setSpeed(1));
  await page.waitForTimeout(8000);
  await settled();
  if (process.env.TW_CLOCK !== 'run') await page.evaluate(() => window.tw.clock.setRunning(false));
  console.log(stamp(), 'settled', process.env.TW_CLOCK === 'run' ? 'clock running at 4 game minutes a second' : 'clock stopped');

  for (const name of EDITS) {
    // "@view": the camera on the edited line's middle stop, zoomed in (its zone solved fine), before the edit
    if (name.endsWith('@view')) {
      await page.evaluate((kind) => {
        const w = window.tw.world.value, l = window.__rec.pick(kind), s = w.save.stations.find((x) => x.id === l.stops[l.stops.length >> 1]);
        window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 11.5 });
      }, name.split('@')[0]);
      await page.waitForTimeout(3000);
      await settled();
    }
    const r = await page.evaluate(async (name) => {
      const R = window.__rec, tw = window.tw, w = () => tw.world.value, { td, lf } = R;
      const sleep = (ms) => new Promise((f) => setTimeout(f, ms));
      const [kind, how] = name.split('@');
      const l = R.pick(kind), st = w().lineStats[l.id];
      let op;
      if (kind === 'headway') op ={ op: 'setHeadway', line: Number(l.num), seconds: [l.tph.high, l.tph.medium, l.tph.low].map((n) => (n > 0 ? Math.round((3600 / n) * 1.15) : 0)) };
      else if (kind === 'trains') op = { op: 'setTrainCount', line: Number(l.num), trains: st.trainsNeeded.map((n) => (n > 0 ? n + 1 : 0)) };
      else op = { op: 'setStops', line: Number(l.num), stops: l.stops.filter((_, k) => k !== l.stops.length >> 1).map(Number), autoFast: false };
      const out = { name, line: l.id, note: `${l.name} ${st.km | 0} km, ${l.stops.length} stops`, op, samples: [] };
      const base = R.snap(), t0 = performance.now();
      // as the player's controls send it (game/world.ts `edit`), a stop click as the line tool does
      void R.wd.edit(op, false, true, kind === 'stop' ? 'rows' : false).then((a) => { out.ok = a?.ok; out.answeredAt = Math.round(performance.now() - t0); out.version = w().version; });
      // "@move": the camera then to four of the line's stops in turn, zoomed in (each zone refined aside while the
      // edit's solve is in hand, T-2605)
      if (how === 'move') void (async () => {
        const stops = l.stops.map((id) => w().save.stations.find((s) => s.id === id));
        for (const k of [0.15, 0.85, 0.5, 0.3]) { const s = stops[Math.floor(k * stops.length)]; tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 11.5 }); await sleep(5000); }
      })();
      let quiet = 0;
      while (performance.now() - t0 < 360000) {
        const s = { t: Math.round(performance.now() - t0), busy: R.busy(), npend: lf.linesPending.value.size, lines: R.snap() };
        out.samples.push(s);
        s.final = R.dm.finalWorld.value;
        quiet = !s.busy && !s.npend && s.t > 10000 && s.final >= out.version ? quiet + 1 : 0;
        if (quiet > 40) break;
        await sleep(250);
      }
      out.base = base;
      return out;
    }, name);
    await writeFile(OUT + `${name}.json`, JSON.stringify(r));
    report(r);
    await settled().catch(() => console.log('not settled after', name));
  }
} finally {
  await close();
}
console.log(stamp(), 'done');
if (failures.length) { console.log('FAIL', failures.join('; ')); process.exitCode = 1; } else console.log(`PASS: no line fell more than ${MARGIN * 100}% and every edit landed`);

function report(r) {
  const fin = r.samples.at(-1);
  console.log(`\n=== ${r.name}: ${r.note}, ok ${r.ok} answered ${r.answeredAt} ms; ${r.samples.length} samples, last at ${fin.t} ms busy ${fin.busy} pending ${fin.npend}, max pending ${Math.max(...r.samples.map((s) => s.npend))}`);
  // landed: the whole final shown is of a network made at or after the edit (T-2658), and nothing is pending
  if (fin.busy || fin.npend || !r.ok || !(fin.final >= r.version)) failures.push(`${r.name}: did not land (final of world ${fin.final}, edit at ${r.version})`);
  const b = r.base[r.line], f = fin.lines[r.line];
  const moved = b && f && (Math.abs(f[1] - b[1]) > 1e-3 * Math.abs(b[1]) || Math.abs(f[2] - b[2]) > 1e-3 * Math.abs(b[2]));
  console.log(`  edited line ${r.line}: riders ${b?.[0] | 0} -> ${f?.[0] | 0}, fares/day ${(b?.[1] / 1e6).toFixed(2)}M -> ${(f?.[1] / 1e6).toFixed(2)}M, posted ${b?.[2].toFixed(2)} -> ${f?.[2].toFixed(2)}${moved ? '' : ' (its money unchanged)'}`);
  for (const [label, k] of [['riders', 0], ['fares/day', 1], ['posted income', 2]]) {
    const dipped = [];
    for (const id of Object.keys(r.base)) {
      const b0 = r.base[id][k], f0 = fin.lines[id]?.[k];
      if (!(b0 > 0) || !(f0 > 0)) continue;
      const ref = Math.min(b0, f0);
      let lo = Infinity, at = 0, pend = 0;
      for (const s of r.samples) { const x = s.lines[id]; if (x && x[k] < lo) { lo = x[k]; at = s.t; pend = x[3]; } }
      const dip = 1 - lo / ref;
      if (dip > 0.02) dipped.push({ id, b: +b0.toPrecision(4), f: +f0.toPrecision(4), lo: +lo.toPrecision(4), at, pend, dip: +dip.toFixed(3) });
    }
    dipped.sort((a, b) => b.dip - a.dip);
    const bad = dipped.filter((d) => d.dip > MARGIN);
    console.log(`  ${label}: over ${MARGIN * 100}%: ${bad.length}, over 2%: ${dipped.length}; worst ${JSON.stringify(dipped.slice(0, 5))}`);
    if (bad.length) failures.push(`${r.name} ${label} (${bad.length} lines)`);
    // a figure that fell and stayed (the drop hidden by measuring against the last sample: one zone's riders left on a
    // long line, 98-99% below, T-2793): a small edit never takes FINAL_FALL off a line. (Real answers move a lot: a
    // second stop dropped on Hikari takes line 69's fares from $329M to $197M a day, the same before and after T-2793.
    // Lines under 1,000 riders are left out: NYC's line 103 steps from 394 to 266 on the F's train count.)
    const stayed = Object.keys(r.base).filter((id) => r.base[id][0] >= 1000 && r.base[id][k] > 0 && !((fin.lines[id]?.[k] ?? 0) >= (1 - FINAL_FALL) * r.base[id][k]));
    if (stayed.length) {
      console.log(`  ${label}: ended more than ${FINAL_FALL * 100}% below where it was: ${stayed.map((id) => `${id} ${+r.base[id][k].toPrecision(4)} -> ${+(fin.lines[id]?.[k] ?? 0).toPrecision(4)}`).join(', ')}`);
      failures.push(`${r.name} ${label} ended low (${stayed.length} lines)`);
    }
  }
}
