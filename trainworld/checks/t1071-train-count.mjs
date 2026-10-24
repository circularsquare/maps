// T-1071: changing a line's train count moves no train. The real save (T-007-real-nyc.save), its
// busiest line (not a loop, not on headways), at midday on a weekday; the count raised, lowered, raised
// by several at once and set back. Every frame the renderer draws, every train of every line is placed
// from the same buffers the GPU reads (`tripAt`); trains matched frame to frame may move no more than
// the top speed allows in the frame's game time (plus a metre), and a train may appear or vanish only
// at a terminal (just left it, or standing past its run). Where new buffers arrive (a train count's
// state, a new hour's trips), the old and the new must draw the same trains within a metre at the
// moment the new took effect, terminals again excepted.
//   node trainworld/checks/t1071-train-count.mjs [url]     (default http://localhost:6148/)
//   SPEED=2 (clock speed index, game/clock.ts SPEEDS; 2 is 960x, the fastest; -1 paused)
//   SHOTS=0 (no screenshots)  OP=count|headway (how the line's service is changed)
//   KEEP_FRAMES=1 (write frames-<op>-<speed>.json, tens of MB); FRAMES=<that file> analyses it again
// PLAYWRIGHT_MODULE as smoke.mjs. Output: ../artifacts/trainworld/t1071/ (check-<op>-<speed>.json).
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6148/';
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t1071/', import.meta.url));
const SHOTS = process.env.SHOTS !== '0';
const SPEED = Number(process.env.SPEED ?? 2);
const OP = process.env.OP ?? 'count';
const TAG = `${OP}-${SPEED}`;
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

/** The trains of `a` ([line, run, offset, at a terminal]) that no matching to `b` (same line,
 * offsets within `reach`; a change of run only where both stand at a terminal) can pair when it
 * must pair every one of `a` not at a terminal (`strict`: every one). Kuhn's augmenting paths. */
function unpaired(a, b, reach, strict) {
  const adj = a.map((p) => {
    const out = [];
    for (let j = 0; j < b.length; j++) {
      const q = b[j];
      if (q[0] !== p[0] || Math.abs(p[2] - q[2]) > reach) continue;
      if (p[1] !== q[1] && !(p[3] && q[3])) continue;
      out.push(j);
    }
    return out;
  });
  const to = new Array(b.length).fill(-1);
  const augment = (i, seen) => {
    for (const j of adj[i]) {
      if (seen[j]) continue;
      seen[j] = true;
      if (to[j] < 0 || augment(to[j], seen)) { to[j] = i; return true; }
    }
    return false;
  };
  const out = [];
  for (let i = 0; i < a.length; i++) if ((strict || !a[i][3]) && !augment(i, new Array(b.length).fill(false))) out.push(a[i]);
  return out;
}

/** Per line, frame to frame: every train where one of the frame before could have got to, else at
 * a terminal; and where new buffers arrived, the same trains at the same places at the frame
 * before's time from both (a metre). A matching pairing every required train on both sides exists
 * exactly when each side's can be paired on its own (Mendelsohn-Dulmage). */
function analyse(frames) {
  const VMAX = 160 / 3.6;
  const jumps = [], swaps = [];
  const byLine = (v) => { const m = new Map(); for (const x of v) { if (!m.has(x[0])) m.set(x[0], []); m.get(x[0]).push(x); } return m; };
  const note = (list, f, kind, x, extra) => list.push({ f, line: x[0], kind, run: x[1], d: Math.round(x[2]), term: x[3], t: Math.round(frames[f].t), got: frames[f].got, ...extra });
  for (let f = 1; f < frames.length; f++) {
    const a = frames[f - 1], b = frames[f];
    const dt = b.t - a.t;
    if (dt < 0 || dt > 120) continue;
    let e = f;
    while (e > 0 && !frames[e].mark) e--;
    const after = frames[e].mark ? `${f - e} frames after ${frames[e].mark}` : '';
    const reach = VMAX * dt + 1;
    const la = byLine(a.tr), lb = byLine(b.tr);
    for (const line of new Set([...la.keys(), ...lb.keys()])) {
      const pa = la.get(line) ?? [], pb = lb.get(line) ?? [];
      for (const x of unpaired(pa, pb, reach, false)) note(jumps, f, 'gone', x, { dt: +dt.toFixed(1), after });
      for (const x of unpaired(pb, pa, reach, false)) note(jumps, f, 'new', x, { dt: +dt.toFixed(1), after });
    }
    if (b.swap) {
      // only at a terminal may a train come or go: one standing there for a departure the change
      // gave another leaves service, and a new hour's trips hold only the trains drawn in it
      const lo = byLine(b.swap.old), ls = byLine(b.swap.new);
      for (const line of new Set([...lo.keys(), ...ls.keys()])) {
        const pa = lo.get(line) ?? [], ps = ls.get(line) ?? [];
        for (const x of unpaired(pa, ps, 1, false)) note(swaps, f, 'gone', x, { after });
        for (const x of unpaired(ps, pa, 1, false)) note(swaps, f, 'new', x, { after });
      }
    }
  }
  return { jumps, swaps };
}

// FRAMES=<frames-*.json>: analyse a recorded run again, no browser
if (process.env.FRAMES) {
  const r = analyse(JSON.parse(await readFile(process.env.FRAMES, 'utf8')));
  for (const j of r.jumps) console.log('jump', JSON.stringify(j));
  for (const j of r.swaps) console.log('swap', JSON.stringify(j));
  process.exit(0);
}

await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1071-', size: [1200, 800] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !/openfreemap|AJAXError|favicon|\/fonts\//.test(m.text())) errors.push(m.text()); });
const out = { base: BASE, speed: SPEED, op: OP, errors };
let failed = false;
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1000);
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  log('loaded');
  await demandFinal(page);
  log('demand final');
  const pick = await page.evaluate(() => {
    const tw = window.tw, w = tw.world.value;
    const lines = w.save.lines.filter((l) => { const s = w.lineStats[l.id]; return s && s.status === 'running' && !s.autoS && !s.loop && l.stops.length > 2; });
    lines.sort((a, b) => w.lineStats[b.id].trainsNeeded[1] - w.lineStats[a.id].trainsNeeded[1]);
    const l = lines[0], st = w.lineStats[l.id];
    tw.clock.setRunning(false);
    const day = Math.floor(tw.clock.now() / 86400);
    tw.clock.set((day + ((day % 7) >= 5 ? 2 : 0)) * 86400 + 11.25 * 3600);
    const sts = l.stops.map((s) => w.save.stations.find((x) => x.id === s)).filter(Boolean);
    const lng = sts.map((s) => s.lng), lat = sts.map((s) => s.lat);
    void tw.map.fitBounds([[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]], { padding: 60, duration: 0 });
    return { id: l.num, name: l.name, trains: st.trainsNeeded, roundTrip: st.roundTripByLevel, km: st.km };
  });
  out.line = pick;
  log('line', JSON.stringify(pick));
  // a sampler on every draw: every train, from the buffers the GPU reads, as [line, run, offset along
  // run 0]; a trip's profile is (line * 2 + run) * 3 + level
  await page.evaluate(async () => {
    const tw = window.tw, r = tw.renderer;
    const nw = await import('/src/map/network.ts');
    window.__frames = [];
    window.__mark = '';
    window.__on = false;
    let last = null, lastT = NaN;
    // the buffers the renderer draws from now (the network copied, as a train count replaces its
    // run profiles in place)
    const held = () => ({ n: { ...r.net }, trips: r.trips, ends: r.tripEnds, e0: r.tripsEpoch });
    // every train drawn at `time` from buffers `b`: [line, run, offset along run 0, at a terminal
    // (left it within `fresh` s, or standing past its run's end)]
    const placesAt = (b, time, fresh) => {
      const { n, trips, ends, e0 } = b, tr = [];
      for (let i = 0; i < trips.length / 2; i++) {
        const p = trips[i * 2], dep = trips[i * 2 + 1] + e0;
        const at = nw.tripAt(n, p, dep, time, ends ? ends[i] + e0 : -Infinity);
        if (!at) continue;
        const len = n.meta[p * 4 + 3], tau = time - dep;
        const term = tau <= fresh + 1 || Math.abs(at.s - nw.profileS(n, p, 1e7)) < 0.5;
        tr.push([Math.floor(p / 6), at.run, at.run === 0 ? at.s : len - at.s, term ? 1 : 0]);
      }
      return tr;
    };
    const draw = r.draw.bind(r);
    r.draw = (m, time, ...rest) => {
      const res = draw(m, time, ...rest);
      const n = r.net, trips = r.trips;
      if (!window.__on || !n || !trips) return res;
      const dt = Number.isFinite(lastT) ? time - lastT : 0;
      const now = held();
      const tr = placesAt(now, time, dt);
      // what arrived since the last frame: new trips (T), new run profiles (P), a new hour (E);
      // with them, the trains the old and the new buffers draw when the new took effect (a train
      // count's moment, else the last frame's time)
      const got = last ? (trips !== last.trips ? 'T' : '') + (n.phases !== last.n.phases ? 'P' : '') + (r.tripsEpoch !== last.e0 ? 'E' : '') : '';
      let swap = null;
      if (got) {
        const at = r.stagedAt > lastT && r.stagedAt <= time ? r.stagedAt : lastT;
        swap = { at, old: placesAt(last, at, dt), new: placesAt(now, at, dt) };
      }
      last = now;
      lastT = time;
      window.__frames.push({ t: time, mark: window.__mark, got, tr, swap });
      window.__mark = '';
      return res;
    };
  });
  const edit = (trains) => page.evaluate(async ([id, trains, op, rt]) => {
    window.__mark = 'edit ' + trains.join('/');
    const a = await window.tw.client.edit(op === 'headway'
      ? { op: 'setHeadway', line: id, seconds: trains.map((n, i) => (n ? Math.round(rt[i] / n) : 0)) }
      : { op: 'setTrainCount', line: id, trains });
    return a.ok;
  }, [pick.id, trains, OP, pick.roundTrip]);
  const t = pick.trains;
  await page.evaluate((speed) => { const c = window.tw.clock; if (speed >= 0) { c.setSpeed(speed); c.setRunning(true); } }, SPEED);
  // the trips for the hour set above arrive during the warm-up
  await page.waitForTimeout(3000);
  await page.evaluate(() => { window.__on = true; });
  await page.waitForTimeout(500);
  const steps = [[t[0], t[1] + 4, t[2]], [t[0], Math.max(1, t[1] - 3), t[2]], [t[0], t[1] + 8, t[2]], t];
  out.edits = [];
  let k = 0;
  for (const s of steps) {
    if (SHOTS) await page.screenshot({ path: join(OUT, `${TAG}-edit${k}-a-before.png`) });
    const ok = await edit(s);
    out.edits.push({ trains: s, ok });
    log('edit', s.join('/'), ok);
    if (SHOTS) {
      await page.waitForTimeout(100);
      await page.screenshot({ path: join(OUT, `${TAG}-edit${k}-b-after.png`) });
    }
    await page.waitForTimeout(SPEED >= 2 ? 5000 : 3000);
    k++;
  }
  await page.evaluate(() => { window.__on = false; window.tw.clock.setRunning(false); });
  const frames = await page.evaluate(() => window.__frames);
  out.frameCount = frames.length;
  out.trainsDrawn = [Math.min(...frames.map((f) => f.tr.length)), Math.max(...frames.map((f) => f.tr.length))];
  out.lineTrains = frames.filter((_, i) => i % 40 === 0).map((f) => [Math.round(f.t), f.tr.filter((x) => x[0] === pick.id).length]);
  out.editFrames = frames.map((f, i) => [i, f.mark]).filter((x) => x[1]);
  out.editMs = await page.evaluate(() => window.tw.client.editLog.slice(-4).map((e) => Math.round(e.worker?.total ?? -1)));
  log('edit ms in the worker', out.editMs.join(' '));
  if (process.env.KEEP_FRAMES) await writeFile(join(OUT, `frames-${TAG}.json`), JSON.stringify(frames));
  const r = analyse(frames);
  out.jumps = r.jumps.length;
  out.jumpSample = r.jumps.slice(0, 60);
  out.swaps = r.swaps.length;
  out.swapSample = r.swaps.slice(0, 60);
  out.swapFrames = frames.filter((f) => f.swap).length;
  failed = r.jumps.length > 0 || r.swaps.length > 0 || out.edits.some((e) => !e.ok);
  log('frames', frames.length, 'trains', out.trainsDrawn.join('-'), 'jumps', r.jumps.length, 'buffer swaps', out.swapFrames, 'moved by a swap', r.swaps.length);
  for (const j of r.jumps.slice(0, 10)) log('jump', JSON.stringify(j));
  for (const j of r.swaps.slice(0, 10)) log('swap', JSON.stringify(j));
} catch (e) {
  failed = true;
  out.error = String(e?.stack ?? e);
  log('error', out.error);
} finally {
  await close();
}
await writeFile(join(OUT, `check-${TAG}.json`), JSON.stringify(out, null, 1));
if (errors.length) { failed = true; log('page errors', errors.slice(0, 5)); }
console.log(`${failed ? 'FAIL' : 'PASS'}  t1071 ${OP} changes at speed ${SPEED} move no train`);
process.exit(failed ? 1 : 0);
