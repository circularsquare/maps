// T-2018: an express behind a local never halts between stations unless the train in front of it
// is standing too. A new game with Anita's test track (8 km of straight double track, a station
// every km, a local at all of them and its copy at the ends only), the clock at its fastest for a
// few game hours. Every frame, each express is placed as drawn (the guard's lag included) and so is
// the nearest local ahead of it on its run; a frame where an express has not moved since the last
// one, away from its own stops, is a halt, and it is allowed only when that local has not moved
// either. The series goes to artifacts/trainworld/t2018/. From the maps root:
//   node trainworld/checks/t2018-trail.mjs [url] [locals] [expresses] [hours]
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2018');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6147/';
const LOCALS = Number(process.argv[3] ?? 1), EXPRESSES = Number(process.argv[4] ?? 1), HOURS = Number(process.argv[5] ?? 3);
const TAG = process.env.TAG ?? `${LOCALS}l${EXPRESSES}x`;
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const PTS = Array.from({ length: 9 }, (_, k) => [LNG0 + k * KM, LAT]);

const chrome = await launchChrome({ prefix: 'tw-t2018-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  const built = await page.evaluate(async ([PTS, LOCALS, EXPRESSES]) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < xy.length; k++) { await pause(250); await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false }); }
    await pause(400);
    for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(150); }
    await c.edit({ op: 'constructAll' });
    await pause(400);
    const stops = xy.map(near);
    const local = await c.edit({ op: 'addLine', stops, name: 'Local', colour: '#0a7d3b' });
    const express = await c.edit({ op: 'copyLine', line: local.line });
    await c.edit({ op: 'setStops', line: express.line, stops: [stops[0], stops[stops.length - 1]] });
    await c.edit({ op: 'lineLook', line: express.line, name: 'Express', colour: '#d9467a', letter: 'X' });
    await c.edit({ op: 'setTrainCount', line: local.line, trains: [LOCALS, LOCALS, LOCALS] });
    await c.edit({ op: 'setTrainCount', line: express.line, trains: [EXPRESSES, EXPRESSES, EXPRESSES] });
    await pause(1500);
    const st = tw.world.value.lineStats;
    return { local: local.line, express: express.line, rt: [st[String(local.line)]?.roundTripS, st[String(express.line)]?.roundTripS] };
  }, [PTS, LOCALS, EXPRESSES]);
  console.log('built', JSON.stringify(built));
  await page.evaluate(([lng, lat]) => { window.tw.map.jumpTo({ center: [lng, lat], zoom: 12.6 }); }, PTS[4]);
  await page.waitForTimeout(1500);

  const run = await page.evaluate(async ([HOURS, LOCAL, EXPRESS]) => {
    const tw = window.tw, r = tw.renderer, c = tw.clock;
    const day = Math.floor(c.now() / 86400);
    c.setRunning(false);
    c.set((day + 1 + ((day + 1) % 7 >= 5 ? 2 : 0)) * 86400 + 6.5 * 3600);
    c.setSpeed(2);
    c.setRunning(true);
    const t0 = c.now(), w0 = performance.now();
    // per frame: time, then per drawn train [trip, line, run, s, L, lag]
    const frames = [];
    // per profile: where its run starts and ends, then where its timetable stands between
    const stands = {};
    const standsOf = (n, p) => {
      if (stands[p]) return;
      const ph = n.phases, first = n.meta[p * 4], count = n.meta[p * 4 + 1];
      const l = (first + count - 1) * 4, end = ph[l + 3] < 0 ? ph[l + 1] + (ph[l + 2] * ph[l + 2]) / (-2 * ph[l + 3]) : ph[l + 1];
      const out = [ph[first * 4 + 1], end];
      for (let k = first + 1; k < first + count; k++) if (ph[k * 4 + 2] === 0 && ph[k * 4 + 3] === 0) out.push(ph[k * 4 + 1]);
      stands[p] = out;
    };
    let error = null;
    await new Promise((done) => {
      const step = () => {
        const t = r.guard.lastTime;
        if (c.now() - t0 > HOURS * 3600 || performance.now() - w0 > 240000) return done();
        if (!(t >= t0) || (frames.length && frames[frames.length - 1][0] === t)) return requestAnimationFrame(step);
        try {
          const n = r.net, trains = [];
          for (let i = 0; i < r.tripCount; i++) {
            const prof = r.trips[i * 2];
            const at = tw.tripAt(n, prof, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
            if (!at || (at.line !== LOCAL && at.line !== EXPRESS)) continue;
            standsOf(n, prof);
            trains.push([i, at.line, at.run, +at.s.toFixed(2), prof, +r.tripLag(i).toFixed(1)]);
          }
          frames.push([t, trains]);
        } catch (e) { error = String(e); return done(); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    c.setRunning(false);
    return { frames, stands, error };
  }, [HOURS, built.local, built.express]);
  check(!run.error, `the sampling ran${run.error ? ': ' + run.error : ''}`);
  const f = run.frames;
  const stopsOf = (p) => run.stands[p] ?? [];
  // stands between the stops in the express's timetable (its profiles' stands, but those at a
  // station where it stops)
  const exProfs = [...new Set(f.flatMap((fr) => fr[1].filter((x) => x[1] === built.express).map((x) => x[4])))];
  const timetableStands = exProfs.map((p) => stopsOf(p).slice(2).filter((s) => Math.abs(s - stopsOf(p)[0]) > 60 && Math.abs(s - stopsOf(p)[1]) > 60).length);
  // halts: an express that has not moved since the last frame, more than 60 m from its own stops
  let moving = 0, standing = 0, behindStanding = 0, halts = 0, lagMax = 0;
  const halted = [];
  for (let k = 1; k < f.length; k++) {
    const prev = new Map(f[k - 1][1].map((x) => [x[0], x]));
    for (const x of f[k][1]) {
      if (x[1] !== built.express) continue;
      lagMax = Math.max(lagMax, x[5]);
      const p = prev.get(x[0]);
      if (!p || p[2] !== x[2]) continue;
      if (stopsOf(x[4]).slice(0, 2).some((s) => Math.abs(s - x[3]) < 60)) continue;
      if (x[3] - p[3] > 0.5) { moving++; continue; }
      standing++;
      // the nearest local ahead on the same run, and whether it moved
      let ahead = null;
      for (const y of f[k][1]) if (y[1] === built.local && y[2] === x[2] && y[3] > x[3] && (!ahead || y[3] < ahead[3])) ahead = y;
      const ap = ahead && f[k - 1][1].find((y) => y[0] === ahead[0]);
      // (a local that has finished its run stands at the platform ahead as its next trip, the
      // other run, not yet moved from that run's start)
      const turning = f[k][1].some((y) => y[1] === built.local && y[2] !== x[2] && Math.abs(y[3] - (stopsOf(y[4])[0] ?? 0)) < 1 && f[k - 1][1].some((z) => z[0] === y[0] && z[3] === y[3]));
      if ((ahead && ap && ahead[3] - ap[3] <= 0.5) || (!ahead && turning)) behindStanding++;
      else { halts++; if (halted.length < 10) halted.push({ t: f[k][0], s: x[3], ahead: ahead ? [ahead[3], ap ? ap[3] : null] : null }); }
    }
  }
  const res = { tag: TAG, built, frames: f.length, gameHours: f.length ? +((f[f.length - 1][0] - f[0][0]) / 3600).toFixed(2) : 0, timetableStands, moving, standing, behindStanding, halts, lagMax, halted };
  console.log(JSON.stringify(res));
  await writeFile(join(OUT, `trail-${TAG}.json`), JSON.stringify({ ...res, stands: run.stands, series: f }));
  check(moving > 50, `${TAG}: expresses sampled moving between stations (${moving} frames)`);
  check(timetableStands.every((n) => n === 0), `${TAG}: the express's timetable never stands between its stops (${timetableStands.join(', ')} stands in its profiles)`);
  check(halts === 0, `${TAG}: no express halts between stations with the train ahead moving (${halts} of ${standing} standing frames; ${behindStanding} behind a standing local)`);
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
