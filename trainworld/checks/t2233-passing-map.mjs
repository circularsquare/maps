// T-2233: passing tracks on the map. A new sandbox game with T-2228's test line (180 km of high-speed
// track, 7 stations 30 km apart, a local at all of them and an express at A, D and G, both every 10
// minutes) with passing tracks at B to F, run 2 game hours at 60x with the whole line in view. Every
// frame each train is placed as drawn (its lag included). An express may pass a local only where the
// local stands on a passing station's outer track (one it stops at and the express runs through, as
// the guard's rule), and two trains on one run are never drawn overlapping anywhere else. Then zoom 15
// shots of E, light and dark, with an express passing a local standing there, in
// artifacts/trainworld/t2233/. From the maps root:
//   node trainworld/checks/t2233-passing-map.mjs [url] [hours]
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2233');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6187/';
const HOURS = Number(process.argv[3] ?? 2);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.3, LNG0 = -79.6, N = 7, STEP_KM = 30, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const PTS = Array.from({ length: N }, (_, k) => [LNG0 + k * STEP_KM * KM, LAT]);
/** as trainGuard.ts: a stopping train is on the outer track from its back this much of the lead short
 * of the platform to its front past the lead */
const PASS_IN = 0.2, PASS_LEAD_M = 100;

const chrome = await launchChrome({ prefix: 'tw-t2233-', size: [3000, 700] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const built = await page.evaluate(async ([PTS]) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = PTS.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    const log = [];
    log.push((await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: 0 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 }))?.ok);
    for (let k = 2; k < xy.length; k++) {
      await pause(250);
      log.push((await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: 0 }, pis: [], single: false, hsr: true, design: 320 }))?.ok);
    }
    await pause(500);
    for (const [k, p] of xy.entries()) { log.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 400, name: 'ABCDEFG'[k] }))?.ok); await pause(150); }
    log.push((await c.edit({ op: 'constructAll' }))?.ok);
    await pause(600);
    const stops = xy.map(near);
    const local = (await c.edit({ op: 'addLine', stops, name: 'Local', colour: '#0a7d3b', autoFast: true })).line;
    await c.edit({ op: 'lineFast', line: local, fast: true });
    const express = (await c.edit({ op: 'copyLine', line: local })).line;
    await c.edit({ op: 'setStops', line: express, stops: [stops[0], stops[3], stops[6]] });
    await c.edit({ op: 'lineFast', line: express, fast: true });
    await c.edit({ op: 'lineLook', line: express, name: 'Express', colour: '#d9467a', letter: 'X' });
    await c.edit({ op: 'setHeadway', line: local, seconds: [600, 600, 0] });
    await c.edit({ op: 'setHeadway', line: express, seconds: [600, 600, 0] });
    log.push((await c.edit({ op: 'stationUpgrade', nodes: stops.slice(1, 6), bit: 8, on: true }))?.ok);
    await pause(2500);
    const n = tw.renderer.net;
    return { log, stops, local, express, passing: n.passing ? Array.from(n.passing) : [], cars: [n.lineCars?.[local] ?? 8, n.lineCars?.[express] ?? 8] };
  }, [PTS]);
  console.log('built', JSON.stringify({ ...built, passing: built.passing.length / 5 }));
  check(built.log.every(Boolean), 'the test line was built, passing tracks at B to F');
  // the passing stations each line stops at and runs through, along run 0 (m), half the platform
  const stopsOf = new Map(), through = new Map();
  for (let k = 0; k + 5 <= built.passing.length; k += 5) {
    const [line, d, ph, node, stops] = built.passing.slice(k, k + 5);
    const m = stops ? stopsOf : through;
    if (!m.has(line)) m.set(line, []);
    m.get(line).push([d, ph, node]);
  }
  check((stopsOf.get(built.local)?.length ?? 0) === 5 && (through.get(built.express)?.length ?? 0) === 4, `the local stops at 5 passing stations, the express runs through 4 (${stopsOf.get(built.local)?.length ?? 0}, ${through.get(built.express)?.length ?? 0})`);

  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 10.05 }), PTS[3]);
  await page.waitForTimeout(1500);
  const run = await page.evaluate(async ([HOURS, LINES]) => {
    const tw = window.tw, r = tw.renderer, c = tw.clock;
    const day = Math.floor(c.now() / 86400);
    c.setRunning(false);
    c.set((day + 1 + ((day + 1) % 7 >= 5 ? 2 : 0)) * 86400 + 7 * 3600);
    c.setSpeed(0);
    c.setRunning(true);
    r.guard.resetCounts();
    const t0 = c.now(), w0 = performance.now();
    // per frame: time, then per drawn train [trip, line, run, s, profile, run length, lag]
    const frames = [];
    let error = null;
    await new Promise((done) => {
      const step = () => {
        const t = r.guard.lastTime;
        if (c.now() - t0 > HOURS * 3600 || performance.now() - w0 > 400000) return done();
        if (!(t >= t0) || (frames.length && frames[frames.length - 1][0] === t)) return requestAnimationFrame(step);
        try {
          const n = r.net, trains = [];
          for (let i = 0; i < r.tripCount; i++) {
            const prof = r.trips[i * 2];
            const at = tw.tripAt(n, prof, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
            if (!at || !LINES.includes(at.line)) continue;
            trains.push([i, at.line, at.run, +at.s.toFixed(1), prof, n.meta[prof * 4 + 3], +r.tripLag(i).toFixed(1)]);
          }
          frames.push([t, trains]);
        } catch (e) { error = String(e); return done(); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    c.setRunning(false);
    return { frames, error, passedBy: r.guard.passedBy, wall: performance.now() - w0 };
  }, [HOURS, [built.local, built.express]]);
  check(!run.error, `the sampling ran${run.error ? ': ' + run.error : ''}`);
  const f = run.frames;
  const halfOf = (line) => Math.min(Math.max(1, line === built.local ? built.cars[0] : built.cars[1]) * 20) / 2;
  // the passing station whose outer track a train is on, or null
  const outer = (x) => {
    const [, line, rn, s, , L] = x, h = halfOf(line);
    for (const [d, ph, node] of stopsOf.get(line) ?? []) {
      const at = rn === 0 ? d : L - d;
      if (s - h >= at - ph - PASS_IN * PASS_LEAD_M && s + h <= at + ph + PASS_LEAD_M) return node;
    }
    return null;
  };
  const besides = (a, b) => { const oa = outer(a), ob = outer(b); return (oa !== null && (through.get(b[1]) ?? []).some((p) => p[2] === oa)) || (ob !== null && (through.get(a[1]) ?? []).some((p) => p[2] === ob)); };
  // away from the run's ends, where trains turn (they stand there for their next trip)
  const midRun = (x) => x[3] > 1000 && x[3] < x[5] - 1000;
  let overlaps = 0, besideFrames = 0, passes = 0, passesBad = 0, lagMax = 0;
  const bad = [], where = {};
  const nodeName = new Map(built.stops.map((s, k) => [s, 'ABCDEFG'[k]]));
  for (let k = 0; k < f.length; k++) {
    const tr = f[k][1].filter(midRun);
    for (const x of tr) lagMax = Math.max(lagMax, x[6]);
    for (let a = 0; a < tr.length; a++) for (let b = a + 1; b < tr.length; b++) {
      const x = tr[a], y = tr[b];
      if (x[2] !== y[2]) continue;
      if (Math.abs(x[3] - y[3]) >= halfOf(x[1]) + halfOf(y[1])) continue;
      if (besides(x, y)) { besideFrames++; continue; }
      overlaps++;
      if (bad.length < 10) bad.push({ t: f[k][0], x, y });
    }
    if (!k) continue;
    // an express that was behind a local last frame and is ahead now: where the local is
    const prev = new Map(f[k - 1][1].map((x) => [x[0], x]));
    for (const x of tr) {
      if (x[1] !== built.express) continue;
      const px = prev.get(x[0]);
      if (!px || px[2] !== x[2]) continue;
      for (const y of tr) {
        if (y[1] !== built.local || y[2] !== x[2]) continue;
        const py = prev.get(y[0]);
        if (!py || py[2] !== y[2] || !(px[3] < py[3] && x[3] > y[3])) continue;
        passes++;
        const o = outer(y);
        const name = o === null ? 'none' : nodeName.get(o);
        where[name] = (where[name] ?? 0) + 1;
        if (o === null || !(through.get(built.express) ?? []).some((p) => p[2] === o)) { passesBad++; if (bad.length < 10) bad.push({ pass: true, t: f[k][0], x, y }); }
      }
    }
  }
  const res = { frames: f.length, gameHours: f.length ? +((f[f.length - 1][0] - f[0][0]) / 3600).toFixed(2) : 0, wallS: +(run.wall / 1000).toFixed(0), passes, where, passesBad, overlaps, besideFrames, guardBeside: run.passedBy, lagMax, bad };
  console.log(JSON.stringify(res));
  await writeFile(join(OUT, 'passing-map.json'), JSON.stringify(res));
  check(res.gameHours >= HOURS * 0.95, `${res.gameHours} game hours sampled over ${f.length} frames`);
  check(passes > 0, `expresses passed locals ${passes} times (${Object.entries(where).map(([k, v]) => `${k} ${v}`).join(', ')})`);
  check(passesBad === 0, `every pass was at a passing station the local stops at and the express runs through (${passesBad} elsewhere)`);
  check(overlaps === 0, `no two trains on a run drawn through each other (${overlaps} frame pairs; ${besideFrames} beside each other at passing stations)`);

  // zoom 15 at E with an express passing a local standing there: run on until one is
  const E = built.stops[4];
  const shotAt = await page.evaluate(async ([LOCAL, EXPRESS, eD]) => {
    const tw = window.tw, r = tw.renderer, c = tw.clock;
    c.setSpeed(0);
    c.setRunning(true);
    const w0 = performance.now();
    const found = await new Promise((done) => {
      const step = () => {
        if (performance.now() - w0 > 150000) return done(null);
        const n = r.net, t = r.guard.lastTime, at = [];
        for (let i = 0; i < r.tripCount; i++) {
          const prof = r.trips[i * 2];
          const a = tw.tripAt(n, prof, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
          if (a && (a.line === LOCAL || a.line === EXPRESS)) at.push(a);
        }
        const L = n.meta[0 * 4 + 3];
        const dOf = (a) => (a.run === 0 ? a.s : n.meta[(a.line * 6 + a.run * 3) * 4 + 3] - a.s);
        const loc = at.find((a) => a.line === LOCAL && Math.abs(dOf(a) - eD) < 30);
        const ex = loc && at.find((a) => a.line === EXPRESS && a.run === loc.run && Math.abs(dOf(a) - eD) < 250);
        if (loc && ex) { c.setRunning(false); return done({ x: loc.x, y: loc.y, t, L }); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    return found;
  }, [built.local, built.express, (stopsOf.get(built.local) ?? []).find((p) => p[2] === E)?.[0] ?? 0]);
  check(!!shotAt, `an express caught passing a local standing at E${shotAt ? ' at ' + new Date(shotAt.t * 1000).toISOString().slice(11, 19) : ''}`);
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 15 }), PTS[4]);
  await page.waitForTimeout(1500);
  for (const sh of ['light', 'dark']) {
    await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, sh);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: join(OUT, `passing-z15-${sh}.png`), clip: { x: 300, y: 250, width: 1100, height: 400 } });
  }
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
