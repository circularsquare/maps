// T-2980: trains at a line's ends take a place there (notes/T-2974.md 5). A new sandbox game with
// T-2960's Y: a 3 km trunk from a dead end T to a junction J, line A on to Br, line B on to M, a
// through station 3 km past J with track running 1 km on and a terminal (sidings) there. Both lines
// run 3 trains (COUNT=n), then a 5 minute headway; each for an hour of game time at 60x with T in view. Every
// frame each train is placed as drawn (`tw.tripAt` with its lag and places). Counted: trains standing
// at T at once and how many of those pairs are on one platform; arrivals at T held on the way in
// (their lag grew in the last 600 m); trains drawn in M's sidings. With `?perfOff=bays` (the second
// argument `before`) the old drawing: every train at the end of its arrival track. Then zoom 18 shots
// of T with two trains standing and of M with one in a siding, light and dark, in
// artifacts/trainworld/t2980-*.png. From the maps root:
//   node trainworld/checks/t2980-terminus.mjs [url] [before]
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6410/';
const BEFORE = process.argv[3] === 'before';
/** trains each line runs first (`COUNT`); 4 each fills the flat junction, so trains also stand at T
 * for a path out, holding a platform */
const COUNT = Number(process.env.COUNT ?? 3);
await mkdir(ARTIFACTS, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.3, LNG0 = -79.6, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180)), KMLAT = 1 / 110.57;
const T = [LNG0, LAT], J = [LNG0 + 3 * KM, LAT], M = [LNG0 + 6 * KM, LAT], X = [LNG0 + 7 * KM, LAT], BR = [LNG0 + 5 * KM, LAT + 1.5 * KMLAT];

const chrome = await launchChrome({ prefix: 'tw-t2980-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1${BEFORE ? '&perfOff=bays' : ''}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const built = await page.evaluate(async ([T, J, M, X, BR, COUNT]) => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const log = [];
    const route = async (a, b, free) => {
      const [bx, by] = xy(b);
      const from = free ? { kind: 'free', x: xy(a)[0], y: xy(a)[1], level: 0 } : { kind: 'node', node: near(a) };
      log.push((await c.edit({ op: 'route', from, to: { kind: 'free', x: bx, y: by, level: 0 }, pis: [], single: false }))?.ok);
      await pause(300);
    };
    await route(T, J, true);
    await route(J, M);
    await route(M, X);
    await route(J, BR);
    for (const [p, name] of [[T, 'T'], [M, 'M'], [BR, 'Br']]) { log.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name }))?.ok); await pause(200); }
    log.push((await c.edit({ op: 'constructAll' }))?.ok);
    await pause(600);
    const a = (await c.edit({ op: 'addLine', stops: [near(T), near(BR)], name: 'A', colour: '#0a7d3b' })).line;
    const b = (await c.edit({ op: 'addLine', stops: [near(T), near(M)], name: 'B', colour: '#d9467a' })).line;
    log.push((await c.edit({ op: 'stationUpgrade', nodes: [near(M)], bit: 1, on: true }))?.ok);
    await c.edit({ op: 'setTrainCount', line: a, trains: [COUNT, COUNT, COUNT] });
    await c.edit({ op: 'setTrainCount', line: b, trains: [COUNT, COUNT, COUNT] });
    await pause(2500);
    const info = tw.world.value.save.lines.map((l) => JSON.stringify(l).slice(0, 400));
    return { log, a, b, t: near(T), m: near(M), info };
  }, [T, J, M, X, BR, COUNT]);
  console.log('built', JSON.stringify(built));
  check(built.log.every(Boolean), 'the Y was built, with a terminal at M');

  /** an hour of game time from 07:00 on a weekday: per frame, each train of A and B as drawn */
  const sample = (hours, from) => page.evaluate(async ([LINES, hours, from]) => {
    const tw = window.tw, r = tw.renderer, c = tw.clock;
    c.setRunning(false);
    const day = Math.floor(c.now() / 86400) + 1;
    c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + from * 3600);
    c.setSpeed(0);
    c.setRunning(true);
    // when a trip reaches its last stop: its last phase's stop (network.ts arriveS)
    const arrive = (n, p) => { const k = (n.meta[p * 4] + n.meta[p * 4 + 1] - 1) * 4; return n.phases[k + 3] < 0 ? n.phases[k] + n.phases[k + 2] / -n.phases[k + 3] : n.phases[k]; };
    const t0 = c.now(), w0 = performance.now(), frames = [];
    r.guard.placeLog = [];
    let error = null;
    await new Promise((done) => {
      const step = () => {
        const t = r.guard.lastTime;
        if (c.now() - t0 > hours * 3600 || performance.now() - w0 > 200000) return done();
        if (!(t >= t0) || (frames.length && frames[frames.length - 1][0] === t)) return requestAnimationFrame(step);
        try {
          const n = r.net, out = [];
          for (let i = 0; i < r.tripCount; i++) {
            const prof = r.trips[i * 2], lag = r.tripLag(i), dep = r.trips[i * 2 + 1] + r.tripsEpoch + lag;
            const at = tw.tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, 0, 0, undefined, 0, 0, r.tripBays ?? undefined, i * 4);
            if (!at || !LINES.includes(at.line)) continue;
            const L = n.meta[prof * 4 + 3], d = at.run === 0 ? at.s : L - at.s;
            // id, run, m to its run's end, standing (past its run), across at its middle, m into a siding, lag
            out.push([`${prof}|${(r.trips[i * 2 + 1] + r.tripsEpoch).toFixed(1)}`, at.run, +(L - at.s).toFixed(1), t - dep > arrive(n, prof) ? 1 : 0, +at.across(d + at.ex).toFixed(2), +Math.abs(at.ex).toFixed(1), +lag.toFixed(1)]);
          }
          frames.push([t, out]);
        } catch (e) { error = String(e.stack ?? e); return done(); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    c.setRunning(false);
    return { frames, error, placeLog: r.guard.placeLog, tripCount: r.tripCount, bays: r.tripBays?.length ?? -1, ends: r.tripEnds?.length ?? -1, epoch: r.tripsEpoch, now: c.now() };
  }, [[built.a, built.b], hours, from]);

  const measure = (run, tag) => {
    check(!run.error, `${tag}: the sampling ran${run.error ? ': ' + run.error : ''}`);
    const f = run.frames;
    let pairs = 0, onePlace = 0, sidingFrames = 0;
    const lag600 = new Map(), arrivedLag = new Map();
    const heldAt = new Map();
    for (const [, tr] of f) {
      // run 1 ends at T on both lines
      const atT = tr.filter((x) => x[1] === 1 && x[3]);
      for (let a = 0; a < atT.length; a++) for (let b = a + 1; b < atT.length; b++) {
        pairs++;
        if (Math.sign(atT[a][4]) === Math.sign(atT[b][4])) onePlace++;
      }
      for (const x of tr) {
        if (x[1] === 0 && x[5] > 1) sidingFrames++;
        if (x[1] !== 1) continue;
        if (!x[3] && x[2] <= 600 && !lag600.has(x[0])) lag600.set(x[0], x[6]);
        if (x[3] && !arrivedLag.has(x[0])) arrivedLag.set(x[0], x[6]);
        if (!x[3] && lag600.has(x[0]) && x[6] > lag600.get(x[0]) + 5 && !heldAt.has(x[0])) heldAt.set(x[0], x[2]);
      }
    }
    let arrivals = 0, held = 0, heldS = 0;
    for (const [id, l1] of arrivedLag) {
      const l0 = lag600.get(id);
      if (l0 === undefined) continue;
      arrivals++;
      if (l1 - l0 > 5) { held++; heldS += l1 - l0; }
    }
    const res = { tag, trains: f.reduce((a, x) => a + x[1].length, 0), tripCount: run.tripCount, bays: run.bays, ends: run.ends, epoch: run.epoch, now: run.now, frames: f.length, gameH: f.length ? +((f[f.length - 1][0] - f[0][0]) / 3600).toFixed(2) : 0, pairsAtT: pairs, onOnePlace: onePlace, arrivals, held, heldMeanS: held ? +(heldS / held).toFixed(0) : 0, sidingFrames };
    console.log(JSON.stringify(res));
    if (process.env.PLACE_LOG) console.log(JSON.stringify(run.placeLog?.slice(0, 12)));
    if (process.env.PLACE_LOG) console.log('held at m to the end', JSON.stringify([...heldAt.values()]));
    return res;
  };
  // T in view, close enough for the guard to hold trains there
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 14 }), [T[0] + 1 * KM, T[1]]);
  await page.waitForTimeout(1500);
  const count = measure(await sample(Number(process.env.SAMPLE_H ?? 1), 9), `count ${COUNT}+${COUNT}`);
  await page.evaluate(async ([a, b]) => {
    for (const l of [a, b]) await window.tw.client.edit({ op: 'setHeadway', line: l, seconds: [300, 300, 300] });
    await new Promise((d) => setTimeout(d, 2500));
  }, [built.a, built.b]);
  const hw = measure(await sample(1, 13), 'headway 5 min');
  await writeFile(join(ARTIFACTS, `t2980-terminus${BEFORE ? '-before' : ''}.json`), JSON.stringify({ count, hw }));
  for (const r of [count, hw]) {
    check(r.gameH >= 0.95, `${r.tag}: ${r.gameH} game hours over ${r.frames} frames`);
    if (!BEFORE) {
      check(r.pairsAtT > 0, `${r.tag}: two trains stood at T at once in ${r.pairsAtT} frame pairs`);
      check(r.onOnePlace === 0, `${r.tag}: none of them on one platform (${r.onOnePlace})`);
      // (a few seconds where a train leaving holds its platform a little past the timetable's pick)
      check(r.held <= Math.ceil(0.25 * r.arrivals) || r.heldMeanS <= 20, `${r.tag}: arrivals at T held on the way in: ${r.held} of ${r.arrivals} (mean ${r.heldMeanS} s)`);
      check(r.sidingFrames > 0, `${r.tag}: trains drawn in M's sidings (${r.sidingFrames} frames)`);
    } else console.log(`(before) ${r.tag}: ${r.onOnePlace} pairs on one place, ${r.held} of ${r.arrivals} arrivals held, mean ${r.heldMeanS} s`);
  }

  // shots: T with two trains standing, M with a train in a siding
  const shoot = async (want, lngLat, name) => {
    const found = await page.evaluate(async ([LINES, want]) => {
      const tw = window.tw, r = tw.renderer, c = tw.clock;
      c.setSpeed(0);
      c.setRunning(true);
      const w0 = performance.now();
      const arrive = (n, p) => { const k = (n.meta[p * 4] + n.meta[p * 4 + 1] - 1) * 4; return n.phases[k + 3] < 0 ? n.phases[k] + n.phases[k + 2] / -n.phases[k + 3] : n.phases[k]; };
      return await new Promise((done) => {
        const step = () => {
          if (performance.now() - w0 > 120000) { c.setRunning(false); return done(false); }
          const n = r.net, t = r.guard.lastTime;
          let standT = 0, siding = 0;
          for (let i = 0; i < r.tripCount; i++) {
            const prof = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i);
            const at = tw.tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, 0, 0, undefined, 0, 0, r.tripBays ?? undefined, i * 4);
            if (!at || !LINES.includes(at.line)) continue;
            if (at.run === 1 && t - dep > arrive(n, prof)) standT++;
            if (Math.abs(at.ex) > 150) siding++;
          }
          if (want === 'T' ? standT >= 2 : siding >= 1) { c.setRunning(false); return done(true); }
          requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
    }, [[built.a, built.b], want]);
    check(found, `${name}: the moment came`);
    await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 18 }), lngLat);
    await page.waitForTimeout(1500);
    for (const sh of ['light', 'dark']) {
      await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, sh);
      await page.waitForTimeout(1200);
      await page.screenshot({ path: join(ARTIFACTS, `t2980-${name}${BEFORE ? '-before' : ''}-${sh}.png`), clip: { x: 200, y: 250, width: 1000, height: 400 } });
    }
    await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, 'light');
    await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 14 }), [T[0] + 1 * KM, T[1]]);
  };
  // T's platform runs 200 m back from its end; M's sidings 100-260 m past its middle towards X
  if (!BEFORE) {
    await shoot('T', [T[0] + 0.15 * KM, T[1]], 'dead-end');
    await shoot('M', [M[0] + 0.15 * KM, M[1]], 'terminal');
  }
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
