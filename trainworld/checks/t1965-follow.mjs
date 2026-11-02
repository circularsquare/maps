// T-1965: trains on the same track never drawn through each other. A new game with Anita's kind of
// test track (a straight double-track line, a station every km, a local stopping at all of them and
// an express at the ends only), the clock at its fastest for a few game hours. Every frame counts
// the pairs of trains on the same track (an edge in one direction) whose bodies overlap, as the
// timetable has them (before) and as drawn (after, with the guard's lags), and checks that the GPU
// draws trains where the CPU says. Screenshots of one meeting before and after, and a short frame
// sequence, into artifacts/trainworld/t1965/. From the maps root:
//   node trainworld/checks/t1965-follow.mjs [url] [locals] [expresses] [hours]
// SAVE=<file> CENTER=lng,lat ZOOM=z TAG=name: a save instead, the trains in that view counted.
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, importSave } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1965');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6160/';
const LOCALS = Number(process.argv[3] ?? 1), EXPRESSES = Number(process.argv[4] ?? 1), HOURS = Number(process.argv[5] ?? 3);
const SAVE = process.env.SAVE ?? null;
const TAG = process.env.TAG ?? `${LOCALS}l${EXPRESSES}x`;
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const PTS = Array.from({ length: 9 }, (_, k) => [LNG0 + k * KM, LAT]);

const chrome = await launchChrome({ prefix: 'tw-t1965-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  if (SAVE) {
    await importSave(page, await readFile(SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
    await page.waitForFunction(() => !window.tw.clock.held.peek(), undefined, { timeout: 300000 });
  }
  const built = SAVE ? null : await page.evaluate(async ([PTS, LOCALS, EXPRESSES]) => {
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
    return { local: local.line, express: express.line, rt: [st[String(local.line)]?.roundTripS, st[String(express.line)]?.roundTripS], status: [st[String(local.line)]?.status, st[String(express.line)]?.status] };
  }, [PTS, LOCALS, EXPRESSES]);
  console.log('built', JSON.stringify(built));
  const mid = process.env.CENTER ? process.env.CENTER.split(',').map(Number) : PTS[4];
  // the whole line in view: the guard holds trains in or near the view only
  await page.evaluate(([lng, lat, z]) => { window.tw.map.jumpTo({ center: [lng, lat], zoom: z }); }, [...mid, Number(process.env.ZOOM ?? 12.6)]);
  await page.waitForTimeout(1500);

  // the overlap counter, in the page
  await page.evaluate((inView) => {
    const tw = window.tw, r = tw.renderer;
    /** pairs of trains whose bodies overlap on one track, by the timetable (lag 0) or as drawn */
    window.__overlaps = (t, drawn) => {
      const n = r.net, T = n.pathTable, E = n.pathEntry, PF = n.pathFrom, PL = n.pathLen;
      const byKey = new Map();
      for (let i = 0; i < r.tripCount; i++) {
        const prof = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + (drawn ? r.tripLag(i) : 0);
        const at = tw.tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
        if (!at) continue;
        if (inView) {
          const { x: px, y: py } = tw.map.project(tw.toLngLat(at.x, at.y));
          const c = tw.map.getCanvas();
          if (px < 0 || py < 0 || px > c.clientWidth || py > c.clientHeight) continue;
        }
        const line = at.line, L = n.meta[prof * 4 + 3], cars = Math.max(1, n.lineCars?.[line] ?? 8), len = Math.min(cars * 20, L);
        const s = Math.max(len / 2, Math.min(L - len / 2, at.s));
        const a = s - len / 2, b = s + len / 2;
        for (let k = T[line * 2]; k < T[line * 2] + T[line * 2 + 1]; k++) {
          const start = at.run === 0 ? PF[k] : L - PF[k] - PL[k];
          if (start > b || start + PL[k] < a) continue;
          const key = at.run === 0 ? E[k] : E[k] ^ 1;
          const v = byKey.get(key) ?? [];
          v.push([i, Math.max(a - start, 0), Math.min(b - start, PL[k])]);
          byKey.set(key, v);
        }
      }
      const pairs = new Set();
      let deepest = 0, deepI = -1;
      for (const v of byKey.values()) for (let x = 0; x < v.length; x++) for (let y = x + 1; y < v.length; y++) {
        if (v[x][0] === v[y][0]) continue;
        const o = Math.min(v[x][2], v[y][2]) - Math.max(v[x][1], v[y][1]);
        if (o > 0.5) pairs.add(Math.min(v[x][0], v[y][0]) + ',' + Math.max(v[x][0], v[y][0]));
        if (o > deepest) { deepest = o; deepI = v[x][0]; }
      }
      window.__depth = deepest;
      if (!drawn && deepI >= 0) {
        const at = tw.tripAt(n, r.trips[deepI * 2], r.trips[deepI * 2 + 1] + r.tripsEpoch, t, r.tripEnds ? r.tripEnds[deepI] + r.tripsEpoch : -Infinity);
        window.__where = at ? tw.toLngLat(at.x, at.y) : null;
      }
      if (drawn && pairs.size && (window.__detail ??= []).length < 6) {
        const one = (i) => { const prof = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch; const at = tw.tripAt(n, prof, dep + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity); return { i, prof, dep: dep - t, lag: r.tripLag(i), s: at?.s, end: r.tripEnds ? r.tripEnds[i] + r.tripsEpoch - t : null, dur: n.meta[prof * 4 + 2] }; };
        window.__detail.push({ t, pairs: [...pairs].map((p) => p.split(',').map(Number).map(one)) });
      }
      return pairs.size;
    };
  }, !!SAVE);

  // a weekday morning, the clock at its fastest, every frame counted
  const run = await page.evaluate(async ([HOURS]) => {
    const tw = window.tw, r = tw.renderer, c = tw.clock;
    const day = Math.floor(c.now() / 86400);
    c.setRunning(false);
    c.set((day + 1 + ((day + 1) % 7 >= 5 ? 2 : 0)) * 86400 + 6.5 * 3600);
    c.setSpeed(2);
    c.setRunning(true);
    const t0 = c.now(), frames = [], w0 = performance.now();
    let error = null;
    let worst = null;
    await new Promise((done) => {
      const step = () => {
        // the time of the frame last drawn, which the lags are for
        const t = r.guard.lastTime;
        if (c.now() - t0 > HOURS * 3600 || performance.now() - w0 > 240000) return done();
        if (!(t >= t0)) return requestAnimationFrame(step);
        let before, depth, where, after;
        try {
          before = window.__overlaps(t, false); depth = window.__depth; where = window.__where; after = window.__overlaps(t, true);
        } catch (e) { error = String(e); return done(); }
        frames.push([t, before, after, r.guard.heldCount, r.guard.guardedCount, r.guardMs]);
        // the deepest overlap away from the ends, for the pictures
        if (before > 0 && depth < 150 && (!worst || depth > worst[1])) worst = [t, depth, where];
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    c.setRunning(false);
    return { frames, worst, error };
  }, [HOURS]);
  const f = run.frames;
  check(!run.error, `the count ran${run.error ? ': ' + run.error : ''}`);
  const sum = (k) => f.reduce((s, x) => s + x[k], 0), withAny = (k) => f.filter((x) => x[k] > 0).length;
  const res = { tag: TAG, built, frames: f.length, gameHours: f.length ? +((f[f.length - 1][0] - f[0][0]) / 3600).toFixed(2) : 0, beforePairs: sum(1), beforeFrames: withAny(1), afterPairs: sum(2), afterFrames: withAny(2), heldFrames: withAny(3), guardMsAvg: +(sum(5) / f.length).toFixed(4), guardMsMax: +Math.max(...f.map((x) => x[5])).toFixed(3) };
  console.log(JSON.stringify(res));
  console.log(JSON.stringify(await page.evaluate(() => window.__detail ?? null)));
  await writeFile(join(OUT, `follow-${TAG}.json`), JSON.stringify({ ...res, frames: f }));
  check(res.beforeFrames > 0, `${TAG}: the timetable draws trains through each other (${res.beforeFrames} of ${f.length} frames)`);
  check(res.afterFrames === 0, `${TAG}: none drawn through each other (${res.afterFrames} frames, ${res.afterPairs} pairs)`);

  // the GPU draws the trains where the CPU has them, lags and all
  const gpu = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, n = r.net, t = tw.clock.now();
    const pos = r.probe(t);
    let worst = 0, n2 = 0;
    for (let i = 0; i < r.tripCount; i++) {
      const at = tw.tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
      if (!at) continue;
      n2++;
      worst = Math.max(worst, Math.hypot(pos[i * 2] - at.x, pos[i * 2 + 1] - at.y));
    }
    return { n: n2, worst };
  });
  check(gpu.n > 0 && gpu.worst < 1, `${TAG}: GPU places ${gpu.n} trains as the CPU does (worst ${gpu.worst.toFixed(3)} units)`);

  // pictures: a meeting, as drawn and with the guard off, and a short sequence of it
  if (run.worst) {
    const shot = async (name) => { await page.waitForTimeout(1300); await page.screenshot({ path: join(OUT, `${TAG}-${name}.png`) }); };
    // the camera on the meeting, close in
    const where = run.worst[2];
    if (where) await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 15.6 }), where);
    // from a game minute and a half before it, a frame every half a second of wall time at 60x
    await page.evaluate((t) => { const c = window.tw.clock; c.setRunning(false); c.set(t - 90); c.setSpeed(0); c.setRunning(true); }, run.worst[0]);
    for (let k = 0; k < 8; k++) {
      await page.waitForTimeout(500);
      await page.screenshot({ path: join(OUT, `${TAG}-seq-${k}.png`) });
    }
    await page.evaluate((t) => { const c = window.tw.clock; c.setRunning(false); c.set(t); c.setSpeed(0); c.setRunning(true); }, run.worst[0] - 30);
    await page.waitForTimeout(500);
    await page.evaluate(() => window.tw.clock.setRunning(false));
    await shot('after');
    console.log('at the shot', JSON.stringify(await page.evaluate(() => {
      const tw = window.tw, r = tw.renderer, n = r.net, t = tw.clock.now();
      const out = { t, worst: null, guardT: r.guard.lastTime, trains: [] };
      for (let i = 0; i < r.tripCount; i++) {
        const at = tw.tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
        if (at) out.trains.push({ i, line: at.line, run: at.run, s: Math.round(at.s), lag: Math.round(r.tripLag(i)), ll: tw.toLngLat(at.x, at.y).map((v) => +v.toFixed(5)) });
      }
      return out;
    })), 'worst', JSON.stringify(run.worst), 'where', JSON.stringify(where));
    // (a redraw needs the camera to move a hair while the clock is stopped)
    const nudge = () => page.evaluate(() => { const m = window.tw.map, c = m.getCenter(); m.jumpTo({ center: [c.lng + 1e-7, c.lat] }); });
    await page.evaluate(() => { window.tw.renderer.follow = false; });
    await nudge();
    await shot('before');
    await page.evaluate(() => { window.tw.renderer.follow = true; });
  }
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
