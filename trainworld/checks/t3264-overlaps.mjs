// T-3264: every pair of trains drawn through each other, whatever the cause (notes/T-3264.md, "A
// general probe"). Each frame places every train in view as drawn (`tw.tripAt` with the renderer's
// lane, band and passing shifts, its lag and its places; each car by `trainCarAt`) and projects its
// cars to the screen; two cars of different trains whose centre lines come closer than a quarter of
// their two drawn widths together (half a train, for two of one width) are through each other. Each
// pair is sorted by angle (same way, opposite ways, crossing), by whether the two are on one track
// piece (an edge) or different ones, by standing, and by whether either is on a band of shared track
// (T-1799). An event is one pair of trips, however many frames.
//
// Two games: the real New York save, its scenes as the notes' table; then a sandbox game with the
// timing layouts in one game, so the timing kinds show too: t1965's track (a local and an
// express on 8 km of double track, a station a km), t3202's 2 km of single track (two trains a level)
// and t2980's Y (two lines to a dead end with one platform). Per scene it prints the table's classes
// against BASELINE and fails a class more than a quarter (and 5 events) above it. From the maps root:
//   node trainworld/checks/t3264-overlaps.mjs [url] [real|layouts|both] [tag]
// SHOTS=n: pictures of each scene's n longest events into artifacts/trainworld/t3264/ (default 1).
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6131/';
const WHICH = process.argv[3] ?? 'both';
const TAG = process.argv[4] ?? 'run';
const SHOTS = Number(process.env.SHOTS ?? 1);
const OUT = join(ARTIFACTS, 't3264');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

// a sandbox game: t1965's track along LAT_F and t3202's single track along LAT_S in Brooklyn; t2980's Y
// where its check has it (level ground for M's platform)
const LNG0 = -73.985, KM = (lat) => 1 / (111.32 * Math.cos((lat * Math.PI) / 180)), KMLAT = 1 / 110.57;
const LAT_F = 40.652, LAT_S = 40.68, LAT_Y = 40.3, LNG_Y = -79.6;
const F_PTS = Array.from({ length: 9 }, (_, k) => [LNG0 + k * KM(LAT_F), LAT_F]);
const S_A = [LNG0, LAT_S], S_B = [LNG0 + 2 * KM(LAT_S), LAT_S];
const Y = { T: [LNG_Y, LAT_Y], J: [LNG_Y + 3 * KM(LAT_Y), LAT_Y], M: [LNG_Y + 6 * KM(LAT_Y), LAT_Y], X: [LNG_Y + 7 * KM(LAT_Y), LAT_Y], BR: [LNG_Y + 5 * KM(LAT_Y), LAT_Y + 1.5 * KMLAT] };

/** scenes: name, [lng, lat], zoom, speed index (60x, 240x, 960x), game minutes from 07:30 on a weekday */
const REAL = [
  ['midtown', [-73.985, 40.755], 14.5, 0, 20],
  ['midtown', [-73.985, 40.755], 16.5, 0, 10],
  ['midtown', [-73.985, 40.755], 17.5, 0, 10],
  ['brooklyn', [-73.95, 40.7], 12.5, 2, 60],
];
const LAYOUTS = [
  ['follow', F_PTS[4], 12.6, 2, 180],
  ['single', [LNG0 + KM(LAT_S), LAT_S], 14.6, 0, 30],
  ['y-end', [LNG_Y + KM(LAT_Y), LAT_Y], 14, 0, 60],
];

// Master c350115a on 2026-11-17 (the layouts' follow and single: dc1be574, after T-3526; the notes' table of
// 2026-11-12 within 4% of it): events
// by class, one track same way, one track opposite, other track alongside, crossing. Brooklyn at 960x
// counts more the more frames a game hour gets (219 here, 159 on 11-12: 921 opposite then).
const BASELINE = {
  'midtown-z14.5': { same: 0, opposite: 214, alongside: 432, crossing: 506 },
  'midtown-z16.5': { same: 0, opposite: 32, alongside: 46, crossing: 100 },
  'midtown-z17.5': { same: 0, opposite: 1, alongside: 5, crossing: 95 },
  'brooklyn-z12.5': { same: 0, opposite: 1215, alongside: 1843, crossing: 2138 },
  'follow-z12.6': { same: 0, opposite: 0, alongside: 0, crossing: 0 },
  'single-z14.6': { same: 0, opposite: 0, alongside: 0, crossing: 0 },
  'y-end-z14': { same: 0, opposite: 21, alongside: 0, crossing: 0 },
};

const chrome = await launchChrome({ prefix: 'tw-t3264-', size: [1400, 900] });
const errors = [];
const results = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);

  const install = () => page.evaluate(async () => {
    const net = await import('/src/map/network.ts');
    const { laneState } = await import('/src/map/renderer.ts');
    const tw = window.tw, r = tw.renderer, m = tw.map;
    // the train shader's width (renderer TRAIN_PX, narrowing to 3 m as double track parts, T-658)
    const TRAIN_PX = [[10, 3], [11, 5.25], [12, 6], [15, 9]];
    const interp = (z, W) => { if (z <= W[0][0]) return W[0][1]; for (let k = 1; k < W.length; k++) if (z <= W[k][0]) { const [z0, a] = W[k - 1], [z1, b] = W[k]; return a + (b - a) * (z - z0) / (z1 - z0); } return W[W.length - 1][1]; };
    const widthOf = (z, lane, tram) => {
      const ls = laneState(z), mPx = ls.gap / 4, l = ls.t * lane;
      const w0 = Math.max(interp(z, TRAIN_PX) * (tram ? 0.885 : 1), (tram ? 2.4 : 3) * mPx);
      return w0 + ((tram ? 2.4 : 3) * mPx - w0) * l;
    };
    /** a per-sample value of row `geo` at `d` m along it */
    const sampleAt = (n, v, geo, d, stride = 1, off = 0) => {
      if (!v) return null;
      const f0 = n.lineTable[geo * 4], cnt = n.lineTable[geo * 4 + 1], step = n.lineTable[geo * 4 + 3];
      if (cnt < 2) return v[f0 * stride + off];
      const x = Math.max(0, Math.min(cnt - 1, d / step)), i = Math.min(cnt - 2, Math.floor(x));
      return v[(f0 + i) * stride + off] + (v[(f0 + i + 1) * stride + off] - v[(f0 + i) * stride + off]) * (x - i);
    };
    const segD = (a, b, c, d) => {
      const cl = (p, q, r2) => { const dx = r2[0] - q[0], dy = r2[1] - q[1], L = dx * dx + dy * dy || 1e-9; const u = Math.max(0, Math.min(1, ((p[0] - q[0]) * dx + (p[1] - q[1]) * dy) / L)); return Math.hypot(q[0] + u * dx - p[0], q[1] + u * dy - p[1]); };
      const cross = (p, q, r2) => (q[0] - p[0]) * (r2[1] - p[1]) - (q[1] - p[1]) * (r2[0] - p[0]);
      if (Math.sign(cross(a, b, c)) !== Math.sign(cross(a, b, d)) && Math.sign(cross(c, d, a)) !== Math.sign(cross(c, d, b))) return 0;
      return Math.min(cl(a, c, d), cl(b, c, d), cl(c, a, b), cl(d, a, b));
    };
    window.__probeReset = () => { window.__probe = { pairFrames: 0, frames: 0, framesWith: 0, events: new Map(), kinds: {} }; };
    window.__probeReset();
    window.__probeFrame = () => {
      const n = r.net, t = r.guard.lastTime;
      if (!n || !(t > 0) || r.trainsHidden) return;
      const c = m.getCanvas(), Wd = c.clientWidth, Ht = c.clientHeight, z = m.getZoom();
      const P = window.__probe;
      P.frames++;
      const trains = [];
      for (let i = 0; i < r.tripCount; i++) {
        const prof = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i);
        const at = tw.tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, r.laneShiftM, r.bandShiftM, r.sampleBand, r.bandOn, r.passShiftM, r.tripBays ?? undefined, i * 4);
        if (!at) continue;
        const cn = m.project(tw.toLngLat(at.x, at.y));
        if (cn.x < -200 || cn.y < -200 || cn.x > Wd + 200 || cn.y > Ht + 200) continue;
        const raw = n.lineCars?.[at.line] ?? 8;
        if (raw < 1) continue; // buses run on roads, boats on water
        const tram = raw >= 1 && raw % 1 > 0.25;
        const cars = Math.max(1, Math.floor(raw));
        const segs = [];
        for (let k = 0; k < cars; k++) {
          const p = net.trainCarAt(n, at, k);
          const b = m.project(tw.toLngLat(...p.back)), f = m.project(tw.toLngLat(...p.front));
          // (back and front are along the drawn row's run 0: a train drawn as run 1 travels the other way)
          segs.push(at.gr === 0 ? [[b.x, b.y], [f.x, f.y]] : [[f.x, f.y], [b.x, b.y]]);
        }
        const d = at.gr === 0 ? at.s : n.lineTable[at.geo * 4 + 2] - at.s;
        // on a band of shared track: where its line is shifted
        const lane = sampleAt(n, n.sampleLane, at.geo, d) ?? 1;
        const w = widthOf(z, lane, tram);
        const banded = r.bandOn > 0 && Math.abs(sampleAt(n, r.sampleBand ?? n.sampleBand, at.geo, d, 2, 0) ?? 0) > 0.01;
        // the track piece (edge) under its middle, from the drawn row's path entries
        let edge = -1;
        if (n.pathTable && n.pathFrom) {
          const first = n.pathTable[at.geo * 2], cnt = n.pathTable[at.geo * 2 + 1];
          for (let k = first; k < first + cnt; k++) if (n.pathFrom[k] <= d && d <= n.pathFrom[k] + n.pathLen[k]) { edge = n.pathEntry[k] >> 1; break; }
        }
        const tau = t - dep, standing = tau > n.meta[prof * 4 + 2] - 1 || Math.abs(net.profileS(n, prof, tau + 1) - at.s) < 0.05;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const s of segs) for (const q of s) { x0 = Math.min(x0, q[0]); y0 = Math.min(y0, q[1]); x1 = Math.max(x1, q[0]); y1 = Math.max(y1, q[1]); }
        const inView = x1 >= 0 && y1 >= 0 && x0 <= Wd && y0 <= Ht;
        trains.push({ i, prof, line: at.line, run: at.run, edge, segs, w, banded, box: [x0 - w, y0 - w, x1 + w, y1 + w], standing, lag: r.tripLag(i), inView, ll: tw.toLngLat(at.x, at.y) });
      }
      let any = 0;
      for (let a = 0; a < trains.length; a++) for (let b = a + 1; b < trains.length; b++) {
        const A = trains[a], B = trains[b];
        if (!A.inView && !B.inView) continue;
        if (A.box[2] < B.box[0] || B.box[2] < A.box[0] || A.box[3] < B.box[1] || B.box[3] < A.box[1]) continue;
        const lim = (A.w + B.w) / 4;
        let hit = null;
        for (const s of A.segs) { for (const q of B.segs) { if (segD(s[0], s[1], q[0], q[1]) < lim) { hit = [s, q]; break; } } if (hit) break; }
        if (!hit) continue;
        const [s, q] = hit, ax = s[1][0] - s[0][0], ay = s[1][1] - s[0][1], bx = q[1][0] - q[0][0], by = q[1][1] - q[0][1];
        const cos = (ax * bx + ay * by) / ((Math.hypot(ax, ay) * Math.hypot(bx, by)) || 1);
        const angle = cos > 0.85 ? 'same' : cos < -0.85 ? 'opposite' : 'crossing';
        const stand = A.standing && B.standing ? 'both-standing' : A.standing || B.standing ? 'one-standing' : 'moving';
        const held = A.lag > 1 || B.lag > 1 ? 'lagged' : 'on-time';
        const track = A.edge >= 0 && A.edge === B.edge ? 'one-track' : 'other-track';
        const kind = `${angle}/${track}/${stand}/${A.line === B.line ? 'same-line' : 'two-lines'}/${held}/${A.banded || B.banded ? 'band' : 'plain'}`;
        P.pairFrames++;
        any++;
        const key = Math.min(A.i, B.i) + ',' + Math.max(A.i, B.i);
        const ev = P.events.get(key);
        if (ev) { ev.frames++; ev.kinds.add(kind); }
        else P.events.set(key, { frames: 1, kinds: new Set([kind]), t, ll: A.ll.map((v) => +v.toFixed(5)), lines: [A.line, B.line], runs: [A.run, B.run], lags: [Math.round(A.lag), Math.round(B.lag)], widths: [+A.w.toFixed(1), +B.w.toFixed(1)] });
        P.kinds[kind] = (P.kinds[kind] ?? 0) + 1;
      }
      if (any) P.framesWith++;
    };
  });

  /** one scene: the camera on it, the clock from 07:30 on a weekday at its speed, every frame probed */
  const runScene = async (game, [name, centre, zoom, speed, mins]) => {
    const tag = `${name}-z${zoom}`, file = `${TAG}-${game}-${tag}`;
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [centre, zoom]);
    await page.waitForTimeout(2500);
    const res = await page.evaluate(async ([SPEED, MINS]) => {
      window.__probeReset();
      const tw = window.tw, r = tw.renderer, c = tw.clock;
      const day = Math.floor(c.now() / 86400);
      c.setRunning(false);
      c.set((day + 1 + ((day + 1) % 7 >= 5 ? 2 : 0)) * 86400 + 7.5 * 3600);
      c.setSpeed(SPEED);
      c.setRunning(true);
      const t0 = c.now(), w0 = performance.now(), ms = [];
      let err = null;
      await new Promise((done) => {
        const step = () => {
          if (c.now() - t0 > MINS * 60 || performance.now() - w0 > 300000) return done();
          try { window.__probeFrame(); ms.push([r.guardMs, r.guard.guardedCount, r.guard.heldCount]); } catch (e) { err = String(e.stack ?? e); return done(); }
          requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      });
      c.setRunning(false);
      const P = window.__probe, evs = [...P.events.values()];
      const byKind = {};
      for (const e of evs) for (const k of e.kinds) byKind[k] = (byKind[k] ?? 0) + 1;
      const avg = (k) => ms.length ? +(ms.reduce((s, x) => s + x[k], 0) / ms.length).toFixed(3) : 0;
      const p95 = (k) => { const v = ms.map((x) => x[k]).sort((a, b) => a - b); return v.length ? +v[Math.floor(v.length * 0.95)].toFixed(3) : 0; };
      return { err, gameMin: +((c.now() - t0) / 60).toFixed(1), frames: P.frames, framesWith: P.framesWith, pairFrames: P.pairFrames, events: evs.length, eventsByKind: byKind, pairFramesByKind: P.kinds, guardMsAvg: avg(0), guardMsP95: p95(0), guardedAvg: avg(1), heldAvg: avg(2), trips: r.tripCount, longest: evs.sort((a, b) => b.frames - a.frames).slice(0, 12).map((e) => ({ ...e, kinds: [...e.kinds] })) };
    }, [speed, mins]);
    // the notes' classes: an event counts once in each class it was in
    const cls = { same: 0, opposite: 0, alongside: 0, crossing: 0, band: 0, standingBoth: 0, oppositeMoving: 0 };
    for (const [k, v] of Object.entries(res.eventsByKind)) {
      const [angle, track, stand, , , band] = k.split('/');
      const c = angle === 'crossing' ? 'crossing' : track === 'one-track' ? angle : 'alongside';
      cls[c] += v;
      if (band === 'band') cls.band += v;
      if (stand === 'both-standing') cls.standingBoth += v;
      // (one track, opposite ways, at least one moving; both standing is a place at a line's end)
      if (c === 'opposite' && stand !== 'both-standing') cls.oppositeMoving += v;
    }
    const out = { game, tag, scene: `${centre.join(',')},${zoom}`, speed, ...res, classes: cls };
    results.push(out);
    await writeFile(join(OUT, `overlaps-${file}.json`), JSON.stringify(out, null, 1));
    check(!res.err && res.frames > 50, `${game} ${tag}: probed ${res.frames} frames over ${res.gameMin} game min${res.err ? ': ' + res.err : ''}`);
    const base = BASELINE[tag];
    const line = `${game} ${tag}: ${res.framesWith} of ${res.frames} frames with any, ${res.events} events; one track same way ${cls.same}, opposite ${cls.opposite} (${cls.oppositeMoving} moving); other track alongside ${cls.alongside}; crossing ${cls.crossing} (on bands ${cls.band}, both standing ${cls.standingBoth}); guard ${res.guardMsAvg} ms avg, p95 ${res.guardMsP95}`;
    console.log(line);
    if (base) for (const k of ['same', 'opposite', 'alongside', 'crossing']) check(cls[k] <= base[k] * 1.25 + 5, `${tag} ${k}: ${cls[k]} events against ${base[k]} at the baseline`);
    for (const [k, e] of res.longest.slice(0, SHOTS).entries()) {
      await page.evaluate(([t, ll, z]) => { const tw = window.tw, c = tw.clock; c.setRunning(false); c.set(t); tw.map.jumpTo({ center: ll, zoom: Math.max(z, 15.5) }); }, [e.t, e.ll, zoom]);
      await page.waitForTimeout(1500);
      await page.evaluate(() => { const m = window.tw.map, c = m.getCenter(); m.jumpTo({ center: [c.lng + 1e-7, c.lat] }); });
      await page.waitForTimeout(800);
      await page.screenshot({ path: join(OUT, `overlaps-${file}-ev${k}.png`) });
    }
  };

  const scene = runScene;

  const load = async (save) => {
    await page.goto(`${url}?debug=1&new=1&paused=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
    await importSave(page, await readFile(save));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 240000 });
    // (positions do not depend on riders: a clock still held for demand after 6 minutes is let go)
    try { await page.waitForFunction(() => !window.tw.clock.held.peek(), undefined, { timeout: 360000 }); }
    catch { console.log('clock still held: released by hand'); await page.evaluate(() => window.tw.clock.setHeld(false)); }
    await install();
  };
  if (WHICH !== 'layouts') {
    await load(process.env.SAVE ?? REAL_SAVE);
    for (const s of REAL) await scene('real', s);
    // LONDON=<save>: the City at z14.5 too (the notes' "shared London" save, 2,155 rail trips an hour then)
    if (process.env.LONDON) {
      await load(process.env.LONDON);
      await scene('london', ['city', [-0.1, 51.513], 14.5, 0, 20]);
    }
  }

  if (WHICH !== 'real') {
    await page.goto(`${url}?debug=1&new=1&paused=1`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    await page.waitForTimeout(1000);
    const built = await page.evaluate(async ([F, SA, SB, Y]) => {
      const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
      const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
      const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
      const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: 0 }; };
      const log = [];
      const route = async (a, b, single = false, fromFree = false) => { log.push((await c.edit({ op: 'route', from: fromFree ? free(a) : { kind: 'node', node: near(a) }, to: free(b), pis: [], single }))?.ok); await pause(300); };
      const station = async (p, name = '') => { const r = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name }); log.push(r?.ok || `${name}: ${JSON.stringify(r?.issues ?? r)}`); await pause(200); };
      // t1965: 8 km of double track, a station a km, a local at every one and an express at the ends
      await route(F[0], F[1], false, true);
      for (let k = 2; k < F.length; k++) await route(F[k - 1], F[k]);
      for (const p of F) await station(p);
      // t3202: 2 km of single track, a station at each end
      await route(SA, SB, true, true);
      await station(SA, 'A');
      await station(SB, 'B');
      // t2980: the Y, T a dead end with one platform (no terminal), M a terminal
      await route(Y.T, Y.J, false, true);
      await route(Y.J, Y.M);
      await route(Y.M, Y.X);
      await route(Y.J, Y.BR);
      for (const [p, name] of [[Y.T, 'T'], [Y.M, 'M'], [Y.BR, 'Br']]) await station(p, name);
      log.push((await c.edit({ op: 'constructAll' }))?.ok);
      await pause(600);
      const local = await c.edit({ op: 'addLine', stops: F.map(near), name: 'Local', colour: '#0a7d3b' });
      // (an "Internal" issue here has been a WASM older than the app: npm run wasm --prefix app)
      log.push(local?.line !== undefined || 'addLine: ' + JSON.stringify(local).slice(0, 300));
      const express = await c.edit({ op: 'copyLine', line: local.line });
      await c.edit({ op: 'setStops', line: express.line, stops: [near(F[0]), near(F[F.length - 1])] });
      await c.edit({ op: 'lineLook', line: express.line, name: 'Express', colour: '#d9467a', letter: 'X' });
      await c.edit({ op: 'setTrainCount', line: local.line, trains: [1, 1, 1] });
      await c.edit({ op: 'setTrainCount', line: express.line, trains: [1, 1, 1] });
      const single = await c.edit({ op: 'addLine', stops: [near(SA), near(SB)], name: 'Single', colour: '#1f6fd1' });
      await c.edit({ op: 'setTrainCount', line: single.line, trains: [2, 2, 2] });
      const a = await c.edit({ op: 'addLine', stops: [near(Y.T), near(Y.BR)], name: 'A', colour: '#0a7d3b' });
      const b = await c.edit({ op: 'addLine', stops: [near(Y.T), near(Y.M)], name: 'B', colour: '#d9467a' });
      log.push((await c.edit({ op: 'stationUpgrade', nodes: [near(Y.M)], bit: 1, on: true }))?.ok);
      await c.edit({ op: 'setTrainCount', line: a.line, trains: [3, 3, 3] });
      await c.edit({ op: 'setTrainCount', line: b.line, trains: [3, 3, 3] });
      await pause(2500);
      return { log, lines: [local.line, express.line, single.line, a.line, b.line], tracks: tw.world.value.edges.map((e) => e.tracks) };
    }, [F_PTS, S_A, S_B, Y]);
    console.log('built', JSON.stringify({ ...built, tracks: undefined }));
    check(built.log.every(Boolean) && built.lines.every((l) => l !== undefined), 'the three layouts were built');
    await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
    await install();
    for (const s of LAYOUTS) await scene('layouts', s);
  }
  await writeFile(join(OUT, `overlaps-${TAG}.json`), JSON.stringify(results.map((r) => ({ ...r, longest: r.longest.slice(0, 3) })), null, 1));
  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
