// T-3202: two trains on a two-station single-track line never pass through each other on screen, and
// meet side by side at a station. A new sandbox game gets 2 km of two-way single track between
// stations A and B and one line A - B with two trains at every level. The clock runs a weekday
// morning: at zoom 14.6 with the whole line in view, guard off (`renderer.guard.singles`, as before),
// then on at 60 and 600 game seconds a wall second; then at zoom 18 on B, where the train coming from
// A is far out of view. Every frame, the trains drawn on the stretch between the two stops (not
// standing at either) are looked at, and two of opposite directions there at once is a meeting on the
// single track; and two drawn bodies overlapping, as drawn once double track's tracks have parted
// (each its side of a station on single track, `singleSplits`, and its place at a line's end), are
// counted. Prints when each train leaves each stop as drawn; pictures of the two at B, light and dark,
// to artifacts/trainworld/t3202-meet-*.png. From the maps root:
//   node trainworld/checks/t3202-single-meet.mjs [url] [profile dir]
import { join } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6121/';
const profile = process.argv[3];
const OUT = join(ARTIFACTS, 't3202');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const A = [LNG0, LAT], B = [LNG0 + 2 * KM, LAT];
const chrome = await launchChrome({ prefix: 'tw-t3202-', size: [1300, 850], ...(profile ? { profile } : {}) });
const errors = [];
const out = { windows: [] };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  check(await page.evaluate(() => window.tw.renderer.guard.singles), 'the single-track wait is on by default');
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  out.built = await page.evaluate(async ([A, B]) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: 0 }; };
    const route = await c.edit({ op: 'route', from: free(A), to: free(B), pis: [], single: true });
    await pause(400);
    for (const p of [A, B]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    await c.edit({ op: 'constructAll' });
    await pause(500);
    const r = await c.edit({ op: 'addLine', stops: [near(A), near(B)], name: '', colour: '#d9467a' });
    if (r?.line !== undefined) await c.edit({ op: 'setTrainCount', line: r.line, trains: [2, 2, 2] });
    await pause(1500);
    return { route: route?.ok, line: r?.line, tracks: tw.world.value.edges.map((e) => e.tracks) };
  }, [A, B]);
  console.log('built', JSON.stringify(out.built));
  check(out.built.route && out.built.line !== undefined && out.built.tracks.every((t) => t === 1), 'a single-track line A - B with two trains');
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  await page.evaluate(() => {
    // per frame: each drawn train of the line, its run and offset along its run, whether it is on
    // the stretch between the stops (its middle more than 15 m from where it stands at either end),
    // and its metres right of run 0 once double track has parted (2 m on its own side)
    window.__look = () => {
      const tw = window.tw, r = tw.renderer, n = r.net, t = r.guard.lastTime;
      const laneAt = (line, d) => {
        const f0 = n.lineTable[line * 4], cnt = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
        if (!n.sampleLane) return 1;
        const x = Math.max(0, Math.min(cnt - 1, d / step)), i = Math.min(cnt - 2, Math.floor(x));
        return cnt < 2 ? n.sampleLane[f0] : n.sampleLane[f0 + i] + (n.sampleLane[f0 + i + 1] - n.sampleLane[f0 + i]) * (x - i);
      };
      const trains = [];
      for (let i = 0; i < r.tripCount; i++) {
        const p = r.trips[i * 2];
        const pos = tw.tripAt(n, p, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, 0, 0, undefined, 0, 0, r.tripBays ?? undefined, i * 4);
        if (!pos) continue;
        // where it stands at its first and last stop: its offset at the start and the end of its trip
        const dep = r.trips[i * 2 + 1] + r.tripsEpoch, dur = n.meta[p * 4 + 2];
        const s0 = tw.tripAt(n, p, dep, dep)?.s ?? 0, s1 = tw.tripAt(n, p, dep, dep + dur)?.s ?? n.meta[p * 4 + 3];
        const L = n.meta[p * 4 + 3], len = Math.min(Math.max(1, n.lineCars?.[Math.floor(p / 6)] ?? 8) * 20, L);
        const d = Math.max(pos.lo, Math.min(Math.max(pos.lo, pos.hi), pos.run === 0 ? pos.s : L - pos.s));
        const lat = (pos.run === 0 ? 2 : -2) * laneAt(pos.line, d) * pos.across(d);
        trains.push({ i, key: p + '|' + Math.round(dep), run: pos.run, s: pos.s, d, lat, len, lo: s0, hi: s1, on: pos.s > s0 + 15 && pos.s < s1 - 15, lag: r.tripLag(i) });
      }
      return { t, trains };
    };
  });
  const B_CENTER = [LNG0 + 2 * KM - 0.1 * KM, LAT];
  const windows = [[60, false, 14.6], [60, true, 14.6], [600, true, 14.6], [60, true, 18]];
  for (const [rate, on, zoom] of windows) {
    const secs = rate === 60 ? 60 : 20;
    await page.evaluate(([rate, on, zoom, center]) => {
      window.tw.renderer.guard.singles = on;
      const tw = window.tw, c = tw.clock, day = Math.floor(c.now() / 86400) + 1;
      tw.map.jumpTo({ center, zoom });
      c.setRunning(false);
      c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 9 * 3600);
      c.setMultiple(rate / 60);
      c.setRunning(true);
      window.__w = { frames: 0, meet: 0, meetT: [], overlap: 0, overlapT: [], beside: 0, leaves: [], maxLag: 0, prev: new Map() };
      window.__wOn = true;
      const f = () => {
        if (!window.__wOn) return;
        const w = window.__w, x = window.__look();
        w.frames++;
        if (w.frames % 60 === 1 && (w.samples ??= []).length < 40) w.samples.push([Math.round(x.t % 86400), x.trains.map((q) => [q.run, Math.round(q.d), +q.lat.toFixed(1), q.on ? 1 : 0])]);
        const on = x.trains.filter((q) => q.on);
        if (on.some((q) => q.run === 0) && on.some((q) => q.run === 1)) { w.meet++; if (w.meetT.length < 20) w.meetT.push(Math.round(x.t % 86400)); }
        // two drawn bodies over each other anywhere, or beside each other (along the same stretch,
        // 3 m or more apart across: on two tracks)
        let ov = 0, bs = 0;
        for (let a = 0; a < x.trains.length; a++) for (let b = a + 1; b < x.trains.length; b++) {
          const p = x.trains[a], q = x.trains[b];
          if (Math.abs(p.d - q.d) >= (p.len + q.len) / 2 - 1) continue;
          if (Math.abs(p.lat - q.lat) < 2.9) ov++;
          else bs++;
        }
        if (ov) { w.overlap++; if (w.overlapT.length < 20) w.overlapT.push(Math.round(x.t % 86400)); }
        if (bs) w.beside++;
        for (const q of x.trains) {
          w.maxLag = Math.max(w.maxLag, q.lag);
          const was = w.prev.get(q.key);
          if (was !== undefined && !was && q.on && w.leaves.length < 40) w.leaves.push([Math.round(x.t % 86400), q.run]);
          w.prev.set(q.key, q.on);
        }
        requestAnimationFrame(f);
      };
      requestAnimationFrame(f);
    }, [rate, on, zoom, zoom > 16 ? B_CENTER : [LNG0 + KM, LAT]]);
    await page.waitForTimeout(secs * 1000);
    const w = await page.evaluate(() => { window.__wOn = false; const w = window.__w; delete w.prev; return w; });
    const hm = (s) => `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    const tag = `rate ${rate}, zoom ${zoom}${on ? '' : ', wait off'}`;
    console.log(`${tag}: ${w.frames} frames, ${w.meet} with trains of both directions on the single track; ${w.overlap} with two trains drawn over each other, ${w.beside} with two side by side; longest lag ${w.maxLag.toFixed(0)} s`);
    console.log(`  leaving a stop (game time, direction): ${w.leaves.map(([t, r]) => `${hm(t)} ${r === 0 ? 'A>B' : 'B>A'}`).join(', ')}`);
    if (w.meet) console.log(`  meetings at ${w.meetT.map(hm).join(', ')}`);
    if (w.overlap) console.log(`  drawn over each other at ${w.overlapT.map(hm).join(', ')}`);
    out.windows.push({ rate, on, zoom, ...w });
    if (!on) continue;
    check(w.frames > 50, `${tag}: frames drawn (${w.frames})`);
    check(w.leaves.length >= 2, `${tag}: trains ran (${w.leaves.length} departures seen)`);
    check(w.meet === 0, `${tag}: never two trains heading towards each other on the single track (${w.meet} frames)`);
    check(w.overlap === 0, `${tag}: never two trains drawn over each other (${w.overlap} frames)`);
    check(w.beside > 0, `${tag}: two trains met side by side at a station (${w.beside} frames)`);
    // (the one waiting stands about the other's run, 93 s, and 5 s more)
    check(w.maxLag < 150, `${tag}: longest lag under 150 s (${w.maxLag.toFixed(0)} s)`);
  }
  // pictures: the moment both trains are at B, zoom 18
  const found = await page.evaluate(async ([center]) => {
    const tw = window.tw, c = tw.clock, day = Math.floor(c.now() / 86400) + 1;
    tw.map.jumpTo({ center, zoom: 18 });
    c.setRunning(false);
    c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 9 * 3600);
    c.setMultiple(1);
    c.setRunning(true);
    return await new Promise((done) => {
      const t0 = performance.now();
      const step = () => {
        if (performance.now() - t0 > 120000) return done(false);
        const x = window.__look();
        const atB = x.trains.filter((q) => !q.on && q.d > 1500);
        // (the one leaving a few seconds after the other is in: its first move)
        if (atB.length >= 2 && atB.some((q) => q.run === 0 && q.s >= q.hi - 2) && atB.some((q) => q.run === 1 && q.lag > 60)) { c.setRunning(false); return done(true); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, [B_CENTER]);
  check(found, 'both trains standing at B, side by side');
  await page.waitForTimeout(1500);
  for (const sh of ['light', 'dark']) {
    await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, sh);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: join(ARTIFACTS, `t3202-meet-${sh}.png`), clip: { x: 150, y: 225, width: 1000, height: 400 } });
  }
  await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, 'light');
  out.errors = errors;
  await writeFile(join(OUT, 'meet.json'), JSON.stringify(out, null, 1));
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
  check(errors.length === 0, 'no console errors');
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
