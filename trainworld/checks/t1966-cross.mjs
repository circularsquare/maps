// T-1966: trains wait for each other at flat crossings and crossing junction moves, zoomed in to the
// renderer's CONFLICT_ZOOM. A save (the real one by default) on a weekday morning, the clock at 1x
// (real time), 60x and 960x, the camera still
// on each scene, a window with the waits on and one off: per frame the trains standing at a
// conflict and the longest wait so far, the waits ended and the longest, trains let over at the
// cap, the guard's ms, and the pairs of trains on crossing routes both with their bodies over a
// conflict point (no margins) in view. From the maps root:
//   node trainworld/checks/t1966-cross.mjs [url] [save] [scenes json]
// scenes: [[lng, lat, zoom], ...]. Writes artifacts/trainworld/t1966/cross-<save>.json.
// save `test`: t1966-layout.mjs's Y junction and flat crossing in a new game, the camera on them.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';
import { buildTest, TEST_SCENE } from './t1966-layout.mjs';

const OUT = join(ARTIFACTS, 't1966');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6157/';
const save = process.argv[3] ?? REAL_SAVE;
// the real save's busiest flat crossings: Midtown, Lower Manhattan, Long Island City (zoomed out past
// CONFLICT_ZOOM no one waits: pass e.g. [[-73.979, 40.755, 12.5]] to see)
const SCENES = process.argv[4] ? JSON.parse(process.argv[4]) : save === 'test' ? [TEST_SCENE] : [[-73.979, 40.755, 14], [-73.999, 40.725, 14.5], [-73.918, 40.748, 14]];
// speed (game s a wall s), wall seconds a window
const SPEEDS = [[1, 40], [60, 20], [960, 20]];
const chrome = await launchChrome({ prefix: 'tw-t1966-', size: [1400, 900] });
const errors = [];
const out = { save, scenes: [] };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  if (save === 'test') out.built = await buildTest(page);
  else await importSave(page, await readFile(save));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
  await page.waitForFunction(() => !window.tw.clock.held.peek() && !window.tw.clock.capped.peek(), undefined, { timeout: 900000, polling: 1000 });
  out.conflicts = await page.evaluate(() => { const g = window.tw.renderer.guard; return { crossings: g.nCross, moves: g.moveRes.length, pairs: g.movePairs.size / 2, guardedLines: g.guardedAll.reduce((a, x) => a + x, 0), trackLines: g.guardedTrack.reduce((a, x) => a + x, 0) }; });
  console.log('conflicts', JSON.stringify(out.conflicts), out.built ? JSON.stringify(out.built) : '');
  await page.evaluate(() => {
    // pairs of trains on crossing routes with their bodies over one conflict point, in view
    window.__overlaps = () => {
      const tw = window.tw, r = tw.renderer, g = r.guard, n = r.net, t = g.lastTime;
      const T = n.pathTable, E = n.pathEntry, PF = n.pathFrom, PL = n.pathLen, meta = n.meta, M = g.moveRes.length;
      const b = tw.map.getBounds();
      const at = new Map();
      const add = (res, route, i) => { const v = at.get(res); if (v) v.push(route, i); else at.set(res, [route, i]); };
      for (let i = 0; i < r.tripCount; i++) {
        const p = r.trips[i * 2], line = Math.floor(p / 6);
        if (!g.guardedAll[line]) continue;
        const pos = tw.tripAt(n, p, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
        if (!pos || !b.contains(tw.toLngLat(pos.x, pos.y))) continue;
        const run = pos.run, L = meta[p * 4 + 3], h = Math.min(Math.max(1, n.lineCars?.[line] ?? 8) * 20, L) / 2;
        const s0 = pos.s - h, s1 = pos.s + h, d0 = run === 0 ? s0 : L - s1, d1 = run === 0 ? s1 : L - s0;
        const first = T[line * 2], count = T[line * 2 + 1];
        for (let k = first; k < first + count && PF[k] <= d1; k++) {
          const len = PL[k];
          if (PF[k] + len < d0) continue;
          const ent = E[k], start = run === 0 ? PF[k] : L - PF[k] - len;
          const xs = g.crossOn.get(ent >> 1);
          if (xs) for (let q = 0; q < xs.length; q += 2) {
            const fromA = xs[q + 1] * len, sx = start + ((((run === 0 ? ent : ent ^ 1) & 1) === 0) ? fromA : len - fromA);
            if (sx > s0 && sx < s1) add(xs[q], ent >> 1, i);
          }
          if (k + 1 < first + count) {
            const sx = run === 0 ? PF[k + 1] : L - PF[k + 1];
            if (sx > s0 && sx < s1) {
              const m = g.moveIdx.get(run === 0 ? E[k] * 2 ** 26 + E[k + 1] : (E[k + 1] ^ 1) * 2 ** 26 + (E[k] ^ 1));
              if (m !== undefined) add(g.moveRes[m], m, i);
            }
          }
        }
      }
      let pairs = 0;
      for (const [res, v] of at) for (let a = 0; a < v.length; a += 2) for (let c = a + 2; c < v.length; c += 2) {
        if (v[a + 1] === v[c + 1]) continue;
        if (res < g.nCross ? v[a] !== v[c] : g.movePairs.has(v[a] * M + v[c])) pairs++;
      }
      return pairs;
    };
  });
  for (const [lng, lat, zoom] of SCENES) {
    const row = { scene: [lng, lat, zoom], windows: [] };
    for (const [rate, secs] of SPEEDS) for (const on of [true, false]) {
      if (rate === 1 && !on) continue;
      // a weekday at 8:00, so each window starts from the same rush hour
      await page.evaluate(([lng, lat, zoom, rate, on]) => {
        const tw = window.tw, c = tw.clock, day = Math.floor(c.now() / 86400) + 1;
        tw.map.jumpTo({ center: [lng, lat], zoom });
        c.setRunning(false);
        c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 8 * 3600);
        c.setMultiple(rate / 60);
        tw.renderer.guard.crossings = on;
        tw.renderer.guard.resetCounts();
        window.__w = { frames: 0, over: 0, overFrames: 0, held: 0, waitNow: 0, follow: 0, ms: 0, msMax: 0, maxLag: 0, guarded: 0 };
        window.__wOn = true;
        const f = () => {
          if (!window.__wOn) return;
          const r = tw.renderer, g = r.guard, w = window.__w;
          w.frames++;
          const o = window.__overlaps();
          w.over += o; if (o) w.overFrames++;
          w.held = Math.max(w.held, g.conflictHeld); w.waitNow = Math.max(w.waitNow, g.conflictWaitMax);
          w.follow = Math.max(w.follow, g.heldCount); w.guarded = Math.max(w.guarded, g.guardedCount);
          w.ms += r.guardMs; w.msMax = Math.max(w.msMax, r.guardMs);
          for (const v of g.lag) if (v > w.maxLag) w.maxLag = v;
          requestAnimationFrame(f);
        };
        requestAnimationFrame(f);
      }, [lng, lat, zoom, rate, on]);
      await page.waitForTimeout(secs * 1000);
      const w = await page.evaluate(() => { window.__wOn = false; const g = window.tw.renderer.guard, w = window.__w; return { ...w, ms: w.ms / Math.max(1, w.frames), waits: g.conflictWaits, meanWait: g.conflictWaitSum / Math.max(1, g.conflictWaits), longest: g.conflictLongest, letThrough: g.conflictLetThrough }; });
      for (const [k, v] of Object.entries(w)) if (typeof v === 'number') w[k] = +v.toFixed(3);
      const r = { rate, on, ...w };
      console.log(JSON.stringify([lng, lat, zoom]), JSON.stringify(r));
      row.windows.push(r);
    }
    out.scenes.push(row);
  }
  out.errors = errors;
  await writeFile(join(OUT, `cross-${save.replace(/.*[\\/]/, '').replace(/\W+/g, '_')}.json`), JSON.stringify(out, null, 1));
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no console errors');
} finally {
  await chrome.close();
}
