// T-1966: what the waits at flat crossings and crossing junction moves cost a frame, against
// `?perfOff=crossWait` (`renderer.guard.crossings` off). A save (the real one by default), the
// clock at its fastest, the camera still on each scene; windows of 6 s with the waits on and off in
// turn, three of each: the overlay's draw ms a frame (tw.stats), the guard's own ms, the main
// thread's busy ms a frame (CDP TaskDuration), trains guarded and held (on the track or at a
// conflict). From the maps root:
//   node trainworld/checks/t1966-perf.mjs [url] [save] [scenes json]
// scenes: [[lng, lat, zoom], ...]; default Midtown at 14 and 15.5, Long Island City at 14.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1966');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6157/';
const save = process.argv[3] ?? REAL_SAVE;
const SCENES = process.argv[4] ? JSON.parse(process.argv[4]) : [[-73.979, 40.755, 14], [-73.979, 40.755, 15.5], [-73.918, 40.748, 14]];
const chrome = await launchChrome({ prefix: 'tw-t1966q-', size: [1400, 900] });
const errors = [];
const out = { save, scenes: [] };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  const cdp = await chrome.ctx.newCDPSession(page);
  await cdp.send('Performance.enable');
  await page.goto(`${url}?debug=1&new=1&opened=all&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(save));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
  await page.waitForFunction(() => !window.tw.clock.held.peek() && !window.tw.clock.capped.peek(), undefined, { timeout: 900000, polling: 1000 });
  await page.evaluate(() => { const c = window.tw.clock, day = Math.floor(c.now() / 86400) + 1; c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 8 * 3600); c.setSpeed(2); c.setRunning(true); });
  const busy = async () => { const { metrics } = await cdp.send('Performance.getMetrics'); return metrics.find((m) => m.name === 'TaskDuration').value * 1000; };
  for (const [lng, lat, zoom] of SCENES) {
    await page.evaluate(([lng, lat, zoom]) => { window.tw.map.jumpTo({ center: [lng, lat], zoom }); }, [lng, lat, zoom]);
    await page.waitForTimeout(2500);
    const win = { on: [], off: [] };
    for (let rep = 0; rep < 3; rep++) for (const on of [true, false]) {
      await page.evaluate((on) => { window.tw.renderer.guard.crossings = on; window.tw.reset(); window.__g = []; window.__gOn = true; const f = () => { if (!window.__gOn) return; const r = window.tw.renderer; let mx = 0, late = 0; for (const v of r.guard.lag) { if (v > mx) mx = v; if (v > 120) late++; } window.__g.push([r.guardMs, r.guard.guardedCount, r.guard.heldCount + r.guard.conflictHeld, mx, late]); requestAnimationFrame(f); }; requestAnimationFrame(f); }, on);
      const b0 = await busy();
      await page.waitForTimeout(6000);
      const b1 = await busy();
      const s = await page.evaluate(() => { window.__gOn = false; const st = window.tw.stats(); const g = window.__g; return { draws: st.draws, drawMs: st.drawMs, frames: g.length, guardMs: g.reduce((a, x) => a + x[0], 0) / Math.max(1, g.length), guardMax: Math.max(0, ...g.map((x) => x[0])), guarded: Math.max(0, ...g.map((x) => x[1])), held: Math.max(0, ...g.map((x) => x[2])), maxLag: Math.max(0, ...g.map((x) => x[3])), over2min: Math.max(0, ...g.map((x) => x[4])) }; });
      win[on ? 'on' : 'off'].push({ ...s, mainMsPerFrame: (b1 - b0) / Math.max(1, s.frames) });
    }
    const avg = (a, k) => a.reduce((s, x) => s + x[k], 0) / a.length;
    const row = { scene: [lng, lat, zoom], on: { drawMs: avg(win.on, 'drawMs'), mainMs: avg(win.on, 'mainMsPerFrame'), guardMs: avg(win.on, 'guardMs'), guardMax: Math.max(...win.on.map((x) => x.guardMax)), guarded: Math.max(...win.on.map((x) => x.guarded)), held: Math.max(...win.on.map((x) => x.held)), maxLag: Math.max(...win.on.map((x) => x.maxLag)), over2min: Math.max(...win.on.map((x) => x.over2min)) }, off: { drawMs: avg(win.off, 'drawMs'), mainMs: avg(win.off, 'mainMsPerFrame'), guardMs: avg(win.off, 'guardMs') }, trips: await page.evaluate(() => window.tw.renderer.tripCount) };
    for (const k of ['on', 'off']) for (const [x, v] of Object.entries(row[k])) if (typeof v === 'number') row[k][x] = +v.toFixed(3);
    console.log(JSON.stringify(row));
    out.scenes.push(row);
  }
  out.errors = errors;
  await writeFile(join(OUT, `perf-${save.replace(/.*[\\/]/, '').replace(/\W+/g, '_')}.json`), JSON.stringify(out, null, 1));
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no console errors');
} finally {
  await chrome.close();
}
