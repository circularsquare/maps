// T-1966: a train waiting at a flat crossing or a crossing junction move, as a short frame sequence.
// A save (the real one by default) on a weekday morning at 60x, the camera on a scene until a
// train in view stands at a conflict; then the camera on that conflict, the waiting train
// selected, and the clock stepped by hand, a picture each step, into artifacts/trainworld/t1966/.
// From the maps root:
//   node trainworld/checks/t1966-pics.mjs [url] [save | test] [lng,lat,zoom] [kind: any|crossing|junction] [name]
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave } from './lib.mjs';
import { buildTest, TEST_SCENE } from './t1966-layout.mjs';

const OUT = join(ARTIFACTS, 't1966');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6157/';
const save = process.argv[3] ?? REAL_SAVE;
const [lng, lat, zoom] = process.argv[4] ? process.argv[4].split(',').map(Number) : save === 'test' ? TEST_SCENE : [-73.979, 40.755, 14];
const kind = process.argv[5] ?? 'any';
const name = process.argv[6] ?? 'wait';
const STEP_S = 5, STEPS = 10, ZOOM = 16;
const chrome = await launchChrome({ prefix: 'tw-t1966p-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  if (save === 'test') console.log('built', JSON.stringify(await buildTest(page)));
  else await importSave(page, await readFile(save));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
  await page.waitForFunction(() => !window.tw.clock.held.peek() && !window.tw.clock.capped.peek(), undefined, { timeout: 900000, polling: 1000 });
  const found = await page.evaluate(async ([lng, lat, zoom, kind]) => {
    const tw = window.tw, c = tw.clock, r = tw.renderer, g = r.guard, day = Math.floor(c.now() / 86400) + 1;
    tw.map.jumpTo({ center: [lng, lat], zoom });
    c.setRunning(false);
    c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 8 * 3600);
    c.setSpeed(0);
    const seen = new Set();
    for (let k = 0; k < 20000; k++) {
      await new Promise((d) => requestAnimationFrame(d));
      const n = r.net;
      for (let i = 0; i < g.waitRes.length; i++) {
        const res = g.waitRes[i];
        if (res < 0 || seen.has(i)) continue;
        if (kind === 'crossing' && res >= g.nCross) continue;
        if (kind === 'junction' && res < g.nCross) continue;
        const since = g.lastTime - r.tripsEpoch - g.waitFrom[i];
        // standing 10-20 s already: a wait long enough to see
        if (since < 10) continue;
        if (since > 20) { seen.add(i); continue; }
        const p = r.trips[i * 2], line = Math.floor(p / 6), dep = r.trips[i * 2 + 1] + r.tripsEpoch;
        const at = tw.tripAt(n, p, dep + r.tripLag(i), g.lastTime);
        if (!at) continue;
        // the conflict point: its front's hold point plus the margin
        const h = Math.min(Math.max(1, n.lineCars?.[line] ?? 8) * 20, n.meta[p * 4 + 3]) / 2;
        const sx = at.s + h + 20, L = n.meta[p * 4 + 3], d = at.run === 0 ? sx : L - sx;
        const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
        const si = first + Math.max(0, Math.min(count - 1, Math.round(d / step)));
        c.setRunning(false);
        return { t: g.lastTime, res, crossing: res < g.nCross, line, lineId: n.lineIds?.[line], lag: r.tripLag(i), mid: tw.toLngLat(n.samples[si * 2], n.samples[si * 2 + 1]), sel: { kind: 'train', line: String(line), profile: p, dep } };
      }
    }
    return null;
  }, [lng, lat, zoom, kind]);
  console.log('found', JSON.stringify(found));
  if (found) {
    await page.evaluate(([c, sel, z]) => { window.tw.select(sel); window.tw.map.jumpTo({ center: c, zoom: z }); }, [found.mid, found.sel, ZOOM]);
    await page.waitForTimeout(2500);
    for (let k = 0; k < STEPS; k++) {
      const st = await page.evaluate(() => { const g = window.tw.renderer.guard; return { t: g.lastTime, held: g.conflictHeld, wait: +g.conflictWaitMax.toFixed(1) }; });
      console.log(k, JSON.stringify(st));
      await page.screenshot({ path: join(OUT, `${name}-${k + 1}.png`) });
      await page.evaluate((s) => { const c = window.tw.clock; c.set(c.now() + s); }, STEP_S);
      await page.waitForTimeout(700);
    }
  }
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no console errors');
} finally {
  await chrome.close();
}
