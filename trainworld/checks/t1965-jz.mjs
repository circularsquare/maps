// T-1965: on the shared New York save (artifacts/trainworld/t1305-shared/nyc_real.save) the Z is
// locked behind the J (both run Jamaica Center to Broad St): its round trip against the J's, and
// pictures of a Z and the J ahead of it, close in, a few game minutes apart, into
// artifacts/trainworld/t1965/. From the maps root:
//   node trainworld/checks/t1965-jz.mjs [url]
import { readFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, importSave } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1965');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6160/';
const SAVE = join(ARTIFACTS, 't1305-shared', 'nyc_real.save');
const chrome = await launchChrome({ prefix: 'tw-t1965jz-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
  await page.waitForFunction(() => !window.tw.clock.held.peek(), undefined, { timeout: 300000 });
  const info = await page.evaluate(() => {
    const w = window.tw.world.value, find = (x) => w.save.lines.find((l) => l.letter === x || l.name === x);
    const j = find('J'), z = find('Z'), st = (l) => w.lineStats[l.id];
    return { j: j.num, z: z.num, jRt: st(j).roundTripS, zRt: st(z).roundTripS, jStops: j.stops.length, zStops: z.stops.length, jTph: j.tph, zTph: z.tph };
  });
  console.log(JSON.stringify(info));
  // a weekday rush hour, the clock at 60x
  await page.evaluate(() => { const c = window.tw.clock, day = Math.floor(c.now() / 86400) + 1; c.setRunning(false); c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 7.75 * 3600); c.setSpeed(0); c.setRunning(true); window.tw.map.jumpTo({ center: [-73.88, 40.69], zoom: 11.5 }); });
  // wait for a Z with a J ahead of it within 800 m on its track
  const found = await page.evaluate(async ([J, Z]) => {
    const tw = window.tw, r = tw.renderer;
    for (let k = 0; k < 3000; k++) {
      await new Promise((d) => requestAnimationFrame(d));
      const n = r.net, t = r.guard.lastTime, T = n.pathTable, E = n.pathEntry, PF = n.pathFrom, PL = n.pathLen;
      const place = (i) => {
        const p = r.trips[i * 2], at = tw.tripAt(n, p, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
        if (!at) return null;
        const L = n.meta[p * 4 + 3], d = at.run === 0 ? at.s : L - at.s;
        for (let q = T[at.line * 2]; q < T[at.line * 2] + T[at.line * 2 + 1]; q++) if (d >= PF[q] && d <= PF[q] + PL[q]) {
          const into = at.run === 0 ? d - PF[q] : PF[q] + PL[q] - d;
          const key = at.run === 0 ? E[q] : E[q] ^ 1;
          return { key, into, at, lag: r.tripLag(i), profile: p, dep: r.trips[i * 2 + 1] + r.tripsEpoch };
        }
        return null;
      };
      const zs = [], js = [];
      for (let i = 0; i < r.tripCount; i++) {
        const line = Math.floor(r.trips[i * 2] / 6);
        if (line === Z || line === J) { const pl = place(i); if (pl) (line === Z ? zs : js).push(pl); }
      }
      for (const a of zs) for (const b of js) if (a.key === b.key && b.into > a.into && b.into - a.into < 260 && a.lag > 5 && a.at.s > 2000) {
        return { t, gap: b.into - a.into, zLag: a.lag, jLag: b.lag, mid: tw.toLngLat((a.at.x + b.at.x) / 2, (a.at.y + b.at.y) / 2), sel: { kind: 'train', line: String(Z), profile: a.profile, dep: a.dep } };
      }
    }
    return null;
  }, [info.j, info.z]);
  console.log('found', JSON.stringify(found));
  if (found) {
    // the Z selected (its halo), so it reads apart from the J in the same colour
    await page.evaluate(([c, sel]) => { window.tw.clock.setRunning(false); window.tw.select(sel); window.tw.map.jumpTo({ center: c, zoom: 15.6 }); }, [found.mid, found.sel]);
    await page.waitForTimeout(2500);
    await page.screenshot({ path: join(OUT, 'jz-1.png') });
    // a game minute on, then another
    for (const k of [2, 3]) {
      await page.evaluate(() => { const c = window.tw.clock; c.setRunning(true); });
      await page.waitForTimeout(1000);
      await page.evaluate(() => window.tw.clock.setRunning(false));
      await page.waitForTimeout(1500);
      await page.screenshot({ path: join(OUT, `jz-${k}.png`) });
    }
  }
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no console errors');
} finally {
  await chrome.close();
}
