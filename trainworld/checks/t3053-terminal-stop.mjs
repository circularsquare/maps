// T-3053: a train standing at a line's end sits fully on the platform. A ?new=1 sandbox game with a
// 2 km line between two dead-end stations (platform 200 m), 3 trains; the clock runs until a train
// stands at the west end (run 1's end), then a zoom 17.5 shot, before and after a terminal is built
// there (artifacts/trainworld/t3053-<tag>-plain.png and -terminal.png). Also prints how far the
// standing train's front and back are from the platform's ends. From the maps root:
//   node trainworld/checks/t3053-terminal-stop.mjs [url] [tag]
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, buildLine } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6167/';
const tag = process.argv[3] ?? 'after';
const LAT = 40.3, LNG0 = -79.6, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180));
const chrome = await launchChrome({ prefix: 'tw-t3053-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  const THROUGH = process.env.THROUGH === '1';
  const r = THROUGH ? await page.evaluate(async ([P]) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = P.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
    await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
    await pause(300);
    await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[1]) }, to: { kind: 'free', x: xy[2][0], y: xy[2][1], level: -1 }, pis: [], single: false });
    await pause(500);
    for (const p of xy) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    await c.edit({ op: 'constructAll' }); await pause(500);
    const r = await c.edit({ op: 'addLine', stops: [near(xy[2]), near(xy[1])], name: 'T', colour: '#d9467a' });
    await c.edit({ op: 'setTrainCount', line: r.line, trains: [4, 4, 4] });
    return r;
  }, [[[LNG0 + 2 * KM, LAT], [LNG0, LAT], [LNG0 - 2 * KM, LAT]]]) : await buildLine(page, [[LNG0, LAT], [LNG0 + 2 * KM, LAT]], { name: 'T', colour: '#d9467a', trains: [4, 4, 4] });
  console.log('line', JSON.stringify(r));
  await page.waitForTimeout(2000);
  const standing = () => page.evaluate(async (RUN) => {
    const tw = window.tw, rr = tw.renderer, c = tw.clock;
    c.setSpeed(0); c.setRunning(true);
    const arrive = (n, p) => { const k = (n.meta[p * 4] + n.meta[p * 4 + 1] - 1) * 4; return n.phases[k + 3] < 0 ? n.phases[k] + n.phases[k + 2] / -n.phases[k + 3] : n.phases[k]; };
    const w0 = performance.now();
    return await new Promise((done) => {
      const step = () => {
        if (performance.now() - w0 > 60000) { c.setRunning(false); return done(null); }
        try { const n = rr.net, t = rr.guard.lastTime;
        for (let i = 0; i < rr.tripCount; i++) {
          const prof = rr.trips[i * 2], lag = rr.tripLag(i), dep = rr.trips[i * 2 + 1] + rr.tripsEpoch + lag;
          const at = tw.tripAt(n, prof, dep, t, rr.tripEnds ? rr.tripEnds[i] + rr.tripsEpoch : -Infinity, 0, 0, undefined, 0, 0, rr.tripBays ?? undefined, i * 4);
          if (!at || at.run !== RUN) continue;
          if (t - dep > arrive(n, prof) + 1) {
            c.setRunning(false);
            const cars = [0, 7].map((k) => tw.trainCarAt?.(n, at, k)).filter(Boolean);
            return done({ s: at.s, ex: at.ex, L: n.meta[prof * 4 + 3], cars: cars.map((p) => [p.x, p.y]) });
          }
        }
        } catch (e) { c.setRunning(false); return done({ error: String(e.stack ?? e) }); }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, THROUGH ? 0 : 1);
  const shot = async (name) => {
    await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 17.2 }), [LNG0 + 0.07 * KM, LAT]);
    await page.waitForTimeout(1800);
    await page.screenshot({ path: join(ARTIFACTS, `t3053-${tag}-${name}.png`), clip: { x: 200, y: 250, width: 1000, height: 400 } });
  };
  console.log('plain', JSON.stringify(await standing()), errors.slice(0, 3).join(' | '));
  await shot('plain');
  const t = await page.evaluate(([lng, lat, thru]) => { const w = window.tw.world.value, q = window.tw.lngLatToPack(lng, lat); let best = null, bd = 1e12; for (const n of w.nodes.values()) { const d = thru ? Math.hypot(n.x - q.x, n.y - q.y) : n.x; if (d < bd) { bd = d; best = n.id; } } return best; }, [LNG0, LAT, THROUGH]);
  console.log('terminal', await page.evaluate(async (id) => { const ok = (await window.tw.client.edit({ op: 'stationUpgrade', nodes: [id], bit: 1, on: true }))?.ok; await new Promise((d) => setTimeout(d, 2500)); return ok; }, t));
  console.log('terminal run', JSON.stringify(await standing()));
  await shot('terminal');
  console.log(errors.length ? errors.slice(0, 3).join(' | ') : 'no console errors');
} finally { await chrome.close(); }
