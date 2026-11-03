// T-2228: what the sim does today with a local and an express on one double track.
// Builds two test lines in a sandbox (a Tokaido-style high-speed line and a commuter line), runs a
// local and an express on each at a few headways, and reads each line's round trip, its waits and
// the stop times. Output: artifacts/trainworld/t2228/today.json. From the maps root:
//   node <this> [url]
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const OUT = join(ARTIFACTS, 't2228');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6192/';

const chrome = await launchChrome({ prefix: 'tw-t2228-', size: [1400, 900] });
const errors = [];
const result = { cases: [] };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);

  // a case: n stations `km` apart on a straight east from (lat, lng0), express stopping at `xStops`
  const CASES = [
    { name: 'shinkansen', lat: 40.3, lng0: -79.6, n: 7, km: 30, level: -2, hsr: true, platform: 400, xStops: [0, 3, 6],
      runs: [{ l: 600, x: 600 }, { l: 1800, x: 600 }, { l: 1200, x: 300 }] },
    { name: 'shinkansen-unlocked', reuse: 'shinkansen', xStops: [1, 3, 6], runs: [{ l: 600, x: 600 }, { l: 1800, x: 600 }] },
    { name: 'commuter', lat: 40.9, lng0: -79.6, n: 13, km: 2, level: -2, hsr: false, platform: 200, xStops: [0, 4, 8, 12],
      runs: [{ l: 600, x: 600 }, { l: 300, x: 600 }, { l: 900, x: 900 }] },
  ];
  const built = {};
  for (const C of CASES) {
    const out = await page.evaluate(async ([C, prev]) => {
      const tw = window.tw, c = tw.client;
      const pause = (ms) => new Promise((d) => setTimeout(d, ms));
      const log = [];
      let stops;
      let local;
      if (prev) {
        ({ stops, local } = prev);
      } else {
        const KM = 1 / (111.32 * Math.cos((C.lat * Math.PI) / 180));
        const xy = Array.from({ length: C.n }, (_, k) => { const p = tw.lngLatToPack(C.lng0 + k * C.km * KM, C.lat); return [p.x, p.y]; });
        const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
        const r0 = await c.edit({ op: 'route', from: { kind: 'free', x: xy[0][0], y: xy[0][1], level: C.level }, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: C.level }, pis: [], single: false, hsr: C.hsr, design: C.hsr ? 320 : 160 });
        log.push(['route0', r0?.ok, r0?.issue ?? r0?.error]);
        for (let k = 2; k < xy.length; k++) {
          await pause(250);
          const r = await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: C.level }, pis: [], single: false, hsr: C.hsr, design: C.hsr ? 320 : 160 });
          log.push(['route' + k, r?.ok, r?.issue ?? r?.error]);
        }
        await pause(500);
        for (const p of xy) { const r = await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: C.platform, name: '' }); log.push(['station', r?.ok, r?.issue ?? r?.error]); await pause(150); }
        const k = await c.edit({ op: 'constructAll' });
        log.push(['construct', k?.ok, k?.issue ?? k?.error]);
        await pause(600);
        stops = xy.map(near);
        const a = await c.edit({ op: 'addLine', stops, name: 'Local', colour: '#0a7d3b', autoFast: true });
        local = a.line;
        if (C.hsr) await c.edit({ op: 'lineFast', line: local, fast: true });
      }
      const e = await c.edit({ op: 'copyLine', line: local });
      const express = e.line;
      await c.edit({ op: 'setStops', line: express, stops: C.xStops.map((i) => stops[i]) });
      if (C.hsr) await c.edit({ op: 'lineFast', line: express, fast: true });
      await c.edit({ op: 'lineLook', line: express, name: 'Express', colour: '#d9467a', letter: 'X' });
      const read = () => {
        const st = tw.world.value.lineStats;
        const pick = (id) => { const s = st[String(id)]; return s && { km: s.km, rt: s.roundTripByLevel, delay: s.delayS, need: s.trainsNeeded, from: s.fromStartByLevel?.[0], dwell: s.dwellS, turn: s.turnaroundS, fast: s.fast, status: s.status }; };
        return { local: pick(local), express: pick(express) };
      };
      const runs = [];
      for (const R of C.runs) {
        await c.edit({ op: 'setHeadway', line: local, seconds: [R.l, R.l, 0] });
        await c.edit({ op: 'setHeadway', line: express, seconds: [R.x, R.x, 0] });
        await pause(2500);
        const both = read();
        await c.edit({ op: 'setHeadway', line: local, seconds: [0, 0, 0] });
        await pause(2000);
        const alone = read();
        runs.push({ ...R, both, alone });
      }
      // leave the express out of service for the next case
      await c.edit({ op: 'removeLine', line: express });
      await pause(500);
      return { log, stops, local, express, runs };
    }, [C, C.reuse ? built[C.reuse] : null]);
    if (!C.reuse) built[C.name] = { stops: out.stops, local: out.local };
    result.cases.push({ name: C.name, km: C.km, n: C.n, xStops: C.xStops, ...out });
    console.log(C.name, JSON.stringify(out.log), JSON.stringify(out.runs.map((r) => ({ l: r.l, x: r.x, localRt: r.both.local?.rt, exBoth: r.both.express?.rt, exAlone: r.alone.express?.rt, exDelay: r.both.express?.delay }))));
  }
  // a shot of the line panel for the locked Shinkansen pair
  await page.screenshot({ path: join(OUT, 'last-page.png') });
} finally {
  result.errors = errors;
  await writeFile(join(OUT, 'today.json'), JSON.stringify(result, null, 1));
  await chrome.close();
}
console.log('errors', errors.length);
