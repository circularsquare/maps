// T-2185: waits where two lines merge onto shared track. A new game with a double-track Y: trunk
// A - M - J, branches J - B and J - C, lines A-M-B and A-M-C stopping at M, and a short line D - E
// with two trains across the C branch on the level (a flat crossing); for each train count,
// what the sim charges (capacity markers, each line's waits by resource, `lineDelays`) and what the
// screen draws at J zoomed in (the display guard's holds behind the train ahead, waits at the Y's
// crossing pair, the longest lag), then a picture. From the maps root:
//   node trainworld/checks/t2185-merge.mjs [url] [counts json] [shade]
// Writes artifacts/trainworld/t2185/merge.json and merge-<trains>[-dark].png.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2185');
await mkdir(OUT, { recursive: true });
const url = process.argv[2] ?? 'http://localhost:6184/';
const COUNTS = process.argv[3] ? JSON.parse(process.argv[3]) : [2, 4, 10];
const SHADE = process.argv[4] ?? 'light';
const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180)), KMLAT = 1 / 111.32;
const Y = { a: [LNG0, LAT], m: [LNG0 + 1.5 * KM, LAT], j: [LNG0 + 3 * KM, LAT], b: [LNG0 + 5 * KM, LAT - 0.6 * KMLAT], c: [LNG0 + 5 * KM, LAT + 0.9 * KMLAT],
  // a short line D - E across the C branch on the level (a flat crossing), two trains
  d: [LNG0 + 3.9 * KM, LAT + 1.2 * KMLAT], e: [LNG0 + 4.3 * KM, LAT - 0.05 * KMLAT] };
const SCENE = [LNG0 + 3.55 * KM, LAT + 0.25 * KMLAT, 14.4];

const chrome = await launchChrome({ prefix: 'tw-t2185-', size: [1200, 800] });
const errors = [];
const out = { counts: [] };
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); localStorage.setItem('anitabuilder-trouble-open', '1'); } catch {} }, SHADE);
  await page.goto(`${url}?debug=1&new=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  out.built = await page.evaluate(async (Y) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: -1 }; };
    const steps = [];
    steps.push(await c.edit({ op: 'route', from: free(Y.a), to: free(Y.m), pis: [], single: false }));
    await pause(300);
    steps.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(Y.m) }, to: free(Y.j), pis: [], single: false }));
    await pause(300);
    steps.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(Y.j) }, to: free(Y.b), pis: [], single: false }));
    await pause(300);
    steps.push(await c.edit({ op: 'route', from: { kind: 'node', node: near(Y.j) }, to: free(Y.c), pis: [], single: false }));
    await pause(300);
    steps.push(await c.edit({ op: 'route', from: free(Y.d), to: free(Y.e), pis: [], single: false }));
    await pause(500);
    for (const p of [Y.a, Y.m, Y.b, Y.c, Y.d, Y.e]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    await c.edit({ op: 'constructAll' });
    await pause(500);
    const lines = [];
    for (const [t, colour] of [[Y.b, '#e8862a'], [Y.c, '#2a6fdb']]) {
      const r = await c.edit({ op: 'addLine', stops: [near(Y.a), near(Y.m), near(t)], name: '', colour });
      lines.push(r?.line);
      await pause(300);
    }
    const de = await c.edit({ op: 'addLine', stops: [near(Y.d), near(Y.e)], name: '', colour: '#0a7d3b' });
    if (de?.line !== undefined) await c.edit({ op: 'setTrainCount', line: de.line, trains: [2, 2, 2] });
    await pause(300);
    return { lines, cross: de?.line, j: near(Y.j), ok: steps.map((s) => s?.ok), issues: steps.flatMap((s) => s?.issues ?? []) };
  }, Y);
  console.log('built', JSON.stringify(out.built));
  for (const n of COUNTS) {
    await page.evaluate(async ([lines, n]) => {
      const c = window.tw.client;
      for (const l of lines) await c.edit({ op: 'setTrainCount', line: l, trains: [n, n, n] });
    }, [out.built.lines, n]);
    await page.waitForTimeout(1500);
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 0 && window.tw.renderer.tripCount > 0, undefined, { timeout: 180000 });
    await page.waitForFunction(() => !window.tw.clock.held.peek() && !window.tw.clock.capped.peek(), undefined, { timeout: 900000, polling: 1000 });
    const sim = await page.evaluate(async ([lines, j]) => {
      const tw = window.tw, w = tw.world.value;
      const delays = [];
      for (const l of lines) {
        const d = await tw.client.query({ what: 'lineDelays', line: l, level: 0 });
        const rows = [];
        for (let k = 0; k + 4 < (d?.length ?? 0); k += 5) rows.push({ kind: d[k], node: d[k + 1], s: +d[k + 4].toFixed(1) });
        const st = w.lineStats[w.save.lines.find((x) => x.num === l)?.id];
        delays.push({ line: l, rows, headway: st?.autoS?.[0], roundTrip: st?.roundTripByLevel?.[0], delay: st?.delayS?.[0], trains: st?.trainsNeeded?.[0] });
      }
      const junction = await tw.client.query({ what: 'junction', node: j });
      // each marker's distance from J, m
      const jn = w.nodes.get(j), cosLat = Math.cos((jn.lat * Math.PI) / 180);
      const fromJ = (m) => { const [lng, lat] = tw.toLngLat(m.x, m.y); return Math.round(Math.hypot((lng - jn.lng) * cosLat, lat - jn.lat) * 111320); };
      return { markers: (w.capacity[0] ?? []).map((m) => ({ kind: m.kind, rho: +m.rho.toFixed(3), delayS: +m.delayS.toFixed(1), tph: +m.tph.toFixed(1), fromJ: fromJ(m) })), delays, junction: junction ? Array.from(junction).map((x) => +(+x).toFixed(2)) : null };
    }, [out.built.lines, out.built.j]);
    console.log(n, 'trains each', JSON.stringify(sim));
    // the screen at J, a weekday from 8:00 at 60x for 30 wall s
    const screen = await page.evaluate(async ([lng, lat, zoom]) => {
      const tw = window.tw, c = tw.clock, day = Math.floor(c.now() / 86400) + 1;
      tw.map.jumpTo({ center: [lng, lat], zoom });
      c.setRunning(false);
      c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 8 * 3600);
      c.setMultiple(1);
      c.setRunning(true);
      tw.renderer.guard.resetCounts();
      const w = { frames: 0, heldMax: 0, heldSum: 0, conflictHeld: 0, lagMax: 0, lagged30: 0 };
      await new Promise((done) => {
        const t0 = performance.now();
        const f = () => {
          const g = tw.renderer.guard;
          w.frames++;
          w.heldMax = Math.max(w.heldMax, g.heldCount);
          w.heldSum += g.heldCount;
          w.conflictHeld = Math.max(w.conflictHeld, g.conflictHeld);
          let l30 = 0;
          for (const v of g.lag) { if (v > w.lagMax) w.lagMax = v; if (v > 30) l30++; }
          w.lagged30 = Math.max(w.lagged30, l30);
          if (performance.now() - t0 < 30000) requestAnimationFrame(f); else done();
        };
        requestAnimationFrame(f);
      });
      const g = tw.renderer.guard, r = tw.renderer, n = r.net;
      // the trips drawn over 30 s late now: line, run, where along it (m of its length), lag s
      const late = [];
      for (let i = 0; i < r.tripCount && late.length < 24; i++) {
        if (!(g.lag[i] > 30)) continue;
        const p = r.trips[i * 2], pos = tw.tripAt(n, p, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), g.lastTime, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
        late.push([Math.floor(p / 6), Math.floor(p / 3) % 2, pos ? Math.round(pos.s) : null, Math.round(n.meta[p * 4 + 3]), Math.round(g.lag[i])]);
      }
      w.late = late;
      return { ...w, heldMean: +(w.heldSum / Math.max(1, w.frames)).toFixed(2), lagMax: +w.lagMax.toFixed(0), conflictWaits: g.conflictWaits, conflictMean: +(g.conflictWaitSum / Math.max(1, g.conflictWaits)).toFixed(1), conflictLongest: +g.conflictLongest.toFixed(0), letThrough: g.conflictLetThrough };
    }, SCENE);
    console.log(n, 'screen', JSON.stringify(screen));
    await page.evaluate(() => window.tw.clock.setRunning(false));
    await page.waitForTimeout(500);
    // the pointer on J, so a dot there shows its hover
    const at = await page.evaluate((j) => { const tw = window.tw, n = tw.world.value.nodes.get(j), p = tw.map.project([n.lng, n.lat]), r = tw.map.getContainer().getBoundingClientRect(); return [r.left + p.x, r.top + p.y]; }, out.built.j);
    await page.mouse.move(at[0], at[1]);
    await page.waitForTimeout(400);
    screen.tip = await page.evaluate(() => [...document.querySelectorAll('.tool-tag.demand-tag')].filter((e) => e.style.display !== 'none').map((e) => e.innerText).join(' | '));
    screen.trouble = await page.evaluate(() => [...document.querySelectorAll('.trouble-list li')].map((e) => `${e.innerText.replace(/\s+/g, ' ')} [${e.dataset.tip}]`));
    console.log(n, 'tip', JSON.stringify(screen.tip), 'trouble', JSON.stringify(screen.trouble));
    await page.screenshot({ path: join(OUT, `merge-${n}${SHADE === 'dark' ? '-dark' : ''}.png`) });
    out.counts.push({ n, sim, screen });
  }
  out.errors = errors;
  await writeFile(join(OUT, `merge${SHADE === 'dark' ? '-dark' : ''}.json`), JSON.stringify(out, null, 1));
  console.log(errors.length ? 'errors: ' + errors.slice(0, 3).join(' | ') : 'no console errors');
} finally {
  await chrome.close();
}
