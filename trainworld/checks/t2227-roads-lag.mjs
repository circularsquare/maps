// T-2227: what T-1929's road crossings cost the track tool, in headless Chrome. Each run loads a new
// New York game or the real save, with the road tiles on or off (`?perfOff=roads`), then:
//   ghost   the track tool's ghost from Midtown swept round a 3 km circle, a pointer move every frame,
//           twice (cold: the road tiles land meanwhile; warm: they are in): each preview's
//           ask-to-answer ms, the frame gaps, the main thread's long tasks, the "roads" re-asks
//   place   PLACES ground level segments in Brooklyn and Queens, each where no tile has been
//           fetched: the ghost shown 2.5 s, then the click; click to the edit's answer, to its
//           state, the worker's state ms and phase lines read, and the long tasks in the 3 s after
// Runs go in the order of RUNS (default new:on,new:off,new:on,new:off), each in a fresh Chrome, and
// each prints one JSON line; the end prints each case's medians. JSON in artifacts/trainworld/t2226/.
// Usage: node trainworld/checks/t2227-roads-lag.mjs [url] (default http://localhost:6191/)

import { mkdirSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ANGLE, ARTIFACTS, importSave, launchChrome, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6191/';
const RUNS = (process.env.RUNS ?? 'new:on,new:off,new:on,new:off').split(',').map((r) => r.split(':'));
const PLACES = Number(process.env.PLACES ?? 5);
const OUT = join(ARTIFACTS, 't2226');
mkdirSync(OUT, { recursive: true });

async function run(which, roads) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t2227-', size: [1200, 800], args: ANGLE });
  try {
    const page = await ctx.newPage();
    page.on('pageerror', (e) => console.log('pageerror', e.message));    const off = roads === 'off' ? '&perfOff=roads' : '';
    await page.goto(`${BASE}?debug=1&new=1&city=nyc&paused=1&demandWorkers=1${off}&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools && window.twDemand?.client, undefined, { timeout: 180000 });
    if (which === 'real') {
      await importSave(page, await readFile(REAL_SAVE));
      await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) >= 5, undefined, { timeout: 300000, polling: 250 });
    }
    await page.waitForFunction(() => { const d = window.twDemand, v = d?.view?.(); return !!v?.final && !d.client.waiting; }, undefined, { timeout: 900000, polling: 500 });
    await page.waitForTimeout(3000);
    const r = await page.evaluate(async ([PLACES]) => {
      const tw = window.tw, map = tw.map, tools = tw.tools, c = tw.client;
      const ui = await import('/src/game/ui.ts');
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
      const at = (lng, lat) => { const p = map.project([lng, lat]); return { point: { x: p.x, y: p.y }, lngLat: map.unproject([p.x, p.y]), originalEvent: { detail: 1, shiftKey: false } }; };
      const q = (xs, f) => { const s = xs.slice().sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * f))] : -1; };
      const longs = [];
      new PerformanceObserver((l) => { for (const e of l.getEntries()) longs.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: false });
      const longIn = (t0, t1) => longs.filter(([s]) => s >= t0 && s < t1).reduce((a, [, d]) => a + d, 0);
      const ask = c.preview.bind(c);
      let asks = [];
      c.preview = (req, cb) => { const p = { t: performance.now(), ms: -1 }; asks.push(p); return ask(req, (a) => { p.ms = performance.now() - p.t; cb(a); }); };
      const recv = c.receive.bind(c);
      let log = [];
      c.receive = (msg) => {
        const t = performance.now();
        const out = recv(msg);
        if (msg.kind === 'roads' || msg.kind === 'state' || msg.kind === 'edited') log.push({ kind: msg.kind, t, main: performance.now() - t, ms: msg.ms ?? msg.state?.ms, early: msg.early, ok: msg.ok });
        return out;
      };
      ui.tool.value = 'track';
      ui.buildLevel.value = 0;
      ui.keepPlacing.value = false;
      ui.snapReal.value = false;
      ui.snapBeside.value = false;

      // ---- the ghost, cold then warm
      const ghost = async () => {
        const A = [-73.9845, 40.7485];
        map.jumpTo({ center: A, zoom: 12.5 });
        await sleep(1500);
        tools.reset();
        tools.click(at(...A));
        await sleep(300);
        asks = []; log = [];
        const gaps = [];
        const t0 = performance.now();
        let last = t0;
        for (let k = 0; k < 240; k++) {
          const th = (k / 240) * 2 * Math.PI, dlat = 0.027 * Math.cos(th), dlng = 0.0356 * Math.sin(th);
          tools.move(at(A[0] + dlng, A[1] + dlat));
          await frame();
          const now = performance.now();
          gaps.push(now - last);
          last = now;
        }
        await sleep(1500);
        const t1 = performance.now();
        tools.reset();
        const ms = asks.filter((a) => a.ms >= 0).map((a) => a.ms);
        return {
          previews: ms.length, mean: ms.reduce((a, b) => a + b, 0) / Math.max(1, ms.length), p50: q(ms, 0.5), p95: q(ms, 0.95), max: Math.max(0, ...ms),
          gapMean: gaps.reduce((a, b) => a + b, 0) / gaps.length, gapP95: q(gaps, 0.95), gapMax: Math.max(...gaps), longMs: longIn(t0, t1),
          roads: log.filter((l) => l.kind === 'roads').length, states: log.filter((l) => l.kind === 'state').length,
        };
      };
      const cold = await ghost();
      const warm = await ghost();

      // ---- placements, each in a fresh place
      const places = [];
      for (let i = 0; i < PLACES; i++) {
        const S = [-73.975 + 0.03 * i, 40.672], E = [S[0] + 0.012, S[1] + 0.008];
        map.jumpTo({ center: [(S[0] + E[0]) / 2, (S[1] + E[1]) / 2], zoom: 13.5 });
        await sleep(1500);
        tools.reset();
        tools.click(at(...S));
        await sleep(200);
        for (let k = 1; k <= 10; k++) { tools.move(at(S[0] + (E[0] - S[0]) * k / 10, S[1] + (E[1] - S[1]) * k / 10)); await frame(); }
        await sleep(2500);
        tools.move(at(...E));
        await sleep(300);
        log = [];
        const w0 = tw.world.value, t0 = performance.now();
        tools.click(at(...E));
        let worldAt = -1;
        while (performance.now() - t0 < 3000) { await frame(); if (worldAt < 0 && tw.world.value !== w0) worldAt = performance.now() - t0; }
        const ed = log.find((l) => l.kind === 'edited'), st = log.find((l) => l.kind === 'state' && l.t >= (ed?.t ?? 0));
        places.push({
          ok: ed?.ok ?? null, edited: ed ? ed.t - t0 : -1, state: st ? st.t - t0 : -1, world: worldAt,
          stateMainMs: st?.main ?? -1, stateWorkerMs: st?.ms ?? null, editMs: ed?.ms ? { total: ed.ms.total, apply: ed.ms.apply, sLines: ed.ms.sLines, sPhases: ed.ms.sPhases, nPhaseLines: ed.ms.nPhaseLines } : null,
          states: log.filter((l) => l.kind === 'state').length, roads: log.filter((l) => l.kind === 'roads').length, longMs: longIn(t0, t0 + 3000),
        });
        tools.reset();
        await sleep(1000);
      }
      c.preview = ask;
      c.receive = recv;
      return { cold, warm, places };
    }, [PLACES]);
    return r;
  } finally {
    await close();
  }
}

const results = [];
const med = (xs) => { const s = xs.filter((x) => x >= 0).sort((a, b) => a - b); return s.length ? Math.round(s[Math.floor(s.length / 2)] * 10) / 10 : -1; };
for (const [which, roads] of RUNS) {
  const t = Date.now();
  const r = await run(which, roads);
  const row = { which, roads, at: new Date().toISOString(), loadAndRunS: Math.round((Date.now() - t) / 1000), ...r };
  results.push(row);
  const round = (o) => JSON.parse(JSON.stringify(o, (k, v) => (typeof v === 'number' ? Math.round(v * 10) / 10 : v)));
  console.log(`t2227 ${which} roads ${roads}: ${JSON.stringify(round({ cold: r.cold, warm: r.warm }))}`);
  console.log(`t2227 ${which} roads ${roads} places: ${JSON.stringify(round(r.places.map((p) => ({ ok: p.ok, edited: p.edited, state: p.state, world: p.world, main: p.stateMainMs, worker: p.stateWorkerMs, lines: p.editMs?.sLines, phases: p.editMs?.sPhases, nPhase: p.editMs?.nPhaseLines, long: p.longMs }))))}`);
  writeFileSync(join(OUT, `t2227-${process.env.TAG ?? 'run'}.json`), JSON.stringify(results, null, 1));
}
for (const which of ['new', 'real']) for (const roads of ['on', 'off']) {
  const xs = results.filter((r) => r.which === which && r.roads === roads);
  if (!xs.length) continue;
  const g = (pass, f) => med(xs.map((r) => r[pass][f]));
  const p = (f) => med(xs.flatMap((r) => r.places.map(f)));
  console.log(`t2227 ${which} roads ${roads} (n ${xs.length}): ghost cold p50 ${g('cold', 'p50')} p95 ${g('cold', 'p95')} frame p95 ${g('cold', 'gapP95')} long ${g('cold', 'longMs')}; warm p50 ${g('warm', 'p50')} p95 ${g('warm', 'p95')} frame p95 ${g('warm', 'gapP95')} long ${g('warm', 'longMs')}; place edited ${p((q) => q.edited)} state ${p((q) => q.state)} world ${p((q) => q.world)} worker state ${p((q) => (q.stateWorkerMs ? q.stateWorkerMs[0] + q.stateWorkerMs[1] : -1))} main ${p((q) => q.stateMainMs)} long ${p((q) => q.longMs)}`);
}
