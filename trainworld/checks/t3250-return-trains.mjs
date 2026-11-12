// T-3250: a line whose return run takes its own way back (round one-way track) has its return trains drawn on that
// way. Two stations joined by two bows of single track, the south one set one way east and the north one west (a
// player's pair of one-way tracks as double track), and one line between them both ways. The sim runs the line east
// on the south bow and back west on the north (`line_back_path`); this checks that every train drawn on the bows'
// middle stretch is on the bow its run takes, by the CPU's drawn place (`tripAt`) and the GPU's (`probe`), and
// screenshots the pair with trains on it in artifacts/trainworld/t3250/.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6139):
//   node trainworld/checks/t3250-return-trains.mjs [dev url] [tag]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const args = process.argv.slice(2);
const BASE = args.find((a) => /^https?:/.test(a)) ?? 'http://localhost:6139/';
const TAG = args.find((a) => !/^https?:/.test(a)) ?? 'master';
const OUT = join(ARTIFACTS, 't3250');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const RC = [-73.945, 40.672], HALF = 1200, BOW = 250;
const chrome = await launchChrome({ prefix: 'tw-t3250-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.client, undefined, { timeout: 180000 });

  const built = await page.evaluate(async ([c, half, bow]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const res = [];
    const w = P(-half, 0), e = P(half, 0);
    const between = (W, E) => tw.world.value.edges.filter((x) => (x.a === W && x.b === E) || (x.a === E && x.b === W));
    // the south bow first (pack y grows north), then the north bow node to node
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...w, level: 0 }, to: { kind: 'free', ...e, level: 0 }, pis: [{ ...P(0, -bow), level: 0 }], single: true }));
    await pause(400);
    const W = near(w), E = near(e);
    const south = between(W, E)[0];
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: W }, to: { kind: 'node', node: E }, pis: [{ ...P(0, bow), level: 0 }], single: true }));
    await pause(400);
    const north = between(W, E).find((x) => x.id !== south?.id);
    for (const n of [W, E]) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [W, E], name: 'Purple', colour: '#8e44ad' });
    await pause(500);
    // south one way east (W to E), north one way west: right-hand running
    const way = (x, from) => (x.a === from ? 1 : 2);
    res.push(await cl.edit({ op: 'oneWay', edge: south.id, dir: way(south, W) }));
    await pause(400);
    res.push(await cl.edit({ op: 'oneWay', edge: north.id, dir: way(north, E) }));
    await pause(400);
    res.push(await cl.edit({ op: 'setTrainCount', line: line.line, trains: [4, 4, 4] }));
    await pause(1500);
    const st = tw.world.value.lineStats[line.line];
    return {
      W, E, south: south?.id, north: north?.id, line: line.line, ok: line.ok, status: st?.status, path: st?.path,
      ways: tw.world.value.edges.filter((x) => x.id === south?.id || x.id === north?.id).map((x) => [x.id, x.oneWay]),
      refused: res.filter((r) => r && r.ok === false).map((r) => JSON.stringify(r.issues ?? r).slice(0, 100)),
    };
  }, [RC, HALF, BOW]);
  console.log('built', JSON.stringify(built));
  check(built.ok && !built.refused.length && built.south != null && built.north != null, 'two bows of single track between two stations, a line on them', built.refused.join(' | '));
  check(built.ways.every(([, w]) => w === 1 || w === 2), 'both bows run one way', JSON.stringify(built.ways));
  const pathEdges = (built.path ?? []).map((p) => p >> 1);
  // its way back is the north bow, the only other way (`back_path_of`; the main thread has no line rows to read it from)
  check(pathEdges.join() === String(built.south) && built.status === 'running', 'the line runs east on the south bow, so back on the north', `path ${pathEdges} ${built.status}`);

  // a weekday morning, the clock running; every frame, each train on the bows' middle third, north or south of the stations
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 14.3 }), RC);
  await settle(1000);
  const run = await page.evaluate(async ([c, half]) => {
    const tw = window.tw, r = tw.renderer, ck = tw.clock;
    const day = Math.floor(ck.now() / 86400);
    ck.setRunning(false);
    ck.set((day + 1 + ((day + 1) % 7 >= 5 ? 2 : 0)) * 86400 + 8 * 3600);
    ck.setSpeed(1);
    ck.setRunning(true);
    const o = tw.lngLatToPack(c[0], c[1]);
    const t0 = ck.now(), w0 = performance.now();
    // per run: [north, south] counts as the CPU draws them, and the GPU's disagreements
    const cpu = [[0, 0], [0, 0]];
    let gpuWorst = 0, gpuN = 0, shotT = null;
    await new Promise((done) => {
      const step = () => {
        const t = r.guard?.lastTime ?? ck.now();
        if (ck.now() - t0 > 1800 || performance.now() - w0 > 60000) return done();
        const n = r.net;
        const pos = r.probe ? r.probe(t) : null;
        for (let i = 0; i < r.tripCount; i++) {
          const at = tw.tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch + (r.tripLag ? r.tripLag(i) : 0), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
          if (!at) continue;
          const [lng, lat] = tw.toLngLat(at.x, at.y), p = tw.lngLatToPack(lng, lat);
          if (Math.abs(p.x - o.x) > half / 3) continue;
          cpu[at.run][p.y > o.y ? 0 : 1]++;
          if (at.run === 1 && shotT === null && Math.abs(p.x - o.x) < half / 6) shotT = t;
          if (pos) { gpuN++; gpuWorst = Math.max(gpuWorst, Math.hypot(pos[i * 2] - at.x, pos[i * 2 + 1] - at.y)); }
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    return { cpu, gpuWorst, gpuN, tripCount: r.tripCount, shotT };
  }, [RC, HALF]);
  console.log('run', JSON.stringify(run));
  const [r0, r1] = run.cpu;
  check(r0[0] + r0[1] > 0 && r1[0] + r1[1] > 0, 'trains of both runs were drawn on the middle stretch', JSON.stringify(run.cpu));
  check(r0[0] === 0, 'eastbound trains drawn on the south bow only', `north ${r0[0]} south ${r0[1]}`);
  check(r1[1] === 0, 'westbound trains drawn on the north bow only', `north ${r1[0]} south ${r1[1]}`);
  check(run.gpuN > 0 && run.gpuWorst < 1, 'the GPU draws them where the CPU does', `worst ${run.gpuWorst.toFixed(3)} of ${run.gpuN}`);

  // a picture: the moment a westbound train was mid-way, the clock stopped (a redraw needs the camera to move a hair)
  await page.evaluate((t) => { const c = window.tw.clock; c.setRunning(false); if (t !== null) c.set(t); }, run.shotT);
  for (const dx of [0, 1e-7]) {
    await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 15.6 }), [RC[0] + dx, RC[1]]);
    await settle(1000);
  }
  await page.screenshot({ path: join(OUT, `pair-${TAG}.png`) });
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  console.log('FAIL', e?.stack ?? e);
  fail++;
} finally {
  await chrome.close?.();
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
