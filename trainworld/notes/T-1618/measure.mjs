// T-1618: one run, the whole table. Connects to hold.mjs's Chrome (CDP 9338), GPU process pid as argv[2].
// Per state: the GPU process's 3D engine use (typeperf, the Task Manager figure), frames per second
// (rAF frames, MapLibre renders, overlay draws, demand-canvas draws) and GPU ms per draw from timer queries
// on MapLibre's context, the overlay's and the demand canvas's.
import { execFile } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

// PORT, CDP and OUT as hold.mjs's
const PORT = process.env.PORT ?? '6158', CDP = process.env.CDP ?? '9338';
const OUT = process.env.OUT ?? 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/f28006bc-4241-4116-9970-a65fc772bb95/scratchpad/t1618/';
const GPU_PID = process.argv[2];
const ONLY = process.argv[3] ? new RegExp(process.argv[3]) : null;
const SAMPLES = 6;
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));

const typeperf = () => new Promise((res) => execFile('typeperf', [`\\GPU Engine(pid_${GPU_PID}*engtype_3D)\\Utilization Percentage`, '-si', '1', '-sc', String(SAMPLES)], (err, out) => {
  const rows = out.split(/\r?\n/).filter((l) => /^"\d/.test(l)).map((l) => l.split('","').slice(1).reduce((s, v) => s + (parseFloat(v.replace(/"/g, '')) || 0), 0));
  res(rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : NaN);
}));

await page.evaluate(() => {
  if (window.__m) return;
  const tw = window.tw, map = tw.map, m = { raf: 0, rafFrames: 0, lastTs: -1, mapGpu: [], ovGpu: [], dmGpu: [] };
  window.__m = m;
  const oraf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) => oraf((ts) => { m.raf++; if (ts !== m.lastTs) { m.lastTs = ts; m.rafFrames++; } cb(ts); });
  const gl = map.painter.context.gl, ext = gl.getExtension('EXT_disjoint_timer_query_webgl2'), pending = [];
  m.ext = !!ext;
  const orig = map._render;
  map._render = function (ts) {
    let q = null;
    if (ext && pending.length < 16) { q = gl.createQuery(); gl.beginQuery(ext.TIME_ELAPSED_EXT, q); }
    const r = orig.call(this, ts);
    if (q) { gl.endQuery(ext.TIME_ELAPSED_EXT); pending.push(q); }
    return r;
  };
  setInterval(() => {
    while (pending.length && gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)) {
      const q = pending.shift();
      if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) m.mapGpu.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      gl.deleteQuery(q);
    }
  }, 30);
  const hook = (t, arr) => { const ob = t.begin.bind(t); t.begin = () => { const q = t.query; ob(); if (q && t.query !== q && t.ms != null) arr.push(t.ms); }; };
  hook(tw.renderer.timer, m.ovGpu);
  hook(window.twViews.overlay.timer, m.dmGpu);
  m.minChanges = 0;
  tw.clock.minute.subscribe(() => { m.minChanges++; });
  m.snap = () => ({ t: performance.now(), minChanges: m.minChanges, raf: m.raf, rafFrames: m.rafFrames, ov: tw.overlay.frames, map: tw.overlay.mapFrames, dm: window.twViews.overlay.draws, cache: tw.renderer.mainCache?.redraws ?? 0 });
  // toggles: each returns its undo
  const disp = (k, v) => { const was = tw.display.value[k]; tw.display.value = { ...tw.display.value, [k]: v }; return () => { tw.display.value = { ...tw.display.value, [k]: was }; }; };
  const r = tw.renderer, dov = window.twViews.overlay;
  m.toggles = {
    trains: () => disp('trains', false),
    strokes: () => { const a = r.track, b = r.lines, c = r.strokeCache; r.track = r.lines = null; r.strokeCache = false; tw.overlay.invalidate(); return () => { r.track = a; r.lines = b; r.strokeCache = c; tw.overlay.invalidate(); }; },
    // T-1638: the strokes drawn every frame, as before the cache
    nocache: () => { r.strokeCache = false; return () => { r.strokeCache = true; tw.overlay.invalidate(); }; },
    // T-1639: every stroke drawn, unsimplified and unculled, as before it
    old: () => { r.strokeCull = r.strokeLod = false; tw.overlay.invalidate(); return () => { r.strokeCull = r.strokeLod = true; tw.overlay.invalidate(); }; },
    nolod: () => { r.strokeLod = false; tw.overlay.invalidate(); return () => { r.strokeLod = true; tw.overlay.invalidate(); }; },
    overlayDraw: () => { r.draw = () => {}; return () => { delete r.draw; tw.overlay.invalidate(); }; },
    names: () => disp('stationNames', false),
    capacity: () => disp('capacity', false),
    bubbles: () => { const was = dov.showBubbles; dov.setBubblesVisible(false); return () => dov.setBubblesVisible(was); },
    hillshade: () => disp('hillshade', false),
    realNetwork: () => disp('realNetwork', false),
    basemapLabels: () => disp('basemapLabels', false),
    basemap: () => {
      const was = [];
      for (const l of map.getStyle().layers) if (!l.id.startsWith('tw-')) { was.push([l.id, map.getLayoutProperty(l.id, 'visibility') ?? 'visible']); map.setLayoutProperty(l.id, 'visibility', 'none'); }
      return () => { for (const [id, v] of was) map.setLayoutProperty(id, 'visibility', v); };
    },
    demandCanvas: () => { const d = dov.draw; dov.draw = () => {}; dov.canvas.style.display = 'none'; return () => { delete dov.draw; dov.canvas.style.display = ''; dov.invalidate(); }; },
    dpr2: () => { map.setPixelRatio(2); map.resize(); return () => { map.setPixelRatio(1); map.resize(); }; },
    flat: () => disp('projection', 'flat'),
    // T-1640: redraw every frame while the clock runs, as before it
    nothrottle: () => { tw.overlay.throttle = false; tw.overlay.invalidate(); return () => { tw.overlay.throttle = true; tw.overlay.invalidate(); }; },
  };
  m.undo = [];
});

const info = await page.evaluate(() => {
  const tw = window.tw, map = tw.map, o = tw.overlay.gl, d = window.twViews.overlay.gl, g = map.painter.context.gl;
  const attrs = (x) => { const a = x.getContextAttributes(); return { antialias: a.antialias, stencil: a.stencil, samples: x.getParameter(x.SAMPLES), w: x.drawingBufferWidth, h: x.drawingBufferHeight }; };
  return { ext: window.__m.ext, focus: document.hasFocus(), dpr: devicePixelRatio, mapPixelRatio: map.getPixelRatio(), map: attrs(g), overlay: attrs(o), demand: attrs(d), trips: tw.renderer.tripCount, lines: tw.world.value.save.lines.length, stations: tw.world.value.save.stations.length, display: { ...tw.display.value }, layers: map.getStyle().layers.length, gpu: g.getParameter(g.getExtension('WEBGL_debug_renderer_info')?.UNMASKED_RENDERER_WEBGL ?? g.RENDERER) };
});
console.log(JSON.stringify(info));

const JAPAN = { center: [137.5, 37.0], zoom: 4.6 }, TOKYO = { center: [139.75, 35.68], zoom: 11 }, Z15 = { center: [139.767, 35.681], zoom: 15 }, Z95 = { center: [139.75, 35.68], zoom: 9.5 }, Z13 = { center: [139.767, 35.681], zoom: 13 };
// [label, view, clock (null paused, else speed index), toggles, pan px]
const CASES = [
  ['idle', JAPAN, null, [], 0],
  ['run 1x', JAPAN, 0, [], 0],
  ['run fastest', JAPAN, 2, [], 0],
  ['pan, paused', JAPAN, null, [], 1500],
  ['pan, 1x', JAPAN, 0, [], 1500],
  ...['trains', 'strokes', 'overlayDraw', 'names', 'capacity', 'bubbles', 'demandCanvas', 'hillshade', 'realNetwork', 'basemapLabels', 'basemap'].map((k) => [`1x, no ${k}`, JAPAN, 0, [k], 0]),
  ['1x, ours all off', JAPAN, 0, ['overlayDraw', 'names', 'capacity', 'demandCanvas'], 0],
  ['1x, only ours', JAPAN, 0, ['basemap', 'hillshade'], 0],
  ...['hillshade', 'basemap', 'names', 'overlayDraw', 'capacity'].map((k) => [`pan paused, no ${k}`, JAPAN, null, [k], 1500]),
  ['1x, flat map', JAPAN, 0, ['flat'], 0],
  ['1x, dpr 2', JAPAN, 0, ['dpr2'], 0],
  ['Tokyo idle', TOKYO, null, [], 0],
  ['Tokyo 1x', TOKYO, 0, [], 0],
  ['Tokyo fastest', TOKYO, 2, [], 0],
  ['Tokyo 1x, no trains', TOKYO, 0, ['trains'], 0],
  ['Tokyo pan, 1x', TOKYO, 0, [], 1500],
  // T-1638's before: each of these with the stroke cache off
  ...['run 1x', 'run fastest', 'pan, paused', 'pan, 1x'].map((l) => [`${l}, no cache`, JAPAN, l.includes('fastest') ? 2 : l.includes('paused') ? null : 0, ['nocache'], l.startsWith('pan') ? 1500 : 0]),
  ...['1x', 'fastest', 'pan, 1x'].map((l) => [`Tokyo ${l}, no cache`, TOKYO, l === 'fastest' ? 2 : 0, ['nocache'], l.startsWith('pan') ? 1500 : 0]),
  // T-1639: panning with the strokes simplified and culled, and without (old), at three zooms
  ['Tokyo pan, paused', TOKYO, null, [], 1500],
  ['z15 1x', Z15, 0, [], 0],
  ['z15 pan, 1x', Z15, 0, [], 1500],
  ...[['pan, paused', JAPAN, null], ['pan, 1x', JAPAN, 0], ['Tokyo pan, paused', TOKYO, null], ['Tokyo pan, 1x', TOKYO, 0], ['z15 pan, 1x', Z15, 0]].flatMap(([l, v, sp]) => [[`${l}, old`, v, sp, ['old'], 1500], [`${l}, culled only`, v, sp, ['nolod'], 1500]]),
  // T-1640: the redraw rate while the camera is still, with the throttle on and off
  ...[['Japan z4.6 1x', JAPAN, 0], ['Japan z4.6 fastest', JAPAN, 2], ['Kanto z9.5 1x', Z95, 0], ['Tokyo z11 1x', TOKYO, 0], ['Tokyo z11 fastest', TOKYO, 2], ['Tokyo z13 1x', Z13, 0]].flatMap(([l, v, sp]) => [[`T1640 ${l}`, v, sp, [], 0], [`T1640 ${l}, no throttle`, v, sp, ['nothrottle'], 0]]),
];

// Anita's conditions (T-1617): globe, Terrain on
await page.evaluate(() => { const tw = window.tw; tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true }; });
const rows = [];
for (const [label, view, speed, toggles, pan] of CASES) {
  if (ONLY && !ONLY.test(label)) continue;
  await page.evaluate(([view, speed, toggles]) => {
    const tw = window.tw, m = window.__m;
    tw.map.jumpTo(view);
    if (speed === null) tw.clock.setRunning(false);
    else { tw.clock.setSpeed(speed); tw.clock.setRunning(true); }
    for (const k of toggles) m.undo.push(m.toggles[k]());
  }, [view, speed, toggles]);
  await page.waitForTimeout(4000); // tiles and placers settle
  // T-1639: panned by a jump each frame (T-1638's panBy rows did not move), `px` CSS px east over 12 s
  if (pan) await page.evaluate((px) => {
    const map = window.tw.map, c = map.getCenter(), z = map.getZoom(), t0 = performance.now();
    window.__pan = true;
    const step = () => {
      if (!window.__pan) return;
      const t = Math.min(1, (performance.now() - t0) / 12000);
      map.jumpTo({ center: [c.lng + (px * t * 360) / (512 * 2 ** z), c.lat] });
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, pan);
  await page.waitForTimeout(800);
  const a = await page.evaluate(() => { const m = window.__m; m.mapGpu.length = m.ovGpu.length = m.dmGpu.length = 0; return m.snap(); });
  const util = await typeperf();
  const b = await page.evaluate(() => { const m = window.__m, avg = (x) => (x.length ? x.reduce((s, v) => s + v, 0) / x.length : null); return { ...m.snap(), mapGpu: avg(m.mapGpu), ovGpu: avg(m.ovGpu), dmGpu: avg(m.dmGpu), nMap: m.mapGpu.length, nOv: m.ovGpu.length, trips: window.tw.renderer.tripCount, ticking: window.tw.clock.ticking, moving: window.tw.overlay.moving, inst: window.tw.renderer.strokeInstances, lodMs: window.tw.renderer.lodMs }; });
  const s = (b.t - a.t) / 1000, ps = (k) => (b[k] - a[k]) / s;
  const row = { label, minFps: ps('minChanges'), util, rafFps: ps('rafFrames'), mapFps: ps('map'), ovFps: ps('ov'), dmFps: ps('dm'), cacheFps: ps('cache'), mapGpu: b.mapGpu, ovGpu: b.ovGpu, dmGpu: b.dmGpu, trips: b.trips, ticking: b.ticking, moving: b.moving, inst: b.inst, lodMs: b.lodMs };
  rows.push(row);
  console.log(JSON.stringify(row));
  await page.evaluate(() => { const tw = window.tw, m = window.__m; window.__pan = false; tw.map.stop(); while (m.undo.length) m.undo.pop()(); });
  await page.waitForTimeout(500);
}
await page.evaluate(() => window.tw.clock.setRunning(false));

const f = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '-' : v.toFixed(d));
const md = ['| state | GPU % | frames/s (rAF) | MapLibre renders/s | MapLibre GPU ms | overlay draws/s | overlay GPU ms | stroke redraws/s | stroke instances | demand draws/s | demand GPU ms | minute changes/s |', '|---|---|---|---|---|---|---|---|---|---|---|---|',
  ...rows.map((r) => `| ${r.label} | ${f(r.util)} | ${f(r.rafFps, 0)} | ${f(r.mapFps, 0)} | ${f(r.mapGpu, 2)} | ${f(r.ovFps, 0)} | ${f(r.ovGpu, 2)} | ${f(r.cacheFps, 0)} | ${r.inst ?? '-'} | ${f(r.dmFps, 0)} | ${f(r.dmGpu, 2)} | ${f(r.minFps, 1)} |`)].join('\n');
await writeFile(OUT + (ONLY ? 'table-part.md' : 'table.md'), JSON.stringify(info, null, 1) + '\n\n' + md + '\n');
console.log(md);
process.exit(0);
