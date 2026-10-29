// T-1644: panning at zoom 15 (a jumpTo per frame), parts of the overlay's draw switched off by skipping their GL draw calls.
// Per row: overlay GPU ms (timer query, median and mean), MapLibre GPU ms, frames/s, the GPU process's 3D use (typeperf).
// argv: gpu pid, row filter regex (optional). env VIEW, CLOCK=1 to run the clock at 1x.
import { execFile } from 'node:child_process';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const PORT = process.env.PORT ?? '6124', CDP = process.env.CDP ?? '9344';
const GPU_PID = process.argv[2];
const ONLY = process.argv[3] ? new RegExp(process.argv[3]) : null;
const SAMPLES = 5;
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));
const typeperf = () => new Promise((res) => execFile('typeperf', [`\\GPU Engine(pid_${GPU_PID}*engtype_3D)\\Utilization Percentage`, '-si', '1', '-sc', String(SAMPLES)], (err, out) => {
  const rows = out.split(/\r?\n/).filter((l) => /^"\d/.test(l)).map((l) => l.split('","').slice(1).reduce((s, v) => s + (parseFloat(v.replace(/"/g, '')) || 0), 0));
  res(rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : NaN);
}));
const VIEWS = { tokyo15: { center: [139.767, 35.681], zoom: 15 }, tokyo14: { center: [139.767, 35.681], zoom: 14 }, tokyo13: { center: [139.767, 35.681], zoom: 13 }, shinjuku16: { center: [139.7005, 35.6905], zoom: 16 }, osaka15: { center: [135.4959, 34.7025], zoom: 15 } };
const VIEW = process.env.VIEW ?? 'tokyo15';

await page.evaluate(() => {
  if (window.__p) return;
  const tw = window.tw, r = tw.renderer, gl = tw.overlay.gl, map = tw.map;
  const p = { ov: [], mp: [], skip: { progs: new Set(), pred: null, only: null }, st: { kind: 0, mode: 0, casing: 0, cap: 0, full: 0 }, count: {}, collect: false };
  window.__p = p;
  const t = r.timer, ob = t.begin.bind(t);
  t.begin = () => { const q = t.query; ob(); if (q && t.query !== q && t.ms != null) p.ov.push(t.ms); };
  const mgl = map.painter.context.gl, ext = mgl.getExtension('EXT_disjoint_timer_query_webgl2'), pending = [];
  const orig = map._render;
  map._render = function (ts) {
    let q = null;
    if (ext && pending.length < 16) { q = mgl.createQuery(); mgl.beginQuery(ext.TIME_ELAPSED_EXT, q); }
    const res = orig.call(this, ts);
    if (q) { mgl.endQuery(ext.TIME_ELAPSED_EXT); pending.push(q); }
    return res;
  };
  setInterval(() => {
    while (pending.length && mgl.getQueryParameter(pending[0], mgl.QUERY_RESULT_AVAILABLE)) {
      const q = pending.shift();
      if (!mgl.getParameter(ext.GPU_DISJOINT_EXT)) p.mp.push(mgl.getQueryParameter(q, mgl.QUERY_RESULT) / 1e6);
      mgl.deleteQuery(q);
    }
  }, 30);
  const names = new Map([[r.strokeProg, 'stroke'], [r.stationProg, 'station'], [r.bridgeProg, 'bridge'], [r.trainProg, 'train']]);
  let cur = null;
  const ouse = gl.useProgram.bind(gl), odi = gl.drawArraysInstanced.bind(gl), oda = gl.drawArrays.bind(gl);
  const ids = new Map();
  const label = () => names.get(cur) ?? (ids.has(cur) ? ids.get(cur) : (ids.set(cur, 'other' + ids.size), ids.get(cur)));
  gl.useProgram = (pr) => { cur = pr; return ouse(pr); };
  const su = r.strokeU, st = p.st;
  for (const fn of ['uniform1i', 'uniform1f']) {
    const o = gl[fn].bind(gl);
    gl[fn] = (loc, v, ...rest) => { if (loc === su.u_kind) st.kind = v; else if (loc === su.u_lane_mode) st.mode = v; else if (loc === su.u_casing) st.casing = v; else if (loc === su.u_cap) st.cap = v; else if (loc === su.u_full_pass) st.full = v; return o(loc, v, ...rest); };
  }
  const handle = (count, inst, real) => {
    const l = label();
    let key = l;
    if (cur === r.strokeProg) {
      key = `stroke k${st.kind} m${st.mode} c${st.casing > 0 ? 1 : 0} cap${st.cap > 0 ? 1 : 0}`;
      if (p.skip.pred && p.skip.pred(st)) return;
    }
    if (p.skip.progs.has(l) || (p.skip.only && l !== p.skip.only)) return;
    if (p.collect) { const e = (p.count[key] ??= { calls: 0, inst: 0 }); e.calls++; e.inst += inst; }
    real();
  };
  gl.drawArraysInstanced = (mode, first, count, inst) => handle(count, inst, () => odi(mode, first, count, inst));
  gl.drawArrays = (mode, first, count) => handle(count, 1, () => oda(mode, first, count));
  p.skipProg = (...x) => () => { for (const k of x) p.skip.progs.add(k); return () => { for (const k of x) p.skip.progs.delete(k); }; };
  p.skipPred = (f) => () => { p.skip.pred = f; return () => { p.skip.pred = null; }; };
  p.only = (k) => () => { p.skip.only = k; return () => { p.skip.only = null; }; };
  p.vp1 = () => { const vb = gl.viewport.bind(gl); gl.viewport = () => vb(0, 0, 1, 1); return () => { delete gl.viewport; }; };
  p.undo = [];
  p.setSamples = (n) => () => { const c = r.mainCache; const was = c.samples; c.samples = n; c.w = 0; return () => { c.samples = was; c.w = 0; }; };
});

const ROWS = [
  ['all', 'null'],
  ['no strokes', 'p.skipProg("stroke")'],
  ['no stations or bridges', 'p.skipProg("station","bridge")'],
  ['no trains', 'p.skipProg("train")'],
  ['strokes only', 'p.only("stroke")'],
  ['stations only', 'p.only("station")'],
  ['trains only', 'p.only("train")'],
  ['nothing drawn', 'p.only("none")'],
  ['no casing passes', 'p.skipPred((s) => s.casing > 0)'],
  ['no fill passes', 'p.skipPred((s) => s.casing <= 0 && s.cap <= 0)'],
  ['no through-box layer', 'p.skipPred((s) => s.cap > 0)'],
  ['no track strokes', 'p.skipPred((s) => s.kind === 0)'],
  ['no line strokes', 'p.skipPred((s) => s.kind === 1)'],
  ['1x1 viewport', '() => p.vp1()'],
  ['strokes only, 1x1 viewport', '() => { const u1 = p.only("stroke")(), u2 = p.vp1(); return () => { u1(); u2(); }; }'],
  ['cache samples 2', 'p.setSamples(2)'],
  ['cache samples 1', 'p.setSamples(1)'],
  ['all (again)', 'null'],
];
await page.evaluate(([v]) => { const tw = window.tw; tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true }; tw.map.jumpTo(v); tw.clock.setRunning(false); }, [VIEWS[VIEW]]);
await page.waitForTimeout(4000);
const f = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '-' : v.toFixed(d));
const rows = [];
for (const [label, setup] of ROWS) {
  if (ONLY && !ONLY.test(label)) continue;
  await page.evaluate(([setup, view]) => {
    const p = window.__p, tw = window.tw;
    tw.map.jumpTo(view);
    const mk = (0, eval)(`(p) => ${setup}`)(p);
    p.undo.push(mk ? mk() : () => {});
  }, [setup, VIEWS[VIEW]]);
  await page.waitForTimeout(1500);
  await page.evaluate((px) => {
    const map = window.tw.map, c = map.getCenter(), z = map.getZoom(), t0 = performance.now();
    window.__pan = true;
    const step = () => {
      if (!window.__pan) return;
      const t = ((performance.now() - t0) / 12000) % 1;
      map.jumpTo({ center: [c.lng + (px * t * 360) / (512 * 2 ** z), c.lat] });
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, 1200);
  await page.waitForTimeout(1500);
  const a = await page.evaluate(() => { const p = window.__p; p.ov.length = p.mp.length = 0; p.count = {}; p.collect = true; return { t: performance.now(), ov: window.tw.overlay.frames, map: window.tw.overlay.mapFrames }; });
  const util = await typeperf();
  const b = await page.evaluate(() => { const p = window.__p; p.collect = false; const med = (x) => { const s = [...x].sort((u, v) => u - v); return s.length ? s[s.length >> 1] : null; }, avg = (x) => (x.length ? x.reduce((s, v) => s + v, 0) / x.length : null); return { t: performance.now(), ov: window.tw.overlay.frames, ovMed: med(p.ov), ovAvg: avg(p.ov), mpMed: med(p.mp), mpAvg: avg(p.mp), n: p.ov.length, count: p.count, inst: window.tw.renderer.strokeInstances }; });
  const s = (b.t - a.t) / 1000;
  const row = { label, util, fps: (b.ov - a.ov) / s, ovMed: b.ovMed, ovAvg: b.ovAvg, mpMed: b.mpMed, mpAvg: b.mpAvg, n: b.n, inst: b.inst };
  rows.push(row);
  console.log(JSON.stringify(row));
  if (label === 'all') console.log(JSON.stringify(b.count));
  await page.evaluate(() => { window.__pan = false; window.tw.map.stop(); const p = window.__p; while (p.undo.length) p.undo.pop()(); });
  await page.waitForTimeout(600);
}
console.log('| row | GPU % | overlay draws/s | overlay GPU ms (median / mean) | MapLibre GPU ms (median) |\n|---|---|---|---|---|');
for (const r of rows) console.log(`| ${r.label} | ${f(r.util, 1)} | ${f(r.fps, 0)} | ${f(r.ovMed)} / ${f(r.ovAvg)} | ${f(r.mpMed)} |`);
process.exit(0);
