// T-1644: pan at several views with the renderer's own flags: gather on, gather off (ranges), culling off.
// Per row: GPU % (typeperf), overlay GPU ms (timer), CPU ms a frame in renderer.draw and in pickStrokes, ranges, instances.
// argv: gpu pid, views (comma list of names), configs (comma list of gather|ranges|none)
import { execFile } from 'node:child_process';
import { playwright } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';
const PORT = process.env.PORT ?? '6124', CDP = process.env.CDP ?? '9345';
const GPU_PID = process.argv[2];
const VIEWNAMES = (process.argv[3] ?? 'tokyo15').split(',');
const CONFIGS = (process.argv[4] ?? 'gather,ranges').split(',');
const SAMPLES = 5;
const VIEWS = {
  japan: { center: [137.5, 37.0], zoom: 4.6 }, kanto9: { center: [139.75, 35.68], zoom: 9.5 },
  tokyo11: { center: [139.75, 35.68], zoom: 11 }, tokyo12: { center: [139.767, 35.681], zoom: 12 }, tokyo13: { center: [139.767, 35.681], zoom: 13 }, tokyo14: { center: [139.767, 35.681], zoom: 14 },
  tokyo15: { center: [139.767, 35.681], zoom: 15 }, tokyo16: { center: [139.767, 35.681], zoom: 16 }, shinjuku16: { center: [139.7005, 35.6905], zoom: 16 }, shinjuku17: { center: [139.7005, 35.6905], zoom: 17 }, osaka15: { center: [135.4959, 34.7025], zoom: 15 },
};
const { chromium } = await playwright();
const browser = await chromium.connectOverCDP(`http://127.0.0.1:${CDP}`);
const page = browser.contexts()[0].pages().find((p) => p.url().includes(PORT));
const typeperf = () => new Promise((res) => execFile('typeperf', [`\\GPU Engine(pid_${GPU_PID}*engtype_3D)\\Utilization Percentage`, '-si', '1', '-sc', String(SAMPLES)], (err, out) => {
  const rows = out.split(/\r?\n/).filter((l) => /^"\d/.test(l)).map((l) => l.split('","').slice(1).reduce((s, v) => s + (parseFloat(v.replace(/"/g, '')) || 0), 0));
  res(rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : NaN);
}));
await page.evaluate(() => {
  if (window.__c) return;
  const tw = window.tw, r = tw.renderer;
  const c = { ov: [], pick: [], draw: [], ranges: 0, calls: 0 };
  window.__c = c;
  const tm = r.timer, ob = tm.begin.bind(tm);
  tm.begin = () => { const q = tm.query; ob(); if (q && tm.query !== q && tm.ms != null) c.ov.push(tm.ms); };
  const op = r.pickStrokes.bind(r);
  r.pickStrokes = (...a) => { const t0 = performance.now(); const p = op(...a); c.pick.push(performance.now() - t0); if (p?.ranges) c.ranges += p.ranges.length / 2; return p; };
  const od = r.draw.bind(r);
  r.draw = (...a) => { const t0 = performance.now(); const x = od(...a); c.draw.push(performance.now() - t0); return x; };
});
const f = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '-' : v.toFixed(d));
await page.evaluate((m) => { window.__mlv = m === undefined ? undefined : Number(m); }, process.env.MINL);
const out = [];
for (const vn of VIEWNAMES) for (const cfg of CONFIGS) {
  await page.evaluate(([cfg, view]) => {
    const tw = window.tw, r = tw.renderer;
    window.__minL = window.__mlv === undefined ? undefined : () => window.__mlv;
    r.strokeCull = cfg !== 'none';
    r.strokeGather = cfg === 'gather';
    tw.display.value = { ...tw.display.value, projection: 'globe', hillshade: true };
    tw.clock.setRunning(false);
    tw.map.jumpTo(view);
    tw.overlay.invalidate();
  }, [cfg, VIEWS[vn]]);
  await page.waitForTimeout(3500);
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
  const a = await page.evaluate(() => { const c = window.__c; c.ov.length = c.pick.length = c.draw.length = 0; c.ranges = 0; globalThis.__tb = globalThis.__tf = 0; return { t: performance.now(), ov: window.tw.overlay.frames }; });
  const util = await typeperf();
  const b = await page.evaluate(() => { const c = window.__c, med = (x) => { const s = [...x].sort((u, v) => u - v); return s.length ? s[s.length >> 1] : null; }, avg = (x) => (x.length ? x.reduce((s, v) => s + v, 0) / x.length : null); return { t: performance.now(), ov: window.tw.overlay.frames, ovMed: med(c.ov), drawAvg: avg(c.draw), drawMed: med(c.draw), pickAvg: avg(c.pick) * 2, picks: c.pick.length, tb: (globalThis.__tb ?? 0) / Math.max(1, c.pick.length / 2), tf: (globalThis.__tf ?? 0) / Math.max(1, c.pick.length / 2), ranges: c.ranges, inst: window.tw.renderer.strokeInstances }; });
  const s = (b.t - a.t) / 1000;
  const row = { view: vn, cfg, util, fps: (b.ov - a.ov) / s, ovMed: b.ovMed, drawAvg: b.drawAvg, pick: b.pickAvg, tb: b.tb, tf: b.tf, ranges: b.ranges / Math.max(1, b.picks / 2), inst: b.inst };
  out.push(row);
  row.minl = process.env.MINL; console.log(JSON.stringify(row));
  await page.evaluate(() => { window.__pan = false; window.tw.map.stop(); });
  await page.waitForTimeout(500);
}
await page.evaluate(() => { const r = window.tw.renderer; r.strokeCull = true; r.strokeGather = true; });
console.log('| view | config | GPU % | draws/s | overlay GPU ms | renderer.draw CPU ms | pickStrokes CPU ms | ranges a frame | instances |\n|---|---|---|---|---|---|---|---|---|');
for (const r of out) console.log(`| ${r.view} | ${r.cfg} | ${f(r.util, 1)} | ${f(r.fps, 0)} | ${f(r.ovMed)} | ${f(r.drawAvg)} | ${f(r.pick)} | ${f(r.ranges, 0)} | ${r.inst} |`);
process.exit(0);
