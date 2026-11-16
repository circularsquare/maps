// T-076: what playing costs on the real New York save, and how far the fastest train on screen moves a frame.
// Per case (zoom, clock speed, frame cap): CPU of all this Chrome's processes (% of one core), the GPU's 3D
// engine use, overlay draws/s, and per frame the screen step of every train in view (px a 60 Hz frame).
import { execFile } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, startVite, REAL_SAVE } from 'file:///C:/Users/anita/projects/maps/.tw-trees/perf/trainworld/checks/lib.mjs';

const PORT = process.env.PORT ?? '6121';
const OUT = process.env.OUT;
const ONLY = process.env.ONLY ? new RegExp(process.env.ONLY) : null;
const log = (s) => console.log(`${new Date().toISOString().slice(11, 19)} ${s}`);
const ps = (cmd) => new Promise((res) => execFile('powershell', ['-NoProfile', '-Command', cmd], { maxBuffer: 1e7 }, (e, out) => res(out)));
let vite = null;
if (!process.env.NOVITE) { vite = await startVite(Number(PORT)); log('vite up'); }
const { ctx, profile, close } = await launchChrome({ prefix: 'tw-perf-t076-', size: [1920, 1080] });
const prof = profile.replace(/\//g, '\\');
// CPU seconds of every process of this Chrome (its profile dir on the command line), and the GPU process pid
const cpu = async () => {
  const out = await ps(`Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.CommandLine -like '*${profile.split(/[\\/]/).pop()}*' } | ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, ($_.KernelModeTime + $_.UserModeTime), ($(if ($_.CommandLine -match '--type=([a-z-]+)') { $matches[1] } else { 'browser' })) }`);
  const rows = out.trim().split(/\r?\n/).filter(Boolean).map((l) => { const [pid, t, type] = l.trim().split(' '); return { pid: +pid, s: +t / 1e7, type }; });
  return rows;
};
const gpuUtil = (pid, n) => new Promise((res) => execFile('typeperf', [`\\GPU Engine(pid_${pid}*engtype_3D)\\Utilization Percentage`, '-si', '1', '-sc', String(n)], (err, out) => {
  const rows = (out ?? '').split(/\r?\n/).filter((l) => /^"\d/.test(l)).map((l) => l.split('","').slice(1).reduce((s, v) => s + (parseFloat(v.replace(/"/g, '')) || 0), 0));
  res(rows.length ? rows.reduce((a, b) => a + b, 0) / rows.length : NaN);
}));
const page = ctx.pages()[0] ?? await ctx.newPage();
page.on('pageerror', (e) => log('pageerror ' + e.message));
await page.goto(`http://localhost:${PORT}/?debug=1&new=1&paused=1&demandWorkers=2`);
await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
await importSave(page, await readFile(REAL_SAVE));
await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 600000 });
log('save in');
await demandFinal(page, { timeout: 1800000, polling: 2000, lines: 100, held: true });
log('demand final (fresh)');
// the workers quiet: the renderer process under 30% of a core over 5 s
for (let i = 0; i < 60; i++) {
  const a = await cpu(); await page.waitForTimeout(5000); const b = await cpu();
  const r = b.filter((p) => p.type === 'renderer').reduce((s, p) => s + p.s - (a.find((q) => q.pid === p.pid)?.s ?? p.s), 0) / 5;
  log(`renderer ${(100 * r).toFixed(0)}%`);
  if (r < 0.3) break;
}
await page.bringToFront();
await page.evaluate(() => window.dispatchEvent(new Event('focus')));
await page.evaluate(async () => {
  const tw = window.tw, m = { steps: [] };
  window.__m = m;
  // every train in view: its screen step over one 60 Hz frame of game time, px
  m.sample = () => {
    const r = tw.renderer, n = r.net, d = r.trips, N = r.tripCount;
    if (!n || !N) return null;
    const uTime = tw.clock.now() - r.epoch, dt = tw.clock.rate() / 60, W = tw.overlay.cssW, H = tw.overlay.cssH;
    const prof = (p, tau) => {
      const first = n.meta[p * 4], count = n.meta[p * 4 + 1];
      if (!count) return 0;
      let lo = first, hi = first + count - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (n.phases[mid * 4] <= tau) lo = mid; else hi = mid - 1; }
      const t = n.phases[lo * 4], s = n.phases[lo * 4 + 1], v = n.phases[lo * 4 + 2], a = n.phases[lo * 4 + 3];
      let tt = Math.max(0, tau - t);
      if (a < 0) tt = Math.min(tt, v / -a);
      return s + v * tt + 0.5 * a * tt * tt;
    };
    const pos = (p, s) => {
      const line = Math.floor(p / 6), run = Math.floor(p / 3) % 2, dd = run === 0 ? s : n.meta[p * 4 + 3] - s;
      const first = n.lineTable[line * 4], count = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
      const f = Math.max(0, Math.min(count - 1.0001, dd / step)), i = Math.floor(f), k = f - i, j = first + i;
      return tw.overlay.toScreen(n.samples[j * 2] + (n.samples[j * 2 + 2] - n.samples[j * 2]) * k, n.samples[j * 2 + 1] + (n.samples[j * 2 + 3] - n.samples[j * 2 + 1]) * k);
    };
    const steps = [];
    for (let i = 0; i < N; i++) {
      const p = d[i * 2], tau = uTime - d[i * 2 + 1];
      if (n.meta[p * 4 + 1] < 0.5 || tau < 0 || tau > n.meta[p * 4 + 2]) continue;
      const s0 = prof(p, tau), s1 = prof(p, tau + dt);
      const a = pos(p, s0);
      if (!(a[0] >= 0 && a[0] <= W && a[1] >= 0 && a[1] <= H)) continue;
      const b = pos(p, s1);
      steps.push(Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
    steps.sort((x, y) => y - x);
    return { n: steps.length, max: steps[0] ?? 0, p10: steps[Math.floor(steps.length * 0.1)] ?? 0, moving: steps.filter((v) => v > 0.01).length };
  };
  m.origWait = tw.renderer.stillWaitMs.bind(tw.renderer);
  m.cap = (ms) => { tw.renderer.stillWaitMs = ms ? (...a) => Math.max(ms, m.origWait(...a)) : m.origWait; tw.overlay.invalidate(); };
});
const ZOOMS = (process.env.ZOOMS ?? '10,11,12,13,14').split(',').map(Number);
const SPEEDS = [0, 1, 2];
const CAPS = (process.env.CAPS ?? '0,33').split(',').map(Number);
const SECS = Number(process.env.SECS ?? 8);
const CENTER = [-73.985, 40.758];
const cases = [['paused z12', 12, null, 0]];
for (const z of ZOOMS) for (const sp of SPEEDS) for (const cap of CAPS) cases.push([`z${z} ${['1x', '4x', '16x'][sp]}${cap ? ` cap ${cap} ms` : ''}`, z, sp, cap]);
const rows = [];
for (const [label, zoom, speed, cap] of cases) {
  if (ONLY && !ONLY.test(label)) continue;
  await page.evaluate(([zoom, speed, cap, CENTER]) => {
    const tw = window.tw;
    tw.map.jumpTo({ center: CENTER, zoom });
    if (speed === null) tw.clock.setRunning(false);
    else { tw.clock.setSpeed(speed); tw.clock.setRunning(true); }
    window.__m.cap(cap);
    window.dispatchEvent(new Event('focus'));
  }, [zoom, speed, cap, CENTER]);
  await page.waitForTimeout(3000);
  const procs = await cpu();
  const gpuPid = procs.find((p) => p.type === 'gpu-process')?.pid;
  const a = await page.evaluate(() => ({ t: performance.now(), ov: window.tw.overlay.frames }));
  // the train steps, sampled every 250 ms meanwhile
  await page.evaluate(() => { const m = window.__m; m.steps = []; m.timer = setInterval(() => { const s = m.sample(); if (s) m.steps.push(s); }, 250); });
  const g = gpuPid ? gpuUtil(gpuPid, SECS) : Promise.resolve(NaN);
  await page.waitForTimeout(SECS * 1000);
  const util = await g;
  const procs2 = await cpu();
  const b = await page.evaluate(() => { const m = window.__m; clearInterval(m.timer); return { t: performance.now(), ov: window.tw.overlay.frames, steps: m.steps, rate: window.tw.clock.rate(), capped: window.tw.clock.capped.value, ticking: window.tw.clock.ticking, focus: document.hasFocus(), busy: !!window.twDemand.client.waiting, trips: window.tw.renderer.tripCount }; });
  const secs = (b.t - a.t) / 1000;
  const by = {};
  for (const p of procs2) { const q = procs.find((x) => x.pid === p.pid); if (q) by[p.type] = (by[p.type] ?? 0) + (p.s - q.s); }
  const pct = Object.fromEntries(Object.entries(by).map(([k, v]) => [k, +(100 * v / secs).toFixed(1)]));
  const total = +Object.values(pct).reduce((s, v) => s + v, 0).toFixed(1);
  const st = b.steps, med = (f) => { const v = st.map(f).sort((x, y) => x - y); return v.length ? +v[Math.floor(v.length / 2)].toFixed(2) : null; };
  const row = { label, ticking: b.ticking, focus: b.focus, busy: b.busy, cpu: total, by: pct, gpu: +util.toFixed(1), ovFps: +((b.ov - a.ov) / secs).toFixed(1), rate: b.rate, capped: b.capped, inView: med((s) => s.n), moving: med((s) => s.moving), maxStep: med((s) => s.max), maxStepTop: st.length ? +Math.max(...st.map((s) => s.max)).toFixed(2) : null, p10Step: med((s) => s.p10) };
  rows.push(row);
  log(JSON.stringify(row));
}
await page.evaluate(() => { window.tw.clock.setRunning(false); window.__m.cap(0); });
if (OUT) await writeFile(OUT, JSON.stringify(rows, null, 1));
await close();
vite?.child.kill();
process.exit(0);
