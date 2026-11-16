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
const CENTER0 = [-73.985, 40.758];
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

await page.evaluate(([c]) => { const tw = window.tw; tw.map.jumpTo({ center: c, zoom: 12 }); tw.clock.setSpeed(0); tw.clock.setRunning(true); window.dispatchEvent(new Event('focus')); }, [CENTER0]);
await page.waitForTimeout(3000);
const timing = await page.evaluate(async () => {
  const tw = window.tw, r = tw.renderer; tw.overlay.drawMs.clear();
  let g = 0, n = 0, longest = 0, last = performance.now(), gaps = [];
  const t0 = performance.now();
  await new Promise((res) => { const f = () => { g += r.guardMs; n++; const t = performance.now(); gaps.push(t - last); last = t; if (t - t0 < 5000) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  return { overlayDrawMs: +tw.overlay.drawMs.avg().toFixed(3), guardMs: +(g / n).toFixed(3), frames: n };
});
log('timing ' + JSON.stringify(timing));
const cdp = await ctx.newCDPSession(page);
await cdp.send('Profiler.enable');
await cdp.send('Profiler.setSamplingInterval', { interval: 200 });
await cdp.send('Profiler.start');
await page.waitForTimeout(6000);
const { profile: pr } = await cdp.send('Profiler.stop');
const self = new Map(), byId = new Map(pr.nodes.map((n) => [n.id, n]));
const dt = pr.timeDeltas, counts = new Map();
for (let i = 0; i < pr.samples.length; i++) counts.set(pr.samples[i], (counts.get(pr.samples[i]) ?? 0) + (dt[i] ?? 0));
let total = 0;
for (const [id, us] of counts) { const n = byId.get(id), cf = n.callFrame, k = `${cf.functionName || '(anon)'} ${cf.url.split('/').pop()}:${cf.lineNumber + 1}`; self.set(k, (self.get(k) ?? 0) + us); total += us; }
const top = [...self].sort((a, b) => b[1] - a[1]).slice(0, 30).map(([k, us]) => `${(us / 6000 / 10).toFixed(2)}% ${k}`);
log('main thread profile over 6 s, % of the window (idle included in (idle)/(program)):\n' + top.join('\n'));
await page.evaluate(() => window.tw.clock.setRunning(false));
await close();
vite?.child.kill();
process.exit(0);
