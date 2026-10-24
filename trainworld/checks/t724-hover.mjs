// T-724 / T-689: what a pointer move costs on the real save, demand final. With the select and the
// station tool, in four views, a zigzag of real mouse moves (8 px at 60 Hz, a 350 ms rest every 25):
// each move's handlers (window capture to bubble), the timer tasks they start (the resting list, the
// station name), the station tool's snap and name lookup; one view's moves come right after an
// edit. Then wheel scrolls in and out from zoom 11: frame callbacks and the per-frame placers.
//   node trainworld/checks/t724-hover.mjs <url of your own Vite>     (from the maps root; ~4 min)
// Env: LABEL, NOSOLVE=1, PROFILE=1 (a .cpuprofile per view), ONLY=select|station|wheel|probe (with
// PROBE, a page expression run in each view). FAIL when a view's 95th percentile move is over 8 ms.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6155/';
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t724/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const ONLY = process.env.ONLY;

const pct = (a, p) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const r1 = (x) => Math.round(x * 10) / 10;
const summ = (a) => ({ n: a.length, p50: r1(pct(a, 0.5)), p95: r1(pct(a, 0.95)), max: r1(Math.max(0, ...a)) });

const { ctx, close } = await launchChrome({ prefix: 'tw-t724-' });
const out = { label: LABEL, scenes: [] };
await mkdir(OUT, { recursive: true });
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => {
    try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ }
    const W = window;
    W.__mv = []; W.__timers = []; W.__raf = []; W.__wheel = [];
    let t0 = 0;
    addEventListener('mousemove', () => { t0 = performance.now(); }, true);
    addEventListener('mousemove', () => { W.__mv.push(performance.now() - t0); }, false);
    let w0 = 0;
    addEventListener('wheel', () => { w0 = performance.now(); }, { capture: true, passive: true });
    addEventListener('wheel', () => { W.__wheel.push(performance.now() - w0); }, { passive: true });
    const st = W.setTimeout.bind(W);
    W.setTimeout = (fn, ms, ...a) => st(typeof fn === 'function' ? (...b) => { const s = performance.now(); try { return fn(...b); } finally { const d = performance.now() - s; if (d > 2) W.__timers.push([s, d, String(fn).slice(0, 70)]); } } : fn, ms, ...a);
    const raf = W.requestAnimationFrame.bind(W);
    W.requestAnimationFrame = (fn) => raf((t) => { const s = performance.now(); try { return fn(t); } finally { const d = performance.now() - s; if (d > 1) W.__raf.push([s, d]); } });
  });
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  const cdp = await ctx.newCDPSession(page);
  await page.goto(`${BASE}?debug=1&new=1${process.env.QS ?? ''}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const bytes = await readFile(REAL_SAVE);
  await importSave(page, bytes);
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 120000 });
  if (process.env.NOSOLVE !== '1') {
    await demandFinal(page, { polling: 250 }).catch(() => console.log('no final'));
  }
  await page.waitForTimeout(1500);
  // time the station tool's name lookup and snap (sync part)
  await page.evaluate(() => {
    const T = window.tw.tools, W = window;
    W.__name = []; W.__snap = [];
    const nf = T.nameFor.bind(T);
    T.nameFor = (...a) => { const s = performance.now(); const r = nf(...a); W.__name.push(performance.now() - s); return r; };
    const sn = T.snap.bind(T);
    T.snap = (...a) => { const s = performance.now(); const r = sn(...a); W.__snap.push(performance.now() - s); return r; };
  });
  const SCENES = [
    { name: 'Manhattan z12', at: [-73.975, 40.76], zoom: 12 },
    { name: 'Manhattan z14', at: [-73.985, 40.75], zoom: 14 },
    { name: 'Midtown z16', at: [-73.988, 40.752], zoom: 16 },
    { name: 'Brooklyn z14', at: [-73.985, 40.69], zoom: 14 },
  ];
  const clear = () => page.evaluate(() => { const W = window; W.__mv = []; W.__timers = []; W.__raf = []; W.__name = []; W.__snap = []; W.__wheel = []; });
  const grab = () => page.evaluate(() => { const W = window; return { mv: W.__mv, timers: W.__timers, raf: W.__raf.map((x) => x[1]), name: W.__name, snap: W.__snap, wheel: W.__wheel }; });
  const prof = async (on, name) => {
    if (process.env.PROFILE !== '1') return;
    if (on) { await cdp.send('Profiler.enable'); await cdp.send('Profiler.setSamplingInterval', { interval: 100 }); await cdp.send('Profiler.start'); }
    else { const { profile: p } = await cdp.send('Profiler.stop'); await writeFile(join(OUT, `${LABEL}-${name.replace(/\W+/g, '_')}.cpuprofile`), JSON.stringify(p)); }
  };
  if (ONLY === 'probe') {
    for (const sc of SCENES) {
      await page.evaluate(([at, zoom]) => void window.tw.map.jumpTo({ center: at, zoom }), [sc.at, sc.zoom]);
      await page.waitForTimeout(3000);
      console.log(sc.name, await page.evaluate(async (x) => { try { return JSON.stringify(await (0, eval)(x)); } catch (e) { return 'probe error: ' + e.message + e.stack; } }, process.env.PROBE));
    }
  }
  for (const mode of ['select', 'station']) {
    if (ONLY && ONLY !== mode) continue;
    await page.evaluate(async (m) => { const { tool } = await import('/src/game/ui.ts'); tool.value = m; }, mode);
    for (const sc of SCENES) {
      await page.evaluate(([at, zoom]) => void window.tw.map.jumpTo({ center: at, zoom }), [sc.at, sc.zoom]);
      await page.waitForTimeout(3000);
      await page.mouse.move(300, 200);
      await page.waitForTimeout(300);
      if (sc.name === 'Brooklyn z14') {
        // a new network just before the moves: a blueprint route in the view (the first moves after an edit)
        const ok = await page.evaluate(async () => {
          const { lngLatToPack } = await import('/src/game/coords.ts');
          const { edit } = await import('/src/game/world.ts');
          const ll = window.tw.map.unproject([200, 800]);
          const a = lngLatToPack(ll.lng, ll.lat), b = lngLatToPack(ll.lng + 0.01, ll.lat + 0.003);
          const before = window.tw.overlay.renderer.net;
          const r = await edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
          for (let i = 0; i < 100 && window.tw.overlay.renderer.net === before; i++) await new Promise((res) => setTimeout(res, 20));
          return r.ok && window.tw.overlay.renderer.net !== before;
        });
        console.log('    edit made a new network:', ok);
      }
      await clear();
      await prof(true);
      // a zigzag: 8 px steps at 60 Hz, a 350 ms rest every 25 moves
      let k = 0;
      for (let row = 0; row < 4; row++) {
        const y = 250 + row * 120;
        for (let i = 0; i < 100; i++) {
          const x = row % 2 ? 1100 - i * 8 : 300 + i * 8;
          await page.mouse.move(x, y + (i % 7) * 3);
          await page.waitForTimeout(16);
          if (++k % 25 === 0) await page.waitForTimeout(350);
        }
      }
      await page.waitForTimeout(400);
      await prof(false, `${mode}-${sc.name}`);
      const g = await grab();
      const s = { mode, name: sc.name, move: summ(g.mv), over8: g.mv.filter((v) => v > 8).length, timers: summ(g.timers.map((t) => t[1])), timerTop: g.timers.sort((a, b) => b[1] - a[1]).slice(0, 3).map((t) => [r1(t[1]), t[2]]), raf: summ(g.raf), name: summ(g.name), snap: summ(g.snap) };
      out.scenes.push(s);
      console.log(`${mode.padEnd(8)} ${sc.name.padEnd(14)} move p50 ${s.move.p50} p95 ${s.move.p95} max ${s.move.max} (>8ms ${s.over8}/${s.move.n})  snap p95 ${s.snap.p95} max ${s.snap.max}  name n${s.name.n} p50 ${s.name.p50} max ${s.name.max}  timers max ${s.timers.max}  raf p95 ${s.raf.p95} max ${s.raf.max}`);
      for (const t of s.timerTop) console.log('    timer', t[0], t[1].replace(/\s+/g, ' '));
    }
  }
  if (!ONLY || ONLY === 'wheel') {
    await page.evaluate(async () => { const { tool } = await import('/src/game/ui.ts'); tool.value = 'select'; });
    // T-689: wheel scrolls in and out from zoom 11 with a mousemove before each, as T-675 traced
    await page.evaluate(() => void window.tw.map.jumpTo({ center: [-73.98, 40.74], zoom: 11 }));
    await page.waitForTimeout(3000);
    await page.evaluate(() => {
      const hooks = window.tw.overlay.afterMapFrame;
      window.__hook = hooks.map(() => []);
      hooks.forEach((f, i) => { hooks[i] = () => { const s = performance.now(); try { return f(); } finally { const d = performance.now() - s; window.__hook[i].push(d); if (d > 3) (window.__slow ??= []).push([i, Math.round(d * 10) / 10, Math.round(window.tw.map.getZoom() * 100) / 100]); } }; });
    });
    await clear();
    await prof(true);
    const metric = async () => { const { metrics } = await cdp.send('Performance.getMetrics'); const m = Object.fromEntries(metrics.map((x) => [x.name, x.value])); return { task: m.TaskDuration * 1000, layout: m.LayoutDuration * 1000, style: m.RecalcStyleDuration * 1000, script: m.ScriptDuration * 1000 }; };
    await cdp.send('Performance.enable');
    const m0 = await metric();
    const firsts = [];
    for (let rep = 0; rep < 4; rep++) {
      for (const dy of [-100, 100]) {
        await page.mouse.move(700 + rep * 5, 450);
        await page.waitForTimeout(100);
        const before = await page.evaluate(() => window.__raf.length);
        for (let i = 0; i < 12; i++) { await page.mouse.wheel(0, dy); await page.waitForTimeout(30); }
        await page.waitForTimeout(600);
        firsts.push(await page.evaluate((b) => window.__raf.slice(b, b + 3).map((x) => x[1]), before));
      }
    }
    const m1 = await metric();
    await prof(false, 'wheel');
    const g = await grab();
    const hooks = await page.evaluate(() => window.__hook.map((a) => [a.length, Math.round(Math.max(0, ...a) * 10) / 10, Math.round(a.reduce((x, y) => x + y, 0))]));
    const frames = g.raf.length;
    console.log('  slow hooks [hook, ms, zoom]', JSON.stringify(await page.evaluate(() => window.__slow ?? [])));
    console.log('  hooks [calls, max ms, total ms]', JSON.stringify(hooks), ' main ms', Math.round(m1.task - m0.task), 'layout', Math.round(m1.layout - m0.layout), 'style', Math.round(m1.style - m0.style), 'script', Math.round(m1.script - m0.script), 'raf calls', frames);
    const s = { mode: 'wheel', hooks, raf: summ(g.raf), firstFrames: firsts.map((f) => f.map(r1)), wheel: summ(g.wheel), move: summ(g.mv), timers: summ(g.timers.map((t) => t[1])) };
    out.scenes.push(s);
    console.log(`wheel   frames p50 ${s.raf.p50} p95 ${s.raf.p95} max ${s.raf.max}  wheel handler max ${s.wheel.max}  move max ${s.move.max}  first frames ${JSON.stringify(s.firstFrames)}`);
  }
} finally {
  await close();
}
await writeFile(join(OUT, `hover-${LABEL}.json`), JSON.stringify(out, null, 1));
const slow = out.scenes.filter((s) => s.mode !== 'wheel' && s.move.p95 > 8);
const views = out.scenes.filter((s) => s.mode !== 'wheel');
if (views.length) console.log(`${slow.length ? 'FAIL' : 'PASS'}  pointer moves  (95th percentile per view ${views.map((s) => s.move.p95).join(', ')} ms, budget 8${slow.length ? '; over: ' + slow.map((s) => `${s.mode} ${s.name}`).join(', ') : ''})`);
process.exit(slow.length ? 1 : 0);
