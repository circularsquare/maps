// T-871: dragging across Europe zoomed out loads no zone for viewing; zoomed in it does.
//   node trainworld/checks/t871-europe-pan.mjs [url]      (from the maps root)
// With no url it starts its own Vite (checks/dev-nowatch.mjs, port PAN_PORT or 6193). A new game, Homes
// on, real mouse drags London to Vienna at each zoom of ZOOMS (default 6,7,9; FULL=1 runs London to Milan
// at 5.5,7,8,9.5, about 8 minutes). Per zoom: pack loads, water mask fetches, exact outlines fetched,
// main-thread ms a frame, long tasks over 50 ms, the aux worker's WASM MB. Fails when a zoom at or below
// ZONE_LOAD_ZOOM (7.5) opens a pack or fetches a water mask, or the zoomed-in leg opens none.
// QS adds URL switches (QS=&perfOff=zoneZoom: before T-871). Writes ../artifacts/trainworld/t871/pan-LABEL.json.
import { writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChrome, collectErrors, startVite } from './lib.mjs';

const OUT = process.env.OUT ?? fileURLToPath(new URL('../../artifacts/trainworld/t871/', import.meta.url));
const LABEL = process.env.LABEL ?? 'run';
const QS = process.env.QS ?? '';
const FULL = process.env.FULL === '1';
const ZOOMS = (process.env.ZOOMS ?? (FULL ? '5.5,7,8,9.5' : '6,7,9')).split(',').map(Number);
const LOAD_ZOOM = 7.5;
const ROUTE = FULL
  ? [[-0.12, 51.5], [2.35, 48.86], [4.35, 50.85], [4.9, 52.37], [10.0, 53.55], [13.4, 52.52], [14.42, 50.08], [16.37, 48.21], [11.58, 48.14], [8.54, 47.37], [9.19, 45.46]]
  : [[-0.12, 51.5], [2.35, 48.86], [4.35, 50.85], [8.68, 50.11], [11.58, 48.14], [16.37, 48.21]];
/** the zoomed-in leg is short: Paris to Brussels */
const NEAR_ROUTE = [[2.35, 48.86], [3.06, 50.63], [4.35, 50.85]];
const W = 1400, H = 900;

const pct = (a, p) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

async function run(BASE) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-871-', size: [W, H] });
  const errors = [];
  const reqs = [];
  const out = { label: LABEL, base: BASE, qs: QS, zooms: [], errors };
  try {
    const page = await ctx.newPage();
    await page.addInitScript(() => {
      try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', commuters: true, commuterEnd: 'home' }, v: 2 })); } catch { /* */ }
      window.__frames = []; window.__lt = [];
      const tick = (t) => { window.__frames.push(t); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
      try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); }).observe({ type: 'longtask', buffered: true }); } catch { /* */ }
    });
    collectErrors(page, errors, /favicon|openfreemap|AJAXError|noritetsu/, /maplibre-gl/);
    // the workers' fetches come through the page too
    page.on('request', (r) => { const m = /packs\/([^?]+)/.exec(r.url()); if (m) reqs.push(m[1]); });
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    await page.goto(`${BASE}?new=1&debug=1${QS}&v=${Date.now()}#6/48.86/2.35`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw?.map, undefined, { timeout: 180000 });
    await page.waitForTimeout(6000);
    const task = async () => { const { metrics } = await cdp.send('Performance.getMetrics'); return metrics.find((x) => x.name === 'TaskDuration').value * 1000; };
    const state = () => page.evaluate(() => {
      const d = window.twDemand;
      return { loads: d.start.loads.length, aux: Math.round((d.start.workers?.at(-1)?.memBytes ?? 0) / 1048576) };
    });
    // short of FULL, the zooms past LOAD_ZOOM take the short leg only
    const legs = FULL ? ZOOMS.map((z) => [z, ROUTE]) : ZOOMS.map((z) => [z, z <= LOAD_ZOOM ? ROUTE : NEAR_ROUTE]);
    for (const [zoom, route] of legs) {
      await page.evaluate(([at, z]) => { window.tw.map.jumpTo({ center: at, zoom: z }); }, [route[0], zoom]);
      await page.waitForTimeout(4000);
      const s0 = await state(), r0 = reqs.length, m0 = await task();
      const t0 = await page.evaluate(() => performance.now());
      let drags = 0;
      for (const target of route.slice(1)) {
        for (let guard = 0; guard < 40; guard++) {
          const p = await page.evaluate((t) => { const q = window.tw.map.project(t); return [q.x, q.y]; }, target);
          let dx = p[0] - W / 2, dy = p[1] - H / 2;
          const d = Math.hypot(dx, dy);
          if (d < 20) break;
          const k = Math.min(1, 400 / d); dx *= k; dy *= k;
          const sx = W / 2 + dx / 2, sy = H / 2 + dy / 2;
          await page.mouse.move(sx, sy); await page.mouse.down();
          for (let i = 1; i <= 25; i++) { await page.mouse.move(sx - (dx * i) / 25, sy - (dy * i) / 25); await page.waitForTimeout(16); }
          await page.mouse.up(); drags++;
          await page.waitForTimeout(120);
        }
        await page.waitForTimeout(600);
      }
      await page.waitForTimeout(3000);
      const t1 = await page.evaluate(() => performance.now());
      const m1 = await task(), s1 = await state();
      const st = await page.evaluate(([t0, t1]) => {
        const f = window.__frames.filter((t) => t >= t0 && t <= t1), iv = [];
        for (let i = 1; i < f.length; i++) iv.push(f[i] - f[i - 1]);
        return { iv, lt: window.__lt.filter(([s]) => s >= t0 && s <= t1).map(([, d]) => Math.round(d)) };
      }, [t0, t1]);
      const rq = reqs.slice(r0);
      const r = {
        zoom, drags, secs: Math.round((t1 - t0) / 100) / 10,
        packLoads: s1.loads - s0.loads, waterFetches: rq.filter((u) => /\.bin$/.test(u)).length, outlineFetches: rq.filter((u) => /boundary/.test(u)).length,
        mainMsPerFrame: Math.round(((m1 - m0) / Math.max(1, st.iv.length)) * 10) / 10, p95: Math.round(pct(st.iv, 0.95)), over50: st.iv.filter((v) => v > 50).length,
        longTasks: st.lt.length, longMs: st.lt.reduce((a, b) => a + b, 0), longMax: Math.max(0, ...st.lt), auxMB: s1.aux,
      };
      out.zooms.push(r);
      console.log(`z${String(zoom).padEnd(4)} drags ${String(drags).padStart(3)}  pack loads ${String(r.packLoads).padStart(3)}  water ${String(r.waterFetches).padStart(3)}  outlines ${String(r.outlineFetches).padStart(3)}  main ${r.mainMsPerFrame} ms/frame  >50ms frames ${r.over50}  long tasks ${r.longTasks}/${r.longMs} ms (max ${r.longMax})  aux ${r.auxMB} MB`);
    }
  } finally {
    await close();
  }
  return out;
}

const given = process.argv[2];
const vite = given ? null : await startVite(Number(process.env.PAN_PORT ?? 6193));
let out;
try {
  out = await run(given ?? vite.url);
} finally {
  if (vite) vite.child.kill();
}
await mkdir(OUT, { recursive: true });
await writeFile(join(OUT, `pan-${LABEL}.json`), JSON.stringify(out, null, 1));
const far = out.zooms.filter((z) => z.zoom <= LOAD_ZOOM), near = out.zooms.filter((z) => z.zoom > LOAD_ZOOM);
const bad = [
  ...far.filter((z) => z.packLoads || z.waterFetches).map((z) => `z${z.zoom}: ${z.packLoads} pack loads, ${z.waterFetches} water masks`),
  ...(near.length && !near.some((z) => z.packLoads) ? ['zoomed in, no zone opened'] : []),
  ...out.errors.slice(0, 3),
];
console.log(`${bad.length ? 'FAIL' : 'PASS'}  zones load for viewing only zoomed in${bad.length ? ': ' + bad.join('; ') : ''}`);
process.exit(bad.length ? 1 : 0);
