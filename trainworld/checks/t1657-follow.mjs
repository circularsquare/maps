// T-1657 / T-1621: how far station names and attraction marks are from where they belong on each map
// frame while the camera moves, on the real save (New York), with mouse drags (pan) and wheel zooms,
// plus the main thread's ms a frame. Station names are checked against the placement rule (the
// station's screen point plus the gap, below or beside); marks against map.project of the attraction.
//   node trainworld/checks/t1657-follow.mjs <url>      (from the maps root; a Vite with ?debug=1 pages)
import { readFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const BASE = process.argv[2];
if (!BASE) { console.error('usage: node checks/t1657-follow.mjs <url>'); process.exit(2); }
const SCENES = [
  { name: 'pan z14', at: [-73.985, 40.75], zoom: 14, drag: 'pan' },
  { name: 'pan z15', at: [-73.985, 40.755], zoom: 15, drag: 'pan' },
  { name: 'zoom 12-16 and back', at: [-73.98, 40.74], zoom: 12, drag: 'wheel' },
];
const r1 = (x) => Math.round(x * 10) / 10;
const { ctx, close } = await launchChrome({ prefix: 'tw-follow-', size: [1400, 900] });
const errors = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch { /* */ } });
  page.on('pageerror', (e) => errors.push(e.message));
  const cdp = await ctx.newCDPSession(page);
  await cdp.send('Performance.enable');
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&opened=all&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250 }).catch(() => errors.push('demand never final'));
  await page.waitForTimeout(2000);
  await page.evaluate(async () => {
    const geo = await import('/src/map/stationGeometry.ts');
    const tw = window.tw, map = tw.map, box = map.getContainer();
    const S = (window.__follow = { on: false, names: [], marks: [], frames: 0 });
    map.on('render', () => {
      if (!S.on || !tw.overlay.moving) return;
      const cr = box.getBoundingClientRect(), z = map.getZoom(), ss = tw.renderer.stationStyle;
      const gap = Math.max(10, geo.plainBoxHalfPx(ss, z) + 4, ss.boxes ? 0 : ss.radius[1] + 3);
      let nw = 0, nn = 0, mw = 0, mn = 0;
      const W = tw.overlay.cssW, H = tw.overlay.cssH;
      for (const l of tw.labels.labels) {
        if (l.hidden !== false) continue;
        const [sx, sy] = tw.overlay.toScreen(l.x, l.y);
        if (!(sx > -50 && sx < W + 50 && sy > -50 && sy < H + 50)) continue;
        const r = l.el.getBoundingClientRect();
        const ex = l.below ? sx - l.w / 2 : sx + gap, ey = l.below ? sy + gap - 2 : sy - l.h / 2;
        nw = Math.max(nw, Math.hypot(r.left - cr.left - ex, r.top - cr.top - ey));
        nn++;
      }
      for (const m of window.twAttractionMarks?.() ?? []) {
        if (m.el.style.display === 'none' || m.el.style.visibility === 'hidden') continue;
        const p = map.project([m.lng, m.lat]);
        if (!(p.x > 0 && p.x < W && p.y > 0 && p.y < H)) continue;
        const r = m.el.getBoundingClientRect();
        mw = Math.max(mw, Math.hypot(r.left + r.width / 2 - cr.left - p.x, r.top + r.height / 2 - cr.top - p.y));
        mn++;
      }
      S.frames++;
      S.names.push(nw); S.marks.push(mw);
      S.nn = nn; S.mn = mn;
    });
  });
  const metric = async () => {
    const { metrics } = await cdp.send('Performance.getMetrics');
    return Object.fromEntries(metrics.map((x) => [x.name, x.value])).TaskDuration * 1000;
  };
  const cx = 700, cy = 450;
  for (const sc of SCENES) {
    await page.evaluate(([at, zoom]) => { window.tw.map.jumpTo({ center: at, zoom }); }, [sc.at, sc.zoom]);
    await page.waitForTimeout(2500);
    await page.evaluate(() => { const S = window.__follow; S.names = []; S.marks = []; S.frames = 0; S.on = true; });
    const m0 = await metric(), t0 = await page.evaluate(() => performance.now());
    await page.mouse.move(cx, cy);
    if (sc.drag === 'pan') {
      for (const dir of [1, -1]) {
        await page.mouse.move(cx - dir * 200, cy - dir * 60);
        await page.mouse.down();
        for (let i = 1; i <= 50; i++) { await page.mouse.move(cx - dir * 200 + dir * 8 * i, cy - dir * 60 + dir * 2.4 * i); await page.waitForTimeout(16); }
        await page.mouse.up();
        await page.waitForTimeout(150);
      }
    } else {
      for (const dy of [-100, 100]) {
        for (let i = 0; i < 25; i++) { await page.mouse.wheel(0, dy); await page.waitForTimeout(40); }
        await page.waitForTimeout(400);
      }
    }
    const t1 = await page.evaluate(() => performance.now()), m1 = await metric();
    const S = await page.evaluate(() => { const s = window.__follow; s.on = false; return { frames: s.frames, names: s.names, marks: s.marks, nn: s.nn, mn: s.mn }; });
    const sum = (a) => ({ off: a.filter((v) => v > 2).length, worst: r1(Math.max(0, ...a)) });
    const n = sum(S.names), k = sum(S.marks);
    console.log(`${sc.name.padEnd(22)} moving frames ${S.frames}  names (${S.nn} on screen) off>2px ${n.off} worst ${n.worst} px  marks (${S.mn}) off>2px ${k.off} worst ${k.worst} px  main ${r1((m1 - m0) / Math.max(1, S.frames))} ms/frame`);
  }
} finally {
  await close();
}
if (errors.length) console.log('errors:', errors.slice(0, 3).join(' | '));
