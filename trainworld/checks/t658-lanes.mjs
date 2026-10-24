// T-658a: double track opens into its two tracks close in. On the real save (T-007-real-nyc.save):
// the overlay at zoom 12 and 15 is pixel for pixel what the baseline build draws (default 8800, the
// last commit's snapshot); screenshots at zoom 15, 17 and 19 round the busiest transfer station
// and a flat junction, light and dark, traffic thickness on (the default); the overlay's draw time
// at zoom 15 and 17.5; a click on either track at 17.5 picks a line.
//   node trainworld/checks/t658-lanes.mjs [url] [baseline url]   (from the maps root)
// Writes artifacts/trainworld/t658/*.png. Prints PASS or FAIL lines, exits 1 on failure. A few minutes.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { launchChrome, importSave, demandFinal, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6160/';
const REF = process.argv[3] ?? 'http://localhost:8800/trainworld/';
const OUT = fileURLToPath(new URL('../../artifacts/trainworld/t658', import.meta.url));

await mkdir(OUT, { recursive: true });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const bytes = Array.from(await readFile(REAL_SAVE));
const closers = [];

async function open(url, shade) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t658-', size: [1200, 800], args: ANGLE });
  closers.push(close);
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, shade);
  await page.goto(`${url}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { strict: false });
  await page.waitForFunction(() => !document.querySelector('.loading-card'), undefined, { timeout: 300000, polling: 500 });
  await page.waitForTimeout(1500);
  return { ctx, page };
}

// a pixel differs when a channel is more than one level off: a station quad's size (T-713 took the
// crowd's bulge out of it) moves its interpolated edge coverage by one level in a channel
const differs = (a, b, k) => Math.abs(a[k] - b[k]) > 1 || Math.abs(a[k + 1] - b[k + 1]) > 1 || Math.abs(a[k + 2] - b[k + 2]) > 1 || Math.abs(a[k + 3] - b[k + 3]) > 1;
const go = async (page, at, zoom) => {
  await page.evaluate(([c, z]) => void window.tw.map.jumpTo({ center: c, zoom: z }), [at, zoom]);
  await page.waitForTimeout(1800);
};
/** the overlay's pixels right after a draw, base64 */
const pixels = (page) => page.evaluate(() => {
  const ov = window.tw.overlay, gl = ov.gl;
  ov.draw();
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(w * h * 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
  let s = '';
  for (let i = 0; i < px.length; i += 0x8000) s += String.fromCharCode(...px.subarray(i, i + 0x8000));
  return btoa(s);
});

try {
  const dev = await open(BASE, 'light');
  // where to look: the station with the most other lines' strokes within 150 m (the real save gives
  // each line its own track), and a flat junction on double track with lines on it
  const spots = await dev.page.evaluate(() => {
    const w = window.tw.world.value, n = window.tw.renderer.net, L = n.lines;
    const near = (x, y, r) => {
      const ids = new Set();
      for (let i = 0; i < L.count; i++) {
        const mx = (L.seg[i * 4] + L.seg[i * 4 + 2]) / 2, my = (L.seg[i * 4 + 1] + L.seg[i * 4 + 3]) / 2;
        if (Math.abs(mx - x) < r && Math.abs(my - y) < r) ids.add(L.colour[i]);
      }
      return ids.size;
    };
    let best = -1, bestN = -1;
    for (let i = 0; i < n.stationCount; i += 3) {
      const k = near(n.stations[i * 3], n.stations[i * 3 + 1], 200);
      if (k > bestN) { bestN = k; best = i; }
    }
    const station = window.tw.toLngLat(n.stations[best * 3], n.stations[best * 3 + 1]);
    const lined = new Set(L.edge);
    let junction = null;
    for (let k = 0; k < n.junctionIds.length && !junction; k++) {
      const id = String(n.junctionIds[k]);
      const es = w.edges.filter((e) => String(e.a) === id || String(e.b) === id);
      if (es.length >= 3 && es.every((e) => e.tracks === 2 && !e.crossover && lined.has(Number(e.id)))) junction = window.tw.toLngLat(n.junctions[k * 2], n.junctions[k * 2 + 1]);
    }
    // open line: a double-track line stroke 500 m or more from any station, as near the station as that
    let open = null, openD = Infinity;
    const sx = n.stations[best * 3], sy = n.stations[best * 3 + 1];
    for (let i = 0; i < L.count; i += 5) {
      if (L.lane[i * 2] < 255) continue;
      const x = L.seg[i * 4], y = L.seg[i * 4 + 1];
      let far = true;
      for (let j = 0; j < n.stationCount && far; j++) if (Math.hypot(n.stations[j * 3] - x, n.stations[j * 3 + 1] - y) < 650) far = false;
      const d = Math.hypot(x - sx, y - sy);
      if (far && d < openD) { openD = d; open = window.tw.toLngLat(x, y); }
    }
    // T-658d: a turn-back crossover, the first a dead end's scissors if there is one
    const T = n.turnbacks;
    let turnback = null, deadEnd = null;
    if (T?.count) {
      turnback = window.tw.toLngLat(T.seg[0], T.seg[1]);
      for (let i = 1; i < T.count && !deadEnd; i++) {
        // a scissors: two strokes starting at the same point
        if (Math.hypot(T.seg[i * 4] - T.seg[0], T.seg[i * 4 + 1] - T.seg[1]) < 1e-6) continue;
        for (let j = i + 1; j < T.count; j++) if (T.seg[j * 4] === T.seg[i * 4] && T.seg[j * 4 + 1] === T.seg[i * 4 + 1] && T.edge[j] === T.edge[i]) { deadEnd = window.tw.toLngLat(T.seg[i * 4], T.seg[i * 4 + 1]); break; }
      }
    }
    return { station, junction, open, turnback, deadEnd, turnbackSegs: T?.count ?? 0, lines: bestN };
  });
  console.log('spots', JSON.stringify(spots));

  // far zoom unchanged: the same overlay as the baseline build
  const ref = await open(REF, 'light');
  // the same map size in both, whatever the panels beside it
  const fix = (page) => page.evaluate(() => {
    const map = window.tw.map, c = map.getContainer();
    Object.assign(c.style, { position: 'fixed', left: '0px', top: '0px', width: '1000px', height: '700px', zIndex: '0' });
    map.resize();
  });
  await fix(dev.page);
  await fix(ref.page);
  for (const z of [12, 15]) {
    await go(dev.page, spots.station, z);
    await go(ref.page, spots.station, z);
    // line widths follow demand, which can still be settling in one of the two: up to 6 tries
    let a, b, diff = 0;
    for (let tries = 0; tries < 6; tries++) {
      [a, b] = [Buffer.from(await pixels(dev.page), 'base64'), Buffer.from(await pixels(ref.page), 'base64')];
      diff = 0;
      for (let k = 0; k < Math.min(a.length, b.length); k += 4) if (differs(a, b, k)) diff++;
      if (!diff) break;
      await dev.page.waitForTimeout(10000);
    }
    diff = 0;
    const mask = new Uint8Array(a.length);
    for (let k = 0; k < Math.min(a.length, b.length); k += 4) {
      if (differs(a, b, k)) { diff++; mask[k] = 255; mask[k + 3] = 255; }
      else if (a[k + 3]) { mask[k] = mask[k + 1] = mask[k + 2] = 160; mask[k + 3] = 255; }
    }
    if (diff) {
      // where they differ: red, the rest of the drawing grey (flipped: GL rows run bottom up)
      const png = await dev.page.evaluate(([m, w, h]) => {
        const px = Uint8ClampedArray.from(atob(m), (c) => c.charCodeAt(0)), c = document.createElement('canvas');
        c.width = w; c.height = h;
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) img.data.set(px.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
        c.getContext('2d').putImageData(img, 0, 0);
        return c.toDataURL('image/png').split(',')[1];
      }, [Buffer.from(mask).toString('base64'), 1000, a.length / 4000]);
      await writeFile(`${OUT}/diff-z${z}.png`, Buffer.from(png, 'base64'));
    }
    report(`zoom ${z} overlay identical to the baseline`, a.length === b.length && diff === 0, `(${diff} of ${a.length / 4} pixels differ)`);
  }
  await ref.ctx.close();

  // draw cost: the overlay's GPU time (the renderer's timer query) and CPU time, over 40 frames
  const cost = async (z) => {
    await go(dev.page, spots.station, z);
    return dev.page.evaluate(async () => {
      const ov = window.tw.overlay, r = window.tw.renderer, gpu = [], cpu = [];
      for (let k = 0; k < 45; k++) {
        const t0 = performance.now();
        ov.draw();
        cpu.push(performance.now() - t0);
        await new Promise((f) => requestAnimationFrame(() => setTimeout(f, 20)));
        if (k >= 5 && r.gpuMs != null) gpu.push(r.gpuMs);
      }
      const avg = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
      return { gpu: avg(gpu), cpu: avg(cpu.slice(5)) };
    });
  };
  const c15 = await cost(15), c175 = await cost(17.5);
  console.log(`draw ms (gpu, cpu): zoom 15 ${c15.gpu.toFixed(2)}, ${c15.cpu.toFixed(2)}; zoom 17.5 ${c175.gpu.toFixed(2)}, ${c175.cpu.toFixed(2)}`);

  // a click beside the centre line, on either track, picks a line at 17.5
  await go(dev.page, spots.station, 17.5);
  const picks = await dev.page.evaluate(() => {
    const w = window.tw.world.value, n = window.tw.renderer.net;
    // a line stroke near the centre of the view, 200 m or more from any station
    const map = window.tw.map, c = map.project(map.getCenter());
    const out = [];
    for (let i = 0; i < n.lines.count && out.length < 6; i += 7) {
      const x0 = n.lines.seg[i * 4], y0 = n.lines.seg[i * 4 + 1], x1 = n.lines.seg[i * 4 + 2], y1 = n.lines.seg[i * 4 + 3];
      if (n.lines.lane[i * 2] < 255 || n.lines.lane[i * 2 + 1] < 255) continue;
      const [ax, ay] = window.tw.overlay.toScreen(x0, y0), [bx, by] = window.tw.overlay.toScreen(x1, y1);
      const mx = (ax + bx) / 2, my = (ay + by) / 2, len = Math.hypot(bx - ax, by - ay);
      if (len < 4 || Math.hypot(mx - c.x, my - c.y) > 350) continue;
      if (window.tw.renderer.stationAt(mx, my, (x, y) => window.tw.overlay.toScreen(x, y), 40) >= 0) continue;
      const nx = -(by - ay) / len, ny = (bx - ax) / len, gap = window.tw.renderer.laneReachPx / 1.6;
      const side = (s) => window.tw.pick(window.tw.overlay, mx + nx * s * gap, my + ny * s * gap, false)?.kind ?? 'none';
      out.push([side(1), side(-1)]);
    }
    return out;
  });
  report('either track picks the line at zoom 17.5', picks.length > 0 && picks.every(([a, b]) => a === 'line' && b === 'line'), JSON.stringify(picks));

  // T-658b: two trains of one line passing each other on double track, each on its own track
  const pass = await dev.page.evaluate(async () => {
    const { tripAt } = await import('/src/map/network.ts');
    const r = window.tw.renderer, n = r.net, now = (await import('/src/game/clock.ts')).clock.now();
    const at = [];
    for (let i = 0; i < r.tripCount; i++) {
      const p = tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch, now, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
      if (p) at.push({ i, ...p });
    }
    let best = null;
    for (const a of at) for (const b of at) {
      if (a.line !== b.line || a.run !== 0 || b.run !== 1) continue;
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 120 && (!best || d < best.d)) best = { d, a: a.i, b: b.i, x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }
    return best && { ...best, at: window.tw.toLngLat(best.x, best.y), now };
  });
  if (!pass) report('two trains passing found', false);
  else {
    await go(dev.page, pass.at, 17.6);
    const hits = await dev.page.evaluate(async (p) => {
      const { tripAt } = await import('/src/map/network.ts');
      const r = window.tw.renderer, n = r.net, ov = window.tw.overlay;
      const one = (i) => {
        const q = tripAt(n, r.trips[i * 2], r.trips[i * 2 + 1] + r.tripsEpoch, p.now, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, r.laneShiftM);
        const [x, y] = ov.toScreen(q.x, q.y);
        const s = window.tw.pick(ov, x, y, true);
        return { picked: s?.kind === 'train' && s.profile === r.trips[i * 2] && Math.abs(s.dep - (r.trips[i * 2 + 1] + r.tripsEpoch)) < 0.01, x, y };
      };
      return { a: one(p.a), b: one(p.b), shift: r.laneShiftM };
    }, pass);
    report('two trains passing: each picked on its own track', hits.a.picked && hits.b.picked && hits.shift === 2, JSON.stringify({ apartM: pass.d.toFixed(0), ...hits }));
    await dev.page.screenshot({ path: `${OUT}/trains-passing-z17.6-light.png`, clip: { x: 0, y: 0, width: 1000, height: 700 } });
    await go(dev.page, pass.at, 19);
    await dev.page.screenshot({ path: `${OUT}/trains-passing-z19-light.png`, clip: { x: 0, y: 0, width: 1000, height: 700 } });
  }

  const shoot = async (page, shade) => {
    await fix(page);
    for (const [name, at] of [['station', spots.station], ['open', spots.open], ['junction', spots.junction], ['turnback', spots.turnback], ['deadend', spots.deadEnd]]) {
      if (!at) continue;
      for (const z of [15, 16.9, 17.2, 19]) {
        await go(page, at, z);
        await page.screenshot({ path: `${OUT}/${name}-z${z}-${shade}.png`, clip: { x: 0, y: 0, width: 1000, height: 700 } });
      }
    }
  };
  await shoot(dev.page, 'light');
  await dev.ctx.close();
  const dark = await open(BASE, 'dark');
  await shoot(dark.page, 'dark');
  await dark.ctx.close();
} finally {
  for (const close of closers) await close().catch(() => {});
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
