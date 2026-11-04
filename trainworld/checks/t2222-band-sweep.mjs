// T-2222, T-2258: lines side by side drawn where their bands say, at every zoom. On the real New York
// save and Anita's two Tokyo saves (skipped when absent; day 414 is the one her band snap was seen on,
// T-2222), each opened as a load opens it (the camera fitted to its network), in the Line view and,
// once demand is final, the Traffic view: at the zoom the load opened at and at each from 10 to 15 by
// 0.25 the simplified copy the renderer draws there (strokeLod.ts) is compared with the full strokes
// at every vertex it merged away; the full strokes' band shifts are checked for jumps between joined
// segments and for metres past BAND_M_RANGE; and no line or track segment is shorter than
// network.ts' MIN_SEG_M unless it is all of its edge (a shorter one points anywhere, so its band or
// track is shifted the wrong way: the round prong and the band snapping to its own track).
//   node trainworld/checks/t2222-band-sweep.mjs [url] [--line]     (default http://localhost:6189/)
// One PASS or FAIL line per save, view and check; figures in artifacts/trainworld/t2222/sweep.json.
import { existsSync } from 'node:fs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { ARTIFACTS, REAL_SAVE, launchChrome, importSave, collectErrors, demandFinal, GL } from './lib.mjs';

const URL = process.argv.find((a, i) => i > 1 && !a.startsWith('--')) ?? 'http://localhost:6189/';
const LINE_ONLY = process.argv.includes('--line');
const TOKYO = 'C:/Users/anita/Documents/trainworld/Tokyo Seoul Fukuoka Busan (from anita.garden).save';
const DAY414 = `${ARTIFACTS}anita-2026-11-03e/tokyo-day-414.save`;
const SAVES = [['real', REAL_SAVE], ['tokyo', TOKYO], ['day414', DAY414]].filter(([, f]) => existsSync(f));
const ZOOMS = Array.from({ length: 21 }, (_, i) => 10 + i * 0.25);
// (in the page: a merged vertex may sit at most 1 px off its band; a join may step 0.05 line widths
// and 1 m in the Line view, 0.2 line widths in the Traffic view)
const out = {};
let failed = 0;
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

// in the page: the checks on what is drawn now (Line or Traffic view as set)
const sweep = async ({ zooms, at }) => {
  const tw = window.tw, r = tw.renderer, src = r.net.lines;
  const { BAND_RANGE, BAND_M_RANGE, BAND_STRIDE } = await import('/src/map/lineBands.ts');
  const S = BAND_RANGE / 32767, M = BAND_M_RANGE / 32767, bd = src.band, n = src.count, wm = r.trafficWidths ? r.linesWmul : null;
  const ll = (x, y) => { const p = tw.toLngLat(x, y); return [+(p.lng ?? p[0]).toFixed(5), +(p.lat ?? p[1]).toFixed(5)]; };
  const frame = () => new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
  const joined = (i) => src.colour[i] === src.colour[i - 1] && Math.abs(src.seg[i * 4] - src.seg[i * 4 - 2]) + Math.abs(src.seg[i * 4 + 1] - src.seg[i * 4 - 1]) <= 1e-3;
  const jumps = [];
  let clamped = 0;
  for (let i = 0; i < n; i++) {
    const o = i * BAND_STRIDE;
    if (Math.abs(bd[o + 4]) >= 32766 || Math.abs(bd[o + 5]) >= 32766) clamped++;
    if (i === 0 || !joined(i)) continue;
    const a = o - BAND_STRIDE, ds = Math.abs(bd[o] - bd[a + 1]) * S, dm = Math.abs(bd[o + 4] - bd[a + 5]) * M;
    const dt = wm ? Math.abs(wm[i * 3 + 1] - wm[(i - 1) * 3 + 2]) : 0;
    if (ds > 0.05 || dm > 1 || dt > 0.2) jumps.push([src.colour[i], +ds.toFixed(3), +dm.toFixed(1), +dt.toFixed(2), ll(src.seg[i * 4], src.seg[i * 4 + 1])]);
  }
  const zs = [];
  for (const z of zooms) {
    tw.map.jumpTo({ center: at, zoom: z, bearing: 0, pitch: 0 });
    tw.overlay.invalidate();
    for (let k = 0; k < 6; k++) await frame();
    const st = r.lodOf.get(r.lines), bz = Math.ceil(z + Math.log2(devicePixelRatio) - 1e-6), band = st?.bands.get(bz);
    const used = band && band.vao.count < r.lines.count ? band : null;
    const rec = { z, lod: !!used, max: 0, over: 0, worst: [] };
    if (used && r.bandOn > 0) {
      const f = used.first, m = f.length, lb = used.strokes.band;
      for (let k = 0; k < m; k++) {
        const i = f[k], last = (k + 1 < m ? f[k + 1] : n) - 1;
        const x0 = src.seg[i * 4], y0 = src.seg[i * 4 + 1], dx = src.seg[last * 4 + 2] - x0, dy = src.seg[last * 4 + 3] - y0, L2 = dx * dx + dy * dy;
        for (let v = i + 1; v <= last; v++) {
          const t = L2 > 0 ? Math.max(0, Math.min(1, ((src.seg[v * 4] - x0) * dx + (src.seg[v * 4 + 1] - y0) * dy) / L2)) : 0;
          const o = k * BAND_STRIDE, ov = v * BAND_STRIDE;
          const dm = ((lb[o + 4] + (lb[o + 5] - lb[o + 4]) * t) - bd[ov + 4]) * M;
          const ds = wm && used.wm ? used.wm[k * 3 + 1] + (used.wm[k * 3 + 2] - used.wm[k * 3 + 1]) * t - wm[v * 3 + 1] : ((lb[o] + (lb[o + 1] - lb[o]) * t) - bd[ov]) * S;
          const e = Math.abs(ds * r.bandPx + dm * r.bandMPx);
          if (e > rec.max) rec.max = +e.toFixed(2);
          if (e > 1) { rec.over++; if (rec.worst.length < 10) rec.worst.push([+e.toFixed(2), src.colour[v], ll(src.seg[v * 4], src.seg[v * 4 + 1])]); }
        }
      }
    }
    zs.push(rec);
  }
  return { jumps: jumps.length, jumpsAt: jumps.slice(0, 10), clamped, zooms: zs };
};

// in the page: line and track segments shorter than MIN_SEG_M that are not all of their edge's stroke
const short = async () => {
  const tw = window.tw, net = tw.renderer.net;
  const { MIN_SEG_M } = await import('/src/map/network.ts');
  const out = {};
  for (const name of ['lines', 'track']) {
    const S = net[name], n = S.count, at = [];
    let count = 0, whole = 0, wholeMin = Infinity;
    const same = (i, k) => k >= 0 && k < n && S.edge[k] === S.edge[i] && S.colour[k] === S.colour[i];
    for (let i = 0; i < n; i++) {
      const L = Math.hypot(S.seg[i * 4 + 2] - S.seg[i * 4], S.seg[i * 4 + 3] - S.seg[i * 4 + 1]);
      if (L >= MIN_SEG_M) continue;
      // (an edge drawn as one short segment: reported, not failed)
      if (!(same(i, i - 1) || same(i, i + 1))) { whole++; wholeMin = Math.min(wholeMin, L); continue; }
      count++;
      if (at.length < 10) { const p = tw.toLngLat(S.seg[i * 4], S.seg[i * 4 + 1]); at.push([S.colour[i], S.edge[i], +L.toFixed(2), +(p.lng ?? p[0]).toFixed(5), +(p.lat ?? p[1]).toFixed(5)]); }
    }
    out[name] = { count, at, whole, wholeMin: whole ? +wholeMin.toFixed(2) : null };
  }
  return { min: MIN_SEG_M, ...out };
};

const { ctx, close } = await launchChrome({ prefix: 'tw-t2222-', size: [1400, 900], args: GL });
const page = ctx.pages()[0] ?? await ctx.newPage();
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${URL}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  for (const [name, file] of SAVES) {
    await page.evaluate(() => { window.tw.client.lastLoad = null; window.twPrev = [window.tw.renderer.net, window.tw.client.lastLines]; window.tw.client.fitNextGame(); });
    await importSave(page, await readFile(file));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.renderer.net?.lines.count > 0 && window.tw.client.lastLines?.length > 0 && window.tw.renderer.net !== window.twPrev[0] && window.tw.client.lastLines !== window.twPrev[1], undefined, { timeout: 600000 });
    await new Promise((r) => setTimeout(r, 2000));
    const [at, opened] = await page.evaluate(() => { const m = window.tw.map, c = m.getCenter(); return [[c.lng, c.lat], +m.getZoom().toFixed(2)]; });
    const zooms = [...new Set([opened, ...ZOOMS])];
    const sh = await page.evaluate(short);
    out[`${name}-short`] = sh;
    line(sh.lines.count + sh.track.count === 0, `${name}: no line or track segment under ${sh.min} m but its edge's only one`, sh.lines.count + sh.track.count ? `lines ${sh.lines.count}, track ${sh.track.count}, first ${JSON.stringify(sh.lines.at[0] ?? sh.track.at[0])}` : `opened at zoom ${opened}; edges drawn as one short segment: lines ${sh.lines.whole}${sh.lines.whole ? ` (shortest ${sh.lines.wholeMin} m)` : ''}, track ${sh.track.whole}`);
    for (const view of LINE_ONLY ? ['line'] : ['line', 'traffic']) {
      if (view === 'traffic') await demandFinal(page, { polling: 1000, lines: 1 }).catch((e) => console.log(`(${name}) demand wait: ${e.message.split('\n')[0]}`));
      await page.evaluate((v) => {
        const tw = window.tw;
        tw.display.value = { ...tw.display.peek(), lineBands: true, trackColour: v };
        tw.renderer.lineBands = true;
        tw.overlay.invalidate();
      }, view);
      await new Promise((r) => setTimeout(r, 3000));
      const r = await page.evaluate(sweep, { zooms, at });
      out[`${name}-${view}`] = r;
      const bad = r.zooms.filter((z) => z.over > 0);
      line(bad.length === 0, `${name} ${view}: drawn copies on their bands at every zoom`, bad.length ? bad.map((z) => `z${z.z} ${z.over} (max ${z.max} px)`).join(', ') : `max ${Math.max(...r.zooms.map((z) => z.max))} px, ${r.zooms.filter((z) => z.lod).length} of ${r.zooms.length} zooms simplified`);
      line(r.jumps === 0, `${name} ${view}: no band jumps between joined segments`, r.jumps ? `${r.jumps}, first ${JSON.stringify(r.jumpsAt[0])}` : '');
      if (view === 'line') line(r.clamped === 0, `${name}: no band metres past the range`, r.clamped ? `${r.clamped} segment ends` : '');
    }
  }
  line(errors.length === 0, 'no console or GL errors', errors.slice(0, 3).join(' | '));
} finally {
  await mkdir(`${ARTIFACTS}t2222`, { recursive: true });
  await writeFile(`${ARTIFACTS}t2222/sweep.json`, JSON.stringify(out, null, 1));
  await close();
}
process.exit(failed ? 1 : 0);
