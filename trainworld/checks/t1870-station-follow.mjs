// T-1870: a station ghost follows the pointer along track wherever the platform fits, at any zoom.
//   node trainworld/checks/t1870-station-follow.mjs [url] [label]      (from the maps root; a few minutes)
// A fresh game in Brooklyn gets blueprint track: a straight drawn in three clicks (two plain joints),
// a station on one joint and a branch off it (a junction), and a curve through one point. Every
// SAMPLE_M along each piece the track model is asked whether a station fits right there; then, with
// the station tool at zooms 13, 14 and 16, the pointer rests on each sample and the ghost's spot is
// read. A sample "moved" when a station fits at it and the ghost's centre is more than MOVED_M from
// it. Prints counts per zoom (moved, refused where it fits, where it does not fit: moved or refused)
// and writes artifacts/trainworld/t1870/<label>-sweep.json and screenshots of the worst moves.
import { mkdir, writeFile } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6170/';
const LABEL = process.argv[3] ?? 'run';
const OUT = `${ARTIFACTS}t1870/`;
// a third argument picks the zooms ("13,16"; "0" for none: the pictures only)
const SAMPLE_M = 40, MOVED_M = 10, ZOOMS = process.argv[4] ? process.argv[4].split(',').map(Number).filter((z) => z > 0) : [13, 14, 16];
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1870-', size: [1200, 800], args: GL });
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client, P = (lng, lat) => tw.lngLatToPack(lng, lat);
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (lng, lat) => { const p = P(lng, lat); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) bd = d, best = n.id; } return best; };
    const free = (lng, lat) => ({ kind: 'free', ...P(lng, lat), level: 0 });
    const out = [];
    const route = async (from, to, pis = []) => { const r = await c.edit({ op: 'route', from, to, pis, single: false }); out.push(r?.ok ?? false); await pause(300); };
    // the curve, constructed, then a blueprint piece on from its end (a joint of unlike track)
    const pi = P(-73.930, 40.652);
    await route(free(-73.955, 40.640), free(-73.905, 40.640), [{ x: pi.x, y: pi.y, level: 0 }]);
    out.push((await c.edit({ op: 'constructAll' }))?.ok ?? false);
    await pause(500);
    await route({ kind: 'node', node: near(-73.905, 40.640) }, free(-73.890, 40.640));
    // the straight, six clicks (plain joints)
    const xs = [-73.955, -73.945, -73.938, -73.930, -73.921, -73.912, -73.904];
    await route(free(xs[0], 40.672), free(xs[1], 40.672));
    for (let k = 2; k < xs.length; k++) await route({ kind: 'node', node: near(xs[k - 1], 40.672) }, free(xs[k], 40.672));
    // a station on one joint, a branch from another
    out.push((await c.edit({ op: 'addStation', at: { kind: 'node', node: near(-73.921, 40.672) }, platform: 200, name: 'Existing' }))?.ok ?? false);
    await pause(300);
    await route({ kind: 'node', node: near(-73.938, 40.672) }, free(-73.938, 40.662));
    await pause(800);
    return out;
  });
  console.log('track edits ok:', JSON.stringify(made));
  // the samples, and whether a station fits at each (the track model's own answer)
  const samples = await page.evaluate(async ([SAMPLE_M]) => {
    const tw = window.tw, w = tw.world.value, net = tw.renderer.net, R = 6371008.8, RAD = Math.PI / 180;
    const m = (a, b) => Math.hypot((a[0] - b[0]) * Math.cos(a[1] * RAD), a[1] - b[1]) * RAD * R;
    const out = [];
    for (const e of w.edges) {
      const pts = net.edgePts.get(e.id);
      if (!pts) continue;
      const ll = [];
      for (let k = 0; k < pts.length; k += 3) ll.push(tw.toLngLat(pts[k], pts[k + 1]));
      let total = 0;
      const cum = [0];
      for (let k = 1; k < ll.length; k++) cum.push(total += m(ll[k], ll[k - 1]));
      for (let s = SAMPLE_M / 2; s < total; s += SAMPLE_M) {
        let k = 1;
        while (k < cum.length - 1 && cum[k] < s) k++;
        const t = (s - cum[k - 1]) / Math.max(1e-9, cum[k] - cum[k - 1]);
        out.push({ edge: e.id, s: Math.round(s), lng: ll[k - 1][0] + (ll[k][0] - ll[k - 1][0]) * t, lat: ll[k - 1][1] + (ll[k][1] - ll[k - 1][1]) * t });
      }
    }
    for (const p of out) {
      const at = { kind: 'edge', edge: p.edge, ...tw.lngLatToPack(p.lng, p.lat) };
      const r = await new Promise((res) => tw.client.checkStation(at, 200, (ok, issues) => res({ ok, issues })));
      p.fits = r.ok;
      p.fitWhy = r.issues.join(',');
    }
    return out;
  }, [SAMPLE_M]);
  console.log(`samples ${samples.length}, a station fits at ${samples.filter((s) => s.fits).length}`);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'station'; });
  const rect = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { left: r.left, top: r.top, w: r.width, h: r.height }; });
  const results = {};
  for (const zoom of ZOOMS) {
    const rows = [];
    for (const p of samples) {
      const xy = await page.evaluate(([p, zoom]) => {
        const m = window.tw.map;
        let q = m.project([p.lng, p.lat]);
        const r = m.getContainer().getBoundingClientRect();
        if (Math.abs(m.getZoom() - zoom) > 1e-6 || q.x < r.width * 0.15 || q.x > r.width * 0.85 || q.y < r.height * 0.15 || q.y > r.height * 0.85) {
          m.jumpTo({ center: [p.lng, p.lat], zoom });
          q = m.project([p.lng, p.lat]);
          return { x: q.x, y: q.y, jumped: true };
        }
        return { x: q.x, y: q.y, jumped: false };
      }, [p, zoom]);
      if (xy.jumped) await page.waitForTimeout(400);
      await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
      await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
      // the worker's answer for this pointer, else the guess after a while
      await page.waitForFunction(() => { const t = window.tw.tools; return t.answer && t.answer.key === t.altsKey(); }, undefined, { timeout: 1500, polling: 30 }).catch(() => {});
      const got = await page.evaluate(([p]) => {
        const tw = window.tw, t = tw.tools, v = t.verdict(), s = t.stationSpot(), R = 6371008.8, RAD = Math.PI / 180;
        const m = (a, b) => Math.hypot((a[0] - b[0]) * Math.cos(a[1] * RAD), a[1] - b[1]) * RAD * R;
        if (!s || s.kind === 'free') return { kind: s?.kind ?? null, ok: v?.ok ?? null };
        const ll = tw.toLngLat(s.lx, s.ly), ptr = t.pointer ? [t.pointer.lng, t.pointer.lat] : null;
        const node = s.kind === 'node' ? tw.world.value.nodes.get(s.node) : null;
        // an end station's platform runs back from its node: its middle is half a platform in
        const d = m(ll, [p.lng, p.lat]), end = node && node.ports <= 1;
        return { kind: s.kind, ports: node?.ports ?? null, ok: v?.ok ?? null, answered: t.answer?.key === t.altsKey(), why: v?.why ?? '', d: end ? Math.abs(d - 100) : d, ptr: ptr ? m(ptr, [p.lng, p.lat]) : null };
      }, [p]);
      rows.push({ ...p, ...got });
    }
    const fits = rows.filter((r) => r.fits), no = rows.filter((r) => !r.fits);
    const moved = fits.filter((r) => r.ok && r.d > MOVED_M), refused = fits.filter((r) => r.ok === false);
    const sum = {
      zoom, samples: rows.length, fits: fits.length,
      fitsMoved: moved.length, fitsRefused: refused.length, unanswered: rows.filter((r) => !r.answered).length,
      noFitMoved: no.filter((r) => r.ok).length, noFitRefused: no.filter((r) => r.ok === false).length,
      maxMoved: Math.round(Math.max(0, ...moved.map((r) => r.d))),
      movedToNode: moved.filter((r) => r.kind === 'node').length,
    };
    console.log(JSON.stringify(sum));
    results[zoom] = { sum, rows };
    // a picture at the worst move, and at a refusal where it fits
    for (const [name, r] of [['moved', moved.sort((a, b) => b.d - a.d)[0]], ['refused', refused[0]]]) {
      if (!r) continue;
      const xy = await page.evaluate(([p, zoom]) => { const m = window.tw.map; m.jumpTo({ center: [p.lng, p.lat], zoom }); const q = m.project([p.lng, p.lat]); return { x: q.x, y: q.y }; }, [r, zoom]);
      await page.waitForTimeout(500);
      await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
      await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
      await page.waitForTimeout(900);
      await page.screenshot({ path: `${OUT}${LABEL}-z${zoom}-${name}.png`, clip: { x: rect.left + xy.x - 300, y: rect.top + xy.y - 200, width: 600, height: 400 } });
    }
  }
  // 85 m from the plain joint at -73.945, zoom 13: the ghost pulled onto the joint before T-1870
  {
    const at = [-73.944, 40.672];
    const xy = await page.evaluate((at) => { const m = window.tw.map; m.jumpTo({ center: at, zoom: 13 }); const q = m.project(at); return { x: q.x, y: q.y }; }, at);
    await page.waitForTimeout(600);
    await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
    await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
    await page.waitForTimeout(900);
    await page.screenshot({ path: `${OUT}${LABEL}-z13-near-joint.png`, clip: { x: rect.left + xy.x - 150, y: rect.top + xy.y - 100, width: 300, height: 200 } });
  }
  // a short stub branching off the straight's last piece: no room either way, so the junction is named
  const stub = await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, ll = (n) => { const o = tw.world.value.nodes.get(n); return [o.lng, o.lat]; };
    const e = w.edges.find((e) => { const [a, b] = [ll(e.a), ll(e.b)]; return Math.abs(a[1] - 40.672) < 1e-4 && Math.abs(b[1] - 40.672) < 1e-4 && Math.min(a[0], b[0]) < -73.9085 && Math.max(a[0], b[0]) > -73.9085; });
    const from = tw.lngLatToPack(-73.9085, 40.672), to = tw.lngLatToPack(-73.9068, 40.6727);
    const r = await tw.client.edit({ op: 'route', from: { kind: 'edge', edge: e.id, x: from.x, y: from.y }, to: { kind: 'free', x: to.x, y: to.y, level: 0 }, pis: [], single: false });
    await new Promise((d) => setTimeout(d, 800));
    const added = tw.world.value.edges.filter((x) => !w.edges.some((o) => o.id === x.id));
    const s = added.map((x) => ({ id: x.id, m: x.lengthM, a: ll(x.a), b: ll(x.b) })).find((x) => Math.max(x.a[1], x.b[1]) > 40.6722);
    if (!s) return { ok: r?.ok, issues: r?.issues };
    // most of the way out, where the straight is out of reach
    const pts = tw.renderer.net.edgePts.get(s.id), k = Math.floor((pts.length / 3) * (s.a[1] > s.b[1] ? 0.2 : 0.8)) * 3;
    return { ok: r?.ok, m: s.m, mid: tw.toLngLat(pts[k], pts[k + 1]) };
  });
  console.log('stub', JSON.stringify(stub));
  for (const [name, at] of [['stub', stub.mid ?? [-73.9076, 40.6724]], ['on-station', [-73.9215, 40.672]]]) {
    const xy = await page.evaluate((at) => { const m = window.tw.map; m.jumpTo({ center: at, zoom: 16 }); const q = m.project(at); return { x: q.x, y: q.y }; }, at);
    await page.waitForTimeout(600);
    await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
    await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
    await page.waitForTimeout(900);
    const why = await page.evaluate(() => window.tw.tools.verdict()?.why ?? '');
    console.log(`${name}: "${why}"`);
    await page.screenshot({ path: `${OUT}${LABEL}-z16-${name}.png`, clip: { x: rect.left + xy.x - 300, y: rect.top + xy.y - 200, width: 600, height: 400 } });
  }
  // the texts where a station does not fit, as shown
  const whys = {};
  for (const z of ZOOMS) for (const r of results[z].rows) if (r.ok === false) whys[r.why] = (whys[r.why] ?? 0) + 1;
  console.log('refusal texts:', JSON.stringify(whys));
  await writeFile(`${OUT}${LABEL}-sweep.json`, JSON.stringify(results, null, 1));
} catch (e) {
  failed = true;
  console.log(`FAIL  t1870 sweep: ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
