// T-1799: lines sharing track drawn side by side. On the shared-track New York save (T-1305) and the
// real one: the bands' order crosses nowhere a pair of lines runs on together (only where they part
// or meet, which the pairwise rule leaves to its routes), a click on each band of the busiest shared
// edges picks that band's line, and each train is drawn on its line's band.
//   node trainworld/checks/t1799-bands.mjs [url]     (default http://localhost:6197/)
// One PASS or FAIL line per check; figures in artifacts/trainworld/t1799/check.json.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { ARTIFACTS, REAL_SAVE, launchChrome, importSave, collectErrors, GL } from './lib.mjs';

const URL = process.argv[2] ?? 'http://localhost:6197/';
const SAVES = [['shared', `${ARTIFACTS}t1305-shared/nyc_real.save`], ['real', REAL_SAVE]];
const out = {};
let failed = 0;
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1799c-', size: [1400, 900], args: GL });
const page = ctx.pages()[0] ?? await ctx.newPage();
const errors = [];
collectErrors(page, errors);
await page.goto(`${URL}?debug=1&new=1&paused=1&demandWorkers=1`);
await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
for (const [name, file] of SAVES) {
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, await readFile(file));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.renderer.net?.lines.count > 0 && window.tw.client.lastLines?.length > 0, undefined, { timeout: 600000 });
  await new Promise((r) => setTimeout(r, 2000));
  const r = await page.evaluate(async () => {
    const tw = window.tw, n = tw.renderer.net, lines = tw.client.lastLines, edges = tw.client.lastEdges;
    const { lineBands, BAND_RANGE } = await import('/src/map/lineBands.ts');
    const t0 = performance.now();
    const b = lineBands(lines, edges, n.edgePts);
    const ms = performance.now() - t0;
    const ends = new Map(edges.map((e) => [e.id, e]));
    const slot = new Map();
    for (const [e, ids] of b.order) slot.set(e, new Map(ids.map((id, i) => [id, i])));
    // left of `m` in `l`'s travel along path entry p
    const leftOf = (p, l, m) => { const s = slot.get(p >> 1); return (p & 1) === 0 ? s.get(l) < s.get(m) : s.get(l) > s.get(m); };
    // where l takes entry p then q through a node: the other lines doing the same, and those on p that leave by another edge
    let together = 0, cycles = 0, parting = 0, pairsTogether = 0;
    const bad = [];
    const rail = lines.filter((l) => l.ok && !l.bus && l.path.length);
    const steps = new Map();
    for (const l of rail) for (let k = 0; k + 1 < l.path.length; k++) {
      const key = `${l.path[k] >> 1}`;
      if (!steps.has(key)) steps.set(key, []);
      steps.get(key).push({ id: l.id, p: l.path[k], q: l.path[k + 1] });
    }
    const dirAt = (edge, node) => {
      const p = n.edgePts.get(edge), e = ends.get(edge), m = p.length / 3;
      const i0 = e.a === node ? 0 : m - 1, i1 = e.a === node ? Math.min(m - 1, 3) : Math.max(0, m - 4);
      const dx = p[i1 * 3] - p[i0 * 3], dy = -(p[i1 * 3 + 1] - p[i0 * 3 + 1]), d = Math.hypot(dx, dy) || 1;
      return [dx / d, dy / d];
    };
    const turn = (inE, out, node) => { const [ax, ay] = dirAt(inE, node), [bx, by] = dirAt(out, node); return Math.atan2(-ax * by + ay * bx, -ax * bx - ay * by); };
    for (const [, v] of steps) {
      if (!slot.has(v[0].p >> 1)) continue;
      for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) {
        const x = v[i], y = v[j];
        if (x.id === y.id) continue;
        // same direction on this edge only (a pair running opposite ways is seen from the other side too)
        if ((x.p & 1) !== (y.p & 1)) continue;
        const e = ends.get(x.p >> 1), node = (x.p & 1) === 0 ? e.b : e.a;
        if (x.q === y.q) {
          if (!slot.has(x.q >> 1)) continue;
          pairsTogether++;
          // the rule itself, in x's travel, on both edges: the same when the order broke a circle of
          // partings among three or more lines (one of them has to cross somewhere), else a fault
          const ruleLeft = (p) => { const r = b.rule(p >> 1, x.id, y.id); return (p & 1) === 0 ? r < 0 : r > 0; };
          if (leftOf(x.p, x.id, y.id) !== leftOf(x.q, x.id, y.id) && ruleLeft(x.p) === ruleLeft(x.q)) cycles++;
          else if (leftOf(x.p, x.id, y.id) !== leftOf(x.q, x.id, y.id)) { together++; if (bad.length < 8) { const lx = rail.find((l) => l.id === x.id), ly = rail.find((l) => l.id === y.id); const twice = (l, e) => l.path.filter((p) => p >> 1 === e).length; bad.push({ lines: [x.id, y.id], edges: [x.p, x.q], rule: [b.rule(x.p >> 1, x.id, y.id), b.rule(x.q >> 1, x.id, y.id)], orders: [b.order.get(x.p >> 1), b.order.get(x.q >> 1)], twice: [twice(lx, x.p >> 1), twice(ly, x.p >> 1), twice(lx, x.q >> 1), twice(ly, x.q >> 1)] }); } }
        } else if ((x.q >> 1) !== (y.q >> 1)) {
          // they part here: a crossing when the one on the left turns right of the other
          const tx = turn(x.p >> 1, x.q >> 1, node), ty = turn(y.p >> 1, y.q >> 1, node);
          if (tx !== ty && leftOf(x.p, x.id, y.id) !== tx > ty) parting++;
        }
      }
    }
    const shared = [...b.order.values()];
    // the busiest shared edges: a click on each band at zoom 15
    const busiest = [...b.order.entries()].sort((a, c) => c[1].length - a[1].length).slice(0, 6);
    const picks = { ok: 0, bad: 0, over: 0, misses: [] };
    const trains = { n: 0, offPx: [] };
    const L = n.lines;
    for (const [edge, ids] of busiest) {
      // the middle of the edge's first line's strokes
      let seg = -1;
      for (let i = 0; i < L.count && seg < 0; i++) if (L.edge[i] === edge && L.colour[i] === ids[0]) seg = i;
      const segs = [];
      for (let i = seg; i >= 0 && i < L.count && L.edge[i] === edge; i++) segs.push(i);
      const mid = segs[segs.length >> 1];
      const mx = (L.seg[mid * 4] + L.seg[mid * 4 + 2]) / 2, my = (L.seg[mid * 4 + 1] + L.seg[mid * 4 + 3]) / 2;
      const ll = tw.toLngLat(mx, my);
      tw.map.jumpTo({ center: ll, zoom: 15 });
      await new Promise((r) => setTimeout(r, 800));
      const bandPx = tw.renderer.bandPx;
      for (const id of ids) {
        let i = -1, bd = Infinity;
        for (let k = 0; k < L.count; k++) {
          if (L.edge[k] !== edge || L.colour[k] !== id) continue;
          const d = Math.hypot((L.seg[k * 4] + L.seg[k * 4 + 2]) / 2 - mx, (L.seg[k * 4 + 1] + L.seg[k * 4 + 3]) / 2 - my);
          if (d < bd) { bd = d; i = k; }
        }
        const [x0, y0] = tw.overlay.toScreen(L.seg[i * 4], L.seg[i * 4 + 1]), [x1, y1] = tw.overlay.toScreen(L.seg[i * 4 + 2], L.seg[i * 4 + 3]);
        const s = (L.band[i * 4] + L.band[i * 4 + 1]) / 2 * BAND_RANGE / 32767 * bandPx, dl = Math.hypot(x1 - x0, y1 - y0) || 1;
        const px = (x0 + x1) / 2 + (y1 - y0) / dl * s, py = (y0 + y1) / 2 - (x1 - x0) / dl * s;
        const hit = tw.pick(tw.overlay, px, py, false, null, true);
        // a band of a parallel edge's ribbon drawn over this one (four-track trunks, stage 3's corridors) is not this edge's fault
        const other = hit?.kind === 'line' && !ids.includes(Number(hit.line));
        if (hit?.kind === 'line' && hit.line === String(id)) picks.ok++;
        else if (other) picks.over++;
        else { picks.bad++; if (picks.misses.length < 8) picks.misses.push({ edge, line: id, got: hit }); }
      }
      // trains in view on this stretch: their drawn place against their line's band
      const r = tw.renderer, t = tw.clock.now();
      for (let k = 0; k < r.tripCount; k++) {
        const prof = r.trips[k * 2], dep = r.trips[k * 2 + 1] + r.tripsEpoch;
        const at = tw.tripAt(n, prof, dep, t, r.tripEnds ? r.tripEnds[k] + r.tripsEpoch : -Infinity, r.laneShiftM, r.bandShiftM);
        if (!at) continue;
        const [sx, sy] = tw.overlay.toScreen(at.x, at.y);
        if (sx < 0 || sy < 0 || sx > tw.overlay.cssW || sy > tw.overlay.cssH) continue;
        // the nearest of its own line's strokes, shifted as drawn
        let best = Infinity, onShared = false;
        for (let q = 0; q < L.count; q++) {
          if (L.colour[q] !== at.line) continue;
          const [a0, b0] = tw.overlay.toScreen(L.seg[q * 4], L.seg[q * 4 + 1]), [a1, b1] = tw.overlay.toScreen(L.seg[q * 4 + 2], L.seg[q * 4 + 3]);
          const dl2 = Math.hypot(a1 - a0, b1 - b0) || 1, nx = (b1 - b0) / dl2, ny = -(a1 - a0) / dl2;
          const s0 = L.band[q * 4] * BAND_RANGE / 32767 * bandPx, s1 = L.band[q * 4 + 1] * BAND_RANGE / 32767 * bandPx;
          const ax = a0 + nx * s0, ay = b0 + ny * s0, bx = a1 + nx * s1, by = b1 + ny * s1;
          const vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy;
          const u = l2 ? Math.max(0, Math.min(1, ((sx - ax) * vx + (sy - ay) * vy) / l2)) : 0;
          const d = Math.hypot(ax + u * vx - sx, ay + u * vy - sy);
          if (d < best) { best = d; onShared = slot.has(L.edge[q]); }
        }
        if (onShared) { trains.n++; trains.offPx.push(best); }
      }
    }
    trains.offPx.sort((a, c) => a - c);
    return { ms, sharedEdges: shared.length, maxLines: Math.max(0, ...shared.map((v) => v.length)), pairsTogether, together, cycles, parting, bad, picks, trains: { n: trains.n, p50: trains.offPx[trains.offPx.length >> 1] ?? null, p95: trains.offPx[Math.floor(trains.offPx.length * 0.95)] ?? null, max: trains.offPx.at(-1) ?? null } };
  });
  out[name] = r;
  line(r.together === 0, `${name}: bands never cross where two lines run on together`, `${r.pairsTogether} pairs through nodes together, ${r.together} cross against the pairwise rule, ${r.cycles} where three or more lines' partings go round a circle; ${r.parting} cross where they part or meet; ${r.sharedEdges} shared edges, up to ${r.maxLines} lines; order in ${r.ms.toFixed(0)} ms`);
  if (r.sharedEdges) {
    line(r.picks.bad === 0, `${name}: a click on each band of the busiest shared edges picks its line`, `${r.picks.ok} of ${r.picks.ok + r.picks.bad + r.picks.over}, ${r.picks.over} under a parallel edge's band${r.picks.misses.length ? ' ' + JSON.stringify(r.picks.misses) : ''}`);
    line(r.trains.n > 0 && r.trains.p95 < 1.5, `${name}: trains drawn on their line's band`, `${r.trains.n} trains on shared track, off the band's middle p50 ${r.trains.p50?.toFixed(2)} px, p95 ${r.trains.p95?.toFixed(2)}, max ${r.trains.max?.toFixed(2)}`);
  }
}
line(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
await mkdir(`${ARTIFACTS}t1799`, { recursive: true });
await writeFile(`${ARTIFACTS}t1799/check.json`, JSON.stringify(out, null, 1));
await close();
process.exit(failed ? 1 : 0);
