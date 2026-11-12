// T-3212: at a station on two-way single track the track is drawn splitting into two platform tracks,
// and the trains stand on them. A new sandbox game gets 3 km of single track A - M - B (stations of
// 200 m, one line with two trains), a double-track line C - D and a tram line E - F (tram stops; tram
// track is never single). Looks at the drawn track's lanes (`Strokes.lane`, 255 two tracks, 0 one) at each station and
// between them, the station flag 16 (a platform each side), and the train lanes (`sampleLane`) against
// the line's drawn lanes all along A - B. Then runs the clock until two trains stand side by side at a
// station and takes pictures of it at a few zooms, light and dark, and of D and F with no trains about,
// to artifacts/trainworld/t3212/<tag>-*.png. From the maps root:
//   node trainworld/checks/t3212-station-split.mjs [url] [tag]
// With tag "before" (the code before T-3212) only the pictures are taken.
import { join } from 'node:path';
import { mkdir, writeFile } from 'node:fs/promises';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6135/';
const tag = process.argv[3] ?? 'after';
const OUT = join(ARTIFACTS, 't3212');
await mkdir(OUT, { recursive: true });
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };

const LAT = 40.652, LNG0 = -73.985, KM = 1 / (111.32 * Math.cos((LAT * Math.PI) / 180)), N = 1 / 111.32;
const A = [LNG0, LAT], M = [LNG0 + 1.5 * KM, LAT], B = [LNG0 + 3 * KM, LAT];
const C = [LNG0, LAT + 0.7 * N], D = [LNG0 + 1.5 * KM, LAT + 0.7 * N];
const E = [LNG0, LAT - 0.7 * N], F = [LNG0 + 1 * KM, LAT - 0.7 * N];
// G: single track one side, double the other; J: a station whose split runs through a plain node K 120 m on into double track
const H0 = [LNG0, LAT - 1.4 * N], G = [LNG0 + 1 * KM, LAT - 1.4 * N], H1 = [LNG0 + 2 * KM, LAT - 1.4 * N];
const J0 = [LNG0, LAT + 1.4 * N], J = [LNG0 + 1 * KM, LAT + 1.4 * N], K = [LNG0 + 1.12 * KM, LAT + 1.4 * N], K1 = [LNG0 + 2 * KM, LAT + 1.4 * N];
const chrome = await launchChrome({ prefix: 'tw-t3212-', size: [1300, 850] });
const errors = [];
const out = {};
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&demandWorkers=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(1000);
  out.built = await page.evaluate(async ({ A, M, B, C, D, E, F, H0, G, H1, J0, J, K, K1 }) => {
    const tw = window.tw, c = tw.client, pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const xy = (p) => { const q = tw.lngLatToPack(p[0], p[1]); return [q.x, q.y]; };
    const near = (p) => { const [x, y] = xy(p); let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const free = (p) => { const [x, y] = xy(p); return { kind: 'free', x, y, level: 0 }; };
    const ok = [];
    ok.push((await c.edit({ op: 'route', from: free(A), to: free(M), pis: [], single: true }))?.ok);
    await pause(300);
    ok.push((await c.edit({ op: 'route', from: { kind: 'node', node: near(M) }, to: free(B), pis: [], single: true }))?.ok);
    await pause(300);
    ok.push((await c.edit({ op: 'route', from: free(C), to: free(D), pis: [], single: false }))?.ok);
    await pause(300);
    ok.push((await c.edit({ op: 'route', from: free(E), to: free(F), pis: [], single: false, tram: true }))?.ok);
    await pause(300);
    for (const [p, q, single] of [[H0, G, true], [G, H1, false], [J0, J, true], [J, K, true], [K, K1, false]]) {
      const from = tw.world.value.nodes.size && [...tw.world.value.nodes.values()].some((n) => Math.hypot(n.x - xy(p)[0], n.y - xy(p)[1]) < 1) ? { kind: 'node', node: near(p) } : free(p);
      ok.push((await c.edit({ op: 'route', from, to: free(q), pis: [], single }))?.ok);
      await pause(300);
    }
    for (const p of [A, M, B, C, D, G, J]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(200); }
    for (const p of [E, F]) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 40, name: '' }); await pause(200); }
    await c.edit({ op: 'constructAll' });
    await pause(500);
    const lines = {};
    for (const [name, stops, colour] of [['amb', [A, M, B], '#d9467a'], ['cd', [C, D], '#2f7fd0'], ['ef', [E, F], '#3a9a4a']]) {
      const r = await c.edit({ op: 'addLine', stops: stops.map(near), name: '', colour });
      if (r?.line !== undefined) await c.edit({ op: 'setTrainCount', line: r.line, trains: [2, 2, 2] });
      lines[name] = r?.line;
      await pause(300);
    }
    await pause(1500);
    const nodes = Object.fromEntries(Object.entries({ A, M, B, C, D, E, F, G, J }).map(([k, p]) => [k, near(p)]));
    return { ok, lines, nodes, edges: tw.world.value.edges.map((e) => [e.id, e.a, e.b, e.tracks, e.kind]) };
  }, { A, M, B, C, D, E, F, H0, G, H1, J0, J, K, K1 });
  console.log('built', JSON.stringify(out.built));
  check(out.built.ok.every(Boolean) && Object.values(out.built.lines).every((l) => l !== undefined), 'the three lines built');
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0 && window.tw.renderer.net?.track?.count > 0, undefined, { timeout: 60000 });
  await page.waitForTimeout(1000);

  // the drawn lanes
  out.lanes = await page.evaluate(async ({ nodes, lines, H0, G, J0, J, K, KM }) => {
    const tw = window.tw, n = tw.renderer.net, T = n.track, Ls = n.lines;
    const st = (id) => { const i = n.stationIds.indexOf(String(id)); return i < 0 ? null : { x: n.stations[i * 3], y: n.stations[i * 3 + 1], flags: n.stations[i * 3 + 2] }; };
    // the lane of the track segment nearest a point (the larger of its two ends' when it is at an end)
    const laneNear = (x, y) => {
      let best = -1, bd = 1e18, bf = 0;
      for (let i = 0; i < T.count; i++) {
        const x0 = T.seg[i * 4], y0 = T.seg[i * 4 + 1], x1 = T.seg[i * 4 + 2], y1 = T.seg[i * 4 + 3];
        const dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy;
        const f = L2 > 0 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / L2)) : 0;
        const d = Math.hypot(x0 + dx * f - x, y0 + dy * f - y);
        if (d < bd - 1e-6) { bd = d; best = i; bf = f; }
      }
      return best < 0 ? -1 : Math.round(T.lane[best * 2] + (T.lane[best * 2 + 1] - T.lane[best * 2]) * bf);
    };
    const s = Object.fromEntries(Object.entries(nodes).map(([k, id]) => [k, st(id)]));
    const r = { flags: {}, at: {} };
    for (const [k, v] of Object.entries(s)) { r.flags[k] = v ? v.flags : -1; r.at[k] = v ? laneNear(v.x, v.y) : -1; }
    const mid = (p, q, f = 0.5) => laneNear(p.x + (q.x - p.x) * f, p.y + (q.y - p.y) * f);
    r.at.AMmid = mid(s.A, s.M);
    r.at.MBmid = mid(s.M, s.B);
    r.at.EFmid = mid(s.E, s.F);
    const geo = await import('/src/map/geo.ts');
    const laneAt = (p) => laneNear(...geo.toLocal(p[0], p[1]));
    r.at.H0Gmid = laneAt([(H0[0] + G[0]) / 2, G[1]]);
    r.at.Gdouble = laneAt([G[0] + 0.03 * KM, G[1]]);
    r.at.J0Jmid = laneAt([(J0[0] + J[0]) / 2, J[1]]);
    r.at.Kdouble = laneAt([K[0] + 0.02 * KM, K[1]]);
    r.at.JK = laneAt([J[0] + 0.11 * KM, K[1]]);
    // train lanes against the line's drawn lanes, along run 0 of A - M - B (by share of the way)
    const line = lines.amb, f0 = n.lineTable[line * 4], cnt = n.lineTable[line * 4 + 1], step = n.lineTable[line * 4 + 3];
    const segs = [];
    for (let i = 0; i < Ls.count; i++) if (Ls.colour[i] === line) segs.push(i);
    const segLen = (i) => Math.hypot(Ls.seg[i * 4 + 2] - Ls.seg[i * 4], Ls.seg[i * 4 + 3] - Ls.seg[i * 4 + 1]);
    const total = segs.length ? Ls.dist[segs[segs.length - 1]] + segLen(segs[segs.length - 1]) - Ls.dist[segs[0]] : 0;
    const L = n.meta[line * 6 * 4 + 3];
    const drawnAt = (share) => {
      const want = Ls.dist[segs[0]] + share * total;
      for (const i of segs) {
        const a = Ls.dist[i], len = segLen(i);
        if (want <= a + len + 1e-6) { const u = len > 0 ? Math.max(0, Math.min(1, (want - a) / len)) : 0; return (Ls.lane[i * 2] + (Ls.lane[i * 2 + 1] - Ls.lane[i * 2]) * u) / 255; }
      }
      return Ls.lane[segs[segs.length - 1] * 2 + 1] / 255;
    };
    let worst = 0, worstAt = 0;
    const profile = [];
    for (let k = 0; k < cnt; k++) {
      const v = n.sampleLane[f0 + k], dv = drawnAt((k * step) / L);
      if (Math.abs(v - dv) > worst) { worst = Math.abs(v - dv); worstAt = k * step; }
      if (k % 5 === 0) profile.push([k * step, +v.toFixed(2), +dv.toFixed(2)]);
    }
    r.trains = { worst, worstAt, L, profile };
    return r;
  }, { ...out.built, H0, G, J0, J, K, KM });
  console.log('lanes', JSON.stringify({ flags: out.lanes.flags, at: out.lanes.at, worst: out.lanes.trains.worst, worstAt: out.lanes.trains.worstAt }));
  if (tag !== 'before') {
    const { at, flags, trains } = out.lanes;
    check(at.M === 255 && at.A === 255 && at.B === 255, `single track drawn as two at A, M, B (lanes ${at.A}, ${at.M}, ${at.B})`);
    check(at.AMmid === 0 && at.MBmid === 0, `one track between the stations (lanes ${at.AMmid}, ${at.MBmid})`);
    check([flags.A, flags.M, flags.B].every((f) => (f & 16) !== 0), 'A, M, B have a platform each side (flag 16)');
    check(at.D === 255 && (flags.D & 16) !== 0, `the double-track station as before (lane ${at.D}, flag 16)`);
    check(at.F === 255 && at.E === 255 && at.EFmid === 255 && (flags.F & 16) !== 0 && (flags.E & 16) !== 0, `the tram stops as before, on double tram track (tram track is never single; lanes ${at.E}, ${at.F}, flag 16)`);
    check(at.G === 255 && at.Gdouble === 255 && at.H0Gmid === 0 && (flags.G & 16) !== 0, `single track meeting double at a station: split on the single side, no taper on the double (lanes ${at.H0Gmid}, ${at.G}, ${at.Gdouble})`);
    check(at.J === 255 && at.JK === 255 && at.Kdouble === 255 && at.J0Jmid === 0, `a split reaching double track within its ease stays two tracks to it (lanes ${at.J0Jmid}, ${at.J}, ${at.JK}, ${at.Kdouble})`);
    check(trains.worst < 0.06, `trains' lanes follow the drawn lanes along A - B (worst ${trains.worst.toFixed(3)} at ${Math.round(trains.worstAt)} m)`);
  }

  const shoot = async (name, center, zoom) => {
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [center, zoom]);
    for (const sh of ['light', 'dark']) {
      await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, sh);
      await page.waitForTimeout(1200);
      await page.screenshot({ path: join(OUT, `${tag}-${name}-z${zoom}-${sh}.png`), clip: { x: 150, y: 225, width: 1000, height: 400 } });
    }
    await page.evaluate((x) => { document.documentElement.dataset.shade = x; window.tw.shade.value = x; }, 'light');
  };
  // D and F at night, no trains about (for before and after to be compared)
  await page.evaluate(() => { const c = window.tw.clock, day = Math.floor(c.now() / 86400) + 1; c.setRunning(false); c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 2.5 * 3600); });
  await page.waitForTimeout(1500);
  await shoot('double', D, 17.7);
  await shoot('tram', F, 17.7);
  await shoot('empty-M', M, 17.7);
  await shoot('empty-B', [B[0] - 0.08 * KM, B[1]], 17.7);
  await shoot('mixed-G', G, 17.7);
  await shoot('plain-J', [J[0] + 0.05 * KM, J[1]], 17.7);

  // two trains standing side by side at a station of A - M - B
  const met = await page.evaluate(async ({ line, nodes }) => {
    const tw = window.tw, c = tw.clock, r = tw.renderer, day = Math.floor(c.now() / 86400) + 1;
    c.setRunning(false);
    c.set((day + (day % 7 >= 5 ? 2 : 0)) * 86400 + 9 * 3600);
    c.setMultiple(1);
    c.setRunning(true);
    const n0 = r.net, L0 = n0.lineTable[line * 4 + 2];
    return await new Promise((done) => {
      const t0 = performance.now();
      const step = () => {
        if (performance.now() - t0 > 150000) return done(null);
        const n = r.net, t = r.guard.lastTime, seen = [];
        for (let i = 0; i < r.tripCount; i++) {
          const p = r.trips[i * 2];
          if (Math.floor(p / 6) !== line) continue;
          const pos = tw.tripAt(n, p, r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i), t, r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity, 0, 0, undefined, 0, 0, r.tripBays ?? undefined, i * 4);
          if (!pos) continue;
          const L = n.meta[p * 4 + 3];
          seen.push({ run: pos.run, d: pos.run === 0 ? pos.s : L - pos.s, L });
        }
        // both within 120 m of the same station (M at half way, B at the end), one each way
        for (const [name, at] of [['M', 0.5], ['B', 1]]) {
          const here = seen.filter((q) => Math.abs(q.d - at * q.L) < 120);
          if (here.length >= 2 && here.some((q) => q.run === 0) && here.some((q) => q.run === 1)) { c.setRunning(false); return done(name); }
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }, { line: out.built.lines.amb, nodes: out.built.nodes });
  console.log('met at', met);
  check(met !== null, 'two trains met side by side at a station of A - M - B');
  if (met) {
    await page.waitForTimeout(1500);
    const where = met === 'M' ? M : [B[0] - 0.08 * KM, B[1]];
    for (const z of [16, 16.7, 17.3, 18]) await shoot(`meet-${met}`, where, z);
  }
  out.errors = errors;
  await writeFile(join(OUT, `${tag}.json`), JSON.stringify(out, null, 1));
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
  check(errors.length === 0, 'no console errors');
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
