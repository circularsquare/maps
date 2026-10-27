// T-1465: a bus route drawn along roads. In a dense grid (Manhattan) and a city with curvy roads (San
// Francisco's Twin Peaks), the real drawing tool is driven with the mouse: points are clicked, the route
// is made with Enter, and the checks are that the route follows streets (every sample along its road
// legs is within a few px of a drawn road of the basemap), that its road length is near the straight
// line's times a plausible detour and below 3 times, that a click far from any road stays straight
// (length counted at 1.2), that the pointer moves cost little main-thread time, that redrawing the route
// keeps its corners, and a save brings it back. Screenshots go to artifacts/trainworld/t1465/. Uses the
// Vite already on `port` (dev-nowatch.mjs). From the maps root:
//   node trainworld/checks/t1465-bus-roads.mjs [port]
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors } from './lib.mjs';

const OUT = join(ARTIFACTS, 't1465');
await mkdir(OUT, { recursive: true });
const port = Number(process.argv[2] ?? 6365);
const url = `http://localhost:${port}/`;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const PLACES = [
  // Manhattan: 8th Ave at 34th St, then up Central Park West, then across to the East Side
  { name: 'manhattan', clicks: [[-73.9965, 40.7510], [-73.9795, 40.7680], [-73.9570, 40.7720]], grid: true },
  // San Francisco, Russian Hill: the crooked part of Lombard Street, over to Hyde and up
  { name: 'sf', clicks: [[-122.4187, 37.8021], [-122.4250, 37.8040], [-122.4220, 37.8080]], grid: false },
  // Lisbon, Alfama: narrow twisting streets
  { name: 'lisbon', clicks: [[-9.1300, 38.7110], [-9.1370, 38.7140], [-9.1400, 38.7180]], grid: false },
  // Dongguan, Shatian: Jiaju Avenue, a divided road, from the roundabout at Hengliu South Road east
  // across Gangkou Boulevard (T-1512: a hairpin at the junction and at the clicks)
  { name: 'dongguan', clicks: [[113.6146, 22.9129], [113.6227, 22.9124], [113.6285, 22.9100]], grid: false, divided: true },
];

const chrome = await launchChrome({ prefix: process.env.PROFILE_PREFIX ?? 'tw-t1465-', profile: process.env.PROFILE_DIR, size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  const map = await page.$('#map');
  const box = await map.boundingBox();

  const screen = (p) => page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p);
  const settle = async () => {
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 });
    await page.waitForTimeout(800);
  };
  await page.evaluate(() => {
    const b = window.tw.tools.bus, times = { move: [], place: [] };
    window.__busTimes = times;
    for (const k of ['move', 'place']) {
      const orig = b[k].bind(b);
      b[k] = (...a) => { const t = performance.now(); const r = orig(...a); times[k].push(performance.now() - t); return r; };
    }
  });

  for (const place of PLACES) {
    console.log(`--- ${place.name}`);
    await page.evaluate(([cl]) => { window.tw.selection.value = null; const lo = [Math.min(...cl.map((p) => p[0])), Math.min(...cl.map((p) => p[1]))], hi = [Math.max(...cl.map((p) => p[0])), Math.max(...cl.map((p) => p[1]))]; window.tw.map.fitBounds([lo, hi], { padding: 140, maxZoom: 16, duration: 0 }); }, [place.clicks]);
    await settle();
    await page.evaluate(() => { window.tw.tools.exit?.(); window.tw.tools.bus.begin(null); window.__busTimes.move.length = 0; window.__busTimes.place.length = 0; });
    const stats0 = await page.evaluate(() => ({ ...window.tw.tools.bus.roads.stats }));
    let prev = null;
    const ends = [];
    for (let i = 0; i < place.clicks.length; i++) {
      const [x, y] = await screen(place.clicks[i]);
      const px = box.x + x, py = box.y + y;
      if (prev) await page.mouse.move(prev[0], prev[1]);
      // walk the pointer there in steps as a hand does
      await page.mouse.move(px, py, { steps: prev ? 30 : 1 });
      await page.waitForTimeout(900);
      if (i === place.clicks.length - 1) await page.screenshot({ path: join(OUT, `${place.name}-ghost.png`) });
      await page.mouse.click(px, py);
      await page.waitForTimeout(300);
      prev = [px, py];
    }
    await page.waitForTimeout(800);
    const st = await page.evaluate(() => ({ stats: { ...window.tw.tools.bus.roads.stats }, times: window.__busTimes }));
    const moves = st.times.move.slice().sort((a, b) => a - b);
    const places = st.times.place.slice().sort((a, b) => a - b);
    const q = (a, p) => a[Math.min(a.length - 1, Math.floor(a.length * p))] ?? 0;
    console.log(`pointer moves ${moves.length}: median ${q(moves, 0.5).toFixed(2)} ms, p95 ${q(moves, 0.95).toFixed(2)} ms, max ${moves.at(-1)?.toFixed(2)} ms; place median ${q(places, 0.5).toFixed(2)} ms, max ${places.at(-1)?.toFixed(2)} ms; legs searched ${st.stats.legs - stats0.legs}, slowest ${st.stats.max.toFixed(2)} ms, ${st.stats.tiles - stats0.tiles} tiles loaded`);
    check(q(moves, 0.95) < 8, `${place.name}: pointer moves stay cheap (p95 under 8 ms)`, `p95 ${q(moves, 0.95).toFixed(2)}`);
    check((moves.at(-1) ?? 0) < 60, `${place.name}: no pointer move over 60 ms`, `max ${moves.at(-1)?.toFixed(1)}`);

    await page.keyboard.press('Enter');
    await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0 && !window.tw.tools.bus.active, undefined, { timeout: 20000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const line = await page.evaluate(() => {
      const w = window.tw.world.value, l = w.save.lines[w.save.lines.length - 1];
      const b = w.lineStats[l.id].bus;
      return { id: l.id, num: l.num, route: b.routeLL, kinds: b.kinds, roadM: b.roadM };
    });
    const n = line.route.length / 2;
    const corners = line.kinds.filter((k) => k & 1).length;
    const roadPts = line.kinds.filter((k) => k & 2).length;
    console.log(`line ${line.num}: ${n} points, ${corners} corners, ${roadPts} on road, ${(line.roadM / 1000).toFixed(2)} km`);
    check(corners === place.clicks.length, `${place.name}: one corner per click`, `${corners}`);
    check(n > place.clicks.length * 2 && roadPts > n / 2, `${place.name}: the route follows roads`, `${n} points, ${roadPts} on road`);
    // length against the straight line through the corners
    const cs = []; for (let i = 0; i < n; i++) if (line.kinds[i] & 1) cs.push([line.route[2 * i], line.route[2 * i + 1]]);
    const m = (a, b) => { const k = Math.cos((a[1] * Math.PI) / 180); return Math.hypot((a[0] - b[0]) * 111320 * k, (a[1] - b[1]) * 110574); };
    let straight = 0; for (let i = 1; i < cs.length; i++) straight += m(cs[i - 1], cs[i]);
    const ratio = line.roadM / straight;
    console.log(`straight through the corners ${(straight / 1000).toFixed(2)} km, road ${(line.roadM / 1000).toFixed(2)} km, ratio ${ratio.toFixed(2)}`);
    check(ratio >= 0.99 && ratio < (place.grid ? 1.6 : 3), `${place.name}: road length is a plausible detour`, `x${ratio.toFixed(2)}`);

    // a hairpin: the route turns back on itself between two stretches of at least 10 m (T-1512)
    const turns = await page.evaluate((line) => {
      const k = Math.cos((line.route[1] * Math.PI) / 180), P = (i) => [line.route[2 * i] * 111320 * k, line.route[2 * i + 1] * 110574];
      const out = [];
      for (let i = 1; i + 1 < line.kinds.length; i++) {
        const a = P(i - 1), b = P(i), c = P(i + 1), u = [b[0] - a[0], b[1] - a[1]], v = [c[0] - b[0], c[1] - b[1]];
        const lu = Math.hypot(...u), lv = Math.hypot(...v);
        if (lu < 10 || lv < 10) continue;
        const cos = (u[0] * v[0] + u[1] * v[1]) / (lu * lv);
        if (cos < -0.94) out.push([i, Math.round(Math.acos(cos) * 180 / Math.PI)]);
      }
      return out;
    }, line);
    check(turns.length === 0, `${place.name}: no hairpin turns`, JSON.stringify(turns));

    // the independent check: sample every ~20 m along the road legs, look for a drawn road within 7 px
    await page.evaluate(() => { window.tw.selection.value = null; });
    const legsAt = []; { let from = 0; for (let i = 1; i < n; i++) if (line.kinds[i] & 1) { legsAt.push([from, i]); from = i; } }
    const onRoad = { total: 0, hit: 0, miss: [] };
    for (const [from, to] of legsAt) {
      if (!(line.kinds[from + 1] & 2)) continue;
      const lng = line.route.filter((_, k) => k % 2 === 0).slice(from, to + 1), lat = line.route.filter((_, k) => k % 2 === 1).slice(from, to + 1);
      await page.evaluate(([lo, hi]) => window.tw.map.fitBounds([lo, hi], { padding: 60, maxZoom: 15.5, duration: 0 }), [[Math.min(...lng), Math.min(...lat)], [Math.max(...lng), Math.max(...lat)]]);
      await settle();
      const r = await page.evaluate(([line, from, to]) => {
        const map = window.tw.map, R = 7, layers = map.getStyle().layers.filter((l) => l['source-layer'] === 'transportation' && l.type === 'line').map((l) => l.id);
        let total = 0, hit = 0; const miss = [];
        for (let i = from + 1; i <= to; i++) {
          const a = [line.route[2 * i - 2], line.route[2 * i - 1]], b = [line.route[2 * i], line.route[2 * i + 1]];
          const pa = map.project(a), pb = map.project(b), len = Math.hypot(pa.x - pb.x, pa.y - pb.y);
          const steps = Math.max(1, Math.round(len / 10));
          for (let s = 0; s <= steps; s++) {
            const x = pa.x + ((pb.x - pa.x) * s) / steps, y = pa.y + ((pb.y - pa.y) * s) / steps;
            if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) continue;
            total++;
            if (map.queryRenderedFeatures([[x - R, y - R], [x + R, y + R]], { layers }).length) hit++;
            else if (miss.length < 5) miss.push([Math.round(x), Math.round(y), +map.getZoom().toFixed(1)]);
          }
        }
        return { total, hit, miss };
      }, [line, from, to]);
      onRoad.total += r.total; onRoad.hit += r.hit; onRoad.miss.push(...r.miss);
    }
    console.log(`on-road samples ${onRoad.hit}/${onRoad.total}`, onRoad.miss.length ? JSON.stringify(onRoad.miss) : '');
    check(onRoad.total > 20 && onRoad.hit / onRoad.total >= 0.97, `${place.name}: the route lies on drawn roads`, `${onRoad.hit}/${onRoad.total}`);

    // selected look at a zoom where it all shows
    await page.evaluate(([id, cl]) => { const lo = [Math.min(...cl.map((p) => p[0])), Math.min(...cl.map((p) => p[1]))], hi = [Math.max(...cl.map((p) => p[0])), Math.max(...cl.map((p) => p[1]))]; window.tw.map.fitBounds([lo, hi], { padding: 140, maxZoom: 16, duration: 0 }); window.tw.selection.value = { kind: 'line', line: String(id) }; }, [line.id, place.clicks]);
    await settle();
    await page.screenshot({ path: join(OUT, `${place.name}-selected.png`) });
    await page.evaluate(() => { window.tw.shade.value = 'dark'; });
    await page.waitForTimeout(900);
    await page.screenshot({ path: join(OUT, `${place.name}-selected-dark.png`) });
    await page.evaluate(() => { window.tw.shade.value = 'light'; });
    await page.waitForTimeout(500);

    // redraw: the tool opens with the corners, and finishing unchanged keeps the same route
    const same = await page.evaluate(async (line) => {
      const tw = window.tw, b = tw.tools.bus;
      tw.tools.exit?.();
      const { editBusRoute } = await import('/src/game/lineActions.ts');
      const l = tw.world.value.save.lines.find((x) => x.id === line.id);
      editBusRoute(l);
      const pts = b.pts.length, legs = b.legs.length;
      await b.finish();
      await new Promise((r) => setTimeout(r, 1200));
      const now = tw.world.value.lineStats[line.id].bus;
      return { pts, legs, same: now.routeLL.length === line.route.length && now.routeLL.every((v, i) => Math.abs(v - line.route[i]) < 1e-6), kinds: now.kinds.join() === line.kinds.join() };
    }, line);
    check(same.pts === place.clicks.length && same.legs === place.clicks.length - 1, `${place.name}: Edit route opens with the corners and their legs`, JSON.stringify(same));
    check(same.same && same.kinds, `${place.name}: redrawing unchanged leaves the route as it was`);
    await page.evaluate(() => window.tw.tools.exit?.());
  }

  // divided roads (T-1512): where the basemap draws a road's two carriageways as separate lines, a leg from
  // one carriageway to the other must cross the gap, not run to the next junction and back. Pairs of
  // parallel roads 8-35 m apart are found in the loaded graph, a point put on each (a fifth in along one,
  // four fifths along the other) and the leg between them asked for.
  for (const [name, c] of [['dongguan', [113.6200, 22.9190]], ['berlin', [13.4380, 52.5171]], ['shenzhen', [114.0500, 22.5420]]]) {
    console.log(`--- divided roads, ${name}`);
    await page.evaluate(([c]) => window.tw.map.jumpTo({ center: c, zoom: 14 }), [c]);
    await page.evaluate(async ([c]) => {
      const roads = window.tw.tools.bus.roads;
      for (let i = 0; i < 100 && !roads.look({ lng: c[0], lat: c[1] }); i++) await new Promise((r) => setTimeout(r, 200));
      // a ring of points around, so the graph holds the neighbouring tiles too
      for (const [dx, dy] of [[0.012, 0], [-0.012, 0], [0, 0.01], [0, -0.01]]) for (let i = 0; i < 100 && !roads.look({ lng: c[0] + dx, lat: c[1] + dy }); i++) await new Promise((r) => setTimeout(r, 200));
    }, [c]);
    const r = await page.evaluate(async ([c]) => {
      const roads = window.tw.tools.bus.roads, g = roads.graph, W = 2 ** 26;
      const lngOf = (x) => (x / W) * 360 - 180, latOf = (y) => (360 / Math.PI) * Math.atan(Math.exp(Math.PI * (1 - (2 * y) / W))) - 90;
      const gy0 = (0.5 - Math.log((1 + Math.sin((c[1] * Math.PI) / 180)) / (1 - Math.sin((c[1] * Math.PI) / 180))) / (4 * Math.PI)) * W;
      const mpu = (40075016.686 / W) * Math.cos((c[1] * Math.PI) / 180);
      const gx0 = ((c[0] + 180) / 360) * W;
      const near = [];
      for (let e = 0; e < g.edges; e++) {
        const a = g.ea[e], b = g.eb[e], mx = (g.gx[a] + g.gx[b]) / 2, my = (g.gy[a] + g.gy[b]) / 2;
        if (Math.abs(mx - gx0) * mpu < 1500 && Math.abs(my - gy0) * mpu < 1500 && Math.hypot(g.gx[b] - g.gx[a], g.gy[b] - g.gy[a]) * mpu >= 150) near.push(e);
      }
      const pairs = [];
      for (let i = 0; i < near.length && pairs.length < 40; i++) {
        const e = near[i], a = g.ea[e], b = g.eb[e], dx = g.gx[b] - g.gx[a], dy = g.gy[b] - g.gy[a], dl = Math.hypot(dx, dy);
        const ux = dx / dl, uy = dy / dl, mx = g.gx[a] + dx / 2, my = g.gy[a] + dy / 2;
        for (let j = i + 1; j < near.length; j++) {
          const f = near[j], p = g.ea[f], q = g.eb[f], ex = g.gx[q] - g.gx[p], ey = g.gy[q] - g.gy[p], el = Math.hypot(ex, ey);
          if (Math.abs((ex * ux + ey * uy) / el) < 0.97) continue;
          const cx = g.gx[p] + ex / 2 - mx, cy = g.gy[p] + ey / 2 - my;
          const along = Math.abs(cx * ux + cy * uy) * mpu, side = Math.abs(-cx * uy + cy * ux) * mpu;
          // the next road across from e's middle, not one beyond another
          const here = { edge: e, t: 0.5, x: mx, y: my, d: 0 };
          if (side >= 8 && side <= 35 && along < 40 && g.across(here).some((c) => c.edge === f)) { pairs.push([e, f]); break; }
        }
      }
      const at = (e, t) => { const a = g.ea[e], b = g.eb[e]; return { lng: lngOf(g.gx[a] + t * (g.gx[b] - g.gx[a])), lat: latOf(g.gy[a] + t * (g.gy[b] - g.gy[a])) }; };
      const turns = (pts) => {
        const k = Math.cos((pts[1] * Math.PI) / 180), P = (i) => [pts[2 * i] * 111320 * k, pts[2 * i + 1] * 110574];
        let n = 0;
        for (let i = 1; i + 1 < pts.length / 2; i++) {
          const a = P(i - 1), b = P(i), d = P(i + 1), u = [b[0] - a[0], b[1] - a[1]], v = [d[0] - b[0], d[1] - b[1]];
          const lu = Math.hypot(...u), lv = Math.hypot(...v);
          if (lu < 10 || lv < 10) continue;
          if ((u[0] * v[0] + u[1] * v[1]) / (lu * lv) < -0.94) n++;
        }
        return n;
      };
      let ok = 0, road = 0, hair = 0, worst = 0, tried = 0;
      for (const [e, f] of pairs) {
        // A in the middle of one carriageway, B 60 m along the other from where A lies across
        const a = g.ea[e], b = g.eb[e], p = g.ea[f], q = g.eb[f], ex = g.gx[q] - g.gx[p], ey = g.gy[q] - g.gy[p], el2 = ex * ex + ey * ey;
        const ax = (g.gx[a] + g.gx[b]) / 2, ay = (g.gy[a] + g.gy[b]) / 2;
        const tp = ((ax - g.gx[p]) * ex + (ay - g.gy[p]) * ey) / el2, step = 60 / mpu / Math.sqrt(el2);
        const tb = tp + step <= 0.97 ? tp + step : tp - step;
        if (tb < 0.03 || tb > 0.97) continue;
        const A = at(e, 0.5), B = at(f, tb);
        let leg = roads.leg(A, B);
        for (let i = 0; i < 50 && !leg.ready; i++) { await new Promise((r) => setTimeout(r, 100)); leg = roads.leg(A, B); }
        const k = Math.cos((A.lat * Math.PI) / 180), straight = Math.hypot((A.lng - B.lng) * 111320 * k, (A.lat - B.lat) * 110574);
        tried++;
        const ratio = leg.m / Math.max(straight, 1), h = turns(leg.pts);
        if (leg.road) road++;
        if (h) hair++;
        worst = Math.max(worst, ratio);
        if (leg.road && ratio < 1.6 && !h) ok++;
      }
      return { pairs: pairs.length, tried, ok, road, hair, worst };
    }, [c]);
    console.log(JSON.stringify(r));
    check(r.pairs >= 5, `${name}: divided roads found`, `${r.pairs} pairs`);
    check(r.hair === 0, `${name}: no hairpin between carriageways`, `${r.hair} of ${r.tried}`);
    check(r.pairs > 0 && r.tried > 0 && r.ok / r.tried >= 0.9, `${name}: a leg across carriageways follows the road and crosses the gap`, `${r.ok}/${r.tried} road legs under 1.6 times the straight line, worst ${r.worst.toFixed(2)}`);
  }

  // a click far from any road stays straight, counted at 1.2: the sea off Manhattan's Battery
  console.log('--- off the roads');
  const a = [-74.045, 40.690], b = [-74.030, 40.682];
  await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [[-74.04, 40.686], 14.5]);
  await settle();
  const far = await page.evaluate(async ([a, b]) => {
    const tw = window.tw, bus = tw.tools.bus;
    tw.tools.exit?.();
    bus.begin(null);
    const click = (p) => { bus.add({ lng: p[0], lat: p[1] }); };
    click(a); click(b);
    await new Promise((r) => setTimeout(r, 1500));
    const built = bus.build(false);
    return { kinds: built.kinds, n: built.kinds.length };
  }, [a, b]);
  console.log(JSON.stringify(far));
  check(far.n >= 2 && !far.kinds.some((k) => k & 2), 'off the roads: the leg stays straight');
  await page.evaluate(() => window.tw.tools.exit?.());

  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} finally {
  await chrome.close();
}
process.exit(fail ? 1 : 0);
