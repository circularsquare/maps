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
