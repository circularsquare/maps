// T-1466 sweep: draw buses of different length, headway and place, one at a time on the real save or a new game; record riders,
// loads and the legs by mode of the homes within 400 m (cellModes). About 3 minutes a bus on the real save. Analysis: notes/T-1466.md.
// node trainworld/checks/t1466-bus-sweep.mjs <real|new> <outfile.json> [url] [ids, comma separated substrings]
// from the maps root, on your own Vite (default port 9301).
import { readFile, writeFile } from 'node:fs/promises';
import { REAL_SAVE, buildLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';
const WORLD = process.argv[2] ?? 'real';
const OUT = process.argv[3] ?? 'out.json';
const BASE = process.argv[4] ?? 'http://localhost:9301/';
const ONLY = process.argv[5] ?? '';

const km = (lng, lat) => [111.32 * Math.cos((lat * Math.PI) / 180), 110.57];
const off = ([lng, lat], bearingDeg, d) => { const [kx, ky] = km(lng, lat); const b = (bearingDeg * Math.PI) / 180; return [lng + (d * Math.sin(b)) / kx, lat + (d * Math.cos(b)) / ky]; };

// station-ended feeder: `station` is a name hint resolved in the page by nearest built station to the given lng/lat
const feeder = (id, near, bearing, straightKm, hw = 480) => ({ id, kind: 'feeder', near, bearing, straightKm, hw });
const free = (id, a, b, hw = 480) => ({ id, kind: 'free', a, b, hw });

const REAL = [
  // Manhattan: an avenue (First/Second Ave direction, tilted 29 degrees), a crosstown
  free('man-ave-10k-8m', [-73.9885, 40.7215], [-73.938, 40.803], 480),
  free('man-ave-10k-4m', [-73.9885, 40.7215], [-73.938, 40.803], 240),
  free('man-ave-5k-8m', [-73.9885, 40.7215], [-73.9625, 40.7625], 480),
  free('man-xtown-3k-8m', [-74.001, 40.760], [-73.9705, 40.7505], 480),
  // Brooklyn feeder into the Flatbush/Midwood station
  feeder('bk-fdr-2k-8m', [-73.95, 40.63], 225, 2, 480),
  feeder('bk-fdr-3k-8m', [-73.95, 40.63], 225, 3, 480),
  feeder('bk-fdr-5k-8m', [-73.95, 40.63], 225, 5, 480),
  feeder('bk-fdr-3k-4m', [-73.95, 40.63], 225, 3, 240),
  feeder('bk-fdr-3k-15m', [-73.95, 40.63], 225, 3, 900),
  feeder('bk-fdr-3k-30m', [-73.95, 40.63], 225, 3, 1800),
  // Queens feeder (as t1464-bus): NE of Jamaica
  feeder('qn-fdr-3k-8m', [-73.80, 40.70], 50, 3, 480),
  feeder('qn-fdr-5k-8m', [-73.80, 40.70], 50, 5, 480),
  // suburb: Mineola and Babylon on Long Island
  feeder('li-mineola-3k-8m', [-73.6405, 40.7404], 0, 3, 480),
  feeder('li-mineola-6k-8m', [-73.6405, 40.7404], 0, 6, 480),
  feeder('li-mineola-6k-15m', [-73.6405, 40.7404], 0, 6, 900),
  feeder('li-babylon-4k-8m', [-73.3247, 40.7006], 0, 4, 480),
  free('li-xtown-6k-8m', [-73.585, 40.765], [-73.515, 40.765], 480),
];
const NEW = [
  feeder('nw-fdr-2k-8m', [-73.87, 40.75], 0, 2, 480),
  feeder('nw-fdr-4k-8m', [-73.87, 40.75], 0, 4, 480),
  feeder('nw-fdr-5k-8m', [-73.87, 40.75], 0, 5, 480),
  feeder('nw-fdr-5k-4m', [-73.87, 40.75], 0, 5, 240),
  feeder('nw-fdr-5k-15m', [-73.87, 40.75], 0, 5, 900),
  feeder('nw-fdr-8k-8m', [-73.87, 40.75], 0, 8, 480),
];
const SCEN = (WORLD === 'real' ? REAL : NEW).filter((s) => ONLY.split(',').some((o) => s.id.includes(o)));

/** everything the page can tell about the current final view */
const snapshot = (page, line) => page.evaluate((line) => {
  const v = window.twDemand.view(), d = v.lines.get(String(line));
  const out = { version: v.version, split: v.split, trips: v.tripsPerDay, rail: v.railTripsPerDay, peakHourFactor: v.peakHourFactor };
  if (d) {
    const lp = window.tw.world.value.lineStats[String(line)];
    const stops = lp?.bus?.stops ?? [];
    const n = d.boardings.length;
    const P = d.board.length;
    const per = (arr) => Array.from({ length: n }, (_, k) => arr.reduce((s, a) => s + a[k] + a[2 * n - 1 - k], 0));
    out.line = {
      riders: d.ridersPerDay, fullest: d.fullestPct, n, stops: stops.map((s) => ({ kind: s.kind, st: s.station ?? null })),
      boardStop: per(d.board), alightStop: per(d.alight), arriveStop: per(d.arrive ?? d.board),
      boardRun: [0, 1].map((r) => d.board.reduce((s, a) => s + a.slice(r * n, r * n + n).reduce((x, y) => x + y, 0), 0)),
      maxLoadPerPeriod: d.loads.map((a) => Math.max(...a)),
      loads: d.loads.map((a) => Array.from(a)),
      km: lp?.km, trainsNeeded: lp?.trainsNeeded, headway: lp?.headway ?? null,
    };
    out.stationTransfers = stops.map((s) => (s.station != null ? v.stations.get(String(s.station))?.transfers ?? null : null));
  }
  return out;
}, line);

/** home and activity legs by mode of the cells within 400 m of the route, now and in the stored baseline */
const corridor = (page, route, store) => page.evaluate(async ({ route, store }) => {
  const c = window.twDemand.client;
  const cc = await c.cityCells(), cm = await c.cellModes(true);
  if (!cc || !cm) return null;
  if (store) window.__cm0 = cm;
  const base = window.__cm0;
  const keys = ['homeRail', 'homeWalk', 'homeDrive', 'workRail', 'workWalk', 'workDrive'];
  const pts = []; for (let i = 0; i < route.length; i += 2) pts.push([route[i], route[i + 1]]);
  const dist = (x, y) => { let d = 1e12; for (let s = 0; s + 1 < pts.length; s++) { const [ax, ay] = pts[s], [bx, by] = pts[s + 1]; const dx = bx - ax, dy = by - ay; const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy || 1))); d = Math.min(d, Math.hypot(x - ax - t * dx, y - ay - t * dy)); } return d; };
  let nm = 0, nearest = 1e12; const mask = new Uint8Array(cc.n);
  for (let k = 0; k < cc.n; k++) { const d = dist(cc.xy[2 * k], cc.xy[2 * k + 1]); nearest = Math.min(nearest, d); if (d <= 400) { mask[k] = 1; nm++; } }
  const sum = (m) => Object.fromEntries(keys.map((key) => { let s = 0; const a = m[key]; for (let k = 0; k < cc.n; k++) if (mask[k]) s += a[k]; return [key, s]; }));
  return { cells: cc.n, nm, nearest, now: sum(cm), base: base ? sum(base) : null, ver: cm.version };
}, { route, store });

const finalWith =(page, line, afterVersion, tmo = 420000) => page.waitForFunction(({ line, afterVersion }) => {
  const d = window.twDemand, s = d?.log?.at(-1), v = d?.view?.();
  if (!v || !s || v.version !== s.version || v.version === 0 || !v.final || d.client.waiting || v.version <= afterVersion) return false;
  if (line === null) return true;
  const l = v.lines.get(String(line));
  return !!l && !l.estimate;
}, { line, afterVersion }, { timeout: tmo, polling: 500 }).catch(async (e) => {
  console.log('diag', JSON.stringify(await page.evaluate((line) => { const d = window.twDemand, v = d.view(); return { ver: v.version, log: d.log.slice(-3).map((x) => x.version), keys: [...v.lines.keys()], final: v.final, waiting: d.client.waiting, has: v.lines.has(String(line)), est: v.lines.get(String(line))?.estimate, lines: v.lines.size, ls: window.tw.world.value.lineStats[String(line)]?.trainsNeeded }; }, line)).slice(0, 800));
  throw e;
});

const b = await launchChrome({ prefix: 'tw-t1466-' });
const results = { world: WORLD, when: new Date().toISOString(), scen: [] };
const errors = [];
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value && window.twDemand?.client, null, { timeout: 180000 });
  let railLine = null;
  if (WORLD === 'real') {
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    await demandFinal(page, { lines: 100 });
  } else {
    const [lng, lat] = [-73.87, 40.75];
    const dLng = 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
    const r = await buildLine(page, [-1, 0, 1].map((k) => [lng + k * dLng, lat]), { trains: [6, 4, 3] });
    railLine = r.line;
    await demandFinal(page);
  }
  await page.waitForFunction(() => window.twDemand.view().tripsPerDay > 0 && window.twDemand.view().final, undefined, { timeout: 300000 });
  const base = await snapshot(page, -1);
  results.base = base;
  await corridor(page, [0, 0, 1, 1], true);
  console.log('baseline', JSON.stringify({ split: base.split, trips: base.trips, rail: base.rail }));

  for (const sc of SCEN) {
    const t0 = Date.now();
    // endpoints
    let from, to;
    if (sc.kind === 'feeder') {
      const st = await page.evaluate(([lng, lat]) => {
        const all = window.tw.world.value.save.stations.filter((s) => s.built);
        const s = all.reduce((b, s) => (Math.hypot((s.lng - lng) * 0.76, s.lat - lat) < Math.hypot((b.lng - lng) * 0.76, b.lat - lat) ? s : b));
        return [s.lng, s.lat, s.name];
      }, sc.near);
      to = [st[0], st[1] + 0.0005];
      from = off(to, sc.bearing, sc.straightKm);
      sc.station = st[2];
    } else { from = sc.a; to = sc.b; }
    const prev = await page.evaluate(() => window.twDemand.view().version);
    const r = await page.evaluate(async ([from, to, hw]) => {
      const tw = window.tw, c = tw.client;
      const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
      const cash0 = tw.world.value.save.cash;
      const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
      const con = await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
      const hwr = await c.edit({ op: 'setHeadway', line: add.line, seconds: [hw, Math.round(hw * 1.25), Math.round(hw * 2.5)] });
      await new Promise((d) => setTimeout(d, 1500));
      const st = tw.world.value.lineStats[String(add.line)];
      return { route, line: add.line, ok: add.ok && con.ok && hwr?.ok !== false, hwr: JSON.stringify(hwr).slice(0, 80), spent: (cash0 - tw.world.value.save.cash) / 1e6, km: st?.km, trainsNeeded: st?.trainsNeeded };
    }, [from, to, sc.hw]);
    await finalWith(page, r.line, prev);
    const snap = await snapshot(page, r.line);
    snap.corridor = await corridor(page, r.route, false);
    console.log(sc.id, 'corridor', JSON.stringify(snap.corridor));
    const straight = Math.hypot((from[0] - to[0]) * km(...from)[0], (from[1] - to[1]) * km(...from)[1]);
    results.scen.push({ ...sc, from, to, straightKm: +straight.toFixed(2), add: r, snap, secs: (Date.now() - t0) / 1000 });
    console.log(sc.id, 'riders', Math.round(snap.line?.riders ?? 0), 'road km', r.km?.toFixed?.(2), 'buses', JSON.stringify(r.trainsNeeded), 'fullest', snap.line?.fullest, 'ok', r.ok, r.hwr);
    await writeFile(OUT, JSON.stringify(results));
    // remove and wait for the baseline back
    const v1 = await page.evaluate(() => window.twDemand.view().version);
    await page.evaluate((l) => window.tw.client.edit({ op: 'removeLine', line: l }), r.line);
    await page.waitForTimeout(2500);
    await finalWith(page, null, v1);
  }
  const base2 = await snapshot(page, -1);
  results.base2 = base2;
  console.log('baseline after', JSON.stringify({ split: base2.split, trips: base2.trips, rail: base2.rail }));
} catch (e) {
  console.log('FAILED', e.message);
  results.error = e.message;
} finally {
  results.errors = errors.slice(0, 5);
  await writeFile(OUT, JSON.stringify(results));
  await b.close();
}
