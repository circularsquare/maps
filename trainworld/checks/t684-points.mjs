// T-684: points count a rider's whole rail journey once, changes included, plus the passenger-km
// the line loads carry; and they land in the ledger as the hour's table says. In a ?new=1 game,
// paused, two Mumbai lines meeting at Dadar are built (the Western line test of t586, and a line
// east to Kurla from a station 190 m from Dadar's), so riders change between them with a walk. On
// the weekday's final solve:
//   - riders change lines (boardings exceed rail trips);
//   - the trips points count are the solve's rail trips (journeys), and those equal boardings less
//     the changes, so a two-line journey scores one trip;
//   - the rest of the points a day is the lines' loads times their stretches' lengths;
// then the clock runs a day and more: a whole game hour's points in the ledger match the hour's
// table, and Mumbai's points (T-694) are the sum of the past 24 whole hours' tables.
// Writes artifacts/trainworld/t684-points.json.
//   node trainworld/checks/t684-points.mjs [url]   (from the maps root; default localhost:6153)
// Prints PASS or FAIL lines and exits 1 on failure. A few minutes.
import { writeFile } from 'node:fs/promises';
import { launchChrome, demandFinal, ANGLE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6153/').replace(/\/?$/, '/');
const OUT = 'artifacts/trainworld/t684-points.json';
const CENTRE = [72.8428, 19.0192];
const WEST = [['Churchgate', 72.8272, 18.9355], ['Marine Lines', 72.8237, 18.9458], ['Charni Road', 72.8182, 18.9518],
  ['Grant Road', 72.8158, 18.9634], ['Mumbai Central', 72.8193, 18.9696], ['Mahalaxmi', 72.8242, 18.9825],
  ['Lower Parel', 72.8303, 18.9957], ['Prabhadevi', 72.8359, 19.0075], ['Dadar', 72.8428, 19.0192],
  ['Matunga Road', 72.8467, 19.0278], ['Mahim', 72.8466, 19.0407], ['Bandra', 72.8402, 19.0550]];
const EAST = [['Dadar East', 72.8445, 19.0186], ['Matunga', 72.8553, 19.0274],['Sion', 72.8636, 19.0466], ['Kurla', 72.8794, 19.0656]];

const { ctx, close } = await launchChrome({ prefix: 'tw-t684-', args: ANGLE });
const results = [];
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const near = (a, b, tol) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));

try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 240000 });
  await page.evaluate(async () => (await import('/src/game/ui.ts')).setDisplay('dayShown', 'weekday'));
  await page.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 11.2 }), CENTRE);
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'mum', undefined, { timeout: 300000 });
  const before = await page.evaluate(() => window.twDemand.view()?.version ?? -1);
  const built = await page.evaluate(async ({ west, east }) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    // a route at `level` through `pts` with a tail before the first and after the last; its nodes
    const lay = async (pts, level) => {
      const nodes = [], start = tail(pts[0], pts[1]);
      const path = [...pts, tail(pts.at(-1), pts.at(-2))];
      let from = { kind: 'free', x: start.x, y: start.y, level };
      for (const p of path) {
        await run({ op: 'route', from, to: { kind: 'free', x: p.x, y: p.y, level }, pis: [], single: false });
        const e = t.world.value.edges.at(-1);
        if (!nodes.length) nodes.push(e.a);
        nodes.push(e.b);
        from = { kind: 'node', node: e.b };
      }
      return nodes;
    };
    const wp = west.map((s) => t.lngLatToPack(s[1], s[2]));
    const wn = await lay(wp, -2);
    const wAt = wn.slice(1, -1);
    for (let i = 0; i < west.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: wAt[i] }, platform: 160, name: west[i][0] });
    // its own track from a station 190 m east of Dadar: a change is a walk between the two
    const ep = east.map((s) => t.lngLatToPack(s[1], s[2]));
    const en = await lay(ep, -3);
    const eAt = en.slice(1, -1);
    for (let i = 0; i < east.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: eAt[i] }, platform: 160, name: east[i][0] });
    const a = (await run({ op: 'addLine', stops: wAt, name: 'Western test', colour: '#e4002b' })).line;
    const b = (await run({ op: 'addLine', stops: eAt, name: 'Kurla test', colour: '#0060a8' })).line;
    await run({ op: 'setTrainCount', line: a, trains: [10, 10, 5] });
    await run({ op: 'setTrainCount', line: b, trains: [8, 8, 4] });
    await run({ op: 'constructAll' });
    return { lines: [a, b] };
  }, { west: WEST, east: EAST });
  await demandFinal(page, { strict: false, after: before, timeout: 900000 });

  // the weekday's figures: boardings, the solve's rail trips, the changes, the points a day
  const day = await page.evaluate(async () => {
    const P = await import('/src/game/points.ts'), D = await import('/src/game/demand.ts');
    const w = window.tw.world.value, d = D.demandByDay.value[0], dv = d.view;
    let boardings = 0, pkm = 0;
    for (const l of w.save.lines) {
      const dm = dv.lines.get(l.id), st = w.lineStats[l.id], n = l.stops.length;
      if (!dm || st?.status !== 'running') continue;
      boardings += dm.ridersPerDay;
      const km = [];
      for (const s of st.stopS) for (let k = 0; k + 1 < n; k++) km.push(Math.abs(s[k + 1] - s[k]) / 1000);
      for (const seg of dm.loads) seg.forEach((v, k) => { pkm += v * km[k]; });
    }
    const trips = [...P.railTripsByRegion(d).values()].reduce((a, b) => a + b, 0);
    const x = await D.complexTransfers([...dv.stations.keys()]);
    const changes = (x?.pairs ?? []).reduce((a, p) => a + p.riders, 0);
    let points = 0;
    for (const r of P.pointsPerDay.value.values()) points += r[0];
    return { boardings, railTrips: dv.railTripsPerDay, trips, changes, pkm, points, perTrip: P.POINTS_PER_TRIP, perKm: P.POINTS_PER_KM };
  });
  const tripPoints = day.points - day.perKm * day.pkm;
  report('riders change lines at Dadar', day.changes > 0 && day.boardings > day.railTrips * 1.01, `(${Math.round(day.boardings)} boardings, ${Math.round(day.railTrips)} rail trips, ${Math.round(day.changes)} changes)`);
  report('the regions\' trips are the network\'s rail trips', near(day.trips, day.railTrips, 0.001), `(${Math.round(day.trips)} of ${Math.round(day.railTrips)})`);
  report('a journey with a change is one trip: rail trips are boardings less changes', near(day.railTrips, day.boardings - day.changes, 0.02), `(${Math.round(day.railTrips)} against ${Math.round(day.boardings - day.changes)})`);
  report('points a day are 20 a trip and 1 a passenger-km of the line loads', near(tripPoints / day.perTrip, day.railTrips, 0.005), `(${Math.round(day.points)} points: ${Math.round(tripPoints / day.perTrip)} trips, ${Math.round(day.pkm)} passenger-km)`);

  // the clock runs a game day and more; each whole hour's points in the ledger are the hour's table,
  // and Mumbai's points (T-694) are the past 24 whole hours'
  const t0 = await page.evaluate(() => { const c = window.tw.clock; c.setSpeed(2); c.setRunning(true); return c.now(); });
  await page.waitForFunction((t) => window.tw.clock.now() > t + 26 * 3600, t0, { timeout: 300000, polling: 500 });
  const hour = await page.evaluate(async (t0) => {
    const c = window.tw.clock;
    c.setRunning(false);
    await new Promise((r) => setTimeout(r, 500));
    const P = await import('/src/game/points.ts'), M = await import('/src/game/money.ts'), L = await import('/src/game/localTime.ts');
    const Pe = await import('/src/game/periods.ts'), Dy = await import('/src/game/days.ts');
    const m = M.moneyState.value, now = m.hours.at(-1).hour;
    const table = (h) => {
      const tl = h * 3600 + L.shiftAt(72.8428, 19.0192), k = Dy.dayKindAt(tl), p = Pe.periodAt(tl);
      let v = 0;
      for (const { regions } of P.linePoints.value) for (const [, per] of regions) v += (per[k] ?? per[0])[p];
      return v;
    };
    const h = now - 1, e = m.hours.find((x) => x.hour === h);
    let expect24 = 0;
    for (let x = now - 24; x < now; x++) expect24 += table(x);
    return { h, ledger: e?.points ?? null, expect: table(h), started: (now - 24) * 3600 >= t0, region: m.regions.mum?.hours.find((x) => x.hour === h)?.points ?? null, past24: P.pointsPast24h.value.get('mum') ?? 0, expect24 };
  }, t0);
  report('a whole hour\'s points in the ledger match the hour\'s table', hour.ledger !== null && near(hour.ledger, hour.expect, 1e-4) && hour.expect > 0, `(hour ${hour.h}: ${Math.round(hour.ledger ?? -1)} against ${Math.round(hour.expect)}; Mumbai's ${Math.round(hour.region ?? -1)})`);
  report('Mumbai\'s points are the past 24 whole hours\' tables', hour.started && near(hour.past24, hour.expect24, 1e-4) && hour.expect24 > 0, `(${Math.round(hour.past24)} against ${Math.round(hour.expect24)}; a day at the weekday's riders ${Math.round(day.points)})`);
  await writeFile(OUT, JSON.stringify({ built, day, hour }, null, 1));
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await close();
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
