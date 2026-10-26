// T-1464: bus lines feeding rail, in a new game and on the real save. A feeder is drawn to a station,
// constructed and solved: it carries riders, the rail line's riders rise, Construct charges US$1M a
// road-km plus US$0.6M a bus, demand gets the corridor (each stop that is no station with its
// stretch of route, the station without), and the save brings the route back. Running costs and
// upkeep are the sim's tests (sim/src/track/wasm/bus_tests.rs). About 6 minutes. `node trainworld/checks/t1464-bus.mjs [url]`
// from the maps root, on your own Vite (port 6164 by default).
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, buildLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6164/';
let failed = 0;
const say = (ok, what, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? `  (${detail})` : ''}`); };

/** Draw a feeder from `from` [lng, lat] to `to`, construct it; the edits' answers and the line's state. */
const feeder = (page, from, to) => page.evaluate(async ([from, to]) => {
  const tw = window.tw, c = tw.client;
  const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
  const cash0 = tw.world.value.save.cash;
  const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
  const con = await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
  await new Promise((d) => setTimeout(d, 1000));
  const st = tw.world.value.lineStats[String(add.line)];
  return { line: add.line, ok: add.ok && con.ok, charge: con.charge, spent: (cash0 - tw.world.value.save.cash) / 1e6, km: st?.km, buses: Math.max(...(st?.trainsNeeded ?? [0])), bus: st?.bus };
}, [from, to]);

/** The demand view's riders a day on `lines` and the corridor demand was given for them. */
const figures = (page, lines) => page.evaluate((lines) => {
  const v = window.twDemand.view(), s = window.twDemand.client.cur?.svc;
  return {
    version: v.version, rail: v.railTripsPerDay,
    riders: Object.fromEntries(lines.map((l) => [l, v.lines.get(String(l))?.ridersPerDay ?? 0])),
    corridor: Object.fromEntries(lines.map((l) => {
      const sl = s?.lines.find((x) => x.id === String(l));
      return [l, sl ? { stops: sl.stops.length, withStretch: sl.catches?.filter((c) => c.length >= 4).length ?? 0, cars: sl.cars } : null];
    })),
    // bus stops that are no station: ids from `bus::STOP_ID_BASE` (2^30) up
    busStations: s?.stations.filter((x) => Number(x.id) >= 2 ** 30).length ?? 0,
  };
}, lines);

const b = await launchChrome({ prefix: 'tw-t1464-' });
const errors = [];
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);

  // ---- a new game: a rail line across Queens, a 5 km feeder from the north to its middle station
  await page.goto(`${BASE}?new=1&debug=1&paused=1`);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value, null, { timeout: 120000 });
  const [lng, lat] = [-73.87, 40.75];
  const dLng = 2 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const r = await buildLine(page, [-1, 0, 1].map((k) => [lng + k * dLng, lat]), { trains: [6, 4, 3] });
  await demandFinal(page);
  const before = await figures(page, [r.line]);
  const f = await feeder(page, [lng, lat + 0.045], [lng, lat + 0.0005]);
  say(f.ok && f.bus?.built, 'new game: feeder made and constructed', `${f.km?.toFixed(2)} km of road, ${f.buses} buses`);
  const want = f.km * 1 + f.buses * 0.6;
  say(Math.abs(f.spent - want) < 0.01, 'new game: Construct charges $1M a road-km and $0.6M a bus', `spent ${f.spent.toFixed(3)}, want ${want.toFixed(3)}`);
  const ends = f.bus?.stops ?? [];
  say(ends.length >= 3 && ends.at(-1).kind === 1 && ends[0].kind === 0, 'new game: stops at the end and the station it reaches', ends.map((s) => s.kind).join(','));
  await demandFinal(page, { after: before.version });
  const after = await figures(page, [r.line, f.line]);
  say(after.riders[f.line] > 1000, 'new game: the feeder carries riders', `${Math.round(after.riders[f.line]).toLocaleString('en-US')} a day`);
  say(after.riders[r.line] > before.riders[r.line], 'new game: the rail line gains riders', `${Math.round(before.riders[r.line])} -> ${Math.round(after.riders[r.line])}`);
  const cor = after.corridor[f.line];
  say(!!cor && cor.withStretch === cor.stops - 1 && cor.cars === 0.5 && after.busStations === cor.stops - 1, 'new game: demand has the corridor', JSON.stringify(cor));
  // the save brings the route back
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const saved = await readFile(await (await dl).path());
  await page.evaluate(() => { window.tw.client.lastLoad = null; });
  await importSave(page, saved);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length === 2, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  const back = await page.evaluate((l) => window.tw.world.value.lineStats[String(l)]?.bus, f.line);
  say(!!back && back.built && back.stops.length === ends.length, 'new game: a saved bus line loads with its route and stops');

  // ---- the real save: two feeders to stations in Queens and Brooklyn
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { lines: 100 });
  const real0 = await figures(page, []);
  const picks = await page.evaluate(() => {
    const st = window.tw.world.value.save.stations.filter((s) => s.built);
    const near = (lng, lat) => st.reduce((b, s) => (Math.hypot(s.lng - lng, s.lat - lat) < Math.hypot(b.lng - lng, b.lat - lat) ? s : b));
    return [near(-73.80, 40.70), near(-73.95, 40.63)].map((s) => [s.lng, s.lat]);
  });
  const feeds = [];
  for (const [i, [a, c]] of picks.entries()) feeds.push(await feeder(page, [a + (i ? -0.03 : 0.03), c + 0.02], [a, c]));
  say(feeds.every((x) => x.ok && x.bus?.stops.some((s) => s.kind === 1)), 'real save: two feeders made, each stopping at its station');
  await demandFinal(page, { lines: 100, after: real0.version });
  const real1 = await figures(page, feeds.map((x) => x.line));
  say(feeds.every((x) => real1.riders[x.line] > 0), 'real save: the feeders carry riders', feeds.map((x) => Math.round(real1.riders[x.line])).join(', '));
  say(real1.rail > real0.rail, 'real save: rail trips a day rise', `${Math.round(real0.rail).toLocaleString('en-US')} -> ${Math.round(real1.rail).toLocaleString('en-US')}`);
  say(feeds.every((x) => real1.corridor[x.line]?.withStretch > 0), 'real save: demand has both corridors');
} catch (e) {
  say(false, 'ran to the end', e.message);
} finally {
  say(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
  await b.close();
}
console.log(failed ? `T-1464 FAILED: ${failed}` : 'T-1464 PASSED');
process.exit(failed ? 1 : 0);
