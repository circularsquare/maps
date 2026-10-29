// T-1750: load a real-network save into a new game, wait for the weekday's final demand, dump the line stats,
// the service the demand solved with, the capacity markers and each line's per-period loads / boardings / arrivals.
// env: CITY, OUT, PORT (6196), WORKERS (1)
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, GL } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const CITY = process.env.CITY, OUT = process.env.OUT;
const SAVE = `C:/Users/anita/projects/maps/trainworld/data/packs/real/${CITY}.save`;
const PORT = process.env.PORT ?? '6196', WORKERS = process.env.WORKERS ?? '1';
const log = (s) => console.log(`${new Date().toISOString().slice(11, 19)} ${s}`);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1750-' + CITY + '-', size: [1200, 800], args: [...GL] });
const errors = [];
try {
  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('pageerror', (e) => { errors.push('pageerror: ' + e.message); log('pageerror ' + e.message.slice(0, 200)); });
  await page.goto(`http://localhost:${PORT}/?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 300000 });
  log('new game up');
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 900000 });
  log('save in');
  const t = Date.now();
  await demandFinal(page, { timeout: 3 * 3600000, polling: 2000, lines: 3 });
  log(`demand final after ${((Date.now() - t) / 1000).toFixed(0)} s`);
  const dump = await page.evaluate(() => {
    const w = window.tw.world.value, days = window.twDemand.days();
    const v0 = days[0]?.view;
    const r1 = (a) => Array.from(a, (x) => Math.round(x * 10) / 10);
    const nameOf = new Map(w.save.stations.map((s) => [s.id, s.name]));
    const out = { stations: w.save.stations.map((s) => [s.id, s.name]), lines: [], capacity: w.capacity, fullTrack: w.fullTrack.map((m) => [...m.entries()]),
      view: { day: v0.day, round: v0.round, final: v0.final, tripsPerDay: v0.tripsPerDay, railTripsPerDay: v0.railTripsPerDay, split: v0.split, peakHourFactor: v0.peakHourFactor, trainShareByPeriod: v0.trainShareByPeriod },
      stationDemand: [...v0.stations.entries()].map(([id, s]) => [id, nameOf.get(id) ?? nameOf.get(Number(id)), s.boardings, s.alightings, s.transfers ?? 0]) };
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st) continue;
      const dm = v0.lines.get(l.id);
      out.lines.push({ id: l.id, num: l.num, name: l.name, stops: l.stops, stopNames: l.stops.map((s) => nameOf.get(s)), saveTph: l.tph, saveTrains: l.trains ?? null,
        stat: { status: st.status, km: st.km, fromStartByLevel: st.fromStartByLevel, roundTripByLevel: st.roundTripByLevel, trainsNeeded: st.trainsNeeded, autoS: st.autoS, delayS: st.delayS,
          cars: st.cars, carsSet: st.carsSet, fast: st.fast, stretches: st.stretches, stopS: st.stopS, dwellS: st.dwellS, turnaroundS: st.turnaroundS, passed: st.passed, reversals: st.reversals, endTurns: st.endTurns, loop: st.loop },
        dm: dm && { ridersPerDay: dm.ridersPerDay, fullestPct: dm.fullestPct, boardings: r1(dm.boardings), tph: dm.tph, cars: dm.cars,
          loads: dm.loads.map(r1), board: dm.board?.map(r1), alight: dm.alight?.map(r1), arrive: dm.arrive?.map(r1) } });
    }
    return out;
  });
  dump.errors = errors; dump.city = CITY;
  await writeFile(OUT, JSON.stringify(dump));
  log(`dumped ${dump.lines.length} lines -> ${OUT}`);
} catch (e) {
  log('FAILED ' + (e?.stack ?? e));
  process.exitCode = 1;
} finally {
  await close();
}
