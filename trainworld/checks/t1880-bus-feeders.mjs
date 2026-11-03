// T-1880: do buses act as feeders? On the real save, one bus at a time: its own profit a day (its fares
// minus running cost and upkeep, as the line panel counts them) against the change in the whole network's
// profit, the rail lines' riders and fares, and rail trips a day. A feeder "works" when its own profit is
// below 0 and the network's rises. About 5 minutes a bus. Analysis: notes/T-1880.md.
//   node trainworld/checks/t1880-bus-feeders.mjs [url] [outfile.json] [ids, comma separated substrings]
//   e.g. `... http://localhost:6164/ out.json throgs@720,freshmeadows` (Throgs Neck at 12 minutes, Fresh Meadows as drawn)
// from the maps root, on your own Vite (port 6164 by default). Money is game dollars (the economy's 100x).
import { readFile, writeFile } from 'node:fs/promises';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6164/';
const OUT = process.argv[3] ?? `${ARTIFACTS}t1880/feeders.json`;
const ONLY = process.argv[4] ?? '';

// `pts` [lng, lat] the clicked points, the bus straight between them; `to` the hint of a station the
// route ends at (the nearest built one, the end put 55 m north of it as T-1466's sweep did); `hw` a
// headway in seconds at high demand (medium 1.25x, low 2.5x), or the game's default for a new bus.
const S = (id, what, pts, to = null, hw = null) => ({ id, what, pts, to, hw });
const ALL = [
  // a dense place with no station to a busy one: Red Hook by Columbia St to Jay St-MetroTech
  S('redhook', 'Red Hook to Jay St-MetroTech', [[-74.0115, 40.6760], [-73.9990, 40.6790]], [-73.9871, 40.6923]),
  // Fresh Meadows (no station within 2.5 km) to the F's terminal
  S('freshmeadows', 'Fresh Meadows to Jamaica-179 St', [[-73.7900, 40.7400]], [-73.7838, 40.7127]),
  // crosstowns linking lines: Fordham Rd (4, B/D, Metro-North, 5) and Church Av (F/G, B/Q, 2/5, East Flatbush)
  S('fordham', 'Fordham Rd crosstown, 4 to the 5', [[-73.9010, 40.8628], [-73.8978, 40.8613], [-73.8904, 40.8616], [-73.8720, 40.8590]], [-73.8554, 40.8590]),
  S('church', 'Church Av crosstown, F/G to East Flatbush', [[-73.9797, 40.6443], [-73.9623, 40.6505], [-73.9496, 40.6508], [-73.9200, 40.6520]]),
  // outer neighbourhoods to a terminal
  S('throgs', 'Throgs Neck to Pelham Bay Park', [[-73.8195, 40.8175], [-73.8230, 40.8350]], [-73.8281, 40.8525]),
  S('laurelton', 'Springfield Gardens to Jamaica-179 St', [[-73.7450, 40.6750], [-73.7700, 40.6950]], [-73.7838, 40.7127]),
];
// `ids` picks scenarios by substring; `id@seconds` runs that one at another headway
const PICK = ONLY ? ONLY.split(',').flatMap((o) => {
  const [id, hw] = o.split('@');
  return ALL.filter((s) => s.id.includes(id)).map((s) => (hw ? { ...s, id: `${s.id}-${Math.round(hw / 60)}m`, hw: +hw } : s));
}) : ALL;

const H = [4, 6, 4, 4, 6], LEVEL = [0, 0, 0, 1, 2]; // game/periods.ts: hours and demand level of each period

/** a weekday's money by line, as the Lines tab counts it (fares from `incomeByDay`, running cost as
 * `runningToday`, upkeep from the line's stats), and the weekday's demand figures. A loaded save shows
 * the average day, which has no income of its own, so the weekday is read from `twDemand.days()`. */
const DAY = 0;
const money = (page) => page.evaluate(([H, LEVEL, di]) => {
  const tw = window.tw, w = tw.world.value, m = tw.moneyState.value, v = window.twDemand.days()[di].view;
  const inc = tw.incomeByDay.value[di];
  // T-2104: the base fare's share as the game takes it, and the even share per boarding before it
  const baseNow = tw.baseFares?.(v) ?? {}, baseEven = tw.baseFares?.(v, 1) ?? {};
  const quiet = di === 1 || di === 2;
  const lines = {};
  let boardings = 0;
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id];
    if (!st || st.status !== 'running') continue;
    const ld = v.lines.get(l.id);
    const tph = ld?.tph ?? [l.tph.high, l.tph.medium, l.tph.low];
    const cars = ld?.cars ?? st.cars;
    const perKm = st.bus ? m.runCostBusKm : st.fast ? m.hsRunCostTrainKm + cars * m.hsRunCostCarKm : m.runCostTrainKm + cars * m.runCostCarKm;
    let run = 0, busKm = 0;
    H.forEach((h, p) => { const lev = LEVEL[p] === 0 && quiet ? 1 : LEVEL[p]; run += tph[lev] * 2 * st.km * perKm * m.economy * h; busKm += tph[lev] * 2 * st.km * h; });
    const lm = inc?.lines.get(l.id);
    const fares = lm ? lm.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 : 0;
    const riders = ld?.ridersPerDay ?? 0;
    boardings += riders;
    const faresEven = fares - (baseNow[l.id] ?? 0) + (baseEven[l.id] ?? 0);
    lines[l.id] = { name: l.name, bus: !!st.bus, km: st.km, riders, fares, faresEven, run, upkeep: st.upkeep, vehKm: busKm, tph, estimate: !!ld?.estimate };
  }
  return { di, version: v.version, final: v.final, rail: v.railTripsPerDay, trips: v.tripsPerDay, split: v.split, boardings, fares: m.fares, economy: m.economy, runCostBusKm: m.runCostBusKm, lines };
}, [H, LEVEL, DAY]);

/** totals of a money snapshot: all lines, rail lines only, and the bus `line` */
const totals = (s, line = null) => {
  const sum = (f) => Object.entries(s.lines).filter(([id, l]) => f(id, l)).reduce((a, [, l]) => ({ riders: a.riders + l.riders, fares: a.fares + l.fares, run: a.run + l.run, upkeep: a.upkeep + l.upkeep }), { riders: 0, fares: 0, run: 0, upkeep: 0 });
  const all = sum(() => true), rail = sum((_, l) => !l.bus), bus = line === null ? null : s.lines[String(line)] ?? null;
  const profit = (x) => x.fares - x.run - x.upkeep;
  return { all: { ...all, profit: profit(all) }, rail: { ...rail, profit: profit(rail) }, bus: bus && { ...bus, profit: profit(bus) }, railTrips: s.rail, boardings: s.boardings, tripsPerBoarding: Math.min(1, s.rail / s.boardings) };
};

/** the weekday's final above version `after` (with `line` solved, not an estimate), from the newest
 * whole solve: the focus pool's rough solves of the zone in view go into the log too and never become
 * the view (lib's demandFinal waits on them for ever), so they are skipped; then 15 s with no newer one */
const finalWith = async (page, line, after, timeout = 900000) => {
  const ready = () => page.waitForFunction(({ line, after, di }) => {
    const d = window.twDemand, s = d?.log?.filter((x) => !x.rough).at(-1), v = d?.days?.()[di]?.view;
    if (!v || !s || v.version !== s.version || v.version === 0 || !v.final || d.client.waiting || v.version <= after || v.lines.size < 100) return false;
    if (line === null) return true;
    const l = v.lines.get(String(line));
    return !!l && !l.estimate;
  }, { line, after, di: DAY }, { timeout, polling: 500 });
  for (;;) {
    await ready();
    const v = await page.evaluate((di) => window.twDemand.days()[di].view.version, DAY);
    await page.waitForTimeout(15000);
    if (await page.evaluate(([di, v]) => window.twDemand.days()[di].view.version === v && window.twDemand.days()[di].view.final, [DAY, v])) return v;
  }
};
const version = (page) => page.evaluate((di) => window.twDemand.days()[di].view?.version ?? 0, DAY);

const b = await launchChrome({ prefix: 'tw-t1880-' });
const results = { when: new Date().toISOString(), base: BASE, scen: [] };
const errors = [];
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=${process.env.DEMAND_WORKERS ?? 2}&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await finalWith(page, null, 0);
  const base = await money(page);
  results.baseline = { ...totals(base), di: base.di, fares: base.fares, economy: base.economy, runCostBusKm: base.runCostBusKm, split: base.split };
  console.log('baseline', JSON.stringify(results.baseline));
  for (const sc of PICK) {
    const t0 = Date.now();
    const pts = sc.pts.map((p) => [...p]);
    if (sc.to) {
      const st = await page.evaluate(([lng, lat]) => {
        const all = window.tw.world.value.save.stations.filter((s) => s.built);
        const s = all.reduce((b, s) => (Math.hypot((s.lng - lng) * 0.76, s.lat - lat) < Math.hypot((b.lng - lng) * 0.76, b.lat - lat) ? s : b));
        return { lng: s.lng, lat: s.lat, name: s.name, id: s.id };
      }, sc.to);
      pts.push([st.lng, st.lat + 0.0005]);
      sc.station = st.name;
    }
    const v0 = await version(page);
    const before = await money(page);
    const r = await page.evaluate(async ([pts, hw]) => {
      const tw = window.tw, c = tw.client;
      const route = pts.flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
      const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
      const con = await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
      const h = hw ? await c.edit({ op: 'setHeadway', line: add.line, seconds: [hw, Math.round(hw * 1.25), Math.round(hw * 2.5)] }) : null;
      await new Promise((d) => setTimeout(d, 1500));
      const st = tw.world.value.lineStats[String(add.line)];
      return { line: add.line, ok: add.ok && con.ok && h?.ok !== false, charge: con.charge, km: st?.km, buses: st?.trainsNeeded, status: st?.status, stops: (st?.bus?.stops ?? []).map((s) => s.kind), stations: (st?.bus?.stops ?? []).filter((s) => s.kind === 1).length };
    }, [pts, sc.hw]);
    await finalWith(page, r.line, v0);
    const after = await money(page);
    const a = totals(after, r.line), z = totals(before);
    const d = {
      busProfit: a.bus?.profit ?? null,
      busProfitEven: a.bus ? a.bus.faresEven - a.bus.run - a.bus.upkeep : null,
      network: a.all.profit - z.all.profit,
      railProfit: a.rail.profit - z.rail.profit,
      railFares: a.rail.fares - z.rail.fares,
      railRiders: a.rail.riders - z.rail.riders,
      railRun: a.rail.run - z.rail.run,
      railTrips: a.railTrips - z.railTrips,
      tripsPerBoarding: a.tripsPerBoarding - z.tripsPerBoarding,
    };
    const works = d.busProfit < 0 && d.network > 0;
    // the rail lines that moved most: fares, and any change in running cost (the trains demand solved with)
    const lines = Object.entries(after.lines).filter(([id, l]) => !l.bus && before.lines[id]).map(([id, l]) => {
      const o = before.lines[id];
      return { id, name: l.name, fares: l.fares - o.fares, riders: l.riders - o.riders, run: l.run - o.run, tph: l.run !== o.run ? [o.tph, l.tph] : undefined };
    });
    const moved = [...lines].sort((x, y) => Math.abs(y.fares) - Math.abs(x.fares)).slice(0, 8);
    const ran = lines.filter((l) => Math.abs(l.run) > 1);
    results.scen.push({ ...sc, pts, add: r, before: z, after: a, d, works, moved, ran, secs: (Date.now() - t0) / 1000 });
    console.log(sc.id, JSON.stringify({ station: sc.station, km: r.km?.toFixed(2), buses: r.buses, stops: r.stops.length, stations: r.stations, riders: Math.round(a.bus?.riders ?? 0), fares: Math.round(a.bus?.fares ?? 0), run: Math.round(a.bus?.run ?? 0), busProfit: Math.round(d.busProfit), busProfitEven: Math.round(d.busProfitEven ?? 0), network: Math.round(d.network), railFares: Math.round(d.railFares), railRiders: Math.round(d.railRiders), railTrips: Math.round(d.railTrips), works, ok: r.ok }));
    await writeFile(OUT, JSON.stringify(results, null, 1));
    const v1 = await version(page);
    await page.evaluate((l) => window.tw.client.edit({ op: 'removeLine', line: l }), r.line);
    await page.waitForTimeout(2500);
    await finalWith(page, null, v1);
  }
  const end = await money(page);
  results.baselineAfter = totals(end);
  console.log('baseline after', JSON.stringify({ profit: results.baselineAfter.all.profit, rail: results.baselineAfter.railTrips }));
} catch (e) {
  console.log('FAILED', e.message);
  results.error = e.message;
} finally {
  results.errors = errors.slice(0, 5);
  await writeFile(OUT, JSON.stringify(results, null, 1));
  await b.close();
}
