// T-1625: load a Japan save into a new game on Vite 6171, wait for the weekday's final demand, dump each
// line's figures to OUT (json). env: SAVE, OUT, LABEL, PORT (6171), WORKERS (2)
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, GL } from '../../checks/lib.mjs';

const SAVE = process.env.SAVE, OUT = process.env.OUT, LABEL = process.env.LABEL ?? 'run';
const PORT = process.env.PORT ?? '6171', WORKERS = process.env.WORKERS ?? '2';
const log = (s) => console.log(`${new Date().toISOString().slice(11, 19)} ${s}`);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1625-', size: [1400, 900], args: [...GL] });
const errors = [];
try {
  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('pageerror', (e) => { errors.push('pageerror: ' + e.message); log('pageerror ' + e.message.slice(0, 200)); });
  page.on('console', (m) => { if (m.type() === 'error' && !/favicon|openfreemap|AJAXError|^\w{1,3}$/.test(m.text())) { errors.push(m.text().slice(0, 300)); log('console.error ' + m.text().slice(0, 200)); } });
  await page.goto(`http://localhost:${PORT}/?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  log('new game up');
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 900000 });
  log('save in: ' + JSON.stringify(await page.evaluate(() => ({ lines: window.tw.world.value.save.lines.length, stations: window.tw.world.value.save.stations.length }))));
  const t = Date.now();
  const tick = setInterval(async () => {
    try {
      const s = await page.evaluate(() => { const d = window.twDemand, v = d?.view?.(); return { round: v?.round, final: v?.final, lines: v?.lines?.size, waiting: d?.client?.waiting }; });
      log(`  demand ${Math.round((Date.now() - t) / 60000)} min: ${JSON.stringify(s)}`);
    } catch { /* page busy */ }
  }, 300000);
  await demandFinal(page, { timeout: 4 * 3600000, polling: 2000, lines: 100 });
  clearInterval(tick);
  log(`demand final after ${((Date.now() - t) / 1000).toFixed(0)} s`);
  const dump = await page.evaluate(() => {
    const w = window.tw.world.value, days = window.twDemand.days();
    const views = days.map((d) => d?.view ?? null);
    const nameOf = new Map(w.save.stations.map((s) => [s.id, s.name]));
    const out = { label: null, hours: [4, 6, 4, 4, 6], levels: ['high', 'high', 'high', 'medium', 'low'], lines: [], days: [] };
    for (const [k, v] of views.entries()) out.days.push(v && { k, day: v.day, final: v.final, round: v.round, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, split: v.split });
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st) continue;
      const n = l.stops.length;
      const km = [], kmAbs = [];
      for (const s of st.stopS ?? []) for (let k = 0; k + 1 < n; k++) { km.push((s[k + 1] - s[k]) / 1000); kmAbs.push(Math.abs(s[k + 1] - s[k]) / 1000); }
      const rec = { id: l.id, name: l.name, n, status: st.status, first: nameOf.get(l.stops[0]), last: nameOf.get(l.stops[n - 1]), km: st.stopS?.[0] ? (st.stopS[0][n - 1] - st.stopS[0][0]) / 1000 : 0,
        cars: st.cars, fast: st.fast, tph: l.tph, rt: st.roundTripByLevel, needed: st.trainsNeeded, delayS: st.delayS, upkeep: st.upkeep, pkm: {}, riders: {} };
      for (const d of [0, 6]) {
        const dm = views[d]?.lines.get(l.id);
        rec.riders[d] = dm?.ridersPerDay ?? 0;
        rec.pkm[d] = (dm?.loads ?? []).map((seg) => { let a = 0; if (seg.length === kmAbs.length) for (let k = 0; k < seg.length; k++) a += seg[k] * kmAbs[k]; return a; });
      }
      out.lines.push(rec);
    }
    const m = window.tw.moneyState.value;
    out.money = m && { economy: m.economy, fares: m.fares, runCostTrainKm: m.runCostTrainKm, runCostCarKm: m.runCostCarKm, hsRunCostTrainKm: m.hsRunCostTrainKm, hsRunCostCarKm: m.hsRunCostCarKm, upkeep: m.upkeep, fleet: m.fleet, fleetFast: m.fleetFast };
    try { const inc = window.tw.incomeByDay.value; out.income = inc.slice(0, 3).map((i) => ({ perHour: i.perHour, rides: i.rides })); } catch { /* */ }
    return out;
  });
  dump.label = LABEL;
  dump.errors = errors;
  await writeFile(OUT, JSON.stringify(dump));
  log(`dumped ${dump.lines.length} lines -> ${OUT}`);
} catch (e) {
  log('FAILED ' + (e?.stack ?? e));
  process.exitCode = 1;
} finally {
  await close();
}
