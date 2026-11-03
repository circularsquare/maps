// T-2104: each line's fares and profit a day under the base fare's share (half by boardings, half by
// passenger-km, at most one base a boarding) against the even share per boarding it replaced, on one
// solved save: the same riders, only the base's split differs. PASS when the network's fares come out
// the same to the cent. Notes: notes/T-2104.md.
//   node trainworld/checks/t2104-fare-share.mjs [url] [save] [out.json] [day index, 0 = weekday]
// from the maps root, on your own Vite (port 6169 by default). Money is game dollars (the economy's 100x).
import { readFile, writeFile } from 'node:fs/promises';
import { ARTIFACTS, REAL_SAVE, collectErrors, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6169/';
const SAVE = process.argv[3] ?? REAL_SAVE;
const OUT = process.argv[4] ?? `${ARTIFACTS}t2104/nyc.json`;
const DAY = +(process.argv[5] ?? 0);
const H = [4, 6, 4, 4, 6], LEVEL = [0, 0, 0, 1, 2]; // game/periods.ts: hours and demand level of each period

/** the newest whole solve's final for day `di` (the focus pool's rough solves skipped), then 15 s with no newer one */
const finalOf = async (page, di, timeout = 1800000) => {
  for (;;) {
    await page.waitForFunction((di) => {
      const d = window.twDemand, s = d?.log?.filter((x) => !x.rough).at(-1), v = d?.days?.()[di]?.view;
      return !!v && !!s && v.version === s.version && v.version > 0 && v.final && !d.client.waiting && v.lines.size > 0;
    }, di, { timeout, polling: 1000 });
    const v = await page.evaluate((di) => window.twDemand.days()[di].view.version, di);
    await page.waitForTimeout(15000);
    if (await page.evaluate(([di, v]) => window.twDemand.days()[di].view.version === v && window.twDemand.days()[di].view.final, [di, v])) return v;
  }
};

const b = await launchChrome({ prefix: 'tw-t2104-' });
const errors = [];
let out = { when: new Date().toISOString(), base: BASE, save: SAVE };
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  const t0 = Date.now();
  await finalOf(page, DAY);
  out = { ...out, solveSecs: (Date.now() - t0) / 1000, ...await page.evaluate(([H, LEVEL, di]) => {
    const tw = window.tw, w = tw.world.value, m = tw.moneyState.value, v = window.twDemand.days()[di].view;
    const inc = tw.incomeByDay.value[di];
    const even = tw.baseFares(v, 1), now = tw.baseFares(v);
    const quiet = di === 1 || di === 2;
    const lines = [];
    let boardings = 0;
    for (const l of w.save.lines) boardings += v.lines.get(l.id)?.ridersPerDay ?? 0;
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st || st.status !== 'running') continue;
      const ld = v.lines.get(l.id);
      const tph = ld?.tph ?? [l.tph.high, l.tph.medium, l.tph.low];
      const cars = ld?.cars ?? st.cars;
      const perKm = st.bus ? m.runCostBusKm : st.fast ? m.hsRunCostTrainKm + cars * m.hsRunCostCarKm : m.runCostTrainKm + cars * m.runCostCarKm;
      let run = 0;
      H.forEach((h, p) => { const lev = LEVEL[p] === 0 && quiet ? 1 : LEVEL[p]; run += tph[lev] * 2 * st.km * perKm * m.economy * h; });
      const lm = inc?.lines.get(l.id);
      const fares = lm ? lm.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 : 0;
      const baseNew = now[l.id] ?? 0, baseEven = even[l.id] ?? 0;
      const riders = ld?.ridersPerDay ?? 0;
      // passenger-km a day and mean ride, the plain km (the share's weight)
      let pkm = 0;
      const s = st.stopS, n = l.stops.length;
      if (ld && s[0]?.length === n && s[1]?.length === n) {
        const km = [];
        for (const r of s) for (let k = 0; k + 1 < n; k++) km.push((r[k + 1] - r[k]) / 1000);
        for (const seg of ld.loads) if (seg.length === km.length) for (let k = 0; k < seg.length; k++) pkm += seg[k] * km[k];
      }
      lines.push({ id: l.id, name: l.name, bus: !!st.bus, fast: !!st.fast, km: st.km, riders, pkm, ride: riders > 0 ? pkm / riders : 0, fares, faresEven: fares - baseNew + baseEven, baseNew, baseEven, run, upkeep: st.upkeep });
    }
    const sum = (f) => lines.reduce((a, l) => a + f(l), 0);
    const network = inc ? inc.perHour.reduce((s, x, p) => s + x * H[p], 0) * 1e6 : 0;
    return { di, version: v.version, rail: v.railTripsPerDay, boardings, economy: m.economy, fares: m.fares, network, sumFares: sum((l) => l.fares), sumFaresEven: sum((l) => l.faresEven), sumBaseNew: sum((l) => l.baseNew), sumBaseEven: sum((l) => l.baseEven), lines };
  }, [H, LEVEL, DAY]) };
  const diff = out.sumFares - out.sumFaresEven;
  out.pass = Math.abs(diff) < 0.01;
  for (const l of out.lines) { l.profit = l.fares - l.run - l.upkeep; l.profitEven = l.faresEven - l.run - l.upkeep; l.change = l.profit - l.profitEven; }
  console.log(`${out.pass ? 'PASS' : 'FAIL'} network fares ${out.sumFares.toFixed(2)} now, ${out.sumFaresEven.toFixed(2)} under the even share (difference ${diff.toFixed(6)}); base ${out.sumBaseNew.toFixed(2)} against ${out.sumBaseEven.toFixed(2)}; ${out.lines.length} lines`);
  const k = (x) => `${(x / 1e3).toFixed(0)}k`;
  for (const l of [...out.lines].sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, 15)) {
    console.log(`${l.name.padEnd(36)} ${l.bus ? 'bus' : l.fast ? 'hsr' : 'rail'} ride ${l.ride.toFixed(1)} km  riders ${Math.round(l.riders)}  profit ${k(l.profitEven)} -> ${k(l.profit)} (${l.change >= 0 ? '+' : ''}${k(l.change)})`);
  }
} catch (e) {
  console.log('FAILED', e.message);
  out.error = e.message;
} finally {
  out.errors = errors.slice(0, 5);
  await writeFile(OUT, JSON.stringify(out, null, 1));
  await b.close();
}
