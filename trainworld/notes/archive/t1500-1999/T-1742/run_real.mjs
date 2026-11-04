// T-1742: load one real-network save into a new game on Vite 6195, wait for the weekday's final demand, dump lines.
// env: CITY (sdj..), OUT (json), PORT (6195), WORKERS (1)
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, GL } from 'file:///C:/Users/anita/projects/maps/trainworld/checks/lib.mjs';

const CITY = process.env.CITY, OUT = process.env.OUT;
const SAVE = `C:/Users/anita/projects/maps/trainworld/data/packs/real/${CITY}.save`;
const PORT = process.env.PORT ?? '6195', WORKERS = process.env.WORKERS ?? '1';
const log = (s) => console.log(`${new Date().toISOString().slice(11, 19)} ${s}`);
const { ctx, close } = await launchChrome({ prefix: 'tw-t1742-' + CITY + '-', size: [1400, 900], args: [...GL] });
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
  log('save in: ' + JSON.stringify(await page.evaluate(() => ({ lines: window.tw.world.value.save.lines.length, stations: window.tw.world.value.save.stations.length, lastLoad: window.tw.client.lastLoad, header: Object.keys(window.tw.world.value.save).slice(0, 30) }))));
  const t = Date.now();
  const tick = setInterval(async () => {
    try {
      const s = await page.evaluate(() => { const d = window.twDemand, v = d?.view?.(); return { round: v?.round, final: v?.final, lines: v?.lines?.size, waiting: d?.client?.waiting }; });
      log(`  demand ${Math.round((Date.now() - t) / 60000)} min: ${JSON.stringify(s)}`);
    } catch { /* page busy */ }
  }, 120000);
  await demandFinal(page, { timeout: 3 * 3600000, polling: 2000, lines: 3 });
  clearInterval(tick);
  log(`demand final after ${((Date.now() - t) / 1000).toFixed(0)} s`);
  const dump = await page.evaluate(() => {
    const w = window.tw.world.value, days = window.twDemand.days();
    const views = days.map((d) => d?.view ?? null);
    const nameOf = new Map(w.save.stations.map((s) => [s.id, s.name]));
    const out = { city: null, lineKeys: Object.keys(w.save.lines[0] ?? {}), statKeys: Object.keys(w.lineStats[w.save.lines[0]?.id] ?? {}), viewKeys: Object.keys(views[0] ?? {}), lines: [], days: [] };
    for (const [k, v] of views.entries()) out.days.push(v && { k, day: v.day, final: v.final, round: v.round, tripsPerDay: v.tripsPerDay, railTripsPerDay: v.railTripsPerDay, split: v.split });
    const vl0 = views[0]?.lines; const first = vl0 && [...vl0.values()][0];
    out.lineViewKeys = first ? Object.keys(first) : [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st) continue;
      const n = l.stops.length;
      const kmAbs = [];
      for (const s of st.stopS ?? []) for (let k = 0; k + 1 < n; k++) kmAbs.push(Math.abs(s[k + 1] - s[k]) / 1000);
      const rec = { id: l.id, name: l.name, kind: l.kind ?? l.type ?? null, operator: l.operator ?? null, n, status: st.status, first: nameOf.get(l.stops[0]), last: nameOf.get(l.stops[n - 1]),
        km: st.stopS?.[0] ? (st.stopS[0][n - 1] - st.stopS[0][0]) / 1000 : 0, cars: st.cars, fast: st.fast, tph: l.tph, pkm: {}, riders: {}, boardings: {} };
      for (const d of [0, 6]) {
        const dm = views[d]?.lines.get(l.id);
        rec.riders[d] = dm?.ridersPerDay ?? 0;
        rec.boardings[d] = dm?.boardings ?? 0;
        rec.pkm[d] = (dm?.loads ?? []).map((seg) => { let a = 0; if (seg.length === kmAbs.length) for (let k = 0; k < seg.length; k++) a += seg[k] * kmAbs[k]; return a; });
      }
      out.lines.push(rec);
    }
    return out;
  });
  dump.errors = errors;
  dump.city = CITY;
  await writeFile(OUT, JSON.stringify(dump));
  log(`dumped ${dump.lines.length} lines -> ${OUT}`);
} catch (e) {
  log('FAILED ' + (e?.stack ?? e));
  process.exitCode = 1;
} finally {
  await close();
}
