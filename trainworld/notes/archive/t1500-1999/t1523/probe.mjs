// T-1523 probe: the real save on our own Vite; per line, per period: demand's riders, the busiest
// segment's load, the account's crowding numbers (line_periods), written as JSON.
// usage: node probe.mjs <port> <out.json>
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const TREE = 'C:/Users/anita/projects/maps/.tw-trees/t1523/trainworld';
const lib = await import(pathToFileURL(`${TREE}/checks/lib.mjs`).href);
const [port, out] = process.argv.slice(2);
const BASE = `http://localhost:${port}/`;
const SCRATCH = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/ee78dc43-077b-4b8e-9cf9-ac5bb7367793/scratchpad/t1523/';

const { ctx, close } = await lib.launchChrome({ prefix: SCRATCH + 'prof-' });
try {
  const page = await ctx.newPage();
  const errors = [];
  lib.collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.tw?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(3000);
  await lib.importSave(page, await readFile(lib.REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  console.log('save loaded');
  await lib.demandFinal(page, { polling: 500, lines: 100, timeout: 900000 });
  console.log('demand final');
  // mode: base, medHigh (Medium trains := High's), highMed (High trains := Medium's)
  const mode = process.argv[4] ?? 'base';
  if (mode !== 'base') {
    const ver = await page.evaluate(() => window.twDemand.view().version);
    const r = await page.evaluate(async (mode) => {
      const w = window.tw.world.value, c = window.tw.client;
      let ok = 0, bad = 0, same = 0;
      for (const l of w.save.lines) {
        const n = w.lineStats[l.id]?.trainsNeeded;
        if (!n) continue;
        const t = mode === 'medHigh' ? [n[0], n[0], n[2]] : [n[1], n[1], n[2]];
        if (t[0] === n[0] && t[1] === n[1]) { same++; continue; }
        const a = await c.edit({ op: 'setTrainCount', line: l.num, trains: t });
        if (a?.ok) ok++; else bad++;
      }
      return { ok, bad, same };
    }, mode);
    console.log('edits', JSON.stringify(r));
    await page.waitForTimeout(5000);
    await lib.demandFinal(page, { polling: 500, after: ver, timeout: 900000 });
    console.log('demand final after edits');
    await page.waitForTimeout(60000);
  }
  // wait for the account to have every line
  const res = await page.evaluate(async () => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const w = window.tw.world.value;
    const pause = (ms) => new Promise((r) => setTimeout(r, ms));
    const lines = w.save.lines;
    const t0 = performance.now();
    let missing = lines.length;
    while (performance.now() - t0 < 300000) {
      missing = 0;
      for (const l of lines) { const v = await queryClock({ what: 'linePeriods', line: l.num, day: 0 }); if (!v || !v.length) missing++; }
      if (missing <= 2) break;
      await pause(3000);
    }
    const { demandByDay } = await import('/src/game/demand.ts');
    const dv = demandByDay.value[0]?.view ?? window.twDemand.view();
    const weekday = !!demandByDay.value[0]?.view;
    const rows = [];
    for (const l of lines) {
      const st = w.lineStats[l.id];
      const d = dv.lines.get(l.id) ?? dv.lines.get(String(l.id)) ?? dv.lines.get(l.num);
      const v = await queryClock({ what: 'linePeriods', line: l.num, day: 0 });
      const per = [];
      for (let p = 0; p < 5; p++) {
        const r = { };
        if (d?.board && d?.alight) {
          const b = d.board[p], a = d.alight[p], n = b.length / 2;
          r.board = b.reduce((s, x) => s + x, 0);
          r.arrive = d.arrive ? d.arrive[p].reduce((s, x) => s + x, 0) : null;
          // busiest segment per direction
          r.maxSeg = [0, 1].map((run) => { let on = 0, best = 0; for (let k = 0; k < n; k++) { on += b[run * n + k] - a[run * n + k]; best = Math.max(best, on); } return best; });
          if (d.arrive) r.maxSegCome = [0, 1].map((run) => { let on = 0, best = 0; const c = d.arrive[p]; for (let k = 0; k < n; k++) { const sh = b[run * n + k] > 0 ? c[run * n + k] / b[run * n + k] : 1; on += c[run * n + k] - a[run * n + k] * Math.max(1, sh); best = Math.max(best, on); } return best; });
        }
        if (v && v.length >= 55) r.acct = Array.from(v.slice(p * 11, p * 11 + 11));
        per.push(r);
      }
      rows.push({ id: l.id, num: l.num, name: l.name, letter: l.letter, stops: l.stops.length, tph: l.tph, cars: st?.cars, rt: st?.roundTripByLevel, need: st?.trainsNeeded, auto: st?.autoS, km: st?.km, status: st?.status, shift: st?.shift, ridersPerDay: d?.ridersPerDay, purposes: Object.fromEntries([...(dv.purposes ?? [])].map(([k, p]) => [k, (p.lines.get(l.id) ?? p.lines.get(String(l.id)))?.ridersPerDay ?? 0])), per });
    }
    const { trainKind } = await import('/src/game/format.ts');
    const m = window.tw.moneyState.peek();
    for (const row of rows) {
      const st = w.lineStats[row.id];
      const k = trainKind(m, st.fast);
      row.perTrainKm = (st.bus ? m.runCostBusKm : k.runCostTrainKm + st.cars * k.runCostCarKm) * m.economy;
    }
    return { missing, weekday, rail: dv.railTripsPerDay, peak: dv.peakHourFactor, fares: window.tw.moneyState.peek()?.fares, rows };
  });
  res.errors = errors.slice(0, 20);
  await writeFile(out, JSON.stringify(res, null, 1));
  console.log('wrote', out, 'missing', res.missing, 'errors', errors.length);
} finally {
  await close();
}
