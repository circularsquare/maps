// T-1536 probe: the real save on our own Vite; per line, per period: demand's riders, the account's
// crowding numbers (line_periods), the line's counts, and the game's own running cost a day.
// Adapted from notes/t1523/probe.mjs. usage: node probe.mjs <port> <out.json>
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const TREE = 'C:/Users/anita/projects/maps/.tw-trees/t1536/trainworld';
const lib = await import(pathToFileURL(`${TREE}/checks/lib.mjs`).href);
const [port, out] = process.argv.slice(2);
const BASE = `http://localhost:${port}/`;
const SCRATCH = 'C:/Users/anita/AppData/Local/Temp/claude/c--Users-anita-projects-maps/ee78dc43-077b-4b8e-9cf9-ac5bb7367793/scratchpad/t1536/';

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
  await page.waitForTimeout(30000);
  const res = await page.evaluate(async () => {
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const P = await import('/src/game/periods.ts');
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
        const r = {};
        if (d?.board && d?.alight) r.board = d.board[p].reduce((s, x) => s + x, 0);
        if (v && v.length >= 55) r.acct = Array.from(v.slice(p * 11, p * 11 + 11));
        per.push(r);
      }
      rows.push({ id: l.id, num: l.num, name: l.name, stops: l.stops.length, tph: l.tph, cars: st?.cars, need: st?.trainsNeeded, km: st?.km, status: st?.status, ridersPerDay: d?.ridersPerDay, per });
    }
    const { trainKind } = await import('/src/game/format.ts');
    const m = window.tw.moneyState.peek();
    for (const row of rows) {
      const st = w.lineStats[row.id];
      const k = trainKind(m, st.fast);
      row.perTrainKm = (st.bus ? m.runCostBusKm : k.runCostTrainKm + st.cars * k.runCostCarKm) * m.economy;
    }
    const LEVEL = { high: 0, medium: 1, low: 2 };
    const levelOfHour = Array.from({ length: 24 }, (_, h) => P.PERIODS[P.periodAt(h * 3600)].demand);
    let runDaily = 0;
    for (let h = 0; h < 24; h++) runDaily += m.runningPerHour[LEVEL[levelOfHour[h]]];
    return {
      missing, weekday, rail: dv.railTripsPerDay, trips: dv.tripsPerDay, peak: dv.peakHourFactor, levelOfHour, periods: P.PERIODS,
      runningPerHour: Array.from(m.runningPerHour), runDaily, rows,
    };
  });
  res.errors = errors.slice(0, 20);
  await writeFile(out, JSON.stringify(res, null, 1));
  console.log('wrote', out, 'missing', res.missing, 'errors', errors.length, 'runDaily', res.runDaily);
} finally {
  await close();
}
