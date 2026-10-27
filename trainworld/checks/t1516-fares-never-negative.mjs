// T-1516: no line's fares go below zero under a tapering fare curve, where a trip's curve extra is
// a discount. A new game (a 40 km rail line east across Queens and Long Island with a 4 km bus feeder
// at its west end) and the real save (a feeder to a Queens station), each under a curve flat past
// 1 km: every running line's fares a day, by money.ts, are at least zero, and the bus's are above
// zero with its share of the curve's extra (printed). About 6 minutes.
//   node trainworld/checks/t1516-fares-never-negative.mjs [url]   (from the maps root; default localhost:6116)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, buildLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6116/';
let failed = 0;
const say = (ok, what, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? `  (${detail})` : ''}`); };

/** Each running line's fares a day (US$, the economy's) and its part from the curve's extra. */
const fares = (page) => page.evaluate(async () => {
  const money = await import('/src/game/money.ts');
  const demand = await import('/src/game/demand.ts');
  const { PERIODS } = await import('/src/game/periods.ts');
  const w = window.tw.world.value, m = money.moneyState.value;
  const dv = demand.demandByDay.value[0].view, inc = money.incomeByDay.value[0];
  const out = [];
  for (const l of w.save.lines) {
    const st = w.lineStats[l.id], lm = inc.lines.get(l.id), dm = dv.lines.get(l.id);
    if (!lm || st?.status !== 'running') continue;
    const day = lm.perHour.reduce((a, v, p) => a + v * (PERIODS[p].to - PERIODS[p].from), 0) * 1e6;
    out.push({ id: l.id, bus: !!st.bus, riders: Math.round(dm?.ridersPerDay ?? 0), fares: Math.round(day), extra: Math.round((dm?.fareX ?? []).reduce((a, b) => a + b, 0) * m.economy) });
  }
  return out;
});

/** A curve at the first stretch's rate for 1 km, then almost flat. */
const taper = (page) => page.evaluate(async () => {
  const f = (await import('/src/game/money.ts')).moneyState.value.fares;
  return (await window.tw.client.edit({ op: 'fares', nodes: [[0, f.base], [1, f.base + f.perKm], [100, f.base + f.perKm + 0.05]] })).ok;
});

const feeder = (page, from, to) => page.evaluate(async ([from, to]) => {
  const tw = window.tw, c = tw.client;
  const route = [from, to].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
  const add = await c.edit({ op: 'addBusLine', route, name: '', colour: '#d7263d' });
  const con = await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] });
  return { line: String(add.line), ok: add.ok && con.ok };
}, [from, to]);

const judge = (where, all, bus) => {
  const neg = all.filter((x) => x.fares < 0);
  say(all.length > 1 && neg.length === 0, `${where}: no line's fares below zero`, neg.length ? neg.map((x) => `${x.id}: ${x.fares}`).join(', ') : `${all.length} lines`);
  const b = all.find((x) => x.id === bus);
  say(!!b && b.fares > 0 && b.extra <= 0, `${where}: the bus keeps a positive fare with its share of the taper`, b ? `fares $${b.fares.toLocaleString('en-US')} a day, of it extra $${b.extra.toLocaleString('en-US')}, ${b.riders.toLocaleString('en-US')} riders` : 'no bus');
};

const b = await launchChrome({ prefix: 'tw-t1516-' });
const errors = [];
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);

  // ---- a new game
  await page.goto(`${BASE}?new=1&debug=1&paused=1`);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value && window.twDemand?.client, null, { timeout: 180000 });
  const [lng, lat] = [-73.87, 40.75];
  const dLng = 5 / (111.32 * Math.cos((lat * Math.PI) / 180));
  const r = await buildLine(page, [0, 1, 2, 3, 4, 5, 6, 7, 8].map((k) => [lng + k * dLng, lat]), { trains: [12, 8, 6] });
  say(r.ok && await taper(page), 'new game: the line and the tapering curve are in');
  const f = await feeder(page, [lng, lat + 0.036], [lng, lat + 0.0005]);
  await page.waitForTimeout(1500);
  await demandFinal(page);
  await page.waitForTimeout(1500);
  judge('new game', await fares(page), f.line);

  // ---- the real save
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  say(await taper(page), 'real save: the tapering curve is in');
  const at = await page.evaluate(() => {
    const st = window.tw.world.value.save.stations.filter((s) => s.built);
    const s = st.reduce((b, s) => (Math.hypot(s.lng + 73.80, s.lat - 40.70) < Math.hypot(b.lng + 73.80, b.lat - 40.70) ? s : b));
    return [s.lng, s.lat];
  });
  const g = await feeder(page, [at[0] + 0.03, at[1] + 0.02], at);
  await page.waitForTimeout(1500);
  await demandFinal(page, { lines: 100 });
  await page.waitForTimeout(1500);
  judge('real save', await fares(page), g.line);
} catch (e) {
  say(false, 'ran to the end', e.message);
} finally {
  say(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
  await b.close();
}
console.log(failed ? `T-1516 FAILED: ${failed}` : 'T-1516 PASSED');
process.exit(failed ? 1 : 0);
