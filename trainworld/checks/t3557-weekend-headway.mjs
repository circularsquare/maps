// T-3557: stepping a headway line's weekend headway in the timetables window makes the weekends run that
// headway. A new game in New York, a 3-stop line built through the play harness (lib/play.mjs), its schedule
// window opened from the line panel, Headway picked for the weekdays; then the Weekends High row's headway
// stepper is pressed longer and shorter. After each press, one line: the headway the stepper asked the sim
// for (the stored weekend headway, `lineStats.weekend[5]`), against the measured gap between the line's real
// High-level departures on the next Saturday (the drawn trips, `lineTrips`, not the sim's settled rate). The
// Medium and Low levels and the weekdays must not move. A control at the end sends the old edit (the same
// counts, no headway: a percent first drops the stored headways) and the departures run faster than stepped,
// which is the bug. From a tree's root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6154):
//   node trainworld/checks/t3557-weekend-headway.mjs [dev url]     (about 4 minutes; default localhost:6154)
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6154/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32]; // km east, north

const g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', ui: { shade: 'light', v: 2 }, prefix: 'tw-t3557-' });
try {
  await g.act.camera(C, 12.5);
  await g.act.tool('track', { type: 'regular', tracks: 2, snap: { real: false, roads: false, beside: false }, keepPlacing: false });
  const A = P(-4, 0), B = P(0, 0.3), D = P(4, 0);
  const run = await g.act.drawTrack([A, B, D]);
  check(run.ok, 'track drawn');
  for (const p of [A, B, D]) await g.act.placeStation(p);
  await g.act.construct();
  const line = await g.act.newLine([A, B, D], { name: 'Headway', zoom: 15 });
  check(line?.stops.length === 3, 'a 3-stop line built');
  await g.act.setTrains([8, 6, 4]);

  await g.page.evaluate(async () => {
    window.__m = { cc: await import('/src/workers/clockClient.ts'), hol: await import('/src/game/holidays.ts'), days: await import('/src/game/days.ts'), w: await import('/src/game/world.ts') };
  });
  const S = (f, arg) => g.page.evaluate(`(async () => (${f.toString()})({ ...window.__m, tw: window.tw }, ${JSON.stringify(arg ?? null)}))()`);
  const who = await S(({ tw }) => { const l = tw.world.value.save.lines.at(-1); return { id: l.id, num: l.num }; });
  await g.page.click('button[data-tip="Detailed schedule"]');
  await g.page.waitForSelector('dialog.sched-panel[open]', { timeout: 10000 });
  await g.page.locator('dialog.sched-panel button:text-is("Headway")').first().click();
  await g.page.waitForFunction((id) => window.tw.world.value.lineStats[id]?.autoS, who.id, { timeout: 30000 });
  await g.page.waitForTimeout(2500);

  // the next Saturday in the line's own time, its clock hour 13:00
  const day = await S(({ tw, hol }, w) => {
    const l = tw.world.value.save.lines.find((x) => x.id === w.id), st = tw.world.value.lineStats[w.id];
    const c = hol.lineCountry(l.stops), shift = st.shift ?? 0, today = Math.floor((tw.clock.now() + shift) / 86400);
    for (let d = today + 2; d < today + 60; d++) if (hol.kindIn(c, d) === 1 && hol.rowIn(c, d) === 1 && hol.rowIn(c, d - 1) === 0) return { d, shift };
    return null;
  }, who);
  check(!!day, 'a Saturday ahead');
  const t0 = day.d * 86400 - day.shift, t1 = t0 + 86400;

  // the median gap between consecutive departures of the line at level `lev` on the Saturday, minutes
  const gap = (lev) => S(async ({ cc }, a) => {
    const trips = await cc.queryClock({ what: 'lineTrips', line: a.num, t0: a.t0, t1: a.t1 });
    const by = new Map();
    for (let i = 0; i + 3 < trips.length; i += 4) {
      const k = trips[i];
      if (k % 3 !== a.lev) continue;
      if (!by.has(k)) by.set(k, []);
      by.get(k).push(trips[i + 1]);
    }
    const gaps = [];
    for (const ts of by.values()) { ts.sort((x, y) => x - y); for (let i = 1; i < ts.length; i++) gaps.push(ts[i] - ts[i - 1]); }
    gaps.sort((x, y) => x - y);
    return gaps.length ? gaps[gaps.length >> 1] / 60 : NaN;
  }, { num: who.num, t0, t1, lev });
  const state = () => S(({ tw }, id) => { const w = tw.world.value, st = w.lineStats[id], l = w.save.lines.find((x) => x.id === id); return { weekend: st.weekend, rest: st.restTph, week: [l.tph.high, l.tph.medium, l.tph.low], auto: st.autoS }; }, who.id);

  const before = await state();
  const mid0 = [await gap(1), await gap(2)];
  console.log('start', JSON.stringify(before), 'gaps medium/low', mid0.map((x) => x.toFixed(2)));
  const press = async (more) => {
    await g.act.click(`${more ? 'More' : 'Less'} Headway on weekends`, { within: 'dialog.sched-panel', nth: 0 });
    await g.page.waitForTimeout(3500);
  };
  let n = 0;
  for (const more of [true, true, true, false, false, false, false, false]) {
    await press(more);
    const s = await state();
    const stepped = (s.weekend?.[5] ?? NaN) / 60, measured = await gap(0), floor = before.auto[0] / 60;
    console.log('   weekend', JSON.stringify(s.weekend), 'rest', JSON.stringify(s.rest));
    // a headway shorter than the track allows (the weekdays' shortest, `floor`) runs at that
const want = Math.max(stepped, floor);
    check(Math.abs(measured - want) <= 0.03 * want + 0.1, `press ${++n} (${more ? 'longer' : 'shorter'}): stepped ${stepped.toFixed(1)} min, measured ${measured.toFixed(1)} min${stepped < floor ? ` (the track holds ${floor.toFixed(1)})` : ''}`);
    check(JSON.stringify(s.week) === JSON.stringify(before.week), `press ${n}: weekdays unchanged`);
    const mid = [await gap(1), await gap(2)];
    check(mid.every((m, i) => Math.abs(m - mid0[i]) <= 0.03 * mid0[i] + 0.1), `press ${n}: Medium and Low still ${mid.map((x) => x.toFixed(1)).join(' and ')} min`);
  }
  const end = await state();
  check(JSON.stringify(end.auto) === JSON.stringify(before.auto), 'weekday headways unchanged', JSON.stringify(end.auto));

  // the old edit: the counts of the 5 minute step with no headway run faster than 5 minutes
  const counts = [4, 5, 4];
  const send = (op) => S(({ w }, o) => w.edit(o), op);
  await send({ op: "weekendCounts", line: who.num, counts, headways: [300, 240, 300] });
  await g.page.waitForTimeout(3500);
  const kept = await gap(0);
  await send({ op: "weekendShare", line: who.num, share: 100 });
  await g.page.waitForTimeout(3000);
  await send({ op: "weekendCounts", line: who.num, counts });
  await g.page.waitForTimeout(3500);
  const old = await gap(0);
  check(Math.abs(kept - 5) <= 0.25, `with the headway: stepped 5.0 min, measured ${kept.toFixed(1)} min`);
  check(old < 5 - 0.1, `the old edit (the same counts, no headway): stepped 5.0 min, measured ${old.toFixed(1)} min, faster`);
  const errs = g.pageErrors();
  check(errs.length === 0, 'no console errors', errs.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await g.close();
}
console.log(fail ? `T-3557 FAILED: ${fail}` : 'T-3557 PASSED');
process.exit(fail ? 1 : 0);
