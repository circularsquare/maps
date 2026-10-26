// T-1174: a demand worker whose WASM panics is replaced, reopens what it held, and the solve is sent
// again; the figures come out as before. On the real save: a panic in an idle solving worker (the
// pair of errors T-1174 saw: a swallowed panic, then "recursive use of an object"), one in a solving
// worker in the solve of a train count edit, and one in the aux worker; after each, the weekday's rail
// trips a day must equal those of the same network solved with no panic, to the trip, a station click
// must still be answered, and no console error (one is "periods with different corridor tables").
// Usage (maps root): node trainworld/checks/t1174-panic-recover.mjs [dev server url]. About 4 minutes.
// Dev server only: `debugPanic` is compiled out of a production build.
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, NOISE, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6174/';
let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };

const { ctx, close } = await launchChrome({ prefix: 'tw-t1174-' });
const errors = [], warnings = [];
try {
  const page = await ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
  collectErrors(page, errors, NOISE, /glyphs/);
  page.on('console', (m) => { if (m.type() === 'warning' && /panicked/.test(m.text())) warnings.push(m.text()); });
  await page.goto(`${BASE}${BASE.includes('?') ? '&' : '?'}debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { polling: 250, lines: 100 });
  const state = () => page.evaluate(async () => {
    const d = window.twDemand, v = d.view();
    const station = window.tw.world.value.save.stations[37]?.id;
    const riders = station ? await d.stationRiders(station) : null;
    return { version: v.version, rail: Math.round(v.railTripsPerDay), status: String(d.client.status), workers: d.client.computeN, station: !!riders };
  });
  const first = await state();
  console.log(`      first final: ${first.rail.toLocaleString('en-US')} rail trips a day, version ${first.version}, ${first.workers} solving workers`);

  // the busiest line's train count, and two more trains at each level: an edit whose solve steps
  // (a solve of an unchanged network keeps every cluster and steps nothing)
  const line = await page.evaluate(() => {
    const w = window.tw.world.value;
    const ls = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id].autoS);
    ls.sort((a, b) => w.lineStats[b.id].trainsNeeded[1] - w.lineStats[a.id].trainsNeeded[1]);
    return { id: ls[0].num, trains: w.lineStats[ls[0].id].trainsNeeded };
  });
  const more = line.trains.map((n) => n + 2);
  const edit = (trains) => page.evaluate(([id, trains]) => window.tw.client.edit({ op: 'setTrainCount', line: id, trains }).then((r) => r.ok), [line.id, trains]);
  const settle = (after) => page.waitForFunction((after) => {
    const d = window.twDemand, s = d.log.at(-1), v = d.view();
    return String(d.client.status).startsWith('failed') || (!!v && s && v.version === s.version && v.final && !d.client.waiting && v.version > after);
  }, after, { timeout: 300000, polling: 250 });
  let last = first;
  await edit(more);
  await settle(last.version);
  const edited = (last = await state());
  await edit(line.trains);
  await settle(last.version);
  last = await state();
  console.log(`      line ${line.id} at ${more.join('/')} trains: ${edited.rail.toLocaleString('en-US')} rail trips a day; back at ${line.trains.join('/')}: ${last.rail.toLocaleString('en-US')}`);

  // [name, worker, when, the edit that follows the panic, the rail trips a day expected after]
  // (an edit and its undo come back a few trips off the first final on the same page, from before
  // T-1174: so an idle panic is held to the figure just before it, the edit to the edit's own)
  const cases = [
    ['idle worker 0, swallowed panic then recursive use', 0, 'now', null, null],
    ['worker 1 mid-solve', Math.min(1, first.workers - 1), 'solve', more, edited.rail],
    ['aux worker', first.workers, 'now', null, null],
  ];
  for (const [name, i, when, trains, wanted] of cases) {
    const t = Date.now(), seen = warnings.length, want = wanted ?? last.rail;
    await page.evaluate(([i, when]) => window.twDemand.client.debugPanic(i, when), [i, when]);
    if (trains) await edit(trains);
    try {
      await settle(last.version);
      const now = await state();
      const warned = warnings.slice(seen).join(' | ');
      report(name, now.rail === want && !now.status.startsWith('failed') && now.station && /test panic \(T-1174\)/.test(warned),
        `(${now.rail.toLocaleString('en-US')} rail trips a day, want ${want.toLocaleString('en-US')}, version ${now.version}, station answered ${now.station}, ${((Date.now() - t) / 1000).toFixed(1)} s; ${warned.slice(0, 220)})`);
      last = now;
    } catch (e) {
      const s = await page.evaluate(() => String(window.twDemand.client.status));
      report(name, false, `${e.message.split('\n')[0]}; status ${s}`);
    }
  }
  report('no console errors', errors.length === 0, errors.slice(0, 5).join(' | '));
} finally {
  await close();
}
console.log(failed ? `T1174 FAILED: ${failed}` : 'T1174 PASSED');
process.exit(failed ? 1 : 0);
