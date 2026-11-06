// T-2622: the 4x clock cap is the same wherever the camera is (game/lineFigures.ts `demandCapped`).
// On a save, the camera alone (a pan to a far zone and back) never caps; then during a loud line edit's
// update and a fare curve edit's, the cap is read from one view, the camera moved to the other, and read again:
// the two must match whenever the cap's own inputs did not move between. From the maps root:
//   node trainworld/checks/t2622-cap-view.mjs [url] [save] [demand workers]   (a few minutes)
import { readFile } from 'node:fs/promises';
import { REAL_SAVE, GL, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const [BASE = 'http://localhost:6173/', SAVE = REAL_SAVE, WORKERS = '1'] = process.argv.slice(2);
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t2622-', size: [1400, 900], args: GL });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=${WORKERS}&v=${Date.now()}`, { timeout: 180000 });
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await demandFinal(page, { held: true, timeout: 900000 });

  // view A: the zone with the most lines; view B: the line farthest from it, else 400 km west
  const { a, b, line } = await page.evaluate(async () => {
    const w = window.tw.world.value, ci = await import('/src/game/cities.ts'), ck = await import('/src/game/clock.ts'), ty = await import('/src/game/types.ts');
    const at = new Map(w.save.stations.map((s) => [s.id, [s.lng, s.lat]]));
    // (a stop dropped from a line with trains at the level in force: loud, and the line reads "loading...")
    const ls = w.save.lines.filter((l) => w.lineStats[l.id]?.status === 'running' && !w.lineStats[l.id].bus && l.stops.length >= 5 && at.has(l.stops[0]))
      .map((l) => { const st = w.lineStats[l.id]; return { num: l.num, at: at.get(l.stops[0]), zone: ci.cityWithin(...at.get(l.stops[0])), trains: st.trainsNeeded[ty.DEMAND_INDEX[ck.demandAt(ck.lineSeconds(st))]] > 0 }; });
    const count = new Map();
    for (const l of ls) count.set(l.zone, (count.get(l.zone) ?? 0) + 1);
    const top = [...count].sort((x, y) => y[1] - x[1])[0][0];
    const home = ls.find((l) => l.zone === top && l.trains);
    const d = (l) => Math.hypot(l.at[0] - home.at[0], l.at[1] - home.at[1]);
    const far = ls.filter((l) => l.zone !== top).sort((x, y) => d(y) - d(x))[0];
    return { a: home.at, b: far ? far.at : [home.at[0] - 5, home.at[1]], line: home.num };
  });
  const look = (c) => page.evaluate((c) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: 11 }); }, c);
  const read = () => page.evaluate(async () => {
    const lf = await import('/src/game/lineFigures.ts'), ec = await import('/src/game/editCap.ts'), d = await import('/src/game/demand.ts'), ld = await import('/src/game/loading.ts');
    const load = ld.loadProgress.value;
    return {
      capped: window.tw.clock.capped.peek(), updating: d.demandUpdating.value,
      // the cap's own inputs: a move in any of them may change it legitimately
      inputs: JSON.stringify([[...lf.linesPending.value], ec.quietEdits.value, ec.wideEdits.value, d.demandBehindFrom.value !== null, d.demandUpdating.value, load ? load.cities : null]),
      tag: document.querySelector('.status-tag .demand-busy')?.textContent ?? '',
    };
  });

  // the camera alone
  let camCap = false;
  for (const c of [b, a, b, a]) {
    await look(c);
    for (let k = 0; k < 16; k++) { if ((await read()).capped) camCap = true; await page.waitForTimeout(250); }
  }
  check(!camCap, 'the camera alone (to a far zone and back, twice) never caps the clock');

  /** an edit's update, the cap read from one view and then the other, back and forth */
  const flip = async (label, fn, arg) => {
    // the last case's update over first (a zone opened by the flips re-solves), the camera home
    await look(a);
    await page.waitForFunction(() => !window.__m.d.demandUpdating.value && !window.twDemand.client.waiting, undefined, { timeout: 900000, polling: 500 });
    await page.waitForTimeout(1000);
    const res = await page.evaluate(fn, arg);
    let started = false;
    for (let k = 0; k < 240 && !started; k++) { started = (await read()).updating; if (!started) await page.waitForTimeout(250); }
    check(started, `${label}: an update started (edit result ${JSON.stringify(res)?.slice(0, 120)})`);
    let same = 0, differ = 0, capped = 0, untagged = 0, here = a;
    const t0 = Date.now();
    // (60 pairs, about two minutes, are enough; the update may run on long after)
    while (started && Date.now() - t0 < 600000 && same + differ < 60) {
      const r0 = await read();
      if (!r0.updating) break;
      here = here === a ? b : a;
      await look(here);
      await page.waitForTimeout(400);
      const r1 = await read();
      if (r1.capped && !r1.tag) untagged++;
      if (r0.inputs === r1.inputs) {
        if (r0.capped === r1.capped) same++; else differ++;
        if (r0.capped) capped++;
      }
      await page.waitForTimeout(1000);
    }
    check(differ === 0 && same >= 3, `${label}: the cap read the same from both views (${same} pairs alike, ${differ} differ)`);
    check(capped > 0, `${label}: the edit capped the clock (${capped} pairs capped)`);
    check(untagged === 0, `${label}: the status tag shows whenever the clock is capped`);
  };
  await page.evaluate(async () => { window.__m = { d: await import('/src/game/demand.ts') }; });
  await flip('the fare curve doubled', async () => {
    const nodes = window.tw.moneyState.peek().fares.nodes.map(([km, f]) => [km, f * 2]);
    return (await import('/src/game/world.ts')).edit({ op: 'fares', nodes });
  });
  await flip('a stop dropped from a line', async (n) => (await import('/src/game/world.ts')).edit({ op: 'dropStop', line: n, index: 1 }), line);
  check(errors.length === 0, 'no console errors' + (errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''));
} finally {
  await chrome.close();
}
process.exitCode = fail ? 1 : 0;
