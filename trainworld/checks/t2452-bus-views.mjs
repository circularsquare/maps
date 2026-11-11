// T-2452: the Line, Traffic, Speed and Profit track displays on bus and ferry lines. Loads T-2195's
// real-ferries.save (artifacts/trainworld/t2195/, five NYC ferry lines) and adds three bus lines, waits for
// demand, then: each bus and ferry line's stretches in the Profit view sum to its weekday fares and its
// upkeep share; the hover on a bus and a ferry stretch in each display; pictures of each display, light and
// dark, in artifacts/trainworld/busviews/. From the maps root, on a Vite of your own:
//   node trainworld/checks/t2452-bus-views.mjs [dev url]
// About 3-4 minutes on 2 demand workers (`DEMAND_WORKERS`); claim 3 cores.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 'busviews');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6189/';
const WORKERS = Number(process.env.DEMAND_WORKERS ?? 2);
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// clicked points of three bus lines near the East River ferry, each leg straight
const BUSES = [
  [[-73.9905, 40.7145], [-73.9840, 40.7175], [-73.9765, 40.7200]],
  [[-73.9600, 40.7140], [-73.9520, 40.7070], [-73.9450, 40.7000]],
  [[-73.9560, 40.7440], [-73.9450, 40.7470], [-73.9350, 40.7500]],
];
const VIEW = { center: [-73.968, 40.728], zoom: 13.2 };

const chrome = await launchChrome({ prefix: 'tw-t2452-', size: [1300, 850], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&demandWorkers=${WORKERS}&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, await readFile(join(ARTIFACTS, 't2195', 'real-ferries.save')));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  const added = await page.evaluate(async (BUSES) => {
    const tw = window.tw, c = tw.client, out = [];
    for (const [i, pts] of BUSES.entries()) {
      const route = pts.flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
      const add = await c.edit({ op: 'addBusLine', route, name: '', colour: ['#d7263d', '#7b3fa0', '#1f8a70'][i] });
      const con = add.ok ? await c.edit({ op: 'construct', edges: [], stations: [], buses: [add.line] }) : null;
      out.push({ line: add.line, ok: !!add.ok && !!con?.ok });
    }
    return out;
  }, BUSES);
  check(added.every((a) => a.ok), 'three bus lines added and constructed', JSON.stringify(added));
  await wait(1500);
  await demandFinal(page, { timeout: 1200000, polling: 1000, lines: 100 });
  await page.evaluate(() => { const tw = window.tw; tw.display.value = { ...tw.display.value, trains: false, dayShown: 'weekday' }; });

  // ---- the Profit view's money: each bus and ferry line's stretches sum to its fares and upkeep share
  const sums = await page.evaluate(async () => {
    const tw = window.tw;
    tw.display.value = { ...tw.display.value, trackColour: 'profit' };
    const { stretchMoney } = await import('/src/game/stretchMoney.ts');
    const { incomeByDay } = await import('/src/game/money.ts');
    const { PERIODS } = await import('/src/game/periods.ts');
    const sm = stretchMoney.value, w = tw.world.value, inc = incomeByDay.value[0];
    if (!sm) return null;
    const out = [];
    for (const l of w.save.lines) {
      const st = w.lineStats[l.id];
      if (!st?.bus || st.status !== 'running') continue;
      const b = sm.buses.get(l.num);
      const li = inc.lines.get(l.id);
      const fares = li ? li.perHour.reduce((a, v, k) => a + v * (PERIODS[k].to - PERIODS[k].from), 0) * 1e6 : 0;
      const got = (b?.stretches ?? []).reduce((a, s) => ({ fares: a.fares + s.fares, upkeep: a.upkeep + s.upkeep, running: a.running + s.running }), { fares: 0, upkeep: 0, running: 0 });
      out.push({ name: l.name, ferry: !!st.bus.ferry, stretches: b?.stretches.length ?? 0, stops: l.stops.length, fares, got, upkeep: st.upkeep, ratio: Array.from(b?.ratio ?? []).map((v) => (v > 1e29 ? null : +v.toFixed(2))) });
    }
    return out;
  });
  check(!!sums && sums.length >= 8, 'bus and ferry lines placed in the Profit view', JSON.stringify(sums?.map((s) => `${s.name} ${s.stretches}/${s.stops - 1}`)));
  for (const s of sums ?? []) {
    check(s.stretches === s.stops - 1 && Math.abs(s.got.fares - s.fares) <= 1e-6 * Math.max(1, s.fares) && Math.abs(s.got.upkeep - s.upkeep) < 1,
      `${s.name}: stretches sum to its weekday fares and upkeep share`, `fares ${s.got.fares.toFixed(0)}/${s.fares.toFixed(0)}, upkeep ${s.got.upkeep.toFixed(2)}/${s.upkeep.toFixed(2)}, running ${s.got.running.toFixed(0)}, ratios ${s.ratio.join(' ')}`);
  }

  // ---- each display: a picture and the hover on a bus and a ferry stretch
  const spots = await page.evaluate(() => {
    const w = window.tw.world.value, out = [];
    for (const l of w.save.lines) {
      const b = w.lineStats[l.id]?.bus;
      if (!b || !b.built) continue;
      // the middles of route segments, from the route's middle outwards
      const n = b.routeLL.length / 2 - 1, at = [];
      for (let d = 0; d < n; d++) for (const i of [Math.floor(n / 2) + d, Math.floor(n / 2) - d - 1]) {
        if (i >= 0 && i < n && at.length < 16) at.push([(b.routeLL[2 * i] + b.routeLL[2 * i + 2]) / 2, (b.routeLL[2 * i + 1] + b.routeLL[2 * i + 3]) / 2]);
      }
      out.push({ name: l.name, ferry: !!b.ferry, at });
    }
    return out;
  });
  // Bus 110 lies under rail track where the pick takes the track first: try every bus line's points
  const busSpots = spots.filter((s) => !s.ferry), bus = busSpots.length ? { name: 'a bus line', ferry: false, at: busSpots.flatMap((s) => s.at) } : undefined, ferry = spots.find((s) => s.ferry);
  for (const view of ['line', 'traffic', 'speed', 'profit', 'height']) {
    await page.evaluate((v) => { const tw = window.tw; tw.display.value = { ...tw.display.value, trackColour: v }; }, view);
    for (const spot of [bus, ferry]) {
      if (!spot) continue;
      const text = await page.evaluate(async ([ats]) => {
        const tw = window.tw;
        const { trackTip } = await import('/src/map/trackHover.ts');
        for (const at of ats) {
          tw.map.jumpTo({ center: at, zoom: 13.5 });
          await new Promise((r) => setTimeout(r, 700));
          const p = tw.map.project(at);
          const rows = trackTip(tw.overlay, p.x, p.y, 13.5);
          const t = (rows ?? []).map((x) => (typeof x === 'string' ? x : x.textContent)).join(' | ');
          if (/(Bus|Ferry) line/.test(t)) return t;
        }
        return null;
      }, [spot.at]);
      const want = view === 'speed' ? new RegExp(`Fastest ${spot.ferry ? 'boat' : 'bus'} \\d+ km/h`) : view === 'height' ? /^(?!.*(riders per|Fastest|Fares cover))/ : view === 'profit' ? new RegExp(`Fares cover [\\d,]+% of costs.*Running ${spot.ferry ? 'boats' : 'buses'} −\\$`) : /riders per weekday/;
      check(!!text && want.test(text), `${view}: the hover on ${spot.name}`, JSON.stringify(text));
    }
    for (const shade of view === 'traffic' || view === 'profit' || view === 'speed' ? ['light', 'dark'] : ['light']) {
      await page.evaluate(([s, VIEW]) => { window.tw.shade.value = s; window.tw.map.jumpTo(VIEW); }, [shade, VIEW]);
      await wait(2500);
      await page.screenshot({ path: join(OUT, `${view}-${shade}.png`) });
    }
    await page.evaluate(() => { window.tw.shade.value = 'light'; });
  }
} catch (e) {
  check(false, 'ran to the end', e.message);
} finally {
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
  await chrome.close();
}
console.log(fail ? 'T-2452 FAILED' : 'T-2452 PASSED');
process.exit(fail ? 1 : 0);
