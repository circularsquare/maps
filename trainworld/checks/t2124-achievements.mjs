// T-2124: achievements. Loads the real New York save and the Tokyo save (each into its own ?new=1 page
// on a dev server, one demand worker), waits for the quiet working-out an older save gets, and prints
// the steps earned and the bars' figures; then a new game: a first short line in Manhattan earns
// nothing, and a second line into another region earns "Cities served" 2 with a toast. Screenshots of
// the panel (light, dark) and the toast go to artifacts/trainworld/t2124/.
//   node trainworld/checks/t2124-achievements.mjs [base url, default http://localhost:6170/] [--only=nyc|tokyo|new]
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, acrossLine, buildLine, collectErrors, demandFinal, importSave, launchChrome } from './lib.mjs';

const BASE = process.argv.find((a) => a.startsWith('http')) ?? 'http://localhost:6170/';
const ONLY = process.argv.find((a) => a.startsWith('--only='))?.slice(7);
const TOKYO = 'C:/Users/anita/Documents/trainworld/Tokyo Seoul Fukuoka Busan (from anita.garden).save';
const OUT = join(ARTIFACTS, 't2124');
// what each save is expected to have earned (notes/T-2124.md section 1, with Anita's tiers)
const EXPECT = {
  nyc: ['cities:2', 'trips:1000000', 'trips:10000000', 'station:500000', 'station:1000000', 'line:500000', 'line:1000000'],
  tokyo: ['cities:2', 'cities:10', 'trips:1000000', 'trips:10000000', 'station:500000', 'line:500000', 'line:1000000', 'border', 'long', 'money', 'bullet'],
  // (Tokyo's 19 cities include stray stations in Spain and the US; Change here 0.73; no city at 20% of
  // its own trips or its trips to other places by train; the Shinkansen's run is 5.4 h)
};
let failed = 0;
const say = (ok, what) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); };

async function page(ctx, shade, extra = '') {
  const p = await ctx.newPage();
  await p.addInitScript((s) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: s, display: { dayShown: 'average' }, v: 2 })); } catch {} }, shade);
  await p.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1${extra}&v=${Date.now()}`);
  await p.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.moneyState.value, undefined, { timeout: 180000 });
  return p;
}

/** the steps earned, the bars' figures and what is behind the firsts */
const readout = (p) => p.evaluate(async () => {
  const a = await import('/src/game/achievements.ts');
  const m = window.tw.moneyState.value;
  const nf = a.networkFigures.value, df = a.demandFigures.value, met = a.metNow.value;
  return {
    earned: Object.keys(m.achievements ?? {}).sort(), achievements: m.achievements ?? null,
    figures: { cities: nf?.cities, trips: df?.trips, station: df?.station, line: df?.line, profit: a.bestProfit.value },
    metNow: Object.keys(met.met).sort(), ready: met.ready,
    firstsNow: { ...Object.fromEntries(nf?.firsts ?? []), ...Object.fromEntries(df?.firsts ?? []) },
  };
});

/** the diagnostics behind the share and long-ride firsts, to set their floors */
const diag = (p) => p.evaluate(async () => {
  const { demandByDay, eachCorridor, roughZones } = await import('/src/game/demand.ts');
  const { cityById } = await import('/src/game/cities.ts');
  const { stationZone } = await import('/src/game/listScope.ts');
  const w = window.tw.world.value, d = demandByDay.value[6];
  const built = new Set(w.save.stations.filter((s) => s.built).map(stationZone).filter(Boolean));
  const places = new Map([...built].map((id) => [cityById(id).place, id]));
  const sums = new Map();
  const c = d.intercity?.corridors;
  if (c) eachCorridor(c, (x, y) => x !== y && (places.has(x) || places.has(y)), (j, v) => {
    for (const p of [c.table.from[j], c.table.to[j]]) if (places.has(p)) { const e = sums.get(p) ?? { all: 0, rail: 0 }; e.all += v.trips; e.rail += v.rail; sums.set(p, e); }
  });
  const inter = [...sums].map(([p, e]) => [cityById(places.get(p)).name, Math.round(e.all), +(100 * e.rail / e.all).toFixed(1)]).sort((x, y) => y[2] - x[2]).slice(0, 8);
  const local = [...built].map((id) => { const v = d.cities.get(id); return [cityById(id).name, v ? +v.split.train.toFixed(1) : null, v?.final, roughZones.value.has(id)]; }).sort((x, y) => (y[1] ?? -1) - (x[1] ?? -1)).slice(0, 6);
  // the longest one-way run of each running line, h
  const runs = w.save.lines.map((l) => [l.name, +((w.lineStats[l.id]?.fromStartS.at(-1) ?? 0) / 3600).toFixed(2)]).sort((x, y) => y[1] - x[1]).slice(0, 4);
  let tr = 0; for (const s of d.view.stations.values()) tr += s.transfers ?? 0;
  return { inter, local, runs, changesPerTrip: +(tr / d.view.railTripsPerDay).toFixed(3), railTrips: Math.round(d.view.railTripsPerDay) };
});

async function loadSave(ctx, name, path, shade) {
  const errors = [];
  const p = await page(ctx, shade);
  collectErrors(p, errors);
  const t0 = Date.now();
  // when each step lands, and the demand result in force then
  await p.evaluate(async () => {
    const { demandByDay, demandUpdating } = await import('/src/game/demand.ts');
    const log = (window.achLog = []);
    let n = -1;
    window.tw.moneyState.subscribe((m) => {
      const k = m?.achievements ? Object.keys(m.achievements).length : -1;
      if (k === n) return;
      n = k;
      const v = demandByDay.peek()[6].view;
      log.push({ s: Math.round(performance.now() / 1000), steps: k, keys: Object.keys(m?.achievements ?? {}).join(' '), version: v?.version, final: v?.final, updating: demandUpdating.peek() });
    });
  });
  await importSave(p, await readFile(path));
  await p.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 180000 });
  await demandFinal(p, { timeout: 1800000, polling: 2000, lines: 0 });
  await p.waitForFunction(() => !!window.tw.moneyState.value?.achievements, undefined, { timeout: 120000 });
  await p.waitForTimeout(2500);
  const r = await readout(p);
  console.log(`${name}: loaded and worked out in ${Math.round((Date.now() - t0) / 1000)} s`);
  console.log(`${name}: earned ${r.earned.length}: ${r.earned.join(' ')}`);
  console.log(`${name}: bars ${JSON.stringify(r.figures)}`);
  console.log(`${name}: firsts met ${JSON.stringify(r.firstsNow)}`);
  console.log(`${name}: diagnostics ${JSON.stringify(await diag(p))}`);
  console.log(`${name}: steps as they landed ${JSON.stringify(await p.evaluate(() => window.achLog))}`);
  const exp = EXPECT[name];
  say(JSON.stringify(r.earned) === JSON.stringify([...exp].sort()), `${name} earns ${exp.length} steps as expected${JSON.stringify(r.earned) === JSON.stringify([...exp].sort()) ? '' : ` (expected ${[...exp].sort().join(' ')})`}`);
  const toasts = await p.evaluate(() => [...document.querySelectorAll('.ach-toast')].map((t) => t.textContent));
  say(toasts.length === 0, `${name} worked out quietly, no toast${toasts.length ? ': ' + toasts.join(' | ') : ''}`);
  say(errors.length === 0, `${name} no errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
  return p;
}

async function shots(p, tag) {
  await p.evaluate(async () => { const { openAchievements } = await import('/src/ui/Achievements.tsx'); openAchievements(); });
  await p.waitForTimeout(800);
  const dock = await p.$('#dock');
  await dock.screenshot({ path: join(OUT, `panel-${tag}.png`) });
  console.log(`shot ${join(OUT, `panel-${tag}.png`)}`);
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t2124b-' });
try {
  if (!ONLY || ONLY === 'nyc') {
    const p = await loadSave(ctx, 'nyc', REAL_SAVE, 'light');
    await shots(p, 'nyc-light');
    await p.close();
  }
  if ((!ONLY || ONLY === 'tokyo') && existsSync(TOKYO)) {
    const p = await loadSave(ctx, 'tokyo', TOKYO, 'light');
    await shots(p, 'tokyo-light');
    await p.evaluate(() => { window.tw.shade.value = 'dark'; });
    await p.waitForTimeout(500);
    await shots(p, 'tokyo-dark');
    // a tick's hover
    const tick = await p.$('.ach-tick.reached');
    if (tick) { await tick.hover(); await p.waitForTimeout(600); await (await p.$('#dock')).screenshot({ path: join(OUT, 'panel-tokyo-tick-hover.png') }); }
    await p.close();
  }
  if (!ONLY || ONLY === 'new') {
    const errors = [];
    const p = await page(ctx, 'light');
    collectErrors(p, errors);
    const first = await acrossLine(p, [-73.985, 40.758], 2);
    await demandFinal(p, { timeout: 600000, polling: 1000, lines: 0 });
    await p.waitForTimeout(3000);
    let r = await readout(p);
    console.log(`new game, first line ${first}: earned ${JSON.stringify(r.earned)}; bars ${JSON.stringify(r.figures)}`);
    say(first !== null && r.earned.length === 0 && r.achievements !== null, 'a new game earns nothing on its first short line');
    // a second line from Bridgeport (New York's region) into New Haven's
    const zones = await p.evaluate(async () => { const { stationZone } = await import('/src/game/listScope.ts'); return window.tw.world.value.save.stations.map(stationZone); });
    const second = await buildLine(p, [[-73.25, 41.17], [-73.12, 41.18], [-73.0, 41.2]], { name: 'Across', trains: [6, 4, 3] });
    await p.waitForFunction(() => !!document.querySelector('.ach-toast'), undefined, { timeout: 60000 }).catch(() => {});
    await p.waitForTimeout(300);
    const toast = await p.evaluate(() => [...document.querySelectorAll('.ach-toast')].map((t) => t.textContent));
    const zones2 = await p.evaluate(async () => { const { stationZone } = await import('/src/game/listScope.ts'); return [...new Set(window.tw.world.value.save.stations.map(stationZone))]; });
    r = await readout(p);
    console.log(`new game, second line ${JSON.stringify(second?.ok)}: zones ${JSON.stringify(zones)} -> ${JSON.stringify(zones2)}; earned ${JSON.stringify(r.earned)}; toasts ${JSON.stringify(toast)}`);
    say(r.earned.includes('cities:2') && toast.some((t) => /Cities served/.test(t)), 'a second line into another region earns Cities served 2 with a toast');
    const map = await p.$('#main');
    if (map) {
      const box = await map.boundingBox();
      await p.screenshot({ path: join(OUT, 'toast-light.png'), clip: { x: box.x, y: box.y + box.height - 160, width: 420, height: 160 } });
      await p.evaluate(() => { window.tw.shade.value = 'dark'; });
      await p.waitForTimeout(300);
      await p.screenshot({ path: join(OUT, 'toast-dark.png'), clip: { x: box.x, y: box.y + box.height - 160, width: 420, height: 160 } });
      console.log(`shot ${join(OUT, 'toast-light.png')}`);
    }
    // gone after about 5.6 s
    await p.waitForTimeout(6500);
    say(await p.evaluate(() => !document.querySelector('.ach-toast')), 'the toast fades after about 5 s');
    say(errors.length === 0, `new game no errors${errors.length ? ': ' + errors.slice(0, 3).join(' | ') : ''}`);
    await p.close();
  }
} finally {
  await close();
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exitCode = failed ? 1 : 0;
