// T-2365: the city view's trips by length graph and the top bar's split hover, in headless Chrome. From the
// maps root:
//   node trainworld/checks/t2365-trip-graph.mjs [url] [--shots]
// On the real New York save (T-007), then a new game in New York:
// 1. The graph's bins add up: train, walking, driving with No route, and planes to the zone's split (its trips a
//    day times each share) plus the city view's To other places trips (train, car, plane), within 0.1%; the far
//    part to To other places exactly; every corridor has a distance.
// 2. The drawn curves keep the trips: each layer's area under its curve (trips per decade, over the axis) is its
//    bins' sum within 1%.
// 3. The panel: "Trips within New York", the graph with its five layers and axis labels, and the graph's hover
//    at 10 km names the 10-15 km band.
// 4. The top bar's hover: "Per day in New York", Trip count and Distance rows, the share table.
// --shots: the city view light and dark and the top bar's hover, in artifacts/trainworld/tripui/.
// Uses the Vite on the url (dev-nowatch.mjs); 2 demand workers, claim 4 cores. About 3 minutes.
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, importSave, launchChrome, REAL_SAVE } from './lib.mjs';

const OUT = join(ARTIFACTS, 'tripui');
const SHOTS = process.argv.includes('--shots');
if (SHOTS) await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6184/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const ZONE = 'nyc';

/** The graph's bins and curves against the zone's figures, worked out in the page from the app's own modules. */
const figures = (page) => page.evaluate(async (zone) => {
  const tl = await import('/src/game/tripLengths.ts');
  const zl = await import('/src/game/zoneLengths.ts');
  const dm = await import('/src/game/demand.ts');
  const { cityById } = await import('/src/game/cities.ts');
  const city = cityById(zone);
  const v = dm.cityDemandViews.value.get(zone);
  const local = zl.lengthsOf(zone);
  const market = dm.intercityDemand.value;
  if (!v || !local || !market) return { error: `view ${!!v}, lengths ${!!local}, market ${!!market}` };
  if (local.version !== dm.demandView.value?.version) return { error: `lengths of solve ${local.version}, view ${dm.demandView.value?.version}` };
  const far = zl.farTrips(city.place, market.corridors);
  // the city view's To other places, as CityInspector sums it
  let all = 0, rail = 0, air = 0;
  dm.eachCorridor(market.corridors, (a, b) => a !== b && (a === city.place || b === city.place), (j, f) => { all += f.trips; rail += f.rail; air += f.air; });
  const L = tl.lengthLayers(local, far);
  const sum = (a) => a.reduce((s, x) => s + x, 0);
  const bins = Object.fromEntries(tl.LAYERS.map((l) => [l, sum(L.n[l])]));
  const [u0, u1] = tl.axisSpan(L);
  const n = 150, c = tl.layerCurves(L, u0, u1, n);
  const area = Object.fromEntries(tl.LAYERS.map((l) => {
    const y = c.y[l];
    let s = 0;
    for (let k = 1; k < n; k++) s += 0.5 * (y[k] + y[k - 1]) * (c.u[k] - c.u[k - 1]);
    return [l, s];
  }));
  const t = v.tripsPerDay, s = v.split;
  return {
    want: { train: (t * s.train) / 100 + rail, walk: (t * s.walk) / 100, driveNone: (t * (s.drive + (s.none ?? 0))) / 100 + (all - rail - air), plane: air, far: all },
    bins, farBins: sum(L.far), area, noKm: far.filter((f) => !Number.isFinite(f.km)).length, corridors: far.length,
    axis: [Math.pow(10, u0), Math.pow(10, u1)], localTrips: t, kmShare: (100 * sum(local.km.rail)) / (sum(local.km.rail) + sum(local.km.walk) + sum(local.km.drive) + sum(local.km.none)),
  };
}, ZONE);

const rel = (a, b) => Math.abs(a - b) / Math.max(1, Math.abs(b));
const M = (v) => (v / 1e6).toFixed(3) + 'M';

async function run(label, page) {
  await page.evaluate((z) => { window.tw.selection.value = { kind: 'city', city: z }; }, ZONE);
  await page.waitForFunction(() => document.querySelector('svg.trip-graph'), undefined, { timeout: 120000, polling: 500 });
  // the market's corridors can land after the zone's final
  let f = await figures(page);
  for (let k = 0; k < 120 && (f.error || f.corridors === 0); k++) { await page.waitForTimeout(500); f = await figures(page); }
  if (f.error) return check(false, `${label}: the graph's figures`, f.error);
  console.log(`  ${label}: ${M(f.localTrips)} trips within, ${M(f.want.far)} to other places over ${f.corridors} corridors; axis ${f.axis[0].toFixed(3)}-${Math.round(f.axis[1])} km; train share of km ${f.kmShare.toFixed(1)}%`);
  const pairs = [['train', f.bins.train, f.want.train], ['walking', f.bins.walk, f.want.walk], ['driving and No route', f.bins.drive + f.bins.none, f.want.driveNone], ['planes', f.bins.plane, f.want.plane], ['to other places', f.farBins, f.want.far]];
  const worst = Math.max(...pairs.map(([, a, b]) => rel(a, b)));
  check(worst < 1e-3 && f.noKm === 0, `${label}: the graph's bins add up to the zone's split and To other places`, pairs.map(([n, a, b]) => `${n} ${M(a)} of ${M(b)}`).join(', ') + `; ${f.noKm} corridors with no distance`);
  const worstArea = Math.max(...Object.keys(f.bins).map((l) => (f.bins[l] > 0 ? rel(f.area[l], f.bins[l]) : 0)));
  check(worstArea < 0.01, `${label}: each layer's drawn area is its trips`, Object.keys(f.bins).map((l) => `${l} ${M(f.area[l])} of ${M(f.bins[l])}`).join(', '));

  const panel = await page.evaluate(() => {
    const g = document.querySelector('svg.trip-graph');
    return { text: document.querySelector('#inspector')?.innerText ?? '', areas: g.querySelectorAll('.tg-area').length, labels: [...g.querySelectorAll('.tg-label')].map((t) => t.textContent) };
  });
  check(/Trips within New York/.test(panel.text) && /By trip length/.test(panel.text) && panel.areas >= 5 && panel.labels.length >= 4,
    `${label}: the city view says "Trips within New York" and draws the graph`, `${panel.areas} areas, labels ${panel.labels.join(' ')}`);
  // the graph's hover at 10 km
  const box = await page.evaluate(() => { document.querySelector('svg.trip-graph').scrollIntoView({ block: 'center' }); return document.querySelector('.tg-hover').getBoundingClientRect().toJSON(); });
  const [u0, u1] = [Math.log10(f.axis[0]), Math.log10(f.axis[1])];
  const at = box.left + ((1 - u0) / (u1 - u0)) * box.width + 2;
  await page.mouse.move(at - 20, box.top + box.height / 2);
  await page.mouse.move(at, box.top + box.height / 2);
  await page.waitForTimeout(400);
  const tag = await page.evaluate(() => [...document.querySelectorAll('.ui-tag')].find((t) => t.style.display !== 'none')?.innerText ?? '');
  check(/Trips of 10-15 km per day/.test(tag) && /Train/.test(tag) && /All/.test(tag), `${label}: the graph's hover at 10 km`, tag.replace(/\n/g, ' | '));
  if (SHOTS) await page.screenshot({ path: join(OUT, `${label}-graph-hover.png`), clip: { x: Math.max(0, box.left - 40), y: Math.max(0, box.top - 260), width: 520, height: 520 } });

  // the top bar's hover
  const split = await page.evaluate(() => document.querySelector('.split-row').getBoundingClientRect().toJSON());
  await page.mouse.move(split.left + 10, split.top + split.height / 2);
  await page.mouse.move(split.left + 20, split.top + split.height / 2);
  await page.waitForFunction(() => /Distance/.test([...document.querySelectorAll('.ui-tag')].find((t) => t.style.display !== 'none')?.innerText ?? ''), undefined, { timeout: 30000, polling: 250 }).catch(() => {});
  const top = await page.evaluate(() => [...document.querySelectorAll('.ui-tag')].find((t) => t.style.display !== 'none')?.innerText ?? '');
  check(/Per day in New York/.test(top) && /Trip count/.test(top) && /Distance/.test(top) && /Walking/.test(top) && /km/.test(top), `${label}: the top bar's hover`, top.replace(/\n/g, ' | '));
  // T-2443: every figure sits inside the tag's box
  const over = await page.evaluate(() => {
    const tag = [...document.querySelectorAll('.ui-tag')].find((t) => t.style.display !== 'none');
    if (!tag) return ['no tag'];
    const box = tag.getBoundingClientRect();
    return [...tag.querySelectorAll('.tip-grid > div')].filter((c) => c.getBoundingClientRect().right > box.right + 0.5).map((c) => c.textContent);
  });
  check(over.length === 0, `${label}: the top bar's figures sit inside its tooltip`, over.join(', '));
  if (SHOTS) await page.screenshot({ path: join(OUT, `${label}-topbar-hover.png`), clip: { x: Math.max(0, split.left - 60), y: 0, width: 720, height: 260 } });
  await page.mouse.move(5, 600);

  if (SHOTS) {
    for (const shade of ['light', 'dark']) {
      await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
      await page.waitForTimeout(800);
      const dock = await page.evaluate(() => { const i = document.querySelector('#inspector'); i.querySelector('svg.trip-graph').scrollIntoView({ block: 'center' }); return i.getBoundingClientRect().toJSON(); });
      await page.screenshot({ path: join(OUT, `${label}-city-${shade}.png`), clip: { x: dock.left, y: dock.top, width: dock.width, height: dock.height } });
    }
    await page.evaluate(() => { window.tw.shade.value = 'light'; });
  }
}

const chrome = await launchChrome({ prefix: 'tw-t2365-', size: [1400, 1000], deviceScaleFactor: SHOTS ? 2 : 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 2 })); } catch {} });
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&city=${ZONE}&demandWorkers=2&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
  await demandFinal(page, { strict: false });
  await run('new-game', page);
  await page.evaluate(() => { window.tw.selection.value = null; });
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await demandFinal(page, { lines: 100, timeout: 600000 });
  await run('real-nyc', page);
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2365 FAILED: ${fail}` : 'T-2365 PASSED');
process.exit(fail ? 1 : 0);
