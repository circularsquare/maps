// T-1177, T-1229: the everyday outings across a zone line in the page. A new game in Boston with a
// line to Providence; the same made-up pairs as t1177-cross-everyday.mjs written into Boston's
// market pack on its way to the page (Playwright route; nothing under data/ is touched). Checked on
// the page's own client: the selection of every Boston cell holds the pairs by purpose with walkers,
// the trip explorer between the places has the everyday section with a walking row, and a
// Saturday's market trips are not a weekday's. A screenshot of each panel goes to
// artifacts/trainworld/t1177/.
//   BASE=http://localhost:6154/ node trainworld/checks/t1177-cross-everyday-page.mjs
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, buildLine, demandFinal } from './lib.mjs';
import { KINDS, TRIPS, pairData } from './t1177-data.mjs';
const BASE = process.env.BASE ?? 'http://localhost:6154/';
const OUT = join(ARTIFACTS, 't1177');
mkdirSync(OUT, { recursive: true });
const { homeMarket, awayMarket, lngLat, withPairs } = pairData('bos', 'pvd');
const { ctx, close } = await launchChrome({ prefix: 't1177-' });
const errors = [];
const out = {};
try {
  await ctx.route(/\/bos\.market\.json/, async (route) => {
    const r = await route.fetch();
    await route.fulfill({ response: r, body: JSON.stringify(withPairs()), headers: { ...r.headers(), 'content-type': 'application/json' } });
  });
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 10.5 }), lngLat[0]);
  await page.waitForFunction(() => window.twDemand.activeCity.peek() === 'bos', undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const line = await buildLine(page, lngLat, { name: 'Cross', wait: { route: 600, station: 300, settle: 800 } });
  assert(line?.ok !== false && line?.line !== undefined, `line built: ${JSON.stringify(line)}`);
  await demandFinal(page, { timeout: 280000, polling: 250, lines: 0 });
  await page.waitForTimeout(1500);
  await demandFinal(page, { timeout: 280000, polling: 250, lines: 0 });

  // trains by level: the 60 km line takes most of a new game's cash, a few trains are what is left
  const trains = await page.evaluate(async (n) => { const r = await window.tw.client.edit({ op: 'setTrainCount', line: n, trains: [3, 2, 2] }); return { ok: r.ok, issues: r.issues, tph: window.tw.world.value.save.lines[0].tph }; }, line.line);
  assert(trains.ok, `trains set: ${JSON.stringify(trains)}`);
  await page.waitForTimeout(1500);
  await demandFinal(page, { timeout: 280000, polling: 250, lines: 0 });
  out.solve = await page.evaluate(() => { const d = window.twDemand, s = d.log.at(-1), v = d.view(), ic = d.intercity(); return { cities: s.cities, markets: s.markets ?? s.market, clusters: s.clusters, lines: v.lines.size, riders: [...v.lines.values()].map((l) => Math.round(l.ridersPerDay)), stations: window.tw.world.value.save.lines.map((l) => l.stops?.length), longer: ic && { trips: Math.round(ic.tripsPerDay), rail: Math.round(ic.railTripsPerDay) } }; });
  console.log('solve', JSON.stringify(out.solve));
  console.log('line', JSON.stringify(await page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines[0]; return { line: { ...l, stops: l.stops?.length }, nodes: w.nodes.size, edges: w.save.edges?.length, tracks: w.save.tracks?.length, keys: Object.keys(w.save) }; })).slice(0, 1500));
  // the selection of every Boston cell as homes, on the client's own reading
  out.selection = await page.evaluate(async ({ away }) => {
    const d = window.twDemand, cells = await d.cityCells();
    const sel = await d.intercitySelection('home', Uint32Array.from({ length: cells.n }, (_, i) => i));
    if (!sel) return null;
    const by = {};
    for (const [k, v] of sel.byPurpose ?? []) by[k] = { all: Math.round(v.starts.all), rail: Math.round(v.starts.rail), walk: Math.round(v.starts.walk ?? 0), places: v.starts.places.map((p) => p.place), zones: v.starts.zones.length };
    return { all: Math.round(sel.starts.all), walk: Math.round(sel.starts.walk ?? 0), places: sel.starts.places.slice(0, 6).map((p) => [p.place, Math.round(p.all), Math.round(p.walk ?? 0)]), by, away, n: cells.n };
  }, { away: awayMarket.place });
  console.log('selection', JSON.stringify(out.selection));
  const s = out.selection;
  assert(s, 'selection answered');
  assert.deepEqual(Object.keys(s.by).sort(), ['school', 'shopping', 'social']);
  // the page shows the average day (a week's: five weekdays, a Saturday and a Sunday) by default:
  // each purpose's own weekend factors (sim days.rs DAY_RATE), so 1.063, 1.173 and 0.739 of a weekday's
  const AVERAGE = [(5 + 1.39 + 1.05) / 7, (5 + 1.54 + 1.67) / 7, (5 + 0.11 + 0.06) / 7];
  KINDS.forEach((k, i) => assert(Math.abs(s.by[['shopping', 'social', 'school'][i]].all - 2 * TRIPS[i] * AVERAGE[i]) < 0.01 * 2 * TRIPS[i], `legs by purpose, ${k}: ${s.by[['shopping', 'social', 'school'][i]].all}`));
  assert(s.by.shopping.walk > 0 && s.by.shopping.rail > 0, 'shopping walks and rides');
  assert.equal(s.by.school.walk, 0);
  assert(s.by.shopping.places.includes(s.away) && s.by.shopping.zones === 1);
  assert(s.walk > 0 && s.places.some(([p, all]) => p === s.away && all >= 2 * sumOf(TRIPS)), 'the all-purposes list holds them');

  // the trip explorer between the places, on the client's own reading
  out.explorer = await page.evaluate(async ({ near, far }) => {
    const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), far, -1, { place: near, gid: -1 });
    return ps && ps.map((p) => p && [Math.round(p.everyday.legs), Math.round(p.everyday.rail), Math.round(p.everyday.walk)]);
  }, { near: homeMarket.place, far: awayMarket.place });
  console.log('explorer', JSON.stringify(out.explorer));
  assert(out.explorer && out.explorer.every((p) => p), 'explorer answered');
  const e = out.explorer.reduce((a, p) => a.map((v, i) => v + p[i]), [0, 0, 0]);
  assert(Math.abs(e[0] - sumOf(TRIPS.map((t, i) => 2 * t * AVERAGE[i]))) < 10 && e[1] > 0 && e[2] > 0, `explorer totals ${e}`);

  // the panel
  const [fx, fy] = await page.evaluate(([a, b]) => { const f = window.tw.lngLatToPack; const p = f(a[0], a[1]), q = f(b[0], b[1]); return [[p.x, p.y], [q.x, q.y]]; }, [lngLat[0], lngLat[2]]);
  await page.evaluate(({ fx, fy, near, far }) => window.tw.select({ kind: 'trip', from: fx, to: fy, far: { place: far, gid: -1 }, near: { place: near, gid: -1 } }), { fx, fy, near: homeMarket.place, far: awayMarket.place });
  await page.waitForFunction(() => document.body.innerText.includes('Everyday trips'), undefined, { timeout: 60000 });
  out.panel = await page.evaluate(() => {
    const h = [...document.querySelectorAll('.h')].find((el) => el.textContent.startsWith('Everyday trips'));
    const box = h?.nextElementSibling;
    return { head: h?.textContent, rows: [...(box?.querySelectorAll('li') ?? [])].map((li) => li.textContent.replace(/\s+/g, ' ').trim()) };
  });
  console.log('panel', JSON.stringify(out.panel));
  assert(out.panel.rows.some((r) => r.startsWith('Walking')) && out.panel.rows.some((r) => r.startsWith('Train')) && out.panel.rows.some((r) => r.startsWith('Car')), 'the explorer has a walking row beside train and car');
  await page.screenshot({ path: join(OUT, 'explorer.png') });

  // a bubble selection's inspector: the walking column and the pairs in the places list
  await page.evaluate(async () => {
    const cells = await window.twDemand.cityCells();
    window.tw.display.value = { ...window.tw.display.value, commuters: true, commuterEnd: 'home' };
    window.tw.select({ kind: 'commuters', end: 'home', cells: Array.from({ length: cells.n }, (_, i) => i), areas: 1 });
  });
  await page.waitForFunction(() => [...document.querySelectorAll('[data-tip]')].some((el) => el.getAttribute('data-tip').includes('on foot')), undefined, { timeout: 60000 });
  await page.screenshot({ path: join(OUT, 'selection.png') });
  out.selectionPanel = await page.evaluate(() => [...document.querySelectorAll('.rows li')].map((li) => li.textContent.replace(/\s+/g, ' ').trim()).filter((t) => /^(To|From|Leave|Arrive)/.test(t)).slice(0, 8));
  console.log('selection panel', JSON.stringify(out.selectionPanel));
  // the map's far end under the purpose filter: the pairs of that purpose alone (one far place),
  // none under Work, the longer trips and the rest under all purposes
  // (zoomed out to places, each far place is a labelled bubble: "Providence · 6k trips")
  await page.evaluate(([a, b]) => void window.tw.map.jumpTo({ center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], zoom: 9 }), [lngLat[0], lngLat[2]]);
  const far = async (value) => {
    await page.selectOption('select[aria-label="Trip purpose on the map"]', value);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `far-${value}.png`) });
    return page.evaluate(() => [...document.querySelectorAll('body *')].filter((el) => !el.children.length && / · [\d.,]+[kKmM]? trips$/.test(el.textContent)).map((el) => el.textContent));
  };
  out.farLabels = { all: await far('all'), shopping: await far('shopping'), school: await far('school'), work: await far('work'), medical: await far('medical') };
  console.log('far labels', JSON.stringify(out.farLabels));
  assert(out.farLabels.shopping.length === 1 && out.farLabels.shopping[0].startsWith('Providence') && out.farLabels.school.length === 1, 'a purpose with pairs draws their far place');
  assert(out.farLabels.work.length === 0 && out.farLabels.medical.length === 0, 'none for a purpose without');
  assert(out.farLabels.all.length > 1, 'all purposes draw the longer trips too');
  await page.selectOption('select[aria-label="Trip purpose on the map"]', 'shopping');
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(OUT, 'shopping-far-end.png') });

  // a Saturday: the market's trips take each purpose's factor, not the weekday's
  out.days = await page.evaluate(() => window.twDemand.days().map((d) => Math.round(d?.intercity?.tripsPerDay ?? NaN)));
  console.log('market trips by day', JSON.stringify(out.days));
  assert(out.days[1] > 0 && out.days[1] !== out.days[0]);
} finally {
  await close();
}
assert.deepEqual(errors, [], 'no console errors');
writeFileSync(join(OUT, 't1177-cross-everyday-page.json'), JSON.stringify(out, null, 1));
console.log('PASS t1177-cross-everyday-page');
function sumOf(xs) { return xs.reduce((s, v) => s + v, 0); }
