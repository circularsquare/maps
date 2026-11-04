// T-2215: the grey "No route" (trips between two land parts no road joins, not by rail; T-2214's outputs) on the map
// and in the panels, in headless Chrome. From the maps root:
//   node trainworld/checks/t2215-no-route.mjs [url]
// For Salamis (Athens, `ath`) and Cheung Chau (Hong Kong, `hon`), a new game in each, Homes bubbles on:
// 1. The zone's mode split adds up to 100% with a No route share, and the bubbles' legs per cell (homes and
//    destinations, rail + walk + drive + no route) add up to the zone's trips (0.1%).
// 2. A bubble's hover over the island: its count is its bar's four parts, a grey No route part among them.
// 3. The city view's "Within" bar has a No route part.
// 4. The trip explorer from the island to the shore across: a No route row, "No road between these two places"
//    under By car and By walking, no car rows.
// 5. A ferry from the island to the shore built (the Ferry tool's water router): the No route share falls and the
//    explorer's train row has trips.
// Screenshots, light and dark, without and with the ferry, in artifacts/trainworld/t2215/. Uses the Vite on the
// url (dev-nowatch.mjs, port 6196 by default); 2 demand workers, claim 3 cores. About 5 minutes.
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const OUT = join(ARTIFACTS, 't2215');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6196/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const t0 = Date.now();
const log = (s) => console.log(`${((Date.now() - t0) / 1000).toFixed(0).padStart(5)} s  ${s}`);

/** Each island: its zone, a point on it (the bubbles' hover and the explorer's start), a point on the shore across
 * within 10 km (the explorer's end, so By walking shows), the ferry's two landings, the view shown and a point on the
 * island in it (the hover). */
const ISLANDS = [
  { name: 'Salamis', zone: 'ath', at: [23.4905, 37.9640], across: [23.5650, 37.9650], ferry: [[23.5268, 37.9589], [23.5600, 37.9628]], view: [23.5400, 37.9580], hover: [23.5220, 37.9600], zoom: 13.4 },
  { name: 'Cheung Chau', zone: 'hon', at: [114.0275, 22.2085], across: [114.0005, 22.2655], ferry: [[114.0290, 22.2105], [113.9990, 22.2640]], view: [114.0135, 22.2365], hover: [114.0275, 22.2085], zoom: 12.7 },
];

/** game/format.ts `count`: 312000 -> "312k", 1.84e6 -> "1.84M" */
const count = (n) => (n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 1 : 2).replace(/\.?0+$/, '') + 'M' : n >= 1e3 ? Math.round(n / 1000) + 'k' : String(Math.round(n)));

async function zoneFigures(page, zone) {
  return page.evaluate(async (zone) => {
    const v = window.twDemand.cities().get(zone);
    const m = await window.twDemand.cellModes();
    const sum = (a) => (a ? a.reduce((s, x) => s + x, 0) : NaN);
    const s = v.split;
    return { split: s, splitSum: s.train + s.walk + s.drive + (s.none ?? 0), trips: v.tripsPerDay,
      home: m ? [sum(m.homeRail), sum(m.homeWalk), sum(m.homeDrive), sum(m.homeNone)] : null,
      work: m ? [sum(m.workRail), sum(m.workWalk), sum(m.workDrive), sum(m.workNone)] : null };
  }, zone);
}

/** Hover the bubbles round `at` until one with a No route part shows its tag: the tag's count, its bar's parts. */
async function hoverIsland(page, at) {
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  const p = await page.evaluate((at) => window.tw.map.project(at), at);
  let best = null;
  for (let r = 0; r <= 90 && !best; r += 4) for (let a = 0, na = r ? Math.max(8, Math.round(r / 2)) : 1; a < na && !best; a++) {
    const x = box.left + p.x + r * Math.cos((2 * a * Math.PI) / na), y = box.top + p.y + r * Math.sin((2 * a * Math.PI) / na);
    await page.mouse.move(x, y);
    await page.waitForTimeout(60);
    const tag = await page.evaluate(() => {
      const t = [...document.querySelectorAll('.demand-tag:not(.rest-list):not(.track-tag)')].find((e) => e.style.display !== 'none' && e.querySelector('.mode-bar'));
      if (!t) return null;
      const parts = [...t.querySelectorAll('.mode-bar i')].map((i) => ({ grow: Number(i.style.flexGrow), bg: i.style.background }));
      return { text: t.innerText.split('\n')[0], parts };
    });
    if (tag && tag.parts.some((q) => /none-grey/.test(q.bg))) best = { ...tag, x, y };
  }
  return best;
}

/** The inspector panel's text */
const panelText = (page) => page.evaluate(() => (document.querySelector('#inspector, .inspector')?.innerText ?? ''));

async function shots(page, isl, tag, label) {
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.waitForTimeout(1500);
    if (tag) { await page.mouse.move(tag.x - 40, tag.y); await page.mouse.move(tag.x, tag.y); await page.waitForTimeout(400); }
    const file = join(OUT, `${isl.zone}-${label}-${shade}.png`);
    await page.screenshot({ path: file });
  }
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
}

async function explore(page, isl) {
  await page.evaluate(([a, b]) => {
    const p = window.tw.lngLatToPack(a[0], a[1]), q = window.tw.lngLatToPack(b[0], b[1]);
    window.tw.selection.value = { kind: 'trip', from: [p.x, p.y], to: [q.x, q.y] };
  }, [isl.at, isl.across]);
  await page.waitForFunction(() => /By car/.test(document.querySelector('#inspector, .inspector')?.innerText ?? ''), undefined, { timeout: 120000, polling: 500 });
  await page.waitForTimeout(800);
  return panelText(page);
}

async function addFerry(page, isl) {
  return page.evaluate(async ([ends, name]) => {
    const tw = window.tw, water = tw.tools.ferry.water;
    const snap = async ([lng, lat]) => {
      const k = 1 / (111320 * Math.cos((lat * Math.PI) / 180)), j = 1 / 110540;
      for (const r of [0, 60, 120, 200, 300, 450]) for (let a = 0; a < (r ? 8 : 1); a++) {
        const s = await water.port([lng + r * Math.cos((a * Math.PI) / 4) * k, lat + r * Math.sin((a * Math.PI) / 4) * j]);
        if (s?.ok) return s;
      }
      return null;
    };
    const ports = [], berths = [];
    for (const e of ends) {
      const s = await snap(e);
      if (!s) return { error: `no port near ${e}` };
      const a = tw.lngLatToPack(s.landing[0], s.landing[1]), b = tw.lngLatToPack(s.berth[0], s.berth[1]);
      ports.push({ x: a.x, y: a.y, bx: b.x, by: b.y, name: '' });
      berths.push(s.berth);
    }
    const r = await water.line(berths.map((at) => ({ at, kind: 1 })), 'check');
    if (!r?.route.ok) return { error: `no route (${r?.route.legs.map((l) => l.reason ?? 'ok').join(', ')})` };
    const route = [];
    for (let i = 0; i < r.route.kinds.length; i++) { const q = tw.lngLatToPack(r.route.ll[2 * i], r.route.ll[2 * i + 1]); route.push(q.x, q.y, r.route.kinds[i], r.route.speeds[i]); }
    const e = await tw.client.edit({ op: 'addFerryLine', ports, route, name, colour: '#1f77b4' });
    if (!e?.ok || e.line === undefined) return { error: `not added (${(e?.issues ?? []).join(', ')})` };
    const c = await tw.client.edit({ op: 'constructLine', line: e.line });
    if (!c?.ok) return { error: `not constructed (${(c?.issues ?? []).join(', ')})` };
    return { line: String(e.line), km: r.route.m / 1000 };
  }, [isl.ferry, `${isl.name} ferry`]);
}

for (const isl of ISLANDS) {
  const chrome = await launchChrome({ prefix: `tw-t2215-${isl.zone}-`, size: [1200, 800], deviceScaleFactor: 1 });
  const page = await chrome.ctx.newPage();
  await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday', commuters: true, commuterEnd: 'home' }, v: 2 })); } catch {} });
  const errors = [];
  collectErrors(page, errors);
  try {
    await page.goto(`${BASE}?debug=1&new=1&city=${isl.zone}&demandWorkers=2&paused=1&v=${Date.now()}`);
    await page.waitForFunction((z) => window.twDemand?.cities?.()?.get(z) && window.tw?.tools?.ferry, isl.zone, { timeout: 300000, polling: 1000 });
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [isl.view, isl.zoom]);
    // the screenshots without the debug readout and the test page's autosave notice
    await page.addStyleTag({ content: '#debug, .save-notice { display: none !important; }' });
    log(`${isl.name}: ${isl.zone} open`);

    // 1. the split and the bubbles' legs add up
    const f = await zoneFigures(page, isl.zone);
    const legs = (a) => a.reduce((s, x) => s + x, 0);
    check(Math.abs(f.splitSum - 100) < 0.01 && (f.split.none ?? 0) > 0, `${isl.name}: ${isl.zone}'s split adds up to 100% with a No route share`, `train ${f.split.train.toFixed(2)}, walk ${f.split.walk.toFixed(2)}, drive ${f.split.drive.toFixed(2)}, no route ${(f.split.none ?? 0).toFixed(2)}`);
    for (const [end, v] of [['homes', f.home], ['destinations', f.work]])
      check(!!v && Math.abs(legs(v) - f.trips) <= 1e-3 * f.trips && v[3] > 0, `${isl.name}: the bubbles' legs at ${end} add up to the zone's trips, No route included`, v ? `${Math.round(legs(v)).toLocaleString('en-US')} of ${Math.round(f.trips).toLocaleString('en-US')}, no route ${Math.round(v[3]).toLocaleString('en-US')}` : 'no cell modes');

    // 2. a bubble's hover over the island (the view set again: a new game's start camera can land after the first)
    await page.waitForTimeout(3000);
    await page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [isl.view, isl.zoom]);
    await page.waitForTimeout(1500);
    let tag = await hoverIsland(page, isl.hover);
    const tagSum = tag ? tag.parts.reduce((s, q) => s + q.grow, 0) : 0;
    check(!!tag && tag.text.split(' ')[0] === count(tagSum), `${isl.name}: a bubble's hover counts its bar's parts, a grey No route among them`, tag ? `"${tag.text}", parts ${tag.parts.map((q) => Math.round(q.grow)).join(' + ')}` : 'no bubble with No route found');
    await shots(page, isl, tag, 'no-ferry');

    // 3. the city view
    await page.evaluate((z) => { window.tw.selection.value = { kind: 'city', city: z }; }, isl.zone);
    await page.waitForTimeout(1200);
    const cityBar = await page.evaluate(() => [...document.querySelectorAll('.zone-trips .seg-none')].map((e) => e.getAttribute('data-tip') ?? e.closest('[data-tip]')?.getAttribute('data-tip') ?? '').join(' | '));
    check(cityBar.length > 0, `${isl.name}: the city view's Within bar has a No route part`, cityBar.slice(0, 160));
    await page.screenshot({ path: join(OUT, `${isl.zone}-city-no-ferry-light.png`) });

    // 4. the trip explorer from the island to the shore across
    let text = await explore(page, isl);
    const noRoad = (text.match(/No road between these two places/g) ?? []).length;
    check(/No route/.test(text) && noRoad === 2 && !/Parking/.test(text), `${isl.name}: the explorer to the shore across has a No route row and no road by car or on foot`, text.split('\n').filter((s) => /No route|No road|Train|Walk|Drive/.test(s)).join(' | ').slice(0, 300));
    await page.screenshot({ path: join(OUT, `${isl.zone}-trip-no-ferry-light.png`) });
    await page.evaluate(() => { window.tw.selection.value = null; });

    // 5. the ferry
    await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
    const before = await page.evaluate(() => window.twDemand.view()?.version ?? 0);
    const made = await addFerry(page, isl);
    check(!made.error, `${isl.name}: a ferry to the shore built`, made.error ?? `${made.km.toFixed(1)} km`);
    if (!made.error) {
      await demandFinal(page, { timeout: 600000, polling: 1000, after: before });
      await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [isl.view, isl.zoom]);
      const g = await zoneFigures(page, isl.zone);
      check((g.split.none ?? 0) < (f.split.none ?? 0) && Math.abs(g.splitSum - 100) < 0.01, `${isl.name}: with the ferry the No route share falls, the split still adds up`, `${(f.split.none ?? 0).toFixed(3)}% -> ${(g.split.none ?? 0).toFixed(3)}%, ferry riders ${Math.round(await page.evaluate((id) => window.twDemand.view()?.lines.get(id)?.ridersPerDay ?? 0, made.line)).toLocaleString('en-US')} a weekday`);
      if (g.home) check(Math.abs(legs(g.home) - g.trips) <= 1e-3 * g.trips, `${isl.name}: with the ferry the bubbles' legs still add up`, `${Math.round(legs(g.home)).toLocaleString('en-US')} of ${Math.round(g.trips).toLocaleString('en-US')}`);
      await page.waitForTimeout(3000);
      tag = await hoverIsland(page, isl.hover);
      await shots(page, isl, tag, 'ferry');
      await page.evaluate((z) => { window.tw.selection.value = { kind: 'city', city: z }; }, isl.zone);
      await page.waitForTimeout(1200);
      await page.screenshot({ path: join(OUT, `${isl.zone}-city-ferry-light.png`) });
      text = await explore(page, isl);
      const train = /Train\s*\n?\s*(\d+)%/.exec(text);
      check(/No road between these two places/.test(text), `${isl.name}: with the ferry the explorer still says no road`, text.split('\n').filter((s) => /No route|No road|Train/.test(s)).join(' | ').slice(0, 300));
      await page.screenshot({ path: join(OUT, `${isl.zone}-trip-ferry-light.png`) });
      if (train) log(`${isl.name}: explorer train share with the ferry ${train[1]}%`);
    }
    check(errors.length === 0, `${isl.name}: no console errors`, errors.slice(0, 3).join(' | '));
  } catch (e) {
    check(false, `${isl.name}: ran to the end`, e.stack ?? e.message);
  } finally {
    await chrome.close();
  }
}
console.log(fail ? `T-2215 FAILED: ${fail}` : 'T-2215 PASSED');
process.exit(fail ? 1 : 0);
