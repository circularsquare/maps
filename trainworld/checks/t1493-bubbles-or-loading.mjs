// T-1493: whenever the zone in view has no demand bubbles, something says it is loading, and the
// bubbles always arrive. Jumps the camera between cities at several zooms, with Destinations on, in a
// new game (and with SAVE=1 the real save), sampling every SAMPLE_MS: the focused zone's bubbles drawn
// (zoomed out, the one bubble per zone), and the loading card, the status tag, the zone's bar or its
// grey. FAIL when a sample past GRACE_MS shows neither, or a stop ends its DWELL_MS without bubbles.
//   BASE=http://localhost:6173/ node trainworld/checks/t1493-bubbles-or-loading.mjs
// SAVE=1 adds the real save; DWELL ms per stop (12000); OUT a JSON of every sample.
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, collectErrors, importSave, demandFinal, REAL_SAVE } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6101/';
const SAVE = process.env.SAVE === '1';
const DWELL = Number(process.env.DWELL ?? 12000);
const SAMPLE_MS = 250;
/** a jump may take this long to show a loading state (the focus follows the camera after it settles) */
const GRACE_MS = 1500;
const OUT = process.env.OUT ?? null;

// [label, lon, lat, zoom]: in a zone, across zone lines, zoomed out to the one bubble per zone, far away
const NEW_STOPS = [
  ['nyc', -73.985, 40.758, 12], ['phl', -75.1636, 39.9526, 11], ['nyc', -73.985, 40.758, 10],
  ['bos', -71.0589, 42.3601, 12], ['nyc-out', -73.985, 40.758, 8], ['tor', -79.3832, 43.6534, 11],
  ['bal', -76.6122, 39.2904, 9], ['lon', -0.1246, 51.5073, 11], ['par', 2.3488, 48.8534, 10],
  ['ber', 13.3777, 52.5163, 12], ['par-out', 2.3488, 48.8534, 8], ['tyo', 139.7671, 35.6812, 11],
  ['wuh', 114.2643, 30.5785, 11], ['sao', -46.6339, -23.5503, 10], ['nyc', -73.985, 40.758, 11],
];
const SAVE_STOPS = [
  ['nyc', -73.985, 40.758, 12], ['jfk', -73.79, 40.66, 11], ['phl', -75.1636, 39.9526, 11],
  ['nyc-out', -73.985, 40.758, 8], ['bos', -71.0589, 42.3601, 11], ['nyc', -73.985, 40.758, 10],
];

const probe = () => {
  const d = window.twDemand, o = window.twViews.overlay, z = window.twZoneLoading?.();
  const lvl = o.lastLevel;
  let near = 0;
  if (o.showBubbles) for (const s of o.nearby.values()) near += s[lvl]?.count ?? 0;
  const tag = document.querySelector('.status-tag .demand-busy')?.textContent ?? null;
  return {
    t: Math.round(performance.now()), zoom: +window.tw.map.getZoom().toFixed(2), focus: d.activeCity.peek(), viewed: z?.viewed ?? null, realm: z?.realmShown ?? true,
    own: o.showBubbles ? o.bubbles[lvl]?.count ?? 0 : 0, near, inter: o.sets.get('inter')?.count ?? 0, visible: o.showBubbles,
    card: !!document.querySelector('.loading-card'), tag, bar: !!z?.bar, grey: !!z?.grey, zoneLoading: !!z?.loading,
    opened: d.client.opened.has(d.activeCity.peek()), bare: d.client.bareOpen.has(d.activeCity.peek()), hasView: d.cities().has(d.activeCity.peek()),
    version: d.view()?.version ?? null, final: !!d.view()?.final,
  };
};

const OVERVIEW_ZOOM = 8.5;
/** bubbles for the zone in view: its own zoomed in, the one bubble per zone zoomed out */
const hasBubbles = (s) => (s.zoom <= OVERVIEW_ZOOM ? s.inter > 0 : s.own > 0);
const loadingShown = (s) => s.card || !!s.tag || s.bar || s.grey;
/** a zone in view whose realm is shown: the rule applies */
const applies = (s) => !!s.viewed && s.realm;

async function run(page, label, stops) {
  const fails = [], rows = [];
  for (const [name, lon, lat, zoom] of stops) {
    await page.evaluate(([lon, lat, z]) => window.tw.map.jumpTo({ center: [lon, lat], zoom: z }), [lon, lat, zoom]);
    const t0 = Date.now(), samples = [];
    let firstBubbles = null, gaps = 0;
    while (Date.now() - t0 < DWELL) {
      await page.waitForTimeout(SAMPLE_MS);
      const s = await page.evaluate(probe);
      s.ms = Date.now() - t0;
      samples.push(s);
      if (!applies(s)) continue;
      if (hasBubbles(s)) { firstBubbles ??= s.ms; continue; }
      if (s.ms > GRACE_MS && !loadingShown(s)) gaps++;
    }
    const last = samples.at(-1);
    const bad = samples.filter((s) => applies(s) && s.ms > GRACE_MS && !hasBubbles(s) && !loadingShown(s));
    const row = { label, stop: name, zoom, focus: last.focus, applies: applies(last), firstBubbles, gaps, end: last };
    rows.push({ ...row, samples });
    const endBare = applies(last) && !hasBubbles(last);
    if (bad.length) fails.push(`${label} ${name} z${zoom}: ${bad.length} samples with neither bubbles nor a loading state (first at ${bad[0].ms} ms: ${JSON.stringify(bad[0])})`);
    if (endBare) fails.push(`${label} ${name} z${zoom}: no bubbles after ${DWELL} ms (${loadingShown(last) ? 'still loading' : 'nothing shown'}: ${JSON.stringify(last)})`);
    console.log(`${label} ${name.padEnd(8)} z${zoom} focus=${last.focus} viewed=${last.viewed} realm=${last.realm} bubbles@${firstBubbles ?? '-'}ms gaps=${gaps} end own=${last.own} inter=${last.inter} card=${last.card} tag=${last.tag} bar=${last.bar} grey=${last.grey}`);
  }
  return { fails, rows };
}

const { ctx, close } = await launchChrome({ prefix: 'tw-t1493-' });
const errors = [], all = [];
let fails = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?new=1&debug=1&v=${Date.now()}`);
  await page.waitForFunction(() => !!window.twDemand && !!window.twViews && !!window.tw?.map, undefined, { timeout: 120000 });
  await demandFinal(page, { held: true, timeout: 240000 });
  // Destinations on, as a new game opens (T-1441)
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'work' }; });
  let r = await run(page, 'new', NEW_STOPS);
  fails.push(...r.fails); all.push(...r.rows);
  if (SAVE) {
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 120000 });
    const before = await page.evaluate(() => window.tw.lastSaved.peek());
    await importSave(page, await readFile(REAL_SAVE));
    await page.waitForFunction(() => (window.tw.world.value?.save?.lines?.length ?? 0) > 100, undefined, { timeout: 240000 });
    await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
    await page.waitForFunction((b) => window.tw.lastSaved.peek() > b, before, { timeout: 60000 });
    await page.goto(`${BASE}?debug=1&continue=1&v=${Date.now()}`);
    await page.waitForFunction(() => !!window.twDemand && !!window.twViews && !!window.tw?.map, undefined, { timeout: 120000 });
    await demandFinal(page, { held: true, timeout: 300000 });
    await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'work' }; });
    r = await run(page, 'save', SAVE_STOPS);
    fails.push(...r.fails); all.push(...r.rows);
  }
} finally {
  await close();
}
if (OUT) await writeFile(OUT, JSON.stringify({ fails, errors, rows: all }, null, 1));
for (const f of fails) console.log('  ' + f);
if (errors.length) console.log('  errors: ' + JSON.stringify(errors.slice(0, 5)));
console.log(fails.length ? `FAIL t1493 bubbles or loading (${fails.length})` : 'PASS t1493 bubbles or loading');
process.exitCode = fails.length ? 1 : 0;
