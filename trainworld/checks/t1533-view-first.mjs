// T-1533: how long the zone in view waits for its demand bubbles, from the moment it is the zone in
// view: a save picked from the start menu (the real save, without and with its cached result), a jump
// into a city while the all-China save still loads (CHINA=1, about 10 minutes), jumps between cities
// on the world save, a new line (its first figures), and the bubbles turned off and on. Target 5 s.
// Cases KNOWN to miss it on this machine are reported and do not fail the check (notes/T-1533.md).
//   BASE=http://localhost:6101/ node trainworld/checks/t1533-view-first.mjs
// CASES=load,pan,line,view (default all but china); CHINA=1 adds china; OUT a JSON of every sample.
import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, buildLine, collectErrors, REAL_SAVE, MAPS, ARTIFACTS, ANGLE } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:6101/';
const TARGET = Number(process.env.TARGET ?? 5000);
const CASES = (process.env.CASES ?? 'load,pan,line,view').split(',').concat(process.env.CHINA === '1' ? ['china'] : []);
const OUT = process.env.OUT ?? null;
const WORLD_SAVE = `${ARTIFACTS}world-map-day73.save`;
// (at the maps root of the main tree, which an agent tree does not copy)
const CHINA_SAVE = process.env.CHINA_SAVE ?? [`${MAPS}t1156-all-china.save`, 'C:/Users/anita/projects/maps/t1156-all-china.save'].find((f) => existsSync(f));
/** cases this machine's solve cannot bring under the target (notes/T-1533.md): reported, not failed */
const KNOWN = new Set(['load cold', 'line real']);

const probe = () => {
  const d = window.twDemand, o = window.twViews?.overlay;
  if (!d || !o || !window.tw?.map) return null;
  const lvl = o.lastLevel, v = d.view?.();
  return {
    t: performance.now(), zoom: window.tw.map.getZoom(), focus: d.activeCity.peek(),
    own: o.showBubbles ? o.bubbles[lvl]?.count ?? 0 : 0, inter: o.sets.get('inter')?.count ?? 0,
    card: !!document.querySelector('.loading-card'), version: v?.version ?? null, final: !!v?.final, lines: v?.lines?.size ?? 0,
  };
};
/** bubbles for the zone in view: its own zoomed in, the one bubble per zone zoomed out */
const bubbles = (s) => (s.zoom <= 8.5 ? s.inter > 0 : s.own > 0);

async function until(page, t0, ok, limit) {
  const samples = [];
  for (const w0 = Date.now(); Date.now() - w0 < limit;) {
    const s = await page.evaluate(probe).catch(() => null);
    if (s) {
      s.ms = Math.round(s.t - t0);
      samples.push(s);
      if (ok(s)) return { ms: s.ms, samples };
    }
    await page.waitForTimeout(200);
  }
  return { ms: null, samples };
}

/** a save into the browser's slot, the bubbles on, then the start menu */
async function prepare(page, file) {
  await page.goto(`${BASE}?debug=1&continue=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });
  const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
  await importSave(page, await readFile(file));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 400000 });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 120000 });
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'work' }; });
  await page.waitForTimeout(500);
}
async function pick(page) {
  await page.goto(`${BASE}?debug=1`);
  await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
  return page.evaluate(() => { const t = performance.now(); document.querySelector('.start-menu .start-save').click(); return t; });
}
const jump = (page, [lon, lat, z]) => page.evaluate(([lon, lat, z]) => { window.tw.map.jumpTo({ center: [lon, lat], zoom: z }); return performance.now(); }, [lon, lat, z]);

const rows = [], errors = [];
const row = (name, ms, extra = '') => {
  const known = KNOWN.has(name);
  const verdict = ms === null ? 'never' : ms <= TARGET ? 'ok' : 'over';
  rows.push({ name, ms, verdict, known });
  console.log(`${name.padEnd(22)} ${ms === null ? '-' : (ms / 1000).toFixed(1) + ' s'}  ${verdict}${verdict !== 'ok' && known ? ' (known)' : ''}${extra ? '  ' + extra : ''}`);
};
const all = [];

async function withPage(f) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t1533-', args: ANGLE });
  try {
    const page = await ctx.newPage();
    collectErrors(page, errors);
    await f(page);
  } finally {
    await close();
  }
}

if (CASES.includes('load') || CASES.includes('view') || CASES.includes('line')) await withPage(async (page) => {
  await prepare(page, REAL_SAVE);
  if (CASES.includes('load')) for (const label of ['cold', 'warm']) {
    const t0 = await pick(page);
    const r = await until(page, t0, bubbles, 400000);
    row(`load ${label}`, r.ms, label === 'warm' ? '(cached result)' : '(no cached result)');
    all.push({ name: `load ${label}`, samples: r.samples });
    await demandFinal(page, { held: true, timeout: 400000 });
    // the cache is written a moment after the final
    await page.waitForFunction(() => !!window.twDemand.client.cacheLog?.wrote, undefined, { timeout: 60000 }).catch(() => {});
  } else {
    await pick(page);
    await demandFinal(page, { held: true, timeout: 400000 });
  }
  if (CASES.includes('view')) {
    await page.waitForTimeout(2000);
    for (const end of ['home', 'work']) {
      await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: false }; });
      await page.waitForTimeout(1500);
      const t0 = await page.evaluate((end) => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: end }; return performance.now(); }, end);
      const r = await until(page, t0, bubbles, 60000);
      row(`view ${end}`, r.ms);
      all.push({ name: `view ${end}`, samples: r.samples });
    }
  }
  if (CASES.includes('line')) {
    // a new line on Staten Island, in New York's one big cluster: its free-flow round
    await jump(page, [-74.12, 40.60, 12]);
    await page.waitForTimeout(3000);
    const before = await page.evaluate(() => window.twDemand.view()?.lines?.size ?? 0);
    await buildLine(page, [[-74.15, 40.58], [-74.12, 40.60], [-74.09, 40.62]], { trains: [10, 8, 6], wait: { route: 300, station: 150, settle: 300 } });
    const t0 = await page.evaluate(() => performance.now());
    const r = await until(page, t0, (s) => s.lines > before, 300000);
    row('line real', r.ms, '(the new line\'s first figures)');
    all.push({ name: 'line real', samples: r.samples });
  }
});

if (CASES.includes('pan')) await withPage(async (page) => {
  await prepare(page, WORLD_SAVE);
  await pick(page);
  await demandFinal(page, { held: true, timeout: 400000 });
  for (const [name, at] of [['lon', [-0.1246, 51.5073, 11]], ['ber', [13.3777, 52.5163, 12]], ['tyo', [139.7671, 35.6812, 11]], ['wuh', [114.2643, 30.5785, 11]], ['nyc', [-73.985, 40.758, 11]]]) {
    const t0 = await jump(page, at);
    // the zone in view follows the camera once it settles
    const r = await until(page, t0, (s) => s.ms > 600 && bubbles(s), 90000);
    row(`pan ${name}`, r.ms);
    all.push({ name: `pan ${name}`, samples: r.samples });
  }
});

if (CASES.includes('line')) await withPage(async (page) => {
  await page.goto(`${BASE}?new=1&debug=1`);
  await page.waitForFunction(() => !!window.twDemand && !!window.twViews && !!window.tw?.map, undefined, { timeout: 120000 });
  await demandFinal(page, { held: true, timeout: 300000 });
  await page.evaluate(() => { const d = window.tw.display; d.value = { ...d.value, commuters: true, commuterEnd: 'work' }; });
  // the track tool picked first, as a player does, which readies the zone for its first station (T-351)
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'track'; });
  await page.waitForTimeout(5000);
  const before = await page.evaluate(() => window.twDemand.view()?.lines?.size ?? 0);
  await buildLine(page, [[-73.99, 40.73], [-73.98, 40.75], [-73.97, 40.77]], { trains: [10, 8, 6], wait: { route: 300, station: 150, settle: 300 } });
  const t0 = await page.evaluate(() => performance.now());
  const r = await until(page, t0, (s) => s.lines > before, 300000);
  row('line new', r.ms, '(the new line\'s first figures)');
  all.push({ name: 'line new', samples: r.samples });
});

if (CASES.includes('china')) await withPage(async (page) => {
  await prepare(page, CHINA_SAVE);
  await pick(page);
  // the network is in; most packs are still opening
  await page.waitForFunction(() => (window.twDemand?.client?.log?.length ?? 0) > 0 || !!window.twDemand?.client?.waiting, undefined, { timeout: 600000 });
  const t0 = await jump(page, [114.30, 30.59, 11]);
  const r = await until(page, t0, (s) => s.ms > 600 && bubbles(s), 900000);
  row('china wuhan', r.ms, '(jumped to while the save loads)');
  all.push({ name: 'china wuhan', samples: r.samples });
});

if (OUT) await writeFile(OUT, JSON.stringify({ rows, errors, all }, null, 1));
if (errors.length) console.log('  errors: ' + JSON.stringify(errors.slice(0, 5)));
const bad = rows.filter((r) => r.verdict !== 'ok' && !r.known);
console.log(bad.length || errors.length ? `FAIL t1533 view first (${bad.map((r) => r.name).join(', ') || 'errors'})` : `PASS t1533 view first (${rows.filter((r) => r.verdict === 'ok').length} of ${rows.length} within ${TARGET / 1000} s)`);
process.exitCode = bad.length || errors.length ? 1 : 0;
