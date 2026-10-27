// T-1530: how long the station bubbles wait for their figures. On the real save: the bubbles turned on
// (Origins, which needs the transfers split, and Total) after the final and while a load solves, and a
// new line's stations under Total and under Origins; and a new line in a new New York game.
//   BASE=http://localhost:9001/ node trainworld/checks/t1530-station-bubbles.mjs
// CASES=on,load,line,new (default all); QUERY more URL parameters for every page (perfOff=splitAhead:
// before T-1530); OUT a JSON of the rows and samples. Prints each time; fails only on page errors or
// a figure never shown.
import { readFile, writeFile } from 'node:fs/promises';
import { launchChrome, importSave, demandFinal, buildLine, collectErrors, REAL_SAVE, ANGLE } from './lib.mjs';

const BASE = process.env.BASE ?? 'http://localhost:9001/';
const CASES = (process.env.CASES ?? 'on,load,line,new').split(',');
const OUT = process.env.OUT ?? null;
const Q = process.env.QUERY ? `&${process.env.QUERY}` : '';

/** the stations' figures now: `ids` the stations watched (all when null) */
const probe = (ids) => {
  const m = window.__t1530;
  if (!m) return null;
  const dv = m.dm.demandView.peek(), su = m.su.stationUse.peek();
  const want = ids ? new Set(ids) : null;
  const has = (map) => !!map && (want ? [...want].some((id) => map.has(id)) : map.size > 0);
  return {
    t: performance.now(), version: dv?.version ?? null, round: dv?.round ?? null, final: !!dv?.final,
    total: has(dv?.stations), split: has(su?.at), splitCurrent: !!su && su.dv === dv,
  };
};

async function until(page, t0, ok, limit, ids = null) {
  const samples = [];
  for (const w0 = Date.now(); Date.now() - w0 < limit;) {
    const s = await page.evaluate(probe, ids).catch(() => null);
    if (s) {
      s.ms = Math.round(s.t - t0);
      samples.push(s);
      if (ok(s)) return { ms: s.ms, samples };
    }
    await page.waitForTimeout(100);
  }
  return { ms: null, samples };
}
const hook = (page) => page.evaluate(async () => {
  window.__t1530 = { su: await import('/src/game/stationUse.ts'), dm: await import('/src/game/demand.ts') };
});
const setUse = (page, mode) => page.evaluate((mode) => { const d = window.tw.display; d.value = { ...d.value, stationUse: mode }; return performance.now(); }, mode);

const rows = [], errors = [], all = [];
const row = (name, ms, extra = '') => {
  rows.push({ name, ms });
  console.log(`${name.padEnd(26)} ${ms === null ? 'never' : (ms / 1000).toFixed(2) + ' s'}${extra ? '  ' + extra : ''}`);
};

async function withPage(f) {
  const { ctx, close } = await launchChrome({ prefix: 'tw-t1530-', args: ANGLE });
  try {
    const page = await ctx.newPage();
    collectErrors(page, errors);
    await f(page);
  } finally {
    await close();
  }
}

async function prepare(page) {
  await page.goto(`${BASE}?debug=1&continue=1&paused=1${Q}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.lastSaved, undefined, { timeout: 180000 });
  const before = await page.evaluate(() => window.tw.lastSaved.peek() ?? 0);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 400000 });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.tw.client.post({ kind: 'flush' }));
  await page.waitForFunction((b) => (window.tw.lastSaved.peek() ?? 0) > b, before, { timeout: 120000 });
}
async function pick(page) {
  await page.goto(`${BASE}?debug=1${Q}`);
  await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
  await hook(page);
  await setUse(page, 'none');
  return page.evaluate(() => { const t = performance.now(); document.querySelector('.start-menu .start-save').click(); return t; });
}

/** a new line through `pts`; the ids of the stations it added */
async function newLine(page, pts) {
  const before = await page.evaluate(() => window.tw.world.value.save.stations.map((s) => s.id));
  await buildLine(page, pts, { trains: [10, 8, 6], wait: { route: 300, station: 150, settle: 300 } });
  const t0 = await page.evaluate(() => performance.now());
  const ids = await page.evaluate((b) => { const had = new Set(b); return window.tw.world.value.save.stations.filter((s) => s.built && !had.has(s.id)).map((s) => s.id); }, before);
  return { t0, ids };
}
async function lineCase(page, name, pts) {
  await setUse(page, 'origins');
  await page.waitForTimeout(1500);
  const { t0, ids } = await newLine(page, pts);
  const r = await until(page, t0, (s) => s.total && s.split, 300000, ids);
  const first = (k) => r.samples.find((s) => s[k])?.ms ?? null;
  row(`${name}: total`, first('total'), `(${ids.length} new stations)`);
  row(`${name}: origins`, first('split'));
  all.push({ name, samples: r.samples });
}

if (CASES.some((c) => ['on', 'load', 'line'].includes(c))) await withPage(async (page) => {
  await prepare(page);
  if (CASES.includes('load')) {
    // the bubbles on (Origins) as the save is picked: Total's figures, then the split
    const t0 = await pick(page);
    await setUse(page, 'origins');
    const r = await until(page, t0, (s) => s.split && s.final && s.splitCurrent, 400000);
    const first = (k) => r.samples.find((s) => s[k])?.ms ?? null;
    row('load: first figures', first('total'), '(from the pick)');
    row('load: first split', first('split'));
    row('load: final split', r.ms);
    all.push({ name: 'load', samples: r.samples });
    await demandFinal(page, { held: true, timeout: 400000 });
  } else {
    await pick(page);
    await demandFinal(page, { held: true, timeout: 400000 });
  }
  if (CASES.includes('on')) {
    await page.waitForTimeout(3000);
    // a fresh page holds no split, so the first time on asks the workers (as a player's first)
    await page.reload();
    await page.waitForSelector('.start-menu .start-save', { timeout: 60000 });
    await hook(page);
    await setUse(page, 'none');
    await page.evaluate(() => document.querySelector('.start-menu .start-save').click());
    await demandFinal(page, { held: true, timeout: 400000 });
    await page.waitForTimeout(3000);
    for (const mode of ['origins', 'transfers']) {
      const t0 = await setUse(page, mode);
      const r = await until(page, t0, (s) => s.split && s.splitCurrent, 120000);
      row(`on after final: ${mode}`, r.ms);
      all.push({ name: `on ${mode}`, samples: r.samples });
      await setUse(page, 'none');
      await page.waitForTimeout(1000);
    }
  }
  if (CASES.includes('line')) {
    await page.evaluate(() => window.tw.map.jumpTo({ center: [-74.12, 40.60], zoom: 12 }));
    await page.waitForTimeout(3000);
    await lineCase(page, 'line real', [[-74.15, 40.58], [-74.12, 40.60], [-74.09, 40.62]]);
  }
});

if (CASES.includes('new')) await withPage(async (page) => {
  await page.goto(`${BASE}?new=1&debug=1${Q}`);
  await page.waitForFunction(() => !!window.twDemand && !!window.twViews && !!window.tw?.map, undefined, { timeout: 120000 });
  await hook(page);
  await demandFinal(page, { held: true, timeout: 300000 });
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'track'; });
  await page.waitForTimeout(5000);
  await lineCase(page, 'line new', [[-73.99, 40.73], [-73.98, 40.75], [-73.97, 40.77]]);
  // a second line, joining the first: its stations' figures come with figures already shown
  await page.waitForFunction(() => window.twDemand.view()?.final && !window.twDemand.client.waiting, undefined, { timeout: 300000 });
  await page.waitForTimeout(2000);
  await lineCase(page, 'line new 2', [[-73.95, 40.70], [-73.96, 40.72], [-73.985, 40.735]]);
});

if (OUT) await writeFile(OUT, JSON.stringify({ rows, errors, all }, null, 1));
if (errors.length) console.log('  errors: ' + JSON.stringify(errors.slice(0, 5)));
const never = rows.filter((r) => r.ms === null);
console.log(never.length || errors.length ? `FAIL t1530 station bubbles (${never.map((r) => r.name).join(', ') || 'errors'})` : `PASS t1530 station bubbles (${rows.length} times)`);
process.exitCode = never.length || errors.length ? 1 : 0;
