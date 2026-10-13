// T-207 browser check: how full trains get in play. Two cases, each measured from demand's result
// (fullest train per line) and the boarding account (each train's own load, platforms leaving
// riders behind) at the morning peak's busiest hour:
//   real: New York's real network (T-007 save);
//   built: a new game with one Manhattan trunk (Harlem to Fulton St under Lexington, 18 stops,
//          elevated, 200 m platforms), at a modest fleet and then at a small one.
// Needs a Vite dev server (BASE, default http://localhost:6931; file watching off) and Playwright
// (PLAYWRIGHT_MODULE, a file URL to its index.mjs). PROFILE_DIR: where the throwaway Chrome profile
// goes. CASES=real,built picks. Run from the maps root: node trainworld/notes/t207-browser.mjs.
// Screenshots: artifacts/trainworld/T-207-*.png. Results: artifacts/trainworld/notes-outputs/t207-browser.json.
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const ART = join(ROOT, '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6931';
const CASES = (process.env.CASES ?? 'real,built').split(',');
const TAG = process.env.TAG ?? '';
const FLEETS = JSON.parse(process.env.FLEETS ?? '[[14,8,4],[7,4,2]]');
const STOPS = [[-73.9377, 40.8042], [-73.9416, 40.7986], [-73.9476, 40.7906], [-73.9559, 40.7795], [-73.9599, 40.7737], [-73.9640, 40.7685], [-73.9677, 40.7626], [-73.9720, 40.7571], [-73.9766, 40.7527], [-73.9821, 40.7461], [-73.9867, 40.7393], [-73.9903, 40.7359], [-73.9911, 40.7300], [-73.9941, 40.7259], [-73.9973, 40.7222], [-74.0003, 40.7188], [-74.0041, 40.7131], [-74.0079, 40.7094]];
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't207-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = {};

/** In the page: the boarding account's hour at game time t (paused there), per line or for one. */
const measure = async (page, t, only) => {
  await page.evaluate(t => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(t); tw.overlay.invalidate(); }, t);
  await page.waitForFunction(t => window.tw.renderer.board && window.tw.renderer.board.epoch === Math.floor(t / 3600) * 3600, t, { timeout: 120000 });
  await page.waitForTimeout(800);
  return page.evaluate(only => {
    const tw = window.tw, r = tw.renderer, b = r.board, n = r.net, w = tw.world.value;
    const dv = window.twDemand.view();
    const rows = [];
    const byLine = new Map();
    for (let i = 0; i < r.tripCount; i++) {
      const prof = r.trips[i * 2], line = Math.floor(prof / 6);
      if (only !== null && line !== only) continue;
      const dep = r.trips[i * 2 + 1];
      if (dep < 0 || dep >= 3600) continue; // trains leaving their first stop this hour
      const o = b.offs[i * 2];
      if (o < 0) continue;
      const l = w.save.lines.find(l => l.num === line);
      const stretches = l.stops.length - 1;
      const cars = n.lineCars?.[line] ?? 8;
      let max = 0;
      for (let k = 0; k < stretches; k++) max = Math.max(max, b.loads[o + k]);
      const share = max / (160 * cars);
      rows.push(share);
      const e = byLine.get(line) ?? { name: l.name, trains: 0, full: 0, max: 0, sum: 0 };
      e.trains++; e.sum += share; e.max = Math.max(e.max, share); if (share > 0.99) e.full++;
      byLine.set(line, e);
    }
    rows.sort((a, b) => a - b);
    const q = p => rows.length ? +rows[Math.min(rows.length - 1, Math.floor(p * rows.length))].toFixed(3) : null;
    let left = 0, plats = 0, arriving = 0;
    for (let k = 0; k < b.queueCount; k++) {
      const Q = b.queues.subarray(k * 10, k * 10 + 10);
      if (only !== null && Q[1] !== only) continue;
      arriving += Q[7];
      if (Q[5] >= 1) { plats++; left += Q[5]; }
    }
    const lines = [...byLine.values()].sort((a, b) => b.max - a.max).slice(0, 8).map(e => ({ name: e.name, trains: e.trains, meanOfCrush: +(e.sum / e.trains).toFixed(2), fullestOfCrush: +e.max.toFixed(2), atCrush: e.full }));
    return {
      trains: rows.length, medianOfCrush: q(0.5), p90OfCrush: q(0.9), maxOfCrush: q(1),
      overSeats: rows.filter(s => s > 44 / 160).length, over80: rows.filter(s => s > 0.8).length, atCrush: rows.filter(s => s > 0.99).length,
      platformsLeavingRiders: plats, leftBehind: Math.round(left), arriving: Math.round(arriving),
      peakHourFactor: dv?.peakHourFactor, lines,
    };
  }, only);
};

/** Demand's own view of a line or all: fullest train % of crush in the high period. */
const demandLines = (page, only) => page.evaluate(only => {
  const dv = window.twDemand.view(), w = window.tw.world.value;
  const rows = [];
  for (const [id, d] of dv.lines) {
    const l = w.save.lines.find(l => l.id === id);
    if (only !== null && l?.num !== only) continue;
    rows.push({ name: l?.name, ridersPerDay: Math.round(d.ridersPerDay), fullestPct: Math.round(d.fullestPct), trains: w.lineStats[id]?.trainsNeeded, cars: window.tw.renderer.net.lineCars?.[l?.num] });
  }
  rows.sort((a, b) => b.fullestPct - a.fullestPct);
  return { lines: rows.length, over90: rows.filter(r => r.fullestPct >= 90).length, over100: rows.filter(r => r.fullestPct >= 100).length, top: rows.slice(0, 10) };
}, only);

try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round === 3, undefined, { timeout: 300000 });
  const day = await page.evaluate(() => Math.floor(window.tw.clock.now() / 86400) * 86400 + 86400);
  if (CASES.includes('real')) {
    const v0 = await page.evaluate(() => window.twDemand.view().version);
    const bytes = await readFile(join(ROOT, 'T-007-real-nyc.save'));
    await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
    await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size > 100, v0, { timeout: 600000 });
    const d = await page.evaluate(() => Math.floor(window.tw.clock.now() / 86400) * 86400);
    out.real = { demand: await demandLines(page, null), am8: await measure(page, d + 8 * 3600 + 1800, null), am7: await measure(page, d + 7 * 3600 + 1800, null), pm17: await measure(page, d + 17 * 3600 + 1800, null), mid12: await measure(page, d + 12 * 3600 + 1800, null) };
    console.log('real', JSON.stringify(out.real));
  }
  if (CASES.includes('save') && process.env.SAVE) {
    // a player's own save (read only; it loads into this throwaway profile)
    const v0 = await page.evaluate(() => window.twDemand.view().version);
    const bytes = await readFile(process.env.SAVE);
    await page.evaluate(data => { const bytes = Uint8Array.from(data).buffer; window.tw.client.post({ kind: 'import', bytes }, [bytes]); }, Array.from(bytes));
    await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3, v0, { timeout: 600000 });
    const d = await page.evaluate(() => Math.floor(window.tw.clock.now() / 86400) * 86400);
    out.save = { demand: await demandLines(page, null), am8: await measure(page, d + 8 * 3600 + 1800, null), pm17: await measure(page, d + 17 * 3600 + 1800, null), mid12: await measure(page, d + 12 * 3600 + 1800, null) };
    out.save.lineStats = await page.evaluate(() => { const w = window.tw.world.value; return w.save.lines.map(l => ({ name: l.name, stops: l.stops.length, trains: w.lineStats[l.id]?.trainsNeeded, roundTripS: Math.round(w.lineStats[l.id]?.roundTripS), cars: window.tw.renderer.net.lineCars?.[l.num] })); });
    console.log('save', JSON.stringify(out.save));
  }
  if (CASES.includes('built')) {
    const v0 = await page.evaluate(() => window.twDemand.view().version);
    await page.evaluate(() => window.tw.client.post({ kind: 'newGame' }));
    await page.waitForFunction(() => window.tw.world.value.save.lines.length === 0, undefined, { timeout: 60000 });
    const line = await page.evaluate(async ([stops, fleet]) => {
      const t = window.tw, run = async op => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
      const nodes = [];
      for (let i = 1; i < stops.length; i++) {
        const b = t.lngLatToPack(...stops[i]);
        const from = i === 1 ? (() => { const a = t.lngLatToPack(...stops[0]); return { kind: 'free', x: a.x, y: a.y, level: 1 }; })() : { kind: 'node', node: nodes.at(-1) };
        await run({ op: 'route', from, to: { kind: 'free', x: b.x, y: b.y, level: 1 }, pis: [], single: false });
        const e = t.world.value.edges.at(-1);
        if (i === 1) nodes.push(e.a);
        nodes.push(e.b);
      }
      for (const [i, n] of nodes.entries()) await run({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: 'Lex ' + (i + 1) });
      const line = (await run({ op: 'addLine', stops: nodes, name: 'Lexington trunk', colour: '#1f6fd1' })).line;
      await run({ op: 'setTrainCount', line, trains: fleet });
      await run({ op: 'constructAll' });
      return line;
    }, [STOPS, FLEETS[0]]);
    out.built = [];
    for (const [k, fleet] of FLEETS.entries()) {
      if (k > 0) {
        const v = await page.evaluate(() => window.twDemand.view().version);
        await page.evaluate(([line, fleet]) => window.tw.client.edit({ op: 'setTrainCount', line, trains: fleet }), [line, fleet]);
        await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3, v, { timeout: 300000 });
      } else {
        await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size === 1, v0, { timeout: 300000 });
      }
      const stats = await page.evaluate(line => { const w = window.tw.world.value, l = w.save.lines.find(l => l.num === line); return { roundTripS: w.lineStats[l.id]?.roundTripByLevel, trains: w.lineStats[l.id]?.trainsNeeded, tph: l.tph }; }, line);
      const res = { fleet, stats, demand: await demandLines(page, line) };
      for (const [key, h] of [['am7', 7.5], ['am8', 8.5], ['am9', 9.5], ['pm17', 17.5], ['mid12', 12.5]]) res[key] = await measure(page, day + h * 3600, line);
      out.built.push(res);
      console.log('built', JSON.stringify(res));
      // the line at 08:30 for a look
      await page.evaluate(() => { window.tw.map.jumpTo({ center: [-73.9745, 40.7550], zoom: 15.2 }); window.tw.overlay.invalidate(); });
      await page.evaluate(t => { window.tw.clock.set(t); window.tw.overlay.invalidate(); }, day + 8.5 * 3600);
      await page.waitForTimeout(2500);
      await page.screenshot({ path: join(ART, `T-207-built-${fleet.join('-')}${TAG}.png`) });
      // a train packed full: the first downtown train this hour at crush on some stretch, half way
      // along that stretch, close up
      const full = await page.evaluate(line => {
        const tw = window.tw, r = tw.renderer, b = r.board, w = tw.world.value;
        const l = w.save.lines.find(l => l.num === line), cars = r.net.lineCars?.[line] ?? 10, n = l.stops.length;
        for (let i = 0; i < r.tripCount; i++) {
          const prof = r.trips[i * 2];
          if (Math.floor(prof / 6) !== line || Math.floor(prof / 3) % 2 !== 0 || b.offs[i * 2] < 0) continue;
          const dep = r.trips[i * 2 + 1] + r.tripsEpoch;
          if (dep < tw.clock.now()) continue;
          for (let k = 0; k < n - 1; k++) {
            if (b.loads[b.offs[i * 2] + k] < 160 * cars - 0.5) continue;
            const times = tw.world.value.lineStats[l.id]?.fromStartByLevel?.[prof % 3] ?? null;
            const a = w.save.stations.find(s => s.id === l.stops[k]), c = w.save.stations.find(s => s.id === l.stops[k + 1]);
            return { dep, k, times, at: [(a.lng + c.lng) / 2, (a.lat + c.lat) / 2] };
          }
        }
        return null;
      }, line);
      res.fullTrain = full && { k: full.k, dep: full.dep };
      if (full) {
        const t = full.dep + (full.times ? (full.times[full.k] + full.times[full.k + 1]) / 2 + 15 : 0);
        await page.evaluate(([t, at]) => { window.tw.map.jumpTo({ center: at, zoom: 15.6 }); window.tw.clock.set(t); window.tw.overlay.invalidate(); }, [t, full.at]);
        await page.waitForTimeout(2500);
        await page.screenshot({ path: join(ART, `T-207-full-train${TAG}.png`) });
      }
    }
  }
  out.errors = errors.filter(e => !/favicon|DevTools/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', `t207-browser${TAG}.json`), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
