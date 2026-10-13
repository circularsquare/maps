// T-230 browser check: why a platform leaves hundreds behind while the line panel said its fullest
// train was 91%. A new game with a line like Anita's G2 (21 stops from Church Av in Brooklyn up
// the G's route through Greenpoint, over the East River to Penn Station, elevated, 100 m platforms
// so 5-car trains, 8 / 4 / 2 trains). After demand's third round, at 08:00-09:00: demand's
// forecast figure (`fullestPct`, the busiest hour's average train), the per-train account's day
// for the line (`lineAccount`), the platform leaving the most riders behind, and the trains
// leaving it: riders aboard as they arrive and leave, and riders left. Screenshots of the line
// panel and that station's panel.
// Needs a Vite dev server (BASE, default http://localhost:6932; file watching off) and Playwright
// (PLAYWRIGHT_MODULE). PROFILE_DIR: where the throwaway Chrome profile goes. TAG names the run.
//   node trainworld/notes/t230-browser.mjs
// Results: artifacts/trainworld/notes-outputs/t230-browser<TAG>.json; screenshots artifacts/trainworld/T-230-*<TAG>.png.
import { writeFile, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const here = dirname(fileURLToPath(import.meta.url));
const ART = join(here, '..', '..', 'artifacts', 'trainworld');
const BASE = process.env.BASE ?? 'http://localhost:6932';
const TAG = process.env.TAG ?? '';
const FLEET = JSON.parse(process.env.FLEET ?? '[8,4,2]');
// Church Av to Court Sq along the G, then over the river to 34 St and Penn Station
const STOPS = [[-73.9797, 40.6443], [-73.9757, 40.6506], [-73.9799, 40.6604], [-73.9898, 40.6702], [-73.9952, 40.6800], [-73.9909, 40.6862], [-73.9850, 40.6884], [-73.9754, 40.6873], [-73.9667, 40.6884], [-73.9600, 40.6889], [-73.9536, 40.6895], [-73.9493, 40.6946], [-73.9502, 40.7003], [-73.9502, 40.7061], [-73.9514, 40.7128], [-73.9513, 40.7245], [-73.9541, 40.7313], [-73.9493, 40.7443], [-73.9456, 40.7470], [-73.9740, 40.7470], [-73.9935, 40.7506]];
const profile = await mkdtemp(join(process.env.PROFILE_DIR ?? tmpdir(), 't230-'));
const ctx = await chromium.launchPersistentContext(profile, { executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, viewport: { width: 1400, height: 900 }, args: ['--use-angle=d3d11', '--ignore-gpu-blocklist'] });
const out = {};
try {
  const page = await ctx.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(BASE + '/?debug=1&new=1&paused=1&v=' + Date.now());
  await page.waitForFunction(() => window.tw?.world.value && window.twDemand?.client && window.twDemand.view()?.round === 3, undefined, { timeout: 300000 });
  const v0 = await page.evaluate(() => window.twDemand.view().version);
  const day = await page.evaluate(() => Math.floor(window.tw.clock.now() / 86400) * 86400 + 86400);
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
    const names = ['Church Av', 'Fort Hamilton Pkwy', '15 St', '4 Av', 'Carroll St', 'Bergen St', 'Hoyt-Schermerhorn', 'Fulton St', 'Clinton-Washington', 'Classon Av', 'Bedford-Nostrand', 'Myrtle-Willoughby', 'Flushing Av', 'Broadway', 'Metropolitan Av', 'Nassau Av', 'Greenpoint Av', '21 St', 'Court Sq', '34 St', 'Penn Station'];
    for (const [i, n] of nodes.entries()) await run({ op: 'addStation', at: { kind: 'node', node: n }, platform: 100, name: names[i] });
    const line = (await run({ op: 'addLine', stops: nodes, name: 'G2', colour: '#6cbe45' })).line;
    await run({ op: 'setTrainCount', line, trains: fleet });
    await run({ op: 'constructAll' });
    return line;
  }, [STOPS, FLEET]);
  await page.waitForFunction(v => window.twDemand.view()?.version > v && window.twDemand.view().round === 3 && window.twDemand.view().lines.size === 1, v0, { timeout: 300000 });
  await page.waitForTimeout(2000);
  out.line = await page.evaluate(line => {
    const w = window.tw.world.value, l = w.save.lines.find(l => l.num === line), st = w.lineStats[l.id], d = window.twDemand.view().lines.get(l.id);
    return { stops: l.stops.length, km: +st.km.toFixed(1), cars: st.cars, trains: st.trainsNeeded, tph: l.tph, ridersPerDay: Math.round(d.ridersPerDay), forecastFullestPct: d.fullestPct };
  }, line);
  out.account = await page.evaluate(async line => Array.from(await window.tw.client.query({ what: 'lineAccount', line }) ?? []), line);
  // 08:00-09:00: the platform leaving the most riders behind, and the trains leaving it
  await page.evaluate(t => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(t); tw.overlay.invalidate(); }, day + 8 * 3600 + 1800);
  await page.waitForFunction(t => window.tw.renderer.board && window.tw.renderer.board.epoch === Math.floor(t / 3600) * 3600, day + 8 * 3600 + 1800, { timeout: 120000 });
  await page.waitForTimeout(1000);
  out.worst = await page.evaluate(line => {
    const tw = window.tw, r = tw.renderer, b = r.board, w = tw.world.value, l = w.save.lines.find(l => l.num === line);
    let best = -1;
    for (let q = 0; q < b.queueCount; q++) if (b.queues[q * 10 + 1] === line && (best < 0 || b.queues[q * 10 + 5] > b.queues[best * 10 + 5])) best = q;
    const Q = b.queues.subarray(best * 10, best * 10 + 10);
    const node = Q[0], run = Q[2];
    const name = w.save.stations.find(s => s.num === node)?.name;
    const n = l.stops.length, k = (run === 0 ? l.stops.map(Number) : l.stops.map(Number).reverse()).indexOf(node);
    const towards = w.save.stations.find(s => s.id === (run === 0 ? l.stops.at(-1) : l.stops[0]))?.name;
    const crush = 160 * (r.net.lineCars?.[line] ?? 5);
    // the hour's trains of that direction: aboard arriving (the stretch before) and leaving
    const trains = [];
    for (let i = 0; i < r.tripCount; i++) {
      const prof = r.trips[i * 2];
      if (Math.floor(prof / 6) !== line || Math.floor(prof / 3) % 2 !== run || b.offs[i * 2] < 0) continue;
      const dep = r.trips[i * 2 + 1];
      const o = b.offs[i * 2];
      const st = s => (run === 0 ? s : n - 2 - s);
      trains.push({ leavesFirstStop: Math.round(dep / 60), arriving: k > 0 ? Math.round(b.loads[o + st(k - 1)]) : 0, leaving: Math.round(b.loads[o + st(k)]) });
    }
    const E = b.events, ev = [];
    for (let j = Q[8]; j < Q[8] + Q[9]; j++) ev.push({ min: Math.round(E[j * 2] / 60), left: Math.round(E[j * 2 + 1]) });
    return { station: name, node, towards, crush, arrivingThisHour: Math.round(Q[7]), leftBehindThisHour: Math.round(Q[5]), longestWaitMin: Math.round(Q[6] / 60), trainsLeaving: ev, trains: trains.sort((a, b) => a.leavesFirstStop - b.leavesFirstStop) };
  }, line);
  console.log(JSON.stringify(out));
  // the line panel, then the station's
  await page.evaluate(line => { const w = window.tw.world.value, l = w.save.lines.find(l => l.num === line); window.tw.select({ kind: 'line', line: l.id }); }, line);
  await page.waitForTimeout(1500);
  out.linePanel = await page.evaluate(() => document.querySelector('#dock-root')?.innerText.split('Stops')[0]?.slice(-900) ?? null);
  await page.screenshot({ path: join(ART, `T-230-line${TAG}.png`) });
  // just before the worst train of the hour at that platform leaves
  const worstEv = out.worst.trainsLeaving.filter(e => e.min >= 0 && e.min < 60).sort((a, b) => b.left - a.left)[0];
  await page.evaluate(([node, t]) => { const tw = window.tw; tw.clock.set(t); tw.select({ kind: 'station', station: String(node) }); const n = tw.world.value.nodes.get(node); tw.map.jumpTo({ center: [n.lng, n.lat], zoom: 15 }); tw.overlay.invalidate(); }, [out.worst.node, day + 8 * 3600 + (worstEv?.min ?? 30) * 60 + 30]);
  await page.waitForTimeout(2000);
  out.stationPanel = await page.evaluate(() => document.querySelector('#dock-root')?.innerText.slice(0, 900) ?? null);
  await page.screenshot({ path: join(ART, `T-230-station${TAG}.png`) });
  out.errors = errors.filter(e => !/favicon|DevTools/.test(e)).slice(0, 20);
} finally {
  await ctx.close();
}
await writeFile(join(here, '../../artifacts/trainworld/notes-outputs', `t230-browser${TAG}.json`), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
