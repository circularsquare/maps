// T-2986: what one platform at a dead end without a terminal does to the real save
// (T-007-real-nyc.save, 109 lines). Imports it into a ?new=1 sandbox game, waits for a demand final,
// counts the line ends (dead ends on double track with and without a terminal, through stations,
// single track), reads each line's riders and the termini dots at high demand; then builds a terminal
// at every double-track dead end a line ends at (what a free one on load would give: the rule
// before T-2986) and reads them again. Writes artifacts/trainworld/t2986-real-ends.json. From the
// maps root:
//   node trainworld/checks/t2986-real-ends.mjs [url]
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { ARTIFACTS, REAL_SAVE, launchChrome, collectErrors, importSave, demandFinal } from './lib.mjs';

const url = process.argv[2] ?? 'http://localhost:6460/';
const chrome = await launchChrome({ prefix: 'tw-t2986-', size: [1400, 900] });
const errors = [];
try {
  const page = await chrome.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${url}?debug=1&new=1&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 100, undefined, { timeout: 180000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await demandFinal(page, { polling: 500, lines: 100, timeout: 600000 });

  /** the line ends, each line's riders and the termini dots, as the game stands */
  const read = () => page.evaluate(() => {
    const tw = window.tw, w = tw.world.value, v = window.twDemand.view();
    const nodeAt = new Map();
    for (const n of w.nodes.values()) nodeAt.set(`${Math.round(n.x)},${Math.round(n.y)}`, n);
    const edgesAt = new Map();
    for (const e of w.edges) for (const k of [e.a, e.b]) { if (!edgesAt.has(k)) edgesAt.set(k, []); edgesAt.get(k).push(e); }
    const stationOf = new Map(w.save.stations.map((s) => [s.id, s]));
    const ends = { deadNoTerminal: 0, deadTerminal: 0, deadSingle: 0, through: 0, other: 0 };
    const deadNodes = new Set(), perLine = [];
    for (const l of w.save.lines) {
      if (l.bus || l.ferry || l.stops.length < 2 || l.stops[0] === l.stops[l.stops.length - 1]) continue;
      let own = 0;
      for (const sid of [l.stops[0], l.stops[l.stops.length - 1]]) {
        const s = stationOf.get(sid);
        const n = s && nodeAt.get(`${Math.round(s.x)},${Math.round(s.y)}`);
        if (!n) { ends.other++; continue; }
        const built = (edgesAt.get(n.id) ?? []).filter((e) => e.built);
        if (n.builtPorts > 1) { ends.through++; continue; }
        if (!built.length || built.every((e) => e.tracks === 1)) { ends.deadSingle++; continue; }
        if ((s.upgrades ?? 0) & 1) ends.deadTerminal++;
        else { ends.deadNoTerminal++; own++; deadNodes.add(n.id); }
      }
      perLine.push({ id: l.id, name: l.name, ends: own, riders: Math.round(v.lines.get(l.id)?.ridersPerDay ?? 0) });
    }
    const dots = (w.capacity[0] ?? []).filter((m) => m.kind === 3 || m.kind === 14).map((m) => ({ kind: m.kind, rho: +m.rho.toFixed(3), delayS: Math.round(m.delayS), tph: +m.tph.toFixed(1), held: m.held }));
    return { ends, deadNodes: [...deadNodes], perLine, riders: perLine.reduce((a, l) => a + l.riders, 0), railTrips: Math.round(v.railTripsPerDay), version: v.version, dots };
  });
  const now = await read();
  console.log('line ends', JSON.stringify(now.ends), 'dead-end nodes without a terminal', now.deadNodes.length);
  console.log('now: riders', now.riders, 'rail trips', now.railTrips, 'termini dots', now.dots.length, 'over', now.dots.filter((d) => d.rho > 1).length, 'kind 14', now.dots.filter((d) => d.kind === 14).length);
  const version = now.version;
  const up = await page.evaluate(async (nodes) => (await window.tw.client.edit({ op: 'stationUpgrade', nodes, bit: 1, on: true }))?.ok, now.deadNodes);
  console.log('terminals built at', now.deadNodes.length, 'dead ends:', up);
  await page.waitForTimeout(3000);
  await demandFinal(page, { polling: 500, after: version, timeout: 600000 });
  const all = await read();
  console.log('with terminals: riders', all.riders, 'rail trips', all.railTrips, 'termini dots', all.dots.length, 'over', all.dots.filter((d) => d.rho > 1).length, 'kind 14', all.dots.filter((d) => d.kind === 14).length);
  const byId = new Map(all.perLine.map((l) => [l.id, l]));
  const lose = now.perLine.map((l) => ({ name: l.name, ends: l.ends, now: l.riders, withTerminals: byId.get(l.id)?.riders ?? 0 }))
    .map((l) => ({ ...l, loss: l.withTerminals - l.now, pct: l.withTerminals ? +((100 * (l.withTerminals - l.now)) / l.withTerminals).toFixed(1) : 0 }))
    .sort((a, b) => b.loss - a.loss);
  console.log('riders without terminals vs with:', now.riders, 'vs', all.riders, `(${(100 * (now.riders - all.riders) / all.riders).toFixed(2)}%)`);
  console.log('rail trips:', now.railTrips, 'vs', all.railTrips, `(${(100 * (now.railTrips - all.railTrips) / all.railTrips).toFixed(2)}%)`);
  console.log('most lost:', JSON.stringify(lose.slice(0, 12)));
  await writeFile(join(ARTIFACTS, 't2986-real-ends.json'), JSON.stringify({ ends: now.ends, deadNodes: now.deadNodes.length, now: { riders: now.riders, railTrips: now.railTrips, dots: now.dots }, withTerminals: { riders: all.riders, railTrips: all.railTrips, dots: all.dots }, lines: lose }, null, 1));
  console.log(errors.length ? `console errors: ${errors.slice(0, 3).join(' | ')}` : 'no console errors');
} finally {
  await chrome.close();
}
