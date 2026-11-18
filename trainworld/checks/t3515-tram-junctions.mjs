// T-3515: tram track snapped to the real network takes the turn a real tram line takes at a junction, in
// Zurich, Melbourne, Amsterdam and Milan, and every snapped route the ghost shows as placeable places on the
// click. Each case is two points a real tram service runs between (from its OSM route relation, about 300 or
// 500 m along it, turning at a junction on the way): the snapped route must be found, no longer than `max` m
// (a detour round a terminal loop is kilometres), accepted by the track model and placed. Then the T-3069
// MODE=price sweep: Zurich's drawn tram lines in view at zoom 14, each start and end on the map (not under a
// panel or the track bar), and every ghost that shows as placeable must place on the click.
// GRAPHS=<dir> serves that folder's .rgr files in place of the data pack's (graphs built but not landed).
// About 3 minutes. From the maps root:
//   [GRAPHS=dir] [ONLY=zurich,melbourne] node trainworld/checks/t3515-tram-junctions.mjs [url]
import { readFile, readdir } from 'node:fs/promises';
import { launchGames } from './lib/play.mjs';

const url = process.argv[2] ?? 'http://localhost:6194/';
const only = process.env.ONLY?.split(',');
// [lon, lat] pairs; `line` the OSM route's ref, for the record
const CASES = [
  // Bahnhofbrucke to Museumstrasse over the Bahnhofquai track (OSM service=siding while the stop is rebuilt;
  // without it the route went round the Bucheggplatz loop, kilometres)
  { city: 'zurich', line: '4', a: [8.5428, 47.37692], b: [8.5394, 47.3789], max: 1000 },
  { city: 'zurich', line: '10', a: [8.545516, 47.378626], b: [8.544278, 47.377946], max: 450 },
  { city: 'melbourne', line: '1', a: [144.967732, -37.796227], b: [144.964403, -37.798206], max: 700 },
  // line 82's single track round street corners: curves the graph's old 3 m thinning cut into long chords
  { city: 'melbourne', line: '82', a: [144.900889, -37.801463], b: [144.89839, -37.798225], max: 700 },
  { city: 'melbourne', line: '82 north', a: [144.893982, -37.792972], b: [144.889582, -37.790723], max: 700 },
  { city: 'amsterdam', line: '1', a: [4.88076, 52.363017], b: [4.883698, 52.362587], max: 450 },
  { city: 'amsterdam', line: '14', a: [4.899427, 52.37677], b: [4.900948, 52.377752], max: 450 },
  { city: 'milan', line: '14', a: [9.18466, 45.468873], b: [9.183391, 45.470093], max: 450 },
  { city: 'milan', line: '24', a: [9.189715, 45.459816], b: [9.18964, 45.460948], max: 450 },
];
const SWEEP = { zurich: [8.54, 47.38] };

const games = await launchGames({ prefix: 'tw-t3515-' });
let failed = 0;
const fail = (msg) => { failed++; console.log('FAIL', msg); };
try {
  if (process.env.GRAPHS) {
    let served = 0;
    for (const f of await readdir(process.env.GRAPHS)) if (f.endsWith('.rgr')) {
      const body = await readFile(process.env.GRAPHS + '/' + f);
      // the context's route: the graphs are fetched by the router's worker, not the page
      await games.chrome.ctx.route('**/realgraph/' + f, (r) => { served++; return r.fulfill({ status: 200, body, contentType: 'application/octet-stream' }); });
    }
    process.on('exit', () => console.log(`graphs served from ${process.env.GRAPHS}: ${served}`));
  }
  const g = await games.open({ url, mode: 'sandbox' });
  // the real network drawn whole while snapping (the layer's All setting), as T-3069's checks run it
  await g.page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.display.value = { ...ui.display.value, realNetwork: 'all' }; });
  const tram = () => g.act.tool('track', { type: 'tram', snap: { real: true } });
  // the router's route at the pointer: its length once found (phase done), else the phase ("none": nothing joins)
  const routeAt = () => g.page.evaluate(() => { const r = window.tw.tools.real; return { phase: r.spot?.phase ?? null, len: r.spot?.phase === 'done' ? r.last?.lengthM ?? null : null }; });
  // the snap's ring needs the drawn real network under both points: wait for its tiles
  // each point moved onto the nearest drawn tram line within 60 px (the case points come from OSM's route ways,
  // which the drawn tiles may place a few metres off); null until those tiles are drawn
  const onDrawn = (pts) => g.page.evaluate((pts) => pts.map((q) => {
    const map = window.tw.map, p = map.project(q);
    const layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real-')).map((l) => l.id);
    let best = null;
    for (const f of map.queryRenderedFeatures([[p.x - 60, p.y - 60], [p.x + 60, p.y + 60]], { layers })) {
      if (f.properties.k !== 'tram') continue;
      const ls = f.geometry.type === 'LineString' ? [f.geometry.coordinates] : f.geometry.type === 'MultiLineString' ? f.geometry.coordinates : [];
      for (const l of ls) for (let i = 1; i < l.length; i++) {
        const a = map.project(l[i - 1]), b = map.project(l[i]), dx = b.x - a.x, dy = b.y - a.y, l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2)) : 0;
        const x = a.x + t * dx, y = a.y + t * dy, d = Math.hypot(x - p.x, y - p.y);
        if (!best || d < best.d) best = { d, x, y };
      }
    }
    if (!best || !window.tw.tools.real.ring(best.x, best.y)) return null;
    return map.unproject([best.x, best.y]).toArray();
  }), pts);
  const ringsAt = async (pts) => {
    for (let k = 0; k < 60; k++) {
      const got = await onDrawn(pts);
      if (got.every(Boolean)) return got;
      await g.page.waitForTimeout(500);
    }
    return null;
  };

  for (const c of CASES) {
    if (only && !only.includes(c.city)) continue;
    await g.act.camera([(c.a[0] + c.b[0]) / 2, (c.a[1] + c.b[1]) / 2], 16);
    await tram();
    const tag = `${c.city} line ${c.line}`;
    const ab = await ringsAt([c.a, c.b]);
    if (!ab) { fail(`${tag}: no drawn tram line within 60 px of both points`); continue; }
    let at = { phase: null, len: null };
    const run = await g.act.drawTrack(ab, { before: async (i) => { if (i === 1) at = await routeAt(); } });
    const s = run.segments[0];
    const kinds = [...new Set((s?.edges ?? []).map((e) => e.kind))];
    const len = at.len;
    console.log(tag.padEnd(18), 'route', len === null ? at.phase : Math.round(len) + ' m', '(max ' + c.max + ')', 'ghost', s?.ghost ? Math.round(s.ghost.lengthM) + ' m ' + (s.ghost.ok ? 'ok' : 'refused') : '-', 'placed', s?.ok ? s.edges.length + ' edges ' + JSON.stringify(kinds) : 'no ' + JSON.stringify(s?.notes));
    if (len === null || len > c.max) fail(`${tag}: the snapped route is ${len === null ? 'not found (' + at.phase + ')' : Math.round(len) + ' m'}, over ${c.max} m: the junction's turn was not taken`);
    else if (!s?.ghost?.ok) fail(`${tag}: the track model refused the route`);
    else if (!s.ok) fail(`${tag}: an accepted route did not place`);
  }

  for (const [city, center] of Object.entries(SWEEP)) {
    if (only && !only.includes(city)) continue;
    await g.act.camera(center, 14);
    await tram();
    await g.page.waitForTimeout(8000);
    // each drawn tram feature in view: its first and last points with 80 to 400 px between
    const cands = await g.page.evaluate(() => {
      const map = window.tw.map;
      const layers = map.getStyle().layers.filter((l) => l.id.startsWith('tw-real-')).map((l) => l.id);
      const out = [];
      for (const f of map.queryRenderedFeatures(undefined, { layers })) {
        if (f.properties.k !== 'tram' || f.geometry.type !== 'LineString') continue;
        const pts = f.geometry.coordinates.map((q) => map.project(q));
        if (pts.length < 3) continue;
        const a = pts[0], b = pts[pts.length - 1], d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 80 || d > 400 || !window.tw.tools.real.ring(a.x, a.y) || !window.tw.tools.real.ring(b.x, b.y)) continue;
        out.push([map.unproject(a).toArray(), map.unproject(b).toArray()]);
      }
      return out;
    });
    let tried = 0, shown = 0, placed = 0, covered = 0;
    for (const [a, b] of cands) {
      if (tried >= 8) break;
      const sc = await g.toScreen([a, b]);
      if (!sc.every((p) => p.onMap)) { covered++; continue; }
      tried++;
      let at = { phase: null };
      const run = await g.act.drawTrack([a, b], { before: async (i) => { if (i === 1) at = await routeAt(); } });
      const s = run.segments[0];
      // placeable: the snapped route found and its ghost accepted (a red "no track" ghost keeps the last ok preview)
      if (at.phase === 'done' && s?.ghost?.ok) {
        shown++;
        if (s.ok) placed++;
        else fail(`${city} sweep: a route shown as placeable (${Math.round(s.ghost.lengthM)} m) did not place on the click, ${a} to ${b}`);
      }
      await g.act.camera(center, 14);
    }
    console.log(`${city} sweep: ${cands.length} candidates, ${covered} under a panel or bar left out, ${tried} tried, ${shown} placeable, ${placed} placed`);
    if (!shown) fail(`${city} sweep: no placeable route to try`);
  }
  const errs = g.pageErrors();
  if (errs.length) fail('page errors: ' + errs.slice(0, 3).join(' | '));
} finally {
  await games.close();
}
console.log(failed ? `${failed} failed` : 'all passed');
process.exitCode = failed ? 1 : 0;
