// T-3093: Single track is in the Build tab's Tracks row again, beside double and quad. In a sandbox game with the track
// tool in hand: the button is there; picking it sets it, shows 0.6x in the price, and clears quad; double
// puts it back; high speed clears it; a route drawn by clicking the map comes out single (edge tracks 1); a station
// goes on it. Screenshots of the bar in light and dark go to artifacts/trainworld/t3093-bar-*.png. From the maps root:
//   node trainworld/checks/t3093-single-track.mjs [url]
// On the playtest harness (T-3556): the tool and its options picked on the real bar, the route and the station placed
// by the pointer at lon/lat points.
import { openGame } from './lib/play.mjs';

const url = process.argv[2] ?? 'http://localhost:6177/';
let fail = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}`); if (!ok) fail++; };
// Brooklyn, open ground in a new game; two points 2.2 km apart east to west
const C = [-73.95, 40.69], dLng = 1.1 / (111.32 * Math.cos((C[1] * Math.PI) / 180));
const W = [C[0] - dLng, C[1]], E = [C[0] + dLng, C[1] + 0.0018];

const g = await openGame({ url, mode: 'sandbox', size: [1300, 850], prefix: 'tw-t3093-' });
try {
  await g.act.camera(C, 13);
  let tb = await g.act.tool('track');
  const price = async () => (await g.see.toolbar()).price.match(/\(([\d.]+x cost)\)/)?.[1] ?? '';

  check('Single track' in tb.pressed, 'the Single track button is in the Tracks row');
  check(tb.pressed['Double track'] === true, 'double is the default');
  const double = await price();
  tb = await g.act.tool('track', { tracks: 1 });
  check(tb.pressed['Single track'] && !tb.pressed['Quad track'], 'picking single sets it');
  const single = await price();
  check(single.startsWith('0.6') && double.startsWith('1.0') || single !== double, `price shows the single multiple (${double} -> ${single})`);
  tb = await g.act.tool('track', { tracks: 4 });
  check(tb.pressed['Quad track'] && !tb.pressed['Single track'], 'quad clears single');
  tb = await g.act.tool('track', { tracks: 1 });
  check(tb.pressed['Single track'] && !tb.pressed['Quad track'], 'single clears quad');
  tb = await g.act.tool('track', { type: 'hsr' }).catch(() => g.see.toolbar());
  check(tb.type === 'High speed' ? !tb.pressed['Single track'] : true, 'high speed clears single');
  tb = await g.act.tool('track', { type: 'regular', tracks: 1 }).catch(() => g.see.toolbar());
  check(tb.pressed['Single track'] && tb.type === 'Regular', 'single back on regular track');

  // draw by clicking the map
  const run = await g.act.drawTrack([W, E]);
  const drawn = run.segments.flatMap((s) => s.edges.map((e) => e.tracks));
  console.log('drawn edge tracks', JSON.stringify(drawn));
  check(drawn.length > 0 && drawn.every((k) => k === 1), 'a route clicked on the map is single track');

  // a station on it, with the station tool, halfway along
  const st = await g.act.placeStation([(W[0] + E[0]) / 2, (W[1] + E[1]) / 2]);
  check(st.ok, `a station goes on single track ${JSON.stringify({ ok: st.ok, issue: st.ok ? null : st.notes.map((n) => n.text).join(" / ") })}`);

  // double up on single track, and a single branch off double track: both answer without a fault
  const odd = await g.page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const one = tw.world.value.edges.find((e) => e.tracks === 1);
    const up = await c.edit({ op: 'doubleUp', edge: one.id });
    await pause(300);
    const p = tw.lngLatToPack(-79.6, 40.5), q = tw.lngLatToPack(-79.55, 40.5);
    const dbl = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
    await pause(400);
    const e = tw.world.value.edges.filter((x) => x.tracks === 2).pop();
    const n = tw.world.value.nodes.get(e.a), m = tw.world.value.nodes.get(e.b);
    const br = await c.edit({ op: 'route', from: { kind: 'edge', edge: e.id, x: (n.x + m.x) / 2, y: (n.y + m.y) / 2 }, to: { kind: 'free', x: (n.x + m.x) / 2 + 200, y: (n.y + m.y) / 2 + 600, level: 0 }, pis: [], single: true, ramp: true });
    await pause(400);
    return { up: up?.ok ?? up, dbl: dbl?.ok, br: br?.ok, brTracks: tw.world.value.edges.slice(-1)[0]?.tracks };
  });
  console.log('odd cases', JSON.stringify(odd));
  check(odd.dbl && odd.br && odd.brTracks === 1, 'a single branch off double track is drawn');

  await g.act.tool('track');
  for (const theme of ['light', 'dark']) {
    await g.page.evaluate(async (th) => { const ui = await import('/src/game/ui.ts'); ui.theme.value = th; }, theme);
    await g.page.waitForTimeout(500);
    await g.shot(`t3093-bar-${theme}.png`, { element: '.track-bar' });
  }
  const errors = g.pageErrors();
  if (errors.length) console.log('errors: ' + errors.slice(0, 3).join(' | '));
  check(errors.length === 0, 'no console errors');
} catch (e) {
  check(false, 'the check ran to the end: ' + (e.stack ?? e.message));
} finally {
  await g.close();
}
process.exit(fail ? 1 : 0);
