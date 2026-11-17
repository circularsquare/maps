// T-3556: the playtest harness (lib/play.mjs) checks itself. A new game in New York: the pointer's lon/lat
// at a projected point is the point asked for; the Build tab's tool and the track bar's options picked by name;
// double track drawn through three points, then single track; stations placed with the station tool; Construct
// blueprints; a line made with New line, named, three stops; its trains typed into the panel's three rows and read
// back from the schedule table; the money figure; a press under a map button or off the map refused; a crop of the panel and of a
// lon/lat box. Then the real save (T-007-real-nyc.save) opened through Saved games, Open a file: 109 lines and
// 1,756 stations, a station pressed on the map opens its panel, demand reaches a final. Between them, by touch (a
// touch screen, taps): two segments of track and a station.
// From the tree root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6139):
//   node trainworld/checks/t3556-play.mjs [dev url]      (about 5 minutes; NEW=0, TOUCH=0, SAVE=0 skip a part)
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { MAPS, REAL_SAVE } from './lib.mjs';
import { openGame } from './lib/play.mjs';
import { STATE } from './lib/reads.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6139/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

// Brooklyn, east of Prospect Park: open ground in a new game
const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32]; // km east, north

let g;
if (process.env.NEW !== '0') {
g = await openGame({ url: BASE, city: 'nyc', ui: { shade: 'light', display: { dayShown: 'weekday' }, v: 2 }, prefix: 'tw-t3556-' });
try {
  await g.act.camera(C, 13);
  // the projection: the pointer the tool records at a projected point is that point
  await g.act.tool('track', { type: 'regular', tracks: 2, snap: { real: false, roads: false, beside: false }, keepPlacing: false });
  const probe = P(0.7, 0.4);
  const [s] = await g.toScreen([probe]);
  await g.driver.move(s.x, s.y);
  await g.page.waitForTimeout(300);
  const ptr = (await g.state.drawing()).pointer;
  // the browser puts a mouse event on whole pixels, so a point is exact to under a pixel on each axis
  const [q] = ptr ? await g.toScreen([ptr]) : [{ x: Infinity, y: Infinity }];
  check(Math.abs(q.x - s.x) <= 0.5 + 1e-6 && Math.abs(q.y - s.y) <= 0.5 + 1e-6, 'the pointer the tool records at a projected point is that point to the pixel', `off by ${(q.x - s.x).toFixed(2)}, ${(q.y - s.y).toFixed(2)} px`);
  const back = (await g.toLngLat([[s.x, s.y]]))[0];
  check(Math.abs(back[0] - probe[0]) < 1e-7 && Math.abs(back[1] - probe[1]) < 1e-7, 'toLngLat undoes toScreen', JSON.stringify(back));
  const tb = await g.see.toolbar();
  check(tb.tool === 'Track' && tb.pressed['Double track'] && tb.type === 'Regular', 'the Build tab\'s Track tool and the bar\'s Regular, Double track', JSON.stringify(tb));

  // double track through three points, one run
  const A = P(-1.5, 0), B = P(0, 0.3), D = P(1.5, 0);
  const run = await g.act.drawTrack([A, B, D]);
  const ends = run.segments.map((x) => x.edges.map((e) => e.tracks).join('')).join(' ');
  check(run.ok && run.segments.every((x) => x.edges.length > 0 && x.edges.every((e) => e.tracks === 2)), 'two segments of double track drawn by the pointer', ends);
  // single track beside it
  await g.act.tool('track', { tracks: 1 });
  check((await g.see.toolbar()).pressed['Single track'], 'Single track picked on the bar');
  const one = await g.act.drawTrack([P(-1.5, -1), P(1.5, -1)]);
  check(one.ok && one.segments[0].edges.every((e) => e.tracks === 1), 'a single track segment', JSON.stringify(one.segments[0].edges.map((e) => e.tracks)));

  // stations on the double track, construct, a line through them
  const st = [];
  for (const p of [A, B, D]) st.push(await g.act.placeStation(p));
  check(st.every((x) => x.ok), 'three stations placed with the station tool', st.map((x) => x.name ?? JSON.stringify(x.notes)).join(', '));
  await g.act.construct();
  const unbuilt = (await g.state.edgesFrom(0)).filter((e) => e.built === false).length;
  check(unbuilt === 0, 'Construct blueprints builds every piece', `${unbuilt} unbuilt`);
  const line = await g.act.newLine([A, B, D], { name: 'Harness', zoom: 15 });
  check(line?.name === 'Harness' && line.stops.length === 3, 'New line, named, three stops', JSON.stringify(line));
  const panel = await g.see.panel();
  check(panel?.title === 'Harness', 'the line\'s panel is open', panel?.title);
  const sched = await g.act.setTrains([3, 2, 2]);
  const trains = sched?.schedule.map((r) => r.trains).join(' ');
  check(trains === '3 2 2', 'trains typed into the schedule rows read back from the table', JSON.stringify(sched?.schedule));
  check(sched.schedule.every((r) => /min/.test(r.headway)), 'each row shows a headway', sched.schedule.map((r) => r.headway).join(', '));
  const money = await g.see.money();
  check(/^\$[\d.]+[MB]$/.test(money?.cash ?? ''), 'the money figure', money?.cash);
  // a press where the map's layer buttons cover it is refused, naming what is there; one off the map too
  const box = await g.page.locator('button:text-is("Destinations")').first().boundingBox();
  const [under] = await g.toLngLat([[box.x + box.width / 2, box.y + box.height / 2]]);
  const refused = await g.act.press(under).then(() => '', (e) => e.message);
  check(/can't press .* is there/.test(refused) && !/outside/.test(refused), 'a press under a map button is refused', refused.slice(0, 140));
  const off = await g.act.press(P(-40, 0)).then(() => '', (e) => e.message);
  check(/outside the view/.test(off), 'a press off the map is refused', off.slice(0, 100));
  const f1 = await g.shot('t3556_panel.png', { element: '#inspector' });
  await g.act.camera(C, 13);
  const clip = await g.page.evaluate(STATE.clipOf, { box: [P(-2, -1.5), P(2, 1)], pad: 10 });
  const f2 = await g.shot('t3556_box.png', { box: [P(-2, -1.5), P(2, 1)], pad: 10 });
  // the box is 4 km by 2.5 km: about 550 by 345 px at zoom 13 here, with the pad
  check(existsSync(f1) && existsSync(f2) && clip.width > 500 && clip.width < 600 && clip.height > 300 && clip.height < 400, 'crops of the panel and of a lon/lat box', `${f1}, ${f2}; box ${Math.round(clip.width)} x ${Math.round(clip.height)} px`);
  const errs = g.pageErrors();
  check(errs.length === 0, 'no console errors (new game)', errs.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the new game part ran to the end', e.stack ?? e.message);
} finally {
  await g.close();
}
}

if (process.env.TOUCH !== '0') {
  g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', touch: true, prefix: 'tw-t3556t-' });
  try {
    await g.act.camera(C, 13);
    await g.act.tool('track', { tracks: 2 });
    const run = await g.act.drawTrack([P(-1.5, 0), P(0, 0.3), P(1.5, 0)]);
    const left = (await g.state.drawing()).start;
    check(run.ok && run.segments.every((x) => x.edges.length > 0) && !left, 'by touch: two segments tapped (a tap, a second tap to confirm), the run stopped from the touch bar', run.segments.map((x) => x.notes.at(-1)?.text).join(' / '));
    const st = await g.act.placeStation(P(-1.5, 0));
    check(st.ok, 'by touch: a station tapped and confirmed', st.name ?? JSON.stringify(st.notes));
  } catch (e) {
    check(false, 'the touch part ran to the end', e.stack ?? e.message);
  } finally {
    await g.close();
  }
}

if (process.env.SAVE !== '0') {
  const save = existsSync(REAL_SAVE) ? REAL_SAVE : join(MAPS, 'trainworld', 'T-007-real-nyc.save');
  g = await openGame({ url: BASE, save, ui: { shade: 'light', display: { dayShown: 'weekday' }, v: 2 }, prefix: 'tw-t3556s-' });
  try {
    const n = await g.state.counts();
    check(n.lines === 109 && n.stations === 1756, 'the real save through Saved games, Open a file', `${n.lines} lines, ${n.stations} stations, ${n.trackKm} km`);
    const [times] = await g.state.stations({ near: [-73.9866, 40.7553], limit: 1 });
    await g.act.camera([times.lng, times.lat], 15.5);
    const p = await g.act.selectAt([times.lng, times.lat]);
    const sel = await g.state.selection();
    check(sel?.kind === 'station' && !!p?.title, 'a station pressed on the map opens its panel', `${times.name}: ${p?.title} / ${p?.subtitle}`);
    await g.waitDemand({ timeout: 300000 });
    check((await g.state.demand()).final, 'demand reaches a final on the real save');
    await g.shot('t3556_station.png', { element: '#inspector' });
    const errs = g.pageErrors();
    check(errs.length === 0, 'no console errors (real save)', errs.slice(0, 3).join(' | '));
  } catch (e) {
    check(false, 'the real save part ran to the end', e.stack ?? e.message);
  } finally {
    await g.close();
  }
}
console.log(fail ? `T-3556 FAILED: ${fail}` : 'T-3556 PASSED');
process.exit(fail ? 1 : 0);
