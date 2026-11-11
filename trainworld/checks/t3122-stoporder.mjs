// T-3122, T-3140, T-3139: stop order on lines with loops, on Anita's Amsterdam save. A new line clicked once per
// station of Line 1 (a stem with a loop at each end) with real clicks on the stop markers stays an open line run both
// ways (a click never loops a line, T-3140); a right click on each marker offering a turn then runs it round both
// loops one way, stopping at every stem station both ways and at each loop station once. The same by edits in
// another order. Then, for Line 1, Line 1a and the new line, the stops the sim says are served one way
// (`LineStats.oneWayAt`) are the half-filled markers while its stops are edited, the half dots in its stop list and the
// chevrons in the station hovers; every marker chevron lies along the track's tangent pointing the way the line's
// trains leave (T-3139), so one-way stops in a row on a plain run point the same way along it.
// Screenshots artifacts/trainworld/t3122-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3122-stoporder.mjs [dev url]   (SHADE=dark)
// About three minutes.
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6183/';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
const SHADE = process.env.SHADE ?? 'light';
/** how far a chevron may lie off the track's tangent, degrees (the drawn polyline's first piece stands in for it) */
const TANGENT_DEG = 4;
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const STEM = ['Johan van Hasseltweg', 'Bankwerkerij', 'Meeuwenlaan', 'De Ruijterkade', 'Mandenmakerssteeg', 'Kromelleboogsteeg', 'Nes', 'Herengracht', 'Weteringschans', 'Hobbemakade'];
const SOUTH = ['Amstelkade', 'President Kennedylaan', 'Europaboulevard', 'Beethovenstraat', 'Parnas', 'Oud-Zuid', 'Apollobuurt'];
const NORTH = ['Pluvierstraat'];
const LOOPS = [...NORTH, ...SOUTH].sort();

const chrome = await launchChrome({ prefix: 'tw-t3122-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);
const out = { when: new Date().toISOString() };
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });

/** a line's stops by name, its loop, and the stations the sim says it serves one way */
const info = (name) => page.evaluate((name) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === name), st = w.lineStats[l.id];
  const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
  return { num: l.num, id: l.id, stops: l.stops.map((s) => nm.get(s)), loop: st.loop, oneWay: st.oneWay, ways: Object.keys(st.oneWayAt).map((s) => nm.get(s)).sort() };
}, name);
const visits = (l, s) => (l.loop ? l.stops.slice(0, -1) : l.stops).filter((x) => x === s).length;
const editStops = (num) => page.evaluate(async (num) => {
  const tw = window.tw, ui = await import('/src/game/ui.ts');
  const l = tw.world.value.save.lines.find((x) => x.num === num);
  tw.selection.value = { kind: 'line', line: l.id };
  ui.tool.value = 'line';
  ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
}, num);
const stopEditing = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = null; });
const centre = (name, zoom) => page.evaluate(([name, zoom]) => {
  const tw = window.tw, s = tw.world.value.save.stations.find((x) => x.name === name), n = tw.world.value.nodes.get(Number(s.id));
  tw.map.jumpTo({ center: tw.packToLngLat ? tw.packToLngLat(n.x, n.y) : [n.lng, n.lat], zoom });
}, [name, zoom]);
const markerOf = (name) => page.evaluate((name) => {
  const tw = window.tw, s = tw.world.value.save.stations.find((x) => x.name === name);
  const m = tw.tools.stops.marks.get(s.id);
  const r = tw.map.getContainer().getBoundingClientRect();
  return m && m.shown ? { x: r.left + m.x, y: r.top + m.y } : null;
}, name);
/** the stations whose marker offers to turn the line round a loop (the right click's tier 7) */
const turnMarkers = () => page.evaluate(() => {
  const tw = window.tw, nm = new Map(tw.world.value.save.stations.map((s) => [s.id, s.name]));
  return [...tw.tools.stops.tiers].filter(([, t]) => t === 'turn').map(([id]) => nm.get(id));
});
/** the stops whose editing marker is half filled, and whose chevron shows */
const halfMarkers = () => page.evaluate(() => {
  const tw = window.tw, nm = new Map(tw.world.value.save.stations.map((s) => [s.id, s.name]));
  const half = [], chev = [];
  for (const [id, m] of tw.tools.stops.marks) {
    if (/gradient/.test(m.el.style.background)) half.push(nm.get(id));
    if (m.chev && m.chev.style.display !== 'none') chev.push(nm.get(id));
  }
  return { half: half.sort(), chev: chev.sort() };
});
/** T-3139: per marker chevron of line `name`, its screen angle, and on screen the way the line's path leaves that
 * station along the drawn track (from the first piece of each leaving edge's polyline): `off` the angle between
 * them, degrees (0 along the way the trains leave, 180 against it), `along` the angle to the track's line either way */
const chevronAngles = (name) => page.evaluate((name) => {
  const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.name === name), net = tw.renderer.net, ov = tw.overlay;
  const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const edges = new Map(w.edges.map((e) => [e.id, e]));
  const norm = (d) => ((d % 360) + 540) % 360 - 180;
  const res = [];
  for (const [id, m] of tw.tools.stops.marks) {
    if (!m.chev || m.chev.style.display === 'none') continue;
    const deg = Number(/rotate\((-?[\d.]+)deg\)/.exec(m.chev.style.transform)?.[1]);
    const node = Number(id), leaving = [];
    for (const p of w.lineStats[l.id].path) {
      const e = edges.get(p >> 1), pts = net.edgePts.get(p >> 1);
      if (!e || !pts) continue;
      const from = (p & 1) === 0 ? e.a : e.b;
      if (Number(from) !== node) continue;
      const n = pts.length / 3;
      const [i, j] = (p & 1) === 0 ? [0, 1] : [n - 1, n - 2];
      const a = ov.toScreen(pts[i * 3], pts[i * 3 + 1]), b = ov.toScreen(pts[j * 3], pts[j * 3 + 1]);
      leaving.push((Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI);
    }
    if (!leaving.length) { res.push({ name: nm.get(id), deg, off: null }); continue; }
    const off = Math.min(...leaving.map((t) => Math.abs(norm(deg - t))));
    res.push({ name: nm.get(id), deg: +deg.toFixed(1), track: leaving.map((t) => +t.toFixed(1)), off: +off.toFixed(1), along: +Math.min(off, 180 - off).toFixed(1) });
  }
  return res;
}, name);
/** the stop list's rows with a half dot, and every row */
const listRows = () => page.evaluate(() => {
  const rows = [...document.querySelectorAll('ul.stops > li[data-stop]')];
  return { half: rows.filter((r) => r.classList.contains('oneway')).map((r) => r.querySelector('.stop-name')?.textContent.trim()).sort(), all: rows.length };
});
/** the stations of a line whose hover shows a chevron beside that line's bullet */
const hoverChevrons = (name) => page.evaluate(async (name) => {
  const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.name === name), net = tw.renderer.net;
  const { stationTip } = await import('/src/map/networkTips.ts');
  const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const out = [];
  for (const id of new Set(l.stops)) {
    const tip = stationTip(tw.overlay, w, net.stationIds.indexOf(id)) ?? [];
    const has = tip.some((t) => t instanceof HTMLElement && [...t.querySelectorAll('.bullet-heading')].some((b) => b.textContent.trim() === l.letter));
    if (has) out.push(nm.get(id));
  }
  return out.sort();
}, name);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const lineDone = (name, want) => page.waitForFunction(([name, want]) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === name); return l && l.stops.length >= want; }, [name, want], { timeout: 15000 }).catch(() => {});

try {
  await page.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await ready();
  const bytes = await readFile(SAVE);
  await page.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
  await page.waitForURL(/load=file/, { timeout: 120000 });
  await ready();
  await page.waitForFunction(() => window.tw.world.value.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
  await demandFinal(page, { timeout: 600000 });
  await page.waitForTimeout(1000);

  // ---- Line 1 as saved: Johan van Hasseltweg was a stop one way only; loading stops it at both passes
  const l1 = await info('Line 1');
  out.line1 = l1;
  check(STEM.every((s) => visits(l1, s) === 2) && LOOPS.every((s) => visits(l1, s) === 1), 'Line 1 as loaded stops at every stem station both ways, each loop station once', JSON.stringify(l1.stops));
  check(same(l1.ways, LOOPS), 'the sim serves Line 1 one way at its loops\' stations only', JSON.stringify(l1.ways));

  // ---- a new line clicked once per station, north to south, with real clicks on the markers
  const first = await page.evaluate(async (name) => {
    const w = window.tw.world.value, s = w.save.stations.find((x) => x.name === name);
    return (await window.tw.client.edit({ op: 'addLine', stops: [Number(s.id)], name: 'Clicked', colour: '#7d3c98' })).line;
  }, NORTH[0]);
  await editStops(first);
  await page.waitForTimeout(1500);
  const order = [...NORTH, ...STEM, ...SOUTH];
  for (const name of order.slice(1)) {
    await centre(name, 14.5);
    await page.waitForTimeout(900);
    const m = await markerOf(name);
    if (!m) { check(false, `marker of ${name} on screen`); continue; }
    await page.mouse.click(m.x, m.y);
    await page.waitForFunction((name) => { const w = window.tw.world.value, s = w.save.stations.find((x) => x.name === name), l = w.save.lines.find((x) => x.name === 'Clicked'); return l.stops.includes(s.id); }, name, { timeout: 15000 }).catch(() => {});
  }
  await page.waitForTimeout(1500);
  const open = await info('Clicked');
  out.clickedOpen = open;
  console.log('clicked', JSON.stringify(open.stops));
  check(open.loop === null && !open.oneWay && open.stops.length === order.length, 'T-3140: clicked once per station, the line stays open, both ways, each station once', `${open.loop} ${open.oneWay} ${open.stops.length} stops`);
  check(open.ways.length === 0, 'no stop of the open line is served one way', JSON.stringify(open.ways));

  // ---- right click each marker that offers a turn round a loop
  const turned = [];
  for (let k = 0; k < 2; k++) {
    await page.waitForTimeout(2000);
    const offers = await turnMarkers();
    if (!offers.length) break;
    await centre(offers[0], 14.5);
    await page.waitForTimeout(1200);
    const m = await markerOf(offers[0]);
    if (!m) { check(false, `marker of ${offers[0]} on screen`); break; }
    const before = (await info('Clicked')).stops.length;
    await page.mouse.click(m.x, m.y, { button: 'right' });
    await lineDone('Clicked', before + 1);
    turned.push(offers[0]);
  }
  await page.waitForTimeout(1500);
  const clicked = await info('Clicked');
  out.clicked = clicked;
  out.turnedAt = turned;
  console.log('turned at', JSON.stringify(turned), JSON.stringify(clicked.stops));
  check(turned.length >= 1, 'a turn offered, right clicked (one turn may close round both loops)', JSON.stringify(turned));
  check(clicked.loop === 'circle' && clicked.oneWay, 'turned at both ends, the line is a circle run one way', `${clicked.loop} ${clicked.oneWay}`);
  check(STEM.every((s) => visits(clicked, s) === 2), 'every stem station a stop both ways', STEM.map((s) => visits(clicked, s)).join(','));
  check(LOOPS.every((s) => visits(clicked, s) === 1), 'every loop station a stop once', LOOPS.map((s) => visits(clicked, s)).join(','));
  check(same(clicked.ways, LOOPS), 'served one way at the loops\' stations only', JSON.stringify(clicked.ways));

  // ---- the same stations by edits, south to north, then each offered turn
  await stopEditing();
  const back = await page.evaluate(async (names) => {
    const tw = window.tw, w = tw.world.value, id = (n) => Number(w.save.stations.find((x) => x.name === n).id);
    const { queryClock } = await import('/src/workers/clockClient.ts');
    const line = (await tw.client.edit({ op: 'addLine', stops: [id(names[0])], name: 'Edited', colour: '#2ca02c' })).line;
    for (const n of names.slice(1)) await tw.client.edit({ op: 'insertStop', line, station: id(n) });
    for (let k = 0; k < 2; k++) {
      const d = await queryClock({ what: 'stopTiers', line });
      let at = -1;
      for (let i = 0; i + 2 < d.length; i += 3) if (d[i + 1] === 7) { at = d[i]; break; }
      if (at < 0) break;
      await tw.client.edit({ op: 'turnEnd', line, station: at });
    }
    return line;
  }, [...[...SOUTH].reverse(), ...[...STEM].reverse(), ...NORTH]);
  await page.waitForTimeout(800);
  const edited = await info('Edited');
  out.edited = edited;
  check(edited.loop === 'circle' && edited.oneWay && STEM.every((s) => visits(edited, s) === 2) && LOOPS.every((s) => visits(edited, s) === 1), 'the same south to north by edits and turns', JSON.stringify(edited.stops));

  // ---- dots, list and hover agree, for Line 1, Line 1a and the clicked line; chevrons along the track
  for (const name of ['Line 1', 'Line 1a', 'Clicked']) {
    const l = await info(name);
    await editStops(l.num);
    await centre(name === 'Line 1a' ? 'Nes' : 'Weteringschans', 12.1);
    await page.waitForTimeout(2500);
    const mk = await halfMarkers(), list = await listRows(), hover = await hoverChevrons(name);
    out[name] = { ways: l.ways, markers: mk, list: list.half, hover };
    check(same(mk.half, l.ways) && same(mk.chev, l.ways), `${name}: half-filled markers and their chevrons are the sim's one-way stops`, `${mk.half.length} half, ${mk.chev.length} chevrons, ${l.ways.length} one way`);
    check(same(list.half, l.ways), `${name}: the stop list's half dots are the same`, `${list.half.length} of ${list.all} rows`);
    check(same(hover, l.ways), `${name}: the hovers' chevrons are the same`, JSON.stringify(hover));
    // T-3139, on each loop up close so the drawn track's first piece is short
    const angles = [];
    for (const at of name === 'Line 1a' ? [] : ['Pluvierstraat', 'Parnas']) {
      await centre(at, 15);
      await page.waitForTimeout(1500);
      angles.push(...(await chevronAngles(name)));
    }
    if (angles.length) {
      out[name].angles = angles;
      const bad = angles.filter((a) => a.off === null || a.off > TANGENT_DEG);
      check(bad.length === 0, `${name}: every chevron lies along the track, pointing the way the trains leave (within ${TANGENT_DEG} degrees)`, `${angles.length} chevrons; off ${JSON.stringify(bad.slice(0, 3))}`);
    }
    if (name === 'Clicked') {
      // the south loop's stops in running order: each next one-way stop's chevron points along the run, so two in a
      // row never point toward each other
      await centre('Parnas', 14.4);
      await page.waitForTimeout(1500);
      const pairs = await page.evaluate(() => {
        const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.name === 'Clicked');
        const way = w.lineStats[l.id].oneWayAt, nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
        const out = [];
        for (let i = 0; i + 1 < l.stops.length; i++) {
          const a = l.stops[i], b = l.stops[i + 1];
          if (way[a] === undefined || way[b] === undefined) continue;
          const ma = tw.tools.stops.marks.get(a), mb = tw.tools.stops.marks.get(b);
          if (!ma?.shown || !mb?.shown) continue;
          const deg = (m) => (Number(/rotate\((-?[\d.]+)deg\)/.exec(m.chev.style.transform)?.[1]) * Math.PI) / 180;
          // from a toward b, a's chevron leads on and b's does not point back at a
          const dx = mb.x - ma.x, dy = mb.y - ma.y, ta = deg(ma), tb = deg(mb);
          out.push({ a: nm.get(a), b: nm.get(b), aOn: Math.cos(ta) * dx + Math.sin(ta) * dy > 0, bOn: Math.cos(tb) * dx + Math.sin(tb) * dy > 0 });
        }
        return out;
      });
      out.pairs = pairs;
      check(pairs.length >= 3 && pairs.every((p) => p.aOn && p.bOn), 'one-way stops in a row on the loop point the same way, along the run', `${pairs.length} pairs; ${JSON.stringify(pairs.filter((p) => !p.aOn || !p.bOn).slice(0, 3))}`);
      await page.screenshot({ path: join(ARTIFACTS, `t3122-south-loop-${SHADE}.png`) });
      await centre('Pluvierstraat', 15.2);
      await page.waitForTimeout(1500);
      await page.screenshot({ path: join(ARTIFACTS, `t3122-north-loop-${SHADE}.png`) });
      await centre('Weteringschans', 12.6);
      await page.waitForTimeout(1500);
      await page.screenshot({ path: join(ARTIFACTS, `t3122-clicked-${SHADE}.png`) });
    }
    await stopEditing();
  }
  // the stop list of the clicked line, not edited
  await page.evaluate(() => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.name === 'Clicked'); window.tw.selection.value = { kind: 'line', line: l.id }; });
  await page.waitForTimeout(1500);
  const shown = await listRows();
  check(same(shown.half, LOOPS), 'the clicked line\'s stop list, not edited, half dots on the loops\' stations', JSON.stringify(shown.half));
  await page.screenshot({ path: join(ARTIFACTS, `t3122-list-${SHADE}.png`) });
} catch (e) {
  console.log('FAIL', e.message);
  fail++;
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
out.errors = errors;
await writeFile(join(ARTIFACTS, 't3122-stoporder.json'), JSON.stringify(out, null, 1));
await chrome.close();
console.log(fail ? `T-3122 FAILED (${fail})` : 'T-3122 PASSED');
process.exit(fail ? 1 : 0);
