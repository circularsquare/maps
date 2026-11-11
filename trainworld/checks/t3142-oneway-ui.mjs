// T-3142 and T-3143: a stop a loop serves one way is a half-filled dot on bus and ferry loops too, and the station
// hover says so. On a new game in New York: a bus ring and a bus route that loops back (stops edited, the markers
// half-filled exactly where the sim says one way, `LineStats.oneWayAt`), a ferry loop (the tool's port circles, the
// stop list's half dots, the port hovers' "One way, towards ..." rows, none once it runs both ways). Then Anita's
// Amsterdam save: Line 1's station hovers say it at its loops' stations only. Screenshots artifacts/trainworld/t3142-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3142-oneway-ui.mjs [dev url]   (SHADE=dark, RAIL=0 skips Amsterdam)
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6194/';
const SHADE = process.env.SHADE ?? 'light';
const RAIL = process.env.RAIL !== '0';
const SAVE = process.env.SAVE ?? join(ARTIFACTS, 't3099-anita-amsterdam.save');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const C = [-73.9535, 40.6505];
const PORT = { whitehall: [-74.0130, 40.7012], stgeorge: [-74.0715, 40.6437], redhook: [-74.0115, 40.6855] };

const chrome = await launchChrome({ prefix: 'tw-t3142-', size: [1400, 900], deviceScaleFactor: 2 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);
const out = { when: new Date().toISOString() };
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = async (ms = 1500) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const stats = (num) => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return { ways: Object.keys(w.lineStats[l.id].oneWayAt), stops: l.stops, loop: w.lineStats[l.id].loop }; }, num);
const editStops = (num) => page.evaluate(async (num) => {
  const tw = window.tw, ui = await import('/src/game/ui.ts');
  const l = tw.world.value.save.lines.find((x) => x.num === num);
  tw.selection.value = { kind: 'line', line: l.id };
  ui.tool.value = 'line';
  ui.lineDraft.value = { line: num, stops: l.stops.map(Number) };
}, num);
const stopEditing = () => page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = null; });
const jump = (center, zoom) => page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); window.tw.overlay.invalidate(); }, [center, zoom]);
const centreOf = (num) => page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), r = w.lineStats[l.id].bus.routeLL; let a = [Infinity, Infinity, -Infinity, -Infinity]; for (let i = 0; i < r.length; i += 2) a = [Math.min(a[0], r[i]), Math.min(a[1], r[i + 1]), Math.max(a[2], r[i]), Math.max(a[3], r[i + 1])]; return [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2]; }, num);

/** the bus editing markers in stop order: half-filled or not, with the stop's id and whether the sim says one way */
const busMarkers = (num) => page.evaluate((num) => {
  const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
  return tw.tools.busStops.items.map((m, i) => ({ id: st.bus.stops[i].id, half: /gradient/.test(m.el.style.background), way: st.oneWayAt[st.bus.stops[i].id] !== undefined, shown: m.shown }));
}, num);

/** the text rows of a station hover for a line: its one-way rows ("One way ..."), by station name */
const hoverRows = (num) => page.evaluate(async (num) => {
  const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.num === num), net = tw.renderer.net;
  const { stationTip } = await import('/src/map/networkTips.ts');
  const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
  const res = {};
  for (const id of new Set(l.stops)) {
    const tip = stationTip(tw.overlay, w, net.stationIds.indexOf(id)) ?? [];
    const rows = tip.filter((t) => t instanceof HTMLElement && t.classList.contains('tag-facts') && /^One way/.test(t.textContent.replace(l.letter, '').trim())).map((t) => t.textContent.replace(l.letter, '').trim());
    res[nm.get(id)] = rows;
  }
  return res;
}, num);

try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus && window.tw.tools?.ferry, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 600000 });

  // ---- bus ring and a route that loops back
  const made = await page.evaluate(async (C) => {
    const tw = window.tw, c = tw.lngLatToPack(C[0], C[1]);
    const sq = (x, y, s) => [x, y, 1, x + s, y, 1, x + s, y + s, 1, x, y + s, 1, x, y, 1];
    const ring = await tw.client.edit({ op: 'addBusLine', route: sq(c.x - 1500, c.y - 1500, 3000), name: 'Ring', colour: '#c0392b' });
    const [x, y] = [c.x + 3000, c.y - 1500];
    const back = await tw.client.edit({ op: 'addBusLine', route: [x, y, 1, x + 2000, y, 1, x + 3000, y, 1, x + 3000, y + 1000, 1, x + 2000, y + 1000, 1, x + 2000, y, 1, x, y, 1], name: 'Back', colour: '#2d7dd2' });
    return { ring: ring.line, back: back.line, c: [c.x, c.y], b: [x, y] };
  }, C);
  check(made.ring !== undefined && made.back !== undefined, 'a bus ring and a route that loops back are added');
  await edit({ op: 'busHand', line: made.ring });
  await edit({ op: 'busHandStop', line: made.ring, x: made.c[0] + 1500, y: made.c[1] - 500, on: true });
  await edit({ op: 'busHand', line: made.back });
  await edit({ op: 'busHandStop', line: made.back, x: made.b[0] + 3000, y: made.b[1] + 400, on: true });
  await edit({ op: 'busHandStop', line: made.back, x: made.b[0] + 1000, y: made.b[1] + 10, on: true });
  await page.waitForTimeout(800);
  const ringC = await centreOf(made.ring), backC = await centreOf(made.back);

  for (const [name, num, at, zoom] of [['ring', made.ring, ringC, 13.6], ['back', made.back, backC, 13.4]]) {
    const s = await stats(num);
    await editStops(num);
    await jump(at, zoom);
    await settle(2500);
    const mk = await busMarkers(num);
    out[name] = { ways: s.ways, markers: mk };
    check(s.ways.length > 0, `the sim serves the ${name} one way at some stops`, `${s.ways.length} ids`);
    check(mk.length > 0 && mk.every((m) => m.shown && m.half === m.way), `${name}: the markers are half-filled exactly at the stops the sim says one way`, mk.map((m) => (m.half ? 'H' : 'F')).join(''));
    if (name === 'back') check(mk.some((m) => !m.way) && mk.some((m) => m.way), 'the loop-back route has stops both ways (its way out) and one way (its loop)', mk.map((m) => (m.way ? 'H' : 'F')).join(''));
    await page.screenshot({ path: join(ARTIFACTS, `t3142-bus-${name}-${SHADE}.png`) });
    await stopEditing();
  }

  // ---- ferry loop
  const ferry = await page.evaluate(async (PORT) => {
    const tw = window.tw, water = tw.tools.ferry.water, ports = [], berths = [];
    const snap = async ([lng, lat]) => {
      const k = 1 / (111320 * Math.cos((lat * Math.PI) / 180)), j = 1 / 110540;
      for (const r of [0, 50, 100, 150]) for (let a = 0; a < (r ? 8 : 1); a++) {
        const s = await water.port([lng + r * Math.cos((a * Math.PI) / 4) * k, lat + r * Math.sin((a * Math.PI) / 4) * j]);
        if (s?.ok) return s;
      }
      return null;
    };
    for (const p of ['whitehall', 'stgeorge', 'redhook']) {
      const s = await snap(PORT[p]);
      if (!s) return { error: `no port at ${p}` };
      const a = tw.lngLatToPack(s.landing[0], s.landing[1]), b = tw.lngLatToPack(s.berth[0], s.berth[1]);
      ports.push({ x: a.x, y: a.y, bx: b.x, by: b.y, name: p });
      berths.push(s.berth);
    }
    const r = await water.line([...berths, berths[0]].map((at) => ({ at, kind: 1 })), 'check');
    if (!r?.route.ok) return { error: `no route (${r?.route.legs.map((l) => l.reason ?? 'ok').join(', ')})` };
    const route = [];
    for (let i = 0; i < r.route.kinds.length; i++) { const q = tw.lngLatToPack(r.route.ll[2 * i], r.route.ll[2 * i + 1]); route.push(q.x, q.y, r.route.kinds[i], r.route.speeds[i]); }
    const e = await tw.client.edit({ op: 'addFerryLine', ports: [...ports, { again: 0 }], route, name: 'Harbour', colour: '#1f77b4' });
    if (!e?.ok) return { error: `not added (${(e?.issues ?? []).join(', ')})` };
    return { line: e.line, centre: [(PORT.whitehall[0] + PORT.stgeorge[0]) / 2, (PORT.whitehall[1] + PORT.stgeorge[1]) / 2] };
  }, PORT);
  if (ferry.error) throw new Error(ferry.error);
  await page.waitForTimeout(1200);
  const fs = await stats(ferry.line);
  check(fs.loop === 'circle' && fs.ways.length === 3, 'the ferry loop is a circle the sim serves one way at its three ports', `${fs.loop} ${fs.ways.length}`);
  // the stop list, selected
  await jump(ferry.centre, 11.8);
  await page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); window.tw.selection.value = { kind: 'line', line: l.id }; }, ferry.line);
  await settle(2000);
  const rows = await page.evaluate(() => [...document.querySelectorAll('ul.stops.ferry-ports > li')].map((r) => r.classList.contains('oneway')));
  check(rows.length === 4 && rows.every(Boolean), 'the ferry loop\'s stop list has a half dot on each of its rows', JSON.stringify(rows));
  // the ferry tool with the line loaded
  await page.evaluate((num) => { window.tw.tools.ferry.begin(num); }, ferry.line);
  await settle(1500);
  const circles = await page.evaluate(() => [...document.querySelectorAll('svg.ferry-draw circle.ferry-port, svg.ferry-draw circle.ferry-end')].map((c) => /url\(#ferry-half/.test(c.getAttribute('style'))));
  check(circles.length >= 3 && circles.every(Boolean), 'the ferry tool draws each port of the loop half-filled', JSON.stringify(circles));
  await page.screenshot({ path: join(ARTIFACTS, `t3142-ferry-loop-${SHADE}.png`) });
  await page.evaluate(() => window.tw.tools.ferry.clear());
  // hovers: one way, then both ways
  let hv = await hoverRows(ferry.line);
  const text = Object.values(hv).flat();
  check(Object.values(hv).length === 3 && Object.values(hv).every((r) => r.length === 1) && text.every((t) => /^One way, towards [A-Za-z ]+$/.test(t)), 'each port\'s hover has one "One way, towards ..." row', JSON.stringify(hv));
  out.ferryHover = hv;
  await edit({ op: 'lineOneWay', line: ferry.line, oneWay: false });
  await page.waitForTimeout(1200);
  hv = await hoverRows(ferry.line);
  check(Object.values(hv).every((r) => r.length === 0), 'run both ways, no port hover says one way', JSON.stringify(hv));
  await edit({ op: 'lineOneWay', line: ferry.line, oneWay: true });
  await page.waitForTimeout(1000);
  await page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); window.tw.selection.value = { kind: 'line', line: l.id }; }, ferry.line);
  await page.evaluate(async () => {
    const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.name === 'Harbour'), net = tw.renderer.net;
    const { stationTip } = await import('/src/map/networkTips.ts');
    const i = net.stationIds.indexOf(l.stops[1]);
    window.__tip = stationTip(tw.overlay, w, i);
    const box = document.createElement('div');
    box.id = 'check-tip';
    box.style.cssText = 'position:fixed;left:40px;top:40px;z-index:99999';
    box.className = 'tool-tag';
    for (const t of window.__tip) { const d = document.createElement('div'); if (typeof t === 'string') d.textContent = t; else d.append(t); box.append(d); }
    document.body.append(box);
  });
  await page.screenshot({ path: join(ARTIFACTS, `t3142-hover-${SHADE}.png`), clip: { x: 0, y: 0, width: 420, height: 260 } });
  await page.evaluate(() => document.getElementById('check-tip')?.remove());
} catch (e) {
  console.log('FAIL', e.message);
  fail++;
}

// ---- Amsterdam: Line 1's hovers say "One way, towards ..." at its loops' stations only
if (RAIL && fail === 0) {
  try {
    const p2 = await chrome.ctx.newPage();
    await p2.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
    collectErrors(p2, errors);
    await p2.goto(`${BASE}?debug=1&paused=1&demandWorkers=1&v=${Date.now()}`);
    await p2.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
    const bytes = await readFile(SAVE);
    await p2.evaluate(async (data) => { const m = await import('/src/workers/clockClient.ts'); await m.importGame(new File([Uint8Array.from(data)], 'game.save')); }, Array.from(bytes));
    await p2.waitForURL(/load=file/, { timeout: 120000 });
    await p2.waitForFunction(() => window.tw?.world?.value?.save.lines.length > 0, undefined, { timeout: 300000, polling: 100 });
    await demandFinal(p2, { timeout: 600000 });
    await p2.waitForTimeout(1000);
    const res = await p2.evaluate(async () => {
      const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.name === 'Line 1'), st = w.lineStats[l.id], net = tw.renderer.net;
      const { stationTip } = await import('/src/map/networkTips.ts');
      const nm = new Map(w.save.stations.map((s) => [s.id, s.name]));
      const one = Object.keys(st.oneWayAt).map((s) => nm.get(s)).sort(), says = [], texts = [];
      for (const id of new Set(l.stops)) {
        const tip = stationTip(tw.overlay, w, net.stationIds.indexOf(id)) ?? [];
        const rows = tip.filter((t) => t instanceof HTMLElement && t.classList.contains('tag-facts') && /One way/.test(t.textContent));
        if (rows.length) { says.push(nm.get(id)); texts.push(rows[0].textContent.trim()); }
      }
      return { one, says: says.sort(), texts };
    });
    out.line1 = res;
    check(same(res.one, res.says) && res.one.length > 0, 'Amsterdam Line 1: the hovers say "One way" at the stations the sim serves one way, and no others', `${res.says.length} of ${res.one.length}; ${res.texts.slice(0, 2).join(' | ')}`);
  } catch (e) {
    console.log('FAIL', e.message);
    fail++;
  }
}
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
out.errors = errors;
await writeFile(join(ARTIFACTS, 't3142-oneway-ui.json'), JSON.stringify(out, null, 1));
await chrome.close();
console.log(fail ? `T-3142 FAILED (${fail})` : 'T-3142 PASSED');
process.exit(fail ? 1 : 0);
