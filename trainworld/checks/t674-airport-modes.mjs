// T-674: after a line to an airport, the airport's hover (the plane on Destinations) shows the
// network's mode split, not the one from before the line. A new Tokyo game (?new=1), Destinations on
// under All purposes; a line from Hamamatsucho through Shinagawa to Haneda's busiest terminal cell;
// after the final solve the plane's hover shares must agree within 1 point with the worker's own
// airport legs summed round that cell (`cellModes("airport")`), and carry some train. Then the
// destination bubble beside the plane reads the same with and without a selection. About three minutes.
//   node trainworld/checks/t674-airport-modes.mjs [url]   (from the maps root; default localhost:6144)
// Prints PASS or FAIL lines and exits 1 on failure. Writes artifacts/trainworld/t674-airport.json.
import { writeFile } from 'node:fs/promises';
import { launchChrome, demandFinal, ANGLE } from './lib.mjs';

const BASE = (process.argv[2] ?? 'http://localhost:6144/').replace(/\/?$/, '/');
const OUT = 'artifacts/trainworld/';
const HANEDA = [139.7810, 35.5494];
const STOPS = [['Hamamatsucho', 139.7570, 35.6553], ['Shinagawa', 139.7387, 35.6285]];
const TOL = 0.01;
const JOIN_M = 2500; // map/airportMarks.ts

const { ctx, close } = await launchChrome({ prefix: 'tw-t674-', args: ANGLE });
const results = [], log = {};
const report = (name, ok, detail = '') => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const waitFinal = (page, after = -1) => demandFinal(page, { strict: false, after, timeout: 900000 });
const shares = (r, w, d) => { const n = r + w + d; return n > 0 ? [r / n, w / n, d / n] : [0, 0, 0]; };
const off = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** The plane's screen place (map frame, CSS px), or null when none is shown near Haneda. */
const plane = (page) => page.evaluate((h) => {
  const t = window.tw, at = t.map.project(h);
  let best = null;
  for (const m of document.querySelectorAll('.airport-mark')) {
    if (m.style.display === 'none') continue;
    const g = m.style.transform.match(/translate\(([-0-9.]+)px, ([-0-9.]+)px\)/);
    if (!g) continue;
    const x = Number(g[1]), y = Number(g[2]), d = Math.hypot(x - at.x, y - at.y);
    if (!best || d < best.d) best = { x, y, d };
  }
  if (!best || best.d > 200) return null;
  const ll = t.map.unproject([best.x, best.y]);
  return { x: best.x, y: best.y, lng: ll.lng, lat: ll.lat };
}, HANEDA);

/** A hover's lines and its mode bar's shares (train, walk, drive). */
async function hoverAt(page, x, y) {
  const lines = await page.evaluate(([x, y]) => window.twViews.hover(x, y), [x, y]);
  if (!lines) return null;
  const cols = await page.evaluate(async () => { const p = await import('/src/game/palette.ts'); return [p.MODE_COLOURS.train, p.MODE_COLOURS.walk, p.MODE_COLOURS.drive]; });
  const bar = lines.find((l) => /flex-grow/.test(l)) ?? '';
  const grow = cols.map((c) => { const m = bar.match(new RegExp(`flex-grow:([0-9.e+-]+);background:${c.replace(/[#()]/g, (s) => '\\' + s)}`)); return m ? Number(m[1]) : 0; });
  return { head: lines[0].replace(/<[^>]+>/g, ''), shares: shares(...grow) };
}

/** The worker's airport legs, summed over the cells within JOIN_M of the plane's cell. */
const truth = (page, p) => page.evaluate(async ([lng, lat, join]) => {
  const cities = await import('/src/game/cities.ts');
  const demand = await import('/src/game/demand.ts');
  const [cells, modes] = await Promise.all([demand.cityCells(), demand.cellModes('airport')]);
  if (!cells || !modes) return null;
  const frame = cities.getWorldFrame(), kx = 111320 * Math.cos((lat * Math.PI) / 180);
  let r = 0, w = 0, dr = 0;
  for (let c = 0; c < cells.n; c++) {
    const all = modes.workRail[c] + modes.workWalk[c] + modes.workDrive[c];
    if (all <= 0) continue;
    const [x, y] = frame.toLonLat(cells.xy[c * 2], cells.xy[c * 2 + 1]);
    if (Math.hypot((x - lng) * kx, (y - lat) * 110540) > join + 50) continue;
    r += modes.workRail[c]; w += modes.workWalk[c]; dr += modes.workDrive[c];
  }
  return [r, w, dr];
}, [p.lng, p.lat, JOIN_M]);

try {
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 240000 });
  await page.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 12 }), HANEDA);
  await page.waitForFunction(() => window.twDemand.activeCity.value === 'tyo', undefined, { timeout: 300000 });
  await waitFinal(page);
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    const t = window.tw;
    t.display.value = { ...t.display.value, commuters: true, commuterEnd: 'work', dayShown: 'weekday' };
    ui.purposeFilter.value = 'all';
  });
  await page.waitForFunction(() => [...document.querySelectorAll('.airport-mark')].some((m) => m.style.display !== 'none'), undefined, { timeout: 120000, polling: 500 });
  const p0 = await plane(page);
  report('a plane at Haneda before the line', !!p0, JSON.stringify(p0));
  if (!p0) throw new Error('no plane');
  log.before = { plane: await hoverAt(page, p0.x, p0.y), truth: await truth(page, p0), bubble: await hoverAt(page, p0.x + 25, p0.y) };

  const before = await page.evaluate(() => window.twDemand.view()?.version ?? -1);
  await page.evaluate(async (stops) => {
    const t = window.tw, run = async (op) => { const r = await t.client.edit(op); if (!r.ok) throw Error(JSON.stringify({ op, r })); return r; };
    const pts = stops.map((s) => t.lngLatToPack(s[1], s[2]));
    const tail = (a, b) => { const dx = a.x - b.x, dy = a.y - b.y, l = Math.hypot(dx, dy); return { x: a.x + 300 * dx / l, y: a.y + 300 * dy / l }; };
    const path = [tail(pts[0], pts[1]), ...pts, tail(pts.at(-1), pts.at(-2))];
    const nodes = [];
    for (let i = 1; i < path.length; i++) {
      const from = i === 1 ? { kind: 'free', x: path[0].x, y: path[0].y, level: -2 } : { kind: 'node', node: nodes.at(-1) };
      await run({ op: 'route', from, to: { kind: 'free', x: path[i].x, y: path[i].y, level: -2 }, pis: [], single: false });
      const e = t.world.value.edges.at(-1);
      if (i === 1) nodes.push(e.a);
      nodes.push(e.b);
    }
    const at = nodes.slice(1, -1);
    for (let i = 0; i < stops.length; i++) await run({ op: 'addStation', at: { kind: 'node', node: at[i] }, platform: 160, name: stops[i][0] });
    const line = (await run({ op: 'addLine', stops: at, name: 'Airport test', colour: '#e4002b' })).line;
    await run({ op: 'setTrainCount', line, trains: [10, 10, 5] });
    await run({ op: 'constructAll' });
  }, [...STOPS, ['Haneda', p0.lng, p0.lat]]);
  await waitFinal(page, before);
  await page.evaluate((ll) => void window.tw.map.jumpTo({ center: ll, zoom: 12 }), HANEDA);
  await page.waitForTimeout(6000);
  const p1 = await plane(page);
  report('a plane at Haneda after the line', !!p1, JSON.stringify(p1));
  if (!p1) throw new Error('no plane after');
  const want = await truth(page, p1);
  const got = await hoverAt(page, p1.x, p1.y);
  log.after = { plane: got, truth: want, bubble: await hoverAt(page, p1.x + 25, p1.y) };
  // the bubbles show a new result at once (T-906)
  await page.evaluate(() => window.tw.clock.setRunning(false));
  await page.waitForTimeout(1500);
  log.after.dayLater = await hoverAt(page, p1.x + 25, p1.y);
  // the destination bubble there, selected: its own figures
  const sel = await page.evaluate(([x, y]) => window.twViews.click(x + 25, y, false), [p1.x, p1.y]);
  if (sel) {
    await page.evaluate((s) => { window.tw.selection.value = s; }, sel);
    await page.waitForTimeout(3000);
    log.after.selected = await hoverAt(page, p1.x + 25, p1.y);
    await page.evaluate(() => { window.tw.selection.value = null; });
    await page.waitForTimeout(3000);
    log.after.unselected = await hoverAt(page, p1.x + 25, p1.y);
  }
  const ws = want && shares(...want);
  report('the worker carries airport trips by train after the line', !!ws && ws[0] > 0.02, JSON.stringify(ws));
  report('the plane\'s hover agrees with the worker within 1 point', !!ws && !!got && off(got.shares, ws) <= TOL, `(hover ${JSON.stringify(got?.shares)}, worker ${JSON.stringify(ws)})`);
  const { selected: s, unselected: u } = log.after;
  report('the destination bubble beside the plane reads the same selected and not', !!s && !!u && s.head === u.head && off(s.shares, u.shares) <= TOL, `(selected ${JSON.stringify(s)}, not ${JSON.stringify(u)})`);
  // the plane and its hover tag, for the eye
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  await page.mouse.move(box.left + p1.x - 3, box.top + p1.y - 3);
  await page.mouse.move(box.left + p1.x, box.top + p1.y, { steps: 3 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${OUT}t674-plane.png`, clip: { x: box.left + p1.x - 300, y: box.top + p1.y - 220, width: 640, height: 440 } });
  report('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
} finally {
  await writeFile(OUT + 't674-airport.json', JSON.stringify(log, null, 1));
  await close();
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
