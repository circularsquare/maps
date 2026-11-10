// T-1869: dragging a blueprint station's platform end, and the Select tool's pick, with real mouse
// events in headless Chrome (a point added to track: t2951-add-point.mjs).
//   node trainworld/checks/t1869-drag.mjs [url]      (from the maps root; about a minute)
// A new game in New York gets a curved blueprint piece and a straight one with a station on it.
// 1. From zoom 16 the station's platform ends have squares; dragging one moves the station by the
//    pointer's move, the end held under the pointer.
// 2. The Select tool's pick of everything shows a square on each movable node; dragging the pick
//    moves the squares with it and shows the curves' speed labels.
// Screenshots (before and during each) to artifacts/trainworld/t1869/. One PASS or FAIL line each.
import { mkdir } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6169/';
const OUT = `${ARTIFACTS}t1869/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1869-', size: [1200, 800], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  // a curve turning 60 degrees between two 400 m legs, and a straight piece with a station on it
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client, a = tw.lngLatToPack(-73.968, 40.778);
    const pi = { x: a.x + 400, y: a.y }, b = { x: pi.x + 400 * Math.cos(Math.PI / 3), y: pi.y + 400 * Math.sin(Math.PI / 3) };
    const ids = () => new Set(tw.world.value.edges.map((e) => e.id));
    let before = ids();
    await c.edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [{ x: pi.x, y: pi.y, level: 0 }], single: false });
    const curve = tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null;
    const s0 = { x: a.x, y: a.y - 700 }, s1 = { x: a.x + 600, y: a.y - 700 };
    before = ids();
    await c.edit({ op: 'route', from: { kind: 'free', x: s0.x, y: s0.y, level: 0 }, to: { kind: 'free', x: s1.x, y: s1.y, level: 0 }, pis: [], single: false });
    const straight = tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null;
    const st = straight === null ? null : await c.edit({ op: 'addStation', at: { kind: 'edge', edge: straight, x: a.x + 300, y: a.y - 700 }, platform: 200, name: '' });
    const station = tw.world.value.save.stations.at(-1)?.num ?? null;
    return { curve, straight, station: st?.ok ? station : null, pi };
  });
  if (made.curve === null || made.station === null) throw new Error('setup failed ' + JSON.stringify(made));
  const setTool = (t) => page.evaluate(async (t) => { (await import('/src/game/ui.ts')).tool.value = t; }, t);
  const jump = (lngLat, zoom) => page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [lngLat, zoom]);
  const lngLatOf = (p) => page.evaluate((p) => window.tw.packToLngLat ? window.tw.packToLngLat(p.x, p.y) : null, p);
  const rect = () => page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { left: r.left, top: r.top }; });
  const packAt = (x, y) => page.evaluate(([x, y]) => { const m = window.tw.map, r = m.getContainer().getBoundingClientRect(), ll = m.unproject([x - r.left, y - r.top]); return window.tw.lngLatToPack(ll.lng, ll.lat); }, [x, y]);
  const drag = async (x, y, dx, dy, shot) => {
    await page.mouse.move(x, y);
    await page.waitForTimeout(200);
    await page.mouse.down();
    // whole px: a mouse event's position is
    for (let i = 1; i <= 10; i++) { await page.mouse.move(Math.round(x + (dx * i) / 10), Math.round(y + (dy * i) / 10)); await page.waitForTimeout(40); }
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}${shot}` });
  };
  const visible = (sel) => page.evaluate((sel) => [...document.querySelectorAll(sel)].filter((m) => m.style.display !== 'none' && m.style.visibility !== 'hidden').map((m) => { const r = m.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }), sel);
  const undo = async () => { await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true)); await page.waitForTimeout(700); };

  await setTool('select');
  const ll = await lngLatOf(made.pi);

  // 1: a platform's end
  const stLL = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return [p.lng, p.lat]; }, made.station);
  await jump(stLL, 16);
  await page.waitForTimeout(1500);
  await page.mouse.move(5, 5);
  const ends = await page.evaluate(async () => (await import('/src/map/dragHandles.ts')).stationEnds(window.tw.overlay));
  await page.screenshot({ path: `${OUT}station-before.png` });
  report(ends.length === 2, 'from zoom 16 the blueprint station has a handle at each platform end', `${ends.length} ends`);
  if (ends.length === 2) {
    const r = await rect(), e = ends[1], ex = Math.round(e.x + r.left), ey = Math.round(e.y + r.top);
    const n0 = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return { x: p.x, y: p.y }; }, made.station);
    const p0 = await packAt(ex, ey), p1 = await packAt(ex + 30, ey + 35);
    await drag(ex, ey, 30, 35, 'station-during.png');
    await page.mouse.up();
    await page.waitForTimeout(900);
    const n1 = await page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return { x: p.x, y: p.y }; }, made.station);
    const moved = Math.hypot(n1.x - n0.x - (p1.x - p0.x), n1.y - n0.y - (p1.y - p0.y));
    const ends1 = await page.evaluate(async () => (await import('/src/map/dragHandles.ts')).stationEnds(window.tw.overlay));
    const held = Math.min(...ends1.map((q) => Math.hypot(q.x + r.left - (ex + 30), q.y + r.top - (ey + 35))));
    await page.screenshot({ path: `${OUT}station-after.png` });
    // the platform follows its track, which bends where the node moved off its line: the end lands near the pointer
    report(moved < 1 && held < 12, 'dragging a platform end moves the station by the pointer\'s move, the end near the pointer', `node off the pointer's move by ${moved.toFixed(2)} m, end ${held.toFixed(1)} px from the pointer`);
    await undo();
  }

  // 2: the Select tool's pick
  await jump(ll ?? [-73.963, 40.778], 14.2);
  await setTool('blueprint');
  const pick = await page.evaluate(async (station) => {
    const bp = await import('/src/game/blueprintPick.ts'), w = window.tw.world.value;
    const edges = w.edges.filter((e) => !e.built).map((e) => e.id);
    bp.bpPicked.value = { edges, stations: [station] };
    const nodes = new Set([station]);
    for (const e of w.edges) if (edges.includes(e.id)) nodes.add(e.a), nodes.add(e.b);
    return { edges: edges.length, nodes: [...nodes] };
  }, made.station);
  await page.waitForTimeout(1200);
  await page.mouse.move(5, 5);
  const sq0 = await visible('.tool-marks > .pi-mark');
  await page.screenshot({ path: `${OUT}pick-before.png` });
  report(sq0.length === pick.nodes.length, 'the pick shows a square on every movable node', `${sq0.length} squares, ${pick.nodes.length} nodes`);
  if (sq0.length) {
    const s = sq0[0];
    await drag(s.x, s.y, 40, 30, 'pick-during.png');
    const sq1 = await visible('.tool-marks > .pi-mark'), labels = await visible('.tool-marks > .pi-label');
    const follow = sq1.length === sq0.length && sq0.every((a) => sq1.some((b) => Math.hypot(b.x - a.x - 40, b.y - a.y - 30) < 2.5));
    report(follow && labels.length >= 1, 'dragging the pick moves its squares and shows the curves\' speeds', `${sq1.length} squares, follow ${follow}, ${labels.length} speed labels`);
    await page.mouse.up();
    await page.waitForTimeout(900);
    await undo();
  }
} catch (e) {
  failed = true;
  console.log(`FAIL  drag check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
