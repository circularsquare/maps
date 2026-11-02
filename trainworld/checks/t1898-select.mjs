// T-1898: the Select tool's box moves only the nodes inside it, the selection bar's Regular / High
// speed, and a 320 km/h blueprint curve keeping its speed through a drag; real mouse events on the
// real save in headless Chrome.
//   node trainworld/checks/t1898-select.mjs [url]      (from the maps root; about two minutes)
// 1. A run of three straight blueprint pieces and a single-track piece beside its middle; a box
//    round the middle piece and the single track picks all four, but a drag of the pick moves only
//    the four nodes inside the box: the run's outer ends stay, its outer pieces stretch.
// 2. High speed in the bar makes the three double pieces high-speed track and counts the single one
//    as not changed; Regular puts them back.
// 3. At 320 km/h (Build tab: High speed) a 3 km straight and a tangent curve from its end at the
//    design radius (zoom 14): bending the curve's middle 15 px sharper keeps 300 km/h or more, 15 px
//    gentler keeps 320, 40 px sharper slows it only to what its legs allow (over 160), and a 15 px
//    drag of the joint keeps both pieces at 300 or more.
// Screenshots to artifacts/trainworld/t1898/. One PASS or FAIL line each.
import { mkdir, readFile } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS, REAL_SAVE, importSave } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6190/';
const OUT = `${ARTIFACTS}t1898/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1898-', size: [1300, 850], args: GL });
let failed = false;
const report = (ok, what, detail = '') => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? '  (' + detail + ')' : ''}`);
};
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.client, undefined, { timeout: 180000 });
  await page.waitForTimeout(1500);
  await importSave(page, await readFile(REAL_SAVE));
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length > 0, undefined, { timeout: 120000 });
  await page.waitForTimeout(1500);
  const lines = await page.evaluate(() => window.tw.world.value.save.lines.length);
  report(lines === 109, 'the real save is loaded', `${lines} lines`);
  // the loading card sits over the map's middle until the city is in
  await page.waitForFunction(() => ![...document.querySelectorAll('.loading-card')].some((e) => e.offsetParent), undefined, { timeout: 300000 }).catch(() => console.log('  (loading card still up)'));

  const setTool = (t) => page.evaluate(async (t) => { (await import('/src/game/ui.ts')).tool.value = t; }, t);
  const jump = (lngLat, zoom) => page.evaluate(([c, z]) => window.tw.map.jumpTo({ center: c, zoom: z }), [lngLat, zoom]);
  // a node's or a pack point's place on the page, px
  const screenOf = (p) => page.evaluate((p) => { const m = window.tw.map, r = m.getContainer().getBoundingClientRect(), q = m.project(window.tw.packToLngLat(p.x, p.y)); return { x: q.x + r.left, y: q.y + r.top }; }, p);
  const nodeAt = (n) => page.evaluate((n) => { const p = window.tw.world.value.nodes.get(n); return p ? { x: p.x, y: p.y } : null; }, n);
  const packPerPx = () => page.evaluate(() => { const m = window.tw.map, c = m.getCenter(), a = m.project(c), b = m.unproject([a.x + 100, a.y]), p = window.tw.lngLatToPack(c.lng, c.lat), q = window.tw.lngLatToPack(b.lng, b.lat); return Math.hypot(q.x - p.x, q.y - p.y) / 100; });
  const notice = () => page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');
  const edgePis = (edge) => page.evaluate(async (edge) => Array.from(await (await import('/src/workers/clockClient.ts')).queryClock({ what: 'edgePis', edge })), edge);
  // the slowest curve of an edge, km/h (edge_pis: flag, then x, y, set radius, level, radius used, km/h, middle x, y)
  const slowest = async (edge) => {
    const d = await edgePis(edge);
    let kmh = Infinity, r = 0, set = 0;
    for (let k = 1; k + 8 <= d.length; k += 8) if (d[k + 4] > 0 && d[k + 5] < kmh) [kmh, r, set] = [d[k + 5], d[k + 4], d[k + 2]];
    return { kmh, r, set };
  };
  const drag = async (x, y, dx, dy, shot, release = true) => {
    await page.mouse.move(Math.round(x), Math.round(y));
    await page.waitForTimeout(150);
    await page.mouse.down();
    for (let i = 1; i <= 10; i++) { await page.mouse.move(Math.round(x + (dx * i) / 10), Math.round(y + (dy * i) / 10)); await page.waitForTimeout(40); }
    await page.waitForTimeout(1200);
    if (shot) await page.screenshot({ path: `${OUT}${shot}` });
    if (release) {
      const v = await page.evaluate(() => window.tw.world.value.version);
      await page.mouse.up();
      // an edit on the real save takes a moment
      await page.waitForFunction((v) => window.tw.world.value.version !== v, v, { timeout: 30000 }).catch(() => console.log('  (no edit after the drag)'));
      await page.waitForTimeout(500);
    }
  };
  const visible = (sel) => page.evaluate((sel) => [...document.querySelectorAll(sel)].filter((m) => m.style.display !== 'none' && m.style.visibility !== 'hidden').map((m) => { const r = m.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; }), sel);

  // a quiet spot on the real save where new track is accepted
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    const ids = () => new Set(tw.world.value.edges.map((e) => e.id));
    const newEdge = (before) => tw.world.value.edges.find((e) => !before.has(e.id))?.id ?? null;
    for (const [lng, lat] of [[-74.29, 40.86], [-73.72, 40.98], [-74.22, 40.54], [-74.4, 40.7]]) {
      const o = tw.lngLatToPack(lng, lat);
      const pts = [0, 1, 2, 3].map((i) => ({ x: o.x + i * 500, y: o.y }));
      const before = ids();
      let a = await c.edit({ op: 'route', from: { kind: 'free', ...pts[0], level: 0 }, to: { kind: 'free', ...pts[1], level: 0 }, pis: [], single: false });
      if (!a.ok) continue;
      const nodes = [];
      const e0 = tw.world.value.edges.find((e) => !before.has(e.id));
      nodes.push(e0.a, e0.b);
      const run = [e0.id];
      for (const k of [2, 3]) {
        const b = ids();
        a = await c.edit({ op: 'route', from: { kind: 'node', node: nodes.at(-1) }, to: { kind: 'free', ...pts[k], level: 0 }, pis: [], single: false });
        if (!a.ok) break;
        const e = tw.world.value.edges.find((x) => !b.has(x.id));
        run.push(e.id);
        nodes.push(e.a === nodes.at(-1) ? e.b : e.a);
      }
      if (run.length < 3) continue;
      const b = ids();
      a = await c.edit({ op: 'route', from: { kind: 'free', x: o.x + 650, y: o.y + 200, level: 0 }, to: { kind: 'free', x: o.x + 850, y: o.y + 200, level: 0 }, pis: [], single: true });
      if (!a.ok) continue;
      const single = newEdge(b);
      const se = tw.world.value.edges.find((e) => e.id === single);
      return { at: [lng, lat], o, run, nodes, single, singleNodes: [se.a, se.b] };
    }
    return null;
  });
  if (!made) throw new Error('no spot on the real save took the test track');

  // 1: the box
  await setTool('blueprint');
  await jump(await page.evaluate((o) => window.tw.packToLngLat(o.x + 750, o.y + 100), made.o), 15);
  await page.waitForTimeout(1500);
  const corners = await Promise.all([made.nodes[1], made.nodes[2], ...made.singleNodes].map(async (n) => screenOf(await nodeAt(n))));
  const L = Math.min(...corners.map((p) => p.x)) - 30, R = Math.max(...corners.map((p) => p.x)) + 30;
  const T = Math.min(...corners.map((p) => p.y)) - 30, B = Math.max(...corners.map((p) => p.y)) + 30;
  await page.mouse.move(Math.round(L), Math.round(T));
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) { await page.mouse.move(Math.round(L + ((R - L) * i) / 10), Math.round(T + ((B - T) * i) / 10)); await page.waitForTimeout(30); }
  await page.screenshot({ path: `${OUT}box-drawn.png` });
  await page.mouse.up();
  await page.waitForTimeout(800);
  const pick = await page.evaluate(async () => { const p = (await import('/src/game/blueprintPick.ts')).bpPicked.value; return { edges: p.edges, nodes: p.nodes ?? null }; });
  const wantEdges = [...made.run, made.single];
  const wantNodes = [made.nodes[1], made.nodes[2], ...made.singleNodes];
  report(wantEdges.every((e) => pick.edges.includes(e)) && pick.nodes?.length === 4 && wantNodes.every((n) => pick.nodes.includes(n)),
    'a box picks the track it touches and keeps the nodes inside it', `${pick.edges.length} pieces, nodes ${JSON.stringify(pick.nodes)}`);
  await page.mouse.move(5, 5);
  await page.waitForTimeout(300);
  const squares = await visible('.tool-marks > .pi-mark:not(.curve)');
  await page.screenshot({ path: `${OUT}box-picked.png` });
  report(squares.length === 4, 'the pick has squares only on the nodes inside the box', `${squares.length} squares`);
  const before = await Promise.all(made.nodes.map(nodeAt)), sBefore = await Promise.all(made.singleNodes.map(nodeAt));
  const at1 = await screenOf(before[1]), per = await packPerPx();
  await drag(at1.x, at1.y, 40, 30, 'box-during.png');
  // an edit on the real save takes a moment
  await page.waitForFunction(([n, x]) => window.tw.world.value.nodes.get(n)?.x !== x, [made.nodes[1], before[1].x], { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);
  const after = await Promise.all(made.nodes.map(nodeAt)), sAfter = await Promise.all(made.singleNodes.map(nodeAt));
  const moved = (a, b) => Math.hypot(b.x - a.x, b.y - a.y);
  const want = 50 * per;
  const inside = [moved(before[1], after[1]), moved(before[2], after[2]), ...sBefore.map((p, i) => moved(p, sAfter[i]))];
  const outside = [moved(before[0], after[0]), moved(before[3], after[3])];
  await page.screenshot({ path: `${OUT}box-after.png` });
  report(inside.every((d) => Math.abs(d - want) < 3 * per) && outside.every((d) => d < 0.01),
    'dragging the pick moves the nodes inside the box and leaves the outer ends', `inside moved ${inside.map((d) => d.toFixed(0)).join(', ')} m (pointer ${want.toFixed(0)}), outside ${outside.map((d) => d.toFixed(2)).join(', ')} m`);

  // 2: Regular / High speed
  const hsrOf = () => page.evaluate((ids) => ids.map((id) => !!window.tw.world.value.edges.find((e) => e.id === id)?.hsr), wantEdges);
  const bar = async (name) => { const b = await page.$('.blueprint-bar'); if (b) await b.screenshot({ path: `${OUT}${name}` }); };
  await bar('bar-before.png');
  await page.evaluate(async () => { window.__notice = (await import('/src/game/world.ts')).notice; });
  const said = async (was) => { await page.waitForFunction((was) => (window.__notice.value?.text ?? '') !== was, was, { timeout: 30000 }).catch(() => {}); await page.waitForTimeout(500); };
  let was = await notice();
  await page.click('.blueprint-bar button:has-text("High speed")');
  await said(was);
  const h1 = await hsrOf(), n1 = await notice();
  await bar('bar-high-speed.png');
  await page.screenshot({ path: `${OUT}high-speed.png` });
  report(h1.slice(0, 3).every(Boolean) && !h1[3] && n1 === '3 of 4 pieces set to high-speed track. 1 could not change.', 'High speed makes the picked double track high speed, single track left', `${JSON.stringify(h1)}, "${n1}"`);
  was = await notice();
  await page.click('.blueprint-bar button:has-text("Regular")');
  await said(was);
  const h2 = await hsrOf(), n2 = await notice();
  report(h2.every((v) => !v) && n2 === '3 of 4 pieces set to regular track. 1 could not change.', 'Regular puts it back', `${JSON.stringify(h2)}, "${n2}"`);

  // 3: 320 km/h
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.setTrackType('hsr'); ui.designSpeed.value = 320; });
  const hs = await page.evaluate(async (o) => {
    const tw = window.tw, c = tw.client;
    const ids = () => new Set(tw.world.value.edges.map((e) => e.id));
    const q0 = { x: o.x - 2000, y: o.y - 2500 }, q1 = { x: o.x + 1000, y: o.y - 2500 };
    let b = ids();
    let a = await c.edit({ op: 'route', from: { kind: 'free', ...q0, level: 0 }, to: { kind: 'free', ...q1, level: 0 }, pis: [], single: false, through: true, design: 320, hsr: true });
    if (!a.ok) return { err: 'straight ' + a.issues.join() };
    const straight = tw.world.value.edges.find((e) => !b.has(e.id)).id, joint = a.node;
    b = ids();
    const off = 15 * Math.PI / 180;
    a = await c.edit({ op: 'route', from: { kind: 'node', node: joint }, to: { kind: 'free', x: q1.x + 5000 * Math.cos(off), y: q1.y - 5000 * Math.sin(off), level: 0 }, pis: [], single: false, through: true, design: 320, hsr: true });
    if (!a.ok) return { err: 'curve ' + a.issues.join() };
    const curve = tw.world.value.edges.find((e) => !b.has(e.id)).id;
    return { straight, curve, joint };
  }, made.o);
  if (hs.err) throw new Error(hs.err);
  await setTool('select');
  const s0 = await slowest(hs.curve);
  report(s0.kmh >= 319 && s0.set > 7000, 'the curve is drawn for 320 km/h', `${s0.kmh.toFixed(0)} km/h, radius ${s0.r.toFixed(0)} m (set ${s0.set.toFixed(0)})`);
  const midOf = async () => { const d = await edgePis(hs.curve); return screenOf({ x: d[7], y: d[8] }); };
  const handle = async () => { const m = await midOf(), marks = await visible('.tool-marks > .pi-mark.curve'); return marks.sort((a, b) => Math.hypot(a.x - m.x, a.y - m.y) - Math.hypot(b.x - m.x, b.y - m.y))[0]; };
  await jump(await page.evaluate(async (e) => { const d = Array.from(await (await import('/src/workers/clockClient.ts')).queryClock({ what: 'edgePis', edge: e })); return window.tw.packToLngLat(d[7], d[8]); }, hs.curve), 14);
  await page.waitForTimeout(1500);
  await page.mouse.move(5, 5);
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}hsr-before.png` });
  const undo = async () => { await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true)); await page.waitForTimeout(1000); };
  // towards the curve's PI bends it sharper, away from it gentler
  const towardPi = async (h) => { const d = await edgePis(hs.curve), p = await screenOf({ x: d[1], y: d[2] }), l = Math.hypot(p.x - h.x, p.y - h.y) || 1; return [(p.x - h.x) / l, (p.y - h.y) / l]; };
  for (const [name, sign, px, ok] of [['sharper', 1, 15, (k) => k >= 300], ['gentler', -1, 15, (k) => k >= 319], ['much sharper', 1, 40, (k) => k > 165 && k < 319]]) {
    const h = await handle();
    if (!h) { report(false, `a bend ${name}`, 'no curve handle'); continue; }
    const [vx, vy] = await towardPi(h);
    await drag(h.x, h.y, sign * px * vx, sign * px * vy, `hsr-bend-${name.replace(' ', '-')}.png`);
    const s = await slowest(hs.curve);
    report(ok(s.kmh), `a ${px} px bend ${name}: ${px === 40 ? 'slows only as far as its legs need' : 'keeps the curve fast'}`, `${s.kmh.toFixed(0)} km/h, radius ${s.r.toFixed(0)} m (set ${s.set.toFixed(0)})`);
    await undo();
  }
  {
    const j = await screenOf(await nodeAt(hs.joint));
    await drag(j.x, j.y, 0, 15, 'hsr-joint.png');
    const a = await slowest(hs.straight), b = await slowest(hs.curve);
    report(a.kmh >= 300 && b.kmh >= 300, 'a 15 px drag of the joint keeps both pieces fast', `straight's new curve ${a.kmh.toFixed(0)} km/h (radius ${a.r.toFixed(0)} m), the curve ${b.kmh.toFixed(0)} km/h`);
    await undo();
  }
} catch (e) {
  failed = true;
  console.log(`FAIL  T-1898 check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
