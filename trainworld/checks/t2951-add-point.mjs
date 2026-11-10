// T-2951: adding a point to placed blueprint track and dragging it, with real mouse events in
// headless Chrome; the round curve handles are gone.
//   node trainworld/checks/t2951-add-point.mjs [url]      (from the maps root; about a minute)
// 1. Anita's case: two segments drawn with the track tool, the middle dragged with the select tool:
//    every square drawn sits on the track, none round.
// 2. A double click on blueprint track adds a point there (one undo takes it away), and does not zoom.
// 3. Dragging the new point bends the track through it; one undo for the drag, one for the point.
// 4. The track inspector's Add point puts one halfway along.
// 5. A double click on constructed track adds nothing and says why.
// Screenshots to artifacts/trainworld/trackvertex/. One PASS or FAIL line each.
import { mkdir } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6310/';
const OUT = `${ARTIFACTS}trackvertex/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t2951-', size: [1200, 800], args: GL });
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
  await page.evaluate(() => window.tw.map.jumpTo({ center: [-73.97, 40.775], zoom: 15 }));
  await page.waitForTimeout(1500);
  const setTool = (t) => page.evaluate(async (t) => { (await import('/src/game/ui.ts')).tool.value = t; }, t);
  const counts = () => page.evaluate(() => ({ edges: window.tw.world.value.edges.length, nodes: [...window.tw.world.value.nodes.values()].filter((n) => !n.bus && !n.port).length }));
  const zoom = () => page.evaluate(() => window.tw.map.getZoom());
  const undo = async () => { await page.evaluate(() => window.tw.client.edit({ op: 'undo' }, true)); await page.waitForTimeout(900); };
  const rect = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { left: r.left, top: r.top }; });
  const click = async (x, y) => { await page.mouse.move(x, y); await page.waitForTimeout(150); await page.mouse.click(x, y); await page.waitForTimeout(700); };
  const drag = async (x, y, dx, dy) => {
    await page.mouse.move(x, y); await page.waitForTimeout(200); await page.mouse.down();
    for (let i = 1; i <= 10; i++) { await page.mouse.move(Math.round(x + (dx * i) / 10), Math.round(y + (dy * i) / 10)); await page.waitForTimeout(40); }
    await page.waitForTimeout(1000);
    await page.mouse.up(); await page.waitForTimeout(1500);
  };
  // each point's mark drawn, page px, and its distance from the drawn track (px, to the nearest
  // segment); a selected piece's corner squares (map/edgeEdit.ts, `.drag`) sit off it by design
  const marks = () => page.evaluate(() => {
    const ov = window.tw.overlay, net = ov.renderer.net, r = window.tw.map.getContainer().getBoundingClientRect(), out = [];
    for (const m of document.querySelectorAll('.tool-marks > .pi-mark:not(.drag), .tool-marks > .end-mark')) {
      if (m.style.display === 'none') continue;
      const b = m.getBoundingClientRect(), px = b.left + b.width / 2 - r.left, py = b.top + b.height / 2 - r.top;
      let best = 1e9;
      for (const pts of net.edgePts.values()) {
        let [x0, y0] = ov.toScreen(pts[0], pts[1]);
        for (let k = 3; k + 1 < pts.length; k += 3) {
          const [x1, y1] = ov.toScreen(pts[k], pts[k + 1]), dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy || 1;
          const u = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / l2));
          best = Math.min(best, Math.hypot(x0 + u * dx - px, y0 + u * dy - py));
          [x0, y0] = [x1, y1];
        }
      }
      out.push({ round: getComputedStyle(m).borderRadius !== '0px' && m.classList.contains('pi-mark'), x: b.left + b.width / 2, y: b.top + b.height / 2, off: best });
    }
    return out;
  });
  // the drawn piece's point halfway along it on screen, page px
  const halfway = (edge) => page.evaluate(([e, r]) => {
    const ov = window.tw.overlay, pts = ov.renderer.net.edgePts.get(e), s = [];
    for (let k = 0; k + 1 < pts.length; k += 3) s.push(ov.toScreen(pts[k], pts[k + 1]));
    let total = 0;
    for (let i = 1; i < s.length; i++) total += Math.hypot(s[i][0] - s[i - 1][0], s[i][1] - s[i - 1][1]);
    for (let i = 1, run = 0; i < s.length; i++) {
      const l = Math.hypot(s[i][0] - s[i - 1][0], s[i][1] - s[i - 1][1]);
      if (run + l >= total / 2) { const u = (total / 2 - run) / l; return { x: Math.round(s[i - 1][0] + u * (s[i][0] - s[i - 1][0]) + r.left), y: Math.round(s[i - 1][1] + u * (s[i][1] - s[i - 1][1]) + r.top) }; }
      run += l;
    }
    return null;
  }, [edge, rect]);
  const nodeScreen = (id) => page.evaluate(([id, r]) => { const n = window.tw.world.value.nodes.get(id), p = window.tw.map.project([n.lng, n.lat]); return { x: p.x + r.left, y: p.y + r.top }; }, [id, rect]);

  // 1: her case
  await setTool('track');
  await page.waitForTimeout(300);
  await click(300, 450);
  await click(600, 400);
  await click(900, 350);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(500);
  await setTool('select');
  await page.waitForTimeout(800);
  const drawn = await page.evaluate(() => window.tw.world.value.edges.map((e) => ({ id: e.id, a: e.a, b: e.b })));
  const mid = drawn.length === 2 ? drawn[0].b : -1;
  if (mid < 0 || drawn[1].a !== mid) throw new Error('setup: two pieces through a middle node, got ' + JSON.stringify(drawn));
  const m0 = await nodeScreen(mid);
  await drag(m0.x, m0.y, 0, -80);
  await page.mouse.move(5, 5);
  await page.waitForTimeout(600);
  const m1 = await nodeScreen(mid);
  let ms = await marks();
  await page.screenshot({ path: `${OUT}1-middle-dragged.png` });
  report(Math.hypot(m1.x - m0.x, m1.y - m0.y + 80) < 2, 'the middle point goes where it is dragged', `moved ${(m1.x - m0.x).toFixed(1)}, ${(m1.y - m0.y).toFixed(1)} px`);
  report(ms.length >= 3 && ms.every((m) => !m.round && m.off < 3), 'after the drag no round handles, and every square sits on the track', ms.map((m) => `${m.round ? 'round' : 'square'} ${m.off.toFixed(1)} px`).join(', '));

  // 2: a double click on the right piece adds a point
  const c0 = await counts(), z0 = await zoom();
  const on = await halfway(drawn[1].id);
  await page.mouse.move(on.x, on.y);
  await page.waitForTimeout(200);
  await page.mouse.dblclick(on.x, on.y);
  await page.waitForTimeout(1500);
  await page.mouse.move(5, 5);
  const c1 = await counts(), z1 = await zoom();
  await page.screenshot({ path: `${OUT}2-point-added.png` });
  const added = await page.evaluate((before) => [...window.tw.world.value.nodes.values()].find((n) => !n.bus && !n.port && !before.includes(n.id))?.id ?? -1, drawn.flatMap((e) => [e.a, e.b]));
  const pAdd = added >= 0 ? await nodeScreen(added) : null;
  report(c1.edges === c0.edges + 1 && c1.nodes === c0.nodes + 1 && pAdd && Math.hypot(pAdd.x - on.x, pAdd.y - on.y) < 3, 'a double click on blueprint track adds a point where it was clicked', `${c0.edges} -> ${c1.edges} pieces, point ${pAdd ? Math.hypot(pAdd.x - on.x, pAdd.y - on.y).toFixed(1) : '-'} px from the click`);
  report(Math.abs(z1 - z0) < 1e-6, 'a double click does not zoom the map', `zoom ${z0.toFixed(3)} -> ${z1.toFixed(3)}`);
  ms = await marks();
  report(ms.some((m) => pAdd && Math.hypot(m.x - pAdd.x, m.y - pAdd.y) < 2), 'the new point has its square');

  // 3: drag it; the track goes through it; one undo each
  if (pAdd) {
    await drag(pAdd.x, pAdd.y, 20, 60);
    await page.mouse.move(5, 5);
    await page.waitForTimeout(600);
    const p2 = await nodeScreen(added);
    ms = await marks();
    await page.screenshot({ path: `${OUT}3-point-dragged.png` });
    const sq = ms.find((m) => Math.hypot(m.x - p2.x, m.y - p2.y) < 2);
    report(Math.hypot(p2.x - pAdd.x - 20, p2.y - pAdd.y - 60) < 2 && sq && sq.off < 3, 'dragging the new point reshapes the track through it', `point moved ${(p2.x - pAdd.x).toFixed(1)}, ${(p2.y - pAdd.y).toFixed(1)} px, track ${sq ? sq.off.toFixed(1) : '-'} px from it`);
    report(ms.every((m) => !m.round && m.off < 3), 'every square still on the track', ms.map((m) => m.off.toFixed(1)).join(', '));
    await undo();
    const p3 = await nodeScreen(added);
    report(Math.hypot(p3.x - pAdd.x, p3.y - pAdd.y) < 1.5 && (await counts()).edges === c1.edges, 'one undo puts the point back where it was added');
    await undo();
    report((await counts()).edges === c0.edges, 'a second undo takes the point away');
  }

  // 4: the inspector's button
  await page.evaluate(async (e) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'track', edge: e }; }, drawn[0].id);
  await page.waitForTimeout(800);
  const btn = page.locator('button[aria-label="Add point"]');
  const has = await btn.count();
  report(has === 1, 'a blueprint piece\'s inspector has Add point');
  if (has) {
    const len0 = await page.evaluate((e) => window.tw.world.value.edges.find((x) => x.id === e).lengthM, drawn[0].id);
    await btn.click();
    await page.waitForTimeout(1500);
    await page.mouse.move(5, 5);
    await page.screenshot({ path: `${OUT}4-inspector-button.png` });
    const len1 = await page.evaluate((e) => window.tw.world.value.edges.find((x) => x.id === e).lengthM, drawn[0].id);
    report((await counts()).edges === c0.edges + 1 && Math.abs(len1 - len0 / 2) < 3, 'Add point splits the piece halfway', `${len0.toFixed(0)} m -> ${len1.toFixed(0)} m`);
    await undo();
  }

  // 5: constructed track
  await page.evaluate(async () => { await window.tw.client.edit({ op: 'constructAll' }); });
  await page.waitForTimeout(1500);
  const c5 = await counts();
  const on5 = await halfway(drawn[0].id);
  await page.mouse.dblclick(on5.x, on5.y);
  await page.waitForTimeout(1500);
  const said = await page.evaluate(() => document.body.innerText.includes('Built track cannot be moved'));
  await page.screenshot({ path: `${OUT}5-built-refused.png` });
  report((await counts()).edges === c5.edges && said, 'a double click on constructed track adds nothing and says why', `said ${said}`);
} catch (e) {
  failed = true;
  console.log(`FAIL  T-2951 check  ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
