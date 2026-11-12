// T-3260..T-3263: the track tool's count, Ctrl's straight line, the road snap's ghost while the pointer moves,
// and quad track on roads. On a new New York game in headless Chrome, the real mouse and keyboard:
//  A (T-3262) the track bar: Single on Regular, then High speed shows double, Regular gives single back; Quad
//    stays through High speed and Tram; Single on High speed keeps High speed and double and says why; the
//    count picked survives a reload.
//  B (T-3260) a segment, then the next with Ctrl held: the ghost's end is on the straight line on from the first
//    (under 1 m off), the placed piece is straight; without Ctrl the same pointer curves; the tag says "Hold Ctrl
//    to draw straight" without Ctrl and not with it.
//  C (T-3263) Snap to roads with every ghost answer held 150 ms (a slow track model): a pointer moving every
//    8 ms for 2 s still updates the ghost (at least 3 times; it was 0 before T-3263); a click then places the road route.
//  D (T-3261) quad track on roads, two segments chained round a corner: both placed, and no node joins three
//    or more pieces (the pairs carry on side by side, no merge at the click).
// Pictures artifacts/trainworld/t3260/check-*.png. From the maps root, on a Vite of your own:
//   node trainworld/checks/t3260-track-bar.mjs [dev url]      (about 2 minutes)
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6143/';
const OUT = join(ARTIFACTS, 't3260');
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t3260-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { if (!sessionStorage.getItem('t3260')) { sessionStorage.setItem('t3260', '1'); localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { tram: false, roads: false, level: 1, keep: true, tracks: 2 } })); } } catch {} });
const errors = [];
collectErrors(page, errors);
const url = `${BASE}?debug=1&new=1&city=nyc&paused=1&demandWorkers=1&v=${Date.now()}`;
const ready = () => page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
// `f` runs in the page with game/ui.ts (a string through the debugger, which the page's CSP leaves alone)
const ui = async (f) => { await page.evaluate(async () => { window.__ui ??= await import('/src/game/ui.ts'); }); return page.evaluate(`(${f.toString()})(window.__ui)`); };
const notice = () => page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');
const state = () => ui((u) => ({ type: u.trackType.value, single: u.singleTrack.value, quad: u.quadTrack.value }));
try {
  await page.goto(url);
  await ready();
  await ui((u) => { u.tool.value = 'track'; });
  await wait(800);
  const bar = page.locator('.track-bar');
  const btn = (label) => bar.locator(`button[aria-label="${label}"]`);
  const type = (name) => bar.locator('.tb-wide button', { hasText: new RegExp(`^${name}$`) });

  // ---- A
  await btn('Single track').click();
  await type('High speed').click();
  let s = await state();
  check(s.type === 'hsr' && !s.single && !s.quad, 'A: Single then High speed: high speed, shown double', JSON.stringify(s));
  await type('Regular').click();
  s = await state();
  check(s.type === 2 && s.single, 'A: Regular again: single comes back', JSON.stringify(s));
  await type('Tram').click();
  await type('Regular').click();
  s = await state();
  check(s.type === 2 && s.single, 'A: through Tram and back: still single', JSON.stringify(s));
  await btn('Quad track').click();
  await type('High speed').click();
  s = await state();
  check(s.type === 'hsr' && s.quad, 'A: Quad then High speed: high-speed quad', JSON.stringify(s));
  await type('Tram').click();
  s = await state();
  check(s.type === 'tram' && s.quad, 'A: then Tram: tram quad', JSON.stringify(s));
  await type('High speed').click();
  await btn('Double track').click();
  await btn('Single track').click();
  s = await state();
  const said = await notice();
  check(s.type === 'hsr' && !s.single && !s.quad, 'A: Single on High speed keeps High speed and double', JSON.stringify(s));
  check(/can't be single track/.test(said), 'A: and says why', said);
  await page.screenshot({ path: join(OUT, 'check-a-hsr-single.png') });
  await type('Regular').click();
  await btn('Single track').click();
  await wait(500);
  await page.reload();
  await ready();
  s = await state();
  check(s.type === 2 && s.single, 'A: single is remembered across a reload', JSON.stringify(s));
  await ui((u) => { u.setDoubleTrack(); u.tool.value = 'track'; u.buildLevel.value = 1; u.keepPlacing.value = true; u.snapRoads.value = false; u.snapReal.value = false; u.snapBeside.value = false; });

  // ---- B: open water-free ground in Queens, zoom 15
  const box = await (await page.$('#map')).boundingBox();
  const go = async (c, z) => {
    await page.evaluate(([c, z]) => { window.tw.selection.value = null; window.tw.map.jumpTo({ center: c, zoom: z }); }, [c, z]);
    await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
    await wait(1500);
  };
  await go([-73.80, 40.735], 15);
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const settle = async () => { await page.waitForFunction(() => { const T = window.tw.tools; return T.lastPreview && T.previewKey === T.wantKey; }, undefined, { timeout: 20000, polling: 20 }).catch(() => {}); await wait(300); };
  await page.mouse.move(cx - 300, cy + 100, { steps: 3 });
  await page.mouse.click(cx - 300, cy + 100);
  await page.mouse.move(cx - 100, cy, { steps: 4 });
  await settle();
  await page.mouse.click(cx - 100, cy);
  await wait(1500);
  // the line on from (cx-100, cy) runs along (200, -100) on screen
  await page.mouse.move(cx + 150, cy + 20, { steps: 4 });
  await settle();
  const tagFree = await page.evaluate(() => window.tw.tools.tag.innerText);
  const curved = await page.evaluate(() => { const a = window.tw.tools.lastPreview?.arc; return a && a.length >= 3 ? a[2] : 0; });
  check(curved > 0, 'B: without Ctrl the segment curves', `arc radius ${Math.round(curved)}`);
  check(/Hold Ctrl to draw straight/.test(tagFree), 'B: the tag says Hold Ctrl to draw straight', tagFree.replace(/\n/g, ' | '));
  await page.screenshot({ path: join(OUT, 'check-b-no-ctrl.png') });
  await page.keyboard.down('Control');
  await page.mouse.move(cx + 151, cy + 21, { steps: 2 });
  await settle();
  const g = await page.evaluate(() => {
    const T = window.tw.tools, pv = T.lastPreview, st = T.start, [sx, sy] = window.tw.toLocal(st.lng, st.lat), [ux, uy] = st.dir;
    const v = pv.verts, ex = v[v.length - 4], ey = v[v.length - 3];
    const k = window.tw.anchor?.()?.k ?? 1;
    return { off: Math.abs((ex - sx) * uy - (ey - sy) * ux), ok: pv.ok, arc: pv.arc && pv.arc.length >= 3 ? pv.arc[2] : 0, tag: T.tag.innerText, n: v.length / 4 };
  });
  check(g.ok && g.off < 1 && g.n === 2, 'B: with Ctrl the ghost ends on the straight line on, no corner', JSON.stringify({ off: g.off, n: g.n, arc: g.arc }));
  check(!/Hold Ctrl/.test(g.tag), 'B: and the tag drops the Ctrl line', g.tag.replace(/\n/g, ' | '));
  await page.screenshot({ path: join(OUT, 'check-b-ctrl-ghost.png') });
  const before = await page.evaluate(() => window.tw.world.value.edges.length);
  await page.mouse.click(cx + 151, cy + 21);
  await wait(2000);
  await page.keyboard.up('Control');
  const placed = await page.evaluate((before) => window.tw.world.value.edges.length - before, before);
  check(placed === 1, 'B: a Ctrl click places the straight piece', `pieces +${placed}`);
  await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');
  await ui((u) => { u.tool.value = 'track'; });

  // ---- C and D: Snap to roads in Midtown
  await go([-73.986, 40.751], 15);
  const scr = async (p) => { const [x, y] = await page.evaluate(([a, b]) => { const q = window.tw.map.project([a, b]); return [q.x, q.y]; }, p); return [box.x + x, box.y + y]; };
  await ui((u) => { u.snapRoads.value = true; u.setDoubleTrack(); });
  await wait(500);
  const A = await scr([-73.9935, 40.7522]), B = await scr([-73.98505, 40.74865]);
  await page.mouse.move(A[0], A[1], { steps: 3 });
  await wait(1500);
  await page.mouse.click(A[0], A[1]);
  await wait(800);
  const sweep = await page.evaluate(async ([a, b]) => {
    const c = window.tw.client, orig = c.preview.bind(c);
    c.preview = (req, cb) => orig(req, (x) => setTimeout(() => cb(x), 150));
    // the fitted route's ghost and tag (the street's own line standing in is not counted)
    const st = window.tw.tools.street, sp = st.display.bind(st);
    let n = 0;
    st.display = (a, q) => { if (a) n++; return sp(a, q); };
    const el = window.tw.map.getCanvasContainer(), t0 = performance.now();
    await new Promise((done) => {
      const id = setInterval(() => {
        const e = performance.now() - t0;
        if (e > 2000) { clearInterval(id); return done(); }
        const f = 0.5 + 0.5 * (e / 2000);
        el.dispatchEvent(new MouseEvent('mousemove', { clientX: a[0] + (b[0] - a[0]) * f, clientY: a[1] + (b[1] - a[1]) * f + 30 * Math.sin(e / 150), bubbles: true }));
      }, 8);
    });
    delete st.display;
    c.preview = orig;
    return n;
  }, [A, B]);
  check(sweep >= 3, 'C: with a slow track model the road ghost still updates while the pointer moves', `${sweep} fitted ghosts in 2 s`);
  await page.mouse.move(B[0], B[1], { steps: 4 });
  await page.waitForFunction(() => window.tw.tools.street.phase === 'done', undefined, { timeout: 20000 }).catch(() => {});
  let e0 = await page.evaluate(() => window.tw.world.value.edges.length);
  await page.mouse.click(B[0], B[1]);
  await wait(2500);
  let e1 = await page.evaluate(() => window.tw.world.value.edges.length);
  check(e1 > e0, 'C: the click then places the road route', `pieces ${e0} -> ${e1}`);
  await page.keyboard.press('Escape');
  await page.keyboard.press('Escape');
  await page.evaluate(() => window.tw.client.edit?.({ op: 'undo' })).catch(() => {});
  await wait(1000);

  // ---- D
  await ui((u) => { u.tool.value = 'track'; u.setQuadTrack(true); u.snapRoads.value = true; });
  await wait(500);
  const D = [[-73.9935, 40.7522], [-73.98505, 40.74865], [-73.98165, 40.7534]];
  const before4 = await page.evaluate(() => window.tw.world.value.edges.map((e) => e.id));
  const oks = [];
  for (let k = 0; k < D.length; k++) {
    const [x, y] = await scr(D[k]);
    await page.mouse.move(x - 15, y - 8, { steps: 2 });
    await page.mouse.move(x, y, { steps: 3 });
    if (k > 0) await page.waitForFunction(() => window.tw.tools.street.phase === 'done' || window.tw.tools.street.phase === 'none', undefined, { timeout: 20000 }).catch(() => {});
    await wait(500);
    if (k > 0) oks.push(await page.evaluate(() => ({ ok: window.tw.tools.lastPreview?.ok, issues: window.tw.tools.lastPreview?.issues })));
    if (k === D.length - 1) await page.screenshot({ path: join(OUT, 'check-d-quad-roads-ghost.png') });
    await page.mouse.click(x, y);
    await wait(2500);
  }
  await page.keyboard.press('Escape');
  await wait(1000);
  await page.screenshot({ path: join(OUT, 'check-d-quad-roads.png') });
  await go(D[1], 17.5);
  await page.screenshot({ path: join(OUT, 'check-d-quad-corner.png') });
  const d = await page.evaluate((before) => {
    const es = window.tw.world.value.edges.filter((e) => !before.includes(e.id)), deg = new Map();
    for (const e of es) for (const n of [e.a, e.b]) deg.set(n, (deg.get(n) ?? 0) + 1);
    return { pieces: es.length, merges: [...deg.values()].filter((v) => v >= 3).length };
  }, before4);
  check(oks.every((o) => o.ok), 'D: both quad road segments can be built', JSON.stringify(oks));
  check(d.pieces >= 4 && d.merges === 0, 'D: the pairs carry on side by side, no merge at the click', JSON.stringify(d));
  check(errors.filter((e) => !/openfreemap|favicon/i.test(e)).length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(OUT, 'check-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3260 FAILED: ${fail}` : 'T-3260 PASSED');
process.exit(fail ? 1 : 0);
