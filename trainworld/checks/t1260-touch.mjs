// T-1260: building by touch. On a phone-sized page with touch (390 x 844, Playwright's hasTouch and
// isMobile), real taps on the map and on the control bar: track by tap, preview, confirm (and a
// second tap on the same spot), a one-finger pan placing nothing, a station on the track, a station
// on open ground, a line through two stations made with taps, the delete tool's cancel, and the
// Select tool's taps and held box. Then a desktop page: two mouse clicks still place a segment with
// no bar. Screenshots in artifacts/trainworld/t1260/.
//   node trainworld/checks/t1260-touch.mjs [url]   (from the maps root; default localhost:6165)
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { launchChrome, ARTIFACTS, GL } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6165/';
const OUT = join(ARTIFACTS, 't1260');
await mkdir(OUT, { recursive: true });
let failed = false;
let current = null;
const check = (ok, msg) => {
  failed ||= !ok;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`);
  if (!ok && process.env.T1260_DEBUG && current) void current.evaluate(() => JSON.stringify({ clicks: window.__t.clicks.slice(-4), s: window.__t.state() })).then((t) => console.log('   ', t));
};
const URL_NEW = `${BASE}?debug=1&new=1&opened=all&paused=1&v=${Date.now()}`;
const HIDE = '#debug, .loading-card, .loading-box { display: none !important; }';

/** In the page: helpers on window.__t. */
const HELPERS = async () => {
  const tw = window.tw, map = tw.map;
  const ui = await import('/src/game/ui.ts');
  const touch = await import('/src/map/touch.ts');
  const pick = await import('/src/game/blueprintPick.ts');
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const w = () => tw.world.value;
  const settle = async (pred, ms = 10000) => { const t = performance.now(); while (performance.now() - t < ms) { if (pred()) return true; await sleep(100); } return false; };
  /** a map point in page px */
  const at = (lng, lat) => { const p = map.project([lng, lat]), r = map.getContainer().getBoundingClientRect(); return { x: r.left + p.x, y: r.top + p.y }; };
  /** the page px of a fraction across and down the map */
  const frac = (fx, fy) => { const r = map.getContainer().getBoundingClientRect(); const x = r.left + r.width * fx, y = r.top + r.height * fy; const ll = map.unproject([x - r.left, y - r.top]); return { x, y, lng: ll.lng, lat: ll.lat }; };
  const state = () => ({
    edges: w().edges.length, stations: w().save.stations.length, lines: w().save.lines.length,
    drawing: tw.tools.drawing, ctl: touch.touchCtl.value, touchInput: touch.touchInput.value,
    picked: pick.bpPicked.value.edges.length + pick.bpPicked.value.stations.length,
    preview: tw.tools.lastPreview ? { ok: tw.tools.lastPreview.ok } : null,
    center: map.getCenter().toArray(), tool: ui.tool.value, tag: document.querySelector('.build-tag')?.textContent ?? '',
  });
  /** the bar's buttons, page px and size */
  const bar = () => [...document.querySelectorAll('.touch-bar button')].map((b) => { const r = b.getBoundingClientRect(); return { tip: b.dataset.tip, x: r.left + r.width / 2, y: r.top + r.height / 2, w: r.width, h: r.height, disabled: b.disabled, left: r.left, right: r.right, top: r.top, bottom: r.bottom }; });
  const clicks = [];
  map.on('click', (e) => clicks.push(`${ui.tool.value}@${Math.round(e.point.x)},${Math.round(e.point.y)}`));
  window.__t = { ui, touch, sleep, settle, at, frac, state, bar, w, clicks };
};

const tapAt = (page, p) => page.touchscreen.tap(Math.round(p.x), Math.round(p.y));
const st = (page) => page.evaluate(() => window.__t.state());
const waitFor = (page, fn, arg, ms = 10000) => page.waitForFunction(fn, arg, { timeout: ms }).then(() => true, () => false);
const bar = (page) => page.evaluate(() => window.__t.bar());
async function tapButton(page, tip) {
  const b = (await bar(page)).find((x) => x.tip === tip);
  if (!b) return false;
  await tapAt(page, b);
  return true;
}
/** a one-finger swipe by CDP touch events, `steps` moves */
async function swipe(page, from, to, steps = 8) {
  const cdp = await page.context().newCDPSession(page);
  const pt = (p) => [{ x: Math.round(p.x), y: Math.round(p.y), id: 1 }];
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: pt(from) });
  for (let i = 1; i <= steps; i++) {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: pt({ x: from.x + (to.x - from.x) * i / steps, y: from.y + (to.y - from.y) * i / steps }) });
    await page.waitForTimeout(16);
  }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await cdp.detach();
  // a tap during the fling stops it and is no tap, as on a phone
  await page.waitForTimeout(200);
  await page.waitForFunction(() => !window.tw.map.isMoving(), undefined, { timeout: 5000 }).catch(() => {});
  await page.waitForTimeout(300);
}
/** a held press then a drag, by CDP touch events */
async function holdDrag(page, from, to, holdMs = 650) {
  const cdp = await page.context().newCDPSession(page);
  const pt = (p) => [{ x: Math.round(p.x), y: Math.round(p.y), id: 1 }];
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: pt(from) });
  await page.waitForTimeout(holdMs);
  for (let i = 1; i <= 8; i++) {
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: pt({ x: from.x + (to.x - from.x) * i / 8, y: from.y + (to.y - from.y) * i / 8 }) });
    await page.waitForTimeout(16);
  }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await cdp.detach();
}

async function open(ctx) {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => { failed = true; console.log('pageerror', e.message); });
  await page.goto(URL_NEW);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.addStyleTag({ content: HIDE });
  current = page;
  await page.evaluate(HELPERS);
  // open ground in Queens at zoom 14, ground level, double track, keep placing on (the default)
  await page.evaluate(() => { const { ui } = window.__t; ui.buildLevel.value = 0; ui.autoLevel.value = false; ui.trackCount.value = 2; ui.highSpeed.value = false; ui.keepPlacing.value = true; ui.snapReal.value = false; window.tw.map.jumpTo({ center: [-73.885, 40.735], zoom: 14 }); });
  await page.waitForTimeout(1500);
  return page;
}

const touchCtx = await launchChrome({ prefix: 'tw-t1260-', size: [390, 844], args: GL, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });
try {
  const page = await open(touchCtx.ctx);
  // the Build tab and the Track tool, by taps
  await page.locator('button.tab[aria-label="Build"]').tap();
  await page.locator('.tool-pick button', { hasText: 'Track' }).tap();
  check((await st(page)).tool === 'track', 'tapping Build and Track picks the track tool');

  const A = await page.evaluate(() => window.__t.frac(0.25, 0.62));
  const B = await page.evaluate(() => window.__t.frac(0.7, 0.45));
  // under the demand buttons, which take the map's top quarter on a phone
  const C = await page.evaluate(() => window.__t.frac(0.85, 0.33));
  const s0 = await st(page);
  await tapAt(page, A);
  await page.waitForTimeout(400);
  let s = await st(page);
  check(s.touchInput && s.drawing && s.edges === s0.edges, 'the first tap sets the start and places nothing');
  check(s.ctl && s.ctl.confirm === null, 'with only a start the bar has no confirm button');
  await page.screenshot({ path: join(OUT, '1-first-tap.png') });

  await tapAt(page, B);
  const previewed = await waitFor(page, () => window.__t.state().preview?.ok && window.__t.state().ctl?.confirm === true);
  s = await st(page);
  check(previewed && s.edges === s0.edges, 'the second tap previews the segment (the ghost buildable) and places nothing');
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, '2-preview.png') });
  const buttons = await bar(page);
  const vp = page.viewportSize();
  check(buttons.length >= 4 && buttons.every((b) => b.w >= 44 && b.h >= 44), `the bar's ${buttons.length} buttons are at least 44 px (${buttons.map((b) => `${b.tip} ${b.w}x${b.h}`).join(', ')})`);
  check(buttons.every((b) => b.left >= 0 && b.right <= vp.width && b.top >= 0), 'the bar is inside the screen');
  check(buttons.every((b) => Math.hypot(b.x - B.x, b.y - B.y) > 40), 'no button is under the tapped point');

  // the bar's elevation button opens the seven elevations; one picked closes them
  await page.locator('.touch-row .btn.lv').tap();
  const lvOpen = await page.evaluate(() => document.querySelectorAll('.touch-levels .btn.lv').length);
  check(lvOpen === 7, `the elevation button opens the seven elevations (${lvOpen})`);
  if (lvOpen) {
    await page.locator('.touch-levels .btn.lv', { hasText: '0' }).first().tap();
    await page.waitForTimeout(200);
  }

  await tapButton(page, 'Place this track');
  const placed = await waitFor(page, (n) => window.__t.state().edges > n, s0.edges);
  s = await st(page);
  check(placed && s.drawing, 'confirm places the segment, and with keep placing the next starts at its end');
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(OUT, '3-confirmed.png') });

  // a one-finger pan while drawing places nothing
  const s1 = await st(page);
  await swipe(page, { x: vp.width * 0.4, y: vp.height * 0.3 }, { x: vp.width * 0.55, y: vp.height * 0.36 });
  await page.waitForTimeout(500);
  s = await st(page);
  check(s.edges === s1.edges && s.ctl?.confirm === null && (s.center[0] !== s1.center[0] || s.center[1] !== s1.center[1]), 'a one-finger pan moves the map and places nothing');
  await swipe(page, { x: vp.width * 0.55, y: vp.height * 0.36 }, { x: vp.width * 0.4, y: vp.height * 0.3 });
  await page.waitForTimeout(500);

  // C, then a second tap on C places it
  const C2 = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [C.lng, C.lat]);
  await tapAt(page, C2);
  await waitFor(page, () => window.__t.state().preview?.ok && window.__t.state().ctl?.confirm === true);
  await page.waitForTimeout(250);
  await tapAt(page, { x: C2.x + 6, y: C2.y - 5 });
  check(await waitFor(page, (n) => window.__t.state().edges > n, s1.edges), 'a second tap on the previewed end places it');
  await page.waitForTimeout(300);
  // back takes the last segment back; stop ends the run
  const s2 = await st(page);
  await tapButton(page, 'Take back the last point');
  check(await waitFor(page, (n) => window.__t.state().edges < n, s2.edges), 'back takes the last segment back');
  await tapAt(page, C2);
  await waitFor(page, () => window.__t.state().ctl?.confirm === true);
  check((await bar(page)).every((b) => b.left >= 0 && b.right <= vp.width), 'the bar keeps inside the screen when it grows');
  await tapButton(page, 'Place this track');
  await waitFor(page, (n) => window.__t.state().edges >= n, s2.edges);
  await tapButton(page, 'Stop drawing');
  await page.waitForTimeout(300);
  s = await st(page);
  check(!s.drawing && !s.ctl && s.edges === s2.edges, 'stop ends the run, the track kept');

  // stations: one on the first piece, one on the second, each tap then confirm
  await page.locator('.tool-pick button', { hasText: 'Station' }).tap();
  const mids = [[A, B], [B, C]].map(([p, q]) => ({ lng: (p.lng + q.lng) / 2, lat: (p.lat + q.lat) / 2 }));
  for (const [i, m] of mids.entries()) {
    const before = await st(page);
    const p = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [m.lng, m.lat]);
    await tapAt(page, p);
    const ready = await waitFor(page, () => window.__t.state().ctl?.confirm === true);
    s = await st(page);
    check(ready && s.stations === before.stations, `station ${i + 1}: a tap on the track shows the ghost and places nothing`);
    if (i === 0) await page.screenshot({ path: join(OUT, '4-station-ghost.png') });
    if (i === 0) await tapButton(page, 'Place the station');
    else await tapAt(page, p);
    check(await waitFor(page, (n) => window.__t.state().stations > n, before.stations), `station ${i + 1}: ${i === 0 ? 'confirm' : 'a second tap'} places it`);
    await page.waitForTimeout(400);
  }
  // a station on open ground: its end, its direction, confirm
  const E = [];
  {
    const before = await st(page);
    const e1 = await page.evaluate(() => window.__t.frac(0.35, 0.8)), e2 = await page.evaluate(() => window.__t.frac(0.55, 0.8));
    E.push(e1, e2);
    await tapAt(page, e1);
    await page.waitForTimeout(300);
    await tapAt(page, e2);
    const ready = await waitFor(page, () => window.__t.state().ctl?.confirm === true);
    s = await st(page);
    check(ready && s.stations === before.stations, 'open ground: two taps show the station and its track, nothing placed');
    await page.screenshot({ path: join(OUT, '5-free-station.png') });
    await tapButton(page, 'Place the station');
    check(await waitFor(page, (n) => window.__t.state().stations > n, before.stations), 'open ground: confirm places the station');
  }

  // a line through the two stations on the track, by taps
  {
    await page.locator('button.tab[aria-label="Lines"]').tap();
    await page.locator('button', { hasText: 'New line' }).tap();
    const editing = await waitFor(page, () => window.__t.ui.tool.value === 'line' && window.__t.ui.lineDraft.value.line !== null);
    check(editing, 'New line starts editing stops');
    await page.evaluate(() => document.activeElement?.blur?.());
    const ids = await page.evaluate(() => window.__t.w().save.stations.slice(-3, -1).map((x) => x.num));
    for (const id of ids) {
      const p = await page.evaluate((id) => { const n = window.__t.w().nodes.get(id); return window.__t.at(n.lng, n.lat); }, id);
      await tapAt(page, p);
      await page.waitForTimeout(700);
    }
    const stops = await page.evaluate(() => window.__t.ui.lineDraft.value.stops.length);
    check(stops === 2, `two taps on the stations make a line of two stops (${stops})`);
    await page.screenshot({ path: join(OUT, '6-line.png') });
    await page.locator('button', { hasText: 'Done' }).first().tap().catch(() => {});
    await page.waitForTimeout(300);
  }

  // delete: a tap shows what would go, cancel keeps it
  {
    await page.locator('button.tab[aria-label="Build"]').tap();
    await page.locator('.tool-pick button', { hasText: 'Delete' }).tap();
    const before = await st(page);
    const m = { lng: A.lng * 0.8 + B.lng * 0.2, lat: A.lat * 0.8 + B.lat * 0.2 };
    const p = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [m.lng, m.lat]);
    await tapAt(page, p);
    const shown = await waitFor(page, () => window.__t.state().ctl?.confirmTip === 'Remove');
    check(shown, 'delete: a tap on track shows what a confirm removes');
    await tapButton(page, 'Cancel');
    await page.waitForTimeout(300);
    s = await st(page);
    check(s.edges === before.edges && !s.ctl, 'delete: cancel removes nothing');
  }

  // Select: a tap adds a piece, a second tap takes it out, a held drag adds a box's worth
  {
    await page.locator('.tool-pick button', { hasText: 'Select' }).tap();
    await page.waitForTimeout(200);
    // the open-ground station, low on the map: the selection bar takes the middle of a phone's map
    const p = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [(E[0].lng + E[1].lng) / 2, (E[0].lat + E[1].lat) / 2]);
    await tapAt(page, p);
    await page.waitForTimeout(300);
    const one = (await st(page)).picked;
    await tapAt(page, p);
    await page.waitForTimeout(300);
    const none = (await st(page)).picked;
    check(one === 1 && none === 0, `select: a tap adds the piece, another takes it out (${one}, ${none})`);
    const c0 = (await st(page)).center;
    const f0 = await page.evaluate(() => window.__t.frac(0.4, 0.85)), f1 = await page.evaluate(() => window.__t.frac(0.55, 0.9));
    await swipe(page, f0, f1);
    s = await st(page);
    check(s.picked === 0 && (s.center[0] !== c0[0] || s.center[1] !== c0[1]), 'select: a one-finger drag pans');
    await swipe(page, f1, f0);
    await page.waitForTimeout(400);
    const a2 = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [A.lng, A.lat]);
    const c2 = await page.evaluate(([lng, lat]) => window.__t.at(lng, lat), [C.lng, C.lat]);
    await holdDrag(page, { x: Math.min(a2.x, c2.x) - 30, y: Math.min(a2.y, c2.y) - 30 }, { x: Math.max(a2.x, c2.x) + 30, y: Math.max(a2.y, c2.y) + 30 });
    await page.waitForTimeout(400);
    s = await st(page);
    check(s.picked >= 3, `select: a held press then a drag picks what its box holds (${s.picked})`);
    // T-1263: the selection bar on the map's bottom edge, one row of the screen's width, its actions icon-only
    const sb = await page.evaluate(() => {
      const b = document.querySelector('.blueprint-bar'), m = window.tw.map.getContainer().getBoundingClientRect();
      if (!b) return null;
      const r = b.getBoundingClientRect();
      const bs = [...b.querySelectorAll('button')].map((x) => x.getBoundingClientRect()).filter((x) => x.width > 0);
      const tops = new Set(bs.map((x) => Math.round(x.top)));
      return { atBottom: Math.abs(r.bottom - m.bottom) < 2, width: r.width, rows: tops.size, labels: [...b.querySelectorAll('.bp-label')].filter((x) => x.getBoundingClientRect().width > 0).length, mapW: m.width,
        fits: b.scrollWidth <= b.clientWidth && bs.every((x) => x.left >= 0 && x.right <= m.width), minH: Math.min(...bs.map((x) => x.height)), minW: Math.min(...bs.map((x) => x.width)) };
    });
    check(!!sb && sb.atBottom && sb.rows === 1 && sb.labels === 0 && sb.fits && sb.minH >= 36 && sb.minW >= 36 && Math.abs(sb.width - sb.mapW) < 2, `select: the bar fits one row of icons on the map's bottom edge at ${sb?.mapW} px, every button at least 36 px, nothing past the edge (${JSON.stringify(sb)})`);
    await page.screenshot({ path: join(OUT, '7-select-box.png') });
    // the one elevation button opens the seven above the bar; a pick closes them and levels the track
    await page.locator('.blueprint-bar .bp-level').tap();
    const opened = await page.evaluate(() => {
      const row = document.querySelector('.blueprint-levels'), bar = document.querySelector('.blueprint-bar')?.getBoundingClientRect();
      const r = row?.getBoundingClientRect(), bs = [...(row?.querySelectorAll('.btn.lv') ?? [])].map((x) => x.getBoundingClientRect());
      return { n: bs.length, above: !!r && !!bar && r.bottom <= bar.top + 1, inside: !!r && r.left >= 0 && r.right <= window.innerWidth, minW: Math.min(...bs.map((x) => x.width)), minH: Math.min(...bs.map((x) => x.height)) };
    });
    check(opened.n === 7 && opened.above && opened.inside && opened.minW >= 36 && opened.minH >= 36, `select: the elevation button opens seven buttons above the bar (${JSON.stringify(opened)})`);
    await page.screenshot({ path: join(OUT, '8-select-levels.png') });
    await page.locator('.blueprint-levels .btn.lv', { hasText: '+1' }).first().tap();
    await page.waitForTimeout(500);
    const after = await page.evaluate(() => ({ open: !!document.querySelector('.blueprint-levels'), label: document.querySelector('.blueprint-bar .bp-level')?.textContent }));
    check(!after.open && after.label === '+1', `select: a pick closes the seven and the button shows it (${JSON.stringify(after)})`);
  }
  await page.close();
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message);
} finally {
  await touchCtx.close();
}

// desktop: unchanged, two clicks place a segment and no bar shows
const deskCtx = await launchChrome({ prefix: 'tw-t1260d-', size: [1200, 800], args: GL });
try {
  const page = await open(deskCtx.ctx);
  await page.keyboard.press('1');
  const A = await page.evaluate(() => window.__t.frac(0.4, 0.6)), B = await page.evaluate(() => window.__t.frac(0.65, 0.4));
  const s0 = await st(page);
  await page.mouse.move(A.x, A.y);
  await page.mouse.click(A.x, A.y);
  await page.mouse.move(B.x, B.y, { steps: 5 });
  await waitFor(page, () => window.__t.state().preview?.ok);
  const s = await st(page);
  check(!s.touchInput && !s.ctl && s.drawing && s.edges === s0.edges, 'desktop: after the first click, the ghost follows the pointer, no bar');
  await page.mouse.click(B.x, B.y);
  check(await waitFor(page, (n) => window.__t.state().edges > n, s0.edges), 'desktop: the second click places the segment at once');
  await page.close();
} catch (e) {
  failed = true;
  console.log('FAIL  ' + e.message);
} finally {
  await deskCtx.close();
}
process.exitCode = failed ? 1 : 0;
