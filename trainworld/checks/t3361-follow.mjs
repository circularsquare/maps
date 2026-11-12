// T-3361: the camera follows a train. A new game, a built line with trains running: the follow button in the
// train panel keeps the train in the middle of the map while it moves; a drag, a close, zooming far out and a
// second press end it; a double click on a train follows it and does not zoom; nothing hooks the frame when
// not following. Screenshots artifacts/trainworld/t3361-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3361-follow.mjs [dev url] [profile dir]
import { ARTIFACTS, acrossLine, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6195/';
const PROFILE = process.argv.slice(2).find((a) => !/^https?:/.test(a));
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const chrome = await launchChrome({ prefix: 'tw-t3361-', profile: PROFILE, size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors, /favicon|openfreemap|AJAXError|nodeGlobal/); // nodeGlobal: a known demand-worker error (notes/T-2747.md)
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
/** the first train drawn with its place on screen: its selection and where it is, px from the map's middle */
const where = () => page.evaluate(() => {
  const tw = window.tw, r = tw.renderer, n = r.net, s = tw.selection.value;
  if (s?.kind !== 'train') return null;
  const i = r.tripIndex(s.profile, s.dep - r.tripsEpoch);
  if (i < 0) return null;
  const at = tw.tripAt(n, s.profile, s.dep + r.tripLag(i), tw.clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
  if (!at) return null;
  const p = tw.map.project(tw.toLngLat(at.x, at.y)), b = tw.map.getCanvas().getBoundingClientRect();
  return { x: p.x, y: p.y, dx: p.x - b.width / 2, dy: p.y - b.height / 2, left: b.left, top: b.top, zoom: tw.map.getZoom() };
});
const state = () => page.evaluate(() => ({ following: !!window.tw.overlay.afterDraw, on: !!document.querySelector('#inspector .title .btn.icon.on'), sel: window.tw.selection.value?.kind ?? null, zoom: window.tw.map.getZoom(), tip: document.querySelector('#inspector .title .btn.icon:last-of-type')?.getAttribute('aria-label') }));
try {
  await page.goto(`${BASE}?debug=1&new=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 300000 });
  const line = await acrossLine(page, [-71.06, 42.36], 6);
  check(line !== null, 'a line is built');
  await wait(2500);
  await page.evaluate(() => { const tw = window.tw; tw.clock.setSpeed(1); tw.map.jumpTo({ center: [-71.06, 42.36], zoom: 12.5 }); });
  // a train that is on its way
  const picked = await page.waitForFunction(() => {
    const tw = window.tw, r = tw.renderer, n = r.net;
    if (!n || !r.tripCount) return false;
    for (let i = 0; i < r.tripCount; i++) {
      const profile = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i);
      const at = tw.tripAt(n, profile, dep, tw.clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
      if (at && at.s > 300) { tw.selection.value = { kind: 'train', line: String(at.line), profile, dep }; return true; }
    }
    return false;
  }, undefined, { timeout: 120000, polling: 500 });
  check(!!picked, 'a train is selected');
  await wait(800);
  let s = await state();
  check(!s.following && !s.on, 'not following to begin with: no frame hook', JSON.stringify(s));
  await page.screenshot({ path: join(ARTIFACTS, 't3361-panel-before.png') });
  const btn = page.locator('#inspector .title .btn.icon').last();
  await btn.hover();
  await wait(300);
  await page.screenshot({ path: join(ARTIFACTS, 't3361-hover.png') });
  await btn.click();
  await wait(1500);
  s = await state();
  check(s.following && s.on, 'the button starts following', JSON.stringify(s));
  // the train stays in the middle while it moves: its distance from the middle, read in each animation frame for 4 s
  const track = (ms) => page.evaluate((ms) => new Promise((res) => {
    const tw = window.tw, r = tw.renderer, offs = [], t0 = performance.now();
    const f = () => {
      const s = tw.selection.value, i = s?.kind === 'train' ? r.tripIndex(s.profile, s.dep - r.tripsEpoch) : -1;
      const at = i >= 0 ? tw.tripAt(r.net, s.profile, s.dep + r.tripLag(i), tw.clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity) : null;
      if (at) { const p = tw.map.project(tw.toLngLat(at.x, at.y)), b = tw.map.getCanvas().getBoundingClientRect(); offs.push(Math.hypot(p.x - b.width / 2, p.y - b.height / 2)); }
      if (performance.now() - t0 < ms) requestAnimationFrame(f); else res(offs);
    };
    requestAnimationFrame(f);
  }), ms);
  const stat = (o) => { const a = [...o].sort((x, y) => x - y); return { n: a.length, med: a[a.length >> 1], p95: a[Math.floor(a.length * 0.95)], max: a.at(-1) }; };
  const offs = stat(await track(4000));
  console.log('      offsets', JSON.stringify(offs));
  check(offs.n > 100 && offs.p95 < 6, 'the train stays in the middle', JSON.stringify(offs));
  const moved = await page.evaluate(() => window.tw.map.getCenter().lng);
  await wait(1500);
  const moved2 = await page.evaluate(() => window.tw.map.getCenter().lng);
  check(Math.abs(moved2 - moved) > 1e-5, 'the map moves with it', `${moved} -> ${moved2}`);
  await page.screenshot({ path: join(ARTIFACTS, 't3361-following.png') });
  // frame rate while following
  const fps = await page.evaluate(() => new Promise((res) => { let n = 0; const t0 = performance.now(); const f = () => { n++; if (performance.now() - t0 < 2000) requestAnimationFrame(f); else res(n / 2); }; requestAnimationFrame(f); }));
  console.log('      frames per second while following', fps);
  // a wheel zoom keeps the train in the middle
  const w0 = await where();
  await page.mouse.move(w0.left + 200, w0.top + 200);
  await page.mouse.wheel(0, -300);
  await wait(1000);
  const z1 = (await state()).zoom;
  const zo = stat(await track(2000));
  check(z1 > w0.zoom + 0.1 && zo.p95 < 6, 'a wheel zoom keeps the train in the middle', `zoom ${w0.zoom.toFixed(2)} -> ${z1.toFixed(2)}, ${JSON.stringify(zo)}`);
  // a drag ends it
  await page.mouse.move(w0.left + 300, w0.top + 300);
  await page.mouse.down();
  await page.mouse.move(w0.left + 360, w0.top + 330, { steps: 5 });
  await page.mouse.up();
  await wait(500);
  s = await state();
  check(!s.following && !s.on && s.sel === 'train', 'a drag ends following and keeps the panel', JSON.stringify(s));
  // again, then the button ends it
  await btn.click();
  await wait(800);
  check((await state()).following, 'following again');
  await btn.click();
  await wait(500);
  s = await state();
  check(!s.following && !s.on, 'the button ends it', JSON.stringify(s));
  // closing the panel ends it
  await btn.click();
  await wait(600);
  await page.evaluate(() => { window.tw.selection.value = null; });
  await wait(300);
  check(!(await state()).following, 'closing the panel ends it');
  // zooming far out ends it
  await page.evaluate(() => { window.tw.map.jumpTo({ zoom: 12.5 }); });
  const again = await page.waitForFunction(() => {
    const tw = window.tw, r = tw.renderer, n = r.net;
    for (let i = 0; i < r.tripCount; i++) {
      const profile = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i);
      const at = tw.tripAt(n, profile, dep, tw.clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
      if (at) { tw.selection.value = { kind: 'train', line: String(at.line), profile, dep }; return true; }
    }
    return false;
  }, undefined, { timeout: 60000, polling: 500 });
  await wait(500);
  await page.locator('#inspector .title .btn.icon').last().click();
  await wait(600);
  check((await state()).following, 'following a second train');
  await page.evaluate(() => { window.tw.map.jumpTo({ zoom: 5 }); });
  await wait(800);
  check(!(await state()).following, 'zooming far out ends it', JSON.stringify(await state()));
  // a double click on a train follows it and does not zoom
  await page.evaluate(() => { window.tw.selection.value = null; window.tw.map.jumpTo({ zoom: 13 }); });
  await wait(600);
  await page.evaluate(() => { window.tw.clock.setRunning(false); });
  await wait(300);
  const at = await page.evaluate(() => {
    const tw = window.tw, r = tw.renderer, n = r.net, b = tw.map.getCanvas().getBoundingClientRect(), out = [];
    for (let i = 0; i < r.tripCount; i++) {
      const profile = r.trips[i * 2], dep = r.trips[i * 2 + 1] + r.tripsEpoch + r.tripLag(i);
      const a = tw.tripAt(n, profile, dep, tw.clock.now(), r.tripEnds ? r.tripEnds[i] + r.tripsEpoch : -Infinity);
      if (!a) continue;
      const p = tw.map.project(tw.toLngLat(a.x, a.y));
      if (p.x > 60 && p.x < b.width - 60 && p.y > 60 && p.y < b.height - 60 && tw.pick(tw.overlay, p.x, p.y, true, null)?.kind === 'train') out.push({ x: b.left + p.x, y: b.top + p.y });
    }
    return out[0];
  });
  check(!!at, 'a train is on screen to double click');
  const z0 = (await state()).zoom;
  await page.mouse.dblclick(at.x, at.y);
  await wait(800);
  s = await state();
  check(s.sel === 'train' && s.following && s.on, 'a double click on a train follows it', JSON.stringify(s));
  check(Math.abs(s.zoom - z0) < 0.01, 'and does not zoom', `${z0} -> ${s.zoom}`);
  await page.screenshot({ path: join(ARTIFACTS, 't3361-dblclick.png') });
  check(errors.length === 0, 'no page errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  console.log('ERROR', e.stack ?? e);
  fail++;
} finally {
  await chrome.close();
}
console.log(fail ? `${fail} FAILED` : 'all passed');
process.exit(fail ? 1 : 0);
