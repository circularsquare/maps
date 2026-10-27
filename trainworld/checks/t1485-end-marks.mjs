// T-1485: the track tool's loose-end rings. A new game in Wuhan with a line and three track stubs;
// with the track tool on, rings show only near the pointer (within 160 px), each on the end of
// track the network draws, none with the pointer off the map. T-1522: none with the pointer on a
// control over the map while keys pan it, near the pointer after a drag pan, none once the tool changes.
// T-1534: rings only on track that is visibly drawn: built and active blueprint yes, a switched-off
// blueprint no, a stub under 4 px long on screen no.
// SHOT=<png> saves the view after the key pans. About a minute.
// `node trainworld/checks/t1485-end-marks.mjs [url]` from the maps root, on your own Vite.
import { buildLine, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6152/';
let failed = 0;
const say = (ok, what, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? `  (${detail})` : ''}`); };

/** the rings shown, CSS px in the map, and each one's distance to the nearest drawn track end */
const rings = (page) => page.evaluate(() => {
  const tw = window.tw, net = tw.renderer.net, box = tw.map.getContainer().getBoundingClientRect();
  const ends = [];
  for (const p of net.edgePts.values()) for (const k of [0, p.length - 3]) ends.push(tw.overlay.toScreen(p[k], p[k + 1]));
  return [...document.querySelectorAll('.end-mark')].filter((m) => m.style.display !== 'none').map((m) => {
    const r = m.getBoundingClientRect(), x = r.x + r.width / 2 - box.x, y = r.y + r.height / 2 - box.y;
    return { x, y, off: Math.min(...ends.map(([a, b]) => Math.hypot(a - x, b - y))) };
  });
});

const b = await launchChrome({ prefix: 'tw-t1485-' });
const errors = [];
try {
  const page = await b.ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?new=1&city=wuh&debug=1&paused=1`);
  await page.waitForFunction(() => window.tw?.client && window.tw.world?.value, null, { timeout: 180000 });
  const [lng, lat] = [114.30, 30.58];
  const d = 3 / (111.32 * Math.cos((lat * Math.PI) / 180));
  await buildLine(page, [-1, 0, 1].map((k) => [lng + k * d, lat]), { trains: [6, 4, 3] });
  await page.evaluate(async () => {
    const tw = window.tw, c = tw.client;
    for (const [a, z] of [[[114.33, 30.45], [114.37, 30.43]], [[114.22, 30.45], [114.19, 30.42]], [[114.42, 30.50], [114.45, 30.47]]]) {
      const p = tw.lngLatToPack(...a), q = tw.lngLatToPack(...z);
      await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
      await new Promise((r) => setTimeout(r, 300));
    }
    await c.edit({ op: 'constructAll' });
  });
  await page.evaluate(() => window.tw.map.jumpTo({ center: [114.31, 30.5], zoom: 10.6 }));
  await page.waitForTimeout(1500);
  const loose = await page.evaluate(() => [...window.tw.world.value.nodes.values()].filter((n) => n.ports === 1 && !(n.platform > 0)).map((n) => { const p = window.tw.map.project([n.lng, n.lat]); return [p.x, p.y]; }));
  say(loose.length === 6, 'six loose ends built', String(loose.length));
  const box = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return [r.x, r.y]; });
  // the pointer by the first stub's first end
  const [px, py] = loose[0];
  await page.mouse.move(box[0] + px + 20, box[1] + py + 20);
  await page.keyboard.press('1');
  await page.waitForTimeout(800);
  const near = await rings(page);
  const wantNear = loose.filter(([x, y]) => Math.hypot(x - px - 20, y - py - 20) <= 160).length;
  say(near.length === wantNear && near.length > 0 && near.length < loose.length, 'track tool: rings only near the pointer', `${near.length} shown, ${wantNear} within 160 px, ${loose.length} in all`);
  say(near.every((r) => r.off < 1.5), 'each ring on a drawn track end', near.map((r) => r.off.toFixed(1)).join(', '));
  await page.mouse.move(box[0] - 5 + 1, box[1] + 300);
  await page.mouse.move(5, 300);
  await page.waitForTimeout(300);
  const off = await rings(page);
  say(off.length === 0, 'no rings with the pointer off the map', String(off.length));
  await page.mouse.move(box[0] + px + 20, box[1] + py + 20);
  await page.keyboard.press('2');
  await page.waitForTimeout(500);
  say((await rings(page)).length === wantNear, 'station tool: the same rings');

  // T-1522: the pointer from the map onto a control laid over it (MapLibre's mousemove stopped there and
  // left the rings round the last map point, where key pans brought rings in and out)
  const ctrl = await page.evaluate(() => {
    const m = window.tw.map, cc = m.getCanvasContainer();
    const el = [...m.getContainer().querySelectorAll('button')].find((b) => !cc.contains(b) && b.offsetParent && b.getBoundingClientRect().width > 0);
    const r = el.getBoundingClientRect();
    return [r.x + r.width / 2, r.y + r.height / 2];
  });
  // the first stub's end brought just under the control, the pointer from there onto it
  await page.evaluate(([dx, dy]) => window.tw.map.panBy([dx, dy], { animate: false }), [box[0] + px - ctrl[0], box[1] + py - ctrl[1] - 150]);
  await page.waitForTimeout(500);
  await page.mouse.move(ctrl[0] + 10, ctrl[1] + 140, { steps: 4 });
  await page.waitForTimeout(200);
  say((await rings(page)).length > 0, 'rings with the pointer on the map under the control');
  await page.mouse.move(ctrl[0], ctrl[1], { steps: 4 });
  await page.waitForTimeout(300);
  say((await rings(page)).length === 0, 'no rings with the pointer on a control over the map', String((await rings(page)).length));
  let most = 0;
  for (const key of ['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft']) {
    await page.keyboard.down(key); await page.waitForTimeout(500); await page.keyboard.up(key); await page.waitForTimeout(500);
    most = Math.max(most, (await rings(page)).length);
  }
  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
  say(most === 0, 'none while the map is panned by keys under it', `${most} at most`);

  // a drag pan: rings round where the pointer is at the end, on the ends where the map now draws them
  await page.evaluate(([lng, lat]) => window.tw.map.jumpTo({ center: [lng, lat], zoom: 10.6 }), [114.31, 30.5]);
  await page.waitForTimeout(800);
  const start = [box[0] + px + 20, box[1] + py + 20];
  const dx = px < (await page.evaluate(() => window.tw.overlay.cssW)) / 2 ? 250 : -250;
  await page.mouse.move(start[0], start[1], { steps: 6 });
  await page.mouse.down();
  await page.mouse.move(start[0] + dx / 2, start[1], { steps: 12 });
  await page.mouse.move(start[0] + dx, start[1], { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(800);
  const dragged = await rings(page), at = [start[0] + dx - box[0], start[1] - box[1]];
  const strays = dragged.filter((r) => Math.hypot(r.x - at[0], r.y - at[1]) > 162 || r.off > 1.5);
  say(dragged.length > 0 && strays.length === 0, 'after a drag pan: rings near the pointer, on track ends', `${dragged.length} shown, ${strays.length} astray`);

  // T-1534: rings only at the loose ends of track drawn so it can be seen: an active blueprint yes, a
  // switched-off one no, a stub under 4 px long on screen no (until the map is zoomed in)
  const stub = (a, z, off) => page.evaluate(async ([a, z, off]) => {
    const tw = window.tw, c = tw.client, before = new Set(tw.world.value.edges.map((e) => e.id));
    const p = tw.lngLatToPack(...a), q = tw.lngLatToPack(...z);
    const r = await c.edit({ op: 'route', from: { kind: 'free', x: p.x, y: p.y, level: 0 }, to: { kind: 'free', x: q.x, y: q.y, level: 0 }, pis: [], single: false });
    await new Promise((res) => setTimeout(res, 300));
    const edges = tw.world.value.edges.filter((e) => !before.has(e.id)).map((e) => e.id);
    if (off && edges.length) await c.edit({ op: 'blueprintOff', edges, stations: [], off: true }, true);
    await new Promise((res) => setTimeout(res, 300));
    return { ok: r.ok, edges };
  }, [a, z, off]);
  /** per edge: how many of its two ends have a ring on them, and its length on screen in px */
  const endRings = (ids) => page.evaluate((ids) => {
    const tw = window.tw, net = tw.renderer.net, box = tw.map.getContainer().getBoundingClientRect();
    const marks = [...document.querySelectorAll('.end-mark')].filter((m) => m.style.display !== 'none').map((m) => { const r = m.getBoundingClientRect(); return [r.x + r.width / 2 - box.x, r.y + r.height / 2 - box.y]; });
    return ids.map((id) => {
      const p = net.edgePts.get(id), ends = [tw.overlay.toScreen(p[0], p[1]), tw.overlay.toScreen(p[p.length - 3], p[p.length - 2])];
      return { rings: ends.filter(([x, y]) => marks.some(([a, b]) => Math.hypot(a - x, b - y) < 3)).length, px: Math.hypot(ends[0][0] - ends[1][0], ends[0][1] - ends[1][1]) };
    });
  }, ids);
  const live = await stub([114.26, 30.40], [114.29, 30.39], false);
  const dead = await stub([114.26, 30.37], [114.29, 30.36], true);
  const tiny = await stub([114.299, 30.375], [114.3002, 30.375], false);
  say(live.ok && dead.ok && tiny.ok && live.edges.length > 0 && dead.edges.length > 0 && tiny.edges.length > 0, 'three blueprint stubs routed', JSON.stringify([live, dead, tiny].map((s) => s.edges)));
  const offNow = await page.evaluate((ids) => window.tw.world.value.edges.filter((e) => ids.includes(e.id)).map((e) => !!e.off), dead.edges);
  say(offNow.length > 0 && offNow.every(Boolean), 'one of them switched off', JSON.stringify(offNow));
  await page.keyboard.press('1');
  await page.evaluate(() => window.tw.map.jumpTo({ center: [114.30, 30.38], zoom: 10.6 }));
  await page.waitForTimeout(800);
  const mid = await page.evaluate(() => [window.tw.overlay.cssW / 2, window.tw.overlay.cssH / 2]);
  await page.mouse.move(box[0] + mid[0], box[1] + mid[1], { steps: 4 });
  await page.waitForTimeout(500);
  const [rl, rd, rt] = await endRings([live.edges[0], dead.edges[0], tiny.edges[0]]);
  say(rl.rings === 2, 'an active blueprint: rings on both ends', JSON.stringify(rl));
  say(rd.rings === 0 && rd.px > 8, 'a switched-off blueprint, long enough to see: no rings', JSON.stringify(rd));
  say(rt.rings === 0 && rt.px < 4, 'a stub under 4 px on screen: no rings', JSON.stringify(rt));
  await page.evaluate(() => window.tw.map.jumpTo({ center: [114.2996, 30.375], zoom: 13.6 }));
  await page.waitForTimeout(800);
  await page.mouse.move(box[0] + mid[0] + 10, box[1] + mid[1] + 5, { steps: 4 });
  await page.waitForTimeout(500);
  const [zt] = await endRings([tiny.edges[0]]);
  say(zt.rings === 2 && zt.px >= 4, 'the same stub zoomed in: rings on both ends', JSON.stringify(zt));

  // the tool changes: gone at once
  await page.keyboard.press('5');
  await page.waitForTimeout(200);
  say((await rings(page)).length === 0, 'select tool: no rings');
  await page.keyboard.press('1');
  await page.waitForTimeout(200);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  say((await rings(page)).length === 0, 'tool put down: no rings');
} catch (e) {
  say(false, 'ran to the end', e.message);
} finally {
  say(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
  await b.close();
}
console.log(failed ? `T-1485 FAILED: ${failed}` : 'T-1485 PASSED');
process.exit(failed ? 1 : 0);
