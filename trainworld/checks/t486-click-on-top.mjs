import { launchChrome, demandFinal, ANGLE } from './lib.mjs';
// T-486: clicks reach only what is drawn on top. In a ?new=1 game with the commuter bubbles on, a
// blueprint track end is drawn on a bubble: a click on the end's square selects the track, and a
// click on the same bubble away from the track still selects the bubble. A blueprint station under
// a bubble: a click selects the station (stations come before bubbles, T-1515), and so does one
// once it is selected. Saves artifacts/trainworld/T-486-station-under-bubble.png.
//   node trainworld/checks/t486-click-on-top.mjs [url]     (from the maps root; default localhost:6135)
// Prints PASS or FAIL lines and exits 1 on failure. About a minute.

const BASE = process.argv[2] ?? 'http://localhost:6135/';

const { ctx, close } = await launchChrome({ prefix: 'tw-t486-', size: [1200, 800], args: ANGLE });
const results = [];
const report = (name, ok, detail) => { results.push(ok); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twViews && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate(() => {
    const tw = window.tw;
    tw.display.value = { ...tw.display.value, commuters: true };
    tw.map.jumpTo({ center: [-73.985, 40.75], zoom: 15 });
  });
  // a bubble: the first screen point (on a grid) whose click the commuter view takes
  const P = await page.waitForFunction(() => {
    const W = window.tw.map.getContainer().clientWidth, H = window.tw.map.getContainer().clientHeight;
    for (let y = H * 0.3; y < H * 0.7; y += 6) for (let x = W * 0.3; x < W * 0.7; x += 6) {
      const s = window.twViews.click(x, y, false);
      if (s && s.kind !== 'track' && window.twViews.click(x + 25, y, false)?.kind === s.kind && window.twViews.click(x - 70, y, false)?.kind === s.kind) return { x, y, kind: s.kind };
    }
    return null;
  }, undefined, { timeout: 120000, polling: 1000 }).then((h) => h.jsonValue());
  // blueprint track from that point 1 km west, so the end's square sits on the bubble
  const ok = await page.evaluate(async (P) => {
    const { lngLatToPack } = await import('/src/game/coords.ts');
    const { edit } = await import('/src/game/world.ts');
    const ll = window.tw.map.unproject([P.x, P.y]);
    const a = lngLatToPack(ll.lng, ll.lat), b = lngLatToPack(ll.lng - 0.012, ll.lat);
    const r = await edit({ op: 'route', from: { kind: 'free', x: a.x, y: a.y, level: 0 }, to: { kind: 'free', x: b.x, y: b.y, level: 0 }, pis: [], single: false });
    return r.ok;
  }, P);
  await page.waitForTimeout(1500);
  const box = await page.evaluate(() => window.tw.map.getContainer().getBoundingClientRect().toJSON());
  const clickAt = async (x, y) => {
    await page.evaluate(() => { window.tw.selection.value = null; });
    await page.mouse.click(box.left + x, box.top + y);
    await page.waitForTimeout(300);
    return page.evaluate(() => window.tw.selection.value);
  };
  const marks = await page.evaluate(() => [...document.querySelectorAll('.pi-mark')].filter((m) => m.style.display !== 'none').length);
  const onEnd = await clickAt(P.x, P.y);
  report('click on a track end over a bubble', ok && marks > 0 && onEnd?.kind === 'track', `(route ${ok}, ${marks} squares drawn, selected ${onEnd?.kind ?? 'nothing'}; the bubble is ${P.kind})`);
  const off = await clickAt(P.x + 25, P.y);
  report('click on the bubble beside it', off?.kind === P.kind, `(selected ${off?.kind ?? 'nothing'})`);
  const tip = await page.evaluate((P) => window.twViews.hover(P.x, P.y), P);
  report('no bubble tag over the square', tip === null, tip ? `(${tip.join(' / ')})` : '');
  // a blueprint station under the bubble: the network canvas is under the bubbles' canvas
  // (map/demandOverlay.ts), but a station is picked before a bubble (T-1515); selected, the station
  // is cut out of the bubbles and drawn on top (the screenshot shows both)
  // a second stretch through the bubble 15 px below the first, its station in the bubble
  const S = await page.evaluate(async (P) => {
    const { edit } = await import('/src/game/world.ts');
    const { lngLatToPack } = await import('/src/game/coords.ts');
    const m = window.tw.map, y = P.y + 15;
    const pk = (x) => { const ll = m.unproject([x, y]); return lngLatToPack(ll.lng, ll.lat); };
    const before = new Set(window.tw.world.value.edges.map((e) => e.id));
    const a = pk(P.x + 200), b = pk(P.x - 200);
    const r1 = await edit({ op: 'route', from: { kind: 'free', ...a, level: 0 }, to: { kind: 'free', ...b, level: 0 }, pis: [], single: false });
    const e = window.tw.world.value.edges.find((x) => !before.has(x.id));
    const r = r1.ok && e ? await edit({ op: 'addStation', at: { kind: 'edge', edge: e.id, ...pk(P.x) }, platform: 200, name: 'T486' }) : r1;
    return { ok: r.ok, issues: r.issues, x: P.x, y };
  }, P);
  await demandFinal(page, { strict: false }); await page.waitForTimeout(3000);
  const sid = await page.evaluate(() => window.tw.world.value.save.stations.at(-1)?.id);
  const bubbleThere = await page.evaluate((S) => window.twViews.click(S.x, S.y, false)?.kind, S);
  await page.screenshot({ path: 'artifacts/trainworld/T-486-station-under-bubble.png', clip: { x: box.left + S.x - 200, y: box.top + S.y - 150, width: 400, height: 300 } });
  const onSt = await clickAt(S.x, S.y);
  report('click on a blueprint station under a bubble', S.ok && !!sid && bubbleThere === P.kind && onSt?.kind === 'station', `(station placed ${S.ok}${S.ok ? '' : ' ' + JSON.stringify(S.issues)}, bubble there ${bubbleThere}, selected ${onSt?.kind ?? 'nothing'})`);
  P.x = S.x; P.y = S.y;
  if (!sid) throw new Error('no station');
  await page.evaluate((sid) => { window.tw.selection.value = { kind: 'station', station: sid }; }, sid);
  await page.waitForTimeout(800);
  await page.mouse.click(box.left + P.x, box.top + P.y);
  await page.waitForTimeout(300);
  const again = await page.evaluate(() => window.tw.selection.value);
  report('click on the selected station over a bubble', again?.kind === 'station' && again.station === sid, `(selected ${again?.kind ?? 'nothing'})`);
} finally {
  await close();
}
process.exit(results.length && results.every(Boolean) ? 0 : 1);
