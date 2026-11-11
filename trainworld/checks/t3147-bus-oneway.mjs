// T-3147: a bus loop's stop rings in the line's own view are half-filled where the sim serves the stop one way
// (`LineStats.oneWayAt`), and the bus stop hover has the chevron beside the line's bullet (no "One way" row).
// A bus ring and a route that loops back on a new New York game. Screenshots artifacts/trainworld/t3147-*.png.
// From the maps root, on a Vite of your own:   node trainworld/checks/t3147-bus-oneway.mjs [dev url]
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6202/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.9535, 40.6505];
const chrome = await launchChrome({ prefix: 'tw-t3147-', size: [1400, 900], deviceScaleFactor: Number(process.env.DSF ?? 2) });
const errors = [];
const page = await chrome.ctx.newPage();
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = async (ms = 1500) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
try {
  await page.goto(`${BASE}?debug=1&new=1&opened=all&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus, undefined, { timeout: 300000 });
  await demandFinal(page, { timeout: 600000 });
  const made = await page.evaluate(async (C) => {
    const tw = window.tw, c = tw.lngLatToPack(C[0], C[1]);
    const sq = (x, y, s) => [x, y, 1, x + s, y, 1, x + s, y + s, 1, x, y + s, 1, x, y, 1];
    const ring = await tw.client.edit({ op: 'addBusLine', route: sq(c.x - 1500, c.y - 1500, 3000), name: 'Ring', colour: '#c0392b' });
    const [x, y] = [c.x + 3000, c.y - 1500];
    const back = await tw.client.edit({ op: 'addBusLine', route: [x, y, 1, x + 2000, y, 1, x + 3000, y, 1, x + 3000, y + 1000, 1, x + 2000, y + 1000, 1, x + 2000, y, 1, x, y, 1], name: 'Back', colour: '#2d7dd2' });
    return { ring: ring.line, back: back.line, c: [c.x, c.y], b: [x, y] };
  }, C);
  await edit({ op: 'busHand', line: made.ring });
  for (const [dx, dy] of [[1500, -500], [-500, 1500], [-1500, 300]]) await edit({ op: 'busHandStop', line: made.ring, x: made.c[0] + dx, y: made.c[1] + dy, on: true });
  await edit({ op: 'busHand', line: made.back });
  await edit({ op: 'busHandStop', line: made.back, x: made.b[0] + 3000, y: made.b[1] + 400, on: true });
  await edit({ op: 'busHandStop', line: made.back, x: made.b[0] + 1000, y: made.b[1] + 10, on: true });
  await page.waitForTimeout(1000);
  for (const [name, num] of [['ring', made.ring], ['back', made.back]]) {
    const at = await page.evaluate((num) => {
      const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), r = w.lineStats[l.id].bus.routeLL;
      let a = [Infinity, Infinity, -Infinity, -Infinity];
      for (let i = 0; i < r.length; i += 2) a = [Math.min(a[0], r[i]), Math.min(a[1], r[i + 1]), Math.max(a[2], r[i]), Math.max(a[3], r[i + 1])];
      window.tw.selection.value = { kind: 'line', line: l.id };
      window.tw.map.jumpTo({ center: [(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], zoom: 13.6 });
      window.tw.overlay.invalidate();
      return true;
    }, num);
    await settle(2500);
    const res = await page.evaluate(async (num) => {
      const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      const { busStopTip } = await import('/src/map/busLayer.ts');
      const b = tw.renderer.buses;
      const stops = st.bus.stops.map((s, i) => {
        const end = i === 0 || i === st.bus.stops.length - 1;
        const pt = tw.map.project([s.lng, s.lat]), sx = pt.x, sy = pt.y;
        const hit = b.stopAt(tw.overlay, sx, sy);
        const tip = hit ? busStopTip(w, hit.line, hit.end, hit.id, tw.overlay) : [];
        return { id: s.id, end, way: st.oneWayAt[s.id] !== undefined, hitId: hit?.id, hitId2: hit?.id, tipLen: tip.length, chev: tip.some((t) => t instanceof HTMLElement && t.querySelector('.bullet-heading .heading-chevron')), text: tip.some((t) => (typeof t === 'string' ? t : t.textContent).includes('One way')) };
      });
      const built = b.built; const dbg = { last: b.last, sc: built.stopCount, p: tw.overlay.toScreen(st.bus.stops[1].x, st.bus.stops[1].y), loc: [built.stops[0], built.stops[1]], pl: [st.bus.stops[1].x, st.bus.stops[1].y] }; 
      let drawn = 0;
      for (let i = 0; i < built.stopCount; i++) if (built.stopLine[i] === num && (built.stopWay[2 * i] || built.stopWay[2 * i + 1])) drawn++;
      return { dbg, stops, drawn, oneWay: Object.keys(st.oneWayAt).length };
    }, num);
    const mid = res.stops.filter((s) => !s.end);
    check(res.oneWay > 0 && mid.length > 0, `${name}: the sim serves some stops one way`, `${res.oneWay} ids`);
    check(mid.every((s) => s.chev === s.way && !s.text), `${name}: the hover has the chevron exactly at one-way stops, no text row`, mid.map((s) => (s.chev ? 'C' : '-')).join(''));
    check(res.drawn === mid.filter((s) => s.way).length, `${name}: the rings carry a half exactly at those stops`, `${res.drawn} drawn`);
    if (name === 'back') check(mid.some((s) => !s.way) && mid.some((s) => s.way), 'the loop-back route has stops both ways and one way');
    await page.screenshot({ path: join(ARTIFACTS, `t3147-bus-${name}.png`) });
    // close on a one-way stop
    await page.evaluate((num) => {
      const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      const s = st.bus.stops.slice(1, -1).find((x) => st.oneWayAt[x.id] !== undefined);
      window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 17.2 });
      window.tw.overlay.invalidate();
    }, num);
    await settle(2000);
    await page.screenshot({ path: join(ARTIFACTS, `t3147-bus-${name}-close.png`), clip: await page.evaluate(() => { const p = window.tw.map.project(window.tw.map.getCenter()), r = window.tw.map.getCanvas().getBoundingClientRect(); return { x: Math.round(p.x + r.left) - 40, y: Math.round(p.y + r.top) - 30, width: 80, height: 60 }; }) });
    // the hover tag of that stop, drawn in a box
    await page.evaluate(async (num) => {
      const tw = window.tw, w = tw.world.value, l = w.save.lines.find((x) => x.num === num), st = w.lineStats[l.id];
      const s = st.bus.stops.slice(1, -1).find((x) => st.oneWayAt[x.id] !== undefined);
      const { busStopTip } = await import('/src/map/busLayer.ts');
      const box = document.createElement('div');
      box.id = 'check-tip'; box.className = 'tool-tag'; box.style.cssText = 'position:fixed;left:700px;top:100px;z-index:99999';
      for (const t of busStopTip(w, num, false, s.id, tw.overlay)) { const d = document.createElement('div'); if (typeof t === 'string') d.textContent = t; else d.append(t); box.append(d); }
      document.body.append(box);
    }, num);
    await page.screenshot({ path: join(ARTIFACTS, `t3147-bus-${name}-hover.png`), clip: { x: 650, y: 70, width: 420, height: 160 } });
    await page.evaluate(() => document.getElementById('check-tip')?.remove());
  }
  // T-3131: a route drawn out and back by hand along one road, a hand stop on its way out lands on both passes
  const oab = await page.evaluate(async (C) => {
    const tw = window.tw, c = tw.lngLatToPack(C[0], C[1]);
    const [x, y] = [c.x - 3000, c.y + 3000];
    const r = await tw.client.edit({ op: 'addBusLine', route: [x, y, 1, x + 3000, y, 1, x + 3000, y + 15, 1, x, y + 15, 1], name: 'Out', colour: '#2a9d4b' });
    return { line: r.line, x, y };
  }, C);
  await edit({ op: 'busHand', line: oab.line });
  await edit({ op: 'busHandStop', line: oab.line, x: oab.x + 1000, y: oab.y + 5, on: true });
  await edit({ op: 'busHandStop', line: oab.line, x: oab.x + 2000, y: oab.y + 5, on: true });
  await page.waitForTimeout(1000);
  const n = await page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); return w.lineStats[l.id].bus.stops.map((s) => Math.round(s.x)); }, oab.line);
  check(n.length === 6, 'out and back by hand: each of two hand stops is on both passes (6 stops with the ends)', n.join(','));
  await page.evaluate((num) => {
    const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num), s = w.lineStats[l.id].bus.stops[1];
    window.tw.selection.value = { kind: 'line', line: l.id };
    window.tw.map.jumpTo({ center: [s.lng, s.lat], zoom: 15.5 });
    window.tw.overlay.invalidate();
  }, oab.line);
  await settle(2500);
  await page.screenshot({ path: join(ARTIFACTS, 't3131-out-and-back.png') });
} catch (e) { console.log('FAIL', e.message); fail++; }
check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `T-3147 FAILED (${fail})` : 'T-3147 PASSED');
process.exit(fail ? 1 : 0);
