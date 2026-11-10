// T-2981: one-way loops. On a new game in New York, a train ring of four stations in Brooklyn (built with the edit
// API, curves at the corners, so trains carry straight on through every station) and a bus loop round a block of
// Manhattan. A new train loop runs both ways and a new bus loop one way; each is switched to the other in the line
// panel's Direction row (a click on the segmented control). Checks the round trip (one lap one way, two both ways),
// the fleet, which way the vehicles go round (all one way, or some each way), the riders from a demand solve before
// and after (printed, with the share boarding run 1 and the stops a rider rides on average), and a save that
// reloads with each loop's direction. Screenshots of the panel light and dark in artifacts/trainworld/t2981-*.png.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t2981-oneway.mjs [dev url]
// About 6 minutes (four demand solves).
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6420/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

// the ring: a square of 2.4 km round Crown Heights, a station at the middle of each side
const RC = [-73.945, 40.672], H = 1200;
// the bus loop: four corners of a block of Manhattan streets, drawn anticlockwise on the map (east, north, west)
const BUS = [[-73.9930, 40.7390], [-73.9770, 40.7390], [-73.9770, 40.7520], [-73.9930, 40.7520]];
const BC = [(BUS[0][0] + BUS[1][0]) / 2, (BUS[0][1] + BUS[2][1]) / 2];

const chrome = await launchChrome({ prefix: 'tw-t2981-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const stats = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num);
  if (!l) return null;
  const st = w.lineStats[l.id];
  return { id: l.id, loop: st.loop, oneWay: st.oneWay, status: st.status, round: st.roundTripByLevel, trains: st.trainsNeeded, lap: st.fromStartByLevel[1].at(-1), dwell: st.dwellS };
}, num);
/** per vehicle within `r` m of `c`, its total turn about `c` in turns over `n` probes `step` game s apart (+ anticlockwise) */
const turns = (c, r, n, step) => page.evaluate(([c, rad, n, step]) => {
  const tw = window.tw, rd = tw.renderer, a = rd.net?.anchor ?? tw.anchor(), t0 = tw.clock.now();
  const ps = Array.from({ length: n }, (_, k) => rd.probe(t0 + k * step));
  const out = [];
  for (let i = 0; i < rd.tripCount; i++) {
    let prev = null, total = 0, live = true, near = false;
    for (const p of ps) {
      if (p[i * 2] >= 1e8) { live = false; break; }
      const [x, y] = tw.toLngLat(p[i * 2], p[i * 2 + 1], a);
      const dx = (x - c[0]) * 111320 * Math.cos((c[1] * Math.PI) / 180), dy = (y - c[1]) * 111320;
      if (Math.hypot(dx, dy) < rad) near = true;
      const ang = Math.atan2(dy, dx);
      if (prev !== null) { let d = ang - prev; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; total += d; }
      prev = ang;
    }
    if (live && near) out.push(total / (2 * Math.PI));
  }
  return out;
}, [c, r, n, step]);
const ways = async (c, r, step) => {
  // the renderer holds the trips of what is in view
  await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 13 }); window.tw.overlay.invalidate(); }, c);
  await settle(1500);
  let t = [];
  for (let k = 0; k < 12 && t.filter((x) => Math.abs(x) > 0.05).length < 3; k++) { await page.waitForTimeout(1500); t = await turns(c, r, 6, step); }
  const live = t.filter((x) => Math.abs(x) > 0.05);
  return { acw: live.filter((x) => x > 0).length, cw: live.filter((x) => x < 0).length, text: t.map((x) => x.toFixed(2)).join(' ') };
};
/** the weekday's final newer than `after`, steady for 8 s */
const final = async (after) => {
  for (;;) {
    await page.waitForFunction((after) => {
      const d = window.twDemand, s = d?.log?.findLast((x) => !d.client.fversions?.has(x.version)), v = d?.view?.();
      return !!v && s && v.version === s.version && v.version > after && v.final && !d.client.waiting;
    }, after, { timeout: 400000, polling: 500 });
    const v = await page.evaluate(() => window.twDemand.view().version);
    await page.waitForTimeout(8000);
    if (await page.evaluate((v) => window.twDemand.view().version === v && window.twDemand.view().final, v)) return v;
  }
};
/** a line's riders a day, the share of boardings on run 1 and the mean stops ridden a boarding */
const riders = (id) => page.evaluate((id) => {
  const l = window.twDemand.view().lines.get(String(id));
  if (!l) return null;
  let b0 = 0, b1 = 0, seg = 0;
  const n = l.boardings.length;
  for (const b of l.board ?? []) for (let k = 0; k < 2 * n; k++) if (k < n) b0 += b[k]; else b1 += b[k];
  for (const q of l.loads) for (const x of q) seg += x;
  return { perDay: Math.round(l.ridersPerDay), run1: b0 + b1 > 0 ? b1 / (b0 + b1) : 0, stops: b0 + b1 > 0 ? seg / (b0 + b1) : 0 };
}, id);
const fmt = (r) => r ? `${r.perDay.toLocaleString('en-US')} a day, ${(100 * r.run1).toFixed(0)}% on run 1, ${r.stops.toFixed(2)} stops a ride` : 'none';
const select = async (id) => { await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, id); await page.waitForTimeout(1200); };
/** a click on the Direction row's choice, as the player clicks it */
const choose = async (label) => {
  const row = page.locator('li', { has: page.locator('span.grow', { hasText: /^Direction$/ }) }).first();
  await row.waitFor({ timeout: 20000 });
  await row.locator('button', { hasText: new RegExp(`^${label}$`) }).first().click();
};
const panelShot = async (file) => {
  const p = await page.$('#inspector, .inspector');
  await page.waitForTimeout(600);
  await (p ?? page).screenshot({ path: join(ARTIFACTS, file) });
};
const out = { when: new Date().toISOString() };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client && window.tw.tools?.bus, undefined, { timeout: 180000 });

  // ---------------------------------------------------------------- the train ring and the bus loop
  const ring = await page.evaluate(async ([c, h]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const mids = [P(h, 0), P(0, h), P(-h, 0), P(0, -h)], corners = [P(h, h), P(-h, h), P(-h, -h), P(h, -h)];
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const nodes = [];
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const lv = -1, pi = (p) => [{ x: p.x, y: p.y, level: lv }];
    const res = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', x: mids[0].x, y: mids[0].y, level: lv }, to: { kind: 'free', x: mids[1].x, y: mids[1].y, level: lv }, pis: pi(corners[0]), single: false }));
    await pause(300);
    nodes.push(near(mids[0]), near(mids[1]));
    for (const k of [2, 3]) {
      res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[k - 1] }, to: { kind: 'free', x: mids[k].x, y: mids[k].y, level: lv }, pis: pi(corners[k - 1]), single: false }));
      await pause(300);
      nodes.push(near(mids[k]));
    }
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[3] }, to: { kind: 'node', node: nodes[0] }, pis: pi(corners[3]), single: false }));
    await pause(500);
    for (const n of nodes) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' })); await pause(200); }
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(500);
    const line = await cl.edit({ op: 'addLine', stops: [...nodes, nodes[0]], name: 'Ring', colour: '#0a7d3b' });
    await pause(500);
    return { line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 80)) };
  }, [RC, H]);
  check(ring.ok && !ring.refused.length, 'the train ring is built with the edit API', ring.refused.join(' | '));
  await edit({ op: 'setTrainCount', line: ring.line, trains: [8, 6, 4] });
  await page.waitForTimeout(800);
  let t = await stats(ring.line);
  check(t?.loop === 'circle' && t.oneWay === false, 'a new train loop is a circle run both ways', `${t?.loop} oneWay ${t?.oneWay}`);
  check(t && Math.abs(t.round[1] - 2 * (t.lap + t.dwell)) < 2, 'both ways its round trip is two laps', `${t?.round[1]} lap ${t?.lap} + ${t?.dwell}`);

  const bus = await page.evaluate(async (pts) => {
    const tw = window.tw;
    const route = [...pts, pts[0]].flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
    const r = await tw.client.edit({ op: 'addBusLine', route, name: '', colour: '#c0392b' });
    const c = r.ok ? await tw.client.edit({ op: 'constructLine', line: r.line }) : null;
    return { line: r.line, ok: r.ok && c?.ok };
  }, BUS);
  check(bus.ok, 'the bus loop is added and constructed');
  await page.waitForTimeout(1500);
  let b = await stats(bus.line);
  check(b?.loop === 'circle' && b.oneWay === true, 'a new bus loop runs one way', `${b?.loop} oneWay ${b?.oneWay}`);
  check(b && Math.abs(b.round[1] - (b.lap + b.dwell)) < 2, 'one way its round trip is one lap', `${b?.round[1]} lap ${b?.lap} + ${b?.dwell}`);
  console.log('ring', JSON.stringify(t), 'bus', JSON.stringify(b));

  // ---------------------------------------------------------------- vehicles and riders as they start
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); tw.overlay.invalidate(); });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  let wr = await ways(RC, 2500, 50), wb = await ways(BC, 1500, 120);
  check(wr.acw >= 1 && wr.cw >= 1, 'both ways: trains go round the ring each way', wr.text);
  check(wb.acw >= 2 && wb.cw === 0, 'one way: every bus goes round anticlockwise, the way it was drawn', wb.text);
  let v = await final(0);
  out.before = { ring: await riders(t.id), bus: await riders(b.id) };
  console.log(`riders: ring both ways ${fmt(out.before.ring)}; bus one way ${fmt(out.before.bus)}`);
  check(out.before.bus && out.before.bus.run1 === 0, 'nobody boards run 1 of a one-way loop', fmt(out.before.bus));

  // ---------------------------------------------------------------- switched in the panel
  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 13.4 }), RC);
  await settle();
  await select(t.id);
  await panelShot('t2981-ring-both-light.png');
  await choose('One way');
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return w.lineStats[l.id].oneWay; }, ring.line, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(1000);
  t = await stats(ring.line);
  check(t.oneWay && Math.abs(t.round[1] - (t.lap + t.dwell)) < 2, 'One way in the panel: the ring\'s round trip is one lap', `${t.round[1]} lap ${t.lap}`);
  check(t.trains[1] === 6, 'the ring keeps its trains, all going one way', `${t.trains}`);
  await panelShot('t2981-ring-one-light.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await page.waitForTimeout(800);
  await panelShot('t2981-ring-one-dark.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
  const p = await page.evaluate(() => (document.querySelector('#inspector, .inspector') ?? document.body).innerText);
  check(/Direction/.test(p) && /One way/.test(p) && /Both ways/.test(p) && /Lap/.test(p), 'the panel has the Direction row and the lap', '');
  // the train count steps by 1 now
  await edit({ op: 'setTrainCount', line: ring.line, trains: [7, 5, 3] });
  await page.waitForTimeout(1000);
  t = await stats(ring.line);
  check(t.trains.join() === '7,5,3', 'an odd fleet stays odd on a one-way loop', `${t.trains}`);
  await edit({ op: 'setTrainCount', line: ring.line, trains: [8, 6, 4] });

  await page.evaluate((c) => window.tw.map.jumpTo({ center: c, zoom: 14.2 }), BC);
  await settle();
  await select(b.id);
  await panelShot('t2981-bus-one-light.png');
  await choose('Both ways');
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return !w.lineStats[l.id].oneWay; }, bus.line, { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(1000);
  b = await stats(bus.line);
  check(!b.oneWay && Math.abs(b.round[1] - 2 * (b.lap + b.dwell)) < 2 && b.trains.every((x) => x % 2 === 0), 'Both ways in the panel: the bus loop\'s round trip is two laps, an even fleet', `${b.round[1]} lap ${b.lap} buses ${b.trains}`);
  await panelShot('t2981-bus-both-light.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'dark'; });
  await page.waitForTimeout(800);
  await panelShot('t2981-bus-both-dark.png');
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).shade.value = 'light'; });
  await page.evaluate(() => { window.tw.selection.value = null; });

  await page.evaluate(() => { const tw = window.tw; tw.clock.set(tw.clock.now() + 1800); tw.overlay.invalidate(); });
  await page.waitForTimeout(1500);
  wr = await ways(RC, 2500, 50); wb = await ways(BC, 1500, 120);
  check(wr.acw >= 2 && wr.cw === 0, 'one way: every train goes round anticlockwise, the way the stops were drawn', wr.text);
  check(wb.acw >= 1 && wb.cw >= 1, 'both ways: buses go round each way', wb.text);
  v = await final(v);
  out.after = { ring: await riders(t.id), bus: await riders(b.id) };
  console.log(`riders: ring one way ${fmt(out.after.ring)}; bus both ways ${fmt(out.after.bus)}`);
  check(out.after.ring && out.after.ring.run1 === 0, 'nobody boards run 1 of the one-way ring', fmt(out.after.ring));
  check(out.after.ring && out.before.ring && out.after.ring.stops > out.before.ring.stops, 'one way, a ride on the ring is longer (the long way round)', `${out.before.ring?.stops.toFixed(2)} -> ${out.after.ring?.stops.toFixed(2)} stops`);
  check(out.after.bus && out.before.bus && out.after.bus.stops < out.before.bus.stops, 'both ways, a ride on the bus loop is shorter', `${out.before.bus?.stops.toFixed(2)} -> ${out.after.bus?.stops.toFixed(2)} stops`);

  // ---------------------------------------------------------------- a save that reloads
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const t2 = await stats(ring.line), b2 = await stats(bus.line);
  check(t2?.oneWay === true && b2?.oneWay === false, 'a saved game loads the ring one way and the bus loop both ways', `${t2?.oneWay} ${b2?.oneWay}`);
  check(t2 && Math.abs(t2.round[1] - t.round[1]) < 1 && b2 && Math.abs(b2.round[1] - b.round[1]) < 1, 'with the same round trips', `${t2?.round[1]} ${b2?.round[1]}`);
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't2981-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
await writeFile(join(ARTIFACTS, 't2981-oneway.json'), JSON.stringify(out, null, 1)).catch(() => {});
console.log(fail ? `T-2981 FAILED: ${fail}` : 'T-2981 PASSED');
process.exit(fail ? 1 : 0);
