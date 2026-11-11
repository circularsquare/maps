// T-2899: bus and ferry loop lines, with the mouse, on a new game in New York. A bus loop drawn in the bus
// tool (four points, then a right click on the first closes it and makes the line, T-3109), its tag "Right click to
// close the loop", the panel's lap, constructed and run with buses going round one way (T-2981), riders from a demand solve,
// an out-and-back bus line beside it unchanged, undo and redo, and a save that reloads with the loop. Then
// a ferry loop over three ports of the harbour the same way (a right click, T-3118). Pictures in artifacts/trainworld/busloop/.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t2899-loops.mjs [dev url]
// About 4 minutes.
import { ARTIFACTS, collectErrors, importSave, launchChrome } from './lib.mjs';
import { mkdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const OUT = join(ARTIFACTS, 'busloop');
await mkdir(OUT, { recursive: true });
const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6340/';
const SHADE = process.env.SHADE ?? 'light';
/** ONLY=ferry skips the bus half, ONLY=bus the ferry half */
const ONLY = process.env.ONLY ?? '';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// four corners of a block of Manhattan streets, and three ports of the harbour
const BUS = [[-73.9930, 40.7390], [-73.9770, 40.7390], [-73.9770, 40.7520], [-73.9930, 40.7520]];
const OPEN = [[-73.9930, 40.7300], [-73.9770, 40.7300]];
const WHITEHALL = [-74.0130, 40.7012], ST_GEORGE = [-74.0715, 40.6437], RED_HOOK = [-74.0155, 40.6745];

const chrome = await launchChrome({ prefix: 'tw-t2899-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript((shade) => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade, display: { dayShown: 'weekday' }, v: 2 })); } catch {} }, SHADE);
collectErrors(page, errors);

const box = () => page.$('#map').then((m) => m.boundingBox());
const at = async (p) => { const b = await box(); const [x, y] = await page.evaluate(([a, c]) => { const q = window.tw.map.project([a, c]); return [q.x, q.y]; }, p); return { x: Math.round(b.x + x), y: Math.round(b.y + y) }; };
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };
const fit = async (pts, pad = 120) => {
  await page.evaluate(([pts, pad]) => window.tw.map.fitBounds([[Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1]))], [Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1]))]], { padding: pad, duration: 0 }), [pts, pad]);
  await settle();
};
const tags = () => page.evaluate(() => [...document.querySelectorAll('.tool-tag')].filter((e) => e.style.display !== 'none').map((e) => e.innerText.replace(/\n/g, ' / ')).join(' // '));
const lines = () => page.evaluate(() => window.tw.world.value.save.lines.map((l) => l.num));
const stats = (num) => page.evaluate((num) => {
  const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num);
  if (!l) return null;
  const st = w.lineStats[l.id], b = st.bus;
  return { id: l.id, name: l.name, loop: st.loop, status: st.status, km: st.km, round: st.roundTripByLevel, trains: st.trainsNeeded, fromStart: st.fromStartByLevel[0], turn: st.turnaroundS, dwell: st.dwellS,
    ferry: !!b?.ferry, stops: b?.stops.map((s) => [s.id, s.kind, s.station]) ?? [], ports: b?.ferry?.ports.map((p) => p.node) ?? [], built: b?.built, roadM: b?.roadM };
}, num);
const panel = () => page.evaluate(() => (document.querySelector('#inspector, .inspector') ?? document.body).innerText);
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);

/** vehicles' angle about `c` at times `t0 + k * step`: per vehicle the total turn in turns (+ one way, - the other) */
const turns = (c, n, step) => page.evaluate(([c, n, step]) => {
  const tw = window.tw, r = tw.renderer, a = r.net?.anchor ?? tw.anchor(), t0 = tw.clock.now();
  const ps = Array.from({ length: n }, (_, k) => r.probe(t0 + k * step));
  const out = [];
  for (let i = 0; i < r.tripCount; i++) {
    let prev = null, total = 0, live = true;
    for (const p of ps) {
      if (p[i * 2] >= 1e8) { live = false; break; }
      const [x, y] = tw.toLngLat(p[i * 2], p[i * 2 + 1], a);
      const ang = Math.atan2((y - c[1]) * 111000, (x - c[0]) * 84000);
      if (prev !== null) { let d = ang - prev; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; total += d; }
      prev = ang;
    }
    if (live) out.push(total / (2 * Math.PI));
  }
  return out;
}, [c, n, step]);

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=2&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools?.bus && window.tw.tools?.ferry, undefined, { timeout: 180000 });

  // ---------------------------------------------------------------- the bus loop
  if (ONLY !== 'ferry') {
  await fit(BUS, 160);
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-bus');
  await page.waitForTimeout(300);
  check(await page.evaluate(() => window.tw.tools.bus.active), 'the Build tab\'s Bus button starts the tool');
  for (let k = 0; k < 4; k++) {
    const q = await at(BUS[k]);
    await page.mouse.move(q.x - 5, q.y - 5);
    await page.mouse.move(q.x, q.y, { steps: 3 });
    await page.waitForTimeout(500);
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.bus.pts.length >= n, k + 1, { timeout: 40000 }).catch(() => {});
  }
  const n4 = await page.evaluate(() => window.tw.tools.bus.pts.length);
  check(n4 === 4, 'four points placed', `${n4}`);
  // over the first point: the tag offers to close the loop
  const first = await at(BUS[0]);
  await page.mouse.move(first.x + 12, first.y + 12);
  await page.mouse.move(first.x, first.y, { steps: 3 });
  await page.waitForTimeout(900);
  const tag = await tags();
  check(/Right click to close loop/.test(tag), 'the tag over the first point says Right click to close loop (T-3109)', tag);
  await page.screenshot({ path: join(OUT, `bus-draw-${SHADE}.png`) });
  const cash0 = await page.evaluate(() => window.tw.world.value.save.cash);
  await page.mouse.click(first.x, first.y, { button: 'right' });
  await page.waitForFunction(() => window.tw.world.value.save.lines.length === 1 && !window.tw.tools.bus.active, undefined, { timeout: 60000 }).catch(() => {});
  const nums = await lines();
  check(nums.length === 1, 'the click makes the line', JSON.stringify(nums));
  const num = nums[0];
  let s = await stats(num);
  check(s?.loop === 'circle', 'it is a circle', s?.loop);
  check(s && s.stops.length >= 3 && s.stops[0][0] === s.stops.at(-1)[0], 'its first and last stop are one stop', JSON.stringify(s?.stops));
  check(s && s.round[0] > 0 && Math.abs(s.round[0] - (s.fromStart.at(-1) + s.dwell)) < 1, 'one way (T-2981), its round trip is the lap and the stand', `${s?.round[0]} lap ${s?.fromStart.at(-1)} + ${s?.dwell}`);
  console.log('bus loop', JSON.stringify({ km: s?.km, roadM: s?.roadM, stops: s?.stops.length, trains: s?.trains }));

  // undo takes the loop line away, redo brings it back whole
  await edit({ op: 'undo' });
  await page.waitForTimeout(800);
  check((await lines()).length === 0, 'undo takes the line away', JSON.stringify(await lines()));
  await edit({ op: 'redo' });
  await page.waitForTimeout(800);
  const redone = await stats(num);
  check(redone?.loop === 'circle' && redone.stops.length === s.stops.length, 'redo brings the loop back', `${redone?.loop} ${redone?.stops.length}`);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, redone.id);
  await page.waitForTimeout(600);

  // the new line's name box is open: keep the name
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const btn = page.locator('.callout button:has-text("Construct for")').first();
  await btn.waitFor({ timeout: 20000 });
  await btn.click();
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return l && w.lineStats[l.id].bus?.built; }, num, { timeout: 30000 });
  s = await stats(num);
  check(s.built && s.status === 'running', 'constructed and running', s.status);
  check(s.trains[0] >= 1, 'it has buses', `${s.trains}`);

  // buses go round, one way, on a clock set to 8:30
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 8.5 * 3600); tw.overlay.invalidate(); });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  await page.waitForTimeout(500);
  const centre = [(BUS[0][0] + BUS[1][0]) / 2, (BUS[0][1] + BUS[2][1]) / 2];
  // 3 game minutes apart over a lap of about 20
  let tn = [];
  for (let k = 0; k < 10 && tn.filter((x) => Math.abs(x) > 0.05).length < 2; k++) { await page.waitForTimeout(1500); tn = await turns(centre, 6, 180); }
  const cw = tn.filter((x) => x < -0.05).length, ccw = tn.filter((x) => x > 0.05).length;
  check(tn.length >= 2 && (cw === 0) !== (ccw === 0), 'buses go round the ring, all one way', `${tn.map((x) => x.toFixed(2)).join(' ')}`);
  const tn2 = await turns(centre, 8, 60);
  check(tn2.every((x) => Math.abs(x) < 1.2), 'and none turns back', tn2.map((x) => x.toFixed(2)).join(' '));
  await page.evaluate(() => { window.tw.selection.value = null; });
  await fit(BUS, 200);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, s.id);
  await page.waitForTimeout(1200);

  // riders from a demand solve
  const riders = await page.waitForFunction((id) => (window.twDemand?.view()?.lines.get(String(id))?.ridersPerDay ?? 0) > 0, s.id, { timeout: 240000 }).then(() => true, () => false);
  const rd = await page.evaluate((id) => window.twDemand?.view()?.lines.get(String(id))?.ridersPerDay ?? 0, s.id);
  check(riders && rd > 0, 'the loop carries riders from a demand solve', `${Math.round(rd)} a day`);
  const p = await panel();
  check(/Lap/.test(p) && !/Round trip/.test(p), 'the panel shows the lap', (p.match(/Lap[^\n]*\n?[^\n]*/) ?? [''])[0].replace(/\n/g, ' '));
  check(/Running/.test(p) && /Dwell/.test(p), 'with running time and dwell', '');
  await page.screenshot({ path: join(OUT, `bus-loop-${SHADE}.png`) });
  // zoomed in on the buses
  const tw = await page.evaluate(() => { const t = window.tw, r = t.renderer, a = r.net?.anchor ?? t.anchor(), t0 = t.clock.now(), q = r.probe(t0); for (let i = 0; i < r.tripCount; i++) if (q[i * 2] < 1e8) return t.toLngLat(q[i * 2], q[i * 2 + 1], a); return null; });
  if (tw) {
    await page.evaluate((c) => { window.tw.map.jumpTo({ center: c, zoom: 15.5 }); window.tw.overlay.invalidate(); }, tw);
    await settle(1000);
    await page.screenshot({ path: join(OUT, `bus-loop-z15-${SHADE}.png`) });
  }

  // ---------------------------------------------------------------- an out-and-back bus line beside it
  await fit([...BUS, ...OPEN], 160);
  const open = await page.evaluate(async (pts) => {
    const tw = window.tw;
    const route = pts.flatMap(([a, b]) => { const p = tw.lngLatToPack(a, b); return [p.x, p.y, 1]; });
    return window.tw.client.edit({ op: 'addBusLine', route, name: '', colour: '#c0392b' });
  }, OPEN);
  check(open.ok, 'an out-and-back bus line added beside it', JSON.stringify(open).slice(0, 100));
  await page.waitForTimeout(1500);
  const so = await stats(open.line);
  check(so && so.loop === null && so.stops[0][0] !== so.stops.at(-1)[0], 'it is no loop and its ends are two stops', JSON.stringify(so?.loop));
  const back = so.round[0] - 2 * so.turn - 2 * so.fromStart.at(-1);
  check(Math.abs(back) < 5, 'its round trip is two runs and two turnarounds as before', `${so.round[0]} vs ${2 * so.fromStart.at(-1) + 2 * so.turn}`);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, so.id);
  await page.waitForTimeout(800);
  const po = await panel();
  check(/Round trip/.test(po) && !/\bLap\b/.test(po), 'its panel still says Round trip', '');
  await edit({ op: 'undo' });

  // ---------------------------------------------------------------- a save that reloads
  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const bytes = await readFile(await (await dl).path());
  await importSave(page, bytes);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file', undefined, { timeout: 120000 });
  await page.waitForTimeout(2500);
  const reloaded = await stats(num);
  check(reloaded && reloaded.loop === 'circle' && reloaded.stops[0][0] === reloaded.stops.at(-1)[0], 'a saved game loads with the bus loop', `${reloaded?.loop} ${reloaded?.stops.length} stops`);

  }
  // ---------------------------------------------------------------- the ferry loop
  if (ONLY !== 'bus') {
  await fit([WHITEHALL, ST_GEORGE, RED_HOOK], 140);
  await page.evaluate(() => window.tw.tools.exit?.());
  await page.click('button.tab[aria-label="Build"]');
  await page.click('button.tool-ferry');
  await page.waitForTimeout(300);
  const ferry = () => page.evaluate(() => { const f = window.tw.tools.ferry; return { ports: f.ports.length, closed: f.closed(), route: f.route ? { ok: f.route.ok, m: f.route.m, refused: f.route.legs.filter((l) => l.reason).length } : null }; });
  const place = async (pt, ports) => {
    const q = await at(pt);
    await page.mouse.move(q.x - 4, q.y - 4);
    await page.mouse.move(q.x, q.y, { steps: 4 });
    await page.waitForTimeout(1500);
    await page.mouse.click(q.x, q.y);
    await page.waitForFunction((n) => window.tw.tools.ferry.ports.length >= n, ports, { timeout: 30000 }).catch(() => {});
  };
  await place(WHITEHALL, 1);
  await place(ST_GEORGE, 2);
  await place(RED_HOOK, 3);
  let f = await ferry();
  check(f.ports === 3, 'three ports placed', `${f.ports}`);
  await page.waitForFunction(() => { const f = window.tw.tools.ferry; return f.route && f.routeKey === f.berthKey(); }, undefined, { timeout: 60000 }).catch(() => {});
  const w0 = await at(WHITEHALL);
  await page.mouse.move(w0.x + 15, w0.y + 15);
  await page.mouse.move(w0.x, w0.y, { steps: 3 });
  await page.waitForTimeout(900);
  const ft = await tags();
  check(/Right click to close loop/.test(ft), 'the tag over the first port says Right click to close loop (T-3118)', ft);
  await page.screenshot({ path: join(OUT, `ferry-draw-${SHADE}.png`) });
  const fcash = await page.evaluate(() => window.tw.world.value.save.cash);
  await page.mouse.click(w0.x, w0.y, { button: 'right' });
  await page.waitForFunction(() => window.tw.world.value.save.lines.length >= 2 && !window.tw.tools.ferry.active, undefined, { timeout: 90000 }).catch(() => {});
  const all = await lines();
  const fnum = all.at(-1);
  let fs = await stats(fnum);
  check(fs?.ferry && fs.loop === 'circle', 'the click makes a ferry loop', `${fs?.ferry} ${fs?.loop} ${fs?.name}`);
  check(fs && fs.ports.length === 4 && fs.ports[0] === fs.ports[3], 'its ports are the first again at the end', JSON.stringify(fs?.ports));
  console.log('ferry loop', JSON.stringify({ km: fs?.km, round: fs?.round, stops: fs?.stops.length }));
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const fbtn = page.locator('.callout button:has-text("Construct for")').first();
  await fbtn.waitFor({ timeout: 20000 });
  const quote = await fbtn.textContent();
  const money0 = await page.evaluate(() => window.tw.world.value.save.cash);
  await fbtn.click();
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return l && w.lineStats[l.id].bus?.built; }, fnum, { timeout: 30000 });
  const spent = (money0 - (await page.evaluate(() => window.tw.world.value.save.cash))) / 1e6;
  fs = await stats(fnum);
  const boats = Math.max(...fs.trains);
  check(Math.abs(spent - (30 + 8 * boats)) < 0.1, 'Construct pays three ports once and the boats', `spent ${spent.toFixed(1)} for ${boats} boats; "${quote}"`);
  check(fs.trains[0] >= 1, 'it has boats', `${fs.trains}`);
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(tw.clock.now() + 2 * 3600); tw.overlay.invalidate(); });
  await page.waitForFunction(() => window.tw.renderer.tripCount > 0, undefined, { timeout: 60000 });
  const fc = [(WHITEHALL[0] + ST_GEORGE[0] + RED_HOOK[0]) / 3, (WHITEHALL[1] + ST_GEORGE[1] + RED_HOOK[1]) / 3];
  // the boats reach the renderer a moment after the clock is moved
  let ft2 = [];
  for (let k = 0; k < 10 && ft2.filter((x) => Math.abs(x) > 0.02).length < 2; k++) { await page.waitForTimeout(1500); ft2 = await turns(fc, 6, 300); }
  console.log('boat turns', ft2.map((x) => x.toFixed(2)).join(' '), 'trips', await page.evaluate(() => window.tw.renderer.tripCount));
  const live = ft2.filter((x) => Math.abs(x) > 0.02);
  check(live.length >= 1 && (live.some((x) => x > 0) !== live.some((x) => x < 0)), 'boats go round the harbour, all one way', ft2.map((x) => x.toFixed(2)).join(' '));
  await fit([WHITEHALL, ST_GEORGE, RED_HOOK], 200);
  await page.evaluate((id) => { window.tw.selection.value = { kind: 'line', line: id }; }, fs.id);
  await page.waitForTimeout(1500);
  const fr = await page.waitForFunction((id) => (window.twDemand?.view()?.lines.get(String(id))?.ridersPerDay ?? 0) > 0, fs.id, { timeout: 240000 }).then(() => true, () => false);
  const frd = await page.evaluate((id) => window.twDemand?.view()?.lines.get(String(id))?.ridersPerDay ?? 0, fs.id);
  console.log(`ferry riders a day: ${Math.round(frd)}`);
  check(fr, 'the ferry loop carries riders (the harbour has stations near its ports)', `${Math.round(frd)} a day`);
  const fp = await panel();
  check(/Lap/.test(fp) && !/Round trip/.test(fp), 'its panel shows the lap', '');
  await page.screenshot({ path: join(OUT, `ferry-loop-${SHADE}.png`) });

  // a port taken out from the panel's list: taking one out of three opens the loop, undo closes it again
  const hits = page.locator('.ferry-ports li .stop-hit');
  check((await hits.count()) === 4, 'the panel lists four stops, each with its cross', `${await hits.count()}`);
  await hits.nth(1).click();
  await page.waitForFunction((n) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === n); return l && w.lineStats[l.id].bus.ferry.ports.length !== 4; }, fnum, { timeout: 60000 }).catch(() => {});
  const cut = await stats(fnum);
  check(cut && cut.loop === null && cut.ports.length === 2, 'taking one out of three opens it into a line of two ports', `${cut?.loop} ${JSON.stringify(cut?.ports)}`);
  await edit({ op: 'undo' });
  await page.waitForTimeout(1000);
  const uncut = await stats(fnum);
  check(uncut?.loop === 'circle' && uncut.ports.length === 4, 'and undo closes it again', `${uncut?.loop} ${uncut?.ports.length}`);

  }
  check(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(OUT, 'died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-2899 FAILED: ${fail}` : 'T-2899 PASSED');
process.exit(fail ? 1 : 0);
