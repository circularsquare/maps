// T-3348 stage 1: per-direction stops, Anita's case (reports/t3348-split-legs.png). A double track splits
// into two one-way single legs 150 m apart, a station on each, and rejoins as double; a line West - East run
// both ways, the stop clicked on the leg its trains run the other way along. The line keeps one stop there,
// on the leg it runs out along, and its return run stops at the other leg's station: its trains board and
// alight each way on their own leg (boarding reads each run's stop node), and demand's riders board the
// way back at the north leg's station (the snapshot's backStops, the kernel's reverse run). Picture in
// artifacts/trainworld/t3348-perdir.png. From the maps root, on a Vite of your own
// (node trainworld/checks/dev-nowatch.mjs 6170):
//   node trainworld/checks/t3348-perdir.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, demandFinal, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6170/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3348-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const edit = (op) => page.evaluate(async (op) => { const r = await window.tw.client.edit(op); await new Promise((d) => setTimeout(d, 500)); return r; }, op);

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const made = await page.evaluate(async ([c]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + dy });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const node = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const res = [];
    const free = (p) => ({ kind: 'free', x: p.x, y: p.y, level: 0 });
    res.push(await cl.edit({ op: 'route', from: free(P(-3000, 0)), to: free(P(-1000, 0)), pis: [], single: false, quad: false }));
    await pause(500);
    res.push(await cl.edit({ op: 'route', from: free(P(1000, 0)), to: free(P(3000, 0)), pis: [], single: false, quad: false }));
    await pause(500);
    const [j0, j1] = [node(P(-1000, 0)), node(P(1000, 0))];
    const leg = {};
    for (const dy of [75, -75]) {
      const had = new Set(tw.world.value.edges.map((e) => e.id));
      res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: j0 }, to: { kind: 'node', node: j1 }, pis: [{ ...P(0, dy), level: -1 }], single: true }));
      await pause(500);
      leg[dy] = tw.world.value.edges.find((e) => !had.has(e.id) && e.tracks === 1);
    }
    // the player sets each leg's way: the south one eastbound, the north one westbound (trains keep right)
    const w = tw.world.value;
    for (const dy of [75, -75]) {
      const e = leg[dy], a = w.nodes.get(e.a), b = w.nodes.get(e.b), east = dy < 0, ab = a.x < b.x;
      res.push(await cl.edit({ op: 'oneWay', edge: e.id, dir: ab === east ? 1 : 2 }));
      await pause(200);
    }
    const ids = {};
    for (const [name, dx, dy, end] of [['West', -3000, 0, true], ['East', 3000, 0, true], ['Noordeinde', 0, 75, false], ['Noordeinde ', 0, -75, false]]) {
      const p = P(dx, dy);
      const at = end ? { kind: 'node', node: node(p) } : { kind: 'edge', edge: leg[dy].id, x: p.x, y: p.y };
      const r = await cl.edit({ op: 'addStation', at, platform: 120, name: name.trim() });
      await pause(300);
      res.push(r);
      ids[dy > 0 ? 'north' : dy < 0 ? 'south' : name] = r.ok ? node(p) : null;
    }
    return { ids, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 100)), legs: tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.oneWay ?? 0) };
  }, [RC]);
  check(!made.refused.length && Object.values(made.ids).every((x) => x !== null), 'double, two one-way legs with a station each, rejoined', JSON.stringify(made));
  const I = made.ids;
  const line = await edit({ op: 'addLine', stops: [I.West, I.East], name: 'L', colour: '#d6336c' });
  check(line?.ok, 'a line West - East, both ways', JSON.stringify(line));
  // the click lands on the north leg, which runs westbound: the way out is eastbound
  const ins = await edit({ op: 'insertStop', line: line.line, station: I.north });
  await wait(800);
  const st = await page.evaluate((num) => { const w = window.tw.world.value, l = w.save.lines.find((x) => x.num === num); const s = w.lineStats[l.id]; return { stops: l.stops.map(Number), backStops: s.backStops?.map(Number) ?? null, back: s.backStops ? 1 : 0, stopS: s.stopS.map((r) => r.length), broken: l.broken ?? false }; }, line.line);
  check(ins?.ok && st.stops.length === 3 && st.stops[1] === I.south, 'one stop at Noordeinde, kept on the leg the way out runs along', JSON.stringify(st));
  check(JSON.stringify(st.backStops) === JSON.stringify([I.West, I.north, I.East]), 'the way back stops at the north leg station', JSON.stringify(st.backStops));
  check(st.back > 0 && st.stopS[0] === 3 && st.stopS[1] === 3 && !st.broken, 'both runs run, the way back round the other leg', JSON.stringify(st));
  await edit({ op: 'setTrainCount', line: line.line, trains: [6, 4, 2] });
  await edit({ op: 'constructAll' });
  await page.evaluate(() => { const tw = window.tw; tw.clock.setRunning(false); tw.clock.set(Math.floor(tw.clock.now() / 86400) * 86400 + 9.2 * 3600); });
  // demand: the way back's riders board and alight at the north leg's station, the way out's at the south's
  await demandFinal(page, { timeout: 600000, polling: 2000, lines: 0 });
  const riders = await page.evaluate(([ids, num]) => {
    const w = window.tw.world.value, v = window.twDemand.view(), l = w.save.lines.find((x) => x.num === num);
    const at = (id) => { const d = v.stations.get(String(id)) ?? v.stations.get(id); return d ? Math.round(d.boardings ?? 0) : 0; };
    return { north: at(ids.north), south: at(ids.south), West: at(ids.West), East: at(ids.East), line: Math.round(v.lines.get(l.id)?.ridersPerDay ?? 0), backStops: (window.tw.world.value.lineStats[l.id].backStops ?? []).map(Number) };
  }, [I, line.line]);
  check(riders.north > 0 && riders.south > 0 && riders.line > 0, 'riders board at both Noordeinde legs, the north one on the way back', JSON.stringify(riders));
  await page.evaluate(async (l) => { (await import('/src/game/ui.ts')).selection.value = { kind: 'line', line: String(l) }; }, line.line);
  await page.evaluate(([c]) => { const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]), q = tw.packToLngLat(o.x, o.y), [lng, lat] = Array.isArray(q) ? q : [q.lng, q.lat]; tw.map.jumpTo({ center: [lng, lat], zoom: 15.4 }); }, [RC]);
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(2500);
  await page.screenshot({ path: join(ARTIFACTS, 't3348-perdir.png') });
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await page.screenshot({ path: join(ARTIFACTS, 't3348-died.png') }).catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3348 FAILED: ${fail}` : 'T-3348 PASSED');
process.exit(fail ? 1 : 0);
