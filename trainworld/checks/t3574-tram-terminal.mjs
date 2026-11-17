// T-3574, T-3575: tram stops take the terminal and passing tracks as rail stations do, and tram track takes crossovers.
// Played through lib/play.mjs in a new game in New York:
//  1. a tram double track with two stops, constructed; the stop's panel row has Terminal and Passing tracks (and no
//     Park and ride); a rail station beside has all four; the terminal's hover says it costs $37.5M (1 km of the stop's
//     own kind, the same as rail at ground off street), pressing it builds it and the station's upgrade bit is set;
//  2. crossing track drawn between two tram doubles is tram track and joins them;
//  3. the crossover edit on a tram quad makes two tram crossover edges, priced as the same on rail track.
// Shot: artifacts/trainworld/review/t3574_tramstop.png, the tram stop's panel.
// From the tree root, on a Vite of your own:   node trainworld/checks/t3574-tram-terminal.mjs [dev url]
import { openGame } from './lib/play.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6157/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.66];
const P = (dx, dy) => [C[0] + dx / (111.32 * Math.cos((C[1] * Math.PI) / 180)), C[1] + dy / 111.32];
const upgrades = (g) => g.page.evaluate(() => [...document.querySelectorAll('#inspector .upgrades button.upgrade')].map((b) => ({ name: b.getAttribute('aria-label'), tip: b.dataset.tip ?? '', on: b.getAttribute('aria-pressed') })));

const g = await openGame({ url: BASE, city: 'nyc', mode: 'sandbox', ui: { shade: 'light', display: { dayShown: 'weekday' }, v: 2 }, prefix: 'tw-t3574-' });
try {
  await g.act.camera(C, 14);
  await g.act.tool('track', { type: 'tram', tracks: 2, level: 0, snap: { real: false, streets: false }, keepPlacing: false });
  const tram = await g.act.drawTrack([P(-1, 0), P(1, 0)]);
  check(tram.ok, 'tram double track drawn');
  const stops = [P(-0.5, 0), P(0.5, 0)];
  for (const p of stops) check((await g.act.placeStation(p)).ok, 'a tram stop placed');
  await g.act.tool('track', { type: 'regular', tracks: 2, level: 0 });
  const rail = await g.act.drawTrack([P(-1, -0.6), P(1, -0.6)]);
  check(rail.ok, 'rail double track drawn');
  check((await g.act.placeStation(P(0, -0.6))).ok, 'a rail station placed');
  await g.act.construct();

  await g.act.selectAt(stops[1]);
  const p = await g.see.panel();
  const ups = await upgrades(g);
  const names = ups.map((u) => u.name);
  console.log('tram stop panel:', p?.title, '|', names.join(', '));
  check(names.includes('Terminal') && names.includes('Passing tracks'), 'a tram stop offers Terminal and Passing tracks', names.join(', '));
  check(!names.includes('Park and ride') && !names.includes('Parking'), 'a tram stop offers no park and ride', names.join(', '));
  const term = ups.find((u) => u.name === 'Terminal');
  console.log('terminal tip:', JSON.stringify(term?.tip));
  check(/\$37\.5M/.test(term?.tip ?? ''), 'the terminal costs $37.5M at a tram stop at ground level (1 km of tram track)', term?.tip);
  await g.shot('C:/Users/anita/projects/maps/artifacts/trainworld/review/t3574_tramstop.png', { element: '#inspector' });
  await g.driver.control({ name: 'Terminal', within: '#inspector' });
  await g.page.waitForTimeout(800);
  const after = await upgrades(g);
  check(after.find((u) => u.name === 'Terminal')?.on === 'true', 'pressing it builds the terminal at the tram stop');
  const bit = await g.page.evaluate(() => window.tw.world.value.save.stations.filter((s) => s.tram).map((s) => s.upgrades));
  check(bit.some((u) => (u & 1) === 1), 'the stop carries the terminal upgrade', JSON.stringify(bit));

  await g.act.selectAt(P(0, -0.6));
  const railNames = (await upgrades(g)).map((u) => u.name);
  console.log('rail station panel:', railNames.join(', '));
  check(['Terminal', 'Passing tracks', 'Waiting room'].every((n) => railNames.some((r) => r.startsWith(n.split(' ')[0]))) && railNames.includes('Parking lot'), 'a rail station still has terminal, parking lot, waiting room, passing tracks', railNames.join(', '));
  const railTip = (await upgrades(g)).find((u) => u.name === 'Terminal')?.tip;
  check(/\$37\.5M/.test(railTip ?? ''), 'a rail station terminal still costs $37.5M', railTip);

  // 2. crossing track drawn by a player between two double tracks, then a line over stations on each: with no
  // crossing track no way joins the stops; with it the line runs, changing track there, on tram as on rail
  const lines = {};
  for (const [kind, y] of [['tram', 1.2], ['regular', 2.4]]) {
    await g.act.camera(P(0, y + 0.15), 14);
    await g.act.tool('track', { type: kind, tracks: 2, snap: kind === 'tram' ? { real: false, streets: false } : { real: false, roads: false } });
    const a = await g.act.drawTrack([P(-1, y), P(1, y)]);
    const b = await g.act.drawTrack([P(-1, y + 0.3), P(1, y + 0.3)]);
    check(a.ok && b.ok, `${kind}: two double tracks drawn side by side`);
    const s1 = await g.act.placeStation(P(-0.6, y));
    const s2 = await g.act.placeStation(P(0.6, y + 0.3));
    check(s1.ok && s2.ok, `${kind}: a stop on each`);
    await g.act.construct();
    const before = await g.page.evaluate(async ([p1, p2]) => {
      const tw = window.tw, w = tw.world.value;
      const near = (ll) => { const p = tw.lngLatToPack(...ll); let best = null, bd = 1e12; for (const n of w.nodes.values()) { if (!n.station && !w.save.stations.some((s) => String(s.num) === String(n.id))) continue; const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
      const r = await tw.client.edit({ op: 'addLine', stops: [near(p1), near(p2)], name: 'probe', colour: '#c0392b' });
      return { ok: r.ok, issues: r.issues ?? null };
    }, [P(-0.6, y), P(0.6, y + 0.3)]);
    check(before.ok === false, `${kind}: no line between the stops before there is crossing track`, JSON.stringify(before));
    await g.act.tool('track', { type: kind, tracks: 2 });
    const link = await g.act.drawTrack([P(-0.1, y), P(0.1, y + 0.3)]);
    check(link.ok, `${kind}: crossing track drawn by the pointer between the two`, JSON.stringify(link.segments.map((x) => x.edges.map((e) => e.tracks))));
    await g.act.construct();
    const line = await g.act.newLine([P(-0.6, y), P(0.6, y + 0.3)], { name: `X ${kind}`, zoom: 15 }).catch((e) => ({ err: e.message }));
    lines[kind] = line;
    check(!!line && !line.err && line.stops?.length === 2, `${kind}: a line over the two stops runs, changing track over the crossing`, JSON.stringify(line).slice(0, 160));
    if (kind === 'tram') {
      const tl = await g.page.evaluate(() => window.tw.world.value.lineStats[window.tw.world.value.save.lines.at(-1).id]?.tram);
      check(tl === true, 'tram: the line is a tram line');
    }
  }
  await g.shot('C:/Users/anita/AppData/Local/Temp/claude/dce1fefcc1d43aef/f0b2a436-f42b-45c6-999f-26d970f36c1a/scratchpad/tramterm/crossing.png');

  // 3. the crossover edit on tram and rail quads, same price at ground level
  const price = {};
  for (const [kind, y] of [['tram', 3], ['regular', 3.6]]) {
    await g.act.camera(P(0, y), 14);
    await g.act.tool('track', { type: kind, tracks: 4 });
    const q = await g.act.drawTrack([P(-1.5, y), P(1.5, y)]);
    if (!q.ok) { check(false, `${kind} quad drawn`, JSON.stringify(q.segments?.map((s) => s.notes))); continue; }
    const r = await g.page.evaluate(async (y) => {
      const tw = window.tw, c = tw.client;
      const w = tw.world.value;
      const p = tw.lngLatToPack(...y);
      const edges = [...w.edges.values()].filter((e) => !e.crossover);
      let best = null, bd = 1e12;
      for (const e of edges) { const a = w.nodes.get(e.a), b = w.nodes.get(e.b); const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2; const d = Math.hypot(mx - p.x, my - p.y); if (d < bd) { bd = d; best = e; } }
      const pre = await new Promise((res) => c.checkCrossover(best.id, p.x, p.y, res));
      return { ok: pre.ok, cost: pre.cost, issues: pre.issues };
    }, y === 3 ? P(0, 3) : P(0, 3.6));
    price[kind] = r;
    console.log(`${kind} crossover preview:`, JSON.stringify(r));
  }
  check(price.tram?.ok === true, 'a tram quad takes a crossover (preview ok)', JSON.stringify(price.tram));
  check(price.tram?.ok && price.regular?.ok && Math.abs(price.tram.cost - price.regular.cost) < 1e-6, 'priced as rail at ground level off street', `${price.tram?.cost} / ${price.regular?.cost}`);
  check(g.pageErrors().length === 0, 'no console errors', g.pageErrors().slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await g.close();
}
console.log(fail ? `T-3574 FAILED: ${fail}` : 'T-3574 PASSED');
process.exit(fail ? 1 : 0);
