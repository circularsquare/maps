// T-3241: rebuilding the one piece between two stations to another level (the players' case: a 4.3 km piece
// between Roissy-en-Brie and Ozoir-la-Ferriere to -2). On a new game in New York: a run of three pieces, a
// station at each end of the 4.3 km middle one (with a corner in it) and at the run's two ends, built, with a
// line over it. Through the Select tool's bar, the middle piece alone, stations not picked: to -2, then to +1.
// Each time both stations go to the level with it, the ramps go on the pieces beyond them (drawn in the pick's
// band before Rebuild, and in the hover's length and price), the cash taken is the quote, and the line runs.
// Then a piece whose station has a junction 250 m beyond it: no room for the ramp, so it is drawn red and the
// refusal says there is not enough room to change level. Screenshots artifacts/trainworld/t3241/.
// From the maps root, on a Vite of your own:
//   node trainworld/checks/t3241-rebuild-stations.mjs [dev url]
// About a minute.
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6134/';
const OUT = join(ARTIFACTS, 't3241');
mkdirSync(OUT, { recursive: true });
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const C = [-73.93, 40.668];

const chrome = await launchChrome({ prefix: 'tw-t3241-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);

const snap = () => page.evaluate(() => {
  const w = window.tw.world.value;
  return {
    cash: w.save.cash,
    edges: w.edges.map((e) => ({ id: Number(e.id), a: Number(e.a), b: Number(e.b), built: e.built, lo: e.levelMin, hi: e.levelMax, len: e.lengthM })),
    nodes: Object.fromEntries([...w.nodes.values()].map((n) => [Number(n.id), n.level])),
    lines: w.save.lines.map((l) => ({ num: Number(l.num), stops: l.stops.map(Number), status: w.lineStats[l.id]?.status })),
  };
});
const quote = (edges, stations, level, kind) => page.evaluate(async ([edges, stations, level, kind]) => {
  const { queryClock } = await import('/src/workers/clockClient.ts');
  const d = await queryClock({ what: 'rebuildQuote', edges, stations, level, kind });
  const refused = d[3], end = 5 + refused + d[1], beyond = [];
  for (let k = end; k + 2 < d.length; k += 3) beyond.push({ edge: d[k], from: d[k + 1], m: d[k + 2] });
  return { price: d[0], changed: Array.from(d.subarray(5 + refused, end), Number), stations: d[2], refused: Array.from(d.subarray(5, 5 + refused), Number), beyond };
}, [edges, stations, level, kind]);
const pick = (edges) => page.evaluate(async (edges) => { (await import('/src/game/blueprintPick.ts')).bpPicked.value = { edges, stations: [], built: edges.length > 0 }; }, edges);
const jump = (lng, lat, zoom) => page.evaluate(([lng, lat, zoom]) => window.tw.map.jumpTo({ center: [lng, lat], zoom }), [lng, lat, zoom]);

/** Build a run from `xs` (m east of C, at `dy` m north), stations at `stops` (indices into xs), constructed, with a line. */
const build = (xs, dy, stops, extra) => page.evaluate(async ([c, xs, dy, stops, extra]) => {
  const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
  const pause = (ms) => new Promise((d) => setTimeout(d, ms));
  // pack metres are Mercator metres: ground metres times 1 / cos(latitude)
  const k = 1 / Math.cos((c[1] * Math.PI) / 180);
  const P = (dx, y = dy) => ({ x: o.x + dx * k, y: o.y + y * k });
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
  const res = [];
  let from = { kind: 'free', ...P(xs[0]), level: 0 };
  for (let i = 1; i < xs.length; i++) {
    const pis = xs[i].corner ? [{ ...P(xs[i].corner[0], dy + xs[i].corner[1]), level: 0 }] : [];
    const x = xs[i].x ?? xs[i];
    res.push(await cl.edit({ op: 'route', from, to: { kind: 'free', ...P(x), level: 0 }, pis, single: false })); await pause(250);
    from = { kind: 'node', node: near(P(x)) };
  }
  const xOf = (v) => v.x ?? v;
  const nodes = xs.map((v) => near(P(xOf(v))));
  if (extra) {
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: nodes[extra.from] }, to: { kind: 'free', ...P(extra.to[0], dy + extra.to[1]), level: 0 }, pis: [], single: false })); await pause(250);
  }
  const st = stops.map((i) => nodes[i]);
  for (const n of st) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 160, name: '' })); await pause(200); }
  res.push(await cl.edit({ op: 'constructAll' })); await pause(500);
  const line = await cl.edit({ op: 'addLine', stops: st, name: 'Through the stations', colour: '#c0392b' });
  await pause(800);
  return { nodes, st, line: line.line, ok: line.ok, refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 160)) };
}, [C, xs, dy, stops, extra]);

const edgeBetween = (s, a, b) => s.edges.find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
const lngAt = (dx) => C[0] + dx / (111320 * Math.cos((C[1] * Math.PI) / 180));
const latAt = (dy) => C[1] + dy / 110540;

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'blueprint'; });

  // the players' case: 2.85 km, the 4.3 km piece between two stations (a corner 300 m off the straight), 2.85 km
  const run = await build([-5000, -2150, { x: 2150, corner: [0, 300] }, 5000], 0, [0, 1, 2, 3]);
  check(run.ok && !run.refused.length, 'the run, its stations and its line are built', run.refused.join(' | '));
  if (!run.ok) throw new Error('no line');
  const [n0, s1, s2, n3] = run.nodes;
  let s = await snap();
  const mid = edgeBetween(s, s1, s2), west = edgeBetween(s, n0, s1), east = edgeBetween(s, s2, n3);
  check(mid && west && east && Math.abs(mid.len - 4300) < 200, 'three pieces, the middle about 4.3 km', JSON.stringify([mid?.len, west?.len, east?.len]));
  const line0 = s.lines.find((l) => l.num === run.line);

  for (const [level, label] of [[-2, '−2'], [1, '+1']]) {
    const before = await snap();
    await jump(C[0], C[1], 12.6);
    await pick([mid.id]);
    await page.waitForTimeout(500);
    await page.locator('.blueprint-bar .btn.lv', { hasText: label }).first().click();
    await page.waitForTimeout(1500);
    const q = await quote([mid.id], [], level, 0);
    console.log(`quote to ${label}:`, JSON.stringify(q));
    check(!q.refused.length && q.stations === 2 && q.changed.includes(mid.id), `${label}: nothing refused, both stations charged`, JSON.stringify(q));
    // the west piece is straight from its station: the ramp ends exactly past the platform the first time; the
    // east one leaves its station on a curve (the piece before it bends), so its ramp waits for the straight after
    const ramp = 80 + 2 + 200 * Math.abs(level - before.nodes[s1]);
    const wb = q.beyond.find((b) => b.edge === west.id);
    if (level === -2) check(wb && Math.abs(wb.m - ramp) < 1, `${label}: on the straight piece the ramp ends ${ramp} m past the station`, JSON.stringify(wb));
    check(q.beyond.length === 2 && q.beyond.every((b) => [west.id, east.id].includes(b.edge) && [s1, s2].includes(b.from) && b.m > 280 && b.m < 1200), `${label}: the ramps go on the pieces beyond the stations`, JSON.stringify(q.beyond));
    const tip = await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).getAttribute('data-tip');
    console.log('hover:', JSON.stringify(tip));
    const km = Number(/([\d.]+) km/.exec(tip ?? '')?.[1]);
    const wantKm = (mid.len + q.beyond.reduce((t, b) => t + b.m, 0)) / 1000;
    check(!/change/.test(tip ?? '') && Math.abs(km - wantKm) < 0.06, `${label}: the hover's length takes in the ramps beyond`, `${tip} (want ${wantKm.toFixed(2)} km)`);
    await page.screenshot({ path: join(OUT, `pick-${level}.png`) });
    // the west station with the band past it, below the loading card in the map's middle
    await jump(lngAt(-2150 + 150), latAt(450), 15);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `pick-${level}-station.png`) });
    await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).click();
    await page.waitForSelector('dialog.confirmation[open]', { timeout: 5000 });
    const ask = await page.locator('dialog.confirmation #confirmation-text').innerText();
    console.log('confirm:', ask);
    check(/2 built stations/.test(ask), `${label}: the question names the two stations`, ask);
    await page.locator('dialog.confirmation .btn.danger').click();
    await page.waitForTimeout(2500);
    s = await snap();
    const m = s.edges.find((e) => e.id === mid.id), w = s.edges.find((e) => e.id === west.id), e = s.edges.find((x) => x.id === east.id);
    check(Math.abs(before.cash - s.cash - q.price * 1e6) < 2, `${label}: the cash taken is the quote`, `taken ${before.cash - s.cash}, quoted ${q.price * 1e6}`);
    check(s.nodes[s1] === level && s.nodes[s2] === level, `${label}: both stations are on the level`, `${s.nodes[s1]} ${s.nodes[s2]}`);
    check(m.lo === level && m.hi === level, `${label}: the piece is on the level`, `${m.lo}..${m.hi}`);
    check(s.nodes[n0] === 0 && s.nodes[n3] === 0 && Math.min(w.lo, e.lo) === Math.min(level, 0) && Math.max(w.hi, e.hi) === Math.max(level, 0), `${label}: the pieces beyond ramp from the level back to ground`, JSON.stringify([w.lo, w.hi, e.lo, e.hi]));
    const ln = s.lines.find((l) => l.num === run.line);
    check(ln && JSON.stringify(ln.stops) === JSON.stringify(line0.stops) && ln.status !== 'broken', `${label}: the line keeps its stops and runs`, ln?.status);
    // the track coloured by level: the piece and both stations on the level, ramps past the stations
    await pick([]);
    await page.getByRole('button', { name: 'Level', exact: true }).first().click();
    await jump(C[0], latAt(1300), 13.4);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `after-${level}-levels.png`) });
    await jump(lngAt(-2150 + 150), latAt(450), 15);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(OUT, `after-${level}-station.png`) });
    await page.getByRole('button', { name: 'Line', exact: true }).first().click();
  }

  // no room: a junction 250 m beyond a station, and a piece with no corners to ramp on itself
  await pick([]);
  const dy = 3000;
  const nr = await build([-3000, -1500, 1500, 1750, 3000], dy, [0, 1, 2, 4], { from: 3, to: [3200, 1200] });
  check(nr.ok && !nr.refused.length, 'the run with a junction is built', nr.refused.join(' | '));
  s = await snap();
  const nmid = edgeBetween(s, nr.nodes[1], nr.nodes[2]);
  await jump(lngAt(0), latAt(dy), 13.4);
  await pick([nmid.id]);
  await page.waitForTimeout(500);
  await page.locator('.blueprint-bar .btn.lv', { hasText: '−2' }).first().click();
  await page.waitForTimeout(1500);
  const qn = await quote([nmid.id], [], -2, 0);
  check(qn.refused.length === 1 && qn.refused[0] === nmid.id && !qn.changed.length, 'no room for the ramp before the junction: the piece is refused', JSON.stringify(qn));
  await page.screenshot({ path: join(OUT, 'no-room.png') });
  await page.locator('.blueprint-bar button', { hasText: 'Rebuild' }).click();
  await page.waitForTimeout(1500);
  const note = await page.evaluate(async () => (await import('/src/game/world.ts')).notice.value?.text ?? '');
  console.log('notice:', note);
  check(/Not enough room to change level/.test(note), 'the refusal says there is no room to change level', note);
  const s9 = await snap();
  check(s9.nodes[nr.nodes[1]] === 0 && s9.nodes[nr.nodes[2]] === 0, 'the stations stay');
} catch (e) {
  fail++;
  console.log('FAIL exception', e?.stack ?? e);
}
check(!errors.length, 'no console or page errors', errors.slice(0, 3).join(' | '));
await chrome.close();
console.log(fail ? `FAIL: ${fail} failed` : 'PASS: all checks');
process.exit(fail ? 1 : 0);
