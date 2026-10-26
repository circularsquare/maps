// T-1500: two lines meeting at a middle city, Shanghai - Wuhan and Wuhan - Beijing, sharing the
// Wuhan station, with eight metro lines at each end so the inter-city riders spread over many
// station pairs (none of them holding 0.5%). The trip explorer between Shanghai and Beijing must
// find the train trip with its change at Wuhan in every period, on a fresh new game and after a
// save and load, and the panel must show it. A screenshot of each panel goes to
// artifacts/trainworld/t1500/. A sandbox, so money refuses nothing; about 4 minutes.
//   node trainworld/checks/t1500-change-trip.mjs [base url, default http://localhost:6150/]
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { ARTIFACTS, launchChrome, collectErrors, demandFinal, importSave, registry } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6150/';
const OUT = join(ARTIFACTS, 't1500');
mkdirSync(OUT, { recursive: true });
const reg = registry();
const place = (id) => reg.cities.find((c) => c.id === id).place;
const SH = [121.4737, 31.2304], WH = [114.3055, 30.5928], BJ = [116.4074, 39.9042];
let fails = 0;
const say = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} t1500 ${what}${detail ? `: ${detail}` : ''}`); if (!ok) fails++; };

/** Lines through points ([lng, lat] each): track, a station at each point (or the existing one
 * there), constructed, the line added with its trains. `join` routes from the station already at
 * the first point. */
const build = (page, specs) => page.evaluate(async (SPECS) => {
  const tw = window.tw, c = tw.client;
  const pause = (ms) => new Promise((d) => setTimeout(d, ms));
  const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p[0], n.y - p[1]); if (d < bd) { bd = d; best = n.id; } } return best; };
  const lines = [];
  for (const { pts, join, name, colour, trains } of SPECS) {
    const xy = pts.map(([lng, lat]) => { const p = tw.lngLatToPack(lng, lat); return [p.x, p.y]; });
    const first = join ? { kind: 'node', node: near(xy[0]) } : { kind: 'free', x: xy[0][0], y: xy[0][1], level: -1 };
    await c.edit({ op: 'route', from: first, to: { kind: 'free', x: xy[1][0], y: xy[1][1], level: -1 }, pis: [], single: false });
    for (let k = 2; k < xy.length; k++) {
      await pause(400);
      await c.edit({ op: 'route', from: { kind: 'node', node: near(xy[k - 1]) }, to: { kind: 'free', x: xy[k][0], y: xy[k][1], level: -1 }, pis: [], single: false });
    }
    await pause(600);
    // the shared station refuses a second one (StationPorts), which is what is wanted
    for (const [k, p] of xy.entries()) if (!(join && k === 0)) { await c.edit({ op: 'addStation', at: { kind: 'node', node: near(p) }, platform: 200, name: '' }); await pause(150); }
    await c.edit({ op: 'constructAll' });
    await pause(600);
    const r = await c.edit({ op: 'addLine', stops: xy.map(near), name, colour });
    if (r?.line !== undefined) await c.edit({ op: 'setTrainCount', line: r.line, trains });
    lines.push(r?.line ?? null);
  }
  return lines;
}, specs);

/** The explorer between Shanghai and Beijing on the client's own reading, each period. */
const explorer = (page) => page.evaluate(async ({ near, far }) => {
  const ps = await window.twDemand.client.marketTrip('home', new Uint32Array(0), far, -1, { place: near, gid: -1 });
  return ps && ps.map((p) => p && { trips: p.trips, rail: p.rail, why: p.why, routes: p.routes.length, lines: p.routes[0] ? p.routes[0].legs.filter((l) => l.kind === 'ride').map((l) => l.line) : [] });
}, { near: place('sha'), far: place('bei') });

async function judge(page, label, lineA, lineB) {
  await demandFinal(page, { timeout: 400000 });
  await page.waitForTimeout(1500);
  await demandFinal(page, { timeout: 400000 });
  const ps = await explorer(page);
  say(!!ps && ps.length === 5 && ps.every((p) => p), `${label}: the explorer answers every period`);
  if (!ps) return;
  const brief = ps.map((p) => p && `${Math.round(p.trips)} trips, rail ${p.rail.toExponential(1)}, ${p.routes} routes [${p.lines}]${p.why ? ` ${p.why}` : ''}`).join('; ');
  say(ps.every((p) => p && !p.why && p.rail > 0), `${label}: rail offered with riders in every period`, brief);
  say(ps.every((p) => p && p.routes > 0 && p.lines.includes(String(lineA)) && p.lines.includes(String(lineB))), `${label}: every period's first route rides both lines, changing at Wuhan`);
  const [a, b] = await page.evaluate(([a, b]) => { const f = window.tw.lngLatToPack; const p = f(a[0], a[1]), q = f(b[0], b[1]); return [[p.x, p.y], [q.x, q.y]]; }, [SH, BJ]);
  await page.evaluate(({ a, b, near, far }) => window.tw.select({ kind: 'trip', from: a, to: b, far: { place: far, gid: -1 }, near: { place: near, gid: -1 } }), { a, b, near: place('sha'), far: place('bei') });
  await page.waitForFunction(() => /By train/.test(document.body.innerText) && !/Calculating demand/.test(document.body.innerText), undefined, { timeout: 60000 });
  await page.waitForTimeout(800);
  const text = await page.evaluate(() => document.body.innerText);
  say(!/No train serves both ends/.test(text) && /Change to/.test(text), `${label}: the panel shows the trip with its change`);
  await page.screenshot({ path: join(OUT, `${label.replace(/\W+/g, '-')}.png`) });
}

const { ctx, close } = await launchChrome({ prefix: 't1500-' });
const errors = [];
try {
  const page = await ctx.newPage();
  collectErrors(page, errors);
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 6 }), WH);
  await page.evaluate(() => window.tw.client.post({ kind: 'sandbox' }));
  await page.waitForTimeout(2000);
  const radial = (c, deg) => Array.from({ length: 12 }, (_, k) => { const r = (0.25 + k * 2.5) / 111.32; return [c[0] + (r * Math.cos((deg * Math.PI) / 180)) / Math.cos((c[1] * Math.PI) / 180), c[1] + r * Math.sin((deg * Math.PI) / 180)]; });
  const metro = [];
  for (const [c, tag] of [[SH, 'S'], [BJ, 'J']]) for (const deg of [20, 65, 110, 155, 200, 245, 290, 335]) metro.push({ pts: radial(c, deg), name: `${tag}${deg}`, colour: '#1f6fb2', trains: [20, 15, 10] });
  const [lineA, lineB, ...rest] = await build(page, [
    { pts: [SH, [118.8, 32.05], WH], name: 'A', colour: '#0a7d3b', trains: [80, 60, 40] },
    { pts: [WH, [113.65, 34.75], BJ], join: true, name: 'B', colour: '#d0312d', trains: [80, 60, 40] },
    ...metro,
  ]);
  const shared = await page.evaluate(([a, b]) => { const ls = window.tw.world.value.save.lines; const A = ls.find((l) => l.id === String(a)) ?? ls[a], B = ls.find((l) => l.id === String(b)) ?? ls[b]; return A && B && A.stops.at(-1) === B.stops[0]; }, [lineA, lineB]);
  say(lineA !== null && lineB !== null && rest.every((l) => l !== null) && shared, 'new game: the two lines and 16 metro lines built, sharing the Wuhan station');
  await judge(page, 'new game', lineA, lineB);

  const dl = page.waitForEvent('download', { timeout: 60000 });
  await page.evaluate(() => window.tw.client.post({ kind: 'export' }));
  const saved = readFileSync(await (await dl).path());
  await page.goto(`${BASE}?debug=1&new=1&paused=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.twDemand?.client, undefined, { timeout: 180000 });
  await importSave(page, saved);
  await page.waitForFunction(() => window.tw.client.lastLoad?.from === 'file' && window.tw.world.value.save.lines.length === 18, undefined, { timeout: 180000 });
  await page.evaluate((c) => void window.tw.map.jumpTo({ center: c, zoom: 6 }), WH);
  await judge(page, 'after load', lineA, lineB);
} finally {
  await close();
}
say(errors.length === 0, 'no console errors', errors.slice(0, 3).join(' | '));
process.exit(fails ? 1 : 0);
