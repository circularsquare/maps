// T-3117, T-3120: how a flyover is laid and drawn. A double main with a station on it, and a double branch drawn
// from that station (the click right after its platform) with the ramp switch on, which lays a flyover (T-3092);
// stations at the ends, a line on each, all built. Checks that the flyover was laid with nothing behind the click
// (two junctions, J where both legs leave the main and J3 where they meet, neither with a flat badge, no flat
// crossing), that it is short along the main, that the branch line's return run is on the line rows (`back`), and
// that a low target speed (T-3121) makes it shorter still. Screenshots J3 (where the branch parts into its two legs),
// J and the whole flyover in light and dark, in artifacts/trainworld/t3120-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6181):
//   node trainworld/checks/t3117-flyover.mjs [dev url]
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';
import { join } from 'node:path';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6181/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };

const RC = [-73.93, 40.675];
const chrome = await launchChrome({ prefix: 'tw-t3120-', size: [1400, 900], deviceScaleFactor: 1 });
const errors = [];
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', display: { dayShown: 'weekday' }, v: 2 })); } catch {} });
collectErrors(page, errors);
const settle = async (ms = 800) => { await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }); await page.waitForTimeout(ms); };

try {
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });

  // a main at `dy` north of the centre, a station on it at x = -1200 and a branch from that station at
  // `design` km/h; the flyover's junctions and their span along the main past the station, m
  const build = (dy, design, lines) => page.evaluate(async ([c, dy, design, lines]) => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, y) => ({ x: o.x + dx, y: o.y + dy + y });
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const res = [];
    const w0 = P(-3000, 0), e0 = P(3000, 0), b0 = P(1800, 2200), s0 = P(-1200, 0);
    const before = new Set(tw.world.value.nodes.keys());
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...w0, level: 0 }, to: { kind: 'free', ...e0, level: 0 }, pis: [], single: false }));
    await pause(400);
    const main = tw.world.value.edges.find((e) => e.tracks === 2 && !before.has(e.a));
    res.push(await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: main.id, ...s0 }, platform: 200, name: '' }));
    await pause(300);
    const S = near(s0);
    res.push(await cl.edit({ op: 'route', from: { kind: 'node', node: S }, to: { kind: 'free', ...b0, level: 0 }, pis: [], single: false, ramp: true, design }));
    await pause(500);
    const [W, E, B] = [near(w0), near(e0), near(b0)];
    let m = { ok: true }, b = { ok: true };
    if (lines) {
      for (const n of [W, E, B]) { res.push(await cl.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 200, name: '' })); await pause(200); }
      res.push(await cl.edit({ op: 'constructAll' }));
      await pause(500);
      m = await cl.edit({ op: 'addLine', stops: [W, E], name: 'Main', colour: '#e6b800' });
      await pause(400);
      b = await cl.edit({ op: 'addLine', stops: [W, B], name: 'Branch', colour: '#f28c00' });
      await pause(800);
    }
    const w = tw.world.value;
    const sNode = w.nodes.get(S);
    const junctions = [...w.nodes.values()].filter((n) => !before.has(n.id) && n.ports >= 3 && n.platform === 0);
    const legs = w.edges.filter((e) => e.tracks === 1 && e.oneWay && !before.has(e.a));
    // the span along the main, from the station's platform end on, that the flyover's junctions take
    const along = junctions.map((n) => (n.lng - sNode.lng) * Math.cos(sNode.lat * Math.PI / 180) * 111320);
    return { refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues ?? r).slice(0, 80)), main: m.line, branch: b.line, ok: m.ok && b.ok,
      junctions: junctions.map((n) => ({ id: n.id, lng: n.lng, lat: n.lat, crosses: !!n.crosses, ports: n.ports })), legs: legs.map((e) => e.id),
      crossings: (w.crossings ?? []).length / 5, span: Math.max(...along) - 100, behind: Math.min(...along) - 100 };
  }, [RC, dy, design, lines]);

  const built = await build(0, 160, true);
  check(!built.refused.length && built.ok, 'a main, a station, a branch from it with a flyover, stations and two lines', built.refused.join(' | '));
  check(built.junctions.length === 2 && built.junctions.every((j) => !j.crosses) && built.junctions.some((j) => j.ports === 4), 'two junctions, J with four ends, neither with a flat badge', JSON.stringify(built.junctions.map((j) => [j.ports, j.crosses])));
  check(built.legs.length >= 2, 'one-way legs laid', `${built.legs.length}`);
  check(built.crossings === 0, 'no flat crossing', `${built.crossings}`);
  check(built.behind >= 0 && built.span < 560, 'nothing behind the click, under 560 m along the main', `J at ${built.behind.toFixed(0)} m past the platform, J3 ${built.span.toFixed(0)} m`);
  const lines = await page.evaluate(() => window.tw.client.lastLines.map((l) => ({ name: l.name, path: l.path, back: l.back ?? [] })));
  const br = lines.find((l) => l.name === 'Branch');
  const own = new Set(br.path.map((p) => p >> 1));
  const backOnly = br.back.map((p) => p >> 1).filter((e) => !own.has(e));
  check(backOnly.length > 0 && backOnly.some((e) => built.legs.includes(e)), 'the branch line\'s way back takes the other leg, on the line rows', `back-only edges ${backOnly}`);
  // T-3121: the same branch at 60 km/h, further north, is shorter along the main
  const slow = await build(3000, 60, false);
  check(!slow.refused.length && slow.junctions.length === 2 && slow.span < built.span, 'at 60 km/h a flyover too, shorter along the main', `${slow.span.toFixed(0)} m vs ${built.span.toFixed(0)} m`);
  console.log(`span along the main past the platform: ${built.span.toFixed(0)} m at 160 km/h, ${slow.span.toFixed(0)} m at 60 km/h`);

  await page.evaluate(() => { window.tw.selection.value = null; });
  await page.getByText(/Loading city data|Calculating demand/).first().waitFor({ state: 'hidden', timeout: 240000 }).catch(() => {});
  // J3 is the junction furthest from the main's line (its latitude), J the one on it
  const js = built.junctions.slice().sort((a, b) => b.lat - a.lat);
  const shot = async (name, at, zoom) => {
    await page.evaluate(([c, z]) => { window.tw.map.jumpTo({ center: c, zoom: z }); window.tw.overlay.invalidate(); }, [at, zoom]);
    await settle(1800);
    await page.screenshot({ path: join(ARTIFACTS, name) });
  };
  const j3 = [js[0].lng, js[0].lat], j = [js[1].lng, js[1].lat];
  for (const shade of ['light', 'dark']) {
    await page.evaluate((s) => { window.tw.shade.value = s; }, shade);
    await page.waitForTimeout(800);
    await shot(`t3120-j3-${shade}.png`, j3, 17.3);
    await shot(`t3120-j-${shade}.png`, j, 17.0);
    await shot(`t3120-whole-${shade}.png`, [(j3[0] + j[0]) / 2, (j3[1] + j[1]) / 2], 15.6);
  }
  await page.evaluate(() => { window.tw.shade.value = 'light'; });
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  console.log('FAIL', e?.stack ?? e);
  fail++;
} finally {
  await chrome.close?.();
}
console.log(fail ? `FAIL: ${fail}` : 'PASS: all');
process.exit(fail ? 1 : 0);
