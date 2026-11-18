// T-3379, T-3380, T-3386, T-3411: single track's loose ends, in a new sandbox game (about 1 minute).
//   a short single piece joined to one track of double track ends on its centre line at its far end (no jog),
//   the Delete tool's red ghost over single track is one track wide,
//   single track that runs one way by its join to a bare end runs both ways once a station is on it (T-3647: a
//   terminus branch; T-3386's way kept through a station on legs joined at both ends is the sim test
//   t3386_a_station_inside_the_run_keeps_its_way),
//   the end of single track takes a one-platform station.
// From the tree root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6111):
//   node trainworld/checks/t3379-single-fixes.mjs [dev url]
import { launchChrome, collectErrors } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6111/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = await launchChrome({ prefix: 'tw-t3379-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
const errors = [];
collectErrors(page, errors);
try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  const r = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, o = tw.lngLatToPack(-73.93, 40.675), out = {};
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const sl = (ms) => new Promise((d) => setTimeout(d, ms));
    const nd = (p) => { let b = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; b = n.id; } } return b; };
    const perM = (lx, ly) => { const [lng, lat] = tw.toLngLat(lx, ly); const [x2, y2] = tw.toLocal(lng, lat + 10 / 111320); return Math.hypot(x2 - lx, y2 - ly) / 10; };
    // a double main to x 0, a single piece on beyond 400, a 25 m piece between them joined to the main's upper track
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(0, 0), level: 0 }, pis: [], single: false, design: 60 });
    await sl(300);
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(25, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: true, design: 60 });
    await sl(300);
    const j = P(0, 2), A = nd(P(0, 0)), B = nd(P(25, 0));
    out.join = await cl.edit({ op: 'route', from: { kind: 'node', node: A }, to: { kind: 'node', node: B }, pis: [], single: true, design: 60, join: [j.x, j.y, null, null] });
    await sl(800);
    const w = tw.world.value, short = w.edges.find((e) => e.tracks === 1 && ((e.a === A && e.b === B) || (e.a === B && e.b === A)));
    const nb = w.nodes.get(B), [bx, by] = tw.toLocal(nb.lng, nb.lat), tr = tw.renderer.net.track;
    let best = Infinity;
    for (let i = 0; i < tr.count; i++) if (tr.edge[i] === short.id) for (const k of [0, 2]) best = Math.min(best, Math.hypot(tr.seg[i * 4 + k] - bx, tr.seg[i * 4 + k + 1] - by));
    out.farOffM = best / perM(bx, by);
    // the Delete tool over the long single piece
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'delete';
    tw.map.jumpTo({ center: (() => { const q = tw.packToLngLat(P(1500, 0).x, P(1500, 0).y); return Array.isArray(q) ? q : [q.lng, q.lat]; })(), zoom: 17 });
    window.__p = P;
    return out;
  });
  check(r.join?.ok, 'a 25 m piece joined to the main is laid', JSON.stringify(r.join?.issues));
  check(r.farOffM < 0.3, 'its far end is on the centre line (no jog)', `${r.farOffM?.toFixed(2)} m`);
  await wait(1500);
  const box = await (await page.$('#map')).boundingBox();
  const [x, y] = await page.evaluate(() => { const tw = window.tw, p = window.__p(1500, 0), q = tw.packToLngLat(p.x, p.y), s = tw.map.project(Array.isArray(q) ? q : [q.lng, q.lat]); return [s.x, s.y]; });
  await page.mouse.move(box.x + x - 30, box.y + y + 2, { steps: 3 });
  await page.mouse.move(box.x + x, box.y + y + 2, { steps: 3 });
  await wait(800);
  const g = await page.evaluate(() => { const s = window.tw.renderer.previewSrc; return s && s.count ? { n: s.count, maxLane: Math.max(...s.lane) } : null; });
  check(g && g.maxLane === 0, 'the Delete ghost over single track is one track wide', JSON.stringify(g));
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'select'; });

  // stations: one inside the single piece (splits it) and one at its far end
  const s = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, P = window.__p, sl = (ms) => new Promise((d) => setTimeout(d, ms));
    const w0 = tw.world.value, long = w0.edges.find((e) => e.tracks === 1 && Math.abs(w0.nodes.get(e.a).x - P(25, 0).x) < 2 && Math.abs(w0.nodes.get(e.b).x - P(3000, 0).x) < 2);
    const before = long.oneWay ?? 0, pt = P(1000, 0);
    await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: long.id, x: pt.x, y: pt.y }, platform: 100, name: '' });
    await sl(500);
    const ways = tw.world.value.edges.filter((e) => e.tracks === 1 && w0.nodes.get(e.b)?.x !== P(25, 0).x).map((e) => e.oneWay ?? 0);
    const end = [...tw.world.value.nodes.values()].find((n) => Math.abs(n.x - P(3000, 0).x) < 2);
    await cl.edit({ op: 'addStation', at: { kind: 'node', node: end.id }, platform: 100, name: '' });
    await sl(500);
    const sw = await cl.edit({ op: 'stationTracks', node: end.id, one: true });
    await sl(300);
    return { before, ways, sw, one: cl.lastNodes.find((n) => n.id === end.id)?.one };
  });
  check(s.before !== 0 && s.ways.length >= 2 && s.ways.every((v) => v === 0), 'one way by its join, and both ways once a station splits it (a terminus branch)', JSON.stringify(s));
  check(s.sw.ok && s.one === 2, 'the end of single track takes a one-platform station', JSON.stringify([s.sw, s.one]));
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `FAILED: ${fail}` : 'PASSED');
process.exit(fail ? 1 : 0);
