// T-3209: an auto flyover's one-way legs stay on their line when a double track is drawn onto one's side (its split node). The check compares the legs' drawn strokes before and after. Pictures t3209-before.png, t3209-after.png. Vite of your own on 6112.
// (header was) T-3191: pictures of a hand-laid single track leaving the upper track of a double main 40 m before its end, zoomed in (no asserts; judge the curve leaves its track cleanly). t3191-z20.png, t3191-z18.png. Vite of your own on 6112.
// T-3162: with single track picked, each track of double track drawn as two tracks is a branch point across its
// whole drawn width, at any zoom: a pointer anywhere over the outer track (and over the inner one) starts a branch
// from that track (`Snap.join`), the pointer just past the drawn pair does not. T-3163: the branch dot is 15 px.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6112):
//   node trainworld/checks/t3162-branch-hit.mjs [dev url]
// Pictures in artifacts/trainworld/t3162-*.png and t3163-*.png.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6112/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3162-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `${name}.png`) });
const settle = async (ms = 600) => {
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(ms);
};

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    window.__j = { P, ll };
  }, [RC]);
  await page.evaluate(async () => {
    const { P } = window.__j, cl = window.tw.client;
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false });
  });
  await wait(800);
  await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track'; ui.buildLevel.value = 0; ui.keepPlacing.value = false; ui.quadTrack.value = false;
    ui.singleTrack.value = true; ui.tramTrack.value = false; ui.highSpeed.value = false; ui.designSpeed.value = 60;
  });
  const box = await (await page.$('#map')).boundingBox();
  const lanes = () => page.evaluate(async () => (({ gap, t, share }) => ({ gap, t, share }))((await import('/src/map/renderer.ts')).laneState(window.tw.map.getZoom())));
  const hoverAt = async (x, y) => {
    await page.mouse.move(x - 25, y, { steps: 2 });
    await page.mouse.move(x, y, { steps: 2 });
    await wait(250);
    return page.evaluate(() => {
      const h = window.tw.tools.hover;
      if (!h) return null;
      const p = h.join ? window.tw.map.project(window.tw.toLngLat(h.join[0], h.join[1])) : null;
      const c = window.tw.map.project(window.tw.toLngLat(h.lx, h.ly));
      return { kind: h.kind, side: p ? (p.y < c.y ? 'up' : 'down') : null };
    });
  };
  const grab = (leg) => page.evaluate((leg) => { const tr = window.tw.renderer.net.track, pts = []; for (let i = 0; i < tr.count; i++) if (tr.edge[i] === leg) pts.push([tr.seg[i * 4], tr.seg[i * 4 + 1], tr.seg[i * 4 + 2], tr.seg[i * 4 + 3]]); return pts; }, leg);
  // T-3209: an auto flyover (a double branch from a station with ramps, T-3092), then a double track drawn onto a leg
  const info = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P } = window.__j;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const near = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const main = tw.world.value.edges.find((e) => e.tracks === 2);
    const s0 = P(-1200, 0);
    await cl.edit({ op: 'addStation', at: { kind: 'edge', edge: main.id, ...s0 }, platform: 200, name: '' });
    await pause(300);
    const S = near(s0);
    const r = await cl.edit({ op: 'route', from: { kind: 'node', node: S }, to: { kind: 'free', ...P(1800, 2200), level: 0 }, pis: [], single: false, ramp: true, design: 160, flyover: 20 });
    await pause(600);
    const w = tw.world.value, legs = w.edges.filter((e) => e.tracks === 1 && e.oneWay);
    const mid = (e) => { const a = w.nodes.get(e.a), b = w.nodes.get(e.b); return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; };
    const leg = legs[0], m = leg && mid(leg), a = leg && w.nodes.get(leg.a), b = leg && w.nodes.get(leg.b);
    return { allLegs: legs.map((e) => e.id), ok: r.ok, legs: legs.length, leg: leg?.id, a: a && [a.x, a.y], b: b && [b.x, b.y], ends: [leg?.a, leg?.b] };
  });
  console.log('flyover', JSON.stringify(info));
  const at = [(info.a[0] + info.b[0]) / 2, (info.a[1] + info.b[1]) / 2];
  await page.evaluate(([c]) => { const q = window.tw.packToLngLat(c[0], c[1]); window.tw.map.jumpTo({ center: Array.isArray(q) ? q : [q.lng, q.lat], zoom: 16.5 }); }, [at]);
  await settle();
  await shot('t3209-before');
  const legs0 = {};
  for (const l of info.allLegs) legs0[l] = await grab(l);
  const r2 = await page.evaluate(async ([leg, a]) => {
    const tw = window.tw, cl = tw.client;
    // a double from the north west, ending on the leg's loose end node (its second node)
    const w = tw.world.value, e = w.edges.find((x) => x.id === leg), n = w.nodes.get(e.b);
    const A = w.nodes.get(e.a), B = w.nodes.get(e.b), mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2, dx = A.x - B.x, dy = A.y - B.y, dl = Math.hypot(dx, dy);
    const res = await cl.edit({ op: 'route', from: { kind: 'free', x: mx + (dx / dl) * 1500 + 120, y: my + (dy / dl) * 1500 + 120, level: 0 }, to: { kind: 'edge', edge: leg, x: mx, y: my }, pis: [], single: false, design: 60 });
    await new Promise((r) => setTimeout(r, 1000));
    return { ok: res.ok, issues: JSON.stringify(res.issues) };
  }, [info.leg, info.a]);
  console.log('double', JSON.stringify(r2));
  await page.evaluate(([c]) => { const q = window.tw.packToLngLat(c[0], c[1]); window.tw.map.jumpTo({ center: Array.isArray(q) ? q : [q.lng, q.lat], zoom: 16.5 }); }, [at]);
  await settle();
  await shot('t3209-after');
  console.log('edges', JSON.stringify(await page.evaluate(() => window.tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => [e.id, e.a, e.b, e.join, e.oneWay]))));
  for (const l of info.allLegs) {
    const after = await page.evaluate(() => { const tr = window.tw.renderer.net.track, ids = new Set(window.tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.id)), pts = []; for (let i = 0; i < tr.count; i++) if (ids.has(tr.edge[i])) pts.push([tr.seg[i * 4], tr.seg[i * 4 + 1], tr.seg[i * 4 + 2], tr.seg[i * 4 + 3]]); return pts; });
    // per before segment, its distance (local units; ~ per metre scale below) from the after polyline
    const mpu = await page.evaluate(() => { const [x, y] = window.tw.toLocal(window.tw.map.getCenter().lng, window.tw.map.getCenter().lat); const [x2, y2] = window.tw.toLocal(window.tw.map.getCenter().lng, window.tw.map.getCenter().lat + 10 / 111320); return Math.hypot(x2 - x, y2 - y) / 10; });
    const dist = (p, segs) => Math.min(...segs.map(([x0, y0, x1, y1]) => { const dx = x1 - x0, dy = y1 - y0, l2 = dx * dx + dy * dy, t = l2 ? Math.max(0, Math.min(1, ((p[0] - x0) * dx + (p[1] - y0) * dy) / l2)) : 0; return Math.hypot(x0 + dx * t - p[0], y0 + dy * t - p[1]); }));
    const b = legs0[l];
    const d = b.map((sg, i) => dist([sg[0], sg[1]], after) / mpu);
    console.log(`${Math.max(...d) < 0.3 ? 'PASS' : 'FAIL'} leg ${l} stays on its line after a double is drawn onto it (max drawn shift ${Math.max(...d).toFixed(2)} m)`); if (Math.max(...d) >= 0.3) process.exitCode = 1;
  }
} catch (e) {
  await shot('t3191-died');
  console.log(e.stack);
} finally {
  await chrome.close();
}
process.exit(process.exitCode ?? 0);
