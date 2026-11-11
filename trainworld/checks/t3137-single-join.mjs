// T-3137 C3: single track drawn off or onto double track takes the track the pointer is over. On a new game, a
// double main running east-west; zoomed in where it draws as two tracks, single track is drawn by the mouse:
//   A off the upper (north) track heading north, B off the lower track heading north (across the upper),
//   C onto the lower track from the south, D onto the upper track from the south (across the lower),
//   E a station at A's far end (a terminus branch, two-way).
// For each: the snap's join point (`Snap.join`) sits on the track hovered, the preview request carries it
// (`RouteAsk.join`), the ghost's end is drawn 2 m off the centre line on that track, the badge shows only where a
// move crosses (B, D, E), and once built the piece's join (`ClockEdge.join`) and its drawn end match. Zoomed out
// the snap has no join. Pictures in artifacts/trainworld/t3137-join-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6197):
//   node trainworld/checks/t3137-single-join.mjs [dev url]
// About 2 minutes.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6197/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];
const NEAR = 18, FAR = 16;

const chrome = await launchChrome({ prefix: 'tw-t3137j-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3137-join-${name}.png`) });
const settle = async (ms = 600) => {
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(ms);
};

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });

  // helpers in the page: pack points near RC, the main's signed offset of a local point in metres (+ is up on screen)
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    const mainIds = () => tw.world.value.edges.filter((e) => e.tracks === 2).map((e) => e.id);
    // local units per metre where (lx, ly) is
    const perM = (lx, ly) => { const [lng, lat] = tw.toLngLat(lx, ly); const [x2, y2] = tw.toLocal(lng, lat + 10 / 111320); return Math.hypot(x2 - lx, y2 - ly) / 10; };
    const offM = (lx, ly) => {
      let best = null;
      for (const id of mainIds()) {
        const pts = tw.renderer.net.edgePts.get(id);
        if (!pts) continue;
        for (let k = 0; k + 5 < pts.length; k += 3) {
          const ux = pts[k + 3] - pts[k], uy = pts[k + 4] - pts[k + 1], l2 = ux * ux + uy * uy;
          if (!(l2 > 0)) continue;
          const f = Math.max(0, Math.min(1, ((lx - pts[k]) * ux + (ly - pts[k + 1]) * uy) / l2));
          const fx = pts[k] + ux * f, fy = pts[k + 1] + uy * f, d = Math.hypot(lx - fx, ly - fy);
          if (!best || d < best.d) best = { d, fx, fy };
        }
      }
      if (!best) return NaN;
      const a = tw.map.project(tw.toLngLat(best.fx, best.fy)), b = tw.map.project(tw.toLngLat(lx, ly));
      return (b.y < a.y ? 1 : -1) * best.d / perM(lx, ly);
    };
    // the requests the drawing tool asks previews for
    const asked = [];
    const preview = tw.client.preview.bind(tw.client);
    tw.client.preview = (req, cb) => { asked.push(req); return preview(req, cb); };
    window.__j = { P, ll, offM, asked, perM };
  }, [RC]);

  // the double main, west to east
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
  const jump = async (p, zoom) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__j.ll(window.__j.P(p[0], p[1])), zoom: z }), [p, zoom]); await settle(); };
  // the screen point of pack offset p, page px
  const at = (p) => page.evaluate((p) => { const q = window.tw.map.project(window.__j.ll(window.__j.P(p[0], p[1]))); return [q.x, q.y]; }, p).then(([x, y]) => [box.x + x, box.y + y]);
  const hover = () => page.evaluate(() => {
    const h = window.tw.tools.hover;
    return h ? { kind: h.kind, join: h.join ? window.__j.offM(h.join[0], h.join[1]) : null } : null;
  });
  // the newest ghost: its join answer, first and last points' offsets, flat badges, chevrons
  const ghost = () => page.evaluate(() => {
    const t = window.tw.tools, a = t.lastPreview, s = window.tw.renderer.previewSrc;
    if (!a || !s || !s.count) return null;
    const { offM, asked } = window.__j;
    const marks = a.marks ? [...a.marks] : [];
    let flat = 0;
    for (let k = 2; k < marks.length; k += 3) if (marks[k] === 1) flat++;
    const req = asked[asked.length - 1];
    return { ok: a.ok, joins: a.joins ? [...a.joins] : null, reqJoin: req?.join ?? null, start: offM(s.seg[0], s.seg[1]), end: offM(s.seg[(s.count - 1) * 4 + 2], s.seg[(s.count - 1) * 4 + 3]), flat, chevrons: window.tw.renderer.oneWayGhost.size };
  });
  const settleGhost = async (want) => {
    const t0 = Date.now();
    let g = null;
    while (Date.now() - t0 < 15000) {
      g = await ghost();
      if (g && g.joins && want(g)) return g;
      await wait(150);
    }
    return g;
  };
  // the single piece built last with an end at the main, its join, its drawn end there and that node's badge
  const built = (before) => page.evaluate(async (before) => {
    const tw = window.tw, { offM } = window.__j;
    const t0 = Date.now();
    for (;;) {
      const w = tw.world.value, n = tw.renderer.net;
      const mains = new Set(w.edges.filter((e) => e.tracks === 2).flatMap((e) => [e.a, e.b]));
      const e = w.edges.find((x) => x.tracks === 1 && !before.includes(x.id) && (mains.has(x.a) || mains.has(x.b)));
      const tr = n?.track;
      if (e && tr) {
        const node = w.nodes.get(mains.has(e.a) ? e.a : e.b);
        const [nx, ny] = tw.toLocal(node.lng, node.lat);
        let best = null;
        for (let i = 0; i < tr.count; i++) {
          if (tr.edge[i] !== e.id) continue;
          for (const k of [0, 2]) {
            const x = tr.seg[i * 4 + k], y = tr.seg[i * 4 + k + 1], d = Math.hypot(x - nx, y - ny);
            if (!best || d < best.d) best = { d, x, y };
          }
        }
        if (best) {
          const end = mains.has(e.a) ? 0 : 1;
          return { id: e.id, join: end === 0 ? (e.join ?? 0) & 3 : (e.join ?? 0) >> 2, way: e.oneWay, drawn: offM(best.x, best.y), crosses: !!tw.client.lastNodes.find((x) => x.id === node.id)?.crosses, node: node.id, far: mains.has(e.a) ? e.b : e.a };
        }
      }
      if (Date.now() - t0 > 15000) return null;
      await new Promise((r) => setTimeout(r, 150));
    }
  }, before);
  const singles = () => page.evaluate(() => window.tw.world.value.edges.filter((e) => e.tracks === 1).map((e) => e.id));
  const near2 = (v, sign) => Math.abs(v - 2 * sign) < 0.35;

  // ---- zoomed out: no join
  {
    await jump([-900, 0], FAR - 1);
    const [x, y] = await at([-900, 0]);
    await page.mouse.move(x, y - 2, { steps: 2 });
    await wait(300);
    const h = await hover();
    check(h?.kind === 'edge' && h.join === null, 'zoomed out (double drawn as one stroke): the snap picks no track', JSON.stringify(h));
  }

  // ---- branching: A off the upper track, B off the lower track, both heading north
  const branch = async (name, p, side, to, wantFlat) => {
    const before = await singles();
    await jump(p, NEAR);
    const [x, y] = await at(p);
    await page.mouse.move(x - 30, y - 4 * side, { steps: 2 });
    await page.mouse.move(x, y - 4 * side, { steps: 2 });
    await wait(300);
    const h = await hover();
    check(h?.kind === 'edge' && near2(h.join, side), `${name}: the snap's join point is on the ${side > 0 ? 'upper' : 'lower'} track, 2 m off the centre line`, JSON.stringify(h));
    await page.mouse.click(x, y - 4 * side);
    await wait(400);
    await jump(p, FAR);
    const [ex, ey] = await at(to);
    await page.mouse.move(ex - 20, ey, { steps: 2 });
    await page.mouse.move(ex, ey, { steps: 2 });
    const g = await settleGhost((g) => g.joins[0] !== 0);
    check(g?.ok && Array.isArray(g.reqJoin) && g.reqJoin[0] !== null && g.reqJoin[2] === null, `${name}: the preview asks with the start's join point`, JSON.stringify(g?.reqJoin));
    check(g?.joins?.[0] > 0 && near2(g.start, side), `${name}: the ghost starts on that track`, `joins ${JSON.stringify(g?.joins)}, start ${g?.start?.toFixed(2)} m`);
    check(g?.flat === (wantFlat ? 1 : 0), `${name}: ${wantFlat ? 'a flat badge (the move crosses the other track)' : 'no flat badge'}`, `${g?.flat}`);
    check(g?.chevrons > 0, `${name}: chevrons on the ghost (a stub runs one way)`, `${g?.chevrons}`);
    await jump(p, NEAR);
    await shot(`${name}-ghost`);
    await jump(p, FAR);
    await page.mouse.move(ex - 10, ey, { steps: 2 });
    await page.mouse.move(ex, ey, { steps: 2 });
    await settleGhost(() => true);
    await page.mouse.click(ex, ey);
    const b = await built(before);
    check(b && b.join === g?.joins?.[0] && near2(b.drawn, side), `${name}: built, joined to that track and drawn from it`, JSON.stringify(b));
    check(b && b.crosses === wantFlat, `${name}: the junction's badge ${wantFlat ? 'shows' : 'does not show'} once built`, `${b?.crosses}`);
    await jump(p, NEAR);
    await shot(`${name}-built`);
    return b;
  };
  // ---- merging: C onto the lower track, D onto the upper track, both from the south
  const merge = async (name, from, p, side, wantFlat) => {
    const before = await singles();
    await jump(from, FAR);
    const [sx, sy] = await at(from);
    await page.mouse.move(sx, sy, { steps: 2 });
    await page.mouse.click(sx, sy);
    await wait(400);
    await jump(p, NEAR);
    const [x, y] = await at(p);
    await page.mouse.move(x - 30, y - 4 * side, { steps: 2 });
    await page.mouse.move(x, y - 4 * side, { steps: 2 });
    await wait(300);
    const h = await hover();
    check(h?.kind === 'edge' && near2(h.join, side), `${name}: the snap's join point is on the ${side > 0 ? 'upper' : 'lower'} track`, JSON.stringify(h));
    const g = await settleGhost((g) => g.joins[1] !== 0);
    check(g?.ok && Array.isArray(g.reqJoin) && g.reqJoin[0] === null && g.reqJoin[2] !== null, `${name}: the preview asks with the end's join point`, JSON.stringify(g?.reqJoin));
    check(g?.joins?.[1] > 0 && near2(g.end, side), `${name}: the ghost ends on that track`, `joins ${JSON.stringify(g?.joins)}, end ${g?.end?.toFixed(2)} m`);
    check(g?.flat === (wantFlat ? 1 : 0), `${name}: ${wantFlat ? 'a flat badge' : 'no flat badge'}`, `${g?.flat}`);
    await shot(`${name}-ghost`);
    await page.mouse.click(x, y - 4 * side);
    const b = await built(before);
    check(b && b.join === g?.joins?.[1] && near2(b.drawn, side), `${name}: built, joined to that track and drawn onto it`, JSON.stringify(b));
    check(b && b.crosses === wantFlat, `${name}: the junction's badge ${wantFlat ? 'shows' : 'does not show'} once built`, `${b?.crosses}`);
    await shot(`${name}-built`);
    return b;
  };

  const A = await branch('A-upper', [-900, 0], 1, [-600, 250], false);
  await branch('B-lower', [-1000, 0], -1, [-1300, 250], true);
  await merge('C-lower', [700, -250], [900, 0], -1, false);
  await merge('D-upper', [1300, -250], [1000, 0], 1, true);

  // ---- E: a station at A's far end makes it a terminus branch: two-way, its junction's move back crosses
  if (A) {
    await page.evaluate(async (n) => { await window.tw.client.edit({ op: 'addStation', at: { kind: 'node', node: n }, platform: 100, name: '' }); }, A.far);
    await wait(1000);
    const e = await page.evaluate((a) => { const w = window.tw.world.value, s = w.edges.find((x) => x.tracks === 1 && (x.a === a.node || x.b === a.node)); return { way: s ? s.oneWay ?? 0 : null, crosses: !!window.tw.client.lastNodes.find((x) => x.id === a.node)?.crosses }; }, A);
    check(e.way === 0 && e.crosses, 'E: with a station at its end the branch runs both ways and its junction shows the badge', JSON.stringify(e));
  }
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).tool.value = 'select'; });
  await page.keyboard.press('Escape');

  // ---- F: a flyover laid by hand from two single tracks (`RouteAsk.join` as the tool sends it): a double piece
  // Q-R east of x 3600, north of the main; leg 1 off the main's upper track at x 1700, north and down to Q; leg 2 off its
  // lower track at x 2000, south of it and then over it on a ramp to Q, arriving south of leg 1
  const F = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P } = window.__j;
    const node = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const mainAt = (x) => {
      const w = tw.world.value, y0 = P(0, 0).y, X = P(x, 0).x;
      return w.edges.find((e) => { const a = w.nodes.get(e.a), b = w.nodes.get(e.b); return e.tracks === 2 && Math.abs(a.y - y0) < 1 && Math.abs(b.y - y0) < 1 && Math.min(a.x, b.x) < X && X < Math.max(a.x, b.x); })?.id;
    };
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const res = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...P(3600, 400), level: 0 }, to: { kind: 'free', ...P(4400, 400), level: 0 }, pis: [], single: false, design: 60 }));
    await pause(400);
    const Q = node(P(3600, 400)), before = new Set(tw.world.value.edges.map((e) => e.id));
    const j1 = P(1700, 2), j2 = P(2000, -2);
    res.push(await cl.edit({ op: 'route', from: { kind: 'edge', edge: mainAt(1700), ...P(1700, 0) }, to: { kind: 'node', node: Q }, pis: [{ ...P(2600, 600), radius: 0, level: 0 }], single: true, through: true, design: 60, join: [j1.x, j1.y, null, null] }));
    await pause(400);
    res.push(await cl.edit({ op: 'route', from: { kind: 'edge', edge: mainAt(2000), ...P(2000, 0) }, to: { kind: 'node', node: Q }, pis: [{ ...P(2400, -200), radius: 0, level: 0 }], single: true, through: true, design: 60, ramp: true, join: [j2.x, j2.y, null, null] }));
    await pause(800);
    const w = tw.world.value, legs = w.edges.filter((e) => e.tracks === 1 && !before.has(e.id));
    const junctions = [node(P(1700, 0)), node(P(2000, 0)), Q];
    return {
      refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues)),
      legs: legs.map((e) => ({ id: e.id, way: e.oneWay ?? 0, join: e.join ?? 0 })),
      badges: junctions.map((n) => !!cl.lastNodes.find((x) => x.id === n)?.crosses),
      crossings: ((m) => { let c = 0; for (let k = 2; m && k < m.length; k += 6) if (m[k] === 0 || m[k] === 4) c++; return c; })(tw.renderer.net?.marks),
    };
  });
  check(!F.refused.length && F.legs.length >= 2, 'F: a flyover laid by hand from two single tracks', F.refused.join(' | '));
  check(F.legs.every((l) => l.way !== 0), 'F: every piece of both legs runs one way', JSON.stringify(F.legs));
  check(F.badges.every((b) => !b) && F.crossings === 0, 'F: no flat badge at either junction on the main or at Q, no flat crossing', `${JSON.stringify(F.badges)}, ${F.crossings} crossings`);
  await jump([1850, 0], 17.6);
  await shot('flyover-hand');

  await jump([-950, 60], 17.6);
  await shot('branches');
  await jump([950, -60], 17.6);
  await shot('merges');
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3137 C3 FAILED: ${fail}` : 'T-3137 C3 PASSED');
process.exit(fail ? 1 : 0);
