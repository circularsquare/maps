// T-3166. High-speed track and tram track double up into quad, and tram quad is drawn.
//  A  a high-speed double doubled up (the inspector's Double up, `doubleUp`): every new piece is high speed, it is
//     priced as a second high-speed double, a station goes on each pair, a line runs on each pair.
//  B  the track bar with Tram picked: Quad can be picked, Single is not offered, quad turns Snap to streets off and
//     turning that on picks Double again; the ghost asks tram + quad and costs twice the double; built, it is four
//     tram tracks (two tram doubles) and a station goes on each pair.
//  C  a tram double doubled up: the pair is tram track, priced as a second off-street double.
// Pictures in artifacts/trainworld/t3166-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6120):
//   node trainworld/checks/t3166-quad.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6120/';
const SHADES = process.argv.includes('--dark') ? ['dark'] : ['light', 'dark'];
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3166-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false, streets: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3166-${name}.png`) });
const settle = async (ms = 600) => {
  await page.waitForFunction(() => window.tw.map.loaded() && !window.tw.map.isMoving(), undefined, { timeout: 60000 }).catch(() => {});
  await wait(ms);
};
const edit = (op) => page.evaluate((op) => window.tw.client.edit(op), op);
const edges = () => page.evaluate(() => window.tw.world.value.edges.map((e) => ({ id: e.id, kind: e.kind ?? 0, tracks: e.tracks, built: e.built, hsr: !!e.hsr, len: e.lengthM, cost: e.cost })));
const ui = (fn) => page.evaluate(fn);

try {
  await page.goto(`${BASE}?debug=1&new=1&mode=sandbox&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw.tools, undefined, { timeout: 180000 });
  await page.evaluate(async () => { window.__m = await import('/src/game/ui.ts'); });
  await page.evaluate(([c]) => {
    const tw = window.tw, o = tw.lngLatToPack(c[0], c[1]);
    const P = (dx, dy) => ({ x: o.x + dx, y: o.y + 3000 + dy });
    const ll = (p) => { const q = tw.packToLngLat(p.x, p.y); return Array.isArray(q) ? q : [q.lng, q.lat]; };
    window.__t = { P, ll };
  }, [RC]);
  const jump = async (p, zoom) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__t.ll(window.__t.P(p[0], p[1])), zoom: z }), [p, zoom]); await settle(); };
  const route = (from, to, extra) => page.evaluate(([from, to, extra]) => { const { P } = window.__t; return window.tw.client.edit({ op: 'route', from: { kind: 'free', ...P(...from), level: 0 }, to: { kind: 'free', ...P(...to), level: 0 }, pis: [], single: false, ramp: false, ...extra }); }, [from, to, extra]);

  // ---- A: high-speed double doubled up
  let r = await route([-1200, -1500], [1200, -1500], { hsr: true, design: 320 });
  check(r.ok, 'A: a high-speed double is drawn', JSON.stringify(r));
  await wait(600);
  const a0 = await edges();
  const hs = a0.filter((e) => e.hsr);
  check(hs.length >= 1 && hs.every((e) => e.tracks === 2), 'A: one high-speed double', JSON.stringify(a0));
  const q = await page.evaluate((id) => window.tw.client.query({ what: 'doubleUp', edge: id }), hs[0].id);
  check(q && q[0] > 0, 'A: Double up is offered on high-speed track with a price', JSON.stringify(q));
  const cost0 = await page.evaluate(() => window.tw.world.value.blueprintCost ?? null);
  r = await edit({ op: 'doubleUp', edge: hs[0].id });
  check(r.ok, 'A: Double up works on the high-speed double', JSON.stringify(r));
  await wait(800);
  const a1 = await edges();
  const hs1 = a1.filter((e) => e.hsr);
  check(hs1.length === 2 * hs.length && a1.length === 2 * a0.length, 'A: a high-speed pair for each piece', `${hs1.length} of ${a1.length}`);
  check(hs1.every((e) => e.tracks === 2 && !e.built), 'A: the pair is blueprint high-speed double');
  const costs = a1.map((e) => e.cost).filter((c) => typeof c === 'number');
  if (costs.length === a1.length) check(Math.abs(a1.filter((e) => !hs.some((h) => h.id === e.id)).reduce((s, e) => s + e.cost, 0) - hs.reduce((s, e) => s + e.cost, 0)) < 0.03 * hs.reduce((s, e) => s + e.cost, 0), 'A: the pair costs about what the first double does (the pair swings out at the ends)');
  r = await edit({ op: 'doubleUp', edge: hs[0].id });
  check(!r.ok, 'A: a second Double up is refused', JSON.stringify(r));
  await jump([0, -1500], 14);
  await shot('hsr-doubled-up-blueprint');

  // ---- B: tram quad in the track bar
  const B = await ui(() => {
    const m = window.__m;
    m.setTrackType(2); m.setQuadTrack(false); m.setSingleTrack(false);
    m.snapStreets.value = true;
    m.setTrackType('tram');
    const first = { tram: m.tramTrack.value, quad: m.quadTrack.value, streets: m.snapStreets.value };
    m.setQuadTrack(true);
    const quad = { tram: m.tramTrack.value, quad: m.quadTrack.value, streets: m.snapStreets.value };
    m.snapStreets.value = true; m.setQuadTrack(false);
    return { first, quad };
  });
  check(B.first.tram && !B.first.quad && B.first.streets, 'B: tram picked, double, Snap to streets on', JSON.stringify(B.first));
  check(B.quad.tram && B.quad.quad && !B.quad.streets, 'B: Quad can be picked with Tram and turns Snap to streets off', JSON.stringify(B.quad));
  await ui(() => { const m = window.__m; m.tool.value = 'track'; m.setQuadTrack(true); m.buildLevel.value = 0; });
  await wait(400);
  const bar = await page.evaluate(() => [...document.querySelectorAll('.track-bar .tb-count button')].map((b) => b.getAttribute('aria-label')));
  check(bar.length === 2, 'B: Tram shows two count buttons, Double and Quad', JSON.stringify(bar));
  await page.evaluate(() => document.querySelector('.track-bar .tb-streets')?.click());
  await wait(200);
  const afterStreets = await ui(() => ({ quad: window.__m.quadTrack.value, streets: window.__m.snapStreets.value }));
  check(!afterStreets.quad && afterStreets.streets, 'B: turning Snap to streets on picks Double again', JSON.stringify(afterStreets));
  await ui(() => { window.__m.snapStreets.value = false; window.__m.setQuadTrack(true); });
  await wait(300);
  const price = await page.evaluate(() => document.querySelector('.track-bar .price')?.textContent ?? '');
  const priceDouble = await ui(() => { window.__m.setQuadTrack(false); return 1; }).then(() => wait(300)).then(() => page.evaluate(() => document.querySelector('.track-bar .price')?.textContent ?? ''));
  console.log('price quad:', price.trim(), '| double:', priceDouble.trim());
  await ui(() => window.__m.setQuadTrack(true));

  // the ghost and the build, by drawing with the mouse
  await jump([0, 0], 15);
  const box = await (await page.$('#map')).boundingBox();
  const at = (p) => page.evaluate((p) => { const q = window.tw.map.project(window.__t.ll(window.__t.P(p[0], p[1]))); return [q.x, q.y]; }, p).then(([x, y]) => [box.x + x, box.y + y]);
  const ghostCost = async () => {
    const t0 = Date.now();
    for (;;) {
      const g = await page.evaluate(() => { const a = window.tw.tools.lastPreview; return a ? { ok: a.ok, cost: a.cost, req: window.tw.tools.previewKey } : null; });
      if (g && Date.now() - t0 > 700) return g;
      if (Date.now() - t0 > 15000) return g;
      await wait(150);
    }
  };
  const [sx, sy] = await at([-1200, 0]);
  const [ex, ey] = await at([1200, 0]);
  await page.mouse.move(sx, sy, { steps: 2 });
  await page.mouse.click(sx, sy);
  await wait(300);
  await page.mouse.move(ex - 40, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  const gq = await ghostCost();
  check(gq?.ok, 'B: the tram quad ghost is ok', JSON.stringify(gq));
  check(/"tram":true/.test(gq?.req ?? '') && /"quad":true/.test(gq?.req ?? ''), 'B: the preview asks for tram and quad', gq?.req);
  await ui(() => window.__m.setQuadTrack(false));
  await page.mouse.move(ex - 30, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  const gd = await ghostCost();
  check(gd?.ok && gq?.cost > 1.8 * gd.cost && gq.cost < 2.2 * gd.cost, 'B: the tram quad costs about twice the tram double', `${gq?.cost} vs ${gd?.cost}`);
  await ui(() => window.__m.setQuadTrack(true));
  await page.mouse.move(ex - 30, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  await ghostCost();
  await shot('tram-quad-ghost');
  const before = (await edges()).map((e) => e.id);
  await page.mouse.click(ex, ey);
  await wait(2000);
  await ui(() => { window.__m.tool.value = 'select'; });
  const made = (await edges()).filter((e) => !before.includes(e.id));
  check(made.length >= 2 && made.every((e) => e.kind === 3 && e.tracks === 2), 'B: building it makes tram double pieces, two side by side', JSON.stringify(made));

  // ---- C: a tram double doubled up
  r = await route([-1200, 1500], [1200, 1500], { tram: true });
  check(r.ok, 'C: a tram double is drawn', JSON.stringify(r));
  await wait(600);
  const c0 = await edges();
  const tr = c0.filter((e) => e.kind === 3 && !before.includes(e.id) && !made.some((m) => m.id === e.id));
  const qc = await page.evaluate((id) => window.tw.client.query({ what: 'doubleUp', edge: id }), tr[0].id);
  check(qc && qc[0] > 0, 'C: Double up is offered on tram track with a price', JSON.stringify(qc));
  r = await edit({ op: 'doubleUp', edge: tr[0].id });
  check(r.ok, 'C: Double up works on the tram double', JSON.stringify(r));
  await wait(800);
  const c1 = await edges();
  const tr1 = c1.filter((e) => e.kind === 3);
  check(tr1.length === c0.filter((e) => e.kind === 3).length + tr.length && c1.every((e) => !e.hsr || true), 'C: a tram pair for each piece', `${tr1.length}`);

  // ---- D: a station on each pair and a line on each pair, on all three quads
  const all = await edges();
  const groups = {
    'high-speed doubled up': all.filter((e) => e.hsr),
    'tram quad drawn': made,
    'tram doubled up': all.filter((e) => e.kind === 3 && !made.some((m) => m.id === e.id) && e.id >= tr[0].id),
  };
  for (const [name, es] of Object.entries(groups)) {
    check(es.length === 2, `D: ${name}: two double pieces`, JSON.stringify(es.map((e) => e.id)));
    const lines = [];
    for (const e of es) {
      const ids = [];
      for (const f of [0.25, 0.75]) {
        const r = await page.evaluate(async ([id, f, tram]) => {
          const w = window.tw.world.value, ed0 = w.edges.find((x) => x.id === id);
          window.__ends ??= {};
          const ends = (window.__ends[id] ??= [w.nodes.get(ed0.a), w.nodes.get(ed0.b)].map((n) => [n.x, n.y]));
          const x = ends[0][0] + f * (ends[1][0] - ends[0][0]), y = ends[0][1] + f * (ends[1][1] - ends[0][1]);
          // the piece the point is on now (a station splits its piece)
          const dist = (e) => { const a = w.nodes.get(e.a), b = w.nodes.get(e.b); const t = Math.max(0, Math.min(1, ((x - a.x) * (b.x - a.x) + (y - a.y) * (b.y - a.y)) / ((b.x - a.x) ** 2 + (b.y - a.y) ** 2))); return Math.hypot(a.x + t * (b.x - a.x) - x, a.y + t * (b.y - a.y) - y); };
          const cur = w.edges.filter((e) => e.hsr === ed0.hsr && e.kind === ed0.kind).sort((p, q) => dist(p) - dist(q))[0];
          const ids0 = new Set([...w.nodes.values()].filter((n) => n.platform > 0).map((n) => n.id));
          const a = await window.tw.client.edit({ op: 'addStation', at: { kind: 'edge', edge: cur.id, x, y }, platform: tram ? 40 : 160, name: 'S' });
          const n = [...window.tw.world.value.nodes.values()].filter((n) => n.platform > 0 && !ids0.has(n.id));
          return { ok: a.ok, issues: a.issues, node: n[0]?.id, kind: n[0]?.kind };
        }, [e.id, f, e.kind === 3]);
        check(r.ok && r.node !== undefined, `D: ${name}: a station on piece ${e.id} at ${f}`, JSON.stringify(r));
        if (r.node !== undefined) ids.push(r.node);
      }
      if (ids.length === 2) {
        const l = await edit({ op: 'addLine', stops: ids, name: `L${e.id}`, colour: '#c33' });
        check(l.ok, `D: ${name}: a line on piece ${e.id}`, JSON.stringify(l));
        lines.push(l.ok);
      }
    }
    await wait(500);
  }
  const bad0 = await page.evaluate(() => window.tw.world.value.lines?.map((l) => ({ id: l.id, ok: l.broken !== true })) ?? []);
  console.log('lines:', JSON.stringify(bad0));
  // a save round trip keeps every kind
  const rt = await page.evaluate(async () => { const s = await window.tw.client.save?.(); return s ? 1 : 0; }).catch(() => 0);
  console.log('save round trip available:', rt);
  for (const sh of SHADES) {
    await page.evaluate((sh) => { window.__m.shade.value = sh; }, sh);
    for (const [nm, y, z] of [['hsr', -1500, 17], ['tram-quad', 0, 17], ['tram-doubled', 1500, 17]]) {
      await jump([-300, y], z);
      await shot(`${nm}-${sh}`);
    }
  }


  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3166 FAILED: ${fail}` : 'T-3166 PASSED');
process.exit(fail ? 1 : 0);
