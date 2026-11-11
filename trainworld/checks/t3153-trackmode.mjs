// T-3153 and T-3126. Delete mode (and the select tool's picking) does not read the track tool's settings, and
// high-speed track can be quad.
//  A  a tram blueprint and a regular one on a new game; with each track-tool setting set (regular, high speed,
//     high speed quad, single, quad, tram, a low target speed, ramp off) and the delete tool in hand, hovering each
//     picks that piece (`Tools.delHover`), and a click removes the tram one with high speed set.
//  B  the track tool with high speed and quad: both signals stay on, the ghost is ok and asks hsr + quad, its price is
//     twice the double's, building it makes four high-speed pieces' worth (two double high-speed edges side by side).
// Pictures in artifacts/trainworld/t3153-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6204):
//   node trainworld/checks/t3153-trackmode.mjs [dev url]
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6204/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];
const ZOOM = 15;

const chrome = await launchChrome({ prefix: 'tw-t3153-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3153-${name}.png`) });
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
    window.__t = { P, ll };
  }, [RC]);

  // two blueprints: tram track west, regular east
  await page.evaluate(async () => {
    const { P } = window.__t, cl = window.tw.client;
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(-1500, 0), level: 0 }, to: { kind: 'free', ...P(-300, 0), level: 0 }, pis: [], single: false, tram: true, ramp: false });
    await cl.edit({ op: 'route', from: { kind: 'free', ...P(300, 0), level: 0 }, to: { kind: 'free', ...P(1500, 0), level: 0 }, pis: [], single: false });
  });
  await wait(1000);
  const edges = await page.evaluate(() => window.tw.world.value.edges.map((e) => ({ id: e.id, kind: e.kind ?? 0, tracks: e.tracks, built: e.built, hsr: !!e.hsr })));
  const tram = edges.find((e) => e.kind >= 2), reg = edges.find((e) => !e.kind && !e.hsr);
  check(!!tram && !!reg, 'A: a tram blueprint and a regular one exist', JSON.stringify(edges));

  const box = await (await page.$('#map')).boundingBox();
  const jump = async (p, zoom) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__t.ll(window.__t.P(p[0], p[1])), zoom: z }), [p, zoom]); await settle(); };
  const at = (p) => page.evaluate((p) => { const q = window.tw.map.project(window.__t.ll(window.__t.P(p[0], p[1]))); return [q.x, q.y]; }, p).then(([x, y]) => [box.x + x, box.y + y]);
  const picked = () => page.evaluate(() => { const d = window.tw.tools.delHover; return d ? { kind: d.kind, edge: d.edge } : null; });
  const setTool = (t) => page.evaluate(async (t) => { (await import('/src/game/ui.ts')).tool.value = t; }, t);
  const settings = {
    regular: { type: 2, quad: false, single: false },
    'high speed': { type: 'hsr', quad: false, single: false },
    'high speed quad': { type: 'hsr', quad: true, single: false },
    single: { type: 2, quad: false, single: true },
    quad: { type: 2, quad: true, single: false },
    tram: { type: 'tram', quad: false, single: false },
    'target 20': { type: 2, quad: false, single: false, design: 20 },
    'ramp off': { type: 2, quad: false, single: false, ramp: false },
  };
  const apply = (s) => page.evaluate(async (s) => {
    const ui = await import('/src/game/ui.ts');
    ui.tool.value = 'track';
    ui.setSingleTrack(false); ui.setQuadTrack(false);
    ui.setTrackType(s.type);
    if (s.quad) ui.setQuadTrack(true);
    if (s.single) ui.setSingleTrack(true);
    ui.designSpeed.value = s.design ?? (s.type === 'hsr' ? 320 : 160);
    ui.autoRamp.value = s.ramp ?? true;
  }, s);

  await jump([0, 0], ZOOM);
  const [tx, ty] = await at([-700, 0]);
  const [rx, ry] = await at([700, 0]);
  for (const [name, s] of Object.entries(settings)) {
    await apply(s);
    await setTool('delete');
    await page.mouse.move(tx - 20, ty + 30, { steps: 2 });
    await page.mouse.move(tx, ty, { steps: 3 });
    await wait(250);
    const a = await picked();
    await page.mouse.move(rx - 20, ry + 30, { steps: 2 });
    await page.mouse.move(rx, ry, { steps: 3 });
    await wait(250);
    const b = await picked();
    check(a?.kind === 'track' && a.edge === tram.id, `A: ${name} set, delete mode picks the tram blueprint`, JSON.stringify(a));
    check(b?.kind === 'track' && b.edge === reg.id, `A: ${name} set, delete mode picks the regular blueprint`, JSON.stringify(b));
    if (name === 'high speed') await shot('delete-tram-with-high-speed');
  }

  // delete the tram blueprint with high speed picked
  await apply(settings['high speed']);
  await setTool('delete');
  await page.mouse.move(tx - 20, ty + 30, { steps: 2 });
  await page.mouse.move(tx, ty, { steps: 3 });
  await wait(300);
  await page.mouse.click(tx, ty);
  await wait(1200);
  const left = await page.evaluate(() => window.tw.world.value.edges.map((e) => e.id));
  check(!left.includes(tram.id) && left.includes(reg.id), 'A: with high speed picked, a click in delete mode removes the tram blueprint only', JSON.stringify(left));

  // ---- B: high-speed quad
  const B = await page.evaluate(async () => {
    const ui = await import('/src/game/ui.ts');
    ui.setTrackType(2); ui.setQuadTrack(false); ui.setSingleTrack(false);
    ui.setTrackType('hsr'); ui.setQuadTrack(true);
    return { hsr: ui.highSpeed.value, quad: ui.quadTrack.value, single: ui.singleTrack.value };
  });
  check(B.hsr && B.quad && !B.single, 'B: high speed and quad are both on', JSON.stringify(B));
  await setTool('track');
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.buildLevel.value = 0; ui.autoRamp.value = false; });
  await jump([0, 1000], 15);
  const [sx, sy] = await at([-1200, 1000]);
  const [ex, ey] = await at([1200, 1000]);
  const ghostCost = async () => {
    const t0 = Date.now();
    for (;;) {
      const g = await page.evaluate(() => { const a = window.tw.tools.lastPreview; return a ? { ok: a.ok, cost: a.cost, req: window.tw.tools.previewKey } : null; });
      if (g && Date.now() - t0 > 700) return g;
      if (Date.now() - t0 > 15000) return g;
      await wait(150);
    }
  };
  await page.mouse.move(sx, sy, { steps: 2 });
  await page.mouse.click(sx, sy);
  await wait(300);
  await page.mouse.move(ex - 40, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  const gq = await ghostCost();
  check(gq?.ok, 'B: the high-speed quad ghost is ok', JSON.stringify(gq));
  check(/"hsr":true/.test(gq?.req ?? '') && /"quad":true/.test(gq?.req ?? ''), 'B: the preview asks for hsr and quad', gq?.req);
  await shot('hsr-quad-ghost');
  // the same route as double, for the price
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).setQuadTrack(false); });
  await page.mouse.move(ex - 30, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  const gd = await ghostCost();
  check(gd?.ok && gq?.cost > 1.8 * gd.cost && gq.cost < 2.2 * gd.cost, 'B: the high-speed quad costs about twice the high-speed double', `${gq?.cost} vs ${gd?.cost}`);
  await page.evaluate(async () => { (await import('/src/game/ui.ts')).setQuadTrack(true); });
  await page.mouse.move(ex - 30, ey, { steps: 2 });
  await page.mouse.move(ex, ey, { steps: 3 });
  await ghostCost();
  const before = await page.evaluate(() => window.tw.world.value.edges.map((e) => e.id));
  await page.mouse.click(ex, ey);
  await wait(2000);
  const made = await page.evaluate((before) => window.tw.world.value.edges.filter((e) => !before.includes(e.id)).map((e) => ({ id: e.id, tracks: e.tracks, hsr: !!e.hsr })), before);
  check(made.length >= 2 && made.every((e) => e.hsr && e.tracks === 2), 'B: building it makes high-speed double pieces, two side by side', JSON.stringify(made));
  await jump([0, 1000], 17);
  await shot('hsr-quad-built');
  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3153 FAILED: ${fail}` : 'T-3153 PASSED');
process.exit(fail ? 1 : 0);
