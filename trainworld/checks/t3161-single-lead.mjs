// T-3161: single track that runs both ways where it joins one track of double track draws the link the track model
// gives it to the other track: a short crossover along the double, from the joined track at the junction to the
// other track (a single-lead junction), where the flat-junction badge is. On a new game: two double tracks crossing
// (H east-west, V north-south), a single chord from H east of the crossing to V north of it.
//   automatic (one way): no crossover, no badge; pinned both ways: a crossover and a badge at each end, the
//   crossover starting on the track the end joins; Reverse (one way, joined to the far tracks): none again.
// Pictures in artifacts/trainworld/t3161-lead-*.png.
// From the maps root, on a Vite of your own (node trainworld/checks/dev-nowatch.mjs 6114):
//   node trainworld/checks/t3161-single-lead.mjs [dev url]
// About a minute.
import { join } from 'node:path';
import { ARTIFACTS, collectErrors, launchChrome } from './lib.mjs';

const BASE = process.argv.slice(2).find((a) => /^https?:/.test(a)) ?? 'http://localhost:6114/';
let fail = 0;
const check = (ok, what, detail = '') => { console.log(`${ok ? 'PASS' : 'FAIL'} ${what}${detail ? `  (${detail})` : ''}`); if (!ok) fail++; };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const RC = [-73.93, 40.675];

const chrome = await launchChrome({ prefix: 'tw-t3161-', size: [1400, 900], deviceScaleFactor: 1 });
const page = await chrome.ctx.newPage();
await page.addInitScript(() => { try { localStorage.setItem('trainworld-ui', JSON.stringify({ shade: 'light', v: 7, build: { level: 0, keep: false } })); } catch {} });
const errors = [];
collectErrors(page, errors);
const shot = (name) => page.screenshot({ path: join(ARTIFACTS, `t3161-lead-${name}.png`) });
const settle = async (ms = 700) => {
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
    const node = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const d = Math.hypot(n.x - p.x, n.y - p.y); if (d < bd) { bd = d; best = n.id; } } return best; };
    const edgeAt = (p) => {
      const w = tw.world.value;
      let best = null, bd = 1e12;
      for (const e of w.edges) {
        if (e.tracks !== 2) continue;
        const a = w.nodes.get(e.a), b = w.nodes.get(e.b), ux = b.x - a.x, uy = b.y - a.y;
        const f = Math.max(0, Math.min(1, ((p.x - a.x) * ux + (p.y - a.y) * uy) / (ux * ux + uy * uy)));
        const d = Math.hypot(p.x - a.x - ux * f, p.y - a.y - uy * f);
        if (d < bd) { bd = d; best = e.id; }
      }
      return best;
    };
    window.__l = { P, ll, node, edgeAt };
  }, [RC]);

  const net = await page.evaluate(async () => {
    const tw = window.tw, cl = tw.client, { P, node, edgeAt } = window.__l;
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const res = [];
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...P(-3000, 0), level: 0 }, to: { kind: 'free', ...P(3000, 0), level: 0 }, pis: [], single: false }));
    await pause(400);
    res.push(await cl.edit({ op: 'route', from: { kind: 'free', ...P(0, -3000), level: 0 }, to: { kind: 'free', ...P(0, 3000), level: 0 }, pis: [], single: false }));
    await pause(400);
    const before = new Set(tw.world.value.edges.map((e) => e.id));
    res.push(await cl.edit({ op: 'route', from: { kind: 'edge', edge: edgeAt(P(800, 0)), ...P(800, 0) }, to: { kind: 'edge', edge: edgeAt(P(0, 800)), ...P(0, 800) }, pis: [], single: true, design: 60 }));
    await pause(400);
    res.push(await cl.edit({ op: 'constructAll' }));
    await pause(800);
    const chord = tw.world.value.edges.filter((e) => e.tracks === 1 && !before.has(e.id)).map((e) => e.id);
    return { refused: res.filter((r) => !r.ok).map((r) => JSON.stringify(r.issues)), chord, H: node(P(800, 0)), V: node(P(0, 800)) };
  });
  check(!net.refused.length && net.chord.length === 1, 'two crossing doubles and a single chord between them', JSON.stringify(net));
  const chord = net.chord[0];

  // the drawn crossovers (`turnbacks`: no line runs, so all are T-3161's): per run of strokes, its double and
  // its lane at the start and the end (0 one track, 255 the other)
  const state = () => page.evaluate(([chord, H, V]) => {
    const tw = window.tw, w = tw.world.value, e = w.edges.find((x) => x.id === chord), t = tw.renderer.net?.turnbacks;
    const badge = (n) => !!tw.client.lastNodes.find((q) => q.id === n)?.crosses;
    const runs = [];
    if (t) for (let i = 0; i < t.count; i++) {
      const first = i === 0 || t.seg[i * 4] !== t.seg[i * 4 - 2] || t.seg[i * 4 + 1] !== t.seg[i * 4 - 1];
      if (first) runs.push({ edge: t.edge[i], lane0: t.lane ? t.lane[i * 2] : null, lane1: null });
      if (runs.length && t.lane) runs[runs.length - 1].lane1 = t.lane[i * 2 + 1];
    }
    return { way: e.oneWay ?? 0, join: e.join ?? 0, badgeH: badge(H), badgeV: badge(V), runs };
  }, [chord, net.H, net.V]);
  const jump = async (p, z) => { await page.evaluate(([p, z]) => window.tw.map.jumpTo({ center: window.__l.ll(window.__l.P(p[0], p[1])), zoom: z }), [p, z]); await settle(); };
  const select = (id) => page.evaluate(async (id) => { (await import('/src/game/ui.ts')).selection.value = id === null ? null : { kind: 'track', edge: id }; }, id);
  await page.waitForFunction(() => !document.body.innerText.includes('Loading city data'), undefined, { timeout: 120000 }).catch(() => {});

  let s = await state();
  check(s.way !== 0 && !s.badgeH && !s.badgeV && s.runs.length === 0, 'automatic: one way, no badge, no crossover', JSON.stringify(s));

  await select(chord);
  await wait(600);
  await page.locator('label.check').first().click();
  await wait(1200);
  await select(null);
  await page.keyboard.press('Escape');
  s = await state();
  check(s.way === 0 && s.badgeH && s.badgeV, 'pinned both ways: a badge at both ends', JSON.stringify(s));
  check(s.runs.length === 2 && s.runs.every((r) => r.lane0 !== null && r.lane0 !== r.lane1 && (r.lane0 === 0 || r.lane0 === 255)), 'a crossover at each end, track to track', JSON.stringify(s.runs));
  for (const [name, p] of [['H', [800, 0]], ['V', [0, 800]]]) {
    await jump(p, 18.4);
    await shot(`both-${name}`);
  }
  await jump([420, 470], 15.8);
  await shot('both-overview');

  // Reverse: one way the other way, joined to the far tracks: no crossover
  await select(chord);
  await wait(600);
  await page.locator('label.check').first().click();
  await wait(1200);
  await page.getByText('Reverse direction', { exact: true }).click();
  await wait(1200);
  await select(null);
  s = await state();
  check(s.way !== 0 && s.runs.length === 0, 'reversed, one way: no crossover', JSON.stringify(s));

  const bad = errors.filter((e) => !/favicon|404/.test(String(e)));
  check(bad.length === 0, 'no console errors', bad.slice(0, 3).join(' | '));
} catch (e) {
  await shot('died').catch(() => {});
  check(false, 'the check ran to the end', e.stack ?? e.message);
} finally {
  await chrome.close();
}
console.log(fail ? `T-3161 FAILED: ${fail}` : 'T-3161 PASSED');
process.exit(fail ? 1 : 0);
