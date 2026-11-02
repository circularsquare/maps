// T-1964: a station over a joint between two drawn pieces of track is placed where the pointer is.
//   node trainworld/checks/t1964-station-joints.mjs [url] [label]      (from the maps root; about a minute)
// A fresh game in Brooklyn gets two curving routes drawn click by click as the track tool draws them
// (tangent arcs), one all blueprint and one whose first half is constructed (joints of built beside
// blueprint). With the station tool at zoom 16 the pointer rests on every plain joint and the ghost
// is read: placed (and how far from the pointer) or refused (and why). Then a click on the first
// blueprint joint places a station there. Prints the counts and one line per refusal; writes
// artifacts/trainworld/t1964/<label>-*.png and <label>.json. Fails when a joint between two blueprint
// pieces is refused or moved more than 10 m.
import { mkdir, writeFile } from 'node:fs/promises';
import { launchChrome, GL, ARTIFACTS } from './lib.mjs';

const BASE = process.argv[2] ?? 'http://localhost:6130/';
const LABEL = process.argv[3] ?? 'run';
const OUT = `${ARTIFACTS}t1964/`;
await mkdir(OUT, { recursive: true });
const { ctx, close } = await launchChrome({ prefix: 'tw-t1964-', size: [1200, 800], args: GL });
let failed = false;
try {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', e.message));
  await page.goto(`${BASE}?debug=1&new=1&paused=1&demandWorkers=1&v=${Date.now()}`);
  await page.waitForFunction(() => window.tw?.world?.value && window.tw?.tools, undefined, { timeout: 180000 });
  await page.waitForTimeout(2000);
  const made = await page.evaluate(async () => {
    const tw = window.tw, c = tw.client, P = (lng, lat) => tw.lngLatToPack(lng, lat);
    const pause = (ms) => new Promise((d) => setTimeout(d, ms));
    const nearest = (p) => { let best = null, bd = 1e12; for (const n of tw.world.value.nodes.values()) { const q = P(n.lng, n.lat), d = Math.hypot(q.x - p.x, q.y - p.y); if (d < bd) bd = d, best = n.id; } return best; };
    const out = [];
    const draw = async (lng, lat, heading, turns) => {
      // the drawing tool's request: tangent from the end, design speed 160, auto ramp
      const req = (from, to) => ({ op: 'route', from, to, pis: [], single: false, through: true, design: 160, ramp: true });
      // pack units per metre here
      const start = P(lng, lat), m = Math.hypot(P(lng + 0.01, lat).x - start.x, P(lng + 0.01, lat).y - start.y) / (0.01 * 111320 * Math.cos(lat * Math.PI / 180));
      let at = { ...start }, h = heading, prev = null;
      const edgesBefore = new Set(tw.world.value.edges.map((e) => e.id));
      for (let i = 0; i <= turns.length; i++) {
        const len = 260 + 90 * (i % 3);
        const t = i === 0 ? 0 : turns[i - 1];
        const to = { x: at.x + len * m * Math.cos(h + t), y: at.y + len * m * Math.sin(h + t) };
        const from = prev === null ? { kind: 'free', ...at, level: 0 } : { kind: 'node', node: prev };
        const r = await c.edit(req(from, { kind: 'free', ...to, level: 0 }));
        out.push(r?.ok ?? false);
        await pause(250);
        prev = nearest(to);
        at = to;
        h += 2 * t;
      }
      return tw.world.value.edges.filter((e) => !edgesBefore.has(e.id)).map((e) => e.id);
    };
    const a = await draw(-73.955, 40.646, 0.15, [0.12, -0.18, 0.2, -0.1, 0.16, -0.22, 0.08]);
    const b = await draw(-73.955, 40.676, -0.1, [-0.15, 0.2, -0.12, 0.18, -0.2, 0.1, -0.14]);
    // clicks a hair off straight ahead, within the heading tolerance: pieces meeting at a slight angle
    await draw(-73.955, 40.662, 0.05, [0.0012, -0.0014, 0.1, 0.0013, -0.0012]);
    // the first half of the second route constructed
    out.push((await c.edit({ op: 'construct', edges: b.slice(0, 4), stations: [] }))?.ok ?? false);
    await pause(800);
    return { out, a, b };
  });
  console.log('track edits ok:', JSON.stringify(made.out));
  const joints = await page.evaluate(() => {
    const w = window.tw.world.value, byNode = new Map();
    for (const e of w.edges) for (const n of [e.a, e.b]) (byNode.get(n) ?? byNode.set(n, []).get(n)).push(e);
    const out = [];
    for (const n of w.nodes.values()) {
      if (n.ports !== 2 || n.platform > 0) continue;
      const es = byNode.get(n.id) ?? [];
      if (es.length !== 2) continue;
      out.push({ node: n.id, lng: n.lng, lat: n.lat, built: es.map((e) => e.built).join('|') });
    }
    return out;
  });
  console.log(`plain joints: ${joints.length}`);
  await page.evaluate(async () => { const ui = await import('/src/game/ui.ts'); ui.tool.value = 'station'; });
  const rect = await page.evaluate(() => { const r = window.tw.map.getContainer().getBoundingClientRect(); return { left: r.left, top: r.top }; });
  const rows = [];
  for (const j of joints) {
    const xy = await page.evaluate((j) => { const m = window.tw.map; m.jumpTo({ center: [j.lng, j.lat], zoom: 16 }); const q = m.project([j.lng, j.lat]); return { x: q.x, y: q.y }; }, j);
    await page.waitForTimeout(400);
    await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
    await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
    await page.waitForFunction(() => { const t = window.tw.tools; return t.answer && t.answer.key === t.altsKey(); }, undefined, { timeout: 3000, polling: 30 }).catch(() => {});
    const got = await page.evaluate((j) => {
      const tw = window.tw, t = tw.tools, v = t.verdict(), s = t.stationSpot(), R = 6371008.8, RAD = Math.PI / 180;
      const m = (a, b) => Math.hypot((a[0] - b[0]) * Math.cos(a[1] * RAD), a[1] - b[1]) * RAD * R;
      const ll = s && s.kind !== 'free' ? tw.toLngLat(s.lx, s.ly) : null;
      return { ok: v?.ok ?? null, why: v?.why ?? '', d: ll ? Math.round(m(ll, [j.lng, j.lat])) : null };
    }, j);
    rows.push({ ...j, ...got });
  }
  const kinds = ['false|false', 'true|true', 'false|true', 'true|false'];
  for (const k of kinds) {
    const rs = rows.filter((r) => r.built === k);
    if (!rs.length) continue;
    console.log(`${k === 'false|false' ? 'blueprint' : k === 'true|true' ? 'built' : 'built beside blueprint'} joints: ${rs.length}, placed ${rs.filter((r) => r.ok).length} (moved over 10 m: ${rs.filter((r) => r.ok && r.d > 10).length}), refused ${rs.filter((r) => r.ok === false).length}`);
  }
  for (const r of rows.filter((r) => !r.ok || r.d > 10)) console.log(`  node ${r.node} (${r.built}): ${r.ok ? `placed ${r.d} m off` : `refused "${r.why}"`}`);
  const bad = rows.filter((r) => r.built !== 'false|true' && r.built !== 'true|false' && (!r.ok || r.d > 10));
  if (bad.length) { failed = true; console.log(`FAIL  ${bad.length} joints of like pieces refused or moved`); }
  // a picture of the ghost on the first blueprint joint, then the click
  const first = rows.find((r) => r.built === 'false|false');
  if (first) {
    const xy = await page.evaluate((j) => { const m = window.tw.map; m.jumpTo({ center: [j.lng, j.lat], zoom: 16 }); const q = m.project([j.lng, j.lat]); return { x: q.x, y: q.y }; }, first);
    await page.waitForTimeout(600);
    await page.mouse.move(rect.left + xy.x - 3, rect.top + xy.y - 2);
    await page.mouse.move(rect.left + xy.x, rect.top + xy.y);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}${LABEL}-ghost.png`, clip: { x: rect.left + xy.x - 300, y: rect.top + xy.y - 200, width: 600, height: 400 } });
    const before = await page.evaluate(() => [...window.tw.world.value.nodes.values()].filter((n) => n.platform > 0).length);
    await page.mouse.click(rect.left + xy.x, rect.top + xy.y);
    await page.waitForTimeout(1500);
    const after = await page.evaluate(() => [...window.tw.world.value.nodes.values()].filter((n) => n.platform > 0).length);
    console.log(`click on a blueprint joint: stations ${before} -> ${after}`);
    if (after !== before + 1) { failed = true; console.log('FAIL  the click placed no station'); }
    await page.mouse.move(rect.left + xy.x + 200, rect.top + xy.y + 150);
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${OUT}${LABEL}-placed.png`, clip: { x: rect.left + xy.x - 300, y: rect.top + xy.y - 200, width: 600, height: 400 } });
  }
  await writeFile(`${OUT}${LABEL}.json`, JSON.stringify(rows, null, 1));
} catch (e) {
  failed = true;
  console.log(`FAIL  t1964 joints: ${e.message}`);
} finally {
  await close();
}
process.exitCode = failed ? 1 : 0;
