// T-2168: the network diagram's faint zone borders (diagram/borders.ts, warp.ts) on all Japan, the shared Northeast
// save and the real New York save. Each save is laid out as the page's worker does, then its zones' coarse outlines
// (`packs/outlines.json`) are put in the graph's frame the way view.ts does (by the stations' own mercator places) and
// drawn. A station is wrong when the nearest border piece drawn (opacity a level or more) within 16 pitches of it puts
// it on another zone's side; stations with no border near are fine (none is drawn there). Passes at 95% or more each.
// Also the whole rings, even-odd (the far parts are never drawn, so this is looser), and the times.
//   node trainworld/checks/t2168-borders.mjs [jpn,ne,nyc]      (about two minutes with Japan; figures in
//   artifacts/trainworld/t2168/figures.json)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { openSave } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing } from '@app/src/diagram/draw.ts';
export { zoneBorders, inRings, LEVELS } from '@app/src/diagram/borders.ts'; export { toMerc01, EARTH_C } from '@app/src/map/geo.ts';
export { makeFrame } from '@app/src/game/frame.ts';`);
const which = (process.argv[2] ?? 'jpn,ne,nyc').split(',');
const TW = new URL('../', import.meta.url);
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
const ORIGIN = { ne: [-73.985, 40.758], nyc: [-73.985, 40.758], jpn: [139.7, 35.7] };
const SEEN = 16;
let failed = 0;
const out = {};
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const city of which) {
  const s = openSave(city);
  const { g } = s.graph();
  const names = [...g.key].map((k) => s.name(k) ?? '');
  const ni = s.api.nodes_info(), xy = new Map();
  for (let i = 0; i < ni.length; i += 10) xy.set(ni[i], [ni[i + 1], ni[i + 2]]);
  s.api.free();
  // the frame as view.ts works it out, from each node's key station
  const fr = L.makeFrame({ kind: 'equirect', lon: ORIGIN[city][0], lat: ORIGIN[city][1] }), k = g.k / L.EARTH_C;
  let fx = 0, fy = 0, w = Infinity, so = Infinity, e = -Infinity, n = -Infinity;
  g.key.forEach((key, i) => {
    const [lng, lat] = fr.toLonLat(...xy.get(key)), [mx, my] = L.toMerc01(lng, lat);
    (fx += mx - g.x[i] * k), (fy += my + g.y[i] * k);
    (w = Math.min(w, lng)), (e = Math.max(e, lng)), (so = Math.min(so, lat)), (n = Math.max(n, lat));
  });
  (fx /= g.key.length), (fy /= g.key.length);
  const shapes = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > e || b[2] < w || b[1] > n || b[3] < so) continue;
    shapes.push({ id: c.id, rings: flat.map((f) => {
      const r = new Float64Array(f.length);
      let x = 0, y = 0;
      for (let i = 0; i < f.length; i += 2) {
        const [mx, my] = L.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q);
        (r[i] = (mx - fx) / k), (r[i + 1] = -(my - fy) / k);
      }
      return r;
    }) });
  }
  const t0 = performance.now();
  const lay = L.layoutDiagram(g);
  const t1 = performance.now();
  const d = L.drawing(g, lay);
  const t2 = performance.now();
  const b = L.zoneBorders(g, lay.x, lay.y, lay.blocks, d.pitch, shapes, true);
  const t3 = performance.now();
  const r = figures(g, lay, d, b, names);
  r.ms = { layout: Math.round(t1 - t0), borders: Math.round(t3 - t2) };
  r.zonesDrawn = b.zones.length;
  r.blocks = lay.blocks.list.length;
  r.pointsDrawn = b.levels.reduce((a, l) => a + l.length / 2, 0);
  out[city] = r;
  console.log(`     ${city}: ${g.key.length} stations, ${r.blocks} blocks, ${r.zonesDrawn} zones; ${r.beside} beside a border, ${r.wrong} on the wrong side (${r.wrongPast1} by more than a pitch); whole rings ${r.ringPct}%; layout ${r.ms.layout} ms, borders ${r.ms.borders} ms`);
  console.log(`     worst: ${r.worst.map((z) => `${z.name} ${z.wrong}/${z.n}`).join(', ')}`);
  say(r.pct >= 95, `${city}: 95% of stations or more on their own zone's side of the borders drawn`, `${r.pct}%`);
}
mkdirSync(`${ARTIFACTS}t2168`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2168/figures.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);

function figures(g, lay, d, b, names) {
  const n = g.key.length, pitch = d.pitch, cell = SEEN * pitch, grid = new Map();
  b.arcs.forEach((a, ai) => {
    for (let k = 0; k + 3 < a.p.length; k += 3) {
      if (Math.round(((a.p[k + 2] + a.p[k + 5]) / 2) * L.LEVELS) < 1) continue;
      const key = `${Math.floor(a.p[k] / cell)},${Math.floor(a.p[k + 1] / cell)}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(ai, k);
    }
  });
  const per = b.zones.map((id) => ({ id, name: REG.cities.find((c) => c.id === id)?.name ?? id, n: 0, wrong: 0 }));
  let all = 0, ring = 0, wrong = 0, past = 0, beside = 0;
  const examples = [];
  for (let i = 0; i < n; i++) {
    const z = b.zoneOf[i];
    if (z < 0) continue;
    all++, per[z].n++;
    if (L.inRings(b.rings[z], lay.x[i], lay.y[i])) ring++;
    const px = lay.x[i], py = lay.y[i], cx = Math.floor(px / cell), cy = Math.floor(py / cell);
    let bd = cell * cell, best = null;
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
      const l = grid.get(`${gx},${gy}`);
      if (!l) continue;
      for (let q = 0; q < l.length; q += 2) {
        const a = b.arcs[l[q]], k = l[q + 1], ax = a.p[k], ay = a.p[k + 1], dx = a.p[k + 3] - ax, dy = a.p[k + 4] - ay, l2 = dx * dx + dy * dy;
        const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0, ex = ax + t * dx - px, ey = ay + t * dy - py, dd = ex * ex + ey * ey;
        if (dd < bd) (bd = dd), (best = [a, ax, ay, dx, dy]);
      }
    }
    if (!best) continue;
    beside++;
    const [a, ax, ay, dx, dy] = best, left = dx * (py - ay) - dy * (px - ax) > 0;
    if ((left ? a.left : a.right) === z) continue;
    wrong++, per[z].wrong++;
    if (bd > pitch * pitch) past++;
    if (examples.length < 10) examples.push(`${names[i]} (${b.zones[z]})`);
  }
  per.sort((p, q) => q.wrong - p.wrong || q.n - p.n);
  return { stations: all, beside, wrong, wrongPast1: past, pct: +(100 * (all - wrong) / all).toFixed(1), ringPct: +(100 * ring / all).toFixed(1), worst: per.slice(0, 6).map(({ name, n, wrong }) => ({ name, n, wrong })), examples };
}
