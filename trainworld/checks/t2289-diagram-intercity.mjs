// T-2289: the network diagram at the intercity scale, on all Japan's real network and the shared Northeast save.
// A stretch is a chain of hops whose lines are all of one colour, through stations with two such hops of that
// colour, whose two ends lie in different zones (the nearest zone centre within 120 km, cities.ts centreWithin);
// a city block is one of layout.ts' cityParts cities (stations with 3 others within 2.5 km, joined within 2.5 km),
// measured as its drawn bounding box's longer side. Passes when no stretch is drawn longer than the longest city
// block; also counts two stations' pills overlapping and stations closer than their half widths plus half a hop.
//   node trainworld/checks/t2289-diagram-intercity.mjs [ne,jpn] [lonely=1]      (about two minutes; figures in
//   artifacts/trainworld/t2289/intercity.json; the second argument overrides the layout's options, to compare)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { openSave, app, waterOf } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, pillSize } from '@app/src/diagram/draw.ts';`);
const which = (process.argv[2] ?? 'ne,jpn').split(',');
const over = process.argv[3] ? Object.fromEntries(process.argv[3].split(',').map((kv) => [kv.split('=')[0], +kv.split('=')[1]])) : {};
const OPTS = { ...L.LAYOUT, ...over };
const REG = JSON.parse(readFileSync(new URL('../app/src/game/registry.json', import.meta.url)));
let failed = 0;
const out = {};
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const city of which) {
  const s = openSave(city);
  const { g } = s.graph();
  const names = [...g.key].map((k) => s.name(k) ?? '');
  const ni = s.api.nodes_info(), xy = new Map();
  for (let i = 0; i < ni.length; i += 10) xy.set(ni[i], [ni[i + 1], ni[i + 2]]);
  // the sea on the ground the page's layout reads (T-2541)
  const { sea } = waterOf(s, g);
  s.api.free();
  // the save's own frame (T-2542)
  const frame = app.makeFrame(s.frameSpec);
  const zone = [...g.key].map((k) => {
    const [lng, lat] = frame.toLonLat(...xy.get(k)), kx = 111320 * Math.cos(lat * Math.PI / 180);
    let best = null, bd = Infinity;
    for (const c of REG.cities) { const d = Math.hypot((lng - c.center[0]) * kx, (lat - c.center[1]) * 111320); if (d < bd) (bd = d), (best = c.id); }
    return bd <= 120000 ? best : null;
  });
  const t0 = performance.now();
  const lay = L.layoutDiagram(g, OPTS, null, undefined, sea);
  const ms = performance.now() - t0;
  const d = L.drawing(g, lay);
  const r = { complexes: g.key.length, ms: Math.round(ms), ...intercity(g, lay, names, zone), overlaps: overlaps(g, d, names), close: closePairs(g, lay) };
  out[city] = r;
  const top = r.stretches.slice(0, 3).map((x) => `${x.from} - ${x.to} ${x.len.toFixed(0)} (${x.hops} hops, ${x.km.toFixed(0)} km)`).join('; ');
  line(r.longest <= r.block, `${city}: no single-line stretch between two cities drawn longer than the longest city block`, `longest city block ${r.block.toFixed(0)} pitches (${r.blockAt}); ${r.stretches.length} stretches, ${r.over.length} longer; longest ${top}`);
  console.log(`     ${city}: ${r.complexes} complexes in ${(ms / 1000).toFixed(1)} s; ${r.size.join(' x ')} pitches across; stretches ${r.total.toFixed(0)} pitches in all; pill overlaps ${r.overlaps.count}${r.overlaps.examples.length ? ' (' + r.overlaps.examples.slice(0, 3).join(', ') + ')' : ''}; close pairs ${r.close}`);
}
mkdirSync(`${ARTIFACTS}t2289`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2289/intercity.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);

function intercity(g, lay, names, zone) {
  const n = g.key.length, m = g.ea.length, pu = lay.pu;
  const { city } = L.cityParts(g);
  const nbr = Array.from({ length: n }, () => []);
  for (let e = 0; e < m; e++) nbr[g.ea[e]].push(e), nbr[g.eb[e]].push(e);
  const colOf = new Array(m);
  for (let e = 0; e < m; e++) {
    const cs = new Set();
    for (let k = g.first[e]; k < g.first[e + 1]; k++) cs.add(g.lines[g.on[k]].colour);
    colOf[e] = cs.size === 1 ? [...cs][0] : null;
  }
  const pass = (v) => nbr[v].length === 2 && colOf[nbr[v][0]] !== null && colOf[nbr[v][0]] === colOf[nbr[v][1]];
  const seen = new Uint8Array(m), stretches = [];
  for (let a = 0; a < n; a++) {
    if (pass(a)) continue;
    for (const e0 of nbr[a]) {
      if (seen[e0] || colOf[e0] === null) continue;
      const path = [a];
      let e = e0, v = a;
      for (;;) {
        seen[e] = 1;
        const w = g.ea[e] === v ? g.eb[e] : g.ea[e];
        path.push(w);
        if (!pass(w)) break;
        const next = nbr[w][0] === e ? nbr[w][1] : nbr[w][0];
        if (seen[next] || colOf[next] !== colOf[e]) break;
        (v = w), (e = next);
      }
      const z = path[path.length - 1];
      if (!zone[a] || !zone[z] || zone[a] === zone[z]) continue;
      let len = 0, km = 0;
      for (let i = 1; i < path.length; i++) {
        len += Math.hypot(lay.x[path[i]] - lay.x[path[i - 1]], lay.y[path[i]] - lay.y[path[i - 1]]);
        km += Math.hypot(g.x[path[i]] - g.x[path[i - 1]], g.y[path[i]] - g.y[path[i - 1]]) / 1000;
      }
      stretches.push({ len: len / pu, hops: path.length - 1, km, from: names[a], to: names[z], za: zone[a], zb: zone[z] });
    }
  }
  stretches.sort((p, q) => q.len - p.len);
  const box = new Map();
  for (let k = 0; k < n; k++) {
    if (city[k] < 0) continue;
    let b = box.get(city[k]);
    if (!b) box.set(city[k], (b = [Infinity, Infinity, -Infinity, -Infinity, k]));
    (b[0] = Math.min(b[0], lay.x[k])), (b[1] = Math.min(b[1], lay.y[k])), (b[2] = Math.max(b[2], lay.x[k])), (b[3] = Math.max(b[3], lay.y[k]));
  }
  let block = 0, blockAt = '';
  for (const b of box.values()) { const side = Math.max(b[2] - b[0], b[3] - b[1]) / pu; if (side > block) (block = side), (blockAt = names[b[4]]); }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) (x0 = Math.min(x0, lay.x[i])), (x1 = Math.max(x1, lay.x[i])), (y0 = Math.min(y0, lay.y[i])), (y1 = Math.max(y1, lay.y[i]));
  return { size: [Math.round((x1 - x0) / pu), Math.round((y1 - y0) / pu)], cities: box.size, block, blockAt, longest: stretches.length ? stretches[0].len : 0, total: stretches.reduce((a, x) => a + x.len, 0), over: stretches.filter((x) => x.len > block).map((x) => `${x.from} - ${x.to}`), stretches: stretches.slice(0, 40) };
}

/** pills of two stations overlapping, each at its drawn size (as t2161) */
function overlaps(g, d, names) {
  const n = g.key.length, adj = new Set();
  for (let e = 0; e < g.ea.length; e++) adj.add(g.ea[e] * n + g.eb[e]);
  const rOf = (i) => L.pillSize(d, d.pillOf[i]).reduce((a, b) => a + b);
  const seg = (p, q) => {
    const pt = (px, py, s) => { const dx = s[2] - s[0], dy = s[3] - s[1], l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((px - s[0]) * dx + (py - s[1]) * dy) / l2)) : 0; return Math.hypot(s[0] + dx * t - px, s[1] + dy * t - py); };
    return Math.min(pt(p[0], p[1], q), pt(p[2], p[3], q), pt(q[0], q[1], p), pt(q[2], q[3], p));
  };
  const cell = 6 * d.pitch, grid = new Map();
  d.pills.forEach((p, i) => { const k = `${Math.floor((p[0] + p[2]) / 2 / cell)},${Math.floor((p[1] + p[3]) / 2 / cell)}`; if (!grid.has(k)) grid.set(k, []); grid.get(k).push(i); });
  const R = Math.ceil(Math.max(...d.pills.map((p) => Math.hypot(p[2] - p[0], p[3] - p[1]))) / cell + 1);
  let count = 0;
  const examples = [];
  d.pills.forEach((p, i) => {
    const cx = Math.floor((p[0] + p[2]) / 2 / cell), cy = Math.floor((p[1] + p[3]) / 2 / cell);
    for (let gx = cx - R; gx <= cx + R; gx++) for (let gy = cy - R; gy <= cy + R; gy++) for (const j of grid.get(`${gx},${gy}`) ?? []) {
      if (j <= i || d.pillOf[j] === d.pillOf[i]) continue;
      if (seg(p, d.pills[j]) < rOf(i) + rOf(j) + p[4] + d.pills[j][4]) { count++; const a = Math.min(d.pillOf[i], d.pillOf[j]), b = Math.max(d.pillOf[i], d.pillOf[j]); if (examples.length < 6) examples.push(`${names[d.pillOf[i]]} | ${names[d.pillOf[j]]}${adj.has(a * n + b) ? ' (one hop)' : ''}`); }
    }
  });
  return { count, examples };
}

/** stations sharing no edge closer than their half widths plus half a hop (as t2160) */
function closePairs(g, lay) {
  const n = g.key.length, adj = new Set(), half = 0.5 * lay.hop;
  for (let e = 0; e < g.ea.length; e++) adj.add(g.ea[e] * n + g.eb[e]);
  const cell = 4 * lay.hop, grid = new Map();
  for (let i = 0; i < n; i++) { const k = `${Math.floor(lay.x[i] / cell)},${Math.floor(lay.y[i] / cell)}`; if (!grid.has(k)) grid.set(k, []); grid.get(k).push(i); }
  let close = 0;
  for (let i = 0; i < n; i++) {
    const cx = Math.floor(lay.x[i] / cell), cy = Math.floor(lay.y[i] / cell);
    for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) for (const j of grid.get(`${gx},${gy}`) ?? []) {
      if (j <= i || adj.has(Math.min(i, j) * n + Math.max(i, j))) continue;
      if (Math.hypot(lay.x[j] - lay.x[i], lay.y[j] - lay.y[i]) < lay.hw[i] + lay.hw[j] + half) close++;
    }
  }
  return close;
}
