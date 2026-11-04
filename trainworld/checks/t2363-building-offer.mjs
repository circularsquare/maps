// T-2363: the station building's offered rectangle follows a platform taken as straight (its chord),
// the direction most platforms share within 15 degrees, ties to the longer total platform length.
// Made-up stations first, then every complex of the real New York save: each offer covers every
// platform, and those whose direction moved off the old rule's (the longest platform's heading at its
// node) are listed. Run from the maps root:
//   node trainworld/checks/t2363-building-offer.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp } from './lib.mjs';

const app = await loadApp(`export * from '@app/src/game/complexes.ts'; export * from '@app/src/game/buildings.ts';
export { PlatformLines } from '@app/src/workers/platformLines.ts';`);
const deg = (a) => a * 180 / Math.PI, rad = (d) => d * Math.PI / 180;
/** directions apart as lines, degrees */
const gap = (a, b) => { const r = ((deg(a - b) % 180) + 180) % 180; return Math.min(r, 180 - r); };
/** rectangles apart, degrees: one turned a right angle is the same rectangle */
const gap90 = (a, b) => { const r = ((deg(a - b) % 90) + 90) % 90; return Math.min(r, 90 - r); };
/** the offered rectangle's first edge direction */
const firstEdge = (r) => Math.atan2(r[3] - r[1], r[2] - r[0]);

// ---- made-up stations, flat frame
let n = 0;
const straight = (x, y, d, length) => ({ id: String(n++), x, y, heading: rad(d), length });
/** a platform bent by `bend` degrees along its length, its chord along `d` */
const curved = (x, y, d, length, bend) => {
  const pts = [], m = 8;
  for (let i = 0; i <= m; i++) {
    const t = (i / m - 0.5) * length, a = rad(d) + rad(bend) * (i / m - 0.5);
    pts.push([t, a]);
  }
  // integrate the tangent from the middle outwards
  const line = new Array(2 * (m + 1)).fill(0), h = length / m, mid = m / 2;
  for (let i = mid + 1; i <= m; i++) { const a = (pts[i][1] + pts[i - 1][1]) / 2; line[2 * i] = line[2 * i - 2] + Math.cos(a) * h; line[2 * i + 1] = line[2 * i - 1] + Math.sin(a) * h; }
  for (let i = mid - 1; i >= 0; i--) { const a = (pts[i][1] + pts[i + 1][1]) / 2; line[2 * i] = line[2 * i + 2] - Math.cos(a) * h; line[2 * i + 1] = line[2 * i + 3] - Math.sin(a) * h; }
  // a heading off the chord, as where the curve is all to one side of the node
  return { id: String(n++), x, y, heading: rad(d + bend / 2), length, line };
};

// one straight platform: along it
assert.ok(gap(app.offeredDirection([straight(0, 0, 37, 200)]), rad(37)) < 1e-6);
// one curved platform bent 40 degrees, its heading at the node 20 degrees off its chord: the chord
const c1 = curved(0, 0, 10, 300, 40);
assert.ok(gap(app.offeredDirection([c1]), rad(10)) < 0.5, `curved: ${deg(app.offeredDirection([c1]))}`);
// three near-parallel platforms (0, 6, 12 degrees) and one long crossing one at 90: the three win,
// along the longest of them
const g3 = [straight(0, 0, 0, 160), straight(0, 20, 6, 180), straight(0, 40, 12, 140), straight(50, 0, 90, 400)];
assert.ok(gap(app.offeredDirection(g3), rad(6)) < 1e-6, `three beat one: ${deg(app.offeredDirection(g3))}`);
// two and two: the pair with the longer total length
const tie = [straight(0, 0, 0, 200), straight(0, 20, 3, 200), straight(50, 0, 90, 300), straight(70, 0, 88, 120)];
assert.ok(gap(app.offeredDirection(tie), rad(90)) < 1e-6, `tie to the longer pair: ${deg(app.offeredDirection(tie))}`);
// 20 degrees apart is not parallel: one and one, the longer one
const apart = [straight(0, 0, 0, 200), straight(0, 20, 20, 220)];
assert.ok(gap(app.offeredDirection(apart), rad(20)) < 1e-6);
// directions near the wrap (178 and 2 degrees) are parallel
const wrap = [straight(0, 0, 178, 100), straight(0, 20, 2, 120), straight(50, 0, 90, 300)];
assert.ok(gap(app.offeredDirection(wrap), rad(2)) < 1e-6, `wrap: ${deg(app.offeredDirection(wrap))}`);
console.log('made-up stations ok');

// ---- the real New York save
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(new URL('../T-007-real-nyc.save', import.meta.url)));
const api = new TrackApi(-73.985, 40.758, true);
assert.ok(api.load(raw.subarray(8 + raw.readUInt32LE(4))), 'save did not load');
const R = 6378137;
const kAt = api.frame_kind() === 2 ? (y) => Math.cosh(y / R) : () => 1;
const lines = new app.PlatformLines();
const ei = api.edges_info(), edges = [];
for (let i = 0; i < ei.length; i += 10) edges.push({ id: ei[i], a: ei[i + 1], b: ei[i + 2] });
lines.sync(edges);
const ni = api.nodes_info(), all = [];
for (let i = 0; i < ni.length; i += 10) {
  if (!(ni[i + 7] > 0) || ni[i + 8] !== 1) continue;
  const line = lines.of(api, ni[i], ni[i + 1], ni[i + 2], ni[i + 7], kAt(ni[i + 2]));
  all.push({ id: String(ni[i]), name: api.node_name(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7], ...(line ? { line } : {}) });
}
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const cx = app.buildComplexes(all, links, kAt);
const byId = new Map(all.map((s) => [s.id, s]));

let moved = 0, total = 0;
const rows = [];
for (const m of cx.members) {
  const st = m.map((id) => byId.get(id)).filter(Boolean);
  const r = app.offeredRectangle(st, kAt);
  if (!r) continue;
  total++;
  // covers every platform's line, half a box and the pad out
  const a = firstEdge(r), c = Math.cos(a), s = Math.sin(a);
  const u = r.filter((_, i) => i % 2 === 0).map((x, i) => x * c + r[2 * i + 1] * s), v = r.filter((_, i) => i % 2 === 0).map((x, i) => -x * s + r[2 * i + 1] * c);
  const [u0, u1, v0, v1] = [Math.min(...u), Math.max(...u), Math.min(...v), Math.max(...v)];
  for (const p of st) {
    const o = app.platformOffsets(p, kAt(p.y)), pad = 20 * kAt(p.y);
    for (let i = 0; i + 1 < o.length; i += 2) {
      const x = p.x + o[i], y = p.y + o[i + 1], pu = x * c + y * s, pv = -x * s + y * c;
      assert.ok(pu >= u0 + pad && pu <= u1 - pad && pv >= v0 + pad && pv <= v1 - pad, `${p.name}: a platform point outside the offer`);
    }
  }
  const lead = st.reduce((x, y) => (y.length > x.length ? y : x));
  const d = gap90(a, lead.heading);
  const dirs = st.map((p) => deg(app.platformChord(p)));
  if (d > 3) moved++;
  rows.push({ name: st.map((p) => p.name).filter((x, i, l) => l.indexOf(x) === i).join(' / '), n: st.length, moved: d, dirs: dirs.map((x) => Math.round(((x % 180) + 180) % 180)).join(' '), bend: Math.max(...st.map((p) => gap(app.platformChord(p), p.heading))) });
}
rows.sort((p, q) => q.moved - p.moved);
console.log(`${total} complexes; ${moved} turned more than 3 degrees from the old rule`);
for (const r of rows.slice(0, 15)) console.log(`  ${r.name}: ${r.n} platforms, turned ${r.moved.toFixed(1)} deg, chords ${r.dirs}, worst chord off its heading ${r.bend.toFixed(1)} deg`);
console.log('t2363 ok');
