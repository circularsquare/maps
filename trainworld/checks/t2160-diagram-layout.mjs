// T-2160: the network diagram's layout (app/src/diagram/layout.ts) on the real New York save and Tokyo's
// real network: every drawn stretch within 0.5 degrees of 0, 45 or 90; no two stations that share no
// edge closer than their half widths plus half a hop; New York in under 5 s, Tokyo in under 8 s; one
// station added (a new complex halfway along a hop) moves the median complex less than a tenth of a hop
// on the rebuild, which starts from the last layout.
//   node trainworld/checks/t2160-diagram-layout.mjs [nyc,tyo] [r2]      (about a minute; `r2` lays out with
//   round 2's options instead of the game's; figures in artifacts/trainworld/t2160/check.json)
import { mkdirSync, writeFileSync } from 'node:fs';
import { openSave, layoutApp } from './diagram-saves.mjs';
import { ARTIFACTS } from './lib.mjs';

const L = await layoutApp();
const which = (process.argv[2] ?? 'nyc,tyo').split(',');
const OPTS = process.argv[3] === 'r2' ? L.LAYOUT_R2 : L.LAYOUT;
const LIMIT = { nyc: 5000, tyo: 8000 };
let failed = 0;
const out = {};
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; };

/** the graph with one more complex halfway along its longest hop among lines (each line on it stops there) */
function withStation(g) {
  let best = -1, bl = 0;
  for (let e = 0; e < g.ea.length; e++) {
    const l = Math.hypot(g.x[g.eb[e]] - g.x[g.ea[e]], g.y[g.eb[e]] - g.y[g.ea[e]]);
    if (l > bl && l < 3000) bl = l, best = e;
  }
  const a = g.ea[best], b = g.eb[best], n = g.key.length;
  const key = Int32Array.from([...g.key, Math.max(...g.key) + 1]);
  const x = Float64Array.from([...g.x, (g.x[a] + g.x[b]) / 2]), y = Float64Array.from([...g.y, (g.y[a] + g.y[b]) / 2]);
  const lines = g.lines.map((l) => {
    const c = [], s = [];
    for (let i = 0; i < l.chain.length; i++) {
      if (i && ((l.chain[i - 1] === a && l.chain[i] === b) || (l.chain[i - 1] === b && l.chain[i] === a))) c.push(n), s.push(1);
      c.push(l.chain[i]);
      s.push(l.stop[i]);
    }
    return { ...l, chain: Int32Array.from(c), stop: Uint8Array.from(s) };
  });
  const em = new Map();
  lines.forEach((l, li) => { for (let i = 1; i < l.chain.length; i++) { const p = Math.min(l.chain[i - 1], l.chain[i]), q = Math.max(l.chain[i - 1], l.chain[i]); const k = p * (n + 1) + q; if (!em.has(k)) em.set(k, new Set()); em.get(k).add(li); } });
  const ks = [...em.keys()].sort((p, q) => p - q), first = [0], on = [];
  for (const k of ks) { on.push(...em.get(k)); first.push(on.length); }
  return { ...g, key, x, y, lines, ea: Int32Array.from(ks, (k) => Math.floor(k / (n + 1))), eb: Int32Array.from(ks, (k) => k % (n + 1)), first: Int32Array.from(first), on: Int32Array.from(on) };
}

for (const name of which) {
  const s = openSave(name);
  const { g } = s.graph();
  s.api.free();
  L.layoutDiagram(g, OPTS); // the code compiled
  const t0 = performance.now();
  const lay = L.layoutDiagram(g, OPTS);
  const ms = performance.now() - t0;
  // every drawn stretch: each piece's segments between vertices
  let worst = 0, stretches = 0;
  for (const p of lay.pieces) for (let i = 2; i < p.verts.length; i += 2) {
    const dx = p.verts[i] - p.verts[i - 2], dy = p.verts[i + 1] - p.verts[i - 1];
    if (Math.hypot(dx, dy) < 1e-6 * lay.hop) continue;
    stretches++;
    let a = (Math.atan2(dy, dx) * 180 / Math.PI) % 45;
    if (a < 0) a += 45;
    worst = Math.max(worst, Math.min(a, 45 - a));
  }
  // spacing between stations that share no edge
  const n = g.key.length, adj = new Set();
  for (let e = 0; e < g.ea.length; e++) adj.add(g.ea[e] * n + g.eb[e]);
  let close = 0, worstShare = Infinity;
  const half = 0.5 * lay.hop;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    if (adj.has(i * n + j)) continue;
    const want = lay.hw[i] + lay.hw[j] + half, d = Math.hypot(lay.x[j] - lay.x[i], lay.y[j] - lay.y[i]);
    if (d < want * 3) worstShare = Math.min(worstShare, d / want);
    if (d < want) close++;
  }
  // a rebuild with one more station
  const last = new Map([...g.key].map((k, i) => [k, [lay.bx[i], lay.by[i]]]));
  const g2 = withStation(g);
  const t1 = performance.now();
  const lay2 = L.layoutDiagram(g2, OPTS, last);
  const ms2 = performance.now() - t1;
  const moved = [...g.key].map((_, i) => Math.hypot(lay2.x[i] - lay.x[i], lay2.y[i] - lay.y[i]) / lay.hop);
  const movedB = [...g.key].map((_, i) => Math.hypot(lay2.bx[i] - lay.bx[i], lay2.by[i] - lay.by[i]) / lay.hop);
  // how compact: the drawing's extent in stroke pitches
  const ext = (v) => Math.max(...v) - Math.min(...v);
  const size = [ext(lay.x) / lay.pu, ext(lay.y) / lay.pu];
  const r = { complexes: n, ms: Math.round(ms), phases: lay.ms, pieces: lay.pieces.length, straight: lay.straight, bent: lay.bent, worstDeg: worst, stretches, close, worstShare, rebuildMs: Math.round(ms2), movedMedian: median(moved), movedP90: [...moved].sort((a, b) => a - b)[Math.floor(n * 0.9)], movedBMedian: median(movedB), sizePitches: size.map(Math.round) };
  out[name] = r;
  line(worst <= 0.5, `${name}: every drawn stretch within 0.5 degrees of 0/45/90`, `worst ${worst.toFixed(3)} of ${stretches} stretches; ${r.pieces} pieces, ${r.straight} straight, ${r.bent} with one bend`);
  line(close === 0, `${name}: no stations closer than their half widths plus half a hop`, `${close} pairs closer; the closest at ${worstShare.toFixed(2)} of its distance`);
  line(ms < LIMIT[name], `${name}: laid out in under ${LIMIT[name] / 1000} s`, `${(ms / 1000).toFixed(2)} s (A ${(lay.ms.a / 1000).toFixed(2)}, B ${(lay.ms.b / 1000).toFixed(2)}, routing ${(lay.ms.route / 1000).toFixed(2)}, order ${(lay.ms.order / 1000).toFixed(2)}); ${n} complexes; ${size.map(Math.round).join(' x ')} pitches across`);
  line(r.movedMedian < 0.1, `${name}: one added station moves the median complex under a tenth of a hop`, `median ${r.movedMedian.toFixed(3)} hops (B ${r.movedBMedian.toFixed(3)}), 90th percentile ${r.movedP90.toFixed(3)}; rebuild ${(ms2 / 1000).toFixed(2)} s`);
}
mkdirSync(`${ARTIFACTS}t2160`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2160/check${OPTS === L.LAYOUT_R2 ? '-r2' : ''}.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
