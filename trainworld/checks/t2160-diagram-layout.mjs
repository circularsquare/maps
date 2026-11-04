// T-2160: the network diagram's layout (app/src/diagram/layout.ts) on the real New York save and Tokyo's
// real network: every drawn stretch within 0.5 degrees of 0, 45 or 90; no two stations that share no
// edge closer than their half widths plus half a hop; New York in under 5 s, Tokyo in under 8 s; one
// station added (a new complex halfway along a hop) moves the median complex less than a tenth of a hop
// on the rebuild, which starts from the last layout; no line doubles back at a station (T-2243: drawn
// turning 180 degrees where the real line turns less than 90, or 135 where it turns less than 45; 135
// where it turns 45 to 90 is counted as a sharp corner, not failed). `nyc-game` and `tyo-game` are the graphs
// the game itself builds from those saves, captured by diagram-game.mjs (T-2255): Node lays them out exactly
// as the page's worker does, which is checked against the capture, and their time is the page worker's.
// T-2290: few near stations drawn on the wrong side of each other, and `l8`, Anita's line 8 game
// (t2290-line8-game.mjs builds it), with line 8 east of Fleetwood and no loop round it.
//   node trainworld/checks/diagram-game.mjs http://localhost:6194/      (first, after any change to the layout)
//   node trainworld/checks/t2160-diagram-layout.mjs [nyc,tyo,nyc-game,tyo-game] [r2]      (about two minutes;
//   `r2` lays out with round 2's options instead of the game's; figures in artifacts/trainworld/t2160/check.json)
import { mkdirSync, writeFileSync } from 'node:fs';
import { openSave, layoutApp } from './diagram-saves.mjs';
import { gameGraph, splitCity, bitsHash } from './diagram-game.mjs';
import { ARTIFACTS } from './lib.mjs';

const L = await layoutApp();
const which = (process.argv[2] ?? 'nyc,tyo,nyc-game,tyo-game,l8,l8-game').split(',');
const OPTS = process.argv[3] === 'r2' ? L.LAYOUT_R2 : L.LAYOUT;
const LIMIT = { nyc: 5000, tyo: 8000, l8: 1000 };
/** at most this many near pairs drawn on the wrong side of each other (T-2290: 42 to 71 before, 10 to 18 after) */
const WRONG_MAX = { nyc: 30, tyo: 30, l8: 0 };
let failed = 0;
const out = {};
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };
const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : 0; };
const turnDeg = (a, b) => Math.abs(Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])) * 180 / Math.PI;

/** each line's stations where its drawing turns 135 degrees or more and the real line less than 90: [line, node, drawn,
 * real, whether it doubles back: a U-turn, or a hairpin where the real line turns less than 45 (else a sharp corner)] */
function doublings(g, lay) {
  const n = g.key.length, at = new Map();
  for (const pc of lay.pieces) for (let i = 0; i + 1 < pc.nodes.length; i++) {
    at.set(pc.nodes[i] * n + pc.nodes[i + 1], [pc, i, 1]);
    at.set(pc.nodes[i + 1] * n + pc.nodes[i], [pc, i + 1, -1]);
  }
  /** the drawn way from u toward w (a hop on one piece) as it leaves u */
  const leaving = (u, w) => {
    const x = at.get(u * n + w);
    if (!x) return null;
    const [pc, i, d] = x, v = pc.verts;
    for (let t = pc.at[i]; t !== pc.at[i + d]; t += d) {
      const dx = v[(t + d) * 2] - v[t * 2], dy = v[(t + d) * 2 + 1] - v[t * 2 + 1];
      if (Math.hypot(dx, dy) > 1e-6 * lay.hop) return [dx, dy];
    }
    return null;
  };
  const out = [];
  g.lines.forEach((l, li) => {
    const c = l.chain;
    for (let i = 1; i + 1 < c.length; i++) {
      if (c[i - 1] === c[i + 1]) continue;
      const a = leaving(c[i], c[i - 1]), b = leaving(c[i], c[i + 1]);
      if (!a || !b) continue;
      const drawn = turnDeg([-a[0], -a[1]], b);
      const real = turnDeg([g.x[c[i]] - g.x[c[i - 1]], g.y[c[i]] - g.y[c[i - 1]]], [g.x[c[i + 1]] - g.x[c[i]], g.y[c[i + 1]] - g.y[c[i]]]);
      if (drawn > 112.5 && real < 90) out.push([li, c[i], Math.round(drawn), Math.round(real), drawn > 157.5 || real < 45]);
    }
  });
  return out;
}

/** T-2290: stations on the wrong side of their neighbours. Each station's 6 nearest in reality that it shares no
 * edge with; a pair counts when its bearing in the drawing is more than 135 degrees off the real one. */
export function wrongSide(g, px, py, names) {
  const n = g.key.length, adj = new Set(), seen = new Set(), out = [];
  for (let e = 0; e < g.ea.length; e++) adj.add(g.ea[e] * n + g.eb[e]);
  const cell = 2000, grid = new Map(), key = (cx, cy) => `${cx},${cy}`;
  for (let i = 0; i < n; i++) {
    const k = key(Math.floor(g.x[i] / cell), Math.floor(g.y[i] / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  let pairs = 0;
  for (let i = 0; i < n; i++) {
    const cx = Math.floor(g.x[i] / cell), cy = Math.floor(g.y[i] / cell), near = [];
    for (let r = 1; near.length < 6 && r < 64; r *= 2) {
      near.length = 0;
      for (let gx = cx - r; gx <= cx + r; gx++) for (let gy = cy - r; gy <= cy + r; gy++) for (const j of grid.get(key(gx, gy)) ?? []) {
        if (j !== i && !adj.has(Math.min(i, j) * n + Math.max(i, j))) near.push([Math.hypot(g.x[j] - g.x[i], g.y[j] - g.y[i]), j]);
      }
    }
    near.sort((a, b) => a[0] - b[0]);
    for (const [rd, j] of near.slice(0, 6)) {
      const k = Math.min(i, j) * n + Math.max(i, j);
      if (seen.has(k)) continue;
      seen.add(k);
      pairs++;
      const dx = px[j] - px[i], dy = py[j] - py[i], rx = g.x[j] - g.x[i], ry = g.y[j] - g.y[i];
      const cos = (dx * rx + dy * ry) / (Math.hypot(dx, dy) * Math.hypot(rx, ry) || 1);
      if (cos < -Math.SQRT1_2) out.push([cos, `${names[i]} | ${names[j]} (${Math.round(Math.acos(cos) * 180 / Math.PI)} degrees, ${(rd / 1000).toFixed(1)} km)`]);
    }
  }
  return { pairs, wrong: out.sort((a, b) => a[0] - b[0]).map((x) => x[1]) };
}

/** T-2290, the line 8 game: Fleetwood west of every stop of line 8 after West Mount Vernon, and on the side of
 * line 8's drawing (its nearest stretch) that it is of the real line, so the line does not loop round it */
function line8(g, lay, names) {
  const f = names.indexOf('Fleetwood'), l8 = g.lines.find((l) => l.colour === '#b5122e');
  if (f < 0 || !l8) return { ok: false, info: 'no Fleetwood or line 8' };
  const after = [...l8.chain].filter((k) => names[k] !== 'West Mount Vernon');
  const west = after.filter((k) => lay.x[k] <= lay.x[f]).map((k) => names[k]);
  const side = (X, Y) => {
    let best = Infinity, s = 0;
    for (let i = 1; i < l8.chain.length; i++) {
      const a = l8.chain[i - 1], b = l8.chain[i], dx = X[b] - X[a], dy = Y[b] - Y[a], l2 = dx * dx + dy * dy;
      const t = Math.max(0, Math.min(1, ((X[f] - X[a]) * dx + (Y[f] - Y[a]) * dy) / l2));
      const d = Math.hypot(X[a] + dx * t - X[f], Y[a] + dy * t - Y[f]);
      if (d < best) (best = d), (s = Math.sign(dx * (Y[f] - Y[a]) - dy * (X[f] - X[a])));
    }
    return s;
  };
  const same = side(lay.x, lay.y) === side(g.x, g.y);
  return { ok: !west.length && same, info: `${west.length ? 'drawn west of Fleetwood: ' + west.join(', ') : 'Fleetwood west of all of them'}; Fleetwood ${same ? 'on' : 'off'} its real side of the line` };
}

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
  const [city, game] = splitCity(name);
  let g, names, cap = null;
  if (game) {
    cap = gameGraph(city);
    if (!cap) { console.log(`SKIP ${name}: no capture; run diagram-game.mjs first`); continue; }
    ({ g } = cap);
    names = cap.text;
  } else {
    const s = openSave(city);
    ({ g } = s.graph());
    names = [...g.key].map((k) => s.name(k) ?? '');
    s.api.free();
  }
  L.layoutDiagram(g, OPTS); // the code compiled
  const t0 = performance.now();
  const lay = L.layoutDiagram(g, OPTS);
  let ms = performance.now() - t0, phases = lay.ms;
  // the game's graph: the page's own time, when the capture is of this layout code
  const pageOk = cap && cap.fresh && OPTS === L.LAYOUT;
  if (cap && !pageOk) console.log(`INFO ${name}: the capture (${cap.captured}) is older than the layout code${cap.graphFresh ? '' : ' and graph.ts'}; times are Node's, and the page is not compared; run diagram-game.mjs`);
  if (pageOk) {
    const same = bitsHash(lay.x, lay.y) === cap.page.places;
    line(same, `${name}: Node lays out the game's graph exactly as the page's worker did`, same ? '' : 'places differ: some engine-dependent maths in the layout (diagram/fmath.ts)');
    (ms = cap.page.ms.total), (phases = cap.page.ms);
  }
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
  const last = L.lastPlaces(g, lay);
  const g2 = withStation(g);
  const t1 = performance.now();
  const lay2 = L.layoutDiagram(g2, OPTS, last);
  const ms2 = performance.now() - t1;
  const moved = [...g.key].map((_, i) => Math.hypot(lay2.x[i] - lay.x[i], lay2.y[i] - lay.y[i]) / lay.hop);
  const movedB = [...g.key].map((_, i) => Math.hypot(lay2.bx[i] - lay.bx[i], lay2.by[i] - lay.by[i]) / lay.hop);
  // how compact: the drawing's extent in stroke pitches
  const ext = (v) => Math.max(...v) - Math.min(...v);
  const size = [ext(lay.x) / lay.pu, ext(lay.y) / lay.pu];
  const turns = doublings(g, lay), back = turns.filter((b) => b[4]), sharp = turns.filter((b) => !b[4]);
  const backAt = [...new Set(back.map((b) => names[b[1]]))], sharpAt = [...new Set(sharp.map((b) => names[b[1]]))];
  const r = { complexes: n, ms: Math.round(ms), phases, pieces: lay.pieces.length, straight: lay.straight, bent: lay.bent, worstDeg: worst, stretches, close, worstShare, rebuildMs: Math.round(ms2), movedMedian: median(moved), movedP90: [...moved].sort((a, b) => a - b)[Math.floor(n * 0.9)], movedBMedian: median(movedB), sizePitches: size.map(Math.round), doublings: turns.map(([li, k, d, rl]) => `${names[k]} (${g.lines[li].colour}) ${d} drawn, ${rl} real`) };
  out[name] = r;
  line(worst <= 0.5, `${name}: every drawn stretch within 0.5 degrees of 0/45/90`, `worst ${worst.toFixed(3)} of ${stretches} stretches; ${r.pieces} pieces, ${r.straight} straight, ${r.bent} with one bend`);
  line(close === 0, `${name}: no stations closer than their half widths plus half a hop`, `${close} pairs closer; the closest at ${worstShare.toFixed(2)} of its distance`);
  line(ms < LIMIT[city], `${name}: laid out in under ${LIMIT[city] / 1000} s${pageOk ? " in the page's worker" : ''}`, `${(ms / 1000).toFixed(2)} s (A ${(phases.a / 1000).toFixed(2)}, B ${(phases.b / 1000).toFixed(2)}, routing ${(phases.route / 1000).toFixed(2)}, order ${(phases.order / 1000).toFixed(2)}); ${n} complexes; ${size.map(Math.round).join(' x ')} pitches across`);
  line(back.length === 0, `${name}: no line doubles back at a station`, `${back.length} lines at ${backAt.length} stations${backAt.length ? ': ' + backAt.slice(0, 6).join(', ') : ''}; sharp corners (135 drawn, 45 to 90 real) ${sharp.length} at ${sharpAt.length}${sharpAt.length ? ': ' + sharpAt.slice(0, 6).join(', ') : ''}`);
  const ws = wrongSide(g, lay.x, lay.y, names);
  r.wrongSide = ws.wrong;
  line(ws.wrong.length <= WRONG_MAX[city], `${name}: at most ${WRONG_MAX[city]} near stations drawn on the wrong side of each other`, `${ws.wrong.length} of ${ws.pairs} pairs more than 135 degrees off their real bearing${ws.wrong.length ? ': ' + ws.wrong.slice(0, 4).join('; ') : ''}`);
  if (city === 'l8') {
    const l = line8(g, lay, names);
    line(l.ok, `${name}: line 8 runs east of Fleetwood, with no loop round it`, l.info);
  }
  line(r.movedMedian < 0.1, `${name}: one added station moves the median complex under a tenth of a hop`, `median ${r.movedMedian.toFixed(3)} hops (B ${r.movedBMedian.toFixed(3)}), 90th percentile ${r.movedP90.toFixed(3)}; rebuild ${(ms2 / 1000).toFixed(2)} s`);
}
mkdirSync(`${ARTIFACTS}t2160`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2160/check${OPTS === L.LAYOUT_R2 ? '-r2' : ''}.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
