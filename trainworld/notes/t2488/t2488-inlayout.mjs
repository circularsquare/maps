// T-2488 spike: the zone borders put into the layout itself, naively. Each border arc (borderLines.ts' arcs, the real
// boundaries of the zones with stations and those they share with neighbours) is sampled every SPACING median real
// hops and added to the graph as a line with no stops, so the layout places and routes it like any line. No anchors
// tie it to the stations beyond what the layout already has for any two nodes (side pairs, pushes apart). Measured
// against the laid out save without borders (T2464_CACHE): the layout's time, and the rail drawing alone: bends,
// length in pitches, and T-2405's hidden stretches.
//   node trainworld/checks/t2488-inlayout.mjs [nyc,ne,tor,jpn]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { deserialize } from 'node:v8';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing } from '@app/src/diagram/draw.ts';
export { lineBorders } from '@app/src/diagram/borderLines.ts'; export { toMerc01 } from '@app/src/map/geo.ts';
export { placeNames, guessWidth, splitName } from '@app/src/diagram/names.ts';`);
const TW = new URL('../', import.meta.url);
const which = (process.argv[2] ?? 'nyc,ne,tor,jpn').split(',');
const SPACING = Number(process.env.T2488_SPACING ?? 1);
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
const BORDER = '#b0b0b1';
const out = {};
mkdirSync(`${ARTIFACTS}diaglines`, { recursive: true });
const decode = (f) => { const o = []; let x = 0, y = 0; for (let i = 0; i < f.length; i += 2) o.push([(x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q]); return o; };

for (const city of which) {
  const C = deserialize(readFileSync(`${process.env.T2464_CACHE}/${city}.bin`));
  const { g, lay, shapes } = C;
  // the zones near, in the graph's frame (as t2488-borders.mjs)
  const s0 = shapes[0], v0 = decode(OUTL.zones[s0.id][0]);
  const [ax, ay] = L.toMerc01(...v0[0]);
  let bi = 1;
  while (Math.abs(L.toMerc01(...v0[bi])[0] - ax) < 1e-9) bi++;
  const [bx] = L.toMerc01(...v0[bi]), r0 = s0.rings[0];
  const k = (bx - ax) / (r0[bi * 2] - r0[0]), fx = ax - r0[0] * k, fy = ay + r0[1] * k;
  let w = Infinity, so = Infinity, e = -Infinity, nn = -Infinity;
  for (const s of shapes) { const b = REG.cities.find((c) => c.id === s.id).box; (w = Math.min(w, b[0])), (so = Math.min(so, b[1])), (e = Math.max(e, b[2])), (nn = Math.max(nn, b[3])); }
  const wide = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > e + 1.5 || b[2] < w - 1.5 || b[1] > nn + 1.5 || b[3] < so - 1.5) continue;
    wide.push({ id: c.id, rings: flat.map((f) => Float64Array.from(decode(f).flatMap(([lng, lat]) => { const [mx, my] = L.toMerc01(lng, lat); return [(mx - fx) / k, -(my - fy) / k]; }))) });
  }
  const arcs = [];
  L.lineBorders(g, lay.x, lay.y, C.d, wide, { arcsOut: arcs });
  // the arcs sampled every SPACING median real hops, junctions shared
  const n0 = g.key.length, m0 = g.ea.length;
  const hops = Float64Array.from({ length: m0 }, (_, q) => Math.hypot(g.x[g.eb[q]] - g.x[g.ea[q]], g.y[g.eb[q]] - g.y[g.ea[q]])).sort();
  const sp = SPACING * hops[m0 >> 1];
  const nx = [...g.x], ny = [...g.y], keys = [...g.key], at = new Map(), lines = [...g.lines], newE = [];
  const node = (px, py, shared) => {
    const kk = `${px},${py}`;
    if (shared && at.has(kk)) return at.get(kk);
    const id = nx.length;
    nx.push(px), ny.push(py), keys.push(1_500_000_000 + id);
    if (shared) at.set(kk, id);
    return id;
  };
  for (const A of arcs) {
    const P = A.pts, chain = [node(P[0], P[1], true)];
    let carry = 0;
    for (let q = 0; q + 3 < P.length; q += 2) {
      const sl = Math.hypot(P[q + 2] - P[q], P[q + 3] - P[q + 1]);
      let t = sp - carry;
      while (t < sl) {
        chain.push(node(P[q] + ((P[q + 2] - P[q]) * t) / sl, P[q + 1] + ((P[q + 3] - P[q + 1]) * t) / sl, false));
        t += sp;
      }
      carry = sl - (t - sp);
    }
    const last = node(P[P.length - 2], P[P.length - 1], true);
    if (chain[chain.length - 1] !== last) chain.push(last);
    if (chain.length < 2 || (chain.length === 2 && chain[0] === chain[1])) continue;
    const li = lines.length;
    lines.push({ id: 900000 + li, colour: BORDER, fast: false, chain: Int32Array.from(chain), stop: new Uint8Array(chain.length) });
    for (let q = 1; q < chain.length; q++) if (chain[q] !== chain[q - 1]) newE.push([Math.min(chain[q], chain[q - 1]), Math.max(chain[q], chain[q - 1]), li]);
  }
  // edges: the rail ones as they were, then the borders' (one line each; a hop two arcs share keeps both)
  const merged = new Map();
  for (const [a, b, li] of newE) { const kk = a * 1e7 + b; if (merged.has(kk)) merged.get(kk)[2].push(li); else merged.set(kk, [a, b, [li]]); }
  const ea = [...g.ea], eb = [...g.eb], first = [...g.first], on = [...g.on], stroke = [...g.stroke];
  for (const [a, b, ls] of merged.values()) {
    ea.push(a), eb.push(b);
    for (const li of ls) on.push(li), stroke.push(li);
    first.push(on.length);
  }
  const g2 = { ...g, key: Int32Array.from(keys), x: Float64Array.from(nx), y: Float64Array.from(ny), lines, ea: Int32Array.from(ea), eb: Int32Array.from(eb), first: Int32Array.from(first), on: Int32Array.from(on), stroke: Int32Array.from(stroke) };
  const t0 = performance.now();
  const lay2 = L.layoutDiagram(g2);
  const ms2 = performance.now() - t0;
  const d1 = C.d, d2 = L.drawing(g2, lay2);
  const r1 = rail(g, lay, d1, n0), r2 = rail(g2, lay2, d2, n0);
  out[city] = { borderNodes: nx.length - n0, borderEdges: merged.size, arcs: arcs.length, layoutMs: [Math.round(C.layoutMs), Math.round(ms2)], before: r1, after: r2, pitchRatio: lay2.pu / lay.pu };
  console.log(`${city}: +${nx.length - n0} border nodes (${n0} stations), layout ${(C.layoutMs / 1000).toFixed(1)} s -> ${(ms2 / 1000).toFixed(1)} s; rail before ${JSON.stringify(r1)}; after ${JSON.stringify(r2)}; pitch x${(lay2.pu / lay.pu).toFixed(3)}`);
  writeFileSync(`${ARTIFACTS}diaglines/inlayout.json`, JSON.stringify(out, null, 1));
  // a render file for t2488-shots.mjs (city `<city>-in`, variant `inlayout`): the border lines drawn as borders
  const text = [...C.names, ...Array(nx.length - n0).fill('')];
  const wd = Float32Array.from(text, (t) => L.guessWidth(t, 12)), w2 = Float32Array.from(text, (t) => { const s = L.splitName(t); return s ? Math.max(L.guessWidth(s[0], 12), L.guessWidth(s[1], 12)) : 0; });
  const plan = L.placeNames(g2, lay2, d2, text, wd, w2, 12);
  const keep = d2.lines.map((l) => l.colour !== BORDER), bcmds = d2.lines.filter((l) => l.colour === BORDER).flatMap((l) => [...l.cmds]);
  const enc = (v) => (ArrayBuffer.isView(v) ? [...v] : Array.isArray(v) ? v.map(enc) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([a, q]) => [a, enc(q)])) : v);
  const render = { pitch: d2.pitch, stroke: d2.stroke, box: d2.box, lines: d2.lines.filter((_, i) => keep[i]).map((l) => ({ colour: l.colour, fast: l.fast, cmds: [...l.cmds] })), pills: d2.pills, pillOf: d2.pillOf, passages: d2.passages, passageOf: d2.passageOf, small: [...d2.small], names: enc(plan), text, x: [...lay2.x], y: [...lay2.y], borders: { inlayout: bcmds } };
  writeFileSync(`${ARTIFACTS}diaglines/render-${city}-in.js`, `window.TW_DATA = window.TW_DATA || {}; window.TW_DATA[${JSON.stringify(city + '-in')}] = ${JSON.stringify(render)};`);
}

/** the rail drawing alone (pieces of stations only): pieces, bends, length in pitches, and T-2405's hidden and half
 * covered stretches (another rail line's middle within half a stroke, its own stroke's lines aside) */
function rail(g, lay, d, n0) {
  let pieces = 0, bends = 0, len = 0;
  for (const pc of lay.pieces) {
    if ([...pc.nodes].some((v) => v >= n0)) continue;
    pieces++;
    const v = pc.verts;
    for (let q = 0; q + 3 < v.length; q += 2) len += Math.hypot(v[q + 2] - v[q], v[q + 3] - v[q + 1]);
    for (let q = 2; q + 3 < v.length; q += 2) {
      const ax = v[q] - v[q - 2], ay = v[q + 1] - v[q - 1], bx = v[q + 2] - v[q], by = v[q + 3] - v[q + 1];
      if (Math.hypot(ax, ay) > 1e-9 && Math.hypot(bx, by) > 1e-9 && Math.abs(ax * by - ay * bx) > 1e-6 * Math.hypot(ax, ay) * Math.hypot(bx, by)) bends++;
    }
  }
  const n = g.key.length, W = d.stroke, near = 0.5 * W, step = 0.2 * d.pitch;
  const mates = new Map();
  for (const pc of lay.pieces) for (let i = 0; i + 1 < pc.nodes.length; i++) {
    const hk = Math.min(pc.nodes[i], pc.nodes[i + 1]) * n + Math.max(pc.nodes[i], pc.nodes[i + 1]);
    for (const grp of pc.strokes) for (const l of grp) mates.set(hk * 65536 + l, new Set(grp));
  }
  const isRail = (i) => g.lines[d.lineOf[i]].colour !== BORDER;
  const cell = 2 * d.pitch, grid = new Map();
  d.ribbons.forEach((r, i) => {
    if (!isRail(i)) return;
    for (let j = 0; j + 3 < r.length; j += 2) {
      for (let gx = Math.floor((Math.min(r[j], r[j + 2]) - near) / cell); gx <= Math.floor((Math.max(r[j], r[j + 2]) + near) / cell); gx++)
        for (let gy = Math.floor((Math.min(r[j + 1], r[j + 3]) - near) / cell); gy <= Math.floor((Math.max(r[j + 1], r[j + 3]) + near) / cell); gy++) {
          const kk = gx * 1e6 + gy;
          if (!grid.has(kk)) grid.set(kk, []);
          grid.get(kk).push(i, j);
        }
    }
  });
  const segDist = (px, py, r, j) => { const ax = r[j], ay = r[j + 1], dx = r[j + 2] - ax, dy = r[j + 3] - ay, l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0; return Math.hypot(ax + dx * t - px, ay + dy * t - py); };
  let stretches = 0, hidden = 0, half = 0;
  d.ribbons.forEach((r, i) => {
    if (!isRail(i)) return;
    const li = d.lineOf[i], ln = g.lines[li], at = d.chainAt[i];
    let j0 = 0;
    for (let j = 1; j <= at.length; j++) {
      if (j < at.length && at[j] === at[j0]) continue;
      const c = at[j0], hopEnd = Math.min(j, at.length - 1);
      if (c + 1 < ln.chain.length) {
        const a = ln.chain[c], b = ln.chain[c + 1], allowed = mates.get((Math.min(a, b) * n + Math.max(a, b)) * 65536 + li);
        let L2 = 0, cov = 0;
        for (let q = j0; q < hopEnd; q++) {
          const x0 = r[q * 2], y0 = r[q * 2 + 1], x1 = r[q * 2 + 2], y1 = r[q * 2 + 3], sl = Math.hypot(x1 - x0, y1 - y0), kq = Math.max(1, Math.ceil(sl / step));
          for (let s = 0; s < kq; s++) {
            const f = (s + 0.5) / kq, px = x0 + (x1 - x0) * f, py = y0 + (y1 - y0) * f;
            L2 += sl / kq;
            const v = grid.get(Math.floor(px / cell) * 1e6 + Math.floor(py / cell));
            let hit = false;
            if (v) for (let q2 = 0; q2 < v.length && !hit; q2 += 2) if (v[q2] !== i && !allowed?.has(d.lineOf[v[q2]]) && segDist(px, py, d.ribbons[v[q2]], v[q2 + 1]) < near) hit = true;
            if (hit) cov += sl / kq;
          }
        }
        if (L2 > 1e-9 * d.pitch) {
          stretches++;
          if (cov >= 0.9 * L2) hidden++;
          if (cov >= 0.5 * L2) half++;
        }
      }
      j0 = j;
    }
  });
  return { pieces, bends, lengthPitches: Math.round(len / lay.pu), stretches, hidden, half };
}
