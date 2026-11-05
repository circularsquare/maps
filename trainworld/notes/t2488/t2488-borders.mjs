// T-2488 spike (notes/T-2488.md): the network diagram's zone borders three ways on laid out saves (T-2464's cache,
// T2464_CACHE=<folder>, which this needs): T-2464's as landed ("now"), T-2464's with a wider margin and closing
// ("wide", Anita's fallback) and the real boundaries routed like lines on the drawing ("lines", diagram/borderLines.ts).
// Each is checked as t2464-borders.mjs does (closed, stations on their own zone's side, octilinear), and the figures and
// a render file (drawing, names, the three borders) are written to artifacts/trainworld/diaglines/.
//   node trainworld/checks/t2488-borders.mjs [nyc,ne,tor,jpn] [variants: now,wide,lines]
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { deserialize } from 'node:v8';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export { pillSize } from '@app/src/diagram/draw.ts';
export { zoneBorders } from '@app/src/diagram/borders.ts'; export { lineBorders } from '@app/src/diagram/borderLines.ts';
export { placeNames, guessWidth, splitName } from '@app/src/diagram/names.ts'; export { toMerc01 } from '@app/src/map/geo.ts';`);
const TW = new URL('../', import.meta.url);
const which = (process.argv[2] ?? 'nyc,ne,tor,jpn').split(',');
const variants = (process.argv[3] ?? 'now,wide,lines').split(',');
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
const CACHE = process.env.T2464_CACHE;
const OUT = `${ARTIFACTS}diaglines/`;
mkdirSync(OUT, { recursive: true });
/** the fallback: the region 5 pitches past the casings (2 now), gaps and bays under about 60 pitches closed (8), the
 * edge fitted within 3 pitches (0.8), corners of 3 */
const WIDE = { margin: 5, close: 30, tolEdge: 3, radius: 3 };
const decode = (f) => {
  const out = [];
  let x = 0, y = 0;
  for (let i = 0; i < f.length; i += 2) out.push([(x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q]);
  return out;
};
const all = {};
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const city of which) {
  const C = deserialize(readFileSync(`${CACHE}/${city}.bin`));
  const { g, lay, d, shapes, names } = C;
  // the graph's frame from a cached shape: graph metres = ((mx - fx) / k, -(my - fy) / k)
  const s0 = shapes[0], v0 = decode(OUTL.zones[s0.id][0]);
  const [ax, ay] = L.toMerc01(...v0[0]);
  let bi = 1;
  while (Math.abs(L.toMerc01(...v0[bi])[0] - ax) < 1e-9) bi++;
  const [bx] = L.toMerc01(...v0[bi]), r0 = s0.rings[0];
  const k = (bx - ax) / (r0[bi * 2] - r0[0]), fx = ax - r0[0] * k, fy = ay + r0[1] * k;
  // the zones near: boxes within about 1.5 degrees of the cached ones', so neighbours have their whole outlines
  let w = Infinity, so = Infinity, e = -Infinity, nn = -Infinity;
  for (const s of shapes) {
    const b = REG.cities.find((c) => c.id === s.id).box;
    (w = Math.min(w, b[0])), (so = Math.min(so, b[1])), (e = Math.max(e, b[2])), (nn = Math.max(nn, b[3]));
  }
  const wide = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > e + 1.5 || b[2] < w - 1.5 || b[1] > nn + 1.5 || b[3] < so - 1.5) continue;
    wide.push({ id: c.id, rings: flat.map((f) => Float64Array.from(decode(f).flatMap(([lng, lat]) => { const [mx, my] = L.toMerc01(lng, lat); return [(mx - fx) / k, -(my - fy) / k]; }))) });
  }
  // names, placed with guessed widths (the page measures them in its font)
  const fontPx = 12, text = names, wd = Float32Array.from(text, (t) => L.guessWidth(t, fontPx)), w2 = Float32Array.from(text, (t) => { const s = L.splitName(t); return s ? Math.max(L.guessWidth(s[0], fontPx), L.guessWidth(s[1], fontPx)) : 0; });
  const plan = L.placeNames(g, lay, d, text, wd, w2, fontPx);
  const res = { stations: 0, layoutMs: Math.round(C.layoutMs), zonesShapes: shapes.length, zonesWide: wide.length };
  const render = { pitch: d.pitch, stroke: d.stroke, box: d.box, lines: d.lines.map((l) => ({ colour: l.colour, fast: l.fast, cmds: [...l.cmds] })), pills: d.pills, pillOf: d.pillOf, passages: d.passages, passageOf: d.passageOf, small: [...d.small], ribbons: [], names: enc(plan), text, x: [...lay.x], y: [...lay.y], borders: {} };
  let dbg = [];
  for (const v of variants) {
    const t0 = performance.now();
    const b = v === 'now' ? L.zoneBorders(g, lay.x, lay.y, d, shapes, true) : v === 'wide' ? L.zoneBorders(g, lay.x, lay.y, d, shapes, true, WIDE) : L.lineBorders(g, lay.x, lay.y, d, wide, { ...JSON.parse(process.env.T2488_OPTS ?? "{}"), debug: (dbg = []) });
    const ms = performance.now() - t0;
    const f = figures(g, lay, d, b, names);
    f.ms = Math.round(ms);
    f.zonesDrawn = b.zones.length;
    if (b.stats) f.stats = { ...b.stats, ms: Object.fromEntries(Object.entries(b.stats.ms).map(([a, q]) => [a, Math.round(q)])) };
    res[v] = f;
    res.stations = f.stations;
    render.borders[v] = [...b.cmds];
    if (v === "lines") { render.borders.guide = dbg; const per = {}; b.zoneOf.forEach((z) => { if (z >= 0) per[b.zones[z]] = (per[b.zones[z]] ?? 0) + 1; }); console.log("     zones:", JSON.stringify(per)); }
    console.log(`     ${city} ${v}: ${f.stations} stations, ${f.zonesDrawn} zones; ${f.runs} runs, ${f.legs} legs, ${f.bends} bends, ${Math.round(f.len / d.pitch)} pitches long; ${f.touched} pills touched; mean clearance ${f.clear.toFixed(2)} pitches; ${ms.toFixed(0)} ms${b.stats ? ' ' + JSON.stringify(f.stats) : ''}`);
    if (f.wrongs.length) console.log(`     wrong: ${f.wrongs.join(', ')}`);
    say(f.open.length === 0, `${city} ${v}: every zone closed`, f.open.join(', '));
    say(f.wrong === 0, `${city} ${v}: every station inside its own zone's borders and no other's`, `${f.wrong} of ${f.stations} wrong (${f.wrongBetween} between blocks)`);
    say(f.offAngle === 0, `${city} ${v}: every run at 0, 45 or 90 degrees`, f.offAngle ? `${f.offAngle} legs off` : '');
  }
  all[city] = res;
  writeFileSync(`${OUT}render-${city}.js`, `window.TW_DATA = window.TW_DATA || {}; window.TW_DATA[${JSON.stringify(city)}] = ${JSON.stringify(render)};`);
}
const old = existsSync(`${OUT}figures.json`) ? JSON.parse(readFileSync(`${OUT}figures.json`)) : {};
writeFileSync(`${OUT}figures.json`, JSON.stringify({ ...old, ...all }, null, 1));
process.exit(failed ? 1 : 0);

/** typed arrays in a name plan as plain arrays */
function enc(v) {
  if (ArrayBuffer.isView(v)) return [...v];
  if (Array.isArray(v)) return v.map(enc);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([a, q]) => [a, enc(q)]));
  return v;
}

/** a run's commands as a polyline, curves sampled */
function flat(c) {
  const p = [];
  let x = 0, y = 0;
  for (let i = 0; i < c.length; ) {
    if (c[i] === 2) {
      for (let s = 1; s <= 8; s++) {
        const u = s / 8, a = (1 - u) * (1 - u), m = 2 * (1 - u) * u, q = u * u;
        p.push(a * x + m * c[i + 1] + q * c[i + 3], a * y + m * c[i + 2] + q * c[i + 4]);
      }
      (x = c[i + 3]), (y = c[i + 4]), (i += 5);
    } else {
      (x = c[i + 1]), (y = c[i + 2]), (i += 3);
      p.push(x, y);
    }
  }
  return p;
}

/** t2464-borders.mjs' figures, plus each run's bends and length and the borders' mean clearance from the casings
 * (sampled each pitch, up to 6 pitches; the stretches within 4 pitches of a crossing left out) */
function figures(g, lay, d, b, names) {
  const n = g.key.length, nz = b.zones.length, runs = b.runs;
  const polys = runs.map((r) => flat(b.cmds.subarray(r.c[0], r.c[1])));
  const open = [];
  const withSt = new Uint8Array(nz);
  for (let i = 0; i < n; i++) if (b.zoneOf[i] >= 0) withSt[b.zoneOf[i]] = 1;
  for (let z = 0; z < nz; z++) {
    if (!withSt[z]) continue;
    const ends = new Map();
    runs.forEach((r) => {
      if (r.closed || (r.left !== z && r.right !== z)) return;
      for (const q of [0, r.p.length - 2]) {
        const key = `${r.p[q].toFixed(6)},${r.p[q + 1].toFixed(6)}`;
        ends.set(key, (ends.get(key) ?? 0) + 1);
      }
    });
    if ([...ends.values()].some((c) => c % 2)) open.push(b.zones[z]);
  }
  const segs = [], rows = new Map(), cells = new Map(), RH = 4 * d.pitch;
  const add = (m, key, s) => (m.has(key) ? m.get(key).push(s) : m.set(key, [s]));
  polys.forEach((p, q) => {
    for (let k = 0; k + 3 < p.length; k += 2) {
      const s = segs.length;
      segs.push([p[k], p[k + 1], p[k + 2], p[k + 3], runs[q].left, runs[q].right]);
      for (let j = Math.floor(Math.min(p[k + 1], p[k + 3]) / RH); j <= Math.floor(Math.max(p[k + 1], p[k + 3]) / RH); j++) {
        add(rows, j, s);
        for (let i = Math.floor(Math.min(p[k], p[k + 2]) / RH); i <= Math.floor(Math.max(p[k], p[k + 2]) / RH); i++) add(cells, `${i},${j}`, s);
      }
    }
  });
  const insideOf = (px, py) => {
    const c = new Uint8Array(nz);
    for (const s of rows.get(Math.floor(py / RH)) ?? []) {
      const [x0, y0, x1, y1, l, r] = segs[s];
      if (y0 > py !== y1 > py && px < ((x1 - x0) * (py - y0)) / (y1 - y0) + x0) {
        if (l >= 0) c[l] ^= 1;
        if (r >= 0) c[r] ^= 1;
      }
    }
    return c;
  };
  const segD = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
    return Math.hypot(px - ax - dx * t, py - ay - dy * t);
  };
  let stations = 0, wrong = 0, wrongBetween = 0, between = 0, touched = 0;
  const wrongs = [];
  const pillR = new Float64Array(n), px = new Float64Array(n), py = new Float64Array(n);
  d.pills.forEach((p, q) => {
    const k = d.pillOf[q], [r, o] = L.pillSize(d, k);
    if (!pillR[k]) (px[k] = (p[0] + p[2]) / 2), (py[k] = (p[1] + p[3]) / 2);
    pillR[k] = Math.max(pillR[k], r + o);
  });
  for (let i = 0; i < n; i++) {
    const z = b.zoneOf[i];
    if (z < 0 || !pillR[i]) continue;
    stations++;
    const btw = lay.blocks.list.length > 1 && lay.blocks.of[i] < 0;
    if (btw) between++;
    const c = insideOf(px[i], py[i]);
    let bad = !c[z];
    for (let o = 0; o < nz && !bad; o++) if (o !== z && withSt[o] && c[o]) bad = true;
    if (bad) {
      wrong++;
      if (btw) wrongBetween++;
      if (wrongs.length < 12) wrongs.push(`${names[i]} (${b.zones[z]})`);
    }
    let near = Infinity;
    const ci = Math.floor(px[i] / RH), cj = Math.floor(py[i] / RH);
    for (let gi = ci - 1; gi <= ci + 1; gi++) for (let gj = cj - 1; gj <= cj + 1; gj++) for (const s of cells.get(`${gi},${gj}`) ?? []) {
      const [x0, y0, x1, y1] = segs[s];
      near = Math.min(near, segD(px[i], py[i], x0, y0, x1, y1));
    }
    if (near < pillR[i]) touched++;
  }
  let legs = 0, offAngle = 0, bends = 0, len = 0;
  for (const r of runs) {
    for (let k = 0; k + 3 < r.p.length; k += 2) {
      const dx = Math.abs(r.p[k + 2] - r.p[k]), dy = Math.abs(r.p[k + 3] - r.p[k + 1]), l = Math.hypot(dx, dy);
      legs++;
      len += l;
      if (Math.min(dx, dy) > 1e-6 * l && Math.abs(dx - dy) > 1e-6 * l) offAngle++;
    }
    bends += Math.max(0, r.p.length / 2 - 2) + (r.closed ? 1 : 0);
  }
  // clearance from the casings along the borders: per pitch of border, the distance past the nearest casing (up to 6)
  const P = d.pitch, CAP = 6 * P, rs = new Map(), RC = 8 * P;
  d.ribbons.forEach((r) => {
    for (let k = 0; k + 3 < r.length; k += 2) {
      const i0 = Math.floor((Math.min(r[k], r[k + 2]) - CAP) / RC), i1 = Math.floor((Math.max(r[k], r[k + 2]) + CAP) / RC);
      const j0 = Math.floor((Math.min(r[k + 1], r[k + 3]) - CAP) / RC), j1 = Math.floor((Math.max(r[k + 1], r[k + 3]) + CAP) / RC);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) add(rs, `${i},${j}`, [r[k], r[k + 1], r[k + 2], r[k + 3]]);
    }
  });
  const clearAt = (qx, qy) => {
    let best = CAP;
    for (const [x0, y0, x1, y1] of rs.get(`${Math.floor(qx / RC)},${Math.floor(qy / RC)}`) ?? []) best = Math.min(best, segD(qx, qy, x0, y0, x1, y1) - P / 2);
    return best;
  };
  let cs = 0, cn = 0, under1 = 0;
  for (const p of polys) {
    for (let k = 0; k + 3 < p.length; k += 2) {
      const sl = Math.hypot(p[k + 2] - p[k], p[k + 3] - p[k + 1]), parts = Math.ceil(sl / P);
      for (let q = 0; q < parts; q++) {
        const f = (q + 0.5) / parts, c = clearAt(p[k] + (p[k + 2] - p[k]) * f, p[k + 1] + (p[k + 3] - p[k + 1]) * f);
        // a crossing: within the casing; the stretch either side of one, nearer than a pitch, is the crossing too
        if (c <= 0) continue;
        if (c >= CAP) continue;
        cs += c / P, cn++;
        if (c < P) under1++;
      }
    }
  }
  return { stations, between, wrong, wrongBetween, wrongs, open, touched, runs: runs.length, legs, offAngle, bends, len, clear: cn ? cs / cn : 0, nearSamples: cn, under1 };
}
