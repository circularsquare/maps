// T-2580: the rubber sheet's water and zone borders (diagram/sheetLines.ts `sheetMap`, diagram/sheetThin.ts) on the
// real New York save and all Japan, run as the page's worker does with the water on: the sheet, the lines on it, the
// rivers on the ground (a city map's narrow rivers), the map. Checked: every edge of the drawn water (the sea's coast,
// the rivers' bands) runs at 0, 45 or 90 degrees, as t2169-water.mjs asks of the flag-off water; no stair steps (a run
// under 4 pitches between turns of opposite sign, next to another such: under STAIRS per 1000 pitches of coast and of
// border); every station on land, its own zone's side and under no band; in New York, the Harlem River drawn; land
// drawn as land and sea as sea away from coasts (T-2591, `landSea`). Reported: jogs, rivers' crossings drawn and lines a
// band crosses where its river does not, and the times. The masks are the page's: the zones the sheet's box meets.
// `far` is New York with a line added out to Binghamton, a network reaching far past its first zones (T-2591).
//   node trainworld/checks/t2580-sheet-water.mjs [nyc,jpn,far]
import { readFileSync } from 'node:fs';
import { openSave, waterOf, masksOf, app as A } from './diagram-saves.mjs';
import { loadApp } from './lib.mjs';

const L = await loadApp(`export { rubberSheet, massAt, sheetZoneBox } from '@app/src/diagram/sheet.ts'; export { sheetLines, sheetMap, riverCellM, SHEET_LINES, carryMap, cutLand } from '@app/src/diagram/sheetLines.ts';
export { realWater } from '@app/src/diagram/waterReal.ts'; export { inRings, stationZones } from '@app/src/diagram/borders.ts'; export { toMerc01 } from '@app/src/map/geo.ts';`);
const TW = new URL('../', import.meta.url);
const which = (process.argv[2] ?? 'nyc,jpn').split(',');
/** stair runs allowed per 1000 pitches of coast or border (before T-2580 New York had 82) */
const STAIRS = 10;
/** the biggest group of samples drawn wrong (land as sea, or sea as land) may cover this many squares of the samples'
 * distance from coasts (T-2591) */
const FALSE_KM2 = 25;
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));

/** the zones near `box` (lng, lat) in the graph's frame, as the page sends them */
function shapesNear(box, [fx, fy, fk]) {
  const out = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > box[2] + 0.5 || b[2] < box[0] - 0.5 || b[1] > box[3] + 0.5 || b[3] < box[1] - 0.5) continue;
    out.push({ id: c.id, rings: flat.map((f) => { const r = new Float64Array(f.length); let x = 0, y = 0; for (let i = 0; i < f.length; i += 2) { const [mx, my] = L.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q); (r[i] = (mx - fx) * fk), (r[i + 1] = -(my - fy) * fk); } return r; }) });
  }
  return out;
}
/** path commands (0 move, 1 line, 2 quadratic) as rings of points */
const flat = (cmd) => {
  const rings = [];
  let cur = null;
  for (let i = 0; i < cmd.length; ) {
    if (cmd[i] === 0) (cur = [cmd[i + 1], cmd[i + 2]]), rings.push(cur), (i += 3);
    else if (cmd[i] === 1) cur.push(cmd[i + 1], cmd[i + 2]), (i += 3);
    else { const ax = cur[cur.length - 2], ay = cur[cur.length - 1]; for (let t = 1; t <= 6; t++) { const u = t / 6; cur.push((1 - u) ** 2 * ax + 2 * u * (1 - u) * cmd[i + 1] + u * u * cmd[i + 3], (1 - u) ** 2 * ay + 2 * u * (1 - u) * cmd[i + 2] + u * u * cmd[i + 4]); } i += 5; }
  }
  return rings;
};
/** runs of path commands off 0, 45 and 90 degrees (corners' curves not counted), as t2169-water.mjs counts them */
function offOcti(cmd, P) {
  let px = 0, py = 0, segs = 0;
  const at = [];
  for (let i = 0; i < cmd.length; ) {
    if (cmd[i] === 2) { (px = cmd[i + 3]), (py = cmd[i + 4]), (i += 5); continue; }
    const x = cmd[i + 1], y = cmd[i + 2];
    if (cmd[i] === 1 && Math.hypot(x - px, y - py) > 0.02 * P) {
      segs++;
      const a = ((Math.atan2(y - py, x - px) * 4) / Math.PI) % 1;
      if (Math.min(Math.abs(a), 1 - Math.abs(a)) > 0.005) at.push(`${(x / P).toFixed(0)},${(y / P).toFixed(0)}`);
    }
    (px = x), (py = y), (i += 3);
  }
  return { segs, at };
}
/** jogs (a run under 4 pitches between turns of opposite sign) and stair runs (a jog next to a jog) in fitted polylines */
function stairs(polys, closed, P) {
  let len = 0, jogs = 0, stair = 0;
  for (const v of polys) {
    const pts = [];
    for (let k = 0; k < v.length; k += 2) pts.push([v[k], v[k + 1]]);
    if (closed && pts.length > 2 && Math.hypot(pts[0][0] - pts.at(-1)[0], pts[0][1] - pts.at(-1)[1]) < 1e-9) pts.pop();
    const m = pts.length, nr = closed ? m : m - 1;
    if (m < 2) continue;
    const rl = [], turn = [], jog = new Uint8Array(nr);
    for (let k = 0; k < nr; k++) { const a = pts[k], b = pts[(k + 1) % m]; rl.push(Math.hypot(b[0] - a[0], b[1] - a[1]) / P); }
    for (let k = 0; k < nr; k++) {
      if (!closed && k === 0) { turn.push(0); continue; }
      const a = pts[(k - 1 + m) % m], b = pts[k], c = pts[(k + 1) % m];
      turn.push(Math.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])));
    }
    for (let k = 0; k < nr; k++) {
      len += rl[k];
      const tOut = closed || k + 1 < nr ? turn[(k + 1) % nr] : 0;
      if (rl[k] < 4 && turn[k] && tOut && turn[k] !== tOut) (jog[k] = 1), jogs++;
    }
    for (let k = 0; k < nr; k++) if (jog[k] && ((closed || k + 1 < nr ? jog[(k + 1) % nr] : 0) || (closed || k > 0 ? jog[(k - 1 + nr) % nr] : 0))) stair++;
  }
  return { pitches: Math.round(len), jogs, stairRuns: stair, per1000: +((1000 * stair) / Math.max(1, len)).toFixed(2) };
}
/** T-2591: the drawn land and sea against the ground the sheet read from the masks, at sample points SAMPLE_RHO from
 * any coast and from the masks' edge (at least 1 km, 8 raster cells), each carried to its drawn place. Land with no
 * station that the masks' edge cuts (cutLand) is drawn as sea on purpose and only counted. Wrong samples are grouped
 * (side by side on the samples' lattice); the biggest group's area on the ground is what fails. */
function landSea(city, lines, M, frame, K) {
  const C = lines.carry, R = C.raster, cm = L.carryMap(C), { cut } = L.cutLand(C), G = M.grid;
  const rho = Math.max(1000, 8 * R.rr), step = rho / 2;
  const covAt = (x, y) => {
    const i = Math.floor((x - R.x0) / R.rr), j = Math.floor((y - R.y0) / R.rr);
    if (i < 0 || j < 0 || i >= R.w || j >= R.h) return false;
    for (let t = R.covStart[j]; t < R.covStart[j + 1]; t++) if (i >= R.covA[t] && i < R.covB[t]) return true;
    return false;
  };
  // the drawn sea on the drawn grid's cells, from its fitted rings (even-odd, as the page fills them)
  const sea = new Uint8Array(G.w * G.h), rows = Array.from({ length: G.h }, () => []);
  for (const r of flat(M.water.sea)) for (let a = 0, m = r.length; a < m; a += 2) {
    const b = (a + 2) % m, ax = r[a] / K, ay = r[a + 1] / K, bx = r[b] / K, by = r[b + 1] / K;
    if (ay === by) continue;
    const j0 = Math.max(0, Math.ceil((Math.min(ay, by) - G.y0) / G.cell - 0.5)), j1 = Math.min(G.h - 1, Math.floor((Math.max(ay, by) - G.y0) / G.cell - 0.5));
    for (let j = j0; j <= j1; j++) { const y = G.y0 + (j + 0.5) * G.cell; if ((ay > y) !== (by > y)) rows[j].push(ax + ((y - ay) * (bx - ax)) / (by - ay)); }
  }
  rows.forEach((xs, j) => { xs.sort((a, b) => a - b); for (let k = 0; k + 1 < xs.length; k += 2) for (let i = Math.max(0, Math.ceil((xs[k] - G.x0) / G.cell - 0.5)); i <= Math.min(G.w - 1, Math.floor((xs[k + 1] - G.x0) / G.cell - 0.5)); i++) sea[j * G.w + i] = 1; });
  const SW = Math.floor((R.w * R.rr) / step), SH = Math.floor((R.h * R.rr) / step), bad = new Int8Array(SW * SH);
  let land = 0, water = 0, cutSea = 0;
  for (let sj = 0; sj < SH; sj++) for (let si = 0; si < SW; si++) {
    const x = R.x0 + (si + 0.5) * step, y = R.y0 + (sj + 0.5) * step, m = L.massAt(R, x, y);
    let away = covAt(x, y);
    for (let dj = -2; dj <= 2 && away; dj++) for (let di = -2; di <= 2 && away; di++) { const px = x + (di * rho) / 2, py = y + (dj * rho) / 2; away = covAt(px, py) && L.massAt(R, px, py) > 0 === m > 0; }
    if (!away) continue;
    const [px, py] = cm.at(x, y), gi = Math.floor((px - G.x0) / G.cell), gj = Math.floor((py - G.y0) / G.cell);
    if (gi < 0 || gj < 0 || gi >= G.w || gj >= G.h) continue;
    const drawnSea = sea[gj * G.w + gi] === 1;
    if (m && cut[m - 1]) { if (drawnSea) cutSea++; continue; }
    m ? land++ : water++;
    if (m && drawnSea) bad[sj * SW + si] = 1;
    else if (!m && !drawnSea) bad[sj * SW + si] = 2;
  }
  // groups of wrong samples, 4 neighbours on the lattice: the biggest of each kind, its area and middle
  const [fx, fy, fk] = frame, lngLat = (x, y) => { const mx = x / fk + fx, my = fy - y / fk; return `${(mx * 360 - 180).toFixed(2)},${((Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI).toFixed(2)}`; };
  const seen = new Uint8Array(SW * SH), big = { 1: { n: 0, at: '' }, 2: { n: 0, at: '' } }, all = { 1: 0, 2: 0 };
  for (let c = 0; c < SW * SH; c++) {
    if (!bad[c] || seen[c]) continue;
    const kind = bad[c], q = [c];
    seen[c] = 1;
    let sx = 0, sy = 0;
    for (let k = 0; k < q.length; k++) {
      const v = q[k], i = v % SW, j = (v - i) / SW;
      (sx += i), (sy += j);
      for (const w of [i > 0 ? v - 1 : -1, i + 1 < SW ? v + 1 : -1, j > 0 ? v - SW : -1, j + 1 < SH ? v + SW : -1]) if (w >= 0 && !seen[w] && bad[w] === kind) (seen[w] = 1), q.push(w);
    }
    all[kind] += q.length;
    if (q.length > big[kind].n) big[kind] = { n: q.length, at: lngLat(R.x0 + (sx / q.length + 0.5) * step, R.y0 + (sy / q.length + 0.5) * step) };
  }
  const km2 = (k) => Math.round((k * step * step) / 1e6);
  return { rhoKm: rho / 1000, land, water, falseSea: all[1], falseLand: all[2], biggestFalseSeaKm2: km2(big[1].n), biggestFalseSeaAt: big[1].at, biggestFalseLandKm2: km2(big[2].n), biggestFalseLandAt: big[2].at, cutDrawnSeaKm2: km2(cutSea), cutKm2: M.info.cutKm2 };
}

const segDist = (px, py, ax, ay, bx, by) => { const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0; return Math.hypot(px - ax - t * dx, py - ay - t * dy); };
const crosses = (a, b) => {
  for (let i = 0; i + 3 < a.length; i += 2) for (let j = 0; j + 3 < b.length; j += 2) {
    const d1 = (a[i + 2] - a[i]) * (b[j + 1] - a[i + 1]) - (a[i + 3] - a[i + 1]) * (b[j] - a[i]), d2 = (a[i + 2] - a[i]) * (b[j + 3] - a[i + 1]) - (a[i + 3] - a[i + 1]) * (b[j + 2] - a[i]);
    const d3 = (b[j + 2] - b[j]) * (a[i + 1] - b[j + 1]) - (b[j + 3] - b[j + 1]) * (a[i] - b[j]), d4 = (b[j + 2] - b[j]) * (a[i + 3] - b[j + 1]) - (b[j + 3] - b[j + 1]) * (a[i + 2] - b[j]);
    if (d1 * d2 < 0 && d3 * d4 < 0) return true;
  }
  return false;
};

/** `g` with a line from its station nearest [lng0, lat0] out to [lng1, lat1], a station every `stepM` (graph metres
 * through `frame`) */
function withLine(g, frame, [lng0, lat0], [lng1, lat1], stepM) {
  const [fx, fy, fk] = frame, xy = (lng, lat) => { const [mx, my] = L.toMerc01(lng, lat); return [(mx - fx) * fk, (fy - my) * fk]; };
  const [ax, ay] = xy(lng0, lat0), [bx, by] = xy(lng1, lat1), n = g.key.length;
  let from = 0;
  for (let i = 1; i < n; i++) if (Math.hypot(g.x[i] - ax, g.y[i] - ay) < Math.hypot(g.x[from] - ax, g.y[from] - ay)) from = i;
  const m = Math.max(1, Math.round(Math.hypot(bx - g.x[from], by - g.y[from]) / stepM)), ln = g.lines.length;
  const X = [...g.x], Y = [...g.y], key = [...g.key], chain = [from], hop = [...g.hop], hopAt = [...g.hopAt], ea = [...g.ea], eb = [...g.eb], first = [...g.first], on = [...g.on], stroke = [...g.stroke];
  let top = key.reduce((a, b) => Math.max(a, b), 0);
  for (let k = 1; k <= m; k++) {
    const i = n + k - 1, a = chain.at(-1);
    X.push(g.x[from] + ((bx - g.x[from]) * k) / m), Y.push(g.y[from] + ((by - g.y[from]) * k) / m), key.push(++top), chain.push(i);
    ea.push(a), eb.push(i), on.push(ln), stroke.push(ln), first.push(on.length), hop.push(X[a], Y[a], X[i], Y[i]), hopAt.push(hop.length / 2);
  }
  return {
    ...g, key: Int32Array.from(key), x: Float64Array.from(X), y: Float64Array.from(Y), ea: Int32Array.from(ea), eb: Int32Array.from(eb), first: Int32Array.from(first), on: Int32Array.from(on), stroke: Int32Array.from(stroke), hop: Float32Array.from(hop), hopAt: Int32Array.from(hopAt),
    lines: [...g.lines, { id: 1e6, colour: '#808080', chain: Int32Array.from(chain), stop: new Uint8Array(chain.length).fill(1) }],
  };
}

for (const city of which) {
  const s = openSave(city === 'far' ? 'nyc' : city), { g: g0 } = s.graph(), { frame, box: box0 } = waterOf(s, g0);
  s.api.free();
  const g = city === 'far' ? withLine(g0, frame, [-74.9, 40.9], [-75.92, 42.1], 4000) : g0;
  // the stations' box, then the masks of the zones the sheet's box meets, as the page reads them
  let box = box0;
  if (city === 'far') box = [Math.min(box0[0], -75.92), box0[1], box0[2], Math.max(box0[3], 42.1)];
  const zb = L.sheetZoneBox(box), masks = masksOf(REG.cities.filter(({ box: b }) => b && zb[0] <= b[2] && zb[2] >= b[0] && zb[1] <= b[3] && zb[3] >= b[1]).map((c) => c.id));
  const sea = A.groundSea(g, masks, frame);
  const t0 = performance.now();
  const sheet = L.rubberSheet(g, { masks, frame, sea });
  const lines = L.sheetLines(g, sheet);
  const t1 = performance.now();
  const cell = L.riverCellM(g), real = cell ? L.realWater(g, masks, frame, sea, cell) : null;
  const t2 = performance.now();
  const M = L.sheetMap(lines, g, shapesNear(box, frame), true, L.SHEET_LINES, true, real);
  const t3 = performance.now();
  const n = g.key.length, lay = lines.layout, K = lines.k, W = M.water;
  console.log(`     ${city}: ${n} stations; sheet and lines ${Math.round(t1 - t0)} ms, rivers on the ground ${Math.round(t2 - t1)} ms, map ${Math.round(t3 - t2)} ms; ${JSON.stringify(M.info)}`);
  // edges
  const coast = offOcti(W.sea, K), riv = W.rivers.map((c) => offOcti(c, K));
  const off = [...coast.at, ...riv.flatMap((r) => r.at)];
  say(off.length === 0, `${city}: every water edge at 0, 45 or 90 degrees`, `${off.length} of ${coast.segs + riv.reduce((t, r) => t + r.segs, 0)} runs off ${off.slice(0, 4).join('; ')}`);
  const st = { coast: stairs(W.seaRings, true, K), borders: stairs(M.borderFits, false, K), rivers: stairs((M.riverFits ?? []).map((f) => Float64Array.from(f.fit, (v) => v * K)), false, K) };
  say(st.coast.per1000 < STAIRS && st.borders.per1000 < STAIRS && st.rivers.per1000 < STAIRS, `${city}: no stair steps in the coast, the rivers or the borders`, JSON.stringify(st));
  // stations
  const seaRings = flat(W.sea);
  const wet = [], under = [];
  for (let i = 0; i < n; i++) {
    if (L.inRings(seaRings, lay.x[i], lay.y[i])) wet.push(i);
    const x = lay.x[i] / K, y = lay.y[i] / K;
    if ((M.riverFits ?? []).some(({ fit, w }) => { for (let q = 0; q + 3 < fit.length; q += 2) if (segDist(x, y, fit[q], fit[q + 1], fit[q + 2], fit[q + 3]) < w / 2) return true; return false; })) under.push(i);
  }
  say(wet.length === 0 && under.length === 0 && M.info.offLand === 0, `${city}: every station on its own land, none in the sea or under a river`, `${wet.length} in the sea, ${under.length} under a band, ${M.info.offLand} off their land in the carried map`);
  // a station whose cell also holds a station of another zone cannot be on its own side: one cell takes one zone, so that
  // pair is reported (a game limit of the drawn grid) and the rest must be none
  {
    const shapes = shapesNear(box, frame), zone = L.stationZones(g, shapes), G0 = M.grid, C = lines.carry, cellOf = (k) => Math.floor((C.sy[k] - G0.y0) / G0.cell) * G0.w + Math.floor((C.sx[k] - G0.x0) / G0.cell);
    const byCell = new Map();
    for (let k = 0; k < n; k++) { const c = cellOf(k); if (c >= 0 && Math.floor((C.sx[k] - G0.x0) / G0.cell) < G0.w) (byCell.get(c) ?? byCell.set(c, []).get(c)).push(k); }
    let shared = 0, other = 0;
    for (let i = 0; i < n; i++) {
      const c = cellOf(i);
      if (zone[i] < 0 || !(c >= 0 && c < G0.w * G0.h) || G0.d[c] === zone[i] + 1) continue;
      const lab = G0.d[c], mx = g.x[i] / frame[2] + frame[0], my = frame[1] - g.y[i] / frame[2], mates = (byCell.get(c) ?? []).filter((k) => k !== i && zone[k] >= 0 && zone[k] !== zone[i]);
      mates.length ? shared++ : other++;
      console.log(`     off its zone: station ${i} at ${(mx * 360 - 180).toFixed(3)},${((Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI).toFixed(3)}, zone ${shapes[zone[i]].id}, its cell is ${shapes[lab - 1]?.id ?? (lab === 0 ? 'water' : 'none')}${mates.length ? `, shared with station ${mates.join()} of ${mates.map((k) => shapes[zone[k]].id).join()}` : ''}`);
    }
    say(other === 0, `${city}: every station on its own zone's side`, `${M.info.wrongZone} not (the map's count), ${shared} of them in a cell shared with another zone's station, ${other} other`);
  }
  const ls = landSea(city, lines, M, frame, K);
  say(ls.biggestFalseSeaKm2 <= FALSE_KM2 * ls.rhoKm ** 2 && ls.biggestFalseLandKm2 <= FALSE_KM2 * ls.rhoKm ** 2, `${city}: land drawn as land and sea as sea, away from coasts`, JSON.stringify(ls));
  // rivers: crossings drawn, lines crossed where the river does not
  if (real && M.riverFits) {
    const hopPoly = new Map();
    for (const pc of lay.pieces) for (let q = 0; q + 1 < pc.nodes.length; q++) { const a = pc.nodes[q], b = pc.nodes[q + 1], pts = []; for (let v = pc.at[q]; v <= pc.at[q + 1]; v++) pts.push(pc.verts[v * 2] / K, pc.verts[v * 2 + 1] / K); hopPoly.set(Math.min(a, b) * n + Math.max(a, b), pts); }
    const hk = (e) => Math.min(g.ea[e], g.eb[e]) * n + Math.max(g.ea[e], g.eb[e]);
    const G = M.grid, wetHop = (pts) => { for (let i = 0; i + 3 < pts.length; i += 2) { const m = Math.ceil(Math.hypot(pts[i + 2] - pts[i], pts[i + 3] - pts[i + 1]) / 0.5); for (let t = 0; t <= m; t++) { const gi = Math.floor((pts[i] + ((pts[i + 2] - pts[i]) * t) / m - G.x0) / G.cell), gj = Math.floor((pts[i + 1] + ((pts[i + 3] - pts[i + 1]) * t) / m - G.y0) / G.cell); if (gi >= 0 && gj >= 0 && gi < G.w && gj < G.h && G.d[gj * G.w + gi] === 0) return true; } } return false; };
    const own = new Map();
    for (const A of real.arcs) for (const c of A.cross) (own.get(A.river) ?? own.set(A.river, new Set()).get(A.river)).add(hk(c.e));
    let expected = 0, drawn = 0, wrong = 0;
    for (const ai of new Set(M.riverFits.map((f) => f.arc))) {
      const fs = M.riverFits.filter((f) => f.arc === ai);
      for (const c of real.arcs[ai].cross) { const pts = hopPoly.get(hk(c.e)); expected++; if (pts && (fs.some((f) => crosses(f.fit, pts)) || wetHop(pts))) drawn++; }
    }
    for (const { arc, fit } of M.riverFits) for (const [k, pts] of hopPoly) if (!own.get(real.arcs[arc].river)?.has(k) && crosses(fit, pts)) wrong++;
    console.log(`     rivers: ${M.riverFits.length} bands; of the drawn arcs' crossings ${drawn} of ${expected} drawn; ${wrong} lines a band crosses where its river does not`);
    if (city === 'nyc') {
      // the Harlem River: the bands of the arcs within 300 m of its middle at Macombs Dam, at least 30 pitches in all
      const [mx, my] = L.toMerc01(-73.9303, 40.8285), hx = (mx - frame[0]) * frame[2], hy = -(my - frame[1]) * frame[2];
      const near = new Set(real.arcs.map((A, ai) => [A, ai]).filter(([A]) => { for (let q = 0; q + 3 < A.pts.length; q += 2) if (segDist(hx, hy, A.pts[q], A.pts[q + 1], A.pts[q + 2], A.pts[q + 3]) < 300) return true; return false; }).map(([, ai]) => ai));
      let len = 0;
      for (const f of M.riverFits) if (near.has(f.arc)) for (let q = 0; q + 3 < f.fit.length; q += 2) len += Math.hypot(f.fit[q + 2] - f.fit[q], f.fit[q + 3] - f.fit[q + 1]);
      say(len >= 30, 'nyc: the Harlem River drawn', `${Math.round(len)} pitches of band, at least ${Math.min(...(M.riverFits.filter((f) => near.has(f.arc)).map((f) => f.w)), Infinity)} pitches wide`);
    }
  }
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exitCode = failed ? 1 : 0;
