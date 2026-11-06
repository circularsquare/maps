// T-2580: the rubber sheet's water and zone borders (diagram/sheetLines.ts `sheetMap`, diagram/sheetThin.ts) on the
// real New York save and all Japan, run as the page's worker does with the water on: the sheet, the lines on it, the
// rivers on the ground (a city map's narrow rivers), the map. Checked: every edge of the drawn water (the sea's coast,
// the rivers' bands) runs at 0, 45 or 90 degrees, as t2169-water.mjs asks of the flag-off water; no stair steps (a run
// under 4 pitches between turns of opposite sign, next to another such: under STAIRS per 1000 pitches of coast and of
// border); every station on land, its own zone's side and under no band; in New York, the Harlem River drawn. Reported:
// jogs, rivers' crossings drawn and lines a band crosses where its river does not, and the times.
//   node trainworld/checks/t2580-sheet-water.mjs [nyc,jpn]
import { readFileSync } from 'node:fs';
import { openSave, waterOf } from './diagram-saves.mjs';
import { loadApp } from './lib.mjs';

const L = await loadApp(`export { rubberSheet } from '@app/src/diagram/sheet.ts'; export { sheetLines, sheetMap, riverCellM, SHEET_LINES } from '@app/src/diagram/sheetLines.ts';
export { realWater } from '@app/src/diagram/waterReal.ts'; export { inRings } from '@app/src/diagram/borders.ts'; export { toMerc01 } from '@app/src/map/geo.ts';`);
const TW = new URL('../', import.meta.url);
const which = (process.argv[2] ?? 'nyc,jpn').split(',');
/** stair runs allowed per 1000 pitches of coast or border (before T-2580 New York had 82) */
const STAIRS = 10;
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
const segDist = (px, py, ax, ay, bx, by) => { const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0; return Math.hypot(px - ax - t * dx, py - ay - t * dy); };
const crosses = (a, b) => {
  for (let i = 0; i + 3 < a.length; i += 2) for (let j = 0; j + 3 < b.length; j += 2) {
    const d1 = (a[i + 2] - a[i]) * (b[j + 1] - a[i + 1]) - (a[i + 3] - a[i + 1]) * (b[j] - a[i]), d2 = (a[i + 2] - a[i]) * (b[j + 3] - a[i + 1]) - (a[i + 3] - a[i + 1]) * (b[j + 2] - a[i]);
    const d3 = (b[j + 2] - b[j]) * (a[i + 1] - b[j + 1]) - (b[j + 3] - b[j + 1]) * (a[i] - b[j]), d4 = (b[j + 2] - b[j]) * (a[i + 3] - b[j + 1]) - (b[j + 3] - b[j + 1]) * (a[i + 2] - b[j]);
    if (d1 * d2 < 0 && d3 * d4 < 0) return true;
  }
  return false;
};

for (const city of which) {
  const s = openSave(city), { g } = s.graph(), { frame, box, masks, sea } = waterOf(s, g);
  s.api.free();
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
  say(M.info.wrongZone === 0, `${city}: every station on its own zone's side`, `${M.info.wrongZone} not`);
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
