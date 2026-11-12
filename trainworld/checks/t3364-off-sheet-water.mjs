// T-3364, T-3362: water the diagram only guesses. With the water on (the sheet path, as the page's worker runs it):
// - an inland network (Nottingham's trams and two rail lines, made here, no save) draws no water at the drawing's edge:
//   past the carried map's edge a cell takes the nearest cell on it, and only big water runs on (it used to be all sea,
//   so the city sat on a round island);
// - New York still draws the sea at the edge (the Atlantic and the Sound run on past it);
// - T-3362: the ground's far grid does not run a strip of guessed land out to sea: on Anita's North America day 476
//   save the ground at 26.6 to 26.8 N between 79.5 and 75 W is sea (a coast cell of the Miami zone's mask ran east
//   along its row; the diagram drew it as a long strip of land south-east of Port St. Lucie).
//   node trainworld/checks/t3364-off-sheet-water.mjs
import { readFileSync } from 'node:fs';
import { openSave, waterOf, masksOf, SAVES, app as A } from './diagram-saves.mjs';
import { loadApp } from './lib.mjs';

const L = await loadApp(`export { rubberSheet, sheetZoneBox } from '@app/src/diagram/sheet.ts'; export { sheetLines, sheetMap, riverCellM } from '@app/src/diagram/sheetLines.ts';
export { realWater } from '@app/src/diagram/waterReal.ts'; export { inRings } from '@app/src/diagram/borders.ts';`);
const TW = new URL('../', import.meta.url);
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

function shapesNear(box, [fx, fy, fk]) {
  const out = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > box[2] + 0.5 || b[2] < box[0] - 0.5 || b[1] > box[3] + 0.5 || b[3] < box[1] - 0.5) continue;
    out.push({ id: c.id, rings: flat.map((f) => { const r = new Float64Array(f.length); let x = 0, y = 0; for (let i = 0; i < f.length; i += 2) { const [mx, my] = A.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q); (r[i] = (mx - fx) * fk), (r[i + 1] = -(my - fy) * fk); } return r; }) });
  }
  return out;
}
const masksFor = (box) => { const zb = L.sheetZoneBox(box); return masksOf(REG.cities.filter(({ box: b }) => b && zb[0] <= b[2] && zb[2] >= b[0] && zb[1] <= b[3] && zb[3] >= b[1]).map((c) => c.id)); };

/** a diagram graph from lines given as [lng, lat] polylines, a station every `stepKm`, joined within 200 m */
function graphOf(lines, stepKm) {
  const lat0 = lines[0][0][1], lng0 = lines[0][0][0], cos = Math.cos((lat0 * Math.PI) / 180), [mx0, my0] = A.toMerc01(lng0, lat0), k = 1 / cos, frame = [mx0, my0, A.EARTH_C / k];
  const xy = ([lng, lat]) => { const [mx, my] = A.toMerc01(lng, lat); return [(mx - mx0) * frame[2], (my0 - my) * frame[2]]; };
  const X = [], Y = [], node = (x, y) => { for (let i = 0; i < X.length; i++) if (Math.hypot(X[i] - x, Y[i] - y) < 200) return i; X.push(x), Y.push(y); return X.length - 1; };
  const gl = [], edges = new Map();
  lines.forEach((pts, li) => {
    const chain = [];
    for (let p = 0; p + 1 < pts.length; p++) {
      const [ax, ay] = xy(pts[p]), [bx, by] = xy(pts[p + 1]), m = Math.max(1, Math.round(Math.hypot(bx - ax, by - ay) / (stepKm * 1000)));
      for (let t = 0; t < m; t++) { const v = node(ax + ((bx - ax) * t) / m, ay + ((by - ay) * t) / m); if (chain.at(-1) !== v) chain.push(v); }
    }
    const [ex, ey] = xy(pts.at(-1)), v = node(ex, ey);
    if (chain.at(-1) !== v) chain.push(v);
    gl.push({ id: li + 1, colour: ['#c23a30', '#0b6db7', '#f2b705', '#3a9b4a'][li % 4], chain: Int32Array.from(chain), stop: new Uint8Array(chain.length).fill(1) });
    for (let i = 0; i + 1 < chain.length; i++) { const a = Math.min(chain[i], chain[i + 1]), b = Math.max(chain[i], chain[i + 1]), key = a * 100000 + b; (edges.get(key) ?? edges.set(key, []).get(key)).push(li); }
  });
  const ks = [...edges.keys()].sort((p, q) => p - q), ea = [], eb = [], first = [0], on = [], hop = [], hopAt = [0];
  for (const key of ks) { const a = Math.floor(key / 100000), b = key % 100000; ea.push(a), eb.push(b); for (const l of new Set(edges.get(key))) on.push(l); first.push(on.length); hop.push(X[a], Y[a], X[b], Y[b]), hopAt.push(hop.length / 2); }
  const g = { key: Int32Array.from(X, (_, i) => i + 1), x: Float64Array.from(X), y: Float64Array.from(Y), lines: gl, ea: Int32Array.from(ea), eb: Int32Array.from(eb), first: Int32Array.from(first), on: Int32Array.from(on), stroke: Int32Array.from(on), mid: [0, 0], k, hop: Float32Array.from(hop), hopAt: Int32Array.from(hopAt) };
  const all = lines.flat(), box = [Math.min(...all.map((p) => p[0])), Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[0])), Math.max(...all.map((p) => p[1]))];
  return { g, frame, box };
}

/** path commands (0 move, 1 line, 2 quadratic) as rings of points */
const flat = (cmd) => {
  const rings = [];
  let cur = null;
  for (let i = 0; i < cmd.length; ) {
    if (cmd[i] === 0) (cur = [cmd[i + 1], cmd[i + 2]]), rings.push(cur), (i += 3);
    else if (cmd[i] === 1) cur.push(cmd[i + 1], cmd[i + 2]), (i += 3);
    else cur.push(cmd[i + 3], cmd[i + 4]), (i += 5);
  }
  return rings;
};
/** the map with the water on; of points along its grid's border (a cell in), the share in the drawn sea */
function edgeWater(g, frame, box) {
  const masks = masksFor(box), sea = A.groundSea(g, masks, frame), sheet = L.rubberSheet(g, { masks, frame, sea }), lines = L.sheetLines(g, sheet);
  const cell = L.riverCellM(g), real = cell ? L.realWater(g, masks, frame, sea, cell) : null;
  const M = L.sheetMap(lines, g, shapesNear(box, frame), true, undefined, true, real), G = M.grid, K = lines.k, rings = flat(M.water.sea);
  let wet = 0, all = 0;
  const at = (i, j) => { all++; if (L.inRings(rings, (G.x0 + (i + 0.5) * G.cell) * K, (G.y0 + (j + 0.5) * G.cell) * K)) wet++; };
  const step = Math.max(1, Math.floor(Math.max(G.w, G.h) / 400));
  for (let i = 1; i < G.w - 1; i += step) at(i, 1), at(i, G.h - 2);
  for (let j = 1; j < G.h - 1; j += step) at(1, j), at(G.w - 2, j);
  return { wet, all, share: wet / all };
}

{
  // Nottingham: tram 1 Hucknall to Clifton, tram 2 Phoenix Park to Toton Lane, rail Attenborough to Newark, rail to Mansfield
  const { g, frame, box } = graphOf([
    [[-1.205, 53.038], [-1.183, 52.997], [-1.15, 52.953], [-1.152, 52.935], [-1.172, 52.903]],
    [[-1.212, 53.0], [-1.18, 52.98], [-1.15, 52.953], [-1.17, 52.94], [-1.215, 52.933], [-1.262, 52.921]],
    [[-1.31, 52.9], [-1.23, 52.928], [-1.146, 52.947], [-1.07, 52.955], [-0.98, 52.975], [-0.81, 53.08]],
    [[-1.146, 52.947], [-1.12, 52.99], [-1.15, 53.06], [-1.2, 53.14]],
  ], 1.5);
  const r = edgeWater(g, frame, box);
  say(r.wet === 0, 'nottingham: no sea at the drawing\'s edge', `${r.wet} of ${r.all} points along the edge in the sea`);
}
{
  const s = openSave('nyc'), { g } = s.graph(), { frame, box } = waterOf(s, g);
  s.api.free();
  const r = edgeWater(g, frame, box);
  say(r.share > 0.2, 'nyc: the sea runs on past the edge', `${(100 * r.share).toFixed(0)}% of points along the edge in the sea (want over 20%)`);
}
{
  SAVES.na476 = { file: new URL('../artifacts/trainworld/t3362/na476.save', TW), lon: 0, lat: 0 };
  const s0 = openSave('na476'), o = s0.api.frame_origin();
  s0.api.free();
  (SAVES.na476.lon = o[0]), (SAVES.na476.lat = o[1]);
  const s = openSave('na476'), { g } = s.graph(), { frame, box } = waterOf(s, g);
  s.api.free();
  const sea = A.groundSea(g, masksFor(box), frame), [fx, fy, fk] = frame, land = [];
  for (let lng = -79.5; lng <= -75; lng += 0.25) for (const lat of [26.6, 26.7, 26.8]) {
    const [mx, my] = A.toMerc01(lng, lat), x = (mx - fx) * fk, y = (fy - my) * fk;
    if (!(sea.isSea(x, y) || sea.massAt(x, y) < 0)) land.push(`${lng},${lat}`);
  }
  say(land.length === 0, 'north america: no guessed land strip east of Palm Beach', `${land.length} of 57 points land ${land.slice(0, 4).join('; ')}`);
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exitCode = failed ? 1 : 0;
