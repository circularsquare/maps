// T-3358: the schematic's rivers keep stations on their own banks. Each river some line crosses is routed after the
// layout as one band (diagram/riverRoute.ts), and the lines are laid out again round stations it finds on the wrong
// bank (sheetLines.ts `sheetLayout`), run here as the page's worker runs them with the water on. Checked, with the
// pairs of stations recounted here: of each pair of stations near a river (within 1.5 km of its course, 3 km of each
// other), the river parts them in the drawing exactly when it does on the ground, at London's centre (the Thames from
// Battersea Park to Greenwich), New York's Harlem River and Newtown Creek (two stations near it, so no pairs) and Hong
// Kong's Tuen Mun River (the routed river once put North Greenwich and the Isle of Dogs across the Thames); every routed
// river is one band with two round ends; the Tuen Mun River drawn with at most 10 bends (22 before the light
// straightening); at most 5 London stations partly under the Thames's band. Reported: wrong pairs on every other routed
// river, and the times.
//   node trainworld/checks/t3358-river-banks.mjs [lon,nyc,hon]
import { readFileSync } from 'node:fs';
import { openSave, SAVES, masksOf, app as A } from './diagram-saves.mjs';
import { loadApp } from './lib.mjs';

const L = await loadApp(`export { rubberSheet, sheetZoneBox } from '@app/src/diagram/sheet.ts'; export { sheetLayout, riverCellM, SHEET_LINES } from '@app/src/diagram/sheetLines.ts';
export { riverCourses, nearestOn, roundPath, crossCount, bendsOf } from '@app/src/diagram/riverRoute.ts';
export { realWater } from '@app/src/diagram/waterReal.ts'; export { toMerc01 } from '@app/src/map/geo.ts';`);
const TW = new URL('../', import.meta.url), REAL = new URL('data/packs/real/', TW);
SAVES.lon ??= { file: new URL('lon.save', REAL), lon: -0.12, lat: 51.5 };
SAVES.hon ??= { file: new URL('hon.save', REAL), lon: 114.17, lat: 22.3 };
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
/** each named river by a station beside it; London's centre between two stations along it */
const PLACES = { lon: [{ river: 'the Thames, Battersea Park to Greenwich', by: 'Westminster', from: 'Battersea Park', to: 'Greenwich', min: 850 }], nyc: [{ river: 'the Harlem River', by: 'Marble Hill-225 St', min: 650 }, { river: 'Newtown Creek', by: 'Greenpoint Av', few: true }], hon: [{ river: 'the Tuen Mun River', by: 'Town Centre', bends: 10, min: 65 }] };
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

function shapesNear(box, [fx, fy, fk]) {
  const out = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > box[2] + 0.5 || b[2] < box[0] - 0.5 || b[1] > box[3] + 0.5 || b[3] < box[1] - 0.5) continue;
    out.push({ id: c.id, rings: flat.map((f) => { const r = new Float64Array(f.length); let x = 0, y = 0; for (let i = 0; i < f.length; i += 2) { const [mx, my] = L.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q); (r[i] = (mx - fx) * fk), (r[i + 1] = -(my - fy) * fk); } return r; }) });
  }
  return out;
}

for (const city of (process.argv[2] ?? 'lon,nyc,hon').split(',')) {
  const s = openSave(city), { g } = s.graph(), { frame, box } = s.ground(g), n = g.key.length;
  const names = Array.from({ length: n }, (_, i) => String(s.name(g.key[i])));
  s.api.free();
  const zb = L.sheetZoneBox(box), masks = masksOf(REG.cities.filter(({ box: b }) => b && zb[0] <= b[2] && zb[2] >= b[0] && zb[1] <= b[3] && zb[3] >= b[1]).map((c) => c.id));
  const sea = A.groundSea(g, masks, frame), sheet = L.rubberSheet(g, { masks, frame, sea }), real = L.realWater(g, masks, frame, sea, L.riverCellM(g));
  const t0 = performance.now();
  const { lines, map } = L.sheetLayout(g, sheet, shapesNear(box, frame), true, real, L.SHEET_LINES, undefined, true);
  const ms = performance.now() - t0, K = lines.k, SX = Array.from(lines.layout.x, (v) => v / K), SY = Array.from(lines.layout.y, (v) => v / K);
  const routes = map.riverRoutes, courses = L.riverCourses(real);
  console.log(`     ${city}: ${n} stations, lines, rivers and map ${Math.round(ms)} ms (${lines.info.riverKeyed ?? 0} stations laid out again); ${routes.length} rivers routed`);
  /** a routed river's pairs recounted: near stations (1.5 km of its course, inside the routed stretch) within 3 km of
   * each other, parted on the ground by the course (its stretch's ends a station's own place along it) or not */
  const wrongOf = (R, keep = () => true) => {
    const C = courses[R.course], P = L.roundPath(R.path, 2.5);
    const near = [];
    for (let i = 0; i < n; i++) { const r = L.nearestOn(C.pts, g.x[i], g.y[i]); if (r.d <= 1500 && r.d >= 30 && r.inside && keep(i, r.s)) near.push([i, r.s]); }
    // the routed stretch along the course: from the drawn path's ends taken back to the nearest station along it
    let pairs = 0, wrong = 0;
    const at = [];
    for (let a = 0; a < near.length; a++) for (let b = a + 1; b < near.length; b++) {
      const [i] = near[a], [j] = near[b];
      if (Math.hypot(g.x[i] - g.x[j], g.y[i] - g.y[j]) > 3000) continue;
      // only pairs the routed river's own count holds (stations projecting inside its stretch)
      if (!R.pairsSet.has(Math.min(i, j) * n + Math.max(i, j))) continue;
      pairs++;
      const ground = L.crossCount(g.x[i], g.y[i], g.x[j], g.y[j], C.pts) % 2, drawn = L.crossCount(SX[i], SY[i], SX[j], SY[j], P) % 2;
      if (ground !== drawn) wrong++, at.push(`${names[i]} / ${names[j]}`);
    }
    return { pairs, wrong, at };
  };
  for (const R of routes) { R.pairsSet = new Set(); for (let q = 0; q < R.pairs.length; q += 3) R.pairsSet.add(Math.min(R.pairs[q], R.pairs[q + 1]) * n + Math.max(R.pairs[q], R.pairs[q + 1])); }
  // one band each, two round ends: the bands' runs join end to end, and two end stubs a river
  const ends = (map.water.riverEnds ?? []).reduce((t, c) => t + c.filter((v, q) => q % 3 === 0 && v === 0).length, 0);
  let gaps = 0;
  for (const R of routes) {
    const fits = map.riverFits.filter((f) => f.arcs[0] === R.arcs[0]);
    for (let q = 1; q < fits.length; q++) { const a = fits[q - 1].fit, b = fits[q].fit; if (Math.hypot(a[a.length - 2] - b[0], a[a.length - 1] - b[1]) > 1e-6) gaps++; }
  }
  say(gaps === 0 && ends >= 2 * routes.length, `${city}: every routed river one band with round ends`, `${routes.length} rivers, ${gaps} breaks, ${ends} round ends`);
  for (const P of PLACES[city] ?? []) {
    const st = names.indexOf(P.by);
    let R = null, bd = Infinity;
    for (const Q of routes) for (const a of Q.arcs) { const d = L.nearestOn(real.arcs[a].pts, g.x[st], g.y[st]).d; if (d < bd) (bd = d), (R = Q); }
    if (!R || bd > 1500) { say(false, `${city}: ${P.river} routed`, 'no routed river near ' + P.by); continue; }
    let keep = () => true;
    if (P.from) {
      const C = courses[R.course], fa = names.findIndex((x) => x.startsWith(P.from)), fb = names.findIndex((x) => x.startsWith(P.to));
      const s0 = L.nearestOn(C.pts, g.x[fa], g.y[fa]).s, s1 = L.nearestOn(C.pts, g.x[fb], g.y[fb]).s;
      keep = (i, s) => s >= Math.min(s0, s1) && s <= Math.max(s0, s1);
    }
    const w = wrongOf(R, keep);
    // (a floor on the pairs: the recount covers the pairs the routing lists, so a list grown short would pass quietly)
    say((P.few || w.pairs >= P.min) && w.wrong === 0, `${city}: ${P.river}, every pair of stations on its own banks`, `${w.wrong} of ${w.pairs} pairs wrong ${w.at.slice(0, 4).join('; ')}; lines crossed wrongly ${R.hopWrong}, crossings missed ${R.hopMissed}`);
    if (P.bends) say(R.bends <= P.bends, `${city}: ${P.river} drawn with at most ${P.bends} bends`, `${R.bends}`);
    if (city === 'lon') say(R.under.length <= 5, 'lon: at most 5 stations partly under the Thames', `${R.under.length}: ${R.under.map((i) => names[i]).join(', ')}`);
  }
  const rest = routes.map((R) => [R, wrongOf(R)]).filter(([, w]) => w.wrong);
  console.log(`     ${city}: wrong pairs on routed rivers anywhere: ${rest.reduce((t, [, w]) => t + w.wrong, 0)} ${rest.map(([, w]) => w.at.slice(0, 2).join('; ')).join(' | ')}`);
}
console.log(failed ? `${failed} FAILED` : 'all passed');
process.exitCode = failed ? 1 : 0;
