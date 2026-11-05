// T-2161: the network diagram's pills (app/src/diagram/draw.ts) on the real New York save and Tokyo's real
// network: no pill wider than two pitches across (Times Sq, 14 St and 59 St named), no two stations' pills
// overlapping (each at its drawn size, a local stop's smaller, T-2272), no hook at the stations that had one (T-2244, T-2255), and how many stations need more than one
// pill. `nyc-game` and `tyo-game` are the game's own graphs (diagram-game.mjs; run it first).
//   node trainworld/checks/t2161-diagram-pills.mjs [nyc,tyo,nyc-game,tyo-game]      (about a minute; figures in
//   artifacts/trainworld/t2161/pills.json)
import { mkdirSync, writeFileSync } from 'node:fs';
import { openSave, waterOf, gameSea } from './diagram-saves.mjs';
import { gameGraph, splitCity, bitsHash, pillsFlat } from './diagram-game.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const A = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, pillSize, PILL_R, PILL_OUTLINE, STROKE_SHARE } from '@app/src/diagram/draw.ts';`);
const which = (process.argv[2] ?? 'nyc,tyo,nyc-game,tyo-game').split(',');
// the save's node names, and the game's complex names (the most lines stopping, diagram/names.ts' pickName)
const NAMED = { nyc: ['Times Sq-42 St', '14 St-Union Sq', '59 St-Columbus Circle', '34 St-Penn Station', 'New York Penn'], tyo: ['新宿', '代々木', '渋谷', '池袋'] };
/** stations that had hooks (T-2206, T-2244, T-2255) */
const HOOKLESS = { nyc: ['Times Sq-42 St', '34 St-Penn Station', 'New York Penn', 'Atlantic Av-Barclays Ctr', 'Grand Central-42 St', 'Grand Central'], tyo: ['新宿', '西日暮里'] };
let failed = 0;
const out = {};
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

const segDist = (p, q) => {
  // two segments [x0, y0, x1, y1]: 0 when they cross, else the least endpoint-to-segment distance
  const cross = (ax, ay, bx, by, cx, cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d1 = cross(p[0], p[1], p[2], p[3], q[0], q[1]), d2 = cross(p[0], p[1], p[2], p[3], q[2], q[3]);
  const d3 = cross(q[0], q[1], q[2], q[3], p[0], p[1]), d4 = cross(q[0], q[1], q[2], q[3], p[2], p[3]);
  if (((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0)) && d1 && d2 && d3 && d4) return 0;
  const pt = (px, py, s) => {
    const dx = s[2] - s[0], dy = s[3] - s[1], l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((px - s[0]) * dx + (py - s[1]) * dy) / l2)) : 0;
    return Math.hypot(s[0] + dx * t - px, s[1] + dy * t - py);
  };
  return Math.min(pt(p[0], p[1], q), pt(p[2], p[3], q), pt(q[0], q[1], p), pt(q[2], q[3], p));
};

for (const name of which) {
  const [city, game] = splitCity(name);
  let g, names, cap = null, sea = null;
  if (game) {
    cap = gameGraph(city);
    if (!cap) { console.log(`SKIP ${name}: no capture; run diagram-game.mjs first`); continue; }
    ({ g } = cap);
    names = cap.text;
    sea = gameSea(cap);
  } else {
    const s = openSave(city);
    ({ g } = s.graph());
    names = [...g.key].map((k) => s.name(k));
    ({ sea } = waterOf(s, g));
    s.api.free();
  }
  // with the sea on the ground, as the page's worker lays it out (T-2541)
  const lay = A.layoutDiagram(g, A.LAYOUT, null, undefined, sea);
  const d = A.drawing(g, lay);
  if (cap?.fresh) {
    const same = bitsHash(pillsFlat(d.pills)) === cap.page.pills;
    line(same, `${name}: Node draws the game's graph exactly as the page's worker did`, same ? '' : 'pills differ: some engine-dependent maths in the layout or drawing (diagram/fmath.ts)');
  } else if (cap) console.log(`INFO ${name}: the capture (${cap.captured}) is older than the layout code; the page is not compared; run diagram-game.mjs`);
  // a pill's half width with its outline (a local stop's is smaller, T-2272)
  const rOf = (i) => A.pillSize(d, d.pillOf[i]).reduce((a, b) => a + b);
  // across: a pill's full thickness, outline included, in pitches
  const across = d.pills.map((p, i) => (2 * rOf(i) + 2 * p[4]) / d.pitch);
  const named = {};
  for (const nm of NAMED[city]) {
    const ks = new Set(names.flatMap((x, i) => (x && x.split('/').some((part) => part === nm) ? [i] : [])));
    const mine = d.pills.map((p, i) => [p, i]).filter(([, i]) => ks.has(d.pillOf[i]));
    if (!ks.size) continue;
    named[nm] = { pills: mine.length, widest: Math.max(0, ...mine.map(([, i]) => across[i])), longest: Math.max(0, ...mine.map(([p, i]) => (Math.hypot(p[2] - p[0], p[3] - p[1]) + 2 * rOf(i)) / d.pitch)) };
  }
  // pills of two stations overlapping
  let overlaps = 0;
  const cell = 6 * d.pitch, grid = new Map();
  d.pills.forEach((p, i) => {
    const k = `${Math.floor((p[0] + p[2]) / 2 / cell)},${Math.floor((p[1] + p[3]) / 2 / cell)}`;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  });
  const reach = Math.max(...d.pills.map((p) => Math.hypot(p[2] - p[0], p[3] - p[1]))) / cell + 1;
  const pairs = [];
  d.pills.forEach((p, i) => {
    const cx = Math.floor((p[0] + p[2]) / 2 / cell), cy = Math.floor((p[1] + p[3]) / 2 / cell), R = Math.ceil(reach);
    for (let gx = cx - R; gx <= cx + R; gx++) for (let gy = cy - R; gy <= cy + R; gy++) for (const j of grid.get(`${gx},${gy}`) ?? []) {
      if (j <= i || d.pillOf[j] === d.pillOf[i]) continue;
      const q = d.pills[j];
      if (segDist(p, q) < rOf(i) + rOf(j) + p[4] + q[4]) { overlaps++; if (pairs.length < 8) pairs.push(`${names[d.pillOf[i]]} | ${names[d.pillOf[j]]}`); }
    }
  });
  const perStation = new Map();
  for (const k of d.pillOf) perStation.set(k, (perStation.get(k) ?? 0) + 1);
  const several = [...perStation.values()].filter((v) => v > 1).length;
  const widest = Math.max(...across);
  const hookNames = d.hooks.map((k) => names[k]);
  const hooked = HOOKLESS[city].filter((nm) => hookNames.some((x) => x && x.split('/').includes(nm)));
  out[name] = { pills: d.pills.length, stations: perStation.size, several, passages: d.passages.length, widest, overlaps, overlapExamples: pairs, named, hooks: hookNames };
  line(widest <= 2.001, `${name}: no pill wider than two pitches across`, `widest ${widest.toFixed(2)}; ${Object.entries(named).map(([k, v]) => `${k} ${v.pills} pill${v.pills === 1 ? '' : 's'}, ${v.longest.toFixed(1)} pitches long`).join('; ')}`);
  line(overlaps === 0, `${name}: no two stations' pills overlap`, `${overlaps} pairs${pairs.length ? ', e.g. ' + pairs.slice(0, 3).join(', ') : ''}`);
  line(hooked.length === 0, `${name}: no hook (a round stop off the pill's axis) at ${HOOKLESS[city].join(', ')}`, `${hooked.length ? 'hooks at ' + hooked.join(', ') + '; ' : ''}${hookNames.length} stations with a hook in all${hookNames.length ? ': ' + hookNames.slice(0, 8).join(', ') : ''}`);
  console.log(`     ${name}: ${perStation.size} stations, ${several} with more than one pill, ${d.passages.length} passages`);
}
mkdirSync(`${ARTIFACTS}t2161`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2161/pills.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
