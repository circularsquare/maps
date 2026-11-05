// T-2464: the network diagram's zone borders (diagram/borders.ts) on all Japan, the shared Northeast save, the real
// New York save and Toronto's real network. Each save is laid out and drawn as the page's worker does, then its zones'
// coarse outlines (`packs/outlines.json`) are put in the graph's frame the way view.ts does (by the stations' own
// mercator places), which decide each station's zone, and the borders are drawn. Checked, as drawn (corners rounded):
// every zone with stations is closed (each end of a border run is met an even number of times among its zone's runs);
// every station drawn (its pill's middle) is inside its own zone's borders (even-odd over them) and outside every
// other's, between blocks too;
// every border is made of runs at 0, 45 and 90 degrees. Reported: pills a border touches, and the times.
//   node trainworld/checks/t2464-borders.mjs [jpn,ne,nyc,tor]      (about a minute and a half with Japan; figures in
//   artifacts/trainworld/t2464/figures.json; T2464_CACHE=<folder> reads and writes laid out saves there)
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { serialize, deserialize } from 'node:v8';
import { openSave, SAVES, waterOf } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, pillSize } from '@app/src/diagram/draw.ts';
export { zoneBorders } from '@app/src/diagram/borders.ts'; export { toMerc01 } from '@app/src/map/geo.ts';`);
const TW = new URL('../', import.meta.url);
SAVES.tor ??= { file: new URL('data/packs/real/tor.save', TW), lon: -79.38, lat: 43.65 };
const which = (process.argv[2] ?? 'jpn,ne,nyc,tor').split(',');
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
const CACHE = process.env.T2464_CACHE;
let failed = 0;
const out = {};
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

/** the save laid out and drawn, with its zones' shapes in the graph's frame */
function laidOut(city) {
  const file = CACHE && `${CACHE}/${city}.bin`;
  if (file && existsSync(file)) return deserialize(readFileSync(file));
  const s = openSave(city);
  const { g } = s.graph();
  const names = [...g.key].map((k) => s.name(k) ?? '');
  // the frame as view.ts works it out, from each node's key station, in the save's own frame (T-2542); and the sea on
  // the ground the page's layout reads (T-2541)
  const { frame: [fx, fy, K], box: [w, so, e, n], sea } = waterOf(s, g), k = 1 / K;
  s.api.free();
  const shapes = [];
  for (const c of REG.cities) {
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > e || b[2] < w || b[1] > n || b[3] < so) continue;
    shapes.push({ id: c.id, name: c.name, rings: flat.map((f) => {
      const r = new Float64Array(f.length);
      let x = 0, y = 0;
      for (let i = 0; i < f.length; i += 2) {
        const [mx, my] = L.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q);
        (r[i] = (mx - fx) / k), (r[i + 1] = -(my - fy) / k);
      }
      return r;
    }) });
  }
  const t0 = performance.now();
  const lay = L.layoutDiagram(g, L.LAYOUT, null, undefined, sea);
  const layoutMs = performance.now() - t0;
  const d = L.drawing(g, lay);
  const r = { g, lay, d, shapes, names, layoutMs };
  if (file) writeFileSync(file, serialize(r));
  return r;
}

for (const city of which) {
  const { g, lay, d, shapes, names, layoutMs } = laidOut(city);
  const t0 = performance.now();
  const b = L.zoneBorders(g, lay.x, lay.y, d, shapes, true);
  const ms = performance.now() - t0;
  const r = figures(g, lay, d, b, names);
  r.ms = { layout: Math.round(layoutMs ?? 0), borders: Math.round(ms) };
  r.zonesDrawn = b.zones.length;
  r.blocks = lay.blocks.list.length;
  out[city] = r;
  console.log(`     ${city}: ${r.stations} stations in ${r.zonesDrawn} zones, ${r.blocks} blocks, ${r.between} between blocks; ${r.runs} border runs, ${r.legs} legs; ${r.touched} pills touched by a border; layout ${r.ms.layout} ms, borders ${r.ms.borders} ms`);
  if (r.wrongs.length) console.log(`     wrong: ${r.wrongs.join(', ')}`);
  say(r.open.length === 0, `${city}: every zone with stations is closed`, r.open.join(', '));
  say(r.wrong === 0, `${city}: every station inside its own zone's borders and no other's`, `${r.wrong} of ${r.stations} wrong (${r.wrongBetween} between blocks)`);
  say(r.offAngle === 0, `${city}: every border run at 0, 45 or 90 degrees`, r.offAngle ? `${r.offAngle} legs off` : '');
}
mkdirSync(`${ARTIFACTS}t2464`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2464/figures.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);

/** a run's commands (toPath's) as a polyline, curves sampled */
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

function figures(g, lay, d, b, names) {
  const n = g.key.length, nz = b.zones.length, runs = b.runs;
  const polys = runs.map((r) => flat(b.cmds.subarray(r.c[0], r.c[1])));
  // closed: per zone, each open run's ends met an even number of times
  const open = [];
  for (let z = 0; z < nz; z++) {
    const ends = new Map();
    runs.forEach((r) => {
      if (r.closed || (r.left !== z && r.right !== z)) return;
      for (const k of [0, r.p.length - 2]) {
        const key = `${r.p[k].toFixed(6)},${r.p[k + 1].toFixed(6)}`;
        ends.set(key, (ends.get(key) ?? 0) + 1);
      }
    });
    if ([...ends.values()].some((c) => c % 2)) open.push(b.zones[z]);
  }
  // the drawn borders' segments, by rows (for the even-odd rays) and cells (for the pills touched)
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
  /** per zone, whether (px, py) is inside its borders (even-odd, a ray to the east) */
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
  let stations = 0, wrong = 0, wrongBetween = 0, between = 0, touched = 0;
  const wrongs = [];
  const segD = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy, t = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2)) : 0;
    return Math.hypot(px - ax - dx * t, py - ay - dy * t);
  };
  // per station drawn, its (first) pill's middle and half width: the station as drawn (its axis may pass beside its
  // place in the layout)
  const pillR = new Float64Array(n), px = new Float64Array(n), py = new Float64Array(n);
  d.pills.forEach((p, q) => {
    const k = d.pillOf[q], [r, o] = L.pillSize(d, k);
    if (!pillR[k]) (px[k] = (p[0] + p[2]) / 2), (py[k] = (p[1] + p[3]) / 2);
    pillR[k] = Math.max(pillR[k], r + o);
  });
  for (let i = 0; i < n; i++) {
    const z = b.zoneOf[i];
    // a station drawn: with a pill (one passed and never stopped at has none)
    if (z < 0 || !pillR[i]) continue;
    stations++;
    const btw = lay.blocks.list.length > 1 && lay.blocks.of[i] < 0;
    if (btw) between++;
    const c = insideOf(px[i], py[i]);
    let bad = !c[z];
    for (let o = 0; o < nz && !bad; o++) if (o !== z && c[o]) bad = true;
    if (bad) {
      wrong++;
      if (btw) wrongBetween++;
      if (wrongs.length < 10) wrongs.push(`${names[i]} (${b.zones[z]})`);
    }
    {
      let near = Infinity;
      const ci = Math.floor(px[i] / RH), cj = Math.floor(py[i] / RH);
      for (let gi = ci - 1; gi <= ci + 1; gi++) for (let gj = cj - 1; gj <= cj + 1; gj++) for (const s of cells.get(`${gi},${gj}`) ?? []) {
        const [x0, y0, x1, y1] = segs[s];
        near = Math.min(near, segD(px[i], py[i], x0, y0, x1, y1));
      }
      if (near < pillR[i]) touched++;
    }
  }
  // octilinear: the fitted runs' legs
  let legs = 0, offAngle = 0;
  for (const r of runs) for (let k = 0; k + 3 < r.p.length; k += 2) {
    const dx = Math.abs(r.p[k + 2] - r.p[k]), dy = Math.abs(r.p[k + 3] - r.p[k + 1]), l = Math.hypot(dx, dy);
    legs++;
    if (Math.min(dx, dy) > 1e-6 * l && Math.abs(dx - dy) > 1e-6 * l) offAngle++;
  }
  return { stations, between, wrong, wrongBetween, wrongs, open, touched, runs: runs.length, legs, offAngle };
}
