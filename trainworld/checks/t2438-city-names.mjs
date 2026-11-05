// T-2438: the network diagram's city names far out (diagram/cityNames.ts) on all Japan, the shared Northeast save and
// the real New York save. Each save is laid out as the page's worker does, its zones found from their coarse outlines
// the way t2168-borders.mjs does, and its city names placed. Passes when the real New York save (one block) is named
// "New York" alone at every city level, the Northeast names New York and Philadelphia, no two names meet at any
// level, every name lies beside its block's box (within a gap of 4 px plus its own height), and below the switch no
// station names are drawn while above it no city names are. Also how much line the names cover.
// T-2447: every block has a name and none repeats; where a bigger block of the zone has the zone's name, the
// block takes the nearest city or town of the zone's place list (as the page loads it); New York, the Northeast
// and Toronto are named as before, one name per block, none given.
//   node trainworld/checks/t2438-city-names.mjs [jpn,ne,nyc,tor]      (about two minutes with Japan; figures in
//   artifacts/trainworld/t2438/figures.json)
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { openSave, SAVES } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing } from '@app/src/diagram/draw.ts';
export { zoneBorders } from '@app/src/diagram/borders.ts'; export { toMerc01, EARTH_C } from '@app/src/map/geo.ts';
export { makeFrame } from '@app/src/game/frame.ts'; export * from '@app/src/diagram/cityNames.ts';
export { placeNames, levelAt, guessWidth, NAME_LINE } from '@app/src/diagram/names.ts'; export { zoneName } from '@app/src/game/zoneNames.ts';`);
const which = (process.argv[2] ?? 'jpn,ne,nyc,tor').split(',');
const TW = new URL('../', import.meta.url);
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
const ORIGIN = { ne: [-73.985, 40.758], nyc: [-73.985, 40.758], jpn: [139.7, 35.7], tor: [-79.3832, 43.6534] };
SAVES.tor = { file: new URL('data/packs/real/tor.save', TW), lon: -79.3832, lat: 43.6534 };
const FONT = 12;
let failed = 0;
const out = {};
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const city of which) {
  const s = openSave(city);
  const { g } = s.graph();
  const text = [...g.key].map((k) => s.name(k) ?? '');
  const ni = s.api.nodes_info(), xy = new Map();
  for (let i = 0; i < ni.length; i += 10) xy.set(ni[i], [ni[i + 1], ni[i + 2]]);
  s.api.free();
  // the zones' outlines in the graph's frame, as view.ts sends them
  const fr = L.makeFrame({ kind: 'equirect', lon: ORIGIN[city][0], lat: ORIGIN[city][1] }), k = g.k / L.EARTH_C;
  let fx = 0, fy = 0, w = Infinity, so = Infinity, e = -Infinity, n = -Infinity;
  g.key.forEach((key, i) => {
    const [lng, lat] = fr.toLonLat(...xy.get(key)), [mx, my] = L.toMerc01(lng, lat);
    (fx += mx - g.x[i] * k), (fy += my + g.y[i] * k);
    (w = Math.min(w, lng)), (e = Math.max(e, lng)), (so = Math.min(so, lat)), (n = Math.max(n, lat));
  });
  (fx /= g.key.length), (fy /= g.key.length);
  const shapes = [], byId = new Map();
  for (const c of REG.cities) {
    byId.set(c.id, c);
    const b = c.box, flat = OUTL.zones[c.id];
    if (!b || !flat || b[0] > e || b[2] < w || b[1] > n || b[3] < so) continue;
    shapes.push({ id: c.id, rings: flat.map((f) => {
      const r = new Float64Array(f.length);
      let x = 0, y = 0;
      for (let i = 0; i < f.length; i += 2) {
        const [mx, my] = L.toMerc01((x += f[i]) / OUTL.q, (y += f[i + 1]) / OUTL.q);
        (r[i] = (mx - fx) / k), (r[i + 1] = -(my - fy) / k);
      }
      return r;
    }) });
  }
  const lay = L.layoutDiagram(g);
  const d = L.drawing(g, lay);
  const widths = text.map((t) => L.guessWidth(t, FONT));
  const plan = L.placeNames(g, lay, d, text, widths, widths.map(() => 0), FONT);
  const b = L.zoneBorders(g, lay.x, lay.y, d, shapes);
  const zones = b.zones.map((id) => { const c = byId.get(id), name = L.zoneName(id, c.name); return { name, width: L.guessWidth(name, FONT), trips: c.localTrips ?? 0, city: c.kind !== 'zone' }; });
  // T-2447: the first placing is as the page does it before any place list has arrived; the zones that asked
  // (cn0.want) then get theirs, from data/packs/<id>.places.json as the page loads them, and it is placed again
  const cn0 = L.placeCityNames(d, g, lay.blocks, b.zoneOf, zones, FONT, b.zones);
  const before = cn0.text.filter(Boolean).length;
  for (const id of cn0.want) {
    const zi = b.zones.indexOf(id), f = new URL(`data/packs/${id}.places.json`, TW);
    if (!existsSync(f)) { zones[zi].towns = []; continue; }
    const pf = JSON.parse(readFileSync(f));
    const ps = pf.places.map(([name, kind, lng, lat, pop]) => ({ name, kind: pf.kinds[kind], lng, lat, pop }));
    const vil = ps.filter((p) => p.kind === 'village').sort((p, q) => q.pop - p.pop).slice(0, 30);
    zones[zi].towns = [...ps.filter((p) => p.kind === 'city' || p.kind === 'town'), ...vil].map((p) => {
      const [mx, my] = L.toMerc01(p.lng, p.lat);
      return { name: p.name, width: L.guessWidth(p.name, FONT), x: (mx - fx) / k, y: -(my - fy) / k, big: p.kind !== 'village' };
    });
  }
  const cn = cn0.want.length ? L.placeCityNames(d, g, lay.blocks, b.zoneOf, zones, FONT, b.zones) : cn0;
  const r = figures(d, lay.blocks, cn);
  r.ms = Math.round(cn.ms);
  r.blocks = lay.blocks.list.length;
  r.named = new Set(cn.text.filter(Boolean)).size;
  r.namedBefore = before;
  r.unnamed = cn.text.filter((t) => !t).length;
  r.repeats = cn.text.filter((t, i) => t && cn.text.indexOf(t) !== i);
  r.given = cn.text.map((t, i) => [t, cn0.text[i]]).filter(([t, o]) => t && !o).map(([t]) => t);
  out[city] = r;
  console.log(`     ${city}: ${r.blocks} blocks, ${before} named before, ${r.named} names after; placed per level ${r.perLevel.join(' ')}; covering ${r.coverMedian} px of line (median), ${r.coverMax} at most; ${r.ms} ms`);
  console.log(`     at ${cn.levels[r.wholeLevel].toFixed(2)} px: ${r.whole.join(', ')}`);
  console.log(`     given to blocks that had none: ${r.given.join(', ') || '-'}`);
  say(r.unnamed === 0, `${city}: every block has a name`, `${r.unnamed} without`);
  say(!r.repeats.length, `${city}: no name twice`, r.repeats.join(', '));
  if (city !== 'jpn') say(r.given.length === 0 && before === r.blocks, `${city}: names as before (one per block, none given)`);
  if (city === 'nyc') say(cn.at.every((l) => l.block.length === 1 && cn.text[l.block[0]] === 'New York'), 'nyc: "New York" alone at every city level', r.whole.join(', '));
  if (city === 'ne') say(cn.at.every((l) => [...l.block].map((x) => cn.text[x]).sort().join() === 'New York,Philadelphia'), 'ne: New York and Philadelphia at every city level', r.whole.join(', '));
  say(!r.meet, `${city}: no two city names meet at any level`, r.meet || '');
  say(!r.far, `${city}: every city name beside its block's box`, r.far || '');
  const below = L.CITY_SWITCH * 0.99, above = L.CITY_SWITCH * 1.01;
  say(L.levelAt(plan, below) < 0 && L.cityLevelAt(cn, above) < 0 && L.cityLevelAt(cn, below) >= 0 && L.levelAt(plan, above) >= 0,
    `${city}: city names below ${L.CITY_SWITCH.toFixed(2)} px a pitch, station names from it up`);
}
mkdirSync(`${ARTIFACTS}t2438`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2438/figures.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);

function figures(d, blocks, cn) {
  const seg = [];
  for (const rb of d.ribbons) for (let j = 0; j + 3 < rb.length; j += 2) seg.push([rb[j], rb[j + 1], rb[j + 2], rb[j + 3]]);
  const covers = [];
  let meet = '', far = '', wholeLevel = 0;
  cn.levels.forEach((P, li) => {
    const s = P / d.pitch, at = cn.at[li], boxes = [];
    if (P <= 0.5) wholeLevel = li;
    for (let i = 0; i < at.block.length; i++) {
      const nb = L.cityBox(cn, li, i, s), bb = blocks.list[at.block[i]].box, H = (FONT * L.NAME_LINE) / s, gap = 4 / s + H;
      for (const o of boxes) if (nb[0] < o[2] && nb[2] > o[0] && nb[1] < o[3] && nb[3] > o[1]) meet ||= `${cn.text[at.block[i]]} at ${P.toFixed(2)} px`;
      boxes.push(nb);
      const dx = Math.max(bb[0] - nb[2], 0, nb[0] - bb[2]), dy = Math.max(bb[1] - nb[3], 0, nb[1] - bb[3]);
      if (dx > gap + 1e-6 || dy > gap + 1e-6) far ||= `${cn.text[at.block[i]]} at ${P.toFixed(2)} px, ${(Math.max(dx, dy) * s).toFixed(0)} px off`;
      let c = 0;
      for (const q of seg) c += clip(q, nb);
      covers.push(c * s);
    }
  });
  covers.sort((a, b) => a - b);
  return { perLevel: cn.at.map((l) => l.block.length), coverMedian: +(covers[covers.length >> 1] ?? 0).toFixed(1), coverMax: +(covers[covers.length - 1] ?? 0).toFixed(1), meet, far, wholeLevel, whole: [...cn.at[wholeLevel].block].map((x) => cn.text[x]) };
}

function clip([x0, y0, x1, y1], b) {
  const dx = x1 - x0, dy = y1 - y0, p = [-dx, dx, -dy, dy], q = [x0 - b[0], b[2] - x0, y0 - b[1], b[3] - y0];
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i] < 0) return 0; continue; }
    const r = q[i] / p[i];
    if (p[i] < 0) t0 = Math.max(t0, r); else t1 = Math.min(t1, r);
    if (t0 > t1) return 0;
  }
  return (t1 - t0) * Math.hypot(dx, dy);
}
