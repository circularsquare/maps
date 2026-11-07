// T-2438, T-2521, T-2581: the network diagram's city names far out (diagram/cityNames.ts) on all Japan, the shared
// Northeast save, Toronto, the real New York save and Anita's North America save (na, where it is on disk). Each save is
// laid out as the page's worker does, its zones found from their coarse outlines the way t2168-borders.mjs does, and
// its city names made. Passes when the real New York save (one block) is named "New York" alone, the Northeast names
// New York and Philadelphia, every block has a name and none repeats (T-2447), painted into a recording canvas the
// names sit on their anchors, are 1.25 to 2.5 station-name sizes and meet no other, each anchor lies on its block's own
// city stations, and below the switch no station names are drawn while above it no city names are.
//   node trainworld/checks/t2438-city-names.mjs [jpn,ne,nyc,tor,na]      (about two minutes with Japan)
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { openSave, SAVES, waterOf } from './diagram-saves.mjs';
import { ARTIFACTS, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing } from '@app/src/diagram/draw.ts';
export { zoneBorders } from '@app/src/diagram/borders.ts'; export { toMerc01 } from '@app/src/map/geo.ts';
export * from '@app/src/diagram/cityNames.ts';
export { placeNames, levelAt, guessWidth, NAME_LINE } from '@app/src/diagram/names.ts'; export { zoneName } from '@app/src/game/zoneNames.ts';`);
const TW = new URL('../', import.meta.url);
const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
const OUTL = JSON.parse(readFileSync(new URL('data/packs/outlines.json', TW)));
SAVES.tor = { file: new URL('data/packs/real/tor.save', TW), lon: -79.3832, lat: 43.6534 };
// T-2581: Anita's North America game, Washington's block holding the lines to Atlanta
SAVES.na = { file: new URL('../artifacts/trainworld/anita-2026-11-05/north-america-day-361.save', TW), lon: -73.985, lat: 40.758 };
const which = (process.argv[2] ?? `jpn,ne,nyc,tor${existsSync(SAVES.na.file) ? ',na' : ''}`).split(',');
const FONT = 12;
let failed = 0;
const out = {};
const t0 = performance.now();
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const city of which) {
  const s = openSave(city);
  const { g } = s.graph();
  const text = [...g.key].map((k) => s.name(k) ?? '');
  // the zones' outlines in the graph's frame, as view.ts sends them, from the save's own frame (T-2542); and the sea on
  // the ground the page's layout reads (T-2541)
  const { frame: [fx, fy, K], box: [w, so, e, n], sea } = waterOf(s, g), k = 1 / K;
  s.api.free();
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
  const lay = L.layoutDiagram(g, L.LAYOUT, null, undefined, sea);
  const d = L.drawing(g, lay);
  const widths = text.map((t) => L.guessWidth(t, FONT));
  const plan = L.placeNames(g, lay, d, text, widths, widths.map(() => 0), FONT);
  const b = L.zoneBorders(g, lay.x, lay.y, d, shapes);
  const zones = b.zones.map((id) => { const c = byId.get(id), name = L.zoneName(id, c.name); return { name, width: L.guessWidth(name, FONT), trips: c.localTrips ?? 0, city: c.kind !== 'zone' }; });
  // T-2447: the first placing is as the page does it before any place list has arrived; the zones that asked
  // (cn0.want) then get theirs, from data/packs/<id>.places.json as the page loads them, and it is placed again
  const cn0 = L.placeCityNames(g, lay, lay.blocks, b.zoneOf, zones, FONT, b.zones);
  // T-2640: the entries after the blocks' are zones with stations and no named block; "names as before" is of the blocks
  const nbk = lay.blocks.list.length, before = cn0.text.slice(0, nbk).filter(Boolean).length;
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
  const cn = cn0.want.length ? L.placeCityNames(g, lay, lay.blocks, b.zoneOf, zones, FONT, b.zones) : cn0;
  const r = { ms: 0, blocks: lay.blocks.list.length };
  r.ms = Math.round(performance.now() - t0);
  r.named = new Set(cn.text.filter(Boolean)).size;
  r.namedBefore = before;
  r.unnamed = cn.text.slice(0, nbk).filter((t) => !t).length;
  r.zoneNames = cn.text.length - nbk;
  r.repeats = cn.text.filter((t, i) => t && cn.text.indexOf(t) !== i);
  r.given = cn.text.slice(0, nbk).map((t, i) => [t, cn0.text[i]]).filter(([t, o]) => t && !o).map(([t]) => t);
  out[city] = r;
  console.log(`     ${city}: ${r.blocks} blocks, ${before} named before, ${r.named} names after, ${r.zoneNames} of them zones without a block`);
  console.log(`     by stations: ${[...cn.order].slice(0, 40).map((q) => `${cn.text[q]} ${cn.rank[q]}${q >= nbk ? 'z' : ''} (${Math.round(cn.mx[q] / d.pitch)},${Math.round(cn.my[q] / d.pitch)} pitches)`).join('; ')}`);
  console.log(`     given to blocks that had none: ${r.given.join(', ') || '-'}`);
  say(r.unnamed === 0, `${city}: every block has a name`, `${r.unnamed} without`);
  say(!r.repeats.length, `${city}: no name twice`, r.repeats.join(', '));
  if (city !== 'jpn') say(r.given.length === 0 && before === r.blocks, `${city}: names as before (one per block, none given)`);
  if (city === 'nyc') say(cn.text[cn.order[0]] === 'New York' && cn.order.filter((x) => x < nbk).length === 1, 'nyc: "New York" the one block name, first', [...cn.order].map((x) => cn.text[x]).join(', '));
  if (city === 'ne') say(['New York', 'Philadelphia'].every((n) => [...cn.order].some((x) => x < nbk && cn.text[x] === n)) && cn.order.filter((x) => x < nbk).length === 2, 'ne: New York and Philadelphia the block names', [...cn.order].map((x) => cn.text[x]).join(', '));
  // T-2521: painted into a recording canvas at several pitches, each name is centred on its block's box, at least
  // BIG_MIN station-name sizes, and no two names meet
  const box = d.box, W = 1600, H = 1000;
  for (const px of [0.2, 0.5, 1, 1.4]) {
    const scale = (px / d.pitch) , calls = [], rec = { save() {}, restore() {}, setTransform() {}, fillText: (t, x, y) => calls.push({ t, x, y, f: parseFloat(rec.font) }), font: '', fillStyle: '', globalAlpha: 1, textBaseline: '', textAlign: '' };
    for (const where of [0.5, 0.25]) {
      calls.length = 0;
      const cx = box[0] + (box[2] - box[0]) * where, cy = box[1] + (box[3] - box[1]) * where;
      L.paintCityNames(rec, cn, d.pitch, W, H, cx, cy, scale, 1, { ink: '#777', font: 'sans' });
      let bad = '';
      // T-2640: at most the screen's count, and at least FEW_NAMES where that many have their middle in view
      const cap = Math.min(L.MAX_NAMES, Math.max(L.MIN_NAMES, Math.round((W * H) / L.NAME_AREA_PX)));
      const inView = [...cn.order].filter((q) => { const ax = W / 2 + (cn.mx[q] - cx) * scale, ay = H / 2 - (cn.my[q] - cy) * scale; return ax >= 0 && ax <= W && ay >= 0 && ay <= H; }).length;
      if (calls.length > cap) bad ||= `${calls.length} names over the cap ${cap}`;
      if (calls.length < Math.min(L.FEW_NAMES, inView)) bad ||= `${calls.length} names where ${inView} are in view`;
      for (const c of calls) {
        const b = [...cn.order].find((q) => cn.text[q] === c.t);
        const ax = W / 2 + (cn.mx[b] - cx) * scale, ay = H / 2 - (cn.my[b] - cy) * scale;
        // a name whose middle is off screen moves into view
        if (ax >= 0 && ax <= W && ay >= 0 && ay <= H && (Math.abs(c.x - ax) > 1e-6 || Math.abs(c.y - ay) > 1e-6)) bad ||= `${c.t} off centre`;
        if (!(c.x >= 0 && c.x <= W && c.y >= 0 && c.y <= H)) bad ||= `${c.t} off screen`;
        if (c.f < 1.25 * FONT - 1e-6 || c.f > 2.5 * FONT + 1e-6) bad ||= `${c.t} at ${c.f.toFixed(1)} px`;
      }
      for (let i = 0; i < calls.length; i++) for (let j = 0; j < i; j++) {
        const a = calls[i], o = calls[j], wa = L.guessWidth(a.t, a.f), wo = L.guessWidth(o.t, o.f), ha = a.f * L.NAME_LINE, ho = o.f * L.NAME_LINE;
        if (Math.abs(a.x - o.x) < (wa + wo) / 2 && Math.abs(a.y - o.y) < (ha + ho) / 2) bad ||= `${a.t} meets ${o.t}`;
      }
      say(!bad, `${city}: at ${px} px a pitch (view ${where}) ${calls.length} names, centred, 1.25 to 2.5 sizes, apart`, bad);
    }
  }
  // T-2674: the whole drawing in a 1400 by 900 view shows names spread over it, not the busiest only: in North America
  // at least 2 in the west third, 1 in the middle third and 1 in the east third; in Japan one in each of the south,
  // middle and north thirds
  {
    const bw = box[2] - box[0], bh = box[3] - box[1], WW = +(process.env.WHOLE_W ?? 1400), HH = +(process.env.WHOLE_H ?? 900), sc = Math.min((0.96 * WW) / bw, (0.96 * HH) / bh), cx = (box[0] + box[2]) / 2, cy = (box[1] + box[3]) / 2, got = [];
    L.paintCityNames({ save() {}, restore() {}, setTransform() {}, fillText: (t, x, y) => got.push({ t, x, y }), font: '', fillStyle: '', globalAlpha: 1, textBaseline: '', textAlign: '' }, cn, d.pitch, WW, HH, cx, cy, sc, 1, { ink: '#777', font: 'sans' });
    console.log(`     whole drawing: ${got.map((c) => `${c.t} (${Math.round(c.x)},${Math.round(c.y)})`).join('; ')}`);
    const third = (v, n, k) => got.filter((c) => (c[v] / (v === 'x' ? WW : HH)) * 3 >= k && (c[v] / (v === 'x' ? WW : HH)) * 3 < k + 1).length >= n;
    if (city === 'na') say(third('x', 2, 0) && third('x', 1, 1) && third('x', 1, 2), 'na: the whole drawing has names in its west (2), middle and east thirds', got.map((c) => c.t).join(', '));
    if (city === 'jpn') say(third('y', 1, 0) && third('y', 1, 1) && third('y', 1, 2), 'jpn: the whole drawing has names in its north, middle and south thirds', got.map((c) => c.t).join(', '));
  }
  // T-2581: each name sits inside the drawn box of its block's own city stations (a block's lines out of its city,
  // Washington's to Atlanta, once moved it far off)
  let off = '';
  for (const q of cn.order) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < lay.blocks.of.length; i++) if (lay.blocks.of[i] === q && lay.blocks.core[i]) (x0 = Math.min(x0, lay.x[i])), (y0 = Math.min(y0, lay.y[i])), (x1 = Math.max(x1, lay.x[i])), (y1 = Math.max(y1, lay.y[i]));
    if (x0 <= x1 && !(cn.mx[q] >= x0 && cn.mx[q] <= x1 && cn.my[q] >= y0 && cn.my[q] <= y1)) off += `${cn.text[q]} `;
  }
  say(!off, `${city}: every name on its own city's stations`, off);
  const below = L.CITY_SWITCH * 0.99, above = L.CITY_SWITCH * 1.01;
  const nCalls = (p) => { let n = 0; L.paintCityNames({ save() {}, setTransform() {}, fillText: () => n++ }, cn, d.pitch, W, H, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, p / d.pitch, 1, { ink: '', font: '' }); return n; };
  say(L.levelAt(plan, below) < 0 && nCalls(above) >= 0 && L.levelAt(plan, above) >= 0,
    `${city}: no station names below ${L.CITY_SWITCH.toFixed(2)} px a pitch, station names from it up (the city names at both)`);
}
mkdirSync(`${ARTIFACTS}t2438`, { recursive: true });
writeFileSync(`${ARTIFACTS}t2438/figures.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
