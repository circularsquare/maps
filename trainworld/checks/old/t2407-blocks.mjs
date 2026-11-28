// T-2407: the network diagram's city blocks (layout.ts' block stage, notes/T-2407.md) on a coast-to-coast mock:
// New York's real network at New York, Sendai's (trimmed to 35 km) standing in for Chicago and Toronto's for Los
// Angeles, joined by one fast line through Philadelphia ... Las Vegas (notes/T-2406.md). Laid out as the page's
// worker does: the cities alone, the line added (a rebuild), the same graph fresh (a page load), then two edits
// rebuilt (a stop added to the line; a New York line removed). Then the real New York save alone, one block.
//   node trainworld/checks/t2407-blocks.mjs [jpn] [out=<folder>]    (about a minute; with jpn, all Japan too, a
//   minute or two more). Figures and the drawn cases in artifacts/trainworld/t2407/[<folder>/]; t2407-draw.py
//   makes pictures of them.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { app, openSave } from './diagram-saves.mjs';
import { TrackApi } from '../app/src/wasm/track/sim.js';
import { ARTIFACTS, REAL_SAVE, loadApp } from './lib.mjs';

globalThis.Path2D ??= class { moveTo() {} lineTo() {} quadraticCurveTo() {} };
const L = await loadApp(`export * from '@app/src/diagram/layout.ts'; export { drawing, pillSize } from '@app/src/diagram/draw.ts';`);
const SUB = process.argv.find((a) => a.startsWith('out='))?.slice(4);
const OUT = `${ARTIFACTS}t2407/${SUB ? SUB + '/' : ''}`;
mkdirSync(OUT, { recursive: true });
const TW = new URL('../', import.meta.url);
let failed = 0;
const say = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

/** a save opened as the clock worker would hand it to the diagram, with each node's lon, lat */
function openFile(file, lon, lat) {
  const raw = gunzipSync(readFileSync(file));
  const api = new TrackApi(lon, lat, true);
  if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error(`${file} did not load`);
  const frame = app.makeFrame({ kind: 'equirect', lon, lat });
  app.setAnchor(app.makeAnchor(lon, lat, 1));
  const local = (x, y) => { const [lng, la] = frame.toLonLat(x, y); return app.toLocal(lng, la); };
  const ei = api.edges_info(), edges = [], pts = new Map();
  for (let i = 0; i < ei.length; i += 10) {
    const id = ei[i];
    edges.push({ id, a: ei[i + 1], b: ei[i + 2] });
    const r = api.edge_render(id), p = new Float32Array(r.length);
    for (let k = 0; k < r.length; k += 3) { const [x, y] = local(r[k], r[k + 1]); p[k] = x; p[k + 1] = y; p[k + 2] = r[k + 2]; }
    pts.set(id, p);
  }
  const li = api.lines_info(), lines = [];
  for (let i = 0; i < li.length; i += 22) {
    const id = li[i];
    lines.push({ id, ok: li[i + 2] === 1, bus: api.is_bus(id) || undefined, path: [...api.line_path(id)], stops: [...api.line_stops(id)], colour: '#' + li[i + 1].toString(16).padStart(6, '0'), name: api.line_name(id), fast: !api.is_bus(id) && api.line_fast(id) });
  }
  const ni = api.nodes_info(), geo = [], xy = new Map();
  let top = 0;
  for (let i = 0; i < ni.length; i += 10) {
    top = Math.max(top, ni[i]);
    xy.set(ni[i], frame.toLonLat(ni[i + 1], ni[i + 2]));
    if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
  }
  const fl = api.station_links(), links = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = app.buildComplexes(geo, links);
  const complexOf = new Uint32Array(top + 1).fill(0xffffffff);
  for (const g of geo) complexOf[Number(g.id)] = cx.of.get(g.id);
  const g = app.diagramGraph(lines, edges, pts, complexOf, app.besideSamples(app.bandEdges(lines), pts, new Map(edges.map((e) => [e.id, e]))));
  const name = (n) => api.node_name(n);
  const lonlat = (n) => xy.get(n);
  return { g, name, lonlat, free: () => api.free() };
}

// ---- the mock: three cities moved onto the US map, one line between them
const LON0 = -96, LAT0 = 38, KX = 111320 * Math.cos((LAT0 * Math.PI) / 180), KY = 111320;
const proj = (lng, lat) => [(lng - LON0) * KX, (lat - LAT0) * KY];
const SRC = [
  { tag: 'New York', file: REAL_SAVE, src: [-73.985, 40.758], to: [-73.985, 40.758], hub: /Penn/, off: 0 },
  { tag: 'Chicago (Sendai)', file: new URL('data/packs/real/sdj.save', TW), src: [140.8824, 38.2601], to: [-87.63, 41.88], hub: /^仙台駅$/, off: 10000000, R: 35000 },
  { tag: 'Los Angeles (Toronto)', file: new URL('data/packs/real/tor.save', TW), src: [-79.3832, 43.6534], to: [-118.24, 34.05], hub: /^Union/, off: 20000000 },
];
const STOPS = [
  ['New York'], ['Philadelphia', -75.18, 39.956], ['Harrisburg', -76.88, 40.26], ['Pittsburgh', -80.0, 40.44], ['Cleveland', -81.69, 41.5],
  ['Toledo', -83.54, 41.65], ['Chicago (Sendai)'], ['Omaha', -95.93, 41.26], ['Denver', -104.99, 39.74], ['Grand Junction', -108.55, 39.06],
  ['Salt Lake City', -111.89, 40.76], ['Las Vegas', -115.14, 36.17], ['Los Angeles (Toronto)'],
];
/** the edit: a stop added to the line between Omaha and Denver */
const EXTRA = ['North Platte', -100.77, 41.12];

const key = [], X = [], Y = [], name = [], block = [], lines = [], edges = new Map(), hubs = {};
SRC.forEach((s, bi) => {
  const o = openFile(s.file, s.src[0], s.src[1]);
  const g0 = o.g, [cx, cy] = proj(...s.to), ks = Math.cos((s.src[1] * Math.PI) / 180) * 111320;
  const g = s.R ? trim(g0, (i) => { const [lng, lat] = o.lonlat(g0.key[i]); return Math.hypot((lng - s.src[0]) * ks, (lat - s.src[1]) * 111320) < s.R; }) : g0;
  const base = key.length;
  for (let i = 0; i < g.key.length; i++) {
    const [lng, lat] = o.lonlat(g.key[i]);
    key.push(g.key[i] + s.off);
    X.push(cx + (lng - s.src[0]) * ks);
    Y.push(cy + (lat - s.src[1]) * 111320);
    const nm = o.name(g.key[i]) ?? '';
    name.push(nm);
    block.push(bi);
    if (s.hub.test(nm) && hubs[s.tag] === undefined) hubs[s.tag] = base + i;
  }
  const l0 = lines.length;
  for (const l of g.lines) lines.push({ id: l.id + s.off, colour: l.colour, fast: l.fast, chain: Int32Array.from(l.chain, (c) => c + base), stop: l.stop });
  for (let e = 0; e < g.ea.length; e++) {
    const a = g.ea[e] + base, b = g.eb[e] + base, k = `${a},${b}`;
    edges.set(k, [a, b, []]);
    for (let j = g.first[e]; j < g.first[e + 1]; j++) edges.get(k)[2].push([g.on[j] + l0, g.stroke[j] + l0]);
  }
  o.free();
});
/** the graph with only the lines whose every node passes `keep`, and their nodes */
function trim(g, keep) {
  const okL = g.lines.map((l) => [...l.chain].every(keep));
  const used = new Set();
  g.lines.forEach((l, i) => { if (okL[i]) for (const c of l.chain) used.add(c); });
  const ids = [...used].sort((a, b) => a - b), re = new Map(ids.map((v, i) => [v, i]));
  const lmap = new Map(), ls = [];
  g.lines.forEach((l, i) => { if (okL[i]) { lmap.set(i, ls.length); ls.push({ ...l, chain: Int32Array.from(l.chain, (c) => re.get(c)) }); } });
  const ea = [], eb = [], first = [0], on = [], stroke = [];
  for (let e = 0; e < g.ea.length; e++) {
    if (!used.has(g.ea[e]) || !used.has(g.eb[e])) continue;
    const ons = [];
    for (let j = g.first[e]; j < g.first[e + 1]; j++) if (lmap.has(g.on[j])) ons.push([lmap.get(g.on[j]), lmap.get(g.stroke[j]) ?? lmap.get(g.on[j])]);
    if (!ons.length) continue;
    ea.push(re.get(g.ea[e])); eb.push(re.get(g.eb[e])); on.push(...ons.map((p) => p[0])); stroke.push(...ons.map((p) => p[1])); first.push(on.length);
  }
  return { key: Int32Array.from(ids, (i) => g.key[i]), x: Float64Array.from(ids, (i) => g.x[i]), y: Float64Array.from(ids, (i) => g.y[i]), lines: ls, ea: Int32Array.from(ea), eb: Int32Array.from(eb), first: Int32Array.from(first), on: Int32Array.from(on), stroke: Int32Array.from(stroke), mid: g.mid, k: g.k };
}
const nCity = key.length;
const stopNode = ([nm, lng, lat], k) => {
  if (lng === undefined) return hubs[nm];
  key.push(30000000 + k); const [x, y] = proj(lng, lat); X.push(x); Y.push(y); name.push(nm); block.push(-1);
  return key.length - 1;
};
const chain = STOPS.map(stopNode);
const extra = stopNode(EXTRA, 100);
const chain2 = [...chain.slice(0, 8), extra, ...chain.slice(8)];
const nyLast = lines.findIndex((l, i) => i + 1 < lines.length && block[lines[i + 1].chain[0]] !== 0) ;
// T-2417: a New York line ending at Coney Island extended one stop, 2 km west toward Sea Gate (east, the stop lands
// past Brighton Beach's stations, so the hop is drawn across them, about 5 hops, with blocks or without)
const coney = name.findIndex((nm, i) => block[i] === 0 && /Coney Island/.test(nm));
const coneyLine = lines.findIndex((l) => block[l.chain[0]] === 0 && (l.chain[0] === coney || l.chain[l.chain.length - 1] === coney));
key.push(30000000 + 200); X.push(X[coney] - 2000); Y.push(Y[coney] - 300); name.push('Coney Island extension'); block.push(0);
const coneyExt = key.length - 1;

/** the graph: the cities, with the line along `ch` if given, without line `drop` if given; nodes not on any
 * line left out (keys kept, so a rebuild finds each node again) */
function graph(ch, drop = -1, ext = false) {
  let ls = lines.filter((_, i) => i !== drop);
  const es = new Map();
  // each edge's entries: [line, the line whose stroke it is drawn in], as indices into ls
  for (const [k, v] of edges) {
    const on = v[2].filter((p) => p[0] !== drop).map((p) => [ls.indexOf(lines[p[0]]), ls.indexOf(lines[p[1] === drop ? p[0] : p[1]])]);
    if (on.length) es.set(k, [v[0], v[1], on]);
  }
  if (ch) {
    ls.push({ id: 99999999, colour: '#c2185b', fast: true, chain: Int32Array.from(ch), stop: new Uint8Array(ch.length).fill(1) });
    for (let i = 1; i < ch.length; i++) {
      const a = Math.min(ch[i - 1], ch[i]), b = Math.max(ch[i - 1], ch[i]), k = `${a},${b}`;
      if (!es.has(k)) es.set(k, [a, b, []]);
      es.get(k)[2].push([ls.length - 1, ls.length - 1]);
    }
  }
  if (ext) {
    const li = ls.indexOf(lines[coneyLine]), l = ls[li], atEnd = l.chain[l.chain.length - 1] === coney;
    ls = ls.map((x, i) => (i !== li ? x : { ...l, chain: Int32Array.from(atEnd ? [...l.chain, coneyExt] : [coneyExt, ...l.chain]), stop: Uint8Array.from(atEnd ? [...l.stop, 1] : [1, ...l.stop]) }));
    es.set(`${coney},${coneyExt}`, [coney, coneyExt, [[li, li]]]);
  }
  const used = new Set();
  for (const l of ls) for (const c of l.chain) used.add(c);
  const ids = [...used].sort((a, b) => a - b), re = new Map(ids.map((v, i) => [v, i]));
  const E = [...es.values()].map((e) => [re.get(e[0]), re.get(e[1]), e[2]]).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const first = new Int32Array(E.length + 1), on = [], stroke = [];
  E.forEach((e, i) => {
    first[i] = on.length;
    const seen = new Set();
    for (const [l, s] of e[2]) if (!seen.has(l)) seen.add(l), on.push(l), stroke.push(s);
  });
  first[E.length] = on.length;
  return {
    ids, key: Int32Array.from(ids, (i) => key[i]), x: Float64Array.from(ids, (i) => X[i]), y: Float64Array.from(ids, (i) => Y[i]),
    lines: ls.map((l) => ({ ...l, chain: Int32Array.from(l.chain, (c) => re.get(c)) })),
    ea: Int32Array.from(E, (e) => e[0]), eb: Int32Array.from(E, (e) => e[1]), first, on: Int32Array.from(on), stroke: Int32Array.from(stroke), mid: [0, 0], k: 1,
  };
}

const out = {};
/** a case's figures, and its drawing for t2407-draw.py */
function dump(tag, g, lay, ch, extraFig = {}) {
  const d = L.drawing(g, lay), n = g.key.length, pu = lay.pu;
  const cityOf = g.ids.map((i) => block[i]);
  const boxes = SRC.map((s, bi) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) if (cityOf[i] === bi) (x0 = Math.min(x0, lay.x[i])), (y0 = Math.min(y0, lay.y[i])), (x1 = Math.max(x1, lay.x[i])), (y1 = Math.max(y1, lay.y[i]));
    return { tag: s.tag, box: [x0, y0, x1, y1] };
  });
  const at = new Map(g.ids.map((v, i) => [v, i]));
  const line = ch ? ch.map((c) => ({ name: name[c], x: lay.x[at.get(c)], y: lay.y[at.get(c)] })) : [];
  // the line's drawn length from the New York hub to the Los Angeles hub, pitches
  let span = 0;
  for (let i = 1; i < line.length; i++) span += Math.hypot(line[i].x - line[i - 1].x, line[i].y - line[i - 1].y);
  let overlaps = 0;
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i].box, b = boxes[j].box;
    if (a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]) overlaps++, console.log(`     overlap ${boxes[i].tag} ${a.map((v) => Math.round(v / pu))} | ${boxes[j].tag} ${b.map((v) => Math.round(v / pu))}`);
  }
  const fig = {
    ms: Math.round(lay.ms.total), msParts: Object.fromEntries(Object.entries(lay.ms).map(([k, v]) => [k, Math.round(v)])), pu: +pu.toFixed(2), hop: +lay.hop.toFixed(1),
    blocks: lay.blocks ? lay.blocks.list.map((b, k) => {
      const of = [...lay.blocks.of.keys()].filter((i) => lay.blocks.of[i] === k);
      return `${SRC[cityOf[of[0]]]?.tag ?? 'line'} ${of.length} (${of.filter((i) => lay.blocks.core[i]).length} city) ${Math.round((b.box[2] - b.box[0]) / pu)}x${Math.round((b.box[3] - b.box[1]) / pu)} moved ${Math.round(b.move[0] / pu)},${Math.round(b.move[1] / pu)}`;
    }) : null,
    spanPitches: Math.round(span / pu), nyWidth: Math.round(Math.max(boxes[0].box[2] - boxes[0].box[0], boxes[0].box[3] - boxes[0].box[1]) / pu),
    boxes: boxes.map((b) => [b.tag, Math.round((b.box[2] - b.box[0]) / pu), Math.round((b.box[3] - b.box[1]) / pu)]), overlaps, ...extraFig,
  };
  console.log(`     ${tag}: ${JSON.stringify(fig)}`);
  if (process.env.T2407_DEBUG) console.log('       B boxes', SRC.map((s, bi) => { let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity; for (let i = 0; i < n; i++) if (cityOf[i] === bi) (x0 = Math.min(x0, lay.bx[i])), (y0 = Math.min(y0, lay.by[i])), (x1 = Math.max(x1, lay.bx[i])), (y1 = Math.max(y1, lay.by[i])); return [x0, y0, x1, y1].map((v) => Math.round(v / pu)).join(','); }).join(' | '));
  if (process.env.T2407_DEBUG && ch) for (const c of ch) { const i = at.get(c); console.log(`       ${name[c].padEnd(24)} B ${Math.round(lay.bx[i] / pu)},${Math.round(lay.by[i] / pu)}  drawn ${Math.round(lay.x[i] / pu)},${Math.round(lay.y[i] / pu)}`); }
  out[tag] = fig;
  writeFileSync(`${OUT}${tag}.json`, JSON.stringify({
    tag, fig, pitch: d.pitch, stroke: d.stroke,
    ribbons: d.ribbons.map((r, i) => ({ c: d.lines[i].colour, f: d.lines[i].fast, p: Array.from(r, (v) => Math.round(v)) })),
    pills: d.pills.map((p, i) => [...p.map(Math.round), ...L.pillSize(d, d.pillOf[i]).map((v) => +v.toFixed(1))]),
    boxes, line, pos: { x: Array.from(lay.x, Math.round), y: Array.from(lay.y, Math.round) },
  }));
  return { fig, boxes };
}
/** how a rebuild moved each city: its stations' median move (the block's move, pitches), and each station's move
 * beyond it (hops): the median and 95th percentile over all city stations */
function moved(gA, layA, gB, layB) {
  const atA = new Map([...gA.key].map((k, i) => [k, i]));
  const per = SRC.map(() => []);
  gB.ids.forEach((v, i) => {
    const j = atA.get(gB.key[i]);
    if (j === undefined || block[v] < 0) return;
    per[block[v]].push([i, layB.x[i] - layA.x[j], layB.y[i] - layA.y[j]]);
  });
  const med = (a) => { const s = Float64Array.from(a).sort(); return s.length ? s[s.length >> 1] : 0; };
  const drift = [], mvs = [];
  for (const list of per) {
    const mx = med(list.map((p) => p[1])), my = med(list.map((p) => p[2]));
    mvs.push([mx, my]);
    for (const p of list) drift.push(Math.hypot(p[1] - mx, p[2] - my) / layB.hop);
  }
  drift.sort((a, b) => a - b);
  return {
    blockMovesPitches: mvs.map(([x, y]) => Math.round(Math.hypot(x, y) / layB.pu)),
    // against New York's move
    relMovesPitches: mvs.map(([x, y]) => Math.round(Math.hypot(x - mvs[0][0], y - mvs[0][1]) / layB.pu)),
    driftMedianHops: +drift[drift.length >> 1].toFixed(3), driftP95Hops: +drift[Math.floor(drift.length * 0.95)].toFixed(2),
  };
}

const g0 = graph(null), g1 = graph(chain), g2 = graph(chain2), g3 = graph(chain2, nyLast);
const lay0 = L.layoutDiagram(g0);
const f0 = dump('cities', g0, lay0, null);
const lay1 = L.layoutDiagram(g1, L.LAYOUT, L.lastPlaces(g0, lay0));
const m1 = moved(g0, lay0, g1, lay1);
const f1 = dump('rebuild', g1, lay1, chain, m1);
const layF = L.layoutDiagram(g1);
const fF = dump('fresh', g1, layF, chain);
const lay2 = L.layoutDiagram(g2, L.LAYOUT, L.lastPlaces(g1, lay1));
const m2 = moved(g1, lay1, g2, lay2);
dump('edit-stop', g2, lay2, chain2, m2);
const lay3 = L.layoutDiagram(g3, L.LAYOUT, L.lastPlaces(g2, lay2));
const m3 = moved(g2, lay2, g3, lay3);
dump('edit-nyline', g3, lay3, chain2, m3);

// T-2417: the Coney Island extension on the game with the line, rebuilt
const g4 = graph(chain, -1, true), lay4 = L.layoutDiagram(g4, L.LAYOUT, L.lastPlaces(g1, lay1));
const m4 = moved(g1, lay1, g4, lay4), at4 = new Map(g4.ids.map((v, i) => [v, i]));
const extHops = Math.hypot(lay4.x[at4.get(coneyExt)] - lay4.x[at4.get(coney)], lay4.y[at4.get(coneyExt)] - lay4.y[at4.get(coney)]) / lay4.hop;
dump('edit-coney', g4, lay4, chain, { ...m4, extHops: +extHops.toFixed(2) });

for (const [tag, f] of [['fresh', fF], ['rebuild', f1]]) say(f.fig.spanPitches < 2 * f.fig.nyWidth, `${tag}: New York to Los Angeles drawn under 2 New York widths`, `${f.fig.spanPitches} pitches, New York ${f.fig.nyWidth}`);
for (const [tag, f] of [['cities', f0], ['fresh', fF], ['rebuild', f1]]) say(f.fig.overlaps === 0, `${tag}: no two city boxes overlap`, `${f.fig.overlaps}`);
const grew = f1.fig.boxes.map((b, i) => [b[0], b[1] / f0.fig.boxes[i][1], b[2] / f0.fig.boxes[i][2]]).filter((b) => b[1] > 1.02 || b[2] > 1.02);
say(!grew.length, 'no city box grows over 2% when the line is added (a rebuild)', grew.length ? grew.map((b) => `${b[0]} x${b[1].toFixed(2)} y${b[2].toFixed(2)}`).join(', ') : f1.fig.boxes.map((b) => b.join(' ')).join('; '));
say(m1.driftMedianHops < 0.1, 'adding the line: the median city station moves under 0.1 hops beyond its block', `${m1.driftMedianHops} (95th percentile ${m1.driftP95Hops}); blocks moved ${m1.blockMovesPitches.join(', ')} pitches`);
// a stop between Omaha and Denver lengthens that stretch's targets, so Los Angeles moves out; New York and
// Chicago should not move against each other
say(m2.blockMovesPitches[0] <= 5 && m2.relMovesPitches[1] <= 5 && m2.driftMedianHops < 0.1, 'a stop added to the line: New York and Chicago move 5 pitches or less, the median station under 0.1 hops beyond its block', `moves ${m2.blockMovesPitches.join(', ')} pitches (against New York ${m2.relMovesPitches.join(', ')}); ${m2.driftMedianHops} hops (95th ${m2.driftP95Hops})`);
say(Math.max(...m3.blockMovesPitches) <= 5 && m3.driftMedianHops < 0.1, 'a New York line removed: blocks move 5 pitches or less, the median station under 0.1 hops beyond its block', `moves ${m3.blockMovesPitches.join(', ')} pitches; ${m3.driftMedianHops} hops (95th ${m3.driftP95Hops})`);

say(extHops < 3 && Math.max(...m4.blockMovesPitches) <= 5, 'T-2417: a stop 2 km past Coney Island is drawn under 3 hops from it, blocks moving 5 pitches or less', `${extHops.toFixed(2)} hops; moves ${m4.blockMovesPitches.join(', ')} pitches; ${m4.driftMedianHops} hops (95th ${m4.driftP95Hops})`);

// ---- the real New York save: one block, laid out as without blocks
{
  const s = openSave('nyc'), { g } = s.graph();
  s.api.free();
  const lay = L.layoutDiagram(g);
  let h = 0;
  for (let i = 0; i < g.key.length; i++) h = (h * 31 + Math.round(lay.x[i] * 1000) + 7 * Math.round(lay.y[i] * 1000)) % 2147483647;
  out.nyc = { complexes: g.key.length, ms: Math.round(lay.ms.total), blocks: lay.blocks ? lay.blocks.list.length : null, active: lay.blocks ? lay.blocks.active : null, hash: h };
  console.log(`     nyc: ${JSON.stringify(out.nyc)}`);
  say(!lay.blocks || (lay.blocks.list.length === 1 && !lay.blocks.active), 'the real New York save is one block, the stage idle', `${out.nyc.blocks} blocks`);
}

// ---- all Japan: Okinawa packed near Kyushu
if (process.argv.includes('jpn')) {
  const s = openSave('jpn'), { g } = s.graph();
  const names = [...g.key].map((k) => s.name(k) ?? '');
  s.api.free();
  const t0 = performance.now();
  const lay = L.layoutDiagram(g);
  const ms = performance.now() - t0, pu = lay.pu;
  const r = { complexes: g.key.length, ms: Math.round(ms), msParts: Object.fromEntries(Object.entries(lay.ms).map(([k, v]) => [k, Math.round(v)])) };
  if (lay.blocks) {
    const B = lay.blocks;
    r.blocks = B.list.length;
    r.free = [...B.of].filter((b) => b < 0).length;
    const size = B.list.map(() => 0);
    for (const b of B.of) if (b >= 0) size[b]++;
    r.biggest = B.list.map((b, k) => [k, size[k], b]).sort((p, q) => q[1] - p[1]).slice(0, 6)
      .map(([k, s, b]) => `${names[B.of.indexOf(k)]} ${s} ${Math.round((b.box[2] - b.box[0]) / pu)}x${Math.round((b.box[3] - b.box[1]) / pu)}`);
    // Okinawa: the block of Naha's monorail
    const oki = names.map((nm, i) => (/那覇空港|首里|おもろまち/.test(nm) ? B.of[i] : -1)).filter((b) => b >= 0);
    if (oki.length) {
      const ob = oki[0], bo = B.list[ob].box;
      let best = Infinity, near = -1;
      B.list.forEach((b, k) => {
        if (k === ob) return;
        const dx = Math.max(0, b.box[0] - bo[2], bo[0] - b.box[2]), dy = Math.max(0, b.box[1] - bo[3], bo[1] - b.box[3]), d = Math.hypot(dx, dy);
        if (d < best) (best = d), (near = k);
      });
      const at = B.of.indexOf(near), nb = B.list[near].box;
      r.okinawa = { gapPitches: Math.round(best / pu), nearest: names[at], nearestWidth: Math.round(Math.max(nb[2] - nb[0], nb[3] - nb[1]) / pu), okinawaWidth: Math.round(Math.max(bo[2] - bo[0], bo[3] - bo[1]) / pu) };
    }
  }
  out.jpn = r;
  console.log(`     jpn: ${JSON.stringify(r)}`);
  if (r.okinawa) say(r.okinawa.gapPitches <= r.okinawa.nearestWidth, 'all Japan: Okinawa within a box width of its nearest block', JSON.stringify(r.okinawa));
}
writeFileSync(`${OUT}figures.json`, JSON.stringify(out, null, 1));
process.exit(failed ? 1 : 0);
