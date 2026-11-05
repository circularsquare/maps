// T-2159, T-2160: a save opened in the track model (WASM) and turned into what the clock worker hands
// the network diagram: the rail lines, edges, each edge's local polyline, each node's complex and
// corridors.ts' beside table, then the diagram graph. Shared by the t2159 and t2160 checks.
import { existsSync, readFileSync, openSync, readSync, closeSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

export const app = await loadApp(`export { diagramGraph, bandEdges } from '@app/src/diagram/graph.ts';
export { besideSamples } from '@app/src/map/corridors.ts';
export { makeAnchor, setAnchor, toLocal } from '@app/src/map/geo.ts';
export { makeFrame } from '@app/src/game/frame.ts';
export { buildComplexes } from '@app/src/game/complexes.ts';
export { groundSea } from '@app/src/diagram/waterReal.ts';
export { toMerc01, EARTH_C } from '@app/src/map/geo.ts';`);
/** the layout's modules (a bundle of their own: they share nothing with the graph's) */
export const layoutApp = () => loadApp(`export * from '@app/src/diagram/layout.ts';`);
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });

const TW = new URL('../', import.meta.url);
/** the saves the checks use, with the frame origin their pack metres are measured from */
export const SAVES = {
  nyc: { file: REAL_SAVE, lon: -73.985, lat: 40.758 },
  tyo: { file: new URL('data/packs/real/tyo.save', TW), lon: 139.7, lat: 35.7 },
  jpn: { file: new URL('data/work/realnet/jpn_real.save', TW), lon: 139.7, lat: 35.7 },
  // T-2289: New York and Philadelphia, the shared Northeast save
  ne: { file: new URL('T-125-northeast.save', TW), lon: -73.985, lat: 40.758 },
  // T-2290: Anita's line 8 game as t2290-line8-game.mjs builds it
  l8: { file: new URL('../artifacts/trainworld/t2290/line8.save', TW), lon: -73.985, lat: 40.758 },
  // T-2405: expresses on tracks of their own beside locals, as t2405-express-game.mjs builds it
  xp: { file: new URL('../artifacts/trainworld/t2405/express.save', TW), lon: -73.985, lat: 40.758 },
};

/** the save `name` opened: its model, the inputs the clock worker has, and a `graph()` that builds
 * the diagram graph from them (timed: `ms` the graph alone, `besideMs` the beside table first) */
export function openSave(name) {
  const s = SAVES[name];
  const raw = gunzipSync(readFileSync(s.file));
  const api = new TrackApi(s.lon, s.lat, true);
  if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error(`${name} did not load`);
  // the save's own frame (T-2540: Japan's saves are at 139.7671, 35.6812, not the 139.7, 35.7 above, which put every
  // place 6 km west of its water)
  const o = api.frame_origin(), kind = api.frame_kind();
  const frameSpec = { kind: kind === 2 ? 'merc' : kind === 1 ? 'lcc' : 'equirect', lon: o[0], lat: o[1] };
  const frame = app.makeFrame(frameSpec);
  app.setAnchor(app.makeAnchor(s.lon, s.lat, 1));
  const local = (x, y) => { const [lng, lat] = frame.toLonLat(x, y); return app.toLocal(lng, lat); };
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
  const ni = api.nodes_info(), geo = [];
  let top = 0, stations = 0;
  for (let i = 0; i < ni.length; i += 10) {
    top = Math.max(top, ni[i]);
    if (ni[i + 7] > 0) { stations++; geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] }); }
  }
  const fl = api.station_links(), links = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = app.buildComplexes(geo, links);
  const complexOf = new Uint32Array(top + 1).fill(0xffffffff);
  for (const g of geo) complexOf[Number(g.id)] = cx.of.get(g.id);
  const ends = new Map(edges.map((e) => [e.id, e]));
  const graph = () => {
    const t0 = performance.now();
    const beside = app.besideSamples(app.bandEdges(lines), pts, ends);
    const t1 = performance.now();
    const g = app.diagramGraph(lines, edges, pts, complexOf, beside);
    const t2 = performance.now();
    return { g, ms: t2 - t1, besideMs: t1 - t0 };
  };
  /** the graph's frame in mercator (diagram.worker.ts' ZoneFrame) and its stations' box, lng, lat (call before
   * api.free()) */
  const ground = (g) => {
    const xy = new Map();
    for (let i = 0; i < ni.length; i += 10) xy.set(ni[i], [ni[i + 1], ni[i + 2]]);
    const k = g.k / app.EARTH_C;
    let fx = 0, fy = 0, w = Infinity, so = Infinity, e = -Infinity, n = -Infinity;
    g.key.forEach((key, i) => {
      const [lng, lat] = frame.toLonLat(...xy.get(key)), [mx, my] = app.toMerc01(lng, lat);
      (fx += mx - g.x[i] * k), (fy += my + g.y[i] * k);
      (w = Math.min(w, lng)), (e = Math.max(e, lng)), (so = Math.min(so, lat)), (n = Math.max(n, lat));
    });
    return { frame: [fx / g.key.length, fy / g.key.length, 1 / k], box: [w, so, e, n] };
  };
  return { api, lines, edges, pts, complexOf, cx, stations, graph, frameSpec, ground, name: (n) => api.node_name(n) };
}

const REG = JSON.parse(readFileSync(new URL('app/src/game/registry.json', TW)));
/** the water masks of the zones whose boxes meet `box` (lng, lat), read from the packs as the page fetches them */
export function masksNear(box) {
  return masksOf(REG.cities.filter(({ box: b }) => b && !(b[0] > box[2] + 0.3 || b[2] < box[0] - 0.3 || b[1] > box[3] + 0.3 || b[3] < box[1] - 0.3)).map((c) => c.id));
}

/** the water masks of zones `ids` */
export function masksOf(ids) {
  const out = [], PACKS = new URL('data/packs/', TW);
  for (const id of ids) {
    if (!existsSync(new URL(`${id}.json`, PACKS))) continue;
    const head = JSON.parse(readFileSync(new URL(`${id}.json`, PACKS)));
    const row = head.arrays.find((a) => a.name === 'water_row'), xs = head.arrays.find((a) => a.name === 'water_x');
    if (!head.water || !row || !xs) continue;
    const fd = openSync(new URL(`${id}.bin`, PACKS), 'r');
    const rows = new Uint32Array(row.count), runs = new Uint16Array(xs.count);
    readSync(fd, new Uint8Array(rows.buffer), 0, row.count * 4, row.offset);
    readSync(fd, new Uint8Array(runs.buffer), 0, xs.count * 2, xs.offset);
    closeSync(fd);
    const w = head.water;
    out.push({ id, cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1], rows, runs, lon: head.origin.lon, lat: head.origin.lat });
  }
  return out;
}

/** T-2541: the save's graph `g` with what the page's layout worker has of the water: its frame, box, masks and the sea
 * on the ground (waterReal.ts' groundSea), which layoutDiagram takes; `s` the open save (before api.free()) */
export function waterOf(s, g) {
  const { frame, box } = s.ground(g), masks = masksNear(box);
  return { frame, box, masks, sea: app.groundSea(g, masks, frame) };
}

/** T-2541: the sea a game capture's layout read (diagram-game.mjs `gameGraph`), or null for a capture without it */
export const gameSea = (cap) => (cap?.frame && cap.zones ? app.groundSea(cap.g, masksOf(cap.zones), cap.frame) : null);
