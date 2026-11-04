// T-2159, T-2160: a save opened in the track model (WASM) and turned into what the clock worker hands
// the network diagram: the rail lines, edges, each edge's local polyline, each node's complex and
// corridors.ts' beside table, then the diagram graph. Shared by the t2159 and t2160 checks.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

export const app = await loadApp(`export { diagramGraph, bandEdges } from '@app/src/diagram/graph.ts';
export { besideSamples } from '@app/src/map/corridors.ts';
export { makeAnchor, setAnchor, toLocal } from '@app/src/map/geo.ts';
export { makeFrame } from '@app/src/game/frame.ts';
export { buildComplexes } from '@app/src/game/complexes.ts';`);
/** the layout's modules (a bundle of their own: they share nothing with the graph's) */
export const layoutApp = () => loadApp(`export * from '@app/src/diagram/layout.ts';`);
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });

const TW = new URL('../', import.meta.url);
/** the saves the checks use, with the frame origin their pack metres are measured from */
export const SAVES = {
  nyc: { file: REAL_SAVE, lon: -73.985, lat: 40.758 },
  tyo: { file: new URL('data/packs/real/tyo.save', TW), lon: 139.7, lat: 35.7 },
  jpn: { file: new URL('data/work/realnet/jpn_real.save', TW), lon: 139.7, lat: 35.7 },
  // T-2290: Anita's line 8 game as t2290-line8-game.mjs builds it
  l8: { file: new URL('../artifacts/trainworld/t2290/line8.save', TW), lon: -73.985, lat: 40.758 },
};

/** the save `name` opened: its model, the inputs the clock worker has, and a `graph()` that builds
 * the diagram graph from them (timed: `ms` the graph alone, `besideMs` the beside table first) */
export function openSave(name) {
  const s = SAVES[name];
  const raw = gunzipSync(readFileSync(s.file));
  const api = new TrackApi(s.lon, s.lat, true);
  if (!api.load(raw.subarray(8 + raw.readUInt32LE(4)))) throw new Error(`${name} did not load`);
  const frame = app.makeFrame({ kind: 'equirect', lon: s.lon, lat: s.lat });
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
    lines.push({ id, ok: li[i + 2] === 1, bus: api.is_bus(id) || undefined, path: [...api.line_path(id)], stops: [...api.line_stops(id)], colour: '#' + li[i + 1].toString(16).padStart(6, '0'), name: api.line_name(id) });
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
  return { api, lines, edges, pts, complexOf, cx, stations, graph, name: (n) => api.node_name(n) };
}
