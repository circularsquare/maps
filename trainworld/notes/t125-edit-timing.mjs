// T-125: what the world frame costs an edit. The same 300 route edits (preview, then add; 15 to
// 30 km, in tunnels, around New York and Philadelphia, both city water rasters loaded) on an empty
// network in the legacy frame and in the world frame. Run from the maps root after the WASM is
// built: node trainworld/notes/t125-edit-timing.mjs. Writes T-125-edit-timing.json.
import { readFile, writeFile } from 'node:fs/promises';
const mod = await import('../app/src/wasm/track/sim.js');
await mod.default({ module_or_path: await readFile('trainworld/app/src/wasm/track/sim_bg.wasm') });
const { TrackApi } = mod;
async function water(api) {
  api.clear_water();
  for (const id of ['nyc', 'phl']) {
    const head = JSON.parse(await readFile(`trainworld/data/packs/${id}.json`, 'utf8'));
    const bin = await readFile(`trainworld/data/packs/${id}.bin`);
    const row = head.arrays.find(a => a.name === 'water_row'), xs = head.arrays.find(a => a.name === 'water_x');
    const rows = new Uint32Array(bin.buffer.slice(bin.byteOffset + row.offset, bin.byteOffset + row.offset + row.count * 4));
    const runs = new Uint16Array(bin.buffer.slice(bin.byteOffset + xs.offset, bin.byteOffset + xs.offset + xs.count * 2));
    const w = head.water;
    api.add_pack_water(w.cell_m, w.origin_m[0], w.origin_m[1], w.size[0], w.size[1], rows, runs, head.origin.lon, head.origin.lat);
  }
}
let seed = 12345;
const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
async function run(world) {
  const api = new TrackApi(-73.985, 40.758, true);
  api.set_empty_frame(world);
  await water(api);
  api.set_cash(1e9);
  seed = 12345;
  const centres = [[-73.985, 40.758], [-75.1636, 39.9526]];
  const t = { preview: [], add: [], refused: 0 };
  for (let i = 0; i < 300; i++) {
    const [lon, lat] = centres[i % 2];
    const a = api.from_lonlat(lon + (rnd() - 0.5) * 0.5, lat + (rnd() - 0.5) * 0.4);
    const ang = rnd() * Math.PI * 2, len = 15000 + rnd() * 15000;
    const ends = new Float64Array([0, 0, a[0], a[1], -2, 0, 0, a[0] + len * Math.cos(ang), a[1] + len * Math.sin(ang), -2]);
    let t0 = performance.now();
    const p = api.preview_route(ends, new Float64Array(0), 2, false);
    t.preview.push(performance.now() - t0);
    if (p[0] !== 1) { t.refused++; continue; }
    t0 = performance.now();
    const r = api.add_route(ends, new Float64Array(0), 2, false);
    t.add.push(performance.now() - t0);
    if (r !== 0) t.refused++;
  }
  const stat = a => { const s = [...a].sort((x, y) => x - y); return { n: s.length, meanMs: +(s.reduce((x, y) => x + y, 0) / s.length).toFixed(3), medianMs: +s[s.length >> 1].toFixed(3), p95Ms: +s[Math.floor(s.length * 0.95)].toFixed(3) }; };
  // loading a big save and drawing it
  const t1 = performance.now();
  const bytes = api.save();
  const saveMs = performance.now() - t1;
  const t2 = performance.now();
  api.load(bytes);
  const loadMs = performance.now() - t2;
  const t3 = performance.now();
  let pts = 0;
  const ei = api.edges_info();
  for (let i = 0; i < ei.length; i += 10) pts += api.edge_render(ei[i]).length / 3;
  return { frame: world ? 'world (Lambert conformal)' : 'legacy (equirectangular)', preview: stat(t.preview), add: stat(t.add), refused: t.refused, edges: ei.length / 10, saveBytes: bytes.length, saveMs: +saveMs.toFixed(2), loadMs: +loadMs.toFixed(2), renderAllMs: +(performance.now() - t3).toFixed(2), renderPoints: pts };
}
const out = [];
for (let rep = 0; rep < 2; rep++) for (const world of [false, true]) { const r = await run(world); if (rep === 1) out.push(r); }
console.log(JSON.stringify(out, null, 1));
await writeFile('artifacts/trainworld/notes-outputs/T-125-edit-timing.json', JSON.stringify(out, null, 2) + '\n');
