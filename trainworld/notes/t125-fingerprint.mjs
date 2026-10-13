// T-125: a fingerprint of a track save as the track model derives it, to show that a legacy save
// loads with identical geometry, lengths, costs and service after the world-frame change.
//   node trainworld/notes/t125-fingerprint.mjs <label> [save file]
// Loads the WASM currently in app/src/wasm/track (run it once before a rebuild and once after),
// both city water rasters as the clock worker installs them, and the save (default: the real NYC
// network fixture). Then, as the shared Northeast saves do, builds a Philadelphia line in the same
// legacy frame and fingerprints that too. Writes artifacts/trainworld/notes-outputs/T-125-fingerprint-<label>.json.
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
const label = process.argv[2] ?? 'run';
const savePath = process.argv[3] ?? 'trainworld/T-007-real-nyc.save';
const mod = await import('../app/src/wasm/track/sim.js');
await mod.default({ module_or_path: await readFile('trainworld/app/src/wasm/track/sim_bg.wasm') });
const { TrackApi } = mod;
const raw = gunzipSync(await readFile(savePath));
const hlen = raw.readUInt32LE(4);
const header = JSON.parse(raw.subarray(8, 8 + hlen).toString('utf8'));
const bytes = new Uint8Array(raw.subarray(8 + hlen));
const api = new TrackApi(-73.985, 40.758, true);
api.clear_water();
for (const id of ['nyc', 'phl']) {
  const head = JSON.parse(await readFile(`trainworld/data/packs/${id}.json`, 'utf8'));
  const bin = await readFile(`trainworld/data/packs/${id}.bin`);
  const row = head.arrays.find(a => a.name === 'water_row'), xs = head.arrays.find(a => a.name === 'water_x');
  const rows = new Uint32Array(bin.buffer.slice(bin.byteOffset + row.offset, bin.byteOffset + row.offset + row.count * 4));
  const runs = new Uint16Array(bin.buffer.slice(bin.byteOffset + xs.offset, bin.byteOffset + xs.offset + xs.count * 2));
  const w = head.water, rad = Math.PI / 180, R = 6371008.8, o = { lon: -73.985, lat: 40.758 };
  if (api.add_pack_water) api.add_pack_water(w.cell_m, w.origin_m[0], w.origin_m[1], w.size[0], w.size[1], rows, runs, head.origin.lon, head.origin.lat);
  else api.append_water(w.cell_m, w.origin_m[0], w.origin_m[1], w.size[0], w.size[1], rows, runs,
    R * (head.origin.lon - o.lon) * rad * Math.cos(o.lat * rad), R * (head.origin.lat - o.lat) * rad, Math.cos(o.lat * rad) / Math.cos(head.origin.lat * rad), 1);
}
if (!api.load(bytes)) throw new Error('load failed');
const hash = a => createHash('sha256').update(Buffer.from(new Float64Array(a).buffer)).digest('hex').slice(0, 16);
const sum = (a, k, step) => { let s = 0; for (let i = k; i < a.length; i += step) s += a[i]; return s; };
function fingerprint() {
  const ei = api.edges_info(), li = api.lines_info(), ni = api.nodes_info();
  const render = [];
  for (let i = 0; i < ei.length; i += 10) render.push(...api.edge_render(ei[i]));
  return {
    saveSha: createHash('sha256').update(api.save()).digest('hex').slice(0, 16), saveBytes: api.save().length,
    edges: ei.length / 10, nodes: ni.length / 10, lines: li.length / 22,
    edgeLengthKm: sum(ei, 5, 10) / 1000, edgeCost: sum(ei, 6, 10),
    totalCost: api.total_cost(), builtCost: api.built_cost(), blueprintCost: api.blueprint_cost(), cash: api.cash(),
    runningPerHour: [0, 1, 2].map(l => api.running_cost(l)), fleetNeeded: api.cars_needed(),
    lineLengthKm: sum(li, 9, 22) / 1000,
    hashes: { edges: hash(ei), lines: hash(li), nodes: hash(ni), render: hash(render), phases: hash(api.phases()), meta: hash(api.profile_meta()) },
    renderPoints: render.length / 3,
  };
}
const out = { label, save: savePath, magicIn: Buffer.from(bytes.subarray(0, 4)).toString('latin1'), header: { cash: header.cash, clock: header.clock, fleet: header.fleet, fares: header.fares, ledgerDays: header.ledger?.length, city: header.city }, legacy: fingerprint() };
// A Philadelphia line in the shared Northeast frame (legacy: Philadelphia is 100 km west, 90 km south of New York).
const R = 6371008.8, rad = Math.PI / 180, px = R * (-75.1636 + 73.985) * rad * Math.cos(40.758 * rad), py = R * (39.9526 - 40.758) * rad;
const ok = (r, what) => { if (r !== 0) throw new Error(`${what} refused: ${api.issue_kinds()}`); return r; };
api.set_cash(header.cash); api.set_fleet(header.fleet);
const free = (x, y, level) => [0, 0, x, y, level];
ok(api.add_route(new Float64Array([...free(px, py - 2000, -2), ...free(px, py + 2000, -2)]), new Float64Array(0), 2, false), 'route');
const nodeAt = (x, y) => { const ni = api.nodes_info(); let best = 0, d = 1e18; for (let i = 0; i < ni.length; i += 10) { const dd = Math.hypot(ni[i + 1] - x, ni[i + 2] - y); if (dd < d) { d = dd; best = ni[i]; } } return best; };
const b = nodeAt(px, py + 2000);
ok(api.add_route(new Float64Array([1, b, 0, 0, 0, ...free(px + 30000, py + 20000, -2)]), new Float64Array([px + 8000, py + 9000, 0, -2]), 2, false), 'long route');
const a = nodeAt(px, py - 2000), c = nodeAt(px + 30000, py + 20000);
ok(api.add_station(1, a, 0, 0, 200, 'Philadelphia south'), 'station a');
ok(api.add_station(1, c, 0, 0, 200, 'Philadelphia northeast'), 'station c');
const line = api.new_line_id();
ok(api.set_line_with_trains(line, new Uint32Array([a, c]), 4, 4, 2, 30, 180, 0, 0x17a488, 'Philadelphia check'), 'line');
ok(api.construct_all(), 'construct all');
out.withPhiladelphia = fingerprint();
await writeFile(`artifacts/trainworld/notes-outputs/T-125-fingerprint-${label}.json`, JSON.stringify(out, null, 2) + '\n');
console.log(JSON.stringify(out));
