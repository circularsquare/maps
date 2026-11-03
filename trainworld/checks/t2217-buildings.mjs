// T-2217: station buildings in the track model (WASM, as the clock worker runs it) and the app's
// members, on the real New York save: a building round Times Sq-42 St and 42 St-Bryant Pk is paid
// by its floor, the track model's members and game/buildings.ts's agree, it joins the two complexes
// into one, it rides in the save, and a save it has been removed from is byte for byte the one
// before. Run from the maps root:
//   node trainworld/checks/t2217-buildings.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp } from './lib.mjs';

const app = await loadApp(`export * from '@app/src/game/complexes.ts'; export * from '@app/src/game/buildings.ts';
export { ComplexBuilder } from '@app/src/game/complexBuilder.ts'; export { PlatformLines } from '@app/src/workers/platformLines.ts';`);
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(new URL('../T-007-real-nyc.save', import.meta.url)));
const n = raw.readUInt32LE(4);
const api = new TrackApi(-73.985, 40.758, true);
assert.ok(api.load(raw.subarray(8 + n)), 'save did not load');
api.set_cash(1e6);

const R = 6378137;
const kAt = api.frame_kind() === 2 ? (y) => Math.cosh(y / R) : () => 1;
// every station's platform line as the clock worker works it out (workers/platformLines.ts)
const lines = new app.PlatformLines();
const ei = api.edges_info(), edges = [];
for (let i = 0; i < ei.length; i += 10) edges.push({ id: ei[i], a: ei[i + 1], b: ei[i + 2] });
lines.sync(edges);
const ni = api.nodes_info(), all = [];
for (let i = 0; i < ni.length; i += 10) {
  if (!(ni[i + 7] > 0)) continue;
  const line = lines.of(api, ni[i], ni[i + 1], ni[i + 2], ni[i + 7], kAt(ni[i + 2]));
  all.push({ id: String(ni[i]), name: api.node_name(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7], built: ni[i + 8] === 1, ...(line ? { line } : {}) });
}
const geo = all.filter((s) => s.built);
const fl = api.station_links(), links = [];
for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
const before = app.buildComplexes(geo, links, kAt);

const pick = geo.filter((s) => /^Times Sq|Bryant P/.test(s.name));
const names = [...new Set(pick.map((s) => s.name))];
assert.ok(names.some((s) => s.startsWith('Times Sq')) && names.some((s) => /Bryant/.test(s)), names.join(', '));
const cBefore = new Set(pick.map((s) => before.of.get(s.id)));
assert.ok(cBefore.size >= 2, `the stations start in ${cBefore.size} complex`);

// the rectangle round their nodes, 30 m out
const k = kAt(pick[0].y), pad = 30 * k;
const xs = pick.map((s) => s.x), ys = pick.map((s) => s.y);
const [x0, x1, y0, y1] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
const corners = Float64Array.from([x0, y0, x1, y0, x1, y1, x0, y1]);
const quote = api.building_quote(0xffffffff, corners);
assert.equal(api.issue_kinds(), '', 'refused: ' + api.issue_kinds());
const saved = api.save();
const cash = api.cash();
assert.equal(api.add_building(corners, ''), 0, api.issue_kinds());
const id = api.last_building(), floor = api.building_area(id);
assert.ok(Math.abs(floor - quote[0]) < 1e-6 && Math.abs(cash - api.cash() - 0.002 * floor) < 1e-6 && Math.abs(api.building_charge() - 0.002 * floor) < 1e-6, 'US$2,000 a m²');
const name = api.building_name(id);

// the track model's members and the app's are the same stations
const buildings = app.buildingsFrom(api.buildings_info(), (b) => api.building_name(b));
assert.equal(buildings.length, 1);
const simMembers = [...api.building_members(id)].map(String).sort();
const appMembers = app.buildingMembers(buildings, all, kAt)[0].slice().sort();
assert.deepEqual(appMembers, simMembers, 'members');
assert.equal(quote[2], simMembers.length);
for (const s of pick) assert.ok(simMembers.includes(s.id), `${s.name} is in`);

// one complex now, the same from the patching builder
const joined = [...links, ...app.buildingLinks(buildings, geo, kAt)];
const after = app.buildComplexes(geo, joined, kAt);
assert.equal(new Set(pick.map((s) => after.of.get(s.id))).size, 1, 'one complex');
assert.ok(after.members.length <= before.members.length - (cBefore.size - 1));
const cb = new app.ComplexBuilder();
cb.update(geo, links, kAt, 'merc');
const patched = cb.update(geo, app.withBuildingLinks(links, buildings, geo, kAt), kAt, 'merc');
assert.deepEqual(patched.members.map((m) => m.join()), after.members.map((m) => m.join()), 'patched as built whole');

// the save carries it; removing it leaves the save as it was
const withHall = api.save();
const b = new TrackApi(-73.985, 40.758, true);
assert.ok(b.load(withHall));
assert.deepEqual([...b.buildings_info()], [...api.buildings_info()]);
assert.equal(b.building_name(0), name);
assert.deepEqual(Buffer.compare(Buffer.from(b.save()), Buffer.from(withHall)), 0, 'loads and saves the same');
assert.equal(TrackApi.save_unread(withHall), 0);
const cashNow = api.cash();
assert.equal(api.remove_building(id), 0);
assert.equal(api.cash(), cashNow, 'nothing refunded');
assert.equal(Buffer.compare(Buffer.from(api.save()), Buffer.from(saved)), 0, 'without it: the save from before, byte for byte');

console.log(`t2217 ok: "${name}" ${Math.round(floor)} m², US$${(0.002 * floor).toFixed(1)}M, ${simMembers.length} stations (${names.join(', ')}), ${cBefore.size} complexes made one`);
