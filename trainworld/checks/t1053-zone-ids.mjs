// Run from maps: node trainworld/checks/t1053-zone-ids.mjs
// Saves naming removed zones (T-1053): every zone id HEAD's registry has and the working one lacks
// has an entry in game/zone-successors.json (else: python trainworld/pipeline/zone_successors.py),
// every successor is registered or itself in the table, and saveFile's `zoneNow` and
// `currentZones` map a header's zone ids as docs/architecture.md 11 says.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { build } from '../app/node_modules/vite/dist/node/index.js';

const root = resolve('trainworld/app');
const REG = 'trainworld/app/src/game/registry.json';
const ids = (reg) => new Set(reg.cities.map((c) => c.id));
const have = ids(JSON.parse(readFileSync(REG, 'utf8')));
const table = JSON.parse(readFileSync('trainworld/app/src/game/zone-successors.json', 'utf8'));
const head = ids(JSON.parse(execFileSync('git', ['show', `HEAD:${REG}`], { encoding: 'utf8', maxBuffer: 1 << 28 })));
const gone = [...head].filter((z) => !have.has(z) && !(z in table));
assert.deepEqual(gone, [], `zones removed since HEAD with no successor: ${gone.join(' ')}; run python trainworld/pipeline/zone_successors.py`);
for (const [z, to] of Object.entries(table)) assert.ok(to === null || have.has(to) || to in table, `${z}'s successor ${to} is neither registered nor in the table`);

const bundle = await build({ configFile: false, logLevel: 'silent', plugins: [{
  name: 't1053', resolveId(id) { if (id.endsWith('t1053-entry')) return '\0t1053-entry'; },
  load(id) { if (id === '\0t1053-entry') return `export { zoneNow, currentZones, isNortheastSave } from ${JSON.stringify(root + '/src/workers/saveFile.ts')};`; },
}], build: { write: false, minify: false, lib: { entry: 't1053-entry', formats: ['es'] } } });
const code = (Array.isArray(bundle) ? bundle[0] : bundle).output.find((o) => o.type === 'chunk').code;
const { zoneNow, currentZones, isNortheastSave } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));

// zoneNow: registered ids as they are, removed ones through the table (chains too), else null
const reg = new Set(['nyc', 'hfd', 'nhv']), at = (z) => reg.has(z);
const t = { wnd: 'hfd', nlc: 'hfd', lit: null, old: 'wnd', a: 'b', b: 'a' };
assert.equal(zoneNow('nyc', at, t), 'nyc');
assert.equal(zoneNow('wnd', at, t), 'hfd');
assert.equal(zoneNow('old', at, t), 'hfd', 'followed through a later removal');
assert.equal(zoneNow('lit', at, t), null, 'removed with no successor');
assert.equal(zoneNow('made-up', at, t), null);
assert.equal(zoneNow('a', at, t), null, 'a loop ends');
// the real table: every registered id maps to itself
for (const z of have) assert.equal(zoneNow(z), z);

// currentZones on a header naming removed ids
const now = (z) => zoneNow(z, at, t);
const day = (d, fares) => ({ day: d, fares, running: 1, trains: 0, build: 0, rides: 10 });
const hour = (h, fares) => ({ hour: h, fares, running: 0, trains: 0, build: 0 });
const h = {
  game: 'anitabuilder', format: 1, city: 'nyc', region: 'northeast', clock: 400000, cash: 1, fleet: 0, fares: { base: 1.5, perKm: 0.1 }, ledger: [],
  cities: ['nyc', 'wnd', 'nlc', 'lit', 'hfd', 'made-up'], start: 'wnd', opened: ['nyc', 'wnd', 'made-up', 'nlc', 'hfd'],
  regions: { hfd: { ledger: [day(2, 5)], hours: [hour(50, 1)] }, wnd: { ledger: [day(2, 3), day(3, 4)], hours: [hour(50, 2)] }, 'made-up': { ledger: [day(2, 7)], hours: [] }, '': { ledger: [day(2, 1)], hours: [] } },
  stars: { wnd: 300, nlc: 200, hfd: 250, lit: 100 },
  notes: { 4: [3, 'wnd'], 5: [7, 'nyc'], 6: [2, 'made-up'] },
};
const c = currentZones(h, now);
assert.deepEqual(c.cities, ['nyc', 'hfd']);
assert.equal(c.start, 'hfd');
assert.deepEqual(c.opened, ['nyc', 'hfd'], 'mapped, in order, once each, unknown dropped');
assert.deepEqual(Object.keys(c.regions).sort(), ['', 'hfd']);
assert.deepEqual(c.regions.hfd.ledger.map((d) => [d.day, d.fares]), [[2, 5], [2, 3], [3, 4]], 'removed zone ledgers added to the successor (loadRegions sums a day listed twice)');
assert.deepEqual(c.regions[''].ledger.map((d) => [d.day, d.fares]).sort((a, b) => a[1] - b[1]), [[2, 1], [2, 7]],'no successor: outside every zone');
assert.deepEqual(c.stars, { hfd: 200 }, 'the earliest star; none for a zone dropped');
assert.deepEqual(c.notes, { 4: [3, 'hfd'], 5: [7, 'nyc'], 6: [2, 'made-up'] });
assert.equal(h.cities.length, 6, 'the header passed in is not changed');
assert.equal(currentZones({ ...h, start: 'lit' }, now).start, undefined);
// a header naming only registered zones comes back equal
const plain = { ...h, cities: ['nyc', 'hfd'], start: 'nyc', opened: ['nyc', 'hfd'], regions: { hfd: h.regions.hfd, '': h.regions[''] }, stars: { hfd: 1 }, notes: { 5: [7, 'nyc'] } };
assert.deepEqual(currentZones(plain, now), plain);
// legacy NYC saves (no world, no cities) and this world's open; another world's do not
const world = JSON.parse(readFileSync(REG, 'utf8')).world;
assert.ok(isNortheastSave({ ...h, region: undefined, cities: undefined }));
assert.ok(isNortheastSave({ ...h, region: world }));
assert.ok(!isNortheastSave({ ...h, region: world + '-other' }));
console.log(`t1053 zone ids: PASS (${Object.keys(table).length} successors, ${head.size - [...head].filter((z) => have.has(z)).length} ids removed since HEAD)`);
