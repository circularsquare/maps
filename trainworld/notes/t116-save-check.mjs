import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { build } from '../app/node_modules/vite/dist/node/index.js';

const bundle = await build({ configFile: false, logLevel: 'silent', build: {
  write: false, minify: false,
  lib: { entry: 'trainworld/app/src/workers/saveFile.ts', formats: ['es'] },
} });
const output = (Array.isArray(bundle) ? bundle[0] : bundle).output;
const api = await import('data:text/javascript;base64,' + Buffer.from(output.find(o => o.type === 'chunk').code).toString('base64'));
const track = new Uint8Array(await readFile('trainworld/T-007-real-nyc.save'));
const old = { game: 'anitabuilder', format: 1, city: 'nyc', clock: 145123,
  cash: 2534.5, fleet: 414, fares: { base: 1.5, perKm: .1 },
  ledger: [{ day: 1, fares: 2.75, running: .25, trains: 5, build: 125 }] };
const cases = [];
for (const [name, header] of [['legacy NYC', old], ['Northeast', { ...old, region: 'northeast', cities: ['nyc', 'phl'] }]]) {
  const bytes = await api.encodeSave(header, track);
  const decoded = await api.decodeSave(bytes);
  assert.deepEqual(decoded.header, header);
  assert.deepEqual(decoded.track, track);
  assert(api.isNortheastSave(decoded.header));
  cases.push({ name, bytes: bytes.length, trackBytes: decoded.track.length, cash: decoded.header.cash, clock: decoded.header.clock });
}
for (const foreign of [{ ...old, city: 'phl' }, { ...old, region: 'other' }, { ...old, cities: ['nyc', 'other'] }]) assert(!api.isNortheastSave(foreign));
await writeFile('artifacts/trainworld/notes-outputs/T-116-save-check.json', JSON.stringify({ cases, incompatibleRegionsRejected: true }, null, 2) + '\n');
console.log(JSON.stringify(cases));
