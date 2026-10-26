// Run from maps: node trainworld/checks/t1419-carry.mjs
// T-1419: a line's riders carried onto an edited stop list (game/carryLoads.ts) and read through
// lineLoadsFor while demand has no figures for the new stops.
import assert from 'node:assert/strict';
import { loadApp } from './lib.mjs';
globalThis.location ??= { search: '' };
globalThis.document ??= { documentElement: { dataset: {}, style: { setProperty() {}, removeProperty() {} } }, addEventListener() {} };
const { carryLoads, lineLoadsFor } = await loadApp(`
 export { carryLoads } from '@app/src/game/carryLoads.ts';
 export { lineLoadsFor } from '@app/src/game/demand.ts';`);

// A B C D: forward 10 20 30, back (run 1 in its own order: D->C, C->B, B->A) 3 2 1
const old = [Float32Array.from([10, 20, 30, 3, 2, 1])];
const fwd = (p, n) => Array.from(p.subarray(0, n - 1));
const back = (p, n) => Array.from(p.subarray(n - 1));

// a stop removed: the busier side
let c = carryLoads(old, ['A', 'B', 'C', 'D'], ['A', 'C', 'D']);
assert.deepEqual(fwd(c[0], 3), [20, 30]);
assert.deepEqual(back(c[0], 3), [3, 2]); // D->C 3, C->A max(C->B 2, B->A 1)
// a stop added between two: both halves the old whole
c = carryLoads(old, ['A', 'B', 'C', 'D'], ['A', 'B', 'X', 'C', 'D']);
assert.deepEqual(fwd(c[0], 5), [10, 20, 20, 30]);
assert.deepEqual(back(c[0], 5), [3, 2, 2, 1]);
// a new end stop: its stretch has no figures
c = carryLoads(old, ['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D', 'E']);
assert.deepEqual(fwd(c[0], 5), [10, 20, 30, 0]);
assert.deepEqual(back(c[0], 5), [0, 3, 2, 1]);
c = carryLoads(old, ['A', 'B', 'C', 'D'], ['X', 'A', 'B', 'C', 'D']);
assert.deepEqual(fwd(c[0], 5), [0, 10, 20, 30]);
assert.deepEqual(back(c[0], 5), [3, 2, 1, 0]);
// a stop swapped for a station of the same complex
c = carryLoads(old, ['A', 'B', 'C', 'D'], ['A', 'B2', 'C', 'D'], (s) => s.replace('2', ''));
assert.deepEqual(fwd(c[0], 4), [10, 20, 30]);
// rebuilt beyond recognition, or one stop shared: nothing
assert.equal(carryLoads(old, ['A', 'B', 'C', 'D'], ['A', 'X', 'Y', 'Z']), null);
assert.equal(carryLoads(old, ['A', 'B', 'C', 'D'], ['P', 'Q', 'R', 'D']), null);
// a loop's first stop comes again at the end
c = carryLoads([Float32Array.from([1, 2, 3, 6, 5, 4])], ['A', 'B', 'C', 'A'], ['A', 'B', 'C', 'A']);
assert.deepEqual(Array.from(c[0]), [1, 2, 3, 6, 5, 4]);
// every period carried
assert.equal(carryLoads([old[0], old[0]], ['A', 'B', 'C', 'D'], ['A', 'C', 'D']).length, 2);

// lineLoadsFor: exact figures win, carried ones only for stops the solve was not for
const dv = { lines: new Map([['1', { loads: old, stops: ['A', 'B', 'C', 'D'] }]]), purposes: new Map() };
assert.equal(lineLoadsFor(dv, '1', 'all', ['A', 'B', 'C', 'D']), old);
assert.deepEqual(fwd(lineLoadsFor(dv, '1', 'all', ['A', 'C', 'D'])[0], 3), [20, 30]);
assert.equal(lineLoadsFor(dv, '1', 'all', ['A', 'C', 'D']), lineLoadsFor(dv, '1', 'all', ['A', 'C', 'D']), 'cached');
assert.equal(lineLoadsFor(dv, '1', 'all', ['P', 'Q', 'R']), undefined);
assert.equal(lineLoadsFor(dv, '1', 'all', ['P', 'Q', 'R', 'S']), old, 'same count: as stored');
assert.equal(lineLoadsFor(dv, '1', 'all'), old, 'no stops asked: as stored');
assert.equal(lineLoadsFor({ lines: new Map([['1', { loads: old }]]), purposes: new Map() }, '1', 'all', ['A', 'C', 'D']), old, 'no stops stored: as before');
// a purpose with no riders: zeros of the carried length
assert.deepEqual(Array.from(lineLoadsFor(dv, '1', 'work', ['A', 'C', 'D'])[0]), [0, 0, 0, 0]);
console.log('PASS  t1419 carried line riders');
