// The station tool's choice among overlapping tracks and its end snap (T-469, T-671).
// Run from maps: node trainworld/checks/t671-station-snap-check.mjs
import assert from 'node:assert/strict';
import { build } from '../app/node_modules/vite/dist/node/index.js';
import { resolve } from 'node:path';
const bundle = await build({ configFile: false, logLevel: 'silent', build: { write: false, minify: false, lib: { entry: resolve('trainworld/app/src/map/stationSnap.ts'), formats: ['es'] } } });
const output = (Array.isArray(bundle) ? bundle[0] : bundle).output;
const { orderOverlapping, firstAccepted, stationHere, onlyEndIssues, fractionAlong } = await import('data:text/javascript;base64,' + Buffer.from(output.find(o => o.type === 'chunk').code).toString('base64'));

// track A: 1 - 9 - 2 - 5 (junction), edges 10, 12, 11 of 100 m, 100 m, 1000 m; track B: 3 - 4, edge 20 of 1000 m
const E = (id, a, b, lengthM, built = true) => ({ id, a, b, lengthM, built });
const edges = [E(10, 1, 9, 100), E(12, 9, 2, 100), E(11, 2, 5, 1000), E(20, 3, 4, 1000), E(30, 5, 6, 500), E(31, 5, 7, 500), E(32, 5, 8, 500)];
const plain = { platform: 0, ports: 2 };
const nodes = new Map([[1, { platform: 0, ports: 1 }], [9, plain], [2, plain], [3, { platform: 0, ports: 1 }], [4, { platform: 0, ports: 1 }], [5, { platform: 0, ports: 3 }], [6, plain], [7, plain], [8, plain]]);
const none = new Set();
const choose = (...a) => orderOverlapping(...a)[0];
const at = (edge, frac) => ({ edge, frac });

assert.equal(stationHere(at(10, 0.5), 200, edges, nodes), false, 'no station anywhere');
assert.equal(choose([at(10, 0.5), at(20, 0.5)], 200, edges, nodes, none), 0, 'nothing differs: the nearest');

// a station on the joint at 2 (reach 100 m each way): near it on A the platform would meet it
nodes.set(2, { platform: 200, ports: 2 });
assert.equal(stationHere(at(11, 0.05), 200, edges, nodes), true, '50 m from it');
assert.equal(stationHere(at(11, 0.5), 200, edges, nodes), false, '500 m from it');
assert.equal(stationHere(at(10, 0.9), 200, edges, nodes), true, 'through the plain joint at 9: 110 m');
assert.equal(stationHere(at(10, 0.5), 200, edges, nodes), true, '150 m, inside the two reaches (100 + 100)');
assert.equal(stationHere(at(10, 0.5), 100, edges, nodes), false, '150 m is not inside 50 + 100: a shorter platform fits');
assert.equal(choose([at(11, 0.05), at(20, 0.05)], 200, edges, nodes, none), 1, 'the track the station would not meet, though further');
assert.equal(choose([at(20, 0.05), at(11, 0.05)], 200, edges, nodes, none), 0);

// a junction ends the walk
nodes.set(2, plain);
nodes.set(6, { platform: 200, ports: 1 });
assert.equal(stationHere(at(11, 0.99), 200, edges, nodes), false, 'a station past a junction is another track');

// a terminal's platform runs its full length from the node
nodes.set(6, plain);
nodes.set(4, { platform: 200, ports: 1 });
assert.equal(stationHere(at(20, 0.75), 200, edges, nodes), true, '250 m from a terminal station, inside 100 + 200');
assert.equal(stationHere(at(20, 0.5), 200, edges, nodes), false);

// both meet a station: blueprint before built, then the last station's track, then the nearest
nodes.set(2, { platform: 200, ports: 2 });
const a = at(11, 0.05), b = at(20, 0.95);
assert.equal(choose([a, b], 200, edges, nodes, none), 0, 'both meet one, both built: the nearest');
edges.find((e) => e.id === 20).built = false;
assert.equal(choose([a, b], 200, edges, nodes, none), 1, 'blueprint before built');
edges.find((e) => e.id === 20).built = true;
assert.equal(choose([a, b], 200, edges, nodes, new Set([20])), 1, 'the last station\'s track');
// free of stations beats blueprint with one
edges.find((e) => e.id === 11).built = false;
assert.equal(choose([a, at(20, 0.5)], 200, edges, nodes, none), 1, 'built and free beats blueprint and meeting a station');

assert.equal(onlyEndIssues(['PlatformTooLong']), true);
assert.equal(onlyEndIssues(['PlatformJoint']), true);
assert.equal(onlyEndIssues(['PlatformTooLong', 'GroundOverWater']), false, 'another refusal keeps refusing');
assert.equal(onlyEndIssues([]), false);

const pts = [0, 0, 0, 100, 0, 0, 100, 100, 0];
assert.equal(fractionAlong(pts, 10, 0), 0.05);
assert.equal(fractionAlong(pts, 100, 150), 1, 'past the end is the end');
assert.equal(fractionAlong([0, 0, 0, 100, 0, 0, 100, 10, 0, 5, 10, 0], 60, 0) < 0.5, true, 'along the track, not as the crow flies');
// the first spot the track model accepts wins, whatever ranks first (T-671, Anita: never suggest a
// station where one already stands while another spot under the pointer works)
const refuses = { node7: ['StationPorts'], edge1: ['PlatformTooLong', 'GroundOverWater'], edge2: ['StationInUse'], edge3: ['PlatformTooLong'], 'node9': null };
const ask = async (s) => refuses[s] === undefined ? null : refuses[s];
const never = () => false;
const none2 = () => null;
let r = await firstAccepted(['node7', 'edge1', 'edge4'], ask, none2, never);
assert.deepEqual(r, { spot: 'edge4', issues: [] }, 'a refused node and a refused track: the next that is taken');
r = await firstAccepted(['node7', 'edge2'], ask, none2, never);
assert.deepEqual(r, { spot: null, issues: ['StationPorts'] }, 'none taken: the first one\'s reason');
r = await firstAccepted(['edge3', 'edge2'], ask, (s) => (s === 'edge3' ? 'node9' : null), never);
assert.deepEqual(r, { spot: 'node9', issues: [] }, 'refused only for the track end: the end node');
r = await firstAccepted(['edge1'], ask, () => 'node9', never);
assert.equal(r.spot, null, 'another refusal beside the end one is not moved to the end');
r = await firstAccepted(['edge3', 'edge4'], ask, () => 'node7', never);
assert.equal(r.spot, 'edge4', 'an end node that is refused moves on to the next spot');
r = await firstAccepted(['node7', 'edge4'], ask, none2, () => true);
assert.deepEqual(r, { spot: null, issues: [] }, 'stops when the pointer has moved on');
console.log('t671-station-snap-check ok');
