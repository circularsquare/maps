// Picking follows visible instance/layer stacking, including smaller bubbles drawn over large
// ones. Run from maps: node trainworld/notes/t133-picking-check.mjs
import assert from 'node:assert/strict';
import { build } from '../app/node_modules/vite/dist/node/index.js';
import { resolve } from 'node:path';
const root = resolve('trainworld/app');
const bundle = await build({ configFile: false, logLevel: 'silent', plugins: [{
  name: 'picking-check',
  resolveId(id) { if (id.endsWith('picking-check')) return '\0picking-check'; },
  load(id) { if (id === '\0picking-check') return `
    export { NetworkRenderer } from ${JSON.stringify(root + '/src/map/renderer.ts')};
    export { pick } from ${JSON.stringify(root + '/src/map/pick.ts')};
    export { discAt } from ${JSON.stringify(root + '/src/map/demandHover.ts')};
    export { startClock } from ${JSON.stringify(root + '/src/game/clock.ts')};
    export { CapacityMarkers } from ${JSON.stringify(root + '/src/map/capacityMarkers.ts')};
    export { selection, tool } from ${JSON.stringify(root + '/src/game/ui.ts')};
    export { addFinder, startHover } from ${JSON.stringify(root + '/src/map/demandHover.ts')};
  `; },
}], build: { write: false, minify: false, lib: { entry: 'picking-check', formats: ['es'] } } });
const output = (Array.isArray(bundle) ? bundle[0] : bundle).output;
class Element {
  style = {}; children = []; listeners = {}; offsetWidth = 40; offsetHeight = 20;
  classList = { add() {} };
  appendChild(el) { this.children.push(el); }
  replaceChildren(...els) { this.children = els; }
  getBoundingClientRect() { return { left: 20, top: 30, bottom: 50 }; }
  addEventListener(type, f) { (this.listeners[type] ??= []).push(f); }
}
globalThis.location = { search: '' };
globalThis.document = { visibilityState: 'visible', documentElement: { dataset: {} }, addEventListener() {}, createElement: () => new Element(), fonts: { ready: Promise.resolve() } };
const { NetworkRenderer, pick, discAt, startClock, CapacityMarkers, selection, tool, addFinder, startHover } = await import('data:text/javascript;base64,' + Buffer.from(output.find(o => o.type === 'chunk').code).toString('base64'));
startClock(0).setRunning(false);
const F = a => new Float32Array(a);
const strokes = (ys, colours, edges = colours) => ({ count: ys.length, seg: F(ys.flatMap(y => [-30, y, 30, y])), colour: new Uint32Array(colours), edge: new Uint32Array(edges) });
const n = {
  stations: F([0, 0, 0, 3, 0, 0]), stationIds: ['under', 'over'], stationCount: 2,
  lines: strokes([0, 3], [1, 2]), track: strokes([0, 3], [10, 20]),
  junctionIds: [], junctions: F([]),
  meta: F(Array.from({ length: 7 }, (_, i) => i === 0 || i === 6 ? [0, 1, 100, 100] : [0, 0, 0, 0]).flat()),
  phases: F([0, 0, 0, 0]), lineTable: F([0, 2, 100, 100, 2, 2, 100, 100]),
  samples: F([0, 0, 100, 0, 3, 0, 103, 0]),
};
const ov = { camMatrix: [], toScreen: (x, y) => [x, y], renderer: Object.assign(new NetworkRenderer(), {
  net: n, slotPx: 6, trips: F([0, 0, 6, 0]), tripEpoch: 0, tripCount: 2, detailedTrains: false,
}) };
assert.equal(pick(ov, 0, 0, true).line, '1', 'last train wins even if another is closer');
assert.equal(pick(ov, 0, 0, false).station, 'over', 'last station wins overlapping hits');
n.stationCount = 0;
assert.equal(pick(ov, 0, 0, false).line, '2', 'last line stroke wins');
assert.equal(pick(ov, 0, 0, false, { kind: 'line', line: '2' }).edge, 20, 'selected line still cycles to track');
assert.equal(pick(ov, 0, 0, false, { kind: 'line', line: '1' }).edge, 20, 'selected highlight pass beats later ordinary line strokes');
ov.renderer.slotPx = 0;
assert.equal(pick(ov, 0, 0, false).edge, 20, 'hidden line strokes cannot win in height/speed mode');
assert.equal(pick(ov, 500, 500, true), null, 'empty map remains empty');
const discs = F([0, 0, 10, 0, 0, 0, 1, 0, 3, 0, 4, 0, 0, 0, 1, 0]);
assert.equal(discAt(discs, 0, 0), 1, 'top smaller disc beats closer underlying centre');
assert.equal(discAt(discs, -8, 0), 0, 'exposed part of underlying disc remains selectable');
assert.equal(discAt(discs, 13, 0, 3), 1, 'hit radius follows size control');
assert.equal(discAt(discs, 50, 50), -1);
const container = new Element();
const map = { getContainer: () => container, getZoom: () => 13, on() {}, unproject: () => ({ lng: -73.98, lat: 40.73 }) };
const markers = new CapacityMarkers(container, { ...ov, afterMapFrame: [], canvas: { clientWidth: 1000, clientHeight: 1000 } }, map);
markers.set({ capacity: [[{ x: 100, y: 100, rho: 0.95, delayS: 60, kind: 4 }]], nodes: new Map([[42, { id: 42, ports: 3, lng: -73.98, lat: 40.73 }]]) }, 0);
const mark = container.children[0].children[0];
const event = { clientX: 145, clientY: 140, shiftKey: true, buttons: 0, target: { closest: () => mark }, stopPropagation() { this.stopped = true; } };
let covered = true;
addFinder(100, () => covered ? ['Top bubble'] : null);
const topSelection = { kind: 'commuters', end: 'home', cells: [3], areas: 1 };
markers.demandClick = (x, y, add) => { assert.deepEqual([x, y, add], [125, 110, true]); return covered ? topSelection : undefined; };
mark.onclick(event);
assert.equal(selection.peek(), topSelection, 'bubble above DOM capacity tag wins click');
assert.equal(event.stopped, true);
mark.onmousemove(event);
assert.equal(container.children[1].style.display, 'none', 'covered capacity tooltip is suppressed');
startHover(map);
container.listeners.mousemove[0](event);
assert.equal(container.children[2].style.display, '', 'demand hover still works over a DOM capacity target');
covered = false;
mark.onclick(event);
assert.deepEqual(selection.peek(), { kind: 'junction', node: 42 }, 'exposed capacity tag retains junction link');
mark.onmousemove(event);
assert.equal(container.children[1].style.display, '', 'exposed capacity tooltip remains available');
tool.value = 'station'; selection.value = null;
mark.onclick(event);
assert.equal(selection.peek(), null, 'construction tools are not overridden by a capacity junction link');
console.log('T-133 picking checks passed: network/disc stacking, hidden layers, selected highlights, capacity clicks/hover, and tool priority.');
