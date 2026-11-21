// T-2624: a local result shown period by period (`localStep`, app/src/workers/demandSteps.ts). With every period
// in, a step is the result's own lines; with some, the lines the edit changed take the solved periods' change in
// proportion at their level and every other line keeps its shown figures for the unsolved periods; steps start
// from the views shown before the first, so they never compound; the view's version, final flag, zones and
// purposes stay as shown (a step is never shown as final).
//   node trainworld/checks/t2624-local-steps.mjs
import assert from 'node:assert/strict';
import { localStep } from '../app/src/workers/demandSteps.ts';

const P = 5, levels = ['high', 'high', 'high', 'medium', 'low'];
const stops = ['a', 'b', 'c'], n = stops.length;
// a line's figures: per period boardings and alightings at its 2n route nodes, loads on its 2(n-1) stretches
function line(scale, opts = {}) {
  const board = [], alight = [], loads = [];
  for (let q = 0; q < P; q++) {
    const f = scale[q];
    board.push(Float32Array.from({ length: 2 * n }, (_, k) => f * (k + 1)));
    alight.push(Float32Array.from({ length: 2 * n }, (_, k) => f * (2 * n - k)));
    loads.push(Float32Array.from({ length: 2 * (n - 1) }, (_, k) => f * 10 * (k + 1)));
  }
  const boardings = stops.map(() => 0);
  for (const b of board) for (let k = 0; k < n; k++) boardings[k] += b[k] + b[2 * n - 1 - k];
  return { ridersPerDay: boardings.reduce((x, y) => x + y, 0), fullestPct: 50, boardings, loads, board, alight, arrive: board, fareX: scale.map((f) => f), baseX: scale.map(() => 0), stops, tph: [10, 10, 10, 6, 4], cars: 8, ...opts };
}
function stationsOf(lines) {
  const m = new Map();
  for (const l of lines) for (let q = 0; q < P; q++) for (let k = 0; k < n; k++) {
    const s = m.get(stops[k]) ?? { boardings: 0, alightings: 0 };
    s.boardings += l.board[q][k] + l.board[q][2 * n - 1 - k];
    s.alightings += l.alight[q][k] + l.alight[q][2 * n - 1 - k];
    m.set(stops[k], s);
  }
  return m;
}
function day(lines, version, final) {
  const map = new Map(Object.entries(lines));
  const view = { version, day: 0, round: 3, final, split: { train: 1, walk: 1, drive: 98, none: 0 }, trainShareByPeriod: [], peakHourFactor: [], tripsPerDay: 1000, railTripsPerDay: 100, lines: map, stations: stationsOf(Object.values(lines)), purposes: new Map([['work', { lines: new Map(), stations: new Map() }]]) };
  return { view, cities: new Map([['nyc', { ...view, tripsPerDay: 500, railTripsPerDay: 50 }]]), intercity: null };
}
const was = { L: line([100, 200, 150, 50, 20]), M: line([80, 160, 120, 40, 10]) };
// the edit adds trains to L (its tph changes): every period of L up 10%, M down a little and noisily
const now = { L: line([110, 220, 165, 55, 22], { tph: [11, 11, 11, 7, 5] }), M: line([82, 150, 118, 41, 9]) };
const shown = day(was, 3, true), est = day(now, 4, false);
const layout = new Map([['L', { stops }], ['M', { stops }]]);
const keep = () => true;
const step = (solved, from = shown) => localStep(from, shown, est, solved, levels, ['L', 'M'], layout, layout, keep);
const close = (a, b, msg) => assert.ok(Math.abs(a - b) <= 1e-3 * Math.max(1, Math.abs(b)), `${msg}: ${a} against ${b}`);

// every period in: the result's own lines and stations
const all = step([true, true, true, true, true]);
for (const id of ['L', 'M']) {
  close(all.view.lines.get(id).ridersPerDay, now[id].ridersPerDay, `${id} whole`);
  for (let q = 0; q < P; q++) assert.deepEqual(Array.from(all.view.lines.get(id).board[q]), Array.from(now[id].board[q]));
}
for (const [s, v] of est.view.stations) close(all.view.stations.get(s).boardings, v.boardings, `station ${s} whole`);

// the morning peak alone: L's high periods move by the morning's 10%, its medium and low stay; M keeps every
// unsolved period as shown
const am = step([true, false, false, false, false]);
const L = am.view.lines.get('L'), M = am.view.lines.get('M');
const sum = (v) => v.reduce((x, y) => x + y, 0);
close(sum(L.board[1]), 1.1 * sum(was.L.board[1]), 'L midday scaled');
close(sum(L.board[3]), sum(was.L.board[3]), 'L evening as shown');
close(sum(M.board[0]), sum(now.M.board[0]), 'M morning solved');
close(sum(M.board[1]), sum(was.M.board[1]), 'M midday as shown');
close(L.ridersPerDay, sum(L.boardings), 'L riders are its boardings');
const dL = L.ridersPerDay - was.L.ridersPerDay, fL = now.L.ridersPerDay - was.L.ridersPerDay;
assert.ok(dL > 0.5 * fL && dL <= fL, `L's first step shows ${dL} of ${fL}`);
// stations move by the lines' change
for (const s of stops) {
  const d = (l) => sum([0, 1, 2, 3, 4].map((q) => l.board[q][stops.indexOf(s)] + l.board[q][2 * n - 1 - stops.indexOf(s)]));
  close(am.view.stations.get(s).boardings, shown.view.stations.get(s).boardings + d(L) + d(M) - d(was.L) - d(was.M), `station ${s} step`);
}
// never final: the shown view's version and flag, zones and purposes as they were
assert.equal(am.view.version, 3);
assert.equal(am.view.final, true);
assert.equal(am.view.purposes, shown.view.purposes);
assert.equal(am.cities.get('nyc').railTripsPerDay, 50);
assert.equal(am.cities.get('nyc').lines, am.view.lines);

// steps start from the views before the first: a second step on top of the first equals it made at once
const two = step([true, true, false, false, false]), onTop = step([true, true, false, false, false], am);
for (const id of ['L', 'M']) close(onTop.view.lines.get(id).ridersPerDay, two.view.lines.get(id).ridersPerDay, `${id} second step`);
for (const s of stops) close(onTop.view.stations.get(s).boardings, two.view.stations.get(s).boardings, `station ${s} second step`);

// new stops (a station added on L): the unsolved periods on the new stops, sized to the shown riders times the change
const stops2 = ['a', 'x', 'b', 'c'];
const n2 = stops2.length;
const L2 = { ...now.L, stops: stops2, board: now.L.board.map((b) => Float32Array.from({ length: 2 * n2 }, (_, k) => b[k % (2 * n)])), alight: now.L.alight.map((b) => Float32Array.from({ length: 2 * n2 }, (_, k) => b[k % (2 * n)])), loads: now.L.loads.map(() => new Float32Array(2 * (n2 - 1)).fill(5)) };
const est2 = day({ L: L2, M: now.M }, 4, false);
est2.view.stations.set('x', { boardings: 1, alightings: 1 });
const lay2 = new Map([['L', { stops: stops2 }], ['M', { stops }]]);
const st2 = localStep(shown, shown, est2, [true, false, false, false, false], levels, ['L', 'M'], lay2, layout, keep);
const l2 = st2.view.lines.get('L');
assert.equal(l2.boardings.length, n2);
close(sum(l2.board[1]), sum(was.L.board[1]) * sum(L2.board[0]) / sum(was.L.board[0]), 'new stops midday');
close(sum(l2.board[4]), sum(was.L.board[4]), 'new stops night as shown');
assert.ok(st2.view.stations.get('x').boardings > 0);

// a line the fares rule holds keeps its shown figures
const held = localStep(shown, shown, est, [true, false, false, false, false], levels, ['L', 'M'], layout, layout, (id) => id !== 'L');
assert.equal(held.view.lines.get('L'), was.L);
console.log('t2624 local steps: ok');
