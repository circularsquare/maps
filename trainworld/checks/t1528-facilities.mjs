// T-1528: where a complex's car park and waiting room are placed (map/facilityPlace.ts), on made-up
// layouts: one lot and one room per complex, clear of the boxes, the track, each other and the name
// label where there is room; a lot still placed when nothing fits.
// Run from maps: node trainworld/checks/t1528-facilities.mjs
import assert from 'node:assert/strict';
import { loadApp } from './lib.mjs';
const { placeFacilities, segHitsRect, LOT_STRIDE, BAY_M, PLATFORM_POINTS, ST_ROOM, ST_PARK, ST_FLIP } = await loadApp(`
 export * from '@app/src/map/facilityPlace.ts';
 export { PLATFORM_POINTS, ST_ROOM, ST_PARK, ST_FLIP } from '@app/src/map/stationGeometry.ts';`);

const LAT = 40.7, perM = 1 / Math.cos(LAT * Math.PI / 180);
let pass = 0, fail = 0;
const check = (name, fn) => { try { fn(); pass++; console.log(`PASS ${name}`); } catch (e) { fail++; console.log(`FAIL ${name}: ${e.message}`); } };

/** A layout of straight stations along x (m): each { x, y, len, bits, complex }, with a track through
 * each 1 km either way, plus `extra` tracks (pairs of points, m). */
function layout(defs, extra = []) {
  const st = [], stations = new Float32Array(defs.length * 3), line = new Float32Array(defs.length * 2 * PLATFORM_POINTS), shape = new Float32Array(defs.length * 4);
  const edgePts = new Map(), waysAt = new Map(), complexOf = [];
  defs.forEach((d, i) => {
    const id = i + 1;
    st.push({ id, lat: LAT, platform: d.len, stationBuilt: true });
    stations.set([d.x * perM, d.y * perM, (d.bits & 4 ? ST_ROOM : 0) | (d.bits & 2 ? ST_PARK : 0)], i * 3);
    for (let k = 0; k < PLATFORM_POINTS; k++) line.set([(-d.len / 2 + d.len * k / (PLATFORM_POINTS - 1)) * perM, 0], (i * PLATFORM_POINTS + k) * 2);
    shape.set([0, 0, 0, 6 * perM], i * 4);
    edgePts.set(id * 10, new Float32Array([d.x * perM, d.y * perM, 0, (d.x + 1000) * perM, d.y * perM, 0]));
    edgePts.set(id * 10 + 1, new Float32Array([d.x * perM, d.y * perM, 0, (d.x - 1000) * perM, d.y * perM, 0]));
    waysAt.set(id, [id * 10, id * 10 + 1]);
    complexOf[id] = d.complex ?? id;
  });
  extra.forEach(([x0, y0, x1, y1], k) => edgePts.set(9000 + k, new Float32Array([x0 * perM, y0 * perM, 0, x1 * perM, y1 * perM, 0])));
  const lots = placeFacilities(st, stations, line, shape, edgePts, waysAt, Uint32Array.from(complexOf, (v) => v ?? 0xffffffff));
  const out = [];
  for (let o = 0; o < lots.length; o += LOT_STRIDE) {
    const hl = lots[o + 4], ux = lots[o + 2] / (hl * perM), uy = lots[o + 3] / (hl * perM);
    out.push({ r: { x: lots[o], y: lots[o + 1], ux, uy, hl, hd: lots[o + 5] }, bays: lots[o + 6], station: lots[o + 8], served: lots[o + 9] });
  }
  return { lots: out, flags: [...Array(defs.length)].map((_, i) => Math.round(stations[i * 3 + 2])), edgePts, defs };
}
/** whether a lot comes within `pad` m of any track or any platform */
const clashes = (L, r, pad) => {
  for (const p of L.edgePts.values()) if (segHitsRect(p[0], p[1], p[3], p[4], r, perM, pad)) return true;
  return false;
};

check('a lone station: one lot, clear of the track by 6 m, north of an east-west platform (its label goes south)', () => {
  const L = layout([{ x: 0, y: 0, len: 200, bits: 2 }]);
  assert.equal(L.lots.length, 1);
  const { r, bays } = L.lots[0];
  assert.equal(bays, 1);
  assert.equal(r.hd, BAY_M / 2);
  assert.ok(!clashes(L, r, 6), 'clear of the track');
  assert.ok(r.y < 0, `north of the station (local y runs south): y ${r.y}`);
  assert.ok(Math.abs(r.y / perM) - r.hd >= 20 - 1e-6, 'past a zoom 14 box');
  assert.ok(Math.abs(r.hl * 2 - 200) <= 2.5, `as long as the platform: ${r.hl * 2}`);
});

check('waiting room and car park: the room on one side, the lot clear of it', () => {
  const L = layout([{ x: 0, y: 0, len: 200, bits: 6 }]);
  assert.equal(L.lots.length, 1);
  assert.ok(L.flags[0] & ST_ROOM, 'the room is drawn');
  // + is the left of the platform's way (+x), so local -y; ST_FLIP puts the room there
  const roomY = L.flags[0] & ST_FLIP ? -1 : 1;
  const { r } = L.lots[0];
  const sameSide = Math.sign(r.y) === roomY;
  // the room reaches about 36 m out and 30 m along from the middle
  assert.ok(!sameSide || Math.abs(r.y / perM) - r.hd > 36 || Math.abs(r.x / perM) - r.hl > 30, `lot ${JSON.stringify(r)} on the room's side`);
});

check('a complex of two: one lot of two bays for both, one waiting room', () => {
  const L = layout([{ x: 0, y: 0, len: 200, bits: 6, complex: 1 }, { x: 0, y: 40, len: 160, bits: 6, complex: 1 }]);
  assert.equal(L.lots.length, 1, 'drawn once');
  assert.equal(L.lots[0].bays, 2);
  assert.equal(L.lots[0].served, 2);
  assert.equal(L.flags.filter((f) => f & ST_ROOM).length, 1, 'one room');
  assert.ok(!clashes(L, L.lots[0].r, 6), 'clear of both tracks');
});

check('two complexes side by side: two lots that do not meet', () => {
  const L = layout([{ x: 0, y: 0, len: 200, bits: 2 }, { x: 0, y: 150, len: 200, bits: 2 }]);
  assert.equal(L.lots.length, 2);
  const [a, b] = L.lots.map((l) => l.r);
  const gap = Math.abs(a.y - b.y) / perM - a.hd - b.hd;
  assert.ok(gap >= 4 || Math.abs(a.x - b.x) / perM - a.hl - b.hl >= 4, `lots ${gap.toFixed(1)} m apart`);
});

check('the station only on one side is open: the lot goes there', () => {
  // tracks every 12 m to the north of the station, from 10 m to 120 m out
  const extra = [];
  for (let y = -10; y >= -120; y -= 12) extra.push([-600, y, 600, y]);
  const L = layout([{ x: 0, y: 0, len: 200, bits: 2 }], extra);
  assert.equal(L.lots.length, 1);
  assert.ok(L.lots[0].r.y > 0, 'south, past the label rather than on the track: ' + JSON.stringify(L.lots[0].r));
  assert.ok(!clashes(L, L.lots[0].r, 6));
});

check('nowhere fits: a lot is still drawn', () => {
  const extra = [];
  for (let y = -160; y <= 160; y += 10) if (y) extra.push([-600, y, 600, y]);
  const L = layout([{ x: 0, y: 0, len: 200, bits: 2 }], extra);
  assert.equal(L.lots.length, 1);
});

check('no facilities: no lots, flags untouched', () => {
  const L = layout([{ x: 0, y: 0, len: 200, bits: 0 }]);
  assert.equal(L.lots.length, 0);
  assert.equal(L.flags[0], 0);
});

console.log(`${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
