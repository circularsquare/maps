// T-2935: the line panel's "N trains held out of service" is the trains the sim keeps out of service.
//   node trainworld/checks/t2935-held-count.mjs
// In Node on the track model (WASM), on the real save: lines given more trains than their track takes, by hand and on
// shared track, one level over the limit by only part of a train. Per line and level, the panel's count (`heldTrains`
// of `line_held`) against the trains bought minus the most drawn at any moment of a day. Exits 1 on any difference.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';
import { loadApp, REAL_SAVE } from './lib.mjs';

const { heldTrains } = await loadApp('src/game/capacityKinds.ts');
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
const raw = gunzipSync(readFileSync(REAL_SAVE));
const n = raw.readUInt32LE(4);
const fresh = () => { const a = new TrackApi(-73.985, 40.758, true); if (!a.load(raw.subarray(8 + n))) throw new Error('save did not load'); a.set_cash(1e12); return a; };

const [E, F, FIVE] = [15, 16, 4];
const DAY = 86400 * 10, STEP = 127;
const bought = (a, id) => { const li = a.lines_info(); for (let i = 0; i < li.length; i += 22) if (li[i] === id) return [...li.subarray(i + 13, i + 16)]; };
/** per level, the most trains of `id` drawn at one moment of a day, moments with another level's trains left out */
function drawn(a, id) {
  const most = [0, 0, 0];
  for (let t = DAY; t < DAY + 86400; t += STEP) {
    const trips = a.line_trips(id, t - 20000, t + 1), c = [0, 0, 0];
    for (let i = 0; i < trips.length; i += 4) if (trips[i + 1] <= t && t < trips[i + 2]) c[trips[i] % 3]++;
    const all = c[0] + c[1] + c[2];
    c.forEach((x, lev) => { if (x === all) most[lev] = Math.max(most[lev], x); });
  }
  return most;
}
let failed = 0;
function compare(a, id, what) {
  const b = bought(a, id), h = a.line_held(id), d = drawn(a, id);
  [0, 1, 2].forEach((lev) => {
    if (!d[lev]) return;
    const panel = heldTrains(h[7 * lev]), out = b[lev] - d[lev];
    const ok = panel === out;
    if (!ok) failed++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}, ${['high', 'medium', 'low'][lev]}: bought ${b[lev]}, at most ${d[lev]} running, panel ${panel} (line_held ${h[7 * lev].toFixed(3)})`);
  });
}
const byHand = (id, trains, what) => {
  const a = fresh();
  if (a.set_train_counts(id, ...trains) !== 0) throw new Error(a.issues());
  compare(a, id, what);
  a.free();
};
byHand(E, [43, 38, 30], 'E by hand, over its limit');
byHand(F, [70, 66, 40], 'F by hand, over its limit');
byHand(FIVE, [60, 57, 20], '5 by hand, over its limit');
byHand(E, [33, 32, 10], 'E by hand, a whole train, then part of one, over');
const stops = new Uint32Array(fresh().line_stops(E));
for (const trains of [24, 30, 40]) {
  const a = fresh(), o = a.new_line_id();
  if (a.set_line_with_trains(o, stops, trains, 1, 1, 30, 180, 0, 0x123456, 'Second') !== 0) throw new Error(a.issues());
  compare(a, E, `E sharing its track with ${trains} more trains`);
  compare(a, o, `the line sharing E's track, ${trains} trains`);
  a.free();
}
process.exit(failed ? 1 : 0);
