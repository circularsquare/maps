// T-1277: fares by national income. A rider in New York and one in Mumbai, on the same small line
// (the WASM check's three-cell fixture), set up as the demand worker sets up each zone's
// population: the country's minutes a fare dollar (`set_fare_weight`) and the curve times the
// country's scale (`set_fares`). Each pays the starting curve times their own scale, a player's
// scale replaces the income's, and at the income's scales a fare weighs about the same everywhere.
//   node trainworld/checks/t1277-fare-scale.mjs
import { readFileSync } from 'node:fs';
import { loadApp } from './lib.mjs';
import init, { DemandApi } from '../app/src/wasm/demand/sim.js';

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url)) });
const app = await loadApp("export * from '@app/src/game/income.ts'; export { START_FARE } from '@app/src/game/fareCurve.ts';");
const { fareScale, fareWeight, incomeScale, START_FARE } = app;

function fixture() {
  const arrays = [], parts = [];
  let offset = 0;
  const weights = { x_m: [0, 4000, 8000], y_m: [0, 0, 0], pop: [1000, 2000, 500], jobs: [1000, 1000, 1500] };
  for (const [name, v] of Object.entries(weights)) {
    const p = new Float32Array(v);
    arrays.push({ name, per: 'cell', dtype: 'f32', count: v.length, offset });
    parts.push(new Uint8Array(p.buffer));
    offset += p.byteLength;
  }
  const bin = new Uint8Array(offset);
  let at = 0;
  for (const p of parts) { bin.set(p, at); at += p.length; }
  return [JSON.stringify({ format: 1, city: 'fixture', cells: 3, arrays }), bin];
}
const [h, b] = fixture();

/** The base fare a rider from `country` pays and its perceived minutes, with the player's `scales`. */
function ride(country, scales = {}) {
  const a = new DemandApi(h, b);
  const times = new Float32Array(18);
  for (let level = 0; level < 3; level++) { times[level * 3] = 60; times[level * 3 + 1] = 60; times[(level + 3) * 3 + 1] = 60; times[(level + 3) * 3 + 2] = 60; }
  a.set_network(new Float32Array([0, 0, 4000, 0, 8000, 0]), new Uint32Array([3]), new Uint32Array([0, 1, 2]), times, new Float32Array([12, 12, 12]), new Float32Array([4]));
  const s = fareScale(country, scales);
  a.set_fare_weight(fareWeight(country));
  a.set_fares(START_FARE.base * s, START_FARE.perKm * s, new Float32Array([4, 4, 0, 0, 4, 4]));
  a.solve(0);
  const t = a.trip(0, 0, 0, 8000, 0);
  a.free();
  return { scale: s, perceived: t[22], usd: t[22] / t[23] };
}

let fails = 0;
const check = (ok, msg) => { if (!ok) { fails++; console.log('  ' + msg); } };
const near = (x, y, rel) => Math.abs(x - y) <= rel * Math.abs(y);

check(incomeScale('IN') === 1, `India's scale ${incomeScale('IN')}, the baseline's, not 1`);
check(incomeScale('US') === 2.8, `the US scale ${incomeScale('US')}, not 2.8`);
check(near(fareWeight('US'), 4.42, 0.01) && near(fareWeight('IN'), 12.2, 0.01), `minutes a dollar: US ${fareWeight('US')}, India ${fareWeight('IN')}`);
const ny = ride('US'), mum = ride('IN'), own = ride('US', { US: 1.6 });
check(near(ny.usd, START_FARE.base * 2.8, 1e-4), `New York's base fare $${ny.usd}, not ${START_FARE.base} x 2.8`);
check(near(mum.usd, START_FARE.base, 1e-4), `Mumbai's base fare $${mum.usd}, not ${START_FARE.base}`);
check(near(own.usd, START_FARE.base * 1.6, 1e-4), `New York at the player's 1.6x pays $${own.usd}`);
check(near(ny.perceived, mum.perceived, 0.05), `at the income's scales the base fare weighs ${ny.perceived} min in New York, ${mum.perceived} in Mumbai`);
console.log(`${fails ? 'FAIL' : 'PASS'} t1277 fare scales: base fare New York $${ny.usd.toFixed(2)} (${ny.scale}x, ${ny.perceived.toFixed(1)} min), Mumbai $${mum.usd.toFixed(2)} (${mum.scale}x, ${mum.perceived.toFixed(1)} min), New York at 1.6x $${own.usd.toFixed(2)}`);
process.exitCode = fails ? 1 : 0;
