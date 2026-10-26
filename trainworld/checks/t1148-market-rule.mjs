// T-1148: the main thread's copy of the market's place-pair rule (app/src/game/marketRule.ts)
// against the sim's (sim/src/demand/intercity.rs) on the world's rows: for a few thousand pairs of
// places (near ones, far ones by road, across the Channel, to and from Hokkaido, across seas), the
// distance, the car's minutes to the far place, the reference minutes, the corridor's plane and car
// shares and the trips a day; and the longer trips' period shares on every kind of day. Run after
// any change to the market's rule on either side.
//   node trainworld/checks/t1148-market-rule.mjs
import { readFileSync } from 'node:fs';
import { loadApp, marketRows, registry } from './lib.mjs';
import init, { MarketPacks, day_factors } from '../app/src/wasm/demand/sim.js';

await init({ module_or_path: readFileSync(new URL('../app/src/wasm/demand/sim_bg.wasm', import.meta.url)) });
// game/cities.ts reads the view from the URL as it loads
globalThis.location ??= { search: '', hash: '', href: 'http://localhost/' };
const rule = await loadApp("export * from '@app/src/game/marketRule.ts'; export { marketWorld } from '@app/src/game/cities.ts';");
const reg = registry();
const rowsText = marketRows();
const rows = JSON.parse(rowsText);
rule.setMarketRows(rows);
// T-1277: the world as the game gives it the sim, its rows carrying each country's fare weight
const world = rule.marketWorld(rowsText);
const api = new MarketPacks(world).market();
// a market with no network, so `market_trip_between` answers from the rows
api.set_network_ll(new Float64Array(0), new Uint32Array(0), new Uint32Array(0), new Float32Array(0), new Float32Array(0), new Float32Array(0));
api.solve(0);
const modes = api.corridor_modes(), trips = api.corridor_trips();
const N = reg.market.places.length;
const corridor = (a, b) => { const i = Math.min(a, b), j = Math.max(a, b); return i * N - (i * (i - 1)) / 2 + (j - i); };

let fails = 0;
const fail = (msg) => { if (fails++ < 20) console.log('  ' + msg); };
const near = (x, y, abs, rel = 0) => Math.abs(x - y) <= abs + rel * Math.abs(y);

// pairs: every place with a row, kinds of pair sampled by a fixed generator
const places = [...new Set(Object.values(rows).map((r) => r.place))].sort((a, b) => a - b);
const row = new Map(Object.values(rows).map((r) => [r.place, r]));
let seed = 1148;
const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 2 ** 32);
const pick = () => places[Math.floor(rnd() * places.length)];
const kinds = { near: [], far: [], channel: [], hokkaido: [], sea: [] };
const want = 400;
const named = (n) => reg.market.places.findIndex((p) => p.name === n);
const fixed = [['London', 'Paris'], ['Paris', 'London'], ['Tokyo', 'Sapporo'], ['New York', 'Boston'], ['New York', 'Philadelphia']].map(([a, b]) => [named(a), named(b)]).filter(([a, b]) => a >= 0 && b >= 0);
for (let tries = 0; tries < 2e6 && Object.values(kinds).some((k) => k.length < want); tries++) {
  const a = pick(), b = pick();
  if (a === b) continue;
  const p = row.get(a), q = row.get(b);
  const km = rule.placeKm(p, q);
  const hk = (r) => r.country === 'JP' && r.lat > 41.4;
  const k = p.road !== q.road ? 'sea' : (p.country === 'GB') !== (q.country === 'GB') ? 'channel' : hk(p) !== hk(q) ? 'hokkaido' : km < 250 ? 'near' : 'far';
  if (kinds[k].length < want) kinds[k].push([a, b]);
}
const pairs = [...fixed, ...Object.values(kinds).flat()];
let worst = { car: 0, ref: 0, air: 0, trips: 0 };
for (const [a, b] of pairs) {
  const h = api.market_trip_between(0, 0, a, -1, 0, b, -1);
  const c = rule.corridorOf(a, b), car = rule.carBetween(a, b);
  const label = `${reg.market.places[a].name} - ${reg.market.places[b].name}`;
  if (!near(c.km, h[8], 1e-6, 1e-7)) fail(`${label}: km ${c.km} against the sim's ${h[8]}`);
  const simCar = h[7] < 0 ? NaN : h[7];
  if (Number.isNaN(simCar) !== !car || (car && !near(car.total, simCar, 0.02, 1e-5))) fail(`${label}: car ${car?.total} min against the sim's ${simCar}`);
  const ref = c.refMin;
  if (!near(ref, h[17], 0.02, 1e-5)) fail(`${label}: reference ${ref} min against the sim's ${h[17]}`);
  const k = corridor(a, b);
  // T-1277: a corridor splits by the mean of its two ways, a plane weighing the departing country's fare
  const back = rule.corridorOf(b, a), air = 0.5 * (c.air + back.air), carShare = c.road ? 1 - air : 0;
  if (!near(air, modes[2 * k], 1e-4) || !near(carShare, modes[2 * k + 1], 1e-4)) fail(`${label}: plane ${air} car ${carShare} against the sim's ${modes[2 * k]} ${modes[2 * k + 1]}`);
  const t = rule.placeTrips(a, b);
  if (!near(2 * t, trips[k], 1e-6, 1e-5)) fail(`${label}: ${t} trips each way against the sim's ${trips[k] / 2}`);
  if (car) worst.car = Math.max(worst.car, Math.abs(car.total - simCar));
  if (Number.isFinite(ref)) worst.ref = Math.max(worst.ref, Math.abs(ref - h[17]));
  worst.air = Math.max(worst.air, Math.abs(air - modes[2 * k]));
  worst.trips = Math.max(worst.trips, trips[k] > 0 ? Math.abs(2 * t - trips[k]) / trips[k] : 0);
}
console.log(`${fails ? 'FAIL' : 'PASS'} t1148 place pairs: ${pairs.length} (${Object.entries(kinds).map(([k, v]) => `${v.length} ${k}`).join(', ')}), worst car ${worst.car.toFixed(4)} min, reference ${worst.ref.toFixed(4)} min, plane share ${worst.air.toExponential(1)}, trips ${worst.trips.toExponential(1)} of theirs`);

// the longer trips' period shares (days.rs share_out = SHARES x load factor for them)
// day_factors is [5 days][purposes][5 periods] then [5 days][purposes] rates (weekday, Saturday, Sunday, away, average: T-264); the longer trips are the last purpose
const lf = day_factors(), SH = [0.2, 0.35, 0.25, 0.15, 0.05];
const DAYS = 5, PURPOSES = lf.length / (DAYS * 6), LONGER = PURPOSES - 1;
let dayFails = 0;
for (let d = 0; d < DAYS; d++) for (let q = 0; q < 5; q++) {
  const sim = SH[q] * lf[(d * PURPOSES + LONGER) * 5 + q];
  if (!near(rule.longerShare(d, q), sim, 1e-5)) { dayFails++; console.log(`  day ${d} period ${q}: ${rule.longerShare(d, q)} against the sim's ${sim}`); }
}
console.log(`${dayFails ? 'FAIL' : 'PASS'} t1148 longer trips' period shares on ${DAYS} kinds of day`);
// London - Paris for the eye
const [L, P] = fixed[0] ?? [];
if (L >= 0) {
  const c = rule.corridorOf(L, P), car = rule.carBetween(L, P);
  console.log(`  London - Paris: ${c.km.toFixed(0)} km, car ${car.total.toFixed(1)} min (drive ${car.driveMin.toFixed(1)}, running $${car.runUsd.toFixed(2)}, parking ${car.parkMin.toFixed(1)}, ${car.crossings.map((x) => x.kind).join('+')}), plane ${c.flight?.total.toFixed(1)} min, plane share ${(100 * c.air).toFixed(1)}%, ${rule.placeTrips(L, P).toFixed(0)} trips a weekday each way`);
}
process.exit(fails || dayFails ? 1 : 0);
