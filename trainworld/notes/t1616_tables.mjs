// T-1616: tables from t1616/summary.json (made by t1616_money.mjs).
import { readFileSync } from 'node:fs';
const DIR = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1616/';
const rows = JSON.parse(readFileSync(DIR + 'summary.json', 'utf8'));
const d = JSON.parse(readFileSync(DIR + 'lines.json', 'utf8'));
console.log('fares', JSON.stringify(d.money.fares), 'trips', d.trips, 'rail', d.railTrips, 'split', JSON.stringify(d.split));
const M = (v) => (v / 1e6).toFixed(1);
const sum = (rs, k) => rs.reduce((a, r) => a + r[k], 0);
const group = (label, rs) => {
  const f = sum(rs, 'fares'), r = sum(rs, 'running'), u = sum(rs, 'upkeep'), km = sum(rs, 'km'), pkm = sum(rs, 'pkm'), ck = sum(rs, 'carKm'), tk = sum(rs, 'trainKm');
  console.log(`${label.padEnd(22)} n ${String(rs.length).padStart(4)} km ${km.toFixed(0).padStart(6)}  fares ${M(f).padStart(7)} run ${M(r).padStart(7)} upkeep ${M(u).padStart(6)} net ${M(f - r - u).padStart(7)}  losing ${rs.filter((x) => x.net < 0).length}  pkm/car-km ${(pkm / ck).toFixed(1)}  fare/pkm real $${(f / 100 / pkm).toFixed(3)}  run/train-km $${(r / 100 / tk).toFixed(1)}  pkm/route-km/day ${(pkm / km).toFixed(0)}`);
};
group('all', rows);
for (const [a, b] of [[0, 10], [10, 30], [30, 60], [60, 100], [100, 200], [200, 400], [400, 2000]]) group(`km ${a}-${b}`, rows.filter((r) => r.km >= a && r.km < b));
group('16-car (Shinkansen)', rows.filter((r) => r.cars === 16));
group('fast trains', rows.filter((r) => r.fast));
group('other', rows.filter((r) => r.cars !== 16));
// top losers
const by = [...rows].sort((a, b) => a.net - b.net);
console.log('\nworst 25 lines: name, from-to, km, stops, cars, tph h/m/l, fares, run, upkeep, net ($M a game day), pkm/car-km, km/h end to end, real D vs game density');
for (const r of by.slice(0, 25)) console.log(`${(r.name ?? '').slice(0, 14).padEnd(14)} ${(r.from ?? '').slice(0, 8)}-${(r.to ?? '').slice(0, 8)} km ${r.km.toFixed(0)} st ${r.stops} cars ${r.cars}${r.fast ? 'F' : ''} tph ${r.tph.map((v) => v.toFixed(1)).join('/')} f ${M(r.fares)} run ${M(r.running)} up ${M(r.upkeep)} net ${M(r.net)} occ ${r.perCar.toFixed(1)} ${r.kmh.toFixed(0)}km/h D ${r.realD ? r.realD.toFixed(0) : '-'} game ${r.density.toFixed(0)} curveKm ${r.limitKm.curve.toFixed(0)} xover ${r.limitKm.crossover.toFixed(0)} delay ${r.delayS.map((v) => v.toFixed(0)).join('/')}`);
console.log('\nbest 10');
for (const r of by.slice(-10).reverse()) console.log(`${(r.name ?? '').slice(0, 14).padEnd(14)} km ${r.km.toFixed(0)} cars ${r.cars} tph ${r.tph.map((v) => v.toFixed(1)).join('/')} f ${M(r.fares)} run ${M(r.running)} up ${M(r.upkeep)} net ${M(r.net)} occ ${r.perCar.toFixed(1)} D ${r.realD ? r.realD.toFixed(0) : '-'} game ${r.density.toFixed(0)}`);
// concentration of losses
const losses = by.filter((r) => r.net < 0), tot = sum(losses, 'net');
let acc = 0, k = 0;
for (const r of losses) { acc += r.net; k++; if (acc / tot > 0.5) break; }
console.log(`\n${losses.length} losing lines lose ${M(tot)}; the worst ${k} make half; profit of the winners ${M(sum(by.filter((r) => r.net >= 0), 'net'))}`);
// game density vs real D
const withD = rows.filter((r) => r.realD && r.cars !== 16);
const ratios = withD.map((r) => r.density / r.realD).sort((a, b) => a - b);
const q = (p) => ratios[Math.floor(p * (ratios.length - 1))];
console.log(`game riders past a point / MLIT count (non-Shinkansen, ${withD.length} lines): p10 ${q(0.1).toFixed(2)} p25 ${q(0.25).toFixed(2)} median ${q(0.5).toFixed(2)} p75 ${q(0.75).toFixed(2)} p90 ${q(0.9).toFixed(2)}`);
for (const [a, b] of [[0, 30], [30, 100], [100, 2000]]) {
  const s = withD.filter((r) => r.km >= a && r.km < b), rr = s.map((r) => r.density / r.realD).sort((x, y) => x - y);
  console.log(`  km ${a}-${b}: n ${s.length} median ${rr[rr.length >> 1]?.toFixed(2)} weighted ${(sum(s, 'pkm') / s.reduce((x, r) => x + r.realD * r.km, 0)).toFixed(2)}`);
}
const sh = rows.filter((r) => r.cars === 16);
for (const r of sh) console.log(`  shinkansen ${r.name} ${r.from}-${r.to} km ${r.km.toFixed(0)} D ${r.realD?.toFixed(0)} game ${r.density.toFixed(0)} kmh ${r.kmh.toFixed(0)} fast ${r.fast} net ${M(r.net)} fare/pkm $${r.farePerPkmReal.toFixed(3)}`);
// speeds
const sp = rows.filter((r) => r.km > 50);
console.log('\nend-to-end km/h for lines over 50 km: median', sp.map((r) => r.kmh).sort((a, b) => a - b)[sp.length >> 1].toFixed(0));
const ck = sum(rows, 'km'), cur = rows.reduce((a, r) => a + r.limitKm.curve, 0), xo = rows.reduce((a, r) => a + r.limitKm.crossover, 0);
console.log(`stretch km limited by curves ${(100 * cur / ck).toFixed(1)}%, by crossovers ${(100 * xo / ck).toFixed(1)}%`);
