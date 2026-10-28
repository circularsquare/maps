// T-1616: what limits the trains on the all-Japan save: each stretch's limit (`why`: 0 trains, 1 track,
// 2 curves, 3 crossover), the speed reached, and the capacity delay over each line's round trip.
import { readFileSync } from 'node:fs';
const d = JSON.parse(readFileSync('C:/Users/anita/projects/maps/artifacts/trainworld/t1616/lines.json', 'utf8'));
const why = [0, 0, 0, 0], km = [0, 0, 0, 0];
let hops = 0, hopKm = 0, under = 0, underKm = 0;
const delayed = [];
for (const l of d.lines) {
  l.stretchSpeeds.forEach(([reached, limit, w], k) => {
    const h = l.hopKm[k] ?? 0;
    why[w] = (why[w] ?? 0) + 1; km[w] = (km[w] ?? 0) + h;
    hops++; hopKm += h;
    if (limit < 160) { under++; underKm += h; }
  });
  const rt = l.roundTrip?.[0] ?? 0;
  if (rt > 0) delayed.push({ name: l.name, km: l.km, share: (l.delayS?.[0] ?? 0) / rt, delay: l.delayS?.[0] ?? 0 });
}
console.log('stretches by limit (count, km): trains', why[0], km[0].toFixed(0), 'track', why[1], km[1].toFixed(0), 'curves', why[2], km[2].toFixed(0), 'crossover', why[3], km[3].toFixed(0));
console.log('stretches with a limit under 160 km/h:', under, 'of', hops, underKm.toFixed(0), 'km of', hopKm.toFixed(0));
delayed.sort((a, b) => b.share - a.share);
console.log('lines with capacity delay over 5% of the peak round trip:', delayed.filter((x) => x.share > 0.05).length, 'of', delayed.length);
for (const x of delayed.slice(0, 12)) console.log(`  ${x.name} ${x.km.toFixed(0)} km delay ${x.delay.toFixed(0)} s = ${(100 * x.share).toFixed(1)}%`);
// mean spacing and speed by length
for (const [a, b] of [[0, 30], [30, 100], [100, 300], [300, 2000]]) {
  const s = d.lines.filter((l) => l.status === 'running' && l.km >= a && l.km < b && l.cars !== 16);
  const sp = s.reduce((x, l) => x + l.km, 0) / s.reduce((x, l) => x + l.nStops - 1, 0);
  const v = s.reduce((x, l) => x + l.km, 0) / s.reduce((x, l) => x + l.endS / 3600, 0);
  console.log(`km ${a}-${b}: ${s.length} lines, stop every ${sp.toFixed(1)} km, ${v.toFixed(0)} km/h end to end`);
}
