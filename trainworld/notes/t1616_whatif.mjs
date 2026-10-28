// T-1616: rough static what-ifs on the all-Japan save (riders held fixed, so each is an upper bound on
// the money and says nothing of riders lost to dearer fares).
import { readFileSync } from 'node:fs';
const DIR = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1616/';
const rows = JSON.parse(readFileSync(DIR + 'summary.json', 'utf8'));
const d = JSON.parse(readFileSync(DIR + 'lines.json', 'utf8'));
console.log('fareSense', JSON.stringify(d.fareSense));
const M = (v) => (v / 1e6).toFixed(0);
const sum = (rs, f) => rs.reduce((a, r) => a + f(r), 0);
const base = { fares: sum(rows, (r) => r.fares), run: sum(rows, (r) => r.running), up: sum(rows, (r) => r.upkeep) };
const show = (label, f, run, up) => console.log(`${label.padEnd(58)} fares ${M(f).padStart(6)} running ${M(run).padStart(6)} upkeep ${M(up).padStart(5)} net ${M(f - run - up).padStart(6)}`);
show('as loaded', base.fares, base.run, base.up);
// 1. today's default curve (T-1521): past 20 km $0.02155 a km (India units) where the save has $0.00865
const scale = 6.1, X = 100, extraPerKm = 0.02155 - 0.008653;
let extra = 0;
for (const r of rows) {
  const ride = r.riders > 0 ? r.pkm / r.riders : 0;
  extra += r.riders * Math.max(0, ride - 20) * extraPerKm * scale * X;
}
show('1. the default curve on rides past 20 km (riders fixed)', base.fares + extra, base.run, base.up);
// 2. the save's service cut to its riders: each line keeps the cars it needs so its average car carries
// what the Tokaido Shinkansen's does in this save (~60 riders a car-km), never more cars than now
const target = 40;
let run2 = 0;
for (const r of rows) {
  const k = r.perCar > 0 ? Math.min(1, r.perCar / target) : 0.05;
  // a train-km part stays with frequency: cut cars first, down to 2 a train, then trains
  const runTrain = r.trainKm * 4 * X, runCar = r.running - runTrain;
  const carsNow = r.cars, carsNeed = Math.max(2, Math.ceil(carsNow * k));
  const trainCut = carsNow * k < 2 ? Math.max(0.1, (carsNow * k) / 2) : 1;
  run2 += runTrain * trainCut + (runCar * carsNeed) / carsNow * trainCut;
}
show(`2. trains cut to ${target} riders a car-km on average`, base.fares, run2, base.up);
show('1 and 2 together', base.fares + extra, run2, base.up);
// 3. without the lines over 200 km (the N02 long lines run end to end)
const short = rows.filter((r) => r.km < 200);
show('3. lines under 200 km only', sum(short, (r) => r.fares), sum(short, (r) => r.running), sum(short, (r) => r.upkeep));
// 4. upkeep at JR-like real levels instead (x2, x5)
show('4a. upkeep doubled', base.fares, base.run, 2 * base.up);
show('4b. no upkeep at all', base.fares, base.run, 0);
// cost shares
console.log(`cost shares: running ${(100 * base.run / (base.run + base.up)).toFixed(0)}%, upkeep ${(100 * base.up / (base.run + base.up)).toFixed(0)}%`);
const trainPart = sum(rows, (r) => r.trainKm * 4 * X);
console.log(`running: train-km part ${M(trainPart)}, car-km part ${M(base.run - trainPart)}; train-km ${M(sum(rows, (r) => r.trainKm))}M car-km ${M(sum(rows, (r) => r.carKm))}M pkm ${M(sum(rows, (r) => r.pkm))}M a day`);
const upTrack = d.money.upkeep;
console.log('network upkeep', JSON.stringify(upTrack));
