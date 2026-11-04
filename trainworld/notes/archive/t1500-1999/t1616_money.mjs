// T-1616: each line's and city's money a game day on the all-Japan save, from the dump the page made
// (../artifacts/trainworld/t1616/lines.json, regions.json). Writes t1616/summary.json and prints tables.
//   node trainworld/notes/t1616_money.mjs
import { readFileSync, writeFileSync } from 'node:fs';

const DIR = 'C:/Users/anita/projects/maps/artifacts/trainworld/t1616/';
const d = JSON.parse(readFileSync(DIR + 'lines.json', 'utf8'));
const m = d.money, X = m.economy;
const LEVEL = { high: 0, medium: 1, low: 2 };
const levelHours = [0, 0, 0];
d.levels.forEach((lv, p) => { levelHours[LEVEL[lv]] += d.hours[p]; });
const SEATS_SHINK = 80, SEATS = 150;

const rows = [];
for (const l of d.lines) {
  if (l.status !== 'running') continue;
  const runTrain = l.fast ? m.hsRunCostTrainKm : m.runCostTrainKm, runCar = l.fast ? m.hsRunCostCarKm : m.runCostCarKm;
  let trainKm = 0;
  for (let lv = 0; lv < 3; lv++) trainKm += (l.tph[lv] ?? 0) * 2 * l.km * levelHours[lv];
  const carKm = trainKm * l.cars;
  const running = (trainKm * runTrain + carKm * runCar) * X;
  const fares = l.faresByKind[0];
  const faresWeek = (5 * l.faresByKind[0] + l.faresByKind[1] + l.faresByKind[2]) / 7;
  const pkm = l.pkm.reduce((a, v) => a + v, 0);
  // the save's trains came from MLIT's passing count D (people a day past a section, the line's
  // median): peak each way = 0.06 D / seats, clipped to 0.5-24 (pipeline/realnet_n02.py)
  const seats = (l.cars === 16 ? SEATS_SHINK : SEATS) * l.cars;
  const clipped = l.tph[0] <= 0.5001 || l.tph[0] >= 23.999;
  const realD = clipped ? null : (l.tph[0] * seats) / 0.06;
  rows.push({
    density: l.km ? pkm / l.km : 0, realD,
    id: l.id, name: l.name, from: l.firstStop, to: l.lastStop, km: l.km, stops: l.nStops, cars: l.cars, fast: l.fast, tph: l.tph,
    zones: [...new Set(l.zones)].filter(Boolean),
    fares, faresWeek, running, upkeep: l.upkeep, net: fares - running - l.upkeep,
    trainKm, carKm, pkm, riders: l.riders,
    perCar: carKm ? pkm / carKm : 0,
    farePerPkmReal: pkm ? fares / X / pkm : 0,
    endS: l.endS, kmh: l.endS ? l.km / (l.endS / 3600) : 0,
    limitKm: l.limitKm, delayS: l.delayS, passed: l.passed,
  });
}
writeFileSync(DIR + 'summary.json', JSON.stringify(rows));
const sum = (k, rs = rows) => rs.reduce((a, r) => a + r[k], 0);
const M = (v) => (v / 1e6).toFixed(1);
console.log('lines', rows.length, 'fares $M', M(sum('fares')), 'running', M(sum('running')), 'upkeep', M(sum('upkeep')), 'net', M(sum('net')));
