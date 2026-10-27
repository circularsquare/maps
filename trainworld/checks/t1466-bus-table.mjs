// T-1466: the table of notes/T-1466.md from t1466-bus-sweep.mjs output: node trainworld/checks/t1466-bus-table.mjs <sweep.json>...
import { readFileSync } from 'node:fs';
const files = process.argv.slice(2);
const PH = [4, 6, 4];
const HOURS = [14, 4, 6]; // high 6-20, medium 20-24, low 0-6
const rows = [];
for (const f of files) {
  const j = JSON.parse(readFileSync(f, 'utf8'));
  const b0 = j.base;
  const w0 = (b0.split.walk / 100) * b0.trips, d0 = (b0.split.drive / 100) * b0.trips, r0 = b0.rail;
  for (const s of j.scen) {
    const sn = s.snap, l = sn.line;
    if (!l) { rows.push({ id: s.id, missing: true }); continue; }
    const road = s.add.km, hw = s.hw / 60;
    const tn = s.add.trainsNeeded ?? l.trainsNeeded ?? [];
    const busHours = tn.reduce((a, n, i) => a + n * HOURS[i], 0);
    const w1 = (sn.split.walk / 100) * sn.trips, d1 = (sn.split.drive / 100) * sn.trips, r1 = sn.rail;
    const dR = r1 - r0, dW = w1 - w0, dD = d1 - d0;
    // peak hour per direction in the high periods vs crush capacity
    let peak = 0;
    for (let p = 0; p < 3; p++) peak = Math.max(peak, (l.maxLoadPerPeriod[p] / (PH[p])) * sn.peakHourFactor[p]);
    const cap = (60 / hw) * 80;
    const kinds = l.stops.map((x) => x.kind);
    let bSt = 0, aSt = 0, bSt0 = 0;
    kinds.forEach((k, i) => { if (k === 1) { bSt += l.boardStop[i]; aSt += l.alightStop[i]; } });
    const arrive = l.arriveStop.reduce((a, b) => a + b, 0), board = l.boardStop.reduce((a, b) => a + b, 0);
    rows.push({
      id: s.id, straight: s.straightKm, road, hw, buses: Math.max(...tn), busHours,
      riders: Math.round(l.riders), perKm: Math.round(l.riders / road), perBusHr: +(l.riders / busHours).toFixed(0),
      peakHr: Math.round(peak), cap, load: +(peak / cap).toFixed(2), fullest: l.fullest,
      stopsN: kinds.length, kinds: kinds.join(''),
      stationEnd: Math.round(bSt + aSt), stationEndPct: Math.round((100 * (bSt + aSt)) / l.riders),
      dRail: Math.round(dR), dWalk: Math.round(dW), dDrive: Math.round(dD), turnedAway: Math.round(arrive - board),
    });
  }
}
console.table(rows);
