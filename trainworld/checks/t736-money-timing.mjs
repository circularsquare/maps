// T-736: an hour's money (app/src/game/moneyTiming.ts) keeps every week's totals and moves without
// cliffs: fares, rides and running costs summed over a week's hours match the period tables, and the
// largest change from one hour to the next is printed beside the old flat-period booking's.
//   node trainworld/checks/t736-money-timing.mjs
// Prints PASS or FAIL and exits 1 on failure. A few seconds.
import { fileURLToPath } from 'node:url';
import { build } from '../app/node_modules/vite/dist/node/index.js';

const root = fileURLToPath(new URL('../app', import.meta.url));
const bundle = await build({ configFile: false, root, logLevel: 'silent', build: {
  write: false, minify: false, lib: { entry: 'src/game/moneyTiming.ts', formats: ['es'] },
} });
const out = (Array.isArray(bundle) ? bundle[0] : bundle).output;
const m = await import('data:text/javascript;base64,' + Buffer.from(out.find((o) => o.type === 'chunk').code).toString('base64'));

const PER = [[1000, 600, 900, 500, 80], [500, 550, 520, 420, 90]]; // US$ an hour, weekday and weekend, per period
const RUN = [300, 180, 60]; // high, medium, low
const LEVEL = [0, 1, 0, 1, 2]; // each period's level
const FROM = [6, 10, 16, 20, 0], TO = [10, 16, 20, 24, 6];
const c = [m.strengths([1.4, 1.15, 1.4, 1.3, 1.5]), m.strengths([1.3, 1.1, 1.3, 1.2, 1.5])];
const fares = m.table(PER), cost = (_k, p) => RUN[LEVEL[p]];
const periodOf = (h) => FROM.findIndex((f, i) => h >= f && h < TO[i]);
const kindOf = (day) => ((day - 1) % 7 + 7) % 7 >= 5 ? 1 : 0;
const flat = (tab, t) => { const d = Math.floor(t / 86400); return tab(kindOf(d), periodOf(Math.floor(t / 3600) % 24)); };

let ok = true;
for (const lag of [0, 1500, 3000, 5400]) {
  const week0 = 7 * 86400 * 3; // whole weeks, so the pattern repeats on both sides
  let sumF = 0, sumR = 0, sumC = 0, sumFlatF = 0, sumFlatC = 0, stepF = 0, stepFlat = 0, stepC = 0, stepFlatC = 0;
  let prev = null;
  for (let h = 0; h < 7 * 24; h++) {
    const t = week0 + h * 3600;
    const f = m.lagged(fares, c, t, lag), r = m.between(fares, c, t, t + 3600), k = m.lagged(cost, null, t, lag);
    const ff = flat(fares, t), fc = flat(cost, t);
    sumF += f; sumR += r; sumC += k; sumFlatF += ff; sumFlatC += fc;
    if (prev) {
      stepF = Math.max(stepF, Math.abs(f - prev[0])); stepFlat = Math.max(stepFlat, Math.abs(ff - prev[1]));
      stepC = Math.max(stepC, Math.abs(k - prev[2])); stepFlatC = Math.max(stepFlatC, Math.abs(fc - prev[3]));
    }
    prev = [f, ff, k, fc];
  }
  const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
  const pass = near(sumF, sumFlatF) && near(sumR, sumFlatF) && near(sumC, sumFlatC);
  ok &&= pass;
  console.log(`lag ${lag / 60} min: week fares ${sumF.toFixed(3)} rides-shaped ${sumR.toFixed(3)} vs ${sumFlatF}; costs ${sumC.toFixed(3)} vs ${sumFlatC}; largest hour-to-hour change fares ${stepF.toFixed(0)} (flat ${stepFlat}), costs ${stepC.toFixed(0)} (flat ${stepFlatC}) ${pass ? '' : 'MISMATCH'}`);
}
// one weekday and the weekend's start, hour by hour at a 25-minute lag
const day = 7 * 86400 * 3 + 4 * 86400; // a Friday
console.log('Friday 00:00 to Saturday 06:00, fares by hour:', Array.from({ length: 30 }, (_, h) => m.lagged(fares, c, day + h * 3600, 1500).toFixed(0)).join(' '));
console.log(`${ok ? 'PASS' : 'FAIL'}  money timing keeps each week's fares, rides and running costs`);
process.exit(ok ? 0 : 1);
