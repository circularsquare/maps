// T-3373 B stage 2: sums in whole steps (app/src/workers/exactSum.ts) come out the same to the bit in any order and
// split into any partials, as workers holding different populations would add them. Node 22.18+, no browser.
//   node trainworld/checks/t3373-sums.mjs
import { LOAD_SCALE, FARE_SCALE, stepOf, toF32, addInto } from '../app/src/workers/exactSum.ts';

let fails = 0, seed = 3373;
const ok = (cond, what) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${what}`); if (!cond) fails++; };
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const shuffle = (a) => { const b = a.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; };
const bits = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('hex');

// populations' loads on 200 route nodes: f32 values from 1e-4 to 2e5 riders, some zero, some negative (a window's delta)
const N = 200, P = 300;
const pops = Array.from({ length: P }, (_, p) => Float32Array.from({ length: N }, () => { const r = rnd(); return r < 0.3 ? 0 : (p % 17 === 0 ? -1 : 1) * 10 ** (-4 + 9.3 * rnd()); }));
const exactSum = (list, scale = LOAD_SCALE) => { const t = new Float64Array(N); for (const v of list) for (let k = 0; k < N; k++) t[k] += stepOf(v[k], scale); return t; };
const f32Sum = (list) => { const t = new Float32Array(N); for (const v of list) for (let k = 0; k < N; k++) t[k] += v[k]; return t; };

const whole = toF32(exactSum(pops));
let same = true, f32same = true;
for (let trial = 0; trial < 20; trial++) {
  const order = shuffle(pops);
  if (bits(toF32(exactSum(order))) !== bits(whole)) same = false;
  if (bits(f32Sum(order)) !== bits(f32Sum(pops))) f32same = false;
}
ok(same, 'whole-step sums: the same bits in 20 shuffled orders');
ok(!f32same, 'control: plain f32 sums differ between orders (so the test can fail)');

let parted = true;
for (let trial = 0; trial < 20; trial++) {
  const workers = 1 + Math.floor(rnd() * 5), parts = Array.from({ length: workers }, () => []);
  for (const v of shuffle(pops)) parts[Math.floor(rnd() * workers)].push(v);
  const partials = shuffle(parts.map((l) => exactSum(l)));
  const t = partials.reduce((a, b) => addInto(a, b), new Float64Array(N));
  if (bits(toF32(t)) !== bits(whole)) parted = false;
}
ok(parted, 'split over 1-5 workers, partials added in any order: the same bits as one worker');

let off = 0;
const plain = new Float64Array(N); for (const v of pops) for (let k = 0; k < N; k++) plain[k] += v[k];
for (let k = 0; k < N; k++) if (plain[k] !== 0) off = Math.max(off, Math.abs(whole[k] - plain[k]) / Math.abs(plain[k]));
ok(off < 1e-6, `against an f64 sum of the same terms, the largest relative change is ${off.toExponential(2)}`);

const fare = toF32(exactSum(pops.slice(0, 50), FARE_SCALE)), fare2 = toF32(exactSum(shuffle(pops.slice(0, 50)), FARE_SCALE));
ok(bits(fare) === bits(fare2), 'fare scale: the same bits in another order');
ok(stepOf(2 ** 33 - 1) === 2 ** 33 - 1 && stepOf(0.1) * LOAD_SCALE === Math.round(0.1 * LOAD_SCALE), 'a step is a whole number of 2^-20');

console.log(fails ? `${fails} failed` : 'all passed');
process.exit(fails ? 1 : 0);
