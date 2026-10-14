// T-285 check: the bubble selection's everyday far end added up from the workers' packed answers
// (app/src/workers/selectionMerge.ts) against the dense merge demandClient.ts did before, on
// random answers. Every number must be the same to the bit.
//   node --experimental-strip-types trainworld/checks/t285-merge-check.mjs
import { mergeEveryday } from '../app/src/workers/selectionMerge.ts';

let seed = 12345;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const KEYS = ['shopping', 'social', 'school', 'school', 'airport', 'medical']; // university shows as school

/** demandClient.ts `flows` before T-285, from dense answers (30 + 3 nc a kind). */
function oldMerge(dense, kinds, nc, commute, start) {
  let everyday = null;
  for (const v of dense) {
    if (!everyday) everyday = v.slice();
    else for (let k = 0; k < everyday.length; k++) everyday[k] += v[k];
  }
  const stride = 30 + 3 * nc;
  const far = [new Float32Array(nc), new Float32Array(nc), new Float32Array(nc)];
  const shown = new Map();
  const timing = [];
  for (let kind = 0; kind < kinds; kind++) {
    const values = everyday.subarray(kind * stride, (kind + 1) * stride);
    timing.push(values.slice(0, 30));
    for (let m = 0; m < 3; m++) for (let c = 0; c < nc; c++) far[m][c] += values[30 + m * nc + c];
    const key = KEYS[kind];
    let acc = shown.get(key);
    if (!acc) shown.set(key, (acc = { far: [new Float32Array(nc), new Float32Array(nc), new Float32Array(nc)], total: [0, 0, 0] }));
    for (let m = 0; m < 3; m++) for (let c = 0; c < nc; c++) acc.far[m][c] += values[30 + m * nc + c];
    for (let m = 0; m < 3; m++) for (let q = 0; q < 5; q++) acc.total[m] += values[m * 5 + q] + values[15 + m * 5 + q];
  }
  const keep = (f) => {
    const own = [];
    for (let c = 0; c < nc; c++) if (f[0][c] + f[1][c] + f[2][c] > 0) own.push(c);
    return { cells: Uint32Array.from(own), farRail: Float32Array.from(own, (c) => f[0][c]), farWalk: Float32Array.from(own, (c) => f[1][c]), farDrive: Float32Array.from(own, (c) => f[2][c]) };
  };
  const shownOut = new Map([...shown].map(([k, { far: f, total }]) => [k, { far: keep(f), total }]));
  for (let i = 0; i < commute.cells.length; i++) {
    const c = commute.cells[i];
    far[0][c] += commute.farRail[i]; far[1][c] += commute.farWalk[i]; far[2][c] += commute.farDrive[i];
  }
  const legs = [...start];
  for (let kind = 0; kind < kinds; kind++) for (let m = 0; m < 3; m++) {
    const values = everyday.subarray(kind * stride, kind * stride + 30);
    let v = 0; for (let q = 0; q < 5; q++) v += values[m * 5 + q] + values[15 + m * 5 + q];
    legs[m] += v;
  }
  return { timing, shown: shownOut, far: keep(far), legs };
}

const same = (a, b) => a.length === b.length && Array.prototype.every.call(a, (v, i) => Object.is(v, b[i]));
let cases = 0;
for (let trial = 0; trial < 200; trial++) {
  const nc = 1 + Math.floor(rand() * 400), kinds = 6, workers = 1 + Math.floor(rand() * 5);
  // each kind's far cells: the same on every worker (every worker visits the same samples)
  const reach = Array.from({ length: kinds }, () => Array.from({ length: nc }, (_, c) => c).filter(() => rand() < 0.3));
  const dense = [], packed = [];
  for (let w = 0; w < workers; w++) {
    const d = new Float32Array(kinds * (30 + 3 * nc)), p = [];
    for (let kind = 0; kind < kinds; kind++) {
      const base = kind * (30 + 3 * nc), t = [];
      for (let k = 0; k < 30; k++) t.push((d[base + k] = Math.fround(rand() * 1000)));
      const cells = reach[kind], legs = [[], [], []];
      for (const c of cells) for (let m = 0; m < 3; m++) {
        const v = rand() < 0.2 ? 0 : Math.fround(rand() * rand() * 50);
        d[base + 30 + m * nc + c] = v;
        legs[m].push(v);
      }
      p.push(...t, cells.length, ...cells, ...legs[0], ...legs[1], ...legs[2]);
    }
    dense.push(d);
    packed.push(Float32Array.from(p));
  }
  const cc = Array.from({ length: nc }, (_, c) => c).filter(() => rand() < 0.2);
  const commute = { cells: Uint32Array.from(cc), farRail: Float32Array.from(cc, () => Math.fround(rand() * 30)), farWalk: Float32Array.from(cc, () => Math.fround(rand() * 30)), farDrive: Float32Array.from(cc, () => Math.fround(rand() * 30)) };
  const start = [rand() * 1e4, rand() * 1e4, rand() * 1e4];
  const a = oldMerge(dense, kinds, nc, commute, start), b = mergeEveryday(packed, kinds, (k) => KEYS[k], commute, start);
  const fail = (what) => { throw new Error(`trial ${trial}: ${what} differs`); };
  if (!b) fail('null answer');
  a.timing.forEach((t, k) => { if (!same(t, b.timing[k])) fail(`timing of kind ${k}`); });
  for (const f of ['cells', 'farRail', 'farWalk', 'farDrive']) if (!same(a.far[f], b.far[f])) fail(`far ${f}`);
  if (!same(a.legs, b.legs)) fail('totals');
  if (a.shown.size !== b.shown.size) fail('shown purposes');
  for (const [k, v] of a.shown) {
    const u = b.shown.get(k);
    if (!u || !same(v.total, u.total)) fail(`${k} totals`);
    for (const f of ['cells', 'farRail', 'farWalk', 'farDrive']) if (!same(v.far[f], u.far[f])) fail(`${k} ${f}`);
  }
  cases++;
}
console.log(`${cases} random answers: packed merge identical to the dense one`);
