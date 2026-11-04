// T-2291: the automatic line colour avoids a near-duplicate of a line it runs beside (game/colourPick.ts).
//   node trainworld/checks/t2291-line-colour.mjs      (from the maps root; a few seconds, no browser)
// Loads the pure pick code (t2291-load.mjs), then: the palette's own spacing; `besideColours` on a
// shared station, a shared edge, a stop 250 m away and one 1 km away; a crowded city (every palette colour
// in use nearby, one of them Anita's burgundy line) where the old rule gives red beside the burgundy and
// the new one does not; two lines through the same stations, many times over.
import { rmSync } from 'node:fs';
import { dir, loadGame as load } from './t2291-load.mjs';

const { LINE_PALETTE } = await load('palette');
const { pickColour, deltaE, besideColours, settledColour, APART, BESIDE_M } = await load('colourPick');

let failed = 0;
const report = (name, ok, detail = '') => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

// 1. the palette's own spacing
let minDE = Infinity, pair = '';
for (let i = 0; i < LINE_PALETTE.length; i++) for (let j = i + 1; j < LINE_PALETTE.length; j++) {
  const d = deltaE(LINE_PALETTE[i], LINE_PALETTE[j]);
  if (d < minDE) { minDE = d; pair = LINE_PALETTE[i] + ' ' + LINE_PALETTE[j]; }
}
report('palette: no pair closer than APART (CIEDE2000)', minDE >= APART, `closest ${pair} at ${minDE.toFixed(1)}, APART ${APART}`);

// 2. who is beside
const M = 111_320; // m per degree of latitude
const stn = { a: [-73.9, 40.8], b: [-73.89, 40.8], c: [-73.88, 40.8], near: [-73.89, 40.8 + 250 / M], far: [-73.89, 40.8 + 1000 / M], x: [-73.5, 40.9] };
const at = (s) => stn[s] ?? null;
const lines = [
  { id: '1', colour: '#111111', stops: ['a', 'b', 'c'] }, // the new line
  { id: '2', colour: '#222222', stops: ['b', 'x'] }, // shares a station
  { id: '3', colour: '#333333', stops: ['near', 'x'] }, // a stop 250 m from one of ours
  { id: '4', colour: '#444444', stops: ['far', 'x'] }, // 1 km away
  { id: '5', colour: '#555555', stops: ['x'] }, // shares track only
  { id: '6', colour: '#666666', stops: ['x'] }, // nothing
];
const beside = besideColours(lines[0], lines, [['1', '5'], ['6']], at).sort();
report('beside: shared station, a stop within BESIDE_M, shared edge; not 1 km away', beside.join() === '#222222,#333333,#555555', `${beside.join()} (BESIDE_M ${BESIDE_M})`);

// 3. a crowded city with a burgundy line beside the new one: every other palette colour in use nearby
const rnd = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
const BURGUNDY = '#b5122e';
const crowd = [BURGUNDY, ...LINE_PALETTE.filter((c) => c !== '#d7263d')];
let oldBad = 0, newWorst = Infinity, newBad = 0;
for (let s = 1; s <= 300; s++) {
  const o = pickColour(crowd, 20, undefined, rnd(s));
  const nw = settledColour(o, [BURGUNDY], 20, rnd(s + 1000));
  const dOld = deltaE(BURGUNDY, o), dNew = deltaE(BURGUNDY, nw);
  if (dOld < APART) oldBad++;
  newWorst = Math.min(newWorst, dNew); if (dNew < APART) newBad++;
}
const oldPick = pickColour(crowd, 20, undefined, rnd(1));
report('crowded city: the old rule picks a near-duplicate of the burgundy line', deltaE(BURGUNDY, oldPick) < APART, `old pick ${oldPick}, ${deltaE(BURGUNDY, oldPick).toFixed(1)} from burgundy`);
report('crowded city: the beside rule keeps every pick at least APART from it', newBad === 0 && oldBad > 0, `300 runs, old rule too close ${oldBad}, new ${newBad}, smallest new gap ${newWorst.toFixed(1)}`);

// 4. two lines through the same stations: the first takes what the old rule gives in a crowded city, the second settles beside it
let identical = 0, tooClose = 0, minGap = Infinity;
for (let s = 1; s <= 500; s++) {
  const first = pickColour(LINE_PALETTE, 20, undefined, rnd(s)); // every colour nearby: the old rule is a coin toss
  const second = pickColour(LINE_PALETTE, 20, undefined, rnd(s * 7 + 3));
  if (second === first) identical++;
  const settled = settledColour(second, [first], 20, rnd(s * 13 + 5));
  const gap = deltaE(first, settled);
  minGap = Math.min(minGap, gap);
  if (gap < APART) tooClose++;
}
report('two lines through the same stations: old rule gives identical colours sometimes', identical > 0, `${identical} of 500`);
report('two lines through the same stations: the settled colours are at least APART', tooClose === 0, `smallest gap ${minGap.toFixed(1)} over 500 pairs`);

// 5. nothing beside, or already clear: the colour stays
report('nothing beside: the colour stays', settledColour('#d7263d', [], 20) === '#d7263d');
report('already clear of what is beside: the colour stays', settledColour('#1f6fd1', ['#d7263d', '#1a9a4a'], 20) === '#1f6fd1');

rmSync(dir, { recursive: true, force: true });
console.log(failed ? `FAIL  ${failed} check(s) failed` : 'PASS  all');
process.exit(failed ? 1 : 0);
