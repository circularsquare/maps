// T-1380: the complexes kept up to date from changed stations (game/complexBuilder.ts) against a
// full build (game/complexes.ts `buildComplexes`), after random series of edits: stations added
// (at the end and in the middle of the list), removed, moved by metres or far, resized, links made
// and undone, a second builder for the same list with no links, a change of frame. Every step must
// give the same numbering, not only the same grouping.
//   node trainworld/checks/t1380-complexes.mjs [seeds]      (from the maps root)
import { loadApp } from './lib.mjs';

const { ComplexBuilder, buildComplexes } = await loadApp(
  "export { ComplexBuilder } from '@app/src/game/complexBuilder.ts'; export { buildComplexes } from '@app/src/game/complexes.ts';");

const SEEDS = Number(process.argv[2] ?? 30);
let fails = 0, steps = 0, worked = 0, workedN = 0;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

function same(a, b) {
  if (a.members.length !== b.members.length || a.of.size !== b.of.size) return 'sizes ' + a.members.length + '/' + b.members.length;
  for (const [id, c] of b.of) if (a.of.get(id) !== c) return `of ${id}: ${a.of.get(id)} vs ${c}`;
  for (let c = 0; c < b.members.length; c++) if (a.members[c].join() !== b.members[c].join()) return `members ${c}`;
  // the Map's order is part of what consumers see
  if ([...a.of.keys()].join() !== [...b.of.keys()].join()) return 'of order';
  return null;
}

for (let seed = 1; seed <= SEEDS; seed++) {
  const rnd = rng(seed * 7919);
  const merc = seed % 3 === 0;
  const kAt = (y) => (merc ? Math.cosh(y / 6378137) : 1);
  const key = merc ? 'merc' : 'flat';
  // stations scattered in a few clusters so that many are within 100 m or platform reach of others
  let nextId = 1;
  const make = () => {
    const cl = Math.floor(rnd() * 12);
    return { id: String(nextId++), x: cl * 3000 + (rnd() - 0.5) * 700, y: (cl % 4) * 2500 + (rnd() - 0.5) * 700, heading: rnd() * Math.PI * 2, length: 60 + 20 * Math.floor(rnd() * 18) };
  };
  let st = Array.from({ length: 150 + Math.floor(rnd() * 400) }, make);
  let links = [];
  const full = new ComplexBuilder(), plain = new ComplexBuilder();
  const stepsN = 120;
  for (let step = 0; step < stepsN; step++) {
    const r = rnd();
    const batch = rnd() < 0.2 ? 1 + Math.floor(rnd() * 5) : 1;
    for (let b = 0; b < batch; b++) {
      const q = rnd();
      if (q < 0.18) st = [...st, make()];
      else if (q < 0.26) { const i = Math.floor(rnd() * (st.length + 1)); st = [...st.slice(0, i), make(), ...st.slice(i)]; }
      else if (q < 0.4 && st.length > 20) { const i = Math.floor(rnd() * st.length); st = st.filter((_, j) => j !== i); }
      else if (q < 0.55) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, x: s.x + (rnd() - 0.5) * 120, y: s.y + (rnd() - 0.5) * 120 } : s); }
      else if (q < 0.62) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, x: rnd() * 36000, y: rnd() * 10000 } : s); }
      else if (q < 0.7) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, length: 60 + 20 * Math.floor(rnd() * 18), heading: rnd() * Math.PI * 2 } : s); }
      else if (q < 0.8 && st.length > 2) { const a = st[Math.floor(rnd() * st.length)].id, c = st[Math.floor(rnd() * st.length)].id; if (a !== c) links = [...links, [a, c]]; }
      else if (q < 0.86 && links.length) links = links.filter((_, j) => j !== Math.floor(rnd() * links.length));
      else if (q < 0.9) { /* nothing changes */ }
      else if (q < 0.93) { const i = Math.floor(rnd() * st.length); links = [...links, [st[i].id, String(nextId + 5)]]; } // a link to a station not there yet
      else if (q < 0.96) { st = st.map((s) => ({ ...s })); } // new objects, same values
      else { const m = Math.min(st.length - 1, 1 + Math.floor(rnd() * 30)); st = st.filter((_, j) => j >= m); } // many removed at once
    }
    // fresh objects each step, as the world's are
    const input = st.map((s) => ({ ...s }));
    const got = full.update(input, links, kAt, key), want = buildComplexes(input, links, kAt);
    const got2 = plain.update(input, [], kAt, key), want2 = buildComplexes(input, [], kAt);
    steps++;
    if (full.lastWorked >= 0) { worked += full.lastWorked; workedN++; }
    for (const [g, w, name] of [[got, want, 'links'], [got2, want2, 'plain']]) {
      const d = same(g, w);
      if (d) { fails++; if (fails < 10) console.log(`FAIL seed ${seed} step ${step} (${name}, ${input.length} stations, ${links.length} links): ${d}`); }
    }
    if (fails > 10) break;
  }
}
console.log(`${fails ? 'FAIL' : 'PASS'}  t1380-complexes: ${steps} steps over ${SEEDS} seeds, ${fails} differ; patched steps changed ${workedN ? (worked / workedN).toFixed(1) : 0} stations on average`);
process.exit(fails ? 1 : 0);
