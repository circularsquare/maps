// T-1380: the complexes kept up to date from changed stations (game/complexBuilder.ts) against a
// full build (game/complexes.ts `buildComplexes`), after random series of edits: stations added
// (at the end and in the middle of the list), removed, moved by metres or far, resized, links made
// and undone, pairs kept apart and let go (T-4048), a second builder for the same list with no links,
// a change of frame. Every step must give the same numbering, not only the same grouping. T-4300: a third builder is told
// only the stations that changed (`patch`), as the clock worker and the page tell theirs.
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
  if (a.members.size !== b.members.size || a.of.size !== b.of.size) return 'sizes ' + a.members.size + '/' + b.members.size;
  for (const [id, c] of b.of) if (a.of.get(id) !== c) return `of ${id}: ${a.of.get(id)} vs ${c}`;
  for (const [c, m] of b.members) if (a.members.get(c)?.join() !== m.join()) return `members ${c}`;
  return null;
}
const sameGeo = (p, q) => p.x === q.x && p.y === q.y && p.heading === q.heading && p.length === q.length && (p.line === q.line || (!!p.line && !!q.line && p.line.length === q.line.length && p.line.every((v, i) => v === q.line[i])));
/** the stations that differ from `was` to `now` (each as it is now, null where gone) */
function changesOf(was, now) {
  const before = new Map(was.map((s) => [s.id, s])), out = [];
  for (const s of now) { const w = before.get(s.id); if (!w || !sameGeo(w, s)) out.push([s.id, s]); before.delete(s.id); }
  for (const id of before.keys()) out.push([id, null]);
  return out;
}

for (let seed = 1; seed <= SEEDS; seed++) {
  const rnd = rng(seed * 7919);
  const merc = seed % 3 === 0;
  const kAt = (y) => (merc ? Math.cosh(y / 6378137) : 1);
  const key = merc ? 'merc' : 'flat';
  // stations scattered in a few clusters so that many are within 100 m or platform reach of others
  let nextId = 1;
  // T-1859: half the stations carry a drawn platform line, curved, some from the node (a dead end)
  const lined = (s) => {
    if (rnd() < 0.5) return s;
    const k = kAt(s.y), L = s.length * k, bend = (rnd() - 0.5) * 0.006, from = rnd() < 0.3 ? 0 : -L / 2, line = [];
    for (let i = 0; i < 7; i++) {
      const t = from + (L * i) / 6, a = s.heading + bend * t;
      line.push(Math.cos(a) * t, Math.sin(a) * t);
    }
    return { ...s, line: Float32Array.from(line) };
  };
  const make = () => {
    const cl = Math.floor(rnd() * 12);
    return lined({ id: String(nextId++), x: cl * 3000 + (rnd() - 0.5) * 700, y: (cl % 4) * 2500 + (rnd() - 0.5) * 700, heading: rnd() * Math.PI * 2, length: 60 + 20 * Math.floor(rnd() * 18) });
  };
  let st = Array.from({ length: 150 + Math.floor(rnd() * 400) }, make);
  let links = [], apart = [];
  const full = new ComplexBuilder(), plain = new ComplexBuilder(), told = new ComplexBuilder();
  let was = null;
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
      else if (q < 0.7) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? lined({ ...s, line: undefined, length: 60 + 20 * Math.floor(rnd() * 18), heading: rnd() * Math.PI * 2 }) : s); }
      else if (q < 0.8 && st.length > 2) { const a = st[Math.floor(rnd() * st.length)].id, c = st[Math.floor(rnd() * st.length)].id; if (a !== c) links = [...links, [a, c]]; }
      else if (q < 0.86 && links.length) links = links.filter((_, j) => j !== Math.floor(rnd() * links.length));
      else if (q < 0.9) { /* nothing changes */ }
      else if (q < 0.93) { const i = Math.floor(rnd() * st.length); links = [...links, [st[i].id, String(nextId + 5)]]; } // a link to a station not there yet
      else if (q < 0.96) { st = st.map((s) => ({ ...s })); } // new objects, same values
      else { const m = Math.min(st.length - 1, 1 + Math.floor(rnd() * 30)); st = st.filter((_, j) => j >= m); } // many removed at once
    }
    // T-4048: a station kept apart from its nearest, or a pair let go again
    const qa = rnd();
    if (qa < 0.15 && st.length > 2) {
      const s = st[Math.floor(rnd() * st.length)];
      const o = st.filter((x) => x !== s).sort((p, q) => Math.hypot(p.x - s.x, p.y - s.y) - Math.hypot(q.x - s.x, q.y - s.y))[0];
      apart = [...apart, [s.id, o.id]];
    } else if (qa < 0.22 && apart.length) apart = apart.filter((_, j) => j !== Math.floor(rnd() * apart.length));
    const ak = new Set(apart.map(([a, b]) => (a < b ? `${a}|${b}` : `${b}|${a}`)));
    // fresh objects each step, as the world's are
    const input = st.map((s) => ({ ...s }));
    const got = full.update(input, links, kAt, key, ak), want = buildComplexes(input, links, kAt, ak);
    const got2 = plain.update(input, [], kAt, key, ak), want2 = buildComplexes(input, [], kAt, ak);
    const got3 = was && told.ready(key) ? told.patch(changesOf(was, input), links, kAt, key, ak) : told.update(input, links, kAt, key, ak);
    was = input;
    steps++;
    if (full.lastWorked >= 0) { worked += full.lastWorked; workedN++; }
    for (const [g, w, name] of [[got, want, 'links'], [got2, want2, 'plain'], [got3, want, 'told']]) {
      const d = same(g, w);
      if (d) { fails++; if (fails < 10) console.log(`FAIL seed ${seed} step ${step} (${name}, ${input.length} stations, ${links.length} links): ${d}`); }
    }
    if (fails > 10) break;
  }
}
console.log(`${fails ? 'FAIL' : 'PASS'}  t1380-complexes: ${steps} steps over ${SEEDS} seeds, ${fails} differ; patched steps changed ${workedN ? (worked / workedN).toFixed(1) : 0} stations on average`);
process.exit(fails ? 1 : 0);
