// T-1380: the station names kept up to date from changed stations (map/labelNames.ts `NameIndex`)
// against the whole list worked out from nothing (`labelGroups`), after random series of edits:
// stations added (end and middle of the list), removed, moved, renamed, built, upgraded, lines made,
// their stops changed, lines removed, the opening-day bunting, links, and lists given again with
// the same objects (as the client does) or with new equal ones. Every step: same names in the same
// order with the same fields, and the removes and inserts it reports turn the last list into this one.
//   node trainworld/checks/t1380-labels.mjs [seeds]      (from the maps root)
import { loadApp } from './lib.mjs';

const { NameIndex, labelGroups, linesAt, ComplexBuilder, buildComplexes } = await loadApp(
  "export { NameIndex, labelGroups, linesAt } from '@app/src/map/labelNames.ts'; export { ComplexBuilder } from '@app/src/game/complexBuilder.ts'; export { buildComplexes } from '@app/src/game/complexes.ts';");

const SEEDS = Number(process.argv[2] ?? 30);
let fails = 0, steps = 0, patched = 0, fulls = 0, opsTotal = 0;
const NAMES = ['Central', 'Park', 'North', 'Harbour', 'Central', 'Old Town', 'Airport', 'Mill', 'Park', 'Bridge', 'Court', 'Quay'];

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

const FIELDS = ['name', 'lng', 'lat', 'heading', 'built', 'park', 'rib', 'lines'];
function diff(list, want) {
  if (list.length !== want.length) return `lengths ${list.length}/${want.length}`;
  for (let i = 0; i < want.length; i++) for (const f of FIELDS) if (list[i][f] !== want[i][f]) return `name ${i} ${f}: ${list[i][f]} vs ${want[i][f]}`;
  return null;
}

for (let seed = 1; seed <= SEEDS; seed++) {
  const rnd = rng(seed * 104729);
  const kAt = () => 1;
  let nextId = 1, nextLine = 1;
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const make = () => {
    const cl = Math.floor(rnd() * 10), x = cl * 3000 + (rnd() - 0.5) * 500, y = (cl % 3) * 2500 + (rnd() - 0.5) * 500;
    const id = String(nextId++);
    return { id, num: nextId, name: pick(NAMES), lng: 100 + x / 1e5, lat: 30 + y / 1e5, x, y, level: 0, length: 80 + 20 * Math.floor(rnd() * 8), built: rnd() < 0.7, upgrades: rnd() < 0.15 ? 1 : 0, heading: rnd() * 6.28 };
  };
  let st = Array.from({ length: 120 + Math.floor(rnd() * 200) }, make);
  let lines = [];
  const mkLine = () => {
    const n = 2 + Math.floor(rnd() * 6), stops = [];
    for (let i = 0; i < n; i++) stops.push(pick(st).id);
    return { id: String(nextLine++), num: nextLine, letter: 'A', name: 'L', colour: '#000000', stops, tph: { high: 1, medium: 1, low: 1 } };
  };
  for (let i = 0; i < 12; i++) lines.push(mkLine());
  let rib = null;
  const cxb = new ComplexBuilder(), idx = new NameIndex();
  let mirror = [];
  for (let step = 0; step < 150; step++) {
    const batch = rnd() < 0.2 ? 1 + Math.floor(rnd() * 4) : 1;
    let stationsSame = rnd() < 0.15, linesSame = rnd() < 0.2;
    for (let b = 0; b < batch && !stationsSame; b++) {
      const q = rnd();
      if (q < 0.15) st = [...st, make()];
      else if (q < 0.22) { const i = Math.floor(rnd() * (st.length + 1)); st = [...st.slice(0, i), make(), ...st.slice(i)]; }
      else if (q < 0.34 && st.length > 20) { const i = Math.floor(rnd() * st.length); st = st.filter((_, j) => j !== i); }
      else if (q < 0.46) { const i = Math.floor(rnd() * st.length); const dx = (rnd() - 0.5) * 150, dy = (rnd() - 0.5) * 150; st = st.map((s, j) => j === i ? { ...s, x: s.x + dx, y: s.y + dy, lng: s.lng + dx / 1e5, lat: s.lat + dy / 1e5 } : s); }
      else if (q < 0.54) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, name: pick(NAMES) } : s); }
      else if (q < 0.62) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, built: !s.built } : s); }
      else if (q < 0.68) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, upgrades: s.upgrades ? 0 : 1 } : s); }
      else if (q < 0.74) { const i = Math.floor(rnd() * st.length); st = st.map((s, j) => j === i ? { ...s, length: 80 + 20 * Math.floor(rnd() * 8), heading: rnd() * 6.28 } : s); }
      else if (q < 0.8) { st = st.map((s) => ({ ...s })); } // equal objects
      else if (q < 0.83) { const m = Math.min(st.length - 1, 1 + Math.floor(rnd() * 30)); st = st.filter((_, j) => j >= m); }
      else if (q < 0.9) { const i = Math.floor(rnd() * st.length); const to = pick(st); st = st.map((s, j) => j === i ? { ...s, x: to.x + (rnd() - 0.5) * 40, y: to.y + (rnd() - 0.5) * 40 } : s); } // onto another station
    }
    for (let b = 0; b < batch && !linesSame; b++) {
      const q = rnd();
      if (q < 0.15) lines = [...lines, mkLine()];
      else if (q < 0.3 && lines.length > 2) { const i = Math.floor(rnd() * lines.length); lines = lines.filter((_, j) => j !== i); }
      else if (q < 0.75) { const i = Math.floor(rnd() * lines.length); lines = lines.map((l, j) => j === i ? { ...l, stops: rnd() < 0.5 ? [...l.stops, pick(st).id] : l.stops.slice(0, -1).length >= 2 ? l.stops.slice(0, -1) : l.stops } : l); }
      else if (q < 0.85) { lines = lines.map((l) => ({ ...l, stops: [...l.stops] })); } // equal objects
      else { const i = Math.floor(rnd() * lines.length); lines = lines.map((l, j) => j === i ? { ...l, stops: l.stops.map((s) => (rnd() < 0.3 ? pick(st).id : s)) } : l); }
    }
    if (rnd() < 0.1) rib = rib ? null : new Map(st.filter(() => rnd() < 0.05).map((s) => [s.id, 'linear-gradient(red,red)']));
    else if (rib && rnd() < 0.1) rib = new Map(rib);
    // stations or lines named by a stop may be gone: stops are ids, not checked against the stations, as in the app
    const useSt = st, useLines = lines;
    const cx = cxb.update(useSt, [], kAt, 'flat');
    const plan = idx.plan(useSt, useLines, cx, rib, 1);
    let patch = { full: false, ops: [] };
    if (plan) patch = idx.apply(plan, cx, rib);
    steps++;
    if (plan) { if (patch.full) fulls++; else { patched++; opsTotal += patch.ops.length; } }
    const want = labelGroups(useSt, linesAt(useLines), buildComplexes(useSt, [], kAt), rib);
    let d = diff(idx.list, want);
    if (!d) {
      if (patch.full) mirror = idx.list.map((g) => g.uid);
      else {
        for (const op of patch.ops) { if (op.kind === 'remove') mirror.splice(op.at, 1); else mirror.splice(op.at, 0, op.group.uid); }
      }
      if (mirror.join() !== idx.list.map((g) => g.uid).join()) d = 'ops do not turn the last list into this one';
    }
    if (d) { fails++; console.log(`FAIL seed ${seed} step ${step} (${useSt.length} stations, ${useLines.length} lines): ${d}`); mirror = idx.list.map((g) => g.uid); if (fails > 8) break; }
  }
  if (fails > 8) break;
}
console.log(`${fails ? 'FAIL' : 'PASS'}  t1380-labels: ${steps} steps over ${SEEDS} seeds, ${fails} differ; ${patched} patched (${(opsTotal / Math.max(1, patched)).toFixed(1)} ops on average), ${fulls} built whole`);
process.exit(fails ? 1 : 0);
