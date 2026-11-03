// T-2159: the network diagram's graph (app/src/diagram/graph.ts) on the real New York save, Tokyo's
// real network and all of Japan's: its counts near notes/T-2146.md section 1's table (the Python
// prototype's), the 2 and 3 on the 1's edges from 96 St to Times Sq, New York's graph in under 50 ms
// (corridors.ts' beside table, which the bands build anyway, timed apart).
//   node trainworld/checks/t2159-diagram-graph.mjs [nyc,tyo,jpn]      (about a minute with jpn)
import { openSave } from './diagram-saves.mjs';

const WANT = { nyc: { complexes: 855, lines: 109, edges: 968 }, tyo: { complexes: 1447, lines: 133, edges: 1654 }, jpn: { complexes: 9106, lines: 709, edges: 9561 } };
const which = (process.argv[2] ?? 'nyc,tyo,jpn').split(',');
let failed = 0;
const line = (ok, name, info) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${info ? ': ' + info : ''}`); };

for (const name of which) {
  const s = openSave(name);
  // the first call compiles the code; the polylines' arcs and boxes are kept by polyline after, so a
  // build from new copies of every polyline is a game's first build (the timed one), and the calls
  // after it a rebuild with the edges unchanged
  // after it; the fastest of five (the machine is shared with other sessions' jobs)
  s.graph();
  const colds = [];
  for (let r = 0; r < 5; r++) {
    for (const [id, p] of s.pts) s.pts.set(id, p.slice());
    globalThis.gc?.();
    colds.push(s.graph());
  }
  const cold = colds.reduce((a, b) => (b.ms < a.ms ? b : a));
  const runs = [1, 2, 3].map(() => s.graph());
  const { g } = runs[0];
  const ms = cold.ms;
  const w = WANT[name];
  const near = (a, b) => Math.abs(a - b) <= 0.1 * b;
  line(near(g.key.length, w.complexes) && near(g.ea.length, w.edges) && near(g.lines.length, w.lines),
    `${name}: counts near the prototype's`, `${s.stations} stations, ${g.key.length} complexes on lines (${w.complexes}), ${g.lines.length} lines (${w.lines}), ${g.ea.length} edges (${w.edges}); graph ${cold.ms.toFixed(1)} ms first, ${Math.min(...runs.map((r) => r.ms)).toFixed(1)} ms again; beside table ${cold.besideMs.toFixed(0)} ms first, ${runs[0].besideMs.toFixed(1)} ms when kept`);
  if (name === 'nyc') {
    line(ms < 50, 'nyc: the graph in under 50 ms', `${ms.toFixed(1)} ms`);
    const byName = (n) => g.lines.find((l, i) => s.lines.find((x) => x.id === l.id)?.name.replace(/^subway /, '') === n);
    const nodeName = (k) => s.name(g.key[k]);
    const one = byName('1');
    const ix = (l, pre) => [...l.chain].findIndex((k) => nodeName(k).startsWith(pre));
    const i = ix(one, '96 St'), j = ix(one, 'Times Sq');
    const stretch = [...one.chain].slice(Math.min(i, j), Math.max(i, j) + 1);
    const edgeOf = (a, b) => { for (let e = 0; e < g.ea.length; e++) if (g.ea[e] === Math.min(a, b) && g.eb[e] === Math.max(a, b)) return e; return -1; };
    const res = [];
    for (const nm of ['2', '3']) {
      const l = byName(nm), li = g.lines.indexOf(l);
      let shared = 0;
      for (let k = 1; k < stretch.length; k++) {
        const e = edgeOf(stretch[k - 1], stretch[k]);
        if (e >= 0 && [...g.on.subarray(g.first[e], g.first[e + 1])].includes(li)) shared++;
      }
      res.push([nm, shared, stretch.length - 1]);
    }
    line(i >= 0 && j >= 0 && res.every(([, a, b]) => a === b && b > 0), 'nyc: the 2 and 3 share the 1\'s edges from 96 St to Times Sq',
      `${stretch.map(nodeName).map((n) => n.split(' / ')[0]).join(', ')}; ${res.map(([n, a, b]) => `${n} on ${a} of ${b}`).join(', ')}`);
  }
  s.api.free();
}
process.exit(failed ? 1 : 0);
