// T-1896: the gap a stop cannot cross, worked out by the track model (WASM) in Node.
//   node trainworld/checks/t1896-gap.mjs
// 1. A made-up track: a main line stopping 500 m short of the next stretch, a branch built up to
//    the main line but not joined, and a stub far off: each gap is where it should be.
// 2. The real save (New York, 109 lines): stations no track joins to a line (stop tier 3), PER_LINE
//    of them for each line, get their gap, at most MAX_MS each; the slowest is printed.
// Exits 1 on any failure.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import init, { TrackApi } from '../app/src/wasm/track/sim.js';

const MAX_MS = 200;
/** cut-off stations tried per line, spread over its list */
const PER_LINE = 40;
await init({ module_or_path: readFileSync(new URL('../app/src/wasm/track/sim_bg.wasm', import.meta.url)) });
let failed = 0;
const check = (ok, what) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${what}`); if (!ok) failed++; };

// ---- 1. made-up track, x east and y north in metres
{
  const api = new TrackApi(-73.985, 40.758, true);
  api.set_cash(1e9);
  const must = (r, what) => { if (r !== 0) throw new Error(`${what}: ${api.issue_kinds()}`); };
  const route = (x0, y0, x1, y1) => must(api.add_route(new Float64Array([0, 0, x0, y0, 0, 0, 0, x1, y1, 0]), new Float64Array(0), 2, false), 'route');
  const edges = () => { const e = api.edges_info(), out = []; for (let i = 0; i < e.length; i += 10) out.push(e[i]); return out; };
  const nodes = () => { const n = api.nodes_info(), out = []; for (let i = 0; i < n.length; i += 10) out.push({ id: n[i], x: n[i + 1], y: n[i + 2] }); return out; };
  const edgeAt = (x, y) => {
    let best = null, bd = Infinity;
    for (const e of edges()) {
      const p = api.edge_polyline(e, 0);
      for (let k = 0; k + 3 < p.length; k += 2) {
        const [ax, ay, bx, by] = [p[k], p[k + 1], p[k + 2], p[k + 3]];
        const t = Math.max(0, Math.min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 || 1)));
        const d = Math.hypot(ax + t * (bx - ax) - x, ay + t * (by - ay) - y);
        if (d < bd) [bd, best] = [d, e];
      }
    }
    return best;
  };
  const station = (x, y, name) => {
    must(api.add_station(2, edgeAt(x, y), x, y, 200, name), 'station ' + name);
    return nodes().reduce((b, n) => (Math.hypot(n.x - x, n.y - y) < Math.hypot(b.x - x, b.y - y) ? n : b)).id;
  };
  route(0, 0, 9000, 0);
  route(9500, 0, 15000, 0);
  route(5000, 400, 5000, 3000);
  route(30000, 8000, 32000, 8000);
  const a = station(1000, 0, 'A'), b = station(3000, 0, 'B');
  const far = station(12000, 0, 'C'), branch = station(5000, 2000, 'D'), stub = station(31000, 8000, 'E');
  const id = api.new_line_id();
  must(api.set_line_with_trains(id, new Uint32Array([a, b]), 0, 0, 0, 30, 180, 0, 0xff0000, 'probe'), 'line');
  const origin = api.frame_origin();
  // lng/lat back to metres near the origin, plenty for a metre's check over 30 km
  const R = 6371008.8, rad = Math.PI / 180;
  const metres = (lng, lat) => [(lng - origin[0]) * rad * R * Math.cos(origin[1] * rad), (lat - origin[1]) * rad * R];
  const gap = (n) => { const g = api.stop_gap(id, n); return g.length === 4 ? [...metres(g[0], g[1]), ...metres(g[2], g[3])] : null; };
  const near = (g, want, tol) => !!g && Math.hypot(g[0] - want[0], g[1] - want[1]) < tol && Math.hypot(g[2] - want[2], g[3] - want[3]) < tol;
  const show = (g) => (g ? g.map((v) => Math.round(v)).join(',') : 'none');
  check(near(gap(far), [9000, 0, 9500, 0], 20), `end to end: the gap is between the two track ends (${show(gap(far))})`);
  check(near(gap(branch), [5000, 0, 5000, 400], 20), `a branch short of the main line: from the main line to its end (${show(gap(branch))})`);
  check(near(gap(stub), [9000, 0, 30000, 8000], 60), `a stub far off: from the nearest end (${show(gap(stub))})`);
  check(gap(b) === null, 'a stop on the line has no gap');
}

// ---- 2. the real save
{
  const raw = gunzipSync(readFileSync(new URL('../T-007-real-nyc.save', import.meta.url)));
  const n = raw.readUInt32LE(4);
  const api = new TrackApi(-73.985, 40.758, true);
  if (!api.load(raw.subarray(8 + n))) throw new Error('save did not load');
  const li = api.lines_info();
  const lines = [];
  for (let i = 0; i < li.length; i += 22) lines.push(li[i]);
  let asked = 0, found = 0, worst = 0, worstAt = '', total = 0;
  for (const line of lines) {
    const t = api.stop_tiers(line), cut = [];
    for (let i = 0; i + 2 < t.length; i += 3) if (t[i + 1] === 3) cut.push(t[i]);
    const step = Math.max(1, Math.ceil(cut.length / PER_LINE));
    for (let k = 0; k < cut.length; k += step) {
      const t0 = performance.now();
      const g = api.stop_gap(line, cut[k]);
      const ms = performance.now() - t0;
      asked++;
      total += ms;
      if (g.length === 4) found++;
      if (ms > worst) [worst, worstAt] = [ms, `line ${line}, station ${cut[k]}`];
    }
  }
  console.log(`real save: ${lines.length} lines, ${asked} stations cut off from a line, ${found} with a gap; ${(total / Math.max(1, asked)).toFixed(1)} ms each on average, slowest ${worst.toFixed(1)} ms (${worstAt})`);
  check(asked === 0 || found > 0, 'the real save: stations cut off from a line get a gap');
  check(worst <= MAX_MS, `the real save: every gap within ${MAX_MS} ms`);
}
process.exit(failed ? 1 : 0);
