// T-1850: what changed in the network the demand workers got between two solves, line by line (by
// id), outside the edited line: the inputs a far cluster's `sig` hashes.
import { readFileSync } from 'node:fs';
const nets = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const [va, vb] = [process.argv[3], process.argv[4]];
const A = nets[va], B = nets[vb];
const P = 5;
function lines(x) {
  const n = x.net, out = new Map();
  let g = 0;
  const S2 = 2 * n.stops.length;
  x.ids.lines.forEach((id, l) => {
    const m = n.lineN[l];
    const span = n.span && n.span.length === P * S2 ? Array.from({ length: P }, (_, q) => n.span.slice(q * S2 + 2 * g, q * S2 + 2 * g + 2 * m)) : null;
    out.set(id, {
      stops: n.stops.slice(g, g + m).map((s) => x.ids.stations[s]),
      times: n.times.slice(6 * g, 6 * (g + m)), tph: n.tph.slice(3 * l, 3 * l + 3), cars: n.cars[l],
      hopKm: n.hopKm.slice(2 * g, 2 * (g + m)), span, fareW: n.fareW ? n.fareW.slice(2 * g, 2 * g + 2 * m) : null,
      catchN: n.catchN ? n.catchN.slice(g, g + m) : null,
    });
    g += m;
  });
  return out;
}
const la = lines(A), lb = lines(B);
console.log(`v${va}: ${la.size} lines, ${A.net.stXY.length / 2} stations; v${vb}: ${lb.size} lines, ${B.net.stXY.length / 2} stations`);
console.log('fare same:', JSON.stringify(A.net.fare) === JSON.stringify(B.net.fare), 'frame same:', JSON.stringify(A.net.frame) === JSON.stringify(B.net.frame));
const maxd = (x, y) => { if (!x || !y) return x === y ? 0 : Infinity; if (x.length !== y.length) return Infinity; let d = 0; for (let i = 0; i < x.length; i++) d = Math.max(d, Math.abs(x[i] - y[i])); return d; };
const rel = (x, y) => { let d = 0; for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) d = Math.max(d, Math.abs(x[i] - y[i]) / Math.max(1e-9, Math.abs(x[i]))); return d; };
let changed = 0;
const kinds = {};
for (const [id, a] of la) {
  const b = lb.get(id);
  if (!b) { console.log('line gone', id); continue; }
  const why = [];
  if (a.stops.join() !== b.stops.join()) why.push('stops');
  if (maxd(a.times, b.times)) why.push(`times max ${maxd(a.times, b.times).toFixed(3)} s`);
  if (maxd(a.tph, b.tph)) why.push(`tph ${a.tph.map((v) => v.toFixed(3))} -> ${b.tph.map((v) => v.toFixed(3))} (${(100 * rel(a.tph, b.tph)).toFixed(2)}%)`);
  if (a.cars !== b.cars) why.push(`cars ${a.cars} -> ${b.cars}`);
  if (maxd(a.hopKm, b.hopKm)) why.push('hopKm');
  if (a.span || b.span) { let d = 0; for (let q = 0; q < P; q++) d = Math.max(d, maxd(a.span?.[q], b.span?.[q])); if (d) why.push(`span max ${d.toFixed(3)}`); }
  if (maxd(a.fareW, b.fareW)) why.push('fareW');
  if (maxd(a.catchN, b.catchN)) why.push('catchN');
  if (why.length) { changed++; for (const w of why) { const k = w.split(' ')[0]; kinds[k] = (kinds[k] ?? 0) + 1; } if (changed <= 25) console.log(id, why.join('; ')); }
}
for (const id of lb.keys()) if (!la.has(id)) console.log('new line', id);
console.log(`${changed} of ${la.size} old lines changed; by input: ${JSON.stringify(kinds)}`);
// stations by id: position, complex grouping
const st = (x) => new Map(x.ids.stations.map((id, s) => [id, [x.net.stXY[2 * s], x.net.stXY[2 * s + 1], x.net.complex?.[s]]]));
const sa = st(A), sb = st(B);
let moved = 0, cx = 0;
const cxA = new Map(), cxB = new Map();
for (const [id, [x, y, c]] of sa) { const b = sb.get(id); if (!b) continue; if (b[0] !== x || b[1] !== y) moved++; }
console.log(`stations moved: ${moved}; station order same for the old ones: ${A.ids.stations.join() === B.ids.stations.slice(0, A.ids.stations.length).join()}; lines order same: ${A.ids.lines.join() === B.ids.lines.slice(0, A.ids.lines.length).join()}`);
console.log('ups', !!A.net.ups, !!B.net.ups, 'passages', A.net.passages?.length, B.net.passages?.length);
