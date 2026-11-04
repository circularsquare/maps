// T-1862: from tokyo.mjs's output, Tokyo's lines after the Ceuta line against the load's final (to
// the bit) and against a fresh load of the save with the line (lines with 10,000+ riders, 0.5%).
import { readFileSync } from 'node:fs';
const o = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const L = o.loadFinal.lines, A = o.afterLine.lines, F = o.fresh.lines;
const old = Object.keys(L);
let bitDiff = 0, worst = 0, worstId = null, big = 0, over = 0;
const off = [];
for (const id of old) {
  const a = A[id], l = L[id];
  if (!a) { bitDiff++; continue; }
  if (!Object.is(a.r, l.r) || a.b.length !== l.b.length || a.b.some((v, k) => !Object.is(v, l.b[k]))) bitDiff++;
  const f = F[id];
  if (!f || a.r < 10000) continue;
  big++;
  const rel = Math.abs(a.r - f.r) / f.r;
  if (rel > worst) { worst = rel; worstId = id; }
  if (rel > 0.005) { over++; off.push(`${id}: ${Math.round(a.r)} vs ${Math.round(f.r)} (${(100 * rel).toFixed(3)}%)`); }
}
const newIds = Object.keys(A).filter((id) => !L[id]);
console.log(JSON.stringify({ workers: o.workers, loadS: o.loadS, updMs: o.updMs, freshS: o.freshS, oldLines: old.length, newLines: newIds.map((id) => [id, Math.round(A[id].r), F[id] ? Math.round(F[id].r) : null]), notToTheBit: bitDiff, big10k: big, over05: over, worstPct: +(100 * worst).toFixed(4), worstId, errors: o.errors }));
for (const s of off.slice(0, 10)) console.log(s);
for (const x of o.log.filter((x) => x.kind === 'plan' || x.kind === 'done' || (x.kind === 'solve' && x.dir === '>'))) console.log(JSON.stringify(x));
for (const c of o.console) console.log(c);
