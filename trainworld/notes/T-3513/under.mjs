// self time of every function under frames matching a pattern: node under.mjs <regex> a.cpuprofile ...
import { readFileSync } from 'node:fs';
const NM = new Map(JSON.parse(readFileSync(process.env.NAMES, 'utf8')));
const clean = (n) => { const m = n.match(/^wasm-function\[(\d+)\]/); const x = m ? NM.get(+m[1]) ?? n : n; return x.replace(/::h[0-9a-f]{16}$/, '').replace(/[a-z_]+\[[0-9a-f]{16}\]::/g, '').slice(0, 110); };
const pat = new RegExp(process.argv[2]);
const agg = new Map(); let total = 0;
for (const f of process.argv.slice(3)) {
  const prof = JSON.parse(readFileSync(f, 'utf8'));
  const byId = new Map(prof.nodes.map((x) => [x.id, x]));
  const parent = new Map();
  for (const x of prof.nodes) for (const c of x.children ?? []) parent.set(c, x.id);
  const selfUs = new Map();
  for (let i = 0; i < prof.samples.length; i++) selfUs.set(prof.samples[i], (selfUs.get(prof.samples[i]) ?? 0) + (prof.timeDeltas[i + 1] ?? 0));
  for (const [id, us] of selfUs) {
    let under = false;
    for (let p = id; p !== undefined; p = parent.get(p)) if (pat.test(clean(byId.get(p).callFrame.functionName))) { under = true; break; }
    if (!under) continue;
    const n = clean(byId.get(id).callFrame.functionName);
    total += us; agg.set(n, (agg.get(n) ?? 0) + us);
  }
}
console.log(`under /${process.argv[2]}/: ${(total / 1e6).toFixed(1)} s`);
for (const [n, us] of [...agg].sort((a, b) => b[1] - a[1]).slice(0, 25)) console.log(`${(us / 1e6).toFixed(1).padStart(7)} s ${(100 * us / total).toFixed(1).padStart(5)}%  ${n}`);
