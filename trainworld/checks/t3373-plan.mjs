// T-3373 B: unit tests of the holding planner (app/src/workers/demandPlan.ts), loaded in Node (22.18+ strips
// its types). No browser.   node trainworld/checks/t3373-plan.mjs
import { planHolding, everywhere, measure, spanOf } from '../app/src/workers/demandPlan.ts';

let fails = 0;
const ok = (cond, what) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${what}`); if (!cond) fails++; };
const P = [126, 125, 120, 120, 67];
const total = P.reduce((s, x) => s + x, 0);
const unit = (id, share, mem) => ({ id, cost: P.map((c) => (c * share * 1000) / total), mem });
const two = (w) => (w === 0 ? [0, 3, 4] : [1, 2]);
const byIndex = (n) => (w) => [0, 1, 2, 3, 4].filter((q) => q % n === w);
const max = (a) => Math.max(...a);

function sound(plan, units, n, room) {
  for (const u of units) {
    const hs = plan.holders[u.id], sv = plan.solver[u.id];
    if (!hs?.length || new Set(hs).size !== hs.length || hs.some((w) => w < 0 || w >= n)) return `${u.id}: bad holders ${hs}`;
    if (sv.length !== u.cost.length || sv.some((w) => !hs.includes(w))) return `${u.id}: a period solved off its holders ${sv} / ${hs}`;
  }
  const m = measure(units, n, plan.holders, plan.solver, room);
  if (m.load.some((x, w) => Math.abs(x - plan.load[w]) > 1e-6) || m.mem.some((x, w) => x !== plan.mem[w])) return 'loads or memory not its own';
  return '';
}

// a many-zone save: a market 30% of the passes, 170 cities of falling size (all-China's shape)
const china = [unit('market', 0.3, 450), ...Array.from({ length: 170 }, (_, i) => unit('z' + String(i).padStart(3, '0'), 0.7 * (1 / (i + 4)) / Array.from({ length: 170 }, (_, k) => 1 / (k + 4)).reduce((s, x) => s + x, 0), 3 + 300 / (i + 4)))];
// New York's shape: one city 3/4 of the work
const nyc = [unit('nyc', 0.75, 130), unit('market', 0.2, 185), unit('nhv', 0.03, 15), unit('hfd', 0.02, 15)];

for (const [name, units] of [['many zones', china], ['one dominant city', nyc]]) {
  for (const n of [1, 2, 3]) {
    const room = 1e12;
    const p = planHolding({ units, workers: n, room }, n === 2 ? two : byIndex(n));
    ok(!sound(p, units, n, room), `${name}, ${n} workers: every period of every unit solved by exactly one of its holders ${sound(p, units, n, room)}`);
    const today = everywhere(units, n, n === 2 ? two : byIndex(n));
    ok(max(p.load) <= max(today.load) + 1e-9, `${name}, ${n} workers: never slower than today's pool (${max(p.load).toFixed(0)} against ${max(today.load).toFixed(0)})`);
  }
}

{
  const p = planHolding({ units: china, workers: 1, room: 1e12 });
  ok(Object.values(p.holders).every((h) => h.length === 1 && h[0] === 0), 'one worker: everything held once on worker 0 (stage A)');
  const all = everywhere(china, 3, byIndex(3));
  ok(Object.values(all.holders).every((h) => h.length === 3) && !sound(all, china, 3, Infinity), 'everything everywhere is a plan too (today\'s pool)');
}

{
  const room = 1100;
  const p = planHolding({ units: china, workers: 2, room }, two);
  ok(p.fits && p.mem.every((m) => m <= room), `many zones, 2 workers within ${room} each: ${p.mem.map((m) => m.toFixed(0)).join(', ')}`);
  ok(p.holders.market.length === 1, 'many zones, 2 workers: the market (30%) held once');
  const even = (p.load[0] + p.load[1]) / 2;
  ok(max(p.load) <= even * 1.15 + 1e-9, `many zones, 2 workers: busiest within 15% of the even share (${(max(p.load) / even).toFixed(3)})`);
}

{
  const p = planHolding({ units: nyc, workers: 3, room: 1e12 }, byIndex(3));
  ok(p.holders.nyc.length >= 2 && new Set(p.solver.nyc).size >= 2, `one dominant city, 3 workers: the city copied with its periods split (${p.holders.nyc}, ${p.solver.nyc})`);
  const small = [unit('nyc', 0.75, 130), unit('market', 0.2, 140), unit('nhv', 0.03, 15), unit('hfd', 0.02, 15)];
  const tight = planHolding({ units: small, workers: 3, room: 144 }, byIndex(3));
  ok(tight.fits && tight.holders.nyc.length === 1, 'one dominant city, 3 workers, no room for a copy: held once, within room');
}

{
  const a = planHolding({ units: china, workers: 3, room: 900 }, byIndex(3));
  const b = planHolding({ units: china.slice().reverse(), workers: 3, room: 900 }, byIndex(3));
  ok(JSON.stringify(a.holders) === JSON.stringify(Object.fromEntries(Object.keys(a.holders).map((k) => [k, b.holders[k]]))) && JSON.stringify(a.solver) === JSON.stringify(Object.fromEntries(Object.keys(a.solver).map((k) => [k, b.solver[k]]))), 'the same plan whatever order the units come in');
  const ties = [unit('b', 0.5, 1), unit('a', 0.5, 1)];
  const t = planHolding({ units: ties, workers: 2, room: 10 });
  ok(t.holders.a[0] === 0 && t.holders.b[0] === 1, 'equal units: ties broken by id, then worker index');
}

{
  // a market-heavy save whose small zones take longer to open than to solve: some go to the market's worker
  const zones = Array.from({ length: 10 }, (_, i) => ({ ...unit('z' + i, 0.04, 5), open: 60 }));
  const units = [{ ...unit('market', 0.6, 400), open: 15 }, ...zones];
  const plain = planHolding({ units: units.map(({ open, ...u }) => u), workers: 2, room: 1e12, over: Infinity });
  const p = planHolding({ units, workers: 2, room: 1e12, over: Infinity });
  const m = p.holders.market[0];
  ok(spanOf(units, p) < spanOf(units, plain) && zones.some((z) => p.holders[z.id][0] === m), `opening counted: zones join the market's worker, load ${spanOf(units, p).toFixed(0)} against ${spanOf(units, plain).toFixed(0)}`);
}

{
  const p = planHolding({ units: [unit('big', 1, 50)], workers: 2, room: 10 });
  ok(!p.fits, 'a unit past every worker\'s room: the plan says it does not fit');
}

console.log(fails ? `${fails} failed` : 'all passed');
process.exit(fails ? 1 : 0);
