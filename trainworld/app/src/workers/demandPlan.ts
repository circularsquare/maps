/**
 * T-3373 B: which solving workers hold which populations, and which holder solves each period
 * (notes/T-3373-B-plan.md "What is copied today, and what B changes"). Pure, with no imports, so a
 * check loads it in Node (`checks/t3373-plan.mjs`). The game uses `Holding` under `?split=1`; `planHolding` not yet.
 *
 * A unit is what is held together: a city's commutes with its everyday layer (they share their
 * cells), or the market (every cluster's part shares one composition). Today's pool is the plan
 * that holds every unit on every worker (`everywhere`); stage A's is one worker.
 */

/** the market's unit: every zone's market pack, one composition */
export const MARKET_UNIT = "market";
/** the market's share of a load's passes before any are measured (stage 1 measured 23-65%) */
const MARKET_SHARE = 0.4;

/**
 * T-3373 B stage 3 (`?split=1`): the page's holding plan as packs are first posted. Each unit has one holder, kept
 * for good once placed (WASM never shrinks, and the page keeps no copy of a pack): the market on the last worker,
 * a city on the worker it adds least to, by `cost` (its trips), counting the market as MARKET_SHARE of the work.
 * Copies (a dominant city's periods split, the market twice) and costs from measured passes are stage 8's.
 */
export class Holding {
  private of = new Map<string, number>();
  private cost = new Map<string, number>();
  private n: number;
  constructor(n: number) { this.n = n; }
  get workers() { return this.n; }
  private marketAt() { return this.n - 1; }
  /** each worker's load as placed, in cost units, the market counted on its worker before it is placed (a save's
   * zones are dealt before its market packs arrive) */
  loads(): number[] {
    const load = new Array<number>(this.n).fill(0);
    let cities = 0;
    for (const [u, w] of this.of) if (u !== MARKET_UNIT) { load[w] += this.cost.get(u) ?? 0; cities += this.cost.get(u) ?? 0; }
    load[this.of.get(MARKET_UNIT) ?? this.marketAt()] += (cities * MARKET_SHARE) / (1 - MARKET_SHARE);
    return load;
  }
  private place(unit: string, cost: number, among: number): number {
    if (unit === MARKET_UNIT) return among - 1;
    const load = this.loads().slice(0, among), k = MARKET_SHARE / (1 - MARKET_SHARE), mh = this.of.get(MARKET_UNIT) ?? among - 1;
    let best = 0, bestLoad = Infinity;
    for (let w = 0; w < among; w++) {
      const l = load[w] + cost + (w === mh ? k * cost : 0);
      if (l < bestLoad - 1e-9) { best = w; bestLoad = l; }
    }
    return best;
  }
  /** `unit`'s holder, placed now if it has none */
  holder(unit: string, cost = 1): number {
    let w = this.of.get(unit);
    if (w === undefined) {
      this.cost.set(unit, Math.max(1, cost));
      w = unit === MARKET_UNIT ? this.marketAt() : this.place(unit, Math.max(1, cost), this.n);
      this.of.set(unit, w);
    }
    return w;
  }
  placed(unit: string): number | undefined { return this.of.get(unit); }
  /** the pool is cut to `n`: units held past it are placed again among the first `n` (returned); the rest stay */
  shrink(n: number): string[] {
    const moved = [...this.of].filter(([, w]) => w >= n).map(([u]) => u);
    for (const u of moved) this.of.delete(u);
    this.n = n;
    for (const u of moved.sort((a, b) => (a === MARKET_UNIT ? -1 : b === MARKET_UNIT ? 1 : (this.cost.get(b) ?? 0) - (this.cost.get(a) ?? 0) || (a < b ? -1 : 1)))) {
      this.of.set(u, u === MARKET_UNIT ? this.marketAt() : this.place(u, this.cost.get(u) ?? 1, n));
    }
    return moved;
  }
}

export interface PlanUnit {
  /** a zone id, or "market" */
  id: string;
  /** ms per period of a whole solve (its passes over every crowding round) */
  cost: readonly number[];
  /** bytes it adds to a worker that holds it */
  mem: number;
  /** ms to open its packs on a worker (each holder opens it), before a load's solve can start */
  open?: number;
}

export interface Plan {
  /** each unit's holders, ascending */
  holders: Record<string, number[]>;
  /** each unit's solving worker per period, one of its holders */
  solver: Record<string, number[]>;
  /** each worker's ms of passes */
  load: number[];
  /** each worker's bytes of units held */
  mem: number[];
  /** every worker within `room` */
  fits: boolean;
}

export interface PlanInput {
  units: readonly PlanUnit[];
  workers: number;
  /** bytes of units a worker may hold */
  room: number;
  /** a copy is added while the busiest worker is over the even share by more than this */
  over?: number;
}

const sum = (a: readonly number[]) => a.reduce((s, x) => s + x, 0);
const byCost = (a: PlanUnit, b: PlanUnit) => sum(b.cost) - sum(a.cost) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** the plan's loads and memory, from its holders and solvers */
export function measure(units: readonly PlanUnit[], workers: number, holders: Record<string, number[]>, solver: Record<string, number[]>, room = Infinity): Plan {
  const load = new Array<number>(workers).fill(0), mem = new Array<number>(workers).fill(0);
  for (const u of units) {
    for (const w of holders[u.id]) mem[w] += u.mem;
    u.cost.forEach((ms, q) => { load[solver[u.id][q]] += ms; });
  }
  return { holders, solver, load, mem, fits: mem.every((m) => m <= room) };
}

/** a load's length under a plan: the slowest worker's opening, then the busiest worker's passes */
export function spanOf(units: readonly PlanUnit[], plan: Plan): number {
  const open = plan.load.map(() => 0);
  for (const u of units) for (const w of plan.holders[u.id]) open[w] += u.open ?? 0;
  return Math.max(...open) + Math.max(...plan.load);
}

/** every unit on every worker, period q solved by `periodsOf`'s owner of q (today's pool) */
export function everywhere(units: readonly PlanUnit[], workers: number, periodsOf: (w: number) => readonly number[], room = Infinity): Plan {
  const all = Array.from({ length: workers }, (_, w) => w);
  const owner = (q: number) => all.find((w) => periodsOf(w).includes(q)) ?? 0;
  const holders: Record<string, number[]> = {}, solver: Record<string, number[]> = {};
  for (const u of units) { holders[u.id] = all.slice(); solver[u.id] = u.cost.map((_, q) => owner(q)); }
  return measure(units, workers, holders, solver, room);
}

/** deal unit u's periods over `hs`, each to the holder whose load is least (periods costliest first) */
function dealPeriods(u: PlanUnit, hs: readonly number[], load: number[]): number[] {
  const qs = u.cost.map((_, q) => q).sort((a, b) => u.cost[b] - u.cost[a] || a - b);
  const out = new Array<number>(u.cost.length).fill(hs[0]);
  for (const q of qs) {
    let best = hs[0];
    for (const w of hs) if (load[w] < load[best]) best = w;
    out[q] = best;
    load[best] += u.cost[q];
  }
  return out;
}

const leastOf = (cands: number[], load: readonly number[]) => cands.reduce((b, w) => (load[w] < load[b] ? w : b), cands[0]);

/**
 * The plan B uses: each unit to the least loaded worker with room, costliest first; then, while the
 * busiest worker is more than `over` past the even share, a copy of its costliest unit that shortens
 * the busiest worker on the least loaded worker with room, the unit's periods dealt over its holders.
 * Returns `everywhere` instead when that fits and is shorter, so it is never slower than today's pool.
 */
export function planHolding({ units, workers, room, over = 0.15 }: PlanInput, periodsOf?: (w: number) => readonly number[]): Plan {
  const order = units.slice().sort(byCost);
  const load = new Array<number>(workers).fill(0), mem = new Array<number>(workers).fill(0);
  const holders: Record<string, number[]> = {}, solver: Record<string, number[]> = {};
  const all = Array.from({ length: workers }, (_, w) => w);
  for (const u of order) {
    const fit = all.filter((w) => mem[w] + u.mem <= room);
    // none with room: the worker with the most room left (the plan then does not fit)
    const w = fit.length ? leastOf(fit, load) : all.reduce((b, x) => (mem[x] < mem[b] ? x : b), 0);
    holders[u.id] = [w];
    solver[u.id] = u.cost.map(() => w);
    load[w] += sum(u.cost);
    mem[w] += u.mem;
  }
  const mean = sum(load) / workers;
  for (let guard = 0; guard < units.length * workers; guard++) {
    const top = load.reduce((b, x, w) => (x > load[b] ? w : b), 0), worst = load[top];
    if (worst <= mean * (1 + over)) break;
    // the units the busiest worker solves a period of, costliest there first
    const onTop = (u: PlanUnit) => sum(u.cost.filter((_, q) => solver[u.id][q] === top));
    const mine = order.filter((u) => solver[u.id].includes(top)).sort((a, b) => onTop(b) - onTop(a) || byCost(a, b));
    let done = false;
    for (const u of mine) {
      const free = all.filter((w) => !holders[u.id].includes(w) && mem[w] + u.mem <= room);
      if (!free.length) continue;
      const v = leastOf(free, load);
      const hs = [...holders[u.id], v].sort((a, b) => a - b);
      const trial = load.slice();
      u.cost.forEach((ms, q) => { trial[solver[u.id][q]] -= ms; });
      const dealt = dealPeriods(u, hs, trial);
      if (Math.max(...trial) >= worst - 1e-9) continue;
      holders[u.id] = hs;
      solver[u.id] = dealt;
      trial.forEach((x, w) => { load[w] = x; });
      mem[v] += u.mem;
      done = true;
      break;
    }
    if (!done) break;
  }
  // a load waits for every worker's opening, then solves: where opening is known, move units held
  // once to the worker that most shortens the slowest opening plus the busiest solve
  if (units.some((u) => u.open)) {
    const open = new Array<number>(workers).fill(0);
    for (const u of units) for (const w of holders[u.id]) open[w] += u.open ?? 0;
    const span = () => Math.max(...open) + Math.max(...load);
    const once = order.filter((u) => holders[u.id].length === 1).sort((a, b) => (a.id < b.id ? -1 : 1));
    for (let guard = 0; guard < units.length * workers; guard++) {
      const now = span();
      let best: { u: PlanUnit; to: number; s: number } | null = null;
      for (const u of once) {
        const from = holders[u.id][0], c = sum(u.cost), o = u.open ?? 0;
        for (const to of all) {
          if (to === from || mem[to] + u.mem > room) continue;
          open[from] -= o; open[to] += o; load[from] -= c; load[to] += c;
          const s = span();
          open[from] += o; open[to] -= o; load[from] += c; load[to] -= c;
          if (s < now - 1e-6 * Math.max(1, now) && (!best || s < best.s - 1e-9)) best = { u, to, s };
        }
      }
      if (!best) break;
      const { u, to } = best, from = holders[u.id][0];
      open[from] -= u.open ?? 0; open[to] += u.open ?? 0; load[from] -= sum(u.cost); load[to] += sum(u.cost);
      mem[from] -= u.mem; mem[to] += u.mem;
      holders[u.id] = [to];
      solver[u.id] = u.cost.map(() => to);
    }
  }
  const plan = measure(units, workers, holders, solver, room);
  if (periodsOf && units.length) {
    const today = everywhere(units, workers, periodsOf, room);
    if (today.fits && spanOf(units, today) < spanOf(units, plan) - 1e-9) return today;
  }
  return plan;
}
