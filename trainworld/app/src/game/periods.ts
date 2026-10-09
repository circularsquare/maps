// The five periods of the day (SPEC 4.2, 6.3): when each demand level is scheduled, and the
// periods demand is solved for. The definition is `PERIOD_HOURS` and `PERIOD_LEVEL` in
// sim/src/track/params.rs; this is its copy for the main thread, which has no WASM. The demand
// workers report the Rust table when they start and the demand client checks the two agree
// (workers/demandClient.ts), so an edit to one without the other shows up as a console error.

import type { Demand } from "./types";

export interface Period {
  name: string;
  /** hours of the day, [from, to) */
  from: number;
  to: number;
  demand: Demand;
}

export const PERIODS: readonly Period[] = [
  { name: "Morning peak", from: 6, to: 10, demand: "high" },
  { name: "Midday", from: 10, to: 16, demand: "medium" },
  { name: "Evening peak", from: 16, to: 20, demand: "high" },
  { name: "Evening", from: 20, to: 24, demand: "medium" },
  { name: "Night", from: 0, to: 6, demand: "low" },
];

const LEVEL_INDEX: Record<Demand, number> = { high: 0, medium: 1, low: 2 };

/** The period at game time `t` (seconds), as an index into PERIODS. */
export function periodAt(t: number): number {
  const h = (Math.floor(t / 3600) % 24 + 24) % 24;
  return PERIODS.findIndex((p) => h >= p.from && h < p.to);
}

/** Check against the Rust table (per period: from, to, level index, ...; `stride` numbers each). */
export function periodsMatch(rust: ArrayLike<number>, stride: number): boolean {
  if (rust.length !== PERIODS.length * stride) return false;
  return PERIODS.every((p, i) => rust[i * stride] === p.from && rust[i * stride + 1] === p.to && rust[i * stride + 2] === LEVEL_INDEX[p.demand]);
}
