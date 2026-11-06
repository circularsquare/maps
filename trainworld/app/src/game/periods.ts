// The five periods of the day (SPEC 4.2, 6.3): when each demand level is scheduled (high 6-20,
// medium 20-24, low 0-6), and the periods demand is solved for. The definition is `PERIOD_HOURS`
// and `PERIOD_LEVEL` in sim/src/track/params.rs; this is its copy for the main thread, which has
// no WASM. The demand
// workers report the Rust table when they start and the demand client checks the two agree
// (workers/demandClient.ts), so an edit to one without the other shows up as a console error.

import { t } from "../i18n";
import type { Demand } from "./types";

export interface Period {
  name: string;
  /** hours of the day, [from, to) */
  from: number;
  to: number;
  demand: Demand;
}

export const PERIODS: readonly Period[] = [
  { get name() { return t("periods.morningPeak"); }, from: 6, to: 10, demand: "high" },
  { get name() { return t("periods.midday"); }, from: 10, to: 16, demand: "high" },
  { get name() { return t("periods.eveningPeak"); }, from: 16, to: 20, demand: "high" },
  { get name() { return t("periods.evening"); }, from: 20, to: 24, demand: "medium" },
  { get name() { return t("periods.night"); }, from: 0, to: 6, demand: "low" },
];

const LEVEL_INDEX: Record<Demand, number> = { high: 0, medium: 1, low: 2 };

/** An hour of the day as the clock writes it, "06:00"; the day's end is "24:00" (T-2271). */
export const hourText = (h: number): string => `${String(h).padStart(2, "0")}:00`;
/** Hours `a` to `b` as players read them: "06:00-10:00". */
export const hourSpan = (a: number, b: number): string => `${hourText(a)}-${hourText(b)}`;

/** The period at time `t` (seconds), as an index into PERIODS. Callers pass local time (T-309):
 * each region's periods run on its own hours. */
export function periodAt(t: number): number {
  const h = (Math.floor(t / 3600) % 24 + 24) % 24;
  return PERIODS.findIndex((p) => h >= p.from && h < p.to);
}

/** The hours of the day a demand level is scheduled, [from, to) in order, neighbouring periods of
 * the same level joined. */
export function levelHours(level: Demand): [number, number][] {
  const out: [number, number][] = [];
  for (const p of [...PERIODS].sort((a, b) => a.from - b.from)) {
    if (p.demand !== level) continue;
    const last = out[out.length - 1];
    if (last && last[1] === p.from) last[1] = p.to;
    else out.push([p.from, p.to]);
  }
  return out;
}

/** Check against the Rust table (per period: from, to, level index, ...; `stride` numbers each). */
export function periodsMatch(rust: ArrayLike<number>, stride: number): boolean {
  if (rust.length !== PERIODS.length * stride) return false;
  return PERIODS.every((p, i) => rust[i * stride] === p.from && rust[i * stride + 1] === p.to && rust[i * stride + 2] === LEVEL_INDEX[p.demand]);
}
