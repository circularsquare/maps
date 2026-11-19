// The five periods of the day (SPEC 4.2, 6.3): when each demand level is scheduled (high 6-20,
// medium 20-24, low 0-6), and the periods demand is solved for. The definition is `PERIOD_HOURS`
// and `PERIOD_ROW` in sim/src/track/params.rs; this is its copy for the main thread, which has
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

/** T-3542: each period's code for a message's select ("in the morning peak", "at midday", ...) */
export const PERIOD_CODE = ["morning", "midday", "afternoon", "evening", "night"] as const; // i18n-ignore (select codes)

/** T-3542: the rows a line of 3, 4 or 5 rows sets its trains in, in the day's order: each its code
 * (`schedule.rowName`) and its periods. Three rows are high, medium and low demand; four split midday
 * out of high (rush is the morning and evening peaks), five the afternoon too (sim params.rs
 * `ROW_PLACES`, `PERIOD_PLACE`). */
export interface Row { code: "high" | "medium" | "low" | "rush" | "morning" | "midday" | "afternoon" | "evening" | "night"; periods: number[] }
export const ROWS_OF: Record<3 | 4 | 5, Row[]> = {
  3: [{ code: "high", periods: [0, 1, 2] }, { code: "medium", periods: [3] }, { code: "low", periods: [4] }],
  4: [{ code: "rush", periods: [0, 2] }, { code: "midday", periods: [1] }, { code: "evening", periods: [3] }, { code: "night", periods: [4] }],
  5: [{ code: "morning", periods: [0] }, { code: "midday", periods: [1] }, { code: "afternoon", periods: [2] }, { code: "evening", periods: [3] }, { code: "night", periods: [4] }],
};
/** The rows of a line of `n` rows (3 where `n` is none of 3, 4, 5). */
export const rowsOf = (n: number): Row[] => ROWS_OF[(n === 4 || n === 5 ? n : 3) as 3 | 4 | 5];
/** The row of a line of `n` rows that period `p` is in. */
export const rowOfPeriod = (n: number, p: number): number => rowsOf(n).findIndex((r) => r.periods.includes(p));
/** A row's hours, [from, to) in order, neighbouring periods joined (as `levelHours`). */
export function rowHours(r: Row): [number, number][] {
  const out: [number, number][] = [];
  for (const p of r.periods.map((i) => PERIODS[i]).sort((a, b) => a.from - b.from)) {
    const last = out[out.length - 1];
    if (last && last[1] === p.from) last[1] = p.to;
    else out.push([p.from, p.to]);
  }
  return out;
}

/** T-3542: a demand network's run times per stop, two runs by five periods (`(run * 5 + period) * n
 * + k` from the line's stops x 10; sim/src/demand/api.rs `TIMES_PER_STOP`); trains an hour are one
 * per period. */
export const TIMES_PER_STOP = 2 * PERIODS.length;

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

/** T-3542: trains a day from trains an hour in each period (`tphP`): each level's periods at one rate
 * taken as one stretch, so a line of three rows gives the sum by level to the bit. */
export function trainsADay(tphP: readonly number[]): number {
  let out = 0;
  for (const lev of ["high", "medium", "low"] as const) {
    const ps = PERIODS.flatMap((p, i) => (p.demand === lev ? [i] : []));
    if (ps.every((i) => tphP[i] === tphP[ps[0]])) out += tphP[ps[0]] * ps.reduce((s, i) => s + PERIODS[i].to - PERIODS[i].from, 0);
    else for (const i of ps) out += tphP[i] * (PERIODS[i].to - PERIODS[i].from);
  }
  return out;
}

/** Check against the Rust table (per period: from, to, level index, ...; `stride` numbers each). */
export function periodsMatch(rust: ArrayLike<number>, stride: number): boolean {
  if (rust.length !== PERIODS.length * stride) return false;
  return PERIODS.every((p, i) => rust[i * stride] === p.from && rust[i * stride + 1] === p.to && rust[i * stride + 2] === LEVEL_INDEX[p.demand]);
}
