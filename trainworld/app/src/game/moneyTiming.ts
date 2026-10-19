// When an hour's money lands (T-736, notes/T-736.md). Demand gives fares, rides and points per
// game hour of each period and kind of day, and the timetable a running cost per hour of each
// demand level; this turns those period tables into each clock hour's amount:
// - fares, rides and points follow the boarding account's arrivals over the period (the in-period
//   shapes of sim/src/track/boarding.rs, copied here: `SHAPE`, `shape_cum`, `shape_strength`);
// - fares and points land when the ride ends, and running costs as the trains run: a trailing
//   window as long as a typical one-way run, so a timetable or day change ramps over it.
// A period's total and the money over any run of whole days stay what the tables give. Plain
// functions: the clock worker imports them.

import { PERIODS } from "./periods";
import { dayKindOf } from "./days";

const enum Shape { Tent, Falling, Valley }
/** boarding.rs `SHAPE`, in PERIODS order */
const SHAPE: Shape[] = [Shape.Tent, Shape.Tent, Shape.Tent, Shape.Falling, Shape.Valley];

/** boarding.rs `shape_cum`: the share of a period's arrivals by fraction `x` of it. */
function shapeCum(s: Shape, c: number, x: number): number {
  const tent = x <= 0.5 ? 2 * x * x - x : 3 * x - 2 * x * x - 1;
  const g = s === Shape.Tent ? tent : s === Shape.Falling ? x - x * x : -tent;
  return x + c * g;
}

/** boarding.rs `shape_strength`: the shape's strength for a busiest-hour `factor`. */
function shapeStrength(s: Shape, hours: number, factor: number): number {
  const even = Math.round(hours) % 2 === 0;
  const busiest = s === Shape.Tent && even ? 1 - 2 / hours : s === Shape.Valley ? 1 - 2 / hours : 1 - 1 / hours;
  return busiest <= 0 ? 0 : Math.min(1, Math.max(0, (factor - 1) / busiest));
}

/** Each kind of day's shape strengths per period, from its busiest-hour factors (the first five
 * numbers of the riders `packRiders` sends). */
export function strengths(peak: ArrayLike<number>): number[] {
  return PERIODS.map((p, i) => shapeStrength(SHAPE[i], p.to - p.from, peak[i] ?? 1));
}

/** A table by kind of day and period: per game hour, evenly over the period. */
export type PerHour = (kind: number, period: number) => number;

/** What `per` adds up to over local seconds [u0, u1): shaped as the boarding account's arrivals by
 * `c[kind][period]`, or evenly where `c` is null. */
export function between(per: PerHour, c: number[][] | null, u0: number, u1: number): number {
  let sum = 0;
  for (let u = u0; u < u1;) {
    const day = Math.floor(u / 86400), k = dayKindOf(day);
    const s = (u - day * 86400) / 3600;
    const p = PERIODS.findIndex((x) => s >= x.from && s < x.to);
    const P = PERIODS[p], a = day * 86400 + P.from * 3600, len = (P.to - P.from) * 3600;
    const end = Math.min(u1, a + len);
    const v = per(k, p);
    if (v) {
      const cc = c ? (c[k] ?? c[0])?.[p] ?? 0 : 0;
      sum += v * (P.to - P.from) * (shapeCum(SHAPE[p], cc, (end - a) / len) - shapeCum(SHAPE[p], cc, (u - a) / len));
    }
    u = end;
  }
  return sum;
}

/** Samples of the trailing window: every hour's amount sums the same samples, so consecutive hours
 * hand on exactly what the window moves. */
const SAMPLES = 6;

/** The local hour starting at `tl`'s amount of `per` when it lands up to `lag` seconds after it
 * starts (evenly over that window): a ride ends or a train runs on after it set out. */
export function lagged(per: PerHour, c: number[][] | null, tl: number, lag: number): number {
  if (!(lag > 0)) return between(per, c, tl, tl + 3600);
  let sum = 0;
  for (let j = 0; j < SAMPLES; j++) {
    const v = (lag * (j + 0.5)) / SAMPLES;
    sum += between(per, c, tl - v, tl + 3600 - v);
  }
  return sum / SAMPLES;
}

/** A per-kind-of-day, per-period table as `PerHour` (a table with one kind serves both). */
export const table = (t: number[][] | undefined): PerHour => (k, p) => (t?.[k] ?? t?.[0])?.[p] ?? 0;

/** The lag for the network's money: the typical one-way run of its running lines, seconds, each
 * weighted by its car-km at the medium level. */
export function runLag(lines: { tph: number; lengthM: number; cars: number; oneWayS: number }[]): number {
  let w = 0, s = 0;
  for (const l of lines) {
    const x = l.tph * l.lengthM * l.cars;
    if (!(x > 0) || !(l.oneWayS > 0)) continue;
    w += x;
    s += x * l.oneWayS;
  }
  return w > 0 ? Math.min(7200, s / w) : 0;
}
