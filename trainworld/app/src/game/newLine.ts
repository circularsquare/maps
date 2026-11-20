// What a new line starts with, by mode (T-2919): the one place for these. The clock worker passes
// them to the sim when it makes the line; nothing lets the player change dwell or turnaround after.

import type { NewHeadways } from "../workers/saveFile";
import { PERIODS, rowOfPeriod, rowsOf } from "./periods";

type Levels = [number, number, number];

export const NEW_LINE = {
  /** Trains on the whole line at high, medium and low demand: none, so a line can be made first and
   * its trains decided later (T-2123; Anita, 2026-11-09). */
  train: { trains: [0, 0, 0] as Levels, dwellS: 30, turnaroundS: 180 },
  /** A bus every 8, 15 and 30 minutes (Anita, 2026-11-09; 8, 10 and 20 before); the dwell is stood at
   * train stations only. */
  bus: { headwayS: [480, 900, 1800] as Levels, dwellS: 30, turnaroundS: 120 },
  /** A boat every 20, 30 and 60 minutes (T-2273; the low-demand boat is T-3400). */
  ferry: { headwayS: [1200, 1800, 3600] as Levels, dwellS: 60, turnaroundS: 300 },
  /** T-3892: a ferry line over `LONG_FERRY_KM`, a boat an hour by day and none at night: at every 20 minutes a long
   * crossing runs more boats than it fills (notes/T-3892.md 1). */
  ferryLong: { headwayS: [3600, 3600, 0] as Levels },
};
/** T-3892: the route km past which a new ferry line starts at `NEW_LINE.ferryLong`'s headways. */
export const LONG_FERRY_KM = 40;

/** T-3767: a default's headways one per period, morning to night (game/periods.ts `PERIODS`): five kept as they
 * are; three (high, medium and low) with high's in the morning, at midday and in the afternoon, as a switch up to
 * more rows copies them. */
export function headwaysByPeriod(h: readonly number[]): number[] {
  return h.length === 5 ? [...h] : [h[0], h[0], h[0], h[1], h[2]];
}

/** T-3767: a default's headways for a line of `n` rows, in the day's order: each row the shortest of its periods'
 * that is some (0 where none is), as a switch down to fewer rows joins them. */
export function headwaysIn(h: readonly number[], n: number): number[] {
  const p = headwaysByPeriod(h);
  return rowsOf(n).map((r) => Math.min(...r.periods.map((i) => p[i]).filter((s) => s > 0), Infinity)).map((s) => (Number.isFinite(s) ? s : 0));
}

/** T-3767: the default a line of `n` rows with headways `rows` (the day's order) gives: three for three rows, else
 * one per period. */
export function headwaysOfRows(rows: readonly number[], n: number): number[] {
  if (n === 3) return [...rows];
  return PERIODS.map((_, i) => rows[rowOfPeriod(n, i)] ?? 0);
}

/** The headways a new line of `kind` and `n` rows (3, 4 or 5; T-3767) starts with, in the day's order: the
 * player's default for it (T-3543, T-3703), else a bus's or ferry's own above (a ferry line of `km` route km over
 * `LONG_FERRY_KM` the long one's, T-3892); null for a rail kind with none set (it starts on counts). */
export function headwaysFor(kind: keyof NewHeadways, set: NewHeadways, n = 3, km = 0): number[] | null {
  const own = kind === "ferry" && km > LONG_FERRY_KM ? NEW_LINE.ferryLong.headwayS : kind === "bus" || kind === "ferry" ? NEW_LINE[kind].headwayS : null;
  const h = set[kind] ?? own;
  return h ? headwaysIn(h, n) : null;
}
