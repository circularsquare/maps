// What a new line starts with, by mode (T-2919): the one place for these. The clock worker passes
// them to the sim when it makes the line; nothing lets the player change dwell or turnaround after.

import type { NewHeadways } from "../workers/saveFile";

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
};

/** The headways a new line of `kind` starts with: the player's default for it (T-3543, T-3703), else a bus's or
 * ferry's own above; null for a rail kind with none set (it starts on counts). */
export function headwaysFor(kind: keyof NewHeadways, set: NewHeadways): Levels | null {
  const h = set[kind];
  if (h) return [h[0], h[1], h[2]];
  return kind === "bus" || kind === "ferry" ? [...NEW_LINE[kind].headwayS] : null;
}
