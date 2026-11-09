// What a new line starts with, by mode (T-2919): the one place for these. The clock worker passes
// them to the sim when it makes the line; nothing lets the player change dwell or turnaround after.

type Levels = [number, number, number];

export const NEW_LINE = {
  /** Trains on the whole line at high, medium and low demand: none, so a line can be made first and
   * its trains decided later (T-2123; Anita, 2026-11-09). */
  train: { trains: [0, 0, 0] as Levels, dwellS: 30, turnaroundS: 180 },
  /** A bus every 8, 15 and 30 minutes (Anita, 2026-11-09; 8, 10 and 20 before); the dwell is stood at
   * train stations only. */
  bus: { headwayS: [480, 900, 1800] as Levels, dwellS: 30, turnaroundS: 120 },
  /** A boat every 20 and 30 minutes, none at low demand (T-2273). */
  ferry: { headwayS: [1200, 1800, 0] as Levels, dwellS: 60, turnaroundS: 300 },
};
