// The demand views catch up over a game day (T-156, Anita 2026-10-12). Demand itself has no ramp:
// riders, loads and fares follow a new network at once. What moves gradually is the mode colour
// (and size) of the map's bubbles: after a new result lands, a bubble shows the old result's legs
// times (1 - f) plus the new one's times f, where f is the share of a day's trips that have
// departed since the result landed, as if each person makes their trips once a day and switches
// on their next one. f follows the game clock, not wall time: paused, nothing moves; at the
// fastest speed a day passes in 90 s.
//
// The day's trips depart by the commute's period profile (docs/demand.md 4.2): trips to work
// 70/12/6/4/8% and trips home 2/18/58/16/6% over the five periods, so both legs together
// 36/15/32/10/7%, spread evenly over each period's hours. The everyday purposes leave later in
// the day, but the commute is the profile the model states for every city; a rough curve is
// enough here.

import { PERIODS } from "../game/periods";

/** Share of a day's trip legs departing in each period, PERIODS order. */
const PERIOD_SHARE = [0.36, 0.15, 0.32, 0.1, 0.07];

/** Share departing in each hour of the day, and the cumulative share at each hour's start. */
const HOURLY = new Float64Array(24);
const CUM = new Float64Array(25);
PERIODS.forEach((p, i) => {
  for (let h = p.from; h < p.to; h++) HOURLY[h] = PERIOD_SHARE[i] / (p.to - p.from);
});
for (let h = 0; h < 24; h++) CUM[h + 1] = CUM[h] + HOURLY[h];

/** Days' worth of trips departed by game time `t` (seconds): one a day, by the profile. */
export function tripsDone(t: number): number {
  const day = Math.floor(t / 86400);
  const h = (t - day * 86400) / 3600;
  const i = Math.min(23, Math.floor(h));
  return day + CUM[i] + HOURLY[i] * (h - i);
}

/** The share caught up at game time `t` of a catch-up that began at `start` (`tripsDone`), 0-1.
 * The clock going back (a game loaded) counts as caught up. */
export function caughtUp(start: number, t: number): number {
  const f = tripsDone(t) - start;
  return f >= 1 || f < 0 ? 1 : f;
}

/** Redraw while catching up when the share has moved this much, and no more often than this. */
export const CATCH_STEP = 0.01;
export const CATCH_WALL_MS = 250;
