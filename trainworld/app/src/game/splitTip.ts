// The hover of a mode split bar (T-550): the top bar's, and a row of the Money tab's Cities section.

import { count } from "./format";
import type { ModeSplit } from "./types";

/** The split in rides (trip legs) a day, `all` their total (`modeRides`). */
export type SplitRides = ModeSplit & { all: number };

/** "Rides a day in New York on a weekday: Train 18.5% (8.81M), Walking ..., Driving ..., all 47.6M
 * rides". `where` is the phrase after "Rides a day" (" in New York on a weekday"), `note` what
 * follows (" (an early estimate)"). With no split yet every share is "–". */
export function splitTip(where: string, split: ModeSplit | null, rides: SplitRides | null, note: string): string {
  const s = split ?? { train: 0, walk: 0, drive: 0 };
  const pct = (v: number) => (split ? v.toFixed(1) + "%" : "–");
  const body = rides
    ? `Train ${pct(s.train)} (${count(rides.train)}), Walking ${pct(s.walk)} (${count(rides.walk)}), Driving ${pct(s.drive)} (${count(rides.drive)}), all ${count(rides.all)} rides`
    : `Train ${pct(s.train)}, Walking ${pct(s.walk)}, Driving ${pct(s.drive)}`;
  return `Rides a day${where}: ${body}${note}`;
}
