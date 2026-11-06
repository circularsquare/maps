// Day and month names in the current language, for text that builds a date from the game's own day
// numbers (T-2632). Reads the language signal through tDate, so a caller that renders redraws on a change.

import { tDate } from "./index";

/** The full weekday name; `index` 0 is Monday (as game/days.ts `weekdayOf`). */
export function weekdayLong(index: number): string {
  // 1 January 2024 was a Monday
  return tDate(Date.UTC(2024, 0, 1 + (((index % 7) + 7) % 7)), { weekday: "long", timeZone: "UTC" });
}
