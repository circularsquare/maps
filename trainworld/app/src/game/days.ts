// Kinds of day (T-260, docs/demand.md 4.2): weekday and weekend. The calendar is
// `day_kind_at` in sim/src/track/params.rs (the boarding account reads it); this is its copy for
// the main thread and the clock worker. Game day 1, a new game's first, is a Monday, so days 6 and
// 7 of each week are the weekend. The demand result is a weekday's; a weekend is worked out from
// it per purpose and period (sim/src/demand/days.rs, workers/demandClient.ts). Plain data and one
// signal: the clock worker imports it too.

import { signal } from "@preact/signals";

/** 0 weekday, 1 weekend: an index into the per-day results. */
export type DayKind = 0 | 1;
export const DAY_KINDS: DayKind[] = [0, 1];
export const DAY_KIND_NAME = ["Weekday", "Weekend"] as const;
const WEEKDAY_NAME = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Day of the week of game day `day`, 0 Monday. */
export const weekdayOf = (day: number): number => ((day - 1) % 7 + 7) % 7;
export const weekdayName = (day: number): string => WEEKDAY_NAME[weekdayOf(day)];
export const dayKindOf = (day: number): DayKind => (weekdayOf(day) >= 5 ? 1 : 0);
/** The kind of day at game time `t` (seconds). */
export const dayKindAt = (t: number): DayKind => dayKindOf(Math.floor(t / 86400));

/** The kind of day the game clock is in, kept by game/clock.ts as the minutes pass (main thread
 * only). The demand views show it unless the player chose one (game/ui.ts `shownDay`, T-266). */
export const currentDay = signal<DayKind>(0);
