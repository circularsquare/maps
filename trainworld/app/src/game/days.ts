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

/** The days a demand view can show (T-464): the two kinds, then the average day (5 weekdays and 2
 * weekend days over 7, sim/src/demand/days.rs). The index into `demandByDay` and the `day` of every
 * demand query. The clock and the boarding account only ever see the two kinds. */
export type DemandDay = DayKind | 2;
export const DEMAND_DAYS: DemandDay[] = [0, 1, 2];
export const AVERAGE_DAY = 2;
/** How a figure for the day shown ends: "12,400 trip legs a day". The Day choice names the day. */
export const DEMAND_DAY_PHRASE = ["a weekday", "a weekend day", "a day"] as const;
const WEEKDAY_NAME = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Day of the week of game day `day`, 0 Monday. */
export const weekdayOf = (day: number): number => ((day - 1) % 7 + 7) % 7;
export const weekdayName = (day: number): string => WEEKDAY_NAME[weekdayOf(day)];
export const dayKindOf = (day: number): DayKind => (weekdayOf(day) >= 5 ? 1 : 0);
/** The kind of day at time `t` (seconds). Callers pass local time (T-309, game/localTime.ts): a
 * line's or a view's, never the bare game clock outside New York's time zone. */
export const dayKindAt = (t: number): DayKind => dayKindOf(Math.floor(t / 86400));

/** The kind of day it is in the local time in view (T-309), kept by game/clock.ts as the minutes
 * pass and the view pans (main thread only). */
export const currentDay = signal<DayKind>(0);
