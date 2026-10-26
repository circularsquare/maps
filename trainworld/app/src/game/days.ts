// Kinds of day (T-260, T-262, docs/demand-timing.md): weekday, Saturday and Sunday. The calendar is
// `day_kind_at` in sim/src/track/params.rs (the boarding account reads it); this is its copy for
// the main thread and the clock worker. Game day 1, a new game's first, is a Monday, so days 6 and
// 7 of each week are Saturday and Sunday where a place's country says nothing else (its holidays
// and own weekend, game/holidayCalendar.ts). The demand result is a weekday's; a Saturday and a
// Sunday are worked out from it per purpose and period (sim/src/demand/days.rs,
// workers/demandClient.ts). Plain data and one signal: the clock worker imports it too.

import { signal } from "@preact/signals";

/** 0 weekday, 1 Saturday, 2 Sunday, 3 a superholiday's away day (T-264, a scaled Sunday): an index
 * into the per-day results. */
export type DayKind = 0 | 1 | 2 | 3;
export const DAY_KINDS: DayKind[] = [0, 1, 2, 3];
export const AWAY_DAY = 3;
export const DAY_KIND_NAME = ["Weekday", "Saturday", "Sunday", "Holiday day"] as const;

/** The days a demand view can show (T-464): the four kinds, then the average day (5 weekdays, a
 * Saturday and a Sunday over 7, sim/src/demand/days.rs). The index into `demandByDay` and the `day`
 * of every demand query. The clock and the boarding account only ever see the four kinds. */
export type DemandDay = DayKind | 4;
export const DEMAND_DAYS: DemandDay[] = [0, 1, 2, 3, 4];
export const AVERAGE_DAY = 4;
/** How a figure for the day shown ends: "12,400 trip legs per day". The Day choice names the day. */
export const DEMAND_DAY_PHRASE = ["per weekday", "per Saturday", "per Sunday", "per holiday day", "per day"] as const;
/** "points on a weekday": the same, as the points hovers end */
export const DEMAND_DAY_ON = ["on a weekday", "on a Saturday", "on a Sunday", "on a holiday day", "a day"] as const;
const WEEKDAY_NAME = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Day of the week of game day `day`, 0 Monday. */
export const weekdayOf = (day: number): number => ((day - 1) % 7 + 7) % 7;
export const weekdayName = (day: number): string => WEEKDAY_NAME[weekdayOf(day)];
export const dayKindOf = (day: number): DayKind => { const w = weekdayOf(day); return w === 5 ? 1 : w === 6 ? 2 : 0; };
/** T-861, T-862: the calendar is a 365-day year with no leap day. Game day 1 is Monday January 27
 * (T-862's pick, DECISIONS.md). The weekday stays the day number's (above), so
 * each year starts a weekday later, as real non-leap years do. */
export const DAY1_OF_YEAR = 26;
const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const MONTH_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const MONTH_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
/** Day of the year of game day `day`, 0 January 1. */
export const dayOfYear = (day: number): number => ((day - 1 + DAY1_OF_YEAR) % 365 + 365) % 365;
/** The weekday (0 Monday) of January 1 of game day `day`'s year: which real year's holidays it
 * takes (game/holidays.ts). */
export const newYearWeekday = (day: number): number => weekdayOf(day - dayOfYear(day));
/** Month (0 January) and day of the month of game day `day`. */
export function dateOf(day: number): { month: number; date: number } {
  let d = dayOfYear(day), month = 0;
  while (d >= MONTH_DAYS[month]) d -= MONTH_DAYS[month++];
  return { month, date: d + 1 };
}
/** "Feb 24" */
export function shortDate(day: number): string {
  const { month, date } = dateOf(day);
  return `${MONTH_SHORT[month]} ${date}`;
}
/** "Mon Feb 24" */
export function dateName(day: number): string {
  return `${weekdayName(day)} ${shortDate(day)}`;
}

/** The kind of day at time `t` (seconds). Callers pass local time (T-309, game/localTime.ts): a
 * line's or a view's, never the bare game clock outside New York's time zone. */
export const dayKindAt = (t: number): DayKind => dayKindOf(Math.floor(t / 86400));

/** The kind of day it is in the local time in view (T-309), kept by game/clock.ts as the minutes
 * pass and the view pans (main thread only). */
export const currentDay = signal<DayKind>(0);

/** T-264, kept by game/holidays.ts where the player has built: the superholiday whose away days are
 * within the next 21 days (its name, the first if several; null for none), and whether one is within
 * the next 8. The away day's fares and points are worked out apart from Sunday's only while one is
 * near, and its boarding account only while one is soon; otherwise they are Sunday's, which no line
 * reads meanwhile. */
export const awayNear = signal<string | null>(null);
export const awaySoon = signal(false);
