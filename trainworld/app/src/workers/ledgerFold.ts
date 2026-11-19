// T-3757: the day ledgers (the network's and each region's) keep every day for the History chart's all time, which
// past HISTORY_COLS weeks draws a column per month or run of months. Once a ledger spans that long, its days older
// than DAYS_KEPT fold into one entry per calendar month: the chart's sums come out the same, and a ledger grows by
// 12 entries a year instead of 365 (notes/T-3757.md).

import { dateOf, DAY1_OF_YEAR, weekdayOf } from "../game/days";
import type { DayLedger } from "./saveFile";

/** The History chart's most columns of days; past it a column is a week, then a month, then several months. */
export const HISTORY_COLS = 90;
/** Days kept one by one, the day now among them: the 30-day chart, the loan's week and yesterday's figures read them. */
export const DAYS_KEPT = 120;

/** The month count of game day `d` since the calendar's start (365-day years). */
export const monthIndex = (d: number): number => Math.floor((d - 1 + DAY1_OF_YEAR) / 365) * 12 + dateOf(d).month;

/** Monday-to-Sunday weeks from game day `first` to `last`, both counted. */
export const weeksSpanned = (first: number, last: number): number => Math.floor((last - weekdayOf(last) - (first - weekdayOf(first))) / 7) + 1;

/** Fold `list`'s days before `today - DAYS_KEPT` into one entry per month, in place, once its own entries span more
 * weeks than the chart has columns (the chart reaches at least its last entry, so it groups by month). A folded entry
 * keeps its month's earliest day, so the chart's first day stays. */
export function foldLedger(list: DayLedger[], today: number): void {
  if (list.length < 2 || weeksSpanned(list[0].day, list[list.length - 1].day) <= HISTORY_COLS) return;
  const cut = today - DAYS_KEPT;
  const out: DayLedger[] = [];
  let i = 0;
  // (never the last entry: a region quiet since keeps where its chart ends)
  for (; i < list.length - 1 && list[i].day < cut; i++) {
    const e = list[i], into = out[out.length - 1];
    if (into && monthIndex(into.day) === monthIndex(e.day)) addFolded(into, e);
    else out.push(e);
  }
  if (out.length < i) list.splice(0, i, ...out);
}

function addFolded(to: DayLedger, e: DayLedger) {
  to.fares += e.fares;
  to.running += e.running;
  to.trains += e.trains;
  to.build += e.build;
  if (e.upkeep) to.upkeep = (to.upkeep ?? 0) + e.upkeep;
  if (e.loan) to.loan = (to.loan ?? 0) + e.loan;
  if (e.rides) to.rides = (to.rides ?? 0) + e.rides;
  if (e.points) to.points = (to.points ?? 0) + e.points;
}
