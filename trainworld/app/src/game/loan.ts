// T-1661: the loan. One at a time, of one of these sizes (US$M), repaid in full; its interest is
// `LOAN_RATE` of the loan a game day, drained evenly through the day like upkeep.

import type { DayLedger } from "../workers/saveFile";

export const LOAN_AMOUNTS = [1000, 5000, 10000];
export const LOAN_RATE = 0.01;

/** T-1906: a loan's interest a day may not be more than the profit a day over this many of the
 * last whole game days (fewer in a younger game): a week, so weekends and weekdays both count. */
export const LOAN_DAYS = 7;

/** T-1906: profit a day, US$M (fares less running costs and upkeep; building and new trains are
 * spending, not losses), averaged over the last `LOAN_DAYS` whole days before `today`; null before
 * a whole day with fares. */
export function loanProfit(ledger: DayLedger[], today: number): { perDay: number; days: number } | null {
  const whole = ledger.filter((d) => d.day < today).slice(-LOAN_DAYS);
  if (!whole.some((d) => d.fares > 0)) return null;
  const sum = whole.reduce((s, d) => s + d.fares - d.running - (d.upkeep ?? 0), 0);
  return { perDay: sum / whole.length, days: whole.length };
}

/** Whether a loan of `amount` may be taken: its interest a day within `loanProfit`'s; before any
 * profit, only the smallest. */
export function loanAllowed(amount: number, profit: ReturnType<typeof loanProfit>): boolean {
  return profit ? amount * LOAN_RATE <= profit.perDay : amount === LOAN_AMOUNTS[0];
}
