// T-1661, T-2022: the loan. One at a time, of one of these sizes (US$M), repaid by itself in
// `LOAN_TERM` equal daily payments (an annuity at `LOAN_RATE` a game day), drained evenly through
// each day like upkeep, whatever the cash.

import type { DayLedger, SaveHeader } from "../workers/saveFile";

export const LOAN_AMOUNTS = [1000, 5000, 10000];
export const LOAN_RATE = 0.01;
/** T-2022: game days of payments, every size alike */
export const LOAN_TERM = 60;

/** A loan out: `amount` borrowed and `pay` a day (US$M), for `days` more game days (fractional). */
export interface Loan { amount: number; pay: number; days: number }

/** The equal payment a day that clears `amount` in `days` at `LOAN_RATE`, interest included. */
export function loanPayment(amount: number, days = LOAN_TERM): number {
  return amount * LOAN_RATE / (1 - (1 + LOAN_RATE) ** -days);
}

export function newLoan(amount: number): Loan {
  return { amount, pay: loanPayment(amount), days: LOAN_TERM };
}

/** What paying the loan off now costs: the payments left, less the interest they would carry. */
export function loanBalance(l: Loan): number {
  return l.pay * (1 - (1 + LOAN_RATE) ** -l.days) / LOAN_RATE;
}

/** `dayPart` of a game day passes: the payment due for it, US$M, with `l.days` brought down; the
 * last payment stops where the term ends. */
export function loanDrain(l: Loan, dayPart: number): number {
  const part = Math.min(l.days, Math.max(0, dayPart));
  l.days = l.days - part < 1e-9 ? 0 : l.days - part;
  return l.pay * part;
}

/** The loan in a save's header: none, ours, or (before T-2022) an amount alone, which starts a
 * full term from the load. */
export function loanFromSave(h: Pick<SaveHeader, "loan" | "loanPay" | "loanDays">): Loan | null {
  const amount = Math.max(0, h.loan ?? 0);
  if (!(amount > 0)) return null;
  if (h.loanPay === undefined || h.loanDays === undefined) return newLoan(amount);
  return h.loanDays > 0 ? { amount, pay: h.loanPay, days: h.loanDays } : null;
}

export function loanToSave(l: Loan | null): Pick<SaveHeader, "loan" | "loanPay" | "loanDays"> {
  return l && l.days > 0 ? { loan: l.amount, loanPay: l.pay, loanDays: l.days } : {};
}

/** T-1906: a loan's payment a day may not be more than the profit a day over this many of the
 * last whole game days (fewer in a younger game): a week, so weekends and weekdays both count. */
export const LOAN_DAYS = 7;

/** T-1906: profit a day, US$M (fares minus running costs and upkeep; building and new trains are
 * spending, not losses), averaged over the last `LOAN_DAYS` whole days before `today`; null before
 * a whole day with fares. */
export function loanProfit(ledger: DayLedger[], today: number): { perDay: number; days: number } | null {
  const whole = ledger.filter((d) => d.day < today).slice(-LOAN_DAYS);
  if (!whole.some((d) => d.fares > 0)) return null;
  const sum = whole.reduce((s, d) => s + d.fares - d.running - (d.upkeep ?? 0), 0);
  return { perDay: sum / whole.length, days: whole.length };
}

/** Whether a loan of `amount` may be taken: its payment a day within `loanProfit`'s (T-2022);
 * before any profit, only the smallest. */
export function loanAllowed(amount: number, profit: ReturnType<typeof loanProfit>): boolean {
  return profit ? loanPayment(amount) <= profit.perDay : amount === LOAN_AMOUNTS[0];
}
