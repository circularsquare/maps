// Small text helpers shared by the trip and demand panels.

import { wholePeople } from "../../game/trip";

/** A share of 0 to 1 as a whole percent; "<1%" for one too small to round to a percent. */
export const pctShare = (v: number): string => (v > 0 && v < 0.005 ? "<1%" : `${Math.round(100 * v)}%`);

/** `n` of `of` as a whole percent; "<1%" for a count too small to round to one. */
export const pctOf = (n: number, of: number): string => {
  const p = of > 0 ? (100 * n) / of : 0;
  return n >= 0.5 && p < 0.5 ? "<1%" : Math.round(p) + "%";
};

/** Minutes, at least one: "6 min". */
export const minText = (m: number): string => `${Math.max(1, Math.round(m))} min`;

/** To one decimal place, a trailing .0 dropped. */
export const num1 = (v: number): string => String(Math.round(v * 10) / 10);

/** T-911: shares as whole people out of `n` (null or 0: the model's own shares). */
export const roundTo = (shares: number[], n: number | null, pooled = -1): number[] => (n ? wholePeople(shares, n, pooled).map((p) => p / n) : shares);
