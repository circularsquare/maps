// Small text helpers shared by the trip and demand panels.

import { capitalise } from "../../game/format";
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

const weight = (v: number) => String(Math.round(v * 100) / 100);

/** One term of an effective time equation: "walk 8 × 1.5", "wait 3", "fare $2.90 × 3". */
export interface Term {
  label: string;
  /** the real amount, already formatted ("" where the term is a count of its own minutes) */
  real: string;
  weight: number;
  perceived: number;
}

/** The effective time hover (T-699): only the equation, in the order of the rows, "Walk 8 × 1.5 +
 * wait 3 + ride 6 = 21 min". The total is the sum of the terms' unrounded minutes. */
export function equation(terms: Term[], total = terms.reduce((s, x) => s + x.perceived, 0)): string {
  const text = terms.map((x, i) => {
    const minus = x.perceived < 0;
    const body = `${x.label}${x.real ? " " + x.real : " " + num1(Math.abs(x.perceived))}${x.real && Math.abs(x.weight - 1) >= 0.005 ? ` × ${weight(x.weight)}` : ""}`;
    return i === 0 ? body : `${minus ? "−" : "+"} ${body}`;
  }).join(" ");
  return `${capitalise(text)} = ${Math.round(total)} min`;
}

/** T-911: shares as whole people out of `n` (null or 0: the model's own shares). */
export const roundTo = (shares: number[], n: number | null, pooled = -1): number[] => (n ? wholePeople(shares, n, pooled).map((p) => p / n) : shares);
