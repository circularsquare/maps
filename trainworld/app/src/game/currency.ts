// T-2798: the currency the player reads money in. Display only: every figure in the game, the saves and
// the sim stay US$, and `game/format.ts` multiplies by a fixed rate on the way to the screen (and divides
// on the way back for what the player types or steps). One table, so a rate changes in one place
// (docs/DECISIONS.md, T-2798).

export const CURRENCIES = ["USD", "GBP", "EUR", "CNY", "JPY", "KRW"] as const;
export type Currency = (typeof CURRENCIES)[number];

export interface CurrencySpec {
  /** what the picker says, in no language in particular */
  label: string;
  symbol: string;
  /** how many of it one US dollar buys: a round figure, never updated */
  rate: number;
  /** decimals in a price (a fare, a car's cost): none for yen and won */
  dp: number;
  /** the finest price a player types or drags to, in the currency: a cent, a yen, ten won */
  grain: number;
  /** a click on a fare's + or - moves it this far, in the currency */
  fareStep: number;
  /** the finest rate per km a player types or steps to, in the currency (a fare per distance, to a tenth of that typed) */
  rateGrain: number;
}

export const CURRENCY: Record<Currency, CurrencySpec> = {
  USD: { label: "USD $", symbol: "$", rate: 1, dp: 2, grain: 0.01, fareStep: 0.25, rateGrain: 0.01 },
  GBP: { label: "GBP £", symbol: "£", rate: 0.8, dp: 2, grain: 0.01, fareStep: 0.25, rateGrain: 0.01 },
  EUR: { label: "EUR €", symbol: "€", rate: 0.9, dp: 2, grain: 0.01, fareStep: 0.25, rateGrain: 0.01 },
  CNY: { label: "RMB ¥", symbol: "¥", rate: 7, dp: 2, grain: 0.01, fareStep: 2, rateGrain: 0.05 },
  JPY: { label: "JPY ¥", symbol: "¥", rate: 150, dp: 0, grain: 1, fareStep: 50, rateGrain: 1 },
  KRW: { label: "KRW ₩", symbol: "₩", rate: 1300, dp: 0, grain: 10, fareStep: 500, rateGrain: 10 },
};

export const isCurrency = (s: unknown): s is Currency => typeof s === "string" && (CURRENCIES as readonly string[]).includes(s);

/** `v` to the nearest multiple of `g`; a fraction grain divides by its whole inverse, so 2.9 on a grain of
 * 0.01 is 2.9 and not 2.9000000000000004. */
export function roundTo(v: number, g: number): number {
  if (g >= 1) return Math.round(v / g) * g;
  const inv = Math.round(1 / g);
  return Math.round(v * inv) / inv;
}
