// T-2798: the currency the player reads money in. Display only: every figure in the game, the saves and
// the sim stay US$, and `game/format.ts` multiplies by a fixed rate on the way to the screen (and divides
// on the way back for what the player types or steps). One table, so a rate changes in one place
// (docs/DECISIONS.md, T-2798). T-2860: every currency of a country the game has that is not tiny.

export interface CurrencySpec {
  /** what the picker says: the code and the symbol, in no language in particular */
  label: string;
  /** written before the figure; one that ends in a letter carries its space ("CHF 2.90") */
  symbol: string;
  /** for a currency that locally writes a plain "$": the symbol where it has to be told apart ("A$") */
  full?: string;
  /** how many of it one US dollar buys: a round figure, never updated */
  rate: number;
  /** decimals in a price (a fare, a car's cost): none where a price is a whole number */
  dp: number;
  /** the finest price a player types or drags to, in the currency: a cent, a yen, ten won. The game stores a
   * fare to about 0.0012 US$, so a currency of rate r needs a grain over r * 0.0012 (10.5 gives 0.05, not 0.01). */
  grain: number;
  /** a click on a fare's + or - moves it this far, in the currency */
  fareStep: number;
  /** the finest rate per km a player types or steps to, in the currency (a fare per distance, to a tenth of that typed) */
  rateGrain: number;
}

const spec = (label: string, symbol: string, rate: number, dp: number, grain: number, fareStep: number, rateGrain: number, full?: string): CurrencySpec =>
  ({ label, symbol, rate, dp, grain, fareStep, rateGrain, full });

const TABLE = {
  USD: spec("USD $", "$", 1, 2, 0.01, 0.25, 0.01),
  GBP: spec("GBP £", "£", 0.8, 2, 0.01, 0.25, 0.01),
  EUR: spec("EUR €", "€", 0.9, 2, 0.01, 0.25, 0.01),
  CNY: spec("RMB ¥", "¥", 7, 2, 0.01, 2, 0.05),
  JPY: spec("JPY ¥", "¥", 150, 0, 1, 50, 1),
  KRW: spec("KRW ₩", "₩", 1300, 0, 10, 500, 10),
  CAD: spec("CAD $", "$", 1.4, 2, 0.01, 0.25, 0.01, "C$"),
  AUD: spec("AUD $", "$", 1.5, 2, 0.01, 0.25, 0.01, "A$"),
  INR: spec("INR ₹", "₹", 90, 0, 1, 20, 0.5),
  MXN: spec("MXN $", "$", 18, 2, 0.5, 5, 0.1, "MX$"),
  AED: spec("AED", "AED ", 3.7, 2, 0.01, 1, 0.05),
  ARS: spec("ARS $", "$", 1400, 0, 10, 500, 10, "AR$"),
  BDT: spec("BDT ৳", "৳", 120, 0, 1, 20, 0.5),
  BRL: spec("BRL R$", "R$", 5.5, 2, 0.01, 1, 0.05),
  BYN: spec("BYN", "Br ", 3.3, 2, 0.01, 1, 0.05),
  CHF: spec("CHF", "CHF ", 0.8, 2, 0.01, 0.25, 0.01),
  CLP: spec("CLP $", "$", 950, 0, 10, 250, 10, "CL$"),
  COP: spec("COP $", "$", 4000, 0, 50, 1000, 50, "CO$"),
  CZK: spec("CZK Kč", "Kč ", 22, 0, 1, 5, 0.1),
  DKK: spec("DKK kr", "kr ", 6.8, 2, 0.01, 2, 0.1),
  EGP: spec("EGP E£", "E£", 50, 0, 1, 10, 0.5),
  HKD: spec("HKD $", "$", 7.8, 2, 0.01, 2, 0.1, "HK$"),
  HUF: spec("HUF Ft", "Ft ", 360, 0, 5, 100, 5),
  IDR: spec("IDR Rp", "Rp ", 16000, 0, 100, 5000, 100),
  MYR: spec("MYR RM", "RM ", 4.4, 2, 0.01, 1, 0.05),
  NGN: spec("NGN ₦", "₦", 1500, 0, 10, 500, 10),
  NOK: spec("NOK kr", "kr ", 10.5, 2, 0.05, 2, 0.1),
  NZD: spec("NZD $", "$", 1.7, 2, 0.01, 0.25, 0.01, "NZ$"),
  PEN: spec("PEN S/", "S/ ", 3.7, 2, 0.01, 1, 0.05),
  PHP: spec("PHP ₱", "₱", 58, 0, 1, 10, 0.5),
  PKR: spec("PKR Rs", "Rs ", 280, 0, 1, 50, 1),
  PLN: spec("PLN zł", "zł ", 3.8, 2, 0.01, 1, 0.05),
  RON: spec("RON lei", "lei ", 4.5, 2, 0.01, 1, 0.05),
  RSD: spec("RSD din", "din ", 105, 0, 1, 20, 1),
  RUB: spec("RUB ₽", "₽", 90, 0, 1, 20, 0.5),
  SAR: spec("SAR", "SAR ", 3.75, 2, 0.01, 1, 0.05),
  SEK: spec("SEK kr", "kr ", 10, 2, 0.05, 2, 0.1),
  SGD: spec("SGD $", "$", 1.3, 2, 0.01, 0.25, 0.01, "S$"),
  THB: spec("THB ฿", "฿", 34, 0, 1, 10, 0.5),
  TRY: spec("TRY ₺", "₺", 40, 0, 1, 10, 0.5),
  TWD: spec("TWD NT$", "NT$", 31, 0, 1, 10, 0.5),
  UAH: spec("UAH ₴", "₴", 41, 0, 1, 10, 0.5),
  VND: spec("VND ₫", "₫", 25000, 0, 1000, 5000, 250),
  ZAR: spec("ZAR R", "R ", 18, 2, 0.5, 5, 0.1),
} satisfies Record<string, CurrencySpec>;

export type Currency = keyof typeof TABLE;
export const CURRENCY: Record<Currency, CurrencySpec> = TABLE;

/** The picker's order: the ten most chosen first, then the rest by code. */
const FIRST: Currency[] = ["USD", "EUR", "GBP", "JPY", "CNY", "KRW", "CAD", "AUD", "INR", "MXN"];
export const CURRENCIES: readonly Currency[] = [...FIRST, ...(Object.keys(TABLE) as Currency[]).filter((c) => !FIRST.includes(c)).sort()];

export const isCurrency = (s: unknown): s is Currency => typeof s === "string" && Object.prototype.hasOwnProperty.call(TABLE, s);

/** Region (ISO 3166 alpha-2) to currency, for the countries the game has and the euro area. A region not here gives USD. */
const EURO = "AT BE BG HR CY EE FI FR DE GR IE IT LV LT LU MT NL PT SK SI ES AD MC SM VA ME XK";
const REGION: Record<string, Currency> = {
  US: "USD", EC: "USD", PR: "USD", GB: "GBP", JP: "JPY", CN: "CNY", KR: "KRW", CA: "CAD", AU: "AUD", IN: "INR", MX: "MXN",
  AE: "AED", AR: "ARS", BD: "BDT", BR: "BRL", BY: "BYN", CH: "CHF", LI: "CHF", CL: "CLP", CO: "COP", CZ: "CZK", DK: "DKK", EG: "EGP",
  HK: "HKD", HU: "HUF", ID: "IDR", MY: "MYR", NG: "NGN", NO: "NOK", NZ: "NZD", PE: "PEN", PH: "PHP", PK: "PKR", PL: "PLN",
  RO: "RON", RS: "RSD", RU: "RUB", SA: "SAR", SE: "SEK", SG: "SGD", TH: "THB", TR: "TRY", TW: "TWD", UA: "UAH", VN: "VND", ZA: "ZAR",
};
for (const r of EURO.split(" ")) REGION[r] = "EUR";

/** T-2860: the currency a player with no choice starts on: the region of the first of the browser's languages
 * that has one (es-MX gives MXN, en-GB GBP, de-DE EUR). A language with no region, like plain "es", says
 * nothing; never the city a game starts in. USD when nothing says. */
export function defaultCurrency(langs?: readonly string[]): Currency {
  const list = langs ?? (typeof navigator === "undefined" ? [] : navigator.languages?.length ? navigator.languages : [navigator.language ?? ""]);
  for (const l of list) {
    const region = l.split(/[-_]/).slice(1).find((p) => /^[A-Za-z]{2}$/.test(p));
    const c = region && REGION[region.toUpperCase()];
    if (c) return c;
  }
  return "USD";
}

/** `v` to the nearest multiple of `g`; a fraction grain divides by its whole inverse, so 2.9 on a grain of
 * 0.01 is 2.9 and not 2.9000000000000004. */
export function roundTo(v: number, g: number): number {
  if (g >= 1) return Math.round(v / g) * g;
  const inv = Math.round(1 / g);
  return Math.round(v * inv) / inv;
}
