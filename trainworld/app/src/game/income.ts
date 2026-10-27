// T-1277 (with T-1166, T-1521): fares by national income, in US dollars at market rates. A fare
// dollar costs a rider `FARE_W_REF` x (GDP per head / `FARE_Y_REF`)^-`FARE_E` perceived minutes, by
// the country the trip starts in (the plane's fare by the departing country; the local layers' car
// money stays at 3 minutes a dollar, the market's longer trips weigh it by income group, T-1431,
// below). The player's fare curve is set for `FARE_BASELINE`, a lower-middle-income country, and
// each country charges it times its fare scale: by default its income over the baseline's to the
// same power, rounded to a tenth, or the player's own figure (`Fares.scales`). GDP per head at
// market rates: countryIncome.json `usd` (notes/T-1521/money_table.py, World Bank
// NY.GDP.PCAP.CD). At market rates the exponent is 0.7 where it was 0.5 at PPP: price levels rise
// with income, and 0.7 keeps each country's minutes per unit of its own money as before (T-1521).

import table from "./countryIncome.json";

export const FARE_W_REF = 4.4;
export const FARE_Y_REF = 90000;
export const FARE_E = 0.7;
/** India, the largest lower-middle-income country (GDP per head about $2,700). */
export const FARE_BASELINE = "IN";
/** The player's fare scales go from this to `SCALE_MAX` in steps of `SCALE_STEP`. */
export const SCALE_MIN = 0.1;
export const SCALE_MAX = 19.9;
export const SCALE_STEP = 0.1;

const T = table as { usd: Record<string, number>; gdp: Record<string, number>; car?: Record<string, number>; quint?: Record<string, number[]> };
const GDP = T.usd;
const gdpOf = (country: string | undefined): number => (country && GDP[country]) || GDP[FARE_BASELINE];
const tenth = (v: number) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(v * 10) / 10));

/** Perceived minutes a fare dollar costs a rider from `country`. */
export const fareWeight = (country: string | undefined): number => FARE_W_REF * (gdpOf(country) / FARE_Y_REF) ** -FARE_E;
/** The fare scale `country` has unless the player sets one. */
export const incomeScale = (country: string | undefined): number => tenth((gdpOf(country) / gdpOf(FARE_BASELINE)) ** FARE_E);
/** The fare scale in `country`: the player's, else its income's. */
export function fareScale(country: string | undefined, scales: Record<string, number> | undefined): number {
  const own = country ? scales?.[country] : undefined;
  return own !== undefined && Number.isFinite(own) ? tenth(own) : incomeScale(country);
}
/** T-1286: per country the player set a scale for, that scale over its income's, where not 1: the
 * factor the regional market weighs fares from there at. */
export function fareFactors(scales: Record<string, number> | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  for (const c of Object.keys(scales ?? {})) {
    const f = fareScale(c, scales) / incomeScale(c);
    if (f !== 1) out[c] = f;
  }
  return out;
}
/** A player's scale as kept: a tenth within range. */
export const tidyScale = tenth;

// T-1521: a save from before kept its curve and scales under T-1277's rule (GDP per head at PPP, to
// the 0.5). It loads with its home country's fares weighing exactly as they did (others within the
// scales' rounding): the curve times `oldCurveFactor`, and a scale the player set keeping its factor
// over the income's.
const OLD = { w: 5, y: 60000, e: 0.5 };
const oldGdp = (c: string | undefined) => (c && T.gdp[c]) || T.gdp[FARE_BASELINE];
/** A country's default scale under the old rule. */
export const oldIncomeScale = (c: string | undefined): number =>
  Math.min(9.9, Math.max(0.1, Math.round((oldGdp(c) / T.gdp[FARE_BASELINE]) ** OLD.e * 10) / 10));
/** A curve dollar's perceived minutes in `home` at its default scale, the old rule's over the new one's. */
export const oldCurveFactor = (home: string | undefined): number =>
  (OLD.w * (oldGdp(home) / OLD.y) ** -OLD.e * oldIncomeScale(home)) / (fareWeight(home) * incomeScale(home));
/** A player's scale for `country` from a save under the old rule. */
export const scaleFromOld = (country: string, s: number): number => tidyScale((s / oldIncomeScale(country)) * incomeScale(country));

/** T-1521: what a car costs to run a km in `country`, US$ (fuel and distance tolls,
 * countryIncome.json `car`); the US's where a country has none. */
export const carUsdPerKm = (country: string | undefined): number => (country && T.car?.[country]) || T.car?.US || 0.15;

// T-1431: the regional market's longer trips come from a country's people in `INCOME_GROUPS` equal
// groups by income (the World Bank's income shares by quintile, countryIncome.json `quint`). A
// group's riders weigh every mode's money at the country's minutes a dollar times (its income over
// the mean)^-`GROUP_E` times `LONG_TRIP_MONEY`, and it makes longer trips in proportion to (its
// income over the mean)^`eps`, the market rule's trip rate by income.
export const INCOME_GROUPS = 3;
/** Within a country prices are the same, so T-1277's exponent at PPP. */
export const GROUP_E = 0.5;
/** T-1521: a longer trip's money weighs 0.4 of a local trip's, every mode alike: about US DOT's
 * value of an hour of intercity travel over a local one (all purposes $14.10 local, $20.40 by road
 * and $47.20 by air or high-speed rail, 2016 guidance). */
export const LONG_TRIP_MONEY = 0.4;
const QUINT = T.quint ?? {};
/** One income group: its share of the longer trips and the factor on `fareWeight` it weighs money at. */
export interface IncomeGroup {
  share: number;
  factor: number;
}
/** `country`'s income groups, each one's income over the mean from the quintiles' Lorenz curve
 * (income even within a quintile); `eps` the market rule's. One group with no table. */
export function incomeGroups(country: string | undefined, eps: number): IncomeGroup[] {
  const q = (country && QUINT[country]) || QUINT[FARE_BASELINE];
  if (!q || q.length !== 5) return [{ share: 1, factor: LONG_TRIP_MONEY }];
  const sum = q.reduce((a, b) => a + b, 0);
  const lorenz = (p: number) => {
    const i = Math.min(4, Math.floor(p * 5));
    let below = 0;
    for (let k = 0; k < i; k++) below += q[k] / sum;
    return below + (p * 5 - i) * (q[i] / sum);
  };
  const n = INCOME_GROUPS;
  const rel = Array.from({ length: n }, (_, k) => (lorenz((k + 1) / n) - lorenz(k / n)) * n);
  const trips = rel.map((y) => y ** eps);
  const total = trips.reduce((a, b) => a + b, 0);
  return rel.map((y, k) => ({ share: trips[k] / total, factor: LONG_TRIP_MONEY * y ** -GROUP_E }));
}
