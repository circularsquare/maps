// T-1277 (with T-1166, T-1521): fares by national income, in US dollars at market rates. A fare
// dollar costs a rider `FARE_W_REF` x (GDP per head / `FARE_Y_REF`)^-`FARE_E` perceived minutes, by
// the country the trip starts in (the plane's fare by the departing country; the local layers' car
// money stays at 3 minutes a dollar, the market's longer trips weigh it by income group, T-1431,
// below). Each country charges the player's fare curve times its fare scale: the player's own figure
// (`Fares.scales`), else the default (`defaultScale`): its income scale, (its income over
// `FARE_BASELINE`'s, a lower-middle-income country, to the same power, in tenths), over the start
// country's in a new game (T-1906), whose curve is its start country's fares. GDP per head at market rates: countryIncome.json `usd`
// (notes/T-1521/money_table.py, World Bank NY.GDP.PCAP.CD). At market rates the exponent is 0.7 where it was 0.5 at PPP: price levels rise
// with income, and 0.7 keeps each country's minutes per unit of its own money as before (T-1521).

import table from "./countryIncome.json";

export const FARE_W_REF = 4.4;
export const FARE_Y_REF = 90000;
export const FARE_E = 0.7;
/** India, the largest lower-middle-income country (GDP per head about $2,700). */
export const FARE_BASELINE = "IN";
/** Fare scales, as stored, go from this to `SCALE_MAX`. */
export const SCALE_MIN = 0.1;
export const SCALE_MAX = 19.9;
/** The panel's step, in its own units. */
export const SHOWN_STEP = 0.05;

const T = table as { usd: Record<string, number>; gdp: Record<string, number>; car?: Record<string, number>; quint?: Record<string, number[]> };
const GDP = T.usd;
const gdpOf = (country: string | undefined): number => (country && GDP[country]) || GDP[FARE_BASELINE];
const tenth = (v: number) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(v * 10) / 10));
const hundredth = (v: number) => Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(v * 100) / 100));

/** Perceived minutes a fare dollar costs a rider from `country`. */
export const fareWeight = (country: string | undefined): number => FARE_W_REF * (gdpOf(country) / FARE_Y_REF) ** -FARE_E;
/** The fare scale a country's income gives: a save from before T-1906 charges it where the player
 * set none. */
export const incomeScale = (country: string | undefined): number => tenth((gdpOf(country) / gdpOf(FARE_BASELINE)) ** FARE_E);
/** T-1906: the key in `Fares.scales` holding the income scale of the country a new game started in,
 * whose fares the curve is: every other country charges it times its income scale over that one. A
 * save from before has none, its curve in `FARE_BASELINE`'s units and each country at its income
 * scale. */
export const START = "@";
/** T-1906's first form, in a few saves of 2026-11-02: every country at this one scale. */
export const EVERY = "*";
/** With either key, the player's own scales are kept as multiples of each country's default. */
const relative = (scales: Record<string, number> | undefined) => scales?.[START] !== undefined || scales?.[EVERY] !== undefined;
/** The scale `country` charges unless the player sets one. */
export function defaultScale(country: string | undefined, scales: Record<string, number> | undefined): number {
  const start = scales?.[START], all = scales?.[EVERY];
  if (start !== undefined && Number.isFinite(start) && start > 0) return incomeScale(country) / start;
  return all !== undefined && Number.isFinite(all) ? hundredth(all) : incomeScale(country);
}
/** The fare scale in `country`: the player's, else the default. */
export function fareScale(country: string | undefined, scales: Record<string, number> | undefined): number {
  const own = country && country !== START && country !== EVERY ? scales?.[country] : undefined;
  if (own === undefined || !Number.isFinite(own)) return defaultScale(country, scales);
  return multipleOf(country!, own, scales) * defaultScale(country, scales);
}
/** T-2021: the player's multiple of the country's default fares, to the nearest 0.01x, as the Money
 * tab shows it. A save from before T-1906 keeps the scale itself, so its multiple is that over the
 * default. Every effect of a fare multiplier works from this rounded multiple. */
function multipleOf(country: string, own: number, scales: Record<string, number> | undefined): number {
  return relative(scales) ? hundredth(own) : Math.min(SCALE_MAX, Math.max(0.01, Math.round((own / defaultScale(country, scales)) * 100) / 100));
}
/** T-2021: a player's scale as kept, so that it holds a multiple rounded to 0.01x. */
export function tidyOwn(country: string, v: number, scales: Record<string, number> | undefined): number {
  return relative(scales) ? hundredth(v) : multipleOf(country, v, scales) * defaultScale(country, scales);
}
/** T-1286: per country, its scale over its income's where not 1: the factor the regional market
 * weighs fares from there at. In a game from T-1906 on that is every country in the income table. */
export function fareFactors(scales: Record<string, number> | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  const keys = Object.keys(scales ?? {});
  for (const c of relative(scales) ? new Set([...Object.keys(GDP), ...keys]) : keys) {
    if (c === START || c === EVERY) continue;
    const f = fareScale(c, scales) / incomeScale(c);
    if (f !== 1) out[c] = f;
  }
  return out;
}
/** T-1906: a scale in the panel's terms, a multiple of the country's default (1.00x until the
 * player sets one), to 0.01x, and back to what is stored. */
export const shownScale = (scale: number, country: string, scales: Record<string, number> | undefined): number => Math.round((scale / defaultScale(country, scales)) * 100) / 100;
export const storedScale = (shown: number, country: string, scales: Record<string, number> | undefined): number =>
  tidyOwn(country, relative(scales) ? shown : shown * defaultScale(country, scales), scales);

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
export const scaleFromOld = (country: string, s: number): number => tenth((s / oldIncomeScale(country)) * incomeScale(country));

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
