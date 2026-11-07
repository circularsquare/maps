// T-1277 (with T-1166, T-1521): fares by national income, in US dollars at market rates. A fare
// dollar costs a rider `FARE_W_REF` x (GDP per head / `FARE_Y_REF`)^-`FARE_E` perceived minutes, by
// the country the trip starts in (the plane's fare by the departing country; the local layers' car
// money stays at 3 minutes a dollar, the market's longer trips weigh it by income group, T-1431,
// below). Each country charges the player's fare curve times its fare scale: the player's own figure
// (`Fares.scales`), else the default (`defaultScale`): its income scale, (its income over
// `FARE_BASELINE`'s, a lower-middle-income country, to the same power, in tenths), over the start
// country's in a new game (T-1906), whose curve is its start country's fares. GDP per head at market rates: countryIncome.json `usd`
// (notes/T-1521/money_table.py, World Bank NY.GDP.PCAP.CD). T-1521 had the exponent 0.7 (T-1277's 0.5 at PPP);
// T-2670 (Anita, 2026-11-06, T-2668 option N) makes it 0.5 at market rates for both the fare scale and the
// minutes a dollar, so a default fare still weighs the same with riders everywhere and fares scale more
// gently with income (notes/T-2668.md). The income scale is anchored on the US's T-1521 figure, 11.6, so
// every saved curve charges the US what it did.

import table from "./countryIncome.json";

export const FARE_W_REF = 4.4;
export const FARE_Y_REF = 90000;
export const FARE_E = 0.5;
/** T-2670: the power a save's fares were set under when it carries none (`Fares.income`): T-1521's. */
export const OLD_FARE_E = 0.7;
/** T-2670: the US's income scale, T-1521's, which every other country's is set from. */
export const US_SCALE = 11.6;
/** India, the largest lower-middle-income country (GDP per head about $2,700): its weight is the
 * regional market's (`curveWeight`), and a country without a figure takes its income. */
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

/** T-2669 (T-2668 option N): a country's train running costs over the world price list (the US's),
 * (its GDP per head over the US's)^`RUN_E`, to a hundredth; 1 outside every zone or for a country
 * without a figure. Mostly wages, so they follow income; upkeep, construction and trains stay at
 * world prices. */
export const RUN_E = 0.5;
export const runScale = (country: string | null | undefined): number => {
  const g = country ? GDP[country] : undefined;
  return g ? Math.max(0.01, Math.round((g / GDP.US) ** RUN_E * 100) / 100) : 1;
};
/** T-2669: a line's running cost factor, its running km split between regions (`[zone, share]`,
 * game/regionMoney.ts `lineShares`'s `run`) each at its country's `runScale`; 1 with none. */
export function runFactorOf(shares: readonly [string, number][] | undefined, countryOf: (zone: string) => string | null | undefined): number {
  if (!shares?.length) return 1;
  let f = 0, t = 0;
  for (const [z, s] of shares) { f += s * runScale(z ? countryOf(z) : null); t += s; }
  return t > 0 ? f / t : 1;
}

/** Perceived minutes a fare dollar costs a rider from `country`. */
export const fareWeight = (country: string | undefined): number => FARE_W_REF * (gdpOf(country) / FARE_Y_REF) ** -FARE_E;
/** The fare scale a country's income gives: a save from before T-1906 charges it where the player
 * set none. T-2670: the US's 11.6 times (its income over the US's)^`FARE_E`, in tenths (India 2.0,
 * Mexico 4.6, Japan 7.3). */
export const incomeScale = (country: string | undefined): number => tenth(US_SCALE * (gdpOf(country) / GDP.US) ** FARE_E);
/** T-2670: perceived minutes a dollar of the curve as set, the regional market's weight: every country
 * at its default scale weighs the curve about this (a country's scale times its minutes a dollar is
 * the same everywhere but for the scales' tenths). */
export const curveWeight = (): number => fareWeight(FARE_BASELINE) * incomeScale(FARE_BASELINE);
/** T-2670: a start key (`START`) saved under T-1521's rule, its start country's income scale then, as
 * that country's income scale now. */
export const startFromOld = (s: number): number => tenth(US_SCALE * ((s ** (1 / OLD_FARE_E) * GDP[FARE_BASELINE]) / GDP.US) ** FARE_E);
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
/** The fare scale in `country`: the player's, else the default. A player's scale is kept as a
 * multiple of the country's default with either key, else as the scale itself. */
export function fareScale(country: string | undefined, scales: Record<string, number> | undefined): number {
  const own = country && country !== START && country !== EVERY ? scales?.[country] : undefined;
  if (own === undefined || !Number.isFinite(own)) return defaultScale(country, scales);
  return relative(scales) ? own * defaultScale(country, scales) : own;
}
/** T-2021: a player's scale in a save without a reference country (`Fares.ref`), as it loads: its
 * multiple of the country's default to the nearest 0.01x. */
export function tidyOwn(country: string, v: number, scales: Record<string, number> | undefined): number {
  return relative(scales) ? hundredth(v) : Math.min(SCALE_MAX, Math.max(0.01, Math.round((v / defaultScale(country, scales)) * 100) / 100)) * defaultScale(country, scales);
}
/** T-2679: a player's scale as the worker keeps it: as set, so a multiple of the reference country
 * (`storedScale`) holds exactly. */
export const keepOwn = (v: number): number => Math.min(1000, Math.max(0.001, v));
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
/** T-2679: a country's scale in the panel's terms, a multiple of the reference country's (`refScale`),
 * to 0.01x, and back to what is stored: the multiple to 0.01x, so it charges exactly that many times
 * the reference's scale. */
export const shownScale = (scale: number, refScale: number): number => Math.round((scale / refScale) * 100) / 100;
export function storedScale(shown: number, country: string, scales: Record<string, number> | undefined, refScale: number): number {
  const s = hundredth(shown) * refScale;
  return keepOwn(relative(scales) ? s / defaultScale(country, scales) : s);
}

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
