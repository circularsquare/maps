// T-1277 (with T-1166): fares by national income. A fare dollar costs a rider
// `FARE_W_REF` x (GDP per head / `FARE_Y_REF`)^-`FARE_E` perceived minutes, by the country the trip
// starts in (the plane's fare by the departing country; the car's money stays at 3 minutes a dollar).
// The player's fare curve is set for `FARE_BASELINE`, a lower-middle-income country, and each
// country charges it times its fare scale: by default its income over the baseline's to the same
// power, rounded to a tenth, or the player's own figure (`Fares.scales`). GDP per head at PPP:
// countryIncome.json (notes/T-1277/income_table.py, World Bank NY.GDP.PCAP.PP.KD).

import table from "./countryIncome.json";

export const FARE_W_REF = 5;
export const FARE_Y_REF = 60000;
export const FARE_E = 0.5;
/** India, the largest lower-middle-income country (GDP per head at PPP about $10,000). */
export const FARE_BASELINE = "IN";
/** The player's fare scales go from this to `SCALE_MAX` in steps of `SCALE_STEP`. */
export const SCALE_MIN = 0.1;
export const SCALE_MAX = 9.9;
export const SCALE_STEP = 0.1;

const GDP = (table as { gdp: Record<string, number> }).gdp;
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
/** A player's scale as kept: a tenth within range. */
export const tidyScale = tenth;
