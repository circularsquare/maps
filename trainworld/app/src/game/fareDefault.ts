// T-1662: a country's default fares in round numbers. Each country charges the curve times its scale
// (game/income.ts), which gives fares like $0.789. While the player has set neither the curve nor the
// country's scale, the country's fares are the start curve's rounded: in a save from before T-1906
// its three fares rounded to a step by size; from T-1906 on, the start curve the country would have
// had as a new game's. Demand, income and the Money tab all read this one curve.

import { firstStretch, scaleNodes, START_NODES, type FareNode } from "./fareCurve";
import { EVERY, fareFactor, fareFactors, fareScale, incomeScale, START } from "./income";
import { zoneCountry } from "./cities";

/** `v` to the step for its size: a cent under $0.25, 5 cents under $10, 50 cents under $100, then a dollar, 5 dollars from $1,000. */
export function roundFare(v: number): number {
  const step = v < 0.25 ? 0.01 : v < 10 ? 0.05 : v < 100 ? 0.5 : v < 1000 ? 1 : 5;
  return Math.round(Math.round(v / step) * step * 1e4) / 1e4;
}

const sameNodes = (a: FareNode[], b: FareNode[]) => a.length === b.length && a.every(([k, f], i) => Math.abs(k - b[i][0]) < 1e-6 && Math.abs(f - b[i][1]) < 1e-6);
/** Whether `nodes` are a new game's curve from before T-1906, as the player has not touched it. */
export function isDefaultCurve(nodes: FareNode[]): boolean {
  return sameNodes(nodes, START_NODES);
}

/** The curve charged in `country` while the curve is the game's start curve and the player set no
 * scale for the country: its round fares, with the first stretch as a line; null otherwise (the
 * curve times the country's scale). */
export function roundedDefault(f: { nodes?: FareNode[]; scales?: Record<string, number> }, country: string | undefined): { base: number; perKm: number; nodes: FareNode[] } | null {
  if (!country || !f.nodes || f.scales?.[EVERY] !== undefined || f.scales?.[country] !== undefined) return null;
  const start = f.scales?.[START];
  let nodes: FareNode[];
  if (start !== undefined) {
    if (!sameNodes(f.nodes, startNodesAt(start))) return null;
    nodes = startNodesAt(incomeScale(country));
  } else {
    if (!isDefaultCurve(f.nodes)) return null;
    nodes = scaleNodes(f.nodes, incomeScale(country)).map(([k, v]) => [k, roundFare(v)] as FareNode);
  }
  return { ...firstStretch(nodes), nodes };
}

/** A country's multipliers on the curve as set (T-1277): on the base and on the rate, its fare scale (a rounded
 * default's ratios to the curve's first stretch, T-1662). The one rule for which multiple a fare is charged at: a
 * stop's (its country's) for income, a city population's (its country's) for the fares its riders were solved with
 * (T-3753: income converts the second to the first). */
export function countryFareScales(country: string | undefined, fares: { base: number; perKm: number; nodes?: FareNode[]; scales?: Record<string, number> }): { base: number; rate: number } {
  const scale = fareScale(country, fares.scales), r = roundedDefault(fares, country);
  return { base: r && fares.base > 0 ? r.base / fares.base : scale, rate: r && fares.perKm > 0 ? r.perKm / fares.perKm : scale };
}

type Scaled = { scales?: Record<string, number>; byCity?: boolean; cityScales?: Record<string, number> };
const inCity = new WeakMap<object, Map<string, unknown>>();
/** T-3766: `f` as charged in zone `city` (null: outside every zone). With fares by city its scales keep only the start
 * and every-country keys, plus the city's own multiple under its country's key, so every country rule (the default,
 * the rounded default, the relative storage) applies to the city as it would to its country; else `f` itself. */
export function faresIn<F extends Scaled>(f: F, city: string | null | undefined): F {
  if (!f.byCity) return f;
  let by = inCity.get(f) as Map<string, F> | undefined;
  if (!by) inCity.set(f, (by = new Map()));
  const key = city ?? "";
  let out = by.get(key);
  if (!out) {
    const scales: Record<string, number> = {};
    for (const k of [START, EVERY]) if (f.scales?.[k] !== undefined) scales[k] = f.scales[k];
    const country = zoneCountry(city), own = city ? f.cityScales?.[city] : undefined;
    if (country && own !== undefined && Number.isFinite(own)) scales[country] = own;
    by.set(key, (out = { ...f, scales }));
  }
  return out;
}
/** T-3766: the multipliers on the curve in zone `city` (its country's, or with fares by city its own): the one rule
 * every reader charges by. */
export const cityFareScales = (city: string | null | undefined, f: Parameters<typeof countryFareScales>[1] & Scaled): { base: number; rate: number } =>
  countryFareScales(zoneCountry(city) ?? undefined, faresIn(f, city));
/** T-3766: the fare scale in zone `city`, as `fareScale` gives a country's. */
export const cityScale = (city: string | null | undefined, f: Scaled): number => fareScale(zoneCountry(city) ?? undefined, faresIn(f, city).scales);
/** T-4253: the base fare charged in zone `city` (null: outside every zone), US$, as the Money tab shows it. */
export function baseFareIn(f: Parameters<typeof countryFareScales>[1] & Scaled, city: string | null | undefined): number {
  const country = zoneCountry(city) ?? undefined, here = faresIn(f, city);
  return roundedDefault(here, country)?.base ?? (f.nodes?.[0]?.[1] ?? f.base) * fareScale(country, here.scales);
}

/** T-3766: the market's fare factors (`IntercityApi.set_fare_factors_by`): by country, and with fares by city each
 * city's that has a multiple of its own (the rest take their country's default). */
export function marketFareFactors(f: Scaled): { byCountry: Record<string, number>; byZone: Record<string, number> } {
  const byZone: Record<string, number> = {};
  if (f.byCity) {
    for (const city of Object.keys(f.cityScales ?? {})) {
      const c = zoneCountry(city), s = faresIn(f, city).scales;
      if (c && s?.[c] !== undefined) byZone[city] = fareFactor(c, s);
    }
  }
  return { byCountry: fareFactors(faresIn(f, null).scales), byZone };
}

/** T-1906: a new game's base fare in the US, US$. */
export const START_BASE_US = 2.5;
/** T-1906: a new game's curve in a country of income scale `s`: `START_NODES` times the factor that
 * makes the US's base `START_BASE_US`, times `s` (so the start's fares weigh the same with riders
 * anywhere: Japan $1.30, Germany $1.90, India $0.22), each fare rounded. */
export function startNodesAt(s: number): FareNode[] {
  const k = (START_BASE_US / (START_NODES[0][1] * incomeScale("US"))) * s;
  return scaleNodes(START_NODES, k).map(([km, v]) => [km, roundFare(v)] as FareNode);
}
/** A new game's curve where it starts, in `country`. */
export const startNodes = (country: string | undefined): FareNode[] => startNodesAt(incomeScale(country));
