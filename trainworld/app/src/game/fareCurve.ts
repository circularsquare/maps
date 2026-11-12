// The player's fare curve (T-883, T-1112, T-1143): the fare for a trip against its length, straight
// stretches through three nodes the player places, carried on past the last at its slope. The
// first node is at 0 km (the base fare); each later one is farther and never cheaper, so a fare is
// never below the base and a longer trip never costs less.
//
// `base` and `perKm` are the first stretch as a line (the fare at 0 km and its slope). Demand's
// route search charges them on each ride (`set_fares`), the sim adds the rest of the curve on each
// trip's expected length (`set_fare_curve`, `curveKnots`) and reports that extra on the lines the
// trips ride, by km (T-1516); fare income is `faresFor` plus that extra (game/money.ts), so demand and income
// charge the same trips the same fares. A new game starts at `START_NODES`; an older save's straight
// line becomes nodes on it at 0, 20 and 100 km.

/** [trip km, US$] */
export type FareNode = [number, number];

export interface Fares {
  base: number;
  perKm: number;
  nodes: FareNode[];
  /** T-1277: the player's fare scales by country (ISO alpha-2); the rest follow income
   * (game/income.ts) */
  scales?: Record<string, number>;
  /** T-1277: the country the curve is set for; absent in a save from before, whose curve was
   * New York's fares */
  baseline?: string;
  /** T-1521: "usd" for a curve and scales under the market-rate income rule; absent in a save
   * from before (game/income.ts `oldCurveFactor`) */
  basis?: "usd";
  /** T-2670: the income power its start key (`scales["@"]`) was set under; absent in a save from
   * before, whose key is T-1521's (0.7, game/income.ts `startFromOld`) */
  income?: number;
  /** T-1626: the fare multiplier on high-speed lines; absent for `HSR_FARE` */
  hsr?: number;
  /** T-2194: the fare multiplier on ferry lines; absent for `FERRY_FARE` */
  ferry?: number;
  /** T-3028: the fare multiplier on bus lines; absent for `BUS_FARE` */
  bus?: number;
  /** T-1801: the transfer fare, paid on each boarding after a trip's first, in the curve's units
   * (times a country's scale where charged); absent for 0 */
  transfer?: number;
  /** T-2679: the reference country (ISO alpha-2), whose fares the Money tab's multiples are of: the
   * first built station's until the player picks another. It changes no fare, so demand never sees
   * it. Absent until a station is built, and in a save from before, whose own scales are then
   * rounded as they load (game/income.ts `tidyOwn`). */
  ref?: string;
}

/** T-1801: the transfer fare of `f`, 0 unless set. */
export function transferFare(f: { transfer?: number } | undefined): number {
  const v = f?.transfer;
  return v !== undefined && Number.isFinite(v) && v > 0 ? v : 0;
}
/** A transfer fare the player set, in the curve's units: not below 0, to a hundredth of a cent
 * (as the curve's fares, so a fare typed in a country with a scale reads back as typed). */
export function tidyTransfer(v: number): number {
  return Number.isFinite(v) ? capFare(Math.round(v * 1e4) / 1e4) : 0;
}

/** T-1626: a high-speed line's fares are the curve's times this unless the player sets another.
 * The base is paid once a trip; the rest of a trip's fare is times the share of its km on such
 * lines (docs: the Money section of SPEC.md). */
export const HSR_FARE = 1.25;
export const HSR_FARE_MAX = 5;
/** The high-speed multiplier of `f`, or the default. */
export function hsrFare(f: { hsr?: number } | undefined): number {
  const v = f?.hsr;
  return v !== undefined && Number.isFinite(v) && v >= 0 ? v : HSR_FARE;
}
/** A multiplier the player typed or stepped to: 0 to `HSR_FARE_MAX`, in hundredths. */
export function tidyHsr(v: number): number {
  return Math.min(HSR_FARE_MAX, Math.max(0, Math.round(v * 100) / 100));
}

/** T-2194: a ferry line's fares are the curve's part above the base times this (notes/T-2177.md 3.7),
 * in demand and income alike (T-2195), as the high-speed one. */
export const FERRY_FARE = 1;
/** The ferry multiplier of `f`, or the default. */
export function ferryFare(f: { ferry?: number } | undefined): number {
  const v = f?.ferry;
  return v !== undefined && Number.isFinite(v) && v >= 0 ? v : FERRY_FARE;
}
/** T-3028: a bus line's fares are the curve's part above the base times this; 0 to `HSR_FARE_MAX`
 * as the others, so a long bus can be made as cheap as the player likes. */
export const BUS_FARE = 1;
/** The bus multiplier of `f`, or the default. */
export function busFare(f: { bus?: number } | undefined): number {
  const v = f?.bus;
  return v !== undefined && Number.isFinite(v) && v >= 0 ? v : BUS_FARE;
}
/** The multiplier a line's rides pay on the part of the fare above the base: a ferry's, a
 * bus's, a high-speed line's, else 1. */
export function lineFareMul(line: { fast?: boolean; ferry?: boolean; bus?: boolean }, f: { hsr?: number; ferry?: number; bus?: number } | undefined): number {
  return line.ferry ? ferryFare(f) : line.bus ? busFare(f) : line.fast ? hsrFare(f) : 1;
}

/** The curve as charged in a country whose fare scale is `s`. */
export function scaleNodes(nodes: FareNode[], s: number): FareNode[] {
  return nodes.map(([k, f]) => [k, f * s]);
}
/** `curveKnots` times a country's scale. */
export function scaledKnots(f: { base: number; perKm: number; nodes?: FareNode[] }, s: number): Float32Array {
  const k = curveKnots(f);
  for (let i = 1; i < k.length; i += 2) k[i] *= s;
  return k;
}

export const NODE_KM = [0, 20, 100];
/** A new game's curve's first stretch (T-028, T-883), for the baseline country (T-1277,
 * game/income.ts): New York, at its income's scale of 11.6, starts at $1.00 plus $0.08 a km. */
export const START_FARE = { base: 0.0862, perKm: 0.006895 };
/** A new game's curve (T-1670): one straight line, in New York $1.00 plus 8 cents a km ($2.60 at
 * 20 km, $9.00 at 100). */
export const START_NODES: FareNode[] = [[0, 0.0862], [20, 0.2241], [100, 0.7757]];
/** Nodes keep at least this far apart, km (so f32 keeps them in order in the sim). */
export const NODE_GAP_KM = 0.1;
export const MAX_KM = 5000;
/** T-3303: the most a node of the curve (or the transfer fare) can charge, curve US$: 400 times a US start's
 * $2.50 base. A typed 90 billion otherwise reached every money sum and the demand solve as it was. */
export const FARE_MAX = 1000;
/** A fare held between 0 and `FARE_MAX`; not a number is 0. */
const capFare = (v: number): number => (Number.isNaN(v) ? 0 : Math.min(FARE_MAX, Math.max(0, v)));

export function lineNodes(base: number, perKm: number): FareNode[] {
  return NODE_KM.map((k) => [k, base + perKm * k]);
}

/** The fare for a trip of `km` on the curve through `nodes`, US$. */
export function fareAt(nodes: FareNode[], km: number): number {
  const n = nodes.length;
  if (n === 0) return 0;
  if (n === 1) return nodes[0][1];
  let i = 1;
  while (i < n - 1 && nodes[i][0] <= km) i++;
  const [k0, f0] = nodes[i - 1], [k1, f1] = nodes[i];
  return Math.max(0, f0 + ((f1 - f0) * (km - k0)) / (k1 - k0));
}

/** The first stretch as a line: its fare at 0 km (not below 0) and its slope. */
export function firstStretch(nodes: FareNode[]): { base: number; perKm: number } {
  if (nodes.length < 2) return { base: nodes[0]?.[1] ?? 0, perKm: 0 };
  const [[k0, f0], [k1, f1]] = nodes;
  const perKm = (f1 - f0) / (k1 - k0);
  return { base: Math.max(0, f0 - perKm * k0), perKm };
}

const cents = (v: number) => Math.round(v * 100) / 100;
/** T-1277: the curve's fares are kept finer than cents, so a fare typed in a country with a scale
 * reads back as typed */
const fine = (v: number) => Math.round(v * 1e4) / 1e4;

/** Three nodes, the first at 0 km: in order and apart by `NODE_GAP_KM` within range, fares to a
 * hundredth of a cent, and
 * never falling in fare. Any other curve (an older save's two nodes, or more) becomes its line
 * through its first and last node, at `NODE_KM`. */
export function tidyNodes(nodes: FareNode[]): FareNode[] {
  const sorted = [...nodes].sort((a, b) => a[0] - b[0]);
  let out: FareNode[];
  if (sorted.length === 3 && sorted[0][0] === 0) out = sorted.map(([k, f]) => [k, f] as FareNode);
  else {
    const [k0, f0] = sorted[0], [k1, f1] = sorted[sorted.length - 1];
    const perKm = k1 > k0 ? Math.max(0, (f1 - f0) / (k1 - k0)) : 0;
    out = lineNodes(Math.max(0, f0 - perKm * k0), perKm);
  }
  out[0][0] = 0;
  out[1][0] = Math.min(MAX_KM - NODE_GAP_KM, Math.max(NODE_GAP_KM, cents(out[1][0])));
  out[2][0] = Math.min(MAX_KM, Math.max(out[1][0] + NODE_GAP_KM, cents(out[2][0])));
  out[0][1] = capFare(fine(out[0][1]));
  for (let i = 1; i < 3; i++) out[i][1] = Math.max(out[i - 1][1], capFare(fine(out[i][1])));
  return out;
}

/** Fares from nodes the player set, or from a base and rate (an old save, a check). */
export function faresOf(f: { base?: number; perKm?: number; nodes?: FareNode[] } | undefined, fallback: { base: number; perKm: number }): Fares {
  if (f?.nodes && f.nodes.length >= 2) {
    const nodes = tidyNodes(f.nodes);
    // a base and rate given with nodes on their line stay as given, to the last digit
    if (f.base !== undefined && f.perKm !== undefined && isStraight({ base: f.base, perKm: f.perKm, nodes })) return { base: f.base, perKm: f.perKm, nodes };
    return { ...firstStretch(nodes), nodes };
  }
  const base = Math.max(0, f?.base ?? fallback.base), perKm = Math.max(0, f?.perKm ?? fallback.perKm);
  return { base, perKm, nodes: tidyNodes(lineNodes(base, perKm)) };
}

/** Whether the curve is its first stretch's straight line everywhere. */
export function isStraight(f: Fares): boolean {
  return f.nodes.every(([k, v]) => Math.abs(v - (f.base + f.perKm * k)) < 1e-6);
}

/** The knots the sim takes (`set_fare_curve`): km and US$ in pairs, none for a straight line. */
export function curveKnots(f: { base: number; perKm: number; nodes?: FareNode[] }): Float32Array {
  const full = faresOf(f, f);
  return isStraight(full) ? new Float32Array(0) : new Float32Array(full.nodes.flat());
}

/** The fare taken from `rides` trips riding `km` in all, US$, on the first stretch's line: the base
 * on each trip and the rate on the km (times `mul` on a high-speed line, T-1626), as demand charges
 * them on rides. The rest of a bent curve comes from demand itself (`LineDemand.fareX`). */
export function faresFor(f: Fares, rides: number, km: number, mul = 1): number {
  return f.base * rides + mul * f.perKm * km;
}
