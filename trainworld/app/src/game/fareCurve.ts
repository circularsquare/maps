// The player's fare curve (T-883, T-1112, T-1143): the fare for a trip against its length, straight
// stretches through three nodes the player places, carried on past the last at its slope. The
// first node is at 0 km (the base fare); each later one is farther and never cheaper, so a fare is
// never below the base and a longer trip never costs less.
//
// `base` and `perKm` are the first stretch as a line (the fare at 0 km and its slope). Demand's
// route search charges them on each ride (`set_fares`), the sim adds the rest of the curve on each
// trip's expected length (`set_fare_curve`, `curveKnots`) and reports that extra where the trips
// first board; fare income is `faresFor` plus that extra (game/money.ts), so demand and income
// charge the same trips the same fares. The default curve, and any older save's line, is the line
// through nodes at 0, 20 and 100 km.

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
/** A new game's curve (T-028, T-883), for the baseline country (T-1277, game/income.ts): New York,
 * at its income's scale of 2.8, starts at $1.50 plus $0.10 a km. */
export const START_FARE = { base: 0.5357, perKm: 0.03571 };
/** Nodes keep at least this far apart, km (so f32 keeps them in order in the sim). */
export const NODE_GAP_KM = 0.1;
export const MAX_KM = 5000;

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
  out[0][1] = Math.max(0, fine(out[0][1]));
  for (let i = 1; i < 3; i++) out[i][1] = Math.max(out[i - 1][1], fine(out[i][1]));
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
 * on each trip and the rate on the km, as demand charges them on rides. The rest of a bent curve
 * comes from demand itself (`LineDemand.fareX`). */
export function faresFor(f: Fares, rides: number, km: number): number {
  return f.base * rides + f.perKm * km;
}
