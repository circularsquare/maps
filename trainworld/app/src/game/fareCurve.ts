// The player's fare curve (T-883, T-1112): the fare for a trip against its length, the straight
// line through two nodes the player places, carried on past each at its slope, never below $0.
// The second node is farther and never cheaper, so a longer trip never costs less. The sim and
// `fareAt` still take any number of nodes; the Money tab keeps two (T-1114 may trim the rest).
//
// `base` and `perKm` are the curve's first stretch as a straight line (the fare at 0 km and its
// slope): the demand's route search charges them on each ride, and the sim adds the rest of the
// curve on each trip's length (sim/src/demand/kernel.rs `FareCurve`). The default curve, and any
// old save's base and rate, is that line through two nodes at 1 and 100 km.

/** [trip km, US$] */
export type FareNode = [number, number];

export interface Fares {
  base: number;
  perKm: number;
  nodes: FareNode[];
}

export const NODE_KM = [1, 100];
/** Nodes keep at least this far apart, km (so f32 keeps them in order in the sim). */
export const NODE_GAP_KM = 0.1;
export const MIN_KM = 0.1, MAX_KM = 5000;

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

/** Sorted by distance, apart by `NODE_GAP_KM`, within range, cents, never falling in fare, and
 * only the first and last of more than two (an older curve's). */
export function tidyNodes(nodes: FareNode[]): FareNode[] {
  const out: FareNode[] = [];
  for (const [k, f] of [...nodes].sort((a, b) => a[0] - b[0])) {
    const km = Math.min(MAX_KM, Math.max(MIN_KM, Math.round(k * 100) / 100));
    if (out.length && km < out[out.length - 1][0] + NODE_GAP_KM) continue;
    out.push([km, Math.max(0, Math.round(f * 100) / 100)]);
  }
  if (out.length > 2) out.splice(1, out.length - 2);
  for (let i = 1; i < out.length; i++) out[i][1] = Math.max(out[i][1], out[i - 1][1]);
  return out;
}

/** Fares from nodes the player set, or from a base and rate (an old save, a check). */
export function faresOf(f: { base?: number; perKm?: number; nodes?: FareNode[] } | undefined, fallback: { base: number; perKm: number }): Fares {
  if (f?.nodes && f.nodes.length >= 2) {
    const nodes = tidyNodes(f.nodes);
    // a base and rate given with nodes on their line stay as given, to the last digit
    if (nodes.length >= 2 && f.base !== undefined && f.perKm !== undefined && isStraight({ base: f.base, perKm: f.perKm, nodes })) return { base: f.base, perKm: f.perKm, nodes };
    if (nodes.length >= 2) return { ...firstStretch(nodes), nodes };
  }
  const base = Math.max(0, f?.base ?? fallback.base), perKm = Math.max(0, f?.perKm ?? fallback.perKm);
  return { base, perKm, nodes: lineNodes(base, perKm) };
}

/** Whether the curve is its first stretch's straight line everywhere. */
export function isStraight(f: Fares): boolean {
  const n = f.nodes;
  if (n.length < 2) return true;
  if (n[0][1] - f.perKm * n[0][0] < -1e-9) return false; // clamped at $0 before the first node
  return n.every(([k, v]) => Math.abs(v - (f.base + f.perKm * k)) < 1e-6);
}

/** The knots the sim takes (`set_fare_curve`): km and US$ in pairs, none for a straight line. */
export function curveKnots(f: { base: number; perKm: number; nodes?: FareNode[] }): Float32Array {
  const full = faresOf(f, f);
  return isStraight(full) ? new Float32Array(0) : new Float32Array(full.nodes.flat());
}

/** The fare taken from `rides` trips riding `km` in all, US$: exact for a straight line; for a
 * curve, each trip is taken at the mean length (T-883). */
export function faresFor(f: Fares, rides: number, km: number): number {
  if (isStraight(f)) return f.base * rides + f.perKm * km;
  return rides > 0 ? rides * fareAt(f.nodes, km / rides) : 0;
}
