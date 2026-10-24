// The player's fare curve (T-883, T-1112): the fare for a trip against its length, the straight
// line through two nodes the player places, carried on past each at its slope. The second node is
// farther and never cheaper, so a longer trip never costs less, and the line is never below $0 at
// 0 km (`fitOrigin`, `tidyNodes`), so a fare is never negative. `fareAt` and `tidyNodes` still
// take any number of nodes; the Money tab keeps two.
//
// `base` and `perKm` are the curve as a straight line (the fare at 0 km and its slope), and all
// the sim takes: its route search charges them on each ride (`set_fares`), so demand and income
// (`faresFor`) give the same fare. The default curve, and any old save's base and rate, is that
// line through two nodes at 1 and 100 km.

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
  if (out.length === 2) out[0][1] = Math.max(out[0][1], originFloor(out));
  return out;
}

const upCents = (v: number) => Math.ceil(v * 100 - 1e-9) / 100;
const downCents = (v: number) => Math.floor(v * 100 + 1e-9) / 100;
/** The lowest first-node fare that keeps the line through two nodes at or above $0 at 0 km. */
function originFloor([[k0], [k1, f1]]: FareNode[]): number {
  return upCents((k0 * f1) / k1);
}

/** Two nodes after moving node `moved` (a drag or a typed value): that node is held to where the
 * line is not below $0 at 0 km, the first node's fare no lower, the second's no higher. */
export function fitOrigin(nodes: FareNode[], moved: number): FareNode[] {
  const out = nodes.map((n) => [...n] as FareNode);
  if (out.length !== 2) return out;
  if (moved === 0) out[0][1] = Math.max(out[0][1], originFloor(out));
  else out[1][1] = Math.min(out[1][1], Math.max(out[0][1], downCents((out[0][1] * out[1][0]) / out[0][0])));
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
  return n.every(([k, v]) => Math.abs(v - (f.base + f.perKm * k)) < 1e-6);
}

/** The fare taken from `rides` trips riding `km` in all, US$: the base on each trip and the rate on
 * the km, as demand charges it. */
export function faresFor(f: Fares, rides: number, km: number): number {
  return f.base * rides + f.perKm * km;
}
