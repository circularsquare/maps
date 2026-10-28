// T-1662: a country's default fares in round numbers. The curve is kept in the baseline country's
// units and each country charges it times its scale (game/income.ts), which gives fares like $0.789.
// While the player has set neither the curve nor the country's scale, the country's three fares are
// rounded to a step by size; demand, income and the Money tab all read this one curve.

import { firstStretch, scaleNodes, START_NODES, type FareNode } from "./fareCurve";
import { incomeScale } from "./income";

/** `v` to the step for its size: a cent under $0.25, 5 cents under $10, 50 cents under $100, then a dollar, 5 dollars from $1,000. */
export function roundFare(v: number): number {
  const step = v < 0.25 ? 0.01 : v < 10 ? 0.05 : v < 100 ? 0.5 : v < 1000 ? 1 : 5;
  return Math.round(Math.round(v / step) * step * 1e4) / 1e4;
}

/** Whether `nodes` are a new game's curve, as the player has not touched it. */
export function isDefaultCurve(nodes: FareNode[]): boolean {
  return nodes.length === START_NODES.length && nodes.every(([k, f], i) => Math.abs(k - START_NODES[i][0]) < 1e-6 && Math.abs(f - START_NODES[i][1]) < 1e-6);
}

/** The curve charged in `country` while it is the default one and the player set no scale for the
 * country: its fares rounded, with the first stretch as a line; null otherwise (the curve times
 * the country's scale). */
export function roundedDefault(f: { nodes?: FareNode[]; scales?: Record<string, number> }, country: string | undefined): { base: number; perKm: number; nodes: FareNode[] } | null {
  if (!country || !f.nodes || f.scales?.[country] !== undefined || !isDefaultCurve(f.nodes)) return null;
  const nodes = scaleNodes(f.nodes, incomeScale(country)).map(([k, v]) => [k, roundFare(v)] as FareNode);
  return { ...firstStretch(nodes), nodes };
}
