// T-1662: a country's default fares in round numbers. Each country charges the curve times its scale
// (game/income.ts), which gives fares like $0.789. While the player has set neither the curve nor the
// country's scale, the country's fares are the start curve's rounded: in a save from before T-1906
// its three fares rounded to a step by size; from T-1906 on, the start curve the country would have
// had as a new game's. Demand, income and the Money tab all read this one curve.

import { firstStretch, scaleNodes, START_NODES, type FareNode } from "./fareCurve";
import { EVERY, incomeScale, START } from "./income";

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
