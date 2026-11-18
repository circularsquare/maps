// T-3693: the other pair of a quad track piece, for lighting both pairs of a selected or hovered piece
// (map/quadPairs.ts finds them).

import type { TrackEdge } from "../game/types";
import type { NetworkBuffers } from "./network";
import { quadPairs } from "./quadPairs";

let cache: { n: NetworkBuffers; edges: readonly TrackEdge[]; pairs: Map<number, number> } | null = null;

/** The other pair of the quad track `edge` is one pair of, or -1. */
export function quadTwin(n: NetworkBuffers | null, edges: readonly TrackEdge[] | undefined, edge: number): number {
  if (!n || !edges || edge < 0) return -1;
  if (!cache || cache.n !== n || cache.edges !== edges) cache = { n, edges, pairs: quadPairs(edges, (id) => n.edgePts.get(id)) };
  return cache.pairs.get(edge) ?? -1;
}
