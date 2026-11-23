// T-3693: the other pair of a quad track piece (map/quadPairs.ts finds them), for the track hover's count (T-3714).
// T-4111: found for the one edge hovered, from the edge grid the network carries over edits, not for every edge.

import type { TrackEdge } from "../game/types";
import type { NetworkBuffers } from "./network";
import { QuadLookup } from "./quadPairs";
import { edgesNear } from "./snapIndex";

let cache: { n: NetworkBuffers; edges: readonly TrackEdge[]; look: QuadLookup } | null = null;

/** The other pair of the quad track `edge` is one pair of, or -1. */
export function quadTwin(n: NetworkBuffers | null, edges: readonly TrackEdge[] | undefined, edge: number): number {
  if (!n || !edges || edge < 0) return -1;
  if (!cache || cache.n !== n || cache.edges !== edges) cache = { n, edges, look: new QuadLookup(edges, (id) => n.edgePts.get(id), (x, y, r) => edgesNear(n, x, y, r)) };
  return cache.look.twin(edge);
}
