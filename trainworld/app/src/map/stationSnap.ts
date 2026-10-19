// Where the station tool puts a station (T-469, T-671): which of several overlapping tracks, and
// the end of the track when the platform would not fit the pointer's spot. Pure, so
// checks/t671-station-snap-check.mjs runs it without a browser.

export interface SnapEdge { id: number; a: number; b: number; built: boolean; lengthM: number }
export interface SnapNode { platform: number; ports: number }
/** a spot on a track: the edge, and how far along it from node `a` to node `b`, 0 to 1 */
export interface Spot { edge: number; frac: number }

export function edgesByNode(edges: readonly SnapEdge[]): Map<number, number[]> {
  const byNode = new Map<number, number[]>();
  for (const e of edges) for (const k of [e.a, e.b]) (byNode.get(k) ?? byNode.set(k, []).get(k)!).push(e.id);
  return byNode;
}

/** How far a station's platform reaches along the track from its node, as the track model has it. */
const reachOf = (n: SnapNode) => (n.ports <= 1 ? n.platform : n.platform / 2);
/** the longest platform, m (PLATFORM_MAX in sim/src/track/net.rs) */
const MAX_REACH = 400;

/** Whether a station of `platform` metres at the spot would run into a station already on the
 * track: walking from the spot along the track through joints (stopping at junctions), a station
 * node closer than the two platforms' reaches. */
export function stationHere(spot: Spot, platform: number, edges: readonly SnapEdge[], nodes: ReadonlyMap<number, SnapNode>, byNode = edgesByNode(edges)): boolean {
  const byId = new Map(edges.map((e) => [e.id, e]));
  const e0 = byId.get(spot.edge);
  if (!e0) return false;
  const mine = platform / 2;
  const todo: { node: number; from: number; d: number }[] = [
    { node: e0.a, from: e0.id, d: spot.frac * e0.lengthM },
    { node: e0.b, from: e0.id, d: (1 - spot.frac) * e0.lengthM },
  ];
  const seen = new Set<string>();
  while (todo.length) {
    const { node, from, d } = todo.pop()!;
    const n = nodes.get(node);
    if (n && n.platform > 0) {
      if (d < mine + reachOf(n)) return true;
      continue;
    }
    if (!n || n.ports > 2 || d >= mine + MAX_REACH || seen.has(`${node}:${from}`)) continue;
    seen.add(`${node}:${from}`);
    for (const id of byNode.get(node) ?? []) {
      if (id === from) continue;
      const e = byId.get(id)!;
      todo.push({ node: e.a === node ? e.b : e.a, from: id, d: d + e.lengthM });
    }
  }
  return false;
}

/** The indices of `candidates` (spots under the pointer, nearest first) in the order the station
 * tool tries them: a track where the station would not meet another station before one where it
 * would, then blueprint before built, then the track the last station went on (`onLast`), then
 * the nearest. The tool takes the first one the track model accepts (T-671). */
export function orderOverlapping(candidates: readonly Spot[], platform: number, edges: readonly SnapEdge[], nodes: ReadonlyMap<number, SnapNode>, onLast: ReadonlySet<number>): number[] {
  const byNode = edgesByNode(edges);
  const built = new Map(edges.map((e) => [e.id, e.built]));
  const rank = candidates.map((c) => (stationHere(c, platform, edges, nodes, byNode) ? 4 : 0) + (built.get(c.edge) ? 2 : 0) + (onLast.has(c.edge) ? 0 : 1));
  return candidates.map((_, i) => i).sort((a, b) => rank[a] - rank[b] || a - b);
}

/** The first of `spots`, in order, that `accepts` (the track model's own answer, asked one at a
 * time), or null with the first spot's refusal. A spot refused only for running off its track's
 * end is retried at `endOf(spot)`. Stops early when `stale()`. */
export async function firstAccepted<S>(spots: readonly S[], accepts: (s: S) => Promise<string[] | null>, endOf: (s: S) => S | null, stale: () => boolean): Promise<{ spot: S | null; issues: string[] }> {
  let first: string[] = [];
  for (let i = 0; i < spots.length; i++) {
    const issues = await accepts(spots[i]);
    if (stale()) return { spot: null, issues: [] };
    if (issues === null) return { spot: spots[i], issues: [] };
    if (i === 0) first = issues;
    if (onlyEndIssues(issues)) {
      const end = endOf(spots[i]);
      if (end && (await accepts(end)) === null) return { spot: end, issues: [] };
    }
  }
  return { spot: null, issues: first };
}

/** A refusal only because the platform would run off the end of its track. */
export function onlyEndIssues(issues: readonly string[]): boolean {
  return issues.length > 0 && issues.every((i) => i === "PlatformTooLong" || i === "PlatformJoint");
}

/** How far along `pts` (x, y, z triples from the edge's first node to its last) the point on it
 * nearest (x, y) is, 0 to 1. */
export function fractionAlong(pts: ArrayLike<number>, x: number, y: number): number {
  let total = 0, best = Infinity, at = 0;
  for (let k = 0; k + 5 < pts.length; k += 3) {
    const x0 = pts[k], y0 = pts[k + 1], dx = pts[k + 3] - x0, dy = pts[k + 4] - y0, l2 = dx * dx + dy * dy, l = Math.sqrt(l2);
    const t = l2 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / l2)) : 0;
    const d = (x0 + t * dx - x) ** 2 + (y0 + t * dy - y) ** 2;
    if (d < best) best = d, at = total + t * l;
    total += l;
  }
  return total ? at / total : 0;
}
