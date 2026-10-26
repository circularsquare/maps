// Where the station tool puts a station (T-469, T-671): which of several overlapping tracks, and
// the end of the track when the platform would not fit the pointer's spot. Pure, so
// checks/t671-station-snap-check.mjs runs it without a browser.

export interface SnapEdge { id: number; a: number; b: number; built: boolean; lengthM: number; off?: boolean; tracks?: number; crossover?: boolean }
export interface SnapNode { platform: number; ports: number; bus?: true }
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
export function stationHere(spot: Spot, platform: number, edges: readonly SnapEdge[], nodes: ReadonlyMap<number, SnapNode>, byNode = edgesByNode(edges), byId: ReadonlyMap<number, SnapEdge> = new Map(edges.map((e) => [e.id, e]))): boolean {
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
 * the nearest. The tool takes the first one the track model accepts (T-671). `lay`: the network's
 * indexes when the caller keeps them (it runs on every pointer move, T-714). */
export function orderOverlapping(candidates: readonly Spot[], platform: number, edges: readonly SnapEdge[], nodes: ReadonlyMap<number, SnapNode>, onLast: ReadonlySet<number>, lay: Pick<Layout, "byNode" | "byId"> = layoutOf(edges, nodes)): number[] {
  const rank = candidates.map((c) => (stationHere(c, platform, edges, nodes, lay.byNode, lay.byId) ? 4 : 0) + (lay.byId.get(c.edge)?.built ? 2 : 0) + (onLast.has(c.edge) ? 0 : 1));
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

/** `firstAccepted` answered at once from `issuesOf` (T-695): the station tool's guess at what the
 * track model will take, so the ghost's first frame is where the click would put the station. */
export function likelyFirst<S>(spots: readonly S[], issuesOf: (s: S) => string[], endOf: (s: S) => S | null): { spot: S | null; issues: string[] } {
  let first: string[] = [];
  for (let i = 0; i < spots.length; i++) {
    const issues = issuesOf(spots[i]);
    if (!issues.length) return { spot: spots[i], issues: [] };
    if (i === 0) first = issues;
    if (onlyEndIssues(issues)) {
      const end = endOf(spots[i]);
      if (end && !issuesOf(end).length) return { spot: end, issues: [] };
    }
  }
  return { spot: null, issues: first };
}

export type Candidate = { kind: "node"; node: number } | { kind: "edge"; edge: number; frac: number };

/** What `layoutIssues` reads: the track, and what the page knows besides of each edge. */
export interface Layout {
  nodes: ReadonlyMap<number, SnapNode>;
  byNode: Map<number, number[]>;
  byId: Map<number, SnapEdge>;
  /** where flat crossings are on the edge, metres from its node `a` */
  crossingsOn?: (edge: number) => readonly number[];
}

export function layoutOf(edges: readonly SnapEdge[], nodes: ReadonlyMap<number, SnapNode>, more: Pick<Layout, "crossingsOn"> = {}): Layout {
  return { nodes, byNode: edgesByNode(edges), byId: new Map(edges.map((e) => [e.id, e])), ...more };
}

/** The refusals a station at `c` would meet, as far as the page can tell (T-695): an existing
 * station or junction at a node; the platform's room on each side up to the next station,
 * junction or track end (a drawn route's plain joints joined through as the track model joins
 * them, T-418); a flat crossing under the platform. Water, ramps, crossings that are not flat,
 * curves and cash are the worker's to find. Mirrors `check_station` and `unjoin_near` in
 * sim/src/track. */
export function layoutIssues(c: Candidate, platform: number, lay: Layout): string[] {
  const { nodes, byNode, byId } = lay;
  const out = new Set<string>();
  const ends = (n: number) => (nodes.get(n)?.platform ?? 0) > 0 ? reachOf(nodes.get(n)!) : 0;
  const joint = (n: number) => { const o = nodes.get(n); return !!o && o.platform === 0 && o.ports === 2; };
  const crossed = (edge: number, from: number, to: number) => (lay.crossingsOn?.(edge) ?? []).some((s) => s >= from - 1 && s <= to + 1);
  if (c.kind === "node") {
    const n = nodes.get(c.node);
    if (!n || n.bus || n.platform > 0 || n.ports > 2) return ["StationPorts"];
    const reach = n.ports <= 1 ? platform : platform / 2;
    for (const id of byNode.get(c.node) ?? []) {
      const e = byId.get(id)!;
      let other = e.a === c.node ? e.b : e.a, gap = e.lengthM;
      // at a loose end the joints within the platform are joined through first (T-860)
      if (n.ports <= 1) [other, gap] = joinThrough(e, other, gap, reach + 2, lay, joint);
      if (reach + ends(other) > gap - 1) out.add(joint(other) ? "PlatformJoint" : "PlatformTooLong");
      if (e.a === c.node ? crossed(id, 0, reach) : crossed(id, e.lengthM - reach, e.lengthM)) out.add("PlatformCrossing");
    }
    return [...out];
  }
  const e0 = byId.get(c.edge);
  if (!e0) return [];
  const mine = platform / 2, s0 = c.frac * e0.lengthM;
  if (crossed(e0.id, s0 - mine, s0 + mine)) out.add("PlatformCrossing");
  for (const [start, gap0] of [[e0.a, c.frac * e0.lengthM], [e0.b, (1 - c.frac) * e0.lengthM]] as const) {
    const [node, gap] = joinThrough(e0, start, gap0, mine + 2, lay, joint);
    if (mine + ends(node) > gap - 1) out.add(joint(node) ? "PlatformJoint" : "PlatformTooLong");
  }
  return [...out];
}

/** From `node`, `gap` metres along `e0`, on through plain joints while the gap is under `reach`,
 * as unjoin_near joins them: the node reached and the gap to it. */
function joinThrough(e0: SnapEdge, node: number, gap: number, reach: number, lay: Layout, joint: (n: number) => boolean): [number, number] {
  const same = (o: SnapEdge) => o.built === e0.built && !!o.off === !!e0.off && (o.tracks ?? 1) === (e0.tracks ?? 1) && !o.crossover && !e0.crossover;
  let from = e0.id;
  const seen = new Set([from]);
  while (gap < reach && joint(node)) {
    const next = (lay.byNode.get(node) ?? []).find((id) => id !== from);
    const o = next === undefined ? undefined : lay.byId.get(next);
    if (!o || seen.has(o.id) || !same(o)) break;
    seen.add(o.id);
    gap += o.lengthM;
    node = o.a === node ? o.b : o.a;
    from = o.id;
  }
  return [node, gap];
}

/** A refusal only because the platform would run off the end of its track. */
export function onlyEndIssues(issues: readonly string[]): boolean {
  return issues.length > 0 && issues.every((i) => i === "PlatformTooLong" || i === "PlatformJoint");
}

/** How far along `pts` (x, y, z triples from the edge's first node to its last) the point on it
 * nearest (x, y) is, 0 to 1. */
export function fractionAlong(pts: ArrayLike<number>, x: number, y: number): number {
  return nearestOn(pts, x, y)[0];
}

/** The point on `pts` nearest (x, y): how far along it is, 0 to 1, and how far off, in `pts` units. */
export function nearestOn(pts: ArrayLike<number>, x: number, y: number): [number, number] {
  let total = 0, best = Infinity, at = 0;
  for (let k = 0; k + 5 < pts.length; k += 3) {
    const x0 = pts[k], y0 = pts[k + 1], dx = pts[k + 3] - x0, dy = pts[k + 4] - y0, l2 = dx * dx + dy * dy, l = Math.sqrt(l2);
    const t = l2 ? Math.max(0, Math.min(1, ((x - x0) * dx + (y - y0) * dy) / l2)) : 0;
    const d = (x0 + t * dx - x) ** 2 + (y0 + t * dy - y) ** 2;
    if (d < best) best = d, at = total + t * l;
    total += l;
  }
  return [total ? at / total : 0, Math.sqrt(best)];
}
