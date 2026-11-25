// Grids over the network's nodes and track points (T-724), so the build tools' snap and a station's
// track direction look only at what is near the pointer instead of projecting every node and
// every point of every edge on each pointer move (7-17 ms on the real New York save). Candidates
// come back in the order a full scan meets them (ascending ids, the model's), so the nearest-first ties fall the same way.
// T-4284: an edit's grid is the last one with the records the edit changed laid over it: the older grid passes over the
// records changed since it was built, and their new ones are in a small grid of their own, so a grid follows an edit at
// the cost of what changed. Past LAID_MAX records laid over, the next query builds it whole.

import { checkPatches, sideOf, type WorldDiff, worldDiff } from "../game/worldDiff";
import { timed } from "../game/stepTimes";
import type { WorldView } from "../game/types";
import { CellLevels } from "./cellIndex";
import { anchor, toLocal } from "./geo";
import type { NetworkBuffers } from "./network";

type Node = WorldView["nodes"] extends Map<number, infer N> ? N : never;

/** a grid with more records than this changed since it was built whole is built whole again */
const LAID_MAX = 4096;
const NONE: ReadonlySet<number> = new Set();

/** The indices `levels` holds near a point, ascending and once each; null to scan everything. */
function near(levels: CellLevels, x: number, y: number, reach: number): number[] | null {
  const grid = levels.for(reach);
  if (!grid) return null;
  const out: number[] = [];
  grid.near(x, y, reach, (i) => out.push(i));
  out.sort((a, b) => a - b);
  let m = 0;
  for (let k = 0; k < out.length; k++) if (!m || out[k] !== out[m - 1]) out[m++] = out[k];
  out.length = m;
  return out;
}

/** which nodes a grid holds: track nodes (no bus stop, port or BRT station), junctions (three tracks or more), or what
 * the delete tool picks by its node (a station that is no bus stop or port, a flyover junction) */
type NodeSet = "track" | "junction" | "pick";
const KEEP: Record<NodeSet, (n: Node) => boolean> = {
  // T-1464: a bus stop that is no station is no place to snap to
  track: (n) => !n.bus && !n.port && !n.brtStation,
  junction: (n) => n.ports >= 3,
  pick: (n) => (n.platform > 0 && !n.bus && !n.port) || (n.flying && n.ports >= 3),
};
/** some nodes in a grid, ascending by id */
interface NodeLayer { nodes: Node[]; levels: CellLevels }
/** a world's node grid: `base` with the nodes in `gone` passed over and `top` laid over it, at the anchor `anchor` */
interface NodeGrid { anchor: number; base: NodeLayer; gone: ReadonlySet<number>; top: NodeLayer | null }
const nodeGrids: Record<NodeSet, WeakMap<Map<number, Node>, NodeGrid>> = { track: new WeakMap(), junction: new WeakMap(), pick: new WeakMap() };
/** per set, the world whose grid the next state's is laid on: the last one asked of that the clock client built from a
 * state (a click world, map/clickMap.ts, is laid on its state's); held weakly, so no older world stays for it */
const lastMain: Record<NodeSet, WeakRef<WorldView> | null> = { track: null, junction: null, pick: null };

function nodeLayer(nodes: Node[], now: boolean): NodeLayer {
  const a = anchor();
  const box = () => {
    const b = new Float64Array(nodes.length * 4);
    nodes.forEach((n, i) => {
      const [x, y] = toLocal(n.lng, n.lat, a);
      b[i * 4] = b[i * 4 + 2] = x;
      b[i * 4 + 1] = b[i * 4 + 3] = y;
    });
    return b;
  };
  return { nodes, levels: new CellLevels(box, nodes.length, now) };
}

/** `w`'s grid of `set`: laid on the grid of the world it was built from when the diff says which nodes changed */
function nodeGridOf(w: WorldView, set: NodeSet): NodeGrid {
  const grids = nodeGrids[set], av = anchor().version;
  const had = grids.get(w.nodes);
  if (had && had.anchor === av) return had;
  return timed("grid.nodes", () => {
    const side = sideOf(w), from = side ?? lastMain[set]?.deref() ?? null;
    let g: NodeGrid | null = null;
    // (a click world's is laid on its state's, made first if need be; a state's on the last state's that has one)
    const fg = from && from !== w && from.nodes !== w.nodes && (side || grids.get(from.nodes)?.anchor === av) ? nodeGridOf(from, set) : null;
    const d = fg ? worldDiff(from, w) : null;
    if (fg && d) g = layNodes(fg, d.nodes, set);
    g ??= { anchor: av, base: nodeLayer([...w.nodes.values()].filter(KEEP[set]), false), gone: NONE, top: null };
    grids.set(w.nodes, g);
    if (!side) lastMain[set] = new WeakRef(w);
    if (checkPatches()) timed("dev.check.nodeGrid", () => checkNodeGrid(w, set, g));
    return g;
  });
}

/** `g` with the nodes `changed` laid over it; null when that is too many */
function layNodes(g: NodeGrid, changed: WorldDiff["nodes"], set: NodeSet): NodeGrid | null {
  if (!changed.size) return g;
  if (g.gone.size + changed.size > LAID_MAX) return null;
  const keep = KEEP[set], gone = new Set(g.gone), top: Node[] = [];
  for (const id of changed.keys()) gone.add(id);
  for (const n of g.top?.nodes ?? []) if (!changed.has(n.id)) top.push(n);
  for (const [, [, now]] of changed) if (now && keep(now)) top.push(now);
  top.sort((a, b) => a.id - b.id);
  return { anchor: g.anchor, base: g.base, gone, top: top.length ? nodeLayer(top, true) : null };
}

/** the nodes of `g` near a point, ascending by id; null to scan them all */
function nodesIn(g: NodeGrid, lx: number, ly: number, reach: number): Node[] | null {
  const a = near(g.base.levels, lx, ly, reach);
  if (!a) return null;
  const b = g.top ? near(g.top.levels, lx, ly, reach) : [];
  if (!b) return null;
  const out: Node[] = [], base = g.base.nodes, top = g.top?.nodes ?? [];
  let j = 0;
  for (const i of a) {
    const n = base[i];
    if (g.gone.has(n.id)) continue;
    while (j < b.length && top[b[j]].id < n.id) out.push(top[b[j++]]);
    out.push(n);
  }
  while (j < b.length) out.push(top[b[j++]]);
  return out;
}

/** The world's nodes within `reach` local units of (lx, ly), perhaps with others, in the world's
 * order; null to scan them all. */
export function nodesNear(w: WorldView, lx: number, ly: number, reach: number, set: NodeSet = "track"): Node[] | null {
  return nodesIn(nodeGridOf(w, set), lx, ly, reach);
}

/** T-4125: the node of `set` nearest (lx, ly) in local units, the first in the world's order on a tie (as a scan of
 * every node finds it), from the grid's cells near the point; null with none. */
export function nearestNode(w: WorldView, lx: number, ly: number, set: NodeSet): Node | null {
  const keep = KEEP[set];
  const nearest = (nodes: Iterable<Node>, all: boolean) => {
    let best = Infinity, hit: Node | null = null;
    for (const n of nodes) {
      if (all && !keep(n)) continue;
      const [x, y] = toLocal(n.lng, n.lat);
      const d = (x - lx) ** 2 + (y - ly) ** 2;
      if (d < best) { best = d; hit = n; }
    }
    return { best, hit };
  };
  // every node within the reach is among those near, in the world's order: a nearest one within it is the nearest
  for (let reach = 256; reach <= 65536; reach *= 4) {
    const near = nodesNear(w, lx, ly, reach, set);
    if (!near) break;
    const { best, hit } = nearest(near, false);
    if (hit && best <= reach * reach) return hit;
  }
  return nearest(w.nodes.values(), true).hit;
}

/** DEV (`?patch=check`): a laid grid's answers against a grid of every node, near each node laid over and some others */
function checkNodeGrid(w: WorldView, set: NodeSet, g: NodeGrid) {
  if (g.gone === NONE && !g.top) return;
  const ids = [...w.nodes.keys()];
  if (ids.some((id, i) => i > 0 && !(ids[i - 1] < id))) console.error("T-4284 check: the world's nodes are not in ascending ids, which the node grids' order takes");
  const whole: NodeGrid = { anchor: g.anchor, base: nodeLayer([...w.nodes.values()].filter(KEEP[set]), true), gone: NONE, top: null };
  const at = [...(g.top?.nodes ?? []), ...whole.base.nodes.filter((_, i) => i % Math.max(1, Math.floor(whole.base.nodes.length / 32)) === 0)];
  for (const n of at) {
    const [x, y] = toLocal(n.lng, n.lat);
    for (const reach of [64, 900, 6000]) {
      const a = nodesIn(g, x, y, reach), b = nodesIn(whole, x, y, reach);
      // (the same cells hold the same nodes, so the answers match exactly)
      if (a && b && (a.length !== b.length || a.some((q, i) => q !== b[i]))) { console.error(`T-4284 check: the ${set} node grid laid over differs from one built whole near node ${n.id}`); return; }
    }
  }
}

/** An edge with the track points it has near a place: `ks` the offsets into `pts` (x, y, z triples)
 * of the stretches starting there, ascending. */
interface EdgeNear {
  edge: number;
  pts: Float32Array;
  ks: number[];
}

/** the segments of some edges (ascending by id) in a grid: segment i is edge `edges[seg[2i]]`'s from its point `seg[2i + 1]` */
interface Segs { edges: [number, Float32Array][]; seg: Int32Array; levels: CellLevels }
/** a network's edge grid: `base` with the edges in `gone` passed over and `top` laid over it */
interface EdgeGrid { base: Segs; gone: ReadonlySet<number>; top: Segs | null }

function segsOf(edges: [number, Float32Array][], now: boolean): Segs {
  let count = 0;
  for (const [, pts] of edges) count += Math.max(0, Math.floor(pts.length / 3) - 1);
  const seg = new Int32Array(count * 2);
  let c = 0;
  edges.forEach(([, pts], e) => { for (let k = 0; k + 5 < pts.length; k += 3, c++) seg[c * 2] = e, seg[c * 2 + 1] = k; });
  const box = () => {
    const b = new Float64Array(count * 4);
    for (let i = 0; i < count; i++) {
      const pts = edges[seg[i * 2]][1], k = seg[i * 2 + 1];
      b[i * 4] = pts[k], b[i * 4 + 1] = pts[k + 1], b[i * 4 + 2] = pts[k + 3], b[i * 4 + 3] = pts[k + 4];
    }
    return b;
  };
  return { edges, seg, levels: new CellLevels(box, count, now) };
}

const edgeGrids = new WeakMap<NetworkBuffers, EdgeGrid>();
/** the edge grids still to be laid (`carryEdgeGrid`), worked out when a query first wants them: the grid to lay on and
 * every edge changed since it */
const pendingCarry = new WeakMap<NetworkBuffers, { from: EdgeGrid; touched: Set<number> }>();

/** T-1030, T-4284: the edge grid of `was`, the network before an edit, laid over by the edges the edit changed for `n`
 * after it: `moved` (or `n`'s `edgeStep` from `was`) names them; with neither `n`'s grid is built whole. */
export function carryEdgeGrid(was: NetworkBuffers | null, n: NetworkBuffers, check = false, moved?: Iterable<number>) {
  if (!was || was === n || edgeGrids.has(n) || pendingCarry.has(n)) return;
  const step = n.edgeStep;
  const ids = moved ?? (step && was.serial === step.base ? [...step.set, ...step.gone] : null);
  const g = edgeGrids.get(was), p = pendingCarry.get(was);
  if (!ids || (!g && !p)) return;
  const touched = new Set(p?.touched ?? []);
  for (const id of ids) touched.add(id);
  if (touched.size > LAID_MAX) return;
  pendingCarry.set(n, { from: g ?? p!.from, touched });
  if (check || checkPatches()) timed("dev.check.edgeGrid", () => checkEdgeGrid(n));
}

function edgeGridOf(n: NetworkBuffers): EdgeGrid {
  const had = edgeGrids.get(n);
  if (had) return had;
  return timed("grid.edges", () => {
    const p = pendingCarry.get(n);
    pendingCarry.delete(n);
    const g = (p && layEdges(p.from, n, p.touched)) ?? { base: segsOf([...n.edgePts], false), gone: NONE, top: null };
    edgeGrids.set(n, g);
    return g;
  });
}

/** `g` with the edges `touched` laid over it as `n` has them; null when that is too many */
function layEdges(g: EdgeGrid, n: NetworkBuffers, touched: ReadonlySet<number>): EdgeGrid | null {
  if (g.gone.size + touched.size > LAID_MAX) return null;
  const gone = new Set(g.gone), top: [number, Float32Array][] = [];
  for (const id of touched) gone.add(id);
  for (const e of g.top?.edges ?? []) if (!touched.has(e[0])) top.push(e);
  for (const id of touched) { const pts = n.edgePts.get(id); if (pts) top.push([id, pts]); }
  top.sort((a, b) => a[0] - b[0]);
  const t = top.length ? segsOf(top, true) : null;
  if (t && t.levels.count > LAID_MAX) return null;
  return { base: g.base, gone, top: t };
}

function edgesIn(g: EdgeGrid, lx: number, ly: number, reach: number): EdgeNear[] | null {
  const a = near(g.base.levels, lx, ly, reach);
  if (!a) return null;
  const b = g.top ? near(g.top.levels, lx, ly, reach) : [];
  if (!b) return null;
  const out: EdgeNear[] = [];
  const put = (s: Segs, i: number) => {
    const [id, pts] = s.edges[s.seg[i * 2]], k = s.seg[i * 2 + 1];
    const last = out[out.length - 1];
    if (last && last.edge === id) last.ks.push(k);
    else out.push({ edge: id, pts, ks: [k] });
  };
  const base = g.base, top = g.top;
  let j = 0;
  for (const i of a) {
    const id = base.edges[base.seg[i * 2]][0];
    if (g.gone.has(id)) continue;
    while (top && j < b.length && top.edges[top.seg[b[j] * 2]][0] < id) put(top, b[j++]);
    put(base, i);
  }
  while (top && j < b.length) put(top, b[j++]);
  return out;
}

/** The edges with a stretch of track within `reach` local units of (lx, ly), perhaps with others,
 * in `edgePts` order; null to scan them all. */
export function edgesNear(n: NetworkBuffers, lx: number, ly: number, reach: number): EdgeNear[] | null {
  return edgesIn(edgeGridOf(n), lx, ly, reach);
}

/** DEV (`?patch=check`): a laid grid's answers against a grid of every edge, at each laid edge's points and some others */
function checkEdgeGrid(n: NetworkBuffers) {
  const g = edgeGridOf(n);
  if (g.gone === NONE && !g.top) return;
  const all = [...n.edgePts];
  if (all.some(([id], i) => i > 0 && !(all[i - 1][0] < id))) console.error("T-4284 check: the network's edges are not in ascending ids, which the edge grid's order takes");
  const whole: EdgeGrid = { base: segsOf(all, true), gone: NONE, top: null };
  const at: [number, number][] = [];
  for (const [, pts] of g.top?.edges ?? []) for (let k = 0; k + 2 < pts.length; k += 3 * Math.max(1, Math.floor(pts.length / 24))) at.push([pts[k], pts[k + 1]]);
  for (let e = 0; e < all.length; e += Math.max(1, Math.floor(all.length / 32))) { const pts = all[e][1]; if (pts.length >= 2) at.push([pts[0], pts[1]]); }
  const key = (r: EdgeNear[] | null) => r && r.map((x) => `${x.edge}:${x.ks.join(",")}`).join(" ");
  for (const [x, y] of at)
    for (const reach of [64, 900, 6000]) {
      const a = edgesIn(g, x, y, reach), b = edgesIn(whole, x, y, reach);
      if (a && b && key(a) !== key(b)) { console.error(`T-4284 check: the edge grid laid over differs from one built whole at ${x.toFixed(0)}, ${y.toFixed(0)}`); return; }
    }
}
