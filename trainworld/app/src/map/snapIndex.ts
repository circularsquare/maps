// Grids over the network's nodes and track points (T-724), so the build tools' snap and a station's
// track direction look only at what is near the pointer instead of projecting every node and
// every point of every edge on each pointer move (7-17 ms on the real New York save). Candidates
// come back in the order a full scan meets them, so the nearest-first ties fall the same way.

import type { WorldView } from "../game/types";
import { CellLevels } from "./cellIndex";
import { anchor, toLocal } from "./geo";
import type { NetworkBuffers } from "./network";

type Node = WorldView["nodes"] extends Map<number, infer N> ? N : never;

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

const nodeGrids = new WeakMap<Map<number, Node>, { anchor: number; nodes: Node[]; levels: CellLevels }>();

/** The world's nodes within `reach` local units of (lx, ly), perhaps with others, in the world's
 * order; null to scan them all. */
export function nodesNear(w: WorldView, lx: number, ly: number, reach: number): Node[] | null {
  let g = nodeGrids.get(w.nodes);
  if (!g || g.anchor !== anchor().version) {
    const nodes = [...w.nodes.values()], a = anchor();
    const box = () => {
      const b = new Float64Array(nodes.length * 4);
      nodes.forEach((n, i) => {
        const [x, y] = toLocal(n.lng, n.lat, a);
        b[i * 4] = b[i * 4 + 2] = x;
        b[i * 4 + 1] = b[i * 4 + 3] = y;
      });
      return b;
    };
    nodeGrids.set(w.nodes, (g = { anchor: a.version, nodes, levels: new CellLevels(box, nodes.length) }));
  }
  const idx = near(g.levels, lx, ly, reach);
  return idx && idx.map((i) => g!.nodes[i]);
}

/** An edge with the track points it has near a place: `ks` the offsets into `pts` (x, y, z triples)
 * of the stretches starting there, ascending. */
export interface EdgeNear {
  edge: number;
  pts: Float32Array;
  ks: number[];
}

const edgeGrids = new WeakMap<NetworkBuffers, { edges: [number, Float32Array][]; seg: Int32Array; levels: CellLevels }>();

/** T-1030: the edge grid of the network before an edit carried over to the one after it, the edges
 * whose points are the same object keeping their place in it. */
export function carryEdgeGrid(was: NetworkBuffers | null, n: NetworkBuffers, check = false) {
  // only from a grid already worked out: a chain of pending ones would hold every older network
  if (!was || edgeGrids.has(n) || !edgeGrids.has(was)) return;
  // worked out when a query first wants it, not in the edit's task
  pendingCarry.set(n, () => carryNow(was, n, check));
  if (check) edgeGridOf(n);
}
/** the edge grids still to be carried over (`carryEdgeGrid`) */
const pendingCarry = new WeakMap<NetworkBuffers, () => void>();
function edgeGridOf(n: NetworkBuffers) {
  const p = pendingCarry.get(n);
  if (p) {
    pendingCarry.delete(n);
    p();
  }
  return edgeGrids.get(n);
}
function carryNow(was: NetworkBuffers, n: NetworkBuffers, check: boolean) {
  const g = edgeGridOf(was);
  if (!g || edgeGrids.has(n)) return;
  const oldAt = new Map<Float32Array, number>();
  let o = 0;
  for (const [, pts] of g.edges) oldAt.set(pts, o), (o += Math.max(0, Math.floor(pts.length / 3) - 1));
  const edges = [...n.edgePts];
  const runs: number[] = [];
  let count = 0;
  for (const [, pts] of edges) {
    const c = Math.max(0, Math.floor(pts.length / 3) - 1);
    if (!c) continue;
    const from = oldAt.get(pts) ?? -1, k = runs.length - 3;
    if (k >= 0 && (from < 0 ? runs[k + 2] < 0 : runs[k + 2] >= 0 && runs[k + 2] + runs[k + 1] === from)) runs[k + 1] += c;
    else runs.push(count, c, from);
    count += c;
  }
  const seg = new Int32Array(count * 2);
  let c = 0;
  edges.forEach(([, pts], e) => { for (let k = 0; k + 5 < pts.length; k += 3, c++) seg[c * 2] = e, seg[c * 2 + 1] = k; });
  const box = () => {
    const b = new Float64Array(count * 4);
    for (let i = 0; i < count; i++) itemBox(i, b, i * 4);
    return b;
  };
  const itemBox = (i: number, b: Float64Array, at: number) => {
    const pts = edges[seg[i * 2]][1], k = seg[i * 2 + 1];
    b[at] = pts[k], b[at + 1] = pts[k + 1], b[at + 2] = pts[k + 3], b[at + 3] = pts[k + 4];
  };
  const levels = CellLevels.carried(g.levels, new Int32Array(runs), box, count, itemBox);
  edgeGrids.set(n, { edges, seg, levels });
  const bad = check ? levels.verify() : [];
  if (bad.length) console.error(`T-1030 check: the edge grid carried over differs from one built anew at ${bad.join(", ")}`);
}

/** The edges with a stretch of track within `reach` local units of (lx, ly), perhaps with others,
 * in `edgePts` order; null to scan them all. */
export function edgesNear(n: NetworkBuffers, lx: number, ly: number, reach: number): EdgeNear[] | null {
  let g = edgeGridOf(n);
  if (!g) {
    const edges = [...n.edgePts];
    let count = 0;
    for (const [, pts] of edges) count += Math.max(0, Math.floor(pts.length / 3) - 1);
    const seg = new Int32Array(count * 2);
    const box = () => {
      const b = new Float64Array(count * 4);
      let c = 0;
      edges.forEach(([, pts], e) => {
        for (let k = 0; k + 5 < pts.length; k += 3, c++) {
          b[c * 4] = pts[k], b[c * 4 + 1] = pts[k + 1], b[c * 4 + 2] = pts[k + 3], b[c * 4 + 3] = pts[k + 4];
          seg[c * 2] = e, seg[c * 2 + 1] = k;
        }
      });
      return b;
    };
    edgeGrids.set(n, (g = { edges, seg, levels: new CellLevels(box, count) }));
  }
  const idx = near(g.levels, lx, ly, reach);
  if (!idx) return null;
  const out: EdgeNear[] = [];
  for (const i of idx) {
    const e = g.seg[i * 2], k = g.seg[i * 2 + 1];
    const last = out[out.length - 1];
    if (last && last.edge === g.edges[e][0]) last.ks.push(k);
    else out.push({ edge: g.edges[e][0], pts: g.edges[e][1], ks: [k] });
  }
  return out;
}
