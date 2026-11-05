// Draggable points of blueprint track besides its nodes (T-1869): the middle of each curve, and both
// ends of a blueprint station's platform. map/endMarkers.ts draws them and map/nodeDrag.ts grabs
// them, both through these lists, so the mark drawn and the press that drags agree.

import { packToLngLat } from "../game/coords";
import { world } from "../game/world";
import { queryClock } from "../workers/clockClient";
import { anchor, toLocal } from "./geo";
import type { Overlay } from "./overlay";
import { shapePoints, PLATFORM_POINTS } from "./stationGeometry";

/** The middle of one curve of a blueprint edge: its PI `i`. */
export interface CurveHandle {
  edge: number;
  i: number;
  /** the edge's end nodes */
  a: number;
  b: number;
  /** the curve's middle, local units at `anchorV` */
  lx: number;
  ly: number;
}

/** below this zoom no draggable point is marked (map/endMarkers.ts) or grabbed besides the nodes */
export const MARKS_ZOOM = 9;
/** a press this near a node, px, grabs it: the node drag, the pick's drag and the select tool's marks agree */
export const NODE_HIT_PX = 9;
/** a curve's handle is left out this near either end of its edge, px: the node's square is there */
const NEAR_END_PX = 14;

interface EdgeHandles {
  /** the network's drawn polyline of the edge when they were read: a new one means a new shape */
  pts: Float32Array | undefined;
  /** the curves' middles, pack metres */
  mids: { i: number; x: number; y: number }[];
}

let byEdge = new Map<number, EdgeHandles>();
let list: CurveHandle[] = [];
let listKey = "";
let loadingKey = "";
const listeners: (() => void)[] = [];

/** Call `f` when curve handles arrive after a change. */
export function onCurveHandles(f: () => void) {
  listeners.push(f);
}

/** Every blueprint curve's handle, as last read; a changed network starts a read and gives what is
 * known meanwhile (the changed edges' handles left out until read). */
export function curveHandles(overlay: Overlay): CurveHandle[] {
  const w = world.peek(), net = overlay.renderer.net;
  if (!w || !net) return [];
  const av = anchor().version;
  const key = `${w.version}:${av}`;
  if (key === listKey) return list;
  const edges = w.edges.filter((e) => !e.built && !e.crossover && net.edgePts.has(e.id));
  const stale: number[] = [];
  const next = new Map<number, EdgeHandles>();
  for (const e of edges) {
    const pts = net.edgePts.get(e.id), had = byEdge.get(e.id);
    if (had && had.pts === pts) next.set(e.id, had);
    else stale.push(e.id);
  }
  byEdge = next;
  list = build(edges.map((e) => [e.id, e.a, e.b]));
  if (!stale.length) listKey = key;
  else if (loadingKey !== key) {
    loadingKey = key;
    void Promise.all(stale.map((id) => queryClock({ what: "edgePis", edge: id }).then((d) => [id, d] as const))).then((got) => {
      if (loadingKey !== key) return;
      loadingKey = "";
      const w2 = world.peek(), net2 = overlay.renderer.net;
      if (!w2 || !net2 || `${w2.version}:${anchor().version}` !== key) return;
      for (const [id, d] of got) byEdge.set(id, { pts: net2.edgePts.get(id), mids: d ? mids(d) : [] });
      list = build(w2.edges.filter((e) => byEdge.has(e.id)).map((e) => [e.id, e.a, e.b]));
      listKey = key;
      for (const f of listeners) f();
    });
  }
  return list;
}

/** `TrackApi.edge_pis`' curves (stride 8 after a flag: x, y, set radius, level, radius used, km/h,
 * middle x, y); a PI whose curve is nil has no handle. */
function mids(d: Float64Array): EdgeHandles["mids"] {
  const out: EdgeHandles["mids"] = [];
  for (let k = 1, i = 0; k + 8 <= d.length; k += 8, i++) if (d[k + 4] > 0) out.push({ i, x: d[k + 6], y: d[k + 7] });
  return out;
}

function build(edges: [number, number, number][]): CurveHandle[] {
  const out: CurveHandle[] = [];
  for (const [edge, a, b] of edges) {
    for (const m of byEdge.get(edge)?.mids ?? []) {
      const [lx, ly] = toLocal(...packToLngLat(m.x, m.y));
      out.push({ edge, i: m.i, a, b, lx, ly });
    }
  }
  return out;
}

/** The curve handles drawn now, with their place on screen (px). */
export function visibleCurves(overlay: Overlay): (CurveHandle & { x: number; y: number })[] {
  const w = world.peek();
  if (!w) return [];
  const out: (CurveHandle & { x: number; y: number })[] = [];
  const near = NEAR_END_PX * NEAR_END_PX;
  const at = (id: number) => {
    const n = w.nodes.get(id);
    return n ? overlay.toScreen(...toLocal(n.lng, n.lat)) : null;
  };
  for (const h of curveHandles(overlay)) {
    const [x, y] = overlay.toScreen(h.lx, h.ly);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const pa = at(h.a), pb = at(h.b);
    if (pa && (pa[0] - x) ** 2 + (pa[1] - y) ** 2 < near) continue;
    if (pb && (pb[0] - x) ** 2 + (pb[1] - y) ** 2 < near) continue;
    out.push({ ...h, x, y });
  }
  return out;
}

let movable = { version: -1, nodes: new Set<number>() };
/** The station nodes that can be dragged: no constructed track there and the station unbuilt
 * (map/nodeDrag.ts' rule). */
function movableStations(): Set<number> {
  const w = world.peek();
  if (!w) return movable.nodes;
  if (movable.version !== w.version) {
    const nodes = new Set<number>();
    for (const s of w.save.stations) if (!s.built && w.nodes.get(s.num)?.builtPorts === 0) nodes.add(s.num);
    movable = { version: w.version, nodes };
  }
  return movable.nodes;
}

/** Both ends of every movable blueprint station's platform on screen (px), where stations are drawn
 * as platforms (from STATION_BOX_ZOOM); at a dot both ends are the node. */
export function stationEnds(overlay: Overlay): { node: number; x: number; y: number }[] {
  const r = overlay.renderer, n = r.net;
  if (!n || !r.stationStyle.boxes || r.stationStyle.hidden) return [];
  const out: { node: number; x: number; y: number }[] = [];
  const ok = movableStations();
  if (!ok.size) return out;
  const project = (x: number, y: number) => overlay.toScreen(x, y);
  for (let i = 0; i < n.stationCount; i++) {
    const node = Number(n.stationIds[i]);
    if (!ok.has(node)) continue;
    const s = r.stationShapeOf(i, project);
    if (s.dot > 0) continue;
    // the centre line's ends carried straight on to the box's flat ends (stationPath)
    const q = shapePoints(s), P = PLATFORM_POINTS;
    let total = 0;
    for (let k = 0; k + 1 < P; k++) total += Math.hypot(q[k * 2 + 2] - q[k * 2], q[k * 2 + 3] - q[k * 2 + 1]);
    const ext = s.halfLength - total / 2;
    for (const [k, j] of [[0, 1], [P - 1, P - 2]]) {
      const dx = q[k * 2] - q[j * 2], dy = q[k * 2 + 1] - q[j * 2 + 1], l = Math.hypot(dx, dy) || 1;
      const a = q[k * 2] + (dx / l) * ext, b = q[k * 2 + 1] + (dy / l) * ext;
      out.push({ node, x: s.x + a * s.ux + b * s.nx, y: s.y + a * s.uy + b * s.ny });
    }
  }
  return out;
}
