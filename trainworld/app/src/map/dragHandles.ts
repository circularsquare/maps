// Draggable points of blueprint track besides its nodes (T-1869): both ends of a blueprint station's
// platform. map/endMarkers.ts draws them and map/nodeDrag.ts grabs them, both through this list, so
// the mark drawn and the press that drags agree.

import { world } from "../game/world";
import type { Overlay } from "./overlay";
import { shapePoints, PLATFORM_POINTS } from "./stationGeometry";

/** below this zoom no draggable point is marked (map/endMarkers.ts) or grabbed besides the nodes */
export const MARKS_ZOOM = 9;
/** a press this near a node, px, grabs it: the node drag, the pick's drag and the select tool's marks agree */
export const NODE_HIT_PX = 9;

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
