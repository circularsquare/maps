// T-3540: a station on single track with no passing loop has one platform, drawn to one side of its track: the outside.
import type { ClockEdge } from "../workers/protocol";

/** How far along single track a station looks for the double track it leaves (`stationOutside` rule 1), m. */
const JOIN_SEARCH_M = 3000;

/**
 * The side of a single-track station's platform axis (ax, ay; local units, y down) its one platform is
 * drawn on: 1 the + side (right of the axis), -1 the other. The outside, in this order:
 * 1. Where the track, followed through plain nodes, leaves one track of a double (its `join`), the side
 *    that track is on: so each leg of a double split into two singles (a one-way street pair) has its
 *    platform away from the other leg.
 * 2. On one-way track, the side trains keep to (the right, or the left where they keep left).
 * 3. Otherwise the side facing south-east (east of north-south track, south of east-west), or south-west
 *    where trains keep left, so a left-hand game draws the mirror image of a right-hand one.
 * `at(node)`: the edges at a node; `way(node, edge)`: the edge's points from the node out (x, y, z).
 */
export function stationOutside(node: number, platformEdge: ClockEdge, at: (n: number) => readonly ClockEdge[], way: (node: number, edge: number) => ArrayLike<number>, ax: number, ay: number, keepLeft: boolean): 1 | -1 {
  // the way an edge leaves `node`, unit x, y
  const out = (n: number, id: number): [number, number] => {
    const p = way(n, id);
    for (let i = 3; i + 1 < p.length; i += 3) {
      const dx = p[i] - p[0], dy = p[i + 1] - p[1], l = Math.hypot(dx, dy);
      if (l > 1e-6) return [dx / l, dy / l];
    }
    return [ax, ay];
  };
  // travelling (tx, ty) past the station, its right (or left) side as a side of the axis
  const sideOf = (tx: number, ty: number, right: boolean): 1 | -1 => ((tx * ax + ty * ay >= 0) === right ? 1 : -1);
  let best = Infinity, side: 0 | 1 | -1 = 0;
  for (const e0 of at(node)) {
    let cur = e0, from = node, dist = 0;
    for (let k = 0; k < 64 && dist < JOIN_SEARCH_M && dist < best; k++) {
      const far = cur.a === from ? cur.b : cur.a;
      // (left 1, right 2, looking out from the junction along the piece: towards the station)
      const j = (cur.a === far ? cur.join : cur.join >> 2) & 3;
      dist += cur.lengthM;
      if (j) {
        const [wx, wy] = out(node, e0.id);
        best = dist;
        side = sideOf(-wx, -wy, j === 2);
        break;
      }
      const next = at(far).filter((o) => o.id !== cur.id);
      if (next.length !== 1 || next[0].tracks !== 1 || next[0].crossover) break;
      from = far;
      cur = next[0];
    }
  }
  if (side) return side;
  const right = !keepLeft, e = platformEdge;
  if (e.oneWay) {
    const [wx, wy] = out(node, e.id), along = (e.a === node) === (e.oneWay === 1);
    return along ? sideOf(wx, wy, right) : sideOf(-wx, -wy, right);
  }
  // the side facing south-east (south-west where trains keep left); the + side's normal is (-ay, ax)
  return -ay * (keepLeft ? -1 : 1) + ax >= 0 ? 1 : -1;
}
