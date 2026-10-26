// Words for places on the network, for the inspectors (T-031): the station a stretch of track
// leads to, a compass direction, the station nearest a point. Pack metres, y north.

import { stationsByNum, world } from "./world";
import { groundDist } from "./coords";

const COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"];

export function compass(dx: number, dy: number): string {
  const a = Math.atan2(dy, dx);
  return COMPASS[(Math.round(a / (Math.PI / 4)) + 8) % 8];
}

/** The station nearest a point, or null. */
export function nearestStation(x: number, y: number): { id: string; name: string; d: number; lng: number; lat: number } | null {
  let best: { id: string; name: string; d: number; lng: number; lat: number } | null = null;
  for (const s of world.value?.save.stations ?? []) {
    const d = groundDist(s.x, s.y, x, y);
    if (!best || d < best.d) best = { id: s.id, name: s.name, d, lng: s.lng, lat: s.lat };
  }
  return best;
}

/** T-865: the name of the station nearest a point within `within` metres, else "". */
export const stationNameNear = (x: number, y: number, within = 3000): string => {
  const s = nearestStation(x, y);
  return s && s.d <= within ? s.name : "";
};

/** The first station along track leaving a node by edge `edge` (from its end `end`, 0 = a),
 * carrying on through plain nodes; null if the track ends or splits first. */
export function stationTowards(edge: number, end: number): string | null {
  const w = world.value;
  if (!w) return null;
  const byNode = new Map<number, number[]>();
  for (const e of w.edges) for (const n of [e.a, e.b]) byNode.set(n, [...(byNode.get(n) ?? []), e.id]);
  const byNum = stationsByNum(w);
  let e = w.edges.find((x) => x.id === edge);
  let from = e ? (end === 0 ? e.a : e.b) : -1;
  for (let k = 0; e && k < 200; k++) {
    const next = e.a === from ? e.b : e.a;
    const name = byNum.get(next)?.name;
    if (name) return name;
    const others = (byNode.get(next) ?? []).filter((x) => x !== e!.id);
    if (others.length !== 1) return null;
    from = next;
    e = w.edges.find((x) => x.id === others[0]);
  }
  return null;
}
