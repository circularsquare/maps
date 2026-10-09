// Words for places on the network, for the inspectors (T-031): the station a stretch of track
// leads to, a compass direction, the station nearest a point. Pack metres, y north.

import { world } from "./world";

const COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"];

export function compass(dx: number, dy: number): string {
  const a = Math.atan2(dy, dx);
  return COMPASS[(Math.round(a / (Math.PI / 4)) + 8) % 8];
}

/** The station nearest a point, or null. */
export function nearestStation(x: number, y: number): { name: string; d: number } | null {
  let best: { name: string; d: number } | null = null;
  for (const s of world.value?.save.stations ?? []) {
    const d = Math.hypot(s.x - x, s.y - y);
    if (!best || d < best.d) best = { name: s.name, d };
  }
  return best;
}

/** The first station along track leaving a node by edge `edge` (from its end `end`, 0 = a),
 * carrying on through plain nodes; null if the track ends or splits first. */
export function stationTowards(edge: number, end: number): string | null {
  const w = world.value;
  if (!w) return null;
  const byNode = new Map<number, number[]>();
  for (const e of w.edges) for (const n of [e.a, e.b]) byNode.set(n, [...(byNode.get(n) ?? []), e.id]);
  const names = new Map(w.save.stations.map((s) => [s.num, s.name]));
  let e = w.edges.find((x) => x.id === edge);
  let from = e ? (end === 0 ? e.a : e.b) : -1;
  for (let k = 0; e && k < 200; k++) {
    const next = e.a === from ? e.b : e.a;
    const name = names.get(next);
    if (name) return name;
    const others = (byNode.get(next) ?? []).filter((x) => x !== e!.id);
    if (others.length !== 1) return null;
    from = next;
    e = w.edges.find((x) => x.id === others[0]);
  }
  return null;
}
