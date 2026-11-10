// T-2992: where two bus lines change. Stops of two lines within demand's station merge of each other
// (`DEMAND_MERGE_M`, the kernel's `station_merge_m`) are one demand station, so riders change there
// with no walk, as at a train station a stop belongs to; the map draws them a transfer line, Manual
// stop editing snaps to them, and the bus tool lights the lines a route will meet (T-2993).

import { DEMAND_MERGE_M } from "./complexes";
import type { ClockBus } from "../workers/protocol";

/** Two bus stops of different lines this near each other (ground m) are one place. */
export const BUS_XFER_M = DEMAND_MERGE_M;

type LL = readonly [number, number];
type Stop = ClockBus["stops"][number];

/** Ground metres between two longitude, latitude points (flat, for short distances). */
export function groundM(a: LL, b: LL): number {
  return Math.hypot((a[0] - b[0]) * 111_320 * Math.cos((a[1] * Math.PI) / 180), (a[1] - b[1]) * 111_320);
}

/** Where demand puts a stop: a train station's own place, else the stop's. */
export const stopPlace = (s: Stop): LL => s.at ?? [s.lng, s.lat];

/** The transfers between bus lines' stops: each pair of stops of different lines within
 * `BUS_XFER_M` of each other, not at one station and not on one spot, as the line from one stop to
 * the other (from a stop to the station when the other is a station's): the first stop's line, then
 * lng, lat, lng, lat, flat. */
export function busStopLinks(lines: readonly { line: number; stops: readonly Stop[] }[]): number[] {
  const cell = BUS_XFER_M / 111_320;
  const grid = new Map<string, { line: number; s: Stop }[]>();
  const key = (p: LL) => `${Math.floor(p[0] * Math.cos((p[1] * Math.PI) / 180) / cell)},${Math.floor(p[1] / cell)}`;
  for (const l of lines) for (const s of l.stops) {
    const k = key(stopPlace(s));
    let v = grid.get(k);
    if (!v) grid.set(k, (v = []));
    v.push({ line: l.line, s });
  }
  const out: number[] = [], seen = new Set<string>();
  for (const l of lines) for (const s of l.stops) {
    const p = stopPlace(s), cx = Math.floor(p[0] * Math.cos((p[1] * Math.PI) / 180) / cell), cy = Math.floor(p[1] / cell);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const o of grid.get(`${cx + dx},${cy + dy}`) ?? []) {
        if (o.line === l.line || (s.station >= 0 && o.s.station >= 0)) continue;
        const q = stopPlace(o.s), d = groundM(p, q);
        if (d > BUS_XFER_M || d < 1) continue;
        // the plain stop's line to the other's place, once a pair
        const [from, to] = s.station >= 0 ? [o.s, s] : [s, o.s];
        const id = `${from.id}|${to.id}`, back = `${to.id}|${from.id}`;
        if (seen.has(id) || seen.has(back)) continue;
        seen.add(id);
        const f: LL = [from.lng, from.lat], t = to.station >= 0 ? stopPlace(to) : ([to.lng, to.lat] as LL);
        out.push(s.station >= 0 ? o.line : l.line, f[0], f[1], t[0], t[1]);
      }
    }
  }
  return out;
}

/** `bus::STATION_REACH_M`: a train station this near a route is a stop of it. */
const STATION_REACH_M = 200;
/** A route passing this near one of a new route's crossing stops is the route it crosses there, m. */
const CROSS_PASS_M = 10;

/** The ground metres from a point to a route (lng, lat pairs flat). */
function toRoute(p: LL, ll: readonly number[]): number {
  const kx = 111_320 * Math.cos((p[1] * Math.PI) / 180), ky = 111_320;
  let best = Infinity;
  for (let i = 0; i + 3 < ll.length; i += 2) {
    const ax = (ll[i] - p[0]) * kx, ay = (ll[i + 1] - p[1]) * ky, dx = (ll[i + 2] - ll[i]) * kx, dy = (ll[i + 3] - ll[i + 1]) * ky, l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
    best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy));
  }
  return best;
}

/** T-2993: what a route will connect with, from its stops as the sim plans them (`busQuote`; kind 1
 * a station's, 3 a crossing): the train stations it stops at (the nearest station within
 * `STATION_REACH_M` of each station stop), and the other bus lines it will change with: one with a
 * stop within `BUS_XFER_M` of one of its stops, or whose route passes one of its crossing stops; and
 * `links`, the transfer lines a built line would have: each station stop to its station and each
 * stop to another line's stop near it (not on one spot), lng, lat, lng, lat flat. */
export function routeMeets(stops: readonly { kind: number; lng: number; lat: number }[], stations: readonly { id: string; lng: number; lat: number }[], lines: readonly { line: number; stops: readonly Stop[]; routeLL: readonly number[] }[]): { stations: string[]; buses: number[]; links: number[] } {
  const st = new Set<string>(), buses = new Set<number>(), links: number[] = [];
  for (const s of stops) {
    if (s.kind !== 1) continue;
    let best: { id: string; lng: number; lat: number } | null = null, d = STATION_REACH_M;
    for (const x of stations) {
      const m = groundM([s.lng, s.lat], [x.lng, x.lat]);
      if (m <= d) (d = m), (best = x);
    }
    if (best) st.add(best.id), links.push(s.lng, s.lat, best.lng, best.lat);
  }
  for (const l of lines) {
    let meets = false;
    for (const s of stops) {
      if (s.kind === 3 && toRoute([s.lng, s.lat], l.routeLL) <= CROSS_PASS_M) meets = true;
      if (s.kind === 1) continue;
      for (const o of l.stops) {
        const q = stopPlace(o), m = groundM([s.lng, s.lat], q);
        if (m > BUS_XFER_M) continue;
        meets = true;
        if (m >= 1) links.push(s.lng, s.lat, q[0], q[1]);
      }
    }
    if (meets) buses.add(l.line);
  }
  return { stations: [...st], buses: [...buses], links };
}
