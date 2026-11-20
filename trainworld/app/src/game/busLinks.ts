// T-2992: where two bus lines change. Stops of two lines within demand's station merge of each other
// (`DEMAND_MERGE_M`, the kernel's `station_merge_m`) are one demand station, so riders change there
// with no walk, as at a train station a stop belongs to; Manual stop editing snaps to them, and the bus
// tool lights the lines a route will meet (T-2993). T-3890: `busPlaces` groups the stops into the places
// demand sees (the bus stop panel's scope), with the places a walk away and the connectors the map
// draws between stops that connect but are not on one spot.

import { DEMAND_MERGE_M } from "./complexes";
import type { ClockBus } from "../workers/protocol";
import type { WorldView } from "./types";

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

/** The transfers from bus stops to the train stations they change at: each pair of stops of different
 * lines within `BUS_XFER_M` of each other where one is a station's and the other is not, as the line
 * from the plain stop to the station: the plain stop's line, then lng, lat, lng, lat, flat. Two plain
 * stops are joined by a connector instead (`busPlaces`, T-3890). */
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
        if (o.line === l.line || (s.station >= 0) === (o.s.station >= 0)) continue;
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

/** Demand's change walk between stations (`Params::xfer_walk_m`), ground m. */
export const BUS_WALK_M = 300;

/** A bus line's stop, as the bus stop panel and the connectors read it. */
export interface PlaceStop {
  id: number;
  /** its line's number and id */
  line: number;
  lineId: string;
  /** where it is drawn */
  lng: number;
  lat: number;
  /** a train station's node, else -1 */
  station: number;
  built: boolean;
}

/** One place for demand: bus stops of different lines and train stations within `BUS_XFER_M` of each
 * other, chained, as the kernel's station merge (`merge_stations`) joins them. */
export interface BusPlace {
  stops: PlaceStop[];
  /** the train stations in it (save ids) */
  stations: string[];
  /** every line stopping here, bus and train (ids) */
  lines: string[];
  /** the places and train stations a walk away (`BUS_WALK_M`) with a line this place has not:
   * another place (its index, -1 for train stations no bus stops at), those stations, the shortest
   * walk between the two, ground m; nearest first */
  walks: { place: number; stations: string[]; m: number }[];
}

export interface BusPlaces {
  places: BusPlace[];
  /** each stop's place, by stop id */
  placeOf: Map<number, number>;
  /** the connectors between stops that connect but are drawn apart: within a place by the shortest
   * joins between its plain stops (`walk` false), and between two places a walk apart from their
   * nearest stops where no bus line stops at both (`walk` true) */
  links: { a: PlaceStop; b: PlaceStop; walk: boolean }[];
}

const placesMemo = new WeakMap<WorldView, BusPlaces>();

/** T-3890: the world's bus stops grouped into places (kept per world). Stops of one line are never
 * merged with each other, nor are two lines' stops on one spot joined by a connector. */
export function busPlaces(w: WorldView): BusPlaces {
  const kept = placesMemo.get(w);
  if (kept) return kept;
  type Node = { stop?: PlaceStop; station?: string; p: LL; lines: string[] };
  const nodes: Node[] = [];
  const trainAt = new Map<string, string[]>(), busLines = new Set<string>();
  for (const l of w.save.lines) {
    const b = w.lineStats[l.id]?.bus;
    if (b) {
      if (b.ferry) continue;
      busLines.add(l.id);
      for (const s of b.stops) nodes.push({ stop: { id: s.id, line: l.num, lineId: l.id, lng: s.lng, lat: s.lat, station: s.station, built: b.built }, p: stopPlace(s), lines: [l.id] });
    } else for (const st of new Set(l.stops)) (trainAt.get(st) ?? trainAt.set(st, []).get(st)!).push(l.id);
  }
  const out: BusPlaces = { places: [], placeOf: new Map(), links: [] };
  placesMemo.set(w, out);
  if (!nodes.length) return out;
  for (const st of w.save.stations) nodes.push({ station: st.id, p: [st.lng, st.lat], lines: trainAt.get(st.id) ?? [] });
  // rows of `BUS_WALK_M` of latitude, each cut into columns as wide on the ground at its middle
  const cell = BUS_WALK_M / 111_320;
  const rowOf = (lat: number) => Math.floor(lat / cell);
  const colOf = (lng: number, row: number) => Math.floor((lng * Math.cos(((row + 0.5) * cell * Math.PI) / 180)) / cell);
  const grid = new Map<string, number[]>();
  nodes.forEach((n, i) => {
    const r = rowOf(n.p[1]), k = `${r},${colOf(n.p[0], r)}`;
    (grid.get(k) ?? grid.set(k, []).get(k)!).push(i);
  });
  const parent = Int32Array.from(nodes, (_, i) => i);
  const find = (i: number) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const sameLine = (a: Node, b: Node) => !!a.stop && !!b.stop && a.stop.line === b.stop.line;
  // every pair within the walk, once
  const pairs: [number, number, number][] = [];
  nodes.forEach((n, i) => {
    const r = rowOf(n.p[1]);
    for (let dr = -1; dr <= 1; dr++) {
      const c = colOf(n.p[0], r + dr);
      for (let dc = -1; dc <= 1; dc++) for (const j of grid.get(`${r + dr},${c + dc}`) ?? []) {
        if (j <= i) continue;
        const d = groundM(n.p, nodes[j].p);
        if (d <= BUS_WALK_M) pairs.push([i, j, d]);
      }
    }
  });
  for (const [i, j, d] of pairs) if (d <= BUS_XFER_M && !sameLine(nodes[i], nodes[j])) parent[find(i)] = find(j);
  // each group's stations and lines; the places are the groups holding a stop
  const groups = new Map<number, { stations: string[]; lines: string[]; members: number[]; stop: boolean }>();
  nodes.forEach((n, i) => {
    const root = find(i);
    const g = groups.get(root) ?? groups.set(root, { stations: [], lines: [], members: [], stop: false }).get(root)!;
    g.members.push(i);
    if (n.stop) g.stop = true;
    else g.stations.push(n.station!);
    for (const l of n.lines) if (!g.lines.includes(l)) g.lines.push(l);
  });
  const placeAt = new Map<number, number>();
  const members: number[][] = [];
  for (const [root, g] of groups) {
    if (!g.stop) continue;
    placeAt.set(root, out.places.length);
    const stops = g.members.flatMap((i) => nodes[i].stop ?? []);
    for (const s of stops) out.placeOf.set(s.id, out.places.length);
    out.places.push({ stops, stations: g.stations, lines: g.lines, walks: [] });
    members.push(g.members);
  }
  // a walk away: per pair of groups the shortest walk, and the nearest two bus stops for the connector
  const near = new Map<string, { m: number; stop?: [number, number, number] }>();
  for (const [i, j, d] of pairs) {
    const ri = find(i), rj = find(j);
    if (ri === rj || (!placeAt.has(ri) && !placeAt.has(rj))) continue;
    const k = ri < rj ? `${ri},${rj}` : `${rj},${ri}`;
    const e = near.get(k) ?? near.set(k, { m: Infinity }).get(k)!;
    e.m = Math.min(e.m, d);
    if (nodes[i].stop && nodes[j].stop && (!e.stop || d < e.stop[2])) e.stop = [i, j, d];
  }
  for (const [k, e] of near) {
    const [ri, rj] = k.split(",").map(Number);
    let useful = false;
    for (const [from, to] of [[ri, rj], [rj, ri]] as const) {
      const at = placeAt.get(from), lf = groups.get(from)!.lines, gt = groups.get(to)!;
      if (at === undefined || !gt.lines.some((l) => !lf.includes(l))) continue;
      useful = true;
      out.places[at].walks.push({ place: placeAt.get(to) ?? -1, stations: gt.stations, m: e.m });
    }
    // no connector where a bus line already runs between the two
    const gi = groups.get(ri)!, gj = groups.get(rj)!;
    const shared = gi.lines.some((l) => busLines.has(l) && gj.lines.includes(l));
    if (useful && e.stop && !shared) out.links.push({ a: nodes[e.stop[0]].stop!, b: nodes[e.stop[1]].stop!, walk: true });
  }
  for (const pl of out.places) pl.walks.sort((a, b) => a.m - b.m);
  // within a place: the shortest joins between its plain stops of different lines
  members.forEach((m) => {
    const plain = m.filter((i) => nodes[i].stop && nodes[i].stop!.station < 0);
    if (plain.length < 2) return;
    const joins: [number, number, number][] = [];
    for (let a = 0; a < plain.length; a++) for (let b = a + 1; b < plain.length; b++) {
      const x = nodes[plain[a]], y = nodes[plain[b]];
      if (sameLine(x, y)) continue;
      const d = groundM([x.stop!.lng, x.stop!.lat], [y.stop!.lng, y.stop!.lat]);
      if (d <= BUS_XFER_M) joins.push([d, plain[a], plain[b]]);
    }
    joins.sort((p, q) => p[0] - q[0]);
    const up = new Map<number, number>();
    const top = (i: number): number => { const p = up.get(i); return p === undefined || p === i ? i : top(p); };
    for (const [d, i, j] of joins) {
      const a = top(i), b = top(j);
      if (a === b) continue;
      up.set(a, b);
      if (d >= 1) out.links.push({ a: nodes[i].stop!, b: nodes[j].stop!, walk: false });
    }
  });
  return out;
}

/** The place stop `id` is in, or null. */
export function busPlaceOf(w: WorldView, id: number): BusPlace | null {
  const p = busPlaces(w), at = p.placeOf.get(id);
  return at === undefined ? null : p.places[at];
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
