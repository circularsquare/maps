// Station complexes (T-146, SPEC 8): stations whose platforms nearly touch, taken together, and
// stations the player linked by hand (T-197). Derived from the save's station geometry and links
// whenever the world changes; only the links are saved (with the network, sim/src/track/save.rs).
//
// Two stations are linked when their platforms come within PLATFORM_GAP_M of each other, when
// their centres are within DEMAND_MERGE_M (the distance at which the demand model already counts
// two stations as one, `station_merge_m` in sim/src/demand/kernel.rs), or when the player linked
// them. A complex is everything linked, directly or through other stations (single linkage). A
// station on its own is a complex of one. The second rule keeps every demand station inside one
// complex, so a complex's catchment is the sum of whole demand stations and never counts one twice
// (workers/demandClient.ts). Demand lets riders change between any two stations of a complex
// (T-161): the clock worker sends each constructed station's complex with the network snapshot.
//
// A platform is the straight segment of its length through the station along the track heading
// (curved platforms are taken as straight); levels are ignored, so lines crossing over or under
// each other form a complex too.
//
// Pure geometry, no signals (checks/t146-check.mjs runs it in node); the world's complexes and the
// complex view are in game/complexView.ts. `buildComplexes` builds from nothing and is the reference;
// the world and the clock worker keep the same result up to date with game/complexBuilder.ts (T-1380).

import type { StationId, StationInput } from "./types";

/** Platforms (centre lines) closer than this are one complex, m. DECISIONS 2026-10-12. */
export const PLATFORM_GAP_M = 50;
/** Kept equal to the demand kernel's `station_merge_m`. */
export const DEMAND_MERGE_M = 100;
/** The player can link stations whose centres are at most this far apart, m (T-197). */
export const LINK_MAX_M = 1000;

export interface Complexes {
  /** station id -> complex index */
  of: Map<StationId, number>;
  /** per complex, its stations' ids in save order */
  members: StationId[][];
}

/** Shortest distance between segments p0-p1 and q0-q1. */
export function segmentGap(p0x: number, p0y: number, p1x: number, p1y: number, q0x: number, q0y: number, q1x: number, q1y: number): number {
  const ux = p1x - p0x, uy = p1y - p0y, vx = q1x - q0x, vy = q1y - q0y, wx = p0x - q0x, wy = p0y - q0y;
  const a = ux * ux + uy * uy, b = ux * vx + uy * vy, c = vx * vx + vy * vy, d = ux * wx + uy * wy, e = vx * wx + vy * wy;
  const den = a * c - b * b;
  let s = 0, t = 0;
  if (a < 1e-12 && c < 1e-12) return Math.hypot(wx, wy);
  if (a < 1e-12) t = Math.max(0, Math.min(1, e / c));
  else if (c < 1e-12) s = Math.max(0, Math.min(1, -d / a));
  else {
    s = den > 1e-9 * a * c ? Math.max(0, Math.min(1, (b * e - c * d) / den)) : 0;
    t = (b * s + e) / c;
    if (t < 0) { t = 0; s = Math.max(0, Math.min(1, -d / a)); }
    else if (t > 1) { t = 1; s = Math.max(0, Math.min(1, (b - d) / a)); }
  }
  return Math.hypot(wx + s * ux - t * vx, wy + s * uy - t * vy);
}

/** Group stations into complexes; `links` are the player's pairs (a pair naming a station not in
 * the list is ignored). `kAt` is the world frame's metres per ground metre at a world y
 * (`groundScale` in game/coords.ts; 1 outside the Mercator frame, T-306): platform lengths and
 * the distances above are ground metres. */
export function buildComplexes(stations: readonly Pick<StationInput, "id" | "x" | "y" | "heading" | "length">[], links: readonly (readonly [StationId, StationId])[] = [], kAt: (y: number) => number = () => 1): Complexes {
  const n = stations.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const ends = new Float64Array(n * 4);
  const k = stations.map((s) => kAt(s.y));
  let reach = DEMAND_MERGE_M * k.reduce((m, v) => Math.max(m, v), 1);
  stations.forEach((s, i) => {
    const h = Math.max(0, s.length) / 2 * k[i], dx = Math.cos(s.heading) * h, dy = Math.sin(s.heading) * h;
    ends.set([s.x - dx, s.y - dy, s.x + dx, s.y + dy], i * 4);
    reach = Math.max(reach, 2 * h + PLATFORM_GAP_M * k[i]);
  });
  // a grid as wide as the farthest two linked centres can be: candidates in the 3 x 3 block
  const key = (i: number) => `${Math.floor(stations[i].x / reach)},${Math.floor(stations[i].y / reach)}`;
  const cells = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const k = key(i);
    const c = cells.get(k);
    if (c) c.push(i);
    else cells.set(k, [i]);
  }
  for (let i = 0; i < n; i++) {
    const cx = Math.floor(stations[i].x / reach), cy = Math.floor(stations[i].y / reach);
    for (let gx = cx - 1; gx <= cx + 1; gx++)
      for (let gy = cy - 1; gy <= cy + 1; gy++)
        for (const j of cells.get(`${gx},${gy}`) ?? []) {
          if (j <= i) continue;
          const a = stations[i], b = stations[j];
          const kk = (k[i] + k[j]) / 2;
          const linked = Math.hypot(a.x - b.x, a.y - b.y) <= DEMAND_MERGE_M * kk ||
            segmentGap(ends[i * 4], ends[i * 4 + 1], ends[i * 4 + 2], ends[i * 4 + 3], ends[j * 4], ends[j * 4 + 1], ends[j * 4 + 2], ends[j * 4 + 3]) <= PLATFORM_GAP_M * kk;
          if (!linked) continue;
          const ra = find(i), rb = find(j);
          if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
        }
  }
  if (links.length) {
    const at = new Map(stations.map((s, i) => [s.id, i]));
    for (const [a, b] of links) {
      const i = at.get(a), j = at.get(b);
      if (i === undefined || j === undefined) continue;
      const ra = find(i), rb = find(j);
      if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
    }
  }
  const of = new Map<StationId, number>(), members: StationId[][] = [], index = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const r = find(i);
    let c = index.get(r);
    if (c === undefined) index.set(r, (c = members.push([]) - 1));
    members[c].push(stations[i].id);
    of.set(stations[i].id, c);
  }
  return { of, members };
}

/** The stations in `id`'s complex, itself included (just [id] for a station on its own). */
export function membersOf(c: Complexes, id: StationId): StationId[] {
  const k = c.of.get(id);
  return k === undefined ? [id] : c.members[k];
}
