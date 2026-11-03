// Station buildings (T-2217, notes/T-2208.md 3): an outline the player draws round stations, kept in
// the track model (`Network::buildings`, sim/src/track/building.rs) and saved with the network. Its
// members are every station whose platform touches it, at any level: the platform's centre line as
// the complexes see it (`platformOffsets`, the drawn box's line) inside the outline or within
// BUILDING_TOUCH_M of an edge. Members are worked out again whenever stations or outlines change, never
// saved. A building joins its members into one complex as a passage would (`buildingLinks`, passed
// with the player's links to `buildComplexes` and `ComplexBuilder`); unlike a passage it does not
// make the walk faster. The track model applies the same rule over its own track for its refusals.
//
// Pure geometry, no signals (checks/t2217-buildings.mjs runs it in node).

import { platformOffsets, segmentGap } from "./complexes";
import type { BuildingInput, StationId, StationInput } from "./types";

export type { BuildingInput };

/** A platform this near the outline touches it, ground m: half a drawn box's width (BOX_TOUCH_M);
 * the track model's `building::TOUCH_M`. */
export const BUILDING_TOUCH_M = 8;

type Geo = Pick<StationInput, "id" | "x" | "y" | "heading" | "length" | "line">;

/** The buildings from `TrackApi.buildings_info` (id, corner count n, then n x, y pairs per building),
 * named by `nameOf`. */
export function buildingsFrom(info: ArrayLike<number>, nameOf: (id: number) => string): BuildingInput[] {
  const out: BuildingInput[] = [];
  for (let k = 0; k + 1 < info.length; ) {
    const id = info[k], n = info[k + 1];
    out.push({ id, name: nameOf(id), corners: Array.from({ length: 2 * n }, (_, j) => info[k + 2 + j]) });
    k += 2 + 2 * n;
  }
  return out;
}

/** (x, y) inside the outline (even-odd). */
export function insideOutline(c: ArrayLike<number>, x: number, y: number): boolean {
  const n = c.length / 2;
  let odd = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const ax = c[2 * i], ay = c[2 * i + 1], bx = c[2 * j], by = c[2 * j + 1];
    if ((ay > y) !== (by > y) && x < ax + ((bx - ax) * (y - ay)) / (by - ay)) odd = !odd;
  }
  return odd;
}

/** A platform given as offsets (x, y pairs) from its node (x, y) touches the outline: a point of it
 * inside, or within `tol` of an edge (world units). */
export function platformTouches(c: ArrayLike<number>, x: number, y: number, p: ArrayLike<number>, tol: number): boolean {
  const m = p.length / 2, n = c.length / 2;
  for (let i = 0; i < m; i++) if (insideOutline(c, x + p[2 * i], y + p[2 * i + 1])) return true;
  for (let i = 0; i < Math.max(1, m - 1); i++) {
    const i1 = m > 1 ? i + 1 : i;
    const p0x = x + p[2 * i], p0y = y + p[2 * i + 1], p1x = x + p[2 * i1], p1y = y + p[2 * i1 + 1];
    for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      if (segmentGap(p0x, p0y, p1x, p1y, c[2 * j], c[2 * j + 1], c[2 * j1], c[2 * j1 + 1]) <= tol) return true;
    }
  }
  return false;
}

/** Each building's members, in `stations`' order. `kAt`: world units per ground metre at a world y
 * (`groundScale`). */
export function buildingMembers(buildings: readonly BuildingInput[], stations: readonly Geo[], kAt: (y: number) => number = () => 1): StationId[][] {
  return buildings.map((b) => {
    const c = b.corners, n = c.length / 2;
    if (n < 3) return [];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, my = 0;
    for (let i = 0; i < n; i++) {
      x0 = Math.min(x0, c[2 * i]); x1 = Math.max(x1, c[2 * i]);
      y0 = Math.min(y0, c[2 * i + 1]); y1 = Math.max(y1, c[2 * i + 1]);
      my += c[2 * i + 1] / n;
    }
    const k = kAt(my), tol = BUILDING_TOUCH_M * k;
    const out: StationId[] = [];
    for (const s of stations) {
      // a platform reaches at most about its length from its node
      const r = Math.max(0, s.length) * 1.5 * k + tol;
      if (s.x < x0 - r || s.x > x1 + r || s.y < y0 - r || s.y > y1 + r) continue;
      if (platformTouches(c, s.x, s.y, platformOffsets(s, kAt(s.y)), tol)) out.push(s.id);
    }
    return out;
  });
}

/** The pairs that join each building's members into one complex: each member to the next. */
export function buildingLinks(buildings: readonly BuildingInput[], stations: readonly Geo[], kAt: (y: number) => number = () => 1): [StationId, StationId][] {
  const out: [StationId, StationId][] = [];
  for (const m of buildingMembers(buildings, stations, kAt)) for (let i = 1; i < m.length; i++) out.push([m[i - 1], m[i]]);
  return out;
}

let lastStations: readonly Geo[] | null = null;
let lastKey = "";
let lastLinks: [StationId, StationId][] = [];
/** `links` with the buildings' pairs after them, for the complexes; the buildings' pairs worked out
 * again only when the stations list or an outline changed. `links` itself when there are none. */
export function withBuildingLinks<T extends readonly (readonly [StationId, StationId])[]>(links: T, buildings: readonly BuildingInput[] | undefined, stations: readonly Geo[], kAt: (y: number) => number): T | (readonly [StationId, StationId])[] {
  if (!buildings?.length) return links;
  const key = buildings.map((b) => b.corners.join(",")).join("|");
  if (stations !== lastStations || key !== lastKey) {
    lastStations = stations;
    lastKey = key;
    lastLinks = buildingLinks(buildings, stations, kAt);
  }
  return lastLinks.length ? [...links, ...lastLinks] : links;
}
