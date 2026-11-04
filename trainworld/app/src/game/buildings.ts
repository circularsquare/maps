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

import { BOX_TOUCH_M, platformOffsets, segmentGap } from "./complexes";
import type { BuildingInput, StationId, StationInput } from "./types";

export type { BuildingInput };

/** A platform this near the outline touches it, ground m: half a drawn box's width (BOX_TOUCH_M);
 * the track model's `building::TOUCH_M`. */
export const BUILDING_TOUCH_M = 8;

/** US$M a m² of floor, the track model's `upgrades::BUILDING_PER_M2`: US$2,000. */
export const BUILDING_PER_M2 = 0.002;
/** The rectangle the upgrade row offers stands this far out from the boxes, ground m (T-2218). */
export const OFFER_PAD_M = 15;
/** Snaps turn an edge within this of a right angle to the building's axis (T-2218). */
export const SNAP_RAD = (10 * Math.PI) / 180;

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

// ---------------------------------------------------------------- the tool's shapes (T-2218)
// Pack units throughout: the frame is conformal, so its angles are true ones.

/** Platforms whose directions are within this count as parallel when choosing the offered
 * rectangle's direction (T-2363). */
export const PARALLEL_RAD = (15 * Math.PI) / 180;

/** A platform's direction taken as if it were straight: from its drawn line's first point to its
 * last, or its heading when it has no line. */
export function platformChord(s: Geo): number {
  const l = s.line;
  if (l && l.length >= 4) {
    const dx = l[l.length - 2] - l[0], dy = l[l.length - 1] - l[1];
    if (dx || dy) return Math.atan2(dy, dx);
  }
  return s.heading;
}

/** How far apart two directions are as lines (either way along them), radians, 0 to pi / 2. */
function lineGap(a: number, b: number): number {
  const r = (((a - b) % Math.PI) + Math.PI) % Math.PI;
  return Math.min(r, Math.PI - r);
}

/** The offered rectangle's direction (T-2363): each platform's chord gathers those within PARALLEL_RAD
 * of it; the largest gathering wins, the longer total platform length on a tie, and the rectangle
 * follows the longest platform in it. */
export function offeredDirection(plat: readonly Geo[]): number {
  const dir = plat.map(platformChord);
  let best: number[] = [], bestN = -1, bestLen = -1;
  for (let i = 0; i < plat.length; i++) {
    const g: number[] = [];
    let len = 0;
    for (let j = 0; j < plat.length; j++) if (lineGap(dir[i], dir[j]) <= PARALLEL_RAD) { g.push(j); len += plat[j].length; }
    if (g.length > bestN || (g.length === bestN && len > bestLen)) [best, bestN, bestLen] = [g, g.length, len];
  }
  return dir[best.reduce((a, b) => (plat[b].length > plat[a].length ? b : a))];
}

/** The smallest rectangle round these stations' boxes along `offeredDirection`, OFFER_PAD_M out from
 * them: four corners, flat, its first edge along that direction. Null for no platform. */
export function offeredRectangle(stations: readonly Geo[], kAt: (y: number) => number = () => 1): number[] | null {
  const plat = stations.filter((s) => s.length > 0);
  if (!plat.length) return null;
  const a = offeredDirection(plat);
  const c = Math.cos(a), s = Math.sin(a);
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity, my = 0;
  for (const st of plat) {
    const p = platformOffsets(st, kAt(st.y));
    my += st.y / plat.length;
    for (let i = 0; i + 1 < p.length; i += 2) {
      const x = st.x + p[i], y = st.y + p[i + 1];
      const u = x * c + y * s, v = -x * s + y * c;
      u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
  }
  const pad = (BOX_TOUCH_M / 2 + OFFER_PAD_M) * kAt(my);
  u0 -= pad; u1 += pad; v0 -= pad; v1 += pad;
  return [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].flatMap(([u, v]) => [u * c - v * s, u * s + v * c]);
}

/** The direction of the line along `axis` or a right angle to it nearest the direction `a`, if within
 * `tol` of it (radians); else null. */
export function snapAngle(a: number, axis: number, tol = SNAP_RAD): number | null {
  const q = Math.PI / 2;
  const k = Math.round((a - axis) / q);
  const d = a - axis - k * q;
  return Math.abs(d) <= tol ? axis + k * q : null;
}

type P = { x: number; y: number };

/** The line through `from` along `axis` or a right angle to it that the edge from `from` to `p` is
 * within `tol` of, as its direction; null for none (or no length). */
export function snapLine(from: P, p: P, axis: number, tol = SNAP_RAD): number | null {
  const dx = p.x - from.x, dy = p.y - from.y;
  return dx || dy ? snapAngle(Math.atan2(dy, dx), axis, tol) : null;
}

/** `p` moved onto the line through `o` along direction `a`. */
export function ontoLine(o: P, a: number, p: P): P {
  const c = Math.cos(a), s = Math.sin(a), t = (p.x - o.x) * c + (p.y - o.y) * s;
  return { x: o.x + c * t, y: o.y + s * t };
}

/** Where the lines through `o1` along `a1` and through `o2` along `a2` meet; null when parallel. */
export function meet(o1: P, a1: number, o2: P, a2: number): P | null {
  const c1 = Math.cos(a1), s1 = Math.sin(a1), c2 = Math.cos(a2), s2 = Math.sin(a2);
  const den = c1 * s2 - s1 * c2;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((o2.x - o1.x) * s2 - (o2.y - o1.y) * c2) / den;
  return { x: o1.x + c1 * t, y: o1.y + s1 * t };
}

/** How far the direction from `o` to `p` is off direction `a`, radians (either way along the line). */
function offLine(o: P, p: P, a: number): number {
  const d = Math.atan2(p.y - o.y, p.x - o.x) - a;
  const r = ((d % Math.PI) + Math.PI) % Math.PI;
  return Math.min(r, Math.PI - r);
}

/** A corner at `p` between corners `prev` and `next` (either may be absent), each edge to it turned
 * to a right angle of `axis` when within `tol`. Where both snap, their meeting point when it is within
 * `reach` of `p` (so a corner held near a right angle stays there), else the edge nearer its line
 * alone (so a corner dragged further follows the pointer along it). */
export function snapCorner(p: P, prev: P | null, next: P | null, axis: number, tol = SNAP_RAD, reach = Infinity): P {
  const a = prev ? snapLine(prev, p, axis, tol) : null;
  const b = next ? snapLine(next, p, axis, tol) : null;
  if (a !== null && b !== null) {
    const m = meet(prev!, a, next!, b);
    if (m && Math.hypot(m.x - p.x, m.y - p.y) <= reach) return m;
    return offLine(prev!, p, a) <= offLine(next!, p, b) ? ontoLine(prev!, a, p) : ontoLine(next!, b, p);
  }
  if (a !== null) return ontoLine(prev!, a, p);
  if (b !== null) return ontoLine(next!, b, p);
  return p;
}

/** The outline's floor in pack units², even-odd area (shoelace). */
export function outlineArea(c: ArrayLike<number>): number {
  const n = c.length / 2;
  let s = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) s += c[2 * j] * c[2 * i + 1] - c[2 * i] * c[2 * j + 1];
  return Math.abs(s) / 2;
}
