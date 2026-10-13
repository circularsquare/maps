// Platform shapes (T-144, T-189, T-204), shared by the GPU drawing (renderer.ts STATION_VS),
// picking, the placement ghost and the cut-out above the demand bubbles (demandOverlay.ts).
//
// A platform is drawn where it is on the ground: a band along its track, as long as the platform,
// bending as the track bends (a circular arc through its middle and both ends), with flat ends.
// Through a station the platform is centred on the node; at a dead end it runs from the node along
// the track, as the trains stop at its middle there (sim service.rs). Its width is fixed on the
// ground too (T-204), so the box keeps one shape at every zoom. Zoomed out (below
// STATION_BOX_ZOOM) every station is a dot instead, a complex one dot (T-217, renderer.ts), and
// below STATIONS_ZOOM stations are not drawn (T-205). Selection never changes the size: a selected
// station gets an outline and a glow (T-189). The player's links are drawn as thin bridges between
// two platforms (T-217, `bridgeEnds`, `bridgeShape`).

import { EARTH_C } from "./geo";

export type Project = (x: number, y: number) => [number, number];

/** From this zoom stations are drawn as platforms along the track, further out as dots (T-189;
 * T-204 moved it from 13, Anita: "turn into them sooner, around ~4x more zoomed in"). At 15 a pixel
 * is about 1.8 m at New York's latitude, so a 150-200 m platform is 80-110 px long. */
export const STATION_BOX_ZOOM = 15;
/** Below this zoom neither stations nor trains are drawn or picked, only the lines (T-205,
 * Anita: "hide trains and stations when zoomed out"). At 11 the screen holds about 30 km, a city
 * such as Manhattan with its neighbours; further out the dots and pills crowd the lines. A selected
 * station is still drawn. */
export const STATIONS_ZOOM = 11;
/** Half a platform's true width, m: two tracks either side of an island platform, about 12 m in
 * all. The drawn box is wider (T-204): as wide on the ground as a plain line and its rim are at
 * STATION_BOX_ZOOM, less a quarter (T-239), about 16 m at New York's latitude. */
export const PLATFORM_HALF_M = 6;
/** Local units per CSS px at STATION_BOX_ZOOM: a box's fixed width, set on screen there, in units
 * on the ground (T-204). */
const BOX_UNIT = EARTH_C / (512 * 2 ** STATION_BOX_ZOOM);

/** Where a platform lies, from the station node, in local units (renderer's buffers). */
export interface PlatformShape {
  /** the platform's middle, from the node */
  mx: number;
  my: number;
  /** half the platform along its track at its middle: direction of the track there, length half
   * the platform's length */
  ax: number;
  ay: number;
  /** half the angle the track turns along the platform, rad; positive bends towards the left of
   * (ax, ay), i.e. towards (-ay, ax) */
  phi: number;
  /** the track's centre line along the platform (T-254), PLATFORM_POINTS points (x, y pairs) from
   * the node, evenly spaced along the track from the platform's -axis end to its +axis end; absent:
   * the arc of `phi` */
  pts?: ArrayLike<number>;
}

/** Points a platform's centre line is sampled at (T-254): 12 stretches, 12-17 m on a 150-200 m
 * platform, each turning 7 degrees or less on a 150 m radius curve (9 points showed facets on the
 * outline at zoom 16.5, 34 St-Hudson Yards). The shaders' loops, varyings and the selection
 * texture are sized by it. */
export const PLATFORM_POINTS = 13;

/** The point `d` along a polyline of x, y, z triples; past its end, straight on along its last
 * stretch (`extend`) or clamped to the end. */
function along(p: ArrayLike<number>, d: number, extend = false): [number, number] {
  let left = d;
  const n = p.length / 3;
  let lx = 1, ly = 0;
  for (let k = 0; k + 1 < n; k++) {
    const dx = p[k * 3 + 3] - p[k * 3], dy = p[k * 3 + 4] - p[k * 3 + 1], l = Math.hypot(dx, dy);
    if (l < 1e-9) continue;
    if (left <= l) return [p[k * 3] + (dx * left) / l, p[k * 3 + 1] + (dy * left) / l];
    left -= l;
    lx = dx / l; ly = dy / l;
  }
  if (!n) return [0, 0];
  const ex = p[(n - 1) * 3], ey = p[(n - 1) * 3 + 1];
  return extend ? [ex + lx * left, ey + ly * left] : [ex, ey];
}

/** The first nonzero direction along a polyline, unit; null if it has none. */
function heading(p: ArrayLike<number>): [number, number] | null {
  for (let k = 0; k + 5 < p.length; k += 3) {
    const dx = p[k + 3] - p[k], dy = p[k + 4] - p[k + 1], l = Math.hypot(dx, dy);
    if (l > 1e-9) return [dx / l, dy / l];
  }
  return null;
}

/** A polyline of x, y, z triples turned round. */
export function reversed(p: ArrayLike<number>): Float32Array {
  const n = p.length / 3, out = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) out.set([p[(n - 1 - k) * 3], p[(n - 1 - k) * 3 + 1], p[(n - 1 - k) * 3 + 2]], k * 3);
  return out;
}

/**
 * Where a platform lies on its track. `ways`: the track leaving the station node, one polyline
 * per edge end there (x, y, z triples starting at the node); `first`: the one the platform follows
 * (the edge with the most lines, map/network.ts). The platform carries on through the node along
 * the way most nearly opposite `first`; at a dead end (no other way) it runs from the node along
 * `first`. `half`: half the platform's length, local units.
 */
export function platformShape(ways: ArrayLike<number>[], first: number, half: number): PlatformShape {
  const a = ways[first];
  if (!a || a.length < 6) return { mx: 0, my: 0, ax: half, ay: 0, phi: 0 };
  const ha = heading(a);
  let b: ArrayLike<number> | null = null, best = Infinity;
  for (let i = 0; ha && i < ways.length; i++) {
    const h = i === first ? null : heading(ways[i]);
    if (!h) continue;
    const dot = h[0] * ha[0] + h[1] * ha[1];
    if (dot < best) [best, b] = [dot, ways[i]];
  }
  const node: [number, number] = [a[0], a[1]];
  const mid = b ? node : along(a, half);
  const e1 = b ? along(a, half) : along(a, 2 * half), e0 = b ? along(b, half) : node;
  // the track itself along the platform (T-254, Anita: "ideally we actually fit the track"): from
  // the -axis end to the +axis end, straight on past the end of the track if it stops short
  const pts: number[] = [];
  for (let i = 0; i < PLATFORM_POINTS; i++) {
    const t = -half + (2 * half * i) / (PLATFORM_POINTS - 1);
    const q = b ? (t < 0 ? along(b, -t, true) : along(a, t, true)) : along(a, t + half, true);
    pts.push(q[0] - node[0], q[1] - node[1]);
  }
  const cx = e1[0] - e0[0], cy = e1[1] - e0[1], cl = Math.hypot(cx, cy);
  if (cl < 1e-9 || !ha) return { mx: mid[0] - node[0], my: mid[1] - node[1], ax: (ha?.[0] ?? 1) * half, ay: (ha?.[1] ?? 0) * half, phi: 0, pts };
  const tx = cx / cl, ty = cy / cl, c = cl / 2;
  // the middle's offset to the left of the chord; the ends bend away from it, so the circle's
  // centre is to the right when the middle is to the left
  const s = (mid[0] - (e0[0] + e1[0]) / 2) * -ty + (mid[1] - (e0[1] + e1[1]) / 2) * tx;
  const k = (-2 * s) / (c * c + s * s);
  // at most 140 degrees end to end: track doubling back through a node is no platform shape
  const phi = Math.max(-1.2, Math.min(1.2, k * half));
  return { mx: mid[0] - node[0], my: mid[1] - node[1], ax: tx * half, ay: ty * half, phi, pts };
}

/** A platform's centre line, PLATFORM_POINTS points from its node (local units): its own, or the arc
 * of `phi` through its middle and both ends (a platform built without one). */
export function platformPoints(sh: PlatformShape): ArrayLike<number> {
  if (sh.pts && sh.pts.length >= 2 * PLATFORM_POINTS) return sh.pts;
  const hl = Math.hypot(sh.ax, sh.ay), out: number[] = [];
  const ux = hl > 1e-9 ? sh.ax / hl : 1, uy = hl > 1e-9 ? sh.ay / hl : 0, k = hl > 1e-9 ? sh.phi / hl : 0;
  for (let i = 0; i < PLATFORM_POINTS; i++) {
    const s = -hl + (2 * hl * i) / (PLATFORM_POINTS - 1);
    const ca = Math.abs(k) < 1e-9 ? s : Math.sin(k * s) / k, cb = Math.abs(k) < 1e-9 ? 0 : (1 - Math.cos(k * s)) / k;
    out.push(sh.mx + ca * ux - cb * uy, sh.my + ca * uy + cb * ux);
  }
  return out;
}

/** The track either way from a point on a polyline (x, y, z triples): the stretch towards its end
 * and the stretch back to its start, each starting at the nearest point (the placement ghost's
 * `ways` for a station placed part way along an edge). */
export function waysFrom(p: ArrayLike<number>, x: number, y: number): [Float32Array, Float32Array] {
  const n = p.length / 3;
  let bestD = Infinity, bestK = 0, bestT = 0;
  for (let k = 0; k + 1 < n; k++) {
    const dx = p[k * 3 + 3] - p[k * 3], dy = p[k * 3 + 4] - p[k * 3 + 1], l2 = dx * dx + dy * dy;
    const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - p[k * 3]) * dx + (y - p[k * 3 + 1]) * dy) / l2)) : 0;
    const d = (p[k * 3] + t * dx - x) ** 2 + (p[k * 3 + 1] + t * dy - y) ** 2;
    if (d < bestD) [bestD, bestK, bestT] = [d, k, t];
  }
  const px = p[bestK * 3] + bestT * (p[bestK * 3 + 3] - p[bestK * 3]), py = p[bestK * 3 + 1] + bestT * (p[bestK * 3 + 4] - p[bestK * 3 + 1]);
  const fwd = [px, py, 0], back = [px, py, 0];
  for (let k = bestK + 1; k < n; k++) fwd.push(p[k * 3], p[k * 3 + 1], p[k * 3 + 2]);
  for (let k = bestK; k >= 0; k--) back.push(p[k * 3], p[k * 3 + 1], p[k * 3 + 2]);
  return [new Float32Array(fwd), new Float32Array(back)];
}

/** Dot radius in CSS px, ordinary and transfer stations; a dot also covers the lines through it. */
export function stationRadii(zoom: number): [number, number] {
  return [interpolate(zoom, [[9, 2.2], [12, 3.8], [15, 5.6]]), interpolate(zoom, [[9, 2.8], [12, 5], [15, 7.2]])];
}
export function stationHaloRadius(zoom: number) { return interpolate(zoom, [[9, 18], [12, 40], [15, 56]]); }
/** The dark rim of a station, CSS px. */
export function stationRing(zoom: number) { return interpolate(zoom, [[9, 1.2], [13, 2]]); }
/** A selected station's outline outside its rim, and the glow beyond that, CSS px (T-189). */
export const SEL_OUTLINE_PX = 2;
export const SEL_GLOW_PX = 6;

function interpolate(zoom: number, stops: [number, number][]) {
  for (let i = 1; i < stops.length; i++) if (zoom < stops[i][0]) {
    const [z0, a] = stops[i - 1], [z1, b] = stops[i];
    return a + (b - a) * Math.max(0, (zoom - z0) / (z1 - z0));
  }
  return stops[stops.length - 1][1];
}

/** What sizes a station on screen, CSS px (renderer.ts keeps the last draw's). */
export interface StationStyle {
  /** platforms (true) or dots (zoomed out) */
  boxes: boolean;
  /** dot radius: ordinary, transfer */
  radius: [number, number];
  /** a line's width; 0 when lines are not drawn (Height and Max speed show the track at it) */
  linePx: number;
  ringPx: number;
  /** a line's width and the rim at STATION_BOX_ZOOM: they fix every box's width on the ground (T-204) */
  boxLinePx: number;
  boxRingPx: number;
  /** zoomed out below STATIONS_ZOOM: only selected stations are drawn and picked (T-205) */
  hidden?: boolean;
}

/** A box is this share of the width that covers a plain line and its rim at STATION_BOX_ZOOM
 * (T-239, Anita: "a bit too fat still, maybe 25% narrower"). Lines are not covered any more: in
 * traffic thickness mode (T-227) a line can be wider than the stations on it. */
export const BOX_NARROW = 0.75;
/** The line widths a dot covers, as a share of the line width: the same plain line a box is sized
 * from, whatever the traffic mode draws (T-239). */
const PLAIN_COVER = 0.6;

/** A box's half width at STATION_BOX_ZOOM, CSS px (T-239): one value for every station, in every
 * mode; the shader's `u_fixed` and the CPU twin read it. */
export function boxHalfPxAtBoxZoom(st: Pick<StationStyle, "boxLinePx" | "boxRingPx">) {
  return BOX_NARROW * (0.5 * PLAIN_COVER * st.boxLinePx + st.boxRingPx + 1);
}

/** How wide a box is on the ground, local units: the fixed width above, or the platform's true
 * half width if wider (T-204). */
function boxHalfLoc(st: StationStyle, halfWidth: number) {
  return Math.max(halfWidth, boxHalfPxAtBoxZoom(st) * BOX_UNIT);
}

/** A plain station's box half width on screen at `zoom` on a flat map, CSS px; 0 for dots (where
 * station names go, stationLabels.ts). */
export function plainBoxHalfPx(st: StationStyle, zoom: number) {
  return st.boxes ? (boxHalfLoc(st, 0) / BOX_UNIT) * 2 ** (zoom - STATION_BOX_ZOOM) : 0;
}

/** A station on screen, CSS px: the CPU twin of STATION_VS. */
export interface StationShape {
  /** the platform's middle */
  x: number;
  y: number;
  /** along the platform at its middle, and across it towards the side it bends to (unit) */
  ux: number;
  uy: number;
  nx: number;
  ny: number;
  halfLength: number;
  halfWidth: number;
  /** curvature of the centre line, 1/px, bending towards +n (0 when `pts` gives the line) */
  k: number;
  /** > 0: drawn as a dot of this radius at (x, y) */
  dot: number;
  /** the centre line (T-254): PLATFORM_POINTS points in the shape's own frame (along u, across n,
   * from (x, y)), CSS px; the band is `halfWidth` either side of it, with flat ends. Absent: the arc
   * of `k` (bridges) */
  pts?: Float32Array;
}

/** A shape's centre line in its own frame, PLATFORM_POINTS points: `pts`, or its arc sampled. */
export function shapePoints(s: StationShape): Float32Array {
  if (s.pts) return s.pts;
  const out = new Float32Array(2 * PLATFORM_POINTS), hl = s.halfLength, k = s.k;
  for (let i = 0; i < PLATFORM_POINTS; i++) {
    const t = -hl + (2 * hl * i) / (PLATFORM_POINTS - 1);
    out[i * 2] = Math.abs(k) < 1e-9 ? t : Math.sin(k * t) / k;
    out[i * 2 + 1] = Math.abs(k) < 1e-9 ? 0 : (1 - Math.cos(k * t)) / k;
  }
  return out;
}

/** A point `(a, b)` in a shape's frame to its place along and across the centre line `q`
 * (PLATFORM_POINTS points in the same frame): the arc length from the line's middle, and the
 * signed distance from the nearest point of it (+ to the left of the way it runs); past the ends
 * the end stretches run straight on, so the ends are flat. As SHAPE_GLSL's `polyFrame`. */
export function polyFrame(q: ArrayLike<number>, a: number, b: number): [number, number] {
  let total = 0;
  for (let i = 0; i + 1 < PLATFORM_POINTS; i++) total += Math.hypot(q[i * 2 + 2] - q[i * 2], q[i * 2 + 3] - q[i * 2 + 1]);
  let best = Infinity, along = 0, across = 0, acc = 0;
  for (let i = 0; i + 1 < PLATFORM_POINTS; i++) {
    const ax = q[i * 2], ay = q[i * 2 + 1], dx = q[i * 2 + 2] - ax, dy = q[i * 2 + 3] - ay, l = Math.hypot(dx, dy);
    if (l < 1e-6) continue;
    const ux = dx / l, uy = dy / l, px = a - ax, py = b - ay;
    let t = px * ux + py * uy;
    if (i > 0) t = Math.max(t, 0);
    if (i < PLATFORM_POINTS - 2) t = Math.min(t, l);
    const cx = px - ux * t, cy = py - uy * t, d = Math.hypot(cx, cy);
    if (d < best) {
      best = d;
      along = acc + t;
      across = (ux * py - uy * px) >= 0 ? d : -d;
    }
    acc += l;
  }
  return [along - total / 2, across];
}

/**
 * A station on screen. `x, y`: the node; `sh`: its platform (`PlatformShape`) and `halfWidth`, the
 * platform's true half width in local units; `transfer`: served by more than one line. A box's
 * length and width are both fixed on the ground, so it has one shape at every zoom (T-204), and
 * the same width for every station (T-239).
 */
export function stationShape(x: number, y: number, sh: PlatformShape, halfWidth: number, project: Project, st: StationStyle, transfer: boolean): StationShape {
  const mx = x + sh.mx, my = y + sh.my;
  const [cx, cy] = project(mx, my), [ex, ey] = project(mx + sh.ax, my + sh.ay), [qx, qy] = project(mx - sh.ay, my + sh.ax);
  const len = Math.hypot(ex - cx, ey - cy), nl = Math.hypot(qx - cx, qy - cy);
  const ux = len > 1e-6 ? (ex - cx) / len : 1, uy = len > 1e-6 ? (ey - cy) / len : 0;
  const nx = nl > 1e-6 ? (qx - cx) / nl : -uy, ny = nl > 1e-6 ? (qy - cy) / nl : ux;
  // CSS px per local unit here: the box's length and width both scale with it (T-204)
  const axisLoc = Math.hypot(sh.ax, sh.ay);
  const [ox, oy] = axisLoc > 1e-9 ? [0, 0] : project(mx + 1, my);
  const perLoc = axisLoc > 1e-9 ? len / axisLoc : Math.hypot(ox - cx, oy - cy);
  const hw = boxHalfLoc(st, halfWidth) * perLoc;
  // a dot covers a plain line through it as drawn now (not a wider traffic line, T-239)
  const cover = 0.5 * PLAIN_COVER * st.linePx + st.ringPx + 1;
  // the track's own line along the platform (T-254), in the frame at its middle
  const line = platformPoints(sh), pts = new Float32Array(2 * PLATFORM_POINTS);
  let total = 2 * len;
  if (st.boxes) total = 0;
  for (let i = 0; st.boxes && i < PLATFORM_POINTS; i++) {
    const [px, py] = project(x + line[i * 2], y + line[i * 2 + 1]);
    pts[i * 2] = (px - cx) * ux + (py - cy) * uy;
    pts[i * 2 + 1] = (px - cx) * nx + (py - cy) * ny;
    if (i) total += Math.hypot(pts[i * 2] - pts[i * 2 - 2], pts[i * 2 + 1] - pts[i * 2 - 1]);
  }
  return {
    x: cx, y: cy, ux, uy, nx, ny,
    halfLength: Math.max(total / 2, hw),
    halfWidth: hw,
    k: 0,
    dot: st.boxes ? 0 : Math.max(st.radius[transfer ? 1 : 0], cover),
    ...(st.boxes ? { pts } : {}),
  };
}

/** A point in a station's frame, CSS px: along its centre line from its middle, and across it
 * (towards +n). As SHAPE_GLSL's `polyFrame` (a platform, T-254) or `arc` (a bridge). */
export function stationCoords(s: StationShape, px: number, py: number): [number, number] {
  const dx = px - s.x, dy = py - s.y;
  const a = dx * s.ux + dy * s.uy, b = dx * s.nx + dy * s.ny;
  if (s.pts) return polyFrame(s.pts, a, b);
  if (Math.abs(s.k) < 1e-7) return [a, b];
  const R = 1 / s.k, sg = Math.sign(s.k);
  const qx = a, qy = b - R, r = Math.hypot(qx, qy);
  return [Math.atan2(qx, -sg * qy) * Math.abs(R), R - sg * r];
}

/** Signed distance from a station's edge, CSS px (negative inside): as SHAPE_GLSL's `stationSdf`. */
export function stationDistance(s: StationShape, px: number, py: number): number {
  if (s.dot > 0) return Math.hypot(px - s.x, py - s.y) - s.dot;
  const [a, b] = stationCoords(s, px, py);
  const dx = Math.abs(a) - s.halfLength, dy = Math.abs(b) - s.halfWidth;
  return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0);
}

export function insideStation(s: StationShape, x: number, y: number, padding = 0) {
  return stationDistance(s, x, y) <= padding;
}

/** A link's bridge (T-217, Anita: "if we manually link, this white shape should include a thin
 * bridge"): the white of it is this half width at STATION_BOX_ZOOM, CSS px, fixed on the ground like
 * the boxes (a plain box's white is 2.5 px there), with the station rim along both sides. */
export const BRIDGE_HALF_PX = 1.5;
/** A bridge starts this far inside a platform's end, CSS px at STATION_BOX_ZOOM, so its white
 * crosses the box's rim there and the two shapes join without a seam. */
const BRIDGE_INSET_PX = 6;

/** Points along a platform's centre line, local units (`x, y` its node), `inset` short of each end. */
export function platformLine(x: number, y: number, sh: PlatformShape, inset = 0, count = 9): number[] {
  if (sh.pts && sh.pts.length >= 2 * PLATFORM_POINTS) {
    // the track's own line (T-254), resampled evenly between the insets
    const q = sh.pts, seg: number[] = [];
    let total = 0;
    for (let i = 0; i + 1 < PLATFORM_POINTS; i++) total += (seg[i] = Math.hypot(q[i * 2 + 2] - q[i * 2], q[i * 2 + 3] - q[i * 2 + 1]));
    const h0 = Math.min(inset, total / 2), out: number[] = [];
    for (let c = 0; c < count; c++) {
      let t = h0 + ((total - 2 * h0) * c) / Math.max(1, count - 1), i = 0;
      while (i < PLATFORM_POINTS - 2 && t > seg[i]) t -= seg[i++];
      const f = seg[i] > 1e-9 ? Math.min(1, t / seg[i]) : 0;
      out.push(x + q[i * 2] + (q[i * 2 + 2] - q[i * 2]) * f, y + q[i * 2 + 1] + (q[i * 2 + 3] - q[i * 2 + 1]) * f);
    }
    return out;
  }
  const hl = Math.hypot(sh.ax, sh.ay);
  const mx = x + sh.mx, my = y + sh.my;
  if (hl < 1e-9) return [mx, my];
  const ux = sh.ax / hl, uy = sh.ay / hl, k = sh.phi / hl, h = Math.max(0, hl - inset), out: number[] = [];
  for (let i = 0; i < count; i++) {
    const s = -h + (2 * h * i) / (count - 1);
    const ca = Math.abs(k) < 1e-9 ? s : Math.sin(k * s) / k, cb = Math.abs(k) < 1e-9 ? 0 : (1 - Math.cos(k * s)) / k;
    out.push(mx + ca * ux - cb * uy, my + ca * uy + cb * ux);
  }
  return out;
}

/** Where segments p0-p1 and q0-q1 come nearest: the fraction along each. */
function nearestOnSegments(p0x: number, p0y: number, p1x: number, p1y: number, q0x: number, q0y: number, q1x: number, q1y: number): [number, number] {
  const ux = p1x - p0x, uy = p1y - p0y, vx = q1x - q0x, vy = q1y - q0y, wx = p0x - q0x, wy = p0y - q0y;
  const a = ux * ux + uy * uy, b = ux * vx + uy * vy, c = vx * vx + vy * vy, d = ux * wx + uy * wy, e = vx * wx + vy * wy;
  if (a < 1e-12 && c < 1e-12) return [0, 0];
  if (a < 1e-12) return [0, Math.max(0, Math.min(1, e / c))];
  if (c < 1e-12) return [Math.max(0, Math.min(1, -d / a)), 0];
  const den = a * c - b * b;
  let s = den > 1e-9 * a * c ? Math.max(0, Math.min(1, (b * e - c * d) / den)) : 0;
  let t = (b * s + e) / c;
  if (t < 0) { t = 0; s = Math.max(0, Math.min(1, -d / a)); }
  else if (t > 1) { t = 1; s = Math.max(0, Math.min(1, (b - d) / a)); }
  return [s, t];
}

/** A bridge between two linked platforms (T-217), local units: from the nearest points of their
 * centre lines, each kept BRIDGE_INSET_PX inside its platform's ends, so the bridge leaves each box
 * on the side facing the other. Platforms that overlap get one too; it is inside both. */
export function bridgeEnds(ax: number, ay: number, sa: PlatformShape, bx: number, by: number, sb: PlatformShape): [number, number, number, number] {
  const inset = BRIDGE_INSET_PX * BOX_UNIT;
  const A = platformLine(ax, ay, sa, Math.min(inset, Math.hypot(sa.ax, sa.ay))), B = platformLine(bx, by, sb, Math.min(inset, Math.hypot(sb.ax, sb.ay)));
  const seg = (p: number[], i: number) => (p.length > 2 ? [p[i * 2], p[i * 2 + 1], p[i * 2 + 2], p[i * 2 + 3]] : [p[0], p[1], p[0], p[1]]);
  let best: [number, number, number, number] = [A[0], A[1], B[0], B[1]], bestD = Infinity;
  for (let i = 0; i < Math.max(1, A.length / 2 - 1); i++)
    for (let j = 0; j < Math.max(1, B.length / 2 - 1); j++) {
      const [p0x, p0y, p1x, p1y] = seg(A, i), [q0x, q0y, q1x, q1y] = seg(B, j);
      const [s, t] = nearestOnSegments(p0x, p0y, p1x, p1y, q0x, q0y, q1x, q1y);
      const px = p0x + s * (p1x - p0x), py = p0y + s * (p1y - p0y), qx = q0x + t * (q1x - q0x), qy = q0y + t * (q1y - q0y);
      const d = Math.hypot(px - qx, py - qy);
      if (d < bestD) [bestD, best] = [d, [px, py, qx, qy]];
    }
  return best;
}

/** The smallest circle holding every point (x, y pairs): centre and radius (T-217, a complex's one
 * dot: centred so it reaches its farthest stations evenly, where the mean leans to the side with
 * more platforms). Incremental, exact; fine for the tens of points of a complex. */
export function enclosingCircle(p: [number, number][]): [number, number, number] {
  const inside = (c: [number, number, number], q: [number, number]) => Math.hypot(q[0] - c[0], q[1] - c[1]) <= c[2] * (1 + 1e-9) + 1e-9;
  const two = (a: [number, number], b: [number, number]): [number, number, number] => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.hypot(a[0] - b[0], a[1] - b[1]) / 2];
  const three = (a: [number, number], b: [number, number], c: [number, number]): [number, number, number] => {
    const bx = b[0] - a[0], by = b[1] - a[1], cx = c[0] - a[0], cy = c[1] - a[1], d = 2 * (bx * cy - by * cx);
    if (Math.abs(d) < 1e-12) {
      // in a line: the circle on its two farthest points
      const cs = [two(a, b), two(a, c), two(b, c)];
      return cs.reduce((m, k) => (k[2] > m[2] ? k : m));
    }
    const ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d, uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d;
    return [a[0] + ux, a[1] + uy, Math.hypot(ux, uy)];
  };
  if (!p.length) return [0, 0, 0];
  let c: [number, number, number] = [p[0][0], p[0][1], 0];
  for (let i = 1; i < p.length; i++) {
    if (inside(c, p[i])) continue;
    c = [p[i][0], p[i][1], 0];
    for (let j = 0; j < i; j++) {
      if (inside(c, p[j])) continue;
      c = two(p[i], p[j]);
      for (let k = 0; k < j; k++) if (!inside(c, p[k])) c = three(p[i], p[j], p[k]);
    }
  }
  return c;
}

/** A bridge on screen as a straight box (CSS px), or null when its ends meet: the CPU twin of
 * BRIDGE_VS, for the selection's outline and the cut-out above the bubbles. */
export function bridgeShape(px: number, py: number, qx: number, qy: number, project: Project, st: StationStyle): StationShape | null {
  const [ax, ay] = project(px, py), [bx, by] = project(qx, qy), [ox, oy] = project(px + 1, py);
  const len = Math.hypot(bx - ax, by - ay);
  if (len < 1e-3) return null;
  const ux = (bx - ax) / len, uy = (by - ay) / len;
  const perLoc = Math.hypot(ox - ax, oy - ay);
  return { x: (ax + bx) / 2, y: (ay + by) / 2, ux, uy, nx: -uy, ny: ux, halfLength: len / 2, halfWidth: BRIDGE_HALF_PX * BOX_UNIT * perLoc + st.ringPx, k: 0, dot: 0 };
}

/** An SVG path of a station's outline grown by `grow` px, relative to its middle (the placement
 * ghost). */
export function stationPath(s: StationShape, grow = 0): string {
  if (s.dot > 0) {
    const r = s.dot + grow;
    return `M${-r},0a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0Z`;
  }
  const N = 16, hl = s.halfLength + grow, hw = s.halfWidth + grow, pts: string[] = [];
  const xy = (a: number, b: number) => `${(a * s.ux + b * s.nx).toFixed(2)},${(a * s.uy + b * s.ny).toFixed(2)}`;
  if (s.pts) {
    // the track's own line (T-254): each point offset along the mitred normal, the end points
    // carried straight on to the box's flat ends
    const q = s.pts, P = PLATFORM_POINTS, side: [number, number][] = [], left: string[] = [], right: string[] = [];
    let total = 0;
    for (let i = 0; i + 1 < P; i++) total += Math.hypot(q[i * 2 + 2] - q[i * 2], q[i * 2 + 3] - q[i * 2 + 1]);
    const dir = (i: number): [number, number] => {
      const dx = q[i * 2 + 2] - q[i * 2], dy = q[i * 2 + 3] - q[i * 2 + 1], l = Math.hypot(dx, dy) || 1;
      return [dx / l, dy / l];
    };
    for (let i = 0; i < P; i++) {
      const a = dir(Math.max(0, i - 1)), b = dir(Math.min(P - 2, i));
      let nx = -(a[1] + b[1]), ny = a[0] + b[0];
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl; ny /= nl;
      const m = 1 / Math.max(0.5, nx * -b[1] + ny * b[0]);
      side.push([nx * m, ny * m]);
    }
    const ext = hl - total / 2, d0 = dir(0), d1 = dir(P - 2);
    for (let i = 0; i < P; i++) {
      const e = i === 0 ? -ext : i === P - 1 ? ext : 0, d = i === 0 ? d0 : d1;
      const cx = q[i * 2] + (e ? d[0] * e : 0), cy = q[i * 2 + 1] + (e ? d[1] * e : 0);
      left.push(xy(cx + side[i][0] * hw, cy + side[i][1] * hw));
      right.unshift(xy(cx - side[i][0] * hw, cy - side[i][1] * hw));
    }
    return `M${left.concat(right).join("L")}Z`;
  }
  const at = (t: number, side: number) => {
    // the centre line point at arc length t, and the normal there
    const th = s.k * t;
    const ca = Math.abs(s.k) < 1e-7 ? t : Math.sin(th) / s.k, cb = Math.abs(s.k) < 1e-7 ? 0 : (1 - Math.cos(th)) / s.k;
    const na = -Math.sin(th), nb = Math.cos(th);
    const a = ca + na * side, b = cb + nb * side;
    return `${(a * s.ux + b * s.nx).toFixed(2)},${(a * s.uy + b * s.ny).toFixed(2)}`;
  };
  for (let i = 0; i <= N; i++) pts.push(at(-hl + (2 * hl * i) / N, hw));
  for (let i = N; i >= 0; i--) pts.push(at(-hl + (2 * hl * i) / N, -hw));
  return `M${pts.join("L")}Z`;
}

/** The station shape's distance in GLSL, shared by STATION_VS/FS and the demand cut-out. */
export const SHAPE_GLSL = `
// A point p in a station's frame (along its centre line's tangent at its middle, across towards
// the side it bends to), as arc length along the centre line and distance across it; k in 1/px.
vec2 arc(vec2 p, float k) {
  if (abs(k) < 1e-7) return p;
  float R = 1.0 / k, sg = sign(k);
  vec2 q = p - vec2(0.0, R);
  return vec2(atan(q.x, -sg * q.y) * abs(R), R - sg * length(q));
}
// Signed distance from a box of half size b centred on the origin.
float boxSdf(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
// A point p in a platform's frame as arc length along its centre line q (${PLATFORM_POINTS} points in
// the same frame) from the line's middle, and signed distance across it; the end stretches run
// straight on past the ends, so the ends are flat (T-254, as stationGeometry.ts polyFrame).
vec2 polyFrame(vec2 p, vec2 q[${PLATFORM_POINTS}]) {
  float total = 0.0;
  for (int i = 0; i < ${PLATFORM_POINTS - 1}; i++) total += length(q[i + 1] - q[i]);
  float best = 1e20, along = 0.0, across = 0.0, acc = 0.0;
  for (int i = 0; i < ${PLATFORM_POINTS - 1}; i++) {
    vec2 ab = q[i + 1] - q[i];
    float l = length(ab);
    if (l < 1e-4) continue;
    vec2 u = ab / l, w = p - q[i];
    float t = dot(w, u);
    if (i > 0) t = max(t, 0.0);
    if (i < ${PLATFORM_POINTS - 2}) t = min(t, l);
    float d = length(w - u * t);
    if (d < best) {
      best = d;
      along = acc + t;
      across = u.x * w.y - u.y * w.x >= 0.0 ? d : -d;
    }
    acc += l;
  }
  return vec2(along - 0.5 * total, across);
}
// Signed distance from a station: a dot of radius dotR, or the band along its centre line.
float stationSdf(vec2 p, vec3 size, float dotR) {
  return dotR > 0.0 ? length(p) - dotR : boxSdf(arc(p, size.z), size.xy);
}`;
