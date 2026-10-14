// Coordinates. "Local units" are web mercator metres (mercator 0..1 times the earth's
// circumference) relative to the draw anchor; a metre on the ground is `k` local units, where
// k = 1 / cos(lat) (`kAtY`, `kAtLat`). Keeping them small and relative is what lets float32 hold
// them to a centimetre at street zoom.
//
// The draw anchor follows the view (T-307, docs/world.md "Clock worker, renderer and pick"): when
// the view centre is more than ANCHOR_REACH_M of ground from it at ANCHOR_ZOOM or deeper, the
// anchor moves to the view, snapped to an ANCHOR_GRID_DEG grid so small pans never move it. The
// main thread asks the clock worker for the new anchor (workers/clockClient.ts `followView`); the
// worker rebuilds its draw buffers relative to it, and the main thread switches `anchor()` when that
// state arrives, so the network drawn and every local-unit conversion change together. Caches in
// local units rebuild with the world it comes with, or key on `anchor().version`.
//
// The demand views follow it too (T-317): every disc set (bubbles, a selection's ends, a station's
// catchment, longer trips) is built relative to the anchor in force and keeps that anchor with it
// (`DemandOverlay` draws each set with its own anchor's matrix, the hover finders turn the pointer
// into each set's local units), and is rebuilt when `anchor().version` changes (the signal
// `anchorVersion`, map/anchorVersion.ts). Disc sizes use k at the disc (`kAtY`, `kAtLat`).

export const EARTH_C = 40075016.686;
const EARTH_R = 6371008.8;
const RAD = Math.PI / 180;

/** A draw anchor: where local units are measured from. `version` 0 is HOME_ANCHOR; each move of
 * the anchor takes the next. */
export interface Anchor {
  lng: number;
  lat: number;
  /** its mercator 0..1 position */
  merc: [number, number];
  version: number;
}

function mercY(lat: number) {
  const s = Math.sin(lat * RAD);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}

export function makeAnchor(lng: number, lat: number, version: number): Anchor {
  return { lng, lat, merc: [(lng + 180) / 360, mercY(lat)], version };
}

/** New York: the single origin every view used before T-307, and still the anchor there. */
export const HOME_ANCHOR = makeAnchor(-73.98, 40.73, 0);
/** real metre -> local unit at HOME_ANCHOR's latitude: only for the stress network (synth.ts),
 * which lives at New York. Everything else uses k where it is (`kAtY`, `kAtLat`). */
export const MERC_K = 1 / Math.cos(HOME_ANCHOR.lat * RAD);

let current = HOME_ANCHOR;
/** The anchor local units are measured from now (in a worker: the one it was last told). */
export function anchor(): Anchor {
  return current;
}
export function setAnchor(a: Anchor) {
  current = a;
}

/** The anchor moves when the view centre is farther than this from it, metres of ground... */
export const ANCHOR_REACH_M = 100_000;
/** ...at this zoom or deeper (further out a metre is under a pixel) */
export const ANCHOR_ZOOM = 10;
/** and goes to the nearest point of this grid, degrees: at most about 80 km from the centre */
export const ANCHOR_GRID_DEG = 1;

/** Where the anchor should go for a view centred at (lng, lat) at `zoom`, or null to stay at `a`. */
export function anchorWanted(lng: number, lat: number, zoom: number, a: Anchor): [number, number] | null {
  if (zoom < ANCHOR_ZOOM || !Number.isFinite(lng) || !Number.isFinite(lat)) return null;
  if (groundM(lng, lat, a.lng, a.lat) <= ANCHOR_REACH_M) return null;
  const g = ANCHOR_GRID_DEG;
  const to: [number, number] = [Math.max(-180, Math.min(180, Math.round(lng / g) * g)), Math.max(-80, Math.min(80, Math.round(lat / g) * g))];
  return to[0] === a.lng && to[1] === a.lat ? null : to;
}

/** Great-circle distance, metres. */
function groundM(lng1: number, lat1: number, lng2: number, lat2: number): number {
  const p1 = lat1 * RAD, p2 = lat2 * RAD;
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(((lng2 - lng1) * RAD) / 2) ** 2;
  return 2 * EARTH_R * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** lng/lat -> local units */
export function toLocal(lng: number, lat: number, a: Anchor = current): [number, number] {
  return [((lng + 180) / 360 - a.merc[0]) * EARTH_C, (mercY(lat) - a.merc[1]) * EARTH_C];
}

/** local units -> mercator 0..1 (the input space of MapLibre's mainMatrix) */
export function localToMerc(x: number, y: number, a: Anchor = current): [number, number] {
  return [a.merc[0] + x / EARTH_C, a.merc[1] + y / EARTH_C];
}

/** local units -> lng/lat */
export function toLngLat(x: number, y: number, a: Anchor = current): [number, number] {
  const [mx, my] = localToMerc(x, y, a);
  const lat = (Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI;
  return [mx * 360 - 180, lat];
}

/** Column-major 4x4: local units of anchor `a` -> mercator 0..1. */
export function localToMercMatrix(a: Anchor = current): number[] {
  return [1 / EARTH_C, 0, 0, 0, 0, 1 / EARTH_C, 0, 0, 0, 0, 1, 0, a.merc[0], a.merc[1], 0, 1];
}

/** The isometric latitude at the anchor, rad: k at local y is cosh(this - y * PSI_PER_UNIT). */
export function anchorPsi(a: Anchor = current): number {
  return 2 * Math.PI * (0.5 - a.merc[1]);
}
export const PSI_PER_UNIT = (2 * Math.PI) / EARTH_C;

/** Local units per metre of ground at local y (mercator's scale, 1 / cos(lat) there). */
export function kAtY(y: number, a: Anchor = current): number {
  return Math.cosh(anchorPsi(a) - y * PSI_PER_UNIT);
}
/** Local units of anchor `to` that a point at (0, 0) of anchor `from` is at: add it to positions in
 * `from`'s units to get `to`'s. */
export function anchorShift(from: Anchor, to: Anchor): [number, number] {
  return [(from.merc[0] - to.merc[0]) * EARTH_C, (from.merc[1] - to.merc[1]) * EARTH_C];
}
/** Local units per metre of ground at a latitude. */
export function kAtLat(lat: number): number {
  return 1 / Math.cos(lat * RAD);
}

/** out = a * b, column-major 4x4, in float64; result as float32 for the GPU. */
export function mul4(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}
