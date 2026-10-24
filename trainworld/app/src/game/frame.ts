// World frames (T-125): how the track model's planar metres relate to longitude and latitude. The
// same two frames as sim/src/track/frame.rs, in the same arithmetic; checks/t125-frame-check.mjs
// compares this file with PROJ's values.
//
// - "equirect": x = R dlon cos(lat0), y = R dlat. Every save made before T-125 is in it.
// - "lcc": a spherical Lambert conformal conic tangent at lat0, the frame of new games. Scale
//   1 + dlat^2 / 2 (0.24% at 450 km north or south of the origin), conformal, so arcs stay arcs.
//
// - "merc": spherical Mercator over the whole Earth, the frame of new games since T-306
//   (docs/world.md): x = R lon, y = R atanh(sin lat), no origin (the spec's lon and lat are the
//   game's home origin, kept but not used). Its plane metres are not ground metres: scale
//   k = 1 / cos(lat) = cosh(y / R), so a distance in its metres is divided by k for the ground. The
//   track model's lengths are already ground metres.
//
// A frame is a value, not a global: a demand pack's own frame is an "equirect" at its origin, and a
// game's frame is read from its save.

export type FrameKind = "equirect" | "lcc" | "merc";
export interface FrameSpec { kind: FrameKind; lon: number; lat: number }
export interface Frame extends FrameSpec {
  /** plane metres to [lon, lat] in degrees */
  toLonLat(x: number, y: number): [number, number];
  /** [lon, lat] in degrees to plane metres */
  fromLonLat(lon: number, lat: number): [number, number];
  /** How much longer a metre on the ground is drawn at a latitude (east-west, north-south). */
  scaleAt(lat: number): [number, number];
}

export const EARTH_R = 6371008.8;
/** Metres a degree of longitude spans at the equator (times cos(lat) elsewhere), and a degree of
 * latitude near mid latitudes: the flat approximation over a few km, where a haversine is too dear. */
export const M_PER_DEG = 111_320;
export const M_PER_DEG_LAT = 110_540;
const RAD = Math.PI / 180;

const cache = new Map<string, Frame>();

export function makeFrame(spec: FrameSpec): Frame {
  // Mercator has no origin: every "merc" spec is the one frame (callers compare frames by identity).
  const key = spec.kind === "merc" ? "merc" : `${spec.kind}:${spec.lon}:${spec.lat}`;
  let f = cache.get(key);
  if (!f) cache.set(key, (f = spec.kind === "lcc" ? lcc(spec) : spec.kind === "merc" ? merc(spec) : equirect(spec)));
  return f;
}

/** Mercator scale at plane y: plane metres per ground metre. */
export function mercK(y: number): number {
  return Math.cosh(y / EARTH_R);
}

function merc({ lon, lat }: FrameSpec): Frame {
  return {
    kind: "merc", lon, lat,
    toLonLat: (x, y) => [x / (EARTH_R * RAD), Math.atan(Math.sinh(y / EARTH_R)) / RAD],
    fromLonLat: (lon, lat) => {
      let d = lon % 360;
      d = d > 180 ? d - 360 : d < -180 ? d + 360 : d;
      return [EARTH_R * d * RAD, EARTH_R * Math.atanh(Math.sin(lat * RAD))];
    },
    scaleAt: (lat) => { const k = 1 / Math.cos(lat * RAD); return [k, k]; },
  };
}

function equirect({ lon: lon0, lat: lat0 }: FrameSpec): Frame {
  const cos0 = Math.cos(lat0 * RAD);
  return {
    kind: "equirect", lon: lon0, lat: lat0,
    toLonLat: (x, y) => [lon0 + x / (EARTH_R * cos0 * RAD), lat0 + y / (EARTH_R * RAD)],
    fromLonLat: (lon, lat) => [EARTH_R * (lon - lon0) * RAD * cos0, EARTH_R * (lat - lat0) * RAD],
    scaleAt: (lat) => [cos0 / Math.cos(lat * RAD), 1],
  };
}

function lcc({ lon: lon0, lat: lat0 }: FrameSpec): Frame {
  const phi0 = lat0 * RAD, n = Math.sin(phi0);
  const t0 = Math.tan(Math.PI / 4 + phi0 / 2);
  const rf = EARTH_R * Math.cos(phi0) * Math.pow(t0, n) / n;
  const rho0 = EARTH_R * Math.cos(phi0) / n;
  const wrap = (d: number) => { d %= 360; return d > 180 ? d - 360 : d < -180 ? d + 360 : d; };
  const rho = (lat: number) => rf / Math.pow(Math.tan(Math.PI / 4 + lat * RAD / 2), n);
  return {
    kind: "lcc", lon: lon0, lat: lat0,
    toLonLat: (x, y) => {
      const dy = rho0 - y;
      const r = Math.hypot(x, dy);
      const lat = 2 * Math.atan(Math.pow(rf / r, 1 / n)) - Math.PI / 2;
      return [lon0 + Math.atan2(x, dy) / n / RAD, lat / RAD];
    },
    fromLonLat: (lon, lat) => {
      const r = rho(lat), theta = n * wrap(lon - lon0) * RAD;
      return [r * Math.sin(theta), rho0 - r * Math.cos(theta)];
    },
    scaleAt: (lat) => { const k = n * rho(lat) / (EARTH_R * Math.cos(lat * RAD)); return [k, k]; },
  };
}

/** The frame a track save's bytes are in (`TrackApi.frame_kind`): TWT6 is Mercator (2), TWT5 the
 * Northeast conic (1), older formats the legacy one (0). */
export function frameKindName(kind: number): FrameKind {
  return kind === 2 ? "merc" : kind === 1 ? "lcc" : "equirect";
}
/** The number `TrackApi.set_empty_frame` and `convert_save_frame` take for a frame kind. */
export function frameKindNumber(kind: FrameKind): number {
  return kind === "merc" ? 2 : kind === "lcc" ? 1 : 0;
}
