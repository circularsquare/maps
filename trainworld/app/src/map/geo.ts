// Coordinates. "Local units" are web mercator metres (mercator 0..1 times the earth's
// circumference) relative to a city origin; real metres at the origin's latitude are local / MERC_K.
// Keeping them small and relative is what lets float32 hold them to a centimetre at street zoom.
// One origin for now (New York); a city pack will carry its own.

export const ORIGIN_LNGLAT: [number, number] = [-73.98, 40.73];
export const EARTH_C = 40075016.686;
/** real metre -> local unit at the origin's latitude */
export const MERC_K = 1 / Math.cos((ORIGIN_LNGLAT[1] * Math.PI) / 180);

function mercY(lat: number) {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}

export const ORIGIN_MERC: [number, number] = [(ORIGIN_LNGLAT[0] + 180) / 360, mercY(ORIGIN_LNGLAT[1])];

/** lng/lat -> local units */
export function toLocal(lng: number, lat: number): [number, number] {
  return [((lng + 180) / 360 - ORIGIN_MERC[0]) * EARTH_C, (mercY(lat) - ORIGIN_MERC[1]) * EARTH_C];
}

/** local units -> mercator 0..1 (the input space of MapLibre's mainMatrix) */
export function localToMerc(x: number, y: number): [number, number] {
  return [ORIGIN_MERC[0] + x / EARTH_C, ORIGIN_MERC[1] + y / EARTH_C];
}

/** local units -> lng/lat */
export function toLngLat(x: number, y: number): [number, number] {
  const [mx, my] = localToMerc(x, y);
  const lat = (Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI;
  return [mx * 360 - 180, lat];
}

/** Column-major 4x4: local units -> mercator 0..1. */
export const LOCAL_TO_MERC = [1 / EARTH_C, 0, 0, 0, 0, 1 / EARTH_C, 0, 0, 0, 0, 1, 0, ORIGIN_MERC[0], ORIGIN_MERC[1], 0, 1];

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
