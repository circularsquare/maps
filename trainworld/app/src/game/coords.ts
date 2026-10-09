// Pack metres (the track model's frame: east and north of the city pack's origin, equirectangular,
// notes/T-004.md) to and from lng/lat. The origin comes from the clock worker once it has read
// the pack (`packOrigin` in workers/clockClient.ts); New York's until then.

const R = 6371008.8;
const DEG = 180 / Math.PI;
let origin = { lon: -73.985, lat: 40.758 };

export function setPackOrigin(o: { lon: number; lat: number }) {
  origin = o;
}

export function lngLatToPack(lng: number, lat: number): { x: number; y: number } {
  return { x: (R * (lng - origin.lon) * Math.cos(origin.lat / DEG)) / DEG, y: (R * (lat - origin.lat)) / DEG };
}

export function packToLngLat(x: number, y: number): [number, number] {
  return [origin.lon + (x / (R * Math.cos(origin.lat / DEG))) * DEG, origin.lat + (y / R) * DEG];
}
