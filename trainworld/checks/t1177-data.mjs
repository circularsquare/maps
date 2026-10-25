// T-1177, T-1229: the made-up everyday pairs and the line joining their zones, shared by
// t1177-cross-everyday.mjs (the worker) and t1177-cross-everyday-page.mjs (the page). Boston's and
// Providence's biggest zones, with pairs from the first to the second written into Boston's market
// pack in memory; nothing under data/ is touched.
import { marketPacks } from './lib.mjs';

export const R = 6371008.8, RAD = Math.PI / 180;
export const FRAME = { kind: 'equirect', lon: -73.985, lat: 40.758 };
export const KINDS = [0, 1, 2], TRIPS = [3000, 1000, 500], WALK = [0.3, 0.1, 0.0];
export const NAMES = ['shopping', 'social', 'school', 'university', 'airport', 'medical', 'attractions'];

export function pairData(homeId, awayId) {
  const homeMarket = JSON.parse(marketPacks([homeId])[0].pack), awayMarket = JSON.parse(marketPacks([awayId])[0].pack);
  // the zone nearest the pack's centre among the big ones, so a view there belongs to the pack
  const biggest = (m) => {
    const top = Math.max(...m.zone_ng), d = (z) => Math.hypot(m.zones[z][0] + 0.5, m.zones[z][1] + 0.5);
    return m.zones.map((_, z) => z).filter((z) => m.zone_ng[z] >= 0.7 * top).sort((x, y) => d(x) - d(y))[0];
  };
  const [hz, az] = [biggest(homeMarket), biggest(awayMarket)];
  // a zone's middle: pack metres to lon, lat
  const lonLat = (m, z) => {
    const [i, j] = m.zones[z], x = (i + 0.5) * m.zone_m, y = (j + 0.5) * m.zone_m;
    return [m.frame.lon + x / (R * Math.cos(m.frame.lat * RAD)) / RAD, m.frame.lat + y / R / RAD];
  };
  const xyOf = ([lon, lat]) => [R * (lon - FRAME.lon) * RAD * Math.cos(FRAME.lat * RAD), R * (lat - FRAME.lat) * RAD];
  const [from, to] = [lonLat(homeMarket, hz), lonLat(awayMarket, az)];
  // a station in the middle of each zone, and one a km east of it
  const a = xyOf(from), b = xyOf(to);
  const pts = [a, [a[0] + 1000, a[1]], b, [b[0] + 1000, b[1]]];
  const lngLat = [from, [from[0] + 1000 / (R * Math.cos(from[1] * RAD)) / RAD, from[1]], to, [to[0] + 1000 / (R * Math.cos(to[1] * RAD)) / RAD, to[1]]];
  /** the home pack's market with the made-up pairs written in */
  const withPairs = () => {
    const v = structuredClone(homeMarket);
    v.everyday_home = KINDS.map(() => hz);
    v.everyday_place = KINDS.map(() => awayMarket.place);
    v.everyday_zone = KINDS.map(() => az);
    v.everyday_kind = KINDS;
    v.everyday_trips = TRIPS;
    v.everyday_walk = WALK;
    return v;
  };
  return { homeMarket, awayMarket, hz, az, pts, lngLat, withPairs };
}

/** the line joining the four stations in the world frame, as the worker takes a network */
export function lineNet(pts) {
  const km = pts.slice(1).map((p, k) => Math.hypot(p[0] - pts[k][0], p[1] - pts[k][1]) / 1000);
  const hops = [120, Math.round(km[1] * 60 * 0.8), 120]; // seconds, the long hop at 75 km/h
  const m = pts.length, times = new Array(6 * m).fill(0);
  for (let lev = 0; lev < 3; lev++) for (let k = 0; k < m - 1; k++) { times[lev * m + k] = hops[k]; times[(3 + lev) * m + k + 1] = hops[m - 2 - k]; }
  return { frame: FRAME, stXY: new Float32Array(pts.flat()), complex: new Uint32Array(m).map((_, i) => i), lineN: new Uint32Array([m]), stops: new Uint32Array(m).map((_, i) => i), times: new Float32Array(times), tph: new Float32Array([12, 10, 6]), cars: new Float32Array([6]), fare: { base: 2.9, perKm: 0.1 }, hopKm: new Float32Array([...km, 0, ...km.slice().reverse(), 0]) };
}
