// T-326: the network split into clusters that share no rider. A cluster is the populations (city
// packs, the regional market) and the stations and lines they can reach, joined by lines, station
// complexes and stations close enough to walk between. A cluster's figures depend only on its own
// lines and populations, so each is solved on its own sub-network, and one whose sub-network has
// not changed keeps its result (docs/demand.md "Clusters"). Reach is judged generously: joining two
// clusters that share nothing costs only time, splitting two that share riders would change figures.

import { CITIES, worldToCity, worldToMarket, zoneBox, type CityId } from "../game/cities";
import { PERIODS } from "../game/periods";
import type { DemandNetwork } from "./demandProtocol";

const LEVEL = { high: 0, medium: 1, low: 2 } as const;

/** A city reaches a station within this of its cells' bounding box: past the 4 km walk and the
 * 15 km ride from a home's group centre (`ride_max_m`, up to about 9 km off the cell). */
export const CITY_REACH_M = 30_000;
/** T-396: the regional market reaches a station within this of a market zone's registry box: its
 * sites are the zones' cells, inside the boxes, and its rides reach about 17 km. */
export const MARKET_REACH_M = 30_000;
/** Stations this close are joined: platforms merged into one demand station (100 m) and walking
 * transfers (300 m) lie well inside it. */
export const JOIN_M = 1_000;

const MARKET_BOXES = CITIES.filter((c) => c.market >= 0).map(zoneBox);

/** T-396: whether the regional market's riders can use a station at (lon, lat): within
 * `MARKET_REACH_M` of a market zone's box. What opens the market (workers/demandClient.ts). */
export function marketReaches(lon: number, lat: number): boolean {
  const dy = MARKET_REACH_M / 111_320, dx = dy / Math.cos((lat * Math.PI) / 180);
  return MARKET_BOXES.some(([w, s, e, n]) => lon >= w - dx && lon <= e + dx && lat >= s - dy && lat <= n + dy);
}

export interface Cluster {
  /** the job's cities in it, in the order given */
  cities: CityId[];
  /** the regional market's riders can reach it */
  market: boolean;
  /** T-409: the market's own zones whose stations are in it, ascending, joined by commas (the
   * market's part for this cluster is kept by it); "" for the market no station reaches */
  marketKey: string;
  /** its own network: the cluster's stations (ascending snapshot order) and lines (snapshot order) */
  net: DemandNetwork;
  /** its lines' indices in the whole network (ascending) */
  lines: Uint32Array;
  /** snapshot station -> station of `net`, or -1 */
  stLocal: Int32Array;
  /** route node of `net` -> route node of the whole network */
  nodeGlobal: Uint32Array;
  /** route node of the whole network -> route node of `net`, or -1 */
  nodeLocal: Int32Array;
  /** everything its result depends on, so an unchanged cluster can keep its result */
  sig: string;
  /** T-124: per period, everything that period's result depends on: `sig` less the other demand
   * levels' run times and trains an hour */
  psig: string[];
}

/** A city's cells' bounding box in its own frame: west, south, east, north metres (null: no cells). */
export type Box = [number, number, number, number] | null;

export function cellBox(xy: ArrayLike<number>): Box {
  if (xy.length < 2) return null;
  const b: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let k = 0; k < xy.length; k += 2) {
    const x = xy[k], y = xy[k + 1];
    if (x < b[0]) b[0] = x;
    if (y < b[1]) b[1] = y;
    if (x > b[2]) b[2] = x;
    if (y > b[3]) b[3] = y;
  }
  return b;
}

/** FNV-1a over the bytes of each array, two seeds: a 64-bit key. */
function hashOf(parts: (ArrayBufferView | string)[]): string {
  let h1 = 0x811c9dc5, h2 = 0x01000193 ^ 0x5bd1e995;
  for (const p of parts) {
    const b = typeof p === "string" ? new TextEncoder().encode(p) : new Uint8Array(p.buffer, p.byteOffset, p.byteLength);
    for (let k = 0; k < b.length; k++) {
      h1 = Math.imul(h1 ^ b[k], 0x01000193);
      h2 = Math.imul(h2 ^ b[k], 0x5bd1e995) ^ (h2 >>> 15);
    }
    h1 = Math.imul(h1 ^ 0xff, 0x01000193);
  }
  return (h1 >>> 0).toString(16) + (h2 >>> 0).toString(16);
}

/**
 * The network's clusters, each holding at least one population. `whole` (`?perfOff=clusters`)
 * gives one cluster of everything, the network as it came; `oneMarket` (`?perfOff=split`) joins the
 * market's zones into one node. Lines and stations no population can
 * reach belong to no cluster: nobody rides them, so their figures are zero.
 */
export function splitClusters(n: DemandNetwork, cities: { id: CityId; box: Box }[], marketZones: Float32Array | null, rounds: number, whole = false, oneMarket = false): Cluster[] {
  const market = !!marketZones;
  const S = n.stXY.length / 2, L = n.lineN.length, P = cities.length;
  const lineOff = new Uint32Array(L + 1);
  for (let l = 0; l < L; l++) lineOff[l + 1] = lineOff[l] + n.lineN[l];
  const M = marketZones ? marketZones.length / 4 : 0;
  const parent = new Int32Array(S + P + Math.max(M, 1)).map((_, i) => i);
  const find = (i: number) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const join = (a: number, b: number) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };
  // T-409: one node per zone of the market (`IntercityApi.zone_boxes`, its 6 km zones), joined to
  // the stations within `MARKET_REACH_M` of its sites: every station in its access lists. So every
  // zone's stations are in one cluster, and the market's pairs split by cluster exactly
  const MARKET = S + P;
  if (whole) {
    for (let i = 1; i < parent.length; i++) join(0, i);
  } else {
    for (let l = 0; l < L; l++) for (let k = lineOff[l] + 1; k < lineOff[l + 1]; k++) join(n.stops[k - 1], n.stops[k]);
    if (n.complex && n.complex.length === S) {
      const first = new Map<number, number>();
      for (let s = 0; s < S; s++) {
        const f = first.get(n.complex[s]);
        if (f === undefined) first.set(n.complex[s], s);
        else join(f, s);
      }
    }
    const grid = new Map<string, number[]>();
    const key = (gx: number, gy: number) => `${gx},${gy}`;
    for (let s = 0; s < S; s++) {
      const gx = Math.floor(n.stXY[2 * s] / JOIN_M), gy = Math.floor(n.stXY[2 * s + 1] / JOIN_M);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        for (const t of grid.get(key(gx + dx, gy + dy)) ?? []) {
          if (Math.hypot(n.stXY[2 * s] - n.stXY[2 * t], n.stXY[2 * s + 1] - n.stXY[2 * t + 1]) <= JOIN_M) join(s, t);
        }
      }
      const k = key(gx, gy);
      const cell = grid.get(k);
      if (cell) cell.push(s);
      else grid.set(k, [s]);
    }
    cities.forEach(({ id, box }, i) => {
      if (!box) return;
      for (let s = 0; s < S; s++) {
        const p = worldToCity(n.stXY[2 * s], n.stXY[2 * s + 1], id);
        if (p.x >= box[0] - CITY_REACH_M && p.x <= box[2] + CITY_REACH_M && p.y >= box[1] - CITY_REACH_M && p.y <= box[3] + CITY_REACH_M) join(s, S + i);
      }
    });
    if (marketZones && S) {
      const G = MARKET_REACH_M, cells = new Map<string, number[]>();
      for (let z = 0; z < M; z++) {
        const [w, so, e, no] = marketZones.subarray(4 * z, 4 * z + 4);
        if (!(w <= e)) continue;
        for (let gx = Math.floor((w - G) / G); gx <= Math.floor((e + G) / G); gx++) for (let gy = Math.floor((so - G) / G); gy <= Math.floor((no + G) / G); gy++) {
          const k = `${gx},${gy}`, c = cells.get(k);
          if (c) c.push(z); else cells.set(k, [z]);
        }
      }
      for (let s = 0; s < S; s++) {
        const { x, y } = worldToMarket(n.stXY[2 * s], n.stXY[2 * s + 1]);
        for (const z of cells.get(`${Math.floor(x / G)},${Math.floor(y / G)}`) ?? []) {
          const b = 4 * z;
          if (x >= marketZones[b] - G && x <= marketZones[b + 2] + G && y >= marketZones[b + 1] - G && y <= marketZones[b + 3] + G) join(s, MARKET + z);
        }
      }
    }
    // `?perfOff=split`: the market's zones as one node, as before T-409
    if (oneMarket) for (let k = 1; k < M; k++) join(MARKET, MARKET + k);
  }
  // the clusters, in the order of their first population
  const byRoot = new Map<number, { cities: CityId[]; market: boolean; zones: number[] }>();
  const of = (r: number) => {
    let c = byRoot.get(r);
    if (!c) byRoot.set(r, (c = { cities: [], market: false, zones: [] }));
    return c;
  };
  cities.forEach(({ id }, i) => of(find(S + i)).cities.push(id));
  if (market) {
    // a market zone no station reaches is in no cluster (its pairs carry no rail trip); if none
    // is reached, the market is a cluster of its own with no lines
    const railed = new Set<number>();
    for (let s = 0; s < S; s++) railed.add(find(s));
    let any = whole;
    for (let z = 0; z < M; z++) {
      const r = find(MARKET + z);
      if (!whole && !railed.has(r)) continue;
      const c = of(r);
      c.market = true;
      if (!whole) c.zones.push(z);
      any = true;
    }
    if (!any) of(find(MARKET)).market = true;
  }
  const out: Cluster[] = [];
  const frameKey = JSON.stringify(n.frame);
  for (const [root, c] of byRoot) {
    const stations: number[] = [];
    for (let s = 0; s < S; s++) if (find(s) === root) stations.push(s);
    const lines: number[] = [];
    for (let l = 0; l < L; l++) if (whole || (n.lineN[l] > 0 && find(n.stops[lineOff[l]]) === root)) lines.push(l);
    out.push(subNetwork(n, lineOff, c.cities, c.market, c.zones.join(","), stations, lines, rounds, frameKey));
  }
  return out;
}

function subNetwork(n: DemandNetwork, lineOff: Uint32Array, cities: CityId[], market: boolean, marketKey: string, stations: number[], lines: number[], rounds: number, frameKey: string): Cluster {
  const S = n.stXY.length / 2;
  const stLocal = new Int32Array(S).fill(-1);
  stations.forEach((s, i) => (stLocal[s] = i));
  const stXY = new Float32Array(2 * stations.length);
  stations.forEach((s, i) => { stXY[2 * i] = n.stXY[2 * s]; stXY[2 * i + 1] = n.stXY[2 * s + 1]; });
  // complexes numbered by their first station here: the numbers carry no meaning past equality
  let complex = new Uint32Array(0);
  if (n.complex && n.complex.length === S) {
    const first = new Map<number, number>();
    complex = Uint32Array.from(stations, (s, i) => {
      const f = first.get(n.complex[s]);
      if (f !== undefined) return f;
      first.set(n.complex[s], i);
      return i;
    });
  }
  const nStops = lines.reduce((a, l) => a + n.lineN[l], 0);
  const lineN = Uint32Array.from(lines, (l) => n.lineN[l]);
  const stops = new Uint32Array(nStops), times = new Float32Array(6 * nStops), hopKm = new Float32Array(2 * nStops);
  const tph = new Float32Array(3 * lines.length), cars = new Float32Array(lines.length);
  const nodeGlobal = new Uint32Array(2 * nStops), nodeLocal = new Int32Array(2 * n.stops.length).fill(-1);
  let o = 0;
  lines.forEach((l, i) => {
    const m = n.lineN[l], g = lineOff[l];
    for (let k = 0; k < m; k++) stops[o + k] = stLocal[n.stops[g + k]];
    times.set(n.times.subarray(6 * g, 6 * (g + m)), 6 * o);
    // route nodes: each line's 2m nodes in a row, its runs' stops (`hopKm` alike)
    hopKm.set(n.hopKm.subarray(2 * g, 2 * (g + m)), 2 * o);
    for (let r = 0; r < 2 * m; r++) { nodeGlobal[2 * o + r] = 2 * g + r; nodeLocal[2 * g + r] = 2 * o + r; }
    tph.set(n.tph.subarray(3 * l, 3 * l + 3), 3 * i);
    cars[i] = n.cars[l];
    o += m;
  });
  const net: DemandNetwork = { frame: n.frame, stXY, complex, lineN, stops, times, tph, cars, fare: n.fare, hopKm };
  const head = `${cities.join(",")}|${market}|${marketKey}|${rounds}|${n.fare.base}|${n.fare.perKm}|${frameKey}`;
  const sig = hashOf([head, stXY, complex, lineN, stops, times, tph, cars, hopKm]);
  const levelSig = [0, 1, 2].map((lev) => {
    const lt = new Float32Array(2 * nStops), lp = new Float32Array(lines.length);
    let a = 0;
    lineN.forEach((m, i) => {
      lt.set(times.subarray(6 * a + lev * m, 6 * a + (lev + 1) * m), 2 * a);
      lt.set(times.subarray(6 * a + (3 + lev) * m, 6 * a + (4 + lev) * m), 2 * a + m);
      lp[i] = tph[3 * i + lev];
      a += m;
    });
    return hashOf([`${head}|level ${lev}`, stXY, complex, lineN, stops, lt, lp, cars, hopKm]);
  });
  const psig = PERIODS.map((p) => levelSig[LEVEL[p.demand]]);
  return { cities, market, marketKey, net, lines: Uint32Array.from(lines), stLocal, nodeGlobal, nodeLocal, sig, psig };
}
