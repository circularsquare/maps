// T-326: the network split into clusters that share no rider. A cluster is the populations (city
// packs, the regional market) and the stations and lines they can reach, joined by lines, station
// complexes and stations close enough to walk between. A cluster's figures depend only on its own
// lines and populations, so each is solved on its own sub-network, and one whose sub-network has
// not changed keeps its result (docs/demand.md "Clusters"). Reach is judged generously: joining two
// clusters that share nothing costs only time, splitting two that share riders would change figures.

import { CITIES, worldToCity, worldToLonLat, zoneBox, type CityId } from "../game/cities";
import { PERIODS } from "../game/periods";
import { curveKnots } from "../game/fareCurve";
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

const MARKET_BOXES = CITIES.map((c) => [c.id, zoneBox(c)] as const);

/** T-396, T-377: the zones whose market riders can use a station at (lon, lat): within
 * `MARKET_REACH_M` of the zone's box. What opens a zone's market pack (workers/demandClient.ts). */
export function marketZonesNear(lon: number, lat: number): CityId[] {
  const dy = MARKET_REACH_M / 111_320, dx = dy / Math.cos((lat * Math.PI) / 180);
  return MARKET_BOXES.filter(([, [w, s, e, n]]) => lon >= w - dx && lon <= e + dx && lat >= s - dy && lat <= n + dy).map(([id]) => id);
}

export interface Cluster {
  /** the job's cities in it, in the order given */
  cities: CityId[];
  /** the regional market's riders can reach it */
  market: boolean;
  /** T-409: the market's 6 km zones (their world ids, T-377) whose stations are in it, ascending,
   * joined by commas (the market's part for this cluster is kept by it); "" for the market no
   * station reaches */
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
export function splitClusters(n: DemandNetwork, cities: { id: CityId; box: Box }[], marketZones: Float64Array | null, rounds: number, whole = false, oneMarket = false): Cluster[] {
  const market = !!marketZones;
  const S = n.stXY.length / 2, L = n.lineN.length, P = cities.length;
  const lineOff = new Uint32Array(L + 1);
  for (let l = 0; l < L; l++) lineOff[l + 1] = lineOff[l] + n.lineN[l];
  // T-377: per open 6 km zone its world id, then its sites' box in degrees (`IntercityApi.zone_boxes`)
  const M = marketZones ? marketZones.length / 5 : 0;
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
      // a grid of `GD` degrees; each zone's box widened by the reach at its own latitude
      const GD = 0.5, dyR = MARKET_REACH_M / 111_320, cells = new Map<string, number[]>();
      const pad = (z: number) => {
        const lat = Math.min(85, Math.max(Math.abs(marketZones[5 * z + 2]), Math.abs(marketZones[5 * z + 4])));
        return [dyR / Math.cos((lat * Math.PI) / 180), dyR];
      };
      for (let z = 0; z < M; z++) {
        const [w, so, e, no] = marketZones.subarray(5 * z + 1, 5 * z + 5);
        if (!(w <= e)) continue;
        const [dx, dy] = pad(z);
        for (let gx = Math.floor((w - dx) / GD); gx <= Math.floor((e + dx) / GD); gx++) for (let gy = Math.floor((so - dy) / GD); gy <= Math.floor((no + dy) / GD); gy++) {
          const k = `${gx},${gy}`, c = cells.get(k);
          if (c) c.push(z); else cells.set(k, [z]);
        }
      }
      for (let s = 0; s < S; s++) {
        const [x, y] = worldToLonLat(n.stXY[2 * s], n.stXY[2 * s + 1]);
        for (const z of cells.get(`${Math.floor(x / GD)},${Math.floor(y / GD)}`) ?? []) {
          const b = 5 * z, [dx, dy] = pad(z);
          if (x >= marketZones[b + 1] - dx && x <= marketZones[b + 3] + dx && y >= marketZones[b + 2] - dy && y <= marketZones[b + 4] + dy) join(s, MARKET + z);
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
      if (!whole) c.zones.push(marketZones![5 * z]);
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
    out.push(subNetwork(n, lineOff, c.cities, c.market, c.zones.sort((a, b) => a - b).join(","), stations, lines, rounds, frameKey));
  }
  return out;
}

/**
 * T-756: a population's own network inside its cluster (notes/T-756.md). Each of the cluster's
 * lines with a stop it reaches runs from the stop before its first such stop to the stop after its
 * last (all stops between kept, so no rider changes trains where a line leaves the reach and comes
 * back); the stops outside are only those pads, which no rider of the population boards at.
 */
export interface PopView {
  net: DemandNetwork;
  /** snapshot station -> station of `net`, or -1 */
  stLocal: Int32Array;
  /** route node of `net` -> route node of the whole network */
  nodeGlobal: Uint32Array;
  /** route nodes of `net` that start a run at a pad: their boardings in the crowding input are the
   * riders already aboard there (the whole line's load on the hop leaving it), as the left-behind
   * model, which runs each train from its first stop, needs */
  pads: Uint32Array;
}

/** The view of a population that solves on its whole cluster (the market, `?perfOff=reach`). */
export function clusterView(c: Cluster): PopView {
  return { net: c.net, stLocal: c.stLocal, nodeGlobal: c.nodeGlobal, pads: new Uint32Array(0) };
}

/** T-756: the snapshot stations a city reaches (`CITY_REACH_M` of its cells' box), 1 each. */
export function cityReach(n: DemandNetwork, id: CityId, box: Box): Uint8Array {
  const S = n.stXY.length / 2, out = new Uint8Array(S);
  if (!box) return out;
  for (let s = 0; s < S; s++) {
    const p = worldToCity(n.stXY[2 * s], n.stXY[2 * s + 1], id);
    if (p.x >= box[0] - CITY_REACH_M && p.x <= box[2] + CITY_REACH_M && p.y >= box[1] - CITY_REACH_M && p.y <= box[3] + CITY_REACH_M) out[s] = 1;
  }
  return out;
}

/** T-756: the network of a population reaching the stations marked in `reach`, over the cluster's lines. */
export function reachView(n: DemandNetwork, c: Cluster, reach: Uint8Array): PopView {
  const lineOff = new Uint32Array(n.lineN.length + 1);
  for (let l = 0; l < n.lineN.length; l++) lineOff[l + 1] = lineOff[l] + n.lineN[l];
  const spans: [number, number, number][] = [];
  for (const l of c.lines) {
    const m = n.lineN[l], g = lineOff[l];
    let a = -1, b = -1;
    for (let k = 0; k < m; k++) if (reach[n.stops[g + k]]) { if (a < 0) a = k; b = k; }
    if (a < 0) continue;
    a = Math.max(0, a - 1); b = Math.min(m - 1, b + 1);
    if (b > a) spans.push([l, a, b]);
  }
  const S = n.stXY.length / 2, used = new Uint8Array(S);
  for (const [l, a, b] of spans) for (let k = a; k <= b; k++) used[n.stops[lineOff[l] + k]] = 1;
  const stations: number[] = [];
  for (let s = 0; s < S; s++) if (used[s]) stations.push(s);
  const stLocal = new Int32Array(S).fill(-1);
  stations.forEach((s, i) => (stLocal[s] = i));
  const stXY = new Float32Array(2 * stations.length);
  stations.forEach((s, i) => { stXY[2 * i] = n.stXY[2 * s]; stXY[2 * i + 1] = n.stXY[2 * s + 1]; });
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
  const nStops = spans.reduce((t, [, a, b]) => t + b - a + 1, 0);
  const lineN = Uint32Array.from(spans, ([, a, b]) => b - a + 1);
  const stops = new Uint32Array(nStops), times = new Float32Array(6 * nStops), hopKm = new Float32Array(2 * nStops);
  const tph = new Float32Array(3 * spans.length), cars = new Float32Array(spans.length);
  const nodeGlobal = new Uint32Array(2 * nStops), pads: number[] = [];
  let o = 0;
  spans.forEach(([l, a, b], i) => {
    const m = n.lineN[l], g = lineOff[l], p = b - a + 1;
    for (let k = 0; k < p; k++) stops[o + k] = stLocal[n.stops[g + a + k]];
    for (let r = 0; r < 6; r++) {
      times.set(n.times.subarray(6 * g + r * m + a, 6 * g + r * m + b + 1), 6 * o + r * p);
      // run 0 ends at the span's last stop, run 1 at its first; a whole circle keeps its ride on
      // round through its first stop (T-798)
      if (a > 0 || b < m - 1) times[6 * o + r * p + (r < 3 ? p - 1 : 0)] = 0;
    }
    // route nodes: the span's p stops in order, then reversed (the whole line's alike)
    for (let j = 0; j < p; j++) {
      nodeGlobal[2 * o + j] = 2 * g + a + j;
      nodeGlobal[2 * o + p + j] = 2 * g + m + (m - 1 - (b - j));
      hopKm[2 * o + j] = j + 1 < p ? n.hopKm[2 * g + a + j] : 0;
      hopKm[2 * o + p + j] = j + 1 < p ? n.hopKm[2 * g + m + (m - 1 - (b - j))] : 0;
    }
    if (a > 0) pads.push(2 * o);
    if (b < m - 1) pads.push(2 * o + p);
    tph.set(n.tph.subarray(3 * l, 3 * l + 3), 3 * i);
    cars[i] = n.cars[l];
    o += p;
  });
  const net: DemandNetwork = { frame: n.frame, stXY, complex, lineN, stops, times, tph, cars, fare: n.fare, hopKm, span: spanOf(n, nodeGlobal) };
  return { net, stLocal, nodeGlobal, pads: Uint32Array.from(pads) };
}

/** T-842: the whole network's span for the route nodes `nodeGlobal` names, period by period. */
function spanOf(n: DemandNetwork, nodeGlobal: Uint32Array): Float32Array {
  const all = 2 * n.stops.length, here = nodeGlobal.length;
  if (!n.span || n.span.length !== PERIODS.length * all) return new Float32Array(0);
  const out = new Float32Array(PERIODS.length * here);
  for (let q = 0; q < PERIODS.length; q++) for (let r = 0; r < here; r++) out[q * here + r] = n.span[q * all + nodeGlobal[r]];
  return out.some((v) => v !== 1) ? out : new Float32Array(0);
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
  const span = spanOf(n, nodeGlobal);
  const net: DemandNetwork = { frame: n.frame, stXY, complex, lineN, stops, times, tph, cars, fare: n.fare, hopKm, span };
  const head = `${cities.join(",")}|${market}|${marketKey}|${rounds}|${n.fare.base}|${n.fare.perKm}|${frameKey}${curveKnots(n.fare).join(",")}`;
  const sig = hashOf([head, stXY, complex, lineN, stops, times, tph, cars, hopKm, span]);
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
  // T-842: a period's own span, where there is one
  const nr = 2 * nStops;
  const psig = PERIODS.map((p, q) => span.length ? hashOf([levelSig[LEVEL[p.demand]], span.subarray(q * nr, (q + 1) * nr)]) : levelSig[LEVEL[p.demand]]);
  return { cities, market, marketKey, net, lines: Uint32Array.from(lines), stLocal, nodeGlobal, nodeLocal, sig, psig };
}
