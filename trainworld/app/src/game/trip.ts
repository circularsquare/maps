// The trip explorer (T-195; docs/demand.md "The trip explorer", docs/ui-inspectors.md). One trip
// between two points on the map, read as a commute from a home at the first point to work at the
// second, the way the demand model decides it: the train's routes through the model's strategies,
// the drive, and the chance of each mode in each period. The numbers are the model's own
// (`DemandApi.trip` in sim/src/demand/trip.rs, asked of the demand workers); this file only names
// them. Main thread only.

import { signal } from "@preact/signals";
import { lngLatToPack } from "./coords";
import { PERIODS } from "./periods";
import type { Selection } from "./types";
import { world } from "./world";

/** One leg of a route: a ride on a line, or a walk between stations after a ride. */
export interface TripLeg {
  kind: "ride" | "walk";
  /** the line ridden (a ride's) */
  line: string;
  /** snapshot stations at the two ends */
  from: string;
  to: string;
  /** every station passed, in order, from `from` to `to` (a ride's; a walk's two ends) */
  stations: string[];
  /** a ride's two ends as indices into its line's stops (run 0's order: `k0 > k1` rides run 1),
   * so the map can draw it along the line's track (T-208); -1 for a walk */
  k0: number;
  k1: number;
  stops: number;
  /** real minutes riding or walking, and perceived (a ride's with crowding, without its fare) */
  min: number;
  perceived: number;
  /** a ride's distance fare, perceived minutes */
  farePerceived: number;
  /** the wait before a ride: real minutes (half the combined headway of the useful trains there)
   * and perceived */
  waitMin: number;
  waitPerceived: number;
}

/** A sequence of boardings the train riders take, and its share of them. */
export interface TripRoute {
  /** share of this trip's train riders on the route (0-1) */
  share: number;
  /** index into `TripPeriod.pairs` */
  pair: number;
  legs: TripLeg[];
}

/** A pair of stations to board at and leave from, with the walks to and from them. */
export interface TripPair {
  access: string;
  egress: string;
  /** straight-line metres and perceived minutes of the walk to `access` and from `egress` */
  accessM: number;
  accessPerceived: number;
  egressM: number;
  egressPerceived: number;
  /** the two walks' real minutes (6 km/h, SPEC 4.3); the access entry of a ride has the drive's */
  accessMin: number;
  egressMin: number;
  /** the access entry is a ride (T-371: a car, bus, bike or lift to the station, one mode) rather
   * than a walk, and then the perceived minutes beyond the drive's: the ride's fixed cost and the
   * parking or drop-off at the station */
  accessRide: boolean;
  rideExtra: number;
  /** perceived minutes station to station (waits, rides, changes, distance fare; no base fare) */
  strategy: number;
  share: number;
}

/** Why the train is not a choice for a trip. */
export type NoRail = "" | "noStationStart" | "noStationEnd" | "walksTooLong" | "noRoute";
const WHY: NoRail[] = ["", "noStationStart", "noStationEnd", "walksTooLong", "noRoute"];

/** The trip in one period. Minutes are perceived where named so, else real. */
export interface TripPeriod {
  /** the crowding round the period's costs are from */
  round: number;
  /** world metres of the cells the two points fell in */
  from: [number, number];
  to: [number, number];
  /** straight line between the two cells, and the drive's distance (between the two areas the
   * model groups the places into), metres */
  distM: number;
  driveDistM: number;
  walkShare: number;
  driveMin: number;
  drivePerceived: number;
  parkPerceived: number;
  roadPerceived: number;
  railOffered: boolean;
  why: NoRail;
  /** the train's cost (the logsum over station pairs, perceived minutes); NaN if not offered */
  railPerceived: number;
  /** the train's share of the trips that do not walk, and its utility against the road */
  railOfMotor: number;
  utility: number;
  pWalk: number;
  pTrain: number;
  pDrive: number;
  baseFarePerceived: number;
  minPerUsd: number;
  /** the cheapest perceived walk at each end */
  walkStartPerceived: number;
  walkEndPerceived: number;
  pairs: TripPair[];
  routes: TripRoute[];
  /** share of the train riders on routes too small to list */
  cut: number;
}

export interface Trip {
  version: number;
  round: number;
  /** "x,y>x,y": the points asked about */
  key: string;
  /** per period (game/periods.ts order); null where the trip could not be read */
  periods: (TripPeriod | null)[];
  /** a point is outside the city's cells */
  outside: boolean;
}

/** Route node -> line, snapshot station and its index in the line's stops, from the solve's layout
 * (as `transfers`). */
export type RouteNodes = { line: string; station: string; k: number }[];

const HEAD = 32, PAIR_LEN = 14, LEG_LEN = 12;

/** Read one period's packed answer (layout: `DemandApi.trip`). */
export function readTripPeriod(v: Float64Array, at: RouteNodes): TripPeriod | null {
  if (v.length < HEAD || v[0] !== 1) return null;
  const st = (rn: number) => (rn >= 0 && at[rn] ? at[rn].station : "");
  const np = v[24], nr = v[25];
  const pairs: TripPair[] = [];
  for (let k = 0; k < np; k++) {
    const p = v.subarray(HEAD + k * PAIR_LEN, HEAD + (k + 1) * PAIR_LEN);
    pairs.push({ access: st(p[2]), egress: st(p[3]), accessM: p[4], accessPerceived: p[5], egressM: p[6], egressPerceived: p[7], strategy: p[8], share: p[9], accessMin: p[10], egressMin: p[11], accessRide: p[12] === 1, rideExtra: p[13] });
  }
  const routes: TripRoute[] = [];
  let o = HEAD + np * PAIR_LEN;
  for (let k = 0; k < nr; k++) {
    const pair = v[o], share = v[o + 1], n = v[o + 2];
    o += 3;
    const legs: TripLeg[] = [];
    for (let i = 0; i < n; i++, o += LEG_LEN) {
      const l = v.subarray(o, o + LEG_LEN);
      if (l[0] === 1) {
        const stations: string[] = [];
        for (let rn = l[1]; rn <= l[2]; rn++) stations.push(st(rn));
        legs.push({ kind: "ride", line: at[l[1]]?.line ?? "", from: st(l[1]), to: st(l[2]), stations, k0: at[l[1]]?.k ?? -1, k1: at[l[2]]?.k ?? -1, stops: l[10], min: l[5], perceived: l[6], farePerceived: l[7], waitMin: l[8], waitPerceived: l[9] });
      } else {
        const from = st(l[1]), to = st(l[2]);
        legs.push({ kind: "walk", line: "", from, to, stations: [from, to], k0: -1, k1: -1, stops: 0, min: l[5], perceived: l[6], farePerceived: 0, waitMin: 0, waitPerceived: 0 });
      }
    }
    routes.push({ share, pair, legs });
  }
  const offered = v[13] === 1;
  return {
    round: v[1], from: [v[27], v[28]], to: [v[29], v[30]], distM: v[6], driveDistM: v[7], walkShare: v[8],
    driveMin: v[9], drivePerceived: v[10], parkPerceived: v[11], roadPerceived: v[12],
    railOffered: offered, why: WHY[v[26]] ?? "", railPerceived: offered ? v[14] : NaN, railOfMotor: v[15], utility: v[16],
    pWalk: v[17], pTrain: v[18], pDrive: v[19], baseFarePerceived: v[22], minPerUsd: v[23], walkStartPerceived: v[20], walkEndPerceived: v[21], pairs, routes, cut: v[31],
  };
}

/** A route's totals: real minutes driving to the station (a ride access), walking, waiting, riding
 * and walking between stations, the changes, and the fare in dollars (base fare included). */
export function routeTotals(t: TripPeriod, r: TripRoute) {
  const pair = t.pairs[r.pair];
  const driveMin = pair?.accessRide ? pair.accessMin : 0;
  const walkM = pair ? (pair.accessRide ? 0 : pair.accessM) + pair.egressM : 0;
  const walkMin = pair ? (pair.accessRide ? 0 : pair.accessMin) + pair.egressMin : 0;
  let wait = 0, ride = 0, xfer = 0, fare = t.baseFarePerceived, rides = 0;
  for (const l of r.legs) {
    if (l.kind === "ride") { wait += l.waitMin; ride += l.min; fare += l.farePerceived; rides++; }
    else xfer += l.min;
  }
  return { driveMin, walkMin, walkM, wait, ride, xfer, changes: Math.max(0, rides - 1), fareUsd: fare / t.minPerUsd, total: driveMin + walkMin + wait + ride + xfer };
}

/** One term of a route's perceived total: what it is, its real amount (minutes, or dollars for the
 * fare) with the weight it counts at, and its perceived minutes. */
export interface CostTerm {
  label: string;
  real: number;
  weight: number;
  perceived: number;
  unit: "min" | "usd";
}

/** A route's perceived total as terms, from the worker's own perceived minutes: the walks (or the
 * drive to the station), the waits, the rides with their crowding, the walks between stations and
 * the fare. A change costs its wait and walk only (T-557); the base fare is in the fare. */
export function routeTerms(t: TripPeriod, r: TripRoute): CostTerm[] {
  const pair = t.pairs[r.pair];
  const tot = routeTotals(t, r);
  const out: CostTerm[] = [];
  if (pair) {
    if (pair.accessRide) out.push({ label: "drive", real: pair.accessMin, weight: pair.accessMin > 0 ? (pair.accessPerceived - pair.rideExtra) / pair.accessMin : 1, perceived: pair.accessPerceived - pair.rideExtra, unit: "min" });
    const walkPerc = pair.egressPerceived + (pair.accessRide ? 0 : pair.accessPerceived);
    if (tot.walkMin > 0) out.push({ label: "walk", real: tot.walkMin, weight: walkPerc / tot.walkMin, perceived: walkPerc, unit: "min" });
    if (pair.accessRide) out.push({ label: "station", real: 0, weight: 1, perceived: pair.rideExtra, unit: "min" });
  }
  let waitP = 0, rideP = 0, xferP = 0;
  for (const l of r.legs) {
    if (l.kind === "ride") { waitP += l.waitPerceived; rideP += l.perceived; }
    else xferP += l.perceived;
  }
  out.push({ label: "wait", real: tot.wait, weight: tot.wait > 0 ? waitP / tot.wait : 1, perceived: waitP, unit: "min" });
  out.push({ label: "ride", real: tot.ride, weight: tot.ride > 0 ? rideP / tot.ride : 1, perceived: rideP, unit: "min" });
  if (tot.xfer > 0) out.push({ label: "walk between stations", real: tot.xfer, weight: xferP / tot.xfer, perceived: xferP, unit: "min" });
  out.push({ label: "fare", real: tot.fareUsd, weight: t.minPerUsd, perceived: tot.fareUsd * t.minPerUsd, unit: "usd" });
  return out;
}

let query: ((from: [number, number], to: [number, number]) => Promise<Trip | null>) | null = null;
/** Set by workers/demandClient.ts when the pool starts. */
export function setTripQuery(q: typeof query) {
  query = q;
}
/** Ask the demand workers about a trip, for the solve `demandView` shows. */
export const askTrip = (from: [number, number], to: [number, number]): Promise<Trip | null> => query?.(from, to) ?? Promise.resolve(null);

/** The trip shown (asked for by map/tripView.ts when the selection or the demand result changes). */
export const tripView = signal<{ key: string; trip: Trip | null; asking: boolean } | null>(null);
/** A trip opened from a bubble selection by clicking a far-end bubble (T-208): the trip selection and the one it came from, for the inspector's way back. */
export const tripOpenedFrom = signal<{ trip: Selection; from: Selection; what: string } | null>(null);
/** The period the explorer shows: null follows the clock. */
export const tripPeriod = signal<number | null>(null);
export const PERIOD_NAMES = PERIODS.map((p) => p.name);

/** Where a trip end picked as a station (T-221) is asked about: the station's own position, world
 * metres. The model reads a point as the cell nearest it, so this is the station's own cell, and
 * the model's subzones are the cells of a zone that share their nearest station: the station's cell
 * is in the subzone whose nearest station is this one, so its riders walk the least and board here
 * most. (The busiest cell of its catchment was the other choice; it can be nearer another station,
 * and the trip would then mostly start from that one instead of the station clicked.) */
export function stationPoint(id: string): [number, number] | null {
  const s = world.peek()?.save.stations.find((s) => s.id === id);
  if (!s) return null;
  const p = lngLatToPack(s.lng, s.lat);
  return [p.x, p.y];
}

/** The name of a station picked as a trip's end, if one was. */
export const stationName = (id: string | undefined) => (id ? world.peek()?.save.stations.find((s) => s.id === id)?.name ?? null : null);

export const tripKey = (from: [number, number], to: [number, number]) => `${Math.round(from[0])},${Math.round(from[1])}>${Math.round(to[0])},${Math.round(to[1])}`;
