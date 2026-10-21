// The trip explorer (T-195; docs/demand-views.md "The trip explorer", docs/ui-inspectors.md). Trips
// between two places on the map, from homes at the first to destinations at the second, as the
// view it was opened from counts them (T-586): the purpose filter's trips a day by mode between
// the two places (the same sums as the bubbles' hovers, `tripMix`), then one representative trip
// the way the demand model decides it: the train's routes through the model's strategies, the
// drive, and the chance of each mode in each period. The numbers are the model's own
// (`DemandApi.trip` and `EverydayApi.trip` in sim/src/demand/trip.rs, and the selection's far end,
// asked of the demand workers); this file only names them. Main thread only.

import { signal } from "@preact/signals";
import { lngLatToPack } from "./coords";
import type { Flows } from "./demand";
import { PERIODS } from "./periods";
import { EVERYDAY_KEYS, FILTER_PURPOSES, shownKey, type PurposeFilter, type PurposeKey } from "./purposes";
import type { Selection } from "./types";
import { world } from "./world";

/** One leg of a route: a ride on a line, or a walk after a ride, to another station or between the
 * platforms of one (T-717). */
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
  /** a ride on a circle through its first stop (T-798): the line's stop count, else 0; then
   * `k0 > k1` rides run 0 round past the last stop and `k0 < k1` run 1 */
  wrapN?: number;
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
  /** a ride's fixed cost of the change onto it, perceived minutes (0 on the first ride; T-717) */
  changePerceived: number;
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
  /** world metres of the cells the two points fell in, and the cells (pack ids) */
  from: [number, number];
  to: [number, number];
  cells: [number, number];
  /** walking the whole way between the two cells, real minutes (T-628) */
  walkMin: number;
  /** the weight a minute of walking the whole way counts at */
  walkWeight: number;
  /** an everyday purpose's trips a day between the two cells' areas (0: a commute, or none) */
  pairTrips: number;
  /** straight line between the two cells, and the drive's distance, metres */
  distM: number;
  driveDistM: number;
  /** the drive's length by road, km */
  driveRoadKm: number;
  driveMin: number;
  drivePerceived: number;
  parkPerceived: number;
  /** the car's ownership cost per trip (T-620), included in `roadPerceived` */
  carOwnPerceived: number;
  roadPerceived: number;
  railOffered: boolean;
  why: NoRail;
  /** the train's cost (the logsum over station pairs, perceived minutes); NaN if not offered */
  railPerceived: number;
  /** the train's share of the trips that do not walk, and its utility against the road */
  railOfMotor: number;
  utility: number;
  /** shares of all trips walking the whole way (T-558: walking is a mode in the choice), by
   * train, and by car or other */
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
  /** T-720: the same shares as the demand views count them, for the nearby homes and places the
   * model groups the two cells with, and that group's walk to its nearest station at the start,
   * real minutes (-1 none). Everything else is the two cells' own trip. */
  groupWalk: number;
  groupTrain: number;
  groupDrive: number;
  groupWalkStartMin: number;
}

export interface Trip {
  version: number;
  round: number;
  /** "x,y>x,y": the points asked about */
  key: string;
  /** -1 a commute, else the everyday purpose (`EVERYDAY_KEYS` index) the trip was read as */
  purpose: number;
  /** per period (game/periods.ts order); null where the trip could not be read */
  periods: (TripPeriod | null)[];
  /** a point is outside the city's cells */
  outside: boolean;
}

/** Route node -> line, snapshot station and its index in the line's stops, from the solve's layout
 * (as `transfers`); `o` the line's first route node and `n` its stops. */
export type RouteNodes = { line: string; station: string; k: number; o?: number; n?: number }[];

/** The route node after `rn` on its run: past a circle's last stop, its first (T-798). */
function nextNode(at: RouteNodes, rn: number): number {
  const a = at[rn];
  if (!a || a.o === undefined || !a.n) return rn + 1;
  const start = a.o + (rn - a.o < a.n ? 0 : a.n);
  return rn === start + a.n - 1 ? start : rn + 1;
}

const HEAD = 41, PAIR_LEN = 14, LEG_LEN = 12;

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
        // a ride on a circle may carry on round past its last stop (T-798): its first stop is
        // both route nodes there, named once
        for (let rn = l[1], i = 0; i <= at.length; rn = nextNode(at, rn), i++) {
          if (stations[stations.length - 1] !== st(rn)) stations.push(st(rn));
          if (rn === l[2]) break;
        }
        const wrapN = l[1] > l[2] ? at[l[1]]?.n ?? 0 : 0;
        legs.push({ kind: "ride", line: at[l[1]]?.line ?? "", from: st(l[1]), to: st(l[2]), stations, k0: at[l[1]]?.k ?? -1, k1: at[l[2]]?.k ?? -1, ...(wrapN ? { wrapN } : {}), stops: l[10], min: l[5], perceived: l[6], farePerceived: l[7], waitMin: l[8], waitPerceived: l[9], changePerceived: l[11] });
      } else {
        const from = st(l[1]), to = st(l[2]);
        legs.push({ kind: "walk", line: "", from, to, stations: [from, to], k0: -1, k1: -1, stops: 0, min: l[5], perceived: l[6], farePerceived: 0, waitMin: 0, waitPerceived: 0, changePerceived: 0 });
      }
    }
    routes.push({ share, pair, legs });
  }
  const offered = v[13] === 1;
  return {
    round: v[1], from: [v[27], v[28]], to: [v[29], v[30]], cells: [v[2], v[3]], walkMin: v[33], walkWeight: v[36], pairTrips: v[34], distM: v[6], driveDistM: v[7], driveRoadKm: v[35],
    driveMin: v[9], drivePerceived: v[10], parkPerceived: v[11], carOwnPerceived: v[32], roadPerceived: v[12],
    railOffered: offered, why: WHY[v[26]] ?? "", railPerceived: offered ? v[14] : NaN, railOfMotor: v[15], utility: v[16],
    pWalk: v[17], pTrain: v[18], pDrive: v[19], baseFarePerceived: v[22], minPerUsd: v[23], walkStartPerceived: v[20], walkEndPerceived: v[21], pairs, routes, cut: v[31],
    groupWalk: v[37], groupTrain: v[38], groupDrive: v[39], groupWalkStartMin: v[40],
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

/** One term of a route's perceived total: what it is, its real amount (minutes, dollars for the
 * fare, or a count of changes) with the weight it counts at, and its perceived minutes. */
export interface CostTerm {
  label: string;
  real: number;
  weight: number;
  perceived: number;
  unit: "min" | "usd" | "count";
}

/** A route's perceived total as terms, from the worker's own perceived minutes: the walks (or the
 * drive to the station), the waits, the rides with their crowding, the walks between stations and
 * the fare. A change costs its wait, its walk and a fixed few minutes (T-717); the base fare is in
 * the fare. */
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
  let waitP = 0, rideP = 0, xferP = 0, changeP = 0;
  for (const l of r.legs) {
    if (l.kind === "ride") { waitP += l.waitPerceived; rideP += l.perceived; changeP += l.changePerceived; }
    else xferP += l.perceived;
  }
  out.push({ label: "wait", real: tot.wait, weight: tot.wait > 0 ? waitP / tot.wait : 1, perceived: waitP, unit: "min" });
  out.push({ label: "ride", real: tot.ride, weight: tot.ride > 0 ? rideP / tot.ride : 1, perceived: rideP, unit: "min" });
  if (tot.xfer > 0) out.push({ label: "walk between stations", real: tot.xfer, weight: xferP / tot.xfer, perceived: xferP, unit: "min" });
  if (changeP > 0) out.push(tot.changes > 1
    ? { label: "changes", real: tot.changes, weight: changeP / tot.changes, perceived: changeP, unit: "count" }
    : { label: "change", real: 0, weight: 1, perceived: changeP, unit: "min" });
  out.push({ label: "fare", real: tot.fareUsd, weight: t.minPerUsd, perceived: tot.fareUsd * t.minPerUsd, unit: "usd" });
  return out;
}

let query: ((from: [number, number], to: [number, number], purpose: number) => Promise<Trip | null>) | null = null;
/** Set by workers/demandClient.ts when the pool starts. */
export function setTripQuery(q: typeof query) {
  query = q;
}
/** Ask the demand workers about a trip as a commute (`purpose` -1) or as an everyday purpose
 * (an `EVERYDAY_KEYS` index), for the solve `demandView` shows. */
export const askTrip = (from: [number, number], to: [number, number], purpose = -1): Promise<Trip | null> => query?.(from, to, purpose) ?? Promise.resolve(null);

/** The two places the explorer counts trips between (T-586): the cells at the end the view
 * counts by (a bubble selection's end, else the trip's start as homes), and the cells at the
 * other end (pack cell ids). */
export interface TripPlaces {
  end: "home" | "work";
  near: Uint32Array;
  far: Uint32Array;
}

/** Trips a day (legs both ways) between two places by mode, under the purpose filter. */
export interface TripMix {
  rail: number;
  walk: number;
  drive: number;
  /** each shown purpose's alone, biggest first; under one purpose, that one */
  purposes: { key: PurposeKey; rail: number; walk: number; drive: number }[];
}

/** The far end of `f` (a selection's trips, every purpose and each on its own) summed over the far
 * cells `far`, as the far-end bubbles and their hovers sum it. */
export function tripMix(f: Flows, far: Uint32Array, filter: PurposeFilter): TripMix {
  const inFar = new Set(far);
  const sum = (fe: Pick<Flows, "cells" | "farRail" | "farWalk" | "farDrive">) => {
    let rail = 0, walk = 0, drive = 0;
    for (let i = 0; i < fe.cells.length; i++) if (inFar.has(fe.cells[i])) (rail += fe.farRail[i]), (walk += fe.farWalk[i]), (drive += fe.farDrive[i]);
    return { rail, walk, drive };
  };
  const purposes: TripMix["purposes"] = [];
  for (const { key } of FILTER_PURPOSES) {
    if (filter !== "all" && key !== filter) continue;
    const fe = f.byPurpose?.get(key);
    if (fe) purposes.push({ key, ...sum(fe) });
  }
  purposes.sort((a, b) => b.rail + b.walk + b.drive - (a.rail + a.walk + a.drive));
  const all = filter === "all" ? sum(f) : purposes[0] ?? { rail: 0, walk: 0, drive: 0 };
  return { ...all, purposes };
}

/** The everyday purpose kinds a shown purpose counts (school counts university), in kind order. */
export const kindsOf = (key: PurposeKey): number[] => EVERYDAY_KEYS.flatMap((k, i) => (shownKey(k) === key ? [i] : []));

/** The period with most of a purpose's trips in a selection's far end (`Flows.purposes`: work
 * first, then the everyday kinds in order), or null if it has none. The periods' shares are the
 * purpose's own, the same for every pair of places. */
export function busiestPeriod(f: Flows, key: PurposeKey): number | null {
  const rows = key === "work" ? [f.purposes[0]] : kindsOf(key).map((k) => f.purposes[1 + k]);
  let best = -1, most = 0;
  for (let q = 0; q < PERIODS.length; q++) {
    let n = 0;
    for (const r of rows) if (r) for (let m = 0; m < 3; m++) n += (r.outward[m][q] ?? 0) + (r.returning[m][q] ?? 0);
    if (n > most) (most = n), (best = q);
  }
  return best >= 0 ? best : null;
}

/** What the explorer shows: the trip's routes, the trips between the two places under the
 * view's filter and day (null until counted, or when nothing counts them), and the purpose and
 * period the routes are read for. */
export interface TripShown {
  key: string;
  trip: Trip | null;
  asking: boolean;
  mix: TripMix | null;
  /** the purpose the routes are read as */
  purpose: PurposeKey;
  /** the period with most of its trips (null: none counted) */
  busiest: number | null;
}

/** The trip shown (asked for by map/tripView.ts when the selection, the purpose filter or the
 * demand result changes). */
export const tripView = signal<TripShown | null>(null);
/** A trip opened from a bubble selection by clicking a far-end bubble (T-208): the trip selection,
 * the one it came from (for the inspector's way back) and the two places it counts between. */
export const tripOpenedFrom = signal<{ trip: Selection; from: Selection; places: TripPlaces } | null>(null);
/** The period the explorer shows: null follows the busiest period of the purpose shown. */
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
