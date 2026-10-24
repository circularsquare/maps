// T-1148: the regional market's rule for a pair of places, read from the market rows
// (market-rows.json, docs/regional.md 5.1) on the main thread: the car's and the plane's costs part
// by part, the corridor's split between them, and the places' trips. A mirror of
// sim/src/demand/intercity.rs (`car_min`, `crossing_min`, `air_min`, the per-place-pair rule in
// `build_for`, `place_eachway`) and days.rs (`share_out` for longer trips), so the trip explorer
// answers between places with no station at once and names the parts of the sim's minutes.
// checks/t1148-market-rule.mjs holds it to the sim's figures on a few thousand pairs.

import { signal } from "@preact/signals";
import { CITIES, REGISTRY, widenBox, zoneBox, ZONE_REACH_M } from "./cities";
import { PERIODS } from "./periods";
import type { MarketEnd, MarketPeriod } from "./trip";

/** A place's market row, the fields read here. */
export interface MarketRow {
  place: number;
  lon: number;
  lat: number;
  country: string;
  inc: number;
  road: number;
  g: number;
  t: number;
  a: number;
  ta: number;
  /** trips a day each way with each place under `near_km` */
  near: [number, number][];
  /** each airport: the drive to it (minutes, people weighted) and the places it flies to direct */
  air: [number, number[]][];
  /** perceived minutes to park, mean over the place's zones */
  park: number;
}

interface Rule {
  R: number; A: number; onset_km: [number, number]; d0_km: number; G: number; D1_km: number; eps_t: number;
  tail_onset_km: [number, number]; border_eps: number; border: number; Rt: number; near_km: number;
}
const RULE = REGISTRY.market.rule as Rule;

const EARTH_R = 6371008.8;
export const ROAD_DETOUR = 1.25;
export const CAR_KMH = 85;
/** the drive's fixed minutes, on top of the road at `CAR_KMH` */
export const DRIVE_FIXED_MIN = 15;
export const CAR_USD_PER_KM = 0.15;
/** people sharing the car on a longer trip */
export const CAR_SHARED_BY = 2;
export const MIN_PER_USD = 3;
export const AIR_TERMINAL_MIN = 90;
export const AIR_KMH = 750;
export const AIR_FIXED_MIN = 30;
export const AIR_NO_DIRECT_MIN = 60;
export const AIR_FARE_USD = 100;
export const AIR_USD_PER_KM = 0.05;
const BETA = 0.1, DAMP_MIN = 60;
const OPEN_AREA = new Set(["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS", "CH", "LI", "GB"]);
const OPEN_AREA_BORDER = 0.5;
const HOKKAIDO_LAT = 41.4;

/** A car carried over water: the fare per car in dollars and the minutes beyond the straight drive. */
export interface Crossing { kind: "channel" | "tsugaru"; usd: number; min: number }
const CHANNEL: Crossing = { kind: "channel", usd: 150, min: 60 };
const TSUGARU: Crossing = { kind: "tsugaru", usd: 170, min: 240 };

/** Longer trips' period shares on a weekday, and a Saturday's and a Sunday's trips per weekday's
 * with their own shares (days.rs `DAY_RATE`, `DAY_OUT`). */
const SHARES = [0.2, 0.35, 0.25, 0.15, 0.05];
const WEEKEND: [number, number[]][] = [[1.69, [0.166, 0.314, 0.196, 0.295, 0.029]], [1.48, [0.133, 0.394, 0.285, 0.166, 0.022]]];

/** Longer trips departing in period `q` on demand day `day` (0 weekday, 1 Saturday, 2 Sunday, 3
 * the average), per weekday trip. */
export function longerShare(day: number, q: number): number {
  const on = (d: number) => WEEKEND[d][0] * WEEKEND[d][1][q];
  return day === 0 ? SHARES[q] : day === 3 ? (5 * SHARES[q] + on(0) + on(1)) / 7 : on(Math.min(day, 2) - 1);
}

// ---------------------------------------------------------------- the rows

let rows: Map<number, MarketRow> | null = null;
let loading: Promise<boolean> | null = null;
/** True once the rows are in. */
export const marketRowsReady = signal(false);

/** Use these rows ({zone id: row}, as market-rows.json). */
export function setMarketRows(byZone: Record<string, MarketRow>) {
  rows = new Map();
  // in registry order, as the sim reads them
  for (const c of CITIES) {
    const r = byZone[c.id];
    // a row with no airports or near places leaves them out
    if (r) rows.set(r.place, { ...r, air: r.air ?? [], near: r.near ?? [] });
  }
  marketRowsReady.value = true;
}
/** Fetch the rows once (the same file the demand pool sends its workers). */
export function loadMarketRows(): Promise<boolean> {
  return (loading ??= fetch(new URL("./market-rows.json", import.meta.url).href)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`market rows: ${r.status}`))))
    .then((j) => (setMarketRows(j), true), (err) => { console.error(err); loading = null; return false; }));
}
export const marketRow = (place: number): MarketRow | undefined => rows?.get(place);

// ---------------------------------------------------------------- the rule

const unit = (r: MarketRow) => {
  const lon = (r.lon * Math.PI) / 180, lat = (r.lat * Math.PI) / 180;
  return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
};
/** The great circle between two places' centres, km. */
export function placeKm(a: MarketRow, b: MarketRow): number {
  const u = unit(a), v = unit(b);
  const c = Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
  return ((2 * EARTH_R) / 1000) * Math.asin(Math.min(c / 2, 1));
}

/** The crossings a car between the two places is carried over. */
export function crossingsOf(a: MarketRow, b: MarketRow): Crossing[] {
  const hokkaido = (r: MarketRow) => r.country === "JP" && r.lat > HOKKAIDO_LAT;
  const out: Crossing[] = [];
  if ((a.country === "GB") !== (b.country === "GB")) out.push(CHANNEL);
  if (hokkaido(a) !== hokkaido(b)) out.push(TSUGARU);
  return out;
}

/** The car of a longer trip over `km` (centre to centre) to a place where parking counts `park`
 * perceived minutes: each part's real amount and its perceived minutes. Money is each person's, the
 * car shared by `CAR_SHARED_BY`. */
export interface CarCost {
  km: number;
  roadKm: number;
  /** minutes on the road, the fixed minutes included */
  driveMin: number;
  /** running the car, each person's dollars */
  runUsd: number;
  /** parking where the trip ends, each person's perceived minutes */
  parkMin: number;
  crossings: Crossing[];
  /** perceived minutes in all */
  total: number;
}
export function carCost(km: number, park: number, crossings: Crossing[]): CarCost {
  const roadKm = km * ROAD_DETOUR;
  const driveMin = DRIVE_FIXED_MIN + (roadKm / CAR_KMH) * 60;
  const runUsd = (roadKm * CAR_USD_PER_KM) / CAR_SHARED_BY;
  const parkMin = park / CAR_SHARED_BY;
  const total = driveMin + runUsd * MIN_PER_USD + parkMin + crossings.reduce((s, c) => s + crossingMin(c), 0);
  return { km, roadKm, driveMin, runUsd, parkMin, crossings, total };
}
/** The parts of a car cost the sim gave as `total` perceived minutes over a mean distance of `km`
 * (its 6 km area pairs trip weighted): the drive, its running and the crossings by the rule, which
 * is linear in the distance, and parking (the far areas' mean) the rest. */
export function carPartsOf(km: number, total: number, crossings: Crossing[]): CarCost {
  const c = carCost(km, 0, crossings);
  return { ...c, parkMin: Math.max(0, total - c.total), total };
}
/** A crossing's perceived minutes: its minutes and each person's share of the fare. */
export const crossingMin = (c: Crossing) => c.min + (c.usd / CAR_SHARED_BY) * MIN_PER_USD;

/** A flight between two places by their best pair of airports. */
export interface FlightCost {
  /** the drives to the airport at the start and from the one at the end */
  accessFrom: number;
  accessTo: number;
  direct: boolean;
  /** minutes in the air, the flight's fixed minutes and any change included */
  flightMin: number;
  fareUsd: number;
  total: number;
}
function flightOf(a: MarketRow, b: MarketRow, km: number): FlightCost | null {
  let best: FlightCost | null = null;
  const ok = (x: [number, number[]]) => Number.isFinite(x[0]) && x[0] >= 0;
  for (const x of a.air) {
    if (!ok(x)) continue;
    for (const y of b.air) {
      if (!ok(y)) continue;
      const direct = x[1].includes(b.place) || y[1].includes(a.place);
      const flightMin = (km / AIR_KMH) * 60 + AIR_FIXED_MIN + (direct ? 0 : AIR_NO_DIRECT_MIN);
      const fareUsd = AIR_FARE_USD + AIR_USD_PER_KM * km;
      const total = x[0] + y[0] + AIR_TERMINAL_MIN + flightMin + fareUsd * MIN_PER_USD;
      if (!best || total < best.total) best = { accessFrom: x[0], accessTo: y[0], direct, flightMin, fareUsd, total };
    }
  }
  return best;
}

const modeBeta = (car: number) => BETA * Math.sqrt(Math.min(DAMP_MIN / car, 1));

/** A pair of places as the market costs them, the same whoever builds what. */
export interface Corridor {
  km: number;
  /** under `near_km`: zone to zone pairs, and no plane */
  near: boolean;
  /** one landmass: a car can go */
  road: boolean;
  /** the car with the two places' mean parking: the cost the split is decided on */
  car: CarCost | null;
  flight: FlightCost | null;
  /** the trips not by rail: the plane's share and the car's */
  air: number;
  carShare: number;
  /** the reference mode's perceived minutes: the car's, the plane's where no car goes, or with
   * neither the car's as though a road crossed the water (the sim's `refm`) */
  refMin: number;
}
/** The corridor from place `a` to place `b`, null where either has no row. */
export function corridorOf(a: number, b: number): Corridor | null {
  const p = marketRow(a), q = marketRow(b);
  if (!p || !q || a === b) return null;
  const km = placeKm(p, q);
  const near = km < RULE.near_km;
  const road = p.road === q.road;
  const flight = near ? null : flightOf(p, q, km);
  const car = road ? carCost(km, 0.5 * (parkOf(p) + parkOf(q)), crossingsOf(p, q)) : null;
  let air = 0;
  if (car && flight) {
    const w = Math.exp(Math.max(-80, Math.min(80, -modeBeta(car.total) * (flight.total - car.total))));
    air = w / (1 + w);
  } else if (flight) air = 1;
  const refMin = car ? car.total : flight ? flight.total : carCost(km, 0.5 * (parkOf(p) + parkOf(q)), []).total;
  return { km, near, road, car, flight, air, carShare: car ? 1 - air : 0, refMin };
}
const parkOf = (r: MarketRow) => (Number.isFinite(r.park) ? Math.max(0, r.park) : 0);

/** The car from place `from` to place `to` centre to centre, paying `to`'s parking (the sim's
 * figure between two places' rows, `head_from_rows`); null with no road between them. */
export function carBetween(from: number, to: number): CarCost | null {
  const p = marketRow(from), q = marketRow(to);
  if (!p || !q || from === to || p.road !== q.road) return null;
  return carCost(placeKm(p, q), parkOf(q), crossingsOf(p, q));
}

const ground = (d: number) => {
  const on = Math.min(1, Math.max(0, (d - RULE.onset_km[0]) / (RULE.onset_km[1] - RULE.onset_km[0])));
  return on === 0 ? 0 : on / Math.max(d, RULE.d0_km) ** RULE.A;
};
const tail = (d: number) => {
  const on = Math.min(1, Math.max(0, (d - RULE.tail_onset_km[0]) / (RULE.tail_onset_km[1] - RULE.tail_onset_km[0])));
  return on === 0 || RULE.Rt === 0 ? 0 : on / Math.max(d, RULE.D1_km) ** RULE.G;
};
const income = (r: MarketRow) => Math.min(r.inc, 1);
function border(p: MarketRow, q: MarketRow): number {
  const general = RULE.border * income(p) ** RULE.border_eps;
  if (p.country === q.country) return 1;
  return OPEN_AREA.has(p.country) && OPEN_AREA.has(q.country) ? Math.max(OPEN_AREA_BORDER, general) : general;
}
/** Trips a day from place p to q's attraction at `d` km (`Market::directed`). */
function directed(p: MarketRow, q: MarketRow, d: number): number {
  const rel = income(p) > 0 ? Math.min(income(q) / income(p), 1) ** RULE.eps_t : 1;
  const gr = p.g > 0 ? p.g * ground(d) * q.a : 0;
  const tl = p.t > 0 ? p.t * tail(d) * q.ta * rel : 0;
  return border(p, q) * (gr + tl);
}
/** Two places' longer trips a weekday each way: near ones from the rows, far ones by the rule at
 * their centres (`place_eachway`). */
export function placeTrips(a: number, b: number): number {
  const p = marketRow(a), q = marketRow(b);
  if (!p || !q || a === b) return 0;
  const km = placeKm(p, q);
  if (km < RULE.near_km) return (p.near.find((x) => x[0] === b) ?? q.near.find((x) => x[0] === a))?.[1] ?? 0;
  return 0.5 * (directed(p, q, km) + directed(q, p, km));
}

/** Whether any of `stations` ([lng, lat] each) is near place `place`: inside its zone's box widened
 * by `ZONE_REACH_M`, well past the market's reach to a station (a drive of 15 km from a 6 km area).
 * With none, no train serves the place. */
export function stationNear(place: number, stations: readonly { lng: number; lat: number }[]): boolean {
  const c = CITIES.find((x) => x.place === place);
  if (!c) return true;
  const [w, s, e, n] = widenBox(zoneBox(c), ZONE_REACH_M);
  return stations.some((p) => p.lng >= w && p.lng <= e && p.lat >= s && p.lat <= n);
}

/**
 * The trip explorer's answer between two whole places from the rows alone, each period on demand
 * day `day`: the longer trips from `from` to `to`, the plane's and the car's shares of them as the
 * corridor splits them, the car's minutes to `to` and the distance. No rail: for a pair with no
 * station at one end. Null where either place has no row.
 */
export function rowsAnswer(from: MarketEnd, to: MarketEnd, day: number): (MarketPeriod | null)[] | null {
  const c = corridorOf(from.place, to.place);
  if (!c) return null;
  const trips = placeTrips(from.place, to.place);
  const car = carBetween(from.place, to.place);
  return PERIODS.map((_, q) => {
    const n = trips * longerShare(day, q);
    return {
      round: 0, trips: n, rail: 0, air: n * c.air, car: n * c.carShare,
      railPerceived: NaN, carPerceived: car ? car.total : NaN, refPerceived: c.refMin,
      km: c.km, zonePairs: 0, served: 0, baseFarePerceived: 0, minPerUsd: MIN_PER_USD, why: "noRail", pairs: [], routes: [], cut: 0,
    };
  });
}
