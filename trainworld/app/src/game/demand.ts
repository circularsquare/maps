// What the city's demand came to, as the UI reads it (T-026; SPEC 4, 8). Written only by
// workers/demandClient.ts when a solve lands; components read the signals and the lookups below
// and never write them. The last finished result stays until a newer one replaces it (SPEC 1).
//
// Ids are the UI's: a line or station id as a string (the clock worker's numeric ids go through
// String()).

import { computed, signal } from "@preact/signals";
import { activeCity, REGISTRY, viewedZone, type CityId } from "./cities";
import { DEMAND_DAYS, type DemandDay } from "./days";
import { shownDay } from "./ui";
import type { PurposeFilter, PurposeKey } from "./purposes";
import type { ModeSplit } from "./types";

export interface LineDemand {
  /** boardings onto the line a day, both directions, transfers included */
  ridersPerDay: number;
  /** fullest train at high demand: the worst segment's busiest-hour load in a high-demand
   * period, % of the crush capacity of the trains running then. A forecast average, not shown to
   * the player since T-230: the line panel shows the boarding account's trains instead */
  fullestPct: number;
  /** boardings a day at each stop, both directions, in the line's stop order */
  boardings: number[];
  /** riders per segment and period, `loads[period][segment]`: run 0's segments (stop k to
   * k + 1) and then run 1's in its own travel order; periods as game/periods.ts */
  loads: Float32Array[];
  /** boardings and alightings per period at each route node, `board[period][node]`: run 0's
   * stops, then run 1's, each in its travel order (2 x stops); the boarding account's input (T-164) */
  board?: Float32Array[];
  alight?: Float32Array[];
  /** T-324: from the free-flow estimate of a solve still running. A line new or changed since the
   * shown final result gets these first figures early, and the final round replaces them. */
  estimate?: boolean;
}

export interface StationDemand {
  boardings: number;
  alightings: number;
}

export interface DemandView {
  /** the demand client's solve number this came from */
  version: number;
  /** the day it is for (T-260, T-464): a weekend and the average are worked out from the weekday's solve */
  day: DemandDay;
  /** crowding rounds every period has had: 0 = the free-flow first estimate */
  round: number;
  /** the solve's last round: its figures are final. Earlier rounds are shown only as estimates,
   * where nothing final is shown yet (T-232); a re-solve after an edit keeps the last final
   * result until its own final round */
  final: boolean;
  /** Mode split, % of legs: network-wide here, all modelled local purposes in cityDemandViews. */
  split: ModeSplit;
  trainShareByPeriod: [string, number][];
  /** per period (game/periods.ts order), the busiest hour's riders against the period's mean
   * hour (1.4 in the peaks): segment loads / period hours x this = riders in the busiest hour */
  peakHourFactor: number[];
  tripsPerDay: number;
  railTripsPerDay: number;
  lines: Map<string, LineDemand>;
  stations: Map<string, StationDemand>;
  /** The same lines and stations by trip purpose, for the map's purpose filter (T-158); the
   * purposes add up to `lines` and `stations`. Absent purposes carried no riders. Keyed by the
   * shown purposes (game/purposes.ts `PURPOSES`): "school" counts university too (T-192). */
  purposes?: Map<PurposeKey, PurposeDemand>;
}

/** One purpose's share of `DemandView.lines` and `.stations` (T-158). */
export interface PurposeDemand {
  /** per line id: boardings a day and riders per segment and period, as `LineDemand` */
  lines: Map<string, { ridersPerDay: number; boardings: number[]; loads: Float32Array[] }>;
  stations: Map<string, StationDemand>;
}

/** A line's riders per segment and period as the map draws them: one purpose's, or all. */
export function lineLoadsFor(dv: DemandView | null, lineId: string, purpose: PurposeFilter = "all"): Float32Array[] | undefined {
  if (!dv) return undefined;
  if (purpose === "all") return dv.lines.get(lineId)?.loads;
  // a purpose with no riders anywhere has no entry: zeros, not "unknown"
  const own = dv.purposes?.get(purpose)?.lines.get(lineId)?.loads;
  return own ?? dv.lines.get(lineId)?.loads.map((p) => new Float32Array(p.length));
}

/** A station's boardings and alightings as the map draws them: one purpose's, or all. */
export function stationRidersFor(dv: DemandView | null, id: string, purpose: PurposeFilter = "all"): StationDemand | undefined {
  if (!dv) return undefined;
  if (purpose === "all") return dv.stations.get(id);
  return dv.purposes?.get(purpose)?.stations.get(id) ?? (dv.stations.has(id) ? { boardings: 0, alightings: 0 } : undefined);
}

/** The regional layer (T-117, T-276): trips a day include the commutes between zones; each
 * corridor's trips and rail are both parts together, `commutes` and `commuteRail` the commute legs;
 * T-378: `air` and `car` the rest by plane and by car, `elsewhere` each place's longer trips with
 * places that have no zone, each way. */
export interface IntercityDemand { version: number; round: number; tripsPerDay: number; railTripsPerDay: number; commuteTripsPerDay?: number; commuteRailPerDay?: number; corridors: {from: number; to: number; trips: number; rail: number; commutes?: number; commuteRail?: number; air?: number; car?: number}[]; elsewhere?: number[] }

/** One kind of day's results (T-260, game/days.ts): the network's view, each city's, the market's.
 * Written together by workers/demandClient.ts. */
export interface DayDemand {
  view: DemandView | null;
  cities: Map<CityId, DemandView>;
  intercity: IntercityDemand | null;
}
const noDay = (): DayDemand => ({ view: null, cities: new Map(), intercity: null });
/** The newest finished solve for each day, `DemandDay` order: weekday, weekend, average. */
export const demandByDay = signal<DayDemand[]>(DEMAND_DAYS.map(noDay));

/** The newest finished solve on the day shown (`shownDay`); null until the first lands. */
export const demandView = computed<DemandView | null>(() => demandByDay.value[shownDay.value]?.view ?? null);
/** Local purpose shares (everyday trips included), separate from network-wide totals. */
export const cityDemandViews = computed<Map<CityId, DemandView>>(() => demandByDay.value[shownDay.value]?.cities ?? new Map());
/** Distinct, synthetic intercity trips: never included in local commute denominators. */
export const intercityDemand = computed<IntercityDemand | null>(() => demandByDay.value[shownDay.value]?.intercity ?? null);
/** A solve is running for a newer network than the one `demandView` shows (or, T-328, the shown
 * figures are the cache's and the same network is being solved again). */
export const demandUpdating = signal(false);
/** How far that solve has got: results published so far (the free-flow estimate, then each
 * crowding round) of all it will publish. */
export const demandStep = signal({ done: 0, of: 4 });
/** T-392: the zones the running solve has still to finish (each cluster's land as it is solved);
 * null until the workers say, then read as every zone */
export const demandSolving = signal<Set<CityId> | null>(null);
/** A solve is running that will still change the figures of the zone in view. */
export const viewUpdating = computed(() => demandUpdating.value && (demandSolving.value?.has(activeCity.value) ?? true));
/** T-421: the figures shown for the zone in view are final: the shown result is, or (T-402) the
 * zone's cluster has its final while other clusters still solve and the view is not yet the solve's */
export const viewFinal = computed(() => {
  const v = demandView.value;
  return !!v && (v.final || (!viewUpdating.value && cityDemandViews.value.has(activeCity.value)));
});

/** The mode-split bar's numbers: the viewed zone's own figures (a solve's, or the bare pass's for a
 * zone no station reaches), null until it has them and over dead space (T-318). */
export const modeSplit = computed<ModeSplit | null>(() => !viewedZone.value ? null : cityDemandViews.value.get(activeCity.value)?.split ?? null);

/** T-342: the same split in rides (trip legs) a day, `all` the total of the three. It is what
 * `modeSplit` is a percentage of (the zone's modelled local legs, `DemandView.tripsPerDay`); null
 * until the zone has a solved or bare view, and over dead space. */
export const modeRides = computed<(ModeSplit & { all: number }) | null>(() => {
  const v = !viewedZone.value ? null : cityDemandViews.value.get(activeCity.value);
  if (!v) return null;
  const all = v.tripsPerDay;
  return { all, train: v.railTripsPerDay, walk: all * v.split.walk / 100, drive: all * v.split.drive / 100 };
});

export const lineDemand = (id: string): LineDemand | null => demandView.value?.lines.get(id) ?? null;
export const stationDemand = (id: string): StationDemand | null => demandView.value?.stations.get(id) ?? null;

// --- Data for the demand views (T-078 for T-084; notes/T-078.md) --------------------------------
//
// Asked for on demand, not pushed with every solve: the workers compute them from the solve that
// `demandView` shows and transfer typed arrays (no per-cell objects). Units are commuters a day:
// one person making a trip to work and one home. Every per-cell array is in the city pack's cell
// order (`CityCells`). A query resolves to null while there is no solve yet, or when the workers
// have moved on to a newer network than the one shown; ask again when `demandView` changes.

/** The city's cells and zones. Static for a city: ask once. */
export interface CityCells {
  /** number of cells (New York 202,337) */
  n: number;
  /** x, y per cell in the fixed NYC world frame (game/coords.ts turns them into lng/lat) */
  xy: Float32Array;
  /** H3 index (resolution 9) of each cell */
  h3: BigUint64Array;
  /** gravity zone of each cell (2 km squares, the unit of where people work) */
  zone: Uint32Array;
  /** zone side in metres, and x, y of each zone's centre (2 per zone) */
  zoneM: number;
  zoneXY: Float32Array;
}

/** Local trip legs a day per cell by mode, home and activity ends, both directions. */
export interface CellModes {
  /** the `DemandView.version` these belong to */
  version: number;
  /** and its day (T-260, T-464) */
  day: DemandDay;
  /** living in the cell: by rail, walking the whole way, driving (car, bus, taxi) */
  homeRail: Float32Array;
  homeWalk: Float32Array;
  homeDrive: Float32Array;
  /** activity destinations (work plus implemented everyday purposes), the same three */
  workRail: Float32Array;
  workWalk: Float32Array;
  workDrive: Float32Array;
}

/** Cells and the commuters from each, most first. */
export interface CellRiders {
  cells: Uint32Array;
  riders: Float32Array;
  total: number;
}

/** Zones and commuters, most first (at most `top` of them). */
export interface ZoneRiders {
  zones: Uint32Array;
  riders: Float32Array;
}

/**
 * One station's riders (`stationRiders(id)`). Stations whose platforms are within 100 m of each
 * other are one station for demand (SPEC 4.4), so a pair of them answers the same. Asked for a
 * station complex (T-146, `members`), the answer is the sum over its distinct demand stations.
 */
export interface StationRiders {
  version: number;
  stationId: string;
  /** the stations summed: `members.join(",")`, just the station's id outside complex view */
  scope: string;
  /** commuters living in each cell who use this station at the home end (its catchment) */
  home: CellRiders;
  /** commuters working in each cell who use this station at the work end */
  work: CellRiders;
  /** the home riders (commutes) by how they reach the station: on foot, or by a ride (car, bus,
   * bike or lift, T-371); commuters a day, adding up to `home.total` */
  walked: number;
  rode: number;
  /** where the home riders work, by zone */
  homeRidersWorkIn: ZoneRiders;
  /** where the work riders live, by zone */
  workRidersLiveIn: ZoneRiders;
}

/**
 * Where a set of cells' commuters go (`flows(end, cells)`, T-098 for the bubbles, notes/T-098.md):
 * everyone living (end "home") or working (end "work") in the selected cells, every mode, spread
 * over the cells at the other end. Units as `CellModes`: commuters a day.
 */
export interface Flows {
  version: number;
  end: "home" | "work";
  /** the selected commuters by mode; the same as `cellModes` summed over the selected cells */
  rail: number;
  walk: number;
  drive: number;
  /** cells at the other end with commuters from (or to) the selection, pack order */
  cells: Uint32Array;
  /** per far-end cell, commuters a day */
  farRail: Float32Array;
  farWalk: Float32Array;
  farDrive: Float32Array;
  /** commuters in the far cells left out (the smallest, together 1% of the selection), by mode:
   * the far cells plus these add up to rail, walk, drive */
  cutRail: number;
  cutWalk: number;
  cutDrive: number;
  /** Same local trip legs as the bubbles, separated by activity and departure direction. */
  purposes: PurposeTrips[];
  /** The far end and totals of each local purpose on its own, for the map's purpose filter
   * (T-158); the fields mean what the ones above mean. Absent: the selection has none of it. */
  byPurpose?: Map<PurposeKey, FarEnd>;
}

export type FarEnd = Pick<Flows, "rail" | "walk" | "drive" | "cells" | "farRail" | "farWalk" | "farDrive" | "cutRail" | "cutWalk" | "cutDrive">;

export interface PurposeTrips {
  name: string;
  /** Five departure periods per mode: rail, walk, drive. */
  outward: number[][];
  returning: number[][];
}

/** Registered places in pipeline/intercity.py order: the indices in the arrays below. */
export const INTERCITY_PLACES = REGISTRY.market.places.map(p => p.name);
/** The places as the player reads them. `own` is the place the selection itself lies in: the trips
 * that stay within it are all 40 km or longer (the market has none shorter). */
export function intercityPlaceName(place: number, own = -1): string {
  const name = REGISTRY.market.places[place]?.regionName ?? "";
  return place === own ? `Other ${name} places` : name;
}

/** Longer trips (non-commute, T-117) to or from one place, a day. */
export interface PlaceTrips {
  /** index into INTERCITY_PLACES */
  place: number;
  /** all modes, and by rail; T-378: by plane and by car (a station's: 0) */
  all: number;
  rail: number;
  air: number;
  car: number;
  /** where the trips' far ends are, mean position in the world frame (NaN with no trips) */
  x: number;
  y: number;
}

/** Longer trips to or from one far 6 km zone, a day (T-120): drawn as one bubble. */
export interface ZoneTrips {
  /** Exact market membership from the demand pack, not inferred from a zone's centroid. */
  place: number;
  /** the zone's own position, pack metres (as `PlaceTrips.x`) */
  x: number;
  y: number;
  /** all modes, and by rail (a station's: both the riders) */
  all: number;
  rail: number;
}

/** One direction of a selection's or a station's longer trips, by place at the far end. */
export interface PlaceFlows {
  all: number;
  rail: number;
  air: number;
  car: number;
  /** T-378: longer trips with places that have no zone (not in `all`) */
  elsewhere: number;
  /** the places with trips, biggest (all modes) first */
  places: PlaceTrips[];
  /** the far zones with trips, biggest first; their sums equal the places' */
  zones: ZoneTrips[];
}

/**
 * A bubble selection's longer trips (`intercitySelection(end, cells)`, T-118): the market's own
 * trips from the 6 km zones the selected cells lie in, each zone's share given to the 600 m squares
 * by people (starts) or attraction (ends) and the selected part of every square counted. Units:
 * trips a day, and `rail` the part by train. Not commuters.
 */
export interface IntercitySelection {
  version: number;
  /** trips starting in the selected cells, by where they end */
  starts: PlaceFlows;
  /** trips ending in the selected cells, by where they start */
  ends: PlaceFlows;
  /** share of the selected squares that the market has sites for (1 = all) */
  matched: number;
}

/** A station's longer-trip rail riders (`intercityStation(id)`): starting here, ending here, by place. */
export interface IntercityStation {
  version: number;
  stationId: string;
  /** as `StationRiders.scope` */
  scope: string;
  starts: PlaceFlows;
  ends: PlaceFlows;
}

/** Riders changing from one line at one station to another line at another (or the same) station
 * a day, every period and purpose (T-176). Both ends name a snapshot station and line; a walking
 * transfer has two different stations. */
export interface TransferPair {
  fromStation: string;
  fromLine: string;
  toStation: string;
  toLine: string;
  riders: number;
}

/** The transfers at a station complex (`complexTransfers(members)`): every pair with either end at
 * a member, both directions of a line merged. */
export interface ComplexTransfers {
  version: number;
  round: number;
  /** as `StationRiders.scope`: the members asked about, joined by commas */
  scope: string;
  pairs: TransferPair[];
}

export interface DemandQueries {
  transfers(members: string[]): Promise<ComplexTransfers | null>;
  platformTransfers(): Promise<Float32Array | null>;
  cityCells(city?: CityId): Promise<CityCells | null>;
  cellModes(purpose?: PurposeFilter, city?: CityId): Promise<CellModes | null>;
  stationRiders(stationId: string, top?: number, members?: string[]): Promise<StationRiders | null>;
  flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null>;
  intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null>;
  intercityStation(stationId: string, members?: string[]): Promise<IntercityStation | null>;
}

let queries: DemandQueries | null = null;
/** Set by workers/demandClient.ts when the pool starts. */
export function setDemandQueries(q: DemandQueries) {
  queries = q;
}
/** The city's cells (null until the pool has opened the city). `city`: another zone's than the
 * focused one (a nearby zone's bubbles, T-276); it waits for that zone to open. */
export const cityCells = (city?: CityId): Promise<CityCells | null> => queries?.cityCells(city) ?? Promise.resolve(null);
/** Local trip legs a day per cell by mode, for the solve `demandView` shows. One 4.9 MB answer in
 * New York. `purpose`: one purpose's legs only (T-158); longer trips have no local legs (zeros).
 * `city`: a nearby zone's (T-276), not cached; null while that zone is not in the solve shown. */
export const cellModes = (purpose: PurposeFilter = "all", city?: CityId): Promise<CellModes | null> => queries?.cellModes(purpose, city) ?? Promise.resolve(null);
/** A station's catchment and where its riders go, for the solve `demandView` shows; `top` zones
 * per direction (default 20). A few kB; ask on click. `members`: a station complex's stations,
 * summed (T-146). */
export const stationRiders = (stationId: string, top?: number, members?: string[]): Promise<StationRiders | null> =>
  queries?.stationRiders(stationId, top, members) ?? Promise.resolve(null);
/** Where the commuters living (or working) in `cells` (pack cell ids) work (or live), by mode, for
 * the solve `demandView` shows. Asked on click; null while there is no solve or the workers have
 * moved on to a newer network. */
export const flows = (end: "home" | "work", cells: Uint32Array): Promise<Flows | null> =>
  queries?.flows(end, cells) ?? Promise.resolve(null);
/** The longer (non-commute) trips of the people living or working in `cells`, by place, for the
 * solve `demandView` shows. Null while there is no solve or the workers moved on. */
export const intercitySelection = (end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> =>
  queries?.intercitySelection(end, cells) ?? Promise.resolve(null);
/** A station's longer-trip riders by place, or a complex's (`members`). Null as above. */
export const intercityStation = (stationId: string, members?: string[]): Promise<IntercityStation | null> =>
  queries?.intercityStation(stationId, members) ?? Promise.resolve(null);
/** Transfers a day at a station complex's members (T-176), for the solve `demandView` shows. A few
 * kB; asked when the complex map shows. Null while there is no solve or the workers moved on. */
export const complexTransfers = (members: string[]): Promise<ComplexTransfers | null> =>
  queries?.transfers(members) ?? Promise.resolve(null);
/** Every change between platforms a day, for the solve `demandView` shows (T-554): per pair the
 * line, run and stop (in the run's order) riders leave, the same where they board, and riders, as
 * `TrackApi.set_transfers` reads them. Null while there is no solve or the workers moved on. */
export const platformTransfers = (): Promise<Float32Array | null> => queries?.platformTransfers() ?? Promise.resolve(null);
