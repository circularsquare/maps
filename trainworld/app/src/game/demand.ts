// What the city's demand came to, as the UI reads it (T-026; SPEC 4, 8). Written only by
// workers/demandClient.ts when a solve lands; components read the signals and the lookups below
// and never write them. The last finished result stays until a newer one replaces it (SPEC 1).
//
// Ids are the UI's: a line or station id as a string (the clock worker's numeric ids go through
// String()).

import { computed, signal } from "@preact/signals";
import { activeCity, REGISTRY, type CityId } from "./cities";
import type { ModeSplit } from "./types";

export interface LineDemand {
  /** boardings onto the line a day, both directions, transfers included */
  ridersPerDay: number;
  /** The separately modelled regional market's boardings onto this line. */
  intercityRidersPerDay?: number;
  /** fullest train at high demand: the worst segment's busiest-hour load in a high-demand
   * period, % of the crush capacity of the trains running then */
  fullestPct: number;
  /** boardings a day at each stop, both directions, in the line's stop order */
  boardings: number[];
  /** riders per segment and period, `loads[period][segment]`: run 0's segments (stop k to
   * k + 1) and then run 1's in its own travel order; periods as game/periods.ts */
  loads: Float32Array[];
}

export interface StationDemand {
  boardings: number;
  alightings: number;
}

export interface DemandView {
  /** the demand client's solve number this came from */
  version: number;
  /** crowding rounds every period has had: 0 = the free-flow first estimate */
  round: number;
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
}

/** The newest finished solve; null until the first lands. */
export const demandView = signal<DemandView | null>(null);
/** The split with no rail at all, known once the city pack is open (before any solve). */
export const baseSplit = signal<ModeSplit | null>(null);
/** Local purpose shares (NYC everyday trips included), separate from network-wide totals. */
export const cityDemandViews = signal<Map<CityId, DemandView>>(new Map());
export const cityBaseSplits = signal<Map<CityId, ModeSplit>>(new Map());
/** Distinct, synthetic intercity trips: never included in local commute denominators. */
export const intercityDemand = signal<{version: number; round: number; tripsPerDay: number; railTripsPerDay: number; corridors: {from: number; to: number; trips: number; rail: number}[] } | null>(null);
/** A solve is running for a newer network than the one `demandView` shows. */
export const demandUpdating = signal(false);
/** How far that solve has got: results published so far (the free-flow estimate, then each
 * crowding round) of all it will publish. */
export const demandStep = signal({ done: 0, of: 4 });

/** The mode-split bar's numbers (null while the pack is still loading). */
export const modeSplit = computed<ModeSplit | null>(() => cityDemandViews.value.get(activeCity.value)?.split ?? cityBaseSplits.value.get(activeCity.value) ?? baseSplit.value);

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
 * other are one station for demand (SPEC 4.4), so a pair of them answers the same.
 */
export interface StationRiders {
  version: number;
  stationId: string;
  /** commuters living in each cell who use this station at the home end (its catchment) */
  home: CellRiders;
  /** commuters working in each cell who use this station at the work end */
  work: CellRiders;
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
}

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
  /** all modes, and by rail */
  all: number;
  rail: number;
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
  starts: PlaceFlows;
  ends: PlaceFlows;
}

export interface DemandQueries {
  cityCells(): Promise<CityCells | null>;
  cellModes(): Promise<CellModes | null>;
  stationRiders(stationId: string, top?: number): Promise<StationRiders | null>;
  flows(end: "home" | "work", cells: Uint32Array): Promise<Flows | null>;
  intercitySelection(end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null>;
  intercityStation(stationId: string): Promise<IntercityStation | null>;
}

let queries: DemandQueries | null = null;
/** Set by workers/demandClient.ts when the pool starts. */
export function setDemandQueries(q: DemandQueries) {
  queries = q;
}
/** The city's cells (null until the pool has opened the city). */
export const cityCells = (): Promise<CityCells | null> => queries?.cityCells() ?? Promise.resolve(null);
/** Commuters a day per cell by mode, for the solve `demandView` shows. One 4.9 MB answer in New York. */
export const cellModes = (): Promise<CellModes | null> => queries?.cellModes() ?? Promise.resolve(null);
/** A station's catchment and where its riders go, for the solve `demandView` shows; `top` zones
 * per direction (default 20). A few kB; ask on click. */
export const stationRiders = (stationId: string, top?: number): Promise<StationRiders | null> =>
  queries?.stationRiders(stationId, top) ?? Promise.resolve(null);
/** Where the commuters living (or working) in `cells` (pack cell ids) work (or live), by mode, for
 * the solve `demandView` shows. Asked on click; null while there is no solve or the workers have
 * moved on to a newer network. */
export const flows = (end: "home" | "work", cells: Uint32Array): Promise<Flows | null> =>
  queries?.flows(end, cells) ?? Promise.resolve(null);
/** The longer (non-commute) trips of the people living or working in `cells`, by place, for the
 * solve `demandView` shows. Null while there is no solve or the workers moved on. */
export const intercitySelection = (end: "home" | "work", cells: Uint32Array): Promise<IntercitySelection | null> =>
  queries?.intercitySelection(end, cells) ?? Promise.resolve(null);
/** A station's longer-trip riders by place. Null as above. */
export const intercityStation = (stationId: string): Promise<IntercityStation | null> =>
  queries?.intercityStation(stationId) ?? Promise.resolve(null);
