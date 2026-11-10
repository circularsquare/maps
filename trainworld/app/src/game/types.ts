// Game data as the main thread sees it. Two kinds, kept apart on purpose:
//
// - `Save`: what the player made, shaped like the save file (SPEC 11). The clock worker owns the
//   real copy (SPEC 6.5); the main thread holds a read-only snapshot and changes it only by
//   sending edit operations (game/world.ts -> workers/clockClient.ts).
// - `*Stats`: results the workers compute from it (run times, trains needed, money). Never saved.
//
// `WorldView` bundles the newest snapshot of both, rebuilt by workers/clockClient.ts from each
// state message of the clock worker (T-025). Demand results (riders, the mode split) are in
// game/demand.ts.
//
// Ids: the track model's numeric ids. Lines and stations are also keyed by `String(id)` (the UI
// and the demand module use strings); `num` holds the number.

export type LineId = string;
export type StationId = string;
export type EdgeId = number;
export type NodeId = number;
export type Level = -3 | -2 | -1 | 0 | 1 | 2 | 3;
export const LEVELS: Level[] = [3, 2, 1, 0, -1, -2, -3];

/** Scheduling bands (SPEC 6.3): high = the two peaks, medium = midday and evening, low = night. */
export type Demand = "high" | "medium" | "low";
export const DEMANDS: Demand[] = ["high", "medium", "low"];
export const DEMAND_INDEX: Record<Demand, number> = { high: 0, medium: 1, low: 2 };
/** Trains an hour per demand level. */
export type Tph = Record<Demand, number>;

// ---------- the save: player inputs ----------

export interface StationInput {
  id: StationId;
  /** the track model's node id */
  num: number;
  name: string;
  lng: number;
  lat: number;
  /** world metres east and north of the frame's origin (game/frame.ts) */
  x: number;
  y: number;
  level: Level;
  /** platform length, m (SPEC 6.1: 60-400 in 20 m steps) */
  length: number;
  /** constructed; false = blueprint (SPEC 6.4) */
  built: boolean;
  /** a blueprint switched off (T-333): Construct skips it */
  off?: boolean;
  /** its upgrades, `UPGRADE` bits (game/upgrades.ts, T-1281) */
  upgrades?: number;
  /** the player typed its name (T-2683): a complex shows it above the others' */
  named?: true;
  /** track heading through it, rad, counter-clockwise from east */
  heading: number;
  /** its platform's centre line along its track, as drawn: x, y offsets from x, y in world units
   * (T-1859, workers/platformLines.ts); absent: straight along `heading` */
  line?: ArrayLike<number>;
  /** T-2194: a port, for ferries (no track or platform), and its berth, longitude and latitude */
  port?: [number, number];
  /** a tram stop: tram track's own 20-80 m platform (T-2926) */
  tram?: true;
}

export interface LineInput {
  id: LineId;
  num: number;
  /** one or two characters on the line's chip */
  letter: string;
  /** the symbol the player set, "" while none: the chip then shows the name's first letter (T-305) */
  symbol?: string;
  name: string;
  /** "#rrggbb", from the saturated line palette or player-set */
  colour: string;
  stops: StationId[];
  tph: Tph;
  /** the player's line group (T-1791), absent for none, and its name, "" for its oldest line's */
  group?: number;
  groupName?: string;
  /** T-1813: the group's bullet text and colour set by hand, absent or "" for none */
  groupSymbol?: string;
  groupColour?: string;
}

/** A stretch of track between two nodes (SPEC 6.2). */
export interface TrackEdge {
  id: EdgeId;
  a: NodeId;
  b: NodeId;
  tracks: 1 | 2;
  built: boolean;
  /** a blueprint switched off (T-333): Construct skips it */
  off?: boolean;
  /** high-speed track (T-817): always double */
  hsr?: boolean;
  /** `TrackApi.edge_kind`: 2 street tram, 3 tram (T-2926); absent for rail */
  kind?: number;
  /** one-way single track (T-3094): 1 trains run a to b only, 2 b to a only; absent both ways */
  oneWay?: 1 | 2;
  /** one diagonal of a crossover (T-338): fixed shape, deleted with the other */
  crossover?: boolean;
  /** a crossover diagonal's other diagonal, -1 if none */
  partner?: EdgeId;
  lengthM: number;
  /** build cost, US$ */
  cost: number;
  /** lowest curve speed limit on it, km/h */
  vminKmh: number;
  levelMin: Level;
  levelMax: Level;
  /** lines whose path uses it */
  lines: LineId[];
}

export interface TrackNode {
  id: NodeId;
  /** pack metres */
  x: number;
  y: number;
  lng: number;
  lat: number;
  level: Level;
  /** edge ends here, and how many of them are constructed */
  ports: number;
  builtPorts: number;
  flying: boolean;
  /** station platform, 0 = none */
  platform: number;
  /** T-1464: a bus stop that is no station: no track, never picked or snapped to */
  bus?: true;
  /** T-2194: a port: a station with no track, so no place to snap track to */
  port?: true;
  /** `TrackApi.node_kind` where it is a tram's (2 or 3, T-2926); absent for rail */
  kind?: number;
}

export interface Save {
  /** US$ */
  cash: number;
  stations: StationInput[];
  lines: LineInput[];
  /** stations the player linked into one complex (T-197), each pair once */
  links: [StationId, StationId][];
  /** the station buildings (T-2217, game/buildings.ts) */
  buildings?: BuildingInput[];
}

/** A station building as the track model lists it (T-2217, `TrackApi.buildings_info`). */
export interface BuildingInput {
  id: number;
  name: string;
  /** corners, x, y pairs, world units, in the order drawn */
  corners: number[];
}

// ---------- results from the workers ----------

export type LineStatus =
  /** carries trains */
  | "running"
  /** stops and path fine, some of it still blueprint */
  | "planned"
  /** its path lost track, or a stop went */
  | "broken";

export interface LineStats {
  status: LineStatus;
  km: number;
  /** time from the start of the line to each stop, s, dwells included (the first is 0), at the
   * demand level now in force */
  fromStartS: number[];
  /** the same per demand level (high, medium, low) */
  fromStartByLevel: number[][];
  /** whole round trip including dwells and turnarounds, s, at the level now in force */
  roundTripS: number;
  /** per demand level: round trip s, trains needed, capacity delay over the round trip s */
  roundTripByLevel: [number, number, number];
  trainsNeeded: [number, number, number];
  /** headway in seconds per level when the trains follow it (T-625, 0: no service), else null */
  autoS: [number, number, number] | null;
  delayS: [number, number, number];
  /** cars per train it runs, and the length the player set (T-733: 0 for the most its shortest platform fits) */
  cars: number;
  carsSet: number;
  /** high-speed trains (T-817) */
  fast: boolean;
  /** a line on tram track (T-2926) */
  tram: boolean;
  /** its share of the upkeep of the track and stations it uses, US$ a game day (T-867) */
  upkeep: number;
  /** per stretch between stops, run 0: the top speed reached and the limit there, km/h, and what
   * sets the limit (0 the trains, 1 ordinary track, 2 curves, 3 a crossover) */
  stretches: { reached: number; limit: number; why: number }[];
  /** where each stop is along run 0's and run 1's path, m */
  stopS: [number[], number[]];
  /** its edges first stop to last, `edge << 1 | direction` (T-2420) */
  path: number[];
  /** dwell at each intermediate stop, and the layover at each end, s */
  dwellS: number;
  turnaroundS: number;
  /** seconds the line's local time runs ahead of the game clock (T-309, game/localTime.ts) */
  shift: number;
  /** the stations it passes without stopping, in path order; gap i lies between stop i and i + 1 (T-646) */
  passed: { gap: number; node: number }[];
  /** the stops (indices) where its trains reverse partway, standing the turnaround (T-797) */
  reversals: number[];
  /** the bare track ends (node ids) where its trains reverse between stops, standing the turnaround (T-797 B3) */
  endTurns: number[];
  /** T-2895: the player set its stop order by hand (`TrackApi::line_hand`) */
  hand: boolean;
  /** a loop (T-798): its last stop is its first; a circle carries on round there, a balloon turns
   * back. Run both ways, each way has its own trains, half the line's (`loopHalves`). */
  loop: "circle" | "balloon" | null;
  /** T-2981: a loop that runs one way, round in the direction it was drawn, with all its trains */
  oneWay: boolean;
  /** T-2987: the player's speed limit for the line's trains, km/h; 0 for none */
  limit: number;
  /** T-1464: a bus line's route and stops (its stops that are no station are in `nodes` only,
   * never in the save's stations); absent for a rail line */
  bus?: import("../workers/protocol").ClockBus;
}

export interface ModeSplit {
  train: number;
  walk: number;
  drive: number;
  /** T-2215: trips no way serves (between two land parts, not by rail), "No route"; absent as 0 */
  none?: number;
}

/** A loop run both ways (T-798, T-2981): its round trip is two laps, its trains half each way. */
export function loopHalves(st: Pick<LineStats, "loop" | "oneWay">): boolean {
  return !!st.loop && !st.oneWay;
}

/** The split of `trips` with `rail`, `walk` and `none` (T-2215) of them, % of trips; driving is
 * the rest. */
export function splitOf(trips: number, rail: number, walk: number, none: number): ModeSplit {
  const pct = (v: number) => (trips > 0 ? (100 * v) / trips : 0);
  const n = pct(none);
  return { train: pct(rail), walk: pct(walk), drive: Math.max(0, 100 - pct(rail) - pct(walk) - n), none: n };
}

/** Construction totals. Cash, trains, fares and running costs: game/money.ts (T-028). */
export interface MoneyStats {
  /** what is constructed cost, and what the blueprint would, US$ */
  builtValue: number;
  blueprintCost: number;
}

/** Construction model's grouped quote, US$ (game/quotes.ts asks for it). Track quantities are
 * metres; others are counts. */
export interface BlueprintItem {
  kind: number;
  level: Level;
  tracks: number;
  wet: boolean;
  /** a track row's terrain class (T-1839: ramps are not told apart), 0 for the rest */
  ground: number;
  quantity: number;
  cost: number;
}

export interface WorldView {
  /** the clock worker's state version */
  version: number;
  /** busy resources per demand level (SPEC 6.2), local units */
  capacity: { x: number; y: number; rho: number; delayS: number; kind: number; tph: number; held: boolean }[][];
  /** T-819: per demand level, the edges whose track is at capacity, with the fuller direction's
   * utilisation and trains an hour */
  fullTrack: Map<EdgeId, { rho: number; tph: number }>[];
  save: Save;
  lineStats: Record<LineId, LineStats>;
  edges: TrackEdge[];
  nodes: Map<NodeId, TrackNode>;
  money: MoneyStats;
  canUndo: boolean;
  canRedo: boolean;
}

// ---------- selection ----------

/** The demand bubbles' end: trips from homes, to destinations, or both together (T-1903). */
export type BubbleEnd = "home" | "work" | "both";
/** A bubble selection taken one way, for the trip explorer, which counts trips from homes or to destinations. */
export type CommutersOneWay = Extract<Selection, { kind: "commuters" }> & { end: "home" | "work" };

/** What the inspector shows. `null` = nothing selected, and the inspector is empty. */
export type Selection =
  | { kind: "line"; line: LineId }
  /** a line group's view (T-1791): its group number */
  | { kind: "group"; group: number }
  /** T-2949: a stop that two or more bus lines share: their ids, and the id of one of its stops (its name) */
  | { kind: "busStop"; lines: LineId[]; stop: string }
  | { kind: "station"; station: StationId }
  | { kind: "track"; edge: EdgeId }
  | { kind: "junction"; node: NodeId }
  /** a train: its profile (line, run, demand level) and departure, absolute game seconds */
  | { kind: "train"; line: LineId; profile: number; dep: number }
  /** commuter bubbles picked on the map (T-097): the cells (city pack order) of `areas` bubbles,
   * at the homes or the jobs end, or both (T-1903) */
  | { kind: "commuters"; end: BubbleEnd; cells: number[]; areas: number }
  /** a city's longer-trips bubble picked zoomed far out (T-186): the registry's city id */
  | { kind: "city"; city: string }
  /** T-2395: the all cities view, the Cities tab's first row */
  | { kind: "world" }
  /** T-2124: the achievements, opened from the Cities tab */
  | { kind: "achievements" }
  /** the trip explorer (T-195): where a trip starts and ends, world metres; `to` null until the
   * second click. An end picked by clicking a station (T-221) is at the station's position and
   * names it. */
  | { kind: "trip"; from: [number, number]; to: [number, number] | null; fromStation?: StationId; toStation?: StationId;
      /** T-866: the far end is another zone's: a market place, or one of its 6 km zones (`gid`, -1
       * the whole place); the trips are the regional market's */
      far?: { place: number; gid: number };
      /** T-885: the near end is a market place or 6 km zone too, rather than the focused zone's cells */
      near?: { place: number; gid: number } }
  | null;
