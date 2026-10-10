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
  /** metres east and north of the city pack's origin */
  x: number;
  y: number;
  level: Level;
  /** platform length, m (SPEC 6.1: 60-400 in 20 m steps) */
  length: number;
  /** constructed; false = blueprint (SPEC 6.4) */
  built: boolean;
  /** track heading through it, rad, counter-clockwise from east */
  heading: number;
}

export interface LineInput {
  id: LineId;
  num: number;
  /** one or two characters on the line's chip */
  letter: string;
  name: string;
  /** "#rrggbb", from the saturated line palette or player-set */
  colour: string;
  stops: StationId[];
  tph: Tph;
}

/** A stretch of track between two nodes (SPEC 6.2). */
export interface TrackEdge {
  id: EdgeId;
  a: NodeId;
  b: NodeId;
  tracks: 1 | 2;
  built: boolean;
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
}

export interface Save {
  version: 0;
  city: string;
  /** game seconds since day 0 00:00 */
  clock: number;
  /** US$ */
  cash: number;
  stations: StationInput[];
  lines: LineInput[];
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
  delayS: [number, number, number];
  cars: number;
  /** what constructing the rest of its track and stations costs, US$ (0 when running) */
  buildCost: number;
  /** what the trains it needs will cost when it is constructed, after spare cars, US$ (T-028) */
  trainCost: number;
  /** where each stop is along run 0's and run 1's path, m */
  stopS: [number[], number[]];
  /** dwell at each intermediate stop, and the layover at each end, s */
  dwellS: number;
  turnaroundS: number;
}

export interface ModeSplit {
  train: number;
  walk: number;
  drive: number;
}

export interface CityStats {
  name: string;
  population: number;
  jobs: number;
  split: ModeSplit;
  /** train share of trips by period, % */
  trainShareByPeriod: [string, number][];
}

/** Construction totals. Cash, trains, fares and running costs: game/money.ts (T-028). */
export interface MoneyStats {
  /** what is constructed cost, and what the blueprint would, US$ */
  builtValue: number;
  blueprintCost: number;
  blueprintItems: BlueprintItem[];
}

/** Construction model's grouped quote, US$. Track quantities are metres; others are counts. */
export interface BlueprintItem {
  kind: number;
  level: Level;
  tracks: number;
  wet: boolean;
  ramp: boolean;
  quantity: number;
  cost: number;
}

export interface WorldView {
  /** the clock worker's state version */
  version: number;
  /** busy resources per demand level (SPEC 6.2), local units */
  capacity: { x: number; y: number; rho: number; delayS: number; kind: number }[][];
  save: Save;
  lineStats: Record<LineId, LineStats>;
  edges: TrackEdge[];
  nodes: Map<NodeId, TrackNode>;
  city: CityStats;
  money: MoneyStats;
  canUndo: boolean;
  canRedo: boolean;
}

// ---------- selection ----------

/** What the inspector shows. `null` = nothing selected, and the inspector is empty. */
export type Selection =
  | { kind: "line"; line: LineId }
  | { kind: "station"; station: StationId }
  | { kind: "track"; edge: EdgeId }
  | { kind: "junction"; node: NodeId }
  /** a train: its profile (line, run, demand level) and departure, absolute game seconds */
  | { kind: "train"; line: LineId; profile: number; dep: number }
  /** commuter bubbles picked on the map (T-097): the cells (city pack order) of `areas` bubbles,
   * at the homes or the jobs end */
  | { kind: "commuters"; end: "home" | "work"; cells: number[]; areas: number }
  | { kind: "cell"; cell: string }
  | { kind: "zone"; zone: number }
  | null;
