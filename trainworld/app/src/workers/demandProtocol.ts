// Messages between the main thread and the demand workers (T-026, T-021; notes/T-026.md).
//
// Each worker owns a copy of the city (parsed from the pack it is sent once) and a fixed set of
// the day's five periods (game/periods.ts). A `solve` carries the whole network, packed; the
// worker answers with one `result` per period and round as each lands (round 0 = the free-flow
// first estimate, then crowding rounds), then `done`. A newer `solve` arriving mid-way supersedes
// the running one at the next period boundary: the old version's remaining results are never
// sent.

/**
 * The network packed for `DemandApi.set_network` (sim/src/demand/api.rs). Stations are indices
 * into the snapshot's station list; lines are in the snapshot's order.
 */
export interface DemandNetwork {
  /** x, y per station, metres east and north of the pack origin */
  stXY: Float32Array;
  /** stops per line */
  lineN: Uint32Array;
  /** station index of every stop, line after line */
  stops: Uint32Array;
  /**
   * Riding seconds, 6 per stop: for each line, per run (0 = stops in order, 1 = reversed) and
   * demand level (high, medium, low), `n` values indexed by stop k in the line's own order.
   * Run 0: stop k to k + 1 (0 at the last stop); run 1: stop k to k - 1 (0 at the first).
   * Includes the dwell at the stop the train leaves.
   */
  times: Float32Array;
  /** trains an hour, 3 per line: high, medium, low */
  tph: Float32Array;
  /** cars per train, per line */
  cars: Float32Array;
  fare: { base: number; perKm: number };
  /** km per route node: each line's forward stops then reverse stops, 0 at each end */
  hopKm: Float32Array;
}

export type ToDemand =
  /** the city pack (header text and .bin) and the periods this worker solves */
  | { kind: "init"; header: string; bin: ArrayBuffer; periods: number[] }
  /** solve every period of this worker on `net`: free flow, then `rounds` crowding rounds */
  | { kind: "solve"; version: number; net: DemandNetwork; rounds: number }
  /**
   * Queries for the demand views (T-078, notes/T-078.md), answered between periods. `id` matches
   * the reply; `version` is the solve they ask about: a worker that has moved on to a newer
   * network answers with nulls.
   */
  /** the city's cells and zones (static; any worker) */
  | { kind: "cells"; id: number }
  /** rail trips by subzone over this worker's periods (`DemandApi.sub_sums`) */
  | { kind: "subSums"; id: number; version: number }
  /** commuters a day per cell by mode from every worker's `subSums` added up (one worker) */
  | { kind: "cellModes"; id: number; version: number; sums: Float32Array }
  /** one station's riders over this worker's periods (`DemandApi.station_riders`); `station`
   * indexes the solve's station list (`DemandNetwork.stXY`) */
  | { kind: "station"; id: number; version: number; station: number }
  /** a selection's rail trips over this worker's periods (`DemandApi.flow_rail`, T-098); `end`
   * 0 homes, 1 jobs; `cells` pack cell ids */
  | { kind: "flowRail"; id: number; version: number; end: number; cells: Uint32Array }
  /** the selection's commuters by mode spread over the far end, from every worker's `flowRail`
   * added up (`DemandApi.flows`, one worker) */
  | { kind: "flows"; id: number; version: number; end: number; cells: Uint32Array; sums: Float32Array };

export type FromDemand =
  | {
      kind: "ready";
      /** WASM start and pack parse + gravity rebuild, ms */
      wasmMs: number;
      openMs: number;
      /** a solve on a small network, so V8 has optimised the kernel before the first real one */
      warmMs: number;
      /** DemandApi.info(): cells, zones, home-to-work trips a day, walk share with no rail, open ms */
      info: number[];
      /** demand_periods(): 6 numbers per period (from, to, level, to-work share, to-home share, peak-hour factor) */
      periods: number[];
      /** WASM linear memory, bytes */
      memBytes: number;
    }
  | { kind: "failed"; error: string }
  | {
      kind: "result";
      version: number;
      period: number;
      /** DemandApi.summary(): round, trips, rail, walk, ms, pairs, crowding stats */
      summary: number[];
      /** per route node (line by line, run 0 stops then run 1 stops): riders in the period */
      seg: Float32Array;
      board: Float32Array;
      alight: Float32Array;
      /** busiest-hour load on the segment leaving the node, share of crush capacity */
      loadOfCrush: Float32Array;
      /** this stage in the worker, ms, and set_network's ms when this was the version's first stage */
      ms: number;
      setupMs: number;
      memBytes: number;
    }
  | { kind: "done"; version: number; memBytes: number }
  | { kind: "cells"; id: number; xy: Float32Array; h3: BigUint64Array; zone: Uint32Array; zoneXY: Float32Array }
  /** `mask`: bit q set for each period included; null sums: no network, or not `version` */
  | { kind: "subSums"; id: number; version: number; mask: number; sums: Float32Array | null }
  /** six blocks of one value per cell (home rail, walk, drive, work rail, walk, drive) */
  | { kind: "cellModes"; id: number; version: number; modes: Float32Array | null }
  | { kind: "station"; id: number; version: number; mask: number; packed: Float32Array | null }
  | { kind: "flowRail"; id: number; version: number; mask: number; sums: Float32Array | null }
  /** `[rail, walk, drive, cut rail, cut walk, cut drive, n, n cells.., rail.., walk.., drive..]` */
  | { kind: "flows"; id: number; version: number; packed: Float32Array | null };
