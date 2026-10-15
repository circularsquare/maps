import type { CityId } from "../game/cities";
import type { FrameSpec } from "../game/frame";
// Messages between the main thread and the demand workers (T-026, T-021; notes/T-026.md).
//
// Each worker owns a copy of the city (parsed from the pack it is sent once) and a fixed set of
// the day's five periods (game/periods.ts). A `solve` carries the whole network, packed; the
// worker answers with one `result` per period and round as each lands (round 0 = the free-flow
// first estimate, then crowding rounds), then `done`. A worker solves each cluster of the network
// on its own (T-326, demandClusters.ts); every answer is in the whole network's stations and route
// nodes. A newer `solve` arriving mid-way supersedes
// the running one at the next period boundary: the old version's remaining results are never
// sent.

/**
 * The network packed for `DemandApi.set_network` (sim/src/demand/api.rs). Stations are indices
 * into the snapshot's station list; lines are in the snapshot's order.
 */
export interface DemandNetwork {
  /** the world frame of `stXY` (T-125): the loaded game's */
  frame: FrameSpec;
  /** x, y per station in that frame; each pack receives a copy in its own frame */
  stXY: Float32Array;
  /** each station's complex (T-161, `DemandApi.set_complexes`): equal numbers are one complex */
  complex: Uint32Array;
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
  /** start the WASM with the periods this worker solves; city packs (header text and .bin) and the
   * regional market may come here or in `open`s after it (T-149) */
  | { kind: "init"; packs: { id: CityId; header: string; bin: ArrayBuffer }[]; intercity?: string | Uint8Array; periods: number[]; aux?: boolean;
      /** T-173: open a small synthetic everyday layer first (`everyday_warm_up`), before `ready` */
      warmEveryday?: boolean;
      /** T-325, the aux worker: each solving worker's periods, in worker order, so a zone it holds
       * alone answers period group by period group as the solving workers would */
      groups?: number[][] }
  /** open more city packs and/or the regional market (T-149); answered by `opened`. The market is
   * the pack's text or, from the page, its UTF-8 bytes (T-180: decoded in the worker; the aux
   * worker, which never opens it, gets none). `bare` (T-325, aux worker only): a zone no station
   * reaches, opened for its bubbles and city figures alone (no rail, so no solve needs it); a
   * later `open` of the same pack without `bare` adds it to the solves as well */
  | { kind: "open"; packs: { id: CityId; header: string; bin: ArrayBuffer }[]; intercity?: string | Uint8Array; bare?: boolean }
  /** solve every period of this worker on `net`: free flow, then `rounds` crowding rounds */
  | { kind: "solve"; version: number; net: DemandNetwork; rounds: number;
      /** T-327: start each crowding round only once `go` allows it */
      barrier?: boolean;
      /** T-326 `?perfOff=clusters`: solve the network as one cluster */
      whole?: boolean;
      /** T-409 `?perfOff=split`: the market's zones as one, so its clusters merge through it */
      oneMarket?: boolean;
      /** T-124 `?perfOff=periods`: a cluster solved solves every period, none kept */
      fresh?: boolean;
      /** T-393: the city in view: its cluster is solved first */
      first?: CityId }
  /** T-327: every worker has finished `round - 1` of `version`, so this worker may start `round`.
   * A worker ahead waits at the boundary holding its last round, so every worker holds the round
   * after the published one at some moment, and a question about it can be answered (the final
   * waits on the slowest worker either way). `group` (T-402, 0 when absent): the solve's group of
   * clusters the round is of, as `plan.ahead` says (0 the cluster solved ahead, 1 the rest) */
  | { kind: "go"; version: number; round: number; group?: number }
  /**
   * Queries for the demand views (T-078, notes/T-078.md), answered between periods. `id` matches
   * the reply; `version` is the solve they ask about: a worker that has moved on to a newer
   * network answers with nulls. Every query also carries `day` (T-260, game/ui.ts `shownDay`,
   * 0 when absent): the kind of day its answer weighs the periods for. A question that reads the
   * periods' state may also carry `wait` (T-327): if this worker has not yet reached `round` of
   * `version`, it is answered as soon as it has, just before that round's last `result`.
   */
  /** the city's cells and zones (static; any worker) */
  /** `frame`: the world frame the answer's coordinates are wanted in (T-125) */
  | { kind: "cells"; id: number; city: CityId; frame: FrameSpec }
  /** rail trips by subzone over this worker's periods (`DemandApi.sub_sums`) */
  | { kind: "subSums"; id: number; city: CityId; version: number; round: number }
  /** commuters a day per cell by mode from every worker's `subSums` added up (one worker) */
  | { kind: "cellModes"; id: number; city: CityId; version: number; round: number; sums: Float32Array }
  /** one station's riders over this worker's periods (`DemandApi.station_riders`); `station`
   * indexes the solve's station list (`DemandNetwork.stXY`) */
  | { kind: "station"; id: number; city: CityId; version: number; round: number; station: number }
  /** a selection's rail trips over this worker's periods (`DemandApi.flow_rail`, T-098); `end`
   * 0 homes, 1 jobs; `cells` pack cell ids */
  | { kind: "flowRail"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array }
  | { kind: "purposes"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array }
  /** the selection's commuters by mode spread over the far end, from every worker's `flowRail`
   * added up (`DemandApi.flows`, one worker) */
  | { kind: "flows"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array; sums: Float32Array }
  /** longer trips (T-118) of a bubble selection's cells over this worker's periods
   * (`IntercityApi.selection_flows`); `end` 0 homes, 1 jobs; `cells` pack cell ids of `city` */
  | { kind: "icSel"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array;
      /** T-336: the selection's squares, from the aux worker's `bareSquares`, for a zone open bare
       * there (this worker does not hold it); `cells` and `end` are then unread */
      sq?: { xy: Float32Array; share: Float32Array } }
  /** a station's longer-trip rail riders over this worker's periods (`IntercityApi.station_flows`) */
  | { kind: "icStation"; id: number; city: CityId; version: number; round: number; station: number }
  /** one everyday purpose's local trip legs per cell by mode over this worker's periods
   * (`EverydayApi.cell_modes_of`, T-158); `purpose` indexes `EVERYDAY_KEYS` (game/purposes.ts) */
  | { kind: "purposeModes"; id: number; city: CityId; version: number; round: number; purpose: number }
  /** transfers over this worker's periods (T-176), every population and purpose together, with
   * either end at one of `nodes` (route nodes, the result arrays' order) */
  | { kind: "transfers"; id: number; city: CityId; version: number; round: number; nodes: Uint32Array }
  /** the trip explorer (T-195): one trip from `from` to `to` (world metres) as a commute, in every
   * period this worker holds (`DemandApi.trip`) */
  | { kind: "trip"; id: number; city: CityId; version: number; round: number; from: [number, number]; to: [number, number] }
  /** T-325, the aux worker: a bare zone's legs per cell with no rail, any version. `purpose` -1:
   * commutes and every everyday purpose, -2: commutes alone, k >= 0: everyday kind k alone */
  | { kind: "bare"; id: number; city: CityId; version: number; round: number; purpose: number }
  /** T-336, the aux worker: a bubble selection of a bare zone, as `flows` and `purposes` answer
   * for a solved one; any version */
  | { kind: "bareFlows"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array }
  /** T-336, the aux worker: a bare zone's selection as the regional market's squares, for the
   * solving workers' `icSel` */
  | { kind: "bareSquares"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array };

/** T-325: a bare zone's figures per period (index q), as a solve's `result` would carry them:
 * `citySummary`'s numbers and `cityParts`'s, the latter in the order of its own `keys`. */
export interface BareFigures { summary: number[][]; parts: number[][]; keys: string[] }

export type FromDemand =
  | {
      kind: "ready";
      cityInfo: Record<CityId, number[]>;
      /** WASM start and pack parse + gravity rebuild, ms */
      wasmMs: number;
      openMs: number;
      /** a solve on a small network, so V8 has optimised the kernel before the first real one */
      warmMs: number;
      /** DemandApi.info(): cells, zones, home-to-work trips a day, walk share with no rail, open ms
       * (the home city's; absent when `init` carried no pack for it) */
      info?: number[];
      /** demand_periods(): 6 numbers per period (from, to, level, to-work share, to-home share, peak-hour factor) */
      periods: number[];
      /** T-260 `day_factors()`: load factor per kind of day, purpose and period, then each day's
       * trip rate per purpose (sim/src/demand/days.rs) */
      days: number[];
      /** WASM linear memory, bytes */
      memBytes: number;
    }
  /** an `open` is done: its cities' info as `ready`'s, and whether it opened the market */
  | { kind: "opened"; cityInfo: Record<CityId, number[]>; market: boolean; openMs: number; warmMs: number; memBytes: number;
      /** T-325: the packs were opened bare (aux worker), with each one's figures */
      bare?: Record<CityId, BareFigures> }
  | { kind: "failed"; error: string }
  | {
      kind: "result";
      version: number;
      period: number;
      /** DemandApi.summary(): round, trips, rail, walk, ms, pairs, crowding stats */
      summary: number[];
      citySummary: Record<CityId, number[]>;
      /** T-260: per city, trips, rail trips and walked trips of each purpose (`parts.keys` order,
       * the market's longer trips left out), so a weekend's split can be worked out */
      cityParts?: Record<CityId, number[]>;
      intercitySummary: number[];
      /** T-276: IntercityApi.part_summary(): longer trips and rail, then commute legs between
       * zones and rail (both in `intercitySummary`) */
      intercityParts?: number[];
      /** per market pair (from, to, longer trips, rail) */
      corridors: number[];
      /** T-276: per market pair in `corridors` order, commute legs between zones and rail legs */
      commuteCorridors?: number[];
      /** per route node (line by line, run 0 stops then run 1 stops): riders in the period */
      seg: Float32Array;
      board: Float32Array;
      alight: Float32Array;
      /** busiest-hour load on the segment leaving the node, share of crush capacity */
      loadOfCrush: Float32Array;
      /**
       * The same `seg`, `board` and `alight` split by trip purpose (T-158, the map's purpose
       * filter): `keys` (game/purposes.ts `PurposeKey`) and, per key in that order, one block of
       * one value per route node. The purposes add up to the totals. Crowding is never split.
       */
      parts?: { keys: string[]; seg: Float32Array; board: Float32Array; alight: Float32Array };
      /** this stage in the worker, ms, and set_network's ms when this was the version's first stage */
      ms: number;
      setupMs: number;
      memBytes: number;
      /** T-392: one cluster's final figures alone, sent the moment it is solved while the solve goes
       * on with other clusters: its cities (only theirs in `citySummary`), whether the market is in it
       * (else the intercity fields are empty), its lines (indices into the network's); every array
       * still the whole network's, zero off the cluster. T-402: in a solve with the barrier, the
       * cluster solved ahead (`plan.ahead`) sends one per round, the round in `summary[0]` */
      cluster?: { cities: CityId[]; market: boolean; lines: Uint32Array };
    }
  /** T-392: at a solve's start, the cities of each cluster it solves (in order) and of those it
   * keeps. `ahead` (T-402, a solve with the barrier): the cities of the cluster in view, stepped
   * through its rounds alone (group 0) before the other clusters together (group 1) */
  | { kind: "plan"; version: number; solving: CityId[][]; kept: CityId[]; ahead?: CityId[] }
  /** `clusters` (T-326): how many kept their result, and how many were solved; `periods` (T-124):
   * of this worker's periods in the clusters solved, how many were kept and how many solved */
  | { kind: "done"; version: number; memBytes: number; clusters?: [number, number]; periods?: [number, number] }
  /** a worker's `sub_sums` for one city once all its periods have `round` (sent unasked, T-119) */
  | { kind: "sums"; version: number; round: number; city: CityId; mask: number; sums: Float32Array; everyday: Float32Array | null }
  | { kind: "cells"; id: number; city: CityId; xy: Float32Array; h3: BigUint64Array; zone: Uint32Array; zoneXY: Float32Array }
  /** `mask`: bit q set for each period included; null sums: no network, or not `version` */
  | { kind: "subSums"; id: number; city: CityId; version: number; mask: number; sums: Float32Array | null; everyday: Float32Array | null }
  /** `everyday`: per purpose kind 30 timing values, m, then m far cells, their rail, walk and
   * drive legs (`EverydayApi.selection_sparse_day`, T-285) */
  | { kind: "purposes"; id: number; city: CityId; version: number; mask: number; work: Float32Array | null; everyday: Float32Array | null; names: string[] }
  /** six blocks of one value per cell (home rail, walk, drive, work rail, walk, drive) */
  | { kind: "cellModes"; id: number; city: CityId; version: number; modes: Float32Array | null }
  | { kind: "station"; id: number; city: CityId; version: number; mask: number; packed: Float32Array | null }
  | { kind: "flowRail"; id: number; city: CityId; version: number; mask: number; sums: Float32Array | null }
  /** `[rail, walk, drive, cut rail, cut walk, cut drive, n, n cells.., rail.., walk.., drive..]` */
  | { kind: "flows"; id: number; city: CityId; version: number; packed: Float32Array | null }
  /** `mask` as `subSums`; `packed` as `IntercityApi.selection_flows` (null: no network, or not `version`) */
  | { kind: "icSel"; id: number; city: CityId; version: number; mask: number; packed: Float64Array | null }
  /** `packed` as `IntercityApi.station_flows` */
  | { kind: "icStation"; id: number; city: CityId; version: number; mask: number; packed: Float64Array | null }
  /** six blocks of one value per cell as `cellModes`, trip legs; empty when the city has no such
   * purpose; null: no network, or not `version` and `round` */
  | { kind: "purposeModes"; id: number; city: CityId; version: number; mask: number; modes: Float32Array | null }
  /** triples (route node arriving, route node leaving, riders), each pair once, summed over this
   * worker's periods; null: no network, or not `version` and `round` */
  | { kind: "transfers"; id: number; city: CityId; version: number; mask: number; pairs: Float32Array | null }
  /** one `DemandApi.trip` answer per period in `periods` (the cells' x, y turned into world
   * metres); null: no network, or not `version` and `round` */
  | { kind: "trip"; id: number; city: CityId; version: number; mask: number; periods: number[]; answers: Float64Array[] | null }
  /** T-325: `commute` as `cellModes`'s six blocks (commuters, not legs); `parts` one block set per
   * solving worker's periods (`masks`), the everyday legs as that worker's `sums` message would
   * carry them (empty arrays for a kind the zone lacks; none without an everyday layer). Null
   * `commute`: not open bare here */
  | { kind: "bare"; id: number; city: CityId; commute: Float32Array | null; masks: number[]; parts: Float32Array[] }
  /** T-336: `packed` as `flows`'s (null: not open bare here); per solving worker's periods
   * (`masks`), `work` and `everyday` as that worker's `purposes` answer would carry them
   * (`everyday` empty without an everyday layer) */
  | { kind: "bareFlows"; id: number; city: CityId; packed: Float32Array | null; masks: number[]; work: Float32Array[]; everyday: Float32Array[]; names: string[] }
  /** T-336: the squares' centres in the market frame and the share of each selected; null: not
   * open bare here, or outside the regional market */
  | { kind: "bareSquares"; id: number; city: CityId; xy: Float32Array | null; share: Float32Array | null };
