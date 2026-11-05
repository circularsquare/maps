import type { CityId } from "../game/cities";
import type { CorridorTable } from "../game/demand";
import type { FrameSpec } from "../game/frame";

/** `DemandApi.trip`'s answer layout (sim/src/demand/trip.rs `HEAD`, `PAIR_LEN`, `LEG_LEN`). */
export const TRIP_LAYOUT = { HEAD: 48, PAIR_LEN: 20, LEG_LEN: 13 } as const;
/** `IntercityApi::market_trip`'s answer: `MT_HEAD` and `MT_PAIR_LEN` in market_trip.rs. */
export const MARKET_TRIP_LAYOUT = { HEAD: 25, PAIR_LEN: 7 } as const;
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
  /** each station's upgrades (T-1281, `DemandApi.set_upgrades`), absent where none has any */
  ups?: Uint8Array;
  /** station pairs joined by a passage, flat (T-1281, `DemandApi.set_passages`), absent where none */
  passages?: Uint32Array;
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
  fare: { base: number; perKm: number; nodes?: [number, number][]; scales?: Record<string, number>; hsr?: number; transfer?: number };
  /** km per route node: each line's forward stops then reverse stops, 0 at each end */
  hopKm: Float32Array;
  /** T-1626 (`DemandApi.set_fare_mul`): each route node's fare multiplier, as `hopKm`; absent: all 1 */
  fareW?: Float32Array;
  /** T-842 (`DemandApi.set_span`): per period and route node (`q * route nodes + r`, route nodes
   * as `hopKm`), the share of riders with a train due within their tolerance; empty or absent:
   * 1 everywhere */
  span?: Float32Array;
  /** T-1464 (`DemandApi.set_bus`): per stop (parallel to `stops`) the points of its stretch of
   * bus route, 0 for none, and their x, y after one another in `frame`; absent with no buses */
  catchN?: Uint32Array;
  catchXY?: Float32Array;
  /** T-2196 (`EverydayApi.set_ferry`): 1 for each ferry line, 0 for the rest; absent with none */
  ferry?: Uint8Array;
}

/** T-377: a zone's part of the world market (`data/packs/<id>.market.json`). */
export interface MarketPack { id: CityId; pack: string | Uint8Array }

/** T-776: the world market's rows (app/src/game/market-rows.json), as text or UTF-8 bytes; given
 * with or before a worker's first market pack, which cannot open without them. */
export type MarketRows = string | Uint8Array;

/** T-492: a pack a worker could not open: the zone, whether it was its market pack, and why */
export interface OpenRejected { id: CityId; market: boolean; error: string }

/**
 * T-1607: one side of a period shared by two workers (its cities' populations in one, its market
 * in the other): that side's segment loads, boardings and alightings after `round`, summed on the
 * whole network's route nodes (3 x nodes), for cluster `cluster` (index in the job's clusters).
 * The page passes it on to the other side's worker.
 */
/** T-1255: the riders coming to the platforms per route node: the free-flow pass's boardings, at
 * least the rounds' (before the cap), into `free` */
export function comeMax(free: Float32Array, rounds: Float32Array): Float32Array {
  for (let k = 0; k < free.length; k++) free[k] = Math.max(free[k], rounds[k]);
  return free;
}
/** The account's riders coming (`result.arrive`): everyone who would ride where the queue turned
 * riders away (their trains carry no more), the riders who ride elsewhere, so the account's
 * boardings stay demand's; worked out into `come` (`comeMax`) */
export function arriveOf(come: Float32Array, rounds: Float32Array, board: Float32Array): Float32Array {
  for (let k = 0; k < come.length; k++) come[k] = board[k] < 0.999 * rounds[k] ? Math.max(come[k], board[k]) : board[k];
  return come;
}
export type Loads ={ kind: "loads"; version: number; cluster: number; period: number; round: number; side: "cities" | "market"; loads: Float32Array };

export type ToDemand =
  /** start the WASM with the periods this worker solves; city packs (header text and .bin) and
   * zones' market packs may come here or in `open`s after it (T-149) */
  | { kind: "init"; packs: { id: CityId; header: string; bin: ArrayBuffer }[]; intercity?: MarketPack[]; rows?: MarketRows; periods: number[]; aux?: boolean;
      /** T-173: open a small synthetic everyday layer first (`everyday_warm_up`), before `ready` */
      warmEveryday?: boolean;
      /** T-889: open each `open` alone, as it comes (`?perfOff=batchOpen`) */
      oneByOne?: boolean;
      /** T-325, the aux worker: each solving worker's periods, in worker order, so a zone it holds
       * alone answers period group by period group as the solving workers would */
      groups?: number[][];
      /** T-1607: the periods whose market this worker solves in a solve with the barrier (`share`
       * true in every solving worker when any market sits apart from its period's cities) */
      markets?: number[]; share?: boolean }
  /** open more city packs and/or zones' market packs (T-149, T-377); answered by `opened`. A
   * market pack is its text or, from the page, its UTF-8 bytes (T-180: decoded in the worker; the
   * aux worker, which never opens one, gets empty bytes). `bare` (T-325, aux worker only): a zone
   * no station reaches, opened for its bubbles and city figures alone (no rail, so no solve needs
   * it); a later `open` of the same pack without `bare` adds it to the solves as well. `drives`
   * (T-279): the routed drives (market-drives.bin), with the rows. `first` (T-1533): the zone in
   * view's pack, opened ahead of the opens waiting and on its own. A pack's `level` (T-1223, `lod::COARSE` 1;
   * fine when absent): a solving worker holding the zone at another level opens it again at this one
   * (once the solve in hand ends, if it covers the zone); the aux worker opens every zone fine */
  | { kind: "open"; packs: { id: CityId; header: string; bin: ArrayBuffer; level?: number }[]; intercity?: MarketPack[]; rows?: MarketRows; drives?: Uint8Array; bare?: boolean; first?: boolean }
  /** T-301: close city packs and zones' market packs (`markets`) no longer needed, freeing their
   * WASM memory for the next pack; `bare`: the aux worker's bare copies. A pack the solve in hand
   * covers closes when that solve ends, unless an `open` of it comes first. Answered by `closed`. */
  | { kind: "close"; ids: CityId[]; markets?: CityId[]; bare?: boolean }
  /** solve every period of this worker on `net`: free flow, then `rounds` crowding rounds */
  | { kind: "solve"; version: number; net: DemandNetwork; rounds: number;
      /** T-327: start each crowding round only once `go` allows it */
      barrier?: boolean;
      /** T-326 `?perfOff=clusters`: solve the network as one cluster */
      whole?: boolean;
      /** T-409 `?perfOff=split`: the market's zones as one, so its clusters merge through it */
      oneMarket?: boolean;
      /** T-756 `?perfOff=reach`: every population on its whole cluster, not its reach */
      wide?: boolean;
      /** T-780, T-874 `?perfOff=chunks`: the market and the commutes solve every zone pair, none sampled */
      exactMarket?: boolean;
      /** T-124 `?perfOff=periods`: a cluster solved solves every period, none kept */
      fresh?: boolean;
      /** T-393: the city in view: its cluster is solved first */
      first?: CityId;
      /** T-2453: zones whose clusters come next, in order (the zone the city view shows) */
      next?: CityId[];
      /** T-2459: the zone in view is solved elsewhere (the focus pool): with the barrier, `next`'s
       * clusters go before `first`'s */
      focused?: boolean;
      /** T-1533: a rough solve of these zones alone (no market), while the packs a whole solve needs
       * still open: it starts ahead of the opens waiting, which wait for it */
      only?: CityId[];
      /** T-2045: the focus pool's solve of `only`: kept once solved, though its free flow was
       * estimated (no whole solve ever reuses it there), so its zone answers after the next solve */
      keep?: boolean;
      /** T-1538 `?perfOff=estimate`: no period's first figures estimated from another's */
      exact0?: boolean;
      /** T-1543: this solving worker's part k of n: a free-flow round shown before it is solved (a
       * rough solve's, the zone in view's preview after an edit) is one period solved in n parts of
       * its origins, every period estimated from each part, sent as `result`s with `part` that the
       * page adds up (`?perfOff=parts`: none) */
      part?: [number, number];
      /** T-1174: every market part made afresh from the newest composition, as a worker started
       * now has them (sent after a worker is replaced, so all workers' parts are alike again) */
      freshParts?: boolean;
      /** T-1609 `?perfOff=refine`: a cluster whose only new populations are zones opened again at
       * another level is solved whole, not those zones alone against the others' frozen loads */
      noLocal?: boolean;
      /** T-1735 `?perfOff=window`: an edit's local solve leaves the whole market frozen */
      noWindow?: boolean;
      /** T-1862 `?perfOff=noise`: a cluster keeps its result only on exactly the network it solved */
      exactKeep?: boolean;
      /** T-1733: the network's line ids (`lineN` order) and station ids (`stXY` order), so a worker
       * can carry what it holds solved over to the next network's route nodes */
      ids?: { lines: string[]; stations: string[] } }
  /** T-327: every worker has finished `round - 1` of `version`, so this worker may start `round`.
   * A worker ahead waits at the boundary holding its last round, so every worker holds the round
   * after the published one at some moment, and a question about it can be answered (the final
   * waits on the slowest worker either way). `group` (T-402, 0 when absent): the solve's group of
   * clusters the round is of, as `plan.ahead` says (0 the cluster solved ahead, 1 the rest) */
  | { kind: "go"; version: number; round: number; group?: number }
  /** T-2045: a solving worker of the main pool holds its solve and its opening at the next
   * population (`on`) until let go, while the focus pool solves the zone in view first */
  | { kind: "pause"; on: boolean }
  /** T-1733: every worker's local result after an edit is shown and a moment has passed: the whole solve of `version` may start */
  | { kind: "whole"; version: number }
  /** T-1533: the zone in view changed: its pack, if still waiting to open, goes ahead of the others */
  | { kind: "first"; id: CityId }
  /** T-1904: the kind of day shown (`day` as a query's) and the zones whose bubbles are drawn (the
   * zone in view and the nearby zones; none while the bubbles are off): a solve's `sums` for those
   * zones carry that day's too */
  | { kind: "view"; day: number; zones: CityId[] }
  /** T-1008: the page has lost the corridor tables sent: the next results carry theirs again */
  | { kind: "corridorsAgain" }
  /** T-1174, dev server only (`twDemand.client.debugPanic`): make this worker's WASM panic, `now`
   * or in the running solve's next step */
  | { kind: "debugPanic"; when: "now" | "solve" }
  /** T-1205: the pool has shrunk: this worker's periods from now on (`init`'s `periods`, `groups`) */
  | { kind: "periods"; periods: number[]; groups?: number[][]; markets?: number[]; share?: boolean }
  /** T-1607: a period's loads from the worker solving the other side of it (`FromDemand` "loads") */
  | Loads
  /**
   * Queries for the demand views (T-078, notes/T-078.md), answered between periods. `id` matches
   * the reply; `version` is the solve they ask about: a worker that has moved on to a newer
   * network answers with nulls. Every query also carries `day` (T-260, game/ui.ts `shownDay`,
   * 0 when absent): the day its answer weighs the periods for (0 weekday, 1 Saturday, 2 Sunday,
   * 3 the average day, T-464, T-262). A question that reads the
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
  /** the trip explorer (T-195): one trip from `from` to `to` (world metres) in every period this
   * worker holds, as a commute (`DemandApi.trip`), or with `purpose` (T-586: an index into
   * `EVERYDAY_KEYS`) as that everyday purpose (`EverydayApi.trip`; answers 0 for a city without it) */
  | { kind: "trip"; id: number; city: CityId; version: number; round: number; from: [number, number]; to: [number, number]; purpose?: number }
  /** T-866: the trip explorer between zones, in every period this worker holds
   * (`IntercityApi.market_trip`): the longer trips of a selection's cells (as `icSel`) with place
   * `far`, or its 6 km zone `farGid` (-1 the whole place) */
  | { kind: "icTrip"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array; far: number; farGid: number;
      sq?: { xy: Float32Array; share: Float32Array };
      /** T-885: the near end is place `place` (or its 6 km zone `gid`) rather than `cells`
       * (`IntercityApi.market_trip_between`) */
      near?: { place: number; gid: number } }
  /** T-325, the aux worker: a bare zone's legs per cell with no rail, any version. `purpose` -1:
   * commutes and every everyday purpose, -2: commutes alone, k >= 0: everyday kind k alone */
  | { kind: "bare"; id: number; city: CityId; version: number; round: number; purpose: number }
  /** T-336, the aux worker: a bubble selection of a bare zone, as `flows` and `purposes` answer
   * for a solved one; any version */
  | { kind: "bareFlows"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array }
  /** T-336, the aux worker: a bare zone's selection as the regional market's squares, for the
   * solving workers' `icSel` */
  | { kind: "bareSquares"; id: number; city: CityId; version: number; round: number; end: number; cells: Uint32Array }
  /** T-1435: the car legs rail takes over this worker's periods, by pair of the zone's squares
   * (`DemandApi.drive_rail_day`, `EverydayApi.drive_rail_day`) */
  | { kind: "driveRail"; id: number; city: CityId; version: number; round: number }
  /** T-1435: the zone's squares and one table with no rail, any version: `part` 0 the commutes'
   * (`DemandApi.drive_base_day`), 1 the everyday layer's */
  | { kind: "driveBase"; id: number; city: CityId; version: number; round: number; part: 0 | 1 }
  /** T-2365: the zone's trips by length over this worker's periods (`DemandApi.trip_lengths_day`,
   * `EverydayApi.trip_lengths_day`); `bare`: the aux worker's for a zone open bare, every period */
  | { kind: "tripLengths"; id: number; city: CityId; version: number; round: number; bare?: boolean };

/** T-1205: the WASM heap in use and the most it has been since the worker's last solve began
 * (`wasm_heap_bytes`, `wasm_heap_peak`), bytes */
export type Heap = [live: number, peak: number];

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
      /** demand_periods(): 6 numbers per period (from, to, level, to-work share, to-home share, peak-hour factor) */
      periods: number[];
      /** T-260 `day_factors()`: load factor per kind of day, purpose and period, then each day's
       * trip rate per purpose (sim/src/demand/days.rs) */
      days: number[];
      /** WASM linear memory, bytes */
      memBytes: number;
      heap?: Heap;
    }
  /** T-1195: one pack of an `open` is in (a zone's, or with `market` its market pack): the loading
   * bar's progress through a batch, which may take minutes */
  | { kind: "packOpened"; id: CityId; market?: boolean }
  /** an `open` is done: its cities' info as `ready`'s, and the zones whose market packs it opened */
  | { kind: "opened"; cityInfo: Record<CityId, number[]>; markets: CityId[]; openMs: number; warmMs: number; memBytes: number; heap?: Heap;
      /** T-325: the packs were opened bare (aux worker), with each one's figures */
      bare?: Record<CityId, BareFigures>;
      /** T-492: packs of this `open` that did not open (left out of `cityInfo`, `markets`, `bare`) */
      rejected?: OpenRejected[] }
  /** T-301: a `close` is done (WASM memory never shrinks: the freed space is reused) */
  | { kind: "closed"; memBytes: number; heap?: Heap }
  | { kind: "failed"; error: string }
  /** T-1174: the WASM panicked (or ran out of memory): every object it holds may be left borrowed,
   * so the worker does nothing more and the page replaces it. `panic`: `last_panic()`'s text */
  | { kind: "dead"; error: string; panic: string; memBytes: number }
  | {
      kind: "result";
      version: number;
      period: number;
      /** DemandApi.summary(): round, trips, rail, walk, ms, pairs, crowding stats; driving is
       * trips minus rail, walk and (index 13, T-2214) the trips no way serves */
      summary: number[];
      /** T-1300: rail trips per curve dollar dearer on every fare (negative), local and the market's,
       * then each one's reach: its sensitivity-weighted mean trip km over its rider-weighted one */
      fareSense?: [number, number, number, number];
      citySummary: Record<CityId, number[]>;
      /** T-260: per city, trips, rail trips and walked trips of each purpose (`parts.keys` order,
       * the market's longer trips left out), so a weekend's split can be worked out */
      cityParts?: Record<CityId, number[]>;
      intercitySummary: number[];
      /** T-276: IntercityApi.part_summary(): longer trips and rail, then commute legs between
       * zones and rail (both in `intercitySummary`), then (T-1175) everyday legs across a zone
       * line, their rail and walking legs, and their walking legs with no rail */
      intercityParts?: number[];
      /** T-1008: the market's corridor table this result's corridors index (`CorridorTable.key`);
       * the table itself rides along the first time this worker sends its key */
      corridorKey?: string;
      corridorTable?: CorridorTable;
      /** T-1008: the period's factors of the table's longer trips and commute legs */
      corridorShares?: [number, number];
      /** the pairs with rail as (index in the table, rail trips, commute rail legs) */
      corridorRail?: Float64Array;
      /** T-378: per place, its longer trips in the period with places that have no zone, each way */
      elsewhere?: number[];
      /** per route node (line by line, run 0 stops then run 1 stops): riders in the period */
      seg: Float32Array;
      board: Float32Array;
      alight: Float32Array;
      /** T-1255: per route node, the riders coming to the platform in the period: at least `board`,
       * those beyond it riders who would ride if the trains had room (the free-flow pass's), whom
       * the boarding account turns away from a full platform */
      arrive?: Float32Array;
      /** busiest-hour load on the segment leaving the node, share of crush capacity */
      loadOfCrush: Float32Array;
      /** per route node, US$ in the period: the fare curve's extra on the trips first boarding there
       * (T-1143, the sims' `fare_extra`; all 0 for a straight line) */
      fareX?: Float32Array;
      /** riders boarding at each route node after changing lines in the period, every purpose
       * (T-1530, the sims' `transfers_in`), one block of route nodes per kind of day
       * (`DEMAND_DAYS` order), each population's scaled as its `transfers_day` scales its pairs:
       * the stations' transfers without a query */
      xferIn?: Float32Array;
      /**
       * The same `seg`, `board` and `alight` split by trip purpose (T-158, the map's purpose
       * filter): `keys` (game/purposes.ts `PurposeKey`) and, per key in that order, one block of
       * one value per route node. The purposes add up to the totals. Crowding is never split.
       * `lean` (T-1373): the longer trips times their pair's lean, a block each of seg, board and
       * alight; a going-out day's longer trips are theirs plus it, a coming-back day's less it.
       */
      parts?: { keys: string[]; seg: Float32Array; board: Float32Array; alight: Float32Array; lean?: Float32Array };
      /** this stage in the worker, ms, and set_network's ms when this was the version's first stage */
      ms: number;
      setupMs: number;
      memBytes: number;
      /** T-392: one cluster's final figures alone, sent the moment it is solved while the solve goes
       * on with other clusters: its cities (only theirs in `citySummary`), whether the market is in it
       * (else the intercity fields are empty), its lines (indices into the network's); every array
       * still the whole network's, zero off the cluster. T-402: in a solve with the barrier, the
       * cluster solved ahead (`plan.ahead`) sends one per round, the round in `summary[0]` */
      /** T-1733 `local`: the cluster after an edit with only `cities` (the zones the edit touched)
       * solved, against the others' frozen loads, ahead of its whole solve; its lines' figures carry
       * every population's riders, the market's as last solved */
      cluster?: { cities: CityId[]; market: boolean; lines: Uint32Array; local?: boolean };
      /** T-2045: in a solve of `only` zones, the lines it gives figures (indices into the network's) */
      onlyLines?: Uint32Array;
      /** T-1543: one worker's part k of n of the period's origins (`solve.part`): every figure is the
       * part's, to be added to the other parts', the market's only in the worker holding the period.
       * `fareRaw`: the sums `fareSense`'s local reach is worked out from (rail trips per curve
       * dollar, the same times km, rail trips, their km), which add up where reaches do not */
      part?: [number, number];
      fareRaw?: [number, number, number, number];
      /** T-1607: a part's boardings summed as `arrive` reads them (the free-flow pass's, the last
       * round's before the cap), so the page works `arrive` out again from the added parts */
      come?: [Float32Array, Float32Array];
    }
  | Loads
  /** T-392: at a solve's start, the cities of each cluster it solves (in order) and of those it
   * keeps. `ahead` (T-402, a solve with the barrier): the cities of the cluster in view, stepped
   * through its rounds alone (group 0) before the other clusters together (group 1) */
  | { kind: "plan"; version: number; solving: CityId[][]; kept: CityId[]; ahead?: CityId[];
      /** T-1733: how many clusters this worker solves locally first, after an edit */
      local?: number }
  /** `clusters` (T-326): how many kept their result, and how many were solved; `periods` (T-124):
   * of this worker's periods in the clusters solved, how many were kept and how many solved */
  | { kind: "done"; version: number; memBytes: number; heap?: Heap; clusters?: [number, number]; periods?: [number, number]; local?: CityId[] }
  /** a worker's `sub_sums` for one city once all its periods have `round` (sent unasked, T-119) */
  | { kind: "sums"; version: number; round: number; city: CityId; mask: number; sums: Float32Array; everyday: Float32Array | null; day?: number }
  | { kind: "cells"; id: number; city: CityId; xy: Float32Array; h3: BigUint64Array; zone: Uint32Array; zoneXY: Float32Array }
  /** `mask`: bit q set for each period included; null sums: no network, or not `version` */
  | { kind: "subSums"; id: number; city: CityId; version: number; mask: number; sums: Float32Array | null; everyday: Float32Array | null }
  /** `everyday`: per purpose kind 30 timing values, m, then m far cells, their rail, walk and
   * drive legs (`EverydayApi.selection_sparse_day`, T-285) */
  | { kind: "purposes"; id: number; city: CityId; version: number; mask: number; work: Float32Array | null; everyday: Float32Array | null; names: string[] }
  /** eight blocks of one value per cell (home rail, walk, drive, work rail, walk, drive, then No route at
   * home and at work, T-2215: demand.worker.ts `withNone`) */
  | { kind: "cellModes"; id: number; city: CityId; version: number; modes: Float32Array | null }
  | { kind: "station"; id: number; city: CityId; version: number; mask: number; packed: Float32Array | null }
  | { kind: "flowRail"; id: number; city: CityId; version: number; mask: number; sums: Float32Array | null }
  /** `[rail, walk, drive, cut rail, cut walk, cut drive, n, n cells.., rail.., walk.., drive..]` */
  | { kind: "flows"; id: number; city: CityId; version: number; packed: Float32Array | null }
  /** `mask` as `subSums`; `packed` as `IntercityApi.selection_flows` (null: no network, or not `version`) */
  | { kind: "icSel"; id: number; city: CityId; version: number; mask: number; packed: Float64Array | null }
  /** `packed` as `IntercityApi.station_flows` */
  | { kind: "icStation"; id: number; city: CityId; version: number; mask: number; packed: Float64Array | null }
  /** eight blocks of one value per cell as `cellModes`, trip legs; empty when the city has no such
   * purpose; null: no network, or not `version` and `round` */
  | { kind: "purposeModes"; id: number; city: CityId; version: number; mask: number; modes: Float32Array | null }
  /** triples (route node arriving, route node leaving, riders), each pair once, summed over this
   * worker's periods; null: no network, or not `version` and `round` */
  | { kind: "transfers"; id: number; city: CityId; version: number; mask: number; pairs: Float32Array | null }
  /** one `DemandApi.trip` answer per period in `periods` (the cells' x, y turned into world
   * metres); null: no network, or not `version` and `round` */
  | { kind: "trip"; id: number; city: CityId; version: number; mask: number; periods: number[]; answers: Float64Array[] | null }
  /** T-866: one `IntercityApi.market_trip` answer per period in `periods`, from the market part
   * with the most rail (its route nodes the whole network's); null as `trip` */
  | { kind: "icTrip"; id: number; city: CityId; version: number; mask: number; periods: number[]; answers: Float64Array[] | null }
  /** T-325: `commute` as `cellModes`'s eight blocks (commuters, not legs); `parts` one block set per
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
  | { kind: "bareSquares"; id: number; city: CityId; xy: Float32Array | null; share: Float32Array | null }
  /** T-1435: each table sparse (`[m, pair.., legs..]`); null: no network, or not `version` and
   * `round`; `everyday` null for a zone without the layer */
  | { kind: "driveRail"; id: number; city: CityId; version: number; mask: number; commute: Float32Array | null; everyday: Float32Array | null }
  /** T-1435: `squares` as `DriveSquares::packed` with x and y as longitude and latitude; `base` the part's
   * table; both null when the zone (or, for part 1, its everyday layer) is not held here */
  | { kind: "driveBase"; id: number; city: CityId; squares: Float64Array | null; base: Float32Array | null }
  /** T-2365: eight blocks of `trip_length_edges` bins each (trips, rail, walking, no way, then their
   * km); null: not `version` and `round` (or not held here); `everyday` null for a zone without the layer */
  | { kind: "tripLengths"; id: number; city: CityId; version: number; mask: number; edges: Float32Array; commute: Float64Array | null; everyday: Float64Array | null };
