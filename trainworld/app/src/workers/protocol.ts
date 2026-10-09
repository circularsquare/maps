// Messages between the main thread and the workers (SPEC 6.5, 7).
//
// Rules: workers own their data. The main thread sends edits and settings, never waits, and
// swaps in the newest snapshot (a version number on every result). Big arrays travel as
// transferred buffers.
//
// Coordinates in edits are pack metres (east and north of the city pack's origin, the track
// model's frame); what the renderer draws comes back in local units (map/geo.ts). Money in
// messages is US$M, as in the track model.

import type { EdgeId, Level, NodeId } from "../game/types";
import type { DayLedger } from "./saveFile";
import type { NetworkBuffers } from "../map/network";

export type { DayLedger };

// ---------- clock worker: owns the network, run times, capacity, keyframes (T-025) ----------

/** A point of intersection (PI), pack metres: a point the player clicked, where the track turns
 * with a circular curve that cuts the corner (SPEC 6.1, T-092). */
export interface PiInput {
  x: number;
  y: number;
  level: Level;
  /** player-set radius in m; auto when absent (SPEC 6.1) */
  radius?: number;
}

/** Where a drawn route starts or ends: a new node, an existing node, or splitting an edge. */
export type RouteEnd =
  | { kind: "free"; x: number; y: number; level: Level }
  | { kind: "node"; node: NodeId }
  | { kind: "edge"; edge: EdgeId; x: number; y: number };

/** Where a station goes: on a node, or splitting an edge near a point. */
export type StationAt = { kind: "node"; node: NodeId } | { kind: "edge"; edge: EdgeId; x: number; y: number };

/** One edit. Blueprint edits are undoable; construct and removing constructed things are final
 * (SPEC 6.4). Line ids are the track model's numbers. */
export type EditOp =
  /** a route turning at the clicked points `pis`; a junction it makes is flat (T-080) */
  | { op: "route"; from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean }
  | { op: "removeEdge"; edge: EdgeId }
  /** reshape blueprint track (T-064): its PIs replaced (moved, or a radius set), same ends */
  | { op: "edgePis"; edge: EdgeId; pis: PiInput[] }
  /** move a blueprint node (T-093) to pack metres x, y; the blueprint track there follows */
  | { op: "moveNode"; node: NodeId; x: number; y: number }
  /** a whole blueprint stretch on one level, shape kept */
  | { op: "edgeLevel"; edge: EdgeId; level: Level }
  | { op: "flying"; node: NodeId; flying: boolean }
  | { op: "addStation"; at: StationAt; platform: number; name: string }
  | { op: "stationProps"; node: NodeId; platform: number; name: string }
  | { op: "removeStation"; node: NodeId }
  | { op: "construct"; edges: EdgeId[]; stations: NodeId[] }
  | { op: "constructAll" }
  | { op: "constructLine"; line: number }
  | { op: "addLine"; stops: NodeId[]; name: string; colour: string }
  | { op: "setStops"; line: number; stops: NodeId[] }
  | { op: "setFrequency"; line: number; tph: [number, number, number] }
  | { op: "lineLook"; line: number; name: string; colour: string }
  | { op: "removeLine"; line: number }
  /** the fare curve (T-028): US$ a ride plus US$ a km; not undoable, a setting */
  | { op: "fares"; base: number; perKm: number }
  | { op: "undo" }
  | { op: "redo" };

export type ToClock =
  /** `pack`: the city pack's URL without extension (".../packs/nyc"), absolute. `fresh`: start a
   * new game even if there is an autosave (`?new=1`). */
  | { kind: "init"; city: string; pack: string; fresh?: boolean }
  | { kind: "edit"; seq: number; op: EditOp }
  /** what a route would cost and whether it can be built; answered with "preview" */
  | { kind: "preview"; seq: number; from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean }
  /** what giving blueprint track new PIs would do (T-064); answered with "preview" */
  | { kind: "previewEdge"; seq: number; edge: EdgeId; pis: PiInput[] }
  /** what moving a blueprint node would do (T-093); answered with "preview", one part per edge */
  | { kind: "previewNode"; seq: number; node: NodeId; x: number; y: number }
  /** the render epoch the main thread is in (start of the game hour, s): send its trips and the
   * next hour's */
  | { kind: "epoch"; epoch: number }
  /** the game clock (main thread, game/clock.ts) whenever it is set, paused, resumed or changes
   * speed: game seconds `t` at wall time `at` (Date.now()), moving `rate` game s per wall s (0
   * when stopped). The worker settles money by the game hour from it (T-028). */
  | { kind: "clock"; t: number; rate: number; at: number }
  /** fare income in force, US$M per game hour for each of the five periods (from demand riders,
   * game/money.ts) */
  | { kind: "income"; perHour: number[] }
  /** write the autosave now if anything changed (the page is being hidden or closed) */
  | { kind: "flush" }
  /** the game as a file: answered with "exported" */
  | { kind: "export" }
  /** load a save file (T-029); answered with "loaded" */
  | { kind: "import"; bytes: ArrayBuffer }
  /** start again: empty city, $6B, day 1 07:00; answered with "loaded" */
  | { kind: "newGame" }
  /** inspector queries (T-031), answered with "answer": a junction's moves
   * (`TrackApi.junction_info`), where a line waits at a demand level (`line_delays`), a blueprint
   * stretch's PIs with their curves (`edge_pis`, T-064), or a flyover's price (`[US$M]`, T-080) */
  | {
      kind: "query";
      seq: number;
      q: { what: "junction"; node: number } | { what: "lineDelays"; line: number; level: number } | { what: "edgePis"; edge: number } | { what: "flyover"; node: number };
    };

/** Money as the worker keeps it (T-028), US$M unless said. */
export interface MoneyState {
  cash: number;
  /** cars owned, and the cars the running lines need */
  fleet: number;
  carsNeeded: number;
  /** US$M per car, US$ per car-km (the real figure) (sim/src/track/params.rs) */
  carPrice: number;
  runCostCarKm: number;
  /** fares and running costs count this many times over in a game day (T-081, `ECONOMY`) */
  economy: number;
  fares: { base: number; perKm: number };
  /** running cost per game hour at high, medium, low demand */
  runningPerHour: [number, number, number];
  /** recent game days, oldest first; the last is today so far */
  ledger: DayLedger[];
}

export interface ClockEdge {
  id: number;
  a: number;
  b: number;
  tracks: 1 | 2;
  built: boolean;
  lengthM: number;
  /** US$M */
  cost: number;
  vminKmh: number;
  levelMin: number;
  levelMax: number;
}

export interface ClockNode {
  id: number;
  x: number;
  y: number;
  lng: number;
  lat: number;
  level: number;
  ports: number;
  builtPorts: number;
  flying: boolean;
  platform: number;
  stationBuilt: boolean;
  name: string;
  /** track heading through it, rad, counter-clockwise from east */
  heading: number;
}

export interface ClockLine {
  id: number;
  name: string;
  colour: string;
  ok: boolean;
  running: boolean;
  broken: boolean;
  tph: [number, number, number];
  cars: number;
  lengthM: number;
  roundTrip: [number, number, number];
  trains: [number, number, number];
  delay: [number, number, number];
  dwell: number;
  turnaround: number;
  stops: number[];
  /** `edge << 1 | direction`, first stop to last */
  path: number[];
  /** run 0, per demand level: arrival at each stop from the start, s */
  fromStart: number[][];
  stopS: [number[], number[]];
  /** US$M to construct what it still lacks */
  buildCost: number;
  /** US$M for the trains it will need once constructed, after spare cars (T-028) */
  trainCost: number;
}

/** Everything drawn, in local units (map/geo.ts). */
export interface ClockRender {
  /** per edge in `ClockState.edges` order: x, y, height (m) triples, concatenated */
  edgePts: Float32Array;
  /** start of each edge's points (in triples), edges + 1 entries */
  edgeOff: Uint32Array;
  /** keyframes (notes/T-040.md): phases (t, s, v, a), per profile (first, count, trip s, length
   * m), run 0 of each line resampled (x, y, local units), per line id (first sample, count,
   * length m, step m) */
  phases: Float32Array;
  meta: Float32Array;
  samples: Float32Array;
  lineTable: Float32Array;
}

/** A busy resource (SPEC 6.2: 75% and up), local units. `kind`: 0 track, 1 single track,
 * 2 platform, 3 terminus, 4 junction, 5 flat crossing (`ResKind` in sim/src/track/capacity.rs). */
export interface CapacityMarker {
  x: number;
  y: number;
  rho: number;
  delayS: number;
  kind: number;
}

export interface ClockState {
  version: number;
  /** busy resources per demand level (high, medium, low) */
  markers: CapacityMarker[][];
  /** US$M */
  cash: number;
  builtCost: number;
  blueprintCost: number;
  canUndo: boolean;
  canRedo: boolean;
  edges: ClockEdge[];
  nodes: ClockNode[];
  lines: ClockLine[];
  render: ClockRender;
  money: MoneyState;
  /** the renderer's buffers, built in the worker from the rest (T-063); the arrays are shared with
   * `render` */
  net?: NetworkBuffers;
  /** worker ms: building the state, building `net` */
  ms?: number[];
}

export type FromClock =
  | { kind: "ready"; initMs: number; origin: { lon: number; lat: number }; water: boolean; population: number; jobs: number }
  /** an edit's answer; `issues` are the track model's problem kinds (`TrackApi.issue_kinds`) */
  | { kind: "edited"; seq: number; ok: boolean; issues: string[]; charge: number; line?: number }
  | {
      kind: "preview";
      seq: number;
      ok: boolean;
      /** US$M */
      cost: number;
      lengthM: number;
      /** the track's price over its length at the base price (level and water averaged, T-085) */
      mult: number;
      issues: string[];
      /** per vertex: x, y (local units; the middle of its curve), radius m (0 = no arc), speed
       * limit km/h */
      verts: Float64Array;
      /** x, y (local units), height m */
      pts: Float32Array;
      /** several alignments (a node drag moves every edge there, T-093): where each starts in
       * `pts`, in triples, and the end; absent = one */
      parts?: Uint32Array;
    }
  | { kind: "state"; state: ClockState }
  /** trips drawn in [epoch, epoch + 3600): profile index, departure - epoch (notes/T-040.md) */
  | { kind: "trips"; version: number; epoch: number; trips: Float32Array }
  | { kind: "snapshot"; snapshot: NetworkSnapshot }
  /** money changed without a network change (an hour settled, fares set) */
  | { kind: "money"; money: MoneyState }
  /** a game was loaded or started (`from`), at game time `clock`; a failed import says why in
   * `error` and changes nothing */
  | { kind: "loaded"; from: "autosave" | "file" | "new"; clock: number; error?: string }
  /** the save file, for the player to keep */
  | { kind: "exported"; bytes: Uint8Array; day: number }
  /** a query's answer, in the track model's flat layout (pack metres) */
  | { kind: "answer"; seq: number; data: Float64Array }
  /** an autosave was written (Date.now()), its size, and worker ms to encode and to write */
  | { kind: "saved"; at: number; bytes: number; encodeMs: number; totalMs: number }
  | { kind: "error"; message: string };

// ---------- the network as demand sees it (T-025 -> T-026) ----------

/**
 * The service demand runs on: what a passenger can ride. Built by the clock worker after every
 * service change (SPEC 6.5: a line's stops or frequency, a line starting or stopping running, or
 * a station-to-station time moving 5 s or more), debounced ~0.5 s, and delivered on the main
 * thread through `onNetworkSnapshot` in workers/clockClient.ts (the newest one is also in the
 * `networkSnapshot` signal there). notes/T-025.md.
 *
 * Only what carries trains is in it: constructed stations, and lines whose every edge and stop
 * is constructed (a line planned over blueprint track is left out until it runs).
 */
export interface NetworkSnapshot {
  /** increases with every snapshot; a newer one replaces an older one */
  version: number;
  city: string;
  stations: SnapshotStation[];
  lines: SnapshotLine[];
  /** the player's fare curve (T-028): US$ a ride plus US$ a km of the ride. Demand does not read
   * it yet (its fare is a fixed 6 perceived minutes); a change does not send a new snapshot. */
  fare?: { base: number; perKm: number };
}

export interface SnapshotStation {
  /** the track model's node id; stable while the station exists */
  id: number;
  name: string;
  /** metres east and north of the city pack's origin (the pack's own frame, notes/T-004.md) */
  x: number;
  y: number;
  level: Level;
  /** platform length, m */
  platform: number;
}

export interface SnapshotLine {
  /** the track model's line id; stable while the line exists */
  id: number;
  name: string;
  /** station ids in order, first to last (run 0); run 1 is the same stops reversed */
  stops: number[];
  /** trains an hour at high, medium, low demand (SPEC 6.3; periods in sim/src/track/params.rs) */
  tph: [number, number, number];
  /**
   * Per run (0 = stops in order, 1 = reversed), per demand level (0 high, 1 medium, 2 low):
   * seconds from leaving stop i to arriving at stop i + 1, capacity holds on the way included.
   * `hopS[run][level].length === stops.length - 1`; run 1's hops are in its own travel order.
   */
  hopS: number[][][];
  /** dwell at each stop, s, per run and level, in the run's own stop order (0 at both ends) */
  dwellS: number[][][];
  /** layover at each end, s */
  turnaroundS: number;
  /** cars per train (20 m each) */
  cars: number;
}

// ---------- demand workers: a small pool per city (T-026), later one long-distance ----------
// Their messages are in workers/demandProtocol.ts; the main thread hands them each
// NetworkSnapshot through `submitNetwork` in workers/demandClient.ts. notes/T-026.md.

export type { DemandNetwork, FromDemand, ToDemand } from "./demandProtocol";
