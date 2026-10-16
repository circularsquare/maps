// Messages between the main thread and the workers (SPEC 6.5, 7).
//
// Rules: workers own their data. The main thread sends edits and settings, never waits, and
// swaps in the newest snapshot (a version number on every result). Big arrays travel as
// transferred buffers.
//
// Coordinates in edits are world metres (the loaded game's frame, game/frame.ts: the track
// model's frame); what the renderer draws comes back in local units (map/geo.ts). Money in
// messages is US$M, as in the track model.

import type { EdgeId, Level, NodeId } from "../game/types";
import type { FrameSpec } from "../game/frame";
import type { DayLedger, HourLedger, SaveSlot } from "./saveFile";
import type { NetworkBuffers } from "../map/network";

export type { DayLedger, HourLedger };

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
  /** `through` (T-418): a route with no PIs from a heading to open ground is one tangent arc */
  | { op: "route"; from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean }
  | { op: "removeEdge"; edge: EdgeId }
  /** reshape blueprint track (T-064): its PIs replaced (moved, or a radius set), same ends */
  | { op: "edgePis"; edge: EdgeId; pis: PiInput[] }
  /** move a blueprint node (T-093) to pack metres x, y; the blueprint track there follows */
  | { op: "moveNode"; node: NodeId; x: number; y: number }
  /** move several blueprint nodes together by dx, dy pack metres (T-475) */
  | { op: "moveNodes"; nodes: NodeId[]; dx: number; dy: number }
  /** a whole blueprint stretch on one level, shape kept */
  | { op: "edgeLevel"; edge: EdgeId; level: Level }
  | { op: "flying"; node: NodeId; flying: boolean }
  | { op: "addStation"; at: StationAt; platform: number; name: string }
  /** a crossover on track `edge` near x, y and the track running beside it (T-338) */
  | { op: "crossover"; edge: EdgeId; x: number; y: number }
  | { op: "stationProps"; node: NodeId; platform: number; name: string }
  | { op: "removeStation"; node: NodeId }
  | { op: "construct"; edges: EdgeId[]; stations: NodeId[] }
  | { op: "constructAll" }
  /** switch blueprint track and stations off or on (T-333): Construct and the quote skip what is off; one undoable edit */
  | { op: "blueprintOff"; edges: EdgeId[]; stations: NodeId[]; off: boolean }
  | { op: "constructLine"; line: number }
  | { op: "addLine"; stops: NodeId[]; name: string; colour: string }
  | { op: "copyLine"; line: number; name?: string; express?: boolean }
  | { op: "setStops"; line: number; stops: NodeId[] }
  | { op: "setTrainCount"; line: number; trains: [number, number, number] }
  | { op: "lineLook"; line: number; name: string; colour: string; letter?: string }
  | { op: "removeLine"; line: number }
  /** the fare curve (T-028): US$ a ride plus US$ a km; not undoable, a setting */
  | { op: "fares"; base: number; perKm: number }
  /** link two stations into one complex, or undo a link (T-197): node ids; not undoable, a setting */
  | { op: "link"; a: NodeId; b: NodeId; on: boolean }
  | { op: "undo" }
  | { op: "redo" };

export type ToClock =
  /** `pack`: the city pack's URL without extension (".../packs/nyc"), absolute. `fresh`: start a
   * new game even if there is an autosave (`?new=1`). `testPage`: the page is a check or stress
   * page (`?new=1`, `?synth=1`, `?demandTest=1`, T-320): it never writes a slot until the player
   * saves a copy; `fresh` also does not read one. */
  | { kind: "init"; city: string; pack: string; fresh?: boolean; testPage?: boolean; slot?: string }
  /** the view's bounds in degrees after a move (T-302): water masks near it are fetched ahead */
  | { kind: "view"; w: number; s: number; e: number; n: number }
  | { kind: "edit"; seq: number; op: EditOp }
  /** what a route would cost and whether it can be built; answered with "preview" */
  | { kind: "preview"; seq: number; from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean }
  /** what giving blueprint track new PIs would do (T-064); answered with "preview" */
  | { kind: "previewEdge"; seq: number; edge: EdgeId; pis: PiInput[] }
  /** what moving a blueprint node would do (T-093); answered with "preview", one part per edge */
  | { kind: "previewNode"; seq: number; node: NodeId; x: number; y: number }
  /** the same for several nodes moved together (T-475) */
  | { kind: "previewNodes"; seq: number; nodes: NodeId[]; dx: number; dy: number }
  /** whether `addStation` with these arguments would succeed, nothing changed (T-311, the
   * placement ghost's colour); answered with "stationCheck" */
  | { kind: "previewStation"; seq: number; at: StationAt; platform: number }
  /** what the crossover op would do (T-338); answered with "crossoverCheck" */
  | { kind: "previewCrossover"; seq: number; edge: EdgeId; x: number; y: number }
  /** the render epoch the main thread is in (start of the game hour, s): send its trips and the
   * next hour's */
  | { kind: "epoch"; epoch: number }
  /** the draw anchor moved (T-307, map/geo.ts): rebuild every drawn position relative to it and send
   * a full state */
  | { kind: "anchor"; lng: number; lat: number; version: number }
  /** the game clock (main thread, game/clock.ts) whenever it is set, paused, resumed or changes
   * speed: game seconds `t` at wall time `at` (Date.now()), moving `rate` game s per wall s (0
   * when stopped). The worker settles money by the game hour from it (T-028). */
  | { kind: "clock"; t: number; rate: number; at: number }
  /** fare income in force, US$M per game hour for each of the five periods (from demand riders,
   * game/money.ts), one table per kind of day (T-260, game/days.ts `DayKind` order) */
  /** fare income per game hour in each period, per kind of day (US$M); `lines` the same per line
   * (T-293: each line's fares taken today) */
  | { kind: "income"; perHour: number[][]; lines?: { line: number; perHour: number[][] }[] }
  /** demand's riders for the boarding account (T-164): `TrackApi.set_riders`'s packing, built by
   * game/boarding.ts `packRiders` from the demand result, for one kind of day (T-260) */
  | { kind: "riders"; packed: Float32Array; day: number }
  /** write the autosave now if anything changed (the page is being hidden or closed) */
  | { kind: "flush" }
  /** the game as a file: answered with "exported" */
  | { kind: "export" }
  /** load a save file (T-029); answered with "loaded" */
  | { kind: "import"; bytes: ArrayBuffer }
  /** start again: empty city, $6B, day 1 07:00; answered with "loaded" */
  | { kind: "newGame" }
  | { kind: "saveSlot"; action: "refresh" | "create" | "open" | "rename" | "delete"; key?: string; name?: string }
  /** inspector queries (T-031), answered with "answer": a junction's moves
   * (`TrackApi.junction_info`), where a line waits at a demand level (`line_delays`), a blueprint
   * stretch's PIs with their curves (`edge_pis`, T-064), or a flyover's price (`[US$M]`, T-080) */
  | {
      kind: "query";
      seq: number;
      q: { what: "insertStop"; line: number; station: number } | { what: "stopTiers"; line: number } | { what: "junction"; node: number } | { what: "lineDelays"; line: number; level: number } | { what: "edgePis"; edge: number } | { what: "flyover"; node: number } | { what: "lineAccount"; line: number; day: number } | { what: "lineLevels"; line: number; day: number } | { what: "today" } | { what: "yesterday" };
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
  /** recent game days, oldest first; the last is today so far (as of the last whole hour; the
   * "cash" message carries today as fares land, T-284) */
  ledger: DayLedger[];
  /** recent game hours, oldest first, the last this hour so far (T-447) */
  hours: HourLedger[];
}

export interface ClockEdge {
  id: number;
  a: number;
  b: number;
  tracks: 1 | 2;
  built: boolean;
  /** a blueprint switched off (T-333) */
  off: boolean;
  /** one diagonal of a crossover (T-338) */
  crossover: boolean;
  /** the crossover's other diagonal, -1 if none */
  partner: number;
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
  /** a blueprint station switched off (T-333) */
  stationOff: boolean;
  name: string;
  /** track heading through it, rad, counter-clockwise from east */
  heading: number;
}

export interface ClockLine {
  id: number;
  name: string;
  letter: string;
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
  /** seconds its local time runs ahead of the game clock: its first stop's region's (T-309) */
  shift: number;
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
  /** Geometry speed limit km/h per rendered edge segment; indexed as edge vertices. */
  edgeSpeed?: Float32Array;
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
  /** Seven-number cost rows from TrackApi.blueprint_cost_items, amounts in US$M. */
  blueprintItems: number[];
  canUndo: boolean;
  canRedo: boolean;
  edges: ClockEdge[];
  nodes: ClockNode[];
  lines: ClockLine[];
  /** stations the player linked into one complex (T-197): node id pairs, flat */
  links: number[];
  render: ClockRender;
  money: MoneyState;
  /** the renderer's buffers, built in the worker from the rest (T-063); the arrays are shared with
   * `render` */
  net?: NetworkBuffers;
  /** worker ms: building the state, building `net` */
  ms?: number[];
  /** only the service changed (a train count, T-229): no geometry moved, so `net` is not built and
   * the renderer keeps its buffers, taking only the new phase tables and profile meta */
  service?: boolean;
  /** the draw anchor its local units are relative to (T-307, map/geo.ts) */
  anchor?: { lng: number; lat: number; version: number };
}

export type FromClock =
  | { kind: "ready"; initMs: number; frame: FrameSpec }
  /** an edit's answer; `issues` are the track model's problem kinds (`TrackApi.issue_kinds`) */
  /** `ms`: the worker's time on the edit by step (T-229, for `?debug=1`) */
  | { kind: "edited"; seq: number; ok: boolean; issues: string[]; charge: number; line?: number; node?: number; ms?: Record<string, number> }
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
      /** Geometry km/h per consecutive point pair, for a route/edge preview. */
      speeds?: Float32Array;
      /** several alignments (a node drag moves every edge there, T-093): where each starts in
       * `pts`, in triples, and the end; absent = one */
      parts?: Uint32Array;
      /** a tangent route's circle (T-418): centre x, y (local units), ground radius m, 0 */
      arc?: Float64Array;
      /** the draw anchor version `verts` and `pts` are relative to (T-307) */
      anchor?: number;
    }
  | { kind: "state"; state: ClockState }
  /** trips drawn in [epoch, epoch + 3600): profile index, departure - epoch (notes/T-040.md);
   * `ends`, per trip, when it stops being drawn - epoch (its train takes its next departure from
   * the terminal or leaves service, T-206); `nexts`, per trip, the departure - epoch its train
   * takes next (the other run's trip drawn from then), NaN when it leaves service (T-245); `board`, once demand's riders are known: that hour of the boarding account, lined up with
   * `trips` (`TrackApi.boarding_hour`, game/boarding.ts reads it; T-164), the ms it took, the
   * lines worked out again since the last, and the ms spent stepping through them (all, longest) */
  | { kind: "trips"; version: number; epoch: number; trips: Float32Array; ends?: Float32Array; nexts?: Float32Array; board?: Float32Array; boardMs?: number; redone?: number; prepMs?: number; prepMax?: number }
  | { kind: "snapshot"; snapshot: NetworkSnapshot }
  /** money changed without a network change (an hour settled, fares set) */
  | { kind: "money"; money: MoneyState }
  /** T-284: the cash and today's ledger as fares land and running costs drain, several times a wall
   * second while the clock runs; the rest of MoneyState comes with "money" when an hour turns */
  | { kind: "cash"; cash: number; today: MoneyState["ledger"][number]; hour: HourLedger }
  /** a game was loaded or started (`from`), at game time `clock`; a failed import says why in
   * `error` and changes nothing */
  /** `frame`: the game now in the worker; `converted`: its save was in an older frame and was
   * converted to the world's on the way in (T-308) */
  | { kind: "loaded"; from: "autosave" | "file" | "new"; clock: number; frame: FrameSpec; error?: string; converted?: boolean }
  /** the save file, for the player to keep */
  | { kind: "exported"; bytes: Uint8Array; day: number }
  /** a station placement's check (T-311): `issues` as an edit's */
  | { kind: "stationCheck"; seq: number; ok: boolean; issues: string[] }
  /** a crossover preview (T-338): cost US$M, each diagonal's centre line (local x, y, height) */
  | { kind: "crossoverCheck"; seq: number; ok: boolean; issues: string[]; cost: number; parts: Float32Array[] }
  /** a query's answer, in the track model's flat layout (pack metres) */
  | { kind: "answer"; seq: number; data: Float64Array }
  /** an autosave was written (Date.now()), its size, and worker ms to encode and to write */
  | { kind: "saved"; at: number; bytes: number; encodeMs: number; totalMs: number }
  | { kind: "saveSlots"; slots: SaveSlot[]; active: string; writable: boolean; at: number; reason?: string; error?: string }
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
  /** the frame the stations' x, y are in (T-125): the loaded game's */
  frame: FrameSpec;
  stations: SnapshotStation[];
  lines: SnapshotLine[];
  /** The player's fare curve: US$ once per trip plus US$ per km ridden. */
  fare?: { base: number; perKm: number };
}

export interface SnapshotStation {
  /** the track model's node id; stable while the station exists */
  id: number;
  name: string;
  /** world metres east and north of the frame's origin (the snapshot's `frame`, game/frame.ts) */
  x: number;
  y: number;
  level: Level;
  /** platform length, m */
  platform: number;
  /** its station complex among the snapshot's stations (T-161, game/complexes.ts with the
   * player's links): stations with the same number are one complex, and demand lets riders change
   * between any two of them on foot. Absent in snapshots made before T-161: no complexes. */
  complex?: number;
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
  /** Actual track km per segment, in each run's travel order. Legacy snapshots omit it. */
  hopKm?: number[][];
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
