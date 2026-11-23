// Main-thread side of the clock worker (T-025). Never waits on it: requests go out, answers
// arrive as messages. Each "state" becomes the `world` snapshot (game/world.ts) and the
// renderer's buffers; "trips" go to the renderer; network snapshots go to whoever registered
// with `onNetworkSnapshot` (the demand pool, T-026).

import { t, tDate, type MessageKey } from "../i18n";
import { attachClockTrace, editTrace, trace, tracing } from "../game/freezeTrace";
import { effect, signal } from "@preact/signals";
import type { DiagramGraph } from "../diagram/graph";
import { clock } from "../game/clock";
import { activeCity, HOME_CITY, setWorldFrame, viewedZone } from "../game/cities";
import { setDemandFineSink, setGameOpened, setOpenedSink } from "../game/openedZones";
import { setHandOrderSink, takeHandOrder } from "../game/handOrder";
import { setLineShapeSink, takeLineShapes } from "../game/lineShapeState";
import { setNewHeadwaySink, takeNewHeadway } from "../game/newHeadway";
import { bindNewStations } from "../game/stationBinding";
import { setHiddenLinesSink, takeHiddenLines } from "../game/lineVisibility";
import { takeDiagramCities } from "../game/diagramCities";
import { hiddenZones, setGameStart } from "../game/realms";
import { home } from "../game/home";
import { clockStepNow, fitView, holdForGameLoad, loadCannotFinish, markAlive, networkInWithoutTrips, noteClockStep, noteLoad, noteLoadSlot, tripsOff } from "../game/loading";
import { incomeByDay, lastSaved, liveMoney, moneyState, moneyTerms, moneyWorld, setMoneyState } from "../game/money";
import { zonesLoaded } from "../game/zones";
import { chargeShares, lineShares, upkeepShares } from "../game/regionMoney";
import { linePoints } from "../game/points";
import { goalsDue } from "../game/goalRules";
import { AVERAGE_DAY, AWAY_DAY, awaySoon, BACK_DAY, DAY_KINDS, OUT_DAY, rushSoon } from "../game/days";
import { demandByDay, demandView, marketIn, platformTransfers, zoneViews } from "../game/demand";
import { linesPending } from "../game/lineFigures";
import { accountVersion, packArrivals, packRiders } from "../game/boarding";
import { ask, designSpeed, guideFinished, lineDraft, marketOff, selection, tool } from "../game/ui";
import { afterStopEdits, flipSites, rightHand, say, setEditSink, stopsQueued, world, worldIsRows } from "../game/world";
import type { LineInput, LineStats, RowFigs, StationInput, StopOnly, TrackEdge, TrackNode, WorldView } from "../game/types";
import type { ClockLine, ClockState, EditOp, FromClock, MoneyPatch, MoneyState, NetworkSnapshot, PaidFor, PiInput, RouteAsk, StationAt, ToClock } from "./protocol";
import { applyPatch } from "./valuePatch";
import { anchorVersion } from "../map/anchorVersion";
import { mergeNet, type NetworkBuffers } from "../map/network";
import { bandTry } from "../perfFlags";
import { anchor, anchorWanted, EARTH_C, HOME_ANCHOR, makeAnchor, setAnchor, type Anchor } from "../map/geo";
import { loadFailText } from "../game/loadText";
import { noteOpenFailure, openedFine } from "../game/openFailures";
import { decodeSave, isDamagedError, isNewerError, isUncheckedError, readStored, writePending, type GuideSave, type NewGameSpec, type PendingLoad, type SaveSlot } from "./saveFile";
import type { CloudContext } from "./cloud";
import type { FrameSpec } from "../game/frame";
import { dataUrl, TERRAIN_BASE } from "../game/dataBase";
import { packedOutlines } from "../map/coarseOutlines";
import { heapStep } from "./deviceMemory";
import { markDownloaded } from "../game/keepState";
import { saveRefusal } from "../game/memoryCheck";
import { askBeforeLeaving, keepStorage } from "../game/saveSafety";
import { afterPaint, BUSY_MS, expectedMs, hideBusy, learnBusy, showBusy } from "../game/busy";
import { addTerrainGrid, dropTerrainGrids, groundNet, setTerrainIndex } from "../game/terrain";
import { addTownOutline, clearTownOutlines } from "../map/townArea";
import { letterOf } from "../game/lineLetter";

/** Text from the clock worker: it sends message keys (docs/localization.md "Workers"), worded here; anything else as it is. */
/** T-3542: a line of four or five rows' figures (`ClockLine.rows`) as the panels read them. */
function rowFigsOf(r: NonNullable<ClockLine["rows"]>): RowFigs {
  const n = r.set.length, f = (k: number) => Array.from({ length: n }, (_, i) => r.figs[i * 5 + k] ?? 0);
  return { roundTrip: f(0), trains: f(1), delay: f(2), tph: f(3), restTph: f(4), auto: r.auto ?? null, weekend: r.weekend };
}

export function workerText(s: string): string;
export function workerText(s: string | undefined): string | undefined;
export function workerText(s: string | undefined): string | undefined {
  return s && /^(clockWorker|issues|loadText)\.[A-Za-z]+$/.test(s) ? t(s as MessageKey) : s;
}

export const saveSlotsState = signal<Omit<Extract<FromClock, { kind: "saveSlots" }>, "kind">>({ slots: [], active: "autosave", game: 0, writable: true, at: 0, backups: [] });
/** When the game was last saved by hand in this page (T-1251). */
export const savedByHand = signal<number | null>(null);

/** T-1815: the panels' account reads, held while track previews are asked */
const PANEL_READS = new Set(["today", "linesPeriods", "linePeriods", "lineAccount"]);
const PANEL_HOLD_MS = 250;
/** T-2766: the line tool's reads of the track model's lines, which a stop click has changed already */
const STOP_READS = new Set(["stopTiers", "stopGap", "insertStop", "passedStations"]);
/** T-2766: every other read waits until no stop click has been answered for this long, ms (the worker holds a stop
 * click's network as long): the panels ask a dozen after each world, 0.2 to 0.7 s of the worker's on New York's real
 * save, and the next click waited behind them */
const STOP_HOLD_MS = 500;
type ClockQ = Extract<ToClock, { kind: "query" }>["q"];
/** T-3801: the panels' reads, asked again as the clock, the network or the account moves (docs/architecture.md
 * "Workers"). The worker answers in turn, so an edit waits behind every one sent before it: they wait on the page
 * instead, one in the worker at a time, and after each answer the next waits POLL_REST times the worker's own time
 * for it (at most POLL_REST_MAX_MS), so they take at most a third of the worker. A question the page has never had
 * answered (a panel just opened) goes at once. */
const POLL_READS = new Set<ClockQ["what"]>(["today", "linesPeriods", "linePeriods", "lineAccount", "lineHeld", "lineDelays", "lineWaits", "lineMarks",
  "lineTrips", "stationDeps", "lineRelated", "linePairs", "passingSpots", "upgradeCosts", "upgradeCost", "tracksCost", "platformCost", "junction",
  "flyover", "blueprintQuote", "blueprintPieces", "lineCost"]);
const POLL_REST = 2;
const POLL_REST_MAX_MS = 3000;
/** a read unanswered this long, ms, no longer counts as queued (a worker that never answers holds no read for ever) */
const POLL_LOST_MS = 10000;
export function manageSaveSlot(action: Extract<ToClock, { kind: "saveSlot" }>["action"], key?: string, name?: string) {
  // T-1196: a game opened or restored reloads the page into it, T-1197: if it fits
  if (action === "open" && key && key !== saveSlotsState.peek().active) return void ifItFits({ key }, () => switchGame({ slot: key }));
  if (action === "restore" && key) return void ifItFits({ key }, () => switchGame({ load: "restore" }, { kind: "restore", key, from: saveSlotsState.peek().active }));
  // T-1852: deleting the game being played ends it; the page then goes back to the start menu
  if (action === "delete" && key && key === saveSlotsState.peek().active && current?.running) {
    return void current.leave(undefined, toStartMenu, { kind: "saveSlot", action, key });
  }
  // T-3731: a save waits for the stop clicks still being sent, so it holds them
  if (stopsQueued.peek() > 0) return void afterStopEdits().then(() => current?.post({ kind: "saveSlot", action, key, name }));
  current?.post({ kind: "saveSlot", action, key, name });
}

/** T-988: the account the clock worker syncs the slots with; `settle` settles them all again. */
export function postAccount(ctx: CloudContext | null, settle = false) {
  current?.post({ kind: "account", ctx, settle });
}
let accountGone = () => {};
/** T-988: what to do when the worker's upload finds the session's token dead */
export function onAccountGone(f: () => void) { accountGone = f; }
/** Whether the clock worker has started (not on the start menu). */
export const clockRunning = () => !!current?.running;

export const clockWorkerStatus = signal("loading");
/** T-3734: the stamp of the game in hand, `loaded`'s `game` (the worker's load count): per-game page state starts
 * again when it moves (T-2124), and work taken under one stamp is dropped under another (T-3731's stop clicks) */
export const gameStamp = signal(0);
/** The newest network snapshot (what demand runs on); null until the worker has started. */
const networkSnapshot = signal<NetworkSnapshot | null>(null);

export interface EditAnswer {
  ok: boolean;
  issues: string[];
  /** US$M paid */
  charge: number;
  /** the new line's id, for addLine */
  line?: number;
  /** a route's end node, for drawing on from it (T-418) */
  node?: number;
  /** pieces an edgesLevel left out (T-815) */
  skipped?: number;
  /** T-1407: answered before the edit's state, which `world` does not hold yet */
  early?: boolean;
  /** T-2404: an early answer whose state waits for the pointer's rest: the blueprint edges it made
   * or changed, local x, y, z each, for their stand-ins */
  held?: Float32Array[];
}

/** A preview's answer: the worker's "preview" message, but for the sequence number and what only the
 * message's own handling reads. */
export type PreviewAnswer = Omit<Extract<FromClock, { kind: "preview" }>, "kind" | "seq" | "stale" | "anchor">;

/** A crossover preview (T-338): its cost US$M or why not, and its diagonals' centre lines. */
export type CrossoverAnswer = Pick<Extract<FromClock, { kind: "crossoverCheck" }>, "ok" | "issues" | "cost" | "parts">;

const snapshotListeners: ((s: NetworkSnapshot) => void)[] = [];
/** Call `f` with every network snapshot from now on (and the newest one, if there is one). */
export function onNetworkSnapshot(f: (s: NetworkSnapshot) => void) {
  snapshotListeners.push(f);
  if (networkSnapshot.peek()) f(networkSnapshot.peek()!);
}

type StationsAheadSink = (frame: FrameSpec, xy: Float64Array, fine: string[] | null, built: string[][] | null, home: string | undefined) => void;
let stationsAheadSink: StationsAheadSink | null = null;
/** T-889: call `f` with a loading save's built stations, before its first snapshot. */
export function onStationsAhead(f: StationsAheadSink) {
  stationsAheadSink = f;
}
const roadsListeners: (() => void)[] = [];
/** T-1929: call `f` whenever road tiles land in the track model (a ghost priced before them is out of date). */
export function onRoadsLanded(f: () => void) {
  roadsListeners.push(f);
}
let roomSink: (() => void) | null = null;
/** T-3869: call `f` when a save finds no memory (the demand workers are replaced, giving theirs back). */
export function onSaveRoom(f: () => void) {
  roomSink = f;
}
let netAheadSink: ((key: string) => void) | null = null;
/** T-1539: call `f` with a loading save's `saveNetKey`, before its first snapshot. */
export function onNetAhead(f: (key: string) => void) {
  netAheadSink = f;
}


type StateKind = "load" | "redraw" | "network" | "state";
/** T-1200: ms a state takes on the main thread per edge and node, before this page has taken one of
 * its kind; New York's real save unthrottled (notes/T-1200.md). Learnt from then on (game/busy.ts). */
const STATE_MS_GUESS: Record<StateKind, number> = { load: 0.06, redraw: 0.012, network: 0.01, state: 0.006 };

/** The longest a state waits for its trips (T-648, `ClockClient.receive`), ms. */
const HOLD_MS = 250;

/** Flat node pairs (the clock worker's `links`) as station id pairs (T-197). */
function pairsOf(flat: number[]): [string, string][] {
  const out: [string, string][] = [];
  for (let i = 0; i + 1 < flat.length; i += 2) out.push([String(flat[i]), String(flat[i + 1])]);
  return out;
}

/** T-1380: a node's or line's row gives the same view object for as long as the row stays (the worker
 * posts only the rows that changed, T-1029), and the lists are the last ones when no object in them
 * changed: what builds from the stations and lines (complexes, station names) then finds at once that
 * nothing differs. */
const stationViews = new WeakMap<object, StationInput>();
const lineViews = new WeakMap<object, LineInput>();
const statsViews = new WeakMap<object, LineStats>();
let lastStations: StationInput[] = [];
let lastLines: LineInput[] = [];
let lastNodeRows: readonly object[] = [];
let lastNodeMap = new Map<number, TrackNode>();
function sameObjects<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
/** T-2576: a line's distinct edge ids (its path's half-indices), kept per line row */
const lineEdgeIds = new WeakMap<object, number[]>();
let lastEdgeLineRows: readonly ClockState["lines"][number][] = [];
let lastEdgeLines = new Map<number, string[]>();

/** T-3474: a line's whole plan, which still lists the lines on each edge (the line tool reroutes over
 * all of it, not only the part a partly built line runs) */
function planPathOf(l: ClockState["lines"][number]): number[] {
  return l.planPath ?? l.path;
}

/** whether two line rows run over the same edges: the same row, or the same line with the same path */
function samePaths(a: readonly ClockState["lines"][number][], b: readonly ClockState["lines"][number][]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    if (x === y) continue;
    const xb = x.back ?? [], yb = y.back ?? [], xp = planPathOf(x), yp = planPathOf(y);
    if (x.id !== y.id || xp.length !== yp.length || xb.length !== yb.length) return false;
    for (let k = 0; k < xp.length; k++) if (xp[k] !== yp[k]) return false;
    for (let k = 0; k < xb.length; k++) if (xb[k] !== yb[k]) return false;
  }
  return true;
}
let lastEdgeRows: readonly object[] = [];
let lastEdgeViews: TrackEdge[] = [];
/** The UI's track edges from the state's rows and lines (T-2576): an edge row that was in the last state gives
 * its old view while the lines over it are the same, and the lists of lines are kept while every line
 * runs the same path (an edit of a line's trains changes its row, not its path). */
function edgesOf(s: ClockState): TrackEdge[] {
  let edgeLines = lastEdgeLines;
  let linesSame = true;
  if (!sameObjects(s.lines, lastEdgeLineRows)) {
    if (samePaths(s.lines, lastEdgeLineRows)) lastEdgeLineRows = s.lines;
    else {
      linesSame = false;
      edgeLines = new Map<number, string[]>();
      for (const l of s.lines) {
        let ids = lineEdgeIds.get(l);
        // T-3101: a way back that takes other track counts too
        if (!ids) lineEdgeIds.set(l, ids = [...new Set(planPathOf(l).concat(l.back ?? []).map((p) => p >> 1))]);
        const id = String(l.id);
        for (const p of ids) {
          const v = edgeLines.get(p);
          if (v) v.push(id);
          else edgeLines.set(p, [id]);
        }
      }
      lastEdgeLineRows = s.lines;
      lastEdgeLines = edgeLines;
    }
  }
  if (linesSame && s.edges === lastEdgeRows) return lastEdgeViews;
  const none: string[] = [];
  // where the row at i stood in the last state: usually i, one more or less after an edge was
  // removed or added; a row not found there gets a new view
  let shift = 0;
  const lastRows = lastEdgeRows, lastViews = lastEdgeViews;
  const out = s.edges.map((e, i) => {
    let j = i + shift;
    if (lastRows[j] !== e) {
      if (lastRows[j + 1] === e) j = i + ++shift;
      else if (j > 0 && lastRows[j - 1] === e) j = i + --shift;
      else j = -1;
    }
    if (j >= 0 && linesSame) return lastViews[j];
    const ls = edgeLines.get(e.id) ?? none;
    const v = j >= 0 ? lastViews[j] : undefined;
    if (v && sameObjects(v.lines, ls)) return v;
    return {
      id: e.id, a: e.a, b: e.b, tracks: e.tracks, built: e.built, off: e.off, hsr: e.hsr, kind: e.kind || undefined, oneWay: e.oneWay || undefined, join: e.join || undefined, setWay: e.setWay || undefined, autoWay: e.autoWay || undefined, crossover: e.crossover, partner: e.partner, lengthM: e.lengthM, cost: e.cost * 1e6, vminKmh: e.vminKmh, limit: e.limit || undefined, ramp: e.ramp || undefined,
      levelMin: e.levelMin as any, levelMax: e.levelMax as any, lines: ls === none ? [] : ls,
    };
  });
  lastEdgeRows = s.edges;
  lastEdgeViews = out;
  return out;
}
/** The UI's world view from the worker's state. */
function toWorld(s: ClockState): WorldView {
  let stations: StationInput[] = s.nodes
    .filter((n) => n.platform > 0)
    .map((n) => {
      let v = stationViews.get(n);
      if (!v) stationViews.set(n, v = {
        id: String(n.id), num: n.id, name: n.name, lng: n.lng, lat: n.lat, x: n.x, y: n.y, level: n.level as any, length: n.platform, built: n.stationBuilt, off: n.stationOff, upgrades: n.upgrades, heading: n.heading, ...(n.fare ? { fare: n.fare } : {}), ...(n.named ? { named: true as const } : {}), ...(n.own ? { unbound: true as const } : {}), ...(n.hub !== undefined ? { hub: n.hub } : {}), ...(n.line ? { line: n.line } : {}), ...(n.port ? { port: n.port } : {}), ...(n.brtStation ? { brt: true as const } : {}), ...(n.kind === 2 || n.kind === 3 ? { tram: true as const } : {}),
        ...(n.one ? { one: n.one } : {}), ...(n.oneHeading !== undefined ? { oneHeading: n.oneHeading } : {}), ...(n.onePlatform ? { onePlatform: true as const } : {}),
        ...(n.upgradeUpkeep ? { upgradeUpkeep: n.upgradeUpkeep } : {}),
      });
      return v;
    });
  if (sameObjects(stations, lastStations)) stations = lastStations;
  else lastStations = stations;
  // T-3333: a line run on both pairs and its hidden twin keep a row each, so every sum counts both; the twin
  // is marked for the lists to leave out, its line for its panel to add it in
  const twinOfLine = new Map<number, number>(), twins = new Map<string, string>();
  for (const l of s.lines) if (l.twinOf !== undefined) { twinOfLine.set(l.twinOf, l.id); twins.set(String(l.id), String(l.twinOf)); }
  let lines: LineInput[] = s.lines.map((l) => {
    let v = lineViews.get(l);
    if (!v) lineViews.set(l, v = {
      id: String(l.id), num: l.id, letter: l.letter || letterOf(l.name), symbol: l.letter, name: l.name, colour: l.colour, stops: l.stops.map(String),
      ...(l.planStops ? { planStops: l.planStops.map(String), stopStart: l.stopStart ?? 0 } : {}),
      ...(l.group ? { group: l.group, groupName: l.groupName, groupSymbol: l.groupSymbol, groupColour: l.groupColour, ...(l.outer.length ? { outer: l.outer } : {}) } : {}),
      tph: { high: l.tph[0], medium: l.tph[1], low: l.tph[2] },
      tphP: l.tphP,
      ...(l.twinOf !== undefined ? { twinOf: String(l.twinOf) } : {}),
    });
    return v;
  });
  if (sameObjects(lines, lastLines)) lines = lastLines;
  else lastLines = lines;
  const lineStats: Record<string, LineStats> = {};
  for (const l of s.lines) {
    const had = statsViews.get(l), twin = twinOfLine.has(l.id) ? String(twinOfLine.get(l.id)) : undefined;
    if (had && had.twin === twin) {
      lineStats[String(l.id)] = had;
      continue;
    }
    statsViews.set(l, lineStats[String(l.id)] = { ...statsOf(l), ...(twin ? { twin } : {}) });
  }
  function statsOf(l: ClockLine): LineStats {
    return {
      status: l.running ? "running" : l.ok || (!l.broken && l.stops.length < 2) ? "planned" : "broken",
      km: l.lengthM / 1000,
      fromStartS: l.fromStart[0] ?? [],
      fromStartByLevel: l.fromStart,
      roundTripS: l.roundTrip[0],
      roundTripByLevel: l.roundTrip,
      trainsNeeded: l.trains,
      autoS: l.auto,
      delayS: l.delay,
      cars: l.cars,
      carsSet: l.carsSet,
      fast: l.fast,
      tram: l.tram,
      upkeep: l.upkeep * 1e6,
      stretches: Array.from({ length: l.stretchSpeeds.length / 3 }, (_, k) => ({ reached: l.stretchSpeeds[3 * k], limit: l.stretchSpeeds[3 * k + 1], why: l.stretchSpeeds[3 * k + 2] })),
      stopS: l.stopS,
      path: l.path,
      ...(l.back?.length ? { back: l.back } : {}),
      dwellS: l.dwell,
      turnaroundS: l.turnaround,
      shift: l.shift,
      passed: Array.from({ length: l.passed.length >> 1 }, (_, k) => ({ gap: l.passed[k * 2], node: l.passed[k * 2 + 1] })),
      reversals: l.reversals,
      endTurns: l.endTurns,
      oneWayAt: Object.fromEntries(Array.from({ length: l.ways.length >> 1 }, (_, k) => [String(l.ways[2 * k]), l.ways[2 * k + 1]])),
      ...(l.backStops?.length ? { backStops: l.backStops.map(String) } : {}),
      ...(l.oneStops?.length ? { oneStops: l.oneStops.map(String) } : {}),
      ...(l.stopOnly?.length ? { stopOnly: Object.fromEntries(Array.from({ length: l.stopOnly.length >> 1 }, (_, k) => [String(l.stopOnly![2 * k]), l.stopOnly![2 * k + 1] as StopOnly])) } : {}),
      hand: l.hand,
      loop: l.loopKind === 1 ? "circle" : l.loopKind === 2 ? "balloon" : null, // i18n-ignore
      oneWay: l.oneWay,
      ...(l.oneWayByTrack ? { oneWayByTrack: true as const } : {}),
      limit: l.limit,
      ...(l.fareDist ? { fareDist: true as const } : {}),
      ...(l.weekend ? { weekend: l.weekend } : {}),
      ...(l.restTph ? { restTph: l.restTph } : {}),
      ...(l.pairs ? { pairs: l.pairs } : {}),
      ...(l.twinOf !== undefined ? { twinOf: String(l.twinOf) } : {}),
      periods: l.periods,
      periodsOwn: l.periodsOwn,
      hours: l.hours,
      hoursOwn: l.hoursOwn,
      ...(l.rows ? { rows: rowFigsOf(l.rows) } : {}),
      ...(l.bus ? { bus: l.bus } : {}),
    };
  }
  const edges = edgesOf(s);
  // the last map while the rows are the same, so the snap's node grid (map/snapIndex.ts) stays
  let nodes = lastNodeMap;
  if (!sameObjects(s.nodes, lastNodeRows)) {
    nodes = new Map<number, TrackNode>(
      s.nodes.map((n) => [n.id, { id: n.id, x: n.x, y: n.y, lng: n.lng, lat: n.lat, level: n.level as any, ports: n.ports, builtPorts: n.builtPorts, flying: n.flying, platform: n.platform, ...(n.bus ? { bus: true as const } : {}), ...(n.port ? { port: true as const } : {}), ...(n.brtStation ? { brtStation: true as const } : {}), ...(n.kind ? { kind: n.kind } : {}) }]),
    );
    lastNodeRows = s.nodes;
    lastNodeMap = nodes;
  }
  return {
    version: s.version,
    periods: s.periods,
    hours: s.hours,
    capacity: s.markers,
    fullTrack: (s.fullTrack ?? []).map((v) => {
      const m = new Map<number, { rho: number; tph: number }>();
      for (let k = 0; k + 2 < v.length; k += 3) m.set(v[k], { rho: v[k + 1], tph: v[k + 2] });
      return m;
    }),
    save: { cash: s.cash * 1e6, stations, lines, links: pairsOf(s.links ?? []), ...(s.buildings?.length ? { buildings: s.buildings } : {}) },
    lineStats,
    twins,
    edges,
    nodes,
    money: {
      builtValue: s.builtCost * 1e6, blueprintCost: s.blueprintCost * 1e6,
    },
    canUndo: s.canUndo,
    canRedo: s.canRedo,
  };
}

let current: ClockClient | null = null;

/** Download the game as a file (T-029). */
export function exportGame() {
  current?.post({ kind: "export" });
}
/** T-3115: the game in hand's save bytes, for a Community share; null with no game. */
let shareExport: ((bytes: Uint8Array | null) => void) | null = null;
export function exportForShare(): Promise<Uint8Array | null> {
  if (!current || shareExport) return Promise.resolve(null);
  return new Promise((done) => {
    const late = setTimeout(() => shareExport?.(null), 60_000);
    shareExport = (b) => { clearTimeout(late); shareExport = null; done(b); };
    current!.post({ kind: "export", share: true });
  });
}
/** T-3257: view a shared network (bytes `game/sharedSave.ts` checked): the game in hand is saved and the page reloads
 * into the network, held in no slot until the player saves it as `name` (T-3115 opened it as a copy in a new slot);
 * from the start menu, with no game in hand, it waits in storage for the reloaded page. A reason to show when it
 * cannot go, "" once it goes, null when the player chose to stay on a viewed game.
 * T-3851: `trips` false opens it without calculating trips (`?trips=off`), so no memory check: the demand workers
 * are what a big game's memory goes to. `ask` false skips the question before leaving a viewed game (the same
 * network calculated). */
export async function openView(bytes: Uint8Array, name: string, trips = true, ask = true): Promise<string | null> {
  if (trips) {
    const why = await saveRefusal({ bytes });
    if (why && !why.soft) return why.text;
  }
  if (ask && !await leaveViewed()) return null;
  const pending: PendingLoad = { kind: "import", bytes, name, from: saveSlotsState.peek().active, view: true };
  // T-4002: a viewed network opens paused
  const params: Record<string, string> = trips ? { load: "file", paused: "1" } : { load: "file", trips: "off", paused: "1" };
  if (current?.running) { switchNow(params, pending); return ""; }
  if (!await writePending(pending)) return t("clockClient.noStorage");
  location.replace(pageWith(params));
  return "";
}
/** T-3851: the game in hand, opened without calculating trips, opened again with them: a viewed network goes with its
 * edits, a saved game is saved and reopened. A reason to show when it cannot go, "" once it goes. */
export async function calculateTrips(): Promise<string> {
  const v = saveSlotsState.peek().viewing;
  if (!v) {
    const key = saveSlotsState.peek().active;
    if (!key) return t("clockClient.noStorage");
    const why = await saveRefusal({ key });
    if (why && !why.soft) return why.text;
    switchNow({ slot: key });
    return "";
  }
  const bytes = await exportForShare();
  if (!bytes) return t("clockClient.reopenFailed");
  return await openView(bytes, v.name, true, false) ?? "";
}
/** T-3257: before a viewed network (never saved) gives way to another game or the start menu, the player is asked;
 * true to go on. True at once for any other game. */
export function leaveViewed(): Promise<boolean> {
  const v = saveSlotsState.peek().viewing;
  // a network not edited since it opened loses nothing
  if (!v || !current?.running || !world.peek()?.canUndo) return Promise.resolve(true);
  return new Promise((done) => {
    ask.value = { text: t("clockClient.leaveView", { name: v.name }), yes: t("clockClient.leaveViewYes"), no: t("clockClient.leaveViewNo"), run: () => done(true), stay: () => done(false) };
  });
}
/** A save's bytes, in parts (T-3869: put in the file one by one, never joined here), downloaded as
 * "trainsworld-<name>-day-<day>.save". */
function downloadSave(parts: Uint8Array[], name: string, day: number, slot?: string) {
  markDownloaded(slot); // T-1832: the "saved in this browser only" card has nothing more to say
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(parts as BlobPart[], { type: "application/octet-stream" }));
  const named = name.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "").replace(/-+/g, "-").replace(/^-|-$/g, "");
  a.download = `trainsworld-${named || "game"}-day-${day}.save`; // i18n-ignore
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
/** T-1395: a saved game not being played downloaded as its last save in this browser; false if it
 * could not be read. */
export async function exportSlot(slot: SaveSlot): Promise<boolean> {
  const stored = await readStored(slot.key).catch(() => null);
  if (!stored) return false;
  downloadSave([stored.bytes], slot.name, stored.day, slot.key);
  return true;
}
/** Load a save file the player picked, into a new slot named after the file (T-743).
 * `version`: the file is a version of that game (an account backup, T-2037), else a game of its own. */
export async function importGame(file: File, version?: { origin: string; edit: number }) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (await ifItFits({ bytes })) switchGame({ load: "file" }, { kind: "import", bytes, name: file.name.replace(/\.save$/i, ""), from: saveSlotsState.peek().active, ...version });
}
/** T-1612: a save file picked on the start menu, where no game is in hand to reload from: its game is
 * left in storage as after `switchGame`, and the menu's pick is `{ pending: true }`. A reason (to show)
 * when the file cannot open. */
export async function fileChoice(file: File, version?: { origin: string; edit: number }, warn?: boolean): Promise<GameChoice | string | { warn: string }> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  try { await decodeSave(bytes); } catch (err) { return isDamagedError(err) ? t("clockWorker.saveDamaged") : isNewerError(err) ? t("clockWorker.newer") : isUncheckedError(err) ? loadFailText(String(err)) : t("clockClient.notSave"); }
  const why = await saveRefusal({ bytes }, warn);
  if (why) return why.soft ? { warn: why.text } : why.text;
  if (!await writePending({ kind: "import", bytes, name: file.name.replace(/\.save$/i, ""), from: "", ...version })) return t("clockClient.noStorage");
  return { pending: true };
}
/** T-2037: a backup of slot `from` opened from the start menu, as `fileChoice` does a file. */
export async function backupChoice(key: string, from: string): Promise<GameChoice | string> {
  const why = await saveRefusal({ key });
  if (why) return why.text;
  if (!await writePending({ kind: "restore", key, from })) return t("clockClient.noStorage");
  return { pending: true };
}
/** Start a new game in a new slot (T-743), named after its start (T-728) unless `name` is given. T-3478: the
 * game in hand is saved and the page goes to the start menu (`?next=new`), which starts the new game as its
 * own New game does, so a start that fails ends there, never in the game left. */
export function newGame(game?: NewGameSpec, name = "", at?: { center: [number, number]; zoom: number } | null) {
  const spec = game ?? { mode: "normal" };
  switchGame({ next: "new", mode: spec.mode, ...(spec.start ? { city: spec.start } : {}), ...(spec.real ? { real: "1" } : {}), ...(at ? { at: [...at.center, at.zoom].join(",") } : {}) }, { kind: "new", spec, name });
}
/** T-3479: fetch the terrain that failed again now. */
export function retryGround() {
  current?.post({ kind: "retryGround" });
}

/** T-1196: the URL switches that say which game a page opens (ui/StartMenu.tsx `choiceFromUrl`) */
export const CHOICE_PARAMS = ["new", "city", "mode", "synth", "demandTest", "continue", "slot", "load", "real", "next", "at", "trips"];
/** This page's URL with `params` in place of its game switches; none: the start menu (`?paused=1`, a
 * check's paused clock, stays with a game and goes with the menu). */
export function pageWith(params: Record<string, string>): string {
  const u = new URL(location.href);
  for (const k of CHOICE_PARAMS) u.searchParams.delete(k);
  if (!Object.keys(params).length) u.searchParams.delete("paused");
  for (const [k, v] of Object.entries(params)) u.searchParams.set(k, v);
  return u.href;
}
/** T-1196: another game replaces this one by reloading the page into it, which frees everything the
 * last one held (its demand workers' memory never shrinks while they live: 2.5 GB each after all of
 * China). The game in hand is saved first; if it cannot be, the player is told and stays. */
export function switchGame(params: Record<string, string>, pending?: PendingLoad) {
  void leaveViewed().then((go) => { if (go) switchNow(params, pending); });
}
function switchNow(params: Record<string, string>, pending?: PendingLoad) {
  current?.leave(pending, () => location.replace(pageWith(params)));
}
/** The page reloads to the start menu; a page already at that address (it came without a game switch) is reloaded, since replacing it with itself would not be (T-1852). */
function toStartMenu() {
  const to = pageWith({});
  if (to === location.href) location.reload();
  else location.replace(to);
}
/** T-988: the page goes to the start menu, the game in hand saved and let go first; `then` runs just
 * before the reload, and only if the game could be saved. */
export function leaveToMenu(then: () => void) {
  const go = () => { then(); toStartMenu(); };
  if (current) void leaveViewed().then((ok) => { if (ok) current?.leave(undefined, go); });
  else go();
}
/** T-1197: whether the save fits in this device's memory; if not the player is told why. `then` runs if it does. */
async function ifItFits(src: { bytes: Uint8Array } | { key: string }, then?: () => void): Promise<boolean> {
  const why = await saveRefusal(src);
  if (why) say(why.text, true);
  else then?.();
  return !why;
}
/** T-728: make the game a sandbox, for good. */
export function makeSandbox() {
  current?.post({ kind: "sandbox" });
}
/** T-3707: the page's modded marks (game/modded.ts), told to every clock worker this page starts. */
let moddedWhy: string[] = [];
export function postModded(why: string[]) {
  moddedWhy = why;
  current?.post({ kind: "modded", why });
}
/** T-1822: keep the Getting started card's progress in the save. */
export function saveGuide(guide: GuideSave) {
  current?.post({ kind: "guide", guide });
}
/** T-2395: make a place the game's home (a zone id or `WORLD_HOME`), or none. */
export function setHome(place: string | null) {
  if (!current) return;
  home.value = place;
  current.post({ kind: "home", home: place });
}
/** T-2124: achievement steps met (game/achievements.ts); the worker keeps each from the first time. */
export function postAchievements(got: Record<string, Omit<import("./saveFile").Achieved, "at">>) {
  current?.post({ kind: "achieve", got });
}
/** Ask the clock worker an inspector question (T-031); null before the worker is there. */
export function queryClock(q: Extract<ToClock, { kind: "query" }>["q"], soon = false): Promise<Float64Array | null> {
  return current ? current.query(q, soon) : Promise.resolve(null);
}

/** T-3297: every train, tram and bus to the other side; the sites the flip fixed and left (`ToClock` "flipSide"). */
export function flipSide(): Promise<Float64Array | null> {
  return current ? current.flipSide() : Promise.resolve(null);
}

/** Which game the page opens (T-677). */
export interface GameChoice {
  /** `?new=1`: a new game that owns no slot (T-320) */
  fresh?: boolean;
  /** the slot to open; absent, the one opened last */
  slot?: string;
  /** the start menu's New game beside saved games: a new slot of this name */
  newSlot?: string;
  /** a new game: demand opens the viewed zone bare from the start (T-351) */
  newGame?: boolean;
  /** how the new game starts (T-728) */
  game?: NewGameSpec;
  /** T-1196: the game is the one the saves panel left in storage before the page reloaded */
  pending?: boolean;
  /** T-3851: `?trips=off`: the game opens without calculating trips (no demand pool) */
  noTrips?: boolean;
}

/** One request in flight to the worker; a newer one replaces a waiting one. */
class Newest<Q, A extends unknown[]> {
  busy = false;
  waiting: { q: Q; cb: (...a: A) => void } | null = null;
  private cb: ((...a: A) => void) | null = null;
  /** true when `q` can go now; otherwise it waits, in place of an earlier waiting one */
  ask(q: Q, cb: (...a: A) => void): boolean {
    if (this.busy) {
      this.waiting = { q, cb };
      return false;
    }
    this.busy = true;
    this.cb = cb;
    return true;
  }
  /** The answer came: `deliver` hands it to the callback, then the waiting request goes through
   * `go`, also when the callback throws (T-1001). */
  answered(deliver: (cb: ((...a: A) => void) | null) => void, go: (q: Q, cb: (...a: A) => void) => void) {
    this.busy = false;
    try {
      deliver(this.cb);
    } finally {
      const n = this.waiting;
      this.waiting = null;
      if (n) go(n.q, n.cb);
    }
  }
}

/** T-2710: the waits before each new start of a clock worker whose script did not download (as wasmInit.ts's) */
const BOOT_WAITS_MS = [1_000, 4_000];
/** T-4072: how long a clock worker's script may take to come (twice), and its WASM to run once it has, while the
 * page is visible: the script is a few hundred KB and the WASM 2.5 MB, so these allow a very slow connection */
const BOOT_LIMIT_MS = 60_000, READY_LIMIT_MS = 180_000, START_WATCH_MS = 5_000;

export class ClockClient {
  private worker: Worker | null = null;
  private waiting: [ToClock, Transferable[]][] = [];
  private seq = 0;
  private pending = new Map<number, (a: EditAnswer) => void>();
  /** T-3313: the edits and reads not answered yet, and when each went (the worker answers them in order) */
  private sent = new Map<number, { what: string; at: number }>();
  /** T-3313: when the last hour of trips came, and the worker's last error, for a feedback report */
  private tripsAt = NaN;
  private lastTrouble = "";
  /** T-3628: the ground downloads (masks, terrain) that failed, with their errors, at most 5 */
  private groundWhy: string[] = [];
  /** T-3313: one line for a feedback report: the oldest unanswered edit or read and its age, how many wait
   * behind it, the last trips' age, the last error (seconds) */
  diagnosis(): string {
    const now = performance.now(), first = this.sent.values().next().value;
    const ground = groundNet.peek();
    return [
      first ? `oldest unanswered ${first.what} ${Math.round((now - first.at) / 1000)}s, ${this.sent.size} waiting` : "nothing unanswered", // i18n-ignore
      `trips ${Number.isNaN(this.tripsAt) ? "none yet" : Math.round((now - this.tripsAt) / 1000) + "s ago"}`, // i18n-ignore
      `status ${clockWorkerStatus.peek()}`,
      // T-4072: whether the worker started and its script came, and its game load's last step until the game is in
      !this.worker ? "worker not started" : !this.booted ? `worker script not in after ${Math.round((now - this.spawnAt) / 1000)}s${this.respawned ? " (started again)" : ""}` : "", // i18n-ignore
      !world.peek() && clockStepNow() ? `load step ${clockStepNow()}` : "", // i18n-ignore
      ground.pending ? `ground pending ${ground.pending}` : "", // i18n-ignore
      this.groundWhy.length ? `ground failed ${this.groundWhy.join(", ")}` : "", // i18n-ignore
      this.lastTrouble ? `last error ${this.lastTrouble.slice(0, 200)}` : "", // i18n-ignore
    ].filter(Boolean).join("; ");
  }
  /** T-3314: performance.now() when the oldest unanswered edit or read went, or null (game/figuresStopped.ts) */
  unansweredSince(): number | null {
    return this.sent.values().next().value?.at ?? null;
  }
  private stations = new Newest<{ at: StationAt; platform: number }, [ok: boolean, issues: string[], cost: number]>();
  private crossovers = new Newest<{ edge: number; x: number; y: number }, [a: CrossoverAnswer]>();
  private previews = new Newest<(seq: number) => ToClock, [a: PreviewAnswer]>();
  /** whether a preview is in flight (read by checks) */
  private epoch = NaN;
  private sawSlots = false;
  /** the last game load: where it came from and whether its save was converted to the world frame
   * (T-308; read by `?debug=1` checks, the player is told nothing) */
  lastLoad: { from: "autosave" | "file" | "new"; converted: boolean } | null = null;
  /** T-3933: the slot the last "loaded" named ("" for none): noted for a retry once the saves list shows it written */
  private loadedInto = "";
  /** the renderer's hooks, set by main.ts */
  onState: (s: ClockState) => void = () => {};
  /** T-2159: the network diagram's graph, while the diagram is shown (diagram/view.ts) */
  /** T-3409: `cut`: the graphs are of the chosen cities alone; `whole`: the whole network's too, when a share asked */
  onDiagram: (graphs: DiagramGraph[], cut: boolean, whole?: DiagramGraph[]) => void = () => {};
  /** `board`: the boarding account's hour (T-164, game/boarding.ts), once demand has riders */
  onTrips: (epoch: number, version: number, trips: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array, bays?: Float32Array) => void = () => {};
  /** the boarding account's last hours: ms in the worker for the hour, lines worked out again
   * before it and the ms that took in steps (all, longest), bytes; newest last, at most 40
   * (`?debug=1`, notes/T-164.md) */
  boardLog: { ms: number; redone: number; bytes: number; prepMs: number; prepMax: number }[] = [];
  /** the last edits' time in the worker by step, and the main thread's for the state they sent
   * (T-229, `?debug=1`); newest last, at most 40 */
  editLog: { worker?: Record<string, number>; stateMs?: number }[] = [];
  private lastStateMs = 0;
  /** Draw anchors (T-307, map/geo.ts) by version: the one asked for last, and the ones states and
   * previews may still arrive in. `anchor()` switches when the worker's state built at a new one
   * arrives, so the drawn network and every local-unit conversion change together. */
  private anchors = new Map<number, Anchor>([[HOME_ANCHOR.version, HOME_ANCHOR]]);
  private wantedAnchor: Anchor = HOME_ANCHOR;
  /** called after `anchor()` moved, before the state built at it is handed over (main.ts) */
  onAnchor: (a: Anchor) => void = () => {};

  /** The view moved (MapLibre's moveend): ask for a new draw anchor when it has travelled far. */
  followView(lng: number, lat: number, zoom: number) {
    const to = anchorWanted(lng, lat, zoom, this.wantedAnchor);
    if (!to) return;
    const a = makeAnchor(to[0], to[1], this.wantedAnchor.version + 1);
    this.anchors.set(a.version, a);
    for (const v of this.anchors.keys()) if (v < a.version - 8 && v !== anchor().version) this.anchors.delete(v);
    this.wantedAnchor = a;
    this.post({ kind: "anchor", lng: a.lng, lat: a.lat, version: a.version });
  }

  /** The view's bounds after a move (T-302): the worker fetches the water masks near it ahead. */
  followBounds(w: number, s: number, e: number, n: number) {
    this.post({ kind: "view", w, s, e, n, skip: hiddenZones() });
  }

  /** The theme in use (T-543): the worker draws save thumbnails in it. */
  setShade(shade: "light" | "dark") {
    this.post({ kind: "shade", shade });
  }

  /** The anchor of `version`, as the worker reported it. */
  private anchorOf(spec: { lng: number; lat: number; version: number }): Anchor {
    let a = this.anchors.get(spec.version);
    if (!a) this.anchors.set(spec.version, (a = makeAnchor(spec.lng, spec.lat, spec.version)));
    return a;
  }

  /** A preview's positions, relative to anchor `version`, moved to the current anchor's local
   * units in place (an anchor change and a preview crossed). */
  private toCurrentAnchor(version: number | undefined, verts: Float64Array, pts: Float32Array) {
    const from = version === undefined ? undefined : this.anchors.get(version), to = anchor();
    if (!from || from === to) return;
    const dx = (from.merc[0] - to.merc[0]) * EARTH_C, dy = (from.merc[1] - to.merc[1]) * EARTH_C;
    for (let i = 0; i < verts.length; i += 4) { verts[i] += dx; verts[i + 1] += dy; }
    for (let i = 0; i < pts.length; i += 3) { pts[i] += dx; pts[i + 1] += dy; }
  }

  /** Start the worker on the game chosen (T-677: the start menu's pick, or the URL's). Messages
   * posted before this wait and follow the worker's init. */
  start(choice: GameChoice) {
    if (this.worker) return;
    this.spawnWorker();
    this.watchStart();
    noteLoad("clock worker started");
    const q = new URLSearchParams(location.search);
    const fresh = !!choice.fresh;
    // T-320: a check or stress page never writes over a saved game
    this.testPage = fresh || q.get("synth") === "1" || q.get("demandTest") === "1";
    let slot = choice.slot ?? "autosave";
    if (!choice.slot) try { slot = localStorage.getItem("anitabuilder-save-slot") || slot; } catch { /* storage unavailable */ }
    const backupMs = Number(q.get("backupMs")) || undefined;
    const readFails = q.get("debug") === "1" ? Number(q.get("readFails")) || undefined : undefined;
    const maskWaitMs = q.get("debug") === "1" && q.has("maskWaitMs") ? Number(q.get("maskWaitMs")) : undefined;
    const loadPanic = q.get("debug") !== "1" ? undefined : q.get("loadPanic") === "1" ? true : q.get("loadPanic") === "late" ? "late" as const : undefined;
    const room = q.get("debug") === "1" ? q.get("saveRoom") : null;
    const saveRoom = room === "big" || room === "all" ? room : undefined;
    this.patchMode = q.get("patch") ?? "1";
    // (first, before any message posted ahead of the start)
    this.waiting.unshift([{ kind: "init", pack: dataUrl(this.cityId), terrain: TERRAIN_BASE, fresh, testPage: this.testPage, slot, newSlot: choice.newSlot, backupMs, game: choice.game, pending: choice.pending, patch: this.patchMode, guide: !guideFinished.peek(), bandMode: bandTry(), readFails, maskWaitMs, loadPanic, saveRoom, heapStep: heapStep() } satisfies ToClock, choice.game?.real ? [choice.game.real] : []]);
    // T-2802: the page's parse of the outlines, copied, in place of the worker's own fetch and parse
    void packedOutlines().then((packed) => this.post({ kind: "outlines", packed }));
  }

  /** T-2710: the worker's script loaded (its `booted`): until then every message waits here, its
   * transfers whole, so a worker whose script did not download is started again with all of them */
  private booted = false;
  private bootTries = 0;
  /** T-4072: when the worker in hand was made, whether its start was tried again, its WASM running ("ready") */
  private spawnAt = 0;
  private respawned = false;
  private wasmReady = false;
  /** T-4072: the worker's start watched while the page is visible: a script not in after BOOT_LIMIT_MS starts it once
   * more; not in again, or its WASM not running READY_LIMIT_MS after the script, the load fails saying the game's
   * files did not finish downloading (Safari 17.6 players sat 13 minutes on silent workers). `?startMs=N` sets both
   * (checks); `?debug=1&clockStall=boot` drops the worker's `booted` (dev). */
  private watchStart() {
    const q = new URLSearchParams(location.search), n = Number(q.get("startMs"));
    const bootLimit = n > 0 ? n : BOOT_LIMIT_MS, readyLimit = n > 0 ? n : READY_LIMIT_MS;
    let waited = 0, last = performance.now();
    const firstSpawn = this.spawnAt;
    const timer = setInterval(() => {
      const now = performance.now(), step = now - last;
      last = now;
      if (this.wasmReady || world.peek() || clockWorkerStatus.peek().startsWith("failed")) return clearInterval(timer);
      // a hidden or frozen page counts no wait (its downloads may be held)
      if (document.visibilityState !== "visible" || step > 3 * START_WATCH_MS) return;
      waited += step;
      if (!this.booted && waited >= bootLimit) {
        if (!this.respawned) {
          console.warn(`clock worker: its script not in after ${Math.round(waited / 1000)} s; starting it again`); // i18n-ignore
          this.respawned = true;
          waited = 0;
          this.worker?.terminate();
          this.spawnWorker();
          return;
        }
        clearInterval(timer);
        this.failed(`its script did not arrive, started twice, ${Math.round((now - firstSpawn) / 1000)} s`, "timeout"); // i18n-ignore
      } else if (this.booted && waited >= readyLimit) {
        clearInterval(timer);
        this.failed(`its WebAssembly did not start, ${Math.round(waited / 1000)} s after its script`, "timeout"); // i18n-ignore
      }
    }, n > 0 ? Math.min(START_WATCH_MS, Math.max(100, n / 4)) : START_WATCH_MS);
    this.startWaitReset = () => { waited = 0; };
  }
  private startWaitReset = () => {};
  private spawnWorker() {
    const w = new Worker(new URL("./clock.worker.ts", import.meta.url), { type: "module" });
    this.worker = w;
    this.spawnAt = performance.now();
    attachClockTrace(w);
    this.booted = false;
    w.onmessage = (e: MessageEvent<FromClock>) => this.arrive(e.data);
    w.onerror = (e) => {
      if (this.worker !== w) return;
      // a script that did not download: an error with no message, before `booted`
      if (!e.message && !this.booted && this.bootTries < BOOT_WAITS_MS.length) {
        console.warn("clock worker: its script did not download; trying again"); // i18n-ignore
        w.terminate();
        setTimeout(() => { if (this.worker === w) this.spawnWorker(); }, BOOT_WAITS_MS[this.bootTries++]);
        return;
      }
      this.failed(e.message || "its script did not load"); // i18n-ignore
    };
    w.onmessageerror = () => this.failed("a message could not be read"); // i18n-ignore
  }
  private onBooted() {
    // (dev, T-4072: `?debug=1&clockStall=boot` the script never comes; `=ready` it comes and its WASM never starts)
    const stall = import.meta.env.DEV && new URLSearchParams(location.search).get("debug") === "1" ? new URLSearchParams(location.search).get("clockStall") : null;
    if (stall === "boot") return;
    this.booted = true;
    this.startWaitReset();
    noteClockStep("script in"); // i18n-ignore
    if (stall === "ready") return;
    for (const [msg, transfer] of this.waiting) this.worker!.postMessage(msg, transfer);
    this.waiting = [];
  }

  /** T-950: the worker failed; before its first game no game will come, so the card gives way to the
   * failure screen. `storage`: the saved game could not be read; `timeout` (T-4072): its start never finished. */
  private failed(why: string, kind: "error" | "storage" | "timeout" = "error") {
    clockWorkerStatus.value = "failed to load: " + why; // i18n-ignore
    if (world.peek()) return;
    if (kind === "timeout") loadCannotFinish(`${t("clockClient.startTimeout")} (${why})`, true); // i18n-ignore: the reason is for the report
    else loadCannotFinish(t(kind === "storage" ? "clockClient.storageUnreadable" : "clockClient.cannotStart", { why }), true);
  }

  constructor(cityId = HOME_CITY) {
    this.cityId = cityId;
    setEditSink((op, early, game) => this.edit(op, early, game));
    setOpenedSink((zones) => this.post({ kind: "opened", zones }));
    setDemandFineSink((zones, built) => this.post({ kind: "demandFine", zones, built }));
    setHandOrderSink((kind, ids) => this.post({ kind: "listOrder", list: kind === "lines" ? 0 : 1, ids }));
    setLineShapeSink((line, shape) => this.post({ kind: "lineShape", line, shape }));
    setHiddenLinesSink((lines) => this.post({ kind: "hideLines", lines }));
    setNewHeadwaySink((seconds) => this.post({ kind: "newHeadway", seconds }));
    // T-3759: the player's setting for new stations, again on each change (a worker with no game yet drops it,
    // so each load sends it too)
    effect(() => this.post({ kind: "bindNew", on: bindNewStations.value }));
    current = this;
    if (moddedWhy.length) this.post({ kind: "modded", why: moddedWhy });
    if (tripsOff.peek()) this.post({ kind: "moneyFrozen" }); // T-3851
    // The worker settles money from the clock (T-028, T-284): tell it each time the
    // clock is set, paused, resumed or sped up.
    const sendClock = () => this.post({ kind: "clock", t: clock.now(), rate: clock.ticking ? clock.rate() : 0, at: performance.timeOrigin + performance.now(), game: gameStamp.peek() });
    clock.onChange(sendClock);
    sendClock();
    // Fare income from the riders (game/money.ts), whenever riders, fares or lines change.
    // One table per kind of day (T-260): the worker settles each hour by its own day's.
    let lastIncome = "";
    // T-2848: bumped after a rebuild, which cleared the worker's figures for a network that gets no new solve
    const resend = signal(0);
    // T-2773: a line whose riders are being worked out again is paid what it last earned settled, not a
    // half-done answer's (a local solve reads a long line far low): its entries stand in for the new ones
    const settledIncome = new Map<string, { perHour: number[]; rides: number[] }[]>();
    effect(() => {
      void resend.value;
      const inc = incomeByDay.value, pend = linesPending.value;
      const days = inc.map((i) => ({ ...i, perHour: [...i.perHour], rides: [...i.rides], lines: new Map(i.lines) }));
      const live = new Set(days.flatMap((i) => [...i.lines.keys()]));
      for (const id of [...settledIncome.keys()]) if (!live.has(id)) settledIncome.delete(id);
      for (const id of live) {
        const was = settledIncome.get(id);
        if (!pend.has(id) || !was) { settledIncome.set(id, days.map((i) => ({ perHour: i.lines.get(id)?.perHour ?? [], rides: i.lines.get(id)?.rides ?? [] }))); continue; }
        days.forEach((i, d) => {
          const cur = i.lines.get(id), old = was[d];
          if (!cur || !old) return;
          for (let p = 0; p < i.perHour.length; p++) {
            i.perHour[p] += (old.perHour[p] ?? 0) - (cur.perHour[p] ?? 0);
            if (i.rides[p] !== undefined) i.rides[p] += (old.rides[p] ?? 0) - (cur.rides[p] ?? 0);
          }
          i.lines.set(id, { ...cur, perHour: old.perHour, rides: old.rides });
        });
      }
      const per = days.map((i) => i.perHour);
      // T-293: each line's too, for its fares taken today
      const ids = new Set(days.flatMap((i) => [...i.lines.keys()]));
      const lines = [...ids].map((id) => ({
        line: Number(id),
        perHour: days.map((i) => i.lines.get(id)?.perHour ?? (per[0] ?? []).map(() => 0)),
        rides: days.map((i) => i.lines.get(id)?.rides ?? (per[0] ?? []).map(() => 0)),
      }));
      const rides = days.map((i) => i.rides);
      const key = JSON.stringify([per, rides, lines]);
      if (key !== lastIncome) {
        lastIncome = key;
        this.post({ kind: "income", perHour: per, rides, lines });
      }
    });
    // T-531: each line's money by region, whenever lines, riders or zone boundaries change it
    // T-1021: worked out after the state's own frame, in idle time: placing every station and edge
    // midpoint of a big network was 200 ms of a load's first state, and the ledger only needs them by
    // the next hour
    let lastShares = "", sharesSoon = false;
    const postShares = () => {
      sharesSoon = false;
      const lines = [...lineShares.peek()].map(([line, s]) => ({ line, ...s })), upkeep = upkeepShares.peek();
      const key = JSON.stringify([lines, upkeep]);
      if (key !== lastShares) {
        lastShares = key;
        this.post({ kind: "regionShares", lines, upkeep });
      }
    };
    effect(() => {
      // what the two shares read; they are worked out later
      void moneyWorld.value, moneyTerms.value, zonesLoaded.value, demandByDay.value, world.value;
      if (sharesSoon) return;
      sharesSoon = true;
      if (typeof requestIdleCallback === "function") requestIdleCallback(postShares, { timeout: 2000 });
      else setTimeout(postShares, 100);
    });
    // T-684: each line's points by region, whenever riders, lines or zone boundaries change them
    let lastPoints = "";
    effect(() => {
      void resend.value;
      const lines = linePoints.value;
      const key = JSON.stringify(lines);
      if (key !== lastPoints) {
        lastPoints = key;
        this.post({ kind: "points", lines });
      }
    });
    // the view each kind of day's riders were last sent from; the same view is not sent again (T-1209)
    let ridersSent: unknown[] = [];
    // T-715: a load clears the worker's income, shares and points; the next round sends them again
    // even when they match the last game's
    this.forgetSent = () => { lastIncome = lastShares = lastPoints = ""; ridersSent = []; };
    // T-685: a region's star, the first time its rail share on the average day (never the day
    // shown, so switching it earns nothing) reaches the mark in a final result
    effect(() => {
      const m = moneyState.value;
      if (!m) return;
      // (T-3509: the trips within the zone, those between its far-apart parts included, so never
      // from figures before the market's result while the game has one)
      const d = demandByDay.value[AVERAGE_DAY];
      if (!marketIn(d) && !marketOff()) return;
      for (const g of goalsDue(zoneViews(d), m)) this.post({ kind: "star", ...g });
    });
    // Demand's riders for the boarding account (T-164): each new result, every kind of day's
    // (T-260); the trains and platforms of each hour read their own day's.
    effect(() => {
      // T-264, T-1373: a superholiday's days only while one is soon; an empty message clears them,
      // so the account reads the ordinary day's (track/boarding.rs `kind`)
      void resend.value;
      const soon = awaySoon.value, rush = rushSoon.value;
      demandByDay.value.slice(0, DAY_KINDS.length).forEach((d0, day) => {
        const own = day === AWAY_DAY ? soon : day === OUT_DAY || day === BACK_DAY ? rush : null;
        const d = own === false ? { view: null } : d0;
        if (own !== null && !d.view) {
          if (ridersSent[day]) this.post({ kind: "riders", packed: new Float32Array(0), day });
          ridersSent[day] = null;
          return;
        }
        if (d.view && d.view === ridersSent[day]) return;
        ridersSent[day] = d.view;
        const packed = d.view ? packRiders(d.view) : null;
        // T-1255: with the riders who come to the platforms
        const arrive = packed && d.view ? packArrivals(d.view) : null;
        if (packed) this.post({ kind: "riders", packed, day, ...(arrive ? { arrive } : {}) }, arrive ? [packed.buffer, arrive.buffer] : [packed.buffer]);
      });
    });
    // T-554: the changes between platforms of the solve shown, for when changing riders reach
    // their next platform; one question at a time, asked again if the view moved on meanwhile
    let xferAsked: unknown = null, xferBusy = false;
    const askTransfers = () => {
      const dv = demandView.peek();
      if (xferBusy || !dv || dv === xferAsked) return;
      xferBusy = true;
      xferAsked = dv;
      void platformTransfers().catch(() => null).then((packed) => {
        xferBusy = false;
        // a copy: the answer is cached for other askers of the same round
        if (packed) {
          const own = packed.slice();
          this.post({ kind: "transfers", packed: own }, [own.buffer]);
        } else xferAsked = null;
        if (demandView.peek() !== dv) askTransfers();
      });
    };
    effect(() => {
      void resend.value;
      if (demandView.value) askTransfers();
    });
    this.resendFigures = () => { xferAsked = null; resend.value++; };
    // Hidden or closing: write the autosave now (T-029).
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && this.post({ kind: "flush" }));
    addEventListener("pagehide", () => this.post({ kind: "flush" }));
  }

  /** T-2657: write the autosave and wait for its mark (or 1.5 s) before the page reloads */
  async saveBeforeReload(): Promise<void> {
    // T-3731: the stop clicks still being sent first
    await afterStopEdits();
    const from = saveSlotsState.peek().at;
    this.post({ kind: "flush" });
    return new Promise((done) => {
      const stop = effect(() => { if (saveSlotsState.value.at !== from) { stop(); done(); } });
      setTimeout(() => { stop(); done(); }, 1500);
    });
  }

  private cityId: string;
  /** the world before the last state, for the edit that state answers (T-531) */
  private worldBefore: WorldView | null = null;
  /** `?new=1`, `?synth=1` or `?demandTest=1`: no autosave until a copy is saved (T-320) */
  private testPage = false;
  private forgetSent = () => {};
  private resendFigures = () => {};
  /** T-895: the next game loaded was opened by the player, so the camera fits its network (`fitView`) */
  private fitNext = false;
  private fitState = false;
  fitNextGame() { this.fitNext = true; }
  /** the game loading now was opened by the player (T-1539: its built stations fit the camera ahead) */
  get fitPending() { return this.fitNext; }
  /** T-1196: what to do once the worker has saved and let go of the game in hand */
  private onLeft: (() => void) | null = null;
  leave(pending: PendingLoad | undefined, then: () => void, instead?: ToClock) {
    if (this.onLeft) return;
    // a worker that never started has no game to keep
    if (!this.worker) return then();
    this.onLeft = then;
    // T-3731: the game saved on leaving holds the stop clicks still being sent
    if (stopsQueued.peek() > 0) void afterStopEdits().then(() => this.post(instead ?? { kind: "leave", pending }));
    else this.post(instead ?? { kind: "leave", pending });
  }

  get running() { return !!this.worker; }

  private posts = 0;
  private readsOut = new Map<string, { posts: number; p: Promise<Float64Array> }>();
  post(msg: ToClock, transfer: Transferable[] = []) {
    if (tracing() && msg.kind === "edit") trace("clock-post", { seq: msg.seq, edit: editTrace(msg.op) });
    this.posts++;
    if (this.worker && this.booted) this.worker.postMessage(msg, transfer);
    else this.waiting.push([msg, transfer]);
  }

  /** A state waiting for its trips (T-648), and how long it may wait. */
  private held: Extract<FromClock, { kind: "state" }> | null = null;
  private holdTimer = 0;
  /** `?patch=` (T-1029) */
  patchMode = "1";
  /** the last state's edges, nodes and lines, by id and in order, which the worker posts as changes to
   * them (T-648, T-1029) */
  private rows = { edges: new Map<number, ClockState["edges"][number]>(), nodes: new Map<number, ClockState["nodes"][number]>(), lines: new Map<number, ClockState["lines"][number]>() };
  private lastEdges: ClockState["edges"] = [];
  private lastNodes: ClockState["nodes"] = [];
  private lastLines: ClockState["lines"] = [];
  /** the last network posted, as merged (`mergeNet`), and the run profiles posted since */
  private lastNet: NetworkBuffers | null = null;
  private lastService: { phases: Float32Array; meta: Float32Array } | null = null;

  /** T-3803: the last money record posted (as a state's or a "money" message's), and its serial */
  private money: { serial: number; m: MoneyState } | null = null;

  /** The money record back in `money` from the last one and a patch (T-3803); false when the patch was made
   * against a record this page does not hold (the worker is then asked for a whole state). */
  private takeMoney(x: { money?: MoneyState; moneySerial?: number; moneyPatch?: MoneyPatch; checkMoney?: MoneyState }): boolean {
    const p = x.moneyPatch;
    if (p) {
      if (this.money?.serial !== p.base) return false;
      // (a new record each time, as when it came whole)
      x.money = p.patch ? applyPatch(this.money.m, p.patch) : { ...this.money.m };
      this.money = { serial: p.serial, m: x.money };
      delete x.moneyPatch;
    } else if (x.money) this.money = { serial: x.moneySerial ?? NaN, m: x.money };
    if (x.checkMoney) {
      if (JSON.stringify(x.money) !== JSON.stringify(x.checkMoney)) console.error("T-3803 check: the money record rebuilt from changes differs from the whole one");
      delete x.checkMoney;
    }
    return !!x.money;
  }

  /** A state's edges, nodes and lines back in full from the last ones and its `delta` (T-1029);
   * false when they cannot be (the worker is then asked for a whole state). */
  private takeRows(st: ClockState): boolean {
    const d = st.delta;
    const put = <K extends "edges" | "nodes" | "lines">(k: K, last: ClockState[K], delta: { ids: number[] | null; rows: ClockState[K] } | undefined): ClockState[K] | null => {
      const by = this.rows[k] as Map<number, ClockState[K][number]>;
      if (!delta) {
        by.clear();
        for (const r of st[k]) by.set(r.id, r);
        return st[k];
      }
      if (!delta.ids && !delta.rows.length) return last;
      for (const r of delta.rows) by.set(r.id, r);
      const ids = delta.ids ?? last.map((r) => r.id);
      const out = ids.map((id) => by.get(id));
      if (out.some((r) => !r)) return null;
      if (delta.ids) {
        by.clear();
        for (const r of out) by.set(r!.id, r!);
      }
      return out as ClockState[K];
    };
    const edges = put("edges", this.lastEdges, d?.edges), nodes = put("nodes", this.lastNodes, d?.nodes), lines = put("lines", this.lastLines, d?.lines);
    if (!edges || !nodes || !lines) return false;
    st.edges = this.lastEdges = edges;
    st.nodes = this.lastNodes = nodes;
    st.lines = this.lastLines = lines;
    if (st.checkRows) {
      const bad = (["edges", "nodes", "lines"] as const).filter((k) => JSON.stringify(st[k]) !== JSON.stringify(st.checkRows![k]));
      if (bad.length) console.error(`T-1029 check: the ${bad.join(", ")} rebuilt from changes differ from the whole state's`);
      delete st.checkRows;
    }
    delete st.delta;
    return true;
  }

  /** The state's network made whole from the last (`mergeNet`); false when it cannot be. */
  private takeNet(st: ClockState): boolean {
    if (st.service) {
      this.lastService = { phases: st.render.phases, meta: st.render.meta };
      return true;
    }
    const n = st.net;
    if (!n) return true;
    if (!mergeNet(this.lastNet, n, this.lastService)) return false;
    this.lastNet = n;
    this.lastService = null;
    return true;
  }

  private release() {
    window.clearTimeout(this.holdTimer);
    const m = this.held;
    this.held = null;
    if (m) this.takeState(m);
  }

  /** T-1200: messages waiting while the busy tag paints, in order; null when none wait */
  private deferred: FromClock[] | null = null;
  /** the busy tag shown for a state still held for its trips, hidden once it is taken */
  private stateBusy: { token: number; kind: StateKind; units: number } | null = null;

  /** A state expected to freeze the page past BUSY_MS here (a big network, or one redrawn at a new
   * draw anchor) waits a frame with the busy tag up; every message after it waits behind it. */
  private arrive(msg: FromClock) {
    if (msg.kind === "edited") trace("clock-answer", { seq: msg.seq, ok: msg.ok, ms: msg.ms });
    if (msg.kind === "error") trace("clock-error", msg.message);
    if (this.deferred) { this.deferred.push(msg); return; }
    if (msg.kind === "state" && !this.stateBusy) {
      const st = msg.state;
      const units = (st.delta ? st.delta.edges.rows.length + st.delta.nodes.rows.length : st.edges.length + st.nodes.length) + 1;
      const kind = this.stateKind(st);
      if (expectedMs(kind, units, STATE_MS_GUESS[kind]) > BUSY_MS) {
        this.deferred = [msg];
        this.stateBusy = { token: showBusy(t(kind === "load" || kind === "redraw" ? "clockClient.drawing" : "clockClient.updating")), kind, units };
        afterPaint(() => {
          const q = this.deferred ?? [];
          this.deferred = null;
          for (const m of q) this.receive(m);
          // not taken yet: held for its trips, and the tag stays until it is
          if (this.stateBusy && !this.held) { hideBusy(this.stateBusy.token); this.stateBusy = null; }
        });
        return;
      }
      // T-1407: an early-answered edit's state (tens of ms here) after the next paint, so the
      // ghost the track tool asked on the answer shows first
      if (st.ghostFirst) {
        this.deferred = [msg];
        afterPaint(() => {
          const q = this.deferred ?? [];
          this.deferred = null;
          for (const m of q) this.receive(m);
        });
        return;
      }
    }
    this.receive(msg);
  }

  /** a state's kind of work on this thread: a game's first, a redraw at a new draw anchor, a new network, or neither */
  private stateKind(st: ClockState): StateKind {
    return !world.peek() ? "load" : st.anchor && this.anchorOf(st.anchor) !== anchor() ? "redraw" : st.net ? "network" : "state";
  }

  /** T-1407: a paying edit answered ahead of its state, whose money's split is read off that state */
  private earlyShares: { seq: number; line?: number; paid?: PaidFor } | null = null;

  private takeState(msg: Extract<FromClock, { kind: "state" }>) {
    const st = msg.state, sb = this.stateBusy;
    const kind = sb?.kind ?? this.stateKind(st);
    const t0 = performance.now();
    try { this.takeStateNow(msg); } finally {
      learnBusy(kind, sb?.units ?? st.edges.length + st.nodes.length + 1, performance.now() - t0);
      if (sb) { hideBusy(sb.token); this.stateBusy = null; }
    }
  }

  private takeStateNow(msg: Extract<FromClock, { kind: "state" }>) {
    const t0 = performance.now();
    // built at another draw anchor (T-307): switch to it first, so everything rebuilt from
    // this state converts with the anchor its positions are relative to
    const a = msg.state.anchor ? this.anchorOf(msg.state.anchor) : anchor();
    const moved = a !== anchor();
    if (moved) setAnchor(a);
    if (msg.state.net) msg.state.net.anchor = a;
    setMoneyState(msg.state.money!);
    this.worldBefore = world.value;
    worldIsRows.value = !!msg.state.rows;
    if (msg.state.rightHand !== undefined) rightHand.value = msg.state.rightHand;
    world.value = toWorld(msg.state);
    if (world.value && !msg.state.rows) noteLoad("network in the page", { stations: world.value.save.stations.length, lines: world.value.save.lines.length });
    const es = this.earlyShares;
    this.earlyShares = null;
    if (es && world.value) this.post({ kind: "chargeShares", seq: es.seq, ...chargeShares(this.worldBefore, world.value, es.line, es.paid) });
    // T-2766: rows alone leave the renderer's network as it is until the full state
    if (!msg.state.rows) this.onState(msg.state);
    if (!msg.state.rows && this.lastLoad) networkInWithoutTrips(); // T-3851: with trips off, the loaded network is the whole load
    if (moved) {
      anchorVersion.value = a.version; // the demand views rebuild at it (T-317)
      this.onAnchor(a);
    }
    this.lastStateMs = performance.now() - t0;
    if (this.fitState) {
      this.fitState = false;
      fitView.value++;
    }
  }

  private receive(msg: FromClock) {
    const watched = tracing() && (msg.kind === "state" || msg.kind === "trips");
    if (watched) trace("receive-start", { kind: msg.kind });
    const start = performance.now();
    try { return this.receiveMessage(msg); }
    finally { if (watched) trace("receive-end", { kind: msg.kind, ms: performance.now() - start }); }
  }
  private receiveMessage(msg: FromClock) {
    // T-648: the worker posts a state's trips right after it, before anything else. The state waits
    // for them, so its world, network and trips are taken in one task: no frame draws the old trips
    // on the new run profiles (trains jumped, or stayed a frame on a line that had lost its service),
    // and the demand views build once. Not when the draw anchor moves (its listeners need the network
    // at once), and never past HOLD_MS.
    if (this.held && !(msg.kind === "trips" && msg.version < this.held.state.version) && msg.kind !== "cash" && msg.kind !== "loadStep") this.release();
    // T-950: the steps toward a loading card's game (not the money ticks, which come regardless)
    if (msg.kind === "ready" || msg.kind === "loaded" || msg.kind === "state" || msg.kind === "snapshot" || msg.kind === "stationsAhead" || msg.kind === "loadStep") markAlive();
    switch (msg.kind) {
      case "booted":
        this.onBooted();
        break;
      case "ready":
        clockWorkerStatus.value = `ready in ${msg.initMs.toFixed(0)} ms`; // i18n-ignore
        this.wasmReady = true;
        noteClockStep("track model ready"); // i18n-ignore
        setWorldFrame(msg.frame);
        break;
      case "loadStep":
        noteClockStep(msg.step);
        break;
      case "error":
        console.warn("clock worker:", msg.message);
        this.lastTrouble = msg.message;
        break;
      case "recovered":
        console.error("clock worker:", msg.detail);
        this.lastTrouble = msg.detail;
        if (msg.down) clockWorkerStatus.value = "track model down after a panic"; // i18n-ignore
        say(workerText(msg.notice), true);
        break;
      case "refused":
        say(workerText(msg.notice), true);
        break;
      case "failed":
        console.error("clock worker:", msg.error);
        this.failed(msg.error, msg.storage ? "storage" : "error");
        break;
      case "state": {
        const st = msg.state;
        // T-1029: a state that cannot be made whole is dropped, and the worker posts the next whole
        if (!this.takeMoney(st) || !this.takeRows(st) || !this.takeNet(st)) {
          console.error("T-1029: a state could not be taken as changes to the last; asking for it whole");
          this.lastNet = null;
          this.post({ kind: "wholeState" });
          break;
        }
        const a = st.anchor ? this.anchorOf(st.anchor) : anchor();
        if (a === anchor() && world.peek() && !st.noTrips) {
          this.held = msg;
          this.holdTimer = window.setTimeout(() => this.release(), HOLD_MS);
        } else this.takeState(msg);
        break;
      }
      case "money":
        if (!this.takeMoney(msg)) {
          console.error("T-3803: a money record could not be taken as changes to the last; asking for a whole state");
          this.post({ kind: "wholeState" });
          break;
        }
        setMoneyState(msg.money!);
        break;
      case "diagramGraph":
        this.onDiagram(msg.graphs, !!msg.cut, msg.whole);
        break;
      case "cash": // T-284: fares landing, running costs draining
        liveMoney.value = { cash: msg.cash, today: msg.today, hour: msg.hour, regions: msg.regions };
        break;
      case "opened":
        // T-975: a save older than its start zone shows the realm it opens in
        setGameStart(msg.start ?? (msg.zones === null && viewedZone.peek() ? activeCity.peek() : null));
        setGameOpened(msg.zones, viewedZone.peek() ? activeCity.peek() : null);
        break;
      case "listOrder":
        takeHandOrder(msg.lines, msg.stations);
        break;
      case "lineShapes":
        takeLineShapes(msg.shapes);
        break;
      case "newHeadway":
        takeNewHeadway(msg.seconds);
        break;
      case "hiddenLines":
        takeHiddenLines(msg.lines);
        break;
      case "diagramCities":
        takeDiagramCities(msg.cities);
        break;
      case "stationsAhead":
        stationsAheadSink?.(msg.frame, msg.xy, msg.fine, msg.built, msg.home);
        break;
      case "home":
        home.value = msg.home;
        break;
      case "netAhead":
        netAheadSink?.(msg.key);
        break;
      case "terrain":
        addTerrainGrid(msg.key, msg.bytes);
        break;
      case "terrainDrop":
        dropTerrainGrids(msg.keys);
        break;
      case "terrainIndex":
        setTerrainIndex(msg.bytes);
        break;
      case "town":
        addTownOutline(msg.id, msg.outline);
        break;
      case "townClear":
        clearTownOutlines();
        break;
      case "ground":
        groundNet.value = { pending: msg.pending, failed: msg.failed };
        this.groundWhy = msg.why;
        break;
      case "roads":
        for (const f of roadsListeners) f();
        break;
      case "loaded":
        // the game now in the worker may be in the other frame (T-125); first, before anything reads positions
        setWorldFrame(msg.frame);
        this.post({ kind: "bindNew", on: bindNewStations.peek() });
        this.lastLoad = { from: msg.from, converted: !!msg.converted };
        noteLoad(msg.error ? "clock worker: game did not open" : `game open in clock worker (${msg.from})`);
        this.loadedInto = msg.error ? "" : msg.slot ?? "";
        this.fitState = this.fitNext && msg.from !== "new" && !msg.error;
        this.fitNext = false;
        if (msg.error) {
          // T-2820: a panic's message goes in the notice and the console
          if (msg.why) console.error("clock worker: a game did not open:", msg.why); // i18n-ignore
          say(msg.why ? t(msg.error as MessageKey, { why: msg.why }) : workerText(msg.error), true);
          if (msg.from === "file") break;
        }
        // a different game: its clock waits for its demand (T-149); the snapshot follows
        holdForGameLoad(msg.from === "new" ? "new" : msg.slot || saveSlotsState.peek().active || undefined);
        // Ids are renumbered by a load: nothing old may stay selected or half drawn.
        selection.value = null;
        // T-3297: the last side switch's list is this game's only (a switch comes back as "rebuilt")
        if (msg.reason !== "rebuilt") flipSites.value = null;
        tool.value = "select";
        lineDraft.value = { line: null, stops: [] };
        this.forgetSent();
        if (msg.reason === "rebuilt") this.resendFigures(); // T-2848: only here; a real load would send the last game's
        // T-715: the clock the hold above sent is the old game's, which the worker ignores
        gameStamp.value = msg.game;
        clock.set(msg.clock);
        if (msg.from === "file") say(t("clockClient.loaded", { day: Math.floor(msg.clock / 86400) }));
        if (msg.from === "new" && !msg.error) say(t("clockClient.newGame"));
        if (!msg.error && msg.from === "autosave") openedFine(msg.slot);
        break;
      case "openFailed":
        // T-3773: the worker holds no slot now; the start menu says why, and the saved game is as it was
        console.warn("clock worker: a game did not open, back to the start menu:", msg.why ?? msg.error); // i18n-ignore (the worker logged the error)
        noteOpenFailure({ from: msg.from, slot: msg.slot, name: msg.name, error: msg.error, why: msg.why });
        toStartMenu();
        break;
      case "left": {
        const then = this.onLeft;
        this.onLeft = null;
        if (msg.error) say(workerText(msg.error), true);
        else { askBeforeLeaving(false); then?.(); }
        break;
      }
      case "unsaved":
        askBeforeLeaving(msg.on);
        break;
      case "exported": {
        if (msg.share) { shareExport?.(msg.bytes ?? null); break; }
        if (msg.error) { say(workerText(msg.error), true); break; }
        const { slots, active, viewing } = saveSlotsState.value;
        downloadSave(msg.parts ?? (msg.bytes ? [msg.bytes] : []), viewing?.name || slots.find((s) => s.key === active)?.name || this.cityId, msg.day, active);
        break;
      }
      case "room":
        roomSink?.();
        this.post({ kind: "roomMade" });
        break;
      case "saveSlots": {
        const before = saveSlotsState.value;
        saveSlotsState.value = { slots: msg.slots, active: msg.active, game: msg.game, writable: msg.writable, at: msg.at, reason: workerText(msg.reason), error: workerText(msg.error), backups: msg.backups, viewing: msg.viewing };
        lastSaved.value = msg.at;
        // T-3933: the slot a load that dies is opened again from: the one the game went into, once it is written
        if (this.loadedInto && msg.active === this.loadedInto && msg.slots.some((s) => s.key === msg.active && !s.unreadable)) noteLoadSlot(msg.active);
        if (msg.at > 0 && !this.testPage) void keepStorage(); // T-1214: a slot is written
        // a test page does not make its slot the one the next page opens, until it changes (T-320)
        if (msg.active && ((!this.sawSlots && !this.testPage) || before.active !== msg.active)) {
          try { localStorage.setItem("anitabuilder-save-slot", msg.active); } catch { /* storage unavailable */ }
        }
        this.sawSlots = true;
        // the reason a slot is not written stays in the top bar (ui/TopBar.tsx SaveNotice), not here
        if (msg.error) say(workerText(msg.error), true);
        break;
      }
      case "saved":
        lastSaved.value = msg.at;
        break;
      case "accountGone":
        accountGone();
        break;
      case "savedByHand":
        // a refusal is said with the saveSlots message that follows it
        if (msg.at) {
          savedByHand.value = msg.at;
          say(t("clockClient.saved", { time: tDate(msg.at, { hour: "numeric", minute: "2-digit" }) }));
        }
        break;
      case "answer": {
        const f = this.queries.get(msg.seq);
        this.queries.delete(msg.seq);
        this.sent.delete(msg.seq);
        f?.(msg.data);
        this.pollAnswered(msg.seq, msg.ms);
        break;
      }
      case "trips":
        if (msg.board) {
          this.boardLog.push({ ms: msg.boardMs ?? 0, redone: msg.redone ?? 0, bytes: msg.board.byteLength, prepMs: msg.prepMs ?? 0, prepMax: msg.prepMax ?? 0 });
          if (this.boardLog.length > 40) this.boardLog.shift();
          accountVersion.value++;
        }
        this.tripsAt = performance.now();
        this.onTrips(msg.epoch, msg.version, msg.trips, msg.board, msg.ends, msg.nexts, msg.bays);
        break;
      case "edited": {
        const f = this.pending.get(msg.seq);
        this.pending.delete(msg.seq);
        this.sent.delete(msg.seq);
        this.editLog.push({ worker: msg.ms, stateMs: this.lastStateMs });
        if (this.editLog.length > 40) this.editLog.shift();
        // T-531: the world before and after this edit's state say where its money went
        // (an early answer's world comes with its state, T-1407)
        if (msg.ok && msg.charge > 0 && msg.early) this.earlyShares = { seq: msg.seq, line: msg.line, paid: msg.paid };
        else if (msg.ok && msg.charge > 0 && world.value) this.post({ kind: "chargeShares", seq: msg.seq, ...chargeShares(this.worldBefore, world.value, msg.line, msg.paid) });
        f?.({ ok: msg.ok, issues: msg.issues, charge: msg.charge, line: msg.line, node: msg.node, skipped: msg.skipped, early: msg.early, held: msg.held });
        break;
      }
      case "preview": {
        // T-1001: a callback that throws still lets the preview waiting behind it go
        this.previews.answered((cb) => {
          if (msg.stale) return;
          this.toCurrentAnchor(msg.anchor, msg.verts, msg.pts);
          if (msg.arc) this.toCurrentAnchor(msg.anchor, msg.arc, new Float32Array(0));
          if (msg.marks) this.toCurrentAnchor(msg.anchor, new Float64Array(0), msg.marks);
          if (msg.legs) this.toCurrentAnchor(msg.anchor, msg.legs, new Float32Array(0));
          if (msg.ramps) this.toCurrentAnchor(msg.anchor, new Float64Array(0), msg.ramps);
          cb?.({ ok: msg.ok, cost: msg.cost, lengthM: msg.lengthM, mult: msg.mult, ground: msg.ground, issues: msg.issues, verts: msg.verts, pts: msg.pts, speeds: msg.speeds, lanes: msg.lanes, partLanes: msg.partLanes, parts: msg.parts, arc: msg.arc, marks: msg.marks, joins: msg.joins, noFlyover: msg.noFlyover, legs: msg.legs, ramps: msg.ramps });
        }, (q, cb) => this.ask(q, cb));
        break;
      }
      case "stationCheck": {
        this.stations.answered((cb) => cb?.(msg.ok, msg.issues, msg.cost), (q, cb) => this.checkStation(q.at, q.platform, cb));
        break;
      }
      case "crossoverCheck": {
        this.crossovers.answered((cb) => {
          for (const p of msg.parts) this.toCurrentAnchor(msg.anchor, new Float64Array(0), p);
          cb?.(msg);
        }, (q, cb) => this.checkCrossover(q.edge, q.x, q.y, cb));
        break;
      }
      case "snapshot":
        networkSnapshot.value = msg.snapshot;
        for (const f of snapshotListeners) f(msg.snapshot);
        break;
    }
  }

  private queries = new Map<number, (d: Float64Array) => void>();
  /** T-1815: panel reads (`PANEL_READS`) asked while track previews are being asked wait until
   * previews stop for `PANEL_HOLD_MS`, the same ones asked once: each takes 15-50 ms in the clock
   * worker on a big network, and a preview behind it waits that long */
  private lastPreviewAt = -Infinity;
  private heldReads = new Map<string, { q: Extract<ToClock, { kind: "query" }>["q"]; res: ((d: Float64Array) => void)[]; soon: boolean }>();
  private heldTimer = 0;
  /** T-2766: stop clicks sent and not yet answered, when the last was sent, and when the last was answered */
  private stopsOut = 0;
  private lastStopSent = -Infinity;
  private lastStopAt = -Infinity;
  /** a stop click unanswered (for at most 10 s: a worker that never answers holds no read for ever) */
  private stopOut = () => this.stopsOut > 0 && performance.now() - this.lastStopSent < 10000;
  private holdsRead(what: string): boolean {
    const now = performance.now();
    if (this.stopOut() || now - this.lastStopAt < STOP_HOLD_MS) return !STOP_READS.has(what);
    return PANEL_READS.has(what) && (now - this.lastPreviewAt < PANEL_HOLD_MS || this.heldReads.size > 0);
  }
  private releaseHeld() {
    this.heldTimer = 0;
    const wait = this.stopOut() ? STOP_HOLD_MS : Math.max(this.lastPreviewAt + PANEL_HOLD_MS, this.lastStopAt + STOP_HOLD_MS) - performance.now();
    if (wait > 0) {
      this.heldTimer = setTimeout(() => this.releaseHeld(), wait) as unknown as number;
      return;
    }
    const held = [...this.heldReads.values()];
    this.heldReads.clear();
    for (const h of held) void this.query(h.q, h.soon).then((d) => h.res.forEach((r, k) => r(k ? d.slice() : d)));
  }
  /** `soon`: a poll read past its rest (still behind the queued ones), for a count the player just changed */
  query(q: ClockQ, soon = false): Promise<Float64Array> {
    if (this.holdsRead(q.what)) {
      const key = JSON.stringify(q);
      return new Promise((res) => {
        const h = this.heldReads.get(key);
        if (h) { h.res.push(res); h.soon ||= soon; }
        else this.heldReads.set(key, { q, res: [res], soon });
        if (!this.heldTimer) this.releaseHeld();
      });
    }
    // T-2953: the panels each ask the same account read after an edit (the Lines tab, the trouble list, the
    // goals); with nothing posted since the first, one answer serves them all (each its own copy past the account reads)
    const poll = POLL_READS.has(q.what), key = poll || PANEL_READS.has(q.what) ? JSON.stringify(q) : "";
    const same = key ? this.readsOut.get(key) : undefined;
    if (same && same.posts === this.posts) return PANEL_READS.has(q.what) ? same.p : same.p.then((d) => d.slice());
    if (poll && (this.pollWait.has(key) || !this.pollGoes(key, soon))) {
      return new Promise((res) => {
        const h = this.pollWait.get(key);
        if (h) { h.res.push(res); h.soon ||= soon; }
        else this.pollWait.set(key, { q, res: [res], soon });
        this.pumpPolls();
      });
    }
    return this.sendQuery(q, key, poll);
  }
  private sendQuery(q: ClockQ, key: string, poll: boolean): Promise<Float64Array> {
    const seq = ++this.seq;
    this.post({ kind: "query", seq, q });
    const at = performance.now();
    this.sent.set(seq, { what: q.what, at });
    if (poll) this.pollsOut.set(seq, { at, key });
    const p = new Promise<Float64Array>((res) => this.queries.set(seq, res));
    if (key) {
      const read = { posts: this.posts, p };
      this.readsOut.set(key, read);
      void p.then(() => { if (this.readsOut.get(key) === read) this.readsOut.delete(key); });
    }
    return p;
  }

  /** T-3801: poll reads waiting on the page (`POLL_READS`), by question, in the order first asked */
  private pollWait = new Map<string, { q: ClockQ; res: ((d: Float64Array) => void)[]; soon: boolean }>();
  /** poll reads sent and not answered, by seq: when each was sent, and its question */
  private pollsOut = new Map<number, { at: number; key: string }>();
  /** questions answered before: a question new to the page (a panel just opened) skips the rest */
  private pollSeen = new Set<string>();
  /** no poll read before this time (performance.now), after the last one's answer */
  private pollRestUntil = 0;
  private pollTimer = 0;
  /** a poll read may go: a question never answered at once; else none other in flight and the rest over (`soon` skips
   * the rest) */
  private pollGoes(key: string, soon: boolean): boolean {
    if (!this.pollSeen.has(key)) return true;
    const now = performance.now();
    for (const [seq, r] of this.pollsOut) if (now - r.at > POLL_LOST_MS) this.pollsOut.delete(seq);
    return !this.pollsOut.size && (soon || now >= this.pollRestUntil) && (this.pollWait.size === 0 || this.pollWait.has(key));
  }
  /** the first waiting poll read (a `soon` one first) that may go; else a timer for the rest's end (or a lost read's) */
  private pumpPolls() {
    clearTimeout(this.pollTimer);
    this.pollTimer = 0;
    const waiting = [...this.pollWait];
    const first = waiting.find(([, h]) => h.soon) ?? waiting[0];
    if (!first) return;
    const [key, h] = first;
    if (this.pollGoes(key, h.soon)) {
      this.pollWait.delete(key);
      void this.sendQuery(h.q, key, true).then((d) => h.res.forEach((r, k) => r(k ? d.slice() : d)));
      return;
    }
    const now = performance.now();
    let at = h.soon ? now : Math.max(now, this.pollRestUntil);
    for (const r of this.pollsOut.values()) at = Math.max(at, r.at + POLL_LOST_MS);
    this.pollTimer = setTimeout(() => this.pumpPolls(), Math.max(1, at - now)) as unknown as number;
  }
  /** a poll read answered: the rest from the worker's own time for it, then the next waiting one */
  private pollAnswered(seq: number, ms: number | undefined) {
    const r = this.pollsOut.get(seq);
    if (!r) return;
    this.pollsOut.delete(seq);
    this.pollSeen.add(r.key);
    const cost = ms ?? Math.min(performance.now() - r.at, POLL_REST_MAX_MS);
    this.pollRestUntil = Math.max(this.pollRestUntil, performance.now() + Math.min(POLL_REST_MAX_MS, POLL_REST * cost));
    this.pumpPolls();
  }

  /** T-3297: `flipSide` above. */
  flipSide(): Promise<Float64Array> {
    const seq = ++this.seq;
    this.post({ kind: "flipSide", seq });
    return new Promise<Float64Array>((res) => this.queries.set(seq, res));
  }

  /** Send an edit; the promise resolves with the worker's answer. `early` "rows": a stop click's (T-2766), answered
   * once the world holds it, its network a moment after the clicks stop. `game`: the `gameStamp` it was made under,
   * for an edit sent later than made (T-3734). */
  edit(op: EditOp, early: boolean | "rows" = false, game?: number): Promise<EditAnswer> {
    const seq = ++this.seq, stamp = game === undefined ? {} : { game };
    if (early === "rows") {
      this.stopsOut++;
      this.lastStopSent = performance.now();
      this.post({ kind: "edit", seq, op, rows: true, ...stamp });
      return new Promise((res) => this.pending.set(seq, (a) => { this.stopsOut--; this.lastStopAt = performance.now(); res(a); }));
    }
    this.post(early ? { kind: "edit", seq, op, early, ...stamp } : { kind: "edit", seq, op, ...stamp });
    this.sent.set(seq, { what: op.op, at: performance.now() });
    return new Promise((res) => this.pending.set(seq, res));
  }

  /** Ask what a route would cost. One request in flight; newer ones replace a waiting one. */
  preview(req: RouteAsk, cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "preview", seq, ...req }), cb);
  }

  /** What giving blueprint track new PIs would do (T-064); shares the one-in-flight queue. */
  previewEdge(edge: number, pis: PiInput[], cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "previewEdge", seq, edge, pis, design: designSpeed.peek() }), cb);
  }

  /** What moving a blueprint node to pack metres (x, y) would do (T-093); the same queue. */
  previewNode(node: number, x: number, y: number, cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "previewNode", seq, node, x, y, design: designSpeed.peek() }), cb);
  }

  previewNodes(nodes: number[], dx: number, dy: number, cb: (a: PreviewAnswer) => void) {
    this.ask((seq) => ({ kind: "previewNodes", seq, nodes, dx, dy, design: designSpeed.peek() }), cb);
  }

  /** Whether a station could be placed here (T-311): the worker tries `addStation` and rolls it
   * back, `cost` (US$M) what it would add to the blueprint. One request in flight; a newer one
   * replaces a waiting one. */
  checkStation(at: StationAt, platform: number, cb: (ok: boolean, issues: string[], cost: number) => void) {
    this.lastPreviewAt = performance.now();
    if (!this.stations.ask({ at, platform }, cb)) return;
    this.post({ kind: "previewStation", seq: ++this.seq, at, platform });
  }

  /** What a crossover on track `edge` near pack metres (x, y) would cost, or why not (T-338), with
   * its two diagonals to draw. One request in flight; a newer one replaces a waiting one. */
  checkCrossover(edge: number, x: number, y: number, cb: (a: CrossoverAnswer) => void) {
    if (!this.crossovers.ask({ edge, x, y }, cb)) return;
    this.post({ kind: "previewCrossover", seq: ++this.seq, edge, x, y });
  }

  /** Drop a preview waiting to be sent, and the one in flight if it is not yet being worked out (a
   * drag was cancelled or finished, a run ended). */
  dropPreview() {
    this.previews.waiting = null;
    this.staleInFlight();
  }

  /** T-2404: the state of segments placed early, held while the ghost is asked, wanted now */
  placedNow() {
    this.post({ kind: "release" });
  }

  /** the preview in flight, and whether the worker was told it is no longer wanted (T-1039) */
  private previewSeq = 0;
  private previewStaleSent = false;
  private staleInFlight() {
    if (!this.previews.busy || this.previewStaleSent) return;
    this.previewStaleSent = true;
    this.post({ kind: "previewStale", seq: this.previewSeq });
  }

  private ask(msg: (seq: number) => ToClock, cb: (a: PreviewAnswer) => void) {
    this.lastPreviewAt = performance.now();
    if (!this.previews.ask(msg, cb)) {
      // not yet started, or waiting for masks, it is dropped for this one (T-1039)
      this.staleInFlight();
      return;
    }
    this.previewSeq = ++this.seq;
    this.previewStaleSent = false;
    this.post(msg(this.previewSeq));
  }

  /** T-1007, dev server only (`?debug=1`, `tw.client.debugPanic`): make the track model panic. */
  debugPanic(when: "now" | "read" | "afterEdit") {
    if (import.meta.env.DEV) this.post({ kind: "debugPanic", when });
  }

  /** T-2766, dev server only: set the cash, US$M; `fleet`: own only the cars in use. */
  debugCash(cash: number, fleet = false) {
    if (import.meta.env.DEV) this.post({ kind: "debugCash", cash, fleet });
  }

  /** The render epoch the renderer is drawing (start of the game hour). */
  setEpoch(epoch: number) {
    if (epoch === this.epoch) return;
    this.epoch = epoch;
    this.post({ kind: "epoch", epoch });
  }
}
