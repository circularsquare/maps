// Main-thread side of the clock worker (T-025). Never waits on it: requests go out, answers
// arrive as messages. Each "state" becomes the `world` snapshot (game/world.ts) and the
// renderer's buffers; "trips" go to the renderer; network snapshots go to whoever registered
// with `onNetworkSnapshot` (the demand pool, T-026).

import { effect, signal } from "@preact/signals";
import type { DiagramGraph } from "../diagram/graph";
import { clock } from "../game/clock";
import { activeCity, HOME_CITY, setWorldFrame, viewedZone } from "../game/cities";
import { setDemandFineSink, setGameOpened, setOpenedSink } from "../game/openedZones";
import { setHandOrderSink, takeHandOrder } from "../game/handOrder";
import { hiddenZones, setGameStart } from "../game/realms";
import { fitView, holdForGameLoad, loadCannotFinish, markAlive } from "../game/loading";
import { incomeByDay, lastSaved, liveMoney, moneyState, moneyTerms, moneyWorld, setMoneyState } from "../game/money";
import { zonesLoaded } from "../game/zones";
import { chargeShares, lineShares, upkeepShares } from "../game/regionMoney";
import { GOAL_SHARES, linePoints, STAR_SHARE } from "../game/points";
import { AVERAGE_DAY, AWAY_DAY, awaySoon, BACK_DAY, DAY_KINDS, OUT_DAY, rushSoon } from "../game/days";
import { demandByDay, demandView, platformTransfers, roughZones } from "../game/demand";
import { accountVersion, packArrivals, packRiders } from "../game/boarding";
import { designSpeed, guideFinished, lineDraft, selection, tool } from "../game/ui";
import { say, setEditSink, world } from "../game/world";
import type { LineInput, LineStats, StationInput, TrackEdge, TrackNode, WorldView } from "../game/types";
import type { ClockState, EditOp, FromClock, NetworkSnapshot, PiInput, RouteEnd, StationAt, ToClock } from "./protocol";
import { anchorVersion } from "../map/anchorVersion";
import { mergeNet, type NetworkBuffers } from "../map/network";
import { bandTry } from "../perfFlags";
import { anchor, anchorWanted, EARTH_C, HOME_ANCHOR, makeAnchor, setAnchor, type Anchor } from "../map/geo";
import { decodeSave, readStored, writePending, type GuideSave, type NewGameSpec, type PendingLoad, type SaveSlot } from "./saveFile";
import type { CloudContext } from "./cloud";
import type { FrameSpec } from "../game/frame";
import { dataUrl } from "../game/dataBase";
import { markDownloaded } from "../game/keepState";
import { saveRefusal } from "../game/memoryCheck";
import { askBeforeLeaving, keepStorage } from "../game/saveSafety";
import { afterPaint, BUSY_MS, expectedMs, hideBusy, learnBusy, showBusy } from "../game/busy";
import { addTerrainGrid } from "../game/terrain";
import { letterOf } from "../game/lineLetter";

export const saveSlotsState = signal<{ slots: import("./saveFile").SaveSlot[]; active: string; writable: boolean; at: number; reason?: string; error?: string; backups: import("./saveFile").SlotBackup[]; newer?: boolean }>({ slots: [], active: "autosave", writable: true, at: 0, backups: [] });
/** When the game was last saved by hand in this page (T-1251). */
export const savedByHand = signal<number | null>(null);

/** T-1815: the panels' account reads, held while track previews are asked */
const PANEL_READS = new Set(["today", "linesPeriods", "linePeriods", "lineAccount"]);
const PANEL_HOLD_MS = 250;
export function manageSaveSlot(action: "refresh" | "save" | "create" | "open" | "rename" | "delete" | "restore", key?: string, name?: string) {
  // T-1196: a game opened or restored reloads the page into it, T-1197: if it fits
  if (action === "open" && key && key !== saveSlotsState.peek().active) return void ifItFits({ key }, () => switchGame({ slot: key }));
  if (action === "restore" && key) return void ifItFits({ key }, () => switchGame({ load: "restore" }, { kind: "restore", key, from: saveSlotsState.peek().active }));
  // T-1852: deleting the game being played ends it; the page then goes back to the start menu
  if (action === "delete" && key && key === saveSlotsState.peek().active && current?.running) {
    return void current.leave(undefined, toStartMenu, { kind: "saveSlot", action, key });
  }
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
/** T-2124: counts the games loaded into the worker (each `loaded`), so per-game page state starts again */
export const gamesLoaded = signal(0);
/** The newest network snapshot (what demand runs on); null until the worker has started. */
export const networkSnapshot = signal<NetworkSnapshot | null>(null);

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
}

export interface PreviewAnswer {
  ok: boolean;
  cost: number;
  lengthM: number;
  /** cost multiplier: the track's price over its length at the base price (T-085) */
  mult: number;
  /** metres on flat, rolling, hilly and mountain ground where the terrain is charged (T-1270) */
  ground?: number[];
  issues: string[];
  verts: Float64Array;
  pts: Float32Array;
  speeds?: Float32Array;
  /** where each alignment starts in `pts` (triples) when there are several (a node drag) */
  parts?: Uint32Array;
  /** a tangent route's circle (T-418): centre x, y (local units), ground radius m, 0 */
  arc?: Float64Array;
  /** where it makes a crossing or a flat junction (T-629): x, y, kind (0 crossing, 1 junction) */
  marks?: Float32Array;
}

/** A crossover preview (T-338): its cost US$M or why not, and its diagonals' centre lines. */
export type CrossoverAnswer = Pick<Extract<FromClock, { kind: "crossoverCheck" }>, "ok" | "issues" | "cost" | "parts">;

const snapshotListeners: ((s: NetworkSnapshot) => void)[] = [];
/** Call `f` with every network snapshot from now on (and the newest one, if there is one). */
export function onNetworkSnapshot(f: (s: NetworkSnapshot) => void) {
  snapshotListeners.push(f);
  if (networkSnapshot.peek()) f(networkSnapshot.peek()!);
}

let stationsAheadSink: ((frame: FrameSpec, xy: Float64Array, fine: string[] | null, built: string[][] | null) => void) | null = null;
/** T-889: call `f` with a loading save's built stations, before its first snapshot. */
export function onStationsAhead(f: (frame: FrameSpec, xy: Float64Array, fine: string[] | null, built: string[][] | null) => void) {
  stationsAheadSink = f;
}
const roadsListeners: (() => void)[] = [];
/** T-1929: call `f` whenever road tiles land in the track model (a ghost priced before them is out of date). */
export function onRoadsLanded(f: () => void) {
  roadsListeners.push(f);
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
let lastStations: StationInput[] = [];
let lastLines: LineInput[] = [];
function sameObjects<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** The UI's world view from the worker's state. */
function toWorld(s: ClockState, city: { population: number; jobs: number }): WorldView {
  let stations: StationInput[] = s.nodes
    .filter((n) => n.platform > 0)
    .map((n) => {
      let v = stationViews.get(n);
      if (!v) stationViews.set(n, v = {
        id: String(n.id), num: n.id, name: n.name, lng: n.lng, lat: n.lat, x: n.x, y: n.y, level: n.level as any, length: n.platform, built: n.stationBuilt, off: n.stationOff, upgrades: n.upgrades, heading: n.heading, ...(n.line ? { line: n.line } : {}), ...(n.port ? { port: n.port } : {}),
      });
      return v;
    });
  if (sameObjects(stations, lastStations)) stations = lastStations;
  else lastStations = stations;
  let lines: LineInput[] = s.lines.map((l) => {
    let v = lineViews.get(l);
    if (!v) lineViews.set(l, v = {
      id: String(l.id), num: l.id, letter: l.letter || letterOf(l.name), symbol: l.letter, name: l.name, colour: l.colour, stops: l.stops.map(String),
      ...(l.group ? { group: l.group, groupName: l.groupName, groupSymbol: l.groupSymbol, groupColour: l.groupColour } : {}),
      tph: { high: l.tph[0], medium: l.tph[1], low: l.tph[2] },
    });
    return v;
  });
  if (sameObjects(lines, lastLines)) lines = lastLines;
  else lastLines = lines;
  const lineStats: Record<string, LineStats> = {};
  const edgeLines = new Map<number, string[]>();
  for (const l of s.lines) {
    for (const p of new Set(l.path.map((p) => p >> 1))) {
      const v = edgeLines.get(p) ?? [];
      v.push(String(l.id));
      edgeLines.set(p, v);
    }
    lineStats[String(l.id)] = {
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
      upkeep: (l.upkeep ?? 0) * 1e6,
      stretches: Array.from({ length: l.stretchSpeeds.length / 3 }, (_, k) => ({ reached: l.stretchSpeeds[3 * k], limit: l.stretchSpeeds[3 * k + 1], why: l.stretchSpeeds[3 * k + 2] })),
      stopS: l.stopS,
      dwellS: l.dwell,
      turnaroundS: l.turnaround,
      shift: l.shift,
      passed: Array.from({ length: l.passed.length >> 1 }, (_, k) => ({ gap: l.passed[k * 2], node: l.passed[k * 2 + 1] })),
      reversals: l.reversals ?? [],
      endTurns: l.endTurns ?? [],
      loop: l.loopKind === 1 ? "circle" : l.loopKind === 2 ? "balloon" : null,
      ...(l.bus ? { bus: l.bus } : {}),
    };
  }
  const edges: TrackEdge[] = s.edges.map((e) => ({
    id: e.id, a: e.a, b: e.b, tracks: e.tracks, built: e.built, off: e.off, hsr: e.hsr, crossover: e.crossover, partner: e.partner, lengthM: e.lengthM, cost: e.cost * 1e6, vminKmh: e.vminKmh,
    levelMin: e.levelMin as any, levelMax: e.levelMax as any, lines: edgeLines.get(e.id) ?? [],
  }));
  const nodes = new Map<number, TrackNode>(
    s.nodes.map((n) => [n.id, { id: n.id, x: n.x, y: n.y, lng: n.lng, lat: n.lat, level: n.level as any, ports: n.ports, builtPorts: n.builtPorts, flying: n.flying, platform: n.platform, ...(n.bus ? { bus: true as const } : {}), ...(n.port ? { port: true as const } : {}) }]),
  );
  return {
    version: s.version,
    capacity: s.markers,
    fullTrack: (s.fullTrack ?? []).map((v) => {
      const m = new Map<number, { rho: number; tph: number }>();
      for (let k = 0; k + 2 < v.length; k += 3) m.set(v[k], { rho: v[k + 1], tph: v[k + 2] });
      return m;
    }),
    save: { version: 0, city: HOME_CITY, clock: clock.now(), cash: s.cash * 1e6, stations, lines, links: pairsOf(s.links ?? []), ...(s.buildings?.length ? { buildings: s.buildings } : {}) },
    lineStats,
    edges,
    nodes,
    city: { name: "New York", population: city.population, jobs: city.jobs, split: { train: 0, walk: 0, drive: 100 }, trainShareByPeriod: [] },
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
/** A save's bytes downloaded as "trainsworld-<name>-day-<day>.save". */
function downloadSave(bytes: Uint8Array, name: string, day: number, slot?: string) {
  markDownloaded(slot); // T-1832: the "saved in this browser only" card has nothing more to say
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([bytes as BlobPart], { type: "application/octet-stream" }));
  const named = name.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "").replace(/-+/g, "-").replace(/^-|-$/g, "");
  a.download = `trainsworld-${named || "game"}-day-${day}.save`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
/** T-1395: a saved game not being played downloaded as its last save in this browser; false if it
 * could not be read. */
export async function exportSlot(slot: SaveSlot): Promise<boolean> {
  const stored = await readStored(slot.key).catch(() => null);
  if (!stored) return false;
  downloadSave(stored.bytes, slot.name, stored.day, slot.key);
  return true;
}
/** Load a save file the player picked, into a new slot named after the file (T-743). */
/** `version`: the file is a version of that game (an account backup, T-2037), else a game of its own. */
export async function importGame(file: File, version?: { origin: string; edit: number }) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (await ifItFits({ bytes })) switchGame({ load: "file" }, { kind: "import", bytes, name: file.name.replace(/\.save$/i, ""), from: saveSlotsState.peek().active, ...version });
}
/** T-1612: a save file picked on the start menu, where no game is in hand to reload from: its game is
 * left in storage as after `switchGame`, and the menu's pick is `{ pending: true }`. A reason (to show)
 * when the file cannot open. */
export async function fileChoice(file: File, version?: { origin: string; edit: number }): Promise<GameChoice | string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  try { await decodeSave(bytes); } catch { return "This file is not a saved game."; }
  const why = await saveRefusal({ bytes });
  if (why) return why;
  if (!await writePending({ kind: "import", bytes, name: file.name.replace(/\.save$/i, ""), from: "", ...version })) return "Browser storage is unavailable.";
  return { pending: true };
}
/** T-2037: a backup of slot `from` opened from the start menu, as `fileChoice` does a file. */
export async function backupChoice(key: string, from: string): Promise<GameChoice | string> {
  const why = await saveRefusal({ key });
  if (why) return why;
  if (!await writePending({ kind: "restore", key, from })) return "Browser storage is unavailable.";
  return { pending: true };
}
/** Start a new game in a new slot (T-743), named after its start (T-728) unless `name` is given. */
export function newGame(game?: NewGameSpec, name = "") {
  const spec = game ?? { mode: "normal" };
  switchGame({ load: "new", mode: spec.mode, ...(spec.start ? { city: spec.start } : {}), ...(spec.real ? { real: "1" } : {}) }, { kind: "new", spec, name });
}

/** T-1196: the URL switches that say which game a page opens (ui/StartMenu.tsx `choiceFromUrl`) */
export const CHOICE_PARAMS = ["new", "city", "mode", "synth", "demandTest", "continue", "slot", "load", "real"];
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
  if (current) current.leave(undefined, go);
  else go();
}
/** T-1197: whether the save fits in this device's memory; if not the player is told why. `then` runs if it does. */
async function ifItFits(src: { bytes: Uint8Array } | { key: string }, then?: () => void): Promise<boolean> {
  const why = await saveRefusal(src);
  if (why) say(why, true);
  else then?.();
  return !why;
}
/** T-728: make the game a sandbox, for good. */
export function makeSandbox() {
  current?.post({ kind: "sandbox" });
}
/** T-1822: keep the Getting started card's progress in the save. */
export function saveGuide(guide: GuideSave) {
  current?.post({ kind: "guide", guide });
}
/** T-2124: achievement steps met (game/achievements.ts); the worker keeps each from the first time. */
export function postAchievements(got: Record<string, Omit<import("./saveFile").Achieved, "at">>) {
  current?.post({ kind: "achieve", got });
}
/** Ask the clock worker an inspector question (T-031); null before the worker is there. */
export function queryClock(q: Extract<ToClock, { kind: "query" }>["q"]): Promise<Float64Array | null> {
  return current ? current.query(q) : Promise.resolve(null);
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
}

export class ClockClient {
  private worker: Worker | null = null;
  private waiting: [ToClock, Transferable[]][] = [];
  private seq = 0;
  private pending = new Map<number, (a: EditAnswer) => void>();
  private stationBusy = false;
  private stationNext: { at: StationAt; platform: number; cb: (ok: boolean, issues: string[], cost: number) => void } | null = null;
  private stationCb: ((ok: boolean, issues: string[], cost: number) => void) | null = null;
  private crossBusy = false;
  private crossNext: { edge: number; x: number; y: number; cb: (a: CrossoverAnswer) => void } | null = null;
  private crossCb: ((a: CrossoverAnswer) => void) | null = null;
  private previewBusy = false;
  private previewNext: { msg: (seq: number) => ToClock; cb: (a: PreviewAnswer) => void } | null = null;
  private previewCb: ((a: PreviewAnswer) => void) | null = null;
  private city = { population: 0, jobs: 0 };
  private epoch = NaN;
  private sawSlots = false;
  /** the last game load: where it came from and whether its save was converted to the world frame
   * (T-308; read by `?debug=1` checks, the player is told nothing) */
  lastLoad: { from: "autosave" | "file" | "new"; converted: boolean } | null = null;
  /** the renderer's hooks, set by main.ts */
  onState: (s: ClockState) => void = () => {};
  /** T-2159: the network diagram's graph, while the diagram is shown (diagram/view.ts) */
  onDiagram: (g: DiagramGraph) => void = () => {};
  /** `board`: the boarding account's hour (T-164, game/boarding.ts), once demand has riders */
  onTrips: (epoch: number, version: number, trips: Float32Array, board?: Float32Array, ends?: Float32Array, nexts?: Float32Array) => void = () => {};
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
    this.worker = new Worker(new URL("./clock.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<FromClock>) => this.arrive(e.data);
    this.worker.onerror = (e) => this.failed(e.message || "its script did not load");
    this.worker.onmessageerror = () => this.failed("a message could not be read");
    const q = new URLSearchParams(location.search);
    const fresh = !!choice.fresh;
    // T-320: a check or stress page never writes over a saved game
    this.testPage = fresh || q.get("synth") === "1" || q.get("demandTest") === "1";
    let slot = choice.slot ?? "autosave";
    if (!choice.slot) try { slot = localStorage.getItem("anitabuilder-save-slot") || slot; } catch { /* storage unavailable */ }
    const backupMs = Number(q.get("backupMs")) || undefined;
    this.patchMode = q.get("patch") ?? "1";
    this.worker.postMessage({ kind: "init", city: this.cityId, pack: dataUrl(this.cityId), outlines: dataUrl("outlines.json"), fresh, testPage: this.testPage, slot, newSlot: choice.newSlot, backupMs, game: choice.game, pending: choice.pending, patch: this.patchMode, guide: !guideFinished.peek(), bandMode: bandTry() } satisfies ToClock, choice.game?.real ? [choice.game.real] : []);
    for (const [msg, transfer] of this.waiting) this.worker.postMessage(msg, transfer);
    this.waiting = [];
  }

  /** T-950: the worker failed; before its first game no game will come, so the card gives way to the
   * failure screen */
  private failed(why: string) {
    clockWorkerStatus.value = "failed to load: " + why;
    if (!world.peek()) loadCannotFinish(`The train simulation could not start (${why}).`, true);
  }

  constructor(cityId = HOME_CITY) {
    this.cityId = cityId;
    setEditSink((op, early) => this.edit(op, early));
    setOpenedSink((zones) => this.post({ kind: "opened", zones }));
    setDemandFineSink((zones, built) => this.post({ kind: "demandFine", zones, built }));
    setHandOrderSink((kind, ids) => this.post({ kind: "listOrder", list: kind === "lines" ? 0 : 1, ids }));
    current = this;
    // The worker settles money from the clock (T-028, T-284): tell it each time the
    // clock is set, paused, resumed or sped up.
    const sendClock = () => this.post({ kind: "clock", t: clock.now(), rate: clock.ticking ? clock.rate() : 0, at: performance.timeOrigin + performance.now(), game: this.game });
    clock.onChange(sendClock);
    sendClock();
    // Fare income from the riders (game/money.ts), whenever riders, fares or lines change.
    // One table per kind of day (T-260): the worker settles each hour by its own day's.
    let lastIncome = "";
    effect(() => {
      const days = incomeByDay.value;
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
      for (const [id, v] of demandByDay.value[AVERAGE_DAY]?.cities ?? []) {
        // (T-1223: never from rough figures, whose noise could cross a mark)
        if (!v.final || roughZones.value.has(id)) continue;
        if (v.split.train >= STAR_SHARE && m.stars?.[id] === undefined) this.post({ kind: "star", zone: id });
        // T-1437: the higher goals the same way
        for (const g of GOAL_SHARES) if (g !== STAR_SHARE && v.split.train >= g && m.goals?.[id]?.[g] === undefined) this.post({ kind: "star", zone: id, share: g });
      }
    });
    // Demand's riders for the boarding account (T-164): each new result, every kind of day's
    // (T-260); the trains and platforms of each hour read their own day's.
    effect(() => {
      // T-264, T-1373: a superholiday's days only while one is soon; an empty message clears them,
      // so the account reads the ordinary day's (track/boarding.rs `kind`)
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
      if (demandView.value) askTransfers();
    });
    // Hidden or closing: write the autosave now (T-029).
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && this.post({ kind: "flush" }));
    addEventListener("pagehide", () => this.post({ kind: "flush" }));
  }

  private cityId: string;
  /** the world before the last state, for the edit that state answers (T-531) */
  private worldBefore: WorldView | null = null;
  /** `?new=1`, `?synth=1` or `?demandTest=1`: no autosave until a copy is saved (T-320) */
  private testPage = false;
  /** the worker's game the clock messages are for (`loaded`'s `game`, T-715) */
  private game = 0;
  private forgetSent = () => {};
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
    this.post(instead ?? { kind: "leave", pending });
  }

  get running() { return !!this.worker; }

  post(msg: ToClock, transfer: Transferable[] = []) {
    if (this.worker) this.worker.postMessage(msg, transfer);
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
    if (this.deferred) { this.deferred.push(msg); return; }
    if (msg.kind === "state" && !this.stateBusy) {
      const st = msg.state;
      const units = (st.delta ? st.delta.edges.rows.length + st.delta.nodes.rows.length : st.edges.length + st.nodes.length) + 1;
      const kind = this.stateKind(st);
      if (expectedMs(kind, units, STATE_MS_GUESS[kind]) > BUSY_MS) {
        this.deferred = [msg];
        this.stateBusy = { token: showBusy(kind === "load" || kind === "redraw" ? "Drawing the network" : "Updating the network"), kind, units };
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
  private earlyShares: { seq: number; line?: number } | null = null;

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
    setMoneyState(msg.state.money);
    this.worldBefore = world.value;
    world.value = toWorld(msg.state, this.city);
    const es = this.earlyShares;
    this.earlyShares = null;
    if (es && world.value) this.post({ kind: "chargeShares", seq: es.seq, ...chargeShares(this.worldBefore, world.value, es.line) });
    this.onState(msg.state);
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
    // T-648: the worker posts a state's trips right after it, before anything else. The state waits
    // for them, so its world, network and trips are taken in one task: no frame draws the old trips
    // on the new run profiles (trains jumped, or stayed a frame on a line that had lost its service),
    // and the demand views build once. Not when the draw anchor moves (its listeners need the network
    // at once), and never past HOLD_MS.
    if (this.held && !(msg.kind === "trips" && msg.version < this.held.state.version) && msg.kind !== "cash") this.release();
    // T-950: the steps toward a loading card's game (not the money ticks, which come regardless)
    if (msg.kind === "ready" || msg.kind === "loaded" || msg.kind === "state" || msg.kind === "snapshot" || msg.kind === "stationsAhead") markAlive();
    switch (msg.kind) {
      case "ready":
        clockWorkerStatus.value = `ready in ${msg.initMs.toFixed(0)} ms`;
        setWorldFrame(msg.frame);
        break;
      case "error":
        console.warn("clock worker:", msg.message);
        break;
      case "recovered":
        console.error("clock worker:", msg.detail);
        if (msg.down) clockWorkerStatus.value = "track model down after a panic";
        say(msg.notice, true);
        break;
      case "failed":
        console.error("clock worker:", msg.error);
        this.failed(msg.error);
        break;
      case "state": {
        const st = msg.state;
        // T-1029: a state that cannot be made whole is dropped, and the worker posts the next whole
        if (!this.takeRows(st) || !this.takeNet(st)) {
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
        setMoneyState(msg.money);
        break;
      case "diagramGraph":
        this.onDiagram(msg.graph);
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
      case "stationsAhead":
        stationsAheadSink?.(msg.frame, msg.xy, msg.fine, msg.built);
        break;
      case "netAhead":
        netAheadSink?.(msg.key);
        break;
      case "terrain":
        addTerrainGrid(msg.id, msg.bytes);
        break;
      case "roads":
        for (const f of roadsListeners) f();
        break;
      case "loaded":
        // the game now in the worker may be in the other frame (T-125); first, before anything reads positions
        setWorldFrame(msg.frame);
        this.lastLoad = { from: msg.from, converted: !!msg.converted };
        this.fitState = this.fitNext && msg.from !== "new" && !msg.error;
        this.fitNext = false;
        if (msg.error) {
          say(msg.error, true);
          if (msg.from === "file") break;
        }
        // a different game: its clock waits for its demand (T-149); the snapshot follows
        holdForGameLoad(msg.from === "new" ? "new" : saveSlotsState.peek().active || undefined);
        // Ids are renumbered by a load: nothing old may stay selected or half drawn.
        selection.value = null;
        tool.value = "select";
        lineDraft.value = { line: null, stops: [] };
        // T-715: the clock the hold above sent is the old game's, which the worker ignores
        this.game = msg.game;
        this.forgetSent();
        gamesLoaded.value++;
        clock.set(msg.clock);
        if (msg.from === "file") say(`Game loaded, day ${Math.floor(msg.clock / 86400)}.`);
        if (msg.from === "new" && !msg.error) say("New game started.");
        break;
      case "left": {
        const then = this.onLeft;
        this.onLeft = null;
        if (msg.error) say(msg.error, true);
        else { askBeforeLeaving(false); then?.(); }
        break;
      }
      case "unsaved":
        askBeforeLeaving(msg.on);
        break;
      case "exported": {
        const { slots, active } = saveSlotsState.value;
        downloadSave(msg.bytes, slots.find((s) => s.key === active)?.name || this.cityId, msg.day, active);
        break;
      }
      case "saveSlots": {
        const before = saveSlotsState.value;
        saveSlotsState.value = { slots: msg.slots, active: msg.active, writable: msg.writable, at: msg.at, reason: msg.reason, error: msg.error, backups: msg.backups, newer: msg.newer };
        lastSaved.value = msg.at;
        if (msg.at > 0 && !this.testPage) void keepStorage(); // T-1214: a slot is written
        // a test page does not make its slot the one the next page opens, until it changes (T-320)
        if (msg.active && ((!this.sawSlots && !this.testPage) || before.active !== msg.active)) {
          try { localStorage.setItem("anitabuilder-save-slot", msg.active); } catch { /* storage unavailable */ }
        }
        this.sawSlots = true;
        // the reason a slot is not written stays in the top bar (ui/TopBar.tsx SaveNotice), not here
        if (msg.error) say(msg.error, true);
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
          say(`Saved ${new Date(msg.at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`);
        }
        break;
      case "answer": {
        const f = this.queries.get(msg.seq);
        this.queries.delete(msg.seq);
        f?.(msg.data);
        break;
      }
      case "trips":
        if (msg.board) {
          this.boardLog.push({ ms: msg.boardMs ?? 0, redone: msg.redone ?? 0, bytes: msg.board.byteLength, prepMs: msg.prepMs ?? 0, prepMax: msg.prepMax ?? 0 });
          if (this.boardLog.length > 40) this.boardLog.shift();
          accountVersion.value++;
        }
        this.onTrips(msg.epoch, msg.version, msg.trips, msg.board, msg.ends, msg.nexts);
        break;
      case "edited": {
        const f = this.pending.get(msg.seq);
        this.pending.delete(msg.seq);
        this.editLog.push({ worker: msg.ms, stateMs: this.lastStateMs });
        if (this.editLog.length > 40) this.editLog.shift();
        // T-531: the world before and after this edit's state say where its money went
        // (an early answer's world comes with its state, T-1407)
        if (msg.ok && msg.charge > 0 && msg.early) this.earlyShares = { seq: msg.seq, line: msg.line };
        else if (msg.ok && msg.charge > 0 && world.value) this.post({ kind: "chargeShares", seq: msg.seq, ...chargeShares(this.worldBefore, world.value, msg.line) });
        f?.({ ok: msg.ok, issues: msg.issues, charge: msg.charge, line: msg.line, node: msg.node, skipped: msg.skipped, early: msg.early });
        break;
      }
      case "preview": {
        this.previewBusy = false;
        // T-1001: a callback that throws still lets the preview waiting behind it go
        try {
          if (msg.stale) break;
          this.toCurrentAnchor(msg.anchor, msg.verts, msg.pts);
          if (msg.arc) this.toCurrentAnchor(msg.anchor, msg.arc, new Float32Array(0));
          if (msg.marks) this.toCurrentAnchor(msg.anchor, new Float64Array(0), msg.marks);
          this.previewCb?.({ ok: msg.ok, cost: msg.cost, lengthM: msg.lengthM, mult: msg.mult, ground: msg.ground, issues: msg.issues, verts: msg.verts, pts: msg.pts, speeds: msg.speeds, parts: msg.parts, arc: msg.arc, marks: msg.marks });
        } finally {
          const n = this.previewNext;
          this.previewNext = null;
          if (n) this.ask(n.msg, n.cb);
        }
        break;
      }
      case "stationCheck": {
        this.stationBusy = false;
        this.stationCb?.(msg.ok, msg.issues, msg.cost);
        const n = this.stationNext;
        this.stationNext = null;
        if (n) this.checkStation(n.at, n.platform, n.cb);
        break;
      }
      case "crossoverCheck": {
        this.crossBusy = false;
        for (const p of msg.parts) this.toCurrentAnchor(msg.anchor, new Float64Array(0), p);
        this.crossCb?.(msg);
        const n = this.crossNext;
        this.crossNext = null;
        if (n) this.checkCrossover(n.edge, n.x, n.y, n.cb);
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
  private heldReads = new Map<string, { q: Extract<ToClock, { kind: "query" }>["q"]; res: ((d: Float64Array) => void)[] }>();
  private heldTimer = 0;
  private releaseHeld() {
    this.heldTimer = 0;
    const wait = this.lastPreviewAt + PANEL_HOLD_MS - performance.now();
    if (wait > 0) {
      this.heldTimer = setTimeout(() => this.releaseHeld(), wait) as unknown as number;
      return;
    }
    const held = [...this.heldReads.values()];
    this.heldReads.clear();
    for (const h of held) void this.query(h.q).then((d) => h.res.forEach((r, k) => r(k ? d.slice() : d)));
  }
  query(q: Extract<ToClock, { kind: "query" }>["q"]): Promise<Float64Array> {
    if (PANEL_READS.has(q.what) && (performance.now() - this.lastPreviewAt < PANEL_HOLD_MS || this.heldReads.size)) {
      const key = JSON.stringify(q);
      return new Promise((res) => {
        const h = this.heldReads.get(key);
        if (h) h.res.push(res);
        else this.heldReads.set(key, { q, res: [res] });
        if (!this.heldTimer) this.releaseHeld();
      });
    }
    const seq = ++this.seq;
    this.post({ kind: "query", seq, q });
    return new Promise((res) => this.queries.set(seq, res));
  }

  /** Send an edit; the promise resolves with the worker's answer. */
  edit(op: EditOp, early = false): Promise<EditAnswer> {
    const seq = ++this.seq;
    this.post(early ? { kind: "edit", seq, op, early } : { kind: "edit", seq, op });
    return new Promise((res) => this.pending.set(seq, res));
  }

  /** Ask what a route would cost. One request in flight; newer ones replace a waiting one. */
  preview(req: { from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean; design?: number; hsr?: boolean; ramp?: boolean; beside?: [number, number] }, cb: (a: PreviewAnswer) => void) {
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
    if (this.stationBusy) {
      this.stationNext = { at, platform, cb };
      return;
    }
    this.stationBusy = true;
    this.stationCb = cb;
    this.post({ kind: "previewStation", seq: ++this.seq, at, platform });
  }

  /** What a crossover on track `edge` near pack metres (x, y) would cost, or why not (T-338), with
   * its two diagonals to draw. One request in flight; a newer one replaces a waiting one. */
  checkCrossover(edge: number, x: number, y: number, cb: (a: CrossoverAnswer) => void) {
    if (this.crossBusy) {
      this.crossNext = { edge, x, y, cb };
      return;
    }
    this.crossBusy = true;
    this.crossCb = cb;
    this.post({ kind: "previewCrossover", seq: ++this.seq, edge, x, y });
  }

  /** Drop a preview waiting to be sent, and the one in flight if it is not yet being worked out (a
   * drag was cancelled or finished, a run ended). */
  dropPreview() {
    this.previewNext = null;
    this.staleInFlight();
  }

  /** the preview in flight, and whether the worker was told it is no longer wanted (T-1039) */
  private previewSeq = 0;
  private previewStaleSent = false;
  private staleInFlight() {
    if (!this.previewBusy || this.previewStaleSent) return;
    this.previewStaleSent = true;
    this.post({ kind: "previewStale", seq: this.previewSeq });
  }

  private ask(msg: (seq: number) => ToClock, cb: (a: PreviewAnswer) => void) {
    this.lastPreviewAt = performance.now();
    if (this.previewBusy) {
      this.previewNext = { msg, cb };
      // not yet started, or waiting for masks, it is dropped for this one (T-1039)
      this.staleInFlight();
      return;
    }
    this.previewBusy = true;
    this.previewCb = cb;
    this.previewSeq = ++this.seq;
    this.previewStaleSent = false;
    this.post(msg(this.previewSeq));
  }

  /** T-1007, dev server only (`?debug=1`, `tw.client.debugPanic`): make the track model panic. */
  debugPanic(when: "now" | "read" | "afterEdit") {
    if (import.meta.env.DEV) this.post({ kind: "debugPanic", when });
  }

  /** The render epoch the renderer is drawing (start of the game hour). */
  setEpoch(epoch: number) {
    if (epoch === this.epoch) return;
    this.epoch = epoch;
    this.post({ kind: "epoch", epoch });
  }
}
