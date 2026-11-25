// Boarding and the hour's trips, and the snapshot demand runs on (T-2374).

import { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { mercK } from "../game/frame";
import { PERIODS } from "../game/periods";
import { headwaysFor, headwaysIn } from "../game/newLine";
import { BOATS, MEDIUM, type Boat } from "../game/boats";
import { NEIGHBOUR_FARE_M } from "../game/fareZones";
import { assignBays, BAY_LOOKBACK_S, profileTimes } from "../map/terminalBays";
import { NO_PAIRS, buildComplexes, pairKey, type Complexes } from "../game/complexes";
import { ComplexBuilder, complexChanges, complexesDiffer, type ComplexInputs } from "../game/complexBuilder";
import { buildingMembers, buildingsFrom, withBuildingLinks, type BuildingInput } from "../game/buildings";
import { BRT_STATION, IN_BUILDING } from "../game/upgrades";
import { type NewHeadways } from "./saveFile";
import { rowGeo, upgradesOf, type StationGeo } from "./stateRows";
import type { KeepInput } from "../game/types";
import type { ClockBus, ClockNode, EditOp, FerryPortIn, FerryRow, NetworkSnapshot } from "./protocol";
import { demandFares, tidyLineFares } from "./clockFares";
import { followUp, putServiceEditMs, REST_MS, whenNoneWaits } from "./clockFollowUp";
import { platformLineOf, stateRows, termEnds, version } from "./clockMain";
import { dayEntry, gameNow, hourEntry, moneyNow, putTodayGen, regionEntries, shapeHour, todayGen } from "./clockMoney";
import { patchMode } from "./clockPost";
import { api, city, epoch, fares, frame, frameSpec, heldCharges, ledgerDay, midLoad, newLineHeadway, newLineWeekendShare, packLngLat, send, SNAPSHOT_DEBOUNCE_MS, syncHolidays, timing, trackVersion } from "./clockState";

let snapVersion = 0;

export let snapTimer = 0;

/** Lines of the boarding account worked out per step: a few ms on New York's real network, so an
 * edit or the money tick never waits long behind new riders (T-164, notes/T-164.md). */
export const BOARDING_STEP_LINES = 3;
/** the transfers the boarding account holds (T-1810), null after its riders are cleared */
export let xferSet: Float32Array | null = null;
export const sameFloats = (a: Float32Array, b: Float32Array | null) => b !== null && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
let boardingTimer = 0;
/** T-3655: steps a task until this many ms of work: a timer between tasks waits at least 4 ms once nested (the
 * browsers' clamp), as long as a 3-line step's work on London's real save */
const STEP_SLICE_MS = 8;
/** time spent stepping since the last hour sent, and its longest task of steps (`?debug=1`) */
let prep = { ms: 0, max: 0 };
/** T-2404: when the last track preview was asked; while they come, the boarding account's steps
 * wait (one step of a few lines took 0.25-0.75 s on all of Japan, and a ghost asked meanwhile waits
 * for it), at most BOARD_HOLD_MS, after which it steps to the end */
export let previewAt = -Infinity;
const BOARD_HOLD_MS = 3000;
/** since when the steps have waited, 0 while they do not, -1 once they have waited their most */
let boardHeldAt = 0;
export function stepBoarding() {
  clearTimeout(boardingTimer);
  if (!api) return;
  const now = performance.now();
  if (now - previewAt < REST_MS && boardHeldAt >= 0) {
    if (!boardHeldAt) boardHeldAt = now;
    if (now - boardHeldAt < BOARD_HOLD_MS) {
      boardingTimer = setTimeout(stepBoarding, REST_MS - (now - previewAt) + 1) as unknown as number;
      return;
    }
    boardHeldAt = -1;
  }
  putTodayGen(todayGen + 1);
  const t0 = performance.now();
  let more: boolean;
  do more = api.boarding_step(BOARDING_STEP_LINES);
  while (more && performance.now() - t0 < STEP_SLICE_MS);
  const ms = performance.now() - t0;
  prep = { ms: prep.ms + ms, max: Math.max(prep.max, ms) };
  if (more) boardingTimer = setTimeout(stepBoarding, 0) as unknown as number;
  else {
    boardHeldAt = 0;
    sendTrips(true);
    // the lead covers the loads too, so the staged count's trains are drawn with them (T-1071)
    if (boardSince !== null) putServiceEditMs(performance.now() - boardSince);
    boardSince = null;
  }
}

/** T-3655: a service edit's boarding steps before its state. Past them the state and answer go first with the trips
 * without loads, and the steps go on after (`loadsLater`): an edit that moves many lines' service (a count on London's
 * full track, about 25 lines a kind of day) waited 0.7-2.2 s behind them. One step finishes most edits elsewhere (a
 * count on New York's real save), which keep their loads with their state. */
const BOARD_STEPS_BEFORE_STATE = 1;
/** when the service edit whose loads the steps now finish began, for its lead (`serviceEditMs`) */
let boardSince: number | null = null;
/** T-3655: whether the state of a service edit begun at `boardAt` goes before its trips' loads: the boarding account
 * is still not up to date after the edit's steps (`editSteps`). Its steps then go on, and the last sends both hours
 * with their loads; the page draws the staged state once they are here (`ClockState.loadsFollow`). */
/** a service edit's loads are still being worked out (its lead is timed when they go) */
export const loadsOwed = () => boardSince !== null;
export function loadsLater(boardAt: number | undefined): boolean {
  if (boardAt === undefined || !api?.has_riders() || !api.boarding_step_service(0)) return false;
  accountStale();
  boardSince = boardAt;
  return true;
}
/** an edit's boarding steps before its state (`later`'s `step`); `budget`: at most BOARD_STEPS_BEFORE_STATE. T-4240: only
 * the lines whose service changed (the edit's): lines out of date for new riders alone keep their last loads in its
 * trips (`sendSlice`) until `stepBoarding` reaches them, so an edit made while trips are calculated no longer waits for
 * every line's account (4.8 s of a station's 6.1 s on the Asia save) */
export function editSteps(budget: boolean): () => boolean {
  let n = 0;
  return () => {
    if (budget && n++ >= BOARD_STEPS_BEFORE_STATE) return false;
    putTodayGen(todayGen + 1);
    return api!.boarding_step_service(BOARDING_STEP_LINES);
  };
}

let nextHourTimer = 0;
/** T-3804: the hours whose whole slice (trips, and the boarding hour once there are riders) went out since the trips
 * were last sent anew. Only what sends the trips anew changes a slice (an edit's state with trips, new riders or
 * transfers, a calendar change, the boarding account's last step); a state without trips changes no train's run
 * (T-1780), so its version does not count. */
const sentHours = new Set<number>();
/** the next hour's slice after this one's (T-1810), its trips a task later and its boarding hour the task after
 * (T-3804: 50-200 ms together on a 2,647-station game), unless the trips are sent anew first or the account is still
 * being worked out (its last step sends both hours); begun again if an edit or a state came between the two */
function nextHourLater() {
  clearTimeout(nextHourTimer);
  // (T-4281: as `sendTrips`, after a waiting follow-up's state; an edit between the two tasks starts again here)
  const waits = () => !!followUp && !followUp.held;
  if (waits()) return whenNoneWaits(nextHourLater);
  const e = epoch, h = e + 3600;
  const due = () => !!api && e === epoch && !sentHours.has(h) && !accountStale();
  nextHourTimer = setTimeout(() => {
    if (waits()) return whenNoneWaits(nextHourLater);
    if (!due() || !tripsPrelude()) return;
    const at = [version, trackVersion], s = sliceTrips(h, rawTrips(), bayTimes());
    nextHourTimer = setTimeout(() => {
      if (version !== at[0] || trackVersion !== at[1]) return nextHourLater();
      if (due()) sendSlice(s);
    }, 0) as unknown as number;
  }, 0) as unknown as number;
}
/** T-3804: a new hour. Its slice went out at the last turn (as the next hour) or since, and is sent again only if it
 * did not; the next hour's goes a task later. A boarding account that is not up to date (new riders still being worked
 * through, a line's old pattern from before an edit just forgotten, a calendar change) is never worked out here, which
 * took up to 3 s on a 2,647-station game: its steps go on, edits between them, and the last one sends both hours. */
export function turnHour() {
  if (!api || !Number.isFinite(epoch)) return;
  for (const e of sentHours) if (e < epoch) sentHours.delete(e);
  api.set_now(gameNow());
  const changed = syncHolidays(), sent = sentHours.has(epoch);
  // (an hour never sent, as after a jump of the clock: its trains now, their loads once the steps end)
  if (accountStale()) { if (!sent) sendTrips(false, [epoch], false); }
  else if (changed || !sent) sendTrips(true);
  else nextHourLater();
}
/** T-3804: the boarding account is not up to date with the service and the riders; its steps are then set going
 * (their last one sends both hours anew), unless a stop click's follow-up is held: trips sent now would be of the
 * clicks' model ahead of their state (T-1407), and that follow-up steps the account itself before its trips */
function accountStale(): boolean {
  if (!api?.has_riders() || !api.boarding_step(0)) return false;
  if (!followUp?.held?.stops) {
    clearTimeout(boardingTimer);
    boardingTimer = setTimeout(stepBoarding, 0) as unknown as number;
  }
  return true;
}
/** The drawn trips and boarding slices of this hour and the next. `split` (T-1810): the next hour
 * a task later, so a preview asked meanwhile waits for one hour's slice only (~20 ms on shared New
 * York); dropped if the trips are sent again or riders come before it. `withBoard` false: no boarding hour, even with
 * riders (T-3804: the account is not up to date). */
export function sendTrips(split = false, hours?: number[], withBoard = true) {
  // T-1407: never trips of the model ahead of the state they are drawn on; T-4281: so while a follow-up waits they go after
  // its state (the account's last step sent them ahead of a click's, and ran that state whole first, 0.4-1 s on her Asia save)
  if (followUp && !followUp.held) return whenNoneWaits(() => sendTrips(split, hours, withBoard));
  if (!tripsPrelude()) return;
  if (!hours) sentHours.clear();
  if (split && !hours) nextHourLater();
  const rawAt = rawTrips(), times = bayTimes();
  for (const e of hours ?? (split ? [epoch] : [epoch, epoch + 3600])) sendSlice(sliceTrips(e, rawAt, times), withBoard);
}
/** what comes before any hour's slice; false if there is none to send */
function tripsPrelude(): boolean {
  // (T-2404: a held follow-up's blueprint changes no line's run, so its trips are the same; any other waits, `sendTrips`)
  clearTimeout(nextHourTimer);
  if (!api || !Number.isFinite(epoch)) return false;
  putTodayGen(todayGen + 1);
  api.set_now(gameNow());
  syncHolidays();
  timing.trips = 0;
  timing.board = 0;
  timing.bays = 0;
  return true;
}
/** `TrackApi.trips` by hour, each asked once (T-2980: the hour before each one's too, for where trains stand at the
 * ends) */
function rawTrips(): (e: number) => Float32Array {
  const raws = new Map<number, Float32Array>();
  return (e) => {
    let r = raws.get(e);
    if (!r) raws.set(e, (r = api!.trips(e)));
    return r;
  };
}
function bayTimes() {
  if (!termEnds.size) return null;
  const a = api!, t0 = performance.now();
  // T-4149: the phase tables the rows keep, brought up to date (reading every line's anew was 150-400 ms per edit and
  // per hour on a 2,647-station game); `?patch=walk` reads them from the model, as before, for timing against
  const tables = patchMode === "walk" ? (l: number) => a.line_phases(l) : stateRows.phaseTablesNow(a);
  if (patchMode === "check") {
    const bad = [...termEnds.keys()].filter((l) => !sameFloats(a.line_phases(l), tables(l)));
    if (bad.length) console.error(`T-4149 check: the kept phase tables of lines ${bad.slice(0, 8).join(", ")} differ from the model's`);
  }
  const out = profileTimes(a.profile_meta(), termEnds.keys(), tables);
  timing.bays = (timing.bays ?? 0) + performance.now() - t0;
  return out;
}
/** an hour's drawn trips: per trip, profile, departure, end and the departure its train takes next, relative to the
 * epoch (the end, T-206, and the next, T-245, go apart), and their places at the ends (T-2980) */
interface TripSlice { e: number; trips: Float32Array; ends: Float32Array; nexts: Float32Array; bays: Float32Array }
function sliceTrips(e: number, rawAt: (e: number) => Float32Array, times: ReturnType<typeof bayTimes>): TripSlice {
  const tT = performance.now();
  const raw = rawAt(e), n = raw.length / 4;
  const bays = assignBays(raw, e, times ? rawAt(e - BAY_LOOKBACK_S) : null, termEnds, (p) => times?.get(p));
  timing.trips += performance.now() - tT;
  const trips = new Float32Array(n * 2), ends = new Float32Array(n), nexts = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    trips[i * 2] = raw[i * 4];
    trips[i * 2 + 1] = raw[i * 4 + 1];
    ends[i] = raw[i * 4 + 2];
    nexts[i] = raw[i * 4 + 3];
  }
  return { e, trips, ends, nexts, bays };
}
/** An hour's slice to the page, with the boarding account's hour (T-164) once demand's riders are here: each trip's
 * loads and the platforms' crowds. The account is worked out again only for lines whose service or riders changed. */
function sendSlice(s: TripSlice, withBoard = true) {
  const a = api!, { e, trips, ends, nexts, bays } = s, riders = a.has_riders();
  if (withBoard && riders) {
    const t0 = performance.now();
    // (T-4240: a line out of date for new riders alone with its last loads, so an edit's trips never wait for every
    // line's account; the last of `stepBoarding`'s steps sends the hours again with theirs)
    const board = a.boarding_hour_kept(e);
    shapeHour(e, board); // T-284: when this hour's fares land, read before the slice is sent
    timing.board += performance.now() - t0;
    send({ kind: "trips", version, epoch: e, trips, ends, nexts, bays, board, boardMs: performance.now() - t0, redone: a.boarding_redone(), prepMs: prep.ms, prepMax: prep.max, ...(import.meta.env.DEV ? { timing: { trips: timing.trips, bays: timing.bays, board: timing.board } } : {}) }, [trips.buffer, ends.buffer, nexts.buffer, bays.buffer, board.buffer]);
    prep = { ms: 0, max: 0 };
  } else send({ kind: "trips", version, epoch: e, trips, ends, nexts, bays }, [trips.buffer, ends.buffer, nexts.buffer, bays.buffer]);
  if (withBoard || !riders) sentHours.add(e);
}

/** An edit's charge (US$M, `trains` of it for trains) into the money record of the hour and day. */
export function bookCharge(seq: number, charge: number, trains: number) {
  const d = dayEntry(ledgerDay(moneyNow()));
  const e = hourEntry(Math.floor(moneyNow() / 3600));
  d.trains += trains;
  d.build += charge - trains;
  e.trains += trains;
  e.build += charge - trains;
  for (const x of regionEntries("", d.day, e.hour)) { x.trains += trains; x.build += charge - trains; }
  heldCharges.set(seq, { day: d.day, hour: e.hour, build: charge - trains, trains });
}

// ---------------------------------------------------------------- the snapshot demand runs on

/** T-1380: the complexes of the snapshot's built stations and of every station (`complexOfNodes`),
 * each kept up to date from the stations that changed since it was last asked for. */
const snapshotComplexes = new ComplexBuilder();
const nodeComplexes = new ComplexBuilder();
export const groundK = (y: number) => (frame.kind === "merc" ? mercK(y) : 1);
export type { StationGeo };
function stationGeo(a: TrackApi, ni: ArrayLike<number>, i: number): StationGeo {
  const line = platformLineOf(a, ni[i], ni[i + 1], ni[i + 2], ni[i + 7]);
  return { id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7], ...(line ? { line } : {}) };
}
/** T-2217: the station buildings as the track model lists them; "" for a name nobody gave (T-3750) */
function buildingsOf(a: TrackApi) {
  return buildingsFrom(a.buildings_info(), (id) => (a.building_named(id) ? a.building_name(id) : ""));
}
/** The station buildings as shown: one nobody named shows its first member's complex's name (`StateRows.complexName`;
 * T-4300: its members kept with the complexes, `buildingFirst`, as the last rows read them), the part before any slash,
 * as the track model's own default. */
export function shownBuildings(a: TrackApi) {
  const blds = buildingsOf(a);
  if (!blds.some((b) => !b.name)) return blds;
  return blds.map((b) => {
    if (b.name) return b;
    const m = stateRows.buildingFirst(b.id);
    const name = (m !== undefined ? stateRows.complexName(m) : undefined) ?? a.building_name(b.id);
    return { ...b, name: name.split("/")[0].trim() };
  });
}
/** T-3759: rename a station as its panel does: an unbound one takes the name alone (and, alone in its
 * complex, as its part in the complex's name too, so a complex it forms later is called what the player
 * saw); a bound one renames its complex. */
export function renameStation(a: TrackApi, node: number, name: string): number {
  if (a.station_own(node)) return a.set_station_own(node, name, stateRows.complexSize(node) <= 1);
  return renameComplex(a, node, name);
}
/** T-3759: rename the complex station `node` is in: its name source (`nameSource`, the worker's last rows)
 * takes the name as the player's, so every bound station shows it; unbound ones keep their own. */
export function renameComplex(a: TrackApi, node: number, name: string): number {
  if (stateRows.complexSize(node) <= 1 && a.station_own(node)) return a.set_station_own(node, name, true);
  return a.type_station_name(stateRows.headOf(node) ?? node, name);
}
/** T-4259 (omi-key's PR 35): the line panel's Rename stations: each station renamed as `renameStation` does (an
 * unbound one alone, a bound one's complex through its head), all as one undo (`TrackApi.rename_stations`). */
export function renameStations(a: TrackApi, nodes: number[], names: string[]): number {
  const to: number[] = [], as: string[] = [], kinds: number[] = [];
  nodes.forEach((node, i) => {
    const name = names[i]?.trim();
    if (!name) return;
    if (a.station_own(node)) to.push(node), kinds.push(stateRows.complexSize(node) <= 1 ? 1 : 0);
    else to.push(stateRows.headOf(node) ?? node), kinds.push(2);
    as.push(name);
  });
  return to.length ? a.rename_stations(new Uint32Array(to), as, new Uint8Array(kinds)) : 0;
}
const NO_KEEP = 0xffffffff;
/** T-4080: name a bus stop as its panel does: the name goes to the kept stop its place already has (the first of `stops` with
 * one), else a new one at (x, y); an empty name clears it. */
export function renameBusStop(a: TrackApi, stops: number[], x: number, y: number, name: string): number {
  const id = stops.map((s) => a.bus_stop_keep(s)).find((k) => k !== NO_KEEP);
  return a.name_keep(id ?? NO_KEEP, x, y, name);
}
let keepsRead: { epoch: number; list: KeepInput[] } | null = null;
/** T-4080: the kept stops that have a name, read again when the sim says the kept stops changed (`keep_epoch`, which a load
 * starts over: `forgetKeeps`) */
export function namedKeeps(a: TrackApi): KeepInput[] {
  const epoch = a.keep_epoch();
  if (keepsRead?.epoch === epoch) return keepsRead.list;
  const flat = a.keeps(), list: KeepInput[] = [];
  for (let k = 0; k + 2 < flat.length; k += 3) {
    const name = a.keep_name(flat[k]);
    if (name) list.push({ id: flat[k], x: flat[k + 1], y: flat[k + 2], name });
  }
  keepsRead = { epoch, list };
  return list;
}
export const forgetKeeps = () => { keepsRead = null; };
/** T-3793: an edit that changes only what a station or line is called (asked before it is applied): no train's run
 * changes, and the network drawn has no names in it. T-4147: and a new line with no stops. */
export function nameOnly(op: EditOp): boolean {
  switch (op.op) {
    case "renameStation": case "renameComplex": case "renameStations": case "bindStation": case "bindAll": return true;
    // T-4080: a name for a kept stop that exists moves no stop; making or clearing one can (lines near it are planned again)
    case "renameBusStop": return !!op.name && op.stops.some((s) => api!.bus_stop_keep(s) !== NO_KEEP);
    case "stationProps": return op.platform === stateRows.nodeRow(op.node)?.platform;
    // T-4147: a line with no stops draws nothing and carries no train, so its list entry needs only the rows
    case "addLine": return op.stops.length === 0;
    case "lineLook": return op.colour === undefined && op.letter === undefined;
    default: return false;
  }
}
let apartFlat = "", apartKeys: ReadonlySet<string> = NO_PAIRS;
/** T-4048: the pairs the player separated, as `pairKey`s; the same set while they stay the same */
export function apartNow(a: TrackApi): ReadonlySet<string> {
  const fl = a.station_apart(), k = fl.join();
  if (k !== apartFlat) {
    apartFlat = k;
    const s = new Set<string>();
    for (let i = 0; i + 1 < fl.length; i += 2) s.add(pairKey(String(fl[i]), String(fl[i + 1])));
    apartKeys = s.size ? s : NO_PAIRS;
  }
  return apartKeys;
}
const NO_LINKS: readonly (readonly [string, string])[] = [];
let linksFlat: ArrayLike<number> = [], linkPairs = NO_LINKS;
let bldFlat: ArrayLike<number> = [], bldShapes: readonly BuildingInput[] = [];
/** T-4300: what the complexes are made of besides the stations (the player's links T-197, the pairs separated T-4048, the
 * station buildings' outlines T-2217), each the same object while the model's numbers for it are the same */
export function complexInputs(a: TrackApi): ComplexInputs {
  const fl = a.station_links();
  if (!sameNums(fl, linksFlat)) {
    linksFlat = fl;
    const p: [string, string][] = [];
    for (let k = 0; k + 1 < fl.length; k += 2) p.push([String(fl[k]), String(fl[k + 1])]);
    linkPairs = p.length ? p : NO_LINKS;
  }
  const bi = a.buildings_info();
  if (!sameNums(bi, bldFlat)) { bldFlat = bi; bldShapes = buildingsFrom(bi, () => ""); }
  return { links: linkPairs, apart: apartNow(a), buildings: bldShapes, kAt: groundK, frame: frame.kind };
}
function sameNums(p: ArrayLike<number>, q: ArrayLike<number>): boolean {
  if (p.length !== q.length) return false;
  for (let i = 0; i < p.length; i++) if (p[i] !== q[i]) return false;
  return true;
}

/** The complexes of `geo` (each station once) kept by `builder` from the stations that changed since its last call. */
function complexesOf(builder: ComplexBuilder, geo: StationGeo[]): Complexes {
  const c = complexInputs(api!);
  if (patchMode === "0") return buildComplexes(geo, withBuildingLinks(c.links, c.buildings, geo, groundK), groundK, c.apart);
  const cx = builder.update(geo, c.links, groundK, c.frame, c.apart, c.buildings);
  if (patchMode === "check") {
    const d = complexesDiffer(cx, geo, c.links, groundK, c.apart, c.buildings);
    if (d) console.error(`T-1380 check: the complexes patched from changes differ from a whole build (${d})`);
  }
  return cx;
}

export function buildSnapshot(): NetworkSnapshot {
  const a = api!;
  const ni = a.nodes_info();
  const stations: NetworkSnapshot["stations"] = [];
  const geo: StationGeo[] = [];
  const ups = upgradesOf(a);
  // T-3740: the stations' fare zones, where any has one
  const fz = new Map<number, number>(), zl = fares.zones ? a.station_fares() : [];
  for (let k = 0; k + 1 < zl.length; k += 2) fz.set(zl[k], zl[k + 1]);
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] > 0 && ni[i + 8] === 1) {
      const u = ups.get(ni[i]), z = fz.get(ni[i]);
      // T-4041: a tram stop on street track (`TrackKind::StreetTram`) at ground
      const street = ni[i + 3] === 0 && a.node_kind(ni[i]) === 2;
      stations.push({ id: ni[i], name: stateRows.nameOf(a, ni[i], frame), x: ni[i + 1], y: ni[i + 2], level: ni[i + 3] as any, platform: ni[i + 7], ...(u ? { upgrades: u } : {}), ...(z ? { fare: z } : {}), ...(street ? { street: true as const } : {}) });
      geo.push(stationGeo(a, ni, i));
    }
  }
  // T-161: each constructed station's complex, among constructed stations only (a blueprint
  // platform carries nobody, so it joins nothing), with the player's links (T-197); numbered 0, 1, ... by their first
  // station, as demand takes them (workers/demandClient.ts numbers a station of none past them)
  const cx = complexesOf(snapshotComplexes, geo), dense = new Map<number, number>();
  for (const s of stations) {
    const c = cx.of.get(String(s.id));
    if (c === undefined) continue;
    let d = dense.get(c);
    if (d === undefined) dense.set(c, (d = dense.size));
    s.complex = d;
  }
  // T-2220: a change inside a station building is quicker; the bit is the snapshot's alone, never saved
  const blds = buildingsOf(a);
  if (blds.length) {
    const inside = new Set(buildingMembers(blds, geo, groundK).flat());
    for (const s of stations) if (inside.has(String(s.id))) s.upgrades = (s.upgrades ?? 0) | IN_BUILDING;
  }
  // T-3378: a wait at a BRT station counts as in a waiting room; this bit too is the snapshot's alone
  const brt = a.brt_stations();
  if (brt.length) {
    const on = new Set(brt);
    for (const s of stations) if (on.has(Number(s.id))) s.upgrades = (s.upgrades ?? 0) | BRT_STATION;
  }
  // T-1281: the links are passages, walked faster; between constructed stations only
  const built = new Set(geo.map((g) => g.id));
  const passages = complexInputs(a).links.filter(([p, q]) => built.has(p) && built.has(q)).map(([p, q]) => [Number(p), Number(q)] as [number, number]);
  // T-4048: the pairs the player separated, between constructed stations only
  const fa = a.station_apart(), apart: [number, number][] = [];
  for (let k = 0; k + 1 < fa.length; k += 2) if (built.has(String(fa[k])) && built.has(String(fa[k + 1]))) apart.push([fa[k], fa[k + 1]]);
  const li = a.lines_info();
  const lines: NetworkSnapshot["lines"] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue; // running lines only
    const id = li[i];
    const ferry = a.is_ferry(id);
    // T-3892: a boat holds its size's cars' riders, and its fares take its size's multiple
    const boat = (ferry ? a.ferry_boat(id) : MEDIUM) as Boat;
    const bus = a.is_bus(id) ? busSnapshot(a, id, stations) : null;
    const hopS: number[][][] = [], dwellS: number[][][] = [];
    const span = a.line_span(id), back = a.line_back_stops(id);
    for (let run = 0; run < 2; run++) {
      hopS.push([]);
      dwellS.push([]);
      // T-3542: per demand period (its slot's times)
      for (let q = 0; q < PERIODS.length; q++) {
        const arr = a.stop_times_in(id, run, q), dep = a.stop_departures_in(id, run, q);
        hopS[run].push(Array.from({ length: arr.length - 1 }, (_, k) => arr[k + 1] - dep[k]));
        dwellS[run].push(Array.from(arr, (t, k) => (k === 0 || k === arr.length - 1 ? 0 : dep[k] - t)));
      }
    }
    lines.push({
      id, name: a.line_name(id), stops: [...a.line_run_stops(id)], ...(back.length ? { backStops: [...back] } : {}), tph: Array.from(a.line_demand_tph(id)), hopS, dwellS,
      hopKm: [0, 1].map((run) => { const s = a.line_stop_s(id, run); return Array.from({ length: s.length - 1 }, (_, k) => (s[k + 1] - s[k]) / 1000); }),
      ...(!bus && !ferry && a.line_fast(id) ? { fast: true } : {}),
      ...(ferry ? { ferry: true } : {}),
      ...(boat !== MEDIUM ? { boat: boat as 0 | 2 } : {}),
      // T-3915: and a BRT line's the BRT multiple
      ...(bus && a.bus_brt(id) ? { brt: true } : {}),
      ...(!bus && !ferry && a.line_tram(id) ? { tram: true } : {}),
      // T-3758: what fare zones read, only while the fares have areas
      ...(fares.zones && a.line_fare_dist(id) ? { fareDist: true } : {}),
      ...(fares.zones && !bus && !ferry ? passFaresOf(a, id) : {}),
      turnaroundS: li[i + 20], cars: bus ? BUS_DEMAND_CARS : ferry ? BOATS[boat].cars : li[i + 21],
      ...(bus ? { catches: bus } : {}),
      // T-798: a circle's riders ride on round through its first stop, standing the dwell there
      ...(a.line_loop(id) === 1 ? { wrapS: Math.max(1, li[i + 19]) } : {}),
      ...(span.some((v) => v !== 1) ? { span: Array.from(span) } : {}),
    });
  }
  if (fz.size) busStopFares(stations);
  snapStations = stationsKey(ni);
  tidyLineFares(a);
  return { version: ++snapVersion, city, frame: frameSpec(), stations, lines, fare: demandFares(), ...(passages.length ? { passages } : {}), ...(apart.length ? { apart } : {}) };
}

/** T-3740: a bus stop that is no station takes the fare zone of the nearest station with one within
 * `NEIGHBOUR_FARE_M` on the ground (as a new station does, sim `neighbour_fare`), else none, so it pays by
 * distance. Only real stations count, never another stop, so zones don't creep along a route. Worked out for each
 * snapshot, never stored. */
function busStopFares(stations: NetworkSnapshot["stations"]) {
  const cell = NEIGHBOUR_FARE_M * 2, grid = new Map<string, NetworkSnapshot["stations"]>();
  const key = (i: number, j: number) => `${i},${j}`;
  for (const s of stations) {
    if (s.bus || !s.fare) continue;
    const k = key(Math.floor(s.x / cell), Math.floor(s.y / cell));
    const list = grid.get(k);
    if (list) list.push(s); else grid.set(k, [s]);
  }
  if (!grid.size) return;
  for (const b of stations) {
    if (!b.bus) continue;
    const r = NEIGHBOUR_FARE_M * groundK(b.y), i0 = Math.floor((b.x - r) / cell), i1 = Math.floor((b.x + r) / cell), j0 = Math.floor((b.y - r) / cell), j1 = Math.floor((b.y + r) / cell);
    let best = r * r, code = 0;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      for (const s of grid.get(key(i, j)) ?? []) {
        const d = (s.x - b.x) ** 2 + (s.y - b.y) ** 2;
        if (d <= best) { best = d; code = s.fare!; }
      }
    }
    if (code) b.fare = code;
  }
}

/** T-3758: per run and hop, the fare zone codes of the stations line `id` passes without stopping; nothing where
 * it passes none with a zone. */
function passFaresOf(a: TrackApi, id: number): { passFares?: number[][][] } {
  let any = false;
  const runs = [0, 1].map((run) => {
    const v = a.line_pass_fares(id, run), out: number[][] = [];
    for (let k = 0; k < v.length; k += 1 + v[k]) {
      out.push(Array.from(v.subarray(k + 1, k + 1 + v[k])));
      if (v[k]) any = true;
    }
    return out;
  });
  return any ? { passFares: runs } : {};
}

/** T-1464: a bus line's row (`ClockBus`); its stops that are no station join `nodes` once each,
 * flagged `bus`, so every stop of every line has a node row. */
export function busRow(a: TrackApi, id: number, nodes: ClockNode[]): ClockBus {
  if (a.is_ferry(id)) return ferryRow(a, id);
  const r = a.bus_route(id), st = a.bus_stops(id), rs = a.bus_reach(id);
  // T-2982: a stop at a station is on the route beside it; the station's own place, for the transfer's dashes
  const reachAt = new Map<number, [number, number]>();
  for (let k = 0; k + 2 < rs.length; k += 3) reachAt.set(rs[k], [rs[k + 1], rs[k + 2]]);
  const stops: ClockBus["stops"] = [];
  for (let k = 0; k + 7 < st.length; k += 8) {
    const [lng, lat] = packLngLat(st[k + 1], st[k + 2]);
    const s: ClockBus["stops"][number] = { id: st[k], x: st[k + 1], y: st[k + 2], lng, lat, kind: st[k + 4], station: st[k + 5] };
    const sxy = s.station >= 0 ? reachAt.get(s.station) : undefined;
    if (sxy) s.at = packLngLat(sxy[0], sxy[1]);
    stops.push(s);
    if (s.station < 0 && !nodes.some((n) => n.id === s.id)) {
      nodes.push({ id: s.id, x: s.x, y: s.y, lng, lat, level: 0, ports: 0, builtPorts: 0, flying: false, platform: 0, stationBuilt: r[0] === 1, stationOff: false, upgrades: 0, name: "", heading: 0, bus: true });
    }
  }
  // the route is x, y, kind per point (T-1465); a long road-following one is worth converting once
  const route: number[] = [], kinds: number[] = [], street = Array.from(a.bus_street(id));
  for (let k = 1; k + 2 < r.length; k += 3) {
    route.push(r[k], r[k + 1]);
    kinds.push(r[k + 2]);
  }
  const ll = (xy: number[]) => { const out: number[] = []; for (let k = 0; k + 1 < xy.length; k += 2) out.push(...packLngLat(xy[k], xy[k + 1])); return out; };
  const had = busRouteLL.get(id);
  const same = had && had.route.length === route.length && had.route.every((v, i) => v === route[i]);
  const routeLL = same ? had.ll : ll(route);
  if (!same) busRouteLL.set(id, { route, ll: routeLL });
  const hand = a.bus_hand(id), reach: ClockBus["reach"] = [];
  if (hand) {
    for (let k = 0; k + 2 < rs.length; k += 3) {
      const [lng, lat] = packLngLat(rs[k + 1], rs[k + 2]);
      reach.push({ node: rs[k], x: rs[k + 1], y: rs[k + 2], lng, lat });
    }
  }
  const [townM, newM, laneM] = a.bus_lanes(id);
  // T-3944: the lanes' runs with their points as longitude, latitude
  const marks = a.bus_lane_marks(id), lanes: number[] = [];
  for (let k = 0; k + 1 < marks.length; k += 2 + 2 * marks[k + 1]) lanes.push(marks[k], marks[k + 1], ...ll(Array.from(marks.subarray(k + 2, k + 2 + 2 * marks[k + 1]))));
  return { built: r[0] === 1, route, kinds, roadM: a.bus_road_m(id), stops, street, spacing: a.bus_spacing(id), hand, brt: a.bus_brt(id), brtBp: a.bus_brt_bp(id), lanes, townM: townM ?? 0, newM: newM ?? 0, laneM: laneM ?? 0, laneLoad: [...a.bus_lane_load(id)], reach, routeLL, streetLL: ll(street) };
}
/** a bus route's longitude, latitude pairs by line, until the route changes */
const busRouteLL = new Map<number, { route: number[]; ll: number[] }>();

/** T-2194: a ferry line's row: a bus line's (`ClockBus`), its kinds the ferry's, with its speeds and
 * each port's landing and berth (`FerryRow`). Its ports are stations, so no node joins `nodes`. */
function ferryRow(a: TrackApi, id: number): ClockBus {
  const r = a.ferry_route(id), st = a.bus_stops(id), stops: ClockBus["stops"] = [];
  for (let k = 0; k + 7 < st.length; k += 8) {
    const [lng, lat] = packLngLat(st[k + 1], st[k + 2]);
    stops.push({ id: st[k], x: st[k + 1], y: st[k + 2], lng, lat, kind: st[k + 4], station: st[k + 5] });
  }
  const route: number[] = [], kinds: number[] = [], speeds: number[] = [];
  for (let k = 1; k + 3 < r.length; k += 4) {
    route.push(r[k], r[k + 1]);
    kinds.push(r[k + 2]);
    speeds.push(r[k + 3]);
  }
  const had = busRouteLL.get(id);
  const same = had && had.route.length === route.length && had.route.every((v, i) => v === route[i]);
  let routeLL = had?.ll ?? [];
  if (!same) {
    routeLL = [];
    for (let k = 0; k + 1 < route.length; k += 2) routeLL.push(...packLngLat(route[k], route[k + 1]));
    busRouteLL.set(id, { route, ll: routeLL });
  }
  // each port's landing is its stop (`ferry::plan`, in route order); its berth the kind 1 point
  const ports: FerryRow["ports"] = [];
  const nodes = a.line_stops(id);
  let p = 0;
  kinds.forEach((kd, i) => {
    if (kd !== 1 || p >= nodes.length) return;
    const node = nodes[p], s = stops[p++];
    const berth: [number, number] = [routeLL[2 * i], routeLL[2 * i + 1]];
    ports.push({ node, landing: s && s.id === node ? [s.lng, s.lat] : berth, berth });
  });
  return { built: r[0] === 1, route, kinds, roadM: a.ferry_m(id), stops, street: [], spacing: 0, hand: false, brt: false, brtBp: false, lanes: [], townM: 0, newM: 0, laneM: 0, laneLoad: [], reach: [], routeLL, streetLL: [], ferry: { speeds, ports, boat: a.ferry_boat(id) as Boat } };
}

/** T-2194: a ferry line's ports as the network's nodes, new ones added first as blueprint ports; null
 * when one is refused (those added before it are taken back). */
export function addPorts(a: TrackApi, ports: FerryPortIn[]): Uint32Array | null {
  const out: number[] = [];
  let added = 0;
  for (const p of ports) {
    if ("node" in p) { out.push(p.node); continue; }
    if ("again" in p) { out.push(out[p.again]); continue; }
    if (a.add_port(p.x, p.y, p.bx, p.by, p.name) !== 0) {
      for (; added > 0; added--) a.undo();
      return null;
    }
    added++;
    out.push(a.last_port());
  }
  return Uint32Array.from(out);
}

/** A new line's edits' answer: the first's, and where more than one edit made it, every edit's charges. */
type Made = { r: number; line: number; charge?: number; trainCharge?: number };

/** T-3767: a new line (`r` its edit's answer) in a game of four or five rows with a default of its kind
 * (`kind`) whose midday or afternoon differs from its high (the sim starts them at high's): every row's
 * headway, as a second edit. */
export function newLineRows(a: TrackApi, r: number, line: number, kind: keyof NewHeadways, km = 0): Made {
  if (r !== 0) return { r, line };
  const n = a.line_period_count(line), h = n > 3 ? headwaysFor(kind, newLineHeadway, n, km) : null;
  const three = headwaysFor(kind, newLineHeadway, 3, km);
  if (!h || !three || headwaysIn(three, n).every((s, i) => s === h[i])) return { r, line };
  const charge = a.last_charge(), trainCharge = a.last_train_charge();
  if (a.set_line_row_headways(line, Uint32Array.from(h)) !== 0) return { r, line, charge, trainCharge };
  return { r, line, charge: charge + a.last_charge(), trainCharge: trainCharge + a.last_train_charge() };
}

/** T-3588, T-3703: a new bus or ferry line at the weekend percent "Apply to all lines" set for new lines, as one
 * more edit, as a rail line's; every edit's charges. */
export function newLineWeekends(a: TrackApi, made: Made): Made {
  if (made.r !== 0 || newLineWeekendShare === null || newLineWeekendShare === 100) return made;
  const charge = made.charge ?? a.last_charge(), trainCharge = made.trainCharge ?? a.last_train_charge();
  if (a.set_line_weekend_share(made.line, newLineWeekendShare) !== 0) return { ...made, charge, trainCharge };
  return { ...made, charge: charge + a.last_charge(), trainCharge: trainCharge + a.last_train_charge() };
}

/** T-1475: a new bus line's name, "Bus n" with n the sim's next free line number (T-2194: "Ferry n"). */
export function nextBusName(a: TrackApi, word = "Bus "): string { // i18n-ignore: a line's saved name
  const taken = new Set<string>();
  const info = a.lines_info();
  for (let i = 0; i < info.length; i += 22) taken.add(a.line_name(info[i]));
  for (;;) {
    const name = a.next_line_name().replace(/^Line /, word);
    if (!taken.has(name)) return name;
  }
}

/** T-1464: `TrackApi.bus_quote` with each stop's and street stop's longitude and latitude after
 * its x, y: [road m, US$M, 1 if it makes a line, n, then x, y, kind, lng, lat per stop, then m and
 * x, y, lng, lat per street stop, then the run count and start m, end m, km/h, 1 on a trunk road per run]. */
export function busQuoteLL(a: TrackApi, line: number, route: number[]): Float64Array {
  const q = a.bus_quote(line, Float64Array.from(route)), out = [q[0], q[1], q[2], q[3]];
  let k = 4;
  for (let i = 0; i < q[3]; i++, k += 3) out.push(q[k], q[k + 1], q[k + 2], ...packLngLat(q[k], q[k + 1]));
  const m = q[k++];
  out.push(m);
  for (let i = 0; i < m; i++, k += 2) out.push(q[k], q[k + 1], ...packLngLat(q[k], q[k + 1]));
  for (; k < q.length; k++) out.push(q[k]);
  return Float64Array.from(out);
}

/** T-2919: a bus as demand's cars: a third of a car, 53 at the rated load and 80 at crush (`BUS_CARS`
 * in sim/src/demand/network.rs, which knows a bus line by this figure). */
const BUS_DEMAND_CARS = 1 / 3;

/** T-1464: a running bus line's stops that are no station, added to the snapshot's stations once
 * (two lines' stops at one crossing are two stations, merged by demand), and each stop's stretch of
 * route for demand's corridor. */
function busSnapshot(a: TrackApi, id: number, stations: NetworkSnapshot["stations"]): number[][] {
  const st = a.bus_stops(id), have = new Set(stations.map((s) => s.id));
  for (let k = 0; k + 7 < st.length; k += 8) {
    if (st[k + 5] >= 0 || have.has(st[k])) continue;
    stations.push({ id: st[k], name: "", x: st[k + 1], y: st[k + 2], level: 0, platform: 0, bus: true });
    have.add(st[k]);
  }
  const c = a.bus_catches(id), out: number[][] = [];
  for (let k = 0; k < c.length; ) {
    const n = c[k];
    out.push(Array.from(c.subarray(k + 1, k + 1 + 2 * n)));
    k += 1 + 2 * n;
  }
  return out;
}

/** T-1368: the built stations as the last snapshot had them. An edit that changes only them (a
 * station removed after its line, one built or moved on no line) changes no line's service for
 * demand, yet demand's reach and nearby markets read them. */
export let snapStations = "";
export function stationsKey(ni = api!.nodes_info()): string {
  const out: (number | string)[] = [];
  // T-1805: a station's heading is its axis here; a branch splitting the track beside it can turn
  // the heading round, and a few microradians off (it follows the node's first edge end), which
  // moves nothing demand reads: the axis as its doubled angle, to a milliradian
  const axis = (h: number) => `${Math.round(Math.cos(2 * h) * 1e3)};${Math.round(Math.sin(2 * h) * 1e3)}`;
  for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0 && ni[i + 8] === 1) out.push(ni[i], ni[i + 1], ni[i + 2], ni[i + 3], ni[i + 7], axis(ni[i + 9]));
  // T-3740: and their fare zones, where the fares have areas (the snapshot's only then)
  return `${out.join(",")}|${api!.station_upgrades().join(",")}${fares.zones ? `|${api!.station_fares().join(",")}` : ""}`;
}

/** T-1168, T-4300: each node's station complex for `line_related`, the drawn network and the schematic, over every
 * station (blueprints and ports too, unlike the snapshot's: a planned express is related before it is built),
 * 0xffffffff for none; a new array for each change, which only the stations whose complex changed differ in. The
 * complexes follow the state's rows by the reads' changes (`nodeRead`), with the stations edits since the last read
 * touched taken from the model (`nodeLive`, given again from the rows once a read has them). */
let nodeRead = -1;
const nodeLive = new Set<number>();
let nodeArr: { cx: Complexes; of: Uint32Array } | null = null;
const NO_COMPLEX = 0xffffffff;
export function complexOfNodes(): Uint32Array {
  const a = api!, c = complexInputs(a);
  // T-4299: between an edit and its state the stations it touched are read from the model in their place (review21);
  // after a read that left regions for the next one (a rows state's focus: review25) every station is
  // (?patch=0 has no builder to patch: every station, as complexesOf builds them)
  const t = a.peek_touched(), whole = stateRows.behind || t[0] === 0xffffffff || patchMode === "0";
  let cx: Complexes;
  if (whole) {
    const ni = a.nodes_info(), geo: StationGeo[] = [];
    for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0) geo.push(stationGeo(a, ni, i));
    cx = complexesOf(nodeComplexes, geo);
    nodeRead = -1;
    nodeLive.clear();
  } else {
    const touched = new Set(t.subarray(1 + t[0])), since = nodeRead >= 0 && nodeComplexes.ready(c.frame) ? stateRows.changedSince(nodeRead, 1) : null;
    const changes: [string, StationGeo | null][] = [];
    // the stations the reads since changed, and those taken from the model before that no edit since touched, as the rows
    // have them (every one when that is not known)
    const rows = since ? null : complexesOf(nodeComplexes, stateRows.modelNodes().filter((r) => r.platform > 0).map(rowGeo));
    if (since) {
      const put = (id: number) => { const r = stateRows.nodeNow(id); changes.push([String(id), r && r.platform > 0 && !r.bus ? rowGeo(r) : null]); };
      for (const id of since) if (!touched.has(id)) put(id);
      for (const id of nodeLive) if (!since.has(id) && !touched.has(id)) put(id);
    }
    nodeRead = stateRows.lastRead;
    nodeLive.clear();
    // (a touched node the model has no numbers for is gone)
    if (touched.size) {
      const ids = Uint32Array.from(touched), ni = a.nodes_info_of(ids), now = new Map<number, StationGeo>();
      for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0) now.set(ni[i], stationGeo(a, ni, i));
      for (const id of ids) { changes.push([String(id), now.get(id) ?? null]); nodeLive.add(id); }
    }
    cx = rows && !changes.length ? rows : nodeComplexes.patch(changes, c.links, c.kAt, c.frame, c.apart, c.buildings);
  }
  if (patchMode === "check" && !whole) {
    const ni = a.nodes_info(), geo: StationGeo[] = [];
    for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0) geo.push(stationGeo(a, ni, i));
    const d = complexesDiffer(cx, geo, c.links, c.kAt, c.apart, c.buildings);
    if (d) console.error(`T-4300 check: every station's complexes kept from the rows and the edits since differ from a whole build of the model's (${d})`);
  }
  return nodeArray(cx);
}

/** each node's complex in `cx` as an array: the last one copied with the stations `complexChanges` names laid in */
function nodeArray(cx: Complexes): Uint32Array {
  if (nodeArr?.cx === cx) return nodeArr.of;
  const ch = nodeArr ? complexChanges(nodeArr.cx, cx) : null;
  let of: Uint32Array;
  if (ch && nodeArr) {
    let top = nodeArr.of.length - 1;
    for (const id of ch) top = Math.max(top, Number(id));
    if (top < nodeArr.of.length) of = nodeArr.of.slice();
    else { of = new Uint32Array(top + 1).fill(NO_COMPLEX); of.set(nodeArr.of); }
    for (const id of ch) of[Number(id)] = cx.of.get(id) ?? NO_COMPLEX;
  } else {
    let top = 0;
    for (const id of cx.of.keys()) top = Math.max(top, Number(id));
    of = new Uint32Array(top + 1).fill(NO_COMPLEX);
    for (const [id, k] of cx.of) of[Number(id)] = k;
  }
  if (patchMode === "check") {
    const bad = [...cx.of].find(([id, k]) => of[Number(id)] !== k) ?? Array.from(of).findIndex((k, n) => k !== NO_COMPLEX && cx.of.get(String(n)) !== k);
    if (bad !== -1) console.error(`T-4300 check: each node's complex laid from changes differs from the complexes (${bad})`);
  }
  nodeArr = { cx, of };
  return of;
}

export function scheduleSnapshot() {
  clearTimeout(snapTimer);
  if (midLoad) return;
  snapTimer = setTimeout(() => { if (!midLoad) send({ kind: "snapshot", snapshot: buildSnapshot() }); }, SNAPSHOT_DEBOUNCE_MS) as unknown as number;
}
export function putPreviewAt(v: typeof previewAt) { previewAt = v; }
export function putXferSet(v: typeof xferSet) { xferSet = v; }
