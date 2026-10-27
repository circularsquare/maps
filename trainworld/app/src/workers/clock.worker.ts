// The clock worker (SPEC 6.5, 7; T-025): owns the network (the track model's `TrackApi`, T-040),
// applies edits, runs profiles and the capacity pass, and sends the main thread what it draws:
// track polylines, keyframes (shared phase tables) and each game hour's trips (the hourly
// re-base, SPEC 7). After a service change it also builds the network snapshot demand runs on,
// debounced 0.5 s (SPEC 6.5). notes/T-025.md.
//
// Every edit answers with an "edited" message (and, if it applied, a new "state"); the state
// holds the whole network. Fine for a city; per-tile updates are follow-up work (notes/T-025.md).

import init, { TrackApi } from "../wasm/track/sim"; // the track model only (T-051)
import { anchor, makeAnchor, setAnchor, toLocal } from "../map/geo";
import { CITIES, cityById, cityWithin, HOME_CITY, NEW_GAME_FRAME, WORLD_ID, zoneBox, zonesTouching } from "../game/cities";
import { frameKindName, frameKindNumber, M_PER_DEG, makeFrame, mercK, type Frame, type FrameSpec } from "../game/frame";
import { PERIODS } from "../game/periods";
import { letterOf } from "../game/lineLetter";
import { faresOf, START_FARE, type Fares } from "../game/fareCurve";
import { FARE_BASELINE, incomeScale, tidyScale } from "../game/income";
import { SHADES } from "../game/shadeTable";
import { DAY_KINDS, dayKindOf } from "../game/days";
import { kindOfIn, type HolidayData } from "../game/holidayCalendar";
import { shiftAt, timeZoneTable } from "../game/localTime";
import { between, lagged, runLag, strengths, table, type KindOf, type PerHour } from "../game/moneyTiming";
import { networkFromState, newNetCache, runsTrains, type NetworkBuffers } from "../map/network";
import { StrokeSender, strokesOfPieces, type StrokeKind, type StrokeMode } from "../map/strokePost";
import { readOutlinesForPlaces } from "../map/coarseOutlines";
import { buildComplexes, type Complexes } from "../game/complexes";
import { ComplexBuilder } from "../game/complexBuilder";
import { UPGRADE, UPGRADE_BITS } from "../game/upgrades";
import { saveNetKey } from "./demandCache";
import { currentZones, decodeSave, encodeSave, isNortheastSave, isRetiredTrackFormat, trackFrameKind, readStored, SAVE_FORMAT, START_CASH, START_CLOCK, writeStored, listSlots, writeSlot, renameSlot, deleteSlot, leaseSlot, ORIGINAL_SLOT, listBackups, writeBackup, isBackupOf, nextGameName, newGameName, slotName, takePending, writePending, readSync, writeSync, type PendingLoad, type GameMode, type NewGameSpec, type SlotBackup, type SlotLease, type SlotSummary, type StoredSave, type DayLedger, type HourLedger, type SaveHeader } from "./saveFile";
import { StateRows, upgradesOf, type RowEnv } from "./stateRows";
import type { ClockBus, ClockEdge, ClockLine, ClockNode, ClockState, EditOp, FromClock, MoneyState, NetworkSnapshot, PiInput, RegionNow, RouteEnd, StationAt, ToClock } from "./protocol";
import { changedHere, claim, fetchGame, forgetSlot, forkName, GAP_MS, lastListFailure, listGames, newGameId, onTokenGone, sendDeletes, settle as settleCloud, tieNew, tiedTo, upload, writeFetched, type CloudContext } from "./cloud";

/** Run 0 of each line is resampled this often (m) for the renderer's line texture. */
const SAMPLE_STEP = 50;
const SNAPSHOT_DEBOUNCE_MS = 500;
/** Whole trains on a new line across both directions, high/medium/low demand. */
const NEW_LINE_TRAINS: [number, number, number] = [0, 0, 0];
/** T-1464: a new bus line's headways, s, high/medium/low demand: a bus every 8, 10 and 20 minutes. */
const NEW_BUS_HEADWAYS: [number, number, number] = [480, 600, 1200];
/** Default fare curve: `START_FARE` as three nodes. */
const START_FARES: Fares = { ...faresOf(undefined, START_FARE), scales: {}, baseline: FARE_BASELINE };
/** A save's fares (T-1277): one from before the baseline was New York's fares, so its curve is
 * divided by the home country's scale; the player's scales kept. */
function loadFares(f: SaveHeader["fares"] | undefined): Fares {
  if (!f) return { ...START_FARES };
  const s = f.baseline ? 1 : incomeScale(cityById(HOME_CITY).country);
  const g = s === 1 ? f : { base: f.base / s, perKm: f.perKm / s, nodes: f.nodes?.map(([k, v]) => [k, v / s] as [number, number]) };
  const scales: Record<string, number> = {};
  for (const [c, v] of Object.entries(f.scales ?? {})) if (Number.isFinite(v)) scales[c] = tidyScale(v);
  return { ...faresOf(g, START_FARES), scales, baseline: FARE_BASELINE };
}
/** Autosave (T-029): this long after the last one if anything changed, wall ms; at once after a
 * construct or other paid edit, and when the page is hidden. */
const AUTOSAVE_MS = 60_000;
/** T-744: an autosave this long after the slot's newest backup is also written as a backup, wall ms */
let backupMs = 5 * 60_000;
/** Days of money kept in the ledger, and hours in the hourly one (T-447). */
const LEDGER_DAYS = 30;
const LEDGER_HOURS = 48;
const LEVEL_OF = { high: 0, medium: 1, low: 2 } as const;
/** T-867: hours a day at each demand level, which weigh each line's trains a day for its upkeep share */
const LEVEL_HOURS = new Float64Array([0, 1, 2].map((l) => PERIODS.reduce((s, p) => s + (LEVEL_OF[p.demand] === l ? p.to - p.from : 0), 0)));

let api: TrackApi | null = null;
let city = HOME_CITY;
/** the frame of the loaded game (T-125): new games are in the world frame, saves made before it in
 * the legacy one; read from the track model after every load */
let frame: Frame = makeFrame(NEW_GAME_FRAME);
function frameSpec(): FrameSpec { return { kind: frame.kind, lon: frame.lon, lat: frame.lat }; }
function syncFrame() {
  const o = api!.frame_origin();
  frame = makeFrame({ kind: frameKindName(api!.frame_kind()), lon: o[0], lat: o[1] });
}
let version = 0;
let snapVersion = 0;
let epoch = NaN;
let snapTimer = 0;
/** the last edit's time by step, ms (T-229): sent with its answer for `?debug=1` */
let timing: Record<string, number> = {};

// Money and the clock (T-028). The main thread's clock is the game's; it tells this worker each
// time it is set, paused or changes speed, and the worker settles fares and running costs one game
// hour at a time from that.
let fares: Fares = { ...START_FARES };
let ledger: DayLedger[] = [];
let hourLedger: HourLedger[] = [];
/** fare income per game hour in each period, US$M (from the main thread's demand results), per
 * kind of day (T-260, game/days.ts) */
let incomePerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
/** the same per line id (T-293), for each line's fares taken today */
let lineIncome = new Map<number, number[][]>();
/** train rides per game hour as `incomePerHour` (T-565), and per line id */
let ridesPerHour: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
let lineRides = new Map<number, number[][]>();
/** T-736: the boarding account's in-period shape strengths per kind of day and period, and the lag
 * (s) money lands after riders set out (game/moneyTiming.ts) */
const peakC: number[][] = DAY_KINDS.map(() => PERIODS.map(() => 0));
let moneyLag = 0;
const fareHour = (per: number[][] | undefined, tl: number, kindOf?: KindOf) => lagged(table(per), peakC, tl, moneyLag, 0, kindOf);
const rideHour = (per: number[][] | undefined, tl: number, kindOf?: KindOf) => between(table(per), peakC, tl, tl + 3600, kindOf);

/** T-872: the holiday calendar (its own chunk, fetched once), and each line's country: its first
 * stop's zone's (`cityWithin`; "" outside every zone). Its holidays and own weekend decide
 * its kinds of day (T-262). */
let holidayData: HolidayData | null = null;
let lineCountries = new Map<number, string>();
/** each line's first stop, lng and lat (null with none), as the last state was built */
let lineFirstStops = new Map<number, [number, number] | null>();
function placeLineCountries() {
  lineCountries = new Map([...lineFirstStops].map(([id, p]) => {
    const z = p ? cityWithin(p[0], p[1]) : null;
    return [id, z ? cityById(z).country ?? "" : ""];
  }));
}
/** T-1265: the coarse outlines name each station's zone (game/cities.ts `cityWithin`); read once
 * from init, and the lines' countries placed again when they are in */
let outlinesRead: Promise<void> = Promise.resolve();
let holidaysSent = "";
const kindOfLine = (id: number): KindOf => kindOfIn(holidayData, lineCountries.get(id));
function loadHolidays() {
  void import("../game/holidays.json").then((m) => {
    holidayData = m.default as unknown as HolidayData;
    if (syncHolidays()) sendTrips();
  }, () => {});
}
/** Send the track model each line's days whose kind is not the day number's, with their kind
 * (holidays, its country's own weekend, weekend days worked), from yesterday to 8 days ahead in
 * its local time; true if they changed. Cheap when nothing did: called before every read of the
 * boarding account by day. */
function syncHolidays(): boolean {
  if (!api || !holidayData) return false;
  const now = gameNow(), out: number[] = [];
  for (const [id, c] of lineCountries) {
    if (!c) continue;
    const kindOf = kindOfIn(holidayData, c), d0 = Math.floor((now + api.line_shift(id)) / 86400);
    const days: number[] = [];
    for (let d = d0 - 1; d <= d0 + 8; d++) { const k = kindOf(d); if (k !== dayKindOf(d)) days.push(d, k); }
    if (days.length) out.push(id, days.length / 2, ...days);
  }
  const key = out.join();
  if (key === holidaysSent) return false;
  holidaysSent = key;
  api.set_line_holidays(new Float64Array(out));
  return true;
}
// T-835: trains run from half a run before their period to half a run into it; `tail`, the cost a
// day a period adds by running to its own end before no service (`runTail`), spread over it
const costHour = (run: number[], tl: number, tail?: number[]) => lagged(((_k, p) => run[LEVEL_OF[PERIODS[p].demand]] + (tail?.[p] ?? 0) / (PERIODS[p].to - PERIODS[p].from)) as PerHour, null, tl, moneyLag, moneyLag / 2);
/** T-835 (service.rs `period_points`): a period followed by no service runs to its own end, not
 * half a run before it, so its line runs that much longer a day at its level; `run` per hour by
 * level, from `lines_info` entry `i`. Per period, US$M a day. */
function runTail(li: ArrayLike<number>, i: number, run: number[]): number[] {
  const lead = Math.min(7200, Math.max(0, (li[i + 11] - 2 * li[i + 20]) / 4));
  return PERIODS.map((p, k) => {
    const next = PERIODS.findIndex((q) => q.from === p.to % 24);
    const nextTph = next >= 0 ? li[i + 5 + LEVEL_OF[PERIODS[next].demand]] : 0;
    return nextTph > 0 ? 0 : (run[LEVEL_OF[PERIODS[k].demand]] * lead) / 3600;
  });
}
/** T-531: each region's ledgers, kept as the network's (a zone id; "" outside every zone or not yet
 * placed, so the regions add up to the network) */
let regions = new Map<string, { days: DayLedger[]; hours: HourLedger[] }>();
/** T-531: each running line's fares and running cost by region, [region, share] (game/regionMoney.ts) */
let lineShares = new Map<number, { fare: [string, number][]; ride: [string, number][]; run: [string, number][] }>();
/** T-867: how the network's upkeep divides between regions, [region, share] (game/regionMoney.ts) */
let upkeepRegions: [string, number][] = [];
/** T-684: each running line's points per game hour by region and kind of day and period (game/points.ts) */
let linePoints = new Map<number, [string, number[][]][]>();
/** T-685: each region's star, the game second it was earned */
let stars = new Map<string, number>();
/** T-1437: each region's higher goals, by share (%), the game second each was reached */
let goals = new Map<string, Record<string, number>>();
/** T-706: each line's station note and the city it was chosen in */
let lineNotes = new Map<string, [number, string]>();
/** T-1094: the train length last set on any line (0 for the most the platforms fit), a new line's start */
let newLineCars = 0;
/** T-531: paid edits' charges, held under "" until the main thread says how each divides */
const heldCharges = new Map<number, { day: number; hour: number; build: number; trains: number }>();
/** Wall ms on the clock both threads read (`ToClock` "clock"): Date.now() drifts from it, and the
 * worker running ahead of the main thread's clock pulled money time back across midnight (T-488). */
const wallNow = () => performance.timeOrigin + performance.now();
let clk = { t: START_CLOCK, rate: 0, at: wallNow() };
/** Loads so far (T-715): a "clock" message the main thread sent before it saw the latest still
 * carries the previous game's time, and settling to it charged the gap's running costs and then
 * held the money until the clock caught up. */
let game = 0;
/** T-284: money is settled up to this game second, not by whole hours, so the cash moves as fares
 * land. `partHour` is the hour being settled and `partFare`, `partCost` the shares of its fares and
 * running cost already in the cash (each reaches 1 as the hour ends, so an hour still pays its whole
 * fare and cost). */
let settledT = START_CLOCK;
let partHour = Math.floor(START_CLOCK / 3600);
let partFare = (START_CLOCK % 3600) / 3600, partCost = partFare;
/** per game hour (absolute index), the share of its fares paid by each second into it (3601
 * values from 0 to 1), from the boarding account's trains leaving platforms with riders; an hour
 * without one pays evenly (T-284) */
const hourShape = new Map<number, Float32Array>();
/** the cash and today's fares last sent in a "cash" message */
let sentCash = NaN, sentFares = NaN;
/** something to autosave; when the last autosave was written */
let dirty = false;
let savedAt = Date.now();
let saveInFlight: Promise<boolean> | null = null;
/** the next autosave is a save by hand: it goes to the account at once, with `force` (T-988) */
let byHandNext = false;
let activeSlot = ORIGINAL_SLOT;
let activeLease: SlotLease | null = null;
/** another tab saved this slot by hand and took its lease (T-1251): autosave stays off here */
let leaseTaken = false;
function hold(lease: SlotLease | null) {
  activeLease = lease;
  leaseTaken = false;
  void lease?.lost.then(() => {
    if (activeLease !== lease) return;
    activeLease = null;
    leaseTaken = true;
    void publishSlots();
  });
}
/** the start menu's new game (T-677): its slot's name, given at its first write */
let newSlotName: { key: string; name: string } | null = null;
/** the active slot while nothing is stored under it (a new game's, T-677) */
let unwritten: string | null = null;
let storageReady = true;
/** `?new=1`, `?synth=1`, `?demandTest=1` (T-320): no slot is written until "Save as a copy" gives
 * the game a slot of its own; a `?new=1` game has no active slot ("") and reads none */
let testPage = false;
let saveFailure: string | undefined;
/** while the game is being swapped for another slot's (T-743): nothing autosaves */
let switching = false;
/** T-728: the game's mode, start zone and opened zones, kept in the save's header (`opened` absent
 * in a save from before it) */
let mode: GameMode = "normal";
let startZone: string | undefined;
let opened: string[] | undefined = [];
function setMode(m: GameMode) {
  mode = m;
  api!.set_unlimited(m === "sandbox");
}
const slotChanges = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel("anitabuilder-save-slots");
slotChanges?.addEventListener("message", () => { if (api) void publishSlots(); });

async function publishSlots(error?: string) {
  let slots: Awaited<ReturnType<typeof listSlots>> = [];
  let backups: SlotBackup[] = [];
  try {
    slots = await listSlots(); storageReady = true;
    if (activeSlot) backups = await listBackups(activeSlot);
  } catch { storageReady = false; }
  const taken = !!activeSlot && cloudTaken === activeSlot;
  const writable = !!activeLease && storageReady && !saveFailure && !taken;
  const reason = saveFailure || (!storageReady ? "Browser storage is unavailable. Download this game as a file to keep it."
    : taken ? "Autosave is off here because this game was opened on another device."
    : !activeLease ? testPage ? "Autosave is off in this test page. Make a copy to keep the game."
      : leaseTaken ? "Autosave is off here because this game was saved in another tab."
      : navigator.locks ? "Autosave is off here because this slot was opened in another tab. Make a copy, or close the other tab and reload."
      : "This browser cannot protect save slots. Download this game as a file to keep it." : undefined);
  send({ kind: "saveSlots", slots, active: activeSlot, writable, at: slots.find(s => s.key === activeSlot)?.at ?? 0, reason, error, backups });
}

async function flushBeforeSwitch() {
  if (dirty && (!activeLease || cloudTaken === activeSlot)) throw new Error("Make a copy of this game before leaving it, or download it as a file and reload this page.");
  if (!await autosave()) throw new Error("This game could not be saved. Download it as a file before leaving it.");
}

/** T-743: New game, Open a file and a backup's restore each put the game in a slot of its own,
 * named `name`; the game being played is saved first and keeps its slot. `load` replaces the game
 * (throwing changes nothing). A new game's slot is written once something changes, a loaded one's
 * at once. A test page has no slot to keep: the game is replaced in the page and owns none. */
async function intoNewSlot(name: string, load: () => Promise<unknown> | void, writeNow: boolean) {
  if (testPage) {
    await load();
    activeLease?.release();
    activeSlot = "";
    activeLease = null;
    dirty = writeNow;
    await publishSlots();
    return;
  }
  await flushBeforeSwitch();
  const key = "slot-" + crypto.randomUUID();
  const lease = await leaseSlot(key);
  switching = true;
  try {
    await load();
  } catch (err) { lease?.release(); throw err; } finally { switching = false; }
  activeLease?.release();
  activeSlot = key;
  hold(lease);
  unwritten = key;
  newSlotName = { key, name };
  saveFailure = undefined;
  dirty = writeNow;
  savedAt = Date.now();
  if (!writeNow || !await autosave()) await publishSlots();
  slotChanges?.postMessage({ key });
}
// T-988: games on the account (workers/cloud.ts, notes/T-988.md). The main thread posts the account;
// each slot write goes up after it (a save by hand at once, autosaves at most every GAP_MS), and the
// game opened is claimed on the account. All of it runs in one queue of its own, so the game's
// messages never wait on the network; only the swap of the game in hand goes through theirs.
let cloudCtx: CloudContext | null = null;
/** the slot whose game this page has claimed (or uploaded) on the account */
let cloudClaimed = "";
/** the slot another device opened: autosave is off here until a save by hand takes it back */
let cloudTaken = "";
let cloudQueue: Promise<unknown> = Promise.resolve();
const cloudTimers = new Map<string, ReturnType<typeof setTimeout>>();
/** the game's first load is done: the claim can follow */
let initDone = false;
function cloudRun(f: () => Promise<unknown>): Promise<unknown> {
  cloudQueue = cloudQueue.then(f).catch(() => {});
  return cloudQueue;
}
onTokenGone(() => send({ kind: "accountGone" }));

/** After slot `key` was written: up now (`force`, a save by hand), or once the gap since its last
 * upload has passed. `fresh`: the slot's first write, which ties it to the account signed in. */
async function cloudAfterWrite(key: string, force: boolean, fresh: boolean) {
  const ctx = cloudCtx;
  if (!ctx || testPage) return;
  let rec = await readSync(key);
  if (!rec && fresh) rec = await tieNew(ctx, key);
  if (!tiedTo(rec, ctx) || !ctx.on) return;
  const wait = force ? 0 : (rec.upAt ?? 0) + GAP_MS - Date.now();
  if (wait > 0) {
    if (!cloudTimers.has(key)) cloudTimers.set(key, setTimeout(() => { cloudTimers.delete(key); void cloudUp(key); }, wait));
    return;
  }
  clearTimeout(cloudTimers.get(key));
  cloudTimers.delete(key);
  await cloudUp(key, force);
}

/** Slot `key` up as stored. The game in hand is claimed first; a 409 "elsewhere" on it means another
 * device opened it, and this page gives way (a save by hand sends `force` and takes it back). */
function cloudUp(key: string, force = false): Promise<unknown> {
  return cloudRun(() => upNow(key, force));
}
async function upNow(key: string, force: boolean) {
  const ctx = cloudCtx;
  if (!ctx?.on || testPage) return;
  if (key === activeSlot && !force && cloudClaimed !== key) { await openOnAccount(ctx); await publishSlots(); return; }
  if (!force) {
    const [s, r] = await Promise.all([readStored(key), readSync(key)]);
    if (!s || !r || !changedHere(s, r)) return; // went up already
  }
  const r = await upload(ctx, key, force);
  if (key === activeSlot && r === "ok") { cloudClaimed = key; if (cloudTaken === key) cloudTaken = ""; }
  if (key === activeSlot && r === "elsewhere") await giveWay(key);
  await publishSlots();
}

async function giveWay(key: string) {
  cloudTaken = key;
  cloudClaimed = "";
  const rec = await readSync(key);
  if (rec) await writeSync(key, { ...rec, gaveWay: true });
}

/** The page hides: anything waiting for the gap goes now (best effort, a closing page can cut it). */
function cloudFlush() {
  for (const [key, t] of cloudTimers) { clearTimeout(t); cloudTimers.delete(key); void cloudUp(key); }
  if (activeSlot && cloudTaken !== activeSlot) void cloudUp(activeSlot);
}

/** Settle every other slot this page can lock, then claim the game in hand. */
function cloudKick(): Promise<unknown> {
  return cloudRun(async () => {
    const ctx = cloudCtx;
    if (!ctx?.on || testPage || !initDone) return;
    await settleCloud(ctx, activeSlot);
    await openOnAccount(ctx);
    await publishSlots();
  });
}

/** The game in hand claimed on the account. A newer account copy replaces it if nothing changed here
 * (the page loads it in place), else this browser's version carries on as a fork in a new slot and
 * the old slot takes the account's. Offline it plays on and claims when back. */
async function openOnAccount(ctx: CloudContext) {
  const key = activeSlot;
  if (!key || !activeLease || cloudClaimed === key || cloudTaken === key) return;
  let [stored, rec] = await Promise.all([readStored(key), readSync(key)]);
  if (!stored || !tiedTo(rec, ctx)) return; // a new game not written yet, or not on this account
  const list = await listGames(ctx);
  const changed = () => (changedHere(stored!, rec!) || dirty) && !rec!.gaveWay;
  if (!list) {
    // unreachable: the game plays on, "Not on your account yet" while it has changed here
    if (activeSlot === key && changedHere(stored, rec)) { await writeSync(key, { ...rec, error: lastListFailure() }); await publishSlots(); }
    return;
  }
  if (activeSlot !== key) return;
  const g = list.games.find((x) => x.id === rec!.id);
  if (!g) {
    if (list.tombstones.some((t) => t.id === rec!.id)) {
      // deleted on another device: a game changed here is kept as a new account game; one not
      // changed stays in this browser only, so playing it brings back nothing the player deleted
      if (!changed()) { await writeSync(key, { ...rec, acct: undefined, local: true }); return; }
      rec = { ...rec, id: newGameId(), rev: 0, syncedAt: 0 };
      await writeSync(key, rec);
    }
    if (await upload(ctx, key) === "ok") cloudClaimed = key;
    return;
  }
  if (g.rev < rec.rev) { rec = { ...rec, rev: g.rev }; await writeSync(key, rec); }
  if (g.rev === rec.rev) {
    const c = await claim(ctx, rec);
    if (c !== "ok") return; // tried again at the next save
    cloudClaimed = key;
    rec = (await readSync(key)) ?? rec;
    if (rec.gaveWay) await writeSync(key, rec = { ...rec, gaveWay: undefined });
    if (changedHere((await readStored(key)) ?? stored, rec)) await upload(ctx, key);
    return;
  }
  const got = await fetchGame(ctx, g);
  if (!got) return;
  let fork = "";
  // through the game's own queue: nothing edits the game while it is swapped
  messages = messages.then(async () => {
    if (activeSlot !== key || !activeLease) return;
    stored = (await readStored(key)) ?? stored;
    if (!changed()) {
      await writeFetched(ctx, key, got);
      switching = true;
      try { await loadGame(got.stored.bytes, "autosave"); } finally { switching = false; }
      dirty = false;
      savedAt = got.stored.at;
      return;
    }
    // changed on both sides: the game in hand moves to a fork's slot, the account copy into this one
    fork = "slot-" + crypto.randomUUID();
    const lease = await leaseSlot(fork);
    if (!lease) { fork = ""; return; }
    await writeSlot(fork, { ...stored!, name: forkName(stored!.name || "Saved game", stored!.at) });
    await writeSync(fork, { id: newGameId(), acct: ctx.acct, rev: 0, syncedAt: 0 });
    await writeFetched(ctx, key, got);
    activeLease.release();
    activeSlot = fork;
    hold(lease);
    dirty = true; // what changed since the last write goes to the fork's slot
    slotChanges?.postMessage({ key: fork });
  }).catch((err) => send({ kind: "error", message: String(err) }));
  await messages;
  if (fork) {
    if (activeSlot === fork && await upload(ctx, fork) === "ok") cloudClaimed = fork;
  } else if (activeSlot === key) {
    const now = await readSync(key);
    if (now && await claim(ctx, now) === "ok") cloudClaimed = key;
  }
}

/** T-1196: a pending file or backup's bytes and the name of the slot it goes into; null if the backup
 * has gone */
async function pendingGame(p: Exclude<PendingLoad, { kind: "new" }>): Promise<{ bytes: Uint8Array; name: string } | null> {
  if (p.kind === "import") return { bytes: p.bytes, name: p.name || "Loaded game" };
  if (!isBackupOf(p.key, p.from)) return null;
  const backup = await readStored(p.key), slot = await readStored(p.from);
  if (!backup) return null;
  const time = new Date(backup.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return { bytes: backup.bytes, name: `${slotName(p.from, slot)} at ${time}` };
}

/** T-1196: the page reloads into another game, which frees all of this one's memory: the game in
 * hand is saved and its slot let go (the reloaded page's worker takes the next) */
async function leave(msg: Extract<ToClock, { kind: "leave" }>) {
  try {
    if (!testPage) await flushBeforeSwitch();
    if (msg.pending && !await writePending(msg.pending)) throw new Error("Browser storage is unavailable. Download this game as a file to keep it.");
    activeLease?.release();
    activeLease = null;
    send({ kind: "left" });
  } catch (err) { send({ kind: "left", error: (err as Error).message }); }
}

/** the empty network's save, for a new game (keeps the water mask) */
let emptyTrack: Uint8Array | null = null;
let moneyParams = [2.5, 2, 100, 9];

function send(msg: FromClock, transfer: Transferable[] = []) {
  postMessage(msg, { transfer });
}

/** world metres -> lng/lat in the loaded game's frame (game/frame.ts) */
function packLngLat(x: number, y: number): [number, number] {
  return frame.toLonLat(x, y);
}
/** world metres -> local units, relative to the draw anchor the main thread last sent (T-307) */
function packLocal(x: number, y: number): [number, number] {
  const [lng, lat] = packLngLat(x, y);
  return toLocal(lng, lat);
}
function anchorSpec() {
  const a = anchor();
  return { lng: a.lng, lat: a.lat, version: a.version };
}

// ---------------------------------------------------------------- Rust panics (T-1007)
//
// A panic in the track model aborts the call and leaves wasm-bindgen's borrow on the `TrackApi`
// held, so every later call on it throws. The worker then makes a new `TrackApi` and opens the
// network in it as a load does, from the network as it was after the last load or edit (`kept`;
// an edit's panic always goes back to it, any other panic to the live network when it can still be
// read). The main thread logs the panic and tells the player ("recovered").

const EDIT_UNDONE = "That edit failed and was undone.";
const MODEL_RESTORED = "The train simulation failed and was restored to your last edit.";
const MODEL_DOWN = "The train simulation stopped working. Reload the page to go on. Your last autosave is kept.";
/** more panics than this in a minute, or in all, and the model is left down: it is failing for a reason a rebuild does not cure */
const MAX_PANICS_PER_MINUTE = 3;
const MAX_PANICS = 8;

/** the network, cash and fleet after the last load or edit */
let kept: { net: Uint8Array; cash: number; fleet: number; fleetFast: number; fleetBus: number } | null = null;
function keep() {
  try {
    const a = api!;
    kept = { net: a.save(), cash: a.cash(), fleet: a.fleet(), fleetFast: a.fleet_fast(), fleetBus: a.fleet_bus() };
  } catch { /* the model is down; the next panic recovery says so */ }
}

const panicTimes: number[] = [];
let panicCount = 0;
let modelDown = false;
/** when the last recovery ended; an error raised before it came from the model it replaced */
let recoveredAt = 0;
/** `?debug=1` probes (dev server only): the next edit panics after it is applied */
let panicAfterEdit = false;

function isPanic(err: unknown): boolean {
  return err instanceof WebAssembly.RuntimeError || (err instanceof Error && /recursive use of an object|null pointer passed to rust/.test(err.message));
}

/** Panic from a call that panics: an id no model has (a `&mut self` call, so the borrow stays held). */
function panicNow() {
  api!.set_node(0xffffffff, 0, 0, 0, false);
}

/** A new track model over the same WASM instance, set up as the first one is (a panic leaves the
 * old one's borrow held for ever; its memory is left where it is, since freeing a model that
 * stopped half way through a change is not safe). */
function newModel(): TrackApi {
  const a = new TrackApi(NEW_GAME_FRAME.lon, NEW_GAME_FRAME.lat, true);
  api = a;
  worldWaterIn = false;
  // T-309: each region's local time, kept across loads
  a.set_time_zones(timeZoneTable());
  a.set_empty_frame(frameKindNumber(NEW_GAME_FRAME.kind));
  syncFrame();
  a.clear_water();
  a.set_cash(START_CASH);
  return a;
}

async function rebuildModel(fromKept: boolean) {
  const old = api!;
  let k = kept;
  if (!fromKept) {
    try { k = { net: old.save(), cash: old.cash(), fleet: old.fleet(), fleetFast: old.fleet_fast(), fleetBus: old.fleet_bus() }; } catch { /* held: use what was kept */ }
  }
  if (!k) throw new Error("no network was kept");
  (old as unknown as { __destroy_into_raw(): number }).__destroy_into_raw();
  newModel();
  masksIn.clear();
  holidaysSent = "";
  await openNetwork(k.net);
  api!.set_fleet(k.fleet);
  api!.set_fleet_fast(k.fleetFast);
  api!.set_fleet_bus(k.fleetBus);
  api!.set_cash(k.cash);
  setMode(mode);
  restart(gameNow(), "autosave");
  dirty = true;
}

/** Rebuild the track model after a panic and tell the main thread. `at`: when the error was raised. */
async function recover(err: unknown, what: string, notice: string, at: number, fromKept = false) {
  if (modelDown || at <= recoveredAt) return;
  const now = Date.now();
  panicTimes.push(now);
  while (panicTimes[0] < now - 60_000) panicTimes.shift();
  panicCount++;
  const detail = `Rust panic in ${what}: ${err instanceof Error ? err.stack || err.message : String(err)}`;
  try {
    if (panicTimes.length > MAX_PANICS_PER_MINUTE || panicCount > MAX_PANICS) throw new Error("too many panics");
    await rebuildModel(fromKept);
    send({ kind: "recovered", detail, notice });
  } catch (e) {
    modelDown = true;
    send({ kind: "recovered", detail: `${detail}\nnot rebuilt: ${e}`, notice: MODEL_DOWN, down: true });
  }
  recoveredAt = performance.now();
}

/** A panic found outside a message (a timer, a save), rebuilt in turn with the messages. */
function panicLater(err: unknown, what: string) {
  const at = performance.now();
  messages = messages.then(() => recover(err, what, MODEL_RESTORED, at));
}
addEventListener("error", (e) => {
  if (!isPanic(e.error)) return;
  e.preventDefault();
  panicLater(e.error, "a timer");
});
addEventListener("unhandledrejection", (e) => {
  if (!isPanic(e.reason)) return;
  e.preventDefault();
  panicLater(e.reason, "background work");
});

// ---------------------------------------------------------------- water masks on demand (T-302)
//
// A zone's mask is fetched only when the view comes near its box or track is drawn or loaded within
// it (docs/zones.md 6). The rule that keeps building exact: every zone whose box meets existing track
// has its mask installed. An edit or preview that would make new track is first worked out, and
// if its geometry meets a zone whose mask is not in yet, it waits for the mask and is worked out
// again, so ground over water is never accepted because a mask was late.

/** the packs' folder (the init message's `pack` names one pack in it) */
let packBase = "";
interface WaterRaster { cell: number; x0: number; y0: number; w: number; h: number; rows: Uint32Array; runs: Uint16Array; lon: number; lat: number }
const waterFetches = new Map<string, Promise<WaterRaster | null>>();
/** zones whose masks are in the track model (a zone with no mask counts once its head said so) */
const masksIn = new Set<string>();
/** a point of track this near a zone's box needs its mask: the sampled geometry's spacing to spare */
const TRACK_MARGIN_M = 2_000;
/** the view this near a zone's box fetches its mask ahead, when the view is under VIEW_SPAN_M wide */
const VIEW_MARGIN_M = 20_000;
const VIEW_SPAN_M = 400_000;

function fetchWater(id: string): Promise<WaterRaster | null> {
  let p = waterFetches.get(id);
  if (!p) {
    const url = new URL(id, packBase).href;
    p = (async () => {
      const head = await (await fetch(url + ".json")).json();
      const row = head.arrays.find((a: any) => a.name === "water_row");
      const xs = head.arrays.find((a: any) => a.name === "water_x");
      if (!head.water || !row || !xs) return null;
      const start = row.offset, end = xs.offset + xs.count * 2;
      const res = await fetch(url + ".bin", { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (!res.ok) throw new Error(`${res.status} ${url}.bin`);
      const buf = await res.arrayBuffer();
      const base = res.status === 206 ? start : 0;
      const w = head.water;
      return {
        cell: w.cell_m, x0: w.origin_m[0], y0: w.origin_m[1], w: w.size[0], h: w.size[1],
        rows: new Uint32Array(buf, row.offset - base, row.count), runs: new Uint16Array(buf, xs.offset - base, xs.count),
        lon: head.origin.lon, lat: head.origin.lat,
      };
    })();
    waterFetches.set(id, p);
    // a failed fetch is tried again by the next edit or view that needs it
    p.catch((err) => {
      if (waterFetches.get(id) === p) waterFetches.delete(id);
      send({ kind: "error", message: `water mask for ${id} not loaded: ${err}` });
    });
  }
  return p;
}

/** Install the masks of `ids` (fetching those not fetched); resolves once each is in or failed. The
 * raster is laid out in the pack's own frame; the track model puts it in the game's (T-125). */
async function ensureMasks(ids: Iterable<string>): Promise<void> {
  const all = [...new Set(ids)];
  // T-1270: each zone's terrain comes with its mask, but nothing waits for it
  void ensureTerrain(all);
  const todo = all.filter((id) => !masksIn.has(id));
  if (!todo.length) return;
  const got = await Promise.all(todo.map((id) => fetchWater(id).then((r) => ({ r }), () => null)));
  todo.forEach((id, k) => {
    const g = got[k];
    if (!g || masksIn.has(id)) return;
    masksIn.add(id);
    const r = g.r;
    if (r) api!.add_pack_water(r.cell, r.x0, r.y0, r.w, r.h, r.rows, r.runs, r.lon, r.lat);
  });
}

// ---------------------------------------------------------------- terrain classes (T-1270)
//
// A zone's terrain grid (`terrain/<id>.terrain` among the packs, pipeline/zone_terrain.py) is
// fetched with its water mask. It only prices track, so no preview or edit waits for it: track drawn
// before it lands is priced on flat ground and priced again when it does. Construct waits for the
// grids under what it builds, so nothing is paid on a guess.

const terrainFetches = new Map<string, Promise<ArrayBuffer | null>>();
/** zones whose terrain is in the track model (or that have none) */
const terrainIn = new Set<string>();

function fetchTerrain(id: string): Promise<ArrayBuffer | null> {
  let p = terrainFetches.get(id);
  if (!p) {
    p = (async () => {
      const res = await fetch(new URL(`terrain/${id}.terrain`, packBase).href);
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`${res.status}`);
      return await res.arrayBuffer();
    })();
    terrainFetches.set(id, p);
    // a failed fetch is tried again by the next view or edit that needs it
    p.catch(() => {
      if (terrainFetches.get(id) === p) terrainFetches.delete(id);
    });
  }
  return p;
}

/** Install the terrain of `ids` (fetching those not fetched); resolves once each is in or failed. A
 * grid that reprices the blueprint sends the new state, so the quote follows. */
async function ensureTerrain(ids: Iterable<string>): Promise<void> {
  const todo = [...new Set(ids)].filter((id) => !terrainIn.has(id));
  if (!todo.length || !api) return;
  const got = await Promise.all(todo.map((id) => fetchTerrain(id).then((b) => ({ b }), () => null)));
  const a = api;
  if (!a) return;
  const before = a.blueprint_cost();
  todo.forEach((id, k) => {
    const g = got[k];
    if (!g || terrainIn.has(id)) return;
    terrainIn.add(id);
    // a server that answers a missing file with a page instead of a 404 gives no grid either
    if (g.b && a.add_terrain(new Uint8Array(g.b))) send({ kind: "terrain", id, bytes: g.b.slice(0) }); // the tint (T-1270)
  });
  if (Math.abs(a.blueprint_cost() - before) > 1e-9) sendState();
}

// ---------------------------------------------------------------- the world's coarse water (T-1285)
//
// One file for the whole world (`world.water` among the packs, pipeline/world_water.py, about 1 MB),
// deciding water wherever no zone's mask does: the open sea between zones. Fetched once at start;
// previews, edits and loads wait for it as for a zone's mask. A server without it (a 404) leaves the
// sea outside the zones as land, as before.

let worldWaterFetch: Promise<ArrayBuffer | null> | null = null;
/** the current track model has the world's water (or there is none to have) */
let worldWaterIn = false;

function worldWaterIntoModel(): Promise<void> {
  if (worldWaterIn) return Promise.resolve();
  if (!worldWaterFetch) {
    const p = (async () => {
      const res = await fetch(new URL("world.water", packBase).href);
      if (res.status === 404) return null;
      if (!res.ok) throw new Error(`${res.status}`);
      return await res.arrayBuffer();
    })();
    worldWaterFetch = p;
    // a failed fetch is tried again by the next edit; the sea outside the zones is land until then
    p.catch((err) => {
      if (worldWaterFetch === p) worldWaterFetch = null;
      send({ kind: "error", message: `world water mask not loaded: ${err}` });
    });
  }
  return worldWaterFetch.then((b) => {
    if (worldWaterIn || !api) return;
    // a server that answers a missing file with a page gives no mask either
    if (b) api.add_world_water(new Uint8Array(b));
    worldWaterIn = true;
  }, () => {});
}

/** Before Construct: the terrain under every blueprint piece. */
async function terrainForBlueprint() {
  const a = api!;
  const xy: number[] = [];
  const ei = a.edges_info();
  for (let i = 0; i < ei.length; i += 10) {
    if (ei[i + 4] === 1) continue;
    const r = a.edge_render(ei[i]);
    for (let k = 0; k + 2 < r.length; k += 3) xy.push(r[k], r[k + 1]);
  }
  const need = zonesOfPoints(xy).filter((id) => !terrainIn.has(id));
  if (need.length) await ensureTerrain(need);
}

/** The zones within TRACK_MARGIN_M of any of these world-metre points (x, y every `stride`). */
function zonesOfPoints(xy: ArrayLike<number>, stride = 2, start = 0): string[] {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (let k = start; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    if (lon < w) w = lon; if (lon > e) e = lon;
    if (lat < s) s = lat; if (lat > n) n = lat;
  }
  if (!(e >= w)) return [];
  const near = zonesTouching(w, s, e, n, TRACK_MARGIN_M);
  // one box over points far apart (a long route, a whole save) meets zones none of them is near:
  // test those point by point
  if (near.length <= 1) return near.map((c) => c.id);
  const boxes = near.map((c) => {
    const [bw, bs, be, bn] = zoneBox(c);
    const dy = TRACK_MARGIN_M / M_PER_DEG, dx = dy / Math.cos(Math.min(85, Math.max(Math.abs(bs), Math.abs(bn))) * Math.PI / 180);
    return { id: c.id, w: bw - dx, s: bs - dy, e: be + dx, n: bn + dy };
  });
  const hit = new Set<string>();
  for (let k = start; k + 1 < xy.length; k += stride) {
    const [lon, lat] = frame.toLonLat(xy[k], xy[k + 1]);
    for (const b of boxes) if (lon >= b.w && lon <= b.e && lat >= b.s && lat <= b.n) hit.add(b.id);
  }
  return [...hit];
}

// T-1039: previews the main thread no longer wants (`previewStale`, sent outside the queue) are
// answered `stale` without being worked out, or while they wait for masks
let previewStaleSeq = 0;
let wakeStale: (() => void) | null = null;
const stale = (seq: number) => seq <= previewStaleSeq;
type PreviewMsg = Extract<ToClock, { kind: "preview" | "previewEdge" | "previewNode" | "previewNodes" }>;
const isPreview = (m: ToClock): m is PreviewMsg => m.kind === "preview" || m.kind === "previewEdge" || m.kind === "previewNode" || m.kind === "previewNodes";
function sendStale(seq: number) {
  send({ kind: "preview", seq, stale: true, ok: false, cost: 0, lengthM: 0, mult: 0, issues: [], verts: new Float64Array(0), pts: new Float32Array(0) });
}

/** whether the last `withMasks` ended with every mask its geometry meets in */
let masksComplete = false;

/** Work out a preview; while its geometry meets zones whose masks are not in, install them and work it
 * out again. Returns the last result, or null once preview `seq` is no longer wanted (the masks
 * still come in). */
async function withMasks(run: () => Float64Array, points: (out: Float64Array) => number[], seq?: number): Promise<Float64Array | null> {
  masksComplete = false;
  await worldWaterIntoModel();
  for (;;) {
    const out = run();
    const need = zonesOfPoints(points(out)).filter((id) => !masksIn.has(id));
    if (!need.length) {
      masksComplete = true;
      return out;
    }
    if (seq === undefined) await ensureMasks(need);
    else {
      const gone = await Promise.race([ensureMasks(need).then(() => false), new Promise<boolean>((r) => (wakeStale = () => stale(seq) && r(true)))]);
      wakeStale = null;
      if (gone) return null;
    }
    // a mask that could not be fetched (reported) leaves the edit as it was without it
    if (need.every((id) => !masksIn.has(id))) return run();
  }
}

/** world x, y of a route or edge preview's polyline (`preview_route`'s layout) */
function routePoints(out: Float64Array): number[] {
  const nv = out[3], p0 = 5 + nv * 4, xy: number[] = [];
  for (let k = p0; k + 2 < out.length; k += 3) xy.push(out[k], out[k + 1]);
  return xy;
}
/** the same for a node drag's preview (`preview_move_node`) */
function nodePoints(out: Float64Array): number[] {
  const n = out[2], xy: number[] = [];
  let i = 3;
  for (let e = 0; e < n; e++) {
    i += out[i] * 4 + 1;
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) xy.push(out[i], out[i + 1]);
  }
  return xy;
}

/** Before a load (T-737): the masks of the zones the save's nodes and curve points fall in, so the
 * network is derived once (a load is about 3 s in WASM for Japan's 709 lines). `masksForLoaded`
 * still catches a zone only the drawn curves reach. */
async function masksBeforeLoad(net: Uint8Array) {
  if (trackFrameKind(net) !== NEW_GAME_FRAME.kind) return;
  const pts = TrackApi.save_points(net);
  if (!pts.length) return; // undecodable: the load says so
  const shown = frame;
  frame = makeFrame(NEW_GAME_FRAME);
  const missing = zonesOfPoints(pts).filter((id) => !masksIn.has(id));
  frame = shown;
  if (!missing.length) return;
  api!.load(emptyTrack!);
  await ensureMasks(missing);
}

/** Put a network (in the world frame) in the track model, with the masks of the zones it meets. */
async function openNetwork(net: Uint8Array) {
  await worldWaterIntoModel();
  await masksBeforeLoad(net);
  if (!api!.load(net)) throw new Error("This save is damaged.");
  await masksForLoaded(net);
}

/** After a network is loaded: the masks of every zone its track meets, then the network opened again
 * over them if any was missing (each mask installed on a loaded network would rework all of it). */
async function masksForLoaded(net: Uint8Array) {
  const a = api!;
  syncFrame();
  const xy: number[] = [];
  const ni = a.nodes_info();
  for (let i = 0; i < ni.length; i += 10) xy.push(ni[i + 1], ni[i + 2]);
  const ei = a.edges_info();
  for (let i = 0; i < ei.length; i += 10) {
    const r = a.edge_render(ei[i]);
    for (let k = 0; k + 2 < r.length; k += 3) xy.push(r[k], r[k + 1]);
  }
  const missing = zonesOfPoints(xy).filter((id) => !masksIn.has(id));
  if (!missing.length) return;
  a.load(emptyTrack!);
  await ensureMasks(missing);
  if (!a.load(net)) throw new Error("This save is damaged.");
}

// ---------------------------------------------------------------- state for the main thread

function hex(c: number) {
  return "#" + c.toString(16).padStart(6, "0");
}

// What the renderer draws, per edge and per line, in local units, kept between edits: an edit
// re-renders only the edges and lines it dirtied (T-063). Cleared when a game is loaded.
const edgeDrawn = new Map<number, Float32Array>();
const edgeSpeedDrawn = new Map<number, Float32Array>();
const lineDrawn = new Map<number, Float32Array>();
function forget(all = false) {
  if (all) {
    edgeDrawn.clear();
    edgeSpeedDrawn.clear();
    lineDrawn.clear();
    stateRows.reset();
    postWhole();
    return;
  }
  for (const e of api!.dirty_edges()) { edgeDrawn.delete(e); edgeSpeedDrawn.delete(e); }
  for (const l of api!.dirty_lines()) lineDrawn.delete(l);
}

/** The track model's mark rows (`crossings_info`, `preview_marks`) with x, y moved to local units. */
function markPoints(rows: Float64Array, stride: number): Float32Array {
  const out = new Float32Array(rows.length);
  for (let k = 0; k < rows.length; k += stride) {
    const [x, y] = packLocal(rows[k], rows[k + 1]);
    out.set([x, y], k);
    for (let j = 2; j < stride; j++) out[k + j] = rows[k + j];
  }
  return out;
}

/** what `StateRows` needs of the worker */
const rowEnv: RowEnv = { lngLat: packLngLat, levelHours: LEVEL_HOURS, busRow };
/** the rows of the last states, read again only where the track model says something changed (T-1382);
 * `?patch=0` reads them whole */
const stateRows = new StateRows();

function buildState(): ClockState {
  const tq0 = performance.now();
  const a = api!;
  if (patchMode === "0") stateRows.reset();
  stateRows.bind(a, frame);
  const edges = stateRows.edges(a);
  const drawn: Float32Array[] = [];
  for (const { id } of edges) {
    let d = edgeDrawn.get(id);
    if (!d) {
      const r = a.edge_render(id);
      d = new Float32Array(r.length);
      for (let k = 0; k < r.length; k += 3) {
        const [x, y] = packLocal(r[k], r[k + 1]);
        d[k] = x;
        d[k + 1] = y;
        d[k + 2] = r[k + 2];
      }
      edgeDrawn.set(id, d);
      edgeSpeedDrawn.set(id, a.edge_render_speeds(id));
    }
    drawn.push(d);
  }
  const speedParts = edges.map((e) => edgeSpeedDrawn.get(e.id));
  const tq1 = performance.now(); timing.sEdges = tq1 - tq0;
  const ups = upgradesOf(a);
  const nodes = stateRows.nodes(a, rowEnv, ups);
  const stationCount = nodes.length;
  const tq2 = performance.now(); timing.sNodes = tq2 - tq1;
  const lines = stateRows.lines(a, rowEnv, nodes);
  const tq3 = performance.now(); timing.sLines = tq3 - tq2;
  if (patchMode === "check") checkRows({ edges, nodes, lines }, a, ups);
  // a bus line's stops that are no station were added to the nodes by its row
  const busNodes = new Map(nodes.slice(stationCount).map((n) => [n.id, n]));
  lineFirstStops = new Map(lines.map((l) => {
    const n = stateRows.nodeRow(l.stops[0]) ?? busNodes.get(l.stops[0]);
    return [l.id, n ? [n.lng, n.lat] : null];
  }));
  placeLineCountries();
  syncHolidays();
  // Keyframes: phase tables, profile meta, and run 0 of every line resampled.
  const meta = a.profile_meta();
  const nLines = meta.length / 24;
  const lineTable = new Float32Array(Math.max(1, nLines) * 4);
  const runs: Float32Array[] = [];
  let nSamples = 0;
  for (const l of lines) {
    if (!l.ok) continue;
    let d = lineDrawn.get(l.id);
    if (!d) {
      const p = a.run_polyline(l.id, 0, SAMPLE_STEP);
      d = new Float32Array(p.length);
      for (let k = 0; k < p.length; k += 2) d.set(packLocal(p[k], p[k + 1]), k);
      lineDrawn.set(l.id, d);
    }
    runs.push(d);
    lineTable.set([nSamples, d.length / 2, l.lengthM, SAMPLE_STEP], l.id * 4);
    nSamples += d.length / 2;
  }
  const samples = new Float32Array(nSamples * 2);
  runs.reduce((o, d) => (samples.set(d, o), o + d.length), 0);
  const markers = [0, 1, 2].map((lev) => {
    const m = a.markers(lev);
    const out: ClockState["markers"][number] = [];
    for (let k = 0; k < m.length; k += 7) {
      const [x, y] = packLocal(m[k], m[k + 1]);
      out.push({ x, y, rho: m[k + 2], delayS: m[k + 3], kind: m[k + 4], tph: m[k + 5], held: m[k + 6] > 0 });
    }
    return out;
  });
  const crossings = markPoints(a.crossings_info(), 5);
  const tq4 = performance.now(); timing.sRest = tq4 - tq3;
  const fullTrack = [0, 1, 2].map((lev) => a.full_track(lev));
  const builtCost = a.built_cost();
  const blueprintCost = Math.max(a.total_cost() - builtCost, 0);
  const blueprintItems = Array.from(a.blueprint_cost_items());
  const tq5 = performance.now(); timing.sCosts = tq5 - tq4;
  const phases = a.phases();
  timing.sPhases = performance.now() - tq5;
  return {
    version: ++version,
    markers,
    fullTrack,
    crossings,
    cash: a.cash(),
    builtCost,
    blueprintCost,
    blueprintItems,
    canUndo: a.can_undo(),
    canRedo: a.can_redo(),
    edges,
    nodes,
    lines,
    links: Array.from(a.station_links()),
    render: {
      edgeParts: drawn,
      edgeSpeedParts: speedParts,
      phases,
      meta,
      samples,
      lineTable,
    },
    money: moneyState(),
    anchor: anchorSpec(),
  };
}

// ---------------------------------------------------------------- money (T-028)

function gameNow(): number {
  return clk.t + Math.max(0, (wallNow() - clk.at) / 1000) * clk.rate;
}

/** Wall ms the last train count or headway edit took here, from applying it to its trips sent. */
let serviceEditMs = 100;
/** T-1071: wall seconds after the clock a train count or headway edit takes effect: room for this
 * worker's own time on it, twice over, and the main thread's frame, so its new trips are on screen
 * before anything they change is drawn. */
function serviceLeadS(): number {
  return Math.min(2, 0.25 + (2 * serviceEditMs) / 1000);
}

/** Money's now: the clock, or the time settled up to if the clock has been set back behind it, so
 * a charge or the day shown never lands on a day already closed (T-488). */
function moneyNow(): number {
  return Math.max(gameNow(), settledT);
}

/** The entry for `key` in a ledger kept in order of `at`, made if missing (in its place, so a day or
 * hour is never listed twice, T-488); the ledger keeps the newest `cap`. */
function entryIn<T extends { fares: number; running: number; trains: number; build: number; rides?: number }>(list: T[], at: (e: T) => number, key: number, make: () => T, cap: number): T {
  let i = list.length;
  while (i > 0 && at(list[i - 1]) > key) i--;
  if (i > 0 && at(list[i - 1]) === key) return list[i - 1];
  const e = make();
  list.splice(i, 0, e);
  if (list.length > cap) list.splice(0, list.length - cap);
  return e;
}

/** Game day `day`'s ledger entry; the ledger keeps LEDGER_DAYS days. */
function dayEntry(day: number): DayLedger {
  return entryIn(ledger, (d) => d.day, day, () => ({ day, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS);
}

/** The hour's entry in the hourly ledger (T-447); it keeps LEDGER_HOURS hours. */
function hourEntry(hour: number): HourLedger {
  return entryIn(hourLedger, (d) => d.hour, hour, () => ({ hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS);
}

/** T-531: a region's entries for game day `day` and hour `hour`, made if missing. */
function regionEntries(zone: string, day: number, hour: number): [DayLedger, HourLedger] {
  let r = regions.get(zone);
  if (!r) regions.set(zone, (r = { days: [], hours: [] }));
  return [
    entryIn(r.days, (d) => d.day, day, () => ({ day, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS),
    entryIn(r.hours, (d) => d.hour, hour, () => ({ hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS),
  ];
}

/** T-531: every region's today and this hour that has an entry today. */
function regionNow(): RegionNow {
  const day = Math.floor(moneyNow() / 86400), hour = Math.floor(moneyNow() / 3600), out: RegionNow = {};
  for (const [k, r] of regions) {
    const d = r.days[r.days.length - 1], h = r.hours[r.hours.length - 1];
    if (d?.day === day) out[k] = { today: { ...d }, hour: h?.hour === hour ? { ...h } : { hour, fares: 0, running: 0, trains: 0, build: 0, rides: 0 } };
  }
  return out;
}

/** T-531: the running lines' money by region for settling: each line's fare income tables and its
 * running cost per game hour at each demand level (as `running_cost`), with its local time. */
function regionLines() {
  const a = api!, li = a.lines_info(), unit = moneyParams[2] / 1e6;
  const costs: { shift: number; run: number[]; tail: number[]; shares: [string, number][] }[] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue;
    const id = li[i], km = li[i + 9] / 1000, cars = li[i + 21] || li[i + 8];
    const [perTrainKm, perCarKm] = a.line_fast(id) ? [moneyParams[6], moneyParams[5]] : [moneyParams[3], moneyParams[1]];
    const run = [0, 1, 2].map((l) => li[i + 5 + l] * 2 * km * (perTrainKm + perCarKm * cars) * unit);
    costs.push({ shift: a.line_shift(id), run, tail: runTail(li, i, run), shares: lineShares.get(id)?.run ?? [] });
  }
  const fares = [...lineIncome].map(([id, per]) => ({ shift: a.line_shift(id), kindOf: kindOfLine(id), per, shares: lineShares.get(id)?.fare ?? [] }));
  const rides = [...lineRides].map(([id, per]) => ({ shift: a.line_shift(id), kindOf: kindOfLine(id), per, shares: lineShares.get(id)?.ride ?? [] }));
  return { costs, fares, rides };
}

/** T-531: hour `t`'s fares (US$M), running cost (US$M) and train rides (T-565) in each region. A
 * line not yet placed counts under "". */
function regionHour(t: number, rl: ReturnType<typeof regionLines>): Map<string, [number, number, number]> {
  const out = new Map<string, [number, number, number]>();
  const spread = (shares: [string, number][], v: number, k: 0 | 1 | 2) => {
    if (!v) return;
    for (const [z, s] of shares.length ? shares : [["", 1] as [string, number]]) {
      let e = out.get(z);
      if (!e) out.set(z, (e = [0, 0, 0]));
      e[k] += v * s;
    }
  };
  for (const f of rl.fares) spread(f.shares, fareHour(f.per, t + f.shift, f.kindOf), 0);
  for (const f of rl.rides) spread(f.shares, rideHour(f.per, t + f.shift, f.kindOf), 2);
  for (const c of rl.costs) spread(c.shares, costHour(c.run, t + c.shift, c.tail), 1);
  return out;
}

/** T-531: a region's saved ledgers, as `mergeLedger` makes the network's. */
function loadRegions(saved: SaveHeader["regions"]) {
  regions = new Map(Object.entries(saved ?? {}).map(([k, r]) => [k, { days: mergeLedger(r.ledger, (d) => d.day), hours: mergeLedger(r.hours, (d) => d.hour) }]));
  heldCharges.clear();
}

function savedRegions(copy: <T extends object>(e: T) => T = (e) => ({ ...e })): NonNullable<SaveHeader["regions"]> {
  return Object.fromEntries([...regions].map(([k, r]) => [k, { ledger: r.days.map(copy), hours: r.hours.map(copy) }]));
}

/** An entry as the save keeps it (T-567): money to the dollar and rides to a tenth, which the
 * charts cannot tell from the full figures and which gzips to about 60% of them. */
function rounded<T extends object>(e: T): T {
  const out: Record<string, number> = { ...e } as Record<string, number>;
  for (const k of ["fares", "running", "upkeep", "trains", "build"]) if (typeof out[k] === "number") out[k] = Math.round(out[k] * 1e6) / 1e6;
  if (typeof out.rides === "number") out.rides = Math.round(out.rides * 10) / 10;
  if (typeof out.points === "number") out.points = Math.round(out.points);
  return out as T;
}

/** A saved ledger in order with one entry per day or hour: saves written before T-488 can list a
 * day twice, the second time empty, which merging adds back where it belongs. */
function mergeLedger<T extends { fares: number; running: number; trains: number; build: number; upkeep?: number; rides?: number; points?: number }>(list: T[] | undefined, at: (e: T) => number): T[] {
  const by = new Map<number, T>();
  for (const e of list ?? []) {
    const k = at(e), d = by.get(k);
    if (!d) by.set(k, { ...e, rides: e.rides ?? 0, points: e.points ?? 0 });
    else { d.fares += e.fares; d.running += e.running; d.upkeep = (d.upkeep ?? 0) + (e.upkeep ?? 0); d.trains += e.trains; d.build += e.build; d.rides = (d.rides ?? 0) + (e.rides ?? 0); d.points = (d.points ?? 0) + (e.points ?? 0); }
  }
  return [...by.values()].sort((a, b) => at(a) - at(b));
}

/** A load settles again from the save's clock, so what the loaded ledgers hold after it leaves them
 * and its net comes back into the cash; edits' charges stay, on the clock's day and hour. Such
 * entries come from a clock pulled back (T-488, minutes) or, before T-715, a clock of the game open
 * before (up to a year of running costs with no fares, charged again by each such load). The day
 * ledger keeps 30 days, so the running cost is worked out again on the loaded network up to where the
 * hour ledger ends, times the whole days kept over the same worked out again. Returns US$M. */
function unsettleAfter(clockT: number): number {
  const D = Math.floor(clockT / 86400), H = Math.floor(clockT / 3600);
  const last = hourLedger[hourLedger.length - 1];
  if (!last || last.hour <= H) return 0;
  const groups = localGroups();
  const costAt = (h: number) => groups.reduce((s, g) => s + costHour(g.running, h * 3600 + g.shift, g.tail), 0);
  const replay = (from: number, to: number) => {
    let c = 0;
    for (let t = from; t < to;) {
      const h = Math.floor(t / 3600), e = Math.min(to, (h + 1) * 3600);
      c += costAt(h) * (e - t) / 3600;
      t = e;
    }
    return c;
  };
  const whole = ledger.filter((d) => d.day > D && d.day < Math.floor(last.hour / 24));
  const kept = whole.reduce((s, d) => s + d.running, 0), again = whole.reduce((s, d) => s + replay(d.day * 86400, (d.day + 1) * 86400), 0);
  const times = kept > 0 && again > 0 ? kept / again : 1;
  const full = costAt(last.hour) * times;
  const end = last.hour * 3600 + 3600 * (full > 0 ? Math.min(1, last.running / full) : 1);
  const back = times * replay(Math.max(clockT, end - 86400 * 366), end); // settle's year at most
  // T-867: upkeep drains evenly, so its hours since the clock are worked out again the same way
  const upHour = upkeepPerHour(), upEnd = last.hour * 3600 + 3600 * (upHour > 0 ? Math.min(1, (last.upkeep ?? 0) / upHour) : 0);
  const upBack = upHour * Math.min(366 * 24, Math.max(0, upEnd - clockT) / 3600);
  const trim = (days: DayLedger[], hours: HourLedger[]) => {
    const later = days.filter((d) => d.day > D), laterH = hours.filter((e) => e.hour > H);
    const today = laterH.filter((e) => Math.floor(e.hour / 24) === D);
    const fares = [...later, ...today].reduce((s, e) => s + e.fares, 0);
    const keep = { days: days.filter((d) => d.day <= D), hours: hours.filter((e) => e.hour <= H) };
    const d = keep.days.find((x) => x.day === D);
    if (d) for (const e of today) { d.fares -= e.fares; d.running -= e.running; d.upkeep = (d.upkeep ?? 0) - (e.upkeep ?? 0); d.rides = (d.rides ?? 0) - (e.rides ?? 0); d.points = (d.points ?? 0) - (e.points ?? 0); }
    const carry = <T extends DayLedger | HourLedger>(list: T[], from: T[], at: (e: T) => number, key: number, make: () => T, cap: number) => {
      const trains = from.reduce((s, e) => s + e.trains, 0), build = from.reduce((s, e) => s + e.build, 0);
      if (trains || build) { const e = entryIn(list, at, key, make, cap); e.trains += trains; e.build += build; }
    };
    carry(keep.days, later, (x) => x.day, D, () => ({ day: D, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_DAYS);
    carry(keep.hours, laterH, (x) => x.hour, H, () => ({ hour: H, fares: 0, running: 0, trains: 0, build: 0, rides: 0 }), LEDGER_HOURS);
    return { ...keep, fares };
  };
  const net = trim(ledger, hourLedger);
  ledger = net.days;
  hourLedger = net.hours;
  for (const [k, r] of regions) { const t = trim(r.days, r.hours); regions.set(k, { days: t.days, hours: t.hours }); }
  return back + upBack - net.fares;
}

/** T-867: the network's upkeep, US$M a game hour: it drains evenly through the day, trains or none. */
function upkeepPerHour(): number {
  return (api?.upkeep()[0] ?? 0) / 24;
}
function upkeepNow(): MoneyState["upkeep"] {
  const u = api!.upkeep();
  return { track: u[1], stations: u[2], km: u[3], count: u[4], buses: u[5] ?? 0, busKm: u[6] ?? 0 };
}

function moneyState(): MoneyState {
  const a = api!;
  dayEntry(Math.floor(moneyNow() / 86400));
  hourEntry(Math.floor(moneyNow() / 3600));
  return {
    cash: a.cash(),
    mode,
    fleet: a.fleet(),
    carsNeeded: a.cars_needed(),
    fleetFast: a.fleet_fast(),
    carsNeededFast: a.cars_needed_fast(),
    fleetBus: a.fleet_bus(),
    busesNeeded: a.buses_needed(),
    busPrice: moneyParams[11],
    runCostBusKm: moneyParams[12],
    upkeepBusKm: moneyParams[13],
    busRouteKm: moneyParams[14],
    hsCarPrice: moneyParams[4],
    hsRunCostCarKm: moneyParams[5],
    hsRunCostTrainKm: moneyParams[6],
    carPrice: moneyParams[0],
    runCostCarKm: moneyParams[1],
    runCostTrainKm: moneyParams[3],
    upkeepTrackKm: moneyParams[7],
    upkeepStation: moneyParams[8],
    upkeepSingle: moneyParams[9],
    upkeepHsr: moneyParams[10],
    upkeep: upkeepNow(),
    economy: moneyParams[2],
    fares: { ...fares },
    runningPerHour: [0, 1, 2].map((l) => a.running_cost(l)) as [number, number, number],
    ledger: ledger.map((d) => ({ ...d })),
    hours: hourLedger.map((d) => ({ ...d })),
    regions: savedRegions(),
    stars: Object.fromEntries(stars),
    goals: Object.fromEntries(goals),
    notes: Object.fromEntries(lineNotes),
  };
}

function sendMoney() {
  const m = moneyState();
  sentCash = m.cash;
  sentFares = m.ledger[m.ledger.length - 1]?.fares ?? NaN;
  send({ kind: "money", money: m });
}

/** T-284: the cash and today's ledger, if either moved since last sent. */
function sendCash() {
  const cash = api!.cash(), today = dayEntry(Math.floor(moneyNow() / 86400)), hour = hourEntry(Math.floor(moneyNow() / 3600));
  if (cash === sentCash && today.fares === sentFares) return;
  sentCash = cash;
  sentFares = today.fares;
  send({ kind: "cash", cash, today: { ...today }, hour: { ...hour }, regions: regionNow() });
}

/** One local time's fares and rides in one country (T-872): per kind of day and period. */
interface CountryPart { kindOf: KindOf; income: number[][]; rides: number[][] }

/** T-309: the running lines grouped by the local time they run on, each group's running cost per
 * demand level (the same on every kind of day) and its fare income per kind of day and period,
 * split by country (T-872, T-262: each country's holidays and weekend decide its kinds of day). With one local time and
 * one country (every game in the Northeast) the network's totals. */
function localGroups(): { shift: number; running: number[]; tail: number[]; parts: CountryPart[] }[] {
  const a = api!;
  const shifts = [...a.running_shifts()];
  const costs = regionLines().costs;
  const tail = (s: number | null) => PERIODS.map((_, p) => costs.reduce((t, c) => t + (s === null || c.shift === s ? c.tail[p] : 0), 0));
  const countries = [...new Set([...lineIncome.keys(), ...lineRides.keys()].map((id) => lineCountries.get(id) ?? ""))];
  const one = shifts.length <= 1;
  if (one && countries.length <= 1) return [{ shift: shifts[0] ?? 0, running: [0, 1, 2].map((l) => a.running_cost(l)), tail: tail(null), parts: [{ kindOf: kindOfIn(holidayData, countries[0]), income: incomePerHour, rides: ridesPerHour }] }];
  return (one ? [shifts[0] ?? 0] : shifts).map((s) => {
    const sum = (zero: number[][], by: Map<number, number[][]>, c: string) => {
      const out = zero.map((p) => p.map(() => 0));
      for (const [id, per] of by) {
        if ((!one && a.line_shift(id) !== s) || (lineCountries.get(id) ?? "") !== c) continue;
        per.forEach((row, k) => row.forEach((v, p) => { if (out[k]) out[k][p] += v; }));
      }
      return out;
    };
    const parts = countries.map((c) => ({ kindOf: kindOfIn(holidayData, c), income: sum(incomePerHour, lineIncome, c), rides: sum(ridesPerHour, lineRides, c) }));
    return { shift: s, running: [0, 1, 2].map((l) => (one ? a.running_cost(l) : a.running_cost_in(l, s))), tail: tail(one ? null : s), parts };
  });
}

/** Fares in and running costs out up to now (T-284: to the game second, not the whole hour). An
 * hour's fare (`fareHour`: the period's income shaped as the riders arrive, landing as rides end)
 * is paid as its trains pick up riders (`hourShape`), its running cost (`costHour`) evenly; by the
 * hour's end both are paid in full. Returns whether an hour ended. */
function settle(): boolean {
  if (!api) return false;
  const now = gameNow();
  if (now - settledT > 86400 * 366) { // a year at most in one go
    settledT = Math.floor((now - 86400 * 366) / 3600) * 3600;
    partHour = settledT / 3600;
    partFare = partCost = 0;
  }
  if (now <= settledT) return false;
  const groups = localGroups(), rl = regionLines(), up = upkeepPerHour();
  const upShares: [string, number][] = upkeepRegions.length ? upkeepRegions : [["", 1]];
  const pl = [...linePoints].map(([id, regs]) => ({ shift: api!.line_shift(id), kindOf: kindOfLine(id), regs }));
  let cash = api.cash(), ended = false;
  while (settledT < now) {
    const h = Math.floor(settledT / 3600);
    if (h !== partHour) { partHour = h; partFare = partCost = 0; }
    const t = h * 3600, end = Math.min(now, t + 3600), full = end >= t + 3600;
    // T-309: each local time's lines at their own period and day; T-260: the hour's own kind of
    // day, the weekday's until a weekend's has come; T-736: shaped and lagged (fareHour)
    let fare = 0, cost = 0, rides = 0;
    for (const g of groups) {
      const tl = t + g.shift;
      for (const c of g.parts) {
        fare += fareHour(c.income, tl, c.kindOf);
        rides += rideHour(c.rides, tl, c.kindOf);
      }
      cost += costHour(g.running, tl, g.tail);
    }
    const wf = full ? 1 : shareAt(h, end - t), wc = full ? 1 : (end - t) / 3600;
    const df = Math.max(0, wf - partFare), dc = Math.max(0, wc - partCost);
    partFare = Math.max(partFare, wf);
    partCost = Math.max(partCost, wc);
    cash += fare * df - (cost + up) * dc;
    const d = dayEntry(Math.floor(t / 86400));
    d.fares += fare * df;
    d.running += cost * dc;
    d.upkeep = (d.upkeep ?? 0) + up * dc;
    d.rides = (d.rides ?? 0) + rides * df;
    const e = hourEntry(h);
    e.fares += fare * df;
    e.running += cost * dc;
    e.upkeep = (e.upkeep ?? 0) + up * dc;
    e.rides = (e.rides ?? 0) + rides * df;
    for (const [z, [f, c, n]] of regionHour(t, rl)) {
      for (const r of regionEntries(z, d.day, h)) { r.fares += f * df; r.running += c * dc; r.rides = (r.rides ?? 0) + n * df; }
    }
    if (up * dc > 0) for (const [z, s] of upShares) {
      for (const r of regionEntries(z, d.day, h)) r.upkeep = (r.upkeep ?? 0) + up * dc * s;
    }
    // T-684: points land as fares do, each line's at its own local period and kind of day
    if (df > 0) for (const { shift, kindOf, regs } of pl) {
      for (const [z, per] of regs) {
        const v = fareHour(per, t + shift, kindOf) * df;
        if (!(v > 0)) continue;
        d.points = (d.points ?? 0) + v;
        e.points = (e.points ?? 0) + v;
        for (const r of regionEntries(z, d.day, h)) r.points = (r.points ?? 0) + v;
      }
    }
    settledT = full ? t + 3600 : end;
    if (full) { ended = true; hourShape.delete(h); }
  }
  api.set_cash(cash);
  if (kept) kept.cash = cash;
  dirty = true;
  return ended;
}

/** T-446: every line's past 24 hours (`TrackApi.past_day`, `today`'s layout), then each listed
 * line's fares taken in them, US$M, in the same order: the hour now by the share already paid, as
 * `settle` pays the network's, and the hour a day ago by the rest. */
function today(): Float64Array {
  if (settle()) sendMoney();
  return packDays(lineDaysAt(Math.min(gameNow(), settledT), false, undefined, true));
}

/** One line's day, in `TrackApi.today`'s per-line layout, with its fares (US$M). */
interface LineDay { id: number; block: ArrayLike<number>; fare: number }

/** Every line's block of `TrackApi.today(now)` with its fares taken: each hour since the line's
 * local midnight (T-309), at its local period and day, by the share already paid (`closed`: the
 * day is over, every hour paid whole). `shift` keeps the lines on one local time. `past`: the past
 * 24 hours instead (T-446). */
function lineDaysAt(now: number, closed: boolean, shift?: number, past = false): LineDay[] {
  const a = api!;
  syncHolidays();
  const v = past ? a.past_day(now) : a.today(now), count = v[0], hNow = Math.floor(now / 3600);
  const out: LineDay[] = [];
  let i = 1;
  for (let k = 0; k < count; k++) {
    const id = v[i], n = v[i + 1], spans = v[i + 6], len = 15 + 2 * spans + n;
    const sh = a.line_shift(id);
    if (shift === undefined || sh === shift) {
      const per = lineIncome.get(id);
      let f = 0;
      const h0 = past ? hNow - 24 : Math.floor((now + sh) / 86400) * 24 - sh / 3600;
      const wNow = closed ? 1 : hNow === partHour ? partFare : 0;
      if (per) for (let h = h0; h <= hNow; h++) {
        const t = h * 3600 + sh, w = h === hNow ? wNow : past && h === h0 ? 1 - wNow : 1;
        f += fareHour(per, t, kindOfLine(id)) * w;
      }
      out.push({ id, block: v.subarray(i, i + len), fare: f });
    }
    i += len;
  }
  return out;
}

/** The flat array the panels read (game/today.ts `readToday`): the line count, each line's block,
 * then each line's fares. */
function packDays(items: LineDay[]): Float64Array {
  const len = items.reduce((s, d) => s + d.block.length, 1);
  const out = new Float64Array(len + items.length);
  out[0] = items.length;
  let i = 1;
  for (const d of items) { out.set(d.block, i); i += d.block.length; }
  items.forEach((d, k) => { out[i + k] = d.fare; });
  return out;
}

/** The share of hour `h`'s fares paid by `s` seconds into it (T-284). */
function shareAt(h: number, s: number): number {
  const c = hourShape.get(h);
  if (!c) return s / 3600;
  return c[Math.max(0, Math.min(3600, Math.floor(s)))];
}

/** T-284: when hour `epoch`'s fares are paid, from its slice of the boarding account
 * (game/boarding.ts `readBoardHour` has the layout): each train leaving a platform in the hour
 * takes the riders waiting less those it leaves behind, and the hour's fares are paid in
 * proportion as they board. Only the timing within the hour comes from here. */
function shapeHour(epoch: number, b: Float32Array) {
  const h = Math.round(epoch / 3600);
  for (const k of hourShape.keys()) if (k < partHour - 1) hourShape.delete(k);
  const nt = b[0], nl = b[1], nq = b[2], ne = b[3], S = 12;
  const q0 = 5 + 2 * nt + nl, e0 = q0 + nq * S, p0 = e0 + ne * 2;
  const bins = new Float64Array(3600);
  let total = 0;
  for (let q = 0; q < nq; q++) {
    const Q = q0 + q * S, rate = b[Q + 4], first = b[Q + 8], n = b[Q + 9];
    let p = b[Q + 10];
    const pEnd = p + b[Q + 11];
    for (let j = first + 1; j < first + n; j++) {
      const t = b[e0 + j * 2];
      const tPrev = b[e0 + (j - 1) * 2], leftPrev = b[e0 + (j - 1) * 2 + 1];
      // riders changing from other lines' trains since the train before (T-554)
      let changed = 0;
      for (; p < pEnd && b[p0 + p * 2] <= t; p++) if (b[p0 + p * 2] > tPrev) changed += b[p0 + p * 2 + 1];
      if (!(t >= 0 && t < 3600)) continue;
      const w = Math.max(0, leftPrev + rate * Math.max(0, t - tPrev) + changed - b[e0 + j * 2 + 1]);
      bins[Math.floor(t)] += w;
      total += w;
    }
  }
  if (!(total > 0)) { hourShape.delete(h); return; }
  const c = new Float32Array(3601);
  let acc = 0;
  for (let s = 0; s < 3600; s++) { c[s] = acc / total; acc += bins[s]; }
  c[3600] = 1;
  hourShape.set(h, c);
}

// ---------------------------------------------------------------- saves (T-029)

function header(): SaveHeader {
  return { game: "anitabuilder", format: SAVE_FORMAT, city, region: WORLD_ID, cities: CITIES.map(c => c.id), clock: gameNow(), cash: api!.cash(), fleet: api!.fleet(), fleetFast: api!.fleet_fast(), ...(api!.fleet_bus() ? { fleetBus: api!.fleet_bus() } : {}), fares: { ...fares }, ledger: ledger.map(rounded), hours: hourLedger.map(rounded), regions: savedRegions(rounded), stars: Object.fromEntries(stars), goals: Object.fromEntries(goals), notes: Object.fromEntries(lineNotes), mode, start: startZone, opened: opened && [...opened], newCars: newLineCars };
}

/** The game as file bytes. The network and header are taken at once, before anything awaits. */
function saveBytes(): Promise<Uint8Array> {
  settle();
  return encodeSave(header(), api!.save());
}

/** T-241: the saves list's picture of the network and its figures. Read at once, before anything
 * awaits (call it straight after saveBytes); a failure leaves the picture out, never the save. */
// T-251: drawn at twice the size the saves list shows it (240 x 160, T-894), for sharp lines on high-density screens
const THUMB_W = 480, THUMB_H = 320;
/** T-894: one panel per city, the most stations first, at most four; fractions of the picture */
const THUMB_LAYOUTS: [number, number, number, number][][] = [
  [[0, 0, 1, 1]],
  [[0, 0, 0.5, 1], [0.5, 0, 0.5, 1]],
  [[0, 0, 0.6, 1], [0.6, 0, 0.4, 0.5], [0.6, 0.5, 0.4, 0.5]],
  [[0, 0, 0.5, 0.5], [0.5, 0, 0.5, 0.5], [0, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
];
/** canvas px: between panels; a panel's margin, with room at the top for the HTML name over it */
const THUMB_GAP = 4, THUMB_PAD = 14, THUMB_TOP = 34;
/** the theme the page is in (main.ts sends it): a thumbnail is drawn on that theme's land (T-543) */
let thumbShade: "light" | "dark" = "light";
async function slotSummary(): Promise<SlotSummary> {
  // a save never waits long on them: without them a station takes the nearest centre's name
  await Promise.race([outlinesRead, new Promise((go) => setTimeout(go, 3000))]);
  const a = api!;
  // stations and the cities they are in
  const ni = a.nodes_info();
  const byCity = new Map<string, number>();
  const stationXY: { city: string | null; x: number; y: number }[] = [];
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] <= 0) continue;
    const [lng, lat] = packLngLat(ni[i + 1], ni[i + 2]);
    const c = cityWithin(lng, lat); // T-318: a station over dead space is in no city
    if (c) byCity.set(c, (byCity.get(c) ?? 0) + 1);
    const [x, y] = toLocal(lng, lat);
    stationXY.push({ city: c, x, y });
  }
  const stations = stationXY.length;
  const cities = [...byCity].sort((x, y) => y[1] - x[1]).map(([c]) => c);
  const li = a.lines_info();
  const summary: SlotSummary = { cities, lines: li.length / 22, stations, cash: a.cash(), mode };
  try {
    if (typeof OffscreenCanvas === "undefined") return summary;
    // track (grey) and each line's route (its colour), in local units: what the renderer draws
    const track: ArrayLike<number>[] = [];
    const ei = a.edges_info();
    for (let i = 0; i < ei.length; i += 10) {
      const id = ei[i];
      let d: ArrayLike<number> | undefined = edgeDrawn.get(id);
      if (!d) {
        const r = a.edge_render(id), out = new Float32Array(r.length);
        for (let k = 0; k < r.length; k += 3) out.set([...packLocal(r[k], r[k + 1]), r[k + 2]], k);
        d = out;
      }
      const xy = new Float32Array((d.length / 3) * 2);
      for (let k = 0, j = 0; k < d.length; k += 3, j += 2) { xy[j] = d[k]; xy[j + 1] = d[k + 1]; }
      track.push(xy);
    }
    const routes: { colour: string; xy: ArrayLike<number> }[] = [];
    for (let i = 0; i < li.length; i += 22) {
      if (li[i + 2] !== 1) continue;
      const id = li[i];
      let d: ArrayLike<number> | undefined = lineDrawn.get(id);
      if (!d) {
        const p = a.run_polyline(id, 0, SAMPLE_STEP), out = new Float32Array(p.length);
        for (let k = 0; k < p.length; k += 2) out.set(packLocal(p[k], p[k + 1]), k);
        d = out;
      }
      routes.push({ colour: hex(li[i + 1]), xy: d });
    }
    type Box = [number, number, number, number];
    const boxOf = (pts: Iterable<[number, number]>): Box => {
      const b: Box = [Infinity, Infinity, -Infinity, -Infinity];
      for (const [x, y] of pts) { b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y); b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y); }
      return b;
    };
    function* points(xy: ArrayLike<number>): Generator<[number, number]> { for (let k = 0; k < xy.length; k += 2) yield [xy[k], xy[k + 1]]; }
    const drawn = [...track.map((xy) => ({ xy, colour: "" })), ...routes].map((p) => ({ ...p, box: boxOf(points(p.xy)) })).filter((p) => p.box[2] >= p.box[0]);
    const all = boxOf(drawn.flatMap((p) => [[p.box[0], p.box[1]], [p.box[2], p.box[3]]] as [number, number][]));
    if (!(all[2] >= all[0])) return summary; // nothing built or drawn
    // T-894: the biggest cities by stations, each framed on its own stations; a city with under 8%
    // of the biggest's stations is a fringe of its network, not a panel. A network with no station
    // in a city is one panel over all of it, unnamed.
    const most = cities.length ? byCity.get(cities[0])! : 0;
    const shown = cities.filter((c) => byCity.get(c)! >= 0.08 * most).slice(0, THUMB_LAYOUTS.length);
    const frames = shown.length
      ? shown.map((c) => ({ city: c, stations: byCity.get(c)!, box: boxOf(stationXY.filter((p) => p.city === c).map((p) => [p.x, p.y] as [number, number])) }))
      : [{ city: "", stations, box: all }];
    const layout = THUMB_LAYOUTS[frames.length - 1];
    const sh = SHADES[thumbShade];
    const canvas = new OffscreenCanvas(THUMB_W, THUMB_H);
    const ctx = canvas.getContext("2d")!;
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.fillStyle = sh.casing; // the rule between panels
    ctx.fillRect(0, 0, THUMB_W, THUMB_H);
    summary.shade = thumbShade;
    summary.panels = [];
    frames.forEach((f, n) => {
      const [fx, fy, fw, fh] = layout[n];
      // panel in canvas px, the gap split between neighbours
      const px0 = fx * THUMB_W + (fx > 0 ? THUMB_GAP / 2 : 0), py0 = fy * THUMB_H + (fy > 0 ? THUMB_GAP / 2 : 0);
      const px1 = (fx + fw) * THUMB_W - (fx + fw < 1 ? THUMB_GAP / 2 : 0), py1 = (fy + fh) * THUMB_H - (fy + fh < 1 ? THUMB_GAP / 2 : 0);
      const pw = px1 - px0, ph = py1 - py0;
      summary.panels!.push({ city: f.city, stations: f.stations, x: px0 / THUMB_W, y: py0 / THUMB_H, w: pw / THUMB_W, h: ph / THUMB_H });
      // the frame: the city's stations inside the margins, at least 3 km across so a short stretch is not blown up
      const top = f.city ? THUMB_TOP : THUMB_PAD;
      const iw = pw - 2 * THUMB_PAD, ih = ph - top - THUMB_PAD;
      const [x0, y0, x1, y1] = f.box;
      const s = iw / Math.max(x1 - x0, (y1 - y0) * iw / ih, 3000);
      const ox = px0 + THUMB_PAD + iw / 2 - ((x0 + x1) / 2) * s, oy = py0 + top + ih / 2 - ((y0 + y1) / 2) * s;
      // what lies inside the panel, in local units
      const v: Box = [(px0 - ox) / s, (py0 - oy) / s, (px1 - ox) / s, (py1 - oy) / s];
      const inside = (b: Box) => b[2] >= v[0] && b[0] <= v[2] && b[3] >= v[1] && b[1] <= v[3];
      ctx.save();
      ctx.beginPath();
      ctx.rect(px0, py0, pw, ph);
      ctx.clip();
      ctx.fillStyle = sh.land;
      ctx.fillRect(px0, py0, pw, ph);
      // a smaller panel draws thinner strokes
      const k = Math.max(0.8, Math.sqrt((pw * ph) / (THUMB_W * THUMB_H)));
      for (const p of drawn) {
        if (!inside(p.box)) continue;
        ctx.strokeStyle = p.colour || sh.thumbTrack;
        ctx.lineWidth = (p.colour ? 6 : 3) * k;
        ctx.beginPath();
        let lx = NaN, ly = NaN;
        for (let j = 0; j < p.xy.length; j += 2) {
          const qx = ox + p.xy[j] * s, qy = oy + p.xy[j + 1] * s;
          if (j > 0 && j < p.xy.length - 2 && Math.hypot(qx - lx, qy - ly) < 0.8) continue;
          if (j === 0) ctx.moveTo(qx, qy); else ctx.lineTo(qx, qy);
          lx = qx; ly = qy;
        }
        ctx.stroke();
      }
      ctx.restore();
    });
    const png = new Uint8Array(await (await canvas.convertToBlob({ type: "image/png" })).arrayBuffer());
    let bin = "";
    for (let k = 0; k < png.length; k += 0x8000) bin += String.fromCharCode(...png.subarray(k, k + 0x8000));
    summary.thumb = "data:image/png;base64," + btoa(bin);
  } catch { /* no picture */ }
  return summary;
}

function autosave(): Promise<boolean> {
  if (saveInFlight) return saveInFlight.then(ok => dirty && ok ? autosave() : ok);
  if (!api || !dirty) return Promise.resolve(true);
  if (!activeLease || switching || cloudTaken === activeSlot) return Promise.resolve(false);
  // T-677: a new game whose slot was never written is not kept while its network is empty
  if (unwritten === activeSlot && !api.nodes_info().length && !api.lines_info().length) {
    dirty = false;
    return Promise.resolve(true);
  }
  const key = activeSlot; // an in-flight write must never move to the next slot
  const force = byHandNext;
  byHandNext = false;
  dirty = false;
  saveInFlight = (async () => {
    try {
      const day = Math.floor(gameNow() / 86400);
      const t0 = performance.now();
      const bytesP = saveBytes(), summaryP = slotSummary();
      const bytes = await bytesP;
      const encodeMs = performance.now() - t0;
      const summary = await summaryP;
      const at = Date.now();
      const name = newSlotName?.key === key ? newSlotName.name : undefined;
      const fresh = await writeSlot(key, { bytes, at, day, city, ...summary, name });
      if (name) newSlotName = null;
      if (unwritten === key) unwritten = null;
      const backups = await listBackups(key).catch(() => null);
      if (backups && !(at - (backups[0]?.at ?? -Infinity) < backupMs)) await writeBackup(key, { bytes, at, day, city, ...summary });
      void cloudAfterWrite(key, force, fresh); // T-988
      savedAt = at;
      saveFailure = undefined;
      send({ kind: "saved", at, bytes: bytes.length, encodeMs, totalMs: performance.now() - t0 });
      await publishSlots();
      slotChanges?.postMessage({ key });
      return true;
    } catch (err) {
      dirty = true;
      if (isPanic(err)) { panicLater(err, "autosave"); return false; } // not the browser's fault, and rebuilt then saved
      savedAt = Date.now(); // retry on the next autosave interval, rather than every tick
      saveFailure = "The browser could not save this game. Download it as a file to keep your changes.";
      await publishSlots(saveFailure);
      return false;
    } finally { saveInFlight = null; }
  })();
  return saveInFlight;
}

/** T-1251: the slot written now, autosave on or off. A tab without the slot's lease takes it from
 * the tab holding it, whose autosave then stops; an empty new game is written too. */
async function saveByHand() {
  try {
    if (testPage || !activeSlot) throw new Error("This test page keeps no saved game. Make a copy to keep it.");
    if (!activeLease) {
      const lease = await leaseSlot(activeSlot, true);
      if (!lease) throw new Error("This browser cannot protect save slots. Download the game as a file instead.");
      hold(lease);
    }
    // T-988: a game another device opened is taken back, the account's copy kept there as a backup
    if (cloudTaken === activeSlot) cloudTaken = "";
    if (unwritten === activeSlot) unwritten = null;
    saveFailure = undefined;
    dirty = true;
    byHandNext = true;
    if (!await autosave()) throw new Error(saveFailure ?? "This game could not be saved. Download it as a file to keep it.");
    send({ kind: "savedByHand", at: savedAt });
  } catch (err) {
    send({ kind: "savedByHand", error: (err as Error).message });
    await publishSlots((err as Error).message);
  }
}

async function manageSlot(msg: Extract<ToClock, { kind: "saveSlot" }>) {
  try {
    if (msg.action === "refresh") { await publishSlots(); return; }
    if (msg.action === "save") { await saveByHand(); return; }
    if (msg.action === "create") {
      const key = "slot-" + crypto.randomUUID();
      const lease = await leaseSlot(key);
      if (!lease) throw new Error("This browser cannot protect save slots. Download the game as a file instead.");
      try {
        const bytesP = saveBytes(), summaryP = slotSummary();
        const bytes = await bytesP;
        await writeSlot(key, { bytes, at: Date.now(), day: Math.floor(gameNow() / 86400), city, name: msg.name?.trim() || "Saved game", ...await summaryP });
        void cloudAfterWrite(key, false, true); // T-988: a new game on the account
        // Finish earlier writes before releasing the old slot's guard. The new copy is already safe.
        if (activeLease) await autosave();
        activeLease?.release();
        activeSlot = key;
        hold(lease);
        saveFailure = undefined;
        dirty = true;
        savedAt = Date.now();
      } catch (err) { lease.release(); throw err; }
    } else if (msg.action === "open") {
      const key = msg.key;
      if (!key || key === activeSlot) { await publishSlots(); return; }
      await flushBeforeSwitch();
      const lease = await leaseSlot(key);
      try {
        // Read after acquiring the guard: the previous owner may have saved just before release.
        const stored = await readStored(key);
        if (!stored) throw new Error("That save slot no longer exists.");
        const decoded = await decodeSave(stored.bytes);
        if (!isNortheastSave(decoded.header)) throw new Error("This save is for another world.");
        switching = true;
        const converted = await loadGame(stored.bytes, "file").finally(() => { switching = false; });
        activeLease?.release();
        activeSlot = key;
        hold(lease);
        saveFailure = undefined;
        dirty = false;
        savedAt = stored.at;
        if (converted) await saveConverted(key, stored);
      } catch (err) { lease?.release(); throw err; }
    } else if (msg.action === "restore") {
      const key = msg.key, from = activeSlot;
      if (!key || !isBackupOf(key, from)) throw new Error("That backup no longer exists.");
      const backup = await readStored(key), slot = await readStored(from);
      if (!backup) throw new Error("That backup no longer exists.");
      await intoNewSlot(forkName(slotName(from, slot), backup.at), () => loadGame(backup.bytes, "file"), true);
      return;
    } else {
      const key = msg.key;
      if (!key) throw new Error("Choose a save slot first.");
      if (msg.action === "delete" && key === activeSlot) throw new Error("Open another slot before deleting this one.");
      const temporary = key === activeSlot ? null : await leaseSlot(key);
      if (key === activeSlot ? !activeLease : !temporary) throw new Error("That slot is open in another tab. Close it before changing this save.");
      try {
        if (msg.action === "rename") await renameSlot(key, msg.name ?? "");
        else {
          await forgetSlot(key); // T-988: from the account too
          await deleteSlot(key);
        }
      } finally { temporary?.release(); }
      // T-988: a rename goes up at once; a deletion now, or at the next chance
      const ctx = cloudCtx;
      if (ctx && msg.action === "rename") void cloudRun(async () => { if (await upload(ctx, key) === "ok" && key === activeSlot) cloudClaimed = key; await publishSlots(); });
      else if (ctx) void cloudRun(() => sendDeletes(ctx));
    }
    await publishSlots();
    slotChanges?.postMessage({ key: activeSlot });
  } catch (err) { await publishSlots((err as Error).message); }
}

/** whether the game just loaded was converted from an older frame (T-308), for the "loaded" message */
let convertedOnLoad = false;

/** Replace the game with a save's. Throws a player-readable reason, changing nothing. A save whose
 * track is not in the world frame is converted once, through longitude and latitude, and loaded in
 * it (T-308); the return value says so, and the caller then writes the slot again in the new
 * format. Money, fares and the ledger load as saved, but for anything settled after the clock
 * (`unsettleAfter`). With `fresh`, the save is a real network a new game starts from (T-731): only
 * its network, fleet and fares are kept, the rest is a new game's. */
async function loadGame(bytes: Uint8Array, from: "autosave" | "file" | "new", fresh?: NewGameSpec): Promise<boolean> {
  const decoded = await decodeSave(bytes);
  const { track } = decoded;
  const h: SaveHeader = currentZones(fresh
    ? { ...decoded.header, clock: startClock(fresh), cash: START_CASH, ledger: [], hours: [], regions: undefined, stars: undefined, goals: undefined, notes: undefined, newCars: undefined, ...newGameHeader(fresh) }
    : decoded.header);
  if (!isNortheastSave(h)) throw new Error("This save is for another world.");
  if (isRetiredTrackFormat(track)) throw new Error("This save is from an older version of the game and no longer opens.");
  let net = track, converted = false;
  if (trackFrameKind(track) !== NEW_GAME_FRAME.kind) {
    const c = TrackApi.convert_save_frame(track, frameKindNumber(NEW_GAME_FRAME.kind));
    // an unreadable network converts to nothing: let the load below say so
    if (c.length) { net = c; converted = true; }
  }
  // T-889: the demand pool opens the zones the built stations reach while the network is worked out
  const st = TrackApi.save_stations(net);
  if (st.length > 3) send({ kind: "stationsAhead", frame: { kind: frameKindName(st[0]), lon: st[1], lat: st[2] }, xy: st.slice(3) });
  // T-1539: the demand result kept for this save's network can show while the network is worked out
  // (China's 17-42 s); awaited here, since the load below holds the worker
  if (st.length > 3) await saveNetKey(net, h).then((key) => send({ kind: "netAhead", key }), () => {});
  await openNetwork(net);
  api!.set_fleet(h.fleet ?? 0);
  api!.set_fleet_fast(h.fleetFast ?? 0);
  api!.set_fleet_bus(h.fleetBus ?? 0);
  fares = loadFares(h.fares);
  ledger = mergeLedger(h.ledger, (d) => d.day);
  hourLedger = mergeLedger(h.hours, (d) => d.hour);
  loadRegions(h.regions);
  stars = new Map(Object.entries(h.stars ?? {}));
  goals = new Map(Object.entries(h.goals ?? {}));
  lineNotes = new Map(Object.entries(h.notes ?? {}));
  newLineCars = h.newCars ?? 0;
  api!.set_cash(h.cash + unsettleAfter(h.clock));
  setMode(h.mode === "sandbox" ? "sandbox" : "normal");
  startZone = h.start;
  opened = h.opened;
  convertedOnLoad = converted;
  restart(h.clock, from);
  return converted;
}

/** A new game's clock (T-728): the first time from day 1 07:00 on the game clock at which it is
 * 07:00 in the start city, so every start opens on its own morning peak (Tokyo: Tue 2 07:00 there). */
function startClock(spec?: NewGameSpec): number {
  const c = spec?.start ? CITIES.find((x) => x.id === spec.start) : undefined;
  const ahead = c ? shiftAt(c.center[0], c.center[1]) : 0;
  return START_CLOCK + (((-ahead % 86400) + 86400) % 86400);
}

/** A new game's mode, start and opened zones (T-728). */
function newGameHeader(spec?: NewGameSpec): Pick<SaveHeader, "mode" | "start" | "opened"> {
  return { mode: spec?.mode ?? "normal", start: spec?.start, opened: spec?.start ? [spec.start] : [] };
}
function sendOpened() {
  send({ kind: "opened", zones: opened ? [...opened] : null, start: startZone });
}
function setNewGame(spec?: NewGameSpec) {
  const g = newGameHeader(spec);
  setMode(g.mode!);
  startZone = g.start;
  opened = g.opened;
}

/** After a slot's save was converted on load: keep the old-format copy under a key the saves list
 * does not show, then write the slot in the new format. */
async function saveConverted(key: string, old: StoredSave) {
  await writeStored("old-frame-" + key, old);
  dirty = true;
  await autosave();
}

/** A new game: empty, or from `spec.real`'s network (T-731). */
async function newGame(spec?: NewGameSpec) {
  if (spec?.real) {
    await loadGame(new Uint8Array(spec.real), "new", spec);
    return;
  }
  setNewGame(spec);
  api!.load(emptyTrack!);
  api!.set_cash(START_CASH);
  api!.set_fleet(0);
  api!.set_fleet_fast(0);
  api!.set_fleet_bus(0);
  fares = { ...START_FARES };
  ledger = [];
  hourLedger = [];
  loadRegions(undefined);
  stars = new Map();
  goals = new Map();
  lineNotes = new Map();
  newLineCars = 0;
  restart(startClock(spec), "new");
}

/** After a load or a new game: the clock, fresh state, a new snapshot for demand. */
function restart(t: number, from: "autosave" | "file" | "new") {
  clk = { t, rate: clk.rate, at: wallNow() };
  game++;
  // T-284: the loaded hour counts as paid up to the clock, evenly (its trains' timing is not known yet)
  settledT = t;
  partHour = Math.floor(t / 3600);
  partFare = partCost = (t - partHour * 3600) / 3600;
  hourShape.clear();
  incomePerHour =DAY_KINDS.map(() => PERIODS.map(() => 0)); // the main thread sends the new network's once demand lands
  ridesPerHour = DAY_KINDS.map(() => PERIODS.map(() => 0));
  lineIncome = new Map(); // T-715: the old game's per line, which regions' fares were paid from
  lineRides = new Map();
  linePoints = new Map();
  api!.clear_riders(); // and its riders (T-164)
  forget(true);
  syncFrame();
  keep();
  sendOpened();
  send({ kind: "loaded", from, clock: t, frame: frameSpec(), converted: convertedOnLoad || undefined, game });
  convertedOnLoad = false;
  sendState();
  clearTimeout(snapTimer);
  send({ kind: "snapshot", snapshot: buildSnapshot() });
}

/** the last build's pieces of the renderer's buffers (T-496) */
const netCache = newNetCache();
/** the lines with trains at the last full build */
let runningLines = "";

/** `service`: only the service changed (a train count, T-229): the renderer's buffers are not built
 * again, which was about half the edit's time on New York's real network. */
function sendState(service = false, effectiveAt?: number) {
  const t0 = performance.now();
  const s = buildState();
  const t1 = performance.now();
  if (effectiveAt !== undefined) s.effectiveAt = effectiveAt;
  moneyLag = runLag(s.lines.filter((l) => l.ok && l.running).map((l) => ({ tph: l.tph[1], lengthM: l.lengthM, cars: l.cars, oneWayS: l.roundTrip[1] / 2 - l.turnaround })));
  // a line gaining its first train or losing its last changes the draw order (T-533)
  const running = s.lines.filter((l) => l.ok && runsTrains(l)).map((l) => l.id).join();
  const r = s.render;
  // the per-edge arrays stay here (T-1029): the network carries what the main thread needs of them
  s.render = { phases: r.phases, meta: r.meta, samples: r.samples, lineTable: r.lineTable };
  if (service && running === runningLines && posted.net) {
    s.service = true;
    s.ms = [t1 - t0, 0];
    timing.state = t1 - t0;
    keepPosted("phases", r.phases);
    keepPosted("meta", r.meta);
    postState(s, [r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
    timing.post = performance.now() - t1;
    sendTrips();
    return;
  }
  // The renderer's buffers are built here too, so the main thread only uploads them (T-063).
  runningLines = running;
  // T-1528: complexes only where a waiting room or car park is drawn once for one
  const complexOf = s.nodes.some((x) => ((x.upgrades ?? 0) & (UPGRADE.parkRide | UPGRADE.waitingRoom)) !== 0) ? complexOfNodes() : undefined;
  const n = networkFromState({ ...s, render: r, complexOf }, netCache, true);
  const t2 = performance.now();
  const out = (s.net = netForPost(n, patchMode === "check" ? networkFromState({ ...s, render: r, complexOf }) : undefined));
  s.ms = [t1 - t0, t2 - t1];
  timing.state = t1 - t0;
  timing.buffers = t2 - t1;
  const bufs = new Set<ArrayBufferLike>([r.phases.buffer, r.meta.buffer, r.samples.buffer, r.lineTable.buffer]);
  for (const st of [out.track, out.lines]) for (const v of Object.values(st)) if (ArrayBuffer.isView(v) && v.byteLength) bufs.add(v.buffer);
  for (const k of SAME_FIELDS) { const v = out[k]; if (v) bufs.add(v.buffer); }
  for (const v of out.edgePts.values()) bufs.add(v.buffer);
  if (out.patch) for (const v of [out.patch.track, out.patch.lines, out.patch.edgeIds]) bufs.add(v.buffer);
  postState(s, [...bufs] as Transferable[]);
  timing.post = performance.now() - t2;
  sendTrips();
}

// ---------------------------------------------------------------- posting only what changed (T-1029)

/** `?patch=` of the page: "0" posts every state and network whole, "check" also builds each network
 * from scratch and compares (the client compares the merged state too); otherwise changes only */
let patchMode = "1";
/** the network's typed arrays that, when as last posted, are left out of the message (`NetPatch.same`) */
const SAME_FIELDS = ["phases", "meta", "samples", "lineTable", "lineCars", "lineColours", "sampleLane", "sampleStretch", "sampleStroke", "stations", "stationAxis", "stationShape", "stationLine", "stationLength", "stationLinkIdx", "stationLinks", "lots", "junctions", "marks"] as const;
type SameField = (typeof SAME_FIELDS)[number];
/** What the client holds: the last network posted (its serial, draw anchor, a copy of each
 * `SAME_FIELDS` array and its edges' points) and the rows of the last state, by id. Cleared to post
 * the next state whole. */
const posted = {
  net: null as null | { serial: number; anchor: number; pts: Map<number, Float32Array> },
  arrays: new Map<string, ArrayBufferView>(),
  edges: new Map<number, ClockEdge>(), nodes: new Map<number, ClockNode>(), lines: new Map<number, ClockLine>(),
  edgeIds: [] as number[], nodeIds: [] as number[], lineIds: [] as number[],
};
const sender = new StrokeSender();
function postWhole() {
  posted.net = null;
  sender.forget();
  posted.arrays.clear();
  posted.edges.clear(); posted.nodes.clear(); posted.lines.clear();
  posted.edgeIds = []; posted.nodeIds = []; posted.lineIds = [];
}

const bytesOf = (v: ArrayBufferView) => new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
function sameBytes(a: ArrayBufferView | undefined, b: ArrayBufferView): boolean {
  if (!a || a.byteLength !== b.byteLength || a.constructor !== b.constructor) return false;
  if (a.byteLength % 4 === 0 && a.byteOffset % 4 === 0 && b.byteOffset % 4 === 0) {
    const x = new Int32Array(a.buffer, a.byteOffset, a.byteLength / 4), y = new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  }
  const x = bytesOf(a), y = bytesOf(b);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}
function keepPosted(k: string, v: ArrayBufferView) {
  posted.arrays.set(k, (v as Float32Array).slice());
}

/** The network as it is posted: whole (a copy of its edges' points in one buffer), or with
 * `patch` and the fields the client already has left out. */
function netForPost(n: NetworkBuffers, scratch?: NetworkBuffers): NetworkBuffers {
  const anchorNow = anchor().version;
  const pts = n.edgePts;
  const whole = patchMode === "0" || !n.patch || !posted.net || posted.net.serial !== n.patch.base || posted.net.anchor !== anchorNow;
  const pieces = n.strokePieces!;
  const out: NetworkBuffers = { ...n };
  delete out.strokePieces;
  const t0 = performance.now();
  const modes = {} as Record<StrokeKind, StrokeMode>, ranges = {} as Record<StrokeKind, Int32Array>;
  for (const kind of ["track", "lines"] as const) {
    const r = sender.post(kind, pieces[kind], n[kind].count, whole ? null : n.patch![kind], whole ? 0 : kind === "track" ? n.patch!.oldTrack : n.patch!.oldLines);
    out[kind] = r.strokes;
    modes[kind] = r.mode;
    ranges[kind] = r.ranges;
  }
  timing.strokes = performance.now() - t0;
  timing.modes = (["whole", "same", "patch"] as const).indexOf(modes.track) * 10 + (["whole", "same", "patch"] as const).indexOf(modes.lines);
  if (scratch) {
    const full = { track: strokesOfPieces("track", pieces.track), lines: strokesOfPieces("lines", pieces.lines) };
    checkNet({ ...n, ...full }, scratch);
    out.checkStrokes = full;
  }
  if (whole) {
    delete out.patch;
    let total = 0;
    for (const p of pts.values()) total += p.length;
    const all = new Float32Array(total), mine = new Map<number, Float32Array>();
    let at = 0;
    for (const [id, p] of pts) {
      all.set(p, at);
      mine.set(id, all.subarray(at, at + p.length));
      at += p.length;
    }
    out.edgePts = mine;
    for (const k of SAME_FIELDS) { const v = n[k]; if (v) keepPosted(k, v); }
  } else {
    const patch = { ...n.patch!, same: [] as string[], modes, ranges };
    for (const k of SAME_FIELDS) {
      const v = n[k];
      if (!v) continue;
      if (sameBytes(posted.arrays.get(k), v)) {
        patch.same.push(k);
        (out as Record<SameField, unknown>)[k] = undefined;
      } else keepPosted(k, v);
    }
    const changed = new Map<number, Float32Array>();
    const was = posted.net!.pts;
    for (const [id, p] of pts) if (was.get(id) !== p) changed.set(id, p.slice());
    out.edgePts = changed;
    out.patch = patch;
  }
  posted.net = { serial: n.serial!, anchor: anchorNow, pts };
  return out;
}

/** T-1029's check (`?patch=check`): the network built from the cache against one built from scratch. */
function checkNet(n: NetworkBuffers, full: NetworkBuffers) {
  const bad: string[] = [];
  const cmp = (k: string, a: unknown, b: unknown) => {
    if (ArrayBuffer.isView(a) && ArrayBuffer.isView(b)) { if (!sameBytes(a, b)) bad.push(k); }
    else if (JSON.stringify(a) !== JSON.stringify(b)) bad.push(k);
  };
  for (const k of Object.keys(full) as (keyof NetworkBuffers & string)[]) {
    if (k === "serial" || k === "patch" || k === "anchor") continue;
    const a = n[k], b = full[k];
    if (k === "edgePts") {
      const x = a as Map<number, Float32Array>, y = b as Map<number, Float32Array>;
      if (x.size !== y.size || [...x.keys()].join() !== [...y.keys()].join() || [...y].some(([id, p]) => !sameBytes(x.get(id), p))) bad.push(k);
    } else if (b && typeof b === "object" && "seg" in b) for (const f of Object.keys(b)) cmp(`${k}.${f}`, (a as any)?.[f], (b as any)[f]);
    else cmp(k, a, b);
  }
  // the patch's runs rebuild the strokes from the last build's
  if (bad.length) console.error(`T-1029 check: the network built from the cache differs from a full build in ${bad.join(", ")}`);
}

/** `?patch=check`: the rows just read against a fresh read of the whole model (T-1382). The fresh
 * read leaves the kept rows alone and its bus rows are the same ones. Pricing a planned line (an ok
 * line not running) runs a trial that moves headway lines a little, so with one the lines are not
 * compared here; checks/t1382-rows.mjs compares them with two models. */
function checkRows(got: { edges: ClockEdge[]; nodes: ClockNode[]; lines: ClockLine[] }, a: TrackApi, ups: Map<number, number>) {
  const full = new StateRows().read(a, rowEnv, frame, ups);
  const planned = got.lines.some((l) => l.ok && !l.running);
  const keys: ("edges" | "nodes" | "lines")[] = planned ? ["edges", "nodes"] : ["edges", "nodes", "lines"];
  const bad = keys.filter((k) => !sameValue(got[k], full[k]));
  if (bad.length) console.error(`T-1382 check: the ${bad.join(", ")} read from changes differ from a full read`);
}

/** Rows field for field, arrays and typed arrays by value. */
function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return Number.isNaN(a) && Number.isNaN(b);
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!sameValue(a[i], b[i])) return false;
    return true;
  }
  if (ArrayBuffer.isView(a)) return ArrayBuffer.isView(b) && sameBytes(a, b);
  const x = a as Record<string, unknown>, y = b as Record<string, unknown>;
  const kx = Object.keys(x);
  if (kx.length !== Object.keys(y).length) return false;
  for (const k of kx) if (!sameValue(x[k], y[k])) return false;
  return true;
}

/** The rows that differ from those last posted, with every row's id in order (null when the ids
 * are as last posted); `posted` takes the new rows. */
function rowsDelta<T extends { id: number }>(rows: T[], by: Map<number, T>, ids: number[]): { ids: number[] | null; rows: T[] } {
  const changed: T[] = [];
  let sameIds = rows.length === ids.length;
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (sameIds && ids[i] !== r.id) sameIds = false;
    if (!sameValue(by.get(r.id), r)) changed.push(r);
  }
  const out = { ids: sameIds ? null : rows.map((r) => r.id), rows: changed };
  by.clear();
  for (const r of rows) by.set(r.id, r);
  return out;
}

/** Post a state, its edges, nodes and lines as only those that changed since the last state posted
 * (T-648, T-1029: copying them to the main thread and back into objects was most of a state's time
 * there on a big network). */
function postState(s: ClockState, transfer: Transferable[]) {
  if (patchMode === "0" || !posted.edgeIds.length && !posted.nodeIds.length && !posted.lineIds.length) {
    rowsDelta(s.edges, posted.edges, posted.edgeIds);
    rowsDelta(s.nodes, posted.nodes, posted.nodeIds);
    rowsDelta(s.lines, posted.lines, posted.lineIds);
    posted.edgeIds = s.edges.map((e) => e.id); posted.nodeIds = s.nodes.map((n) => n.id); posted.lineIds = s.lines.map((l) => l.id);
    send({ kind: "state", state: s }, transfer);
    return;
  }
  const de = rowsDelta(s.edges, posted.edges, posted.edgeIds), dn = rowsDelta(s.nodes, posted.nodes, posted.nodeIds), dl = rowsDelta(s.lines, posted.lines, posted.lineIds);
  if (de.ids) posted.edgeIds = de.ids;
  if (dn.ids) posted.nodeIds = dn.ids;
  if (dl.ids) posted.lineIds = dl.ids;
  const full = patchMode === "check" ? { edges: s.edges, nodes: s.nodes, lines: s.lines } : undefined;
  send({ kind: "state", state: { ...s, edges: [], nodes: [], lines: [], delta: { edges: de, nodes: dn, lines: dl }, ...(full ? { checkRows: full } : {}) } }, transfer);
}

/** Lines of the boarding account worked out per step: a few ms on New York's real network, so an
 * edit or the money tick never waits long behind new riders (T-164, notes/T-164.md). */
const BOARDING_STEP_LINES = 3;
let boardingTimer = 0;
/** time spent stepping since the last hour sent, and its longest step (`?debug=1`) */
let prep = { ms: 0, max: 0 };
function stepBoarding() {
  clearTimeout(boardingTimer);
  if (!api) return;
  const t0 = performance.now();
  const more = api.boarding_step(BOARDING_STEP_LINES);
  const ms = performance.now() - t0;
  prep = { ms: prep.ms + ms, max: Math.max(prep.max, ms) };
  if (more) boardingTimer = setTimeout(stepBoarding, 0) as unknown as number;
  else sendTrips();
}

function sendTrips() {
  if (!api || !Number.isFinite(epoch)) return;
  api.set_now(gameNow());
  syncHolidays();
  timing.trips = 0;
  timing.board = 0;
  for (const e of [epoch, epoch + 3600]) {
    // per trip: profile, departure, end and the departure its train takes next, relative to the
    // epoch; the end (T-206) and the next (T-245) go apart
    const tT = performance.now();
    const raw = api.trips(e), n = raw.length / 4;
    timing.trips += performance.now() - tT;
    const trips = new Float32Array(n * 2), ends = new Float32Array(n), nexts = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      trips[i * 2] = raw[i * 4];
      trips[i * 2 + 1] = raw[i * 4 + 1];
      ends[i] = raw[i * 4 + 2];
      nexts[i] = raw[i * 4 + 3];
    }
    // The boarding account's hour (T-164), once demand's riders are here: each trip's loads and
    // the platforms' crowds. The account is worked out again only for lines whose service or
    // riders changed.
    if (api.has_riders()) {
      const t0 = performance.now();
      const board = api.boarding_hour(e);
      shapeHour(e, board); // T-284: when this hour's fares land, read before the slice is sent
      timing.board += performance.now() - t0;
      send({ kind: "trips", version, epoch: e, trips, ends, nexts, board, boardMs: performance.now() - t0, redone: api.boarding_redone(), prepMs: prep.ms, prepMax: prep.max }, [trips.buffer, ends.buffer, nexts.buffer, board.buffer]);
      prep = { ms: 0, max: 0 };
    } else send({ kind: "trips", version, epoch: e, trips, ends, nexts }, [trips.buffer, ends.buffer, nexts.buffer]);
  }
}

/** An edit's charge (US$M, `trains` of it for trains) into the money record of the hour and day. */
function bookCharge(seq: number, charge: number, trains: number) {
  const d = dayEntry(Math.floor(moneyNow() / 86400));
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
const groundK = (y: number) => (frame.kind === "merc" ? mercK(y) : 1);
function complexesOf(builder: ComplexBuilder, geo: { id: string; x: number; y: number; heading: number; length: number }[], links: [string, string][]): Complexes {
  if (patchMode === "0") return buildComplexes(geo, links, groundK);
  const cx = builder.update(geo, links, groundK, frame.kind);
  if (patchMode === "check") {
    const full = buildComplexes(geo, links, groundK);
    if (cx.of.size !== full.of.size || cx.members.length !== full.members.length || [...full.of].some(([id, c]) => cx.of.get(id) !== c) || full.members.some((m, i) => m.join() !== cx.members[i].join())) {
      console.error("T-1380 check: the complexes patched from changes differ from a whole build");
    }
  }
  return cx;
}

function buildSnapshot(): NetworkSnapshot {
  const a = api!;
  const ni = a.nodes_info();
  const stations: NetworkSnapshot["stations"] = [];
  const geo: { id: string; x: number; y: number; heading: number; length: number }[] = [];
  const ups = upgradesOf(a);
  for (let i = 0; i < ni.length; i += 10) {
    if (ni[i + 7] > 0 && ni[i + 8] === 1) {
      const u = ups.get(ni[i]);
      stations.push({ id: ni[i], name: stateRows.nameOf(a, ni[i], frame), x: ni[i + 1], y: ni[i + 2], level: ni[i + 3] as any, platform: ni[i + 7], ...(u ? { upgrades: u } : {}) });
      geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
    }
  }
  // T-161: each constructed station's complex, among constructed stations only (a blueprint
  // platform carries nobody, so it joins nothing), with the player's links (T-197)
  const fl = a.station_links(), links: [string, string][] = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = complexesOf(snapshotComplexes, geo, links);
  stations.forEach((s) => (s.complex = cx.of.get(String(s.id))));
  // T-1281: the links are passages, walked faster; between constructed stations only
  const built = new Set(geo.map((g) => g.id));
  const passages = links.filter(([p, q]) => built.has(p) && built.has(q)).map(([p, q]) => [Number(p), Number(q)] as [number, number]);
  const li = a.lines_info();
  const lines: NetworkSnapshot["lines"] = [];
  for (let i = 0; i < li.length; i += 22) {
    if (li[i + 3] !== 1) continue; // running lines only
    const id = li[i];
    const bus = a.is_bus(id) ? busSnapshot(a, id, stations) : null;
    const hopS: number[][][] = [], dwellS: number[][][] = [];
    const span = a.line_span(id);
    for (let run = 0; run < 2; run++) {
      hopS.push([]);
      dwellS.push([]);
      for (let lev = 0; lev < 3; lev++) {
        const arr = a.stop_times(id, run, lev), dep = a.stop_departures(id, run, lev);
        hopS[run].push(Array.from({ length: arr.length - 1 }, (_, k) => arr[k + 1] - dep[k]));
        dwellS[run].push(Array.from(arr, (t, k) => (k === 0 || k === arr.length - 1 ? 0 : dep[k] - t)));
      }
    }
    lines.push({
      id, name: a.line_name(id), stops: [...a.line_stops(id)], tph: [li[i + 5], li[i + 6], li[i + 7]], hopS, dwellS,
      hopKm: [0, 1].map((run) => { const s = a.line_stop_s(id, run); return Array.from({ length: s.length - 1 }, (_, k) => (s[k + 1] - s[k]) / 1000); }),
      turnaroundS: li[i + 20], cars: bus ? BUS_DEMAND_CARS : li[i + 21],
      ...(bus ? { catches: bus } : {}),
      // T-798: a circle's riders ride on round through its first stop, standing the dwell there
      ...(a.line_loop(id) === 1 ? { wrapS: Math.max(1, li[i + 19]) } : {}),
      ...(span.some((v) => v !== 1) ? { span: Array.from(span) } : {}),
    });
  }
  snapStations = stationsKey(ni);
  return { version: ++snapVersion, city, frame: frameSpec(), stations, lines, fare: { ...fares }, ...(passages.length ? { passages } : {}) };
}

/** T-1464: a bus line's row (`ClockBus`); its stops that are no station join `nodes` once each,
 * flagged `bus`, so every stop of every line has a node row. */
function busRow(a: TrackApi, id: number, nodes: ClockNode[]): ClockBus {
  const r = a.bus_route(id), st = a.bus_stops(id);
  const stops: ClockBus["stops"] = [];
  for (let k = 0; k + 7 < st.length; k += 8) {
    const [lng, lat] = packLngLat(st[k + 1], st[k + 2]);
    const s = { id: st[k], x: st[k + 1], y: st[k + 2], lng, lat, kind: st[k + 4], station: st[k + 5] };
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
  return { built: r[0] === 1, route, kinds, roadM: a.bus_road_m(id), stops, street, routeLL, streetLL: ll(street) };
}
/** a bus route's longitude, latitude pairs by line, until the route changes */
const busRouteLL = new Map<number, { route: number[]; ll: number[] }>();

/** T-1475: a new bus line's name, "Bus n" with n the sim's next free line number. */
function nextBusName(a: TrackApi): string {
  const taken = new Set<string>();
  const info = a.lines_info();
  for (let i = 0; i < info.length; i += 22) taken.add(a.line_name(info[i]));
  for (;;) {
    const name = a.next_line_name().replace(/^Line /, "Bus ");
    if (!taken.has(name)) return name;
  }
}

/** T-1464: `TrackApi.bus_quote` with each stop's and street stop's longitude and latitude after
 * its x, y: [road m, US$M, 1 if it makes a line, n, then x, y, kind, lng, lat per stop, then m and
 * x, y, lng, lat per street stop]. */
function busQuoteLL(a: TrackApi, line: number, route: number[]): Float64Array {
  const q = a.bus_quote(line, Float64Array.from(route)), out = [q[0], q[1], q[2], q[3]];
  let k = 4;
  for (let i = 0; i < q[3]; i++, k += 3) out.push(q[k], q[k + 1], q[k + 2], ...packLngLat(q[k], q[k + 1]));
  const m = q[k++];
  out.push(m);
  for (let i = 0; i < m; i++, k += 2) out.push(q[k], q[k + 1], ...packLngLat(q[k], q[k + 1]));
  return Float64Array.from(out);
}

/** T-1464: a bus as demand's cars: 0.5 of a 160-rider car, its 80 at crush. */
const BUS_DEMAND_CARS = 0.5;

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
let snapStations = "";
function stationsKey(ni = api!.nodes_info()): string {
  const out: number[] = [];
  for (let i = 0; i < ni.length; i += 10) if (ni[i + 7] > 0 && ni[i + 8] === 1) out.push(ni[i], ni[i + 1], ni[i + 2], ni[i + 3], ni[i + 7], ni[i + 9]);
  return `${out.join(",")}|${api!.station_upgrades().join(",")}`;
}

/** T-1168: each node's station complex for `line_related`, over every station (blueprints too,
 * unlike the snapshot's: a planned express is related before it is built), 0xffffffff for none;
 * worked out again when the world changes. */
let complexNodes: { version: number; of: Uint32Array } | null = null;
function complexOfNodes(): Uint32Array {
  if (complexNodes?.version === version) return complexNodes.of;
  const a = api!, ni = a.nodes_info();
  const geo: { id: string; x: number; y: number; heading: number; length: number }[] = [];
  let top = 0;
  for (let i = 0; i < ni.length; i += 10) {
    top = Math.max(top, ni[i]);
    if (ni[i + 7] > 0) geo.push({ id: String(ni[i]), x: ni[i + 1], y: ni[i + 2], heading: ni[i + 9], length: ni[i + 7] });
  }
  const fl = a.station_links(), links: [string, string][] = [];
  for (let k = 0; k + 1 < fl.length; k += 2) links.push([String(fl[k]), String(fl[k + 1])]);
  const cx = complexesOf(nodeComplexes, geo, links);
  const of = new Uint32Array(top + 1).fill(0xffffffff);
  for (const g of geo) of[Number(g.id)] = cx.of.get(g.id)!;
  complexNodes = { version, of };
  return of;
}

function scheduleSnapshot() {
  clearTimeout(snapTimer);
  snapTimer = setTimeout(() => send({ kind: "snapshot", snapshot: buildSnapshot() }), SNAPSHOT_DEBOUNCE_MS) as unknown as number;
}

// ---------------------------------------------------------------- edits

function endArr(e: RouteEnd): number[] {
  if (e.kind === "free") return [0, 0, e.x, e.y, e.level];
  if (e.kind === "node") return [1, e.node, 0, 0, 0];
  return [2, e.edge, e.x, e.y, 0];
}
function freeEnds(at: Extract<StationAt, { kind: "free" }>): Float64Array {
  return new Float64Array([0, 0, at.x0, at.y0, at.level, 0, 0, at.x1, at.y1, at.level]);
}
function pisArr(p: PiInput[]): Float64Array {
  return new Float64Array(p.flatMap((v) => [v.x, v.y, v.radius ?? 0, v.level]));
}
function colourInt(c: string) {
  return parseInt(c.slice(1), 16);
}

function apply(op: EditOp): { r: number; line?: number; node?: number; skipped?: number } {
  const a = api!;
  switch (op.op) {
    case "route": {
      a.set_route_tangent(!!op.through);
      a.set_design_speed(op.design ?? 160);
      a.set_route_hsr(!!op.hsr);
      const r = a.add_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : op.quad ? 4 : 2, false);
      return { r, node: r === 0 ? a.route_end() : undefined };
    }
    case "edgePis":
      return { r: a.set_edge_pis(op.edge, pisArr(op.pis)) };
    case "moveNode":
      return { r: a.move_node(op.node, op.x, op.y) };
    case "moveNodes":
      return { r: a.move_nodes(new Uint32Array(op.nodes), op.dx, op.dy) };
    case "edgeLevel":
      return { r: a.set_edge_level(op.edge, op.level) };
    case "edgesLevel": {
      const r = a.set_edges_level(new Uint32Array(op.edges), op.level);
      return { r, skipped: a.last_skipped() };
    }
    case "removeEdge":
      return { r: a.delete_edge(op.edge) };
    case "removePieces":
      return { r: a.delete_pieces(new Uint32Array(op.edges), new Uint32Array(op.stations)) };
    case "flying":
      return { r: a.set_flying(op.node, op.flying) };
    case "stationUpgrade":
      return { r: a.set_station_upgrade(new Uint32Array(op.nodes), op.bit, op.on) };
    case "addStation":
      return op.at.kind === "node"
        ? { r: a.add_station(1, op.at.node, 0, 0, op.platform, op.name) }
        : op.at.kind === "edge"
          ? { r: a.add_station(2, op.at.edge, op.at.x, op.at.y, op.platform, op.name) }
          : (a.set_route_hsr(!!op.at.hsr), { r: a.add_free_station(freeEnds(op.at), op.at.quad ? 4 : 2, op.platform, op.name) });
    case "crossover":
      return { r: a.add_crossover(op.edge, op.x, op.y) };
    case "doubleUp":
      return { r: a.double_up(op.edge) };
    case "stationProps":
      return { r: a.set_station_props(op.node, op.platform, op.name) };
    case "removeStation":
      return { r: a.remove_station(op.node) };
    case "construct":
      return { r: a.construct_with_buses(new Uint32Array(op.edges), new Uint32Array(op.stations), new Uint32Array(op.buses ?? [])) };
    case "constructAll":
      return { r: a.construct_all() };
    case "blueprintOff":
      return { r: a.set_blueprint_off(new Uint32Array(op.edges), new Uint32Array(op.stations), op.off) };
    case "constructLine":
      return { r: a.construct_line(op.line) };
    case "addBusLine": {
      const id = a.new_line_id();
      const [h, m, l] = NEW_BUS_HEADWAYS;
      return { r: a.set_bus_line(id, Float64Array.from(op.route), h, m, l, 30, 120, colourInt(op.colour), op.name || nextBusName(a)), line: id };
    }
    case "busRoute":
      return { r: a.set_bus_route(op.line, Float64Array.from(op.route)) };
    case "addLine": {
      const id = a.new_line_id();
      const [h, m, l] = NEW_LINE_TRAINS;
      return { r: a.set_line_with_trains(id, new Uint32Array(op.stops), h, m, l, 30, 180, newLineCars, colourInt(op.colour), op.name || a.next_line_name()), line: id };
    }
    case "copyLine": {
      const id = a.new_line_id();
      const existing = new Set<string>();
      const info = a.lines_info();
      let colour = 0;
      for (let i = 0; i < info.length; i += 22) {
        existing.add(a.line_name(info[i]));
        if (info[i] === op.line) colour = info[i + 1];
      }
      const base = op.name?.trim() || `${a.line_name(op.line)} copy`;
      let name = base, n = 2;
      while (existing.has(name)) name = `${base} ${n++}`;
      const r = a.copy_line(op.line, id, name);
      // T-1475: a line with no symbol of its own shows its name's, which a copy's name would change:
      // the copy gets the symbol its line shows
      if (r === 0 && !a.line_symbol(op.line)) a.set_line_badge(id, name, colour, letterOf(a.line_name(op.line)));
      return { r, line: id };
    }
    case "setStops":
      return { r: a.set_stops(op.line, new Uint32Array(op.stops)) };
    case "insertStop": {
      // one message instead of a query and then setStops: a round trip less, and no stale stops
      const stops = a.insert_stop(op.line, op.station);
      return stops.length ? { r: a.set_stops(op.line, Uint32Array.from(stops)) } : { r: INSERT_REFUSED };
    }
    case "setTrainCount":
      return { r: a.set_train_counts(op.line, ...op.trains) };
    case "setHeadway":
      return { r: op.seconds ? a.set_auto_headway(op.line, ...op.seconds) : a.clear_auto_headway(op.line) };
    case "lineLook":
      return { r: op.letter === undefined
        ? a.set_line_look(op.line, op.name, colourInt(op.colour))
        : a.set_line_badge(op.line, op.name, colourInt(op.colour), op.letter.trim()) };
    case "removeLine":
      return { r: a.remove_line(op.line) };
    case "lineCars": {
      const cars = Math.max(0, Math.min(255, Math.round(op.cars)));
      const r = a.set_line_cars(op.line, cars);
      if (r === 0) newLineCars = cars;
      return { r };
    }
    case "lineFast":
      return { r: a.set_line_fast(op.line, op.fast) };
    case "fares":
      fares = { ...(op.nodes ? faresOf({ nodes: op.nodes }, fares) : faresOf({ base: op.base, perKm: op.perKm }, fares)), scales: fares.scales, baseline: FARE_BASELINE };
      return { r: 0 };
    case "fareScale": {
      const scales = { ...fares.scales };
      if (op.scale === null || !Number.isFinite(op.scale)) delete scales[op.country];
      else scales[op.country] = tidyScale(op.scale);
      fares = { ...fares, scales };
      return { r: 0 };
    }
    case "link":
      return { r: op.on ? a.link_stations(op.a, op.b) : a.unlink_stations(op.a, op.b) };
    case "undo":
      return { r: a.undo() };
    case "redo":
      return { r: a.redo() };
  }
}

/** `apply`'s answer for an insertStop with nowhere to go (the track model's own codes are >= 0) */
const INSERT_REFUSED = -1;

function issues(): string[] {
  return api!.issue_kinds().split("\n").filter(Boolean);
}

/** The "doubleUp" query's answer (protocol.ts). */
function doubleUpAnswer(edge: number): Float64Array {
  const price = api!.preview_double_up(edge);
  if (price >= 0) return new Float64Array([price]);
  const kinds = api!.issue_kinds();
  return new Float64Array([-1, ...Array.from(kinds, (c) => c.charCodeAt(0))]);
}

// ---------------------------------------------------------------- messages

let ready: Promise<void> | null = null;

/** the zones whose masks the view fetches ahead (T-302) */
let viewZones: string[] = [];
function viewMasks(): Promise<void> {
  return api && viewZones.length ? ensureMasks(viewZones) : Promise.resolve();
}

let messages = Promise.resolve();
onmessage = (e: MessageEvent<ToClock>) => {
  const msg = e.data;
  if (msg.kind === "view") {
    // outside the queue, so the fetch starts at once; a mask is installed between messages
    const midLat = (msg.s + msg.n) / 2, wide = (msg.e - msg.w) * M_PER_DEG * Math.cos(midLat * Math.PI / 180) > VIEW_SPAN_M;
    const cx = (msg.w + msg.e) / 2;
    const skip = new Set(msg.skip);
    viewZones = (wide ? zonesTouching(cx, midLat, cx, midLat, VIEW_MARGIN_M) : zonesTouching(msg.w, msg.s, msg.e, msg.n, VIEW_MARGIN_M)).map((c) => c.id).filter((id) => !skip.has(id));
    for (const id of viewZones) if (!masksIn.has(id)) void fetchWater(id).catch(() => {});
    void viewMasks();
    return;
  }
  if (msg.kind === "shade") {
    thumbShade = msg.shade;
    return;
  }
  if (msg.kind === "account") {
    const before = cloudCtx;
    cloudCtx = msg.ctx;
    if (msg.ctx?.on && (msg.settle || !before?.on || before.acct !== msg.ctx.acct)) void cloudKick();
    return;
  }
  if (msg.kind === "previewStale") {
    previewStaleSeq = Math.max(previewStaleSeq, msg.seq);
    wakeStale?.();
    return;
  }
  messages =messages.then(() => processMessage(msg)).catch(err => send({ kind: "error", message: String(err) }));
};
async function processMessage(msg: ToClock) {
  if (msg.kind === "init") {
    if (msg.patch) patchMode = msg.patch;
    // World identity and coordinates never follow the map's focused city.
    city = HOME_CITY;
    packBase = msg.pack;
    outlinesRead = readOutlinesForPlaces(msg.outlines).then(() => {
      if (!api) return;
      placeLineCountries();
      if (syncHolidays()) sendTrips();
    });
    ready = (async () => {
      const t0 = performance.now();
      await init();
      const model = newModel();
      loadHolidays();
      emptyTrack = model.save();
      moneyParams = [...model.money_params()];
      void viewMasks(); // the view sent before the track model was up
      void worldWaterIntoModel();
      send({ kind: "ready", initMs: performance.now() - t0, frame: frameSpec() });
      // The last game, unless the page asked for a new one (T-029).
      testPage = !!(msg.fresh || msg.testPage);
      if (msg.backupMs && msg.backupMs > 0) backupMs = msg.backupMs;
      // T-1196: the game picked in the saves panel before the page reloaded; a file or backup goes
      // into a slot of its own below, the slot it was picked from kept to fall back on
      const pending = msg.pending ? await takePending() : null;
      let incoming: { bytes: Uint8Array; name: string } | null = null;
      if (pending?.kind === "new") {
        msg.newSlot = pending.name || newGameName(await listSlots().catch(() => []), pending.spec);
        msg.game = pending.spec;
      } else if (pending) {
        incoming = await pendingGame(pending);
        msg.slot = pending.from;
      }
      if (msg.fresh) {
        // T-320: a new game for a check owns no slot, so nothing it does can reach a saved game
        activeSlot = "";
        activeLease = null;
      } else if (msg.newSlot) {
        activeSlot = "slot-" + crypto.randomUUID();
        newSlotName = { key: activeSlot, name: msg.newSlot };
        hold(await leaseSlot(activeSlot));
      } else {
        activeSlot = msg.slot?.startsWith("slot-") ? msg.slot : ORIGINAL_SLOT;
        if (activeSlot !== ORIGINAL_SLOT && !await readStored(activeSlot)) activeSlot = ORIGINAL_SLOT;
        // a stress page reads the slot but does not take it, so it never writes it
        hold(msg.testPage ? null : await leaseSlot(activeSlot));
      }
      await publishSlots();
      if (incoming) {
        try {
          const g = incoming;
          await intoNewSlot(g.name, () => loadGame(g.bytes, "file"), true);
          return;
        } catch (err) {
          send({ kind: "loaded", from: "file", clock: START_CLOCK, frame: frameSpec(), error: (err as Error).message, game });
        }
      }
      const stored = msg.fresh || msg.newSlot ? null : await readStored(activeSlot);
      if (!stored && activeSlot) unwritten = activeSlot;
      if (msg.fresh || msg.newSlot) {
        // T-728, T-731: the start menu's choice; a real network is written to its slot at once
        if (msg.game?.real) {
          try {
            await newGame(msg.game);
            dirty = true;
            void autosave();
            return;
          } catch (err) {
            send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The real network could not be loaded. " + (err as Error).message, frame: frameSpec(), game });
          }
        }
        const empty = msg.game && { ...msg.game, real: undefined };
        // a start on another clock than New York's starts again at its own morning
        if (startClock(empty) !== START_CLOCK) { await newGame(empty); return; }
        setNewGame(empty);
      }
      if (stored) {
        try {
          if (await loadGame(stored.bytes, "autosave") && activeLease) await saveConverted(activeSlot, stored);
          return;
        } catch (err) {
          // Keep the unreadable save aside, and its slot as it is: the new game gets a slot of its own (T-743).
          await writeStored("unreadable-" + stored.at, stored);
          if (activeLease) {
            activeLease.release();
            activeSlot = "slot-" + crypto.randomUUID();
            newSlotName = { key: activeSlot, name: nextGameName(await listSlots().catch(() => [])) };
            unwritten = activeSlot;
            hold(await leaseSlot(activeSlot));
            await publishSlots();
          }
          send({ kind: "loaded", from: "new", clock: START_CLOCK, error: "The last game could not be loaded. " + (err as Error).message, frame: frameSpec(), game });
        }
      }
      keep();
      sendOpened();
      sendState();
      send({ kind: "snapshot", snapshot: buildSnapshot() });
    })().catch((err) => send({ kind: "failed", error: String(err) }));
    // T-988: the game in hand claimed on the account, the other slots settled
    void ready.then(() => { initDone = true; void cloudKick(); });
    addEventListener("online", () => void cloudKick());
    setInterval(tick, 1000);
    setInterval(liveMoney, LIVE_MONEY_MS);
    return;
  }
  await ready;
  if (msg.kind === "leave") return await leave(msg);
  if (!api) return;
  if (modelDown) return failAnswer(msg);
  if (isPreview(msg) && stale(msg.seq)) return sendStale(msg.seq);
  try {
    if (msg.kind === "saveSlot") return await manageSlot(msg);
    if (msg.kind === "import" || msg.kind === "export" || msg.kind === "newGame") return await whole(msg);
    await handle(msg);
  } catch (err) {
    if (isPanic(err)) await recover(err, msg.kind, msg.kind === "edit" ? EDIT_UNDONE : MODEL_RESTORED, performance.now(), msg.kind === "edit");
    else send({ kind: "error", message: String(err) });
    failAnswer(msg);
  } finally {
    if (msg.kind === "edit" || (msg.kind === "saveSlot" && msg.action !== "save") || msg.kind === "import" || msg.kind === "newGame") trackVersion++;
  }
};

/** T-1039: the network's version (any edit or load moves it on), and the route request whose
 * preview found every mask its geometry needs in, at that version: its edit needs no second look */
let trackVersion = 0;
let maskedRoute = "";
type RouteAsk = { from: RouteEnd; to: RouteEnd; pis: PiInput[]; single: boolean; quad?: boolean; through?: boolean; design?: number; hsr?: boolean };
const routeKey = (r: RouteAsk) => `${trackVersion}|${JSON.stringify([r.from, r.to, r.pis, !!r.single, !!r.quad, !!r.through, r.design ?? 160, !!r.hsr])}`;

/** An answer for a message the track model could not work out, so nothing waits on it for ever. */
function failAnswer(msg: ToClock) {
  if (msg.kind === "edit") send({ kind: "edited", seq: msg.seq, ok: false, issues: ["Internal"], charge: 0 });
  if (msg.kind === "previewStation") send({ kind: "stationCheck", seq: msg.seq, ok: false, issues: ["Internal"], cost: 0 });
  if (msg.kind === "previewCrossover") send({ kind: "crossoverCheck", seq: msg.seq, ok: false, issues: ["Internal"], cost: 0, parts: [] });
  if (msg.kind === "preview" || msg.kind === "previewEdge" || msg.kind === "previewNode" || msg.kind === "previewNodes") send({ kind: "preview", seq: msg.seq, ok: false, cost: 0, lengthM: 0, mult: 0, issues: ["Internal"], verts: new Float64Array(0), pts: new Float32Array(0) });
  if (msg.kind === "query") send({ kind: "answer", seq: msg.seq, data: new Float64Array(0) });
}

/** Once a wall second: autosave when due. */
function tick() {
  if (!api || modelDown) return;
  if (dirty && Date.now() - savedAt >= AUTOSAVE_MS) void autosave();
}

/** T-284: how often the cash is settled and sent while the clock runs, wall ms: 15 game seconds at
 * normal speed, 4 game minutes at the fastest. */
const LIVE_MONEY_MS = 250;
function liveMoney() {
  if (!api || modelDown || !(clk.rate > 0)) return;
  if (settle()) sendMoney();
  else sendCash();
}

/** Messages that replace or copy the whole game. */
async function whole(msg: Extract<ToClock, { kind: "import" | "export" | "newGame" }>) {
  if (msg.kind === "export") {
    const bytes = await saveBytes();
    send({ kind: "exported", bytes, day: Math.floor(gameNow() / 86400) }, [bytes.buffer]);
  } else {
    try {
      const name = msg.name?.trim();
      if (msg.kind === "import") await intoNewSlot(name || "Loaded game", () => loadGame(new Uint8Array(msg.bytes), "file"), true);
      else {
        const spec = msg.game;
        await intoNewSlot(name || newGameName(await listSlots().catch(() => []), spec ?? { mode: "normal" }), () => newGame(spec), !!spec?.real);
      }
    } catch (err) {
      if (isPanic(err)) throw err;
      // "file": the main thread keeps the game it has
      send({ kind: "loaded", from: "file", clock: gameNow(), frame: frameSpec(), error: (err as Error).message, game });
    }
  }
}

/** Before an edit that makes new track: the masks its geometry needs (worked out as its preview). */
async function masksFor(op: EditOp) {
  const a = api!;
  if (op.op === "route" && maskedRoute === routeKey(op)) return;
  if (op.op === "route") await withMasks(() => (a.set_route_tangent(!!op.through), a.set_design_speed(op.design ?? 160), a.set_route_hsr(!!op.hsr), a.preview_route(new Float64Array([...endArr(op.from), ...endArr(op.to)]), pisArr(op.pis), op.single ? 1 : op.quad ? 4 : 2, false)), routePoints);
  else if (op.op === "edgePis") await withMasks(() => a.preview_edge(op.edge, pisArr(op.pis)), routePoints);
  else if (op.op === "moveNode") await withMasks(() => a.preview_move_node(op.node, op.x, op.y), nodePoints);
  else if (op.op === "moveNodes") await withMasks(() => a.preview_move_nodes(new Uint32Array(op.nodes), op.dx, op.dy), nodePoints);
}

async function handle(msg: ToClock) {
  if (!api) return;
  if (settle()) sendMoney();
  // an edit's new departures start now; trains already running keep their places (T-206)
  api.set_now(gameNow());
  if (msg.kind === "opened") {
    opened = [...msg.zones];
    dirty = true;
  } else if (msg.kind === "sandbox") {
    if (mode === "sandbox") return;
    setMode("sandbox");
    dirty = true;
    sendMoney();
    void autosave(); // the slot's card shows the mark at once
  } else if (msg.kind === "edit") {
    if (msg.op.op === "fares" || msg.op.op === "fareScale") {
      apply(msg.op);
      dirty = true;
      sendMoney();
      scheduleSnapshot();
      send({ kind: "edited", seq: msg.seq, ok: true, issues: [], charge: 0 });
      return;
    }
    if (msg.op.op === "link") {
      // T-197: a setting outside the undo history, like fares; the complexes change, so the
      // drawing (links in the state) and demand (the snapshot's complexes) both follow
      const { r } = apply(msg.op);
      // T-1281: a passage between constructed stations is paid at once
      const charge = r === 0 && msg.op.on ? api.link_charge() : 0;
      if (r === 0) {
        dirty = true;
        if (charge > 0) bookCharge(msg.seq, charge, 0);
        sendState();
        scheduleSnapshot();
        keep();
      }
      send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : msg.op.on ? issues() : [], charge });
      return;
    }
    await worldWaterIntoModel();
    await masksFor(msg.op);
    if (msg.op.op === "construct" || msg.op.op === "constructAll" || msg.op.op === "constructLine") await terrainForBlueprint();
    const tA = performance.now();
    // T-1071: a train count takes effect a moment ahead, once its trips are on screen
    const service = msg.op.op === "setTrainCount" || msg.op.op === "setHeadway";
    if (service) {
      api.set_now(gameNow());
      api.set_edit_lead(clk.rate * serviceLeadS());
    }
    const { r, line, node, skipped } = apply(msg.op);
    api.set_edit_lead(0); // a refused edit leaves it set
    if (import.meta.env.DEV && panicAfterEdit) { panicAfterEdit = false; panicNow(); }
    const tB = performance.now();
    timing = {};
    const charge = r === 0 ? api.last_charge() : 0;
    // The new state first, so the world is up to date when the edit's answer arrives.
    if (r === 0) {
      dirty = true;
      if (charge > 0) bookCharge(msg.seq, charge, api.last_train_charge());
      forget();
      const demand = api.dirty_demand_lines();
      sendState(service, service ? api.edit_time() : undefined);
      if (service) serviceEditMs = performance.now() - tA;
      if (demand.length || msg.op.op === "undo" || msg.op.op === "redo" || stationsKey() !== snapStations) scheduleSnapshot();
    }
    send({ kind: "edited", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : r === INSERT_REFUSED ? ["InsertStop"] : issues(), charge, line, node, skipped, ms: { apply: tB - tA, ...timing, total: performance.now() - tA } });
    if (r === 0) keep();
    // Paid for something: keep it (T-029, autosave on construct).
    if (charge > 0) void autosave();
  } else if (msg.kind === "clock") {
    if (msg.game === game) clk = { t: msg.t, rate: msg.rate, at: msg.at };
  } else if (msg.kind === "income") {
    incomePerHour = msg.perHour.map((p) => p.slice(0, PERIODS.length));
    lineIncome = new Map((msg.lines ?? []).map((l) => [l.line, l.perHour]));
    ridesPerHour = msg.rides.map((p) => p.slice(0, PERIODS.length));
    lineRides = new Map((msg.lines ?? []).map((l) => [l.line, l.rides]));
  } else if (msg.kind === "regionShares") {
    lineShares = new Map(msg.lines.map((l) => [l.line, { fare: l.fare, ride: l.ride, run: l.run }]));
    upkeepRegions = msg.upkeep ?? [];
  } else if (msg.kind === "points") {
    linePoints = new Map(msg.lines.map((l) => [l.line, l.regions]));
  } else if (msg.kind === "star" && msg.share !== undefined) {
    const g = goals.get(msg.zone) ?? {};
    if (g[msg.share] === undefined) {
      goals.set(msg.zone, { ...g, [msg.share]: Math.floor(gameNow()) });
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "star") {
    if (!stars.has(msg.zone)) {
      stars.set(msg.zone, Math.floor(gameNow()));
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "lineNotes") {
    let added = false;
    for (const [id, v] of Object.entries(msg.notes)) if (!lineNotes.has(id)) { lineNotes.set(id, v); added = true; }
    if (added) {
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "chargeShares") {
    // T-531: move a paid edit's charge from "" to the regions it was spent in
    const c = heldCharges.get(msg.seq);
    heldCharges.delete(msg.seq);
    if (c) {
      const move = (shares: [string, number][], v: number, k: "build" | "trains") => {
        if (!v || !shares.length) return;
        for (const x of regionEntries("", c.day, c.hour)) x[k] -= v;
        for (const [z, s] of shares) for (const x of regionEntries(z, c.day, c.hour)) x[k] += v * s;
      };
      move(msg.build, c.build, "build");
      move(msg.trains, c.trains, "trains");
      dirty = true;
      sendMoney();
    }
  } else if (msg.kind === "riders") {
    // a new demand result: the boarding account follows it (T-164), a few lines at a time, one
    // kind of day's riders a message (T-260)
    // an empty one clears the kind (an away day's once none is near, T-264)
    if (peakC[msg.day ?? 0] && msg.packed.length >= PERIODS.length) peakC[msg.day ?? 0] = strengths(msg.packed.subarray(0, PERIODS.length));
    api.set_riders(msg.packed, msg.day ?? 0);
    if (msg.arrive) api.set_arrivals(msg.arrive, msg.day ?? 0);
    stepBoarding();
  } else if (msg.kind === "transfers") {
    // T-554: when riders changing lines reach their next platform; the account is unchanged
    api.set_transfers(msg.packed);
    sendTrips();
  } else if (msg.kind === "flush") {
    void autosave().then(cloudFlush);
  } else if (import.meta.env.DEV && msg.kind === "debugPanic") {
    if (msg.when === "afterEdit") panicAfterEdit = true;
    else if (msg.when === "read") api.edge_polyline(0xffffffff, 0);
    else panicNow();
  } else if (msg.kind === "query") {
    const q = msg.q;
    const data =
      q.what === "insertStop" ? api.insert_stop(q.line, q.station)
      : q.what === "stopTiers" ? api.stop_tiers(q.line)
      : q.what === "passedStations" ? api.passed_stations(q.line)
      : q.what === "junction" ? api.junction_info(q.node)
      : q.what === "edgePis" ? api.edge_pis(q.edge)
      : q.what === "flyover" ? new Float64Array([api.flyover_cost(q.node)])
      : q.what === "upgradeCost" ? new Float64Array(UPGRADE_BITS.map((b) => api!.station_upgrade_cost(q.node, b)))
      : q.what === "doubleUp" ? doubleUpAnswer(q.edge)
      : q.what === "lineAccount" ? api.line_account(q.line, q.day ?? 0)
      : q.what === "linePeriods" ? api.line_periods(q.line, q.day)
      : q.what === "today" ? today()
      : q.what === "lineHeld" ? api.line_held(q.line)
      : q.what === "lineMarks" ? api.line_marks(q.line)
      : q.what === "lineRelated" ? api.line_related(q.line, complexOfNodes())
      : q.what === "removal" ? new Float64Array(api.removal_lines(new Uint32Array(q.edges), new Uint32Array(q.stations)))
      : q.what === "busQuote" ? busQuoteLL(api, q.line, q.route)
      : api.line_delays(q.line, q.level);
    send({ kind: "answer", seq: msg.seq, data }, [data.buffer]);
  } else if (msg.kind === "previewStation") {
    const at = msg.at;
    const r = at.kind === "node" ? api.preview_station(1, at.node, 0, 0, msg.platform)
      : at.kind === "edge" ? api.preview_station(2, at.edge, at.x, at.y, msg.platform)
      : (api.set_route_hsr(!!at.hsr), api.preview_free_station(freeEnds(at), at.quad ? 4 : 2, msg.platform));
    send({ kind: "stationCheck", seq: msg.seq, ok: r === 0, issues: r === 0 ? [] : issues(), cost: r === 0 ? api.station_cost() : 0 });
  } else if (msg.kind === "previewCrossover") {
    const out = api.preview_crossover(msg.edge, msg.x, msg.y);
    const ok = out[0] === 1;
    const parts: Float32Array[] = [];
    for (let k = 2; k < out.length; ) {
      const n = out[k++], pts = new Float32Array(n * 3);
      for (let j = 0; j < n; j++, k += 3) {
        const [x, y] = packLocal(out[k], out[k + 1]);
        pts.set([x, y, out[k + 2]], j * 3);
      }
      parts.push(pts);
    }
    send({ kind: "crossoverCheck", seq: msg.seq, ok, issues: ok ? [] : issues(), cost: out[1], parts, anchor: anchor().version }, parts.map((p) => p.buffer));
  } else if (msg.kind === "previewNode" || msg.kind === "previewNodes") {
    const a = api;
    const out = await withMasks(() => msg.kind === "previewNode" ? a.preview_move_node(msg.node, msg.x, msg.y) : a.preview_move_nodes(new Uint32Array(msg.nodes), msg.dx, msg.dy), nodePoints, msg.seq);
    if (out) sendNodePreview(msg.seq, out);
    else sendStale(msg.seq);
  } else if (msg.kind === "preview" || msg.kind === "previewEdge") {
    const a = api;
    const out = await withMasks(() => msg.kind === "preview"
      ? (a.set_route_tangent(!!msg.through), a.set_design_speed(msg.design ?? 160), a.set_route_hsr(!!msg.hsr), a.preview_route(new Float64Array([...endArr(msg.from), ...endArr(msg.to)]), pisArr(msg.pis), msg.single ? 1 : msg.quad ? 4 : 2, false))
      : a.preview_edge(msg.edge, pisArr(msg.pis)), routePoints, msg.seq);
    if (!out) return sendStale(msg.seq);
    if (msg.kind === "preview") maskedRoute = masksComplete ? routeKey(msg) : "";
    let arc: Float64Array | undefined;
    const ra = msg.kind === "preview" ? a.route_arc() : null;
    if (ra?.length === 3) {
      const [x, y] = packLocal(ra[0], ra[1]);
      arc = new Float64Array([x, y, ra[2], 0]);
    }
    const ok = out[0] === 1, nv = out[3];
    const verts = new Float64Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      const [x, y] = packLocal(out[5 + i * 4], out[6 + i * 4]);
      verts.set([x, y, out[7 + i * 4], out[8 + i * 4]], i * 4);
    }
    const p0 = 5 + nv * 4;
    const pts = new Float32Array(out.length - p0);
    for (let k = p0; k < out.length; k += 3) {
      const [x, y] = packLocal(out[k], out[k + 1]);
      pts.set([x, y, out[k + 2]], k - p0);
    }
    // Explicit shape also permits type-checking before the track WASM declarations regenerate.
    const speeds = (api as TrackApi & { preview_render_speeds(): Float32Array }).preview_render_speeds();
    if (speeds.length !== Math.max(0, pts.length / 3 - 1)) throw new Error("Preview speed/point length mismatch");
    const marks = markPoints((api as TrackApi & { preview_marks(): Float64Array }).preview_marks(), 3);
    const ground = msg.kind === "preview" ? Array.from(a.preview_ground()) : undefined;
    send({ kind: "preview", seq: msg.seq, ok, cost: out[1], lengthM: out[2], mult: out[4], ground, issues: ok ? [] : issues(), verts, pts, speeds, arc, anchor: anchor().version, marks }, [verts.buffer, pts.buffer, speeds.buffer, marks.buffer]);
  } else if (msg.kind === "epoch") {
    if (msg.epoch !== epoch) {
      epoch = msg.epoch;
      sendTrips();
    }
  } else if (msg.kind === "wholeState") {
    // the client could not take a state as changes to the last (T-1029)
    postWhole();
    if (api) sendState();
  } else if (msg.kind === "anchor") {
    // the view travelled far (T-307): every drawn position again, relative to the new anchor
    if (msg.version !== anchor().version) {
      setAnchor(makeAnchor(msg.lng, msg.lat, msg.version));
      forget(true);
      sendState();
    }
  }
}

/** A node drag's preview (`TrackApi.preview_move_node`, T-093): every edge at the node, its
 * vertices one after another and its polyline as one part of `pts`. */
function sendNodePreview(seq: number, out: Float64Array) {
  const ok = out[0] === 1, n = out[2];
  const verts: number[] = [], pts: number[] = [], parts: number[] = [0];
  let i = 3;
  for (let e = 0; e < n; e++) {
    const nv = out[i++];
    for (let k = 0; k < nv; k++, i += 4) verts.push(...packLocal(out[i], out[i + 1]), out[i + 2], out[i + 3]);
    const np = out[i++];
    for (let k = 0; k < np; k++, i += 3) pts.push(...packLocal(out[i], out[i + 1]), out[i + 2]);
    parts.push(pts.length / 3);
  }
  const v = new Float64Array(verts), p = new Float32Array(pts), pa = new Uint32Array(parts);
  const marks = markPoints((api as TrackApi & { preview_marks(): Float64Array }).preview_marks(), 3);
  send({ kind: "preview", seq, ok, cost: out[1], lengthM: 0, mult: 0, issues: ok ? [] : issues(), verts: v, pts: p, parts: pa, anchor: anchor().version, marks }, [v.buffer, p.buffer, pa.buffer, marks.buffer]);
}
